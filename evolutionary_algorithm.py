import numpy as np
import pandas as pd
import subprocess
import os
from tqdm import tqdm

# Constants
DATA_PATH = "../data/S4_pillar_sims/grid_data.csv"
OUTPUT_DIR = "./evolution_results"
TEMP_DIR = f"{OUTPUT_DIR}/temp"
WAVELENGTH_RANGE = (3, 5)
POPULATION_SIZE = 20
GENERATIONS = 50
HEIGHT_RANGE = (1, 10)  # microns
RADIUS_RANGE = (0.05 * 2.7, 0.45 * 2.7)  # microns

# Fitness function: maximize transmission peak height and duration
def fitness_function(transmission_values, wavelengths):
    mask = (wavelengths >= WAVELENGTH_RANGE[0]) & (wavelengths <= WAVELENGTH_RANGE[1])
    transmission_filtered = transmission_values[mask]
    peak_height = np.max(transmission_filtered)
    peak_duration = np.sum(transmission_filtered > 0.8 * peak_height)  # Duration above 80% of peak
    return peak_height + peak_duration

# Generate initial population from grid data
def initialize_population(data_path, population_size):
    data = pd.read_csv(data_path)
    unique_configs = data[['height', 'radii list']].drop_duplicates()
    population = unique_configs.sample(n=population_size, replace=True).to_dict('records')
    return population

# Mutate a configuration
def mutate(config):
    config['height'] = np.clip(config['height'] + np.random.uniform(-0.5, 0.5), *HEIGHT_RANGE)
    config['radii list'] = [
        np.clip(r + np.random.uniform(-0.05, 0.05), *RADIUS_RANGE) for r in config['radii list']
    ]
    return config

# Crossover between two configurations
def crossover(parent1, parent2):
    child = {
        'height': np.mean([parent1['height'], parent2['height']]),
        'radii list': [
            np.mean([r1, r2]) for r1, r2 in zip(parent1['radii list'], parent2['radii list'])
        ]
    }
    return child

# Evaluate fitness of a configuration
def evaluate_fitness(config):
    radii_args = " ".join(map(str, config['radii list']))
    output_file = f"{TEMP_DIR}/temp_{config['height']}_{radii_args.replace(' ', '_')}.csv"
    subprocess.run([
        "python3", "simulation_task.py",
        "--r1", str(config['radii list'][0]),
        "--r2", str(config['radii list'][1]),
        "--r3", str(config['radii list'][2]),
        "--r4", str(config['radii list'][3]),
        "--height", str(config['height']),
        "--output", output_file
    ])
    data = pd.read_csv(output_file)
    wavelengths = data['wavelength'].values
    transmission = data['transmission'].values
    return fitness_function(transmission, wavelengths)

# Run evolutionary algorithm
def run_evolutionary_algorithm():
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    population = initialize_population(DATA_PATH, POPULATION_SIZE)

    for generation in range(GENERATIONS):
        print(f"Generation {generation + 1}/{GENERATIONS}")
        fitness_scores = []
        for config in tqdm(population):
            fitness = evaluate_fitness(config)
            fitness_scores.append((config, fitness))
        
        # Select top individuals
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        top_individuals = [x[0] for x in fitness_scores[:POPULATION_SIZE // 2]]

        # Generate new population
        new_population = []
        for _ in range(POPULATION_SIZE):
            if np.random.rand() < 0.5:  # Mutation
                parent = np.random.choice(top_individuals)
                new_population.append(mutate(parent.copy()))
            else:  # Crossover
                parent1, parent2 = np.random.choice(top_individuals, size=2, replace=False)
                new_population.append(crossover(parent1, parent2))
        
        population = new_population

        # Save generation results
        results = pd.DataFrame([{'height': ind['height'], 'radii list': ind['radii list']} for ind in population])
        results.to_csv(f"{OUTPUT_DIR}/generation_{generation + 1}.csv", index=False)

    print("Evolutionary algorithm completed. Results saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    run_evolutionary_algorithm()
