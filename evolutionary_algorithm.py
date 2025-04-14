import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import subprocess
import ast

# Constants
DATA_PATH = "../data/grid_data.csv"
OUTPUT_DIR = "./evolution_results"
TEMP_DIR = f"{OUTPUT_DIR}/temp"
PARAMS_FILE = f"{TEMP_DIR}/params.txt"
WAVELENGTH_RANGE = (3, 5)
POPULATION_SIZE = 20
GENERATIONS = 50
HEIGHT_RANGE = (1, 10)  # microns
RADIUS_RANGE = (0.05 * 2.7, 0.45 * 2.7)  # microns

# Fitness function: maximize transmission peak height and duration
def fitness_function(transmission_values):
    peak_height = np.max(transmission_values)
    peak_duration = np.sum(transmission_values > 0.8 * peak_height)  # Duration above 80% of peak
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

# Write configurations to a parameter file for parallel execution
def write_params_file(population, params_file):
    with open(params_file, 'w') as f:
        for config in population:
            radii_list = config['radii list']
            radii_args = " ".join(map(str, radii_list))
            f.write(f"{radii_args} {config['height']}\n")
"""
# Run a single simulation
def run_simulation_task(config, temp_dir):
        radii_args = " ".join(map(str, config['radii list']))
        output_file = f"{temp_dir}/temp_{'_'.join(map(str, config['radii list']))}_{config['height']}.csv"
        config['radii list'] = ast.literal_eval(config['radii list'])
        subprocess.run([
            "python3", "simulation_task.py",
            "--r1", str(config['radii list'][0]),
            "--r2", str(config['radii list'][1]),
            "--r3", str(config['radii list'][2]),
            "--r4", str(config['radii list'][3]),
            "--height", str(config['height']),
            "--output", output_file
            ])
        return output_file
"""
# Aggregate results from temporary files
def aggregate_results(temp_dir, output_file):
    temp_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.csv')]
    if not temp_files:
        raise RuntimeError(f"No simulation output files found in {temp_dir}. Check for errors in the simulation tasks.")
    with open(output_file, 'w') as outfile:
        # Write header from the first file
        with open(temp_files[0], 'r') as infile:
            outfile.write(infile.readline())
        # Append data from all files
        for temp_file in temp_files:
            with open(temp_file, 'r') as infile:
                next(infile)  # Skip header
                outfile.writelines(infile)

# Run evolutionary algorithm
def run_evolutionary_algorithm():
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    population = initialize_population(DATA_PATH, POPULATION_SIZE)

    for generation in range(GENERATIONS):
        print(f"Generation {generation + 1}/{GENERATIONS}")
        
        # Write population to params file
        write_params_file(population, PARAMS_FILE)
        from IPython import embed; embed()
        # Run simulations in parallel using GNU parallel
        subprocess.run([
            "parallel", "--jobs", "128", "--bar", "--verbose", "--colsep", " ",
            "python3 simulation_task.py --r1 {1} --r2 {2} --r3 {3} --r4 {4} --height {5}"
            f"--output {TEMP_DIR}/temp_{{1}}_{{2}}_{{3}}_{{4}}_{{5}}.csv",
            ":::", PARAMS_FILE
        ])
        """

        #Run simulations sequentially

        for config in tqdm(population, desc=f"Running simulations for generation {generation + 1}"):
            run_simulation_task(config, TEMP_DIR)

        """
        # Aggregate results
        generation_output = f"{OUTPUT_DIR}/generation_{generation + 1}.csv"
        aggregate_results(TEMP_DIR, generation_output)
        


        # Evaluate fitness and select top individuals
        data = pd.read_csv(generation_output)
        #from IPython import embed; embed()
        fitness_scores = []
        for _, row in data.iterrows():
            transmission_values = np.array(row['transmission'])
            fitness = fitness_function(transmission_values)
            row_dict = row.to_dict()
            row_dict['radii list'] = ast.literal_eval(row_dict['radii list'])
            fitness_scores.append((row_dict, fitness))
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

    print("Evolutionary algorithm completed. Results saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    run_evolutionary_algorithm()
