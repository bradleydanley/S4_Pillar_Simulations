import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import subprocess
import ast
import time
import re

# Constants
DATA_PATH = "../data/grid_data.csv"
OUTPUT_DIR = "./evolution_results"
TEMP_DIR = f"{OUTPUT_DIR}/temp"
PARAMS_FILE = f"{TEMP_DIR}/params.txt"
WAVELENGTH_RANGE = (3, 5)
POPULATION_SIZE = 128
GENERATIONS = 10
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
# Ensure radii_list is parsed as a Python list
            radii_list = ast.literal_eval(config['radii list']) if isinstance(config['radii list'], str) else config['radii list']
            radii_list = [round(float(r), 5) for r in radii_list]  # Convert to float and round to 5 decimals
            height = round(float(config['height']), 5)  # Round height to 5 decimals
            f.write(f"{radii_list} {height}\n")  # Write radii_list and height

# Function to run simulations in parallel using the same approach as run_parallel.sh
def run_simulations_in_parallel(params_file, temp_dir, failed_file, generation, max_jobs=128, batch_timeout=80):
    job_count = 0
    batch_pids = []
    batch_params = []

    def wait_batch():
        nonlocal batch_pids, batch_params
        print(f"Waiting for a batch of {len(batch_pids)} job(s)...")
        start_time = time.time()

        while True:  # Corrected line
            all_finished = True
            for pid in batch_pids:
                if os.system(f"kill -0 {pid} 2>/dev/null") == 0:
                    all_finished = False
                    break

            elapsed = time.time() - start_time
            if elapsed > batch_timeout:
                print(f"Batch timeout reached after {elapsed} seconds. Killing remaining tasks...")
                for i, pid in enumerate(batch_pids):
                    if os.system(f"kill -0 {pid} 2>/dev/null") == 0:
                        os.system(f"kill -9 {pid} 2>/dev/null")
                        with open(failed_file, 'a') as f:
                            f.write(f"Job [{batch_params[i]}] failed (timeout after {elapsed} seconds).\n")
                break
            if all_finished:
                print("Batch completed successfully.")
                break
            time.sleep(5)

        batch_pids = []
        batch_params = []

    with open(params_file, 'r') as f:
        for line in f:
            #if generation > 1:
            #    from IPython import embed; embed(); exit()
            #print(f"Raw line: {line.strip()}")  # Debugging: Print the raw line
            string = re.split(r'(?<=\])', line.replace(" ", ""))  # Manually parse the line to extract parameters
            #print(f"Split string: {string}")  # Debugging: Print the split string
            radii_list = ast.literal_eval(string[0])
            r1, r2, r3, r4 = map(float, radii_list[:])  # First 4 values are radii
            h = float(ast.literal_eval(string[1]))  # Last value is height
            core = job_count % max_jobs
            output_file = f"{temp_dir}/temp_{r1}_{r2}_{r3}_{r4}_{h}.csv"
            #print(f"Launching job with parameters: r1={r1}, r2={r2}, r3={r3}, r4={r4}, height={h} on core {core}")
            pid = os.fork()
            if pid == 0:  # Child process
                os.system(f"taskset -c {core} python3 simulation_task.py --r1 {r1} --r2 {r2} --r3 {r3} --r4 {r4} --height {h} --output {output_file}")
                os._exit(0)
            else:  # Parent process
                batch_pids.append(pid)
                batch_params.append(f"r1={r1} r2={r2} r3={r3} r4={r4} h={h}")
                job_count += 1
            if job_count % max_jobs == 0:
                wait_batch()
    if batch_pids:
        wait_batch()

# Aggregate results from temporary files and include fitness values
def aggregate_results_with_fitness(temp_dir, output_file):
    temp_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.csv')]
    if not temp_files:
        raise RuntimeError(f"No simulation output files found in {temp_dir}. Check for errors in the simulation tasks.")
    with open(output_file, 'w') as outfile:
        # Write header with an additional 'fitness' column
        with open(temp_files[0], 'r') as infile:
            header = infile.readline().strip()
            outfile.write(f"{header},fitness\n")
        # Append data from all files with fitness values
        for temp_file in temp_files:
            with open(temp_file, 'r') as infile:
                next(infile)  # Skip header
                for line in infile:
                    row = line.strip().split(',')
                    transmission_values = np.array(ast.literal_eval(row[3]))  # Assuming 'transmission' is the 4th column
                    fitness = fitness_function(transmission_values)
                    outfile.write(f"{line.strip()},{fitness}\n")

# Run evolutionary algorithm
def run_evolutionary_algorithm():
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    population = initialize_population(DATA_PATH, POPULATION_SIZE)
    for generation in range(GENERATIONS):
        print(f"Generation {generation + 1}/{GENERATIONS}")
        # Write population to params file
        write_params_file(population, PARAMS_FILE)
        # Run simulations in parallel
        run_simulations_in_parallel(PARAMS_FILE, TEMP_DIR, f"{OUTPUT_DIR}/failed_jobs.txt", generation+1)
        # Aggregate results with fitness values
        generation_output = f"{OUTPUT_DIR}/generation_{generation + 1}.csv"
        aggregate_results_with_fitness(TEMP_DIR, generation_output)
        # Evaluate fitness and select top individuals
        data = pd.read_csv(generation_output)
        fitness_scores = []
        for _, row in data.iterrows():
            row_dict = row.to_dict()
            row_dict['radii list'] = ast.literal_eval(row_dict['radii list'])
            fitness_scores.append((row_dict, row['fitness']))
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
