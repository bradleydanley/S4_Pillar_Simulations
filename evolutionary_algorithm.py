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
GENERATIONS = 50  # Increased from 10 to 50
HEIGHT_RANGE = (1, 10)  # microns
RADIUS_RANGE = (0.05 * 2.7, 0.45 * 2.7)  # microns

# Fitness function: maximize transmission peak height and duration
def fitness_function(transmission_values, wavelengths):
    #from IPython import embed; embed(); exit()
    peak_height = np.max(transmission_values)
    peak_duration = np.sum(transmission_values > 0.8 * peak_height)  # Duration above 80% of peak
    area_under_curve = np.trapezoid(transmission_values, wavelengths)

    return round(2 * peak_height + peak_duration + area_under_curve, 6)  # Weight peak height more

# Generate initial population from grid data
def initialize_population(data_path, population_size):
    data = pd.read_csv(data_path)
    unique_configs = data[['height', 'radii list']].drop_duplicates()
    population = unique_configs.sample(n=population_size, replace=False).to_dict('records')
    return population

# Mutate a configuration
def mutate(config):
    # Introduce larger random changes within the allowed range
    config['height'] = np.clip(config['height'] + np.random.uniform(-1.0, 1.0), *HEIGHT_RANGE)
    config['radii list'] = [
        np.clip(r + np.random.uniform(-0.1, 0.1), *RADIUS_RANGE) for r in config['radii list']
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

    # Load the grid search CSV file into memory for quick lookup
    grid_search_file = DATA_PATH
    if os.path.exists(grid_search_file):
        grid_data = pd.read_csv(grid_search_file)
    else:
        grid_data = None

    def wait_batch():
        nonlocal batch_pids, batch_params
        print(f"Waiting for a batch of {len(batch_pids)} job(s)...")
        start_time = time.time()

        while True:
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
            # Parse the parameters
            string = re.split(r'(?<=\])', line.replace(" ", ""))
            radii_list = ast.literal_eval(string[0])
            r1, r2, r3, r4 = map(float, radii_list[:])
            h = float(ast.literal_eval(string[1]))
            core = job_count % max_jobs
            output_file = f"{temp_dir}/temp_{r1}_{r2}_{r3}_{r4}_{h}.csv"

            # Check if the configuration exists in the grid search CSV
            if grid_data is not None:
                match = grid_data[
                    (grid_data['height'] == h) &
                    (grid_data['radii list'] == str([r1, r2, r3, r4]))
                ]
                if not match.empty:
                    # Copy the matching row(s) to the temporary output file
                    match.to_csv(output_file, index=False)
                    #from IPython import embed; embed(); exit()
                    print(f"Configuration found in grid search. Skipping simulation for r1={r1}, r2={r2}, r3={r3}, r4={r4}, height={h}")
                else:

                    # Launch the simulation if the configuration is not found
                    print(f"Launching job with parameters: r1={r1}, r2={r2}, r3={r3}, r4={r4}, height={h} on core {core}")
                    pid = os.fork()
                    if pid == 0:  # Child process
                        os.system(f"taskset -c {core} python3 simulation_task.py --r1 {r1} --r2 {r2} --r3 {r3} --r4 {r4} --height {h} --output {output_file}")
                        os._exit(0)
                    else:  # Parent process
                        batch_pids.append(pid)
                        batch_params.append(f"r1={r1} r2={r2} r3={r3} r4={r4} h={h}")
                        job_count += 1

            # Wait for the batch to finish if the maximum number of jobs is reached
            if job_count % max_jobs == 0:
                wait_batch()
                batch_pids = []
                batch_params = []

    # Wait for any remaining jobs in the final batch
    if batch_pids:
        wait_batch()

# Aggregate results from temporary files and include fitness values
def aggregate_results_with_fitness(temp_dir, output_file):
    temp_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.csv')]
    if not temp_files:
        raise RuntimeError(f"No simulation output files found in {temp_dir}. Check for errors in the simulation tasks.")
    with open(output_file, 'w') as outfile:  # This is where the generation file is created
        # Write header with an additional 'fitness' column
        with open(temp_files[0], 'r') as infile:
            header = infile.readline().strip()
            outfile.write(f"{header},fitness\n")  # Add 'fitness' column to the header
        # Append data from all files with fitness values
        for temp_file in temp_files:
            #from IPython import embed; embed(); exit()
            with open(temp_file, 'r') as infile:
                next(infile)  # Skip header
                data_lines = []
                transmission_values = []
                for line in infile:
                    data_lines.append(line)
                    row = line.strip().split(',')
                    transmission_values.append(np.array(ast.literal_eval(row[3])))
                fitness = fitness_function(transmission_values, wavelengths=np.linspace(3, 5, 40))
            
                for line in data_lines:
                    outfile.write(f"{line.strip()},{fitness}\n")  # Write data with fitness to the output file

    # Cleanup: Remove temporary files
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            print(f"Removed temporary file: {temp_file}")
        except Exception as e:
            print(f"Error removing temporary file {temp_file}: {e}")

# Run evolutionary algorithm
def run_evolutionary_algorithm():
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    original_population = initialize_population(DATA_PATH, POPULATION_SIZE * 2)  # Larger initial pool for diversity
    population = original_population[:POPULATION_SIZE]  # Start with a subset of the original population

    for generation in range(GENERATIONS):
        print(f"Generation {generation + 1}/{GENERATIONS}")
        # Write population to params file
        write_params_file(population, PARAMS_FILE)
        # Run simulations in parallel
        run_simulations_in_parallel(PARAMS_FILE, TEMP_DIR, f"{OUTPUT_DIR}/failed_jobs.txt", generation + 1)
        # Aggregate results with fitness values
        generation_output = f"{OUTPUT_DIR}/generation_{generation + 1}.csv"
        aggregate_results_with_fitness(TEMP_DIR, generation_output)  # This is where the generation file is created
        # Evaluate fitness and select top individuals
        data = pd.read_csv(generation_output)
        fitness_scores = []
        for _, row in data.iterrows():
            row_dict = row.to_dict()
            row_dict['radii list'] = ast.literal_eval(row_dict['radii list'])
            fitness_scores.append((row_dict, row['fitness']))
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top 1/3 as parents
        num_parents = POPULATION_SIZE // 3
        parents = [x[0] for x in fitness_scores[:num_parents]]

        # Generate another 1/3 through mutation and crossover
        num_mutated = POPULATION_SIZE // 3
        mutated_population = []
        #from IPython import embed; embed(); exit()
        for _ in range(num_mutated):
            if np.random.rand() < 0.5:  # Mutation
                parent = np.random.choice(parents)
                mutated_population.append(mutate(parent.copy()))
            else:  # Crossover
                parent1, parent2 = np.random.choice(parents, size=2, replace=False)
                mutated_population.append(crossover(parent1, parent2))

        # Randomly sample the remaining 1/3 from the original population
        num_random = POPULATION_SIZE - num_parents - num_mutated
        random_population = np.random.choice(original_population, size=num_random, replace=False).tolist()

        # Combine all three groups to form the new population
        population = parents + mutated_population + random_population

    print("Evolutionary algorithm completed. Results saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    run_evolutionary_algorithm()
