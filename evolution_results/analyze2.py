import os
import sys
import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = './plots/'

def plot_fitness_trends(data, output_dir):
    if 'generation' not in data.columns:
        raise ValueError("The 'generation' column is missing from the data.")
    
    generations = sorted(data['generation'].unique())
    avg_fitness = []
    max_fitness = []

    for gen in generations:
        gen_data = data[data['generation'] == gen]
        avg_fitness.append(gen_data['fitness'].mean())
        max_fitness.append(gen_data['fitness'].max())

    plt.figure()
    plt.plot(generations, avg_fitness, label='Average Fitness', color='blue')
    plt.plot(generations, max_fitness, label='Highest Fitness', color='red')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Trends Across Generations')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'fitness_trends.png'))
    plt.close()

def plot_best_and_average_fitness(data, output_dir):
    if 'generation' not in data.columns:
        raise ValueError("The 'generation' column is missing from the data.")
    
    generations = sorted(data['generation'].unique())
    avg_fitness = []
    best_fitness = []

    for gen in generations:
        gen_data = data[data['generation'] == gen]
        avg_fitness.append(gen_data['fitness'].mean())
        best_fitness.append(gen_data['fitness'].max())

    plt.figure()
    plt.plot(generations, avg_fitness, label='Average Fitness', color='blue', linestyle='--')
    plt.plot(generations, best_fitness, label='Best Fitness', color='red', linestyle='-')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Best and Average Fitness Per Generation')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'best_and_average_fitness.png'))
    plt.close()

def plot_highest_transmission(data, output_dir):
    generations = sorted(data['generation'].unique())
    selected_generations = [generations[0], generations[len(generations)//2], generations[-1]]

    for gen in selected_generations:
        gen_data = data[data['generation'] == gen]
        max_transmission_row = gen_data.loc[gen_data['transmission'].idxmax()]
        radii_list = ast.literal_eval(max_transmission_row['radii list'])  # Convert radii list to Python list
        height = max_transmission_row['height']
        
        # Handle cases where transmission is a single value
        transmission = max_transmission_row['transmission']
        if isinstance(transmission, str):
            transmission = ast.literal_eval(transmission)  # Convert string to list if necessary
        if not isinstance(transmission, (list, np.ndarray)):
            transmission = [transmission] * 40  # Repeat the single value for all wavelengths
        
        wavelength = np.linspace(3, 5, len(transmission))

        plt.figure()
        plt.plot(wavelength, transmission, label=f'Gen {gen} - Height: {height}, Radii: {radii_list}')
        plt.xlabel('Wavelength (microns)')
        plt.ylabel('Transmission')
        plt.title(f'Highest Transmission - Generation {gen}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'highest_transmission_gen_{gen}.png'))
        plt.close()

def plot_best_fitness(data, output_dir, last_generation_file):
    # Load the last generation file
    last_gen_data = pd.read_csv(last_generation_file)
    
    max_fitness_row = last_gen_data.loc[last_gen_data['fitness'].idxmax()]
    radii_list = ast.literal_eval(max_fitness_row['radii list'])  # Convert radii list to Python list
    height = max_fitness_row['height']
    
    # Ensure transmission is a list of values
    transmission = max_fitness_row['transmission']
    if isinstance(transmission, str):
        transmission = ast.literal_eval(transmission)  # Convert string to list if necessary
    if not isinstance(transmission, (list, np.ndarray)):
        raise ValueError("Transmission data is not a list of values.")

    # Generate wavelength range
    wavelength = np.linspace(3, 5, len(transmission))

    irst_gen_data = pd.read_csv('generation2_1.csv')
    middle_gen_data = pd.read_csv('generation2_26.csv')

    # Plot the transmission vs wavelength
    plt.figure()
    plt.plot(wavelength, transmission, label='Transmission', color='blue')
    plt.xlabel('Wavelength (microns)')
    plt.ylabel('Transmission')
    plt.ylim(0, 1)  # Match the y-axis limits from analyze.py
    plt.title(f"Transmission vs Wavelength\nBest Fitness Configuration - Height: {height}, Radii: {radii_list}")
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plot_file = os.path.join(output_dir, 'best_fitness_configuration.png')
    plt.savefig(plot_file)
    print(f"Plot saved to: {plot_file}")

    # Show the plot (if running locally with GUI support)
    #plt.show()

def plot_violin_fitness(data, output_dir):
    generations = sorted(data['generation'].unique())
    fitness_values = [data[data['generation'] == gen]['fitness'].values for gen in generations]

    plt.figure()
    plt.violinplot(fitness_values, showmeans=True, showextrema=True)
    plt.xticks(range(1, len(generations) + 1), generations, rotation=45)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Distribution Across Generations')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'violin_fitness.png'))
    plt.close()

def find_last_generation_file(data_dir):
    """Find the most recent generation CSV file in the specified directory."""
    generation_files = [f for f in os.listdir(data_dir) if f.startswith('generation2_') and f.endswith('.csv')]
    if not generation_files:
        raise FileNotFoundError("No generation files found in the specified directory.")
    
    # Sort files by generation number
    generation_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    return os.path.join(data_dir, generation_files[-1])

def analyze_results(file_path):
    file_path_list = ['generation2_1.csv', 'generation2_26.csv', 'generation2_51.csv']
    plt.figure()
    labels = ['Generation 1', 'Generation 25', 'Generation 50']
    colors = ['blue', 'green', 'red']
    for file_path, labeler, color in zip(file_path_list, labels, colors):
        # Load the CSV file
        data = pd.read_csv(file_path)

        # Find the row with the maximum transmission
        max_transmission_row = data.loc[data['transmission'].idxmax()]
        max_transmission = max_transmission_row['transmission']
        max_transmission_config = {
            "height": max_transmission_row['height'],
            "radii list": ast.literal_eval(max_transmission_row['radii list'])
        }

        # Find the row with the maximum fitness
        max_fitness_row = data.loc[data['fitness'].idxmax()]
        max_fitness = max_fitness_row['fitness']
        max_fitness_config = {
            "height": max_fitness_row['height'],
            "radii list": ast.literal_eval(max_fitness_row['radii list'])
        }

        # Calculate the average fitness
        avg_fitness = data['fitness'].mean()

        # Print the results
        print("Analysis Results:")
        print(f"Max Transmission: {max_transmission}")
        print(f"Configuration for Max Transmission: {max_transmission_config}")
        print(f"Max Fitness: {max_fitness}")
        print(f"Configuration for Max Fitness: {max_fitness_config}")
        print(f"Average Fitness: {avg_fitness}")

        # Filter data for the configuration with the best fitness
        best_fitness_data = data[
            (data['height'] == max_transmission_config['height']) &
            (data['radii list'] == str(max_transmission_config['radii list']))
        ]

        # Plot transmission vs wavelength
        
        plt.plot(best_fitness_data['wavelength'], best_fitness_data['transmission'], color=color, label=labeler)
        
    plt.xlabel('Wavelength (microns)')
    plt.ylabel('Transmission')
    plt.ylim(0, 1)
    plt.title(f"Transmission vs Wavelength\nBest Fitness Configuration - Height: {max_fitness_config['height']}, Radii: {max_fitness_config['radii list']}")
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plot_file = './plots/best_fitness_tranmission_generations.png'
    plt.savefig(plot_file)
    print(f"Plot saved to: {plot_file}")
    plt.show()
    # Show the plot (if running locally with GUI support)
    #plt.show()

    

def process_generation_files(data_dir):
    """Process all generation CSV files and calculate max and average fitness per generation."""
    generation_files = [
        f for f in os.listdir(data_dir) if f.startswith('generation2_') and f.endswith('.csv')
    ]
    if not generation_files:
        raise FileNotFoundError("No generation files found in the specified directory.")

    # Sort files by generation number
    generation_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    max_fitness_per_generation = []
    avg_fitness_per_generation = []
    generations = []

    for gen_file in generation_files:
        gen_number = int(gen_file.split('_')[1].split('.')[0])
        generations.append(gen_number)

        # Load the CSV file
        file_path = os.path.join(data_dir, gen_file)
        data = pd.read_csv(file_path)

        # Ensure the 'fitness' column exists
        if 'fitness' not in data.columns:
            raise ValueError(f"The file {gen_file} does not contain a 'fitness' column.")

        # Calculate max and average fitness
        max_fitness = data['fitness'].max()
        avg_fitness = data['fitness'].mean()

        max_fitness_per_generation.append(max_fitness)
        avg_fitness_per_generation.append(avg_fitness)

    return generations, max_fitness_per_generation, avg_fitness_per_generation

def plot_fitness_trends(generations, max_fitness, avg_fitness, output_dir):
    """Plot max and average fitness trends across generations."""
    plt.figure()
    plt.plot(generations, avg_fitness, label='Average Fitness', color='blue', linestyle='--')
    plt.plot(generations, max_fitness, label='Max Fitness', color='red', linestyle='-')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Max and Average Fitness Across Generations')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_file = os.path.join(output_dir, 'fitness_trends_across_generations.png')
    plt.savefig(plot_file)
    print(f"Fitness trends plot saved to: {plot_file}")

    # Show the plot (if running locally with GUI support)
    #plt.show()

def analyze_all_generations(data_dir):
    """Analyze all generation files and plot fitness trends."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process all generation files
    generations, max_fitness, avg_fitness = process_generation_files(data_dir)

    # Plot fitness trends
    plot_fitness_trends(generations, max_fitness, avg_fitness, OUTPUT_DIR)
    analyze_results(os.path.join(data_dir, find_last_generation_file(data_dir)))
if __name__ == "__main__":
    data_dir = './'
    try:
        analyze_all_generations(data_dir)
    except Exception as e:
        print(f"Error: {e}")
