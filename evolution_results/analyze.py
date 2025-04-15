import sys
import pandas as pd
import ast

def analyze_results(file_path):
    try:
        # Load the CSV file
        data = pd.read_csv(file_path)

        # Ensure the 'fitness' column exists
        if 'fitness' not in data.columns:
            raise ValueError("The provided file does not contain a 'fitness' column.")

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

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze.py <path_to_csv_file>")
    else:
        analyze_results(sys.argv[1])
