import sys
import pandas as pd
import ast
import matplotlib.pyplot as plt

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

        # Filter data for the configuration with the best fitness
        best_fitness_data = data[
            (data['height'] == max_fitness_config['height']) &
            (data['radii list'] == str(max_fitness_config['radii list']))
        ]

        # Plot transmission vs wavelength
        plt.figure()
        plt.plot(best_fitness_data['wavelength'], best_fitness_data['transmission'], label='Transmission', color='blue')
        plt.xlabel('Wavelength (microns)')
        plt.ylabel('Transmission')
        plt.ylim(0, 1)
        from IPython import embed; embed()
        plt.title(f"Transmission vs Wavelength\nBest Fitness Configuration - Height:{max_fitness_config['height'].item()}, Radii:{max_fitness_config['radii list']}")
        plt.legend()
        plt.grid(True)

        # Save the plot to a file
        plot_file = file_path.replace('.csv', '_best_fitness_plot.png')
        plt.savefig(plot_file)
        print(f"Plot saved to: {plot_file}")

        # Show the plot (if running locally with GUI support)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze.py <path_to_csv_file>")
    else:
        analyze_results(sys.argv[1])
