#!/bin/bash
set -e

# Define directories for outputs
OUTPUT_DIR="./plots/grid_data"
TEMP_DIR="${OUTPUT_DIR}/temp"
mkdir -p "$TEMP_DIR"

# Notify that simulation tasks are about to start
echo "Generating parameter combinations and launching simulations..."

# Generate the list of parameter combinations.
# For radii, we use 10 values from np.linspace(radii_max, radii_min, 10)
# where radii_min = 0.05*2.7 and radii_max = 0.45*2.7,
# and for heights, 10 values from 10 to 1.
python3 - <<'EOF'
import numpy as np
import itertools

radii_min = 0.05 * 2.7
radii_max = 0.45 * 2.7
radiis = np.linspace(radii_max, radii_min, 10)
heights = np.linspace(10, 1, 10)

with open("params.txt", "w") as f:
    for r1, r2, r3, r4, h in itertools.product(radiis, radiis, radiis, radiis, heights):
        # Each line: r1 r2 r3 r4 height
        f.write(f"{r1} {r2} {r3} {r4} {h}\n")
EOF

# Use GNU parallel to run the simulation tasks concurrently.
# --jobs 128 tells it to run 128 tasks concurrently.
# --bar shows a progress bar, and --verbose will print the commands being executed.
parallel --jobs 128 --bar --verbose --colsep ' ' \
    'python3 simulation_task.py --r1 {1} --r2 {2} --r3 {3} --r4 {4} --height {5} --output '"$TEMP_DIR"'/temp_{1}_{2}_{3}_{4}_{5}.csv' \
    :::: params.txt

# After all simulations are complete, aggregate the temporary CSV files into the final CSV.
FINAL_OUTPUT="${OUTPUT_DIR}/grid_data.csv"
# Write the header once (from the first file found)
head -n 1 "$(find "$TEMP_DIR" -type f | head -n 1)" > "$FINAL_OUTPUT"
# Append the data (skip headers) from every temporary file
for file in "$TEMP_DIR"/*.csv; do
    tail -n +2 "$file" >> "$FINAL_OUTPUT"
done

echo "All simulations complete. Aggregated results saved to $FINAL_OUTPUT"

