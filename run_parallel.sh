#!/bin/bash
set -e

# Define directories for outputs
OUTPUT_DIR="./plots/grid_data"
TEMP_DIR="${OUTPUT_DIR}/temp"
mkdir -p "$TEMP_DIR"

# File to log failed jobs
FAILED_FILE="failed_jobs.txt"
# Clear previous log file
> "$FAILED_FILE"

echo "Generating parameter combinations..."

# Generate parameter combinations using an embedded Python script.
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

# Maximum jobs per batch and timeout in seconds (3 minutes)
MAX_JOBS=128
BATCH_TIMEOUT=180
job_count=0

# Arrays to keep track of the current batch's process IDs and parameters
declare -a batch_pids=()
declare -a batch_params=()

# Function to wait for the current batch of jobs with a timeout.
wait_batch() {
  echo "Waiting for a batch of ${#batch_pids[@]} job(s)..."
  start_time=$(date +%s)
  
  while true; do
    all_finished=true
    for pid in "${batch_pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        all_finished=false
        break
      fi
    done

    if $all_finished; then
      echo "Batch completed successfully."
      break
    fi

    now=$(date +%s)
    elapsed=$(( now - start_time ))
    if [ $elapsed -gt $BATCH_TIMEOUT ]; then
      echo "Batch timeout reached after ${elapsed} seconds. Killing remaining tasks..."
      for i in "${!batch_pids[@]}"; do
        pid="${batch_pids[$i]}"
        if kill -0 "$pid" 2>/dev/null; then
          kill -9 "$pid" 2>/dev/null
          # Log the failed job details to the failed jobs file.
          echo "Job [${batch_params[$i]}] failed (timeout after ${elapsed} seconds)." >> "$FAILED_FILE"
        fi
      done
      break
    fi
    sleep 5
  done
  
  # Wait for all background jobs to exit (should return quickly if already killed)
  for pid in "${batch_pids[@]}"; do
    wait "$pid" 2>/dev/null
  done
}

echo "Launching simulations..."

# Process each parameter set from the generated file.
while read r1 r2 r3 r4 h; do
  # Use round-robin core assignment (0-indexed)
  core=$(( job_count % MAX_JOBS ))
  output_file="${TEMP_DIR}/temp_${r1}_${r2}_${r3}_${r4}_${h}.csv"
  
  echo "Launching job with parameters: r1=$r1, r2=$r2, r3=$r3, r4=$r4, height=$h on core $core"
  
  # Launch the simulation bound to a specific core via taskset.
  taskset -c "$core" python3 simulation_task.py --r1 "$r1" --r2 "$r2" --r3 "$r3" --r4 "$r4" --height "$h" --output "$output_file" &
  pid=$!
  batch_pids+=("$pid")
  batch_params+=("r1=$r1 r2=$r2 r3=$r3 r4=$r4 h=$h")
  
  job_count=$((job_count+1))
  
  # If we've launched MAX_JOBS jobs, wait for the batch to finish (or timeout).
  if [ $(( job_count % MAX_JOBS )) -eq 0 ]; then
    wait_batch
    # Reset batch arrays
    batch_pids=()
    batch_params=()
  fi
done < params.txt

# Wait for any remaining jobs in the final batch.
if [ ${#batch_pids[@]} -gt 0 ]; then
  wait_batch
fi

echo "All simulations finished. Aggregating results..."

# Aggregate temporary CSV files into the final CSV.
FINAL_OUTPUT="${OUTPUT_DIR}/grid_data.csv"
# Write the header (taken from the first temporary file)
head -n 1 "$(find "$TEMP_DIR" -type f | head -n 1)" > "$FINAL_OUTPUT"
# Append data (skipping headers) from every temporary file
for file in "$TEMP_DIR"/*.csv; do
    tail -n +2 "$file" >> "$FINAL_OUTPUT"
done

echo "All simulations complete. Aggregated results saved to $FINAL_OUTPUT"
echo "Failed jobs (if any) are logged in $FAILED_FILE"