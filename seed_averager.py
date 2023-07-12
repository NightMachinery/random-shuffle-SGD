#!/usr/bin/env python3
##
import os
import glob
import json
import numpy as np

# Root directory where the logs are stored
root_dir = './results'

# Directory to store the averages
avg_dir = './seed_avg'

# If the average directory doesn't exist, create it
os.makedirs(avg_dir, exist_ok=True)

# Extract the batch size folders
batch_dirs = glob.glob(f"{root_dir}/batch_size_*")

# For each batch size
for bd in batch_dirs:
    batch_size = os.path.basename(bd).split('_')[-1]

    # Extract the algorithm files for each seed
    alg_files = glob.glob(f"{bd}/seed_*/*.log")

    # Group by algorithms
    alg_dict = {}
    for af in alg_files:
        alg_name = os.path.basename(af).split('.')[0]
        if alg_name not in alg_dict:
            alg_dict[alg_name] = []

        with open(af, 'r') as f:
            # Assume the log file contains a JSON array of numbers
            data = json.loads(f.read())
            alg_dict[alg_name].append(data)

    # Calculate and write out averages
    for alg, results in alg_dict.items():
        avg_results = np.mean(results, axis=0).tolist()

        # Form the output path
        out_path = os.path.join(avg_dir, f"batch_size_{batch_size}_{alg}.log")

        with open(out_path, 'w') as f:
            f.write(json.dumps(avg_results))

print("Completed the computation of averages.")
