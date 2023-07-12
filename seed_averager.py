#!/usr/bin/env python3
##
import os
import glob
import json
import numpy as np
try:
    from icecream import ic, colorize as ic_colorize

    # ic.configureOutput(outputFunction=lambda s: print(ic_colorize(s)))
    ic.configureOutput(outputFunction=lambda s: print(s))
except ImportError:
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)


# Root directory where the logs are stored
root_dir = './results'

# Directory to store the averages
avg_dir = f'{root_dir}/seed_avg'

# If the average directory doesn't exist, create it
os.makedirs(avg_dir, exist_ok=True)

# Extract the batch size folders
batch_dirs = glob.glob(f"{root_dir}/batch_size_*")

# For each batch size
for bd in batch_dirs:
    batch_size = os.path.basename(bd).split('_')[-1]

    # Extract the algorithm files for each seed
    alg_files = glob.glob(f"{bd}/seed_*/*.log")
    if len(alg_files) == 0:
        continue

    print(f'\n----------------Batch Size {batch_size}-----------------\n')

    # Group by algorithms
    alg_dict = {}
    for af in alg_files:
        seed_num = os.path.basename(os.path.dirname(af)).split('_')[-1]

        alg_name = os.path.basename(af).split('.')[0]
        if alg_name not in alg_dict:
            alg_dict[alg_name] = []

        with open(af, 'r') as f:
            # Assume the log file contains a JSON array of numbers
            data = json.loads(f.read())
            ic(alg_name, seed_num, data)

            alg_dict[alg_name].append(data)

    print('\n----------------Averages-----------------\n')
    # Calculate and write out averages
    for alg, results in alg_dict.items():
        avg_results = np.mean(results, axis=0).tolist()

        # Round results to 4 decimal places
        avg_results = [round(num, 4) for num in avg_results]

        # Form the output path
        out_path = os.path.join(avg_dir, f"batch_size_{batch_size}_{alg}.log")

        ic(alg, avg_results)

        with open(out_path, 'w') as f:
            f.write(json.dumps(avg_results))

print('\n----------------Completed-----------------\n')
