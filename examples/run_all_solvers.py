# filename: examples/run_all_solvers.py

import subprocess
import os
import sys

# --- Configuration ---
# In run_all_solvers.py
solvers = ["fomaml", "reptile", "neumann", "maml"] # <-- Add 'penalty'
# --- IMPORTANT: Correct Path Configuration ---
# Get the absolute path to the directory where this script is located (i.e., .../imaml_dev/examples)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the data directory by going up one level and then into omniglot/python
# This makes the script robust and runnable from anywhere.
DATA_PATH = os.path.abspath(os.path.join(script_dir, "..", "omniglot/python"))

# Add a check to ensure the data directory actually exists.
if not os.path.isdir(DATA_PATH):
    print(f"FATAL ERROR: Data directory not found at the calculated path: {DATA_PATH}")
    print("Please ensure your omniglot data is located at '<project_root>/omniglot/python'")
    sys.exit(1) # Exit the script if the data isn't found
# --- End of Path Fix ---


META_STEPS = 2000
TASK_MB_SIZE = 16
N_WAY = 5
K_SHOT = 1
USE_GPU = "True"

# --- Experiment Loop ---
for solver in solvers:
    # Save results in a directory at the project root level
    save_dir = os.path.join(script_dir, "..", f"results/{solver}")
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*50)
    print(f" RUNNING EXPERIMENT FOR: {solver.upper()}")
    print(f" Results will be saved in: {save_dir}")
    print("="*50 + "\n")

    cmd = [
        "python",
        "other_methods.py",
        "--method", solver,
        "--save_dir", save_dir,
        "--data_dir", DATA_PATH, # Pass the absolute path
        "--meta_steps", str(META_STEPS),
        "--task_mb_size", str(TASK_MB_SIZE),
        "--N_way", str(N_WAY),
        "--K_shot", str(K_SHOT),
        "--use_gpu", USE_GPU
    ]
    
    if solver == "maml":
        print("WARNING: 'maml' method is very slow due to second-order gradients.")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Experiment for {solver.upper()} failed with exit code {e.returncode}.")
        continue

print("\nAll experiments finished.")