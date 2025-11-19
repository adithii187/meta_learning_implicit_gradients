
import subprocess
import os
import sys
import argparse

SOLVER_CONFIGS = {
    "fomaml":   {"outer_lr": "1e-3"},
    "reptile_new":  {"outer_lr": "0.01"},  # Uses a tuned, higher learning rate
    "neumann":  {"outer_lr": "1e-3"},
    "maml":     {"outer_lr": "1e-3"},
    "penalty":  {"outer_lr": "1e-3"}
}

# --- Command-line Argument Parsing ---
parser = argparse.ArgumentParser(description="Run a single meta-learning experiment.")
parser.add_argument('--method', type=str, required=True, choices=SOLVER_CONFIGS.keys(),
                    help="The solver method to run.")
parser.add_argument('--outer_lr', type=str, default=None, help="Override the default outer learning rate.")

args = parser.parse_args()

solver_to_run = args.method
config = SOLVER_CONFIGS[solver_to_run]

if args.outer_lr:
    config['outer_lr'] = args.outer_lr
    print(f"Overriding outer_lr to {args.outer_lr}")

# --- Path Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(script_dir, "..", "omniglot/python"))

if not os.path.isdir(DATA_PATH):
    print(f"FATAL ERROR: Data directory not found at: {DATA_PATH}")
    sys.exit(1)

# --- Hyperparameter Configuration ---
META_STEPS = 2000
TASK_MB_SIZE = 16
N_WAY = 5
K_SHOT = 1
USE_GPU = "True"

# --- Experiment Execution ---
save_dir = os.path.join(script_dir, "..", f"results/{solver_to_run}")
os.makedirs(save_dir, exist_ok=True)

print("\n" + "="*50)
print(f" RUNNING EXPERIMENT FOR: {solver_to_run.upper()}")
print(f" Using Config: {config}")
print(f" Results will be saved in: {save_dir}")
print("="*50 + "\n")

cmd = [
    "python", "other_methods.py",
    "--method", solver_to_run,
    "--save_dir", save_dir,
    "--data_dir", DATA_PATH,
    "--outer_lr", config["outer_lr"],
    "--meta_steps", str(META_STEPS),
    "--task_mb_size", str(TASK_MB_SIZE),
    "--N_way", str(N_WAY),
    "--K_shot", str(K_SHOT),
    "--use_gpu", USE_GPU
]

try:
    subprocess.run(cmd, check=True)
    print("\nExperiment finished successfully.")
except subprocess.CalledProcessError as e:
    print(f"\nERROR: Experiment for {solver_to_run.upper()} failed.")