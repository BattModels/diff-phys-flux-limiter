import os
import sys
import subprocess
from itertools import product

# -----------------------
# Sweep definitions
# -----------------------
# (A) Width sweep: fixed depth
width_sweep = [32, 64, 128]
fixed_layers_for_width = 5

# (B) Depth sweep: fixed width
depth_sweep = [3, 5, 7]
fixed_hidden_for_depth = 64

# Optional: keep activation fixed for this sweep
activation = "relu" # "tanh"

# Optinal: Fix cg for now
cg_list = [8]

# Optional: wandb naming prefix
sweep_tag = "net_size_sweep"

# Path to training script
script_path = os.path.abspath("src/train_fl_linear_adv.py")

# -----------------------
# Helper to run one job
# -----------------------
def run_one(n_hidden: int, n_layers: int, cg: int, extra_overrides=None):
    extra_overrides = extra_overrides or []

    # Hydra overrides
    overrides = [
        f"net.n_hidden={n_hidden}",
        f"net.n_layers={n_layers}",
        f"net.activation={activation}",
        f"data.CG={cg}",
        # Make each run write to a unique hydra dir:
        f"hydra.run.dir=outputs/{sweep_tag}/CG{cg}_L{n_layers}_H{n_hidden}",
        # If you want distinct wandb run names:
        f"wandb.log=True",
        f"+wandb.run_name=CG{cg}_L{n_layers}_H{n_hidden}",
    ] + extra_overrides

    cmd = [sys.executable, script_path] + overrides
    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)

# -----------------------
# Execute sweep
# -----------------------
# 1) Width sweep (depth fixed)
for cg, n_hidden in product(cg_list, width_sweep):
    run_one(n_hidden=n_hidden, n_layers=fixed_layers_for_width, cg=cg)

# 2) Depth sweep (width fixed)
for cg, n_layers in product(cg_list, depth_sweep):
    run_one(n_hidden=fixed_hidden_for_depth, n_layers=n_layers, cg=cg)

print("\nAll sweeps finished.")

