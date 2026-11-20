import os
import json
import numpy as np
from fbm import FBM
from joblib import Parallel, delayed
from tqdm import tqdm

# ============================================================
# GENERAL SETTINGS
# ============================================================
np.random.seed(5)

# ============================================================
# GENERAL FUNCTIONS
# ============================================================
def format_param(value, decimals=2):
    """
    Convert a numeric value to a string for use in file names.
    Ej: 0.75 -> '0p75'
    """
    return str(round(value, decimals)).replace('.', 'p')

def A(x, f):
    """
    Compute the drift term of the AMOC system.
    """
    return -(1 + mu * np.abs(x - 1)) * x + f

def generate_fgn(seed, nt, hurst, length):
    """
    Generate a fractional Gaussian noise (fGn) trajectory.
    """
    np.random.seed(seed)
    f = FBM(n=nt, hurst=hurst, length=length, method='daviesharte')
    return f.fgn()[:nt - 1]

# ============================================================
# SYSTEM PARAMETERS
# ============================================================

# Physical system parameters
mu = 4.2       # nondimensional
ax = 1.7e-4    # 1/C
ay = 7.5e-4    # 1/psu
xx = 25        # C
cc = 5145      # Sv
yd = 5.67      # psu
fd = 0.46      # Sv
td = 124       # years

# ============================================================
# TIME MESH
# ============================================================
ntest = 20000
dt = 0.5
t = np.arange(0, 700, dt)
nt = len(t)
t_adim = t / td
dt_adim = t_adim[1] - t_adim[0]

# FWF
force_temporal = 1.9 / 1000 * t
force_adimensional = force_temporal / fd

# Diffusion parameters
H = 0.75
sigma0 = 0.0
sigma1 = 0.0

# Auxiliary derived strings
H_str = format_param(H)
sigma0_str = format_param(sigma0)
sigma1_str = format_param(sigma1)

is_deterministic_case = (sigma0 == 0.0 and sigma1 == 0.0)

# Chunking and parallelization parameters
chunk_size = 500
n_jobs = -1
assert ntest % chunk_size == 0, "ntest must be a multiple of chunk_size"
n_chunks = ntest // chunk_size

# ============================================================
# SIMULATION
# ============================================================
X = np.zeros((ntest, nt))
X[:, 0] = (ax * xx - (20 + 2 * np.random.randn(ntest)) / cc) / yd / ay

for c in tqdm(range(n_chunks), desc="Trajectory chunks"):
    idx_start = c * chunk_size
    idx_end = idx_start + chunk_size
    X_chunk = X[idx_start:idx_end, :]

    if is_deterministic_case:
        # No noise contribution, skip fgn generation entirely to save time
        dF_chunk = np.zeros((chunk_size, nt - 1))
    else:
        seeds = np.random.randint(0, 2**31 - 1, size=chunk_size)
        results = Parallel(n_jobs=n_jobs)(
            delayed(generate_fgn)(seeds[k], nt, H, t_adim[-1] + dt_adim)
            for k in range(chunk_size)
        )
        dF_chunk = np.array(results)

    for n in range(1, nt):
        X_chunk[:, n] = (X_chunk[:, n - 1] +
                         dt_adim * A(X_chunk[:, n - 1], force_adimensional[n - 1]) +
                         (sigma0 + sigma1 * X_chunk[:, n - 1]) * dF_chunk[:, n - 1])

QfluxX = cc * (ax * xx - ay * X * yd)

# ============================================================
# SAVE ENSEMBLE OUTPUT
# ============================================================
output_dir = f"Figura3_H{H_str}"
os.makedirs(output_dir, exist_ok=True)
savepath_npy = os.path.join(output_dir, f"figura_3_(H={H_str}-sigma0={sigma0_str}-sigma1={sigma1_str}).npy")
np.save(savepath_npy, QfluxX.astype(np.float32))
print(f"Saved ensemble output to {savepath_npy}")

# ============================================================
# DETERMINISTIC CROSSING TIME, ONLY COMPUTED FOR THE REFERENCE CASE
# ============================================================
if is_deterministic_case:
    QfluxX_mean = QfluxX.mean(axis=0)
    zero_crossings = np.where(QfluxX_mean < 0)[0]
    if zero_crossings.size > 0:
        t_crossing_det = t[zero_crossings[0]]
        print(f"Deterministic reference crosses zero at t = {t_crossing_det:.1f} years (index {zero_crossings[0]})")
    else:
        t_crossing_det = None
        print("The deterministic reference does not cross zero within the simulated time range")

    crossing_info = {"t_crossing_det": float(t_crossing_det) if t_crossing_det is not None else None}
    crossing_path = os.path.join(output_dir, "deterministic_crossing.json")
    with open(crossing_path, "w") as f:
        json.dump(crossing_info, f, indent=2)
    print(f"Saved deterministic crossing info to {crossing_path}")
