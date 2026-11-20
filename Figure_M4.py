# -*- coding: utf-8 -*-
import os
import numpy as np
from fbm import FBM
from tqdm import tqdm
from joblib import Parallel, delayed

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
# MODEL PARAMETERS
# ============================================================

# Physical system parameters
mu = 4.2       # nondimensional
yd = 5.67      # psu
fd = 0.46      # Sv
td = 124       # years
ax = 1.7e-4    # 1/C
ay = 7.5e-4    # 1/psu
xx = 25        # C
cc = 5145      # Sv

# Time mesh
ntest = 100000
dt_adim = 0.01
t_adim = np.arange(0, 4000 / td, dt_adim)
t = t_adim * td
nt = len(t)

# Diffusion parameters
H = 0.75
sigma0 = 0.2
sigma1 = 0.2
scenario = 0  # 0 selects scenario A, 1 selects scenario B

# Strings derived automatically (do not touch)
H_str = format_param(H)
sigma0_str = format_param(sigma0)
sigma1_str = format_param(sigma1)
scenario_str = 'A' if scenario == 0 else 'B'

# Chunking and parallelization parameters
chunk_size = 1000
n_jobs = -1
assert ntest % chunk_size == 0, "ntest must be a multiple of chunk_size"
n_chunks = ntest // chunk_size

# ============================================================
# OVERSHOOTING FRESHWATER FORCING
# ============================================================

# Current and limit freshwater forcing values
FWF0 = 0.53
FWFlim = 0.565

# Coordinates [Blue curve, Red curve]
gamma = np.array([0.01910 * 0.07, 5 * 0.01910 * 0.07])
mu0 = np.array([1.8e-3, 5 * 1.8e-3])
mu1 = np.array([2e-7, 10 * 2e-7])

mut = np.zeros(nt)
FWF = np.zeros(nt)
for i in range(nt):
    mut[i] = mu0[scenario] + mu1[scenario] * t[i]
    FWF[i] = (FWF0 + gamma[scenario] * t[i]
              - (1 - np.exp(-mut[i] * t[i])) * (gamma[scenario] * t[i] - (FWFlim - FWF0)))

if scenario == 0:
    AAA = (0.78 - 0.53) / (np.max(FWF) - 0.53)
else:
    AAA = (0.84 - 0.53) / (np.max(FWF) - 0.53)
BBB = 0.53 * (1 - AAA)
FWF = FWF * AAA + BBB

forces_adim = FWF / fd

# ============================================================
# SIMULATION
# ============================================================
X = np.zeros((ntest, nt))
X[:, 0] = (ax * xx - (20 + 2 * np.random.randn(ntest)) / cc) / yd / ay

for c in tqdm(range(n_chunks), desc="Trajectory chunks"):
    idx_start = c * chunk_size
    idx_end = idx_start + chunk_size
    X_chunk = X[idx_start:idx_end, :]

    seeds = np.random.randint(0, 2**31 - 1, size=chunk_size)
    results = Parallel(n_jobs=n_jobs)(
        delayed(generate_fgn)(seeds[k], nt, H, t_adim[-1] + dt_adim)
        for k in range(chunk_size)
    )
    dF_chunk = np.array(results)

    for n in range(1, nt):
        X_chunk[:, n] = (X_chunk[:, n - 1]
                         + dt_adim * A(X_chunk[:, n - 1], forces_adim[n - 1])
                         + (sigma0 + sigma1 * X_chunk[:, n - 1]) * dF_chunk[:, n - 1])

Xdim = yd * X
QfluxX = cc * (ax * xx - ay * Xdim)

# ============================================================
# SAVING RAW ENSEMBLE OUTPUT
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "Figura4_raw")
os.makedirs(output_dir, exist_ok=True)

savepath_npy = os.path.join(
    output_dir,
    f'figura_4_({scenario_str}-H={H_str}-sigma0={sigma0_str}-sigma1={sigma1_str}).npy'
)
np.save(savepath_npy, QfluxX.astype(np.float32))
print(f"Saved ensemble output to {savepath_npy}")
