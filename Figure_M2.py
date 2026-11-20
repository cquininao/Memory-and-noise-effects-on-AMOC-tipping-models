import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, FormatStrFormatter
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


def generar_fgn(seed, nt, hurst, length):
    """
    Generate a fractional Gaussian noise (fGn) trajectory.
    """
    np.random.seed(seed)
    f = FBM(n=nt, hurst=hurst, length=length, method='daviesharte')
    return f.fgn()[:nt - 1]


# ============================================================
# MODEL PARAMETERS
# ============================================================

# Parámetros físicos del sistema
mu = 4.2       # nondimensional
ax = 1.7e-4    # 1/C
ay = 7.5e-4    # 1/psu
xx = 25        # C
cc = 5145      # Sv
yd = 5.67      # psu
fd = 0.46      # Sv
td = 124       # years

# Diffusion parameters
H = 0.75
sigma0 = 0.0
sigma1 = 0.4

# Auxiliary derived strings
H_str = format_param(H)
sigma0_str = format_param(sigma0)
sigma1_str = format_param(sigma1)

# Time mesh
ntest = 10000
dt = 0.5
t = np.arange(0, 1500, dt)
nt = len(t)
t_adim = t / td
dt_adim = t_adim[1] - t_adim[0]

# FWF sweep
forces = np.arange(0, 1, 1e-2)
forces_adim = forces / fd
nf = len(forces)

# Chunking and parallelization parameters
chunk_size = 1000
n_jobs = -1
assert ntest % chunk_size == 0, "ntest must be a multiple of chunk_size"
n_chunks = ntest // chunk_size


# ============================================================
# SIMULATION
# ============================================================

Xdim = np.zeros((ntest, nf))

for nfi, fi in enumerate(tqdm(forces_adim, desc="Forzantes")):
    X = (ax * xx - (20 + 2 * np.random.randn(ntest)) / cc) / yd / ay

    for c in range(n_chunks):
        idx_start = c * chunk_size
        idx_end = idx_start + chunk_size

        seeds = np.random.randint(0, 2**31 - 1, size=chunk_size)

        resultados = Parallel(n_jobs=n_jobs)(
            delayed(generar_fgn)(seeds[k], nt, H, t_adim[-1] + dt_adim)
            for k in range(chunk_size)
        )
        dF_chunk = np.array(resultados)

        X_chunk = X[idx_start:idx_end]
        for n in range(1, nt):
            X_chunk[:] = (X_chunk + dt_adim * A(X_chunk, fi) +
                          (sigma0 + sigma1 * X_chunk) * dF_chunk[:, n - 1])

    Xdim[:, nfi] = yd * X

QfluxX = cc * (ax * xx - ay * Xdim)


# ============================================================
# POST-PROCESSING: DENSITY AS A FUNCTION OF FWF
# ============================================================

nx = 101
x = np.linspace(-10, 25, nx)
dx = x[1] - x[0]

PDF = np.zeros((nx - 1, nf))
for ff in range(nf):
    qAMOC = QfluxX[:, ff]
    hist, _ = np.histogram(qAMOC, bins=x)
    PDF[:, ff] = hist

PDFn = PDF / np.sum(PDF, axis=0) / dx

FF, XX = np.meshgrid(forces, x[1:])


# ============================================================
# PLOT: BIFURCATION DIAGRAM DENSITY vs FORCING
# ============================================================

plt.figure(figsize=(10, 6))
niveles = np.linspace(1e-3, 1, 200)
contour = plt.contourf(FF, XX, np.maximum(1e-3, PDFn), cmap='viridis',
                        levels=niveles, norm=LogNorm(vmin=1e-3, vmax=1e0))
plt.ylim([-10, 25])
plt.xlim([0, 1])
plt.xlabel('Fresh water forcing : f[Sv]', fontsize=14)
plt.ylabel('Strength of the AMOC : q[Sv]', fontsize=14)

cbar = plt.colorbar(ticks=LogLocator(base=10.0))
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))

postpath = f'figura_2_(H={H_str}-sigma0={sigma0_str}-sigma1={sigma1_str}).pdf'
plt.savefig(postpath)
