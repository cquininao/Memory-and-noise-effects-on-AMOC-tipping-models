import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, FormatStrFormatter
from fbm import FBM
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.signal import argrelextrema

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


def find_peaks_and_valley(values, nbins=101, value_range=(-10, 25)):
    """
    Find the ON peak (global maximum), the OFF peak (local maximum to the
    left of the ON peak), and the valley (density minimum between them).

    Return (Y_off_peak, Y_valley, Y_on_peak). If there is no local OFF peak
    (e.g., for low H), Y_off_peak and Y_valley are None.
    """
    hist, bin_edges = np.histogram(values, bins=nbins, range=value_range, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    idx_on_peak = np.argmax(hist)
    Y_on_peak = bin_centers[idx_on_peak]

    idx_local_max = argrelextrema(hist, np.greater)[0]
    idx_local_max_left = idx_local_max[idx_local_max < idx_on_peak]

    if len(idx_local_max_left) == 0:
        return None, None, Y_on_peak

    idx_off_peak = idx_local_max_left[np.argmax(hist[idx_local_max_left])]
    idx_valley = idx_off_peak + np.argmin(hist[idx_off_peak:idx_on_peak + 1])

    Y_off_peak = bin_centers[idx_off_peak]
    Y_valley = bin_centers[idx_valley]

    return Y_off_peak, Y_valley, Y_on_peak


def get_fused_excursions(y, y_off_confirm, y_on_recovery):
    """
    Identify core events (Q < y_off_confirm) and merge consecutive events
    if the bridge between them never reaches y_on_recovery (no recovery to
    the ON state).

    Return the durations (in time steps) of the merged excursions. The last
    excursion is excluded if it is censored, i.e., if no recovery occurs
    before the end of the data.
    """
    core_mask = y < y_off_confirm
    n = len(y)
    padded = np.concatenate(([False], core_mask, [False]))
    diffs = np.diff(padded.astype(int))
    starts = list(np.where(diffs == 1)[0])
    ends = list(np.where(diffs == -1)[0])

    if len(starts) == 0:
        return np.array([])

    fused_starts = [starts[0]]
    fused_ends = [ends[0]]

    for i in range(1, len(starts)):
        gap = y[fused_ends[-1]:starts[i]]
        if gap.max() < y_on_recovery:
            fused_ends[-1] = ends[i]
        else:
            fused_starts.append(starts[i])
            fused_ends.append(ends[i])

    # Check censoring of the last excursion
    e_ultima = fused_ends[-1]
    if e_ultima >= n:
        censurado = True
    else:
        cola = y[e_ultima:]
        censurado = cola.max() < y_on_recovery if len(cola) > 0 else True

    if censurado:
        fused_starts = fused_starts[:-1]
        fused_ends = fused_ends[:-1]

    durations = np.array(fused_ends) - np.array(fused_starts)
    return durations


# ============================================================
# MODEL PARAMETERS
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

# FWF
forces = 0.50
forces_adim = forces / fd

# Diffusion parameters
H = 0.75
sigma0 = 0.2
sigma1 = 0.4

# Auxiliary derived strings
H_str = format_param(H)
sigma0_str = format_param(sigma0)
sigma1_str = format_param(sigma1)

# Time mesh
ntest = 10000
dt = 0.5
t = np.arange(0, 6000, dt)
nt = len(t)
t_adim = t / td
dt_adim = t_adim[1] - t_adim[0]

# Chunking and parallelization parameters
chunk_size = 1000
n_jobs = -1
assert ntest % chunk_size == 0, "ntest must be a multiple of chunk_size"
n_chunks = ntest // chunk_size

#  Physical criterion for a "core event" in the OFF state (flux reversal)
Y_off_confirm = 0.0  # Sv

# Recovery criterion: fractions of the sigma1-specific ON peak
recovery_fractions = [0.8, 0.9]


# ============================================================
# SIMULATION
# ============================================================

X = np.zeros((ntest, nt))
X[:, 0] = (ax * xx - (20 + 2 * np.random.randn(ntest)) / cc) / yd / ay

for c in tqdm(range(n_chunks), desc="Chunks"):
    idx_start = c * chunk_size
    idx_end = idx_start + chunk_size

    seeds = np.random.randint(0, 2**31 - 1, size=chunk_size)

    resultados = Parallel(n_jobs=n_jobs)(
        delayed(generar_fgn)(seeds[k], nt, H, t_adim[-1] + dt_adim)
        for k in range(chunk_size)
    )
    dF_chunk = np.array(resultados)

    X_chunk = X[idx_start:idx_end, :]
    for n in range(1, nt):
        X_chunk[:, n] = (X_chunk[:, n - 1] + dt_adim * A(X_chunk[:, n - 1], forces_adim) +
                          (sigma0 + sigma1 * X_chunk[:, n - 1]) * dF_chunk[:, n - 1])

Xdim = yd * X
QfluxX = cc * (ax * xx - ay * Xdim)

# ============================================================
# GENERAL DIAGNOSTICS: PEAKS AND VALLEYS
# ============================================================

idx_estacionario = t > 500  # Discard the initial transient
Q_estacionario = QfluxX[:, idx_estacionario]
valores_pool = Q_estacionario.ravel()

Y_off_peak, Y_valley, Y_on_peak = find_peaks_and_valley(valores_pool)

es_bimodal = Y_off_peak is not None

if es_bimodal:
    print(f"OFF peak={Y_off_peak:.3f} | Valley={Y_valley:.3f} | ON peak={Y_on_peak:.3f}")
else:
    print(f"Unimodal distribution (no local OFF peak detected) | ON peak={Y_on_peak:.3f}")

# Numerical diagnostics of how "real" the OFF peak and the valley are
hist_full, edges_full = np.histogram(valores_pool, bins=101, range=(-10, 25), density=True)
centers_full = 0.5 * (edges_full[:-1] + edges_full[1:])

idx_on = np.argmin(np.abs(centers_full - Y_on_peak))

if es_bimodal:
    idx_off = np.argmin(np.abs(centers_full - Y_off_peak))
    idx_valley_ = np.argmin(np.abs(centers_full - Y_valley))

    prominencia_off = hist_full[idx_off] - hist_full[idx_valley_]
    prominencia_on = hist_full[idx_on] - hist_full[idx_valley_]

    print(f"OFF prominence (peak height - valley height): {prominencia_off:.5f}")
    print(f"ON prominence (peak height - valley height): {prominencia_on:.5f}")
    print(f"OFF/ON ratio: {prominencia_off / prominencia_on:.3f}\n")
else:
    prominencia_off = np.nan
    prominencia_on = np.nan
    print("Prominences not applicable (unimodal distribution).\n")

# Visual check: full histogram + zoom on the OFF region
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(valores_pool, bins=101, range=(-10, 25), density=True, color='tab:gray')
if es_bimodal:
    axes[0].axvline(Y_off_peak, color='tab:blue', linestyle=':', label=f'OFF peak={Y_off_peak:.2f}')
    axes[0].axvline(Y_valley, color='tab:green', linestyle='--', label=f'Valley={Y_valley:.2f}')
axes[0].axvline(Y_on_peak, color='tab:purple', linestyle=':', label=f'ON peak={Y_on_peak:.2f}')
axes[0].axvline(Y_off_confirm, color='black', linestyle='-', linewidth=1, label='Y_off_confirm=0')
axes[0].set_xlabel('Y (Sv)', fontsize=11)
axes[0].set_ylabel('Density', fontsize=11)
axes[0].set_title('Full distribution', fontsize=11)
axes[0].legend(fontsize=7)

if es_bimodal:
    margen_zoom = 5
    rango_zoom = (-10, Y_valley + margen_zoom)

    hist_zoom, edges_zoom = np.histogram(valores_pool, bins=101, range=rango_zoom, density=True)
    centers_zoom = 0.5 * (edges_zoom[:-1] + edges_zoom[1:])
    ancho_zoom = edges_zoom[1] - edges_zoom[0]

    axes[1].bar(centers_zoom, hist_zoom, width=ancho_zoom, color='tab:gray', align='center')
    axes[1].axvline(Y_off_peak, color='tab:blue', linestyle=':', label=f'OFF peak={Y_off_peak:.2f}')
    axes[1].axvline(Y_valley, color='tab:green', linestyle='--', label=f'Valley={Y_valley:.2f}')
    axes[1].axvline(Y_off_confirm, color='black', linestyle='-', linewidth=1, label='Y_off_confirm=0')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Y (Sv)', fontsize=11)
    axes[1].set_ylabel('Density (log scale)', fontsize=11)
    axes[1].set_title('Zoom: OFF region', fontsize=11)
    axes[1].legend(fontsize=7)
else:
    axes[1].text(0.5, 0.5, 'No OFF peak detected\n(unimodal distribution)',
                 ha='center', va='center', fontsize=11, transform=axes[1].transAxes)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

plt.tight_layout()
plt.savefig(f'figura_check_peaks_(H={H_str}-sigma0={sigma0_str}-sigma1={sigma1_str}).pdf')

# ============================================================
# EXCURSION ANALYSIS: DURATION (EMPIRICAL CDF)
# ============================================================

results_summary = {}

for frac in recovery_fractions:
    y_on_recovery = frac * Y_on_peak

    all_durations_pasos = []
    for ni in range(ntest):
        durs = get_fused_excursions(Q_estacionario[ni], Y_off_confirm, y_on_recovery)
        all_durations_pasos.extend(durs)

    all_durations_pasos = np.array(all_durations_pasos)
    duraciones_anos = all_durations_pasos * dt

    results_summary[frac] = {
        'y_on_recovery': y_on_recovery,
        'n_excursions': len(duraciones_anos),
        'mean': duraciones_anos.mean(),
        'median': np.median(duraciones_anos),
        'durations': duraciones_anos
    }

    print(f"--- recovery fraction={frac} (Y_on_recovery={y_on_recovery:.3f} Sv) ---")
    print(f"N excursiones (uncensored): {len(duraciones_anos)}")
    print(f"Media: {duraciones_anos.mean():.1f} years | Median: {np.median(duraciones_anos):.1f} years\n")

# --- Comparación entre criterios de recuperación mediante función de supervivencia ---
plt.figure(figsize=(6, 4))
for frac in recovery_fractions:
    duraciones = np.sort(results_summary[frac]['durations'])
    n = len(duraciones)
    cdf = np.arange(1, n + 1) / n
    plt.plot(duraciones, 1 - cdf, label=f'recovery={frac} (n={n})')

plt.xlabel('Duration of AMOC OFF excursions (years)', fontsize=12)
plt.ylabel('Empirical CDF', fontsize=12)
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig(f'figura_duraciones_cdf_(H={H_str}-sigma0={sigma0_str}-sigma1={sigma1_str}).pdf')

# ============================================================
# POST-PROCESSING: DENSITY
# ============================================================

nx = 101
x = np.linspace(-10, 25, nx)
dx = x[1] - x[0]

tplot = t[::50]
QfluxX_slicing = QfluxX[:, ::50]
nt_plot = len(tplot)

PDFx = np.zeros((nx - 1, nt_plot))
for n in range(nt_plot):
    PDFx[:, n] = np.histogram(QfluxX_slicing[:, n], bins=x)[0]

PDFxn = PDFx / np.sum(PDFx, axis=0) / dx
TT, XX = np.meshgrid(tplot, x[0:-1])

# ============================================================
# GRAPHICS: DENSITY
# ============================================================

fig = plt.figure(figsize=(8, 3.5))
gs = fig.add_gridspec(1, 2, width_ratios=[25, 1], wspace=0.05)
ax_main = fig.add_subplot(gs[0])
cax = fig.add_subplot(gs[1])

niveles = np.linspace(1e-3, 1, 200)
contour = ax_main.contourf(TT, XX, np.maximum(1e-3, PDFxn), levels=niveles,
                             cmap='viridis', norm=LogNorm(vmin=1e-3, vmax=1e0))
ax_main.set_xlim([t[0], t[-1]])
ax_main.set_ylim([x[0], x[-1]])
ax_main.set_xlabel('Time (years)', fontsize=12)
ax_main.set_ylabel('Strength of the AMOC (Sv)', fontsize=12)
ax_main.tick_params(axis='both', labelsize=12)

cbar = fig.colorbar(contour, cax=cax, ticks=LogLocator(base=10.0))
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))
cbar.ax.tick_params(labelsize=12)
fig.subplots_adjust(left=0.10, right=0.88, bottom=0.20, top=0.95, wspace=0.05)

postpath = f'figura_1_densidad_(H={H_str}-sigma0={sigma0_str}-sigma1={sigma1_str}).pdf'
plt.savefig(postpath)

# ============================================================
# PLOT: ILLUSTRATIVE TRAJECTORIES (showing that excursions exist)
# ============================================================

mtraj = 5
minimos = np.min(QfluxX, axis=1)
indices_minimos = np.argsort(minimos)
qplot = QfluxX[indices_minimos[:mtraj]]

fig = plt.figure(figsize=(8, 3.5))
gs = fig.add_gridspec(1, 2, width_ratios=[25, 1], wspace=0.05)
ax_main = fig.add_subplot(gs[0])
cax = fig.add_subplot(gs[1])
cax.axis('off')

for k in range(mtraj):
    ax_main.plot(t, qplot[k])

ax_main.axhline(Y_off_confirm, color='black', linestyle='--', linewidth=1, alpha=0.7)
ax_main.set_ylim([x[0], x[-1]])
ax_main.set_xlim([t[0], t[-1]])
ax_main.set_xlabel('Time (years)', fontsize=12)
ax_main.set_ylabel('Strength of the AMOC (Sv)', fontsize=12)
ax_main.tick_params(axis='both', labelsize=12)
ax_main.grid(True, linestyle='--', alpha=0.7)

fig.subplots_adjust(left=0.10, right=0.88, bottom=0.20, top=0.95, wspace=0.05)
postpath = f'figura_1_trayectorias_(H={H_str}-sigma0={sigma0_str}-sigma1={sigma1_str}).pdf'
plt.savefig(postpath)

# ============================================================
# SAVING SUMMARY RESULTS (for later comparison across sigma1)
# ============================================================

nombre_archivo = f'resumen_excursiones_(H={H_str}-sigma0={sigma0_str}-sigma1={sigma1_str}).npz'

datos_a_guardar = {
    'H': H,
    'sigma0': sigma0,
    'sigma1': sigma1,
    'Y_off_peak': Y_off_peak if es_bimodal else np.nan,
    'Y_valley': Y_valley if es_bimodal else np.nan,
    'Y_on_peak': Y_on_peak,
    'prominencia_off': prominencia_off,
    'prominencia_on': prominencia_on,
}

for frac in recovery_fractions:
    frac_str = format_param(frac)
    datos_a_guardar[f'durations_frac{frac_str}'] = results_summary[frac]['durations']
    datos_a_guardar[f'n_excursions_frac{frac_str}'] = results_summary[frac]['n_excursions']
    datos_a_guardar[f'mean_frac{frac_str}'] = results_summary[frac]['mean']
    datos_a_guardar[f'median_frac{frac_str}'] = results_summary[frac]['median']

np.savez(nombre_archivo, **datos_a_guardar)
print(f"Summary saved to: {nombre_archivo}")