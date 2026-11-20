import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tqdm import tqdm


# ============================================================
# GENERAL FUNCTIONS
# ============================================================

def format_param(value, decimals=2):
    """
    Convert a numeric value to a string for use in file names.
    Ej: 0.75 -> '0p75'
    """
    return str(round(value, decimals)).replace('.', 'p')

def compute_flo_fixed_threshold(QfluxX, ntest, Y_off_confirm=0.0):
    """
    Compute flo(t), the fraction of trajectories with AMOC flux below
    Y_off_confirm at each time step.
    """
    nt = QfluxX.shape[1]
    flo = np.zeros(nt)
    for ff in range(nt):
        flo[ff] = np.sum(QfluxX[:, ff] < Y_off_confirm) / ntest
    return flo

def interpolate_at(t, y, t_query):
    """
    Linearly interpolate y(t) at a single time t_query.
    """
    return float(np.interp(t_query, t, y))


# ============================================================
# PARAMETERS
# ============================================================
dt = 0.5
t = np.arange(0, 700, dt)
nt = len(t)

ruta = "./Figura3_H0p75/"

# Physical criterion for the OFF state, flux reversal
Y_off_confirm = 0.0

# Smoothing for display only; flo(t) remains unsmoothed for the analysis.
smoothing_window = 21
smoothing_polyorder = 1

# System parameters needed for the closed-form saddle-node calculation
mu = 4.2
fd = 0.46
ramp_rate = 1.9 / 1000  # Sv per year, matches force_temporal in the main script

# ============================================================
# CLOSED-FORM SADDLE-NODE BIFURCATION (deterministic, constant forcing)
# ============================================================
# The ON state lives on the branch x < 1, where A(x,f) = mu*x^2 - (1+mu)*x + f.
# The saddle-node occurs when the discriminant of this quadratic vanishes.
f_fold_adim = (1 + mu)**2 / (4 * mu)
f_fold_dim = fd * f_fold_adim
t_crit_constant_forcing = f_fold_dim / ramp_rate

print(f"f_fold (dimensionless) = {f_fold_adim:.6f}")
print(f"f_fold (Sv) = {f_fold_dim:.6f}")
print(f"t_crit_constant_forcing (years) = {t_crit_constant_forcing:.2f}\n")

# ============================================================
# INFER H AND sigma0 FROM THE FOLDER AND FILE NAMES
# ============================================================
folder_name = os.path.basename(os.path.normpath(ruta))
HTest_str = folder_name.replace("Figura3_H", "")
HTest = float(HTest_str.replace("p", "."))

archivos_all = [f for f in os.listdir(ruta) if f.endswith(".npy")]
if len(archivos_all) == 0:
    raise FileNotFoundError(f"No .npy files found in {ruta}")

pos0 = archivos_all[0].find("=", archivos_all[0].find("sigma0"))
pos1 = archivos_all[0].find("-", pos0)
sigma0Test_str = archivos_all[0][pos0 + 1:pos1]
sigma0Test = float(sigma0Test_str.replace("p", "."))

print(f"Inferred H = {HTest} (from folder name), sigma0 = {sigma0Test} (from first file found)\n")

# ============================================================
# READ DETERMINISTIC CROSSING TIME
# ============================================================
crossing_path = os.path.join(ruta, "deterministic_crossing.json")
if not os.path.exists(crossing_path):
    raise FileNotFoundError(
        "deterministic_crossing.json not found. "
        "Run the main script first with sigma0=0.0 and sigma1=0.0 "
        "to generate the deterministic reference before continuing."
    )
with open(crossing_path, "r") as f:
    crossing_info = json.load(f)
t_crossing_det = crossing_info["t_crossing_det"]

# ============================================================
# READ ENSEMBLE FILES AND COMPUTE flo(t) FOR EACH sigma1
# ============================================================
flo_by_sigma1 = {}
ntest = None

for nombre in tqdm(archivos_all, desc="Reading ensemble files"):
    pos0 = nombre.find("=")
    pos1 = nombre.find("-")
    H_str = nombre[pos0 + 1:pos1].replace("p", ".")
    H = float(H_str)

    pos0 = nombre.find("=", pos0 + 1)
    pos1 = nombre.find("-", pos1 + 1)
    sigma0_str = nombre[pos0 + 1:pos1].replace("p", ".")
    sigma0 = float(sigma0_str)

    pos0 = nombre.find("=", pos0 + 1)
    pos1 = nombre.find(")")
    sigma1_str = nombre[pos0 + 1:pos1].replace("p", ".")
    sigma1 = float(sigma1_str)

    if H == HTest and sigma0 == sigma0Test and sigma1 > 0:
        QfluxX = np.load(os.path.join(ruta, nombre))
        if ntest is None:
            ntest = QfluxX.shape[0]
        elif ntest != QfluxX.shape[0]:
            raise ValueError(
                f"Inconsistent ntest across files, expected {ntest} but "
                f"got {QfluxX.shape[0]} in {nombre}"
            )
        flo_by_sigma1[sigma1] = compute_flo_fixed_threshold(QfluxX, ntest, Y_off_confirm=Y_off_confirm)

if ntest is None:
    raise ValueError(f"No matching files found for H={HTest}, sigma0={sigma0Test}, sigma1>0 in {ruta}")

# ============================================================
# PRINT TABLE, flo AT BOTH REFERENCE TIMES, FOR EACH sigma1
# ============================================================
print(f"{'sigma1':>8} | {'flo at t_crit_constant_forcing':>32} | {'flo at t_crossing_det':>22}")
print("-" * 68)
for sigma1 in sorted(flo_by_sigma1.keys()):
    flo_raw = flo_by_sigma1[sigma1]
    flo_at_t_crit = interpolate_at(t, flo_raw, t_crit_constant_forcing)
    flo_at_t_crossing = interpolate_at(t, flo_raw, t_crossing_det)
    print(f"{sigma1:>8.2f} | {flo_at_t_crit:>32.4f} | {flo_at_t_crossing:>22.4f}")

# ============================================================
# PLOT, FIXED THRESHOLD RESULT WITH BOTH REFERENCE TIMES
# ============================================================
plt.figure(figsize=(8, 3.5))

for sigma1 in sorted(flo_by_sigma1.keys()):
    flo_raw = flo_by_sigma1[sigma1]
    flo_smooth = savgol_filter(flo_raw, window_length=smoothing_window, polyorder=smoothing_polyorder)
    plt.plot(t, flo_smooth, label=rf"$\sigma_1$ = {sigma1}")

# Constant-forcing threshold, vertical line with a vertically rotated
# text annotation instead of a legend entry
plt.plot([t_crit_constant_forcing, t_crit_constant_forcing], [0, 1], ':', color='gray', linewidth=2)
plt.annotate('Constant-forcing threshold',
             xy=(t_crit_constant_forcing, 0.5),
             xytext=(-8, 0), textcoords='offset points',
             fontsize=8, color='gray', ha='right', va='center', rotation=90)

# Deterministic collapse under the ramp, vertical line with a vertically
# rotated text annotation instead of a legend entry
plt.plot([t_crossing_det, t_crossing_det], [0, 1], '--k', linewidth=2)
plt.annotate('Deterministic collapse',
             xy=(t_crossing_det, 0.05),
             xytext=(8, 0), textcoords='offset points',
             fontsize=8, color='black', ha='left', va='center', rotation=0)

plt.legend(loc='lower left')
plt.xlim([252, 700])
plt.ylim([0, 1])
plt.xlabel('Time (years)', fontsize=12)
plt.ylabel('Probability of AMOC OFF State', fontsize=12)
plt.tick_params(axis='both', labelsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

postpath = f'figura_3_(H={HTest_str}-sigma0={sigma0Test_str}).pdf'
plt.savefig(postpath)
plt.show()
