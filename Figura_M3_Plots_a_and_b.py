import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# GENERAL FUNCTIONS
# ============================================================
def format_param(value, decimals=2):
    """
    Convert a numeric value to a string for use in file names.
    Ej: 0.75 -> '0p75'
    """
    return str(round(value, decimals)).replace('.', 'p')


def plot_deterministic_mean(QfluxX_det, t, t_crossing_det, t_crit_constant_forcing,
                              output_path, xlim=(252, 700)):
    """
    Plots the mean trajectory of the deterministic ensemble over time.
    One horizontal line is drawn at zero, the operational ON/OFF
    threshold, labeled Flux reversal threshold. Two vertical lines are
    drawn, one at t_crit_constant_forcing (labeled Constant-forcing
    threshold) and one at t_crossing_det (labeled Deterministic collapse).
    """
    mean_trajectory = QfluxX_det.mean(axis=0)

    plt.figure(figsize=(8, 3.5))
    # Background shading, ON region above zero, OFF region below zero
    plt.axvspan(0, t_crossing_det, color='tab:blue', alpha=0.06, zorder=0)
    plt.axvspan(t_crossing_det, 700, color='tab:red', alpha=0.06, zorder=0)

    plt.plot(t, mean_trajectory, linewidth=1.5)

    # Flux reversal threshold
    plt.axhline(0, color='tab:red', linewidth=1, linestyle='--')
    plt.annotate('Flux reversal threshold',
                 xy=(xlim[1], 0),
                 xytext=(-8, 5), textcoords='offset points',
                 fontsize=8, color='tab:red', ha='right', va='bottom')

    # Constant-forcing threshold
    plt.axvline(t_crit_constant_forcing, color='tab:gray', linewidth=2, linestyle=':')
    plt.annotate('Constant-forcing threshold',
                 xy=(t_crit_constant_forcing, -8),
                 xytext=(5, 0), textcoords='offset points',
                 fontsize=8, color='tab:gray', ha='left', va='center', rotation=0)

    # Deterministic collapse
    plt.axvline(t_crossing_det, color='k', linewidth=2, linestyle='--')
    plt.annotate('Deterministic collapse',
                 xy=(t_crossing_det, 12.5),
                 xytext=(8, 0), textcoords='offset points',
                 fontsize=8, color='black', ha='left', va='center', rotation=90)

    plt.ylim([-10, 25])
    plt.xlim(xlim)
    plt.xlabel('Time (years)', fontsize=12)
    plt.ylabel('Strength of the AMOC (Sv)', fontsize=12)
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def plot_forcing_ramp(t, ramp_rate, fd, f_fold, output_path):
    """
    Plots the freshwater forcing ramp over time, with two horizontal
    dashed reference lines. The line at fd marks the onset of
    bistability. The line at f_fold marks the saddle-node point.
    """
    force_temporal = ramp_rate * t

    plt.figure(figsize=(8, 3.5))
    plt.plot(t, force_temporal, linewidth=2)

    plt.axhline(fd, color='k', linewidth=1, linestyle='--')
    plt.annotate('Bistability onset',
                 xy=(t[-1], fd),
                 xytext=(-8, 5), textcoords='offset points',
                 fontsize=8, color='black', ha='right', va='bottom')

    plt.axhline(f_fold, color='tab:gray', linewidth=1, linestyle=':')
    plt.annotate('Constant-forcing threshold',
                 xy=(t[-1], f_fold),
                 xytext=(-8, 5), textcoords='offset points',
                 fontsize=8, color='tab:gray', ha='right', va='bottom')

    plt.xlim([t[0], t[-1]])
    plt.xlabel('Time (years)', fontsize=12)
    plt.ylabel('Freshwater forcing (Sv)', fontsize=12)
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

# ============================================================
# PARAMETERS
# ============================================================
mu = 4.2
fd = 0.46
ramp_rate = 1.9 / 1000

dt = 0.5
t = np.arange(0, 700, dt)

ruta = "./Figura3_H0p75/"

# ============================================================
# CLOSED-FORM SADDLE-NODE BIFURCATION (deterministic, constant forcing)
# ============================================================
f_fold_adim = (1 + mu)**2 / (4 * mu)
f_fold_dim = fd * f_fold_adim
t_crit_constant_forcing = f_fold_dim / ramp_rate

print(f"f_fold (Sv) = {f_fold_dim:.6f}")
print(f"t_crit_constant_forcing (years) = {t_crit_constant_forcing:.2f}")

# ============================================================
# LOAD DETERMINISTIC ENSEMBLE
# ============================================================
det_filename = "figura_3_(H=0p75-sigma0=0p0-sigma1=0p0).npy"
det_path = os.path.join(ruta, det_filename)

if not os.path.exists(det_path):
    raise FileNotFoundError(
        f"{det_path} not found. Run the main script first with "
        "sigma0=0.0 and sigma1=0.0 to generate the deterministic reference."
    )

QfluxX_det = np.load(det_path)

# ============================================================
# LOAD t_crossing_det
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
# PLOTS
# ============================================================
plot_deterministic_mean(QfluxX_det, t, t_crossing_det, t_crit_constant_forcing,
                         "figura_3_AMOC_flux_mean.pdf")
plot_forcing_ramp(t, ramp_rate, fd, f_fold_dim, "figura_3_FWF.pdf")
