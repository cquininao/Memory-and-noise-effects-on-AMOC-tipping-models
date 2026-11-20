# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# GENERAL FUNCTIONS
# ============================================================

def A(x, f):
    """
    Drift term of the AMOC system.
    """
    return -(1 + mu * np.abs(x - 1)) * x + f

def compute_overshooting_forcing(t, scenario):
    """
    Recomputes the deterministic overshooting freshwater forcing for
    the given scenario, 0 for scenario A, 1 for scenario B. Returns the
    forcing array FWF, with the same length as t.
    """
    nt_local = len(t)

    FWF0 = 0.53
    FWFlim = 0.565

    gamma = np.array([0.01910 * 0.07, 5 * 0.01910 * 0.07])
    mu0 = np.array([1.8e-3, 5 * 1.8e-3])
    mu1 = np.array([2e-7, 10 * 2e-7])

    mut = np.zeros(nt_local)
    FWF = np.zeros(nt_local)
    for i in range(nt_local):
        mut[i] = mu0[scenario] + mu1[scenario] * t[i]
        FWF[i] = (FWF0 + gamma[scenario] * t[i]
                  - (1 - np.exp(-mut[i] * t[i])) * (gamma[scenario] * t[i] - (FWFlim - FWF0)))

    if scenario == 0:
        AAA = (0.78 - 0.53) / (np.max(FWF) - 0.53)
    else:
        AAA = (0.84 - 0.53) / (np.max(FWF) - 0.53)
    BBB = 0.53 * (1 - AAA)
    FWF = FWF * AAA + BBB

    return FWF

def compute_deterministic_mean(forces_adim, t_adim, dt_adim, ntest):
    """
    Integrates the noise-free AMOC system over the full pool of
    initial conditions, and returns the
    mean trajectory in physical units of Sv.
    """
    nt_local = len(t_adim)
    X = np.zeros((ntest, nt_local))
    X[:, 0] = (ax * xx - (20 + 2 * np.random.randn(ntest)) / cc) / yd / ay

    for n in range(1, nt_local):
        X[:, n] = X[:, n - 1] + dt_adim * A(X[:, n - 1], forces_adim[n - 1])

    Xdim = yd * X
    QfluxX = cc * (ax * xx - ay * Xdim)
    return QfluxX.mean(axis=0)

def find_first_crossing_below(values, t):
    """
    Finds the first time at which values crosses below zero. Returns
    None if the trajectory never goes negative.
    """
    idx = np.where(values < 0)[0]
    if idx.size == 0:
        return None
    return t[idx[0]]


def find_first_recovery(values, t, t_reference):
    """
    Finds the first time after t_reference at which values crosses
    back above zero. Returns None if no such recovery occurs
    ."""
    idx_ref = int(np.searchsorted(t, t_reference))
    if idx_ref >= len(values):
        return None
    idx_recovery = np.where(values[idx_ref:] > 0)[0]
    if idx_recovery.size == 0:
        return None
    return t[idx_ref + idx_recovery[0]]


def find_first_crossing_above_threshold(values, t, threshold):
    """
    Finds the first time at which values exceeds the given threshold.
    Returns None if the threshold is never reached.
    """
    idx = np.where(values > threshold)[0]
    if idx.size == 0:
        return None
    return t[idx[0]]

def plot_deterministic_mean(mean_trajectory, t, t_crossing_det, t_recovery_det,
                              t_crit_constant_forcing, output_path):
    """Plots the mean trajectory of the deterministic ensemble over the
    full simulated time range. Background shading marks ON regions in
    light blue and OFF regions in light red, built from however many
    crossings are found, collapse only, or collapse and recovery. A
    horizontal line at zero marks the flux reversal threshold. Vertical
    lines mark the constant-forcing threshold, the deterministic
    collapse, and, if present, the deterministic recovery."""
    plt.figure(figsize=(8, 3.5))

    # Background shading, built from the sequence of ON, OFF, ON regions
    t_start = t[0]
    t_end = t[-1]
    regions = []
    cursor = t_start
    state = 'ON'
    if t_crossing_det is not None:
        regions.append((cursor, t_crossing_det, state))
        cursor = t_crossing_det
        state = 'OFF'
    if t_recovery_det is not None:
        regions.append((cursor, t_recovery_det, state))
        cursor = t_recovery_det
        state = 'ON'
    regions.append((cursor, t_end, state))

    color_by_state = {'ON': 'tab:blue', 'OFF': 'tab:red'}
    for region_start, region_end, region_state in regions:
        plt.axvspan(region_start, region_end, color=color_by_state[region_state],
                    alpha=0.06, zorder=0)

    plt.plot(t, mean_trajectory, linewidth=1.5)

    # Flux reversal threshold
    plt.axhline(0, color='tab:red', linewidth=1, linestyle='--')
    plt.annotate('Flux reversal threshold',
                 xy=(t_end, 0),
                 xytext=(-8, 5), textcoords='offset points',
                 fontsize=8, color='tab:red', ha='right', va='bottom')

    # Constant-forcing threshold
    if t_crit_constant_forcing is not None:
        plt.axvline(t_crit_constant_forcing, color='tab:gray', linewidth=2, linestyle=':')
        plt.annotate('Constant-forcing threshold',
                     xy=(t_crit_constant_forcing, -8),
                     xytext=(5, 0), textcoords='offset points',
                     fontsize=8, color='tab:gray', ha='left', va='center', rotation=0)

    # Deterministic collapse
    if t_crossing_det is not None:
        plt.axvline(t_crossing_det, color='k', linewidth=2, linestyle='--')
        plt.annotate('Deterministic collapse',
                     xy=(t_crossing_det, 12.5),
                     xytext=(8, 0), textcoords='offset points',
                     fontsize=8, color='black', ha='left', va='center', rotation=90)

    # Deterministic recovery
    if t_recovery_det is not None:
        plt.axvline(t_recovery_det, color='darkgreen', linewidth=2, linestyle='-.')
        plt.annotate('Deterministic recovery',
                     xy=(t_recovery_det, 20),
                     xytext=(8, 0), textcoords='offset points',
                     fontsize=8, color='darkgreen', ha='left', va='center', rotation=90)

    plt.ylim([-10, 25])
    plt.xlim([t_start, t_end])
    plt.xlabel('Time (years)', fontsize=12)
    plt.ylabel('Strength of the AMOC (Sv)', fontsize=12)
    plt.tick_params(axis='both', labelsize=12)
    plt.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.20)
    plt.savefig(output_path)
    plt.show()


def plot_forcing_ramp(t, FWF, fd, f_fold_dim, output_path):
    """
    Plots the overshooting freshwater forcing over time, with two
    horizontal dashed reference lines. The line at fd marks the onset
    of bistability, labeled Bistability onset. The line at f_fold_dim
    marks the saddle-node point, labeled Constant-forcing threshold.
    """
    plt.figure(figsize=(8, 3.5))
    plt.plot(t, FWF, linewidth=2)

    plt.axhline(fd, color='k', linewidth=1, linestyle='--')
    plt.annotate('Bistability onset',
                 xy=(t[-1], fd),
                 xytext=(-8, 5), textcoords='offset points',
                 fontsize=8, color='black', ha='right', va='bottom')

    plt.axhline(f_fold_dim, color='tab:gray', linewidth=1, linestyle=':')
    plt.annotate('Constant-forcing threshold',
                 xy=(t[-1], f_fold_dim),
                 xytext=(-8, 5), textcoords='offset points',
                 fontsize=8, color='tab:gray', ha='right', va='bottom')

    plt.xlim([t[0], t[-1]])
    plt.xlabel('Time (years)', fontsize=12)
    plt.ylabel('Freshwater forcing (Sv)', fontsize=12)
    plt.tick_params(axis='both', labelsize=12)
    plt.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.20)
    plt.savefig(output_path)
    plt.show()


# ============================================================
# SYSTEM PARAMETERS
# Must match exactly what is used in the main simulation script.
# ============================================================
mu = 4.2
ax = 1.7e-4
ay = 7.5e-4
xx = 25
cc = 5145
yd = 5.67
fd = 0.46

# Time mesh
ntest_det = 100000
td = 124
dt_adim = 0.01
t_adim = np.arange(0, 4000 / td, dt_adim)
t = t_adim * td
nt = len(t)

f_fold_adim = (1 + mu) ** 2 / (4 * mu)
f_fold_dim = fd * f_fold_adim
print(f"f_fold (Sv) = {f_fold_dim:.6f}")

script_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# MAIN LOOP
# ============================================================
np.random.seed(5)

for scenario_str, scenario in [('A', 0), ('B', 1)]:
    FWF = compute_overshooting_forcing(t, scenario)
    forces_adim = FWF / fd

    mean_trajectory = compute_deterministic_mean(forces_adim, t_adim, dt_adim, ntest_det)

    t_crossing_det = find_first_crossing_below(mean_trajectory, t)
    t_recovery_det = None
    if t_crossing_det is not None:
        t_recovery_det = find_first_recovery(mean_trajectory, t, t_crossing_det)

    t_crit_constant_forcing = find_first_crossing_above_threshold(FWF, t, f_fold_dim)

    print(f"Scenario {scenario_str}")
    print(f"  t_crossing_det (years) = {t_crossing_det}")
    print(f"  t_recovery_det (years) = {t_recovery_det}")
    print(f"  t_crit_constant_forcing (years) = {t_crit_constant_forcing}")

    mean_plot_path = os.path.join(script_dir, f'figura_4_AMOC_flux_mean_({scenario_str}).pdf')
    plot_deterministic_mean(mean_trajectory, t, t_crossing_det, t_recovery_det,
                             t_crit_constant_forcing, mean_plot_path)

    forcing_plot_path = os.path.join(script_dir, f'figura_4_FWF_({scenario_str}).pdf')
    plot_forcing_ramp(t, FWF, fd, f_fold_dim, forcing_plot_path)
