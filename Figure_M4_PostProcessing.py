# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, FormatStrFormatter
from matplotlib.cm import ScalarMappable

# ============================================================
# LIST OF FILES SELECTED FOR PLOTTING
# ============================================================
SELECTED_FILES = [
    #"figura_4_(A-H=0p5-sigma0=0p2-sigma1=0p0).npy",
    #"figura_4_(A-H=0p5-sigma0=0p2-sigma1=0p2).npy",
    "figura_4_(A-H=0p75-sigma0=0p2-sigma1=0p2).npy",
    #"figura_4_(B-H=0p5-sigma0=0p2-sigma1=0p0).npy",
    #"figura_4_(B-H=0p5-sigma0=0p2-sigma1=0p2).npy",
    "figura_4_(B-H=0p75-sigma0=0p2-sigma1=0p2).npy",
]

# ============================================================
# TIME AND SPACE GRID
# ============================================================
td = 124
dt_adim = 0.01
t_adim = np.arange(0, 4000 / td, dt_adim)
t = t_adim * td
nt = len(t)

nx = 101
x = np.linspace(-10, 25, nx)
dx = x[1] - x[0]
FF, XX = np.meshgrid(t, x[1:])

# Shared color scale, same for every panel and for the standalone bar
VMIN = 1e-3
VMAX = 1e0
NIVELES = np.linspace(1e-4, 1, 200)

# ============================================================
# FILE NAME PARSING
# ============================================================

FILENAME_PATTERN = re.compile(
    r'figura_4_\(([AB])-H=([0-9p]+)-sigma0=([0-9p]+)-sigma1=([0-9p]+)\)\.npy'
)

def parse_params_from_filename(filename):
    """
    Extracts scenario, H, sigma0 and sigma1 strings from the raw
    output file name. Raises an error if the file name does not match
    the expected pattern.
    """
    basename = os.path.basename(filename)
    match = FILENAME_PATTERN.match(basename)
    if match is None:
        raise ValueError(f"File name does not match the expected pattern, {basename}")
    scenario_str, H_str, sigma0_str, sigma1_str = match.groups()
    return scenario_str, H_str, sigma0_str, sigma1_str

# ============================================================
# MAIN
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(script_dir, "Figura4_raw")

for filename in SELECTED_FILES:
    raw_path = os.path.join(raw_dir, filename)

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"{raw_path} not found, check the file name")

    scenario_str, H_str, sigma0_str, sigma1_str = parse_params_from_filename(raw_path)

    QfluxX = np.load(raw_path)

    if QfluxX.shape[1] != nt:
        raise ValueError(
            f"Time grid mismatch for {filename}, "
            f"expected nt={nt}, got {QfluxX.shape[1]}. "
            "Check that dt_adim and the time range match the simulation script."
        )

    # --------------------------------------------------------
    # DENSITY HISTOGRAM PER TIME COLUMN
    # --------------------------------------------------------
    PDFx = np.zeros((nx - 1, nt))
    for n in range(nt):
        PDFx[:, n] = np.histogram(QfluxX[:, n], bins=x)[0]

    PDFxn = PDFx / np.sum(PDFx, axis=0) / dx

    # --------------------------------------------------------
    # FIGURE, DENSITY, NO COLORBAR
    # --------------------------------------------------------
    plt.figure(figsize=(8, 3.5))
    plt.contourf(FF, XX, np.maximum(1e-3, PDFxn), levels=NIVELES,
                 cmap='viridis', norm=LogNorm(vmin=VMIN, vmax=VMAX))
    plt.xlim([t[0], t[-1]])
    plt.ylim([x[0], x[-1]])
    plt.xlabel('Time (years)', fontsize=12)
    plt.ylabel('Strength of the AMOC (Sv)', fontsize=12)
    plt.tick_params(axis='both', labelsize=12)
    plt.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.20)

    savepath_density = os.path.join(
        script_dir,
        f'figura_M4_({scenario_str}-H={H_str}-sigma0={sigma0_str}-sigma1={sigma1_str}).pdf'
    )
    plt.savefig(savepath_density)
    plt.close()
    print(f"Saved {savepath_density}")

# ============================================================
# STANDALONE COLORBAR
# ============================================================
fig_cbar = plt.figure(figsize=(8, 1.0))
ax_cbar = fig_cbar.add_axes([0.10, 0.5, 0.80, 0.3])

sm = ScalarMappable(norm=LogNorm(vmin=VMIN, vmax=VMAX), cmap='viridis')
cbar = fig_cbar.colorbar(sm, cax=ax_cbar, orientation='horizontal',
                          ticks=LogLocator(base=10.0))
cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.0e'))
cbar.ax.tick_params(labelsize=12)

savepath_colorbar = os.path.join(script_dir, 'figura_M4_colorbar.pdf')
fig_cbar.savefig(savepath_colorbar)
plt.close(fig_cbar)
print(f"Saved {savepath_colorbar}")
