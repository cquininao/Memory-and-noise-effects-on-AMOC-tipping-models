import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def generate_lighter_shades(color, n=3):
    base = mcolors.to_rgb(color)
    shades = [(1 - i/n) * np.array(base) + (i/n) * np.array([1, 1, 1]) for i in range(n)]
    return shades

# Enable LaTeX rendering for text
plt.rcParams['text.usetex'] = True

# Initial parameters
mu = 4.2  # nondimensional
ax = 1.7e-4  # 1/C
ay = 7.5e-4  # 1/psu
xx = np.array([25, 21, 17])  # C
cc = 5145  # Sv
yd = 5.67  # psu
fd = 0.46  # Sv
td = 124  # years

# Threshold values in dimensional units (Sv)
bistability_onset = 0.46
constant_forcing_threshold = 0.74

# Forcing values and simulation range
forces = np.arange(0, 1, 0.001)
forces_adim = forces / fd

# Steady states
ceros = np.zeros((3, len(forces)))
ceros[0, :] = (mu - 1 + np.sqrt((mu - 1)**2 + 4 * mu * forces_adim)) / (2 * mu)
ceros[0, :] *= (np.imag(ceros[0, :]) == 0) * (ceros[0, :] > 1)

ceros[1, :] = (1 + mu - np.sqrt((1 + mu)**2 - 4 * mu * forces_adim)) / (2 * mu)
ceros[1, :] *= (np.imag(ceros[1, :]) == 0) * (ceros[1, :] < 1)

ceros[2, :] = (1 + mu + np.sqrt((1 + mu)**2 - 4 * mu * forces_adim)) / (2 * mu)
ceros[2, :] *= (np.imag(ceros[2, :]) == 0) * (ceros[2, :] < 1)

f1 = forces[ceros[0, :] > 0]
f2 = forces[ceros[1, :] > 0]
f3 = forces[ceros[2, :] > 0]

# Corresponding AMOC strength values
yf1 = np.zeros((len(xx), len(f1)))
yf2 = np.zeros((len(xx), len(f2)))
yf3 = np.zeros((len(xx), len(f3)))

for i in range(len(xx)):
    yf1[i, :] = cc * (ax * xx[i] - ay * yd * ceros[0, ceros[0, :] > 0])
    yf2[i, :] = cc * (ax * xx[i] - ay * yd * ceros[1, ceros[1, :] > 0])
    yf3[i, :] = cc * (ax * xx[i] - ay * yd * ceros[2, ceros[2, :] > 0])

# Plot
plt.figure(figsize=(10, 6))

base_color = '#1f77b4'
shades = generate_lighter_shades(base_color, n=len(xx))

for i, shade in enumerate(shades):
    plt.plot(f1, yf1[i, :], color=shade, linewidth=2.0, label=f'$\Delta$T : {xx[i]}')
    plt.plot(f2, yf2[i, :], color=shade, linewidth=2.0)
    plt.plot(f3, yf3[i, :], linestyle='--', color=shade, linewidth=2.0)

# Threshold lines
plt.axvline(bistability_onset, color='black', linestyle='--', linewidth=1.5)
plt.axvline(constant_forcing_threshold, color='gray', linestyle=':', linewidth=1.5)
offset = 0.01  # small horizontal offset, in units of the x-axis (Sv)

plt.text(bistability_onset - offset, 22, 'Bistability onset', rotation=0,
         verticalalignment='top', horizontalalignment='right', fontsize=11, color='black')
plt.text(constant_forcing_threshold + offset, 15, 'Constant-forcing threshold', rotation=90,
         verticalalignment='top', horizontalalignment='left', fontsize=11, color='gray')

plt.grid(True)
plt.grid(True, alpha=0.3)
plt.ylim([-10, 25])
plt.xlim([0, 1])
plt.xlabel('Fresh water forcing : f [Sv]', fontsize=14)
plt.ylabel('Strength of the AMOC : q [Sv]', fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("figura_S0_Sept_2026.pdf")
plt.show()
