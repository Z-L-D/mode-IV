#!/usr/bin/env python3
"""Diagnose WHERE the chi2 comes from in ell-space."""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

T_CMB = 2.7255e6  # uK

# Load data
planck = np.loadtxt("planck_tt_binned.txt")
p_ell, p_Dl, p_sig = planck[:,0], planck[:,1], (planck[:,2]+planck[:,3])/2

# Load QNM model (latest output)
import glob, os
cl_files = [f for f in glob.glob("class_qnm/output/*_cl.dat") if 'lensed' not in f]
latest = max(cl_files, key=os.path.getmtime)
print(f"Using: {latest}")
cl = np.loadtxt(latest)
cl_ell = cl[:,0]
cl_tt = cl[:,1] * T_CMB**2

# Also load LCDM
lcdm_files = [f for f in glob.glob("class_qnm/output/*_cl.dat") if 'lensed' not in f]
# We need a reference LCDM run — use the Planck best-fit column
p_bf = planck[:,4] if planck.shape[1] > 4 else None

# Interpolate QNM to Planck ells
mask = p_ell >= 30
ell = p_ell[mask]
Dl = p_Dl[mask]
sig = p_sig[mask]

f_qnm = interp1d(cl_ell, cl_tt, kind='cubic')
qnm_at_planck = f_qnm(ell)

# Per-bin chi2
chi2_per_bin = ((Dl - qnm_at_planck) / sig)**2
residual = (Dl - qnm_at_planck) / sig  # signed

# Cumulative chi2
cumul_chi2 = np.cumsum(chi2_per_bin)
total_chi2 = cumul_chi2[-1]

# Split by ell ranges
ranges = [
    ("Low ell (30-200)", 30, 200),
    ("Peak 1 region (200-350)", 200, 350),
    ("Trough 1 (350-450)", 350, 450),
    ("Peak 2 region (450-650)", 450, 650),
    ("Trough 2 (650-750)", 650, 750),
    ("Peak 3 region (750-950)", 750, 950),
    ("Mid ell (950-1500)", 950, 1500),
    ("High ell (1500-2500)", 1500, 2500),
]

print(f"\nTotal chi2 = {total_chi2:.1f}, chi2/dof = {total_chi2/(len(ell)-1):.3f}")
print(f"\n{'Region':>30s}  {'chi2':>8s}  {'N_bins':>6s}  {'chi2/bin':>8s}  {'% of total':>10s}")
print("-" * 75)

for name, l_lo, l_hi in ranges:
    rmask = (ell >= l_lo) & (ell < l_hi)
    if np.sum(rmask) == 0:
        continue
    region_chi2 = np.sum(chi2_per_bin[rmask])
    n_bins = np.sum(rmask)
    print(f"{name:>30s}  {region_chi2:8.1f}  {n_bins:6d}  {region_chi2/n_bins:8.2f}  {region_chi2/total_chi2*100:9.1f}%")

# Mean residual by region (shows DIRECTION of mismatch)
print(f"\n{'Region':>30s}  {'mean resid':>10s}  {'direction':>10s}")
print("-" * 55)
for name, l_lo, l_hi in ranges:
    rmask = (ell >= l_lo) & (ell < l_hi)
    if np.sum(rmask) == 0:
        continue
    mr = np.mean(residual[rmask])
    direction = "QNM HIGH" if mr < 0 else "QNM LOW"
    print(f"{name:>30s}  {mr:10.2f}  {direction:>10s}")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Top: Data vs model
ax = axes[0]
ax.errorbar(ell, Dl, yerr=sig, fmt='.', ms=3, color='gray', alpha=0.5, label='Planck 2018')
ax.plot(cl_ell, cl_tt, 'b-', lw=1, alpha=0.7, label='QNM')
if p_bf is not None:
    ax.plot(p_ell[mask], planck[mask,4], 'r-', lw=1, alpha=0.5, label='Planck best-fit')
ax.set_ylabel('D_l [μK²]')
ax.set_title('TT Power Spectrum')
ax.legend()
ax.set_xlim(0, 2500)

# Middle: Signed residual
ax = axes[1]
ax.plot(ell, residual, 'b.', ms=3, alpha=0.5)
ax.axhline(0, color='k', ls='-', lw=0.5)
ax.axhline(3, color='r', ls='--', alpha=0.3)
ax.axhline(-3, color='r', ls='--', alpha=0.3)
# Smoothed residual
window = 5
if len(residual) > window:
    smooth_r = np.convolve(residual, np.ones(window)/window, mode='same')
    ax.plot(ell, smooth_r, 'r-', lw=2, alpha=0.7, label=f'{window}-bin smooth')
ax.set_ylabel('(Data - QNM) / σ')
ax.set_title('Signed Residual')
ax.legend()
ax.set_xlim(0, 2500)

# Bottom: Cumulative chi2 fraction
ax = axes[2]
ax.plot(ell, cumul_chi2 / total_chi2 * 100, 'b-', lw=2)
ax.set_xlabel('ℓ')
ax.set_ylabel('Cumulative χ² (%)')
ax.set_title(f'Cumulative χ² (total = {total_chi2:.0f})')
ax.set_xlim(0, 2500)
ax.grid(True, alpha=0.3)
# Mark 50% and 80% levels
for pct in [50, 80]:
    idx = np.searchsorted(cumul_chi2 / total_chi2 * 100, pct)
    if idx < len(ell):
        ax.axhline(pct, color='gray', ls=':', alpha=0.5)
        ax.axvline(ell[idx], color='r', ls=':', alpha=0.5)
        ax.text(ell[idx]+20, pct+2, f'ℓ={ell[idx]:.0f}', fontsize=9, color='r')

plt.tight_layout()
plt.savefig('chi2_diagnostic.png', dpi=150)
print(f"\nSaved: chi2_diagnostic.png")
