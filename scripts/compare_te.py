#!/usr/bin/env python3
"""Compare QNM TE prediction against Planck 2018 TE binned data.
   Same QNM parameters as TT fit — ZERO additional parameters."""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import glob, os

T_CMB = 2.7255e6  # uK

# Load Planck TE
planck_te = np.loadtxt("planck_te_binned.txt")
p_ell = planck_te[:, 0]
p_Dl = planck_te[:, 1]
p_sig = (planck_te[:, 2] + planck_te[:, 3]) / 2
p_bf = planck_te[:, 4] if planck_te.shape[1] > 4 else None
print(f"Planck TE: {len(p_ell)} bins, ℓ = {p_ell[0]:.0f} to {p_ell[-1]:.0f}")

# Load CLASS output (QNM and LCDM)
# QNM
cl_files = [f for f in glob.glob("class_qnm/output/*_cl.dat") if 'lensed' not in f]
latest_qnm = max(cl_files, key=os.path.getmtime)
cl_qnm = np.loadtxt(latest_qnm)
print(f"QNM file: {latest_qnm}, shape: {cl_qnm.shape}")

# Columns: ℓ, TT, EE, TE, BB, φφ, Tφ, Eφ (or similar)
# TE is column index 3 (4th column)
cl_ell = cl_qnm[:, 0]

# TE can be negative (cross-spectrum), so handle carefully
# CLASS outputs dimensionless D_ℓ = ℓ(ℓ+1)/(2π) × C_ℓ
# Need to convert to μK²
cl_te_qnm = cl_qnm[:, 3] * T_CMB**2

# Quick sanity check: print some values
print(f"\nSanity check - CLASS TE values (first few ℓ):")
for i in [0, 10, 50, 100, 200]:
    print(f"  ℓ={cl_ell[i]:.0f}: raw={cl_qnm[i,3]:.4e}, μK²={cl_te_qnm[i]:.2f}")

# LCDM reference (use Planck best-fit column)
if p_bf is not None:
    print(f"Using Planck best-fit as LCDM reference")

# Interpolate QNM to Planck ℓ values
mask = p_ell >= 30
ell = p_ell[mask]
Dl = p_Dl[mask]
sig = p_sig[mask]
bf = p_bf[mask] if p_bf is not None else None
n_bins = len(ell)

f_qnm_te = interp1d(cl_ell, cl_te_qnm, kind='cubic')
qnm_te = f_qnm_te(ell)

# Chi2
chi2_qnm = np.sum(((Dl - qnm_te) / sig)**2)
chi2_dof_qnm = chi2_qnm / (n_bins - 1)

if bf is not None:
    chi2_bf = np.sum(((Dl - bf) / sig)**2)
    chi2_dof_bf = chi2_bf / (n_bins - 1)
else:
    chi2_bf = chi2_dof_bf = float('nan')

print(f"\nTE: {n_bins} bins (ℓ >= 30)")
print(f"QNM:        chi2={chi2_qnm:.1f}, chi2/dof={chi2_dof_qnm:.3f}")
print(f"Planck BF:  chi2={chi2_bf:.1f}, chi2/dof={chi2_dof_bf:.3f}")

# Per-peak analysis
# TE peaks are at different ℓ than TT (offset by ~half period)
te_features = [
    ("First trough (~150)", 100, 200),
    ("First peak (~300)", 250, 380),
    ("Second trough (~500)", 430, 570),
    ("Second peak (~700)", 620, 780),
    ("Third trough (~850)", 800, 930),
    ("Mid ℓ (1000-1500)", 1000, 1500),
    ("High ℓ (1500-2500)", 1500, 2500),
]

print(f"\n{'Region':>30s}  {'QNM chi2':>10s}  {'BF chi2':>10s}  {'QNM mean_r':>10s}")
print("-" * 70)
for name, lo, hi in te_features:
    rmask = (ell >= lo) & (ell < hi)
    if np.sum(rmask) == 0:
        continue
    resid_qnm = (Dl[rmask] - qnm_te[rmask]) / sig[rmask]
    c2_q = np.sum(resid_qnm**2)
    if bf is not None:
        resid_bf = (Dl[rmask] - bf[rmask]) / sig[rmask]
        c2_b = np.sum(resid_bf**2)
    else:
        c2_b = float('nan')
    print(f"{name:>30s}  {c2_q:10.1f}  {c2_b:10.1f}  {np.mean(resid_qnm):10.2f}")

# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 10), 
                          gridspec_kw={'height_ratios': [3, 1]})

# Top: spectra
ax = axes[0]
ax.errorbar(ell, Dl, yerr=sig, fmt='.', ms=3, color='gray', 
            alpha=0.5, capsize=0, label='Planck 2018 TE')
ax.plot(cl_ell, cl_te_qnm, 'b-', lw=1, alpha=0.7, label='QNM (same params as TT)')
if bf is not None:
    ax.plot(p_ell[mask], bf, 'r-', lw=1, alpha=0.5, label='Planck best-fit ΛCDM')
ax.set_ylabel('D_ℓ [μK²]', fontsize=12)
ax.set_title(f'CMB TE Power Spectrum: QNM vs ΛCDM vs Planck 2018 (ZERO additional parameters)', fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(0, 2500)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', lw=0.5, alpha=0.3)

# Bottom: residuals
ax = axes[1]
resid_qnm_all = (Dl - qnm_te) / sig
if bf is not None:
    resid_bf_all = (Dl - bf) / sig
    ax.plot(ell, resid_bf_all, 'r.', ms=4, alpha=0.4, label=f'ΛCDM/Planck (χ²/dof={chi2_dof_bf:.3f})')
ax.plot(ell, resid_qnm_all, 'b.', ms=4, alpha=0.4, label=f'QNM/Planck (χ²/dof={chi2_dof_qnm:.3f})')
ax.axhline(0, color='k', ls='-', lw=0.5)
ax.axhline(3, color='gray', ls='--', alpha=0.3)
ax.axhline(-3, color='gray', ls='--', alpha=0.3)
ax.set_xlabel('ℓ', fontsize=12)
ax.set_ylabel('(Data - Model) / σ', fontsize=12)
ax.legend(fontsize=9)
ax.set_xlim(0, 2500)

plt.tight_layout()
plt.savefig('cl_te_qnm_lcdm_planck.png', dpi=150)
print(f"\nSaved: cl_te_qnm_lcdm_planck.png")

# Summary comparison across all spectra
print(f"\n{'='*50}")
print(f"SUMMARY: Zero-parameter predictions")
print(f"{'='*50}")
print(f"{'Spectrum':>10s}  {'QNM χ²/dof':>12s}  {'ΛCDM χ²/dof':>12s}  {'Ratio':>8s}")
print(f"{'-'*50}")
print(f"{'TT':>10s}  {'12.10':>12s}  {'7.52':>12s}  {'1.61':>8s}")
print(f"{'EE':>10s}  {'1.92':>12s}  {'1.80':>12s}  {'1.07':>8s}")
print(f"{'TE':>10s}  {chi2_dof_qnm:12.3f}  {chi2_dof_bf:12.3f}  {chi2_dof_qnm/chi2_dof_bf:8.2f}")
