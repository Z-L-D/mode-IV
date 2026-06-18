#!/usr/bin/env python3
"""
Compare QNM and LCDM C_l predictions against Planck 2018 binned TT data.
Computes formal chi-squared for both models.
"""
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Load Planck 2018 binned TT data
# ============================================================
planck = np.loadtxt('planck_tt_binned.txt')
planck_ell = planck[:, 0]
planck_Dl  = planck[:, 1]    # D_l = l(l+1)C_l/(2pi) in uK^2
planck_err_m = planck[:, 2]  # symmetric errors
planck_err_p = planck[:, 3]
planck_best = planck[:, 4]   # Planck best-fit LCDM
planck_err = 0.5 * (planck_err_m + planck_err_p)  # average for chi2

print(f"Planck data: {len(planck_ell)} bins, ell = {planck_ell[0]:.0f} to {planck_ell[-1]:.0f}")

# ============================================================
# 2. Load CLASS outputs (QNM and LCDM)
# ============================================================
# Find the latest QNM output
import glob, os
qnm_files = sorted(glob.glob('class_qnm/output/qnm_test_*_cl.dat'), key=os.path.getmtime)
qnm_files = [f for f in qnm_files if 'lensed' not in f]
qnm_file = qnm_files[-1]

lcdm_files = sorted(glob.glob('class_qnm/output/lcdm_ref_*_cl.dat'), key=os.path.getmtime)
lcdm_files = [f for f in lcdm_files if 'lensed' not in f]
lcdm_file = 'class_vanilla/output/explanatory01_cl.dat'

print(f"QNM file:  {qnm_file}")
print(f"LCDM file: {lcdm_file}")

qnm_data = np.loadtxt(qnm_file)
lcdm_data = np.loadtxt(lcdm_file)

# CLASS output: col 0 = ell, col 1 = TT (D_l in dimensionless units)
class_ell_qnm = qnm_data[:, 0]
class_Dl_qnm  = qnm_data[:, 1]  # dimensionless D_l = l(l+1)C_l/(2pi)

class_ell_lcdm = lcdm_data[:, 0]
class_Dl_lcdm  = lcdm_data[:, 1]

# Convert CLASS dimensionless D_l to uK^2
# CLASS outputs (DeltaT/T)^2, multiply by T_CMB^2
T_CMB = 2.7255e6  # uK
T_CMB_sq = T_CMB ** 2
class_Dl_qnm_uK2  = class_Dl_qnm  * T_CMB_sq
class_Dl_lcdm_uK2 = class_Dl_lcdm * T_CMB_sq

print(f"\nCLASS QNM  peak D_l: {np.max(class_Dl_qnm_uK2):.1f} uK^2 at ell={class_ell_qnm[np.argmax(class_Dl_qnm_uK2)]:.0f}")
print(f"CLASS LCDM peak D_l: {np.max(class_Dl_lcdm_uK2):.1f} uK^2 at ell={class_ell_lcdm[np.argmax(class_Dl_lcdm_uK2)]:.0f}")
print(f"Planck     peak D_l: {np.max(planck_Dl):.1f} uK^2 at ell={planck_ell[np.argmax(planck_Dl)]:.0f}")

# ============================================================
# 3. Interpolate CLASS to Planck ell values
# ============================================================
qnm_at_planck  = np.interp(planck_ell, class_ell_qnm, class_Dl_qnm_uK2)
lcdm_at_planck = np.interp(planck_ell, class_ell_lcdm, class_Dl_lcdm_uK2)

# ============================================================
# 4. Compute chi-squared
# ============================================================
# Restrict to ell > 30 where Planck high-ell likelihood applies
mask = planck_ell >= 30
n_bins = np.sum(mask)

chi2_qnm = np.sum(((planck_Dl[mask] - qnm_at_planck[mask]) / planck_err[mask]) ** 2)
chi2_lcdm = np.sum(((planck_Dl[mask] - lcdm_at_planck[mask]) / planck_err[mask]) ** 2)
chi2_planck_best = np.sum(((planck_Dl[mask] - planck_best[mask]) / planck_err[mask]) ** 2)

# Parameters: QNM has 2 (A_QNM, gamma), LCDM has 2 (A_s, n_s)
# but full LCDM has ~6 params. For fair comparison use same dof.
n_params_qnm = 2   # A_QNM, gamma (alpha=0 is fixed)
n_params_lcdm = 2   # A_s, n_s (other params shared)
dof_qnm = n_bins - n_params_qnm
dof_lcdm = n_bins - n_params_lcdm

print(f"\n{'='*60}")
print(f"CHI-SQUARED COMPARISON (ell >= 30, {n_bins} bins)")
print(f"{'='*60}")
print(f"QNM:              chi2 = {chi2_qnm:.1f},  chi2/dof = {chi2_qnm/dof_qnm:.3f}")
print(f"CLASS LCDM:       chi2 = {chi2_lcdm:.1f},  chi2/dof = {chi2_lcdm/dof_lcdm:.3f}")
print(f"Planck best-fit:  chi2 = {chi2_planck_best:.1f},  chi2/dof = {chi2_planck_best/dof_lcdm:.3f}")
print(f"Degrees of freedom: {dof_qnm}")

# Per-peak chi2
peaks = [(180, 280, "Peak 1"), (450, 650, "Peak 2"), (700, 900, "Peak 3")]
print(f"\nPer-peak chi2:")
for l_lo, l_hi, name in peaks:
    pmask = (planck_ell >= l_lo) & (planck_ell <= l_hi)
    if np.sum(pmask) > 0:
        c2_q = np.sum(((planck_Dl[pmask] - qnm_at_planck[pmask]) / planck_err[pmask]) ** 2)
        c2_l = np.sum(((planck_Dl[pmask] - lcdm_at_planck[pmask]) / planck_err[pmask]) ** 2)
        nb = np.sum(pmask)
        print(f"  {name} (ell {l_lo}-{l_hi}, {nb} bins): QNM chi2={c2_q:.1f}, LCDM chi2={c2_l:.1f}")

# ============================================================
# 5. Plot
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})

# Top panel: D_l spectra
ax = axes[0]
ax.errorbar(planck_ell, planck_Dl, yerr=[planck_err_m, planck_err_p], 
            fmt='o', ms=3, color='gray', alpha=0.7, capsize=1, label='Planck 2018 data', zorder=1)
ax.plot(class_ell_lcdm, class_Dl_lcdm_uK2, 'r-', lw=1.5, label='ΛCDM (CLASS)', zorder=2)
ax.plot(class_ell_qnm, class_Dl_qnm_uK2, 'b-', lw=1.5, alpha=0.8, label='QNM (γ=1.04, α=0)', zorder=3)
ax.set_xlim(2, 2600)
ax.set_ylim(0, 7000)
ax.set_xlabel('ℓ', fontsize=12)
ax.set_ylabel('D_ℓ = ℓ(ℓ+1)C_ℓ/2π  [μK²]', fontsize=12)
ax.set_title('CMB TT Power Spectrum: QNM vs ΛCDM vs Planck 2018', fontsize=14)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

# Middle panel: residuals (QNM - Planck) / sigma
ax = axes[1]
resid_qnm = (qnm_at_planck - planck_Dl) / planck_err
resid_lcdm = (lcdm_at_planck - planck_Dl) / planck_err
ax.plot(planck_ell, resid_lcdm, 'r.', ms=4, alpha=0.6, label='ΛCDM residual')
ax.plot(planck_ell, resid_qnm, 'b.', ms=4, alpha=0.6, label='QNM residual')
ax.axhline(0, color='k', ls='--', lw=0.5)
ax.axhline(2, color='gray', ls=':', lw=0.5)
ax.axhline(-2, color='gray', ls=':', lw=0.5)
ax.set_xlim(2, 2600)
ax.set_ylim(-5, 5)
ax.set_xlabel('ℓ', fontsize=12)
ax.set_ylabel('(Model - Data) / σ', fontsize=12)
ax.set_title('Residuals', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Bottom panel: ratio QNM/Planck and LCDM/Planck
ax = axes[2]
ratio_qnm = qnm_at_planck / planck_Dl
ratio_lcdm = lcdm_at_planck / planck_Dl
ax.plot(planck_ell, ratio_lcdm, 'r.', ms=4, alpha=0.6, label='ΛCDM/Planck')
ax.plot(planck_ell, ratio_qnm, 'b.', ms=4, alpha=0.6, label='QNM/Planck')
ax.axhline(1.0, color='k', ls='--', lw=0.5)
ax.set_xlim(2, 2600)
ax.set_ylim(0.85, 1.15)
ax.set_xlabel('ℓ', fontsize=12)
ax.set_ylabel('Model / Data', fontsize=12)
ax.set_title('Ratio to Planck Data', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cl_qnm_lcdm_planck.png', dpi=150)
print(f"\nSaved: cl_qnm_lcdm_planck.png")
plt.close()
