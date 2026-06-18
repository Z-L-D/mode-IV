#!/usr/bin/env python3
"""Redo all three Planck comparisons with LENSED QNM spectra."""
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

T_CMB = 2.7255e6

# Load lensed QNM
lensed = np.loadtxt('class_qnm/output/qnm_test_100_cl_lensed.dat')
cl_ell = lensed[:, 0]
cl_tt = lensed[:, 1] * T_CMB**2
cl_ee = lensed[:, 2] * T_CMB**2
cl_te = lensed[:, 3] * T_CMB**2

print(f"Lensed output: ℓ = {cl_ell[0]:.0f} to {cl_ell[-1]:.0f}, {len(cl_ell)} multipoles")

results = {}

for spec_name, cl_col, planck_file in [
    ("TT", cl_tt, "data/planck/planck_tt_binned.txt"),
    ("EE", cl_ee, "data/planck/planck_ee_binned.txt"),
    ("TE", cl_te, "data/planck/planck_te_binned.txt"),
]:
    planck = np.loadtxt(planck_file)
    p_ell = planck[:, 0]
    p_Dl = planck[:, 1]
    p_sig = (planck[:, 2] + planck[:, 3]) / 2
    p_bf = planck[:, 4] if planck.shape[1] > 4 else None
    
    mask = p_ell >= 30
    ell = p_ell[mask]
    Dl = p_Dl[mask]
    sig = p_sig[mask]
    bf = p_bf[mask] if p_bf is not None else None
    n_bins = len(ell)
    
    # Interpolate QNM lensed to Planck ells
    # Check ell range overlap
    valid_ell = (ell >= cl_ell.min()) & (ell <= cl_ell.max())
    ell_v = ell[valid_ell]
    Dl_v = Dl[valid_ell]
    sig_v = sig[valid_ell]
    bf_v = bf[valid_ell] if bf is not None else None
    
    f_qnm = interp1d(cl_ell, cl_col, kind='cubic')
    qnm_v = f_qnm(ell_v)
    
    chi2_qnm = np.sum(((Dl_v - qnm_v) / sig_v)**2)
    n_v = len(ell_v)
    
    if bf_v is not None:
        chi2_bf = np.sum(((Dl_v - bf_v) / sig_v)**2)
    else:
        chi2_bf = float('nan')
    
    results[spec_name] = (chi2_qnm, n_v, chi2_bf)
    
    print(f"\n{spec_name}: {n_v} bins (ℓ >= 30)")
    print(f"  QNM LENSED: chi2={chi2_qnm:.1f}, chi2/dof={chi2_qnm/(n_v-1):.3f}")
    print(f"  Planck BF:  chi2={chi2_bf:.1f}, chi2/dof={chi2_bf/(n_v-1):.3f}")
    
    # Regional breakdown for TT
    if spec_name == "TT":
        resid = (Dl_v - qnm_v) / sig_v
        regions = [
            ("Low ℓ (30-200)", 30, 200),
            ("Peak 1 (200-350)", 200, 350),
            ("Trough 1 (350-450)", 350, 450),
            ("Peak 2 (450-650)", 450, 650),
            ("Trough 2 (650-750)", 650, 750),
            ("Peak 3 (750-950)", 750, 950),
            ("Mid ℓ (950-1500)", 950, 1500),
            ("High ℓ (1500-2500)", 1500, 2500),
        ]
        print(f"\n  Regional breakdown:")
        print(f"  {'Region':>25s}  {'chi2':>8s}  {'N':>4s}  {'chi2/bin':>8s}  {'mean_r':>8s}  {'dir':>10s}")
        for name, lo, hi in regions:
            rmask = (ell_v >= lo) & (ell_v < hi)
            if np.sum(rmask) == 0: continue
            c2 = np.sum(((Dl_v[rmask] - qnm_v[rmask])/sig_v[rmask])**2)
            mr = np.mean((Dl_v[rmask] - qnm_v[rmask])/sig_v[rmask])
            d = "QNM HIGH" if mr < 0 else "QNM LOW"
            print(f"  {name:>25s}  {c2:8.1f}  {np.sum(rmask):4d}  {c2/np.sum(rmask):8.2f}  {mr:8.2f}  {d:>10s}")

# Summary
print(f"\n{'='*60}")
print(f"SUMMARY: QNM LENSED vs Planck 2018")
print(f"{'='*60}")
print(f"{'Spectrum':>10s}  {'QNM χ²/dof':>12s}  {'ΛCDM χ²/dof':>12s}  {'Ratio':>8s}")
print(f"{'-'*50}")
for spec in ["TT", "EE", "TE"]:
    c2_q, n, c2_b = results[spec]
    dof = n - 1
    print(f"{spec:>10s}  {c2_q/dof:12.3f}  {c2_b/dof:12.3f}  {c2_q/c2_b:8.2f}")

# Comparison: unlensed vs lensed
print(f"\n{'='*60}")
print(f"BEFORE vs AFTER (lensing correction)")
print(f"{'='*60}")
print(f"{'Spectrum':>10s}  {'UNLENSED':>12s}  {'LENSED':>12s}  {'Improvement':>12s}")
print(f"{'-'*50}")
unlensed_values = {"TT": 12.096, "EE": 1.922, "TE": 3.848}
for spec in ["TT", "EE", "TE"]:
    c2_q, n, _ = results[spec]
    lensed_val = c2_q / (n-1)
    unlensed_val = unlensed_values[spec]
    improvement = (1 - lensed_val/unlensed_val) * 100
    print(f"{spec:>10s}  {unlensed_val:12.3f}  {lensed_val:12.3f}  {improvement:+11.1f}%")

# Plot all three
fig, axes = plt.subplots(3, 2, figsize=(16, 16))

for row, (spec_name, cl_col, planck_file) in enumerate([
    ("TT", cl_tt, "data/planck/planck_tt_binned.txt"),
    ("EE", cl_ee, "data/planck/planck_ee_binned.txt"),
    ("TE", cl_te, "data/planck/planck_te_binned.txt"),
]):
    planck = np.loadtxt(planck_file)
    p_ell = planck[:,0]; p_Dl = planck[:,1]
    p_sig = (planck[:,2]+planck[:,3])/2
    p_bf = planck[:,4] if planck.shape[1] > 4 else None
    
    mask = p_ell >= 30
    ell = p_ell[mask]; Dl = p_Dl[mask]; sig = p_sig[mask]
    bf = p_bf[mask] if p_bf is not None else None
    
    valid_ell = (ell >= cl_ell.min()) & (ell <= cl_ell.max())
    ell_v = ell[valid_ell]; Dl_v = Dl[valid_ell]; sig_v = sig[valid_ell]
    bf_v = bf[valid_ell] if bf is not None else None
    
    f_qnm = interp1d(cl_ell, cl_col, kind='cubic')
    qnm_v = f_qnm(ell_v)
    
    c2_q, n, c2_b = results[spec_name]
    
    # Spectrum
    ax = axes[row, 0]
    ax.errorbar(ell_v, Dl_v, yerr=sig_v, fmt='.', ms=3, color='gray', alpha=0.5)
    ax.plot(cl_ell, cl_col, 'b-', lw=1, alpha=0.7, label='QNM (lensed)')
    if bf_v is not None:
        ax.plot(ell_v, bf_v, 'r-', lw=1, alpha=0.5, label='ΛCDM best-fit')
    ax.set_ylabel('D_ℓ [μK²]')
    ax.set_title(f'{spec_name}: QNM lensed vs Planck 2018')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 2500)
    if spec_name == "TE":
        ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    
    # Residual
    ax = axes[row, 1]
    resid_q = (Dl_v - qnm_v) / sig_v
    ax.plot(ell_v, resid_q, 'b.', ms=4, alpha=0.4, 
            label=f'QNM (χ²/dof={c2_q/(n-1):.2f})')
    if bf_v is not None:
        resid_b = (Dl_v - bf_v) / sig_v
        ax.plot(ell_v, resid_b, 'r.', ms=4, alpha=0.4,
                label=f'ΛCDM (χ²/dof={c2_b/(n-1):.2f})')
    ax.axhline(0, color='k', lw=0.5)
    ax.axhline(3, color='gray', ls='--', alpha=0.3)
    ax.axhline(-3, color='gray', ls='--', alpha=0.3)
    ax.set_ylabel(f'{spec_name} residual (σ)')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 2500)

axes[2, 0].set_xlabel('ℓ')
axes[2, 1].set_xlabel('ℓ')

plt.suptitle('QNM LENSED vs Planck 2018: TT, EE, TE (zero additional parameters)', 
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('cl_all_lensed_planck.png', dpi=150)
print(f"\nSaved: cl_all_lensed_planck.png")
