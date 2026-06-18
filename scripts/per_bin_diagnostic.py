#!/usr/bin/env python3
"""
per_bin_diagnostic.py — Per-bin QNM vs ΛCDM comparison for TT, EE, TE
Run from ~/qnm-boltzmann/

Shows where QNM beats ΛCDM and where it doesn't, bin by bin.
"""
import numpy as np
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

T_CMB = 2.7255e6  # μK

# Load QNM lensed output (current best)
qnm = np.loadtxt('class_qnm/output/qnm_test_100_cl_lensed.dat')
qnm_ell = qnm[:, 0]
qnm_tt = qnm[:, 1] * T_CMB**2
qnm_ee = qnm[:, 2] * T_CMB**2
qnm_te = qnm[:, 3] * T_CMB**2

# Load ΛCDM best-fit (vanilla CLASS with default params)
# Generate if needed: run vanilla CLASS with Planck best-fit
# For now, use the Planck BestFit column from the binned data
planck_tt = np.loadtxt('planck_tt_binned.txt')
planck_ee = np.loadtxt('planck_ee_binned.txt')
planck_te = np.loadtxt('planck_te_binned.txt')

spectra = [
    ('TT', planck_tt, qnm_tt, 1),
    ('EE', planck_ee, qnm_ee, 2),
    ('TE', planck_te, qnm_te, 3),
]

fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('Per-Bin Diagnostic: QNM (f=0.28, δ=−0.60, A_L=1.33) vs ΛCDM vs Planck',
             fontsize=14, y=0.98)

for row, (name, planck_data, qnm_cl, col_idx) in enumerate(spectra):

    mask = planck_data[:, 0] >= 30
    p = planck_data[mask]
    ell = p[:, 0]
    dl_data = p[:, 1]
    err_minus = np.abs(p[:, 2])
    err_plus = np.abs(p[:, 3])
    err = 0.5 * (err_minus + err_plus)

    # ΛCDM best-fit from the Planck file (column 4 if it exists)
    if p.shape[1] > 4:
        dl_lcdm = p[:, 4]
    else:
        dl_lcdm = p[:, -1]  # last column is usually best-fit

    # QNM interpolated to Planck ℓ values
    qnm_interp = interp1d(qnm_ell, qnm_cl, kind='cubic',
                          bounds_error=False, fill_value=0)
    dl_qnm = qnm_interp(ell)

    # Residuals in units of σ
    resid_qnm = (dl_data - dl_qnm) / err
    resid_lcdm = (dl_data - dl_lcdm) / err

    # Chi2 per bin
    chi2_qnm = resid_qnm**2
    chi2_lcdm = resid_lcdm**2

    # Who wins each bin
    qnm_wins = chi2_qnm < chi2_lcdm
    n_qnm_wins = np.sum(qnm_wins)
    n_lcdm_wins = np.sum(~qnm_wins)

    # Print per-bin table
    print(f"\n{'='*85}")
    print(f"  {name}: QNM wins {n_qnm_wins}/{len(ell)} bins, "
          f"ΛCDM wins {n_lcdm_wins}/{len(ell)} bins")
    print(f"{'='*85}")
    print(f"  {'ℓ':>7} {'Data':>10} {'QNM':>10} {'ΛCDM':>10} "
          f"{'QNM_σ':>7} {'ΛCDM_σ':>7} {'χ²_QNM':>7} {'χ²_ΛCDM':>7} {'Winner':>8}")
    print(f"  {'-'*78}")

    for i in range(len(ell)):
        winner = "QNM" if chi2_qnm[i] < chi2_lcdm[i] else "ΛCDM"
        advantage = abs(chi2_lcdm[i] - chi2_qnm[i])
        marker = f" ({'+'if winner=='QNM' else '-'}{advantage:.1f})"
        print(f"  {ell[i]:7.0f} {dl_data[i]:10.1f} {dl_qnm[i]:10.1f} {dl_lcdm[i]:10.1f} "
              f"{resid_qnm[i]:+7.2f} {resid_lcdm[i]:+7.2f} "
              f"{chi2_qnm[i]:7.2f} {chi2_lcdm[i]:7.2f} {winner:>5}{marker}")

    # Summary stats
    print(f"\n  Summary:")
    print(f"    QNM  total χ²: {np.sum(chi2_qnm):.1f}  (χ²/dof = {np.sum(chi2_qnm)/len(ell):.3f})")
    print(f"    ΛCDM total χ²: {np.sum(chi2_lcdm):.1f}  (χ²/dof = {np.sum(chi2_lcdm)/len(ell):.3f})")
    print(f"    QNM wins: {n_qnm_wins} bins ({100*n_qnm_wins/len(ell):.0f}%)")
    print(f"    Bins where QNM beats ΛCDM by >1σ²: "
          f"{np.sum((chi2_lcdm - chi2_qnm) > 1)}")
    print(f"    Bins where ΛCDM beats QNM by >1σ²: "
          f"{np.sum((chi2_qnm - chi2_lcdm) > 1)}")

    # ---- PLOTS ----

    # Column 1: Spectra comparison
    ax = axes[row, 0]
    ax.errorbar(ell, dl_data, yerr=[err_minus, err_plus],
                fmt='.', color='gray', alpha=0.5, ms=4, label='Planck')
    ax.plot(ell, dl_lcdm, 'b-', lw=1.5, alpha=0.7, label='ΛCDM BF')
    ax.plot(ell, dl_qnm, 'r-', lw=1.5, alpha=0.7, label='QNM')
    ax.set_ylabel(f'{name} D_ℓ [μK²]')
    ax.set_title(f'{name} Spectrum')
    ax.legend(fontsize=7)
    if row == 2:
        ax.set_xlabel('ℓ')

    # Column 2: Residuals in σ
    ax = axes[row, 1]
    ax.bar(ell - 3, resid_qnm, width=6, alpha=0.6, color='red', label='QNM')
    ax.bar(ell + 3, resid_lcdm, width=6, alpha=0.6, color='blue', label='ΛCDM')
    ax.axhline(0, color='k', lw=0.5)
    ax.axhline(2, color='k', ls=':', alpha=0.3)
    ax.axhline(-2, color='k', ls=':', alpha=0.3)
    ax.set_ylabel(f'{name} Residual (σ)')
    ax.set_title(f'{name} Residuals')
    ax.legend(fontsize=7)
    if row == 2:
        ax.set_xlabel('ℓ')

    # Column 3: Chi2 per bin comparison
    ax = axes[row, 2]
    delta_chi2 = chi2_lcdm - chi2_qnm  # positive = QNM wins
    colors = ['green' if d > 0 else 'red' for d in delta_chi2]
    ax.bar(ell, delta_chi2, width=12, color=colors, alpha=0.7)
    ax.axhline(0, color='k', lw=1)
    ax.set_ylabel(f'Δχ² (ΛCDM − QNM)')
    ax.set_title(f'{name}: Green = QNM wins, Red = ΛCDM wins')
    if row == 2:
        ax.set_xlabel('ℓ')

    # Annotate with win count
    ax.text(0.02, 0.95, f'QNM wins {n_qnm_wins}/{len(ell)}',
            transform=ax.transAxes, fontsize=9, va='top',
            color='green' if n_qnm_wins > n_lcdm_wins else 'red')

plt.tight_layout()
plt.savefig('per_bin_diagnostic.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved: per_bin_diagnostic.png")

# ================================================================
# OVERALL SUMMARY
# ================================================================
print(f"\n{'='*85}")
print("OVERALL SUMMARY")
print(f"{'='*85}")
for name, planck_data, qnm_cl, col_idx in spectra:
    mask = planck_data[:, 0] >= 30
    p = planck_data[mask]
    err = 0.5 * (np.abs(p[:, 2]) + np.abs(p[:, 3]))
    qnm_interp = interp1d(qnm_ell, qnm_cl, kind='cubic',
                          bounds_error=False, fill_value=0)
    chi2_q = np.sum(((p[:, 1] - qnm_interp(p[:, 0])) / err)**2)
    dl_lcdm = p[:, 4] if p.shape[1] > 4 else p[:, -1]
    chi2_l = np.sum(((p[:, 1] - dl_lcdm) / err)**2)
    n = len(p)
    wins = np.sum(((p[:, 1] - qnm_interp(p[:, 0])) / err)**2 <
                  ((p[:, 1] - dl_lcdm) / err)**2)
    print(f"  {name}: QNM χ²/dof={chi2_q/n:.3f}, ΛCDM χ²/dof={chi2_l/n:.3f}, "
          f"ratio={chi2_q/chi2_l:.2f}×, QNM wins {wins}/{n} bins ({100*wins/n:.0f}%)")
