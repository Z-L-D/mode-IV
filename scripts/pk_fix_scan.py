#!/usr/bin/env python3
"""
Test modifications to the QNM P(k) to close the high-k deficit.
Identifies the minimal change that flattens P_QNM/P_LCDM.
"""
import numpy as np
import matplotlib.pyplot as plt

# Current parameters
omega_I_leaver = [0.08896, 0.27392, 0.47828, 0.70514, 0.95, 1.20, 1.45, 1.70310]
k0 = np.pi / 4450.6
beta = 1.0
N_modes = 300
gamma = 1.04
A_qnm = 2.85e-8
A_s = 2.1e-9
n_s = 0.9649

def qnm_pk(k, gamma_val=gamma, N=N_modes, width_factor=1.0, 
            asymp_coeff=8.0, continuum=0.0, gamma_running=0.0):
    """QNM P(k) with modification parameters."""
    # Compute normalization
    w_sum = 0.0
    for n in range(N):
        g_eff = gamma_val + gamma_running * np.log(1 + n/10.0)
        w_sum += 1.0/(n+1)**g_eff
    
    mode_sum = 0.0
    for n in range(N):
        kn = k0 * (n+1) * beta
        if n < 8:
            omega_I_n = omega_I_leaver[n]
        else:
            omega_I_n = (2.0*n + 1.0) / asymp_coeff
        
        sigma_n = omega_I_n * k0 / (2*np.pi) * width_factor
        g_eff = gamma_val + gamma_running * np.log(1 + n/10.0)
        w_n = (1.0/(n+1)**g_eff) / w_sum
        
        dk = k - kn
        lorentz = sigma_n**2 / (dk**2 + sigma_n**2)
        asym = (k + kn)**2 / kn**2
        mode_sum += w_n * lorentz * asym
    
    mode_sum += continuum
    return A_qnm * mode_sum

def lcdm_pk(k):
    return A_s * (k/0.05)**(n_s - 1)

# k range
k_vals = np.logspace(np.log10(k0*5), np.log10(k0*280), 500)
D_A = 14000

# Reference: current QNM
pk_current = np.array([qnm_pk(k) for k in k_vals])
pk_lcdm = np.array([lcdm_pk(k) for k in k_vals])
ratio_current = pk_current / pk_lcdm

# Test modifications
modifications = {
    "Current (baseline)": {},
    "Width ×1.3": {"width_factor": 1.3},
    "Width ×1.5": {"width_factor": 1.5},
    "Width ×2.0": {"width_factor": 2.0},
    "Asymp coeff 7.0": {"asymp_coeff": 7.0},
    "Asymp coeff 6.0": {"asymp_coeff": 6.0},
    "N_modes=500": {"N": 500},
    "N_modes=1000": {"N": 1000},
    "gamma=1.02": {"gamma_val": 1.02},
    "gamma=1.00": {"gamma_val": 1.00},
    "Continuum 0.001": {"continuum": 0.001},
    "Continuum 0.003": {"continuum": 0.003},
    "Continuum 0.005": {"continuum": 0.005},
    "gamma_running=-0.01": {"gamma_running": -0.01},
    "gamma_running=-0.02": {"gamma_running": -0.02},
}

# Evaluate each
print(f"{'Modification':>25s}  {'ratio@500':>10s}  {'ratio@1000':>10s}  {'ratio@1500':>10s}  {'ratio@2000':>10s}  {'ratio@2500':>10s}  {'max_dev_peaks':>14s}")
print("-" * 100)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

ell_targets = [500, 1000, 1500, 2000, 2500]
k_targets = [ell/D_A for ell in ell_targets]

best_name = None
best_score = 999

for name, params in modifications.items():
    pk_mod = np.array([qnm_pk(k, **params) for k in k_vals])
    ratio = pk_mod / pk_lcdm
    
    # Evaluate at targets
    ratios_at_targets = []
    for kt in k_targets:
        idx = np.argmin(np.abs(k_vals - kt))
        ratios_at_targets.append(ratio[idx])
    
    # Peak region deviation (ℓ=200-900, want this close to 1.0)
    peak_mask = (k_vals > 200/D_A) & (k_vals < 900/D_A)
    max_dev_peaks = np.max(np.abs(ratio[peak_mask] - 1.0))
    
    # Score: mean absolute deviation from 1.0 at high ell
    high_ell_mask = (k_vals > 1000/D_A) & (k_vals < 2500/D_A)
    score = np.mean(np.abs(ratio[high_ell_mask] - 1.0))
    
    if score < best_score and max_dev_peaks < 0.03:
        best_score = score
        best_name = name
    
    vals_str = "  ".join(f"{r:10.4f}" for r in ratios_at_targets)
    print(f"{name:>25s}  {vals_str}  {max_dev_peaks:14.4f}")
    
    if name != "Current (baseline)":
        axes[0].semilogx(k_vals, ratio, '-', lw=1, alpha=0.6, label=name)

# Plot current as thick line
axes[0].semilogx(k_vals, ratio_current, 'k-', lw=2.5, label='Current', alpha=0.8)
axes[0].axhline(1.0, color='gray', ls='--', lw=0.5)
axes[0].set_ylabel('P_QNM / P_LCDM')
axes[0].set_title('Effect of Modifications on Primordial Spectrum Ratio')
axes[0].legend(fontsize=7, ncol=3, loc='lower left')
axes[0].set_ylim(0.85, 1.15)
axes[0].grid(True, alpha=0.3)

# Mark ell regions
for ell_mark in [220, 540, 810, 1500, 2000]:
    axes[0].axvline(ell_mark/D_A, color='gray', ls=':', alpha=0.3)

# Bottom: best candidate vs current
if best_name:
    print(f"\nBest modification (min high-ell deviation, peak deviation < 3%): {best_name}")
    pk_best = np.array([qnm_pk(k, **modifications[best_name]) for k in k_vals])
    ratio_best = pk_best / pk_lcdm
    
    axes[1].semilogx(k_vals, ratio_current, 'r-', lw=2, label='Current')
    axes[1].semilogx(k_vals, ratio_best, 'b-', lw=2, label=f'Best: {best_name}')
    axes[1].axhline(1.0, color='gray', ls='--', lw=0.5)
    axes[1].set_xlabel('k [Mpc⁻¹]')
    axes[1].set_ylabel('P_QNM / P_LCDM')
    axes[1].set_title(f'Best Modification: {best_name}')
    axes[1].legend()
    axes[1].set_ylim(0.9, 1.1)
    axes[1].grid(True, alpha=0.3)
    
    # Print predicted chi2 improvement
    # Current high-ell deficit causes ~600 chi2 (60% of 992)
    # If ratio goes from 0.936 to X at ell=2500, improvement is proportional
    idx_2500 = np.argmin(np.abs(k_vals - 2500/D_A))
    old_deficit = 1 - ratio_current[idx_2500]
    new_deficit = 1 - ratio_best[idx_2500]
    print(f"\nDeficit at ℓ~2500: {old_deficit*100:.1f}% → {new_deficit*100:.1f}%")
    print(f"Estimated chi2 reduction: ~{(1-(new_deficit/old_deficit)**2)*600:.0f} from high-ℓ region")

plt.tight_layout()
plt.savefig('pk_fix_scan.png', dpi=150)
print(f"\nSaved: pk_fix_scan.png")
