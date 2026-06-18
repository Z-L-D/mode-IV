#!/usr/bin/env python3
"""
Fix the ω_R discontinuity and implement proper mode coupling.

The Motl-Neitzke asymptotic formula gives:
  ω_R → ln(3)/(8πM) = 0.0437 (M=1 units) = ln(3)/(4π) in 2M=1 units
  ω_I → (2n+1)/(8M) = (2n+1)/4 in 2M=1 units

But ω_R approaches the limit GRADUALLY. Nollert (1993) showed:
  ω_R(n) ≈ ln(3)/(4π) + C/(n+1)^(1/2) for large n

We need to smoothly interpolate from Leaver (n<8) to asymptotic (n→∞).
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import glob, os

# Leaver values (l=2, s=2)
leaver = {
    0: (0.74734, 0.08896),
    1: (0.69342, 0.27392),
    2: (0.60211, 0.47828),
    3: (0.50302, 0.70514),
    4: (0.41583, 0.94685),
    5: (0.34615, 1.19567),
    6: (0.29131, 1.44796),
    7: (0.24834, 1.70200),
}

asymp_wr = np.log(3) / (4*np.pi)  # 0.08742

# ============================================================
# Smooth interpolation for ω_R
# ============================================================

# The Leaver values show ω_R decreasing roughly as 1/√n
# Fit: ω_R(n) = asymp + A/(n+1)^p
# Using n=7 value to calibrate:
# 0.24834 = 0.08742 + A/8^p
# So A = (0.24834 - 0.08742) * 8^p

# Try different power laws
print("Fitting ω_R approach to asymptotic limit:")
print(f"{'p':>5s}  {'A':>8s}  {'ω_R(8)':>8s}  {'ω_R(20)':>8s}  {'ω_R(50)':>8s}  {'ω_R(100)':>8s}")
print("-" * 55)

for p in [0.5, 0.75, 1.0, 1.5]:
    A = (0.24834 - asymp_wr) * 8**p
    for n_test in [8, 20, 50, 100]:
        pass
    wr_8 = asymp_wr + A/(9)**p
    wr_20 = asymp_wr + A/(21)**p
    wr_50 = asymp_wr + A/(51)**p
    wr_100 = asymp_wr + A/(101)**p
    print(f"{p:5.2f}  {A:8.4f}  {wr_8:8.5f}  {wr_20:8.5f}  {wr_50:8.5f}  {wr_100:8.5f}")

# Use p=0.5 (Nollert's result) as default
p_fit = 0.5
A_fit = (0.24834 - asymp_wr) * 8**p_fit

def omega_R_smooth(n):
    """Smoothly interpolated ω_R for all n."""
    if n < len(leaver):
        return leaver[n][0]
    return asymp_wr + A_fit / (n+1)**p_fit

def omega_I_smooth(n):
    """ω_I - Leaver for n<8, asymptotic for n≥8."""
    if n < len(leaver):
        return leaver[n][1]
    return (2*n+1) / 4.0

# Compare old vs new
print("\nMode frequencies: Current (discontinuous) vs Smooth:")
print(f"{'n':>3s}  {'ω_R old':>10s}  {'ω_R smooth':>10s}  {'ω_I':>10s}  {'Q_old':>8s}  {'Q_smooth':>8s}  {'Δω_R(n,n+1)':>12s}")
print("-" * 75)

for n in range(20):
    wr_old = leaver[n][0] if n < 8 else asymp_wr
    wr_new = omega_R_smooth(n)
    wi = omega_I_smooth(n)
    Q_old = wr_old/wi if wi > 0 else 0
    Q_new = wr_new/wi if wi > 0 else 0
    
    if n > 0:
        wr_new_prev = omega_R_smooth(n-1)
        dwr = wr_new_prev - wr_new
    else:
        dwr = 0
    
    print(f"{n:3d}  {wr_old:10.5f}  {wr_new:10.5f}  {wi:10.5f}  {Q_old:8.3f}  {Q_new:8.3f}  {dwr:12.5f}")

# ============================================================
# P(k) with smooth ω_R and coupling
# ============================================================

k0 = np.pi / 4450.6
N_modes = 300
gamma = 1.04
A_qnm = 2.85e-8
A_s, n_s = 2.1e-9, 0.9649
D_A = 14000

def pk_smooth_coupled(k, eta=0.0, use_smooth_wr=True):
    """P(k) with smooth ω_R transition and optional coupling."""
    w_sum = sum(1.0/(n+1)**gamma for n in range(N_modes))
    
    # Compute per-mode quantities
    kn = np.array([k0*(n+1) for n in range(N_modes)])
    wr = np.array([omega_R_smooth(n) if use_smooth_wr else 
                   (leaver[n][0] if n < 8 else asymp_wr) 
                   for n in range(N_modes)])
    wi = np.array([omega_I_smooth(n) for n in range(N_modes)])
    sigma = wi * k0 / (2*np.pi)
    w = np.array([(1.0/(n+1)**gamma)/w_sum for n in range(N_modes)])
    
    # Lorentzians
    dk = k - kn
    L = sigma**2 / (dk**2 + sigma**2)
    asym = (k + kn)**2 / kn**2
    
    # Standard sum
    mode_sum = np.sum(w * L * asym)
    
    # Cross-terms with smooth ω_R
    if eta != 0:
        cross = 0.0
        for n in range(min(N_modes-1, 50)):  # only first 50 matter
            dwr = wr[n] - wr[n+1]
            if abs(dwr) > 1e-10:
                phase = dwr * (k/k0) * 2 * np.pi
                amp = np.sqrt(w[n]*w[n+1]) * np.sqrt(L[n]*L[n+1]*asym[n]*asym[n+1])
                cross += amp * np.cos(phase)
        mode_sum += eta * cross
    
    return max(A_qnm * mode_sum, 1e-30)

# Compare: current vs smooth vs smooth+coupling
k_vals = np.logspace(np.log10(k0*10), np.log10(k0*280), 2000)
ell_vals = k_vals * D_A
pk_lcdm = np.array([A_s*(k/0.05)**(n_s-1) for k in k_vals])

configs = [
    ("Current (discontinuous ω_R)", False, 0.0),
    ("Smooth ω_R, no coupling", True, 0.0),
    ("Smooth ω_R, η=0.5", True, 0.5),
    ("Smooth ω_R, η=1.0", True, 1.0),
    ("Smooth ω_R, η=2.0", True, 2.0),
]

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

print("\n\nP_QNM/P_LCDM comparison:")
print(f"{'Config':>35s}  {'ℓ=220':>7s}  {'ℓ=810':>7s}  {'ℓ=1500':>7s}  {'ℓ=1750':>7s}  {'ℓ=2000':>7s}  {'ℓ=2500':>7s}")
print("-" * 85)

for name, smooth, eta in configs:
    pk = np.array([pk_smooth_coupled(k, eta, smooth) for k in k_vals])
    ratio = pk / pk_lcdm
    
    # Print key ratios
    vals = []
    for ell_t in [220, 810, 1500, 1750, 2000, 2500]:
        idx = np.argmin(np.abs(ell_vals - ell_t))
        vals.append(ratio[idx])
    vals_str = "  ".join(f"{v:7.4f}" for v in vals)
    print(f"{name:>35s}  {vals_str}")
    
    axes[0].semilogx(k_vals, ratio, '-', lw=1.5, label=name, alpha=0.7)
    
    # Detail view
    detail = (ell_vals > 1200) & (ell_vals < 2600)
    axes[1].plot(ell_vals[detail], ratio[detail], '-', lw=1.5, label=name, alpha=0.7)

axes[0].axhline(1.0, color='k', ls='--', lw=0.5)
axes[0].set_ylabel('P_QNM / P_LCDM')
axes[0].set_title('Effect of Smooth ω_R + Coupling')
axes[0].legend(fontsize=8)
axes[0].set_ylim(0.90, 1.10)
axes[0].grid(True, alpha=0.3)

axes[1].axhline(1.0, color='k', ls='--', lw=0.5)
axes[1].set_xlabel('ℓ')
axes[1].set_ylabel('P_QNM / P_LCDM')
axes[1].set_title('Detail: ℓ 1200-2600 (Coupling Region)')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('coupling_fix.png', dpi=150)
print(f"\nSaved: coupling_fix.png")
