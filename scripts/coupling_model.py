#!/usr/bin/env python3
"""
QNM Mode Coupling in the Primordial Spectrum
=============================================

The current P(k) is a sum of independent Lorentzians (|C_n|²).
The FULL power spectrum includes cross-terms from the complex
Green's function: Σ C_n C_m* interference terms.

These produce oscillatory modulation at beat frequencies
Δω = ω_R,n - ω_R,m between adjacent modes.

The cross-terms are STRONGEST where:
1. Adjacent Lorentzians overlap (high n, dense packing)
2. The modes transition from oscillation to damping dominated
3. The relative phase between modes is coherent

This is the "coupling" feature predicted at ℓ~1700-1900.
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import glob, os

# ============================================================
# QNM parameters
# ============================================================

# Full Leaver values: (omega_R, omega_I) in 2M=1 units
leaver_full = [
    (0.74734, 0.08896),  # n=0
    (0.69342, 0.27392),  # n=1
    (0.60211, 0.47828),  # n=2
    (0.50302, 0.70514),  # n=3
    (0.41583, 0.94685),  # n=4
    (0.34615, 1.19567),  # n=5
    (0.29131, 1.44796),  # n=6
    (0.24834, 1.70200),  # n=7
]

def qnm_omega(n):
    """Return (omega_R, omega_I) for mode n."""
    if n < len(leaver_full):
        return leaver_full[n]
    else:
        # Asymptotic: omega_R → ln(3)/(8πM), omega_I → (2n+1)/(8M)
        # In 2M=1 units: omega_R → ln(3)/(4π), omega_I → (2n+1)/4
        return (np.log(3)/(4*np.pi), (2*n+1)/4.0)

k0 = np.pi / 4450.6
beta = 1.0
N_modes = 300
gamma = 1.04
A_qnm = 2.85e-8
A_s, n_s = 2.1e-9, 0.9649
D_A = 14000
T_CMB = 2.7255e6

# ============================================================
# Analyze mode structure at the transition
# ============================================================

print("QNM Mode Structure at Critical Damping Transition")
print("=" * 60)
print(f"{'n':>3s}  {'ω_R':>8s}  {'ω_I':>8s}  {'Q=ω_R/ω_I':>10s}  {'Δω_R':>8s}  {'k_n':>10s}  {'ℓ_n':>6s}")
print("-" * 65)

for n in range(15):
    wr, wi = qnm_omega(n)
    Q = wr / wi if wi > 0 else float('inf')
    kn = k0 * (n+1) * beta
    ell_n = kn * D_A
    if n > 0:
        wr_prev, _ = qnm_omega(n-1)
        dwr = wr_prev - wr  # frequency splitting
    else:
        dwr = 0
    print(f"{n:3d}  {wr:8.5f}  {wi:8.5f}  {Q:10.3f}  {dwr:8.5f}  {kn:10.6f}  {ell_n:6.0f}")

# Critical damping: Q = ω_R/ω_I crosses 1
print("\nCritical damping (Q=1) between n=2 (Q=1.26) and n=3 (Q=0.71)")
print("This is where oscillation → damping transition occurs")

# ============================================================
# P(k) with mode coupling
# ============================================================

def pk_with_coupling(k, coupling_strength=0.0):
    """
    P(k) with inter-mode interference terms.
    
    Standard: P(k) = A × Σ w_n × L_n(k) × asym_n
    
    With coupling: P(k) = A × [Σ w_n × L_n + η × Σ w_n w_{n+1} × X_{n,n+1}]
    
    where X_{n,n+1} is the cross-term between adjacent modes:
    X = 2 × cos(φ_n - φ_{n+1}) × √(L_n × L_{n+1})
    
    The phase difference is: Δφ = (ω_R,n - ω_R,n+1) × k / k0
    This produces oscillatory modulation at the beat frequency.
    """
    w_sum = sum(1.0/(n+1)**gamma for n in range(N_modes))
    
    # First pass: compute individual Lorentzians
    L = np.zeros(N_modes)
    w = np.zeros(N_modes)
    kn_arr = np.zeros(N_modes)
    wr_arr = np.zeros(N_modes)
    
    for n in range(N_modes):
        kn = k0 * (n+1) * beta
        kn_arr[n] = kn
        wr_arr[n] = qnm_omega(n)[0]
        _, wi = qnm_omega(n)
        sigma_n = wi * k0 / (2*np.pi)
        w[n] = (1.0/(n+1)**gamma) / w_sum
        dk = k - kn
        L[n] = sigma_n**2 / (dk**2 + sigma_n**2)
        asym = (k + kn)**2 / kn**2
        L[n] *= asym
    
    # Standard mode sum
    mode_sum = np.sum(w * L)
    
    # Cross-terms between adjacent modes
    if coupling_strength != 0:
        cross_sum = 0.0
        for n in range(N_modes - 1):
            # Phase difference from real frequency splitting
            delta_omega_R = wr_arr[n] - wr_arr[n+1]
            # The beat phase depends on k relative to the mode spacing
            phase = delta_omega_R * (k / k0) * 2 * np.pi
            
            # Cross-term amplitude: geometric mean of adjacent Lorentzians
            cross_amp = np.sqrt(L[n] * L[n+1]) * np.sqrt(w[n] * w[n+1])
            
            # Interference: constructive or destructive depending on phase
            cross_sum += cross_amp * np.cos(phase)
        
        mode_sum += coupling_strength * cross_sum
    
    pk = A_qnm * mode_sum
    return max(pk, 1e-30)

# ============================================================
# Compute P(k) with different coupling strengths
# ============================================================

k_vals = np.logspace(np.log10(k0*5), np.log10(k0*280), 2000)
ell_vals = k_vals * D_A

pk_base = np.array([pk_with_coupling(k, 0.0) for k in k_vals])
pk_lcdm = np.array([A_s * (k/0.05)**(n_s-1) for k in k_vals])

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Test different coupling strengths
coupling_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
colors = plt.cm.viridis(np.linspace(0, 1, len(coupling_values)))

for eta, color in zip(coupling_values, colors):
    pk_coupled = np.array([pk_with_coupling(k, eta) for k in k_vals])
    ratio = pk_coupled / pk_lcdm
    
    axes[0].semilogx(k_vals, ratio, '-', color=color, lw=1.5, 
                      alpha=0.7, label=f'η={eta}')
    
    # Focus on the coupling region (ℓ 1500-2200)
    region = (ell_vals > 1400) & (ell_vals < 2300)
    axes[1].plot(ell_vals[region], ratio[region], '-', color=color, 
                 lw=1.5, alpha=0.7, label=f'η={eta}')

axes[0].axhline(1.0, color='k', ls='--', lw=0.5)
axes[0].set_ylabel('P_QNM / P_LCDM')
axes[0].set_title('Primordial Spectrum Ratio: Effect of Mode Coupling')
axes[0].legend(fontsize=9)
axes[0].set_ylim(0.85, 1.15)
axes[0].grid(True, alpha=0.3)
for ell_mark in [1000, 1500, 2000]:
    axes[0].axvline(ell_mark/D_A, color='gray', ls=':', alpha=0.3)

axes[1].axhline(1.0, color='k', ls='--', lw=0.5)
axes[1].set_xlabel('ℓ')
axes[1].set_ylabel('P_QNM / P_LCDM')
axes[1].set_title('Coupling Region Detail (ℓ 1400-2300)')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

# Bottom: compare to Planck residuals
# Load Planck and QNM C_ell
planck = np.loadtxt("planck_tt_binned.txt")
p_ell = planck[:,0]
p_Dl = planck[:,1]
p_sig = (planck[:,2]+planck[:,3])/2
p_bf = planck[:,4]

cl_files = [f for f in glob.glob("class_qnm/output/*_cl.dat") if 'lensed' not in f]
latest = max(cl_files, key=os.path.getmtime)
cl = np.loadtxt(latest)
cl_ell_arr = cl[:,0]
cl_tt_arr = cl[:,1] * T_CMB**2

mask = p_ell >= 30
ell_p = p_ell[mask]
Dl_p = p_Dl[mask]
sig_p = p_sig[mask]
bf_p = p_bf[mask]

f_qnm = interp1d(cl_ell_arr, cl_tt_arr, kind='cubic')
qnm_p = f_qnm(ell_p)
resid_qnm = (Dl_p - qnm_p) / sig_p
resid_lcdm = (Dl_p - bf_p) / sig_p

# What the coupling predicts: oscillatory modulation
# The beat frequency Δω_R between n=2 and n=3 (at critical damping)
# maps to an ℓ-space oscillation period
delta_wr_crit = leaver_full[2][0] - leaver_full[3][0]  # 0.60211 - 0.50302 = 0.0991
ell_beat = 2 * np.pi * D_A * k0 / delta_wr_crit
print(f"\nCritical damping beat frequency:")
print(f"  Δω_R(n=2,3) = {delta_wr_crit:.5f}")
print(f"  Beat period in ℓ-space: Δℓ ≈ {ell_beat:.0f}")

# Also check higher overtone beats
for n in range(8):
    wr_n, _ = qnm_omega(n)
    wr_n1, _ = qnm_omega(n+1)
    dwr = wr_n - wr_n1
    if dwr > 0:
        ell_b = 2 * np.pi * D_A * k0 / dwr
        print(f"  Beat(n={n},{n+1}): Δω_R={dwr:.5f}, Δℓ ≈ {ell_b:.0f}")

ax = axes[2]
hi = ell_p > 800
ax.plot(ell_p[hi], resid_qnm[hi], 'b.-', ms=4, lw=0.5, alpha=0.5, label='QNM residual')
ax.plot(ell_p[hi], resid_lcdm[hi], 'r.-', ms=4, lw=0.5, alpha=0.5, label='ΛCDM residual')
# Smooth
w = 3
if np.sum(hi) > w:
    ax.plot(ell_p[hi], np.convolve(resid_qnm[hi], np.ones(w)/w, 'same'), 
            'b-', lw=2.5, alpha=0.8)
    ax.plot(ell_p[hi], np.convolve(resid_lcdm[hi], np.ones(w)/w, 'same'),
            'r-', lw=2.5, alpha=0.8)
ax.axhline(0, color='k', lw=0.5)

# Mark predicted beat frequencies
for n in range(4):
    wr_n, _ = qnm_omega(n)
    wr_n1, _ = qnm_omega(n+1)
    dwr = wr_n - wr_n1
    if dwr > 0:
        ell_b = 2 * np.pi * D_A * k0 / dwr
        if 800 < ell_b < 2500:
            ax.axvline(ell_b, color='green', ls='--', alpha=0.5, lw=1.5)
            ax.text(ell_b, 8, f'Beat({n},{n+1})\nΔℓ={ell_b:.0f}', 
                    fontsize=7, ha='center', color='green')

ax.set_xlabel('ℓ')
ax.set_ylabel('Residual (σ)')
ax.set_title('High-ℓ Residuals with Predicted Beat Frequencies')
ax.legend(fontsize=10)
ax.set_xlim(800, 2500)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('coupling_analysis.png', dpi=150)
print(f"\nSaved: coupling_analysis.png")

# ============================================================
# Key test: implement coupling in CLASS and check chi2
# ============================================================

# Find optimal coupling strength by comparing P(k) ratio to residuals
print("\nOptimal coupling strength scan:")
print(f"{'η':>6s}  {'ratio@1500':>10s}  {'ratio@1750':>10s}  {'ratio@2000':>10s}  {'ratio@2250':>10s}  {'ratio@2500':>10s}")

for eta in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    ratios = []
    for ell_t in [1500, 1750, 2000, 2250, 2500]:
        k_t = ell_t / D_A
        pk_c = pk_with_coupling(k_t, eta)
        pk_l = A_s * (k_t/0.05)**(n_s-1)
        ratios.append(pk_c / pk_l)
    vals = "  ".join(f"{r:10.4f}" for r in ratios)
    print(f"{eta:6.1f}  {vals}")
