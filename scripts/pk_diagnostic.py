#!/usr/bin/env python3
"""Compare the primordial P(k) shape between QNM and LCDM."""

import numpy as np
import matplotlib.pyplot as plt

# QNM parameters (from primordial.c)
omega_I_leaver = [0.08896, 0.27392, 0.47828, 0.70514, 0.95, 1.20, 1.45, 1.70310]
k0 = np.pi / 4450.6  # from qnm_rs = 4450.6 Mpc
beta = 1.0
N_modes = 300
gamma = 1.04
alpha = 0.0
A_qnm = 2.85e-8

# LCDM parameters
A_s = 2.1e-9
n_s = 0.9649
k_pivot = 0.05

def qnm_pk(k):
    """Compute QNM P(k)."""
    w_sum = sum(1.0/(n+1)**gamma for n in range(N_modes))
    mode_sum = 0.0
    for n in range(N_modes):
        kn = k0 * (n+1) * beta
        if n < 8:
            omega_I_n = omega_I_leaver[n]
        else:
            omega_I_n = (2*n + 1) / 8.0
        sigma_n = omega_I_n * k0 / (2*np.pi)
        w_n = (1.0/(n+1)**gamma) / w_sum
        dk = k - kn
        lorentz = sigma_n**2 / (dk**2 + sigma_n**2)
        asym = (k + kn)**2 / kn**2
        mode_sum += w_n * lorentz * asym
    pk = A_qnm * (k/0.05)**alpha * mode_sum
    return max(pk, 1e-30)

def lcdm_pk(k):
    """Standard LCDM power law."""
    return A_s * (k/k_pivot)**(n_s - 1)

# Compute over k range
k_vals = np.logspace(np.log10(k0*0.5), np.log10(k0*300), 2000)

pk_qnm = np.array([qnm_pk(k) for k in k_vals])
pk_lcdm = np.array([lcdm_pk(k) for k in k_vals])

# Find which k corresponds to which ell (rough: ell ~ k * D_A, D_A ~ 14000 Mpc)
D_A = 14000  # approximate angular diameter distance to LSS
ell_approx = k_vals * D_A

print(f"k0 = {k0:.6e} Mpc^-1")
print(f"k0*300 = {k0*300:.6e} Mpc^-1")
print(f"Mode spacing: dk = {k0*beta:.6e} Mpc^-1")
print(f"sigma_0 = {omega_I_leaver[0]*k0/(2*np.pi):.6e} Mpc^-1")
print(f"sigma_100 = {(201/8)*k0/(2*np.pi):.6e} Mpc^-1")
print(f"Q_0 = k_1/sigma_0 = {k0*beta / (omega_I_leaver[0]*k0/(2*np.pi)):.2f}")
print(f"Q_100 = k_100/sigma_100 = {100*k0*beta / ((201/8)*k0/(2*np.pi)):.2f}")

# Key diagnostic: ratio P_QNM / P_LCDM as function of k
ratio = pk_qnm / pk_lcdm

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Top: P(k) comparison
ax = axes[0]
ax.loglog(k_vals, pk_qnm, 'b-', lw=1.5, label='QNM', alpha=0.7)
ax.loglog(k_vals, pk_lcdm, 'r--', lw=1.5, label='LCDM')
ax.set_xlabel('k [Mpc⁻¹]')
ax.set_ylabel('P(k) [Mpc³]')
ax.set_title('Primordial Power Spectrum')
ax.legend()
ax.grid(True, alpha=0.3)

# Add ell scale on top
ax2 = ax.twiny()
ell_ticks = [100, 200, 500, 1000, 2000]
k_ticks = [ell/D_A for ell in ell_ticks]
ax2.set_xlim(ax.get_xlim())
ax2.set_xscale('log')
ax2.set_xticks(k_ticks)
ax2.set_xticklabels([str(l) for l in ell_ticks])
ax2.set_xlabel('Approximate ℓ')

# Middle: Ratio
ax = axes[1]
ax.semilogx(k_vals, ratio, 'b-', lw=1.5)
ax.axhline(1.0, color='k', ls='--', lw=0.5)
ax.set_xlabel('k [Mpc⁻¹]')
ax.set_ylabel('P_QNM / P_LCDM')
ax.set_title('Ratio QNM / LCDM')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 2)

# Mark key ell regions
for ell_mark, label in [(220, 'Peak 1'), (540, 'Peak 2'), (1000, 'ℓ=1000'), (1500, 'ℓ=1500'), (2000, 'ℓ=2000')]:
    k_mark = ell_mark / D_A
    ax.axvline(k_mark, color='gray', ls=':', alpha=0.5)
    ax.text(k_mark, 1.8, label, fontsize=8, ha='center')

# Bottom: Effective spectral index
ax = axes[2]
# d ln P / d ln k for both
dlnk = np.diff(np.log(k_vals))
dlnpk_qnm = np.diff(np.log(pk_qnm)) / dlnk
dlnpk_lcdm = np.diff(np.log(pk_lcdm)) / dlnk
k_mid = np.sqrt(k_vals[:-1] * k_vals[1:])

# Smooth the QNM spectral index
window = 50
if len(dlnpk_qnm) > window:
    smooth_ns = np.convolve(dlnpk_qnm, np.ones(window)/window, mode='same')
    ax.semilogx(k_mid, smooth_ns + 1, 'b-', lw=2, label=f'QNM (smoothed, w={window})')
ax.semilogx(k_mid, dlnpk_lcdm + 1, 'r--', lw=1.5, label=f'LCDM (n_s={n_s})')
ax.axhline(1.0, color='gray', ls=':', alpha=0.5, label='Scale invariant')
ax.set_xlabel('k [Mpc⁻¹]')
ax.set_ylabel('Effective n_s')
ax.set_title('Running Spectral Index')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0.8, 1.1)

plt.tight_layout()
plt.savefig('pk_diagnostic.png', dpi=150)
print("\nSaved: pk_diagnostic.png")

# Quantitative: ratio at key k values
print("\nP_QNM / P_LCDM at key scales:")
for ell_target in [220, 540, 810, 1000, 1500, 2000, 2500]:
    k_target = ell_target / D_A
    idx = np.argmin(np.abs(k_vals - k_target))
    print(f"  ℓ~{ell_target}: k={k_target:.5f}, ratio = {ratio[idx]:.4f}")
