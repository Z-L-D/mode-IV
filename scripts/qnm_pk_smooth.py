#!/usr/bin/env python3
"""QNM P(k) with smooth (averaged) cavity envelope replacing Dirichlet j2."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn

H0_kmsMpc = 67.36
c_kmps = 299792.458
H0 = H0_kmsMpc / c_kmps
rs = 1.0 / H0
k0 = np.pi / rs
beta_geom = 1.0

omega_R_leaver = np.array([0.37367, 0.34671, 0.30105, 0.25150, 0.205, 0.165, 0.130, 0.10932])
omega_I_leaver = np.array([0.08896, 0.27392, 0.47828, 0.70514, 0.95, 1.20, 1.45, 1.70310])
omega_R_asymp = np.log(3) / (8 * np.pi)

def get_omega_I(n):
    if n < 8: return omega_I_leaver[n]
    return (2*n + 1) / 8.0

def get_omega_R(n):
    if n < 8: return omega_R_leaver[n]
    return omega_R_asymp + (omega_R_leaver[7] - omega_R_asymp) * np.exp(-(n - 7) / 3.0)

N_MODES = 300

def compute_spectrum(k_array, A_QNM, envelope_type='smooth', model='B', gamma=1.0):
    """
    envelope_type:
      'dirichlet' = [j2(k*rs)]^2          (hard wall, creates zeros)
      'smooth'    = 1/(2*(k*rs)^2)         (averaged, smooth)
      'tapered'   = j2^2 smoothed by running average
      'none'      = 1.0                     (pure mode sum)
    """
    Pk = np.zeros_like(k_array)
    
    kn_arr = np.zeros(N_MODES)
    sigma_arr = np.zeros(N_MODES)
    w_arr = np.zeros(N_MODES)
    
    for n in range(N_MODES):
        kn_arr[n] = k0 * (n + 1) * beta_geom
        sigma_arr[n] = get_omega_I(n) * k0 / (2.0 * np.pi)
        if model == 'A':
            w_arr[n] = 1.0
        elif model == 'B':
            w_arr[n] = 1.0 / (n + 1)**gamma
    w_arr /= np.sum(w_arr)
    
    for i, k in enumerate(k_array):
        if k <= 0: continue
        x = k * rs
        
        # Envelope selection
        if envelope_type == 'dirichlet':
            env = spherical_jn(2, x)**2
        elif envelope_type == 'smooth':
            if x < 3.0:
                env = spherical_jn(2, x)**2  # use exact at low x
            else:
                env = 1.0 / (2.0 * x * x)   # averaged at high x
        elif envelope_type == 'none':
            env = 1.0
        else:
            env = 1.0
        
        if env < 1e-30: continue
        
        mode_sum = 0.0
        for n in range(N_MODES):
            kn = kn_arr[n]
            sigma_n = sigma_arr[n]
            dk = k - kn
            lorentz = 1.0 / (dk**2 + sigma_n**2)
            asym = (k + kn)**2 / kn**2
            mode_sum += w_arr[n] * lorentz * asym
        
        Pk[i] = A_QNM * env * mode_sum
    return Pk

def P_inflation(k_array, A_s=2.1e-9, n_s=0.9649, k_pivot=0.05):
    return A_s * (k_array / k_pivot)**(n_s - 1.0)

k = np.logspace(-4, -0.3, 3000)
Pk_inf = P_inflation(k)
D_rec = 14000
k_ell200 = 200.0 / D_rec
k_ell2500 = 2500.0 / D_rec

# Try multiple amplitude normalizations for the smooth envelope
# Start with A_QNM adjusted to match inflation at pivot
A_test = 3.3e-9  # ~2.1e-9 / 0.63 from the Model B ratio

print("Computing smooth envelope spectra...")
Pk_smooth_B = compute_spectrum(k, A_test, envelope_type='smooth', model='B', gamma=1.0)
Pk_smooth_B15 = compute_spectrum(k, A_test*0.5, envelope_type='smooth', model='B', gamma=1.5)
Pk_smooth_B07 = compute_spectrum(k, A_test*2.0, envelope_type='smooth', model='B', gamma=0.7)
Pk_dirichlet = compute_spectrum(k, A_test, envelope_type='dirichlet', model='B', gamma=1.0)

fig, axes = plt.subplots(3, 1, figsize=(14, 14), gridspec_kw={'height_ratios': [3, 3, 1]})

# Panel 1: Full range
ax1 = axes[0]
ax1.loglog(k, Pk_smooth_B, 'b-', lw=2, label='Smooth env, gamma=1.0', alpha=0.9)
ax1.loglog(k, Pk_smooth_B15, 'g-', lw=1.5, label='Smooth env, gamma=1.5', alpha=0.7)
ax1.loglog(k, Pk_smooth_B07, 'm-', lw=1.5, label='Smooth env, gamma=0.7', alpha=0.7)
ax1.loglog(k, Pk_dirichlet, 'c-', lw=0.5, label='Dirichlet env (original)', alpha=0.4)
ax1.loglog(k, Pk_inf, 'r--', lw=2, label='Inflation (n_s=0.9649)')
ax1.axvspan(k_ell200, k_ell2500, alpha=0.1, color='green', label='Planck range')
ax1.set_xlim(1e-4, 0.5)
ax1.set_xlabel('k [Mpc^-1]')
ax1.set_ylabel('P(k)')
ax1.set_title('QNM with Smooth Cavity Envelope vs Inflation')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel 2: Planck zoom
ax2 = axes[1]
ax2.loglog(k, Pk_smooth_B, 'b-', lw=2, label='gamma=1.0', alpha=0.9)
ax2.loglog(k, Pk_smooth_B15, 'g-', lw=1.5, label='gamma=1.5', alpha=0.7)
ax2.loglog(k, Pk_smooth_B07, 'm-', lw=1.5, label='gamma=0.7', alpha=0.7)
ax2.loglog(k, Pk_inf, 'r--', lw=2, label='Inflation')
ax2.axvspan(k_ell200, k_ell2500, alpha=0.1, color='green')
ax2.set_xlim(0.008, 0.25)
ax2.set_xlabel('k [Mpc^-1]')
ax2.set_ylabel('P(k)')
ax2.set_title('Planck Range — Smooth Envelope')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Panel 3: Ratio
ax3 = axes[2]
for Pk_test, label, color in [
    (Pk_smooth_B, 'gamma=1.0', 'b'),
    (Pk_smooth_B15, 'gamma=1.5', 'g'),
    (Pk_smooth_B07, 'gamma=0.7', 'm'),
]:
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(Pk_inf > 0, Pk_test / Pk_inf, np.nan)
    ax3.semilogx(k, ratio, color=color, lw=1.5, label=label, alpha=0.8)
ax3.axhline(1.0, color='r', ls='--', alpha=0.5)
ax3.axvspan(k_ell200, k_ell2500, alpha=0.1, color='green')
ax3.set_xlim(0.008, 0.25)
ax3.set_ylim(0, 3)
ax3.set_xlabel('k [Mpc^-1]')
ax3.set_ylabel('P_QNM / P_inflation')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('qnm_pk_smooth.png', dpi=150, bbox_inches='tight')
print("Saved: qnm_pk_smooth.png")

# Effective spectral index
print("\n" + "="*60)
print("EFFECTIVE SPECTRAL INDEX — SMOOTH ENVELOPE")
print("="*60)
print("\n  k (Mpc^-1)    n_eff(g=1.0)   n_eff(g=1.5)   n_eff(g=0.7)   n_eff(inf)")
for kt in [0.01, 0.02, 0.05, 0.1, 0.15]:
    dk = kt * 0.02
    i_lo = np.argmin(np.abs(k - (kt - dk)))
    i_hi = np.argmin(np.abs(k - (kt + dk)))
    results = []
    for Pk_test in [Pk_smooth_B, Pk_smooth_B15, Pk_smooth_B07]:
        if Pk_test[i_lo] > 0 and Pk_test[i_hi] > 0:
            n_eff = np.log(Pk_test[i_hi]/Pk_test[i_lo]) / np.log(k[i_hi]/k[i_lo])
        else:
            n_eff = float('nan')
        results.append(n_eff)
    print(f"  {kt:.3f}         {results[0]:+.4f}        {results[1]:+.4f}        {results[2]:+.4f}        -0.0351")

# Amplitude at pivot
print(f"\nAt k = 0.05 Mpc^-1:")
idx = np.argmin(np.abs(k - 0.05))
print(f"  Inflation:  {Pk_inf[idx]:.3e}")
print(f"  Smooth g=1: {Pk_smooth_B[idx]:.3e} (ratio {Pk_smooth_B[idx]/Pk_inf[idx]:.2f})")
print(f"  Smooth g=1.5: {Pk_smooth_B15[idx]:.3e} (ratio {Pk_smooth_B15[idx]/Pk_inf[idx]:.2f})")
print(f"  Smooth g=0.7: {Pk_smooth_B07[idx]:.3e} (ratio {Pk_smooth_B07[idx]/Pk_inf[idx]:.2f})")
