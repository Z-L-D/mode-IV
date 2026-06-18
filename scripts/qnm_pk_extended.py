#!/usr/bin/env python3
"""Extended QNM P(k) with ~300 cavity modes, not just 8."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn

H0_kmsMpc = 67.36
c_kmps = 299792.458
H0 = H0_kmsMpc / c_kmps
rs = 1.0 / H0
k0 = np.pi / rs
beta_geom = 1.0
A_QNM = 2.1e-9

# Leaver 1985 exact values for n=0-7
omega_R_leaver = np.array([0.37367, 0.34671, 0.30105, 0.25150, 0.205, 0.165, 0.130, 0.10932])
omega_I_leaver = np.array([0.08896, 0.27392, 0.47828, 0.70514, 0.95, 1.20, 1.45, 1.70310])

# Asymptotic formula (Motl & Neitzke 2003) for n >> 1
# omega_R -> ln(3)/(8*pi) ≈ 0.0439
# omega_I -> (2n+1)/8
omega_R_asymp = np.log(3) / (8 * np.pi)

def get_omega_I(n):
    """QNM imaginary frequency for overtone n."""
    if n < 8:
        return omega_I_leaver[n]
    return (2*n + 1) / 8.0

def get_omega_R(n):
    """QNM real frequency for overtone n."""
    if n < 8:
        return omega_R_leaver[n]
    # Smooth interpolation from n=7 value to asymptotic
    # Exponential approach: omega_R_asymp + (omega_R[7] - omega_R_asymp) * exp(-(n-7)/3)
    return omega_R_asymp + (omega_R_leaver[7] - omega_R_asymp) * np.exp(-(n - 7) / 3.0)

N_MODES = 300  # Extend to cover Planck range and beyond

def P_QNM_extended(k_array, model='A'):
    """
    Extended QNM power spectrum with N_MODES cavity modes.
    
    Models:
      'A' = equal excitation (all modes same weight)
      'B' = power-law decay (w_n ~ 1/(n+1))
      'C' = thermal (w_n ~ exp(-omega_I * t_form))
    """
    Pk = np.zeros_like(k_array)
    
    # Precompute mode parameters
    kn_arr = np.zeros(N_MODES)
    sigma_arr = np.zeros(N_MODES)
    w_arr = np.zeros(N_MODES)
    
    for n in range(N_MODES):
        kn_arr[n] = k0 * (n + 1) * beta_geom
        omega_I_n = get_omega_I(n)
        sigma_arr[n] = omega_I_n * k0 / (2.0 * np.pi)
        
        if model == 'A':
            w_arr[n] = 1.0
        elif model == 'B':
            w_arr[n] = 1.0 / (n + 1)
        elif model == 'C':
            # Formation timescale ~ 1 Gyr
            t_form_Gyr = 1.0
            tau_n_Gyr = 1.0 / (omega_I_n * 2 * H0 * 3.086e19 / 3.156e16)
            w_arr[n] = np.exp(-t_form_Gyr / tau_n_Gyr)
    
    # Normalize weights
    w_arr /= np.sum(w_arr)
    
    for i, k in enumerate(k_array):
        if k <= 0:
            continue
        
        # Geometric envelope
        x = k * rs
        j2 = spherical_jn(2, x)
        envelope = j2**2
        
        if envelope < 1e-30:
            continue
        
        # Sum over ALL modes
        mode_sum = 0.0
        for n in range(N_MODES):
            kn = kn_arr[n]
            sigma_n = sigma_arr[n]
            dk = k - kn
            lorentz = 1.0 / (dk**2 + sigma_n**2)
            asym = (k + kn)**2 / kn**2
            mode_sum += w_arr[n] * lorentz * asym
        
        Pk[i] = A_QNM * envelope * mode_sum
    
    return Pk

def P_inflation(k_array, A_s=2.1e-9, n_s=0.9649, k_pivot=0.05):
    return A_s * (k_array / k_pivot)**(n_s - 1.0)

# ============================================================
# Compute
# ============================================================

k = np.logspace(-4, -0.3, 3000)  # extend to k = 0.5 Mpc^-1

print("Computing P_QNM Model A (equal excitation)...")
Pk_A = P_QNM_extended(k, model='A')
print("Computing P_QNM Model B (power-law decay)...")
Pk_B = P_QNM_extended(k, model='B')
print("Computing P_inflation...")
Pk_inf = P_inflation(k)

D_rec = 14000
k_ell200 = 200.0 / D_rec
k_ell2500 = 2500.0 / D_rec

# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(14, 14), gridspec_kw={'height_ratios': [3, 3, 1]})

# --- Panel 1: Full comparison ---
ax1 = axes[0]
ax1.loglog(k, Pk_A, 'b-', linewidth=1.5, label='QNM Model A (equal excitation, N=300)', alpha=0.8)
ax1.loglog(k, Pk_B, 'c-', linewidth=1.5, label='QNM Model B (1/(n+1) decay, N=300)', alpha=0.8)
ax1.loglog(k, Pk_inf, 'r--', linewidth=2, label='Inflation (n_s=0.9649)')
ax1.axvspan(k_ell200, k_ell2500, alpha=0.1, color='green', label='Planck TT range (l=200-2500)')

# Mark first 8 QNM modes
for n in range(8):
    kn = k0 * (n+1) * beta_geom
    ax1.axvline(kn, color='gray', alpha=0.2, linestyle=':')

ax1.set_xlabel('k [Mpc^-1]')
ax1.set_ylabel('P(k)')
ax1.set_title('Extended QNM Spectrum (300 modes) vs Inflation')
ax1.legend(fontsize=9)
ax1.set_xlim(1e-4, 0.5)
ax1.grid(True, alpha=0.3)

# --- Panel 2: Zoom on Planck range ---
ax2 = axes[1]
ax2.loglog(k, Pk_A, 'b-', linewidth=1.5, label='QNM Model A', alpha=0.8)
ax2.loglog(k, Pk_B, 'c-', linewidth=1.5, label='QNM Model B', alpha=0.8)
ax2.loglog(k, Pk_inf, 'r--', linewidth=2, label='Inflation')
ax2.axvspan(k_ell200, k_ell2500, alpha=0.1, color='green')
ax2.set_xlabel('k [Mpc^-1]')
ax2.set_ylabel('P(k)')
ax2.set_title('Zoom: Planck Range')
ax2.legend(fontsize=9)
ax2.set_xlim(0.005, 0.3)
ax2.grid(True, alpha=0.3)

# --- Panel 3: Ratio to inflation ---
ax3 = axes[2]
with np.errstate(divide='ignore', invalid='ignore'):
    ratio_A = np.where(Pk_inf > 0, Pk_A / Pk_inf, np.nan)
    ratio_B = np.where(Pk_inf > 0, Pk_B / Pk_inf, np.nan)
ax3.semilogx(k, ratio_A, 'b-', linewidth=1.5, label='Model A / inflation', alpha=0.8)
ax3.semilogx(k, ratio_B, 'c-', linewidth=1.5, label='Model B / inflation', alpha=0.8)
ax3.axhline(1.0, color='r', linestyle='--', alpha=0.5)
ax3.set_xlabel('k [Mpc^-1]')
ax3.set_ylabel('P_QNM / P_inflation')
ax3.set_xlim(0.005, 0.3)
ax3.set_ylim(0, 5)
ax3.grid(True, alpha=0.3)
ax3.axvspan(k_ell200, k_ell2500, alpha=0.1, color='green')
ax3.legend(fontsize=9)

plt.tight_layout()
plt.savefig('qnm_pk_extended.png', dpi=150, bbox_inches='tight')
print("\nSaved: qnm_pk_extended.png")

# ============================================================
# Diagnostics
# ============================================================

print("\n" + "="*60)
print("EXTENDED QNM DIAGNOSTICS")
print("="*60)
print(f"\nTotal modes: {N_MODES}")
print(f"Highest mode: k_{N_MODES-1} = {k0 * N_MODES * beta_geom:.4e} Mpc^-1")
print(f"  -> ell ~ {k0 * N_MODES * beta_geom * D_rec:.0f}")
print(f"\nPlanck range: k = [{k_ell200:.4e}, {k_ell2500:.4e}]")
print(f"  -> needs modes n ~ {int(k_ell200/k0)}-{int(k_ell2500/k0)}")

# Check amplitude match in Planck range
k_test = 0.05  # pivot scale
idx = np.argmin(np.abs(k - k_test))
print(f"\nAt pivot k = {k_test} Mpc^-1:")
print(f"  P_inflation = {Pk_inf[idx]:.3e}")
print(f"  P_QNM (A)   = {Pk_A[idx]:.3e}  (ratio = {Pk_A[idx]/Pk_inf[idx]:.2f})")
print(f"  P_QNM (B)   = {Pk_B[idx]:.3e}  (ratio = {Pk_B[idx]/Pk_inf[idx]:.2f})")

# Check mode Q and overlap at CMB scales
n_cmb = int(k_ell200 / k0)
print(f"\nMode properties at first Planck peak (n~{n_cmb}):")
kn = k0 * (n_cmb + 1)
sigma_n = get_omega_I(n_cmb) * k0 / (2*np.pi)
print(f"  k_n = {kn:.4e}, sigma = {sigma_n:.4e}")
print(f"  Q = {kn/(2*sigma_n):.1f}")
print(f"  Peak spacing dk = {k0:.4e}")
print(f"  Overlap: sigma/dk = {sigma_n/k0:.2f}")
if sigma_n > k0:
    print("  -> HEAVY OVERLAP: modes blend into smooth continuum")
else:
    print("  -> PARTIAL OVERLAP: individual peaks may be resolvable")

# ============================================================
# Effective spectral index in the Planck range
# ============================================================
print("\n" + "="*60)
print("EFFECTIVE SPECTRAL INDEX")
print("="*60)

# Compute n_eff = d(ln P)/d(ln k) at several points in the Planck range
k_test_points = [0.01, 0.02, 0.05, 0.1, 0.15]
print("\n  k (Mpc^-1)    n_eff (Model B)    n_eff (inflation)")

for kt in k_test_points:
    dk = kt * 0.01  # 1% step for numerical derivative
    idx_lo = np.argmin(np.abs(k - (kt - dk)))
    idx_hi = np.argmin(np.abs(k - (kt + dk)))
    
    if Pk_B[idx_lo] > 0 and Pk_B[idx_hi] > 0:
        n_eff_B = np.log(Pk_B[idx_hi] / Pk_B[idx_lo]) / np.log(k[idx_hi] / k[idx_lo])
    else:
        n_eff_B = float('nan')
    
    n_eff_inf = 0.9649 - 1.0  # constant by construction
    
    print(f"  {kt:.3f}           {n_eff_B:+.4f}             {n_eff_inf:+.4f}")

print(f"\nInflation: n_s - 1 = -0.0351 everywhere (by construction)")
print(f"QNM: n_eff varies with k (from mode structure + j2 envelope)")
