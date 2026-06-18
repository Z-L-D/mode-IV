#!/usr/bin/env python3
"""
Option 3: Strip the envelope. Test whether the bare QNM mode structure
produces the right SHAPE for the CMB, independent of overall tilt.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn
from scipy.optimize import curve_fit

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

N_MODES = 300

def bare_mode_sum(k_array, gamma=1.0):
    """Pure mode sum — no cavity envelope at all."""
    kn_arr = np.zeros(N_MODES)
    sigma_arr = np.zeros(N_MODES)
    w_arr = np.zeros(N_MODES)
    
    for n in range(N_MODES):
        kn_arr[n] = k0 * (n + 1) * beta_geom
        sigma_arr[n] = get_omega_I(n) * k0 / (2.0 * np.pi)
        w_arr[n] = 1.0 / (n + 1)**gamma
    w_arr /= np.sum(w_arr)
    
    Pk = np.zeros_like(k_array)
    for i, k in enumerate(k_array):
        if k <= 0: continue
        s = 0.0
        for n in range(N_MODES):
            kn = kn_arr[n]
            sigma_n = sigma_arr[n]
            dk = k - kn
            lorentz = 1.0 / (dk**2 + sigma_n**2)
            asym = (k + kn)**2 / kn**2
            s += w_arr[n] * lorentz * asym
        Pk[i] = s
    return Pk

def P_inflation(k_array, A_s=2.1e-9, n_s=0.9649, k_pivot=0.05):
    return A_s * (k_array / k_pivot)**(n_s - 1.0)

# ============================================================
# Compute bare mode sums
# ============================================================

k = np.logspace(-4, -0.3, 3000)
Pk_inf = P_inflation(k)
D_rec = 14000
k_ell200 = 200.0 / D_rec
k_ell2500 = 2500.0 / D_rec

print("Computing bare mode sums...")
bare_10 = bare_mode_sum(k, gamma=1.0)
bare_15 = bare_mode_sum(k, gamma=1.5)
bare_07 = bare_mode_sum(k, gamma=0.7)

# ============================================================
# Fit natural tilt of bare mode sum in Planck range
# ============================================================

planck_mask = (k >= k_ell200) & (k <= k_ell2500)
k_planck = k[planck_mask]

def power_law(lnk, lnA, alpha):
    return lnA + alpha * lnk

for gamma_val, bare in [(1.0, bare_10), (1.5, bare_15), (0.7, bare_07)]:
    bare_planck = bare[planck_mask]
    valid = bare_planck > 0
    if np.sum(valid) > 10:
        popt, _ = curve_fit(power_law, np.log(k_planck[valid]), np.log(bare_planck[valid]))
        natural_tilt = popt[1]
        print(f"  gamma={gamma_val}: natural tilt of bare mode sum = {natural_tilt:+.3f}")
        print(f"    need tilt correction of {(-0.035 - natural_tilt):+.3f} to match inflation")

# ============================================================
# Apply tilt correction: multiply bare sum by k^correction
# ============================================================

print("\nApplying tilt corrections...")

# For gamma=1.0, find what k^alpha makes the tilt match n_s-1=-0.035
# If bare tilt is T, we need alpha = -0.035 - T
for gamma_val, bare in [(1.0, bare_10), (1.5, bare_15)]:
    bare_planck = bare[planck_mask]
    valid = bare_planck > 0
    popt, _ = curve_fit(power_law, np.log(k_planck[valid]), np.log(bare_planck[valid]))
    natural_tilt = popt[1]
    alpha_correction = -0.035 - natural_tilt
    
    # Corrected spectrum: bare * k^alpha * A_norm
    corrected = bare * k**alpha_correction
    
    # Normalize to match inflation at pivot
    idx_pivot = np.argmin(np.abs(k - 0.05))
    A_norm = Pk_inf[idx_pivot] / corrected[idx_pivot]
    corrected *= A_norm
    
    if gamma_val == 1.0:
        Pk_corrected_10 = corrected
        alpha_10 = alpha_correction
    else:
        Pk_corrected_15 = corrected
        alpha_15 = alpha_correction

print(f"  gamma=1.0: alpha_correction = {alpha_10:+.3f}")
print(f"  gamma=1.5: alpha_correction = {alpha_15:+.3f}")

# ============================================================
# THE KEY DIAGNOSTIC: residual structure after tilt removal
# ============================================================

# If QNM mode structure produces features BEYOND a smooth power law,
# those features are the cavity's signature. The ratio of corrected
# QNM to inflation reveals mode structure independent of tilt.

fig, axes = plt.subplots(4, 1, figsize=(14, 18), 
                          gridspec_kw={'height_ratios': [3, 3, 2, 2]})

# Panel 1: Bare mode sums (no envelope)
ax1 = axes[0]
ax1.loglog(k, bare_10, 'b-', lw=1.5, label='Bare modes gamma=1.0', alpha=0.8)
ax1.loglog(k, bare_15, 'g-', lw=1.5, label='Bare modes gamma=1.5', alpha=0.8)
ax1.loglog(k, bare_07, 'm-', lw=1.5, label='Bare modes gamma=0.7', alpha=0.8)
ax1.axvspan(k_ell200, k_ell2500, alpha=0.1, color='green', label='Planck range')
ax1.set_xlim(1e-4, 0.5)
ax1.set_xlabel('k [Mpc^-1]')
ax1.set_ylabel('Mode sum (arb units)')
ax1.set_title('Panel 1: Bare QNM Mode Sum (No Envelope)')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel 2: Tilt-corrected vs inflation
ax2 = axes[1]
ax2.loglog(k, Pk_corrected_10, 'b-', lw=2, 
           label=f'QNM corrected (g=1.0, k^{{{alpha_10:+.2f}}})', alpha=0.9)
ax2.loglog(k, Pk_corrected_15, 'g-', lw=1.5, 
           label=f'QNM corrected (g=1.5, k^{{{alpha_15:+.2f}}})', alpha=0.7)
ax2.loglog(k, Pk_inf, 'r--', lw=2, label='Inflation')
ax2.axvspan(k_ell200, k_ell2500, alpha=0.1, color='green')
ax2.set_xlim(0.005, 0.3)
ax2.set_xlabel('k [Mpc^-1]')
ax2.set_ylabel('P(k)')
ax2.set_title('Panel 2: Tilt-Corrected QNM vs Inflation (Planck Zoom)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Panel 3: Ratio (corrected QNM / inflation) — THE MONEY PLOT
# If this is flat ~1.0, mode structure adds nothing beyond a power law.
# If it has structure, the modes create features that survive tilt correction.
ax3 = axes[2]
with np.errstate(divide='ignore', invalid='ignore'):
    ratio_10 = np.where(Pk_inf > 0, Pk_corrected_10 / Pk_inf, np.nan)
    ratio_15 = np.where(Pk_inf > 0, Pk_corrected_15 / Pk_inf, np.nan)
ax3.semilogx(k, ratio_10, 'b-', lw=1.5, label='gamma=1.0', alpha=0.8)
ax3.semilogx(k, ratio_15, 'g-', lw=1.5, label='gamma=1.5', alpha=0.7)
ax3.axhline(1.0, color='r', ls='--', alpha=0.5)
ax3.axvspan(k_ell200, k_ell2500, alpha=0.1, color='green')
ax3.set_xlim(0.005, 0.3)
ax3.set_ylim(0.5, 1.5)
ax3.set_xlabel('k [Mpc^-1]')
ax3.set_ylabel('P_QNM_corrected / P_inflation')
ax3.set_title('Panel 3: Residual Mode Structure After Tilt Correction (THE KEY)')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Panel 4: Fractional deviation from smooth power law
# This shows whether mode structure creates ~1% features or ~100% features
ax4 = axes[3]
# Fit a smooth power law to the corrected spectrum in Planck range
corr_planck = Pk_corrected_10[planck_mask]
valid = corr_planck > 0
popt_smooth, _ = curve_fit(power_law, np.log(k_planck[valid]), np.log(corr_planck[valid]))
smooth_fit = np.exp(power_law(np.log(k), *popt_smooth))

fractional_dev = (Pk_corrected_10 - smooth_fit) / smooth_fit
ax4.semilogx(k, fractional_dev * 100, 'b-', lw=1, alpha=0.8)
ax4.axhline(0, color='r', ls='--', alpha=0.5)
ax4.axvspan(k_ell200, k_ell2500, alpha=0.1, color='green')
ax4.set_xlim(0.005, 0.3)
ax4.set_ylim(-50, 50)
ax4.set_xlabel('k [Mpc^-1]')
ax4.set_ylabel('% deviation from smooth power law')
ax4.set_title('Panel 4: Mode Structure Amplitude (% deviation from smooth trend)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('qnm_pk_option3.png', dpi=150, bbox_inches='tight')
print("\nSaved: qnm_pk_option3.png")

# ============================================================
# Summary diagnostics
# ============================================================

print("\n" + "="*60)
print("OPTION 3 DIAGNOSTIC SUMMARY")
print("="*60)

# Effective tilt of corrected spectrum
print("\nEffective tilt of corrected spectrum (should be ~-0.035):")
for kt in [0.01, 0.02, 0.05, 0.1, 0.15]:
    dk_step = kt * 0.02
    i_lo = np.argmin(np.abs(k - (kt - dk_step)))
    i_hi = np.argmin(np.abs(k - (kt + dk_step)))
    if Pk_corrected_10[i_lo] > 0 and Pk_corrected_10[i_hi] > 0:
        n_eff = np.log(Pk_corrected_10[i_hi]/Pk_corrected_10[i_lo]) / np.log(k[i_hi]/k[i_lo])
        print(f"  k={kt:.3f}: n_eff = {n_eff:+.4f}  (target: -0.0351)")

# RMS deviation from smooth power law in Planck range
dev_planck = fractional_dev[planck_mask]
valid_dev = np.isfinite(dev_planck)
rms_dev = np.sqrt(np.mean(dev_planck[valid_dev]**2))
max_dev = np.max(np.abs(dev_planck[valid_dev]))
print(f"\nMode structure amplitude in Planck range:")
print(f"  RMS deviation from smooth: {rms_dev*100:.1f}%")
print(f"  Max deviation from smooth: {max_dev*100:.1f}%")

if rms_dev < 0.01:
    print("\n  -> Mode structure is <1%: effectively a smooth power law.")
    print("     QNM modes blend into continuum. No distinctive features.")
    print("     Could fit Planck but indistinguishable from inflation.")
elif rms_dev < 0.10:
    print("\n  -> Mode structure is 1-10%: detectable features.")
    print("     This is the interesting regime — features large enough")
    print("     to test against Planck but small enough to not be ruled out.")
elif rms_dev < 0.50:
    print("\n  -> Mode structure is 10-50%: large oscillatory residuals.")
    print("     Would produce visible wiggles in C_l that Planck doesn't see")
    print("     (unless they happen to align with acoustic peak structure).")
else:
    print("\n  -> Mode structure is >50%: dominant oscillations.")
    print("     Incompatible with Planck unless features coincide with peaks.")
