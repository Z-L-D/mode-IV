#!/usr/bin/env python3
"""
qnm_periodogram.py — Search for discrete mode structure in unbinned Planck CMB

Prediction: QNM mode spacing Δk = k₀ ≈ 7.06e-4 Mpc⁻¹ maps to
Δℓ ≈ k₀ × D_A ≈ 7.06e-4 × 14000 ≈ 10 multipoles.

If the primordial spectrum is a sum of discrete modes, residuals
(data - smooth model) should show periodic structure at this spacing,
strongest in the damping tail (ℓ > 1000) where mode overlap is partial.

ΛCDM, with its perfectly smooth P(k), predicts no such periodicity.
"""
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

T_CMB = 2.7255e6  # μK

# ================================================================
# LOAD DATA
# ================================================================

tt_data = np.loadtxt('planck_tt_unbinned.txt')
ee_data = np.loadtxt('planck_ee_unbinned.txt')

# Also load QNM and ΛCDM CLASS outputs for comparison
try:
    qnm_cl = np.loadtxt('class_qnm/output/qnm_test_100_cl_lensed.dat')
    qnm_ell = qnm_cl[:, 0]
    qnm_tt_cl = qnm_cl[:, 1] * T_CMB**2
    qnm_ee_cl = qnm_cl[:, 2] * T_CMB**2
    has_qnm = True
    print("Loaded QNM CLASS output")
except:
    has_qnm = False
    print("No QNM CLASS output found")

# ================================================================
# ANALYSIS FUNCTION
# ================================================================

def analyze_spectrum(ell, dl, err, name, qnm_interp=None,
                     ell_ranges=None):
    """
    Compute periodogram of residuals for a given spectrum.
    """
    if ell_ranges is None:
        ell_ranges = {
            'Full (30-2500)': (30, 2500),
            'Peaks (200-800)': (200, 800),
            'Damping tail (1000-2000)': (1000, 2000),
            'High ℓ (1500-2500)': (1500, 2500),
        }

    results = {}

    for label, (ell_min, ell_max) in ell_ranges.items():
        mask = (ell >= ell_min) & (ell <= ell_max) & np.isfinite(dl) & (err > 0)
        e = ell[mask]
        d = dl[mask]
        s = err[mask]

        if len(e) < 50:
            continue

        # Subtract smooth baseline using Savitzky-Golay filter
        # Window must be odd, ~10% of data length
        win = min(len(d) // 5, 201)
        if win % 2 == 0:
            win += 1
        win = max(win, 11)

        smooth = savgol_filter(d, win, 3)
        resid = (d - smooth) / s  # weighted residuals

        # Lomb-Scargle periodogram (scipy)
        from scipy.signal import lombscargle
        # Angular frequency grid: periods from 5 to 100 multipoles
        freq_min = 1.0 / 100.0
        freq_max = 1.0 / 4.0
        freq = np.linspace(freq_min, freq_max, 2000)
        angular_freq = 2.0 * np.pi * freq

        # Normalize residuals to zero mean
        resid_centered = resid - np.mean(resid)
        power = lombscargle(e, resid_centered, angular_freq, normalize=True)
        period = 1.0 / freq

        # Peak
        peak_idx = np.argmax(power)
        peak_period = period[peak_idx]
        peak_power = power[peak_idx]
        # Approximate FAP: Baluev (2008) for normalized periodogram
        M = len(freq)  # number of independent frequencies
        fap = 1.0 - (1.0 - np.exp(-peak_power))**M
        fap = min(fap, 1.0)

        results[label] = {
            'freq': freq, 'power': power, 'period': period,
            'peak_period': peak_period, 'peak_power': peak_power,
            'fap': fap, 'ell': e, 'resid': resid, 'smooth': smooth,
            'data': d, 'err': s,
        }

        # Also compute periodogram of QNM residuals if available
        if qnm_interp is not None:
            qnm_at_ell = qnm_interp(e)
            resid_qnm = (d - qnm_at_ell) / s
            resid_qnm_centered = resid_qnm - np.mean(resid_qnm)
            power_qnm = lombscargle(e, resid_qnm_centered, angular_freq,
                                    normalize=True)
            results[label]['power_qnm'] = power_qnm
            results[label]['resid_qnm'] = resid_qnm

    return results

# ================================================================
# QNM PREDICTION
# ================================================================

# Cavity parameters
H0 = 67.36  # km/s/Mpc
c_kms = 299792.458
rs = c_kms / H0
k0 = np.pi / rs
D_A = 14000.0  # Mpc, angular diameter distance to last scattering

predicted_delta_ell = k0 * D_A
print(f"\nQNM PREDICTION:")
print(f"  k₀ = π/r_s = {k0:.6e} Mpc⁻¹")
print(f"  D_A ≈ {D_A:.0f} Mpc")
print(f"  Predicted Δℓ = k₀ × D_A = {predicted_delta_ell:.1f} multipoles")
print(f"  Predicted frequency = {1/predicted_delta_ell:.4f} cycles/multipole")

# ================================================================
# RUN ANALYSIS
# ================================================================

# Prepare QNM interpolation
qnm_tt_interp = None
qnm_ee_interp = None
if has_qnm:
    qnm_tt_interp = interp1d(qnm_ell, qnm_tt_cl, kind='cubic',
                              bounds_error=False, fill_value=0)
    qnm_ee_interp = interp1d(qnm_ell, qnm_ee_cl, kind='cubic',
                              bounds_error=False, fill_value=0)

print(f"\n{'='*70}")
print("TT ANALYSIS")
print(f"{'='*70}")

tt_ell = tt_data[:, 0]
tt_dl = tt_data[:, 1]
tt_err = 0.5 * (np.abs(tt_data[:, 2]) + np.abs(tt_data[:, 3]))

tt_results = analyze_spectrum(tt_ell, tt_dl, tt_err, 'TT',
                              qnm_interp=qnm_tt_interp)

for label, r in tt_results.items():
    print(f"\n  {label}:")
    print(f"    Peak period: {r['peak_period']:.1f} multipoles "
          f"(predicted: {predicted_delta_ell:.1f})")
    print(f"    Peak power: {r['peak_power']:.4f}")
    print(f"    False alarm prob: {r['fap']:.4e}")
    near_pred = np.argmin(np.abs(r['period'] - predicted_delta_ell))
    print(f"    Power at predicted period ({predicted_delta_ell:.1f}): "
          f"{r['power'][near_pred]:.4f}")

print(f"\n{'='*70}")
print("EE ANALYSIS")
print(f"{'='*70}")

ee_ell = ee_data[:, 0]
ee_dl = ee_data[:, 1]
ee_err = 0.5 * (np.abs(ee_data[:, 2]) + np.abs(ee_data[:, 3]))

ee_results = analyze_spectrum(ee_ell, ee_dl, ee_err, 'EE',
                              qnm_interp=qnm_ee_interp)

for label, r in ee_results.items():
    print(f"\n  {label}:")
    print(f"    Peak period: {r['peak_period']:.1f} multipoles "
          f"(predicted: {predicted_delta_ell:.1f})")
    print(f"    Peak power: {r['peak_power']:.4f}")
    print(f"    False alarm prob: {r['fap']:.4e}")
    near_pred = np.argmin(np.abs(r['period'] - predicted_delta_ell))
    print(f"    Power at predicted period ({predicted_delta_ell:.1f}): "
          f"{r['power'][near_pred]:.4f}")

# ================================================================
# PLOTS
# ================================================================

fig, axes = plt.subplots(4, 2, figsize=(16, 18))
fig.suptitle(f'QNM Mode Spacing Search — Predicted Δℓ = {predicted_delta_ell:.1f}',
             fontsize=14, y=0.98)

for col, (spec_name, spec_results) in enumerate([('TT', tt_results),
                                                   ('EE', ee_results)]):
    # Row 1: Periodogram — damping tail
    ax = axes[0, col]
    key = 'Damping tail (1000-2000)'
    if key in spec_results:
        r = spec_results[key]
        ax.plot(r['period'], r['power'], 'b-', lw=0.8,
                label='Data − smooth')
        if 'power_qnm' in r:
            ax.plot(r['period'], r['power_qnm'], 'r-', lw=0.8,
                    alpha=0.7, label='Data − QNM')
        ax.axvline(predicted_delta_ell, color='green', ls='--', lw=2,
                   label=f'QNM prediction ({predicted_delta_ell:.1f})')
        ax.set_xlabel('Period (multipoles)')
        ax.set_ylabel('Lomb-Scargle Power')
        ax.set_title(f'{spec_name} Periodogram — ℓ = 1000-2000')
        ax.legend(fontsize=7)
        ax.set_xlim(4, 60)

    # Row 2: Periodogram — high ℓ
    ax = axes[1, col]
    key = 'High ℓ (1500-2500)'
    if key in spec_results:
        r = spec_results[key]
        ax.plot(r['period'], r['power'], 'b-', lw=0.8,
                label='Data − smooth')
        if 'power_qnm' in r:
            ax.plot(r['period'], r['power_qnm'], 'r-', lw=0.8,
                    alpha=0.7, label='Data − QNM')
        ax.axvline(predicted_delta_ell, color='green', ls='--', lw=2,
                   label=f'QNM prediction ({predicted_delta_ell:.1f})')
        ax.set_xlabel('Period (multipoles)')
        ax.set_ylabel('Lomb-Scargle Power')
        ax.set_title(f'{spec_name} Periodogram — ℓ = 1500-2500')
        ax.legend(fontsize=7)
        ax.set_xlim(4, 60)

    # Row 3: Periodogram — full range
    ax = axes[2, col]
    key = 'Full (30-2500)'
    if key in spec_results:
        r = spec_results[key]
        ax.plot(r['period'], r['power'], 'b-', lw=0.8,
                label='Data − smooth')
        if 'power_qnm' in r:
            ax.plot(r['period'], r['power_qnm'], 'r-', lw=0.8,
                    alpha=0.7, label='Data − QNM')
        ax.axvline(predicted_delta_ell, color='green', ls='--', lw=2,
                   label=f'QNM prediction ({predicted_delta_ell:.1f})')
        ax.set_xlabel('Period (multipoles)')
        ax.set_ylabel('Lomb-Scargle Power')
        ax.set_title(f'{spec_name} Periodogram — Full Range')
        ax.legend(fontsize=7)
        ax.set_xlim(4, 60)

    # Row 4: Residuals in the damping tail
    ax = axes[3, col]
    key = 'Damping tail (1000-2000)'
    if key in spec_results:
        r = spec_results[key]
        ax.plot(r['ell'], r['resid'], 'b.', ms=2, alpha=0.5,
                label='Data − smooth')
        if 'resid_qnm' in r:
            ax.plot(r['ell'], r['resid_qnm'], 'r.', ms=2, alpha=0.5,
                    label='Data − QNM')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel('ℓ')
        ax.set_ylabel('Weighted Residual (σ)')
        ax.set_title(f'{spec_name} Residuals — ℓ = 1000-2000')
        ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('qnm_periodogram.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved: qnm_periodogram.png")

# ================================================================
# QUANTITATIVE SUMMARY
# ================================================================

print(f"\n{'='*70}")
print("SUMMARY — Is there a signal at the predicted QNM period?")
print(f"{'='*70}")

for spec_name, spec_results in [('TT', tt_results), ('EE', ee_results)]:
    print(f"\n  {spec_name}:")
    for label, r in spec_results.items():
        near_pred = np.argmin(np.abs(r['period'] - predicted_delta_ell))
        power_at_pred = r['power'][near_pred]
        # What fraction of the periodogram exceeds this power?
        frac_above = np.mean(r['power'] > power_at_pred)
        print(f"    {label}:")
        print(f"      Power at Δℓ={predicted_delta_ell:.1f}: {power_at_pred:.4f} "
              f"(top {100*frac_above:.0f}% of frequencies)")
        if 'power_qnm' in r:
            power_qnm_at_pred = r['power_qnm'][near_pred]
            print(f"      QNM residual power at same period: {power_qnm_at_pred:.4f}")
            if power_qnm_at_pred < power_at_pred:
                print(f"      >>> QNM ABSORBS {(1-power_qnm_at_pred/power_at_pred)*100:.0f}% "
                      f"of the signal (predicted if QNM captures mode structure)")
            else:
                print(f"      >>> QNM does NOT absorb the signal")
