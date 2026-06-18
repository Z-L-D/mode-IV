#!/usr/bin/env python3
"""
BAO Feature Analysis: QNM vs LCDM
===================================
Extracts BAO oscillation features from CLASS matter power spectra.
Compares QNM and LCDM BAO amplitude, damping, and shape.

Reads: bao_qnm_pk.dat, bao_lcdm_pk.dat (from CLASS)
Outputs: bao_comparison.png, bao_results.txt
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

# ============================================================
# Eisenstein-Hu no-wiggle reference spectrum
# (standard tool for isolating BAO feature)
# ============================================================

def eisenstein_hu_nowiggle(k_h, omega_m, omega_b, h, T_cmb=2.7255):
    """
    Eisenstein & Hu (1998) transfer function WITHOUT BAO wiggles.
    k_h in h/Mpc units.
    Returns T(k) (transfer function, not power spectrum).
    """
    # Physical densities
    om_mh2 = omega_m * h**2  # but we already have omega_m = Om*h^2? No.
    # omega_m here is Omega_m (dimensionless), omega_b is Omega_b
    # Actually, CLASS uses omega_b = Omega_b * h^2, omega_cdm = Omega_cdm * h^2
    # Let me use physical densities directly
    
    omh2 = omega_m  # omega_m = Omega_m * h^2 (physical)
    obh2 = omega_b  # omega_b = Omega_b * h^2 (physical)
    
    theta_cmb = T_cmb / 2.7
    
    # Eq. 26 - sound horizon
    s = 44.5 * np.log(9.83 / omh2) / np.sqrt(1 + 10 * obh2**0.75)
    
    # Eq. 31 - alpha_gamma
    alpha_gamma = 1 - 0.328 * np.log(431 * omh2) * (obh2 / omh2) + \
                  0.38 * np.log(22.3 * omh2) * (obh2 / omh2)**2
    
    # Eq. 30 - effective shape parameter
    gamma_eff = omh2 / h * (alpha_gamma + (1 - alpha_gamma) / (1 + (0.43 * k_h * s)**4))
    
    # Eq. 28 - transfer function (no wiggles)
    q = k_h * theta_cmb**2 / gamma_eff
    L = np.log(2 * np.e + 1.8 * q)
    C = 14.2 + 731.0 / (1 + 62.5 * q)
    T = L / (L + C * q**2)
    
    return T


def compute_nowiggle_pk(k_h, pk, omega_m_phys, omega_b_phys, h):
    """
    Compute the no-wiggle reference P(k) using Eisenstein-Hu.
    
    The idea: P_nw(k) = P_primordial(k) * T_nw(k)^2 * (normalization)
    We normalize by matching to the actual P(k) in a smooth sense.
    """
    T_nw = eisenstein_hu_nowiggle(k_h, omega_m_phys, omega_b_phys, h)
    
    # The no-wiggle P(k) shape
    pk_nw_shape = k_h * T_nw**2  # P(k) ~ k * T^2 for n_s ~ 1
    
    # Normalize to match the actual P(k) on average
    # Use broad smoothing to get the envelope
    log_ratio = np.log(pk / pk_nw_shape)
    
    # Fit a smooth polynomial to the log ratio
    mask = (k_h > 0.01) & (k_h < 0.5) & np.isfinite(log_ratio)
    if np.sum(mask) < 10:
        return pk_nw_shape * np.exp(np.median(log_ratio[mask]))
    
    coeffs = np.polyfit(np.log(k_h[mask]), log_ratio[mask], 3)
    smooth_ratio = np.polyval(coeffs, np.log(k_h))
    
    pk_nw = pk_nw_shape * np.exp(smooth_ratio)
    return pk_nw


# ============================================================
# Alternative: Savitzky-Golay smoothing for no-wiggle
# ============================================================

def smooth_nowiggle(k, pk, window=51):
    """Extract no-wiggle by Savitzky-Golay smoothing in log space."""
    log_pk = np.log(pk)
    log_k = np.log(k)
    
    # Interpolate to uniform log spacing
    log_k_uniform = np.linspace(log_k.min(), log_k.max(), len(k))
    log_pk_uniform = np.interp(log_k_uniform, log_k, log_pk)
    
    # Smooth
    if window > len(log_pk_uniform):
        window = len(log_pk_uniform) // 2 * 2 - 1
    if window < 5:
        window = 5
    
    log_pk_smooth = savgol_filter(log_pk_uniform, window, 3)
    
    # Interpolate back
    pk_smooth = np.exp(np.interp(log_k, log_k_uniform, log_pk_smooth))
    return pk_smooth


# ============================================================
# Load CLASS output
# ============================================================

def load_class_pk(filepath):
    """Load CLASS matter power spectrum output.
    
    CLASS pk format: columns are k (h/Mpc) and P(k) (Mpc/h)^3
    at different redshifts. First column is k, subsequent are P(k,z).
    """
    data = []
    header_lines = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                header_lines.append(line)
                continue
            if not line:
                continue
            parts = line.split()
            try:
                row = [float(x) for x in parts]
                data.append(row)
            except ValueError:
                continue
    
    if not data:
        return None, None
    
    data = np.array(data)
    k = data[:, 0]  # k in h/Mpc
    pk = data[:, 1]  # P(k) at z=0 (first redshift)
    
    # Check for additional redshifts
    pk_all = {}
    pk_all['z0'] = pk
    if data.shape[1] > 2:
        pk_all['z1'] = data[:, 2]  # z=0.5
    if data.shape[1] > 3:
        pk_all['z2'] = data[:, 3]  # z=1.0
    if data.shape[1] > 4:
        pk_all['z3'] = data[:, 4]  # z=2.0
    
    return k, pk_all


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("BAO Feature Analysis: QNM vs ΛCDM")
print("=" * 60)

# Load data
print("\nLoading CLASS matter power spectra...")

k_qnm, pk_qnm_all = load_class_pk('bao_qnm_pk.dat')
k_lcdm, pk_lcdm_all = load_class_pk('bao_lcdm_pk.dat')

if k_qnm is None or k_lcdm is None:
    print("ERROR: Could not load P(k) files.")
    print("Make sure bao_qnm_pk.dat and bao_lcdm_pk.dat exist.")
    print("Run bao_extraction.sh first.")
    exit(1)

pk_qnm = pk_qnm_all['z0']
pk_lcdm = pk_lcdm_all['z0']

print(f"  QNM: {len(k_qnm)} k-values, k = {k_qnm[0]:.4e} to {k_qnm[-1]:.4e} h/Mpc")
print(f"  ΛCDM: {len(k_lcdm)} k-values, k = {k_lcdm[0]:.4e} to {k_lcdm[-1]:.4e} h/Mpc")

# Cosmological parameters
h = 0.6736
omega_b = 0.02237  # physical
omega_cdm = 0.12    # physical
omega_m = omega_b + omega_cdm  # physical Omega_m h^2

# ============================================================
# Extract BAO features
# ============================================================

print("\nExtracting BAO features...")

# Method 1: Savitzky-Golay smoothing
pk_qnm_nw = smooth_nowiggle(k_qnm, pk_qnm, window=101)
pk_lcdm_nw = smooth_nowiggle(k_lcdm, pk_lcdm, window=101)

# BAO ratio: P(k)/P_nw(k) - 1
bao_qnm = pk_qnm / pk_qnm_nw - 1
bao_lcdm = pk_lcdm / pk_lcdm_nw - 1

# ============================================================
# Analysis
# ============================================================

print("\nBAO feature comparison:")

# BAO region: k = 0.02 - 0.35 h/Mpc
bao_mask_qnm = (k_qnm > 0.02) & (k_qnm < 0.35)
bao_mask_lcdm = (k_lcdm > 0.02) & (k_lcdm < 0.35)

# RMS amplitude of BAO wiggles
bao_rms_qnm = np.sqrt(np.mean(bao_qnm[bao_mask_qnm]**2))
bao_rms_lcdm = np.sqrt(np.mean(bao_lcdm[bao_mask_lcdm]**2))

print(f"  BAO RMS amplitude (k=0.02-0.35):")
print(f"    QNM:  {bao_rms_qnm:.4f}")
print(f"    ΛCDM: {bao_rms_lcdm:.4f}")
print(f"    Ratio: {bao_rms_qnm/bao_rms_lcdm:.4f}")

# Peak-by-peak comparison
# Find BAO peaks in the LCDM spectrum
from scipy.signal import find_peaks

peaks_lcdm, _ = find_peaks(bao_lcdm[bao_mask_lcdm], height=0.01, distance=20)
troughs_lcdm, _ = find_peaks(-bao_lcdm[bao_mask_lcdm], height=0.01, distance=20)

k_bao = k_lcdm[bao_mask_lcdm]

print(f"\n  BAO peaks in ΛCDM spectrum:")
for i, p in enumerate(peaks_lcdm[:5]):
    print(f"    Peak {i+1}: k = {k_bao[p]:.4f} h/Mpc, amplitude = {bao_lcdm[bao_mask_lcdm][p]:.4f}")

# Direct P(k) ratio
print(f"\n  Direct P(k) ratio QNM/ΛCDM at BAO scales:")
# Interpolate to common k grid
k_common = np.logspace(np.log10(0.02), np.log10(0.35), 200)
pk_qnm_interp = np.interp(k_common, k_qnm, pk_qnm)
pk_lcdm_interp = np.interp(k_common, k_lcdm, pk_lcdm)
ratio = pk_qnm_interp / pk_lcdm_interp

print(f"    k=0.03: ratio = {np.interp(0.03, k_common, ratio):.4f}")
print(f"    k=0.05: ratio = {np.interp(0.05, k_common, ratio):.4f}")
print(f"    k=0.10: ratio = {np.interp(0.10, k_common, ratio):.4f}")
print(f"    k=0.15: ratio = {np.interp(0.15, k_common, ratio):.4f}")
print(f"    k=0.20: ratio = {np.interp(0.20, k_common, ratio):.4f}")
print(f"    k=0.30: ratio = {np.interp(0.30, k_common, ratio):.4f}")
print(f"    Mean ratio (0.02-0.35): {np.mean(ratio):.4f}")
print(f"    Std ratio (0.02-0.35): {np.std(ratio):.4f}")

# BAO feature DIFFERENCE
bao_qnm_interp = np.interp(k_common, k_qnm, bao_qnm)
bao_lcdm_interp = np.interp(k_common, k_lcdm, bao_lcdm)
bao_diff = bao_qnm_interp - bao_lcdm_interp

print(f"\n  BAO feature shape difference (QNM - ΛCDM):")
print(f"    Max |difference|: {np.max(np.abs(bao_diff)):.4f}")
print(f"    RMS difference: {np.sqrt(np.mean(bao_diff**2)):.4f}")
print(f"    Mean difference: {np.mean(bao_diff):.4f}")

# ============================================================
# PLOTS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: Full P(k) comparison
ax = axes[0, 0]
ax.loglog(k_qnm, pk_qnm, 'b-', linewidth=1.5, alpha=0.8, label='QNM')
ax.loglog(k_lcdm, pk_lcdm, 'r-', linewidth=1.5, alpha=0.8, label='ΛCDM')
ax.axvline(0.02, color='gray', linestyle=':', alpha=0.3)
ax.axvline(0.35, color='gray', linestyle=':', alpha=0.3)
ax.set_xlabel('k [h/Mpc]', fontsize=13)
ax.set_ylabel('P(k) [(Mpc/h)³]', fontsize=13)
ax.set_title('Matter Power Spectrum at z = 0', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.set_xlim(0.005, 0.5)
ax.grid(True, alpha=0.2)

# Panel 2: P(k) ratio
ax = axes[0, 1]
ax.semilogx(k_common, ratio, 'b-', linewidth=2)
ax.axhline(1.0, color='r', linestyle='--', alpha=0.5)
ax.fill_between(k_common, 0.99, 1.01, alpha=0.1, color='green', label='±1% (DESI precision)')
ax.set_xlabel('k [h/Mpc]', fontsize=13)
ax.set_ylabel('P_QNM / P_ΛCDM', fontsize=13)
ax.set_title('Matter P(k) Ratio', fontsize=14, fontweight='bold')
ax.set_xlim(0.02, 0.35)
ax.set_ylim(0.9, 1.1)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.2)

# Panel 3: BAO feature comparison
ax = axes[1, 0]
ax.plot(k_qnm[bao_mask_qnm], bao_qnm[bao_mask_qnm], 'b-', linewidth=1.5, 
        alpha=0.8, label='QNM')
ax.plot(k_lcdm[bao_mask_lcdm], bao_lcdm[bao_mask_lcdm], 'r-', linewidth=1.5, 
        alpha=0.8, label='ΛCDM')
ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax.set_xlabel('k [h/Mpc]', fontsize=13)
ax.set_ylabel('P(k)/P_nw(k) - 1', fontsize=13)
ax.set_title('BAO Feature (oscillations isolated)', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.set_xlim(0.02, 0.35)
ax.grid(True, alpha=0.2)

# Panel 4: BAO difference
ax = axes[1, 1]
ax.plot(k_common, bao_diff, 'purple', linewidth=2)
ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax.fill_between(k_common, -0.005, 0.005, alpha=0.1, color='green', 
                label='±0.5% (est. DESI sensitivity)')
ax.set_xlabel('k [h/Mpc]', fontsize=13)
ax.set_ylabel('BAO_QNM - BAO_ΛCDM', fontsize=13)
ax.set_title('BAO Feature Shape Difference', fontsize=14, fontweight='bold')
ax.set_xlim(0.02, 0.35)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('bao_comparison.png', dpi=150, bbox_inches='tight')
print("\nSaved: bao_comparison.png")

# ============================================================
# Multi-redshift comparison (if available)
# ============================================================

if 'z1' in pk_qnm_all and 'z1' in pk_lcdm_all:
    print("\n\nMulti-redshift P(k) ratios:")
    z_labels = ['z=0', 'z=0.5', 'z=1.0', 'z=2.0']
    z_keys = ['z0', 'z1', 'z2', 'z3']
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, (zk, zl) in enumerate(zip(z_keys, z_labels)):
        if zk in pk_qnm_all and zk in pk_lcdm_all:
            pk_q = pk_qnm_all[zk]
            pk_l = pk_lcdm_all[zk]
            r = np.interp(k_common, k_qnm, pk_q) / np.interp(k_common, k_lcdm, pk_l)
            ax2.semilogx(k_common, r, color=colors[i], linewidth=2, label=zl)
            print(f"  {zl}: mean ratio = {np.mean(r):.4f}, std = {np.std(r):.4f}")
    
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('k [h/Mpc]', fontsize=13)
    ax2.set_ylabel('P_QNM / P_ΛCDM', fontsize=13)
    ax2.set_title('Matter P(k) Ratio at Multiple Redshifts', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.set_xlim(0.02, 0.35)
    ax2.set_ylim(0.85, 1.15)
    ax2.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('bao_redshift_evolution.png', dpi=150, bbox_inches='tight')
    print("  Saved: bao_redshift_evolution.png")

# ============================================================
# Summary
# ============================================================

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"""
BAO feature comparison at z = 0:
  BAO RMS: QNM = {bao_rms_qnm:.4f}, ΛCDM = {bao_rms_lcdm:.4f} (ratio {bao_rms_qnm/bao_rms_lcdm:.3f})
  P(k) ratio: {np.mean(ratio):.4f} ± {np.std(ratio):.4f} across BAO range
  BAO shape difference: RMS = {np.sqrt(np.mean(bao_diff**2)):.4f}

Interpretation:
  If ratio ≈ 1.00 ± 0.01: BAO features are indistinguishable
  If ratio deviates > 1%: potentially detectable by DESI (k-dependent)
  If BAO shape differs > 0.5%: potentially detectable in BAO reconstruction
""")

# Save numerical results
with open('bao_results.txt', 'w') as f:
    f.write("BAO Feature Analysis Results\n")
    f.write(f"BAO RMS QNM: {bao_rms_qnm:.6f}\n")
    f.write(f"BAO RMS LCDM: {bao_rms_lcdm:.6f}\n")
    f.write(f"RMS ratio: {bao_rms_qnm/bao_rms_lcdm:.6f}\n")
    f.write(f"P(k) mean ratio: {np.mean(ratio):.6f}\n")
    f.write(f"P(k) std ratio: {np.std(ratio):.6f}\n")
    f.write(f"BAO shape RMS diff: {np.sqrt(np.mean(bao_diff**2)):.6f}\n")
    f.write(f"\nk, ratio, bao_qnm, bao_lcdm, bao_diff\n")
    for i in range(len(k_common)):
        f.write(f"{k_common[i]:.6f}, {ratio[i]:.6f}, {bao_qnm_interp[i]:.6f}, "
                f"{bao_lcdm_interp[i]:.6f}, {bao_diff[i]:.6f}\n")

print("Saved: bao_results.txt")
