#!/usr/bin/env python3
"""
QNM Parameter Scan: Omega_b and H0 optimization
================================================
Scans over baryon density and Hubble constant to minimize
chi2 against Planck 2018 TT binned data.

QNM primordial spectrum params are FIXED: gamma=1.04, alpha=0.00
Only cosmological transport params are varied.

Run from ~/qnm-boltzmann/
Requires: CLASS compiled in class_qnm/, planck_tt_binned.txt
"""

import subprocess
import numpy as np
import os
import sys
import shutil
import re

# ============================================================
# Configuration
# ============================================================

CLASS_DIR = os.path.expanduser("~/qnm-boltzmann/class_qnm")
CLASS_BIN = os.path.join(CLASS_DIR, "class")
INI_TEMPLATE = os.path.join(CLASS_DIR, "qnm_test.ini")
OUTPUT_DIR = os.path.join(CLASS_DIR, "output")
PLANCK_FILE = os.path.expanduser("~/qnm-boltzmann/planck_tt_binned.txt")

# Parameter grid
# CLASS default: omega_b = 0.02237, H0 = 0.6736
# Planck best-fit: omega_b = 0.02237, H0 = 0.6736
# Scan range: ±5% around defaults
OMEGA_B_VALUES = [0.0210, 0.0215, 0.0220, 0.02237, 0.0226, 0.0230, 0.0235, 0.0240]
H0_VALUES = [0.640, 0.650, 0.660, 0.670, 0.6736, 0.680, 0.690, 0.700]

# Fixed QNM params
QNM_GAMMA = 1.04
QNM_ALPHA = 0.00
QNM_A = 2.85e-8

# ============================================================
# Load Planck data
# ============================================================

def load_planck_tt(filepath):
    """Load Planck 2018 binned TT data."""
    data = np.loadtxt(filepath)
    # Columns: ell_center, D_ell, sigma_minus, sigma_plus
    ell = data[:, 0]
    D_ell = data[:, 1]
    # Use average of asymmetric errors
    sigma = (data[:, 2] + data[:, 3]) / 2
    return ell, D_ell, sigma

# ============================================================
# Modify ini file
# ============================================================

def create_ini(omega_b, H0, ini_path):
    """Create modified ini file with specified params."""
    # Read template
    with open(INI_TEMPLATE, 'r') as f:
        content = f.read()
    
    # Replace omega_b (CLASS uses omega_b = Omega_b * h^2)
    content = re.sub(r'omega_b\s*=\s*[\d.eE+-]+', f'omega_b = {omega_b}', content)
    
    # Replace H0
    content = re.sub(r'^h\s*=\s*[\d.eE+-]+', f'h = {H0}', content, flags=re.MULTILINE)
    
    # Ensure QNM params are correct
    content = re.sub(r'qnm_gamma\s*=\s*[\d.eE+-]+', f'qnm_gamma = {QNM_GAMMA}', content)
    content = re.sub(r'qnm_alpha\s*=\s*[\d.eE+-]+', f'qnm_alpha = {QNM_ALPHA}', content)
    content = re.sub(r'qnm_A\s*=\s*[\d.eE+-]+', f'qnm_A = {QNM_A}', content)
    
    # Write modified ini
    with open(ini_path, 'w') as f:
        f.write(content)

# ============================================================
# Run CLASS and compute chi2
# ============================================================

def run_class_and_chi2(omega_b, H0, planck_ell, planck_Dl, planck_sigma):
    """Run CLASS with given params, return chi2 against Planck."""
    
    # Create temporary ini
    tmp_ini = os.path.join(CLASS_DIR, "scan_tmp.ini")
    create_ini(omega_b, H0, tmp_ini)
    
    # Run CLASS
    try:
        result = subprocess.run(
            [CLASS_BIN, "scan_tmp.ini"],
            cwd=CLASS_DIR,
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f"  CLASS failed: {result.stderr[:200]}")
            return np.inf
    except subprocess.TimeoutExpired:
        print(f"  CLASS timed out")
        return np.inf
    except Exception as e:
        print(f"  CLASS error: {e}")
        return np.inf
    
    # Find output file
    output_files = sorted(
        [f for f in os.listdir(OUTPUT_DIR) if 'scan_tmp' in f and '_cl.dat' in f and 'lensed' not in f],
        key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)),
        reverse=True
    )
    
    if not output_files:
        # Try any recent cl.dat file
        output_files = sorted(
            [f for f in os.listdir(OUTPUT_DIR) if '_cl.dat' in f and 'lensed' not in f],
            key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)),
            reverse=True
        )
    
    if not output_files:
        print(f"  No output file found")
        return np.inf
    
    output_path = os.path.join(OUTPUT_DIR, output_files[0])
    
    # Load CLASS output
    try:
        cl_data = np.loadtxt(output_path)
        cl_ell = cl_data[:, 0]
        cl_tt = cl_data[:, 1]  # D_ell^TT in muK^2
    except Exception as e:
        print(f"  Failed to load output: {e}")
        return np.inf
    
    # Interpolate CLASS to Planck ell values
    from scipy.interpolate import interp1d
    
    # Only use ell >= 30 (skip low-ell cosmic variance dominated bins)
    mask = planck_ell >= 30
    p_ell = planck_ell[mask]
    p_Dl = planck_Dl[mask]
    p_sig = planck_sigma[mask]
    
    # Interpolate
    valid = (cl_ell >= p_ell.min()) & (cl_ell <= p_ell.max())
    if np.sum(valid) < 10:
        return np.inf
    
    interp_func = interp1d(cl_ell[valid], cl_tt[valid], kind='cubic', fill_value='extrapolate')
    model_at_planck = interp_func(p_ell)
    
    # Chi2
    chi2 = np.sum(((p_Dl - model_at_planck) / p_sig) ** 2)
    dof = len(p_ell) - 1  # subtract 1 for amplitude freedom
    
    # Clean up output files from this run
    for f in output_files[:3]:
        try:
            os.remove(os.path.join(OUTPUT_DIR, f))
        except:
            pass
    
    return chi2

# ============================================================
# Main scan
# ============================================================

if __name__ == "__main__":
    print("QNM Omega_b / H0 Parameter Scan")
    print("=" * 60)
    print(f"Fixed: gamma={QNM_GAMMA}, alpha={QNM_ALPHA}, A={QNM_A}")
    print(f"Scanning: omega_b = {OMEGA_B_VALUES}")
    print(f"          H0 = {H0_VALUES}")
    print(f"Grid: {len(OMEGA_B_VALUES)} x {len(H0_VALUES)} = {len(OMEGA_B_VALUES)*len(H0_VALUES)} runs")
    print()
    
    # Check prerequisites
    if not os.path.exists(CLASS_BIN):
        print(f"ERROR: CLASS binary not found at {CLASS_BIN}")
        sys.exit(1)
    if not os.path.exists(PLANCK_FILE):
        print(f"ERROR: Planck data not found at {PLANCK_FILE}")
        sys.exit(1)
    if not os.path.exists(INI_TEMPLATE):
        print(f"ERROR: ini template not found at {INI_TEMPLATE}")
        sys.exit(1)
    
    # Load Planck data
    planck_ell, planck_Dl, planck_sigma = load_planck_tt(PLANCK_FILE)
    n_bins = np.sum(planck_ell >= 30)
    print(f"Planck TT: {n_bins} bins (ell >= 30)")
    
    # Run grid scan
    results = np.zeros((len(OMEGA_B_VALUES), len(H0_VALUES)))
    
    total = len(OMEGA_B_VALUES) * len(H0_VALUES)
    count = 0
    
    for i, ob in enumerate(OMEGA_B_VALUES):
        for j, h0 in enumerate(H0_VALUES):
            count += 1
            print(f"[{count}/{total}] omega_b={ob:.5f}, h={h0:.4f} ... ", end='', flush=True)
            
            chi2 = run_class_and_chi2(ob, h0, planck_ell, planck_Dl, planck_sigma)
            results[i, j] = chi2
            chi2_dof = chi2 / (n_bins - 1)
            
            print(f"chi2={chi2:.1f}, chi2/dof={chi2_dof:.3f}")
    
    # ============================================================
    # Results
    # ============================================================
    
    print("\n" + "=" * 60)
    print("RESULTS GRID: chi2/dof")
    print("=" * 60)
    
    # Header
    header = f"{'omega_b':>10s} |"
    for h0 in H0_VALUES:
        header += f" {h0:>7.2f}"
    print(header)
    print("-" * len(header))
    
    for i, ob in enumerate(OMEGA_B_VALUES):
        row = f"{ob:10.5f} |"
        for j, h0 in enumerate(H0_VALUES):
            chi2_dof = results[i, j] / (n_bins - 1)
            row += f" {chi2_dof:7.3f}"
        print(row)
    
    # Find minimum
    min_idx = np.unravel_index(np.argmin(results), results.shape)
    best_ob = OMEGA_B_VALUES[min_idx[0]]
    best_h0 = H0_VALUES[min_idx[1]]
    best_chi2 = results[min_idx]
    best_chi2_dof = best_chi2 / (n_bins - 1)
    
    print(f"\nBEST FIT:")
    print(f"  omega_b = {best_ob:.5f}")
    print(f"  H0 = {best_h0:.2f}")
    print(f"  chi2 = {best_chi2:.1f}")
    print(f"  chi2/dof = {best_chi2_dof:.3f}")
    print(f"  (compare to QNM default: chi2/dof = 12.24)")
    print(f"  (compare to LCDM default: chi2/dof = 7.52)")
    print(f"  (compare to Planck best-fit: chi2/dof = 0.80)")
    
    # LCDM comparison
    print(f"\nLCDM best-fit: omega_b=0.02237, H0=0.6736")
    print(f"QNM best-fit:  omega_b={best_ob:.5f}, h={best_h0:.4f}")
    if abs(best_ob - 0.02237) / 0.02237 > 0.02:
        print(f"  *** omega_b DIFFERS by {(best_ob-0.02237)/0.02237*100:.1f}% ***")
        print(f"  This is a TESTABLE PREDICTION via BBN deuterium abundance")
    if abs(best_h0 - 0.6736) / 0.6736 > 0.01:
        print(f"  *** H0 DIFFERS by {(best_h0-0.6736)/0.6736*100:.1f}% ***")
        print(f"  This is testable with Hubble tension measurements")
    
    # Save results
    np.savetxt('omega_b_H0_chi2_grid.txt', results,
               header=f'Rows: omega_b = {OMEGA_B_VALUES}\nCols: H0 = {H0_VALUES}\nValues: chi2')
    
    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Chi2 heatmap
        chi2_dof_grid = results / (n_bins - 1)
        im = ax1.imshow(chi2_dof_grid, origin='lower', aspect='auto',
                        extent=[H0_VALUES[0], H0_VALUES[-1], 
                                OMEGA_B_VALUES[0]*1000, OMEGA_B_VALUES[-1]*1000],
                        cmap='RdYlBu_r')
        ax1.set_xlabel('h', fontsize=12)
        ax1.set_ylabel('ω_b × 1000', fontsize=12)
        ax1.set_title('QNM χ²/dof vs Cosmological Parameters', fontsize=13)
        plt.colorbar(im, ax=ax1, label='χ²/dof')
        ax1.plot(best_h0, best_ob*1000, 'w*', ms=15, mew=2, label=f'Best: ({best_h0:.1f}, {best_ob:.5f})')
        ax1.plot(0.6736, 0.02237*1000, 'k+', ms=15, mew=2, label='ΛCDM best-fit')
        ax1.legend(fontsize=9)
        
        # Chi2 slices
        ax2.plot(OMEGA_B_VALUES, chi2_dof_grid[:, np.argmin(np.abs(np.array(H0_VALUES)-best_h0))],
                'b-o', lw=2, label=f'h={best_h0:.4f} (best)')
        ax2.plot(OMEGA_B_VALUES, chi2_dof_grid[:, np.argmin(np.abs(np.array(H0_VALUES)-0.6736))],
                'r--s', lw=1.5, label='H₀=0.6736 (ΛCDM)')
        ax2.axhline(0.80, color='green', ls=':', lw=1, label='Planck best-fit')
        ax2.axhline(7.52, color='gray', ls=':', lw=1, label='ΛCDM default')
        ax2.set_xlabel('ω_b', fontsize=12)
        ax2.set_ylabel('χ²/dof', fontsize=12)
        ax2.set_title('χ²/dof vs ω_b', fontsize=13)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('omega_b_H0_scan.png', dpi=150)
        print("\nSaved: omega_b_H0_scan.png")
    except Exception as e:
        print(f"\nPlot failed: {e}")
    
    # Restore original ini
    create_ini(0.02237, 0.6736, INI_TEMPLATE)
    # Reset to our best QNM params
    with open(INI_TEMPLATE, 'r') as f:
        content = f.read()
    content = re.sub(r'qnm_A\s*=\s*[\d.eE+-]+', f'qnm_A = {QNM_A}', content)
    with open(INI_TEMPLATE, 'w') as f:
        f.write(content)
    print("Restored original ini file.")
