#!/usr/bin/env python3
"""
joint_scan.py — Scan (f_grey, phase_delta, A_QNM, A_L) jointly
Run from ~/qnm-boltzmann/

Each CLASS run takes ~10s. Coarse grid: ~200 runs ≈ 30 minutes.
"""
import numpy as np
import subprocess
import os
import sys
from scipy.interpolate import interp1d

# ================================================================
# PLANCK DATA
# ================================================================
T_CMB = 2.7255e6  # μK

planck_tt = np.loadtxt('planck_tt_binned.txt')
planck_ee = np.loadtxt('planck_ee_binned.txt')
planck_te = np.loadtxt('planck_te_binned.txt')

def chi2_spectrum(cl_file, planck_data, col_idx, min_ell=30):
    """Compute chi2 of QNM vs Planck for one spectrum."""
    try:
        cl = np.loadtxt(cl_file)
    except:
        return 1e10
    
    cl_ell = cl[:, 0]
    cl_vals = cl[:, col_idx] * T_CMB**2
    
    interp = interp1d(cl_ell, cl_vals, kind='cubic',
                      bounds_error=False, fill_value=0)
    
    mask = planck_data[:, 0] >= min_ell
    p = planck_data[mask]
    ell_p = p[:, 0]
    dl_p = p[:, 1]
    err_p = 0.5 * (np.abs(p[:, 2]) + np.abs(p[:, 3]))
    
    dl_qnm = interp(ell_p)
    resid = (dl_p - dl_qnm) / err_p
    
    return np.sum(resid**2), len(resid)

# ================================================================
# CLASS RUNNER
# ================================================================
CLASS_DIR = os.path.expanduser("~/qnm-boltzmann/class_qnm")
CLASS_BIN = os.path.join(CLASS_DIR, "class")
BASE_INI = os.path.join(CLASS_DIR, "qnm_test.ini")

def run_class(A_qnm, f_grey, phase_delta, A_L, run_id):
    """Run CLASS with given parameters, return chi2 dict."""
    
    root = f"output/scan_{run_id:04d}_"
    ini_path = os.path.join(CLASS_DIR, f"scan_{run_id:04d}.ini")
    
    # Read base ini
    with open(BASE_INI, 'r') as f:
        base = f.read()
    
    # Override parameters
    lines = base.strip().split('\n')
    new_lines = []
    skip_keys = ['qnm_A', 'qnm_f_grey', 'qnm_phase_delta', 'A_L', 'root',
                 'write parameters']
    for line in lines:
        key = line.split('=')[0].strip() if '=' in line else ''
        if key not in skip_keys:
            new_lines.append(line)
    
    new_lines.append(f'qnm_A = {A_qnm:.6e}')
    new_lines.append(f'qnm_f_grey = {f_grey:.4f}')
    new_lines.append(f'qnm_phase_delta = {phase_delta:.4f}')
    if A_L != 1.0:
        new_lines.append(f'A_L = {A_L:.4f}')
    new_lines.append(f'root = {root}')
    
    with open(ini_path, 'w') as f:
        f.write('\n'.join(new_lines) + '\n')
    
    # Run CLASS
    result = subprocess.run([CLASS_BIN, ini_path],
                          capture_output=True, timeout=120,
                          cwd=CLASS_DIR)
    
    if result.returncode != 0:
        return None
    
    # Find lensed output
    cl_file = os.path.join(CLASS_DIR, root + "cl_lensed.dat")
    # CLASS may append numbers — find the actual file
    import glob
    candidates = glob.glob(os.path.join(CLASS_DIR, f"output/scan_{run_id:04d}_*cl_lensed.dat"))
    if not candidates:
        return None
    cl_file = max(candidates, key=os.path.getmtime)
    
    # Compute chi2
    tt_chi2, tt_n = chi2_spectrum(cl_file, planck_tt, 1)
    ee_chi2, ee_n = chi2_spectrum(cl_file, planck_ee, 2)
    te_chi2, te_n = chi2_spectrum(cl_file, planck_te, 3)
    
    # Clean up
    for f_clean in glob.glob(os.path.join(CLASS_DIR, f"output/scan_{run_id:04d}_*")):
        os.remove(f_clean)
    os.remove(ini_path)
    
    return {
        'tt': tt_chi2 / tt_n,
        'ee': ee_chi2 / ee_n,
        'te': te_chi2 / te_n,
        'combined': (tt_chi2 + ee_chi2 + te_chi2) / (tt_n + ee_n + te_n),
        'tt_raw': tt_chi2,
        'ee_raw': ee_chi2,
        'te_raw': te_chi2,
    }

# ================================================================
# PARAMETER GRID
# ================================================================

# Phase 1: Coarse grid
f_values = [0.0, 0.10, 0.20, 0.25, 0.30, 0.35]
d_values = [0.0, -0.25, -0.50, -0.75]
A_L_values = [1.00, 1.15, 1.25, 1.30]

# A_QNM scales with f_grey: more f → less integrated area → need more A
# Approximate: A = 2.85e-8 / (1 - f/2) for the area correction
def estimate_A(f, A_base=2.85e-8):
    return A_base / (1.0 - f * 0.45)

# For each (f, d, A_L), scan A around the estimate
A_offsets = [0.95, 1.00, 1.05]

print("=" * 80)
print("JOINT PARAMETER SCAN")
print("=" * 80)
print(f"Grid: {len(f_values)} f × {len(d_values)} d × {len(A_L_values)} A_L "
      f"× {len(A_offsets)} A_scale = {len(f_values)*len(d_values)*len(A_L_values)*len(A_offsets)} runs")
print(f"Estimated time: {len(f_values)*len(d_values)*len(A_L_values)*len(A_offsets)*12/60:.0f} minutes")
print()

# ================================================================
# RUN SCAN
# ================================================================

results = []
run_id = 0
total = len(f_values) * len(d_values) * len(A_L_values) * len(A_offsets)

print(f"{'#':>4} {'f':>5} {'delta':>6} {'A_L':>5} {'A(e-8)':>7} "
      f"{'TT':>7} {'EE':>7} {'TE':>7} {'comb':>7}")
print("-" * 65)

for f in f_values:
    A_est = estimate_A(f)
    for d in d_values:
        for al in A_L_values:
            for a_scale in A_offsets:
                A = A_est * a_scale
                run_id += 1
                
                try:
                    r = run_class(A, f, d, al, run_id)
                except Exception as e:
                    r = None
                
                if r is None:
                    print(f"{run_id:4d} {f:5.2f} {d:6.2f} {al:5.2f} {A*1e8:7.3f}   FAILED")
                    continue
                
                results.append({
                    'f': f, 'd': d, 'A_L': al, 'A': A,
                    **r
                })
                
                flag = " ***" if r['tt'] < 2.0 else " **" if r['tt'] < 2.5 else ""
                print(f"{run_id:4d} {f:5.2f} {d:6.2f} {al:5.2f} {A*1e8:7.3f} "
                      f"{r['tt']:7.3f} {r['ee']:7.3f} {r['te']:7.3f} "
                      f"{r['combined']:7.3f}{flag}")
                
                sys.stdout.flush()

# ================================================================
# FIND BEST
# ================================================================
print(f"\n{'=' * 80}")
print("TOP 10 CONFIGURATIONS (by combined chi2/dof)")
print("=" * 80)

results.sort(key=lambda x: x['combined'])
print(f"{'rank':>4} {'f':>5} {'delta':>6} {'A_L':>5} {'A(e-8)':>7} "
      f"{'TT':>7} {'EE':>7} {'TE':>7} {'comb':>7}")
print("-" * 65)
for i, r in enumerate(results[:10]):
    print(f"{i+1:4d} {r['f']:5.2f} {r['d']:6.2f} {r['A_L']:5.2f} "
          f"{r['A']*1e8:7.3f} {r['tt']:7.3f} {r['ee']:7.3f} "
          f"{r['te']:7.3f} {r['combined']:7.3f}")

print(f"\n{'=' * 80}")
print("TOP 10 CONFIGURATIONS (by TT chi2/dof)")
print("=" * 80)
results.sort(key=lambda x: x['tt'])
print(f"{'rank':>4} {'f':>5} {'delta':>6} {'A_L':>5} {'A(e-8)':>7} "
      f"{'TT':>7} {'EE':>7} {'TE':>7} {'comb':>7}")
print("-" * 65)
for i, r in enumerate(results[:10]):
    print(f"{i+1:4d} {r['f']:5.2f} {r['d']:6.2f} {r['A_L']:5.2f} "
          f"{r['A']*1e8:7.3f} {r['tt']:7.3f} {r['ee']:7.3f} "
          f"{r['te']:7.3f} {r['combined']:7.3f}")

# Save all results
np.savez('joint_scan_results.npz',
         results=results,
         f_values=f_values,
         d_values=d_values,
         A_L_values=A_L_values)

print(f"\nResults saved to joint_scan_results.npz")
print(f"Total runs: {len(results)} successful out of {run_id} attempted")
