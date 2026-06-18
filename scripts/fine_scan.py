#!/usr/bin/env python3
"""
fine_scan.py — Fine grid around the sweet spot
Run from ~/qnm-boltzmann/

Focused on f=0.15-0.35, d=-0.25 to -0.75, A=3.0-3.6e-8, A_L=1.15-1.35
~200 runs ≈ 35 minutes
"""
import numpy as np
import subprocess
import os
import sys
import glob
from scipy.interpolate import interp1d

T_CMB = 2.7255e6

planck_tt = np.loadtxt('planck_tt_binned.txt')
planck_ee = np.loadtxt('planck_ee_binned.txt')
planck_te = np.loadtxt('planck_te_binned.txt')

def chi2_spectrum(cl_file, planck_data, col_idx, min_ell=30):
    try:
        cl = np.loadtxt(cl_file)
    except:
        return 1e10, 0
    cl_ell = cl[:, 0]
    cl_vals = cl[:, col_idx] * T_CMB**2
    interp = interp1d(cl_ell, cl_vals, kind='cubic',
                      bounds_error=False, fill_value=0)
    mask = planck_data[:, 0] >= min_ell
    p = planck_data[mask]
    err_p = 0.5 * (np.abs(p[:, 2]) + np.abs(p[:, 3]))
    resid = (p[:, 1] - interp(p[:, 0])) / err_p
    return np.sum(resid**2), len(resid)

CLASS_DIR = os.path.expanduser("~/qnm-boltzmann/class_qnm")
CLASS_BIN = os.path.join(CLASS_DIR, "class")

def run_class(A, f, d, al, run_id):
    root = f"output/fscan_{run_id:04d}_"
    ini_path = os.path.join(CLASS_DIR, f"fscan_{run_id:04d}.ini")

    ini_text = f"""h = 0.6736
T_cmb = 2.7255
omega_b = 0.02237
omega_cdm = 0.12
tau_reio = 0.0544
output = tCl,pCl,lCl
lensing = yes
l_max_scalars = 2500
qnm_mode = yes
qnm_A = {A:.6e}
qnm_gamma = 1.04
qnm_alpha = 0.0
qnm_beta_geom = 1.0
qnm_N_modes = 300
qnm_rs = 4450.6
qnm_f_grey = {f:.4f}
qnm_phase_delta = {d:.4f}
A_L = {al:.4f}
root = {root}
"""
    with open(ini_path, 'w') as fh:
        fh.write(ini_text)

    result = subprocess.run([CLASS_BIN, ini_path],
                          capture_output=True, timeout=120,
                          cwd=CLASS_DIR)
    if result.returncode != 0:
        return None

    candidates = glob.glob(os.path.join(CLASS_DIR, f"output/fscan_{run_id:04d}_*cl_lensed.dat"))
    if not candidates:
        return None
    cl_file = max(candidates, key=os.path.getmtime)

    tt_chi2, tt_n = chi2_spectrum(cl_file, planck_tt, 1)
    ee_chi2, ee_n = chi2_spectrum(cl_file, planck_ee, 2)
    te_chi2, te_n = chi2_spectrum(cl_file, planck_te, 3)

    # Clean up
    for fc in glob.glob(os.path.join(CLASS_DIR, f"output/fscan_{run_id:04d}_*")):
        os.remove(fc)
    if os.path.exists(ini_path):
        os.remove(ini_path)

    return {
        'tt': tt_chi2 / tt_n, 'ee': ee_chi2 / ee_n, 'te': te_chi2 / te_n,
        'combined': (tt_chi2 + ee_chi2 + te_chi2) / (tt_n + ee_n + te_n),
    }

# Fine grid
f_values = [0.15, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35]
d_values = [-0.25, -0.40, -0.50, -0.60, -0.75]
A_values = [3.05e-8, 3.15e-8, 3.25e-8, 3.30e-8, 3.35e-8, 3.45e-8]
AL_values = [1.20, 1.25, 1.30, 1.35]

# Not all combos — fix A_L at 1.25 and 1.30 for the grid, scan others
# Total: 7 * 5 * 6 * 2 = 420... too many. Let's be smarter.
# Fix d=-0.50 first, scan f × A × A_L
# Then fix best f/A, scan d × A_L

print("=" * 75)
print("PHASE 1: Fix d=-0.50, scan f × A × A_L")
print("=" * 75)

results_p1 = []
run_id = 0
total_p1 = len(f_values) * len(A_values) * len(AL_values)
print(f"Runs: {total_p1} ≈ {total_p1*12/60:.0f} min")
print(f"{'#':>4} {'f':>5} {'d':>6} {'A_L':>5} {'A(e-8)':>7} "
      f"{'TT':>7} {'EE':>7} {'TE':>7} {'comb':>7}")
print("-" * 65)

d_fixed = -0.50
for f in f_values:
    for A in A_values:
        for al in AL_values:
            run_id += 1
            try:
                r = run_class(A, f, d_fixed, al, run_id)
            except:
                r = None
            if r is None:
                print(f"{run_id:4d} {f:5.2f} {d_fixed:6.2f} {al:5.2f} {A*1e8:7.3f}   FAILED")
                continue
            results_p1.append({'f': f, 'd': d_fixed, 'A_L': al, 'A': A, **r})
            flag = " ***" if r['tt'] < 2.0 else " **" if r['tt'] < 2.5 else ""
            print(f"{run_id:4d} {f:5.2f} {d_fixed:6.2f} {al:5.2f} {A*1e8:7.3f} "
                  f"{r['tt']:7.3f} {r['ee']:7.3f} {r['te']:7.3f} "
                  f"{r['combined']:7.3f}{flag}")
            sys.stdout.flush()

# Find best from phase 1
results_p1.sort(key=lambda x: x['combined'])
best = results_p1[0]
print(f"\nBest Phase 1: f={best['f']}, A={best['A']*1e8:.3f}e-8, "
      f"A_L={best['A_L']}, TT={best['tt']:.3f}, comb={best['combined']:.3f}")

# Phase 2: Fix best f and A, scan d × A_L finely
print(f"\n{'=' * 75}")
print(f"PHASE 2: Fix f={best['f']}, A={best['A']*1e8:.3f}e-8, scan d × A_L")
print(f"{'=' * 75}")

d_fine = [-0.20, -0.30, -0.40, -0.50, -0.60, -0.70, -0.80, -1.00]
AL_fine = [1.15, 1.20, 1.25, 1.28, 1.30, 1.33, 1.35, 1.40]

results_p2 = []
total_p2 = len(d_fine) * len(AL_fine)
print(f"Runs: {total_p2} ≈ {total_p2*12/60:.0f} min")
print(f"{'#':>4} {'f':>5} {'d':>6} {'A_L':>5} {'A(e-8)':>7} "
      f"{'TT':>7} {'EE':>7} {'TE':>7} {'comb':>7}")
print("-" * 65)

for d in d_fine:
    for al in AL_fine:
        run_id += 1
        try:
            r = run_class(best['A'], best['f'], d, al, run_id)
        except:
            r = None
        if r is None:
            continue
        results_p2.append({'f': best['f'], 'd': d, 'A_L': al, 'A': best['A'], **r})
        flag = " ***" if r['tt'] < 2.0 else " **" if r['tt'] < 2.5 else ""
        print(f"{run_id:4d} {best['f']:5.2f} {d:6.2f} {al:5.2f} {best['A']*1e8:7.3f} "
              f"{r['tt']:7.3f} {r['ee']:7.3f} {r['te']:7.3f} "
              f"{r['combined']:7.3f}{flag}")
        sys.stdout.flush()

# Final results
all_results = results_p1 + results_p2
all_results.sort(key=lambda x: x['combined'])

print(f"\n{'=' * 75}")
print("TOP 15 OVERALL (by combined)")
print("=" * 75)
print(f"{'rank':>4} {'f':>5} {'delta':>6} {'A_L':>5} {'A(e-8)':>7} "
      f"{'TT':>7} {'EE':>7} {'TE':>7} {'comb':>7}")
print("-" * 65)
for i, r in enumerate(all_results[:15]):
    print(f"{i+1:4d} {r['f']:5.2f} {r['d']:6.2f} {r['A_L']:5.2f} "
          f"{r['A']*1e8:7.3f} {r['tt']:7.3f} {r['ee']:7.3f} "
          f"{r['te']:7.3f} {r['combined']:7.3f}")

all_results.sort(key=lambda x: x['tt'])
print(f"\n{'=' * 75}")
print("TOP 15 OVERALL (by TT)")
print("=" * 75)
print(f"{'rank':>4} {'f':>5} {'delta':>6} {'A_L':>5} {'A(e-8)':>7} "
      f"{'TT':>7} {'EE':>7} {'TE':>7} {'comb':>7}")
print("-" * 65)
for i, r in enumerate(all_results[:15]):
    print(f"{i+1:4d} {r['f']:5.2f} {r['d']:6.2f} {r['A_L']:5.2f} "
          f"{r['A']*1e8:7.3f} {r['tt']:7.3f} {r['ee']:7.3f} "
          f"{r['te']:7.3f} {r['combined']:7.3f}")

np.savez('fine_scan_results.npz', results=all_results)
print(f"\nSaved to fine_scan_results.npz")
print(f"Total: {len(all_results)} successful runs")
