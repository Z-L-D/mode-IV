#!/usr/bin/env python3
"""Omega_b / h parameter scan v2 - simplified, robust."""

import subprocess
import numpy as np
import os
import re
import glob
from scipy.interpolate import interp1d

CLASS_DIR = os.path.expanduser("~/qnm-boltzmann/class_qnm")
WORK_DIR = os.path.expanduser("~/qnm-boltzmann")

# Load Planck
planck = np.loadtxt(os.path.join(WORK_DIR, "planck_tt_binned.txt"))
p_ell = planck[:, 0]
p_Dl = planck[:, 1]
p_sig = (planck[:, 2] + planck[:, 3]) / 2
mask = p_ell >= 30
p_ell, p_Dl, p_sig = p_ell[mask], p_Dl[mask], p_sig[mask]
n_bins = len(p_ell)
print(f"Planck: {n_bins} bins")

def get_latest_cl():
    """Find the most recent non-lensed cl file."""
    files = glob.glob(os.path.join(CLASS_DIR, "output", "*_cl.dat"))
    files = [f for f in files if 'lensed' not in f]
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def run_one(omega_b, h_val):
    """Modify ini, run CLASS, compute chi2."""
    ini_path = os.path.join(CLASS_DIR, "qnm_test.ini")
    
    # Read current ini
    with open(ini_path, 'r') as f:
        lines = f.readlines()
    
    # Modify parameters
    new_lines = []
    for line in lines:
        if line.strip().startswith('h ='):
            new_lines.append(f'h = {h_val}\n')
        elif line.strip().startswith('omega_b ='):
            new_lines.append(f'omega_b = {omega_b}\n')
        else:
            new_lines.append(line)
    
    with open(ini_path, 'w') as f:
        f.writelines(new_lines)
    
    # Run CLASS
    result = subprocess.run(
        ["./class", "qnm_test.ini"],
        cwd=CLASS_DIR,
        capture_output=True, text=True, timeout=120
    )
    
    if result.returncode != 0:
        print(f"FAILED: {result.stderr[:100]}")
        return np.inf
    
    # Get output
    cl_file = get_latest_cl()
    if cl_file is None:
        return np.inf
    
    cl = np.loadtxt(cl_file)
    cl_ell = cl[:, 0]
    cl_tt = cl[:, 1] * (2.7255e6)**2
    
    # Interpolate to Planck ells
    valid = (cl_ell >= p_ell.min() - 10) & (cl_ell <= p_ell.max() + 10)
    f_interp = interp1d(cl_ell[valid], cl_tt[valid], kind='cubic')
    model = f_interp(p_ell)
    
    chi2 = np.sum(((p_Dl - model) / p_sig) ** 2)
    return chi2

# Grid
OB_VALS = [0.0215, 0.0220, 0.02237, 0.0226, 0.0230, 0.0235]
H_VALS = [0.650, 0.660, 0.670, 0.6736, 0.680, 0.690]

print(f"Grid: {len(OB_VALS)} x {len(H_VALS)} = {len(OB_VALS)*len(H_VALS)} runs")
print(f"{'':>10s} |" + "".join(f" h={h:.4f}" for h in H_VALS))
print("-" * (12 + 9*len(H_VALS)))

results = {}
count = 0
total = len(OB_VALS) * len(H_VALS)

for ob in OB_VALS:
    row = f"ob={ob:.5f} |"
    for h in H_VALS:
        count += 1
        chi2 = run_one(ob, h)
        chi2_dof = chi2 / (n_bins - 1)
        results[(ob, h)] = chi2
        row += f" {chi2_dof:7.2f}"
        print(f"  [{count}/{total}] ob={ob:.5f} h={h:.4f} chi2/dof={chi2_dof:.3f}")
    print(row)

# Best
best_key = min(results, key=results.get)
best_chi2 = results[best_key]
print(f"\nBEST: omega_b={best_key[0]:.5f}, h={best_key[1]:.4f}")
print(f"  chi2/dof = {best_chi2/(n_bins-1):.3f}")
print(f"  (was 12.24 at defaults)")

# Restore defaults
run_one(0.02237, 0.6736)
print("Restored defaults.")
