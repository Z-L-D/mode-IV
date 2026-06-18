#!/usr/bin/env python3
"""Omega_b / h scan using LENSED output."""
import subprocess
import numpy as np
import os
import glob
from scipy.interpolate import interp1d

CLASS_DIR = os.path.expanduser("~/qnm-boltzmann/class_qnm")
WORK_DIR = os.path.expanduser("~/qnm-boltzmann")
T_CMB = 2.7255e6

planck = np.loadtxt(os.path.join(WORK_DIR, "planck_tt_binned.txt"))
p_ell = planck[:,0]; p_Dl = planck[:,1]
p_sig = (planck[:,2]+planck[:,3])/2
mask = p_ell >= 30
p_ell, p_Dl, p_sig = p_ell[mask], p_Dl[mask], p_sig[mask]
n_bins = len(p_ell)

def get_latest_lensed():
    files = glob.glob(os.path.join(CLASS_DIR, "output", "*_cl_lensed.dat"))
    if not files: return None
    return max(files, key=os.path.getmtime)

def run_one(omega_b, h_val):
    ini_path = os.path.join(CLASS_DIR, "qnm_test.ini")
    with open(ini_path, 'r') as f:
        lines = f.readlines()
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
    
    result = subprocess.run(["./class", "qnm_test.ini"],
        cwd=CLASS_DIR, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        return np.inf
    
    cl_file = get_latest_lensed()
    if cl_file is None:
        return np.inf
    
    cl = np.loadtxt(cl_file)
    cl_ell = cl[:,0]
    cl_tt = cl[:,1] * T_CMB**2
    
    valid = (cl_ell >= p_ell.min()-10) & (cl_ell <= p_ell.max()+10)
    f_interp = interp1d(cl_ell[valid], cl_tt[valid], kind='cubic')
    model = f_interp(p_ell)
    return np.sum(((p_Dl - model) / p_sig)**2)

# Finer grid centered near defaults
OB = [0.0218, 0.0220, 0.0222, 0.02237, 0.0226, 0.0228, 0.0230, 0.0232]
H = [0.650, 0.655, 0.660, 0.665, 0.670, 0.6736, 0.680, 0.685]

print(f"LENSED Omega_b/h scan: {len(OB)}x{len(H)}={len(OB)*len(H)} runs")
print(f"{'':>12s}|" + "".join(f"  h={h:.4f}" for h in H))
print("-" * (14 + 10*len(H)))

best_chi2 = 999999
best_params = (0,0)
count = 0

for ob in OB:
    row = f"ob={ob:.5f} |"
    for h in H:
        count += 1
        chi2 = run_one(ob, h)
        chi2_dof = chi2 / (n_bins - 1)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_params = (ob, h)
        row += f"  {chi2_dof:7.3f}"
        print(f"  [{count}/{len(OB)*len(H)}] ob={ob:.5f} h={h:.4f} chi2/dof={chi2_dof:.3f}")
    print(row)

print(f"\nBEST: omega_b={best_params[0]:.5f}, h={best_params[1]:.4f}")
print(f"  chi2 = {best_chi2:.1f}")
print(f"  chi2/dof = {best_chi2/(n_bins-1):.3f}")
print(f"  (was 4.217 at defaults with lensed)")
print(f"  (LCDM best-fit: 0.794)")

# Restore defaults
run_one(0.02237, 0.6736)
print("Restored defaults.")
