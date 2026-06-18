#!/usr/bin/env python3
"""
probe_class_impl.py — Inspect your actual CLASS QNM implementation
Run from ~/qnm-boltzmann/

Answers the 5 verification questions by reading your source files
and comparing CLASS output against the standalone P(k).
"""
import os
import re
import glob
import subprocess
import numpy as np

BASE = os.path.expanduser("~/qnm-boltzmann")
CLASS_DIR = os.path.join(BASE, "class_qnm")

print("=" * 65)
print("CLASS QNM IMPLEMENTATION PROBE")
print("=" * 65)

# ================================================================
# Q1: Which Lorentzian form?
# ================================================================
print("\n--- Q1: Lorentzian form ---")
prim_c = os.path.join(CLASS_DIR, "source", "primordial.c")
if os.path.exists(prim_c):
    with open(prim_c, 'r') as f:
        src = f.read()
    
    # Find lines containing lorentz/Lorentz/LORENTZ or the formula
    lines = src.split('\n')
    relevant = []
    for i, line in enumerate(lines):
        low = line.lower()
        if any(x in low for x in ['lorentz', 'dk*dk', 'dk * dk',
                                    'sigma_n', 'sigma*sigma',
                                    'eps*eps', 'eps2', 'fano',
                                    'asymmetr', 'asym_n', 'asym ']):
            relevant.append((i+1, line.rstrip()))
    
    if relevant:
        print(f"  Found {len(relevant)} relevant lines in primordial.c:")
        for lineno, line in relevant[:25]:
            print(f"    L{lineno:4d}: {line}")
    else:
        print("  No obvious Lorentzian formula found — checking broader context...")
        # Look for the QNM function
        for i, line in enumerate(lines):
            if 'qnm' in line.lower() and ('(' in line or 'void' in line or 
                                           'double' in line or 'int' in line):
                # Print surrounding context
                start = max(0, i-2)
                end = min(len(lines), i+30)
                print(f"  QNM function found near L{i+1}:")
                for j in range(start, end):
                    print(f"    L{j+1:4d}: {lines[j].rstrip()}")
                break
else:
    print(f"  ERROR: {prim_c} not found")
    # Try to find it
    for root, dirs, files in os.walk(CLASS_DIR):
        for f in files:
            if f == 'primordial.c':
                print(f"  Found at: {os.path.join(root, f)}")

# ================================================================
# Q2: j₂ envelope?
# ================================================================
print("\n--- Q2: j₂ envelope ---")
if os.path.exists(prim_c):
    has_j2 = any(x in src.lower() for x in ['bessel', 'j2', 'j_2',
                                              'sph_bessel', 'envelope',
                                              'j2_val', 'jl'])
    if has_j2:
        for i, line in enumerate(lines):
            low = line.lower()
            if any(x in low for x in ['bessel', 'j2', 'j_2', 'envelope',
                                       'sph_bessel']):
                print(f"    L{i+1:4d}: {line.rstrip()}")
        print("  >>> j₂ envelope IS present in the code")
    else:
        print("  No j₂/bessel/envelope references found")
        print("  >>> j₂ envelope is NOT present")

# ================================================================
# Q3: beta_geom value
# ================================================================
print("\n--- Q3: beta_geom ---")
ini_files = glob.glob(os.path.join(CLASS_DIR, "*.ini")) + \
            glob.glob(os.path.join(CLASS_DIR, "**/*.ini"), recursive=True)
found_beta = False
for ini_path in ini_files:
    with open(ini_path, 'r') as f:
        for line in f:
            if 'beta' in line.lower() or 'qnm' in line.lower():
                print(f"  {os.path.basename(ini_path)}: {line.rstrip()}")
                found_beta = True
if not found_beta:
    print("  No beta_geom found in .ini files")
    # Check the C source for defaults
    if os.path.exists(prim_c):
        for i, line in enumerate(lines):
            if 'beta' in line.lower():
                print(f"  primordial.c L{i+1}: {line.rstrip()}")

# Also check input.c for parameter reading
input_c = os.path.join(CLASS_DIR, "source", "input.c")
if os.path.exists(input_c):
    with open(input_c, 'r') as f:
        inp_src = f.read()
    inp_lines = inp_src.split('\n')
    for i, line in enumerate(inp_lines):
        if 'beta' in line.lower() and 'qnm' in line.lower():
            print(f"  input.c L{i+1}: {line.rstrip()}")

# ================================================================
# Q4: Asymptotic omega_I formula
# ================================================================
print("\n--- Q4: Asymptotic omega_I ---")
if os.path.exists(prim_c):
    for i, line in enumerate(lines):
        low = line.lower()
        if any(x in low for x in ['omega_i', 'omega_im', 'asymptot',
                                    '2*n+1', '2*n + 1', '(2*n',
                                    'motl', 'neitzke', 'damping']):
            print(f"    L{i+1:4d}: {line.rstrip()}")

# ================================================================
# Q5: Compare P(k) from CLASS output
# ================================================================
print("\n--- Q5: P(k) comparison ---")

# Find CLASS P(k) output files
pk_files = glob.glob(os.path.join(CLASS_DIR, "output", "*pk*")) + \
           glob.glob(os.path.join(CLASS_DIR, "output", "*primordial*")) + \
           glob.glob(os.path.join(CLASS_DIR, "output", "*.dat"))

print(f"  Found {len(pk_files)} output files:")
for f in pk_files[:10]:
    print(f"    {os.path.relpath(f, BASE)}")

# Try to find a file with P(k) data
pk_data = None
pk_file_used = None
for pf in pk_files:
    try:
        # Try reading with various comment chars
        for comment in ['#', '%', '!']:
            try:
                data = np.loadtxt(pf, comments=comment)
                if data.ndim == 2 and data.shape[1] >= 2:
                    # Check if first column looks like k values
                    if data[0, 0] > 0 and data[0, 0] < 1:
                        pk_data = data
                        pk_file_used = pf
                        break
            except:
                continue
        if pk_data is not None:
            break
    except:
        continue

if pk_data is not None:
    print(f"\n  Using: {os.path.relpath(pk_file_used, BASE)}")
    print(f"  Shape: {pk_data.shape}")
    print(f"  k range: [{pk_data[0,0]:.6e}, {pk_data[-1,0]:.6e}]")
    print(f"  Columns: {pk_data.shape[1]}")
    print(f"\n  First 5 rows:")
    for row in pk_data[:5]:
        print(f"    {row}")
    
    # If this is a P(k) file, compare against standalone
    k_class = pk_data[:, 0]
    pk_class = pk_data[:, 1]
    
    # Standalone computation (inline)
    omega_I_leaver = [0.08896, 0.27392, 0.47828, 0.70514,
                      0.94684, 1.19565, 1.44766, 1.70105]
    H0, c_kms = 67.36, 299792.458
    rs = c_kms / H0
    k0 = np.pi / rs
    gamma = 1.04
    N_modes = 300
    w = np.array([1.0/(n+1)**gamma for n in range(N_modes)])
    w /= w.sum()
    kn_arr = np.array([k0*(n+1) for n in range(N_modes)])
    sig_arr = np.array([(omega_I_leaver[n] if n<8 else (2*n+1)/8.0)*k0/(2*np.pi)
                        for n in range(N_modes)])
    def pk_at(k_val):
        pk = 0.0
        for n in range(N_modes):
            dk = k_val - kn_arr[n]
            eps2 = (dk/sig_arr[n])**2
            prof = 1.0/(1.0+eps2)
            asym = (k_val+kn_arr[n])**2 / kn_arr[n]**2
            pk += w[n]*prof*asym
        return 2.85e-8 * pk

    k_test = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    print(f"\n  {'k':>10}  {'P_standalone':>14}  {'P_CLASS':>14}  {'ratio':>10}")
    print(f"  {'-'*52}")
    for kt in k_test:
        idx = np.argmin(np.abs(k_class - kt))
        if abs(k_class[idx] - kt) / kt < 0.1:  # within 10% of target k
            p_stand = pk_at(kt)
            p_class = pk_class[idx]
            ratio = p_class / p_stand if p_stand > 0 else float('nan')
            print(f"  {kt:>10.4f}  {p_stand:>14.6e}  {p_class:>14.6e}  {ratio:>10.6f}")
else:
    print("  No P(k) output file found automatically.")
    print("  Try running CLASS and checking output/ for primordial spectrum files.")

# ================================================================
# Also dump the header of primordial.h for the QNM struct
# ================================================================
print("\n--- BONUS: primordial.h QNM fields ---")
prim_h = os.path.join(CLASS_DIR, "include", "primordial.h")
if os.path.exists(prim_h):
    with open(prim_h, 'r') as f:
        h_lines = f.readlines()
    in_qnm = False
    for i, line in enumerate(h_lines):
        if 'qnm' in line.lower() or 'QNM' in line:
            in_qnm = True
        if in_qnm:
            print(f"    L{i+1:4d}: {line.rstrip()}")
            if '};' in line or (in_qnm and line.strip() == '' and i > 5):
                # Check if we've printed enough context
                pass
        # Also catch the enum
        if 'qnm' in line.lower() and ('enum' in line.lower() or 
                                        'spec_type' in line.lower()):
            start = max(0, i-3)
            end = min(len(h_lines), i+5)
            for j in range(start, end):
                print(f"    L{j+1:4d}: {h_lines[j].rstrip()}")
else:
    print(f"  {prim_h} not found")

# ================================================================
# Check the .ini file for ALL QNM parameters
# ================================================================
print("\n--- QNM .ini parameters ---")
for ini_path in ini_files:
    with open(ini_path, 'r') as f:
        ini_content = f.read()
    if 'qnm' in ini_content.lower():
        print(f"\n  File: {os.path.relpath(ini_path, BASE)}")
        for line in ini_content.split('\n'):
            line_s = line.strip()
            if line_s and not line_s.startswith('#'):
                # Print all non-comment lines that might be QNM params
                if any(x in line.lower() for x in ['qnm', 'gamma', 'alpha',
                                                     'beta', 'amplitude',
                                                     'n_modes', 'rs',
                                                     'cavity']):
                    print(f"    {line_s}")

print("\n" + "=" * 65)
print("Paste this entire output back for analysis.")
print("=" * 65)
