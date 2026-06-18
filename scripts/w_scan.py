import subprocess, numpy as np, os, glob
from scipy.interpolate import interp1d
from numpy.polynomial import polynomial as P

CLASS_DIR = os.path.expanduser("~/qnm-boltzmann/class_qnm")
WORK_DIR = os.path.expanduser("~/qnm-boltzmann")
T_CMB = 2.7255e6

planck = {}
for spec, col, pfile in [("TT",1,"planck_tt_binned.txt"),
                          ("EE",2,"planck_ee_binned.txt"),
                          ("TE",3,"planck_te_binned.txt")]:
    p = np.loadtxt(os.path.join(WORK_DIR, pfile))
    m = p[:,0] >= 30
    planck[spec] = (p[m,0], p[m,1], (p[m,2]+p[m,3])/2, col)

def run_w(w0, wa=0.0):
    """Run CLASS with dynamical dark energy w(a) = w0 + wa*(1-a)."""
    ini = os.path.join(CLASS_DIR, "qnm_test.ini")
    with open(ini) as f: lines = f.readlines()
    
    # Remove any existing fluid params
    out = [l for l in lines if not any(l.strip().startswith(p) for p in 
           ['Omega_Lambda', 'w0_fld', 'wa_fld', 'Omega_fld', 
            'use_ppf', 'cs2_fld', 'fluid_equation_of_state'])]
    
    # Remove trailing newlines then add fluid params
    while out and out[-1].strip() == '':
        out.pop()
    
    if abs(w0 - (-1.0)) < 1e-6 and abs(wa) < 1e-6:
        # Standard cosmological constant - don't add fluid params
        pass
    else:
        # Add dark energy fluid
        out.append(f'\n# Dark energy fluid\n')
        out.append(f'Omega_fld = 0.6847\n')  # Same as Omega_Lambda default
        out.append(f'w0_fld = {w0}\n')
        out.append(f'wa_fld = {wa}\n')
        out.append(f'cs2_fld = 1.0\n')
        # Need to remove Omega_Lambda if it exists implicitly
        # CLASS computes it from flatness by default
    
    with open(ini,'w') as f: f.writelines(out)
    
    r = subprocess.run(["./class","qnm_test.ini"], cwd=CLASS_DIR, 
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f"  CLASS failed: {r.stderr[:150]}")
        return None
    
    cl_files = glob.glob(os.path.join(CLASS_DIR,"output","*_cl_lensed.dat"))
    if not cl_files:
        print("  No lensed output")
        return None
    cl_file = max(cl_files, key=os.path.getmtime)
    cl = np.loadtxt(cl_file)
    cl_ell = cl[:,0]
    
    results = {}
    for spec, (ell, Dl, sig, col) in planck.items():
        v = (ell >= cl_ell.min()) & (ell <= cl_ell.max())
        f = interp1d(cl_ell, cl[:,col]*T_CMB**2, kind='cubic')
        resid = (Dl[v] - f(ell[v])) / sig[v]
        c2 = np.sum(resid**2) / (np.sum(v)-1)
        results[spec] = c2
        
        if spec == "TT":
            # Tilt and runs test
            coeffs = P.polyfit(ell[v], resid, 1)
            results['tilt'] = coeffs[1] * 1000
            
            signs = resid > 0
            runs = 1 + np.sum(signs[1:] != signs[:-1])
            n = len(signs)
            n_pos, n_neg = np.sum(signs), np.sum(~signs)
            exp_r = 1 + 2*n_pos*n_neg/n
            std_r = np.sqrt(2*n_pos*n_neg*(2*n_pos*n_neg-n)/(n**2*(n-1)))
            results['z'] = (runs - exp_r)/std_r if std_r > 0 else 0
    
    return results

# First: scan constant w
print("="*65)
print("CONSTANT w SCAN (wa=0)")
print("="*65)
print(f"{'w0':>7s} | {'TT':>7s} {'EE':>7s} {'TE':>7s} {'SUM':>7s} | {'tilt':>7s} {'z':>7s}")
print("-"*60)

for w0 in [-0.90, -0.95, -1.00, -1.05, -1.10, -1.15, -1.20]:
    r = run_w(w0)
    if r is None:
        print(f"{w0:7.2f} | FAILED")
        continue
    s = r["TT"] + r["EE"] + r["TE"]
    print(f"{w0:7.2f} | {r['TT']:7.3f} {r['EE']:7.3f} {r['TE']:7.3f} {s:7.3f} | {r['tilt']:+7.3f} {r['z']:7.2f}")

# Then: try CPL approximation of the phase-structured values
# Phase-structured: w(z<0.5)=-1.10, w(0.5<z<8)=-1.354, w(z>8)=-0.513
# CPL: w(a) = w0 + wa*(1-a), at z=0: w=w0, at z→∞: w=w0+wa
# Rough fit: w0 ≈ -1.10, wa ≈ 0.6 (so at high z: w ≈ -0.50)
print(f"\n{'='*65}")
print("CPL PARAMETERIZATION (w = w0 + wa*(1-a))")
print("="*65)
print(f"{'w0':>7s} {'wa':>7s} | {'TT':>7s} {'EE':>7s} {'TE':>7s} {'SUM':>7s} | {'tilt':>7s} {'z':>7s}")
print("-"*65)

for w0, wa in [(-1.00, 0.0), (-1.10, 0.0), (-1.10, 0.3), (-1.10, 0.6), 
               (-1.05, 0.3), (-1.05, 0.5), (-1.15, 0.3), (-1.15, 0.6),
               (-1.20, 0.5), (-1.20, 0.7)]:
    r = run_w(w0, wa)
    if r is None:
        print(f"{w0:7.2f} {wa:7.2f} | FAILED")
        continue
    s = r["TT"] + r["EE"] + r["TE"]
    print(f"{w0:7.2f} {wa:7.2f} | {r['TT']:7.3f} {r['EE']:7.3f} {r['TE']:7.3f} {s:7.3f} | {r['tilt']:+7.3f} {r['z']:7.2f}")

# Restore standard LCDM
ini = os.path.join(CLASS_DIR, "qnm_test.ini")
with open(ini) as f: content = f.read()
# Remove fluid params
import re
content = re.sub(r'\n# Dark energy fluid\n.*?(?=\n[a-zA-Z]|\Z)', '', content, flags=re.DOTALL)
content = re.sub(r'Omega_fld.*?\n', '', content)
content = re.sub(r'w0_fld.*?\n', '', content)
content = re.sub(r'wa_fld.*?\n', '', content)
content = re.sub(r'cs2_fld.*?\n', '', content)
with open(ini, 'w') as f: f.write(content)
print("\nRestored standard LCDM ini.")
