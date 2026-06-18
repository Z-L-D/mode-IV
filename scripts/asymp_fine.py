import subprocess, numpy as np, os, glob, re, shutil
from scipy.interpolate import interp1d
from numpy.polynomial import polynomial as P

CLASS_DIR = os.path.expanduser("~/qnm-boltzmann/class_qnm")
WORK_DIR = os.path.expanduser("~/qnm-boltzmann")
T_CMB = 2.7255e6
PRIM_C = os.path.join(CLASS_DIR, "source/primordial.c")

planck = {}
for spec, col, pfile in [("TT",1,"planck_tt_binned.txt"),
                          ("EE",2,"planck_ee_binned.txt"),
                          ("TE",3,"planck_te_binned.txt")]:
    p = np.loadtxt(os.path.join(WORK_DIR, pfile))
    m = p[:,0] >= 30
    planck[spec] = (p[m,0], p[m,1], (p[m,2]+p[m,3])/2, col)

shutil.copy2(PRIM_C, PRIM_C + ".backup_fine")

def run_with_denom(D):
    with open(PRIM_C, 'r') as f:
        code = f.read()
    code_new = re.sub(
        r'\(2\.0\*n \+ 1\.0\) / [\d.]+',
        f'(2.0*n + 1.0) / {D:.4f}',
        code
    )
    with open(PRIM_C, 'w') as f:
        f.write(code_new)
    
    r = subprocess.run(["make", "class"], cwd=CLASS_DIR, 
                       capture_output=True, timeout=120)
    if r.returncode != 0: return None
    
    r = subprocess.run(["./class", "qnm_test.ini"], cwd=CLASS_DIR,
                       capture_output=True, timeout=120)
    if r.returncode != 0: return None
    
    cl_files = glob.glob(os.path.join(CLASS_DIR, "output", "*_cl_lensed.dat"))
    if not cl_files: return None
    cl = np.loadtxt(max(cl_files, key=os.path.getmtime))
    cl_ell = cl[:,0]
    
    results = {}
    for spec, (ell, Dl, sig, col) in planck.items():
        v = (ell >= cl_ell.min()) & (ell <= cl_ell.max())
        f = interp1d(cl_ell, cl[:,col]*T_CMB**2, kind='cubic')
        resid = (Dl[v] - f(ell[v])) / sig[v]
        c2 = np.sum(resid**2) / (np.sum(v)-1)
        results[spec] = c2
        
        if spec == "TT":
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

print("Fine asymptotic denominator scan (D = 8.0 to 8.5)")
print(f"{'D':>6s} | {'TT':>7s} {'EE':>7s} {'TE':>7s} {'SUM':>7s} | {'tilt':>7s} {'z':>7s}")
print("-" * 60)

for D in [8.0, 8.05, 8.10, 8.15, 8.20, 8.25, 8.30, 8.35, 8.40, 8.50]:
    r = run_with_denom(D)
    if r is None:
        print(f"{D:6.2f} | FAILED")
        continue
    s = r["TT"] + r["EE"] + r["TE"]
    print(f"{D:6.2f} | {r['TT']:7.3f} {r['EE']:7.3f} {r['TE']:7.3f} {s:7.3f} | {r['tilt']:+7.3f} {r['z']:7.2f}")

# Restore
shutil.copy2(PRIM_C + ".backup_fine", PRIM_C)
subprocess.run(["make", "class"], cwd=CLASS_DIR, capture_output=True, timeout=120)
subprocess.run(["./class", "qnm_test.ini"], cwd=CLASS_DIR, capture_output=True, timeout=120)
print("\nRestored original.")
