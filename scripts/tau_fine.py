import subprocess, numpy as np, os, glob
from scipy.interpolate import interp1d

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

def run_one(tau):
    ini = os.path.join(CLASS_DIR, "qnm_test.ini")
    with open(ini) as f: lines = f.readlines()
    out = []
    for l in lines:
        if l.strip().startswith('tau_reio ='): out.append(f'tau_reio = {tau}\n')
        else: out.append(l)
    with open(ini,'w') as f: f.writelines(out)
    
    subprocess.run(["./class","qnm_test.ini"], cwd=CLASS_DIR, 
                   capture_output=True, timeout=120)
    
    cl_file = max(glob.glob(os.path.join(CLASS_DIR,"output","*_cl_lensed.dat")), 
                  key=os.path.getmtime)
    cl = np.loadtxt(cl_file)
    cl_ell = cl[:,0]
    
    results = {}
    for spec, (ell, Dl, sig, col) in planck.items():
        v = (ell >= cl_ell.min()) & (ell <= cl_ell.max())
        f = interp1d(cl_ell, cl[:,col]*T_CMB**2, kind='cubic')
        c2 = np.sum(((Dl[v]-f(ell[v]))/sig[v])**2)
        results[spec] = c2 / (np.sum(v)-1)
    return results

TAU = np.arange(0.042, 0.058, 0.002)

print("Fine tau scan (omega_cdm=0.12 fixed, LENSED)")
print(f"{'tau':>7s} | {'TT':>7s} {'EE':>7s} {'TE':>7s} {'SUM':>7s}")
print("-" * 45)

best_sum = 999
best_tau = 0

for tau in TAU:
    r = run_one(tau)
    s = r["TT"] + r["EE"] + r["TE"]
    if s < best_sum:
        best_sum = s
        best_tau = tau
    print(f"{tau:7.4f} | {r['TT']:7.3f} {r['EE']:7.3f} {r['TE']:7.3f} {s:7.3f}")

print(f"\nBEST: tau={best_tau:.4f}, combined={best_sum:.3f}")

# Ultra-fine around best
print(f"\nUltra-fine around tau={best_tau:.4f}:")
for tau in np.arange(best_tau-0.003, best_tau+0.004, 0.001):
    r = run_one(tau)
    s = r["TT"] + r["EE"] + r["TE"]
    marker = " ***" if s < best_sum else ""
    if s < best_sum:
        best_sum = s
        best_tau = tau
    print(f"  {tau:7.4f} | {r['TT']:7.3f} {r['EE']:7.3f} {r['TE']:7.3f} {s:7.3f}{marker}")

print(f"\nFINAL BEST: tau={best_tau:.4f}")
print(f"  TT:  chi2/dof at best tau")
print(f"  Combined: {best_sum:.3f}")
print(f"  Planck constraint: tau = 0.054 ± 0.007")
print(f"  QNM preference: tau = {best_tau:.4f} ({(best_tau-0.054)/0.007:.1f}σ from Planck)")

# Restore
run_one(0.0544)
print("Restored.")
