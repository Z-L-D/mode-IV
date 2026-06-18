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

def run_one(tau, A_qnm):
    ini = os.path.join(CLASS_DIR, "qnm_test.ini")
    with open(ini) as f: lines = f.readlines()
    out = []
    for l in lines:
        if l.strip().startswith('tau_reio ='): out.append(f'tau_reio = {tau}\n')
        elif l.strip().startswith('qnm_A ='): out.append(f'qnm_A = {A_qnm}\n')
        else: out.append(l)
    with open(ini,'w') as f: f.writelines(out)
    
    r = subprocess.run(["./class","qnm_test.ini"], cwd=CLASS_DIR, 
                       capture_output=True, timeout=120)
    if r.returncode != 0: return {s: np.inf for s in ["TT","EE","TE"]}
    
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

# Joint scan: tau × amplitude
TAU = [0.046, 0.047, 0.048, 0.049, 0.050, 0.051, 0.052]
AMP = [2.80e-8, 2.82e-8, 2.85e-8, 2.88e-8, 2.90e-8, 2.92e-8, 2.95e-8]

print(f"Joint tau x A scan: {len(TAU)}x{len(AMP)}={len(TAU)*len(AMP)} runs (LENSED)")
print(f"{'tau':>7s} {'A':>9s} | {'TT':>7s} {'EE':>7s} {'TE':>7s} {'SUM':>7s}")
print("-" * 55)

best_sum = 999
best_params = None

for tau in TAU:
    for A in AMP:
        r = run_one(tau, A)
        s = r["TT"] + r["EE"] + r["TE"]
        if s < best_sum:
            best_sum = s
            best_params = (tau, A)
            marker = " ***"
        else:
            marker = ""
        print(f"{tau:7.4f} {A:9.2e} | {r['TT']:7.3f} {r['EE']:7.3f} {r['TE']:7.3f} {s:7.3f}{marker}")

print(f"\nBEST: tau={best_params[0]:.4f}, A={best_params[1]:.2e}")
print(f"  Combined = {best_sum:.3f}")

# Run best point and get detailed breakdown
r = run_one(best_params[0], best_params[1])
print(f"\n  TT = {r['TT']:.3f}")
print(f"  EE = {r['EE']:.3f}")
print(f"  TE = {r['TE']:.3f}")

# Comparison
print(f"\n  vs LCDM:  TT=0.794  EE=0.968  TE=1.119")
print(f"  Ratios:   TT={r['TT']/0.794:.2f}x  EE={r['EE']/0.968:.2f}x  TE={r['TE']/1.119:.2f}x")

# Restore defaults
run_one(0.0544, 2.85e-8)
print("\nRestored defaults.")
