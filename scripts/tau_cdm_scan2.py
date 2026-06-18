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

def run_one(tau, omega_cdm):
    ini = os.path.join(CLASS_DIR, "qnm_test.ini")
    with open(ini) as f: lines = f.readlines()
    out = []
    for l in lines:
        if l.strip().startswith('tau_reio ='): out.append(f'tau_reio = {tau}\n')
        elif l.strip().startswith('omega_cdm ='): out.append(f'omega_cdm = {omega_cdm}\n')
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

TAU = [0.040, 0.050, 0.0544, 0.060, 0.070]
CDM = [0.10, 0.11, 0.12, 0.13, 0.14]

print("Scanning tau_reio and omega_cdm (LENSED)")
print(f"{'tau':>6s} {'ocdm':>6s} | {'TT':>7s} {'EE':>7s} {'TE':>7s} {'SUM':>7s}")
print("-" * 55)

best_sum = 999
best_params = None

for tau in TAU:
    for ocdm in CDM:
        r = run_one(tau, ocdm)
        s = r["TT"] + r["EE"] + r["TE"]
        if s < best_sum:
            best_sum = s
            best_params = (tau, ocdm)
        print(f"{tau:6.4f} {ocdm:5.2f}  | {r['TT']:7.3f} {r['EE']:7.3f} {r['TE']:7.3f} {s:7.3f}")

print(f"\nBEST: tau={best_params[0]:.4f}, omega_cdm={best_params[1]:.2f}")
print(f"  Combined = {best_sum:.3f}")
print(f"  Default combined = {4.217+1.261+1.406:.3f}")

# Restore
run_one(0.0544, 0.12)
print("Restored.")
