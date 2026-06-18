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

def run_one(tau, omega_b, h):
    ini = os.path.join(CLASS_DIR, "qnm_test.ini")
    with open(ini) as f: lines = f.readlines()
    out = []
    for l in lines:
        if l.strip().startswith('tau_reio ='): out.append(f'tau_reio = {tau}\n')
        elif l.strip().startswith('omega_b ='): out.append(f'omega_b = {omega_b}\n')
        elif l.strip().startswith('h ='): out.append(f'h = {h}\n')
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

# Focused 3-param grid along the degeneracy direction
# τ controls amplitude, ω_b controls peak ratio, h controls peak positions
configs = [
    # (tau, omega_b, h) - default LCDM values and variations
    (0.0544, 0.02237, 0.6736),  # LCDM default
    (0.0490, 0.02237, 0.6736),  # τ optimized only
    (0.0544, 0.02260, 0.6700),  # ω_b/h optimized only
    (0.0490, 0.02260, 0.6700),  # both optimized
    # Fine variations around combined optimum
    (0.0500, 0.02237, 0.6700),
    (0.0500, 0.02250, 0.6700),
    (0.0500, 0.02260, 0.6700),
    (0.0510, 0.02237, 0.6700),
    (0.0510, 0.02250, 0.6700),
    (0.0510, 0.02260, 0.6700),
    (0.0520, 0.02237, 0.6700),
    (0.0520, 0.02250, 0.6700),
    (0.0520, 0.02260, 0.6700),
    (0.0530, 0.02237, 0.6700),
    (0.0530, 0.02250, 0.6700),
    (0.0530, 0.02260, 0.6700),
    (0.0540, 0.02237, 0.6700),
    (0.0540, 0.02250, 0.6700),
    (0.0540, 0.02260, 0.6700),
    # Try with h=0.6736 (LCDM default)
    (0.0500, 0.02237, 0.6736),
    (0.0510, 0.02237, 0.6736),
    (0.0520, 0.02237, 0.6736),
    (0.0530, 0.02237, 0.6736),
    # Try h=0.6650 (lower)
    (0.0520, 0.02237, 0.6650),
    (0.0530, 0.02237, 0.6650),
    (0.0540, 0.02237, 0.6650),
    (0.0544, 0.02237, 0.6650),
]

print(f"Joint 3-param scan: {len(configs)} points (LENSED)")
print(f"{'tau':>7s} {'ob':>8s} {'h':>7s} | {'TT':>7s} {'EE':>7s} {'TE':>7s} {'SUM':>7s}")
print("-" * 60)

best_sum = 999
best_cfg = None

for tau, ob, h in configs:
    r = run_one(tau, ob, h)
    s = r["TT"] + r["EE"] + r["TE"]
    if s < best_sum:
        best_sum = s
        best_cfg = (tau, ob, h)
        marker = " ***"
    else:
        marker = ""
    print(f"{tau:7.4f} {ob:8.5f} {h:7.4f} | {r['TT']:7.3f} {r['EE']:7.3f} {r['TE']:7.3f} {s:7.3f}{marker}")

print(f"\nBEST: tau={best_cfg[0]:.4f}, omega_b={best_cfg[1]:.5f}, h={best_cfg[2]:.4f}")
print(f"  Combined = {best_sum:.3f}")
print(f"  LCDM default combined = 6.884")

# Restore
run_one(0.0544, 0.02237, 0.6736)
print("Restored.")
