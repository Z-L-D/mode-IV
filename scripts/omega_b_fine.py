import subprocess, numpy as np, os, glob
from scipy.interpolate import interp1d

CLASS_DIR = os.path.expanduser("~/qnm-boltzmann/class_qnm")
T_CMB = 2.7255e6

planck = np.loadtxt("planck_tt_binned.txt")
mask = planck[:,0] >= 30
p_ell, p_Dl, p_sig = planck[mask,0], planck[mask,1], (planck[mask,2]+planck[mask,3])/2
n_bins = len(p_ell)

def run_one(ob, h):
    ini = os.path.join(CLASS_DIR, "qnm_test.ini")
    with open(ini) as f: lines = f.readlines()
    out = []
    for l in lines:
        if l.strip().startswith('h ='): out.append(f'h = {h}\n')
        elif l.strip().startswith('omega_b ='): out.append(f'omega_b = {ob}\n')
        else: out.append(l)
    with open(ini,'w') as f: f.writelines(out)
    subprocess.run(["./class","qnm_test.ini"], cwd=CLASS_DIR, capture_output=True, timeout=120)
    cl_file = max(glob.glob(CLASS_DIR+"/output/*_cl_lensed.dat"), key=os.path.getmtime)
    cl = np.loadtxt(cl_file)
    f = interp1d(cl[:,0], cl[:,1]*T_CMB**2, kind='cubic')
    return np.sum(((p_Dl - f(p_ell))/p_sig)**2)

# Fine grid along the valley
OB = np.arange(0.0224, 0.0230, 0.0002)
H = np.arange(0.665, 0.676, 0.001)

print(f"Fine scan: {len(OB)}x{len(H)}={len(OB)*len(H)} runs")
print(f"{'':>12s}|" + "".join(f" h={h:.3f}" for h in H))
print("-"*(14+8*len(H)))

best = (999999, 0, 0)
for ob in OB:
    row = f"ob={ob:.4f}  |"
    for h in H:
        chi2 = run_one(ob, h)
        dof = chi2/(n_bins-1)
        if chi2 < best[0]: best = (chi2, ob, h)
        row += f" {dof:6.3f}"
    print(row)

print(f"\nBEST: omega_b={best[1]:.5f}, h={best[2]:.4f}")
print(f"  chi2/dof = {best[0]/(n_bins-1):.3f}")

# Also compute EE and TE at best point
run_one(best[1], best[2])
cl_file = max(glob.glob(CLASS_DIR+"/output/*_cl_lensed.dat"), key=os.path.getmtime)
cl = np.loadtxt(cl_file)
cl_ell = cl[:,0]

for spec, col, pfile in [("EE",2,"planck_ee_binned.txt"),("TE",3,"planck_te_binned.txt")]:
    p = np.loadtxt(pfile)
    m = p[:,0] >= 30
    ell, Dl, sig = p[m,0], p[m,1], (p[m,2]+p[m,3])/2
    v = (ell >= cl_ell.min()) & (ell <= cl_ell.max())
    f = interp1d(cl_ell, cl[:,col]*T_CMB**2, kind='cubic')
    c2 = np.sum(((Dl[v]-f(ell[v]))/sig[v])**2)
    print(f"  {spec}: chi2/dof = {c2/(np.sum(v)-1):.3f}")

# Restore
run_one(0.02237, 0.6736)
