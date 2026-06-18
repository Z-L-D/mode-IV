import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import glob, os

T_CMB = 2.7255e6

# Load LENSED QNM
cl = np.loadtxt(max([f for f in glob.glob("class_qnm/output/*_cl_lensed.dat")], key=os.path.getmtime))
cl_ell, cl_tt = cl[:,0], cl[:,1]*T_CMB**2

# Load Planck
planck = np.loadtxt("planck_tt_binned.txt")
mask = planck[:,0] >= 30
ell = planck[mask,0]; Dl = planck[mask,1]
sig = (planck[mask,2]+planck[mask,3])/2; bf = planck[mask,4]

valid = (ell >= cl_ell.min()) & (ell <= cl_ell.max())
ell, Dl, sig, bf = ell[valid], Dl[valid], sig[valid], bf[valid]

f = interp1d(cl_ell, cl_tt, kind='cubic')
resid_qnm = (Dl - f(ell)) / sig
resid_lcdm = (Dl - bf) / sig

print("RESIDUAL STRUCTURE (LENSED QNM)")
print("=" * 55)

# Runs test
signs = resid_qnm > 0
runs = 1 + np.sum(signs[1:] != signs[:-1])
n_pos, n_neg = np.sum(signs), np.sum(~signs)
n = len(signs)
exp_runs = 1 + 2*n_pos*n_neg/n
std_runs = np.sqrt(2*n_pos*n_neg*(2*n_pos*n_neg-n)/(n**2*(n-1)))
z = (runs - exp_runs)/std_runs if std_runs > 0 else 0

print(f"\nRuns test:")
print(f"  QNM: {runs} runs (expected {exp_runs:.1f}, z={z:.2f})")
signs_l = resid_lcdm > 0
runs_l = 1 + np.sum(signs_l[1:] != signs_l[:-1])
exp_l = 1 + 2*np.sum(signs_l)*np.sum(~signs_l)/n
std_l = np.sqrt(2*np.sum(signs_l)*np.sum(~signs_l)*(2*np.sum(signs_l)*np.sum(~signs_l)-n)/(n**2*(n-1)))
z_l = (runs_l - exp_l)/std_l if std_l > 0 else 0
print(f"  LCDM: {runs_l} runs (expected {exp_l:.1f}, z={z_l:.2f})")

# Autocorrelation
print(f"\nAutocorrelation:")
for name, r in [("QNM", resid_qnm), ("LCDM", resid_lcdm)]:
    ac = [np.corrcoef(r[:-lag], r[lag:])[0,1] for lag in range(1,6)]
    print(f"  {name}: " + "  ".join(f"lag{i+1}={a:.3f}" for i,a in enumerate(ac)))

# Zero crossings at high ell
print(f"\nHigh-ell zero crossings (ell>1000):")
hi = ell > 1000
for name, r in [("QNM", resid_qnm[hi]), ("LCDM", resid_lcdm[hi])]:
    xings = np.sum(r[1:]*r[:-1] < 0)
    exp = (len(r)-1)/2
    print(f"  {name}: {xings} crossings (expected {exp:.0f})")

# QNM-LCDM correlation
print(f"\nQNM-LCDM residual correlation:")
for name, lo, hi_val in [("All", 30, 2500), ("l<1000", 30, 1000), 
                          ("l>1000", 1000, 2500), ("1400-2000", 1400, 2000)]:
    m = (ell >= lo) & (ell < hi_val)
    if np.sum(m) < 3: continue
    r, p = pearsonr(resid_qnm[m], resid_lcdm[m])
    print(f"  {name:>12s}: r={r:.3f}, p={p:.3f}")

# Regional statistics
print(f"\nRegional means:")
for name, lo, hi_val in [("30-500",30,500),("500-1000",500,1000),
                          ("1000-1500",1000,1500),("1500-2500",1500,2500)]:
    m = (ell >= lo) & (ell < hi_val)
    if np.sum(m) < 2: continue
    print(f"  {name:>12s}: QNM={np.mean(resid_qnm[m]):+.2f}σ (std={np.std(resid_qnm[m]):.2f}), LCDM={np.mean(resid_lcdm[m]):+.2f}σ (std={np.std(resid_lcdm[m]):.2f})")

# Was the old z=-4.14 mostly lensing?
print(f"\nCOMPARISON:")
print(f"  Unlensed runs test: z=-4.14 (highly significant clustering)")
print(f"  Lensed runs test:   z={z:.2f}")
if abs(z) < 2:
    print(f"  → Clustering RESOLVED by lensing correction!")
else:
    print(f"  → Clustering persists (real structure)")
