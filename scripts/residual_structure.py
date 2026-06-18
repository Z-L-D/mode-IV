#!/usr/bin/env python3
"""
Analyze STRUCTURE of QNM residuals vs LCDM residuals.
If QNM mode coupling is real, QNM residuals should show:
1. Oscillatory structure at high ell (mode beating)
2. Different power spectrum than LCDM residuals (non-white)
3. Sign change near the critical damping transition
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib.pyplot as plt
import glob, os

T_CMB = 2.7255e6

# Load Planck
planck = np.loadtxt("planck_tt_binned.txt")
p_ell = planck[:,0]
p_Dl = planck[:,1]
p_sig = (planck[:,2] + planck[:,3]) / 2
p_bf = planck[:,4]  # Planck best-fit LCDM

mask = p_ell >= 30
ell = p_ell[mask]
Dl = p_Dl[mask]
sig = p_sig[mask]
bf = p_bf[mask]

# Load QNM (reset to gamma=1.04 for this analysis)
cl_files = [f for f in glob.glob("class_qnm/output/*_cl.dat") if 'lensed' not in f]
latest = max(cl_files, key=os.path.getmtime)
cl = np.loadtxt(latest)
cl_ell = cl[:,0]
cl_tt = cl[:,1] * T_CMB**2
f_qnm = interp1d(cl_ell, cl_tt, kind='cubic')
qnm_at_planck = f_qnm(ell)

# Residuals
resid_qnm = (Dl - qnm_at_planck) / sig  # QNM residuals in sigma
resid_lcdm = (Dl - bf) / sig              # LCDM residuals in sigma

print("RESIDUAL STRUCTURE ANALYSIS")
print("=" * 60)

# 1. Basic statistics in ell bins
print("\n1. Residual statistics by ell range:")
print(f"{'Region':>25s}  {'QNM mean':>9s} {'QNM std':>8s} {'LCDM mean':>10s} {'LCDM std':>9s}")
print("-" * 70)

ranges = [
    ("ℓ 30-500", 30, 500),
    ("ℓ 500-1000", 500, 1000),
    ("ℓ 1000-1500", 1000, 1500),
    ("ℓ 1500-2000", 1500, 2000),
    ("ℓ 2000-2500", 2000, 2500),
]

for name, lo, hi in ranges:
    m = (ell >= lo) & (ell < hi)
    if np.sum(m) < 2:
        continue
    print(f"{name:>25s}  {np.mean(resid_qnm[m]):9.3f} {np.std(resid_qnm[m]):8.3f} "
          f"{np.mean(resid_lcdm[m]):10.3f} {np.std(resid_lcdm[m]):9.3f}")

# 2. Autocorrelation of residuals
# White noise has zero autocorrelation at lag > 0
# Structured residuals have non-zero autocorrelation
print("\n2. Residual autocorrelation (lag 1-5 bins):")
for name, resid in [("QNM", resid_qnm), ("LCDM", resid_lcdm)]:
    print(f"  {name}:", end="")
    for lag in range(1, 6):
        if len(resid) > lag:
            ac = np.corrcoef(resid[:-lag], resid[lag:])[0, 1]
            print(f"  lag{lag}={ac:.3f}", end="")
    print()

# 3. Runs test: do signs alternate randomly or cluster?
print("\n3. Runs test (sign clustering):")
for name, resid in [("QNM", resid_qnm), ("LCDM", resid_lcdm)]:
    signs = resid > 0
    runs = 1 + np.sum(signs[1:] != signs[:-1])
    n_pos = np.sum(signs)
    n_neg = np.sum(~signs)
    n = len(signs)
    # Expected runs for random sequence
    expected_runs = 1 + 2*n_pos*n_neg / n
    std_runs = np.sqrt(2*n_pos*n_neg*(2*n_pos*n_neg - n) / (n**2 * (n-1)))
    z = (runs - expected_runs) / std_runs if std_runs > 0 else 0
    print(f"  {name}: {runs} runs (expected {expected_runs:.1f}, z={z:.2f})")
    if abs(z) > 2:
        print(f"    → SIGNIFICANT clustering (z={z:.2f}, p<0.05)")
    else:
        print(f"    → Consistent with random")

# 4. High-ell oscillation analysis
# Focus on ell 1000-2500 where QNM residuals dominate
print("\n4. High-ell residual oscillation analysis (ℓ > 1000):")
hi_mask = ell > 1000
if np.sum(hi_mask) > 10:
    ell_hi = ell[hi_mask]
    r_qnm_hi = resid_qnm[hi_mask]
    r_lcdm_hi = resid_lcdm[hi_mask]
    
    # Count zero crossings
    qnm_crossings = np.sum(r_qnm_hi[1:] * r_qnm_hi[:-1] < 0)
    lcdm_crossings = np.sum(r_lcdm_hi[1:] * r_lcdm_hi[:-1] < 0)
    n_hi = len(r_qnm_hi)
    expected_crossings = (n_hi - 1) / 2  # for random
    
    print(f"  QNM zero crossings:  {qnm_crossings} (expected {expected_crossings:.0f} for random)")
    print(f"  LCDM zero crossings: {lcdm_crossings} (expected {expected_crossings:.0f} for random)")
    
    if qnm_crossings < expected_crossings * 0.7:
        print(f"  → QNM residuals are CLUSTERED (fewer crossings = coherent structure)")
    elif qnm_crossings > expected_crossings * 1.3:
        print(f"  → QNM residuals OSCILLATE faster than random")

# 5. KEY TEST: Do QNM and LCDM residuals correlate?
# If they do, both models miss the same features → instrumental/foreground
# If they don't, the misses are model-specific → physics content
print("\n5. QNM vs LCDM residual correlation:")
for name, lo, hi in [("All ℓ", 30, 2500), ("ℓ < 1000", 30, 1000), 
                      ("ℓ > 1000", 1000, 2500), ("ℓ 1500-2000", 1500, 2000)]:
    m = (ell >= lo) & (ell < hi)
    if np.sum(m) < 3:
        continue
    r = np.corrcoef(resid_qnm[m], resid_lcdm[m])[0, 1]
    print(f"  {name:>15s}: r = {r:.3f}", end="")
    if r > 0.7:
        print(" → CORRELATED (same features missed by both)")
    elif r < 0.3:
        print(" → UNCORRELATED (different physics content)")
    else:
        print(" → Moderate")

# 6. Difference: what does QNM see that LCDM doesn't?
diff = resid_qnm - resid_lcdm  # positive = QNM misses more, negative = LCDM misses more
print("\n6. QNM - LCDM residual difference (where models diverge):")
for name, lo, hi in ranges:
    m = (ell >= lo) & (ell < hi)
    if np.sum(m) < 2:
        continue
    print(f"  {name:>25s}: mean diff = {np.mean(diff[m]):+.3f}σ", end="")
    if np.mean(diff[m]) > 1:
        print(" → QNM needs more power here")
    elif np.mean(diff[m]) < -1:
        print(" → QNM has too much power here")
    else:
        print(" → Models comparable")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

ax = axes[0]
ax.plot(ell, resid_qnm, 'b.', ms=4, alpha=0.5, label='QNM residuals')
ax.plot(ell, resid_lcdm, 'r.', ms=4, alpha=0.5, label='LCDM residuals')
w = 5
if len(resid_qnm) > w:
    ax.plot(ell, np.convolve(resid_qnm, np.ones(w)/w, 'same'), 'b-', lw=2, alpha=0.7)
    ax.plot(ell, np.convolve(resid_lcdm, np.ones(w)/w, 'same'), 'r-', lw=2, alpha=0.7)
ax.axhline(0, color='k', lw=0.5)
ax.set_ylabel('Residual (σ)')
ax.set_title('QNM vs ΛCDM Residuals')
ax.legend()
ax.set_xlim(0, 2500)

ax = axes[1]
ax.plot(ell, diff, 'k.', ms=4, alpha=0.5)
if len(diff) > w:
    ax.plot(ell, np.convolve(diff, np.ones(w)/w, 'same'), 'k-', lw=2, alpha=0.7)
ax.axhline(0, color='gray', lw=0.5)
ax.set_ylabel('QNM - LCDM residual (σ)')
ax.set_title('Where Models Diverge')
ax.set_xlim(0, 2500)
ax.axvspan(1500, 1800, alpha=0.1, color='red', label='Coupling region?')
ax.legend()

ax = axes[2]
# Scatter plot: QNM vs LCDM residuals
colors = np.where(ell < 1000, 'blue', np.where(ell < 1500, 'green', 'red'))
for c, label in [('blue', 'ℓ<1000'), ('green', 'ℓ 1000-1500'), ('red', 'ℓ>1500')]:
    m = colors == c
    ax.scatter(resid_lcdm[m], resid_qnm[m], c=c, s=15, alpha=0.6, label=label)
ax.plot([-5, 12], [-5, 12], 'k--', lw=0.5, alpha=0.5)
ax.set_xlabel('LCDM residual (σ)')
ax.set_ylabel('QNM residual (σ)')
ax.set_title('Residual Correlation')
ax.legend()
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residual_structure.png', dpi=150)
print(f"\nSaved: residual_structure.png")
