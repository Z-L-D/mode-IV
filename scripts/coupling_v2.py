#!/usr/bin/env python3
"""
QNM Mode Coupling v2: Fixed normalization + proper cross-terms.
"""
import numpy as np
import matplotlib.pyplot as plt
import glob, os
from scipy.interpolate import interp1d

# Leaver values (2M=1 units)
leaver = [
    (0.74734, 0.08896), (0.69342, 0.27392), (0.60211, 0.47828),
    (0.50302, 0.70514), (0.41583, 0.94685), (0.34615, 1.19567),
    (0.29131, 1.44796), (0.24834, 1.70310),
]

asymp_wr = np.log(3) / (4*np.pi)  # 0.08742
# Nollert smooth approach: ω_R(n) = asymp + A/√(n+1)
A_wr = (leaver[7][0] - asymp_wr) * np.sqrt(8)

k0 = np.pi / 4450.6
N_modes = 300
gamma = 1.04
A_qnm = 2.85e-8
A_s, n_s = 2.1e-9, 0.9649
D_A = 14000

def omega_R(n):
    if n < 8: return leaver[n][0]
    return asymp_wr + A_wr / np.sqrt(n+1)

def omega_I(n):
    if n < 8: return leaver[n][1]
    return (2.0*n + 1.0) / 8.0  # FIXED: /8.0 not /4.0

def pk_coupled(k, eta=0.0):
    """P(k) with interference cross-terms between adjacent modes."""
    w_sum = sum(1.0/(n+1)**gamma for n in range(N_modes))
    
    # Pre-compute all mode quantities
    kn = np.array([k0*(n+1) for n in range(N_modes)])
    wi = np.array([omega_I(n) for n in range(N_modes)])
    wr = np.array([omega_R(n) for n in range(N_modes)])
    sigma = wi * k0 / (2*np.pi)
    w = np.array([(1.0/(n+1)**gamma)/w_sum for n in range(N_modes)])
    
    dk = k - kn
    L = sigma**2 / (dk**2 + sigma**2)
    asym = (k + kn)**2 / kn**2
    
    # Standard mode sum (diagonal terms)
    mode_sum = np.sum(w * L * asym)
    
    # Cross-terms: interference between adjacent modes
    if eta != 0:
        cross = 0.0
        for n in range(N_modes - 1):
            dwr = wr[n] - wr[n+1]
            if abs(dwr) < 1e-12:
                continue
            
            # Phase from ω_R difference
            # Physical: the cross-spectral density oscillates at Δω_R
            # In k-space, this maps to oscillation with period ~2π/(Δω_R/k0)
            phase = dwr * (k / k0) * 2 * np.pi
            
            # Cross-term amplitude: product of adjacent mode amplitudes
            amp = np.sqrt(w[n]*w[n+1]) * np.sqrt(L[n]*asym[n]) * np.sqrt(L[n+1]*asym[n+1])
            
            cross += amp * np.cos(phase)
        
        mode_sum += eta * cross
    
    return max(A_qnm * (k/0.05)**0 * mode_sum, 1e-30)

def pk_lcdm(k):
    return A_s * (k/0.05)**(n_s-1)

# ============================================================
# Verify normalization is correct
# ============================================================

k_test = np.array([220, 540, 810, 1500, 2000, 2500]) / D_A
print("Normalization check (η=0, should match pk_diagnostic.py):")
for ell, kt in zip([220, 540, 810, 1500, 2000, 2500], k_test):
    ratio = pk_coupled(kt, 0) / pk_lcdm(kt)
    print(f"  ℓ={ell}: P_QNM/P_LCDM = {ratio:.4f}")

# ============================================================
# Scan coupling strength
# ============================================================

k_vals = np.logspace(np.log10(k0*10), np.log10(k0*280), 2000)
ell_vals = k_vals * D_A

print(f"\nCoupling strength scan:")
print(f"{'η':>6s}  {'ℓ=220':>8s} {'ℓ=540':>8s} {'ℓ=810':>8s} {'ℓ=1500':>8s} {'ℓ=1750':>8s} {'ℓ=2000':>8s} {'ℓ=2500':>8s}")
print("-" * 75)

fig, axes = plt.subplots(3, 1, figsize=(14, 14))

best_eta = 0
best_score = 999

for eta in [0.0, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]:
    pk = np.array([pk_coupled(k, eta) for k in k_vals])
    ratio = pk / np.array([pk_lcdm(k) for k in k_vals])
    
    vals = []
    for ell_t in [220, 540, 810, 1500, 1750, 2000, 2500]:
        idx = np.argmin(np.abs(ell_vals - ell_t))
        vals.append(ratio[idx])
    
    # Score: want ratios close to 1.0 everywhere
    peak_dev = max(abs(v-1) for v in vals[:3])
    high_dev = np.mean([abs(v-1) for v in vals[3:]])
    score = peak_dev + high_dev
    
    if score < best_score and peak_dev < 0.03:
        best_score = score
        best_eta = eta
    
    vals_str = " ".join(f"{v:8.4f}" for v in vals)
    print(f"{eta:6.1f}  {vals_str}")
    
    if eta in [0, 1, 5, 10, 20]:
        axes[0].semilogx(k_vals, ratio, '-', lw=1.5, label=f'η={eta}', alpha=0.7)

print(f"\nBest η = {best_eta} (score = {best_score:.4f})")

axes[0].axhline(1.0, color='k', ls='--', lw=0.5)
axes[0].set_ylabel('P_QNM / P_LCDM')
axes[0].set_title('Mode Coupling Effect on P(k)')
axes[0].legend(fontsize=9)
axes[0].set_ylim(0.90, 1.15)
axes[0].grid(True, alpha=0.3)

# Detail of coupling region
for eta in [0, best_eta]:
    pk = np.array([pk_coupled(k, eta) for k in k_vals])
    ratio = pk / np.array([pk_lcdm(k) for k in k_vals])
    detail = (ell_vals > 1200) & (ell_vals < 2600)
    axes[1].plot(ell_vals[detail], ratio[detail], '-', lw=2, 
                 label=f'η={eta}', alpha=0.8)

axes[1].axhline(1.0, color='k', ls='--', lw=0.5)
axes[1].set_xlabel('ℓ')
axes[1].set_ylabel('P_QNM / P_LCDM')
axes[1].set_title(f'Coupling Region Detail (best η={best_eta})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Compare coupling oscillation structure to Planck residuals
T_CMB = 2.7255e6
planck = np.loadtxt("planck_tt_binned.txt")
p_ell = planck[:,0]; p_Dl = planck[:,1]
p_sig = (planck[:,2]+planck[:,3])/2; p_bf = planck[:,4]

cl_files = [f for f in glob.glob("class_qnm/output/*_cl.dat") if 'lensed' not in f]
latest = max(cl_files, key=os.path.getmtime)
cl = np.loadtxt(latest)
f_qnm = interp1d(cl[:,0], cl[:,1]*T_CMB**2, kind='cubic')

mask = p_ell >= 30
ell_p = p_ell[mask]; Dl_p = p_Dl[mask]; sig_p = p_sig[mask]; bf_p = p_bf[mask]
resid_qnm = (Dl_p - f_qnm(ell_p)) / sig_p
resid_lcdm = (Dl_p - bf_p) / sig_p

hi = ell_p > 800
ax = axes[2]
ax.plot(ell_p[hi], resid_qnm[hi], 'b.-', ms=5, lw=0.5, alpha=0.5, label='QNM residual')
ax.plot(ell_p[hi], resid_lcdm[hi], 'r.-', ms=5, lw=0.5, alpha=0.5, label='ΛCDM residual')
# Smooth
w = 3
ax.plot(ell_p[hi], np.convolve(resid_qnm[hi], np.ones(w)/w, 'same'), 'b-', lw=2.5)
ax.plot(ell_p[hi], np.convolve(resid_lcdm[hi], np.ones(w)/w, 'same'), 'r-', lw=2.5)
ax.axhline(0, color='k', lw=0.5)

# Overlay: where does coupling ADD power? (ratio > 1 with coupling vs without)
if best_eta > 0:
    pk_no = np.array([pk_coupled(k, 0) for k in k_vals])
    pk_yes = np.array([pk_coupled(k, best_eta) for k in k_vals])
    boost = pk_yes / pk_no
    # Scale and overlay on residual plot
    detail2 = (ell_vals > 800) & (ell_vals < 2600)
    boost_scaled = (boost[detail2] - 1) * 50  # scale for visibility
    ax.plot(ell_vals[detail2], boost_scaled, 'g-', lw=1.5, alpha=0.6,
            label=f'Coupling boost (η={best_eta}, scaled)')

ax.set_xlabel('ℓ')
ax.set_ylabel('Residual (σ)')
ax.set_title('Residuals + Coupling Boost Prediction')
ax.legend(fontsize=9)
ax.set_xlim(800, 2500)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('coupling_v2.png', dpi=150)
print(f"\nSaved: coupling_v2.png")
