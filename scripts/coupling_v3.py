#!/usr/bin/env python3
"""
QNM Mode Coupling v3: Properly normalized.

Physical model: P(k) = |Σ √w_n × ψ_n(k) × e^{iφ_n}|²

Expanding: P = Σ w_n|ψ_n|² + 2 Σ_{n<m} √(w_n w_m) |ψ_n||ψ_m| cos(Δφ)

The cross/diagonal RATIO at any k is bounded. We parameterize:
  P(k) = P_diagonal(k) × [1 + η × modulation(k)]

where modulation oscillates between -1 and +1, and η controls
what fraction of the maximum coherent interference is realized.

η=0: fully incoherent (current model)
η=1: maximum possible interference at every k
Physical η is probably 0.01-0.10 (partial coherence from collapse)
"""
import numpy as np
import matplotlib.pyplot as plt
import glob, os
from scipy.interpolate import interp1d

leaver = [
    (0.74734, 0.08896), (0.69342, 0.27392), (0.60211, 0.47828),
    (0.50302, 0.70514), (0.41583, 0.94685), (0.34615, 1.19567),
    (0.29131, 1.44796), (0.24834, 1.70310),
]
asymp_wr = np.log(3) / (4*np.pi)
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
    return (2.0*n + 1.0) / 8.0

def pk_v3(k, eta=0.0):
    """
    P(k) with coupling as multiplicative modulation.
    
    Step 1: compute diagonal sum (standard Lorentzians)
    Step 2: compute modulation from adjacent-mode interference
    Step 3: P = diagonal × (1 + η × modulation)
    """
    w_sum = sum(1.0/(n+1)**gamma for n in range(N_modes))
    
    # Per-mode arrays
    kn = np.array([k0*(n+1) for n in range(N_modes)])
    wr = np.array([omega_R(n) for n in range(N_modes)])
    wi = np.array([omega_I(n) for n in range(N_modes)])
    sigma = wi * k0 / (2*np.pi)
    w = np.array([(1.0/(n+1)**gamma)/w_sum for n in range(N_modes)])
    
    dk = k - kn
    L = sigma**2 / (dk**2 + sigma**2)
    asym = (k + kn)**2 / kn**2
    
    # Diagonal sum
    diag = np.sum(w * L * asym)
    
    if eta == 0 or diag < 1e-30:
        return max(A_qnm * diag, 1e-30)
    
    # Modulation from adjacent-mode cross-terms
    # Only include pairs where BOTH modes have significant amplitude at this k
    # (Lorentzian > 0.01 of peak)
    cross_num = 0.0   # numerator (oscillatory)
    cross_den = 0.0   # denominator (maximum possible interference)
    
    for n in range(min(N_modes-1, 100)):
        # Only include if both modes contribute here
        if L[n] < 0.001 and L[n+1] < 0.001:
            continue
        
        dwr = wr[n] - wr[n+1]
        if abs(dwr) < 1e-12:
            continue
        
        # Phase: ω_R difference × position in k-space
        # Use k/k0 as the natural unit
        phase = dwr * k / (k0 * 0.5)  # scale so oscillation visible
        
        # Amplitude of this pair's contribution
        pair_amp = np.sqrt(w[n]*w[n+1] * L[n]*asym[n] * L[n+1]*asym[n+1])
        
        cross_num += pair_amp * np.cos(phase)
        cross_den += pair_amp  # maximum (all constructive)
    
    # Normalized modulation: between -1 and +1
    if cross_den > 1e-30:
        modulation = cross_num / cross_den
    else:
        modulation = 0.0
    
    # Relative coupling strength: cross_den/diag tells us how
    # large the interference CAN be relative to the diagonal
    max_coupling = cross_den / diag
    
    # Final spectrum
    pk = A_qnm * diag * (1.0 + eta * max_coupling * modulation)
    return max(pk, 1e-30)

# ============================================================
# Scan and analysis
# ============================================================

k_vals = np.logspace(np.log10(k0*10), np.log10(k0*280), 3000)
ell_vals = k_vals * D_A
pk_lcdm = np.array([A_s*(k/0.05)**(n_s-1) for k in k_vals])

# First check: what does the modulation look like?
print("Modulation structure analysis:")
mod_vals = []
for k in k_vals:
    w_sum = sum(1.0/(n+1)**gamma for n in range(N_modes))
    kn = np.array([k0*(n+1) for n in range(N_modes)])
    wr_arr = np.array([omega_R(n) for n in range(N_modes)])
    wi_arr = np.array([omega_I(n) for n in range(N_modes)])
    sigma = wi_arr * k0 / (2*np.pi)
    w = np.array([(1.0/(n+1)**gamma)/w_sum for n in range(N_modes)])
    dk = k - kn
    L = sigma**2 / (dk**2 + sigma**2)
    asym = (k + kn)**2 / kn**2
    diag = np.sum(w * L * asym)
    
    cross_num = 0.0
    cross_den = 0.0
    for n in range(min(N_modes-1, 100)):
        if L[n] < 0.001 and L[n+1] < 0.001:
            continue
        dwr = wr_arr[n] - wr_arr[n+1]
        if abs(dwr) < 1e-12:
            continue
        phase = dwr * k / (k0 * 0.5)
        pair_amp = np.sqrt(w[n]*w[n+1] * L[n]*asym[n] * L[n+1]*asym[n+1])
        cross_num += pair_amp * np.cos(phase)
        cross_den += pair_amp
    
    if cross_den > 0:
        mod_vals.append((cross_num/cross_den, cross_den/diag))
    else:
        mod_vals.append((0, 0))

modulation = np.array([m[0] for m in mod_vals])
max_strength = np.array([m[1] for m in mod_vals])

print(f"  Max coupling strength (cross_den/diag): {max_strength.max():.4f}")
print(f"  Typical modulation range: [{modulation.min():.4f}, {modulation.max():.4f}]")
print(f"  At ℓ=1500: strength={max_strength[np.argmin(np.abs(ell_vals-1500))]:.4f}")
print(f"  At ℓ=2000: strength={max_strength[np.argmin(np.abs(ell_vals-2000))]:.4f}")

# Scan η
print(f"\nCoupling scan:")
print(f"{'η':>6s}  {'ℓ=220':>7s} {'ℓ=540':>7s} {'ℓ=810':>7s} {'ℓ=1500':>7s} {'ℓ=1750':>7s} {'ℓ=2000':>7s} {'ℓ=2500':>7s}")
print("-" * 70)

fig, axes = plt.subplots(3, 1, figsize=(14, 14))

for eta in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]:
    pk = np.array([pk_v3(k, eta) for k in k_vals])
    ratio = pk / pk_lcdm
    
    vals = []
    for ell_t in [220, 540, 810, 1500, 1750, 2000, 2500]:
        idx = np.argmin(np.abs(ell_vals - ell_t))
        vals.append(ratio[idx])
    vals_str = " ".join(f"{v:7.4f}" for v in vals)
    print(f"{eta:6.2f}  {vals_str}")
    
    if eta in [0, 0.05, 0.10, 0.20, 0.50]:
        axes[0].semilogx(k_vals, ratio, '-', lw=1.5, label=f'η={eta}', alpha=0.7)

axes[0].axhline(1.0, color='k', ls='--', lw=0.5)
axes[0].set_ylabel('P_QNM / P_LCDM')
axes[0].set_title('Coupling Effect (properly normalized)')
axes[0].legend(fontsize=9)
axes[0].set_ylim(0.90, 1.15)
axes[0].grid(True, alpha=0.3)

# Modulation structure
axes[1].plot(ell_vals, modulation, 'b-', lw=0.5, alpha=0.5)
# Smooth
w_sm = 30
axes[1].plot(ell_vals, np.convolve(modulation, np.ones(w_sm)/w_sm, 'same'), 
             'b-', lw=2, label='Modulation pattern')
axes[1].plot(ell_vals, max_strength, 'r-', lw=1.5, label='Max coupling strength')
axes[1].set_xlabel('ℓ')
axes[1].set_ylabel('Modulation / Strength')
axes[1].set_title('Interference Modulation Pattern')
axes[1].legend()
axes[1].set_xlim(0, 2500)
axes[1].grid(True, alpha=0.3)

# Compare to Planck residuals
T_CMB = 2.7255e6
planck = np.loadtxt("planck_tt_binned.txt")
p_ell = planck[:,0]; p_Dl = planck[:,1]
p_sig = (planck[:,2]+planck[:,3])/2; p_bf = planck[:,4]

cl_files = [f for f in glob.glob("class_qnm/output/*_cl.dat") if 'lensed' not in f]
latest = max(cl_files, key=os.path.getmtime)
cl = np.loadtxt(latest)
f_qnm = interp1d(cl[:,0], cl[:,1]*T_CMB**2, kind='cubic')

mask = p_ell >= 30
ell_p = p_ell[mask]
resid_qnm = (p_Dl[mask] - f_qnm(ell_p)) / p_sig[mask]
resid_lcdm = (p_Dl[mask] - p_bf[mask]) / p_sig[mask]

ax = axes[2]
hi = ell_p > 500
ax.plot(ell_p[hi], resid_qnm[hi], 'b.-', ms=4, lw=0.5, alpha=0.4)
# Smooth QNM residual
w_s = 3
ax.plot(ell_p[hi], np.convolve(resid_qnm[hi], np.ones(w_s)/w_s, 'same'), 
        'b-', lw=2.5, label='QNM residual (smoothed)')

# Overlay modulation pattern scaled to residual amplitude
mod_at_planck = np.interp(ell_p[hi], ell_vals, modulation)
strength_at_planck = np.interp(ell_p[hi], ell_vals, max_strength)
# The residual ~ deficit × coupling_correction
# If coupling fills the deficit, modulation should anti-correlate with residual
ax.plot(ell_p[hi], -mod_at_planck * strength_at_planck * 20, 'g-', lw=2, alpha=0.6,
        label='Coupling prediction (inverted, scaled)')

ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('ℓ')
ax.set_ylabel('Residual (σ)')
ax.set_title('QNM Residuals vs Coupling Modulation Pattern')
ax.legend(fontsize=9)
ax.set_xlim(500, 2500)
ax.grid(True, alpha=0.3)

# Correlation between residual and modulation prediction
from scipy.stats import pearsonr
r_corr, p_val = pearsonr(resid_qnm[hi], -mod_at_planck[hi] * strength_at_planck[hi])
print(f"\nCorrelation between QNM residual and coupling prediction:")
print(f"  r = {r_corr:.4f}, p = {p_val:.4f}")
if p_val < 0.05:
    print(f"  SIGNIFICANT at p<0.05")

plt.tight_layout()
plt.savefig('coupling_v3.png', dpi=150)
print(f"\nSaved: coupling_v3.png")
