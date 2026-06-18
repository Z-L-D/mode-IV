#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn

H0_kmsMpc = 67.36
c_kmps = 299792.458
H0 = H0_kmsMpc / c_kmps
rs = 1.0 / H0
k0 = np.pi / rs
beta_geom = 1.0
A_QNM = 2.1e-9

omega_R = np.array([0.37367, 0.34671, 0.30105, 0.25150, 0.205, 0.165, 0.130, 0.10932])
omega_I = np.array([0.08896, 0.27392, 0.47828, 0.70514, 0.95, 1.20, 1.45, 1.70310])
N_modes = 8
w = np.ones(N_modes) / N_modes

def P_QNM(k_array):
    Pk = np.zeros_like(k_array)
    for i, k in enumerate(k_array):
        if k <= 0:
            continue
        x = k * rs
        j2 = spherical_jn(2, x)
        envelope = j2**2
        if envelope < 1e-30:
            continue
        mode_sum = 0.0
        for n in range(N_modes):
            kn = k0 * (n + 1) * beta_geom
            sigma_n = omega_I[n] * k0 / (2.0 * np.pi)
            dk = k - kn
            lorentz = 1.0 / (dk**2 + sigma_n**2)
            asym = (k + kn)**2 / kn**2
            mode_sum += w[n] * lorentz * asym
        Pk[i] = A_QNM * envelope * mode_sum
    return Pk

def P_inflation(k_array, A_s=2.1e-9, n_s=0.9649, k_pivot=0.05):
    return A_s * (k_array / k_pivot)**(n_s - 1.0)

k = np.logspace(-4, -0.5, 2000)
Pk_qnm = P_QNM(k)
Pk_inf = P_inflation(k)
k_modes = [k0 * (n+1) * beta_geom for n in range(N_modes)]
D_rec = 14000

fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
ax1 = axes[0]
ax1.loglog(k, Pk_qnm, 'b-', linewidth=2, label='P_QNM(k)')
ax1.loglog(k, Pk_inf, 'r--', linewidth=1.5, label=f'P_inflation (n_s=0.9649)')
for n, kn in enumerate(k_modes):
    if kn < k[-1]:
        ax1.axvline(kn, color='gray', alpha=0.3, linestyle=':')
k_ell200 = 200.0 / D_rec
k_ell2500 = 2500.0 / D_rec
ax1.axvspan(k_ell200, k_ell2500, alpha=0.1, color='green', label='Planck TT range')
ax1.set_xlabel('k [Mpc^-1]')
ax1.set_ylabel('P(k)')
ax1.set_title('QNM vs Inflationary Primordial Power Spectrum')
ax1.legend()
ax1.set_xlim(1e-4, 0.3)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
with np.errstate(divide='ignore', invalid='ignore'):
    ratio = np.where(Pk_inf > 0, Pk_qnm / Pk_inf, np.nan)
ax2.semilogx(k, ratio, 'b-', linewidth=1.5)
ax2.axhline(1.0, color='r', linestyle='--', alpha=0.5)
ax2.set_xlabel('k [Mpc^-1]')
ax2.set_ylabel('P_QNM / P_inflation')
ax2.set_xlim(1e-4, 0.3)
ax2.set_ylim(0, 5)
ax2.grid(True, alpha=0.3)
ax2.axvspan(k_ell200, k_ell2500, alpha=0.1, color='green')

plt.tight_layout()
plt.savefig('qnm_pk_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: qnm_pk_comparison.png")

print("\n" + "="*60)
print("QNM PRIMORDIAL SPECTRUM DIAGNOSTICS")
print("="*60)
print(f"\nCavity size rs = {rs:.1f} Mpc")
print(f"Fundamental k0 = {k0:.4e} Mpc^-1")
print(f"Fundamental wavelength = {2*np.pi/k0:.0f} Mpc")
print(f"\nQNM resonance positions:")
for n in range(N_modes):
    kn = k0 * (n+1) * beta_geom
    sigma_n = omega_I[n] * k0 / (2*np.pi)
    ell_n = kn * D_rec
    print(f"  n={n}: k_n = {kn:.4e}, sigma = {sigma_n:.4e}, ell ~ {ell_n:.0f}, Q = {kn/(2*sigma_n):.1f}")
print(f"\nPlanck first peak: k ~ {k_ell200:.4e} Mpc^-1")
k_max = k0 * N_modes * beta_geom
print(f"Highest QNM mode: k_7 = {k_max:.4e} Mpc^-1 (ell ~ {k_max*D_rec:.0f})")
if k_max < k_ell200:
    print("\n*** BUMP DESERT: modes don't reach Planck range ***")
else:
    print("\nModes extend into Planck range.")
