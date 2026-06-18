import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit

leaver = [(0.74734,0.17792),(0.69342,0.54783),(0.60211,0.95654),
          (0.50302,1.41030),(0.41583,1.89369),(0.34615,2.39134),
          (0.29131,2.89591),(0.24834,3.40400)]

def qnm_freq(n):
    if n < len(leaver):
        return complex(leaver[n][0], -leaver[n][1])
    wr = np.log(3)/(4*np.pi)
    wi = (2*n+1)/4
    return complex(wr, -wi)

def excitation_real_omega(n, R0):
    """
    Mode excitation using REAL part of QNM frequency only.
    
    Physical reasoning: during collapse, the source couples to
    the mode's oscillation frequency (ω_R). The damping rate (ω_I)
    describes energy loss AFTER the cavity forms, not during excitation.
    
    C_n = ∫ Q̈(t) exp(i ω_R t) dt
    """
    omega_R = qnm_freq(n).real
    tau_ff = np.pi * R0**1.5 / 2
    c1 = tau_ff / np.pi
    R0_sq = R0**2

    def integrand_real(eta):
        if eta > np.pi - 1e-10: return 0.0
        cos_e, sin_e = np.cos(eta), np.sin(eta)
        dt_de = c1 * (1 + cos_e)
        if abs(dt_de) < 1e-15: return 0.0
        d2Q = -R0_sq/(2*c1) * cos_e / dt_de
        t = c1 * (eta + sin_e)
        return d2Q * np.cos(omega_R * t) * dt_de

    def integrand_imag(eta):
        if eta > np.pi - 1e-10: return 0.0
        cos_e, sin_e = np.cos(eta), np.sin(eta)
        dt_de = c1 * (1 + cos_e)
        if abs(dt_de) < 1e-15: return 0.0
        d2Q = -R0_sq/(2*c1) * cos_e / dt_de
        t = c1 * (eta + sin_e)
        return d2Q * np.sin(omega_R * t) * dt_de

    re, _ = quad(integrand_real, 0, np.pi-1e-8, limit=200)
    im, _ = quad(integrand_imag, 0, np.pi-1e-8, limit=200)
    return re**2 + im**2

def power_law(n, A, gamma):
    return A / (n+1)**gamma

print("Exact O-S collapse, real ω only (physical excitation)")
print("="*60)
print(f"{'R0':>5s}  {'tau_ff':>8s}  {'tau/M':>6s}  {'gamma':>8s}  {'|g-1.04|':>8s}")
print("-"*60)

import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

n_modes = 30
ns = np.arange(n_modes)
all_results = []

for R0 in [1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]:
    tau_ff = np.pi * R0**1.5 / 2
    tau_over_M = tau_ff * 2  # convert to GM/c³ units
    
    try:
        exc = np.array([excitation_real_omega(n, R0) for n in range(n_modes)])
    except Exception as e:
        print(f"{R0:5.1f}  {tau_ff:8.4f}  {tau_over_M:6.2f}  ERROR: {e}")
        continue
    
    if exc[0] <= 0 or not np.all(np.isfinite(exc)):
        print(f"{R0:5.1f}  {tau_ff:8.4f}  {tau_over_M:6.2f}  INVALID")
        continue
    
    exc_norm = exc / exc[0]
    mask = (ns >= 1) & (exc_norm > 1e-15) & np.isfinite(exc_norm)
    
    # Log-space fit (more robust)
    log_exc = np.log(exc_norm[mask])
    log_n = np.log(ns[mask] + 1)
    slope, intercept = np.polyfit(log_n, log_exc, 1)
    g = -slope
    
    # Also try curve_fit
    try:
        popt, pcov = curve_fit(power_law, ns[mask], exc_norm[mask],
                               p0=[1.0, max(g, 0.1)], maxfev=10000)
        g_cf = popt[1]
        method = "both"
    except:
        g_cf = g
        method = "log"
    
    # Use log-fit gamma as primary (more robust for power laws)
    print(f"{R0:5.1f}  {tau_ff:8.4f}  {tau_over_M:6.2f}  {g:8.4f}  {abs(g-1.04):8.4f}  (cf:{g_cf:.3f})")
    all_results.append((R0, tau_ff, g, exc_norm.copy()))
    
    ax1.semilogy(ns[:20], exc_norm[:20], 'o-', ms=3, alpha=0.7,
                 label=f'R₀={R0}, γ={g:.2f}')

# Reference
ax1.semilogy(ns[:20], 1/(ns[:20]+1)**1.04, 'k--', lw=2.5, label='γ=1.04 (CMB)')
ax1.semilogy(ns[:20], 1/(ns[:20]+1)**1.00, color='gray', ls=':', lw=1.5, label='γ=1.00')
ax1.set_xlabel('Overtone n', fontsize=12)
ax1.set_ylabel('w_n / w_0', fontsize=12)
ax1.set_title('Mode Excitation (ω_R only)', fontsize=13)
ax1.legend(fontsize=7, ncol=2)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 19)

# γ vs R₀
if all_results:
    R0s = [r[0] for r in all_results]
    gammas = [r[2] for r in all_results]
    ax2.semilogx(R0s, gammas, 'bo-', ms=8, lw=2)
    ax2.axhline(1.04, color='r', ls='--', lw=2, label='γ=1.04 (CMB)')
    ax2.axhline(1.00, color='gray', ls=':', lw=1, label='γ=1.00')
    ax2.set_xlabel('R₀ / r_s', fontsize=12)
    ax2.set_ylabel('Derived γ', fontsize=12)
    ax2.set_title('γ vs Initial Radius', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gamma_derivation_exact.png', dpi=150)
print(f"\nSaved: gamma_derivation_exact.png")

# Summary
if all_results:
    best = min(all_results, key=lambda x: abs(x[2]-1.04))
    print(f"\nBest match: R₀ = {best[0]:.2f} r_s, γ = {best[2]:.4f}")
    print(f"Deviation from 1.04: {abs(best[2]-1.04):.4f} ({abs(best[2]-1.04)/1.04*100:.1f}%)")
    
    # Does γ cross 1.04 at any R₀?
    gammas = [r[2] for r in all_results]
    if max(gammas) >= 1.04:
        print(f"γ = 1.04 IS achievable in the scanned range")
    else:
        print(f"Maximum γ = {max(gammas):.4f} at R₀ = {all_results[gammas.index(max(gammas))][0]:.1f}")
        print(f"γ = 1.04 requires either:")
        print(f"  - Larger R₀ (more extended collapse)")
        print(f"  - Different source profile (not uniform density)")
        print(f"  - Rotation (Kerr corrections)")
