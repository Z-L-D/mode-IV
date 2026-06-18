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

def excitation(n, R0):
    """
    Physical mode excitation with damping absorbed into integrand.
    
    Instead of: C_n = ∫ Q̈ exp(iωt) dt, then multiply by exp(-|ω_I|τ)
    Compute:    C_n = ∫ Q̈ exp(iω_R t) exp(-|ω_I|(τ-t)) dt
    
    This is bounded because (τ-t) >= 0, so exp term <= 1.
    """
    omega = qnm_freq(n)
    omega_R = omega.real
    omega_I_abs = abs(omega.imag)
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
        # Bounded exponential: oscillation × decay from remaining time
        phase = omega_R * t
        decay = -omega_I_abs * (tau_ff - t)
        exp_factor = np.exp(decay) * np.cos(phase)
        return d2Q * exp_factor * dt_de

    def integrand_imag(eta):
        if eta > np.pi - 1e-10: return 0.0
        cos_e, sin_e = np.cos(eta), np.sin(eta)
        dt_de = c1 * (1 + cos_e)
        if abs(dt_de) < 1e-15: return 0.0
        d2Q = -R0_sq/(2*c1) * cos_e / dt_de
        t = c1 * (eta + sin_e)
        phase = omega_R * t
        decay = -omega_I_abs * (tau_ff - t)
        exp_factor = np.exp(decay) * np.sin(phase)
        return d2Q * exp_factor * dt_de

    re, _ = quad(integrand_real, 0, np.pi-1e-8, limit=200)
    im, _ = quad(integrand_imag, 0, np.pi-1e-8, limit=200)
    return re**2 + im**2

def power_law(n, A, gamma):
    return A / (n+1)**gamma

print("Exact O-S collapse with physical damping")
print("="*55)
print(f"{'R0':>5s}  {'tau_ff':>8s}  {'gamma':>8s}  {'|g-1.04|':>8s}  {'method':>8s}")
print("-"*55)

n_modes = 30
ns = np.arange(n_modes)

import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22']

for idx, R0 in enumerate([1.1, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]):
    tau_ff = np.pi * R0**1.5 / 2
    
    try:
        exc = np.array([excitation(n, R0) for n in range(n_modes)])
    except:
        print(f"{R0:5.1f}  {tau_ff:8.4f}  OVERFLOW")
        continue
    
    if exc[0] <= 0 or not np.all(np.isfinite(exc)):
        print(f"{R0:5.1f}  {tau_ff:8.4f}  INVALID")
        continue
    
    exc_norm = exc / exc[0]
    mask = (ns >= 1) & (exc_norm > 1e-15) & np.isfinite(exc_norm)
    
    method = "curvefit"
    try:
        popt, pcov = curve_fit(power_law, ns[mask], exc_norm[mask], 
                               p0=[1.0, 1.5], maxfev=10000)
        g = popt[1]
        gerr = np.sqrt(pcov[1,1]) if np.isfinite(pcov[1,1]) else 0
    except:
        # Log-space linear fit as fallback
        log_exc = np.log(exc_norm[mask])
        log_n = np.log(ns[mask] + 1)
        slope, intercept = np.polyfit(log_n, log_exc, 1)
        g = -slope
        gerr = 0
        method = "log-fit"
    
    print(f"{R0:5.1f}  {tau_ff:8.4f}  {g:8.4f}  {abs(g-1.04):8.4f}  {method:>8s}")
    
    c = colors[idx % len(colors)]
    ax1.semilogy(ns[:20], exc_norm[:20], 'o-', ms=3, color=c, 
                 alpha=0.7, label=f'R₀={R0}, γ={g:.2f}')

# Reference lines
ax1.semilogy(ns[:20], 1/(ns[:20]+1)**1.04, 'k--', lw=2.5, label='γ=1.04 (CMB)')
ax1.semilogy(ns[:20], 1/(ns[:20]+1)**1.00, color='gray', ls=':', lw=1.5, label='γ=1.00')
ax1.set_xlabel('Overtone n', fontsize=12)
ax1.set_ylabel('w_n / w_0', fontsize=12)
ax1.set_title('Mode Excitation: Exact O-S with Damping', fontsize=13)
ax1.legend(fontsize=7, ncol=2)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 19)
ax1.set_ylim(1e-4, 2)

ax2.text(0.5, 0.5, 'See terminal output\nfor γ vs R₀ table', 
         transform=ax2.transAxes, ha='center', va='center', fontsize=14)
ax2.set_title('Summary', fontsize=13)

plt.tight_layout()
plt.savefig('gamma_derivation_exact.png', dpi=150)
print(f"\nSaved: gamma_derivation_exact.png")
