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
    omega_R = qnm_freq(n).real
    tau_ff = np.pi * R0**1.5 / 2
    c1 = tau_ff / np.pi
    R0_sq = R0**2
    def integ_re(eta):
        if eta > np.pi-1e-10: return 0.0
        ce, se = np.cos(eta), np.sin(eta)
        dt = c1*(1+ce)
        if abs(dt)<1e-15: return 0.0
        d2Q = -R0_sq/(2*c1)*ce/dt
        t = c1*(eta+se)
        return d2Q*np.cos(omega_R*t)*dt
    def integ_im(eta):
        if eta > np.pi-1e-10: return 0.0
        ce, se = np.cos(eta), np.sin(eta)
        dt = c1*(1+ce)
        if abs(dt)<1e-15: return 0.0
        d2Q = -R0_sq/(2*c1)*ce/dt
        t = c1*(eta+se)
        return d2Q*np.sin(omega_R*t)*dt
    re,_ = quad(integ_re, 0, np.pi-1e-8, limit=200)
    im,_ = quad(integ_im, 0, np.pi-1e-8, limit=200)
    return re**2 + im**2

n_modes = 25
ns = np.arange(n_modes)

print("Fine scan: R₀ = 3.0 to 5.0")
print(f"{'R0':>6s}  {'tau_ff':>8s}  {'gamma':>8s}  {'|g-1.04|':>8s}")
print("-"*40)

results = []
for R0 in np.arange(3.0, 5.1, 0.2):
    tau_ff = np.pi * R0**1.5 / 2
    exc = np.array([excitation(n, R0) for n in range(n_modes)])
    if exc[0] <= 0: continue
    exc_norm = exc / exc[0]
    mask = (ns >= 1) & (exc_norm > 1e-15) & np.isfinite(exc_norm)
    log_exc = np.log(exc_norm[mask])
    log_n = np.log(ns[mask] + 1)
    slope, _ = np.polyfit(log_n, log_exc, 1)
    g = -slope
    results.append((R0, tau_ff, g))
    print(f"{R0:6.2f}  {tau_ff:8.4f}  {g:8.4f}  {abs(g-1.04):8.4f}")

# Interpolate to find exact R₀ for γ = 1.04
if len(results) >= 2:
    R0s = np.array([r[0] for r in results])
    gammas = np.array([r[2] for r in results])
    
    # Find crossing
    for i in range(len(gammas)-1):
        if (gammas[i] - 1.04) * (gammas[i+1] - 1.04) < 0:
            # Linear interpolation
            R0_cross = R0s[i] + (1.04 - gammas[i]) * (R0s[i+1] - R0s[i]) / (gammas[i+1] - gammas[i])
            tau_cross = np.pi * R0_cross**1.5 / 2
            print(f"\n*** γ = 1.04 at R₀ = {R0_cross:.3f} r_s ***")
            print(f"    τ_ff = {tau_cross:.4f} (2M=1)")
            print(f"    τ_ff = {tau_cross*2:.4f} M")
            print(f"    R₀ = {R0_cross:.3f} × 2GM/c²")
            print(f"    R₀/r_s = {R0_cross:.3f}")
            
            # Physical meaning
            r_isco = 6  # Schwarzschild ISCO in r_s units (= 6M for r_s = 2M = 1)
            r_photon = 1.5  # Photon sphere = 3M = 1.5 r_s
            print(f"\n    Photon sphere: r = {r_photon:.1f} r_s")
            print(f"    ISCO:          r = {r_isco:.1f} r_s")
            print(f"    R₀:            r = {R0_cross:.2f} r_s")
            
            if R0_cross < r_photon:
                print(f"    → Inside photon sphere (ultra-compact)")
            elif R0_cross < r_isco:
                print(f"    → Between photon sphere and ISCO")
            else:
                print(f"    → Outside ISCO")
            break
