import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit

# QNM frequencies (2M=1)
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
    omega = qnm_freq(n)
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
        val = d2Q * np.exp(1j * omega * t) * dt_de
        return val.real

    def integrand_imag(eta):
        if eta > np.pi - 1e-10: return 0.0
        cos_e, sin_e = np.cos(eta), np.sin(eta)
        dt_de = c1 * (1 + cos_e)
        if abs(dt_de) < 1e-15: return 0.0
        d2Q = -R0_sq/(2*c1) * cos_e / dt_de
        t = c1 * (eta + sin_e)
        val = d2Q * np.exp(1j * omega * t) * dt_de
        return val.imag

    re, _ = quad(integrand_real, 0, np.pi-1e-8, limit=200)
    im, _ = quad(integrand_imag, 0, np.pi-1e-8, limit=200)
    C_n_sq = re**2 + im**2
    
    # PHYSICAL CORRECTION: subtract the exponential growth
    # Energy = |C_n|² × exp(-2|ω_I| × τ_ff)
    damping = np.exp(-2 * abs(omega.imag) * tau_ff)
    return C_n_sq * damping

def power_law(n, A, gamma):
    return A / (n+1)**gamma

print("R0     tau_ff   gamma    |g-1.04|")
print("-"*45)

n_modes = 25
ns = np.arange(n_modes)

for R0 in [1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]:
    tau_ff = np.pi * R0**1.5 / 2
    exc = np.array([excitation(n, R0) for n in range(n_modes)])
    
    if exc[0] <= 0 or not np.all(np.isfinite(exc)):
        print(f"{R0:5.1f}  {tau_ff:8.4f}  FAILED")
        continue
    
    exc_norm = exc / exc[0]
    mask = (ns >= 1) & (exc_norm > 0) & np.isfinite(exc_norm)
    
    try:
        popt, pcov = curve_fit(power_law, ns[mask], exc_norm[mask], 
                               p0=[1.0, 1.5], maxfev=10000)
        g = popt[1]
        print(f"{R0:5.1f}  {tau_ff:8.4f}  {g:8.4f}  {abs(g-1.04):8.4f}")
    except:
        # Try log-space fit
        log_exc = np.log(exc_norm[mask])
        log_n = np.log(ns[mask] + 1)
        slope, intercept = np.polyfit(log_n, log_exc, 1)
        g = -slope
        print(f"{R0:5.1f}  {tau_ff:8.4f}  {g:8.4f}  {abs(g-1.04):8.4f}  (log fit)")
