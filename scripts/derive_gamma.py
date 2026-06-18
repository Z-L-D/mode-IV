#!/usr/bin/env python3
"""
Derivation of the QNM mode weighting exponent γ from first principles.
=====================================================================

The question: why does γ = 1.04 give the best CMB fit?

The mode weighting w_n = 1/(n+1)^γ encodes how much energy each
QNM overtone receives during the cavity formation event (gravitational
collapse). γ is determined by the overlap integral between the
collapse perturbation (source) and the QNM eigenfunctions.

Physics chain:
1. QNM eigenfunctions from the Regge-Wheeler equation
2. Oppenheimer-Snyder collapse source profile
3. Overlap integral → mode excitation amplitude
4. Fit → γ

We use geometric units: G = c = 1, M = 1/2 (so r_s = 2M = 1).
"""

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ============================================================
# 1. Schwarzschild QNM frequencies (Leaver values + asymptotic)
# ============================================================

# Using 2M = 1 convention (M = 1/2)
# Standard Leaver values for l=2, s=2 (gravitational), n=0..7
# Format: (M*omega_R, M*omega_I) with M = 1/2
# So omega = 2 * M*omega for our convention

# Leaver values in the standard 2M=1 convention
leaver_2M1 = [
    (0.74734, 0.17792),   # n=0
    (0.69342, 0.54783),   # n=1
    (0.60211, 0.95654),   # n=2
    (0.50302, 1.41030),   # n=3
    (0.41583, 1.89369),   # n=4
    (0.34615, 2.39134),   # n=5
    (0.29131, 2.89591),   # n=6
    (0.24834, 3.40400),   # n=7
]

def qnm_frequency(n, l=2):
    """Return complex QNM frequency for mode n (2M=1 units)."""
    if n < len(leaver_2M1):
        wr, wi = leaver_2M1[n]
        return complex(wr, -wi)  # convention: damped modes have Im < 0
    else:
        # Motl-Neitzke asymptotic formula
        wr = np.log(3) / (4 * np.pi)  # = ln(3)/(8πM) with M=1/2
        wi = (2*n + 1) / 4            # = (2n+1)/(8M) with M=1/2
        return complex(wr, -wi)

print("QNM frequencies (2M=1 units):")
for n in range(12):
    omega = qnm_frequency(n)
    print(f"  n={n:2d}: ω = {omega.real:.5f} - {-omega.imag:.5f}i")

# ============================================================
# 2. Regge-Wheeler potential
# ============================================================

def tortoise(r, rs=1.0):
    """Tortoise coordinate r* = r + rs*ln(r/rs - 1)"""
    return r + rs * np.log(np.abs(r/rs - 1))

def V_regge_wheeler(r, l=2, s=2, rs=1.0):
    """Regge-Wheeler potential for spin-s perturbations."""
    f = 1 - rs/r
    return f * (l*(l+1)/r**2 + rs*(1 - s**2)/r**3)

# Evaluate potential on a grid
r_grid = np.linspace(1.001, 20, 10000)  # r_s = 1, so start just outside horizon
rstar_grid = tortoise(r_grid)
V_grid = V_regge_wheeler(r_grid)

# Find potential peak
V_peak_idx = np.argmax(V_grid)
r_peak = r_grid[V_peak_idx]
rstar_peak = rstar_grid[V_peak_idx]
V_max = V_grid[V_peak_idx]
print(f"\nPotential peak: r = {r_peak:.4f}, r* = {rstar_peak:.4f}, V_max = {V_max:.6f}")

# ============================================================
# 3. WKB eigenfunctions (approximate)
# ============================================================

def wkb_eigenfunction(rstar_vals, omega_n, V_func_rstar, normalize=True):
    """
    WKB approximation for QNM eigenfunction.
    
    In the classically allowed region (ω² > V): oscillating
    In the classically forbidden region (ω² < V): evanescent
    
    For QNMs with complex ω, all regions are "allowed" in some sense.
    We use the real part of ω for the WKB phase.
    """
    omega_R = omega_n.real
    omega_I = omega_n.imag  # negative for damped modes
    
    psi = np.zeros(len(rstar_vals), dtype=complex)
    
    for i, rs in enumerate(rstar_vals):
        # Local wavenumber
        V_here = np.interp(rs, rstar_grid, V_grid)
        k2 = omega_R**2 - V_here
        
        if k2 > 0:
            k = np.sqrt(k2)
            # Phase accumulated from left boundary
            # Approximate: use local k
            phase = omega_R * rs  # simplified; full version would integrate k(r*)
            psi[i] = np.sin(phase + n * np.pi * rs / (rstar_grid[-1] - rstar_grid[0])) / np.sqrt(k + 0.01)
        else:
            k = np.sqrt(-k2)
            psi[i] = np.exp(-k * abs(rs - rstar_peak)) / np.sqrt(k + 0.01)
    
    if normalize:
        norm = np.sqrt(np.trapezoid(np.abs(psi)**2, rstar_vals))
        if norm > 0:
            psi /= norm
    
    return psi

# ============================================================
# 4. More rigorous: direct numerical integration of R-W equation
# ============================================================

def rw_eigenfunction_numerical(omega_n, r_grid, l=2, s=2, rs=1.0):
    """
    Numerically integrate the Regge-Wheeler equation:
    d²ψ/dr*² + [ω² - V(r*)] ψ = 0
    
    Start near the horizon with ingoing wave BC:
    ψ ~ exp(-iω r*) as r* → -∞
    
    Integrate outward.
    """
    # Convert r grid to r* grid
    rstar = tortoise(r_grid, rs)
    
    # Set up ODE: let y1 = ψ, y2 = dψ/dr*
    # dy1/dr* = y2
    # dy2/dr* = [V(r) - ω²] y1
    
    def ode_rhs(rstar_val, y):
        # Interpolate V at this r* value
        V_val = np.interp(rstar_val, rstar, V_regge_wheeler(r_grid, l, s, rs))
        return [y[1], (V_val - omega_n**2) * y[0]]
    
    # Initial conditions: ingoing wave at horizon
    # ψ ~ exp(-iω r*), dψ/dr* ~ -iω exp(-iω r*)
    rstar_start = rstar[0]
    phase = -1j * omega_n * rstar_start
    y0 = [np.exp(phase), -1j * omega_n * np.exp(phase)]
    
    # Integrate
    sol = solve_ivp(ode_rhs, [rstar[0], rstar[-1]], y0, 
                    t_eval=rstar, rtol=1e-10, atol=1e-12, method='DOP853')
    
    if sol.success:
        psi = sol.y[0]
        # Normalize in the region near the potential peak
        mask = (r_grid > 1.2) & (r_grid < 5.0)
        norm = np.sqrt(np.trapezoid(np.abs(psi[mask])**2, rstar[mask]))
        if norm > 0:
            psi /= norm
        return psi, rstar
    else:
        return None, rstar

# ============================================================
# 5. Source function: Oppenheimer-Snyder collapse
# ============================================================

def os_source(r, R_initial=10.0, rs=1.0):
    """
    Oppenheimer-Snyder collapse density perturbation.
    
    Uniform density sphere collapsing from radius R_initial.
    At the moment of horizon formation, the matter is at r = rs.
    
    The gravitational wave source is the quadrupole moment:
    S(r) ∝ r² ρ(r) for quadrupole coupling
    
    For uniform density inside R_initial:
    S(r) = r² for r < R_surface, 0 otherwise
    
    At horizon formation, R_surface ≈ rs = 1
    """
    return np.where(r < rs * 1.5, r**2 * (1 - rs/r)**0.5, 0)

def simple_source(r, rs=1.0):
    """
    Simplified source: quadrupole moment of collapsing matter.
    The perturbation couples to ψ through the quadrupole overlap.
    
    For a smooth collapse, the source is approximately:
    S(r) ~ (r/rs - 1)^(1/2) * (r/rs)^2 near the horizon
    
    This has a square-root onset at the horizon (r = rs)
    that determines the high-frequency spectral content.
    """
    x = r / rs
    result = np.zeros_like(r)
    mask = x > 1.0
    result[mask] = np.sqrt(x[mask] - 1) * x[mask]**2 * np.exp(-(x[mask]-1)/3)
    return result

# ============================================================
# 6. Compute overlap integrals
# ============================================================

print("\n" + "="*60)
print("Computing mode excitation amplitudes...")
print("="*60)

# Use a fine r grid for integration
r_fine = np.linspace(1.0005, 15.0, 50000)
rstar_fine = tortoise(r_fine)

# Source function in r-space
S_r = simple_source(r_fine)

# Also compute source in r*-space (need Jacobian: dr*/dr = 1/(1-rs/r))
dr_drstar = 1 - 1.0/r_fine  # = f(r) = 1 - rs/r
S_rstar = S_r * dr_drstar  # S(r) dr = S(r) (dr/dr*) dr*

n_modes = 40
excitation = np.zeros(n_modes)
excitation_alt = np.zeros(n_modes)

# Method 1: Frequency-domain approach
# The excitation of mode n from a time-domain source S(t) is:
# C_n = ∫ S(t) exp(iω_n t) dt
# For our case, the "time" is the collapse timescale τ.
# The source has a characteristic frequency ω_source ~ 1/τ_collapse
# where τ_collapse ~ πM = π/2 in our units.

tau_collapse = np.pi / 2  # O-S free-fall time in 2M=1 units

print(f"\nCollapse timescale: τ = {tau_collapse:.4f} (2M=1 units)")
print(f"Corresponding frequency: ω_collapse = {1/tau_collapse:.4f}")

# For a source with time profile:
# S(t) = (1 - (t/τ)²)^(1/2) for |t| < τ  (semicircular profile)
# The Fourier transform is a Bessel function:
# S̃(ω) = τ × J₁(ωτ) / (ωτ)

from scipy.special import j1 as bessel_j1

for n in range(n_modes):
    omega_n = qnm_frequency(n)
    
    # Method 1: Frequency-domain (Fourier of collapse profile)
    # Use |ω_n| since we want the spectral amplitude
    omega_abs = abs(omega_n)
    omega_phys = np.sqrt(omega_n.real**2 + omega_n.imag**2)
    
    # Semicircular collapse profile → Bessel function
    x = omega_phys * tau_collapse
    if x > 0:
        # |S̃(ω)|² for semicircular pulse
        ft_amp = abs(bessel_j1(x) / x) if x > 0.01 else 0.5
    else:
        ft_amp = 0.5
    
    excitation[n] = ft_amp**2
    
    # Method 2: Simple spectral model
    # dE/dω for gravitational collapse ~ ω⁴/(1 + (ωτ)²)^3
    # (Quadrupole formula for dust collapse)
    omega_R = abs(omega_n.real)
    omega_I = abs(omega_n.imag)
    
    # Effective frequency for spectral evaluation
    omega_eff = np.sqrt(omega_R**2 + omega_I**2)
    x2 = (omega_eff * tau_collapse)**2
    
    excitation_alt[n] = omega_eff**4 / (1 + x2)**3

# Method 3: Spatial overlap with numerical eigenfunctions
print("\nComputing numerical eigenfunctions and spatial overlaps...")
excitation_spatial = np.zeros(n_modes)

for n in range(min(n_modes, 25)):  # limit to first 25 for speed
    omega_n = qnm_frequency(n)
    
    try:
        psi, rstar = rw_eigenfunction_numerical(omega_n, r_fine)
        if psi is not None:
            # Overlap integral: C_n = ∫ S(r*) × ψ_n(r*) dr*
            integrand = S_rstar * psi
            C_n = np.trapezoid(integrand, rstar)
            excitation_spatial[n] = abs(C_n)**2
    except Exception as e:
        print(f"  n={n}: numerical integration failed ({e})")
        excitation_spatial[n] = np.nan

# ============================================================
# 7. Fit γ for each method
# ============================================================

def power_law(n, A, gamma):
    return A / (n + 1)**gamma

print("\n" + "="*60)
print("FITTING γ")
print("="*60)

ns = np.arange(n_modes)

for name, exc in [("Freq-domain (Bessel)", excitation), 
                   ("Quadrupole spectral", excitation_alt)]:
    # Normalize
    exc_norm = exc / exc[0] if exc[0] > 0 else exc
    
    # Fit power law to n >= 1
    mask = (ns >= 1) & (exc_norm > 0) & np.isfinite(exc_norm)
    if np.sum(mask) < 3:
        print(f"\n{name}: insufficient data for fit")
        continue
    
    try:
        popt, pcov = curve_fit(power_law, ns[mask], exc_norm[mask], 
                               p0=[1.0, 1.0], maxfev=10000)
        A_fit, gamma_fit = popt
        gamma_err = np.sqrt(pcov[1,1])
        
        print(f"\n{name}:")
        print(f"  γ = {gamma_fit:.4f} ± {gamma_err:.4f}")
        print(f"  A = {A_fit:.4f}")
        print(f"  Target: γ = 1.04")
        print(f"  Deviation from target: {abs(gamma_fit - 1.04):.4f}")
        
        # Print first few values
        for nn in [0, 1, 2, 5, 10, 20]:
            if nn < len(exc_norm):
                print(f"    n={nn}: w_n = {exc_norm[nn]:.6f}, fit = {power_law(nn, *popt):.6f}")
    except Exception as e:
        print(f"\n{name}: fit failed ({e})")

# Spatial overlap
mask_s = (ns[:25] >= 1) & (excitation_spatial[:25] > 0) & np.isfinite(excitation_spatial[:25])
if np.sum(mask_s) >= 3:
    exc_s_norm = excitation_spatial[:25] / excitation_spatial[0] if excitation_spatial[0] > 0 else excitation_spatial[:25]
    try:
        popt_s, pcov_s = curve_fit(power_law, ns[:25][mask_s], exc_s_norm[mask_s],
                                    p0=[1.0, 1.0], maxfev=10000)
        gamma_s = popt_s[1]
        gamma_s_err = np.sqrt(pcov_s[1,1])
        print(f"\nSpatial overlap (numerical eigenfunctions):")
        print(f"  γ = {gamma_s:.4f} ± {gamma_s_err:.4f}")
        print(f"  Target: γ = 1.04")
    except Exception as e:
        print(f"\nSpatial overlap: fit failed ({e})")

# ============================================================
# 8. Explore collapse timescale dependence
# ============================================================

print("\n" + "="*60)
print("COLLAPSE TIMESCALE SCAN")
print("="*60)
print("How does γ depend on the collapse timescale τ?")

tau_values = np.linspace(0.5, 5.0, 20)
gamma_vs_tau = []

for tau in tau_values:
    exc_tau = np.zeros(n_modes)
    for n in range(n_modes):
        omega_n = qnm_frequency(n)
        omega_phys = np.sqrt(omega_n.real**2 + omega_n.imag**2)
        x = omega_phys * tau
        if x > 0.01:
            exc_tau[n] = (bessel_j1(x) / x)**2
        else:
            exc_tau[n] = 0.25
    
    exc_tau_norm = exc_tau / exc_tau[0] if exc_tau[0] > 0 else exc_tau
    mask_t = (ns >= 1) & (exc_tau_norm > 0) & np.isfinite(exc_tau_norm)
    
    try:
        popt_t, _ = curve_fit(power_law, ns[mask_t], exc_tau_norm[mask_t], p0=[1.0, 1.0])
        gamma_vs_tau.append((tau, popt_t[1]))
    except:
        gamma_vs_tau.append((tau, np.nan))

gamma_vs_tau = np.array(gamma_vs_tau)

print(f"\n  {'τ':>6s}  {'γ':>8s}  {'|γ-1.04|':>10s}")
print(f"  {'-'*6}  {'-'*8}  {'-'*10}")
for tau, gamma in gamma_vs_tau:
    if np.isfinite(gamma):
        print(f"  {tau:6.3f}  {gamma:8.4f}  {abs(gamma-1.04):10.4f}")

# Find τ that gives γ closest to 1.04
best_idx = np.nanargmin(np.abs(gamma_vs_tau[:,1] - 1.04))
best_tau = gamma_vs_tau[best_idx, 0]
best_gamma = gamma_vs_tau[best_idx, 1]
print(f"\nBest match: τ = {best_tau:.3f}, γ = {best_gamma:.4f}")
print(f"Physical interpretation: τ = {best_tau:.3f} × (2M) = {best_tau:.3f} × r_s")
print(f"  = {best_tau/np.pi:.4f} × π (compare to O-S free-fall time τ_ff = π/2)")

# ============================================================
# 9. Plot
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top left: excitation amplitudes
ax = axes[0,0]
for name, exc, color in [("Bessel", excitation, 'b'), 
                          ("Quadrupole", excitation_alt, 'r')]:
    exc_norm = exc / exc[0] if exc[0] > 0 else exc
    ax.semilogy(ns, exc_norm, 'o-', ms=4, color=color, label=name, alpha=0.7)

# Overplot power law with γ=1.04
ax.semilogy(ns, 1/(ns+1)**1.04, 'k--', lw=2, label='γ=1.04 (CMB fit)')
ax.semilogy(ns, 1/(ns+1)**1.00, 'gray', ls=':', lw=1, label='γ=1.00 (scale-free)')
ax.set_xlabel('Overtone n')
ax.set_ylabel('w_n / w_0')
ax.set_title('Mode Excitation Amplitude')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 35)

# Top right: γ vs τ
ax = axes[0,1]
ax.plot(gamma_vs_tau[:,0], gamma_vs_tau[:,1], 'bo-', ms=5)
ax.axhline(1.04, color='r', ls='--', lw=2, label='γ=1.04 (CMB)')
ax.axhline(1.00, color='gray', ls=':', lw=1, label='γ=1.00 (scale-free)')
ax.axvline(np.pi/2, color='g', ls='--', lw=1, label=f'τ=π/2 (O-S)')
ax.set_xlabel('Collapse timescale τ (2M=1 units)')
ax.set_ylabel('Fitted γ')
ax.set_title('γ vs Collapse Timescale')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Bottom left: source function
ax = axes[1,0]
ax.plot(r_fine, simple_source(r_fine), 'b-', lw=2)
ax.set_xlabel('r / r_s')
ax.set_ylabel('S(r)')
ax.set_title('Collapse Source Function')
ax.set_xlim(0.9, 8)
ax.grid(True, alpha=0.3)

# Bottom right: Regge-Wheeler potential
ax = axes[1,1]
ax.plot(rstar_grid, V_grid, 'b-', lw=2)
ax.set_xlabel('r* / r_s')
ax.set_ylabel('V(r*)')
ax.set_title('Regge-Wheeler Potential (l=2, s=2)')
ax.set_xlim(-10, 20)
ax.grid(True, alpha=0.3)

# Mark QNM frequencies
for n in range(5):
    omega = qnm_frequency(n)
    ax.axhline(omega.real**2, color='r', ls=':', alpha=0.3)

plt.tight_layout()
plt.savefig('gamma_derivation.png', dpi=150)
print(f"\nSaved: gamma_derivation.png")
