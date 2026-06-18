#!/usr/bin/env python3
"""
Modified gamma_fine.py — with GR angular momentum
===================================================
Replaces the Newtonian cycloid with full GR radial dynamics 
including angular momentum. Tests whether L = L_ISCO at 
R₀ = ISCO gives γ = 1.04.

Units: 2M = 1 (so r_s = 1, M = 0.5, ISCO at R = 3.0)
"""
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

# QNM frequencies (same as gamma_fine.py)
leaver = [(0.74734,0.17792),(0.69342,0.54783),(0.60211,0.95654),
          (0.50302,1.41030),(0.41583,1.89369),(0.34615,2.39134),
          (0.29131,2.89591),(0.24834,3.40400)]

def qnm_freq(n):
    if n < len(leaver):
        return complex(leaver[n][0], -leaver[n][1])
    wr = np.log(3)/(4*np.pi)
    wi = (2*n+1)/4
    return complex(wr, -wi)

# GR effective potential (2M = 1 units)
def V_eff_GR(R, L):
    """V_eff = (1 - 1/R)(1 + L²/R²)"""
    if R <= 1.0: return 1e10
    return (1 - 1.0/R) * (1 + L**2/R**2)

# ISCO parameters in 2M = 1 units
R_ISCO = 3.0
E_ISCO = np.sqrt(8.0/9.0)   # ≈ 0.9428
L_ISCO = np.sqrt(3.0)        # 2√3 M = 2√3 × 0.5 = √3 ≈ 1.732

print("="*60)
print("Rotating O-S Collapse → γ (GR dynamics)")  
print("="*60)
print(f"ISCO: R = {R_ISCO:.1f} (2M=1), E = {E_ISCO:.4f}, L = {L_ISCO:.4f}")
print(f"V_eff(ISCO) = {V_eff_GR(R_ISCO, L_ISCO):.6f}, E² = {E_ISCO**2:.6f}")

def compute_trajectory(R0, L, E=None, eps=0.001):
    """
    Integrate GR radial geodesic with angular momentum.
    Returns t_coord[], R[] arrays.
    
    Proper time equations:
      dR/dτ = -√(E² - V_eff(R))
      dt/dτ = E / (1 - 1/R)
      dφ/dτ = L / R²
    """
    if E is None:
        # Start from rest: E² = V_eff(R0, L) minus small perturbation
        E = np.sqrt(V_eff_GR(R0, L)) * (1 - eps)
    
    def rhs(tau, y):
        R, t_c, phi = y
        if R <= 1.01: return [0, 0, 0]
        f = 1 - 1.0/R
        Veff = f * (1 + L**2/R**2)
        radicand = E**2 - Veff
        if radicand < 0: radicand = 0
        dRdtau = -np.sqrt(radicand)
        dtdtau = E / f
        dphidtau = L / R**2
        return [dRdtau, dtdtau, dphidtau]
    
    def event_horizon(tau, y):
        return y[0] - 1.02
    event_horizon.terminal = True
    
    sol = solve_ivp(rhs, [0, 500], [R0, 0, 0],
                    method='DOP853', max_step=0.02,
                    events=event_horizon,
                    rtol=1e-12, atol=1e-14,
                    dense_output=True)
    
    n_pts = 50000
    tau = np.linspace(0, sol.t[-1]*0.999, n_pts)
    y = sol.sol(tau)
    R, t_c, phi = y[0], y[1], y[2]
    
    # Trim to valid
    valid = R > 1.05
    return t_c[valid], R[valid], phi[valid]

def excitation_from_trajectory(t_arr, R_arr, phi_arr, n, L):
    """
    Compute QNM excitation amplitude for mode n from the trajectory.
    
    The quadrupole moment for a collapsing shell with angular momentum:
    Q_total(t) = R(t)² [m=0 contribution, axisymmetric compression]
    
    For m=±2 (orbital modulation), Q₂₂ ∝ R² cos(2φ)
    
    Excitation = |∫ Q̈(t) e^{iωt} dt|²
    
    We use integration by parts to avoid numerical differentiation:
    ∫ Q̈ e^{iωt} dt = -ω² ∫ Q e^{iωt} dt  (for Q → 0 at boundaries)
    
    So |A_n|² ∝ ω⁴ |∫ Q(t) e^{iωt} dt|²
    """
    omega_R = qnm_freq(n).real
    
    # Q = R² (m=0, dominant for radial collapse)
    Q = R_arr**2
    
    # Add m=2 contribution if L > 0
    if L > 0.01:
        Q_m2 = R_arr**2 * np.cos(2*phi_arr)
        Q = Q + Q_m2  # total quadrupole
    
    # Numerical integration using trapezoidal rule
    dt = np.diff(t_arr)
    t_mid = 0.5*(t_arr[:-1] + t_arr[1:])
    Q_mid = 0.5*(Q[:-1] + Q[1:])
    
    re_int = np.sum(Q_mid * np.cos(omega_R * t_mid) * dt)
    im_int = np.sum(Q_mid * np.sin(omega_R * t_mid) * dt)
    
    # Excitation ∝ ω⁴ |∫Q e^{iωt}|²
    return omega_R**4 * (re_int**2 + im_int**2)

def extract_gamma(excitations, n_modes=25):
    """Fit power law to excitation amplitudes: w_n ∝ (n+1)^(-γ)"""
    ns = np.arange(n_modes)
    exc = np.array(excitations[:n_modes])
    if exc[0] <= 0: return np.nan
    exc_norm = exc / exc[0]
    mask = (ns >= 1) & (exc_norm > 1e-15) & np.isfinite(exc_norm)
    if np.sum(mask) < 3: return np.nan
    log_exc = np.log(exc_norm[mask])
    log_n = np.log(ns[mask] + 1)
    slope, _ = np.polyfit(log_n, log_exc, 1)
    return -slope

# ============================================================
# First: reproduce gamma_fine.py result at L = 0
# ============================================================
print("\n--- Verification: L = 0, comparing with gamma_fine.py ---")
n_modes = 25

# Original Newtonian cycloid result from gamma_fine.py
from subprocess import run as subprocess_run
# (We'll compare manually)

for R0 in [3.0, 3.2, 3.4, 3.6, 4.0]:
    t_arr, R_arr, phi_arr = compute_trajectory(R0, L=0, eps=0.001)
    exc = [excitation_from_trajectory(t_arr, R_arr, phi_arr, n, 0) for n in range(n_modes)]
    g = extract_gamma(exc)
    print(f"  R0={R0:.1f}, GR L=0: γ = {g:.4f}, final_R = {R_arr[-1]:.3f}, t_max = {t_arr[-1]:.1f}")

# ============================================================
# Main: scan L from 0 to L_ISCO at R₀ = ISCO
# ============================================================
print(f"\n{'='*60}")
print(f"L scan at R₀ = {R_ISCO} (ISCO)")
print(f"{'='*60}")
print(f"{'L':>6s} {'L/L_ISCO':>10s} {'γ':>8s} {'|γ-1.04|':>8s} {'orbits':>8s}")
print("-"*50)

L_scan = np.linspace(0, L_ISCO*0.999, 15)
results_L = []

for L_test in L_scan:
    try:
        t_arr, R_arr, phi_arr = compute_trajectory(R_ISCO, L=L_test, eps=0.005)
        if len(t_arr) < 100: 
            results_L.append((L_test, np.nan, 0))
            continue
        exc = [excitation_from_trajectory(t_arr, R_arr, phi_arr, n, L_test) 
               for n in range(n_modes)]
        g = extract_gamma(exc)
        orbits = phi_arr[-1]/(2*np.pi)
        results_L.append((L_test, g, orbits))
        print(f"{L_test:6.3f} {L_test/L_ISCO:10.3f} {g:8.4f} {abs(g-1.04):8.4f} {orbits:8.2f}")
    except Exception as e:
        results_L.append((L_test, np.nan, 0))
        print(f"{L_test:6.3f} {L_test/L_ISCO:10.3f}  FAILED: {e}")

# ============================================================
# Also scan R₀ at several L values
# ============================================================
print(f"\n{'='*60}")
print(f"R₀ scan at several L values")
print(f"{'='*60}")

for L_fixed in [0, L_ISCO*0.5, L_ISCO*0.8, L_ISCO*0.9, L_ISCO*0.95]:
    print(f"\n  L = {L_fixed:.3f} ({L_fixed/L_ISCO:.2f} × L_ISCO):")
    for R0 in np.arange(3.0, 4.5, 0.2):
        try:
            t_arr, R_arr, phi_arr = compute_trajectory(R0, L=L_fixed, eps=0.005)
            if len(t_arr) < 100: continue
            exc = [excitation_from_trajectory(t_arr, R_arr, phi_arr, n, L_fixed) 
                   for n in range(n_modes)]
            g = extract_gamma(exc)
            marker = " *** " if abs(g - 1.04) < 0.03 else ""
            print(f"    R0={R0:.1f}: γ = {g:.4f} {marker}")
        except:
            pass

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")

valid_results = [(L, g, o) for L, g, o in results_L if not np.isnan(g)]
if valid_results:
    Ls = [r[0] for r in valid_results]
    gs = [r[1] for r in valid_results]
    
    # Find L that gives γ = 1.04
    for i in range(len(gs)-1):
        if (gs[i] - 1.04) * (gs[i+1] - 1.04) < 0:
            L_cross = Ls[i] + (1.04 - gs[i]) * (Ls[i+1] - Ls[i]) / (gs[i+1] - gs[i])
            print(f"\n*** γ = 1.04 at L = {L_cross:.4f} (L/L_ISCO = {L_cross/L_ISCO:.4f}) ***")
            break
    else:
        if gs:
            print(f"\nγ = 1.04 not found in L range. γ ranges from {min(gs):.3f} to {max(gs):.3f}")
            if min(gs) > 1.04:
                print("All γ values ABOVE 1.04 — angular momentum makes γ LARGER, not smaller")
                print("The hypothesis that L_ISCO reduces γ from 1.34 to 1.04 is WRONG")
            elif max(gs) < 1.04:
                print("All γ values BELOW 1.04")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
if valid_results:
    ax.plot([r[0] for r in valid_results], [r[1] for r in valid_results], 'bo-', lw=2, ms=8)
ax.axhline(1.04, color='gold', ls='--', lw=2, label='Target γ=1.04')
ax.axvline(L_ISCO, color='green', ls='--', lw=1.5, alpha=0.5, label=f'L_ISCO={L_ISCO:.3f}')
ax.set_xlabel('L (2M=1 units)', fontsize=13)
ax.set_ylabel('γ', fontsize=13)
ax.set_title(f'γ vs Angular Momentum at R₀ = ISCO', fontsize=13, fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
# Show a sample trajectory
t_demo, R_demo, phi_demo = compute_trajectory(R_ISCO, L=L_ISCO*0.9, eps=0.005)
x_demo = R_demo * np.cos(phi_demo)
y_demo = R_demo * np.sin(phi_demo)
ax.plot(x_demo, y_demo, 'b-', lw=1)
th = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(th), np.sin(th), 'k-', lw=2, label='Horizon')
ax.plot(1.5*np.cos(th), 1.5*np.sin(th), 'k:', alpha=0.5, label='Photon sphere')
ax.set_aspect('equal'); ax.legend(); ax.grid(alpha=0.3)
ax.set_title(f'Trajectory: R₀=ISCO, L=0.9×L_ISCO', fontweight='bold')

plt.tight_layout()
plt.savefig('gamma_rotating.png', dpi=150)
print("\nSaved: gamma_rotating.png")
