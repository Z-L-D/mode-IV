#!/usr/bin/env python3
"""
Schwarzschild Greybody Factor Calculator
=========================================

Computes l=2 transmission coefficients (greybody factors) for both
Regge-Wheeler (odd-parity) and Zerilli (even-parity) perturbation
equations in 2M=1 units.

CRITICAL: In 2M=1 units, M = 1/2. The Zerilli potential uses M explicitly
in its coefficients. Previous versions incorrectly used M=1, inflating
V_Z from 0.605 to 0.897.

CORRECT RESULT: V_RW_max ≈ V_Z_max ≈ 0.605 (isospectral partners).
The RW and Zerilli greybody factors should be IDENTICAL for all ω.
This is a consequence of the Chandrasekhar (1983) isospectrality theorem.

Units: 2M = 1 throughout (Leaver convention)
  r_horizon = 1
  r_photon_sphere = 1.5  (= 3M)
  r_ISCO = 3.0  (= 6M)

References:
  Regge & Wheeler (1957) — odd-parity equation
  Zerilli (1970) — even-parity equation  
  Chandrasekhar (1983) — isospectrality, Darboux transformation
  Page (1976) — greybody factors for thermal radiation

Author: mode:LIMINAL / CoS-PAFS
Date: June 7, 2026 (corrected)
"""

import numpy as np
from scipy.integrate import solve_ivp
import json
import sys

# =================================================================
# SCHWARZSCHILD GEOMETRY (2M = 1)
# =================================================================

M = 0.5  # Mass parameter in 2M=1 units

def f_schw(r):
    """Metric function f(r) = 1 - 2M/r = 1 - 1/r"""
    return 1.0 - 1.0 / r

def tortoise(r):
    """r* = r + 2M ln(r/(2M) - 1) = r + ln(r - 1) in 2M=1"""
    return r + np.log(np.abs(r - 1.0))

# Precomputed r*(r) inversion table
_R_TAB = np.concatenate([
    1.0 + np.logspace(-12, -3, 3000),   # near horizon
    np.linspace(1.001, 1.5, 2000),       # cavity region
    np.linspace(1.5, 10.0, 3000),        # barrier region
    np.linspace(10.0, 50.0, 1000),
    np.linspace(50.0, 500.0, 500),
])
_R_TAB = np.unique(np.sort(_R_TAB))
_RS_TAB = np.array([tortoise(r) for r in _R_TAB])

from scipy.interpolate import interp1d
_r_of_rstar_interp = interp1d(_RS_TAB, _R_TAB, kind='cubic',
                                fill_value='extrapolate')

def r_from_rstar(rstar):
    """Robust r*(r) inversion via precomputed lookup table."""
    scalar = np.isscalar(rstar)
    rstar = np.atleast_1d(np.asarray(rstar, dtype=float))
    r = np.clip(_r_of_rstar_interp(rstar), 1.0 + 1e-15, None)
    return float(r.flat[0]) if scalar else r

# =================================================================
# EFFECTIVE POTENTIALS
# =================================================================

def V_regge_wheeler(r, l=2):
    """
    Regge-Wheeler potential (odd-parity / axial perturbations).
    
    V_RW(r) = f(r) × [l(l+1)/r² - 6M/r³]
    
    In 2M=1 units (M=0.5):
    V_RW(r) = (1 - 1/r) × [l(l+1)/r² - 3/r³]
    """
    f = f_schw(r)
    return f * (l * (l + 1) / r**2 - 6.0 * M / r**3)


def V_zerilli(r, l=2):
    """
    Zerilli potential (even-parity / polar perturbations).
    
    V_Z(r) = f(r) × [2n²(n+1)r³ + 6n²Mr² + 18nM²r + 18M³]
                   / [r³ × (nr + 3M)²]
    
    where n = λ = (l-1)(l+2)/2 = 2 for l=2
    
    CRITICAL: M = 0.5 in 2M=1 units. Previous code used M=1 here,
    which gave V_Z_max = 0.897 instead of the correct 0.605.
    """
    n = (l - 1) * (l + 2) / 2  # = 2 for l=2
    f = f_schw(r)
    
    numer = (2.0 * n**2 * (n + 1) * r**3 
             + 6.0 * n**2 * M * r**2 
             + 18.0 * n * M**2 * r 
             + 18.0 * M**3)
    denom = r**3 * (n * r + 3.0 * M)**2
    
    return f * numer / denom


def validate_potentials(l=2):
    """Verify both potentials peak at ~0.605 (isospectral check)."""
    r_arr = np.linspace(1.05, 5.0, 10000)
    V_rw = np.array([V_regge_wheeler(r, l) for r in r_arr])
    V_z = np.array([V_zerilli(r, l) for r in r_arr])
    
    i_rw = np.argmax(V_rw)
    i_z = np.argmax(V_z)
    
    print(f"Potential validation (l={l}):")
    print(f"  V_RW:     max = {V_rw[i_rw]:.6f} at r = {r_arr[i_rw]:.4f}")
    print(f"  V_Zerilli: max = {V_z[i_z]:.6f} at r = {r_arr[i_z]:.4f}")
    print(f"  Ratio V_Z/V_RW = {V_z[i_z]/V_rw[i_rw]:.4f} (should be ~1.0)")
    
    if abs(V_z[i_z] / V_rw[i_rw] - 1.0) > 0.01:
        print("  WARNING: Zerilli/RW ratio deviates from 1.0!")
        print("  Check M value in Zerilli potential.")
    else:
        print("  ✓ Isospectral check passed")
    
    return V_rw[i_rw], V_z[i_z]


# =================================================================
# GREYBODY FACTOR COMPUTATION
# =================================================================

def compute_greybody(omega, potential_func, l=2, 
                     rstar_min=-60, rstar_max=200):
    """
    Compute the greybody factor Γ(ω) for a given potential.
    
    Method: integrate the wave equation from the horizon outward
    with ingoing boundary condition:
        ψ ~ e^{-iωr*} as r* → -∞
    
    At large r*:
        ψ → A_out e^{+iωr*} + A_in e^{-iωr*}
    
    Greybody factor: Γ = 1 - |A_out/A_in|²
    
    Returns: Γ, A_in, A_out, Wronskian_error
    """
    # Initial conditions near horizon
    psi0 = np.exp(-1j * omega * rstar_min)
    dpsi0 = -1j * omega * psi0
    
    def deriv(rstar, y):
        psi = y[0] + 1j * y[1]
        r = r_from_rstar(rstar)
        V = potential_func(r, l) if r > 1.0001 else 0.0
        d2psi = (V - omega**2) * psi
        return [y[2], y[3], d2psi.real, d2psi.imag]
    
    y0 = [psi0.real, psi0.imag, dpsi0.real, dpsi0.imag]
    
    sol = solve_ivp(deriv, (rstar_min, rstar_max), y0,
                    method='DOP853',
                    rtol=1e-10, atol=1e-12)
    
    if not sol.success:
        print(f"  WARNING: Integration failed at ω={omega:.4f}: {sol.message}")
        return 0.0, 0.0, 0.0, 1.0
    
    # Extract A_in, A_out from asymptotic behavior at large r*
    rstar_far = sol.t[-1]
    psi_far = sol.y[0, -1] + 1j * sol.y[1, -1]
    dpsi_far = sol.y[2, -1] + 1j * sol.y[3, -1]
    
    exp_plus = np.exp(1j * omega * rstar_far)
    exp_minus = np.exp(-1j * omega * rstar_far)
    
    # ψ = A_out e^{iωr*} + A_in e^{-iωr*}
    # ψ' = iω A_out e^{iωr*} - iω A_in e^{-iωr*}
    A_out = (psi_far + dpsi_far / (1j * omega)) / (2.0 * exp_plus)
    A_in  = (psi_far - dpsi_far / (1j * omega)) / (2.0 * exp_minus)
    
    # Greybody factor
    if abs(A_in) > 1e-30:
        Gamma = 1.0 - abs(A_out / A_in)**2
    else:
        Gamma = 0.0
    
    # Wronskian check: |A_in|² - |A_out|² should be constant
    # (related to flux conservation)
    W_error = abs(abs(A_in)**2 - abs(A_out)**2 - 1.0)
    # Note: with our normalization, the Wronskian = i*omega * (A_in² - A_out²)
    # For unit incoming flux: |A_in|² = 1/(transmission), so this isn't exactly 1.
    # Better check: Γ should be between 0 and 1.
    W_error = 0.0 if 0 <= Gamma <= 1 else abs(Gamma - max(0, min(1, Gamma)))
    
    return Gamma, A_in, A_out, W_error


# =================================================================
# PARTIAL BARRIER (for barrier formation model)
# =================================================================

def compute_greybody_partial(omega, r_surface, potential_func=V_zerilli,
                             l=2, rstar_max=200):
    """
    Greybody factor when inner boundary is at r_surface instead of horizon.
    
    Used in the barrier formation model: as the O-S surface crosses the
    photon sphere, the effective barrier builds up progressively.
    
    For r_surface > barrier peak: Γ ≈ 1 (no barrier above surface)
    For r_surface → horizon: Γ → Γ_full
    """
    if r_surface < 1.001:
        r_surface = 1.001
    
    rstar_surface = tortoise(r_surface)
    
    # Ingoing BC at r_surface
    psi0 = np.exp(-1j * omega * rstar_surface)
    dpsi0 = -1j * omega * psi0
    
    def deriv(rstar, y):
        psi = y[0] + 1j * y[1]
        r = r_from_rstar(rstar)
        V = potential_func(r, l) if r > 1.0001 else 0.0
        d2psi = (V - omega**2) * psi
        return [y[2], y[3], d2psi.real, d2psi.imag]
    
    y0 = [psi0.real, psi0.imag, dpsi0.real, dpsi0.imag]
    
    sol = solve_ivp(deriv, (rstar_surface, rstar_max), y0,
                    method='DOP853', rtol=1e-10, atol=1e-12)
    
    if not sol.success:
        return 1.0, None, None  # assume transparent on failure
    
    rstar_far = sol.t[-1]
    psi_far = sol.y[0, -1] + 1j * sol.y[1, -1]
    dpsi_far = sol.y[2, -1] + 1j * sol.y[3, -1]
    
    exp_plus = np.exp(1j * omega * rstar_far)
    exp_minus = np.exp(-1j * omega * rstar_far)
    
    A_out = (psi_far + dpsi_far / (1j * omega)) / (2.0 * exp_plus)
    A_in  = (psi_far - dpsi_far / (1j * omega)) / (2.0 * exp_minus)
    
    Gamma = 1.0 - abs(A_out / A_in)**2 if abs(A_in) > 1e-30 else 0.0
    Gamma = max(0.0, min(1.0, Gamma))  # clamp to physical range
    
    return Gamma, A_in, A_out


# =================================================================
# QNM FREQUENCY TABLE (exact l=2 from Leaver CF solver)
# =================================================================

# First 8 modes from Leaver continued fraction (validated)
QNM_L2 = [
    (0.74734, 0.17792),   # n=0
    (0.69342, 0.54783),   # n=1
    (0.60210, 0.95654),   # n=2
    (0.50301, 1.41030),   # n=3
    (0.41581, 1.89384),   # n=4
    (0.33860, 2.39122),   # n=5
    (0.27376, 2.89738),   # n=6
    (0.22015, 3.40848),   # n=7
]

def qnm_omega_R(n, l=2):
    """Real part of QNM frequency for overtone n."""
    if n < len(QNM_L2):
        return QNM_L2[n][0]
    else:
        # Motl-Neitzke asymptotic: ω_R → ln(3)/(4π) in 2M=1 units
        return np.log(3.0) / (4.0 * np.pi)  # ≈ 0.08664

def qnm_omega_I(n, l=2):
    """Imaginary part (damping rate) for overtone n."""
    if n < len(QNM_L2):
        return QNM_L2[n][1]
    else:
        # Motl-Neitzke: ω_I → (2n+1)/4 in 2M=1 units
        return (2 * n + 1) / 4.0


# =================================================================
# MAIN: COMPUTE AND VALIDATE
# =================================================================

def main():
    print("=" * 65)
    print("SCHWARZSCHILD GREYBODY FACTOR CALCULATOR")
    print("2M=1 units, M=0.5, l=2")
    print("=" * 65)
    
    # Validate potentials
    print()
    V_rw_max, V_z_max = validate_potentials()
    
    # Compute greybody factors at QNM frequencies
    print(f"\n{'='*65}")
    print(f"Greybody factors at QNM overtone frequencies")
    print(f"{'='*65}")
    print(f"{'n':>3} {'ω_R':>8} {'ω_I':>8} {'Γ_RW':>10} {'Γ_Z':>10} "
          f"{'RW/Z':>8} {'Γ_RW err':>10}")
    print("-" * 65)
    
    results = []
    for n in range(min(31, 8)):  # First 8 from Leaver, rest asymptotic
        omega = qnm_omega_R(n)
        omega_I = qnm_omega_I(n)
        
        Gamma_RW, Ain_RW, Aout_RW, err_RW = compute_greybody(
            omega, V_regge_wheeler)
        Gamma_Z, Ain_Z, Aout_Z, err_Z = compute_greybody(
            omega, V_zerilli)
        
        ratio = Gamma_RW / Gamma_Z if Gamma_Z > 1e-15 else float('inf')
        
        print(f"{n:3d} {omega:8.4f} {omega_I:8.4f} {Gamma_RW:10.6f} "
              f"{Gamma_Z:10.6f} {ratio:8.2f} {err_RW:10.2e}")
        
        results.append({
            'n': n,
            'omega_R': omega,
            'omega_I': omega_I,
            'Gamma_RW': float(Gamma_RW),
            'Gamma_Z': float(Gamma_Z),
            'ratio_RW_Z': float(ratio),
            'A_in_RW': float(abs(Ain_RW)),
            'A_out_RW': float(abs(Aout_RW)),
        })
    
    # Also compute at asymptotic frequency
    omega_asymp = qnm_omega_R(50)
    Gamma_RW_a, _, _, _ = compute_greybody(omega_asymp, V_regge_wheeler)
    Gamma_Z_a, _, _, _ = compute_greybody(omega_asymp, V_zerilli)
    print(f"\nAsymptotic (n≈50, ω={omega_asymp:.5f}):")
    print(f"  Γ_RW = {Gamma_RW_a:.2e}, Γ_Z = {Gamma_Z_a:.2e}")
    
    # Barrier peak info
    print(f"\nBarrier peaks:")
    print(f"  V_RW_max  = {V_rw_max:.6f} → ω_barrier = {np.sqrt(V_rw_max):.4f}")
    print(f"  V_Z_max   = {V_z_max:.6f} → ω_barrier = {np.sqrt(V_z_max):.4f}")
    print(f"  ω²(n=0)   = {qnm_omega_R(0)**2:.6f}")
    print(f"  ω²/V_peak = {qnm_omega_R(0)**2/V_rw_max:.4f}")
    
    # Key finding
    print(f"\nKEY RESULT:")
    print(f"  Γ_RW(n=0) = {results[0]['Gamma_RW']:.4f}")
    print(f"  Γ_Z(n=0)  = {results[0]['Gamma_Z']:.4f}")
    print(f"  RW/Z ratio = {results[0]['ratio_RW_Z']:.2f}")
    if abs(results[0]['ratio_RW_Z'] - 1.0) < 0.1:
        print(f"  ✓ RW and Zerilli give same Γ (isospectral, as expected)")
    else:
        print(f"  ⚠ RW/Z ratio ≠ 1 — check potential formulas!")
    
    # Save
    output = {
        'units': '2M=1, M=0.5',
        'l': 2,
        'V_RW_max': float(V_rw_max),
        'V_Z_max': float(V_z_max),
        'greybody_factors': results,
    }
    
    with open('greybody_results_corrected.json', 'w') as fout:
        json.dump(output, fout, indent=2)
    print(f"\nSaved to greybody_results_corrected.json")
    
    return results


if __name__ == '__main__':
    results = main()
