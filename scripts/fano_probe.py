#!/usr/bin/env python3
"""
fano_probe.py — Quick probe to verify P(k) implementation
Run this FIRST. Paste the output back so I can verify alignment
with your CLASS implementation before running the full Fano test.
"""
import numpy as np

# === Parameters from definitive paper outline ===
H0 = 67.36  # km/s/Mpc
c_kms = 299792.458  # km/s
rs = c_kms / H0  # Mpc
k0 = np.pi / rs

gamma = 1.04
beta_geom = 1.0
A_QNM = 2.85e-8
N_modes = 300

# Leaver l=2 gravitational Schwarzschild (omega_I * M, dimensionless)
omega_I_leaver = [0.08896, 0.27392, 0.47828, 0.70514,
                  0.94684, 1.19565, 1.44766, 1.70105]

def omega_I(n):
    if n < 8:
        return omega_I_leaver[n]
    return (2*n + 1) / 8.0  # Motl-Neitzke

# Derived arrays
w = np.array([1.0/(n+1)**gamma for n in range(N_modes)])
w /= w.sum()
kn = np.array([k0 * (n+1) * beta_geom for n in range(N_modes)])
sigma = np.array([omega_I(n) * k0 / (2*np.pi) for n in range(N_modes)])

# Compute P(k) at diagnostic k values
k_test = np.array([0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20])

def pk_at(k_val):
    pk = 0.0
    for n in range(N_modes):
        dk = k_val - kn[n]
        eps2 = (dk / sigma[n])**2
        lorentz = 1.0 / (1.0 + eps2)          # = sigma^2/(dk^2+sigma^2)
        asym = (k_val + kn[n])**2 / kn[n]**2   # P(k->0) = 0 enforcer
        pk += w[n] * lorentz * asym
    return A_QNM * pk

print("=" * 60)
print("QNM P(k) PROBE — Verify against your CLASS implementation")
print("=" * 60)
print(f"r_s     = {rs:.4f} Mpc")
print(f"k_0     = {k0:.8e} Mpc^-1")
print(f"k_1     = {kn[0]:.8e} Mpc^-1")
print(f"k_300   = {kn[-1]:.8e} Mpc^-1")
print(f"sigma_1 = {sigma[0]:.8e} Mpc^-1")
print(f"sigma_8 = {sigma[7]:.8e} Mpc^-1")
print(f"sigma_9 = {sigma[8]:.8e} Mpc^-1  (first asymptotic)")
print(f"sigma_300= {sigma[-1]:.8e} Mpc^-1")
print(f"w_1     = {w[0]:.8e}")
print(f"w_300   = {w[-1]:.8e}")
print(f"\nP(k) at diagnostic points:")
print(f"{'k':>12} {'P(k)':>15}")
print("-" * 30)
for k_val in k_test:
    print(f"{k_val:>12.5f} {pk_at(k_val):>15.6e}")

# Key questions for Zac:
print("\n" + "=" * 60)
print("QUESTIONS — please confirm or correct:")
print("=" * 60)
print("""
1. Does your primordial.c use the Lorentzian as:
   sigma^2 / (dk^2 + sigma^2)   [peak=1]
   or 1/(dk^2 + sigma^2)        [peak=1/sigma^2]?

2. Is there a j_2(k*rs)^2 envelope multiplying the mode sum?
   (Implementation guide had it, definitive outline doesn't)

3. Is beta_geom = 1.0 in your current .ini?

4. For n>=8 asymptotic, is omega_I = (2n+1)/8 correct?
   Or is it (2n+1)/(8M) with M != 1?

5. Do the P(k) values above match what CLASS computes
   when you print P(k) at these same k values?
""")
