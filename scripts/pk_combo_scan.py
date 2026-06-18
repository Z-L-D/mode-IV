#!/usr/bin/env python3
"""Find optimal continuum + gamma combination."""
import numpy as np

omega_I_leaver = [0.08896, 0.27392, 0.47828, 0.70514, 0.95, 1.20, 1.45, 1.70310]
k0 = np.pi / 4450.6
A_qnm, A_s, n_s = 2.85e-8, 2.1e-9, 0.9649
D_A = 14000

def qnm_pk(k, gamma_val, continuum):
    w_sum = sum(1.0/(n+1)**gamma_val for n in range(300))
    mode_sum = 0.0
    for n in range(300):
        kn = k0*(n+1)
        omega_I_n = omega_I_leaver[n] if n < 8 else (2.0*n+1)/8.0
        sigma_n = omega_I_n * k0 / (2*np.pi)
        w_n = (1.0/(n+1)**gamma_val) / w_sum
        dk = k - kn
        lorentz = sigma_n**2 / (dk**2 + sigma_n**2)
        asym = (k+kn)**2 / kn**2
        mode_sum += w_n * lorentz * asym
    mode_sum += continuum
    return A_qnm * mode_sum

def lcdm_pk(k):
    return A_s * (k/0.05)**(n_s - 1)

# Target: ratio = 1.0 at ALL ell values simultaneously
ell_targets = [220, 540, 810, 1000, 1500, 2000, 2500]
k_targets = [ell/D_A for ell in ell_targets]

print(f"{'gamma':>7s} {'cont':>7s} |", end="")
for ell in ell_targets:
    print(f" ℓ={ell:>4d}", end="")
print(f" | {'score':>7s}  {'peak_err':>8s}")
print("-" * 90)

best_score = 999
best_params = None

for gamma_val in np.arange(1.00, 1.06, 0.005):
    for cont in np.arange(0.000, 0.010, 0.001):
        ratios = []
        for kt in k_targets:
            r = qnm_pk(kt, gamma_val, cont) / lcdm_pk(kt)
            ratios.append(r)
        
        # Peak error: max deviation at ℓ=220,540,810
        peak_err = max(abs(r-1) for r in ratios[:3])
        # High-ell error: mean deviation at ℓ=1500,2000,2500
        high_err = np.mean([abs(r-1) for r in ratios[4:]])
        # Total score
        score = peak_err + high_err
        
        if score < best_score:
            best_score = score
            best_params = (gamma_val, cont)
        
        if abs(gamma_val - 1.04) < 0.001 or score < 0.02:
            vals = " ".join(f" {r:6.4f}" for r in ratios)
            print(f"{gamma_val:7.3f} {cont:7.4f} |{vals} | {score:7.4f}  {peak_err:8.4f}")

gamma_best, cont_best = best_params
print(f"\nBEST: gamma={gamma_best:.3f}, continuum={cont_best:.4f}, score={best_score:.4f}")

# Show best result in detail
print(f"\nBest combo ratios:")
for ell, kt in zip(ell_targets, k_targets):
    r = qnm_pk(kt, gamma_best, cont_best) / lcdm_pk(kt)
    print(f"  ℓ={ell}: P_QNM/P_LCDM = {r:.4f} ({(r-1)*100:+.2f}%)")

# What this means for CLASS
print(f"\nTo implement in CLASS primordial.c:")
print(f"  1. Change gamma from 1.04 to {gamma_best:.3f} in qnm_test.ini")
print(f"  2. Add 'mode_sum += {cont_best:.4f};' after the mode loop")
print(f"     This represents the branch cut (non-resonant) contribution")
