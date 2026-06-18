import numpy as np

omega_I_leaver = [0.08896, 0.27392, 0.47828, 0.70514, 0.95, 1.20, 1.45, 1.70310]
k0 = np.pi / 4450.6
A_qnm, A_s, n_s = 2.85e-8, 2.1e-9, 0.9649
D_A = 14000

def qnm_pk(k, gamma_val=1.04, cont=0.0, branch_amp=0.0, branch_pow=2.0):
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
    # Branch cut: grows with k (non-resonant GF contribution)
    mode_sum += cont + branch_amp * (k/k0)**branch_pow / (1 + (k/k0)**branch_pow)
    return A_qnm * mode_sum

def lcdm_pk(k):
    return A_s * (k/0.05)**(n_s - 1)

ell_targets = [220, 540, 810, 1000, 1500, 2000, 2500]
k_targets = [ell/D_A for ell in ell_targets]

print("Fine scan: gamma + branch cut (saturating power law)")
print(f"{'gamma':>7s} {'br_amp':>7s} {'br_pow':>7s} |", end="")
for ell in ell_targets:
    print(f"  {ell:>5d}", end="")
print(f" |  score")
print("-" * 85)

best_score = 999
best_params = None

for gamma_val in np.arange(1.020, 1.045, 0.005):
    for br_amp in [0.0, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01]:
        for br_pow in [0.5, 1.0, 1.5, 2.0]:
            if br_amp == 0 and br_pow > 0.5:
                continue
            ratios = []
            for kt in k_targets:
                r = qnm_pk(kt, gamma_val, 0, br_amp, br_pow) / lcdm_pk(kt)
                ratios.append(r)
            
            peak_err = max(abs(r-1) for r in ratios[:3])
            all_err = np.mean([abs(r-1) for r in ratios])
            
            if peak_err > 0.02:
                continue
            
            if all_err < best_score:
                best_score = all_err
                best_params = (gamma_val, br_amp, br_pow)
            
            vals = " ".join(f" {r:6.4f}" for r in ratios)
            print(f"{gamma_val:7.3f} {br_amp:7.4f} {br_pow:7.1f} |{vals} | {all_err:7.4f}")

if best_params:
    g, ba, bp = best_params
    print(f"\nBEST: gamma={g:.3f}, branch_amp={ba:.4f}, branch_pow={bp:.1f}")
    print(f"\nFor CLASS primordial.c, after the mode loop add:")
    print(f"  mode_sum += {ba} * pow(k/k0, {bp}) / (1.0 + pow(k/k0, {bp}));")
    print(f"And set gamma={g:.3f}")
    
    print(f"\nDetailed ratios:")
    for ell, kt in zip(ell_targets, k_targets):
        r = qnm_pk(kt, g, 0, ba, bp) / lcdm_pk(kt)
        print(f"  ℓ={ell}: {r:.4f} ({(r-1)*100:+.2f}%)")
