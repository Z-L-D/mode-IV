"""
Two-channel (prompt + ringdown) interference model for QNM overtone weights.
Zero free parameters: all inputs are production-measured quantities.

Inputs (all from PQM production state / CPM waveform):
  - Leaver l=2 Schwarzschild eigenvalues (2M=1), n=0..7
  - |A_prompt/A_ring| = 1.63  (measured at fundamental, FFT at omega_R)
  - Delta_phi_0 = -43.7 deg   (measured prompt-ringdown phase at omega_R)
  - Half-period relation: channels anchored at epochs separated by
    Delta_tau = pi / omega_R0  => phase slope pi in (omega_R,n / omega_R,0)
  - Berti-Cardoso 2006 asymptotic excitation factors:
    |B_n| = 1 / (3 sqrt(ln^2 3 + ((2n+1) pi)^2))

Model:
  w_n  proportional to  |A_r,n + A_p e^{i phi_n}|^2 / sigma_n
  A_r,n = |B_n|/|B_0|   (ring channel carries the excitation factor)
  A_p   = 1.63          (prompt channel broadband, flat across low n)
  phi_n = Delta_phi_0 + slope * (omega_R,n/omega_R,0 - 1)
  sigma_n proportional to omega_I,n   (code ground-truth convention)

Pre-registered targets (June 10 master findings):
  H1/H0 = 0.136 (systematic range 0.09-0.15, +/-25%)
  H2/H0 <~ 0.05 (upper bound only)
  |C1|^2/|C0|^2 = 4.5/11.7 = 0.385 (+/- ~27%)
  Band power-law extrapolation refs: 0.486 (n=1), 0.319 (n=2)
  Impulsive (lifetime) refs: 0.325 (n=1)
"""

import numpy as np

# ---------- Leaver l=2 eigenvalues (2M=1), n = 0..7 ----------
leaver = [(0.74734, 0.17792), (0.69342, 0.54783), (0.60211, 0.95654),
          (0.50302, 1.41030), (0.41583, 1.89369), (0.34615, 2.39134),
          (0.29131, 2.89591), (0.24834, 3.40400)]
wR = np.array([m[0] for m in leaver])
wI = np.array([m[1] for m in leaver])
nn = np.arange(len(leaver))

# ---------- measured anchors at n=0 ----------
Rch    = 1.63                      # |A_prompt / A_ring| at fundamental
dphi0  = np.deg2rad(-43.7)         # measured phase difference at omega_R0

# ---------- GATE: reconstruct f = 0.27 from the anchors ----------
tot0   = 1.0 + Rch**2 + 2.0*Rch*np.cos(dphi0)
f_spec = Rch**2 / tot0
eta    = 2.0*Rch*np.cos(dphi0) / tot0
f_rec  = f_spec * (1.0 - eta)
print("=== CONSISTENCY GATE (must reproduce production values) ===")
print(f"  f_spectral = {f_spec:.4f}   (production: 0.44)")
print(f"  eta_cross  = {eta:.4f}   (production: 0.39)")
print(f"  f          = {f_rec:.4f}   (production: 0.27)")
print()

# ---------- Berti-Cardoso asymptotic excitation factors ----------
Bn = 1.0 / (3.0*np.sqrt(np.log(3.0)**2 + ((2*nn+1)*np.pi)**2))
Ar = Bn / Bn[0]                     # ring amplitudes normalized to n=0

def run_model(slope, sigma_mode='wI', Ar_mode='BC'):
    """slope: phase accumulated per unit (omega_R/omega_R0), in radians."""
    if Ar_mode == 'BC':
        A_r = Ar
    elif Ar_mode == 'simple':                  # pure 1/(2n+1)
        A_r = 1.0/(2*nn+1.0)
    phi  = dphi0 + slope*(wR/wR[0] - 1.0)
    amp2 = A_r**2 + Rch**2 + 2.0*A_r*Rch*np.cos(phi)
    sig  = wI if sigma_mode == 'wI' else wI/wR
    H    = amp2 / sig
    return H/H[0], amp2/amp2[0], np.rad2deg(phi)

# ---------- reference curves ----------
H_powerlaw  = (nn+1.0)**(-1.04)           # band extrapolation
H_impulsive = wI[0]/wI                    # lifetime weighting

# ---------- primary model: pi slope (half-period relation) ----------
Hr_pi,  C2r_pi,  phi_pi  = run_model(np.pi)
Hr_2pi, C2r_2pi, _       = run_model(2*np.pi)   # rest-anchored variant
Hr_0,   C2r_0,   _       = run_model(0.0)       # no phase evolution
Hr_sR,  _,       _       = run_model(np.pi, sigma_mode='wI_over_wR')

print("=== MODEL A (pi slope, sigma ~ omega_I) vs references ===")
print(f"{'n':>2} {'phi_n(deg)':>10} {'H_n/H_0':>9} {'powerlaw':>9} "
      f"{'impulsive':>9} {'measured':>20}")
meas = {1: '0.136 (0.09-0.15)', 2: '<~0.05 (bound)'}
for i in nn:
    m = meas.get(i, '--  (unmeasured)')
    print(f"{i:>2} {phi_pi[i]:>10.1f} {Hr_pi[i]:>9.4f} {H_powerlaw[i]:>9.4f} "
          f"{H_impulsive[i]:>9.4f} {m:>20}")
print()

print("=== |C_n|^2/|C_0|^2 (amplitude-squared ratios, no sigma) ===")
print(f"  n=1: model {C2r_pi[1]:.3f}   measured 0.385 +/- ~0.10")
print(f"  n=2: model {C2r_pi[2]:.3f}   measured <~ 0.103")
print()

# ---------- suppression accounting ----------
def frac_captured(model_val, pl_val, meas_val):
    need = np.log(pl_val/meas_val)
    got  = np.log(pl_val/model_val)
    return got/need

print("=== SUPPRESSION CAPTURED (log-scale fraction of PL -> measured) ===")
print(f"  n=1: {100*frac_captured(Hr_pi[1], H_powerlaw[1], 0.136):.0f}% "
      f"(model {Hr_pi[1]:.3f}, PL 0.486, measured 0.136)")
print(f"  n=2: {100*frac_captured(Hr_pi[2], H_powerlaw[2], 0.05):.0f}% "
      f"(model {Hr_pi[2]:.3f}, PL 0.319, bound 0.05)")
print()

# ---------- robustness scan ----------
print("=== ROBUSTNESS: H1/H0 and H2/H0 across assumptions ===")
print(f"  {'variant':<38} {'H1/H0':>7} {'H2/H0':>7}")
rows = [
    ("pi slope, sigma~wI (PRIMARY)",        Hr_pi[1],  Hr_pi[2]),
    ("2pi slope (rest-point anchor)",       Hr_2pi[1], Hr_2pi[2]),
    ("no phase evolution (slope 0)",        Hr_0[1],   Hr_0[2]),
    ("pi slope, sigma~wI/wR (paper conv.)", Hr_sR[1],  Hr_sR[2]),
]
Hr_s, _, _ = run_model(np.pi, Ar_mode='simple')
rows.append(("pi slope, A_r ~ 1/(2n+1) exactly", Hr_s[1], Hr_s[2]))
for name, h1, h2 in rows:
    print(f"  {name:<38} {h1:>7.4f} {h2:>7.4f}")
print()

# ---------- the n=3-7 prediction table (proj_cn.py target) ----------
print("=== PREDICTION: n=3-7 weights (pre-registered for proj_cn.py) ===")
print(f"  {'n':>2} {'H_n/H_0 (pi)':>13} {'H_n/H_0 (2pi)':>14} {'powerlaw':>9}")
for i in range(3, 8):
    print(f"  {i:>2} {Hr_pi[i]:>13.4f} {Hr_2pi[i]:>14.4f} {H_powerlaw[i]:>9.4f}")
