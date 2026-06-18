#!/bin/bash
# generate_v3_weights.sh
# Generates qnm_weights_v3.dat from the two-channel interference model.
# Headerless format (fscanf-safe). One H_n/H_0 per line, n=0..300.
# For n=0-7: two-channel model values. For n>=8: power-law (n+1)^{-1.04}.
#
# Usage: bash generate_v3_weights.sh [output_path]
# Default output: ~/qnm-boltzmann/qnm_weights_v3.dat

OUTPATH="${1:-$HOME/qnm-boltzmann/qnm_weights_v3.dat}"

python3 - "$OUTPATH" << 'PYEOF'
import sys, numpy as np

outpath = sys.argv[1]

# Leaver l=2 eigenvalues (2M=1), n=0..7
leaver = [(0.74734,0.17792),(0.69342,0.54783),(0.60211,0.95654),
          (0.50302,1.41030),(0.41583,1.89369),(0.34615,2.39134),
          (0.29131,2.89591),(0.24834,3.40400)]
wR = np.array([m[0] for m in leaver])
wI = np.array([m[1] for m in leaver])
nn = np.arange(len(leaver))

# Measured CPM anchors
Rch   = 1.63
dphi0 = np.deg2rad(-43.7)
slope = np.pi

# Berti-Cardoso excitation factors
Bn = 1.0 / (3.0*np.sqrt(np.log(3.0)**2 + ((2*nn+1)*np.pi)**2))
Ar = Bn / Bn[0]

# Two-channel model
phi  = dphi0 + slope*(wR/wR[0] - 1.0)
amp2 = Ar**2 + Rch**2 + 2.0*Ar*Rch*np.cos(phi)
H    = amp2 / wI
H    = H / H[0]  # normalize to n=0

# Write 301 weights: model at low n, power law at high n, smooth crossover
# The model gives H[7] ~ 0.021 but power law at n=8 gives ~ 0.102.
# Sharp handoff would create a Boltzmann artifact.
# Strategy: loglinear crossover from model to PL over n = 5..12
# (Schroeder scale: modes resolved below n~6, merged by n~12)
gamma = 1.04
PL = np.array([(i+1)**(-gamma) for i in range(301)])

weights = np.zeros(301)
n_lo, n_hi = 5, 12  # crossover zone
for i in range(301):
    if i < n_lo:
        weights[i] = H[i] if i < 8 else PL[i]
    elif i > n_hi:
        weights[i] = PL[i]
    else:
        # smooth blend in log space
        t = (i - n_lo) / (n_hi - n_lo)  # 0 at n_lo, 1 at n_hi
        t = 0.5 * (1 - np.cos(np.pi * t))  # cosine ease
        h_model = H[i] if i < 8 else H[7] * (H[7]/H[6])**(i - 7)
        log_blend = (1 - t) * np.log(h_model) + t * np.log(PL[i])
        weights[i] = np.exp(log_blend)

with open(outpath, 'w') as f:
    for i in range(301):
        f.write(f"{i} {weights[i]:.10e}\n")

print(f"Wrote {outpath}")
print(f"  n=0: {weights[0]:.6f}")
print(f"  n=1: {weights[1]:.6f}  (PL would be {PL[1]:.6f})")
print(f"  n=2: {weights[2]:.6f}  (PL would be {PL[2]:.6f})")
print(f"  n=5: {weights[5]:.6f}  (crossover start)")
print(f"  n=7: {weights[7]:.6f}  (PL: {PL[7]:.6f})")
print(f"  n=8: {weights[8]:.6f}  (PL: {PL[8]:.6f})")
print(f"  n=12: {weights[12]:.6f}  (crossover end, PL: {PL[12]:.6f})")
print(f"  n=13: {weights[13]:.6f}  (pure PL: {PL[13]:.6f})")
PYEOF

echo "Weight file ready: $OUTPATH"
