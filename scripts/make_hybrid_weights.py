#!/usr/bin/env python3
"""
make_hybrid_weights.py — build weight tables for the CPM low-n test.

Baseline shape: w_n = (n+1)^-1.04 for all n (identical to qnm_gamma=1.04).
V1: replace n=1 with the CPM-measured height ratio  H1/H0 = 0.136.
V2: V1 plus the n=2 upper bound                     H2/H0 = 0.050.
Everything n>=3 stays exactly baseline, so the acoustic band is
untouched by construction and the test isolates ell<~30 / S_1/2.

Measured values from extract_overtones.py (June 10, 2026 run):
  |C0|^2 = 11.73 +- 0.13   -> H0 = 65.9   (anchor)
  |C1|^2 =  4.5  +- 1.2    -> H1/H0 = 0.136 (range 0.09-0.15)
  |C2|^2 <~ 1.2  (bound)   -> H2/H0 <~ 0.05

Sanity guard: verifies n>=3 entries match (n+1)^-1.04 exactly and that
the files have N_MODES entries before writing.
"""
import numpy as np

N_MODES = 300
GAMMA = 1.04

n = np.arange(N_MODES)
base = (n + 1.0) ** -GAMMA

variants = {
    "v1": {1: 0.136},
    "v2": {1: 0.136, 2: 0.050},
}

for name, repl in variants.items():
    w = base.copy()
    for idx, val in repl.items():
        w[idx] = val * w[0]          # ratios are relative to H0
    assert np.allclose(w[3:], base[3:]), "band contaminated!"
    assert len(w) == N_MODES
    fn = f"qnm_weights_{name}.dat"
    np.savetxt(fn, np.column_stack([n, w]), fmt=["%d", "%.10e"],
               header=f"CPM low-n hybrid {name}: " +
                      ", ".join(f"H{k}/H0={v}" for k, v in repl.items()) +
                      f"; n>=3 = (n+1)^-{GAMMA}")
    changed = ", ".join(f"w[{k}]: {base[k]:.4f} -> {w[k]:.4f}"
                        for k in repl)
    print(f"{fn}: {changed}")

print("\nFiles written. Reminder: ini paths are relative to class_qnm/,")
print("so use qnm_weight_file = ../qnm_weights_v1.dat")
