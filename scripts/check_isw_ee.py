#!/usr/bin/env python3
"""
check_isw_ee.py — differential verification of the late-ISW removal run.

Usage: python3 check_isw_ee.py BASELINE_CL_FILE NOISW_CL_FILE

Pre-registered expectations (both runs at the SAME qnm_A):
  EE : IDENTICAL to output precision (flag touches T sources only).
       PASS if max rel dev < 1e-10; FAIL otherwise -> test invalid.
  TE : near-identical above the mask (T-side cross term only).
       Expect max rel dev (ell>=30) at the sub-percent level.
  TT : large deficit at ell <~ 10-20 (late ISW removed), shrinking
       through ell ~ 30-60, small above. This is the signature.
"""
import sys
import numpy as np

base = np.loadtxt(sys.argv[1])
noisw = np.loadtxt(sys.argv[2])

assert base.shape == noisw.shape, \
    f"shape mismatch: {base.shape} vs {noisw.shape} — different lmax/format?"
assert np.array_equal(base[:, 0], noisw[:, 0]), "ell grids differ"

ell = base[:, 0]
cols = {"TT": 1, "EE": 2, "TE": 3}

def reldev(c):
    a, b = base[:, c], noisw[:, c]
    denom = np.maximum(np.abs(a), 1e-300)
    return np.abs(b - a) / denom

print(f"{'spectrum':>8s} {'max rel dev (all ell)':>22s} "
      f"{'max rel dev (ell>=30)':>22s}")
devs = {}
for name, c in cols.items():
    d = reldev(c)
    devs[name] = d
    print(f"{name:>8s} {d.max():22.3e} {d[ell >= 30].max():22.3e}")

ee = devs["EE"].max()
print()
if ee < 1e-10:
    print(f"EE IDENTITY: PASS (max rel dev {ee:.1e})")
elif ee < 1e-6:
    print(f"EE IDENTITY: MARGINAL ({ee:.1e}) — output rounding? "
          f"verify both runs used identical ini apart from the flag.")
else:
    print(f"EE IDENTITY: *** FAIL *** ({ee:.1e}) — the contributions "
          f"flag changed more than temperature sources. TEST INVALID; "
          f"do not interpret TT/S_1/2 from this run.")

print("\nTT late-ISW removal signature (Delta D_ell / D_ell, baseline ref):")
print(f"{'ell':>5s} {'D_base':>12s} {'D_noisw':>12s} {'rel change':>11s}")
for l in [2, 5, 10, 20, 30, 50, 100, 500, 1500]:
    i = np.argmin(np.abs(ell - l))
    a, b = base[i, 1], noisw[i, 1]
    print(f"{int(ell[i]):5d} {a:12.5e} {b:12.5e} {(b-a)/a:11.4f}")

te = devs["TE"][ell >= 30].max()
print(f"\nTE (ell>=30) max rel dev: {te:.3e} "
      f"({'as expected, small' if te < 0.05 else 'LARGER than expected — inspect'})")
print("\nIf EE passed: proceed to refit/compare/s_half on the no-ISW file.")
