#!/usr/bin/env python3
"""
parity_readout.py — even/odd parity statistic of the QNM model's own
low-ell TT spectrum, read from existing CLASS lensed output files.

  R^TT(lmax) = mean[ D_ell, even ell in 2..lmax ] /
               mean[ D_ell, odd  ell in 3..lmax ]

Planck observes an ODD-multipole excess: R^TT < 1 at lmax ~ 20-30
(point-parity anomaly, ~2-3 sigma depending on estimator/mask).

PRE-REGISTRATION (decided before reading any number):
  Naive comb placement puts teeth at ell ~ 10.05*(n+1) = 10, 20, 30...
  i.e. near EVEN multipoles -> naive expectation R > 1: the WRONG SIGN
  versus Planck. Projection tails (j_ell skews power below k*chi) and
  profile asymmetry (delta < 0) could flip this. This is a risk-bearing
  test: R substantially > 1 is a real tension to log honestly; R < 1
  is an unforced parity prediction. Either outcome is information.

Usage:
  python3 parity_readout.py FILE1 [FILE2 ...]
e.g.
  python3 parity_readout.py \
    class_qnm/output/baseline_for_isw_test.dat \
    $(ls -t class_qnm/output/qnm_noisw_*_cl_lensed.dat | head -1) \
    $(ls -t class_qnm/output/qnm_combined_*_cl_lensed.dat | head -1)
"""
import sys
import numpy as np

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

files = sys.argv[1:]
data = {}
for f in files:
    d = np.loadtxt(f)
    ell = d[:, 0].astype(int)
    Dl = d[:, 1]                      # ell(ell+1)C_ell/2pi (units cancel in R)
    data[f] = dict(zip(ell, Dl))

short = [f.split('/')[-1][:28] for f in files]

print("=" * 70)
print("R^TT(lmax) = <D_even>/<D_odd>,  ell in [2, lmax]")
print("Planck qualitative reference: R < 1 (odd excess), strongest lmax~20-30")
print("=" * 70)
print(f"{'lmax':>5s} " + " ".join(f"{s:>28s}" for s in short))
for lmax in range(3, 62):
    row = []
    for f in files:
        ev = [data[f][l] for l in range(2, lmax + 1, 2) if l in data[f]]
        od = [data[f][l] for l in range(3, lmax + 1, 2) if l in data[f]]
        row.append(np.mean(ev) / np.mean(od) if ev and od else np.nan)
    flag = "  <-- Planck's strongest range" if 20 <= lmax <= 30 else ""
    print(f"{lmax:5d} " + " ".join(f"{r:28.4f}" for r in row) + flag)

print("\n" + "=" * 70)
print("Low-ell D_ell (first file), comb teeth expected near ell ~ 10(n+1):")
print("=" * 70)
f0 = files[0]
for l in range(2, 43):
    if l in data[f0]:
        tooth = ""
        for n in range(5):
            if abs(l - 10.05 * (n + 1)) < 0.6:
                tooth = f"   <-- comb tooth n={n}"
        bar = "#" * int(60 * data[f0][l] / max(data[f0][m]
                        for m in range(2, 43) if m in data[f0]))
        print(f"  l={l:3d}  {data[f0][l]:.4e}  {bar}{tooth}")

print("\nREADING GUIDE:")
print(" - R(lmax) persistently > 1.1 over lmax 15-35: even excess, wrong")
print("   sign vs Planck -> log as tension; boundary calc gains a target.")
print(" - R(lmax) persistently < 0.9: unforced parity prediction -> the")
print("   model natively prefers odd multipoles; strong paper material.")
print(" - R oscillating about 1 with ~10-multipole period: comb teeth")
print("   visible but parity-neutral; the anomaly then needs the x0")
print("   ell<->ell+1 leakage mechanism rather than native placement.")
