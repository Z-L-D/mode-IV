#!/usr/bin/env python3
"""
s_half_check.py — compute C(theta) and S_1/2 from a class_qnm lensed
output file, for relative comparison between baseline and lifetime runs.

S_1/2 = integral over cos(theta) in [-1, 1/2] of C(theta)^2.

IMPORTANT: conventions (l_max, lensed vs unlensed, pixel window) shift
the absolute number. If the original script that produced
S_1/2 = 7,661 muK^4 exists, use THAT for the paper number and use this
one only for the baseline-vs-lifetime RATIO, computed identically on
both outputs.

Usage:
  python3 s_half_check.py [path_to_cl_lensed.dat] [l_max]
  defaults: class_qnm/output/qnm_test_100_cl_lensed.dat, l_max=100
"""
import sys
import numpy as np
from numpy.polynomial.legendre import legval

T_CMB = 2.7255e6

path = sys.argv[1] if len(sys.argv) > 1 else \
    'class_qnm/output/qnm_test_100_cl_lensed.dat'
lmax = int(sys.argv[2]) if len(sys.argv) > 2 else 100

d = np.loadtxt(path)
ell = d[:, 0].astype(int)
# CLASS outputs D_l = l(l+1)C_l/2pi (dimensionless); convert to C_l in muK^2
Dl_tt = d[:, 1] * T_CMB**2
Cl = 2.0 * np.pi * Dl_tt / (ell * (ell + 1.0))

m = ell <= lmax
ell, Cl = ell[m], Cl[m]

# C(theta) = sum_l (2l+1)/(4pi) C_l P_l(cos theta), l >= 2
coeffs = np.zeros(lmax + 1)
for l, c in zip(ell, Cl):
    if l >= 2:
        coeffs[l] = (2 * l + 1) / (4 * np.pi) * c

x = np.linspace(-1.0, 0.5, 4001)          # cos(theta) in [-1, 1/2]
C_theta = legval(x, coeffs)
S_half = np.trapezoid(C_theta**2, x)

print(f"File:  {path}")
print(f"l_max: {lmax}")
print(f"S_1/2 = {S_half:,.0f} muK^4")
print(f"(Planck ~1,065; LCDM ~35,000-48,000; prior QNM baseline 7,661 "
      f"-- compare RATIOS across runs with this script, not absolutes)")
