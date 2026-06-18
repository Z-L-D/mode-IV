#!/usr/bin/env python3
"""
refit_aqnm.py — analytic A_QNM rescale from a completed class_qnm run.

Reads the same lensed output file and Planck binned files as
compare_all_lensed.py, computes the multiplicative rescale s that
minimizes chi2 (linear-response approximation: Cl scales with A_QNM),
and prints the new qnm_A for the ini.

Lensing makes the true response slightly superlinear in A, so iterate:
apply the printed qnm_A, rerun CLASS, run this again. Converged when
s is within ~0.5% of 1.

Usage:
  python3 refit_aqnm.py [current_qnm_A]
  (default current_qnm_A = 3.33e-8; pass the value actually in the ini)
"""
import sys
import numpy as np
from scipy.interpolate import interp1d

T_CMB = 2.7255e6
A_current = float(sys.argv[1]) if len(sys.argv) > 1 else 3.33e-8

lensed = np.loadtxt('class_qnm/output/qnm_test_100_cl_lensed.dat')
cl_ell = lensed[:, 0]
spectra = {
    "TT": (lensed[:, 1] * T_CMB**2, "planck_tt_binned.txt"),
    "EE": (lensed[:, 2] * T_CMB**2, "planck_ee_binned.txt"),
    "TE": (lensed[:, 3] * T_CMB**2, "planck_te_binned.txt"),
}

# Accumulate over chosen spectra. TT-only is the paper's primary metric;
# joint is reported for information.
def optimal_scale(spec_names):
    num, den = 0.0, 0.0
    for name in spec_names:
        cl_col, planck_file = spectra[name]
        planck = np.loadtxt(planck_file)
        p_ell, p_Dl = planck[:, 0], planck[:, 1]
        p_sig = (planck[:, 2] + planck[:, 3]) / 2
        mask = (p_ell >= 30) & (p_ell >= cl_ell.min()) & (p_ell <= cl_ell.max())
        ell, Dl, sig = p_ell[mask], p_Dl[mask], p_sig[mask]
        q = interp1d(cl_ell, cl_col, kind='cubic')(ell)
        num += np.sum(Dl * q / sig**2)
        den += np.sum(q * q / sig**2)
    return num / den

def chi2_at_scale(spec_name, s):
    cl_col, planck_file = spectra[spec_name]
    planck = np.loadtxt(planck_file)
    p_ell, p_Dl = planck[:, 0], planck[:, 1]
    p_sig = (planck[:, 2] + planck[:, 3]) / 2
    mask = (p_ell >= 30) & (p_ell >= cl_ell.min()) & (p_ell <= cl_ell.max())
    ell, Dl, sig = p_ell[mask], p_Dl[mask], p_sig[mask]
    q = interp1d(cl_ell, cl_col, kind='cubic')(ell)
    return np.sum(((Dl - s * q) / sig)**2) / (len(ell) - 1)

s_tt = optimal_scale(["TT"])
s_joint = optimal_scale(["TT", "EE", "TE"])

print(f"Current qnm_A: {A_current:.6e}")
print(f"\nOptimal rescale (TT only):      s = {s_tt:.5f}")
print(f"Optimal rescale (TT+EE+TE):     s = {s_joint:.5f}")
print(f"\nProjected chi2/dof at TT-optimal scale (linear approx):")
for name in ["TT", "EE", "TE"]:
    print(f"  {name}: {chi2_at_scale(name, s_tt):.3f}  "
          f"(at s=1: {chi2_at_scale(name, 1.0):.3f})")
print(f"\n>>> Set in ini:  qnm_A = {A_current * s_tt:.6e}   (TT-anchored)")
print(f"    (joint-anchored alternative: qnm_A = {A_current * s_joint:.6e})")
print("\nThen rerun CLASS and run this script again. Converged when s ~ 1.")
