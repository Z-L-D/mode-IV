#!/usr/bin/env python3
"""
C(theta) from CLASS C_ell output — PRECISE VERSION
Run this in your class_qnm directory after generating C_ell files.

Usage: python3 c_theta_from_class.py qnm_cl.dat lcdm_cl.dat
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre

def load_class_cl(filepath):
    """Load CLASS C_ℓ output. Returns (ell, D_ell) arrays."""
    data = np.loadtxt(filepath, comments='#')
    ell = data[:, 0].astype(int)
    # CLASS outputs D_ℓ = ℓ(ℓ+1)C_ℓ/(2π) in μK²
    Dl = data[:, 1]  # TT
    Cl = Dl * 2 * np.pi / (ell * (ell + 1))
    return ell, Cl, Dl

def compute_c_theta(ell, Cl, theta_arr):
    c_theta = np.zeros_like(theta_arr)
    for i, theta in enumerate(theta_arr):
        cos_t = np.cos(np.radians(theta))
        for j, l in enumerate(ell):
            if l < 2: continue
            Pl = legendre(l)(cos_t)
            c_theta[i] += (2*l + 1) / (4*np.pi) * Cl[j] * Pl
    return c_theta

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 c_theta_from_class.py qnm_cl.dat lcdm_cl.dat")
        sys.exit(1)
    
    ell_q, Cl_q, Dl_q = load_class_cl(sys.argv[1])
    ell_l, Cl_l, Dl_l = load_class_cl(sys.argv[2])
    
    theta = np.linspace(1, 179, 300)
    
    # Use only ℓ < 100 for large-angle correlation
    mask_q = ell_q <= 100
    mask_l = ell_l <= 100
    
    ct_q = compute_c_theta(ell_q[mask_q], Cl_q[mask_q], theta)
    ct_l = compute_c_theta(ell_l[mask_l], Cl_l[mask_l], theta)
    
    plt.figure(figsize=(10, 6))
    plt.plot(theta, ct_l, "r-", lw=2, label="LCDM")
    plt.plot(theta, ct_q, "b-", lw=2, label="QNM")
    plt.axhline(0, color="gray", alpha=0.3)
    plt.axvline(60, color="green", ls="--", alpha=0.5, label="60 deg")
    plt.xlabel("theta (degrees)")
    plt.ylabel("C(theta)")
    plt.title("Angular Correlation Function from CLASS output")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("c_theta_class.png", dpi=150)
    print("Saved: c_theta_class.png")
