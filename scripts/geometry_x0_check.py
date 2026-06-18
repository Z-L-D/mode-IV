#!/usr/bin/env python3
"""
geometry_x0_check.py — single-displacement consistency test.
Question: are (1) Planck's hemispherical dipole-modulation direction and
(2) the Cold Spot's offset from the ideal Y20 nodal ring consistent
projections of ONE observer displacement x0?

Model (first order in eps = x0/chi_LS):
  cos(theta_eff) = cos(theta) + eps*( a.xhat - cos(theta) * n.xhat )
  Apparent node (cos=1/sqrt3) shifts by
  d_theta(phi) = eps*[ cos(alpha)*sin(theta*) - cos(theta*)*sin(alpha)*cos(phi) ]
  phi = azimuth around axis from the displacement's projection;
  alpha = angle(axis, displacement).
PASS pre-registration: offset SIGN at the Cold Spot azimuth matches for
one sign of x0, magnitude within factor ~3 of the measured 0.3 deg.
All input directions carry 10-25 deg literature uncertainties — this is
a consistency check, not precision astrometry. Substitute the paper's
exact cavity axis before quoting anything.
"""
import numpy as np

def unit(l_deg, b_deg):
    l, b = np.radians(l_deg), np.radians(b_deg)
    return np.array([np.cos(b)*np.cos(l), np.cos(b)*np.sin(l), np.sin(b)])

# --- literature inputs (galactic l, b) ---
AXES = {"Land-Magueijo (260,60)": unit(260, 60),
        "Copi quad+oct (250,60)": unit(250, 60)}
CS   = unit(209, -57)            # Cold Spot center
DIP  = unit(227, -15)            # Planck dipole-modulation direction (+-20deg)
A_mod   = 0.07                   # modulation amplitude
chi_LS  = 14160.0                # Mpc comoving
k0      = np.pi/4450.0           # Mpc^-1
x0      = A_mod/k0               # naive pricing -> ~99 Mpc  (O(1) model factor)
eps     = x0/chi_LS
ct, st  = 1/np.sqrt(3), np.sqrt(2/3)   # ideal node theta* = 54.7356 deg
th_star = np.degrees(np.arccos(ct))

print(f"x0 (naive A/k0 pricing) = {x0:.1f} Mpc -> eps = {eps:.5f} "
      f"({np.degrees(eps):.3f} deg scale)\n")

for name, ax in AXES.items():
    # Cold Spot polar angle from axis (use hemisphere closest to ring)
    cthCS = np.dot(ax, CS)
    thCS  = np.degrees(np.arccos(abs(cthCS)))   # fold: Y20 ring exists both sides
    offset_meas = thCS - th_star                # + = outside ring (larger theta)
    # azimuth of CS and dipole around the axis
    def azimuth(v, ax, ref):
        vp  = v - np.dot(v, ax)*ax; vp /= np.linalg.norm(vp)
        rp  = ref - np.dot(ref, ax)*ax; rp /= np.linalg.norm(rp)
        cosp = np.clip(np.dot(vp, rp), -1, 1)
        sgn  = np.sign(np.dot(ax, np.cross(rp, vp)))
        return np.degrees(np.arccos(cosp))*(1 if sgn >= 0 else -1)
    alpha = np.degrees(np.arccos(abs(np.dot(ax, DIP))))  # fold to <=90
    phiCS = azimuth(CS, ax, DIP)
    for sign, lab in [(+1, "+x0 toward dipole dir"), (-1, "-x0 (opposite)")]:
        a_dot_x = sign*abs(np.dot(ax, DIP))
        s_alpha = np.sin(np.radians(alpha))
        dth = eps*(a_dot_x*st - ct*sign*s_alpha*np.cos(np.radians(phiCS)))
        dth_deg = np.degrees(dth)
        match_sign = (np.sign(dth_deg) == np.sign(offset_meas))
        ratio = abs(dth_deg/offset_meas) if offset_meas != 0 else np.inf
        verdict = "SIGN MATCH" if match_sign else "sign mismatch"
        mag = "mag OK(<3x)" if 1/3 <= ratio <= 3 else f"mag ratio {ratio:.2f}"
        print(f"[{name}]  alpha(axis,dipole)={alpha:5.1f}  "
              f"thetaCS={thCS:6.2f} (offset {offset_meas:+.2f} deg)  "
              f"phiCS={phiCS:+7.1f}")
        print(f"   {lab:26s}: predicted ring shift at CS azimuth = "
              f"{dth_deg:+.3f} deg   -> {verdict}, {mag}")
    print()
print("NOTE: the measured offset sign (CS inside vs outside the ideal ring)")
print("must be taken from the paper's Y20 analysis — the fold above assumes")
print("|offset|; verify the sign against your published 0.3 deg figure.")
