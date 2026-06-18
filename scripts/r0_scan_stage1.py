#!/usr/bin/env python3
"""
r0_scan_stage1.py — R0 sensitivity of the CPM-derived (f, delta).

Pre-registered branches (decided BEFORE running):
  A) chi2(R0) minimum near 3.39  -> CMB selects the collapse radius;
     the half-period relation becomes a candidate selection principle
     for a data-selected number.
  B) chi2(R0) flat               -> f, delta insensitive to R0; their
     derived status becomes unconditional; contamination worry deleted.
  C) minimum elsewhere           -> production (f, delta) suboptimal;
     paper chi2 improves at true minimum; pi-relation demoted.
Every branch strengthens the paper.

Stage 1 (this script): diamond runs at each R0, derive the full chain
  f_spectral, eta_cross -> f ;  |A_p/A_r|, dphi -> delta
CONTROL GATE: the production-grid run at R0=3.39 must reproduce the
published values (f_spec=0.44, eta=0.39, ratio=1.63, dphi=-43.7deg,
f=0.27, delta=-0.56) to ~few %. If it does not, STOP — convention
drift — do not run stage 2.

Run from ~/qnm-boltzmann (needs cpm_diamond.py + greybody_calc.py).
"""
import json
import numpy as np
import cpm_diamond as cd
from greybody_calc import QNM_L2

R0_SCAN = [3.00, 3.20, 3.39, 3.394, 3.60, 4.00]
GRID_SCAN = dict(h=0.15, rstar_obs=20.0, u_range=(-10, 70), v_range=(0, 110))
GRID_PROD = dict(h=0.15, rstar_obs=20.0, u_range=(-10, 55), v_range=(0, 95))

wR0, wI0 = QNM_L2[0][0], QNM_L2[0][1]


def derive_chain(t_ret, Psi_obs, separation_factor=1.0):
    """f_spectral, eta_cross, complex prompt/ring ratio at omega_R0
    -> (f, delta). Mirrors analyze_waveform conventions exactly."""
    max_psi = np.max(np.abs(Psi_obs))
    if max_psi < 1e-10:
        return None
    thresh = 0.005 * max_psi
    idx_start = np.argmax(np.abs(Psi_obs) > thresh)
    idx_peak = np.argmax(np.abs(Psi_obs))
    t_sep = t_ret[idx_peak] + separation_factor / wI0

    m_full = t_ret >= t_ret[idx_start]
    m_prompt = (t_ret >= t_ret[idx_start]) & (t_ret < t_sep)
    m_ring = t_ret >= t_sep

    dt = t_ret[1] - t_ret[0]
    freqs = np.fft.rfftfreq(len(t_ret), d=dt) * 2 * np.pi
    F = lambda m: np.fft.rfft(np.where(m, Psi_obs, 0.0)) * dt
    fft_full, fft_prompt, fft_ring = F(m_full), F(m_prompt), F(m_ring)
    i0 = np.argmin(np.abs(freqs - wR0))

    Pt = np.abs(fft_full[i0])**2
    Pp = np.abs(fft_prompt[i0])**2
    Pr = np.abs(fft_ring[i0])**2
    f_spec = Pp / Pt
    eta_cross = (Pt - Pp - Pr) / Pt
    f_derived = f_spec * (1.0 - eta_cross)

    z = fft_prompt[i0] / fft_ring[i0]
    ratio = np.abs(z)
    dphi = np.angle(z)
    delta = -2.0 * np.cos(dphi) * ratio * (wI0 / wR0)

    return dict(f_spectral=float(f_spec), eta_cross=float(eta_cross),
                f=float(f_derived), ratio=float(ratio),
                dphi_deg=float(np.degrees(dphi)), delta=float(delta),
                t_peak=float(t_ret[idx_peak]),
                n_ring=int(m_ring.sum()))


def run(R0, grid):
    t_ret, Psi_obs, traj, meta = cd.diamond_integrate(R0=R0, **grid)
    out = derive_chain(t_ret, Psi_obs)
    out['R0'] = R0
    out['max_Psi'] = float(np.max(np.abs(Psi_obs)))
    return out


print("=" * 64)
print("CONTROL: production grid, R0 = 3.39")
print("=" * 64)
ctrl = run(3.39, GRID_PROD)
pub = dict(f_spectral=0.44, eta_cross=0.39, f=0.27, ratio=1.63,
           dphi_deg=-43.7, delta=-0.56)
print(f"\n{'quantity':>12s} {'control':>10s} {'published':>10s} {'dev%':>8s}")
ok = True
for k, v in pub.items():
    c = ctrl[k]
    dev = abs(c - v) / abs(v) * 100
    flag = "" if dev < 8 else "  <-- DRIFT"
    if dev >= 8:
        ok = False
    print(f"{k:>12s} {c:10.4f} {v:10.4f} {dev:8.2f}{flag}")
if not ok:
    print("\nCONTROL GATE FAILED — convention drift vs published values.")
    print("Do NOT proceed to stage 2. Investigate before trusting the scan.")
else:
    print("\nControl gate PASSED.")

print("\n" + "=" * 64)
print("R0 SCAN (uniform extended grid)")
print("=" * 64)
rows = []
for R0 in R0_SCAN:
    print(f"\n----- R0 = {R0} -----")
    r = run(R0, GRID_SCAN)
    rows.append(r)
    print(f"  f_spec={r['f_spectral']:.4f} eta={r['eta_cross']:.4f} "
          f"-> f={r['f']:.4f} | ratio={r['ratio']:.3f} "
          f"dphi={r['dphi_deg']:.1f}deg -> delta={r['delta']:.4f} "
          f"| peak t={r['t_peak']:.1f} n_ring={r['n_ring']}")

print("\n" + "=" * 64)
print(f"{'R0':>6s} {'f':>8s} {'delta':>8s} {'f_spec':>8s} "
      f"{'eta':>7s} {'ratio':>7s} {'dphi':>8s}")
for r in rows:
    print(f"{r['R0']:6.3f} {r['f']:8.4f} {r['delta']:8.4f} "
          f"{r['f_spectral']:8.4f} {r['eta_cross']:7.4f} "
          f"{r['ratio']:7.3f} {r['dphi_deg']:8.1f}")

json.dump(dict(control_production_grid=ctrl, control_gate_passed=ok,
               scan=rows, published=pub),
          open('r0_scan_results.json', 'w'), indent=2)
print("\nWrote r0_scan_results.json")
print("If the control gate passed, proceed: python3 r0_scan_stage2.py")
