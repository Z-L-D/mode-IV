#!/usr/bin/env python3
"""
r0_scan_stage2.py — chi2(R0) through the Boltzmann solver.

For each R0 in r0_scan_results.json: build an ini from the canonical
baseline with (qnm_f_grey, qnm_phase_delta) set to that R0's derived
(f, delta), run class, compute the amplitude-optimal projected
chi2/dof for TT, EE, TE (the few-% rescale regime where tonight's
projections matched converged runs to ~0.01). The candidate minimum
should then be converged properly (apply qnm_A, rerun) before any
number enters the paper.

Also runs the CONFIG DISAMBIGUATION pair to settle the Sec. II C
prose tension: (0.27,-0.56) vs (0.28,-0.60), both at the production
A=3.33e-8 / A_L=1.31.

Run from ~/qnm-boltzmann. Expects /tmp/freeze_test.ini (or
class_qnm/qnm_baseline_paper.ini) and class_qnm/class built with the
weight-mode patch (mode 0 default is fine here).
"""
import json, os, shutil, subprocess
import numpy as np
from scipy.interpolate import interp1d

T_CMB = 2.7255e6
BASE_INI_CANDIDATES = ['/tmp/freeze_test.ini',
                       'class_qnm/qnm_baseline_paper.ini']
CLASS_DIR = 'class_qnm'

base_ini = next((p for p in BASE_INI_CANDIDATES if os.path.exists(p)), None)
assert base_ini, "No baseline ini found — recreate freeze_test.ini first."
print(f"Baseline ini: {base_ini}")

scan = json.load(open('r0_scan_results.json'))
assert scan.get('control_gate_passed'), \
    "Stage 1 control gate did not pass — refusing to run stage 2."

planck = {}
for name, fn in [("TT", "planck_tt_binned.txt"),
                 ("EE", "planck_ee_binned.txt"),
                 ("TE", "planck_te_binned.txt")]:
    d = np.loadtxt(fn)
    planck[name] = d


def projected_chi2(cl_file):
    lensed = np.loadtxt(cl_file)
    ell_q = lensed[:, 0]
    cols = {"TT": lensed[:, 1]*T_CMB**2, "EE": lensed[:, 2]*T_CMB**2,
            "TE": lensed[:, 3]*T_CMB**2}
    # TT-anchored optimal rescale
    def pieces(name):
        p = planck[name]
        m = (p[:, 0] >= 30) & (p[:, 0] >= ell_q.min()) & (p[:, 0] <= ell_q.max())
        ell, Dl, sig = p[m, 0], p[m, 1], (p[m, 2]+p[m, 3])/2
        q = interp1d(ell_q, cols[name], kind='cubic')(ell)
        return Dl, sig, q
    Dl, sig, q = pieces("TT")
    s = np.sum(Dl*q/sig**2)/np.sum(q*q/sig**2)
    out = {"s": s}
    for name in ("TT", "EE", "TE"):
        Dl, sig, q = pieces(name)
        out[name] = float(np.sum(((Dl - s*q)/sig)**2)/(len(Dl)-1))
    return out


def run_config(tag, f_val, delta_val, extra=None):
    ini = os.path.join(CLASS_DIR, f'r0scan_{tag}.ini')
    lines = [l for l in open(base_ini)
             if not l.startswith(('root', 'qnm_f_grey', 'qnm_phase_delta'))]
    lines += [f"root = output/r0scan_{tag}_\n",
              f"qnm_f_grey = {f_val}\n",
              f"qnm_phase_delta = {delta_val}\n"]
    if extra:
        lines += [e + "\n" for e in extra]
    open(ini, 'w').writelines(lines)

    r = subprocess.run(['./class', f'r0scan_{tag}.ini'],
                       cwd=CLASS_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  CLASS FAILED for {tag}:\n{r.stderr[-500:]}")
        return None
    outs = sorted([os.path.join(CLASS_DIR, 'output', x)
                   for x in os.listdir(os.path.join(CLASS_DIR, 'output'))
                   if x.startswith(f'r0scan_{tag}_') and
                   x.endswith('_cl_lensed.dat')],
                  key=os.path.getmtime)
    if not outs:
        print(f"  No lensed output for {tag}")
        return None
    return projected_chi2(outs[-1])


print("\n================ chi2(R0) scan ================")
print(f"{'R0':>6s} {'f':>8s} {'delta':>8s} {'s_opt':>8s} "
      f"{'TT':>7s} {'EE':>7s} {'TE':>7s}")
results = []
for row in scan['scan']:
    R0, f_v, d_v = row['R0'], row['f'], row['delta']
    res = run_config(f"R{str(R0).replace('.','p')}", f_v, d_v)
    if res:
        results.append((R0, f_v, d_v, res))
        print(f"{R0:6.3f} {f_v:8.4f} {d_v:8.4f} {res['s']:8.4f} "
              f"{res['TT']:7.3f} {res['EE']:7.3f} {res['TE']:7.3f}")

print("\n================ config disambiguation ================")
print("(settles the Sec. II C prose tension)")
for tag, fv, dv in [("derivedcfg", 0.27, -0.56), ("tablecfg", 0.28, -0.60)]:
    res = run_config(tag, fv, dv)
    if res:
        print(f"  (f={fv}, delta={dv}) at production A,A_L: "
              f"TT={res['TT']:.3f} EE={res['EE']:.3f} TE={res['TE']:.3f} "
              f"(s={res['s']:.4f})")

if results:
    best = min(results, key=lambda r: r[3]['TT'])
    print(f"\nTT minimum on scan grid: R0 = {best[0]} "
          f"(TT = {best[3]['TT']:.3f})")
    print("Converge the minimum candidate (apply printed qnm_A = "
          "3.33e-8 * s_opt, rerun) before quoting numbers.")
    json.dump([{'R0': r[0], 'f': r[1], 'delta': r[2], **r[3]}
               for r in results],
              open('r0_scan_chi2.json', 'w'), indent=2)
    print("Wrote r0_scan_chi2.json")
