#!/usr/bin/env python3
"""
extract_overtones.py — extract QNM excitation amplitudes |C_n|^2 from the
CPM diamond waveform by linear least-squares ringdown fitting with the
frequencies FIXED at the exact Leaver values (only amplitudes free).

Model on t >= t0:
    Psi(t) = sum_n e^{-wI_n (t-t0)} [ a_n cos(wR_n (t-t0)) + b_n sin(...) ]
Linear in (a_n, b_n)  ->  lstsq, no nonlinear fitting, no frequency drift.

Honesty machinery:
  * t0 is scanned from the waveform peak forward; a real amplitude is one
    that PLATEAUS across a t0 window. Drifting amplitude = contaminated.
  * Design-matrix condition number reported per (N_modes, t0).
  * Fit residual RMS reported; adding a mode must reduce it materially
    or the mode is not detected.

Output:
  * stability table per overtone
  * CLASS heights H_n = |C_n|^2 / omega_I_n (shape only; A_QNM absorbs scale)
  * comparison of H ratios against (n+1)^-1.04 and lifetime 1/omega_I
  * cpm_weights_lown.dat (n, H_n normalized) for qnm_weight_mode = 2 hybrids

Usage: python3 extract_overtones.py [cpm_diamond_results.json]
"""
import sys, json
import numpy as np

# Exact Leaver l=2 frequencies, 2M=1 units (matches repo convention)
LEAVER = [(0.74734, 0.17792), (0.69342, 0.54783), (0.60211, 0.95654),
          (0.50302, 1.41030), (0.41583, 1.89369), (0.34615, 2.39134)]

path = sys.argv[1] if len(sys.argv) > 1 else 'cpm_diamond_results.json'
d = json.load(open(path))
t = np.array(d['t_ret'], dtype=float)
psi = np.array(d['Psi_obs'], dtype=float)

dt = np.median(np.diff(t))
i_pk = int(np.argmax(np.abs(psi)))
t_pk = t[i_pk]
T_end = t[-1]
efolds_n0 = (T_end - t_pk) * LEAVER[0][1]
print(f"Waveform: {len(t)} pts, dt={dt:.4f}, span [{t[0]:.2f}, {T_end:.2f}]")
print(f"Peak |Psi|={np.max(np.abs(psi)):.4f} at t={t_pk:.2f}; "
      f"post-peak = {T_end-t_pk:.1f} = {efolds_n0:.1f} fundamental e-folds")

def design(tt, N):
    cols = []
    for n in range(N):
        wr, wi = LEAVER[n]
        e = np.exp(-wi * tt)
        cols += [e * np.cos(wr * tt), e * np.sin(wr * tt)]
    return np.vstack(cols).T

def fit(t0, N):
    m = t >= t0
    if m.sum() < 4 * N:
        return None
    tt = t[m] - t0
    A = design(tt, N)
    coef, res, rank, sv = np.linalg.lstsq(A, psi[m], rcond=None)
    model = A @ coef
    rms = np.sqrt(np.mean((psi[m] - model) ** 2))
    cond = sv[0] / sv[-1]
    # |C_n|^2 referenced to common epoch t0: amplitude at t0 of each mode
    C2 = np.array([coef[2*n]**2 + coef[2*n+1]**2 for n in range(N)])
    return C2, rms, cond

# ---- t0 scan for each model size ----------------------------------
t0_grid = np.arange(t_pk + 0.5, t_pk + 12.0 + 1e-9, 0.5)
print("\n================ t0 stability scan ================")
results = {}
for N in (1, 2, 3, 4):
    print(f"\n--- N_modes = {N} ---")
    print(f"{'t0':>6s} {'rms':>10s} {'cond':>10s} " +
          " ".join(f"{'|C%d|^2'%n:>11s}" for n in range(N)))
    rows = []
    for t0 in t0_grid:
        r = fit(t0, N)
        if r is None:
            continue
        C2, rms, cond = r
        rows.append((t0, rms, cond, C2))
        print(f"{t0:6.1f} {rms:10.3e} {cond:10.2e} " +
              " ".join(f"{c:11.4e}" for c in C2))
    results[N] = rows

# ---- amplitude referencing ----------------------------------------
# C_n extracted at varying t0 must be propagated to a COMMON epoch to be
# comparable: C_n(t_ref) = C_n(t0) * exp(+wI_n (t0 - t_ref))  (amplitudes),
# i.e. |C|^2 factor exp(2 wI (t0 - t_ref)). A mode is genuinely detected
# when the t_ref-referenced |C_n|^2 plateaus over a t0 window.
t_ref = t_pk + 2.0
print(f"\n====== |C_n|^2 referenced to common epoch t_ref = {t_ref:.2f} ======")
print("(plateau across t0 = detection; monotonic drift = contamination)")
for N in (2, 3, 4):
    print(f"\n--- N_modes = {N} ---")
    print(f"{'t0':>6s} " + " ".join(f"{'C%d^2@ref'%n:>12s}" for n in range(N)))
    for (t0, rms, cond, C2) in results[N]:
        ref = [C2[n] * np.exp(2*LEAVER[n][1]*(t0 - t_ref)) for n in range(N)]
        print(f"{t0:6.1f} " + " ".join(f"{c:12.4e}" for c in ref))

# ---- pick reported values from the plateau (median over early window) --
print("\n====== Extracted weights (median over t0 in [pk+1, pk+4]) ======")
N_report = 3
window = [(r if (t_pk+1.0) <= r[0] <= (t_pk+4.0) else None)
          for r in results[N_report]]
window = [r for r in window if r]
C2ref = np.array([[r[3][n]*np.exp(2*LEAVER[n][1]*(r[0]-t_ref))
                   for n in range(N_report)] for r in window])
C2med = np.median(C2ref, axis=0)
C2spr = (np.percentile(C2ref, 84, axis=0) -
         np.percentile(C2ref, 16, axis=0)) / 2

wI = np.array([LEAVER[n][1] for n in range(N_report)])
H = C2med / wI                      # CLASS heights: energy / linewidth
Hn = H / H[0]
He = (C2spr / wI) / H[0]

pow104 = (np.arange(N_report)+1.0)**-1.04
life = (1.0/wI) / (1.0/wI[0])
print(f"\n{'n':>2s} {'|C|^2':>11s} {'+-':>10s} {'H_n=C2/wI':>11s} "
      f"{'H/H0':>8s} {'+-':>7s} {'(n+1)^-1.04':>12s} {'lifetime':>9s}")
for n in range(N_report):
    print(f"{n:2d} {C2med[n]:11.4e} {C2spr[n]:10.2e} {H[n]:11.4e} "
          f"{Hn[n]:8.4f} {He[n]:7.4f} {pow104[n]:12.4f} {life[n]:9.4f}")

np.savetxt('cpm_weights_lown.dat',
           np.column_stack([np.arange(N_report), Hn, He]),
           fmt=['%d', '%.6e', '%.6e'],
           header='n  H_n/H_0  err   (CPM-extracted, epoch t_ref=peak+2)')
print("\nWrote cpm_weights_lown.dat")
print("\nREAD THE STABILITY TABLES BEFORE BELIEVING THE SUMMARY:")
print(" - C0 should plateau everywhere (it will).")
print(" - C1 plateau over several t0 steps = real first-overtone measurement.")
print(" - C2: expect drift/blowup; treat as upper bound unless it plateaus.")
print(" - cond > ~1e8 at small t0 means the early window is degenerate;")
print("   trust only t0 regions with cond below that.")
