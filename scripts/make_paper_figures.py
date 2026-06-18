#!/usr/bin/env python3
"""
make_paper_figures.py
Regenerates the four figures referenced in paper4 (.tex placeholders):

  Figure 1: fig_cmb_comparison.pdf  - TT/EE/TE: QNM vs LCDM vs Planck
  Figure 2: fig_pk_comparison.pdf   - P(k) mode sum vs power law + ratio
  Figure 3: fig_chi2_D.pdf          - chi2(D) selectivity curve
  Figure 4: fig_cmb_anomaly.pdf     - Y20 anomaly geometry

Figures 2 and 4 are fully self-contained (computed from model formulas).
Figures 1 and 3 require data files - paths are auto-discovered where
possible, otherwise edit the CONFIG block below.

Run from the repo root:  python3 scripts/make_paper_figures.py
Outputs land in figures/ as both .pdf (paper) and .png (preview).
"""

import os, sys, glob, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =================================================================
# CONFIG - edit these paths if auto-discovery fails
# =================================================================
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isdir(os.path.join(REPO, 'figures')):
    REPO = os.getcwd()   # fallback: run from anywhere, write to ./figures
FIGDIR = os.path.join(REPO, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# Figure 1 inputs
QNM_CL_FILE   = None   # CLASS lensed output (canonical configuration)
PLANCK_TT     = os.path.join(REPO, 'data/planck/planck_tt_binned.txt')
PLANCK_EE     = os.path.join(REPO, 'data/planck/planck_ee_binned.txt')
PLANCK_TE     = os.path.join(REPO, 'data/planck/planck_te_binned.txt')

# Figure 3 input
DSCAN_FILE    = None   # two columns: D, chi2_dof (auto-discovers *D_scan*/*chi2_D*)

T_CMB_uK = 2.7255e6    # CLASS dimensionless -> muK^2

# Production QNM parameters (qnm_baseline_paper.ini)
A_QNM   = 3.33e-8
GAMMA   = 1.04
F_GREY  = 0.27
DELTA   = -0.56
RS_MPC  = 4450.6
K0      = np.pi / RS_MPC          # 7.06e-4 Mpc^-1
N_MODES = 300
# LCDM reference primordial
A_S     = 2.100e-9
N_S     = 0.9649
K_PIVOT = 0.05

OMEGA_I_LEAVER = [0.17792, 0.54783, 0.95654, 1.41030,
                  1.89369, 2.39134, 2.89591, 3.40400]

def style():
    plt.rcParams.update({
        'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
        'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'figure.dpi': 150, 'savefig.bbox': 'tight',
        'axes.linewidth': 0.8, 'lines.linewidth': 1.2,
    })

def save(fig, name):
    for ext in ('pdf', 'png'):
        path = os.path.join(FIGDIR, f'{name}.{ext}')
        fig.savefig(path)
    print(f'  wrote figures/{name}.pdf (+.png)')
    plt.close(fig)

# =================================================================
# Auto-discovery helpers
# =================================================================
def find_first(patterns, where):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(where, pat)))
        if hits:
            return hits[0]
    return None

def discover_from_compare_script():
    """Pull quoted .dat/.txt filenames out of compare_all_lensed.py."""
    cand = find_first(['compare_all_lensed.py'], os.path.join(REPO, 'scripts')) \
        or find_first(['compare_all_lensed.py'], REPO) \
        or find_first(['compare_all_lensed.py'], os.path.expanduser('~/qnm-boltzmann'))
    if not cand:
        return []
    txt = open(cand).read()
    return re.findall(r"['\"]([^'\"]+\.(?:dat|txt|csv))['\"]", txt)

# =================================================================
# FIGURE 2: P(k) comparison  (self-contained)
# =================================================================
def qnm_pk(k):
    """QNM mode sum, mirroring class_qnm primordial.c (weight_mode 0)."""
    w = np.array([1.0/(n+1)**GAMMA for n in range(N_MODES)])
    w /= w.sum()
    pk = np.zeros_like(k)
    for n in range(N_MODES):
        kn = K0 * (n + 1)
        oI = OMEGA_I_LEAVER[n] if n < 8 else (2.0*n + 1.0)/8.0
        sigma = oI * K0 / (2.0*np.pi)
        eps = (k - kn) / sigma
        L1 = 1.0/(1.0 + eps**2)
        L2 = L1*L1
        Lb = (1.0 - F_GREY)*L1 + F_GREY*L2
        pk += w[n] * Lb * (1.0 - DELTA*eps*L1)
    return A_QNM * pk

def lcdm_pk(k):
    return A_S * (k/K_PIVOT)**(N_S - 1.0)

def fig2():
    print('Figure 2: P(k) comparison')
    k = np.logspace(np.log10(2e-4), np.log10(0.30), 6000)
    pq, pl = qnm_pk(k), lcdm_pk(k)
    # Display normalization: match LCDM at the pivot scale.
    # (Absolute normalizations differ by the Lorentzian sum integral;
    #  cross-check the ratio panel against pk_diagnostic.py values.)
    i_piv = np.argmin(np.abs(k - K_PIVOT))
    # average over a band around pivot to avoid sitting on a mode peak
    band = (k > 0.8*K_PIVOT) & (k < 1.25*K_PIVOT)
    norm = np.mean(pl[band]) / np.mean(pq[band])
    pq_n = pq * norm

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.4), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1],
                                                'hspace': 0.08})
    ax1.loglog(k, pl, 'r--', label=r'$\Lambda$CDM power law')
    ax1.loglog(k, pq_n, 'b-', lw=0.8, label='QNM mode sum')
    ax1.set_ylabel(r'$P_{\cal R}(k)$')
    ax1.legend(loc='lower left', frameon=False)
    ax1.set_ylim(3e-10, 8e-9)

    ratio = pq_n / pl
    ax2.semilogx(k, ratio, 'b-', lw=0.8)
    ax2.axhline(1.0, color='k', lw=0.5, alpha=0.6)
    ax2.axvspan(0.006, 0.20, color='green', alpha=0.15,
                label=r'CMB core $k=0.006$-$0.20\,{\rm Mpc}^{-1}$')
    ax2.set_xlabel(r'$k\ [{\rm Mpc}^{-1}]$')
    ax2.set_ylabel(r'$P_{\rm QNM}/P_{\Lambda{\rm CDM}}$')
    ax2.set_ylim(0.0, 2.0)
    ax2.legend(loc='upper right', frameon=False)

    core = (k > 0.006) & (k < 0.20)
    print(f'  CMB-core ratio range: {ratio[core].min():.3f} - '
          f'{ratio[core].max():.3f}  (paper quotes 0.89-1.01)')
    save(fig, 'fig_pk_comparison')

# =================================================================
# FIGURE 4: Y20 anomaly geometry  (self-contained)
# =================================================================
def fig4():
    print('Figure 4: CMB anomaly Y20 geometry')
    theta = np.linspace(0, 180, 1000)
    th = np.deg2rad(theta)
    Y20 = 0.25*np.sqrt(5/np.pi)*(3*np.cos(th)**2 - 1)
    node = np.rad2deg(np.arccos(1/np.sqrt(3)))   # 54.74 deg

    anomalies = [
        # (name, predicted theta, measured deviation deg, marker, color)
        ('Quadrupole/octupole\n(axis-defining)', 0.0,   0.0, 'o', 'tab:blue'),
        ('Cold Spot',                       node,  0.3, 's', 'tab:red'),
        ('Hemispherical\nasymmetry',        90.0,  2.7, '^', 'tab:green'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8),
                                   gridspec_kw={'width_ratios': [1.5, 1]})
    ax1.plot(theta, Y20, 'k-', lw=1.4)
    ax1.axhline(0, color='gray', lw=0.5, alpha=0.6)
    ax1.axvline(node, color='gray', ls=':', lw=0.8)
    ax1.text(node+1.5, ax1.get_ylim()[0]*0+0.42,
             f'node\n{node:.1f}°', fontsize=7, color='gray')
    for name, th0, dev, mk, c in anomalies:
        y0 = 0.25*np.sqrt(5/np.pi)*(3*np.cos(np.deg2rad(th0))**2 - 1)
        ax1.plot(th0, y0, mk, color=c, ms=8, zorder=5)
        # mirror at 180-theta for the southern intersections
    ax1.set_xlabel(r'$\theta$ from cavity axis $(232.3^\circ,\ +66.9^\circ)$ [deg]')
    ax1.set_ylabel(r'$Y_2^0(\theta)$')
    ax1.set_xlim(0, 180)

    names = [a[0] for a in anomalies]
    devs  = [a[1+1] for a in anomalies]
    colors= [a[4] for a in anomalies]
    ypos = np.arange(len(names))
    ax2.barh(ypos, devs, color=colors, alpha=0.75, height=0.55)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(names, fontsize=7)
    ax2.set_xlabel('angular deviation [deg]')
    ax2.invert_yaxis()
    for y, d in zip(ypos, devs):
        ax2.text(d + 0.07, y, f'{d:.1f}°', va='center', fontsize=7)
    ax2.set_xlim(0, 3.6)
    ax2.text(0.97, 0.97,
             'Cold Spot: $p=0.007$ ($2.4\\sigma$;\n$2.1\\sigma$ after trials)',
             transform=ax2.transAxes, ha='right', va='top', fontsize=6.5)

    save(fig, 'fig_cmb_anomaly')

# =================================================================
# FIGURE 1: TT/EE/TE comparison  (needs data files)
# =================================================================
def load_class_cl(path):
    """CLASS *_cl_lensed.dat: l, TT, EE, TE, ... (dimensionless D_l)."""
    d = np.loadtxt(path)
    ell = d[:, 0]
    return ell, d[:, 1]*T_CMB_uK**2, d[:, 2]*T_CMB_uK**2, d[:, 3]*T_CMB_uK**2

def fig1():
    print('Figure 1: CMB TT/EE/TE comparison')
    global QNM_CL_FILE

    if QNM_CL_FILE is None:
        QNM_CL_FILE = find_first(
            ['qnm_test_100_cl_lensed.dat', '*.bak_production'],
            os.path.join(REPO, 'results')) \
            or find_first(['qnm_test_100_cl_lensed.dat'],
                os.path.expanduser('~/qnm-boltzmann/class_qnm/output'))

    for name, path in [('QNM C_l', QNM_CL_FILE), ('Planck TT', PLANCK_TT),
                        ('Planck EE', PLANCK_EE), ('Planck TE', PLANCK_TE)]:
        if path is None or not os.path.exists(path):
            print(f'  SKIPPED — missing: {name} ({path})')
            return

    # Load QNM C_ell from CLASS output (dimensionless D_ell)
    d = np.loadtxt(QNM_CL_FILE)
    ell_q = d[:, 0]
    qTT, qEE, qTE = d[:, 1]*T_CMB_uK**2, d[:, 2]*T_CMB_uK**2, d[:, 3]*T_CMB_uK**2

    # Planck binned: 5 columns (ell, D_ell, -sigma, +sigma, BestFit)
    # All already in muK^2
    pTT = np.loadtxt(PLANCK_TT)
    pEE = np.loadtxt(PLANCK_EE)
    pTE = np.loadtxt(PLANCK_TE)

    panels = [
        ('TT', qTT, pTT, r'$\mathcal{D}_\ell^{TT}\ [\mu\mathrm{K}^2]$'),
        ('EE', qEE, pEE, r'$\mathcal{D}_\ell^{EE}\ [\mu\mathrm{K}^2]$'),
        ('TE', qTE, pTE, r'$\mathcal{D}_\ell^{TE}\ [\mu\mathrm{K}^2]$'),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.5), sharex=True,
                             gridspec_kw={'hspace': 0.08})
    for ax, (name, qcl, planck, ylabel) in zip(axes, panels):
        ell_p = planck[:, 0]
        dl_p  = planck[:, 1]
        sig_p = 0.5*(planck[:, 2] + planck[:, 3])  # average of -/+ sigma
        bf_p  = planck[:, 4]  # LCDM best-fit

        # QNM theory curve
        ax.plot(ell_q, qcl, 'b-', lw=0.9, label='QNM', zorder=3)
        # LCDM best-fit (from Planck file)
        ax.plot(ell_p, bf_p, 'r--', lw=0.9, label=r'$\Lambda$CDM', zorder=2)
        # Planck data
        ax.errorbar(ell_p, dl_p, yerr=sig_p, fmt='k.', ms=2.5, lw=0.5,
                    capsize=0, label='Planck 2018', zorder=1)
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right', frameon=False, fontsize=7)

        if name == 'TT':
            ax.set_ylim(0, 6500)
        elif name == 'EE':
            ax.set_ylim(-5, 50)
            ax.set_xlim(2, 2500)

    axes[-1].set_xlabel(r'$\ell$')
    axes[-1].set_xlim(2, 2500)
    save(fig, 'fig_cmb_comparison')

# =================================================================
# FIGURE 3: chi2(D) selectivity  (needs scan data)
# =================================================================
def fig3():
    print('Figure 3: chi2(D) selectivity')
    global DSCAN_FILE
    if DSCAN_FILE is None:
        for pd in [os.path.join(REPO, 'results'), REPO,
                   os.path.expanduser('~/qnm-boltzmann')]:
            DSCAN_FILE = DSCAN_FILE or find_first(
                ['*D_scan*', '*dscan*', '*chi2_D*', '*d_scan*'], pd)
    if DSCAN_FILE is None:
        print('  SKIPPED - no D-scan data found (*D_scan*, *chi2_D*).')
        print('  Point DSCAN_FILE at the scan results (columns: D, chi2_dof)')
        print('  or rerun the scan; the paper quotes min at D=8.0,')
        print('  rising to chi2/dof ~ 50 at D = 7.0 and 9.0.')
        return
    d = np.loadtxt(DSCAN_FILE, comments='#', skiprows=1)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(d[:, 0], d[:, 1], 'bo-', ms=4, lw=1.0)
    ax.axvline(8.0, color='k', ls='--', lw=0.8)
    ax.text(8.05, 0.92, 'Motl-Neitzke\n$D=8$', transform=ax.get_xaxis_transform(),
            fontsize=7, va='top')
    ax.set_xlabel(r'$D$ in $\omega_I = (2n+1)/D$  ($M=1$)')
    ax.set_ylabel(r'TT $\chi^2/{\rm dof}$')
    save(fig, 'fig_chi2_D')

# =================================================================
if __name__ == '__main__':
    style()
    print(f'Repo root: {REPO}')
    print(f'Figures  : {FIGDIR}\n')
    fig2()
    fig4()
    fig1()
    fig3()
    print('\nDone. Check figures/ and uncomment the \\includegraphics lines '
          'in the .tex.')
