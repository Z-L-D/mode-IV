#!/usr/bin/env python3
"""
C(theta) Angular Correlation from CLASS output
Usage: python3 c_theta_from_class_v2.py <qnm_cl_lensed.dat> <lcdm_cl_lensed.dat>
"""
import sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

def legendre_val(l, x):
    if l == 0: return 1.0
    if l == 1: return x
    p0, p1 = 1.0, x
    for n in range(2, l+1):
        p2 = ((2*n-1)*x*p1 - (n-1)*p0) / n
        p0, p1 = p1, p2
    return p1

def load_class_cl(filepath):
    T_CMB_sq = (2.7255e6)**2
    ells, Dl_raw = [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line: continue
            parts = line.split()
            try:
                ells.append(int(parts[0]))
                Dl_raw.append(float(parts[1]))
            except: continue
    ells, Dl_raw = np.array(ells), np.array(Dl_raw)
    Dl_uK2 = Dl_raw * T_CMB_sq
    Cl_uK2 = Dl_uK2 * 2*np.pi / (ells*(ells+1))
    return ells, Cl_uK2, Dl_uK2

def compute_c_theta(ells, Cl, theta_arr, ell_max=100):
    ct = np.zeros(len(theta_arr))
    mask = ells <= ell_max
    el, cl = ells[mask], Cl[mask]
    for i, t in enumerate(theta_arr):
        cos_t = np.cos(np.radians(t))
        for j, l in enumerate(el):
            ct[i] += (2*l+1)/(4*np.pi) * cl[j] * legendre_val(int(l), cos_t)
    return ct

def s_half(ct, theta):
    mask = theta >= 60
    return -np.trapezoid(ct[mask]**2, np.cos(np.radians(theta[mask])))

if len(sys.argv) < 3:
    print("Usage: python3 c_theta_from_class_v2.py <qnm_cl_lensed.dat> <lcdm_cl_lensed.dat>")
    sys.exit(1)

print("="*60)
print("C(theta) from CLASS — QNM vs LCDM")
print("="*60)

print(f"\nQNM:  {sys.argv[1]}")
ell_q, Cl_q, Dl_q = load_class_cl(sys.argv[1])
print(f"  {len(ell_q)} multipoles, l={ell_q[0]} to {ell_q[-1]}")

print(f"LCDM: {sys.argv[2]}")
ell_l, Cl_l, Dl_l = load_class_cl(sys.argv[2])
print(f"  {len(ell_l)} multipoles, l={ell_l[0]} to {ell_l[-1]}")

print(f"\nD_l ratio (QNM/LCDM):")
for l in [2,3,5,10,20,30,50]:
    if l in ell_q and l in ell_l:
        dq, dl = Dl_q[ell_q==l][0], Dl_l[ell_l==l][0]
        print(f"  l={l:3d}: {dq:10.1f} / {dl:10.1f} = {dq/dl:.4f}")

print(f"\nComputing C(theta)...")
theta = np.linspace(1, 179, 300)
ct_q = compute_c_theta(ell_q, Cl_q, theta)
ct_l = compute_c_theta(ell_l, Cl_l, theta)

sq, sl = s_half(ct_q, theta), s_half(ct_l, theta)

print(f"\nS_1/2 statistic:")
print(f"  LCDM:   {sl:.1f} uK^4")
print(f"  QNM:    {sq:.1f} uK^4  ({sq/sl*100:.1f}% of LCDM)")
print(f"  Planck: ~1065 uK^4     (~2.5% of LCDM)")
print(f"  Ratio QNM/LCDM: {sq/sl:.4f}")

print(f"\nC(theta) at key angles:")
for t in [10,30,60,90,120,150,170]:
    idx = np.argmin(np.abs(theta-t))
    r = ct_q[idx]/ct_l[idx] if abs(ct_l[idx])>1e-10 else 0
    print(f"  {t:>5}deg: LCDM={ct_l[idx]:10.1f}  QNM={ct_q[idx]:10.1f}  ratio={r:.3f}")

rms_l = np.sqrt(np.mean(ct_l[theta>=60]**2))
rms_q = np.sqrt(np.mean(ct_q[theta>=60]**2))
print(f"\nRMS C(theta>60): LCDM={rms_l:.1f}  QNM={rms_q:.1f}  ({rms_q/rms_l*100:.1f}%)")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
ml, mq = ell_l<=100, ell_q<=100
axes[0,0].plot(ell_l[ml], Dl_l[ml], 'r-', lw=2, label='LCDM')
axes[0,0].plot(ell_q[mq], Dl_q[mq], 'b-', lw=2, label='QNM')
axes[0,0].set_xlabel('l'); axes[0,0].set_ylabel('D_l (uK^2)')
axes[0,0].set_title('Low-l Power (CLASS)'); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

ce = np.intersect1d(ell_q[mq], ell_l[ml])
rat = np.array([Dl_q[ell_q==l][0]/Dl_l[ell_l==l][0] for l in ce])
axes[0,1].plot(ce, rat, 'b-', lw=2)
axes[0,1].axhline(1, color='r', ls='--', alpha=0.3)
axes[0,1].set_xlabel('l'); axes[0,1].set_ylabel('QNM/LCDM')
axes[0,1].set_title('Power Ratio'); axes[0,1].set_ylim(0,1.5); axes[0,1].grid(alpha=0.3)

axes[1,0].plot(theta, ct_l, 'r-', lw=2.5, label='LCDM')
axes[1,0].plot(theta, ct_q, 'b-', lw=2.5, label='QNM')
axes[1,0].axhline(0, color='gray', alpha=0.5)
axes[1,0].axvline(60, color='green', ls='--', lw=2, alpha=0.5, label='60 deg')
axes[1,0].set_xlabel('theta (deg)'); axes[1,0].set_ylabel('C(theta) uK^2')
axes[1,0].set_title(f'Angular Correlation\nS_1/2(QNM) = {sq/sl*100:.1f}% of LCDM')
axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

bars = axes[1,1].bar([0,1,2], [sl,sq,1065], color=['r','b','k'], alpha=0.7, width=0.5)
for b,v in zip(bars,[sl,sq,1065]):
    axes[1,1].text(b.get_x()+b.get_width()/2, v+sl*0.02, f'{v/sl*100:.1f}%', ha='center', fontsize=12, fontweight='bold')
axes[1,1].set_xticks([0,1,2]); axes[1,1].set_xticklabels(['LCDM','QNM','Planck obs'])
axes[1,1].set_ylabel('S_1/2 (uK^4)'); axes[1,1].set_title('S_1/2 Statistic')
axes[1,1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('c_theta_class.png', dpi=150)
print(f"\nSaved: c_theta_class.png")
