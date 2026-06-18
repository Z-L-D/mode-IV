#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

cl_qnm = np.loadtxt('class_qnm/output/qnm_test_12_cl.dat')
cl_lcdm = np.loadtxt('class_vanilla/output/explanatory01_cl.dat')

print(f"QNM shape: {cl_qnm.shape}, ell: {cl_qnm[0,0]:.0f}-{cl_qnm[-1,0]:.0f}")
print(f"LCDM shape: {cl_lcdm.shape}, ell: {cl_lcdm[0,0]:.0f}-{cl_lcdm[-1,0]:.0f}")

ell_q, tt_q = cl_qnm[:,0], cl_qnm[:,1]
ell_l, tt_l = cl_lcdm[:,0], cl_lcdm[:,1]

fig, axes = plt.subplots(3, 1, figsize=(14,14), gridspec_kw={'height_ratios':[3,2,2]})

ax1 = axes[0]
ax1.plot(ell_l, tt_l, 'r-', lw=2, label='LCDM', alpha=0.8)
ax1.plot(ell_q, tt_q, 'b-', lw=2, label='QNM', alpha=0.8)
ax1.set_xlabel('l'); ax1.set_ylabel('l(l+1)Cl/2pi [uK^2]')
ax1.set_title('CMB TT Power Spectrum: QNM vs LCDM')
ax1.legend(fontsize=12); ax1.set_xlim(2,2500); ax1.grid(True, alpha=0.3)

ell_min = int(max(ell_q[0], ell_l[0]))
ell_max = int(min(ell_q[-1], ell_l[-1]))
ell_c = np.arange(ell_min, ell_max+1)
fq = interp1d(ell_q, tt_q, kind='linear', bounds_error=False)
fl = interp1d(ell_l, tt_l, kind='linear', bounds_error=False)
tq = fq(ell_c); tl = fl(ell_c)

ratio = np.where(tl>0, tq/tl, np.nan)
ax2 = axes[1]
ax2.plot(ell_c, ratio, 'b-', lw=1, alpha=0.8)
ax2.axhline(1.0, color='r', ls='--'); ax2.set_xlim(2,2500); ax2.set_ylim(0,3)
ax2.set_xlabel('l'); ax2.set_ylabel('QNM/LCDM'); ax2.set_title('Ratio')
ax2.grid(True, alpha=0.3)

resid = np.where(tl>0, (tq-tl)/tl*100, np.nan)
ax3 = axes[2]
ax3.plot(ell_c, resid, 'b-', lw=1, alpha=0.8)
ax3.axhline(0, color='r', ls='--'); ax3.set_xlim(2,2500); ax3.set_ylim(-100,100)
ax3.set_xlabel('l'); ax3.set_ylabel('% diff'); ax3.set_title('(QNM-LCDM)/LCDM %')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cl_qnm_vs_lcdm.png', dpi=150, bbox_inches='tight')
print("Saved: cl_qnm_vs_lcdm.png")

v = np.isfinite(ratio)
m = (ell_c>=200)&(ell_c<=2500)&v
print(f"\nMean ratio l=200-2500: {np.mean(ratio[m]):.3f}")
for peak, ell in [('1',220),('2',540),('3',810)]:
    print(f"Peak {peak} (l~{ell}): QNM={fq(ell):.1f}, LCDM={fl(ell):.1f}, ratio={fq(ell)/fl(ell):.3f}")
