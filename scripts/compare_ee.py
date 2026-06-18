import numpy as np
import matplotlib.pyplot as plt

# Planck EE binned
ee_p = np.loadtxt('planck_ee_binned.txt')
p_ell, p_Dl, p_errm, p_errp, p_best = ee_p[:,0], ee_p[:,1], ee_p[:,2], ee_p[:,3], ee_p[:,4]
p_err = 0.5*(p_errm + p_errp)

# CLASS QNM and LCDM
qnm = np.loadtxt('class_qnm/output/qnm_test_12_cl.dat')
lcdm = np.loadtxt('class_vanilla/output/explanatory01_cl.dat')

T2 = (2.7255e6)**2  # uK^2

q_ell, q_ee = qnm[:,0], qnm[:,2] * T2
l_ell, l_ee = lcdm[:,0], lcdm[:,2] * T2

# Interpolate to Planck ells
q_at_p = np.interp(p_ell, q_ell, q_ee)
l_at_p = np.interp(p_ell, l_ell, l_ee)

# Chi2 (ell >= 30)
m = p_ell >= 30
n = np.sum(m)
chi2_q = np.sum(((p_Dl[m] - q_at_p[m])/p_err[m])**2)
chi2_l = np.sum(((p_Dl[m] - l_at_p[m])/p_err[m])**2)
chi2_b = np.sum(((p_Dl[m] - p_best[m])/p_err[m])**2)

print(f"EE: {n} bins (ell >= 30)")
print(f"QNM:        chi2={chi2_q:.1f}, chi2/dof={chi2_q/(n-2):.3f}")
print(f"CLASS LCDM: chi2={chi2_l:.1f}, chi2/dof={chi2_l/(n-2):.3f}")
print(f"Planck BF:  chi2={chi2_b:.1f}, chi2/dof={chi2_b/(n-2):.3f}")

# Ratio at key ells
for target in [400, 600, 800, 1000]:
    idx = np.argmin(np.abs(p_ell - target))
    print(f"  ell~{p_ell[idx]:.0f}: QNM/Planck={q_at_p[idx]/p_Dl[idx]:.3f}, LCDM/Planck={l_at_p[idx]/p_Dl[idx]:.3f}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios':[3,1]})
ax1.errorbar(p_ell, p_Dl, yerr=[p_errm, p_errp], fmt='o', ms=3, color='gray', alpha=0.7, capsize=1, label='Planck 2018 EE')
ax1.plot(l_ell, l_ee, 'r-', lw=1.5, label='ΛCDM')
ax1.plot(q_ell, q_ee, 'b-', lw=1.5, alpha=0.8, label='QNM (same params as TT)')
ax1.set_xlim(2, 2100)
ax1.set_ylim(0, 45)
ax1.set_ylabel('D_ℓ [μK²]')
ax1.set_title('CMB EE Power Spectrum: QNM vs ΛCDM vs Planck 2018 (ZERO additional parameters)')
ax1.legend()
ax1.grid(True, alpha=0.3)

r_q = q_at_p / p_Dl
r_l = l_at_p / p_Dl
ax2.plot(p_ell, r_l, 'r.', ms=4, alpha=0.6, label='ΛCDM/Planck')
ax2.plot(p_ell, r_q, 'b.', ms=4, alpha=0.6, label='QNM/Planck')
ax2.axhline(1, color='k', ls='--', lw=0.5)
ax2.set_xlim(2, 2100)
ax2.set_ylim(0.7, 1.3)
ax2.set_xlabel('ℓ')
ax2.set_ylabel('Model / Data')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cl_ee_qnm_lcdm_planck.png', dpi=150)
print("Saved: cl_ee_qnm_lcdm_planck.png")
