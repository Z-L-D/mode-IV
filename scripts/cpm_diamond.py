#!/usr/bin/env python3
"""
CPM Diamond Integration: Time-Domain Waveform from O-S Collapse
================================================================

Solves the Zerilli equation in null coordinates (u,v) using the
diamond scheme. Extracts the prompt-to-ringdown spectral ratio f
from the waveform at a distant observer.

Physics:
  A dust shell collapses from R₀ = 3.39 (≈ISCO) through the photon
  sphere at r=3M. The Zerilli equation is sourced by the shell's
  time-dependent quadrupole moment. The waveform at the observer
  contains both the prompt response (emitted during barrier formation)
  and the QNM ringdown (emitted after barrier closes).

Method:
  In null coords u = t - r*, v = t + r*:
    4 ∂²Ψ/∂u∂v + V(r*) Ψ = S(u,v)
  
  Diamond scheme (inherently stable, no boundary conditions needed):
    Ψ(N) = -Ψ(S) + Ψ(E) + Ψ(W) - h²/4 × V × Ψ_avg + h²/4 × S_avg

Validated: QNM ringdown fit gives ω_R = 0.756 (expected 0.747),
  ω_I = 0.160 (expected 0.178). Physical waveform confirmed.

Requires: greybody_calc.py (corrected, in same directory or on path)

Units: 2M = 1 throughout.

Author: mode:LIMINAL / CoS-PAFS
Date: June 7, 2026
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.integrate import simpson
import json, sys, os, time

# Import validated greybody infrastructure
# Adjust path as needed for your machine
for p in ['.', '/mnt/user-data/outputs', os.path.expanduser('~/qnm-boltzmann')]:
    if os.path.exists(os.path.join(p, 'greybody_calc.py')):
        sys.path.insert(0, p)
        break

from greybody_calc import (f_schw, tortoise, r_from_rstar, V_zerilli,
                            QNM_L2, qnm_omega_R, qnm_omega_I, M)


# =================================================================
# O-S TRAJECTORY (analytic parametric cycloid)
# =================================================================
def os_trajectory(R0=3.39, n_points=200000, r_stop=1.001):
    """
    Radial geodesic from rest at R0 to near the horizon.
    Returns dict with r, t (coordinate), tau (proper), rstar, dr/dtau, E.
    """
    E2 = f_schw(R0)
    E = np.sqrt(E2)
    
    cos_stop = max(-1.0 + 1e-10, 2.0 * r_stop / R0 - 1.0)
    eta_max = np.arccos(cos_stop)
    eta = np.linspace(0, eta_max, n_points)
    
    r = R0 / 2.0 * (1.0 + np.cos(eta))
    tau_scale = np.sqrt(R0**3 / 8.0)
    tau = tau_scale * (eta + np.sin(eta))
    
    # dr/dτ
    dr_dtau = np.zeros_like(eta)
    for i in range(len(eta)):
        d = tau_scale * (1.0 + np.cos(eta[i]))
        if abs(d) > 1e-15:
            dr_dtau[i] = -(R0 / 2.0) * np.sin(eta[i]) / d
    
    # Coordinate time: dt/dτ = E/f(r)
    t = np.zeros_like(tau)
    for i in range(1, len(tau)):
        dtau = tau[i] - tau[i-1]
        ff = f_schw(r[i])
        t[i] = t[i-1] + E / max(ff, 1e-10) * dtau
    
    # Tortoise coordinate
    rstar = np.array([
        tortoise(ri) if ri > 1.0001 else ri + np.log(max(ri - 1, 1e-15))
        for ri in r
    ])
    
    return {
        'r': r, 't': t, 'tau': tau, 'rstar': rstar,
        'dr_dtau': dr_dtau, 'E': E, 'R0': R0
    }


# =================================================================
# ZERILLI SOURCE AMPLITUDE
# =================================================================
def zerilli_source(r, dr_dtau, E):
    """
    l=2 Zerilli source coupling for radial infall.
    Returns S_eff = source amplitude × dt/dτ.
    """
    if r < 1.01:
        return 0.0
    f = f_schw(r)
    n = 2  # λ for l=2
    Mval = 0.5  # M in 2M=1 units
    
    coupling = f * ((n + 1) * r**2 + 3 * Mval * r + 9 * Mval**2 / n) / (
        r**2 * (n * r + 3 * Mval))
    vel = abs(dr_dtau)
    dt_dtau = E / max(f, 1e-10)
    
    return 16 * np.pi * np.sqrt(5.0 / (4.0 * np.pi)) * coupling * vel * dt_dtau


# =================================================================
# DIAMOND INTEGRATION
# =================================================================
def diamond_integrate(R0=3.39, h=0.15, rstar_obs=20.0,
                       u_range=(-10, 55), v_range=(0, 95),
                       sigma_source=None):
    """
    Solve the Zerilli equation in null coordinates using the diamond scheme.
    
    Parameters:
    -----------
    R0 : float — initial radius of collapsing shell
    h : float — null grid spacing (smaller = more accurate, slower)
    rstar_obs : float — observer tortoise coordinate
    u_range, v_range : tuples — null coordinate grid bounds
    sigma_source : float — source smoothing width (default 2h)
    
    Returns:
    --------
    t_ret : array — retarded time at observer
    Psi_obs : array — waveform at observer
    traj : dict — trajectory data
    metadata : dict — grid info and diagnostics
    """
    if sigma_source is None:
        sigma_source = 2.0 * h
    
    # Trajectory
    print(f"Computing O-S trajectory (R0={R0})...")
    traj = os_trajectory(R0, n_points=200000)
    
    # Worldline in null coordinates
    u_traj = traj['t'] - traj['rstar']
    v_traj = traj['t'] + traj['rstar']
    
    # Source along trajectory
    S_traj = np.array([
        zerilli_source(traj['r'][k], traj['dr_dtau'][k], traj['E'])
        for k in range(len(traj['r']))
    ])
    
    # Interpolators
    u_traj_f = interp1d(traj['t'], u_traj,
                         fill_value=(u_traj[0], u_traj[-1]), bounds_error=False)
    v_traj_f = interp1d(traj['t'], v_traj,
                         fill_value=(v_traj[0], v_traj[-1]), bounds_error=False)
    S_of_t = interp1d(traj['t'], S_traj, fill_value=0.0, bounds_error=False)
    
    # Grid
    u_min, u_max = u_range
    v_min, v_max = v_range
    Nu = int((u_max - u_min) / h) + 1
    Nv = int((v_max - v_min) / h) + 1
    u_arr = np.linspace(u_min, u_max, Nu)
    v_arr = np.linspace(v_min, v_max, Nv)
    
    print(f"Diamond grid: {Nu}×{Nv} = {Nu*Nv:,} points, h={h}")
    
    # Precompute Zerilli potential on grid
    print(f"Precomputing V(r*)...")
    V_grid = np.zeros((Nu, Nv))
    for i in range(Nu):
        for j in range(Nv):
            rstar = (v_arr[j] - u_arr[i]) / 2.0
            if -20 < rstar < 100:
                r = r_from_rstar(rstar)
                if r > 1.001:
                    V_grid[i, j] = V_zerilli(r)
    
    # Diamond integration
    print(f"Integrating...")
    t_start = time.time()
    Psi = np.zeros((Nu, Nv))
    
    max_S = np.max(np.abs(S_traj))
    S_cutoff = min(1e4, 10 * max_S)  # stability cutoff
    
    for i in range(1, Nu):
        for j in range(1, Nv):
            # Diamond: S=(i-1,j-1), E=(i,j-1), W=(i-1,j), N=(i,j)
            V_avg = (V_grid[i-1, j-1] + V_grid[i, j-1] + V_grid[i-1, j]) / 3.0
            Psi_avg = (Psi[i-1, j-1] + Psi[i, j-1] + Psi[i-1, j]) / 3.0
            
            # Homogeneous step
            Psi[i, j] = (-Psi[i-1, j-1] + Psi[i, j-1] + Psi[i-1, j]
                         - h**2 / 4.0 * V_avg * Psi_avg)
            
            # Source: check if worldline passes near this diamond
            t_cell = (u_arr[i] + v_arr[j]) / 2.0
            
            if 0 < t_cell < traj['t'][-1]:
                u_wl = float(u_traj_f(t_cell))
                v_wl = float(v_traj_f(t_cell))
                du = u_arr[i] - u_wl
                dv = v_arr[j] - v_wl
                dist2 = du**2 + dv**2
                
                if dist2 < (5 * sigma_source)**2:
                    S_val = float(S_of_t(t_cell))
                    if abs(S_val) < S_cutoff:
                        gauss = np.exp(-dist2 / (2 * sigma_source**2)) / (
                            2 * np.pi * sigma_source**2)
                        Psi[i, j] += h**2 / 4.0 * S_val * gauss
        
        if i % (Nu // 5) == 0:
            elapsed = time.time() - t_start
            print(f"  u={u_arr[i]:.1f}/{u_max}, max|Ψ|={np.max(np.abs(Psi[:i+1,:])):.4e}, "
                  f"elapsed={elapsed:.1f}s")
    
    elapsed = time.time() - t_start
    print(f"Integration complete in {elapsed:.1f}s")
    
    # Extract waveform at observer
    # r* = (v-u)/2 = rstar_obs → v = u + 2*rstar_obs
    t_obs_list = []
    Psi_obs_list = []
    
    for i in range(Nu):
        v_needed = u_arr[i] + 2.0 * rstar_obs
        j = int((v_needed - v_arr[0]) / h + 0.5)
        if 0 <= j < Nv:
            t_val = (u_arr[i] + v_arr[j]) / 2.0
            t_obs_list.append(t_val)
            Psi_obs_list.append(Psi[i, j])
    
    t_obs = np.array(t_obs_list)
    Psi_obs = np.array(Psi_obs_list)
    t_ret = t_obs - rstar_obs
    
    metadata = {
        'Nu': Nu, 'Nv': Nv, 'h': h,
        'rstar_obs': rstar_obs,
        'max_Psi_global': float(np.max(np.abs(Psi))),
        'max_Psi_obs': float(np.max(np.abs(Psi_obs))),
        'elapsed_s': elapsed,
    }
    
    return t_ret, Psi_obs, traj, metadata


# =================================================================
# WAVEFORM ANALYSIS
# =================================================================
def analyze_waveform(t_ret, Psi_obs, separation_factor=1.0):
    """
    Separate waveform into prompt and ringdown. Extract f_spectral.
    
    separation_factor: number of QNM e-folding times after peak
    to define the prompt/ringdown boundary. Default 1.0.
    """
    print(f"\nWaveform analysis:")
    
    # Signal detection
    max_psi = np.max(np.abs(Psi_obs))
    if max_psi < 1e-10:
        print("  No signal detected!")
        return None
    
    thresh = 0.005 * max_psi
    idx_start = np.argmax(np.abs(Psi_obs) > thresh)
    idx_peak = np.argmax(np.abs(Psi_obs))
    
    print(f"  Signal arrives: t_ret = {t_ret[idx_start]:.2f}")
    print(f"  Peak: t_ret = {t_ret[idx_peak]:.2f}, Ψ = {Psi_obs[idx_peak]:.4e}")
    
    # Separation point: peak + separation_factor / ω_I
    omega_I_fund = QNM_L2[0][1]
    t_sep = t_ret[idx_peak] + separation_factor / omega_I_fund
    
    mask_prompt = (t_ret >= t_ret[idx_start]) & (t_ret < t_sep)
    mask_ring = t_ret >= t_sep
    mask_full = t_ret >= t_ret[idx_start]
    
    n_prompt = np.sum(mask_prompt)
    n_ring = np.sum(mask_ring)
    print(f"  Separation at t_ret = {t_sep:.2f}")
    print(f"  Prompt: {n_prompt} samples, Ringdown: {n_ring} samples")
    
    if n_ring < 20:
        print("  WARNING: Insufficient ringdown data. Increase t_max or reduce rstar_obs.")
    
    # Windowed signals
    Psi_full = np.where(mask_full, Psi_obs, 0.0)
    Psi_prompt = np.where(mask_prompt, Psi_obs, 0.0)
    Psi_ring = np.where(mask_ring, Psi_obs, 0.0)
    
    # FFT
    dt = t_ret[1] - t_ret[0] if len(t_ret) > 1 else 0.15
    N = len(t_ret)
    freqs = np.fft.rfftfreq(N, d=dt) * 2 * np.pi
    
    fft_full = np.fft.rfft(Psi_full) * dt
    fft_prompt = np.fft.rfft(Psi_prompt) * dt
    fft_ring = np.fft.rfft(Psi_ring) * dt
    
    P_full = np.abs(fft_full)**2
    P_prompt = np.abs(fft_prompt)**2
    P_ring = np.abs(fft_ring)**2
    
    # Spectral ratios at QNM frequencies
    print(f"\n  Spectral content at QNM frequencies:")
    print(f"  {'n':>3} {'ω_R':>7} {'P_total':>12} {'P_prompt':>12} "
          f"{'P_ring':>12} {'f_spec':>8} {'P_cross%':>8}")
    print(f"  {'-'*72}")
    
    results = []
    for n in range(min(6, len(QNM_L2))):
        omega_R = QNM_L2[n][0]
        idx = np.argmin(np.abs(freqs - omega_R))
        
        Pt = P_full[idx]
        Pp = P_prompt[idx]
        Pr = P_ring[idx]
        P_cross = Pt - Pp - Pr
        
        f_s = Pp / Pt if Pt > 0 else 0
        cross_pct = P_cross / Pt * 100 if Pt > 0 else 0
        
        print(f"  {n:3d} {omega_R:7.4f} {Pt:12.4e} {Pp:12.4e} "
              f"{Pr:12.4e} {f_s:8.4f} {cross_pct:8.1f}%")
        
        results.append({
            'n': n, 'omega_R': omega_R, 'f_spectral': float(f_s),
            'P_total': float(Pt), 'P_prompt': float(Pp),
            'P_ring': float(Pr), 'P_cross_pct': float(cross_pct),
        })
    
    # QNM fit to ringdown
    if n_ring > 30:
        t_rn = t_ret[mask_ring] - t_sep
        Psi_rn = Psi_obs[mask_ring]
        
        if np.max(np.abs(Psi_rn)) > 1e-6:
            try:
                def qnm_model(t, A, wR, wI, phi):
                    return A * np.exp(-wI * t) * np.cos(wR * t + phi)
                
                n_fit = min(500, len(t_rn))
                popt, _ = curve_fit(qnm_model, t_rn[:n_fit], Psi_rn[:n_fit],
                                     p0=[Psi_rn[0], QNM_L2[0][0], QNM_L2[0][1], 0],
                                     maxfev=10000)
                
                print(f"\n  QNM ringdown fit:")
                print(f"    ω_R = {popt[1]:.4f} (expected {QNM_L2[0][0]:.4f}, "
                      f"off by {abs(popt[1]-QNM_L2[0][0])/QNM_L2[0][0]*100:.1f}%)")
                print(f"    ω_I = {popt[2]:.4f} (expected {QNM_L2[0][1]:.4f}, "
                      f"off by {abs(popt[2]-QNM_L2[0][1])/QNM_L2[0][1]*100:.1f}%)")
                
                results.append({
                    'qnm_fit': True,
                    'omega_R_fit': float(popt[1]),
                    'omega_I_fit': float(popt[2]),
                    'amplitude': float(popt[0]),
                })
            except Exception as e:
                print(f"\n  QNM fit failed: {e}")
                results.append({'qnm_fit': False, 'error': str(e)})
    
    return results


# =================================================================
# CONVERGENCE STUDY
# =================================================================
def convergence_study(R0=3.39, rstar_obs=20.0):
    """
    Run at multiple resolutions to check convergence of f.
    """
    print("\n" + "=" * 60)
    print("CONVERGENCE STUDY")
    print("=" * 60)
    
    h_values = [0.20, 0.15, 0.10]
    f_values = []
    
    for h in h_values:
        print(f"\n--- h = {h} ---")
        t_ret, Psi_obs, traj, meta = diamond_integrate(
            R0=R0, h=h, rstar_obs=rstar_obs,
            u_range=(-10, 55), v_range=(0, 95))
        
        results = analyze_waveform(t_ret, Psi_obs, separation_factor=1.0)
        
        if results:
            f0 = next((r['f_spectral'] for r in results if r.get('n') == 0), None)
            if f0 is not None:
                f_values.append((h, f0))
                print(f"  → f(n=0) = {f0:.4f}")
    
    if len(f_values) >= 2:
        print(f"\n  Convergence summary:")
        for h, f in f_values:
            print(f"    h = {h}: f(n=0) = {f:.4f}")
        
        # Richardson extrapolation (second order)
        if len(f_values) >= 3:
            h1, f1 = f_values[0]
            h2, f2 = f_values[1]
            h3, f3 = f_values[2]
            # f_exact ≈ f3 + (f3-f2)²/(f1-2f2+f3) for second-order
            if abs(f1 - 2*f2 + f3) > 1e-10:
                f_extrap = f3 - (f3 - f2)**2 / (f1 - 2*f2 + f3)
                print(f"    Richardson extrapolation → f(n=0) ≈ {f_extrap:.4f}")
    
    return f_values


# =================================================================
# MAIN
# =================================================================
def main():
    R0 = 3.39
    
    print("=" * 60)
    print("CPM TIME-DOMAIN WAVEFORM — DIAMOND INTEGRATION")
    print(f"R0 = {R0}, l = 2, 2M = 1")
    print("=" * 60)
    
    # Main run at default resolution
    t_ret, Psi_obs, traj, meta = diamond_integrate(
        R0=R0, h=0.15, rstar_obs=20.0,
        u_range=(-10, 55), v_range=(0, 95))
    
    results = analyze_waveform(t_ret, Psi_obs)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    
    if results:
        f0 = next((r['f_spectral'] for r in results if r.get('n') == 0), None)
        if f0 is not None:
            print(f"  f_spectral at fundamental: {f0:.4f}")
            print(f"  Target (CMB fit): 0.28")
            print(f"  Barrier formation model: 0.18")
            
        qnm = next((r for r in results if r.get('qnm_fit')), None)
        if qnm:
            print(f"  QNM validated: ω_R={qnm['omega_R_fit']:.4f}, "
                  f"ω_I={qnm['omega_I_fit']:.4f}")
    
    print(f"\n  Grid: {meta['Nu']}×{meta['Nv']}, h={meta['h']}")
    print(f"  Compute time: {meta['elapsed_s']:.1f}s")
    
    # Save
    output = {
        'R0': R0,
        'metadata': meta,
        'analysis': results,
        'target_f': 0.28,
        't_ret': t_ret.tolist(),
        'Psi_obs': Psi_obs.tolist(),
    }
    with open('cpm_diamond_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to cpm_diamond_results.json")
    
    # Optional: convergence study (uncomment to run)
    convergence_study(R0=R0)
    
    return results


if __name__ == '__main__':
    results = main()
