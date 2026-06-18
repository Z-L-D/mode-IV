#!/usr/bin/env python3
"""
class_runner.py — Shared utility for running CLASS and finding output
======================================================================
Handles CLASS's auto-incrementing output filenames correctly.

CLASS writes output like:
    output/qnm_test_100_01_cl_lensed.dat
    output/qnm_test_100_02_cl_lensed.dat
    ...

This module runs CLASS and finds the NEWEST lensed cl file.
"""

import subprocess
import os
import glob
import time


def find_newest_lensed_cl(search_dir, pattern="*cl_lensed*"):
    """Find the most recently modified lensed Cl file.
    
    Mimics: ls -t output/*_cl_lensed.dat | head -1
    """
    search_path = os.path.join(search_dir, pattern)
    candidates = glob.glob(search_path)
    
    if not candidates:
        # Try broader patterns
        for alt_pattern in ["*lensed*.dat", "*lensed_cl*", "*_cl_lensed*"]:
            search_path = os.path.join(search_dir, alt_pattern)
            candidates = glob.glob(search_path)
            if candidates:
                break
    
    if not candidates:
        return None
    
    # Sort by modification time, newest first (same as ls -t)
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def run_class_and_get_output(class_bin, ini_path, class_dir, output_search_dir):
    """Run CLASS and return path to the newest lensed Cl file.
    
    Args:
        class_bin: path to CLASS binary
        ini_path: path to .ini file
        class_dir: working directory for CLASS
        output_search_dir: directory where CLASS writes output files
        
    Returns:
        path to newest lensed cl file, or None on failure
    """
    # Record files BEFORE the run
    pre_files = set()
    if os.path.exists(output_search_dir):
        pre_files = set(glob.glob(os.path.join(output_search_dir, "*lensed*")))
    
    # Record timestamp before run
    t_before = time.time()
    
    # Run CLASS
    try:
        result = subprocess.run(
            [class_bin, ini_path],
            capture_output=True, text=True, timeout=120,
            cwd=class_dir
        )
        if result.returncode != 0:
            print(f"  CLASS failed: {result.stderr[:300]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  CLASS timed out")
        return None
    
    # Find the NEW file (created after our run started)
    if os.path.exists(output_search_dir):
        post_files = set(glob.glob(os.path.join(output_search_dir, "*lensed*")))
        new_files = post_files - pre_files
        
        if new_files:
            # Sort new files by mtime, newest first
            newest = sorted(new_files, key=os.path.getmtime, reverse=True)[0]
            return newest
    
    # Fallback: find any lensed file modified after we started
    lensed_file = find_newest_lensed_cl(output_search_dir)
    if lensed_file and os.path.getmtime(lensed_file) > t_before:
        return lensed_file
    
    # Last resort: just get the newest one regardless
    if lensed_file:
        print(f"  WARNING: Using pre-existing file {os.path.basename(lensed_file)}")
        return lensed_file
    
    return None


def parse_lensed_cl(filepath):
    """Parse a CLASS lensed Cl file.
    
    Returns dict with keys: ells, tt, ee, te (all lists).
    Values are D_ell = ell*(ell+1)/(2*pi) * C_ell in muK^2.
    """
    ells = []
    tt_vals = []
    ee_vals = []
    te_vals = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    ells.append(int(parts[0]))
                    tt_vals.append(float(parts[1]))
                    ee_vals.append(float(parts[2]))
                    te_vals.append(float(parts[3]))
                except (ValueError, IndexError):
                    continue
    
    if not ells:
        return None
    
    return {"ells": ells, "tt": tt_vals, "ee": ee_vals, "te": te_vals}


def load_planck_data(filepath):
    """Load Planck binned data: returns list of (ell_center, D_ell, sigma)."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    data.append((float(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    continue
    return data


def compute_chi2(planck_data, class_data, spectrum="tt"):
    """Compute chi²/dof for a given spectrum."""
    if class_data is None:
        return float('inf'), 0
    
    class_vals = class_data[spectrum]
    chi2 = 0.0
    n_bins = 0
    
    for ell_c, d_ell_planck, sigma in planck_data:
        idx = int(round(ell_c)) - 2
        if idx < 0 or idx >= len(class_vals) or sigma <= 0:
            continue
        chi2 += ((class_vals[idx] - d_ell_planck) / sigma) ** 2
        n_bins += 1
    
    return chi2 / max(n_bins, 1), n_bins


def compute_regional_chi2(planck_data, class_data, spectrum="tt"):
    """Compute chi² by ell region for TT diagnostic."""
    if class_data is None:
        return {}
    
    regions = {
        "low_ell_30_200": (30, 200),
        "peak1_200_350": (200, 350),
        "trough1_350_450": (350, 450),
        "peak2_450_650": (450, 650),
        "trough2_650_750": (650, 750),
        "peak3_750_950": (750, 950),
        "mid_ell_950_1500": (950, 1500),
        "high_ell_1500_2500": (1500, 2500),
    }
    
    class_vals = class_data[spectrum]
    results = {}
    
    for name, (ell_min, ell_max) in regions.items():
        chi2 = 0.0
        n = 0
        for ell_c, d_ell_planck, sigma in planck_data:
            if ell_c < ell_min or ell_c >= ell_max:
                continue
            idx = int(round(ell_c)) - 2
            if idx < 0 or idx >= len(class_vals) or sigma <= 0:
                continue
            chi2 += ((class_vals[idx] - d_ell_planck) / sigma) ** 2
            n += 1
        results[name] = chi2 / max(n, 1) if n > 0 else float('inf')
    
    return results
