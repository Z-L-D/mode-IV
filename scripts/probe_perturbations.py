#!/usr/bin/env python3
"""
probe_perturbations.py — Map the CLASS perturbation module
Run from ~/qnm-boltzmann/

Goal: Identify exactly where to modify CLASS to replace growing CDM
perturbations with a fixed gravitational potential template.

We need to find:
  1. The derivative function (perturb_derivs or equivalent)
  2. δ'_cdm and θ'_cdm evolution equations
  3. The Poisson equation / Einstein constraint (where Φ, Ψ are computed)
  4. Where the metric source terms feed into the Boltzmann hierarchy
  5. The ISW contribution (∝ Φ' + Ψ') — this is what changes most
"""
import os
import re

BASE = os.path.expanduser("~/qnm-boltzmann")
CLASS_DIR = os.path.join(BASE, "class_qnm")

# Files to inspect
perturb_c = os.path.join(CLASS_DIR, "source", "perturbations.c")
perturb_h = os.path.join(CLASS_DIR, "include", "perturbations.h")

def search_file(filepath, patterns, context=3, max_per_pattern=8):
    """Search file for patterns, print surrounding context."""
    if not os.path.exists(filepath):
        print(f"  FILE NOT FOUND: {filepath}")
        return
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for pat_label, pat in patterns:
        matches = []
        regex = re.compile(pat, re.IGNORECASE)
        for i, line in enumerate(lines):
            if regex.search(line):
                matches.append(i)
        
        if matches:
            print(f"\n  [{pat_label}] — {len(matches)} matches")
            for m in matches[:max_per_pattern]:
                start = max(0, m - context)
                end = min(len(lines), m + context + 1)
                for j in range(start, end):
                    marker = ">>>" if j == m else "   "
                    print(f"  {marker} L{j+1:5d}: {lines[j].rstrip()}")
                if m != matches[min(max_per_pattern-1, len(matches)-1)]:
                    print(f"  {'─'*60}")
            if len(matches) > max_per_pattern:
                print(f"  ... and {len(matches)-max_per_pattern} more matches")

print("=" * 70)
print("CLASS PERTURBATION MODULE PROBE")
print("=" * 70)

# ================================================================
# PART 1: Find the CDM evolution equations
# ================================================================
print(f"\n{'='*70}")
print("PART 1: CDM perturbation equations (δ'_cdm, θ'_cdm)")
print(f"{'='*70}")

search_file(perturb_c, [
    ("delta_cdm derivative", r"delta.*cdm.*=.*(-|).*theta"),
    ("theta_cdm derivative", r"theta.*cdm.*=.*(-|).*a_prime_over_a"),
    ("dy\[.*cdm\]", r"dy\[.*cdm"),
    ("ppw->pvec.*cdm", r"ppw.*pvec.*cdm|pvecmetric.*cdm"),
], context=4, max_per_pattern=5)

# ================================================================
# PART 2: Find the Poisson / Einstein equations
# ================================================================
print(f"\n{'='*70}")
print("PART 2: Poisson equation / metric potentials (Φ, Ψ)")
print(f"{'='*70}")

search_file(perturb_c, [
    ("Poisson equation", r"poisson|4.*pi.*G|k2.*phi|k2.*psi"),
    ("metric potential phi", r"phi\b.*=.*rho|phi\b.*delta|pvecmetric\[.*phi"),
    ("metric potential psi", r"psi\b.*=|pvecmetric\[.*psi"),
    ("Einstein constraint", r"einstein|constraint|00.*equation"),
], context=3, max_per_pattern=5)

# ================================================================
# PART 3: Find the ISW contribution (Φ' + Ψ')
# ================================================================
print(f"\n{'='*70}")
print("PART 3: ISW effect (Φ' + Ψ', metric_prime)")
print(f"{'='*70}")

search_file(perturb_c, [
    ("phi_prime or phi'", r"phi_prime|phi'|dphi|phi_dot"),
    ("psi_prime or psi'", r"psi_prime|psi'|dpsi|psi_dot"),
    ("ISW", r"[iI][sS][wW]"),
    ("metric source", r"metric.*source|source.*metric"),
], context=3, max_per_pattern=5)

# ================================================================
# PART 4: Find the perturb_derivs function signature
# ================================================================
print(f"\n{'='*70}")
print("PART 4: Main derivative function")
print(f"{'='*70}")

search_file(perturb_c, [
    ("perturb_derivs function", r"int\s+perturb_derivs\s*\("),
    ("perturb_einstein function", r"int\s+perturb_einstein\s*\("),
    ("perturb_total_stress", r"int\s+perturb_total_stress"),
], context=5, max_per_pattern=3)

# ================================================================
# PART 5: Index definitions for CDM in the perturbation vector
# ================================================================
print(f"\n{'='*70}")
print("PART 5: CDM index definitions (where in the ODE vector)")
print(f"{'='*70}")

search_file(perturb_h, [
    ("index_pt_delta_cdm", r"index_pt_delta_cdm"),
    ("index_pt_theta_cdm", r"index_pt_theta_cdm"),
    ("index_mt_phi", r"index_mt_phi|index_mt_psi"),
], context=2, max_per_pattern=5)

# Also check perturbations.c for index assignments
search_file(perturb_c, [
    ("cdm index assignment", r"index_pt_delta_cdm\s*=|index_pt_theta_cdm\s*="),
], context=2, max_per_pattern=5)

# ================================================================
# PART 6: How many lines is perturbations.c?
# ================================================================
print(f"\n{'='*70}")
print("PART 6: File sizes")
print(f"{'='*70}")

for fpath in [perturb_c, perturb_h]:
    if os.path.exists(fpath):
        with open(fpath, 'r') as f:
            n = len(f.readlines())
        print(f"  {os.path.basename(fpath)}: {n} lines")

# ================================================================
# PART 7: Extract the FULL CDM section from perturb_derivs
# ================================================================
print(f"\n{'='*70}")
print("PART 7: Full CDM block in derivative function")
print(f"{'='*70}")
print("  (Looking for a contiguous block with 'cdm' references)")

if os.path.exists(perturb_c):
    with open(perturb_c, 'r') as f:
        lines = f.readlines()
    
    # Find regions with dense CDM references
    cdm_lines = []
    for i, line in enumerate(lines):
        if 'cdm' in line.lower() and ('delta' in line.lower() or 
                                       'theta' in line.lower() or
                                       'dy[' in line.lower() or
                                       'y[' in line.lower()):
            cdm_lines.append(i)
    
    if cdm_lines:
        # Find clusters of CDM lines (within 30 lines of each other)
        clusters = []
        current = [cdm_lines[0]]
        for i in range(1, len(cdm_lines)):
            if cdm_lines[i] - cdm_lines[i-1] < 30:
                current.append(cdm_lines[i])
            else:
                clusters.append(current)
                current = [cdm_lines[i]]
        clusters.append(current)
        
        # Print the largest cluster with full context
        largest = max(clusters, key=len)
        start = max(0, largest[0] - 5)
        end = min(len(lines), largest[-1] + 10)
        
        print(f"  Largest CDM block: lines {start+1}-{end+1} "
              f"({len(largest)} CDM references)")
        for j in range(start, end):
            marker = ">>>" if j in largest else "   "
            print(f"  {marker} L{j+1:5d}: {lines[j].rstrip()}")

# ================================================================
# PART 8: Check for perturb_einstein — the constraint equation solver
# ================================================================
print(f"\n{'='*70}")
print("PART 8: perturb_einstein function body (first 80 lines)")
print(f"{'='*70}")

if os.path.exists(perturb_c):
    in_func = False
    func_start = None
    brace_count = 0
    
    for i, line in enumerate(lines):
        if re.search(r'int\s+perturb_einstein\s*\(', line):
            in_func = True
            func_start = i
        if in_func:
            brace_count += line.count('{') - line.count('}')
            if func_start is not None and i - func_start < 80:
                print(f"  L{i+1:5d}: {line.rstrip()}")
            if brace_count <= 0 and i > func_start + 5:
                print(f"  ... function ends at L{i+1}")
                break
    
    if func_start is None:
        print("  perturb_einstein function not found — may be named differently")
        # Try alternative names
        for alt in ['einstein', 'metric', 'poisson']:
            for i, line in enumerate(lines):
                if re.search(rf'int\s+perturb_{alt}\s*\(', line):
                    print(f"  Found: perturb_{alt} at L{i+1}")

print(f"\n{'='*70}")
print("Paste this output back for the modification plan.")
print(f"{'='*70}")
