#!/usr/bin/env python3
"""
swap_figure_placeholders.py
Replaces \fbox placeholder blocks with \includegraphics in paper4.tex.
Usage: python3 swap_figure_placeholders.py paper4.tex
"""
import sys, shutil

if len(sys.argv) < 2:
    print("Usage: python3 swap_figure_placeholders.py paper4.tex")
    sys.exit(1)

texpath = sys.argv[1]
shutil.copy2(texpath, texpath + '.bak')

with open(texpath) as f:
    lines = f.readlines()

figures = {
    'fig_cmb_comparison': ('\\textwidth', 'figures/fig_cmb_comparison.pdf'),
    'fig_pk_comparison':  ('\\columnwidth', 'figures/fig_pk_comparison.pdf'),
    'fig_chi2_D':         ('\\columnwidth', 'figures/fig_chi2_D.pdf'),
    'fig_cmb_anomaly':    ('\\columnwidth', 'figures/fig_cmb_anomaly.pdf'),
}

out = []
skip_fbox = False
count = 0

for line in lines:
    # Skip any line that's part of an \fbox placeholder block
    if skip_fbox:
        if '\\fbox{' in line or (skip_fbox and line.strip().startswith('\\fbox')):
            skip_fbox = False  # this line IS the fbox, skip it, done
            continue
        skip_fbox = False

    # Check for commented includegraphics
    matched = False
    for pattern, (width, path) in figures.items():
        if f'%\\includegraphics' in line and pattern in line:
            # Replace with uncommented version pointing to figures/
            out.append(f'\\includegraphics[width={width}]{{{path}}}\n')
            skip_fbox = True  # next non-blank line is the fbox to skip
            count += 1
            print(f'  + {path}')
            matched = True
            break
    
    if not matched:
        # Also skip standalone fbox lines if we encounter them
        stripped = line.strip()
        if stripped.startswith('\\fbox{\\parbox') and any(p in stripped for p in figures):
            continue
        out.append(line)

with open(texpath, 'w') as f:
    f.writelines(out)

print(f'\n{count}/4 figures inserted. Backup at {texpath}.bak')
