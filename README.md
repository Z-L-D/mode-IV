# A Quasinormal Mode Template for the Primordial Power Spectrum

**Doyle 2026** · [arXiv: pending] · mode:LIMINAL

A primordial power spectrum built from Schwarzschild black hole
quasinormal modes, tested against Planck 2018 CMB data using a
modified [CLASS](https://github.com/lesgourg/class_public) Boltzmann
solver. Two fitted parameters. Five derived from general relativity.
χ²/dof = 2.16 (TT), 1.26 (EE), 1.62 (TE).

---

## Quick start (3 commands)

If you already have a Linux system with `gcc`, `make`, `git`, and `python3` with NumPy:

```bash
git clone https://github.com/[YOUR_USERNAME]/pqm-paper4.git
cd pqm-paper4
bash setup.sh
```

`setup.sh` does everything: downloads CLASS v3.3.4, applies the QNM patch,
builds the solver, runs it, and prints the result. Takes about 2 minutes.

Expected output:
```
=== Verification passed ===
TT chi2/dof = 2.162
EE chi2/dof = 1.256
TE chi2/dof = 1.617
S_1/2       = 7,212 muK^4
md5         = d8e306e44da798f89da639d095261824
```

---

## Full walkthrough (from scratch)

### 1. Get a Linux environment

**Already on Linux?** Skip to step 2.

**On Windows?** Install WSL (Windows Subsystem for Linux):
```powershell
# Run in PowerShell as Administrator
wsl --install -d Ubuntu
```
Restart, set a username/password when prompted, then open Ubuntu from the Start menu.

**On macOS?** The commands below work directly in Terminal. You may need
to install Xcode command line tools (`xcode-select --install`) and
Homebrew (`brew install gcc make git python3`).

### 2. Install dependencies

```bash
# Ubuntu / Debian / WSL
sudo apt update
sudo apt install -y build-essential gcc make git python3 python3-pip python3-numpy
```

That's it. CLASS has no external library dependencies beyond a C compiler.

### 3. Clone this repository

```bash
git clone https://github.com/[YOUR_USERNAME]/pqm-paper4.git
cd pqm-paper4
```

### 4. Build the QNM-modified CLASS

The repository ships a patch file (`qnm_fork.patch`) containing only the
physics modifications against vanilla CLASS v3.3.4. The setup script
downloads CLASS from its official repository, applies the patch, copies
in the configuration files, and compiles:

```bash
bash setup.sh
```

Or, step by step if you prefer:

```bash
# Download CLASS v3.3.4 (pinned to exact commit for reproducibility)
git clone https://github.com/lesgourg/class_public class_qnm
cd class_qnm
git checkout e85808324f51fc694d12e3ed7439552a3c3f9540

# Apply the QNM modifications (~500 lines of physics in 5 files)
git apply ../qnm_fork.patch

# Copy configuration and weight files into the CLASS directory
cp ../configs/qnm_verify.ini .
cp ../weights/*.dat .

# Build (takes ~30 seconds)
make -j4 class

cd ..
```

### 5. Run the verification

```bash
# Run CLASS with the paper's production configuration
cd class_qnm
./class qnm_verify.ini

# CLASS appends a numeric suffix to output files — copy to the canonical name
cp $(ls -t output/qnm_test_100_*_cl_lensed.dat | head -1) \
   output/qnm_test_100_cl_lensed.dat

# Compare against Planck 2018 binned data
cd ..
python3 scripts/compare_all_lensed.py
python3 scripts/s_half_check.py
```

You should see:
```
TT: QNM LENSED chi2/dof = 2.162    (ΛCDM best-fit: 0.794)
EE: QNM LENSED chi2/dof = 1.256    (ΛCDM best-fit: 0.968)
TE: QNM LENSED chi2/dof = 1.617    (ΛCDM best-fit: 1.119)
S_1/2 = 7,212 muK^4                (Planck observed: ~1,065)
```

### 6. Verify the checksum (optional)

The output is bit-deterministic on a given compiler. On the verified
build environment (gcc on Ubuntu):

```bash
md5sum class_qnm/output/qnm_test_100_cl_lensed.dat
# Expected: d8e306e44da798f89da639d095261824
```

Different compilers or optimization levels may produce slightly different
floating-point results. The χ² digits (2.162, 1.256, 1.617) are robust
across platforms; the md5 is the strong check on identical environments.

---

## What the patch changes

The QNM modifications touch 5 files in CLASS v3.3.4. Everything else
(background cosmology, thermodynamics, perturbation evolution, lensing)
is unmodified upstream code.

| File | What changed |
|------|-------------|
| `source/primordial.c` | QNM mode sum replaces inflationary P(k) |
| `include/primordial.h` | New parameter declarations |
| `source/input.c` | .ini parser for QNM parameters |
| `source/perturbations.c` | Late-ISW suppression switch |
| `include/perturbations.h` | ISW flag declaration |

The full diff is human-readable in `qnm_fork.patch` (~500 lines of physics,
the rest is whitespace). The modified source files are also browsable in
`class_qnm_mod/` without needing to apply the patch.

---

## Repository layout

```
pqm-paper4/
├── setup.sh                 # One-command build + verify
├── run_verify.sh            # Re-run verification (after setup)
├── fetch_and_patch.sh       # Download CLASS + apply patch (no verify)
├── qnm_fork.patch           # Unified diff against CLASS v3.3.4
├── class_qnm_mod/           # Modified files (browsable without patch)
├── BASE_COMMIT.txt          # Exact upstream commit hash
│
├── configs/                  # CLASS .ini configurations
│   ├── qnm_verify.ini       # Paper's production config (the one to run)
│   ├── qnm_baseline_paper.ini
│   ├── qnm_cpm_v3.ini       # Two-channel model, n=0-12
│   └── qnm_cpm_v4.ini       # Two-channel model, n=1-2 only
│
├── weights/                  # Per-mode weight tables
│   ├── qnm_weights_v1.dat   # CPM-measured n=1 weight (H1/H0 = 0.136)
│   ├── qnm_weights_v2.dat   # CPM-measured n=1 + n=2
│   ├── qnm_weights_v3.dat   # Zero-parameter two-channel, n=0-12
│   └── qnm_weights_v4.dat   # Zero-parameter two-channel, n=1-2
│
├── data/planck/              # Planck 2018 binned power spectra
│   ├── planck_tt_binned.txt  # 5 columns: ell, D_ell, -sigma, +sigma, BestFit
│   ├── planck_ee_binned.txt
│   └── planck_te_binned.txt
│
├── results/                  # Archived outputs for key configurations
├── figures/                  # Paper figures (PDF + PNG)
├── scripts/                  # Analysis and figure generation
│   ├── compare_all_lensed.py # Chi-squared comparison against Planck
│   ├── s_half_check.py       # S_1/2 angular correlation statistic
│   └── make_paper_figures.py # Regenerate all four paper figures
│
├── paper/                    # LaTeX source
│   ├── paper4.tex
│   └── paper4.bib
│
└── class_qnm/               # [Generated by setup.sh — not tracked in git]
```

---

## Key results at a glance

| Configuration | A_QNM | TT χ²/dof | EE | TE | S₁/₂ (μK⁴) |
|---|---|---|---|---|---|
| **Baseline (paper)** | **3.33e-8** | **2.162** | **1.256** | **1.617** | **7,212** |
| + Measured n=1 weight | 3.12e-8 | 2.154 | 1.252 | 1.608 | 4,390 |
| + Measured n=1,2 | 2.97e-8 | 2.154 | 1.246 | 1.618 | 4,060 |
| + No late-ISW | 3.33e-8 | 2.160 | 1.256 | 1.666 | 2,326 |
| Combined (weights + ISW) | 3.12e-8 | 2.158 | 1.252 | 1.657 | 2,306 |
| Planck observed | — | — | — | — | ~1,065 |
| ΛCDM best-fit | — | 0.794 | 0.968 | 1.119 | ~35,000–48,000 |

The baseline is what `setup.sh` reproduces. Other configurations use
the weight files in `weights/` with `qnm_weight_mode = 2` in the .ini.

---

## Running other configurations

To test a different weight profile (e.g., the CPM-measured overtone weights):

```bash
cd class_qnm

# Copy the weight file and config
cp ../weights/qnm_weights_v1.dat .
cp ../configs/qnm_cpm_v1.ini .

# Edit the .ini to set the correct amplitude (see table above)
# Then run:
./class qnm_cpm_v1.ini

# Copy output and compare
cp $(ls -t output/qnm_test_100_*_cl_lensed.dat | head -1) \
   output/qnm_test_100_cl_lensed.dat
cd .. && python3 scripts/compare_all_lensed.py
```

### Amplitude renormalization

When you change low-overtone weights, the internal normalization
(`w_n = w_table[n] / sum`) shifts the overall amplitude. The converged
amplitude for any weight file is:

```
A_new = 3.33e-8 × (5.609 - Δ) / 5.609
```

where Δ is the total weight removed from the power-law profile.
Verified to 0.1% across three configurations — no scan needed.

---

## Regenerating figures

```bash
python3 scripts/make_paper_figures.py
```

Produces `figures/fig_cmb_comparison.pdf` (Fig. 1), `fig_pk_comparison.pdf`
(Fig. 2), `fig_chi2_D.pdf` (Fig. 3), and `fig_cmb_anomaly.pdf` (Fig. 4).
Figures 2 and 4 are self-contained; Figures 1 and 3 require the CLASS
output and D-scan data respectively.

---

## Provenance

- **CLASS base**: commit `e85808324f51fc694d12e3ed7439552a3c3f9540` (v3.3.4)
- **Patch**: 5 modified files, ~500 lines of physics
- **Planck data**: publicly available binned power spectra from Planck 2018 release
- **Build verified**: bit-deterministic output (md5 `d8e306e44da798f89da639d095261824`)

---

## License

This repository is licensed under the **GNU General Public License v2.0 or later**
(GPLv2+), consistent with the CLASS Boltzmann solver on which the fork is based.
See [LICENSE](LICENSE) for the full text.

---

## Citation

If you use this code or build on these results, please cite:

```bibtex
@article{Doyle2026_QNM,
  title   = {mode-IV: A Quasinormal Mode Template for the Primordial Power Spectrum},
  author  = {Doyle, Zac},
  year    = {2026},
  note    = {arXiv: [pending]},
  doi     = {10.5281/zenodo.20740495}
}
```

and the CLASS solver:

```bibtex
@article{Blas2011,
  title   = {The Cosmic Linear Anisotropy Solving System ({CLASS}). {II}. Approximation schemes},
  author  = {Blas, Diego and Lesgourgues, Julien and Tram, Thomas},
  journal = {JCAP},
  volume  = {2011},
  number  = {07},
  pages   = {034},
  year    = {2011},
}
```
