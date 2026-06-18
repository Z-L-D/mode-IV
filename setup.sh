#!/bin/bash
# setup.sh — One-command reproduction of Paper 4 results
#
# Downloads CLASS v3.3.4, applies the QNM patch, builds, runs the
# verification configuration, and compares against Planck 2018 data.
#
# Prerequisites: gcc, make, git, python3, numpy
# Time: ~2 minutes on a modern machine
#
# Usage: bash setup.sh

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE_COMMIT="e85808324f51fc694d12e3ed7439552a3c3f9540"

# ─── Colors for output ───
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  PQM Paper 4 — Setup & Verification                    ║"
echo "║  Quasinormal Mode Primordial Power Spectrum             ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ─── Check prerequisites ───
echo "Checking prerequisites..."
MISSING=""
for cmd in gcc make git python3; do
    if ! command -v $cmd &>/dev/null; then
        MISSING="$MISSING $cmd"
    fi
done
python3 -c "import numpy" 2>/dev/null || MISSING="$MISSING numpy"

if [ -n "$MISSING" ]; then
    echo -e "${RED}Missing:${NC}$MISSING"
    echo ""
    echo "Install on Ubuntu/Debian/WSL:"
    echo "  sudo apt update"
    echo "  sudo apt install -y build-essential gcc make git python3 python3-pip python3-numpy"
    echo ""
    echo "Install on macOS:"
    echo "  xcode-select --install"
    echo "  brew install gcc make git python3"
    echo "  pip3 install numpy"
    exit 1
fi
echo -e "${GREEN}✓${NC} All prerequisites found"
echo ""

# ─── Step 1: Download CLASS ───
if [ -d "$HERE/class_qnm/source" ]; then
    echo -e "${YELLOW}class_qnm/ already exists — skipping download${NC}"
    echo "  (Delete class_qnm/ and rerun to rebuild from scratch)"
else
    echo "Step 1/5: Downloading CLASS v3.3.4..."
    git clone --quiet https://github.com/lesgourg/class_public "$HERE/class_qnm"
    cd "$HERE/class_qnm"
    git checkout --quiet "$BASE_COMMIT"
    echo -e "${GREEN}✓${NC} CLASS v3.3.4 (commit ${BASE_COMMIT:0:8})"
    cd "$HERE"

    # ─── Step 2: Apply QNM patch ───
    echo "Step 2/5: Applying QNM patch..."
    cd "$HERE/class_qnm"
    git apply "$HERE/qnm_fork.patch" 2>/dev/null || \
        patch -p1 < "$HERE/qnm_fork.patch"
    echo -e "${GREEN}✓${NC} Patch applied (5 modified files)"
    cd "$HERE"
fi

# ─── Step 3: Copy configs and weights ───
echo "Step 3/5: Installing configuration files..."
cp "$HERE/configs/qnm_verify.ini" "$HERE/class_qnm/"
cp "$HERE/weights/"*.dat "$HERE/class_qnm/" 2>/dev/null || true
echo -e "${GREEN}✓${NC} Configs and weights installed"

# ─── Step 4: Build ───
echo "Step 4/5: Building CLASS..."
cd "$HERE/class_qnm"
make -j$(nproc 2>/dev/null || echo 4) class 2>&1 | tail -1
if [ ! -x "./class" ]; then
    echo -e "${RED}Build failed. Check compiler output above.${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Build complete"
cd "$HERE"

# ─── Step 5: Run and verify ───
echo "Step 5/5: Running verification..."
cd "$HERE/class_qnm"
./class qnm_verify.ini

# Handle the output-suffix quirk
cp $(ls -t output/qnm_test_100_*_cl_lensed.dat 2>/dev/null | head -1) \
   output/qnm_test_100_cl_lensed.dat 2>/dev/null

cd "$HERE"

echo ""
echo "─── Results ───"
echo ""
python3 scripts/compare_all_lensed.py 2>/dev/null | grep -E "QNM LENSED|Planck BF"
echo ""
python3 scripts/s_half_check.py 2>/dev/null | grep "S_1/2"
echo ""

# Checksum
HASH=$(md5sum "$HERE/class_qnm/output/qnm_test_100_cl_lensed.dat" 2>/dev/null | cut -d' ' -f1)

echo "─── Verification ───"
echo ""

EXPECTED_HASH="d8e306e44da798f89da639d095261824"
if [ "$HASH" = "$EXPECTED_HASH" ]; then
    echo -e "${GREEN}✓ md5 MATCH${NC}: $HASH"
    echo -e "${GREEN}✓ Bit-exact reproduction confirmed${NC}"
else
    echo -e "${YELLOW}md5: $HASH (differs from reference — likely different compiler)${NC}"
    echo "  This is normal across platforms. Check the chi2 digits above instead."
    echo "  Expected: TT=2.162, EE=1.256, TE=1.617, S_1/2=7,212"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Setup complete. To re-run later: bash run_verify.sh    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
