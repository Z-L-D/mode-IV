#!/bin/bash
# fetch_and_patch.sh — reconstruct class_qnm from upstream CLASS + patch
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE=e85808324f51fc694d12e3ed7439552a3c3f9540

if [ ! -d class_qnm ]; then
    echo "Cloning CLASS (pinned to base commit $BASE)..."
    git clone https://github.com/lesgourg/class_public class_qnm
    cd class_qnm && git checkout $BASE && cd ..
fi

echo "Applying QNM patch..."
cd class_qnm
git apply "$HERE/qnm_fork.patch" || patch -p1 < "$HERE/qnm_fork.patch"

echo "Copying fork-only files (modified sources for browsing)..."
(cd "$HERE/class_qnm_mod" && tar cf - .) | tar xf -

echo "Copying configs and weight files..."
cp "$HERE"/configs/*.ini . 2>/dev/null || true
cp "$HERE"/weights/*.dat . 2>/dev/null || true

echo "Building..."
make -j4 class

echo ""
echo "Reproduce the headline result:"
echo "  ./class qnm_verify.ini"
echo "  python3 ../scripts/compare_all_lensed.py"
