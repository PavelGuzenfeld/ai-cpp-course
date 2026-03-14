#!/usr/bin/env bash
# Build, test, and package the tracker-utils package.
#
# Usage:
#   cd /workspace/ai-cpp-l9
#   bash build_and_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo " Step 1: Install the package"
echo "============================================"
pip install . --verbose 2>&1 | tail -20
echo ""

echo "============================================"
echo " Step 2: Verify import"
echo "============================================"
python3 -c "
from tracker_utils import BBox, __version__
print(f'  Package version: {__version__}')
b = BBox(10, 20, 100, 50)
print(f'  BBox: {b}')
print(f'  Area: {b.area}')
print(f'  Center: ({b.cx}, {b.cy})')
print('  Import OK')
"
echo ""

echo "============================================"
echo " Step 3: Run tests"
echo "============================================"
pytest test_package.py test_integration_package.py -v
echo ""

echo "============================================"
echo " Step 4: Build a wheel"
echo "============================================"
rm -rf dist/
pip wheel . -w dist/ --no-deps
echo ""

echo "============================================"
echo " Step 5: Wheel contents"
echo "============================================"
WHEEL=$(ls dist/*.whl 2>/dev/null | head -1)
if [ -n "$WHEEL" ]; then
    echo "  Wheel: $(basename "$WHEEL")"
    echo "  Size:  $(du -h "$WHEEL" | cut -f1)"
    echo ""
    echo "  Contents:"
    unzip -l "$WHEEL" | grep -E '^\s+[0-9]' | awk '{print "    " $4}'
else
    echo "  ERROR: No wheel found in dist/"
    exit 1
fi

echo ""
echo "============================================"
echo " Done. Wheel saved to: dist/"
echo "============================================"
