#!/usr/bin/env bash
set -euo pipefail

# Optional external GPL-3.0 tools fetcher (keeps GPL binaries out of Git).
# Installs into ./tools/ (gitignored) and verifies executables.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="${ROOT_DIR}/tools"
mkdir -p "${TOOLS_DIR}"

echo "[quadvox] External GPL-3.0 tool installation guide:"
echo "  1. Blender 5.3 (GPL-3.0): https://www.blender.org/download/"
echo "     Ensure 'blender' is on PATH or export QUADVOX_BLENDER_BIN=/path/to/blender"
echo ""
echo "  2. QuadWild Bi-MDF (GPL-3.0): https://github.com/cgg-bern/quadwild-bimdf"
echo "     Clone/extract release binaries into: ${TOOLS_DIR}/quadwild/"
echo "     or export QUADVOX_QUADWILD_DIR=/path/to/quadwild"
echo ""
echo "  3. ReQuad Blender Extension (GPL-3.0): https://github.com/mlstr0m/requad"
echo "     Extract extension + linux engine into: ${TOOLS_DIR}/requad/"
echo "     or export QUADVOX_REQUAD_DIR=/path/to/requad"
echo ""

if command -v blender >/dev/null 2>&1; then
  echo "[OK] Found blender on PATH: $(command -v blender)"
else
  echo "[WARN] 'blender' not found on PATH."
fi
