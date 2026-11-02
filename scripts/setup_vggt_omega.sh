#!/usr/bin/env bash
# Fetch the VGGT-Omega source at the commit used for the final entry into third_party/vggt-omega.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6
if [ ! -d third_party/vggt-omega/.git ]; then
  git clone https://github.com/facebookresearch/vggt-omega.git third_party/vggt-omega
fi
git -C third_party/vggt-omega checkout --quiet "$COMMIT"
echo "vggt-omega at $(git -C third_party/vggt-omega rev-parse HEAD)"
