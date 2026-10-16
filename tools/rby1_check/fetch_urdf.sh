#!/usr/bin/env bash
# Sparse-clone rby1-sdk (models/rby1a only) into tools/rby1_check/_ext/rby1-sdk (gitignored).
set -euo pipefail
DEST="$(cd "$(dirname "$0")" && pwd)/_ext/rby1-sdk"
if [ -d "$DEST/.git" ]; then
  git -C "$DEST" pull --ff-only
else
  mkdir -p "$(dirname "$DEST")"
  git clone --depth 1 --filter=blob:none --sparse https://github.com/RainbowRobotics/rby1-sdk.git "$DEST"
  git -C "$DEST" sparse-checkout set models/rby1a
fi
git -C "$DEST" log -1 --format='rby1-sdk %H %cd'
ls "$DEST/models/rby1a/urdf"
