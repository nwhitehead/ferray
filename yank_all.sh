#!/bin/bash
# Yank all ferray crates from crates.io (version 0.1.0)
# Run: bash yank_all.sh

CRATES=(
  ferray-core-macros
  ferray-core
  ferray-ufunc
  ferray-stats
  ferray-io
  ferray-linalg
  ferray-fft
  ferray-random
  ferray-polynomial
  ferray-window
  ferray-strings
  ferray-ma
  ferray-stride-tricks
  ferray-numpy-interop
  ferray-autodiff
  ferray
)

VERSION="0.1.0"

for crate in "${CRATES[@]}"; do
  echo "Yanking $crate@$VERSION..."
  cargo yank --version "$VERSION" "$crate"
  echo ""
done

echo "Done."
