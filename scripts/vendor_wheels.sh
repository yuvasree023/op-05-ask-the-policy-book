#!/usr/bin/env bash
# Pre-submission step (not part of reproduce.sh): download wheels for every
# pinned requirement into vendor/wheels/ so reproduce.sh can install with
# zero network access (the reviewer's isolated container restricts egress to
# a metered gateway - EVALUATION_PROTOCOL.md).
#
# Build this on the same OS/glibc and CPython minor version as the review
# target (plain x86_64 Linux, python3.10+); pip resolves wheels for the host
# it runs on.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p vendor/wheels
# One wheelhouse for CPython 3.10-3.13 on x86_64 Linux (glibc >= 2.28), so the
# install works offline whichever of those the review image has.
for v in 310 311 312 313; do
  pip3 download -q --only-binary=:all: --implementation cp --python-version "$v" \
    --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 \
    --platform manylinux_2_28_x86_64 -r requirements.txt -d vendor/wheels
done
echo "wheelhouse ready: $(du -sh vendor/wheels | cut -f1), $(ls vendor/wheels | wc -l) wheels"
