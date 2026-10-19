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
pip3 download --only-binary=:all: -r requirements.txt -d vendor/wheels
echo "wheelhouse ready: $(du -sh vendor/wheels | cut -f1), $(ls vendor/wheels | wc -l) wheels"
