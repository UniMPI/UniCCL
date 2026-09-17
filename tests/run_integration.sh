#!/usr/bin/env bash
# run_integration.sh - run the real-backend integration test (M2.3).
#
# Requires a GPU host or a tf-builder image with NCCL and/or RCCL:
#   docker run --gpus=all -it tf-builder/ubuntu-24.04-gcc13-cuda /bin/bash
#   docker run --device=/dev/kfd --device=/dev/dri -it tf-builder/ubuntu-24.04-gcc13-rocm /bin/bash
#
# Usage:
#   ./run_integration.sh [nccl|rccl] [world_size] [path-to-test_integration]
#
# The backend is selected through the environment exactly like production:
# setting XCCL_BACKEND here is equivalent to exporting it as usual.
set -euo pipefail

BACKEND="${1:-nccl}"
WORLD="${2:-2}"
BIN="${3:-./test_integration}"

if [ ! -x "$BIN" ]; then
    echo "error: $BIN not found or not executable" >&2
    exit 2
fi

UID_FILE="$(mktemp)"
trap 'rm -f "$UID_FILE"' EXIT

echo "== XCCL integration: backend=$BACKEND world_size=$WORLD =="
pids=()
for ((r = 0; r < WORLD; r++)); do
    XCCL_BACKEND="$BACKEND" \
    XCCL_TEST_WORLD_SIZE="$WORLD" \
    XCCL_TEST_WORLD_RANK="$r" \
    XCCL_TEST_UID_FILE="$UID_FILE" \
        "$BIN" &
    pids+=("$!")
done

rc=0
for p in "${pids[@]}"; do
    wait "$p" || rc=1
done
if [ "$rc" -ne 0 ]; then
    echo "== integration FAILED (backend=$BACKEND) ==" >&2
    exit 1
fi
echo "== integration PASSED (backend=$BACKEND, world=$WORLD) =="
exit 0
