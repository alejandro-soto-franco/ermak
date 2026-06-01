#!/usr/bin/env bash
# Run a command inside a systemd memory-capped scope, so an ermak build or
# simulation can never OOM this machine. This is the OS-level backstop beneath
# the in-process MemoryBudget guardrails: even a bug that ignored the budget is
# killed by the kernel inside the scope instead of taking down the box.
#
#   scripts/run-bounded.sh cargo test
#   ERMAK_MEM_MAX=12G scripts/run-bounded.sh cargo build --release --features gpu
#   ERMAK_MEM_MAX=4G  scripts/run-bounded.sh cargo run --release --example crowding_sweep
#
# Caps (override via env):
#   ERMAK_MEM_MAX   hard RAM ceiling for the scope   (default 8G)
#   ERMAK_SWAP_MAX  swap ceiling on top of RAM        (default 2G)
#
# Note: no `set -u` (the box's shell-harness aborts on unset vars in some envs).
set -eo pipefail

MEM_MAX="${ERMAK_MEM_MAX:-8G}"
SWAP_MAX="${ERMAK_SWAP_MAX:-2G}"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 64
fi

if ! command -v systemd-run >/dev/null 2>&1; then
  echo "run-bounded: systemd-run not found; running UNCAPPED: $*" >&2
  exec "$@"
fi

echo "run-bounded: MemoryMax=$MEM_MAX MemorySwapMax=$SWAP_MAX :: $*" >&2
exec systemd-run --user --scope --quiet \
  -p MemoryMax="$MEM_MAX" \
  -p MemorySwapMax="$SWAP_MAX" \
  -- "$@"
