#!/usr/bin/env bash
# Demonstrate that local inference makes no outbound network connections.
#
# The model must already be cached -- the first download obviously uses the
# network. Run `make pull MODEL=<key>` first, then this.
#
# Method: start a generation, then poll lsof for sockets owned by that process
# tree and report anything that is not loopback.

set -uo pipefail

MODEL_KEY="${1:-12b}"
cd "$(dirname "$0")/.."

MODEL_ID="$(uv run python scripts/resolve_model.py "$MODEL_KEY")" || exit 1

echo "Model:  $MODEL_ID"
echo "Method: generate while polling lsof for non-loopback sockets"
echo

LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT

uv run mlx_lm.generate \
  --model "$MODEL_ID" \
  --prompt "Say the single word: offline" \
  --max-tokens 32 >"$LOG" 2>&1 &
PID=$!

EXTERNAL=""
while kill -0 "$PID" 2>/dev/null; do
  # -n/-P avoid DNS and service lookups, which would themselves hit the network.
  SOCKETS="$(lsof -nP -i -a -p "$PID" 2>/dev/null | tail -n +2)"
  if [ -n "$SOCKETS" ]; then
    HITS="$(echo "$SOCKETS" | grep -vE '127\.0\.0\.1|\[::1\]|localhost' || true)"
    [ -n "$HITS" ] && EXTERNAL+="$HITS"$'\n'
  fi
  sleep 0.25
done
wait "$PID"
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
  echo "Generation failed -- the check is inconclusive."
  echo "Make sure the model is cached: make pull MODEL=$MODEL_KEY"
  echo
  tail -20 "$LOG"
  exit 1
fi

echo "--- model output ---"
tail -5 "$LOG"
echo "--------------------"
echo

if [ -n "$EXTERNAL" ]; then
  echo "OUTBOUND CONNECTIONS OBSERVED:"
  echo "$EXTERNAL"
  echo "Investigate before sending private data through this model."
  exit 1
fi

echo "PASS  No non-loopback sockets observed during inference."
echo
echo "For a stronger guarantee, turn Wi-Fi off entirely and re-run:"
echo "  networksetup -setairportpower en0 off && make gen MODEL=$MODEL_KEY"
