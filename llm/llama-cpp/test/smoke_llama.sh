#!/usr/bin/env bash
# Smoke test on real hardware: full GPU offload + one completion via llama-server.
# Usage: ./smoke_llama.sh [image] [port]
set -euo pipefail

IMAGE="${1:-narandill/llama-cpp:0.8.0}"
PORT="${2:-18080}"
NAME="llama-cpp-smoke"
MODEL_DIR="${MODEL_DIR:-/home/narandill/edge-containers/task/models}"
MODEL_FILE="Qwen_Qwen3-1.7B-Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF/resolve/main/${MODEL_FILE}"
MODEL="$MODEL_DIR/$MODEL_FILE"

mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL" ]; then
  echo "[smoke] downloading $MODEL_FILE (~1.1 GB)"
  wget -q --show-progress -O "$MODEL" "$MODEL_URL"
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "[smoke] starting $IMAGE (NVIDIA_DISABLE_REQUIRE=1 until host is on JP 7.2.1)"
docker run -d --name "$NAME" --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 \
  -p "${PORT}:8080" -v "${MODEL_DIR}:/data:ro" \
  "$IMAGE" \
  llama-server -m "/data/${MODEL_FILE}" -ngl 999 --host 0.0.0.0 --port 8080 -c 4096 -fa on >/dev/null

cleanup() {
  docker logs "$NAME" 2>&1 | grep -E "offload|CUDA|graph|error|failed" | tail -15 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[smoke] waiting for /health"
for i in $(seq 1 120); do
  ST=$(curl -s -o /tmp/smoke_health.json -w '%{http_code}' "http://localhost:${PORT}/health" || true)
  [ "$ST" = "200" ] && break
  sleep 2
  [ "$i" = 120 ] && { echo "[smoke] server did not become healthy"; exit 1; }
done
echo "[smoke] healthy: $(cat /tmp/smoke_health.json)"

echo "[smoke] sending completion"
RESP=$(curl -s "http://localhost:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say exactly: JETSON OK"}],"max_tokens":200,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}')
REPLY=$(echo "$RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null || true)
echo "[smoke] reply: ${REPLY}"
case "$REPLY" in *JETSON*) : ;; *) echo "[smoke] unexpected reply: $RESP"; exit 1 ;; esac

echo "[smoke] GPU layers/timings from server log:"
docker logs "$NAME" 2>&1 | grep -E "offloaded|load_tensors: .*CUDA|prompt eval time|eval time" | tail -6

echo "[smoke] PASS"
