#!/usr/bin/env bash
# GPU smoke test (Thor): provider check + sentence-chunked stream + CPU EP run.
# Waits (never kills) if the GPU is busy. Usage: ./smoke_tts.sh <image>
set -euo pipefail
IMAGE="${1:?usage: smoke_tts.sh <image>}"
OUT="$(mktemp -d /tmp/paradee-smoke-XXXXXX)"
RUNTIME=(--runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1)

# Co-tenants are fine (125GB Thor); wait only under real memory pressure.
# NEVER kill anyone's processes.
for i in $(seq 1 90); do
  USED=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')
  [ "${USED:-0}" -lt 100000 ] && break
  echo "[smoke] GPU busy (${USED} MiB in use) - sleeping 20s"; sleep 20
done
echo "[smoke] GPU check: ${USED:-0} MiB in use by co-tenants (proceeding)"

echo "[smoke] 1/3 sentence-chunked stream (default providers: CUDA first)"
docker run --rm "${RUNTIME[@]}" -v "$OUT":/data \
  --entrypoint paradee-stream "$IMAGE" \
  "Hello from Jetson Thor. This is the second sentence for chunking." -o /data/paradee_stream.wav
S=$(stat -c%s "$OUT/paradee_stream.wav"); echo "[smoke] wav: $S bytes"

echo "[smoke] 2/3 CUDA EP sanity: session providers for the baked int8 onnx"
docker run --rm "${RUNTIME[@]}" --entrypoint /opt/venv/bin/python "$IMAGE" - <<'PY'
import glob, onnxruntime as rt
m = glob.glob("/opt/hf-cache/**/paradee_int8.onnx", recursive=True) or glob.glob("/opt/hf-cache/**/*.onnx", recursive=True)
assert m, "no onnx found in hf cache"
sess = rt.InferenceSession(m[0], providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("providers:", sess.get_providers())
print("inputs:", [(i.name, i.shape) for i in sess.get_inputs()])
print("outputs:", [(o.name, o.shape) for o in sess.get_outputs()])
PY

echo "[smoke] 3/3 CPU EP forced (safe fallback)"
docker run --rm "${RUNTIME[@]}" -e CUDA_VISIBLE_DEVICES="" -v "$OUT":/data \
  --entrypoint paradee-say "$IMAGE" "CPU fallback check." -o /data/paradee_cpu.wav
S2=$(stat -c%s "$OUT/paradee_cpu.wav"); echo "[smoke] cpu wav: $S2 bytes"

[ "$S" -gt 30000 ] && [ "$S2" -gt 30000 ] && echo "[smoke] PASS (artifacts in $OUT)" \
  || { echo "[smoke] FAIL: wav too small"; exit 1; }
