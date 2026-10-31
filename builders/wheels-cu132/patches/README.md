Build-time patches vendored here, applied during build. Pinned to upstream tags.

0001-wyoming-faster-whisper-v3.8.1-nexus-env.patch
    Applied to wyoming-faster-whisper v3.8.1 (git clone at tag, then git apply).
    Re-based version of the fork's wyoming-whisper build.sh seds:
      - inject _detect_host() (LAN IP via dummy UDP socket)
      - WHISPER_PROGRAM_NAME env override for the Info event name
      - WHISPER_ADVERTISE_HOST env override (or auto-detect) for zeroconf host
    Needed so the container advertises a routable LAN address for mDNS
    discovery and keeps the fork's "nexus-whisper-stt" naming.

0002-wyoming-piper-v2.5.2-nexus-env.patch
    Same treatment for wyoming-piper v2.5.2 (PIPER_PROGRAM_NAME /
    PIPER_ADVERTISE_HOST).

piper setup.py onnxruntime strip is applied via sed in the Dockerfile
(v1.8.0 hard-depends on onnxruntime; our GPU-built onnxruntime-gpu wheel is
force-installed last and wins - same reconciliation upstream's Dockerfile.gpu
documents).
