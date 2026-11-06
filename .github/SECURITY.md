# Security

## Report a Vulnerability

Please do **not** open public issues for security bugs. Use GitHub's private
vulnerability disclosure (*Security → Report a vulnerability* on this
repository), or contact the maintainer (**@narandill**) directly.

## Scope

voice-containers is a set of self-contained voice images (`narandill/<engine>`)
built from the Dockerfiles under `packages/` — speech recognition, speech
synthesis and a local LLM server for NVIDIA Jetson edge GPUs. Almost all
functionality is upstream software running unmodified; vulnerabilities in
those components belong with the upstream projects.

Two properties of the stack worth knowing:

- **The voice servers are unauthenticated by design.** The Wyoming / HTTP
  endpoints bind host ports from inside the container and trust the LAN — the
  host-side port binding is the entire network boundary. Do not expose them
  beyond a trusted network without an authenticating reverse proxy in front.
- **The harness is a local CLI, not a service.** `bench/` orchestrates docker
  and the GPU from the host and adds no network listeners of its own; the
  voice-api contract (`docs/voice-api.md`) is exercised by
  `tools/voice_api_contract_test.py` against a local reference stub, and
  `deploy/` only composes the already-built images.

The images are built `FROM` Ubuntu 24.04 and NVIDIA CUDA / JetPack-based
runtime images and redistribute those components under their own licenses —
see [`NOTICE`](../NOTICE) and the license files inside each image.
