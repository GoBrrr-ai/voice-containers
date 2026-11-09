# Post-mortem — `enable-cuda-compat` kills container create

**Platform:** AGX Orin · JetPack 7.2.1 (L4T R39, rev 2.1) · driver 595.78 · nvidia-container-toolkit **1.19.1**
**Status:** fixed on `main` — commit `9ea3f0c` (2026-11-09) · **User impact:** none (pre-deploy); blocked stack validation for one session

---

## 1. Intro — is an image from this repo affected? (run this first)

The failure needs **two** things at once, and neither alone suffices:

- **Host side:** JetPack 7.x userspace with container-toolkit `1.19.1`, whose generated
  CDI spec carries the `createContainer` hook `enable-cuda-compat`
  (`/var/run/cdi/nvidia.yaml`), **and** the daemon's `default-runtime` set to `nvidia`
  (it is, on this host — so *every* `docker run`, even without `--runtime`, executes the hook).
- **Image side:** the final stage ships the Jetson-flavoured CUDA forward-compat pair
  `/usr/local/cuda/compat/` **and `/usr/local/cuda/compat_orin/`** (the 595.58.03 libs
  that ship with the `cuda` apt packages on cu13.2 Jetson images).

### The 60-second check

```bash
# 1) host preconditions
cat /etc/nv_tegra_release                        # R39 ... => JP 7.x family
nvidia-ctk --version                             # 1.19.1 observed
grep -c enable-cuda-compat /var/run/cdi/nvidia.yaml
docker info | grep -i 'default runtime'          # nvidia => hooks run always

# 2) does the image carry the trigger?
docker run --rm --entrypoint sh IMAGE -c 'ls -d /usr/local/cuda/compat* 2>/dev/null'

# 3) does create actually die?  (hooks run at create, before the process starts)
docker run --rm --runtime nvidia --entrypoint true IMAGE; echo exit=$?
```

### Decision table

| final stage contains | observed create | verdict |
|---|---|---|
| `compat/` **and** `compat_orin/` (595.x-era, cu13.2 jetson base) | **panic, exit 125** | affected → strip (§5) or host fix (§7) |
| `compat/` only, e.g. `libcuda.so.580.95.05` (server-style base) | OK | not affected; stripping harmless |
| no `compat*` at all | OK | fine (post-fix images, CPU-only images) |
| anything, on a host without the hook / other ctk | OK | bug is host-conditional |

### Batch audit — every image built from this repo

```bash
for i in $(docker images --format '{{.Repository}}:{{.Tag}}' | grep '^narandill/'); do
  printf '%-42s ' "$i"
  docker run --rm --entrypoint sh "$i" -c 'ls -d /usr/local/cuda/compat* 2>/dev/null | tr "\n" " "; echo'
done
```

Recorded 2026-11-09 on the Orin (post-fix state):

| image | compat in final stage | `docker run --runtime nvidia` |
|---|---|---|
| `narandill/llama-cpp:0.8.0` | none (stripped, `9ea3f0c`) | OK |
| `narandill/nemotron-streaming-stt:0.2.0` | none (stripped, `9ea3f0c`) | OK |
| `narandill/phonon31m-tts:latest` | none (CPU-only image) | OK (n/a) |
| `narandill/mjolnir-vllm:latest` | `compat/` only — `libcuda.so.580.95.05`, no `compat_orin` | OK |
| `narandill/edge-wheels:cu132` | compat pair present | **panic, exit 125 — reproducible today** |

`edge-wheels:cu132` is a build-stage image (buildkit doesn't run CDI hooks), so its
panic is latent, not blocking — but it is the canonical live repro for §1 check 3.
`builders/parakeet-cpp` has no local image yet: run the audit on it after the first build.

**CI gate (recommended):** after every image build, on a Jetson runner:
`docker run --rm --runtime nvidia --entrypoint true IMAGE || exit 1` — this exact
failure then breaks the build instead of the user's morning.

## 2. Summary — what happened, and what we changed

During the 2026-11-09 stack rebuild, both new GPU runtime images
(`narandill/llama-cpp`, `narandill/nemotron-streaming-stt`) died at **container
create** — never reaching entrypoint or logs — with runc reporting
`error running createContainer hook #2: exit status 2` and a Go panic from
`nvidia-cdi-hook`. Every retry, `--rm`, and compose attempt failed identically; the
previously-working `mjolnir-vllm` image kept working, which localised the cause to
the images, not the runtime.

**The change** (all of it — commit `9ea3f0c`, 2 files, +15 lines, comments included):

```dockerfile
# packages/llm/llama-cpp/Dockerfile   (after ldconfig, ~L122)
# packages/stt/nemotron-streaming-stt/Dockerfile (after venv/pip stage, ~L65)
RUN rm -rf /usr/local/cuda/compat /usr/local/cuda/compat_orin
```

No entrypoint changes, no base-image bumps, no flag changes. Both images rebuilt,
retagged `0.8.0` / `0.2.0` + `latest`, and verified end-to-end (§6).

## 3. Background — how a "compat hook" can kill `docker run`

On JetPack 7 the userspace NVIDIA libs live under `usr/lib/aarch64-linux-gnu/nvidia`
(`TARGET_USERSPACE_LIB_DIR`), and the container runtime stack is CDI-based:
`nvidia-container-toolkit 1.19.1` maintains `/var/run/cdi/nvidia.yaml`, whose every
device spec chains `createContainer` hooks (`nvidia-cdi-hook …`). runc executes these
**after rootfs setup, before the container starts** — a nonzero hook exit aborts the
whole create. The chain on this host (verbatim, spec lines 611–617):

```yaml
- hookName: createContainer
  path: /usr/bin/nvidia-cdi-hook
  args: [nvidia-cdi-hook, enable-cuda-compat,
         --host-cuda-version=13.2,
         --cuda-compat-container-root=/usr/local/cuda/compat_orin]
  env: [NVIDIA_CTK_DEBUG=false]
# ...followed by createContainer update-ldcache --folder /usr/lib/aarch64-linux-gnu/nvidia ...
```

`enable-cuda-compat` implements CUDA's forward-compat promise: *if the image ships
newer userspace CUDA than the host driver can run, prefer the image's
`compat*/libcuda`*. To decide, the hook **parses the ELF header/notes of the compat
libraries found in the image** (`cudacompat.getContainerForwardCompatDir` →
`useCompatLibraries` → `GetCUDACompatElfHeaderFromReader` → `trim`).

## 4. Failure analysis

Verbatim, from a live reproduction (see §1 check 3 against `edge-wheels:cu132`):

```text
docker: Error response from daemon: failed to create task for container:
  failed to create shim task: OCI runtime create failed: runc create failed:
  error during container init: error running createContainer hook #2:
  exit status 2, stdout: , stderr:
panic: runtime error: slice bounds out of range [:73] with capacity 71
goroutine 1 [running]:
nvidia-cdi-hook/cudacompat.trim(...)                     cuda-elf-header.go:100
cudacompat.GetCUDACompatElfHeaderFromReader(...)         cuda-elf-header.go:86
cudacompat.command.useCompatLibraries(...)               cudacompat.go:197
cudacompat.command.getContainerForwardCompatDir(...)
```

Failure chain: `docker run` → default runtime `nvidia` → runc `createContainer` →
hook #2 `enable-cuda-compat` → ELF-note parse of the image's forward-compat libs →
`trim()` slices `[:73]` of a 71-capacity buffer → **panic → hook exit 2 → container
create aborted (docker exit 125)**. The container's own code never executes, so
there are no container logs by construction.

**Trigger specificity (the part that cost the afternoon):** compat presence alone is
*not* the trigger.

- The two broken images inherit the **Jetson apt-packages layout**: `compat/` **and**
  `compat_orin/`, cu13.2-era `595.58.03` libs — and the generated hook's
  `--cuda-compat-container-root` aims exactly at `compat_orin`. Their ELF notes trip
  the `trim()` slice math (`[:73]` vs cap 71 — two bytes, every time).
- `mjolnir-vllm` ships **only `compat/`** (`580.95.05`, server-style base): the hook
  finds no `compat_orin`, exits cleanly — which is why the previously-working image
  kept working and the bug looked "random".

The precise ELF field that upsets `trim()` was **not bisected**; the panic is
deterministic per library vintage, and no public fix for 1.19.1 was located at
writing (re-check `nvidia-ctk` release notes >1.19.1). Empirics and repro above are
the contract; treat exact-note forensics as an upstream follow-up (§7.3).

**Why stripping is semantically safe.** Forward-compat libs are consulted only when
they are *newer* than the host driver API. Host: driver `595.78` / CUDA 13.2; images
baked: `595.58.03` / CUDA 13.2 — never newer, so the hook could only ever no-op;
`libcuda.so.1` is host-injected by the toolkit at run time regardless (the images'
ldconfig paths and `NVIDIA_VISIBLE_DEVICES` semantics are untouched). With the dirs
gone, the hook finds nothing, exits 0, and create succeeds — verified, not assumed.

## 5. The fix, precisely

`9ea3f0c` — one `RUN` per Dockerfile, placed in the **final runtime stage** *after*
all other layers that could reintroduce the dirs (llama: post-`ldconfig`; stt: post
pip/venv assembly), so a rebuild cache hit can't resurrect them:

```dockerfile
RUN rm -rf /usr/local/cuda/compat /usr/local/cuda/compat_orin
```

Alternatives considered and rejected:

- **Pin/downgrade ctk on the host** — host policy, fragile, helps one host.
- **Hand-edit `/var/run/cdi/nvidia.yaml`** — the spec is regenerated; edits don't survive.
- **Strip in the base images** — bases aren't ours; the consumer Dockerfile owns its final stage.
- **Keep dirs + `docker run --runtime runc`** — loses GPU injection; pointless for GPU images.

## 6. Verification record (2026-11-09)

- `docker run --rm --runtime nvidia --entrypoint true` → exit 0 on both fixed images;
  `edge-wheels:cu132` still 125 + panic (expected — unfixed by design, it's the repro).
- Full 5-service stack (`ai-stack` compose) up and healthy; every lane proven
  end-to-end the same day — decisions, chat, embeddings, STT (CUDA EP), TTS, and the
  tts→pcm→resample→stts round trip. Details: `ai-stack/MODELS.md` (verification
  stamp 2026-11-09).
- Post-fix images re-tagged `0.8.0` / `0.2.0` + `latest`; hub push pending (the push
  overwrites those tags — bump to `-r2` first if the pre-fix builds matter).

## 7. Required / recommended follow-ups

1. **Repo-wide pattern (required for every future GPU runtime image):** final stage
   ends with the strip `RUN`, plus the comment block from `9ea3f0c`. The
   `builders/parakeet-cpp` runtime image, when it exists, inherits this rule.
2. **CI create-smoke (recommended):** the §1 check 3 as a mandatory post-build step on
   ARM/Jetson runners — turns this class of failure into a red build.
3. **Upstream (recommended):** file the panic against `nvidia-container-toolkit`
   1.19.1 with the §4 trace + the edge-wheels repro; ask for the fixed release, then
   re-audit per §1 — but **keep the strip**: it is a no-op-cost invariant, not a
   workaround to unwind.
4. **Escape hatch** (only if an image ever needs real forward-compat, i.e. image
   CUDA *newer* than a host driver): systemd drop-in that post-edits the generated
   CDI spec to drop hook #2, or build the image from a CUDA ≤ host-driver base.
   Document the host's ctk version next to the deployment, since the bug is
   version-conditional.
5. **AGENTS.md note:** one line — "GPU runtime images: strip
   `/usr/local/cuda/compat*` in the final stage; see
   `docs/postmortem-cuda-compat-hook.md`".

## 8. Appendix — evidence log (2026-11-09, host `192.168.0.138`)

```text
/etc/nv_tegra_release : # R39 (release), REVISION: 2.1, GCID: 46758480 ... DATE: 2026-08-07
                        TARGET_USERSPACE_LIB_DIR=nvidia
driver                : Orin (nvgpu), 595.78      (nvidia-smi)
toolkit               : nvidia-container-toolkit 1.19.1-1 arm64
                        libnvidia-container1 1.19.1-1, cli commit 598a9cbb
CDI specs             : /var/run/cdi/nvidia.yaml (runtime), /etc/cdi/nvidia-pva.yaml
daemon.json           : "default-runtime": "nvidia"   <- hooks on every plain docker run
hook (spec L611)      : createContainer nvidia-cdi-hook enable-cuda-compat
                        --host-cuda-version=13.2
                        --cuda-compat-container-root=/usr/local/cuda/compat_orin
panic                 : slice bounds out of range [:73] with capacity 71
                        cudacompat.trim cuda-elf-header.go:100
                        ...GetCUDACompatElfHeaderFromReader :86
                        ...useCompatLibraries cudacompat.go:197
docker exit           : 125 (create phase; no container logs exist)
fix                   : 9ea3f0c "drop cuda compat dirs from llama-cpp and stt images"
                        packages/llm/llama-cpp/Dockerfile   | 8 ++++++++
                        packages/stt/nemotron-streaming-stt/Dockerfile | 7 ++++++++
mjolnir contrast      : compat/ only, libcuda.so.580.95.05, no compat_orin -> hook no-op, create OK
```
