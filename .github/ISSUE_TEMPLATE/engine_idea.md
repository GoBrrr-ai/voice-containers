---
name: Engine idea
about: Propose an engine / model for the voice stack
labels: engine-idea
---

**Upstream project / model**
URL, license, size — and whether an aarch64 path for sm_87 + sm_110 exists.

**Why it earns a slot**
What the current podium lacks: latency (TTFA/RTF), throughput, quality,
streaming behaviour, or VRAM budget.

**Probe results**
Anything measured already (CPU counts too) — attach numbers + how.

**Reminders**
A new package means `packages/<task>/<name>` plus registration in all four
lists (`Makefile`, both `scripts/sweep-*.sh`, workflows), a `bench/names.py`
entry, and a README re-render. Nothing is pushed — images stay local.
