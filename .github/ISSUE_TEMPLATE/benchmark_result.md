---
name: Benchmark result
about: Report numbers from your box (no PR required)
labels: benchmark
---

**Cohort**
Task (stt/tts), device, power profile, and the frozen manifest (seed 42) —
cohorts are never ranked across a mix.

**Run**
`run_id`, the config used, and `n_ok` vs the manifest line count
(completeness — incomplete rows never replace full ones).

**Numbers**
Headline metrics; attach `results/<run_id>/` or the `benchmarks/raw/` export.

**Environment**
JetPack/L4T, `bench.cli selfcheck`, GPU co-tenants during the run.
