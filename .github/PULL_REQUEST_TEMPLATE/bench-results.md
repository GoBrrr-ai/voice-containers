## Run provenance

- run id(s) / date / host:
- config: `bench/configs/…yaml`; manifest (frozen, seed 42):
- power profile + GPU co-tenancy during the run:

## Completeness & artifacts

- [ ] `n_ok` == manifest line count (incomplete runs never supplant full ones)
- [ ] raw artifacts attached — `results/<run_id>/` or `benchmarks/raw/<run_id>/`
      (baseline-class runs publish raw, no audio)
- [ ] `benchmarks/history.jsonl` appended by `bench run`, not edited

## Rendered

- [ ] `make bench-plot` re-run; README/chart diff included; one cohort per
      comparison — never mixed
