# voice-containers — build, verify, smoke and benchmark all projects.
# The project roster lives in scripts/projects.txt (one path per line) — the
# single source of truth consumed here and by scripts/sweep-*.sh + workflows.
PROJECTS := $(shell tr '\n' ' ' < scripts/projects.txt)

.PHONY: builders build-all verify-all smoke-all lint test bench-prepare bench-selfcheck bench-plot bench-prune

builders: ## shared artifact builders (cached; instant on replay)
	cd builders/parakeet-cpp && ./build.sh
	cd builders/wheels-cu132 && ./build.sh

build-all: ## docker build every engine image (tags narandill/<name>:<version>)
	@set -e; for p in $(PROJECTS); do echo "== build $$p"; (cd $$p && ./build.sh); done

verify-all: ## cuobjdump arch verification (no GPU touched)
	SMOKE=0 ./scripts/sweep-smoke.sh

smoke-all: ## GPU smoke suite (waits for free VRAM; never kills co-tenants)
	SMOKE=1 ./scripts/sweep-smoke.sh

lint: ## pre-commit over the whole repo
	pre-commit run --all-files

test: ## unit tests + 80% coverage gate — pure helpers, no GPU/docker (bench venv)
	bench/.venv/bin/python -m pytest

contract: ## voice-api v1 contract test (reference stub, no GPU/docker) — run before campaigns
	python3 tools/voice_api_contract_test.py

bench-prepare: ## download datasets + build frozen manifests (spec 3.4)
	bench/.venv/bin/python -m bench.cli prepare --dataset librispeech --split test-clean --lang en --n 500 --seed 42 --task stt
	bench/.venv/bin/python -m bench.cli prepare --dataset librispeech --split test-clean --lang en --n 500 --seed 42 --task tts

bench-selfcheck: ## device/CUDA/probe/engine availability report (spec 9.3)
	bench/.venv/bin/python -m bench.cli selfcheck

bench-plot: ## charts + matrix + README.md from benchmarks/history.jsonl (no inference)
	bench/.venv/bin/python -m bench.cli plot --readme

bench-prune: ## delete superseded benchmarks/raw dirs (latest per task; --yes to apply)
	bench/.venv/bin/python -m bench.cli prune
