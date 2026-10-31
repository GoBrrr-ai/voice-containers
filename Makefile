# edge-containers — build, verify, smoke and benchmark all projects.
PROJECTS := llama-cpp wyoming-faster-whisper wyoming-piper1-tts nemotron-unified-asr \
            parakeet-redux-stt phonon2-stt kokoro-tts phonon31m-tts paradee-tts

.PHONY: builders build-all verify-all smoke-all lint bench-prepare bench-selfcheck bench-plot

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

bench-prepare: ## download datasets + build frozen manifests (PRD 3.4)
	bench/.venv/bin/python -m bench.cli prepare --dataset librispeech --split test-clean --lang en --n 500 --seed 42 --task stt
	bench/.venv/bin/python -m bench.cli prepare --dataset librispeech --split test-clean --lang en --n 500 --seed 42 --task tts

bench-selfcheck: ## device/CUDA/probe/engine availability report (PRD 9.3)
	bench/.venv/bin/python -m bench.cli selfcheck

bench-plot: ## charts + matrix + README.md from benchmarks/history.jsonl (no inference)
	bench/.venv/bin/python -m bench.cli plot --readme
