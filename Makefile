# edge-containers — build, verify, smoke and benchmark all projects.
PROJECTS := llama-cpp wyoming-faster-whisper wyoming-piper1-tts nemotron-unified-asr \
            parakeet-redux-stt phonon2-stt kokoro-tts phonon31m-tts paradee-tts

.PHONY: builders build-all verify-all smoke-all lint bench-prepare bench-selfcheck

builders: ## shared artifact builders (cached; instant on replay)
	cd builders/parakeet-cpp && ./build.sh
	cd builders/wheels-cu132 && ./build.sh

build-all: ## docker build every engine image (tags futureproofhomes/<name>)
	@set -e; for p in $(PROJECTS); do echo "== build $$p"; (cd $$p && ./build.sh); done

verify-all: ## cuobjdump arch verification (CUDA images)
	@set -e; for p in $(PROJECTS); do \
	  if [ -x $$p/test/verify_arch.sh ]; then echo "== verify $$p"; \
	    $$p/test/verify_arch.sh futureproofhomes/$$p-thor:0.1.0 || \
	    $$p/test/verify_arch.sh futureproofhomes/$$p-thor:0.8.0; \
	  else echo "== verify $$p (skipped: CPU-only)"; fi; done

smoke-all: ## GPU smoke suite (waits for free VRAM; never kills co-tenants)
	@set -e; for p in $(PROJECTS); do echo "== smoke $$p"; \
	  (cd $$p && ./test/smoke_*.sh futureproofhomes/$$p-thor:0.1.0); done

lint: ## pre-commit over the whole repo
	pre-commit run --all-files

bench-prepare: ## download datasets + build frozen manifests (PRD 3.4)
	python -m bench.cli prepare --dataset librispeech --split test-clean --lang en --n 500 --seed 42
	python -m bench.cli prepare --dataset libritts_r --split test.clean --lang en --n 500 --seed 42

bench-selfcheck: ## device/CUDA/probe/engine availability report (PRD 9.3)
	python -m bench.cli selfcheck
