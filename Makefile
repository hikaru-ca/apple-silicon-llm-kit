.DEFAULT_GOAL := help
SHELL := /bin/bash

# ---- Tunables (override on the command line, e.g. `make gen MODEL=31b`) ----
# 26b-moe, not 12b: the 12B Unified architecture needs mlx-vlm, which
# mlx_lm.server cannot load. See configs/models.toml.
MODEL       ?= 26b-moe
PROMPT      ?= Explain unified memory on Apple silicon in two sentences.
MAX_TOKENS  ?= 512
PORT        ?= 8080
UI_PORT     ?= 8443

# Thinking mode. Off by default: it costs several times the tokens and adds no
# quality on summarization, translation, extraction or classification. Turn it
# on for multi-step reasoning, and raise MAX_TOKENS when you do -- the reasoning
# is spent out of the same budget as the answer, so a short reply can still be
# truncated before it starts.
#
# Measured on Gemma 4 E2B, identical prompt, temp 0:
#   THINK=off   47 tokens
#   THINK=on   309 tokens   (6.6x, same answer)
#
# The two CLIs spell this differently, which is easy to trip over:
# mlx_lm.generate takes --chat-template-config, mlx_lm.server takes
# --chat-template-args.
THINK       ?= off
ifeq ($(THINK),on)
  THINK_JSON := {"enable_thinking":true}
else
  THINK_JSON := {"enable_thinking":false}
endif

# Training defaults sized for a 64 GB machine.
DATA        ?= data/lora
ADAPTER     ?= adapters/$(MODEL)
ITERS       ?= 600
BATCH_SIZE  ?= 4
NUM_LAYERS  ?= 16
LEARN_RATE  ?= 1e-5

RUN    := uv run
MODEL_ID = $(shell $(RUN) python scripts/resolve_model.py $(MODEL))

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  MODEL keys:"
	@$(RUN) python scripts/resolve_model.py --list 2>/dev/null || echo "    (run 'make setup' first)"

# ---------------------------------------------------------------- environment
.PHONY: setup
setup: ## Create the project-local venv and install pinned dependencies
	uv sync --extra dev
	@echo "OK. Environment is project-local at .venv (nothing installed globally)."

.PHONY: doctor
doctor: ## Print machine + MLX capability summary
	@$(RUN) python scripts/doctor.py

# ----------------------------------------------------------------- inference
.PHONY: pull
pull: ## Download a model into the local HF cache
	$(RUN) python scripts/pull.py $(MODEL)

.PHONY: gen
gen: ## One-shot generation: make gen MODEL=31b PROMPT="..." [THINK=on]
	$(RUN) mlx_lm.generate \
		--model $(MODEL_ID) \
		--prompt "$(PROMPT)" \
		--max-tokens $(MAX_TOKENS) \
		--chat-template-config '$(THINK_JSON)'

.PHONY: serve
serve: ## Start an OpenAI-compatible server on localhost
	@echo "Serving $(MODEL_ID) at http://127.0.0.1:$(PORT)/v1 (local only, THINK=$(THINK))"
	$(RUN) mlx_lm.server \
		--model $(MODEL_ID) \
		--port $(PORT) \
		--chat-template-args '$(THINK_JSON)'

.PHONY: up
up: ## Launch the stack: make up MODEL=31b-qat [THINK=on] [EXPOSE=lan] [AUTH=1] [UI=0]
	$(RUN) python scripts/launch.py \
		--model $(MODEL) \
		--think $(THINK) \
		--max-tokens $(MAX_TOKENS) \
		--expose $(or $(EXPOSE),loopback) \
		--port $(PORT) \
		--ui-port $(UI_PORT) \
		$(if $(AUTH),--auth,--no-auth) \
		$(if $(filter 0,$(UI)),--no-ui,--ui)

.PHONY: token
token: ## Generate a token into .env with 0600 permissions (FORCE=1 to rotate)
	@if [ -f .env ] && [ -z "$(FORCE)" ]; then \
		echo "error: .env already exists. Rotate with: make token FORCE=1"; exit 1; fi
	@# 0600 from the start: a shell redirect would create it 0644 first, leaving
	@# the token world-readable for the moment in between.
	@python3 -c 'import os, pathlib, secrets; \
p = pathlib.Path(".env"); \
fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600); \
os.write(fd, f"MLX_API_TOKEN={secrets.token_urlsafe(32)}\n".encode()); \
os.close(fd); os.chmod(p, 0o600)'
	@ls -l .env | awk '{print "Wrote .env (" $$1 ")"}'
	@echo 'Load it into this shell:  set -a; . ./.env; set +a'

.PHONY: down
down: ## Stop a running stack and wait for its ports to be released
	@pkill -f "scripts/launch.py|scripts/auth_proxy.py|mlx_lm.server" 2>/dev/null || true
	@# A terminated launcher gives its children up to 10s to exit, so the ports
	@# stay bound for a moment. Returning before they are free makes an
	@# immediately following `make up` fail on a stack that is already gone.
	@for i in $$(seq 1 20); do \
		if ! lsof -nP -iTCP:$(PORT),$(UI_PORT) -sTCP:LISTEN >/dev/null 2>&1; then \
			echo "Stopped."; exit 0; \
		fi; \
		sleep 0.5; \
	done; \
	echo "warning: ports still bound after 10s; check with: lsof -nP -iTCP:$(PORT) -sTCP:LISTEN"

.PHONY: restart
restart: down up ## Stop whatever is running, then launch: make restart MODEL=31b-qat

.PHONY: serve-auth
serve-auth: ## Bearer-token proxy in front of the server: make serve-auth LISTEN=0.0.0.0:8443
	@test -n "$$MLX_API_TOKEN" || { \
		echo "error: MLX_API_TOKEN is not set."; \
		echo "  cp .env.example .env, generate a token, then:  set -a; . ./.env; set +a"; \
		exit 1; }
	$(RUN) python scripts/auth_proxy.py \
		--listen $(or $(LISTEN),127.0.0.1:8443) \
		--upstream 127.0.0.1:$(PORT)

.PHONY: convert
convert: ## Quantize an arbitrary HF model: make convert HF=org/model QBITS=4
	@test -n "$(HF)" || { echo "error: set HF=<org/model>"; exit 1; }
	$(RUN) mlx_lm.convert \
		--hf-path $(HF) \
		--mlx-path models/$(notdir $(HF))-$(or $(QBITS),4)bit \
		-q --q-bits $(or $(QBITS),4)

# -------------------------------------------------------------- fine-tuning
.PHONY: lora-train
lora-train: ## LoRA/QLoRA fine-tune (QLoRA is automatic on a quantized base)
	$(RUN) mlx_lm.lora \
		--model $(MODEL_ID) \
		--train \
		--data $(DATA) \
		--adapter-path $(ADAPTER) \
		--iters $(ITERS) \
		--batch-size $(BATCH_SIZE) \
		--num-layers $(NUM_LAYERS) \
		--learning-rate $(LEARN_RATE) \
		--grad-checkpoint

.PHONY: lora-test
lora-test: ## Generate using a trained adapter
	$(RUN) mlx_lm.generate \
		--model $(MODEL_ID) \
		--adapter-path $(ADAPTER) \
		--prompt "$(PROMPT)" \
		--max-tokens $(MAX_TOKENS) \
		--chat-template-config '$(THINK_JSON)'

.PHONY: lora-fuse
lora-fuse: ## Fuse the adapter into the base weights as a standalone model
	$(RUN) mlx_lm.fuse \
		--model $(MODEL_ID) \
		--adapter-path $(ADAPTER) \
		--save-path models/$(MODEL)-fused

# ------------------------------------------------------------------- quality
.PHONY: lint
lint: ## Lint and format-check
	$(RUN) ruff check .
	$(RUN) ruff format --check .

.PHONY: fmt
fmt: ## Auto-format
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

.PHONY: leakcheck
leakcheck: ## Scan tracked files for private information before pushing
	$(RUN) python scripts/check_leak.py

.PHONY: verify-offline
verify-offline: ## Prove inference makes no outbound connections
	./scripts/verify_offline.sh $(MODEL)

.PHONY: check
check: lint leakcheck ## Everything CI runs
