# apple-silicon-llm-kit

A reproducible setup for running and fine-tuning open LLMs locally on Apple
silicon with [MLX](https://github.com/ml-explore/mlx).

The point of this repository is not the model — anyone can `pip install mlx-lm`.
The point is that the environment is **pinned, reproducible, and verifiably
offline**, so it can be trusted with data that must not leave the machine.

## Why MLX

MLX is Apple's own array framework. Its arrays live in unified memory, so CPU
and GPU operate on the same buffers with no host-to-device copy. Two practical
consequences:

- **Large models fit.** Memory is not capped by a discrete GPU's VRAM. A 64 GB
  Mac can hold a 31B model comfortably.
- **Fine-tuning works locally.** `mlx_lm.lora` trains LoRA, QLoRA, DoRA and full
  fine-tunes on the same machine. This is the capability llama.cpp does not
  have, and the main reason to prefer MLX over GGUF tooling.

The trade-offs are real and worth stating: MLX is Apple-only, it does not
enforce JSON schemas during sampling the way llama.cpp's grammars do, and its
throughput advantage narrows or reverses past roughly 40K tokens of context.

## Requirements

- Apple silicon (M1 or newer). There is no fallback — MLX does not run on x86.
- macOS with Metal support.
- [`uv`](https://docs.astral.sh/uv/) for dependency management.
- Memory determines which models are usable. Run `make doctor` to find out.

## Quickstart

```bash
make setup                  # create .venv and install pinned deps
make doctor                 # check the machine and print what fits
make pull  MODEL=12b        # download weights into the HF cache
make gen   MODEL=12b PROMPT="Explain LoRA in two sentences."
```

`make help` lists every target and every model key.

## Models

Keys are defined in [`configs/models.toml`](configs/models.toml) and resolve to
`mlx-community` repositories on the Hugging Face Hub.

| Key | Model | Weights | `make up` | Notes |
|---|---|---|---|---|
| `e2b` | Gemma 4 E2B | ~2.5 GB | yes | Smoke tests. Too small to trust on facts |
| `e4b` | Gemma 4 E4B 8-bit | ~9 GB | yes | Community build, not an official one |
| `12b` | Gemma 4 12B | ~8 GB | **no** | `gemma4_unified`; needs mlx-vlm |
| `12b-qat` | Gemma 4 12B QAT | ~8 GB | **no** | `gemma4_unified`; needs mlx-vlm |
| `26b-moe` | Gemma 4 26B A4B | ~16 GB | yes | **Default.** MoE, ~4B active per token |
| `31b` | Gemma 4 31B | ~19 GB | yes | Flagship dense |
| `31b-qat` | Gemma 4 31B QAT | ~19 GB | yes | Best quality here |

The two 12B entries are listed because they exist and are worth knowing about,
not because they work. `mlx-lm` 0.31.3 is the newest release there is, and it
predates the Gemma 4 12B Unified architecture (June 2026), so `mlx_lm.server`
cannot load it. `mlx-vlm` can, but is not wired into the server path yet.

This matters more than it looks, because `mlx_lm.server` loads weights lazily:
an unsupported model starts up, answers `/v1/models`, and only fails when
someone finally sends a message — as an HTTP 404 that explains nothing.
`make up` therefore checks `runtime` in the registry and refuses up front.

Sizes are weights only. KV cache grows on top and scales with context length,
so budget extra headroom before pushing context to six figures.

Any Hugging Face id or local path also works directly:

```bash
make gen MODEL=mlx-community/gemma-4-e2b-it-4bit
```

## Launching a stack

`make up` starts the pieces you ask for and stops them together on Ctrl-C.

```bash
make up MODEL=e2b                      # web UI + API on loopback, no token
make up MODEL=31b THINK=on             # bigger model, reasoning on
make up MODEL=12b UI=0                 # API only, for scripts
make up MODEL=12b EXPOSE=lan AUTH=1    # reachable from a phone, token required
```

| Option | Values | Default | Effect |
|---|---|---|---|
| `MODEL` | model key or HF id | `12b` | Which weights to load |
| `THINK` | `on` / `off` | `off` | Gemma thinking mode |
| `MAX_TOKENS` | integer | `512` | Default generation budget |
| `UI` | `0` to disable | on | Serve the web page |
| `AUTH` | `1` to enable | off | Require a bearer token |
| `EXPOSE` | `loopback` / `lan` | `loopback` | Who can reach it |

The model server always binds to loopback. Anything that needs to be reachable
from elsewhere goes through the front door instead, so **exposing the stack and
authenticating it are one decision**: `EXPOSE=lan` turns `AUTH` on by itself,
and the proxy refuses to start unauthenticated on a non-loopback address. That
combination is a mistake rather than a preference, so it is rejected instead of
warned about.

If the weights are missing, the launcher asks before downloading rather than
silently pulling several gigabytes.

### The web UI

A single self-contained page at `web/index.html`: streaming replies, model
picker, a per-message thinking toggle, and Gemma's reasoning folded into a
collapsible block. Conversation history lives in the browser tab and is gone
when you close it — nothing is written to disk.

It loads no fonts, scripts or styles from any CDN. A local model behind a page
that phones out for a stylesheet is not actually private, and the page has to
keep working with the network off.

It is deliberately small. For saved conversations, RAG, multiple users or mixing
in hosted providers' API keys, put Open WebUI in front of the same API instead.

## Serving

```bash
make serve MODEL=12b PORT=8080
```

This exposes an OpenAI-compatible API at `http://127.0.0.1:8080/v1`, which any
OpenAI client can use by overriding its base URL.

### Authentication

`mlx_lm.server` has no authentication of any kind — no tokens, no accounts, no
flags. Its only protection is the default bind address of `127.0.0.1`, which
keeps it unreachable from the network. That default is doing all of the work:
`--host 0.0.0.0` publishes an open model endpoint to everyone on the LAN.

So on loopback, nothing more is needed. To reach it from another device, put the
bundled proxy in front instead of moving the server off loopback:

```bash
cp .env.example .env                                  # then generate a token
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
set -a; . ./.env; set +a

make serve                                            # stays on 127.0.0.1:8080
make serve-auth LISTEN=0.0.0.0:8443                   # authenticated, in another shell
```

Clients then send `Authorization: Bearer <token>` (or `x-api-key`, for
Anthropic-style clients). Requests without a valid credential get a 401.
Streaming passes through chunk by chunk, so token-by-token output still works.

The proxy is standard library only — an authentication component is the last
place to add dependencies. Its limits are deliberate and worth stating plainly:
the token is sent in plaintext, so this is for a trusted LAN rather than the
internet; there is one shared token rather than per-user accounts; and there is
no rate limiting. For anything more exposed, terminate TLS in front of it.

## Thinking mode

Gemma 4 can emit its reasoning into a separate `thought` channel before
answering. Those reasoning tokens are generated like any others: they cost time,
and they are drawn from the same `--max-tokens` budget as the answer. A request
with too small a budget can spend all of it thinking and return no answer at
all.

`THINK` controls it, and defaults to `off`:

```bash
make gen MODEL=12b PROMPT="..."            # thinking off
make gen MODEL=12b PROMPT="..." THINK=on MAX_TOKENS=1500
```

Measured on Gemma 4 E2B, identical prompt, `temp 0`:

| | Tokens | Time | Answer |
|---|---|---|---|
| `THINK=off` | 47 | 0.4 s | correct |
| `THINK=on` | 309 | 1.8 s | correct |

Six and a half times the tokens for the same answer — because this prompt needs
no reasoning. Thinking pays for itself on multi-step problems (arithmetic,
constraint satisfaction, debugging, planning) and is close to pure overhead on
summarization, translation, extraction, classification and formatting.

For batch work the difference compounds: 6x tokens on ten thousand records is
the difference between an afternoon and a day. Leave it off unless a specific
task measurably improves with it.

One inconsistency worth knowing, since the flags are not named alike:
`mlx_lm.generate` takes `--chat-template-config`, while `mlx_lm.server` takes
`--chat-template-args`. Both accept `{"enable_thinking": true|false}`. The
`THINK` variable hides this.

## Fine-tuning

QLoRA is automatic: if the base model is quantized, adapters train on top of the
quantized weights with no extra flags.

```bash
cp examples/sample_train.jsonl data/lora/train.jsonl   # replace with real data
make lora-train MODEL=12b ITERS=600
make lora-test  MODEL=12b PROMPT="..."
make lora-fuse  MODEL=12b                              # standalone merged model
```

Defaults (`BATCH_SIZE=4`, `NUM_LAYERS=16`, `--grad-checkpoint`) are sized for a
64 GB machine. See [`data/README.md`](data/README.md) for the expected format.

Full fine-tuning is out of reach for 12B and above on a single machine —
optimizer state alone exceeds available memory. Use LoRA, or rent a GPU.

## Privacy

Local inference reads static weight files. Weights are never modified by
inference, prompts are not retained between runs, and nothing is transmitted.
Network access is needed only to download weights.

That claim is checkable rather than merely asserted:

```bash
make verify-offline MODEL=12b
```

This runs a generation while polling `lsof` for sockets owned by the process and
fails if any non-loopback connection appears. For a stronger guarantee, disable
Wi-Fi and run `make gen` — a cached model does not need the network at all.

## Keeping private data out of a public repository

Two independent layers:

1. **`.gitignore`** excludes `models/`, `adapters/`, `data/`, `.env` and local
   agent configuration.
2. **`make leakcheck`** scans tracked files for credentials, email addresses,
   private IPs, internal hostnames and absolute home paths. It runs in CI.

The scanner supports a second, private denylist at `.leakcheck-local.txt` for
concrete internal names. That file is gitignored on purpose: **a committed list
of secret words is itself a disclosure.** CI therefore runs the generic patterns
only, and the private list is enforced on the developer machine before pushing.

See [`.leakcheck-local.txt.example`](.leakcheck-local.txt.example).

## Dependency policy

- Every Python dependency is pinned with `==`. No ranges, no `latest`.
- Pins are stable releases that were at least three days old when added.
- `uv.lock` is committed; CI installs with `--locked` and fails on drift.
- GitHub Actions are pinned to full commit SHAs with the tag in a trailing
  comment, so Renovate can still propose upgrades.
- Everything installs into a project-local `.venv`. Nothing global.

`make check` enforces all of this locally; `scripts/check_pins.py` enforces it
in CI.

## License

MIT. See [LICENSE](LICENSE).

Model weights carry their own licenses. Gemma 4 is Apache 2.0; other models on
the Hub vary, and some entries in `configs/models.toml` are community builds
rather than official releases. Check before redistributing anything.
