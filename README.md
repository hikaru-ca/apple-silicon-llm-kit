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

| Key | Model | Weights | Notes |
|---|---|---|---|
| `e2b` | Gemma 4 E2B | ~2.5 GB | Smoke tests |
| `12b` | Gemma 4 12B | ~8 GB | Everyday default |
| `12b-qat` | Gemma 4 12B QAT | ~8 GB | Better quality at the same size |
| `26b-moe` | Gemma 4 26B A4B | ~16 GB | MoE, ~4B active per token |
| `31b` | Gemma 4 31B | ~19 GB | Flagship dense |
| `31b-qat` | Gemma 4 31B QAT | ~19 GB | Preferred 31B |

Sizes are weights only. KV cache grows on top and scales with context length,
so budget extra headroom before pushing context to six figures.

Any Hugging Face id or local path also works directly:

```bash
make gen MODEL=mlx-community/gemma-4-e2b-it-4bit
```

## Serving

```bash
make serve MODEL=12b PORT=8080
```

This exposes an OpenAI-compatible API at `http://127.0.0.1:8080/v1`, which any
OpenAI client can use by overriding its base URL. It binds to loopback; do not
expose it to a network without adding authentication.

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
