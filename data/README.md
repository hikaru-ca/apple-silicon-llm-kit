# data/

Everything in this directory is gitignored except this file.

Fine-tuning data lives here and must never be committed. Training sets tend to
be the most sensitive thing in a machine learning repository: they are the raw
material, not a derived artifact, and they are easy to add without thinking.

## Layout expected by `make lora-train`

```
data/lora/
  train.jsonl
  valid.jsonl
  test.jsonl      # optional
```

## Format

One JSON object per line. `mlx_lm.lora` accepts several schemas; the chat
format below is the most convenient for instruction tuning.

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

A minimal, entirely synthetic example lives at `examples/sample_train.jsonl`.
Copy it to `data/lora/train.jsonl` to check the pipeline end to end before
putting real data anywhere near this directory.

## Sizing

A few hundred examples is enough to shift tone and format. Changing what the
model actually knows takes thousands, and is usually better solved with
retrieval than with fine-tuning.
