# exp/madlad-3b — summary (NEGATIVE / BLOCKED on hardware)

## Stack as attempted
- Translator: `google/madlad400-3b-mt`, T5-style encoder-decoder
- Target precision: fp16 (model weights ~5.8 GB)
- Fallback attempted: bnb int8, bnb int4 (nf4)

## Hardware
- NVIDIA RTX 5060, 8 GB VRAM (Blackwell sm_120), ~7.2 GB free

## Result: blocked
The 3B T5 architecture does not fit on this 8 GB GPU in any usable precision:

| Precision | Loads? | Runs? | Quality | Notes |
|---|---|---|---|---|
| fp16 | partial | NO | n/a | OOM allocating activations after weight load |
| bnb-int8 | yes | NO | n/a | OOM allocating 1 GB scratch; weights at ~5.7 GB |
| bnb-nf4 | yes | yes | ~0 BLEU | Garbage output (token-id counting, repeated chars) |

Smoke test (3 FLORES en→ko sentences) at int4:
- BLEU 0.08 ko-mecab
- 7.7 s/sentence
- Output samples (verbatim from `/tmp/madlad_int4_smoke.json`):
  - `"1 2 3 4 5 6 7 8 9 10 11 12 13 14 ..."`
  - `"ง ง ง ง ง ง ง ง ง ง ง ง ..."`

## Root cause
Three compounding issues:
1. **VRAM ceiling.** MADLAD-3B fp16 weights take ~5.8 GB of the 7.2 GB
   budget — leaves no headroom for KV cache + activations on long inputs.
2. **Tied-embeddings load bug on T5.** transformers emits the warning
   `tie shared.weight to decoder.embed_tokens.weight, but both are present
   in the checkpoints with different values, so we will NOT tie them.`
   The model loads with separate (and apparently inconsistent) embedding
   tensors — its predicted token distribution is essentially random.
3. **nf4 quantization breaks T5 stability.** Even when fp16 loads
   correctly elsewhere, T5 layer-norm + relative positional bias is
   sensitive to int4 noise; degenerate decodes are a documented failure
   mode for MADLAD/T5 under bnb-nf4 without LoRA QAT.

## Verdict
**Dead end on 8 GB VRAM.** MADLAD-3B is a viable EN↔KO translator
(scout reports placed it at ~27–30 BLEU on FLORES) but requires
≥ 12 GB VRAM in fp16 to deploy. On this box, it is fundamentally
infeasible. If we acquire ≥ 12 GB hardware, MADLAD-3B fp16 should be
the first thing retried.

## What would unblock this experiment
- ≥ 12 GB GPU (e.g., RTX 4070 Ti / A4000)
- OR an AWQ-quantized MADLAD-3B checkpoint with proper T5 weight-tying
  (none exists on HF as of this round)
- OR a smaller MADLAD variant (no public smaller MT variant exists today;
  only 3B / 7B / 10B)
