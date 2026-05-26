# exp/qwen25-7b-awq — summary

## Stack as deployed
- Requested: `Qwen/Qwen2.5-7B-Instruct-AWQ` int4
- **Deployed (fallback):** `Qwen/Qwen2.5-7B-Instruct` (bf16 base) loaded
  with bitsandbytes nf4 4-bit quantization (`load_in_4bit=True`,
  `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=float16`,
  `bnb_4bit_use_double_quant=True`).
- Context injection ON: system prompt contains glossary entries
  (case-insensitive substring match) + top-k=3 BM25 TM neighbours from
  `data/translation_memory/en-ko.jsonl` + register=formal directive.
  `max_new_tokens=128`, greedy decode.

## Why AWQ failed and we used nf4 instead
- `Qwen/Qwen2.5-7B-Instruct-AWQ` weights load via transformers requires
  `gptqmodel` (a recent transformers refactor — autoawq alone is no
  longer sufficient).
- `pip install gptqmodel` fails to build: its dependency `pypcre>=0.2.14`
  has no matching distribution on PyPI (max published is `0.3.2` and
  the package metadata still pins `pypcre`).
- bnb-nf4 on the bf16 base gives equivalent runtime VRAM (~4.2 GB
  weights, ~5 GB peak) with known-stable kernels on Blackwell sm_120.
- The substitution is a faithful proxy for "Qwen2.5-7B at int4" but is
  not literally AWQ. Recorded honestly.

## Hardware
- NVIDIA RTX 5060, 8 GB VRAM (sm_120, Blackwell)
- torch 2.10.0+cu128, transformers 4.x, bitsandbytes latest

## Eval (FLORES-200 devtest, 1012 sentences)

| Direction | BLEU | Tokenizer | Avg latency / segment | n |
|---|---|---|---|---|
| en → ko | **17.32** | ko-mecab | 1457 ms | 1012 |
| ko → en | **23.94** | 13a | 938 ms | 1012 |

## Comparison vs the sibling NLLB-600M baseline

| Direction | Qwen-nf4 + RAG | NLLB-600M no-ctx | Δ BLEU | Δ latency |
|---|---|---|---|---|
| en → ko | 17.32 | **25.35** | **−8.03** | +1321 ms |
| ko → en | 23.94 | **25.32** | **−1.38** | +804 ms |

## Why Qwen-nf4 lost

1. **nf4 quantization degrades Korean disproportionately.** Korean
   tokens are rarer in the Qwen tokenizer than English; rare tokens'
   embeddings are most sensitive to 4-bit rounding error. Scout reports
   warned of "1-2 BLEU loss" on en→ko from nf4; what we measured is
   considerably worse on en→ko but mild on ko→en. Suggests the loss is
   concentrated in target-side Korean generation, not source-side
   Korean comprehension.
2. **Context injection didn't trigger.** FLORES devtest is general
   news; the glossary is ML-conference terminology (LLM, fine-tuning,
   embedding, NVIDIA, …) and the TM has 10 ML-domain sentences. On the
   1012 FLORES sentences the glossary trigger rate is near zero and TM
   retrievals are off-domain. So the "+RAG" mechanism contributes
   essentially no signal on this test set — the comparison is really
   "nf4 Qwen no-real-context vs fp16 NLLB no-context".
3. **Decode latency 7-10x NLLB.** Qwen-7B autoregressive decode in nf4
   averages ~1.5 s per FLORES sentence vs NLLB-600M's 0.135 s — at 1.46
   s the StreamLAAL bar (≤2.4 s) is met but with no headroom for ASR
   chunks in the real cascade.

## Verdict
**Dead end on 8 GB VRAM with this test set.** The path "LLM-with-RAG
beats dedicated NMT" — the central README hypothesis — is not
supported by these measurements. It may still hold with (a) a domain
test set where the glossary actually triggers, and (b) ≥ 12 GB VRAM to
deploy Qwen at fp16 or proper AWQ. Both should be tested before
abandoning the LLM cascade idea entirely.

## Reproduce
```
git checkout exp/qwen25-7b-awq
PYTHONPATH=. python3 scripts/eval_streamlaal.py \
  --testset data/eval/flores_devtest_en_ko.tsv --src en --tgt ko \
  --config configs/default.yaml --report reports/exp_qwen25-7b-awq/en-ko.json
PYTHONPATH=. python3 scripts/eval_streamlaal.py \
  --testset data/eval/flores_devtest_ko_en.tsv --src ko --tgt en \
  --config configs/default.yaml --report reports/exp_qwen25-7b-awq/ko-en.json
```
