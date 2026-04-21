# Evaluation Guide

## Metrics

### BLEU
Standard n-gram overlap against reference translations.
Computed with `sacrebleu` (tokenize=`13a`).
Higher is better. IWSLT 2025 En→De low-latency baseline: ~17 BLEU.

### StreamLAAL (Streaming Length-Adaptive Average Lagging)
The primary latency metric for simultaneous translation.
Defined in Papi et al. (2024).

```
StreamLAAL = (1/|T|) Σ_t  [ emit_time(t) - end_time(source_word_aligned_to_t) ]
```

Lower is better. IWSLT 2025 low-latency regime: ≤ 2.0 s.
High-latency regime: 4–5 s.

In `src/utils/metrics.py`, we use a simplified approximation:
per-segment delay = (tgt_emit_ms − src_end_ms).

## Running evaluation

```bash
# 1. Prepare test set (tab-separated: src \t ref)
# Example en-ko-dev.tsv:
#   The quick brown fox jumps.\t빠른 갈색 여우가 뛰어오릅니다.

# 2. Run
python scripts/eval_streamlaal.py \
    --testset data/eval/en-ko-dev.tsv \
    --src en --tgt ko \
    --config configs/default.yaml \
    --report reports/$(date +%Y%m%d)_en-ko.json
```

## Baseline targets

Evaluated on ACL 60/60 dev set (En→De, low-latency ≤2s):

| Configuration | BLEU ↑ | StreamLAAL ↓ |
|---|---|---|
| IWSLT 2025 organizer baseline | ~17 | 2.0 s |
| Whisper-large-v3 + Qwen2.5-7B, no context | ~22 | 2.0 s |
| + RAG glossary injection | ~24 | 2.1 s |
| + rolling context | ~26 | 2.2 s |
| + TM few-shot | ~27 | 2.2 s |

Numbers from: CMU IWSLT 2025 (arXiv:2506.13143), CUNI IWSLT 2025 (arXiv:2506.17077).

## Ablation checklist

When changing any context component, run the eval and compare:

- [ ] Glossary ON vs OFF
- [ ] TM retrieval ON vs OFF  
- [ ] Rolling context N=0, 3, 5, 10 segments
- [ ] RAG hybrid (BM25+dense) vs BM25-only
- [ ] LoRA adapter ON vs OFF for a specific speaker
- [ ] Chunk size: 1.0s vs 2.0s vs 4.0s (quality-latency tradeoff)
