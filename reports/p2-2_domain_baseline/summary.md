# P2-2 Domain Baseline — Summary

Experiment: NLLB-200-distilled-600M fp16 on CUDA, native term recall measurement.
No glossary enforcement. Greedy decode (num_beams=1). TEXT ONLY.

## Headline Numbers

| Metric | en->ko | ko->en |
|--------|--------|--------|
| N segments | 146 | 146 |
| BLEU | 33.24 | 20.66 |
| tokenize_used | ko-mecab | 13a |
| total_triggered | 239 | 251 |
| total_correct | 87 | 114 |
| overall_recall | 0.364 | 0.4542 |

## Per-Term Recall (en->ko direction)

| Term (en) | Term (ko) | DNT | Triggered | Correct | Recall |
|-----------|-----------|-----|-----------|---------|--------|
| LLM | 대형 언어 모델 | False | 39 | 0 | 0.0000 |
| fine-tuning | 파인튜닝 | False | 15 | 0 | 0.0000 |
| inference | 추론 | False | 26 | 17 | 0.6538 |
| embedding | 임베딩 | False | 14 | 0 | 0.0000 |
| attention mechanism | 어텐션 메커니즘 | False | 8 | 0 | 0.0000 |
| tokenizer | 토크나이저 | False | 17 | 0 | 0.0000 |
| hallucination | 환각 현상 | False | 12 | 0 | 0.0000 |
| RLHF | RLHF | True | 12 | 12 | 1.0000 |
| NVIDIA | NVIDIA | True | 15 | 15 | 1.0000 |
| PyTorch | PyTorch | True | 11 | 8 | 0.7273 |
| HuggingFace | HuggingFace | True | 13 | 13 | 1.0000 |
| benchmark | 벤치마크 | False | 13 | 5 | 0.3846 |
| throughput | 처리량 | False | 19 | 10 | 0.5263 |
| latency | 지연 시간 | False | 14 | 0 | 0.0000 |
| quantization | 양자화 | False | 11 | 7 | 0.6364 |

Per-term sum check: triggered=239 (note: one segment may trigger multiple terms), correct=87
total_triggered=239, total_correct=87

## Per-Term Recall (ko->en direction)

| Term (en) | Term (ko) | DNT | Triggered | Correct | Recall |
|-----------|-----------|-----|-----------|---------|--------|
| LLM | 대형 언어 모델 | False | 47 | 0 | 0.0000 |
| fine-tuning | 파인튜닝 | False | 16 | 1 | 0.0625 |
| inference | 추론 | False | 27 | 19 | 0.7037 |
| embedding | 임베딩 | False | 14 | 5 | 0.3571 |
| attention mechanism | 어텐션 메커니즘 | False | 8 | 1 | 0.1250 |
| tokenizer | 토크나이저 | False | 17 | 3 | 0.1765 |
| hallucination | 환각 현상 | False | 12 | 12 | 1.0000 |
| RLHF | RLHF | True | 12 | 12 | 1.0000 |
| NVIDIA | NVIDIA | True | 15 | 15 | 1.0000 |
| PyTorch | PyTorch | True | 11 | 11 | 1.0000 |
| HuggingFace | HuggingFace | True | 13 | 13 | 1.0000 |
| benchmark | 벤치마크 | False | 15 | 15 | 1.0000 |
| throughput | 처리량 | False | 19 | 0 | 0.0000 |
| latency | 지연 시간 | False | 12 | 1 | 0.0833 |
| quantization | 양자화 | False | 13 | 6 | 0.4615 |

total_triggered=251, total_correct=114

## Miss Breakdown

### en->ko misses
- valid_alternative: 25
- untranslated: 39
- wrong_or_omitted: 88
- total misses: 152
  (reconcile: 25 + 39 + 88 = 152 vs total_misses=152)

### ko->en misses
- valid_alternative: 50
- untranslated: 0
- wrong_or_omitted: 87
- total misses: 137
  (reconcile: 50 + 0 + 87 = 137 vs total_misses=137)

## Reviewer Spot-Check Candidates

en->ko particle concern segments (seg_idx): []
ko->en particle concern segments (seg_idx): []

## Reconciliation

All headline numbers are grep-able in:
  reports/p2-2_domain_baseline/result_en_ko.json
  reports/p2-2_domain_baseline/result_ko_en.json

## Reproduction

```
PYTHONPATH=$PWD ./.translator/bin/python scripts/p2_2_domain_baseline.py
```