# P3-1 LoRA Adaptation — Experiment Summary

Experiment: LoRA adapter on NLLB-200-distilled-600M for canonical terminology adherence.
P2-2 baseline: non-DNT recall en->ko 20.7% / ko->en 31.5%.

## Adapter Configuration

- Base model: facebook/nllb-200-distilled-600M
- LoRA rank: 16
- LoRA alpha: 32
- Target modules: ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']
- LoRA dropout: 0.05
- Trainable params: 8,650,752 (1.387%)
- Total params: 623,724,544
- LR: 0.0002, warmup_steps: 100
- Epochs run: 3, chosen_epoch: 3
- Effective batch: 16 (per_step=4 x grad_accum=4)
- Train losses per epoch: [1.4387, 0.1463, 0.0688]
- Dev losses per epoch: [0.2522, 0.095, 0.0783]
- Best dev loss: 0.0783
- Total optimizer steps: 357
- Training time: 168.2s
- Peak VRAM (train): 3465.5 MB = 3.38 GB
- ru_maxrss (train peak RSS): 4767112 KB

## Headline Numbers (N segments shown per row)

### Canonical Non-DNT Recall (ML Glossary Slice, N=146 per direction)

| Metric | en->ko | ko->en |
|--------|--------|--------|
| Base non-DNT recall | 20.2% (38/188) | 31.5% (63/200) |
| Adapter non-DNT recall | 94.7% (178/188) | 85.5% (171/200) |
| Delta (pp) | +74.5 pp | +54.0 pp |
| Base overall recall | 36.0% (86/239) | 45.4% (114/251) |
| Adapter overall recall | 95.8% (229/239) | 88.4% (222/251) |

**P2-2 baseline (reference):** non-DNT en->ko=20.7% (39/188), ko->en=31.5% (63/200)

### Pass Gate Assessment — Canonical Adherence
- en->ko non-DNT >= 35%: 94.7% => PASS
- ko->en non-DNT >= 45%: 85.5% => PASS

### Slice BLEU (ML Glossary Slice)

| Metric | en->ko (ko-mecab) | ko->en (13a) |
|--------|--------|--------|
| Base BLEU | 33.07 | 20.65 |
| Adapter BLEU | 47.46 | 26.48 |
| Regression | -14.39 | -5.82 |

### FLORES BLEU Regression (N=1012 per direction)

| Metric | en->ko (ko-mecab) | ko->en (13a) |
|--------|--------|--------|
| Base BLEU | 25.05 | 24.88 |
| Adapter BLEU | 21.36 | 21.48 |
| Regression | 3.7 | 3.4 |

### Pass Gate Assessment — FLORES Regression
- FLORES en->ko regression <= 0.5: 3.7 => FAIL
- FLORES ko->en regression <= 0.5: 3.4 => FAIL

## Overall Verdict: FAIL

Failed gates: FLORES en->ko regression 3.7 > 0.5 gate, FLORES ko->en regression 3.4 > 0.5 gate

## Per-Term Recall Delta (en->ko)

| Term (en) | Term (ko) | DNT | Base recall | Adapter recall | Delta (pp) |
|-----------|-----------|-----|-------------|----------------|------------|
| LLM | 대형 언어 모델 | no | 0.0% (0/39) | 92.3% (36/39) | +92.3 pp |
| fine-tuning | 파인튜닝 | no | 0.0% (0/15) | 100.0% (15/15) | +100.0 pp |
| inference | 추론 | no | 65.4% (17/26) | 76.9% (20/26) | +11.5 pp |
| embedding | 임베딩 | no | 0.0% (0/14) | 100.0% (14/14) | +100.0 pp |
| attention mechanism | 어텐션 메커니즘 | no | 0.0% (0/8) | 100.0% (8/8) | +100.0 pp |
| tokenizer | 토크나이저 | no | 0.0% (0/17) | 100.0% (17/17) | +100.0 pp |
| hallucination | 환각 현상 | no | 0.0% (0/12) | 91.7% (11/12) | +91.7 pp |
| RLHF | RLHF | YES | 100.0% (12/12) | 100.0% (12/12) | +0.0 pp |
| NVIDIA | NVIDIA | YES | 100.0% (15/15) | 100.0% (15/15) | +0.0 pp |
| PyTorch | PyTorch | YES | 72.7% (8/11) | 100.0% (11/11) | +27.3 pp |
| HuggingFace | HuggingFace | YES | 100.0% (13/13) | 100.0% (13/13) | +0.0 pp |
| benchmark | 벤치마크 | no | 38.5% (5/13) | 100.0% (13/13) | +61.5 pp |
| throughput | 처리량 | no | 47.4% (9/19) | 100.0% (19/19) | +52.6 pp |
| latency | 지연 시간 | no | 0.0% (0/14) | 100.0% (14/14) | +100.0 pp |
| quantization | 양자화 | no | 63.6% (7/11) | 100.0% (11/11) | +36.4 pp |

## Per-Term Recall Delta (ko->en)

| Term (en) | Term (ko) | DNT | Base recall | Adapter recall | Delta (pp) |
|-----------|-----------|-----|-------------|----------------|------------|
| LLM | 대형 언어 모델 | no | 0.0% (0/47) | 97.9% (46/47) | +97.9 pp |
| fine-tuning | 파인튜닝 | no | 6.2% (1/16) | 93.8% (15/16) | +87.5 pp |
| inference | 추론 | no | 70.4% (19/27) | 96.3% (26/27) | +25.9 pp |
| embedding | 임베딩 | no | 35.7% (5/14) | 7.1% (1/14) | -28.6 pp |
| attention mechanism | 어텐션 메커니즘 | no | 12.5% (1/8) | 100.0% (8/8) | +87.5 pp |
| tokenizer | 토크나이저 | no | 17.6% (3/17) | 76.5% (13/17) | +58.8 pp |
| hallucination | 환각 현상 | no | 100.0% (12/12) | 100.0% (12/12) | +0.0 pp |
| RLHF | RLHF | YES | 100.0% (12/12) | 100.0% (12/12) | +0.0 pp |
| NVIDIA | NVIDIA | YES | 100.0% (15/15) | 100.0% (15/15) | +0.0 pp |
| PyTorch | PyTorch | YES | 100.0% (11/11) | 100.0% (11/11) | +0.0 pp |
| HuggingFace | HuggingFace | YES | 100.0% (13/13) | 100.0% (13/13) | +0.0 pp |
| benchmark | 벤치마크 | no | 100.0% (15/15) | 100.0% (15/15) | +0.0 pp |
| throughput | 처리량 | no | 0.0% (0/19) | 94.7% (18/19) | +94.7 pp |
| latency | 지연 시간 | no | 8.3% (1/12) | 83.3% (10/12) | +75.0 pp |
| quantization | 양자화 | no | 46.2% (6/13) | 53.8% (7/13) | +7.7 pp |

## Hardware / Memory

- Peak VRAM (inference, torch.cuda.max_memory_allocated): 2526 MB = 2.47 GB
- ru_maxrss (true peak RSS, eval process): 5406388 KB = 5279.7 MB = 5.16 GB
- Eval time: 195.3s

## Reviewer Particle Check Candidates

en->ko particle concern segments (adapter): []
ko->en particle concern segments (adapter): []

## Reconciliation

All headline numbers are grep-able in:
  reports/p3-1_lora/result_slice_en_ko.json
  reports/p3-1_lora/result_slice_ko_en.json
  reports/p3-1_lora/result_flores_en_ko.json
  reports/p3-1_lora/result_flores_ko_en.json
  reports/p3-1_lora/training_metadata.json

Per-segment dumps for reviewer:
  reports/p3-1_lora/slice_outputs_en_ko.json
  reports/p3-1_lora/slice_outputs_ko_en.json

## Reproduction

```bash
# Train:
PYTHONPATH=$PWD ./.translator/bin/python scripts/p3_1_lora_train.py
# Eval:
PYTHONPATH=$PWD ./.translator/bin/python scripts/p3_1_lora_eval.py
```