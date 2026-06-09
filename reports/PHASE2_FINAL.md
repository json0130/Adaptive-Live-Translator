# Phase 2 — GPU Validation: Final Report

**Status: CLOSED.** Phase-2 goal was to stand up a working voice-to-voice
en↔ko translator on GPU by composing the Phase-1-validated components, and
confirm the architecture works end-to-end once the CPU hardware ceilings no
longer apply. Outcome: **GPU composition, VRAM, and translation quality are
validated; the real-time latency goal is NOT met** — and the reason is
architectural, not hardware. Phase-3 terminology baseline is measured.

Hardware (corrected at preflight): **RTX 5060, 8 GB (Blackwell sm_120)** —
the plan's "RTX 4070 / 12 GB" was phantom hardware. All gates were
re-baselined to the 8 GB card before any experiment ran.

Eval protocol: the Phase-1 locked protocol (FLORES devtest, Fleurs paired
manifest N=270, ML glossary slice N=146, ko-mecab / 13a tokenization) —
preserved so numbers compare across phases.

---

## Results by step

### P2-0 — Environment rebuild (gated) — **PASS**
- torch `2.11.0+cpu` → `2.11.0+cu128`; CUDA runs on sm_120 (fp16 matmul
  verified). faster-whisper (CTranslate2) + NLLB fp16 (transformers) +
  MeloTTS all load and run on CUDA with no silent CPU fallback.
- Locked **ko-mecab BLEU tokenizer restored** (`src/utils/metrics.py` now
  uses `mecab_ko_dic.MECAB_ARGS`; MeloTTS's unidic install had hijacked
  MeCab's default dict). Behaviour-preserving.
- Env casualty recorded: MeloTTS hard-pins `transformers==4.27.4`
  (downgrade from 5.5.4); torch cu128 survived; only RAG-only deps
  (flagembedding/sentence-transformers) broke — unused in Phase 2.

### P2-1 — GPU pipeline composition — **PARTIAL (e2e FALSIFIED)**
Full stack: faster-whisper-large-v3 `int8_float16` + LocalAgreement-2 →
NLLB-600M transformers fp16 → MeloTTS-KR, all CUDA. N=270 each direction.

The runner reported "PROMISING — architecture validated"; **reviewer +
manager overturned the headline** (3rd headline correction in the project).

| Gate | en→ko | ko→en | Verdict |
|---|---|---|---|
| BLEU (≥24.03/19.39; regression ≤1.0) | 23.63 | 19.49 | PASS (no >1.0 regression; R4-1 was N=50 vs N=270 here) |
| Peak VRAM ≤7 GB (inference, nvidia-smi) | 5.02 GB | 4.85 GB | PASS |
| First-emission ≤1.5s | 1.853s | 1.732s | FAIL |
| **E2E ≤2.5s (honest)** | **4.115s** | **6.110s** | **FALSIFIED (>3s)** |

The runner's reported e2e (2.30s/1.88s) summed ASR **first-emission** + MT +
TTS, but MT runs on the **full** transcript (it cannot start until ASR
completes). Honest e2e (`asr_e2e_ms + mt_ms + tts_ms`, verified from
committed JSON) is 4.115s / 6.110s — both exceed the 2.5s gate and the 3s
falsify line (174/270 and 240/270 segments individually >3s).

**Finding:** the pipeline is a *batch pipeline in streaming clothes*. e2e is
dominated by full streaming-ASR decode (`asr_e2e_ms` 3.7s / 6.0s), inflated
because LA-2 re-decodes the whole growing buffer every window (buffer
trimming disabled, O(n²)), and MT is non-incremental. This reconfirms
Phase-1's bottleneck — **ASR gates latency, not the translator** — now on
GPU, but for a *different reason* than CPU (CPU was first-emission compute;
GPU is full-utterance streaming-decode cost + non-incremental MT).
- ko→en is **not voice-to-voice**: MeloTTS is Korean-only, so `tts_ms=0`
  for English output. The voice-to-voice claim holds for en→ko only.
- Real fix = **incremental MT** (translate committed ASR chunks) — an
  architecture change the plan explicitly scoped out. Buffer trimming would
  help but stays batch.

### P2-2 — Domain terminology baseline — **MEASURED (Phase-3 target)**
NLLB native term recall on the ML glossary slice (N=146 each direction, no
glossary enforcement — that axis is closed). Manager-verified from committed
JSON; manager linguistic spot-check found no grammar damage.

| Metric | en→ko | ko→en |
|---|---|---|
| BLEU (glossary slice) | 33.24 (ko-mecab) | 20.66 (13a) |
| Overall term recall | 36.4% (87/239) | 45.4% (114/251) |
| **Non-DNT term recall** | **20.7% (39/188)** | **31.5% (63/200)** |
| DNT pass-through | 94.1% (48/51) | 100% (51/51) |

(Corrected: runner's summary.md states ko→en non-DNT 32.1%; the per-term
table gives 63/200 = **31.5%** — runner subtracted 55 DNT triggers vs the
correct 51.)

**Miss analysis (the Phase-3 input):** the misses are **canonical-adherence
and wrong-term, not grammar breakage**:
- `LLM` 0% both directions — en→ko leaves it untranslated (`LLM를`); ko→en
  expands to "large language model" instead of the abbreviation (all
  valid_alternative). A style/adherence gap, not a correctness error.
- Genuine wrong-term: `fine-tuning→정렬` (alignment), `throughput→processing
  power`, `latency→delay`.
- Valid short forms scored as misses: `환각` for 환각 현상, `지연` for 지연 시간.
- `inference` highest natural recall (65–70%); DNT (RLHF/NVIDIA/HuggingFace)
  near-perfect; PyTorch occasionally transliterated (파이토치).

So a Phase-3 LoRA's job is **canonical-term preference**, not fixing broken
translation — feasible but needs enough per-term examples.

---

## Comparison table

| Step | en→ko BLEU | ko→en BLEU | StreamLAAL / latency | VRAM | Verdict |
|---|---|---|---|---|---|
| P2-1 (Fleurs N=270) | 23.63 | 19.49 | first-emit 1.85/1.73s; **honest e2e 4.12/6.11s** | 5.02 GB peak | composition+VRAM+quality OK; **e2e falsified** |
| P2-2 (glossary N=146) | 33.24 | 20.66 | n/a (text) | (text, translator only) | baseline measured |
| Phase-1 R4-1 (Fleurs N=50, CPU) | 24.03 | 19.39 | e2e ~5.9/4.3s | ~7.6 GB RAM | (reference baseline) |

Note: a true StreamLAAL (lag) metric was **never captured** in P2-1 — only
ASR first-emission and (mis-computed) e2e. For a simultaneous translator the
lag metric is the right yardstick; its absence is a methodology gap to fix
if the latency axis is reopened.

---

## Negative / honest results
- The "working real-time voice-to-voice system" goal was **not achieved**:
  e2e latency fails honestly on the cascaded-batch design.
- The runner's optimistic P2-1 headline was wrong; the corrected verdict
  came from path-consistent latency accounting (the R4-1 lesson, repeated).
- ko→en is text-out, not voice-out, with the current Korean-only TTS.
- P2-2 native recall (~21% / ~31% non-DNT) is low — confirming a real
  terminology gap — but the cause is canonical preference, not correctness,
  which changes what (if anything) LoRA should optimize.

## What is validated (carry forward)
- GPU composition of the Phase-1 stack works end-to-end with no OOM on 8 GB.
- int8_float16 ASR was the right call: 5.02 GB inference peak vs the 7.16 GB
  fp16 static-load estimate — comfortably under the 7 GB gate.
- Translation quality on GPU matches Phase-1 (no regression >1.0 BLEU).
- **8 GB has no headroom for co-resident LoRA training** (5 GB inference +
  ~2 GB free). Phase-3 LoRA training must run with the inference stack
  unloaded — a hard architectural constraint.

## Recommended next direction (Phase 3 decision inputs)
1. **Latency (if reopened):** the lever is **incremental/simultaneous MT**
   (translate committed ASR chunks), plus enabling ASR buffer trimming and a
   proper StreamLAAL metric. This is an architecture change, a deliberate
   scope decision — not another tuning run.
2. **Terminology (LoRA):** target to beat is non-DNT canonical recall
   **20.7% (en→ko) / 31.5% (ko→en)**. Since misses are adherence/style, a
   small LoRA or even a constrained-decode-free preference dataset may move
   it — but training must be a separate, inference-unloaded event on 8 GB.
3. Do **not** reopen decode-time glossary enforcement (closed in Phase 1).

## Termination rationale
P2-0 PASS, P2-1 e2e falsified (architectural finding, fix is out-of-scope
incremental MT), P2-2 baseline measured. The working-system goal is not met
on the cascaded-batch architecture, the cause is a clean publishable finding,
and the Phase-3 target is quantified. Per the budget rule, the phase closes
here without an additional latency-tuning run (option A deferred to a Phase-3
scope decision). Detailed code + per-segment JSON live on branches
`phase2/gpu-pipeline` (P2-1) and `phase2/domain-baseline` (P2-2).
