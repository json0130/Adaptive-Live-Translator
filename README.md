# Adaptive Live Translator — Postmortem

A context-aware, real-time speech translation system (ASR → context → translator
→ TTS) for Korean ↔ English, targeting a laptop-CPU deployment. The goal was a
voice-to-voice cascade that adapts to domain/terminology/speaker **and** runs
under hard edge gates: **first-audio latency < 3.5 s** and **co-resident RAM
< 4 GB** on an 8-core CPU with ~7 GB free RAM, no GPU.

> **Status: CLOSED research artifact — not a deployable product.** Five rounds
> of experiments converged on an architectural-limit finding: this CPU/7 GB-laptop
> class **cannot** meet the 3.5 s latency and 4 GB RAM gates simultaneously for
> cascaded ko↔en voice translation. The bottlenecks are Whisper first-emission and
> MeloTTS per-utterance memory growth — **not** the translator, which was the axis
> four rounds optimized. Details below. This README documents what was learned, not
> a roadmap.

---

## The research arc (five rounds)

Each round below links to its full report under [reports/](reports/).

**Round 1 — GPU baseline (RTX 5060, 8 GB VRAM). Verdict: SATURATION.**
On FLORES-200 devtest, a dedicated NMT — `NLLB-200-distilled-600M` fp16 — set the
real baseline at **25.35 / 25.32 BLEU** (en→ko / ko→en) at ~135 ms/seg and ~3 GB,
beating the README's old placeholder. Qwen2.5-7B (nf4, forced by 8 GB) **lost**
(17.32 en→ko), prompt-injection RAG on NMT was **catastrophic** (25.35 → 5.56),
and MADLAD-3B OOM'd. Conclusion: the cascade architecture wasn't the bottleneck —
VRAM and the LLM-quantization penalty on Korean were. See
[ROUND1_REPORT.md](reports/ROUND1_REPORT.md).

**Round 2 — CPU pivot (laptop, no GPU). Verdict: quality survives, latency/TTS don't.**
`NLLB-600M` converted to **CT2 int8** held quality on CPU (**25.24 / 24.96 BLEU**,
~2 GB) — the int8-Korean thesis held. `faster-whisper-medium` int8 was the right
ASR for CPU. But the batch voice→voice pipeline missed the 3.5 s gate (4.2–5.0 s,
because batch ASR waits for the whole utterance), and **Korean TTS was the single
weakest link**: espeak-ng was robotic enough that round-trip ASR couldn't recover
the words (71.66 % WER). See [ROUND2_REPORT.md](reports/ROUND2_REPORT.md).

**Round 3 — clear the gates. Verdict: ASR + TTS fixed; glossary failed.**
Streaming ASR via **LocalAgreement-2** dropped first-audio latency to **2.95 s (en,
PASS)** / 3.89 s (ko, near-miss). **MeloTTS-KR** (MIT-licensed, from source) took
Korean TTS from unintelligible to **13.64 % round-trip WER** — the strongest
result of the project. The glossary attempt (token-level logit-bias on NLLB)
**FAILED**: a reviewer's Korean-particle spot-check found it breaks grammar (case-
particle swaps) on multi-term sentences, invisible to corpus BLEU. See
[ROUND3_REPORT.md](reports/ROUND3_REPORT.md).

**Round 4 — compose end-to-end. Verdict: the bottleneck is NOT the translator.**
The full pipeline composed, but quality was **flat** (≈ −0.5 BLEU vs the R2 e2e run
on the same manifest — NLLB passes audio-chain quality through but adds none),
latency **failed** (the small-model final ASR pass pushed the real quality-path
latency to ~10 s, ~3× the gate), and RAM **busted** the 4 GB budget by 89 %
(7.57 GB peak, growing from 3.8 GB across the run — a MeloTTS allocation leak). A
third glossary mechanism (phrase-constrained decoding) also failed. The verified
finding: **latency is gated by ASR, RAM by TTS; the translator/context axis is
saturated.** (A reviewer corrected the runner's inflated headline here — the second
consecutive round review overturned a result.) See
[ROUND4_FINAL.md](reports/ROUND4_FINAL.md).

**Round 5 — attack the actual bottlenecks. Verdict: BOTH FAIL → ARCHITECTURAL LIMIT.**
Two experiments, one per gate. **R5-1 single-pass ASR** (remove the final re-decode
so the latency path == the quality path) reached a best of **3.87 s (ko→en)** and
5.47 s (en→ko) — **no variant meets 3.5 s** in either direction; Whisper CPU first-
emission alone is ~3.3 s (base) / ~4.9 s (small). **R5-2 TTS-RAM slim** (disable
MeloTTS's BERT normalizer) saved ~0.84 GB *at load* but the per-utterance allocation
growth still climbed to **~6.6 GB by segment 30** — **busts 4 GB**. Terminated on
budget. See [ROUND5_FINAL.md](reports/ROUND5_FINAL.md).

---

## The finding

On this **CPU / ~7 GB-laptop class with no GPU**, a cascaded ko↔en voice translator
**cannot meet the 3.5 s latency and 4 GB co-resident-RAM gates simultaneously**. The
two gates are blocked by distinct, hardware-rooted bottlenecks — and neither is the
translator, which is where four of five rounds spent their effort:

1. **Latency floor — ASR.** CPU Whisper first-emission is ~3.3 s on its own
   (`faster-whisper-base`), already at/over the gate before TTS runs. Removing the
   small-model final pass (R5-1) was the right move but is not enough.
2. **Memory ceiling — TTS.** MeloTTS-KR synthesis accumulates to ~6.6 GB over a run.
   Stripping its `kykim/bert-kor-base` normalizer saves only ~0.84 GB at load and
   nothing about the runtime growth.

To proceed you would change the **hardware/engine class** (a GPU/NPU to collapse
Whisper first-emission and TTS synthesis; or a fundamentally lighter Korean TTS
engine) — not the translator. These are architecture decisions, not another
experiment in this loop.

### What works

- **`NLLB-200-distilled-600M`, CT2 int8 on CPU** — preserves translation quality
  (≤ 0.4 BLEU vs fp16 GPU; ~25 BLEU both directions), ~2 GB. The translator is solid.
- **MeloTTS-KR** — intelligible Korean TTS (13.64 % round-trip WER), **MIT-licensed**,
  installed from source.
- **Streaming ASR with LocalAgreement-2** (`faster-whisper-small`/`-base` int8) —
  clears the latency gate for English (2.95 s) and gets Korean close.

### What doesn't

- **Decode-time glossary/terminology enforcement** — failed across **three**
  mechanisms on NLLB-600M (prompt-prefix in R1, token logit-bias in R3,
  phrase-constrained decoding in R4). The axis is closed: NMT terminology adherence
  at decode time was not solvable here without breaking Korean grammar or beam
  stability.
- **The CPU cascade against its deployment gates** — see the finding above.

### The one live lead (if anyone resumes)

The **MeloTTS per-utterance allocation growth** is a concrete, isolatable bug and
the single most promising CPU-side lead. Load-time RAM after disabling the BERT
normalizer is already **2.93 GB — under budget**; it is the runtime growth to
~6.6 GB that busts the 4 GB gate. If that leak can be bounded, R5-2's verdict could
move. **This diagnosis has not been started** and is a separate future decision, not
part of this close-out. All other axes (translator, glossary, context) are closed.

---

## Architecture

```
 ┌────────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐
 │  Mic /     │───▶│  Streaming  │───▶│  Context     │───▶│  Streaming   │───▶│  Streaming │
 │  RTP       │    │  ASR        │    │  Builder     │    │  Translator  │    │  TTS       │
 │            │    │  (Whisper + │    │  (RAG +      │    │  (Qwen2.5 +  │    │  (CosyVoice│
 │            │    │   AlignAtt) │    │   profile)   │    │   InfiniSST  │    │   2)       │
 │            │    │             │    │              │    │   pattern)   │    │            │
 └────────────┘    └──────┬──────┘    └──────┬───────┘    └──────┬───────┘    └─────┬──────┘
                          │                  │                   │                  │
                          ▼                  ▼                   ▼                  ▼
                   ┌─────────────────────────────────────────────────────────┐
                   │  State: rolling transcript, KV cache, speaker profile,  │
                   │         glossary hits, TM retrievals                    │
                   └─────────────────────────────────────────────────────────┘
```

> Note: the diagram shows the original scaffold's intended components (Whisper /
> Qwen / CosyVoice). The configuration that actually survived experiments is
> `faster-whisper` (streaming, LocalAgreement-2) → `NLLB-200-distilled-600M`
> CT2 int8 → MeloTTS-KR, on CPU. See the finding above.

---

## Folder structure

```
adaptive-live-translator/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
├── Makefile
│
├── configs/                        # YAML configs per component
│   ├── default.yaml
│   ├── asr.yaml
│   ├── translator.yaml
│   └── context.yaml
│
├── src/
│   ├── asr/                        # Streaming ASR (Whisper + AlignAtt)
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract ASRStreamer
│   │   ├── whisper_streaming.py    # Whisper large-v3 wrapper
│   │   └── align_att.py            # AlignAtt simultaneous policy
│   │
│   ├── translator/                 # LLM-based streaming translator
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract Translator
│   │   ├── qwen_translator.py      # Qwen2.5-7B with KV-cache reuse
│   │   └── policies.py             # LocalAgreement / wait-k
│   │
│   ├── context/                    # The "adaptive" part
│   │   ├── __init__.py
│   │   ├── rag.py                  # Hybrid BM25 + dense retriever
│   │   ├── glossary.py             # Glossary + DNT list handling
│   │   ├── translation_memory.py   # TM lookup for few-shot
│   │   └── prompt_builder.py       # Assembles final LLM prompt
│   │
│   ├── personalization/            # Per-speaker adaptation
│   │   ├── __init__.py
│   │   ├── speaker_profile.py      # JSON profile CRUD
│   │   ├── lora_loader.py          # Hot-swap LoRA adapters
│   │   └── speaker_id.py           # Optional speaker diarization
│   │
│   ├── tts/                        # Streaming TTS
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── cosyvoice.py            # CosyVoice 2 wrapper
│   │
│   ├── pipeline/                   # Orchestration
│   │   ├── __init__.py
│   │   ├── session.py              # A translation session (per speaker/meeting)
│   │   └── streaming_loop.py       # The main async read/write loop
│   │
│   ├── api/                        # HTTP / WebSocket entrypoints
│   │   ├── __init__.py
│   │   ├── server.py               # FastAPI app
│   │   └── ws_handler.py           # WebSocket audio streaming
│   │
│   └── utils/
│       ├── __init__.py
│       ├── audio.py
│       ├── logging.py
│       └── metrics.py              # BLEU / StreamLAAL latency tracking
│
├── data/
│   ├── glossaries/                 # One JSON per domain/meeting
│   ├── translation_memory/         # Parallel TM per language pair
│   ├── speaker_profiles/           # Per-user JSON profiles
│   └── lora_adapters/              # Trained speaker adapters (*.safetensors)
│
├── scripts/
│   ├── download_models.py          # Pulls Whisper, Qwen, CosyVoice from HF
│   ├── build_tm_index.py           # BM25 + FAISS index over a TM
│   ├── train_speaker_lora.py       # Fine-tune a LoRA adapter on speaker data
│   └── eval_streamlaal.py          # Offline latency/quality eval
│
├── tests/
│   ├── test_asr.py
│   ├── test_translator.py
│   ├── test_rag.py
│   └── test_pipeline.py
│
├── notebooks/
│   └── 01_smoke_test.ipynb
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yaml
│
└── docs/
    ├── architecture.md
    ├── prompt_template.md
    └── evaluation.md
```

---

## References

- Koshkin et al., *LLMs Are Zero-Shot Context-Aware Simultaneous Translators*, EMNLP 2024
- Papi et al., *AlignAtt*, Interspeech 2023
- Ouyang et al., *InfiniSST*, ACL Findings 2025
- CMU IWSLT 2025 submission (arXiv:2506.13143)
- CUNI IWSLT 2025 submission (arXiv:2506.17077)
- Hu et al., *LoRA*, ICLR 2022
