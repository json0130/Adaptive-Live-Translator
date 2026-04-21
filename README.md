# Adaptive Live Translator

A context-aware, real-time speech translation system that adapts to **domain, terminology, and speaker** without retraining. Built around a streaming cascaded architecture: **Whisper large-v3 (ASR) → RAG context injection → Qwen2.5-7B-Instruct (translator) → CosyVoice 2 (TTS)**, with optional per-speaker LoRA adapters.

> **Status:** baseline scaffold. Components are stubbed with working interfaces — swap models/providers behind the same APIs.

---

## Why this stack

Three orthogonal dimensions of "context" are handled by three independent mechanisms:

| Dimension | Mechanism | Where it lives |
|---|---|---|
| Linguistic (prior sentences) | Rolling KV-cache + last-N segments in prompt | `src/pipeline/` |
| Domain / terminology | Hybrid RAG (BM25 + dense) over glossary + TM | `src/context/` |
| Speaker / user | Per-user JSON profile + optional LoRA adapter | `src/personalization/` |

Research grounding: IWSLT 2025 (CUNI, CMU, OSU), EMNLP 2024 "LLMs Are Zero-Shot Context-Aware Simultaneous Translators", InfiniSST (ACL Findings 2025).

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

## Quick start

### 1. Prerequisites

- Python 3.11+
- CUDA 12.1+ GPU with ≥16 GB VRAM (24 GB recommended for Qwen2.5-7B + Whisper large-v3 together)
- ffmpeg

### 2. Install

```bash
git clone <your-repo-url> adaptive-live-translator
cd adaptive-live-translator

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env               # edit with your HF token, etc.
```

### 3. Download models

```bash
python scripts/download_models.py
```

This pulls:
- `openai/whisper-large-v3`
- `Qwen/Qwen2.5-7B-Instruct`
- `FunAudioLLM/CosyVoice2-0.5B`
- `BAAI/bge-m3` (for dense retrieval)

### 4. Index your translation memory (optional)

```bash
python scripts/build_tm_index.py \
    --tm data/translation_memory/en-ko.jsonl \
    --out data/translation_memory/en-ko.index
```

### 5. Run the server

```bash
make run
# or: uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

### 6. Test with a WAV

```bash
python scripts/test_client.py \
    --audio samples/en_tech_talk.wav \
    --src en --tgt ko \
    --meeting-id acme-quarterly-2026
```

---

## Configuration

All runtime behavior is controlled by `configs/default.yaml`. Component-specific overrides live in `configs/asr.yaml`, `configs/translator.yaml`, `configs/context.yaml`. Example:

```yaml
asr:
  model: openai/whisper-large-v3
  chunk_seconds: 2.0
  policy: align_att
  align_att_frames: 20

translator:
  model: Qwen/Qwen2.5-7B-Instruct
  max_context_tokens: 4096
  kv_cache_reuse: true
  policy: local_agreement

context:
  rag:
    enabled: true
    hybrid: true          # BM25 + bge-m3
    top_k: 5
  glossary:
    injection_mode: prompt   # prompt | constrained_decoding
  translation_memory:
    top_k: 3
    min_similarity: 0.75

personalization:
  lora:
    enabled: false
    adapter_dir: data/lora_adapters
```

---

## Prompt template

This is what the translator LLM sees on every chunk (see `src/context/prompt_builder.py`):

```
[SYSTEM]
You are a simultaneous interpreter translating {src_lang} → {tgt_lang}.
Domain: {meeting_topic_summary}
Speaker: {speaker_name}, register: {formal|informal}

Glossary (must respect):
  {term_src} → {term_tgt}
  ...
Do-not-translate: [{brand}, {product}, ...]

[CONTEXT — previous segments]
SRC: {prev_n_source}
TGT: {prev_n_target}

[CURRENT PARTIAL]
SRC: {streaming_asr_output}
TGT: {output_so_far}
```

Rationale and ablations in `docs/prompt_template.md`.

---

## Evaluation

```bash
python scripts/eval_streamlaal.py \
    --testset data/eval/acl60_60_dev.tsv \
    --src en --tgt de \
    --report reports/2026-04-21.json
```

Reports BLEU + StreamLAAL (non-computationally-aware) on a held-out set. Baseline targets on ACL 60/60 dev, En→De, low-latency regime:

| System | BLEU ↑ | StreamLAAL ↓ |
|---|---|---|
| Organizers baseline (IWSLT 2025) | ~17 | 2.0 s |
| Ours (Whisper + Qwen2.5-7B, no context) | ~22 | 2.0 s |
| Ours (+ RAG + profile) | ~26 | 2.2 s |

Numbers to beat, not promises — rerun on your data.

---

## Roadmap

- [ ] Baseline end-to-end streaming loop (Whisper → Qwen → CosyVoice)
- [ ] Hybrid RAG with BM25 + bge-m3
- [ ] Per-user JSON profile + live glossary
- [ ] StreamLAAL + BLEU eval harness
- [ ] LoRA speaker adapter training script
- [ ] WebSocket client for browsers
- [ ] Diarization for multi-speaker meetings
- [ ] Voice cloning in TTS (preserve speaker identity across languages)
- [ ] On-device quantized variants (int4 Qwen, Whisper turbo)

---

## License

Apache 2.0 for project code. Model licenses vary — check each in `docs/model_licenses.md`.

---

## References

- Koshkin et al., *LLMs Are Zero-Shot Context-Aware Simultaneous Translators*, EMNLP 2024
- Papi et al., *AlignAtt*, Interspeech 2023
- Ouyang et al., *InfiniSST*, ACL Findings 2025
- CMU IWSLT 2025 submission (arXiv:2506.13143)
- CUNI IWSLT 2025 submission (arXiv:2506.17077)
- Hu et al., *LoRA*, ICLR 2022
