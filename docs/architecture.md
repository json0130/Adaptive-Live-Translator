# Architecture

## Component diagram

```
Browser / Client App
        │ WebSocket (binary PCM + JSON)
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ FastAPI Server  (src/api/)                                       │
│                                                                  │
│  POST /sessions  →  SessionConfig  →  TranslationSession        │
│  WS   /ws/{id}   →  ws_handler     →  run_streaming_loop        │
└────────────────────────────┬────────────────────────────────────┘
                             │ asyncio tasks
        ┌────────────────────┴──────────────────────┐
        │                                           │
        ▼                                           ▼
┌───────────────┐                       ┌───────────────────────┐
│  audio_queue  │                       │   output_queue        │
│  (PCM chunks) │                       │   (TranslationEvent)  │
└───────┬───────┘                       └───────────────────────┘
        │                                           ▲
        ▼                                           │
┌──────────────────────────────────────────────────┴─────────────┐
│  run_streaming_loop  (src/pipeline/streaming_loop.py)           │
│                                                                 │
│   PCM  →  [ASR: WhisperStreaming + AlignAtt]                    │
│               │ ASRChunk.delta (source text)                    │
│               ▼                                                 │
│           [Context: PromptBuilder]                              │
│            ├── Glossary.hits_for(src_delta)                     │
│            ├── TranslationMemory.retrieve(src_delta)            │
│            ├── HybridRetriever (BM25 + bge-m3)                 │
│            └── rolling_context (last N segments)               │
│               │ system_prompt (str)                             │
│               ▼                                                 │
│           [Translator: QwenTranslator]                          │
│            ├── KV cache reuse (InfiniSST pattern)              │
│            └── LocalAgreement policy                           │
│               │ TranslationChunk.delta (target text)           │
│               ▼                                                 │
│           [TTS: CosyVoice 2]  (optional)                       │
│               │ np.ndarray PCM audio                           │
│               ▼                                                 │
│           TranslationEvent → output_queue                      │
└────────────────────────────────────────────────────────────────┘
```

## KV cache reuse (InfiniSST pattern)

The translator maintains a rolling KV cache across chunk turns. Each new
source chunk becomes a new "user" turn in the multi-turn conversation:

```
Turn 1: [SYS prompt] [USR: chunk_1] [AST: translation_1]
Turn 2:              [USR: chunk_2] [AST: translation_2]
Turn 3:              [USR: chunk_3] [AST: translation_3]
```

The KV cache for the system prompt and all previous turns is reused —
only the new `chunk_N` tokens are freshly encoded each time. This reduces
inference cost from O(N²) to O(N) as the session grows.

When context exceeds `max_context_tokens`, a sliding window drops the
oldest turn while keeping the system prompt permanently cached.

## Retrieval pipeline

For each ASR delta, three retrieval steps run in parallel before the LLM:

```
src_delta
    ├──► BM25 over glossary entries         → top-k_bm25
    ├──► bge-m3 dense cosine over glossary  → top-k_dense
    └──► Reciprocal Rank Fusion (k=60)      → top-k merged
```

The merged results are formatted into the system prompt as:
- Exact glossary term overrides (MUST-follow)
- Do-Not-Translate list (keep verbatim)
- TM few-shot examples (for style guidance)

## Latency budget

```
Source audio chunk     2000 ms  (chunk_seconds = 2.0)
VAD + ASR              ~200 ms  (faster-whisper)
RAG retrieval          ~5 ms    (BM25 + cached embeddings)
LLM decode             ~100 ms  (Qwen2.5-7B int4, A100)
TTS (optional)         ~150 ms  (CosyVoice 2 streaming)
─────────────────────────────────
Total overhead         ~455 ms
StreamLAAL estimate    ~2.2 s
```

## Data flow for speaker adaptation

```
Session start
    │
    ├── Look up SpeakerProfile (JSON)
    │      ├── register, topic, glossary overrides, DNT list
    │      └── lora_adapter name
    │
    ├── Load LoRA adapter (if enabled)
    │      └── HF PEFT hot-swap onto base Whisper
    │
    └── Inject profile into SessionContext
           └── PromptBuilder uses register + topic in every prompt
```
