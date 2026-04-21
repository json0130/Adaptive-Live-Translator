# Prompt Template Reference

Every chunk sent to the LLM uses the template below.
The system prompt is built once per chunk by `PromptBuilder.build_system_prompt()`.
The user turn is built by `PromptBuilder.build_user_turn()`.

## System prompt

```
You are a simultaneous interpreter translating EN → KO.
Translate accurately and naturally. Emit only the translation — no explanations.
Register: formal.
Domain/topic: {meeting_topic_summary}
Speaker: {speaker_name}

Glossary (MUST respect these translations):
  LLM → 대형 언어 모델
  fine-tuning → 파인튜닝
Do-Not-Translate (keep as-is): NVIDIA, PyTorch

Translation examples (for style/terminology reference):
  SRC: The model achieved state-of-the-art results.
  TGT: 모델이 최첨단 결과를 달성했습니다.

Previous segments (for discourse continuity):
  SRC: Good morning, everyone.
  TGT: 안녕하세요, 여러분.
  SRC: Today we discuss neural scaling laws.
  TGT: 오늘은 신경망 스케일링 법칙에 대해 논의합니다.
```

## User turn (per chunk)

```
Translate:
We trained a large language model on a hundred billion tokens.

Partial translation so far (continue from here):
우리는 천억
```

## Design decisions

### Why RAG instead of a full glossary in every prompt?
Full glossaries can have thousands of entries. Injecting all of them:
- Inflates token count → higher cost + latency
- Distracts the LLM (confirmed empirically by Intento 2024)

We inject only the entries whose source term matches the current chunk
via BM25 + dense retrieval, typically 0–10 entries per chunk.

### Why rolling context instead of the full transcript?
Full transcripts easily exceed 4096 tokens for a 10-minute talk (see
EuroLLM context study in CUNI IWSLT 2025). We keep the last 5 segments,
which is enough for pronoun resolution and topic continuity.

### Why "assistant priming" (tgt_so_far in user turn)?
Injecting the partial target into the user turn and asking the model to
"continue from here" — rather than starting fresh every time — is the
"response priming" trick from Koshkin et al. (EMNLP 2024). It prevents
the model from generating preambles, apologies, or duplicated text.

### Register adaptation
If the speaker profile says `register: informal`, the system prompt line
changes to `Register: informal.` This reliably shifts the LLM output
from formal honorifics (합쇼체) to informal speech (해요체) in Korean,
from vous to tu in French, etc. — no retraining required.
