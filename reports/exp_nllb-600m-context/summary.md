# exp/nllb-600m-context — summary (NEGATIVE)

## Stack as deployed
- Translator: `facebook/nllb-200-distilled-600M`, fp16, cuda
- Context injection ON for en→ko: glossary entries (case-insensitive
  substring match) + top-k=3 BM25-ranked TM neighbours prepended to the
  source text as `GLOSS: …  TM: …  ||| <source>`
- For ko→en, context injection is skipped (TM/glossary are en-source-keyed)

## Hardware
- NVIDIA RTX 5060, 8 GB VRAM
- Other GPU consumer (Ollama qwen2.5:7b) stopped to free VRAM

## Eval (FLORES-200 devtest, 1012 sentences)

| Direction | BLEU | vs baseline (NLLB-600M no-context) | Latency |
|---|---|---|---|
| en → ko (with context) | **5.56** | **−19.79** (catastrophic regression) | 258 ms |
| ko → en (no context applied) | **25.32** | 0.00 | 131 ms |

## Why en→ko collapsed
The NMT encoder-decoder model treats the entire input string as text
to translate. When we prepend `TM: <english src> => <korean tgt>` pairs
in front of the actual source sentence, NLLB obediently translates
the TM examples themselves and often stops there (max_new_tokens budget
spent on TM pairs). Sample output:

> Input:  `TM: We used quantization … => 메모리 줄이기 위해 양자화를 사용했습니다. ||| <real source>`
> Output: `TM: 우리는 메모리 요구 사항을 줄이기 위해 양자화를 사용했습니다.`

The TM examples become the prediction. The actual source sentence
contributes only token-budget pressure.

The 5.56 BLEU floor is not zero because (a) trivial sentences pass
through, and (b) glossary terms occasionally show up in the output
correctly (`ko-mecab` is forgiving on partial matches).

## What this confirms
The approach-scout's skeptical prediction was correct: **prompt-injection
RAG is an LLM technique. NMT encoder-decoder models like NLLB cannot
distinguish "this is context, that is the source" without an
instruction-following prior.** The mechanism that works for Qwen2.5
(README's "+RAG → +4 BLEU") does not transfer to NLLB.

This is consistent with the scout's claim that NMT models gain
substantially less from in-context examples than instruction-tuned LLMs
(Intento 2024; IWSLT 2025 terminology submissions). What we found here
is stronger than "less gain" — we found a **negative gain** because the
NMT model literally translates the examples.

## What would actually help an NMT model
- A trained terminology / tag mechanism (NLLB has no such API)
- Constrained decoding (target-side trie) — testable on NLLB but
  requires a custom LogitsProcessor (not attempted this round)
- Source-side surface substitution (replace glossary source terms
  with their target translations before tokenization) — clean and
  cheap; would be the right round-2 if we keep NLLB as translator

## Verdict
**Dead end as built.** Confirms the architecture finding: prompt-
injection RAG requires an instruction-following translator. Cascade
with an NMT model needs a different context-injection mechanism (source-
substitution or constrained decoding), not prompt prefix.

## Reproduce
```
git checkout exp/nllb-600m-context
PYTHONPATH=. python3 scripts/eval_streamlaal.py \
  --testset data/eval/flores_devtest_en_ko.tsv --src en --tgt ko \
  --config configs/default.yaml --report reports/exp_nllb-600m-context/en-ko.json
PYTHONPATH=. python3 scripts/eval_streamlaal.py \
  --testset data/eval/flores_devtest_ko_en.tsv --src ko --tgt en \
  --config configs/default.yaml --report reports/exp_nllb-600m-context/ko-en.json
```
