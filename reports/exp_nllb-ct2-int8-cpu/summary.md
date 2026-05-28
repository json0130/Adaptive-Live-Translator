# exp/nllb-ct2-int8-cpu — summary (PASS — CPU baseline established)

## Stack as deployed
- Translator: `facebook/nllb-200-distilled-600M` converted to CTranslate2 INT8
  via `ct2-transformers-converter --quantization int8`.
- Decode: greedy (beam_size=1), max_decoding_length=256, intra_threads=8.
- No context injection (RAG/glossary/TM/rolling all disabled — translator-only baseline).
- ctranslate2 4.x, transformers 4.x, sentencepiece, sacrebleu+mecab-ko.

## Hardware
- Laptop CPU, no GPU used for this experiment.
- 8-core, AVX2.

## Eval (FLORES-200 devtest, 1012 sentences, locked protocol)

| Direction | BLEU | Tokenizer | Avg latency / segment | vs round-1 fp16 GPU |
|---|---|---|---|---|
| en → ko | **25.24** | ko-mecab | 689 ms | **−0.11** (25.35) |
| ko → en | **24.96** | 13a | 314 ms | **−0.36** (25.32) |

Both directions inside the ±0.55 BLEU window the protocol locked as
"int8 preserved quality". Falsification thresholds (any direction < 24.3,
or decode < 15 tok/s, or RSS > 1.2 GB) were not hit.

## What the Haiku runner got wrong (and how this got fixed)

The first attempt declared "DEAD END — INT8 broke Korean" based on
spurious decoding loops on long sentences (e.g. sentence 1 emitted
~300 repeated apostrophes; sentence 3 emitted `ĠSaraĠDanius…`).

That diagnosis was wrong. The real bug was in the runner's
`NllbCt2Translator._lazy_load`: they passed
`AutoTokenizer.from_pretrained(..., fix_mistral_regex=True)` to silence
a transformers warning. **That flag is for Mistral tokenizers, not NLLB.**
Applying it makes the NllbTokenizer emit GPT-2 / BPE `Ġ`-prefixed tokens
instead of the SentencePiece `▁`-prefixed tokens the model was trained
on, which silently corrupts long-sentence translations into degenerate
loops. The model and the int8 quantization were never the problem.

After removing the flag, FLORES devtest yields the numbers above —
matching round 1 to within 0.4 BLEU.

Lesson recorded for future spawn rounds:
**suppressing warnings without understanding their context can corrupt
data silently and the runner won't notice.** A Haiku-tier diagnosis
that contradicts a strong prior (CT2 int8 NMT being well-understood)
should be verified by the manager before being accepted.

## Verdict
**PASS — establishes the CPU baseline for round 2.** The translator
quality thesis ("int8 on a multilingual seq2seq is fundamentally
different from nf4 on a generative LM") is supported. CPU latency
budget is met with a 4-5x margin on both directions. Move on to A2
(ASR feasibility on real Fleurs audio).

## Reproduce
```
git checkout exp/nllb-ct2-int8-cpu
ct2-transformers-converter \
  --model facebook/nllb-200-distilled-600M \
  --output_dir models/nllb-600m-ct2-int8 \
  --quantization int8 \
  --copy_files tokenizer.json special_tokens_map.json tokenizer_config.json sentencepiece.bpe.model

PYTHONPATH=. python3 scripts/eval_streamlaal.py \
  --testset data/eval/flores_devtest_en_ko.tsv --src en --tgt ko \
  --config configs/default.yaml \
  --report reports/exp_nllb-ct2-int8-cpu/en-ko.json

PYTHONPATH=. python3 scripts/eval_streamlaal.py \
  --testset data/eval/flores_devtest_ko_en.tsv --src ko --tgt en \
  --config configs/default.yaml \
  --report reports/exp_nllb-ct2-int8-cpu/ko-en.json
```
