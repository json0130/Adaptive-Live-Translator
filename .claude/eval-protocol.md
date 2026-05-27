# Fixed Eval Protocol — adaptive-live-translator

Every experiment, every round, is scored EXACTLY this way. Do not vary
any parameter below. If the protocol must change, it changes for ALL
experiments and the comparison table resets.

## Language pairs
- en -> ko  (primary)
- ko -> en  (secondary)

## Test sets (all frozen — no experiment trains or tunes on these)

### MT-only text reference (text-in, text-out)
- en->ko: `data/eval/flores_devtest_en_ko.tsv`
- ko->en: `data/eval/flores_devtest_ko_en.tsv`
  - FLORES+ devtest, 1012 sentences, CC-BY-SA.
  - Built by `scripts/build_flores_eval.py`.
  - Kept locked from round 1 as the apples-to-apples translator-only
    reference. Round 1 NLLB-600M fp16 GPU numbers on this set: 25.35
    en->ko / 25.32 ko->en (sacrebleu ko-mecab / 13a).

### Audio / end-to-end reference (round 2+)
- paired manifest: `data/eval/fleurs_pair_manifest.tsv`
  - 270 parallel utterances from Fleurs test (en_us + ko_kr), CC-BY-4.0.
  - Built by `scripts/build_fleurs_eval.py`.
  - Columns: `id, en_audio, ko_audio, en_text, ko_text`.
- Per-language audio manifests for ASR-only eval:
  - `data/eval/fleurs_en_us_test/manifest.tsv` (350 utterances)
  - `data/eval/fleurs_ko_kr_test/manifest.tsv` (270 utterances)
- Use this set for ANY experiment that touches ASR (and end-to-end
  voice-to-voice). MT-only experiments still use FLORES so their
  numbers stay comparable to round-1.

### Terminology / glossary slice
- `data/eval/ml_glossary_slice_en_ko.tsv` (146 sentences, en source)
- `data/eval/ml_glossary_slice_ko_en.tsv` (146 sentences, ko source — mirror)
  - Hand-curated ML/AI domain sentence pairs in which every row
    triggers at least one entry from `data/glossaries/ml-conference-en-ko.json`.
    Trigger rate is 100% by construction.
  - Used by terminology / constrained-decoding experiments. FLORES
    devtest stays as the parallel "general news" check for
    BLEU non-regression. Both must be reported.

## Metrics
- BLEU (sacrebleu).
  - en target: `--tokenize 13a`
  - ko target: `--tokenize ko-mecab` (LOCKED. Fall back to `char`
    only if mecab-ko-dic refuses to install AND record the fallback
    in `bleu_tokenize_used` so reviewer can verify.)
- ASR: WER (en, normalized via jiwer); WER + CER (ko, ko-mecab tokenized
  for word level).
- TTS: ASR-round-trip WER on synthesized audio against the original text.
- RTFx (decode wall-clock / audio duration). Real-time-capable = RTFx < 1.0.
- Peak resident memory (RSS, MB).
- Per-segment wall-clock latency.

In TEXT mode the latency number is translator-only.
In AUDIO mode it includes ASR + translator, which is the meaningful
end-to-end number for the live cascade.

## Command templates

### TEXT mode (MT-only)
```
PYTHONPATH=. python3 scripts/eval_streamlaal.py \
  --testset data/eval/flores_devtest_en_ko.tsv \
  --src en --tgt ko \
  --config configs/default.yaml \
  --report reports/<branch>/en-ko.json
```

### AUDIO mode (ASR + MT)
```
PYTHONPATH=. python3 scripts/eval_streamlaal.py \
  --audio-manifest data/eval/fleurs_pair_manifest.tsv \
  --src en --tgt ko \
  --config configs/default.yaml \
  --report reports/<branch>/audio_en-ko.json
```

### ASR-only
```
PYTHONPATH=. python3 scripts/eval_asr.py \
  --manifest data/eval/fleurs_en_us_test/manifest.tsv --lang en \
  --model Systran/faster-whisper-large-v3 --compute-type int8 \
  --report reports/<branch>/asr_en.json
```

### Cloud reference (NOT an experiment, ceiling row only)
```
PYTHONPATH=. python3 scripts/eval_cloud_baseline.py \
  --testset data/eval/flores_devtest_en_ko.tsv --src en --tgt ko \
  --report reports/cloud_baseline/en-ko.json
```

## Baseline to beat (round 2)
- MT-only on FLORES (text mode, GPU fp16 from round 1):
  - en->ko: **25.35 BLEU**
  - ko->en: **25.32 BLEU**
- Round 1 termination criterion stands: a CPU experiment WINS the
  translator axis if it lands within -0.55 BLEU on either direction
  (i.e. en->ko >= 24.80, ko->en >= 24.80) AND fits the laptop budget
  (see below).

## Laptop budget (round 2 target)
- Translator: <= 1.0 GB RSS, >= 20 tokens/sec decode on 8-core x86 CPU.
- ASR: <= 1.5 GB RSS, RTFx < 1.0.
- TTS: <= 1.0 GB RSS, RTFx < 1.0.
- Total peak memory: <= 4.0 GB RSS for the end-to-end pipeline.
- End-to-end audio-in to first-audio-out latency: <= 3.5 s.

## Rules
- Same test set, same parameters, every experiment.
- Report BOTH directions every round.
- No tuning on the test set. Ever.
- A correctly-measured negative result is a valid finding.
- Reviewer (NOT the experiment-runner) does any subjective spot-checks
  (e.g. Korean particle agreement on 30 sampled glossary-triggered
  sentences for constrained-decoding experiments).
