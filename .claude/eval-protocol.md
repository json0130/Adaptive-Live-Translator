# Fixed Eval Protocol — adaptive-live-translator

Every experiment, every round, is scored EXACTLY this way. Do not vary
any parameter below. If the protocol must change, it changes for ALL
experiments and the comparison table resets.

## Language pairs (Phase 1)
- en -> ko  (primary — has a baseline to beat)
- ko -> en  (secondary — first run establishes the baseline)
Other pairs are deferred to a later phase.

## Test set
- en->ko: data/eval/flores_devtest_en_ko.tsv
  (FLORES+ devtest, 1012 sentences, CC-BY-SA. Built by
  scripts/build_flores_eval.py from the openlanguagedata/flores_plus
  dataset on HF. Frozen at first build — never re-tuned, never re-split.)
- ko->en: data/eval/flores_devtest_ko_en.tsv (mirror of the same 1012
  sentences, columns swapped).
- The test set is frozen for the whole research effort. No experiment
  may train or tune on it.

## Latency regime
- Low-latency regime (matches the README baseline table).
- Same chunk_seconds and streaming policy budget across all experiments
  unless the streaming policy IS the variable being tested.

## Metrics
- BLEU (sacrebleu).
  - en target: --tokenize 13a (sacrebleu default)
  - ko target: --tokenize ko-mecab  (LOCKED. If the harness machine
    cannot install mecab-ko-dic, fall back to --tokenize char AND reset
    the comparison table — char and ko-mecab are not interchangeable.)
- StreamLAAL, non-computationally-aware
- Peak VRAM (GB)
- Wall-clock latency per segment (if available)

## Command template
scripts/eval_streamlaal.py \
  --testset <test-set-path> \
  --src en --tgt ko \
  --report reports/<branch-name>/en-ko.json
(and the reverse for ko->en)

## Baseline to beat (from README, en->ko, low-latency)
- No context:        ~22 BLEU / 2.0 s StreamLAAL
- + RAG + profile:   ~26 BLEU / 2.2 s StreamLAAL
- ko->en: no baseline yet — round 1 establishes it.

## Success bar
An approach wins if it beats the en->ko baseline by >= +2 BLEU WITHOUT
StreamLAAL regressing past ~2.4 s, and reviewer confirms the result is
sound and reproducible.

## Rules
- Same test set, same parameters, every experiment.
- Report BOTH directions every round.
- No tuning on the test set. Ever.
- A correctly-measured negative result is a valid finding.