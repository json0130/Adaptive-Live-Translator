# Fixed Eval Protocol — adaptive-live-translator

Every experiment, every round, is scored EXACTLY this way. Do not vary
any parameter below. If the protocol must change, it changes for ALL
experiments and the comparison table resets.

## Language pairs (Phase 1)
- en -> ko  (primary — has a baseline to beat)
- ko -> en  (secondary — first run establishes the baseline)
Other pairs are deferred to a later phase.

## Test set
- en->ko: data/translation_memory/en-ko held-out split, OR an agreed
  public dev set. Pick ONE and record the exact path here: <FILL IN>
- ko->en: the reverse split of the same set: <FILL IN>
- The test set is frozen for the whole research effort. No experiment
  may train or tune on it.

## Latency regime
- Low-latency regime (matches the README baseline table).
- Same chunk_seconds and streaming policy budget across all experiments
  unless the streaming policy IS the variable being tested.

## Metrics
- BLEU (sacrebleu, default tokenization; for ko use a consistent
  tokenizer — record which: <FILL IN>)
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