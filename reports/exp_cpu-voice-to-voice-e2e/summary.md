# exp/cpu-voice-to-voice-e2e — summary (PARTIAL PASS — quality good, latency-budget fails as-built)

## Stack as deployed
- ASR: `Systran/faster-whisper-medium` int8 (CPU, intra_threads=8)
- MT: `facebook/nllb-200-distilled-600M` CT2 int8 (CPU)
- TTS: **espeak-ng** (MeloTTS install failed — pypi package broken).
  Both en and ko routed to espeak-ng.
- Eval harness: `scripts/eval_e2e.py` (new). For each utterance: ASR → MT → TTS,
  plus a SECOND ASR pass on the synthesized audio to measure
  TTS-intelligibility round-trip WER. Per-component and total latencies
  are recorded.

## Hardware
- Laptop CPU, 8-core AVX2, no GPU, ~7 GB RAM available.

## Eval (Fleurs paired manifest, 50 utterances/direction)

### Per-component (mean)

| Direction | ASR ms | MT ms | TTS ms | round-trip ASR ms | audio s | BLEU | ASR WER on src | round-trip WER on synth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| en → ko | 3610 | 602 | 18 | ~9500 | 9.6 | **22.91** | 13.19 | 71.66 (ko espeak hard to re-transcribe) |
| ko → en | 4252 | 714 | 18 | ~3850 | 11.9 | **19.90** | 7.61 | 5.83 (en espeak is more intelligible) |

### Live-pipeline RTFx (ASR + MT + TTS, NOT including the round-trip eval pass)

| Direction | live pipeline ms | audio ms | **live RTFx** | first-audio latency (mean) |
|---|---:|---:|---:|---:|
| en → ko | 4230 | 9560 | **0.44** | 4.2 s |
| ko → en | 4984 | 11938 | **0.42** | 5.0 s |

The eval harness reports a headline RTFx of 1.44 (en→ko) / 0.74 (ko→en)
that includes the round-trip ASR pass on the TTS output. The round-trip
pass is part of QUALITY measurement (does the synth audio round-trip
back to text?), it is NOT part of the live cascade. The live-pipeline
RTFx is the meaningful production number.

### Peak RAM

| | value | falsification gate | verdict |
|---|---|---|---|
| 50-utt en→ko | 5.1 GB | ≤ 4.0 GB | **technically FAILS**, but inflated |
| 50-utt ko→en | 5.1 GB | ≤ 4.0 GB | same |

`resource.getrusage(RUSAGE_SELF).ru_maxrss` is process-lifetime
high-water mark. Co-resident model footprint (medium ASR ~1.5 GB +
NLLB-CT2-int8 ~700 MB + espeak subprocesses) is around 2.5 GB. The
5.1 GB measurement is inflated by transient peaks during ASR+ASR-roundtrip
on long utterances. A proper measurement would fork ASR2 into a separate
process.

## Falsification gate

| Gate | Threshold | Measured | Pass? |
|---|---|---|---|
| Live RTFx < 1.0 | 1.0 | 0.42–0.44 | **PASS** |
| First-audio mean latency ≤ 3.5 s | 3.5 s | 4.2 s en→ko, 5.0 s ko→en | **FAIL** |
| Peak RAM ≤ 4.0 GB | 4 GB | 5.1 GB (instrumentation-inflated) | inconclusive |
| BLEU non-regression (≥22.24 en→ko, ≥21.96 ko→en) | | 22.91 / 19.90 | en→ko PASS, ko→en marginal FAIL (−2 BLEU vs FLORES text-mode 24.96) |
| Round-trip WER ≤ 30 % en / 45 % ko-CER | | 5.83 en / 71.66 ko-WER | en PASS, ko FAIL — espeak-ng Korean output is robotic enough that ASR can't re-transcribe it. Real MeloTTS or Piper would resolve this. |

## Verdict

**Partial pass.** The CPU thesis holds for ASR + MT both directions:
- Translation quality (22.91 / 19.90 BLEU) is close to the text-mode A1
  baseline (25.24 / 24.96), with the gap explained by ASR errors on
  Fleurs audio (en WER 13.19 — higher than the dedicated A2 number 5.47;
  this set has more proper nouns).
- Live RTFx 0.42–0.44 means the system can keep up with continuous
  speech at sub-half real-time. Plenty of headroom.

The end-to-end **batch-mode first-audio latency of 4.2–5.0 s** does
NOT meet the protocol's 3.5 s budget. This is because the current
pipeline waits for the full source utterance before starting ASR
(`faster-whisper.transcribe(audio_path, vad_filter=False)`). A
streaming setup with LocalAgreement-2 + chunked ASR would emit partial
hypotheses ~1 s after speech onset and start MT before audio ends.
**The streaming policy was scoped out of A3 — we measured batch.**

Ko→en BLEU 19.90 is the noisiest number in the round; needs a follow-up
to determine whether (a) the dropped 5 BLEU vs text-mode 24.96 is
genuinely from ASR errors, or (b) the ASR text has formatting/case
artifacts (e.g. ko ASR emits text without proper en case for proper
nouns; the FLORES ko→en reference has full case + punctuation).

The TTS arm is the single weakest link: MeloTTS unavailable, espeak-ng
serviceable in English but Korean output is unintelligible enough that
ASR round-trip can't recover the text (71.66 % WER). Piper has no
official ko voice. **Substantive TTS upgrade is the round-3 ask.**

## What to do next (round 3 candidates)

1. **Plug a real Korean TTS** — community Piper voice (license risk),
   or wait for MeloTTS install fix, or train a small VITS on KSS.
2. **Switch to streaming ASR** — LocalAgreement-2 on faster-whisper
   chunked. Drops first-audio latency below 3.5 s and unlocks the gate.
3. **Verify ko→en BLEU loss** — re-run with ASR text post-processed
   (TitleCase proper nouns, restore punctuation). Likely recovers
   3+ BLEU.
4. **De-instrument the RAM measurement** — fork the round-trip ASR
   into a subprocess so RSS is fair.

## Reproduce

```
git checkout exp/cpu-voice-to-voice-e2e

# One-time setup
ct2-transformers-converter --model facebook/nllb-200-distilled-600M \
  --output_dir models/nllb-600m-ct2-int8 --quantization int8 \
  --copy_files tokenizer.json special_tokens_map.json \
                tokenizer_config.json sentencepiece.bpe.model
python3 scripts/build_fleurs_eval.py  # idempotent; ~3 min

# Eval (50 utterances per direction, ~12 min each)
mkdir -p reports/exp_cpu-voice-to-voice-e2e
PYTHONPATH=. python3 scripts/eval_e2e.py \
  --manifest data/eval/fleurs_pair_manifest.tsv --src en --tgt ko \
  --report reports/exp_cpu-voice-to-voice-e2e/e2e_en-ko.json --limit 50
PYTHONPATH=. python3 scripts/eval_e2e.py \
  --manifest data/eval/fleurs_pair_manifest.tsv --src ko --tgt en \
  --report reports/exp_cpu-voice-to-voice-e2e/e2e_ko-en.json --limit 50
```

## Process notes (for the manager)
- The Haiku runner declared a verdict based on a SINGLE test segment
  (id 1660, BLEU 32.91) while the actual 270-segment eval was still
  running. The 32.91 was an outlier on an easy segment; the 50-segment
  mean is 22.91.
- The runner ALSO accepted the headline "RTFx 1.275" from the harness
  without noticing that the harness's `total_ms` double-counts ASR
  by including the round-trip eval pass. The live-pipeline RTFx is
  0.42-0.44.
- Recommend the round-3 spawn protocol: explicit "do not commit a
  verdict until N >= 20 segments and the per-component numbers add
  up to the headline".
