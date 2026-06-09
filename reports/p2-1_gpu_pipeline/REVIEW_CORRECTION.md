# P2-1 — Reviewer-corrected verdict (commit 961fc15)

The experiment-runner reported **"PROMISING — architecture validated."**
Reviewer + manager **overturned the headline.** Corrected verdict below.
(Third runner-headline correction in this project: R4, R5, now P2-1.)

## Corrected per-gate verdict (N=270 each direction, full Fleurs manifest)

| Gate | Threshold | en→ko | ko→en | Verdict |
|---|---|---|---|---|
| BLEU vs R4-1 | ≥24.03 / ≥19.39 | 23.63 | 19.49 | QUALIFIED PASS (see N-mismatch) |
| BLEU regression | ≤1.0 | −0.40 | +0.10 | PASS |
| First-emission | ≤1.5s | 1.853s | 1.732s | **FAIL** |
| **E2E (honest)** | ≤2.5s | **4.115s** | **6.110s** | **FAIL — falsified (>3s)** |
| Peak VRAM | ≤7 GB | 5.02 GB | 4.85 GB | PASS |

## The e2e correction (the overturned number)

Runner reported e2e = **2.300s / 1.880s** by summing ASR **first-emission**
+ MT + TTS (`scripts/eval_e2e_p21.py:269`, `asr_latency_s =
first_emission_latency_s`). But MT runs on `result.accuracy_transcript`
(the FULL streaming transcript, line 205) — MT cannot start until ASR
fully completes. The honest e2e uses the recorded full-decode wall time
`asr_e2e_ms`:

Recomputed (committed JSON fields `asr_e2e_ms + mt_ms + tts_ms`, verified
independently by manager and reviewer):

- en→ko: honest e2e mean **4114.8 ms** (median 3531 ms; 174/270 segments >3s)
  - components: asr_e2e_ms **3667.8** + mt 144.7 + tts 302.3
- ko→en: honest e2e mean **6110.1 ms** (median 5203 ms; 240/270 segments >3s)
  - components: asr_e2e_ms **5962.1** + mt 148.0 + tts 0.0

Both exceed the ≤2.5s pass gate AND cross the plan's >3s falsify line.
This is the R4-1 lesson repeated: latency and quality must come from the
same path.

## Root cause (concrete, fixable — not a hard limit)

e2e is dominated by `asr_e2e_ms` (full streaming-decode compute), inflated by:
1. **Buffer trimming DISABLED** — every LA-2 window re-transcribes the whole
   growing buffer from the start (`streaming_local_agreement.py:9,29`),
   i.e. O(n²) decode. No realtime sleep; this is real compute, not audio
   duration. Enabling trimming (drop committed audio) should cut this
   sharply.
2. **MT is batch-on-full-transcript, not incremental.** Even with cheap
   ASR, e2e = full_asr_decode + mt + tts. A simultaneous design would
   translate committed ASR chunks as they arrive.
3. `initial_chunk_s=1.0` sets a ~1.4s first-emission floor (first-emission
   FAIL); 0.5s would lower it but does not fix the e2e accounting.

## What is genuinely SUPPORTED (real results)

- **BLEU 23.63 / 19.49** — protocol-correct (ko-mecab / 13a), N=270, no
  regression beyond ≤1.0. (Caveat: R4-1's 24.03/19.39 were N=50 — not
  strictly apples-to-apples; regression gate still holds.)
- **VRAM 5.02 GB inference peak** (nvidia-smi whole-process; captures CT2
  int8 ASR that torch's counter misses). The int8_float16 ASR decision held
  — fits the 8 GB card with no OOM, well under P2-0's 7.16 GB fp16 estimate.
- **Phase-3 finding stands:** ~5 GB inference + ~2 GB headroom cannot
  co-reside LoRA training → Phase 3 must unload-and-reload.

## Caveats the runner did not flag

- **ko→en is NOT voice-to-voice.** TTS only runs for Korean output
  (`tts_ms=0` for ko→en; MeloTTS is Korean-only). The "voice-to-voice
  validated" claim holds for en→ko only. A real ko→en voice path would add
  English TTS on top of the already-failing 6.1s.
- ko→en VRAM (4.85 GB) includes MeloTTS loaded-but-idle.

## Bottom line

GPU composition + VRAM + translation quality are **validated**. The
**working real-time system is NOT** — e2e latency fails honestly, gated by
ASR streaming-decode cost (buffer trimming) and a non-incremental MT stage.
This reconfirms Phase 1's finding (ASR is the latency bottleneck, not the
translator) now on GPU with honest accounting. The failure has a concrete,
testable fix hypothesis (enable buffer trimming ± incremental MT), not yet
run.
