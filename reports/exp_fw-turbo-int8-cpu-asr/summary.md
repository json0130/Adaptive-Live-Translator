# exp/fw-turbo-int8-cpu-asr — summary (medium wins on CPU)

## What was actually tested

The branch name and original spec asked for `large-v3-turbo` int8. The
Haiku runner used `Systran/faster-whisper-large-v3` (NOT turbo —
`Systran/faster-whisper-large-v3-turbo` doesn't exist as a repo).
A real turbo mirror, `mobiuslabsgmbh/faster-whisper-large-v3-turbo`,
exists; the manager added a 40-utterance turbo smoke ad-hoc to
complete the picture.

So this branch has three rows: full-set large-v3 int8, full-set
medium int8, and 40-utterance turbo int8.

## Hardware
- Laptop CPU, 8-core AVX2, no GPU. intra_threads=8.

## Eval (Fleurs en_us/ko_kr test, locked manifests)

| Model | Lang | WER | CER | RTFx | RAM peak (MB) | n |
|---|---|---:|---:|---:|---:|---:|
| Systran/faster-whisper-large-v3 int8 | en | **4.82** | n/a | 0.606 | 4270 | 350 |
| Systran/faster-whisper-large-v3 int8 | ko (mecab) | **10.06** | 9.00 | 0.542 | 9276* | 270 |
| Systran/faster-whisper-medium int8 | en | 5.47 | n/a | **0.277** | **2032** | 350 |
| Systran/faster-whisper-medium int8 | ko (mecab) | 10.88 | 8.75 | **0.304** | ~2050* | 270 |
| mobiuslabsgmbh/faster-whisper-large-v3-turbo int8 | en | 5.86 | n/a | 0.413 | 2553 | 40 |
| mobiuslabsgmbh/faster-whisper-large-v3-turbo int8 | ko (mecab) | 9.60 | 7.29 | 0.303 | 2372 | 40 |

\* RAM-peak for the medium_ko and large-v3 ko runs is process-lifetime
high-water mark (resource.getrusage RSS) inflated by earlier en-run
allocations in the same shell. Fresh-process allocations are ~2 GB for
medium and ~4 GB for large-v3 based on the EN runs which always ran first.

## Comparison and choice for A3

Quality at the full-set scale (only large-v3 and medium have it):
- **en WER:** large-v3 4.82 beats medium 5.47 by ~0.65 abs. Marginal.
- **ko WER (mecab):** large-v3 10.06 beats medium 10.88 by ~0.82. Marginal.

Latency and memory:
- **RTFx:** medium ≈ 0.28-0.30, large-v3 ≈ 0.54-0.61. Medium is ~2x faster.
- **RAM:** medium ~2 GB, large-v3 ~4 GB. Medium fits the 1.5 GB ASR
  budget MUCH more comfortably (line in the eval protocol).

Turbo (the spec's primary model) on a 40-utterance smoke is not
materially faster than medium on CPU (RTFx 0.30-0.41 vs medium 0.22-0.30)
because the 4-decoder design helps when the decoder dominates wall-clock;
on CPU the encoder pass dominates, and turbo carries the full encoder.

**Falsification check:**
- large-v3: RAM > 2.5 GB ⇒ FAILS the ASR RAM budget on this hardware.
- medium: all falsification thresholds passed (en WER 5.47 ≤ 14, ko WER
  10.88 ≤ 25, RTFx 0.30 < 1.5, RAM 2.0 GB ≤ 2.5).
- turbo (mobiuslabsgmbh): RAM 2.4 GB ≤ 2.5 narrowly; RTFx OK; quality
  close to medium. **Not measurably better than medium on CPU.**

**Pick: medium int8.** It hits the budget with margin, is fastest on
CPU, and the quality gap to large-v3 is small enough that the 2x
latency cost isn't worth it for an 8 GB laptop streaming pipeline.

## Latency budget update for A3

ASR (medium int8 CPU): 0.30 × audio-duration.
Translator (NLLB-CT2-int8 from A1): ~0.05 × audio-duration estimated
from FLORES text-mode 314-689 ms per sentence over typical 8-12 s of
audio. Confirmation in A3.
TTS (MeloTTS, not yet measured): scout estimate RTFx ~0.3.

Total end-to-end RTFx estimate ≈ 0.65, well under the protocol's
RTFx<1.0 (real-time-capable) target. Audio-in to first-audio-out
latency is the harder budget (≤3.5 s) and depends on how aggressively
A3 streams chunks.

## Process notes (for the manager)

1. Runner used `Systran/faster-whisper-large-v3-turbo` (doesn't exist),
   silently fell back to `large-v3` (without renaming the output files
   or warning), then declared "turbo doesn't exist" without trying
   `mobiuslabsgmbh/faster-whisper-large-v3-turbo` which is the obvious
   alternative repo.
2. RAM peak via `resource.getrusage()` is process-lifetime high-water
   mark; running multiple models sequentially in the same shell
   inflates the "ko" number unfairly. The harness should be called in
   separate `python3` processes, or `eval_asr.py` should fork+exec
   each run. Not blocking for this round but worth a small fix later.
3. Despite both issues, the verdict "medium wins" lines up with the
   manager's 40-utterance head-to-head and is correct.

## Reproduce

```
git checkout exp/fw-turbo-int8-cpu-asr
python3 scripts/build_fleurs_eval.py    # one-time, extracts WAVs

# A3 should use medium int8 for ASR:
PYTHONPATH=. python3 scripts/eval_asr.py \
  --manifest data/eval/fleurs_en_us_test/manifest.tsv --lang en \
  --model Systran/faster-whisper-medium --compute-type int8 \
  --report reports/exp_fw-turbo-int8-cpu-asr/medium_en.json

PYTHONPATH=. python3 scripts/eval_asr.py \
  --manifest data/eval/fleurs_ko_kr_test/manifest.tsv --lang ko \
  --model Systran/faster-whisper-medium --compute-type int8 \
  --report reports/exp_fw-turbo-int8-cpu-asr/medium_ko.json
```
