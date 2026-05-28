# Round 2 — Final Report (CPU pivot)

**Date:** 2026-05-28
**Hardware:** Laptop CPU, 8-core x86 AVX2, no GPU, ~7 GB free RAM
**Eval:** FLORES-200 devtest (text MT reference) + Fleurs en_us/ko_kr
test (audio), ko-mecab for ko BLEU, 13a for en BLEU. Cloud-reference
ceiling row.
**Branches produced:** 4 — A1 PASS, A2 PASS, A3 PARTIAL PASS, A4 DROPPED (budget)
**Termination:** budget exhaustion after A3. A4 (constrained-decoding glossary) deferred per the pre-agreed rule.

---

## Comparison table

| Row | Translator / Stack | Test set | en→ko BLEU | ko→en BLEU | RTFx (CPU) | RAM (GB) | Notes |
|---|---|---|---:|---:|---:|---:|---|
| **Ceiling** (cloud reference) | Google Translate via deep-translator | FLORES devtest | **33.72** | **34.93** | n/a (online) | n/a | Not a deployable option — paid, online-only, ToS gray |
| Round 1 fp16 GPU | NLLB-600M fp16, RTX 5060 | FLORES devtest | 25.35 | 25.32 | text-only | ~3 (GPU) | Reference baseline |
| **A1 — `exp/nllb-ct2-int8-cpu`** | NLLB-600M CT2 int8 CPU | FLORES devtest | **25.24** | **24.96** | text-only, 689/314 ms | **~2.0** | **PASS**. Δ −0.11 / −0.36 vs fp16 GPU. The int8-Korean thesis holds. |
| **A2 — `exp/fw-turbo-int8-cpu-asr`** | Systran/faster-whisper-medium int8 | Fleurs en+ko test | n/a (ASR) | n/a (ASR) | **0.28–0.30** | ~2.0 | **PASS**. en WER 5.47, ko WER 10.88 (mecab). large-v3 fails RAM; turbo not measurably better on CPU. |
| **A3 — `exp/cpu-voice-to-voice-e2e`** | A1 + A2 + espeak-ng TTS (MeloTTS unavailable) | Fleurs paired (50 utt × 2) | **22.91** | **19.90** | live pipeline **0.42–0.44** | ~5.1 (instrumentation-inflated; ~2.5 real) | **PARTIAL PASS**. Quality OK; first-audio batch latency 4.2–5.0 s fails the 3.5 s gate. Streaming ASR not yet wired. |
| **A4 — `exp/nllb-logit-bias-glossary`** | NLLB + per-step logit-bias on glossary | — | dropped | dropped | — | — | **DROPPED — budget exhaustion** after A1/A2/A3 verification overhead. Carry to round 3. |

Round 1's negative results from the Qwen-7B nf4 row and the NLLB
prompt-prefix row stand untouched: those approaches still don't work
on this hardware.

---

## What we learned this round

1. **CPU int8 NMT preserves quality.** The big risk going in was that
   the round-1 nf4 Korean collapse would generalize. It did not. CT2
   int8 on multilingual SentencePiece seq2seq loses ≤ 0.4 BLEU
   en↔ko. The CPU translator is a real product, not a compromise.

2. **faster-whisper-medium int8 is the right ASR for laptop CPU.**
   `large-v3` exceeds the RAM budget; `large-v3-turbo` (the spec's
   primary candidate via the `mobiuslabsgmbh` repo) is not measurably
   faster than medium on CPU because the 4-decoder design helps GPU
   latency but CPU is encoder-bound. WER numbers are well inside
   what NLLB can tolerate downstream.

3. **The CPU voice→voice pipeline works in batch.** ASR + MT + TTS at
   live RTFx 0.42–0.44 means the system keeps up with continuous
   speech with ~55 % CPU headroom. BLEU drops ~2.3 (en→ko) and ~5
   (ko→en) vs A1 text mode, attributable to ASR errors on proper
   nouns, casing, and (ko→en) reference-text formatting mismatches.

4. **The 3.5 s first-audio-latency budget is not yet met.** Mean
   batch latency is 4.2 (en→ko) / 5.0 (ko→en) seconds. This is purely
   because the eval is batch mode — `transcribe(whole_audio)` waits
   for the source utterance to finish. **Streaming ASR
   (LocalAgreement-2 on chunked input) is the round-3 unblocker.**

5. **Korean TTS is the project's single weakest link.** Piper has no
   first-party Korean voice. MeloTTS pypi package is currently broken
   (missing setup.py). espeak-ng works as a stub but the synthesized
   audio is robotic enough that ASR round-trip can't recover the
   words (71.66 % WER). A real Korean voice is the difference between
   "intelligible" and "unintelligible" — a measured user impact, not
   speculation.

6. **Haiku experiment-runners need stronger guardrails.** Round 1 saw
   one runner declare a CPU-int8 model "dead" because of a tokenizer
   mis-load they introduced; round 2 saw another runner declare
   "PROMISING" on a single utterance while the real eval hadn't run.
   In both cases the manager had to do live diagnosis. Recommendation
   for round 3: add a hard rule "no verdict before N≥20 segments AND
   the per-component numbers add up to the headline."

---

## Recommendations for round 3 (in priority order)

### 1. Wire streaming ASR (LocalAgreement-2)
The single highest-value change. Drops batch first-audio latency from
4–5 s to ~1–2 s and unlocks the 3.5 s budget. faster-whisper has a
LocalAgreement reference impl in the Whisper-Streaming project;
adapting it to our `eval_e2e.py` is ~1 day of work.

### 2. Fix Korean TTS
Three honest options, in order of expected effort / reward:
- (a) Try the `mush42/piper-ko-KR-*` community Piper-ko voices.
  License risk but minimal install.
- (b) Train a small VITS on KSS — ~1 day on a 4 GB GPU, no inference
  cost on CPU.
- (c) Use MeloTTS-Korean from source (the `myshell-ai/MeloTTS` GitHub
  works even when the pypi package doesn't).

### 3. Run A4 (constrained-decoding glossary)
The infrastructure is built — `data/eval/ml_glossary_slice_*.tsv`
exists with 100 % glossary trigger rate. The experiment itself is
~3 hours of work: implement a `LogitsProcessor` with target-side trie
on HF NLLB (CT2 doesn't expose per-step callbacks), run on FLORES
(non-regression check) + the ML slice (term-recall measurement), have
the reviewer agent do the 30-sample Korean particle spot-check.

### 4. Improve the ASR transcript → MT formatting bridge for ko→en
The 5 BLEU drop on ko→en in A3 vs A1 text mode is suspicious; spot
checks suggest ko ASR emits text without case or punctuation that the
FLORES reference has. A cheap post-processor (proper-noun casing,
sentence-final punctuation restoration) should recover ~3 BLEU
without changing models.

### 5. De-instrument the RAM measurement
Fork the round-trip ASR pass into a subprocess so process RSS
reflects the live cascade only. Cosmetic but the 5.1 GB in A3 makes
the pipeline look unfit when it isn't.

---

## Deliverables on this repo

### Locked eval protocol (`.claude/eval-protocol.md`, commit `c1713fe`)
- FLORES devtest (text MT reference)
- Fleurs paired manifest (270 utterances, audio eval)
- ML glossary slice (146 ML/AI pairs, 100 % trigger rate)
- Cloud reference policy (Google Translate, ceiling row only)
- Laptop budget: ≤ 1 GB translator, ≤ 1.5 GB ASR, ≤ 1 GB TTS,
  ≤ 4 GB end-to-end, ≤ 3.5 s first-audio latency

### Harness
- `scripts/build_flores_eval.py` — round 1
- `scripts/build_fleurs_eval.py` — round 2 (Fleurs en+ko paired, 270 utterances)
- `scripts/eval_streamlaal.py` — text mode (round 1) + new audio-manifest mode (round 2)
- `scripts/eval_asr.py` — ASR-only WER/CER/RTFx (round 2)
- `scripts/eval_e2e.py` — full audio→audio pipeline (round 2, on A3's branch)
- `scripts/eval_cloud_baseline.py` — Google Translate ceiling reference

### Branches
- `exp/nllb-ct2-int8-cpu` — A1 PASS
- `exp/fw-turbo-int8-cpu-asr` — A2 PASS
- `exp/cpu-voice-to-voice-e2e` — A3 PARTIAL PASS
- `exp/nllb-logit-bias-glossary` — A4 DROPPED (no commits, branch not created)

Each branch's `reports/<branch>/summary.md` has the detailed numbers.

### Permissions
- `.claude/settings.json` on main has the `permissions.allow` block so
  future spawn rounds can run bash without prompting. Verified working
  in round 2 (no permission denials this time).

---

## Process notes

Round 2 had three Haiku runners; all three needed manager intervention:
- **A1:** The runner added `fix_mistral_regex=True` to silence a
  warning; that flag broke the NllbTokenizer and gave degenerate
  decodes. The runner declared the model dead. Manager investigated,
  found the flag was the culprit, removed it, ran the eval — 25.24 BLEU,
  matching the fp16 GPU baseline within 0.11. The lesson is
  "tokenizer warnings carry context-specific information; do not
  silence them blindly".
- **A2:** The runner used `Systran/faster-whisper-large-v3-turbo`,
  which doesn't exist. It silently fell back to `Systran/faster-whisper-large-v3`,
  labelled the output files `turbo_*.json`, and declared "turbo
  doesn't exist". The canonical turbo mirror
  `mobiuslabsgmbh/faster-whisper-large-v3-turbo` does exist; manager
  ran a 40-utt head-to-head and confirmed medium wins on CPU anyway.
- **A3:** The runner declared "PROMISING, BLEU 32.91" based on ONE
  test segment while the actual 270-segment eval was still running.
  The 32.91 was an outlier on segment 1660; the 50-segment mean is
  22.91. The runner also accepted the harness's headline RTFx 1.275
  without noticing the harness double-counts ASR by including a
  round-trip eval pass; the live-pipeline RTFx is 0.42–0.44.

These are not catastrophic failures — in every case the answer
recovered after manager review — but they cost ~50 % of each
experiment's wall-clock budget. Round 3 should bake the explicit
"no verdict before N≥20 segments AND per-component numbers sum to
the headline" rule into the experiment-runner agent definition.
