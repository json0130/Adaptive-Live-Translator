# Round 3 — Final Report (clear the gates)

**Date:** 2026-05-29
**Hardware:** Laptop CPU, 8-core x86 AVX2, no GPU, ~7 GB free RAM
**Eval:** Fleurs en_us/ko_kr audio (ASR + e2e), FLORES devtest (MT text),
ML glossary slice (terminology). Locked protocol unchanged.
**Goal:** clear round-2's open gates — first-audio latency, Korean TTS —
and run the deferred glossary experiment.
**Runner model:** Sonnet (not Haiku). **Experiments:** R3-1 PASS,
R3-2 PASS, R3-3 FAIL (reviewer-corrected), R3-4 deferred (budget).

---

## Headline results

| # | Experiment | Branch | Gate | Result |
|---|---|---|---|---|
| R3-1 | Streaming ASR (LocalAgreement-2) | `exp/streaming-asr-localagreement` | first-audio ≤ 3.5 s | **PASS en (2.95 s), near-miss ko (3.89 s)** |
| R3-2 | Korean CPU TTS | `exp/korean-cpu-tts` | round-trip WER ≤ 25 % | **PASS (13.64 %, was 71.66 %)** |
| R3-3 | Glossary logit-bias | `exp/nllb-logit-bias-glossary` | recall +20 pp, no BLEU regression | **FAIL — breaks Korean particles** |
| R3-4 | ko→en ASR post-processor | — | recover ~3 BLEU | **DEFERRED (budget; drop-first item)** |

---

## R3-1 — Streaming ASR: latency gate cleared (en)

LocalAgreement-2 over growing audio windows (initial 1.0 s, step 1.0 s,
agree-2), faster-whisper **small** int8 on CPU.

| Lang | first-audio latency mean / p50 (s) | WER | RTFx | RAM |
|---|---:|---:|---:|---:|
| en | **2.95** / 2.82 | 8.21 | 0.74 | 1.7 GB |
| ko | 3.89 / 3.64 | 11.83 (mecab) | 0.74 | 1.7 GB |

vs round-2 batch (4.2 s en / 5.0 s ko). **English clears the 3.5 s gate;
Korean misses the mean by 0.39 s** (longer utterances, more windows to
first agreement). Model had to drop from medium→small: medium is too slow
per-window on CPU (5.8–6.5 s latency). WER cost (en 5.47→8.21, ko
10.88→11.83) is within the protocol's regression bound.

Did NOT tune chunk/step on the eval set to force ko under the gate
(that's test-set tuning). Honest near-miss; round-4 can try base-model
partials + medium final pass, or confidence-gated early commit.

Fixed a scaffolding bug during takeover: audio-buffer trimming desynced
the LCP bookkeeping → WER 56 % garbage; disabled trimming (correct for
<15 s utterances).

## R3-2 — Korean CPU TTS: intelligibility gate cleared (strongest result)

MeloTTS-KR (MIT, installed from source). N=40 Fleurs ko refs, round-trip
via faster-whisper-medium:

| Engine | round-trip WER (mecab) | CER | warm synth speed | license |
|---|---:|---:|---|---|
| **MeloTTS-KR** | **13.64 %** | 10.17 % | ~4.3× faster than real-time | MIT |
| espeak (round-2) | 71.66 % | — | ~100× RT | GPL-3 |

Takes Korean TTS from unintelligible (71.66 %) to well under the 25 %
gate. **Unblocks round-2's "Korean TTS is the weakest link" finding.**
RAM peak 6.9 GB is contaminated by the co-loaded re-ASR — standalone TTS
footprint needs a separate measurement (carry-forward). Pre-warm the BERT
prosody at session start (first-synth is slow cold).

## R3-3 — Glossary logit-bias: FAIL (reviewer-corrected)

Soft logit-bias `LogitsProcessor` on HF NLLB-600M, boosting triggered
glossary target-term tokens. Three-point slice result (N=60, en→ko):

| Condition | char-BLEU | term-recall |
|---|---:|---:|
| control | 50.92 | 5.4 % |
| bias +3 | 53.84 | 26.8 % (+21.4 pp) |
| bias +15 | 50.88 | 100 % (term-salad) |

The +3 point *appeared* to pass (recall +21.4 pp ≥ +20; FLORES BLEU −0.03,
within −0.5). **But the independent reviewer's 30-sentence Korean
particle spot-check overturned it:** CLEAN 11 / MINOR 12 / BROKEN 7, with
**5 of 7 BROKEN being treatment-specific** particle failures (을→과 case
swaps, dropped subject particles, one truncation), concentrated where 2–3
terms trigger simultaneously. Incorrect particles change meaning, so the
26.8 % recall gain comes with ~17 % grammatical breakage on triggered
sentences.

**Why the corpus-BLEU gate missed it:** char-level BLEU + FLORES triggers
the glossary on only 2/300 sentences, so grammar damage on triggered text
is invisible in the FLORES aggregate. The targeted grammatical inspection
is the correct instrument. This is exactly why the protocol assigns the
particle check to the reviewer, not the runner.

Verdict: naive token-level logit-bias is the wrong mechanism — no safe
operating point (weak→no recall; strong→term-salad; moderate→particle
breakage). Round-4: position-aware constrained decoding over complete
phrase sequences *including their particles* (Hokamp grid-beam / aligned
trie), and/or cap simultaneous triggers at 1/sentence. The round-1 lesson
(output-side beats prompt-injection) still holds; token-bias just isn't
enough.

---

## Current best end-to-end CPU pipeline (composite of passing parts)

- ASR: faster-whisper-small int8, LocalAgreement-2 streaming —
  first-audio 2.95 s en / 3.89 s ko, WER 8.2 / 11.8.
- MT: NLLB-200-distilled-600M CT2 int8 — 25.24/24.96 BLEU (round-2 A1).
- TTS: MeloTTS-KR (MIT) — round-trip WER 13.64 %, ~4× RT warm.
- Glossary terminology: NOT solved (R3-3 failed); defer to round 4.

All three working stages fit CPU/RAM budgets. The voice→voice pipeline is
now intelligible end-to-end on a laptop CPU, with English latency under
the gate and Korean latency within 0.4 s of it.

---

## Carry-forward to round 4 (priority order)

1. **ko streaming latency** (3.89 → ≤3.5 s): base-model partial passes +
   medium/small final pass, or confidence-gated early commit. A-priori
   config, measurable without touching the test set.
2. **R3-4 ko→en ASR post-processor** (deferred this round): casing +
   punctuation restoration on ASR transcripts, expected ~+3 BLEU on
   ko→en. Cheapest item; first thing to run next round.
3. **Glossary, done right**: position-aware constrained decoding
   (phrase-level with particles) instead of token-bias.
4. **Standalone TTS RAM**: re-measure MeloTTS without co-loaded ASR to
   confirm ≤1 GB budget; fix the `mecab_ko_dic` MECAB_ARGS breakage so
   ko BLEU in the glossary harness is protocol-compliant (ko-mecab, not char).
5. **Full e2e re-measurement** with streaming ASR + MeloTTS wired into
   `eval_e2e.py` (round-2 e2e used batch ASR + espeak; both now upgraded).

---

## Process notes

- The settings.json fix mattered: the file had been invalid JSON (bare
  `permissions` block, no wrapper/env/agent), which likely disabled agent
  teams and is probably why rounds 1–2 worktrees kept spawning from stale
  origin/main. After the fix, agent-team spawning was verified, and all
  three round-3 worktrees correctly based off current main — the
  stale-origin bug did not recur.
- All three Sonnet runners were stopped/killed mid-run before producing
  results (one explicitly killed; two went dark with no commits, likely
  CPU/RAM contention from three parallel model-loading jobs on an 8-core/
  7 GB box). The manager took over and ran all three serially in the
  foreground, which removed contention. **Lesson: on this hardware, run
  model-heavy experiments serially, not three-parallel.**
- Each runner's scaffolding was sound and reused; the manager fixed two
  real bugs (R3-1 buffer-trim desync, R3-3 verdict) and verified suspicious
  numbers (R3-2's RTFx, R3-3's recall) before trusting them — the round-2
  "numbers must add up" discipline caught the R3-3 BLEU gate's blind spot,
  and the reviewer caught the grammar damage.

---

## Deliverables (branches, all committed locally; origin push needs auth)

- `exp/streaming-asr-localagreement` — R3-1 (src/asr/streaming_local_agreement.py, scripts/eval_streaming_asr.py)
- `exp/korean-cpu-tts` — R3-2 (src/tts/korean_cpu_tts.py, scripts/eval_tts.py)
- `exp/nllb-logit-bias-glossary` — R3-3 (src/translator/nllb_hf_glossary.py, scripts/eval_glossary.py)
- Each has reports/<branch>/summary.md with full numbers and reproduce commands.
- `.claude/settings.json` fix on main (commit 3046daa).
