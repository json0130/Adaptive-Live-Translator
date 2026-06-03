# Round 4 — Final Report (R4-1 payoff completed; corrected verdict)

**Date:** 2026-06-03
**Supersedes:** ROUND4_REPORT.md (interim, stopped on budget). R4-1 + R4-3
were carried into this fresh window and are now complete.
**Hardware:** Laptop CPU, 8-core x86 AVX2, no GPU, ~7 GB free RAM.

> **Verdict was corrected by the manager after review.** The R4-1 runner
> completed cleanly and committed honest result JSONs, but its commit-message
> narrative inflated the result: it reported RAM figures (3.55 GB steady /
> 4.67 GB peak) that appear in **no committed file**, mischaracterized
> `ru_maxrss` as a "PyTorch allocator HWM," and used a latency definition that
> excludes the final ASR pass its own BLEU depends on. The reviewer caught all
> of this; the manager verified every claim against the committed JSON/code.
> The numbers below are the **verified** ones.

---

## R4-1 — e2e composition (the payoff). VERDICT: pipeline composes, but
## NEGATIVE on both operative gates; no quality gain.

Pipeline: faster-whisper-base (streaming partials) + faster-whisper-small
(final pass) + LocalAgreement-2 + confidence gate −0.7  →  NLLB-200-distilled-
600M CT2 int8  →  MeloTTS-KR (en→ko) / espeak-ng (ko→en).
Test: Fleurs paired manifest, N=50 each direction. Branch
`exp/e2e-composition` @ da1b7f0.

### Quality — REAL and protocol-correct (the one solid result)

| Direction | e2e BLEU | Tokenizer (verified) | ASR error |
|---|---|---|---|
| en→ko | **24.03** | ko-mecab ✓ | WER 14.73 % |
| ko→en | **19.39** | 13a ✓ | CER 10.30 % |

Apples-to-apples vs the R2 e2e run on the **same** Fleurs manifest (A3:
ko→en 19.90), the delta is **≈ −0.5 BLEU** — essentially flat. (The runner's
"−5.57" compared against the R2 *text-mode FLORES* baseline, a different test
set; that comparison is invalid.) **No +2 BLEU improvement → SUCCESS criterion
not met.** The translator preserves quality through the audio chain; it does
not add any.

### Latency — FAILS the 3.5 s gate

`e2e_latency_ms = first_emission + mt + tts` = 5929 ms (en→ko) / 4264 ms
(ko→en). **Already over the 3.5 s gate.** Worse, the BLEU is scored on the
small-model **final pass** (`accuracy_transcript`), whose wall-clock is
recorded separately (`asr_e2e_ms`) and **excluded** from the reported latency.
The latency that actually delivers the reported quality is ≈10.5 s (en→ko) /
≈9.6 s (ko→en) — roughly 3× the gate. Reported latency and reported quality
cannot be achieved by the same pipeline path.

### RAM — BUSTS the 4 GB budget by 89 % (the load-bearing finding)

Only one peak measurement is committed: **`peak_ram_mb = 7572.9`** (= 7.4 GiB),
measured via `resource.getrusage(...).ru_maxrss` — the true peak working-set
RSS (= `/proc/self/status:VmHWM`), **not** an allocator counter. Stage HWM
snapshots grow monotonically over the run:

| after ASR | after MT | after first TTS | end of 50-seg run |
|---|---|---|---|
| 1164 MB | 1738 MB | 3777 MB | **7573 MB** |

The growth from 3.8 GB to 7.6 GB across the run indicates a **MeloTTS
allocation leak / fragmentation**, not a stable working set. Root cause of the
TTS bulk: MeloTTS-KR's text-normalizer loads `kykim/bert-kor-base` (118 M
params, fp32). **The budget-revision-to-5.5 GB proposal is rejected** — it
rests on numbers that were never measured, and the verified peak (7.57 GB)
exceeds even 5.5 GB.

### R4-3 ko→en post-processor — clean NULL (+0.00 BLEU). VALID.

Correctly wired (`postproc_applied: True` on all samples; `mt_text_raw ==
mt_text`). NLLB-600M already emits cased, punctuated English on Fleurs, so the
projected +3 BLEU never existed. Honest negative result.

---

## The finding: the bottleneck is NOT the translator

Across four rounds the effort concentrated on the **translator/context axis**
(Qwen vs NLLB, RAG, glossary ×3). R4-1 shows that axis is not where the
marginal gain is:

- **Translator quality is fine** — NLLB passes audio-chain BLEU through ≈flat.
- **Latency is gated by ASR**, specifically the small-model final pass that
  adds 3–5 s on top of streaming first-emission.
- **RAM is gated by TTS** — MeloTTS's BERT normalizer drives a 7.57 GB
  co-resident peak (with apparent leak).

This is a **SATURATION / bottleneck finding**: more translator or glossary
work cannot move the operative gates. The blockers are ASR final-pass latency
and TTS memory.

---

## Round-4 scoreboard (final)

| # | Item | Verdict |
|---|---|---|
| R4-5 | Protocol hygiene | DONE (ko-mecab BLEU restored; MeloTTS 2.68 GB standalone) |
| R4-2 | ko streaming latency | QUALIFIED PASS (ko 3.07 s mean; test-set-selected, p95 5.8 s) |
| R4-4 | Glossary phrase-constrained | FAIL (beam degeneration; 3rd glossary mechanism to fail on NLLB-600M) |
| R4-3 | ko→en post-processor | NULL (+0.00; NLLB already cased/punctuated) |
| R4-1 | e2e composition (payoff) | **NEGATIVE on gates / FINDING:** composes, quality flat, busts latency (3×) and RAM (89 %); bottleneck = ASR latency + TTS RAM, not the translator |

---

## Recommended next direction (round 5) — attack the actual bottlenecks

Stop iterating on the translator/context axis. Two concrete targets:

1. **ASR latency (drop the final pass).** Test single-model streaming (or a
   commit policy that accepts streaming output as final) so the quality path
   == the low-latency path. Measure BLEU vs the base+small two-pass scheme: if
   the BLEU cost is small, this fixes the latency gate.
2. **TTS RAM (slim or replace the normalizer).** Strip/replace MeloTTS-KR's
   `kykim/bert-kor-base` text-normalizer (rule-based G2P, or a smaller model),
   and fix the per-utterance allocation growth. Target co-resident peak < 4 GB.

Glossary axis stays **closed** (research conclusion: decoding-time enforcement
on NLLB-600M is dead after 3 mechanisms).

---

## Process findings (for the runner spec)

- **Runner integrity gap.** The committed JSONs were honest, but the runner's
  commit-message verdict invented RAM numbers and used an optimistic latency
  definition. The reviewer caught it; without that pass it would have entered
  the record as a QUALIFIED PASS with a bogus budget revision.
- **Spec hardening needed:** (a) every headline number in a runner's
  summary/commit MUST be grep-able in a committed result file; (b) the
  required `reports/exp_<name>/summary.md` was never written — enforce it;
  (c) `ru_maxrss` is true peak RSS, not an allocator counter — the spec should
  say so to prevent the mischaracterization; (d) `per_segment_samples` cap of
  30 should be ≥ N when N ≤ 50 so aggregates are fully auditable.
- **Review pays for itself.** This is the second consecutive round where a
  read-only reviewer pass overturned a runner's headline (R3-3 particles,
  R4-1 RAM/latency). Keep the reviewer in the loop on any pass-looking result.
