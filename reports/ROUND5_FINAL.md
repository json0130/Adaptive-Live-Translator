# Round 5 — Final Report (terminated on BUDGET; outcome: ARCHITECTURAL LIMIT)

**Date:** 2026-06-04
**Hardware:** Laptop CPU, 8-core x86 AVX2, no GPU, ~7 GB free RAM.
**Scope (strict, human-approved):** exactly two experiments, each owning one
gate. R5-1 = the 3.5 s end-to-end latency gate. R5-2 = the < 4 GB co-resident
RSS gate. No third experiment. Translator/context and glossary axes CLOSED.
**Termination:** BUDGET — R5-2's runner hit a session-limit wall mid-run
(reset deferred). Per the standing budget rule, the manager took over, salvaged
and verified the partial result, and did NOT start a re-run or a third
experiment.

---

## Outcome: BOTH FAIL → ARCHITECTURAL LIMIT

Per the round-5 three-category framing (ROUND5_PLAN.md): neither standard fix
moves its gate on this CPU hardware class. This is a publishable **negative
finding**, not a partial pass.

| Experiment | Gate | Result | Verdict |
|---|---|---|---|
| R5-1 single-pass ASR | e2e latency < 3.5 s | best 3.87 s (ko→en), 5.47 s (en→ko) | **FAIL** |
| R5-2 TTS-RAM slim | co-resident peak < 4 GB | run peak ~6.6 GB by seg 30 | **FAIL** (partial, well-supported) |

The two operative gates are blocked by **distinct, hardware-rooted
bottlenecks** — and crucially, neither is the thing four prior rounds optimized
(the translator/context axis). R4-1 first showed this (SATURATION); round 5
confirms it from two independent angles.

---

## R5-1 — single-pass ASR. VERDICT: FAIL (clean, N=50, reviewed by manager)

Branch `exp/asr-single-pass` @ f426909. Removed the small-model final re-decode
so the transcript MT consumes IS the streaming output (fixing R4-1's
honesty defect: latency and quality now come from the same path). Two variants,
N=50 each direction, ko-mecab/13a verified, full 50-segment dumps, numbers
reconciled and grep-able.

| Direction | Variant | BLEU | e2e latency mean | vs two-pass BLEU |
|---|---|---|---|---|
| en→ko | V1 small | 22.31 (ko-mecab) | 7.14 s | −1.72 |
| en→ko | V2 base | 19.92 | 5.47 s | −4.11 |
| ko→en | V1 small | 15.60 (13a) | 5.45 s | −3.79 |
| ko→en | V2 base | 13.47 | **3.87 s** (best) | −5.92 |

**No variant meets 3.5 s in either direction.** Single-pass did the right
thing (honest path, wall-clock ~10.5 s → ~8 s) but cannot reach the gate.
Bottlenecks pinned:
- **en→ko: TTS.** MeloTTS-KR alone is ~3.0 s/utt; ASR first-emission (~2 s)
  already blows the per-stage budget before TTS.
- **ko→en: ASR first-emission.** Any faster-whisper model on CPU emits its
  first token at ~3.3 s (base) / ~4.9 s (small) — over the gate on its own.

## R5-2 — TTS-RAM slim. VERDICT: FAIL (partial run; result well-supported)

Branch `exp/tts-ram-slim`. Approach: disable MeloTTS-KR's `kykim/bert-kor-base`
text-normalizer (`disable_bert=True`). Runner died on a session limit at ~30/50
segments — no committed JSON, no round-trip-WER (intelligibility) measurement,
no summary.md. The salvaged run log (committed at
`reports/exp_tts-ram-slim/en_ko_N50_slim.log`) nonetheless settles the RAM gate:

| Point in run | peak_hwm (ru_maxrss) | instantaneous VmRSS |
|---|---|---|
| baseline | 38 MB | 38 MB |
| after ASR | 1164 MB | 933 MB |
| after MT | 1738 MB | 1638 MB |
| after TTS load + prewarm | **2934 MB** | 2936 MB |
| 10/50 segments | 6336 MB | 6133 MB |
| 20/50 segments | 6637 MB | 6290 MB |
| 30/50 segments | 6637 MB | **6362 MB** |

Two findings:
1. **Disabling the BERT normalizer helped only at load** (3777 → 2934 MB,
   ~0.84 GB saved) — under 4 GB *at load*, confirming the normalizer was ~1.5 GB
   of the bulk.
2. **The per-utterance allocation growth — the harder sub-problem — is
   unsolved and dominates.** Instantaneous VmRSS (not just the monotonic HWM)
   climbs to ~6.1 GB by segment 10 and ~6.4 GB by segment 30. The run busts the
   4 GB gate by segment 10 and plateaus near 6.6 GB.

**FAIL is well-supported despite the partial run:** the gate is the peak over
the run, and the run already sits at ~6.6 GB by segment 30 — completing to 50
cannot bring it under 4 GB. (Caveat held honestly: intelligibility after
disabling BERT was never measured, so even the load-time saving is unconfirmed
as usable. It does not change the verdict — the RAM gate fails on the growth
regardless.) The real fix is bounding the per-synthesis allocation leak, which
was not reached before the budget wall.

---

## The finding (round 4 + round 5 together)

The adaptive-live-translator's deployability on this CPU/7 GB-laptop class is
**not** gated by translation quality (NLLB passes audio-chain BLEU through
≈flat) or by any translator/context mechanism (glossary dead after 3 rounds;
translator axis saturated). It is gated by two hardware-rooted limits the
standard fixes do not move:

1. **Latency floor** — CPU Whisper first-emission (~3.3 s) + MeloTTS synthesis
   (~3 s) each individually approach or exceed the 3.5 s gate. Removing the ASR
   final pass (R5-1) does not help enough.
2. **Memory ceiling** — MeloTTS synthesis leaks/accumulates to ~6.6 GB over a
   run; removing its BERT normalizer (R5-2) saves only ~0.84 GB at load and
   does nothing about the runtime growth.

**Recommendation:** treat the 3.5 s / 4 GB targets as infeasible for a
CPU-only deployment of this cascade. To proceed, change the hardware/engine
class, not the translator:
- a GPU (or NPU) to collapse Whisper first-emission and TTS synthesis time, and
- a fundamentally lighter Korean TTS engine (or a fixed MeloTTS that bounds
  per-utterance allocation — the leak itself is a concrete, isolatable bug and
  the single most promising CPU-side lead if the effort resumes).

These are architecture/hardware decisions, not another experiment in this loop.

---

## Carry-forward (only if a future window revisits the CPU path)
- **R5-2 clean completion** is the one genuinely-unfinished item: run all 50,
  add the round-trip-WER intelligibility guardrail, and — the real prize —
  diagnose and bound the MeloTTS per-utterance allocation growth. If that leak
  is fixable, R5-2's verdict could move; the load-time RAM (2.93 GB) is already
  under budget. This is the only lead that could overturn a round-5 FAIL.
- Do NOT reopen the latency gate on CPU (R5-1 closed it) or the
  translator/glossary axes (closed since rounds 3–4).

---

## Process notes
- **Third consecutive round a reviewer/manager check overturned or corrected a
  runner's headline** (R3-3 particles, R4-1 RAM/latency, R5-2 here required
  salvage). The hardened spec (committed at 950808f) held where it was tested:
  R5-1 produced fully-reconciled, grep-able numbers with full per-segment dumps
  and a mandatory summary.md — a clean contrast to R4-1.
- **Repo damage from a botched git-push, repaired.** The R5-2 session produced
  a junk commit `a011fbf` ("h") that deleted `.claude/next-session-plan.md`;
  it reached main and origin (same "stray h / botched push redirect" pattern as
  the earlier `cfff008` cleanup). The plan file was restored forward from
  950808f; history was NOT rewritten (a011fbf is already on origin). **Action
  item:** the `git push` path keeps generating an "h" artifact — the push
  workflow needs fixing before the next round.
- Two of five round-4/5 runners died on session-limit walls. The
  manager-takeover-and-verify pattern contained the loss both times, but the
  recurring session ceilings are themselves the budget signal that ends here.
