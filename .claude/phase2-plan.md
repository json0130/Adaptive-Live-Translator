# Phase 2 — GPU Validation (working voice-to-voice translator)

**Status of Phase 1:** closed as a research artifact (rounds 1-5, see
README postmortem). Five rounds proved the translator/context axis is
saturated and CPU-only deployment is gated by hardware ceilings
(Whisper ~3.3s first-emission, MeloTTS RAM growth to ~6.6 GB). Closed
clean. Not reopening any of those axes.

**Phase 2 goal:** stand up a working voice-to-voice en↔ko translator on
GPU (RTX 4070, 12 GB VRAM, desktop RAM headroom), composing the
Phase-1-validated components, and confirm the architecture works
end-to-end when the hardware ceilings don't apply. **This is not round 6.**
It's a fresh project with a different target (working system, not gate
chase) on different hardware (GPU, not CPU-only).

**Phase 2 is NOT for:** training new models, replacing the translator,
re-running closed glossary mechanisms, or chasing latency below 3.5s.
Those are Phase 3 questions, decided after Phase 2 lands.

---

## What changes vs Phase 1

| Axis | Phase 1 (CPU) | Phase 2 (GPU) |
|---|---|---|
| Hardware | 8-core CPU, 7 GB free RAM, no GPU | RTX 4070 (12 GB VRAM), desktop RAM |
| Primary goal | Meet 3.5s/4GB gates | Working system end-to-end |
| Translator | NLLB-600M CT2-int8 (CPU) | NLLB-600M fp16 (GPU) — same model, no quantization |
| ASR | faster-whisper base+small int8, streaming | faster-whisper large-v3 fp16, streaming (GPU floor ~0.5s first-emission) |
| TTS | MeloTTS-KR (RAM-leaky on CPU) | MeloTTS-KR on GPU (RAM headroom removes the gate) |
| Eval | FLORES text + Fleurs audio, locked | SAME locked protocol — preserves comparability across phases |
| Adaptive aspect | Glossary tried 3 ways, all failed at decode time | Glossary triggers measured on domain slice — sets baseline for Phase 3 LoRA |

The locked eval protocol from Phase 1 stays locked. Same FLORES devtest,
same Fleurs paired manifest, same ML glossary slice (146 pairs), same
ko-mecab tokenization, same 13a English tokenization. The whole point of
freezing those was so numbers from any phase compare to any other.

---

## Phase 2 experiments (3 experiments, gate-driven, NOT chained)

### P2-1 — GPU pipeline composition   [PAYOFF — the working system]
Wire the Phase-1 components onto GPU and measure the full pipeline
end-to-end on the locked Fleurs paired manifest.

- Branch: `phase2/gpu-pipeline`
- Stack:
  - ASR: faster-whisper-large-v3 fp16 on CUDA, LocalAgreement-2 streaming
  - MT: NLLB-200-distilled-600M, transformers fp16 on CUDA (NOT CT2 — see
    P2-3 rationale)
  - TTS: MeloTTS-KR on CUDA
- Eval: full Fleurs paired manifest (N=270 each direction, NOT a 50-sample
  smoke — Phase 1 lesson: N≥20 verdicts, full N when N≤300)
- Pass gates:
  - en↔ko BLEU ≥ Phase 1's wired-pipeline numbers (R4-1: 24.03 / 19.39)
  - first-emission latency ≤ 1.5s (GPU should crush the 3.5s CPU floor)
  - e2e latency ≤ 2.5s (with the final ASR pass included honestly, per the
    R4-1 lesson — latency and quality must come from the same path)
  - peak VRAM ≤ 8 GB (leaves headroom for P2-3 glossary loading)
- Falsifies: any-direction BLEU regresses >1 vs Phase 1; e2e latency >3s;
  VRAM >10 GB (would block Phase 3 LoRA loading)
- Why first: this IS the working system. Everything else is incremental.

### P2-2 — Domain-slice quality baseline   [SETS PHASE 3 TARGET]
Phase 1 had no working measurement of glossary impact on a domain test
set — FLORES only triggered the glossary 2/300 times, so the "adaptive"
aspect was never actually measured. With GPU headroom we can finally do
this honestly.

- Branch: `phase2/domain-baseline`
- Stack: P2-1's pipeline, no glossary changes — measure NLLB's *native*
  terminology behavior on the ML glossary slice (146 pairs, 100% trigger).
- Eval: the locked ML glossary slice, both directions; report:
  - BLEU (ko-mecab / 13a)
  - Term recall (how often does NLLB get the glossary target right on its own?)
  - Per-segment dump of triggered-term outcomes
- This is NOT an experiment with a pass/fail gate. It's a baseline
  measurement that defines what Phase 3 LoRA would need to beat.
- Why second: until we know NLLB's native term recall on this slice, we
  can't say what "adaptive" needs to add. Phase 1's R3-3 measured it at
  ~5-10% control recall on a 60-sample subset; we need the real number on
  the full slice.

### P2-3 — Quantization quality cost (optional, drop if budget tight)
The eventual goal is "back to CPU once the architecture is proven." This
experiment measures what quantization actually costs on GPU output, so
when we re-quantize for CPU we know the expected delta.

- Branch: `phase2/quantization-deltas`
- Stack: P2-1's GPU pipeline, but swap the translator between three
  configurations: fp16 (P2-1 baseline), int8 (CT2 GPU int8), int4 (if
  available). Same ASR, same TTS.
- Eval: locked FLORES devtest (text-only — isolates the translator effect),
  both directions, both gates ignored (this is a quality-cost measurement,
  not a deployment gate).
- Report: BLEU delta vs fp16 for each quantization level.
- Use: tells us in advance whether CPU int8 will retain quality (Phase 1's
  A1 said yes, but on a 600M model — this confirms it under GPU-comparable
  conditions and serves as the methodology for Phase 3 quantization.)
- Drop priority: this is the first thing to cut if the window tightens.
  P2-1 is non-negotiable; P2-2 sets the next phase; P2-3 is nice-to-have.

---

## Termination criteria

Stop the loop when ANY of these is true:
- **Success:** P2-1 passes (working voice-to-voice system on GPU), P2-2's
  baseline is measured. P2-3 is bonus. Phase closes; Phase 3 decision
  (LoRA for terminology, yes/no/different mechanism) becomes the next
  planning question.
- **Failure:** P2-1 fails — GPU composition doesn't beat Phase 1 numbers
  or VRAM is unmanageable. This would be a strong surprise and a real
  finding; report it and stop.
- **Budget:** 5-hour window exhausted, as always. Phase 1 lesson: stop
  cleanly, don't extend into a third session wall.

## Process rules (inherited from Phase 1, carry forward)

1. **Serial runs.** One runner at a time. Phase 1 proved three-parallel
   crashes hardware; on GPU this is less of a concern but the discipline
   stands — staged debugging is faster than parallel debugging when each
   experiment is novel.
2. **Verdict discipline.** No verdict before N≥20 segments. Per-component
   numbers must reconcile to the headline. Never silence warnings to
   suppress errors. (Already in `.claude/agents/experiment-runner.md`
   from commit 950808f.)
3. **Reviewer-mandate.** Any experiment whose gate is corpus-level but
   whose failure mode is localized linguistic damage MUST have a
   reviewer-led targeted check. (P2-2 might trigger this — if NLLB's
   native term outputs look surprisingly good or bad, reviewer spot-checks
   the Korean particle agreement.)
4. **Plans-in-git, not chat.** This file IS the plan. Update it in-repo,
   don't carry forward in chat alone.
5. **`git pushmain`.** Standard going forward per `.claude/PUSH_WORKFLOW.md`.
6. **Grep-able headline numbers.** Every number in a verdict must appear
   in a committed file. (Commit 950808f rule.)

## Hardware/environment preflight

Before spawning P2-1, the manager must verify:
- CUDA available, RTX 4070 detected
- PyTorch CUDA build installed (the CPU `torch` wheel from Phase 1
  won't work — needs replacement)
- faster-whisper, transformers, MeloTTS load on CUDA without falling back
  to CPU silently (Phase 1 R3-2 lesson: silent fallback is a failure mode)
- Available VRAM ≥ 10 GB at idle (leaves room for all three model loads)

These checks are inline manager work, not a spawned experiment. Stop
and report if any fail before spawning.

---

## What this Phase is NOT

Worth stating explicitly to keep scope tight:

- **Not a CPU revisit.** CPU-only deployment is parked. If Phase 2 lands
  and Phase 3 LoRA lands, we revisit CPU as a Phase 4 question with a
  proven system to compress.
- **Not a translator retraining.** NLLB-600M is the translator. Period.
- **Not a glossary re-run.** Decode-time enforcement is closed. P2-2
  measures the *baseline* the future LoRA approach would need to beat;
  it does not re-attempt the closed mechanisms.
- **Not a new architecture.** Cascaded ASR→MT→TTS, same as Phase 1.
  Direct speech-to-speech was deferred at the round-1 design-space
  survey; reopening it is a separate, larger decision.

---

## Kickoff prompt (paste into a fresh manager session)

> Act as research-manager. Read README.md (now the Phase-1 postmortem)
> and `.claude/phase2-plan.md` for the new project scope. Phase 1 is
> closed; Phase 2 is a fresh project with a different goal and
> different hardware (RTX 4070, 12 GB VRAM).
>
> Enter plan mode. First, do the hardware/environment preflight from
> phase2-plan.md (CUDA available, PyTorch CUDA build, no silent CPU
> fallbacks, ≥10 GB VRAM free). Report results.
>
> Then confirm or re-sequence the three experiments (P2-1 GPU pipeline,
> P2-2 domain baseline, P2-3 quantization deltas), with P2-3 as the
> drop-first item if budget tightens. Tell me if you'd re-sequence
> given GPU dependencies you find on preflight.
>
> Stop and wait for my sign-off before spawning any experiment-runner.
> Same operational rules as Phase 1: serial runs, verdict discipline,
> grep-able numbers, `git pushmain` for any push.