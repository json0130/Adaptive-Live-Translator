# Phase 2 — GPU Validation (working voice-to-voice translator)

**Status of Phase 1:** closed as a research artifact (rounds 1-5, see
README postmortem). Five rounds proved the translator/context axis is
saturated and CPU-only deployment is gated by hardware ceilings
(Whisper ~3.3s first-emission, MeloTTS RAM growth to ~6.6 GB). Closed
clean. Not reopening any of those axes.

**Phase 2 goal:** stand up a working voice-to-voice en↔ko translator on
GPU (RTX 5060, 8 GB VRAM, desktop RAM headroom), composing the
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
| Hardware | 8-core CPU, 7 GB free RAM, no GPU | RTX 5060 (8 GB VRAM, ~7 GB free), desktop RAM |
| Primary goal | Meet 3.5s/4GB gates | Working system end-to-end |
| Translator | NLLB-600M CT2-int8 (CPU) | NLLB-600M fp16 (GPU) — same model, no quantization |
| ASR | faster-whisper base+small int8, streaming | faster-whisper large-v3 `int8_float16`, streaming (fp16 OOMs the 8 GB card co-resident — see P2-1) |
| TTS | MeloTTS-KR (RAM-leaky on CPU) | MeloTTS-KR on GPU (RAM headroom removes the gate) |
| Eval | FLORES text + Fleurs audio, locked | SAME locked protocol — preserves comparability across phases |
| Adaptive aspect | Glossary tried 3 ways, all failed at decode time | Glossary triggers measured on domain slice — sets baseline for Phase 3 LoRA |

The locked eval protocol from Phase 1 stays locked. Same FLORES devtest,
same Fleurs paired manifest, same ML glossary slice (146 pairs), same
ko-mecab tokenization, same 13a English tokenization. The whole point of
freezing those was so numbers from any phase compare to any other.

---

## Phase 2 experiments (1 gated env step + 3 experiments, gate-driven, NOT chained)

### P2-0 — Environment rebuild   [GATED STEP — must pass before P2-1]
Not a spawned experiment; manager-run env work. But it is its own **gated
step**, not "inline prep": the cu128 torch swap on Blackwell (sm_120) plus
the MeloTTS-from-source install (Phase 1 R3-2 was a ~40-min hunt) is real
risk and must clear an explicit exit criterion before P2-1 is spawned.

- Tasks:
  - Replace the Phase-1 CPU torch wheel (`2.11.0+cpu`) with a CUDA build
    carrying **sm_120 (Blackwell)** kernels — cu128 or newer; torchaudio
    must match.
  - Reinstall **MeloTTS from source** (`myshell-ai/MeloTTS`, MIT). Two
    known hazards, both checked AFTER install: (a) its deps can clobber the
    freshly-installed CUDA torch; (b) it can upgrade `mecab_ko_dic` and
    silently break the locked ko-mecab BLEU tokenizer — the exact incident
    recorded in `src/utils/metrics.py`.
- **Exit criterion (translator-only CUDA smoke-load):** NLLB-600M fp16
  (transformers, CUDA), the CT2-int8 translator (CUDA), and
  faster-whisper-large-v3 (CUDA, float16) all load and run on the GPU with
  **no silent CPU fallback**; `torch.cuda.is_available()` is True with
  sm_120 in the arch list; and the locked ko-mecab tokenizer still segments
  (no char-BLEU fallback). MeloTTS load is attempted and reported but is
  **NOT** a blocker: if MeloTTS-on-Blackwell fails, P2-0 still passes and
  P2-3 (translator-only, text) stays a valid fallback experiment while the
  TTS issue is worked separately.
- Falsifies P2-0: torch CUDA won't init on sm_120; faster-whisper or the
  translator silently fall back to CPU; ko-mecab tokenizer breaks.
- Manager runs P2-0, reports, and waits for sign-off before P2-1.

**P2-0 results (2026-06-09) — PASS.** Manager-run; grep-able numbers
(scripts: `.claude/p2_0_smoke.py`, `.claude/p2_0_final.py`; freezes:
`.claude/p2-0_pip_freeze_{before,after}.txt`):
- torch swap: `2.11.0+cpu` -> `2.11.0+cu128`; `cuda.is_available()=True`;
  `arch_list` includes `sm_120`; fp16 matmul executes on GPU (127 ms).
- Exit criterion (translator-only CUDA smoke-load): **PASS**. NLLB-600M fp16
  (transformers, `cuda:0`), CT2-int8 (`device=cuda`, `int8_float16`), and
  faster-whisper-large-v3 (`cuda`, `float16`) all load + translate/transcribe
  both directions with **no silent CPU fallback**; ASR `lang_prob=1.00` on
  en + ko Fleurs samples.
- ko-mecab eval tokenizer: **RESTORED**. `compute_bleu(..., tokenize="ko-mecab")`
  returns `tokenize_used=ko-mecab` (was MeCab-absent -> char fallback). Fix
  in `src/utils/metrics.py`: use `mecab_ko_dic.MECAB_ARGS` (bundles
  `-r mecabrc -d dicdir`) because MeloTTS's unidic install hijacks MeCab's
  default dict; behaviour-preserving (same morpheme segmentation).
- MeloTTS on Blackwell: **PASS** (bonus; was non-blocking). Loads
  `device=cuda`; Korean synth ok (190,621 samples @ 44.1 kHz). Required
  `python -m unidic download` (melo/text/japanese.py inits a tagger at
  import) + first-run download of `kykim/bert-kor-base`.
- ENV CASUALTY (recorded): MeloTTS hard-pins `transformers==4.27.4`, so the
  install downgraded transformers `5.5.4 -> 4.27.4` (+ tokenizers `0.13.3`,
  librosa `0.9.1`, numpy `1.26.4`). **torch cu128 survived.** Broke
  flagembedding / sentence-transformers (RAG-only — not used in Phase 2).
  NLLB fp16, CT2, faster-whisper, sacrebleu all re-verified on the
  downgraded env.

**P2-1 VRAM RISK (key finding feeding P2-1).** Full P2-1 stack co-resident
(ASR large-v3 fp16 + NLLB fp16 + MeloTTS) static-load peak = **7.16 GB
used** of 8.07 GB total (~1.12 GB desktop baseline -> ~6.04 GB process).
Breakdown: ASR ~4.07 GB, +NLLB ~1.14 GB, +MeloTTS ~0.83 GB. This is the
*load* peak; inference activations add on top, so the re-baselined ≤7 GB
gate is **tight** and OOM is a live risk under load. P2-1 likely needs ASR
`compute_type=int8_float16` (~2 GB vs ~4 GB) or sequential loading to hold
the gate — call this out in the P2-1 spec.

### P2-1 — GPU pipeline composition   [PAYOFF — the working system]
Wire the Phase-1 components onto GPU and measure the full pipeline
end-to-end on the locked Fleurs paired manifest.

- Branch: `phase2/gpu-pipeline`
- Stack:
  - ASR: faster-whisper-large-v3 **`int8_float16`** on CUDA, LocalAgreement-2
    streaming. **Changed from fp16 after P2-0** (signed off 2026-06-09): the
    full-stack fp16 static-load peak was 7.16 GB of 8.07 GB — with inference
    activations stacking, fp16 is *expected to OOM mid-run*, not merely
    "tight". int8 ASR frees ~2 GB. Quality basis: Phase-1 small-int8 ASR ran
    11.83% ko / 8.21% en WER; large-v3-int8 should beat that even quantized.
    Architecture validation (P2-1's actual purpose) is unchanged by the
    precision choice.
  - MT: NLLB-200-distilled-600M, transformers fp16 on CUDA (NOT CT2 — see
    P2-3 rationale)
  - TTS: MeloTTS-KR on CUDA
- Eval: full Fleurs paired manifest (N=270 each direction, NOT a 50-sample
  smoke — Phase 1 lesson: N≥20 verdicts, full N when N≤300)
- Required report fields (signed off 2026-06-09):
  - **Log the chosen ASR config explicitly** (model, device, compute_type).
  - **Measure ACTUAL peak VRAM during a representative inference segment**
    (e.g. `torch.cuda.max_memory_allocated` + `nvidia-smi` sampling), NOT
    just the static-load number. This is the Phase-3 input: does 8 GB ever
    hold inference + LoRA training co-resident, or must Phase 3
    unload-and-reload?
- Pass gates:
  - en↔ko BLEU ≥ Phase 1's wired-pipeline numbers (R4-1: 24.03 / 19.39)
  - first-emission latency ≤ 1.5s (GPU should crush the 3.5s CPU floor)
  - e2e latency ≤ 2.5s (with the final ASR pass included honestly, per the
    R4-1 lesson — latency and quality must come from the same path)
  - peak VRAM ≤ 7 GB (re-baselined for the 8 GB card — must fit the ~7 GB
    free at idle; the old ≤8 GB / Phase-3-LoRA-headroom rationale is void,
    see "What this Phase is NOT")
- Falsifies: any-direction BLEU regresses >1 vs Phase 1; e2e latency >3s;
  peak VRAM exceeds the ~7 GB free budget (OOM on the 8 GB card)
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

## Preflight results & re-baselined gates (2026-06-08, lab box)

Preflight run inline by the manager. The plan's original "RTX 4070 / 12 GB"
was **phantom hardware** — corrected throughout this file. Actual GPU:
**RTX 5060, 8 GB (Blackwell sm_120)**, the same card as Phase-1 R1. Both
available machines are 8 GB (home 3060 is the backup box).

Findings:
- GPU: RTX 5060, 8.15 GB total, **~7.0 GB free at idle** (driver 580.159,
  CUDA 13.0).
- torch was `2.11.0+cpu` (Phase-1 CPU wheel) → `cuda.is_available()=False`.
  Resolved by **P2-0**.
- MeloTTS not installed (`No module named 'melo'`). Resolved by **P2-0**.
- CTranslate2 4.7.1 already detects the GPU (fp16/int8 CUDA compute types
  available) — the ASR + CT2-translator paths may run without torch.
- All locked eval data present: FLORES devtest 1012/1012, ML glossary slice
  146, Fleurs paired manifest 270. Weights on disk (whisper-large-v3 CT2,
  NLLB-600M fp16, nllb-600m-ct2-int8). No downloads needed.

Re-baselined gates (8 GB card, not the phantom 12 GB):
- **Dropped** the "≥10 GB VRAM free" preflight check — impossible on 8 GB.
- **P2-1 peak-VRAM gate: ≤ 7 GB** (was ≤ 8 GB) — must fit the ~7 GB free.
- The silent-CPU-fallback check is preserved and folded into **P2-0**'s
  exit criterion (R3-2 lesson: silent fallback is a failure mode).

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
- **Not a co-resident-LoRA target — architectural finding, not a footnote.**
  On 8 GB there is **no VRAM headroom for Phase-3 LoRA alongside the
  inference stack**: the full pipeline already wants ~the whole card
  (P2-1 gate is ≤7 GB of ~7 GB free). The original plan's assumption that
  VRAM headroom would hold LoRA was tied to the phantom 12 GB and **does
  not survive on 8 GB**. Consequence: if Phase 3 LoRA happens, training
  (and any LoRA-loaded inference) must be a **separate event with the
  inference stack fully unloaded — never co-resident**. Record this as a
  hard constraint feeding the Phase-3 LoRA yes/no/different-mechanism
  decision.

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