# Phase 3 — LoRA Terminology Adaptation

**Status of Phase 2:** closed. GPU architecture validates end-to-end (BLEU
23.63/19.49, VRAM 5.02 GB on 8 GB card), but the stack is "batch pipeline
in streaming clothes" — MT is non-incremental, so honest e2e is 4-6s vs
the 2.5s gate. Terminology baseline established: NLLB's native
canonical-adherence is 20.7% en→ko / 31.5% ko→en on the locked ML
glossary slice (146 pairs). Phase 1's decode-time glossary enforcement
is closed (3 mechanisms, 3 failures).

**Phase 3 goal:** train a LoRA adapter on NLLB-200-distilled-600M that
raises canonical-adherence on the locked ML glossary slice above the
P2-2 baseline, while preserving general translation quality (no
regression on FLORES devtest beyond −0.5 BLEU).

**Path A commitment:** LoRA is the chosen mechanism. Phase 3 does NOT
scout alternative mechanisms (preference fine-tuning, RAG, post-edit) —
those are a future Phase 4 comparison if Phase 3 results warrant.

---

## Why LoRA, given Phase 2's "preference not accuracy" finding

P2-2 framed the problem as canonical-adherence (model produces valid
alternatives, you want the canonical one). LoRA is built for accuracy
problems, so we are deliberately stretching the tool. Phase 3 measures
whether that stretch is worthwhile.

Concrete risk being accepted: LoRA may improve canonical-adherence on
the slice but degrade FLORES BLEU because it shifts the underlying
distribution. The success criteria below explicitly track BOTH the
slice AND FLORES to detect this tradeoff.

If Phase 3 finds LoRA improves the slice AND preserves FLORES, the
project has a working adaptive translator. If it improves the slice
but tanks FLORES, that's a finding that LoRA is the wrong tool for
this problem (informing Phase 4's mechanism choice). Both outcomes are
useful.

---

## Hardware constraint — 8 GB workflow

Phase 2 confirmed: 8 GB has NO co-resident-LoRA-training-and-inference
headroom. Phase 3 workflow must be:

1. Train LoRA adapter with inference stack UNLOADED (only base model +
   LoRA layers + optimizer state in VRAM)
2. Save adapter weights to disk
3. Load adapter into the existing P2-1 inference pipeline
4. Run eval (slice + FLORES) with adapter applied

This is operationally heavier than Phase 2's spawn-and-run pattern.
Each iteration is: train → save → swap → eval → analyze. No co-resident
training-and-measurement.

VRAM estimate for training-only:
- NLLB-600M fp16 base weights: ~1.2 GB
- LoRA adapter weights (rank 16, target attention layers): ~10 MB
- Optimizer state (AdamW fp32, 2× LoRA params): ~80 MB
- Activations + gradients (batch 4, seq 128): ~3-4 GB
- Estimated peak: ~5-6 GB → comfortable on 8 GB

If batch size or sequence length push this over 7 GB, gradient
accumulation is the lever (smaller per-step batch, accumulate gradients,
same effective batch).

---

## Phase 3 experiments — 3 experiments, gate-driven

### P3-0 — Training data preparation   [PREREQUISITE, manager-inline]
LoRA needs training data, not just an eval slice. The locked ML
glossary slice (146 pairs) is too small to train on — and using it
for training would invalidate it as an eval set.

Manager builds a separate training corpus from the same domain sources
(arXiv abstracts, HF blog, PyTorch docs) referenced in Phase 1's slice
construction. Target: 1500-3000 sentence pairs containing glossary
terms, with the canonical target translation. NEVER overlaps with the
locked 146-pair eval slice (the eval slice stays frozen, untouched).

Exit criteria:
- Training corpus committed at `data/eval/ml_glossary_train.{en_ko,ko_en}.tsv`
- Per-pair glossary trigger count documented
- Zero overlap with eval slice (verified by hash diff)
- Train/dev split: 90/10 (dev set is for early-stopping decisions only,
  NEVER for final reporting — that's the eval slice)

Manager inline; no spawn. Drop-blocker: without P3-0, P3-1 cannot run.

### P3-1 — LoRA training and adapter creation   [PAYOFF]
Train a LoRA adapter on NLLB-600M using the P3-0 training corpus.

- Branch: `phase3/lora-terminology`
- Stack:
  - Base: NLLB-200-distilled-600M (HF transformers, fp16, frozen)
  - LoRA: peft library, rank 16, alpha 32, target_modules=["q_proj",
    "k_proj", "v_proj", "out_proj"] on all encoder + decoder attention
    layers
  - Optimizer: AdamW, lr 2e-4 (LoRA-typical), warmup 100 steps
  - Train: ~3 epochs on the training corpus, batch 4 (gradient
    accumulation to effective 16 if needed)
  - Inference stack: UNLOADED during training
- Eval: on the locked ML glossary slice (146 pairs), both directions,
  AND on FLORES devtest (regression check), both directions
- Pass gates:
  - Canonical-adherence en→ko ≥ 35% (P2-2 baseline 20.7%, +14 pp target)
  - Canonical-adherence ko→en ≥ 45% (P2-2 baseline 31.5%, +14 pp target)
  - FLORES BLEU regression ≤ 0.5 in either direction
  - Reviewer 30-sample Korean particle check: no treatment-specific
    breakage > 10% (carry-forward from R3-3 lesson)
- Falsifies: either canonical-adherence lift < 5 pp; OR FLORES BLEU
  drops > 1.0; OR particle breakage > 15%
- Reviewer-mandate (non-negotiable, same as R3-3): a corpus-level pass
  can hide localized grammar damage. Reviewer does the particle check
  independently before the verdict is trusted.

### P3-2 — Adapter behavior characterization   [BANKS THE FINDING]
P3-1 produces a pass/fail/regression result. P3-2 characterizes WHY,
which is what makes the result useful (or, on failure, useful anyway).

- Branch: `phase3/lora-analysis` (or fold onto P3-1's branch)
- Three measurements on the trained adapter, all on the locked eval
  slice:
  1. Per-term recall — which glossary entries does the adapter actually
     learn? Are improvements concentrated on common terms, or uniform?
  2. Off-target degradation — for sentences with NO glossary triggers,
     does the adapter change output? (If yes, it's leaking; if no, it's
     well-bounded.)
  3. Multi-trigger sentences — R3-3 showed that 2-3 triggers per
     sentence caused particle breakage. Does the LoRA adapter handle
     these correctly, or repeat the failure mode at a different layer?

- This is a measurement experiment, not a gate experiment. Reports
  findings; no pass/fail.
- Manager inline; no spawn (single-model eval on small slice).

---

## Termination criteria

Stop when ANY of these is true:
- **Success:** P3-1 passes both gates + reviewer particle check; P3-2
  characterizes adapter behavior. Phase 3 closes with a working
  adaptive translator and a defensible mechanism choice.
- **Partial success:** P3-1 lifts canonical-adherence but regresses
  FLORES > 1.0 → finding is "LoRA is wrong tool for preference
  problems on NLLB-600M, but the lift is real." This is itself
  publishable; informs Path B (lighter mechanism scout) for Phase 4.
- **Failure:** P3-1 lifts < 5 pp → LoRA didn't take. Possible causes:
  training corpus too small, rank too low, wrong target modules.
  Documented as "LoRA with [config] does not move this baseline."
- **Budget:** 5-hour window exhausted, as always. Phase 1 lesson holds.

## Process rules (carry forward)
- Serial runs (no parallel)
- Verdict discipline: N≥20, grep-able headlines, reconcile to summary.md
- Reviewer-mandate on the particle check, NON-NEGOTIABLE
- Plans in git, not chat
- `git pushmain` for any push; explicit `git push origin <branch>` for
  branch pushes
- `ru_maxrss` = true peak RSS, not allocator counter (P2-1 lesson)
- "Valid Korean output" means grammatically plausible on ≥5 manually
  checked sentences, not just non-degenerate tokens (R4-4 lesson)

## Preflight (manager runs before P3-0)
- `peft` library installed in cu128 venv
- Training corpus source materials available (arXiv abstracts, etc.)
- Disk space for adapter checkpoints + training data
- VRAM headroom verified by loading NLLB-600M + simulated LoRA layers
  + dummy optimizer state (don't train, just verify the load fits)

---

## What this phase is NOT

- NOT scouting alternative adaptation mechanisms (deferred to Phase 4
  per Path A commitment)
- NOT solving the batch-pipeline-in-streaming-clothes architectural
  finding (that's Phase 4 / Question 1 territory; orthogonal)
- NOT a CPU revisit
- NOT a new translator (NLLB-600M is committed)
- NOT additional language pairs (en↔ko only)

---

## Kickoff prompt

> Act as research-manager. Read reports/PHASE2_FINAL.md and
> .claude/phase3-plan.md. Phase 2 closed with the architecture
> validated, the terminology baseline measured (20.7%/31.5%), and the
> batch-pipeline finding documented. Phase 3 trains a LoRA adapter on
> NLLB-600M to lift canonical-adherence above the P2-2 baseline while
> preserving FLORES quality.
>
> Enter plan mode. First, run the preflight: peft installed in cu128
> venv, training corpus source materials available, VRAM headroom for
> training (NLLB-600M base + LoRA + optimizer state) fits 8 GB.
>
> Then confirm or re-sequence the three items (P3-0 training data prep
> inline, P3-1 LoRA training spawn, P3-2 characterization inline). Flag
> any issues with the proposed LoRA config (rank 16, alpha 32, attention
> target modules, lr 2e-4, 3 epochs).
>
> The reviewer particle check on P3-1 is mandatory and non-negotiable
> — same as R3-3. A corpus-level pass without that check is not a pass.
>
> Stop and wait for my sign-off before spawning P3-1. P3-0 and P3-2 are
> inline manager work; only P3-1 needs a runner spawn.