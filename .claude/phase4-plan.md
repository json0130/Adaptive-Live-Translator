# Phase 4 — LoRA Diversity Retrain

**Status of Phase 3:** closed as partial success. LoRA mechanism validated
(+74.5/+54.0 pp canonical recall), FLORES regressed −3.5 BLEU due to
low-diversity templated corpus overfit (train loss 1.44 → 0.07 on 1900
repetitive pairs). Reviewer particle check passed (4% breakage). One term
(`embedding` ko→en) regressed from 35.7% → 7.1%.

**Phase 4 goal:** retrain the same LoRA on a grammatical-AND-diverse corpus
with regularization, to recover FLORES while keeping the canonical lift.
Settles whether LoRA on NLLB-600M is diversity-fixable or has an inherent
diversity-vs-adherence tension.

**Pre-registered outcomes (both publishable, neither is "failure"):**
- **PASS:** FLORES regression ≤ 0.5 BLEU AND canonical recall ≥ 70%
  both directions → working adaptive translator, deployable on this axis.
  (Phase 2's batch-pipeline finding still blocks full deployment.)
- **PARTIAL:** Canonical recall holds but FLORES still regresses > 1.0 →
  LoRA on small NMT has inherent diversity-vs-adherence tension. Real finding.
- **MECHANISM-NEGATIVE:** Canonical recall drops sharply AND FLORES still
  regresses → corpus diversity was not the bottleneck; LoRA is genuinely
  the wrong tool. Cleanest finding of the three.

---

## Phase 4 experiments — 2 experiments, one window

### P4-0 — Hybrid corpus construction   [PREREQUISITE, manager-inline]

The Phase 3 report flagged the strengthened-prompt + filtered Qwen pipeline
preserved its grammatical-for-most-terms output at
`.claude/p3_0_qwen_raw_*.jsonl`. The user's v1 spot-check identified specific
terms where Qwen worked vs. where it failed.

Build the hybrid at **term granularity**, not sentence granularity:
- Qwen-good terms (from user's v1 spot-check list) → use Qwen v1 pairs
- Qwen-failed terms → use templated v2 pairs
- Multi-trigger sentences → keep curated set from v2 (only place those exist)

**If the user did not produce an explicit Qwen-good-terms list during v1
spot-check:** manager pauses and asks for it before building. Do NOT guess
which terms were Qwen-good — that's the methodology error that started
Phase 3's FLORES regression.

Exit criteria:
- Hybrid corpus committed at `data/eval/ml_glossary_train_hybrid.{en_ko,ko_en}.tsv`
- Per-term stats: pair count, source (Qwen/templated), syntactic position
  diversity (term appears in subject / object / oblique positions — track
  this, since P3-1's `embedding` regression was a position-collapse failure)
- Hash-verified zero overlap with frozen 146-pair eval slice (same rule as P3-0)
- Train/dev split 90/10 (dev for early-stop only, NEVER for final report)
- Spot-check sample (~60 pairs, stratified by term and source) for user review

User sign-off required before P4-1 spawns. Same teeth as P3-0 — bail
criterion: > 20% ungrammatical OR > 30% non-canonical.

### P4-1 — LoRA retrain with regularization   [PAYOFF]

Same LoRA architecture as P3-1 (rank 16, α 32, target q/k/v/out_proj +
fc1/fc2) — controlling the architecture variable so the result attributes
cleanly to corpus + regularization changes.

- Branch: `phase4/lora-diversity`
- Stack: NLLB-200-distilled-600M frozen base, LoRA adapter, peft 0.3.0
- **Regularization changes vs P3-1 (apply ALL — varying levers individually
  burns windows):**
  1. **Early stopping on dev FLORES BLEU**, NOT on training loss. Train loss
     hitting 0.07 was the over-train signal P3-1 missed. New rule: hold out
     a 100-pair FLORES dev sample, eval every 0.5 epoch, stop when FLORES
     dev BLEU starts dropping. Maximum 3 epochs as hard ceiling.
  2. **Anchor batch.** Mix 20% general FLORES-style training pairs into each
     batch (not glossary-triggered) to keep the model grounded in general
     distribution. Source: subsample FLORES train (NEVER devtest — devtest
     stays the locked eval).
  3. **Lower α: 16 instead of 32.** Halves LoRA's distribution-shift
     strength. Conservative move; combined with the above gives three
     independent regularization levers.
- Eval: locked 146-pair ML glossary slice (canonical recall + per-term
  breakdown) AND FLORES devtest (regression check), both directions
- Pass gates: see "pre-registered outcomes" above
- Reviewer-mandate (non-negotiable): 30-sample Korean particle check on
  multi-trigger sentences. Same as P3-1.
- **NEW: per-term regression flag.** P3-2 found `embedding` ko→en regressed
  35.7% → 7.1% — the headline average hid a specific failure. P4-1 must
  flag ANY term whose adapter recall is below its base recall, even if the
  overall average improves. A "pass" with a regressed term is NOT a pass
  without explicit acknowledgment.

P3-2-style characterization (per-term recall, off-target check,
multi-trigger) folds into P4-1's report rather than a separate run.

---

## Termination criteria
- Any of the three pre-registered outcomes (PASS / PARTIAL / MECHANISM-NEGATIVE)
- 5-hour budget exhaustion
- Corpus sign-off rejected (bail; phase splits across windows)

## Hard scope limits (carry from Phase 1 lessons)
- **NO third experiment.** P4-0 + P4-1 = Phase 4 closes. If results are
  unclear, that itself is the finding; do not extend into a P4-2 to
  "characterize further."
- **NO architectural changes.** No incremental MT, no different translator,
  no new language pairs. Those are separate phases if they happen.
- **NO retraining iterations within Phase 4.** One LoRA train, one eval.
  Multi-run variants ("try rank 32 too") = scope creep.
- **Reviewer particle check is mandatory.** Same rule as R3-3 and P3-1.

## Process rules (carry forward)
- Serial runs
- Verdict discipline (N≥20, grep-able, reconcile to summary.md)
- ru_maxrss = true peak RSS
- "Valid output" means N≥5 manually-checked grammatically-plausible
  sentences, not just non-degenerate tokens (R4-4 lesson)
- Plans-in-git
- `git pushmain` for main; explicit `git push origin <branch>` for branches

## Preflight (manager runs before P4-0)
- peft 0.3.0 still installed and working (P3-0 verified this)
- The preserved Qwen raw at `.claude/p3_0_qwen_raw_*.jsonl` is accessible
- User's v1 Qwen-good-terms list available, or user prompted for it
- VRAM headroom (same as Phase 3, since architecture unchanged)

---

## What this phase is NOT

- NOT a third corpus-building attempt if the hybrid spot-check fails (bail = phase split, not pivot)
- NOT a LoRA architecture search (rank, target_modules locked at P3-1's config)
- NOT addressing the Phase 2 batch-pipeline finding (orthogonal; separate phase)
- NOT a CPU revisit
- NOT additional language pairs

---

## Kickoff prompt

> Act as research-manager. Read reports/PHASE3_FINAL.md and
> .claude/phase4-plan.md. Phase 3 closed with LoRA mechanism validated and
> FLORES regressed due to templated corpus overfit. Phase 4 is a diversity
> retrain with regularization to recover FLORES while keeping the canonical
> lift.
>
> Enter plan mode. Verify preflight: peft 0.3.0 working, preserved Qwen raw
> accessible at .claude/p3_0_qwen_raw_*.jsonl, ask me for my v1 Qwen-good-terms
> list before building the hybrid (do NOT guess which terms held — that's the
> methodology error that started Phase 3's regression).
>
> Then confirm or re-sequence the two items (P4-0 hybrid corpus, P4-1 retrain).
> Flag any issue with the three regularization levers (early-stop on FLORES dev
> not train loss, 20% FLORES-anchor batch, α 16 instead of 32). The per-term
> regression flag in P4-1's eval is mandatory — a passing average that hides a
> specific term regression is NOT a pass.
>
> Stop and wait for my sign-off after the corpus checkpoint, same teeth as
> Phase 3 — non-ceremonial bail criterion.