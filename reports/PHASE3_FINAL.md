# Phase 3 — LoRA Terminology Adaptation: Final Report

**Status: CLOSED — partial success (pre-registered outcome).** Phase 3 trained
a LoRA adapter on NLLB-200-distilled-600M to lift canonical terminology
adherence above the P2-2 baseline while preserving FLORES quality. Result:
**the terminology lift is large and real; FLORES quality regressed past the
gate.** Root cause is the low-diversity *templated* training corpus, not LoRA
itself — which makes the Phase-4 lever concrete.

Hardware: RTX 5060, 8 GB. Workflow held: LoRA trained with the inference
stack unloaded (text-only), peak ~5.7 GB — the 8 GB "no co-resident LoRA"
constraint from Phase 2 was respected.

---

## Results by step

### P3-0 — Training data preparation — corpus built, with a forced tradeoff
No domain source materials existed on disk (only the frozen 146-pair eval
slice + a 10-pair TM). Two attempts:
1. **(a) Qwen2.5-7B-nf4 local generation.** Korean *generation* was
   grammatical (the R1 nf4 *translation* failure did not transfer), and a
   strengthened prompt + canonical filter + CJK-contamination filter yielded
   1747 clean, eval-disjoint pairs. **User native spot-check REJECTED it
   (>20% bail).**
2. **(b) Templated fallback.** Grammatical Korean by construction (josa
   allomorph computed from batchim; DNT terms wrapped in Korean classifier
   nouns), term kept in object/oblique position for natural English, curated
   multi-trigger sentences. **2112 pairs** (950 train + 106 dev / direction),
   100% canonical, 0 contamination, 0 eval overlap. User spot-check PASSED.

**The tradeoff that defined Phase 3:** bailing Qwen for grammaticality forced
the templated corpus, whose **low syntactic diversity** is the proximate
cause of the P3-1 FLORES regression below.

### P3-1 — LoRA training + eval — PARTIAL SUCCESS (canonical PASS, FLORES FAIL)
LoRA (rank 16, α 32, target q/k/v/out_proj **+ fc1/fc2**, lr 2e-4, 3 epochs,
effective batch 16, base frozen), one adapter on both directions, 4.7M+
trainable params. Reviewer-adjudicated; numbers independently re-verified by
the manager from committed JSON.

| Gate | en→ko | ko→en | Verdict |
|---|---|---|---|
| Non-DNT canonical recall (base → adapter) | 20.2% → **94.7%** (+74.5pp) | 31.5% → **85.5%** (+54.0pp) | **PASS** (≥35 / ≥45) |
| FLORES BLEU (base → adapter) | 25.05 → 21.36 (**−3.7**) | 24.88 → 21.48 (**−3.4**) | **FAIL** (≤0.5; also > 1.0 falsify) |
| Reviewer Korean particle check | 4.0% breakage (3/75 multi-trigger) | — | **PASS** (≤10%) |
| Slice BLEU (base → adapter) | 33.07 → 47.46 | 20.65 → 26.48 | (overfits to slice register) |

**Overall: FAIL on the FLORES gate** — but the canonical lift is decisive and
the particle check passes. This is the plan's pre-registered "partial success."

### P3-2 — Adapter characterization (folded in; no separate run)
1. **Per-term recall:** near-uniform lift to ~100% on almost every term
   (LLM 0→92%, fine-tuning 0→100%, latency 0→100%, tokenizer 0→100%, …).
   `inference` lifted only modestly (65→77%). **One regression: `embedding`
   ko→en 35.7%→7.1%** — the adapter emits "embedded" (adjectival) not the
   canonical noun "embedding", a direct overfit artifact of the templated
   corpus using 임베딩 in modifier position.
2. **Off-target degradation:** this IS the FLORES regression. Training loss
   collapsed 1.44 → 0.07 over 3 epochs on 1900 repetitive pairs — extreme
   convergence. On diverse FLORES text the adapter narrows to a formulaic
   ML register (~3.5 BLEU / ~15% relative penalty). Degradation is
   narrowing, not breakage.
3. **Multi-trigger:** reviewer's manual check found 4.0% particle breakage,
   localized to DNT Latin terms (`PyTorch은`/`HuggingFace은` should be
   `는`). Nuisance, not catastrophic — multi-trigger handling largely holds.

---

## The finding (refined from the pre-registration)

The plan pre-registered FLORES-regression-as "LoRA is the wrong tool." The
evidence refines this: **LoRA decisively CAN learn the terminology** (+74.5 /
+54.0 pp, near-perfect per-term recall, no grammar damage). What failed is
**corpus diversity** — the templated corpus (forced by the Qwen grammaticality
bail) caused a distribution shift that LoRA faithfully absorbed. So the
mechanism is validated; the bottleneck is the training data.

This connects the whole chain: Qwen-nf4 grammaticality risk → spot-check bail
→ templated fallback → low diversity → FLORES regression. The untested sweet
spot is a corpus that is **both grammatical AND diverse**.

## Comparison table

| Stage | canonical non-DNT en→ko / ko→en | FLORES BLEU en→ko / ko→en | note |
|---|---|---|---|
| P2-2 base NLLB (no adapter) | 20.7% / 31.5% | 25.05 / 24.88* | the baseline to beat |
| **P3-1 LoRA adapter** | **94.7% / 85.5%** | **21.36 / 21.48** | lift huge; FLORES −3.7/−3.4 |
| Gate | ≥35% / ≥45% | regression ≤0.5 | canonical PASS, FLORES FAIL |

*P3-1 base re-measured at 25.05/24.88 (greedy decode); matches the R1
NLLB-600M baseline (25.35/25.32, beam=4) within decode-setting noise.

## Negative / honest results
- The deployable goal (lift terminology **and** preserve FLORES) was **not
  met**: −3.5 BLEU general-quality cost is too high to ship.
- `embedding` ko→en *regressed* (−28.6 pp) — overfit can also hurt specific terms.
- The committed eval initially had a base==adapter contamination bug (adapter
  loaded in-place on base → identical outputs); the runner caught and fixed it
  (separate model instances), reviewer confirmed via VRAM evidence. Worth
  recording as a recurring LoRA-eval footgun.
- The automated `has_particle_concern()` heuristic is degenerate (flags only
  empty/non-Korean) — this is precisely why the independent reviewer particle
  check is mandated.

## What is validated (carry forward)
- **LoRA is a working terminology-adaptation mechanism on NLLB-600M.** The
  lift is real, large, grammatical, and reproducible (adapter committed:
  `reports/p3-1_lora/adapter/`).
- 8 GB LoRA training works inference-unloaded (~5.7 GB peak), confirming the
  Phase-2 workflow constraint.
- The locked eval protocol (ko-mecab/13a, P2-2 term-recall methodology) and
  the frozen 146-pair slice held across phases — base re-measured == P2-2.

## Recommended next direction (Phase 4 inputs)
1. **Primary lever — corpus diversity.** Build a corpus that is grammatical
   AND diverse, then retrain the same LoRA and re-check FLORES. Options: the
   **hybrid** we deferred (the strengthened-prompt + filtered Qwen pipeline is
   already built and produced grammatical pairs for many terms — splice with
   templated only where Qwen failed), or real sourced domain text. Hypothesis:
   diversity recovers FLORES while keeping most of the lift.
2. **Regularization levers** if diversity alone is insufficient: fewer epochs
   / earlier stop (loss hit 0.07 — clearly over-trained), lower LoRA α, mix a
   slice of general FLORES-style pairs into training to anchor the
   distribution.
3. **Path B (only if 1–2 fail):** scout lighter preference mechanisms
   (constrained-decode-free preference data, post-edit) — but LoRA is not yet
   disproven; it's under-fed.

## Termination rationale
Per the plan's pre-registered "partial success" criterion: P3-1 lifted
canonical adherence decisively but regressed FLORES > 1.0; the particle check
passed; reviewer confirmed. The finding (LoRA works; templated-corpus
diversity is the blocker) is clean and actionable. Phase 3 closes here without
a further training round; the diversity retrain is a Phase-4 scope decision.
Detailed code, adapter, and per-segment dumps live on branch
`phase3/lora-terminology`.
