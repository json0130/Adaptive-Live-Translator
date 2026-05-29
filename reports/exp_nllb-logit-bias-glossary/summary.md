# R3-3 — Constrained-decoding glossary (soft logit-bias) — summary

## Goal
Round 1 proved prompt-injection RAG breaks NLLB (translates the examples).
Terminology adaptation for NMT must act on the OUTPUT distribution. This
experiment tests a soft logit-bias `LogitsProcessor` on HF
`facebook/nllb-200-distilled-600M` (greedy): bias the token sequences of
triggered glossary target terms.

Success gate (locked): term-recall(treatment) − term-recall(control)
≥ +20 pp on the glossary slice, AND FLORES BLEU within −0.5 of control.

## Results

### Glossary slice (ml_glossary_slice_en_ko, N=60, en→ko)

| Condition | char-BLEU | term-recall | recall Δ vs control |
|---|---:|---:|---:|
| control (no bias) | 50.92 | 5.4% | — |
| **treatment bias=+3** | **53.84** | **26.8%** | **+21.4 pp** |
| treatment bias=+15 | 50.88 | 100.0% | +94.6 pp |

### FLORES devtest non-regression (N=300, en→ko, general news)

| Condition | char-BLEU | triggered sents |
|---|---:|---:|
| control | 36.23 | 2 / 300 |
| treatment bias=+3 | 36.20 | 2 / 300 |
| Δ | **−0.03** | — |

## Gate verdict: QUALIFIED PASS at bias=+3
- Slice recall lift **+21.4 pp** (5.4 → 26.8) clears the ≥+20 pp gate.
- FLORES BLEU **−0.03** clears the within-−0.5 non-regression bound.

## But: the implementation has a design ceiling
The `GlossaryLogitsProcessor` perpetually boosts the *first token* of every
not-yet-emitted term at *every* decode step. This maximises term presence
*anywhere* in the output rather than at the grammatically correct position:
- At +3, the bias is gentle: recall rises modestly to 26.8% (most terms
  still missed) with no grammar harm (slice BLEU even ticks up +2.9).
- At +15, recall hits 100% **but the terms get jammed at the sentence
  start, out of order**, e.g.:
    `환각 현상 파인튜닝은 영역별 데이터에 따라 ...` (both forced terms
    dumped before the actual clause)
    `벤치마크 대형 언어 모델에 대한 새로운 LLM는 ...`
  Slice BLEU falls back to control level (50.88) — the recall gain is
  cancelled by word-order destruction. There is **no bias value that
  gives high recall AND clean grammar** with this design.

Root causes:
1. Perpetual first-token boost (no position awareness).
2. Subword-boundary mismatch: the term's standalone tokenization
   (`encode(term)`) differs from how the model emits it mid-sentence
   (leading-space `▁` variants), so the biased token IDs often don't line
   up with the in-context emission.

## Verdict
Soft logit-bias **technically clears the round-3 gate at +3** (recall
+21.4 pp, no FLORES regression), confirming the round-1 lesson that
output-side biasing — unlike input prompt-injection — does NOT break
NLLB. But it is a weak, low-ceiling mechanism: you cannot push recall
high without grammar collapse. Proper position-aware lexically-constrained
decoding (Hokamp & Liu grid beam search, or a trie that activates only
when the source-term's aligned position is being decoded) is the
round-4 path to high recall without front-loading damage.

## Korean particle spot-check: PENDING REVIEWER
Per protocol, the 30-sample Korean particle-agreement spot-check on the
treatment outputs is the REVIEWER's job, not self-certified here. The
verbatim +3 treatment ko outputs are saved in:
`reports/exp_nllb-logit-bias-glossary/slice_treatment_b3.json` (all_hyps /
all_results). The +15 outputs (slice_treatment_b15.json) are also saved
to illustrate the front-loading failure mode.

## Tokenizer caveat
eval_glossary.py reports char-BLEU, not the protocol's ko-mecab: the
installed `mecab_ko_dic` is missing the `MECAB_ARGS` attribute that
sacrebleu's TokenizerKoMecab expects in this environment, so it fell back
to char. All BLEU DELTAS here are valid (control and treatment use the
identical char tokenizer), but the absolute char-BLEU values are NOT
comparable to A1's ko-mecab 25.24 — char inflates Korean BLEU. Fixing the
mecab_ko_dic version is a harness follow-up.

## Reproduce
```
git checkout exp/nllb-logit-bias-glossary
# slice, three conditions:
PYTHONPATH=. python3 scripts/eval_glossary.py --testset data/eval/ml_glossary_slice_en_ko.tsv \
  --src en --tgt ko --mode control   --report reports/exp_nllb-logit-bias-glossary/slice_control_b0.json --limit 60
PYTHONPATH=. python3 scripts/eval_glossary.py --testset data/eval/ml_glossary_slice_en_ko.tsv \
  --src en --tgt ko --mode treatment --bias 3  --report reports/exp_nllb-logit-bias-glossary/slice_treatment_b3.json --limit 60
PYTHONPATH=. python3 scripts/eval_glossary.py --testset data/eval/ml_glossary_slice_en_ko.tsv \
  --src en --tgt ko --mode treatment --bias 15 --report reports/exp_nllb-logit-bias-glossary/slice_treatment_b15.json --limit 60
# FLORES non-regression:
PYTHONPATH=. python3 scripts/eval_glossary.py --testset data/eval/flores_devtest_en_ko.tsv \
  --src en --tgt ko --mode control   --report reports/exp_nllb-logit-bias-glossary/flores_control_b0.json --limit 300
PYTHONPATH=. python3 scripts/eval_glossary.py --testset data/eval/flores_devtest_en_ko.tsv \
  --src en --tgt ko --mode treatment --bias 3 --report reports/exp_nllb-logit-bias-glossary/flores_treatment_b3.json --limit 300
```

---

## REVIEWER VERDICT (independent Korean particle spot-check) — OVERTURNS the qualified pass

A native-Korean reviewer agent inspected 30 triggered-term sentences at
bias=+3 (control vs treatment). Result:

- CLEAN 11 / MINOR 12 / BROKEN 7.
- **5 of the 7 BROKEN are treatment-specific** (not in control) — all
  particle/structure failures caused by the bias forcing:
  - S6: `처리량과 비교` (comitative 과) vs control `처리량을 비교` (object 을)
        — "compared WITH throughput" vs "compared throughput". Meaning changed.
  - S12: `처리량으로` (instrument 으로) vs control `출력은` (topic 은) — subject became instrument.
  - S14: `처리량 2x 향상` — subject particle dropped entirely.
  - S23: `처리량 향상` vs control `처리량이 향상` — subject particle 이 dropped.
  - S4: sentence truncated to a dangling connective when 3 terms triggered at once.
- Pattern: damage concentrates where 2–3 terms trigger simultaneously;
  the decoder places the forced tokens but the surrounding case-marking
  glue breaks.

**Reviewer's verdict: bias=+3 is NOT grammatically safe; particle damage
is present at +3, not only +15.** The 26.8% recall gain comes with ~17%
treatment-specific grammatical breakage on triggered sentences. Incorrect
particles change meaning (을 vs 과), so the effect is not cosmetic.

## REVISED MANAGER VERDICT: R3-3 = FAIL (corrected from qualified pass)

The corpus-BLEU non-regression check (−0.03 on FLORES) MISSED this because
(a) BLEU was char-level and (b) FLORES triggers the glossary on only
2/300 sentences, so grammar damage on triggered text is invisible in the
FLORES aggregate. The reviewer's targeted grammatical inspection is the
correct instrument and it fails the approach.

Soft token-level logit-bias is the wrong mechanism: it has no good
operating point (weak bias → no recall; strong bias → term-salad; even
moderate bias → particle breakage on multi-term sentences). Round-4 path:
position-aware constrained decoding over COMPLETE phrase sequences that
include their particles (Hokamp grid-beam / aligned trie), and/or cap
simultaneous triggers at 1 per sentence. This is a clean, well-diagnosed
NEGATIVE — the round-1 lesson (output-side beats prompt-injection) still
holds, but naive token-bias is not sufficient.
