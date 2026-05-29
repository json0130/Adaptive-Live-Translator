# R4-5 — Protocol hygiene (inline, manager)

## 1. ko-mecab BLEU restored (was silently falling back to char)

**Problem.** `sacrebleu`'s `BLEU(tokenize='ko-mecab')` raised AttributeError
in this environment and silently fell back to char-BLEU. Root cause: a
version mismatch between the `mecab_ko` binding and the `mecab_ko_dic`
dictionary package — sacrebleu's `TokenizerKoMecab` expects attributes
(`MECAB_ARGS`, then `DICDIR`) that the newer `mecab_ko_dic` dropped (it now
exposes `dictionary_path` / `model_path`). The regression was introduced
mid-project when R3-2's MeloTTS install upgraded `mecab_ko_dic`. Round-3
R3-3 reported char-BLEU as a result.

**Fix.** `src/utils/metrics.compute_bleu` no longer uses sacrebleu's
ko-mecab tokenizer. For `tokenize="ko-mecab"` it now pre-tokenizes hyp/ref
with a direct `MeCab.Tagger(-d <dictionary_path> -Owakati)` and scores with
sacrebleu `tokenize="none"`. Same morpheme segmentation, no version
coupling. Verified: identical→100.0, different→13.7, reports `"ko-mecab"`
(not the char fallback). The 13a path for en targets is unchanged.

Impact: all round-4 experiments that score Korean BLEU through the locked
harness now produce protocol-compliant ko-mecab numbers. A1's round-2
25.24 was real ko-mecab (it ran before the break); R3-3's char numbers
should be re-read as char (deltas valid, absolutes not comparable).

## 2. Standalone MeloTTS-KR RAM measured

Round-3 reported 6.9 GB peak for MeloTTS, but that was contaminated by the
co-loaded faster-whisper-medium re-ASR. Measured standalone (MeloTTS only,
3 warm synths):

**Standalone MeloTTS-KR peak RSS: ~2.68 GB.**

This clears intelligibility (R3-2: 13.64% round-trip WER) but **exceeds
the ≤1 GB per-stage TTS budget** in the eval protocol.

### Composition risk flagged to R4-1
Co-resident pipeline estimate:
- streaming ASR (faster-whisper-small int8): ~1.7 GB
- MT (NLLB-200-distilled-600M CT2 int8): ~0.7 GB
- TTS (MeloTTS-KR): ~2.6 GB
- **Total ≈ 5 GB — over the ≤4 GB total-pipeline budget.**

R4-1 must measure real co-resident peak RSS and will likely find the
4 GB total budget is busted by MeloTTS. Mitigations to consider:
lazy-load/unload TTS between utterances, a lighter Korean voice, or
revising the 4 GB budget (the original 8 GB-laptop assumption may tolerate
5 GB). This is a real finding, not a blocker — but R4-1's verdict on "do
the wins compose" must account for it.
