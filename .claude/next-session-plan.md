# Next-session plan (resume round 4, then round-5 planning)

Committed to the repo deliberately — prior round plans kept living only in
chat and not surviving ("uncommitted work doesn't travel"). This is the
authoritative handoff for the next window. Human-approved 2026-06-02.

## Current state (end of round 4)
- main @ 92266ae (= origin/main). Round reports R1–R4 in reports/.
- R4-5 DONE (ko-mecab BLEU restored on main; MeloTTS standalone RAM 2.68 GB).
- R4-2 QUALIFIED PASS — exp/ko-streaming-latency (ko first-emission 3.07 s
  mean, WER 13.93 mecab; caveats: test-set-selected config, p95 5.82 s).
- R4-4 FAIL — exp/glossary-phrase-constrained (beam degeneration; 3rd
  glossary mechanism to fail on NLLB-600M).
- NOT done: R4-1 (e2e composition, the payoff), R4-3 (ko→en post-proc).
- Stopped on budget: two runners hit session limits in round 4.

## Do these IN ORDER next window

### 1. R4-1 — e2e composition (PAYOFF, do first)
Wire the upgraded stages into scripts/eval_e2e.py and measure on the
Fleurs paired manifest (data/eval/fleurs_pair_manifest.tsv):
- ASR: R4-2 config (faster-whisper base+small, LocalAgreement-2 +
  confidence gate −0.7) from exp/ko-streaming-latency.
- MT: NLLB-200-distilled-600M CT2 int8 (A1).
- TTS: MeloTTS-KR (exp/korean-cpu-tts).
Report: end-to-end latency, ko-mecab BLEU (R4-5 fix makes this automatic),
and **REAL co-resident peak RSS** — this is the load-bearing measurement.

**RAM reconciliation is the whole point.** R4-5 says co-resident ≈ 5 GB
(MeloTTS 2.68 + ASR ~1.7 + MT ~0.7) likely BUSTS the 4 GB total budget.
If R4-1 confirms the bust: that is a FINDING, not a failure. Do NOT stop at
"wins don't compose." Explore the cheapest fix, in this order:
  (a) MeloTTS lazy-load / unload between utterances (TTS is bursty; it
      doesn't need to be resident during ASR+MT). Likely the cheapest win.
  (b) a lighter Korean voice (re-evaluate round-3 TTS options for RAM).
  (c) revise the 4 GB total budget upward — it was set for an 8 GB laptop;
      5 GB may be acceptable. A budget-revision is a legitimate outcome.
Run runners SERIALLY (round-3 lesson — three-parallel crashed the box).

### 2. R4-3 — ko→en post-processor (inline, fold onto R4-1's branch)
Cheapest item. Casing + terminal-punctuation restoration on the English
hypothesis (sacrebleu 13a is case/punct sensitive). Expected ~+3 BLEU on
ko→en. Measure on the same e2e run: BLEU with vs without post-proc.

## Carry-forward (not next-window-blocking)

### R4-2 robustness
The 3.07 s ko latency margin was grid-searched on the ko test set
(optimistic) and p95 is 5.82 s. Confirm on a held-out ko set (e.g. a
different Fleurs/CommonVoice slice) and address the tail before claiming
the gate is robustly cleared.

### Glossary axis — RESEARCH CONCLUSION, not a TODO
Decoding-time glossary enforcement on NLLB-600M is DEAD: three rounds,
three mechanisms, three failures (R1 prompt-injection, R3 token-bias,
R4 constraint-beam). Do NOT try a 4th decoding-time variant (the
1-trigger-cap idea is a fourth try at a broken paradigm — skip it).
Round-5 is a PLANNING question, to decide AFTER R4-1, between:
  (a) LoRA fine-tune NLLB on term-containing parallel data so terminology
      is LEARNED with correct particles (not forced), or
  (b) accept NLLB's native terminology and drop the hard-enforcement
      requirement entirely.
Do not pre-commit to (a) or (b) now.

## Standing operational rules (hold across rounds)
- Runners are Sonnet; spec at .claude/agents/experiment-runner.md has the
  hardened verdict-discipline rules (N≥20, reconcile, don't silence
  warnings, verify before declaring broken).
- Run model-heavy experiments SERIALLY, never three-parallel (8-core/7 GB).
- Worktrees from dead agents leave stale locks; force-unlock + prune at
  round start.
- Verify .claude/settings.json is valid JSON before spawning (a broken
  edit silently disabled agent teams in rounds 1–2).
- ko BLEU must read `bleu_tokenize: ko-mecab` in outputs; if it says char,
  the R4-5 mecab fix didn't load — investigate, don't accept char.
- Plan files must be committed to the repo, not left in chat.
