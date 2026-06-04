# Round 5 plan — attack the two real bottlenecks (strict scope)

Human-approved 2026-06-04. Committed so it travels (chat-only plans don't
survive). Follows the R4-1 SATURATION finding: the translator/context axis is
done; the operative gates are blocked by **ASR final-pass latency** and
**TTS RAM**.

## Hard scope (do not expand)
- **Two experiments total. No third.** If both fail → terminate clean; that is
  a publishable finding ("standard fixes don't move the gates → architectural
  limit"), NOT a reason to try a third.
- **Glossary axis: CLOSED** (3 mechanisms failed on NLLB-600M).
- **Translator/context axis: CLOSED** (R4-1 saturation finding).
- Serial runs only. ko-mecab guard on all Korean BLEU. Reviewer pass on any
  result that looks like a pass.

## The two experiments (each owns ONE gate; judged only on its own axis)

### R5-1 — single-pass ASR (owns the 3.5 s LATENCY gate)
Branch `exp/asr-single-pass`. Drop the small-model final pass so the quality
path == the low-latency path (today's two-pass scheme re-decodes with the
small model after streaming, adding 3-5 s that the BLEU depends on but the
reported latency excluded). Measure the BLEU cost of single-model streaming
vs the R4-2 base+small two-pass scheme. Both directions. GATE: e2e latency
(audio-in -> first-audio-out) <= 3.5 s with no hidden final pass.

### R5-2 — TTS RAM slim (owns the < 4 GB RAM gate)
Branch `exp/tts-ram-slim`. Replace MeloTTS-KR's `kykim/bert-kor-base`
text-normalizer (rule-based G2P or a smaller model) and fix the per-utterance
allocation growth (R4-1: peak grew 3.8 -> 7.6 GB over 50 segments). GATE:
co-resident peak RSS < 4 GB while keeping the R3-2 intelligibility gate
(<= 25 % round-trip WER). Measure peak via committed ru_maxrss + instantaneous
VmRSS samples.

## Report framing — THREE honest outcome categories (not a pass/fail scale)
A split is a distinct, informative outcome, NOT a mushy "partial pass."

| Outcome | Meaning | Verdict |
|---|---|---|
| Both pass | Standard fixes move both gates | **Deployable product** on this HW class |
| Split (one passes) | Characterizes which axis is tractable, which isn't | **Distinct outcome** — name the tractable axis and the hard blocker |
| Both fail | Standard fixes don't move the gates | **Architectural limit** — publishable negative finding |

The round-5 report must classify the result into exactly one of these three
and, on a split, name which axis (ASR-latency vs TTS-RAM) is tractable on this
hardware class and which is the blocker.
