# R3-2 — Korean CPU TTS — summary

## Question (this round)
Does ANY CPU Korean TTS clear the 25% round-trip-WER intelligibility
gate? (Round-2 espeak-ng gave 71.66% — unintelligible.) Picking the
production voice is a separate later decision.

## Answer: YES — MeloTTS Korean clears it decisively.

## Method
- Engine: **MeloTTS** (`myshell-ai/MeloTTS`, installed from source; the
  pypi `melotts` 0.1.2 also works once unidic is present). Korean VITS +
  `kykim/bert-kor-base` prosody. License **MIT**.
- For each of N=40 Fleurs ko_text references: synthesize Korean audio,
  resample 44.1 kHz→16 kHz, re-transcribe with faster-whisper-medium int8,
  compute round-trip WER (ko-mecab) and CER vs the original reference.

## Results (N=40)

| Engine | round-trip WER (mecab) | round-trip CER | synth speed (warm) | RAM peak | license | gate ≤25% |
|---|---:|---:|---|---:|---|---|
| **MeloTTS-KR** | **13.64%** | 10.17% | ~2–4× faster than real-time | 6.9 GB* | MIT | **PASS** |
| espeak-ng (round-2 baseline) | 71.66% | — | ~100× RT | <20 MB | GPL-3 | FAIL |

\* RAM peak (6.9 GB) is the **combined** process high-water mark and
includes the co-loaded faster-whisper-medium ASR (for re-transcription)
plus the BERT-kor prosody model. It is NOT the standalone TTS footprint
and must not be read as the deployable TTS RAM. A separate standalone
measurement is needed (next round); the MeloTTS VITS + BERT alone is
~1.5–2 GB from the scout estimate.

## Synthesis speed note
A one-off manual probe early in takeover showed ~16 s synth for 5 s
audio (cold path — BERT prosody lazy-loads and the first real sentences
include graph warming). The N=40 eval aggregate is the trustworthy
number: **91.9 s total synth wall-clock for 391.5 s of audio = ~4.3×
faster than real-time warm**, per-sentence 1.4–4.8 s. So MeloTTS is
real-time-capable on CPU once warm; budget the warm-up as a one-time
session-start cost.

## Verdict
**PASS — the strongest result of round 3.** MeloTTS Korean takes the
round-trip WER from espeak's unintelligible 71.66% down to 13.64%, well
under the 25% gate, at MIT license and faster-than-real-time warm synth.
This unblocks the "Korean TTS is the project's weakest link" finding from
round 2. The Korean voice arm of the cascade is now viable on CPU.

## Caveats / next-round follow-ups
1. **Standalone TTS RAM**: re-measure MeloTTS alone (no co-loaded ASR) to
   confirm it fits the ≤1 GB TTS budget; the 6.9 GB here is contaminated.
2. **Warm-up cost**: first-synth latency includes BERT-kor load + graph
   warming (several seconds). Pre-warm at session start so it doesn't hit
   the first user utterance.
3. **Production voice decision** (deferred): MeloTTS is single-speaker;
   if voice-cloning / speaker-identity preservation matters (README
   roadmap), that's a separate evaluation.

## Reproduce
```
git checkout exp/korean-cpu-tts
pip3 install --user git+https://github.com/myshell-ai/MeloTTS.git
python3 -m unidic download
PYTHONPATH=. python3 scripts/eval_tts.py --engine melo --limit 40 \
  --report reports/exp_korean-cpu-tts/tts_ko_melo.json
```
