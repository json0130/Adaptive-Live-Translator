# Model Licenses

Every model downloaded by `scripts/download_models.py` has its own license.
**Check each one before production/commercial use.**

| Component | Model | License | Commercial use | Notes |
|---|---|---|---|---|
| ASR | `openai/whisper-large-v3` | MIT | ✅ Yes | Permissive |
| Translator | `Qwen/Qwen2.5-7B-Instruct` | Apache 2.0 | ✅ Yes | Check Qwen license addendum for very high-traffic deployments |
| Embeddings | `BAAI/bge-m3` | MIT | ✅ Yes | Permissive |
| TTS | `FunAudioLLM/CosyVoice2-0.5B` | Apache 2.0 | ✅ Yes | Voice cloning with real-person consent only |
| Speaker ID (optional) | `pyannote/speaker-diarization-3.1` | MIT | ✅ Yes | Requires HF token acceptance |

## This project's code

Apache 2.0.

## Data

The sample glossary, TM, and speaker profile under `data/` are released under
CC-BY-4.0 — they contain synthetic English/Korean examples. Your own data
stays with you.

## Voice cloning disclosure

CosyVoice 2 can clone a speaker's voice from ~10 seconds of audio. When
using this feature:
- **Always obtain explicit consent** from the person whose voice you are cloning
- Do not clone voices of public figures without permission
- Comply with local biometric / likeness laws (EU AI Act, CA SB 1001, etc.)
- Add audible or inaudible watermarks to synthesised audio where feasible

See https://github.com/FunAudioLLM/CosyVoice for the upstream terms.

## Third-party APIs (optional)

If you swap the local stack for API providers via `.env`:

- **OpenAI GPT-4o / Realtime API** — [OpenAI TOS](https://openai.com/policies/terms-of-use)
- **Anthropic Claude** — [Anthropic TOS](https://www.anthropic.com/legal/consumer-terms)
- **Deepgram Nova-3** — [Deepgram TOS](https://deepgram.com/terms)

Their privacy terms around recorded audio differ significantly — check
before sending user audio through any of them.
