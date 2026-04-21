# Getting Started in 5 Minutes

The fastest path from `git clone` to translating your first sentence.

---

## Prereqs (30 seconds)

- Python 3.11+
- `ffmpeg` installed system-wide
- (For real inference) CUDA-capable GPU with ≥16 GB VRAM
- (For browser demo) modern Chrome/Firefox/Safari

---

## Step 1 — Setup (1 minute)

```bash
git clone <your-repo-url> adaptive-live-translator
cd adaptive-live-translator

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

cp .env.example .env
# Edit .env — at minimum set HF_TOKEN if using gated models
```

---

## Step 2 — Smoke test without downloading models (1 minute)

```bash
pytest tests/ -v
```

All tests use stub models and complete in a few seconds. This confirms
your environment is wired up correctly **before** you spend time
downloading ~30 GB of weights.

---

## Step 3 — Download models (2 minutes setup + background download)

```bash
# Full download (~30 GB)
python scripts/download_models.py

# Or minimal — skip TTS if you only need text output
python scripts/download_models.py --skip-tts
```

Models cache to `./models/`. You only do this once.

---

## Step 4 — Start the server (30 seconds)

```bash
make run
# → Uvicorn running on http://0.0.0.0:8000
```

Check it's alive:

```bash
curl http://localhost:8000/health
# → {"status":"ok","active_sessions":0}
```

---

## Step 5 — Translate something (30 seconds)

### Option A: Browser demo (easiest)

Open `demo/index.html` in your browser, fill in languages, click **Start Recording**,
and talk. You'll see live source + target text with latency numbers.

### Option B: CLI with a WAV file

```bash
python scripts/test_client.py \
    --audio samples/en_tech_talk.wav \
    --src en --tgt ko \
    --topic "AI research talk"
```

### Option C: curl + websocat

```bash
# Create session
SESSION_ID=$(curl -s -X POST http://localhost:8000/sessions \
    -H "Content-Type: application/json" \
    -d '{"src_lang":"en","tgt_lang":"ko","speaker_id":"alice"}' \
    | jq -r .session_id)

# Stream audio
websocat ws://localhost:8000/ws/$SESSION_ID < audio.pcm
```

---

## Step 6 — Make it adapt (the interesting part)

Add a glossary on the fly for an upcoming meeting:

```bash
curl -X POST http://localhost:8000/speakers/alice/glossary \
    -H "Content-Type: application/json" \
    -d '{"src":"Project Aurora","tgt":"Project Aurora","dnt":true}'
```

Or send glossary entries directly over the WebSocket at session start — see
`demo/index.html` for the JSON shape.

The next sentence translated will use your new glossary term **immediately**,
no retraining.

---

## What to read next

- `docs/architecture.md` — how the components wire together
- `docs/prompt_template.md` — exactly what the LLM sees
- `docs/evaluation.md` — how to measure BLEU + StreamLAAL on your own data
- `README.md` — configuration reference
