"""P3-0 corpus generation via local Qwen2.5-7B-Instruct (nf4).

Generates en->ko and ko->en ML-domain parallel pairs INDEPENDENTLY (separate
sentence sets per direction, no round-trip), each constrained to a glossary
canonical term. Output: raw JSONL (one record per pair) for downstream
filter/split/stats. Run with --test first to eyeball Korean quality.

Usage:
  .qwen_gen/bin/python .claude/p3_0_gen_corpus.py --direction en_ko --calls-per-term 5 --per-call 15 --out .claude/p3_0_raw_en_ko.jsonl
  add --test for a single term / single call sanity batch.
"""
import argparse, glob, json, re, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

GLOSSARY = json.load(open("data/glossaries/ml-conference-en-ko.json"))
ENTRIES = GLOSSARY["entries"]

def load_qwen():
    path = glob.glob("models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/")[0]
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb,
                                                 device_map="cuda", torch_dtype=torch.float16)
    model.eval()
    return tok, model

def build_prompt(entry, direction, n):
    src, tgt, dnt = entry["src"], entry["tgt"], entry["dnt"]
    if direction == "en_ko":
        if dnt:
            ko_rule = f"keep '{src}' UNCHANGED in the Korean (do not translate or transliterate it)"
        else:
            ko_rule = (f"write EXACTLY '{tgt}' in the Korean and MUST NOT keep the English "
                       f"'{src}' anywhere in the Korean sentence")
        user = (f"Generate {n} diverse, natural English sentences from ML/AI papers, blogs, or "
                f"docs that each use the term '{src}'. For each, give a fluent, grammatical Korean "
                f"translation that MUST {ko_rule}. Write ALL Korean in Hangul only (no "
                f"Chinese/Japanese characters). Vary structure, length, and context. "
                f"Output STRICT JSONL, one object per line: {{\"en\": \"...\", \"ko\": \"...\"}}. No commentary.")
    else:  # ko_en
        if dnt:
            ko_term = src  # DNT term appears verbatim in Korean too
        else:
            ko_term = tgt
        en_rule = (f"The English MUST contain '{src}' verbatim and MUST NOT paraphrase it"
                   if not dnt else f"The English MUST keep '{src}' unchanged")
        user = (f"Generate {n} diverse, natural Korean sentences from the ML/AI domain that each use "
                f"the term '{ko_term}'. For each, give a fluent English translation. {en_rule}. "
                f"Write ALL Korean in Hangul only (no Chinese/Japanese characters). "
                f"Vary structure, length, and context. Output STRICT JSONL, one object per line: "
                f"{{\"ko\": \"...\", \"en\": \"...\"}}. No commentary.")
    return [{"role": "system", "content": "You write high-quality bilingual ML-domain training data. Output only JSONL."},
            {"role": "user", "content": user}]

def parse_jsonl(text):
    out = []
    for line in text.splitlines():
        line = line.strip().strip("`").strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            o = json.loads(line)
            if "en" in o and "ko" in o and o["en"].strip() and o["ko"].strip():
                out.append({"en": o["en"].strip(), "ko": o["ko"].strip()})
        except Exception:
            continue
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["en_ko", "ko_en"], required=True)
    ap.add_argument("--calls-per-term", type=int, default=5)
    ap.add_argument("--per-call", type=int, default=15)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()

    tok, model = load_qwen()
    print(f"[gen] Qwen loaded; VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    entries = ENTRIES[:1] if a.test else ENTRIES
    calls = 1 if a.test else a.calls_per_term
    n = 0
    fout = sys.stdout if a.test else open(a.out, "w")
    t0 = time.time()
    for entry in entries:
        for _ in range(calls):
            msgs = build_prompt(entry, a.direction, a.per_call)
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = model.generate(ids, max_new_tokens=1100, do_sample=True,
                                     temperature=0.8, top_p=0.9, pad_token_id=tok.eos_token_id)
            text = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
            pairs = parse_jsonl(text)
            for p in pairs:
                p["term"] = entry["src"]; p["dnt"] = entry["dnt"]; p["direction"] = a.direction
                if a.test:
                    print(json.dumps(p, ensure_ascii=False))
                else:
                    fout.write(json.dumps(p, ensure_ascii=False) + "\n"); n += 1
            if a.test:
                print(f"[test] term={entry['src']} parsed={len(pairs)} raw_chars={len(text)}", flush=True)
    if not a.test:
        fout.close()
        print(f"[gen] wrote {n} raw pairs to {a.out} in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
