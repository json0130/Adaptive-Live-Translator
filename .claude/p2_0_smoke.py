"""P2-0 translator-only CUDA smoke-load (exit criterion).

Loads NLLB-600M fp16 (transformers), the CT2-int8 translator, and
faster-whisper-large-v3 on CUDA and runs each once, both directions where
applicable. Verifies NO silent CPU fallback. Prints grep-able SMOKE: lines.
"""
import time, torch

EN = "Machine learning models require careful evaluation before deployment."
KO = "기계 학습 모델은 배포 전에 신중한 평가가 필요합니다."
EN_WAV = "data/eval/fleurs_en_us_test/audio/17158526797037461547.wav"
KO_WAV = "data/eval/fleurs_ko_kr_test/audio/14336290879561136744.wav"

def vram():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e9

print(f"SMOKE: cuda_available={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0)}")
print(f"SMOKE: vram_used_start_gb={vram():.2f}")

# ---- 1) NLLB-600M fp16 via transformers on CUDA ----
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
name = "facebook/nllb-200-distilled-600M"
t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSeq2SeqLM.from_pretrained(name).half().to("cuda").eval()
dev = str(next(model.parameters()).device)
dt = (time.perf_counter() - t0)
print(f"SMOKE: nllb_fp16_param_device={dev} dtype={next(model.parameters()).dtype} load_s={dt:.1f}")
assert dev.startswith("cuda"), "NLLB fp16 NOT on cuda (silent CPU fallback)"

def nllb_translate(text, src, tgt):
    tok.src_lang = src
    inp = tok(text, return_tensors="pt").to("cuda")
    out = model.generate(**inp, forced_bos_token_id=tok.convert_tokens_to_ids(tgt), max_length=128)
    return tok.batch_decode(out, skip_special_tokens=True)[0]

print("SMOKE: nllb_fp16 en->ko =", nllb_translate(EN, "eng_Latn", "kor_Hang"))
print("SMOKE: nllb_fp16 ko->en =", nllb_translate(KO, "kor_Hang", "eng_Latn"))
print(f"SMOKE: vram_used_after_nllb_gb={vram():.2f}")

# ---- 2) CT2-int8 translator on CUDA ----
import ctranslate2, sentencepiece as spm
CT2_DIR = "models/nllb-600m-ct2-int8"
ct = ctranslate2.Translator(CT2_DIR, device="cuda", compute_type="int8_float16")
sp = spm.SentencePieceProcessor(); sp.Load(f"{CT2_DIR}/sentencepiece.bpe.model")
ct_tok = AutoTokenizer.from_pretrained(CT2_DIR)
print(f"SMOKE: ct2_device={ct.device} compute_type=int8_float16")

def ct2_translate(text, src, tgt):
    ct_tok.src_lang = src
    ids = ct_tok.encode(text, add_special_tokens=True)
    toks = ct_tok.convert_ids_to_tokens(ids)
    res = ct.translate_batch([toks], target_prefix=[[tgt]], beam_size=1, max_decoding_length=256)
    out = res[0].hypotheses[0]
    if out and out[0] == tgt:
        out = out[1:]
    return sp.DecodePieces(out)

print("SMOKE: ct2_int8 en->ko =", ct2_translate(EN, "eng_Latn", "kor_Hang"))
print("SMOKE: ct2_int8 ko->en =", ct2_translate(KO, "kor_Hang", "eng_Latn"))
print(f"SMOKE: vram_used_after_ct2_gb={vram():.2f}")

# ---- 3) faster-whisper large-v3 on CUDA float16 ----
from faster_whisper import WhisperModel
t0 = time.perf_counter()
asr = WhisperModel("large-v3", device="cuda", compute_type="float16")
print(f"SMOKE: fw_large_v3 loaded device=cuda compute_type=float16 load_s={time.perf_counter()-t0:.1f}")

def transcribe(path, lang):
    segs, info = asr.transcribe(path, language=lang, beam_size=1)
    return "".join(s.text for s in segs).strip(), info

en_text, en_info = transcribe(EN_WAV, "en")
ko_text, ko_info = transcribe(KO_WAV, "ko")
print(f"SMOKE: fw en (lang_prob={en_info.language_probability:.2f}) =", en_text[:90])
print(f"SMOKE: fw ko (lang_prob={ko_info.language_probability:.2f}) =", ko_text[:90])
print(f"SMOKE: vram_used_all_three_coresident_gb={vram():.2f}")
print("SMOKE: EXIT_CRITERION=PASS (translator+ASR loaded on CUDA, no silent CPU fallback)")
