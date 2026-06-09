"""P2-0 final verification: locked ko-mecab + P2-1 actual-stack VRAM peak."""
import time, torch, traceback

def used():
    free, total = torch.cuda.mem_get_info(); return (total-free)/1e9, total/1e9

# ---- locked ko-mecab tokenizer (eval-protocol integrity) ----
from src.utils.metrics import compute_bleu
s, tok = compute_bleu(["기계 학습 모델은 평가가 필요합니다"],
                      ["기계 학습 모델은 평가가 필요합니다"], tokenize="ko-mecab")
print(f"FINAL: ko_mecab_tokenize_used={tok} (self-bleu={s:.1f})")
assert tok == "ko-mecab", f"ko-mecab FELL BACK to {tok}"

u0,total = used(); print(f"FINAL: vram_total_gb={total:.2f} baseline_used_gb={u0:.2f}")

# ---- P2-1 actual stack: ASR + NLLB-fp16 + MeloTTS, all CUDA ----
from faster_whisper import WhisperModel
asr = WhisperModel("large-v3", device="cuda", compute_type="float16")
segs,_ = asr.transcribe("data/eval/fleurs_ko_kr_test/audio/14336290879561136744.wav", language="ko", beam_size=1)
_ = "".join(s.text for s in segs)
u1,_ = used(); print(f"FINAL: vram_after_asr_gb={u1:.2f}")

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
nllb_tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
nllb = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").half().to("cuda").eval()
u2,_ = used(); print(f"FINAL: vram_after_asr+nllb_gb={u2:.2f}")

melo_ok = False
try:
    from melo.api import TTS
    t0=time.perf_counter()
    melo = TTS(language="KR", device="cuda")
    spk = melo.hps.data.spk2id["KR"]
    print(f"FINAL: melo_device={melo.device} load_s={time.perf_counter()-t0:.1f}")
    melo.tts_to_file("안녕하세요. 기계 학습 평가입니다.", spk, "/tmp/p2_0_melo.wav", speed=1.0, quiet=True)
    import soundfile as sf
    d,sr = sf.read("/tmp/p2_0_melo.wav")
    u3,_ = used(); print(f"FINAL: vram_ALL_THREE_coresident_gb={u3:.2f}  (P2-1 stack peak)")
    print(f"FINAL: melo_synth_ok samples={len(d)} sr={sr}")
    melo_ok = True
except Exception as e:
    u3,_ = used()
    print(f"FINAL: MELO_LOAD_FAILED vram_at_failure_gb={u3:.2f} err={type(e).__name__}: {str(e)[:160]}")
    print("FINAL: (non-blocking — P2-3 translator-only fallback protected)")

print(f"FINAL: melotts_on_blackwell={'PASS' if melo_ok else 'FAIL(non-blocking)'}")
