"""P3-0 templated corpus generation (fallback (b) after Qwen spot-check bail).

Grammatical Korean by construction (josa allomorph from batchim via Unicode;
Latin DNT terms get a Korean classifier noun). English keeps the term in
object/oblique position to avoid subject-verb agreement issues, with a
natural per-term noun phrase. Curated, semantically-sensible multi-trigger
sentences. Writes the raw JSONL format the build script consumes.
"""
import json, random, itertools
random.seed(17)
GL = json.load(open("data/glossaries/ml-conference-en-ko.json"))["entries"]

def _jong(ch):
    if '가' <= ch <= '힣': return (ord(ch) - 0xAC00) % 28
    return None
def J(word, kind):
    j = _jong(word[-1]); has = (j is not None and j != 0)
    if kind == "obj":  return word + ("을" if has else "를")
    if kind == "top":  return word + ("은" if has else "는")
    if kind == "subj": return word + ("이" if has else "가")
    raise ValueError(kind)

# en_phrase = natural English noun phrase (canonical term is a substring of it)
EN_PHRASE = {"LLM": "LLMs", "fine-tuning": "fine-tuning", "inference": "inference",
             "embedding": "embeddings", "attention mechanism": "attention mechanisms",
             "tokenizer": "tokenizers", "hallucination": "hallucinations", "RLHF": "RLHF",
             "NVIDIA": "NVIDIA hardware", "PyTorch": "the PyTorch framework",
             "HuggingFace": "the HuggingFace library", "benchmark": "benchmarks",
             "throughput": "throughput", "latency": "latency", "quantization": "quantization"}
DNT_CLASS = {"RLHF": "RLHF 기법", "NVIDIA": "NVIDIA 하드웨어",
             "PyTorch": "PyTorch 프레임워크", "HuggingFace": "HuggingFace 라이브러리"}
UNITS = [{"term": e["src"], "dnt": e["dnt"], "en": EN_PHRASE[e["src"]],
          "ko": (DNT_CLASS[e["src"]] if e["dnt"] else e["tgt"])} for e in GL]

ACTOR = [("Researchers", "연구진은"), ("The team", "연구팀은"), ("Engineers", "엔지니어들은"),
         ("We", "우리는"), ("The authors", "저자들은"), ("The lab", "연구실은")]
CTX = [("production systems", "프로덕션 시스템"), ("recent research", "최근 연구"),
       ("large-scale settings", "대규모 환경"), ("our experiments", "우리 실험"),
       ("real-time applications", "실시간 애플리케이션"), ("industrial pipelines", "산업 파이프라인")]
ASP = [("performance", "성능"), ("efficiency", "효율성"), ("accuracy", "정확도"),
       ("scalability", "확장성"), ("reliability", "신뢰성"), ("robustness", "견고성")]

# frames: term in object/oblique position only (no subject-verb agreement risk)
def frames_en_ko(en, ko, a, c, s):
    return [
        (f"{a[0]} evaluated {en} in {c[0]}.",
         f"{a[1]} {c[1]}에서 {J(ko,'obj')} 평가했습니다."),
        (f"{a[0]} improved {en} to boost {s[0]}.",
         f"{a[1]} {J(s[1],'obj')} 높이기 위해 {J(ko,'obj')} 개선했습니다."),
        (f"Research on {en} is active in {c[0]}.",
         f"{ko}에 대한 연구가 {c[1]}에서 활발히 진행되고 있습니다."),
        (f"{a[0]} measured the impact of {en} on {s[0]}.",
         f"{a[1]} {J(ko,'subj')} {s[1]}에 미치는 영향을 측정했습니다."),
        (f"Understanding {en} is important for robust systems.",
         f"{J(ko,'obj')} 이해하는 것은 견고한 시스템에 중요합니다."),
        (f"{a[0]} optimized {en} for better {s[0]} in {c[0]}.",
         f"{a[1]} {c[1]}에서 더 나은 {J(s[1],'obj')} 위해 {J(ko,'obj')} 최적화했습니다."),
        (f"Recent advances in {en} have improved {s[0]}.",
         f"{ko} 분야의 최근 발전은 {J(s[1],'obj')} 개선했습니다."),
        (f"Without proper {en}, systems often underperform.",
         f"적절한 {ko} 없이는 시스템이 종종 제대로 작동하지 않습니다."),
    ]

def frames_ko_en(en, ko, a, c, s):
    return [
        (f"{a[1]} {c[1]}에서 {J(ko,'obj')} 연구했습니다.",
         f"{a[0]} studied {en} in {c[0]}."),
        (f"{a[1]} {J(s[1],'obj')} 높이기 위해 {J(ko,'obj')} 최적화했습니다.",
         f"{a[0]} optimized {en} to increase {s[0]}."),
        (f"{ko}에 대한 연구가 {c[1]}에서 활발히 진행되고 있습니다.",
         f"Research on {en} is active in {c[0]}."),
        (f"{a[1]} {J(ko,'subj')} {s[1]}에 미치는 영향을 측정했습니다.",
         f"{a[0]} measured the impact of {en} on {s[0]}."),
        (f"이 논문은 {J(ko,'obj')} 자세히 분석합니다.",
         f"This paper analyzes {en} in detail."),
        (f"{a[1]} 더 나은 {J(s[1],'obj')} 위해 {J(ko,'obj')} 개선했습니다.",
         f"{a[0]} improved {en} for better {s[0]}."),
        (f"{ko} 분야의 최근 발전은 {J(s[1],'obj')} 개선했습니다.",
         f"Recent advances in {en} have improved {s[0]}."),
        (f"적절한 {ko} 없이는 시스템이 종종 제대로 작동하지 않습니다.",
         f"Without proper {en}, systems often underperform."),
    ]

# curated, semantically-correct multi-trigger sentences (distinct per direction)
# (english, korean, primary_term) — primary term's canonical appears in both sides
MULTI_EN_KO = [
    ("Fine-tuning LLMs reduced hallucinations.", "대형 언어 모델을 파인튜닝하여 환각 현상을 줄였습니다.", "fine-tuning"),
    ("Quantization reduced inference latency.", "양자화는 추론 지연 시간을 줄였습니다.", "quantization"),
    ("RLHF reduced hallucinations in LLMs.", "RLHF 기법은 대형 언어 모델의 환각 현상을 줄였습니다.", "RLHF"),
    ("The attention mechanism improves LLM accuracy.", "어텐션 메커니즘은 대형 언어 모델의 정확도를 향상시킵니다.", "attention mechanism"),
    ("The tokenizer affects embedding quality.", "토크나이저는 임베딩 품질에 영향을 미칩니다.", "tokenizer"),
    ("Fine-tuning improved benchmark accuracy.", "파인튜닝은 벤치마크 정확도를 높였습니다.", "benchmark"),
]
# stored as (english, korean, term) for a consistent append loop; direction=ko_en
MULTI_KO_EN = [
    ("Quantization improved inference throughput.", "양자화는 추론 처리량을 개선했습니다.", "quantization"),
    ("Researchers fine-tuned LLMs to raise benchmark scores.", "연구진은 대형 언어 모델을 파인튜닝하여 벤치마크 점수를 높였습니다.", "LLM"),
    ("Quantization lowered inference latency on NVIDIA hardware.", "양자화는 NVIDIA 하드웨어에서 추론 지연 시간을 낮췄습니다.", "NVIDIA"),
    ("The attention mechanism affects LLM throughput.", "어텐션 메커니즘은 대형 언어 모델의 처리량에 영향을 줍니다.", "attention mechanism"),
    ("Tokenizer choice determines embedding quality.", "토크나이저 선택은 임베딩 품질을 좌우합니다.", "tokenizer"),
    ("RLHF improved the reliability of LLMs.", "RLHF 기법은 대형 언어 모델의 신뢰성을 높였습니다.", "RLHF"),
]

def gen(direction, per_term=70):
    rows = []
    for u in UNITS:
        combos = list(itertools.product(ACTOR, CTX, ASP)); random.shuffle(combos)
        seen = set()
        for a, c, s in combos:
            fr = frames_en_ko(u["en"], u["ko"], a, c, s) if direction == "en_ko" \
                 else frames_ko_en(u["en"], u["ko"], a, c, s)
            for item in fr:
                en, ko = item if direction == "en_ko" else (item[1], item[0])
                if (en, ko) in seen: continue
                seen.add((en, ko))
                rows.append({"en": en, "ko": ko, "term": u["term"], "dnt": u["dnt"], "direction": direction})
                if len(seen) >= per_term: break
            if len(seen) >= per_term: break
    return rows

def main():
    for d, multi in [("en_ko", MULTI_EN_KO), ("ko_en", MULTI_KO_EN)]:
        rows = gen(d, per_term=70)
        for en, ko, term in multi:
            rows.append({"en": en, "ko": ko, "term": term, "dnt": False, "direction": d})
        random.shuffle(rows)
        with open(f".claude/p3_0_raw_{d}.jsonl", "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[tmpl] {d}: wrote {len(rows)} raw pairs")

if __name__ == "__main__":
    main()
