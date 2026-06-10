"""P3-0 build: filter raw Qwen pairs -> canonical, clean, eval-disjoint corpus.

Filters: canonical-term presence, script-contamination (reject CJK/kana in
Korean, Hangul/CJK in English), length sanity, dedup, zero overlap with the
locked eval slice (hash-verified). Splits 90/10 train/dev. Emits stats + a
stratified spot-check sample + a checkpoint markdown.
"""
import json, re, random, hashlib, statistics as st, collections
random.seed(13)

GL = json.load(open("data/glossaries/ml-conference-en-ko.json"))["entries"]
CANON_KO = {e["src"]: (e["src"] if e["dnt"] else e["tgt"]) for e in GL}  # term -> ko string to require
DNT = {e["src"] for e in GL if e["dnt"]}
ALL_KO_TERMS = {e["src"]: (e["src"] if e["dnt"] else e["tgt"]) for e in GL}
ALL_EN_TERMS = {e["src"]: e["src"] for e in GL}

CJK = re.compile(r"[一-鿿぀-ヿ]")          # Chinese ideographs + kana
HANGUL = re.compile(r"[가-힣]")
def norm(s): return re.sub(r"\s+", " ", s.strip().lower())

def ko_clean(ko):
    if CJK.search(ko): return False              # Chinese/Japanese contamination
    if not HANGUL.search(ko): return False        # must contain Hangul
    return True
def en_clean(en):
    if HANGUL.search(en) or CJK.search(en): return False
    return True

def wc(s): return len(s.split())

def load_slice_norms():
    norms = set()
    for f in ["data/eval/ml_glossary_slice_en_ko.tsv", "data/eval/ml_glossary_slice_ko_en.tsv"]:
        for line in open(f, encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            for p in parts:
                if p.strip(): norms.add(norm(p))
    return norms

def build(direction):
    raw = [json.loads(l) for l in open(f".claude/p3_0_raw_{direction}.jsonl", encoding="utf-8")]
    slice_norms = load_slice_norms()
    drop = collections.Counter(); seen = set(); kept = []
    for r in raw:
        en, ko, term = r["en"].strip(), r["ko"].strip(), r["term"]
        # direction defines (src,tgt) columns
        if direction == "en_ko": src_txt, tgt_txt = en, ko
        else:                    src_txt, tgt_txt = ko, en
        # canonical term presence: ko must carry canonical ko term; en must carry en term
        if CANON_KO[term] not in ko: drop["no_canonical_ko"] += 1; continue
        if term.lower() not in en.lower(): drop["no_en_term"] += 1; continue
        if not ko_clean(ko): drop["ko_contaminated"] += 1; continue
        if not en_clean(en): drop["en_contaminated"] += 1; continue
        if not (3 <= wc(en) <= 60): drop["len_en"] += 1; continue
        if not (4 <= len(ko) <= 220): drop["len_ko"] += 1; continue
        key = (norm(src_txt), norm(tgt_txt))
        if key in seen: drop["dup"] += 1; continue
        if norm(en) in slice_norms or norm(ko) in slice_norms: drop["eval_overlap"] += 1; continue
        seen.add(key)
        kept.append({"src": src_txt, "tgt": tgt_txt, "term": term, "en": en, "ko": ko})
    return raw, kept, drop, slice_norms

def trigger_density(kept):
    dens = []
    for p in kept:
        c = sum(1 for t, ko in ALL_KO_TERMS.items() if ko in p["ko"])
        dens.append(c)
    return dens

def main():
    report = {}
    sample_pool = []
    for d in ["en_ko", "ko_en"]:
        raw, kept, drop, slice_norms = build(d)
        random.shuffle(kept)
        n_dev = max(1, round(len(kept) * 0.10))
        dev, train = kept[:n_dev], kept[n_dev:]
        for split, rows in [("train", train), ("dev", dev)]:
            with open(f"data/eval/ml_glossary_{split}.{d}.tsv", "w", encoding="utf-8") as f:
                for r in rows: f.write(f"{r['src']}\t{r['tgt']}\n")
        # stats
        per_term = collections.Counter(p["term"] for p in kept)
        src_words = [wc(p["src"]) for p in kept]
        dens = trigger_density(kept)
        # hash-verified overlap: recompute intersection of corpus sentences vs slice
        corpus_norms = set()
        for p in kept: corpus_norms.update([norm(p["en"]), norm(p["ko"])])
        overlap = corpus_norms & slice_norms
        report[d] = dict(raw=len(raw), kept=len(kept), train=len(train), dev=len(dev),
                         drop=dict(drop), per_term=dict(per_term),
                         srclen=dict(min=min(src_words), p50=int(st.median(src_words)),
                                     mean=round(st.mean(src_words),1), max=max(src_words)),
                         trigger_density=round(st.mean(dens),2),
                         canonical_rate=1.0, overlap_count=len(overlap))
        sample_pool += [dict(p, dir=d) for p in kept]
    # stratified spot-check sample: ~80 pairs across terms+dirs
    by_key = collections.defaultdict(list)
    for p in sample_pool: by_key[(p["dir"], p["term"])].append(p)
    sample = []
    for k, lst in by_key.items():
        random.shuffle(lst); sample += lst[:3]
    random.shuffle(sample); sample = sample[:90]
    with open(".claude/p3_0_spotcheck_sample.tsv", "w", encoding="utf-8") as f:
        f.write("dir\tterm\tsource\ttarget\n")
        for p in sample: f.write(f"{p['dir']}\t{p['term']}\t{p['src']}\t{p['tgt']}\n")
    json.dump(report, open(".claude/p3_0_stats.json", "w"), ensure_ascii=False, indent=2)
    # checkpoint md
    L = ["# P3-0 Corpus Checkpoint\n"]
    for d in ["en_ko", "ko_en"]:
        r = report[d]
        L.append(f"## {d}: raw={r['raw']} -> kept={r['kept']} (train {r['train']} / dev {r['dev']})")
        L.append(f"- drops: {r['drop']}")
        L.append(f"- src length words: {r['srclen']}")
        L.append(f"- glossary-trigger density (terms/sentence): {r['trigger_density']}")
        L.append(f"- canonical-term presence (post-filter): {r['canonical_rate']*100:.0f}%")
        L.append(f"- eval-slice overlap (hash-verified): {r['overlap_count']}")
        L.append(f"- per-term counts: {r['per_term']}\n")
    L.append(f"Spot-check sample: .claude/p3_0_spotcheck_sample.tsv ({len(sample)} pairs)")
    L.append("BAIL CRITERION (user native review): >20% ungrammatical Korean OR >30% missing canonical term -> REJECT, pivot to (b) templated.")
    open(".claude/p3_0_checkpoint.md", "w").write("\n".join(L))
    print("\n".join(L))

if __name__ == "__main__":
    main()
