"""P3-1 LoRA Evaluation — compare base NLLB vs LoRA adapter on:
1. ML glossary slice (146 each direction) — canonical adherence
2. FLORES devtest (1012 each direction) — BLEU regression check

Reuses P2-2 methodology for apples-to-apples comparison.
Saves per-segment dumps and summary for reviewer particle check.

Usage:
    PYTHONPATH=$PWD ./.translator/bin/python scripts/p3_1_lora_eval.py
"""
from __future__ import annotations

import json
import os
import re
import resource
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from loguru import logger
from src.utils.metrics import compute_bleu

# ---- Paths ----
GLOSSARY_PATH     = REPO / "data/glossaries/ml-conference-en-ko.json"
SLICE_EN_KO       = REPO / "data/eval/ml_glossary_slice_en_ko.tsv"
SLICE_KO_EN       = REPO / "data/eval/ml_glossary_slice_ko_en.tsv"
FLORES_EN_KO      = REPO / "data/eval/flores_devtest_en_ko.tsv"
FLORES_KO_EN      = REPO / "data/eval/flores_devtest_ko_en.tsv"
ADAPTER_DIR       = REPO / "reports/p3-1_lora/adapter"
REPORT_DIR        = REPO / "reports/p3-1_lora"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG_EN = "eng_Latn"
TGT_LANG_KO = "kor_Hang"
SRC_LANG_KO = "kor_Hang"
TGT_LANG_EN = "eng_Latn"
MAX_NEW_TOKENS = 128
NUM_BEAMS = 1   # greedy decode — matches P2-2 methodology for apples-to-apples comparison


def load_tsv(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
    return pairs


def load_glossary(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["entries"]


def translate_batch(
    model,
    tokenizer,
    texts: list[str],
    src_lang: str,
    tgt_lang: str,
    device: torch.device,
    num_beams: int = NUM_BEAMS,
) -> list[str]:
    """Translate a list of source texts."""
    tokenizer.src_lang = src_lang
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        generated = model.generate(
            **enc,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=num_beams,
        )

    outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return outputs


def check_term_in_output(expected_term: str, output: str, case_sensitive: bool = False) -> bool:
    if case_sensitive:
        return expected_term in output
    return expected_term.lower() in output.lower()


def has_particle_concern(output_ko: str) -> bool:
    if len(output_ko.strip()) < 3:
        return True
    hangul_count = sum(1 for c in output_ko if '가' <= c <= '힣' or 'ᄀ' <= c <= 'ᇿ')
    if hangul_count == 0 and len(output_ko) > 5:
        return True
    return False


def run_slice_eval(
    base_model,
    adapter_model,
    tokenizer,
    pairs: list[tuple[str, str]],
    entries: list[dict],
    src_lang: str,
    tgt_lang: str,
    direction: str,
    device: torch.device,
    batch_size: int = 8,
) -> dict:
    """Run slice eval for one direction, both base and adapter."""
    logger.info(f"Slice eval: {direction} — {len(pairs)} segments")

    srcs = [s for s, r in pairs]
    refs = [r for s, r in pairs]

    # Translate all with base model
    logger.info(f"  Translating with BASE model...")
    base_outputs = []
    for i in range(0, len(srcs), batch_size):
        batch = srcs[i:i+batch_size]
        outs = translate_batch(base_model, tokenizer, batch, src_lang, tgt_lang, device)
        base_outputs.extend(outs)
        if (i // batch_size + 1) % 5 == 0:
            logger.info(f"    base: {i+len(batch)}/{len(srcs)}")

    # Translate all with adapter model
    logger.info(f"  Translating with ADAPTER model...")
    adapter_outputs = []
    for i in range(0, len(srcs), batch_size):
        batch = srcs[i:i+batch_size]
        outs = translate_batch(adapter_model, tokenizer, batch, src_lang, tgt_lang, device)
        adapter_outputs.extend(outs)
        if (i // batch_size + 1) % 5 == 0:
            logger.info(f"    adapter: {i+len(batch)}/{len(srcs)}")

    # BLEU
    if tgt_lang == TGT_LANG_KO:
        base_bleu, base_tok = compute_bleu(base_outputs, refs, tokenize="ko-mecab")
        adapter_bleu, adapter_tok = compute_bleu(adapter_outputs, refs, tokenize="ko-mecab")
        assert base_tok == "ko-mecab", f"ABORT: ko-mecab fallback on base: {base_tok}"
        assert adapter_tok == "ko-mecab", f"ABORT: ko-mecab fallback on adapter: {adapter_tok}"
    else:
        base_bleu, base_tok = compute_bleu(base_outputs, refs, tokenize="13a")
        adapter_bleu, adapter_tok = compute_bleu(adapter_outputs, refs, tokenize="13a")
        assert base_tok == "13a", f"ABORT: 13a fallback on base: {base_tok}"
        assert adapter_tok == "13a", f"ABORT: 13a fallback on adapter: {adapter_tok}"

    logger.info(f"  Slice BLEU — base: {base_bleu:.2f}, adapter: {adapter_bleu:.2f}")

    # Term recall per-segment
    per_term_base   = [{"triggered": 0, "correct": 0} for _ in entries]
    per_term_adapter = [{"triggered": 0, "correct": 0} for _ in entries]

    per_segment = []
    for seg_idx, (src_text, ref_text) in enumerate(pairs):
        base_out    = base_outputs[seg_idx]
        adapter_out = adapter_outputs[seg_idx]

        triggered_terms_base    = []
        triggered_terms_adapter = []

        for entry_idx, entry in enumerate(entries):
            if direction == "en_ko":
                trigger_term = entry["src"]
                expected_output_term = entry["tgt"]
                pattern = r"\b" + re.escape(trigger_term.lower()) + r"\b"
                triggered = bool(re.search(pattern, src_text.lower()))
            else:
                trigger_term = entry["tgt"]
                expected_output_term = entry["src"]
                triggered = trigger_term in src_text

            if triggered:
                per_term_base[entry_idx]["triggered"] += 1
                per_term_adapter[entry_idx]["triggered"] += 1

                if entry.get("dnt", False):
                    found_base    = check_term_in_output(entry["src"], base_out, case_sensitive=True)
                    found_adapter = check_term_in_output(entry["src"], adapter_out, case_sensitive=True)
                else:
                    if direction == "en_ko":
                        found_base    = check_term_in_output(expected_output_term, base_out, case_sensitive=True)
                        found_adapter = check_term_in_output(expected_output_term, adapter_out, case_sensitive=True)
                    else:
                        found_base    = check_term_in_output(expected_output_term, base_out, case_sensitive=False)
                        found_adapter = check_term_in_output(expected_output_term, adapter_out, case_sensitive=False)

                if found_base:
                    per_term_base[entry_idx]["correct"] += 1
                if found_adapter:
                    per_term_adapter[entry_idx]["correct"] += 1

                triggered_terms_base.append({
                    "entry_src": entry["src"],
                    "entry_tgt": entry["tgt"],
                    "dnt": entry.get("dnt", False),
                    "expected_target": expected_output_term,
                    "found": found_base,
                })
                triggered_terms_adapter.append({
                    "entry_src": entry["src"],
                    "entry_tgt": entry["tgt"],
                    "dnt": entry.get("dnt", False),
                    "expected_target": expected_output_term,
                    "found": found_adapter,
                })

        particle_concern_base    = has_particle_concern(base_out)    if tgt_lang == TGT_LANG_KO else False
        particle_concern_adapter = has_particle_concern(adapter_out) if tgt_lang == TGT_LANG_KO else False

        per_segment.append({
            "seg_idx": seg_idx,
            "src": src_text,
            "reference": ref_text,
            "base_out": base_out,
            "adapter_out": adapter_out,
            "triggered_terms_base": triggered_terms_base,
            "triggered_terms_adapter": triggered_terms_adapter,
            "particle_concern_base": particle_concern_base,
            "particle_concern_adapter": particle_concern_adapter,
        })

    # Aggregate term recall
    def aggregate_term_stats(per_term_stats, entries):
        total_triggered = sum(t["triggered"] for t in per_term_stats)
        total_correct   = sum(t["correct"]   for t in per_term_stats)

        # Separate DNT vs non-DNT
        dnt_triggered   = sum(per_term_stats[i]["triggered"] for i, e in enumerate(entries) if e.get("dnt", False))
        dnt_correct     = sum(per_term_stats[i]["correct"]   for i, e in enumerate(entries) if e.get("dnt", False))
        nondnt_triggered = sum(per_term_stats[i]["triggered"] for i, e in enumerate(entries) if not e.get("dnt", False))
        nondnt_correct   = sum(per_term_stats[i]["correct"]   for i, e in enumerate(entries) if not e.get("dnt", False))

        per_term_list = []
        for entry_idx, entry in enumerate(entries):
            t = per_term_stats[entry_idx]
            recall = (t["correct"] / t["triggered"]) if t["triggered"] > 0 else None
            per_term_list.append({
                "src": entry["src"],
                "tgt": entry["tgt"],
                "dnt": entry.get("dnt", False),
                "triggered": t["triggered"],
                "correct": t["correct"],
                "recall": round(recall, 4) if recall is not None else None,
            })

        return {
            "total_triggered": total_triggered,
            "total_correct": total_correct,
            "overall_recall": round(total_correct / total_triggered, 4) if total_triggered > 0 else 0.0,
            "dnt_triggered": dnt_triggered,
            "dnt_correct": dnt_correct,
            "dnt_recall": round(dnt_correct / dnt_triggered, 4) if dnt_triggered > 0 else 0.0,
            "nondnt_triggered": nondnt_triggered,
            "nondnt_correct": nondnt_correct,
            "nondnt_recall": round(nondnt_correct / nondnt_triggered, 4) if nondnt_triggered > 0 else 0.0,
            "per_term": per_term_list,
        }

    base_stats    = aggregate_term_stats(per_term_base,    entries)
    adapter_stats = aggregate_term_stats(per_term_adapter, entries)

    logger.info(f"  Base    overall_recall={base_stats['overall_recall']:.4f} "
                f"({base_stats['total_correct']}/{base_stats['total_triggered']}), "
                f"non-DNT={base_stats['nondnt_recall']:.4f} "
                f"({base_stats['nondnt_correct']}/{base_stats['nondnt_triggered']})")
    logger.info(f"  Adapter overall_recall={adapter_stats['overall_recall']:.4f} "
                f"({adapter_stats['total_correct']}/{adapter_stats['total_triggered']}), "
                f"non-DNT={adapter_stats['nondnt_recall']:.4f} "
                f"({adapter_stats['nondnt_correct']}/{adapter_stats['nondnt_triggered']})")

    return {
        "direction": direction,
        "n_segments": len(pairs),
        "bleu": {
            "base": round(base_bleu, 2),
            "adapter": round(adapter_bleu, 2),
            "regression": round(base_bleu - adapter_bleu, 2),
            "tokenize_used": base_tok,
        },
        "base": base_stats,
        "adapter": adapter_stats,
        "per_segment": per_segment,
    }


def run_flores_eval(
    base_model,
    adapter_model,
    tokenizer,
    pairs: list[tuple[str, str]],
    src_lang: str,
    tgt_lang: str,
    direction: str,
    device: torch.device,
    batch_size: int = 8,
) -> dict:
    """FLORES BLEU regression check — translate all, compute BLEU for base and adapter."""
    logger.info(f"FLORES eval: {direction} — {len(pairs)} segments")

    srcs = [s for s, r in pairs]
    refs = [r for s, r in pairs]

    logger.info(f"  Translating FLORES with BASE model...")
    base_outputs = []
    for i in range(0, len(srcs), batch_size):
        batch = srcs[i:i+batch_size]
        outs = translate_batch(base_model, tokenizer, batch, src_lang, tgt_lang, device)
        base_outputs.extend(outs)
        if (i // batch_size + 1) % 20 == 0:
            logger.info(f"    base: {i+len(batch)}/{len(srcs)}")

    logger.info(f"  Translating FLORES with ADAPTER model...")
    adapter_outputs = []
    for i in range(0, len(srcs), batch_size):
        batch = srcs[i:i+batch_size]
        outs = translate_batch(adapter_model, tokenizer, batch, src_lang, tgt_lang, device)
        adapter_outputs.extend(outs)
        if (i // batch_size + 1) % 20 == 0:
            logger.info(f"    adapter: {i+len(batch)}/{len(srcs)}")

    if tgt_lang == TGT_LANG_KO:
        base_bleu, base_tok = compute_bleu(base_outputs, refs, tokenize="ko-mecab")
        adapter_bleu, adapter_tok = compute_bleu(adapter_outputs, refs, tokenize="ko-mecab")
        assert base_tok == "ko-mecab", f"ABORT: ko-mecab fallback on base FLORES: {base_tok}"
        assert adapter_tok == "ko-mecab", f"ABORT: ko-mecab fallback on adapter FLORES: {adapter_tok}"
    else:
        base_bleu, base_tok = compute_bleu(base_outputs, refs, tokenize="13a")
        adapter_bleu, adapter_tok = compute_bleu(adapter_outputs, refs, tokenize="13a")
        assert base_tok == "13a", f"ABORT: 13a fallback on base FLORES: {base_tok}"
        assert adapter_tok == "13a", f"ABORT: 13a fallback on adapter FLORES: {adapter_tok}"

    regression = round(base_bleu - adapter_bleu, 2)
    logger.info(f"  FLORES BLEU — base: {base_bleu:.2f}, adapter: {adapter_bleu:.2f}, regression: {regression:.2f}")

    return {
        "direction": direction,
        "n_segments": len(pairs),
        "base_bleu": round(base_bleu, 2),
        "adapter_bleu": round(adapter_bleu, 2),
        "regression": regression,
        "tokenize_used": base_tok,
    }


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(REPORT_DIR / "eval.log", level="DEBUG")

    t_start = time.perf_counter()

    # ---- Load data ----
    logger.info("Loading eval data...")
    glossary_entries = load_glossary(GLOSSARY_PATH)
    pairs_slice_en_ko = load_tsv(SLICE_EN_KO)
    pairs_slice_ko_en = load_tsv(SLICE_KO_EN)
    pairs_flores_en_ko = load_tsv(FLORES_EN_KO)
    pairs_flores_ko_en = load_tsv(FLORES_KO_EN)

    assert len(pairs_slice_en_ko) == 146, f"Expected 146, got {len(pairs_slice_en_ko)}"
    assert len(pairs_slice_ko_en) == 146, f"Expected 146, got {len(pairs_slice_ko_en)}"
    assert len(pairs_flores_en_ko) == 1012, f"Expected 1012, got {len(pairs_flores_en_ko)}"
    assert len(pairs_flores_ko_en) == 1012, f"Expected 1012, got {len(pairs_flores_ko_en)}"
    logger.info(f"Slice: {len(pairs_slice_en_ko)} each direction, FLORES: {len(pairs_flores_en_ko)} each direction")

    device = torch.device("cuda")

    # ---- Load tokenizer ----
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    logger.info(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ---- Load BASE model (for baseline measurements) ----
    logger.info(f"Loading BASE model {MODEL_NAME}...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    ).to(device)
    base_model.eval()
    logger.info("Base model loaded.")

    vram_after_base_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    logger.info(f"VRAM after base model: {vram_after_base_mb:.0f} MB")

    # ---- Load ADAPTER model (separate instance so base_model is unaffected) ----
    # IMPORTANT: PeftModel.from_pretrained modifies the passed model in-place.
    # Load a SEPARATE base model instance for the adapter to avoid contaminating base_model.
    logger.info(f"Loading ADAPTER from {ADAPTER_DIR}...")
    from peft import PeftModel
    adapter_base = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    ).to(device)
    adapter_model = PeftModel.from_pretrained(
        adapter_base,
        str(ADAPTER_DIR),
    )
    adapter_model.eval()
    logger.info("Adapter model loaded.")

    vram_after_adapter_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    logger.info(f"VRAM after adapter: {vram_after_adapter_mb:.0f} MB")

    peak_vram_inference_mb = vram_after_adapter_mb

    # ---- Slice eval ----
    logger.info("\n=== SLICE EVAL ===")
    slice_en_ko = run_slice_eval(
        base_model, adapter_model, tokenizer,
        pairs_slice_en_ko, glossary_entries,
        SRC_LANG_EN, TGT_LANG_KO, "en_ko", device,
    )
    peak_vram_inference_mb = max(peak_vram_inference_mb, torch.cuda.max_memory_allocated(device) / 1024**2)

    slice_ko_en = run_slice_eval(
        base_model, adapter_model, tokenizer,
        pairs_slice_ko_en, glossary_entries,
        SRC_LANG_KO, TGT_LANG_EN, "ko_en", device,
    )
    peak_vram_inference_mb = max(peak_vram_inference_mb, torch.cuda.max_memory_allocated(device) / 1024**2)

    # ---- FLORES eval ----
    logger.info("\n=== FLORES EVAL ===")
    flores_en_ko = run_flores_eval(
        base_model, adapter_model, tokenizer,
        pairs_flores_en_ko, SRC_LANG_EN, TGT_LANG_KO, "en_ko", device,
    )
    peak_vram_inference_mb = max(peak_vram_inference_mb, torch.cuda.max_memory_allocated(device) / 1024**2)

    flores_ko_en = run_flores_eval(
        base_model, adapter_model, tokenizer,
        pairs_flores_ko_en, SRC_LANG_KO, TGT_LANG_EN, "ko_en", device,
    )
    peak_vram_inference_mb = max(peak_vram_inference_mb, torch.cuda.max_memory_allocated(device) / 1024**2)

    # ---- Memory ----
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    total_eval_time = time.perf_counter() - t_start
    logger.info(f"Peak VRAM (inference): {peak_vram_inference_mb:.0f} MB = {peak_vram_inference_mb/1024:.2f} GB")
    logger.info(f"ru_maxrss: {rss_kb} KB = {rss_kb/1024:.1f} MB = {rss_kb/1024/1024:.2f} GB")
    logger.info(f"Total eval time: {total_eval_time:.1f}s")

    # ---- Save results ----
    # Per-segment dump for reviewer
    slice_outputs_en_ko = [
        {
            "seg_idx": rec["seg_idx"],
            "src": rec["src"],
            "ref": rec["reference"],
            "base_out": rec["base_out"],
            "adapter_out": rec["adapter_out"],
            "triggered_terms": rec["triggered_terms_adapter"],  # adapter triggers for reviewer
            "particle_concern_adapter": rec["particle_concern_adapter"],
        }
        for rec in slice_en_ko["per_segment"]
    ]
    slice_outputs_ko_en = [
        {
            "seg_idx": rec["seg_idx"],
            "src": rec["src"],
            "ref": rec["reference"],
            "base_out": rec["base_out"],
            "adapter_out": rec["adapter_out"],
            "triggered_terms": rec["triggered_terms_adapter"],
            "particle_concern_adapter": rec["particle_concern_adapter"],
        }
        for rec in slice_ko_en["per_segment"]
    ]

    out_slice_en_ko = REPORT_DIR / "slice_outputs_en_ko.json"
    out_slice_ko_en = REPORT_DIR / "slice_outputs_ko_en.json"
    out_slice_en_ko.write_text(json.dumps(slice_outputs_en_ko, ensure_ascii=False, indent=2), encoding="utf-8")
    out_slice_ko_en.write_text(json.dumps(slice_outputs_ko_en, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Per-segment dumps saved: {out_slice_en_ko}, {out_slice_ko_en}")

    # Full result JSONs
    result_slice_en_ko = {
        "experiment": "P3-1_lora_eval",
        "direction": "en_ko",
        "dataset": "ml_glossary_slice",
        "n_segments": slice_en_ko["n_segments"],
        "bleu": slice_en_ko["bleu"],
        "base": slice_en_ko["base"],
        "adapter": slice_en_ko["adapter"],
        "per_segment": slice_en_ko["per_segment"],
    }
    result_slice_ko_en = {
        "experiment": "P3-1_lora_eval",
        "direction": "ko_en",
        "dataset": "ml_glossary_slice",
        "n_segments": slice_ko_en["n_segments"],
        "bleu": slice_ko_en["bleu"],
        "base": slice_ko_en["base"],
        "adapter": slice_ko_en["adapter"],
        "per_segment": slice_ko_en["per_segment"],
    }
    result_flores_en_ko = {
        "experiment": "P3-1_lora_eval",
        "direction": "en_ko",
        "dataset": "flores_devtest",
        "n_segments": flores_en_ko["n_segments"],
        "base_bleu": flores_en_ko["base_bleu"],
        "adapter_bleu": flores_en_ko["adapter_bleu"],
        "regression": flores_en_ko["regression"],
        "tokenize_used": flores_en_ko["tokenize_used"],
    }
    result_flores_ko_en = {
        "experiment": "P3-1_lora_eval",
        "direction": "ko_en",
        "dataset": "flores_devtest",
        "n_segments": flores_ko_en["n_segments"],
        "base_bleu": flores_ko_en["base_bleu"],
        "adapter_bleu": flores_ko_en["adapter_bleu"],
        "regression": flores_ko_en["regression"],
        "tokenize_used": flores_ko_en["tokenize_used"],
    }

    (REPORT_DIR / "result_slice_en_ko.json").write_text(
        json.dumps(result_slice_en_ko, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "result_slice_ko_en.json").write_text(
        json.dumps(result_slice_ko_en, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "result_flores_en_ko.json").write_text(
        json.dumps(result_flores_en_ko, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "result_flores_ko_en.json").write_text(
        json.dumps(result_flores_ko_en, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("All result JSONs saved.")

    # ---- Write summary ----
    write_summary(
        slice_en_ko, slice_ko_en,
        flores_en_ko, flores_ko_en,
        peak_vram_inference_mb, rss_kb,
        total_eval_time,
        REPORT_DIR,
    )
    logger.info("Summary written. Eval complete.")


def write_summary(
    slice_en_ko, slice_ko_en,
    flores_en_ko, flores_ko_en,
    peak_vram_inference_mb: float,
    rss_kb: int,
    eval_time_s: float,
    report_dir: Path,
):
    """Write summary.md with all headline numbers grep-able in result JSON files."""
    # Load training metadata if exists
    meta_path = report_dir / "training_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    lines = []
    lines.append("# P3-1 LoRA Adaptation — Experiment Summary")
    lines.append("")
    lines.append("Experiment: LoRA adapter on NLLB-200-distilled-600M for canonical terminology adherence.")
    lines.append("P2-2 baseline: non-DNT recall en->ko 20.7% / ko->en 31.5%.")
    lines.append("")

    # Adapter config from metadata
    if meta:
        lc = meta.get("lora_config", {})
        tr = meta.get("training", {})
        par = meta.get("params", {})
        lines.append("## Adapter Configuration")
        lines.append("")
        lines.append(f"- Base model: {meta.get('model', MODEL_NAME)}")
        lines.append(f"- LoRA rank: {lc.get('rank', 16)}")
        lines.append(f"- LoRA alpha: {lc.get('lora_alpha', 32)}")
        lines.append(f"- Target modules: {lc.get('target_modules', ['q_proj','k_proj','v_proj','out_proj','fc1','fc2'])}")
        lines.append(f"- LoRA dropout: {lc.get('lora_dropout', 0.05)}")
        lines.append(f"- Trainable params: {par.get('trainable', 'N/A'):,} ({par.get('trainable_pct', 'N/A')}%)")
        lines.append(f"- Total params: {par.get('total', 'N/A'):,}")
        lines.append(f"- LR: {tr.get('lr', 2e-4)}, warmup_steps: {tr.get('warmup_steps', 100)}")
        lines.append(f"- Epochs run: {tr.get('n_epochs_run', 'N/A')}, chosen_epoch: {tr.get('chosen_epoch', 'N/A')}")
        lines.append(f"- Effective batch: {tr.get('effective_batch', 16)} (per_step={tr.get('batch_size_per_step',4)} x grad_accum={tr.get('grad_accum',4)})")
        lines.append(f"- Train losses per epoch: {tr.get('train_losses', [])}")
        lines.append(f"- Dev losses per epoch: {tr.get('dev_losses', [])}")
        lines.append(f"- Best dev loss: {tr.get('best_dev_loss', 'N/A')}")
        lines.append(f"- Total optimizer steps: {tr.get('total_optimizer_steps', 'N/A')}")
        lines.append(f"- Training time: {tr.get('total_train_time_s', 'N/A')}s")
        lines.append(f"- Peak VRAM (train): {meta.get('peak_vram_train_mb', 'N/A')} MB = {meta.get('peak_vram_train_mb', 0)/1024:.2f} GB")
        lines.append(f"- ru_maxrss (train peak RSS): {meta.get('ru_maxrss_kb', 'N/A')} KB")
        lines.append("")

    lines.append("## Headline Numbers (N segments shown per row)")
    lines.append("")
    lines.append("### Canonical Non-DNT Recall (ML Glossary Slice, N=146 per direction)")
    lines.append("")

    b_en = slice_en_ko["base"]
    a_en = slice_en_ko["adapter"]
    b_ko = slice_ko_en["base"]
    a_ko = slice_ko_en["adapter"]

    en_base_nondnt_pct    = b_en["nondnt_recall"] * 100
    en_adapter_nondnt_pct = a_en["nondnt_recall"] * 100
    ko_base_nondnt_pct    = b_ko["nondnt_recall"] * 100
    ko_adapter_nondnt_pct = a_ko["nondnt_recall"] * 100
    en_delta_pp = en_adapter_nondnt_pct - en_base_nondnt_pct
    ko_delta_pp = ko_adapter_nondnt_pct - ko_base_nondnt_pct

    lines.append(f"| Metric | en->ko | ko->en |")
    lines.append(f"|--------|--------|--------|")
    lines.append(f"| Base non-DNT recall | {en_base_nondnt_pct:.1f}% ({b_en['nondnt_correct']}/{b_en['nondnt_triggered']}) | {ko_base_nondnt_pct:.1f}% ({b_ko['nondnt_correct']}/{b_ko['nondnt_triggered']}) |")
    lines.append(f"| Adapter non-DNT recall | {en_adapter_nondnt_pct:.1f}% ({a_en['nondnt_correct']}/{a_en['nondnt_triggered']}) | {ko_adapter_nondnt_pct:.1f}% ({a_ko['nondnt_correct']}/{a_ko['nondnt_triggered']}) |")
    lines.append(f"| Delta (pp) | {en_delta_pp:+.1f} pp | {ko_delta_pp:+.1f} pp |")
    lines.append(f"| Base overall recall | {b_en['overall_recall']*100:.1f}% ({b_en['total_correct']}/{b_en['total_triggered']}) | {b_ko['overall_recall']*100:.1f}% ({b_ko['total_correct']}/{b_ko['total_triggered']}) |")
    lines.append(f"| Adapter overall recall | {a_en['overall_recall']*100:.1f}% ({a_en['total_correct']}/{a_en['total_triggered']}) | {a_ko['overall_recall']*100:.1f}% ({a_ko['total_correct']}/{a_ko['total_triggered']}) |")
    lines.append("")

    lines.append("**P2-2 baseline (reference):** non-DNT en->ko=20.7% (39/188), ko->en=31.5% (63/200)")
    lines.append("")

    lines.append("### Pass Gate Assessment — Canonical Adherence")
    gate_en = en_adapter_nondnt_pct >= 35.0
    gate_ko = ko_adapter_nondnt_pct >= 45.0
    lines.append(f"- en->ko non-DNT >= 35%: {en_adapter_nondnt_pct:.1f}% => {'PASS' if gate_en else 'FAIL'}")
    lines.append(f"- ko->en non-DNT >= 45%: {ko_adapter_nondnt_pct:.1f}% => {'PASS' if gate_ko else 'FAIL'}")
    lines.append("")

    lines.append("### Slice BLEU (ML Glossary Slice)")
    lines.append("")
    lines.append(f"| Metric | en->ko (ko-mecab) | ko->en (13a) |")
    lines.append(f"|--------|--------|--------|")
    lines.append(f"| Base BLEU | {slice_en_ko['bleu']['base']:.2f} | {slice_ko_en['bleu']['base']:.2f} |")
    lines.append(f"| Adapter BLEU | {slice_en_ko['bleu']['adapter']:.2f} | {slice_ko_en['bleu']['adapter']:.2f} |")
    lines.append(f"| Regression | {slice_en_ko['bleu']['regression']:.2f} | {slice_ko_en['bleu']['regression']:.2f} |")
    lines.append("")

    lines.append("### FLORES BLEU Regression (N=1012 per direction)")
    lines.append("")
    lines.append(f"| Metric | en->ko (ko-mecab) | ko->en (13a) |")
    lines.append(f"|--------|--------|--------|")
    lines.append(f"| Base BLEU | {flores_en_ko['base_bleu']:.2f} | {flores_ko_en['base_bleu']:.2f} |")
    lines.append(f"| Adapter BLEU | {flores_en_ko['adapter_bleu']:.2f} | {flores_ko_en['adapter_bleu']:.2f} |")
    lines.append(f"| Regression | {flores_en_ko['regression']:.2f} | {flores_ko_en['regression']:.2f} |")
    lines.append("")

    flores_gate_en = flores_en_ko["regression"] <= 0.5
    flores_gate_ko = flores_ko_en["regression"] <= 0.5
    lines.append("### Pass Gate Assessment — FLORES Regression")
    lines.append(f"- FLORES en->ko regression <= 0.5: {flores_en_ko['regression']:.2f} => {'PASS' if flores_gate_en else 'FAIL'}")
    lines.append(f"- FLORES ko->en regression <= 0.5: {flores_ko_en['regression']:.2f} => {'PASS' if flores_gate_ko else 'FAIL'}")
    lines.append("")

    all_pass = gate_en and gate_ko and flores_gate_en and flores_gate_ko
    lines.append(f"## Overall Verdict: {'PASS' if all_pass else 'FAIL'}")
    lines.append("")
    if all_pass:
        lines.append("All four pass gates met. Adapter lifts canonical adherence above baseline while preserving FLORES quality.")
    else:
        failures = []
        if not gate_en: failures.append(f"en->ko non-DNT {en_adapter_nondnt_pct:.1f}% < 35% gate")
        if not gate_ko: failures.append(f"ko->en non-DNT {ko_adapter_nondnt_pct:.1f}% < 45% gate")
        if not flores_gate_en: failures.append(f"FLORES en->ko regression {flores_en_ko['regression']:.2f} > 0.5 gate")
        if not flores_gate_ko: failures.append(f"FLORES ko->en regression {flores_ko_en['regression']:.2f} > 0.5 gate")
        lines.append(f"Failed gates: {', '.join(failures)}")
    lines.append("")

    lines.append("## Per-Term Recall Delta (en->ko)")
    lines.append("")
    lines.append("| Term (en) | Term (ko) | DNT | Base recall | Adapter recall | Delta (pp) |")
    lines.append("|-----------|-----------|-----|-------------|----------------|------------|")
    for bi, ai in zip(b_en["per_term"], a_en["per_term"]):
        assert bi["src"] == ai["src"]
        br = f"{bi['recall']*100:.1f}%" if bi["recall"] is not None else "N/A"
        ar = f"{ai['recall']*100:.1f}%" if ai["recall"] is not None else "N/A"
        if bi["recall"] is not None and ai["recall"] is not None:
            delta = f"{(ai['recall'] - bi['recall'])*100:+.1f} pp"
        else:
            delta = "N/A"
        dnt_flag = "YES" if bi["dnt"] else "no"
        lines.append(f"| {bi['src']} | {bi['tgt']} | {dnt_flag} | {br} ({bi['correct']}/{bi['triggered']}) | {ar} ({ai['correct']}/{ai['triggered']}) | {delta} |")

    lines.append("")
    lines.append("## Per-Term Recall Delta (ko->en)")
    lines.append("")
    lines.append("| Term (en) | Term (ko) | DNT | Base recall | Adapter recall | Delta (pp) |")
    lines.append("|-----------|-----------|-----|-------------|----------------|------------|")
    for bi, ai in zip(b_ko["per_term"], a_ko["per_term"]):
        assert bi["src"] == ai["src"]
        br = f"{bi['recall']*100:.1f}%" if bi["recall"] is not None else "N/A"
        ar = f"{ai['recall']*100:.1f}%" if ai["recall"] is not None else "N/A"
        if bi["recall"] is not None and ai["recall"] is not None:
            delta = f"{(ai['recall'] - bi['recall'])*100:+.1f} pp"
        else:
            delta = "N/A"
        dnt_flag = "YES" if bi["dnt"] else "no"
        lines.append(f"| {bi['src']} | {bi['tgt']} | {dnt_flag} | {br} ({bi['correct']}/{bi['triggered']}) | {ar} ({ai['correct']}/{ai['triggered']}) | {delta} |")

    lines.append("")
    lines.append("## Hardware / Memory")
    lines.append("")
    lines.append(f"- Peak VRAM (inference, torch.cuda.max_memory_allocated): {peak_vram_inference_mb:.0f} MB = {peak_vram_inference_mb/1024:.2f} GB")
    lines.append(f"- ru_maxrss (true peak RSS, eval process): {rss_kb} KB = {rss_kb/1024:.1f} MB = {rss_kb/1024/1024:.2f} GB")
    lines.append(f"- Eval time: {eval_time_s:.1f}s")
    lines.append("")

    lines.append("## Reviewer Particle Check Candidates")
    lines.append("")
    pc_en = [rec["seg_idx"] for rec in slice_en_ko["per_segment"] if rec.get("particle_concern_adapter")]
    pc_ko = [rec["seg_idx"] for rec in slice_ko_en["per_segment"] if rec.get("particle_concern_adapter")]
    lines.append(f"en->ko particle concern segments (adapter): {pc_en}")
    lines.append(f"ko->en particle concern segments (adapter): {pc_ko}")
    lines.append("")

    lines.append("## Reconciliation")
    lines.append("")
    lines.append("All headline numbers are grep-able in:")
    lines.append("  reports/p3-1_lora/result_slice_en_ko.json")
    lines.append("  reports/p3-1_lora/result_slice_ko_en.json")
    lines.append("  reports/p3-1_lora/result_flores_en_ko.json")
    lines.append("  reports/p3-1_lora/result_flores_ko_en.json")
    lines.append("  reports/p3-1_lora/training_metadata.json")
    lines.append("")
    lines.append("Per-segment dumps for reviewer:")
    lines.append("  reports/p3-1_lora/slice_outputs_en_ko.json")
    lines.append("  reports/p3-1_lora/slice_outputs_ko_en.json")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("# Train:")
    lines.append("PYTHONPATH=$PWD ./.translator/bin/python scripts/p3_1_lora_train.py")
    lines.append("# Eval:")
    lines.append("PYTHONPATH=$PWD ./.translator/bin/python scripts/p3_1_lora_eval.py")
    lines.append("```")

    summary_path = report_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
