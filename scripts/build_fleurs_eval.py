#!/usr/bin/env python
"""Build frozen Fleurs audio eval set for en<->ko, paired by utterance ID.

Writes:
  data/eval/fleurs_en_us_test/   audio_*.wav + manifest.tsv (id, audio_path, text)
  data/eval/fleurs_ko_kr_test/   audio_*.wav + manifest.tsv (id, audio_path, text)
  data/eval/fleurs_pair_manifest.tsv   (utt_id \\t en_audio \\t ko_audio \\t en_text \\t ko_text)

The pair manifest contains exactly the utterance IDs that appear in BOTH
en_us and ko_kr Fleurs test splits — 270 sentences as of Fleurs v1.

Source: google/fleurs on Hugging Face (CC-BY-4.0).
Each language's test.tar.gz contains 16kHz WAVs; we extract them as-is.

This script is idempotent and refuses to overwrite a built test set —
delete the data/eval/fleurs_*_test directories manually to force a rebuild.

Usage:
    python3 scripts/build_fleurs_eval.py
"""
from __future__ import annotations

import csv
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download


LANGS = [("en_us", "english"), ("ko_kr", "korean")]


def _load_tsv(tsv_path: Path) -> dict[str, dict]:
    """Group rows by utterance ID. Multiple speakers per ID — keep the first."""
    rows: dict[str, dict] = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            utt_id, audio_filename, raw_text = parts[0], parts[1], parts[2]
            if utt_id in rows:
                continue
            rows[utt_id] = {
                "id": utt_id,
                "audio_filename": audio_filename,
                "text": raw_text.strip(),
            }
    return rows


def _materialize_lang(lang_code: str, out_dir: Path) -> dict[str, dict]:
    print(f"[fleurs] downloading metadata + audio for {lang_code}")
    tsv_path = Path(
        hf_hub_download(
            "google/fleurs",
            f"data/{lang_code}/test.tsv",
            repo_type="dataset",
        )
    )
    rows = _load_tsv(tsv_path)
    print(f"[fleurs] {lang_code}: {len(rows)} unique utterances")

    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Download and extract the audio tarball if we don't already have all files.
    needed_files = {r["audio_filename"] for r in rows.values()}
    missing = [f for f in needed_files if not (audio_dir / f).exists()]
    if missing:
        print(f"[fleurs] extracting {len(missing)} missing wavs from tarball")
        tar_path = Path(
            hf_hub_download(
                "google/fleurs",
                f"data/{lang_code}/audio/test.tar.gz",
                repo_type="dataset",
            )
        )
        with tarfile.open(tar_path, "r:gz") as tar:
            members = [m for m in tar.getmembers() if Path(m.name).name in needed_files]
            for m in members:
                m.name = Path(m.name).name  # flatten path
                tar.extract(m, audio_dir)
    return rows


def main():
    repo_root = Path(__file__).resolve().parent.parent
    eval_root = repo_root / "data" / "eval"
    pair_manifest = eval_root / "fleurs_pair_manifest.tsv"

    if pair_manifest.exists():
        print(f"[fleurs] Frozen Fleurs eval already built: {pair_manifest}")
        print("[fleurs] Refusing to overwrite. Delete it to force rebuild.")
        return

    lang_to_rows: dict[str, dict[str, dict]] = {}
    for code, _ in LANGS:
        out_dir = eval_root / f"fleurs_{code}_test"
        rows = _materialize_lang(code, out_dir)
        # Per-language manifest.
        with open(out_dir / "manifest.tsv", "w", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["id", "audio_path", "text"])
            for r in rows.values():
                w.writerow([r["id"], f"audio/{r['audio_filename']}", r["text"]])
        lang_to_rows[code] = rows

    # Paired manifest by utterance ID.
    common_ids = sorted(
        set(lang_to_rows["en_us"]) & set(lang_to_rows["ko_kr"]),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    print(f"[fleurs] paired ko<->en utterances: {len(common_ids)}")
    with open(pair_manifest, "w", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["id", "en_audio", "ko_audio", "en_text", "ko_text"])
        for uid in common_ids:
            en, ko = lang_to_rows["en_us"][uid], lang_to_rows["ko_kr"][uid]
            w.writerow([
                uid,
                f"fleurs_en_us_test/audio/{en['audio_filename']}",
                f"fleurs_ko_kr_test/audio/{ko['audio_filename']}",
                en["text"], ko["text"],
            ])
    print(f"[fleurs] wrote {pair_manifest}")
    print("[fleurs] FROZEN. Do not modify, retune, or re-split.")


if __name__ == "__main__":
    main()
