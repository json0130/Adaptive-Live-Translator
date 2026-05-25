#!/usr/bin/env python
"""Build the frozen eval TSVs from FLORES-200 devtest for en<->ko.

Writes:
  data/eval/flores_devtest_en_ko.tsv   (cols: src_en \\t ref_ko)
  data/eval/flores_devtest_ko_en.tsv   (cols: src_ko \\t ref_en)

These files are FROZEN for the entire research effort. The fixed eval
protocol pins these exact paths.

Source: official FLORES-200 release tarball from Facebook (CC-BY-SA-4.0).
1012 parallel sentences, identical across language pairs.

  https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz

We use the original tarball (not the HF gated mirror openlanguagedata/
flores_plus) so the build does not require an HF login.

Usage:
    python scripts/build_flores_eval.py
"""
from __future__ import annotations

import io
import tarfile
import urllib.request
from pathlib import Path

TARBALL_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
EN_MEMBER = "./flores200_dataset/devtest/eng_Latn.devtest"
KO_MEMBER = "./flores200_dataset/devtest/kor_Hang.devtest"


def _read_lines_from_tar(tar: tarfile.TarFile, member: str) -> list[str]:
    f = tar.extractfile(member)
    if f is None:
        raise RuntimeError(f"missing {member} in tarball")
    return f.read().decode("utf-8").splitlines()


def main():
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "data" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    en_ko_path = out_dir / "flores_devtest_en_ko.tsv"
    ko_en_path = out_dir / "flores_devtest_ko_en.tsv"

    if en_ko_path.exists() and ko_en_path.exists():
        print(f"[build_flores_eval] Already built: {en_ko_path}, {ko_en_path}")
        print("[build_flores_eval] Refusing to overwrite a frozen test set.")
        print("[build_flores_eval] Delete them manually if you really mean it.")
        return

    cache = repo_root / ".cache" / "flores200_dataset.tar.gz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        print(f"[build_flores_eval] Downloading {TARBALL_URL}")
        urllib.request.urlretrieve(TARBALL_URL, cache)
    else:
        print(f"[build_flores_eval] Using cached {cache}")

    with tarfile.open(cache, "r:gz") as tar:
        en_lines = _read_lines_from_tar(tar, EN_MEMBER)
        ko_lines = _read_lines_from_tar(tar, KO_MEMBER)

    assert len(en_lines) == len(ko_lines), (
        f"length mismatch: {len(en_lines)} vs {len(ko_lines)}"
    )
    n = len(en_lines)
    print(f"[build_flores_eval] {n} parallel sentences")

    def _clean(s: str) -> str:
        return s.strip().replace("\t", " ").replace("\n", " ")

    with open(en_ko_path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(f"{_clean(en_lines[i])}\t{_clean(ko_lines[i])}\n")

    with open(ko_en_path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(f"{_clean(ko_lines[i])}\t{_clean(en_lines[i])}\n")

    print(f"[build_flores_eval] Wrote {en_ko_path}")
    print(f"[build_flores_eval] Wrote {ko_en_path}")
    print("[build_flores_eval] FROZEN. Do not modify, retune, or re-split.")


if __name__ == "__main__":
    main()
