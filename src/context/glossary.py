"""Glossary + Do-Not-Translate list management.

Glossary format (JSON):
  {
    "src_lang": "en",
    "tgt_lang": "ko",
    "entries": [
      {"src": "LLM", "tgt": "LLM", "dnt": true},
      {"src": "inference", "tgt": "추론"}
    ]
  }
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GlossaryEntry:
    src: str
    tgt: str
    dnt: bool = False          # Do-Not-Translate — copy src as-is


@dataclass
class Glossary:
    entries: list[GlossaryEntry] = field(default_factory=list)
    src_lang: str = "en"
    tgt_lang: str = "en"

    # ------------------------------------------------------------------ I/O

    @classmethod
    def from_json(cls, path: str | Path) -> "Glossary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            src_lang=data.get("src_lang", "en"),
            tgt_lang=data.get("tgt_lang", "en"),
            entries=[GlossaryEntry(**e) for e in data.get("entries", [])],
        )

    @classmethod
    def empty(cls, src_lang: str = "en", tgt_lang: str = "en") -> "Glossary":
        return cls(entries=[], src_lang=src_lang, tgt_lang=tgt_lang)

    def add_entry(self, src: str, tgt: str, dnt: bool = False) -> None:
        self.entries.append(GlossaryEntry(src=src, tgt=tgt, dnt=dnt))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "src_lang": self.src_lang,
                    "tgt_lang": self.tgt_lang,
                    "entries": [
                        {"src": e.src, "tgt": e.tgt, "dnt": e.dnt}
                        for e in self.entries
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # ---------------------------------------------------------------- lookup

    def hits_for(self, source_text: str, max_entries: int = 30) -> list[GlossaryEntry]:
        """Return entries whose src term appears in source_text (case-insensitive)."""
        text_lower = source_text.lower()
        hits: list[GlossaryEntry] = []
        for entry in self.entries:
            pattern = r"\b" + re.escape(entry.src.lower()) + r"\b"
            if re.search(pattern, text_lower):
                hits.append(entry)
            if len(hits) >= max_entries:
                break
        return hits

    # ------------------------------------------------------- prompt helpers

    def to_prompt_block(self, hits: list[GlossaryEntry]) -> str:
        """Format relevant entries for the LLM prompt."""
        if not hits:
            return ""
        dnt_terms = [e.src for e in hits if e.dnt]
        trans_pairs = [(e.src, e.tgt) for e in hits if not e.dnt]

        lines: list[str] = []
        if trans_pairs:
            lines.append("Glossary (MUST respect these translations):")
            for src, tgt in trans_pairs:
                lines.append(f"  {src} → {tgt}")
        if dnt_terms:
            lines.append(f"Do-Not-Translate (keep as-is): {', '.join(dnt_terms)}")

        return "\n".join(lines)
