"""Speaker profile — per-user JSON store.

Stored at: data/speaker_profiles/{speaker_id}.json

Schema:
  {
    "speaker_id": "alice",
    "display_name": "Alice Chen",
    "default_src_lang": "en",
    "default_tgt_lang": "ko",
    "register": "formal",
    "topic_summary": "Machine learning research",
    "glossary_entries": [
      {"src": "fine-tuning", "tgt": "파인튜닝", "dnt": false}
    ],
    "dnt_list": ["NVIDIA", "PyTorch"],
    "lora_adapter": "alice_whisper_v1",
    "corrections": [
      {"src": "inference", "wrong": "인퍼런스", "correct": "추론", "count": 3}
    ]
  }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger


@dataclass
class CorrectionEntry:
    src: str
    wrong: str
    correct: str
    count: int = 1


@dataclass
class SpeakerProfile:
    speaker_id: str
    display_name: str = ""
    default_src_lang: str = "en"
    default_tgt_lang: str = "ko"
    register: str = "formal"
    topic_summary: str = ""
    glossary_entries: list[dict] = field(default_factory=list)
    dnt_list: list[str] = field(default_factory=list)
    lora_adapter: str | None = None
    corrections: list[CorrectionEntry] = field(default_factory=list)

    # -------------------------------------------------------------- I/O

    @classmethod
    def from_json(cls, path: str | Path) -> "SpeakerProfile":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        corrections = [CorrectionEntry(**c) for c in data.pop("corrections", [])]
        return cls(**{k: v for k, v in data.items() if k != "corrections"},
                   corrections=corrections)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self._to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _to_dict(self) -> dict:
        d = {
            "speaker_id": self.speaker_id,
            "display_name": self.display_name,
            "default_src_lang": self.default_src_lang,
            "default_tgt_lang": self.default_tgt_lang,
            "register": self.register,
            "topic_summary": self.topic_summary,
            "glossary_entries": self.glossary_entries,
            "dnt_list": self.dnt_list,
            "corrections": [
                {"src": c.src, "wrong": c.wrong, "correct": c.correct, "count": c.count}
                for c in self.corrections
            ],
        }
        if self.lora_adapter:
            d["lora_adapter"] = self.lora_adapter
        return d

    # ---------------------------------------------------- correction tracking

    def record_correction(self, src: str, wrong: str, correct: str) -> None:
        for c in self.corrections:
            if c.src == src and c.wrong == wrong:
                c.count += 1
                return
        self.corrections.append(CorrectionEntry(src=src, wrong=wrong, correct=correct))

    def top_corrections(self, n: int = 10) -> list[CorrectionEntry]:
        return sorted(self.corrections, key=lambda c: c.count, reverse=True)[:n]


class SpeakerProfileStore:
    """Load / cache / save speaker profiles from a directory."""

    def __init__(self, profiles_dir: str | Path) -> None:
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, SpeakerProfile] = {}

    def get(self, speaker_id: str) -> SpeakerProfile | None:
        if speaker_id in self._cache:
            return self._cache[speaker_id]
        path = self.profiles_dir / f"{speaker_id}.json"
        if path.exists():
            profile = SpeakerProfile.from_json(path)
            self._cache[speaker_id] = profile
            return profile
        return None

    def get_or_create(self, speaker_id: str, **defaults) -> SpeakerProfile:
        profile = self.get(speaker_id)
        if profile is None:
            profile = SpeakerProfile(speaker_id=speaker_id, **defaults)
            self._cache[speaker_id] = profile
            logger.info(f"Created new speaker profile: {speaker_id}")
        return profile

    def save(self, profile: SpeakerProfile) -> None:
        path = self.profiles_dir / f"{profile.speaker_id}.json"
        profile.save(path)
        self._cache[profile.speaker_id] = profile

    def list_speakers(self) -> list[str]:
        return [p.stem for p in self.profiles_dir.glob("*.json")]
