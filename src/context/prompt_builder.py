"""PromptBuilder — assembles the full LLM system prompt from all context signals.

Prompt structure (see docs/prompt_template.md):

  [SYSTEM]
  You are a simultaneous interpreter translating {src} → {tgt}.
  Domain: {topic_summary}
  Speaker: {name}, register: {formal|informal}

  {glossary_block}

  {tm_block}

  [CONTEXT — previous segments]
  SRC: ...
  TGT: ...

The CURRENT PARTIAL input + partial translation is appended per-chunk by the
streaming loop (not here) so the KV cache can be reused across turns.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .glossary import Glossary, GlossaryEntry
from .translation_memory import TMEntry


@dataclass
class SessionContext:
    """All metadata known about the current translation session."""
    src_lang: str = "en"
    tgt_lang: str = "ko"
    topic_summary: str = ""
    speaker_name: str = "Unknown"
    register: str = "formal"            # "formal" | "informal"
    prev_src_segments: list[str] = field(default_factory=list)
    prev_tgt_segments: list[str] = field(default_factory=list)
    max_prev_segments: int = 5


class PromptBuilder:
    """Builds the system prompt fed to the translator LLM."""

    def __init__(self, cfg: dict) -> None:
        self.max_glossary_entries = cfg["context"]["glossary"]["max_entries_in_prompt"]
        self.max_prev_segments = cfg["context"]["rolling_context"]["prev_segments"]

    # ---------------------------------------------------------- main method

    def build_system_prompt(
        self,
        ctx: SessionContext,
        glossary_hits: list[GlossaryEntry],
        tm_hits: list[TMEntry],
    ) -> str:
        """Return the full system prompt string."""
        parts: list[str] = []

        # --- Header
        parts.append(self._header(ctx))

        # --- Glossary block
        glossary = Glossary(entries=glossary_hits, src_lang=ctx.src_lang, tgt_lang=ctx.tgt_lang)
        gloss_block = glossary.to_prompt_block(glossary_hits[: self.max_glossary_entries])
        if gloss_block:
            parts.append(gloss_block)

        # --- TM block
        if tm_hits:
            tm_lines = ["Translation examples (for style/terminology reference):"]
            for e in tm_hits:
                tm_lines.append(f"  SRC: {e.src}")
                tm_lines.append(f"  TGT: {e.tgt}")
            parts.append("\n".join(tm_lines))

        # --- Rolling context
        context_block = self._rolling_context(ctx)
        if context_block:
            parts.append(context_block)

        return "\n\n".join(parts)

    # ---------------------------------------------------------- helpers

    @staticmethod
    def _header(ctx: SessionContext) -> str:
        lines = [
            f"You are a simultaneous interpreter translating {ctx.src_lang.upper()} → {ctx.tgt_lang.upper()}.",
            f"Translate accurately and naturally. Emit only the translation — no explanations.",
            f"Register: {ctx.register}.",
        ]
        if ctx.topic_summary:
            lines.append(f"Domain/topic: {ctx.topic_summary}")
        if ctx.speaker_name and ctx.speaker_name != "Unknown":
            lines.append(f"Speaker: {ctx.speaker_name}")
        return "\n".join(lines)

    def _rolling_context(self, ctx: SessionContext) -> str:
        n = self.max_prev_segments
        prev_src = ctx.prev_src_segments[-n:]
        prev_tgt = ctx.prev_tgt_segments[-n:]
        if not prev_src:
            return ""
        lines = ["Previous segments (for discourse continuity):"]
        for s, t in zip(prev_src, prev_tgt):
            lines.append(f"  SRC: {s}")
            lines.append(f"  TGT: {t}")
        return "\n".join(lines)

    # ------------------------------------------- incremental user message

    @staticmethod
    def build_user_turn(src_partial: str, tgt_so_far: str = "") -> str:
        """Format the per-chunk user turn.

        The LLM sees: source partial → continues the target.
        """
        msg = f"Translate:\n{src_partial}"
        if tgt_so_far:
            msg += f"\n\nPartial translation so far (continue from here):\n{tgt_so_far}"
        return msg
