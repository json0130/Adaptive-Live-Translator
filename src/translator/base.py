"""Abstract Translator interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class TranslationChunk:
    text: str          # cumulative translation so far
    delta: str         # new tokens since last chunk
    is_final: bool     # segment complete


class Translator(ABC):
    """A streaming translator that accepts source chunks and emits target chunks."""

    @abstractmethod
    async def translate_stream(
        self,
        src_chunks: AsyncIterator[str],
        *,
        src_lang: str,
        tgt_lang: str,
        system_prompt: str,
    ) -> AsyncIterator[TranslationChunk]:
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear KV cache between sessions."""
        ...
