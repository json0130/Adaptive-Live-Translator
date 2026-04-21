"""Qwen2.5-7B-Instruct translator with KV-cache reuse (InfiniSST pattern).

Key ideas:
  - Format as multi-turn chat: each new source chunk is a new user turn,
    each translation is the assistant turn.
  - Reuse KV cache across turns — only encode the delta.
  - Sliding window: keep system prompt + last N turns pinned.
"""
from __future__ import annotations

from typing import AsyncIterator

from loguru import logger

from .base import TranslationChunk, Translator


class QwenTranslator(Translator):
    """Qwen2.5-7B-Instruct streaming translator."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.model_name = cfg["translator"]["model"]
        self.max_new_tokens = cfg["translator"]["max_new_tokens"]
        self.max_context_tokens = cfg["translator"]["max_context_tokens"]
        self._model = None
        self._tokenizer = None
        self._past_kv = None  # KV cache across turns

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        logger.info(f"Loading translator: {self.model_name}")
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self._model = AutoModelForCausalLM.from_pretrained(
        #     self.model_name, torch_dtype="bfloat16", device_map="cuda"
        # )
        self._model = "STUB"
        self._tokenizer = "STUB"

    def reset(self) -> None:
        self._past_kv = None

    async def translate_stream(
        self,
        src_chunks: AsyncIterator[str],
        *,
        src_lang: str,
        tgt_lang: str,
        system_prompt: str,
    ) -> AsyncIterator[TranslationChunk]:
        self._lazy_load()

        cumulative = ""
        async for src_text in src_chunks:
            # TODO: build chat message with reused KV cache
            # messages = [
            #     {"role": "system", "content": system_prompt},
            #     {"role": "user", "content": src_text},
            # ]
            # inputs = self._tokenizer.apply_chat_template(
            #     messages, add_generation_prompt=True, return_tensors="pt"
            # ).to("cuda")
            # out = self._model.generate(
            #     inputs, past_key_values=self._past_kv,
            #     max_new_tokens=self.max_new_tokens,
            #     do_sample=False, use_cache=True, return_dict_in_generate=True,
            # )
            # self._past_kv = out.past_key_values
            # delta = self._tokenizer.decode(out.sequences[0][inputs.shape[1]:])
            delta = "[stub translation]"
            cumulative += delta
            yield TranslationChunk(text=cumulative, delta=delta, is_final=True)
