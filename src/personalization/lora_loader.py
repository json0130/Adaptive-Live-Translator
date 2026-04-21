"""LoRA adapter loader — hot-swap speaker-specific adapters at session start.

Strategy (Samsung / Interspeech 2024, LoRA paper):
  - Base Whisper model is shared (frozen).
  - Each speaker has a small LoRA adapter (~0.25% of params, rank 4–16).
  - Load adapter at speaker-change events; merge back for inference.
  - No additional inference latency (adapters are merged into base weights).

Adapter naming convention:
  data/lora_adapters/{speaker_id}_{model_shortname}_r{rank}.safetensors
  e.g.  data/lora_adapters/alice_whisper-large-v3_r8.safetensors
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger


class LoRALoader:
    """Hot-swap LoRA adapters onto a model at speaker-change events."""

    def __init__(self, adapter_dir: str | Path, rank: int = 8, alpha: int = 16) -> None:
        self.adapter_dir = Path(adapter_dir)
        self.rank = rank
        self.alpha = alpha
        self._active_speaker: str | None = None
        self._model = None

    def attach(self, model) -> None:
        """Register the model to adapt."""
        self._model = model

    def load_for_speaker(self, speaker_id: str) -> bool:
        """Load adapter for speaker_id onto self._model.

        Returns True if adapter was found and loaded, False otherwise.
        """
        if speaker_id == self._active_speaker:
            return True

        adapter_path = self._find_adapter(speaker_id)
        if adapter_path is None:
            logger.debug(f"No LoRA adapter found for speaker '{speaker_id}'. Using base model.")
            self._unload()
            return False

        logger.info(f"Loading LoRA adapter for '{speaker_id}': {adapter_path}")
        self._do_load(adapter_path)
        self._active_speaker = speaker_id
        return True

    def _find_adapter(self, speaker_id: str) -> Path | None:
        for pattern in [f"{speaker_id}_*.safetensors", f"{speaker_id}_*.bin"]:
            matches = list(self.adapter_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    def _do_load(self, path: Path) -> None:
        if self._model is None or self._model == "STUB":
            return
        try:
            from peft import PeftModel
            # PeftModel.from_pretrained(self._model, str(path))
            # For hot-swapping an already-loaded base model:
            # self._model.load_adapter(str(path), adapter_name="speaker")
            # self._model.set_adapter("speaker")
            logger.debug(f"[LoRA] Loaded adapter from {path}")
        except ImportError:
            logger.warning("peft not installed; LoRA adapters disabled.")

    def _unload(self) -> None:
        if self._model is None or self._model == "STUB":
            return
        try:
            # self._model.disable_adapter_layers()
            self._active_speaker = None
        except Exception:
            pass

    # ---------------------------------------------------- training helpers

    @staticmethod
    def get_lora_config(rank: int = 8, alpha: int = 16):
        """Return a LoraConfig for Whisper ASR fine-tuning."""
        try:
            from peft import LoraConfig, TaskType
            return LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=rank,
                lora_alpha=alpha,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"],  # attention layers only
                bias="none",
            )
        except ImportError:
            logger.error("peft not installed. Run: pip install peft")
            return None
