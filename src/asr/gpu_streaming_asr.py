"""GPU LocalAgreement-2 streaming ASR for P2-1.

Thin subclass of LocalAgreementASR that loads faster-whisper on CUDA
with int8_float16 compute type. All streaming logic (LA-2, confidence gate,
timing model) is inherited unchanged.

P2-1 config:
  model_id = "large-v3"
  device = "cuda"
  compute_type = "int8_float16"

Note: faster-whisper on CUDA uses CTranslate2 under the hood (not torch),
so torch.cuda.memory_allocated() will show 0 for ASR. Use nvidia-smi for
the whole-process VRAM reading (which includes the CTranslate2 allocation).
"""
from __future__ import annotations

from typing import Optional

from loguru import logger

from src.asr.streaming_local_agreement import LocalAgreementASR


class GpuLocalAgreementASR(LocalAgreementASR):
    """LocalAgreement-2 ASR on GPU (faster-whisper, int8_float16).

    Identical behavior to LocalAgreementASR except models load on CUDA.
    final_model_id is not used for P2-1 (single model, large-v3 is the
    accuracy model — no separate final-pass model needed).
    """

    def __init__(
        self,
        model_id: str = "large-v3",
        compute_type: str = "int8_float16",
        device: str = "cuda",
        initial_chunk_s: float = 1.0,
        step_s: float = 1.0,
        beam_size: int = 1,
        confidence_gate_threshold: Optional[float] = None,
    ) -> None:
        # Pass cpu_threads=1 (irrelevant for CUDA but required by parent __init__)
        super().__init__(
            model_id=model_id,
            compute_type=compute_type,
            cpu_threads=1,
            initial_chunk_s=initial_chunk_s,
            step_s=step_s,
            beam_size=beam_size,
            confidence_gate_threshold=confidence_gate_threshold,
            final_model_id=None,  # large-v3 is the single high-accuracy model
        )
        self._device = device
        logger.info(
            f"GpuLocalAgreementASR: model={model_id}, device={device}, "
            f"compute_type={compute_type}, initial_chunk={initial_chunk_s}s, "
            f"step={step_s}s, beam={beam_size}, conf_gate={confidence_gate_threshold}"
        )

    def _ensure_loaded(self) -> None:
        """Load faster-whisper on CUDA (overrides parent's CPU load)."""
        if self._model is None:
            from faster_whisper import WhisperModel
            logger.info(
                f"Loading faster-whisper {self.model_id} on {self._device} "
                f"compute_type={self.compute_type}"
            )
            self._model = WhisperModel(
                self.model_id,
                device=self._device,
                compute_type=self.compute_type,
                # cpu_threads not used for CUDA; omit to avoid confusing CT2
            )
            logger.info("faster-whisper loaded on CUDA")

    def _ensure_final_model_loaded(self) -> None:
        """No final model for P2-1 — large-v3 is the single model."""
        pass  # final_model_id=None; parent already handles this
