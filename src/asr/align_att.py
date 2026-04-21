"""AlignAtt simultaneous policy (Papi et al., Interspeech 2023).

Given partial source and partial target, detect where to stop generating by
checking whether the most-attended source frame is behind a safety threshold.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlignAttConfig:
    start_seconds: float = 2.0
    num_frames: int = 20
    threshold_layers: str = "all"  # "all" | "last"


class AlignAttPolicy:
    """Decides READ (wait for more audio) vs WRITE (emit translation).

    TODO: implement attention-based stopping criterion.
    """

    def __init__(self, cfg: AlignAttConfig) -> None:
        self.cfg = cfg

    def should_write(
        self,
        *,
        attention_weights,  # [num_layers, num_heads, tgt_len, src_len]
        src_len: int,
        elapsed_seconds: float,
    ) -> bool:
        """Return True if safe to emit the next token."""
        if elapsed_seconds < self.cfg.start_seconds:
            return False
        # TODO: argmax over src frames for current target step,
        # check it's at least `num_frames` behind the latest frame.
        return True
