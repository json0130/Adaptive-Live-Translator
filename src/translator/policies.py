"""Simultaneous translation policies: wait-k, LocalAgreement."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class LocalAgreementConfig:
    history_size: int = 2  # agreement across N consecutive partial outputs


class LocalAgreementPolicy:
    """LocalAgreement-2: commit the longest common prefix across the last
    N partial translations (Liu et al., 2020).
    """

    def __init__(self, cfg: LocalAgreementConfig) -> None:
        self.cfg = cfg
        self._history: deque[str] = deque(maxlen=cfg.history_size)
        self._committed: str = ""

    def update(self, partial: str) -> str:
        """Add a new partial, return newly committed prefix."""
        self._history.append(partial)
        if len(self._history) < self.cfg.history_size:
            return ""

        prefix = self._history[0]
        for p in list(self._history)[1:]:
            prefix = _common_prefix(prefix, p)

        new = prefix[len(self._committed):]
        self._committed = prefix
        return new

    def reset(self) -> None:
        self._history.clear()
        self._committed = ""


def _common_prefix(a: str, b: str) -> str:
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return a[:i]
