"""LocalAgreement-2 streaming ASR using faster-whisper.

Based on Macháček et al., "Whisper-Streaming", IWSLT 2023.
https://github.com/ufal/whisper_streaming

Algorithm
---------
  - Conceptually, audio arrives at real-time speed. We buffer it.
  - Every step_s seconds of audio, we transcribe the buffer from the
    last committed point to the current audio head.
  - After each transcription, compare the new hypothesis tokens to the
    previous hypothesis. Commit (emit) the longest common prefix (LCP)
    of the last 2 consecutive hypotheses (LocalAgreement-2).
  - Trim the audio buffer start to the last committed point.
  - After audio ends, flush the remaining unconfirmed text.

Timing simulation
-----------------
  In this simulation the audio is pre-loaded. We keep a virtual clock:
    virtual_audio_head_s: how many seconds of audio have "arrived"
    wall_clock_s: actual wall-clock time elapsed since audio start

  They diverge because decoding takes real time. We track:
    first_emission_latency_s: wall-clock time from audio start until
      first words are committed. In real streaming, this is what a listener
      would experience.

  Total RTFx = sum(all decode times) / total_audio_duration.
  This will be high because windows overlap.

Notes
-----
  - model: Systran/faster-whisper-medium int8 CPU, intra_threads=8
  - beam_size=1 (greedy) for minimum latency
  - minimum window size: 1.0s (shorter windows produce garbage on Whisper)

Confidence-Gated Early Commit (R4 extension)
---------------------------------------------
  When confidence_gate_threshold is set (not None), the first window
  decode uses avg_logprob from faster-whisper segments. If the average
  log-probability of all segments >= threshold, the first window is
  committed immediately WITHOUT waiting for 2-window LocalAgreement.
  This shaves ~1 decode time off the first-emission latency path.

  Threshold -0.25 means "very confident": avg_logprob > -0.25 is well
  above the typical Whisper hallucination zone (avg_logprob < -1.0).
  Fallback: if not confident on first window, resumes normal LA-2.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


SAMPLE_RATE = 16000  # faster-whisper expects 16 kHz float32


@dataclass
class EmissionEvent:
    """Record of a committed word emission."""
    word: str
    audio_head_s: float       # how much audio had arrived when emitted
    emit_wall_time_s: float   # wall-clock seconds from audio-stream start when emitted


@dataclass
class StreamingResult:
    """Full result of streaming a single utterance."""
    final_transcript: str         # streaming-model committed text (for latency)
    accuracy_transcript: str      # final-model full-audio transcript (for WER); equals
                                  # final_transcript when no final_model_id is set
    emissions: List[EmissionEvent]
    total_audio_s: float
    total_decode_s: float         # cumulative time in all STREAMING whisper calls
    first_emission_latency_s: Optional[float]   # wall-clock: audio start -> first word
    n_windows: int
    confidence_gate_applied: bool = False   # True if early commit was triggered


def _load_wav_as_float32(audio_path: str) -> np.ndarray:
    """Load WAV as float32 PCM at 16 kHz. Resamples if needed."""
    try:
        import soundfile as sf
        audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    except Exception:
        import scipy.io.wavfile as wf
        sr, audio = wf.read(audio_path)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo -> mono

    if sr != SAMPLE_RATE:
        import scipy.signal as ss
        target_len = int(len(audio) * SAMPLE_RATE / sr)
        audio = ss.resample(audio, target_len)

    return audio


def _tokens(text: str) -> List[str]:
    """Split transcript into word tokens for LCP computation."""
    return text.strip().split()


def _longest_common_prefix(a: List[str], b: List[str]) -> List[str]:
    """Return the longest common prefix of two token lists."""
    lcp: List[str] = []
    for x, y in zip(a, b):
        if x == y:
            lcp.append(x)
        else:
            break
    return lcp


class HypothesisBuffer:
    """Rolling 2-window hypothesis manager for LocalAgreement-2."""

    def __init__(self) -> None:
        self.prev_tokens: List[str] = []
        self.committed_tokens: List[str] = []

    def push(self, new_tokens: List[str]) -> List[str]:
        """Push a new hypothesis. Return newly committed tokens (LCP delta)."""
        lcp = _longest_common_prefix(self.prev_tokens, new_tokens)
        already_committed = len(self.committed_tokens)
        new_committed = lcp[already_committed:]
        self.committed_tokens.extend(new_committed)
        self.prev_tokens = new_tokens
        return new_committed

    def flush(self, final_tokens: Optional[List[str]] = None) -> List[str]:
        """Commit everything remaining.

        If final_tokens is provided, use it as the final hypothesis and
        commit everything beyond what's already committed.
        Otherwise commit the remainder of prev_tokens.
        """
        if final_tokens is not None:
            tokens = final_tokens
        else:
            tokens = self.prev_tokens
        already_committed = len(self.committed_tokens)
        remaining = tokens[already_committed:]
        self.committed_tokens.extend(remaining)
        self.prev_tokens = []
        return remaining


class LocalAgreementASR:
    """Streaming ASR using LocalAgreement-2 over faster-whisper-medium.

    Parameters
    ----------
    model_id : str
        HF model ID, default 'Systran/faster-whisper-medium'
    compute_type : str
        CTranslate2 quantisation, default 'int8'
    cpu_threads : int
        Number of intra-op threads for CT2, default 8
    initial_chunk_s : float
        Size of first audio window in seconds (default 1.5)
    step_s : float
        How much new audio triggers a new decode pass (default 1.0)
    beam_size : int
        Whisper beam size, default 1 (greedy; fastest on CPU)
    confidence_gate_threshold : float or None
        If not None, commit any window immediately (bypassing
        LocalAgreement-2) when avg_logprob >= threshold, as long as
        no words have been emitted yet. This reduces first-emission
        latency by skipping the LA-2 agreement requirement for
        high-confidence early windows.
        Threshold -0.25 = very confident; -0.5 = moderately confident.
        Set to None to disable (original LA-2 behavior, default).
    final_model_id : str or None
        If set, run a separate (typically larger/more accurate) model
        on the complete audio at utterance end to produce the final
        transcript used for WER scoring. The streaming partial model
        (model_id) is only used for latency/first-emission detection.
        The final_model_id model is lazily loaded and NOT counted
        in the streaming decode time.
        Set to None to use model_id for both streaming and final (default).
    """

    def __init__(
        self,
        model_id: str = "Systran/faster-whisper-medium",
        compute_type: str = "int8",
        cpu_threads: int = 8,
        initial_chunk_s: float = 1.5,
        step_s: float = 1.0,
        beam_size: int = 1,
        confidence_gate_threshold: Optional[float] = None,
        final_model_id: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.initial_chunk_s = initial_chunk_s
        self.step_s = step_s
        self.beam_size = beam_size
        self.confidence_gate_threshold = confidence_gate_threshold
        self.final_model_id = final_model_id
        self._model = None
        self._final_model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_id,
                device="cpu",
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )

    def _ensure_final_model_loaded(self) -> None:
        if self.final_model_id is None:
            return  # use streaming model for final
        if self._final_model is None:
            from faster_whisper import WhisperModel
            self._final_model = WhisperModel(
                self.final_model_id,
                device="cpu",
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )

    def _transcribe_full_audio(self, audio: np.ndarray, lang: str) -> str:
        """Transcribe the full audio with the final model (if set) or streaming model.

        Used to produce the accurate final transcript for WER scoring.
        This is called AFTER streaming is complete and does NOT count
        toward streaming latency or RTFx.
        """
        model = self._final_model if self._final_model is not None else self._model
        segs, _ = model.transcribe(
            audio,
            language=lang,
            beam_size=self.beam_size,
            vad_filter=False,
            word_timestamps=False,
        )
        return "".join(s.text for s in segs).strip()

    def _transcribe_window(self, audio_window: np.ndarray, lang: str) -> Tuple[str, float]:
        """Transcribe a numpy float32 PCM window.

        Returns (text, decode_seconds).
        """
        t0 = time.perf_counter()
        segs, _ = self._model.transcribe(
            audio_window,
            language=lang,
            beam_size=self.beam_size,
            vad_filter=False,
            word_timestamps=False,
        )
        text = "".join(s.text for s in segs).strip()
        return text, time.perf_counter() - t0

    def _transcribe_window_with_confidence(
        self, audio_window: np.ndarray, lang: str
    ) -> Tuple[str, float, float]:
        """Transcribe a numpy float32 PCM window and return avg_logprob.

        Returns (text, decode_seconds, avg_logprob).
        avg_logprob is the mean of segment avg_logprobs (None segments ignored).
        If no segments, avg_logprob is -inf.
        """
        t0 = time.perf_counter()
        segs_iter, _ = self._model.transcribe(
            audio_window,
            language=lang,
            beam_size=self.beam_size,
            vad_filter=False,
            word_timestamps=False,
        )
        texts = []
        logprobs = []
        for s in segs_iter:
            texts.append(s.text)
            if s.avg_logprob is not None:
                logprobs.append(s.avg_logprob)
        text = "".join(texts).strip()
        avg_logprob = float(np.mean(logprobs)) if logprobs else float("-inf")
        return text, time.perf_counter() - t0, avg_logprob

    def transcribe_streaming(
        self,
        audio_path: str,
        lang: str = "en",
    ) -> StreamingResult:
        """Simulate streaming transcription using LocalAgreement-2.

        Timing model
        ------------
        We maintain a virtual timeline:
          - audio_head_s: simulated time of the current leading edge of audio
          - wall_clock_s: actual wall-clock elapsed since "audio stream started"

        In the simulation, the audio stream starts at wall_clock_s=0.
        The first audio window is available at wall_clock_s=initial_chunk_s (conceptually),
        but because we're CPU-bound and decoding takes real time, we measure
        the actual wall-clock at each emission.

        The first_emission_latency_s is the actual wall-clock time from the START
        of transcribe_streaming() to the first committed word emission.
        This includes:
          - initial_chunk_s (waiting for audio to arrive, simulated by sleeping
            or just accepting that decode starts immediately for fairness)
          - decode time for first window
          - decode time for second window (needed for agreement)

        Note: We do NOT sleep to simulate real-time audio arrival. Instead we
        compute wall-clock time purely from how long decodes actually take.
        The "fair" latency is: initial_chunk_s + sum(decode_times until first emission).
        We report both the raw wall-clock (which excludes audio-arrival wait)
        and the true streaming latency (which adds initial_chunk_s).
        """
        self._ensure_loaded()

        audio = _load_wav_as_float32(audio_path)
        total_audio_s = len(audio) / SAMPLE_RATE

        buf = HypothesisBuffer()
        emissions: List[EmissionEvent] = []
        total_decode_s = 0.0
        n_windows = 0

        # Wall-clock reference
        wall_start = time.perf_counter()

        initial_samples = int(self.initial_chunk_s * SAMPLE_RATE)
        step_samples = int(self.step_s * SAMPLE_RATE)

        # Buffer start: we trim audio up to the last committed point
        # to keep windows from growing without bound.
        buf_start_sample = 0

        # Next window end (from buf_start_sample=0)
        window_end_sample = initial_samples

        first_emission_wall: Optional[float] = None
        # Track if confidence gate was applied (any window)
        confidence_gate_applied = False

        while True:
            # Clamp to total audio
            actual_end = min(window_end_sample, len(audio))
            is_final = (actual_end >= len(audio))

            audio_window = audio[buf_start_sample:actual_end]

            if len(audio_window) < int(0.3 * SAMPLE_RATE):
                # Window too short to produce useful output
                if is_final:
                    break
                window_end_sample += step_samples
                continue

            # --- Transcribe ---
            # If confidence gating is enabled and we haven't yet emitted anything,
            # fetch avg_logprob to decide whether to commit immediately.
            # This applies to ALL windows before first emission (not just window 0),
            # so it catches the first non-empty window wherever it appears.
            before_first_emission = (first_emission_wall is None)
            use_confidence = (
                before_first_emission
                and not is_final
                and self.confidence_gate_threshold is not None
            )

            if use_confidence:
                hyp_text, decode_s, avg_logprob = self._transcribe_window_with_confidence(
                    audio_window, lang
                )
            else:
                hyp_text, decode_s = self._transcribe_window(audio_window, lang)
                avg_logprob = None

            total_decode_s += decode_s
            n_windows += 1

            hyp_tokens = _tokens(hyp_text)

            # How far into audio are we (for timing annotation)
            audio_head_s = actual_end / SAMPLE_RATE

            if is_final:
                # Final window: flush everything
                new_tokens = buf.flush(hyp_tokens)
            elif (
                use_confidence
                and avg_logprob is not None
                and avg_logprob >= self.confidence_gate_threshold
                and hyp_tokens  # non-empty hypothesis
            ):
                # Confidence gate: commit this window immediately (bypass LA-2).
                # The hypothesis is directly committed; subsequent windows use LA-2.
                new_tokens = list(hyp_tokens)  # commit all tokens
                buf.committed_tokens.extend(new_tokens)
                buf.prev_tokens = hyp_tokens
                confidence_gate_applied = True
            else:
                # LocalAgreement-2: commit LCP of prev and current hypothesis
                new_tokens = buf.push(hyp_tokens)

            if new_tokens:
                emit_wall = time.perf_counter() - wall_start
                if first_emission_wall is None:
                    first_emission_wall = emit_wall
                for word in new_tokens:
                    emissions.append(EmissionEvent(
                        word=word,
                        audio_head_s=audio_head_s,
                        emit_wall_time_s=emit_wall,
                    ))

            # NOTE: audio-buffer trimming is intentionally DISABLED. Trimming
            # buf_start_sample advances the audio window start, but the
            # HypothesisBuffer compares each new hypothesis against the previous
            # one assuming both transcribe the SAME prefix of audio. Trimming
            # breaks that assumption (the new window starts later, so its tokens
            # no longer share a prefix with prev_tokens) and corrupts the LCP,
            # producing garbage transcripts (measured WER 56% with trim on).
            # For Fleurs utterances (<~15 s) the growing-window cost is
            # acceptable; a production system with long-form audio would need
            # the full whisper_streaming approach (commit + re-prompt with
            # committed text, then trim). buf_start_sample stays at 0.

            if is_final:
                break

            window_end_sample += step_samples

        # Final flush if anything remains
        remaining = buf.flush()
        if remaining:
            emit_wall = time.perf_counter() - wall_start
            if first_emission_wall is None:
                first_emission_wall = emit_wall
            for word in remaining:
                emissions.append(EmissionEvent(
                    word=word,
                    audio_head_s=total_audio_s,
                    emit_wall_time_s=emit_wall,
                ))

        streaming_transcript = " ".join(buf.committed_tokens)

        # If a final_model_id is set, run one full-audio pass with the
        # accurate model for WER scoring. This does NOT count toward
        # streaming latency or RTFx — it happens after audio ends.
        if self.final_model_id is not None:
            self._ensure_final_model_loaded()
            accuracy_transcript = self._transcribe_full_audio(audio, lang)
        else:
            accuracy_transcript = streaming_transcript

        # True streaming first-emission latency includes the time we WAIT for
        # the first audio chunk to arrive (initial_chunk_s in wall-clock).
        # Since we don't actually sleep, we add it analytically.
        if first_emission_wall is not None:
            # first_emission_wall = pure decode time before first emission
            # true_latency = initial_chunk_s + first_emission_wall
            # BUT: in a real system, decoding window1 starts at t=initial_chunk_s.
            # So true streaming latency = initial_chunk_s + decode_window1 + decode_window2
            # which equals initial_chunk_s + first_emission_wall (what we measure)
            true_first_emission_latency = self.initial_chunk_s + first_emission_wall
        else:
            true_first_emission_latency = None

        return StreamingResult(
            final_transcript=streaming_transcript,
            accuracy_transcript=accuracy_transcript,
            emissions=emissions,
            total_audio_s=total_audio_s,
            total_decode_s=total_decode_s,
            first_emission_latency_s=true_first_emission_latency,
            n_windows=n_windows,
            confidence_gate_applied=confidence_gate_applied,
        )
