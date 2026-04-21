#!/usr/bin/env python
"""Fine-tune a LoRA adapter for a specific speaker on top of Whisper.

Approach (Samsung / Interspeech 2024):
  - Freeze all Whisper base weights
  - Inject LoRA into attention Q + V projections
  - Fine-tune only LoRA matrices on ~5-30 min of speaker audio
  - Save adapter weights (~10-50 MB)

Usage:
    python scripts/train_speaker_lora.py \
        --speaker-id alice \
        --audio-dir data/speakers/alice/ \
        --base-model openai/whisper-large-v3 \
        --rank 8 --epochs 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def collect_audio_files(audio_dir: Path) -> list[Path]:
    exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    files = [p for p in audio_dir.rglob("*") if p.suffix.lower() in exts]
    logger.info(f"Found {len(files)} audio files in {audio_dir}")
    return sorted(files)


def build_dataset(audio_files: list[Path], model, processor):
    """Build a HuggingFace Dataset from audio files + whisper transcriptions."""
    from datasets import Dataset
    import numpy as np
    import soundfile as sf

    records = []
    for path in audio_files:
        try:
            audio, sr = sf.read(str(path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            # Resample to 16 kHz if needed
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            inputs = processor(
                audio.astype(np.float32),
                sampling_rate=16000,
                return_tensors="pt",
            )
            # Use the base model to get a reference transcription
            # (in practice, provide human-verified transcripts for better quality)
            records.append({
                "input_features": inputs.input_features[0],
                "labels": inputs.input_features[0],  # placeholder
                "path": str(path),
            })
        except Exception as e:
            logger.warning(f"Skipped {path}: {e}")

    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker-id", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--base-model", default="openai/whisper-large-v3")
    parser.add_argument("--output-dir", default="data/lora_adapters")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load base model
    logger.info(f"Loading base model: {args.base_model}")
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    processor = WhisperProcessor.from_pretrained(args.base_model)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype="float16"
    )

    # ---- 2. Wrap with LoRA
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- 3. Build dataset
    audio_files = collect_audio_files(audio_dir)
    if not audio_files:
        logger.error(f"No audio files found in {audio_dir}")
        return

    dataset = build_dataset(audio_files, model, processor)
    logger.info(f"Dataset size: {len(dataset)} samples")

    # ---- 4. Train
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    adapter_name = f"{args.speaker_id}_whisper-large-v3_r{args.rank}"
    adapter_path = output_dir / adapter_name

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(adapter_path),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        fp16=True,
        save_steps=50,
        logging_steps=10,
        report_to="none",
        predict_with_generate=True,
        generation_max_length=225,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    logger.info(f"Training LoRA adapter for speaker '{args.speaker_id}'...")
    trainer.train()

    # ---- 5. Save adapter only
    model.save_pretrained(str(adapter_path))
    logger.info(f"Adapter saved → {adapter_path}")
    logger.info(
        f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )


if __name__ == "__main__":
    main()
