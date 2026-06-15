"""P3-1 LoRA Training — NLLB-200-distilled-600M canonical terminology adaptation.

Trains a LoRA adapter on the ML glossary training corpus (950 en->ko + 950 ko->en)
to lift canonical terminology adherence above P2-2 baseline.

Spec: rank 16, alpha 32, target_modules q/k/v/out_proj + fc1/fc2, lr 2e-4,
      ~3 epochs, per-step batch 4, grad-accum to effective 16.
      Base model frozen. Early-stop on dev loss.

Saves adapter to reports/p3-1_lora/adapter/ and data/lora_adapters/nllb-terminology-v1/

Usage:
    PYTHONPATH=$PWD ./.translator/bin/python scripts/p3_1_lora_train.py
"""
from __future__ import annotations

import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from loguru import logger

# ---- Paths ----
TRAIN_EN_KO = REPO / "data/eval/ml_glossary_train.en_ko.tsv"
TRAIN_KO_EN = REPO / "data/eval/ml_glossary_train.ko_en.tsv"
DEV_EN_KO   = REPO / "data/eval/ml_glossary_dev.en_ko.tsv"
DEV_KO_EN   = REPO / "data/eval/ml_glossary_dev.ko_en.tsv"
ADAPTER_DIR  = REPO / "reports/p3-1_lora/adapter"
ADAPTER_COPY = REPO / "data/lora_adapters/nllb-terminology-v1"
LOG_DIR      = REPO / "reports/p3-1_lora"

ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_COPY.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---- Hyperparameters ----
MODEL_NAME   = "facebook/nllb-200-distilled-600M"
SRC_LANG_EN  = "eng_Latn"
TGT_LANG_KO  = "kor_Hang"
SRC_LANG_KO  = "kor_Hang"
TGT_LANG_EN  = "eng_Latn"
MAX_SEQ_LEN  = 128
BATCH_SIZE   = 4       # per-step batch size
GRAD_ACCUM   = 4       # effective batch = 4 * 4 = 16
LR           = 2e-4
WARMUP_STEPS = 100
N_EPOCHS     = 3
LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
PATIENCE     = 3       # early-stop patience (# eval steps without improvement)


def load_tsv(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
    return pairs


def make_dataset(
    pairs: list[tuple[str, str]],
    src_lang: str,
    tgt_lang: str,
    label: str = "",
) -> list[dict]:
    """Each item: {"src": ..., "tgt": ..., "src_lang": ..., "tgt_lang": ...}"""
    return [
        {"src": s, "tgt": t, "src_lang": src_lang, "tgt_lang": tgt_lang}
        for s, t in pairs
    ]


def tokenize_batch(
    tokenizer,
    batch: list[dict],
    max_length: int = MAX_SEQ_LEN,
) -> dict:
    """Tokenize a batch of mixed-direction examples.

    Each item in the batch may have a different src/tgt lang.
    We process per-item since tokenizer src_lang/tgt_lang changes per item.
    Then stack into tensors.

    Label handling: labels[labels == pad_token_id] = -100 so loss ignores padding.
    """
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for item in batch:
        tokenizer.src_lang = item["src_lang"]
        tokenizer.tgt_lang = item["tgt_lang"]

        enc = tokenizer(
            item["src"],
            text_target=item["tgt"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids_list.append(enc["input_ids"].squeeze(0))
        attention_mask_list.append(enc["attention_mask"].squeeze(0))
        lbl = enc["labels"].squeeze(0)
        lbl[lbl == tokenizer.pad_token_id] = -100
        labels_list.append(lbl)

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
    }


def compute_dev_loss(model, tokenizer, dev_data: list[dict], device: torch.device) -> float:
    """Compute average cross-entropy loss on dev set."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for start in range(0, len(dev_data), BATCH_SIZE):
            batch = dev_data[start : start + BATCH_SIZE]
            enc = tokenize_batch(tokenizer, batch)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            labels = enc["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            total_batches += 1

    model.train()
    return total_loss / max(total_batches, 1)


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(LOG_DIR / "train.log", level="DEBUG")

    t_start = time.perf_counter()

    # ---- Load data ----
    logger.info("Loading training data...")
    train_en_ko = load_tsv(TRAIN_EN_KO)
    train_ko_en = load_tsv(TRAIN_KO_EN)
    dev_en_ko   = load_tsv(DEV_EN_KO)
    dev_ko_en   = load_tsv(DEV_KO_EN)

    logger.info(f"Train: {len(train_en_ko)} en->ko + {len(train_ko_en)} ko->en = {len(train_en_ko)+len(train_ko_en)} total")
    logger.info(f"Dev: {len(dev_en_ko)} en->ko + {len(dev_ko_en)} ko->en = {len(dev_en_ko)+len(dev_ko_en)} total")

    # Combine train data (both directions)
    train_data = (
        make_dataset(train_en_ko, SRC_LANG_EN, TGT_LANG_KO, "en->ko") +
        make_dataset(train_ko_en, SRC_LANG_KO, TGT_LANG_EN, "ko->en")
    )
    dev_data = (
        make_dataset(dev_en_ko, SRC_LANG_EN, TGT_LANG_KO, "en->ko") +
        make_dataset(dev_ko_en, SRC_LANG_KO, TGT_LANG_EN, "ko->en")
    )
    logger.info(f"Combined train: {len(train_data)}, dev: {len(dev_data)}")

    # ---- Load model + tokenizer ----
    logger.info(f"Loading {MODEL_NAME} in fp16...")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    )

    # ---- Apply LoRA ----
    logger.info("Applying LoRA config...")
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Count trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} = {trainable_params/total_params*100:.2f}%")

    device = torch.device("cuda")
    model = model.to(device)

    # ---- Optimizer ----
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LinearLR, SequentialLR

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=0.01,
    )

    # ---- Compute steps ----
    steps_per_epoch = math.ceil(len(train_data) / (BATCH_SIZE * GRAD_ACCUM))
    total_steps = steps_per_epoch * N_EPOCHS
    logger.info(f"Steps per epoch: {steps_per_epoch}, total steps: {total_steps}")

    # Warmup: linear increase for first WARMUP_STEPS optimizer steps
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_STEPS)
    # Decay: linear decay from lr to 0 over remaining steps
    remaining_steps = max(total_steps - WARMUP_STEPS, 1)
    decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=remaining_steps)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[WARMUP_STEPS])

    # ---- Training loop ----
    logger.info("Starting training...")
    import random
    random.seed(42)

    best_dev_loss = float("inf")
    patience_counter = 0
    optimizer_step = 0
    chosen_epoch = 0
    train_losses = []
    dev_losses = []

    peak_vram_train_mb = 0.0

    model.train()
    for epoch in range(N_EPOCHS):
        epoch_start = time.perf_counter()
        # Shuffle train data each epoch
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        shuffled = [train_data[i] for i in indices]

        running_loss = 0.0
        n_micro_steps = 0
        optimizer.zero_grad()

        for micro_step_idx, start in enumerate(range(0, len(shuffled), BATCH_SIZE)):
            batch = shuffled[start : start + BATCH_SIZE]
            if not batch:
                continue

            enc = tokenize_batch(tokenizer, batch)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            labels = enc["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()

            running_loss += outputs.loss.item()
            n_micro_steps += 1

            # Track VRAM peak
            if torch.cuda.is_available():
                cur_vram_mb = torch.cuda.max_memory_allocated(device) / 1024**2
                if cur_vram_mb > peak_vram_train_mb:
                    peak_vram_train_mb = cur_vram_mb

            # Optimizer step every GRAD_ACCUM micro-steps
            if (micro_step_idx + 1) % GRAD_ACCUM == 0 or (start + BATCH_SIZE) >= len(shuffled):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step += 1

                if optimizer_step % 10 == 0:
                    avg_train_loss = running_loss / n_micro_steps
                    logger.info(f"  step {optimizer_step}, train_loss={avg_train_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")

        # End of epoch: compute dev loss
        avg_epoch_loss = running_loss / max(n_micro_steps, 1)
        dev_loss = compute_dev_loss(model, tokenizer, dev_data, device)
        epoch_time = time.perf_counter() - epoch_start
        logger.info(
            f"Epoch {epoch+1}/{N_EPOCHS}: train_loss={avg_epoch_loss:.4f}, "
            f"dev_loss={dev_loss:.4f}, time={epoch_time:.1f}s, opt_steps={optimizer_step}"
        )
        train_losses.append(round(avg_epoch_loss, 4))
        dev_losses.append(round(dev_loss, 4))

        # Early stopping
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            patience_counter = 0
            chosen_epoch = epoch + 1
            logger.info(f"  New best dev_loss={best_dev_loss:.4f}. Saving adapter...")
            model.save_pretrained(str(ADAPTER_DIR))
            tokenizer.save_pretrained(str(ADAPTER_DIR))
        else:
            patience_counter += 1
            logger.info(f"  No improvement. Patience {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    total_train_time = time.perf_counter() - t_start
    logger.info(f"Training complete. Best dev_loss={best_dev_loss:.4f} at epoch {chosen_epoch}")
    logger.info(f"Total training time: {total_train_time:.1f}s")
    logger.info(f"Peak VRAM (train): {peak_vram_train_mb:.0f} MB = {peak_vram_train_mb/1024:.2f} GB")

    # ru_maxrss (true peak RSS, in KB on Linux)
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logger.info(f"ru_maxrss (peak RSS): {rss_kb} KB = {rss_kb/1024:.1f} MB = {rss_kb/1024/1024:.2f} GB")

    # Also copy adapter to data/lora_adapters/
    import shutil
    shutil.copytree(str(ADAPTER_DIR), str(ADAPTER_COPY), dirs_exist_ok=True)
    logger.info(f"Adapter copied to {ADAPTER_COPY}")

    # ---- Save training metadata ----
    metadata = {
        "experiment": "P3-1_lora_training",
        "model": MODEL_NAME,
        "lora_config": {
            "rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "target_modules": LORA_TARGETS,
            "lora_dropout": LORA_DROPOUT,
        },
        "training": {
            "n_train": len(train_data),
            "n_dev": len(dev_data),
            "n_epochs_run": len(train_losses),
            "chosen_epoch": chosen_epoch,
            "batch_size_per_step": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "effective_batch": BATCH_SIZE * GRAD_ACCUM,
            "lr": LR,
            "warmup_steps": WARMUP_STEPS,
            "total_optimizer_steps": optimizer_step,
            "train_losses": train_losses,
            "dev_losses": dev_losses,
            "best_dev_loss": round(best_dev_loss, 4),
            "total_train_time_s": round(total_train_time, 1),
        },
        "params": {
            "trainable": trainable_params,
            "total": total_params,
            "trainable_pct": round(trainable_params / total_params * 100, 4),
        },
        "peak_vram_train_mb": round(peak_vram_train_mb, 1),
        "ru_maxrss_kb": rss_kb,
        "adapter_path": str(ADAPTER_DIR),
    }

    metadata_path = LOG_DIR / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Training metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
