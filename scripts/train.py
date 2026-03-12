"""
Training script for the BERT + CRF NER model.

Usage:
    .venv/bin/python scripts/train.py --data data/train.jsonl
    .venv/bin/python scripts/train.py --data data/fixtures.jsonl --epochs 2 --batch-size 2
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup

from seqeval.metrics import classification_report, f1_score

from data.schema_spec import load_dataset
from extractor.model import BertCRFForNER, NERDataset, IGNORE_LABEL_ID
from extractor.schema import ID_TO_LABEL, NERConfig, NERSample

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
VAL_SPLIT = 0.1
SEED = 42
CHECKPOINT_DIR = "checkpoints/best"


def seed_everything(seed: int) -> None:
    """Seed all RNGs for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker a deterministic seed derived from the
    global torch seed so shuffled batches are reproducible across runs."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BERT+CRF NER model")
    p.add_argument("--data", type=str, required=True, help="Path to JSONL training data")
    p.add_argument("--val-split", type=float, default=VAL_SPLIT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR)
    return p.parse_args()


def load_and_split(
    path: str, val_split: float, seed: int,
) -> tuple[list[NERSample], list[NERSample]]:
    samples = load_dataset(path)
    logger.info(f"Loaded {len(samples)} samples from {path}")

    if len(samples) < 2:
        raise ValueError(
            f"Need at least 2 samples to create a train/val split, got {len(samples)}"
        )

    rng = random.Random(seed)
    rng.shuffle(samples)

    split_idx = max(1, min(len(samples) - 1, int(len(samples) * (1 - val_split))))
    train, val = samples[:split_idx], samples[split_idx:]

    logger.info(f"Split: {len(train)} train, {len(val)} val")
    return train, val


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    return device


def collect_predictions(
    model: BertCRFForNER,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[list[str]], list[list[str]]]:
    """Run model.decode() on a DataLoader, return (true_tags, pred_tags)."""
    model.eval()
    all_true: list[list[str]] = []
    all_pred: list[list[str]] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        word_starts = batch["word_starts"].to(device)
        num_words = batch["num_words"].to(device)
        labels = batch["labels"]

        decoded = model.decode(input_ids, attention_mask, word_starts, num_words)

        for i, pred_ids in enumerate(decoded):
            n = num_words[i].item()
            true_ids = labels[i, :n].tolist()

            true_tags = [ID_TO_LABEL.get(t, "O") for t in true_ids]
            pred_tags = [ID_TO_LABEL.get(p, "O") for p in pred_ids[:n]]
            all_true.append(true_tags)
            all_pred.append(pred_tags)

    return all_true, all_pred


def _build_param_groups(
    model: BertCRFForNER, cfg: NERConfig,
) -> list[dict]:
    """Separate parameters into groups with proper weight-decay and LR.

    Per HuggingFace recommendations (https://huggingface.co/transformers/training.html):
      - bias and LayerNorm.weight receive **no** weight decay.
      - The task head (classifier + CRF) uses a higher LR than the
        pre-trained BERT backbone.
    """
    no_decay = {"bias", "LayerNorm.weight"}
    head_modules = {"classifier", "crf"}

    backbone_decay, backbone_no_decay = [], []
    head_decay, head_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = any(name.startswith(m) for m in head_modules)
        is_no_decay = any(nd in name for nd in no_decay)

        if is_head:
            (head_no_decay if is_no_decay else head_decay).append(param)
        else:
            (backbone_no_decay if is_no_decay else backbone_decay).append(param)

    return [
        {"params": backbone_decay, "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay},
        {"params": backbone_no_decay, "lr": cfg.learning_rate, "weight_decay": 0.0},
        {"params": head_decay, "lr": cfg.head_learning_rate, "weight_decay": cfg.weight_decay},
        {"params": head_no_decay, "lr": cfg.head_learning_rate, "weight_decay": 0.0},
    ]


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    cfg = NERConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
    )

    train_samples, val_samples = load_and_split(args.data, args.val_split, args.seed)

    tokenizer = BertTokenizerFast.from_pretrained(cfg.model_name)
    train_ds = NERDataset(train_samples, tokenizer, cfg.max_seq_length)
    val_ds = NERDataset(val_samples, tokenizer, cfg.max_seq_length)

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        worker_init_fn=_worker_init_fn, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        worker_init_fn=_worker_init_fn,
    )

    device = get_device()

    model = BertCRFForNER(cfg).to(device)
    logger.info(
        f"Model: {cfg.model_name}, CRF={'on' if cfg.use_crf else 'off'}, "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    optimizer = AdamW(
        _build_param_groups(model, cfg),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    best_f1 = 0.0
    best_epoch = -1
    ckpt_dir = Path(args.checkpoint_dir)

    logger.info(
        f"Training: {cfg.num_epochs} epochs, "
        f"batch_size={cfg.batch_size}, lr={cfg.learning_rate}, "
        f"warmup={warmup_steps}/{total_steps} steps"
    )

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_starts = batch["word_starts"].to(device)
            num_words = batch["num_words"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            output = model(input_ids, attention_mask, word_starts, num_words, labels)
            loss = output["loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        elapsed = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            f"Epoch {epoch}/{cfg.num_epochs} — "
            f"loss: {avg_loss:.4f}, lr: {current_lr:.2e}, time: {elapsed:.1f}s"
        )

        true_tags, pred_tags = collect_predictions(model, val_loader, device)

        report = classification_report(true_tags, pred_tags, zero_division=0)
        epoch_f1 = f1_score(true_tags, pred_tags, zero_division=0)
        logger.info(f"Val F1: {epoch_f1:.4f}\n{report}")

        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            best_epoch = epoch

            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / "model.pt")
            tokenizer.save_pretrained(str(ckpt_dir))
            (ckpt_dir / "config.json").write_text(
                cfg.model_dump_json(indent=2)
            )
            logger.info(f"  ↑ New best — saved to {ckpt_dir}/")

    logger.info(
        f"\nDone. Best val F1: {best_f1:.4f} at epoch {best_epoch}. "
        f"Checkpoint: {ckpt_dir}/"
    )


if __name__ == "__main__":
    main()
