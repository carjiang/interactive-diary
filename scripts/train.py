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
from thought_trace.extractor import get_device
from thought_trace.extractor.model import BertCRFForNER, NERDataset, IGNORE_LABEL_ID
from thought_trace.extractor.schema import ID_TO_LABEL, NERConfig, NERSample

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
    p.add_argument("--extra-data", type=str, nargs="*", default=[],
                   help="Additional JSONL paths merged into training data (loaded with strict=False)")
    p.add_argument("--eval-file", type=str, default=None,
                   help="Held-out test JSONL — evaluated once after training, never used for checkpointing")
    p.add_argument("--val-split", type=float, default=VAL_SPLIT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--head-lr", type=float, default=1e-3,
                   help="LR for classifier + CRF head (default: 1e-3)")
    p.add_argument("--model-name", type=str, default="bert-base-uncased",
                   help="HuggingFace model identifier or local path")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-crf", action="store_true", help="Disable CRF layer")
    p.add_argument("--freeze-layers", type=int, default=0,
                   help="Number of bottom BERT layers to freeze")
    p.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR)
    return p.parse_args()


def load_and_split(
    path: str, val_split: float, seed: int, extra_paths: list[str] | None = None,
) -> tuple[list[NERSample], list[NERSample]]:
    samples = load_dataset(path)
    logger.info("Loaded %d samples from %s", len(samples), path)
    for ep in (extra_paths or []):
        extra = load_dataset(ep, strict=False)
        logger.info("Loaded %d extra samples from %s", len(extra), ep)
        samples.extend(extra)

    if len(samples) < 2:
        raise ValueError(
            f"Need at least 2 samples to create a train/val split, got {len(samples)}"
        )

    rng = random.Random(seed)
    rng.shuffle(samples)

    split_idx = max(1, min(len(samples) - 1, int(len(samples) * (1 - val_split))))
    train, val = samples[:split_idx], samples[split_idx:]

    logger.info("Split: %d train, %d val", len(train), len(val))
    return train, val


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
        model_name=args.model_name,
        dropout=args.dropout,
        use_crf=not args.no_crf,
        freeze_bert_layers=args.freeze_layers,
        learning_rate=args.lr,
        head_learning_rate=args.head_lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
    )

    train_samples, val_samples = load_and_split(args.data, args.val_split, args.seed, args.extra_data)

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
    logger.info("Using device: %s", device)

    model = BertCRFForNER(cfg).to(device)
    logger.info(
        "Model: %s, CRF=%s, params=%s",
        cfg.model_name,
        "on" if cfg.use_crf else "off",
        f"{sum(p.numel() for p in model.parameters()):,}",
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
        "Training: %d epochs, batch_size=%d, lr=%s, warmup=%d/%d steps",
        cfg.num_epochs, cfg.batch_size, cfg.learning_rate,
        warmup_steps, total_steps,
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
            "Epoch %d/%d — loss: %.4f, lr: %.2e, time: %.1fs",
            epoch, cfg.num_epochs, avg_loss, current_lr, elapsed,
        )

        true_tags, pred_tags = collect_predictions(model, val_loader, device)

        report = classification_report(true_tags, pred_tags, zero_division=0)
        report_dict = classification_report(true_tags, pred_tags, zero_division=0, output_dict=True)
        epoch_f1 = f1_score(true_tags, pred_tags, zero_division=0)
        entity_f1_str = "  ".join(
            f"{e}={report_dict.get(e, {}).get('f1-score', 0.0):.3f}"
            for e in ["PARTICIPANT", "ACTION", "CONTENT", "BELIEF_CUE", "TEMPORAL", "PERCEPTION"]
        )
        logger.info("Val F1: %.4f  [%s]\n%s", epoch_f1, entity_f1_str, report)

        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            best_epoch = epoch

            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / "model.pt")
            tokenizer.save_pretrained(str(ckpt_dir))
            (ckpt_dir / "config.json").write_text(
                cfg.model_dump_json(indent=2)
            )
            logger.info("  New best — saved to %s/", ckpt_dir)

    logger.info(
        "Done. Best val F1: %.4f at epoch %d. Checkpoint: %s/",
        best_f1, best_epoch, ckpt_dir,
    )

    if args.eval_file:
        logger.info("Running held-out test evaluation on %s", args.eval_file)
        test_samples = load_dataset(args.eval_file)
        test_ds = NERDataset(test_samples, tokenizer, cfg.max_seq_length)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)
        model.load_state_dict(
            torch.load(ckpt_dir / "model.pt", map_location=device, weights_only=True)
        )
        true_tags, pred_tags = collect_predictions(model, test_loader, device)
        test_f1 = f1_score(true_tags, pred_tags, zero_division=0)
        test_report = classification_report(true_tags, pred_tags, zero_division=0)
        logger.info("TEST F1: %.4f\n%s", test_f1, test_report)


if __name__ == "__main__":
    main()
