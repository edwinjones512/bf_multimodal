#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on validation loss and Wikipedia QA.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt
    python evaluate.py --checkpoint checkpoints/best.pt --qa-samples 100
"""

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from config import DataConfig, EvalConfig, SPECIAL_TOKENS, TrainConfig
from data.collate import collate_multimodal
from data.modality import modality_bucket
from data.multimodal_dataset import MultimodalDataset, subset_dataset, train_val_split_positions
from eval.metrics import validation_loss
from eval.qa_eval import evaluate_qa
from eval.wiki_qa import load_qa_set
from inference import load_model


def resolve_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_val_loader(
    data_path: Path,
    tokenizer: Tokenizer,
    model_cfg,
    data_root: Path,
    batch_size: int,
) -> DataLoader:
    """Match train.py validation setup using the checkpoint model config."""
    probe = MultimodalDataset(data_path, tokenizer, model_cfg=model_cfg, data_root=data_root)
    _, val_pos = train_val_split_positions(probe, TrainConfig().val_fraction)
    val_text_pos = [
        p for p in val_pos
        if modality_bucket(probe.records[probe.active_indices[p]]) == "text"
    ] or val_pos
    val_dataset = subset_dataset(
        MultimodalDataset(
            data_path,
            tokenizer,
            model_cfg=model_cfg,
            max_seq_len=model_cfg.max_seq_len,
            bos_id=SPECIAL_TOKENS["<bos>"],
            eos_id=SPECIAL_TOKENS["<eos>"],
            pad_id=SPECIAL_TOKENS["<pad>"],
            data_root=data_root,
        ),
        val_text_pos,
    )
    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_multimodal,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model on val loss + Wikipedia QA")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DataConfig().corpus_path)
    parser.add_argument("--tokenizer", type=Path, default=DataConfig().tokenizer_path)
    parser.add_argument("--qa-eval", type=Path, default=EvalConfig().wiki_qa_path)
    parser.add_argument("--qa-samples", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device in ("auto", "cpu")
        else "cuda"
    )

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    model = load_model(args.checkpoint, device)
    model_cfg = model.config
    data_root = Path.cwd()
    amp_dtype = resolve_amp_dtype(device)

    val_loader = build_val_loader(
        args.data,
        tokenizer,
        model_cfg,
        data_root,
        batch_size=TrainConfig().batch_size,
    )

    val_loss, val_acc = validation_loss(
        model, val_loader, device, amp=device.type == "cuda", amp_dtype=amp_dtype
    )
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation token accuracy: {val_acc:.4f}")

    if args.qa_eval.exists():
        qa_pairs = load_qa_set(args.qa_eval)
        qa_metrics = evaluate_qa(
            model, tokenizer, qa_pairs, device,
            max_samples=args.qa_samples or EvalConfig().qa_eval_samples,
        )
        print(f"Wikipedia QA F1: {qa_metrics['qa_f1']:.4f}")
        print(f"Wikipedia QA contains-accuracy: {qa_metrics['qa_contains_acc']:.4f}")
        print(f"QA samples evaluated: {qa_metrics['qa_count']}")
    else:
        print(f"QA file not found: {args.qa_eval}")


if __name__ == "__main__":
    main()