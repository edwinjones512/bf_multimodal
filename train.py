#!/usr/bin/env python3
"""
Train the multimodal model on the merged corpus.

Loads paired audio/image/video + text when present in corpus.jsonl. Uses a
mixed-batch sampler, optional encoder freeze warmup, speech mel loss, and
Wikipedia QA + LibriVox audio-caption eval.

Usage:
    python train.py --device cuda
    python train.py --device cuda --multimodal-fraction 0.25 --max-video-per-batch 2
    python train.py --train-speech --freeze-encoders-steps 500
    python train.py --text-only   # legacy text-only mode
    python train.py --resume checkpoints/best.pt

On Colab, checkpoints are written to /content/transformer_local/checkpoints and
periodically synced to the mounted Drive checkpoints/ directory to avoid Drive
timeouts during training. Run sync_checkpoints.py --watch 300 for background sync.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm

from config import (
    DataConfig,
    EvalConfig,
    ModelConfig,
    PROJECT_ROOT,
    SPECIAL_TOKENS,
    TrainConfig,
    data_path_candidates,
    resolve_data_path,
)
from data.collate import collate_multimodal
from data.modality import corpus_modality_counts, modality_bucket
from data.multimodal_dataset import (
    MultimodalDataset,
    subset_dataset,
    train_val_split_positions,
)
from data.sampler import MixedBatchSampler
from eval.audio_caption_eval import evaluate_audio_caption, holdout_book_ids_from_manifest, load_audio_eval_set
from eval.metrics import validation_loss
from eval.qa_eval import evaluate_qa
from eval.wiki_qa import load_qa_set
from model import MultimodalModel
from utils.drive_sync import (
    is_colab,
    resolve_checkpoint_dirs,
    resolve_checkpoint_file,
    resolve_metrics_path,
    seed_local_from_drive,
    sync_directory,
)


def get_lr(step: int, warmup: int, max_steps: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available. Install a CUDA-enabled PyTorch build.")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def save_checkpoint(
    model: MultimodalModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: Path,
    metrics: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": model.config,
        "metrics": metrics or {},
    }
    torch.save(payload, path)


def append_metrics(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def clear_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def resolve_amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def sanitize_gradients(model: torch.nn.Module) -> bool:
    """Zero out non-finite gradient entries. Returns True if any were sanitized."""
    found_bad = False
    for param in model.parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            found_bad = True
            torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0, out=param.grad)
    return found_bad


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multimodal model with eval")
    parser.add_argument("--data", type=Path, default=DataConfig().corpus_path)
    parser.add_argument("--tokenizer", type=Path, default=DataConfig().tokenizer_path)
    parser.add_argument("--checkpoint-dir", type=Path, default=DataConfig().checkpoint_dir)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--qa-eval", type=Path, default=EvalConfig().wiki_qa_path)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save step_N.pt snapshots every N optimizer steps (default: 100000)",
    )
    parser.add_argument(
        "--latest-every",
        type=int,
        default=None,
        help="Overwrite latest.pt every N optimizer steps (default: 1000)",
    )
    parser.add_argument("--qa-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--small", action="store_true", help="Use smaller model for Colab T4 GPU")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Override max sequence length")
    parser.add_argument("--val-batch-size", type=int, default=None, help="Smaller batch for validation (saves VRAM)")
    parser.add_argument("--skip-qa-eval", action="store_true", help="Skip Wikipedia QA eval during training")
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Sliding-window attention size (0 = full causal, default from config)",
    )
    parser.add_argument("--multimodal-fraction", type=float, default=None)
    parser.add_argument("--max-audio-per-batch", type=int, default=None)
    parser.add_argument("--max-video-per-batch", type=int, default=None)
    parser.add_argument("--max-images-per-batch", type=int, default=None)
    parser.add_argument("--freeze-encoders-steps", type=int, default=None)
    parser.add_argument("--train-speech", action="store_true")
    parser.add_argument("--skip-audio-eval", action="store_true")
    parser.add_argument("--text-only", action="store_true", help="Legacy text-only training")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override base learning rate")
    parser.add_argument(
        "--drive-checkpoint-dir",
        type=Path,
        default=None,
        help="Google Drive checkpoint directory for periodic sync (Colab default: project checkpoints/)",
    )
    parser.add_argument(
        "--drive-sync-every",
        type=int,
        default=None,
        help="Sync local checkpoints to Drive every N optimizer steps (default: same as --latest-every)",
    )
    parser.add_argument(
        "--no-drive-sync",
        action="store_true",
        help="Disable periodic checkpoint sync to Google Drive",
    )
    args = parser.parse_args()

    checkpoint_dir, drive_checkpoint_dir = resolve_checkpoint_dirs(
        args.checkpoint_dir,
        args.drive_checkpoint_dir,
    )
    metrics_path = resolve_metrics_path(checkpoint_dir, args.metrics_path)
    drive_sync_every = args.drive_sync_every
    enable_drive_sync = (
        not args.no_drive_sync
        and drive_checkpoint_dir is not None
        and drive_sync_every != 0
    )

    if is_colab() and checkpoint_dir != args.checkpoint_dir:
        print(f"Colab: writing checkpoints to local storage at {checkpoint_dir}")
        if enable_drive_sync:
            print(f"Colab: syncing checkpoints to Drive at {drive_checkpoint_dir}")
            seed_local_from_drive(checkpoint_dir, drive_checkpoint_dir)

    model_cfg = ModelConfig.colab() if args.small else ModelConfig()
    if args.max_seq_len:
        model_cfg.max_seq_len = args.max_seq_len
    if args.window_size is not None:
        model_cfg.sliding_window = args.window_size
    train_cfg = TrainConfig()
    eval_cfg = EvalConfig()

    if args.epochs:
        train_cfg.max_epochs = args.epochs
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.grad_accum_steps:
        train_cfg.grad_accum_steps = args.grad_accum_steps
    if args.eval_every:
        train_cfg.eval_every = args.eval_every
    if args.save_every:
        train_cfg.save_every = args.save_every
    if args.latest_every:
        train_cfg.latest_every = args.latest_every
    if args.qa_samples:
        eval_cfg.qa_eval_samples = args.qa_samples
    if args.num_workers is not None:
        train_cfg.num_workers = args.num_workers
    if args.multimodal_fraction is not None:
        train_cfg.multimodal_fraction = args.multimodal_fraction
    if args.max_audio_per_batch is not None:
        train_cfg.max_audio_per_batch = args.max_audio_per_batch
    if args.max_video_per_batch is not None:
        train_cfg.max_video_per_batch = args.max_video_per_batch
    if args.max_images_per_batch is not None:
        train_cfg.max_images_per_batch = args.max_images_per_batch
    if args.freeze_encoders_steps is not None:
        train_cfg.freeze_encoders_steps = args.freeze_encoders_steps
    if args.train_speech:
        train_cfg.train_speech = True
    if args.learning_rate is not None:
        train_cfg.learning_rate = args.learning_rate
    if drive_sync_every is None:
        drive_sync_every = train_cfg.latest_every

    def maybe_sync_drive(step: int, *, force: bool = False) -> None:
        if not enable_drive_sync:
            return
        if force or (drive_sync_every > 0 and step % drive_sync_every == 0):
            sync_directory(checkpoint_dir, drive_checkpoint_dir)

    effective_batch = train_cfg.batch_size * train_cfg.grad_accum_steps
    ref_batch = 8
    if effective_batch != ref_batch:
        train_cfg.learning_rate *= math.sqrt(ref_batch / effective_batch)

    device = resolve_device(args.device)
    use_amp = train_cfg.amp and device.type == "cuda" and not args.no_amp
    amp_dtype = resolve_amp_dtype(device)
    val_batch_size = args.val_batch_size or train_cfg.batch_size
    amp_label = f"{amp_dtype}".replace("torch.", "") if use_amp else "off"
    print(f"Device: {device} (AMP: {use_amp}, dtype: {amp_label})")
    print(
        f"Model: d_model={model_cfg.d_model}, layers={model_cfg.n_layers}, "
        f"max_seq_len={model_cfg.max_seq_len}, sliding_window={model_cfg.sliding_window}"
    )
    if device.type == "cuda":
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    if not args.tokenizer.exists():
        print(f"Tokenizer not found: {args.tokenizer}")
        print("Run: python process_all.py")
        sys.exit(1)
    if not args.data.exists():
        print(f"Training data not found: {args.data}")
        print("Run: python process_all.py")
        sys.exit(1)

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    data_root = Path.cwd()

    if args.text_only:
        from data.dataset import CorpusDataset, train_val_split

        full_dataset = CorpusDataset(
            args.data,
            tokenizer,
            max_seq_len=model_cfg.max_seq_len,
            bos_id=SPECIAL_TOKENS["<bos>"],
            eos_id=SPECIAL_TOKENS["<eos>"],
            pad_id=SPECIAL_TOKENS["<pad>"],
            lazy=True,
        )
        train_dataset, val_dataset = train_val_split(full_dataset, train_cfg.val_fraction)
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            num_workers=train_cfg.num_workers,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=device.type == "cuda",
        )
        use_multimodal = False
    else:
        probe = MultimodalDataset(args.data, tokenizer, model_cfg=model_cfg, data_root=data_root)
        mod_counts = corpus_modality_counts(probe.records)
        print(
            f"Corpus modalities: text={mod_counts['text']:,} audio={mod_counts['audio']:,} "
            f"image={mod_counts['image']:,} video={mod_counts['video']:,}"
        )

        train_pos, val_pos = train_val_split_positions(probe, train_cfg.val_fraction)
        train_dataset = subset_dataset(
            MultimodalDataset(
                args.data,
                tokenizer,
                model_cfg=model_cfg,
                max_seq_len=model_cfg.max_seq_len,
                bos_id=SPECIAL_TOKENS["<bos>"],
                eos_id=SPECIAL_TOKENS["<eos>"],
                pad_id=SPECIAL_TOKENS["<pad>"],
                data_root=data_root,
                include_speech_target=train_cfg.train_speech,
            ),
            train_pos,
        )
        val_text_pos = [
            p for p in val_pos
            if modality_bucket(probe.records[probe.active_indices[p]]) == "text"
        ] or val_pos
        val_dataset = subset_dataset(
            MultimodalDataset(
                args.data,
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

        batch_sampler = MixedBatchSampler(
            train_dataset,
            batch_size=train_cfg.batch_size,
            multimodal_fraction=train_cfg.multimodal_fraction,
            max_audio_per_batch=train_cfg.max_audio_per_batch,
            max_images_per_batch=train_cfg.max_images_per_batch,
            max_video_per_batch=train_cfg.max_video_per_batch,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_multimodal,
            num_workers=train_cfg.num_workers,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            collate_fn=collate_multimodal,
            num_workers=train_cfg.num_workers,
            pin_memory=device.type == "cuda",
        )
        use_multimodal = True

    qa_pairs = []
    qa_eval_path = resolve_data_path(args.qa_eval)
    if args.skip_qa_eval:
        print("Skipping Wikipedia QA eval (--skip-qa-eval)")
    elif qa_eval_path is not None:
        qa_pairs = load_qa_set(qa_eval_path)
        print(f"Loaded {len(qa_pairs):,} Wikipedia QA eval questions from {qa_eval_path}")
    else:
        print("QA eval file not found. Checked:")
        for candidate in data_path_candidates(args.qa_eval):
            print(f"  {candidate.resolve()} (exists={candidate.exists()})")
        print(f"  project_root={PROJECT_ROOT}; cwd={Path.cwd()}")
        print("Run: python eval/build_wiki_qa.py")

    audio_eval_items: list[dict] = []
    if use_multimodal:
        if args.skip_audio_eval:
            print("Skipping LibriVox audio eval (--skip-audio-eval)")
        elif eval_cfg.audio_eval_path.exists():
            holdout = holdout_book_ids_from_manifest(DataConfig().librivox_dir / "books.jsonl")
            audio_eval_items = load_audio_eval_set(
                eval_cfg.audio_eval_path,
                data_root=data_root,
                holdout_book_ids=holdout,
            )
            print(f"Loaded {len(audio_eval_items):,} audio eval clips")
        else:
            print(f"Audio eval file not found ({eval_cfg.audio_eval_path})")

    model = MultimodalModel(model_cfg).to(device)
    if train_cfg.freeze_encoders_steps > 0:
        model.set_encoders_trainable(False)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=2**14)

    start_step = 0
    best_val_loss = float("inf")
    evals_without_improvement = 0

    resume_path = (
        resolve_checkpoint_file(
            args.resume,
            write_dir=checkpoint_dir,
            sync_dir=drive_checkpoint_dir,
        )
        if args.resume
        else None
    )
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        if ckpt.get("metrics", {}).get("val_loss") is not None:
            best_val_loss = ckpt["metrics"]["val_loss"]
        print(f"Resumed from step {start_step} ({resume_path})")
    elif args.resume:
        print(f"Resume checkpoint not found: {args.resume}")

    steps_per_epoch = math.ceil(len(train_loader) / train_cfg.grad_accum_steps)
    max_steps = args.max_steps or (steps_per_epoch * train_cfg.max_epochs)
    if args.max_steps:
        train_cfg.warmup_steps = min(train_cfg.warmup_steps, max(100, max_steps // 10))
    global_step = start_step
    micro_step = 0
    accum_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    print(f"Training on {len(train_dataset):,} records, validating on {len(val_dataset):,}")
    print(f"Max optimizer steps: {max_steps:,}")
    print(f"Learning rate: {train_cfg.learning_rate:.2e} (warmup {train_cfg.warmup_steps:,} steps)")
    pbar = tqdm(total=max_steps, initial=start_step, desc="Training")

    for epoch in range(train_cfg.max_epochs):
        if use_multimodal and hasattr(train_loader, "batch_sampler"):
            train_loader.batch_sampler.set_epoch(epoch)

        for batch in train_loader:
            if global_step >= max_steps:
                break

            if (
                use_multimodal
                and train_cfg.freeze_encoders_steps > 0
                and global_step == train_cfg.freeze_encoders_steps
            ):
                model.set_encoders_trainable(True)
                print(f"Unfroze modality encoders at step {global_step}")

            text_ids = batch["text_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            forward_kwargs: dict = {
                "text_ids": text_ids,
                "labels": labels,
                "label_smoothing": train_cfg.label_smoothing,
            }
            if use_multimodal:
                if batch.get("audio") is not None:
                    forward_kwargs["audio"] = batch["audio"].to(device, non_blocking=True)
                if batch.get("images") is not None:
                    forward_kwargs["images"] = batch["images"].to(device, non_blocking=True)
                if batch.get("video") is not None:
                    forward_kwargs["video_frames"] = batch["video"].to(device, non_blocking=True)
                if batch.get("speech_target") is not None:
                    forward_kwargs["speech_target"] = batch["speech_target"].to(device, non_blocking=True)
                    forward_kwargs["speech_loss_weight"] = train_cfg.speech_loss_weight

            lr = get_lr(global_step, train_cfg.warmup_steps, max_steps, train_cfg.learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            with torch.autocast(
                device_type=device.type, enabled=use_amp, dtype=amp_dtype
            ):
                out = model(**forward_kwargs)
                loss = out["loss"] / train_cfg.grad_accum_steps

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print(f"\n[warn] Non-finite loss ({loss_value}) at step {global_step}, skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                micro_step = 0
                accum_loss = 0.0
                if device.type == "cuda":
                    scaler.update()
                continue

            scaler.scale(loss).backward()
            accum_loss += loss_value * train_cfg.grad_accum_steps
            micro_step += 1

            if micro_step % train_cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                if sanitize_gradients(model):
                    print(f"\n[warn] Sanitized non-finite gradients at step {global_step}.")
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                if not math.isfinite(grad_norm):
                    print(
                        f"\n[warn] Non-finite grad norm at step {global_step}, "
                        "skipping optimizer step."
                    )
                    optimizer.zero_grad(set_to_none=True)
                    micro_step = 0
                    accum_loss = 0.0
                    scaler.update()
                    continue
                if grad_norm > 0:
                    scaler.step(optimizer)
                else:
                    print(
                        f"\n[warn] Zero grad norm at step {global_step}; "
                        "advancing without weight update."
                    )
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                pbar.update(1)
                pbar.set_postfix(loss=f"{accum_loss:.4f}", lr=f"{lr:.2e}")
                accum_loss = 0.0

                if global_step % train_cfg.eval_every == 0:
                    clear_cuda_cache(device)
                    val_loss, val_acc = validation_loss(
                        model, val_loader, device, amp=use_amp, amp_dtype=amp_dtype
                    )
                    clear_cuda_cache(device)
                    metrics = {
                        "step": global_step,
                        "val_loss": val_loss,
                        "val_token_acc": val_acc,
                    }

                    if qa_pairs and math.isfinite(val_loss):
                        qa_metrics = evaluate_qa(
                            model,
                            tokenizer,
                            qa_pairs,
                            device,
                            max_samples=eval_cfg.qa_eval_samples,
                            max_tokens=eval_cfg.qa_max_tokens,
                        )
                        model.train()
                        clear_cuda_cache(device)
                        metrics.update(qa_metrics)
                    elif qa_pairs:
                        print(f"\n[eval step {global_step}] skipping QA eval (non-finite val_loss)")

                    if audio_eval_items:
                        audio_metrics = evaluate_audio_caption(
                            model,
                            tokenizer,
                            audio_eval_items,
                            device,
                            sample_rate=model_cfg.audio_sample_rate,
                            max_duration_sec=model_cfg.audio_duration_sec,
                            max_samples=eval_cfg.audio_eval_samples,
                            max_tokens=eval_cfg.audio_max_tokens,
                        )
                        model.train()
                        clear_cuda_cache(device)
                        metrics.update(audio_metrics)

                    postfix = {"loss": f"{val_loss:.4f}", "lr": f"{lr:.2e}"}
                    if qa_pairs:
                        postfix["qa_f1"] = f"{metrics.get('qa_f1', 0):.3f}"
                    if audio_eval_items:
                        postfix["audio_f1"] = f"{metrics.get('audio_caption_f1', 0):.3f}"
                    pbar.set_postfix(**postfix)

                    append_metrics(metrics_path, metrics)
                    print(
                        f"\n[eval step {global_step}] val_loss={val_loss:.4f} "
                        f"val_acc={val_acc:.4f}"
                        + (
                            f" qa_f1={metrics.get('qa_f1', 0):.4f} "
                            f"qa_contains={metrics.get('qa_contains_acc', 0):.4f}"
                            if qa_pairs
                            else ""
                        )
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        evals_without_improvement = 0
                        save_checkpoint(
                            model, optimizer, global_step,
                            checkpoint_dir / "best.pt",
                            metrics=metrics,
                        )
                        maybe_sync_drive(global_step, force=True)
                    else:
                        evals_without_improvement += 1

                    if evals_without_improvement >= train_cfg.early_stop_patience:
                        print("Early stopping: validation loss stopped improving.")
                        global_step = max_steps
                        break

                if global_step % train_cfg.save_every == 0:
                    save_checkpoint(
                        model, optimizer, global_step,
                        checkpoint_dir / f"step_{global_step}.pt",
                    )
                    maybe_sync_drive(global_step, force=True)
                if global_step % train_cfg.latest_every == 0:
                    save_checkpoint(
                        model, optimizer, global_step,
                        checkpoint_dir / "latest.pt",
                    )
                    maybe_sync_drive(global_step)

        if global_step >= max_steps:
            break

    save_checkpoint(model, optimizer, global_step, checkpoint_dir / "final.pt")
    maybe_sync_drive(global_step, force=True)
    pbar.close()
    print(f"Training complete.")
    print(f"  final : {checkpoint_dir / 'final.pt'}")
    if (checkpoint_dir / "best.pt").exists():
        print(f"  best  : {checkpoint_dir / 'best.pt'} (val_loss={best_val_loss:.4f})")
    if enable_drive_sync:
        print(f"  drive : {drive_checkpoint_dir}")
    print(f"  metrics: {metrics_path}")


if __name__ == "__main__":
    main()