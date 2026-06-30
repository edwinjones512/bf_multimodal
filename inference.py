#!/usr/bin/env python3
"""
Run multimodal prompts against the trained model.

Supports text, images, videos, and audio in a single prompt.

Usage:
    python inference.py --prompt "The capital of France is"
    python inference.py --prompt "Describe this image" --image photo.jpg
    python inference.py --prompt "What happens in this video?" --video clip.mp4
    python inference.py --prompt "Describe this sound" --audio clip.wav
    python inference.py --prompt "Say hello" --speech-out reply.wav
    python inference.py --prompt "Compare these" --image a.jpg --audio clip.wav --video scene.mp4
"""

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from config import DataConfig, ModelConfig, SPECIAL_TOKENS
from model import MultimodalModel
from utils.media import load_audio, load_image, load_video_frames, save_audio


def _remap_legacy_state_dict(state_dict: dict) -> dict:
    """Support checkpoints saved before backbone rename."""
    if not any(key.startswith("gpt.") for key in state_dict):
        return state_dict
    return {
        (f"backbone.{key[4:]}" if key.startswith("gpt.") else key): value
        for key, value in state_dict.items()
    }


def load_model(checkpoint: Path, device: torch.device) -> MultimodalModel:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", ModelConfig())
    model = MultimodalModel(config)
    model.load_state_dict(_remap_legacy_state_dict(ckpt["model"]))
    model.to(device)
    model.eval()
    return model


def decode(tokenizer: Tokenizer, token_ids: list[int]) -> str:
    special = set(SPECIAL_TOKENS.keys())
    filtered = [t for t in token_ids if t not in special]
    return tokenizer.decode(filtered, skip_special_tokens=True)


def encode_text_prompt(tokenizer: Tokenizer, prompt: str) -> list[int]:
    """Match training format: <bos> + tokenized text (build_inputs adds <text> marker)."""
    return [SPECIAL_TOKENS["<bos>"]] + tokenizer.encode(prompt).ids


def run_prompt(
    model: MultimodalModel,
    tokenizer: Tokenizer,
    prompt: str,
    images: list[Path] | None = None,
    video: Path | None = None,
    audio: list[Path] | None = None,
    max_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    rope_scale: float | None = None,
    device: torch.device = torch.device("cpu"),
) -> str:
    cfg = model.config
    prompt_ids = encode_text_prompt(tokenizer, prompt)
    text_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    image_tensor = None
    if images:
        tensors = [load_image(p, cfg.image_size) for p in images]
        image_tensor = torch.cat(tensors, dim=0).to(device)

    video_tensor = None
    if video:
        video_tensor = load_video_frames(video, cfg.n_video_frames, cfg.image_size).to(device)

    audio_tensor = None
    if audio:
        clips = [
            load_audio(p, cfg.audio_sample_rate, cfg.audio_duration_sec).squeeze(0)
            for p in audio
        ]
        audio_tensor = torch.stack(clips).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            text_ids=text_ids,
            images=image_tensor,
            video_frames=video_tensor,
            audio=audio_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=SPECIAL_TOKENS["<eos>"],
            rope_scale=rope_scale,
        )

    new_ids = output_ids[0].tolist()[len(prompt_ids):]
    return decode(tokenizer, new_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal inference")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--image", type=Path, action="append", default=[], help="Image file(s)")
    parser.add_argument("--video", type=Path, default=None, help="Video file")
    parser.add_argument("--audio", type=Path, action="append", default=[], help="Audio file(s)")
    parser.add_argument("--speech-out", type=Path, default=None, help="Save generated speech to WAV")
    parser.add_argument("--speech-only", action="store_true", help="Only generate speech (skip text)")
    parser.add_argument("--checkpoint", type=Path, default=DataConfig().checkpoint_dir / "best.pt")
    parser.add_argument("--tokenizer", type=Path, default=DataConfig().tokenizer_path)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--rope-scale",
        type=float,
        default=None,
        help="NTK RoPE scaling for longer context at inference (e.g. 2.0 = ~2x context)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Train first: python train.py")
        return
    if not args.tokenizer.exists():
        print(f"Tokenizer not found: {args.tokenizer}")
        return

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    model = load_model(args.checkpoint, device)

    modalities = []
    if args.image:
        modalities.append(f"{len(args.image)} image(s)")
    if args.video:
        modalities.append("1 video")
    if args.audio:
        modalities.append(f"{len(args.audio)} audio clip(s)")
    modalities.append("text")
    print(f"Modalities: {', '.join(modalities)}")
    print(f"Prompt: {args.prompt}")
    print("-" * 40)

    cfg = model.config
    prompt_ids = encode_text_prompt(tokenizer, args.prompt)
    text_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    image_tensor = None
    if args.image:
        tensors = [load_image(p, cfg.image_size) for p in args.image]
        image_tensor = torch.cat(tensors, dim=0).to(device)

    video_tensor = None
    if args.video:
        video_tensor = load_video_frames(
            args.video, cfg.n_video_frames, cfg.image_size
        ).to(device)

    audio_tensor = None
    if args.audio:
        clips = [
            load_audio(p, cfg.audio_sample_rate, cfg.audio_duration_sec).squeeze(0)
            for p in args.audio
        ]
        audio_tensor = torch.stack(clips).to(device)

    if not args.speech_only:
        response = run_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            images=args.image or None,
            video=args.video,
            audio=args.audio or None,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            rope_scale=args.rope_scale,
            device=device,
        )
        print(response)

    if args.speech_out:
        with torch.no_grad():
            speech = model.generate_speech(
                text_ids=text_ids,
                images=image_tensor,
                video_frames=video_tensor,
                audio=audio_tensor,
                temperature=args.temperature,
                rope_scale=args.rope_scale,
            )
        save_audio(args.speech_out, speech.squeeze(0), cfg.audio_sample_rate)
        print(f"Saved speech to {args.speech_out}")


if __name__ == "__main__":
    main()