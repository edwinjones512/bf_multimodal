import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig, SPECIAL_TOKENS
from .audio_decoder import AudioDecoder
from .audio_encoder import AudioEncoder, LogMelSpectrogram
from .transformer import DecoderTransformer
from .vision_encoder import VisionEncoder
from .video_encoder import VideoEncoder


class MultimodalModel(nn.Module):
    """
    Multimodal transformer that accepts text, images, videos, and audio, and can output speech.

    Input sequence:
        [<image> img_tokens...] [<video> vid_tokens...] [<audio> aud_tokens...] [<text>] text_tokens...
    Speech output sequence:
        [<speech>] speech_latent_tokens... -> waveform via AudioDecoder
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()

        self.backbone = DecoderTransformer(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            d_ff=self.config.d_ff,
            max_seq_len=self.config.max_seq_len,
            dropout=self.config.dropout,
            pad_token_id=SPECIAL_TOKENS["<pad>"],
            rope_theta=self.config.rope_theta,
            rope_scale=self.config.rope_scale,
            inference_max_seq_len=self.config.inference_max_seq_len,
            sliding_window=self.config.sliding_window,
        )

        self.vision_encoder = VisionEncoder(
            d_model=self.config.d_model,
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
        )
        self.video_encoder = VideoEncoder(
            d_model=self.config.d_model,
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
            n_frames=self.config.n_video_frames,
        )
        self.audio_encoder = AudioEncoder(
            d_model=self.config.d_model,
            sample_rate=self.config.audio_sample_rate,
            n_mels=self.config.n_mel_bins,
            n_fft=self.config.audio_n_fft,
            hop_length=self.config.audio_hop_length,
            max_samples=self.config.max_audio_samples,
            n_tokens=self.config.n_audio_tokens,
        )
        self.audio_decoder = AudioDecoder(
            d_model=self.config.d_model,
            sample_rate=self.config.audio_sample_rate,
            n_mels=self.config.n_mel_bins,
            n_fft=self.config.audio_n_fft,
            hop_length=self.config.audio_hop_length,
            max_samples=self.config.max_audio_samples,
            n_tokens=self.config.n_audio_tokens,
        )

        self.modality_proj = nn.Linear(self.config.d_model, self.config.d_model)
        self.speech_latent_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model),
            nn.GELU(),
            nn.Linear(self.config.d_model, self.config.d_model),
        )
        self._mel_extractor = LogMelSpectrogram(
            sample_rate=self.config.audio_sample_rate,
            n_fft=self.config.audio_n_fft,
            hop_length=self.config.audio_hop_length,
            n_mels=self.config.n_mel_bins,
        )

    def _encode_modality(self, tensor: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        """Run modality encoders in fp32; AMP fp16 breaks STFT/conv backward."""
        with torch.autocast(device_type=tensor.device.type, enabled=False):
            if tensor.is_floating_point():
                tensor = tensor.float()
            tokens = encoder(tensor)
            return self.modality_proj(tokens)

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return self._encode_modality(images, self.vision_encoder)

    def _encode_videos(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode videos, optionally one sample at a time to limit peak VRAM."""
        micro = self.config.video_encode_micro_batch
        if micro <= 0 or frames.shape[0] <= micro:
            return self._encode_modality(frames, self.video_encoder)

        chunks: list[torch.Tensor] = []
        for start in range(0, frames.shape[0], micro):
            chunk = frames[start : start + micro]
            chunks.append(self._encode_modality(chunk, self.video_encoder))
            if frames.is_cuda:
                torch.cuda.empty_cache()
        return torch.cat(chunks, dim=0)

    def _encode_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        return self._encode_modality(waveform, self.audio_encoder)

    def build_inputs(
        self,
        text_ids: torch.Tensor,
        images: torch.Tensor | None = None,
        video_frames: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build combined input_ids and inputs_embeds for a multimodal prompt.

        Returns:
            input_ids: placeholder token IDs (for generation continuation)
            inputs_embeds: full embedded sequence
        """
        B = text_ids.shape[0]
        device = text_ids.device

        segments: list[torch.Tensor] = []
        id_segments: list[torch.Tensor] = []

        if images is not None:
            img_tokens = self._encode_images(images)
            marker = torch.full((B, 1), SPECIAL_TOKENS["<image>"], device=device, dtype=torch.long)
            segments.append(
                torch.cat(
                    [self.backbone.token_emb(marker), img_tokens],
                    dim=1,
                )
            )
            id_segments.append(
                torch.cat(
                    [marker, torch.zeros(B, img_tokens.shape[1], device=device, dtype=torch.long)],
                    dim=1,
                )
            )

        if video_frames is not None:
            vid_tokens = self._encode_videos(video_frames)
            marker = torch.full((B, 1), SPECIAL_TOKENS["<video>"], device=device, dtype=torch.long)
            segments.append(
                torch.cat(
                    [self.backbone.token_emb(marker), vid_tokens],
                    dim=1,
                )
            )
            id_segments.append(
                torch.cat(
                    [marker, torch.zeros(B, vid_tokens.shape[1], device=device, dtype=torch.long)],
                    dim=1,
                )
            )

        if audio is not None:
            aud_tokens = self._encode_audio(audio)
            marker = torch.full((B, 1), SPECIAL_TOKENS["<audio>"], device=device, dtype=torch.long)
            segments.append(
                torch.cat(
                    [self.backbone.token_emb(marker), aud_tokens],
                    dim=1,
                )
            )
            id_segments.append(
                torch.cat(
                    [marker, torch.zeros(B, aud_tokens.shape[1], device=device, dtype=torch.long)],
                    dim=1,
                )
            )

        text_marker = torch.full((B, 1), SPECIAL_TOKENS["<text>"], device=device, dtype=torch.long)
        text_embeds = torch.cat(
            [self.backbone.token_emb(text_marker), self.backbone.token_emb(text_ids)],
            dim=1,
        )
        text_ids_full = torch.cat([text_marker, text_ids], dim=1)

        segments.append(text_embeds)
        id_segments.append(text_ids_full)

        inputs_embeds = torch.cat(segments, dim=1)
        input_ids = torch.cat(id_segments, dim=1)

        max_len = self.config.max_seq_len
        if inputs_embeds.shape[1] > max_len:
            inputs_embeds = inputs_embeds[:, -max_len:, :]
            input_ids = input_ids[:, -max_len:]

        return input_ids, inputs_embeds

    def _text_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        label_smoothing: float,
    ) -> torch.Tensor:
        seq_len = logits.shape[1]
        text_len = min(labels.shape[1], seq_len)
        aligned_labels = torch.full(
            (labels.shape[0], seq_len),
            SPECIAL_TOKENS["<pad>"],
            dtype=labels.dtype,
            device=labels.device,
        )
        text_start = seq_len - text_len
        aligned_labels[:, text_start:] = labels[:, :text_len]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = aligned_labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=SPECIAL_TOKENS["<pad>"],
            label_smoothing=label_smoothing,
        )

    def _speech_mel_loss(
        self,
        text_ids: torch.Tensor,
        speech_target: torch.Tensor,
        images: torch.Tensor | None = None,
        video_frames: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_ids, inputs_embeds = self.build_inputs(text_ids, images, video_frames, audio)
        B = text_ids.shape[0]
        device = text_ids.device
        n_speech_tokens = self.config.n_audio_tokens

        speech_marker = torch.full(
            (B, 1), SPECIAL_TOKENS["<speech>"], device=device, dtype=torch.long
        )
        embeds = torch.cat([inputs_embeds, self.backbone.token_emb(speech_marker)], dim=1)
        ids = torch.cat([input_ids, speech_marker], dim=1)

        latents: list[torch.Tensor] = []
        for _ in range(n_speech_tokens):
            hidden = self.backbone.forward_hidden(ids, inputs_embeds=embeds)
            latent = self.speech_latent_head(hidden[:, -1, :])
            latents.append(latent)
            embeds = torch.cat([embeds, latent.unsqueeze(1)], dim=1)
            ids = torch.cat([ids, torch.zeros(B, 1, device=device, dtype=torch.long)], dim=1)

        predicted = self.audio_decoder(torch.stack(latents, dim=1))
        target_mel = self._mel_extractor(speech_target)
        pred_mel = self._mel_extractor(predicted)
        min_frames = min(target_mel.shape[-1], pred_mel.shape[-1])
        return F.mse_loss(pred_mel[..., :min_frames], target_mel[..., :min_frames])

    def forward(
        self,
        text_ids: torch.Tensor,
        images: torch.Tensor | None = None,
        video_frames: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        speech_target: torch.Tensor | None = None,
        speech_loss_weight: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        input_ids, inputs_embeds = self.build_inputs(text_ids, images, video_frames, audio)
        logits = self.backbone(input_ids, inputs_embeds=inputs_embeds)

        result: dict[str, torch.Tensor] = {"logits": logits, "input_ids": input_ids}

        if labels is not None:
            # Cross-entropy over 32k vocab is unstable in fp16; compute in fp32.
            text_loss = self._text_loss(logits.float(), labels, label_smoothing)
            result["loss"] = text_loss
            result["text_loss"] = text_loss

        if speech_target is not None:
            speech_loss = self._speech_mel_loss(
                text_ids, speech_target, images, video_frames, audio
            )
            result["speech_loss"] = speech_loss
            if "loss" in result:
                result["loss"] = result["loss"] + speech_loss_weight * speech_loss
            else:
                result["loss"] = speech_loss

        return result

    @torch.no_grad()
    def generate(
        self,
        text_ids: torch.Tensor,
        images: torch.Tensor | None = None,
        video_frames: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        eos_token_id: int | None = None,
        rope_scale: float | None = None,
    ) -> torch.Tensor:
        self.eval()
        input_ids, inputs_embeds = self.build_inputs(text_ids, images, video_frames, audio)
        return self.backbone.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_token_id,
            inputs_embeds=inputs_embeds,
            rope_scale=rope_scale,
        )

    @torch.no_grad()
    def generate_speech(
        self,
        text_ids: torch.Tensor,
        images: torch.Tensor | None = None,
        video_frames: torch.Tensor | None = None,
        audio: torch.Tensor | None = None,
        n_speech_tokens: int | None = None,
        temperature: float = 0.8,
        rope_scale: float | None = None,
    ) -> torch.Tensor:
        """
        Generate speech audio from a multimodal prompt.

        Returns:
            (B, samples) mono waveform tensor.
        """
        self.eval()
        if rope_scale is not None:
            self.backbone.set_rope_scale(rope_scale)

        n_speech_tokens = n_speech_tokens or self.config.n_audio_tokens
        input_ids, inputs_embeds = self.build_inputs(text_ids, images, video_frames, audio)
        B = text_ids.shape[0]
        device = text_ids.device

        speech_marker = torch.full(
            (B, 1), SPECIAL_TOKENS["<speech>"], device=device, dtype=torch.long
        )
        embeds = torch.cat([inputs_embeds, self.backbone.token_emb(speech_marker)], dim=1)
        ids = torch.cat([input_ids, speech_marker], dim=1)

        latents: list[torch.Tensor] = []
        for _ in range(n_speech_tokens):
            hidden = self.backbone.forward_hidden(ids, inputs_embeds=embeds)
            h = hidden[:, -1, :] / max(temperature, 1e-5)
            latent = self.speech_latent_head(h)
            latents.append(latent)
            embeds = torch.cat([embeds, latent.unsqueeze(1)], dim=1)
            ids = torch.cat(
                [ids, torch.zeros(B, 1, device=device, dtype=torch.long)],
                dim=1,
            )

        audio_latents = torch.stack(latents, dim=1)
        return self.audio_decoder(audio_latents)

    def set_encoders_trainable(self, trainable: bool) -> None:
        for module in (
            self.vision_encoder,
            self.video_encoder,
            self.audio_encoder,
            self.audio_decoder,
            self.modality_proj,
        ):
            for param in module.parameters():
                param.requires_grad = trainable