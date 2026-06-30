import math
import struct
import wave

import torch

from config import ModelConfig, SPECIAL_TOKENS
from model import MultimodalModel
from model.audio_decoder import AudioDecoder, MelVocoder
from model.audio_encoder import AudioEncoder, LogMelSpectrogram
from utils.media import load_audio, make_sine_wave, save_audio


def _write_wav(path, waveform: torch.Tensor, sample_rate: int = 16000) -> None:
    samples = (waveform.clamp(-1, 1) * 32767).short().tolist()
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def test_log_mel_spectrogram_shape():
    mel = LogMelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=80)
    waveform = make_sine_wave(duration_sec=1.0)
    out = mel(waveform.unsqueeze(0))
    assert out.shape[0] == 1
    assert out.shape[1] == 80
    assert out.shape[2] > 10


def test_audio_encoder_output_shape():
    cfg = ModelConfig(
        d_model=128,
        n_audio_tokens=16,
        audio_duration_sec=1.0,
        audio_sample_rate=16000,
    )
    encoder = AudioEncoder(
        d_model=cfg.d_model,
        sample_rate=cfg.audio_sample_rate,
        n_mels=cfg.n_mel_bins,
        n_fft=cfg.audio_n_fft,
        hop_length=cfg.audio_hop_length,
        max_samples=cfg.max_audio_samples,
        n_tokens=cfg.n_audio_tokens,
        n_heads=4,
        n_layers=1,
    )
    waveform = make_sine_wave(duration_sec=1.0).unsqueeze(0)
    tokens = encoder(waveform)
    assert tokens.shape == (1, cfg.n_audio_tokens, cfg.d_model)


def test_load_audio_from_wav(tmp_path):
    wav_path = tmp_path / "tone.wav"
    _write_wav(wav_path, make_sine_wave(frequency=220.0, duration_sec=0.5))

    audio = load_audio(wav_path, sample_rate=16000, max_duration_sec=1.0)
    assert audio.shape == (1, 16000)
    assert audio.abs().max() > 0


def test_multimodal_build_inputs_with_audio():
    cfg = ModelConfig(
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=256,
        vocab_size=1000,
        image_size=112,
        patch_size=16,
        n_video_frames=2,
        audio_duration_sec=1.0,
        n_audio_tokens=8,
    )
    model = MultimodalModel(cfg)
    text_ids = torch.randint(9, 50, (2, 16))
    audio = make_sine_wave(duration_sec=1.0).unsqueeze(0).expand(2, -1)

    input_ids, inputs_embeds = model.build_inputs(text_ids, audio=audio)

    assert inputs_embeds.shape[0] == 2
    assert inputs_embeds.shape[2] == cfg.d_model
    assert input_ids.shape[1] == inputs_embeds.shape[1]
    assert (input_ids[:, 0] == SPECIAL_TOKENS["<audio>"]).all()


def test_multimodal_forward_with_audio():
    cfg = ModelConfig(
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=256,
        vocab_size=1000,
        image_size=112,
        patch_size=16,
        n_video_frames=2,
        audio_duration_sec=1.0,
        n_audio_tokens=8,
    )
    model = MultimodalModel(cfg)
    text_ids = torch.randint(9, 50, (2, 16))
    labels = text_ids.clone()
    audio = make_sine_wave(duration_sec=1.0).unsqueeze(0).expand(2, -1)

    out = model(text_ids=text_ids, audio=audio, labels=labels)

    assert out["loss"].item() > 0
    assert out["logits"].shape[0] == 2
    assert out["logits"].shape[2] == cfg.vocab_size


def test_multimodal_generate_with_audio():
    cfg = ModelConfig(
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=256,
        vocab_size=1000,
        audio_duration_sec=1.0,
        n_audio_tokens=8,
    )
    model = MultimodalModel(cfg)
    model.eval()
    text_ids = torch.tensor([[10, 11, 12]])
    audio = make_sine_wave(duration_sec=1.0).unsqueeze(0)

    with torch.no_grad():
        output = model.generate(text_ids=text_ids, audio=audio, max_new_tokens=4)

    assert output.shape[0] == 1
    assert output.shape[1] >= text_ids.shape[1]


def test_audio_decoder_output_shape():
    cfg = ModelConfig(
        d_model=128,
        n_audio_tokens=16,
        audio_duration_sec=1.0,
        audio_sample_rate=16000,
    )
    decoder = AudioDecoder(
        d_model=cfg.d_model,
        sample_rate=cfg.audio_sample_rate,
        n_mels=cfg.n_mel_bins,
        n_fft=cfg.audio_n_fft,
        hop_length=cfg.audio_hop_length,
        max_samples=cfg.max_audio_samples,
        n_tokens=cfg.n_audio_tokens,
        n_heads=4,
        n_layers=1,
    )
    latents = torch.randn(2, cfg.n_audio_tokens, cfg.d_model)
    waveform = decoder(latents)
    assert waveform.shape == (2, cfg.max_audio_samples)


def test_mel_vocoder_produces_waveform():
    mel = LogMelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=80)
    vocoder = MelVocoder(sample_rate=16000, n_fft=400, hop_length=160, n_mels=80, n_iter=4)
    waveform = make_sine_wave(frequency=440.0, duration_sec=0.25).unsqueeze(0)
    log_mel = mel(waveform)
    reconstructed = vocoder(log_mel)
    assert reconstructed.shape[0] == 1
    assert reconstructed.shape[1] > 1000
    assert reconstructed.abs().max() > 0


def test_generate_speech_returns_waveform():
    cfg = ModelConfig(
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=256,
        vocab_size=1000,
        audio_duration_sec=1.0,
        n_audio_tokens=8,
    )
    model = MultimodalModel(cfg)
    model.eval()
    text_ids = torch.tensor([[10, 11, 12]])

    with torch.no_grad():
        speech = model.generate_speech(text_ids=text_ids, n_speech_tokens=8)

    assert speech.shape == (1, cfg.max_audio_samples)
    assert speech.abs().max() > 0


def test_save_and_load_generated_speech(tmp_path):
    cfg = ModelConfig(
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=256,
        vocab_size=1000,
        audio_duration_sec=0.5,
        audio_sample_rate=16000,
        n_audio_tokens=8,
    )
    model = MultimodalModel(cfg)
    model.eval()

    with torch.no_grad():
        speech = model.generate_speech(
            text_ids=torch.tensor([[12, 13, 14]]),
            n_speech_tokens=cfg.n_audio_tokens,
        )

    out_path = tmp_path / "speech.wav"
    save_audio(out_path, speech.squeeze(0), cfg.audio_sample_rate)
    loaded = load_audio(out_path, sample_rate=cfg.audio_sample_rate, max_duration_sec=cfg.audio_duration_sec)
    assert loaded.shape == (1, cfg.max_audio_samples)
    assert loaded.abs().max() > 0