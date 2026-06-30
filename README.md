# Multimodal Transformer

A from-scratch multimodal language model that accepts text, images, video, and audio in a single decoder-only transformer. Training data comes from Wikipedia, Gutenberg, GitHub, COCO, LibriVox, and MSR-VTT, merged into a unified `corpus.jsonl`.

## Important: designed for cloud GPU training

**This model is intended to be trained on cloud infrastructure with large GPU clusters**, not on a single consumer GPU or a short Colab session.

Training from scratch on a broad multimodal corpus requires:

- **Hundreds of thousands to millions** of optimizer steps
- **Multi-GPU or multi-node** setups for reasonable wall-clock time at full model size
- **Stable, high-bandwidth storage** for datasets and checkpoints (not a mounted Google Drive folder)

Colab (even Pro+) is useful for **prototyping**, debugging the data pipeline, and running the smaller `--small` config. It is not a substitute for cluster-scale training. Expect slow convergence, session timeouts, and Drive I/O issues if you rely on Colab for long runs.

For production-quality results, plan on cloud VMs with local NVMe storage, distributed training, or fine-tuning a pretrained vision-language model instead of training this architecture from scratch on a laptop.

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.10+ and a CUDA-enabled PyTorch build are recommended for training.

## Quick start

### 1. Prepare data

```bash
python check_data_ready.py          # see what is missing
python process_all.py               # build corpus, tokenizer, wiki QA eval set
python process_all.py --whisper-device cuda   # faster LibriVox transcription on GPU
```

Individual download/process scripts live in the repo root (`download_*.py`, `process_*.py`).

### 2. Train

Full model (needs a large GPU):

```bash
python train.py --device cuda
```

Colab / single T4 prototype (`ModelConfig.colab()` — smaller layers, less VRAM):

```bash
python train.py --device cuda --small --max-steps 500000 --resume checkpoints/best.pt
```

Text-only mode (faster iteration, no vision/audio/video I/O):

```bash
python train.py --device cuda --text-only
```

### 3. Evaluate and run inference

```bash
python evaluate.py --checkpoint checkpoints/best.pt
python inference.py --checkpoint checkpoints/best.pt --prompt "The capital of France is"
python inference.py --checkpoint checkpoints/best.pt --prompt "Describe this image" --image photo.jpg
```

## Colab notes

When running on Google Colab:

- **Checkpoints** are written to `/content/transformer_local/checkpoints` and synced to `checkpoints/` on Drive to avoid Drive timeouts during saves.
- **Background sync:** `python sync_checkpoints.py --watch 300`
- **Drive I/O errors** on image/audio paths are skipped gracefully; affected batches fall back to text-only. Use `--num-workers 0` if errors are frequent.
- **Resume** after disconnect: `python train.py --device cuda --small --resume checkpoints/best.pt`

## Project layout

| Path | Purpose |
|------|---------|
| `config.py` | Model, training, and data paths |
| `train.py` | Main training loop |
| `model/` | Transformer backbone and modality encoders |
| `data/` | Dataset, collation, mixed-batch sampler |
| `eval/` | Validation loss, Wikipedia QA, audio-caption metrics |
| `checkpoints/` | Saved model weights and `training_metrics.jsonl` |
| `utils/drive_sync.py` | Colab local ↔ Drive checkpoint sync |

## Tests

```bash
python -m pytest tests/ -v
```