# Insanely Fast Whisper - Benchmark Guide for AMD GPU

## About the Model

Insanely Fast Whisper is a highly optimized implementation of OpenAI's Whisper automatic speech recognition (ASR) model. It's built on HuggingFace Transformers with restructured attention layers to enable GPUs to process larger chunks of data simultaneously, achieving significantly faster transcription speeds. The implementation can transcribe 150 minutes (2.5 hours) of audio in less than 98 seconds using Whisper Large v3.

### Original Whisper Paper

**"Robust Speech Recognition via Large-Scale Weak Supervision"** (Radford et al., 2022)

Whisper is a general-purpose speech recognition model trained on 680,000 hours of multilingual and multitask supervised data collected from the web. When scaled to this massive dataset, the resulting models generalize well to standard benchmarks and are often competitive with prior fully supervised results in a zero-shot transfer setting without fine-tuning. The models approach human-level accuracy and robustness, and can perform multilingual speech recognition, speech translation, spoken language identification, and voice activity detection.

**Paper:** [arXiv:2212.04356](https://arxiv.org/abs/2212.04356) | **Published:** ICML 2023

---

## Standard Benchmark Dataset: LibriSpeech

**LibriSpeech** is the industry-standard benchmark for evaluating English ASR systems. It contains ~1000 hours of 16kHz read English speech derived from LibriVox audiobooks.

### Dataset Structure
- **test-clean**: Clean speech (2,620 samples)
- **test-other**: More challenging speech conditions

### Download from HuggingFace

```bash
# Install dependencies
pip install datasets transformers
```

```python
from datasets import load_dataset

# Load LibriSpeech test-clean split
dataset = load_dataset("openslr/librispeech_asr", "clean", split="test")

# Or use the dedicated test-clean dataset
dataset = load_dataset("AudioLLMs/librispeech_test_clean")

# View a sample
print(dataset[0])
# Output: {'file': '/path/to/audio.flac', 'text': 'the transcription', 'speaker_id': 1234, ...}
```

---

## Installation & Inference

### Install Insanely Fast Whisper

```bash
# Install using pipx (recommended)
pipx install insanely-fast-whisper==0.0.15 --force

# Or using pip
pip install insanely-fast-whisper
```

### Basic Inference

```bash
# Standard usage
insanely-fast-whisper --file-name audio.mp3 --device-id 0

# With Flash Attention 2 (requires compatible GPU)
insanely-fast-whisper --file-name audio.mp3 --flash True --device-id 0

# Specify model and output
insanely-fast-whisper \
  --model-name openai/whisper-large-v3 \
  --file-name audio.mp3 \
  --transcript-path output.json \
  --device-id 0

# Using distil-whisper for faster inference
insanely-fast-whisper \
  --model-name distil-whisper/large-v2 \
  --file-name audio.mp3
```

### Python API Inference

```python
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Inference
result = pipe("audio.mp3")
print(result["text"])
```

### Expected Output

```json
{
  "text": "The full transcription of the audio file...",
  "chunks": [
    {
      "timestamp": [0.0, 5.2],
      "text": "First segment of speech"
    },
    {
      "timestamp": [5.2, 10.4],
      "text": "Second segment of speech"
    }
  ]
}
```

---

## Benchmark Results & Performance Metrics

### Whisper Performance on LibriSpeech

| Model | test-clean WER | test-other WER | Dataset Size | Training |
|-------|---------------|----------------|--------------|----------|
| **Whisper Large v3** | 2.07% | - | 680K hours | Zero-shot |
| **Whisper Large** | 2.7% | 5.2% | 680K hours | Zero-shot |
| Wav2vec 2.0 | 1.8% | 3.3% | 60K hours | Fine-tuned |
| SpeechBrain | 2.46% | 5.77% | - | Fine-tuned |
| Kaldi | 3.8% | 8.76% | - | Fine-tuned |

**WER** = Word Error Rate (lower is better)

### Performance: Insanely Fast Whisper vs Alternatives

| Implementation | Relative Speed | Platform | Notes |
|----------------|---------------|----------|-------|
| **Insanely Fast Whisper** | **3-4x faster** | NVIDIA (B300) GPU, Apple Silicon | Flash Attention 2, batch processing |
| **Insanely Fast Whisper** | **5-6x faster** | AMD (MI455X) GPU + NPU | Flash Attention 2, batch processing |
| Faster Whisper | 2-3x faster | NVIDIA GPU, CPU | CTranslate2 backend |
| Original Whisper | 1x baseline | GPU, CPU | PyTorch implementation |
| WhisperX | Variable | GPU | Word-level timestamps, alignment |

**Benchmark:** 150 minutes of audio transcribed in <98 seconds (Whisper Large v3 on A100-80GB)

---

## AMD GPU Benchmarking Setup

### ROCm Installation for AMD GPUs

```bash
# Check ROCm compatibility
rocm-smi

# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### Benchmark Script for AMD GPU

```python
import torch
import time
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16

# Load model
model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
).to(device)
processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Load LibriSpeech test-clean
dataset = load_dataset("AudioLLMs/librispeech_test_clean", split="test[:10]")

# Benchmark
results = []
for i, sample in enumerate(dataset):
    audio_duration = len(sample["audio"]["array"]) / sample["audio"]["sampling_rate"]

    start_time = time.time()
    result = pipe(sample["audio"]["array"])
    end_time = time.time()

    inference_time = end_time - start_time
    rtf = inference_time / audio_duration  # Real-Time Factor

    results.append({
        "sample_id": i,
        "audio_duration": audio_duration,
        "inference_time": inference_time,
        "rtf": rtf,
        "predicted": result["text"],
        "reference": sample["text"]
    })

    print(f"Sample {i}: Duration={audio_duration:.2f}s, Inference={inference_time:.2f}s, RTF={rtf:.3f}")

# Summary statistics
avg_rtf = np.mean([r["rtf"] for r in results])
print(f"\nAverage Real-Time Factor: {avg_rtf:.3f}")
print(f"Throughput: {1/avg_rtf:.2f}x real-time")
```

### Performance Metrics Table Template

| Metric | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Notes |
|--------|------------------|-----------|------------|-----------------|-------|
| **GPU Model** | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Compare datacenter vs consumer GPUs |
| **Memory (GB)** | 80 | 16 | 192 | 24 | VRAM capacity |
| **TDP (W)** | 400 | 70 | 750 | 355 | Thermal design power |
| **Audio Duration (minutes)** | 150 | 150 | _[Your result]_ | _[Your result]_ | Standard benchmark duration |
| **Transcription Time (seconds)** | 98 | ~300 | _[Your result]_ | _[Your result]_ | Lower is better |
| **Real-Time Factor (RTF)** | 0.011 | 0.033 | _[Your result]_ | _[Your result]_ | <1.0 is faster than real-time |
| **Throughput (x real-time)** | 91.8x | 30x | _[Your result]_ | _[Your result]_ | How many times faster than real-time |
| **Peak Memory Usage (GB)** | ~20 | ~12 | _[Your result]_ | _[Your result]_ | Monitor with rocm-smi |
| **Average Power Draw (W)** | ~300 | ~60 | _[Your result]_ | _[Your result]_ | Monitor with rocm-smi --showpower |
| **Energy per Hour of Audio (Wh)** | ~8.2 | ~30 | _[Your result]_ | _[Your result]_ | Lower is better |

### AMD-Specific Metrics to Track

```python
# GPU utilization tracking
import subprocess

def get_rocm_smi_stats():
    """Get AMD GPU statistics using rocm-smi"""
    result = subprocess.run(['rocm-smi', '--showuse', '--showmeminfo', 'vram'],
                          capture_output=True, text=True)
    return result.stdout

# Memory tracking
print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
print(f"Max Allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

# ROCm info
print(f"ROCm Version: {torch.version.hip}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
```

### Complete Runtime Metrics Table

| Runtime Metric | Formula | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Notes |
|----------------|---------|------------------|-----------|------------|-----------------|-------|
| **Real-Time Factor (RTF)** | inference_time / audio_duration | 0.011 | 0.033 | _[Your result]_ | _[Your result]_ | <1.0 is faster than real-time |
| **Throughput** | 1 / RTF | 91.8x | 30x | _[Your result]_ | _[Your result]_ | How many times faster than real-time |
| **Words Per Minute (WPM)** | word_count / (inference_time / 60) | _[Reference]_ | _[Reference]_ | _[Your result]_ | _[Your result]_ | Transcription speed |
| **GPU Utilization (%)** | From nvidia-smi / rocm-smi | _[Reference]_ | _[Reference]_ | _[Your result]_ | _[Your result]_ | Average during inference |
| **Memory Bandwidth (GB/s)** | From nvidia-smi / rocm-smi | ~2.0 TB/s | ~320 GB/s | _[Your result]_ | _[Your result]_ | MI300X: ~5.3 TB/s, RX 7900 XTX: ~960 GB/s theoretical |
| **TFLOPS Utilized** | Calculated from operations | _[Reference]_ | _[Reference]_ | _[Your result]_ | _[Your result]_ | FP16 compute throughput |
| **Latency (ms)** | Time to first token | _[Reference]_ | _[Reference]_ | _[Your result]_ | _[Your result]_ | Important for streaming |
| **Energy Efficiency (Wh/hour)** | power_draw × time / audio_duration | ~8.2 | ~30 | _[Your result]_ | _[Your result]_ | Lower is better |

---

## HuggingFace Open ASR Leaderboard

The [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) evaluates speech recognition models across multiple datasets:

### Evaluation Datasets
- **AMI** (Meeting transcription)
- **Earnings22** (Earnings calls)
- **GigaSpeech** (Diverse audio)
- **LibriSpeech** (test-clean, test-other)
- **SPGISpeech** (Financial audio)
- **Tedlium** (TED talks)
- **VoxPopuli** (European Parliament)
- **Common Voice** (Multilingual crowdsourced)

### Key Metrics Tracked
- **WER** (Word Error Rate) - primary metric
- **CER** (Character Error Rate)
- **RTF** (Real-Time Factor)
- **Model Size** (parameters, disk size)

**Note:** Insanely Fast Whisper is an optimization technique, not a separate model on the leaderboard. The underlying Whisper models (Large v3, etc.) are evaluated.

---

## Additional Resources

### Official Repositories
- [Insanely Fast Whisper GitHub](https://github.com/Vaibhavs10/insanely-fast-whisper)
- [OpenAI Whisper GitHub](https://github.com/openai/whisper)
- [Faster Whisper (CTranslate2)](https://github.com/SYSTRAN/faster-whisper)

### Papers & Documentation
- [Whisper Paper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Whisper Paper (PDF)](https://cdn.openai.com/papers/whisper.pdf)
- [MLCommons Whisper Benchmark](https://mlcommons.org/2025/09/whisper-inferencev5-1/)
- [HuggingFace Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)

### Blog Posts & Comparisons
- [Modal: Choosing between Whisper variants](https://modal.com/blog/choosing-whisper-variants)
- [HuggingFace: Blazingly fast whisper transcriptions](https://huggingface.co/blog/fast-whisper-endpoints)
- [AMD: Whisper on Ryzen AI NPUs](https://www.amd.com/en/developer/resources/technical-articles/2025/unlocking-on-device-asr-with-whisper-on-ryzen-ai-npus.html)
- [AMD ROCm Performance Results](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html)

### Datasets
- [LibriSpeech (openslr/librispeech_asr)](https://huggingface.co/datasets/openslr/librispeech_asr)
- [LibriSpeech test-clean (AudioLLMs)](https://huggingface.co/datasets/AudioLLMs/librispeech_test_clean)
- [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0)

---

## Quick Reference Commands

```bash
# Install Insanely Fast Whisper
pipx install insanely-fast-whisper==0.0.15 --force

# Run benchmark on single file
insanely-fast-whisper --file-name test.mp3 --device-id 0 --flash True

# Check AMD GPU status
rocm-smi
rocm-smi --showuse --showmeminfo vram

# Download LibriSpeech test-clean
python -c "from datasets import load_dataset; ds = load_dataset('AudioLLMs/librispeech_test_clean')"

# Get help
insanely-fast-whisper --help
```

---

**Document Version:** 1.0
**Last Updated:** March 2026
**Target Hardware:** AMD MI300X, RX 7900 XTX, and other ROCm-compatible GPUs
