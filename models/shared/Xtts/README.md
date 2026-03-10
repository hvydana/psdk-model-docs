# XTTS - Benchmark Guide for AMD GPU

## About the Model

XTTS (Cross-lingual Text-to-Speech) is a massively multilingual zero-shot text-to-speech model developed by Coqui AI. Built on the Tortoise architecture with novel enhancements, XTTS enables high-quality voice cloning with just a 6-second audio clip and supports real-time streaming synthesis with <200ms latency. The model achieves state-of-the-art performance across 16 languages and can generate natural-sounding speech in any supported language using voice samples from speakers in different languages.

### Original XTTS Paper

**"XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model"** (Casanova et al., 2024)

XTTS addresses limitations in existing multilingual TTS systems that were restricted to just a few high/medium resource languages. The model expands support across low-resource languages by building on the Tortoise architecture with modifications enabling multilingual training capability, enhanced voice cloning performance, and accelerated training and inference speeds. XTTS is trained on over 27,281 hours of speech data across 16 languages and achieves state-of-the-art results in naturalness, acoustic quality, and human likeness across most supported languages.

**Paper:** [arXiv:2406.04904](https://arxiv.org/abs/2406.04904) | **Published:** INTERSPEECH 2024

**Supported Languages:** English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu), Korean (ko)

---

## Standard Benchmark Datasets for TTS

### LibriTTS

**LibriTTS** is the industry-standard benchmark for multi-speaker English TTS systems. It contains approximately 585 hours of read English speech at 24kHz sampling rate, derived from the original LibriSpeech corpus and designed specifically for TTS research.

### Dataset Structure
- **train-clean-100**: Clean training speech (100 hours)
- **train-clean-360**: Clean training speech (360 hours)
- **train-other-500**: Other training conditions (500 hours)
- **dev-clean**: Development set with clean speech
- **dev-other**: Development set with challenging conditions
- **test-clean**: Test set with clean speech
- **test-other**: Test set with challenging conditions

### Download from HuggingFace

```bash
# Install dependencies
pip install datasets transformers
```

```python
from datasets import load_dataset

# Load LibriTTS test-clean split
dataset = load_dataset("cdminix/libritts-aligned", "clean", split="test")

# Or load the full LibriTTS dataset
dataset = load_dataset("parler-tts/libritts_r_filtered", split="train.clean.100")

# View a sample
print(dataset[0])
# Output: {'audio': {'array': [...], 'sampling_rate': 24000}, 'text_normalized': '...', 'speaker_id': 1234, ...}
```

### VCTK (CSTR VCTK Corpus)

**VCTK** is a multi-speaker English corpus comprising nearly 44,000 short clips from 109 native speakers with various accents. Recorded at 48kHz, it provides excellent resources for training models that need to handle accent variations and speaker diversity.

```python
from datasets import load_dataset

# Load VCTK dataset
dataset = load_dataset("vctk", split="train")

# View a sample
print(dataset[0])
# Output: {'audio': {...}, 'text': '...', 'speaker_id': 'p225', 'accent': 'English', ...}
```

### LJSpeech

**LJSpeech** consists of 13,100 short audio clips totaling roughly 24 hours, featuring a single speaker reading passages from non-fiction books. It's commonly used for single-speaker TTS model training and evaluation.

```python
from datasets import load_dataset

# Load LJSpeech dataset
dataset = load_dataset("lj_speech", split="train")

# View a sample
print(dataset[0])
# Output: {'audio': {...}, 'text': '...', 'normalized_text': '...', 'id': '...'}
```

---

## Installation & Inference

### Install XTTS via Coqui TTS

```bash
# Install PyTorch with CUDA support first (for NVIDIA GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Coqui TTS (maintained fork)
pip install coqui-tts

# Or install the original TTS package
pip install TTS
```

### Basic Inference

```bash
# Command line usage with voice cloning
tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --text "Hello world! This is a test of XTTS voice cloning." \
    --speaker_wav /path/to/reference/speaker.wav \
    --language_idx en \
    --use_cuda true \
    --out_path output.wav

# Multi-language example (Turkish)
tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --text "Bugün okula gitmek istemiyorum." \
    --speaker_wav /path/to/reference/speaker.wav \
    --language_idx tr \
    --use_cuda true \
    --out_path output.wav
```

### Python API Inference (High-level)

```python
from TTS.api import TTS
import torch

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize TTS with XTTS v2
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Generate speech by cloning a voice
tts.tts_to_file(
    text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
    speaker_wav="/path/to/reference/speaker.wav",
    language="en",
    file_path="output.wav"
)

# Cross-lingual voice cloning (English speaker, French text)
tts.tts_to_file(
    text="Bonjour, comment allez-vous aujourd'hui?",
    speaker_wav="/path/to/english/speaker.wav",
    language="fr",
    file_path="output_french.wav"
)
```

### Python API Inference (Low-level)

```python
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load configuration
config = XttsConfig()
config.load_json("/path/to/xtts/config.json")

# Initialize model
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="/path/to/xtts/", eval=True)
model.to(device)

# Synthesize speech
outputs = model.synthesize(
    "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    config,
    speaker_wav="/path/to/reference/speaker.wav",
    gpt_cond_len=3,
    language="en",
)

# Save audio
import torchaudio
torchaudio.save("output.wav", torch.tensor(outputs["wav"]).unsqueeze(0), 24000)
```

### Expected Output

The model generates a WAV file with 24kHz sampling rate containing synthesized speech that mimics the voice characteristics of the reference speaker while speaking the input text.

**Audio Properties:**
- Sampling rate: 24,000 Hz
- Channels: 1 (mono)
- Bit depth: 16-bit PCM (typical)
- Duration: Variable based on input text length

---

## Benchmark Results & Performance Metrics

### XTTS Performance on Standard Benchmarks

| Model | Naturalness (CMOS) | Speaker Similarity (SECS) | Character Error Rate (CER) | Languages | Training Data |
|-------|-------------------|---------------------------|---------------------------|-----------|---------------|
| **XTTS v2** | **Superior** | 0.85+ | **Low** | 16 | 27,281 hours |
| XTTS v1 | Good | 0.80+ | Moderate | 13 | ~15,000 hours |
| YourTTS | Moderate | 0.75+ | Moderate | 6 | ~3,000 hours |
| Mega-TTS 2 | Good | 0.78+ | Moderate | 8 | ~10,000 hours |
| StyleTTS 2 | Superior | 0.88+ | Very Low | 1 (en) | 585 hours |

**Metrics:**
- **CMOS** = Comparative Mean Opinion Score (higher is better, measures naturalness)
- **SECS** = Speaker Encoder Cosine Similarity (higher is better, 0-1 scale)
- **CER** = Character Error Rate (lower is better, measures intelligibility)

### XTTS Specific Performance

According to the INTERSPEECH 2024 paper:
- **Naturalness, Acoustic Quality, Human Likeness**: State-of-the-art across most supported languages
- **Speaker Similarity**: Slightly lower than monolingual systems (expected due to multilingual training complexity)
- **Zero-shot Voice Cloning**: Requires only 6 seconds of reference audio
- **Streaming Latency**: <200ms for real-time applications
- **Cross-lingual Capability**: Can clone voices across different languages

### Real-Time Performance Comparison

| Implementation | RTF (Real-Time Factor) | Latency | Platform | Notes |
|----------------|------------------------|---------|----------|-------|
| **XTTS v2** | **0.15-0.30** | <200ms | GPU | Streaming capable |
| XTTS v1 | 0.35-0.50 | ~300ms | GPU | Slower than v2 |
| Coqui TTS (VITS) | 0.10-0.20 | <150ms | GPU | Faster but less natural |
| StyleTTS2 | 0.20-0.35 | ~250ms | GPU | Good quality-speed balance |
| Piper TTS | 0.05-0.10 | <100ms | CPU/GPU | Very fast, lower quality |

**RTF** = Real-Time Factor (processing_time / audio_duration, <1.0 means faster than real-time)

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

# Install Coqui TTS
pip install coqui-tts
```

### Benchmark Script for AMD GPU

```python
import torch
import time
from datasets import load_dataset
from TTS.api import TTS
import numpy as np
import librosa

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load XTTS model
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
tts = TTS(model_name).to(device)

# Load LibriTTS test dataset
dataset = load_dataset("cdminix/libritts-aligned", "clean", split="test[:10]")

# Prepare reference speaker
reference_speaker = dataset[0]["audio"]["array"]
import tempfile
import soundfile as sf

# Save reference audio temporarily
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_ref:
    sf.write(temp_ref.name, reference_speaker, 24000)
    reference_path = temp_ref.name

# Benchmark
results = []
for i, sample in enumerate(dataset):
    text = sample["text_normalized"]

    # Measure synthesis time
    start_time = time.time()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_out:
        tts.tts_to_file(
            text=text,
            speaker_wav=reference_path,
            language="en",
            file_path=temp_out.name
        )
        output_path = temp_out.name

    end_time = time.time()

    # Calculate audio duration
    audio_output, sr = librosa.load(output_path, sr=24000)
    audio_duration = len(audio_output) / sr

    synthesis_time = end_time - start_time
    rtf = synthesis_time / audio_duration  # Real-Time Factor

    results.append({
        "sample_id": i,
        "text_length": len(text),
        "audio_duration": audio_duration,
        "synthesis_time": synthesis_time,
        "rtf": rtf,
        "text": text[:50] + "..." if len(text) > 50 else text
    })

    print(f"Sample {i}: Duration={audio_duration:.2f}s, Synthesis={synthesis_time:.2f}s, RTF={rtf:.3f}")

# Summary statistics
avg_rtf = np.mean([r["rtf"] for r in results])
avg_duration = np.mean([r["audio_duration"] for r in results])
avg_synthesis = np.mean([r["synthesis_time"] for r in results])

print(f"\n=== Benchmark Results ===")
print(f"Average Audio Duration: {avg_duration:.2f}s")
print(f"Average Synthesis Time: {avg_synthesis:.2f}s")
print(f"Average Real-Time Factor: {avg_rtf:.3f}")
print(f"Throughput: {1/avg_rtf:.2f}x real-time" if avg_rtf > 0 else "N/A")
```

### Performance Metrics Table Template

| Metric | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Notes |
|--------|------------------|-----------|------------|-----------------|-------|
| **GPU Model** | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Compare datacenter vs consumer GPUs |
| **Memory (GB)** | 80 | 16 | 192 | 24 | VRAM capacity |
| **TDP (W)** | 400 | 70 | 750 | 355 | Thermal design power |
| **Average Text Length (chars)** | 150 | 150 | _[Your result]_ | _[Your result]_ | Standard benchmark text |
| **Audio Duration (seconds)** | 10.0 | 10.0 | _[Your result]_ | _[Your result]_ | Generated audio length |
| **Synthesis Time (seconds)** | 1.5 | 3.0 | _[Your result]_ | _[Your result]_ | Lower is better |
| **Real-Time Factor (RTF)** | 0.15 | 0.30 | _[Your result]_ | _[Your result]_ | <1.0 is faster than real-time |
| **Throughput (x real-time)** | 6.7x | 3.3x | _[Your result]_ | _[Your result]_ | How many times faster than real-time |
| **Peak Memory Usage (GB)** | ~8 | ~6 | _[Your result]_ | _[Your result]_ | Monitor with rocm-smi |
| **Average Power Draw (W)** | ~280 | ~55 | _[Your result]_ | _[Your result]_ | Monitor with rocm-smi --showpower |
| **Energy per Minute of Audio (Wh)** | ~0.7 | ~2.75 | _[Your result]_ | _[Your result]_ | Lower is better |
| **First Chunk Latency (ms)** | ~150 | ~200 | _[Your result]_ | _[Your result]_ | Important for streaming |

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

# Power monitoring during synthesis
def monitor_power():
    """Monitor GPU power consumption"""
    result = subprocess.run(['rocm-smi', '--showpower'],
                          capture_output=True, text=True)
    return result.stdout
```

### Complete Runtime Metrics Table

| Runtime Metric | Formula | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Notes |
|----------------|---------|------------------|-----------|------------|-----------------|-------|
| **Real-Time Factor (RTF)** | synthesis_time / audio_duration | 0.15 | 0.30 | _[Your result]_ | _[Your result]_ | <1.0 is faster than real-time |
| **Throughput** | 1 / RTF | 6.7x | 3.3x | _[Your result]_ | _[Your result]_ | How many times faster than real-time |
| **Characters Per Second** | char_count / synthesis_time | ~100 | ~50 | _[Your result]_ | _[Your result]_ | Text processing speed |
| **GPU Utilization (%)** | From nvidia-smi / rocm-smi | ~75 | ~85 | _[Your result]_ | _[Your result]_ | Average during inference |
| **Memory Bandwidth (GB/s)** | From nvidia-smi / rocm-smi | ~2.0 TB/s | ~320 GB/s | _[Your result]_ | _[Your result]_ | MI300X: ~5.3 TB/s, RX 7900 XTX: ~960 GB/s theoretical |
| **TFLOPS Utilized** | Calculated from operations | ~50 | ~8 | _[Your result]_ | _[Your result]_ | FP16 compute throughput |
| **First Chunk Latency (ms)** | Time to first audio chunk | ~150 | ~200 | _[Your result]_ | _[Your result]_ | Critical for streaming |
| **Energy Efficiency (Wh/min)** | power_draw × time / audio_duration | ~0.7 | ~2.75 | _[Your result]_ | _[Your result]_ | Lower is better |

### Advanced Benchmarking with Audio Quality Metrics

```python
import torch
import torchaudio
from TTS.api import TTS
from datasets import load_dataset
import numpy as np
from pesq import pesq
from pystoi import stoi

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Load test dataset
dataset = load_dataset("cdminix/libritts-aligned", "clean", split="test[:5]")

quality_results = []
for i, sample in enumerate(dataset):
    # Get reference audio
    reference_audio = sample["audio"]["array"]
    text = sample["text_normalized"]

    # Synthesize
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp:
        tts.tts_to_file(
            text=text,
            speaker_wav=reference_audio,
            language="en",
            file_path=temp.name
        )

        # Load synthesized audio
        synthesized, sr = torchaudio.load(temp.name)
        synthesized = synthesized.numpy()[0]

    # Ensure same length for comparison
    min_len = min(len(reference_audio), len(synthesized))
    ref = reference_audio[:min_len]
    syn = synthesized[:min_len]

    # Calculate quality metrics (if reference audio available)
    try:
        # PESQ (Perceptual Evaluation of Speech Quality): 1.0-4.5, higher is better
        pesq_score = pesq(24000, ref, syn, 'wb')

        # STOI (Short-Time Objective Intelligibility): 0-1, higher is better
        stoi_score = stoi(ref, syn, 24000, extended=False)

        quality_results.append({
            "sample": i,
            "pesq": pesq_score,
            "stoi": stoi_score
        })

        print(f"Sample {i}: PESQ={pesq_score:.2f}, STOI={stoi_score:.3f}")
    except Exception as e:
        print(f"Sample {i}: Quality metrics calculation failed - {e}")

# Summary
if quality_results:
    avg_pesq = np.mean([r["pesq"] for r in quality_results])
    avg_stoi = np.mean([r["stoi"] for r in quality_results])
    print(f"\nAverage PESQ: {avg_pesq:.2f} (1.0-4.5 scale)")
    print(f"Average STOI: {avg_stoi:.3f} (0-1 scale)")
```

---

## TTS Evaluation Metrics

### Objective Metrics

| Metric | Description | Range | Better | Usage |
|--------|-------------|-------|--------|-------|
| **MCD** | Mel-Cepstral Distortion - measures spectral difference | 0-10+ dB | Lower | Naturalness |
| **F0 RMSE** | Root Mean Square Error of fundamental frequency | 0-100+ Hz | Lower | Pitch accuracy |
| **F0 CORR** | Correlation of fundamental frequency | 0-1 | Higher | Pitch similarity |
| **CER** | Character Error Rate from ASR | 0-100% | Lower | Intelligibility |
| **WER** | Word Error Rate from ASR | 0-100% | Lower | Intelligibility |
| **PESQ** | Perceptual Evaluation of Speech Quality | 1.0-4.5 | Higher | Perceived quality |
| **STOI** | Short-Time Objective Intelligibility | 0-1 | Higher | Intelligibility |
| **RTF** | Real-Time Factor (synthesis_time/audio_duration) | 0+ | Lower | Speed |
| **SECS** | Speaker Encoder Cosine Similarity | 0-1 | Higher | Speaker similarity |

### Subjective Metrics

| Metric | Description | Range | Better | Usage |
|--------|-------------|-------|--------|-------|
| **MOS** | Mean Opinion Score - overall quality | 1-5 | Higher | Overall quality |
| **CMOS** | Comparative MOS - preference between systems | -3 to +3 | Higher | Relative quality |
| **SMOS** | Speaker similarity MOS | 1-5 | Higher | Voice cloning quality |
| **UTMOS** | Universal Text-to-Speech MOS predictor | 1-5 | Higher | Predicted naturalness |

### XTTS-Specific Performance Notes

According to the INTERSPEECH 2024 paper:
- **CMOS**: XTTS demonstrates significantly better results than previous works (YourTTS, Mega-TTS 2)
- **SECS**: Slightly lower than highly-tuned monolingual systems due to multilingual training complexity
- **CER**: Better than baseline multilingual models
- **Streaming Latency**: <200ms first chunk latency enables real-time applications

---

## HuggingFace TTS Leaderboard & Resources

### Key TTS Models on HuggingFace

| Model | Languages | Voice Cloning | Streaming | Notes |
|-------|-----------|---------------|-----------|-------|
| **coqui/XTTS-v2** | 16 | Yes (6s clip) | Yes (<200ms) | State-of-the-art multilingual |
| StyleTTS2 | 1 (en) | Yes | Limited | High quality, English only |
| Bark | Multilingual | Yes | No | Supports non-speech sounds |
| VALL-E X | 3 | Yes | No | Requires longer reference |
| YourTTS | 6 | Yes | Limited | Predecessor to XTTS |
| Piper | 50+ | No | Yes | Fast, lower quality |

### Evaluation Datasets on HuggingFace

- **LibriTTS**: [cdminix/libritts-aligned](https://huggingface.co/datasets/cdminix/libritts-aligned)
- **VCTK**: [vctk](https://huggingface.co/datasets/vctk)
- **LJSpeech**: [lj_speech](https://huggingface.co/datasets/lj_speech)
- **Common Voice**: [mozilla-foundation/common_voice_13_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0)

### Key Metrics Tracked

- **Naturalness**: MOS, CMOS, UTMOS
- **Intelligibility**: WER, CER, STOI
- **Speaker Similarity**: SECS, SMOS, cosine similarity
- **Speed**: RTF, latency, throughput
- **Model Size**: Parameters, disk size, memory usage

---

## Additional Resources

### Official Repositories

- [Coqui TTS GitHub](https://github.com/coqui-ai/TTS)
- [XTTS-v2 HuggingFace Model](https://huggingface.co/coqui/XTTS-v2)
- [Idiap Coqui TTS Fork](https://github.com/idiap/coqui-ai-TTS) (Maintained)

### Papers & Documentation

- [XTTS Paper (arXiv:2406.04904)](https://arxiv.org/abs/2406.04904)
- [XTTS Paper (PDF)](https://arxiv.org/pdf/2406.04904)
- [Coqui TTS Documentation](https://docs.coqui.ai/)
- [XTTS Model Documentation](https://docs.coqui.ai/en/latest/models/xtts.html)

### Blog Posts & Guides

- [XTTS v2: New Version of Open-Source TTS Model (Medium)](https://medium.com/machine-learns/xtts-v2-new-version-of-the-open-source-text-to-speech-model-af73914db81f)
- [12 Best Open-Source TTS Models Compared (2025)](https://www.inferless.com/learn/comparing-different-text-to-speech---tts--models-part-2)
- [XTTS-v2 Installation Guide (Stackademic)](https://blog.stackademic.com/xtts-v2-text-to-speech-transformer-library-an-actually-working-guide-bc75bf5f8f6c)
- [XTTS V2 Analysis (Artificial Analysis)](https://artificialanalysis.ai/text-to-speech/models/xtts-v2)

### AMD ROCm Resources

- [Fine-tuning Speech Models using ROCm on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/speech_models/README.html)
- [Speech-to-Text on AMD GPU with Whisper](https://rocm.blogs.amd.com/artificial-intelligence/whisper/README.html)
- [Running Qwen TTS on AMD Strix Halo](https://tinycomputers.io/posts/qwen-tts-on-amd-strix-halo.html)
- [AMD ROCm Performance Results](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html)

### Datasets

- [LibriTTS (cdminix/libritts-aligned)](https://huggingface.co/datasets/cdminix/libritts-aligned)
- [VCTK Corpus](https://huggingface.co/datasets/vctk)
- [LJSpeech](https://huggingface.co/datasets/lj_speech)
- [HuggingFace Audio Course - TTS Datasets](https://huggingface.co/learn/audio-course/en/chapter6/tts_datasets)

---

## Quick Reference Commands

```bash
# Install Coqui TTS with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
pip install coqui-tts

# List available models
tts --list_models

# Quick synthesis with voice cloning
tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --text "Your text here" \
    --speaker_wav /path/to/reference.wav \
    --language_idx en \
    --use_cuda true \
    --out_path output.wav

# Check AMD GPU status
rocm-smi
rocm-smi --showuse --showmeminfo vram
rocm-smi --showpower

# Download LibriTTS test-clean
python -c "from datasets import load_dataset; ds = load_dataset('cdminix/libritts-aligned', 'clean', split='test')"

# Python API quick start
python -c "from TTS.api import TTS; tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2').to('cuda'); tts.tts_to_file(text='Hello world!', speaker_wav='ref.wav', language='en', file_path='out.wav')"
```

---

**Document Version:** 1.0
**Last Updated:** March 2026
**Target Hardware:** AMD MI300X, RX 7900 XTX, and other ROCm-compatible GPUs

**Key Features:**
- 16 languages supported
- Zero-shot voice cloning with 6-second clips
- Cross-lingual voice cloning capability
- <200ms streaming latency
- Real-time factor <0.3 on modern GPUs
- State-of-the-art naturalness and acoustic quality