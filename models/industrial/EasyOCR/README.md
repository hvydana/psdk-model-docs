# EasyOCR - Benchmark Guide for AMD GPU

## About the Model

EasyOCR is a ready-to-use Optical Character Recognition (OCR) toolkit that supports 80+ languages and all popular writing scripts including Latin, Chinese, Arabic, Devanagari, Cyrillic, and more. Built on PyTorch, it uses a two-stage approach combining state-of-the-art deep learning models for text detection and recognition, offering an excellent balance between simplicity, performance, and multilingual support. EasyOCR is particularly effective for structured documents like receipts, PDFs, and bills, while also handling scene text in natural images.

### Underlying Research Papers

**1. CRAFT: "Character Region Awareness for Text Detection"** (Baek et al., 2019)

CRAFT is a scene text detection method that effectively detects text areas by exploring each character and the affinity between characters. The model uses a convolutional neural network to produce character region scores and affinity scores. Extensive experiments on six benchmarks, including TotalText and CTW-1500 datasets containing highly curved texts, demonstrate that character-level text detection significantly outperforms previous state-of-the-art detectors. CRAFT guarantees high flexibility in detecting complicated scene text images, including arbitrarily-oriented, curved, or deformed texts.

**Paper:** [arXiv:1904.01941](https://arxiv.org/abs/1904.01941) | **Published:** CVPR 2019

**2. CRNN: "An End-to-End Trainable Neural Network for Image-based Sequence Recognition"** (Shi et al., 2015)

The CRNN architecture integrates feature extraction, sequence modeling, and transcription into a unified framework. It possesses four distinctive properties: (1) end-to-end trainable, (2) handles sequences of arbitrary lengths without character segmentation, (3) not confined to any predefined lexicon, and (4) generates an effective yet compact model. The architecture consists of convolutional layers for feature extraction, recurrent layers (LSTM) for sequence modeling, and a CTC decoder for transcription.

**Paper:** [arXiv:1507.05717](https://arxiv.org/abs/1507.05717) | **Published:** IEEE TPAMI 2017

---

## Standard Benchmark Datasets

EasyOCR performance is evaluated across multiple industry-standard benchmarks for scene text recognition and detection.

### 1. ICDAR 2015 (Incidental Scene Text)

**ICDAR 2015** (Robust Reading Competition) is a comprehensive benchmark for evaluating text detection and recognition in natural scenes, focusing on "incidental" scene text captured in the wild.

#### Dataset Structure
- **Training set**: 4,468 word instances in 1,000 images
- **Test set**: 2,077 word instances in 500 images
- **Tasks**: Text Localization (Task 1), Word Recognition (Task 3), End-to-end Text Spotting (Task 4)

#### Download from HuggingFace

```bash
# Install dependencies
pip install datasets transformers
```

```python
from datasets import load_dataset

# Load ICDAR 2015 dataset
dataset = load_dataset("MiXaiLL76/ICDAR2015_OCR")

# Dataset contains train (4.47k samples) and test (2.08k samples) splits
print(f"Train samples: {len(dataset['train'])}")
print(f"Test samples: {len(dataset['test'])}")

# View a sample
print(dataset['train'][0])
# Output: {'image': PIL.Image, 'text': 'recognized text', 'bbox': [...], ...}
```

### 2. Street View Text (SVT)

**SVT** is harvested from Google Street View, containing challenging street-level text with perspective distortion, motion blur, and low resolution.

#### Dataset Structure
- **Total**: 647 words, 3,796 letters in 249-350 images
- **Training**: 100 images with 211 words
- **Test**: 250 images with 514 words

#### Download

The SVT dataset is available from the IAPR TC11 website:
- **Official Source**: [IAPR TC11 SVT Dataset](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)
- **Download Size**: 118 MB (complete dataset with annotations)

```bash
# Download and extract
wget http://www.iapr-tc11.org/dataset/SVT/svt.zip
unzip svt.zip
```

### 3. IIIT 5K-Word (IIIT5K)

**IIIT5K** is from the 2012 BMVC paper "Scene Text Recognition using Higher Order Language Priors" and is used for text recognition with natural scene images at word-level granularity.

#### Download from HuggingFace

```python
from datasets import load_dataset

# Multiple versions available on HuggingFace
dataset = load_dataset("HuggingFaceM4/IIIT-5K")
# Or: load_dataset("MiXaiLL76/IIIT5K_OCR")

# View a sample
print(dataset['train'][0])
```

### 4. Additional Benchmark Datasets

EasyOCR is also evaluated on:
- **IC03** (ICDAR 2003): Structured scene text
- **IC13** (ICDAR 2013): Focused text reading
- **SVTP** (Street View Text - Perspective): Perspective-distorted text
- **CUTE80**: Curved text benchmark
- **TotalText**: Arbitrary-shaped text
- **CTW-1500**: Curved text in the wild
- **COCO-Text**: Text in context

---

## Installation & Inference

### Install EasyOCR

```bash
# Install using pip
pip install easyocr

# For GPU support, ensure PyTorch with CUDA/ROCm is installed first
# See AMD GPU Benchmarking Setup section below
```

### Basic Inference

```bash
# Command-line usage (if CLI is available)
easyocr -l en -f image.jpg --detail 1 --gpu True
```

### Python API Inference

```python
import easyocr

# Create reader (supports 80+ languages)
# Language codes: 'en' (English), 'ch_sim' (Chinese Simplified),
# 'fr' (French), 'de' (German), 'es' (Spanish), etc.
reader = easyocr.Reader(['en'], gpu=True)

# Perform OCR on single image
result = reader.readtext('image.jpg')

# Print results
for detection in result:
    bbox, text, confidence = detection
    print(f"Text: {text}, Confidence: {confidence:.2f}")
```

### Advanced Usage

```python
import easyocr
import cv2
import matplotlib.pyplot as plt

# Initialize reader with multiple languages
reader = easyocr.Reader(['en', 'fr', 'de'], gpu=True)

# Read text with detailed output
result = reader.readtext(
    'image.jpg',
    detail=1,           # 1 for bounding boxes, 0 for text only
    paragraph=False,    # True to combine text into paragraphs
    decoder='greedy',   # 'greedy' or 'beamsearch'
    beamWidth=5,        # Beam width for beamsearch decoder
    batch_size=1,       # Batch size for inference
    workers=0,          # Number of workers for image loading
    allowlist=None,     # Allow only specific characters
    blocklist=None,     # Block specific characters
    min_size=10,        # Minimum text size to detect
    contrast_ths=0.1,   # Text/background contrast threshold
    adjust_contrast=0.5 # Contrast adjustment factor
)

# Visualize results
img = cv2.imread('image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

for detection in result:
    bbox, text, confidence = detection
    # Draw bounding box
    top_left = tuple([int(val) for val in bbox[0]])
    bottom_right = tuple([int(val) for val in bbox[2]])
    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(img, text, top_left, cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 1)

plt.imshow(img)
plt.show()
```

### Expected Output

```python
[
    ([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], 'detected text', 0.95),
    ([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], 'another text', 0.87),
    # ... more detections
]
```

Each result tuple contains:
- **Bounding box**: List of 4 corner points [top-left, top-right, bottom-right, bottom-left]
- **Text**: Recognized text string
- **Confidence**: Recognition confidence score (0.0 to 1.0)

---

## Benchmark Results & Performance Metrics

### EasyOCR vs Competing OCR Engines

| OCR Engine | Architecture | Best Use Case | Speed | Accuracy | Languages |
|------------|--------------|---------------|-------|----------|-----------|
| **EasyOCR** | CRAFT + CRNN | Structured docs, Multilingual | Medium | Good | 80+ |
| **PaddleOCR** | DBNet + CRNN | Production, High accuracy | Fast | Excellent | 80+ |
| **Tesseract 5.x** | LSTM-based | Legacy systems, Open source | Medium | Good | 100+ |
| **RapidOCR** | Lightweight CNN | Edge devices, Fast inference | Very Fast | Fair | Limited |
| **GPT-4o Vision** | Transformer-based | Complex layouts, VLM tasks | Slow | Excellent | 50+ |
| **Gemini 1.5** | Transformer-based | Multimodal understanding | Slow | Excellent | 100+ |

**Source:** Comparative analysis from [IntuitionLabs OCR Technical Analysis](https://intuitionlabs.ai/articles/non-llm-ocr-technologies) and [CodeSOTA OCR Benchmarks](https://www.codesota.com/ocr)

### Performance on Standard Benchmarks

| Dataset | EasyOCR | PaddleOCR | Tesseract 4.x | GPT-4o | Notes |
|---------|---------|-----------|---------------|---------|-------|
| **ICDAR 2015** | ~75-80% | ~85-90% | ~65-70% | ~92-95% | Word-level accuracy |
| **SVT** | ~75% | ~82% | ~68% | ~90% | Street View Text |
| **IIIT5K** | ~78% | ~84% | ~70% | ~93% | Scene text recognition |
| **Structured Docs** | ~85-90% | ~92-95% | ~88-92% | ~95-98% | Receipts, PDFs, bills |

**Notes:**
- EasyOCR performs better on structured documents (receipts, PDFs) compared to complex scene text
- Vision-Language Models (GPT-4o, Gemini) achieve higher accuracy but with significantly slower inference
- Accuracy varies based on image quality, text orientation, and language

### Model Architecture & Size

| Component | Architecture | Parameters | Purpose |
|-----------|-------------|------------|---------|
| **Detection** | CRAFT (CNN-based) | ~20M | Locates text regions in images |
| **Recognition** | CRNN (CNN + LSTM + CTC) | ~15M | Recognizes characters in detected regions |
| **Total Model Size** | - | ~35M | Combined detection + recognition |
| **VRAM Usage** | - | ~2 GB | Single image inference |

### Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Model Loading Time** | 3-30 seconds | Depends on hardware; 25-30s on Colab T4, ~3s on RTX GPUs |
| **VRAM Consumption** | ~2 GB | 1.9 GB allocated, 2.5 GB cached (Colab testing) |
| **Parallel Processing** | 15-20 images | On RTX 6000 (24GB) at 1080x1919 resolution |
| **Inference Speed** | 0.5-3 seconds | Per image, varies by resolution and hardware |
| **Batch Processing Speedup** | 30-60% | With optimized CPU/GPU pipeline |

**Source:** [Medium - EasyOCR Performance Optimization](https://medium.com/@phuocnguyen90/i-accidentally-doubled-the-speed-of-easyocr-3779ec951424)

---

## AMD GPU Benchmarking Setup

### ROCm Installation for AMD GPUs

```bash
# Check ROCm compatibility and current installation
rocm-smi
rocminfo

# Install PyTorch with ROCm support (ROCm 6.2)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Verify PyTorch installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}'); print(f'ROCm Version: {torch.version.hip if hasattr(torch.version, \"hip\") else \"N/A\"}')"
```

### Install EasyOCR with ROCm

```bash
# IMPORTANT: Install PyTorch with ROCm FIRST before installing EasyOCR
# This prevents CUDA version conflicts

# After PyTorch is installed, install EasyOCR
pip install easyocr

# Verify installation
python -c "import easyocr; print('EasyOCR installed successfully')"
```

**Note:** EasyOCR does not have official ROCm support as of 2024, but it works with PyTorch ROCm backend since it's PyTorch-based. There is an [open issue requesting ROCm support](https://github.com/JaidedAI/EasyOCR/issues/1271) on the EasyOCR GitHub repository.

### Benchmark Script for AMD GPU

```python
import torch
import easyocr
import time
import numpy as np
from datasets import load_dataset
from PIL import Image
import io

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"ROCm Version: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")
    print(f"PyTorch Version: {torch.__version__}")

# Initialize EasyOCR reader
print("Loading EasyOCR model...")
start_load = time.time()
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
load_time = time.time() - start_load
print(f"Model loaded in {load_time:.2f} seconds")

# Load ICDAR 2015 test dataset
print("Loading ICDAR 2015 dataset...")
dataset = load_dataset("MiXaiLL76/ICDAR2015_OCR", split="test[:100]")

# Benchmark metrics
results = []
total_chars = 0
total_words = 0

print("\nRunning benchmark...")
for i, sample in enumerate(dataset):
    # Get image
    img = sample['image']
    ground_truth = sample.get('text', '')

    # Convert PIL image to bytes for EasyOCR
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Measure inference time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    result = reader.readtext(img_byte_arr, detail=1)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    inference_time = time.time() - start_time

    # Extract predicted text
    predicted_text = ' '.join([detection[1] for detection in result])
    avg_confidence = np.mean([detection[2] for detection in result]) if result else 0.0

    # Calculate metrics
    word_count = len(ground_truth.split())
    char_count = len(ground_truth)
    total_words += word_count
    total_chars += char_count

    results.append({
        'sample_id': i,
        'inference_time': inference_time,
        'word_count': word_count,
        'char_count': char_count,
        'predicted': predicted_text,
        'reference': ground_truth,
        'confidence': avg_confidence,
        'num_detections': len(result)
    })

    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(dataset)} images...")

# Calculate summary statistics
avg_inference_time = np.mean([r['inference_time'] for r in results])
median_inference_time = np.median([r['inference_time'] for r in results])
std_inference_time = np.std([r['inference_time'] for r in results])
total_time = sum([r['inference_time'] for r in results])
throughput = len(results) / total_time  # images per second
avg_confidence = np.mean([r['confidence'] for r in results])

# GPU metrics
if torch.cuda.is_available():
    max_memory_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
    max_memory_reserved = torch.cuda.max_memory_reserved() / 1024**3  # GB

# Print summary
print("\n" + "="*70)
print("BENCHMARK SUMMARY")
print("="*70)
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Total Images: {len(results)}")
print(f"Total Time: {total_time:.2f} seconds")
print(f"Model Load Time: {load_time:.2f} seconds")
print(f"\nInference Metrics:")
print(f"  Average Inference Time: {avg_inference_time:.3f} seconds/image")
print(f"  Median Inference Time: {median_inference_time:.3f} seconds/image")
print(f"  Std Dev: {std_inference_time:.3f} seconds")
print(f"  Throughput: {throughput:.2f} images/second")
print(f"  Average Confidence: {avg_confidence:.2%}")
print(f"\nMemory Usage:")
if torch.cuda.is_available():
    print(f"  Max Memory Allocated: {max_memory_allocated:.2f} GB")
    print(f"  Max Memory Reserved: {max_memory_reserved:.2f} GB")
else:
    print(f"  Running on CPU (no GPU metrics)")
print("="*70)
```

### Performance Metrics Table Template

| Metric | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Notes |
|--------|------------------|-----------|------------|-----------------|-------|
| **GPU Model** | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Compare datacenter vs consumer GPUs |
| **Memory (GB)** | 80 | 16 | 192 | 24 | VRAM capacity |
| **TDP (W)** | 400 | 70 | 750 | 355 | Thermal design power |
| **Model Load Time (s)** | ~3 | ~25-30 | _[Your result]_ | _[Your result]_ | One-time overhead |
| **Avg Inference Time (s/image)** | ~0.5 | ~1.5-2.0 | _[Your result]_ | _[Your result]_ | Per image, 1080p |
| **Throughput (images/s)** | ~2.0 | ~0.5-0.7 | _[Your result]_ | _[Your result]_ | Single image pipeline |
| **Batch Throughput (images/s)** | ~5-8 | ~1.5-2.5 | _[Your result]_ | _[Your result]_ | Optimized batch processing |
| **Peak Memory Usage (GB)** | ~2.5 | ~2.5 | _[Your result]_ | _[Your result]_ | Monitor with rocm-smi |
| **Average Power Draw (W)** | ~150 | ~50 | _[Your result]_ | _[Your result]_ | During inference |
| **Energy per 100 Images (Wh)** | ~2.1 | ~4.2 | _[Your result]_ | _[Your result]_ | Lower is better |

### AMD-Specific Metrics to Track

```python
import subprocess
import torch

def get_rocm_smi_stats():
    """Get AMD GPU statistics using rocm-smi"""
    try:
        # GPU utilization and memory
        result = subprocess.run(
            ['rocm-smi', '--showuse', '--showmeminfo', 'vram'],
            capture_output=True,
            text=True,
            timeout=5
        )
        print("ROCm SMI Output:")
        print(result.stdout)

        # Power consumption
        power_result = subprocess.run(
            ['rocm-smi', '--showpower'],
            capture_output=True,
            text=True,
            timeout=5
        )
        print("\nPower Consumption:")
        print(power_result.stdout)

        return result.stdout
    except Exception as e:
        print(f"Error getting ROCm stats: {e}")
        return None

# Memory tracking
if torch.cuda.is_available():
    print(f"\nPyTorch Memory Stats:")
    print(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    print(f"  Max Allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    # ROCm/GPU info
    print(f"\nGPU Information:")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Compute Capability: {torch.cuda.get_device_capability(0)}")
    if hasattr(torch.version, 'hip'):
        print(f"  ROCm Version: {torch.version.hip}")

    # Get ROCm SMI stats
    print("\n" + "="*70)
    get_rocm_smi_stats()
```

### Complete Runtime Metrics Table

| Runtime Metric | Formula | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Notes |
|----------------|---------|------------------|-----------|------------|-----------------|-------|
| **Inference Latency (ms)** | Time per single image | ~500 | ~1500-2000 | _[Your result]_ | _[Your result]_ | Lower is better |
| **Throughput (imgs/s)** | 1 / avg_inference_time | ~2.0 | ~0.5-0.7 | _[Your result]_ | _[Your result]_ | Single image mode |
| **Batch Throughput (imgs/s)** | batch_size / batch_time | ~5-8 | ~1.5-2.5 | _[Your result]_ | _[Your result]_ | Optimized batching |
| **Chars Per Second** | total_chars / total_time | _[Reference]_ | _[Reference]_ | _[Your result]_ | _[Your result]_ | Recognition speed |
| **GPU Utilization (%)** | From nvidia-smi / rocm-smi | ~60-80% | ~50-70% | _[Your result]_ | _[Your result]_ | Average during inference |
| **Memory Bandwidth (GB/s)** | From specs/monitoring | ~2.0 TB/s | ~320 GB/s | ~5.3 TB/s | ~960 GB/s | Theoretical max |
| **Memory Efficiency (%)** | Actual / Theoretical BW | _[Reference]_ | _[Reference]_ | _[Your result]_ | _[Your result]_ | How well bandwidth is utilized |
| **Average Confidence** | Mean of all detections | ~85-90% | ~85-90% | _[Your result]_ | _[Your result]_ | Model confidence scores |
| **Energy Efficiency (imgs/Wh)** | images / (power × time) | ~13.3 | ~4.2 | _[Your result]_ | _[Your result]_ | Higher is better |

---

## OCR Benchmarking Leaderboards & Resources

### CodeSOTA OCR Leaderboard

The [CodeSOTA OCR Benchmark](https://www.codesota.com/ocr) provides independent benchmarks tracking state-of-the-art OCR models across multiple datasets including ICDAR, SROIE, and custom evaluation sets.

### Key Metrics Tracked
- **Accuracy** (Character and Word level)
- **WER** (Word Error Rate) - lower is better
- **CER** (Character Error Rate) - lower is better
- **Inference Speed** (images/second, latency)
- **Model Size** (parameters, disk space)
- **Language Support** (number of supported languages)

### OCRBench v2

[OCRBench v2](https://arxiv.org/html/2501.00321v2) is an improved benchmark for evaluating large multimodal models on visual text localization and reasoning, providing comprehensive evaluation across diverse OCR tasks.

### Standard Evaluation Datasets Summary

| Dataset | Type | Size | Use Case |
|---------|------|------|----------|
| **ICDAR 2015** | Scene Text | 4.5k train, 2k test | Incidental scene text, arbitrary orientation |
| **SVT** | Street View | 647 words | Perspective distortion, low resolution |
| **IIIT5K** | Scene Text | 5,000 words | Word-level recognition benchmark |
| **IC03** | Scene Text | 509 images | Structured scene text |
| **IC13** | Scene Text | 462 images | Focused reading |
| **SVTP** | Street View | 645 images | Perspective-distorted text |
| **CUTE80** | Scene Text | 80 images | Curved text |
| **TotalText** | Scene Text | 1,555 images | Arbitrary-shaped text |
| **CTW-1500** | Scene Text | 1,500 images | Curved text in the wild |
| **SROIE** | Receipt/Document | 1,000 receipts | Structured document OCR |

---

## Additional Resources

### Official Repositories
- [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR) - Official repository by JaidedAI
- [CRAFT-pytorch GitHub](https://github.com/clovaai/CRAFT-pytorch) - Official CRAFT implementation
- [CRNN Original Implementation](https://github.com/bgshih/crnn) - Original CRNN code

### Papers & Documentation
- [CRAFT Paper (arXiv:1904.01941)](https://arxiv.org/abs/1904.01941) - Character Region Awareness for Text Detection
- [CRNN Paper (arXiv:1507.05717)](https://arxiv.org/abs/1507.05717) - End-to-End Trainable Neural Network for Sequence Recognition
- [ICDAR 2015 Competition Paper (arXiv:1506.03184)](https://arxiv.org/abs/1506.03184) - Text Reading in the Wild
- [EasyOCR Official Documentation](https://www.jaided.ai/easyocr/) - Installation and usage guides

### Benchmarks & Comparisons
- [CodeSOTA OCR Benchmarks](https://www.codesota.com/ocr) - Independent ML benchmarks
- [IntuitionLabs OCR Technical Analysis](https://intuitionlabs.ai/articles/non-llm-ocr-technologies) - Modern OCR engines comparison
- [E2E Networks OCR Models Guide](https://www.e2enetworks.com/blog/complete-guide-open-source-ocr-models-2025) - Open-source OCR comparison 2025
- [OCRBench v2 (arXiv:2501.00321)](https://arxiv.org/html/2501.00321v2) - VLM evaluation on OCR tasks
- [GitHub: OCR Benchmark 2025](https://github.com/agentic-ai-forge/ocr-benchmark-2025) - PaddleOCR vs EasyOCR comparison

### AMD ROCm Resources
- [AMD ROCm Documentation](https://rocm.docs.amd.com/) - Official ROCm documentation
- [PyTorch for AMD ROCm](https://pytorch.org/blog/pytorch-for-amd-rocm-platform-now-available-as-python-package/) - PyTorch ROCm installation guide
- [Install PyTorch for ROCm on Radeon](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-pytorch.html) - Radeon GPU setup
- [EasyOCR ROCm Support Issue](https://github.com/JaidedAI/EasyOCR/issues/1271) - Community discussion on AMD GPU support

### Datasets
- [ICDAR 2015 on HuggingFace](https://huggingface.co/datasets/MiXaiLL76/ICDAR2015_OCR) - 6.68k total samples
- [IIIT5K on HuggingFace](https://huggingface.co/datasets/HuggingFaceM4/IIIT-5K) - IIIT 5K-Word dataset
- [SVT Dataset (IAPR TC11)](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset) - Street View Text
- [Robust Reading Competition](https://rrc.cvc.uab.es/) - ICDAR competitions and datasets
- [OCR Datasets Collection](https://github.com/xinke-wang/OCRDatasets) - Comprehensive OCR dataset list

### Optimization Guides
- [Medium: EasyOCR Speed Optimization](https://medium.com/@phuocnguyen90/i-accidentally-doubled-the-speed-of-easyocr-3779ec951424) - Performance tuning techniques
- [FreeCodeCamp: Fine-Tune EasyOCR](https://www.freecodecamp.org/news/how-to-fine-tune-easyocr-with-a-synthetic-dataset/) - Custom dataset training
- [Qualcomm AI Hub: EasyOCR](https://aihub.qualcomm.com/compute/models/easyocr) - Mobile deployment optimization

---

## Quick Reference Commands

```bash
# Install EasyOCR (after PyTorch with ROCm)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
pip install easyocr

# Basic Python usage
python -c "import easyocr; reader = easyocr.Reader(['en']); print(reader.readtext('image.jpg', detail=0))"

# Check AMD GPU status
rocm-smi
rocm-smi --showuse --showmeminfo vram
rocm-smi --showpower

# Download ICDAR 2015 dataset
python -c "from datasets import load_dataset; ds = load_dataset('MiXaiLL76/ICDAR2015_OCR')"

# Monitor GPU during inference
watch -n 1 rocm-smi

# Memory profiling
python -c "import torch; print(f'VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB')"
```

---

## Troubleshooting AMD GPU Issues

### CUDA/ROCm Conflict
If you encounter "CUDA not available" errors even with ROCm installed:

```bash
# Ensure PyTorch with ROCm is installed BEFORE EasyOCR
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
pip install easyocr --no-deps  # Install without dependencies
pip install pillow opencv-python-headless scipy scikit-image python-bidi pyyaml ninja
```

### Out of Memory Errors
If you encounter OOM errors:

```python
# Reduce batch size or process sequentially
reader = easyocr.Reader(['en'], gpu=True)

# Process images one at a time
for img_path in image_list:
    result = reader.readtext(img_path)
    # Clear cache periodically
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### ROCm Compatibility Check
```bash
# Verify ROCm installation
rocminfo | grep "Name:"
/opt/rocm/bin/hipconfig --version

# Test PyTorch ROCm
python -c "import torch; print(f'ROCm: {torch.version.hip}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

---

**Document Version:** 1.0
**Last Updated:** March 2026
**Target Hardware:** AMD MI300X, RX 7900 XTX, and other ROCm-compatible GPUs
**EasyOCR Version:** 1.7+
**ROCm Version:** 6.2+
