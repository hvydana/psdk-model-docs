# EfficientNetV2 - Benchmark Guide for AMD GPU

**Navigation:** [🏠 Home]({{ site.baseurl }}/) | [📑 Models Index]({{ site.baseurl }}/MODELS_INDEX) | [📝 Contributing]({{ site.baseurl }}/CONTRIBUTING)

---

## About the Model

EfficientNetV2 is a family of convolutional neural networks that achieve state-of-the-art accuracy while being significantly faster to train and smaller in parameter size compared to previous models. Built upon the success of EfficientNet, EfficientNetV2 introduces training-aware neural architecture search (NAS) and scaling to jointly optimize model accuracy, training speed, and parameter efficiency. The architecture uses a combination of Fused-MBConv layers and progressive learning techniques to achieve superior performance.

### Original EfficientNetV2 Paper

**"EfficientNetV2: Smaller Models and Faster Training"** (Tan & Le, 2021)

EfficientNetV2 trains 5x-11x faster than state-of-the-art models while using up to 6.8x fewer parameters. By pretraining on ImageNet21k, EfficientNetV2 achieves 87.3% top-1 accuracy on ImageNet ILSVRC2012, outperforming the Vision Transformer (ViT-L/16) by 2.0% accuracy while training 5x-11x faster using the same computing resources. The key innovations include: (1) training-aware NAS that optimizes for both accuracy and training efficiency, (2) enriched search space with new operations like Fused-MBConv, and (3) progressive learning that adaptively adjusts regularization strength based on image size.

**Paper:** [arXiv:2104.00298](https://arxiv.org/abs/2104.00298) | **Published:** ICML 2021 | **Authors:** Google Research, Brain Team

---

## Standard Benchmark Dataset: ImageNet ILSVRC2012

**ImageNet ILSVRC2012** is the industry-standard benchmark for evaluating image classification models. It contains 1,000 object classes spanning diverse categories from animals to vehicles to everyday objects.

### Dataset Structure
- **Training set**: 1,281,167 images across 1,000 classes
- **Validation set**: 50,000 images (50 per class)
- **Test set**: 100,000 images
- **Total size**: 155.84 GiB

### Download from HuggingFace

```bash
# Install dependencies
pip install datasets transformers pillow
```

```python
from datasets import load_dataset

# Load ImageNet-1k validation set
dataset = load_dataset("imagenet-1k", split="validation")

# Alternative: Load from mlx-vision
dataset = load_dataset("mlx-vision/imagenet-1k", split="validation")

# View a sample
print(dataset[0])
# Output: {'image': <PIL.Image>, 'label': 65}

# Access image and label
from PIL import Image
image = dataset[0]['image']  # PIL Image object
label = dataset[0]['label']   # Integer label (0-999)
```

**Note:** You must register at [https://image-net.org/download-images](https://image-net.org/download-images) to get official download access for the full dataset (ILSVRC2012_img_train.tar, ILSVRC2012_img_val.tar).

---

## Installation & Inference

### Install timm (PyTorch Image Models)

```bash
# Install timm library
pip install timm torch torchvision pillow

# For AMD GPU support with ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
pip install timm
```

### Basic Inference

```bash
# Using timm CLI (if available)
python -m timm.models.efficientnet --model tf_efficientnetv2_s.in21k --pretrained
```

### Python API Inference

```python
import torch
import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# Load model
model = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True)
model.eval()

# Get model-specific transforms
config = resolve_data_config({}, model=model)
transform = create_transform(**config)

# Load and preprocess image
img = Image.open('image.jpg').convert('RGB')
tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Inference
with torch.no_grad():
    output = model(tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Get top-5 predictions
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(5):
    print(f"Class {top5_catid[i].item()}: {top5_prob[i].item():.4f}")
```

### Advanced Inference with GPU

```python
from urllib.request import urlopen
from PIL import Image
import timm
import torch

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load image from URL
img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

# Create model and move to GPU
model = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True)
model = model.eval().to(device)

# Get model-specific transforms
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

# Inference
output = model(transforms(img).unsqueeze(0).to(device))
top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)

print("Top-5 Predictions:")
for i in range(5):
    print(f"  Class {top5_class_indices[0][i].item()}: {top5_probabilities[0][i].item():.2f}%")
```

### Available Model Variants

```python
# EfficientNetV2 Small (ImageNet-21k pretrained)
model = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True)

# EfficientNetV2 Medium (ImageNet-21k pretrained)
model = timm.create_model('tf_efficientnetv2_m.in21k', pretrained=True)

# EfficientNetV2 Large (ImageNet-21k pretrained)
model = timm.create_model('tf_efficientnetv2_l.in21k', pretrained=True)

# EfficientNetV2 XL (ImageNet-21k pretrained)
model = timm.create_model('tf_efficientnetv2_xl.in21k', pretrained=True)

# EfficientNetV2 B0-B3 variants
model = timm.create_model('tf_efficientnetv2_b0.in1k', pretrained=True)
model = timm.create_model('tf_efficientnetv2_b1.in1k', pretrained=True)
model = timm.create_model('tf_efficientnetv2_b2.in1k', pretrained=True)
model = timm.create_model('tf_efficientnetv2_b3.in21k', pretrained=True)

# Ross Wightman's custom variants
model = timm.create_model('efficientnetv2_rw_s.ra2_in1k', pretrained=True)
model = timm.create_model('efficientnetv2_rw_m.agc_in1k', pretrained=True)
```

### Expected Output

```python
# Classification output
{
  "class_id": 281,
  "class_name": "tabby_cat",
  "probability": 0.8234,
  "top5_predictions": [
    {"class_id": 281, "name": "tabby_cat", "prob": 0.8234},
    {"class_id": 282, "name": "tiger_cat", "prob": 0.1245},
    {"class_id": 285, "name": "Egyptian_cat", "prob": 0.0421},
    {"class_id": 287, "name": "lynx", "prob": 0.0034},
    {"class_id": 283, "name": "Persian_cat", "prob": 0.0029}
  ]
}
```

---

## Benchmark Results & Performance Metrics

### EfficientNetV2 Performance on ImageNet

| Model | ImageNet-1k Top-1 | ImageNet-21k Top-1 | Params (M) | FLOPs (B) | Training Speed |
|-------|------------------|-------------------|------------|-----------|----------------|
| **EfficientNetV2-S** | 84.9% | 86.2% | 24.3 | 8.4 | 1x |
| **EfficientNetV2-M** | 85.2% | 86.7% | 54.1 | 24.7 | 1x |
| **EfficientNetV2-L** | 85.7% | **87.3%** | 119.5 | 56.3 | 1x |
| **EfficientNetV2-XL** | - | **87.3%** | 208.1 | 93.7 | 1x |
| EfficientNet-B7 | 84.4% | - | 66.3 | 37.0 | **0.09x** (11x slower) |
| ViT-L/16 (21k) | - | 85.3% | 307 | 190.7 | **0.2x** (5x slower) |
| NFNet-F3 | 85.7% | - | 254 | 115.9 | **0.33x** (3x slower) |

**Key Observations:**
- EfficientNetV2-L achieves 87.3% top-1 accuracy on ImageNet with ImageNet-21k pretraining
- 2.0% higher accuracy than ViT-L/16 while training 5x faster
- Up to 6.8x fewer parameters than comparable models
- EfficientNetV2-M matches EfficientNet-B7 accuracy while training 11x faster

### Performance: EfficientNetV2 vs Alternatives

| Architecture | ImageNet Top-1 | Params (M) | Inference Speed (V100) | Training Speed | Notes |
|--------------|----------------|------------|----------------------|----------------|-------|
| **EfficientNetV2-L** | **87.3%** | 119.5 | Fast | **5-11x faster** | Progressive learning, Fused-MBConv |
| EfficientNetV1-B7 | 84.4% | 66.3 | Medium | 1x baseline | Compound scaling |
| ResNet-152 | 78.3% | 60.2 | Medium | Slower | Traditional CNN |
| ViT-L/16 | 85.3% | 307 | Fast | 1x baseline | Pure transformer |
| NFNet-F3 | 85.7% | 254 | Fast | 3x faster | Normalizer-Free |
| ConvNeXt-L | 86.8% | 197.8 | Fast | Medium | Modernized CNN |

**Inference Throughput** (V100 GPU, FP16, batch size 128):
- EfficientNetV2-S: ~2,100 images/sec
- EfficientNetV2-M: ~1,300 images/sec
- EfficientNetV2-L: ~700 images/sec
- **Up to 3x faster inference** compared to EfficientNet-B7

---

## AMD GPU Benchmarking Setup

### ROCm Installation for AMD GPUs

```bash
# Check ROCm compatibility
rocm-smi

# Verify GPU information
rocminfo | grep "Marketing Name"

# Install PyTorch with ROCm support (ROCm 6.2)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Install timm library
pip install timm pillow

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### Benchmark Script for AMD GPU

```python
import torch
import timm
import time
import numpy as np
from PIL import Image
from datasets import load_dataset
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner

print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"ROCm Version: {torch.version.hip}")

# Load model
model_name = 'tf_efficientnetv2_s.in21k'
model = timm.create_model(model_name, pretrained=True)
model = model.eval().to(device)

# Get transforms
config = resolve_data_config({}, model=model)
transform = create_transform(**config)

# Load ImageNet validation set (subset for quick testing)
print("Loading ImageNet validation set...")
dataset = load_dataset("imagenet-1k", split="validation[:100]", trust_remote_code=True)

# Benchmark
print(f"\nBenchmarking {model_name}...")
results = []
total_correct = 0

# Warmup
print("Warming up...")
for i in range(10):
    sample = dataset[i]
    img_tensor = transform(sample['image']).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model(img_tensor)

# Actual benchmark
print("Running benchmark...")
for i, sample in enumerate(dataset):
    img = sample['image']
    true_label = sample['label']

    # Prepare input
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Measure inference time
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        output = model(img_tensor)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to ms

    # Get prediction
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class].item()

    # Check accuracy
    is_correct = (predicted_class == true_label)
    total_correct += int(is_correct)

    results.append({
        "sample_id": i,
        "inference_time_ms": inference_time,
        "predicted_class": predicted_class,
        "true_label": true_label,
        "confidence": confidence,
        "correct": is_correct
    })

    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(dataset)} images")

# Summary statistics
inference_times = [r["inference_time_ms"] for r in results]
avg_time = np.mean(inference_times)
std_time = np.std(inference_times)
min_time = np.min(inference_times)
max_time = np.max(inference_times)
throughput = 1000 / avg_time  # images per second
accuracy = (total_correct / len(dataset)) * 100

print("\n" + "="*60)
print("BENCHMARK RESULTS")
print("="*60)
print(f"Model: {model_name}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Number of samples: {len(dataset)}")
print(f"\nInference Time Statistics:")
print(f"  Average: {avg_time:.2f} ms")
print(f"  Std Dev: {std_time:.2f} ms")
print(f"  Min: {min_time:.2f} ms")
print(f"  Max: {max_time:.2f} ms")
print(f"\nThroughput: {throughput:.2f} images/second")
print(f"Top-1 Accuracy: {accuracy:.2f}%")

if torch.cuda.is_available():
    print(f"\nGPU Memory:")
    print(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    print(f"  Max Allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
```

### Batch Inference Benchmark

```python
import torch
import timm
import time
import numpy as np
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'tf_efficientnetv2_s.in21k'
batch_sizes = [1, 4, 8, 16, 32, 64, 128]

# Load model
model = timm.create_model(model_name, pretrained=True)
model = model.eval().to(device)

# Get transforms
config = resolve_data_config({}, model=model)
transform = create_transform(**config)

# Create dummy input
input_size = config['input_size']
dummy_img = Image.new('RGB', (input_size[1], input_size[2]), color='red')
dummy_tensor = transform(dummy_img)

print(f"Benchmarking {model_name} with different batch sizes...")
print(f"Input size: {input_size}")

for batch_size in batch_sizes:
    # Create batch
    batch = dummy_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(batch)

    # Benchmark
    times = []
    for _ in range(50):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.time()
        with torch.no_grad():
            _ = model(batch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        times.append(time.time() - start)

    avg_time = np.mean(times) * 1000  # Convert to ms
    throughput = batch_size / (avg_time / 1000)  # images/sec

    print(f"Batch size {batch_size:3d}: {avg_time:6.2f} ms/batch, {throughput:7.1f} images/sec")

    if torch.cuda.is_available():
        mem_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"                  Peak memory: {mem_allocated:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
```

### Performance Metrics Table Template

| Metric | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Notes |
|--------|------------------|-----------|------------|-----------------|-------|
| **GPU Model** | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Compare datacenter vs consumer GPUs |
| **Memory (GB)** | 80 | 16 | 192 | 24 | VRAM capacity |
| **TDP (W)** | 400 | 70 | 750 | 355 | Thermal design power |
| **Inference Time (ms)** | ~2.0 | ~5.0 | _[Your result]_ | _[Your result]_ | Single image, FP16 |
| **Throughput (images/sec)** | ~500 | ~200 | _[Your result]_ | _[Your result]_ | Batch size 1, higher is better |
| **Throughput (BS=32)** | ~2000 | ~800 | _[Your result]_ | _[Your result]_ | Batch size 32, FP16 |
| **Peak Memory Usage (GB)** | ~3.5 | ~3.5 | _[Your result]_ | _[Your result]_ | EfficientNetV2-S, batch size 32 |
| **Average Power Draw (W)** | ~250 | ~60 | _[Your result]_ | _[Your result]_ | Monitor with rocm-smi --showpower |
| **Energy per 1000 Images (Wh)** | ~0.14 | ~0.08 | _[Your result]_ | _[Your result]_ | Lower is better |
| **Top-1 Accuracy (%)** | 86.2% | 86.2% | _[Your result]_ | _[Your result]_ | ImageNet-21k pretrained |

### AMD-Specific Metrics to Track

```python
# GPU utilization tracking
import subprocess

def get_rocm_smi_stats():
    """Get AMD GPU statistics using rocm-smi"""
    try:
        # GPU utilization
        result = subprocess.run(['rocm-smi', '--showuse'],
                              capture_output=True, text=True)
        print("GPU Utilization:")
        print(result.stdout)

        # Memory info
        result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'],
                              capture_output=True, text=True)
        print("\nMemory Info:")
        print(result.stdout)

        # Power consumption
        result = subprocess.run(['rocm-smi', '--showpower'],
                              capture_output=True, text=True)
        print("\nPower Consumption:")
        print(result.stdout)
    except FileNotFoundError:
        print("rocm-smi not found. Make sure ROCm is installed.")

# Memory tracking
print("PyTorch Memory Statistics:")
print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
print(f"Max Allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

# ROCm info
if torch.cuda.is_available():
    print(f"\nROCm Version: {torch.version.hip}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Device Count: {torch.cuda.device_count()}")
```

### Complete Runtime Metrics Table

| Runtime Metric | Formula | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Notes |
|----------------|---------|------------------|-----------|------------|-----------------|-------|
| **Latency (ms)** | Single image inference time | ~2.0 | ~5.0 | _[Your result]_ | _[Your result]_ | Lower is better, FP16 |
| **Throughput (imgs/sec)** | 1000 / latency_ms | ~500 | ~200 | _[Your result]_ | _[Your result]_ | Batch size 1 |
| **Batch Throughput (BS=32)** | batch_size / (batch_time_ms / 1000) | ~2000 | ~800 | _[Your result]_ | _[Your result]_ | Optimal batch size |
| **GPU Utilization (%)** | From nvidia-smi / rocm-smi | ~95% | ~90% | _[Your result]_ | _[Your result]_ | Average during inference |
| **Memory Bandwidth (GB/s)** | From nvidia-smi / rocm-smi | ~2.0 TB/s | ~320 GB/s | _[Your result]_ | _[Your result]_ | MI300X: ~5.3 TB/s, RX 7900 XTX: ~960 GB/s |
| **TFLOPS Utilized** | Calculated from operations | ~150 | ~65 | _[Your result]_ | _[Your result]_ | FP16 compute throughput |
| **Energy Efficiency (Wh/1k imgs)** | (power_W × time_s) / 3600 | ~0.14 | ~0.08 | _[Your result]_ | _[Your result]_ | Lower is better |
| **Memory Footprint (GB)** | Max memory allocated | ~3.5 | ~3.5 | _[Your result]_ | _[Your result]_ | EfficientNetV2-S, BS=32 |

---

## Model Variants Comparison

### EfficientNetV2 Family

| Model | Params (M) | FLOPs (B) | ImageNet-1k Top-1 | ImageNet-21k Top-1 | Input Size | Use Case |
|-------|-----------|-----------|------------------|-------------------|-----------|----------|
| **EfficientNetV2-B0** | 7.1 | 0.72 | 78.7% | - | 224×224 | Edge devices, mobile |
| **EfficientNetV2-B1** | 8.1 | 1.2 | 80.1% | - | 240×240 | Mobile, embedded |
| **EfficientNetV2-B2** | 10.1 | 1.7 | 81.6% | - | 260×260 | Mobile, embedded |
| **EfficientNetV2-B3** | 14.4 | 3.0 | 83.1% | 85.7% | 300×300 | General purpose |
| **EfficientNetV2-S** | 24.3 | 8.4 | 84.9% | 86.2% | 384×384 | Balanced efficiency |
| **EfficientNetV2-M** | 54.1 | 24.7 | 85.2% | 86.7% | 480×480 | High accuracy |
| **EfficientNetV2-L** | 119.5 | 56.3 | 85.7% | **87.3%** | 480×480 | State-of-the-art |
| **EfficientNetV2-XL** | 208.1 | 93.7 | - | **87.3%** | 512×512 | Maximum accuracy |

### timm Model Names

```python
# Official TensorFlow weights ported by Ross Wightman
'tf_efficientnetv2_b0.in1k'        # EfficientNetV2-B0, ImageNet-1k
'tf_efficientnetv2_b1.in1k'        # EfficientNetV2-B1, ImageNet-1k
'tf_efficientnetv2_b2.in1k'        # EfficientNetV2-B2, ImageNet-1k
'tf_efficientnetv2_b3.in21k'       # EfficientNetV2-B3, ImageNet-21k
'tf_efficientnetv2_s.in21k'        # EfficientNetV2-S, ImageNet-21k
'tf_efficientnetv2_m.in21k'        # EfficientNetV2-M, ImageNet-21k
'tf_efficientnetv2_l.in21k'        # EfficientNetV2-L, ImageNet-21k
'tf_efficientnetv2_xl.in21k'       # EfficientNetV2-XL, ImageNet-21k

# Ross Wightman's custom training
'efficientnetv2_rw_s.ra2_in1k'     # Custom S variant
'efficientnetv2_rw_m.agc_in1k'     # Custom M variant with Adaptive Gradient Clipping
'efficientnetv2_rw_t.ra2_in1k'     # Custom Tiny variant
```

---

## Use Cases

### Industrial Applications

#### Quality Control and Defect Detection
```python
# Fine-tune EfficientNetV2 for defect classification
import torch
import torch.nn as nn
import timm

# Load pretrained model
model = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True, num_classes=5)
# 5 classes: [No defect, Scratch, Crack, Dent, Discoloration]

# Replace classifier head for your specific task
num_classes = 5
model.reset_classifier(num_classes)

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Example inference on manufacturing parts
img = Image.open('part_image.jpg')
transform = create_transform(**resolve_data_config({}, model=model))
img_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img_tensor)
    defect_probabilities = torch.nn.functional.softmax(output[0], dim=0)

defect_class = torch.argmax(defect_probabilities).item()
confidence = defect_probabilities[defect_class].item()

print(f"Defect Type: {defect_class}, Confidence: {confidence:.2%}")
```

#### Product Categorization
- Automated inventory classification
- SKU recognition in warehouses
- Product quality grading
- Assembly line monitoring

### Healthcare Applications

#### Medical Image Classification
```python
# Medical imaging example (X-rays, CT scans, etc.)
model = timm.create_model('tf_efficientnetv2_m.in21k', pretrained=True, num_classes=3)
# 3 classes: [Normal, Abnormal, Requires_Expert_Review]

model.reset_classifier(3)
model = model.to(device)

# Process medical image
medical_img = Image.open('xray_image.jpg').convert('RGB')
img_tensor = transform(medical_img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img_tensor)
    diagnosis_probs = torch.nn.functional.softmax(output[0], dim=0)

print(f"Classification: {['Normal', 'Abnormal', 'Requires Review'][torch.argmax(diagnosis_probs).item()]}")
```

**Medical Imaging Use Cases:**
- Chest X-ray abnormality detection
- Skin lesion classification
- Retinal disease screening
- Histopathology slide analysis
- CT/MRI scan classification

---

## Transfer Learning Guide

### Fine-tuning for Custom Dataset

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
from timm.data import create_transform, resolve_data_config

# Configuration
num_classes = 10  # Your custom number of classes
batch_size = 32
learning_rate = 1e-4
num_epochs = 50

# Load pretrained model
model = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True, num_classes=num_classes)
model = model.to(device)

# Data transforms
train_config = resolve_data_config({}, model=model)
train_transform = create_transform(**train_config, is_training=True)
val_transform = create_transform(**train_config, is_training=False)

# Setup optimizer and loss
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    val_accuracy = 100. * correct / len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"  Val Loss: {val_loss/len(val_loader):.4f}")
    print(f"  Val Accuracy: {val_accuracy:.2f}%")

# Save model
torch.save(model.state_dict(), 'efficientnetv2_custom.pth')
```

### Feature Extraction

```python
# Use EfficientNetV2 as feature extractor (frozen backbone)
model = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True, num_classes=0)
model = model.eval().to(device)

# Extract features
with torch.no_grad():
    features = model(img_tensor)  # Shape: [batch_size, feature_dim]

print(f"Feature vector shape: {features.shape}")
# EfficientNetV2-S: [batch_size, 1280]
# Use features for similarity search, clustering, etc.
```

---

## Additional Resources

### Official Repositories
- [EfficientNetV2 Official (TensorFlow)](https://github.com/google/automl/tree/master/efficientnetv2)
- [timm - PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)
- [Ross Wightman's timm](https://github.com/rwightman/timm)

### Papers & Documentation
- [EfficientNetV2 Paper (arXiv:2104.00298)](https://arxiv.org/abs/2104.00298)
- [EfficientNetV2 Paper (PDF)](https://arxiv.org/pdf/2104.00298)
- [Google AI Blog: Toward Fast and Accurate Neural Networks](https://research.google/blog/toward-fast-and-accurate-neural-networks-for-image-recognition/)
- [timm Documentation](https://huggingface.co/docs/timm/index)
- [Original EfficientNet Paper (arXiv:1905.11946)](https://arxiv.org/abs/1905.11946)

### Model Hub
- [timm EfficientNetV2 Models on HuggingFace](https://huggingface.co/models?library=timm&search=efficientnetv2)
- [tf_efficientnetv2_s.in21k](https://huggingface.co/timm/tf_efficientnetv2_s.in21k)
- [tf_efficientnetv2_m.in21k](https://huggingface.co/timm/tf_efficientnetv2_m.in21k)
- [tf_efficientnetv2_l.in21k](https://huggingface.co/timm/tf_efficientnetv2_l.in21k)
- [efficientnetv2_rw_m.agc_in1k](https://huggingface.co/timm/efficientnetv2_rw_m.agc_in1k)

### Blog Posts & Tutorials
- [Google Releases EfficientNetV2 - TowardsDataScience](https://towardsdatascience.com/google-releases-efficientnetv2-a-smaller-faster-and-better-efficientnet-673a77bdd43c/)
- [Review: EfficientNetV2 - Medium](https://medium.com/aiguys/review-efficientnetv2-smaller-models-and-faster-training-47d4215dcdfb)
- [Papers Explained: EfficientNetV2](https://ritvik19.medium.com/papers-explained-efficientnetv2-a7a1e4113b89)
- [AMD ROCm Performance Results](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html)

### Datasets
- [ImageNet (Official)](https://image-net.org/)
- [ImageNet-1k on HuggingFace](https://huggingface.co/datasets/imagenet-1k)
- [ImageNet ILSVRC2012](https://image-net.org/challenges/LSVRC/2012/)
- [mlx-vision/imagenet-1k](https://huggingface.co/datasets/mlx-vision/imagenet-1k)

---

## Quick Reference Commands

```bash
# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
pip install timm pillow datasets

# Check AMD GPU status
rocm-smi
rocm-smi --showuse --showmeminfo vram
rocm-smi --showpower

# Verify PyTorch + ROCm
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# List available EfficientNetV2 models in timm
python -c "import timm; models = timm.list_models('*efficientnetv2*'); print('\n'.join(models))"

# Quick inference test
python -c "
import timm
import torch
model = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True)
print(f'Model loaded: {model.default_cfg[\"architecture\"]}')
print(f'Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')
"

# Download ImageNet validation set (requires authentication)
# Register at https://image-net.org/download-images first
# python -c "from datasets import load_dataset; ds = load_dataset('imagenet-1k', split='validation')"
```

---

## Runtime Support

This model is compatible with:
- **PyTorch** (via timm)
- **ONNX Runtime** (export from PyTorch)
- **TensorFlow** (official implementation)
- **TensorRT** (NVIDIA optimization)
- **ROCm** (AMD GPU acceleration)
- **RyzenAI Stack** (AMD NPU deployment)

### ONNX Export Example

```python
import torch
import timm

model = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True)
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 384, 384)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "efficientnetv2_s.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("Model exported to efficientnetv2_s.onnx")
```

---

**Document Version:** 1.0
**Last Updated:** March 2026
**Target Hardware:** AMD MI300X, RX 7900 XTX, and other ROCm-compatible GPUs
**Model Segments:** Industrial, Healthcare
**Primary Use Cases:** Image Classification, Transfer Learning, Quality Control, Medical Image Analysis
