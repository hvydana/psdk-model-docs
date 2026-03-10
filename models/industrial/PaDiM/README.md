# PaDiM - Benchmark Guide for AMD GPU

**Navigation:** [🏠 Home]({{ site.baseurl }}/) | [📑 Models Index]({{ site.baseurl }}/MODELS_INDEX) | [📝 Contributing]({{ site.baseurl }}/CONTRIBUTING)

---

## About the Model

PaDiM (Patch Distribution Modeling) is a state-of-the-art framework for anomaly detection and localization in industrial inspection applications. It uses pretrained convolutional neural networks (CNNs) to extract patch-level features and models the normal class distribution using multivariate Gaussian distributions. PaDiM excels at detecting and localizing defects in manufacturing processes without requiring retraining, making it ideal for industrial quality control where only normal samples are available during training (one-class learning).

### Original PaDiM Paper

**"PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization"** (Defard et al., 2020)

PaDiM concurrently detects and localizes anomalies in images using a one-class learning approach. The framework leverages pretrained CNN backbones for patch embedding and employs multivariate Gaussian distributions to create a probabilistic representation of the normal class. By exploiting correlations between different semantic levels of the CNN, PaDiM achieves superior anomaly localization compared to existing methods. The low computational complexity and high accuracy make PaDiM particularly suitable for industrial deployment.

**Paper:** [arXiv:2011.08785](https://arxiv.org/abs/2011.08785) | **Published:** ICPR 2021

---

## Standard Benchmark Dataset: MVTec AD

**MVTec AD (Anomaly Detection)** is the industry-standard benchmark for evaluating unsupervised anomaly detection methods in industrial inspection scenarios. It contains 5,354 high-resolution images divided into 15 different object and texture categories with pixel-precise annotations of all anomalies.

### Dataset Structure

**Texture Categories (5):**
- **Carpet**: Fabric texture variations
- **Grid**: Regular grid patterns
- **Leather**: Natural leather surface
- **Tile**: Ceramic tile patterns
- **Wood**: Wood grain textures

**Object Categories (10):**
- **Bottle**: Glass bottles
- **Cable**: Electrical cables
- **Capsule**: Pharmaceutical capsules
- **Hazelnut**: Nuts with shell
- **Metal Nut**: Metal threaded nuts
- **Pill**: Pharmaceutical pills
- **Screw**: Metal screws
- **Toothbrush**: Plastic toothbrushes
- **Transistor**: Electronic components
- **Zipper**: Fabric zippers

### Defect Types
The dataset includes over 70 different types of defects including scratches, dents, contaminations, structural changes, color variations, and missing components.

### Download MVTec AD

**Official Download:** [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

**Alternative Sources:**
- [Kaggle MVTec AD](https://www.kaggle.com/datasets/ipythonx/mvtec-ad)
- [IEEE DataPort](https://ieee-dataport.org/documents/mvtec-ad)

```python
# Using Anomalib library (recommended)
from anomalib.data import MVTec

# Initialize MVTec dataset
datamodule = MVTec(
    root="./datasets/MVTec",
    category="bottle",  # Choose from 15 categories
    image_size=256,
    train_batch_size=32,
    eval_batch_size=32,
)

# Setup will download dataset if not present
datamodule.setup()

# Access data
train_data = datamodule.train_dataloader()
test_data = datamodule.test_dataloader()
```

**Manual Download:**
```bash
# Download from official source
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz

# Extract dataset
tar -xf mvtec_anomaly_detection.tar.xz

# Dataset structure:
# mvtec_anomaly_detection/
# ├── bottle/
# │   ├── train/good/
# │   ├── test/good/
# │   ├── test/broken_large/
# │   └── ground_truth/broken_large/
# └── [other categories...]
```

---

## Installation & Inference

### Install Anomalib with PaDiM

```bash
# Install Anomalib with ROCm support for AMD GPUs
pip install anomalib

# Or install with ROCm backend explicitly
uv pip install "anomalib[rocm]"

# Install PyTorch with ROCm support (if not already installed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# Verify installation
python -c "import anomalib; print(f'Anomalib version: {anomalib.__version__}')"
```

### Basic Inference with Python API

```python
from anomalib.data import MVTec
from anomalib.models import Padim
from anomalib.engine import Engine
import torch

# Check device availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize data module
datamodule = MVTec(
    root="./datasets/MVTec",
    category="bottle",
    image_size=256,
    train_batch_size=32,
    eval_batch_size=32,
)

# Initialize PaDiM model
model = Padim(
    backbone="resnet18",  # Options: resnet18, wide_resnet50_2
    layers=["layer1", "layer2", "layer3"],  # CNN layers for feature extraction
    pre_trained=True,
    n_features=100,  # Reduced dimensionality (100 for ResNet18, 550 for WideResNet50)
)

# Initialize engine
engine = Engine(
    accelerator="gpu",  # Use "gpu" for AMD/NVIDIA GPUs
    devices=1,
    logger=False,
)

# Train (fit Gaussian distributions)
engine.train(datamodule=datamodule, model=model)

# Test
predictions = engine.test(datamodule=datamodule, model=model)
```

### Inference on Single Image

```python
from anomalib.deploy import OpenVINOInferencer
from PIL import Image
import matplotlib.pyplot as plt

# Load trained model
inferencer = OpenVINOInferencer(
    path="results/padim/mvtec/bottle/weights/openvino/model.bin",
    metadata="results/padim/mvtec/bottle/weights/openvino/metadata.json",
)

# Run inference on single image
image_path = "test_image.png"
predictions = inferencer.predict(image=image_path)

# Visualize results
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(Image.open(image_path))
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(predictions.anomaly_map, cmap='jet')
plt.title("Anomaly Map")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(predictions.segmentations)
plt.title("Segmentation Mask")
plt.axis('off')

plt.tight_layout()
plt.savefig("anomaly_detection_result.png")
```

### Using Alternative Backbones

```python
# ResNet18 (faster, lower memory, good accuracy)
model_r18 = Padim(
    backbone="resnet18",
    layers=["layer1", "layer2", "layer3"],
    n_features=100,
)

# WideResNet50 (slower, higher memory, best accuracy)
model_wr50 = Padim(
    backbone="wide_resnet50_2",
    layers=["layer1", "layer2", "layer3"],
    n_features=550,
)

# EfficientNet-B5 (balanced performance)
model_effnet = Padim(
    backbone="efficientnet_b5",
    layers=["blocks.2", "blocks.4", "blocks.6"],
    n_features=300,
)
```

### Expected Output

```json
{
  "image_path": "test_image.png",
  "pred_score": 0.8742,
  "pred_label": "Anomaly",
  "anomaly_map": "numpy_array[256, 256]",
  "segmentations": "binary_mask[256, 256]",
  "box_labels": [
    {
      "label": "Anomaly",
      "confidence": 0.8742,
      "bbox": [120, 85, 200, 165]
    }
  ]
}
```

---

## Benchmark Results & Performance Metrics

### PaDiM Performance on MVTec AD (Image-Level AUROC %)

| Category | ResNet18-Rd100 | WideResNet50-Rd550 | Improvement | Class Type |
|----------|----------------|---------------------|-------------|------------|
| **Carpet** | 98.8 | 99.0 | +0.2% | Texture |
| **Grid** | 93.6 | 96.5 | +2.9% | Texture |
| **Leather** | 99.0 | 98.9 | -0.1% | Texture |
| **Tile** | 91.7 | 93.9 | +2.2% | Texture |
| **Wood** | 94.0 | 94.1 | +0.1% | Texture |
| **Bottle** | 98.1 | 98.2 | +0.1% | Object |
| **Cable** | 94.9 | 96.8 | +1.9% | Object |
| **Capsule** | 98.2 | 98.6 | +0.4% | Object |
| **Hazelnut** | 97.9 | 97.9 | 0.0% | Object |
| **Metal Nut** | 96.7 | 97.1 | +0.4% | Object |
| **Pill** | 94.6 | 96.1 | +1.5% | Object |
| **Screw** | 97.2 | 98.3 | +1.1% | Object |
| **Toothbrush** | 98.6 | 98.7 | +0.1% | Object |
| **Transistor** | 96.8 | 97.5 | +0.7% | Object |
| **Zipper** | 97.6 | 98.4 | +0.8% | Object |
| **Texture Avg** | 95.3 | 96.5 | +1.2% | - |
| **Object Avg** | 97.1 | 97.8 | +0.7% | - |
| **Overall Avg** | 96.5 | 97.3 | +0.8% | - |

**AUROC** = Area Under Receiver Operating Characteristic (higher is better, 100% = perfect)

### Pixel-Level Anomaly Localization Performance

| Metric | ResNet18 | WideResNet50 | Description |
|--------|----------|--------------|-------------|
| **Pixel AUROC (%)** | 97.5 | 98.1 | Pixel-wise anomaly detection accuracy |
| **PRO Score (%)** | 92.1 | 93.5 | Per-Region Overlap metric |
| **Average Precision** | 94.2 | 95.8 | Precision-recall curve area |

### Comparison with Other Methods on MVTec AD

| Method | Image AUROC (%) | Pixel AUROC (%) | PRO Score (%) | Year |
|--------|----------------|-----------------|---------------|------|
| **PaDiM (WR50-Rd550)** | **97.3** | **98.1** | **93.5** | 2020 |
| PatchCore | 99.1 | 98.1 | 95.3 | 2021 |
| FastFlow | 99.4 | 98.5 | 94.2 | 2021 |
| SPADE | 85.5 | 96.5 | 88.9 | 2020 |
| DFM | 95.1 | 93.8 | - | 2020 |
| AutoEncoder | 84.5 | 89.2 | - | Baseline |

**Note:** PatchCore achieves higher accuracy but requires significantly more memory. PaDiM offers the best balance of accuracy, speed, and memory efficiency.

### Computational Efficiency Comparison

| Method | Training Time* | Inference Time* | Memory Usage | Complexity |
|--------|---------------|-----------------|--------------|------------|
| **PaDiM-R18** | ~3 min | ~15 ms/image | ~2 GB | Low |
| **PaDiM-WR50** | ~8 min | ~45 ms/image | ~4 GB | Medium |
| PatchCore | ~10 min | ~60 ms/image | ~8 GB | High |
| FastFlow | ~25 min | ~20 ms/image | ~3 GB | Medium |

*Approximate times on NVIDIA RTX 3090, single category training

---

## AMD GPU Benchmarking Setup

### ROCm Installation for AMD GPUs

```bash
# Check ROCm compatibility
rocm-smi

# Verify ROCm version
rocminfo | grep "Marketing Name"

# Install PyTorch with ROCm support (ROCm 6.2)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# For ROCm 6.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

# Verify CUDA/ROCm availability in PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Install Anomalib with ROCm support
uv pip install "anomalib[rocm]"
```

### Benchmark Script for AMD GPU

```python
import torch
import time
import numpy as np
from anomalib.data import MVTec
from anomalib.models import Padim
from anomalib.engine import Engine
from pathlib import Path

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Configuration
CATEGORY = "bottle"  # Test category
BACKBONE = "wide_resnet50_2"  # or "resnet18"
N_FEATURES = 550  # 550 for WR50, 100 for R18

# Initialize data module
datamodule = MVTec(
    root="./datasets/MVTec",
    category=CATEGORY,
    image_size=256,
    train_batch_size=32,
    eval_batch_size=1,  # Single image for inference timing
)
datamodule.setup()

# Initialize model
model = Padim(
    backbone=BACKBONE,
    layers=["layer1", "layer2", "layer3"],
    pre_trained=True,
    n_features=N_FEATURES,
)

# Initialize engine
engine = Engine(
    accelerator="gpu",
    devices=1,
    logger=False,
)

# Training phase (fitting Gaussian distributions)
print("\n=== Training Phase ===")
train_start = time.time()
engine.train(datamodule=datamodule, model=model)
train_end = time.time()
train_time = train_end - train_start

print(f"Training time: {train_time:.2f} seconds")
print(f"Memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

# Inference benchmarking
print("\n=== Inference Benchmarking ===")
test_loader = datamodule.test_dataloader()

inference_times = []
for i, batch in enumerate(test_loader):
    if i >= 100:  # Benchmark on 100 images
        break

    # Warmup
    if i < 5:
        _ = model(batch)
        continue

    # Measure inference time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    predictions = model(batch)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()

    inference_times.append(end_time - start_time)

# Calculate statistics
avg_inference = np.mean(inference_times)
std_inference = np.std(inference_times)
min_inference = np.min(inference_times)
max_inference = np.max(inference_times)
throughput = 1.0 / avg_inference

print(f"\nInference Statistics (n={len(inference_times)}):")
print(f"  Average: {avg_inference*1000:.2f} ms/image")
print(f"  Std Dev: {std_inference*1000:.2f} ms")
print(f"  Min: {min_inference*1000:.2f} ms")
print(f"  Max: {max_inference*1000:.2f} ms")
print(f"  Throughput: {throughput:.2f} images/second")

# Memory statistics
print(f"\nMemory Usage:")
print(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"  Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
print(f"  Max Allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

# Test phase (full evaluation)
print("\n=== Test Phase ===")
test_start = time.time()
test_results = engine.test(datamodule=datamodule, model=model)
test_end = time.time()
test_time = test_end - test_start

print(f"Test time: {test_time:.2f} seconds")
print(f"Image AUROC: {test_results[0]['image_AUROC']:.4f}")
print(f"Pixel AUROC: {test_results[0]['pixel_AUROC']:.4f}")
```

### Performance Metrics Table Template

| Metric | NVIDIA A100-80GB | NVIDIA RTX 3090 | AMD MI300X | AMD RX 7900 XTX | Notes |
|--------|------------------|-----------------|------------|-----------------|-------|
| **GPU Model** | NVIDIA A100-80GB | NVIDIA RTX 3090 | AMD MI300X | AMD RX 7900 XTX | Compare datacenter vs consumer GPUs |
| **Memory (GB)** | 80 | 24 | 192 | 24 | VRAM capacity |
| **TDP (W)** | 400 | 350 | 750 | 355 | Thermal design power |
| **Training Time (s)** | ~180 | ~240 | _[Your result]_ | _[Your result]_ | Fitting Gaussians on 1 category |
| **Inference (ms/image)** | ~12 | ~15 | _[Your result]_ | _[Your result]_ | Single 256x256 image |
| **Throughput (img/s)** | ~83 | ~67 | _[Your result]_ | _[Your result]_ | Images processed per second |
| **Image AUROC (%)** | 97.3 | 97.3 | _[Your result]_ | _[Your result]_ | Detection accuracy (MVTec bottle) |
| **Pixel AUROC (%)** | 98.1 | 98.1 | _[Your result]_ | _[Your result]_ | Localization accuracy |
| **Peak Memory (GB)** | ~4 | ~4 | _[Your result]_ | _[Your result]_ | WideResNet50 backbone |
| **Average Power (W)** | ~280 | ~300 | _[Your result]_ | _[Your result]_ | During inference |
| **Energy/1K Images (Wh)** | ~0.93 | ~1.25 | _[Your result]_ | _[Your result]_ | Lower is better |

### AMD-Specific Metrics to Track

```python
# GPU utilization tracking
import subprocess

def get_rocm_smi_stats():
    """Get AMD GPU statistics using rocm-smi"""
    result = subprocess.run(
        ['rocm-smi', '--showuse', '--showmeminfo', 'vram', '--showpower'],
        capture_output=True,
        text=True
    )
    return result.stdout

# Monitor during inference
print("=== ROCm GPU Stats ===")
print(get_rocm_smi_stats())

# Memory tracking with PyTorch
print("\n=== PyTorch Memory Stats ===")
print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
print(f"Max Allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

# ROCm information
if torch.cuda.is_available():
    print("\n=== ROCm Information ===")
    print(f"ROCm Version: {torch.version.hip}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Number of devices: {torch.cuda.device_count()}")
```

### Complete Runtime Metrics Table

| Runtime Metric | Formula | NVIDIA A100 | NVIDIA RTX 3090 | AMD MI300X | AMD RX 7900 XTX | Notes |
|----------------|---------|-------------|-----------------|------------|-----------------|-------|
| **Inference Latency (ms)** | time_per_image | 12 | 15 | _[Your result]_ | _[Your result]_ | Single image processing |
| **Throughput (img/s)** | 1000 / latency | 83.3 | 66.7 | _[Your result]_ | _[Your result]_ | Batch size = 1 |
| **Batch Throughput (img/s)** | batch_size / batch_time | ~250 | ~200 | _[Your result]_ | _[Your result]_ | Batch size = 32 |
| **GPU Utilization (%)** | From rocm-smi | ~85 | ~80 | _[Your result]_ | _[Your result]_ | Average during inference |
| **Memory Bandwidth (GB/s)** | From rocm-smi | ~2000 | ~936 | _[Your result]_ | _[Your result]_ | MI300X: ~5300 theoretical |
| **TFLOPS Utilized** | Calculated | ~150 | ~35 | _[Your result]_ | _[Your result]_ | FP16 compute throughput |
| **Training Time (min)** | Total fitting time | 3.0 | 4.0 | _[Your result]_ | _[Your result]_ | Single category (bottle) |
| **Energy Efficiency (Wh/1K img)** | power × time / 1000 | 0.93 | 1.25 | _[Your result]_ | _[Your result]_ | Lower is better |
| **Cost per 1M images ($)** | Based on cloud pricing | ~0.50 | ~0.15 | _[Your result]_ | _[Your result]_ | Estimated operational cost |

---

## Additional Benchmark Datasets

### MVTec AD 2 (Extended Dataset)

**MVTec AD 2** expands the original benchmark with 8 new anomaly detection scenarios and over 8,000 high-resolution images.

**Download:** [MVTec AD 2 Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad-2)

### BTAD (BeanTech Anomaly Detection)

Real-world industrial dataset with 2,540 images across 3 product categories with various defect types.

**Download:** [BTAD on Kaggle](https://www.kaggle.com/datasets/thtuan/btad-beantech-anomaly-detection)

```python
# Using BTAD with Anomalib
from anomalib.data import Folder

datamodule = Folder(
    root="./datasets/BTAD",
    normal_dir="./datasets/BTAD/01/train/ok",
    abnormal_dir="./datasets/BTAD/01/test/ko",
    mask_dir="./datasets/BTAD/01/ground_truth/ko",
    image_size=256,
)
```

### VisA (Visual Anomaly Detection)

12 object categories with 10,821 images including 9,621 normal and 1,200 anomalous samples.

### Real-IAD (Real-World Industrial Anomaly Detection)

Large-scale multi-view dataset with 151,050 high-resolution images of 30 different objects.

---

## Using PaDiM with Anomalib CLI

```bash
# Train PaDiM on MVTec bottle category
anomalib train \
    --model padim \
    --data anomalib.data.MVTec \
    --data.category bottle \
    --data.image_size 256 \
    --model.backbone wide_resnet50_2 \
    --model.layers [layer1,layer2,layer3] \
    --trainer.accelerator gpu \
    --trainer.devices 1 \
    --trainer.max_epochs 1

# Test trained model
anomalib test \
    --model padim \
    --data anomalib.data.MVTec \
    --data.category bottle \
    --ckpt_path results/padim/mvtec/bottle/weights/model.ckpt

# Predict on new images
anomalib predict \
    --model padim \
    --data anomalib.data.MVTec \
    --data.category bottle \
    --ckpt_path results/padim/mvtec/bottle/weights/model.ckpt \
    --return_predictions

# Export to ONNX for deployment
anomalib export \
    --model padim \
    --ckpt_path results/padim/mvtec/bottle/weights/model.ckpt \
    --export_type onnx \
    --input_size [256,256]

# Export to OpenVINO for Intel inference optimization
anomalib export \
    --model padim \
    --ckpt_path results/padim/mvtec/bottle/weights/model.ckpt \
    --export_type openvino
```

---

## Advanced Configuration

### Custom Configuration File (config.yaml)

```yaml
model:
  class_path: anomalib.models.Padim
  init_args:
    backbone: wide_resnet50_2
    layers:
      - layer1
      - layer2
      - layer3
    pre_trained: true
    n_features: 550

data:
  class_path: anomalib.data.MVTec
  init_args:
    root: ./datasets/MVTec
    category: bottle
    image_size: 256
    train_batch_size: 32
    eval_batch_size: 32
    num_workers: 8

trainer:
  accelerator: gpu
  devices: 1
  max_epochs: 1
  log_every_n_steps: 10

optimization:
  export_type: onnx

logging:
  logger: [tensorboard, wandb]
  log_images: true
```

```bash
# Train with config file
anomalib train --config config.yaml
```

### Multi-Category Training Script

```python
from anomalib.data import MVTec
from anomalib.models import Padim
from anomalib.engine import Engine
import pandas as pd

# MVTec categories
categories = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

results = []

for category in categories:
    print(f"\n{'='*50}")
    print(f"Training on category: {category}")
    print(f"{'='*50}")

    # Initialize data module
    datamodule = MVTec(
        root="./datasets/MVTec",
        category=category,
        image_size=256,
        train_batch_size=32,
    )

    # Initialize model
    model = Padim(
        backbone="wide_resnet50_2",
        layers=["layer1", "layer2", "layer3"],
        n_features=550,
    )

    # Initialize engine
    engine = Engine(
        accelerator="gpu",
        devices=1,
        default_root_dir=f"results/{category}",
    )

    # Train and test
    engine.train(datamodule=datamodule, model=model)
    test_results = engine.test(datamodule=datamodule, model=model)

    # Collect results
    results.append({
        "category": category,
        "image_auroc": test_results[0]["image_AUROC"],
        "pixel_auroc": test_results[0]["pixel_AUROC"],
    })

# Save results
df = pd.DataFrame(results)
df.to_csv("padim_all_categories_results.csv", index=False)
print("\n=== Final Results ===")
print(df)
print(f"\nAverage Image AUROC: {df['image_auroc'].mean():.4f}")
print(f"Average Pixel AUROC: {df['pixel_auroc'].mean():.4f}")
```

---

## Additional Resources

### Official Repositories

- [Anomalib Library](https://github.com/open-edge-platform/anomalib) - Production-ready implementation
- [PaDiM Implementation (xiahaifeng1995)](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master)
- [PaDiM Implementation (taikiinoue45)](https://github.com/taikiinoue45/PaDiM)
- [WE-PaDiM (Wavelet-Enhanced)](https://github.com/BioHPC/WE-PaDiM)

### Papers & Documentation

- [PaDiM Paper (arXiv:2011.08785)](https://arxiv.org/abs/2011.08785)
- [PaDiM Paper (HAL PDF)](https://cea.hal.science/cea-03251821/document)
- [Anomalib Documentation](https://anomalib.readthedocs.io/)
- [PaDiM in Anomalib](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/padim.html)

### Datasets

- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [MVTec AD 2 Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad-2)
- [BTAD Dataset (Kaggle)](https://www.kaggle.com/datasets/thtuan/btad-beantech-anomaly-detection)
- [MVTec AD (IEEE DataPort)](https://ieee-dataport.org/documents/mvtec-ad)

### Blog Posts & Tutorials

- [PaDiM: Machine Learning for Defect Detection (Medium)](https://medium.com/axinc-ai/padim-a-machine-learning-model-for-detecting-defective-products-without-retraining-5daa6f203377)
- [Anomalib in 15 Minutes](https://anomalib.readthedocs.io/en/latest/markdown/get_started/anomalib.html)
- [MVTec AD Anomaly Detection with Anomalib (Kaggle)](https://www.kaggle.com/code/ipythonx/mvtec-ad-anomaly-detection-with-anomalib-library)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch on ROCm Installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html)

### Leaderboards & Benchmarks

- [MVTec AD Benchmark (Papers With Code)](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad)
- [Industrial Anomaly Detection Papers (GitHub)](https://github.com/M-3LAB/awesome-industrial-anomaly-detection)
- [Industrial Anomaly Detection Datasets (GitHub)](https://github.com/SSRheart/industrial-anomaly-detection-dataset)

---

## Quick Reference Commands

```bash
# Install Anomalib with ROCm support
uv pip install "anomalib[rocm]"

# Install PyTorch with ROCm 6.2
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# Check AMD GPU status
rocm-smi
rocm-smi --showuse --showmeminfo vram --showpower

# Download MVTec AD dataset (requires manual download from official site)
# Visit: https://www.mvtec.com/company/research/datasets/mvtec-ad

# Train PaDiM on bottle category
anomalib train --model padim --data anomalib.data.MVTec --data.category bottle

# Export trained model to ONNX
anomalib export --model padim --ckpt_path results/padim/mvtec/bottle/weights/model.ckpt --export_type onnx

# Get help
anomalib --help
python -m anomalib.models.padim --help
```

---

## Performance Optimization Tips

### For AMD GPUs

1. **Use ROCm-optimized PyTorch**: Install PyTorch with ROCm backend for best performance
2. **Enable mixed precision**: Use FP16 for faster inference with minimal accuracy loss
3. **Optimize batch size**: Test different batch sizes (16, 32, 64) to maximize GPU utilization
4. **Use WideResNet50**: Better accuracy but requires more memory; use ResNet18 for memory-constrained scenarios
5. **Reduce dimensionality**: Lower n_features (e.g., 100 for R18) for faster inference
6. **Enable TensorCore/Matrix operations**: Ensure ROCm drivers support optimized matrix operations

### Memory Management

```python
# Clear cache periodically
torch.cuda.empty_cache()

# Use gradient checkpointing for large models
model.gradient_checkpointing_enable()

# Monitor memory during training
print(f"Memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
```

### Inference Optimization

```python
# Use torch.no_grad() for inference
with torch.no_grad():
    predictions = model(batch)

# Use torch.cuda.amp for mixed precision
from torch.cuda.amp import autocast

with autocast():
    predictions = model(batch)

# Export to ONNX Runtime for deployment
# ONNX Runtime provides optimized inference across platforms
```

---

**Document Version:** 1.0
**Last Updated:** March 2026
**Target Hardware:** AMD MI300X, RX 7900 XTX, and other ROCm-compatible GPUs

---

## Sources

- [PaDiM Paper (arXiv)](https://arxiv.org/abs/2011.08785)
- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [Anomalib Documentation](https://anomalib.readthedocs.io/)
- [PyTorch ROCm Installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html)
- [Industrial Anomaly Detection Survey](https://github.com/M-3LAB/awesome-industrial-anomaly-detection)
