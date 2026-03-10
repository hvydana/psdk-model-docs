# MobileNetV3 - Benchmark Guide for AMD GPU

**Navigation:** [🏠 Home]({{ site.baseurl }}/) | [📑 Models Index]({{ site.baseurl }}/MODELS_INDEX) | [📝 Contributing]({{ site.baseurl }}/CONTRIBUTING)

---

## About the Model

MobileNetV3 is a state-of-the-art efficient convolutional neural network (CNN) architecture designed for mobile and edge devices. It represents the third generation of MobileNet models, optimized through hardware-aware neural architecture search (NAS) combined with the NetAdapt algorithm. MobileNetV3 delivers significant improvements in accuracy while maintaining or reducing latency compared to previous versions, making it ideal for industrial computer vision applications on resource-constrained devices.

### Original MobileNetV3 Paper

**"Searching for MobileNetV3"** (Howard et al., 2019)

MobileNetV3 is tuned to mobile phone CPUs through a combination of hardware-aware network architecture search (NAS) complemented by the NetAdapt algorithm and subsequently improved through novel architecture advances. The paper presents two models: MobileNetV3-Large for high-accuracy use cases and MobileNetV3-Small for resource-constrained environments. Key innovations include complementary search techniques, new efficient versions of nonlinearities practical for mobile settings, efficient network design, and new efficient segmentation decoder.

MobileNetV3-Large achieves 75.2% top-1 accuracy on ImageNet at 219M MACs with 51-61ms latency on Google Pixel phones, representing a 3.2% accuracy improvement over MobileNetV2 with 15-20% reduced latency. MobileNetV3-Small achieves 4.6-6.6% higher accuracy while reducing latency by 5% compared to MobileNetV2.

**Paper:** [arXiv:1905.02244](https://arxiv.org/abs/1905.02244) | **Published:** ICCV 2019

---

## Standard Benchmark Dataset: ImageNet

**ImageNet** is the industry-standard benchmark for evaluating image classification systems. The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) dataset contains ~1.2 million training images and 50,000 validation images across 1,000 classes.

### Dataset Structure
- **Training set**: 1,281,167 images across 1,000 classes
- **Validation set**: 50,000 images (50 per class)
- **Test set**: 100,000 images (100 per class)
- **Image resolution**: Variable (typically 224x224 after preprocessing)

### Download from HuggingFace

```bash
# Install dependencies
pip install datasets transformers torchvision
```

```python
from datasets import load_dataset

# Load ImageNet-1k dataset
dataset = load_dataset("imagenet-1k", split="validation")

# Or use a smaller subset for testing
dataset = load_dataset("imagenet-1k", split="validation[:100]")

# View a sample
print(dataset[0])
# Output: {'image': <PIL.Image>, 'label': 65, ...}
```

### Additional Benchmark Datasets for Industrial Applications

**COCO (Common Objects in Context)**
- 330K images (>200K labeled)
- 1.5 million object instances
- 80 object categories
- Used for object detection, segmentation, keypoint detection

**CIFAR-10/100**
- CIFAR-10: 60,000 32x32 color images in 10 classes
- CIFAR-100: 60,000 images in 100 classes
- Lightweight dataset for quick benchmarking

```python
# Load CIFAR-10
from datasets import load_dataset
cifar10 = load_dataset("cifar10", split="test")

# Load COCO for object detection
coco = load_dataset("detection-datasets/coco", split="validation")
```

---

## Installation & Inference

### Install PyTorch with ROCm for AMD GPUs

```bash
# Install PyTorch with ROCm 6.2 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### Basic Inference Using TorchVision

```python
import torch
from torchvision import models, transforms
from PIL import Image

# Load pretrained MobileNetV3-Large
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
model.eval()

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess image
img = Image.open("example.jpg")
input_tensor = preprocess(img).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    output = model(input_tensor)

# Get predicted class
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)

print(f"Top prediction: Class {top5_catid[0].item()} with probability {top5_prob[0].item():.4f}")
```

### MobileNetV3-Small for Resource-Constrained Devices

```python
# Load MobileNetV3-Small variant
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
model.eval()
model = model.to(device)

# Same inference pipeline as above
```

### Batch Inference for Throughput Benchmarking

```python
import torch
from torchvision import models, transforms
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
model = model.to(device)
model.eval()

# Create random batch of images
batch_size = 32
dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)

# Warmup
for _ in range(10):
    with torch.no_grad():
        _ = model(dummy_input)

# Benchmark
num_iterations = 100
torch.cuda.synchronize() if torch.cuda.is_available() else None

start_time = time.time()
for _ in range(num_iterations):
    with torch.no_grad():
        _ = model(dummy_input)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

end_time = time.time()

total_images = batch_size * num_iterations
throughput = total_images / (end_time - start_time)
latency_per_image = (end_time - start_time) / total_images * 1000  # ms

print(f"Throughput: {throughput:.2f} images/sec")
print(f"Latency per image: {latency_per_image:.2f} ms")
print(f"Batch latency: {(end_time - start_time) / num_iterations * 1000:.2f} ms")
```

### Expected Output

```json
{
  "predicted_class": 285,
  "class_name": "Egyptian cat",
  "confidence": 0.9234,
  "top_5_predictions": [
    {"class": 285, "name": "Egyptian cat", "probability": 0.9234},
    {"class": 281, "name": "tabby cat", "probability": 0.0521},
    {"class": 282, "name": "tiger cat", "probability": 0.0183},
    {"class": 287, "name": "lynx", "probability": 0.0032},
    {"class": 286, "name": "cougar", "probability": 0.0018}
  ]
}
```

---

## Benchmark Results & Performance Metrics

### MobileNetV3 Performance on ImageNet

| Model | Top-1 Accuracy | Top-5 Accuracy | Parameters | MACs | Latency (Pixel 3) | Notes |
|-------|---------------|----------------|------------|------|-------------------|-------|
| **MobileNetV3-Large** | 75.2% | 92.2% | 5.4M | 219M | 51-61 ms | Main variant |
| **MobileNetV3-Large (1.25x)** | 76.6% | - | - | 356M | - | Higher accuracy |
| **MobileNetV3-Small** | 67.4% | 87.0% | 2.5M | 66M | ~15 ms | Efficient variant |
| **MobileNetV3-Small (0.75x)** | 65.4% | - | 2.0M | 44M | - | Ultra-efficient |
| MobileNetV2 (1.0) | 72.0% | 91.0% | 3.4M | 300M | ~75 ms | Previous generation |
| ResNet-50 | 76.0% | 93.0% | 25.6M | 4.1B | ~300 ms | Comparison baseline |

**Improvements over MobileNetV2:**
- MobileNetV3-Large: +3.2% accuracy, -15-20% latency
- MobileNetV3-Small: +4.6-6.6% accuracy, -5% latency

### Performance on Other Benchmarks

| Dataset | Task | MobileNetV3-Large | MobileNetV3-Small | Notes |
|---------|------|------------------|-------------------|-------|
| **COCO** | Object Detection (mAP) | ~22% | ~17% | 27% faster than MobileNetV2 |
| **CIFAR-10** | Classification | 95.49% | 72.54% | Model size <8 MB |
| **Tiny ImageNet** | Classification | - | 72.54% | Sub-0.1 ms on Tesla P100 |
| **PASCAL VOC** | Segmentation | - | - | 15% latency reduction with optimizations |

### Inference Speed Comparisons

| Implementation | Platform | FPS | Latency | Notes |
|----------------|----------|-----|---------|-------|
| **MobileNetV3-Large ONNX** | NVIDIA GPU | ~170 FPS | ~6 ms | Optimized inference |
| **MobileNetV3-Large** | CPU (single core) | ~24-30 FPS | ~33-40 ms | Real-time capable |
| **MobileNetV3-Small** | Tesla P100 | - | <0.1 ms | Ultra-low latency |
| **MobileNetV3 vs ResNet50** | CPU | 6x faster | - | Similar accuracy |
| **MobileNetV3-Large** | iPhone/iPad (older) | 30+ FPS | ~33 ms | Real-time mobile |

**Real-Time Factor (RTF):** For video processing at 30 FPS, models achieving >30 FPS are faster than real-time (RTF < 1.0)

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
from torchvision import models, transforms
from datasets import load_dataset
import time
import numpy as np

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"ROCm Version: {torch.version.hip}")

# Load model
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
model = model.to(device).to(torch_dtype)
model.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageNet validation subset
print("Loading ImageNet validation dataset...")
dataset = load_dataset("imagenet-1k", split="validation[:100]", trust_remote_code=True)

# Benchmark
results = []
warmup_iterations = 10

print("Running warmup iterations...")
for i in range(warmup_iterations):
    sample = dataset[i % len(dataset)]
    input_tensor = preprocess(sample["image"]).unsqueeze(0).to(device).to(torch_dtype)
    with torch.no_grad():
        _ = model(input_tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

print("Running benchmark...")
for i, sample in enumerate(dataset):
    input_tensor = preprocess(sample["image"]).unsqueeze(0).to(device).to(torch_dtype)

    # Measure inference time
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds

    # Get prediction
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class].item()

    results.append({
        "sample_id": i,
        "inference_time_ms": inference_time,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "ground_truth": sample["label"]
    })

    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(dataset)} images...")

# Calculate metrics
inference_times = [r["inference_time_ms"] for r in results]
avg_latency = np.mean(inference_times)
std_latency = np.std(inference_times)
min_latency = np.min(inference_times)
max_latency = np.max(inference_times)
p50_latency = np.percentile(inference_times, 50)
p95_latency = np.percentile(inference_times, 95)
p99_latency = np.percentile(inference_times, 99)
throughput = 1000 / avg_latency  # images per second
fps = throughput

# Calculate accuracy
correct = sum(1 for r in results if r["predicted_class"] == r["ground_truth"])
accuracy = correct / len(results) * 100

# GPU memory usage
if torch.cuda.is_available():
    allocated_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved_memory = torch.cuda.memory_reserved() / 1024**3  # GB
    max_allocated_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
else:
    allocated_memory = reserved_memory = max_allocated_memory = 0

# Print summary
print("\n" + "="*60)
print("BENCHMARK RESULTS")
print("="*60)
print(f"Model: MobileNetV3-Large")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Samples: {len(results)}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"\nLatency Metrics:")
print(f"  Average: {avg_latency:.2f} ms")
print(f"  Std Dev: {std_latency:.2f} ms")
print(f"  Min: {min_latency:.2f} ms")
print(f"  Max: {max_latency:.2f} ms")
print(f"  P50: {p50_latency:.2f} ms")
print(f"  P95: {p95_latency:.2f} ms")
print(f"  P99: {p99_latency:.2f} ms")
print(f"\nThroughput Metrics:")
print(f"  Images/sec: {throughput:.2f}")
print(f"  FPS: {fps:.2f}")
print(f"\nMemory Usage:")
print(f"  Allocated: {allocated_memory:.2f} GB")
print(f"  Reserved: {reserved_memory:.2f} GB")
print(f"  Peak: {max_allocated_memory:.2f} GB")
print("="*60)

# Export results
import json
with open("mobilenetv3_benchmark_results.json", "w") as f:
    json.dump({
        "model": "MobileNetV3-Large",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "num_samples": len(results),
        "accuracy": accuracy,
        "latency": {
            "avg_ms": avg_latency,
            "std_ms": std_latency,
            "min_ms": min_latency,
            "max_ms": max_latency,
            "p50_ms": p50_latency,
            "p95_ms": p95_latency,
            "p99_ms": p99_latency
        },
        "throughput": {
            "images_per_sec": throughput,
            "fps": fps
        },
        "memory": {
            "allocated_gb": allocated_memory,
            "reserved_gb": reserved_memory,
            "peak_gb": max_allocated_memory
        },
        "detailed_results": results
    }, f, indent=2)

print(f"\nDetailed results saved to: mobilenetv3_benchmark_results.json")
```

### Batch Processing Benchmark for Higher Throughput

```python
# Batch benchmark for maximum throughput
batch_sizes = [1, 4, 8, 16, 32, 64]
batch_results = {}

for batch_size in batch_sizes:
    print(f"\nBenchmarking batch size: {batch_size}")

    # Create random batch
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device).to(torch_dtype)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Benchmark
    num_iterations = 100
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    end_time = time.time()

    total_images = batch_size * num_iterations
    total_time = end_time - start_time
    throughput = total_images / total_time
    latency_per_batch = total_time / num_iterations * 1000  # ms
    latency_per_image = total_time / total_images * 1000  # ms

    batch_results[batch_size] = {
        "throughput": throughput,
        "latency_per_batch_ms": latency_per_batch,
        "latency_per_image_ms": latency_per_image
    }

    print(f"  Throughput: {throughput:.2f} images/sec")
    print(f"  Latency per batch: {latency_per_batch:.2f} ms")
    print(f"  Latency per image: {latency_per_image:.2f} ms")

# Print optimal batch size
optimal_batch = max(batch_results.items(), key=lambda x: x[1]["throughput"])
print(f"\nOptimal batch size: {optimal_batch[0]} with throughput: {optimal_batch[1]['throughput']:.2f} images/sec")
```

### Performance Metrics Table Template

| Metric | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Notes |
|--------|------------------|-----------|------------|-----------------|-------|
| **GPU Model** | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Compare datacenter vs consumer GPUs |
| **Memory (GB)** | 80 | 16 | 192 | 24 | VRAM capacity |
| **TDP (W)** | 400 | 70 | 750 | 355 | Thermal design power |
| **Average Latency (ms)** | ~2.5 | ~8.0 | _[Your result]_ | _[Your result]_ | Single image inference |
| **P95 Latency (ms)** | ~3.0 | ~9.5 | _[Your result]_ | _[Your result]_ | 95th percentile |
| **P99 Latency (ms)** | ~3.5 | ~11.0 | _[Your result]_ | _[Your result]_ | 99th percentile |
| **Throughput (images/sec)** | ~400 | ~125 | _[Your result]_ | _[Your result]_ | Single batch |
| **Batch-32 Throughput** | ~2000 | ~500 | _[Your result]_ | _[Your result]_ | Optimal batch size |
| **FPS** | ~400 | ~125 | _[Your result]_ | _[Your result]_ | Frames per second |
| **Peak Memory Usage (GB)** | ~2.5 | ~1.8 | _[Your result]_ | _[Your result]_ | Monitor with rocm-smi |
| **Average Power Draw (W)** | ~250 | ~50 | _[Your result]_ | _[Your result]_ | Monitor with rocm-smi --showpower |
| **Energy per 1000 Images (Wh)** | ~0.17 | ~0.11 | _[Your result]_ | _[Your result]_ | Lower is better |

### AMD-Specific Metrics to Track

```python
# GPU utilization tracking
import subprocess

def get_rocm_smi_stats():
    """Get AMD GPU statistics using rocm-smi"""
    result = subprocess.run(['rocm-smi', '--showuse', '--showmeminfo', 'vram'],
                          capture_output=True, text=True)
    return result.stdout

def get_power_usage():
    """Get AMD GPU power usage"""
    result = subprocess.run(['rocm-smi', '--showpower'],
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

# During benchmark loop
print("\nGPU Stats:")
print(get_rocm_smi_stats())
print("\nPower Usage:")
print(get_power_usage())
```

### Complete Runtime Metrics Table

| Runtime Metric | Formula | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Notes |
|----------------|---------|------------------|-----------|------------|-----------------|-------|
| **Latency (ms)** | inference_time × 1000 | ~2.5 | ~8.0 | _[Your result]_ | _[Your result]_ | Single image inference time |
| **Throughput (images/sec)** | 1000 / latency_ms | ~400 | ~125 | _[Your result]_ | _[Your result]_ | Images processed per second |
| **FPS** | Same as throughput | ~400 | ~125 | _[Your result]_ | _[Your result]_ | Frames per second |
| **Batch Throughput (batch=32)** | batch_size × 1000 / batch_latency_ms | ~2000 | ~500 | _[Your result]_ | _[Your result]_ | Maximum throughput with batching |
| **GPU Utilization (%)** | From nvidia-smi / rocm-smi | ~85% | ~90% | _[Your result]_ | _[Your result]_ | Average during inference |
| **Memory Bandwidth (GB/s)** | From nvidia-smi / rocm-smi | ~2.0 TB/s | ~320 GB/s | _[Your result]_ | _[Your result]_ | MI300X: ~5.3 TB/s, RX 7900 XTX: ~960 GB/s theoretical |
| **TFLOPS Utilized** | Calculated from operations | ~312 | ~65 | _[Your result]_ | _[Your result]_ | FP16 compute throughput |
| **Energy Efficiency (images/Wh)** | throughput / power_draw × 3600 | ~5760 | ~9000 | _[Your result]_ | _[Your result]_ | Higher is better |
| **Real-Time Factor (RTF)** | latency / (1000/30) | 0.075 | 0.24 | _[Your result]_ | _[Your result]_ | For 30 FPS video: <1.0 is faster than real-time |

---

## Model Variants and Multipliers

MobileNetV3 supports different width multipliers to trade off between accuracy and computational cost:

### MobileNetV3-Large Variants

| Multiplier | Parameters | MACs | Top-1 Accuracy | Use Case |
|------------|-----------|------|----------------|----------|
| **1.25x** | ~7.5M | 356M | 76.6% | High accuracy |
| **1.0x** (default) | 5.4M | 219M | 75.2% | Balanced |
| **0.75x** | ~4.0M | 155M | 73.3% | Efficient |

### MobileNetV3-Small Variants

| Multiplier | Parameters | MACs | Top-1 Accuracy | Use Case |
|------------|-----------|------|----------------|----------|
| **1.0x** (default) | 2.5M | 66M | 67.4% | Small baseline |
| **0.75x** | 2.0M | 44M | 65.4% | Ultra-efficient |
| **0.5x** | ~1.3M | ~25M | ~62% | Extreme resource constraint |
| **0.35x** | ~1.0M | ~15M | ~58% | Minimal footprint |

### Loading Different Variants in PyTorch

```python
from torchvision import models

# MobileNetV3-Large (default 1.0x)
model_large = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)

# MobileNetV3-Small (default 1.0x)
model_small = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

# For custom multipliers, you need to build from source or use timm
import timm
model_large_125 = timm.create_model('mobilenetv3_large_100', pretrained=True)
model_small_075 = timm.create_model('mobilenetv3_small_075', pretrained=True)
```

---

## Industrial Use Cases

MobileNetV3 is particularly well-suited for industrial computer vision applications:

### Quality Inspection
- **Defect detection** on manufacturing lines
- **Surface inspection** for scratches, dents, cracks
- **Assembly verification** to ensure correct component placement
- **Real-time classification** of product quality (pass/fail)

### Edge Deployment
- **Embedded vision systems** on industrial cameras
- **Robotic vision** for pick-and-place operations
- **Automated optical inspection (AOI)** systems
- **Mobile inspection devices** for field technicians

### Example: Industrial Defect Classification

```python
import torch
from torchvision import models, transforms
from PIL import Image

# Load MobileNetV3-Small for edge device
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

# Fine-tune for binary classification (defect/no-defect)
num_classes = 2  # defect, no-defect
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)

# Load fine-tuned weights (after training on your industrial dataset)
# model.load_state_dict(torch.load("defect_classifier.pth"))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# Inference on industrial image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open("product_image.jpg")
input_tensor = preprocess(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class].item()

class_names = ["No Defect", "Defect"]
print(f"Prediction: {class_names[predicted_class]}")
print(f"Confidence: {confidence:.2%}")
```

---

## Optimization Techniques for AMD GPUs

### Mixed Precision Inference

```python
import torch
from torchvision import models

model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
model = model.to("cuda").half()  # FP16 precision
model.eval()

# Use torch.autocast for automatic mixed precision
with torch.cuda.amp.autocast():
    with torch.no_grad():
        output = model(input_tensor.half())
```

### TorchScript Optimization

```python
# Convert to TorchScript for optimized inference
model.eval()
example_input = torch.randn(1, 3, 224, 224).to("cuda")
traced_model = torch.jit.trace(model, example_input)

# Save traced model
traced_model.save("mobilenetv3_large_traced.pt")

# Load and use
loaded_model = torch.jit.load("mobilenetv3_large_traced.pt")
with torch.no_grad():
    output = loaded_model(input_tensor)
```

### ONNX Export for ROCm

```python
import torch.onnx

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224).to("cuda")
torch.onnx.export(
    model,
    dummy_input,
    "mobilenetv3_large.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# Use with ONNX Runtime (ROCm backend)
import onnxruntime as ort

# Create ROCm execution provider session
providers = ['ROCMExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("mobilenetv3_large.onnx", providers=providers)

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
result = session.run([output_name], {input_name: input_data.numpy()})
```

---

## HuggingFace Timm Library

The [timm](https://github.com/huggingface/pytorch-image-models) library provides additional MobileNetV3 variants and pretrained weights:

### Available Models in Timm

```python
import timm

# List all MobileNetV3 models
models = timm.list_models('mobilenetv3*', pretrained=True)
print(models)
# Output: ['mobilenetv3_large_075', 'mobilenetv3_large_100',
#          'mobilenetv3_small_050', 'mobilenetv3_small_075',
#          'mobilenetv3_small_100', ...]

# Load with timm
model = timm.create_model('mobilenetv3_large_100', pretrained=True)
model.eval()

# Get model info
print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
```

### Timm Inference Example

```python
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image

# Load model
model = timm.create_model('mobilenetv3_large_100', pretrained=True)
model.eval()

# Get model-specific transforms
config = resolve_data_config({}, model=model)
transform = create_transform(**config)

# Inference
img = Image.open('image.jpg')
tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Get top-5 predictions
top5_prob, top5_idx = torch.topk(probabilities, 5)
for i in range(5):
    print(f"Class {top5_idx[i]}: {top5_prob[i]:.4f}")
```

---

## Additional Resources

### Official Repositories
- [TorchVision MobileNetV3 Implementation](https://pytorch.org/vision/main/models/mobilenetv3.html)
- [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models)
- [TensorFlow MobileNetV3](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)
- [Official Paper Implementation](https://github.com/d-li14/mobilenetv3.pytorch)

### Papers & Documentation
- [MobileNetV3 Paper (arXiv:1905.02244)](https://arxiv.org/abs/1905.02244)
- [MobileNetV3 Paper (PDF)](https://arxiv.org/pdf/1905.02244)
- [TorchVision MobileNetV3 Blog Post](https://pytorch.org/blog/torchvision-mobilenet-v3-implementation/)
- [MobileNetV4 Paper (arXiv:2404.10518)](https://arxiv.org/abs/2404.10518) - Latest evolution
- [MoGA: Searching Beyond MobileNetV3 (arXiv:1908.01314)](https://arxiv.org/abs/1908.01314)

### Benchmark Resources
- [Papers With Code - MobileNetV3](https://paperswithcode.com/method/mobilenetv3)
- [ImageNet Leaderboard](https://paperswithcode.com/sota/image-classification-on-imagenet)
- [MMPretrain MobileNetV3 Documentation](https://mmpretrain.readthedocs.io/en/latest/papers/mobilenet_v3.html)
- [HuggingFace MobileNet Baselines](https://huggingface.co/blog/rwightman/mobilenet-baselines)

### Optimization Guides
- [Optimizing Faster RCNN MobileNetV3 for Real-Time Inference](https://debuggercafe.com/optimizing-faster-rcnn-mobilenetv3-for-real-time-inference-on-cpu/)
- [AMD ROCm Performance Results](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html)
- [ONNX Runtime ROCm Execution Provider](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html)
- [PyTorch ROCm Installation Guide](https://pytorch.org/get-started/locally/)

### Datasets
- [ImageNet (HuggingFace)](https://huggingface.co/datasets/imagenet-1k)
- [COCO Dataset](https://cocodataset.org/)
- [CIFAR-10/100](https://huggingface.co/datasets/cifar10)
- [Label Studio - Computer Vision Benchmarks](https://labelstud.io/learningcenter/what-benchmarks-are-essential-for-evaluating-computer-vision-ai-systems/)

---

## Quick Reference Commands

```bash
# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Install additional libraries
pip install timm onnxruntime-rocm datasets

# Check AMD GPU status
rocm-smi
rocm-smi --showuse --showmeminfo vram
rocm-smi --showpower

# Download ImageNet validation set
python -c "from datasets import load_dataset; ds = load_dataset('imagenet-1k', split='validation[:100]')"

# Run benchmark script
python mobilenetv3_benchmark.py

# Export to ONNX
python -c "import torch; from torchvision import models; m = models.mobilenet_v3_large(weights='DEFAULT'); torch.onnx.export(m, torch.randn(1,3,224,224), 'model.onnx')"

# List available timm models
python -c "import timm; print('\n'.join(timm.list_models('mobilenetv3*', pretrained=True)))"
```

---

**Document Version:** 1.0
**Last Updated:** March 2026
**Target Hardware:** AMD MI300X, RX 7900 XTX, and other ROCm-compatible GPUs
**Model Segments:** Industrial, Healthcare
**Runtime Support:** ONNX-RT, RyzenAI Stack, PyTorch ROCm
