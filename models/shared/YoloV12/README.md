# YOLOv12 - Benchmark Guide for AMD GPU

## About the Model

YOLOv12 is an attention-centric real-time object detection model that represents the latest evolution in the YOLO (You Only Look Once) family. It introduces a novel attention-based architecture that matches the speed of previous CNN-based YOLO models while harnessing the performance benefits of attention mechanisms. The model leverages Area Attention to reduce computational complexity and Residual Efficient Layer Aggregation Networks (R-ELAN) to enhance feature aggregation, achieving state-of-the-art accuracy on standard benchmarks while maintaining real-time inference speeds.

### Original YOLOv12 Paper

**"YOLOv12: Attention-Centric Real-Time Object Detectors"** (Sun et al., 2025)

YOLOv12 proposes an attention-centric YOLO framework that achieves superior performance across multiple scales. The key innovation lies in the Area Attention mechanism, which reduces computational complexity from O(n²) to O(n√n) while maintaining strong representational capacity. Combined with R-ELAN for efficient feature aggregation, YOLOv12 achieves remarkable results: YOLOv12-N reaches 40.6% mAP with just 1.64 ms latency on a T4 GPU, outperforming YOLOv10-N and YOLOv11-N by 2.1% and 1.2% mAP respectively. The model comes in five scales (N, S, M, L, X) to accommodate different deployment scenarios from edge devices to datacenter GPUs.

**Paper:** [arXiv:2502.12524](https://arxiv.org/abs/2502.12524) | **Published:** February 2025 | **Conference:** NeurIPS 2025

---

## Standard Benchmark Dataset: MS COCO

**Microsoft COCO (Common Objects in Context)** is the industry-standard benchmark for evaluating object detection, segmentation, and captioning systems. It contains over 200,000 labeled images with annotations for 80 object categories.

### Dataset Structure
- **train2017**: 118,287 images with 860,001 bounding boxes
- **val2017**: 5,000 images with 36,781 bounding boxes
- **test-dev2017**: Test set for challenge submissions
- **80 object categories**: Person, vehicle, animal, and household object classes

### Download from HuggingFace

```bash
# Install dependencies
pip install datasets pycocotools
```

```python
from datasets import load_dataset

# Load COCO 2017 dataset (recommended)
dataset = load_dataset("rafaelpadilla/coco2017")

# Load training split
train_dataset = load_dataset("rafaelpadilla/coco2017", split="train")

# Load validation split
val_dataset = load_dataset("rafaelpadilla/coco2017", split="val")

# Alternative: Load with more options
import datasets as ds
dataset = ds.load_dataset(
    "shunk031/MSCOCO",
    year=2017,
    coco_task="instances",
    decode_rle=True  # Decode Run-length Encoding to binary mask
)

# View a sample
print(val_dataset[0])
# Output: {'image': <PIL.Image>, 'objects': {'bbox': [...], 'category': [...], ...}}
```

### Alternative Benchmark: Pascal VOC

```python
# Load Pascal VOC dataset
voc_dataset = load_dataset("detection-datasets/pascal_voc", split="train")
```

---

## Installation & Inference

### Install YOLOv12

```bash
# Clone the official repository
git clone https://github.com/sunsmarterjie/yolov12
cd yolov12

# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install roboflow supervision flash-attn --upgrade
pip install -r requirements.txt
pip install -e .

# Alternative: Using Ultralytics (if supported)
pip install ultralytics
```

### Basic Inference

```bash
# Using Ultralytics CLI
yolo detect predict model=yolo12n.pt source=image.jpg device=0

# Specify custom confidence threshold
yolo detect predict model=yolo12n.pt source=image.jpg conf=0.25 device=0

# Run on video
yolo detect predict model=yolo12n.pt source=video.mp4 device=0

# Save results
yolo detect predict model=yolo12n.pt source=image.jpg save=True project=runs/detect device=0
```

### Python API Inference

```python
from ultralytics import YOLO
import torch

# Check device availability
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load a COCO-pretrained YOLOv12n model
model = YOLO("yolo12n.pt")

# Run inference on a single image
results = model("path/to/image.jpg", device=device)

# Process results
for result in results:
    boxes = result.boxes  # Bounding boxes
    masks = result.masks  # Segmentation masks (if available)
    probs = result.probs  # Class probabilities

    # Print detections
    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        print(f"Class: {class_id}, Confidence: {confidence:.2f}, BBox: {bbox}")

# Display results
results[0].show()

# Save results
results[0].save("output.jpg")
```

### Batch Inference

```python
from ultralytics import YOLO
import cv2
import glob

model = YOLO("yolo12n.pt")

# Process multiple images
image_paths = glob.glob("images/*.jpg")
results = model(image_paths, device="cuda:0", batch=16)

# Process results
for i, result in enumerate(results):
    print(f"Image {i}: {len(result.boxes)} objects detected")
    result.save(f"output_{i}.jpg")
```

### Expected Output

```json
{
  "image_path": "path/to/image.jpg",
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.92,
      "bbox": [100, 150, 400, 600],
      "bbox_format": "xyxy"
    },
    {
      "class_id": 2,
      "class_name": "car",
      "confidence": 0.87,
      "bbox": [450, 300, 800, 550],
      "bbox_format": "xyxy"
    }
  ],
  "inference_time_ms": 1.64,
  "image_shape": [640, 640]
}
```

---

## Benchmark Results & Performance Metrics

### YOLOv12 Performance on MS COCO

| Model | mAP⁵⁰⁻⁹⁵ | mAP⁵⁰ | Params (M) | FLOPs (G) | Latency (ms) T4 | Dataset |
|-------|----------|-------|------------|-----------|-----------------|---------|
| **YOLOv12-N** | **40.6%** | **56.1%** | 2.8 | 6.5 | **1.64** | COCO val2017 |
| **YOLOv12-S** | **47.1%** | **64.2%** | 9.1 | 21.5 | **2.89** | COCO val2017 |
| **YOLOv12-M** | **51.8%** | **69.3%** | 25.3 | 68.7 | **5.41** | COCO val2017 |
| **YOLOv12-L** | **53.6%** | **71.2%** | 43.6 | 122.4 | **8.12** | COCO val2017 |
| **YOLOv12-X** | **54.9%** | **72.5%** | 56.9 | 157.8 | **11.23** | COCO val2017 |

**mAP** = mean Average Precision (higher is better)
**Latency** = Inference time on NVIDIA T4 GPU with TensorRT FP16

### Comparison with Other YOLO Versions

| Model | mAP⁵⁰⁻⁹⁵ | Params (M) | FLOPs (G) | Latency (ms) | Improvement |
|-------|----------|------------|-----------|--------------|-------------|
| YOLOv10-N | 38.5% | 2.3 | 6.7 | 1.58 | Baseline |
| YOLOv11-N | 39.4% | 2.6 | 6.5 | 1.62 | +0.9% mAP |
| **YOLOv12-N** | **40.6%** | 2.8 | 6.5 | 1.64 | **+2.1% mAP vs v10** |
| YOLOv10-S | 46.3% | 7.2 | 21.6 | 2.79 | Baseline |
| YOLOv11-S | 47.0% | 9.4 | 21.5 | 2.84 | +0.7% mAP |
| **YOLOv12-S** | **47.1%** | 9.1 | 21.5 | 2.89 | **+0.8% mAP vs v10** |
| RT-DETR-R18 | 46.5% | 20 | 60 | 4.58 | Baseline |
| RT-DETRv2-R18 | 48.3% | 16 | 55 | 5.01 | +1.8% mAP |
| **YOLOv12-S** | **47.1%** | **9.1** | **21.5** | **2.89** | **42% faster, 64% params** |

### Performance: YOLO Family Evolution

| Implementation | mAP⁵⁰⁻⁹⁵ | Speed | Key Innovation | Notes |
|----------------|----------|-------|----------------|-------|
| **YOLOv12** | **40.6%-54.9%** | **1.64-11.23 ms** | Area Attention + R-ELAN | Attention-centric architecture |
| YOLOv11 | 39.4%-54.7% | 1.62-11.16 ms | Improved C3k2, SPPF | Enhanced feature extraction |
| YOLOv10 | 38.5%-54.4% | 1.58-10.70 ms | NMS-free training | End-to-end detection |
| YOLOv9 | 39.3%-53.9% | 1.70-11.60 ms | GELAN, PGI | Programmable gradient info |
| YOLOv8 | 37.3%-53.9% | 1.47-10.46 ms | C2f module | Anchor-free detection |
| YOLOv5 | 28.0%-50.7% | 1.21-9.20 ms | CSPNet backbone | Widely adopted baseline |

**Benchmark:** MS COCO val2017, NVIDIA T4 GPU with TensorRT FP16 precision

---

## AMD GPU Benchmarking Setup

### ROCm Installation for AMD GPUs

```bash
# Check ROCm compatibility
rocm-smi

# Install PyTorch with ROCm support (ROCm 6.2)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Check ROCm version
python -c "import torch; print(f'ROCm Version: {torch.version.hip}')"
```

### AMD GPU Environment Setup

```bash
# Set environment variables for AMD integrated GPUs (required for stability)
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # Adjust based on your GPU
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# For better performance
export PYTORCH_ROCM_ARCH=gfx1100  # Adjust for your architecture
export GPU_MAX_HEAP_SIZE=100
export GPU_MAX_ALLOC_PERCENT=100
```

### Benchmark Script for AMD GPU

```python
import torch
import time
from datasets import load_dataset
from ultralytics import YOLO
import numpy as np
import subprocess

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"ROCm Version: {torch.version.hip}")

# Load YOLOv12 model
model = YOLO("yolo12n.pt")
model.to(device)

# Load COCO validation dataset
print("Loading COCO dataset...")
dataset = load_dataset("rafaelpadilla/coco2017", split="val[:100]")

# Warmup
print("Warming up GPU...")
for i in range(10):
    dummy_results = model(dataset[0]["image"], device=device, verbose=False)

# Benchmark
print("Starting benchmark...")
results = []
total_objects = 0

for i, sample in enumerate(dataset):
    # Get image
    image = sample["image"]

    # Measure inference time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    detection_results = model(image, device=device, verbose=False)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()

    inference_time_ms = (end_time - start_time) * 1000
    num_detections = len(detection_results[0].boxes)
    total_objects += num_detections

    results.append({
        "sample_id": i,
        "inference_time_ms": inference_time_ms,
        "num_detections": num_detections,
    })

    if (i + 1) % 10 == 0:
        print(f"Processed {i+1}/{len(dataset)} images, "
              f"Avg time: {np.mean([r['inference_time_ms'] for r in results]):.2f}ms")

# Summary statistics
avg_inference_time = np.mean([r["inference_time_ms"] for r in results])
min_inference_time = np.min([r["inference_time_ms"] for r in results])
max_inference_time = np.max([r["inference_time_ms"] for r in results])
std_inference_time = np.std([r["inference_time_ms"] for r in results])
fps = 1000 / avg_inference_time

print("\n" + "="*60)
print("BENCHMARK RESULTS")
print("="*60)
print(f"Model: YOLOv12-N")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Images processed: {len(results)}")
print(f"Total objects detected: {total_objects}")
print(f"Average inference time: {avg_inference_time:.2f} ms")
print(f"Min inference time: {min_inference_time:.2f} ms")
print(f"Max inference time: {max_inference_time:.2f} ms")
print(f"Std deviation: {std_inference_time:.2f} ms")
print(f"Throughput: {fps:.2f} FPS")

# Memory statistics
if torch.cuda.is_available():
    print(f"\nMemory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"Memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
```

### Performance Metrics Table Template

| Metric | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Notes |
|--------|------------------|-----------|------------|-----------------|-------|
| **GPU Model** | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Compare datacenter vs consumer GPUs |
| **Memory (GB)** | 80 | 16 | 192 | 24 | VRAM capacity |
| **TDP (W)** | 400 | 70 | 750 | 355 | Thermal design power |
| **Model Size** | YOLOv12-N | YOLOv12-N | _[Your result]_ | _[Your result]_ | Model variant tested |
| **Inference Time (ms)** | ~1.2 | 1.64 | _[Your result]_ | _[Your result]_ | Lower is better |
| **Throughput (FPS)** | ~833 | 610 | _[Your result]_ | _[Your result]_ | Frames per second |
| **Batch Size** | 1 | 1 | _[Your result]_ | _[Your result]_ | Images processed simultaneously |
| **mAP⁵⁰⁻⁹⁵** | 40.6% | 40.6% | _[Your result]_ | _[Your result]_ | Accuracy on COCO val2017 |
| **Peak Memory Usage (GB)** | ~15 | ~8 | _[Your result]_ | _[Your result]_ | Monitor with rocm-smi |
| **Average Power Draw (W)** | ~250 | ~55 | _[Your result]_ | _[Your result]_ | Monitor with rocm-smi --showpower |
| **Energy per 1000 Images (Wh)** | ~0.08 | ~0.09 | _[Your result]_ | _[Your result]_ | Lower is better |

### AMD-Specific Metrics to Track

```python
# GPU utilization tracking
import subprocess

def get_rocm_smi_stats():
    """Get AMD GPU statistics using rocm-smi"""
    result = subprocess.run(['rocm-smi', '--showuse', '--showmeminfo', 'vram', '--showpower'],
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

# Print rocm-smi stats during benchmark
print("\nGPU Statistics:")
print(get_rocm_smi_stats())
```

### Complete Runtime Metrics Table

| Runtime Metric | Formula | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Notes |
|----------------|---------|------------------|-----------|------------|-----------------|-------|
| **Throughput (FPS)** | 1000 / inference_time_ms | 833 | 610 | _[Your result]_ | _[Your result]_ | Higher is better |
| **Latency (ms)** | Time per image | 1.2 | 1.64 | _[Your result]_ | _[Your result]_ | Lower is better |
| **Objects Per Second** | detections × FPS | _[Reference]_ | _[Reference]_ | _[Your result]_ | _[Your result]_ | Detection throughput |
| **GPU Utilization (%)** | From nvidia-smi / rocm-smi | ~95% | ~95% | _[Your result]_ | _[Your result]_ | Average during inference |
| **Memory Bandwidth (GB/s)** | From nvidia-smi / rocm-smi | ~2.0 TB/s | ~320 GB/s | _[Your result]_ | _[Your result]_ | MI300X: ~5.3 TB/s, RX 7900 XTX: ~960 GB/s theoretical |
| **TFLOPS Utilized** | Calculated from operations | _[Reference]_ | _[Reference]_ | _[Your result]_ | _[Your result]_ | FP16 compute throughput |
| **Batch Processing (imgs/s)** | batch_size / time | _[Reference]_ | _[Reference]_ | _[Your result]_ | _[Your result]_ | With batch size > 1 |
| **Energy Efficiency (imgs/Wh)** | (3600 × FPS) / power_draw | _[Reference]_ | _[Reference]_ | _[Your result]_ | _[Your result]_ | Higher is better |

---

## Computer Vision Model Leaderboard

The [Roboflow Computer Vision Leaderboard](https://leaderboard.roboflow.com/) tracks performance of object detection models across multiple benchmarks:

### Evaluation Datasets
- **MS COCO** (Common Objects in Context) - 80 classes
- **Objects365** - Large-scale detection with 365 classes
- **LVIS** - Long-tail object detection with 1,203 classes
- **Pascal VOC** - Classic benchmark with 20 classes
- **Open Images** - 600 object classes

### Key Metrics Tracked
- **mAP⁵⁰⁻⁹⁵** (mean Average Precision) - primary metric
- **mAP⁵⁰** (mAP at IoU threshold 0.5)
- **Inference Speed** (FPS, latency in ms)
- **Model Size** (parameters, FLOPs)
- **Hardware Efficiency** (performance per watt)

### Top Performers (2026)
| Model | mAP⁵⁰⁻⁹⁵ | Latency (ms) | Status |
|-------|----------|--------------|--------|
| RF-DETR-Medium | 54.7% | 4.52 | SOTA end-to-end |
| **YOLOv12-X** | **54.9%** | **11.23** | **Latest YOLO** |
| YOLOv11-X | 54.7% | 11.16 | Previous gen |
| YOLOv10-X | 54.4% | 10.70 | Previous gen |
| YOLO11 | Similar to YOLOv5 | Variable | Ultralytics |

**Note:** YOLOv12 represents the cutting edge of the YOLO family as of early 2025, with attention-based architecture improvements.

---

## Training YOLOv12 on Custom Dataset

### Prepare Custom Dataset

```bash
# Dataset structure (YOLO format)
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/

# Each label file contains: class_id x_center y_center width height (normalized)
```

### Training Script

```python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yolo12n.pt")

# Train on custom dataset
results = model.train(
    data="custom_dataset.yaml",  # Path to dataset config
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # GPU device
    workers=8,
    patience=50,
    save=True,
    project="runs/train",
    name="yolov12_custom"
)

# Validate
metrics = model.val()
print(f"mAP50-95: {metrics.box.map}")
print(f"mAP50: {metrics.box.map50}")
```

### Dataset Configuration (custom_dataset.yaml)

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test  # optional

# Classes
nc: 80  # number of classes
names: ['person', 'bicycle', 'car', ...]  # class names
```

---

## Additional Resources

### Official Repositories
- [YOLOv12 GitHub](https://github.com/sunsmarterjie/yolov12)
- [Ultralytics YOLOv12 Documentation](https://docs.ultralytics.com/models/yolo12/)
- [Ultralytics YOLO (General)](https://github.com/ultralytics/ultralytics)

### Papers & Documentation
- [YOLOv12 Paper (arXiv:2502.12524)](https://arxiv.org/abs/2502.12524)
- [YOLOv12 HTML Version](https://arxiv.org/html/2502.12524v1)
- [YOLO Evolution Overview (arXiv:2510.09653)](https://arxiv.org/html/2510.09653v2)
- [Roboflow Computer Vision Leaderboard](https://leaderboard.roboflow.com/)

### Blog Posts & Comparisons
- [Best Object Detection Models 2025: RF-DETR, YOLOv12 & Beyond](https://blog.roboflow.com/best-object-detection-models/)
- [How to Train YOLOv12 on Custom Dataset](https://blog.roboflow.com/train-yolov12-model/)
- [YOLOv12: Object Detection meets Attention](https://learnopencv.com/yolov12/)
- [YOLOv12 Tutorial from Beginners to Experts](https://medium.com/@zainshariff6506/a-simple-yolov12-tutorial-from-beginners-to-experts-e6e518c3daf4)
- [AMD ROCm: YOLO Installation Guide](https://phazertech.com/tutorials/rocm.html)

### AMD GPU Resources
- [AMD GPU YOLOv8 Integration](https://github.com/harakas/amd_igpu_yolo_v8)
- [Deploying Object Detection on AMD AI PC with NPU](https://www.amd.com/en/developer/resources/technical-articles/2026/deploying-object-detection-model-on-amd-ai-pc.html)
- [AMD ROCm Hub - AI Performance Results](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html)

### Datasets
- [MS COCO 2017 (rafaelpadilla/coco2017)](https://huggingface.co/datasets/rafaelpadilla/coco2017)
- [MS COCO (shunk031/MSCOCO)](https://huggingface.co/datasets/shunk031/MSCOCO)
- [COCO Detection Datasets](https://huggingface.co/datasets/detection-datasets/coco)
- [COCO8 (Small Sample)](https://huggingface.co/datasets/Ultralytics/COCO8)
- [Pascal VOC](https://huggingface.co/datasets/detection-datasets/pascal_voc)

---

## Quick Reference Commands

```bash
# Install YOLOv12
git clone https://github.com/sunsmarterjie/yolov12
cd yolov12
pip install -r requirements.txt
pip install -e .

# Install PyTorch with ROCm (AMD GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Run inference on single image
yolo detect predict model=yolo12n.pt source=image.jpg device=0

# Run inference on video
yolo detect predict model=yolo12n.pt source=video.mp4 device=0

# Check AMD GPU status
rocm-smi
rocm-smi --showuse --showmeminfo vram --showpower

# Download COCO dataset
python -c "from datasets import load_dataset; ds = load_dataset('rafaelpadilla/coco2017')"

# Train on custom dataset
yolo detect train data=custom.yaml model=yolo12n.pt epochs=100 imgsz=640 device=0

# Validate model
yolo detect val model=yolo12n.pt data=coco.yaml device=0

# Export model (ONNX, TensorRT, etc.)
yolo export model=yolo12n.pt format=onnx
```

---

**Document Version:** 1.0
**Last Updated:** March 2026
**Target Hardware:** AMD MI300X, RX 7900 XTX, and other ROCm-compatible GPUs
