# MobileSAM - Benchmark Guide for AMD GPU

**Navigation:** [🏠 Home](/) | [📑 Models Index](/MODELS_INDEX) | [📝 Contributing](/CONTRIBUTING)

---

## About the Model

MobileSAM (Mobile Segment Anything Model) is a highly optimized implementation of Meta's Segment Anything Model (SAM) designed specifically for mobile and edge devices. It replaces SAM's heavy ViT-H encoder (632M parameters) with a lightweight Tiny-ViT encoder (5M parameters), achieving approximately 60× reduction in model size while maintaining competitive segmentation quality. MobileSAM can process images in just 10-12ms on GPU, making it 5-7× faster than the original SAM and suitable for real-time applications on resource-constrained devices.

### Original SAM Paper

**"Segment Anything"** (Kirillov et al., 2023)

The Segment Anything (SA) project introduces a new task, model, and dataset for image segmentation. Using an efficient model in a data collection loop, Meta AI built the largest segmentation dataset to date with over 1 billion masks on 11 million licensed and privacy-respecting images. SAM enables zero-shot, promptable object segmentation with remarkable generalization capabilities. The model can be prompted with points, boxes, masks, or text to segment any object in an image without fine-tuning.

**Paper:** [arXiv:2304.02643](https://arxiv.org/abs/2304.02643) | **Published:** ICCV 2023

### MobileSAM Paper

**"Faster Segment Anything: Towards Lightweight SAM for Mobile Applications"** (Zhang et al., 2023)

MobileSAM addresses the computational challenges of deploying SAM on mobile devices through a novel decoupled knowledge distillation approach. The model distills knowledge from SAM's heavy image encoder to a lightweight Tiny-ViT encoder while keeping the original prompt encoder and mask decoder unchanged. This ensures full compatibility with SAM's pipeline while achieving significant efficiency gains. MobileSAM was trained on a single GPU using only 100k images (1% of the original dataset) in less than one day.

**Paper:** [arXiv:2306.14289](https://arxiv.org/abs/2306.14289) | **Published:** 2023

---

## Standard Benchmark Datasets

### SA-1B Dataset

**SA-1B** (Segment Anything 1-Billion) is the largest segmentation dataset ever created, containing 11 million images and over 1.1 billion high-quality segmentation masks. It serves as the primary training dataset for SAM and the evaluation benchmark for promptable segmentation models.

#### Dataset Structure
- **Images**: 11 million diverse, high-resolution images
- **Masks**: 1.1 billion segmentation masks
- **Annotations**: Multiple masks per image with varying levels of granularity
- **Quality**: Human audit of 500 images (~50,000 masks) showed 94% with IoU >0.90, 97% with IoU >0.75

#### Download from Meta AI

```bash
# SA-1B dataset is available at Meta AI's official website
# Visit: https://ai.meta.com/datasets/segment-anything/
```

### COCO Dataset

**COCO** (Common Objects in Context) is the industry-standard benchmark for object detection and segmentation. It contains 330K images with detailed annotations for 80 object categories, making it ideal for evaluating zero-shot segmentation performance.

#### Dataset Structure
- **train2017**: 118K images for training
- **val2017**: 5K images for validation
- **test-dev**: Development test set
- **80 object categories**: Person, vehicle, animal, furniture, etc.

#### Download from HuggingFace

```bash
# Install dependencies
pip install datasets transformers
```

```python
from datasets import load_dataset

# Load COCO dataset
dataset = load_dataset("detection-datasets/coco", trust_remote_code=True)

# Or use specific split
coco_val = load_dataset("detection-datasets/coco", split="validation")

# View a sample
print(coco_val[0])
# Output: {'image': <PIL.Image>, 'objects': {'bbox': [...], 'category': [...]}}
```

---

## Installation & Inference

### Install MobileSAM

```bash
# Install from official GitHub repository
git clone https://github.com/ChaoningZhang/MobileSAM.git
cd MobileSAM

# Install dependencies
pip install -r requirements.txt

# Or install from HuggingFace implementation
git clone -b add_mixin https://github.com/NielsRogge/MobileSAM.git
cd MobileSAM
pip install -e .
```

### Download Model Weights

```bash
# Create weights directory
mkdir -p weights

# Download MobileSAM checkpoint
wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt -O weights/mobile_sam.pt

# Or download via Python from HuggingFace
python -c "from mobile_sam import MobileSAM; model = MobileSAM.from_pretrained('nielsr/mobilesam')"
```

### Basic Inference - Point Prompt

```python
import torch
import numpy as np
from PIL import Image
from mobile_sam import sam_model_registry, SamPredictor

# Load model
model_type = "vit_t"
sam_checkpoint = "weights/mobile_sam.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

# Initialize predictor
predictor = SamPredictor(mobile_sam)

# Load and set image
image = Image.open("example.jpg")
image = np.array(image.convert("RGB"))
predictor.set_image(image)

# Segment with point prompt
input_point = np.array([[500, 375]])  # [x, y] coordinates
input_label = np.array([1])  # 1 = foreground, 0 = background

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

# masks: (num_masks, H, W) boolean array
# scores: confidence scores for each mask
# Use the mask with highest score
best_mask = masks[np.argmax(scores)]
```

### Basic Inference - Box Prompt

```python
# Segment with bounding box prompt
input_box = np.array([425, 600, 700, 875])  # [x1, y1, x2, y2]

masks, scores, logits = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
)
```

### Batch Processing Multiple Images

```python
import glob
import time

# Process directory of images
image_paths = glob.glob("images/*.jpg")
results = []

for img_path in image_paths:
    image = Image.open(img_path)
    image = np.array(image.convert("RGB"))

    start_time = time.time()
    predictor.set_image(image)

    # Example: segment object at center
    h, w = image.shape[:2]
    center_point = np.array([[w // 2, h // 2]])
    center_label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=center_point,
        point_labels=center_label,
        multimask_output=True,
    )

    inference_time = time.time() - start_time

    results.append({
        "image": img_path,
        "inference_time": inference_time,
        "mask": masks[np.argmax(scores)],
        "score": float(np.max(scores))
    })

    print(f"Processed {img_path}: {inference_time*1000:.2f}ms, Score: {np.max(scores):.3f}")
```

### HuggingFace API Inference

```python
from mobile_sam import MobileSAM, SamPredictor
import torch
from PIL import Image
import numpy as np

# Load from HuggingFace Hub
model = MobileSAM.from_pretrained("nielsr/mobilesam")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device=device)

# Initialize predictor
predictor = SamPredictor(model)

# Load image
image = Image.open("example.jpg")
image_array = np.array(image)

# Set image
predictor.set_image(image_array)

# Predict with point
point = np.array([[500, 375]])
label = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=point,
    point_labels=label,
    multimask_output=True,
)

print(f"Generated {len(masks)} masks with scores: {scores}")
```

### Expected Output

```python
# Masks shape: (3, H, W) for multimask_output=True
# Each mask is a boolean array indicating segmented pixels
# Scores: [0.95, 0.89, 0.76] - confidence for each mask
# Logits: (3, 256, 256) - raw output logits

# Example visualization
import matplotlib.pyplot as plt

def show_mask(mask, ax, color=None):
    if color is None:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# Display best mask
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(image)
show_mask(masks[np.argmax(scores)], ax)
plt.axis('off')
plt.show()
```

---

## Benchmark Results & Performance Metrics

### MobileSAM Performance on COCO (Zero-Shot)

| Model | Encoder Size | Total Params | mIoU (1-box) | mIoU (1-point) | Model Size (MB) | Inference Time (ms/image) |
|-------|-------------|--------------|--------------|----------------|-----------------|---------------------------|
| **SAM (ViT-H)** | 632M | 641M | 77.3 | - | ~2,400 | 456 |
| **SAM (ViT-L)** | 308M | 312M | 76.0 | - | ~1,200 | 233 |
| **SAM (ViT-B)** | 89M | 94M | 73.2 | - | ~375 | 131 |
| **MobileSAM (ViT-Tiny)** | 5M | 9.66M | 74.5 | 72.8 | ~39 | 10-12 |
| **FastSAM** | - | ~68M | 69.3 | - | ~270 | 40-50 |
| **EfficientSAM-Ti** | - | ~10M | 76.4 | 74.3 | ~40 | 15-20 |
| **EdgeSAM** | - | ~9.5M | 75.7 | 73.9 | ~38 | 13-15 |

**mIoU** = mean Intersection over Union (higher is better, 0-100 scale)

**Key Findings:**
- MobileSAM achieves only ~2.8 points lower mIoU than SAM ViT-H while being 66× smaller
- Outperforms FastSAM by 5.2 mIoU points while being 7× smaller
- Inference time reduced from 456ms to 10-12ms (38× faster than ViT-H)
- Nearly matches SAM ViT-B performance with 10× fewer parameters

### Performance: MobileSAM vs Alternatives

| Implementation | Parameters | Model Size | Relative Speed | Relative Size | Platform Support |
|----------------|-----------|------------|----------------|---------------|------------------|
| **MobileSAM** | 9.66M | 39 MB | **38× faster than SAM-H** | **66× smaller** | GPU, CPU, Mobile |
| SAM ViT-H | 641M | 2,400 MB | 1× baseline | 1× baseline | GPU (high-end) |
| SAM ViT-L | 312M | 1,200 MB | ~2× faster | ~2× smaller | GPU |
| SAM ViT-B | 94M | 375 MB | ~3.5× faster | ~6× smaller | GPU, CPU |
| FastSAM | 68M | 270 MB | ~9× faster | ~9× smaller | GPU |
| EfficientSAM-Ti | 10M | 40 MB | ~30× faster | ~60× smaller | GPU, CPU, Mobile |

**Benchmark Context:** Inference times measured on single GPU (NVIDIA T4 equivalent)

### Encoder Architecture Comparison

| Encoder | Parameters | Input Size | Embed Dim | Depth | Inference Time |
|---------|-----------|------------|-----------|-------|----------------|
| **ViT-H (SAM)** | 632M | 1024×1024 | 1280 | 32 layers | ~450ms |
| **ViT-L (SAM)** | 308M | 1024×1024 | 1024 | 24 layers | ~230ms |
| **ViT-B (SAM)** | 89M | 1024×1024 | 768 | 12 layers | ~125ms |
| **Tiny-ViT (MobileSAM)** | 5.78M | 1024×1024 | 384-576 | 4 stages | ~8ms |

---

## AMD GPU Benchmarking Setup

### ROCm Installation for AMD GPUs

```bash
# Check ROCm compatibility
rocm-smi

# Install PyTorch with ROCm support (ROCm 6.2+)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Check ROCm version
python -c "import torch; print(f'ROCm Version: {torch.version.hip}')"
```

### Benchmark Script for AMD GPU

```python
import torch
import numpy as np
import time
from PIL import Image
from datasets import load_dataset
from mobile_sam import sam_model_registry, SamPredictor

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"ROCm Version: {torch.version.hip if torch.cuda.is_available() else 'N/A'}")

# Load MobileSAM model
model_type = "vit_t"
sam_checkpoint = "weights/mobile_sam.pt"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

# Initialize predictor
predictor = SamPredictor(mobile_sam)

# Load COCO validation dataset
print("Loading COCO validation dataset...")
dataset = load_dataset("detection-datasets/coco", split="validation[:100]", trust_remote_code=True)

# Benchmark metrics
results = []
total_inference_time = 0
total_images = 0

for idx, sample in enumerate(dataset):
    image = np.array(sample["image"].convert("RGB"))
    h, w = image.shape[:2]

    # Warmup on first image
    if idx == 0:
        predictor.set_image(image)
        center_point = np.array([[w // 2, h // 2]])
        center_label = np.array([1])
        _ = predictor.predict(point_coords=center_point, point_labels=center_label, multimask_output=True)
        continue

    # Benchmark inference
    start_time = time.time()

    # Set image (includes encoder forward pass)
    predictor.set_image(image)

    # Get center point as prompt
    center_point = np.array([[w // 2, h // 2]])
    center_label = np.array([1])

    # Predict mask
    masks, scores, logits = predictor.predict(
        point_coords=center_point,
        point_labels=center_label,
        multimask_output=True,
    )

    # Synchronize GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()
    inference_time = end_time - start_time

    # Calculate metrics
    best_mask_idx = np.argmax(scores)
    best_score = scores[best_mask_idx]

    results.append({
        "image_id": idx,
        "inference_time": inference_time,
        "mask_score": float(best_score),
        "image_size": (h, w)
    })

    total_inference_time += inference_time
    total_images += 1

    print(f"Image {idx}: {inference_time*1000:.2f}ms, Score: {best_score:.3f}, Size: {w}x{h}")

# Summary statistics
avg_inference_time = total_inference_time / total_images
images_per_second = 1 / avg_inference_time
avg_score = np.mean([r["mask_score"] for r in results])

print(f"\n{'='*60}")
print(f"BENCHMARK SUMMARY")
print(f"{'='*60}")
print(f"Total images processed: {total_images}")
print(f"Average inference time: {avg_inference_time*1000:.2f}ms")
print(f"Throughput: {images_per_second:.2f} images/second ({images_per_second*60:.0f} images/min)")
print(f"Average mask confidence: {avg_score:.3f}")
print(f"Min inference time: {min([r['inference_time'] for r in results])*1000:.2f}ms")
print(f"Max inference time: {max([r['inference_time'] for r in results])*1000:.2f}ms")
```

### mIoU Calculation with Ground Truth

```python
import torch
import numpy as np
from datasets import load_dataset
from mobile_sam import sam_model_registry, SamPredictor

def calculate_iou(pred_mask, gt_mask):
    """Calculate Intersection over Union between predicted and ground truth masks"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    if union == 0:
        return 0.0

    return intersection / union

# Load model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
mobile_sam = sam_model_registry["vit_t"](checkpoint="weights/mobile_sam.pt")
mobile_sam.to(device=device)
mobile_sam.eval()
predictor = SamPredictor(mobile_sam)

# Load COCO dataset with annotations
dataset = load_dataset("detection-datasets/coco", split="validation[:100]", trust_remote_code=True)

iou_scores = []

for sample in dataset:
    image = np.array(sample["image"].convert("RGB"))
    predictor.set_image(image)

    # Process each object annotation
    if "objects" in sample and sample["objects"]["bbox"]:
        for bbox in sample["objects"]["bbox"]:
            # bbox format: [x, y, width, height] (COCO format)
            x, y, w, h = bbox
            box_prompt = np.array([x, y, x + w, y + h])

            # Predict mask with box prompt
            masks, scores, _ = predictor.predict(
                box=box_prompt[None, :],
                multimask_output=False,
            )

            # Get ground truth mask (if available)
            # Note: This is simplified - actual COCO annotations need decoding
            # For real benchmarks, use pycocotools to decode segmentation masks

            pred_mask = masks[0]
            # iou = calculate_iou(pred_mask, gt_mask)
            # iou_scores.append(iou)

# Calculate mean IoU
# mean_iou = np.mean(iou_scores)
# print(f"Mean IoU: {mean_iou:.4f}")
```

### Performance Metrics Table Template

| Metric | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Notes |
|--------|------------------|-----------|------------|-----------------|-------|
| **GPU Model** | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Compare datacenter vs consumer GPUs |
| **Memory (GB)** | 80 | 16 | 192 | 24 | VRAM capacity |
| **TDP (W)** | 400 | 70 | 750 | 355 | Thermal design power |
| **Batch Size** | 100 | 100 | _[Your result]_ | _[Your result]_ | Images processed in sequence |
| **Avg Inference Time (ms/image)** | ~8-10 | ~10-12 | _[Your result]_ | _[Your result]_ | Lower is better |
| **Throughput (images/sec)** | 100-125 | 83-100 | _[Your result]_ | _[Your result]_ | Higher is better |
| **Throughput (images/min)** | 6,000-7,500 | 5,000-6,000 | _[Your result]_ | _[Your result]_ | Higher is better |
| **Peak Memory Usage (GB)** | ~2-3 | ~1.5-2 | _[Your result]_ | _[Your result]_ | Monitor with rocm-smi |
| **Average Power Draw (W)** | ~150-200 | ~50-60 | _[Your result]_ | _[Your result]_ | Monitor with rocm-smi --showpower |
| **Energy per 1K Images (Wh)** | ~1.5-2 | ~0.6-1 | _[Your result]_ | _[Your result]_ | Lower is better |
| **mIoU (COCO val, 1-box)** | 74.5 | 74.5 | _[Your result]_ | _[Your result]_ | Should match across GPUs |

### AMD-Specific Metrics to Track

```python
import subprocess
import torch

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
        print("\nPower Draw:")
        print(result.stdout)

    except FileNotFoundError:
        print("rocm-smi not found. Please ensure ROCm is installed.")

# PyTorch memory tracking
if torch.cuda.is_available():
    print(f"\nPyTorch CUDA Memory Stats:")
    print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    # ROCm info
    print(f"\nROCm Info:")
    print(f"ROCm Version: {torch.version.hip}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Device Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

# Call during benchmarking
get_rocm_smi_stats()
```

### Complete Runtime Metrics Table

| Runtime Metric | Formula | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Notes |
|----------------|---------|------------------|-----------|------------|-----------------|-------|
| **Inference Time (ms)** | Time per image | 8-10 | 10-12 | _[Your result]_ | _[Your result]_ | Lower is better |
| **Throughput (img/sec)** | 1000 / inference_time | 100-125 | 83-100 | _[Your result]_ | _[Your result]_ | Higher is better |
| **GPU Utilization (%)** | From nvidia-smi / rocm-smi | 85-95% | 80-90% | _[Your result]_ | _[Your result]_ | During inference |
| **Memory Bandwidth (GB/s)** | From nvidia-smi / rocm-smi | ~2.0 TB/s | ~320 GB/s | _[Your result]_ | _[Your result]_ | MI300X: ~5.3 TB/s, RX 7900 XTX: ~960 GB/s theoretical |
| **TFLOPS Utilized** | Calculated from operations | ~50-80 | ~15-25 | _[Your result]_ | _[Your result]_ | FP16/BF16 compute throughput |
| **Encoder Time (ms)** | Image encoder forward pass | 6-8 | 8-10 | _[Your result]_ | _[Your result]_ | ~80% of total time |
| **Decoder Time (ms)** | Mask decoder forward pass | 2-4 | 2-4 | _[Your result]_ | _[Your result]_ | ~20% of total time |
| **Energy Efficiency (Wh/1K img)** | power_draw × time / 1000 | 1.5-2 | 0.6-1 | _[Your result]_ | _[Your result]_ | Lower is better |

---

## Segmentation Leaderboards & Benchmarks

### Common Evaluation Metrics

Segmentation models are typically evaluated using the following metrics:

- **mIoU** (mean Intersection over Union) - Primary metric for segmentation quality
- **Pixel Accuracy** - Percentage of correctly classified pixels
- **Boundary F1** - Precision and recall at object boundaries
- **Inference Time** - Time to process a single image (ms)
- **Model Size** - Total parameters and disk size (MB)

### Zero-Shot Segmentation Performance

MobileSAM excels in zero-shot segmentation tasks where the model segments objects without task-specific training:

| Dataset | Task | SAM (ViT-H) mIoU | MobileSAM mIoU | Performance Gap |
|---------|------|------------------|----------------|-----------------|
| **COCO val** | 1-box prompt | 77.3 | 74.5 | -2.8 points |
| **COCO val** | 1-point prompt | - | 72.8 | - |
| **SA-1B subset** | Multi-prompt | ~78-80 | 76-78 | ~2-3 points |
| **LVIS v1** | Instance segmentation | 74.2 | ~71-72 | ~2-3 points |

### Promptable Segmentation Benchmark

Different prompt types yield varying segmentation quality:

| Prompt Type | Description | SAM mIoU | MobileSAM mIoU | Use Case |
|-------------|-------------|----------|----------------|----------|
| **1 Box** | Bounding box around object | 77.3 | 74.5 | When object detection exists |
| **1 Point (center)** | Single center point | ~75 | 72.8 | Quick interaction |
| **5 Points** | Multiple foreground points | ~78 | 76.2 | More accurate segmentation |
| **Point + Box** | Combined prompts | ~79 | 77.0 | Maximum accuracy |
| **Mask** | Previous mask refinement | ~80 | 78.5 | Iterative refinement |

### Mobile & Edge Device Benchmarks

Performance on resource-constrained devices:

| Device | CPU/GPU | MobileSAM Time | SAM ViT-B Time | Speedup | Notes |
|--------|---------|----------------|----------------|---------|-------|
| **NVIDIA Jetson Orin** | GPU | ~15-20ms | ~150-200ms | 8-10× | Edge AI platform |
| **NVIDIA Jetson Xavier NX** | GPU | ~30-40ms | ~300-400ms | 8-10× | Embedded device |
| **Qualcomm Snapdragon 8 Gen 2** | NPU | ~100-150ms | Not feasible | - | Mobile phone |
| **Apple M1** | Neural Engine | ~50-80ms | ~400-600ms | 6-8× | Mac/iPad |
| **Intel i5 (CPU only)** | CPU | ~3000ms | >10,000ms | 3-4× | No GPU acceleration |
| **ARM Cortex-A78 (CPU)** | CPU | ~200-300ms | Not feasible | - | Mobile CPU |

---

## Additional Resources

### Official Repositories

- [MobileSAM GitHub](https://github.com/ChaoningZhang/MobileSAM) - Official implementation
- [MobileSAM HuggingFace (nielsr)](https://huggingface.co/nielsr/mobilesam) - PyTorch weights
- [MobileSAM HuggingFace (qualcomm)](https://huggingface.co/qualcomm/MobileSam) - Optimized for Qualcomm devices
- [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything) - Original SAM
- [Ultralytics MobileSAM](https://docs.ultralytics.com/models/mobile-sam/) - Ultralytics implementation

### Papers & Documentation

- [MobileSAM Paper (arXiv:2306.14289)](https://arxiv.org/abs/2306.14289)
- [MobileSAM Paper (PDF)](https://arxiv.org/pdf/2306.14289)
- [SAM Paper (arXiv:2304.02643)](https://arxiv.org/abs/2304.02643)
- [SAM Paper (PDF)](https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf)
- [MobileSAMv2 Paper (arXiv:2312.09579)](https://arxiv.org/abs/2312.09579) - Faster "segment everything"
- [Kornia MobileSAM Docs](https://kornia.readthedocs.io/en/latest/models/mobile_sam.html)

### Blog Posts & Comparisons

- [EmergentMind: MobileSAM Overview](https://www.emergentmind.com/topics/mobilesam)
- [Ikomia: Master MobileSAM](https://www.ikomia.ai/blog/mobile-sam-faster-segment-anything-model)
- [Medium: Brief Review of MobileSAM](https://sh-tsang.medium.com/brief-review-faster-segment-anything-towards-lightweight-sam-for-mobile-applications-c226bd0a3a25)
- [Lightly AI: SAM and Friends](https://www.lightly.ai/blog/segment-anything-model-and-friends)
- [AMD ROCm Performance Results](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html)
- [ROCm PyTorch Inference Docs](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/pytorch-inference.html)

### Datasets

- [SA-1B Dataset](https://ai.meta.com/datasets/segment-anything/) - Official Meta AI dataset page
- [COCO Dataset](https://cocodataset.org/) - Official COCO website
- [COCO on HuggingFace](https://huggingface.co/datasets/detection-datasets/coco)
- [LVIS Dataset](https://www.lvisdataset.org/) - Large Vocabulary Instance Segmentation

### Related Models & Variants

- [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) - YOLOv8-based fast SAM
- [EfficientSAM](https://arxiv.org/abs/2312.00863) - Leveraged masked image pretraining
- [EdgeSAM](https://arxiv.org/abs/2312.06660) - Prompt-in-the-loop distillation
- [SAM 2](https://github.com/facebookresearch/sam2) - Video segmentation support
- [NanoSAM](https://github.com/NVIDIA-AI-IOT/nanosam) - NVIDIA TensorRT optimized

---

## Quick Reference Commands

```bash
# Clone MobileSAM repository
git clone https://github.com/ChaoningZhang/MobileSAM.git
cd MobileSAM

# Install dependencies
pip install -r requirements.txt

# Download model weights
mkdir -p weights
wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt -O weights/mobile_sam.pt

# Install PyTorch with ROCm support
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# Check AMD GPU status
rocm-smi
rocm-smi --showuse --showmeminfo vram --showpower

# Download COCO dataset
python -c "from datasets import load_dataset; ds = load_dataset('detection-datasets/coco', split='validation[:100]', trust_remote_code=True)"

# Run inference on single image
python mobile_sam_inference.py --image example.jpg --checkpoint weights/mobile_sam.pt

# Get model info
python -c "from mobile_sam import sam_model_registry; model = sam_model_registry['vit_t'](checkpoint='weights/mobile_sam.pt'); print(f'Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')"
```

### Sample Inference Script

Save as `mobile_sam_inference.py`:

```python
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mobile_sam import sam_model_registry, SamPredictor
import argparse

def show_mask(mask, ax, color=None):
    if color is None:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def main(image_path, checkpoint_path):
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mobile_sam = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
    mobile_sam.to(device=device)
    mobile_sam.eval()

    # Load image
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))

    # Initialize predictor
    predictor = SamPredictor(mobile_sam)
    predictor.set_image(image)

    # Segment center point
    h, w = image.shape[:2]
    point = np.array([[w // 2, h // 2]])
    label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=point,
        point_labels=label,
        multimask_output=True,
    )

    # Display results
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    for i, (mask, score) in enumerate(zip(masks, scores)):
        axes[i+1].imshow(image)
        show_mask(mask, axes[i+1])
        axes[i+1].set_title(f"Mask {i+1}, Score: {score:.3f}")
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.savefig("segmentation_result.png")
    print(f"Results saved to segmentation_result.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", default="weights/mobile_sam.pt", help="Path to model checkpoint")
    args = parser.parse_args()

    main(args.image, args.checkpoint)
```

---

## Sources & References

This documentation was compiled using information from the following sources:

- [MobileSAM Paper (arXiv:2306.14289)](https://arxiv.org/abs/2306.14289)
- [Segment Anything Paper (arXiv:2304.02643)](https://arxiv.org/abs/2304.02643)
- [MobileSAM GitHub Repository](https://github.com/ChaoningZhang/MobileSAM)
- [MobileSAM HuggingFace (nielsr/mobilesam)](https://huggingface.co/nielsr/mobilesam)
- [MobileSAM HuggingFace (qualcomm/MobileSam)](https://huggingface.co/qualcomm/MobileSam)
- [Ultralytics MobileSAM Documentation](https://docs.ultralytics.com/models/mobile-sam/)
- [EmergentMind MobileSAM Overview](https://www.emergentmind.com/topics/mobilesam)
- [SA-1B Dataset Information](https://ai.meta.com/datasets/segment-anything/)
- [COCO Dataset](https://cocodataset.org/)
- [EdgeSAM Paper (arXiv:2312.06660)](https://arxiv.org/abs/2312.06660)
- [MobileSAMv2 Paper (arXiv:2312.09579)](https://arxiv.org/abs/2312.09579)
- [ROCm PyTorch Documentation](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/pytorch-inference.html)
- [AMD ROCm Performance Results](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html)

---

**Document Version:** 1.0
**Last Updated:** March 2026
**Target Hardware:** AMD MI300X, RX 7900 XTX, and other ROCm-compatible GPUs
**Model Version:** MobileSAM (ViT-Tiny encoder, 9.66M parameters)
