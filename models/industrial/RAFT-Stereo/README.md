# RAFT-Stereo - Benchmark Guide for AMD GPU

**Navigation:** [🏠 Home](/) | [📑 Models Index](/MODELS_INDEX) | [📝 Contributing](/CONTRIBUTING)

---

## About the Model

RAFT-Stereo is a state-of-the-art deep learning architecture for stereo matching and depth estimation from rectified stereo image pairs. Built upon the RAFT (Recurrent All-Pairs Field Transforms) optical flow architecture, RAFT-Stereo introduces multi-level convolutional GRUs that efficiently propagate information across the image for accurate disparity estimation. The model achieves real-time performance (up to 26 FPS) while maintaining high accuracy, making it suitable for industrial applications requiring both speed and precision in depth estimation.

### Original RAFT-Stereo Paper

**"RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching"** (Lipson, Teed, and Deng, 2021)

RAFT-Stereo is a deep architecture for rectified stereo based on the optical flow network RAFT. The model constructs multi-level 4D correlation volumes from all-pairs feature similarities, and uses multi-level convolutional GRUs to index these correlation volumes and iteratively update a disparity field. By introducing slow-fast GRUs, shared backbones, and resolution hierarchies, RAFT-Stereo can perform accurate real-time inference at 5-26 FPS on KITTI-sized images. The model ranks first on the Middlebury leaderboard, outperforming the next best method on 1px error by 29%, and outperforms all published work on the ETH3D two-view stereo benchmark.

**Paper:** [arXiv:2109.07547](https://arxiv.org/abs/2109.07547) | **Published:** 3DV 2021 (Best Student Paper Award)

---

## Standard Benchmark Datasets

RAFT-Stereo is evaluated on multiple industry-standard stereo matching benchmarks:

### 1. SceneFlow (Primary Training Dataset)

**SceneFlow** is a large-scale synthetic dataset containing 35,454 training image pairs and 4,370 test image pairs. It includes three sub-datasets: FlyingThings3D, Driving, and Monkaa, with fully disjoint textures and 3D model categories between train and test splits.

#### Dataset Structure
- **FlyingThings3D**: 39,000+ stereo frames at 960x540 resolution
- **Driving**: Automotive scenes
- **Monkaa**: Animated sequences
- **Ground Truth**: Dense disparity maps and optical flow

#### Download from Official Source

```bash
# SceneFlow datasets are available from the official website
# Download via BitTorrent (recommended) or direct download
# URL: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
```

```python
# Using torchvision (PyTorch integration)
from torchvision.datasets import SceneFlowStereo
import torchvision.transforms as transforms

# Load SceneFlow dataset
dataset = SceneFlowStereo(
    root='/path/to/sceneflow',
    variant='FlyingThings3D',
    pass_name='clean',
    split='train'
)

# View a sample
left_img, right_img, disparity = dataset[0]
print(f"Disparity shape: {disparity.shape}")
```

### 2. KITTI Stereo 2015

**KITTI Stereo 2015** is the standard real-world benchmark for autonomous driving applications, containing 200 training scenes and 200 test scenes captured from a moving vehicle.

#### Dataset Structure
- **Training**: 200 stereo pairs with ground truth
- **Test**: 200 stereo pairs (ground truth on evaluation server)
- **Resolution**: Variable (typically ~1240×376)
- **Ground Truth**: Semi-automatic from Velodyne LiDAR and CAD models

#### Download and Evaluation

```bash
# Download from official KITTI website
# URL: https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo

# Evaluation metrics:
# - D1: Percentage of disparity outliers (error >3px or >5%)
# - Separate metrics for non-occluded (noc) and all (occ) pixels
```

```python
# Using torchvision
from torchvision.datasets import Kitti2015Stereo

# Load KITTI 2015
dataset = Kitti2015Stereo(
    root='/path/to/kitti2015',
    split='train'
)

# View a sample
sample = dataset[0]
left_img, right_img, disparity = sample[0], sample[1], sample[2]
```

### 3. Middlebury Stereo

**Middlebury** is a high-precision indoor stereo benchmark with 15 training image pairs, known for its challenging sub-pixel accuracy requirements.

- **Evaluation Threshold**: 2px error
- **High Resolution**: Various resolutions, typically >1000px
- **Use Case**: Best for in-the-wild images and fine-grained depth estimation

### 4. ETH3D

**ETH3D** is a grayscale stereo benchmark providing 27 training image pairs with high-quality ground truth.

- **Evaluation Threshold**: 1px error (most stringent)
- **Characteristics**: Challenging outdoor and indoor scenes
- **Format**: Grayscale images

---

## Installation & Inference

### Install RAFT-Stereo

```bash
# Clone the official repository
git clone https://github.com/princeton-vl/RAFT-Stereo.git
cd RAFT-Stereo

# Create conda environment
conda env create -f environment.yaml
conda activate raftstereo

# Or install dependencies manually
conda create -n raftstereo python=3.8
conda activate raftstereo
pip install torch torchvision torchaudio
pip install matplotlib tensorboard scipy opencv-python
```

### Download Pretrained Models

```bash
# Download all pretrained models
chmod ug+x download_models.sh
./download_models.sh

# Or download manually from Google Drive
# Available models:
# - raftstereo-middlebury.pth (recommended for general use)
# - raftstereo-eth3d.pth
# - raftstereo-sceneflow.pth
# - raftstereo-realtime.pth (fast variant)
```

### Basic Inference

```bash
# Standard inference with Middlebury model
python demo.py \
  --restore_ckpt models/raftstereo-middlebury.pth \
  --corr_implementation alt \
  --mixed_precision \
  -l datasets/Middlebury/MiddEval3/testF/*/im0.png \
  -r datasets/Middlebury/MiddEval3/testF/*/im1.png

# Fast real-time inference
python demo.py \
  --restore_ckpt models/raftstereo-realtime.pth \
  --shared_backbone \
  --n_downsample 3 \
  --n_gru_layers 2 \
  --slow_fast_gru \
  --valid_iters 7 \
  --corr_implementation reg_cuda \
  --mixed_precision

# ETH3D inference
python demo.py \
  --restore_ckpt models/raftstereo-eth3d.pth \
  -l datasets/ETH3D/two_view_testing/*/im0.png \
  -r datasets/ETH3D/two_view_testing/*/im1.png

# Save disparity as numpy arrays
python demo.py \
  --restore_ckpt models/raftstereo-middlebury.pth \
  --save_numpy \
  -l left_image.png \
  -r right_image.png
```

### Python API Inference

```python
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
sys.path.append('core')
from raft_stereo import RAFTStereo
from utils.utils import InputPadder

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = RAFTStereo(args)  # args from argparse with model config
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load('models/raftstereo-middlebury.pth'))
model = model.module
model.to(device)
model.eval()

# Load and preprocess images
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)

left_img = load_image('left.png')
right_img = load_image('right.png')

# Pad images to multiple of 8
padder = InputPadder(left_img.shape, divis_by=32)
left_img, right_img = padder.pad(left_img, right_img)

# Inference
with torch.no_grad():
    _, disparity = model(left_img, right_img, iters=32, test_mode=True)

# Remove padding and convert to numpy
disparity = padder.unpad(disparity)
disparity = disparity.cpu().numpy().squeeze()

# Compute depth (if baseline and focal length are known)
# depth = (baseline * focal_length) / disparity
```

### Using torchvision (Prototype)

```python
import torch
from torchvision.prototype.models.depth.stereo import raft_stereo_base, raft_stereo_realtime

# Load pretrained model
model = raft_stereo_base(weights='DEFAULT')
model.eval()
model = model.cuda()

# Or use realtime variant
model_rt = raft_stereo_realtime(weights='DEFAULT')

# Inference
with torch.no_grad():
    disparity_predictions = model(left_img, right_img)
    final_disparity = disparity_predictions[-1]  # Last iteration output
```

### Expected Output

```python
# Disparity map shape: (H, W) or (B, H, W)
# Values represent horizontal pixel displacement
# Typical range: 0 to max_disparity (e.g., 192 for KITTI)

# Visualization
import matplotlib.pyplot as plt
plt.imshow(disparity, cmap='turbo')
plt.colorbar(label='Disparity (pixels)')
plt.title('Stereo Disparity Map')
plt.savefig('disparity_output.png')
```

---

## Benchmark Results & Performance Metrics

### RAFT-Stereo Performance on Standard Benchmarks

| Dataset | Metric | RAFT-Stereo | PSMNet | GwcNet | GANet | Notes |
|---------|--------|-------------|--------|--------|-------|-------|
| **KITTI 2015** | D1-all (%) | 1.94 | 2.32 | 2.11 | 1.93 | Test set, online leaderboard |
| **KITTI 2015** | D1-noc (%) | 1.61 | - | - | - | Non-occluded pixels only |
| **Middlebury** | avg-all (px) | **0.80** | 1.13 | - | 1.48 | Ranked #1 at publication |
| **ETH3D** | >1px (%) | **3.67** | 6.02 | - | 5.58 | Best published at submission |
| **SceneFlow** | EPE (px) | 0.38 | 1.09 | 0.76 | 0.84 | End-Point Error |

**Performance Highlights:**
- **Middlebury**: 29% improvement over next best method on 1px error
- **ETH3D**: Outperforms all published work on two-view stereo
- **KITTI**: Competitive with state-of-the-art, 2nd among published methods

### Inference Speed Comparison

| Model Variant | FPS (KITTI res.) | Runtime (ms) | Accuracy Trade-off | Configuration |
|--------------|------------------|--------------|---------------------|---------------|
| **RAFT-Stereo Standard** | ~7.6 | 132 | Best accuracy | Full resolution, 32 iters |
| **RAFT-Stereo Slow-Fast** | ~20 | 50 | Minimal degradation | Bi-level GRU (1/8, 1/16) |
| **RAFT-Stereo Realtime** | **26** | **38** | Good accuracy | Shared backbone, 7 iters |
| **RAFT-Stereo Minimal** | 5 | 200 | Reduced accuracy | Fewer scales |
| PSMNet | ~0.5 | 2000 | High accuracy | Slower, 3D cost volume |
| GANet | ~0.3 | 3000+ | High accuracy | Very slow |

**Speed vs Accuracy:** RAFT-Stereo achieves 52% runtime reduction with slow-fast GRU while maintaining competitive accuracy (5.91 D1 error vs 6.5 for DSMNet).

### Real-Time Performance Metrics

| Implementation | Throughput (FPS) | Latency (ms) | Resolution | GPU Model | Notes |
|----------------|------------------|--------------|------------|-----------|-------|
| **Realtime Config** | 26 | 38 | 1240×376 | RTX 2080 Ti | Shared backbone, 7 iters |
| **Slow-Fast Config** | 20 | 50 | 1240×376 | RTX 2080 Ti | Bi-level GRU |
| **Standard Config** | 7.6 | 132 | 1240×376 | RTX 2080 Ti | Full accuracy |
| **High-Res Config** | ~3-5 | 200-300 | 1920×1080 | RTX 3090 | Custom resolution |

---

## AMD GPU Benchmarking Setup

### ROCm Installation for AMD GPUs

```bash
# Check ROCm compatibility
rocm-smi

# Verify GPU is supported
rocminfo | grep "Name"

# Install PyTorch with ROCm support (ROCm 6.2)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Check ROCm version
python -c "import torch; print(f'ROCm Version: {torch.version.hip if hasattr(torch.version, \"hip\") else \"N/A\"}')"
```

### Benchmark Script for AMD GPU

```python
import torch
import torch.nn.functional as F
import numpy as np
import time
from PIL import Image
import sys
sys.path.append('core')
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
import argparse

# Device setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Load model
def load_model(ckpt_path, args):
    model = RAFTStereo(args)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(ckpt_path))
    model = model.module
    model.to(device)
    model.eval()
    return model

# Image loading
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)

# Benchmark configuration
args = argparse.Namespace(
    hidden_dims=[128]*3,
    corr_implementation='reg',
    shared_backbone=False,
    corr_levels=4,
    corr_radius=4,
    n_downsample=2,
    n_gru_layers=3,
    slow_fast_gru=False,
    mixed_precision=True
)

# Load model
model = load_model('models/raftstereo-middlebury.pth', args)

# Benchmark on KITTI dataset
import glob
left_images = sorted(glob.glob('datasets/KITTI/2015/training/image_2/*_10.png'))
right_images = sorted(glob.glob('datasets/KITTI/2015/training/image_3/*_10.png'))

results = []
warmup_runs = 5

print("Starting benchmark...")
for idx, (left_path, right_path) in enumerate(zip(left_images[:20], right_images[:20])):
    # Load images
    left_img = load_image(left_path)
    right_img = load_image(right_path)

    # Pad to multiple of 32
    padder = InputPadder(left_img.shape, divis_by=32)
    left_img, right_img = padder.pad(left_img, right_img)

    # Warmup
    if idx < warmup_runs:
        with torch.no_grad():
            _, _ = model(left_img, right_img, iters=32, test_mode=True)
        torch.cuda.synchronize()
        continue

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        _, disparity = model(left_img, right_img, iters=32, test_mode=True)

    torch.cuda.synchronize()
    end_time = time.time()

    inference_time = (end_time - start_time) * 1000  # ms
    fps = 1000.0 / inference_time

    # Get memory stats
    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB

    results.append({
        'sample': idx - warmup_runs,
        'inference_time_ms': inference_time,
        'fps': fps,
        'memory_allocated_gb': memory_allocated,
        'memory_reserved_gb': memory_reserved
    })

    print(f"Sample {idx-warmup_runs}: {inference_time:.2f}ms ({fps:.2f} FPS) | "
          f"Memory: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")

# Summary statistics
avg_time = np.mean([r['inference_time_ms'] for r in results])
avg_fps = np.mean([r['fps'] for r in results])
avg_memory = np.mean([r['memory_allocated_gb'] for r in results])
max_memory = torch.cuda.max_memory_allocated() / 1024**3

print(f"\n=== Benchmark Summary ===")
print(f"Average Inference Time: {avg_time:.2f} ms")
print(f"Average FPS: {avg_fps:.2f}")
print(f"Average Memory Usage: {avg_memory:.2f} GB")
print(f"Peak Memory Usage: {max_memory:.2f} GB")
```

### Performance Metrics Table Template

| Metric | NVIDIA RTX 2080 Ti | NVIDIA RTX 3090 | AMD MI300X | AMD RX 7900 XTX | Notes |
|--------|-------------------|-----------------|------------|-----------------|-------|
| **GPU Model** | NVIDIA RTX 2080 Ti | NVIDIA RTX 3090 | AMD MI300X | AMD RX 7900 XTX | Reference vs target GPUs |
| **Memory (GB)** | 11 | 24 | 192 | 24 | VRAM capacity |
| **TDP (W)** | 250 | 350 | 750 | 355 | Thermal design power |
| **Resolution** | 1240×376 | 1240×376 | _[Your result]_ | _[Your result]_ | KITTI standard resolution |
| **Inference Time (ms)** | 132 | ~80 | _[Your result]_ | _[Your result]_ | Standard config, 32 iters |
| **FPS (Standard)** | 7.6 | ~12 | _[Your result]_ | _[Your result]_ | Full accuracy mode |
| **FPS (Realtime)** | 26 | ~40 | _[Your result]_ | _[Your result]_ | Fast config, 7 iters |
| **Peak Memory Usage (GB)** | ~8 | ~10 | _[Your result]_ | _[Your result]_ | Monitor with rocm-smi |
| **Average Power Draw (W)** | ~200 | ~280 | _[Your result]_ | _[Your result]_ | Monitor with rocm-smi --showpower |
| **Disparity EPE** | 0.38 | 0.38 | _[Your result]_ | _[Your result]_ | SceneFlow End-Point Error |
| **KITTI D1-all (%)** | 1.94 | 1.94 | _[Your result]_ | _[Your result]_ | Accuracy should match |

### AMD-Specific Metrics to Track

```python
import subprocess
import torch

# GPU utilization tracking
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
        print("rocm-smi not found. Install ROCm tools.")

# Memory tracking during inference
print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
print(f"Max Allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

# ROCm info
if torch.cuda.is_available():
    print(f"\n=== ROCm Configuration ===")
    if hasattr(torch.version, 'hip'):
        print(f"ROCm Version: {torch.version.hip}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Device Count: {torch.cuda.device_count()}")

# Get detailed stats
get_rocm_smi_stats()
```

### Complete Runtime Metrics Table

| Runtime Metric | Formula | NVIDIA RTX 2080 Ti | NVIDIA RTX 3090 | AMD MI300X | AMD RX 7900 XTX | Notes |
|----------------|---------|-------------------|-----------------|------------|-----------------|-------|
| **Throughput (FPS)** | 1000 / inference_time_ms | 7.6 | ~12 | _[Your result]_ | _[Your result]_ | Standard config |
| **Realtime Throughput (FPS)** | 1000 / inference_time_ms | 26 | ~40 | _[Your result]_ | _[Your result]_ | Fast config |
| **Disparity Estimation Rate (Mpx/s)** | (H × W × FPS) / 1e6 | ~3.5 | ~5.6 | _[Your result]_ | _[Your result]_ | Million pixels per second |
| **GPU Utilization (%)** | From nvidia-smi / rocm-smi | ~95 | ~95 | _[Your result]_ | _[Your result]_ | During inference |
| **Memory Bandwidth Utilization** | From nvidia-smi / rocm-smi | ~400 GB/s | ~750 GB/s | _[Your result]_ | _[Your result]_ | MI300X: ~5.3 TB/s, RX 7900 XTX: ~960 GB/s theoretical |
| **TFLOPS Utilized** | Calculated from operations | _[Reference]_ | _[Reference]_ | _[Your result]_ | _[Your result]_ | FP16/FP32 compute |
| **Latency to First Disparity (ms)** | Time to first valid output | ~30 | ~20 | _[Your result]_ | _[Your result]_ | Important for real-time apps |
| **Energy per Frame (J)** | (power_watts × time_s) | ~26 | ~37 | _[Your result]_ | _[Your result]_ | Energy efficiency |
| **Power Efficiency (FPS/W)** | FPS / power_draw | 0.038 | 0.043 | _[Your result]_ | _[Your result]_ | Higher is better |

### Optimization Flags for AMD GPUs

```python
# ROCm-specific optimizations
import torch

# Enable TF32 on AMD (if supported)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Use mixed precision with autocast
from torch.cuda.amp import autocast

with torch.no_grad():
    with autocast(enabled=True):
        _, disparity = model(left_img, right_img, iters=32, test_mode=True)

# Benchmark with different batch sizes (if applicable)
# Note: RAFT-Stereo typically processes single image pairs
# but batch processing can be implemented for multiple pairs

# Memory optimization: gradient checkpointing (if training)
# For inference, ensure model is in eval mode to disable dropout/batchnorm updates
```

---

## Stereo Matching Leaderboards

### KITTI Stereo 2015 Leaderboard

The [KITTI Stereo Benchmark](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) is the primary evaluation platform for stereo matching in autonomous driving.

**Top Methods (at RAFT-Stereo publication):**
1. **RAFT-Stereo**: 1.94% D1-all (2nd among published methods)
2. **GANet**: 1.93% D1-all
3. **PSMNet**: 2.32% D1-all
4. **GwcNet**: 2.11% D1-all

**Evaluation Metrics:**
- **D1-all**: % of pixels with error >3px or >5%
- **D1-noc**: Same metric for non-occluded pixels only
- **D1-fg**: Foreground pixels only

### Middlebury Stereo Leaderboard

The [Middlebury Stereo Benchmark](https://vision.middlebury.edu/stereo/eval3/) provides high-precision evaluation with challenging indoor scenes.

**RAFT-Stereo Performance:**
- **Ranked #1** at time of publication
- **avg-all**: 0.80 pixels (29% better than next best)
- **Evaluation**: 2px error threshold

### ETH3D Two-View Stereo Benchmark

The [ETH3D Benchmark](https://www.eth3d.net/overview) provides high-quality ground truth with 1px accuracy requirements.

**RAFT-Stereo Performance:**
- **Best published work** at submission time
- **>1px error**: 3.67% (vs 6.02% for PSMNet)
- **Most stringent evaluation** among standard benchmarks

### Key Metrics Tracked

| Metric | Description | Lower is Better |
|--------|-------------|-----------------|
| **EPE (End-Point Error)** | Average L2 distance between predicted and ground truth disparity | Yes |
| **D1 / >Npx** | Percentage of pixels with error >N pixels or >5% | Yes |
| **avg-all** | Average error across all pixels | Yes |
| **avg-noc** | Average error for non-occluded pixels only | Yes |
| **Runtime** | Inference time per image pair | Yes |

---

## Additional Resources

### Official Repositories

- [RAFT-Stereo GitHub (Princeton VL)](https://github.com/princeton-vl/RAFT-Stereo)
- [PyTorch Vision - RAFT-Stereo](https://github.com/pytorch/vision/blob/main/torchvision/prototype/models/depth/stereo/raft_stereo.py)
- [Original RAFT (Optical Flow)](https://github.com/princeton-vl/RAFT)

### Papers & Documentation

- [RAFT-Stereo Paper (arXiv:2109.07547)](https://arxiv.org/abs/2109.07547)
- [RAFT Paper (arXiv:2003.12039)](https://arxiv.org/abs/2003.12039) - Original optical flow architecture
- [Papers with Code - RAFT-Stereo](https://paperswithcode.com/paper/raft-stereo-multilevel-recurrent-field)
- [IEEE 3DV 2021 Paper](https://ieeexplore.ieee.org/document/9665883/)

### Benchmark Websites

- [KITTI Stereo Benchmark](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
- [Middlebury Stereo Evaluation](https://vision.middlebury.edu/stereo/eval3/)
- [ETH3D Benchmark](https://www.eth3d.net/overview)
- [SceneFlow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

### Related Work & Improvements

- [IGEV-Stereo](https://github.com/gangweix/IGEV) - Iterative Geometry Encoding Volume (CVPR 2023, TPAMI 2025)
- [NeRF-Supervised Deep Stereo](https://github.com/fabiotosi92/NeRF-Supervised-Deep-Stereo) - Using neural rendering for training data
- [SEA-RAFT](https://arxiv.org/abs/2405.14793) - Simple, Efficient, Accurate RAFT improvements

### AMD ROCm Resources

- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [AMD ROCm Performance Results](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html)
- [PyTorch ROCm Installation Guide](https://pytorch.org/get-started/locally/)
- [ROCm GitHub](https://github.com/ROCm)

### Datasets

- [SceneFlow (FlyingThings3D, Driving, Monkaa)](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/)
- [Middlebury Stereo Datasets](https://vision.middlebury.edu/stereo/data/)
- [ETH3D Benchmark](https://www.eth3d.net/datasets)

### Blog Posts & Tutorials

- [PyTorch Torchvision Stereo Matching Documentation](https://docs.pytorch.org/vision/main/models/raft.html)
- [Scene Flow Estimation Papers](https://paperswithcode.com/task/scene-flow-estimation)
- [Stereo Vision Tutorial - OpenCV](https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html)

---

## Quick Reference Commands

```bash
# Install RAFT-Stereo
git clone https://github.com/princeton-vl/RAFT-Stereo.git
cd RAFT-Stereo
conda env create -f environment.yaml
conda activate raftstereo

# Download pretrained models
./download_models.sh

# Run inference (Middlebury model recommended for general use)
python demo.py \
  --restore_ckpt models/raftstereo-middlebury.pth \
  --corr_implementation alt \
  --mixed_precision \
  -l left_image.png \
  -r right_image.png

# Fast real-time inference
python demo.py \
  --restore_ckpt models/raftstereo-realtime.pth \
  --shared_backbone \
  --n_downsample 3 \
  --n_gru_layers 2 \
  --slow_fast_gru \
  --valid_iters 7 \
  --corr_implementation reg_cuda \
  --mixed_precision

# Check AMD GPU status
rocm-smi
rocm-smi --showuse --showmeminfo vram --showpower

# Evaluate on KITTI 2015
python evaluate_stereo.py \
  --restore_ckpt models/raftstereo-sceneflow.pth \
  --dataset kitti

# Evaluate on SceneFlow
python evaluate_stereo.py \
  --restore_ckpt models/raftstereo-sceneflow.pth \
  --dataset sceneflow

# Save output as numpy arrays
python demo.py \
  --restore_ckpt models/raftstereo-middlebury.pth \
  --save_numpy \
  -l left.png \
  -r right.png
```

---

## Performance Optimization Tips

### Speed Optimizations

1. **Use Realtime Model**: Trade minimal accuracy for 3-4x speedup
2. **Mixed Precision**: Enable with `--mixed_precision` flag (FP16)
3. **Correlation Implementation**: Use `reg_cuda` for speed, `alt` for memory efficiency
4. **Reduce Iterations**: Fewer GRU iterations (7-12 vs 32) for faster inference
5. **Shared Backbone**: Use `--shared_backbone` to reduce parameters

### Memory Optimizations

1. **Use `alt` Correlation**: Significantly reduces memory on high-resolution images
2. **Downsample Input**: Process at lower resolution, upsample disparity
3. **Gradient Checkpointing**: If training/fine-tuning (not needed for inference)
4. **Batch Size 1**: RAFT-Stereo designed for single pair processing

### Accuracy Optimizations

1. **More Iterations**: Use 32+ GRU iterations for best results
2. **Multi-Scale**: Keep all resolution levels (don't downsample aggressively)
3. **Fine-tuning**: Train on target domain (e.g., fine-tune on KITTI for driving)
4. **Ensemble**: Average multiple models (Middlebury + ETH3D)

---

## Industrial Use Cases

### Autonomous Vehicles
- **Real-time depth estimation** for obstacle detection
- **Scene understanding** at 20-26 FPS on automotive hardware
- **Benchmarked on KITTI** - industry standard for driving scenarios

### Robotics & Navigation
- **Mobile robot navigation** with stereo cameras
- **Obstacle avoidance** in industrial environments
- **Bin picking** and **grasp planning** with depth estimation

### 3D Reconstruction
- **Industrial inspection** with stereo camera rigs
- **Quality control** - 3D surface measurement
- **Reverse engineering** - capturing 3D geometry

### Augmented Reality
- **Real-time depth** for AR applications
- **Occlusion handling** with accurate disparity
- **Scene understanding** for AR content placement

---

**Document Version:** 1.0
**Last Updated:** March 2026
**Target Hardware:** AMD MI300X, RX 7900 XTX, and other ROCm-compatible GPUs
**Model Version:** RAFT-Stereo (3DV 2021)