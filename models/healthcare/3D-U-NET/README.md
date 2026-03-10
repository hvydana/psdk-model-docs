# 3D U-Net - Benchmark Guide for AMD GPU

**Navigation:** [🏠 Home](/) | [📑 Models Index](/MODELS_INDEX) | [📝 Contributing](/CONTRIBUTING)

---

## About the Model

3D U-Net is a groundbreaking deep learning architecture for volumetric medical image segmentation. It extends the original 2D U-Net architecture to three dimensions, enabling dense volumetric segmentation from sparse annotations. The model employs a symmetric encoder-decoder structure with skip connections, allowing it to learn from limited annotated data while maintaining spatial context across entire 3D volumes. 3D U-Net has become a foundational architecture for medical imaging tasks involving CT scans, MRI volumes, and other 3D medical data.

### Original 3D U-Net Paper

**"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"** (Çiçek et al., 2016)

3D U-Net builds upon the success of the original 2D U-Net by extending all operations to three dimensions. The architecture features a contracting path to capture context and a symmetric expanding path for precise localization. The network can be trained end-to-end from very few images and achieves excellent performance on challenging 3D segmentation tasks. The key innovation is its ability to learn dense predictions from sparsely annotated training data, making it particularly valuable for medical imaging where expert annotations are expensive and time-consuming to obtain.

**Paper:** [arXiv:1606.06650](https://arxiv.org/abs/1606.06650) | **Published:** MICCAI 2016

**Authors:** Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger

---

## Standard Benchmark Datasets

### 1. BraTS (Brain Tumor Segmentation)

**BraTS** is the premier benchmark for evaluating brain tumor segmentation algorithms. The Multimodal Brain Tumor Image Segmentation Benchmark contains multi-contrast MRI scans with expert annotations for different tumor regions.

#### Dataset Structure
- **Modalities**: T1, T1-weighted with contrast (T1ce), T2-weighted, FLAIR
- **Annotations**: Enhancing tumor, peritumoral edema, necrotic core
- **BraTS 2020**: 369 training cases, 125 validation cases
- **BraTS 2023**: Extended with additional cases and synthesis tasks

#### Download from Official Source

```bash
# Register and download from official BraTS website
# Visit: http://braintumorsegmentation.org/

# Alternative: Use Kaggle dataset
# https://www.kaggle.com/datasets/awsaf49/brats2020-training-data
```

```python
# Example loading script (requires downloaded data)
import nibabel as nib
import numpy as np

# Load BraTS case
def load_brats_case(case_path):
    """
    Load a BraTS case with all modalities
    """
    t1 = nib.load(f"{case_path}_t1.nii.gz").get_fdata()
    t1ce = nib.load(f"{case_path}_t1ce.nii.gz").get_fdata()
    t2 = nib.load(f"{case_path}_t2.nii.gz").get_fdata()
    flair = nib.load(f"{case_path}_flair.nii.gz").get_fdata()
    seg = nib.load(f"{case_path}_seg.nii.gz").get_fdata()

    # Stack modalities
    volume = np.stack([t1, t1ce, t2, flair], axis=-1)

    return volume, seg

# Example usage
volume, segmentation = load_brats_case("path/to/BraTS20_Training_001/BraTS20_Training_001")
print(f"Volume shape: {volume.shape}")  # (240, 240, 155, 4)
print(f"Segmentation shape: {segmentation.shape}")  # (240, 240, 155)
```

### 2. Medical Segmentation Decathlon

**Medical Segmentation Decathlon** is a comprehensive benchmark featuring 10 distinct segmentation tasks across different organs and pathologies. It's designed to test the generalization capabilities of segmentation algorithms.

#### Ten Tasks
1. **Task01**: Liver and Liver Tumor (CT)
2. **Task02**: Brain Tumors (MRI)
3. **Task03**: Hippocampus (MRI)
4. **Task04**: Lung Tumors (CT)
5. **Task05**: Prostate (MRI)
6. **Task06**: Cardiac (MRI)
7. **Task07**: Pancreas and Pancreatic Tumor (CT)
8. **Task08**: Colon Cancer (CT)
9. **Task09**: Hepatic Vessels (CT)
10. **Task10**: Spleen (CT)

#### Download from HuggingFace

```bash
# Install dependencies
pip install datasets nibabel
```

```python
from datasets import load_dataset

# Load Medical Segmentation Decathlon
# Option 1: From HuggingFace (if available)
dataset = load_dataset("Novel-BioMedAI/Medical_Segmentation_Decathlon", "Task01_BrainTumour")

# Option 2: Download from official source
# Visit: http://medicaldecathlon.com/
# Or AWS: https://registry.opendata.aws/msd/
```

### 3. KiTS (Kidney Tumor Segmentation)

**KiTS** is a challenge dataset for semantic segmentation of kidneys, kidney tumors, and kidney cysts in contrast-enhanced CT imaging.

#### Dataset Evolution
- **KiTS19**: 210 training cases, 90 test cases
- **KiTS21**: Expanded with additional segmentation classes
- **KiTS23**: 489 training cases, 110 test cases, added nephrogenic phase

#### Download

```bash
# Clone KiTS23 repository
git clone https://github.com/neheller/kits23.git
cd kits23

# Download data (requires registration)
python -m starter_code.get_imaging
```

```python
# Load KiTS data
import nibabel as nib

def load_kits_case(case_id):
    """
    Load a KiTS case
    """
    imaging = nib.load(f"kits23/dataset/case_{case_id:05d}/imaging.nii.gz")
    segmentation = nib.load(f"kits23/dataset/case_{case_id:05d}/segmentation.nii.gz")

    return imaging.get_fdata(), segmentation.get_fdata()

# Example
ct_scan, mask = load_kits_case(0)
print(f"CT shape: {ct_scan.shape}, Mask shape: {mask.shape}")
```

---

## Installation & Inference

### Option 1: Install MONAI (Recommended)

**MONAI** (Medical Open Network for AI) is the industry-standard PyTorch-based framework for medical imaging with optimized 3D U-Net implementations.

```bash
# Install PyTorch with CUDA/ROCm support first
# For AMD GPUs with ROCm:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Install MONAI
pip install monai[all]

# Verify installation
python -c "import monai; print(f'MONAI version: {monai.__version__}')"
```

### Option 2: Install Standalone PyTorch 3D U-Net

```bash
# Option A: fepegar/unet (simplest)
pip install unet

# Option B: wolny/pytorch-3dunet (feature-rich)
git clone https://github.com/wolny/pytorch-3dunet.git
cd pytorch-3dunet
pip install -r requirements.txt
pip install -e .

# Option C: ellisdg/3DUnetCNN
git clone https://github.com/ellisdg/3DUnetCNN.git
cd 3DUnetCNN
pip install -r requirements.txt
```

### Basic Inference with MONAI

```python
import torch
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Spacingd, Orientationd, ScaleIntensityRanged,
    EnsureTyped
)
import nibabel as nib

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define 3D U-Net model
model = UNet(
    spatial_dims=3,
    in_channels=4,  # Four MRI modalities
    out_channels=4,  # Background + 3 tumor regions
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

# Load pre-trained weights (if available)
# model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

# Preprocessing pipeline
transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    Orientationd(keys=["image"], axcodes="RAS"),
    ScaleIntensityRanged(
        keys=["image"], a_min=-175, a_max=250,
        b_min=0.0, b_max=1.0, clip=True
    ),
    EnsureTyped(keys=["image"]),
])

# Inference function
def segment_volume(image_path):
    """
    Segment a 3D medical volume
    """
    data = {"image": image_path}
    data = transforms(data)

    with torch.no_grad():
        inputs = data["image"].unsqueeze(0).to(device)
        outputs = model(inputs)
        prediction = torch.argmax(outputs, dim=1).cpu().numpy()

    return prediction[0]

# Example usage
prediction = segment_volume("path/to/volume.nii.gz")
print(f"Prediction shape: {prediction.shape}")
```

### Training Script with MONAI

```python
import torch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import DataLoader, Dataset
from torch.optim import Adam

# Model setup
model = UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=4,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

# Loss and optimizer
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = Adam(model.parameters(), lr=1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

# Training loop
def train_epoch(model, loader, optimizer, loss_function):
    model.train()
    epoch_loss = 0

    for batch_data in loader:
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)

# Example training
epochs = 100
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, loss_function)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")
```

### Expected Output

```json
{
  "segmentation_shape": [155, 240, 240],
  "classes": {
    "0": "background",
    "1": "necrotic_and_non_enhancing_tumor",
    "2": "peritumoral_edema",
    "3": "enhancing_tumor"
  },
  "metrics": {
    "dice_whole_tumor": 0.891,
    "dice_tumor_core": 0.834,
    "dice_enhancing_tumor": 0.782
  }
}
```

---

## Benchmark Results & Performance Metrics

### 3D U-Net Performance on BraTS

| Model | Whole Tumor Dice | Tumor Core Dice | Enhancing Tumor Dice | Parameters | Year |
|-------|-----------------|----------------|---------------------|------------|------|
| **3D U-Net (Original)** | 0.86 | 0.72 | 0.63 | ~19M | 2016 |
| **3D U-Net (MONAI)** | 0.89 | 0.83 | 0.78 | ~19M | 2020+ |
| **nnU-Net** | 0.92 | 0.87 | 0.82 | Variable | 2020 |
| **Attention U-Net 3D** | 0.88 | 0.81 | 0.76 | ~25M | 2019 |
| **V-Net** | 0.85 | 0.78 | 0.72 | ~45M | 2016 |

**Dice Score** = Dice Similarity Coefficient (higher is better, range 0-1)

### Performance on Medical Segmentation Decathlon

| Task | Organ/Pathology | 3D U-Net Dice | nnU-Net Dice | Dataset Size | Modality |
|------|----------------|---------------|--------------|--------------|----------|
| **Task01** | Liver Tumor | 0.94 | 0.95 | 131 cases | CT |
| **Task02** | Brain Tumor | 0.68 | 0.73 | 484 cases | MRI |
| **Task03** | Hippocampus | 0.88 | 0.90 | 260 cases | MRI |
| **Task04** | Lung Tumor | 0.69 | 0.73 | 63 cases | CT |
| **Task05** | Prostate | 0.76 | 0.78 | 32 cases | MRI |
| **Task06** | Cardiac | 0.91 | 0.93 | 20 cases | MRI |
| **Task07** | Pancreas | 0.81 | 0.84 | 281 cases | CT |
| **Task08** | Colon | 0.55 | 0.61 | 126 cases | CT |
| **Task09** | Hepatic Vessels | 0.72 | 0.76 | 303 cases | CT |
| **Task10** | Spleen | 0.96 | 0.96 | 41 cases | CT |

### KiTS Challenge Performance

| Model | Kidney Dice | Tumor Dice | Cyst Dice | Average | Competition |
|-------|------------|-----------|-----------|---------|-------------|
| **3D U-Net Baseline** | 0.957 | 0.820 | 0.785 | 0.854 | KiTS19 |
| **nnU-Net** | 0.974 | 0.851 | 0.832 | 0.886 | KiTS19 |
| **Top KiTS21 Solution** | 0.978 | 0.873 | 0.869 | 0.907 | KiTS21 |
| **Top KiTS23 Solution** | 0.982 | 0.889 | 0.881 | 0.917 | KiTS23 |

### Segmentation Metrics Explained

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| **Dice Score** | 2 × TP / (2 × TP + FP + FN) | 0.0 - 1.0 | >0.9 excellent, >0.7 good, <0.5 poor |
| **IoU (Jaccard)** | TP / (TP + FP + FN) | 0.0 - 1.0 | >0.8 excellent, >0.5 good |
| **Hausdorff Distance** | max(h(A,B), h(B,A)) | 0 - ∞ | Lower is better, measures boundary accuracy |
| **Sensitivity (Recall)** | TP / (TP + FN) | 0.0 - 1.0 | Percentage of actual positives detected |
| **Specificity** | TN / (TN + FP) | 0.0 - 1.0 | Percentage of actual negatives correctly identified |

---

## AMD GPU Benchmarking Setup

### ROCm Installation for AMD GPUs

```bash
# Check ROCm compatibility
rocm-smi

# Install PyTorch with ROCm support (ROCm 6.2)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Install MONAI
pip install monai[all]

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### Benchmark Script for AMD GPU

```python
import torch
import time
import numpy as np
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    EnsureTyped, Spacingd
)
import nibabel as nib

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"ROCm Version: {torch.version.hip if torch.cuda.is_available() else 'N/A'}")

# Model configuration
model = UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=4,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Benchmark inference
def benchmark_inference(model, input_shape=(1, 4, 128, 128, 128), num_iterations=100):
    """
    Benchmark inference speed
    """
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Synchronize GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for i in range(num_iterations):
            start_time = time.time()
            output = model(dummy_input)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.time()
            times.append(end_time - start_time)

    # Calculate statistics
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    # Memory statistics
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated(device) / 1024**3
        reserved_memory = torch.cuda.memory_reserved(device) / 1024**3
        max_memory = torch.cuda.max_memory_allocated(device) / 1024**3
    else:
        allocated_memory = reserved_memory = max_memory = 0

    return {
        "mean_time": mean_time,
        "std_time": std_time,
        "min_time": min_time,
        "max_time": max_time,
        "throughput": 1.0 / mean_time,
        "allocated_memory_gb": allocated_memory,
        "reserved_memory_gb": reserved_memory,
        "max_memory_gb": max_memory
    }

# Run benchmark with different input sizes
input_sizes = [
    (1, 4, 64, 64, 64),    # Small
    (1, 4, 128, 128, 128), # Medium
    (1, 4, 160, 192, 128), # BraTS typical size
]

print("\n=== Inference Benchmark Results ===")
for size in input_sizes:
    print(f"\nInput shape: {size}")
    results = benchmark_inference(model, input_shape=size, num_iterations=50)

    print(f"  Mean inference time: {results['mean_time']*1000:.2f} ± {results['std_time']*1000:.2f} ms")
    print(f"  Min time: {results['min_time']*1000:.2f} ms")
    print(f"  Max time: {results['max_time']*1000:.2f} ms")
    print(f"  Throughput: {results['throughput']:.2f} volumes/sec")
    print(f"  Peak memory: {results['max_memory_gb']:.2f} GB")

# Training benchmark
def benchmark_training(model, input_shape=(2, 4, 96, 96, 96), num_iterations=50):
    """
    Benchmark training speed
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)

    # Warmup
    for _ in range(5):
        inputs = torch.randn(input_shape).to(device)
        labels = torch.randint(0, 4, (input_shape[0], 1, *input_shape[2:])).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for i in range(num_iterations):
        inputs = torch.randn(input_shape).to(device)
        labels = torch.randint(0, 4, (input_shape[0], 1, *input_shape[2:])).to(device)

        start_time = time.time()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()
        times.append(end_time - start_time)

    times = np.array(times)

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "throughput": input_shape[0] / np.mean(times)
    }

print("\n=== Training Benchmark Results ===")
train_results = benchmark_training(model, input_shape=(2, 4, 96, 96, 96), num_iterations=30)
print(f"Mean training time per batch: {train_results['mean_time']*1000:.2f} ± {train_results['std_time']*1000:.2f} ms")
print(f"Training throughput: {train_results['throughput']:.2f} samples/sec")
```

### Performance Metrics Table Template

| Metric | NVIDIA A100-80GB | NVIDIA V100-32GB | AMD MI300X | AMD RX 7900 XTX | Notes |
|--------|------------------|------------------|------------|-----------------|-------|
| **GPU Model** | NVIDIA A100-80GB | NVIDIA V100-32GB | AMD MI300X | AMD RX 7900 XTX | Compare datacenter vs consumer GPUs |
| **Memory (GB)** | 80 | 32 | 192 | 24 | VRAM capacity |
| **TDP (W)** | 400 | 300 | 750 | 355 | Thermal design power |
| **Input Size** | 128×128×128 | 128×128×128 | _[Your result]_ | _[Your result]_ | Typical volume size |
| **Batch Size** | 2 | 2 | _[Your result]_ | _[Your result]_ | Training batch size |
| **Inference Time (ms)** | 45 | 68 | _[Your result]_ | _[Your result]_ | Per volume, FP16 |
| **Training Time (ms/batch)** | 380 | 520 | _[Your result]_ | _[Your result]_ | Forward + backward pass |
| **Throughput (volumes/sec)** | 22.2 | 14.7 | _[Your result]_ | _[Your result]_ | Inference throughput |
| **Peak Memory Usage (GB)** | 8.5 | 8.5 | _[Your result]_ | _[Your result]_ | For 128³ input |
| **Average Power Draw (W)** | 320 | 250 | _[Your result]_ | _[Your result]_ | During training |
| **Energy per Volume (Wh)** | 0.004 | 0.005 | _[Your result]_ | _[Your result]_ | Inference energy cost |

### AMD-Specific Metrics to Track

```python
import subprocess

def get_rocm_smi_stats():
    """Get AMD GPU statistics using rocm-smi"""
    result = subprocess.run(
        ['rocm-smi', '--showuse', '--showmeminfo', 'vram', '--showpower'],
        capture_output=True, text=True
    )
    return result.stdout

# GPU Memory tracking
print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
print(f"Max Allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

# ROCm info
print(f"ROCm Version: {torch.version.hip}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")

# Print ROCm-SMI stats
print("\n=== ROCm-SMI Statistics ===")
print(get_rocm_smi_stats())
```

### Complete Runtime Metrics Table

| Runtime Metric | Formula | NVIDIA A100-80GB | NVIDIA V100-32GB | AMD MI300X | AMD RX 7900 XTX | Notes |
|----------------|---------|------------------|------------------|------------|-----------------|-------|
| **Dice Score** | 2×TP/(2×TP+FP+FN) | 0.89 | 0.89 | _[Your result]_ | _[Your result]_ | On BraTS validation |
| **Inference Latency (ms)** | Per volume | 45 | 68 | _[Your result]_ | _[Your result]_ | 128³ volume, FP16 |
| **Training Speed (samples/sec)** | Batch_size / time | 5.3 | 3.8 | _[Your result]_ | _[Your result]_ | Batch size 2 |
| **GPU Utilization (%)** | From nvidia-smi / rocm-smi | 95 | 92 | _[Your result]_ | _[Your result]_ | Average during training |
| **Memory Bandwidth Utilization** | Achieved / Peak | 85% | 78% | _[Your result]_ | _[Your result]_ | % of theoretical peak |
| **TFLOPS Utilized** | Calculated from operations | ~180 | ~120 | _[Your result]_ | _[Your result]_ | FP16 compute throughput |
| **Memory Efficiency** | Memory used / Available | 10.6% | 26.6% | _[Your result]_ | _[Your result]_ | For 128³ input |
| **Energy Efficiency (inferences/kWh)** | 3600 / (power × time) | 252,000 | 176,000 | _[Your result]_ | _[Your result]_ | Higher is better |

---

## State-of-the-Art: nnU-Net

**nnU-Net** (no-new-Net) is a self-configuring framework that automatically adapts U-Net architectures to any medical segmentation task, achieving state-of-the-art results without manual intervention.

### nnU-Net vs Standard 3D U-Net

| Feature | 3D U-Net | nnU-Net |
|---------|----------|---------|
| **Architecture** | Fixed | Self-configuring |
| **Preprocessing** | Manual | Automated based on dataset |
| **Patch Size** | Fixed | Dataset-adaptive |
| **Data Augmentation** | User-defined | Automatically optimized |
| **Postprocessing** | Minimal | Adaptive with ensemble |
| **Performance (avg Dice)** | 0.75-0.85 | 0.85-0.92 |

### Installing nnU-Net

```bash
# Install nnU-Net
pip install nnunetv2

# Set environment variables
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

### Paper & Resources

**"nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation"** (Isensee et al., 2020)

Published in Nature Methods | [arXiv:1809.10486](https://arxiv.org/abs/1809.10486)

---

## Additional Resources

### Official Repositories & Implementations

- [MONAI Framework](https://github.com/Project-MONAI/MONAI) - Official medical imaging framework
- [MONAI Tutorials](https://github.com/Project-MONAI/tutorials) - Extensive tutorials and examples
- [wolny/pytorch-3dunet](https://github.com/wolny/pytorch-3dunet) - Feature-rich PyTorch implementation
- [fepegar/unet](https://github.com/fepegar/unet) - Simple pip-installable 3D U-Net
- [nnU-Net Repository](https://github.com/MIC-DKFZ/nnUNet) - State-of-the-art self-configuring framework

### Papers & Documentation

- [3D U-Net Paper (arXiv:1606.06650)](https://arxiv.org/abs/1606.06650)
- [Original 2D U-Net Paper (arXiv:1505.04597)](https://arxiv.org/abs/1505.04597)
- [nnU-Net Paper (Nature Methods)](https://www.nature.com/articles/s41592-020-01008-z)
- [Medical Segmentation Decathlon Paper (arXiv:2106.05735)](https://arxiv.org/abs/2106.05735)
- [MONAI Documentation](https://docs.monai.io/)

### Benchmark Challenges & Leaderboards

- [BraTS Challenge](http://braintumorsegmentation.org/) - Brain tumor segmentation benchmark
- [Medical Segmentation Decathlon](http://medicaldecathlon.com/) - Multi-organ segmentation
- [KiTS Challenge](https://kits-challenge.org/) - Kidney tumor segmentation
- [Grand Challenges](https://grand-challenge.org/) - Collection of medical imaging challenges

### Datasets

- [BraTS 2020 on Kaggle](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)
- [Medical Segmentation Decathlon on HuggingFace](https://huggingface.co/datasets/Novel-BioMedAI/Medical_Segmentation_Decathlon)
- [Medical Segmentation Decathlon on AWS](https://registry.opendata.aws/msd/)
- [KiTS23 GitHub](https://github.com/neheller/kits23)

### Blog Posts & Tutorials

- [MONAI: Medical Imaging with PyTorch](https://learnopencv.com/monai-medical-imaging-pytorch/)
- [3D Medical Image Segmentation with MONAI & U-Net](https://www.analyticsvidhya.com/blog/2024/03/guide-on-3d-medical-image-segmentation-with-monai-unet/)
- [Creating U-Net with PyTorch for Medical Segmentation](https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-model-building-6ab09d6a0862/)
- [AMD ROCm Performance Results](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html)
- [Understanding Medical Image Segmentation Metrics](https://medium.com/mastering-data-science/understanding-evaluation-metrics-in-medical-image-segmentation-d289a373a3f)

### Related Architectures

- **V-Net**: Fully convolutional network for volumetric medical image segmentation
- **Attention U-Net**: U-Net with attention gates for better feature focus
- **U-Net++**: Nested U-Net architecture with dense skip connections
- **SegResNet**: Residual U-Net for medical image segmentation
- **TransUNet**: Transformer-enhanced U-Net

---

## Quick Reference Commands

```bash
# Install MONAI with ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
pip install monai[all]

# Install standalone 3D U-Net
pip install unet

# Install nnU-Net
pip install nnunetv2

# Check AMD GPU status
rocm-smi
rocm-smi --showuse --showmeminfo vram --showpower

# Run MONAI 3D U-Net training example
python -c "from monai.networks.nets import UNet; print('MONAI 3D U-Net ready')"

# Monitor GPU during training
watch -n 1 rocm-smi

# Check PyTorch + ROCm
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'ROCm: {torch.version.hip}')"
```

---

## Performance Optimization Tips

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

# Enable mixed precision
scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad()

    with autocast():
        outputs = model(inputs)
        loss = loss_function(outputs, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Gradient Checkpointing

```python
# Reduce memory usage by trading compute for memory
from torch.utils.checkpoint import checkpoint

# In model forward pass
def forward(self, x):
    x = checkpoint(self.encoder, x)
    x = checkpoint(self.decoder, x)
    return x
```

### Patch-Based Training for Large Volumes

```python
from monai.transforms import RandCropByPosNegLabeld

# Random crop patches during training
transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(96, 96, 96),
        pos=1,
        neg=1,
        num_samples=4,
    ),
])
```

---

**Document Version:** 1.0
**Last Updated:** March 2026
**Target Hardware:** AMD MI300X, RX 7900 XTX, and other ROCm-compatible GPUs