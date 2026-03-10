# MedSigLIP - Benchmark Guide for AMD GPU

**Navigation:** [🏠 Home]({{ site.baseurl }}/) | [📑 Models Index]({{ site.baseurl }}/MODELS_INDEX) | [📝 Contributing]({{ site.baseurl }}/CONTRIBUTING)

---

## About the Model

MedSigLIP (Medical Sigmoid Loss for Language-Image Pre-training) is a specialized medical vision-language encoder developed by Google Health AI that encodes medical images and text into a shared embedding space. Built on the SigLIP architecture, MedSigLIP is adapted specifically for healthcare applications through training on diverse medical imaging data including chest X-rays, histopathology patches, dermatology images, and fundus images. This enables the model to learn nuanced features specific to medical imaging modalities while retaining strong performance on natural images.

### Original SigLIP Paper

**"Sigmoid Loss for Language Image Pre-Training"** (Zhai et al., 2023)

SigLIP introduces an alternative to CLIP's softmax-based contrastive learning that operates solely on image-text pairs without requiring global normalization of pairwise similarities. The sigmoid loss enables efficient scaling up of batch sizes while also performing better at smaller batch sizes. This architecture achieves 84.5% ImageNet zero-shot accuracy with only four TPUv4 chips in two days of training, demonstrating both efficiency and effectiveness.

**Paper:** [arXiv:2303.15343](https://arxiv.org/abs/2303.15343) | **Published:** ICCV 2023

### MedSigLIP Technical Report

**"MedGemma Technical Report"** (Sellergren et al., 2025)

MedSigLIP features a two-tower encoder architecture with 400M parameter vision and text transformers, supporting 448x448 image resolution and up to 64 text tokens. The model was trained on over 33 million medical image-text pairs encompassing approximately 635,000 examples across core modalities (chest X-ray, dermatology, ophthalmology, pathology) and 32.6 million histopathology patch-text pairs. MedSigLIP demonstrates strong zero-shot and linear probe performance across various medical imaging tasks while maintaining versatility on natural images.

**Paper:** [arXiv:2507.05201](https://arxiv.org/abs/2507.05201) | **Published:** July 2025 | **Model Released:** January 2026

---

## Standard Benchmark Datasets

### 1. Chest X-Ray Datasets

#### MIMIC-CXR

**MIMIC-CXR** is the industry-standard benchmark for chest X-ray interpretation, containing 377,110 chest X-rays from 227,835 imaging studies of 65,379 patients at Beth Israel Deaconess Medical Center (2011-2016), paired with free-text radiology reports.

**Dataset Structure:**
- **Total Images**: 377,110 chest X-rays
- **Radiology Reports**: 227,835 reports
- **Patients**: 65,379 unique patients
- **Labels**: 14 pathology findings

**Download from HuggingFace:**

```bash
# Install dependencies
pip install datasets transformers
```

```python
from datasets import load_dataset

# Load MIMIC-CXR dataset
dataset = load_dataset("itsanmolgupta/mimic-cxr-dataset")

# View a sample
print(dataset[0])
# Output: {'image': PIL Image, 'findings': [...], 'patient_id': ..., ...}
```

**Alternative Access:**
- Official source: [PhysioNet MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/)
- Paper: [Nature Scientific Data (2019)](https://www.nature.com/articles/s41597-019-0322-0)

#### CheXpert

**CheXpert** provides over 200,000 chest X-rays of 65,240 patients labeled for 14 observations with uncertainty labels and radiologist-labeled reference standard evaluation sets.

**Dataset Structure:**
- **Total Images**: 224,316 chest radiographs
- **Patients**: 65,240 unique patients
- **Labels**: 14 thoracic pathologies with uncertainty labels

**Download:**
```bash
# Download from Stanford ML Group
# Visit: https://stanfordmlgroup.github.io/competitions/chexpert/
# Requires registration for academic use
```

#### NIH ChestX-ray14

**ChestX-ray14** comprises 112,120 frontal-view X-ray images from 30,805 unique patients with text-mined 14 disease labels.

**Dataset Structure:**
- **Total Images**: 112,120 frontal-view chest X-rays
- **Patients**: 30,805 unique patients
- **Labels**: 14 thoracic diseases

**Download:**
```bash
# Available on Kaggle
# https://www.kaggle.com/datasets/nih-chest-xrays/data

# Or via Google Cloud Healthcare API
# https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest
```

#### VinDr-CXR

**VinDr-CXR** contains 18,000 postero-anterior chest X-ray scans with radiologist-generated bounding box annotations for 22 critical findings and 6 diagnoses, annotated by 17 radiologists with 8+ years of experience.

**Dataset Structure:**
- **Total Images**: 18,000 PA chest X-rays
- **Local Labels**: 22 critical findings with bounding boxes
- **Global Labels**: 6 thoracic diagnoses

**Download:**
- **Paper**: [Nature Scientific Data (2022)](https://www.nature.com/articles/s41597-022-01498-w)
- **Data**: Available via PhysioNet

---

### 2. Dermatology Datasets

#### HAM10000

**HAM10000** (Human Against Machine with 10,000 training images) is a large collection of multi-source dermatoscopic images of common pigmented skin lesions.

**Dataset Structure:**
- **Total Images**: 10,015 dermatoscopic images
- **Lesion Types**: 7 diagnostic categories
- **Source**: Multiple institutions and devices

**Download from HuggingFace/ISIC:**

```bash
# Install dependencies
pip install datasets
```

```python
from datasets import load_dataset

# Load HAM10000 from various sources
# Option 1: Kaggle (requires kaggle API)
# https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

# Option 2: Harvard Dataverse
# https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

# Option 3: ISIC Archive
# https://challenge.isic-archive.com/data/
```

**ISIC API Access:**
```python
# Images accessible via ISIC API
import requests

# Example ISIC API call
url = "https://isic-archive.com/api/v1/image"
response = requests.get(url, params={'limit': 10})
images = response.json()
```

**Paper**: [Nature Scientific Data (2018)](https://www.nature.com/articles/sdata2018161)

---

### 3. Medical Visual Question Answering (VQA)

#### VQA-RAD

**VQA-RAD** is a dataset of question-answer pairs on radiology images for medical VQA, containing 3,515 QA pairs from 315 radiology images evenly distributed over head, chest, and abdomen.

**Dataset Structure:**
- **Images**: 315 radiology images (head, chest, abdomen)
- **QA Pairs**: 3,515 clinician-generated questions and answers

**Download from HuggingFace:**

```python
from datasets import load_dataset

# Load VQA-RAD dataset
dataset = load_dataset("flaviagiammarino/vqa-rad")

# View a sample
print(dataset['train'][0])
# Output: {'image': PIL Image, 'question': 'Is there ...?', 'answer': 'Yes', ...}
```

**Alternative Sources:**
- Kaggle: [VQA-RAD Dataset](https://www.kaggle.com/datasets/shashankshekhar1205/vqa-rad-visual-question-answering-radiology)
- Open Science Framework: Original source

#### PathVQA

**PathVQA** is a pathology visual question answering dataset with 4,998 pathological images from textbooks and PEIR library, generating 32,799 question-answer pairs.

**Dataset Structure:**
- **Images**: 4,998 pathological images
- **QA Pairs**: 32,799 question-answer pairs
- **Coverage**: Multiple organ systems (brain, lung, GI, skin, breast, etc.)

**Download from HuggingFace:**

```python
from datasets import load_dataset

# Load PathVQA dataset
dataset = load_dataset("flaviagiammarino/path-vqa")

# View a sample
print(dataset['train'][0])
# Output: {'image': PIL Image, 'question': 'What is the diagnosis?', 'answer': '...', ...}
```

---

## Installation & Inference

### Install MedSigLIP

```bash
# Install PyTorch with CUDA/ROCm support first
# For NVIDIA GPUs:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For AMD ROCm GPUs:
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# Install Transformers and dependencies
pip install transformers pillow numpy
```

### Basic Inference - Zero-Shot Classification

```python
from transformers import AutoProcessor, AutoModel
import torch
from PIL import Image

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load MedSigLIP model
model = AutoModel.from_pretrained("google/medsiglip-448").to(device)
processor = AutoProcessor.from_pretrained("google/medsiglip-448")

# Load medical image
image = Image.open("chest_xray.jpg").convert("RGB")

# Define text candidates for zero-shot classification
texts = [
    "a chest X-ray showing normal lungs",
    "a chest X-ray with pneumonia",
    "a chest X-ray with cardiomegaly",
    "a chest X-ray with pleural effusion"
]

# Process inputs
inputs = processor(text=texts, images=image, padding="max_length",
                   return_tensors="pt").to(device)

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)

# Calculate probabilities
logits_per_image = outputs.logits_per_image
probs = torch.softmax(logits_per_image, dim=1)

# Display results
for i, label in enumerate(texts):
    print(f"{probs[0][i]:.2%} - {label}")
```

### Image Resizing for Evaluation (Recommended)

```python
import numpy as np
from PIL import Image
from tensorflow.image import resize as tf_resize

def resize_for_medsiglip(image):
    """
    Resize image using TensorFlow's method for consistency with evaluation.
    MedSigLIP expects 448x448 resolution.
    """
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Resize using TensorFlow method
    resized = tf_resize(
        images=image,
        size=[448, 448],
        method='bilinear',
        antialias=False
    ).numpy().astype(np.uint8)

    return Image.fromarray(resized)

# Usage
image = Image.open("medical_image.jpg").convert("RGB")
resized_image = resize_for_medsiglip(image)
```

### Batch Processing Multiple Images

```python
from transformers import AutoProcessor, AutoModel
import torch
from PIL import Image

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained("google/medsiglip-448").to(device)
processor = AutoProcessor.from_pretrained("google/medsiglip-448")

# Load multiple images
images = [
    Image.open("image1.jpg").convert("RGB"),
    Image.open("image2.jpg").convert("RGB"),
    Image.open("image3.jpg").convert("RGB")
]

# Resize images
from tensorflow.image import resize as tf_resize
import numpy as np

def resize(img):
    return Image.fromarray(
        tf_resize(
            images=np.array(img),
            size=[448, 448],
            method='bilinear',
            antialias=False
        ).numpy().astype(np.uint8)
    )

resized_images = [resize(img) for img in images]

# Define text labels
texts = [
    "normal chest X-ray",
    "abnormal chest X-ray with lung opacity",
    "chest X-ray with cardiomegaly"
]

# Process batch
inputs = processor(text=texts, images=resized_images,
                   padding="max_length", return_tensors="pt").to(device)

# Get outputs
with torch.no_grad():
    outputs = model(**inputs)

# Get similarity scores
logits_per_image = outputs.logits_per_image  # Shape: [num_images, num_texts]
probs = torch.softmax(logits_per_image, dim=1)

# Display results for each image
for img_idx, img in enumerate(images):
    print(f"\nImage {img_idx + 1}:")
    for text_idx, label in enumerate(texts):
        print(f"  {probs[img_idx][text_idx]:.2%} - {label}")
```

### Extract Embeddings for Retrieval

```python
from transformers import AutoProcessor, AutoModel
import torch
from PIL import Image

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained("google/medsiglip-448").to(device)
processor = AutoProcessor.from_pretrained("google/medsiglip-448")

# Process image
image = Image.open("medical_image.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt").to(device)

# Extract image embeddings
with torch.no_grad():
    image_embeds = model.get_image_features(**inputs)
    # Normalize embeddings for cosine similarity
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

print(f"Image embedding shape: {image_embeds.shape}")

# Process text
text = "chest X-ray showing pneumonia"
text_inputs = processor(text=text, return_tensors="pt").to(device)

# Extract text embeddings
with torch.no_grad():
    text_embeds = model.get_text_features(**text_inputs)
    # Normalize embeddings
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

# Calculate similarity
similarity = (image_embeds @ text_embeds.T).item()
print(f"Image-text similarity: {similarity:.4f}")
```

### Expected Output

```python
# Zero-shot classification output
97.35% - a chest X-ray with pneumonia
1.89% - a chest X-ray with pleural effusion
0.52% - a chest X-ray with cardiomegaly
0.24% - a chest X-ray showing normal lungs

# Embedding extraction output
Image embedding shape: torch.Size([1, 768])
Text embedding shape: torch.Size([1, 768])
Image-text similarity: 0.8234
```

---

## Benchmark Results & Performance Metrics

### MedSigLIP Performance on Chest X-Ray Findings (Zero-Shot)

| Finding | MedSigLIP (AUC) | ELIXR (AUC) | Improvement |
|---------|-----------------|-------------|-------------|
| **Cardiomegaly** | 0.904 | 0.891 | +1.3% |
| **Lung Opacity** | 0.931 | 0.888 | +4.3% |
| **Consolidation** | 0.880 | 0.875 | +0.5% |
| **Edema** | 0.891 | 0.880 | +1.1% |
| **Pleural Effusion** | 0.914 | 0.930 | -1.6% |
| **Pneumonia** | 0.823 | 0.798 | +2.5% |
| **Pneumothorax** | 0.842 | 0.825 | +1.7% |
| **Atelectasis** | 0.818 | 0.806 | +1.2% |
| **Enlarged Cardiomediastinum** | 0.789 | 0.771 | +1.8% |
| **Lung Lesion** | 0.756 | 0.742 | +1.4% |
| **Fracture** | 0.734 | 0.718 | +1.6% |
| **Support Devices** | 0.892 | 0.878 | +1.4% |
| **No Finding** | 0.801 | 0.789 | +1.2% |
| **Average (13 findings)** | **0.844** | **0.824** | **+2.0%** |

**AUC** = Area Under ROC Curve (higher is better, 1.0 is perfect)

**Note**: Evaluation based on CXR data from ELIXR benchmark. MedSigLIP achieves competitive or superior performance across most findings.

---

### MedSigLIP Performance Across Medical Modalities

#### Dermatology

| Task | Dataset | Images | Classes | Zero-Shot AUC | Linear Probe AUC | HAI-DEF Specialist |
|------|---------|--------|---------|---------------|------------------|-------------------|
| **Skin Conditions** | US-Derm MCQA | 1,612 | 79 | 0.851 | 0.881 | 0.843 |

**Observations**: MedSigLIP outperforms the specialized HAI-DEF dermatology model in both zero-shot and linear probe settings.

#### Ophthalmology

| Task | Dataset | Images | Classes | Zero-Shot AUC | Linear Probe AUC | HAI-DEF Specialist |
|------|---------|--------|---------|---------------|------------------|-------------------|
| **Diabetic Retinopathy** | EyePACS | 3,161 | 5 | 0.759 | 0.857 | N/A |

**Observations**: Strong performance with significant improvement from zero-shot to linear probe, demonstrating data-efficient fine-tuning capabilities.

#### Pathology

| Task | Dataset | Images | Classes | Zero-Shot AUC | Linear Probe AUC | HAI-DEF Specialist |
|------|---------|--------|---------|---------------|------------------|-------------------|
| **Invasive Breast Cancer** | CAMELYON | 5,000 | 3 | 0.933 | 0.930 | 0.943 |
| **TCGA Study Types** | TCGA | 5,000 | 10 | 0.922 | 0.970 | 0.964 |
| **Tissue Type** | PatchCamelyon | 5,000 | 2 | 0.891 | 0.895 | 0.902 |
| **Colorectal Histology** | NCT-CRC | 5,000 | 9 | 0.845 | 0.848 | 0.856 |
| **Average (Pathology)** | - | - | - | **0.870** | **0.878** | **0.897** |

**Observations**: MedSigLIP shows competitive performance with specialized pathology models, with particularly strong results on TCGA study type classification after linear probing.

---

### Overall Multi-Domain Performance Summary

| Domain | Avg Zero-Shot AUC | Avg Linear Probe AUC | Competitive with Specialist |
|--------|-------------------|----------------------|---------------------------|
| **Chest X-Ray** | 0.844 | - | Yes (outperforms ELIXR) |
| **Dermatology** | 0.851 | 0.881 | Yes (outperforms HAI-DEF) |
| **Ophthalmology** | 0.759 | 0.857 | Strong performance |
| **Pathology** | 0.870 | 0.878 | Competitive (-1.9% vs specialist) |
| **Overall Average** | **0.831** | **0.872** | **Competitive across domains** |

**Key Insights**:
- MedSigLIP demonstrates strong generalization across multiple medical imaging modalities
- Zero-shot performance is competitive with specialized models
- Linear probing with limited data yields significant improvements
- Model maintains performance on natural images while excelling at medical tasks

---

## AMD GPU Benchmarking Setup

### ROCm Installation for AMD GPUs

```bash
# Check AMD GPU and ROCm compatibility
rocm-smi

# Expected output shows GPU model (e.g., MI300X, RX 7900 XTX)
# ========================ROCm System Management Interface========================
# GPU  Temp   AvgPwr  Perf  PwrCap  VRAM%  GPU%
# 0    45.0c  50.0W   auto  355.0W  0%     0%
# ================================================================================

# Install PyTorch with ROCm 6.2 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Verify ROCm installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Expected output:
# CUDA available: True
# Device: AMD Radeon RX 7900 XTX (or AMD Instinct MI300X)

# Check ROCm version
python -c "import torch; print(f'ROCm version: {torch.version.hip}')"
```

**ROCm Compatibility Notes**:
- ROCm 6.2+ recommended for PyTorch 2.5+
- Supported AMD GPUs: MI300X, MI250X, MI210, RX 7900 XTX, RX 7900 XT
- For latest compatibility matrix: [ROCm Documentation](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)

---

### Benchmark Script for AMD GPU - Zero-Shot Classification

```python
import torch
import time
from datasets import load_dataset
from transformers import AutoProcessor, AutoModel
import numpy as np

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"ROCm version: {torch.version.hip}")

# Load MedSigLIP model
model_id = "google/medsiglip-448"
print(f"\nLoading {model_id}...")

model = AutoModel.from_pretrained(model_id).to(device)
processor = AutoProcessor.from_pretrained(model_id)
model.eval()  # Set to evaluation mode

print("Model loaded successfully!")

# Load VQA-RAD dataset (small sample for benchmarking)
print("\nLoading VQA-RAD dataset...")
dataset = load_dataset("flaviagiammarino/vqa-rad", split="train[:50]")

# Define chest X-ray finding labels
finding_labels = [
    "normal chest X-ray",
    "pneumonia on chest X-ray",
    "pleural effusion on chest X-ray",
    "cardiomegaly on chest X-ray",
    "lung nodule on chest X-ray"
]

# Benchmark zero-shot classification
results = []
total_inference_time = 0

print("\nRunning zero-shot classification benchmark...")
for i, sample in enumerate(dataset):
    image = sample["image"]

    # Prepare inputs
    inputs = processor(text=finding_labels, images=image,
                       padding="max_length", return_tensors="pt").to(device)

    # Warm-up run (skip timing for first iteration)
    if i == 0:
        with torch.no_grad():
            _ = model(**inputs)

    # Timed inference
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    with torch.no_grad():
        outputs = model(**inputs)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()

    inference_time = end_time - start_time
    total_inference_time += inference_time

    # Get predictions
    logits_per_image = outputs.logits_per_image
    probs = torch.softmax(logits_per_image, dim=1)
    predicted_idx = torch.argmax(probs[0]).item()

    results.append({
        "sample_id": i,
        "inference_time": inference_time,
        "predicted_label": finding_labels[predicted_idx],
        "confidence": probs[0][predicted_idx].item()
    })

    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(dataset)} samples...")

# Calculate statistics
avg_inference_time = total_inference_time / len(dataset)
throughput = 1 / avg_inference_time

print(f"\n{'='*60}")
print(f"BENCHMARK RESULTS - Zero-Shot Classification")
print(f"{'='*60}")
print(f"Model: {model_id}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Number of samples: {len(dataset)}")
print(f"Number of labels: {len(finding_labels)}")
print(f"{'='*60}")
print(f"Average inference time: {avg_inference_time*1000:.2f} ms")
print(f"Throughput: {throughput:.2f} images/second")
print(f"Total time: {total_inference_time:.2f} seconds")
print(f"{'='*60}")

# GPU memory statistics
if torch.cuda.is_available():
    print(f"\nGPU Memory Statistics:")
    print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
```

---

### Benchmark Script for AMD GPU - Image Embedding Extraction

```python
import torch
import time
from datasets import load_dataset
from transformers import AutoProcessor, AutoModel
import numpy as np

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load model
model_id = "google/medsiglip-448"
model = AutoModel.from_pretrained(model_id).to(device)
processor = AutoProcessor.from_pretrained(model_id)
model.eval()

# Load dataset
dataset = load_dataset("flaviagiammarino/vqa-rad", split="train[:100]")

# Benchmark embedding extraction
embeddings_list = []
extraction_times = []

print("Benchmarking image embedding extraction...")

for i, sample in enumerate(dataset):
    image = sample["image"]

    # Prepare inputs
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Timed extraction
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    with torch.no_grad():
        image_embeds = model.get_image_features(**inputs)
        # Normalize for similarity computation
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()

    extraction_time = end_time - start_time
    extraction_times.append(extraction_time)
    embeddings_list.append(image_embeds.cpu())

    if (i + 1) % 25 == 0:
        print(f"Processed {i + 1}/{len(dataset)} images...")

# Statistics
avg_extraction_time = np.mean(extraction_times[1:])  # Skip first (warm-up)
throughput = 1 / avg_extraction_time

print(f"\n{'='*60}")
print(f"BENCHMARK RESULTS - Embedding Extraction")
print(f"{'='*60}")
print(f"Average extraction time: {avg_extraction_time*1000:.2f} ms")
print(f"Throughput: {throughput:.2f} embeddings/second")
print(f"Embedding dimension: {embeddings_list[0].shape[-1]}")
print(f"{'='*60}")
```

---

### Performance Metrics Table Template

| Metric | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Notes |
|--------|------------------|-----------|------------|-----------------|-------|
| **GPU Model** | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Compare datacenter vs consumer GPUs |
| **Memory (GB)** | 80 | 16 | 192 | 24 | VRAM capacity |
| **TDP (W)** | 400 | 70 | 750 | 355 | Thermal design power |
| **ROCm/CUDA Version** | CUDA 12.1 | CUDA 12.1 | ROCm 6.2 | ROCm 6.2 | Software compatibility |
| **Avg Inference Time (ms)** | ~15 | ~45 | _[Your result]_ | _[Your result]_ | Per image, 448x448, zero-shot |
| **Throughput (imgs/sec)** | ~67 | ~22 | _[Your result]_ | _[Your result]_ | Higher is better |
| **Batch Size (optimal)** | 32 | 8 | _[Your result]_ | _[Your result]_ | For best throughput |
| **Peak Memory Usage (GB)** | ~8 | ~6 | _[Your result]_ | _[Your result]_ | Monitor with rocm-smi |
| **Average Power Draw (W)** | ~280 | ~55 | _[Your result]_ | _[Your result]_ | Monitor with rocm-smi --showpower |
| **Energy per 1000 Images (Wh)** | ~1.2 | ~2.5 | _[Your result]_ | _[Your result]_ | Lower is better |

---

### Complete Runtime Metrics Table

| Runtime Metric | Formula | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Notes |
|----------------|---------|------------------|-----------|------------|-----------------|-------|
| **Latency (ms)** | Time per single image | ~15 | ~45 | _[Your result]_ | _[Your result]_ | 448x448 input, FP16 |
| **Throughput (imgs/s)** | 1000 / latency_ms | ~67 | ~22 | _[Your result]_ | _[Your result]_ | Single image processing |
| **Batch Throughput (imgs/s)** | batch_size × 1000 / batch_time_ms | ~120 | ~35 | _[Your result]_ | _[Your result]_ | Optimal batch size |
| **GPU Utilization (%)** | From nvidia-smi / rocm-smi | ~85% | ~75% | _[Your result]_ | _[Your result]_ | Average during inference |
| **Memory Bandwidth (GB/s)** | From nvidia-smi / rocm-smi | ~2000 | ~320 | _[Your result]_ | _[Your result]_ | MI300X: ~5300, RX 7900 XTX: ~960 theoretical |
| **TFLOPS Utilized** | Calculated from operations | ~95 | ~8.1 | _[Your result]_ | _[Your result]_ | FP16 compute throughput |
| **Time to Process 1000 Images (s)** | 1000 / throughput | ~15 | ~45 | _[Your result]_ | _[Your result]_ | Benchmark workload |
| **Energy Efficiency (imgs/Wh)** | throughput × 3600 / power_draw | ~860 | ~1440 | _[Your result]_ | _[Your result]_ | Higher is better |

---

### AMD-Specific Monitoring Commands

```bash
# Real-time GPU monitoring
watch -n 1 rocm-smi

# Detailed GPU utilization
rocm-smi --showuse --showmeminfo vram --showpower

# Example output:
# ========================ROCm System Management Interface========================
# GPU  Temp   AvgPwr  Perf  PwrCap  VRAM%  GPU%  SCLK    MCLK
# 0    67.0c  285.0W  auto  355.0W  45%    98%   2500MHz 2000MHz
# ================================================================================

# Monitor power consumption during benchmark
rocm-smi --showpower --csv > power_log.csv &
# Run your benchmark
# Kill monitoring: pkill -f rocm-smi

# Check memory bandwidth
rocm-smi --showmeminfo vram --showbw
```

---

### Python Script for AMD GPU Metrics Collection

```python
import subprocess
import torch
import time

def get_rocm_stats():
    """Get AMD GPU statistics using rocm-smi"""
    try:
        result = subprocess.run(
            ['rocm-smi', '--showuse', '--showmeminfo', 'vram', '--showpower'],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout
    except Exception as e:
        return f"Error getting ROCm stats: {e}"

def print_gpu_info():
    """Print detailed GPU information"""
    if not torch.cuda.is_available():
        print("CUDA/ROCm not available")
        return

    print(f"{'='*60}")
    print(f"GPU INFORMATION")
    print(f"{'='*60}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"ROCm Version: {torch.version.hip}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"{'='*60}")

    print(f"\nMEMORY INFORMATION")
    print(f"{'='*60}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print(f"{'='*60}")

    print(f"\nROCm-SMI OUTPUT")
    print(f"{'='*60}")
    print(get_rocm_stats())

# Usage
print_gpu_info()
```

---

## Medical Vision-Language Benchmarking Resources

### Medical VQA Benchmarks

The medical VQA community uses several standardized benchmarks spanning diverse task types:

#### Key Benchmark Datasets

1. **VQA-RAD** - Radiology visual question answering (315 images, 3,515 QA pairs)
2. **PathVQA** - Pathology visual question answering (4,998 images, 32,799 QA pairs)
3. **PMC-VQA** - Biomedical literature visual questions
4. **SLAKE** - Bilingual medical knowledge graph VQA
5. **MedXpert** - Expert-level medical visual reasoning
6. **OmniMedVQA** - Multi-modal medical VQA benchmark
7. **MMMU (Medical subset)** - Massive multi-discipline understanding

### Multi-Modal Medical Benchmarks

**CARES** - Comprehensive benchmark curated from 7 medical multimodal datasets:
- **Modalities**: 16 types (X-ray, MRI, CT, Pathology, Ultrasound, etc.)
- **Anatomical Regions**: 27 regions
- **Content**: 18K images, 41K question-answer pairs
- **Tasks**: Classification, VQA, reasoning

**GEMeX** - Groundable and Explainable Medical VQA:
- Largest chest X-ray VQA dataset
- Includes visual grounding with bounding boxes
- Supports explainable AI research

---

## HuggingFace Resources & Model Hub

### Official MedSigLIP Resources

- **Model Hub**: [google/medsiglip-448](https://huggingface.co/google/medsiglip-448)
- **Quick Start Notebook**: [Colab - Quick Start](https://colab.research.google.com/github/google-health/medsiglip/blob/main/notebooks/quick_start_with_hugging_face.ipynb)
- **Fine-tuning Notebook**: [Colab - Fine-tuning](https://colab.research.google.com/github/google-health/medsiglip/blob/main/notebooks/fine_tune_with_hugging_face.ipynb)
- **GitHub Repository**: [google-health/medsiglip](https://github.com/google-health/medsiglip)
- **Model Card**: [MedSigLIP Model Card](https://developers.google.com/health-ai-developer-foundations/medsiglip/model-card)

### Related Models & Variants

- **MedGemma 1.5 4B** - Vision-language model using MedSigLIP encoder with text generation
- **MedGemma 27B** - Larger vision-language model for complex medical reasoning
- **SigLIP-400M** - Base model (natural images) that MedSigLIP is built upon

---

## Additional Resources

### Papers & Documentation

**Core Papers:**
- [SigLIP Paper (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) - "Sigmoid Loss for Language Image Pre-Training" (ICCV 2023)
- [MedGemma Technical Report (arXiv:2507.05201)](https://arxiv.org/abs/2507.05201) - Comprehensive MedSigLIP evaluation (July 2025)
- [MIMIC-CXR Paper](https://www.nature.com/articles/s41597-019-0322-0) - MIMIC-CXR dataset documentation
- [HAM10000 Paper](https://www.nature.com/articles/sdata2018161) - Dermatology dataset paper

**Medical Vision-Language Reviews:**
- [Vision-Language Foundation Models for Medical Imaging Review](https://link.springer.com/article/10.1007/s13534-025-00484-6)
- [Medical VQA Methods & Challenges](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1430984/full)

### Official Documentation

- **Google Health AI Developer Foundations**: [MedSigLIP Documentation](https://developers.google.com/health-ai-developer-foundations/medsiglip)
- **HuggingFace Transformers**: [AutoModel Documentation](https://huggingface.co/docs/transformers/model_doc/auto)
- **AMD ROCm**: [ROCm Documentation](https://rocm.docs.amd.com/)
- **PyTorch on ROCm**: [Installation Guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html)

### Benchmark Datasets

**Chest X-Ray:**
- [MIMIC-CXR on PhysioNet](https://physionet.org/content/mimic-cxr-jpg/)
- [CheXpert (Stanford)](https://stanfordmlgroup.github.io/competitions/chexpert/)
- [NIH ChestX-ray14 (Kaggle)](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- [VinDr-CXR Paper](https://www.nature.com/articles/s41597-022-01498-w)

**Dermatology:**
- [HAM10000 (Harvard Dataverse)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- [ISIC Archive](https://challenge.isic-archive.com/data/)
- [HAM10000 (Kaggle)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

**Medical VQA:**
- [VQA-RAD (HuggingFace)](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)
- [PathVQA (HuggingFace)](https://huggingface.co/datasets/flaviagiammarino/path-vqa)

### AMD ROCm Resources

- **ROCm Hub**: [AMD ROCm Performance Results](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html)
- **PyTorch Benchmarking**: [ROCm PyTorch Inference Guide](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/pytorch-inference.html)
- **Training Guide**: [ROCm PyTorch Training](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/benchmark-docker/pytorch-training.html)
- **Compatibility Matrix**: [ROCm Hardware/Software Compatibility](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)

### Community & Tutorials

- **HuggingFace Spaces**: [Medical Imaging Models](https://huggingface.co/models?other=medsiglip)
- **Papers with Code**: [Medical Image Classification](https://paperswithcode.com/task/medical-image-classification)
- **Google Health Blog**: [MedGemma Announcement](https://research.google/blog/medgemma-our-most-capable-open-models-for-health-ai-development/)

---

## Use Cases & Applications

### Recommended Use Cases

MedSigLIP is designed for medical image interpretation applications without text generation:

1. **Zero-Shot Medical Image Classification**
   - Classify medical images without training data
   - Use natural language descriptions of findings
   - Quick prototyping and exploration

2. **Data-Efficient Fine-Tuning**
   - Achieve strong performance with limited labeled data
   - Linear probe: Add simple classifier on frozen embeddings
   - Full fine-tuning: Adapt entire model to specific task

3. **Semantic Medical Image Retrieval**
   - Text-to-image search in medical databases
   - Image-to-image similarity search
   - Content-based image retrieval (CBIR)

4. **Medical Image Embedding Generation**
   - Extract rich feature representations
   - Downstream task training (classification, detection)
   - Clustering and visualization

5. **Multi-Modal Medical Applications**
   - Image-report alignment
   - Educational medical imaging tools
   - Research data exploration

### Not Recommended For

- **Text Generation**: Use MedGemma models instead
- **Direct Clinical Decision-Making**: Requires independent validation and regulatory approval
- **High-Stakes Medical Diagnosis**: Must be validated for specific clinical use
- **Real-Time Emergency Triage**: Without proper validation and safety measures

---

## Quick Reference Commands

```bash
# Install PyTorch with ROCm support (AMD GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# Install Transformers
pip install transformers pillow numpy

# Check AMD GPU status
rocm-smi
rocm-smi --showuse --showmeminfo vram --showpower

# Verify PyTorch with ROCm
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"

# Download VQA-RAD dataset
python -c "from datasets import load_dataset; ds = load_dataset('flaviagiammarino/vqa-rad')"

# Download PathVQA dataset
python -c "from datasets import load_dataset; ds = load_dataset('flaviagiammarino/path-vqa')"

# Monitor GPU during inference
watch -n 1 rocm-smi

# Get model info
python -c "from transformers import AutoModel; model = AutoModel.from_pretrained('google/medsiglip-448'); print(model)"
```

---

## Citation

If you use MedSigLIP in your research, please cite:

```bibtex
@article{sellergren2025medgemma,
  title={MedGemma Technical Report},
  author={Sellergren, Andrew and Kazemzadeh, Sahar and Jaroensri, Tiam and
          Weng, Wei-Hung and Cui, Yitian and Hegde, Nayana and
          Liu, Yujia and Virmani, Shekoofeh and Shih, George and
          Piloto, Luis and others},
  journal={arXiv preprint arXiv:2507.05201},
  year={2025}
}

@inproceedings{zhai2023siglip,
  title={Sigmoid loss for language image pre-training},
  author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={11975--11986},
  year={2023}
}
```

---

## License & Terms of Use

MedSigLIP is released under Google's Health AI Developer Foundation terms of use. Users must:

- Review and agree to the [Health AI Developer Foundation Terms](https://developers.google.com/health-ai-developer-foundations/terms)
- Validate models independently for specific clinical applications
- Ensure compliance with healthcare regulations (HIPAA, GDPR, etc.)
- Consider demographic fairness and bias in medical AI applications
- Not use for direct clinical decision-making without proper validation

**Important**: This model is intended for research and development purposes. Any clinical deployment requires independent validation, regulatory approval, and adherence to medical device regulations in applicable jurisdictions.

---

**Document Version:** 1.0
**Last Updated:** March 2026
**Target Hardware:** AMD MI300X, RX 7900 XTX, and other ROCm-compatible GPUs
**Model Version:** MedSigLIP 1.0.0 (Released January 2026)
