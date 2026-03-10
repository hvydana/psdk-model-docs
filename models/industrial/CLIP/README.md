# CLIP - Benchmark Guide for AMD GPU

## About the Model

CLIP (Contrastive Language-Image Pretraining) is a neural network trained on 400 million (image, text) pairs that can be instructed in natural language to predict the most relevant text snippet given an image, without directly optimizing for the task. CLIP learns visual concepts from natural language supervision, enabling zero-shot transfer to downstream tasks without task-specific training data. The model bridges computer vision and natural language processing by learning a shared embedding space where images and their textual descriptions are aligned.

### Original CLIP Paper

**"Learning Transferable Visual Models From Natural Language Supervision"** (Radford et al., 2021)

CLIP demonstrates that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. The model achieves 76.2% top-1 accuracy on ImageNet zero-shot without using any of the 1.28 million training examples, matching the performance of the original supervised ResNet-50.

**Paper:** [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) | **Published:** ICML 2021

---

## Standard Benchmark Datasets

CLIP is typically evaluated on multiple vision benchmarks for zero-shot classification and image-text retrieval tasks.

### 1. ImageNet-1K (Primary Classification Benchmark)

**ImageNet** is the industry-standard benchmark for evaluating image classification systems. It contains 1.28 million training images and 50,000 validation images across 1,000 object categories.

#### Dataset Structure
- **Validation set**: 50,000 images across 1,000 classes
- **Image resolution**: Variable (typically resized to 224x224 or 384x384)
- **Task**: Zero-shot image classification

#### Download from HuggingFace

```bash
# Install dependencies
pip install datasets transformers pillow
```

```python
from datasets import load_dataset

# Load ImageNet-1K validation set
# Note: ImageNet requires manual download and agreement to terms
# You can use imagenet-1k on HuggingFace with proper authentication
dataset = load_dataset("imagenet-1k", split="validation")

# View a sample
print(dataset[0])
# Output: {'image': <PIL.Image>, 'label': 123}

# For zero-shot evaluation, you'll need class names
imagenet_classes = dataset.features['label'].names
print(f"Number of classes: {len(imagenet_classes)}")
```

### 2. COCO Caption (Image-Text Retrieval)

**COCO Caption** contains over 330,000 images with 5 human-generated captions per image, totaling 1.5 million captions. It's widely used for evaluating image-text retrieval performance.

#### Dataset Structure
- **Validation set**: ~5,000 images with 5 captions each
- **Task**: Image-to-text and text-to-image retrieval

#### Download from HuggingFace

```python
from datasets import load_dataset

# Load COCO captions
dataset = load_dataset("HuggingFaceM4/COCO", split="validation")

# View a sample
print(dataset[0])
# Output: {'image': <PIL.Image>, 'sentences': {'raw': 'caption text', ...}}
```

### 3. Flickr30k (Image-Text Retrieval)

**Flickr30k** contains 31,783 images collected from Flickr, with 5 reference captions per image describing the depicted scenes, objects, and events.

#### Download from HuggingFace

```python
from datasets import load_dataset

# Load Flickr30k
dataset = load_dataset("nlphuji/flickr30k", split="test")

# View a sample
print(dataset[0])
# Output: {'image': <PIL.Image>, 'caption': ['caption1', 'caption2', ...]}
```

---

## Installation & Inference

### Install Required Libraries

```bash
# Install PyTorch with ROCm support (for AMD GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# Install Transformers and other dependencies
pip install transformers pillow requests datasets
```

### Basic Inference - Zero-Shot Image Classification

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests

# Load model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Define candidate labels
labels = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]

# Process inputs
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

# Get predictions
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

# Print results
for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob.item():.4f}")
```

### Advanced Inference - Image-Text Similarity

```python
from transformers import CLIPProcessor, CLIPModel
import torch

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Process multiple images and texts
images = [...]  # List of PIL Images
texts = ["a photo of a cat", "a photo of a dog", "a scenic landscape"]

inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

    # Normalize embeddings
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    # Calculate similarity
    similarity = (image_embeds @ text_embeds.T) * 100
    print(f"Similarity scores:\n{similarity}")
```

### Optimized Inference with Flash Attention

```python
from transformers import CLIPModel, CLIPProcessor
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load with optimizations
model = CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14",
    torch_dtype=torch_dtype,
    attn_implementation="sdpa"  # Scaled Dot Product Attention
).to(device)

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Enable inference mode
model.eval()

# Your inference code here
```

### Expected Output

```python
# Zero-shot classification output
{
  "predictions": [
    {"label": "a photo of a cat", "score": 0.9245},
    {"label": "a photo of a dog", "score": 0.0512},
    {"label": "a photo of a bird", "score": 0.0243}
  ]
}

# Image-text similarity output
{
  "similarity_matrix": [
    [92.5, 12.3, 8.1],   # Image 1 similarities with each text
    [15.2, 88.7, 9.3],   # Image 2 similarities with each text
    [7.8, 11.2, 91.5]    # Image 3 similarities with each text
  ]
}
```

---

## Benchmark Results & Performance Metrics

### CLIP Performance on ImageNet Zero-Shot Classification

| Model | ImageNet Top-1 | ImageNet Top-5 | Parameters | Training Data | Architecture |
|-------|----------------|----------------|------------|---------------|--------------|
| **CLIP ViT-L/14** | 75.5% | 95.3% | ~428M | 400M pairs | Vision Transformer Large |
| **CLIP ViT-B/32** | 63.2% | 87.8% | ~151M | 400M pairs | Vision Transformer Base |
| **CLIP ViT-B/16** | 68.3% | 90.1% | ~149M | 400M pairs | Vision Transformer Base |
| **CLIP ResNet-50** | 59.6% | 86.2% | ~102M | 400M pairs | ResNet-50 |
| **CLIP ResNet-101** | 62.3% | 87.6% | ~119M | 400M pairs | ResNet-101 |
| OpenCLIP ViT-H/14 | 78.0% | ~96% | ~632M | LAION-2B | Vision Transformer Huge |
| OpenCLIP ViT-g/14 | 80.1% | ~97% | ~1.8B | LAION-2B | Vision Transformer Giant |
| CLIPA-v2 H/14 | 81.1% | - | ~632M | DataComp-1B | Vision Transformer Huge |

**Note:** OpenCLIP and CLIPA are open-source implementations trained on larger datasets

### CLIP Performance on Image-Text Retrieval

#### COCO Caption Benchmark (5K test set)

| Model | Image→Text R@1 | Image→Text R@5 | Text→Image R@1 | Text→Image R@5 | Average |
|-------|----------------|----------------|----------------|----------------|---------|
| **CLIP ViT-L/14** | 58.4% | 81.5% | 37.8% | 63.4% | 60.3% |
| **CLIP ViT-B/32** | 48.2% | 73.1% | 30.1% | 54.8% | 51.6% |
| **CLIP ViT-B/16** | 52.7% | 76.9% | 33.6% | 58.2% | 55.4% |
| OpenCLIP ViT-H/14 | 63.5% | 85.2% | 42.1% | 68.9% | 64.9% |

**R@K** = Recall at K (percentage of correct matches in top-K predictions)

#### Flickr30k Benchmark (1K test set)

| Model | Image→Text R@1 | Image→Text R@5 | Text→Image R@1 | Text→Image R@5 | Average |
|-------|----------------|----------------|----------------|----------------|---------|
| **CLIP ViT-L/14** | 88.0% | 98.7% | 68.7% | 90.6% | 86.5% |
| **CLIP ViT-B/32** | 77.5% | 95.3% | 56.3% | 81.9% | 77.8% |
| **CLIP ViT-B/16** | 81.8% | 97.1% | 61.5% | 85.7% | 81.5% |

### Inference Speed Comparison

| Model | Batch Size | Images/Second | Latency (ms) | Platform | Notes |
|-------|------------|---------------|--------------|----------|-------|
| **CLIP ViT-B/32** | 1 | 145 | 6.9 | NVIDIA A100 | Fastest variant |
| **CLIP ViT-B/16** | 1 | 98 | 10.2 | NVIDIA A100 | Balanced speed/accuracy |
| **CLIP ViT-L/14** | 1 | 52 | 19.2 | NVIDIA A100 | Best accuracy |
| **CLIP ViT-B/32** | 32 | 2,850 | 11.2 (batch) | NVIDIA A100 | High throughput |
| **CLIP ViT-L/14** | 16 | 625 | 25.6 (batch) | NVIDIA A100 | Large model batched |

**Note:** Latency measured for single image inference; batch latency is time per batch / batch size

---

## AMD GPU Benchmarking Setup

### ROCm Installation for AMD GPUs

```bash
# Check ROCm compatibility and version
rocm-smi
rocminfo | grep "Name:"

# Install PyTorch with ROCm support
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}'); print(f'ROCm version: {torch.version.hip if torch.cuda.is_available() else \"N/A\"}')"
```

### Benchmark Script for AMD GPU - Zero-Shot Classification

```python
import torch
import time
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from tqdm import tqdm

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16

print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"ROCm Version: {torch.version.hip if torch.cuda.is_available() else 'N/A'}")

# Load model with optimizations
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    attn_implementation="sdpa"
).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

model.eval()

# Load ImageNet validation subset (or use your own dataset)
# Note: Full ImageNet requires authentication
print("Loading dataset...")
dataset = load_dataset("imagenet-1k", split="validation[:100]", use_auth_token=True)
class_names = dataset.features['label'].names

# Prepare text prompts for zero-shot classification
text_prompts = [f"a photo of a {class_name}" for class_name in class_names]

# Encode text prompts once (they're the same for all images)
print("Encoding text prompts...")
text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

# Benchmark image encoding and classification
print("Running benchmark...")
results = []
correct = 0
total = 0

for i, sample in enumerate(tqdm(dataset)):
    image = sample['image']
    true_label = sample['label']

    # Measure inference time
    start_time = time.time()

    # Process image
    image_inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        # Get image features
        image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        # Calculate similarity with all text prompts
        similarity = (image_features @ text_features.T) * 100
        predicted_label = similarity.argmax().item()

    end_time = time.time()
    inference_time = end_time - start_time

    # Track accuracy
    if predicted_label == true_label:
        correct += 1
    total += 1

    results.append({
        "sample_id": i,
        "inference_time_ms": inference_time * 1000,
        "predicted_label": predicted_label,
        "true_label": true_label,
        "correct": predicted_label == true_label
    })

# Calculate statistics
accuracy = correct / total
avg_inference_time = np.mean([r["inference_time_ms"] for r in results])
std_inference_time = np.std([r["inference_time_ms"] for r in results])
throughput = 1000 / avg_inference_time  # images per second

print(f"\n{'='*60}")
print(f"Benchmark Results")
print(f"{'='*60}")
print(f"Model: {model_id}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Samples: {total}")
print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
print(f"Average Inference Time: {avg_inference_time:.2f} ± {std_inference_time:.2f} ms")
print(f"Throughput: {throughput:.2f} images/second")
print(f"{'='*60}")

# Memory statistics
if torch.cuda.is_available():
    print(f"\nMemory Statistics:")
    print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
```

### Benchmark Script for AMD GPU - Image-Text Retrieval

```python
import torch
import time
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16

# Load model
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)
processor = CLIPProcessor.from_pretrained(model_id)
model.eval()

# Load COCO validation dataset
print("Loading COCO dataset...")
dataset = load_dataset("HuggingFaceM4/COCO", split="validation[:100]")

# Benchmark retrieval
print("Running image-text retrieval benchmark...")
start_time = time.time()

all_image_embeds = []
all_text_embeds = []

for sample in dataset:
    image = sample['image']
    captions = sample['sentences']['raw'][:1]  # Use first caption

    # Process inputs
    inputs = processor(text=captions, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)

        all_image_embeds.append(image_embeds.cpu())
        all_text_embeds.append(text_embeds.cpu())

# Stack embeddings
all_image_embeds = torch.cat(all_image_embeds, dim=0)
all_text_embeds = torch.cat(all_text_embeds, dim=0)

# Calculate similarity matrix
similarity_matrix = (all_image_embeds @ all_text_embeds.T) * 100

# Calculate Recall@K for image-to-text retrieval
def recall_at_k(similarity, k=1):
    ranks = similarity.argsort(descending=True, dim=1)
    correct = (ranks[:, :k] == torch.arange(len(similarity)).unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()

i2t_r1 = recall_at_k(similarity_matrix, k=1)
i2t_r5 = recall_at_k(similarity_matrix, k=5)
t2i_r1 = recall_at_k(similarity_matrix.T, k=1)
t2i_r5 = recall_at_k(similarity_matrix.T, k=5)

end_time = time.time()
total_time = end_time - start_time

print(f"\nRetrieval Benchmark Results:")
print(f"Image→Text R@1: {i2t_r1:.4f}")
print(f"Image→Text R@5: {i2t_r5:.4f}")
print(f"Text→Image R@1: {t2i_r1:.4f}")
print(f"Text→Image R@5: {t2i_r5:.4f}")
print(f"Total Time: {total_time:.2f} seconds")
print(f"Time per sample: {(total_time/len(dataset))*1000:.2f} ms")
```

### Performance Metrics Table Template

| Metric | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Notes |
|--------|------------------|-----------|------------|-----------------|-------|
| **GPU Model** | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Compare datacenter vs consumer GPUs |
| **Memory (GB)** | 80 | 16 | 192 | 24 | VRAM capacity |
| **TDP (W)** | 400 | 70 | 750 | 355 | Thermal design power |
| **Model** | ViT-B/32 | ViT-B/32 | _[Your result]_ | _[Your result]_ | CLIP model variant |
| **Batch Size** | 32 | 16 | _[Your result]_ | _[Your result]_ | Maximum stable batch size |
| **Inference Time (ms)** | 6.9 | 11.2 | _[Your result]_ | _[Your result]_ | Single image, batch size 1 |
| **Throughput (imgs/sec)** | 145 | 89 | _[Your result]_ | _[Your result]_ | Single image inference |
| **Batch Throughput (imgs/sec)** | 2,850 | 1,425 | _[Your result]_ | _[Your result]_ | Maximum batch size |
| **Peak Memory Usage (GB)** | ~8.5 | ~6.2 | _[Your result]_ | _[Your result]_ | Monitor with rocm-smi |
| **Average Power Draw (W)** | ~280 | ~55 | _[Your result]_ | _[Your result]_ | During inference |
| **Energy per 1K Images (Wh)** | ~0.54 | ~0.62 | _[Your result]_ | _[Your result]_ | Lower is better |
| **ImageNet Top-1 Accuracy** | 63.2% | 63.2% | _[Expected: 63.2%]_ | _[Expected: 63.2%]_ | Zero-shot accuracy |

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
        print("\nPower Consumption:")
        print(result.stdout)
    except FileNotFoundError:
        print("rocm-smi not found. Please ensure ROCm is installed.")

# PyTorch memory tracking
print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
print(f"Max Allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

# ROCm information
if torch.cuda.is_available():
    print(f"\nROCm Version: {torch.version.hip}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

# Get detailed stats
get_rocm_smi_stats()
```

### Complete Runtime Metrics Table

| Runtime Metric | Formula | NVIDIA A100-80GB | NVIDIA T4 | AMD MI300X | AMD RX 7900 XTX | Notes |
|----------------|---------|------------------|-----------|------------|-----------------|-------|
| **Latency (ms)** | Time for single image | 6.9 | 11.2 | _[Your result]_ | _[Your result]_ | Lower is better |
| **Throughput (imgs/sec)** | 1000 / latency | 145 | 89 | _[Your result]_ | _[Your result]_ | Higher is better |
| **Batch Latency (ms)** | Time for batch / batch_size | 11.2 | 17.8 | _[Your result]_ | _[Your result]_ | Batch size 32/16 |
| **Batch Throughput (imgs/sec)** | batch_size × 1000 / total_time | 2,850 | 1,425 | _[Your result]_ | _[Your result]_ | Maximum throughput |
| **GPU Utilization (%)** | From nvidia-smi / rocm-smi | ~85% | ~75% | _[Your result]_ | _[Your result]_ | Average during inference |
| **Memory Bandwidth (GB/s)** | From nvidia-smi / rocm-smi | ~2.0 TB/s | ~320 GB/s | _[Your result]_ | _[Your result]_ | MI300X: ~5.3 TB/s theoretical |
| **TFLOPS Utilized** | Calculated from operations | ~156 | ~65 | _[Your result]_ | _[Your result]_ | FP16 compute throughput |
| **Energy Efficiency (imgs/Wh)** | throughput / power_draw × 3600 | 1,863 | 5,818 | _[Your result]_ | _[Your result]_ | Higher is better |
| **Time to Process 10K Images (sec)** | 10000 / throughput | 69 | 112 | _[Your result]_ | _[Your result]_ | Practical benchmark |

---

## CLIP Leaderboards and Benchmarks

### CLIP Benchmark Repository

The [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark) repository provides standardized evaluation across multiple datasets:

#### Supported Datasets for Zero-Shot Classification
- **ImageNet-1K** - General object classification (1,000 classes)
- **ImageNet-V2** - Distribution shift robustness
- **ImageNet-Sketch** - Sketch recognition
- **ImageNet-A** - Adversarial examples
- **ImageNet-R** - Renditions and art
- **CIFAR-10** - 10 object categories
- **CIFAR-100** - 100 object categories
- **STL-10** - Higher resolution classification
- **PASCAL VOC 2007** - Multi-label classification
- **Caltech-101** - 101 object categories
- **Food-101** - Food classification
- **Oxford Pets** - Pet breed classification
- **Stanford Cars** - Car model classification
- **FGVC Aircraft** - Aircraft variant recognition

#### Supported Datasets for Retrieval
- **COCO Caption** - 5K test set
- **Flickr30k** - 1K test set
- **Flickr8k** - 1K test set

### Key Metrics Tracked

#### Classification Metrics
- **Top-1 Accuracy** - Primary metric for classification
- **Top-5 Accuracy** - Correct label in top 5 predictions
- **Mean Accuracy** - Average across all datasets

#### Retrieval Metrics
- **R@1** - Recall at 1 (top-1 retrieval accuracy)
- **R@5** - Recall at 5 (top-5 retrieval accuracy)
- **R@10** - Recall at 10 (top-10 retrieval accuracy)
- **Mean Recall** - Average of R@1, R@5, R@10

#### Efficiency Metrics
- **Inference Time** - Latency per image
- **Throughput** - Images processed per second
- **Model Size** - Number of parameters
- **VRAM Usage** - Peak GPU memory

### Running CLIP Benchmark

```bash
# Install CLIP benchmark
pip install clip-benchmark

# Run benchmark on ImageNet
clip_benchmark eval --dataset=imagenet1k \
                    --model=ViT-B-32 \
                    --pretrained=openai \
                    --output=results.json

# Run on multiple datasets
clip_benchmark eval --dataset=cifar10,cifar100,imagenet1k \
                    --model=ViT-B-32 \
                    --pretrained=openai \
                    --batch_size=64 \
                    --num_workers=4
```

---

## Additional Resources

### Official Repositories
- [OpenAI CLIP GitHub](https://github.com/openai/CLIP)
- [OpenCLIP (Open Source Implementation)](https://github.com/mlfoundations/open_clip)
- [CLIP Benchmark Repository](https://github.com/LAION-AI/CLIP_benchmark)
- [HuggingFace Transformers CLIP](https://huggingface.co/docs/transformers/model_doc/clip)

### Papers & Documentation
- [CLIP Paper (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020)
- [CLIP Paper (PDF)](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language.pdf)
- [CLIPA-v2: Scaling CLIP Training](https://arxiv.org/abs/2306.15658)
- [OpenCLIP: Reproducible scaling laws](https://arxiv.org/abs/2212.07143)

### Blog Posts & Tutorials
- [OpenAI: CLIP - Connecting text and images](https://openai.com/index/clip/)
- [HuggingFace and AMD Partnership](https://huggingface.co/blog/huggingface-and-amd)
- [Running HuggingFace Models on AMD ROCm](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/hugging-face-models.html)
- [Zero-Shot Image Classification with CLIP (Pinecone)](https://www.pinecone.io/learn/series/image-search/zero-shot-image-classification-clip/)
- [AMD ROCm Performance Results](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html)

### Pre-trained Models on HuggingFace
- [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) - 151M parameters
- [openai/clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16) - 149M parameters
- [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) - 428M parameters
- [laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) - 632M parameters
- [laion/CLIP-ViT-g-14-laion2B-s12B-b42K](https://huggingface.co/laion/CLIP-ViT-g-14-laion2B-s12B-b42K) - 1.8B parameters

### Datasets on HuggingFace
- [ImageNet-1K](https://huggingface.co/datasets/imagenet-1k) - Requires authentication
- [COCO Caption](https://huggingface.co/datasets/HuggingFaceM4/COCO)
- [Flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k)
- [CIFAR-10](https://huggingface.co/datasets/cifar10)
- [CIFAR-100](https://huggingface.co/datasets/cifar100)

### AMD ROCm Resources
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [PyTorch with ROCm](https://pytorch.org/get-started/locally/)
- [Optimum-AMD for HuggingFace](https://github.com/huggingface/optimum-amd)
- [TensorWave: PyTorch and HuggingFace with ROCm 7](https://tensorwave.com/blog/open-source-ai-frameworks-on-amd-how-to-use-pytorch-and-hugging-face-with-rocm-7)

---

## Quick Reference Commands

```bash
# Install PyTorch with ROCm support
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# Install Transformers and dependencies
pip install transformers pillow datasets

# Install CLIP Benchmark
pip install clip-benchmark

# Check AMD GPU status
rocm-smi
rocm-smi --showuse --showmeminfo vram --showpower

# Run zero-shot classification benchmark
python benchmark_classification.py

# Run image-text retrieval benchmark
python benchmark_retrieval.py

# Verify PyTorch + ROCm installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}, ROCm: {torch.version.hip if torch.cuda.is_available() else \"N/A\"}')"
```

---

**Document Version:** 1.0
**Last Updated:** March 2026
**Target Hardware:** AMD MI300X, RX 7900 XTX, and other ROCm-compatible GPUs
