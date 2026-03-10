# LLaMA 3.2 Vision 11B - Benchmark Guide for AMD GPU

## About the Model

LLaMA 3.2 Vision 11B is Meta's first multimodal vision-language model in the LLaMA series, combining image understanding with text generation capabilities. Built on top of the LLaMA 3.1 text-only model, it uses a separately trained vision adapter that integrates with the pre-trained language model to enable visual reasoning, image captioning, document understanding, and visual question answering. The model has 10.7 billion parameters and supports a context length of 128,000 tokens, making it suitable for complex multimodal reasoning tasks.

### Original LLaMA 3.2 Paper

**"The Llama 3 Herd of Models"** (Meta AI, 2024)

LLaMA 3.2 Vision models are pretrained and instruction-tuned multimodal large language models optimized for visual recognition, image reasoning, captioning, and answering general questions about images. The models were trained on 6 billion image-text pairs with knowledge current up to December 2023. They excel at document-level understanding including charts and graphs, image captioning, and visual grounding tasks such as directionally pinpointing objects in images based on natural language descriptions. The 11B and 90B Vision models represent Meta's advancement in combining visual and linguistic understanding for both commercial and research applications.

**Paper:** [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) | **Released:** September 25, 2024

---

## Standard Benchmark Datasets

### 1. MMMU (Massive Multi-discipline Multimodal Understanding)

**MMMU** is a comprehensive benchmark designed to evaluate multimodal models on college-level knowledge and reasoning tasks across diverse disciplines.

#### Dataset Structure
- **Development**: 150 samples (for few-shot/in-context learning)
- **Validation**: 900 samples (for debugging and hyperparameter selection)
- **Test**: 10,500 samples (full evaluation with released answers)

#### Coverage
- **Disciplines**: 6 core areas (Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, Tech & Engineering)
- **Subjects**: 30 college-level subjects
- **Subfields**: 183 specialized subfields
- **Image Types**: 30 highly heterogeneous image types

#### Download from HuggingFace

```bash
# Install dependencies
pip install datasets transformers
```

```python
from datasets import load_dataset

# Load MMMU validation split
dataset = load_dataset("MMMU/MMMU", split="validation")

# Or use the merged version from lmms-lab
dataset = load_dataset("lmms-lab/MMMU", split="validation")

# View a sample
print(dataset[0])
# Output: {'question': '...', 'image': <PIL.Image>, 'options': [...], 'answer': '...'}
```

### 2. VQAv2 (Visual Question Answering v2)

**VQAv2** is an industry-standard benchmark for testing systems on detailed image understanding and complex reasoning.

#### Download from HuggingFace

```python
from datasets import load_dataset

# Load VQAv2 validation split
dataset = load_dataset("HuggingFaceM4/VQAv2", split="validation")

# View a sample
print(dataset[0])
# Output: {'question': 'What is in the image?', 'image': <PIL.Image>, 'answers': [...]}
```

### 3. TextVQA

**TextVQA** benchmarks visual reasoning based on text within images, requiring models to read and interpret text to answer questions.

#### Download from HuggingFace

```python
from datasets import load_dataset

# Load TextVQA validation split
dataset = load_dataset("lmms-lab/textvqa", split="validation")

# View a sample
print(dataset[0])
```

### 4. DocVQA

**DocVQA** evaluates models' ability to comprehend and retrieve information from text within document images. Performance is reported using the ANLS (Average Normalized Levenshtein Similarity) metric.

#### Download from HuggingFace

```python
from datasets import load_dataset

# Load DocVQA test subset
dataset = load_dataset("vidore/docvqa_test_subsampled", split="test")

# View a sample
print(dataset[0])
```

---

## Installation & Inference

### Install PyTorch with ROCm Support

```bash
# Install PyTorch for AMD GPUs with ROCm 6.2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Install transformers
pip install -U transformers accelerate pillow

# Verify ROCm installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### Basic Inference with Transformers

```python
from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import requests

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Model ID
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# Load model and processor
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

# Load image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Prepare prompt
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"}
        ]
    }
]

# Process inputs
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(image, input_text, return_tensors="pt").to(device)

# Generate response
output = model.generate(**inputs, max_new_tokens=256)
response = processor.decode(output[0], skip_special_tokens=True)
print(response)
```

### Inference with vLLM (Optimized)

```bash
# Install vLLM
pip install vllm

# Start vLLM server
vllm serve meta-llama/Llama-3.2-11B-Vision-Instruct \
    --enforce-eager \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.95
```

### Python API with vLLM

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Initialize model
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
llm = LLM(
    model=model_id,
    max_model_len=4096,
    max_num_seqs=16,
    enforce_eager=True,
    gpu_memory_utilization=0.95
)

# Prepare sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256
)

# Prepare prompt with image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "path/to/image.jpg"}},
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }
]

# Generate
outputs = llm.chat(messages, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
```

### Expected Output

```json
{
  "text": "The image shows a red sports car parked on a city street. The car appears to be a modern luxury vehicle with sleek design elements. In the background, there are buildings and trees visible, suggesting an urban environment.",
  "model": "meta-llama/Llama-3.2-11B-Vision-Instruct",
  "usage": {
    "prompt_tokens": 156,
    "completion_tokens": 48,
    "total_tokens": 204
  }
}
```

---

## Benchmark Results & Performance Metrics

### LLaMA 3.2 Vision Performance on Standard Benchmarks

| Model | MMMU | VQAv2 | DocVQA | TextVQA | Parameters | Training |
|-------|------|-------|--------|---------|------------|----------|
| **LLaMA 3.2 11B Vision** | 50.7% | 75.2% | 88.4% | ~73% | 10.7B | Zero-shot |
| **LLaMA 3.2 90B Vision** | 60.3% | 78.1% | 90.1% | ~75% | 90B | Zero-shot |
| Qwen2.5-VL 7B | 54.0% | 76.5% | 91.2% | 74.8% | 7B | Zero-shot |
| MiniCPM-V 2.6 | 43.0% | 72.1% | 85.3% | 73.0% | 8B | Zero-shot |
| GPT-4V | 56.8% | 77.2% | 88.4% | 78.0% | Unknown | Zero-shot |

**Note:** Higher percentages indicate better performance. MMMU measures college-level multimodal understanding, VQAv2 tests visual question answering, DocVQA evaluates document understanding with ANLS metric, and TextVQA assesses text-reading in images.

### Performance: LLaMA 3.2 Vision vs Alternatives

| Implementation | Size | MMMU | VQAv2 | Platform Support | Notes |
|----------------|------|------|-------|------------------|-------|
| **LLaMA 3.2 11B Vision** | 10.7B | 50.7% | 75.2% | NVIDIA, AMD ROCm | Best balance of performance and efficiency |
| **LLaMA 3.2 90B Vision** | 90B | 60.3% | 78.1% | NVIDIA, AMD MI300X | Requires high VRAM (192GB on single MI300X) |
| Qwen2.5-VL 7B | 7B | 54.0% | 76.5% | NVIDIA, AMD | Outperforms 11B on some tasks despite smaller size |
| MiniCPM-V 2.6 | 8B | 43.0% | 72.1% | NVIDIA, AMD | More compact, lower accuracy |

---

## AMD GPU Benchmarking Setup

### ROCm Installation for AMD GPUs

```bash
# Check ROCm compatibility
rocm-smi

# Install PyTorch with ROCm 6.2 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Install additional dependencies
pip install transformers accelerate vllm pillow datasets

# Verify installation
python -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}'); print(f'ROCm Version: {torch.version.hip}')"
```

### Benchmark Script for AMD GPU (MMMU)

```python
import torch
import time
from datasets import load_dataset
from transformers import MllamaForConditionalGeneration, AutoProcessor
import numpy as np
from PIL import Image

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16

print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Load model
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
print(f"Loading model: {model_id}")

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

# Load MMMU validation dataset (subset for quick testing)
print("Loading MMMU validation dataset...")
dataset = load_dataset("MMMU/MMMU", split="validation[:10]")

# Benchmark
results = []
correct = 0
total = 0

for i, sample in enumerate(dataset):
    # Prepare the question
    question = sample["question"]
    image = sample["image"]

    # Create prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{question}\nProvide your answer."}
            ]
        }
    ]

    # Process inputs
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, return_tensors="pt").to(device)

    # Measure inference time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()

    inference_time = end_time - start_time
    predicted = processor.decode(output[0], skip_special_tokens=True)

    # Track results
    results.append({
        "sample_id": i,
        "question": question,
        "inference_time": inference_time,
        "predicted": predicted,
        "reference": sample.get("answer", "N/A")
    })

    total += 1

    # Memory stats
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"Sample {i}: Time={inference_time:.2f}s | Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    else:
        print(f"Sample {i}: Time={inference_time:.2f}s")

# Summary statistics
avg_inference_time = np.mean([r["inference_time"] for r in results])
throughput = 1 / avg_inference_time

print(f"\n=== Benchmark Summary ===")
print(f"Average Inference Time: {avg_inference_time:.2f}s per sample")
print(f"Throughput: {throughput:.2f} samples/second")
print(f"Total Samples Processed: {total}")

if torch.cuda.is_available():
    print(f"Peak Memory Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")
    print(f"Peak Memory Reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f}GB")
```

### Benchmark Script for AMD GPU (VQAv2)

```python
import torch
import time
from datasets import load_dataset
from transformers import MllamaForConditionalGeneration, AutoProcessor
import numpy as np

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16

# Load model
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

# Load VQAv2 validation dataset
print("Loading VQAv2 validation dataset...")
dataset = load_dataset("HuggingFaceM4/VQAv2", split="validation[:50]")

# Benchmark
results = []
for i, sample in enumerate(dataset):
    question = sample["question"]
    image = sample["image"]

    # Prepare prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Question: {question}\nAnswer:"}
            ]
        }
    ]

    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, return_tensors="pt").to(device)

    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=64)
    end_time = time.time()

    inference_time = end_time - start_time
    predicted = processor.decode(output[0], skip_special_tokens=True)

    results.append({
        "sample_id": i,
        "inference_time": inference_time,
        "predicted": predicted,
        "ground_truth": sample.get("answers", [])
    })

    print(f"Sample {i}: Time={inference_time:.2f}s | Question: {question[:50]}...")

# Summary
avg_time = np.mean([r["inference_time"] for r in results])
print(f"\nAverage Inference Time: {avg_time:.2f}s")
print(f"Throughput: {1/avg_time:.2f} samples/second")
```

### Performance Metrics Table Template

| Metric | NVIDIA A100-80GB | NVIDIA H100-80GB | AMD MI300X | AMD RX 7900 XTX | Notes |
|--------|------------------|------------------|------------|-----------------|-------|
| **GPU Model** | NVIDIA A100-80GB | NVIDIA H100-80GB | AMD MI300X | AMD RX 7900 XTX | Compare datacenter vs consumer GPUs |
| **Memory (GB)** | 80 | 80 | 192 | 24 | VRAM capacity |
| **TDP (W)** | 400 | 700 | 750 | 355 | Thermal design power |
| **Model Fits in Memory** | Yes (BF16) | Yes (BF16) | Yes (BF16) | Yes (BF16/FP16) | 11B model loads in ~22GB |
| **Batch Size** | 16 | 32 | 32 | 8 | Max batch size for inference |
| **Inference Time (s/sample)** | ~2.5 | ~1.8 | _[Your result]_ | _[Your result]_ | Time per MMMU question |
| **Throughput (samples/s)** | 0.40 | 0.55 | _[Your result]_ | _[Your result]_ | Questions answered per second |
| **Peak Memory Usage (GB)** | ~35 | ~35 | _[Your result]_ | _[Your result]_ | Monitor with rocm-smi |
| **Average Power Draw (W)** | ~320 | ~480 | _[Your result]_ | _[Your result]_ | Monitor with rocm-smi --showpower |
| **Energy per Sample (Wh)** | ~0.22 | ~0.24 | _[Your result]_ | _[Your result]_ | Lower is better |
| **MMMU Accuracy (%)** | 50.7 | 50.7 | _[Your result]_ | _[Your result]_ | Should match reference |
| **VQAv2 Accuracy (%)** | 75.2 | 75.2 | _[Your result]_ | _[Your result]_ | Should match reference |

### AMD-Specific Metrics to Track

```python
# GPU utilization tracking
import subprocess

def get_rocm_smi_stats():
    """Get AMD GPU statistics using rocm-smi"""
    result = subprocess.run(['rocm-smi', '--showuse', '--showmeminfo', 'vram'],
                          capture_output=True, text=True)
    return result.stdout

def get_power_stats():
    """Get AMD GPU power consumption"""
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

# During inference loop
print(get_rocm_smi_stats())
print(get_power_stats())
```

### Complete Runtime Metrics Table

| Runtime Metric | Formula | NVIDIA A100-80GB | NVIDIA H100-80GB | AMD MI300X | AMD RX 7900 XTX | Notes |
|----------------|---------|------------------|------------------|------------|-----------------|-------|
| **Tokens/Second** | tokens_generated / inference_time | ~85 | ~120 | _[Your result]_ | _[Your result]_ | Generation speed |
| **Latency (ms)** | Time to first token | ~800 | ~600 | _[Your result]_ | _[Your result]_ | Lower is better |
| **GPU Utilization (%)** | From nvidia-smi / rocm-smi | ~85% | ~90% | _[Your result]_ | _[Your result]_ | Average during inference |
| **Memory Bandwidth (GB/s)** | From nvidia-smi / rocm-smi | ~2.0 TB/s | ~3.3 TB/s | _[Your result]_ | _[Your result]_ | MI300X: ~5.3 TB/s, RX 7900 XTX: ~960 GB/s theoretical |
| **TFLOPS Utilized** | Calculated from operations | ~150 | ~250 | _[Your result]_ | _[Your result]_ | BF16 compute throughput |
| **Batch Processing Time (s)** | time_for_batch / batch_size | ~40 | ~30 | _[Your result]_ | _[Your result]_ | Batch size 16 |
| **Energy Efficiency (samples/kWh)** | 3600 / (power_draw × time_per_sample) | ~5600 | ~5300 | _[Your result]_ | _[Your result]_ | Higher is better |

---

## Fine-tuning on AMD GPUs

### LoRA Fine-tuning Script

```python
# Fine-tuning with AMD MI300X using LoRA
# Based on AMD ROCm documentation

import torch
from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Configuration
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
device = "cuda:0"

# Load model
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training arguments
training_args = TrainingArguments(
    output_dir="./llama-3.2-vision-lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="adamw_torch",
)

# Note: Training time on AMD MI300X
# - 11B model LoRA fine-tuning: ~23 hours
# - 90B model QLoRA fine-tuning: ~13 hours
```

---

## Docker Setup for AMD GPUs

### Using Official ROCm PyTorch Docker

```bash
# Pull the official ROCm PyTorch image
docker pull rocm/pytorch:rocm6.2.1_ubuntu20.04_py3.9_pytorch_release_2.3.0

# Run container with GPU access
docker run -it \
    --device /dev/dri \
    --device /dev/kfd \
    --network host \
    --ipc host \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    -v $HOME:$HOME \
    --shm-size 64G \
    --name llama_vision_env \
    rocm/pytorch:rocm6.2.1_ubuntu20.04_py3.9_pytorch_release_2.3.0

# Inside the container, install dependencies
pip install transformers accelerate vllm datasets pillow
```

---

## Multimodal Vision-Language Leaderboard

### Key Benchmark Comparisons

The following benchmarks are commonly used to evaluate vision-language models:

#### Evaluation Datasets
- **MMMU** (Massive Multi-discipline Multimodal Understanding) - College-level knowledge
- **VQAv2** (Visual Question Answering v2) - General visual reasoning
- **TextVQA** - Text-reading in natural images
- **DocVQA** - Document understanding and OCR
- **OCRBench** - OCR capabilities across languages
- **ScienceQA** - Science question answering with diagrams
- **POPE** (Polling-based Object Probing Evaluation) - Object hallucination testing
- **GQA** - Compositional question answering

### Key Metrics Tracked
- **Accuracy** (%) - primary metric for most tasks
- **ANLS** (Average Normalized Levenshtein Similarity) - for DocVQA
- **CIDEr** - for image captioning tasks
- **Inference Speed** (samples/second)
- **Memory Usage** (GB)

**Note:** LLaMA 3.2 Vision models are evaluated on the HuggingFace Open VLM Leaderboard and various academic benchmarks. The 11B model balances performance with efficiency, while the 90B model achieves state-of-the-art results on many tasks.

---

## Additional Resources

### Official Repositories & Documentation
- [LLaMA 3.2 11B Vision on HuggingFace](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision)
- [LLaMA 3.2 11B Vision Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
- [Meta AI Official Blog Post](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)
- [LLaMA Documentation - Vision Capabilities](https://www.llama.com/docs/how-to-guides/vision-capabilities/)

### Papers & Research
- [The Llama 3 Herd of Models (arXiv:2407.21783)](https://arxiv.org/abs/2407.21783)
- [MMMU Benchmark Paper (arXiv:2311.16502)](https://arxiv.org/abs/2311.16502)

### AMD ROCm Resources
- [AMD ROCm: LLaMA 3.2 Vision Inference Guide](https://rocm.blogs.amd.com/artificial-intelligence/llama3_2_vision/README.html)
- [AMD ROCm: Fine-tuning LLaMA 3.2 Vision](https://rocm.blogs.amd.com/software-tools-optimization/fine-tune-llama3.2/README.html)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [AMD ROCm AI Performance Results](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html)

### Blog Posts & Tutorials
- [Hands-On Guide to LLaMA 3.2 Vision](https://www.labellerr.com/blog/hands-on-llama-3-2-vision/)
- [AWS: Vision Use Cases with LLaMA 3.2](https://aws.amazon.com/blogs/machine-learning/vision-use-cases-with-llama-3-2-11b-and-90b-models-from-meta/)
- [Running LLaMA 3.2 on Cloud GPU with Transformers](https://blog.ori.co/how-to-run-llama3.2-on-a-cloud-gpu-with-transformers)

### Datasets
- [MMMU (Official)](https://huggingface.co/datasets/MMMU/MMMU)
- [MMMU (lmms-lab merged)](https://huggingface.co/datasets/lmms-lab/MMMU)
- [VQAv2](https://huggingface.co/datasets/HuggingFaceM4/VQAv2)
- [TextVQA](https://huggingface.co/datasets/lmms-lab/textvqa)
- [DocVQA](https://huggingface.co/datasets/vidore/docvqa_test_subsampled)

### Tools & Frameworks
- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM Vision Language Examples](https://docs.vllm.ai/en/latest/getting_started/examples/offline_inference_vision_language.html)
- [Ollama LLaMA 3.2 Vision](https://ollama.com/library/llama3.2-vision:11b)

---

## Quick Reference Commands

```bash
# Install PyTorch with ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Install dependencies
pip install transformers accelerate vllm pillow datasets

# Check AMD GPU status
rocm-smi
rocm-smi --showuse --showmeminfo vram
rocm-smi --showpower

# Verify PyTorch ROCm installation
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Download MMMU dataset
python -c "from datasets import load_dataset; ds = load_dataset('MMMU/MMMU', split='validation')"

# Download VQAv2 dataset
python -c "from datasets import load_dataset; ds = load_dataset('HuggingFaceM4/VQAv2', split='validation')"

# Run vLLM server
vllm serve meta-llama/Llama-3.2-11B-Vision-Instruct --enforce-eager --max-num-seqs 16

# Pull and run ROCm Docker container
docker run -it --device /dev/dri --device /dev/kfd --group-add video --shm-size 64G rocm/pytorch:rocm6.2.1_ubuntu20.04_py3.9_pytorch_release_2.3.0
```

---

## Known Limitations

### Current Constraints
- **Single Image Input**: Currently only one leading image is supported per inference (multi-image support in development)
- **English Only for Vision**: For image+text applications, English is the only supported language (text-only supports 8 languages)
- **Context Window**: 128,000 tokens maximum (prompt + response combined)
- **GPU Memory**: 11B model requires ~22-24GB VRAM in BF16 precision

### Platform Support
- **Fully Supported**: AMD MI300X, NVIDIA A100/H100, NVIDIA RTX 4090
- **Supported with Limitations**: AMD RX 7900 XTX (may require smaller batch sizes)
- **Not Recommended**: GPUs with <20GB VRAM (consider quantized versions)

---

**Document Version:** 1.0
**Last Updated:** March 2026
**Target Hardware:** AMD MI300X, RX 7900 XTX, and other ROCm-compatible GPUs
**Model Version:** LLaMA 3.2 Vision 11B (Released September 25, 2024)
