# DeepSeek-R1 - Benchmark Guide for AMD GPU

## About the Model

DeepSeek-R1 is a first-generation reasoning model that achieves performance comparable to OpenAI-o1 across math, code, and reasoning tasks. Built on the DeepSeek-V3 architecture with 671 billion parameters (37 billion activated per forward pass), R1 is fine-tuned using Reinforcement Learning to improve reasoning and Chain-of-Thought output. The model excels at complex reasoning tasks and can think through problems step-by-step before providing answers.

### Original DeepSeek-R1 Paper

**"DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"** (DeepSeek-AI, 2025)

DeepSeek-R1 directly applies reinforcement learning (RL) to the base model without relying on supervised fine-tuning (SFT) as a preliminary step. Using Group Relative Policy Optimization (GRPO) as the RL framework, the model develops strong reasoning capabilities. The architecture employs Mixture of Experts (MoE) design with Multi-Head Latent Attention (MLA) for memory-efficient inference. DeepSeek has open-sourced DeepSeek-R1-Zero, DeepSeek-R1, and six dense models distilled from DeepSeek-R1 based on Llama and Qwen at 1.5B, 7B, 8B, 14B, 32B, and 70B parameters.

**Paper:** [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) | **Published:** January 2025

---

## Standard Benchmark Datasets

DeepSeek-R1 is evaluated on industry-standard benchmarks for reasoning, mathematics, and coding capabilities.

### 1. MMLU (Massive Multitask Language Understanding)

**MMLU** evaluates LLMs through multiple-choice questions covering 57 subjects including math, history, law, computer science, and ethics. It tests knowledge across college-level disciplines.

**Download from HuggingFace**

```bash
# Install dependencies
pip install datasets transformers
```

```python
from datasets import load_dataset

# Load MMLU dataset
dataset = load_dataset("cais/mmlu", "all")

# View a sample
print(dataset['test'][0])
# Output: {'question': '...', 'choices': ['A', 'B', 'C', 'D'], 'answer': 0, 'subject': 'history'}
```

### 2. MATH Dataset

**MATH** consists of 12,500 challenging competition mathematics problems from the AMC 10, AMC 12, AIME, and other competitions. Each problem has a full step-by-step solution.

**Download from HuggingFace**

```python
from datasets import load_dataset

# Load MATH dataset
dataset = load_dataset("hendrycks/competition_math")

# Or use EleutherAI mirror
dataset = load_dataset("EleutherAI/hendrycks_math")

# View a sample
print(dataset['test'][0])
# Output: {'problem': '...', 'level': 'Level 5', 'type': 'Algebra', 'solution': '...'}
```

### 3. GSM8K (Grade School Math 8K)

**GSM8K** contains 8,500 grade-school-level, linguistically diverse mathematics word problems requiring 2-8 steps to solve.

**Download from HuggingFace**

```python
from datasets import load_dataset

# Load GSM8K dataset
dataset = load_dataset("openai/gsm8k", "main")

# Or use socratic version with chain-of-thought solutions
dataset = load_dataset("openai/gsm8k", "socratic")

# View a sample
print(dataset['test'][0])
# Output: {'question': '...', 'answer': '#### 42'}
```

### 4. HumanEval

**HumanEval** evaluates code generation through 164 programming problems. Each problem includes a function signature, docstring, body, and unit tests.

**Download from HuggingFace**

```python
from datasets import load_dataset

# Load HumanEval dataset
dataset = load_dataset("openai/openai_humaneval")

# View a sample
print(dataset['test'][0])
# Output: {'task_id': 'test/0', 'prompt': 'def...', 'canonical_solution': '...',
#          'test': 'def check...', 'entry_point': 'function_name'}
```

**Security Warning:** HumanEval executes untrusted model-generated code. Use only within a robust security sandbox.

---

## Installation & Inference

### Install DeepSeek-R1

```bash
# Install dependencies
pip install torch transformers accelerate

# For AMD GPUs, install ROCm-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# For optimized inference, install SGLang
pip install "sglang[all]"

# Or install vLLM
pip install vllm
```

### Download Model Weights

```bash
# Using huggingface-cli (recommended for large models)
pip install -U "huggingface_hub[cli]"

# Download full DeepSeek-R1 (671B parameters)
huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1 --local-dir ./DeepSeek-R1

# Download distilled versions (smaller, faster)
huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Llama-70B --local-dir ./DeepSeek-R1-70B
huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --local-dir ./DeepSeek-R1-32B
huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local-dir ./DeepSeek-R1-7B

# Download quantized GGUF versions (lower memory)
huggingface-cli download bartowski/DeepSeek-R1-GGUF --include "DeepSeek-R1-Q8_0/*" --local-dir ./DeepSeek-R1-GGUF
```

### Basic Inference with Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Setup device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load model (use distilled version for better performance)
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    device_map="auto",
    trust_remote_code=True
)

# Prepare input
prompt = "Please reason step by step, and put your final answer within \\boxed{}: What is 25 * 47?"

# Enforce reasoning by starting with <think>
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

# Generate
outputs = model.generate(
    inputs,
    max_new_tokens=8192,
    temperature=0.6,
    do_sample=True,
    top_p=0.95
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Inference with SGLang Server (Recommended for AMD GPUs)

```bash
# Launch SGLang server with tensor parallelism
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --trust-remote-code \
    --tp 2 \
    --port 30000

# For FP8 quantization (lower memory)
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-R1 \
    --trust-remote-code \
    --tp 8 \
    --quantization fp8 \
    --port 30000
```

```python
# Client code to use SGLang server
import requests

url = "http://localhost:30000/v1/chat/completions"
payload = {
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "messages": [
        {"role": "user", "content": "Solve: What is the derivative of x^2 + 3x?"}
    ],
    "temperature": 0.6,
    "max_tokens": 8192
}

response = requests.post(url, json=payload)
print(response.json()["choices"][0]["message"]["content"])
```

### Inference with vLLM

```python
from vllm import LLM, SamplingParams

# Initialize model
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
llm = LLM(
    model=model_id,
    tokenizer=model_id,
    tensor_parallel_size=1,
    max_model_len=8192,
    trust_remote_code=True
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.6,
    max_tokens=8192,
    top_p=0.95
)

# Generate
prompts = ["Solve this problem step by step: If a train travels 120 miles in 2 hours, what is its average speed?"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### Expected Output Format

DeepSeek-R1 produces reasoning chains with explicit thinking steps:

```
<think>
To solve this problem, I need to find the average speed.
Speed = Distance / Time
Distance = 120 miles
Time = 2 hours
Speed = 120 / 2 = 60 miles per hour
</think>

The train's average speed is 60 miles per hour.
```

**Best Practices:**
- Set temperature within 0.5-0.7 range (0.6 recommended)
- Avoid system prompts; include all instructions in user prompt
- For math problems, include "Please reason step by step, and put your final answer within \boxed{}"
- Enforce reasoning by prompting with "<think>\n" at the beginning

---

## Benchmark Results & Performance Metrics

### DeepSeek-R1 Performance on Standard Benchmarks

| Benchmark | DeepSeek-R1 (671B) | OpenAI o1-1217 | GPT-4o | Claude 3.5 Sonnet | Metric | Notes |
|-----------|-------------------|----------------|---------|-------------------|--------|-------|
| **MMLU** | **90.8%** | 91.8% | 88.0% | 88.7% | Accuracy | 57 subjects, college-level knowledge |
| **MMLU-Pro** | **84.0%** | 85.5% | 72.6% | 78.0% | Accuracy | Harder version with more complex questions |
| **GPQA Diamond** | **71.5%** | 75.7% | 49.9% | 65.0% | Accuracy | Graduate-level science questions |
| **MATH-500** | **97.3%** | 96.4% | 74.6% | 78.3% | Accuracy | Competition mathematics |
| **AIME 2024** | **79.8%** | 79.2% | 9.3% | 16.0% | Pass@1 | American Invitational Mathematics Exam |
| **GSM8K** | **98.9%** | 96.4% | 94.8% | 96.4% | Accuracy | Grade school math word problems |
| **HumanEval** | **97.3%** | 92.7% | 90.2% | 92.0% | Pass@1 | Python code generation |
| **LiveCodeBench** | **65.8%** | 67.0% | 45.2% | 51.7% | Pass@1 | Real-world coding tasks |
| **Codeforces** | **2029 Elo** | 2067 Elo | 1258 Elo | 1318 Elo | Rating | Competitive programming (96.3% percentile) |

**Key Observations:**
- DeepSeek-R1 achieves performance comparable to OpenAI-o1 across reasoning, math, and code tasks
- Significant improvements over previous generation models (GPT-4o, Claude 3.5)
- Strong performance on complex reasoning (GPQA, AIME) and coding (Codeforces)

### Distilled Model Performance

| Model | Parameters | MMLU | MATH | HumanEval | Training Base | Efficiency |
|-------|-----------|------|------|-----------|---------------|------------|
| **DeepSeek-R1** | 671B (37B active) | 90.8% | 97.3% | 97.3% | DeepSeek-V3 | Full model |
| **R1-Distill-Llama-70B** | 70B | 88.3% | 92.7% | 89.6% | Llama-3.1-70B | ~2x faster |
| **R1-Distill-Qwen-32B** | 32B | 86.7% | 90.8% | 85.4% | Qwen2.5-32B | ~4x faster |
| **R1-Distill-Qwen-14B** | 14B | 84.1% | 87.3% | 79.3% | Qwen2.5-14B | ~8x faster |
| **R1-Distill-Qwen-7B** | 7B | 81.9% | 82.6% | 73.2% | Qwen2.5-7B | ~15x faster |
| **R1-Distill-Llama-8B** | 8B | 80.4% | 79.1% | 70.7% | Llama-3.1-8B | ~15x faster |
| **R1-Distill-Qwen-1.5B** | 1.5B | 68.2% | 58.9% | 48.8% | Qwen2.5-1.5B | ~40x faster |

**Note:** Distilled models (32B and 70B) retain most reasoning capability at significantly lower computational cost.

---

## AMD GPU Benchmarking Setup

### ROCm Installation for AMD GPUs

```bash
# Check AMD GPU and ROCm compatibility
rocm-smi

# Install PyTorch with ROCm 6.2 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
python -c "import torch; print(f'ROCm Version: {torch.version.hip}')"

# Install SGLang with ROCm support (recommended for inference)
pip install "sglang[all]"

# Install lm-evaluation-harness for benchmarking
pip install lm-eval
```

### Benchmark Script for AMD GPU

```python
import torch
import time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import re

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16

print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"ROCm Version: {torch.version.hip}")

# Load model (using distilled version for practical benchmarking)
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    device_map="auto",
    trust_remote_code=True
)

# Load benchmark datasets
print("Loading datasets...")
gsm8k = load_dataset("openai/gsm8k", "main", split="test[:100]")
humaneval = load_dataset("openai/openai_humaneval", split="test[:50]")

# Benchmark GSM8K
print("\n=== GSM8K Benchmark ===")
gsm8k_results = []
gsm8k_correct = 0

for i, sample in enumerate(gsm8k):
    question = sample["question"]
    true_answer = sample["answer"].split("####")[-1].strip()

    # Prepare prompt
    prompt = f"Please reason step by step, and put your final answer within \\boxed{{}}: {question}"
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=2048,
            temperature=0.6,
            do_sample=True,
            top_p=0.95
        )
    inference_time = time.time() - start_time

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract answer
    boxed_match = re.search(r'\\boxed{([^}]+)}', response)
    predicted_answer = boxed_match.group(1).strip() if boxed_match else ""

    # Check correctness
    is_correct = predicted_answer == true_answer
    if is_correct:
        gsm8k_correct += 1

    gsm8k_results.append({
        "sample_id": i,
        "inference_time": inference_time,
        "correct": is_correct,
        "tokens_generated": len(outputs[0]) - len(inputs[0])
    })

    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/100 samples, Accuracy: {gsm8k_correct/(i+1)*100:.1f}%")

# GSM8K Summary
avg_gsm8k_time = np.mean([r["inference_time"] for r in gsm8k_results])
avg_tokens = np.mean([r["tokens_generated"] for r in gsm8k_results])
gsm8k_accuracy = (gsm8k_correct / len(gsm8k_results)) * 100

print(f"\nGSM8K Results:")
print(f"  Accuracy: {gsm8k_accuracy:.2f}%")
print(f"  Avg Inference Time: {avg_gsm8k_time:.2f}s")
print(f"  Avg Tokens Generated: {avg_tokens:.0f}")
print(f"  Tokens/Second: {avg_tokens/avg_gsm8k_time:.2f}")

# Memory statistics
print(f"\nMemory Statistics:")
print(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"  Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
print(f"  Max Allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
```

### SGLang Benchmark Script (Recommended)

```python
import time
import requests
from datasets import load_dataset
import numpy as np

# Launch server first:
# python -m sglang.launch_server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --trust-remote-code --tp 2

url = "http://localhost:30000/v1/chat/completions"

# Load dataset
dataset = load_dataset("openai/gsm8k", "main", split="test[:100]")

results = []
correct = 0

for i, sample in enumerate(dataset):
    question = sample["question"]
    true_answer = sample["answer"].split("####")[-1].strip()

    payload = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "messages": [
            {"role": "user", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}: {question}"}
        ],
        "temperature": 0.6,
        "max_tokens": 2048
    }

    start_time = time.time()
    response = requests.post(url, json=payload)
    inference_time = time.time() - start_time

    result = response.json()
    predicted_text = result["choices"][0]["message"]["content"]

    # Extract answer and check correctness
    import re
    boxed_match = re.search(r'\\boxed{([^}]+)}', predicted_text)
    predicted_answer = boxed_match.group(1).strip() if boxed_match else ""

    is_correct = predicted_answer == true_answer
    if is_correct:
        correct += 1

    results.append({
        "inference_time": inference_time,
        "correct": is_correct,
        "tokens": result["usage"]["completion_tokens"]
    })

    if (i + 1) % 10 == 0:
        print(f"Progress: {i+1}/100, Accuracy: {correct/(i+1)*100:.1f}%, Avg Time: {np.mean([r['inference_time'] for r in results]):.2f}s")

# Summary
print(f"\n=== Benchmark Results ===")
print(f"Accuracy: {correct/len(results)*100:.2f}%")
print(f"Avg Inference Time: {np.mean([r['inference_time'] for r in results]):.2f}s")
print(f"Avg Tokens/Second: {np.mean([r['tokens']/r['inference_time'] for r in results]):.2f}")
```

### AMD-Specific Performance Monitoring

```python
import subprocess
import torch

def get_rocm_smi_stats():
    """Get AMD GPU statistics using rocm-smi"""
    try:
        # GPU utilization
        util_result = subprocess.run(['rocm-smi', '--showuse'],
                                    capture_output=True, text=True)

        # Memory info
        mem_result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'],
                                   capture_output=True, text=True)

        # Power consumption
        power_result = subprocess.run(['rocm-smi', '--showpower'],
                                     capture_output=True, text=True)

        return {
            'utilization': util_result.stdout,
            'memory': mem_result.stdout,
            'power': power_result.stdout
        }
    except Exception as e:
        return f"Error: {e}"

# PyTorch memory tracking
print("=== PyTorch Memory Stats ===")
print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
print(f"Max Allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

# ROCm info
print(f"\n=== ROCm Info ===")
print(f"ROCm Version: {torch.version.hip}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")

# Get ROCm-SMI stats
print(f"\n=== ROCm-SMI Stats ===")
stats = get_rocm_smi_stats()
print(stats['utilization'])
print(stats['memory'])
print(stats['power'])
```

### Performance Metrics Table Template

| Metric | NVIDIA H200 | AMD MI300X | AMD MI350X | AMD RX 7900 XTX | Notes |
|--------|-------------|------------|------------|-----------------|-------|
| **GPU Model** | NVIDIA H200 | AMD MI300X | AMD MI350X | AMD RX 7900 XTX | Compare datacenter vs consumer GPUs |
| **Memory (GB)** | 141 | 192 | 288 | 24 | HBM3e capacity |
| **Memory Bandwidth (GB/s)** | 4800 | 5300 | 6000 | 960 | Theoretical peak |
| **TDP (W)** | 700 | 750 | 1000 | 355 | Thermal design power |
| **Model Size** | R1-Distill-32B | R1-Distill-32B | R1-Distill-32B | R1-Distill-7B | Recommended model size |
| **Precision** | FP8 | FP8 | FP8 | FP16 | Quantization format |
| **Batch Size** | 64 | 64 | 64 | 16 | Concurrent requests |
| **Input Tokens** | 3200 | 3200 | 3200 | 1024 | Average prompt length |
| **Output Tokens** | 800 | 800 | 800 | 512 | Average response length |
| **Throughput (tokens/s)** | _[Reference]_ | _[Your result]_ | _[Your result]_ | _[Your result]_ | Higher is better |
| **Latency (ms)** | _[Reference]_ | _[Your result]_ | _[Your result]_ | _[Your result]_ | Time to first token |
| **Peak Memory Usage (GB)** | ~80 | _[Your result]_ | _[Your result]_ | _[Your result]_ | Monitor with rocm-smi |
| **GPU Utilization (%)** | ~95 | _[Your result]_ | _[Your result]_ | _[Your result]_ | Average during inference |
| **Average Power Draw (W)** | ~600 | _[Your result]_ | _[Your result]_ | _[Your result]_ | Monitor with rocm-smi --showpower |
| **GSM8K Accuracy (%)** | 98.5 | _[Your result]_ | _[Your result]_ | _[Your result]_ | Math reasoning benchmark |
| **HumanEval Pass@1 (%)** | 89.0 | _[Your result]_ | _[Your result]_ | _[Your result]_ | Code generation benchmark |

### AMD MI300X vs NVIDIA H200 Benchmark Results

**From AMD's February 2025 Testing:**

| Configuration | Throughput Advantage | Latency Advantage | Test Conditions |
|---------------|---------------------|-------------------|-----------------|
| **MI300X (8 GPUs) vs H200 (8 GPUs)** | **2X-5X higher** at same latency | **60% lower** at same batch size | DeepSeek-R1 FP8, SGLang 0.4.2 |
| **Concurrency: 16 requests** | **5X higher throughput** | - | 3200 input / 800 output tokens |
| **Batch Size: 64** | **75% higher throughput** | **60% lower latency** | Same configuration |

**Key Optimizations Achieved:**
- AITER block-scale GEMM: **up to 2X boost**
- AITER block-scale fused MoE: **up to 3X boost**
- AITER MLA for decode: **up to 17X boost**
- AITER MHA for prefill: **up to 14X boost**

**AMD RX 7900 XTX Consumer GPU Performance:**
- Outperforms NVIDIA RTX 4090 by **up to 13%** in DeepSeek-R1 inference
- Best consumer GPU for local deployment
- Recommended for 7B-14B distilled models

### Complete Runtime Metrics Table

| Runtime Metric | Formula | NVIDIA H200 | AMD MI300X | AMD MI350X | AMD RX 7900 XTX | Notes |
|----------------|---------|-------------|------------|------------|-----------------|-------|
| **Throughput (tokens/s)** | output_tokens / inference_time | _[Reference]_ | _[Your result]_ | _[Your result]_ | _[Your result]_ | Higher is better |
| **Latency (ms)** | Time to first token | _[Reference]_ | _[Your result]_ | _[Your result]_ | _[Your result]_ | Lower is better |
| **GPU Utilization (%)** | From nvidia-smi / rocm-smi | ~95% | _[Your result]_ | _[Your result]_ | _[Your result]_ | Target >90% |
| **Memory Bandwidth Utilized (GB/s)** | From monitoring tools | _[Reference]_ | _[Your result]_ | _[Your result]_ | _[Your result]_ | % of theoretical peak |
| **TFLOPS Utilized** | Calculated from operations | _[Reference]_ | _[Your result]_ | _[Your result]_ | _[Your result]_ | FP8/FP16 compute throughput |
| **Energy Efficiency (tokens/Wh)** | throughput / power_draw | _[Reference]_ | _[Your result]_ | _[Your result]_ | _[Your result]_ | Higher is better |
| **Cost per Million Tokens** | Based on cloud pricing | $0.50-1.00 | $0.40-0.80 | $0.35-0.70 | Local (free) | Estimated |

---

## Using lm-evaluation-harness for Comprehensive Benchmarking

### Installation

```bash
# Install lm-evaluation-harness
pip install lm-eval

# Clone repository for access to all tasks
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```

### Running Standard Benchmarks

```bash
# Run MMLU benchmark
lm_eval --model hf \
    --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,trust_remote_code=True,dtype=float16 \
    --tasks mmlu \
    --device cuda:0 \
    --batch_size 8

# Run GSM8K benchmark
lm_eval --model hf \
    --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,trust_remote_code=True,dtype=float16 \
    --tasks gsm8k \
    --device cuda:0 \
    --batch_size 8

# Run HumanEval benchmark
lm_eval --model hf \
    --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,trust_remote_code=True,dtype=float16 \
    --tasks humaneval \
    --device cuda:0 \
    --batch_size 1

# Run multiple benchmarks at once
lm_eval --model hf \
    --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,trust_remote_code=True,dtype=float16 \
    --tasks mmlu,gsm8k,humaneval,arc_challenge \
    --device cuda:0 \
    --batch_size 8 \
    --output_path ./results/
```

### Using vLLM Backend (Faster)

```bash
# Run with vLLM for better performance
lm_eval --model vllm \
    --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,tensor_parallel_size=2,dtype=float16,trust_remote_code=True \
    --tasks mmlu,gsm8k \
    --batch_size auto

# Run with SGLang backend
lm_eval --model sglang \
    --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,tp_size=2 \
    --tasks mmlu,math,humaneval \
    --batch_size auto
```

---

## HuggingFace Open LLM Leaderboard

DeepSeek-R1 and its distilled variants are evaluated on the [HuggingFace Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), which tracks performance across multiple benchmarks.

### Leaderboard Evaluation Tasks

- **MMLU** (Massive Multitask Language Understanding) - 57 subjects
- **GPQA** (Graduate-level Google-Proof Q&A) - Science questions
- **MATH** - Competition mathematics
- **IFEval** - Instruction following evaluation
- **BBH** (Big Bench Hard) - Challenging reasoning tasks
- **MUSR** (Multi-task understanding and reasoning) - Complex reasoning

### Key Metrics Tracked

- **Average Score** - Normalized performance across all tasks
- **Model Size** - Parameters and memory footprint
- **License** - MIT license (commercial use allowed)
- **Architecture** - MoE (Mixture of Experts)

### DeepSeek-R1 Leaderboard Performance

| Model | Average | MMLU | GPQA | MATH | IFEval | BBH | License |
|-------|---------|------|------|------|--------|-----|---------|
| **DeepSeek-R1** | ~85-90 | 90.8 | 71.5 | 97.3 | High | High | MIT |
| **R1-Distill-70B** | ~80-85 | 88.3 | 68.0 | 92.7 | High | High | MIT |
| **R1-Distill-32B** | ~75-80 | 86.7 | 65.0 | 90.8 | High | High | MIT |

**Note:** DeepSeek-R1 competes with the top proprietary models while being fully open-source under MIT license.

---

## Additional Resources

### Official Repositories

- [DeepSeek-R1 GitHub](https://github.com/deepseek-ai/DeepSeek-R1)
- [DeepSeek-R1 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1)
- [HuggingFace Open-R1 (Reproduction)](https://github.com/huggingface/open-r1)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [vLLM GitHub](https://github.com/vllm-project/vllm)

### Papers & Documentation

- [DeepSeek-R1 Paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948)
- [DeepSeek-R1 Technical Report (PDF)](https://arxiv.org/pdf/2501.12948)
- [DeepSeek-V3 Technical Paper](https://arxiv.org/abs/2412.19437)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [lm-evaluation-harness Documentation](https://github.com/EleutherAI/lm-evaluation-harness)

### Blog Posts & Performance Analysis

- [AMD: Unlock DeepSeek-R1 Inference Performance on MI300X](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html)
- [AMD: Supercharge DeepSeek-R1 Inference on MI300X](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html)
- [AMD: Speed Up DeepSeek R1 with 4-bit Quantization](https://www.amd.com/en/blogs/2025/speed-up-deepseek-r1-distill-4-bit-performance-and.html)
- [NVIDIA: DeepSeek-R1 Model Card](https://build.nvidia.com/deepseek-ai/deepseek-r1/modelcard)
- [DEV Community: DeepSeek R1 Guide (2026)](https://dev.to/lemondata_dev/deepseek-r1-guide-architecture-benchmarks-and-practical-usage-in-2026-m8f)
- [SitePoint: DeepSeek R1 Local Deployment Guide](https://www.sitepoint.com/deepseek-r1-local-deployment-guide-2026/)
- [BentoML: Complete Guide to DeepSeek Models](https://www.bentoml.com/blog/the-complete-guide-to-deepseek-models-from-v3-to-r1-and-beyond)
- [Sebastian Raschka: Technical Tour of DeepSeek Models](https://magazine.sebastianraschka.com/p/technical-deepseek)

### AMD ROCm Resources

- [ROCm Installation Guide](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html)
- [AMD MI300X Specifications](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
- [ROCm 7.0 Benchmark Documentation](https://rocm.docs.amd.com/en/docs-7.0-docker/benchmark-docker/inference-sglang-deepseek-r1-fp8.html)
- [AMD Inference Performance Guide](https://www.amd.com/en/developer/resources/technical-articles/2026/inference-performance-on-amd-gpus.html)

### Benchmark Datasets

- [MMLU (cais/mmlu)](https://huggingface.co/datasets/cais/mmlu)
- [MATH (hendrycks/competition_math)](https://huggingface.co/datasets/hendrycks/competition_math)
- [GSM8K (openai/gsm8k)](https://huggingface.co/datasets/openai/gsm8k)
- [HumanEval (openai/openai_humaneval)](https://huggingface.co/datasets/openai/openai_humaneval)
- [MMLU-Pro (TIGER-Lab/MMLU-Pro)](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)

### Community & Support

- [HuggingFace DeepSeek-R1 Discussions](https://huggingface.co/deepseek-ai/DeepSeek-R1/discussions)
- [AMD ROCm Forums](https://community.amd.com/t5/rocm/ct-p/amd-rocm)
- [SGLang Discord](https://discord.gg/sglang)
- [DeepSeek Official Website](https://www.deepseek.com/)

---

## Quick Reference Commands

```bash
# Check AMD GPU and ROCm status
rocm-smi
rocm-smi --showuse --showmeminfo vram --showpower

# Install PyTorch with ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Download DeepSeek-R1 models
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --local-dir ./models/R1-32B
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local-dir ./models/R1-7B

# Launch SGLang server
python -m sglang.launch_server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --trust-remote-code --tp 2

# Run benchmarks with lm-eval
lm_eval --model hf --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,dtype=float16,trust_remote_code=True --tasks mmlu,gsm8k,humaneval --device cuda:0

# Monitor GPU during inference
watch -n 1 rocm-smi

# Check PyTorch CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

---

## Troubleshooting

### Common Issues on AMD GPUs

**Issue: CUDA not detected**
```bash
# Solution: Reinstall PyTorch with ROCm
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

**Issue: Out of memory errors**
```bash
# Solution: Use smaller model or enable quantization
# For 24GB VRAM (RX 7900 XTX): Use 7B or 14B distilled models
# For 192GB VRAM (MI300X): Can run 32B or 70B models with FP8
```

**Issue: Slow inference**
```bash
# Solution: Use SGLang with tensor parallelism
python -m sglang.launch_server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tp 2 --quantization fp8
```

**Issue: Model loading errors**
```bash
# Solution: Add trust_remote_code flag
# Always use trust_remote_code=True when loading DeepSeek models
```

---

**Document Version:** 1.0
**Last Updated:** March 2026
**Target Hardware:** AMD MI300X, MI350X, MI355X, RX 7900 XTX, and other ROCm-compatible GPUs
**ROCm Version:** 6.2+ (7.0+ recommended for latest optimizations)
