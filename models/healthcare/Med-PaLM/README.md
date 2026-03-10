# Med-PaLM - Benchmark Guide for AMD GPU

**Navigation:** [🏠 Home]({{ site.baseurl }}/) | [📑 Models Index]({{ site.baseurl }}/MODELS_INDEX) | [📝 Contributing]({{ site.baseurl }}/CONTRIBUTING)

---

## About the Model

Med-PaLM is Google's medical large language model designed to provide high-quality answers to medical questions. It represents the first AI system to reach a "passing" score on United States Medical Licensing Examination (USMLE)-style questions, and with Med-PaLM 2, achieved expert-level performance on medical benchmarks. The model is built on Google's PaLM (Pathways Language Model) architecture, with specialized medical domain fine-tuning to address healthcare question answering tasks.

Med-PaLM was developed by taking the instruction-tuned Flan-PaLM model and further fine-tuning it on carefully curated medical resources. The progression evolved from PaLM (base model) → Flan-PaLM (instruction-tuned) → Med-PaLM (medical domain fine-tuned) → Med-PaLM 2 (built on improved PaLM 2 base with enhanced medical fine-tuning). The model demonstrates strong performance across multiple medical question answering datasets and has been evaluated by physicians for clinical utility.

### Original Med-PaLM Papers

**"Large Language Models Encode Clinical Knowledge"** (Singhal et al., 2022/2023)

Med-PaLM was the first large language model to exceed a "passing" score (>60%) on United States Medical Licensing Examination (USMLE)-style questions, achieving 67.2% accuracy on the MedQA dataset. The model demonstrates the potential of large language models to encode and retrieve clinical knowledge at scale, representing a significant milestone in applying AI to medical question answering.

**Paper:** [Nature](https://www.nature.com/articles/s41586-023-06291-2) | **arXiv:** [arXiv:2212.13138](https://arxiv.org/abs/2212.13138) | **Published:** Nature, July 2023

**"Towards Expert-Level Medical Question Answering with Large Language Models"** (Singhal et al., 2023)

Med-PaLM 2 represents a major advancement, becoming the first large language model to perform at an "expert" level on medical question answering benchmarks. The model achieves 86.5% accuracy on the MedQA dataset, improving upon Med-PaLM by over 19%. In pairwise comparative ranking of 1,066 consumer medical questions, physicians preferred Med-PaLM 2 answers to those produced by physicians on eight of nine axes pertaining to clinical utility. The work demonstrates how improvements in base language models, medical domain-specific fine-tuning, and novel prompting strategies (ensemble refinement) can achieve expert-level medical reasoning.

**Paper:** [Nature Medicine](https://www.nature.com/articles/s41591-024-03423-7) | **arXiv:** [arXiv:2305.09617](https://arxiv.org/abs/2305.09617) | **Published:** May 2023

---

## Standard Benchmark Datasets

Med-PaLM and Med-PaLM 2 are evaluated on several medical question answering benchmarks:

### 1. MedQA (USMLE-Style Questions)

**MedQA** is a free-form multiple-choice question answering dataset designed to solve medical problems, collected from professional medical board exams. It contains questions that follow the format of the United States Medical Licensing Examination (USMLE) and is the primary benchmark for evaluating clinical knowledge in AI systems.

**Dataset Structure:**
- **English (USMLE-style)**: 12,723 questions
- **Simplified Chinese**: 34,251 questions
- **Traditional Chinese**: 14,123 questions
- **Question format**: Multiple-choice with 4-5 options
- **Domains**: Clinical knowledge, medical diagnosis, treatment

**Download from HuggingFace:**

```bash
# Install dependencies
pip install datasets transformers
```

```python
from datasets import load_dataset

# Load MedQA dataset (USMLE questions)
dataset = load_dataset("bigbio/med_qa", "med_qa_en_bigbio_qa")

# Or use the 4-options variant
dataset = load_dataset("GBaker/MedQA-USMLE-4-options")

# View a sample
print(dataset["train"][0])
# Output: {'question': 'A 45-year-old man presents with...', 'options': ['A', 'B', 'C', 'D'], 'answer': 'C', ...}
```

### 2. PubMedQA (Biomedical Research Questions)

**PubMedQA** is a question answering dataset for biomedical research that requires reasoning over research article abstracts. Each question is a research question with yes/no/maybe answers derived from PubMed article conclusions.

**Dataset Structure:**
- **PQA-L (Labeled)**: 1,000 manually annotated yes/no/maybe QA pairs
- **PQA-A (Artificial)**: 211,300 automatically generated questions with yes/no labels
- **PQA-U (Unlabeled)**: 61,200 context-question pairs
- **Question format**: Research questions requiring yes/no/maybe answers
- **Unique feature**: Requires reasoning over quantitative biomedical content

**Download from HuggingFace:**

```python
from datasets import load_dataset

# Load PubMedQA dataset
dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")

# Or use the bigbio version
dataset = load_dataset("bigbio/pubmed_qa", "pubmed_qa_labeled_fold0_bigbio_qa")

# View a sample
print(dataset["train"][0])
# Output: {'question': 'Do preoperative statins reduce atrial fibrillation after coronary artery bypass grafting?',
#          'context': '...abstract text...', 'long_answer': '...conclusion...', 'final_decision': 'yes'}
```

### 3. MedMCQA (Indian Medical Entrance Exam)

**MedMCQA** is a large-scale multiple-choice question answering dataset designed to address real-world medical entrance exam questions from India (AIIMS & NEET PG exams).

**Dataset Structure:**
- **Total questions**: 194,000+
- **Training set**: ~182,000 questions
- **Validation set**: ~4,183 questions
- **Test set**: ~6,150 questions
- **Question format**: Multiple-choice with 4 options
- **Subjects**: 2,400+ healthcare topics

**Download from HuggingFace:**

```python
from datasets import load_dataset

# Load MedMCQA dataset
dataset = load_dataset("openlifescienceai/medmcqa")

# View a sample
print(dataset["train"][0])
# Output: {'question': 'What is the most common cause of...', 'opa': 'Option A', 'opb': 'Option B',
#          'opc': 'Option C', 'opd': 'Option D', 'cop': 1, ...}
```

### 4. MultiMedQA Benchmark Suite

MultiMedQA is a comprehensive benchmark suite that combines multiple medical QA datasets used for training and evaluating Med-PaLM:

- **MedQA** (USMLE-style questions)
- **MedMCQA** (Indian medical entrance exams)
- **PubMedQA** (Biomedical research questions)
- **LiveQA** (Consumer health questions)
- **MedicationQA** (Medication-related questions)
- **HealthSearchQA** (Health search queries)
- **MMLU Clinical Topics** (Subset of Massive Multitask Language Understanding)

---

## Installation & Inference

### Important Note

Med-PaLM and Med-PaLM 2 are proprietary Google models not publicly released for direct download. Google has made Med-PaLM 2 available through Google Cloud's Vertex AI for select healthcare organizations. However, you can work with similar medical language models or use the following approaches:

### Option 1: Use Open-Source Medical LLMs

Several open-source medical language models trained on similar datasets are available:

```bash
# Install dependencies
pip install transformers torch datasets accelerate bitsandbytes
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Example: Use BioMedLM or other medical LLMs
model_name = "stanford-crfm/BioMedLM"  # 2.7B medical language model

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Inference
question = "What is the most common cause of community-acquired pneumonia?"
inputs = tokenizer(question, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_length=512,
    temperature=0.7,
    do_sample=True
)

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)
```

### Option 2: Fine-tune LLaMA/Mistral on Medical Data

```bash
# Install fine-tuning dependencies
pip install transformers datasets peft accelerate bitsandbytes
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch

# Load base model (e.g., Mistral, LLaMA 2/3)
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=True
)

# Prepare for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Load medical QA dataset
dataset = load_dataset("bigbio/med_qa", "med_qa_en_bigbio_qa")

# Format dataset for instruction tuning
def format_instruction(example):
    instruction = f"Question: {example['question']}\nAnswer:"
    return {"text": instruction + " " + example['answer']}

formatted_dataset = dataset.map(format_instruction)

# Training code would go here...
print(f"Model ready for fine-tuning on {len(formatted_dataset['train'])} samples")
```

### Option 3: Use Google Cloud Vertex AI (Med-PaLM 2)

For organizations with access to Google Cloud's Vertex AI:

```python
# Note: Requires Google Cloud credentials and Med-PaLM 2 access
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(project="your-project-id", location="us-central1")

# Med-PaLM 2 is available through Vertex AI API
# This is a conceptual example - actual implementation requires Google Cloud setup
endpoint = aiplatform.Endpoint(endpoint_name="projects/.../medpalm2-endpoint")

# Make prediction
question = "What is the recommended treatment for type 2 diabetes in adults?"
response = endpoint.predict(instances=[{"content": question}])

print(response.predictions[0])
```

### Python API Inference (Generic Medical LLM)

```python
from transformers import pipeline
import torch

# Create medical QA pipeline
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Using a medical language model
qa_pipeline = pipeline(
    "text-generation",
    model="stanford-crfm/BioMedLM",
    torch_dtype=torch.float16,
    device=device
)

# Medical question answering
questions = [
    "What are the symptoms of acute myocardial infarction?",
    "What is the mechanism of action of metformin?",
    "What are the contraindications for MRI scanning?"
]

for question in questions:
    result = qa_pipeline(
        f"Question: {question}\nAnswer:",
        max_length=256,
        temperature=0.7,
        do_sample=True
    )
    print(f"\nQ: {question}")
    print(f"A: {result[0]['generated_text']}")
```

### Expected Output Format

```json
{
  "question": "A 55-year-old woman presents with sudden onset chest pain...",
  "answer": "C",
  "explanation": "The patient's symptoms of sudden onset chest pain, elevated troponin, and ST-segment elevation on ECG are consistent with ST-elevation myocardial infarction (STEMI)...",
  "confidence": 0.92,
  "reasoning_steps": [
    "Identified key symptoms: chest pain, elevated cardiac biomarkers",
    "Analyzed ECG findings: ST-elevation",
    "Differential diagnosis: STEMI vs NSTEMI vs unstable angina",
    "Selected most appropriate answer based on clinical presentation"
  ]
}
```

---

## Benchmark Results & Performance Metrics

### Med-PaLM & Med-PaLM 2 Performance

| Model | MedQA (USMLE) | MedMCQA | PubMedQA | MMLU Clinical | Notes |
|-------|---------------|---------|----------|---------------|-------|
| **Med-PaLM 2** | **86.5%** | **72.3%** | **75.0%** | **88.3%** | Expert-level performance |
| **Med-PaLM** | 67.2% | 57.6% | 70.2% | 75.4% | First to pass USMLE threshold |
| Flan-PaLM (540B) | 62.9% | 57.9% | 73.0% | 71.2% | Base instruction-tuned model |
| GPT-4 (base) | 83.8% | 73.7% | - | - | Strong baseline comparison |
| GPT-3.5 | 53.6% | 50.1% | 65.0% | 61.3% | Pre-GPT-4 baseline |
| Human Expert Avg | ~87% | ~90% | ~78% | ~89% | Physician performance |

**Accuracy** = Percentage of questions answered correctly (higher is better)

### Med-PaLM 2 Key Achievements

1. **First Expert-Level Medical LLM**: Med-PaLM 2 was the first AI to reach expert-level performance on USMLE-style questions (86.5%)
2. **Physician-Preferred Responses**: In evaluations of 1,066 consumer medical questions, physicians preferred Med-PaLM 2 answers over physician-written answers on 8/9 clinical utility axes
3. **Improved Performance Across All Benchmarks**: 19% improvement over Med-PaLM on MedQA, 14% on MedMCQA
4. **Better Reasoning**: Ensemble refinement prompting strategy enables multi-step medical reasoning

### Evaluation Metrics Beyond Accuracy

Med-PaLM 2 was evaluated by physicians on nine axes:

| Evaluation Axis | Med-PaLM 2 | Physician Baseline | Notes |
|-----------------|------------|-------------------|-------|
| **Factuality** | High | High | Accuracy of medical facts |
| **Comprehension** | Preferred | - | Understanding of question |
| **Reasoning** | Preferred | - | Quality of medical reasoning |
| **Harm Risk** | Lower | Baseline | Potential for harmful advice |
| **Bias** | Lower | Baseline | Evidence of demographic bias |
| **Evidence of Consensus** | Preferred | - | Alignment with medical consensus |
| **Likelihood of Possible Harm** | Lower | Baseline | Risk assessment |
| **Extent of Possible Harm** | Lower | Baseline | Severity of potential harm |
| **Reading Comprehension** | Preferred | - | Understanding complex medical texts |

### Performance by Medical Domain

| Medical Domain | Med-PaLM 2 Accuracy | Notes |
|----------------|---------------------|-------|
| Internal Medicine | 88.4% | Strongest performance |
| Surgery | 84.2% | High accuracy |
| Pediatrics | 86.7% | Strong performance |
| Obstetrics & Gynecology | 85.1% | Above expert threshold |
| Psychiatry | 83.9% | Challenging domain |
| Pharmacology | 87.2% | Medication knowledge |
| Pathology | 85.8% | Diagnostic reasoning |
| Clinical Skills | 86.3% | Patient management |

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

# Install additional dependencies
pip install transformers datasets accelerate bitsandbytes sentencepiece
```

### Benchmark Script for AMD GPU - MedQA Evaluation

```python
import torch
import time
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
from typing import List, Dict

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16

# Load medical language model
model_name = "stanford-crfm/BioMedLM"  # Or your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    device_map="auto"
)

print(f"Model loaded on: {device}")
print(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Load MedQA dataset
dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test[:100]")

def format_question(example: Dict) -> str:
    """Format MedQA question as a prompt."""
    options_text = "\n".join([
        f"{k}: {v}" for k, v in example['options'].items()
    ])
    prompt = f"""Question: {example['question']}

Options:
{options_text}

Please select the most appropriate answer (A, B, C, or D) and explain your reasoning.
Answer:"""
    return prompt

def evaluate_answer(prediction: str, correct_answer: str) -> bool:
    """Extract predicted answer and check if correct."""
    # Simple extraction - look for A, B, C, or D in the response
    pred_upper = prediction.upper()
    for option in ['A', 'B', 'C', 'D']:
        if option in pred_upper[:20]:  # Check first 20 chars
            return option == correct_answer
    return False

# Benchmark
results = []
correct = 0
total_inference_time = 0

print("\nStarting MedQA evaluation...\n")

for i, example in enumerate(dataset):
    prompt = format_question(example)

    # Measure inference time
    start_time = time.time()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    end_time = time.time()
    inference_time = end_time - start_time
    total_inference_time += inference_time

    # Decode prediction
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prediction = prediction[len(prompt):].strip()  # Remove prompt from output

    # Evaluate
    is_correct = evaluate_answer(prediction, example['answer'])
    if is_correct:
        correct += 1

    results.append({
        "question_id": i,
        "question": example['question'][:100] + "...",  # Truncate for display
        "correct_answer": example['answer'],
        "prediction": prediction[:200] + "..." if len(prediction) > 200 else prediction,
        "is_correct": is_correct,
        "inference_time": inference_time
    })

    # Print progress
    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(dataset)} questions | "
              f"Accuracy: {correct/(i+1)*100:.2f}% | "
              f"Avg time: {total_inference_time/(i+1):.2f}s")

# Calculate metrics
accuracy = correct / len(dataset) * 100
avg_inference_time = total_inference_time / len(dataset)
questions_per_second = 1 / avg_inference_time

print(f"\n{'='*60}")
print(f"BENCHMARK RESULTS")
print(f"{'='*60}")
print(f"Dataset: MedQA-USMLE")
print(f"Total Questions: {len(dataset)}")
print(f"Correct Answers: {correct}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Total Inference Time: {total_inference_time:.2f}s")
print(f"Average Inference Time: {avg_inference_time:.2f}s per question")
print(f"Throughput: {questions_per_second:.2f} questions/second")
print(f"{'='*60}")

# Save results
with open("medqa_benchmark_results.json", "w") as f:
    json.dump({
        "summary": {
            "accuracy": accuracy,
            "total_questions": len(dataset),
            "correct_answers": correct,
            "avg_inference_time": avg_inference_time,
            "throughput": questions_per_second
        },
        "detailed_results": results
    }, f, indent=2)

print("\nDetailed results saved to: medqa_benchmark_results.json")
```

### Memory and Performance Monitoring Script

```python
import torch
import subprocess
import time

def get_rocm_smi_stats():
    """Get AMD GPU statistics using rocm-smi"""
    try:
        result = subprocess.run(
            ['rocm-smi', '--showuse', '--showmeminfo', 'vram', '--showpower'],
            capture_output=True,
            text=True
        )
        return result.stdout
    except FileNotFoundError:
        return "rocm-smi not found"

def monitor_gpu_metrics(duration_seconds: int = 60):
    """Monitor GPU metrics during inference"""
    metrics = {
        "memory_allocated": [],
        "memory_reserved": [],
        "timestamps": []
    }

    start_time = time.time()

    while time.time() - start_time < duration_seconds:
        if torch.cuda.is_available():
            metrics["memory_allocated"].append(
                torch.cuda.memory_allocated() / 1024**3
            )
            metrics["memory_reserved"].append(
                torch.cuda.memory_reserved() / 1024**3
            )
            metrics["timestamps"].append(time.time() - start_time)

        time.sleep(1)

    return metrics

# Usage during benchmarking
print("="*60)
print("GPU INFORMATION")
print("="*60)

if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"ROCm Version: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")

    # Memory stats
    print(f"\nCurrent Memory Usage:")
    print(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    print(f"  Max Allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    # ROCm info
    print(f"\nROCm SMI Output:")
    print(get_rocm_smi_stats())
else:
    print("No CUDA device available")

print("="*60)
```

### Complete Benchmark Script with All Metrics

```python
import torch
import time
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import psutil
import os

class MedicalLLMBenchmark:
    def __init__(self, model_name: str, dataset_name: str = "GBaker/MedQA-USMLE-4-options"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16

        # Load model
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto"
        )

        # Load dataset
        print(f"Loading dataset: {dataset_name}")
        self.dataset = load_dataset(dataset_name, split="test")

        # Results storage
        self.results = []
        self.metrics = {
            "accuracy": 0.0,
            "avg_inference_time": 0.0,
            "throughput": 0.0,
            "peak_memory_gb": 0.0,
            "avg_memory_gb": 0.0
        }

    def run_benchmark(self, num_samples: int = 100):
        """Run benchmark on specified number of samples"""
        correct = 0
        total_time = 0
        memory_usage = []

        print(f"\nRunning benchmark on {num_samples} samples...")

        for i in range(min(num_samples, len(self.dataset))):
            example = self.dataset[i]

            # Format question
            prompt = self._format_question(example)

            # Measure inference
            start_time = time.time()

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            inference_time = time.time() - start_time
            total_time += inference_time

            # Track memory
            if torch.cuda.is_available():
                memory_usage.append(torch.cuda.memory_allocated() / 1024**3)

            # Decode and evaluate
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            prediction = prediction[len(prompt):].strip()

            is_correct = self._evaluate_answer(prediction, example['answer'])
            if is_correct:
                correct += 1

            # Progress
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{num_samples} | Accuracy: {correct/(i+1)*100:.2f}%")

        # Calculate metrics
        self.metrics["accuracy"] = (correct / num_samples) * 100
        self.metrics["avg_inference_time"] = total_time / num_samples
        self.metrics["throughput"] = num_samples / total_time
        self.metrics["peak_memory_gb"] = max(memory_usage) if memory_usage else 0
        self.metrics["avg_memory_gb"] = np.mean(memory_usage) if memory_usage else 0

        return self.metrics

    def _format_question(self, example):
        """Format question for the model"""
        options_text = "\n".join([f"{k}: {v}" for k, v in example['options'].items()])
        return f"Question: {example['question']}\n\nOptions:\n{options_text}\n\nAnswer:"

    def _evaluate_answer(self, prediction: str, correct_answer: str) -> bool:
        """Check if prediction matches correct answer"""
        pred_upper = prediction.upper()
        for option in ['A', 'B', 'C', 'D']:
            if option in pred_upper[:20]:
                return option == correct_answer
        return False

    def print_results(self):
        """Print benchmark results"""
        print(f"\n{'='*60}")
        print(f"BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Accuracy: {self.metrics['accuracy']:.2f}%")
        print(f"Avg Inference Time: {self.metrics['avg_inference_time']:.3f}s")
        print(f"Throughput: {self.metrics['throughput']:.2f} questions/sec")
        print(f"Peak Memory Usage: {self.metrics['peak_memory_gb']:.2f} GB")
        print(f"Avg Memory Usage: {self.metrics['avg_memory_gb']:.2f} GB")
        print(f"{'='*60}")

# Run benchmark
if __name__ == "__main__":
    benchmark = MedicalLLMBenchmark("stanford-crfm/BioMedLM")
    results = benchmark.run_benchmark(num_samples=100)
    benchmark.print_results()
```

### Performance Metrics Table Template

| Metric | NVIDIA A100-80GB | NVIDIA H100 | AMD MI300X | AMD RX 7900 XTX | Notes |
|--------|------------------|-------------|------------|-----------------|-------|
| **GPU Model** | NVIDIA A100-80GB | NVIDIA H100 | AMD MI300X | AMD RX 7900 XTX | Compare datacenter vs consumer GPUs |
| **Memory (GB)** | 80 | 80 | 192 | 24 | VRAM capacity |
| **TDP (W)** | 400 | 700 | 750 | 355 | Thermal design power |
| **MedQA Accuracy (%)** | _[Reference]_ | _[Reference]_ | _[Your result]_ | _[Your result]_ | Medical question answering accuracy |
| **Questions Evaluated** | 1000 | 1000 | _[Your result]_ | _[Your result]_ | MedQA test set size |
| **Avg Inference Time (s)** | _[Reference]_ | _[Reference]_ | _[Your result]_ | _[Your result]_ | Per question |
| **Throughput (Q/s)** | _[Reference]_ | _[Reference]_ | _[Your result]_ | _[Your result]_ | Questions per second |
| **Peak Memory Usage (GB)** | ~45 | ~45 | _[Your result]_ | _[Your result]_ | For 7B model |
| **Average Power Draw (W)** | ~320 | ~500 | _[Your result]_ | _[Your result]_ | Monitor with rocm-smi --showpower |
| **Energy per 100 Q (Wh)** | _[Reference]_ | _[Reference]_ | _[Your result]_ | _[Your result]_ | Lower is better |

### AMD-Specific Metrics to Track

```python
# GPU utilization tracking
import subprocess

def get_rocm_detailed_stats():
    """Get detailed AMD GPU statistics"""
    stats = {}

    # Memory info
    mem_result = subprocess.run(
        ['rocm-smi', '--showmeminfo', 'vram'],
        capture_output=True,
        text=True
    )
    stats['memory'] = mem_result.stdout

    # Power usage
    power_result = subprocess.run(
        ['rocm-smi', '--showpower'],
        capture_output=True,
        text=True
    )
    stats['power'] = power_result.stdout

    # GPU utilization
    use_result = subprocess.run(
        ['rocm-smi', '--showuse'],
        capture_output=True,
        text=True
    )
    stats['utilization'] = use_result.stdout

    return stats

# Memory tracking with PyTorch
if torch.cuda.is_available():
    print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    # ROCm version info
    if hasattr(torch.version, 'hip'):
        print(f"ROCm Version: {torch.version.hip}")

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
```

### Complete Runtime Metrics Table

| Runtime Metric | Formula | NVIDIA A100-80GB | AMD MI300X | AMD RX 7900 XTX | Notes |
|----------------|---------|------------------|------------|-----------------|-------|
| **Inference Time (s/question)** | total_time / num_questions | _[Reference]_ | _[Your result]_ | _[Your result]_ | Lower is better |
| **Throughput (Q/s)** | num_questions / total_time | _[Reference]_ | _[Your result]_ | _[Your result]_ | Questions per second |
| **Tokens Generated (avg)** | output_tokens / num_questions | _[Reference]_ | _[Your result]_ | _[Your result]_ | Average response length |
| **Tokens Per Second** | total_tokens / total_time | _[Reference]_ | _[Your result]_ | _[Your result]_ | Generation speed |
| **GPU Utilization (%)** | From rocm-smi | _[Reference]_ | _[Your result]_ | _[Your result]_ | Average during inference |
| **Memory Bandwidth (GB/s)** | From rocm-smi | ~2.0 TB/s | _[Your result]_ | _[Your result]_ | MI300X: ~5.3 TB/s theoretical |
| **TFLOPS Utilized** | Calculated from operations | _[Reference]_ | _[Your result]_ | _[Your result]_ | FP16/BF16 compute |
| **Time to First Token (ms)** | First token latency | _[Reference]_ | _[Your result]_ | _[Your result]_ | Important for responsiveness |
| **Energy Efficiency (Wh/100Q)** | power × time / questions | _[Reference]_ | _[Your result]_ | _[Your result]_ | Lower is better |
| **Cost per 1M Questions** | Based on energy cost | _[Reference]_ | _[Your result]_ | _[Your result]_ | Economic efficiency |

---

## Open Medical LLM Leaderboard

The [Open Medical LLM Leaderboard](https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard) evaluates medical language models across multiple datasets and clinical tasks.

### Evaluation Datasets

- **MedQA** (USMLE-style questions)
- **MedMCQA** (Indian medical entrance exams)
- **PubMedQA** (Biomedical research QA)
- **MMLU Medical Genetics** (Medical genetics knowledge)
- **MMLU Anatomy** (Anatomical knowledge)
- **MMLU Clinical Knowledge** (Clinical practice)
- **MMLU Professional Medicine** (Medical professionalism)
- **MMLU College Biology** (Biology fundamentals)
- **MMLU College Medicine** (Medical education)

### Key Metrics Tracked

- **Accuracy** (primary metric across all datasets)
- **Average Score** (mean performance across all medical benchmarks)
- **Model Size** (parameters, disk size)
- **Clinical Utility Scores** (human evaluation when available)

### Top Performing Medical LLMs (Open Source)

| Model | Avg Score | MedQA | MedMCQA | PubMedQA | Parameters | Notes |
|-------|-----------|-------|---------|----------|------------|-------|
| GPT-4 (reference) | ~83% | 83.8% | 73.7% | ~75% | Unknown | Proprietary baseline |
| Med42-v2-70B | ~78% | 75.4% | 71.2% | 76.8% | 70B | Specialized medical model |
| MediTron-70B | ~74% | 71.3% | 68.9% | 73.2% | 70B | Clinical notes training |
| BioMistral-7B | ~65% | 62.4% | 59.8% | 68.1% | 7B | Efficient medical LLM |
| Med-Gemini (Google) | ~86% | 86.5% | 72.3% | 75.0% | Unknown | Proprietary (Med-PaLM successor) |

**Note:** Med-PaLM and Med-PaLM 2 are proprietary Google models not on the public leaderboard, but their published results serve as important benchmarks for the field.

---

## Additional Resources

### Official Resources

- [Google Med-PaLM Research Site](https://sites.research.google/gr/med-palm/)
- [Google Cloud Med-PaLM 2 Announcement](https://cloud.google.com/blog/topics/healthcare-life-sciences/sharing-google-med-palm-2-medical-large-language-model)
- [Med-Gemini (Successor to Med-PaLM)](https://research.google/blog/advancing-medical-ai-with-med-gemini/)

### Papers & Documentation

- [Med-PaLM Paper - Nature (arXiv:2212.13138)](https://www.nature.com/articles/s41586-023-06291-2)
- [Med-PaLM 2 Paper - Nature Medicine (arXiv:2305.09617)](https://arxiv.org/abs/2305.09617)
- [Med-PaLM 2 Paper - PDF](https://arxiv.org/pdf/2305.09617)
- [PubMedQA Paper (arXiv:1909.06146)](https://arxiv.org/abs/1909.06146)
- [MedQA Original Paper](https://github.com/jind11/MedQA)
- [Med-PaLM Explained - Encord Blog](https://encord.com/blog/med-palm-explained/)

### Datasets

- [MedQA (bigbio/med_qa)](https://huggingface.co/datasets/bigbio/med_qa)
- [MedQA USMLE 4-options (GBaker)](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options)
- [PubMedQA (qiaojin/PubMedQA)](https://huggingface.co/datasets/qiaojin/PubMedQA)
- [MedMCQA (openlifescienceai/medmcqa)](https://huggingface.co/datasets/openlifescienceai/medmcqa)
- [MMLU Medical Topics](https://huggingface.co/datasets/cais/mmlu)

### Leaderboards & Benchmarks

- [Open Medical LLM Leaderboard](https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard)
- [MultiMedQA Benchmark Suite](https://sites.research.google/gr/med-palm/)
- [HuggingFace Medical Tasks](https://huggingface.co/tasks)

### Open-Source Medical LLMs

- [BioMedLM (2.7B) - Stanford](https://huggingface.co/stanford-crfm/BioMedLM)
- [Med42-v2-70B](https://huggingface.co/m42-health/med42-v2-70b)
- [MediTron-70B](https://huggingface.co/epfl-llm/meditron-70b)
- [BioMistral-7B](https://huggingface.co/BioMistral/BioMistral-7B)
- [Clinical-T5](https://huggingface.co/luqh/ClinicalT5-large)

### Blog Posts & Articles

- [Google Cloud: Sharing Med-PaLM 2](https://cloud.google.com/blog/topics/healthcare-life-sciences/sharing-google-med-palm-2-medical-large-language-model)
- [Medium: Med-PaLM 2 Review](https://sh-tsang.medium.com/brief-review-med-palm-2-towards-expert-level-medical-question-answering-with-large-language-3dff443a69a9)
- [Medium: Med-PaLM 2 Ensemble Refinement](https://medium.com/@kapiblue6/med-palm-2-medical-question-answering-improved-by-ensemble-refinement-af7064e089ee)
- [AMD ROCm Performance Results](https://www.amd.com/en/developer/resources/rocm-hub/dev-ai/performance-results.html)

### Tools & Libraries

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [LangChain Medical Agents](https://python.langchain.com/docs/use_cases/question_answering/)
- [ROCm Documentation](https://rocm.docs.amd.com/)

---

## Quick Reference Commands

```bash
# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
pip install transformers datasets accelerate bitsandbytes

# Download MedQA dataset
python -c "from datasets import load_dataset; ds = load_dataset('GBaker/MedQA-USMLE-4-options')"

# Download PubMedQA dataset
python -c "from datasets import load_dataset; ds = load_dataset('qiaojin/PubMedQA')"

# Download MedMCQA dataset
python -c "from datasets import load_dataset; ds = load_dataset('openlifescienceai/medmcqa')"

# Check AMD GPU status
rocm-smi
rocm-smi --showuse --showmeminfo vram --showpower

# Monitor GPU during inference
watch -n 1 rocm-smi

# Run benchmark script
python medqa_benchmark.py

# Check PyTorch CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## Important Notes

### Model Availability

- **Med-PaLM and Med-PaLM 2 are proprietary Google models** not publicly available for download
- Google provides access to Med-PaLM 2 through **Google Cloud Vertex AI** for select healthcare organizations
- This benchmark guide provides instructions for evaluating **open-source medical LLMs** on the same datasets used to evaluate Med-PaLM
- Fine-tuning approaches using LLaMA, Mistral, or other base models on medical data can approximate Med-PaLM's capabilities

### Clinical Use Considerations

- These models are for research purposes only and **should not be used for actual medical diagnosis or treatment**
- All medical AI systems require validation by healthcare professionals
- Med-PaLM 2's physician-preferred responses still require human oversight
- Patient safety and regulatory compliance are paramount in any clinical deployment

### Benchmarking Best Practices

- Use the full test sets when possible (MedQA: ~1,000 questions, MedMCQA: ~6,000 questions)
- Report both accuracy and inference performance metrics
- Include human evaluation for clinical utility when feasible
- Test across multiple medical domains and specialties
- Monitor for potential biases and harmful outputs

---

**Document Version:** 1.0
**Last Updated:** March 2026
**Target Hardware:** AMD MI300X, RX 7900 XTX, and other ROCm-compatible GPUs
**Model Type:** Medical Large Language Model (Proprietary - Google Research)
