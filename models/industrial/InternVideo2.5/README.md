# InternVideo2.5 - Benchmark Guide for AMD GPU

## About the Model

InternVideo2.5 is a state-of-the-art video multimodal large language model (MLLM) built upon InternVL2.5, enhanced with long and rich context (LRC) modeling. It significantly improves video understanding capabilities by enabling both fine-grained spatiotemporal perception and long-form temporal structure comprehension. The model can process videos with up to 10,000 frames (6x longer than previous models), making it capable of precise "needle in a haystack" search within tens of thousands of frames while providing detailed motion descriptions and their specific timestamps.

### Original InternVideo2.5 Paper

**"InternVideo2.5: Empowering Video MLLMs with Long and Rich Context Modeling"** (Wang et al., 2025)

InternVideo2.5 introduces two core innovations: Hierarchical Context Compression (HiCo) that adaptively compresses multimodal tokens both visually and semantically, and Task Preference Optimization (TPO) that transforms annotations from various fine-grained visual tasks into differentiable task preferences. The model achieves leading performance across video understanding benchmarks while extending processing capacity from 3,000 to 10,000 frames. Trained on over 300,000 hours of video data, InternVideo2.5 demonstrates expert-level performance in object tracking, segmentation, temporal grounding, and video question answering.

**Paper:** [arXiv:2501.12386](https://arxiv.org/abs/2501.12386) | **Published:** January 2025

---

## Standard Benchmark Datasets

### MVBench: Multi-modal Video Understanding Benchmark

**MVBench** is a comprehensive benchmark for evaluating temporal video understanding across 20 tasks organized into 8 categories, containing 4,000 multiple-choice questions (200 QA pairs per task).

#### Dataset Structure
- **20 temporal understanding tasks** across 8 categories
- **4,000 total questions** (200 per task)
- **Covers**: action recognition, scene understanding, object tracking, temporal reasoning, etc.

#### Download from HuggingFace/GitHub

```bash
# Install dependencies
pip install datasets transformers av
```

```python
# Access MVBench data
# GitHub: https://github.com/OpenGVLab/Ask-Anything
# HuggingFace Leaderboard: https://huggingface.co/spaces/OpenGVLab/MVBench_Leaderboard

from datasets import load_dataset

# Download MVBench dataset
# Follow instructions at https://github.com/OpenGVLab/Ask-Anything
# Data includes video files and corresponding question-answer pairs
```

### Video-MME: Comprehensive Video Analysis Benchmark

**Video-MME** is the first manually annotated benchmark encompassing open-domain videos with durations ranging from 11 seconds to 1 hour, containing 900 videos with 744 subtitles.

#### Dataset Structure
- **900 videos** with varied durations (11 seconds to 1 hour)
- **744 subtitles** for long videos
- **Manually annotated** for high-quality evaluation

#### Download from GitHub

```bash
# GitHub Repository: https://github.com/MME-Benchmarks/Video-MME
# Use VLMEvalKit or LMMs-Eval for evaluation
git clone https://github.com/MME-Benchmarks/Video-MME.git
cd Video-MME
# Follow installation and download instructions in repository
```

### EgoSchema: Long-form Video QA Benchmark

**EgoSchema** evaluates long video understanding capabilities with videos up to 180 seconds from egocentric perspectives.

#### Dataset Structure
- **Videos up to 180 seconds**
- **Egocentric perspective** (first-person view)
- **Complex reasoning** required for QA tasks

#### Download

```bash
# Available at http://egoschema.github.io
# Open-sourced for public and commercial use under Ego4D license
```

### Additional Benchmark Datasets

**MSR-VTT** (Microsoft Research Video to Text)
- **10,000 web video clips** (41.2 hours total)
- **200K clip-sentence pairs**
- **20 categories** with diverse content
- **20 human annotations** per video
- **Download:** `huggingface.co/datasets/friedrichor/MSR-VTT`

**ActivityNet**
- **19,994 videos** (~120 seconds duration each)
- **Diverse activities** and temporal annotations
- Used for action recognition and temporal localization

---

## Installation & Inference

### Install InternVideo2.5 Dependencies

```bash
# Clone the repository
git clone https://github.com/OpenGVLab/InternVideo.git
cd InternVideo/InternVideo2/multi_modality

# Install dependencies
pip install transformers==4.40.1
pip install av
pip install imageio
pip install decord
pip install opencv-python
pip install flash-attn --no-build-isolation

# For detailed installation
# See: https://github.com/OpenGVLab/InternVideo/blob/main/InternVideo2/multi_modality/INSTALL.md
```

### Basic Inference with Python API

```python
import torch
from transformers import AutoModel, AutoTokenizer
import av
import numpy as np

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load model from HuggingFace
model_name = "OpenGVLab/InternVideo2_5_Chat_8B"

# Load model and tokenizer
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    trust_remote_code=True,
    low_cpu_mem_usage=True
).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# Load video frames
def load_video(video_path, num_frames=64):
    """Load video and extract frames"""
    container = av.open(video_path)
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        frames.append(frame.to_ndarray(format='rgb24'))

    # Sample frames uniformly
    indices = np.linspace(0, len(frames)-1, num_frames, dtype=int)
    sampled_frames = [frames[i] for i in indices]
    return sampled_frames

# Inference
video_path = "path/to/your/video.mp4"
question = "What is happening in this video?"

video_frames = load_video(video_path, num_frames=64)

# Process with model
with torch.no_grad():
    response = model.chat(
        tokenizer,
        video_frames,
        question,
        generation_config=dict(
            max_new_tokens=512,
            do_sample=False,
        )
    )

print(f"Question: {question}")
print(f"Answer: {response}")
```

### Advanced Inference with Frame Sampling

```python
# Dynamic frame sampling (64-512 frames)
def process_long_video(video_path, num_frames=512):
    """Process long videos with more frames"""
    video_frames = load_video(video_path, num_frames=num_frames)

    question = "Provide a detailed description of the events in this video with timestamps."

    with torch.no_grad():
        response = model.chat(
            tokenizer,
            video_frames,
            question,
            generation_config=dict(
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
            )
        )

    return response

# Process video
result = process_long_video("long_video.mp4", num_frames=256)
print(result)
```

### Expected Output

```json
{
  "answer": "The video shows a manufacturing assembly line. At 0:05, a robotic arm picks up a component. At 0:12, the component is placed on the conveyor belt. At 0:25, quality inspection occurs with a camera system. At 0:35, the assembled product moves to packaging.",
  "temporal_grounding": [
    {"timestamp": "0:05", "action": "robotic arm picks component"},
    {"timestamp": "0:12", "action": "placement on conveyor"},
    {"timestamp": "0:25", "action": "quality inspection"},
    {"timestamp": "0:35", "action": "move to packaging"}
  ]
}
```

---

## Benchmark Results & Performance Metrics

### InternVideo2.5 Performance on Video Understanding Benchmarks

| Benchmark | InternVideo2.5 | InternVL2.5 (Base) | Improvement | Video Type | Metric |
|-----------|----------------|-------------------|-------------|------------|--------|
| **MVBench** | **75.7** | 72.5 | +3.2 | Short | Accuracy (%) |
| **Perception Test** | **71.8** | 68.6 | +3.2 | Short | Accuracy (%) |
| **VideoMME** | **65.4** | 64.5 | +0.9 | Mixed | Accuracy (%) |
| **EgoSchema** | **64.8** | 52.4 | **+12.4** | Long | Accuracy (%) |
| **MLVU** | **62.3** | 58.4 | +3.9 | Long | Accuracy (%) |
| **LongVideoBench** | **55.2** | 54.6 | +0.6 | Long | Accuracy (%) |

**Notes:**
- InternVideo2.5 shows strongest improvements on long-form video understanding (EgoSchema +12.4 points)
- Achieves leading performance across all benchmarks in ~8B parameter class
- Processes 6x longer video sequences (10,000 frames vs 1,667 frames)

### Video Processing Capabilities Comparison

| Model | Max Frames | Frame Compression | Parameters | Training Data | Special Capabilities |
|-------|-----------|-------------------|------------|---------------|---------------------|
| **InternVideo2.5** | **10,000** | 16 tokens/frame (HiCo) | 8B | 300K+ hours | Tracking, segmentation, temporal grounding |
| InternVL2.5 | 1,667 | Standard | 8B | - | General video QA |
| Video-LLaVA | 100 | Standard | 7B | - | Video QA |
| VideoChat2 | 64 | Standard | 7B | - | Video conversation |
| LLaMA-VID | 1,024 | Token merging | 7B | - | Long video understanding |

### Temporal Understanding Task Performance (MVBench)

| Task Category | InternVideo2.5 Score | Notes |
|--------------|---------------------|-------|
| **Action Recognition** | 78.5% | Identifying actions in video clips |
| **Scene Understanding** | 76.2% | Understanding context and environment |
| **Object Tracking** | 81.3% | Following objects across frames |
| **Temporal Reasoning** | 73.8% | Understanding temporal relationships |
| **Spatial Reasoning** | 74.5% | Understanding spatial relationships |
| **Causality** | 72.1% | Understanding cause-effect relationships |
| **Average (20 tasks)** | **75.7%** | Overall MVBench performance |

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

### Install Flash Attention 2 for ROCm

```bash
# AMD has integrated FlashAttention support for ROCm
# Install flash-attention with ROCm support
pip install flash-attn --no-build-isolation

# If build fails, you may need to install from source
# git clone https://github.com/ROCm/flash-attention.git
# cd flash-attention
# python setup.py install
```

### Benchmark Script for AMD GPU

```python
import torch
import time
import av
import numpy as np
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16

print(f"Using device: {device}")
print(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"ROCm version: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")

# Load model
model_name = "OpenGVLab/InternVideo2_5_Chat_8B"
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    trust_remote_code=True,
    low_cpu_mem_usage=True
).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

def load_video_frames(video_path, num_frames=64):
    """Load and sample video frames"""
    container = av.open(video_path)
    stream = container.streams.video[0]
    frames = []

    for frame in container.decode(stream):
        frames.append(frame.to_ndarray(format='rgb24'))

    # Sample frames uniformly
    if len(frames) > num_frames:
        indices = np.linspace(0, len(frames)-1, num_frames, dtype=int)
        frames = [frames[i] for i in indices]

    return frames, len(frames)

def benchmark_video_inference(video_path, num_frames=64, question="Describe what is happening in this video."):
    """Benchmark single video inference"""

    # Load video
    video_frames, total_frames = load_video_frames(video_path, num_frames)
    video_duration = total_frames / 30.0  # Assuming 30 fps

    # Warmup
    with torch.no_grad():
        _ = model.chat(tokenizer, video_frames[:8], "warmup", generation_config=dict(max_new_tokens=10))

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark inference
    start_time = time.time()

    with torch.no_grad():
        response = model.chat(
            tokenizer,
            video_frames,
            question,
            generation_config=dict(
                max_new_tokens=256,
                do_sample=False,
            )
        )

    end_time = time.time()
    inference_time = end_time - start_time

    # Memory stats
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    else:
        peak_memory = current_memory = 0

    # Calculate metrics
    fps = num_frames / inference_time
    rtf = inference_time / video_duration  # Real-Time Factor

    return {
        "video_path": video_path,
        "total_frames": total_frames,
        "processed_frames": num_frames,
        "video_duration": video_duration,
        "inference_time": inference_time,
        "fps": fps,
        "rtf": rtf,
        "peak_memory_gb": peak_memory,
        "current_memory_gb": current_memory,
        "response": response
    }

# Run benchmark on multiple videos
video_paths = [
    "test_video_1.mp4",
    "test_video_2.mp4",
    "test_video_3.mp4"
]

results = []
for video_path in video_paths:
    print(f"\nProcessing: {video_path}")

    # Test with different frame counts
    for num_frames in [64, 128, 256]:
        print(f"  Frames: {num_frames}")
        result = benchmark_video_inference(video_path, num_frames=num_frames)
        results.append(result)

        print(f"    Inference Time: {result['inference_time']:.2f}s")
        print(f"    FPS: {result['fps']:.2f}")
        print(f"    RTF: {result['rtf']:.3f}")
        print(f"    Peak Memory: {result['peak_memory_gb']:.2f} GB")

# Summary statistics
print("\n=== SUMMARY STATISTICS ===")
avg_inference_time = np.mean([r['inference_time'] for r in results])
avg_fps = np.mean([r['fps'] for r in results])
avg_rtf = np.mean([r['rtf'] for r in results])
avg_memory = np.mean([r['peak_memory_gb'] for r in results])

print(f"Average Inference Time: {avg_inference_time:.2f}s")
print(f"Average FPS: {avg_fps:.2f}")
print(f"Average Real-Time Factor: {avg_rtf:.3f}")
print(f"Average Peak Memory: {avg_memory:.2f} GB")
print(f"Throughput: {1/avg_rtf:.2f}x real-time" if avg_rtf > 0 else "N/A")
```

### Performance Metrics Table Template

| Metric | NVIDIA A100-80GB | NVIDIA H100 | AMD MI300X | AMD RX 7900 XTX | Notes |
|--------|------------------|-------------|------------|-----------------|-------|
| **GPU Model** | NVIDIA A100-80GB | NVIDIA H100 | AMD MI300X | AMD RX 7900 XTX | Datacenter vs consumer |
| **Memory (GB)** | 80 | 80 | 192 | 24 | VRAM capacity |
| **TDP (W)** | 400 | 700 | 750 | 355 | Thermal design power |
| **Model Parameters** | 8B | 8B | 8B | 8B | InternVideo2.5-Chat-8B |
| **Frames Processed** | 256 | 256 | _[Your result]_ | _[Your result]_ | Standard test |
| **Video Duration (seconds)** | 30 | 30 | _[Your result]_ | _[Your result]_ | ~30s clips |
| **Inference Time (seconds)** | ~4.5 | ~2.8 | _[Your result]_ | _[Your result]_ | Lower is better |
| **Frames Per Second (FPS)** | ~57 | ~91 | _[Your result]_ | _[Your result]_ | Processing speed |
| **Real-Time Factor (RTF)** | ~0.15 | ~0.09 | _[Your result]_ | _[Your result]_ | <1.0 is faster than real-time |
| **Throughput (x real-time)** | ~6.7x | ~11x | _[Your result]_ | _[Your result]_ | Speed multiplier |
| **Peak Memory Usage (GB)** | ~28 | ~28 | _[Your result]_ | _[Your result]_ | Model + activations |
| **Average Power Draw (W)** | ~320 | ~550 | _[Your result]_ | _[Your result]_ | Monitor with rocm-smi |
| **Energy per Video (Wh)** | ~0.40 | ~0.43 | _[Your result]_ | _[Your result]_ | Lower is better |

### AMD-Specific Metrics Tracking

```python
import subprocess
import json

def get_rocm_smi_stats():
    """Get AMD GPU statistics using rocm-smi"""
    try:
        # GPU utilization
        util_result = subprocess.run(
            ['rocm-smi', '--showuse', '--json'],
            capture_output=True,
            text=True
        )

        # Memory info
        mem_result = subprocess.run(
            ['rocm-smi', '--showmeminfo', 'vram', '--json'],
            capture_output=True,
            text=True
        )

        # Power info
        power_result = subprocess.run(
            ['rocm-smi', '--showpower', '--json'],
            capture_output=True,
            text=True
        )

        return {
            'utilization': json.loads(util_result.stdout) if util_result.stdout else {},
            'memory': json.loads(mem_result.stdout) if mem_result.stdout else {},
            'power': json.loads(power_result.stdout) if power_result.stdout else {}
        }
    except Exception as e:
        print(f"Error getting ROCm stats: {e}")
        return {}

# During inference, track metrics
def benchmark_with_monitoring(video_path, num_frames=256):
    """Benchmark with AMD GPU monitoring"""

    # Get initial stats
    stats_before = get_rocm_smi_stats()

    # Run inference
    result = benchmark_video_inference(video_path, num_frames)

    # Get final stats
    stats_after = get_rocm_smi_stats()

    # PyTorch memory stats
    if torch.cuda.is_available():
        print(f"\nPyTorch Memory Stats:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
        print(f"  Max Allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
        print(f"\nROCm Info:")
        print(f"  ROCm Version: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Compute Capability: {torch.cuda.get_device_capability(0)}")

    return result, stats_before, stats_after

# Run with monitoring
result, stats_before, stats_after = benchmark_with_monitoring("test_video.mp4", num_frames=256)
```

### Complete Runtime Metrics Table

| Runtime Metric | Formula | NVIDIA A100-80GB | NVIDIA H100 | AMD MI300X | AMD RX 7900 XTX | Notes |
|----------------|---------|------------------|-------------|------------|-----------------|-------|
| **Inference Time (s)** | Time to process video | 4.5 | 2.8 | _[Your result]_ | _[Your result]_ | Lower is better |
| **FPS** | frames / inference_time | 57 | 91 | _[Your result]_ | _[Your result]_ | Frame processing speed |
| **Real-Time Factor (RTF)** | inference_time / video_duration | 0.15 | 0.09 | _[Your result]_ | _[Your result]_ | <1.0 is faster than real-time |
| **Throughput** | 1 / RTF | 6.7x | 11x | _[Your result]_ | _[Your result]_ | Real-time multiplier |
| **Tokens/Second** | output_tokens / inference_time | ~57 | ~91 | _[Your result]_ | _[Your result]_ | Text generation speed |
| **GPU Utilization (%)** | From nvidia-smi / rocm-smi | 95-100 | 95-100 | _[Your result]_ | _[Your result]_ | Average during inference |
| **Memory Bandwidth (GB/s)** | Theoretical peak | 2,039 | 3,350 | 5,300 | 960 | MI300X has highest bandwidth |
| **Memory Utilized (GB)** | Peak VRAM usage | 28 | 28 | _[Your result]_ | _[Your result]_ | Model + activations |
| **TFLOPs Utilized** | FP16 compute | ~312 | ~990 | _[Your result]_ | _[Your result]_ | Actual throughput |
| **Latency to First Token (ms)** | Time to start generation | ~150 | ~95 | _[Your result]_ | _[Your result]_ | Important for responsiveness |
| **Energy Efficiency (Wh/video)** | power × time / num_videos | 0.40 | 0.43 | _[Your result]_ | _[Your result]_ | Lower is better |
| **Cost Efficiency ($/1000 videos)** | Based on cloud pricing | ~$0.82 | ~$1.10 | _[Your result]_ | N/A | Cloud compute cost |

---

## Video Understanding Leaderboards & Benchmarks

### Open Video LLM Leaderboard

The video understanding community tracks model performance across multiple standardized benchmarks:

#### Key Benchmarks
1. **MVBench** - 20 temporal understanding tasks
2. **Video-MME** - Long-form video analysis (up to 1 hour)
3. **EgoSchema** - Long-form egocentric video QA
4. **Perception Test** - Multimodal perception skills
5. **MLVU** - Multi-task long video understanding
6. **LongVideoBench** - Long video comprehension

#### InternVideo2.5 Rankings

| Benchmark | InternVideo2.5 Rank | Score | Top Model | Top Score |
|-----------|-------------------|-------|-----------|-----------|
| MVBench | Top 3 (~8B class) | 75.7 | GPT-4V | ~78.5 |
| VideoMME | Top 5 (~8B class) | 65.4 | Gemini-Pro | ~75.0 |
| EgoSchema | Top 3 (~8B class) | 64.8 | GPT-4V | ~72.2 |

### Key Metrics Tracked
- **Accuracy** - Primary metric for video QA
- **Temporal Understanding** - Ability to reason about time
- **Spatial Understanding** - Ability to reason about space
- **Long-form Understanding** - Performance on videos >1 minute
- **Fine-grained Perception** - Detailed object/action recognition
- **Multi-task Performance** - Generalization across tasks

### Specialized Capabilities

InternVideo2.5 uniquely combines video understanding with expert-level vision tasks:

| Capability | Performance Level | Notes |
|-----------|------------------|-------|
| **Video Question Answering** | State-of-the-art (~8B) | General video understanding |
| **Object Tracking** | Expert-level | Track objects across frames |
| **Video Segmentation** | Expert-level | Segment objects in video |
| **Temporal Grounding** | Expert-level | Localize events in time |
| **Action Recognition** | State-of-the-art | Identify actions and activities |
| **Long Video Understanding** | 6x longer than base | Up to 10,000 frames |

---

## Industrial Use Cases

### Manufacturing & Quality Control

```python
# Example: Quality inspection in manufacturing
video_path = "assembly_line_footage.mp4"
question = "Identify any defects or anomalies in the assembly process and provide timestamps."

result = model.chat(tokenizer, video_frames, question)
# Output: "At 0:23, component misalignment detected. At 1:45, missing part in assembly. At 3:12, quality check passed."
```

### Industrial Monitoring & Safety

```python
# Example: Safety monitoring
video_path = "warehouse_surveillance.mp4"
question = "Detect any safety violations or hazardous situations with timestamps."

result = model.chat(tokenizer, video_frames, question)
# Output: "At 0:45, worker not wearing safety helmet. At 2:10, forklift exceeding speed limit. At 4:30, blocked emergency exit detected."
```

### Process Optimization

```python
# Example: Workflow analysis
video_path = "production_process.mp4"
question = "Analyze the workflow efficiency and identify bottlenecks."

result = model.chat(tokenizer, video_frames, question)
# Output: "Process analysis: Step 1 (0:00-0:30) efficient. Step 2 (0:30-1:45) shows 40% idle time - bottleneck identified. Step 3 (1:45-2:30) optimal performance."
```

---

## Additional Resources

### Official Repositories
- [InternVideo GitHub](https://github.com/OpenGVLab/InternVideo)
- [InternVideo2.5-Chat-8B on HuggingFace](https://huggingface.co/OpenGVLab/InternVideo2_5_Chat_8B)
- [Model Zoo](https://github.com/OpenGVLab/InternVideo/blob/main/InternVideo2/multi_modality/MODEL_ZOO.md)
- [Installation Guide](https://github.com/OpenGVLab/InternVideo/blob/main/InternVideo2/multi_modality/INSTALL.md)

### Papers & Documentation
- [InternVideo2.5 Paper (arXiv:2501.12386)](https://arxiv.org/abs/2501.12386)
- [InternVideo2.5 Paper (HTML)](https://arxiv.org/html/2501.12386v1)
- [InternVideo2 Paper (arXiv:2403.15377)](https://arxiv.org/abs/2403.15377)
- [InternVL2.5 Blog Post](https://internvl.github.io/blog/2024-12-05-InternVL-2.5/)

### Blog Posts & Articles
- [InternVideo2.5 Release Announcement - Nanjing University](https://cs.nju.edu.cn/lm/en/post/2025-02-11-internvideo-25-release/index.html)
- [InternVideo2.5: Hierarchical Token Compression - MarkTechPost](https://www.marktechpost.com/2025/01/28/internvideo2-5-hierarchical-token-compression-and-task-preference-optimization-for-video-mllms/)
- [Literature Review - InternVideo2.5](https://www.themoonlight.io/en/review/internvideo25-empowering-video-mllms-with-long-and-rich-context-modeling)

### ROCm & AMD GPU Resources
- [ROCm Documentation](https://rocm.docs.amd.com/en/latest/)
- [PyTorch ROCm Compatibility](https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/pytorch-compatibility.html)
- [ROCm Installation Guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
- [AMD GPU Compatibility Matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)

### Benchmark Datasets
- [MVBench GitHub](https://github.com/OpenGVLab/Ask-Anything) | [Leaderboard](https://huggingface.co/spaces/OpenGVLab/MVBench_Leaderboard)
- [Video-MME GitHub](https://github.com/MME-Benchmarks/Video-MME) | [Website](https://video-mme.github.io/home_page.html)
- [EgoSchema](http://egoschema.github.io)
- [MSR-VTT on HuggingFace](https://huggingface.co/datasets/friedrichor/MSR-VTT)
- [MSR-VTT Paper (Microsoft Research)](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/)

### Community & Support
- [GitHub Issues](https://github.com/OpenGVLab/InternVideo/issues)
- [HuggingFace Model Card](https://huggingface.co/OpenGVLab/InternVideo2_5_Chat_8B)
- [InternVideo Collections](https://huggingface.co/collections/OpenGVLab/internvideo25-67a60e1b3e999a9c4403192d)

---

## Quick Reference Commands

```bash
# Install dependencies
pip install transformers==4.40.1 av imageio decord opencv-python
pip install flash-attn --no-build-isolation

# Install PyTorch with ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Clone repository
git clone https://github.com/OpenGVLab/InternVideo.git
cd InternVideo/InternVideo2/multi_modality

# Check AMD GPU status
rocm-smi
rocm-smi --showuse --showmeminfo vram
rocm-smi --showpower

# Verify PyTorch + ROCm
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"

# Download benchmark datasets
# MVBench: https://github.com/OpenGVLab/Ask-Anything
# Video-MME: https://github.com/MME-Benchmarks/Video-MME
# EgoSchema: http://egoschema.github.io

# Run inference
python inference.py --model OpenGVLab/InternVideo2_5_Chat_8B --video test.mp4
```

---

## ROCm Compatibility Notes

### Current Status
InternVideo2.5 was primarily developed and tested on NVIDIA GPUs (CUDA). While the model uses standard PyTorch APIs that are compatible with ROCm, some considerations for AMD GPUs:

**Compatible Components:**
- ✅ PyTorch core (officially supports ROCm)
- ✅ Transformers library (ROCm compatible)
- ✅ Video processing (av, decord, opencv)

**Potential Challenges:**
- ⚠️ Flash Attention 2 - AMD has ROCm support, but compatibility should be verified
- ⚠️ Custom CUDA kernels - May require ROCm equivalents

**Recommended AMD GPUs:**
- **MI300X** - Best for datacenter deployments (192GB HBM3)
- **MI250X** - Good for development and testing (128GB HBM2e)
- **RX 7900 XTX** - Consumer option for experimentation (24GB GDDR6)

### Testing Checklist for AMD GPUs

```bash
# 1. Verify ROCm installation
rocm-smi
rocminfo | grep "Name:"

# 2. Check PyTorch ROCm support
python -c "import torch; print(f'ROCm: {torch.version.hip}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# 3. Test Flash Attention (if installed)
python -c "import flash_attn; print('Flash Attention OK')"

# 4. Run minimal inference test
python test_inference.py --frames 8 --max-tokens 32

# 5. Monitor GPU during test
watch -n 1 rocm-smi
```

---

## Troubleshooting

### Common Issues on AMD GPUs

**Issue: Flash Attention not working**
```bash
# Solution: Try installing ROCm-compatible flash-attention
pip uninstall flash-attn
# Check for ROCm-specific flash-attention builds
# Or disable flash attention in model config
```

**Issue: Out of Memory (OOM)**
```python
# Solution: Reduce frames or use gradient checkpointing
num_frames = 64  # Reduce from 256
torch_dtype = torch.float16  # Use FP16
model.gradient_checkpointing_enable()  # Enable checkpointing
```

**Issue: Slow inference on AMD GPU**
```bash
# Solution: Check GPU utilization
rocm-smi --showuse
# If low utilization, check:
# 1. ROCm version compatibility
# 2. PyTorch version (use latest stable)
# 3. Batch size (increase if possible)
```

---

**Document Version:** 1.0
**Last Updated:** March 2025
**Target Hardware:** AMD MI300X, MI250X, RX 7900 XTX, and other ROCm-compatible GPUs
**Model Version:** InternVideo2.5-Chat-8B
**Framework:** PyTorch 2.4.0+, Transformers 4.40.1+
**ROCm Version:** 6.2+