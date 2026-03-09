# In-Vehicle Voice Assistant

## Overview
Hybrid heterogeneous voice assistant system for automotive applications

## Category
Automotive IVI System

## Components
### Encoder Models
- Whisper Encoder (Speech-to-Text)
- CLIP Vision Encoder
- Other Encoders

### Decoder Models
- LLM Decoder (Language Understanding)
- TTS Decoder (Text-to-Speech)

### Diffusion Models
- Image/Audio Generation

## Pipeline Flow
```
Voice Input → Whisper Encoder → LLM Processing → Response Generation → TTS Output
Vision Input → CLIP Encoder → Multi-modal Understanding → Action
```

## APU Hybrid Execution Strategy
- **CPU**: Encoder models, Preprocessing
- **GPU/NPU**: Decoder models, Diffusion models
- Optimized for Kraken2 and broader market

## Use Cases
- In-car voice commands
- Natural language interaction
- Multi-modal vehicle control
- Driver assistance

## Description
Heterogeneous hybrid execution system for in-vehicle voice assistance, leveraging APU capabilities for efficient encoder-decoder-diffusion model inference.

## Inference Runtime
- vLLM for LLM
- ONNX-RT for encoders
- Lemonade for orchestration

## Hardware Support
- APU Hybrid (CPU+GPU+NPU)
- Kraken2 platform
- RyzenAI Stack

---
*Source: In Vehicle Voice Assist. Usage to Inference Mapping – Hybrid, APU Hybrid Execution for Encoder-Decoder-Diffusion Models*
