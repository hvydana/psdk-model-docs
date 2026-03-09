# CogACT - Cognitive Action System

## Overview
Agentic robotics system for cognitive action planning and execution

## Category
Agentic/Robotics System

## Components
- Multi-Modal VLA Model
- OpenVLA
- Vision Encoder
- Language Model (Llama 2 7B)
- Action Decoder

## Pipeline Flow
```
Vision Input → Vision Encoder → Multi-Modal VLA → Language Understanding → Action Planning → Action Execution
```

## Use Cases
- Autonomous robotics
- Task planning and execution
- Vision-guided manipulation
- Cognitive decision making

## Description
CogACT is an agentic robotics system that combines vision, language, and action models to enable autonomous decision-making and task execution in robotics applications.

## Inference Runtime
- vLLM
- ONNX-RT for vision components

## Hardware Support
- APU (CPU+GPU) Hybrid Execution
- RyzenAI Stack

---
*Source: Agentic/Robotics (Ex. CogACT) Usage Mapping to Inference Pipeline*
