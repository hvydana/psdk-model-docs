# Veo Follow-Cam

## Overview
Industrial application for intelligent camera tracking and following

## Category
Industrial Application System

## Components
- Playerball Detection Models
  - playerball_detect_hrglass
  - 3d_unet
  - 3d_unet_transf_encoder
- Follow-cam Model
- Pano-Proj (CV Algorithm)
- Follow-cam Stich (CV Shader Algorithm)

## Pipeline Flow
```
Camera Input → GStreamer → Player Detection → 3D Tracking → Follow-cam Control → Panoramic Stitching
```

## Use Cases
- Sports analytics
- Industrial monitoring
- Automated camera tracking
- Multi-camera systems

## Description
Industrial application pipeline for intelligent camera following and tracking, using multiple detection models and computer vision algorithms for real-time tracking and panoramic video stitching.

## Inference Runtime
- GStreamer + ONNX-RT
- Gst-ONNX-RT
- GstHIP for acceleration

## Hardware Support
- APU (CPU+GPU) execution
- RyzenAI Stack

---
*Source: Industrial Application Usage Pipeline (Ex. Veo Follow-Cam)*
