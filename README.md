# DACNet

Official implementation of **DACNet: Detail-Aware Context Modeling and Coherent Refinement for Camouflaged Object Detection**.

DACNet is a camouflaged object detection framework that combines coherent detail reconstruction and regulated global refinement for accurate structure-preserving segmentation.

## News

- Code and evaluation scripts are released.
- Pretrained models and prediction maps will be provided.

## Method Overview

DACNet contains two main components:

- **Coherent Detail Reconstruction Module (CDRM)**: preserves fine structures during top-down decoding with reconstruction-friendly upsampling and feature alignment.
- **Integrated Context Module (ICM)**: enhances global consistency with state space modeling while preserving weak boundary cues.
