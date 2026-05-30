# DACNet

Official implementation of **DACNet: Detail-Aware Context Modeling with Coherent Refinement for Camouflaged Object Detection**.

## Note

This code is directly related to our manuscript currently submitted to *Signal, Image and Video Processing*:

**Detail-Aware Context Modeling and Coherent Refinement for Camouflaged Object Detection**

Readers who use this repository are encouraged to cite the corresponding manuscript once it is published or publicly available.

## Requirements

The code is based on PyTorch. A typical environment is:

```bash
conda create -n dacnet python=3.10 -y
conda activate dacnet
```

```bash
pip install -r requirements.txt
```

### VMamba / SS2D Dependency

DACNet uses **SS2D** from the official VMamba project for state-space global context modeling. Please obtain the VMamba implementation from:

```text
https://github.com/MzeroMiko/VMamba
```

## Pretrained Weights

### PVTv2-B4 ImageNet Pretrained Weight

The PVTv2 encoder requires the ImageNet-pretrained PVTv2-B4 checkpoint. Please download `pvt_v2_b4.pth` from the official PVT project:

```text
https://github.com/whai362/PVT
```

## Dataset

We use the commonly adopted COD training and testing datasets, including CAMO, COD10K, and NC4K. Please download these datasets from their official sources or the links provided by the original benchmark papers.

This repository does not redistribute the datasets. After downloading, please organize the training data as:

```text
data/TrainDataset/
├── Image/
└── GT_Object/
```

## Training

```bash
python train.py
```

## Inference

Place the trained checkpoint as:

```text
checkpoint.pth
```

Then run:

```bash
python tests.py
```

## Evaluation

Evaluation scripts are provided under:

```text
evaltools/
```
## Model Weights

The trained DACNet model weights are available at [Baidu Netdisk](https://pan.baidu.com/s/1le6Ft1GQThNXzHZXK2gsPw?pwd=jwwf).
