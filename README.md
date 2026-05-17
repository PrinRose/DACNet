# DACNet

Official implementation of **DACNet: Detail-Aware Context Modeling with Coherent Refinement for Camouflaged Object Detection**.

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

## Training

Before training, make sure that:

1. The training dataset is placed under `data/TrainDataset/`.
2. The PVTv2-B4 pretrained weight is placed at `pretrained/pvt_v2_b4.pth`.
3. VMamba/SS2D can be imported correctly.

Then run:

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


Please update the BibTeX entry after the paper information is finalized.
