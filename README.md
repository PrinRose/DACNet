# DACNet

Official implementation of **DACNet: Detail-Aware Context Modeling with Coherent Refinement for Camouflaged Object Detection**.

## News

- Code for training and inference has been released.
- Evaluation tools are provided under `evaltools/`.

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

You should make sure that the `SS2D` implementation can be imported by the DACNet code. One common way is to place or link the VMamba code in the repository so that the import path used by the model is valid, for example:

```text
DACNet/
├── VMamba/
│   └── vmamba.py
```

or adjust the import path in the model file according to your local project layout.

Please install the dependencies required by VMamba following the official VMamba instructions. The exact packages may depend on your CUDA, PyTorch, and compiler versions. If the selective scan CUDA extension fails to build, first check that your CUDA version, PyTorch version, and `nvcc` are compatible.

## Pretrained Weights

### PVTv2-B4 ImageNet Pretrained Weight

The PVTv2 encoder requires the ImageNet-pretrained PVTv2-B4 checkpoint. Please download `pvt_v2_b4.pth` from the official PVT project:

```text
https://github.com/whai362/PVT
```

The release page for PVTv2 ImageNet weights is:

```text
https://github.com/whai362/PVT/releases/tag/v2
```

After downloading, place the file as:

```text
DACNet/pretrained/pvt_v2_b4.pth
```

This path matches the default setting in `train.py`:

```python
"PVT_PRETRAINED_PATH": "./pretrained/pvt_v2_b4.pth"
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
