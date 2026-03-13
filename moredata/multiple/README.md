
# SWAN Synthetic Prestack Dataset for Multiple Attenuation

High-quality **synthetic prestack seismic dataset** designed for **machine learning and deep learning research in seismic multiple attenuation and primary reconstruction**.

This dataset provides **paired inputs and ground-truth labels**, enabling supervised learning experiments with modern architectures such as **U-Net, CNNs, Transformers, and physics-guided neural networks**.

---

## Overview

The **SWAN Synthetic Prestack Multiple Dataset** simulates realistic seismic acquisition scenarios where **multiples contaminate primary reflections**.

The dataset contains:

- Raw prestack seismic gathers  
- Optimal multiple prediction models  
- Ground-truth primaries  

These components allow machine learning models to **learn how to separate primaries from multiples**.

---

## Dataset Structure

The dataset is distributed as a single NumPy archive:

```
SWAN_syn_prestack_multiple.npz
```

It contains two arrays:

```
data   : (2, 512, 256, 256)
label  : (1, 512, 256, 256)
```

### Inputs

```
data[0, i, :, :]  → Raw prestack data
data[1, i, :, :]  → Optimal multiple model
```

Each sample has a spatial size of:

```
256 × 256
```

representing a **seismic image (time × trace)**.

### Output

```
label[0, i, :, :] → Ground-truth primary
```

Thus the supervised learning problem is:

```
(raw data, multiple model) → primary reflections
```

or mathematically:

```
f(raw, multiple) ≈ primary
```

---

## Data Dimensions

| Component | Shape | Description |
|---|---|---|
| Inputs | 2 × 512 × 256 × 256 | Raw data + multiple model |
| Labels | 1 × 512 × 256 × 256 | Ground-truth primary |
| Samples | 512 | Total dataset size |
| Image size | 256 × 256 | Time × trace |

For deep learning frameworks such as **PyTorch**, the tensors are typically rearranged to:

```
(N, C, H, W)

Inputs  → (512, 2, 256, 256)
Labels  → (512, 1, 256, 256)
```

---

## Example Data Visualization

Example components in one sample:

| Raw Data | Multiple Model | Ground Truth Primary |
|---|---|---|
| seismic gather with multiples | predicted multiple energy | clean primary reflections |

Example images can be placed in a `docs/` directory:

```
docs/sample_raw.png
docs/sample_multiple.png
docs/sample_primary.png
```

---

## Download

Download the dataset from

https://utexas.box.com/s/1wopeu2sff519otbus13sriqkgvvf553

---

## Quick Loading Example

```python
import numpy as np

data = np.load("SWAN_syn_prestack_multiple.npz")

x = data["data"]
y = data["label"]

print(x.shape)
print(y.shape)
```

Output:

```
(2, 512, 256, 256)
(1, 512, 256, 256)
```

Reorder for PyTorch:

```python
x = x.transpose(1,0,2,3)
y = y.transpose(1,0,2,3)

print(x.shape)
print(y.shape)
```

Output:

```
(512, 2, 256, 256)
(512, 1, 256, 256)
```

---

## Research Applications

This dataset is suitable for studying:

### Seismic Multiple Attenuation
Learning to suppress surface-related or internal multiples.

### Primary Reconstruction
Recovering clean primaries from contaminated gathers.

### Physics-Guided Machine Learning
Combining deep learning with physical constraints or wave-equation models.

### Image-to-Image Translation
Training deep learning architectures such as:

- U-Net
- ResNet
- Swin Transformer
- Diffusion Models
- Physics-Informed Neural Networks

---

## Example Learning Task

Train a neural network to learn

```
Input  : Raw gather + multiple model
Output : Primary reflections
```

Example architecture:

```
2-channel input
     ↓
U-Net encoder-decoder
     ↓
1-channel predicted primary
```

---

## Citation

If you use this dataset in academic work, please cite:

```
SWAN Synthetic Prestack Dataset for Multiple Attenuation
```

---
