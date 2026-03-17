# SWAN: Seismic Waveforms dataset for Automatic Neural-network processing

![Dataset Version](https://img.shields.io/badge/Dataset-v1.0-blue.svg)
![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)
![Python](https://img.shields.io/badge/Python-3.x-green.svg)

**SWAN** is a comprehensive and standardized benchmark designed to advance data-driven seismic signal processing. By aggregating diverse synthetic and real seismic waveforms spanning a wide range of geological structures, noise conditions, propagation environments, and acquisition geometries, SWAN provides a unified, AI-ready foundation for training highly generalizable models.

## 📖 Overview

Deep learning progress in seismic data processing is often constrained by a lack of large-scale, standardized datasets. SWAN addresses this bottleneck by providing:
- **Massive Scale**: 537,373 non-overlapping $128 \times 128$ wavefield patches.
- **Rich Diversity**: Extracted from 20 synthetic benchmark models and real field surveys across various global geological regions.
- **AI-Ready Format**: Consistently formatted, patch-level normalized within `[-1, 1]`, and saved in compressed `.npz` format for immediate integration into PyTorch/TensorFlow pipelines.
- **Comprehensive Metadata**: Includes source details, spatial positioning, original amplitudes, and quality indicators (e.g., zero-value ratios).

## 📊 Dataset Composition

The dataset is grouped into four major categories, spanning both prestack (shot gathers) and poststack (migrated sections) domains:

| Category | Patches | Percentage | Key Sources |
| :--- | :--- | :--- | :--- |
| **Synthetic Prestack** | 325,493 | ~60.6% | BP Models (1994, 2004, 2.5D, TTI), Marmousi, Pluto, Amoco |
| **Synthetic Poststack**| 74,523 | ~13.9% | SEAM Phase I (inline/xline slices) |
| **Real Prestack** | 6,969 | ~1.3% | USGS Alaska, Gulf of Mexico (Stratton3D, Oz Yilmaz) |
| **Real Poststack** | 130,388 | ~24.3% | Taranaki Basin (NZ), North Sea F3, Teapot Dome (US) |

*(For a detailed breakdown, please see [DATASET_SUMMARY.txt](DATASET_SUMMARY.txt))*

## 💾 Download

The SWAN dataset files are hosted on the UT box (https://utexas.box.com/s/cziybf0ktzvt5dt3okqrk0nnzqahcakd). Please download the `.npz` files into the `SWAN` folder or update the file paths in your scripts accordingly.

- `SWAN_syn_prestack.npz` (18 GB) — [Download Link](https://utexas.box.com/s/ntlnd31yghabsyc2umid68cxs9lp1oco) 
- `SWAN_syn_poststack.npz` (3.9 GB) — [Download Link](https://utexas.box.com/s/9xsl1zb4iee71rvjdj1thj3pwvb5mkjl) 
- `SWAN_real_prestack.npz` (372 MB) — [Download Link](https://utexas.box.com/s/ex48pyuhllp9fg32c4niinl0609denug) 
- `SWAN_real_poststack.npz` (6.9 GB) — [Download Link](https://utexas.box.com/s/9q61ndvav17ny1poc3cg5yhrjktcahf3) 

## 🚀 Getting Started

### 1. Repository Structure

```text
SWAN/
├── README.md               # This documentation
├── DATASET_SUMMARY.txt     # In-depth statistical breakdown
├── Main.pdf                # Accompanying research paper (Details on SWAN)
├── create_50k_dataset.py    # Script to sample a 50k dataset for training
├── dataset/                # Directory to store the downloaded .npz files
└── DEMO/                   # Visualization scripts and sample output images
    ├── visualize_samples.py
    └── samples_4_types.png
```

### 2. Loading the Data

SWAN uses the standard NumPy compressed format (`.npz`). You can easily load it using Python:

```python
import numpy as np

# Load a specific category
data = np.load('dataset/SWAN_syn_prestack.npz')

# Access the wavefield patches (Shape: N x 128 x 128)
patches = data['patches']

# Access metadata
dataset_names = data['dataset_name']
zero_ratios = data['zero_ratio']

print(f"Loaded {len(patches)} patches.")

# Example: Filter high-quality patches (less than 5% zero values)
mask = zero_ratios < 0.05
high_quality_patches = patches[mask]
```

### 3. Generating a 50k Training Subset
SWAN includes a script (`create_50k_dataset.py`) to generate a representative, 50,000-patch training dataset. The script randomly samples from all four categories with predefined ratios (40% syn_prestack, 20% syn_poststack, 10% real_prestack, 30% real_poststack):

```bash
python create_50k_dataset.py
```

This will produce a `50k_Train/` folder containing individual `.npy` patches, suitable for building data loaders in PyTorch or TensorFlow for your custom neural network or foundation model.

### 4. Visualization Examples
Explore the diversity of the dataset using the scripts in `DEMO/`. For example, `visualize_samples.py` will generate a visualization highlighting differences between the four types of data.

```bash
python DEMO/visualize_samples.py
```

Example visualization output:
![Samples 4 Types](DEMO/samples_4_types.png)

## 📌 Usage Guidelines & Reproducibility
- **Padding Details**: Some surveys (e.g., Marmousi, Alaska) retain padding traces. This information is available in the metadata key `zero_lines_in_left`.
- **Quality Control**: The `zero_ratio` metadata allows for thresholding out empty or non-informative patches based on your model's robustness.
- **Denormalization**: Patches are scaled to `[-1, 1]` for DL efficiency. Original true amplitudes can be restored using the corresponding `patch_max_value`.

## 📎 Citation

If you use the SWAN dataset in your research, please cite:

Gong, X., Fomel, S., and Chen, Y., 2026.  
Training a generalizable diffusion model for seismic data processing using a large-scale open-source waveform dataset.  
arXiv:2603.13645.  
https://arxiv.org/abs/2603.13645

BibTeX:

@article{gong2026swan,
  title={Training a generalizable diffusion model for seismic data processing using a large-scale open-source waveform dataset},
  author={Gong, Xinyue and Fomel, Sergey and Chen, Yangkang},
  journal={arXiv preprint arXiv:2603.13645},
  year={2026}
}

## Maintainer

Xinyue Gong

For any questions regarding the dataset or scripts, please open an issue in this repository.  
For dataset structure and statistics, please see `DATASET_SUMMARY.txt`.


