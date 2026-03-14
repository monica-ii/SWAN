# SWAN Random Resolution Dataset

A lightweight synthetic dataset designed for **seismic resolution
enhancement experiments**, **machine learning prototyping**, and
**educational demonstrations**.

This dataset contains **random reflectivity models** that allow users to
generate **custom low-resolution and high-resolution seismic data** by
convolving with wavelets of different frequency content.\
Unlike traditional supervised datasets that provide fixed input--output
pairs, this dataset provides **only the reflectivity**, enabling
flexible testing across a wide range of frequency scenarios.

------------------------------------------------------------------------

## Overview

The **SWAN Random Resolution dataset** is a synthetic dataset created
for **rapid experimentation with seismic temporal resolution
enhancement** methods.

Each sample in the dataset is a **2D reflectivity image**, which can be
converted into seismic data by convolution with a wavelet:

s(t,x) = r(t,x) \* w(t)

where

-   r(t,x) -- reflectivity model\
-   w(t) -- seismic wavelet\
-   s(t,x) -- simulated seismic data\
-   -   -- convolution operator

By using **different wavelets**, users can easily simulate:

-   **Low-resolution seismic data** (low dominant frequency)
-   **High-resolution seismic data** (high dominant frequency)

This allows researchers to construct **supervised training pairs** for
deep learning models such as:

Low-resolution seismic → High-resolution seismic

without being restricted to a single predefined frequency band.

------------------------------------------------------------------------

## Dataset Motivation

Many deep learning studies on seismic resolution enhancement rely on
datasets with **fixed wavelets**, which limits their applicability to
real-world acquisition conditions.

This dataset addresses that limitation by providing **reflectivity
only**, allowing users to generate training pairs with **arbitrary
wavelets**.

Key goals include:

-   Rapid experimentation with **resolution enhancement algorithms**
-   Flexible generation of **supervised training datasets**
-   Educational demonstrations of **wavelet-controlled seismic
    resolution**
-   Benchmarking ML/DL models for **temporal resolution recovery**

------------------------------------------------------------------------

## Dataset Structure

The dataset is distributed as a single NumPy archive:

SWAN_random_resolution.npz

### Keys

  Key       Description
  --------- ---------------------
  patches   Reflectivity images

### Data Shape

patches.shape = (N, 1024, 64)

where

-   **N** -- number of samples\
-   **1024** -- time samples\
-   **64** -- spatial traces

Each sample represents a **synthetic reflectivity gather**.

------------------------------------------------------------------------

## Download

Dataset download link: https://utexas.box.com/s/y4y3mhisjbuz1ssq18zavo03064dbnls


------------------------------------------------------------------------

## Example: Generating Low- and High-Resolution Data

Below is a simple example showing how to create training pairs.

``` python
import numpy as np
from scipy.signal import convolve

data = np.load("SWAN_random_resolution.npz")
R = data["patches"]

def ricker(f, dt, length=0.2):
    t = np.arange(-length/2, length/2, dt)
    w = (1 - 2*(np.pi*f*t)**2) * np.exp(-(np.pi*f*t)**2)
    return w

# Example frequencies
f_low = 15
f_high = 40

dt = 0.002

w_low = ricker(f_low, dt)
w_high = ricker(f_high, dt)

reflectivity = R[0]

low_res = convolve(reflectivity, w_low[:,None], mode="same")
high_res = convolve(reflectivity, w_high[:,None], mode="same")
```

This generates a **supervised training pair**:

low_res → high_res

------------------------------------------------------------------------

## Potential Applications

The dataset is suitable for a wide range of research topics, including:

### Machine Learning for Seismic Processing

-   Deep learning resolution enhancement
-   Super-resolution networks
-   U-Net based restoration
-   Transformer-based reconstruction

### Self-Supervised / Unsupervised Learning

-   Deep Image Prior
-   Noise2Noise style training
-   Physics-guided neural networks

### Signal Processing Research

-   Wavelet bandwidth effects
-   Temporal resolution analysis
-   Sparse reflectivity inversion

### Education and Teaching

-   Demonstrating the relationship between **reflectivity, wavelets, and
    resolution**
-   Teaching **seismic forward modeling**
-   Prototyping ML workflows for geophysics students

------------------------------------------------------------------------

## Why Reflectivity Instead of Seismic Data?

Providing only reflectivity has several advantages:

-   Avoids restricting experiments to a **fixed frequency band**
-   Allows **custom wavelet design**
-   Enables **fast generation of many training scenarios**
-   Supports both **supervised and self-supervised learning**

Users can therefore simulate many acquisition scenarios:

Reflectivity → Low-frequency wavelet → Low resolution seismic\
Reflectivity → High-frequency wavelet → High resolution seismic

------------------------------------------------------------------------

## Recommended Workflow

A typical supervised workflow for resolution enhancement is:

Reflectivity\
│\
▼\
Wavelet Convolution\
│\
├── Low-frequency wavelet → Low-resolution seismic\
└── High-frequency wavelet → High-resolution seismic\
│\
▼\
Deep Learning Model\
│\
▼\
Resolution Enhanced Seismic

------------------------------------------------------------------------

## Citation

The SWAN benchmark dataset
