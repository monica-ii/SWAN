# SWAN_real_vscan Dataset

A real seismic velocity-analysis dataset for research on **seismic
imaging, velocity picking, and machine learning in exploration
geophysics**.

------------------------------------------------------------------------

## Overview

`SWAN_real_vscan.npz` is a structured dataset derived from **real marine
seismic data from the Gulf of Mexico**.\
The dataset provides:

-   **CMP gathers**
-   **Velocity spectra**
-   **Accurately picked NMO velocity curves**

It is designed to support research and development in:

-   seismic velocity analysis
-   automated NMO velocity picking
-   machine learning / deep learning for seismic processing
-   seismic imaging workflows
-   velocity-model building

The dataset contains **paired seismic observations and
expert-interpreted velocity picks**, making it particularly useful for
**supervised and self-supervised learning methods**.

------------------------------------------------------------------------

## Dataset Origin

The dataset originates from **marine seismic surveys in the Gulf of
Mexico**, one of the most geophysically complex and economically
important hydrocarbon basins in the world.

Characteristics of the region include:

-   complex sedimentary structures
-   strong velocity variations
-   layered stratigraphy
-   challenging imaging conditions

These characteristics make the dataset highly valuable for testing
**advanced seismic processing algorithms and machine learning models**.

------------------------------------------------------------------------

## Dataset Download

Download the dataset here: https://utexas.box.com/s/4g60d772kldhu9vhi6cfhpq3q9jribkk

------------------------------------------------------------------------

## File Format

The dataset is distributed as a **NumPy NPZ archive**:

    SWAN_real_vscan.npz

This format allows easy loading using Python.

Example:

``` python
import numpy as np

data = np.load("SWAN_real_vscan.npz")
print(data.keys())
```

------------------------------------------------------------------------

## Dataset Structure

The NPZ file contains the following arrays.

  Key      Shape                   Description
  -------- ----------------------- ----------------------------
  `data`   `(nsample, 1000, 48)`   CMP gathers
  `vel`    `(nsample, 1000, 51)`   Velocity spectra
  `vnmo`   `(nsample, 1000)`       Picked NMO velocity curves
  `t`      `(1000,)`               Time axis
  `x`      `(nsample,)`            CMP location axis
  `h`      `(48,)`                 Offset axis
  `v`      `(51,)`                 Velocity axis

Typical dataset dimensions:

    nsample = 250 CMP locations
    nt = 1000 time samples
    nh = 48 offsets
    nv = 51 velocity samples

------------------------------------------------------------------------

## Data Description

### 1. CMP Gathers (`data`)

CMP gathers represent the raw seismic data arranged by **common
midpoint**.

Dimensions:

    (nsamples, time, offset)

Typical size:

    250 × 1000 × 48

Each gather contains seismic traces recorded at different offsets.

These gathers are the primary input for **velocity analysis and NMO
correction**.

------------------------------------------------------------------------

### 2. Velocity Spectra (`vel`)

Velocity spectra are computed from CMP gathers to evaluate **velocity
semblance or coherence** across time and velocity.

Dimensions:

    (nsamples, time, velocity)

Typical size:

    250 × 1000 × 51

These spectra are commonly used by geophysicists to identify optimal
**stacking velocities**.

------------------------------------------------------------------------

### 3. NMO Velocity Picks (`vnmo`)

`vnmo` contains the **expert-picked Normal Moveout (NMO) velocities**
derived from the velocity spectra.

Dimensions:

    (nsamples, time)

Each curve represents the velocity function:

    v_NMO(t)

These curves are essential for:

-   NMO correction
-   stacking
-   migration velocity modeling

------------------------------------------------------------------------

### 4. Axis Information

  Variable   Description
  ---------- ------------------
  `t`        time axis
  `x`        CMP position
  `h`        offset
  `v`        velocity samples

These axes allow physical interpretation and correct plotting.

Example:

``` python
time = data['t']
offset = data['h']
velocity = data['v']
```

------------------------------------------------------------------------

## Example Visualization

Example Python code to visualize the dataset.

### CMP Gather

``` python
plt.imshow(data['data'][0], aspect='auto')
```

### Velocity Spectrum

``` python
plt.imshow(data['vel'][0], aspect='auto')
```

### Velocity Spectrum with Pick

``` python
plt.imshow(data['vel'][0], aspect='auto')
plt.plot(data['vnmo'][0], data['t'])
```

------------------------------------------------------------------------

## Potential Research Applications

The dataset supports research in several areas:

### Seismic Machine Learning

-   automatic NMO velocity picking
-   CNN/U-Net velocity spectrum interpretation
-   self-supervised velocity estimation
-   velocity spectrum denoising
-   representation learning in seismic data

### Seismic Imaging

-   velocity model building
-   NMO correction
-   stacking velocity estimation
-   migration velocity analysis
-   imaging quality assessment

### Seismic Processing Algorithms

-   semblance analysis
-   Radon transform velocity estimation
-   deep learning assisted velocity analysis
-   hybrid physics-guided ML methods

------------------------------------------------------------------------

## Why This Dataset Matters

Velocity analysis remains one of the **most critical and labor-intensive
steps in seismic imaging workflows**.

Accurate velocity models directly affect:

-   NMO correction
-   stacking quality
-   migration accuracy
-   final subsurface images

Traditional velocity picking relies heavily on **manual
interpretation**, which is time-consuming and subjective.

This dataset enables the development of **data-driven velocity analysis
algorithms**, including:

-   automated velocity picking
-   AI-assisted interpretation
-   physics-guided machine learning
-   real-time seismic processing

------------------------------------------------------------------------

## Impact for Seismic AI Research

The dataset provides a valuable benchmark for developing **machine
learning methods for seismic imaging**, including:

-   supervised learning with expert labels
-   weakly-supervised velocity estimation
-   self-supervised representation learning
-   physics-informed neural networks

It bridges the gap between:

    raw seismic data
            ↓
    velocity analysis
            ↓
    seismic imaging

making it useful for both **geophysical research and industrial
applications**.

------------------------------------------------------------------------

## Citation

If you use this dataset in research, please cite:

    SWAN_real_vscan dataset
    Gulf of Mexico seismic velocity analysis dataset


------------------------------------------------------------------------
