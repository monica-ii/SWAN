# SWAN_real_arrival Dataset

### A Lightweight Earthquake Arrival-Picking Dataset for Deep Learning Education

## Overview

**SWAN_real_arrival** is a simplified seismic waveform dataset designed
for **educational and research purposes in deep learning--based seismic
arrival picking**.

The dataset contains **100,000 three-component seismic waveforms**
extracted from broadband stations of the **Texas Seismological Network
(TexNet)**. Each waveform includes annotated **P-wave and S-wave arrival
samples**, enabling straightforward training and benchmarking of machine
learning models.

This dataset is derived from the **TXED (Texas Earthquake Dataset for
AI)** and repackaged into a compact **NumPy `.npz` format** to make it
easier for students, researchers, and educators to experiment with deep
learning workflows.

The goal of this dataset is to provide a **clean, easy-to-use
benchmark** for:

-   Deep learning arrival picking
-   Seismic signal processing education
-   Machine learning demonstrations
-   Algorithm prototyping

------------------------------------------------------------------------

## Data Source

The dataset is extracted from:

**TXED: The Texas Earthquake Dataset for AI**

GitHub repository:\
https://github.com/aaspip/txed

The **SWAN_real_arrival** dataset is a **lightweight subset** designed
specifically for **arrival-picking experiments and tutorials**.

------------------------------------------------------------------------

## Dataset Characteristics

  Property            Value
  ------------------- ----------------------------------------------------
  Number of samples   **100,000**
  Waveform length     **6000 samples**
  Channels            **3 components (Z, N, E)**
  Sampling rate       **100 Hz**
  Time window         **60 seconds**
  Labels              **P arrival sample index, S arrival sample index**
  File format         **NumPy `.npz`**
  Data type           `float32`

Each waveform is recorded from **broadband seismic stations** of the
**Texas Seismological Network (TexNet)**.

------------------------------------------------------------------------

## Dataset Structure

The dataset is stored in a single file:

    SWAN_real_arrival.npz

### Keys inside the `.npz` file

  -----------------------------------------------------------------------
  Key                     Shape                   Description
  ----------------------- ----------------------- -----------------------
  `waveforms`             `(100000, 6000, 3)`     Three-component seismic
                                                  waveforms

  `P`                     `(100000,)`             P-wave arrival sample
                                                  index

  `S`                     `(100000,)`             S-wave arrival sample
                                                  index

  `ids`                   `(100000,)`             Event--station
                                                  identifier
  -----------------------------------------------------------------------

### Component Order

    waveforms[i,:,0] → Z component
    waveforms[i,:,1] → N component
    waveforms[i,:,2] → E component

------------------------------------------------------------------------

## Download

Dataset download link:

    https://utexas.box.com/s/4utpv3m8e3rwy7k83895o7aul3cub5co

------------------------------------------------------------------------

## Example: Loading the Dataset

``` python
import numpy as np

data = np.load("SWAN_real_arrival.npz", allow_pickle=True)

waveforms = data["waveforms"]
P = data["P"]
S = data["S"]
ids = data["ids"]

print(waveforms.shape)
print(P.shape, S.shape)
```

Expected output:

    (100000, 6000, 3)
    (100000,) (100000,)

------------------------------------------------------------------------

## Example: Visualizing Waveforms with Arrivals

``` python
import numpy as np
import matplotlib.pyplot as plt

data = np.load("SWAN_real_arrival.npz", allow_pickle=True)

waveforms = data["waveforms"]
P = data["P"]
S = data["S"]

i = 0
x = waveforms[i]
p = int(P[i])
s = int(S[i])

fig, ax = plt.subplots(3,1,sharex=True,figsize=(10,6))

labels = ["Z","N","E"]

for k in range(3):
    ax[k].plot(x[:,k],'k',linewidth=0.8)
    ax[k].axvline(p,color='r',label='P arrival')
    ax[k].axvline(s,color='b',label='S arrival')
    ax[k].set_ylabel(labels[k])
    ax[k].grid(alpha=0.3)

ax[0].legend()
ax[-1].set_xlabel("Sample")

plt.show()
```

------------------------------------------------------------------------

## Convert Sample Index to Time

Sampling rate = **100 Hz**

    time (seconds) = sample / 100

Example:

``` python
p_time = P[i] / 100
s_time = S[i] / 100

print(p_time, s_time)
```

------------------------------------------------------------------------

## Example: PyTorch Dataset

``` python
import torch
from torch.utils.data import Dataset
import numpy as np

class ArrivalDataset(Dataset):

    def __init__(self, npz_file):
        d = np.load(npz_file, allow_pickle=True)
        self.x = d["waveforms"]
        self.p = d["P"]
        self.s = d["S"]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        x = torch.tensor(self.x[idx].T).float()
        p = torch.tensor(self.p[idx])
        s = torch.tensor(self.s[idx])

        return x, p, s


dataset = ArrivalDataset("SWAN_real_arrival.npz")

print(len(dataset))
```

------------------------------------------------------------------------

## Potential Applications

This dataset is suitable for experimenting with:

### Deep Learning Arrival Picking

-   CNN phase picking
-   U-Net phase picking
-   Transformer-based pickers
-   PhaseNet-style models

### Signal Processing

-   waveform denoising
-   time-frequency analysis
-   feature extraction

### Seismic AI Education

-   supervised learning
-   label regression
-   classification tasks
-   uncertainty estimation

------------------------------------------------------------------------

## Citation

If you use this dataset, please cite the original TXED dataset:

    TXED: The Texas Earthquake Dataset for AI
    https://github.com/aaspip/txed

