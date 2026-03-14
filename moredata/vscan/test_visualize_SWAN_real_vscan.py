#!/usr/bin/env python3
"""
Visualize the SWAN velocity-analysis dataset: SWAN_real_vscan.npz

Dataset keys
------------
data : shape (nsample, nt, nh)
    CMP gathers
vel  : shape (nsample, nt, nv)
    Velocity spectra
vnmo : shape (nsample, nt)
    Picked NMO velocity curve
t    : shape (nt,)
    Time axis
x    : shape (nsample,) or similar
    CMP/location axis
h    : shape (nh,)
    Offset axis
v    : shape (nv,)
    Velocity axis

Typical dimensions for this dataset
-----------------------------------
nsample = 250
nt = 1000
nh = 48
nv = 51

Usage
-----
1) Put this script in the same folder as SWAN_real_vscan.npz
2) Run:
       python visualize_swan_vscan.py

You can also modify the USER SETTINGS section below.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================
NPZ_FILE = "SWAN_real_vscan.npz"

# Example CMP indices to visualize individually
EXAMPLE_IDXS = [0, 10, 50, 100, 150, 200]

# Number of examples in montage plots
N_MONTAGE = 6

# Colormaps
CMP_CMAP = "gray"
VEL_CMAP = "jet"

# Percentile clipping for cleaner display
CMP_CLIP_PERCENTILE = 99.0
VEL_CLIP_PERCENTILE = 99.0

# Save figures?
SAVE_FIGURES = False
FIGURE_DIR = "figures_swan_vscan"

# Show figures interactively?
SHOW_FIGURES = True


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_dataset(npz_file):
    data = np.load(npz_file)

    required_keys = ["data", "vel", "vnmo", "t", "x", "h", "v"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {npz_file}")

    gathers = data["data"]   # (nsample, nt, nh)
    velspec = data["vel"]    # (nsample, nt, nv)
    vnmo = data["vnmo"]      # (nsample, nt)
    t = data["t"]            # (nt,)
    x = data["x"]            # (nsample,) or similar
    h = data["h"]            # (nh,)
    v = data["v"]            # (nv,)

    return gathers, velspec, vnmo, t, x, h, v


def print_dataset_summary(gathers, velspec, vnmo, t, x, h, v):
    print("=" * 70)
    print("SWAN velocity-analysis dataset summary")
    print("=" * 70)
    print(f"data   shape: {gathers.shape}")
    print(f"vel    shape: {velspec.shape}")
    print(f"vnmo   shape: {vnmo.shape}")
    print(f"t      shape: {t.shape}, range = [{t.min():.4f}, {t.max():.4f}]")
    print(f"x      shape: {x.shape}, range = [{x.min():.4f}, {x.max():.4f}]")
    print(f"h      shape: {h.shape}, range = [{h.min():.4f}, {h.max():.4f}]")
    print(f"v      shape: {v.shape}, range = [{v.min():.4f}, {v.max():.4f}]")
    print("=" * 70)


def robust_clip(arr, percentile=99.0, symmetric=True):
    if symmetric:
        amax = np.percentile(np.abs(arr), percentile)
        return -amax, amax
    else:
        amin = np.percentile(arr, 100 - percentile)
        amax = np.percentile(arr, percentile)
        return amin, amax


def savefig_maybe(fig, filename):
    if SAVE_FIGURES:
        ensure_dir(FIGURE_DIR)
        fig.savefig(os.path.join(FIGURE_DIR, filename), dpi=200, bbox_inches="tight")


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================
def plot_cmp_gather(gathers, t, h, x, idx):
    """
    Plot one CMP gather: time vs offset.
    """
    fig, ax = plt.subplots(figsize=(6, 8))

    g = gathers[idx]
    vmin, vmax = robust_clip(g, percentile=CMP_CLIP_PERCENTILE, symmetric=True)

    im = ax.imshow(
        g,
        aspect="auto",
        cmap=CMP_CMAP,
        extent=[h[0], h[-1], t[-1], t[0]],
        vmin=vmin,
        vmax=vmax,
    )

    cmp_label = x[idx] if len(x) > idx else idx
    ax.set_title(f"CMP Gather (index={idx}, x={cmp_label:.3f})")
    ax.set_xlabel("Offset (km)")
    ax.set_ylabel("Time (s)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Amplitude")

    fig.tight_layout()
    savefig_maybe(fig, f"cmp_gather_{idx:03d}.png")
    return fig, ax


def plot_velocity_spectrum(velspec, t, v, x, idx):
    """
    Plot one velocity spectrum: time vs velocity.
    """
    fig, ax = plt.subplots(figsize=(7, 8))

    s = velspec[idx]
    vmin, vmax = robust_clip(s, percentile=VEL_CLIP_PERCENTILE, symmetric=False)

    im = ax.imshow(
        s,
        aspect="auto",
        cmap=VEL_CMAP,
        extent=[v[0], v[-1], t[-1], t[0]],
        vmin=vmin,
        vmax=vmax,
    )

    cmp_label = x[idx] if len(x) > idx else idx
    ax.set_title(f"Velocity Spectrum (index={idx}, x={cmp_label:.3f})")
    ax.set_xlabel("Velocity (km/s)")
    ax.set_ylabel("Time (s)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Spectrum amplitude")

    fig.tight_layout()
    savefig_maybe(fig, f"velspec_{idx:03d}.png")
    return fig, ax


def plot_velocity_spectrum_with_vnmo(velspec, vnmo, t, v, x, idx):
    """
    Plot one velocity spectrum with picked vnmo overlaid.
    """
    fig, ax = plt.subplots(figsize=(7, 8))

    s = velspec[idx]
    pick = vnmo[idx]
    vmin, vmax = robust_clip(s, percentile=VEL_CLIP_PERCENTILE, symmetric=False)

    im = ax.imshow(
        s,
        aspect="auto",
        cmap=VEL_CMAP,
        extent=[v[0], v[-1], t[-1], t[0]],
        vmin=vmin,
        vmax=vmax,
    )

    ax.plot(pick, t, "w-", lw=2.0, label="Picked $v_{\\mathrm{NMO}}$")
    ax.plot(pick, t, "k--", lw=1.0, alpha=0.8)

    cmp_label = x[idx] if len(x) > idx else idx
    ax.set_title(f"Velocity Spectrum + Picked VNMO (index={idx}, x={cmp_label:.3f})")
    ax.set_xlabel("Velocity (km/s)")
    ax.set_ylabel("Time (s)")
    ax.legend(loc="upper right")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Spectrum amplitude")

    fig.tight_layout()
    savefig_maybe(fig, f"velspec_vnmo_{idx:03d}.png")
    return fig, ax


def plot_qc_panel(gathers, velspec, vnmo, t, h, v, x, idx):
    """
    QC panel:
      1) CMP gather
      2) velocity spectrum
      3) velocity spectrum + vnmo
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 7))

    g = gathers[idx]
    s = velspec[idx]
    pick = vnmo[idx]

    gmin, gmax = robust_clip(g, percentile=CMP_CLIP_PERCENTILE, symmetric=True)
    smin, smax = robust_clip(s, percentile=VEL_CLIP_PERCENTILE, symmetric=False)

    cmp_label = x[idx] if len(x) > idx else idx

    im0 = axes[0].imshow(
        g,
        aspect="auto",
        cmap=CMP_CMAP,
        extent=[h[0], h[-1], t[-1], t[0]],
        vmin=gmin,
        vmax=gmax,
    )
    axes[0].set_title(f"CMP Gather\nindex={idx}, x={cmp_label:.3f}")
    axes[0].set_xlabel("Offset (km)")
    axes[0].set_ylabel("Time (s)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        s,
        aspect="auto",
        cmap=VEL_CMAP,
        extent=[v[0], v[-1], t[-1], t[0]],
        vmin=smin,
        vmax=smax,
    )
    axes[1].set_title("Velocity Spectrum")
    axes[1].set_xlabel("Velocity (km/s)")
    axes[1].set_ylabel("Time (s)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(
        s,
        aspect="auto",
        cmap=VEL_CMAP,
        extent=[v[0], v[-1], t[-1], t[0]],
        vmin=smin,
        vmax=smax,
    )
    axes[2].plot(pick, t, "w-", lw=2.0, label="$v_{\\mathrm{NMO}}$")
    axes[2].plot(pick, t, "k--", lw=1.0, alpha=0.8)
    axes[2].set_title("Velocity Spectrum + Pick")
    axes[2].set_xlabel("Velocity (km/s)")
    axes[2].set_ylabel("Time (s)")
    axes[2].legend(loc="upper right")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    savefig_maybe(fig, f"qc_panel_{idx:03d}.png")
    return fig, axes


def plot_vnmo_profile(vnmo, t, x):
    """
    Plot 2D NMO velocity profile over all CMPs: time vs CMP.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        vnmo.T,
        aspect="auto",
        cmap="jet",
        extent=[x[0], x[-1], t[-1], t[0]],
    )

    ax.set_title("2D NMO Velocity Profile")
    ax.set_xlabel("CMP location (km)")
    ax.set_ylabel("Time (s)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("$v_{\\mathrm{NMO}}$")

    fig.tight_layout()
    savefig_maybe(fig, "vnmo_profile_2d.png")
    return fig, ax


def plot_multiple_cmp_gathers(gathers, t, h, x, idxs):
    """
    Plot several CMP gathers in a montage.
    """
    n = len(idxs)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, idx in zip(axes, idxs):
        g = gathers[idx]
        vmin, vmax = robust_clip(g, percentile=CMP_CLIP_PERCENTILE, symmetric=True)

        im = ax.imshow(
            g,
            aspect="auto",
            cmap=CMP_CMAP,
            extent=[h[0], h[-1], t[-1], t[0]],
            vmin=vmin,
            vmax=vmax,
        )
        cmp_label = x[idx] if len(x) > idx else idx
        ax.set_title(f"CMP {idx} (x={cmp_label:.3f})")
        ax.set_xlabel("Offset (km)")
        ax.set_ylabel("Time (s)")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Multiple CMP Gathers", fontsize=14)
    fig.tight_layout()
    savefig_maybe(fig, "cmp_gathers_montage.png")
    return fig, axes


def plot_multiple_velocity_spectra(velspec, vnmo, t, v, x, idxs):
    """
    Plot several velocity spectra with picked vnmo curves in a montage.
    """
    n = len(idxs)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, idx in zip(axes, idxs):
        s = velspec[idx]
        pick = vnmo[idx]
        vmin, vmax = robust_clip(s, percentile=VEL_CLIP_PERCENTILE, symmetric=False)

        im = ax.imshow(
            s,
            aspect="auto",
            cmap=VEL_CMAP,
            extent=[v[0], v[-1], t[-1], t[0]],
            vmin=vmin,
            vmax=vmax,
        )
        ax.plot(pick, t, "w-", lw=1.6)
        ax.plot(pick, t, "k--", lw=0.8, alpha=0.8)

        cmp_label = x[idx] if len(x) > idx else idx
        ax.set_title(f"CMP {idx} (x={cmp_label:.3f})")
        ax.set_xlabel("Velocity (km/s)")
        ax.set_ylabel("Time (s)")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Multiple Velocity Spectra with Picked VNMO", fontsize=14)
    fig.tight_layout()
    savefig_maybe(fig, "velspec_montage.png")
    return fig, axes


def plot_average_gather_and_average_spectrum(gathers, velspec, t, h, v):
    """
    Plot mean CMP gather and mean velocity spectrum across all samples.
    This is useful for dataset-level QC.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    avg_gather = np.mean(gathers, axis=0)
    avg_spec = np.mean(velspec, axis=0)

    gmin, gmax = robust_clip(avg_gather, percentile=CMP_CLIP_PERCENTILE, symmetric=True)
    smin, smax = robust_clip(avg_spec, percentile=VEL_CLIP_PERCENTILE, symmetric=False)

    im0 = axes[0].imshow(
        avg_gather,
        aspect="auto",
        cmap=CMP_CMAP,
        extent=[h[0], h[-1], t[-1], t[0]],
        vmin=gmin,
        vmax=gmax,
    )
    axes[0].set_title("Average CMP Gather")
    axes[0].set_xlabel("Offset (km)")
    axes[0].set_ylabel("Time (s)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        avg_spec,
        aspect="auto",
        cmap=VEL_CMAP,
        extent=[v[0], v[-1], t[-1], t[0]],
        vmin=smin,
        vmax=smax,
    )
    axes[1].set_title("Average Velocity Spectrum")
    axes[1].set_xlabel("Velocity (km/s)")
    axes[1].set_ylabel("Time (s)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    savefig_maybe(fig, "average_gather_and_spectrum.png")
    return fig, axes


# ============================================================
# MAIN
# ============================================================
def main():
    gathers, velspec, vnmo, t, x, h, v = load_dataset(NPZ_FILE)
    print_dataset_summary(gathers, velspec, vnmo, t, x, h, v)

    nsample = gathers.shape[0]

    # Make sure requested indices are valid
    example_idxs = [i for i in EXAMPLE_IDXS if 0 <= i < nsample]
    if len(example_idxs) == 0:
        example_idxs = [0]

    montage_idxs = np.linspace(0, nsample - 1, N_MONTAGE, dtype=int).tolist()

    # --------------------------------------------------------
    # Dataset-level plots
    # --------------------------------------------------------
    plot_vnmo_profile(vnmo, t, x)
    plot_average_gather_and_average_spectrum(gathers, velspec, t, h, v)
    plot_multiple_cmp_gathers(gathers, t, h, x, montage_idxs)
    plot_multiple_velocity_spectra(velspec, vnmo, t, v, x, montage_idxs)

    # --------------------------------------------------------
    # Individual example plots
    # --------------------------------------------------------
    for idx in example_idxs:
        plot_cmp_gather(gathers, t, h, x, idx)
        plot_velocity_spectrum(velspec, t, v, x, idx)
        plot_velocity_spectrum_with_vnmo(velspec, vnmo, t, v, x, idx)
        plot_qc_panel(gathers, velspec, vnmo, t, h, v, x, idx)

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
