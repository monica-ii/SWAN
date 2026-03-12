"""
Author: Xinyue Gong
Create SWAN 50k training dataset
Randomly sample 50,000 training samples from four SWAN datasets according to the ratio of 4:2:1:3

Data source ratio:
- SWAN_syn_prestack.npz:   40% (20,000 patches)
- SWAN_syn_poststack.npz:  20% (10,000 patches)
- SWAN_real_prestack.npz:  10% (5,000 patches)
- SWAN_real_poststack.npz: 30% (15,000 patches)

Output format: patch_XXXXX.npy (consistent with CRDM1/data_train naming convention)
Random seed: 42 (to ensure reproducibility)
"""

import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# Configuration
RANDOM_SEED = 42
TOTAL_SAMPLES = 50000
OUTPUT_DIR = "/data/SWAN/5W_Train"

# Data sources and ratios
DATA_SOURCES = {
    'syn_prestack': {
        'path': '/data/SWAN/SWAN_syn_prestack.npz',
        'ratio': 0.4,  # 40%
        'count': 20000
    },
    'syn_poststack': {
        'path': '/data/SWAN/SWAN_syn_poststack.npz',
        'ratio': 0.2,  # 20%
        'count': 10000
    },
    'real_prestack': {
        'path': '/data/SWAN/SWAN_real_prestack.npz',
        'ratio': 0.1,  # 10%
        'count': 5000
    },
    'real_poststack': {
        'path': '/data/SWAN/SWAN_real_poststack.npz',
        'ratio': 0.3,  # 30%
        'count': 15000
    }
}

def main():
    print("="*70)
    print("Create SWAN 50k Training Dataset")
    print("="*70)

    # Set random seed
    np.random.seed(RANDOM_SEED)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Validate total count
    total_count = sum(src['count'] for src in DATA_SOURCES.values())
    assert total_count == TOTAL_SAMPLES, f"Total sample count mismatch: {total_count} != {TOTAL_SAMPLES}"

    print(f"\nData sampling plan:")
    print(f"{'Data Source':<25} {'Ratio':<8} {'Samples':<10} {'File Path'}")
    print("-"*70)
    for name, info in DATA_SOURCES.items():
        print(f"{name:<25} {info['ratio']*100:>5.1f}%   {info['count']:>8}    {info['path']}")
    print(f"{'Total':<25} {'100.0%':<8} {TOTAL_SAMPLES:>8}")
    print("-"*70)

    # Process each data source
    global_idx = 0

    for source_name, source_info in DATA_SOURCES.items():
        print(f"\nProcessing: {source_name}")
        print(f"  File: {source_info['path']}")

        # Load npz file
        data = np.load(source_info['path'])
        patches_key = 'patches'

        if patches_key not in data:
            raise KeyError(f"'{patches_key}' not found in {source_info['path']}. Available keys: {list(data.keys())}")

        all_patches = data[patches_key]
        total_available = all_patches.shape[0]

        print(f"  Available patches: {total_available:,}")
        print(f"  Need to sample: {source_info['count']:,}")

        if total_available < source_info['count']:
            raise ValueError(
                f"Insufficient samples in data source {source_name}!\n"
                f"  Needed: {source_info['count']:,}\n"
                f"  Available: {total_available:,}"
            )

        # Randomly sample indices (without replacement)
        sampled_indices = np.random.choice(
            total_available,
            size=source_info['count'],
            replace=False
        )
        sampled_indices = np.sort(sampled_indices)  # Sort to improve access efficiency

        print(f"  Sampled index range: [{sampled_indices.min()}, {sampled_indices.max()}]")

        # Extract and save patches
        print(f"  Saving patches...")
        for local_idx, data_idx in enumerate(tqdm(sampled_indices, desc=f"  {source_name}")):
            patch = all_patches[data_idx]

            # Check patch shape
            if patch.shape != (128, 128):
                print(f"\n  Warning: patch {data_idx} has abnormal shape: {patch.shape}, skipping")
                continue

            # Save as npy file, naming format: patch_XXXXX.npy
            output_path = os.path.join(OUTPUT_DIR, f"patch_{global_idx:05d}.npy")
            np.save(output_path, patch)

            global_idx += 1

        data.close()
        print(f"  ✓ {source_name} completed")

    # Verify output
    print("\n" + "="*70)
    print("Dataset creation completed!")
    print("="*70)

    saved_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npy')])
    print(f"\nNumber of generated files: {len(saved_files)}")
    print(f"Target count: {TOTAL_SAMPLES}")

    if len(saved_files) == TOTAL_SAMPLES:
        print("✓ Correct number of files")
    else:
        print(f"✗ Warning: File count mismatch ({len(saved_files)} != {TOTAL_SAMPLES})")

    # Display statistics
    print(f"\nFile naming range: {saved_files[0]} to {saved_files[-1]}")

    # Calculate total size
    total_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in saved_files)
    print(f"Total size: {total_size / (1024**3):.2f} GB")

    # Randomly check a few files
    print("\nRandom sampling check:")
    check_indices = np.random.choice(len(saved_files), size=min(5, len(saved_files)), replace=False)
    for idx in sorted(check_indices):
        filepath = os.path.join(OUTPUT_DIR, saved_files[idx])
        patch = np.load(filepath)
        print(f"  {saved_files[idx]}: shape={patch.shape}, dtype={patch.dtype}, "
              f"min={patch.min():.3f}, max={patch.max():.3f}")

    print("\n" + "="*70)
    print("Dataset is ready for training!")
    print(f"Data path: {OUTPUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
