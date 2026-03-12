#!/usr/bin/env python3
"""
Visualize patches from four types of SWAN datasets
12 rows by 20 columns, 3 rows for each type, distinguished by different color borders
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
DATA_FILES = [
    ('./dataset/SWAN_real_poststack.npz', 'Real Poststack', '#FF6B6B'),    # Red
    ('./dataset/SWAN_real_prestack.npz', 'Real Prestack', '#4ECDC4'),      # Cyan
    ('./dataset/SWAN_syn_poststack.npz', 'Syn Poststack', '#45B7D1'),      # Blue
    ('./dataset/SWAN_syn_prestack.npz', 'Syn Prestack', '#96CEB4'),        # Green
]
OUTPUT_FILE = './dataset/DEMO/samples_4_types.png'

N_ROWS = 12
N_COLS = 20
ROWS_PER_TYPE = 3
PATCHES_PER_TYPE = ROWS_PER_TYPE * N_COLS  # 60 per type

print("=" * 80)
print("SWAN Four Datasets Visualization")
print("=" * 80)

def select_low_zero_patches(data, n_samples, seed=42):
    """Select patches with the lowest ratio of zero values"""
    zero_ratio = data['zero_ratio']
    # Sort by zero ratio, select the smallest
    sorted_indices = np.argsort(zero_ratio)
    
    # Randomly select from the top 10% (to avoid always selecting the same ones)
    top_n = min(len(sorted_indices), max(n_samples * 5, 100000))
    top_indices = sorted_indices[:top_n]
    
    np.random.seed(seed)
    selected = np.random.choice(top_indices, min(n_samples, len(top_indices)), replace=False)
    return selected

# Load all data and select patches
all_patches = []
all_colors = []

for file_path, name, color in DATA_FILES:
    print(f"\nLoading: {file_path}")
    data = np.load(file_path)
    print(f"  Total patches: {len(data['patches']):,}")
    
    # Select patches with fewer zero values
    indices = select_low_zero_patches(data, PATCHES_PER_TYPE)
    patches = data['patches'][indices]
    zero_ratios = data['zero_ratio'][indices]
    
    print(f"  Selected {len(patches)} patches")
    print(f"  Zero ratio: {zero_ratios.min():.3f} - {zero_ratios.max():.3f}")
    
    all_patches.append(patches)
    all_colors.extend([color] * len(patches))

# Create a large figure
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(40, 24))
fig.patch.set_facecolor('white')

# Draw patches
patch_idx = 0
for type_idx, (file_path, name, color) in enumerate(DATA_FILES):
    start_row = type_idx * ROWS_PER_TYPE
    patches = all_patches[type_idx]
    
    for local_idx in range(min(PATCHES_PER_TYPE, len(patches))):
        row = start_row + local_idx // N_COLS
        col = local_idx % N_COLS
        ax = axes[row, col]
        
        # Display patch
        ax.imshow(patches[local_idx], cmap='seismic', aspect='auto',
                  vmin=-1, vmax=1, interpolation='bilinear')
        
        # Set colored borders
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

# Process empty positions
for row_idx in range(N_ROWS):
    for col_idx in range(N_COLS):
        type_idx = row_idx // ROWS_PER_TYPE
        local_row = row_idx % ROWS_PER_TYPE
        local_idx = local_row * N_COLS + col_idx
        if local_idx >= len(all_patches[type_idx]):
            axes[row_idx, col_idx].axis('off')

plt.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04,
                    wspace=0.02, hspace=0.02)

# Add color legend (Top right corner)
# legend_x = 0.99
# legend_y = 0.98
# for i, (_, name, color) in enumerate(DATA_FILES):
#     fig.text(legend_x, legend_y - i*0.025, f'█ {name}', 
#              color=color, fontsize=14, fontweight='bold',
#              ha='right', va='top')

# Save
print(f"\nSaving: {OUTPUT_FILE}")
plt.savefig(OUTPUT_FILE, dpi=64, bbox_inches='tight', facecolor='white')
print(f"  File size: {Path(OUTPUT_FILE).stat().st_size / 1024 / 1024:.1f} MB")
print("\n✓ Done!")
print("=" * 80)
plt.close()
