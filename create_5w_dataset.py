"""
创建SWAN 5W训练数据集
按照 4:2:1:3 的比例从四个SWAN数据集中随机采样生成50000个训练样本

数据源比例:
- SWAN_syn_prestack.npz:   40% (20,000 patches)
- SWAN_syn_poststack.npz:  20% (10,000 patches)
- SWAN_real_prestack.npz:  10% (5,000 patches)
- SWAN_real_poststack.npz: 30% (15,000 patches)

输出格式: patch_XXXXX.npy (与CRDM1/data_train命名方式一致)
随机种子: 42 (确保可重复性)
"""

import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# 配置
RANDOM_SEED = 42
TOTAL_SAMPLES = 50000
OUTPUT_DIR = "/data/SWAN/5W_Train"

# 数据源和比例
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
    print("创建SWAN 5W训练数据集")
    print("="*70)

    # 设置随机种子
    np.random.seed(RANDOM_SEED)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n输出目录: {OUTPUT_DIR}")

    # 验证总数
    total_count = sum(src['count'] for src in DATA_SOURCES.values())
    assert total_count == TOTAL_SAMPLES, f"总样本数不匹配: {total_count} != {TOTAL_SAMPLES}"

    print(f"\n数据采样计划:")
    print(f"{'数据源':<25} {'比例':<8} {'样本数':<10} {'文件路径'}")
    print("-"*70)
    for name, info in DATA_SOURCES.items():
        print(f"{name:<25} {info['ratio']*100:>5.1f}%   {info['count']:>8}    {info['path']}")
    print(f"{'总计':<25} {'100.0%':<8} {TOTAL_SAMPLES:>8}")
    print("-"*70)

    # 处理每个数据源
    global_idx = 0

    for source_name, source_info in DATA_SOURCES.items():
        print(f"\n正在处理: {source_name}")
        print(f"  文件: {source_info['path']}")

        # 加载npz文件
        data = np.load(source_info['path'])
        patches_key = 'patches'

        if patches_key not in data:
            raise KeyError(f"'{patches_key}' not found in {source_info['path']}. Available keys: {list(data.keys())}")

        all_patches = data[patches_key]
        total_available = all_patches.shape[0]

        print(f"  可用patches: {total_available:,}")
        print(f"  需要采样: {source_info['count']:,}")

        if total_available < source_info['count']:
            raise ValueError(
                f"数据源 {source_name} 的样本数不足!\n"
                f"  需要: {source_info['count']:,}\n"
                f"  可用: {total_available:,}"
            )

        # 随机采样索引（不重复）
        sampled_indices = np.random.choice(
            total_available,
            size=source_info['count'],
            replace=False
        )
        sampled_indices = np.sort(sampled_indices)  # 排序以提高访问效率

        print(f"  采样索引范围: [{sampled_indices.min()}, {sampled_indices.max()}]")

        # 提取并保存patches
        print(f"  保存patches...")
        for local_idx, data_idx in enumerate(tqdm(sampled_indices, desc=f"  {source_name}")):
            patch = all_patches[data_idx]

            # 检查patch形状
            if patch.shape != (128, 128):
                print(f"\n  警告: patch {data_idx} 形状异常: {patch.shape}, 跳过")
                continue

            # 保存为npy文件，命名格式: patch_XXXXX.npy
            output_path = os.path.join(OUTPUT_DIR, f"patch_{global_idx:05d}.npy")
            np.save(output_path, patch)

            global_idx += 1

        data.close()
        print(f"  ✓ {source_name} 完成")

    # 验证输出
    print("\n" + "="*70)
    print("数据集创建完成!")
    print("="*70)

    saved_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npy')])
    print(f"\n生成文件数: {len(saved_files)}")
    print(f"目标数量: {TOTAL_SAMPLES}")

    if len(saved_files) == TOTAL_SAMPLES:
        print("✓ 文件数量正确")
    else:
        print(f"✗ 警告: 文件数量不匹配 ({len(saved_files)} != {TOTAL_SAMPLES})")

    # 显示统计信息
    print(f"\n文件命名范围: {saved_files[0]} 到 {saved_files[-1]}")

    # 计算总大小
    total_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in saved_files)
    print(f"总大小: {total_size / (1024**3):.2f} GB")

    # 抽样检查几个文件
    print("\n随机抽样检查:")
    check_indices = np.random.choice(len(saved_files), size=min(5, len(saved_files)), replace=False)
    for idx in sorted(check_indices):
        filepath = os.path.join(OUTPUT_DIR, saved_files[idx])
        patch = np.load(filepath)
        print(f"  {saved_files[idx]}: shape={patch.shape}, dtype={patch.dtype}, "
              f"min={patch.min():.3f}, max={patch.max():.3f}")

    print("\n" + "="*70)
    print("数据集可用于训练!")
    print(f"数据路径: {OUTPUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
