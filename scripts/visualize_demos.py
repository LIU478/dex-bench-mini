"""
Visualize demonstration dataset statistics.

Usage:
  python scripts/visualize_demos.py \
    --hdf5 experiments/exp09_demo_collection/outputs/demos.hdf5 \
    --out  experiments/exp09_demo_collection/outputs/demo_stats.png
"""
import argparse
import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--hdf5", type=str, required=True)
parser.add_argument("--out",  type=str,
    default="experiments/exp09_demo_collection/outputs/demo_stats.png")
args = parser.parse_args()

with h5py.File(args.hdf5, 'r') as f:
    num_demos  = f.attrs['num_demos']
    ep_lengths = f['stats/ep_lengths'][:]
    ep_returns = f['stats/ep_returns'][:]

    # 收集所有轨迹的 joint actions（前 16 维）
    all_actions = []
    all_rewards = []
    for i in range(num_demos):
        grp = f['data'][f'demo_{i}']
        all_actions.append(grp['actions'][:])
        all_rewards.append(grp['rewards'][:])

all_actions = np.concatenate(all_actions, axis=0)  # (total_steps, 16)
all_rewards = np.concatenate(all_rewards, axis=0)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(f'LEAP Hand Demo Dataset — {num_demos} episodes', fontsize=13)
fig.patch.set_facecolor('white')

COLOR = '#E8593C'

# 图1: episode 长度分布
axes[0,0].hist(ep_lengths, bins=30, color=COLOR, alpha=0.8, edgecolor='white')
axes[0,0].axvline(np.mean(ep_lengths), color='gray', lw=1.5, ls='--',
                  label=f'mean={np.mean(ep_lengths):.0f}')
axes[0,0].set_title('Episode length distribution')
axes[0,0].set_xlabel('Steps')
axes[0,0].set_ylabel('Count')
axes[0,0].legend(frameon=False)
axes[0,0].spines[['top','right']].set_visible(False)

# 图2: episode return 分布
axes[0,1].hist(ep_returns, bins=30, color='#3B8BD4', alpha=0.8, edgecolor='white')
axes[0,1].axvline(np.mean(ep_returns), color='gray', lw=1.5, ls='--',
                  label=f'mean={np.mean(ep_returns):.1f}')
axes[0,1].set_title('Episode return distribution')
axes[0,1].set_xlabel('Total reward')
axes[0,1].set_ylabel('Count')
axes[0,1].legend(frameon=False)
axes[0,1].spines[['top','right']].set_visible(False)

# 图3: 每步 reward 分布
axes[0,2].hist(all_rewards, bins=50, color='#2ecc71', alpha=0.8, edgecolor='white')
axes[0,2].set_title('Per-step reward distribution')
axes[0,2].set_xlabel('Reward')
axes[0,2].set_ylabel('Count')
axes[0,2].spines[['top','right']].set_visible(False)

# 图4: 各关节动作范围（箱线图）
joint_names = [f'j{i}' for i in range(16)]
axes[1,0].boxplot(all_actions, labels=joint_names,
                  patch_artist=True,
                  boxprops=dict(facecolor=COLOR, alpha=0.6),
                  medianprops=dict(color='white', lw=2),
                  whiskerprops=dict(color='gray'),
                  capprops=dict(color='gray'),
                  flierprops=dict(marker='.', alpha=0.2, ms=2))
axes[1,0].set_title('Action range per joint')
axes[1,0].set_xlabel('Joint index')
axes[1,0].set_ylabel('Action value')
axes[1,0].tick_params(axis='x', labelsize=8)
axes[1,0].spines[['top','right']].set_visible(False)

# 图5: 各关节动作均值
joint_means = all_actions.mean(axis=0)
joint_stds  = all_actions.std(axis=0)
x = np.arange(16)
axes[1,1].bar(x, joint_means, yerr=joint_stds, color=COLOR, alpha=0.7,
              capsize=3, error_kw={'elinewidth': 1})
axes[1,1].set_title('Mean action per joint (±std)')
axes[1,1].set_xlabel('Joint index')
axes[1,1].set_ylabel('Mean action')
axes[1,1].set_xticks(x)
axes[1,1].spines[['top','right']].set_visible(False)

# 图6: 数据集概览文字
axes[1,2].axis('off')
summary = (
    f"Dataset Summary\n\n"
    f"Total demos:      {num_demos}\n"
    f"Total steps:      {len(all_rewards):,}\n"
    f"Obs dim:          96\n"
    f"Action dim:       16\n\n"
    f"Ep length:\n"
    f"  mean = {np.mean(ep_lengths):.1f}\n"
    f"  std  = {np.std(ep_lengths):.1f}\n"
    f"  min  = {ep_lengths.min()}\n"
    f"  max  = {ep_lengths.max()}\n\n"
    f"Ep return:\n"
    f"  mean = {np.mean(ep_returns):.1f}\n"
    f"  std  = {np.std(ep_returns):.1f}"
)
axes[1,2].text(0.05, 0.95, summary, transform=axes[1,2].transAxes,
               fontsize=10, verticalalignment='top',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))

plt.tight_layout()
os.makedirs(os.path.dirname(args.out), exist_ok=True)
fig.savefig(args.out, dpi=150, bbox_inches='tight', facecolor='white')
print(f"[viz] Saved: {args.out}")
