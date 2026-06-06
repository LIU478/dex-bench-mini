"""
Replay trajectories from HDF5 demo file and verify data integrity.

Usage:
  python scripts/replay_traj.py \
    --hdf5 experiments/exp09_demo_collection/outputs/demos.hdf5 \
    --num_replay 5
"""
import argparse
import os
import h5py
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--hdf5", type=str, required=True)
parser.add_argument("--num_replay", type=int, default=5)
args = parser.parse_args()

print(f"[replay] Loading: {args.hdf5}")
print(f"[replay] File size: {os.path.getsize(args.hdf5)/1e6:.1f} MB")

with h5py.File(args.hdf5, 'r') as f:
    # 全局属性
    print("\n=== Dataset Info ===")
    for k, v in f.attrs.items():
        print(f"  {k}: {v}")

    # 统计
    stats = f['stats']
    ep_lengths = stats['ep_lengths'][:]
    ep_returns = stats['ep_returns'][:]

    print(f"\n=== Statistics ({f.attrs['num_demos']} demos) ===")
    print(f"  ep_len:    mean={np.mean(ep_lengths):.1f}, "
          f"std={np.std(ep_lengths):.1f}, "
          f"min={ep_lengths.min()}, max={ep_lengths.max()}")
    print(f"  ep_return: mean={np.mean(ep_returns):.1f}, "
          f"std={np.std(ep_returns):.1f}, "
          f"min={ep_returns.min():.1f}, max={ep_returns.max():.1f}")

    # 回放几条轨迹
    print(f"\n=== Replaying {args.num_replay} demos ===")
    data = f['data']
    for i in range(min(args.num_replay, len(data))):
        grp = data[f'demo_{i}']
        obs     = grp['obs'][:]
        actions = grp['actions'][:]
        rewards = grp['rewards'][:]

        print(f"\n  demo_{i}:")
        print(f"    ep_len={len(obs)}, ep_return={rewards.sum():.2f}")
        print(f"    obs shape:     {obs.shape}, "
              f"range=[{obs.min():.3f}, {obs.max():.3f}]")
        print(f"    actions shape: {actions.shape}, "
              f"range=[{actions.min():.3f}, {actions.max():.3f}]")
        print(f"    rewards:       mean={rewards.mean():.3f}, "
              f"sum={rewards.sum():.2f}")

        # 检查 NaN/Inf
        has_nan = np.isnan(obs).any() or np.isnan(actions).any()
        has_inf = np.isinf(obs).any() or np.isinf(actions).any()
        print(f"    data quality:  nan={has_nan}, inf={has_inf}")

print("\n[replay] Verification complete.")
