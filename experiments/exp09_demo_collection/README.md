# Experiment 09: LEAP Hand Demonstration Collection

## Overview

Collected 200 high-quality demonstration trajectories from the official pre-trained LEAP Hand policy for imitation learning and offline RL research.

## Dataset Details

### Statistics
- **Total demonstrations**: 200
- **Total transitions**: 56,729
- **Observation dimension**: 96 (hand state + cube state)
- **Action dimension**: 16 (joint commands)
- **Average episode length**: 283.6 ± 55.4 steps
- **Average episode return**: 14,646.1 ± 2,870.3
- **File size**: 23.7 MB (HDF5 with gzip compression)

### Data Quality
- **Success rate**: ~180/200 episodes completed full 300 steps
- **Failure cases**: ~20 episodes with early termination (cube dropped) — intentionally included for diversity
- **Data integrity**: All trajectories verified free of NaN/Inf values
- **Action clipping**: All actions properly normalized to [-1, 1]

### Collection Settings
- **Policy**: Official pre-trained checkpoint (`checkpoint_official_pretrained.pth`)
- **Num environments**: 16 (parallel rollouts)
- **Max steps per episode**: 300
- **Physics step size**: 0.008333 s
- **Rendering step size**: 0.033333 s

## File Structure
exp09_demo_collection/
├── outputs/
│   ├── demos.hdf5              # Main dataset (200 episodes)
│   └── demo_stats.png          # Statistical visualization
├── logs/
│   └── collect_200.log         # Collection script stdout/stderr
└── README.md                   # This file

## HDF5 Format

The `demos.hdf5` file is organized as follows:
demos.hdf5
├── Attributes
│   ├── num_demos: 200
│   ├── obs_dim: 96
│   ├── act_dim: 16
│   ├── task: "Isaac-Reorient-Cube-Leap"
│   ├── checkpoint: path to official pretrained weights
│   └── created_at: "2026-06-06 14:01:02"
├── data/
│   ├── demo_0/
│   │   ├── obs (300, 96) float32
│   │   ├── actions (300, 16) float32
│   │   ├── rewards (300,) float32
│   │   ├── dones (300,) bool
│   │   └── Attributes: ep_len, ep_return
│   ├── demo_1/
│   │   └── ...
│   └── ...
└── stats/
├── ep_lengths (200,) — length of each episode
├── ep_returns (200,) — total return of each episode
└── Attributes: mean_ep_len, mean_ep_return, std_ep_return

## Usage Examples

### Load and inspect dataset

```python
import h5py
import numpy as np

with h5py.File("demos.hdf5", "r") as f:
    # Get dataset info
    num_demos = f.attrs["num_demos"]
    
    # Load a single demo
    demo_0 = f["data/demo_0"]
    obs = demo_0["obs"][:]        # (300, 96)
    actions = demo_0["actions"][:]# (300, 16)
    rewards = demo_0["rewards"][:]# (300,)
    
    # Get statistics
    ep_lengths = f["stats/ep_lengths"][:]
    ep_returns = f["stats/ep_returns"][:]
```

### Verify data integrity

```bash
~/autodl-tmp/dex-bench-mini/dex-env/bin/python \
  ~/autodl-tmp/dex-bench-mini/scripts/replay_traj.py \
  --hdf5 demos.hdf5 --num_replay 10
```

### Generate statistics visualization

```bash
~/autodl-tmp/dex-bench-mini/dex-env/bin/python \
  ~/autodl-tmp/dex-bench-mini/scripts/visualize_demos.py \
  --hdf5 demos.hdf5 \
  --out demo_stats_new.png
```

## Key Observations

1. **High success rate**: ~90% of trajectories are full-length (300 steps), indicating strong policy performance
2. **Action diversity**: Joint actions span the full [-1, 1] range, with different joints using different action distributions
3. **Stable rewards**: Return distribution is tight and high (~14,646 mean), showing consistent behavior cloning quality
4. **Natural failures**: Short episodes (~20) capture realistic failure modes when cube is dropped
5. **Good for imitation learning**: High returns + stable behavior make this ideal for BC/DAgger approaches

## Collection Method

Trajectories collected using `collect_demos.py`:
- Deterministic policy rollouts (no exploration noise)
- 16 parallel environments for efficiency (~450 fps)
- Official pre-trained weights ensure high-quality demonstrations
- Episode reset on termination or 300-step limit
- RNN state properly managed for consistent rollouts
- All data compressed with gzip to minimize storage

## Next Steps

This dataset can be used for:
- **Behavior cloning**: Direct supervised learning from demonstrations
- **DAgger**: Interactive learning by expert querying
- **Offline RL**: CQL, IQL, or other algorithms that learn from fixed data
- **Imitation learning**: GAIL, ValueDICE, and other RL-from-demos methods
- **Pretraining**: Initialize RL agents with BC pretraining for faster convergence

## Metadata

- **Task**: Reorienting a cube using LEAP Hand dexterous manipulator
- **Environment**: Isaac Lab with PhysX GPU acceleration
- **Simulator**: Isaac Sim 4.5
- **RL Algorithm**: PPO (official pre-trained policy)
- **Collection Date**: 2026-06-06
- **Collection Time**: ~8 minutes for 200 episodes

---

For technical details on the collection script, see `../../scripts/collect_demos.py`.
