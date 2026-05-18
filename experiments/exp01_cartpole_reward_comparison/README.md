# Experiment 01: Cartpole Reward Function Comparison

## Hypothesis

增大 `pole_pos` 的惩罚权重（-1.0 → -3.0），预期 policy 会更激进地优化
杆子位置，收敛路径可能不同，但最终性能相近（任务本身已可解）。

## Method

| 配置 | Baseline | Modified |
|------|----------|----------|
| pole_pos weight | -1.0 | **-3.0** |
| alive weight | 1.0 | 1.0 |
| terminating weight | -2.0 | -2.0 |
| num_envs | 64 | 64 |
| max_iterations | 500 | 500 |
| Algorithm | PPO (rsl_rl) | PPO (rsl_rl) |
| Hardware | RTX 4090D 24GB | RTX 4090D 24GB |

## Results

![Reward 对比](comparison.png)

| 指标 | Baseline | Modified |
|------|----------|----------|
| 最终 Mean Reward | **4.96** | **4.89** |
| 最终 Episode Length | **300.00** | **299.02** |
| cart_out_of_bounds | 0.0000 | 0.0000 |
| 计算速度 | ~3000 steps/s | ~3000 steps/s |
| 总训练时间 | ~2m 47s | ~2m 49s |

## Findings

1. **两者都完全收敛**：Episode Length 均接近最大值 300，
   `cart_out_of_bounds` 均为 0，说明两个 policy 都能稳定平衡

2. **Modified 最终 reward 略低（4.89 vs 4.96）**：
   原因是 pole_pos 的惩罚项绝对值更大（-3.0 × deviation），
   在同样的平衡质量下，惩罚更多，总 reward 数值更低

3. **任务本质没有区别**：两者的 alive=1.0 说明策略质量相同，
   差异只在 reward 数值的计算方式上

## Discussion

这个实验揭示了 reward shaping 的一个重要原则：

**reward weight 影响 policy 优化的「力度」，但不一定影响最终质量**。

- Baseline 已经达到最优（Episode Length = 300 = 最大值）
- Modified 在同样收敛的情况下，reward 数值因惩罚项更大而偏低
- 这说明：在任务已经可解的情况下，调整 weight 主要影响数值尺度，
  而非 policy 本身的能力

**对灵巧手任务的启示**：
在 contact-rich 任务（如 cube reorientation）中，reward weight 的设计
至关重要，因为任务本身不容易收敛，weight 的选择会直接决定 policy
是否能学到有效策略。这与 Cartpole 这种"容易收敛"的任务有本质区别。

## Key Takeaway

reward weight 调整 ≠ 性能提升。在简单任务中，weight 主要影响数值尺度；
在复杂任务中，weight 才决定收敛成败。

## Code

修改版任务代码在 `cartpole_modified/` 目录。  
核心改动：`cartpole_env_cfg.py` → `RewardsCfg.pole_pos.weight: -1.0 → -3.0`
