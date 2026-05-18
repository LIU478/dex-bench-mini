# Experiment 02: Allegro Hand Cube Reorientation

## Task
使用 Allegro Hand（16-DoF 四指手）完成 cube reorientation 任务：
将立方体从随机初始姿态重定向到目标姿态。

## Setup

| 配置 | 值 |
|------|----|
| Task | Isaac-Repose-Cube-Allegro-v0 |
| Robot | Allegro Hand（16 joints，4 fingers）|
| Algorithm | PPO（rsl_rl）|
| num_envs | 1024 |
| max_iterations | 2000 |
| Hardware | RTX 4090D 24GB |
| 训练时长 | 33 分 58 秒 |
| 计算速度 | 25,058 steps/s |

## Reward Components

| 组件 | 最终值 | 作用 |
|------|--------|------|
| track_orientation_inv_l2 | 0.7969 | 姿态跟踪（越大越好）|
| success_bonus | 0.0058 | 成功奖励 |
| action_rate_l2 | -0.2164 | 动作平滑性惩罚 |
| action_l2 | -0.0116 | 动作幅度惩罚 |
| joint_vel_l2 | -0.0029 | 关节速度惩罚 |

## Training Results

| 指标 | 数值 |
|------|------|
| 最终 Mean Reward | **12.31** |
| 最终 Episode Length | **533.10** |
| 姿态误差（orientation_error）| 1.4072 rad（~80°）|
| 连续成功率（consecutive_success）| 1.4% |
| object_out_of_reach | 20.8% |

## Analysis

**2000 iter 对于 Allegro Hand 任务属于早期训练阶段**：

| 任务 | 收敛所需 iter | 难度 |
|------|-------------|------|
| Cartpole（2-DoF）| ~100 | 低 |
| Allegro Repose（16-DoF）| 5000~20000 | 高 |

已观察到明确的学习信号：
- track_orientation_inv_l2 为正值，policy 在学习追踪目标姿态
- Mean Reward 从接近 0 上升到 12.31
- 训练过程无崩溃

## Checkpoint

训练好的 policy 存放在 `checkpoint/model_1999.pt`

## Key Takeaway

contact-rich 灵巧手任务需要比简单平衡任务多一个数量级的训练量，
reward 设计的好坏直接决定能否收敛。
