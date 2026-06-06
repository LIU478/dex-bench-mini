# exp08: LEAP Hand Cube Reorientation — Baseline

## 上游
- LEAP_Hand_Isaac_Lab commit: c576e6a
- IsaacLab commit: 3e73d6dd7

## 配置
| 参数 | 值 |
|------|-----|
| Task | Isaac-Reorient-Cube-Leap |
| Algorithm | PPO (rl_games) |
| num_envs | 1024 |
| max_iterations | 20000 |
| Hardware | RTX 4090D 24GB |
| 训练时长 | ~11.3 小时 |
| 计算速度 | ~20000 steps/s |

## 训练结果
| 指标 | 数值 |
|------|------|
| 最终 reward (last 500 iter avg) | 462.3 |
| 最高 reward | 61711 (iter ~1000) |
| consecutive_successes (final) | 0.121 |
| best checkpoint (自训练) | ep_18400, rew=567 |

## 关键观察
- iter 0-7500：ADR 课程学习阶段，reward 在万级，连续成功数峰值 16.5
- iter 7500：ADR 难度推高超出当前策略能力，reward 骤降，属正常现象
- iter 8000+：策略在更高难度配置下重新学习，reward 稳定在 400-550
- 20k iter 属于中期训练阶段，官方完整训练约需 50k-100k iter

## Checkpoint 说明
| 文件 | 说明 |
|------|------|
| checkpoint_official_pretrained.pth | 官方充分训练权重，用于演示最终效果 |
| checkpoint_best.pth | 自训练最佳 (ep_18400, rew=567) |
| checkpoint_final.pth | 自训练最终 (ep_20000, rew=502) |

## 与 Allegro (exp02) 对比
见 outputs/comparison_plots.png
注意：两者 reward 函数不同，数值不可直接比较
