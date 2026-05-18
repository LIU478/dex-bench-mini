# dex-bench-mini

基于 NVIDIA Isaac Lab v2.1 的灵巧手 RL 训练 Benchmark

[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.5-green)](https://developer.nvidia.com/isaac-sim)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-v2.1.0-blue)](https://github.com/isaac-sim/IsaacLab)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red)](https://pytorch.org)

## 实验结果

### Experiment 01: Cartpole Reward Function Comparison

![Cartpole Comparison](experiments/exp01_cartpole_reward_comparison/comparison.png)

通过修改 `pole_pos` 惩罚权重（-1.0 → -3.0），验证 reward shaping 对
PPO 训练收敛的影响。**结论：在已可解任务中，weight 调整主要影响
reward 数值尺度，不影响最终 policy 质量。**

| 配置 | 最终 Reward | Episode Length |
|------|------------|----------------|
| Baseline (w=-1.0) | **4.96** | 300.00（满分）|
| Modified (w=-3.0) | 4.89 | 299.02 |

---

### Experiment 02 & 03: Allegro Hand Cube Reorientation

![Allegro Analysis](experiments/exp03_allegro_analysis/allegro_analysis.png)

在 RTX 4090D 上训练 Allegro Hand（16-DoF）完成 cube reorientation 任务。

| 指标 | 数值 |
|------|------|
| 最终 Mean Reward | **12.31** |
| 计算速度 | **25,058 steps/s** |
| 训练时长 | 33 分 58 秒 |
| track_orientation | 0.7969 |

---

## 实验列表

| 实验 | 内容 | 关键结果 |
|------|------|---------|
| [Exp01](experiments/exp01_cartpole_reward_comparison/) | Cartpole reward 对比实验 | 4.96 vs 4.89 |
| [Exp02](experiments/exp02_allegro_repose/) | Allegro Hand 2000 iter 训练 | Mean Reward 12.31 |
| [Exp03](experiments/exp03_allegro_analysis/) | Allegro 训练数据分析 | 多维度指标可视化 |

## 环境
Isaac Sim 4.5 + Isaac Lab v2.1.0
PyTorch 2.5.1 + CUDA 12.1
RTX 4090D 24GB (AutoDL)
Python 3.10

## 学习笔记

| 笔记 | 内容 |
|------|------|
| [Python 高级特性](notes/day04_python.md) | 函数式编程、生成器、OOP |
| [PyTorch 基础](notes/day04_pytorch.md) | Tensor、autograd、训练循环 |
| [Day 4 总结](notes/day04_summary.md) | 学习反思与简历连接点 |
