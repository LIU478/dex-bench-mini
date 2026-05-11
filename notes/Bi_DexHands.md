# Bi-DexHands (PKU-MARL)

> PKU-MARL 开源的双手灵巧操作（Bimanual Dexterous Manipulation）强化学习 Benchmark  
> 核心方向：双手协作、多智能体强化学习、高自由度灵巧控制、GPU 并行 RL 训练

---

# 1. 项目简介

Bi-DexHands 是北京大学 PKU-MARL 团队提出的一个基于 Isaac Gym 的双手灵巧操作强化学习平台（Benchmark）。

其核心目标是研究：

- 双手协同操作（Bimanual Manipulation）
- 多智能体强化学习（Multi-Agent Reinforcement Learning）
- 高维连续控制（High-DOF Continuous Control）
- 长时序操作任务（Long-Horizon Tasks）
- GPU 并行强化学习训练
- 灵巧手 Sim-to-Real 泛化

与传统单机械臂 manipulation 不同，Bi-DexHands 更强调：

```text
两个高自由度灵巧手之间的协同控制与复杂操作

例如：

- 双手交接物体
- 双手接抛物体
- 双手开门
- 双手开瓶盖
- 双手工具操作

这些任务涉及：

- contact-rich manipulation（复杂接触操作）
- force coordination（力协调）
- timing synchronization（时序同步）
- long-horizon planning（长时序规划）

因此难度远高于普通 RL benchmark。
```

---

# 2. 项目核心特点

## 2.1 基于 Isaac Gym（不是 Isaac Lab）

Bi-DexHands 构建于：

```text
Isaac Gym
```

注意：

```text
不是 Isaac Lab
```

Isaac Gym 是 NVIDIA 提供的：

```text
GPU-based physics simulation framework
```

核心特点：

- Physics Simulation 在 GPU
- RL Training 在 GPU
- Tensor API 在 GPU

因此：

```text
能够实现超大规模并行强化学习训练
```

例如：

- 1024 environments
- 4096 environments
- 8192 environments

同时并行训练。

这是 Isaac Gym 在 RL 领域极其重要的原因。

---

## 2.2 双手灵巧操作（Bimanual Dexterous Manipulation）

Bi-DexHands 最大卖点：

```text
双手协作 manipulation
```

传统 manipulation：

```text
单机械臂
```

而 Bi-DexHands：

```text
双灵巧手
```

因此：

- action space 更大
- state space 更复杂
- coordination 更困难
- reward design 更困难

属于典型：

```text
High-DOF Continuous Control Problem
```

---

## 2.3 GPU 并行强化学习

Isaac Gym 的核心革命：

```text
Physics + RL 全 GPU 化
```

传统 RL：

```text
CPU physics
+
GPU neural network
```

CPU 与 GPU 之间频繁通信：

- 很慢
- 吞吐量低

而 Isaac Gym：

```text
GPU physics
+
GPU RL
+
GPU tensor pipeline
```

因此：

- PPO 训练速度暴涨
- sample throughput 极高
- 更适合 manipulation RL

---

## 2.4 长时序任务（Long-Horizon Tasks）

很多任务：

```text
不是一步完成
```

而是：

```text
多阶段 skill chaining
```

例如：

```text
抓取 → 调整姿态 → 对准 → 插入
```

因此：

- exploration 更困难
- sparse reward 更严重
- policy switching 更重要

这类任务与：

- Sequential Dexterity
- Skill Chaining
- Hierarchical RL

密切相关。

---

# 3. 系统架构

## 3.1 Simulation Layer（仿真层）

底层仿真：

```text
Isaac Gym
```

负责：

- 刚体动力学
- 接触模拟
- collision
- articulation
- joint constraints

---

## 3.2 Robot Layer（机器人层）

机器人：

```text
Shadow Hand
```

通常：

- 双手
- 高自由度
- 多关节

Shadow Hand 常见：

```text
24 DoF
```

双手：

```text
48+ DoF
```

因此：

- 控制维度极高
- 训练难度巨大

---

## 3.3 RL Layer（强化学习层）

支持算法：

- PPO
- TRPO
- SAC
- TD3
- DDPG
- MAPPO
- HAPPO

等算法。

---

## 3.4 Multi-Agent Layer（多智能体层）

双手：

```text
Left Hand Agent
+
Right Hand Agent
```

因此天然适合：

```text
Multi-Agent Reinforcement Learning
```

研究重点：

- cooperation
- coordination
- communication
- credit assignment

---

# 4. 强化学习算法体系

---

# 4.1 Single-Agent Reinforcement Learning

## 支持算法

| 算法 | 全称 |
|---|---|
| PPO | Proximal Policy Optimization |
| TRPO | Trust Region Policy Optimization |
| DDPG | Deep Deterministic Policy Gradient |
| TD3 | Twin Delayed DDPG |
| SAC | Soft Actor-Critic |

---

## 4.1.1 TRPO

TRPO：

```text
Trust Region Policy Optimization
```

核心思想：

```text
每次 policy 更新不要变化太大
```

避免：

```text
policy collapse
```

TRPO 使用：

```math
D_{KL}(\pi_{\theta_{old}} || \pi_\theta)
```

限制新旧策略差异。

优点：

- 理论稳定
- monotonic improvement

缺点：

- 二阶优化
- Hessian approximation
- conjugate gradient
- 实现复杂

---

## 4.1.2 PPO（最核心）

PPO：

```text
Proximal Policy Optimization
```

可以理解为：

```text
TRPO 的工程简化版
```

核心思想：

```text
clip policy update
```

防止：

```text
policy 更新过大导致训练崩盘
```

---

### PPO 核心公式

## 1. Probability Ratio

```math
r_t(\theta)
=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{old}}(a_t|s_t)}
```

含义：

```text
新策略与旧策略在同一动作上的概率比值
```

---

## 2. Clipped Surrogate Objective

```math
L^{CLIP}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
r_t(\theta)\hat{A}_t,
\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
\right)
\right]
```

核心作用：

```text
限制 policy 更新幅度
```

其中：

```math
\epsilon \approx 0.2
```

---

## 3. GAE Advantage

```math
\hat{A}_t
=
\sum_{l=0}^{\infty}
(\gamma \lambda)^l \delta_{t+l}
```

其中：

```math
\delta_t
=
r_t
+
\gamma V(s_{t+1})
-
V(s_t)
```

GAE 的作用：

- 降低 variance
- 稳定训练
- 提高 advantage estimation 质量

---

## PPO 优点

- 简单
- 稳定
- sample efficient
- 易并行
- 工程实现容易

因此：

```text
PPO 几乎成为现代机器人 RL 默认算法
```

包括：

- Isaac Gym
- Isaac Lab
- Humanoid
- Dexterous Manipulation

大量项目都默认 PPO。

---

## 4.1.3 DDPG

DDPG：

```text
Deep Deterministic Policy Gradient
```

特点：

- Actor-Critic
- Off-policy
- Deterministic policy

适合：

- joint control
- torque control
- finger pose control

问题：

- 训练不稳定
- Q-value overestimation

---

## 4.1.4 TD3

TD3：

```text
Twin Delayed DDPG
```

是 DDPG 改进版。

主要解决：

- Q overestimation
- high variance

方法：

- twin Q network
- delayed actor update
- target policy smoothing

因此：

```text
比 DDPG 更稳定
```

---

## 4.1.5 SAC

SAC：

```text
Soft Actor-Critic
```

核心思想：

```text
Maximum Entropy RL
```

不仅最大化 reward：

```math
\mathbb{E}[R]
```

还最大化：

```math
\mathcal{H}(\pi)
```

即：

```text
鼓励 exploration
```

优点：

- exploration 强
- off-policy
- sample efficient

非常适合：

```text
复杂 manipulation
```

---

# 4.2 Multi-Agent Reinforcement Learning

## 支持算法

| 算法 | 全称 |
|---|---|
| IPPO | Independent PPO |
| MAPPO | Multi-Agent PPO |
| HAPPO | Heterogeneous-Agent PPO |
| HATRPO | Heterogeneous-Agent TRPO |
| MADDPG | Multi-Agent DDPG |

---

## 4.2.1 Multi-Agent RL 核心思想

双手系统：

```text
不是一个 agent
```

而是：

```text
多个 agent 协同
```

例如：

- 左手
- 右手

因此需要：

- coordination
- cooperation

---

## 4.2.2 CTDE

MARL 最重要思想：

```text
Centralized Training
Decentralized Execution
```

简称：

```text
CTDE
```

含义：

### 训练阶段

共享：

- 全局 observation
- 全局 reward
- joint state

### 执行阶段

每个 agent：

```text
独立决策
```

---

## 4.2.3 MAPPO

MAPPO：

```text
PPO 的多智能体版本
```

适用于：

- 双手机器人
- 多机器人
- coordination tasks

---

## 4.2.4 HAPPO

HAPPO：

```text
Heterogeneous-Agent PPO
```

重点：

```text
异构 agent
```

即：

- 不同 agent 能力不同
- 左右手结构不同
- 功能不同

---

# 5. Multi-Task Reinforcement Learning

| 算法 | 含义 |
|---|---|
| MTPPO | Multi-Task PPO |
| MTTRPO | Multi-Task TRPO |
| MTSAC | Multi-Task SAC |

核心思想：

传统：

```text
一个任务一个 policy
```

Multi-task RL：

```text
一个 policy 学多个任务
```

目标：

- 泛化能力
- policy reuse
- task transfer

---

# 6. Meta Reinforcement Learning

| 算法 | 含义 |
|---|---|
| ProMP | Meta RL 方法 |

Meta RL 核心思想：

```text
学习如何快速学习
```

目标：

- few-shot adaptation
- fast transfer
- task generalization

---

# 7. 任务列表

| 能力 | 对应任务 |
|---|---|
| 双手交接 | Over |
| 动态接物 | Catch |
| 双手搬运 | Lift |
| 开门 | Door |
| 精细旋转 | BottleCap |
| 工具操作 | Scissors |
| 精细拆装 | PenCap |
| Pick & Place | GraspAndPlace |

---

# 8. 任务分析

---

## 8.1 Over（双手交接）

目标：

```text
一个手将物体交给另一个手
```

核心难点：

- timing
- coordination
- stable grasp

---

## 8.2 Catch（动态接物）

目标：

```text
动态接住飞来的物体
```

难点：

- trajectory prediction
- dynamic manipulation
- fast response

---

## 8.3 Lift（双手搬运）

目标：

```text
双手共同搬运物体
```

难点：

- force balance
- synchronization
- object stabilization

---

## 8.4 Door（开门）

属于：

```text
contact-rich manipulation
```

需要：

- force control
- contact reasoning
- trajectory planning

---

## 8.5 BottleCap（开瓶盖）

经典 dexterous manipulation。

需要：

- 一只手固定
- 一只手旋转

涉及：

- torque control
- fine-grained finger manipulation

---

## 8.6 Scissors（剪刀）

属于：

```text
tool-use manipulation
```

需要：

- 双手协同
- precise coordination

---

## 8.7 PenCap（开笔帽）

涉及：

- precision grasping
- insertion/removal

---

# 9. 核心挑战

---

## 9.1 High-DOF Control

双手：

```text
48+ DoF
```

导致：

- state space 极大
- action space 极大

---

## 9.2 Sparse Reward

很多任务：

```text
只有最终成功才有 reward
```

导致：

```text
exploration difficulty
```

---

## 9.3 Coordination

双手：

```text
必须同步
```

否则：

- object drop
- unstable contact
- task failure

---

## 9.4 Sim-to-Real

仿真：

```text
Isaac Gym
```

真实：

```text
Real Robot
```

存在：

```text
domain gap
```

包括：

- friction
- latency
- sensor noise
- contact mismatch

---

# 10. 我的理解

Bi-DexHands 本质上是在解决：

```text
如何让两个高自由度灵巧手协同完成复杂 manipulation
```

这个问题。

它的重要性不仅仅在于：

```text
“做几个双手任务”
```

而是：

- 提供 benchmark
- 提供 RL baseline
- 推动 MARL manipulation
- 推动 dexterous RL
- 推动 GPU-scale RL

因此：

```text
Bi-DexHands 更像是：

双手 manipulation 领域的标准研究平台
```

---

# 11. 我的疑问

- 为什么 Isaac Gym 比 Mujoco 更适合 RL？
- PPO 为什么适合 dexterous manipulation？
- MAPPO 如何解决 credit assignment？
- 双手 observation 如何共享？
- sim-to-real 如何减少 domain gap？
- contact-rich manipulation 为什么困难？
- PPO 与 SAC 在 manipulation 中谁更强？

---