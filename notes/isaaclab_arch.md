# Isaac Lab 架构理解

> 一份关于 Isaac Lab 模块化机器人强化学习框架的学习笔记，包含核心概念、Workflow 对比，以及如何将其应用到 `dex-bench-mini` 项目。

---

## 目录

- [一句话总结](#一句话总结)
- [两种 Workflow](#两种-workflow)
  - [Manager-based 是什么](#manager-based-是什么)
  - [Direct 是什么](#direct-是什么)
- [Manager-based 核心 Cfg](#manager-based-核心-cfg)
- [Isaac Lab 的运行逻辑](#isaac-lab-的运行逻辑)
- [RL 环境本质](#rl-环境本质)
- [ManagerBasedEnv vs ManagerBasedRLEnv](#managerbasedenv-vs-managerbasedrlenv)
- [Isaac Lab 为什么强](#isaac-lab-为什么强)
- [dex-bench-mini 怎么用](#dex-bench-mini-怎么用)
- [后续扩展路线](#后续扩展路线)
- [我的理解](#我的理解)

---

## 一句话总结

Isaac Lab 本质上是一个 **模块化机器人强化学习框架**。

它把以下部分全部拆成独立模块：

- 场景（Scene）
- 动作（Action）
- 观测（Observation）
- 奖励（Reward）
- 随机化（Randomization）
- 终止条件（Termination）

因此使用 Isaac Lab：

> **不是“写一个巨大的 `env.py`”，而是“像搭积木一样构建 RL 环境”。**

---

## 两种 Workflow

| 特性         | Manager-based | Direct     |
| ------------ | ------------- | ---------- |
| 模块化       | 强            | 弱         |
| 上手难度     | 中            | 高         |
| 灵活性       | 中            | 强         |
| 开发效率     | 高            | 中         |
| 可维护性     | 强            | 弱         |
| 推荐场景     | 标准 RL 任务  | 特殊需求   |
| 推荐程度     | ⭐⭐⭐⭐⭐     | ⭐⭐⭐     |

### Manager-based 是什么

Manager-based 把 RL 环境拆成多个 Manager 管理，例如：

- `ObservationManager`
- `ActionManager`
- `RewardManager`
- `EventManager`
- `TerminationManager`

每个 Manager 负责一种功能。因此：

> **Reward 不再写死在 `env.step()` 里，而是独立配置。**

### Direct 是什么

Direct 模式下所有逻辑自己手写，包括：

- observation
- reward
- physics logic
- reset
- action apply

**优点：** 更灵活
**缺点：** 大型项目很难维护

---

## Manager-based 核心 Cfg

### SceneCfg

定义环境里有什么，例如：

- robot
- table
- object
- camera
- light

### ObservationsCfg

定义 agent 能看到什么，即 **observation space**。

常见 observation：

- joint position
- joint velocity
- object pose
- RGB image
- depth image

### ActionsCfg

定义 agent 能控制什么，即 **action space**。

常见 action：

- torque control
- joint position control
- joint velocity control

### RewardsCfg

定义什么是“好行为”，本质是 **reward terms 加权求和**。

常见 reward：

- distance reward
- grasp reward
- lift reward
- stability reward

### TerminationsCfg

定义 episode 什么时候结束，例如：

- time out
- object drop
- robot fall
- out of bounds

### EventCfg

定义特殊事件，例如：

- reset randomization
- domain randomization
- startup randomization

常见随机化项：

- friction
- mass
- texture
- light

主要用于 **sim-to-real**。

---

## Isaac Lab 的运行逻辑

环境运行流程：

```text
policy
  ↓
action
  ↓
ActionManager
  ↓
Physics Simulation
  ↓
ObservationManager
  ↓
RewardManager
  ↓
TerminationManager
  ↓
返回：obs, reward, done
```

---

## RL 环境本质

Isaac Lab 的 RL 环境本质上是一个 **MDP（Markov Decision Process）**。

对应关系：

| MDP        | Isaac Lab     |
| ---------- | ------------- |
| State      | Observation   |
| Action     | Action        |
| Reward     | Reward        |
| Transition | Physics       |
| Done       | Termination   |

---

## ManagerBasedEnv vs ManagerBasedRLEnv

### ManagerBasedEnv（普通环境）

包含：

- scene
- observation
- action
- event

适合：

- teleoperation
- robot control
- 数据采集

**不包含** reward 与 termination。

### ManagerBasedRLEnv（强化学习环境）

在 `ManagerBasedEnv` 的基础上额外增加：

- `RewardsCfg`
- `TerminationsCfg`
- `CurriculumCfg`
- `CommandsCfg`

因此可以直接用于 **PPO / SAC** 等算法训练。

---

## Isaac Lab 为什么强

### GPU Physics

| 传统方式                          | Isaac Lab                              |
| --------------------------------- | -------------------------------------- |
| CPU physics + GPU neural network  | GPU physics + GPU RL + GPU tensor pipe |

因此可以支持 **4096 / 8192 env 并行训练**。

### 模块化

- 研究 reward → 只改 `RewardsCfg`
- 研究 observation → 只改 `ObservationsCfg`

### Manipulation 友好

官方大量支持：

- manipulation
- humanoid
- locomotion
- dexterous hand

---

## dex-bench-mini 怎么用

### Workflow 选择

`dex-bench-mini` 适合 **Manager-based workflow**，原因：

- 更简单
- 更适合 manipulation
- 官方支持更多
- 更容易扩展

### 参考模板

最适合参考的官方任务：

> **`Isaac-Lift-Cube-Franka-v0`**

它已经实现：

- grasping
- object interaction
- reward design
- manipulation pipeline

### 修改方向

#### 1. 改 `SceneCfg`

将 **Franka Arm** 替换为 **LEAP Hand**，包括：

- URDF
- articulation
- joint config

#### 2. 改 `ObservationsCfg`

增加：

- finger joint states
- object pose
- contact state

后续还可以加入：

- RGB camera
- tactile sensor

#### 3. 改 `ActionsCfg`

控制 LEAP Hand joints，例如使用 **joint position target**。

#### 4. 改 `RewardsCfg`

核心是设计 manipulation reward，例如：

- distance reward
- contact reward
- grasp reward
- lift reward
- stability reward

---

## 后续扩展路线

### 单手 Manipulation

- grasp
- lift
- rotate
- reorient

### 双手 Manipulation

- handover
- bimanual lift
- tool use

### Vision-based RL

加入：

- RGB camera
- depth camera

实现 **vision-based manipulation**。

### Sim-to-Real

加入 domain randomization，实现 **simulation → real robot transfer**。

---

## 我的理解

我认为 Isaac Lab 最大的价值并不只是“能跑 RL”，而是：

> **提供了一套标准化的机器人 RL 开发框架。**

它把以下部分全部模块化：

- physics
- RL
- sensors
- rewards
- randomization
- robot assets

因此研究者不需要每次都从零开始写环境，而是 **像搭积木一样构建机器人任务**。