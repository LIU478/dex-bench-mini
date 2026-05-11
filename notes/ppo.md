# PPO (Proximal Policy Optimization)

---

## 🧠 一句话总结

PPO 是改进的 Policy Gradient，通过 clip 机制限制 policy 更新幅度，从而避免训练不稳定或崩溃。

---

## ❓ 解决了什么问题

- TRPO 计算复杂（需要二阶优化）
- PPO 用一阶优化替代 TRPO
- 通过 clipping 或 KL 约束控制更新幅度
- 提升稳定性 + 易实现性

---

## 📌 核心公式（必须掌握）

### 1️⃣ Probability Ratio

r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)

---

### 2️⃣ Clipped Surrogate Objective

L(θ) = E_t [ min( r_t(θ) A_t,
clip(r_t(θ), 1-ε, 1+ε) A_t ) ]

---

### 3️⃣ GAE Advantage

A_t = R_t - V(s_t)

---

## 🔁 算法流程（简化伪代码）

1. 用 π_θ 采样数据
2. 计算 reward-to-go
3. 计算 advantage
4. 更新 policy（PPO-Clip）
5. 更新 value function
6. 重复

---

## ⚙️ 关键超参数

- clip ratio ε = 0.1 ~ 0.2
- GAE λ = 0.95
- discount γ = 0.99

---

## 💡 我的理解

PPO 本质是在“提升好动作概率”的同时，通过 clip 限制更新幅度，让 policy 每次只做小幅度优化，从而保证训练稳定性。

---

## ❓ 我的疑问

- clip 为什么等价于 KL 约束？
- advantage 为什么可以这样定义？
- PPO 和 SAC 本质区别？