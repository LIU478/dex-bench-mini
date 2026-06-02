"""
加载已训练的 SAC+HER 模型，在 Push 任务上运行推理
第一次实验：成功率为80%
"""
import panda_mujoco_gym
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os

# ── 配置 ──────────────────────────────────────────────────────
ENV_ID     = "FrankaPushSparse-v0"
MODEL_PATH = "./models/final_model"  # 不需要加 .zip

# ── 加载模型 ──────────────────────────────────────────────────
print(f"加载模型: {MODEL_PATH}.zip")
# HER 模型加载必须传入环境
env = gym.make(ENV_ID, render_mode='rgb_array')
model = SAC.load(MODEL_PATH, env=env)
print("模型加载成功！")
print(f"  策略网络: {model.policy}")

# ── 创建评估环境 ───────────────────────────────────────────────
#env = gym.make(ENV_ID, render_mode='rgb_array')
obs, info = env.reset(seed=42)

print(f"\n环境信息:")
print(f"  观测空间: {env.observation_space}")
print(f"  动作空间: {env.action_space}")

# ── 运行推理，收集数据 ─────────────────────────────────────────
N_EPISODES = 10   # 运行 10 个 episode
N_STEPS    = 50   # 每个 episode 最多 50 步

all_rewards    = []
all_successes  = []
frames         = []

print(f"\n开始评估 {N_EPISODES} 个 episode...")

for ep in range(N_EPISODES):
    obs, info   = env.reset()
    ep_reward   = 0
    ep_success  = False
    ep_frames   = []

    for step in range(N_STEPS):
        # 用训练好的模型预测动作（deterministic=True 关掉探索噪声）
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward

        # 每 episode 的第 0、10、25、49 步截图
        if step in [0, 10, 25, 49]:
            frame = env.render()
            if frame is not None:
                ep_frames.append((ep, step, frame))

        if info.get('is_success', False):
            ep_success = True

        if terminated or truncated:
            break

    all_rewards.append(ep_reward)
    all_successes.append(ep_success)
    frames.extend(ep_frames[:2])  # 每个 episode 取前 2 帧

    status = "✅ 成功" if ep_success else "❌ 失败"
    print(f"  Episode {ep+1:2d}: reward={ep_reward:6.2f}  {status}")

env.close()

success_rate = sum(all_successes) / N_EPISODES * 100
print(f"\n评估结果:")
print(f"  成功率:     {success_rate:.1f}%  ({sum(all_successes)}/{N_EPISODES})")
print(f"  平均 reward: {np.mean(all_rewards):.3f}")
print(f"  最高 reward: {max(all_rewards):.3f}")

# ── 画图 ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle(f'SAC+HER Trained Model Evaluation\n'
             f'Task: {ENV_ID}  |  '
             f'Success Rate: {success_rate:.1f}%  |  '
             f'Avg Reward: {np.mean(all_rewards):.3f}',
             fontsize=13, fontweight='bold')

# 上半部分：仿真截图（最多 8 帧）
n_frames = min(8, len(frames))
for i in range(n_frames):
    ax = fig.add_subplot(3, 8, i + 1)
    ep, step, frame = frames[i]
    ax.imshow(frame)
    ax.set_title(f'Ep{ep+1} s{step}', fontsize=7)
    ax.axis('off')

# 左下：每个 episode 的 reward
ax = fig.add_subplot(3, 2, 3)
colors = ['green' if s else 'red' for s in all_successes]
bars = ax.bar(range(1, N_EPISODES+1), all_rewards, color=colors, alpha=0.8)
ax.set_title('Reward per Episode\n(green=success, red=fail)')
ax.set_xlabel('Episode')
ax.set_ylabel('Total Reward')
ax.axhline(y=np.mean(all_rewards), color='blue',
           linestyle='--', linewidth=2, label=f'Mean={np.mean(all_rewards):.2f}')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 右下：成功率饼图
ax = fig.add_subplot(3, 2, 4)
success_count = sum(all_successes)
fail_count    = N_EPISODES - success_count
ax.pie([success_count, fail_count],
       labels=[f'Success\n{success_count}', f'Fail\n{fail_count}'],
       colors=['#2ecc71', '#e74c3c'],
       autopct='%1.0f%%', startangle=90,
       textprops={'fontsize': 11})
ax.set_title(f'Success Rate: {success_rate:.1f}%')

# 左下下：reward 分布
ax = fig.add_subplot(3, 2, 5)
ax.hist(all_rewards, bins=10, color='steelblue', alpha=0.7, edgecolor='white')
ax.axvline(x=np.mean(all_rewards), color='red',
           linestyle='--', linewidth=2, label=f'Mean={np.mean(all_rewards):.2f}')
ax.set_title('Reward Distribution')
ax.set_xlabel('Reward')
ax.set_ylabel('Count')
ax.legend()
ax.grid(True, alpha=0.3)

# 右下下：统计摘要
ax = fig.add_subplot(3, 2, 6)
ax.axis('off')
stats = [
    ('model path',     MODEL_PATH + '.zip'),
    ('algorithm',        'SAC + HER'),
    ('task',        ENV_ID),
    ('assess episodes', str(N_EPISODES)),
    ('successeful',      f'{success_rate:.1f}%'),
    ('mean reward', f'{np.mean(all_rewards):.4f}'),
    ('MAX reward', f'{max(all_rewards):.4f}'),
    ('MIN reward', f'{min(all_rewards):.4f}'),
]
ax.text(0.05, 0.97, 'Evaluation Summary',
        fontsize=11, fontweight='bold',
        transform=ax.transAxes, va='top')
for i, (k, v) in enumerate(stats):
    y = 0.85 - i * 0.11
    ax.text(0.05, y, f'{k}:', fontsize=9,
            transform=ax.transAxes, va='top')
    ax.text(0.55, y, v, fontsize=9, fontweight='bold',
            color='steelblue', transform=ax.transAxes, va='top')

plt.tight_layout()
plt.savefig('eval_result.png', dpi=150, bbox_inches='tight')
print(f"\n图已保存: eval_result.png")
print("✅ 评估完成！")
# ── 实时 Viewer 回放 ──────────────────────────────────────────
import time
print("\n打开实时仿真窗口（关闭窗口退出）...")

viewer_env = gym.make(ENV_ID, render_mode='human')
obs, info  = viewer_env.reset()

for ep in range(20):
    obs, info  = viewer_env.reset()
    ep_reward  = 0
    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = viewer_env.step(action)
        ep_reward += reward
        time.sleep(0.02)
        if done or truncated:
            break
    status = "SUCCESS" if info.get('is_success') else "FAIL"
    print(f"Viewer Episode {ep+1}: {status}, reward={ep_reward:.2f}")

viewer_env.close()