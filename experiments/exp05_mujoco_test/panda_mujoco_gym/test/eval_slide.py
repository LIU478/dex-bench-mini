"""
评估 Franka Slide 任务的 SAC+HER 模型
输出成功率、奖励统计，并生成评估图
"""

import panda_mujoco_gym
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np
import matplotlib
matplotlib.use('Agg')   # 非交互后端，避免 GUI 阻塞
import matplotlib.pyplot as plt
import os

# ========== 配置 ==========
ENV_ID = "FrankaSlideSparse-v0"           # Slide 任务
MODEL_PATH = "./models/Slide_50_model"   # 可修改为你的模型路径（不需要 .zip 后缀）
N_EPISODES = 20                           # 评估回合数
MAX_STEPS = 50                            # 每个回合最大步数
OUTPUT_IMG = "eval_slide_result.png"      # 输出图片文件名
# =========================

# 加载模型
print(f"加载模型: {MODEL_PATH}.zip")
env = gym.make(ENV_ID, render_mode='rgb_array')
model = SAC.load(MODEL_PATH, env=env)
print("模型加载成功！")

# 创建评估环境（用于收集数据，不渲染）
eval_env = gym.make(ENV_ID, render_mode='rgb_array')
obs, info = eval_env.reset()

print(f"\n环境信息:")
print(f"  观测空间: {eval_env.observation_space}")
print(f"  动作空间: {eval_env.action_space}")

# 运行评估
all_rewards = []
all_successes = []
frames = []   # 存储 (episode, step, frame)

print(f"\n开始评估 {N_EPISODES} 个 episode...")

for ep in range(N_EPISODES):
    obs, info = eval_env.reset()
    ep_reward = 0.0
    ep_success = False
    ep_frames = []

    for step in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        ep_reward += reward

        # 每 episode 的 step 0, 10, 25, 49 截图
        if step in [0, 10, 25, MAX_STEPS - 1]:
            frame = eval_env.render()
            if frame is not None:
                ep_frames.append((ep, step, frame))

        if info.get('is_success', False):
            ep_success = True

        if terminated or truncated:
            break

    all_rewards.append(ep_reward)
    all_successes.append(ep_success)
    frames.extend(ep_frames[:2])  # 每个 episode 最多存前 2 帧（避免图片太多）

    status = "✅ 成功" if ep_success else "❌ 失败"
    print(f"  Episode {ep+1:2d}: reward={ep_reward:6.2f}  {status}")

eval_env.close()
env.close()

success_rate = sum(all_successes) / N_EPISODES * 100
print(f"\n评估结果:")
print(f"  成功率:     {success_rate:.1f}%  ({sum(all_successes)}/{N_EPISODES})")
print(f"  平均 reward: {np.mean(all_rewards):.3f}")
print(f"  最高 reward: {max(all_rewards):.3f}")
print(f"  最低 reward: {min(all_rewards):.3f}")

# ========== 绘图 ==========
fig = plt.figure(figsize=(16, 10))
fig.suptitle(f'SAC+HER Model Evaluation on Slide Task\n'
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

# 左下：每个 episode 的 reward（绿色成功，红色失败）
ax = fig.add_subplot(3, 2, 3)
colors = ['green' if s else 'red' for s in all_successes]
bars = ax.bar(range(1, N_EPISODES+1), all_rewards, color=colors, alpha=0.8)
ax.set_title('Reward per Episode (green=success, red=fail)')
ax.set_xlabel('Episode')
ax.set_ylabel('Total Reward')
ax.axhline(y=np.mean(all_rewards), color='blue', linestyle='--', linewidth=2, label=f'Mean={np.mean(all_rewards):.2f}')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 右下：成功率饼图
ax = fig.add_subplot(3, 2, 4)
success_count = sum(all_successes)
fail_count = N_EPISODES - success_count
ax.pie([success_count, fail_count],
       labels=[f'Success\n{success_count}', f'Fail\n{fail_count}'],
       colors=['#2ecc71', '#e74c3c'],
       autopct='%1.0f%%', startangle=90,
       textprops={'fontsize': 11})
ax.set_title(f'Success Rate: {success_rate:.1f}%')

# 左下下：reward 分布直方图
ax = fig.add_subplot(3, 2, 5)
ax.hist(all_rewards, bins=10, color='steelblue', alpha=0.7, edgecolor='white')
ax.axvline(x=np.mean(all_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(all_rewards):.2f}')
ax.set_title('Reward Distribution')
ax.set_xlabel('Reward')
ax.set_ylabel('Count')
ax.legend()
ax.grid(True, alpha=0.3)

# 右下下：统计摘要表格
ax = fig.add_subplot(3, 2, 6)
ax.axis('off')
stats = [
    ('Task', 'FrankaSlideSparse-v0'),
    ('Model path', MODEL_PATH + '.zip'),
    ('Algorithm', 'SAC + HER'),
    ('Evaluation episodes', str(N_EPISODES)),
    ('Success rate', f'{success_rate:.1f}%'),
    ('Mean reward', f'{np.mean(all_rewards):.4f}'),
    ('Max reward', f'{max(all_rewards):.4f}'),
    ('Min reward', f'{min(all_rewards):.4f}'),
]
ax.text(0.05, 0.97, 'Evaluation Summary', fontsize=11, fontweight='bold', transform=ax.transAxes, va='top')
for i, (k, v) in enumerate(stats):
    y = 0.85 - i * 0.11
    ax.text(0.05, y, f'{k}:', fontsize=9, transform=ax.transAxes, va='top')
    ax.text(0.55, y, v, fontsize=9, fontweight='bold', color='steelblue', transform=ax.transAxes, va='top')

plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=150, bbox_inches='tight')
print(f"\n评估图已保存: {OUTPUT_IMG}")
print("✅ 评估完成！")