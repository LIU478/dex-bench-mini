"""
演示了"随机策略下三种任务的表现
panda_mujoco_gym 三种任务演示
Slide / Push / PickAndPlace
"""
import panda_mujoco_gym
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# ── 三种任务配置 ──────────────────────────────────────────────
TASKS = [
    {
        'name': 'Slide',
        'env_id': 'FrankaSlideDense-v0',
        'desc': '推动滑块到目标位置（低摩擦桌面）',
        'color': 'steelblue',
    },
    {
        'name': 'Push',
        'env_id': 'FrankaPushDense-v0',
        'desc': '推方块到目标位置（正常摩擦桌面）',
        'color': 'darkorange',
    },
    {
        'name': 'PickAndPlace',
        'env_id': 'FrankaPickAndPlaceDense-v0',
        'desc': '抓取方块放到目标位置（需要夹爪控制）',
        'color': 'tomato',
    },
]

# ── 收集每个任务的数据 ────────────────────────────────────────
results = {}

for task in TASKS:
    print(f"\n{'='*50}")
    print(f"任务: {task['name']}")
    print(f"描述: {task['desc']}")
    print(f"环境: {task['env_id']}")
    print('='*50)

    # rgb_array 模式：不弹窗，直接渲染到数组
    env = gym.make(task['env_id'], render_mode='rgb_array')
    obs, info = env.reset(seed=42)

    print(f"观测维度: {env.observation_space.shape}")
    print(f"动作维度: {env.action_space.shape}")
    print(f"动作范围: [{env.action_space.low.min():.1f}, {env.action_space.high.max():.1f}]")

    # 运行 200 步，收集数据
    rewards    = []
    successes  = []
    frames     = []
    N_STEPS    = 200

    for step in range(N_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(reward)
        successes.append(info.get('is_success', 0))

        # 每 20 步截一帧
        if step % 40 == 0:
            frame = env.render()
            if frame is not None:
                frames.append((step, frame))

        if terminated or truncated:
            obs, info = env.reset()

    env.close()

    results[task['name']] = {
        'rewards':   rewards,
        'successes': successes,
        'frames':    frames,
        'task':      task,
    }

    print(f"总 reward: {sum(rewards):.2f}")
    print(f"成功次数: {sum(successes)}")
    print(f"捕获帧数: {len(frames)}")

# ── 画图 ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Franka Panda MuJoCo - Three Tasks Demo\n'
             'Slide / Push / PickAndPlace (Random Policy)',
             fontsize=14, fontweight='bold')

task_names = ['Slide', 'Push', 'PickAndPlace']

for row, name in enumerate(task_names):
    data = results[name]
    color = data['task']['color']

    # 左列：仿真截图（4帧）
    frames = data['frames']
    for col, (step, frame) in enumerate(frames[:4]):
        ax = fig.add_subplot(3, 6, row*6 + col + 1)
        ax.imshow(frame)
        ax.set_title(f'{name}\nstep={step}', fontsize=8)
        ax.axis('off')

    # 右列：reward 曲线
    ax = fig.add_subplot(3, 6, row*6 + 5)
    ax.plot(data['rewards'], color=color, linewidth=1.5, alpha=0.8)
    ax.set_title(f'{name} Reward', fontsize=9)
    ax.set_xlabel('Step', fontsize=8)
    ax.set_ylabel('Reward', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 最右列：统计
    ax = fig.add_subplot(3, 6, row*6 + 6)
    ax.axis('off')
    stats = [
        ('Task',      name),
        ('Env ID',    data['task']['env_id'].replace('-v0','')),
        ('Obs dim',   str(len(data['rewards']))),
        ('Total reward', f"{sum(data['rewards']):.1f}"),
        ('Successes', str(int(sum(data['successes'])))),
        ('Avg reward',f"{np.mean(data['rewards']):.3f}"),
    ]
    for i, (k, v) in enumerate(stats):
        ax.text(0.05, 0.92 - i*0.15, f'{k}:', fontsize=8,
                transform=ax.transAxes, va='top')
        ax.text(0.55, 0.92 - i*0.15, v, fontsize=8,
                fontweight='bold', color=color,
                transform=ax.transAxes, va='top')

plt.tight_layout()
plt.savefig('franka_three_tasks.png', dpi=150, bbox_inches='tight')
print(f"\n图已保存: franka_three_tasks.png")
print("✅ 三种任务演示完成！")