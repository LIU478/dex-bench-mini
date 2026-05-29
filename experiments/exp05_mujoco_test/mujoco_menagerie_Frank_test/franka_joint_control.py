"""
MuJoCo Franka Panda 关节控制实验
基于 mujoco_menagerie 官方模型
"""
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# 加载真实 Franka 模型（修改为你的实际路径）
MODEL_PATH = r"D:\work\github\dex-bench-mini\experiments\exp05_mujoco_test\mujoco_menagerie\franka_emika_panda\scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)

print("=" * 50)
print("Franka Panda 模型信息")
print("=" * 50)
print(f"关节数:   {model.njnt}")
print(f"执行器数: {model.nu}")
print(f"自由度:   {model.nv}")
print(f"几何体数: {model.ngeom}")

print("\n关节列表:")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"  joint[{i}]: {name}")

print("\n执行器列表:")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  actuator[{i}]: {name}")

# ── 仿真：正弦关节控制 ────────────────────────────────────
N = 1000
n_joints = 7

qpos_log = np.zeros((N, n_joints))
qvel_log = np.zeros((N, n_joints))
ctrl_log = np.zeros((N, model.nu))
time_log = np.zeros(N)

print(f"\n开始仿真 {N} 步...")

for step in range(N):
    t = step * model.opt.timestep
    # 每个关节施加不同相位的正弦控制
    for i in range(min(model.nu, n_joints)):
        data.ctrl[i] = 0.3 * np.sin(2*np.pi*0.5*t + i*np.pi/4)
    mujoco.mj_step(model, data)
    qpos_log[step] = data.qpos[:n_joints]
    qvel_log[step] = data.qvel[:n_joints]
    ctrl_log[step] = data.ctrl[:]
    time_log[step] = t

print("仿真完成！")
print(f"  关节角范围: [{qpos_log.min():.3f}, {qpos_log.max():.3f}] rad")
print(f"  关节速度范围: [{qvel_log.min():.3f}, {qvel_log.max():.3f}] rad/s")

# ── 画图 ──────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle('Franka Panda - Sinusoidal Joint Control\n(mujoco_menagerie official model)',
             fontsize=13, fontweight='bold')

colors = plt.cm.tab10(np.linspace(0, 0.7, n_joints))

# 图1：关节角
ax = axes[0]
for i in range(n_joints):
    ax.plot(time_log, qpos_log[:, i],
            color=colors[i], linewidth=1.5, label=f'q{i+1}')
ax.set_title('Joint Positions')
ax.set_ylabel('Angle (rad)')
ax.legend(fontsize=7, ncol=4, loc='upper right')
ax.grid(True, alpha=0.3)

# 图2：关节速度
ax = axes[1]
for i in range(n_joints):
    ax.plot(time_log, qvel_log[:, i],
            color=colors[i], linewidth=1.5, label=f'dq{i+1}')
ax.set_title('Joint Velocities')
ax.set_ylabel('Velocity (rad/s)')
ax.legend(fontsize=7, ncol=4, loc='upper right')
ax.grid(True, alpha=0.3)

# 图3：控制输入
ax = axes[2]
for i in range(min(model.nu, n_joints)):
    ax.plot(time_log, ctrl_log[:, i],
            color=colors[i], linewidth=1.5, label=f'ctrl{i+1}')
ax.set_title('Control Inputs (Sinusoidal)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Control Signal')
ax.legend(fontsize=7, ncol=4, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('franka_joint_control.png', dpi=150, bbox_inches='tight')
print("\n图已保存: franka_joint_control.png")

# ── Viewer 可视化 ─────────────────────────────────────────
print("\n打开 3D 可视化窗口（关闭窗口退出）...")
mujoco.mj_resetData(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        t = data.time
        for i in range(min(model.nu, n_joints)):
            data.ctrl[i] = 0.3 * np.sin(2*np.pi*0.5*t + i*np.pi/4)
        mujoco.mj_step(model, data)
        viewer.sync()
        elapsed = time.time() - step_start
        time.sleep(max(0, model.opt.timestep - elapsed))