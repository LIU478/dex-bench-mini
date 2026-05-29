"""
MuJoCo Franka Panda PID 位置控制实验
基于 mujoco_menagerie 官方模型
控制方式：PD 控制器跟踪正弦目标轨迹
"""
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

MODEL_PATH = r"D:\work\github\dex-bench-mini\experiments\exp05_mujoco_test\mujoco_menagerie\franka_emika_panda\scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)

print("=" * 50)
print("Franka Panda PID 位置控制实验")
print("=" * 50)
print(f"关节数:   {model.njnt}")
print(f"执行器数: {model.nu}")
print(f"自由度:   {model.nv}")

# ── PID 控制器 ────────────────────────────────────────────────
class PIDController:
    """
    PID 控制器
    输出 = Kp * error + Ki * integral + Kd * derivative
    """
    def __init__(self, kp, ki, kd, dt):
        self.kp  = kp
        self.ki  = ki
        self.kd  = kd
        self.dt  = dt
        self.integral   = 0.0
        self.prev_error = 0.0

    def compute(self, target, current):
        error            = target - current
        self.integral   += error * self.dt
        derivative       = (error - self.prev_error) / self.dt
        self.prev_error  = error
        return (self.kp * error +
                self.ki * self.integral +
                self.kd * derivative)

    def reset(self):
        self.integral   = 0.0
        self.prev_error = 0.0

# ── 为 7 个臂关节各建一个 PID 控制器 ─────────────────────────
n_arm_joints = 7
dt = model.opt.timestep

# PD 参数（Ki=0 即纯 PD，更稳定）
Kp = 50.0
Ki = 0.1
Kd = 5.0

controllers = [PIDController(Kp, Ki, Kd, dt) for _ in range(n_arm_joints)]

# ── 目标轨迹：各关节不同幅度的正弦 ──────────────────────────
# 幅度设置：在关节限位范围内
amplitudes = [0.5, 0.3, 0.4, 0.3, 0.5, 0.3, 0.4]
freqs      = [0.5, 0.4, 0.6, 0.5, 0.7, 0.4, 0.6]

def get_target(t):
    return np.array([
        amplitudes[i] * np.sin(2*np.pi*freqs[i]*t)
        for i in range(n_arm_joints)
    ])

# ── 仿真 ──────────────────────────────────────────────────────
N = 2000  # 步数 = 4 秒

time_log       = np.zeros(N)
qpos_log       = np.zeros((N, n_arm_joints))
target_log     = np.zeros((N, n_arm_joints))
error_log      = np.zeros((N, n_arm_joints))
ctrl_log       = np.zeros((N, n_arm_joints))

print(f"\n开始 PID 仿真 {N} 步（{N*dt:.1f} 秒）...")

for step in range(N):
    t      = data.time
    target = get_target(t)

    # 对每个关节计算 PID 输出
    for i in range(n_arm_joints):
        ctrl = controllers[i].compute(target[i], data.qpos[i])
        # 限制控制输出幅度，防止过大
        data.ctrl[i] = np.clip(ctrl, -50.0, 50.0)

    mujoco.mj_step(model, data)

    # 记录数据
    time_log[step]   = t
    qpos_log[step]   = data.qpos[:n_arm_joints]
    target_log[step] = target
    error_log[step]  = target - data.qpos[:n_arm_joints]
    ctrl_log[step]   = data.ctrl[:n_arm_joints]

print("仿真完成！")
print(f"  最大跟踪误差: {np.abs(error_log).max():.4f} rad")
print(f"  稳态平均误差: {np.abs(error_log[N//2:]).mean():.4f} rad")
print(f"  最大控制输出: {np.abs(ctrl_log).max():.2f} N·m")

# ── 画图（6 张子图）──────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle(
    f'Franka Panda PID Position Control\n'
    f'Kp={Kp}, Ki={Ki}, Kd={Kd}  |  '
    f'Max error: {np.abs(error_log).max():.4f} rad',
    fontsize=13, fontweight='bold')

colors = plt.cm.tab10(np.linspace(0, 0.7, n_arm_joints))

# 图1：关节1 目标 vs 实际（单关节详细对比）
ax = axes[0, 0]
ax.plot(time_log, target_log[:, 0],
        color='red', linewidth=2, linestyle='--', label='Target q1')
ax.plot(time_log, qpos_log[:, 0],
        color='steelblue', linewidth=2, label='Actual q1')
ax.set_title('Joint 1: Target vs Actual (Detail)')
ax.set_ylabel('Angle (rad)')
ax.legend()
ax.grid(True, alpha=0.3)

# 图2：所有关节跟踪误差
ax = axes[0, 1]
for i in range(n_arm_joints):
    ax.plot(time_log, error_log[:, i],
            color=colors[i], linewidth=1.2, label=f'e{i+1}')
ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
ax.set_title('Tracking Error (all joints)')
ax.set_ylabel('Error (rad)')
ax.legend(fontsize=7, ncol=4)
ax.grid(True, alpha=0.3)

# 图3：所有关节实际位置
ax = axes[1, 0]
for i in range(n_arm_joints):
    ax.plot(time_log, qpos_log[:, i],
            color=colors[i], linewidth=1.5, label=f'q{i+1}')
ax.set_title('Actual Joint Positions')
ax.set_ylabel('Angle (rad)')
ax.legend(fontsize=7, ncol=4)
ax.grid(True, alpha=0.3)

# 图4：控制输出（力矩）
ax = axes[1, 1]
for i in range(n_arm_joints):
    ax.plot(time_log, ctrl_log[:, i],
            color=colors[i], linewidth=1.2, label=f'τ{i+1}')
ax.set_title('PID Control Output (Torque)')
ax.set_ylabel('Torque (N·m)')
ax.legend(fontsize=7, ncol=4)
ax.grid(True, alpha=0.3)

# 图5：误差的 RMS（均方根）随时间变化
ax = axes[2, 0]
window = 50
rms_error = np.array([
    np.sqrt(np.mean(error_log[max(0,i-window):i+1]**2))
    for i in range(N)
])
ax.plot(time_log, rms_error, color='crimson', linewidth=2)
ax.fill_between(time_log, rms_error, alpha=0.2, color='crimson')
ax.set_title('RMS Tracking Error (sliding window)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('RMS Error (rad)')
ax.grid(True, alpha=0.3)

# 图6：统计摘要
ax = axes[2, 1]
ax.axis('off')
stats = [
    ('Kp / Ki / Kd',         f'{Kp} / {Ki} / {Kd}'),
    ('N',               f'{N}'),
    ('Time',               f'{time_log[-1]:.1f} s'),
    ('time step',               f'{dt*1000:.1f} ms'),
    ('Maximum Tracking Error',           f'{np.abs(error_log).max():.4f} rad'),
    ('Steady-State Mean Error',           f'{np.abs(error_log[N//2:]).mean():.4f} rad'),
    ('Max control output',           f'{np.abs(ctrl_log).max():.2f} N·m'),
]
ax.text(0.05, 0.97, 'PID Experiment Summary',
        fontsize=11, fontweight='bold',
        transform=ax.transAxes, va='top')
for i, (k, v) in enumerate(stats):
    y = 0.85 - i * 0.12
    ax.text(0.05, y, f'{k}:', fontsize=9,
            transform=ax.transAxes, va='top')
    ax.text(0.65, y, v, fontsize=9, fontweight='bold',
            color='steelblue', transform=ax.transAxes, va='top')

plt.tight_layout()
plt.savefig('franka_pid_control.png', dpi=150, bbox_inches='tight')
print("\n图已保存: franka_pid_control.png")
print("✅ PID 控制实验完成！")

# ── Viewer 可视化 ─────────────────────────────────────────────
print("\n打开 3D 可视化窗口（关闭窗口退出）...")
mujoco.mj_resetData(model, data)
for c in controllers:
    c.reset()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        t      = data.time
        target = get_target(t)
        for i in range(n_arm_joints):
            ctrl = controllers[i].compute(target[i], data.qpos[i])
            data.ctrl[i] = np.clip(ctrl, -50.0, 50.0)
        mujoco.mj_step(model, data)
        viewer.sync()
        elapsed = time.time() - step_start
        time.sleep(max(0, dt - elapsed))