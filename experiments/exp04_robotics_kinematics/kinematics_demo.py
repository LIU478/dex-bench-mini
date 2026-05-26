"""
机器人运动学与动力学分析
基于 Peter Corke Robotics Toolbox for Python
- 运动学分析: Franka Panda (7-DoF)
- 动力学分析: Puma560 (6-DoF, 有完整动力学参数)
"""
import roboticstoolbox as rtb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spatialmath import SE3

# ============================================================
# Part 1: 正运动学 (Franka Panda)
# ============================================================
print("=" * 50)
print("Part 1: 正运动学 (FK) - Franka Panda")
print("=" * 50)

robot = rtb.models.Panda()
q_home = robot.qr
T_end = robot.fkine(q_home)
print(f"关节角 q_home: {np.round(q_home, 3)}")
print(f"末端位置 (x,y,z): {np.round(T_end.t, 4)}")
print(f"末端姿态 (RPY): {np.round(T_end.rpy(), 4)}")

# ============================================================
# Part 2: 逆运动学 (Franka Panda)
# ============================================================
print("\n" + "=" * 50)
print("Part 2: 逆运动学 (IK) - Franka Panda")
print("=" * 50)

T_target = SE3(0.5, 0.1, 0.3) * SE3.Rz(np.pi)
print(f"目标末端位置: {T_target.t}")
ik_sol = robot.ikine_LM(T_target, q0=q_home)
print(f"IK 求解成功: {ik_sol.success}")
T_verify = robot.fkine(ik_sol.q)
err = np.linalg.norm(T_verify.t - T_target.t)
print(f"验证位置误差: {err*1000:.3f} mm")

# ============================================================
# Part 3: 雅可比矩阵 (Franka Panda)
# ============================================================
print("\n" + "=" * 50)
print("Part 3: 雅可比矩阵 - Franka Panda")
print("=" * 50)

J = robot.jacobe(q_home)
print(f"雅可比矩阵形状: {J.shape}")
print(f"条件数: {np.linalg.cond(J):.2f}  (越小越远离奇异点)")

# ============================================================
# Part 4: 轨迹规划 (Franka Panda)
# ============================================================
print("\n" + "=" * 50)
print("Part 4: 轨迹规划 - Franka Panda")
print("=" * 50)

q_start = robot.qr
q_end   = np.array([0.5, -0.3, 0.2, -2.0, 0.1, 1.8, 0.5])
traj    = rtb.jtraj(q_start, q_end, t=50)
positions = np.array([robot.fkine(q).t for q in traj.q])
print(f"轨迹点数: {len(traj.q)}")
print(f"末端位移: {np.round(positions[-1] - positions[0], 4)} m")

# ============================================================
# Part 5: 动力学 (Puma560 - 有完整动力学参数)
# ============================================================
print("\n" + "=" * 50)
print("Part 5: 动力学分析 - Puma560 (有完整动力学参数)")
print("=" * 50)

puma = rtb.models.DH.Puma560()
q_puma = puma.qr
qd_zero = np.zeros(6)

# 重力补偿力矩
tau_g = puma.gravload(q_puma)
print(f"重力补偿力矩: {np.round(tau_g, 3)} N·m")

# 惯性矩阵
M = puma.inertia(q_puma)
print(f"惯性矩阵形状: {M.shape}")
print(f"主惯量（对角线）: {np.round(np.diag(M), 4)}")

# 逆动力学（给定运动状态 → 关节力矩）
qdd_zero = np.zeros(6)
tau_rne = puma.rne(q_puma, qd_zero, qdd_zero)
print(f"逆动力学力矩（静止）: {np.round(tau_rne, 3)} N·m")

# 沿轨迹计算重力力矩
traj_puma = rtb.jtraj(puma.qr, puma.qz, t=50)
tau_traj_puma = np.array([puma.gravload(q) for q in traj_puma.q])

# ============================================================
# Part 6: 可操作性分析 (Franka Panda)
# ============================================================
print("\n" + "=" * 50)
print("Part 6: 可操作性分析 - Franka Panda")
print("=" * 50)

manip_home = robot.manipulability(q_home)
print(f"Home 构型可操作性: {manip_home:.4f}")
manip_traj = [robot.manipulability(q) for q in traj.q]
cond_traj  = [np.linalg.cond(robot.jacobe(q)) for q in traj.q]
print(f"轨迹最小可操作性: {min(manip_traj):.4f}")
print(f"轨迹最大条件数:   {max(cond_traj):.1f}")

# ============================================================
# 画图
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Robot Kinematics & Dynamics Analysis\n'
             'Franka Panda (FK/IK/Jacobian/Trajectory) + Puma560 (Dynamics)',
             fontsize=12, fontweight='bold')

colors7 = plt.cm.tab10(np.linspace(0, 0.7, 7))

# 图1：关节轨迹
ax = axes[0, 0]
for i in range(7):
    ax.plot(traj.q[:, i], color=colors7[i], linewidth=1.5, label=f'q{i+1}')
ax.set_title('Joint Trajectory (Franka Panda)')
ax.set_xlabel('Step'); ax.set_ylabel('Angle (rad)')
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

# 图2：末端轨迹
ax = axes[0, 1]
for i, (label, color) in enumerate(zip(['X','Y','Z'],['red','green','blue'])):
    ax.plot(positions[:, i], color=color, linewidth=2, label=label)
ax.set_title('End-Effector Cartesian Trajectory')
ax.set_xlabel('Step'); ax.set_ylabel('Position (m)')
ax.legend(); ax.grid(True, alpha=0.3)

# 图3：关节速度
ax = axes[0, 2]
for i in range(7):
    ax.plot(traj.qd[:, i], color=colors7[i], linewidth=1.5)
ax.set_title('Joint Velocity (5th-order Polynomial)')
ax.set_xlabel('Step'); ax.set_ylabel('Velocity (rad/s)')
ax.grid(True, alpha=0.3)

# 图4：Puma560 重力力矩沿轨迹
colors6 = plt.cm.tab10(np.linspace(0, 0.6, 6))
ax = axes[1, 0]
for i in range(6):
    ax.plot(tau_traj_puma[:, i], color=colors6[i],
            linewidth=1.5, label=f'τ{i+1}')
ax.set_title('Gravity Torque along Trajectory (Puma560)')
ax.set_xlabel('Step'); ax.set_ylabel('Torque (N·m)')
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

# 图5：雅可比条件数
ax = axes[1, 1]
ax.plot(cond_traj, color='purple', linewidth=2)
ax.axhline(y=100, color='red', linestyle='--', label='Singularity threshold')
ax.set_title('Jacobian Condition Number (Franka Panda)')
ax.set_xlabel('Step'); ax.set_ylabel('Condition Number')
ax.legend(); ax.grid(True, alpha=0.3)

# 图6：可操作性
ax = axes[1, 2]
ax.plot(manip_traj, color='orange', linewidth=2)
ax.fill_between(range(len(manip_traj)), manip_traj, alpha=0.25, color='orange')
ax.set_title('Manipulability Index - Yoshikawa (Franka Panda)')
ax.set_xlabel('Step'); ax.set_ylabel('Manipulability')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('robotics_analysis.png', dpi=150, bbox_inches='tight')
print("\n图表已保存: robotics_analysis.png")
print("✅ 所有分析完成！")