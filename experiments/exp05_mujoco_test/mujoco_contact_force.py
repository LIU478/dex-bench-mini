# mujoco_contact_force.py
"""
MuJoCo 接触力分析实验
场景：球体落地碰撞 + 盒子施压
分析：法向力、切向力、接触点位置、摩擦锥
"""
import mujoco
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches 
import mujoco.viewer

# ── XML 模型定义 ──────────────────────────────────────────────
XML = """
<mujoco model="contact_analysis">

  <option timestep="0.002" gravity="0 0 -9.81" 
          integrator="RK4" cone="elliptic">
    <flag contact="enable"/>
  </option>

  <asset>
    <material name="ground_mat" rgba="0.8 0.8 0.8 1" 
              reflectance="0.1"/>
    <material name="ball_mat"   rgba="0.9 0.3 0.2 1"/>
    <material name="box_mat"    rgba="0.2 0.5 0.9 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse="1 1 1"/>

    <!-- 地面 -->
    <geom name="ground" type="plane" size="2 2 0.1"
          material="ground_mat" friction="0.8 0.005 0.0001"
          condim="4"/>

    <!-- 球体：从高处自由落下 -->
    <body name="ball" pos="0 0 1.5">
      <freejoint name="ball_joint"/>
      <geom name="ball_geom" type="sphere" size="0.08"
            material="ball_mat" mass="0.5"
            friction="0.8 0.005 0.0001" condim="4"/>
    </body>

    <!-- 盒子：缓慢从上方压下 -->
    <body name="box" pos="0.3 0 0.5">
      <freejoint name="box_joint"/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.05"
            material="box_mat" mass="1.0"
            friction="0.6 0.005 0.0001" condim="4"/>
    </body>
  </worldbody>
</mujoco>
"""

# ── 初始化模型 ────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_string(XML)
data  = mujoco.MjData(model)

print(f"模型加载成功")
print(f"  几何体数: {model.ngeom}")
print(f"  自由度:   {model.nv}")
print(f"  时间步:   {model.opt.timestep} s")

# ── 数据记录 ──────────────────────────────────────────────────
N_STEPS = 2000   # 仿真步数 = 4 秒

time_log          = []
n_contacts_log    = []
normal_force_log  = []   # 球-地面 法向力
tangent_force_log = []   # 球-地面 切向力合力
ball_z_log        = []   # 球的高度
box_z_log         = []   # 盒子高度
all_contact_pos   = []   # 所有接触点位置（用于可视化）

# 获取 geom id
ball_geom_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
ground_geom_id= mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
box_geom_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")

print(f"\n开始仿真 {N_STEPS} 步（{N_STEPS * model.opt.timestep:.1f} 秒）...")
data.qvel[0] = 2.0  # 给球横向初速度
# ── 仿真主循环 ────────────────────────────────────────────────
for step in range(N_STEPS):
    mujoco.mj_step(model, data)
    t = data.time

    # 记录时间和接触数
    time_log.append(t)
    n_contacts_log.append(data.ncon)

    # 记录物体高度
    ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    box_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    ball_z_log.append(data.xpos[ball_body_id, 2])
    box_z_log.append(data.xpos[box_body_id,  2])

    # 分析每个接触点
    max_normal  = 0.0
    max_tangent = 0.0

    for i in range(data.ncon):
        contact = data.contact[i]

        # 只关注球-地面的接触
        g1 = contact.geom[0]
        g2 = contact.geom[1]
        is_ball_ground = (
            (g1 == ball_geom_id and g2 == ground_geom_id) or
            (g1 == ground_geom_id and g2 == ball_geom_id)
        )

        if is_ball_ground:
            # 提取 6D 接触力
            force_torque = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force_torque)

            normal_f  = abs(force_torque[0])
            tangent_f = np.sqrt(force_torque[1]**2 + force_torque[2]**2)

            if normal_f > max_normal:
                max_normal  = normal_f
                max_tangent = tangent_f

            # 记录接触点位置
            all_contact_pos.append({
                'time': t,
                'pos':  contact.pos.copy(),
                'normal': normal_f,
                'tangent': tangent_f
            })

    normal_force_log.append(max_normal)
    tangent_force_log.append(max_tangent)

print(f"仿真完成！")
print(f"  最大法向力: {max(normal_force_log):.2f} N")
print(f"  最大切向力: {max(tangent_force_log):.4f} N")
print(f"  总接触事件: {len(all_contact_pos)}")

# 转换为 numpy 数组
time_arr    = np.array(time_log)
normal_arr  = np.array(normal_force_log)
tangent_arr = np.array(tangent_force_log)
ball_z_arr  = np.array(ball_z_log)
n_con_arr   = np.array(n_contacts_log)

# ── 画图 ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle('MuJoCo Contact Force Analysis\nSphere Drop + Box Compression',
             fontsize=14, fontweight='bold')

# ── 图1：球的高度轨迹 ──────────────────────────────────────
ax1 = fig.add_subplot(3, 3, 1)
ax1.plot(time_arr, ball_z_arr, color='tomato', linewidth=2, label='Ball Z')
ax1.axhline(y=0.08, color='gray', linestyle='--',
            alpha=0.7, label='Ground contact (r=0.08)')
ax1.set_title('Ball Height vs Time')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Height (m)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# ── 图2：法向力 ───────────────────────────────────────────
ax2 = fig.add_subplot(3, 3, 2)
ax2.plot(time_arr, normal_arr, color='steelblue', linewidth=1.5)
ax2.fill_between(time_arr, normal_arr, alpha=0.2, color='steelblue')
ax2.set_title('Normal Force (Ball-Ground)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Force (N)')
ax2.grid(True, alpha=0.3)

# 标注冲击峰值
peak_idx = np.argmax(normal_arr)
if normal_arr[peak_idx] > 1:
    ax2.annotate(f'Impact\n{normal_arr[peak_idx]:.1f} N',
                 xy=(time_arr[peak_idx], normal_arr[peak_idx]),
                 xytext=(time_arr[peak_idx]+0.2, normal_arr[peak_idx]*0.8),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=9, color='red')

# ── 图3：切向力 ───────────────────────────────────────────
ax3 = fig.add_subplot(3, 3, 3)
ax3.plot(time_arr, tangent_arr, color='darkorange', linewidth=1.5)
ax3.fill_between(time_arr, tangent_arr, alpha=0.2, color='darkorange')
ax3.set_title('Tangential Force (Friction)')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Force (N)')
ax3.grid(True, alpha=0.3)

# ── 图4：法向力 vs 切向力（摩擦锥分析）────────────────────
ax4 = fig.add_subplot(3, 3, 4)
mu = 0.8  # 摩擦系数
mask = normal_arr > 0.5  # 只看有接触的时刻
if mask.sum() > 0:
    ax4.scatter(normal_arr[mask], tangent_arr[mask],
                c=time_arr[mask], cmap='viridis',
                s=5, alpha=0.6)
    # 画摩擦锥边界
    n_range = np.linspace(0, normal_arr.max() * 1.1, 100)
    ax4.plot(n_range, mu * n_range, 'r--',
             linewidth=2, label=f'Friction cone (μ={mu})')
    ax4.set_title('Friction Cone Analysis')
    ax4.set_xlabel('Normal Force (N)')
    ax4.set_ylabel('Tangential Force (N)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    plt.colorbar(ax4.collections[0], ax=ax4, label='Time (s)')

# ── 图5：接触点数量随时间变化 ──────────────────────────────
ax5 = fig.add_subplot(3, 3, 5)
ax5.step(time_arr, n_con_arr, color='purple', linewidth=1.5, where='post')
ax5.fill_between(time_arr, n_con_arr, alpha=0.2,
                 color='purple', step='post')
ax5.set_title('Number of Contact Points')
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Count')
ax5.set_ylim(-0.1, n_con_arr.max() + 1)
ax5.grid(True, alpha=0.3)

# ── 图6：法向力频谱（FFT 分析）────────────────────────────
ax6 = fig.add_subplot(3, 3, 6)
if normal_arr.max() > 1:
    fft_vals = np.abs(np.fft.rfft(normal_arr - normal_arr.mean()))
    fft_freq = np.fft.rfftfreq(len(normal_arr),
                                d=model.opt.timestep)
    ax6.plot(fft_freq[:len(fft_freq)//4],
             fft_vals[:len(fft_freq)//4],
             color='green', linewidth=1.5)
    ax6.set_title('Normal Force FFT Spectrum')
    ax6.set_xlabel('Frequency (Hz)')
    ax6.set_ylabel('Amplitude')
    ax6.grid(True, alpha=0.3)

# ── 图7：摩擦系数利用率 ───────────────────────────────────
ax7 = fig.add_subplot(3, 3, 7)
mask2 = normal_arr > 1.0
if mask2.sum() > 0:
    utilization = np.where(mask2,
                           tangent_arr / (mu * normal_arr + 1e-8),
                           0)
    ax7.plot(time_arr, utilization, color='crimson', linewidth=1.5)
    ax7.axhline(y=1.0, color='red', linestyle='--',
                linewidth=2, label='Slip threshold (=1.0)')
    ax7.set_title('Friction Utilization Rate\n(>1 means slip)')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Tangential / (μ × Normal)')
    ax7.set_ylim(0, 1.5)
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

# ── 图8：冲击分析（放大首次碰撞）────────────────────────
ax8 = fig.add_subplot(3, 3, 8)
first_contact_idx = np.where(normal_arr > 1.0)[0]
if len(first_contact_idx) > 0:
    start = max(0, first_contact_idx[0] - 20)
    end   = min(len(time_arr), first_contact_idx[0] + 80)
    ax8.plot(time_arr[start:end] * 1000,
             normal_arr[start:end],
             color='steelblue', linewidth=2)
    ax8.set_title('Impact Detail (First Contact)')
    ax8.set_xlabel('Time (ms)')
    ax8.set_ylabel('Normal Force (N)')
    ax8.grid(True, alpha=0.3)

# ── 图9：统计摘要 ─────────────────────────────────────────
ax9 = fig.add_subplot(3, 3, 9)
ax9.axis('off')

peak_n = normal_arr.max()
peak_t = tangent_arr.max()
avg_n  = normal_arr[normal_arr > 0.5].mean() \
         if (normal_arr > 0.5).sum() > 0 else 0
contact_time = (normal_arr > 0.5).sum() * model.opt.timestep

stats = [
    ('Peak Normal Force',    f'{peak_n:.2f} N'),
    ('Peak Tangential Force',f'{peak_t:.4f} N'),
    ('Avg Normal (contact)', f'{avg_n:.2f} N'),
    ('Contact Duration',     f'{contact_time:.3f} s'),
    ('Friction Coeff (set)', f'{mu}'),
    ('Timestep',             f'{model.opt.timestep*1000:.1f} ms'),
    ('Sim Duration',         f'{time_arr[-1]:.1f} s'),
    ('Total Steps',          f'{N_STEPS}'),
]

ax9.text(0.05, 0.97, 'Simulation Summary',
         fontsize=11, fontweight='bold',
         transform=ax9.transAxes, va='top')
for i, (k, v) in enumerate(stats):
    y = 0.85 - i * 0.11
    ax9.text(0.05, y, f'{k}:', fontsize=9,
             transform=ax9.transAxes, va='top')
    ax9.text(0.7, y, v, fontsize=9, fontweight='bold',
             transform=ax9.transAxes, va='top', color='steelblue')

plt.tight_layout()
out = 'mujoco_contact_analysis.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\n图已保存: {out}")
print("✅ 接触力分析完成！")