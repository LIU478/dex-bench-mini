import mujoco
import mujoco.viewer
import numpy as np
import time

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

model = mujoco.MjModel.from_xml_string(XML)
data  = mujoco.MjData(model)
data.qvel[0] = 2.0

print("打开 3D 窗口，按 Ctrl+C 退出...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        # 控制仿真速度和真实时间一致
        elapsed = time.time() - step_start
        time.sleep(max(0, model.opt.timestep - elapsed))