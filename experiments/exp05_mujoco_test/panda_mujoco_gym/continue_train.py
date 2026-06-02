"""
继续训练已有的 Push 模型（方案 A）
"""
import panda_mujoco_gym
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os

ENV_ID = "FrankaPushSparse-v0"
os.makedirs("./models/", exist_ok=True)

env = DummyVecEnv([lambda: Monitor(gym.make(ENV_ID))])

print("加载已有模型继续训练...")
model = SAC.load("./models/final_model", env=env)

# 修复：让模型重新收集经验填充空 buffer
model.learning_starts = 1000

# 关键：必须用 reset_num_timesteps=True
# 让 num_timesteps 归零，这样 learning_starts 才生效
model.learn(total_timesteps=500_000, reset_num_timesteps=True)

model.save("./models/final_model_1M")
print("训练完成！新模型: ./models/final_model_1M.zip")