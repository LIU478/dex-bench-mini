import panda_mujoco_gym
import time
import gymnasium as gym

env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="human")
obs, info = env.reset()
print("Successful")
print("observation_space:", env.observation_space)
print("action_space:", env.action_space)

for step in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
    time.sleep(0.02)

env.close()
print("Over！")
