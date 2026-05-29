'''
随机动作，没有加载预训练模型
'''
import panda_mujoco_gym, gymnasium as gym, time

env = gym.make('FrankaSlideSparse-v0', render_mode='human')
obs, info = env.reset()
print('Push 任务启动，按 Ctrl+C 退出')
for _ in range(2000):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, info = env.reset()
    time.sleep(0.02)
env.close()