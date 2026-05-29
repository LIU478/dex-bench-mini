import panda_mujoco_gym 
import gymnasium as gym 
envs = [e for e in gym.envs.registry.keys() if 'franka' in e.lower() or 'panda' in e.lower() or 'pick' in e.lower()] 
print('available env:') 
[print(' ', e) for e in envs] 
