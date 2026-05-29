import panda_mujoco_gym  # 导入你的环境
import gymnasium as gym  
from stable_baselines3 import SAC  # 导入强化学习算法
from stable_baselines3.her import HerReplayBuffer 
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os

# ========== 配置部分 ==========
ENV_ID = "FrankaPushSparse-v0"  # 任务: 可选择 Push, Slide, PickAndPlace
TOTAL_TIMESTEPS = 500_000        # 总训练步数: Push任务论文推荐50万步
LOG_DIR = "./logs/"              # 日志保存路径
SAVE_DIR = "./models/"           # 模型保存路径
# =============================

# 创建保存模型的文件夹
os.makedirs(SAVE_DIR, exist_ok=True)

# 创建环境，并包装一下以便记录训练数据
env = Monitor(gym.make(ENV_ID))
# 因为环境观测是字典格式，需要包装一下
env = DummyVecEnv([lambda: env])

# 创建模型
# 使用SAC算法，并启用HER (Hindsight Experience Replay)
model = SAC(
    "MultiInputPolicy",          # 处理字典格式的观测
    env,
    replay_buffer_class=HerReplayBuffer,  # 开启HER，提高学习效率
    replay_buffer_kwargs={
        'n_sampled_goal': 4,
        'goal_selection_strategy': 'future'
    },
    verbose=1,
    tensorboard_log=LOG_DIR,     # 可选，用于记录训练曲线
)

# 开始训练
print(f"开始训练 {ENV_ID} 任务，共 {TOTAL_TIMESTEPS} 步...")
model.learn(total_timesteps=TOTAL_TIMESTEPS)

# 训练完成后，保存最终的模型
model.save(f"{SAVE_DIR}/final_model")
print(f"训练完成！模型已保存至 {SAVE_DIR}/final_model.zip")