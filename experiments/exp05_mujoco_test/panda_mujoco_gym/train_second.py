"""
针对Push：从 500k 模型加载，降低学习率，带早停继续训练到 1M 步
适用于 FrankaPushSparse-v0 任务，SAC + HER
"""

import panda_mujoco_gym
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.her import HerReplayBuffer
import os

# ========== 配置 ==========
ENV_ID = "FrankaPushSparse-v0"
CHECKPOINT_500K = "./models/final_model.zip"   # 已有的 500k 模型路径
TOTAL_TIMESTEPS_TARGET = 1_000_000                  # 最终目标步数
STEPS_TO_CONTINUE = TOTAL_TIMESTEPS_TARGET - 500_000  # 还需训练步数
LOG_DIR = "./logs/"
BEST_MODEL_DIR = "./models/best/"
EVAL_LOG_DIR = "./eval_logs/"
# =========================

# 创建保存目录
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_LOG_DIR, exist_ok=True)

# ---- 1. 创建训练环境（与保存模型时一致）----
train_env = Monitor(gym.make(ENV_ID))
train_env = DummyVecEnv([lambda: train_env])

# ---- 2. 加载 500k 模型 ----
print(f"加载模型: {CHECKPOINT_500K}")
model = SAC.load(CHECKPOINT_500K, env=train_env, print_system_info=True)

# ---- 3. 降低学习率并重建优化器 ----
NEW_LEARNING_RATE = 1e-4   # 原默认是 3e-4
model.learning_rate = NEW_LEARNING_RATE
# 重建优化器使新学习率生效
model.policy.optimizer = model.policy._build_optimizer(learning_rate=model.learning_rate)

# 可选：同时调整熵系数 (避免过早确定性策略)
# model.ent_coef = 0.05cd

print(f"学习率已降低至 {model.learning_rate}")

# ---- 4. 创建独立的评估环境 ----
eval_env = Monitor(gym.make(ENV_ID))
eval_env = DummyVecEnv([lambda: eval_env])

# ---- 5. 创建早停回调 ----
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=BEST_MODEL_DIR,     # 最佳模型保存路径
    log_path=EVAL_LOG_DIR,                   # 评估日志路径
    eval_freq=10_000,                        # 每 10k 步评估一次
    deterministic=True,                      # 使用确定性动作
    render=False,
    n_eval_episodes=20,                      # 每次评估跑 20 个 episode
    callback_on_new_best=None,
    verbose=1
)

# ---- 6. 继续训练 ----
print(f"从当前步数 {model.num_timesteps} 继续训练 {STEPS_TO_CONTINUE} 步，总计目标 {TOTAL_TIMESTEPS_TARGET} 步")
model.learn(
    total_timesteps=STEPS_TO_CONTINUE,
    callback=eval_callback,
    reset_num_timesteps=False   # 不重置时间步，保证连续性
)

# ---- 7. 保存最终模型 ----
final_model_path = "./models/final_model_1M_finetuned"
model.save(final_model_path)
print(f"训练完成！最终模型保存在 {final_model_path}.zip")
print(f"最佳模型保存在 {BEST_MODEL_DIR}/best_model.zip")

# ---- 8. 清理环境 ----
train_env.close()
eval_env.close()