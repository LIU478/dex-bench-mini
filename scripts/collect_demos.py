"""Collect LEAP Hand demonstration data using trained policy."""
import argparse
import math
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_demos", type=int, default=200)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--max_steps_per_ep", type=int, default=300)
parser.add_argument("--out", type=str,
    default="/root/autodl-tmp/dex-bench-mini/experiments/exp09_demo_collection/outputs/demos.hdf5")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
import h5py
import yaml
import gymnasium as gym
import LEAP_Isaaclab  # noqa

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper, RlGamesGpuEnv
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

print(f"[collect] checkpoint : {args.checkpoint}")
print(f"[collect] num_demos  : {args.num_demos}")

ckpt_dir = os.path.dirname(args.checkpoint)
agent_yaml = None
for up in range(5):
    candidate = os.path.normpath(
        os.path.join(ckpt_dir, *([".."] * up), "params", "agent.yaml"))
    if os.path.exists(candidate):
        agent_yaml = candidate
        break
if agent_yaml is None:
    agent_yaml = os.path.join(
        os.path.expanduser("~"),
        "external/LEAP_Hand_Isaac_Lab/source/LEAP_Isaaclab/LEAP_Isaaclab"
        "/tasks/leap_hand_reorient/agents/rl_games_ppo_cfg.yaml")
print(f"[collect] agent_yaml : {agent_yaml}")

with open(agent_yaml) as f:
    agent_cfg = yaml.safe_load(f)

rl_device    = agent_cfg["params"]["config"].get("device", "cuda:0")
clip_obs     = agent_cfg["params"]["env"].get("clip_observations", math.inf)
clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

env_cfg = parse_env_cfg("Isaac-Reorient-Cube-Leap", device=rl_device, num_envs=args.num_envs)
base_env = gym.make("Isaac-Reorient-Cube-Leap", cfg=env_cfg)
env = RlGamesVecEnvWrapper(base_env, rl_device, clip_obs, clip_actions)

vecenv.register("IsaacRlgWrapper",
    lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs))
env_configurations.register("rlgpu",
    {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

agent_cfg["params"]["load_checkpoint"] = True
agent_cfg["params"]["load_path"] = args.checkpoint
agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

runner = Runner()
runner.load(agent_cfg)
agent = runner.create_player()
agent.restore(args.checkpoint)
agent.reset()

obs = env.reset()
if isinstance(obs, dict):
    obs = obs["obs"]
_ = agent.get_batch_size(obs, 1)
if agent.is_rnn:
    agent.init_rnn()

obs_dim = base_env.observation_space.shape[-1]
act_dim = base_env.action_space.shape[-1]
print(f"[collect] obs_dim={obs_dim}, act_dim={act_dim}, num_envs={args.num_envs}")

os.makedirs(os.path.dirname(args.out), exist_ok=True)

ep_buffers = [{"obs": [], "actions": [], "rewards": [], "dones": []}
              for _ in range(args.num_envs)]
collected = 0
all_episodes = []
t0 = time.time()
step_count = 0

print("[collect] Starting collection loop (prints every 100 steps and each saved demo)...")

while collected < args.num_demos:
    with torch.inference_mode():
        obs_t = agent.obs_to_torch(obs)
        actions = agent.get_action(obs_t, is_deterministic=True)
        obs_np = obs_t.cpu().numpy()
        actions_np = actions.cpu().numpy() if isinstance(actions, torch.Tensor) else np.array(actions)
        next_obs, rewards, dones, info = env.step(actions)
        if isinstance(next_obs, dict):
            next_obs = next_obs["obs"]
        if agent.is_rnn and agent.states is not None:
            dones_bool = dones.bool() if isinstance(dones, torch.Tensor) else torch.tensor(dones, dtype=torch.bool)
            if dones_bool.any():
                for s in agent.states:
                    s[:, dones_bool, :] = 0.0

    rewards_np = rewards.cpu().numpy() if isinstance(rewards, torch.Tensor) else np.array(rewards)
    dones_np   = dones.cpu().numpy()   if isinstance(dones, torch.Tensor) else np.array(dones)

    step_count += 1
    if step_count % 100 == 0:
        fps = (step_count * args.num_envs) / (time.time() - t0)
        print(f"[collect] step={step_count} | collected={collected}/{args.num_demos} | fps={fps:.0f}")

    for i in range(args.num_envs):
        ep_buffers[i]["obs"].append(obs_np[i].copy())
        ep_buffers[i]["actions"].append(actions_np[i].copy())
        ep_buffers[i]["rewards"].append(float(rewards_np[i]))
        ep_buffers[i]["dones"].append(bool(dones_np[i]))

        if dones_np[i] or len(ep_buffers[i]["obs"]) >= args.max_steps_per_ep:
            ep = {
                "obs":     np.array(ep_buffers[i]["obs"],     dtype=np.float32),
                "actions": np.array(ep_buffers[i]["actions"], dtype=np.float32),
                "rewards": np.array(ep_buffers[i]["rewards"], dtype=np.float32),
                "dones":   np.array(ep_buffers[i]["dones"],   dtype=bool),
            }
            all_episodes.append(ep)
            collected += 1
            ep_len = len(ep_buffers[i]["obs"])
            ep_ret = float(ep["rewards"].sum())
            print(f"[collect]   saved demo {collected}/{args.num_demos} (env={i}, len={ep_len}, ret={ep_ret:.1f})")
            ep_buffers[i] = {"obs": [], "actions": [], "rewards": [], "dones": []}
            if collected >= args.num_demos:
                break

    obs = next_obs

ep_lengths = [len(e["obs"]) for e in all_episodes]
ep_returns = [float(e["rewards"].sum()) for e in all_episodes]

print(f"[collect] Writing {len(all_episodes)} episodes to {args.out}...")

with h5py.File(args.out, "w") as f:
    f.attrs["num_demos"]  = len(all_episodes)
    f.attrs["obs_dim"]    = obs_dim
    f.attrs["act_dim"]    = act_dim
    f.attrs["task"]       = "Isaac-Reorient-Cube-Leap"
    f.attrs["checkpoint"] = args.checkpoint
    f.attrs["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    data_grp = f.create_group("data")
    for idx, ep in enumerate(all_episodes):
        grp = data_grp.create_group(f"demo_{idx}")
        grp.create_dataset("obs",     data=ep["obs"],     compression="gzip")
        grp.create_dataset("actions", data=ep["actions"], compression="gzip")
        grp.create_dataset("rewards", data=ep["rewards"], compression="gzip")
        grp.create_dataset("dones",   data=ep["dones"],   compression="gzip")
        grp.attrs["ep_len"]    = len(ep["obs"])
        grp.attrs["ep_return"] = float(ep["rewards"].sum())
    stats = f.create_group("stats")
    stats.create_dataset("ep_lengths", data=np.array(ep_lengths))
    stats.create_dataset("ep_returns", data=np.array(ep_returns))
    stats.attrs["mean_ep_len"]    = float(np.mean(ep_lengths))
    stats.attrs["mean_ep_return"] = float(np.mean(ep_returns))
    stats.attrs["std_ep_return"]  = float(np.std(ep_returns))

print("[collect] Done.")
print(f"  demos        : {len(all_episodes)}")
print(f"  mean ep_len  : {np.mean(ep_lengths):.1f}")
print(f"  mean ep_ret  : {np.mean(ep_returns):.1f}")
print(f"  file size    : {os.path.getsize(args.out)/1e6:.1f} MB")

env.close()
simulation_app.close()
