#!/usr/bin/env python3

import sys
import subprocess
import platform
import torch
# Virtual display
from pyvirtualdisplay import Display
import gymnasium as gym
from huggingface_sb3 import load_from_hub, package_to_hub
from huggingface_hub import notebook_login
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

def check_nvidia_driver():
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            print("NVIDIA driver check failed:\n", result.stderr)
            return False
        print(result.stdout.splitlines()[0])  # prints NVIDIA-SMI header line
        return True
    except FileNotFoundError:
        print("Error: nvidia-smi not found. NVIDIA driver may not be installed.")
        return False

def check_pytorch_cuda():
    print("CUDA available in PyTorch:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device count:", torch.cuda.device_count())
        print("GPU Name [0]:", torch.cuda.get_device_name(0))
        print("PyTorch CUDA build:", torch.version.cuda)
        return True
    else:
        print("Platform:", platform.platform())
        return False

# Check prerequs for GPU processing
if not (check_nvidia_driver()):
   sys.exit("Error: NVIDIA driver is missing. Exiting.")
if not (check_pytorch_cuda()):
   sys.exit("PyTorch CUDA support is missing. Exiting.")
   
# Setup virtual display
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

# First, we create our environment called LunarLander-v2
env = make_vec_env("LunarLander-v2", n_envs=16) 
env.reset()

# We added some parameters to accelerate the training
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    device="cuda",
    verbose=1
)

# Train it for 1,000,000 timesteps
model.learn(total_timesteps=50000000)
# Save the model
model_name = "ppo-LunarLander-v2"
model.save(model_name)

eval_env = Monitor(gym.make("LunarLander-v2"))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
