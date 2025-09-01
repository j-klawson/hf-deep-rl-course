#!/usr/bin/env python3

import sys
# Virtual display
from pyvirtualdisplay import Display
import gymnasium
from huggingface_sb3 import load_from_hub, package_to_hub
from huggingface_hub import (
    notebook_login,
)  # To log to our Hugging Face account to be able to upload models to the Hub.
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
import torch

print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    print("name[0]:", torch.cuda.get_device_name(0))
else:
    import platform
    print("platform:", platform.platform())
    sys.exit("Error: CUDA is not available. Exiting.")

# Setup virtual display
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

# First, we create our environment called LunarLander-v2
# env = make_vec_env("LunarLander-v2", n_envs=16) 
env = gym.make('LunarLander-v2')
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
model.learn(total_timesteps=1000000)
# Save the model
model_name = "ppo-LunarLander-v2"
model.save(model_name)
