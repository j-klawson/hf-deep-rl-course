#!/usr/bin/env python3
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from huggingface_sb3 import package_to_hub
from huggingface_hub import notebook_login  # or use HF_TOKEN env var

# ---- Config ----
ENV_ID = "LunarLander-v2"          # switch to "LunarLander-v3" if your Gymnasium build requires it
MODEL_PATH = "ppo-LunarLander-v2"  # produced by model.save(...)
MODEL_NAME = "ppo-LunarLander-v2"
MODEL_ARCH = "PPO"
REPO_ID = "j-klawson/ppo-LunarLander-v2"
COMMIT_MSG = "Upload PPO LunarLander-v2 trained agent"

def main():
    # Load trained model
    model = PPO.load(MODEL_PATH)

    # Eval env; rgb_array enables video preview on the Hub
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(ENV_ID, render_mode="rgb_array"))])

    # Optional: quick evaluation printout
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # Push to Hub
    package_to_hub(
        model=model,
        model_name=MODEL_NAME,
        model_architecture=MODEL_ARCH,
        env_id=ENV_ID,
        eval_env=eval_env,
        repo_id=REPO_ID,
        commit_message=COMMIT_MSG,
    )

if __name__ == "__main__":
    main()

