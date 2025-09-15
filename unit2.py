#!/usr/bin/env python3
"""
Unit 2: Q-Learning Implementation for Hugging Face Deep RL Course
Supports both FrozenLake-v1 and Taxi-v3 environments

Usage:
    python unit2.py --frozenlake [--upload]
    python unit2.py --taxi [--upload]
"""

import argparse
import numpy as np
import gymnasium as gym
import random
import imageio
import os
import pickle
from tqdm import tqdm
from pyvirtualdisplay import Display
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save
from pathlib import Path
import datetime
import json

# Virtual display setup
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

def initialize_q_table(state_space, action_space):
    """Initialize Q-table with zeros"""
    return np.zeros((state_space, action_space))

def greedy_policy(qtable, state):
    """Choose action with highest Q-value for given state"""
    return np.argmax(qtable[state][:])

def epsilon_greedy_policy(qtable, state, epsilon, action_space):
    """Epsilon-greedy policy: explore vs exploit"""
    if random.uniform(0, 1) > epsilon:
        return greedy_policy(qtable, state)
    else:
        return action_space.sample()

def train_qlearning(env, qtable, config):
    """Train Q-learning agent"""
    print(f"Training Q-Learning on {config['env_id']} for {config['n_training_episodes']:,} episodes...")

    for episode in tqdm(range(config['n_training_episodes'])):
        # Decay epsilon
        epsilon = config['min_epsilon'] + (config['max_epsilon'] - config['min_epsilon']) * np.exp(-config['decay_rate'] * episode)

        # Reset environment
        state, info = env.reset()
        terminated = False
        truncated = False

        # Play episode
        for step in range(config['max_steps']):
            # Choose action using epsilon-greedy policy
            action = epsilon_greedy_policy(qtable, state, epsilon, env.action_space)

            # Take action and observe outcome
            new_state, reward, terminated, truncated, info = env.step(action)

            # Update Q-value using Bellman equation
            qtable[state][action] = qtable[state][action] + config['learning_rate'] * (
                reward + config['gamma'] * np.max(qtable[new_state]) - qtable[state][action]
            )

            # Check if episode is done
            if terminated or truncated:
                break

            state = new_state

    return qtable

def evaluate_agent(env, qtable, config):
    """Evaluate the trained agent"""
    episode_rewards = []

    for episode in tqdm(range(config['n_eval_episodes']), desc="Evaluating"):
        if config['eval_seed'] and episode < len(config['eval_seed']):
            state, info = env.reset(seed=config['eval_seed'][episode])
        else:
            state, info = env.reset()

        total_reward = 0
        terminated = False
        truncated = False

        for step in range(config['max_steps']):
            action = greedy_policy(qtable, state)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

def record_video(env, qtable, output_path, fps=1):
    """Record a video of the trained agent"""
    images = []
    state, info = env.reset(seed=random.randint(0, 500))
    img = env.render()
    images.append(img)

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = greedy_policy(qtable, state)
        state, reward, terminated, truncated, info = env.step(action)
        img = env.render()
        images.append(img)

    imageio.mimsave(output_path, [np.array(img) for img in images], fps=fps)

def push_to_hub(repo_id, model, env, video_fps=1):
    """Push trained model to Hugging Face Hub"""
    print(f"Pushing model to Hub: {repo_id}")

    _, repo_name = repo_id.split("/")
    api = HfApi()

    # Create repository
    repo_url = api.create_repo(repo_id=repo_id, exist_ok=True)

    # Download existing files
    repo_local_path = Path(snapshot_download(repo_id=repo_id))

    # Save environment-specific metadata
    if hasattr(env.spec, 'kwargs'):
        if env.spec.kwargs.get("map_name"):
            model["map_name"] = env.spec.kwargs.get("map_name")
        if env.spec.kwargs.get("is_slippery") == False:
            model["slippery"] = False

    # Pickle the model
    with open(repo_local_path / "q-learning.pkl", "wb") as f:
        pickle.dump(model, f)

    # Evaluate and create results
    mean_reward, std_reward = evaluate_agent(env, model["qtable"], model)

    evaluate_data = {
        "env_id": model["env_id"],
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "result": mean_reward - std_reward,
        "n_eval_episodes": model["n_eval_episodes"],
        "eval_datetime": datetime.datetime.now().isoformat(),
    }

    with open(repo_local_path / "results.json", "w") as outfile:
        json.dump(evaluate_data, outfile)

    # Create environment name for model card
    env_name = model["env_id"]
    if hasattr(env.spec, 'kwargs'):
        if env.spec.kwargs.get("map_name"):
            env_name += "-" + env.spec.kwargs.get("map_name")
        if env.spec.kwargs.get("is_slippery") == False:
            env_name += "-noSlippery"

    # Create metadata
    metadata = {"tags": [env_name, "q-learning", "reinforcement-learning", "custom-implementation"]}

    eval_metadata = metadata_eval_result(
        model_pretty_name=repo_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
        dataset_pretty_name=env_name,
        dataset_id=env_name,
    )

    metadata = {**metadata, **eval_metadata}

    # Create model card
    model_card = f"""# **Q-Learning** Agent playing **{env_name}**

This is a trained model of a **Q-Learning** agent playing **{env_name}**.

## Results

- **Mean reward**: {mean_reward:.2f} +/- {std_reward:.2f}
- **Result**: {mean_reward - std_reward:.2f} (mean - std)
- **Evaluation episodes**: {model["n_eval_episodes"]}

## Environment Details

- **Environment**: {model["env_id"]}
- **States**: {env.observation_space.n} possible states
- **Actions**: {env.action_space.n} possible actions

## Training Hyperparameters

- **Episodes**: {model["n_training_episodes"]:,}
- **Learning rate**: {model["learning_rate"]}
- **Gamma (discount)**: {model["gamma"]}
- **Epsilon decay**: {model["decay_rate"]}
- **Max epsilon**: {model["max_epsilon"]}
- **Min epsilon**: {model["min_epsilon"]}

## Usage

```python
from huggingface_hub import hf_hub_download
import pickle
import gymnasium as gym
import numpy as np

# Load the Q-table
model_path = hf_hub_download(repo_id="{repo_id}", filename="q-learning.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Create environment
env = gym.make(model["env_id"])

# Use the trained agent
def greedy_policy(qtable, state):
    return np.argmax(qtable[state][:])

state, info = env.reset()
while True:
    action = greedy_policy(model["qtable"], state)
    state, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

"""

    # Write README
    readme_path = repo_local_path / "README.md"
    with readme_path.open("w", encoding="utf-8") as f:
        f.write(model_card)

    # Save metadata
    metadata_save(readme_path, metadata)

    # Record video
    video_path = repo_local_path / "replay.mp4"
    record_video(env, model["qtable"], video_path, video_fps)

    # Upload everything
    api.upload_folder(
        repo_id=repo_id,
        folder_path=repo_local_path,
        path_in_repo=".",
    )

    print(f"Model pushed successfully!")
    print(f"View at: {repo_url}")

    return mean_reward, std_reward

def get_frozenlake_config():
    """Configuration for FrozenLake-v1"""
    return {
        "env_id": "FrozenLake-v1",
        "env_kwargs": {"map_name": "4x4", "is_slippery": False},
        "n_training_episodes": 10000,
        "learning_rate": 0.7,
        "max_steps": 99,
        "gamma": 0.95,
        "max_epsilon": 1.0,
        "min_epsilon": 0.05,
        "decay_rate": 0.0005,
        "n_eval_episodes": 100,
        "eval_seed": [],
        "repo_name": "q-FrozenLake-v1-4x4-noSlippery"
    }

def get_taxi_config():
    """Configuration for Taxi-v3"""
    eval_seed = [16, 54, 165, 177, 191, 191, 120, 80, 149, 178, 48, 38, 6, 125, 174, 73, 50, 172, 100, 148, 146, 6, 25, 40, 68, 148, 49, 167, 9, 97, 164, 176, 61, 7, 54, 55, 161, 131, 184, 51, 170, 12, 120, 113, 95, 126, 51, 98, 36, 135, 54, 82, 45, 95, 89, 59, 95, 124, 9, 113, 58, 85, 51, 134, 121, 169, 105, 21, 30, 11, 50, 65, 12, 43, 82, 145, 152, 97, 106, 55, 31, 85, 38, 112, 102, 168, 123, 97, 21, 83, 158, 26, 80, 63, 5, 81, 32, 11, 28, 148]
    return {
        "env_id": "Taxi-v3",
        "env_kwargs": {},
        "n_training_episodes": 25000,
        "learning_rate": 0.7,
        "max_steps": 99,
        "gamma": 0.95,
        "max_epsilon": 1.0,
        "min_epsilon": 0.05,
        "decay_rate": 0.0005,
        "n_eval_episodes": 100,
        "eval_seed": eval_seed,
        "repo_name": "q-Taxi-v3"
    }

env_id = "Taxi-v3"  # Name of the environment
max_steps = 99  # Max steps per episode
gamma = 0.95  # Discounting rate

# Exploration parameters
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.05  # Minimum exploration probability
decay_rate = 0.005  # Exponential decay rate for exploration prob

def train_environment(config, upload_to_hub=False, username="j-klawson"):
    """Train Q-Learning agent for specified environment"""
    print(f"\n{'='*50}")
    print(f"Training Q-Learning Agent: {config['env_id']}")
    print(f"{'='*50}")

    # Create environment
    env = gym.make(config["env_id"], render_mode="rgb_array", **config["env_kwargs"])

    # Display environment info
    print(f"\nEnvironment Info:")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    print(f"States: {env.observation_space.n}")
    print(f"Actions: {env.action_space.n}")

    # Initialize Q-table
    qtable = initialize_q_table(env.observation_space.n, env.action_space.n)

    # Train agent
    qtable = train_qlearning(env, qtable, config)

    # Evaluate agent
    print(f"\nEvaluating agent...")
    mean_reward, std_reward = evaluate_agent(env, qtable, config)
    result = mean_reward - std_reward

    print(f"\nResults:")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Result (mean - std): {result:.2f}")

    # Create model dictionary
    model = {
        "env_id": config["env_id"],
        "max_steps": config["max_steps"],
        "n_training_episodes": config["n_training_episodes"],
        "n_eval_episodes": config["n_eval_episodes"],
        "eval_seed": config["eval_seed"],
        "learning_rate": config["learning_rate"],
        "gamma": config["gamma"],
        "max_epsilon": config["max_epsilon"],
        "min_epsilon": config["min_epsilon"],
        "decay_rate": config["decay_rate"],
        "qtable": qtable,
    }

    # Upload to Hub if requested
    if upload_to_hub:
        repo_id = f"{username}/{config['repo_name']}"
        push_to_hub(repo_id, model, env)
    else:
        print(f"\nModel trained successfully!")
        print(f"To upload to Hub, use: --upload flag")

    return qtable, env, model

def main():
    parser = argparse.ArgumentParser(description="Train Q-Learning agents for Unit 2")

    # Environment selection (mutually exclusive)
    env_group = parser.add_mutually_exclusive_group(required=True)
    env_group.add_argument("--frozenlake", action="store_true",
                          help="Train FrozenLake-v1 agent")
    env_group.add_argument("--taxi", action="store_true",
                          help="Train Taxi-v3 agent")

    # Options
    parser.add_argument("--upload", action="store_true",
                       help="Upload trained model to Hugging Face Hub")
    parser.add_argument("--username", default="j-klawson",
                       help="Hugging Face username (default: j-klawson)")

    args = parser.parse_args()

    # Select configuration based on environment
    if args.frozenlake:
        config = get_frozenlake_config()
    elif args.taxi:
        config = get_taxi_config()

    # Train the selected environment
    qtable, env, model = train_environment(
        config,
        upload_to_hub=args.upload,
        username=args.username
    )

    print(f"\nTraining completed for {config['env_id']}!")

if __name__ == "__main__":
    main()