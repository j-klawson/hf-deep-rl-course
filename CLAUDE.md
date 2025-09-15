# Claude Context - Hugging Face Deep RL Course

## Project Overview
This is a repository for exercises from the Hugging Face Deep Reinforcement Learning Course (https://huggingface.co/learn/deep-rl-course).

## Current Structure
- `unit1.py` - PPO Lunar Lander implementation (completed)
- `unit1_push.py` - Script to push Unit 1 model to Hugging Face Hub
- `unit2.py` - Q-Learning implementation (in progress)
- `requirements.txt` - Dependencies for the course
- `README.md` - Basic project description

## Git Status
- Current branch: `unit2`
- Modified files: `requirements.txt`
- Untracked: `ppo-LunarLander-v2.zip`

## Unit 1 (Completed)
- **Environment**: LunarLander-v2
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Model**: Trained for 50M timesteps with CUDA acceleration
- **Hub**: Model pushed to `j-klawson/ppo-LunarLander-v2`
- **Issues Fixed**: Removed duplicate gymnasium imports

## Unit 2 (Current Focus)
- **Objective**: Implement Q-Learning from scratch
- **Environments**:
  - FrozenLake-v1 (non-slippery 4x4)
  - Taxi-v3
- **Requirements**:
  - Achieve >= 4.5 result on leaderboard (mean_reward - std_reward)
  - Push Taxi model to Hugging Face Hub
- **Status**: Basic structure started, needs Q-Learning implementation

## Dependencies
- PyTorch with CUDA support
- Stable Baselines3
- Gymnasium with Box2D
- Hugging Face Hub integration
- Virtual display support (pyvirtualdisplay)

## Development Notes
- CUDA checks implemented for GPU acceleration
- Virtual display configured for headless rendering
- Hub integration ready for model sharing
- Code follows executable script pattern with proper shebangs

## Requirements Issues Fixed
- `pickle5` removed (not needed for Python 3.8+, use standard `pickle`)
- `pyyaml==6.0` changed to `pyyaml>=6.0` for compatibility
- `pyglet==1.5.1` requires separate install: `pip install pyglet --index-url https://pypi.org/simple/`