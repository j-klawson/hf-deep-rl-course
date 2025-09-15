# Hugging Face Deep Reinforcement Learning Course

Exercises and implementations for the Hugging Face [Deep Reinforcement Learning Course](https://huggingface.co/learn/deep-rl-course).

## Project Structure

```
├── unit1.py              # PPO Lunar Lander training
├── unit1_push.py         # Push Unit 1 model to Hub
├── unit2.py              # Unified Q-Learning script with CLI
├── gpu_check.py          # GPU validation utilities
├── requirements.txt      # All dependencies
├── CLAUDE.md            # Detailed project context
└── README.md            # This file
```

## Completed Units

### Unit 1: PPO Lunar Lander 
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Environment**: LunarLander-v2
- **Training**: 50M timesteps with CUDA acceleration
- **Score**: 284.97 ± 14.71 (excellent performance)
- **Model**: [j-klawson/ppo-LunarLander-v2](https://huggingface.co/j-klawson/ppo-LunarLander-v2)

### Unit 2: Q-Learning 
- **Algorithm**: Q-Learning (tabular)
- **Environments**:
  - FrozenLake-v1 (4x4, non-slippery) - Score: 1.00 (perfect)
  - Taxi-v3 (available via CLI)
- **Models**:
  - [j-klawson/q-FrozenLake-v1-4x4-noSlippery](https://huggingface.co/j-klawson/q-FrozenLake-v1-4x4-noSlippery)
  - [j-klawson/q-Taxi-v3](https://huggingface.co/j-klawson/q-Taxi-v3)

## Setup

```bash
# Setup Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For pyglet (if needed)
pip install pyglet --index-url https://pypi.org/simple/

# Check GPU support (for Unit 1)
python gpu_check.py
```

## Usage

### Unit 1: PPO Training
```bash
# Train PPO agent (requires GPU)
python unit1.py

# Push trained model to Hub
python unit1_push.py
```

### Unit 2: Q-Learning Training
```bash
# Train FrozenLake agent
python unit2.py --frozenlake

# Train FrozenLake and upload to Hub
python unit2.py --frozenlake --upload

# Train Taxi agent
python unit2.py --taxi

# Train Taxi and upload to Hub
python unit2.py --taxi --upload

# Use different username
python unit2.py --taxi --upload --username your-username
```

## GPU Requirements

- **Unit 1 (PPO)**: Requires CUDA-capable GPU (uses gpu_check.py for validation)
- **Unit 2 (Q-Learning)**: CPU only (NumPy-based, no GPU needed)
