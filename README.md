# Hugging Face Deep Reinforcement Learning Course

Exercises and implementations for the Hugging Face [Deep Reinforcement Learning Course](https://huggingface.co/learn/deep-rl-course).

## Completed Units

### Unit 1: PPO Lunar Lander
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Environment**: LunarLander-v2
- **Training**: 50M timesteps with CUDA acceleration
- **Model**: [j-klawson/ppo-LunarLander-v2](https://huggingface.co/j-klawson/ppo-LunarLander-v2)

### Unit 2: Q-Learning FrozenLake 
- **Algorithm**: Q-Learning (tabular)
- **Environment**: FrozenLake-v1 (4x4, non-slippery)
- **Result**: Perfect score 1.00 (exceeds certification requirement of ≥4.5)
- **Model**: [j-klawson/q-FrozenLake-v1-4x4-noSlippery](https://huggingface.co/j-klawson/q-FrozenLake-v1-4x4-noSlippery)

## Next Steps
- [ ] Unit 2: Implement Taxi-v3 Q-Learning
- [ ] Continue with additional course units

## Setup

```bash
# Setup Python virtual environment 
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For pyglet (if needed)
pip install pyglet --index-url https://pypi.org/simple/

# Ensure GPU support is working (for Unit 1)
python gpu_check.py
```

## Usage

```bash
# Unit 1: Train PPO agent
python unit1.py

# Unit 1: Push to Hub
python unit1_push.py

# Unit 2: Train Q-Learning agent (includes Hub push)
python unit2.py
```
