# Claude Context - Hugging Face Deep RL Course

## Project Overview
This is a repository for exercises from the Hugging Face Deep Reinforcement Learning Course (https://huggingface.co/learn/deep-rl-course).

## Current File Structure
```
├── unit1.py              # PPO Lunar Lander training (50M timesteps, CUDA)
├── unit1_push.py         # Push Unit 1 model to Hugging Face Hub
├── unit2.py              # Unified Q-Learning script with CLI switches
├── gpu_check.py          # GPU validation utilities (standalone + importable)
├── requirements.txt      # All dependencies (with compatibility fixes)
├── CLAUDE.md            # This context file
└── README.md            # User-facing documentation
```

## Git Status
- Current branch: `unit2`
- All major training files completed and functional
- Models pushed to Hugging Face Hub

## Unit 1: PPO Lunar Lander (COMPLETED ✅)
- **Environment**: LunarLander-v2
- **Algorithm**: PPO (Proximal Policy Optimization) via Stable-Baselines3
- **Training**: 50M timesteps with CUDA acceleration
- **Performance**: 284.97 ± 14.71 (excellent score)
- **Hub Model**: `j-klawson/ppo-LunarLander-v2`
- **GPU Requirement**: Mandatory (enforced via gpu_check.py)

## Unit 2: Q-Learning (COMPLETED ✅)
- **Algorithm**: Q-Learning (tabular, from scratch)
- **Implementation**: Unified script with CLI switches
- **Environments**:
  - FrozenLake-v1 (4x4, non-slippery) ✅ Perfect score: 1.00
  - Taxi-v3 ✅ Available via `--taxi` flag
- **Hub Models**:
  - `j-klawson/q-FrozenLake-v1-4x4-noSlippery`
- **GPU Requirement**: None (NumPy-based, CPU only)

## CLI Usage Patterns
```bash
# Unit 1: PPO (GPU required)
python unit1.py                                    # Train agent
python unit1_push.py                              # Push to Hub

# Unit 2: Q-Learning (CPU only)
python unit2.py --frozenlake [--upload]           # Train FrozenLake
python unit2.py --taxi [--upload]                 # Train Taxi
python unit2.py --taxi --upload --username NAME   # Custom username
```

## Technical Architecture

### Unit 1 (PPO)
- **Dependencies**: PyTorch, Stable-Baselines3, CUDA drivers
- **GPU Validation**: Enforced exit if GPU unavailable
- **Training**: Vectorized environments (16 parallel)
- **Hub Integration**: Via `huggingface_sb3.package_to_hub()`

### Unit 2 (Q-Learning)
- **Dependencies**: NumPy, Gymnasium (no PyTorch)
- **Training**: Epsilon-greedy exploration with exponential decay
- **Evaluation**: Deterministic policy with provided seeds
- **Hub Integration**: Custom implementation with model cards

## Development Notes
- **Executable Scripts**: All `.py` files have proper shebangs
- **Virtual Display**: Configured for headless rendering (videos/visualizations)
- **Modular Design**: Unit 2 refactored from separate files into unified CLI
- **Error Handling**: Graceful GPU validation and dependency management
- **Code Quality**: No emojis in output, professional formatting

## Dependency Management
- **Fixed Issues**:
  - `pickle5` → standard `pickle` (Python 3.8+ compatibility)
  - `pyyaml==6.0` → `pyyaml>=6.0` (version flexibility)
  - `pyglet==1.5.1` → separate install from PyPI (index conflict)
- **GPU Libraries**: PyTorch with CUDA 12.1 support
- **RL Libraries**: Stable-Baselines3, Gymnasium, Hugging Face integrations

## Certification Status
- ✅ **Unit 1**: PPO model exceeds performance requirements
- ✅ **Unit 2**: FrozenLake score (1.00) far exceeds requirement (≥4.5)
- 🎯 **Ready for Course Certification**: Both units complete with excellent scores

## Hub Models
- [j-klawson/ppo-LunarLander-v2](https://huggingface.co/j-klawson/ppo-LunarLander-v2) - PPO Lunar Lander
- [j-klawson/q-FrozenLake-v1-4x4-noSlippery](https://huggingface.co/j-klawson/q-FrozenLake-v1-4x4-noSlippery) - Q-Learning FrozenLake

## Next Steps
- Course progression to Unit 3+ as available
- Optional: Train and push Taxi-v3 model for additional practice
- Optional: Experiment with different hyperparameters or environments