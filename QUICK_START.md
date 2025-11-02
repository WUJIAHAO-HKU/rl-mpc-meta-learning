# Quick Start Guide

This guide will help you run your first experiment in 10 minutes.

## 📦 Installation (5 minutes)

### 1. Clone Repository

```bash
git clone https://github.com/WUJIAHAO-HKU/rl-mpc-meta-learning.git
cd rl-mpc-meta-learning
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n rl-mpc python=3.9
conda activate rl-mpc

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

# Or install in development mode (recommended)
pip install -e .
```

## 🚀 Run First Experiment (5 minutes)

### Option A: Use Pre-trained Models (Fastest)

If you've downloaded pre-trained models:

```bash
# 1. Place pre-trained models in models/ directory
# 2. Evaluate directly
python src/evaluation/evaluate_meta_rl.py \
    --robot franka \
    --model models/pretrained_franka.zip \
    --n_episodes 10
```

### Option B: Quick Training Example (approx. 5 minutes)

```bash
# 1. Quick training with small dataset
python src/training/train_meta_pid.py \
    --epochs 10 \
    --batch_size 32 \
    --quick_test

# 2. Check results
ls models/  # Should see meta_pid_quick.pth
```

### Option C: Full Workflow (If You Have Time)

```bash
# Run complete reproduction script (approx. 24 hours)
bash scripts/reproduce_paper_results.sh
```

## 📊 Verify Installation

Run unit tests to ensure everything works:

```bash
python -m pytest tests/ -v
```

## 🎯 Next Steps

### 1. Understand Code Structure

```bash
# View main components
src/
├── networks/          # Neural network definitions
├── environments/      # PyBullet simulation environments
├── training/          # Training scripts
├── evaluation/        # Evaluation scripts
└── visualization/     # Visualization tools
```

### 2. Customize Parameters

Edit configuration files:

```bash
# Edit training config
nano configs/training_config.yaml

# Common modifications:
# - learning_rate: Learning rate
# - batch_size: Batch size
# - n_episodes: Number of training episodes
```

### 3. Visualize Results

```bash
# Generate training curves
python src/visualization/visualize_training_curves.py \
    --log logs/training.log \
    --output results/training_curves.png

# Generate all paper figures
python src/visualization/generate_all_figures_unified.py
```

## 🔧 Troubleshooting

### Q1: PyBullet GUI not showing

```bash
# Install GUI dependencies (Linux)
sudo apt-get install python3-opengl

# Or set in code
p.connect(p.GUI)  # Change to p.connect(p.DIRECT)
```

### Q2: CUDA out of memory

```bash
# Reduce batch size
python train_meta_pid.py --batch_size 16

# Or use CPU
python train_meta_pid.py --device cpu
```

### Q3: Dependency version conflicts

```bash
# Use clean virtual environment
conda create -n rl-mpc-clean python=3.9
conda activate rl-mpc-clean
pip install -r requirements.txt --no-cache-dir
```

## 📚 Learning Resources

1. **Understand Meta-Learning Network**: Read `src/networks/meta_pid_network.py`
2. **Understand RL Policy**: Read `src/networks/rl_policy.py`
3. **Understand Simulation Environment**: Read `src/environments/meta_rl_combined_env.py`
4. **View Full Documentation**: Read [README.md](README.md)

## 🎓 Tutorial Examples

### Example 1: Train Custom Robot

```python
from src.environments.base_env import BaseRobotEnv

# Define your robot
class MyRobotEnv(BaseRobotEnv):
    def __init__(self):
        super().__init__(
            urdf_path="path/to/your/robot.urdf",
            n_joints=7
        )

# Train
# ... (refer to train_meta_rl_combined.py)
```

### Example 2: Custom Reward Function

```python
def custom_reward(error, action, prev_error):
    """Your custom reward function"""
    r_track = -np.linalg.norm(error)
    r_smooth = -0.05 * np.linalg.norm(action)
    r_improve = 5.0 * (np.linalg.norm(prev_error) - np.linalg.norm(error))
    return r_track + r_smooth + r_improve
```

### Example 3: Visualize Robot Motion

```python
import pybullet as p
from src.environments.meta_rl_combined_env import MetaRLCombinedEnv

env = MetaRLCombinedEnv(robot='franka', render=True)
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, done, truncated, info = env.step(action)
    if done:
        break
```

## ✅ Checklist

After completing these steps, you should be able to:

- [ ] Successfully install all dependencies
- [ ] Run test suite without errors
- [ ] Train a small meta-learning model
- [ ] Load and evaluate models
- [ ] Generate basic visualizations

## 💬 Get Help

- **Issue Reporting**: Submit [GitHub Issue](https://github.com/WUJIAHAO-HKU/rl-mpc-meta-learning/issues)
- **Discussion**: Join [Discussions](https://github.com/WUJIAHAO-HKU/rl-mpc-meta-learning/discussions)
- **Email Contact**: u3661739@connect.hku.hk

---

**Good luck with your experiments!** 🎉

If this project helps you, please give us a ⭐️
