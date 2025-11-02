# RL-Enhanced Model Predictive Control with Meta-Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Official implementation of "Reinforcement Learning-Enhanced Model Predictive Control with Meta-Learning for Online Compensation of Dynamic Model Errors".

## 📖 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{wu2025rl,
  title={Reinforcement Learning-Enhanced Model Predictive Control with Meta-Learning for Online Compensation of Dynamic Model Errors},
  author={Wu, Jiahao and others},
  journal={To be published},
  year={2025},
  note={Manuscript in preparation}
}
```

## 🌟 Key Features

- **Meta-Learning Network**: Fast adaptive PID parameter prediction
- **Reinforcement Learning Enhancement**: Online compensation of dynamic model errors
- **Multi-Robot Platforms**: Supports Franka Panda (9-DOF serial) and Laikago (12-DOF quadruped)
- **Robustness Validation**: Resistant to external disturbances and model uncertainties
- **Data Augmentation**: Physics-constrained virtual sample generation

## 💬 Language Notes

- **Documentation**: Full English and Chinese documentation provided
- **Code Comments**: Core modules (`src/networks/`, `src/environments/base_env.py`) have complete English docstrings and comments
- **User-facing APIs**: All public functions and classes have English documentation
- **Some files**: May contain Chinese comments (legacy from development)
- **Contributions Welcome**: We welcome pull requests to improve internationalization

## 🚀 Quick Start

### Requirements

- Python 3.8+
- PyTorch 2.0+
- PyBullet
- NumPy, Matplotlib

### Installation

```bash
# Clone the repository
git clone https://github.com/WUJIAHAO-HKU/rl-mpc-meta-learning.git
cd rl-mpc-meta-learning

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### 1. Train Meta-Learning Network

```bash
# Train with data augmentation
python train_with_augmentation.py

# Or train with original data
python train_meta_pid.py
```

#### 2. Train RL Policy

```bash
# Franka Panda platform
python train_meta_rl_combined.py --robot franka --timesteps 1000000

# Laikago platform
python train_meta_rl_combined.py --robot laikago --timesteps 1000000
```

#### 3. Evaluate Performance

```bash
# Evaluate Franka Panda
python evaluate_meta_rl.py --robot franka --model best_franka_model.zip

# Evaluate Laikago
python evaluate_laikago.py --model best_laikago_model.zip

# Robustness testing
python evaluate_robustness.py --robot franka --disturbance_level 0.3
```

#### 4. Generate Visualizations

```bash
# Generate all paper figures
python generate_all_figures_unified.py

# Visualize training curves
python visualize_training_curves.py --log training_log.txt
```

## 📊 Experimental Results

### Franka Panda (9-DOF)
| Metric | Meta-PID | Meta-PID+RL | Improvement |
|------|----------|-------------|--------|
| MAE (°) | 7.51 | **6.26** | +16.6% |
| RMSE (°) | 29.32 | **25.45** | +13.2% |

### Laikago (12-DOF)
| Metric | Meta-PID | Meta-PID+RL | Improvement |
|------|----------|-------------|--------|
| MAE (°) | 5.91 | **5.91** | 0.0% |
| RMSE (°) | 29.70 | **29.29** | +1.4% |

*Note: The 0.0% MAE improvement for Laikago reflects the "optimization ceiling effect" — when the meta-learning baseline is already near-optimal, RL's marginal gains are limited.*

## 📁 Project Structure

```
rl-mpc-meta-learning/
├── src/
│   ├── networks/          # Neural network architectures
│   ├── environments/      # PyBullet simulation environments
│   ├── training/          # Training scripts
│   ├── evaluation/        # Evaluation scripts
│   └── visualization/     # Visualization tools
├── data/
│   ├── augmented_pid_data.json      # Augmented training data
│   └── best_configs_paper.json      # Best configurations
├── configs/               # Configuration files
├── models/                # Pre-trained models
├── results/               # Experimental results
├── tests/                 # Unit tests
└── README.md
```

## 🔬 Core Algorithms

### 1. Meta-Learning Network Architecture

```python
class MetaPIDNetwork(nn.Module):
    """
    Input: Robot state s_t = [q, q_dot, q_ref, q_ref_dot]
    Output: PID gains [K_p, K_i, K_d] (per joint)
    """
    def __init__(self, state_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.pid_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softplus()  # Ensure positive PID gains
        )
```

### 2. RL Policy Network

```python
class RLPolicy(nn.Module):
    """
    Input: Augmented state [s_t, K_p, K_i, K_d, tracking_error]
    Output: Compensation torque δτ
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # Limit action range
        )
```

### 3. Reward Function Design

```python
def compute_reward(tracking_error, action, prev_error):
    """
    Multi-objective reward function
    """
    # Tracking error penalty
    r_tracking = -np.linalg.norm(tracking_error)
    
    # Action smoothness penalty
    r_smoothness = -0.1 * np.linalg.norm(action)
    
    # Error reduction reward
    r_improvement = 10.0 * (np.linalg.norm(prev_error) - 
                           np.linalg.norm(tracking_error))
    
    return r_tracking + r_smoothness + r_improvement
```

## 🛠️ Advanced Usage

### Custom Robot Platform

```python
from src.environments.base_env import BaseRobotEnv

class CustomRobotEnv(BaseRobotEnv):
    def __init__(self):
        super().__init__(
            urdf_path="path/to/robot.urdf",
            n_joints=7,
            max_torque=100.0
        )
    
    def compute_dynamics(self, q, q_dot):
        # Implement your dynamics model
        pass
```

### Data Augmentation Configuration

Edit `configs/augmentation_config.yaml`:

```yaml
augmentation:
  enabled: true
  samples_per_real: 10
  noise_levels:
    state: 0.01
    pid_gains: 0.05
  physics_constraints:
    enforce_stability: true
    check_controllability: true
```

## 📈 Training Tips

1. **Meta-Learning Pre-training**: Use abundant offline data for good initialization
2. **Staged Training**: Train meta-learning network first, then RL policy
3. **Curriculum Learning**: Gradually transition from simple to complex trajectories
4. **Hyperparameter Tuning**: Use our provided best configurations in `best_configs_paper.json`

## 🧪 Reproduce Paper Results

```bash
# Complete workflow (approx. 24 hours on single GPU)
bash scripts/reproduce_paper_results.sh

# Or step-by-step execution:
# Step 1: Train meta-learning network
python train_with_augmentation.py

# Step 2: Train Franka RL policy
python train_meta_rl_combined.py --robot franka --timesteps 1000000

# Step 3: Train Laikago RL policy
python train_meta_rl_combined.py --robot laikago --timesteps 1000000

# Step 4: Evaluate and generate results
python evaluate_meta_rl.py --robot franka
python evaluate_laikago.py
python generate_all_figures_unified.py
```

## 🤝 Contributing

We welcome contributions of all kinds! Please follow these steps:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 📧 Contact

- Author: WU JIAHAO
- Email: u3661739@connect.hku.hk
- Project Homepage: [https://github.com/WUJIAHAO-HKU/rl-mpc-meta-learning](https://github.com/WUJIAHAO-HKU/rl-mpc-meta-learning)

## 🙏 Acknowledgments

- PyBullet team for the excellent physics simulation engine
- Stable-Baselines3 for RL algorithm implementations
- Anonymous reviewers for their valuable feedback

## 📚 Related Work

If you're interested in this research, you might also find these works interesting:

- [Meta-Learning for Control](https://arxiv.org/abs/xxxx.xxxxx)
- [Model Predictive Control with Neural Networks](https://arxiv.org/abs/xxxx.xxxxx)
- [Reinforcement Learning for Robotics](https://arxiv.org/abs/xxxx.xxxxx)

---

**⭐ If this project helps you, please give us a Star! ⭐**

