# RL-Enhanced Model Predictive Control with Meta-Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

基于强化学习的模型预测控制动力学模型误差在线补偿方法的官方实现。

## 📖 引用

如果您在研究中使用了本代码，请引用我们的论文：

```bibtex
@article{wu2025rl,
  title={Reinforcement Learning-Enhanced Model Predictive Control with Meta-Learning for Online Compensation of Dynamic Model Errors},
  author={Wu, Jiahao and others},
  journal={To be published},
  year={2025},
  note={Manuscript in preparation}
}
```

## 🌟 主要特点

- **元学习网络**：快速自适应PID参数预测
- **强化学习增强**：在线补偿动力学模型误差
- **多机器人平台**：支持Franka Panda（9-DOF串联）和Laikago（12-DOF并联四足）
- **鲁棒性验证**：抗外部扰动和模型不确定性
- **数据增强**：基于物理约束的虚拟样本生成

## 💬 Language Notes

- **Documentation**: Full English and Chinese documentation provided
- **Code Comments**: Core modules (`src/networks/`, `src/environments/base_env.py`) have complete English docstrings and comments
- **User-facing APIs**: All public functions and classes have English documentation
- **Some files**: May contain Chinese comments (legacy from development)
- **Contributions Welcome**: We welcome pull requests to improve internationalization

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- PyBullet
- NumPy, Matplotlib

### 安装

```bash
# 克隆仓库
git clone https://github.com/WUJIAHAO-HKU/rl-mpc-meta-learning.git
cd rl-mpc-meta-learning

# 安装依赖
pip install -r requirements.txt
```

### 基础使用

#### 1. 训练元学习网络

```bash
# 使用数据增强训练
python train_with_augmentation.py

# 或使用原始数据训练
python train_meta_pid.py
```

#### 2. 训练RL策略

```bash
# Franka Panda平台
python train_meta_rl_combined.py --robot franka --timesteps 1000000

# Laikago平台
python train_meta_rl_combined.py --robot laikago --timesteps 1000000
```

#### 3. 评估性能

```bash
# 评估Franka Panda
python evaluate_meta_rl.py --robot franka --model best_franka_model.zip

# 评估Laikago
python evaluate_laikago.py --model best_laikago_model.zip

# 鲁棒性测试
python evaluate_robustness.py --robot franka --disturbance_level 0.3
```

#### 4. 生成可视化结果

```bash
# 生成所有论文图表
python generate_all_figures_unified.py

# 可视化训练曲线
python visualize_training_curves.py --log training_log.txt
```

## 📊 实验结果

### Franka Panda (9-DOF)
| 指标 | Meta-PID | Meta-PID+RL | 改进率 |
|------|----------|-------------|--------|
| MAE (°) | 7.51 | **6.26** | +16.6% |
| RMSE (°) | 29.32 | **25.45** | +13.2% |

### Laikago (12-DOF)
| 指标 | Meta-PID | Meta-PID+RL | 改进率 |
|------|----------|-------------|--------|
| MAE (°) | 5.91 | **5.91** | 0.0% |
| RMSE (°) | 13.80 | **13.74** | +0.4% |

*注：Laikago平台的0.0% MAE改进率反映了"优化天花板效应"——当元学习基线已达到近乎最优时，RL的边际收益受限。*

## 📁 项目结构

```
rl-mpc-meta-learning/
├── src/
│   ├── networks/          # 神经网络架构
│   ├── environments/      # PyBullet仿真环境
│   ├── training/          # 训练脚本
│   ├── evaluation/        # 评估脚本
│   └── visualization/     # 可视化工具
├── data/
│   ├── augmented_pid_data.json      # 增强训练数据
│   └── best_configs_paper.json      # 最佳配置
├── configs/               # 配置文件
├── models/                # 预训练模型
├── results/               # 实验结果
├── tests/                 # 单元测试
└── README.md
```

## 🔬 核心算法

### 1. 元学习网络架构

```python
class MetaPIDNetwork(nn.Module):
    """
    输入: 机器人状态 s_t = [q, q_dot, q_ref, q_ref_dot]
    输出: PID增益 [K_p, K_i, K_d] (每个关节)
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
            nn.Softplus()  # 确保PID增益为正
        )
```

### 2. RL策略网络

```python
class RLPolicy(nn.Module):
    """
    输入: 增强状态 [s_t, K_p, K_i, K_d, tracking_error]
    输出: 补偿力矩 δτ
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # 限制动作范围
        )
```

### 3. 奖励函数设计

```python
def compute_reward(tracking_error, action, prev_error):
    """
    多目标奖励函数
    """
    # 跟踪误差惩罚
    r_tracking = -np.linalg.norm(tracking_error)
    
    # 动作平滑性惩罚
    r_smoothness = -0.1 * np.linalg.norm(action)
    
    # 误差减少奖励
    r_improvement = 10.0 * (np.linalg.norm(prev_error) - 
                           np.linalg.norm(tracking_error))
    
    return r_tracking + r_smoothness + r_improvement
```

## 🛠️ 高级用法

### 自定义机器人平台

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
        # 实现您的动力学模型
        pass
```

### 数据增强配置

编辑 `configs/augmentation_config.yaml`:

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

## 📈 训练技巧

1. **元学习网络预训练**：使用大量离线数据预训练以获得良好初始化
2. **分阶段训练**：先训练元学习网络，再训练RL策略
3. **课程学习**：从简单轨迹逐步过渡到复杂轨迹
4. **超参数调优**：使用我们提供的最佳配置 `best_configs_paper.json`

## 🧪 复现论文结果

```bash
# 完整流程（约需24小时，单GPU）
bash scripts/reproduce_paper_results.sh

# 或分步执行：
# Step 1: 训练元学习网络
python train_with_augmentation.py

# Step 2: 训练Franka RL策略
python train_meta_rl_combined.py --robot franka --timesteps 1000000

# Step 3: 训练Laikago RL策略
python train_meta_rl_combined.py --robot laikago --timesteps 1000000

# Step 4: 评估并生成结果
python evaluate_meta_rl.py --robot franka
python evaluate_laikago.py
python generate_all_figures_unified.py
```

## 🤝 贡献指南

我们欢迎任何形式的贡献！请遵循以下步骤：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📝 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 📧 联系方式

- 作者：[WU JIAHAO]
- 邮箱：[u3661739@connect.hku.hk]
- 项目主页：[https://github.com/WUJIAHAO-HKU/rl-mpc-meta-learning](https://github.com/WUJIAHAO-HKU/rl-mpc-meta-learning)

## 🙏 致谢

- PyBullet团队提供的优秀物理仿真引擎
- Stable-Baselines3提供的RL算法实现
- 匿名审稿人的宝贵意见

## 📚 相关工作

如果您对本研究感兴趣，可能也会对以下工作感兴趣：

- [Meta-Learning for Control](https://arxiv.org/abs/xxxx.xxxxx)
- [Model Predictive Control with Neural Networks](https://arxiv.org/abs/xxxx.xxxxx)
- [Reinforcement Learning for Robotics](https://arxiv.org/abs/xxxx.xxxxx)

---

**⭐ 如果这个项目对您有帮助，请给我们一个Star！⭐**

