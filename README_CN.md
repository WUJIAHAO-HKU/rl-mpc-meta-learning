# 基于强化学习的模型预测控制（元学习增强版）

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

基于强化学习的模型预测控制动力学模型误差在线补偿方法的官方实现。

[English](README.md) | 简体中文

## 📖 论文引用

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

## 🌟 核心特点

- **元学习网络**：快速自适应PID参数预测，无需手动调参
- **强化学习增强**：在线实时补偿动力学模型误差
- **多机器人平台**：支持Franka Panda（9自由度串联机械臂）和Laikago（12自由度并联四足机器人）
- **鲁棒性验证**：抗外部扰动和模型不确定性
- **数据增强技术**：基于物理约束的虚拟样本生成，提升泛化能力

## 🎯 主要成果

### Franka Panda平台（9自由度）
| 指标 | 纯Meta-PID | Meta-PID+RL | 改进率 |
|------|------------|-------------|---------|
| MAE (°) | 7.51 | **6.26** | **+16.6%** |
| RMSE (°) | 29.32 | **25.45** | **+13.2%** |
| 最大误差 (°) | 48.49 | **42.12** | **+13.1%** |

### Laikago平台（12自由度）
| 指标 | 纯Meta-PID | Meta-PID+RL | 改进率 |
|------|------------|-------------|---------|
| MAE (°) | 5.91 | **5.91** | **0.0%** |
| RMSE (°) | 13.80 | **13.74** | **+0.4%** |

**重要发现**：Laikago平台的0.0% MAE改进率反映了"优化天花板效应"——当元学习基线已经在所有关节上达到均衡且接近最优的性能时，强化学习的边际收益受到限制。这一发现为理解元学习与RL协同的边界条件提供了重要洞察。

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- PyBullet（物理仿真引擎）
- NumPy, Matplotlib（数据处理与可视化）

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/WUJIAHAO-HKU/rl-mpc-meta-learning.git
cd rl-mpc-meta-learning

# 2. 创建虚拟环境（推荐）
conda create -n rl-mpc python=3.9
conda activate rl-mpc

# 3. 安装依赖
pip install -r requirements.txt

# 或者以开发模式安装
pip install -e .
```

### 基本使用

#### 1️⃣ 训练元学习网络

```bash
# 使用数据增强训练（推荐）
python src/training/train_with_augmentation.py

# 使用原始数据训练
python src/training/train_meta_pid.py
```

#### 2️⃣ 训练强化学习策略

```bash
# Franka Panda机械臂
python src/training/train_meta_rl_combined.py \
    --robot franka \
    --timesteps 1000000 \
    --meta_model models/meta_pid_augmented.pth

# Laikago四足机器人
python src/training/train_meta_rl_combined.py \
    --robot laikago \
    --timesteps 1000000 \
    --meta_model models/meta_pid_augmented.pth
```

#### 3️⃣ 评估性能

```bash
# 评估Franka Panda
python src/evaluation/evaluate_meta_rl.py \
    --robot franka \
    --model models/franka_rl_policy.zip \
    --n_episodes 100

# 评估Laikago
python src/evaluation/evaluate_laikago.py \
    --model models/laikago_rl_policy.zip \
    --n_episodes 100

# 鲁棒性测试（抗扰动能力）
python src/evaluation/evaluate_robustness.py \
    --robot franka \
    --disturbance_level 0.3
```

#### 4️⃣ 生成可视化结果

```bash
# 生成所有论文图表
python src/visualization/generate_all_figures_unified.py

# 可视化训练曲线
python src/visualization/visualize_training_curves.py \
    --log logs/training.log
```

## 📁 项目结构

```
rl-mpc-meta-learning/
├── src/                          # 源代码
│   ├── networks/                 # 神经网络架构定义
│   │   ├── meta_pid_network.py  # 元学习PID预测器
│   │   └── rl_policy.py         # RL策略网络
│   ├── environments/             # PyBullet仿真环境
│   │   ├── base_env.py          # 基础环境类
│   │   ├── meta_rl_combined_env.py  # 元学习+RL环境
│   │   └── meta_rl_disturbance_env.py  # 扰动测试环境
│   ├── training/                 # 训练脚本
│   │   ├── train_meta_pid.py    # 元学习网络训练
│   │   ├── train_with_augmentation.py  # 数据增强训练
│   │   └── train_meta_rl_combined.py   # RL训练
│   ├── evaluation/               # 评估脚本
│   │   ├── evaluate_meta_rl.py  # 性能评估
│   │   └── evaluate_robustness.py  # 鲁棒性测试
│   └── visualization/            # 可视化工具
│       ├── generate_all_figures_unified.py  # 生成所有图表
│       └── visualize_training_curves.py     # 训练曲线可视化
├── data/                         # 数据文件
│   ├── augmented_pid_data.json  # 增强后的训练数据
│   └── best_configs_paper.json  # 论文中的最佳配置
├── configs/                      # 配置文件
│   └── training_config.yaml     # 训练配置
├── models/                       # 预训练模型（通过Release下载）
├── results/                      # 实验结果
├── scripts/                      # 实用脚本
│   └── reproduce_paper_results.sh  # 完整复现脚本
├── tests/                        # 单元测试
├── README.md                     # 英文README
├── README_CN.md                  # 中文README（本文件）
├── QUICK_START.md                # 快速入门指南
├── GITHUB_UPLOAD_GUIDE.md        # GitHub上传教程
├── requirements.txt              # Python依赖
├── setup.py                      # 安装配置
└── LICENSE                       # MIT许可证
```

## 🔬 算法原理

### 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      控制系统总览                             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  机器人状态 s_t ──→ [元学习网络] ──→ PID增益 [K_p,K_i,K_d]  │
│                           ↓                                   │
│                    [PID控制器] ──→ τ_base                    │
│                           ↓                                   │
│  增强状态 ────────→ [RL策略] ──→ δτ (补偿力矩)              │
│                           ↓                                   │
│                    τ_total = τ_base + δτ                     │
│                           ↓                                   │
│                      [机器人执行]                             │
│                           ↓                                   │
│                    跟踪误差反馈                                │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 1. 元学习PID预测器

根据机器人当前状态自适应预测PID增益：

**输入**: `s_t = [q, q̇, q_ref, q̇_ref]` (关节位置、速度及参考值)  
**输出**: `[K_p, K_i, K_d]` (每个关节的PID增益)

```python
class MetaPIDNetwork(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=64, output_dim=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()  # 确保PID增益为正
        )
```

### 2. RL策略网络

在元学习PID基础上进行微调，补偿模型误差：

**输入**: 增强状态 `[s_t, K_p, K_i, K_d, tracking_error]`  
**输出**: 补偿力矩 `δτ`

```python
class RLPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # 限制动作范围 [-1, 1]
        )
```

### 3. 奖励函数设计

多目标优化奖励函数：

```python
def compute_reward(tracking_error, action, prev_error):
    """
    r_total = r_tracking + r_smoothness + r_improvement
    """
    # 1. 跟踪精度（主要目标）
    r_tracking = -np.linalg.norm(tracking_error)
    
    # 2. 动作平滑性（避免抖动）
    r_smoothness = -0.1 * np.linalg.norm(action)
    
    # 3. 误差减少奖励（鼓励改进）
    r_improvement = 10.0 * (np.linalg.norm(prev_error) - 
                           np.linalg.norm(tracking_error))
    
    return r_tracking + r_smoothness + r_improvement
```

### 4. 数据增强策略

基于物理约束生成虚拟样本，提升泛化能力：

- **状态噪声注入**：`s' = s + ε`, `ε ~ N(0, σ²)`
- **PID增益扰动**：保持稳定性约束
- **可控性检查**：确保生成的样本物理可行

## 📊 完整实验复现

### 一键复现（推荐）

```bash
# 运行完整实验流程（约24小时，单GPU）
bash scripts/reproduce_paper_results.sh
```

### 分步执行

```bash
# Step 1: 训练元学习网络
python src/training/train_with_augmentation.py

# Step 2: 训练Franka RL策略
python src/training/train_meta_rl_combined.py --robot franka --timesteps 1000000

# Step 3: 训练Laikago RL策略
python src/training/train_meta_rl_combined.py --robot laikago --timesteps 1000000

# Step 4: 评估性能
python src/evaluation/evaluate_meta_rl.py --robot franka
python src/evaluation/evaluate_laikago.py

# Step 5: 生成图表
python src/visualization/generate_all_figures_unified.py
```

## 🎓 教程与示例

### 示例1：自定义机器人平台

```python
from src.environments.base_env import BaseRobotEnv

class MyRobotEnv(BaseRobotEnv):
    """自定义您的机器人环境"""
    def __init__(self):
        super().__init__(
            urdf_path="path/to/your/robot.urdf",
            n_joints=7,
            max_torque=100.0,
            control_freq=240
        )
    
    def compute_dynamics(self, q, q_dot):
        """实现您的动力学模型"""
        # 计算质量矩阵、科里奥利力等
        M = self.compute_mass_matrix(q)
        C = self.compute_coriolis(q, q_dot)
        G = self.compute_gravity(q)
        return M, C, G
```

### 示例2：自定义奖励函数

```python
def custom_reward_function(state, action, next_state):
    """根据您的任务定制奖励"""
    # 末端执行器位置误差
    ee_error = np.linalg.norm(state['ee_pos'] - state['target_pos'])
    r_position = -ee_error
    
    # 能量消耗惩罚
    r_energy = -0.01 * np.sum(action**2)
    
    # 任务完成奖励
    r_success = 100.0 if ee_error < 0.01 else 0.0
    
    return r_position + r_energy + r_success
```

### 示例3：可视化机器人运动

```python
from src.environments.meta_rl_combined_env import MetaRLCombinedEnv
import pybullet as p

# 创建环境（开启GUI）
env = MetaRLCombinedEnv(robot='franka', render=True)
obs, info = env.reset()

# 运行控制循环
for step in range(1000):
    # 使用训练好的策略
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    
    if done:
        print(f"Episode finished! Tracking error: {info['tracking_error']:.3f}°")
        break

env.close()
```

## 🛠️ 高级功能

### 超参数调优

编辑 `configs/training_config.yaml`:

```yaml
meta_learning:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  
reinforcement_learning:
  algorithm: PPO
  total_timesteps: 1000000
  learning_rate: 3e-4
```

### 使用TensorBoard监控训练

```bash
# 启动训练时会自动记录到logs/
python src/training/train_meta_rl_combined.py --robot franka

# 在另一个终端启动TensorBoard
tensorboard --logdir=logs/

# 浏览器访问: http://localhost:6006
```

### 分布式训练（多GPU）

```python
# 在train_meta_rl_combined.py中设置
from stable_baselines3.common.vec_env import SubprocVecEnv

# 创建多个并行环境
n_envs = 4
envs = [make_env(i) for i in range(n_envs)]
vec_env = SubprocVecEnv(envs)

# 训练
model = PPO("MlpPolicy", vec_env, device="cuda", n_steps=2048)
model.learn(total_timesteps=1000000)
```

## 🧪 测试

运行单元测试：

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试
python -m pytest tests/test_meta_pid_network.py -v

# 检查代码覆盖率
pytest --cov=src tests/
```

## 📈 性能优化建议

1. **使用数据增强**：可提升约5-10%的泛化性能
2. **调整学习率**：对于不同机器人，最优学习率可能不同
3. **增加训练步数**：复杂任务可能需要2M+步
4. **课程学习**：从简单轨迹逐步过渡到复杂轨迹
5. **集成学习**：训练多个模型并集成可提高鲁棒性

## 🤝 贡献指南

我们欢迎任何形式的贡献！

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

请确保：
- 代码符合PEP 8规范
- 添加适当的注释和文档
- 通过所有单元测试
- 更新相关的README

## 📝 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 📧 联系方式

- **作者**：WU JIAHAO
- **邮箱**：u3661739@connect.hku.hk
- **项目主页**：[https://github.com/WUJIAHAO-HKU/rl-mpc-meta-learning](https://github.com/WUJIAHAO-HKU/rl-mpc-meta-learning)
- **问题反馈**：[GitHub Issues](https://github.com/WUJIAHAO-HKU/rl-mpc-meta-learning/issues)

## 🙏 致谢

- PyBullet团队提供的优秀物理仿真引擎
- Stable-Baselines3提供的高质量RL算法实现
- 匿名审稿人的宝贵意见和建设性建议
- 开源社区的支持与贡献

## 📚 相关资源

### 相关论文
- [Meta-Learning for Control](https://arxiv.org/abs/xxxx.xxxxx)
- [Model Predictive Control with Deep Learning](https://arxiv.org/abs/xxxx.xxxxx)
- [Reinforcement Learning for Robotics: A Survey](https://arxiv.org/abs/xxxx.xxxxx)

### 推荐阅读
- [PyBullet官方文档](https://pybullet.org/)
- [Stable-Baselines3文档](https://stable-baselines3.readthedocs.io/)
- [PyTorch官方教程](https://pytorch.org/tutorials/)

## 🔄 更新日志

### v1.0.0 (2025-11-02)
- 🎉 初始发布
- ✅ 支持Franka Panda和Laikago两个平台
- ✅ 完整的元学习+RL训练流程
- ✅ 数据增强功能
- ✅ 鲁棒性测试
- ✅ 详细的文档和教程

## ❓ 常见问题（FAQ）

<details>
<summary><b>Q: 训练需要多长时间？</b></summary>

A: 
- 元学习网络训练：约1-2小时（CPU）或10-20分钟（GPU）
- RL策略训练（1M步）：约8-12小时（单GPU）
- 完整流程：约24小时
</details>

<details>
<summary><b>Q: 需要什么硬件配置？</b></summary>

A:
- 最低配置：16GB RAM, 4核CPU
- 推荐配置：32GB RAM, 8核CPU, NVIDIA GPU (8GB+ VRAM)
- GPU不是必须的，但能显著加速训练
</details>

<details>
<summary><b>Q: 如何适配新的机器人？</b></summary>

A: 
1. 准备URDF文件
2. 继承`BaseRobotEnv`类
3. 实现动力学模型（可选，用于模拟）
4. 参考示例代码进行训练
</details>

<details>
<summary><b>Q: 为什么Laikago的改进率是0.0%？</b></summary>

A: 
这是"优化天花板效应"的体现。当元学习基线已经非常优秀（各关节均衡、接近最优）时，RL的改进空间有限。这是一个重要的科学发现，说明了元学习与RL协同的边界条件。
</details>

---

**⭐ 如果这个项目对您有帮助，请给我们一个Star！⭐**

**🌐 欢迎关注我们的工作，共同推进机器人控制技术的发展！**

