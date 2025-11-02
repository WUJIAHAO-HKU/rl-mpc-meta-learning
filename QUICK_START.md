# 快速入门指南

本指南将帮助您在10分钟内运行第一个实验。

## 📦 安装（5分钟）

### 1. 克隆仓库

```bash
git clone https://github.com/WUJIAHAO-HKU/rl-mpc-meta-learning.git
cd rl-mpc-meta-learning
```

### 2. 创建虚拟环境（推荐）

```bash
# 使用conda
conda create -n rl-mpc python=3.9
conda activate rl-mpc

# 或使用venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt

# 或作为开发模式安装（推荐）
pip install -e .
```

## 🚀 运行第一个实验（5分钟）

### 选项A：使用预训练模型（最快）

如果您下载了预训练模型：

```bash
# 1. 将预训练模型放到models/目录
# 2. 直接评估
python src/evaluation/evaluate_meta_rl.py \
    --robot franka \
    --model models/pretrained_franka.zip \
    --n_episodes 10
```

### 选项B：快速训练示例（约5分钟）

```bash
# 1. 使用小数据集快速训练元学习网络
python src/training/train_meta_pid.py \
    --epochs 10 \
    --batch_size 32 \
    --quick_test

# 2. 查看结果
ls models/  # 应该看到 meta_pid_quick.pth
```

### 选项C：完整流程（如果有时间）

```bash
# 运行完整复现脚本（约24小时）
bash scripts/reproduce_paper_results.sh
```

## 📊 验证安装

运行单元测试确保一切正常：

```bash
python -m pytest tests/ -v
```

## 🎯 下一步

### 1. 理解代码结构

```bash
# 查看主要组件
src/
├── networks/          # 神经网络定义
├── environments/      # PyBullet仿真环境
├── training/          # 训练脚本
├── evaluation/        # 评估脚本
└── visualization/     # 可视化工具
```

### 2. 自定义参数

编辑配置文件：

```bash
# 编辑训练配置
nano configs/training_config.yaml

# 常见修改：
# - learning_rate: 学习率
# - batch_size: 批量大小
# - n_episodes: 训练轮数
```

### 3. 可视化结果

```bash
# 生成训练曲线
python src/visualization/visualize_training_curves.py \
    --log logs/training.log \
    --output results/training_curves.png

# 生成所有论文图表
python src/visualization/generate_all_figures_unified.py
```

## 🔧 常见问题

### Q1: PyBullet GUI不显示

```bash
# 确保安装了GUI依赖（Linux）
sudo apt-get install python3-opengl

# 或在代码中设置
p.connect(p.GUI)  # 改为 p.connect(p.DIRECT)
```

### Q2: CUDA内存不足

```bash
# 减小批量大小
python train_meta_pid.py --batch_size 16

# 或使用CPU
python train_meta_pid.py --device cpu
```

### Q3: 依赖版本冲突

```bash
# 使用干净的虚拟环境
conda create -n rl-mpc-clean python=3.9
conda activate rl-mpc-clean
pip install -r requirements.txt --no-cache-dir
```

## 📚 学习资源

1. **理解元学习网络**：阅读 `src/networks/meta_pid_network.py`
2. **理解RL策略**：阅读 `src/networks/rl_policy.py`
3. **理解仿真环境**：阅读 `src/environments/meta_rl_combined_env.py`
4. **查看完整文档**：阅读 [README.md](README.md)

## 🎓 教程示例

### 示例1：训练自定义机器人

```python
from src.environments.base_env import BaseRobotEnv

# 定义您的机器人
class MyRobotEnv(BaseRobotEnv):
    def __init__(self):
        super().__init__(
            urdf_path="path/to/your/robot.urdf",
            n_joints=7
        )

# 训练
# ... (参考train_meta_rl_combined.py)
```

### 示例2：自定义奖励函数

```python
def custom_reward(error, action, prev_error):
    """您的自定义奖励"""
    r_track = -np.linalg.norm(error)
    r_smooth = -0.05 * np.linalg.norm(action)
    r_improve = 5.0 * (np.linalg.norm(prev_error) - np.linalg.norm(error))
    return r_track + r_smooth + r_improve
```

### 示例3：可视化机器人运动

```python
import pybullet as p
from src.environments.meta_rl_combined_env import MetaRLCombinedEnv

env = MetaRLCombinedEnv(robot='franka', render=True)
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # 随机动作
    obs, reward, done, truncated, info = env.step(action)
    if done:
        break
```

## ✅ 检查清单

完成以下步骤后，您应该能够：

- [ ] 成功安装所有依赖
- [ ] 运行测试套件无错误
- [ ] 训练一个小型元学习模型
- [ ] 加载和评估模型
- [ ] 生成基本的可视化结果

## 💬 获取帮助

- **问题反馈**：提交 [GitHub Issue](https://github.com/WUJIAHAO-HKU/rl-mpc-meta-learning/issues)
- **讨论交流**：加入 [Discussions](https://github.com/WUJIAHAO-HKU/rl-mpc-meta-learning/discussions)
- **邮件联系**：u3661739@connect.hku.hk

---

**祝您实验顺利！** 🎉

如果本项目对您有帮助，请给我们一个⭐️

