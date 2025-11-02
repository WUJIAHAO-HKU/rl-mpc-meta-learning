# 关于.gitkeep文件的说明

## 什么是.gitkeep？

`.gitkeep` 不是Git的官方功能，而是一个约定俗成的做法。

## 为什么需要.gitkeep？

Git不会跟踪空目录。如果一个目录是空的，Git会忽略它。但是在项目结构中，我们需要保留这些目录以供后续使用。

## 项目中的.gitkeep位置

在本项目中，以下目录包含.gitkeep文件：

```
models/.gitkeep      # 存储训练好的模型文件
results/.gitkeep     # 存储实验结果
logs/.gitkeep        # 存储训练日志
configs/.gitkeep     # 配置文件目录（已有training_config.yaml）
tests/.gitkeep       # 单元测试文件
```

## 使用目的

### 1. models/
- **用途**: 存放训练好的模型
- **示例文件**:
  - `meta_pid_augmented.pth` (元学习网络)
  - `franka_rl_policy.zip` (Franka RL策略)
  - `laikago_rl_policy.zip` (Laikago RL策略)
- **说明**: 由于模型文件通常很大（50-200MB），不适合直接提交到Git仓库。建议通过Git LFS或Release上传。

### 2. results/
- **用途**: 存放实验结果和生成的图表
- **示例文件**:
  - `evaluation_results.json`
  - `tracking_error.png`
  - `training_curves.png`
- **说明**: 运行实验后自动生成

### 3. logs/
- **用途**: 存放训练和评估日志
- **示例文件**:
  - `training.log`
  - `evaluation.log`
  - TensorBoard日志
- **说明**: TensorBoard可以读取这些日志进行可视化

### 4. configs/
- **用途**: 配置文件
- **已有文件**: `training_config.yaml`
- **可添加**: 其他特定实验的配置文件

### 5. tests/
- **用途**: 单元测试
- **建议添加**:
  - `test_meta_pid_network.py`
  - `test_rl_policy.py`
  - `test_environments.py`

## 是否应该删除.gitkeep？

**不应该删除！** 原因：

1. **保持项目结构**: 其他人clone项目后会自动有正确的目录结构
2. **避免路径错误**: 代码中的路径引用不会出错
3. **清晰的组织**: 明确表明这些目录的用途
4. **约定俗成**: 这是开源项目的常见做法

## 使用建议

### 上传模型文件

```bash
# 方法1: 使用Git LFS（推荐大文件）
git lfs install
git lfs track "models/*.pth"
git lfs track "models/*.zip"
git add .gitattributes
git add models/your_model.pth
git commit -m "Add trained model"
git push

# 方法2: 通过GitHub Release上传（推荐）
# 在GitHub网页上创建Release，上传模型文件
# 在README中添加下载链接
```

### 生成结果文件

```bash
# 运行实验后，结果会自动保存到results/
python src/training/train_meta_pid.py
# 模型保存到 models/
# 日志保存到 logs/
# 结果保存到 results/
```

### 添加测试

```bash
# 在tests/目录创建测试文件
# 例如: tests/test_meta_pid_network.py

# 运行测试
pytest tests/
```

## .gitignore配置

项目中的`.gitignore`已经配置为忽略这些目录中的内容，但保留.gitkeep：

```gitignore
# 忽略models/中的大文件，但保留.gitkeep
models/*.zip
models/*.pth
!models/.gitkeep

# 同样的逻辑应用到其他目录
results/*
!results/.gitkeep

logs/*
!logs/.gitkeep

tests/*
!tests/.gitkeep
```

## 总结

- ✅ `.gitkeep`文件是**必要的**，用于保留空目录结构
- ✅ 它们很小（空文件），不会增加仓库大小
- ✅ 这是开源项目的**最佳实践**
- ✅ 帮助其他研究者快速理解项目结构
- ✅ 避免路径错误和混乱

**建议**: 保留所有.gitkeep文件，它们有助于项目的清晰性和可维护性。

