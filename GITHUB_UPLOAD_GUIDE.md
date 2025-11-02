# GitHub上传完整指南

本指南将一步步教您如何将代码上传到GitHub。

## 📋 准备工作

### 1. 注册GitHub账号（如果还没有）

访问 [https://github.com](https://github.com) 并注册账号。

### 2. 安装Git

#### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install git
```

#### Mac:
```bash
brew install git
```

#### Windows:
下载并安装 [Git for Windows](https://git-scm.com/download/win)

### 3. 配置Git

```bash
# 设置您的用户名和邮箱
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 验证配置
git config --list
```

## 🚀 方法一：使用GitHub网页界面（最简单）

### 步骤1: 创建新仓库

1. 登录GitHub
2. 点击右上角的 **+** → **New repository**
3. 填写信息：
   - **Repository name**: `rl-mpc-meta-learning`
   - **Description**: `RL-Enhanced Model Predictive Control with Meta-Learning`
   - **Public/Private**: 选择 Public（公开）
   - ✅ 勾选 **Add a README file**
   - **License**: 选择 MIT License
4. 点击 **Create repository**

### 步骤2: 上传文件

1. 在仓库页面，点击 **Add file** → **Upload files**
2. 将整个 `github_release` 文件夹中的所有文件拖入浏览器
3. 填写提交信息：`Initial commit: RL-MPC project`
4. 点击 **Commit changes**

✅ **完成！** 您的代码已经在GitHub上了。

## 💻 方法二：使用Git命令行（推荐）

### 步骤1: 创建GitHub仓库

同方法一的步骤1，但**不要**勾选 "Add a README file"。

### 步骤2: 初始化本地仓库

```bash
# 进入项目目录
cd /home/wujiahao/基于强化学习的模型预测控制动力学模型误差在线补偿方法研究/rl_pid_linux/meta_learning/github_release

# 初始化Git仓库
git init

# 添加所有文件
git add .

# 查看状态
git status

# 提交
git commit -m "Initial commit: RL-MPC Meta-Learning project"
```

### 步骤3: 连接到GitHub

```bash
# 添加远程仓库（替换成您的GitHub用户名）
git remote add origin https://github.com/YOUR_USERNAME/rl-mpc-meta-learning.git

# 验证远程仓库
git remote -v
```

### 步骤4: 推送到GitHub

```bash
# 推送到主分支
git push -u origin main

# 如果出现错误说分支名是master，使用：
git branch -M main
git push -u origin main
```

### 步骤5: 输入GitHub凭证

首次推送时，会要求输入GitHub用户名和密码。

**注意**: GitHub现在要求使用个人访问令牌（Personal Access Token）而不是密码。

#### 创建Personal Access Token:

1. 登录GitHub
2. 点击右上角头像 → **Settings**
3. 左侧菜单 → **Developer settings** → **Personal access tokens** → **Tokens (classic)**
4. 点击 **Generate new token** → **Generate new token (classic)**
5. 填写信息：
   - **Note**: `RL-MPC Project`
   - **Expiration**: 选择过期时间（建议90天或自定义）
   - **Scopes**: 勾选 `repo` (所有子选项)
6. 点击 **Generate token**
7. **重要**: 复制生成的token（只显示一次！）

#### 使用Token推送:

```bash
git push -u origin main
# 用户名: YOUR_GITHUB_USERNAME
# 密码: 粘贴刚才复制的token
```

✅ **完成！** 代码已上传到GitHub。

## 🔒 方法三：使用SSH（最安全）

### 步骤1: 生成SSH密钥

```bash
# 生成密钥对
ssh-keygen -t ed25519 -C "your.email@example.com"

# 按提示操作（通常直接按Enter使用默认设置）

# 启动SSH代理
eval "$(ssh-agent -s)"

# 添加私钥
ssh-add ~/.ssh/id_ed25519
```

### 步骤2: 添加SSH公钥到GitHub

```bash
# 复制公钥内容
cat ~/.ssh/id_ed25519.pub
# 或在Linux上直接复制到剪贴板
cat ~/.ssh/id_ed25519.pub | xclip -selection clipboard
```

然后：
1. 登录GitHub
2. 点击头像 → **Settings** → **SSH and GPG keys**
3. 点击 **New SSH key**
4. 粘贴公钥内容
5. 点击 **Add SSH key**

### 步骤3: 使用SSH推送

```bash
# 在github_release目录中
cd /home/wujiahao/基于强化学习的模型预测控制动力学模型误差在线补偿方法研究/rl_pid_linux/meta_learning/github_release

git init
git add .
git commit -m "Initial commit"

# 使用SSH URL（注意是git@而不是https://）
git remote add origin git@github.com:YOUR_USERNAME/rl-mpc-meta-learning.git

git branch -M main
git push -u origin main
```

✅ **完成！** 使用SSH更安全，不需要每次输入密码。

## 📝 后续更新代码

当您修改代码后，可以这样更新GitHub：

```bash
# 查看修改
git status

# 添加修改的文件
git add .

# 或添加特定文件
git add src/training/train_meta_pid.py

# 提交修改
git commit -m "描述您的修改内容"

# 推送到GitHub
git push
```

### 常用Git命令

```bash
# 查看提交历史
git log --oneline

# 查看当前状态
git status

# 查看文件差异
git diff

# 创建新分支
git checkout -b feature/new-feature

# 切换分支
git checkout main

# 合并分支
git merge feature/new-feature

# 拉取最新代码
git pull

# 克隆仓库（其他人下载）
git clone https://github.com/YOUR_USERNAME/rl-mpc-meta-learning.git
```

## 🎨 美化您的GitHub仓库

### 1. 添加徽章（Badges）

在README.md顶部已经包含了常用徽章：
- License徽章
- Python版本徽章
- PyTorch徽章

### 2. 添加Topics标签

1. 在仓库页面，点击右侧 **About** 旁的设置图标
2. 在 **Topics** 中添加：
   - `reinforcement-learning`
   - `model-predictive-control`
   - `meta-learning`
   - `robotics`
   - `pytorch`
   - `pybullet`

### 3. 添加描述和网站

在同一个设置中：
- **Description**: `RL-Enhanced MPC with Meta-Learning for Robot Control`
- **Website**: 您的个人主页或论文链接

### 4. 启用GitHub Pages（可选）

1. 仓库 → **Settings** → **Pages**
2. **Source**: 选择 `main` 分支
3. 保存后，您的文档将发布到：
   `https://YOUR_USERNAME.github.io/rl-mpc-meta-learning/`

## 📦 发布Release版本

### 方法1: 网页界面

1. 仓库页面右侧 → **Releases** → **Create a new release**
2. 填写：
   - **Tag version**: `v1.0.0`
   - **Release title**: `v1.0.0 - Initial Release`
   - **Description**: 描述这个版本的主要特性
3. 可以上传预训练模型等大文件
4. 点击 **Publish release**

### 方法2: 命令行

```bash
# 创建并推送标签
git tag -a v1.0.0 -m "Version 1.0.0 - Initial release"
git push origin v1.0.0

# 然后在GitHub网页上创建Release
```

## 🔍 验证上传成功

访问您的仓库：
```
https://github.com/YOUR_USERNAME/rl-mpc-meta-learning
```

应该看到：
- ✅ README.md 显示在首页
- ✅ 所有文件和文件夹
- ✅ 代码可以正常浏览
- ✅ LICENSE 文件存在

## 🐛 常见问题

### Q1: 推送时提示"Permission denied"

**解决**：
```bash
# 检查远程URL
git remote -v

# 如果是https，确保使用了正确的token
# 如果是SSH，确保SSH密钥已正确配置
ssh -T git@github.com
```

### Q2: 文件太大无法推送

**解决**：
```bash
# GitHub单个文件限制100MB
# 对于大文件（如模型），使用Git LFS

# 安装Git LFS
git lfs install

# 跟踪大文件
git lfs track "*.pth"
git lfs track "*.zip"

# 添加.gitattributes
git add .gitattributes

# 正常提交和推送
git add .
git commit -m "Add large files with LFS"
git push
```

### Q3: 忘记添加.gitignore

**解决**：
```bash
# 创建.gitignore（已包含在项目中）
# 删除已跟踪的不需要的文件
git rm -r --cached __pycache__
git rm -r --cached *.pyc

# 提交
git commit -m "Remove cached files"
git push
```

### Q4: 合并冲突

**解决**：
```bash
# 拉取最新代码
git pull

# 如果有冲突，手动编辑冲突文件
# 搜索 <<<<<<< HEAD 标记

# 解决后
git add <resolved-file>
git commit -m "Resolve merge conflict"
git push
```

## 📧 更新论文中的GitHub链接

代码上传后，记得更新论文中的GitHub链接：

```latex
% 在论文的"Code Availability"部分
The source code and trained models are publicly available at:
\url{https://github.com/YOUR_USERNAME/rl-mpc-meta-learning}
```

## ✅ 完成检查清单

- [ ] GitHub账号已创建
- [ ] Git已安装并配置
- [ ] 仓库已创建
- [ ] 所有代码已推送
- [ ] README.md 正常显示
- [ ] LICENSE 文件存在
- [ ] .gitignore 正确配置
- [ ] Topics标签已添加
- [ ] Release版本已创建（可选）
- [ ] 论文中的GitHub链接已更新

## 🎓 Git学习资源

- [Git官方教程](https://git-scm.com/book/zh/v2)
- [GitHub官方文档](https://docs.github.com/cn)
- [交互式Git教程](https://learngitbranching.js.org/?locale=zh_CN)
- [Git Cheat Sheet](https://training.github.com/downloads/zh_CN/github-git-cheat-sheet/)

---

**恭喜！🎉 您已成功将项目上传到GitHub！**

如有任何问题，欢迎提Issue或发邮件联系。

