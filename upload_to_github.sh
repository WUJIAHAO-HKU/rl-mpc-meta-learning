#!/bin/bash
# 一键上传到GitHub脚本
# 使用前请先确保：
# 1. 已在GitHub上创建了仓库
# 2. 已配置好Git用户信息
# 3. 已设置好SSH密钥或Personal Access Token

set -e

echo "============================================"
echo "  GitHub自动上传脚本"
echo "============================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否已初始化git
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}[步骤 1/6]${NC} 初始化Git仓库..."
    git init
    echo -e "${GREEN}✓${NC} Git仓库已初始化"
else
    echo -e "${GREEN}✓${NC} Git仓库已存在"
fi

# 检查Git配置
echo ""
echo -e "${YELLOW}[步骤 2/6]${NC} 检查Git配置..."
if ! git config user.name > /dev/null 2>&1; then
    echo -e "${RED}✗${NC} Git用户名未配置"
    read -p "请输入您的GitHub用户名: " username
    git config user.name "$username"
fi
if ! git config user.email > /dev/null 2>&1; then
    echo -e "${RED}✗${NC} Git邮箱未配置"
    read -p "请输入您的邮箱: " email
    git config user.email "$email"
fi
echo -e "${GREEN}✓${NC} Git配置完成"
echo "  用户名: $(git config user.name)"
echo "  邮箱: $(git config user.email)"

# 获取GitHub仓库URL
echo ""
echo -e "${YELLOW}[步骤 3/6]${NC} 配置远程仓库..."
if git remote get-url origin > /dev/null 2>&1; then
    origin_url=$(git remote get-url origin)
    echo -e "${GREEN}✓${NC} 远程仓库已配置: $origin_url"
    read -p "是否需要更改？(y/N): " change_remote
    if [[ $change_remote =~ ^[Yy]$ ]]; then
        read -p "请输入新的GitHub仓库URL: " new_url
        git remote set-url origin "$new_url"
        echo -e "${GREEN}✓${NC} 远程仓库已更新"
    fi
else
    echo ""
    echo "请选择连接方式："
    echo "  1) HTTPS (需要Personal Access Token)"
    echo "  2) SSH (需要SSH密钥)"
    read -p "选择 (1/2): " connection_type
    
    read -p "请输入您的GitHub用户名: " github_username
    read -p "请输入仓库名 (默认: rl-mpc-meta-learning): " repo_name
    repo_name=${repo_name:-rl-mpc-meta-learning}
    
    if [ "$connection_type" = "1" ]; then
        remote_url="https://github.com/$github_username/$repo_name.git"
    else
        remote_url="git@github.com:$github_username/$repo_name.git"
    fi
    
    git remote add origin "$remote_url"
    echo -e "${GREEN}✓${NC} 远程仓库已配置: $remote_url"
fi

# 添加文件
echo ""
echo -e "${YELLOW}[步骤 4/6]${NC} 添加文件到Git..."
git add .
echo -e "${GREEN}✓${NC} 文件已添加"

# 查看状态
echo ""
echo "即将提交的文件："
git status --short | head -20
file_count=$(git status --short | wc -l)
if [ $file_count -gt 20 ]; then
    echo "... 以及其他 $((file_count - 20)) 个文件"
fi

# 提交
echo ""
echo -e "${YELLOW}[步骤 5/6]${NC} 提交更改..."
read -p "请输入提交信息 (默认: Initial commit): " commit_message
commit_message=${commit_message:-Initial commit: RL-MPC Meta-Learning project}
git commit -m "$commit_message"
echo -e "${GREEN}✓${NC} 更改已提交"

# 推送到GitHub
echo ""
echo -e "${YELLOW}[步骤 6/6]${NC} 推送到GitHub..."
echo ""
echo -e "${YELLOW}注意${NC}: 如果使用HTTPS，密码处请输入Personal Access Token，而不是GitHub密码"
echo ""

# 确保使用main分支
current_branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$current_branch" != "main" ]; then
    git branch -M main
    echo "已切换到main分支"
fi

# 推送
if git push -u origin main; then
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}  ✓ 成功上传到GitHub！${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    
    # 提取仓库URL用于显示
    origin_url=$(git remote get-url origin)
    if [[ $origin_url == git@github.com:* ]]; then
        # SSH格式转换为HTTPS显示
        repo_path=$(echo $origin_url | sed 's/git@github.com://' | sed 's/.git$//')
        display_url="https://github.com/$repo_path"
    else
        # HTTPS格式直接去掉.git
        display_url=$(echo $origin_url | sed 's/.git$//')
    fi
    
    echo "您的仓库地址："
    echo -e "  ${GREEN}$display_url${NC}"
    echo ""
    echo "下一步建议："
    echo "  1. 访问仓库页面，检查文件是否正确上传"
    echo "  2. 编辑仓库描述和Topics标签"
    echo "  3. 启用GitHub Pages（如果需要）"
    echo "  4. 创建Release版本"
    echo "  5. 更新论文中的GitHub链接"
    echo ""
else
    echo ""
    echo -e "${RED}============================================${NC}"
    echo -e "${RED}  ✗ 上传失败${NC}"
    echo -e "${RED}============================================${NC}"
    echo ""
    echo "可能的原因："
    echo "  1. 网络连接问题"
    echo "  2. 认证失败（检查Token或SSH密钥）"
    echo "  3. 仓库不存在（请先在GitHub上创建）"
    echo "  4. 权限问题"
    echo ""
    echo "解决方法："
    echo "  1. 检查网络连接"
    echo "  2. 如果使用HTTPS，确保使用Personal Access Token"
    echo "  3. 如果使用SSH，运行: ssh -T git@github.com 测试连接"
    echo "  4. 查看详细的Git错误信息"
    echo ""
    echo "需要帮助？请参考: GITHUB_UPLOAD_GUIDE.md"
    exit 1
fi

# 可选：创建.gitignore（如果还没有）
if [ ! -f ".gitignore" ]; then
    echo ""
    read -p "是否创建.gitignore文件？(Y/n): " create_gitignore
    if [[ ! $create_gitignore =~ ^[Nn]$ ]]; then
        # .gitignore已经在项目中了，这里只是提示
        echo -e "${GREEN}✓${NC} .gitignore文件已存在"
    fi
fi

echo ""
echo "============================================"
echo "  祝您研究顺利！🎉"
echo "============================================"

