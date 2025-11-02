"""
元学习PID模型训练脚本

功能：
1. 加载训练数据集
2. 训练元学习网络
3. 评估零样本泛化能力
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from meta_learning.meta_pid_optimizer import MetaPIDOptimizer, MetaPIDNetwork, RobotFeatureExtractor


class PIDDataset(Dataset):
    """PID参数数据集"""
    
    def __init__(self, data_points, feature_extractor, normalization_stats=None):
        """
        Args:
            data_points: list of dict, 每个包含features和optimal_pid
            feature_extractor: RobotFeatureExtractor实例
            normalization_stats: 归一化统计量（可选）
        """
        self.data_points = data_points
        self.feature_extractor = feature_extractor
        
        # 提取所有特征和标签
        self.features = []
        self.labels = []
        self.dofs = []
        
        for dp in data_points:
            # 特征
            feature_dict = dp['features']
            feature_vec = np.array([feature_dict[name] 
                                   for name in feature_extractor.feature_names], 
                                  dtype=np.float32)
            
            # 标签（PID参数）
            pid = dp['optimal_pid']
            kp = np.array(pid['Kp'], dtype=np.float32)
            ki = np.array(pid['Ki'], dtype=np.float32)
            kd = np.array(pid['Kd'], dtype=np.float32)
            
            # Pad到最大DOF（7）
            max_dof = 7
            actual_dof = len(kp)
            
            kp_padded = np.zeros(max_dof, dtype=np.float32)
            ki_padded = np.zeros(max_dof, dtype=np.float32)
            kd_padded = np.zeros(max_dof, dtype=np.float32)
            
            kp_padded[:actual_dof] = kp
            ki_padded[:actual_dof] = ki
            kd_padded[:actual_dof] = kd
            
            self.features.append(feature_vec)
            self.labels.append(np.stack([kp_padded, ki_padded, kd_padded]))  # (3, max_dof)
            self.dofs.append(actual_dof)
        
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        
        # 归一化特征
        if normalization_stats is None:
            self.mean = np.mean(self.features, axis=0)
            self.std = np.std(self.features, axis=0) + 1e-8
        else:
            self.mean = normalization_stats['mean']
            self.std = normalization_stats['std']
        
        self.features = (self.features - self.mean) / self.std
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx]),
            torch.FloatTensor(self.labels[idx]),
            self.dofs[idx]
        )
    
    def get_normalization_stats(self):
        return {'mean': self.mean, 'std': self.std}


def weighted_mse_loss(pred, target, dof, max_dof=7):
    """
    加权MSE损失（只计算实际DOF的损失）
    
    Args:
        pred: (batch, 3, max_dof) 预测的PID参数
        target: (batch, 3, max_dof) 目标PID参数
        dof: (batch,) 实际自由度
        max_dof: 最大自由度
    
    Returns:
        loss: 标量损失
    """
    batch_size = pred.shape[0]
    
    # 创建mask
    mask = torch.zeros_like(pred)
    for i, d in enumerate(dof):
        mask[i, :, :d] = 1.0
    
    # 加权MSE
    squared_diff = (pred - target) ** 2
    masked_loss = squared_diff * mask
    
    # 归一化（除以实际元素数量）
    num_elements = mask.sum()
    loss = masked_loss.sum() / (num_elements + 1e-8)
    
    return loss


def relative_error_loss(pred, target, dof):
    """
    相对误差损失（百分比误差）
    
    Args:
        pred: (batch, 3, max_dof)
        target: (batch, 3, max_dof)
        dof: (batch,)
    
    Returns:
        loss: 标量损失
    """
    batch_size = pred.shape[0]
    
    # 创建mask
    mask = torch.zeros_like(pred)
    for i, d in enumerate(dof):
        mask[i, :, :d] = 1.0
    
    # 相对误差: |pred - target| / (target + eps)
    relative_error = torch.abs(pred - target) / (torch.abs(target) + 1e-2)
    masked_error = relative_error * mask
    
    # 平均
    num_elements = mask.sum()
    loss = masked_error.sum() / (num_elements + 1e-8)
    
    return loss


def train_epoch(model, dataloader, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_mse = 0
    total_rel_error = 0
    
    for features, labels, dofs in dataloader:
        features = features.to(device)
        labels = labels.to(device)  # (batch, 3, max_dof)
        
        # 前向传播
        kp_pred, ki_pred, kd_pred = model(features)
        pred = torch.stack([kp_pred, ki_pred, kd_pred], dim=1)  # (batch, 3, max_dof)
        
        # 计算损失
        mse_loss = weighted_mse_loss(pred, labels, dofs)
        rel_loss = relative_error_loss(pred, labels, dofs)
        
        # 组合损失
        loss = mse_loss + 0.1 * rel_loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_rel_error += rel_loss.item()
    
    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'mse': total_mse / num_batches,
        'rel_error': total_rel_error / num_batches
    }


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_mse = 0
    total_rel_error = 0
    
    with torch.no_grad():
        for features, labels, dofs in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            
            kp_pred, ki_pred, kd_pred = model(features)
            pred = torch.stack([kp_pred, ki_pred, kd_pred], dim=1)
            
            mse_loss = weighted_mse_loss(pred, labels, dofs)
            rel_loss = relative_error_loss(pred, labels, dofs)
            loss = mse_loss + 0.1 * rel_loss
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_rel_error += rel_loss.item()
    
    num_batches = len(dataloader)
    return {
        'loss': total_loss / num_batches,
        'mse': total_mse / num_batches,
        'rel_error': total_rel_error / num_batches
    }


def plot_training_curves(train_history, val_history, output_path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics = ['loss', 'mse', 'rel_error']
    titles = ['Total Loss', 'MSE Loss', 'Relative Error']
    
    for ax, metric, title in zip(axes, metrics, titles):
        train_values = [h[metric] for h in train_history]
        val_values = [h[metric] for h in val_history]
        
        ax.plot(train_values, label='Train', linewidth=2)
        ax.plot(val_values, label='Val', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 训练曲线已保存: {output_path}")


def main():
    """主训练流程"""
    print("=" * 80)
    print("元学习PID模型训练")
    print("=" * 80)
    
    # 配置
    data_path = Path('meta_learning/training_data')
    dataset_file = list(data_path.glob('pid_dataset_*.json'))
    
    if not dataset_file:
        print("❌ 未找到训练数据集！")
        print("   请先运行: python meta_learning/collect_training_data.py")
        return
    
    dataset_file = dataset_file[-1]  # 使用最新的
    print(f"\n加载数据集: {dataset_file}")
    
    # 加载数据
    with open(dataset_file, 'r') as f:
        data_points = json.load(f)
    
    print(f"总数据点: {len(data_points)}")
    
    if len(data_points) < 10:
        print("⚠️  数据点太少，建议至少20个以上")
        print("   当前将使用简化训练流程")
    
    # 划分训练/验证集
    train_data, val_data = train_test_split(data_points, test_size=0.2, random_state=42)
    print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}")
    
    # 创建数据集
    feature_extractor = RobotFeatureExtractor()
    train_dataset = PIDDataset(train_data, feature_extractor)
    val_dataset = PIDDataset(val_data, feature_extractor, 
                            normalization_stats=train_dataset.get_normalization_stats())
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    model = MetaPIDNetwork(
        feature_dim=len(feature_extractor.feature_names),
        max_dof=7,
        hidden_dims=[256, 256, 128]
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # 训练
    num_epochs = 200
    best_val_loss = float('inf')
    train_history = []
    val_history = []
    
    print(f"\n开始训练 ({num_epochs} epochs)...")
    print("=" * 80)
    
    for epoch in range(num_epochs):
        # 训练
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        train_history.append(train_metrics)
        
        # 验证
        val_metrics = evaluate(model, val_loader, device)
        val_history.append(val_metrics)
        
        # 学习率调整
        scheduler.step(val_metrics['loss'])
        
        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val MSE: {val_metrics['mse']:.4f} | "
                  f"Val RelErr: {val_metrics['rel_error']:.4f}")
        
        # 保存最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            
            # 创建优化器并保存
            meta_optimizer = MetaPIDOptimizer(device=device)
            meta_optimizer.model = model
            meta_optimizer.normalization_stats = train_dataset.get_normalization_stats()
            
            model_path = Path('meta_learning/models/best_meta_pid.pth')
            model_path.parent.mkdir(parents=True, exist_ok=True)
            meta_optimizer.save(model_path)
            
            if epoch > 0:
                print(f"      💾 最佳模型已保存 (val_loss={best_val_loss:.4f})")
    
    print("\n" + "=" * 80)
    print("训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    
    # 绘制训练曲线
    plot_path = Path('meta_learning/models/training_curves.png')
    plot_training_curves(train_history, val_history, plot_path)
    
    print("\n下一步: 测试零样本泛化能力")
    print("  python meta_learning/test_zero_shot.py")


if __name__ == '__main__':
    main()

