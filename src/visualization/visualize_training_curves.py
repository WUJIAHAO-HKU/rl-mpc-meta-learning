#!/usr/bin/env python3
"""
训练曲线可视化
展示RL训练过程中的奖励、误差和价值函数变化
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def plot_training_curves(eval_npz_path, save_path='training_curves.png'):
    """
    绘制训练曲线
    
    Args:
        eval_npz_path: evaluations.npz文件路径
        save_path: 保存路径
    """
    # 加载评估数据
    data = np.load(eval_npz_path)
    timesteps = data['timesteps']
    results = data['results']  # 平均奖励
    ep_lengths = data['ep_lengths']  # 回合长度
    
    # 处理多维数组（取平均值）
    if len(results.shape) > 1:
        results = np.mean(results, axis=1)
    if len(ep_lengths.shape) > 1:
        ep_lengths = np.mean(ep_lengths, axis=1)
    
    print(f"✅ 加载训练数据: {eval_npz_path}")
    print(f"   训练步数: {timesteps[-1]}")
    print(f"   评估次数: {len(timesteps)}")
    print(f"   最终奖励: {results[-1]:.2f}")
    print(f"   奖励改善: {results[-1] - results[0]:.2f}")
    
    # 创建图表
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. 奖励曲线（原始 + 平滑）
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(timesteps, results, alpha=0.3, color='blue', label='Raw Reward')
    
    # 平滑处理（移动平均）
    window = min(5, len(results))
    if window > 1:
        smoothed = np.convolve(results, np.ones(window)/window, mode='valid')
        smoothed_timesteps = timesteps[:len(smoothed)]
        ax1.plot(smoothed_timesteps, smoothed, color='blue', linewidth=2, label='Smoothed Reward')
    
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Mean Reward', fontsize=12)
    ax1.set_title('RL Training Progress: Reward Evolution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 标注起始和最终奖励
    ax1.annotate(f'Start: {results[0]:.1f}', 
                xy=(timesteps[0], results[0]),
                xytext=(timesteps[0], results[0] + (results[-1] - results[0]) * 0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    ax1.annotate(f'Final: {results[-1]:.1f}', 
                xy=(timesteps[-1], results[-1]),
                xytext=(timesteps[-1] * 0.85, results[-1]),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')
    
    # 2. 回合长度变化
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(timesteps, ep_lengths, color='orange', linewidth=2)
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Episode Length', fontsize=12)
    ax2.set_title('Episode Length During Training', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 奖励改善率（相对于初始值）
    ax3 = fig.add_subplot(gs[1, 1])
    improvement = (results - results[0]) / abs(results[0]) * 100
    ax3.plot(timesteps, improvement, color='green', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.set_xlabel('Training Steps', fontsize=12)
    ax3.set_ylabel('Improvement (%)', fontsize=12)
    ax3.set_title('Reward Improvement Relative to Start', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.fill_between(timesteps, 0, improvement, where=(improvement > 0), 
                     alpha=0.3, color='green', label='Improvement')
    ax3.fill_between(timesteps, 0, improvement, where=(improvement < 0), 
                     alpha=0.3, color='red', label='Degradation')
    ax3.legend()
    
    # 4. 奖励分布（直方图）
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(results, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax4.axvline(np.mean(results), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(results):.2f}')
    ax4.axvline(np.median(results), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(results):.2f}')
    ax4.set_xlabel('Mean Reward', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Reward Distribution', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. 训练阶段分析（前25%、中50%、后25%）
    ax5 = fig.add_subplot(gs[2, 1])
    n = len(results)
    early = results[:n//4]
    middle = results[n//4:3*n//4]
    late = results[3*n//4:]
    
    stages = ['Early\n(0-25%)', 'Middle\n(25-75%)', 'Late\n(75-100%)']
    means = [np.mean(early), np.mean(middle), np.mean(late)]
    stds = [np.std(early), np.std(middle), np.std(late)]
    
    x_pos = np.arange(len(stages))
    bars = ax5.bar(x_pos, means, yerr=stds, alpha=0.7, capsize=5,
                   color=['skyblue', 'lightgreen', 'lightcoral'],
                   edgecolor='black', linewidth=1.5)
    
    ax5.set_xlabel('Training Stage', fontsize=12)
    ax5.set_ylabel('Mean Reward', fontsize=12)
    ax5.set_title('Performance by Training Stage', fontsize=12, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(stages)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上标注数值
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 保存图表
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 训练曲线已保存: {save_path}")
    
    # 统计信息
    print(f"\n📈 训练统计:")
    print(f"   起始奖励:   {results[0]:.2f}")
    print(f"   最终奖励:   {results[-1]:.2f}")
    print(f"   平均奖励:   {np.mean(results):.2f}")
    print(f"   中位奖励:   {np.median(results):.2f}")
    print(f"   标准差:     {np.std(results):.2f}")
    print(f"   最大奖励:   {np.max(results):.2f}")
    print(f"   最小奖励:   {np.min(results):.2f}")
    print(f"   总改善:     {results[-1] - results[0]:.2f} ({(results[-1] - results[0]) / abs(results[0]) * 100:+.2f}%)")
    
    return {
        'timesteps': timesteps,
        'rewards': results,
        'ep_lengths': ep_lengths,
        'initial_reward': results[0],
        'final_reward': results[-1],
        'mean_reward': np.mean(results),
        'improvement': results[-1] - results[0]
    }


def compare_multiple_runs(eval_paths, labels, save_path='training_comparison.png'):
    """
    对比多次训练运行
    
    Args:
        eval_paths: evaluations.npz文件路径列表
        labels: 标签列表
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    for i, (path, label) in enumerate(zip(eval_paths, labels)):
        if not os.path.exists(path):
            print(f"⚠️  文件不存在: {path}")
            continue
        
        data = np.load(path)
        timesteps = data['timesteps']
        results = data['results']
        
        # 奖励曲线
        ax1.plot(timesteps, results, color=colors[i % len(colors)], 
                linewidth=2, label=label, alpha=0.8)
        
        # 累积改善
        improvement = (results - results[0]) / abs(results[0]) * 100
        ax2.plot(timesteps, improvement, color=colors[i % len(colors)],
                linewidth=2, label=label, alpha=0.8)
    
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('Training Curves Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Relative Improvement Comparison')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 对比图已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='可视化RL训练曲线')
    parser.add_argument('--eval_path', 
                        default='logs/meta_rl_panda/evaluations.npz',
                        help='evaluations.npz文件路径')
    parser.add_argument('--output', 
                        default='training_curves.png',
                        help='输出图片路径')
    args = parser.parse_args()
    
    print("="*80)
    print("RL训练曲线可视化")
    print("="*80)
    print(f"评估文件: {args.eval_path}")
    print()
    
    if not os.path.exists(args.eval_path):
        print(f"❌ 错误: 文件不存在 - {args.eval_path}")
        print(f"\n💡 提示: 请先运行训练脚本生成评估数据")
        return
    
    # 绘制训练曲线
    stats = plot_training_curves(args.eval_path, args.output)
    
    print("\n" + "="*80)
    print("✅ 可视化完成！")
    print("="*80)


if __name__ == '__main__':
    main()

