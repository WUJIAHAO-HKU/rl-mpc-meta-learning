#!/usr/bin/env python3
"""
扰动场景鲁棒性测试
对比纯Meta-PID和Meta-PID+RL在不同扰动下的性能
"""

import numpy as np
import pybullet as p
import torch
import argparse
from stable_baselines3 import PPO
from meta_rl_combined_env import MetaRLCombinedEnv
import matplotlib.pyplot as plt


def evaluate_under_disturbance(robot_urdf, disturbance_type, model_path=None, 
                                n_episodes=10, max_steps=5000):
    """
    在特定扰动下评估性能
    
    Args:
        robot_urdf: 机器人URDF路径
        disturbance_type: 扰动类型
        model_path: RL模型路径（None表示纯Meta-PID）
        n_episodes: 测试回合数
        max_steps: 每回合最大步数
    """
    # 创建环境
    env = MetaRLCombinedEnv(
        robot_urdf=robot_urdf, 
        gui=False
    )
    
    # 记录扰动类型（用于统计）
    env.disturbance_type = disturbance_type
    
    # 加载RL模型（如果有）
    model = None
    if model_path is not None:
        model = PPO.load(model_path)
    
    # 记录数据
    all_errors = []
    all_max_errors = []
    all_rewards = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_errors = []
        episode_reward = 0
        
        for step in range(max_steps):
            # 选择动作
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.zeros(2)  # 固定Meta-PID
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 获取实际关节误差
            joint_states = p.getJointStates(env.robot_id, env.controllable_joints)
            q_actual = np.array([s[0] for s in joint_states])
            q_ref = env._get_reference_trajectory()
            
            # 计算误差（角度）
            error_rad = np.linalg.norm(q_ref - q_actual)
            error_deg = np.degrees(error_rad)
            
            episode_errors.append(error_deg)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        # 统计本回合
        all_errors.extend(episode_errors)
        all_max_errors.append(np.max(episode_errors))
        all_rewards.append(episode_reward)
        
        print(f"Episode {episode+1}/{n_episodes}: "
              f"Mean={np.mean(episode_errors):.2f}°, "
              f"Max={np.max(episode_errors):.2f}°, "
              f"Reward={episode_reward:.1f}")
    
    env.close()
    
    # 返回统计结果
    results = {
        'disturbance': disturbance_type,
        'mean_error': np.mean(all_errors),
        'median_error': np.median(all_errors),
        'max_error': np.mean(all_max_errors),  # 平均最大误差
        'std_error': np.std(all_errors),
        'mean_reward': np.mean(all_rewards)
    }
    
    return results


def plot_robustness_comparison(pure_results, rl_results, save_path='robustness_comparison.png'):
    """绘制鲁棒性对比图"""
    disturbances = list(pure_results.keys())
    
    # 提取数据
    pure_mean = [pure_results[d]['mean_error'] for d in disturbances]
    rl_mean = [rl_results[d]['mean_error'] for d in disturbances]
    
    pure_max = [pure_results[d]['max_error'] for d in disturbances]
    rl_max = [rl_results[d]['max_error'] for d in disturbances]
    
    pure_std = [pure_results[d]['std_error'] for d in disturbances]
    rl_std = [rl_results[d]['std_error'] for d in disturbances]
    
    # 计算改善百分比
    improvements = [(pure_mean[i] - rl_mean[i]) / pure_mean[i] * 100 
                    for i in range(len(disturbances))]
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 平均误差对比
    ax = axes[0, 0]
    x = np.arange(len(disturbances))
    width = 0.35
    ax.bar(x - width/2, pure_mean, width, label='Pure Meta-PID', alpha=0.8)
    ax.bar(x + width/2, rl_mean, width, label='Meta-PID + RL', alpha=0.8)
    ax.set_xlabel('Disturbance Type')
    ax.set_ylabel('Mean Error (degrees)')
    ax.set_title('Mean Tracking Error Under Different Disturbances')
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace('_', ' ').title() for d in disturbances], rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. 最大误差对比
    ax = axes[0, 1]
    ax.bar(x - width/2, pure_max, width, label='Pure Meta-PID', alpha=0.8)
    ax.bar(x + width/2, rl_max, width, label='Meta-PID + RL', alpha=0.8)
    ax.set_xlabel('Disturbance Type')
    ax.set_ylabel('Max Error (degrees)')
    ax.set_title('Maximum Tracking Error Under Different Disturbances')
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace('_', ' ').title() for d in disturbances], rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. 标准差对比
    ax = axes[1, 0]
    ax.bar(x - width/2, pure_std, width, label='Pure Meta-PID', alpha=0.8)
    ax.bar(x + width/2, rl_std, width, label='Meta-PID + RL', alpha=0.8)
    ax.set_xlabel('Disturbance Type')
    ax.set_ylabel('Std Dev (degrees)')
    ax.set_title('Error Standard Deviation Under Different Disturbances')
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace('_', ' ').title() for d in disturbances], rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. 改善百分比
    ax = axes[1, 1]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax.bar(x, improvements, alpha=0.8, color=colors)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Disturbance Type')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Performance Improvement with RL Adaptation')
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace('_', ' ').title() for d in disturbances], rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上标注数值
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.1f}%',
                ha='center', va='bottom' if imp > 0 else 'top',
                fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 鲁棒性对比图已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='评估扰动场景鲁棒性')
    parser.add_argument('--robot', default='franka_panda/panda.urdf', help='机器人URDF路径')
    parser.add_argument('--model', default='logs/meta_rl_panda/best_model/best_model', help='RL模型路径')
    parser.add_argument('--disturbances', nargs='+', 
                        default=['none', 'random_force', 'payload', 'param_uncertainty'],
                        help='扰动类型列表')
    parser.add_argument('--n_episodes', type=int, default=10, help='每种扰动的测试回合数')
    parser.add_argument('--max_steps', type=int, default=5000, help='每回合最大步数')
    args = parser.parse_args()
    
    print("="*80)
    print("扰动场景鲁棒性测试")
    print("="*80)
    print(f"机器人: {args.robot}")
    print(f"扰动类型: {args.disturbances}")
    print(f"每种扰动测试回合: {args.n_episodes}")
    print()
    
    # 测试纯Meta-PID
    print("="*80)
    print("测试1: 纯Meta-PID（固定预测值）")
    print("="*80)
    pure_results = {}
    for disturbance in args.disturbances:
        print(f"\n--- 扰动类型: {disturbance} ---")
        result = evaluate_under_disturbance(
            args.robot, disturbance, model_path=None,
            n_episodes=args.n_episodes, max_steps=args.max_steps
        )
        pure_results[disturbance] = result
        print(f"✅ {disturbance}: Mean={result['mean_error']:.2f}°, "
              f"Max={result['max_error']:.2f}°, Std={result['std_error']:.2f}°")
    
    # 测试Meta-PID + RL
    print("\n" + "="*80)
    print("测试2: Meta-PID + RL（动态调整）")
    print("="*80)
    rl_results = {}
    for disturbance in args.disturbances:
        print(f"\n--- 扰动类型: {disturbance} ---")
        result = evaluate_under_disturbance(
            args.robot, disturbance, model_path=args.model,
            n_episodes=args.n_episodes, max_steps=args.max_steps
        )
        rl_results[disturbance] = result
        print(f"✅ {disturbance}: Mean={result['mean_error']:.2f}°, "
              f"Max={result['max_error']:.2f}°, Std={result['std_error']:.2f}°")
    
    # 性能对比总结
    print("\n" + "="*80)
    print("鲁棒性对比总结")
    print("="*80)
    print(f"\n{'扰动类型':<20} {'纯Meta-PID':<15} {'Meta-PID+RL':<15} {'改善':<10}")
    print("-"*80)
    
    total_improvement = 0
    for disturbance in args.disturbances:
        pure_err = pure_results[disturbance]['mean_error']
        rl_err = rl_results[disturbance]['mean_error']
        improvement = (pure_err - rl_err) / pure_err * 100
        total_improvement += improvement
        
        print(f"{disturbance:<20} {pure_err:>8.2f}°      {rl_err:>8.2f}°      {improvement:>+6.2f}%")
    
    avg_improvement = total_improvement / len(args.disturbances)
    print("-"*80)
    print(f"{'平均改善':<20} {'':<15} {'':<15} {avg_improvement:>+6.2f}%")
    
    # 绘制对比图
    plot_robustness_comparison(pure_results, rl_results)
    
    print("\n" + "="*80)
    print("✅ 鲁棒性测试完成！")
    print("="*80)


if __name__ == '__main__':
    main()

