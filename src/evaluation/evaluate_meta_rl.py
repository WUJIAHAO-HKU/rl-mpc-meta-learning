#!/usr/bin/env python3
"""
评估 Meta-PID + RL 组合控制器
对比纯Meta-PID和Meta-PID+RL的性能
"""

import numpy as np
import pybullet as p
import torch
from stable_baselines3 import PPO
from meta_rl_combined_env import MetaRLCombinedEnv
import matplotlib.pyplot as plt
from pathlib import Path


def evaluate_pure_meta_pid(robot_urdf, steps=10000):
    """评估纯Meta-PID（固定预测的PID）"""
    print("\n" + "="*80)
    print("评估 1: 纯Meta-PID（固定预测值）")
    print("="*80)
    
    # 创建环境
    env = MetaRLCombinedEnv(robot_urdf=robot_urdf, gui=False)
    
    obs, _ = env.reset()
    
    # 记录数据
    errors = []
    kp_values = []
    kd_values = []
    rewards = []
    
    # 使用零动作（不调整，保持Meta-PID预测值）
    zero_action = np.zeros(2)
    
    for step in range(steps):
        obs, reward, terminated, truncated, info = env.step(zero_action)
        
        errors.append(info['tracking_error'])
        kp_values.append(info['current_kp'])
        kd_values.append(info['current_kd'])
        rewards.append(reward)
        
        if step % 2000 == 0:
            print(f"Step {step:5d}: error={info['tracking_error']:.4f}, "
                  f"Kp={info['current_kp']:.2f}, Kd={info['current_kd']:.2f}")
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    env.close()
    
    results = {
        'errors': np.array(errors),
        'kp_values': np.array(kp_values),
        'kd_values': np.array(kd_values),
        'rewards': np.array(rewards),
        'total_reward': np.sum(rewards),
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'max_error': np.max(errors)
    }
    
    print(f"\n纯Meta-PID 总奖励: {results['total_reward']:.2f}")
    print(f"纯Meta-PID 平均误差: {results['mean_error']:.4f}")
    print(f"纯Meta-PID 中位误差: {results['median_error']:.4f}")
    print(f"纯Meta-PID 最大误差: {results['max_error']:.4f}")
    print(f"PID参数保持: Kp={np.mean(kp_values):.2f}, Kd={np.mean(kd_values):.2f}")
    
    return results


def evaluate_meta_rl(robot_urdf, model_path, steps=10000):
    """评估Meta-PID+RL（动态调整）"""
    print("\n" + "="*80)
    print("评估 2: Meta-PID + RL（动态调整）")
    print("="*80)
    
    # 加载模型
    model = PPO.load(model_path)
    print(f"✅ 模型加载成功: {model_path}")
    
    # 创建环境
    env = MetaRLCombinedEnv(robot_urdf=robot_urdf, gui=False)
    
    obs, _ = env.reset()
    
    # 记录数据
    errors = []
    kp_values = []
    kd_values = []
    rewards = []
    actions = []
    
    for step in range(steps):
        # RL动作
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        errors.append(info['tracking_error'])
        kp_values.append(info['current_kp'])
        kd_values.append(info['current_kd'])
        rewards.append(reward)
        actions.append(action)
        
        if step % 2000 == 0:
            print(f"Step {step:5d}: error={info['tracking_error']:.4f}, "
                  f"Kp={info['current_kp']:.2f}, Kd={info['current_kd']:.2f}, "
                  f"action={action}")
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    env.close()
    
    results = {
        'errors': np.array(errors),
        'kp_values': np.array(kp_values),
        'kd_values': np.array(kd_values),
        'rewards': np.array(rewards),
        'actions': np.array(actions),
        'total_reward': np.sum(rewards),
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'max_error': np.max(errors)
    }
    
    print(f"\nMeta-PID+RL 总奖励: {results['total_reward']:.2f}")
    print(f"Meta-PID+RL 平均误差: {results['mean_error']:.4f}")
    print(f"Meta-PID+RL 中位误差: {results['median_error']:.4f}")
    print(f"Meta-PID+RL 最大误差: {results['max_error']:.4f}")
    print(f"PID参数范围: Kp=[{np.min(kp_values):.2f}, {np.max(kp_values):.2f}], "
          f"Kd=[{np.min(kd_values):.2f}, {np.max(kd_values):.2f}]")
    
    return results


def plot_comparison(pure_results, rl_results, save_path='meta_rl_comparison.png'):
    """绘制对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 定义统一配色
    color_pure = '#2E86AB'  # 蓝色 - Pure Meta-PID
    color_rl = '#F77F00'    # 橙色 - Meta-PID + RL
    
    # 1. 跟踪误差对比
    ax = axes[0, 0]
    window = 100
    pure_smooth = np.convolve(pure_results['errors'], 
                               np.ones(window)/window, mode='valid')
    rl_smooth = np.convolve(rl_results['errors'], 
                             np.ones(window)/window, mode='valid')
    
    ax.plot(pure_smooth, color=color_pure, label='Pure Meta-PID', 
            alpha=0.8, linewidth=2)
    ax.plot(rl_smooth, color=color_rl, label='Meta-PID + RL', 
            alpha=0.8, linewidth=2)
    ax.set_xlabel('Time Step', fontweight='bold')
    ax.set_ylabel('Tracking Error (normalized)', fontweight='bold')
    ax.set_title('Tracking Error Comparison', fontweight='bold')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 2. 奖励对比
    ax = axes[0, 1]
    pure_reward_smooth = np.convolve(pure_results['rewards'], 
                                      np.ones(window)/window, mode='valid')
    rl_reward_smooth = np.convolve(rl_results['rewards'], 
                                    np.ones(window)/window, mode='valid')
    
    ax.plot(pure_reward_smooth, color=color_pure, label='Pure Meta-PID', 
            alpha=0.8, linewidth=2)
    ax.plot(rl_reward_smooth, color=color_rl, label='Meta-PID + RL', 
            alpha=0.8, linewidth=2)
    ax.set_xlabel('Time Step', fontweight='bold')
    ax.set_ylabel('Reward', fontweight='bold')
    ax.set_title('Reward Comparison', fontweight='bold')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 3. Kp动态调整
    ax = axes[1, 0]
    # 纯Meta-PID：固定值，用蓝色虚线表示
    pure_kp_mean = np.mean(pure_results['kp_values'][:2000])
    ax.axhline(pure_kp_mean, color=color_pure, linestyle='--', linewidth=2.5, 
               label=f'Pure Meta-PID (fixed at {pure_kp_mean:.1f})', alpha=0.8, zorder=5)
    
    # Meta-PID+RL：动态调整，用橙色实线+填充区域
    rl_kp = rl_results['kp_values'][:2000]
    ax.plot(rl_kp, color=color_rl, linewidth=2, 
            label='Meta-PID + RL (adaptive)', alpha=0.9, zorder=3)
    # 添加填充区域显示RL的调整范围
    ax.fill_between(range(len(rl_kp)), pure_kp_mean, rl_kp, 
                    color=color_rl, alpha=0.2, zorder=1)
    
    ax.set_xlabel('Time Step', fontweight='bold')
    ax.set_ylabel('Kp', fontweight='bold')
    ax.set_title('Kp Adjustment (First Episode)', fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 4. Kd动态调整
    ax = axes[1, 1]
    # 纯Meta-PID：固定值，用蓝色虚线表示
    pure_kd_mean = np.mean(pure_results['kd_values'][:2000])
    ax.axhline(pure_kd_mean, color=color_pure, linestyle='--', linewidth=2.5, 
               label=f'Pure Meta-PID (fixed at {pure_kd_mean:.1f})', alpha=0.8, zorder=5)
    
    # Meta-PID+RL：动态调整，用橙色实线+填充区域
    rl_kd = rl_results['kd_values'][:2000]
    ax.plot(rl_kd, color=color_rl, linewidth=2, 
            label='Meta-PID + RL (adaptive)', alpha=0.9, zorder=3)
    # 添加填充区域显示RL的调整范围
    ax.fill_between(range(len(rl_kd)), pure_kd_mean, rl_kd, 
                    color=color_rl, alpha=0.2, zorder=1)
    
    ax.set_xlabel('Time Step', fontweight='bold')
    ax.set_ylabel('Kd', fontweight='bold')
    ax.set_title('Kd Adjustment (First Episode)', fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 对比图已保存: {save_path}")


def main():
    robot_urdf = 'franka_panda/panda.urdf'
    model_path = 'logs/meta_rl_panda/best_model/best_model'
    steps = 10000
    
    print("="*80)
    print("Meta-PID + RL 组合控制器评估")
    print("="*80)
    print(f"机器人: {robot_urdf}")
    print(f"评估步数: {steps}")
    print(f"模型路径: {model_path}")
    
    # 评估1: 纯Meta-PID
    pure_results = evaluate_pure_meta_pid(robot_urdf, steps)
    
    # 评估2: Meta-PID + RL
    rl_results = evaluate_meta_rl(robot_urdf, model_path, steps)
    
    # 性能对比
    print("\n" + "="*80)
    print("性能对比总结")
    print("="*80)
    
    reward_improvement = (rl_results['total_reward'] - pure_results['total_reward']) / abs(pure_results['total_reward']) * 100
    error_improvement = (pure_results['mean_error'] - rl_results['mean_error']) / pure_results['mean_error'] * 100
    
    print(f"\n奖励改善: {rl_results['total_reward'] - pure_results['total_reward']:+.2f} ({reward_improvement:+.2f}%)")
    print(f"误差降低: {pure_results['mean_error'] - rl_results['mean_error']:.4f} ({error_improvement:+.2f}%)")
    
    # 绘制对比图
    plot_comparison(pure_results, rl_results)
    
    print("\n" + "="*80)
    print("✅ 评估完成！")
    print("="*80)


if __name__ == '__main__':
    main()

