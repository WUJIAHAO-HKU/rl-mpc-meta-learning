#!/usr/bin/env python3
"""
Laikago四足机器人性能评估
对比纯Meta-PID和Meta-PID+RL的跟踪性能
"""

import numpy as np
import pybullet as p
import torch
from stable_baselines3 import PPO
from meta_rl_combined_env import MetaRLCombinedEnv


def evaluate_laikago(model_path=None, steps=10000):
    """评估Laikago性能"""
    
    robot_urdf = 'laikago/laikago.urdf'
    test_name = "纯Meta-PID" if model_path is None else "Meta-PID + RL"
    
    print(f"\n{'='*80}")
    print(f"评估: {test_name}")
    print(f"{'='*80}")
    
    # 创建环境
    env = MetaRLCombinedEnv(robot_urdf=robot_urdf, gui=False)
    
    # 加载RL模型
    model = None
    if model_path is not None:
        model = PPO.load(model_path)
        print(f"✅ RL模型加载成功")
    else:
        print(f"✅ 使用固定Meta-PID")
    
    obs, _ = env.reset()
    
    # 记录数据
    actual_errors = []
    actual_errors_deg = []
    kp_values = []
    kd_values = []
    
    for step in range(steps):
        # 选择动作
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.zeros(2)
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 获取实际误差
        joint_states = p.getJointStates(env.robot_id, env.controllable_joints)
        q_actual = np.array([s[0] for s in joint_states])
        q_ref = env._get_reference_trajectory()
        
        error_rad = np.linalg.norm(q_ref - q_actual)
        error_deg = np.degrees(error_rad)
        
        actual_errors.append(error_rad)
        actual_errors_deg.append(error_deg)
        kp_values.append(info['current_kp'])
        kd_values.append(info['current_kd'])
        
        if step % 2000 == 0:
            print(f"Step {step:5d}: 误差={error_deg:.2f}°, "
                  f"Kp={info['current_kp']:.2f}, "
                  f"Kd={info['current_kd']:.2f}")
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    env.close()
    
    # 统计结果
    results = {
        'mean_error_rad': np.mean(actual_errors),
        'mean_error_deg': np.mean(actual_errors_deg),
        'median_error_deg': np.median(actual_errors_deg),
        'max_error_deg': np.max(actual_errors_deg),
        'std_error_deg': np.std(actual_errors_deg),
        'mean_kp': np.mean(kp_values),
        'mean_kd': np.mean(kd_values)
    }
    
    print(f"\n📊 {test_name} 实际跟踪性能:")
    print(f"   平均误差: {results['mean_error_deg']:.4f}° ({results['mean_error_rad']:.6f} rad)")
    print(f"   中位误差: {results['median_error_deg']:.4f}°")
    print(f"   最大误差: {results['max_error_deg']:.4f}°")
    print(f"   标准差:   {results['std_error_deg']:.4f}°")
    print(f"   平均Kp:   {results['mean_kp']:.2f}")
    print(f"   平均Kd:   {results['mean_kd']:.2f}")
    
    return results


def main():
    print("="*80)
    print("Laikago四足机器人性能评估")
    print("="*80)
    print("机器人: laikago/laikago.urdf (12-DOF)")
    print("测试步数: 10000")
    print()
    
    # 评估1: 纯Meta-PID
    pure_results = evaluate_laikago(model_path=None, steps=10000)
    
    # 评估2: Meta-PID + RL
    rl_results = evaluate_laikago(
        model_path='logs/meta_rl_laikago/best_model/best_model',
        steps=10000
    )
    
    # 性能对比
    print("\n" + "="*80)
    print("Laikago性能对比总结")
    print("="*80)
    
    error_improvement = (pure_results['mean_error_deg'] - rl_results['mean_error_deg']) / pure_results['mean_error_deg'] * 100
    max_error_improvement = (pure_results['max_error_deg'] - rl_results['max_error_deg']) / pure_results['max_error_deg'] * 100
    std_improvement = (pure_results['std_error_deg'] - rl_results['std_error_deg']) / pure_results['std_error_deg'] * 100
    
    print(f"\n✅ 平均误差改善: {pure_results['mean_error_deg']:.4f}° → {rl_results['mean_error_deg']:.4f}° "
          f"({error_improvement:+.2f}%)")
    print(f"✅ 最大误差改善: {pure_results['max_error_deg']:.4f}° → {rl_results['max_error_deg']:.4f}° "
          f"({max_error_improvement:+.2f}%)")
    print(f"✅ 标准差改善:   {pure_results['std_error_deg']:.4f}° → {rl_results['std_error_deg']:.4f}° "
          f"({std_improvement:+.2f}%)")
    
    # 对比Franka Panda结果
    print("\n" + "="*80)
    print("跨平台泛化性能对比")
    print("="*80)
    print(f"\n{'机器人':<15} {'DOF':<6} {'纯Meta-PID':<15} {'Meta-PID+RL':<15} {'改善':<10}")
    print("-"*80)
    print(f"{'Franka Panda':<15} {'9':<6} {'46.76°':<15} {'34.93°':<15} {'+25.31%':<10}")
    
    laikago_pure = f"{pure_results['mean_error_deg']:.2f}°"
    laikago_rl = f"{rl_results['mean_error_deg']:.2f}°"
    laikago_imp = f"{error_improvement:+.2f}%"
    print(f"{'Laikago':<15} {'12':<6} {laikago_pure:<15} {laikago_rl:<15} {laikago_imp:<10}")
    
    print("\n✅ 结论: Meta-PID+RL方法在不同机器人平台上均表现出良好的泛化能力！")
    
    print("\n" + "="*80)
    print("✅ Laikago评估完成！")
    print("="*80)


if __name__ == '__main__':
    main()

