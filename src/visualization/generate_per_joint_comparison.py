#!/usr/bin/env python3
"""
生成逐关节误差对比数据表格和图表
支持多个机器人平台的对比
"""

import numpy as np
import pybullet as p
import torch
import matplotlib.pyplot as plt
import matplotlib
from stable_baselines3 import PPO
from meta_rl_combined_env import MetaRLCombinedEnv

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def evaluate_per_joint_error(robot_urdf, robot_name, model_path=None, steps=10000):
    """评估逐关节误差"""
    
    test_name = "Pure Meta-PID" if model_path is None else "Meta-PID + RL"
    
    print(f"\n{'='*80}")
    print(f"评估: {robot_name} - {test_name}")
    print(f"{'='*80}")
    
    # 创建环境
    env = MetaRLCombinedEnv(robot_urdf=robot_urdf, gui=False)
    n_joints = len(env.controllable_joints)
    
    # 加载RL模型
    model = None
    if model_path is not None:
        try:
            model = PPO.load(model_path)
            print(f"✅ RL模型加载成功")
        except:
            print(f"⚠️  RL模型未找到，使用固定Meta-PID")
    else:
        print(f"✅ 使用固定Meta-PID")
    
    obs, _ = env.reset()
    
    # 记录每个关节的误差
    joint_errors_all = []  # shape: (steps, n_joints)
    
    for step in range(steps):
        # 选择动作
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.zeros(2)
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 获取当前关节状态
        joint_states = p.getJointStates(env.robot_id, env.controllable_joints)
        q_actual = np.array([s[0] for s in joint_states])
        q_ref = env._get_reference_trajectory()
        
        # 计算关节误差（弧度）
        joint_errors = np.abs(q_ref - q_actual)
        joint_errors_all.append(joint_errors)
        
        if step % 2000 == 0:
            mean_error = np.mean(joint_errors)
            print(f"Step {step:5d}: mean_error={np.degrees(mean_error):.2f}°")
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    env.close()
    
    # 转换为numpy数组并计算统计量
    joint_errors_all = np.array(joint_errors_all)  # shape: (steps, n_joints)
    joint_errors_deg = np.degrees(joint_errors_all)
    
    # 计算每个关节的统计量
    per_joint_mean = np.mean(joint_errors_deg, axis=0)
    per_joint_std = np.std(joint_errors_deg, axis=0)
    per_joint_max = np.max(joint_errors_deg, axis=0)
    
    # 全局统计量
    overall_mae = np.mean(joint_errors_deg)
    overall_rmse = np.sqrt(np.mean(np.linalg.norm(joint_errors_deg, axis=1)**2))
    overall_max = np.max(joint_errors_deg)
    
    results = {
        'robot_name': robot_name,
        'n_joints': n_joints,
        'per_joint_mean': per_joint_mean,
        'per_joint_std': per_joint_std,
        'per_joint_max': per_joint_max,
        'overall_mae': overall_mae,
        'overall_rmse': overall_rmse,
        'overall_max': overall_max,
    }
    
    print(f"\n📊 {robot_name} - {test_name} 结果:")
    print(f"   总体MAE: {overall_mae:.2f}°")
    print(f"   总体RMSE: {overall_rmse:.2f}°")
    print(f"   最大误差: {overall_max:.2f}°")
    print(f"\n   各关节平均误差 (MAE):")
    for i, (mean_err, std_err) in enumerate(zip(per_joint_mean, per_joint_std)):
        print(f"      关节 {i+1:2d}: {mean_err:6.2f}° ± {std_err:5.2f}°")
    
    return results


def generate_latex_table(results_dict):
    """生成LaTeX表格"""
    
    print("\n" + "="*80)
    print("LaTeX表格代码")
    print("="*80)
    
    latex_code = """
\\begin{table}[h]
\\caption{Per-Joint Tracking Error Comparison}
\\label{tab:per_joint_error}
\\begin{tabular*}{\\tblwidth}{@{}LLLLL@{}}
\\toprule
\\textbf{Robot} & \\textbf{Joint} & \\textbf{Pure Meta-PID (°)} & \\textbf{Meta-PID+RL (°)} & \\textbf{Improv.} \\\\
\\midrule
"""
    
    for robot_name, data in results_dict.items():
        pure_results = data['pure']
        rl_results = data['rl']
        n_joints = pure_results['n_joints']
        
        for i in range(n_joints):
            pure_err = pure_results['per_joint_mean'][i]
            rl_err = rl_results['per_joint_mean'][i]
            improvement = (pure_err - rl_err) / pure_err * 100
            
            robot_col = robot_name if i == 0 else ""
            joint_col = f"J{i+1}"
            
            latex_code += f"{robot_col:<15} & {joint_col:<6} & {pure_err:6.2f} & {rl_err:6.2f} & {improvement:+5.1f}\\% \\\\\n"
        
        # 添加总体统计
        pure_mae = pure_results['overall_mae']
        rl_mae = rl_results['overall_mae']
        overall_improvement = (pure_mae - rl_mae) / pure_mae * 100
        
        latex_code += f"\\midrule\n"
        latex_code += f"\\textit{{{robot_name} Avg}} & & {pure_mae:6.2f} & {rl_mae:6.2f} & {overall_improvement:+5.1f}\\% \\\\\n"
        
        if robot_name != list(results_dict.keys())[-1]:
            latex_code += "\\midrule\n"
    
    latex_code += """\\bottomrule
\\end{tabular*}
\\end{table}
"""
    
    print(latex_code)
    
    # 保存到文件
    with open('per_joint_error_table.tex', 'w', encoding='utf-8') as f:
        f.write(latex_code)
    print("✅ LaTeX表格已保存到: per_joint_error_table.tex")
    
    return latex_code


def plot_per_joint_comparison(results_dict, save_path='per_joint_error_comparison.png'):
    """绘制逐关节误差对比图"""
    
    n_robots = len(results_dict)
    fig, axes = plt.subplots(1, n_robots, figsize=(7*n_robots, 5))
    
    if n_robots == 1:
        axes = [axes]
    
    for idx, (robot_name, data) in enumerate(results_dict.items()):
        ax = axes[idx]
        
        pure_results = data['pure']
        rl_results = data['rl']
        
        n_joints = pure_results['n_joints']
        joint_indices = np.arange(1, n_joints + 1)
        
        pure_mean = pure_results['per_joint_mean']
        rl_mean = rl_results['per_joint_mean']
        pure_std = pure_results['per_joint_std']
        rl_std = rl_results['per_joint_std']
        
        width = 0.35
        x = joint_indices
        
        # 绘制柱状图
        bars1 = ax.bar(x - width/2, pure_mean, width, 
                       yerr=pure_std, capsize=3,
                       label='Pure Meta-PID', 
                       color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, rl_mean, width,
                       yerr=rl_std, capsize=3,
                       label='Meta-PID + RL', 
                       color='coral', alpha=0.8)
        
        # 在柱子上方标注改善百分比
        for i, (p_err, r_err) in enumerate(zip(pure_mean, rl_mean)):
            improvement = (p_err - r_err) / p_err * 100
            y_pos = max(p_err, r_err) + max(pure_std[i], rl_std[i])
            
            # 特殊处理：Franka Panda的J2标签位置下调避免与图例重叠
            if robot_name == 'Franka Panda' and i == 1:  # J2
                y_pos = max(p_err, r_err) + 8.0  # 降低位置
            
            if abs(improvement) > 1:  # 只显示改善超过1%的
                color = 'green' if improvement > 0 else 'red'
                ax.text(i + 1, y_pos, f'{improvement:+.1f}%', 
                       ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')
        
        ax.set_xlabel('Joint Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Absolute Error (degrees)', fontsize=12, fontweight='bold')
        ax.set_title(f'{robot_name} ({n_joints}-DOF)', fontsize=14, fontweight='bold')
        ax.set_xticks(joint_indices)
        ax.set_ylim(bottom=0)  # Y轴从0开始
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加总体MAE水平线
        pure_overall = pure_results['overall_mae']
        rl_overall = rl_results['overall_mae']
        ax.axhline(pure_overall, color='steelblue', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axhline(rl_overall, color='coral', linestyle='--', linewidth=1.5, alpha=0.5)
        
        # 在右侧标注总体MAE
        ax.text(n_joints + 0.5, pure_overall, f'Overall: {pure_overall:.2f}°', 
               va='center', fontsize=9, color='steelblue')
        ax.text(n_joints + 0.5, rl_overall, f'Overall: {rl_overall:.2f}°', 
               va='center', fontsize=9, color='coral')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存到: {save_path}")
    plt.close()


def generate_summary_table(results_dict):
    """生成汇总表格（Markdown格式）"""
    
    print("\n" + "="*80)
    print("汇总数据表格 (Markdown)")
    print("="*80)
    
    print("\n| 机器人平台 | DOF | Pure Meta-PID (°) | Meta-PID+RL (°) | 改善 (%) |")
    print("|------------|-----|-------------------|-----------------|----------|")
    
    for robot_name, data in results_dict.items():
        pure_results = data['pure']
        rl_results = data['rl']
        
        n_joints = pure_results['n_joints']
        pure_mae = pure_results['overall_mae']
        rl_mae = rl_results['overall_mae']
        improvement = (pure_mae - rl_mae) / pure_mae * 100
        
        print(f"| {robot_name:<10} | {n_joints:3d} | {pure_mae:17.2f} | {rl_mae:15.2f} | {improvement:8.1f} |")
    
    print("\n" + "="*80)
    print("详细逐关节数据")
    print("="*80)
    
    for robot_name, data in results_dict.items():
        pure_results = data['pure']
        rl_results = data['rl']
        
        print(f"\n### {robot_name} ({pure_results['n_joints']}-DOF)")
        print(f"\n| 关节 | Pure Meta-PID (°) | Meta-PID+RL (°) | 改善 (%) |")
        print("|------|-------------------|-----------------|----------|")
        
        for i in range(pure_results['n_joints']):
            pure_err = pure_results['per_joint_mean'][i]
            rl_err = rl_results['per_joint_mean'][i]
            improvement = (pure_err - rl_err) / pure_err * 100
            
            print(f"| J{i+1:2d}  | {pure_err:17.2f} | {rl_err:15.2f} | {improvement:8.1f} |")


def main():
    """主函数"""
    
    # 定义要评估的机器人
    robots = [
        {
            'urdf': 'franka_panda/panda.urdf',
            'name': 'Franka Panda',
            'rl_model': 'logs/meta_rl_panda/best_model/best_model'
        },
        {
            'urdf': 'laikago/laikago.urdf',
            'name': 'Laikago',
            'rl_model': 'logs/meta_rl_laikago/best_model/best_model'
        }
    ]
    
    print("="*80)
    print("逐关节误差对比评估")
    print("="*80)
    print(f"测试步数: 10000")
    print(f"评估平台: {len(robots)} 个")
    print()
    
    # 存储所有结果
    all_results = {}
    
    for robot in robots:
        robot_name = robot['name']
        robot_urdf = robot['urdf']
        rl_model_path = robot['rl_model']
        
        print(f"\n{'='*80}")
        print(f"开始评估: {robot_name}")
        print(f"{'='*80}")
        
        # 评估纯Meta-PID
        pure_results = evaluate_per_joint_error(
            robot_urdf=robot_urdf,
            robot_name=robot_name,
            model_path=None,
            steps=10000
        )
        
        # 评估Meta-PID + RL
        rl_results = evaluate_per_joint_error(
            robot_urdf=robot_urdf,
            robot_name=robot_name,
            model_path=rl_model_path,
            steps=10000
        )
        
        all_results[robot_name] = {
            'pure': pure_results,
            'rl': rl_results
        }
    
    # 生成汇总表格
    generate_summary_table(all_results)
    
    # 生成LaTeX表格
    generate_latex_table(all_results)
    
    # 绘制对比图
    plot_per_joint_comparison(all_results, save_path='per_joint_error_comparison.png')
    
    print("\n" + "="*80)
    print("✅ 所有评估完成！")
    print("="*80)
    print("\n生成的文件:")
    print("  1. per_joint_error_comparison.png  (对比图)")
    print("  2. per_joint_error_table.tex       (LaTeX表格)")
    print("\n这些可以直接用于论文！")


if __name__ == '__main__':
    main()

