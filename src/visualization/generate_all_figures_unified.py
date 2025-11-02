#!/usr/bin/env python3
"""
统一生成所有图表和表格，确保数据一致性
同时生成：
1. Figure 3: Per-joint comparison (两个机器人平台)
2. Figure 4: Comprehensive tracking performance (Franka Panda详细分析)
3. LaTeX表格数据
"""

import numpy as np
import pybullet as p
import torch
from stable_baselines3 import PPO
from meta_rl_combined_env import MetaRLCombinedEnv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import json


def setup_publication_style():
    """设置出版级别的图表样式"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 13,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'patch.linewidth': 0.5,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'axes.grid': False,
        'grid.alpha': 0.3,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def evaluate_tracking_performance(robot_urdf, model_path=None, steps=10000, test_name=""):
    """
    评估跟踪性能
    """
    print(f"\n{'='*80}")
    print(f"评估: {test_name}")
    print(f"{'='*80}")
    
    # 创建环境
    env = MetaRLCombinedEnv(robot_urdf=robot_urdf, gui=False)
    
    # 加载RL模型（如果有）
    model = None
    if model_path is not None:
        try:
            model = PPO.load(model_path)
            print(f"✅ RL模型加载成功")
        except Exception as e:
            print(f"⚠️  RL模型加载失败: {e}")
            print(f"   使用固定Meta-PID")
    else:
        print(f"✅ 使用固定Meta-PID（无RL调整）")
    
    obs, _ = env.reset()
    
    # 记录数据
    actual_errors_deg = []  # 总误差 (角度)
    joint_errors = []  # 每个关节的误差 (弧度)
    
    for step in range(steps):
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
        
        # 计算实际误差
        joint_error = np.abs(q_ref - q_actual)  # 每个关节的绝对误差（弧度）
        actual_error_rad = np.linalg.norm(q_ref - q_actual)  # 总误差范数（弧度）
        actual_error_deg = np.degrees(actual_error_rad)  # 转换为角度
        
        actual_errors_deg.append(actual_error_deg)
        joint_errors.append(joint_error)
        
        if step % 2000 == 0:
            print(f"Step {step:5d}: error={actual_error_deg:.2f}°")
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    env.close()
    
    # 转换为numpy数组
    actual_errors_deg = np.array(actual_errors_deg)
    joint_errors = np.array(joint_errors)  # shape: (steps, n_joints)
    
    # 计算统计量
    overall_mae = np.mean(actual_errors_deg)
    overall_rmse = np.sqrt(np.mean(actual_errors_deg**2))
    overall_max = np.max(actual_errors_deg)
    
    # 计算每个关节的统计量
    joint_errors_deg = np.degrees(joint_errors)
    per_joint_mean = np.mean(joint_errors_deg, axis=0)
    per_joint_std = np.std(joint_errors_deg, axis=0)
    
    results = {
        'actual_errors_deg': actual_errors_deg,
        'joint_errors': joint_errors,  # 弧度
        'overall_mae': overall_mae,
        'overall_rmse': overall_rmse,
        'overall_max': overall_max,
        'per_joint_mean': per_joint_mean,
        'per_joint_std': per_joint_std,
        'n_joints': len(per_joint_mean)
    }
    
    print(f"\n📊 {test_name} 结果:")
    print(f"   总体MAE: {overall_mae:.2f}°")
    print(f"   总体RMSE: {overall_rmse:.2f}°")
    print(f"   最大误差: {overall_max:.2f}°")
    print(f"\n   各关节平均误差:")
    for i, (mean_err, std_err) in enumerate(zip(per_joint_mean, per_joint_std)):
        print(f"      关节 {i+1:2d}: {mean_err:6.2f}° ± {std_err:5.2f}°")
    
    return results


def generate_figure3(all_results, save_path='per_joint_error.png'):
    """
    生成Figure 3: Per-joint tracking error comparison across platforms
    """
    setup_publication_style()
    
    n_robots = len(all_results)
    fig, axes = plt.subplots(1, n_robots, figsize=(7*n_robots, 5))
    
    if n_robots == 1:
        axes = [axes]
    
    for idx, (robot_name, data) in enumerate(all_results.items()):
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
                       color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, rl_mean, width,
                       yerr=rl_std, capsize=3,
                       label='Meta-PID + RL', 
                       color='coral', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 在柱子上方标注改善百分比
        for i, (p_err, r_err) in enumerate(zip(pure_mean, rl_mean)):
            improvement = (p_err - r_err) / p_err * 100
            y_pos = max(p_err, r_err) + max(pure_std[i], rl_std[i])
            if abs(improvement) > 1:  # 只显示改善超过1%的
                color = 'green' if improvement > 0 else 'red'
                ax.text(i + 1, y_pos, f'{improvement:+.1f}%', 
                       ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')
        
        ax.set_xlabel('Joint Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Absolute Error (degrees)', fontsize=12, fontweight='bold')
        ax.set_title(f'{robot_name} ({n_joints}-DOF)', fontsize=14, fontweight='bold')
        ax.set_xticks(joint_indices)
        ax.set_ylim(0, None)  # Y轴从0开始，和子图b保持一致
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加总体MAE标注
        pure_overall = pure_results['overall_mae']
        rl_overall = rl_results['overall_mae']
        overall_improvement = (pure_overall - rl_overall) / pure_overall * 100
        
        ax.text(0.02, 0.98, f'Overall: {overall_improvement:+.1f}%', 
               transform=ax.transAxes, ha='left', va='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6),
               fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure 3已保存: {save_path}")
    plt.close()


def generate_figure4(pure_results, rl_results, save_path='Figure4_comprehensive_tracking_performance.png'):
    """
    生成Figure 4: Comprehensive tracking performance (Franka Panda详细分析)
    """
    setup_publication_style()
    
    # 创建2x2子图布局
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 颜色方案
    color_pure = '#4A90E2'  # 蓝色
    color_rl = '#F5A623'    # 橙色
    
    # ========================================================================
    # 子图 (a): Actual Tracking Error Comparison
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    # 平滑处理
    window = 100
    pure_smooth = np.convolve(pure_results['actual_errors_deg'], 
                              np.ones(window)/window, mode='valid')
    rl_smooth = np.convolve(rl_results['actual_errors_deg'], 
                            np.ones(window)/window, mode='valid')
    
    ax1.plot(pure_smooth, label='Pure Meta-PID', color=color_pure, alpha=0.8, linewidth=1.5)
    ax1.plot(rl_smooth, label='Meta-PID + RL', color=color_rl, alpha=0.8, linewidth=1.5)
    ax1.set_xlabel('Time Step', fontweight='bold')
    ax1.set_ylabel('Tracking Error (degrees)', fontweight='bold')
    ax1.set_title('(a) Actual Tracking Error Comparison', fontweight='bold', loc='left')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 计算改善百分比
    improvement = (pure_results['overall_mae'] - rl_results['overall_mae']) / pure_results['overall_mae'] * 100
    ax1.text(0.98, 0.02, f'{improvement:.1f}% improvement with RL adaptation', 
             transform=ax1.transAxes, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9, fontweight='bold')
    
    # ========================================================================
    # 子图 (b): Error Distribution
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    ax2.hist(pure_results['actual_errors_deg'], bins=50, alpha=0.6, 
            color=color_pure, label='Pure Meta-PID', density=True, edgecolor='black', linewidth=0.5)
    ax2.hist(rl_results['actual_errors_deg'], bins=50, alpha=0.6, 
            color=color_rl, label='Meta-PID + RL', density=True, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Tracking Error (degrees)', fontweight='bold')
    ax2.set_ylabel('Density', fontweight='bold')
    ax2.set_title('(b) Error Distribution', fontweight='bold', loc='left')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 添加均值线
    ax2.axvline(pure_results['overall_mae'], color=color_pure, linestyle='--', linewidth=2, alpha=0.7)
    ax2.axvline(rl_results['overall_mae'], color=color_rl, linestyle='--', linewidth=2, alpha=0.7)
    
    # ========================================================================
    # 子图 (c): Per-Joint Error Comparison with Improvement Curve (双Y轴)
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    # 计算各关节平均误差
    mean_joint_errors_pure = np.mean(pure_results['joint_errors'], axis=0)
    mean_joint_errors_rl = np.mean(rl_results['joint_errors'], axis=0)
    
    n_joints = len(mean_joint_errors_pure)
    x = np.arange(n_joints) + 1  # Joint indices starting from 1
    width = 0.35
    
    # 左Y轴：误差值柱状图
    bars1 = ax3.bar(x - width/2, np.degrees(mean_joint_errors_pure), width, 
                     label='Pure Meta-PID', color=color_pure, alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax3.bar(x + width/2, np.degrees(mean_joint_errors_rl), width, 
                     label='Meta-PID + RL', color=color_rl, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax3.set_xlabel('Joint Index', fontweight='bold')
    ax3.set_ylabel('Mean Absolute Error (degrees)', fontweight='bold', color='black')
    ax3.set_title('(c) Per-Joint Error Comparison', fontweight='bold', loc='left')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'J{i}' for i in x])
    ax3.tick_params(axis='y', labelcolor='black')
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 创建右Y轴：改进百分比曲线
    ax3_twin = ax3.twinx()
    
    # 计算每个关节的改进百分比
    improvement_percentages = []
    for i in range(n_joints):
        pure_err = np.degrees(mean_joint_errors_pure[i])
        rl_err = np.degrees(mean_joint_errors_rl[i])
        if pure_err > 0:
            improvement_pct = (pure_err - rl_err) / pure_err * 100
        else:
            improvement_pct = 0
        improvement_percentages.append(improvement_pct)
    
    improvement_percentages = np.array(improvement_percentages)
    
    # 绘制改进百分比曲线（使用深绿色）
    color_improvement = '#2E7D32'  # 深绿色
    line = ax3_twin.plot(x, improvement_percentages, 
                         color=color_improvement, marker='o', markersize=6,
                         linewidth=2.5, label='Improvement (%)', 
                         linestyle='-', alpha=0.9, zorder=10)
    
    # 在数据点上标注改善百分比（J2放在上方，其他放在下方）
    for i, (xi, yi) in enumerate(zip(x, improvement_percentages)):
        if abs(yi) > 1:  # 只显示改善超过1%的
            color_text = 'green' if yi > 0 else 'red'
            
            # J2（i=1，因为索引从0开始）放在曲线上方，其他放在下方
            if i == 1:  # J2
                y_offset = yi + 2.5
                va = 'bottom'
            else:  # 其他关节
                y_offset = yi - 3.0
                va = 'top'
            
            ax3_twin.text(xi, y_offset, f'{yi:+.1f}%', 
                         ha='center', va=va, fontsize=7, 
                         color=color_text, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                 edgecolor=color_text, alpha=0.7, linewidth=1))
    
    ax3_twin.set_ylabel('Improvement (%)', fontweight='bold', color=color_improvement)
    ax3_twin.tick_params(axis='y', labelcolor=color_improvement)
    ax3_twin.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
    
    # 设置右Y轴范围（为下方标注留出更多空间）
    max_abs_improvement = max(abs(improvement_percentages.min()), abs(improvement_percentages.max()))
    ax3_twin.set_ylim(-max_abs_improvement * 0.5, max_abs_improvement * 1.3)
    
    # 合并图例（放在中间上方，避免遮挡数据）
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper center',           # 位置：上方中间
              bbox_to_anchor=(0.5, 0.8),   # 精确位置：水平中心(0.5), 图表内部上方
              framealpha=0.95,              # 背景透明度
              fontsize=8,                   # 字体大小
              edgecolor='gray',             # 边框颜色
              fancybox=True)                # 圆角边框
    
    # 添加改善信息文本框
    joints_improved = np.sum(improvement_percentages > 0)
    avg_joint_improvement = np.mean(improvement_percentages[improvement_percentages > 0]) if joints_improved > 0 else 0
    max_improvement_joint = np.argmax(improvement_percentages) + 1
    max_improvement_value = improvement_percentages[np.argmax(improvement_percentages)]
    
    info_text = f'Joint {max_improvement_joint} benefits most: {max_improvement_value:.1f}% improvement\n{joints_improved}/{n_joints} joints improved, avg {avg_joint_improvement:.1f}%'
    ax3.text(0.98, 0.98, info_text,
             transform=ax3.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6, edgecolor='darkgreen'),
             fontsize=7, fontweight='bold')
    
    # ========================================================================
    # 子图 (d): Cumulative Distribution Function
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    pure_sorted = np.sort(pure_results['actual_errors_deg'])
    rl_sorted = np.sort(rl_results['actual_errors_deg'])
    pure_cdf = np.arange(1, len(pure_sorted)+1) / len(pure_sorted)
    rl_cdf = np.arange(1, len(rl_sorted)+1) / len(rl_sorted)
    
    ax4.plot(pure_sorted, pure_cdf, label='Pure Meta-PID', color=color_pure, linewidth=2, alpha=0.8)
    ax4.plot(rl_sorted, rl_cdf, label='Meta-PID + RL', color=color_rl, linewidth=2, alpha=0.8)
    ax4.set_xlabel('Tracking Error (degrees)', fontweight='bold')
    ax4.set_ylabel('Cumulative Probability', fontweight='bold')
    ax4.set_title('(d) Cumulative Distribution Function', fontweight='bold', loc='left')
    ax4.legend(loc='lower right', framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # 标注关键百分位数的改善
    percentiles = [50, 90]
    for pct in percentiles:
        idx = int(len(pure_sorted) * pct / 100)
        pure_val = pure_sorted[idx]
        rl_val = rl_sorted[idx]
        improvement = (pure_val - rl_val) / pure_val * 100
        ax4.axhline(pct/100, color='gray', linestyle=':', alpha=0.3)
        ax4.text(0.98, pct/100 - 0.05, f'P{pct}: {improvement:+.1f}%', 
                transform=ax4.transAxes, ha='right', va='bottom',
                fontsize=8, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure 4已保存: {save_path}")
    plt.close()


def generate_latex_table(all_results, save_path='per_joint_error_table.tex'):
    """
    生成LaTeX表格
    """
    print("\n" + "="*80)
    print("LaTeX表格代码")
    print("="*80)
    
    latex_code = """\\begin{table*}[!htbp]
\\caption{Per-Joint Tracking Error Comparison Across Platforms}
\\label{tab:per_joint_error}
\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lllll@{}}
\\toprule
\\textbf{Robot} & \\textbf{Joint} & \\textbf{Pure Meta-PID (°)} & \\textbf{Meta-PID+RL (°)} & \\textbf{Improv.} \\\\
\\midrule
"""
    
    for robot_name, data in all_results.items():
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
        
        if robot_name != list(all_results.keys())[-1]:
            latex_code += "\\midrule\n"
    
    latex_code += """\\bottomrule
\\end{tabular*}
\\end{table*}
"""
    
    print(latex_code)
    
    # 保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    print(f"✅ LaTeX表格已保存: {save_path}")
    
    return latex_code


def save_results_json(all_results, save_path='evaluation_results.json'):
    """
    保存评估结果为JSON（方便后续使用）
    """
    # 转换numpy数组为列表
    results_dict = {}
    for robot_name, data in all_results.items():
        results_dict[robot_name] = {}
        for method in ['pure', 'rl']:
            results_dict[robot_name][method] = {
                'overall_mae': float(data[method]['overall_mae']),
                'overall_rmse': float(data[method]['overall_rmse']),
                'overall_max': float(data[method]['overall_max']),
                'per_joint_mean': data[method]['per_joint_mean'].tolist(),
                'per_joint_std': data[method]['per_joint_std'].tolist(),
                'n_joints': int(data[method]['n_joints'])
            }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"✅ 评估结果已保存为JSON: {save_path}")


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
    print("统一评估：生成所有图表和表格")
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
        pure_results = evaluate_tracking_performance(
            robot_urdf=robot_urdf,
            model_path=None,
            steps=10000,
            test_name=f"{robot_name} - Pure Meta-PID"
        )
        
        # 评估Meta-PID + RL
        rl_results = evaluate_tracking_performance(
            robot_urdf=robot_urdf,
            model_path=rl_model_path,
            steps=10000,
            test_name=f"{robot_name} - Meta-PID + RL"
        )
        
        all_results[robot_name] = {
            'pure': pure_results,
            'rl': rl_results
        }
    
    # 生成所有图表和表格
    print("\n" + "="*80)
    print("生成图表和表格")
    print("="*80)
    
    # Figure 3: Per-joint comparison
    generate_figure3(all_results, save_path='per_joint_error.png')
    
    # Figure 4: Comprehensive tracking (Franka Panda only)
    generate_figure4(
        all_results['Franka Panda']['pure'],
        all_results['Franka Panda']['rl'],
        save_path='Figure4_comprehensive_tracking_performance.png'
    )
    
    # LaTeX表格
    generate_latex_table(all_results, save_path='per_joint_error_table.tex')
    
    # 保存JSON结果
    save_results_json(all_results, save_path='evaluation_results.json')
    
    # 打印汇总
    print("\n" + "="*80)
    print("✅ 所有评估完成！")
    print("="*80)
    print("\n生成的文件:")
    print("  1. per_joint_error.png  (Figure 3)")
    print("  2. Figure4_comprehensive_tracking_performance.png  (Figure 4)")
    print("  3. per_joint_error_table.tex  (LaTeX表格)")
    print("  4. evaluation_results.json  (评估结果数据)")
    print("\n📊 关键结果:")
    for robot_name, data in all_results.items():
        pure_mae = data['pure']['overall_mae']
        rl_mae = data['rl']['overall_mae']
        improvement = (pure_mae - rl_mae) / pure_mae * 100
        print(f"  {robot_name}: {improvement:+.1f}% improvement")


if __name__ == '__main__':
    main()

