#!/usr/bin/env python3
"""
训练 Meta-PID + RL 组合控制器
从元学习预测的PID开始，RL进行微调优化
"""

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from pathlib import Path
import os
from meta_rl_combined_env import MetaRLCombinedEnv


def make_env(robot_urdf, rank=0):
    """创建环境"""
    def _init():
        return MetaRLCombinedEnv(
            robot_urdf=robot_urdf,
            gui=False,
            adjustment_range=0.2  # ±20%调整范围
        )
    return _init


def train_meta_rl(robot_urdf='franka_panda/panda.urdf', 
                  total_timesteps=1000000,  # 优化：从200k增加到1M (5倍训练量)
                  n_envs=8,
                  use_gpu=True):
    """
    训练Meta-PID + RL组合控制器
    
    Args:
        robot_urdf: 机器人URDF路径
        total_timesteps: 总训练步数 (默认1M，充分训练)
        n_envs: 并行环境数
        use_gpu: 是否使用GPU
    """
    print("=" * 80)
    print("训练 Meta-PID + RL 组合控制器")
    print("=" * 80)
    
    # 检测GPU
    device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
    print(f"\n🖥️  使用设备: {device}")
    
    # 创建日志目录
    robot_name = Path(robot_urdf).stem
    log_dir = Path(__file__).parent / 'logs' / f'meta_rl_{robot_name}'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 日志目录: {log_dir}")
    
    # 创建并行环境
    print(f"\n🔧 创建{n_envs}个并行环境...")
    
    if n_envs == 1:
        # 单环境
        env = DummyVecEnv([make_env(robot_urdf, 0)])
    else:
        # 多进程环境
        env = SubprocVecEnv([make_env(robot_urdf, i) for i in range(n_envs)])
    
    # 创建评估环境
    eval_env = DummyVecEnv([make_env(robot_urdf, 999)])
    
    print(f"✅ 环境创建成功")
    
    # 创建PPO模型 (优化超参数配置)
    print(f"\n🤖 创建PPO模型...")
    
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=1e-4,           # 优化：从3e-4降低到1e-4，更稳定的学习
        n_steps=2048,                 # 优化：每个环境2048步 (标准PPO配置)
        batch_size=256,               # 优化：从64增加到256，更稳定的梯度估计
        n_epochs=10,                  # 保持10轮更新
        gamma=0.99,                   # 折扣因子
        gae_lambda=0.95,              # GAE lambda
        clip_range=0.2,               # PPO裁剪范围
        ent_coef=0.02,                # 优化：从0.01增加到0.02，增加探索
        vf_coef=0.5,                  # 值函数损失系数
        max_grad_norm=0.5,            # 梯度裁剪
        verbose=1,
        device=device,
        tensorboard_log=str(log_dir / 'tensorboard')
    )
    
    print(f"✅ PPO模型创建成功 (优化配置)")
    print(f"   策略网络: MlpPolicy")
    print(f"   学习率: 1e-4 (降低以提高稳定性)")
    print(f"   Steps per env: 2048 (标准PPO配置)")
    print(f"   批次大小: 256 (增大以提高稳定性)")
    print(f"   熵系数: 0.02 (增加探索)")
    print(f"   总收集步数: {2048 * n_envs} per rollout")
    
    # 设置回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / 'best_model'),
        log_path=str(log_dir / 'eval_logs'),
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=str(log_dir / 'checkpoints'),
        name_prefix='meta_rl'
    )
    
    # 开始训练
    print(f"\n{'='*80}")
    print(f"🚀 开始训练...")
    print(f"{'='*80}")
    print(f"   总步数: {total_timesteps:,}")
    print(f"   并行环境: {n_envs}")
    print(f"   每轮收集: {2048 * n_envs:,} 步")
    print(f"   评估频率: 每10,000步")
    print(f"   检查点: 每20,000步")
    
    # 估算训练时间
    if total_timesteps <= 200000:
        est_time = "~30-45分钟"
    elif total_timesteps <= 500000:
        est_time = "~1.5-2.5小时"
    elif total_timesteps <= 1000000:
        est_time = "~3-5小时"
    else:
        est_time = "~5-10小时"
    
    print(f"\n⏰ 预计训练时间: {est_time}")
    print(f"   (取决于硬件性能，GPU可显著加速)")
    print(f"{'='*80}\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
        
        # 保存最终模型
        final_model_path = log_dir / f'meta_rl_{robot_name}_final.zip'
        model.save(final_model_path)
        
        print(f"\n{'='*80}")
        print(f"✅ 训练完成！")
        print(f"{'='*80}")
        print(f"\n📁 模型保存位置:")
        print(f"   最终模型: {final_model_path}")
        print(f"   最佳模型: {log_dir / 'best_model' / 'best_model.zip'}")
        print(f"   检查点: {log_dir / 'checkpoints'}")
        print(f"\n📊 查看训练曲线:")
        print(f"   tensorboard --logdir {log_dir / 'tensorboard'}")
        print(f"{'='*80}")
        
    except KeyboardInterrupt:
        print(f"\n⏸️  训练被中断")
    finally:
        # 清理资源
        print(f"\n🧹 清理资源...")
        env.close()
        eval_env.close()
        if device == 'cuda':
            torch.cuda.empty_cache()
        print(f"   ✅ 资源清理完成")


# ============================================================================
# 主程序
# ============================================================================
if __name__ == '__main__':
    import sys
    
    # 解析命令行参数
    robot_urdf = sys.argv[1] if len(sys.argv) > 1 else 'franka_panda/panda.urdf'
    total_timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 1000000  # 优化：默认1M步
    n_envs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    
    print("=" * 80)
    print("🎯 优化版本 RL训练脚本")
    print("=" * 80)
    print(f"📈 关键改进:")
    print(f"   • 训练步数: 200k → 1M (5倍)")
    print(f"   • 学习率: 3e-4 → 1e-4 (更稳定)")
    print(f"   • Steps/env: 256 → 2048 (8倍)")
    print(f"   • Batch size: 64 → 256 (4倍)")
    print(f"   • 熵系数: 0.01 → 0.02 (更多探索)")
    print(f"=" * 80)
    print(f"\n💡 使用方法:")
    print(f"   python {sys.argv[0]} [robot_urdf] [total_steps] [n_envs]")
    print(f"\n📝 示例:")
    print(f"   python {sys.argv[0]} franka_panda/panda.urdf 1000000 8")
    print(f"   python {sys.argv[0]} laikago/laikago.urdf 1000000 8")
    print(f"=" * 80)
    print(f"\n")
    
    # 训练
    train_meta_rl(
        robot_urdf=robot_urdf,
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        use_gpu=True
    )

