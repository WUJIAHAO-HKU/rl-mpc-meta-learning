#!/usr/bin/env python3
"""
元学习PID + RL结合环境
使用元学习预测的PID作为初始值，RL进行微调
"""

import numpy as np
import pybullet as p
import pybullet_data
import torch
import torch.nn as nn
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces


# ============================================================================
# 加载元学习PID模型
# ============================================================================
class SimplePIDPredictor(nn.Module):
    """元学习PID预测器"""
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()
        )
    
    def forward(self, x):
        return self.network(x)


def load_meta_pid_model(model_path):
    """加载训练好的元学习PID模型"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = SimplePIDPredictor(input_dim=4, hidden_dim=64, output_dim=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['X_mean'], checkpoint['X_std'], checkpoint['y_mean'], checkpoint['y_std']


def predict_initial_pid(model, robot_features, X_mean, X_std, y_mean, y_std):
    """使用元学习模型预测初始PID"""
    # 标准化
    features_norm = (robot_features - X_mean) / X_std
    
    # 预测
    with torch.no_grad():
        features_t = torch.FloatTensor(features_norm).unsqueeze(0)
        pred_norm = model(features_t).squeeze(0).numpy()
    
    # 反标准化
    pred_log = pred_norm * y_std + y_mean
    pred = np.exp(pred_log)
    
    return pred  # [kp, ki, kd]


# ============================================================================
# Meta-PID + RL 环境
# ============================================================================
class MetaRLCombinedEnv(gym.Env):
    """
    结合元学习PID和RL的环境
    
    特点：
    1. 使用元学习预测的PID作为基准
    2. RL学习小范围调整（±20%）
    3. 快速收敛到最优性能
    """
    
    def __init__(self, robot_urdf='franka_panda/panda.urdf', 
                 meta_model_path=None,
                 gui=False,
                 adjustment_range=0.2):
        """
        Args:
            robot_urdf: 机器人URDF路径
            meta_model_path: 元学习模型路径
            gui: 是否显示GUI
            adjustment_range: RL调整范围（±20%）
        """
        super().__init__()
        
        self.robot_urdf = robot_urdf
        self.gui = gui
        self.adjustment_range = adjustment_range
        
        # 加载元学习模型
        if meta_model_path is None:
            meta_model_path = Path(__file__).parent / 'meta_pid_augmented.pth'
        
        print(f"🔧 加载元学习PID模型: {meta_model_path.name}")
        self.meta_model, self.X_mean, self.X_std, self.y_mean, self.y_std = load_meta_pid_model(meta_model_path)
        
        # 连接PyBullet
        if self.gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)
        
        # 加载机器人
        self.robot_id = p.loadURDF(robot_urdf, [0, 0, 0.5], useFixedBase=True)
        
        # 获取可控关节
        self.controllable_joints = []
        for j in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, j)
            if info[2] != p.JOINT_FIXED:
                self.controllable_joints.append(j)
        
        self.n_dof = len(self.controllable_joints)
        
        # 预测初始PID（元学习）
        robot_features = self._extract_robot_features()
        self.base_pid = predict_initial_pid(
            self.meta_model, robot_features,
            self.X_mean, self.X_std, self.y_mean, self.y_std
        )
        
        print(f"🤖 机器人: {robot_urdf}")
        print(f"   自由度: {self.n_dof}")
        print(f"   元学习预测初始PID:")
        print(f"      Kp = {self.base_pid[0]:.4f}")
        print(f"      Ki = {self.base_pid[1]:.4f}")
        print(f"      Kd = {self.base_pid[2]:.4f}")
        
        # 定义观测空间和动作空间
        # 观测：[q(n_dof), qd(n_dof), error(n_dof), time_in_episode]
        obs_dim = 3 * self.n_dof + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # 动作：[delta_kp_ratio, delta_kd_ratio]（±adjustment_range）
        # 实际PID = base_pid * (1 + delta_ratio)
        self.action_space = spaces.Box(
            low=-adjustment_range, high=adjustment_range, 
            shape=(2,), dtype=np.float32
        )
        
        # 轨迹生成参数
        self.max_steps = 2000  # 约8秒
        self.current_step = 0
        
    def _extract_robot_features(self):
        """提取机器人特征（用于元学习预测）"""
        # 简化：手动设置特征（实际应从URDF提取）
        if 'panda' in self.robot_urdf:
            features = np.array([9, 14.25, 6.55, 0.0])  # [DOF, mass, reach, payload]
        elif 'laikago' in self.robot_urdf:
            features = np.array([12, 11.45, 3.79, 0.0])
        elif 'kuka' in self.robot_urdf or 'iiwa' in self.robot_urdf:
            features = np.array([7, 17.5, 5.75, 0.0])
        else:
            features = np.array([self.n_dof, 15.0, 5.0, 0.0])  # 默认值
        
        return features.astype(np.float32)
    
    def reset(self, seed=None, options=None):
        """重置环境（gymnasium兼容）"""
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
        
        # 重置关节到初始位置
        for j in self.controllable_joints:
            p.resetJointState(self.robot_id, j, 0.0)
        
        self.current_step = 0
        
        obs = self._get_obs()
        info = {}
        
        return obs, info
    
    def _get_obs(self):
        """获取观测"""
        # 获取当前状态
        joint_states = p.getJointStates(self.robot_id, self.controllable_joints)
        q = np.array([s[0] for s in joint_states])
        qd = np.array([s[1] for s in joint_states])
        
        # 参考轨迹
        q_ref = self._get_reference_trajectory()
        
        # 误差
        error = q_ref - q
        
        # 时间归一化
        time_normalized = self.current_step / self.max_steps
        
        obs = np.concatenate([q, qd, error, [time_normalized]])
        
        return obs.astype(np.float32)
    
    def _get_reference_trajectory(self):
        """生成参考轨迹（正弦波）"""
        t = self.current_step * 1./240.
        q_ref = np.array([
            0.3 * np.sin(2 * np.pi * 0.5 * t + i * 0.5) 
            for i in range(self.n_dof)
        ])
        return q_ref
    
    def step(self, action):
        """执行动作"""
        # RL输出：调整比例
        delta_kp_ratio, delta_kd_ratio = action
        
        # 实际PID = 基准PID × (1 + 调整比例)
        current_kp = self.base_pid[0] * (1 + delta_kp_ratio)
        current_kd = self.base_pid[2] * (1 + delta_kd_ratio)
        
        # 使用POSITION_CONTROL应用PID
        q_ref = self._get_reference_trajectory()
        
        p.setJointMotorControlArray(
            self.robot_id,
            self.controllable_joints,
            p.POSITION_CONTROL,
            targetPositions=q_ref,
            positionGains=[current_kp] * self.n_dof,
            velocityGains=[current_kd] * self.n_dof,
            forces=[100.0] * self.n_dof
        )
        
        p.stepSimulation()
        
        # 获取新状态
        obs = self._get_obs()
        
        # 计算奖励
        joint_states = p.getJointStates(self.robot_id, self.controllable_joints)
        q = np.array([s[0] for s in joint_states])
        qd = np.array([s[1] for s in joint_states])
        error = q_ref - q
        
        # 奖励设计（归一化，避免数值爆炸）
        # 1. 归一化跟踪误差（除以sqrt(n_dof)使其与关节数无关）
        tracking_error_norm = np.linalg.norm(error) / np.sqrt(self.n_dof)
        
        # 2. 归一化速度和动作
        velocity_norm = np.linalg.norm(qd) / np.sqrt(self.n_dof)
        action_norm = np.linalg.norm(action)
        
        # 3. 计算奖励（权重更合理）
        reward = (
            -10.0 * tracking_error_norm   # 主要：跟踪误差（归一化后）
            -0.1 * velocity_norm          # 次要：速度平滑
            -0.1 * action_norm            # 次要：动作平滑
        )
        
        # 4. 奖励裁剪（避免极端情况）
        reward = np.clip(reward, -100.0, 10.0)
        
        self.current_step += 1
        
        # gymnasium格式：separated terminated and truncated
        terminated = False  # 没有明确的终止条件
        truncated = (self.current_step >= self.max_steps)  # 时间步数达到上限
        
        info = {
            'tracking_error': tracking_error_norm,  # 归一化后的误差
            'current_kp': current_kp,
            'current_kd': current_kd,
        }
        
        return obs, reward, terminated, truncated, info
    
    def close(self):
        """关闭环境"""
        p.disconnect(self.client)


# ============================================================================
# 测试
# ============================================================================
if __name__ == '__main__':
    print("=" * 80)
    print("测试 Meta-PID + RL 组合环境")
    print("=" * 80)
    
    # 创建环境
    env = MetaRLCombinedEnv(
        robot_urdf='franka_panda/panda.urdf',
        gui=False,
        adjustment_range=0.2  # ±20%调整范围
    )
    
    print(f"\n✅ 环境创建成功")
    print(f"   观测空间: {env.observation_space.shape}")
    print(f"   动作空间: {env.action_space.shape}")
    
    # 测试reset
    obs = env.reset()
    print(f"\n📊 初始观测形状: {obs.shape}")
    
    # 测试step（随机动作）
    print(f"\n🎮 测试10步...")
    for i in range(10):
        action = env.action_space.sample()  # 随机调整
        obs, reward, done, info = env.step(action)
        
        if i % 5 == 0:
            print(f"   Step {i}: reward={reward:.2f}, "
                  f"error={info['tracking_error']:.4f}, "
                  f"Kp={info['current_kp']:.2f}")
    
    env.close()
    print(f"\n✅ 测试完成！")
    
    print(f"\n🎯 下一步:")
    print(f"   python train_meta_rl_combined.py")

