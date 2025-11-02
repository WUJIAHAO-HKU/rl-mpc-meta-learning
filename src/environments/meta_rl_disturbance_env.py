#!/usr/bin/env python3
"""
支持扰动的Meta-PID+RL环境
扩展MetaRLCombinedEnv，添加多种扰动支持
"""

import numpy as np
import pybullet as p
from meta_rl_combined_env import MetaRLCombinedEnv


class MetaRLDisturbanceEnv(MetaRLCombinedEnv):
    """支持扰动的Meta-PID+RL环境"""
    
    def __init__(self, robot_urdf='franka_panda/panda.urdf', 
                 meta_model_path=None,
                 gui=False,
                 adjustment_range=0.2,
                 disturbance_type='none',
                 disturbance_params=None):
        """
        Args:
            disturbance_type: 扰动类型
                - 'none': 无扰动
                - 'random_force': 随机外力
                - 'payload': 负载变化
                - 'param_uncertainty': 参数不确定性
                - 'mixed': 混合扰动
            disturbance_params: 扰动参数字典
        """
        super().__init__(robot_urdf, meta_model_path, gui, adjustment_range)
        
        self.disturbance_type = disturbance_type
        self.disturbance_params = disturbance_params or {}
        
        # 默认扰动参数
        self.default_params = {
            'random_force': {
                'force_range': 20.0,  # 最大外力(N)
                'force_prob': 0.1,    # 施加外力的概率
            },
            'payload': {
                'mass_range': 3.0,    # 负载质量范围(kg)
            },
            'param_uncertainty': {
                'mass_scale': (0.8, 1.2),    # 质量不确定性±20%
                'friction_scale': (0.5, 2.0), # 摩擦不确定性
            }
        }
        
        # 合并参数（仅为当前扰动类型填充默认值，避免意外的扰动组合）
        # 只为当前disturbance_type相关的参数填充默认值
        relevant_keys = []
        if disturbance_type == 'random_force':
            relevant_keys = ['random_force']
        elif disturbance_type == 'payload':
            relevant_keys = ['payload']
        elif disturbance_type == 'param_uncertainty':
            relevant_keys = ['param_uncertainty']
        elif disturbance_type == 'mixed':
            # mixed包含所有三种扰动
            relevant_keys = ['random_force', 'payload', 'param_uncertainty']
        
        for key in relevant_keys:
            if key not in self.disturbance_params and key in self.default_params:
                self.disturbance_params[key] = self.default_params[key]
        
        # 扰动状态
        self.payload_applied = False
        self.param_modified = False
        
        # 应用扰动
        self._apply_disturbance()
    
    def _apply_disturbance(self):
        """应用扰动"""
        if self.disturbance_type == 'none':
            return
        
        elif self.disturbance_type == 'payload':
            self._apply_payload()
        
        elif self.disturbance_type == 'param_uncertainty':
            self._apply_param_uncertainty()
        
        elif self.disturbance_type == 'mixed':
            # 混合扰动：负载 + 参数不确定性
            self._apply_payload()
            self._apply_param_uncertainty()
    
    def _apply_payload(self):
        """应用负载（末端增加质量）"""
        if self.payload_applied:
            return
        
        params = self.disturbance_params['payload']
        mass = np.random.uniform(0, params['mass_range'])
        
        # 获取末端连杆ID（最后一个可控关节）
        end_effector_link = self.controllable_joints[-1]
        
        # 修改末端连杆质量
        dynamics_info = p.getDynamicsInfo(self.robot_id, end_effector_link)
        original_mass = dynamics_info[0]
        new_mass = original_mass + mass
        
        p.changeDynamics(
            self.robot_id,
            end_effector_link,
            mass=new_mass
        )
        
        self.payload_applied = True
        # print(f"✅ 应用负载: {mass:.2f}kg (原质量{original_mass:.2f}kg → 新质量{new_mass:.2f}kg)")
    
    def _apply_param_uncertainty(self):
        """应用参数不确定性"""
        if self.param_modified:
            return
        
        params = self.disturbance_params['param_uncertainty']
        
        # 对所有可控关节应用不确定性
        for joint_id in self.controllable_joints:
            # 质量不确定性
            dynamics_info = p.getDynamicsInfo(self.robot_id, joint_id)
            original_mass = dynamics_info[0]
            mass_scale = np.random.uniform(*params['mass_scale'])
            new_mass = original_mass * mass_scale
            
            # 摩擦不确定性
            friction_scale = np.random.uniform(*params['friction_scale'])
            lateral_friction = dynamics_info[1] * friction_scale
            
            p.changeDynamics(
                self.robot_id,
                joint_id,
                mass=new_mass,
                lateralFriction=lateral_friction
            )
        
        self.param_modified = True
        # print(f"✅ 应用参数不确定性: 质量{params['mass_scale']}, 摩擦{params['friction_scale']}")
    
    def _apply_random_force(self):
        """应用随机外力（每步调用）"""
        params = self.disturbance_params['random_force']
        
        # 以一定概率施加外力
        if np.random.random() < params['force_prob']:
            # 随机选择一个关节
            joint_id = np.random.choice(self.controllable_joints)
            
            # 随机力方向和大小
            force = np.random.uniform(-params['force_range'], params['force_range'], size=3)
            
            # 获取连杆位置
            link_state = p.getLinkState(self.robot_id, joint_id)
            position = link_state[0]
            
            # 施加外力
            p.applyExternalForce(
                self.robot_id,
                joint_id,
                forceObj=force,
                posObj=position,
                flags=p.WORLD_FRAME
            )
    
    def step(self, action):
        """重写step方法，在每步中应用随机外力"""
        # 应用随机外力（如果启用）
        if self.disturbance_type in ['random_force', 'mixed']:
            self._apply_random_force()
        
        # 调用父类step
        return super().step(action)
    
    def reset(self, seed=None, options=None):
        """重写reset方法，重新应用扰动"""
        obs, info = super().reset(seed, options)
        
        # 重置扰动状态
        self.payload_applied = False
        self.param_modified = False
        
        # 重新应用扰动
        self._apply_disturbance()
        
        return obs, info


def test_disturbance_env():
    """测试扰动环境"""
    print("="*80)
    print("测试扰动环境")
    print("="*80)
    
    disturbance_types = ['none', 'random_force', 'payload', 'param_uncertainty', 'mixed']
    
    for dist_type in disturbance_types:
        print(f"\n测试扰动类型: {dist_type}")
        
        env = MetaRLDisturbanceEnv(
            robot_urdf='franka_panda/panda.urdf',
            gui=False,
            disturbance_type=dist_type
        )
        
        obs, _ = env.reset()
        
        for step in range(100):
            action = np.zeros(2)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % 20 == 0:
                print(f"  Step {step}: reward={reward:.2f}, Kp={info['current_kp']:.2f}")
            
            if terminated or truncated:
                break
        
        env.close()
        print(f"✅ {dist_type} 测试完成")
    
    print("\n" + "="*80)
    print("✅ 所有扰动类型测试完成！")
    print("="*80)


if __name__ == '__main__':
    test_disturbance_env()

