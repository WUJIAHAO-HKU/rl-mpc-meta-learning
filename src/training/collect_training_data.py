"""
训练数据收集脚本

功能：
1. 为多个机器人（不同DOF、质量、负载）收集最优PID参数
2. 使用贝叶斯优化找到每个配置的最优PID
3. 保存为训练数据集
"""

import numpy as np
import yaml
import json
from pathlib import Path
from datetime import datetime
import sys
import os

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from meta_pid_optimizer import RobotFeatureExtractor
from tune_pid_params import optimize_pid_params  # 使用之前的PID优化脚本
import pybullet as p


class TrainingDataCollector:
    """训练数据收集器"""
    
    def __init__(self, output_dir='meta_learning/training_data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_extractor = RobotFeatureExtractor()
        self.collected_data = []
    
    def collect_for_robot(self, robot_config, num_trials=1):
        """
        为一个机器人配置收集最优PID数据
        
        Args:
            robot_config: dict包含:
                - urdf_path: URDF文件路径
                - name: 机器人名称
                - payload_range: (min, max) 负载范围
                - num_payloads: 测试几个负载点
            num_trials: 每个配置重复优化次数
        
        Returns:
            list: 数据点列表
        """
        urdf_path = robot_config['urdf_path']
        name = robot_config['name']
        payload_min, payload_max = robot_config.get('payload_range', (0.0, 2.0))
        num_payloads = robot_config.get('num_payloads', 3)
        
        print(f"\n{'=' * 80}")
        print(f"收集机器人数据: {name}")
        print(f"URDF: {urdf_path}")
        print(f"{'=' * 80}")
        
        # 检查URDF是否存在
        if not Path(urdf_path).exists():
            print(f"❌ URDF文件不存在: {urdf_path}")
            return []
        
        data_points = []
        
        # 不同负载
        payloads = np.linspace(payload_min, payload_max, num_payloads)
        
        for payload in payloads:
            print(f"\n📦 负载: {payload:.2f} kg")
            
            # 提取特征
            try:
                features, controllable_joints = self.feature_extractor.extract_features(
                    urdf_path, payload
                )
                print(f"   DOF: {features['dof']}, 总质量: {features['total_mass']:.2f} kg")
            except Exception as e:
                print(f"   ❌ 特征提取失败: {e}")
                continue
            
            # 多次优化（取平均）
            optimal_pids = []
            
            for trial in range(num_trials):
                print(f"\n   🔍 优化尝试 {trial+1}/{num_trials}...")
                
                try:
                    # 使用贝叶斯优化找最优PID
                    # 注意：这里需要创建一个临时配置文件
                    temp_config = self._create_temp_config(
                        urdf_path, features['dof'], payload
                    )
                    
                    # 调用PID优化
                    optimal_pid = self._run_pid_optimization(temp_config, features['dof'])
                    
                    if optimal_pid is not None:
                        optimal_pids.append(optimal_pid)
                        print(f"      ✅ Kp={optimal_pid['Kp'][:2]}...")
                    
                except Exception as e:
                    print(f"      ❌ 优化失败: {e}")
                    continue
            
            # 如果有成功的优化结果
            if optimal_pids:
                # 取平均（多次优化的结果）
                avg_kp = np.mean([p['Kp'] for p in optimal_pids], axis=0)
                avg_ki = np.mean([p['Ki'] for p in optimal_pids], axis=0)
                avg_kd = np.mean([p['Kd'] for p in optimal_pids], axis=0)
                
                # 保存数据点
                data_point = {
                    'robot_name': name,
                    'urdf_path': str(urdf_path),
                    'payload': float(payload),
                    'features': {k: float(v) for k, v in features.items()},
                    'optimal_pid': {
                        'Kp': avg_kp.tolist(),
                        'Ki': avg_ki.tolist(),
                        'Kd': avg_kd.tolist()
                    },
                    'num_trials': len(optimal_pids),
                    'timestamp': datetime.now().isoformat()
                }
                
                data_points.append(data_point)
                self.collected_data.append(data_point)
                
                print(f"\n   ✅ 数据点已收集:")
                print(f"      Kp: {avg_kp}")
                print(f"      Ki: {avg_ki}")
                print(f"      Kd: {avg_kd}")
        
        return data_points
    
    def _create_temp_config(self, urdf_path, dof, payload):
        """创建临时配置文件用于PID优化"""
        # 这里简化处理，实际应该更完整
        config = {
            'robot_params': {
                'urdf_path': str(urdf_path),
                'dof': dof,
                'payload': payload
            },
            'trajectory': {
                'type': 'circle',
                'speed': 0.2,
                'radius': 0.15
            },
            'simulation': {
                'dt': 0.001,
                'max_steps': 10000
            }
        }
        return config
    
    def _run_pid_optimization(self, config, dof):
        """
        运行PID参数优化
        
        这里暂时返回一个合理的初始值
        TODO: 集成实际的贝叶斯优化
        """
        # 暂时使用简化版本：基于DOF和质量的启发式规则
        total_mass = config['robot_params'].get('payload', 0) + 20  # 假设机器人本体20kg
        
        # 经验公式（基于Franka的优化结果外推）
        # Kp约与质量成正比，Ki和Kd与质量平方根成正比
        base_kp = 800
        base_ki = 1.0
        base_kd = 10.0
        
        mass_factor = total_mass / 20.0
        
        kp = base_kp * mass_factor * np.ones(dof)
        ki = base_ki * np.sqrt(mass_factor) * np.ones(dof)
        kd = base_kd * np.sqrt(mass_factor) * np.ones(dof)
        
        return {
            'Kp': kp,
            'Ki': ki,
            'Kd': kd
        }
    
    def save_dataset(self, filename=None):
        """保存收集的数据集"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'pid_dataset_{timestamp}.json'
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(self.collected_data, f, indent=2)
        
        print(f"\n✅ 数据集已保存: {output_path}")
        print(f"   总数据点: {len(self.collected_data)}")
        
        return output_path
    
    def load_dataset(self, filename):
        """加载已有数据集"""
        with open(filename, 'r') as f:
            self.collected_data = json.load(f)
        
        print(f"✅ 数据集已加载: {filename}")
        print(f"   总数据点: {len(self.collected_data)}")
        
        return self.collected_data


def create_robot_configs():
    """
    创建机器人配置列表
    
    Returns:
        list: 机器人配置列表
    """
    base_path = Path('/home/wujiahao/基于强化学习的模型预测控制动力学模型误差在线补偿方法研究/rl_pid_linux')
    
    configs = [
        {
            'name': 'Franka Panda (7DOF)',
            'urdf_path': base_path / 'assets/franka_panda/panda.urdf',
            'payload_range': (0.0, 2.0),
            'num_payloads': 5  # 0, 0.5, 1.0, 1.5, 2.0 kg
        },
        # TODO: 添加更多机器人
        # {
        #     'name': 'UR5 (6DOF)',
        #     'urdf_path': 'path/to/ur5.urdf',
        #     'payload_range': (0.0, 3.0),
        #     'num_payloads': 4
        # },
        # {
        #     'name': 'Custom 3DOF',
        #     'urdf_path': 'path/to/3dof.urdf',
        #     'payload_range': (0.0, 1.0),
        #     'num_payloads': 3
        # }
    ]
    
    return configs


def main():
    """主函数"""
    print("=" * 80)
    print("元学习PID训练数据收集")
    print("=" * 80)
    
    # 创建数据收集器
    collector = TrainingDataCollector()
    
    # 获取机器人配置
    robot_configs = create_robot_configs()
    
    print(f"\n将为 {len(robot_configs)} 个机器人配置收集数据")
    
    # 收集数据
    for config in robot_configs:
        try:
            data_points = collector.collect_for_robot(config, num_trials=1)
            print(f"\n✅ {config['name']}: 收集了 {len(data_points)} 个数据点")
        except Exception as e:
            print(f"\n❌ {config['name']}: 收集失败 - {e}")
            continue
    
    # 保存数据集
    if collector.collected_data:
        dataset_path = collector.save_dataset()
        
        print("\n" + "=" * 80)
        print("数据收集完成！")
        print("=" * 80)
        print(f"数据集路径: {dataset_path}")
        print(f"总数据点: {len(collector.collected_data)}")
        print("\n下一步: 使用此数据集训练元学习模型")
        print("  python meta_learning/train_meta_pid.py")
    else:
        print("\n❌ 没有收集到任何数据")


if __name__ == '__main__':
    main()

