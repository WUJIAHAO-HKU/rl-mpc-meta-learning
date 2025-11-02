#!/usr/bin/env python3
"""
Base Robot Environment for RL Training

This module provides the base class for robot simulation environments
using PyBullet physics engine.
"""

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path


class BaseRobotEnv(gym.Env):
    """
    Base class for robot control environments
    
    This class provides common functionality for robot simulation including:
    - PyBullet initialization
    - Robot loading from URDF
    - Trajectory generation
    - Reward computation
    - State observation
    
    Args:
        urdf_path (str): Path to robot URDF file
        n_joints (int): Number of controllable joints
        max_torque (float): Maximum torque per joint (N⋅m)
        control_freq (int): Control frequency (Hz)
        time_step (float): Simulation time step (s)
        render (bool): Whether to render GUI
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 240}
    
    def __init__(self, 
                 urdf_path,
                 n_joints,
                 max_torque=87.0,
                 control_freq=240,
                 time_step=0.001,
                 render=False):
        super(BaseRobotEnv, self).__init__()
        
        self.urdf_path = urdf_path
        self.n_joints = n_joints
        self.max_torque = max_torque
        self.control_freq = control_freq
        self.time_step = time_step
        self.render_mode = 'human' if render else None
        
        # Initialize PyBullet
        if render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(time_step)
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(n_joints,),
            dtype=np.float32
        )
        
        # Observation: [q, q_dot, q_ref, q_ref_dot, tracking_error]
        obs_dim = n_joints * 5
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Load robot
        self.robot_id = None
        self.joint_indices = []
        self._load_robot()
        
        # Trajectory
        self.trajectory = None
        self.trajectory_time = 0.0
        self.max_episode_steps = 1000
        self.current_step = 0
        
        # Tracking variables
        self.prev_tracking_error = np.zeros(n_joints)
    
    def _load_robot(self):
        """Load robot from URDF file"""
        if not Path(self.urdf_path).exists():
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")
        
        self.robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, 0],
            useFixedBase=True
        )
        
        # Get controllable joint indices
        self.joint_indices = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[2] != p.JOINT_FIXED:
                self.joint_indices.append(i)
        
        if len(self.joint_indices) != self.n_joints:
            print(f"Warning: Found {len(self.joint_indices)} joints, "
                  f"expected {self.n_joints}")
    
    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state
        
        Args:
            seed (int): Random seed
            options (dict): Additional options
        
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset simulation
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        
        # Reload robot
        self._load_robot()
        
        # Generate new trajectory
        self.trajectory = self._generate_trajectory()
        self.trajectory_time = 0.0
        self.current_step = 0
        
        # Reset to initial position
        initial_q = self.trajectory['q'][0]
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_idx, initial_q[i])
        
        # Step simulation
        for _ in range(10):
            p.stepSimulation()
        
        # Get initial observation
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action):
        """
        Execute one time step
        
        Args:
            action (np.ndarray): Control action (normalized torque)
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Scale action to actual torque
        torque = action * self.max_torque
        
        # Apply torque
        for i, joint_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.TORQUE_CONTROL,
                force=torque[i]
            )
        
        # Step simulation
        p.stepSimulation()
        
        # Update time
        self.trajectory_time += self.time_step
        self.current_step += 1
        
        # Get observation
        obs = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward(obs, action)
        
        # Check termination
        terminated = self._is_terminated(obs)
        truncated = self.current_step >= self.max_episode_steps
        
        # Additional info
        info = {
            'tracking_error': self._get_tracking_error(obs),
            'time': self.trajectory_time
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Get current observation"""
        # Get current joint states
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        q = np.array([state[0] for state in joint_states])
        q_dot = np.array([state[1] for state in joint_states])
        
        # Get reference trajectory
        q_ref, q_ref_dot = self._get_reference_trajectory(self.trajectory_time)
        
        # Compute tracking error
        tracking_error = q - q_ref
        
        # Concatenate observation
        obs = np.concatenate([q, q_dot, q_ref, q_ref_dot, tracking_error])
        
        return obs.astype(np.float32)
    
    def _compute_reward(self, obs, action):
        """
        Compute reward function
        
        Multi-objective reward:
        - Tracking accuracy (primary)
        - Action smoothness
        - Improvement over previous step
        """
        # Extract tracking error from observation
        tracking_error = obs[-self.n_joints:]
        
        # Tracking reward (negative L2 norm)
        r_tracking = -np.linalg.norm(tracking_error)
        
        # Smoothness penalty (small action is preferred)
        r_smoothness = -0.1 * np.linalg.norm(action)
        
        # Improvement reward
        error_reduction = np.linalg.norm(self.prev_tracking_error) - \
                         np.linalg.norm(tracking_error)
        r_improvement = 10.0 * error_reduction
        
        # Update previous error
        self.prev_tracking_error = tracking_error.copy()
        
        # Total reward
        reward = r_tracking + r_smoothness + r_improvement
        
        return reward
    
    def _is_terminated(self, obs):
        """Check if episode should terminate"""
        # Terminate if tracking error is too large
        tracking_error = obs[-self.n_joints:]
        max_error = np.max(np.abs(tracking_error))
        
        if max_error > np.pi:  # 180 degrees
            return True
        
        return False
    
    def _get_tracking_error(self, obs):
        """Extract tracking error from observation"""
        return obs[-self.n_joints:]
    
    def _generate_trajectory(self):
        """
        Generate reference trajectory
        
        Override this method in derived classes to implement
        specific trajectory types (quintic, cubic, sinusoidal, etc.)
        """
        raise NotImplementedError("Subclass must implement _generate_trajectory()")
    
    def _get_reference_trajectory(self, t):
        """
        Get reference position and velocity at time t
        
        Args:
            t (float): Current time
        
        Returns:
            tuple: (q_ref, q_ref_dot)
        """
        if self.trajectory is None:
            return np.zeros(self.n_joints), np.zeros(self.n_joints)
        
        # Interpolate trajectory
        idx = int(t / self.time_step)
        if idx >= len(self.trajectory['q']):
            idx = len(self.trajectory['q']) - 1
        
        q_ref = self.trajectory['q'][idx]
        q_ref_dot = self.trajectory['q_dot'][idx]
        
        return q_ref, q_ref_dot
    
    def close(self):
        """Clean up resources"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)


if __name__ == '__main__':
    print("Base robot environment module")
    print("This is a base class - use specific robot implementations")

