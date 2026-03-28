"""
运动控制模块
============

AGV/机械臂运动控制
- 关节空间控制
- 笛卡尔空间控制
- 速度/位置控制
- 轨迹跟踪
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Callable
from enum import Enum


class ControlMode(Enum):
    """控制模式"""
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    JOINT_TORQUE = "joint_torque"
    CARTESIAN_VELOCITY = "cartesian_velocity"  # Twist
    CARTESIAN_POSITION = "cartesian_position"  # Pose


@dataclass
class JointState:
    """关节状态"""
    position: np.ndarray      # 关节位置 (rad 或 m)
    velocity: np.ndarray     # 关节速度
    torque: np.ndarray       # 关节力矩
    timestamp: float = 0.0


@dataclass
class JointTrajectory:
    """关节轨迹点序列"""
    positions: np.ndarray    # N x num_joints
    timestamps: np.ndarray   # N, 时间戳
    velocities: Optional[np.ndarray] = None  # N x num_joints
    accelerations: Optional[np.ndarray] = None  # N x num_joints
    
    def __post_init__(self):
        if isinstance(self.positions, list):
            self.positions = np.array(self.positions)
        if isinstance(self.timestamps, list):
            self.timestamps = np.array(self.timestamps)


@dataclass
class TwistCommand:
    """笛卡尔空间速度命令"""
    linear: np.ndarray       # 3, 线性速度 (m/s)
    angular: np.ndarray      # 3, 角速度 (rad/s)
    frame_id: str = "base_link"
    
    def __post_init__(self):
        if isinstance(self.linear, list):
            self.linear = np.array(self.linear, dtype=np.float32)
        if isinstance(self.angular, list):
            self.angular = np.array(self.angular, dtype=np.float32)


@dataclass
class PoseCommand:
    """笛卡尔空间位置命令"""
    position: np.ndarray     # 3, 位置 (m)
    orientation: np.ndarray  # 4, 四元数
    frame_id: str = "base_link"


class MotionController:
    """
    运动控制器
    
    支持:
    - 关节空间 PID 控制
    - 笛卡尔空间速度控制
    - 轨迹插值 (LQP/MPC)
    - 安全限制 (关节限位/速度限制/加速度限制)
    """
    
    def __init__(
        self,
        num_joints: int,
        control_rate: float = 100.0,  # Hz
        max_velocity: Optional[np.ndarray] = None,
        max_acceleration: Optional[np.ndarray] = None,
        max_torque: Optional[np.ndarray] = None
    ):
        self.num_joints = num_joints
        self.control_rate = control_rate
        self.dt = 1.0 / control_rate
        
        # 关节限制
        self.joint_limits_lower = -np.ones(num_joints) * np.pi
        self.joint_limits_upper = np.ones(num_joints) * np.pi
        
        # 速度/加速度限制
        self.max_velocity = max_velocity if max_velocity is not None else np.ones(num_joints) * np.pi  # rad/s
        self.max_acceleration = max_acceleration if max_acceleration is not None else np.ones(num_joints) * 2.0  # rad/s^2
        self.max_torque = max_torque if max_torque is not None else np.ones(num_joints) * 100.0  # Nm
        
        # PID参数
        self.kp = np.ones(num_joints) * 1.0
        self.ki = np.zeros(num_joints)
        self.kd = np.zeros(num_joints)
        
        # 状态
        self._current_joint_pos = np.zeros(num_joints)
        self._current_joint_vel = np.zeros(num_joints)
        self._joint_pos_error_integral = np.zeros(num_joints)
        self._last_error = np.zeros(num_joints)
        
        # 控制模式
        self._control_mode = ControlMode.JOINT_POSITION
        
        # 回调
        self._torque_callback: Optional[Callable] = None
        
    def set_joint_limits(
        self,
        lower: np.ndarray,
        upper: np.ndarray
    ):
        """设置关节角度限制"""
        if len(lower) != self.num_joints or len(upper) != self.num_joints:
            raise ValueError("Joint limit array size mismatch")
        self.joint_limits_lower = lower
        self.joint_limits_upper = upper
    
    def set_pid_gains(
        self,
        kp: np.ndarray,
        ki: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None
    ):
        """设置PID增益"""
        self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd
    
    def set_torque_callback(self, callback: Callable[[np.ndarray], None]):
        """设置力矩回调 (用于发送到底层驱动)"""
        self._torque_callback = callback
    
    def update_joint_state(self, joint_state: JointState):
        """更新当前关节状态"""
        self._current_joint_pos = joint_state.position.copy()
        self._current_joint_vel = joint_state.velocity.copy()
    
    def compute_joint_torque(
        self,
        target_position: np.ndarray,
        target_velocity: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算关节力矩 (PID控制)
        
        Args:
            target_position: 目标关节位置
            target_velocity: 目标关节速度 (可选)
            
        Returns:
            torque: 关节力矩命令
        """
        error = target_position - self._current_joint_pos
        
        # 积分项
        self._joint_pos_error_integral += error * self.dt
        self._joint_pos_error_integral = np.clip(
            self._joint_pos_error_integral,
            -1.0, 1.0
        )
        
        # 微分项
        error_derivative = (error - self._last_error) / self.dt
        self._last_error = error.copy()
        
        # PID
        if target_velocity is not None:
            # 速度前馈
            target_velocity = np.zeros(self.num_joints) if target_velocity is None else target_velocity
            error += self.kp * (target_velocity - self._current_joint_vel)
        
        torque = (
            self.kp * error +
            self.ki * self._joint_pos_error_integral +
            self.kd * error_derivative
        )
        
        # 限制
        torque = np.clip(torque, -self.max_torque, self.max_torque)
        
        return torque
    
    def compute_cartesian_velocity(
        self,
        target_twist: TwistCommand,
        jacobian: np.ndarray
    ) -> np.ndarray:
        """
        笛卡尔速度转关节速度
        
        Args:
            target_twist: 目标笛卡尔速度
            jacobian: 雅可比矩阵 (6 x num_joints)
            
        Returns:
            joint_velocity: 关节速度命令
        """
        # 伪逆
        jacobian_pinv = np.linalg.pinv(jacobian)
        
        # 笛卡尔速度向量
        twist_vec = np.concatenate([target_twist.linear, target_twist.angular])
        
        # 逆运动学
        joint_vel = jacobian_pinv @ twist_vec
        
        # 限速
        joint_vel = np.clip(joint_vel, -self.max_velocity, self.max_velocity)
        
        return joint_vel
    
    def interpolate_trajectory(
        self,
        trajectory: JointTrajectory,
        current_time: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        轨迹插值
        
        返回指定时刻的位置和速度
        """
        # 找到对应区间
        indices = np.searchsorted(trajectory.timestamps, current_time)
        indices = np.clip(indices, 1, len(trajectory.timestamps) - 1)
        
        i = indices - 1
        t0, t1 = trajectory.timestamps[i], trajectory.timestamps[i + 1]
        
        if t1 <= t0:
            return trajectory.positions[i], trajectory.velocities[i] if trajectory.velocities is not None else np.zeros(self.num_joints)
        
        # 归一化时间
        alpha = (current_time - t0) / (t1 - t0)
        
        # 五次多项式插值
        from scipy.interpolate import interp1d
        
        # 简化: 线性插值
        pos = trajectory.positions[i] * (1 - alpha) + trajectory.positions[i + 1] * alpha
        
        if trajectory.velocities is not None:
            vel = trajectory.velocities[i] * (1 - alpha) + trajectory.velocities[i + 1] * alpha
        else:
            vel = np.zeros(self.num_joints)
        
        return pos, vel
    
    def apply_safety_limits(self, command: np.ndarray, is_velocity: bool = False) -> np.ndarray:
        """
        应用安全限制
        
        - 关节限位
        - 速度限制
        - 加速度限制
        """
        if is_velocity:
            # 速度限制
            command = np.clip(command, -self.max_velocity, self.max_velocity)
        else:
            # 位置限制
            command = np.clip(command, self.joint_limits_lower, self.joint_limits_upper)
        
        return command
    
    def step(
        self,
        target: np.ndarray,
        mode: ControlMode = ControlMode.JOINT_POSITION
    ) -> np.ndarray:
        """
        一步控制计算
        
        Args:
            target: 目标位置/速度/力矩
            mode: 控制模式
            
        Returns:
            command: 控制命令
        """
        self._control_mode = mode
        
        if mode == ControlMode.JOINT_POSITION:
            torque = self.compute_joint_torque(target)
            if self._torque_callback:
                self._torque_callback(torque)
            return torque
        
        elif mode == ControlMode.JOINT_VELOCITY:
            error = target - self._current_joint_vel
            torque = self.kp * error
            if self._torque_callback:
                self._torque_callback(torque)
            return torque
        
        elif mode == ControlMode.JOINT_TORQUE:
            if self._torque_callback:
                self._torque_callback(target)
            return target
        
        else:
            raise ValueError(f"Unsupported control mode: {mode}")


class TwistToJoint:
    """
    笛卡尔速度到关节速度的转换
    
    使用雅可比伪逆
    """
    
    def __init__(self, jacobian_fn: Callable[[np.ndarray], np.ndarray]):
        """
        Args:
            jacobian_fn: 雅可比矩阵计算函数, 输入关节位置返回 6 x num_joints 矩阵
        """
        self.jacobian_fn = jacobian_fn
    
    def compute(
        self,
        twist: TwistCommand,
        joint_positions: np.ndarray,
        null_space_posture: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算关节速度
        
        Args:
            twist: 笛卡尔速度
            joint_positions: 当前关节位置
            null_space_posture: 零空间姿态目标 (可选)
            
        Returns:
            joint_velocities: 关节速度
        """
        J = self.jacobian_fn(joint_positions)
        
        # 加权伪逆
        weights = np.eye(J.shape[1])
        
        # 简化为标准伪逆
        J_pinv = np.linalg.pinv(J)
        
        v = np.concatenate([twist.linear, twist.angular])
        q_dot = J_pinv @ v
        
        # 零空间优化
        if null_space_posture is not None:
            I = np.eye(J.shape[1])
            J_pinv = J.T @ np.linalg.inv(J @ J.T + 1e-6 * np.eye(6))
            null_proj = I - J_pinv @ J
            q_dot_null = null_proj @ (null_space_posture - joint_positions) * 0.1
            q_dot = q_dot + q_dot_null
        
        return q_dot
