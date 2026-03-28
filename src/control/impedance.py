"""
阻抗控制模块
============

柔顺控制实现
- 位置阻抗控制
- 力阻抗控制
- 力位混合控制
- 协作控制
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Callable


@dataclass
class ImpedanceParams:
    """阻抗参数"""
    # 惯性参数
    M: np.ndarray                          # 6x6 惯性矩阵
    # 阻尼参数
    D: np.ndarray                           # 6x6 阻尼矩阵
    # 刚度参数
    K: np.ndarray                           # 6x6 刚度矩阵
    
    def __post_init__(self):
        if isinstance(self.M, (list, tuple)):
            self.M = np.array(self.M, dtype=np.float32)
        if isinstance(self.D, (list, tuple)):
            self.D = np.array(self.D, dtype=np.float32)
        if isinstance(self.K, (list, tuple)):
            self.K = np.array(self.K, dtype=np.float32)
    
    @classmethod
    def default_6d(cls) -> 'ImpedanceParams':
        """默认6维阻抗参数"""
        # 低刚度用于协作
        M_d = np.eye(6) * 5.0
        D_d = np.eye(6) * 50.0
        K_d = np.eye(6) * 200.0
        return cls(M=M_d, D=D_d, K=K_d)
    
    @classmethod
    def high_stiffness(cls) -> 'ImpedanceParams':
        """高刚度参数"""
        M_d = np.eye(6) * 10.0
        D_d = np.eye(6) * 100.0
        K_d = np.eye(6) * 1000.0
        return cls(M=M_d, D=D_d, K=K_d)


class ImpedanceController:
    """
    阻抗控制器
    
    实现: M*Xdd + D*Xd + K*X = F
    其中 X 是位置误差, F 是外力
    """
    
    def __init__(
        self,
        impedance_params: Optional[ImpedanceParams] = None,
        control_rate: float = 100.0
    ):
        """
        Args:
            impedance_params: 阻抗参数
            control_rate: 控制频率 Hz
        """
        self.params = impedance_params or ImpedanceParams.default_6d()
        self.dt = 1.0 / control_rate
        
        # 状态
        self._error_prev = np.zeros(6)
        self._error_dot_prev = np.zeros(6)
        self._desired_pos = np.zeros(3)
        self._desired_orientation = np.eye(3)
        
    def set_impedance_params(self, params: ImpedanceParams):
        """设置阻抗参数"""
        self.params = params
    
    def compute_torque(
        self,
        desired_position: np.ndarray,
        desired_velocity: np.ndarray,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        external_wrench: np.ndarray,
        jacobian: np.ndarray
    ) -> np.ndarray:
        """
        计算关节力矩
        
        Args:
            desired_position: 目标位置 (m)
            desired_velocity: 目标速度 (m/s)
            current_position: 当前位置 (m)
            current_velocity: 当前速度 (m/s)
            external_wrench: 外力旋量 (6,) [Fx, Fy, Fz, Tx, Ty, Tz]
            jacobian: 雅可比矩阵 (6 x n_joints)
            
        Returns:
            joint_torques: 关节力矩 (n_joints,)
        """
        # 位置误差
        pos_error = current_position - desired_position
        
        # 速度误差
        vel_error = current_velocity - desired_velocity
        
        # 构建误差向量 (6维: 线速度 + 角速度误差)
        # 简化: 只考虑线速度
        error = np.zeros(6)
        error[:3] = pos_error
        error[3:] = vel_error
        
        # 误差微分
        error_dot = (error - self._error_prev) / self.dt
        self._error_prev = error
        self._error_dot_prev = error_dot
        
        # 阻抗方程: M*Xdd + D*Xd + K*X = F
        # 稳态: F = K*X + D*Xd (忽略惯性)
        
        # 计算阻抗力
        impedance_force = (
            self.params.K @ error[:3] +
            self.params.D[:3, :3] @ error_dot[:3]
        )
        
        # 外力补偿
        compensated_force = external_wrench[:3] - impedance_force
        
        # 转换到关节空间
        jacobian_transpose = jacobian[:3, :].T
        joint_torques = jacobian_transpose @ compensated_force
        
        return joint_torques
    
    def compute_cartesian_force(
        self,
        desired_pose: np.ndarray,      # 6D pose error
        desired_velocity: np.ndarray,   # 6D velocity error
        external_wrench: np.ndarray,
        use_quaternion: bool = False
    ) -> np.ndarray:
        """
        计算笛卡尔空间力 (用于直接力控制)
        
        Args:
            desired_pose: 目标位姿误差 [dx, dy, dz, droll, dpitch, dyaw]
            desired_velocity: 目标速度误差
            external_wrench: 外力
            use_quaternion: 是否使用四元数
            
        Returns:
            cartesian_force: 笛卡尔力
        """
        # 阻抗力
        M = self.params.M
        D = self.params.D
        K = self.params.K
        
        # 稳态阻抗力
        if desired_velocity is None:
            desired_velocity = np.zeros(6)
        
        impedance_force = (
            K @ desired_pose +
            D @ desired_velocity
        )
        
        # 加上外力补偿
        total_force = impedance_force + external_wrench
        
        return total_force


class AdmittanceController:
    """
    导纳控制器
    
    与阻抗控制相反:
    输入力 -> 输出位置调整
    
    用于: 协作机器人、力适应场景
    """
    
    def __init__(
        self,
        M: float = 10.0,      # 虚拟质量
        D: float = 50.0,      # 虚拟阻尼
        K: float = 200.0,     # 虚拟刚度
        control_rate: float = 100.0
    ):
        self.M = M
        self.D = D
        self.K = K
        self.dt = 1.0 / control_rate
        
        # 状态
        self._velocity = 0.0
        self._position = 0.0
        
    def update(
        self,
        external_force: float,
        desired_position: float,
        dt: Optional[float] = None
    ) -> float:
        """
        更新导纳控制
        
        Args:
            external_force: 外力
            desired_position: 期望位置
            dt: 时间步长
            
        Returns:
            adjusted_position: 调整后的位置
        """
        if dt is not None:
            self.dt = dt
        
        # 导纳方程: M*acc + D*vel + K*pos = F
        # 求 acc = (F - D*vel - K*pos) / M
        
        pos_error = self._position - desired_position
        acc = (external_force - self.D * self._velocity - self.K * pos_error) / self.M
        
        # 积分
        self._velocity += acc * self.dt
        self._position += self._velocity * self.dt
        
        return self._position
    
    def reset(self):
        """重置状态"""
        self._velocity = 0.0
        self._position = 0.0


class ForceImpedanceController:
    """
    力位混合控制器
    
    在不同方向分别使用力控制和位置控制
    """
    
    def __init__(
        self,
        force_axes: np.ndarray,       # 力控方向 (6维, 1=力控, 0=位置控)
        Kp: float = 100.0,
        Kf: float = 1.0,
        control_rate: float = 100.0
    ):
        """
        Args:
            force_axes: 力控方向掩码 (例如 [0,0,1,0,0,0] 表示只控制Z轴力)
            Kp: 位置环增益
            Kf: 力环增益
        """
        self.force_axes = force_axes.astype(bool)
        self.position_axes = ~self.force_axes
        self.Kp = Kp
        self.Kf = Kf
        self.dt = 1.0 / control_rate
        
    def compute_torque(
        self,
        desired_position: np.ndarray,
        desired_force: np.ndarray,
        current_position: np.ndarray,
        current_force: np.ndarray,
        jacobian: np.ndarray
    ) -> np.ndarray:
        """
        计算力位混合控制力矩
        
        Args:
            desired_position: 目标位置
            desired_force: 目标力
            current_position: 当前位置
            current_force: 当前力
            jacobian: 雅可比
        """
        # 位置误差 (仅位置控方向)
        pos_error = np.zeros(6)
        pos_error[self.position_axes] = desired_position - current_position
        
        # 力误差 (仅力控方向)
        force_error = np.zeros(6)
        force_error[self.force_axes] = desired_force - current_force
        
        # 混合控制
        control_effort = np.zeros(6)
        control_effort[self.position_axes] = self.Kp * pos_error[self.position_axes]
        control_effort[self.force_axes] = self.Kf * force_error[self.force_axes]
        
        # 转换到关节空间
        jacobian_T = jacobian.T
        joint_torques = jacobian_T @ control_effort
        
        return joint_torques


class CollaborativeController:
    """
    协作控制
    
    实现:
    - 人类意图估计
    - 拖动示教
    - 安全监控
    """
    
    def __init__(
        self,
        safety_force_limit: float = 50.0,  # N
        safety_velocity_limit: float = 0.5,  # m/s
        reaction_mode: str = "pause"  # "pause" / "backoff"
    ):
        """
        Args:
            safety_force_limit: 安全力限制
            safety_velocity_limit: 安全速度限制
            reaction_mode: 反应模式
        """
        self.safety_force_limit = safety_force_limit
        self.safety_velocity_limit = safety_velocity_limit
        self.reaction_mode = reaction_mode
        
        self._is_safe = True
        
    def check_safety(
        self,
        external_force: np.ndarray,
        velocity: np.ndarray
    ) -> Tuple[bool, str]:
        """
        安全检查
        
        Returns:
            (is_safe, violation_type)
        """
        force_mag = np.linalg.norm(external_force)
        vel_mag = np.linalg.norm(velocity)
        
        if force_mag > self.safety_force_limit:
            self._is_safe = False
            return False, f"force_limit ({force_mag:.1f}N)"
        
        if vel_mag > self.safety_velocity_limit:
            self._is_safe = False
            return False, f"velocity_limit ({vel_mag:.2f}m/s)"
        
        self._is_safe = True
        return True, ""
    
    def get_reaction_torque(
        self,
        external_force: np.ndarray,
        jacobian: np.ndarray
    ) -> np.ndarray:
        """
        计算安全反应力矩
        
        根据反应模式:
        - pause: 零速
        - backoff: 反向力
        """
        if self.reaction_mode == "pause":
            # 零速度命令
            return np.zeros(jacobian.shape[1])
        
        elif self.reaction_mode == "backoff":
            # 反向力
            backoff_force = -external_force * 0.5
            return jacobian[:3, :].T @ backoff_force
        
        return np.zeros(jacobian.shape[1])
