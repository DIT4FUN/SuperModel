# Copyright (C) 2026 焦洋 (Jiao Yang) <jiaoyang@cczu.edu.cn>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
from typing import Tuple, Optional, Callable, Dict, List


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
        
        # 计算阻抗力 (使用前3维的阻抗参数)
        impedance_force = (
            self.params.K[:3, :3] @ error[:3] +
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
        pos_error[self.position_axes] = (
            desired_position[self.position_axes] - current_position[self.position_axes]
        )

        # 力误差 (仅力控方向)
        force_error = np.zeros(6)
        force_error[self.force_axes] = (
            desired_force[self.force_axes] - current_force[self.force_axes]
        )

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


class AdaptiveImpedanceController:
    """
    自适应阻抗控制器
    ==================
    
    在线估计环境参数并自适应调整阻抗系数。
    基于李雅普诺夫稳定性理论实现参数自适应律。
    
    适用场景:
    - 未知刚度环境 (刚度范围 100~10000 N/m)
    - 非结构化环境接触
    - 人机协作中的意图适应
    
    支持AGV等级: M / L / XL / XXL
    """
    
    def __init__(
        self,
        base_params: Optional[ImpedanceParams] = None,
        control_rate: float = 100.0,
        adaptation_rate: float = 0.01,
        env_stiffness_bounds: Tuple[float, float] = (100.0, 10000.0),
        gradient_learning_rate: float = 0.05,
        forgetting_factor: float = 0.98,
        use_lyapunov: bool = True,
    ):
        """
        Args:
            base_params: 基础阻抗参数
            control_rate: 控制频率 Hz
            adaptation_rate: 参数适应率 (0~1)
            env_stiffness_bounds: 环境刚度上下界 [K_min, K_max]
            gradient_learning_rate: 梯度下降学习率
            forgetting_factor: 遗忘因子 (MRAC用)
            use_lyapunov: 是否使用李雅普诺夫稳定性分析
        """
        self.base_params = base_params or ImpedanceParams.default_6d()
        self.dt = 1.0 / control_rate
        self.gamma = adaptation_rate
        self.K_bounds = env_stiffness_bounds
        self.alpha = gradient_learning_rate
        self.forgetting = forgetting_factor
        self.use_lyapunov = use_lyapunov
        
        # 在线估计的环境参数
        self._est_env_stiffness = 1000.0  # N/m 初始估计
        self._est_env_damping = 50.0      # Ns/m 初始估计
        self._est_env_inertia = 5.0       # kg 初始估计
        
        # 参数估计历史 (用于MRAC)
        self._stiffness_history: List[float] = []
        self._damping_history: List[float] = []
        self._inertia_history: List[float] = []
        self._history_len = 100
        
        # 状态
        self._position_error_prev = np.zeros(3)
        self._velocity_error_prev = np.zeros(3)
        self._force_error_prev = 0.0
        self._adaptation_gain_schedule = np.ones(3)  # [K_stiff, K_damp, K_inertia]
        
        # 当前阻抗参数 (自适应调整后)
        self._current_K = self.base_params.K.copy()
        self._current_D = self.base_params.D.copy()
        self._current_M = self.base_params.M.copy()
        
    @property
    def estimated_env_params(self) -> Dict[str, float]:
        """返回当前估计的环境参数"""
        return {
            "stiffness_N_per_m": self._est_env_stiffness,
            "damping_Ns_per_m": self._est_env_damping,
            "inertia_kg": self._est_env_inertia,
        }
    
    @property
    def current_impedance_params(self) -> ImpedanceParams:
        """返回当前自适应后的阻抗参数"""
        return ImpedanceParams(
            M=self._current_M.copy(),
            D=self._current_D.copy(),
            K=self._current_K.copy()
        )
    
    def _clip_stiffness(self, K: float) -> float:
        """限制刚度在合理范围内"""
        return np.clip(K, self.K_bounds[0], self.K_bounds[1])
    
    def _estimate_env_impedance_mrec(
        self,
        position_error: np.ndarray,
        velocity_error: np.ndarray,
        contact_force: np.ndarray,
        dt: float
    ) -> Tuple[float, float, float]:
        """
        MRAC (模型参考自适应控制) 在线估计环境参数
        
        基于力-位移关系的梯度下降估计
        
        Args:
            position_error: 位置误差 (m)
            velocity_error: 速度误差 (m/s)
            contact_force: 接触力 (N)
            dt: 时间步长
            
        Returns:
            (est_K, est_D, est_M)
        """
        force_mag = np.linalg.norm(contact_force)
        pos_mag = np.linalg.norm(position_error)
        vel_mag = np.linalg.norm(velocity_error)
        
        # 力误差
        force_error = force_mag - (
            self._est_env_stiffness * pos_mag +
            self._est_env_damping * vel_mag
        )
        
        # 梯度更新 (带遗忘因子)
        if pos_mag > 1e-6:
            delta_K = self.alpha * self.forgetting * force_error * pos_mag
            self._est_env_stiffness = self._clip_stiffness(
                self._est_env_stiffness + delta_K * dt
            )
        
        if vel_mag > 1e-6:
            delta_D = self.alpha * self.forgetting * force_error * vel_mag
            self._est_env_damping = np.clip(
                self._est_env_damping + delta_D * dt,
                1.0, 500.0
            )
        
        # 惯性估计 (从加速度估算)
        acc_estimate = force_mag / (self._est_env_inertia + 1e-6)
        if abs(acc_estimate) > 1e-4:
            delta_M = self.alpha * self.forgetting * force_error * acc_estimate
            self._est_env_inertia = np.clip(
                self._est_env_inertia + delta_M * dt,
                0.1, 100.0
            )
        
        # 记录历史
        self._stiffness_history.append(self._est_env_stiffness)
        self._damping_history.append(self._est_env_damping)
        self._inertia_history.append(self._est_env_inertia)
        
        if len(self._stiffness_history) > self._history_len:
            self._stiffness_history.pop(0)
            self._damping_history.pop(0)
            self._inertia_history.pop(0)
        
        return self._est_env_stiffness, self._est_env_damping, self._est_env_inertia
    
    def _lyapunov_update(
        self,
        position_error: np.ndarray,
        velocity_error: np.ndarray,
        external_force: np.ndarray
    ) -> None:
        """
        基于李雅普诺夫稳定性理论的参数更新
        
        李雅普诺夫函数: V = 0.5 * e^T * M * e + 0.5 * (K-K_d)^T * Gamma^-1 * (K-K_d)
        
        保证系统稳定性收敛
        """
        e_pos = position_error
        e_vel = velocity_error
        
        # 广义误差
        error = np.concatenate([e_pos, e_vel])
        
        # 环境力估计
        F_env_estimate = (
            self._est_env_stiffness * e_pos +
            self._est_env_damping * e_vel
        )
        
        # 力跟踪误差
        force_error = external_force - F_env_estimate
        
        # 李雅普诺夫梯度下降
        if np.linalg.norm(e_pos) > 1e-6:
            grad_K = np.outer(force_error, e_pos)
            delta_K = self.gamma * np.trace(grad_K)
            self._current_K[:3, :3] = self.base_params.K[:3, :3] + delta_K * self.dt
        
        if np.linalg.norm(e_vel) > 1e-6:
            grad_D = np.outer(force_error, e_vel)
            delta_D = self.gamma * np.trace(grad_D)
            self._current_D[:3, :3] = self.base_params.D[:3, :3] + delta_D * self.dt
        
        # 限制参数范围确保稳定性
        self._current_K = np.clip(
            self._current_K,
            self.base_params.K * 0.1,
            self.base_params.K * 5.0
        )
        self._current_D = np.clip(
            self._current_D,
            self.base_params.D * 0.1,
            self.base_params.D * 5.0
        )
    
    def _gain_schedule_adaptation(
        self,
        contact_phase: str,
        task_type: str = "contact"
    ) -> np.ndarray:
        """
        根据任务阶段的自适应增益调度
        
        Args:
            contact_phase: 'approach' | 'contact' | 'slide' | 'withdraw'
            task_type: 'contact' | 'peg_in_hole' | 'polishing' | 'assembly'
            
        Returns:
            adaptation_gains: [K_stiff_mult, K_damp_mult, K_inertia_mult]
        """
        schedule = {
            "approach":  np.array([0.1, 0.5, 0.2]),   # 低刚度接近
            "contact":   np.array([1.0, 1.0, 1.0]),   # 正常阻抗
            "slide":     np.array([0.5, 2.0, 0.5]),   # 高阻尼滑动
            "withdraw":  np.array([0.2, 0.5, 0.3]),  # 低刚度撤回
        }
        
        task_modifiers = {
            "peg_in_hole": np.array([2.0, 1.5, 3.0]),  # 高增益孔轴装配
            "polishing":   np.array([0.5, 3.0, 0.5]),  # 高阻尼抛光
            "assembly":    np.array([1.5, 1.0, 2.0]),  # 装配任务
        }
        
        base = schedule.get(contact_phase, np.array([1.0, 1.0, 1.0]))
        modifier = task_modifiers.get(task_type, np.array([1.0, 1.0, 1.0]))
        
        return base * modifier
    
    def update(
        self,
        desired_position: np.ndarray,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        external_wrench: np.ndarray,
        jacobian: np.ndarray,
        contact_phase: str = "contact",
        task_type: str = "contact"
    ) -> Tuple[np.ndarray, Dict]:
        """
        自适应阻抗控制主循环
        
        Args:
            desired_position: 目标位置 (m)
            current_position: 当前位置 (m)
            current_velocity: 当前速度 (m/s)
            external_wrench: 外力旋量 (6,) [Fx, Fy, Fz, Tx, Ty, Tz]
            jacobian: 雅可比矩阵
            contact_phase: 接触阶段
            task_type: 任务类型
            
        Returns:
            (joint_torques, info_dict)
        """
        # 误差计算
        pos_error = current_position - desired_position
        vel_error = current_velocity  # 假设desired_velocity=0
        
        # 1. 在线参数估计 (MRAC)
        K_est, D_est, M_est = self._estimate_env_impedance_mrec(
            pos_error, vel_error, external_wrench[:3], self.dt
        )
        
        # 2. 自适应增益调度
        gain_schedule = self._gain_schedule_adaptation(contact_phase, task_type)
        
        # 3. 调整阻抗参数
        if self.use_lyapunov:
            self._lyapunov_update(pos_error, vel_error, external_wrench[:3])
        else:
            # 直接根据估计的环境参数调整
            K_ratio = K_est / 1000.0  # 归一化到默认刚度
            self._current_K[:3, :3] = (
                self.base_params.K[:3, :3] * K_ratio * gain_schedule[0]
            )
            self._current_D[:3, :3] = (
                self.base_params.D[:3, :3] * gain_schedule[1]
            )
            self._current_M[:3, :3] = (
                self.base_params.M[:3, :3] * gain_schedule[2]
            )
        
        # 4. 阻抗控制计算
        impedance_force = (
            self._current_K[:3, :3] @ pos_error +
            self._current_D[:3, :3] @ vel_error
        )
        
        # 加上外力补偿
        compensated_force = external_wrench[:3] - impedance_force
        
        # 转换到关节空间
        jacobian_T = jacobian[:3, :].T
        joint_torques = jacobian_T @ compensated_force
        
        # 记录上一次状态
        self._position_error_prev = pos_error.copy()
        self._velocity_error_prev = vel_error.copy()
        self._force_error_prev = np.linalg.norm(external_wrench[:3])
        
        info = {
            "pos_error_norm": float(np.linalg.norm(pos_error)),
            "vel_error_norm": float(np.linalg.norm(vel_error)),
            "force_magnitude": float(np.linalg.norm(external_wrench[:3])),
            "est_env_K": K_est,
            "est_env_D": D_est,
            "est_env_M": M_est,
            "adaptation_gains": gain_schedule.tolist(),
            "current_K_trace": float(np.trace(self._current_K)),
            "current_D_trace": float(np.trace(self._current_D)),
        }
        
        return joint_torques, info
    
    def reset(self) -> None:
        """重置估计状态"""
        self._est_env_stiffness = 1000.0
        self._est_env_damping = 50.0
        self._est_env_inertia = 5.0
        self._stiffness_history.clear()
        self._damping_history.clear()
        self._inertia_history.clear()
        self._position_error_prev = np.zeros(3)
        self._velocity_error_prev = np.zeros(3)
        self._current_K = self.base_params.K.copy()
        self._current_D = self.base_params.D.copy()
        self._current_M = self.base_params.M.copy()
    
    def get_convergence_metrics(self) -> Dict[str, float]:
        """
        获取参数收敛性指标
        
        Returns:
            包含收敛性指标的字典
        """
        if len(self._stiffness_history) < 10:
            return {"converged": False, "confidence": 0.0}
        
        K_arr = np.array(self._stiffness_history[-50:])
        D_arr = np.array(self._damping_history[-50:])
        
        # 计算方差 (越小说明收敛越好)
        K_std = float(np.std(K_arr))
        D_std = float(np.std(D_arr))
        
        # 归一化置信度
        confidence = max(0.0, 1.0 - (K_std / (self._est_env_stiffness * 0.1 + 1)))
        
        return {
            "converged": K_std < self._est_env_stiffness * 0.05,
            "confidence": confidence,
            "K_variance": K_std,
            "D_variance": D_std,
            "history_length": len(self._stiffness_history),
        }


# === AGV五级阻抗控制规格表 ===
AGV_IMPEDANCE_GRADES = {
    "S": {
        "control_freq_hz": 50,
        "stiffness_range": (50.0, 500.0),
        "damping_range": (10.0, 100.0),
        "inertia_range": (1.0, 10.0),
        "force_limit": 50.0,
        "position_error_limit": 0.05,
        "adaptation_rate": 0.005,
        "convergence_time_s": 5.0,
        "use_lyapunov": False,
        "use_mrac": False,
    },
    "M": {
        "control_freq_hz": 100,
        "stiffness_range": (100.0, 1000.0),
        "damping_range": (20.0, 200.0),
        "inertia_range": (2.0, 20.0),
        "force_limit": 100.0,
        "position_error_limit": 0.02,
        "adaptation_rate": 0.01,
        "convergence_time_s": 2.0,
        "use_lyapunov": False,
        "use_mrac": False,
    },
    "L": {
        "control_freq_hz": 200,
        "stiffness_range": (200.0, 2000.0),
        "damping_range": (50.0, 500.0),
        "inertia_range": (5.0, 50.0),
        "force_limit": 200.0,
        "position_error_limit": 0.01,
        "adaptation_rate": 0.02,
        "convergence_time_s": 1.0,
        "use_lyapunov": True,
        "use_mrac": False,
    },
    "XL": {
        "control_freq_hz": 500,
        "stiffness_range": (300.0, 3000.0),
        "damping_range": (70.0, 700.0),
        "inertia_range": (7.0, 70.0),
        "force_limit": 350.0,
        "position_error_limit": 0.005,
        "adaptation_rate": 0.05,
        "convergence_time_s": 0.5,
        "use_lyapunov": True,
        "use_mrac": True,
    },
    "XXL": {
        "control_freq_hz": 1000,
        "stiffness_range": (500.0, 5000.0),
        "damping_range": (100.0, 1000.0),
        "inertia_range": (10.0, 100.0),
        "force_limit": 500.0,
        "position_error_limit": 0.001,
        "adaptation_rate": 0.1,
        "convergence_time_s": 0.5,
        "use_lyapunov": True,
        "use_mrac": True,
    },
}


def get_impedance_spec(grade: str) -> Dict:
    """
    获取指定AGV等级的阻抗控制规格

    Args:
        grade: AGV等级 (S/M/L/XL/XXL)

    Returns:
        阻抗控制规格字典
    """
    return AGV_IMPEDANCE_GRADES.get(grade, AGV_IMPEDANCE_GRADES['M'])


def list_impedance_capabilities() -> Dict[str, Dict]:
    """
    列出所有AGV等级的阻抗控制能力

    Returns:
        所有等级的完整规格字典
    """
    return dict(AGV_IMPEDANCE_GRADES)


__all__ = [
    'ImpedanceParams',
    'ImpedanceController',
    'AdmittanceController',
    'ForceImpedanceController',
    'CollaborativeController',
    'AdaptiveImpedanceController',
    'AGV_IMPEDANCE_GRADES',
    'get_impedance_spec',
    'list_impedance_capabilities',
]
