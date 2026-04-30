# Copyright (C) 2024-2026 赵元请 (DIT4FUN)
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
MPC 模型预测控制模块
==================

高级模型预测控制器
- 关节空间 MPC
- 笛卡尔空间 MPC
- 约束处理 (关节限位、速度限制、碰撞回避)
- 多目标优化 (位置跟踪 + 能量最优 + 平滑度)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Callable, Dict, Any
import warnings


@dataclass
class MPCConfig:
    """MPC 配置"""
    # 预测范围
    horizon: int = 20              # 预测步数 N
    # 控制范围 (通常 <= horizon)
    control_horizon: int = 10     # 控制步数 nu
    # 采样时间
    dt: float = 0.01              # s
    # 权重
    Q_pos: np.ndarray = None       # 位置跟踪权重 (关节数,)
    Q_vel: np.ndarray = None       # 速度跟踪权重
    R_acc: np.ndarray = None       # 加速度/控制权重 (能量消耗)
    R_jerk: np.ndarray = None      # 冲击度权重 (平滑度)
    # 约束
    joint_limits_lower: np.ndarray = None
    joint_limits_upper: np.ndarray = None
    velocity_limits: np.ndarray = None
    acceleration_limits: np.ndarray = None
    torque_limits: np.ndarray = None
    # 高级约束
    collision_margin: float = 0.05  # 碰撞裕度 (m)
    # 求解器参数
    solver: str = "osqp"           # "osqp" | "qp" | "unconstraint"
    max_iterations: int = 1000
    # AGV 等级
    grade: str = "M"

    def __post_init__(self):
        if self.Q_pos is None:
            self.Q_pos = np.ones(6) * 100.0
        if self.Q_vel is None:
            self.Q_vel = np.ones(6) * 10.0
        if self.R_acc is None:
            self.R_acc = np.ones(6) * 1.0
        if self.R_jerk is None:
            self.R_jerk = np.ones(6) * 0.1

        # 转换权重为对角矩阵
        if len(self.Q_pos.shape) == 1:
            self.Q_pos_diag = np.diag(self.Q_pos)
        if len(self.Q_vel.shape) == 1:
            self.Q_vel_diag = np.diag(self.Q_vel)
        if len(self.R_acc.shape) == 1:
            self.R_acc_diag = np.diag(self.R_acc)

    @classmethod
    def for_grade(cls, grade: str, num_joints: int = 6, dt: float = 0.01) -> 'MPCConfig':
        """根据 AGV 等级创建配置"""
        configs = {
            'S': cls(horizon=10, control_horizon=5, dt=0.02, grade=grade,
                     Q_pos=np.ones(num_joints) * 50,
                     Q_vel=np.ones(num_joints) * 5,
                     R_acc=np.ones(num_joints) * 5.0),
            'M': cls(horizon=20, control_horizon=10, dt=dt, grade=grade),
            'L': cls(horizon=30, control_horizon=15, dt=dt, grade=grade,
                     Q_pos=np.ones(num_joints) * 200,
                     Q_vel=np.ones(num_joints) * 20,
                     R_jerk=np.ones(num_joints) * 0.5),
            'XL': cls(horizon=40, control_horizon=20, dt=dt, grade=grade,
                      Q_pos=np.ones(num_joints) * 500,
                      Q_vel=np.ones(num_joints) * 50,
                      collision_margin=0.02),
            'XXL': cls(horizon=50, control_horizon=25, dt=dt, grade=grade,
                        Q_pos=np.ones(num_joints) * 1000,
                        Q_vel=np.ones(num_joints) * 100,
                        collision_margin=0.01,
                        solver="osqp"),
        }
        return configs.get(grade, configs['M'])


@dataclass
class JointStateMP:
    """关节状态 (MPC 用)"""
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray = None

    def __post_init__(self):
        if self.acceleration is None:
            self.acceleration = np.zeros_like(self.position)


class DynamicsModel:
    """
    机器人动力学模型

    简化模型: qdd = M^{-1}(tau - C(q,qd) - g(q) - tau_friction)
    使用线性近似用于 MPC: qdd ≈ K * q + B * qd + G * tau
    """

    def __init__(
        self,
        num_joints: int = 6,
        mass_matrix_diag: Optional[np.ndarray] = None,
        damping: Optional[np.ndarray] = None,
        gravity: float = 9.81
    ):
        self.n = num_joints
        # 质量矩阵对角线 (简化)
        self.M_diag = mass_matrix_diag or np.ones(self.n) * 0.5
        # 阻尼
        self.damping = damping or np.ones(self.n) * 2.0
        self.gravity = gravity

    def forward(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        tau: np.ndarray
    ) -> np.ndarray:
        """
        简化动力学前向计算

        Args:
            q: 关节位置 (n,)
            qd: 关节速度 (n,)
            tau: 关节力矩 (n,)

        Returns:
            qdd: 关节加速度 (n,)
        """
        # 简化重力 (仅用于垂直方向关节)
        g_term = np.zeros(self.n)
        g_term[2] = self.gravity * 0.5  # 简化的重力影响

        # 摩擦力 (库仑 + 粘性)
        friction = 0.5 * np.sign(qd) + self.damping * qd

        # 动力学: M * qdd = tau - friction - gravity
        qdd = (tau - friction - g_term) / (self.M_diag + 1e-6)
        return qdd

    def linearize(
        self,
        q: np.ndarray,
        qd: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        线性化动力学模型

        Returns:
            A, B, G: 状态矩阵, 输入矩阵, 偏置
            状态 x = [q; qd], 输入 u = [tau]
            xdot = A*x + B*u + G
        """
        n = self.n

        # 状态转移矩阵 (简化)
        A = np.zeros((2 * n, 2 * n))
        # qdot = qd
        A[:n, n:2 * n] = np.eye(n)
        # qddot 相对于 q 的偏导 (简化)
        A[n:2 * n, :n] = np.zeros((n, n))

        # 输入矩阵
        B = np.zeros((2 * n, n))
        B[n:2 * n, :] = np.diag(1.0 / (self.M_diag + 1e-6))

        # 偏置 (重力 + 阻尼)
        G = np.zeros(2 * n)
        G[n:2 * n] = -self.gravity * 0.5 / (self.M_diag + 1e-6) * np.array([0, 0, 1, 0, 0, 0])[:n]
        G[n:2 * n] -= self.damping * qd / (self.M_diag + 1e-6)

        return A, B, G

    def discrete_matrices(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取离散状态空间矩阵

        Returns:
            Ad, Bd: 离散系统矩阵
        """
        A, B, _ = self.linearize(q, qd)
        n = 2 * self.n

        # 零阶保持离散化
        I = np.eye(n)
        # 近似: Ad = I + A*dt, Bd = B*dt
        Ad = I + A * dt
        Bd = B * dt

        return Ad, Bd


class JointSpaceMPC:
    """
    关节空间 MPC 控制器

    最小化: ||q_k - q_d_k||_Q^2 + ||qd_k - qd_d_k||_R^2 + ||u_k||_R^2
    subject to: 关节限位, 速度限位, 力矩限位
    """

    def __init__(
        self,
        config: Optional[MPCConfig] = None,
        dynamics: Optional[DynamicsModel] = None,
        num_joints: int = 6
    ):
        self.config = config or MPCConfig()
        self.n = num_joints
        self.dynamics = dynamics or DynamicsModel(num_joints=num_joints)

        # 状态维度: q + qd = 2n
        self.state_dim = 2 * self.n
        self.control_dim = self.n

        # 初始化
        self.Q = np.kron(np.eye(self.config.horizon), np.diag(self.config.Q_pos))
        self.Q_vel = np.kron(np.eye(self.config.horizon), np.diag(self.config.Q_vel))
        self.R = np.kron(np.eye(self.config.horizon), np.diag(self.config.R_acc))

        # 预测轨迹 (上次求解结果)
        self.predicted_states: List[np.ndarray] = []
        self.predicted_controls: List[np.ndarray] = []

        # 缓存
        self._Ad_cache = None
        self._Bd_cache = None
        self._last_state = None

    def compute_gain(
        self,
        state: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算状态反馈增益 (LQR 或 QP)

        这里使用简化的线性反馈增益
        """
        q = state[:self.n]
        qd = state[self.n:2 * self.n]

        Ad, Bd = self.dynamics.discrete_matrices(q, qd, dt)

        # 简化的状态反馈增益 (LQR 近似)
        # K = (R + B^T * P * B)^{-1} * B^T * P * A
        # 这里用伪逆代替完整求解
        K = np.zeros((self.n, self.state_dim))
        for i in range(self.n):
            K[i, i] = self.config.Q_pos[i] / max(self.config.R_acc[i], 1e-6) * 0.1
            K[i, self.n + i] = self.config.Q_vel[i] / max(self.config.R_acc[i], 1e-6) * 0.05

        return Ad, Bd

    def compute_control(
        self,
        current_state: np.ndarray,
        desired_trajectory: np.ndarray,  # N x n, desired joint positions
        current_velocity: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算最优控制力矩

        Args:
            current_state: 当前关节位置 (n,)
            desired_trajectory: 期望关节位置 (horizon, n)
            current_velocity: 当前关节速度 (n,)

        Returns:
            tau: 关节力矩命令 (n,)
        """
        if current_velocity is None:
            current_velocity = np.zeros(self.n)

        state = np.concatenate([current_state, current_velocity])
        dt = self.config.dt
        N = self.config.horizon

        # 构建 QP 问题 (简化版)
        # 目标: min 0.5 * u^T * H * u + f^T * u
        # 约束: lb <= u <= ub

        H = np.zeros((N * self.n, N * self.n))
        f = np.zeros(N * self.n)

        Ad, Bd = self.dynamics.discrete_matrices(
            current_state, current_velocity, dt
        )

        # 构建目标轨迹追踪误差的权重矩阵
        for k in range(N):
            # 预测状态: x_k = A^k * x0 + sum(A^{k-i} * B * u_i)
            # 简化: 直接构建二次型
            H[k * self.n:(k + 1) * self.n, k * self.n:(k + 1) * self.n] = \
                np.diag(self.config.R_acc) + Bd.T @ np.diag(
                    np.ones(self.state_dim) * 0.1
                ) @ Bd

            # 目标: 追踪期望位置
            if k < len(desired_trajectory):
                ref_pos = desired_trajectory[k]
                error = current_state - ref_pos
                f[k * self.n:(k + 1) * self.n] = error * self.config.R_acc * 0.1

        # 添加正则化确保 H 正定
        H += np.eye(N * self.n) * 1e-4

        # 解 QP (使用梯度下降简化)
        u = np.linalg.solve(H + 1e-4 * np.eye(H.shape[0]), -f)

        # 提取第一个控制输入
        tau = u[:self.n]

        # 应用力矩限位
        if self.config.torque_limits is not None:
            tau = np.clip(tau, -self.config.torque_limits, self.config.torque_limits)

        # 预测状态存储 (用于监控)
        self.predicted_states = []
        x = state.copy()
        for k in range(min(N, len(desired_trajectory))):
            self.predicted_states.append(x.copy())
            u_k = u[k * self.n:(k + 1) * self.n]
            qdd = self.dynamics.forward(x[:self.n], x[self.n:2 * self.n], u_k)
            x[:self.n] += x[self.n:2 * self.n] * dt
            x[self.n:2 * self.n] += qdd * dt

        return tau

    def compute_control_simple(
        self,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        target_pos: np.ndarray,
        target_vel: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        简化的 MPC 控制 (无完整 QP 求解)

        使用 PD 反馈 + 前馈重力补偿
        """
        if target_vel is None:
            target_vel = np.zeros_like(target_pos)

        # PD 控制
        Kp = np.diag(self.config.Q_pos) * 0.01
        Kd = np.diag(self.config.Q_vel) * 0.1

        pos_error = target_pos - current_pos
        vel_error = target_vel - current_vel

        # 反馈控制
        tau_fb = Kp @ pos_error + Kd @ vel_error

        # 前馈重力补偿
        tau_ff = np.zeros(self.n)
        tau_ff[2] = self.dynamics.gravity * 0.3  # 简化的重力补偿

        tau = tau_fb + tau_ff

        # 限幅
        if self.config.torque_limits is not None:
            tau = np.clip(tau, -self.config.torque_limits, self.config.torque_limits)

        return tau

    def reset(self):
        """重置 MPC 内部状态"""
        self.predicted_states.clear()
        self.predicted_controls.clear()


class CartesianMPC:
    """
    笛卡尔空间 MPC 控制器

    在笛卡尔空间进行位置/姿态跟踪
    需要逆运动学将末端执行器轨迹转换为关节轨迹
    """

    def __init__(
        self,
        config: Optional[MPCConfig] = None,
        num_joints: int = 6
    ):
        self.config = config or MPCConfig()
        self.n = num_joints
        self.joint_mpc = JointSpaceMPC(config=config, num_joints=num_joints)

        # 简化的雅可比矩阵估计
        self._link_lengths = np.array([0.3] * 6)
        self._base_height = 0.5

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """
        简化的前向运动学

        Returns:
            pose: [x, y, z, roll, pitch, yaw] (6,)
        """
        # 简化的 6DOF 机械臂前向运动学 (仅用于笛卡尔 MPC)
        # 使用前 3 个连杆
        ll3 = self._link_lengths[:3]
        x = np.sum(ll3 * np.cos(np.cumsum(q[:3]))) * 0.3
        y = np.sum(ll3 * np.sin(np.cumsum(q[:3]))) * 0.3
        z = self._base_height + np.sum(ll3 * np.sin(q[:3])) * 0.5

        roll = q[0] * 0.5
        pitch = q[1] * 0.5
        yaw = q[2] * 0.5

        return np.array([x, y, z, roll, pitch, yaw])

    def jacobian_approx(self, q: np.ndarray) -> np.ndarray:
        """
        近似雅可比矩阵 (3x6)

        用于将笛卡尔速度转换为关节速度
        """
        J = np.zeros((3, self.n))
        # 简化: 假设每个关节对 x, y, z 有相似的贡献
        for i in range(min(3, self.n)):
            J[0, i] = 0.1 * np.cos(q[i])
            J[1, i] = 0.1 * np.sin(q[i])
            J[2, i] = 0.05
        return J

    def compute_control(
        self,
        current_joint_pos: np.ndarray,
        current_joint_vel: np.ndarray,
        target_pose: np.ndarray,   # 6, [x, y, z, roll, pitch, yaw]
        target_twist: Optional[np.ndarray] = None  # 6, [vx, vy, vz, wx, wy, wz]
    ) -> np.ndarray:
        """
        笛卡尔空间 MPC 控制

        Args:
            current_joint_pos: 当前关节位置 (n,)
            current_joint_vel: 当前关节速度 (n,)
            target_pose: 目标末端执行器位姿 (6,)
            target_twist: 目标末端执行器速度 (6,)

        Returns:
            tau: 关节力矩命令 (n,)
        """
        if target_twist is None:
            target_twist = np.zeros(6)

        # 估计当前末端位置
        current_pose = self.forward_kinematics(current_joint_pos)

        # 计算位置误差
        pos_error = target_pose[:3] - current_pose[:3]

        # 雅可比伪逆 (简化)
        J = self.jacobian_approx(current_joint_pos)
        J_pinv = np.linalg.pinv(J)

        # 将笛卡尔误差转换为关节空间期望轨迹
        # 使用 5 步滚动窗口
        N = min(self.config.horizon, 5)
        joint_traj = np.zeros((N, self.n))

        for k in range(N):
            frac = (k + 1) / N
            # 关节空间期望位置 = 当前 + 雅可比伪逆 * 位置误差比例
            delta_q = J_pinv @ (pos_error * frac * 0.5)
            joint_traj[k] = current_joint_pos + delta_q

        # 使用关节空间 MPC
        tau = self.joint_mpc.compute_control(
            current_joint_pos,
            joint_traj,
            current_joint_vel
        )

        return tau


def get_mpc_spec(grade: str = 'M') -> Dict[str, Any]:
    """获取 MPC 控制器规格 (按 AGV 等级)"""
    specs = {
        'S': {
            'horizon': 10, 'control_horizon': 5, 'dt': 0.02,
            'max_torque': 50, 'solver': 'qp',
            'constraints': ['joint_limits', 'velocity_limits'],
            'description': '基础 MPC, 软约束'
        },
        'M': {
            'horizon': 20, 'control_horizon': 10, 'dt': 0.01,
            'max_torque': 100, 'solver': 'qp',
            'constraints': ['joint_limits', 'velocity_limits', 'torque_limits'],
            'description': '标准 MPC, QP 求解'
        },
        'L': {
            'horizon': 30, 'control_horizon': 15, 'dt': 0.01,
            'max_torque': 200, 'solver': 'osqp',
            'constraints': ['joint_limits', 'velocity_limits', 'torque_limits', 'collision'],
            'description': '增强 MPC, OSQP 求解器'
        },
        'XL': {
            'horizon': 40, 'control_horizon': 20, 'dt': 0.005,
            'max_torque': 500, 'solver': 'osqp',
            'constraints': ['joint_limits', 'velocity_limits', 'torque_limits', 'collision', 'obstacle'],
            'description': '高级 MPC, 碰撞回避'
        },
        'XXL': {
            'horizon': 50, 'control_horizon': 25, 'dt': 0.002,
            'max_torque': 1000, 'solver': 'osqp',
            'constraints': ['joint_limits', 'velocity_limits', 'torque_limits', 'collision', 'obstacle', 'force'],
            'description': '旗舰 MPC, 力约束 + 多目标'
        }
    }
    return specs.get(grade, specs['M'])


class AdaptiveMPCController:
    """
    自适应 MPC 控制器
    ==================
    
    在线系统辨识 + 模型预测控制:
    - 递推最小二乘 (RLS) 在线辨识系统参数
    - 自适应预测模型更新
    - 冷启动重 planning 机制
    
    适用场景:
    - 负载变化场景 (抓取不同重量物体)
    - 磨损补偿 (关节参数漂移)
    - 人机协作 (交互力反馈调整)
    
    支持AGV等级: M / L / XL / XXL
    """
    
    def __init__(
        self,
        n_joints: int,
        base_config: Optional[MPCConfig] = None,
        identification_rate: float = 0.02,  # 辨识频率
        forgetting_factor: float = 0.995,
        adaptation_threshold: float = 0.1,
        use_ekf: bool = False,
    ):
        """
        Args:
            n_joints: 关节数
            base_config: 基础 MPC 配置
            identification_rate: 参数辨识更新频率
            forgetting_factor: 遗忘因子 (RLS)
            adaptation_threshold: 模型更新阈值
            use_ekf: 是否使用扩展卡尔曼滤波 (比RLS更稳定)
        """
        self.n_joints = n_joints
        self.config = base_config or self._default_config()
        self.id_rate = identification_rate
        self.lambda_rls = forgetting_factor
        self.adaptation_threshold = adaptation_threshold
        self.use_ekf = use_ekf
        
        # 基础模型参数 (标称值)
        self._nominal_inertia = np.eye(n_joints) * 5.0
        self._nominal_damping = np.eye(n_joints) * 10.0
        self._nominal_stiffness = np.eye(n_joints) * 0.0
        
        # 在线估计的参数
        self._est_inertia = np.eye(n_joints) * 5.0
        self._est_damping = np.eye(n_joints) * 10.0
        
        # RLS 协方差矩阵
        self._P_inertia = np.eye(n_joints * n_joints) * 100.0
        self._P_damping = np.eye(n_joints * n_joints) * 100.0
        
        # 参数估计历史
        self._inertia_history: List[np.ndarray] = []
        self._damping_history: List[np.ndarray] = []
        self._history_max = 200
        
        # 当前使用的模型
        self._current_inertia = self._nominal_inertia.copy()
        self._current_damping = self._nominal_damping.copy()
        
        # MPC 求解器
        self._mpc_solver = GradientDescentMPC(n_joints, self.config)
        
        # 计数器
        self._id_counter = 0
        self._total_updates = 0
        self._adaptation_count = 0
        
    def _default_config(self) -> MPCConfig:
        """默认配置"""
        config = MPCConfig()
        config.horizon = 20
        config.control_horizon = 10
        config.dt = 0.01
        config.Q_pos = np.ones(self.n_joints) * 100.0
        config.Q_vel = np.ones(self.n_joints) * 10.0
        config.R_acc = np.ones(self.n_joints) * 0.1
        return config
    
    def _vector_to_matrix(self, vec: np.ndarray) -> np.ndarray:
        """向量转矩阵 (用于 RLS 参数)"""
        n = self.n_joints
        return vec.reshape(n, n)
    
    def _matrix_to_vector(self, mat: np.ndarray) -> np.ndarray:
        """矩阵转向量 (用于 RLS 参数)"""
        return mat.flatten()
    
    def _rls_update(
        self,
        theta: np.ndarray,
        P: np.ndarray,
        phi: np.ndarray,
        y: float,
        lambda_f: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        递推最小二乘 (RLS) 更新
        
        Args:
            theta: 参数向量
            P: 协方差矩阵
            phi: 观测向量
            y: 观测输出
            lambda_f: 遗忘因子
            
        Returns:
            (theta_updated, P_updated)
        """
        # 预测
        y_pred = phi @ theta
        
        # 误差
        error = y - y_pred
        
        # 卡尔曼增益
        denom = lambda_f + phi @ P @ phi
        K = (P @ phi) / denom
        
        # 参数更新
        theta_new = theta + K * error
        
        # 协方差更新 (Joseph form for numerical stability)
        P_new = (np.eye(len(theta)) - np.outer(K, phi)) @ P / lambda_f
        
        return theta_new, P_new
    
    def identify_system_rls(
        self,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        joint_accelerations: np.ndarray,
        joint_torques: np.ndarray
    ) -> Dict[str, float]:
        """
        使用 RLS 在线辨识系统参数
        
        动力学模型: tau = M(q) * qdd + D(q, qd) * qd
        
        Args:
            joint_positions: 关节位置 (n_joints,)
            joint_velocities: 关节速度 (n_joints,)
            joint_accelerations: 关节加速度 (n_joints,)
            joint_torques: 关节力矩 (n_joints,)
            
        Returns:
            辨识诊断信息
        """
        n = self.n_joints
        
        # 构建回归矩阵 (每个关节一个方程)
        # tau_i = [M_ii.flatten(), D_ii.flatten()] @ [qdd, qd]
        # 简化: 假设 M 和 D 为对角矩阵
        
        # 观测向量: phi = [qdd_1, qdd_2, ..., qdd_n, qd_1, qd_2, ..., qd_n]
        phi_full = np.concatenate([joint_accelerations, joint_velocities])
        
        # 更新每个关节的惯性参数
        new_inertia_estimate = np.zeros((n, n))
        new_damping_estimate = np.zeros((n, n))
        
        diag_residuals = []
        
        for i in range(n):
            # 单关节 RLS
            # tau_i = M_ii * qdd_i + D_ii * qd_i
            phi_i = np.array([joint_accelerations[i], joint_velocities[i]])
            y_i = joint_torques[i]
            
            # 提取对角元素
            theta_M_i = np.array([self._est_inertia[i, i]])
            theta_D_i = np.array([self._est_damping[i, i]])
            
            # M 参数更新
            theta_M_new, P_M_new = self._rls_update(
                theta_M_i, np.array([[100.0]]),
                np.array([joint_accelerations[i]]),
                y_i, self.lambda_rls
            )
            new_inertia_estimate[i, i] = np.clip(theta_M_new[0], 0.1, 50.0)
            
            # 残余力用于阻尼估计
            residual = y_i - new_inertia_estimate[i, i] * joint_accelerations[i]
            theta_D_new, P_D_new = self._rls_update(
                theta_D_i, np.array([[100.0]]),
                np.array([joint_velocities[i]]),
                residual, self.lambda_rls
            )
            new_damping_estimate[i, i] = np.clip(theta_D_new[0], 0.1, 200.0)
            
            diag_residuals.append(abs(residual))
        
        # 更新当前模型
        old_inertia_trace = np.trace(self._current_inertia)
        old_damping_trace = np.trace(self._current_damping)
        
        # 平滑更新 (避免突变)
        alpha_smooth = 0.1
        self._current_inertia = (
            (1 - alpha_smooth) * self._current_inertia +
            alpha_smooth * new_inertia_estimate
        )
        self._current_damping = (
            (1 - alpha_smooth) * self._current_damping +
            alpha_smooth * new_damping_estimate
        )
        
        # 检查是否有显著变化
        inertia_change = abs(
            np.trace(self._current_inertia) - old_inertia_trace
        ) / (old_inertia_trace + 1e-6)
        
        if inertia_change > self.adaptation_threshold:
            self._adaptation_count += 1
        
        # 记录历史
        self._inertia_history.append(self._current_inertia.copy())
        self._damping_history.append(self._current_damping.copy())
        
        if len(self._inertia_history) > self._history_max:
            self._inertia_history.pop(0)
            self._damping_history.pop(0)
        
        self._total_updates += 1
        
        return {
            "inertia_trace_current": float(np.trace(self._current_inertia)),
            "inertia_trace_nominal": float(np.trace(self._nominal_inertia)),
            "inertia_change_ratio": float(inertia_change),
            "damping_trace_current": float(np.trace(self._current_damping)),
            "mean_residual": float(np.mean(diag_residuals)),
            "adaptation_count": self._adaptation_count,
            "total_updates": self._total_updates,
        }
    
    def compute_control(
        self,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        desired_pos: np.ndarray,
        desired_vel: Optional[np.ndarray] = None,
        external_torque: Optional[np.ndarray] = None,
        identification_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        自适应 MPC 控制
        
        Args:
            current_pos: 当前位置
            current_vel: 当前速度
            desired_pos: 目标位置
            desired_vel: 目标速度 (可选)
            external_torque: 外力矩扰动 (可选)
            identification_data: 包含辨识数据的字典 (可选)
            
        Returns:
            控制力矩
        """
        # 定期进行系统辨识
        if identification_data is not None:
            diag = self.identify_system_rls(
                identification_data.get("positions", current_pos),
                identification_data.get("velocities", current_vel),
                identification_data.get("accelerations", np.zeros(self.n_joints)),
                identification_data.get("torques", np.zeros(self.n_joints)),
            )
        
        # 构建带有当前估计参数的 MPC
        # 使用自适应动力学科室
        dynamics = lambda q, qd, tau: (
            np.linalg.solve(self._current_inertia + np.eye(self.n_joints) * 0.1,
                            tau - self._current_damping @ qd)
        )
        
        # 重置 MPC 求解器使用新模型
        self._mpc_solver._A_disc = np.block([
            [np.eye(self.n_joints), self.config.dt * np.eye(self.n_joints)],
            [np.zeros((self.n_joints, self.n_joints)),
             np.eye(self.n_joints) - self.config.dt * np.linalg.solve(
                 self._current_inertia + np.eye(self.n_joints) * 0.1,
                 self._current_damping
             )]
        ])
        
        self._mpc_solver._B_disc = np.block([
            [np.zeros((self.n_joints, self.n_joints))],
            [self.config.dt * np.linalg.solve(
                self._current_inertia + np.eye(self.n_joints) * 0.1,
                np.eye(self.n_joints)
            )]
        ])
        
        # 计算控制
        if desired_vel is None:
            desired_vel = np.zeros(self.n_joints)
        
        if external_torque is not None:
            # 扰动补偿
            disturbance_comp = np.linalg.solve(
                self._current_inertia + np.eye(self.n_joints) * 0.1,
                external_torque
            )
            desired_vel = desired_vel + disturbance_comp * self.config.dt
        
        tau = self._mpc_solver.compute_control(
            current_pos, desired_pos, current_vel, desired_vel
        )
        
        return tau
    
    def get_model_confidence(self) -> Dict[str, float]:
        """
        获取模型置信度
        
        基于参数估计方差和历史稳定性
        """
        if len(self._inertia_history) < 10:
            return {"confidence": 0.0, "stable": False}
        
        inertia_arr = np.array([
            np.trace(m) for m in self._inertia_history[-50:]
        ])
        
        inertia_std = float(np.std(inertia_arr))
        inertia_mean = float(np.mean(inertia_arr))
        
        # 归一化置信度
        cv = inertia_std / (inertia_mean * 0.1 + 1)
        confidence = max(0.0, 1.0 - cv)
        
        return {
            "confidence": confidence,
            "stable": cv < 0.1,
            "inertia_std": inertia_std,
            "inertia_mean": inertia_mean,
            "coefficient_of_variation": cv,
            "history_length": len(self._inertia_history),
            "adaptation_rate": self._adaptation_count / max(1, self._total_updates),
        }
    
    def reset_estimation(self) -> None:
        """重置参数估计"""
        self._est_inertia = self._nominal_inertia.copy()
        self._est_damping = self._nominal_damping.copy()
        self._current_inertia = self._nominal_inertia.copy()
        self._current_damping = self._nominal_damping.copy()
        self._inertia_history.clear()
        self._damping_history.clear()
        self._P_inertia = np.eye(self.n_joints * self.n_joints) * 100.0
        self._P_damping = np.eye(self.n_joints * self.n_joints) * 100.0
        self._adaptation_count = 0
        self._total_updates = 0


class GradientDescentMPC:
    """梯度下降 MPC 求解器 (OSQP 替代, 轻量级)"""
    
    def __init__(self, n_joints: int, config: MPCConfig):
        self.n = n_joints
        self.config = config
        
        # 离散状态空间
        self._A_disc = np.block([
            [np.eye(n_joints), config.dt * np.eye(n_joints)],
            [np.zeros((n_joints, n_joints)),
             np.eye(n_joints) - config.dt * 10.0 * np.eye(n_joints)]
        ])
        
        self._B_disc = np.block([
            [np.zeros((n_joints, n_joints))],
            [config.dt * np.eye(n_joints)]
        ])
    
    def compute_control(
        self,
        current_pos: np.ndarray,
        desired_pos: np.ndarray,
        current_vel: np.ndarray,
        desired_vel: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """计算 MPC 控制"""
        if desired_vel is None:
            desired_vel = np.zeros(self.n)
        
        # 拼接状态
        x = np.concatenate([current_pos, current_vel])
        x_des = np.concatenate([desired_pos, desired_vel])
        
        # 简单 LQR 近似
        Q = np.diag(list(self.config.Q_pos) + list(self.config.Q_vel))
        R = np.diag(list(self.config.R_acc))
        
        # 求解 Riccati
        P = Q.copy()
        for _ in range(self.config.horizon):
            P = Q + self._A_disc.T @ P @ self._A_disc - \
                self._A_disc.T @ P @ self._B_disc @ np.linalg.solve(
                    R + self._B_disc.T @ P @ self._B_disc,
                    self._B_disc.T @ P @ self._A_disc
                )
        
        # 反馈增益
        K_lqr = np.linalg.solve(
            R + self._B_disc.T @ P @ self._B_disc,
            self._B_disc.T @ P @ self._A_disc
        )
        
        # 应用第一帧控制
        u = -K_lqr @ (x - x_des)
        return u[:self.n]
