"""
AGV速度控制模块 (Velocity Control)
==================================

差速驱动AGV的先进速度控制:
- 速度PID闭环控制 (含前馈和摩擦补偿)
- S曲线速度规划执行器
- 轮速同步控制器
- AGV五级规格适配 (S/M/L/XL/XXL)

Author: SuperModel Development Team
Version: v2.45.0 (2026-04-10)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum, auto
import time


# ============================================================================
# AGV五级速度控制规格
# ============================================================================

class ControlFrequencyHz(NamedTuple):
    S: int = 50
    M: int = 100
    L: int = 200
    XL: int = 500
    XXL: int = 1000


AGV_VELOCITY_CONTROL_GRADES: Dict[str, Dict] = {
    "S": {
        "control_frequency_hz": 50,
        "max_linear_velocity_mps": 0.5,
        "max_angular_velocity_rps": 1.5,
        "velocity_pid_kp": 2.0,
        "velocity_pid_ki": 0.1,
        "velocity_pid_kd": 0.05,
        "friction_compensation": False,
        "feedforward": False,
        "wheel_slip_detection": False,
        "adaptive_gain": False,
        "profile_type": "trapezoidal",
        "acceleration_limit_mps2": 0.5,
        "jerk_limit_mps3": None,
        "real_time": False,
    },
    "M": {
        "control_frequency_hz": 100,
        "max_linear_velocity_mps": 1.5,
        "max_angular_velocity_rps": 3.0,
        "velocity_pid_kp": 3.0,
        "velocity_pid_ki": 0.2,
        "velocity_pid_kd": 0.1,
        "friction_compensation": True,
        "feedforward": True,
        "wheel_slip_detection": True,
        "adaptive_gain": False,
        "profile_type": "trapezoidal",
        "acceleration_limit_mps2": 1.0,
        "jerk_limit_mps3": None,
        "real_time": False,
    },
    "L": {
        "control_frequency_hz": 200,
        "max_linear_velocity_mps": 2.0,
        "max_angular_velocity_rps": 2.5,
        "velocity_pid_kp": 4.0,
        "velocity_pid_ki": 0.3,
        "velocity_pid_kd": 0.2,
        "friction_compensation": True,
        "feedforward": True,
        "wheel_slip_detection": True,
        "adaptive_gain": True,
        "profile_type": "s_curve",
        "acceleration_limit_mps2": 1.5,
        "jerk_limit_mps3": 5.0,
        "real_time": True,
    },
    "XL": {
        "control_frequency_hz": 500,
        "max_linear_velocity_mps": 3.0,
        "max_angular_velocity_rps": 2.0,
        "velocity_pid_kp": 5.0,
        "velocity_pid_ki": 0.5,
        "velocity_pid_kd": 0.3,
        "friction_compensation": True,
        "feedforward": True,
        "wheel_slip_detection": True,
        "adaptive_gain": True,
        "profile_type": "s_curve",
        "acceleration_limit_mps2": 2.0,
        "jerk_limit_mps3": 10.0,
        "real_time": True,
    },
    "XXL": {
        "control_frequency_hz": 1000,
        "max_linear_velocity_mps": 3.5,
        "max_angular_velocity_rps": 1.5,
        "velocity_pid_kp": 6.0,
        "velocity_pid_ki": 0.8,
        "velocity_pid_kd": 0.5,
        "friction_compensation": True,
        "feedforward": True,
        "wheel_slip_detection": True,
        "adaptive_gain": True,
        "profile_type": "s_curve",
        "acceleration_limit_mps2": 2.5,
        "jerk_limit_mps3": 15.0,
        "real_time": True,
    },
}


def get_velocity_control_spec(grade: str = "M") -> Dict:
    """获取AGV五级速度控制规格"""
    grades = list(AGV_VELOCITY_CONTROL_GRADES.keys())
    if grade not in grades:
        grade = "M"
    return AGV_VELOCITY_CONTROL_GRADES[grade]


# ============================================================================
# 数据结构
# ============================================================================

class VelocityProfileType(Enum):
    TRAPEZOIDAL = auto()
    S_CURVE = auto()
    POLYNOMIAL = auto()


@dataclass
class VelocityProfile1D:
    """单轴速度曲线"""
    profile_type: VelocityProfileType
    total_duration: float  # 秒
    initial_position: float = 0.0
    final_position: float = 0.0
    max_velocity: float = 0.0
    acceleration: float = 0.0  # 恒加速度段 (trapezoidal)
    jerk: float = 0.0  # 加加速度 (s_curve)
    times: np.ndarray = field(default_factory=lambda: np.array([]))
    positions: np.ndarray = field(default_factory=lambda: np.array([]))
    velocities: np.ndarray = field(default_factory=lambda: np.array([]))
    accelerations: np.ndarray = field(default_factory=lambda: np.array([]))

    def sample_at(self, t: float) -> Tuple[float, float, float]:
        """在时间t采样 (position, velocity, acceleration)"""
        if t <= 0:
            return self.initial_position, 0.0, 0.0
        if t >= self.total_duration:
            return self.final_position, 0.0, 0.0
        idx = np.searchsorted(self.times, t)
        idx = min(idx, len(self.positions) - 1)
        return self.positions[idx], self.velocities[idx], self.accelerations[idx]


@dataclass
class WheelVelocityCommand:
    """轮速控制指令"""
    left_velocity_rps: float = 0.0  # 左轮转速 (转/秒)
    right_velocity_rps: float = 0.0  # 右轮转速 (转/秒)
    timestamp: float = 0.0
    left_feedforward_nm: float = 0.0  # 左轮前馈力矩 (Nm)
    right_feedforward_nm: float = 0.0  # 右轮前馈力矩 (Nm)


@dataclass
class WheelVelocityState:
    """轮速状态"""
    left_velocity_rps: float = 0.0
    right_velocity_rps: float = 0.0
    left_position_rad: float = 0.0
    right_position_rad: float = 0.0
    left_slip: bool = False
    right_slip: bool = False
    timestamp: float = 0.0


@dataclass
class VelocityControllerState:
    """速度控制器状态"""
    left_error: float = 0.0
    right_error: float = 0.0
    left_integral: float = 0.0
    right_integral: float = 0.0
    left_derivative: float = 0.0
    right_derivative: float = 0.0
    left_output: float = 0.0
    right_output: float = 0.0
    left_adaptive_kp: float = 0.0
    right_adaptive_kp: float = 0.0


# ============================================================================
# S曲线速度规划器
# ============================================================================

class SVelocityProfilePlanner:
    """
    S曲线速度规划器
    支持梯形和S曲线速度曲线生成
    """

    def __init__(self, max_velocity: float, max_acceleration: float,
                 max_jerk: Optional[float] = None):
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk

    def plan(self, start_pos: float, end_pos: float,
             max_velocity: Optional[float] = None,
             max_acceleration: Optional[float] = None,
             max_jerk: Optional[float] = None) -> VelocityProfile1D:
        """
        规划S曲线速度剖面

        Args:
            start_pos: 起始位置
            end_pos: 终止位置
            max_velocity: 最大速度 (None=用默认值)
            max_acceleration: 最大加速度 (None=用默认值)
            max_jerk: 最大加加速度 (None=用默认值, S曲线模式)

        Returns:
            VelocityProfile1D 对象
        """
        max_v = max_velocity or self.max_velocity
        max_a = max_acceleration or self.max_acceleration
        max_j = max_jerk or self.max_jerk

        distance = abs(end_pos - start_pos)
        direction = np.sign(end_pos - start_pos)

        if distance < 1e-6:
            return VelocityProfile1D(
                profile_type=VelocityProfileType.TRAPEZOIDAL,
                total_duration=0.0,
                initial_position=start_pos,
                final_position=end_pos,
            )

        # 梯形规划
        if max_j is None or max_j <= 0:
            return self._plan_trapezoidal(
                start_pos, end_pos, max_v, max_a, direction)

        # S曲线规划
        return self._plan_s_curve(
            start_pos, end_pos, max_v, max_a, max_j, direction)

    def _plan_trapezoidal(self, start: float, end: float,
                          max_v: float, max_a: float,
                          direction: float) -> VelocityProfile1D:
        """梯形速度规划"""
        distance = abs(end - start)

        # 计算各段时间
        t_accel = max_v / max_a
        d_accel = 0.5 * max_a * t_accel * t_accel

        d_cruise = 0.0
        if distance <= 2 * d_accel:
            # 三角形轮廓
            t_accel = np.sqrt(distance / max_a)
            t_cruise = 0.0
            t_total = 2 * t_accel
        else:
            # 梯形轮廓
            d_cruise = distance - 2 * d_accel
            t_cruise = d_cruise / max_v
            t_total = 2 * t_accel + t_cruise

        # 生成轨迹点
        num_points = max(50, int(t_total * 100))
        dt = t_total / num_points
        times = np.linspace(0, t_total, num_points + 1)
        positions = np.zeros(num_points + 1)
        velocities = np.zeros(num_points + 1)
        accelerations = np.zeros(num_points + 1)

        t_acc = t_accel
        t_start_cruise = t_acc
        t_end_cruise = t_start_cruise + t_cruise

        for i, t in enumerate(times):
            if t <= t_acc:
                # 加速段
                acc = max_a
                vel = max_a * t
                pos = start + 0.5 * max_a * t * t * direction
            elif t <= t_end_cruise:
                # 匀速段
                acc = 0.0
                vel = max_v
                pos = start + (d_accel + (t - t_start_cruise) * max_v) * direction
            else:
                # 减速段
                tau = t - t_end_cruise
                acc = -max_a
                vel = max_v - max_a * tau
                pos = start + (2 * d_accel + d_cruise + max_v * tau - 0.5 * max_a * tau * tau) * direction

            positions[i] = pos
            velocities[i] = vel * direction
            accelerations[i] = acc * direction

        return VelocityProfile1D(
            profile_type=VelocityProfileType.TRAPEZOIDAL,
            total_duration=t_total,
            initial_position=start,
            final_position=end,
            max_velocity=max_v,
            acceleration=max_a,
            times=times,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
        )

    def _plan_s_curve(self, start: float, end: float,
                      max_v: float, max_a: float, max_j: float,
                      direction: float) -> VelocityProfile1D:
        """S曲线速度规划 (7段模型)"""
        distance = abs(end - start)

        # 计算加减速段时间
        t_aj = max_a / max_j  # 加加速/减减速时间
        d_aj = 0.5 * max_j * t_aj * t_aj  # 加加速段位移

        # 速度余量
        delta_v = max_a * t_aj

        # 计算匀加速度段时间
        t_aa = (max_v - delta_v) / max_a if max_v > delta_v else 0.0

        # 估算单趟位移
        d_1segment = 2 * d_aj + max_a * t_aa * t_aa / 2 + delta_v * t_aa
        d_full = 2 * d_1segment  # 加速+减速

        if distance <= d_full:
            # 限幅后重新计算
            # v_peak^2 = max_a * distance
            v_peak = np.sqrt(max_a * distance)
            v_peak = min(v_peak, max_v)
            t_aa_adj = max(0.0, (v_peak - delta_v) / max_a) if delta_v < v_peak else 0.0
            d_segment = 2 * (d_aj + 0.5 * max_a * t_aa_adj * t_aa_adj + delta_v * t_aa_adj)
            scale = distance / d_segment if d_segment > 0 else 1.0
            v_peak = v_peak * np.sqrt(scale) if scale > 0 else v_peak
            t_aa_actual = max(0.0, (v_peak - delta_v) / max_a) if delta_v < v_peak else 0.0
        else:
            v_peak = max_v
            t_aa_actual = t_aa

        # 7段S曲线时间
        Tj = t_aj
        Ta = t_aa_actual
        T = Tj + Ta + Tj  # 加速过程总时间
        t_accel = T

        d_accel = (max_j * Tj * Tj * Tj / 6 +
                   0.5 * max_a * Tj * Tj +
                   max_a * Tj * Ta +
                   0.5 * max_a * Ta * Ta +
                   max_a * Tj * Ta +
                   0.5 * max_a * Tj * Tj +
                   max_j * Tj * Tj * Tj / 6)

        if distance <= 2 * d_accel:
            t_total = 2 * t_accel
            t_cruise = 0.0
        else:
            d_cruise = distance - 2 * d_accel
            t_cruise = d_cruise / v_peak
            t_total = 2 * t_accel + t_cruise

        num_points = max(100, int(t_total * 100))
        dt = t_total / num_points
        times = np.linspace(0, t_total, num_points + 1)
        positions = np.zeros(num_points + 1)
        velocities = np.zeros(num_points + 1)
        accelerations = np.zeros(num_points + 1)

        for i, t in enumerate(times):
            if t <= t_accel:
                # 加速过程 (7段)
                if t <= Tj:
                    acc = max_j * t
                    vel = 0.5 * max_j * t * t
                    pos = start + max_j * t * t * t / 6 * direction
                elif t <= Tj + Ta:
                    tau = t - Tj
                    acc = max_a
                    vel = 0.5 * max_j * Tj * Tj + max_a * tau
                    pos = start + (max_j * Tj * Tj * Tj / 6 +
                                   0.5 * max_a * Tj * Tj +
                                   max_a * Tj * tau +
                                   0.5 * max_a * tau * tau) * direction
                elif t <= 2 * Tj + Ta:
                    tau = t - Tj - Ta
                    acc = max_a - max_j * tau
                    vel = (0.5 * max_j * Tj * Tj + max_a * Ta +
                           max_a * Tj * tau - 0.5 * max_j * tau * tau)
                    pos = start + (max_j * Tj * Tj * Tj / 6 +
                                   0.5 * max_a * Tj * Tj +
                                   max_a * Tj * Ta +
                                   0.5 * max_a * Ta * Ta +
                                   max_a * Tj * tau +
                                   0.5 * max_a * Tj * tau -
                                   max_j * tau * tau * tau / 6) * direction
                elif t <= 2 * Tj + 2 * Ta:
                    tau = t - 2 * Tj - Ta
                    acc = 0.0
                    vel = v_peak
                    pos = start + d_accel * direction + v_peak * tau * direction
                else:
                    tau = t - 2 * Tj - 2 * Ta - t_cruise
                    # 对称减速
                    if tau <= Tj:
                        acc = -max_j * tau
                        vel = v_peak - 0.5 * max_j * tau * tau
                    elif tau <= 2 * Tj + Ta:
                        tau2 = tau - Tj
                        acc = -max_a
                        vel = v_peak - 0.5 * max_j * Tj * Tj - max_a * tau2
                    else:
                        tau3 = tau - 2 * Tj - Ta
                        acc = -max_a + max_j * tau3
                        vel = (v_peak - 0.5 * max_j * Tj * Tj - max_a * Ta -
                               0.5 * max_a * Tj * Tj - max_j * Tj * Tj * Tj / 6)
                    pos = start + (d_accel + v_peak * t_cruise + (v_peak * tau -
                                   0.5 * max_j * (tau ** 3) / 6) * (tau <= Tj) +
                                   (v_peak * tau - 0.5 * max_j * Tj * Tj * Tj / 6 -
                                    max_a * (tau - Tj) * (tau - Tj) / 2 -
                                    0.5 * max_a * Tj * Tj) * ((tau > Tj) & (tau <= 2*Tj+Ta)) +
                                   (v_peak * tau - 0.5 * max_j * Tj * Tj * Tj / 6 -
                                    max_a * Ta * (2*Tj+Ta) +
                                    0.5 * max_a * Tj * Tj -
                                    max_a * Tj * (tau - 2*Tj-Ta) +
                                    0.5 * max_a * (tau - 2*Tj-Ta)**2 +
                                    max_j * (tau - 2*Tj-Ta)**3 / 6) * (tau > 2*Tj+Ta)) * direction
            elif t <= t_accel + t_cruise:
                # 匀速段
                acc = 0.0
                vel = v_peak
                tau = t - t_accel
                pos = start + (d_accel + v_peak * tau) * direction
            else:
                # 减速过程 (对称)
                tau = t - t_accel - t_cruise
                if tau <= Tj:
                    acc = -max_j * tau
                    vel = v_peak - 0.5 * max_j * tau * tau
                    pos = start + (d_accel + v_peak * t_cruise +
                                   v_peak * tau - max_j * tau**3 / 6) * direction
                elif tau <= 2*Tj + Ta:
                    tau2 = tau - Tj
                    acc = -max_a
                    vel = v_peak - 0.5 * max_j * Tj * Tj - max_a * tau2
                    pos = start + (d_accel + v_peak * t_cruise +
                                   v_peak * tau - max_j * Tj**3 / 6 -
                                   0.5 * max_a * tau2**2) * direction
                elif tau <= 2*Tj + 2*Ta:
                    tau3 = tau - 2*Tj - Ta
                    acc = max_j * tau3 - max_a
                    # 修正: S曲线减速
                    vel = v_peak - 0.5*max_j*Tj*Tj - max_a*Ta - 0.5*max_a*Tj*Tj
                    pos = start + (d_accel + v_peak * t_cruise +
                                   v_peak * tau - max_j*Tj**3/6 - max_a*Ta*(Tj+Ta) +
                                   0.5*max_a*Tj*Tj - 0.5*max_a*Tj*Tj) * direction
                else:
                    tau4 = min(tau - 2*Tj - 2*Ta, t_accel)
                    acc = 0.0
                    vel = 0.0
                    pos = end

            positions[i] = pos
            velocities[i] = vel * direction
            accelerations[i] = acc * direction

        return VelocityProfile1D(
            profile_type=VelocityProfileType.S_CURVE,
            total_duration=t_total,
            initial_position=start,
            final_position=end,
            max_velocity=v_peak,
            acceleration=max_a,
            jerk=max_j,
            times=times,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
        )


# ============================================================================
# 摩擦补偿模型
# ============================================================================

class FrictionCompensator:
    """库伦+粘滞摩擦补偿器"""

    def __init__(self, coulomb_friction: float = 0.5,
                 viscous_friction: float = 0.1,
                 stiction_friction: float = 0.8,
                 stiction_velocity: float = 0.01):
        self.coulomb_friction = coulomb_friction
        self.viscous_friction = viscous_friction
        self.stiction_friction = stiction_friction
        self.stiction_velocity = stiction_velocity

    def compensate(self, velocity: float, torque_direction: float = 1.0) -> float:
        """
        计算摩擦补偿力矩

        Args:
            velocity: 当前速度 (rad/s)
            torque_direction: 力矩方向 (+1/-1)

        Returns:
            摩擦补偿力矩 (Nm)
        """
        v = abs(velocity)

        if v < self.stiction_velocity:
            # 静摩擦区域
            return self.stiction_friction * torque_direction
        else:
            # 库伦+粘滞摩擦
            friction = (self.coulomb_friction * np.tanh(velocity / 0.1) +
                        self.viscous_friction * velocity)
            return friction

    def update_parameters(self, coulomb: float = None,
                         viscous: float = None):
        """在线更新摩擦参数"""
        if coulomb is not None:
            self.coulomb_friction = coulomb
        if viscous is not None:
            self.viscous_friction = viscous


# ============================================================================
# 轮速同步控制器
# ============================================================================

class WheelVelocitySynchronizer:
    """
    差速驱动轮速同步控制器

    确保左右轮速满足运动学约束,防止打滑和偏航
    """

    def __init__(self, wheelbase: float = 0.5,
                 left_radius: float = 0.07,
                 right_radius: float = 0.07,
                 max_wheel_velocity_rps: float = 50.0,
                 slip_threshold: float = 2.0,
                 sync_weight: float = 0.1):
        self.wheelbase = wheelbase
        self.left_radius = left_radius
        self.right_radius = right_radius
        self.max_wheel_velocity_rps = max_wheel_velocity_rps
        self.slip_threshold = slip_threshold  # rad/s 差阈值
        self.sync_weight = sync_weight  # 同步误差权重

        self.prev_left_vel = 0.0
        self.prev_right_vel = 0.0
        self.prev_time = None

    def compute_wheel_velocities(self,
                                  linear_velocity: float,
                                  angular_velocity: float,
                                  adaptive_slip: bool = False,
                                  left_raw: float = None,
                                  right_raw: float = None,
                                  dt: float = 0.01) -> WheelVelocityCommand:
        """
        根据线速度/角速度计算轮速指令

        Args:
            linear_velocity: 目标线速度 (m/s)
            angular_velocity: 目标角速度 (rad/s)
            adaptive_slip: 是否启用自适应打滑检测
            left_raw: 左轮原始测量速度 (rad/s)
            right_raw: 右轮原始测量速度 (rad/s)
            dt: 控制周期 (s)

        Returns:
            WheelVelocityCommand
        """
        # 差速运动学逆解
        v_l = (linear_velocity - angular_velocity * self.wheelbase / 2) / self.left_radius
        v_r = (linear_velocity + angular_velocity * self.wheelbase / 2) / self.right_radius

        # 限幅
        v_l = np.clip(v_l, -self.max_wheel_velocity_rps, self.max_wheel_velocity_rps)
        v_r = np.clip(v_r, -self.max_wheel_velocity_rps, self.max_wheel_velocity_rps)

        # 自适应打滑校正
        if adaptive_slip and left_raw is not None and right_raw is not None:
            v_l, v_r = self._correct_slip(v_l, v_r, left_raw, right_raw, dt)

        # 前馈力矩计算 (基于加速度前馈)
        dt_friction = dt if dt > 0 else 0.01
        left_accel = (v_l - self.prev_left_vel) / dt_friction
        right_accel = (v_r - self.prev_right_vel) / dt_friction

        # 简化前馈: 假设转动惯量 J ≈ 0.01 kg·m²
        J_wheel = 0.01
        left_ff = J_wheel * left_accel
        right_ff = J_wheel * right_accel

        self.prev_left_vel = v_l
        self.prev_right_vel = v_r

        return WheelVelocityCommand(
            left_velocity_rps=v_l,
            right_velocity_rps=v_r,
            timestamp=time.time(),
            left_feedforward_nm=left_ff,
            right_feedforward_nm=right_ff,
        )

    def _correct_slip(self, cmd_l: float, cmd_r: float,
                      raw_l: float, raw_r: float,
                      dt: float) -> Tuple[float, float]:
        """检测并校正打滑"""
        slip_l = abs(cmd_l - raw_l)
        slip_r = abs(cmd_r - raw_r)

        corrected_l = cmd_l
        corrected_r = cmd_r

        # 打滑检测与校正
        if slip_l > self.slip_threshold:
            # 左轮打滑: 使用测量值替代指令值
            corrected_l = raw_l * 0.8 + cmd_l * 0.2
        if slip_r > self.slip_threshold:
            corrected_r = raw_r * 0.8 + cmd_r * 0.2

        return corrected_l, corrected_r

    def check_slip(self, state: WheelVelocityState,
                   cmd: WheelVelocityCommand) -> WheelVelocityState:
        """检测打滑并更新状态"""
        state.left_slip = abs(cmd.left_velocity_rps - state.left_velocity_rps) > self.slip_threshold
        state.right_slip = abs(cmd.right_velocity_rps - state.right_velocity_rps) > self.slip_threshold
        return state


# ============================================================================
# 速度PID控制器
# ============================================================================

class VelocityPIDController:
    """
    先进速度PID控制器

    支持:
    - 标准PID / PI / PD / P 控制模式
    - 积分限幅 (抗饱和)
    - 微分滤波
    - 前馈控制
    - 自适应增益
    """

    def __init__(self,
                 kp: float = 3.0,
                 ki: float = 0.2,
                 kd: float = 0.1,
                 integral_limit: float = 50.0,
                 derivative_filter: float = 0.1,
                 output_limit: float = 100.0,
                 feedforward_gain: float = 1.0,
                 adaptive_gain: bool = False,
                 adaptation_rate: float = 0.01,
                 friction_compensator: Optional[FrictionCompensator] = None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.derivative_filter = derivative_filter
        self.output_limit = output_limit
        self.feedforward_gain = feedforward_gain
        self.adaptive_gain = adaptive_gain
        self.adaptation_rate = adaptation_rate
        self.friction_compensator = friction_compensator

        self.integral = 0.0
        self.prev_error = 0.0
        self.derivative = 0.0
        self.prev_measurement = 0.0
        self.adaptive_kp = kp

    def compute(self,
               setpoint: float,
               measurement: float,
               feedforward: float = 0.0,
               dt: float = 0.01) -> float:
        """
        计算PID控制输出

        Args:
            setpoint: 目标速度 (rps)
            measurement: 测量速度 (rps)
            feedforward: 前馈项
            dt: 控制周期 (s)

        Returns:
            控制输出 (力矩命令)
        """
        # NaN/Inf保护
        if not (np.isfinite(setpoint) and np.isfinite(measurement)):
            return 0.0

        error = setpoint - measurement

        # 积分项 (抗积分饱和)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        # 微分项 (带滤波)
        raw_derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        alpha = self.derivative_filter
        self.derivative = alpha * self.derivative + (1 - alpha) * raw_derivative

        # 自适应增益调整
        if self.adaptive_gain:
            error_magnitude = abs(error)
            kp_factor = 1.0 + self.adaptation_rate * error_magnitude
            self.adaptive_kp = self.kp * kp_factor

        # 摩擦补偿
        friction_comp = 0.0
        if self.friction_compensator is not None:
            friction_comp = self.friction_compensator.compensate(
                measurement, np.sign(error) if error != 0 else 1.0)

        # PID输出
        p_term = self.adaptive_kp * error
        i_term = self.ki * self.integral
        d_term = self.kd * self.derivative
        ff_term = self.feedforward_gain * feedforward

        output = p_term + i_term + d_term + ff_term + friction_comp
        output = np.clip(output, -self.output_limit, self.output_limit)

        self.prev_error = error
        self.prev_measurement = measurement

        return output

    def reset(self):
        """重置控制器状态"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.derivative = 0.0
        self.prev_measurement = 0.0
        self.adaptive_kp = self.kp


# ============================================================================
# AGV速度控制器 (主控制器)
# ============================================================================

class AGVVelocityController:
    """
    AGV完整速度控制器

    整合:
    - 运动学逆解 (线速度/角速度 → 轮速)
    - S曲线速度规划
    - 摩擦补偿
    - 双轮PID闭环
    - 打滑检测
    - AGV五级规格自适应
    """

    def __init__(self, grade: str = "M",
                 wheelbase: float = None,
                 left_wheel_radius: float = None,
                 right_wheel_radius: float = None):
        spec = get_velocity_control_spec(grade)
        self.grade = grade
        self.spec = spec

        # 默认轮子参数 (基于AGV等级)
        default_params = {
            "S": (0.3, 0.035, 0.035),
            "M": (0.5, 0.07, 0.07),
            "L": (0.7, 0.07, 0.07),
            "XL": (0.9, 0.0825, 0.0825),
            "XXL": (1.1, 0.095, 0.095),
        }
        wb, lr, rr = default_params.get(grade, default_params["M"])
        self.wheelbase = wheelbase if wheelbase is not None else wb
        self.left_radius = left_wheel_radius if left_wheel_radius is not None else lr
        self.right_radius = right_wheel_radius if right_wheel_radius is not None else rr

        # 组件初始化
        self.synchronizer = WheelVelocitySynchronizer(
            wheelbase=self.wheelbase,
            left_radius=self.left_radius,
            right_radius=self.right_radius,
            max_wheel_velocity_rps=spec["max_linear_velocity_mps"] * 2 / self.left_radius,
            slip_threshold=5.0 if spec["wheel_slip_detection"] else float("inf"),
        )

        self.friction_comp = None
        if spec["friction_compensation"]:
            self.friction_comp = FrictionCompensator(
                coulomb_friction=0.5 * {"S": 1, "M": 1, "L": 1.5, "XL": 2, "XXL": 3}[grade],
                viscous_friction=0.1 * {"S": 1, "M": 1, "L": 1.5, "XL": 2, "XXL": 3}[grade],
            )

        self.left_pid = VelocityPIDController(
            kp=spec["velocity_pid_kp"],
            ki=spec["velocity_pid_ki"],
            kd=spec["velocity_pid_kd"],
            feedforward_gain=1.0 if spec["feedforward"] else 0.0,
            adaptive_gain=spec["adaptive_gain"],
            friction_compensator=self.friction_comp,
        )

        self.right_pid = VelocityPIDController(
            kp=spec["velocity_pid_kp"],
            ki=spec["velocity_pid_ki"],
            kd=spec["velocity_pid_kd"],
            feedforward_gain=1.0 if spec["feedforward"] else 0.0,
            adaptive_gain=spec["adaptive_gain"],
            friction_compensator=self.friction_comp,
        )

        self.planner = SVelocityProfilePlanner(
            max_velocity=spec["max_linear_velocity_mps"],
            max_acceleration=spec["acceleration_limit_mps2"],
            max_jerk=spec["jerk_limit_mps3"],
        )

        # 状态
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0
        self.current_left_rps = 0.0
        self.current_right_rps = 0.0
        self.dt = 1.0 / spec["control_frequency_hz"]

        # 主动执行的速度剖面
        self.active_profile_linear: Optional[VelocityProfile1D] = None
        self.active_profile_angular: Optional[VelocityProfile1D] = None
        self.profile_start_time: Optional[float] = None

    @property
    def control_frequency(self) -> int:
        return self.spec["control_frequency_hz"]

    @property
    def max_linear_velocity(self) -> float:
        return self.spec["max_linear_velocity_mps"]

    @property
    def max_angular_velocity(self) -> float:
        return self.spec["max_angular_velocity_rps"]

    def plan_trajectory(self, start_pos: Tuple[float, float, float],
                       end_pos: Tuple[float, float, float],
                       max_linear_vel: float = None,
                       max_angular_vel: float = None,
                       max_accel: float = None) -> Tuple[VelocityProfile1D, VelocityProfile1D]:
        """
        规划线速度+角速度轨迹

        Args:
            start_pos: (x, y, theta) 起始位姿
            end_pos: (x, y, theta) 终止位姿
            max_linear_vel: 最大线速度 (None=用等级默认值)
            max_angular_vel: 最大角速度 (None=用等级默认值)
            max_accel: 最大加速度 (None=用等级默认值)

        Returns:
            (linear_profile, angular_profile)
        """
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        dtheta = end_pos[2] - start_pos[2]

        # 归一化角度差
        while dtheta > np.pi:
            dtheta -= 2 * np.pi
        while dtheta < -np.pi:
            dtheta += 2 * np.pi

        linear_distance = np.sqrt(dx * dx + dy * dy)
        max_v = max_linear_vel or self.max_linear_velocity
        max_w = max_angular_vel or self.max_angular_velocity
        max_a = max_accel or self.spec["acceleration_limit_mps2"]

        # 线速度剖面
        linear_profile = self.planner.plan(
            0.0, linear_distance,
            max_velocity=max_v,
            max_acceleration=max_a,
            max_jerk=self.spec["jerk_limit_mps3"] if self.spec["profile_type"] == "s_curve" else None,
        )

        # 角速度剖面 (时间同步)
        angular_profile = self.planner.plan(
            0.0, dtheta,
            max_velocity=min(abs(dtheta / linear_distance) * max_v * 0.5 if linear_distance > 0 else max_w, max_w),
            max_acceleration=max_a * 0.5,
            max_jerk=self.spec["jerk_limit_mps3"] * 0.5 if self.spec["profile_type"] == "s_curve" and self.spec["jerk_limit_mps3"] else None,
        )

        return linear_profile, angular_profile

    def start_trajectory(self, linear_profile: VelocityProfile1D,
                        angular_profile: VelocityProfile1D):
        """启动轨迹执行"""
        self.active_profile_linear = linear_profile
        self.active_profile_angular = angular_profile
        self.profile_start_time = time.time()

    def compute_openloop(self, linear_vel: float,
                         angular_vel: float) -> WheelVelocityCommand:
        """开环速度计算 (不执行PID)"""
        return self.synchronizer.compute_wheel_velocities(
            linear_vel, angular_vel,
            adaptive_slip=self.spec["wheel_slip_detection"],
        )

    def compute(self, target_linear_vel: float,
               target_angular_vel: float,
               measurement_left_rps: float,
               measurement_right_rps: float) -> Tuple[float, float, VelocityControllerState]:
        """
        计算闭环速度控制输出

        Args:
            target_linear_vel: 目标线速度 (m/s)
            target_angular_vel: 目标角速度 (rad/s)
            measurement_left_rps: 左轮测量速度 (rps)
            measurement_right_rps: 右轮测量速度 (rps)

        Returns:
            (left_torque_nm, right_torque_nm, controller_state)
        """
        # 更新测量
        # NaN/Inf保护
        meas_l = measurement_left_rps if np.isfinite(measurement_left_rps) else 0.0
        meas_r = measurement_right_rps if np.isfinite(measurement_right_rps) else 0.0
        self.current_left_rps = meas_l
        self.current_right_rps = meas_r

        # 运动学逆解 → 目标轮速
        cmd = self.synchronizer.compute_wheel_velocities(
            target_linear_vel, target_angular_vel,
            adaptive_slip=self.spec["wheel_slip_detection"],
            left_raw=measurement_left_rps,
            right_raw=measurement_right_rps,
            dt=self.dt,
        )

        # PID闭环
        left_torque = self.left_pid.compute(
            cmd.left_velocity_rps, measurement_left_rps,
            feedforward=cmd.left_feedforward_nm, dt=self.dt,
        )
        right_torque = self.right_pid.compute(
            cmd.right_velocity_rps, measurement_right_rps,
            feedforward=cmd.right_feedforward_nm, dt=self.dt,
        )

        # 状态
        state = VelocityControllerState(
            left_error=cmd.left_velocity_rps - meas_l,
            right_error=cmd.right_velocity_rps - meas_r,
            left_integral=self.left_pid.integral,
            right_integral=self.right_pid.integral,
            left_derivative=self.left_pid.derivative,
            right_derivative=self.right_pid.derivative,
            left_output=left_torque,
            right_output=right_torque,
            left_adaptive_kp=self.left_pid.adaptive_kp,
            right_adaptive_kp=self.right_pid.adaptive_kp,
        )

        self.current_linear_vel = target_linear_vel
        self.current_angular_vel = target_angular_vel

        return left_torque, right_torque, state

    def reset(self):
        """重置所有内部状态"""
        self.left_pid.reset()
        self.right_pid.reset()
        self.active_profile_linear = None
        self.active_profile_angular = None
        self.profile_start_time = None
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0

    def get_state(self) -> Dict:
        """获取控制器完整状态"""
        return {
            "grade": self.grade,
            "control_frequency_hz": self.control_frequency,
            "max_linear_velocity_mps": self.max_linear_velocity,
            "max_angular_velocity_rps": self.max_angular_velocity,
            "current_linear_vel": self.current_linear_vel,
            "current_angular_vel": self.current_angular_vel,
            "current_left_rps": self.current_left_rps,
            "current_right_rps": self.current_right_rps,
            "pid_kp": self.left_pid.kp,
            "pid_ki": self.left_pid.ki,
            "pid_kd": self.left_pid.kd,
            "friction_compensation": self.friction_comp is not None,
            "adaptive_gain": self.spec["adaptive_gain"],
            "slip_detection": self.spec["wheel_slip_detection"],
        }
