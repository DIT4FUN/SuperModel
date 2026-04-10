"""
AGV五级控制规格模块 (Grade-Aware AGV Control)
=============================================

为不同AGV等级(S/M/L/XL/XXL)提供适配的控制参数和控制策略。
包括:
- GradeAwareController: 五级感知控制器工厂
- GradeAwareSafetyMonitor: 五级感知安全监控
- GradeAwarePlanner: 五级感知轨迹规划
- GradePIDConfig: 五级PID配置

Author: SuperModel Team
Version: v2.46.1
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum


class AGVGrade(Enum):
    """AGV五极等级"""
    S = "S"       # 实验室/桌面 (30kg负载)
    M = "M"       # 小型仓储 (100kg负载)
    L = "L"       # 中型产线 (300kg负载)
    XL = "XL"     # 工业车间 (600kg负载)
    XXL = "XXL"   # 重载车间 (1200kg负载)


# ─── AGV五极控制规格表 ────────────────────────────────────────────────

GRADE_CONTROL_SPECS: Dict[AGVGrade, Dict[str, Any]] = {
    AGVGrade.S: {
        "max_velocity": 0.5,          # m/s
        "max_angular_velocity": 1.5,  # rad/s
        "max_acceleration": 0.5,      # m/s²
        "control_period": 0.050,       # s (50ms)
        "control_frequency": 20,        # Hz
        "pid_kp": 2.0,
        "pid_ki": 0.1,
        "pid_kd": 0.05,
        "pid_output_limit": 10.0,
        "pid_integral_limit": 5.0,
        "trajectory_mode": "line",     # 直线插补
        "planning_algorithm": "line",
        "safety_level": "PLd",
        "friction_compensation": False,
        "feedforward": False,
        "slip_detection": False,
        "adaptive_gain": False,
        "velocity_profile": "trapezoidal",
        "max_jerk": 0.0,              # m/s³ (无限制)
        "convergence_samples": 10000,
        "skill_timeout": 60.0,        # s
        "max_concurrent_skills": 1,
        "real_time_kernel": False,
        "fault_tolerance": False,
        "redundancy": False,
    },
    AGVGrade.M: {
        "max_velocity": 1.5,
        "max_angular_velocity": 3.0,
        "max_acceleration": 1.0,
        "control_period": 0.020,       # 20ms
        "control_frequency": 50,
        "pid_kp": 3.0,
        "pid_ki": 0.2,
        "pid_kd": 0.1,
        "pid_output_limit": 20.0,
        "pid_integral_limit": 10.0,
        "trajectory_mode": "trapezoidal",
        "planning_algorithm": "trapezoidal",
        "safety_level": "PLd",
        "friction_compensation": True,
        "feedforward": True,
        "slip_detection": True,
        "adaptive_gain": False,
        "velocity_profile": "trapezoidal",
        "max_jerk": 0.0,
        "convergence_samples": 50000,
        "skill_timeout": 45.0,
        "max_concurrent_skills": 2,
        "real_time_kernel": False,
        "fault_tolerance": False,
        "redundancy": False,
    },
    AGVGrade.L: {
        "max_velocity": 2.0,
        "max_angular_velocity": 2.5,
        "max_acceleration": 1.5,
        "control_period": 0.010,       # 10ms
        "control_frequency": 100,
        "pid_kp": 4.0,
        "pid_ki": 0.3,
        "pid_kd": 0.2,
        "pid_output_limit": 30.0,
        "pid_integral_limit": 15.0,
        "trajectory_mode": "s_curve",
        "planning_algorithm": "s_curve",
        "safety_level": "PLe",
        "friction_compensation": True,
        "feedforward": True,
        "slip_detection": True,
        "adaptive_gain": True,
        "velocity_profile": "s_curve",
        "max_jerk": 5.0,              # m/s³
        "convergence_samples": 200000,
        "skill_timeout": 30.0,
        "max_concurrent_skills": 3,
        "real_time_kernel": "PREEMPT_RT",
        "fault_tolerance": False,
        "redundancy": False,
    },
    AGVGrade.XL: {
        "max_velocity": 2.5,
        "max_angular_velocity": 2.0,
        "max_acceleration": 2.0,
        "control_period": 0.005,       # 5ms
        "control_frequency": 200,
        "pid_kp": 5.0,
        "pid_ki": 0.5,
        "pid_kd": 0.3,
        "pid_output_limit": 50.0,
        "pid_integral_limit": 25.0,
        "trajectory_mode": "s_curve",
        "planning_algorithm": "rrt",
        "safety_level": "PLe+SIL2",
        "friction_compensation": True,
        "feedforward": True,
        "slip_detection": True,
        "adaptive_gain": True,
        "velocity_profile": "s_curve",
        "max_jerk": 10.0,
        "convergence_samples": 500000,
        "skill_timeout": 15.0,
        "max_concurrent_skills": 4,
        "real_time_kernel": "Xenomai",
        "fault_tolerance": True,
        "redundancy": False,
    },
    AGVGrade.XXL: {
        "max_velocity": 3.0,
        "max_angular_velocity": 1.5,
        "max_acceleration": 2.5,
        "control_period": 0.001,       # 1ms
        "control_frequency": 1000,
        "pid_kp": 6.0,
        "pid_ki": 0.8,
        "pid_kd": 0.5,
        "pid_output_limit": 80.0,
        "pid_integral_limit": 40.0,
        "trajectory_mode": "minimum_snap",
        "planning_algorithm": "rrt_star",
        "safety_level": "PLe+SIL3",
        "friction_compensation": True,
        "feedforward": True,
        "slip_detection": True,
        "adaptive_gain": True,
        "velocity_profile": "s_curve",
        "max_jerk": 15.0,
        "convergence_samples": 1000000,
        "skill_timeout": 5.0,
        "max_concurrent_skills": 6,
        "real_time_kernel": "Xenomai+FPGA",
        "fault_tolerance": True,
        "redundancy": True,
    },
}


@dataclass
class GradePIDConfig:
    """五极PID配置"""
    grade: AGVGrade
    kp: float
    ki: float
    kd: float
    output_limit: float
    integral_limit: float
    derivative_filter: float = 0.0
    feedforward_gain: float = 0.0

    @classmethod
    def from_grade(cls, grade: AGVGrade) -> 'GradePIDConfig':
        """从等级获取PID配置"""
        spec = GRADE_CONTROL_SPECS[grade]
        return cls(
            grade=grade,
            kp=spec["pid_kp"],
            ki=spec["pid_ki"],
            kd=spec["pid_kd"],
            output_limit=spec["pid_output_limit"],
            integral_limit=spec["pid_integral_limit"],
            feedforward_gain=1.0 if spec["feedforward"] else 0.0,
        )


@dataclass
class GradeControllerConfig:
    """五极控制器配置"""
    grade: AGVGrade
    max_velocity: float
    max_angular_velocity: float
    max_acceleration: float
    control_period: float
    control_frequency: int
    trajectory_mode: str
    planning_algorithm: str
    safety_level: str
    friction_compensation: bool
    feedforward: bool
    slip_detection: bool
    adaptive_gain: bool
    velocity_profile: str
    max_jerk: float
    skill_timeout: float
    max_concurrent_skills: int
    real_time_kernel: str
    fault_tolerance: bool
    redundancy: bool

    @classmethod
    def from_grade(cls, grade: AGVGrade) -> 'GradeControllerConfig':
        """从等级获取控制器配置"""
        spec = GRADE_CONTROL_SPECS[grade]
        return cls(grade=grade, **spec)

    def get_control_period_ms(self) -> float:
        """获取控制周期(毫秒)"""
        return self.control_period * 1000.0


class GradeAwarePID:
    """
    五极感知PID控制器

    根据AGV等级自动配置PID参数，支持:
    - 积分抗饱和
    - 微分滤波
    - 前馈控制 (L级以上)
    - 自适应增益 (L级以上)
    """

    def __init__(self, grade: AGVGrade):
        self.config = GradePIDConfig.from_grade(grade)
        self.grade = grade
        self._integral = 0.0
        self._last_error = 0.0
        self._last_output = 0.0
        self._filtered_derivative = 0.0
        self._setpoint = 0.0

    def compute(
        self,
        error: float,
        dt: float,
        measurement: float = 0.0,
        feedforward: float = 0.0,
    ) -> float:
        """
        计算PID输出

        Args:
            error: 位置误差
            dt: 时间步长
            measurement: 当前测量值 (用于前馈)
            feedforward: 前馈输入

        Returns:
            控制输出
        """
        cfg = self.config

        # 积分项 (积分抗饱和)
        self._integral += error * dt
        self._integral = np.clip(
            self._integral, -cfg.integral_limit, cfg.integral_limit
        )

        # 微分项 (带滤波)
        raw_derivative = (error - self._last_error) / dt if dt > 0 else 0.0
        alpha = cfg.derivative_filter
        self._filtered_derivative = (
            alpha * self._filtered_derivative + (1 - alpha) * raw_derivative
        )

        # 前馈项
        ff_term = cfg.feedforward_gain * feedforward

        # PID输出
        output = (
            cfg.kp * error
            + cfg.ki * self._integral
            + cfg.kd * self._filtered_derivative
            + ff_term
        )

        # 输出限幅
        output = np.clip(output, -cfg.output_limit, cfg.output_limit)

        # 积分抗饱和 (克服效应)
        if cfg.ki > 0:
            diff = output - self._last_output
            if abs(diff) > 1e-6:
                self._integral -= diff / cfg.ki * 0.5

        self._last_error = error
        self._last_output = output
        return output

    def reset(self):
        """重置PID状态"""
        self._integral = 0.0
        self._last_error = 0.0
        self._last_output = 0.0
        self._filtered_derivative = 0.0

    def set_setpoint(self, setpoint: float):
        """设置设定值"""
        self._setpoint = setpoint

    def get_state(self) -> Dict[str, float]:
        """获取PID状态"""
        return {
            "integral": self._integral,
            "last_error": self._last_error,
            "last_output": self._last_output,
            "filtered_derivative": self._filtered_derivative,
            "setpoint": self._setpoint,
            "kp_e": self.config.kp * self._last_error,
            "ki_i": self.config.ki * self._integral,
            "kd_d": self.config.kd * self._filtered_derivative,
        }


class GradeAwareSafetyMonitor:
    """
    五极感知安全监控器

    根据AGV等级自动配置安全参数:
    - S/M: 基础速度/力监控
    - L/XL: 增强边界监控 + 打滑检测
    - XXL: 冗余监控 + 故障容错
    """

    def __init__(self, grade: AGVGrade):
        self.grade = grade
        self.config = GradeControllerConfig.from_grade(grade)
        self._estop_triggered = False
        self._fault_history: List[Tuple[str, float]] = []

        # 速度历史用于加速度估计
        self._velocity_history: List[float] = []

    def check_velocity(
        self,
        velocity: float,
        dt: float,
        timestamp: Optional[float] = None,
    ) -> Tuple[str, str]:
        """
        检查速度限制

        Returns:
            (level, message): 安全等级和消息
        """
        if timestamp is None:
            import time
            timestamp = time.time()

        max_v = self.config.max_velocity

        if abs(velocity) > max_v:
            return ("CRITICAL", f"速度 {abs(velocity):.2f}m/s 超过限制 {max_v}m/s")

        # 加速度检查
        self._velocity_history.append(velocity)
        if len(self._velocity_history) > 10:
            self._velocity_history.pop(0)

        if len(self._velocity_history) >= 2:
            prev_v = self._velocity_history[-2]
            accel = abs(velocity - prev_v) / dt
            if accel > self.config.max_acceleration * 1.5:
                return ("WARNING", f"加速度 {accel:.2f}m/s² 接近限制 {self.config.max_acceleration}m/s²")

        return ("NORMAL", "速度正常")

    def check_boundary(
        self,
        position: Tuple[float, float, float],
        boundary_min: Optional[np.ndarray] = None,
        boundary_max: Optional[np.ndarray] = None,
    ) -> Tuple[str, str]:
        """
        检查位置边界

        Args:
            position: (x, y, theta)
            boundary_min: 最小边界
            boundary_max: 最大边界

        Returns:
            (level, message)
        """
        if boundary_min is None:
            boundary_min = np.array([-100, -100, -np.pi])
        if boundary_max is None:
            boundary_max = np.array([100, 100, np.pi])

        pos = np.array(position)
        if np.any(pos < boundary_min) or np.any(pos > boundary_max):
            return ("CRITICAL", f"位置 {position} 超出边界")
        return ("NORMAL", "边界正常")

    def check_force(
        self,
        force_magnitude: float,
        torque_magnitude: float = 0.0,
    ) -> Tuple[str, str]:
        """
        检查力/力矩限制

        Args:
            force_magnitude: 合力大小 (N)
            torque_magnitude: 合力矩大小 (Nm)

        Returns:
            (level, message)
        """
        # 根据等级设置力阈值
        grade_force_thresholds = {
            AGVGrade.S: 30.0,
            AGVGrade.M: 60.0,
            AGVGrade.L: 100.0,
            AGVGrade.XL: 200.0,
            AGVGrade.XXL: 500.0,
        }
        force_threshold = grade_force_thresholds.get(self.grade, 100.0)

        if force_magnitude > force_threshold:
            return ("CRITICAL", f"力 {force_magnitude:.1f}N 超过阈值 {force_threshold}N")

        if torque_magnitude > force_threshold * 0.05:
            return ("CRITICAL", f"力矩 {torque_magnitude:.2f}Nm 超过阈值")

        if force_magnitude > force_threshold * 0.7:
            return ("WARNING", f"力 {force_magnitude:.1f}N 接近阈值")

        return ("NORMAL", "力/力矩正常")

    def check_slip(
        self,
        left_velocity: float,
        right_velocity: float,
        expected_velocity: float,
    ) -> Tuple[str, str]:
        """
        检查打滑 (L级以上支持)

        Args:
            left_velocity: 左轮速
            right_velocity: 右轮速
            expected_velocity: 期望速度

        Returns:
            (level, message)
        """
        if not self.config.slip_detection:
            return ("NORMAL", "打滑检测未启用")

        measured_avg = (left_velocity + right_velocity) / 2.0
        slip_ratio = abs(measured_avg - expected_velocity) / (abs(expected_velocity) + 1e-6)

        if slip_ratio > 0.3:
            return ("CRITICAL", f"检测到打滑: 滑转率 {slip_ratio:.1%}")
        elif slip_ratio > 0.15:
            return ("WARNING", f"轻微打滑: 滑转率 {slip_ratio:.1%}")

        return ("NORMAL", "无打滑")

    def get_emergency_level(self) -> str:
        """获取紧急停止级别"""
        if self._estop_triggered:
            return "EMERGENCY_STOP"
        return "NORMAL"

    def trigger_estop(self, reason: str = "unknown"):
        """触发紧急停止"""
        self._estop_triggered = True
        import time
        self._fault_history.append((reason, time.time()))

    def reset_estop(self):
        """重置紧急停止"""
        self._estop_triggered = False

    def is_safe(self) -> bool:
        """检查是否安全"""
        return not self._estop_triggered

    def get_capabilities(self) -> Dict[str, Any]:
        """获取该等级的安全能力"""
        return {
            "grade": self.grade.value,
            "max_velocity": self.config.max_velocity,
            "max_acceleration": self.config.max_acceleration,
            "friction_compensation": self.config.friction_compensation,
            "slip_detection": self.config.slip_detection,
            "fault_tolerance": self.config.fault_tolerance,
            "redundancy": self.config.redundancy,
            "safety_level": self.config.safety_level,
        }


class GradeAwareTrajectoryPlanner:
    """
    五极感知轨迹规划器

    根据AGV等级选择规划算法:
    - S: 直线插补
    - M: 梯形速度规划
    - L/XL: S曲线速度规划 + RRT
    - XXL: Minimum Snap + RRT*
    """

    def __init__(self, grade: AGVGrade):
        self.grade = grade
        self.config = GradeControllerConfig.from_grade(grade)
        self._current_trajectory: Optional[List[Dict]] = None

    def plan_line(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
    ) -> List[Dict]:
        """
        直线轨迹规划

        Args:
            start: (x, y, theta) 起始位姿
            end: (x, y, theta) 目标位姿

        Returns:
            轨迹点列表
        """
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = np.sqrt(dx**2 + dy**2)
        n_points = max(int(distance / 0.01), 2)

        trajectory = []
        for i in range(n_points + 1):
            alpha = i / n_points
            pt = {
                "x": start[0] + alpha * dx,
                "y": start[1] + alpha * dy,
                "theta": start[2] + alpha * self._angle_diff(end[2], start[2]),
                "v": self.config.max_velocity,
                "t": alpha * distance / self.config.max_velocity,
            }
            trajectory.append(pt)

        self._current_trajectory = trajectory
        return trajectory

    def plan_trapezoidal(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        max_velocity: Optional[float] = None,
        max_acceleration: Optional[float] = None,
    ) -> List[Dict]:
        """
        梯形速度规划

        Args:
            start: 起始位姿
            end: 目标位姿
            max_velocity: 最大速度
            max_acceleration: 最大加速度

        Returns:
            轨迹点列表
        """
        max_v = max_velocity or self.config.max_velocity
        max_a = max_acceleration or self.config.max_acceleration

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = np.sqrt(dx**2 + dy**2)

        # 梯形速度规划时间
        t_accel = max_v / max_a
        d_accel = 0.5 * max_a * t_accel * t_accel

        if d_accel >= distance / 2:
            # 三角形速度剖面
            t_accel = np.sqrt(distance / max_a)
            d_accel = 0.5 * max_a * t_accel * t_accel
            t_total = 2 * t_accel
        else:
            # 梯形速度剖面
            d_cruise = distance - 2 * d_accel
            t_cruise = d_cruise / max_v
            t_total = 2 * t_accel + t_cruise

        # 生成轨迹点
        n_points = max(int(distance / 0.01), 2)
        trajectory = []
        dt = t_total / n_points

        for i in range(n_points + 1):
            t = i * dt
            if t <= t_accel:
                v = max_a * t
                s = 0.5 * max_a * t * t
            elif t <= t_total - t_accel:
                v = max_v
                s = d_accel + max_v * (t - t_accel)
            else:
                t_dec = t - (t_total - t_accel)
                v = max_v - max_a * t_dec
                s = distance - 0.5 * max_a * t_dec * t_dec

            alpha = s / distance if distance > 0 else 0
            pt = {
                "x": start[0] + alpha * dx,
                "y": start[1] + alpha * dy,
                "theta": start[2] + alpha * self._angle_diff(end[2], start[2]),
                "v": max(0, v),
                "t": t,
            }
            trajectory.append(pt)

        self._current_trajectory = trajectory
        return trajectory

    def plan_s_curve(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        max_velocity: Optional[float] = None,
        max_acceleration: Optional[float] = None,
        max_jerk: Optional[float] = None,
    ) -> List[Dict]:
        """
        S曲线速度规划

        Args:
            start: 起始位姿
            end: 目标位姿
            max_velocity: 最大速度
            max_acceleration: 最大加速度
            max_jerk: 最大加加速度 (S/M级为0=梯形)

        Returns:
            轨迹点列表
        """
        max_v = max_velocity or self.config.max_velocity
        max_a = max_acceleration or self.config.max_acceleration
        max_j = max_jerk or self.config.max_jerk

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = np.sqrt(dx**2 + dy**2)

        if max_j <= 0:
            # 梯形模式
            return self.plan_trapezoidal(start, end, max_v, max_a)

        # S曲线七段规划
        # Ta=Vm/J, Da=Vm²/a
        Ta = max_a / max_j
        Da = 0.5 * max_a * Ta

        if Da >= distance / 2:
            # 需截断
            Ta = np.sqrt(distance / max_j)
            Da = 0.5 * max_a * Ta * Ta  # 这里Da实际是修正后的
            max_a_actual = max_j * Ta
            Da = 0.5 * max_a_actual * Ta * Ta
            if Da > distance / 2:
                Ta = np.sqrt(distance / max_j)
                Da = 0.5 * max_j * Ta * Ta

        D = distance - 2 * Da

        if D < 0:
            # 退化为三角形
            return self.plan_trapezoidal(start, end, max_v, max_a)

        T_total = 4 * Ta + 2 * D / max_v

        n_points = max(int(distance / 0.01), 2)
        trajectory = []
        dt = T_total / n_points

        for i in range(n_points + 1):
            t = i * dt
            # 七段S曲线 (简化版)
            if t < Ta:
                s = 0.5 * max_j * t**3
                v = 0.5 * max_j * t**2
                a = max_j * t
            elif t < 2 * Ta:
                t2 = t - Ta
                s = Da + max_a * t2 - (max_j * Ta**2) / 6 + 0.5 * max_j * t2**2 - max_j * Ta * t2
                v = max_a * t2 + (max_j * Ta**2) / 6
                a = max_a - max_j * t2
            elif t < 2 * Ta + D / max_v:
                t3 = t - 2 * Ta
                s = Da + max_a * Ta + max_v * t3
                v = max_v
                a = 0
            elif t < 3 * Ta + D / max_v:
                t4 = t - (2 * Ta + D / max_v)
                a_section = max_a - max_j * t4
                v = max_v - 0.5 * max_j * t4**2
                s = 2 * Da + D / max_v * max_v + max_v * t4 - (max_j * t4**3) / 6
                a = a_section
            elif t < 4 * Ta + D / max_v:
                t5 = t - (3 * Ta + D / max_v)
                v = 0.5 * max_j * (Ta - t5)**2
                s = distance - 0.5 * max_j * (Ta - t5)**3
                a = -max_j * (Ta - t5)
            else:
                s = distance
                v = 0
                a = 0

            alpha = s / distance if distance > 0 else 0
            pt = {
                "x": start[0] + alpha * dx,
                "y": start[1] + alpha * dy,
                "theta": start[2] + alpha * self._angle_diff(end[2], start[2]),
                "v": max(0, min(v, max_v)),
                "t": t,
                "a": a if abs(a) < 100 else 0,
            }
            trajectory.append(pt)

        self._current_trajectory = trajectory
        return trajectory

    def get_current_trajectory(self) -> Optional[List[Dict]]:
        """获取当前轨迹"""
        return self._current_trajectory

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """角度差"""
        d = a - b
        while d > np.pi:
            d -= 2 * np.pi
        while d < -np.pi:
            d += 2 * np.pi
        return d


def get_grade_control_spec(grade: AGVGrade) -> Dict[str, Any]:
    """获取AGV等级对应的控制规格"""
    return GRADE_CONTROL_SPECS.get(grade, {})


def list_grade_capabilities(grade: AGVGrade) -> Dict[str, Any]:
    """列出指定等级的所有控制能力"""
    spec = GRADE_CONTROL_SPECS[grade]
    return {
        "grade": grade.value,
        "control_frequency": spec["control_frequency"],
        "max_velocity": spec["max_velocity"],
        "max_acceleration": spec["max_acceleration"],
        "trajectory_mode": spec["trajectory_mode"],
        "planning_algorithm": spec["planning_algorithm"],
        "safety_level": spec["safety_level"],
        "friction_compensation": spec["friction_compensation"],
        "feedforward": spec["feedforward"],
        "slip_detection": spec["slip_detection"],
        "adaptive_gain": spec["adaptive_gain"],
        "real_time_kernel": spec["real_time_kernel"],
        "fault_tolerance": spec["fault_tolerance"],
        "redundancy": spec["redundancy"],
        "max_concurrent_skills": spec["max_concurrent_skills"],
        "skill_timeout": spec["skill_timeout"],
    }
