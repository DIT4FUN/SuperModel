"""
AGV速度控制模块 (Velocity Control)
==================================

为不同AGV等级(S/M/L/XL/XXL)提供适配的速度控制策略。
包括:
- S曲线速度规划器: 平滑加减速, 无冲击
- 摩擦补偿器: 静/动摩擦补偿
- 速度PID控制器: 精确速度闭环
- AGVVelocityController: 五级感知速度控制器

Author: SuperModel Team
Version: v2.60.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


# ─── AGV五级速度控制规格表 ────────────────────────────────────────────

AGV_VELOCITY_CONTROL_GRADES: Dict[str, Dict[str, Any]] = {
    "S": {
        "max_linear_velocity": 0.5,        # m/s
        "max_angular_velocity": 1.5,       # rad/s
        "max_acceleration": 0.5,           # m/s²
        "max_deceleration": 1.0,            # m/s²
        "jerk_limit": 2.0,                 # m/s³
        "control_frequency": 50,           # Hz
        "velocity_kp": 2.0,
        "velocity_ki": 0.1,
        "velocity_kd": 0.05,
        "friction_compensation": False,
        "slip_ratio_limit": 0.15,
        "wheelbase": 0.3,                   # m
        "wheel_radius": 0.05,               # m
        "encoder_resolution": 4096,         # pulses/rev
    },
    "M": {
        "max_linear_velocity": 1.5,         # m/s
        "max_angular_velocity": 2.0,       # rad/s
        "max_acceleration": 1.0,            # m/s²
        "max_deceleration": 2.0,            # m/s²
        "jerk_limit": 5.0,                 # m/s³
        "control_frequency": 100,          # Hz
        "velocity_kp": 3.0,
        "velocity_ki": 0.2,
        "velocity_kd": 0.1,
        "friction_compensation": True,
        "slip_ratio_limit": 0.12,
        "wheelbase": 0.5,                   # m
        "wheel_radius": 0.08,              # m
        "encoder_resolution": 8192,         # pulses/rev
    },
    "L": {
        "max_linear_velocity": 3.0,         # m/s
        "max_angular_velocity": 2.5,        # rad/s
        "max_acceleration": 2.0,            # m/s²
        "max_deceleration": 3.0,           # m/s²
        "jerk_limit": 10.0,                # m/s³
        "control_frequency": 200,           # Hz
        "velocity_kp": 4.0,
        "velocity_ki": 0.3,
        "velocity_kd": 0.15,
        "friction_compensation": True,
        "slip_ratio_limit": 0.10,
        "wheelbase": 0.8,                   # m
        "wheel_radius": 0.12,              # m
        "encoder_resolution": 16384,        # pulses/rev
    },
    "XL": {
        "max_linear_velocity": 5.0,         # m/s
        "max_angular_velocity": 3.0,        # rad/s
        "max_acceleration": 3.0,            # m/s²
        "max_deceleration": 5.0,           # m/s²
        "jerk_limit": 20.0,                # m/s³
        "control_frequency": 500,           # Hz
        "velocity_kp": 5.0,
        "velocity_ki": 0.5,
        "velocity_kd": 0.2,
        "friction_compensation": True,
        "slip_ratio_limit": 0.08,
        "wheelbase": 1.0,                   # m
        "wheel_radius": 0.15,              # m
        "encoder_resolution": 32768,        # pulses/rev
    },
    "XXL": {
        "max_linear_velocity": 8.0,         # m/s
        "max_angular_velocity": 3.5,        # rad/s
        "max_acceleration": 5.0,            # m/s²
        "max_deceleration": 8.0,           # m/s²
        "jerk_limit": 40.0,                # m/s³
        "control_frequency": 1000,          # Hz
        "velocity_kp": 6.0,
        "velocity_ki": 0.8,
        "velocity_kd": 0.3,
        "friction_compensation": True,
        "slip_ratio_limit": 0.05,
        "wheelbase": 1.5,                   # m
        "wheel_radius": 0.20,              # m
        "encoder_resolution": 65536,        # pulses/rev
    },
}


def get_velocity_control_spec(grade: str = "M") -> Dict[str, Any]:
    """获取指定AGV等级的速度控制规格"""
    if grade not in AGV_VELOCITY_CONTROL_GRADES:
        raise ValueError(f"未知AGV等级: {grade}，可用等级: {list(AGV_VELOCITY_CONTROL_GRADES.keys())}")
    return AGV_VELOCITY_CONTROL_GRADES[grade]


def list_velocity_capabilities() -> List[Dict[str, Any]]:
    """列出所有AGV等级的速度控制能力"""
    return [
        {"grade": grade, **spec}
        for grade, spec in AGV_VELOCITY_CONTROL_GRADES.items()
    ]


# ─── 速度规划 ──────────────────────────────────────────────────────────

class VelocityProfileType(Enum):
    """速度曲线类型"""
    TRAPEZOIDAL = "trapezoidal"   # 梯形: 等速段
    S_CURVE = "s_curve"          # S曲线: 平滑无冲击


@dataclass
class VelocityProfile1D:
    """
    一维速度规划器 (S曲线)
    
    生成平滑的速度曲线，限制加速度和急动度(jerk)，消除机械冲击。
    
    Attributes:
        max_velocity: 最大速度 m/s
        max_acceleration: 最大加速度 m/s²
        jerk_limit: 最大急动度 m/s³
    """
    max_velocity: float = 1.0
    max_acceleration: float = 1.0
    jerk_limit: float = 5.0
    
    def plan(
        self,
        start_pos: float,
        end_pos: float,
        start_vel: float = 0.0,
        end_vel: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """
        规划S曲线速度轨迹
        
        Args:
            start_pos: 起始位置 m
            end_pos: 目标位置 m
            start_vel: 起始速度 m/s
            end_vel: 终止速度 m/s
        
        Returns:
            Dict含:
                positions: 位置数组 (N,)
                velocities: 速度数组 (N,)
                accelerations: 加速度数组 (N,)
                timestamps: 时间戳数组 (N,)
                duration: 总时长 s
        """
        distance = abs(end_pos - start_pos)
        sign = np.sign(end_pos - start_pos)
        
        # 计算各阶段时间
        # 加速段时间: ta = (max_a / jerk)
        ta = self.max_acceleration / self.jerk_limit if self.jerk_limit > 0 else 0.0
        # 等加速段速度增量: dv_accel = 0.5 * jerk * ta^2
        dv_accel = 0.5 * self.jerk_limit * ta * ta
        
        # 对称三角形速度曲线(适用于短距离)
        # 若所需速度增量 > 2*dv_accel, 需要等速段
        if dv_accel >= abs(start_vel) and dv_accel >= abs(end_vel):
            # 加速峰值速度
            v_peak = dv_accel
            if v_peak > self.max_velocity:
                v_peak = self.max_velocity
                # 重新计算加速时间
                ta = np.sqrt(2.0 * v_peak / self.jerk_limit)
            
            # 三角形速度曲线: 加速 + 减速, 无等速段
            # 加速段时间 = 减速段时间 = ta
            # 等减速段时间 = td = ta
            total_time = 2.0 * ta
            
            # 生成轨迹
            N = max(20, int(total_time * 200))  # 200Hz采样
            dt = total_time / N
            
            times = np.linspace(0, total_time, N + 1)
            positions = np.zeros(N + 1)
            velocities = np.zeros(N + 1)
            accelerations = np.zeros(N + 1)
            
            # 前半段: 加速 (抛物线)
            for i, t in enumerate(times):
                if t <= ta:
                    acc = self.jerk_limit * t
                    vel = 0.5 * self.jerk_limit * t * t + start_vel
                    pos = start_pos + sign * (start_vel * t + 0.5 * self.jerk_limit * t**3 / 3.0)
                else:
                    # 后半段: 减速
                    t_dec = t - ta
                    acc = self.jerk_limit * ta - self.jerk_limit * t_dec
                    vel = dv_accel + start_vel - 0.5 * self.jerk_limit * t_dec * t_dec
                    pos = (start_pos + sign * (
                        start_vel * ta + self.jerk_limit * ta**3 / 6.0 +
                        (dv_accel + start_vel) * t_dec - self.jerk_limit * t_dec**3 / 6.0
                    ))
                
                if i > 0:
                    dt_i = times[i] - times[i-1]
                    positions[i] = positions[i-1] + vel * dt_i
                else:
                    positions[i] = pos
                velocities[i] = vel
                accelerations[i] = acc
            
            return {
                "positions": positions,
                "velocities": velocities,
                "accelerations": accelerations,
                "timestamps": times,
                "duration": float(total_time),
            }
        else:
            # 完整S曲线: 加速 + 等速 + 减速
            # 计算加速段时间
            ta = self.max_acceleration / self.jerk_limit
            # 计算等加速段达到的速度
            dv_accel = 0.5 * self.jerk_limit * ta * ta
            
            # 达到max_velocity所需的加速时间
            ta_to_vmax = np.sqrt((self.max_velocity - start_vel) / self.jerk_limit + (self.max_velocity - end_vel) / self.jerk_limit)
            if ta_to_vmax > ta * 2:
                ta_to_vmax = ta * 2
            
            # 计算各阶段
            # 等速段时间
            v_cruise = min(self.max_velocity, dv_accel + start_vel, dv_accel + end_vel)
            t_accel = np.sqrt((v_cruise - start_vel) / self.jerk_limit) if self.jerk_limit > 0 else 0.0
            t_decel = np.sqrt((v_cruise - end_vel) / self.jerk_limit) if self.jerk_limit > 0 else 0.0
            
            # 检查距离是否足够
            d_accel = (start_vel + v_cruise) * t_accel / 2.0
            d_decel = (end_vel + v_cruise) * t_decel / 2.0
            d_cruise_needed = distance - d_accel - d_decel
            
            if d_cruise_needed < 0:
                # 降低巡航速度
                v_cruise = dv_accel
                t_accel = ta
                t_decel = ta
                d_accel = dv_accel * ta
                d_decel = dv_accel * ta
                d_cruise_needed = distance - 2.0 * d_accel
            
            t_cruise = max(0, d_cruise_needed / v_cruise) if v_cruise > 0 else 0.0
            total_time = t_accel + t_cruise + t_decel
            
            N = max(20, int(total_time * 200))
            dt = total_time / N
            times = np.linspace(0, total_time, N + 1)
            positions = np.zeros(N + 1)
            velocities = np.zeros(N + 1)
            accelerations = np.zeros(N + 1)
            
            for i, t in enumerate(times):
                if t < t_accel:
                    # 加速段
                    acc = self.jerk_limit * t
                    vel = start_vel + 0.5 * self.jerk_limit * t * t
                elif t < t_accel + t_cruise:
                    # 等速段
                    acc = 0.0
                    vel = v_cruise
                else:
                    # 减速段
                    t_dec = t - t_accel - t_cruise
                    acc = self.jerk_limit * t_decel - self.jerk_limit * t_dec
                    vel = v_cruise - 0.5 * self.jerk_limit * t_dec * t_dec
                
                if i > 0:
                    dt_i = times[i] - times[i-1]
                    positions[i] = positions[i-1] + velocities[i-1] * dt_i
                velocities[i] = max(0, vel) * sign
                accelerations[i] = acc * sign
            
            # 修正最后一点
            positions[-1] = end_pos
            velocities[-1] = end_vel
            
            return {
                "positions": positions,
                "velocities": velocities,
                "accelerations": accelerations,
                "timestamps": times,
                "duration": float(total_time),
            }


@dataclass
class WheelVelocityCommand:
    """车轮速度指令"""
    left_velocity: float   # m/s
    right_velocity: float  # m/s
    timestamp: float


@dataclass
class WheelVelocityState:
    """车轮速度状态"""
    left_velocity: float    # m/s (实测)
    right_velocity: float   # m/s (实测)
    left_position: float     # m (累计行程)
    right_position: float    # m
    timestamp: float
    left_slip: float = 0.0   # 滑移率
    right_slip: float = 0.0


@dataclass
class VelocityControllerState:
    """速度控制器状态"""
    left_velocity_error: float
    right_velocity_error: float
    left_integral: float
    right_integral: float
    left_output: float
    right_output: float
    left_slip_detected: bool
    right_slip_detected: bool
    saturation: bool


# ─── 摩擦补偿 ──────────────────────────────────────────────────────────

class FrictionCompensator:
    """
    摩擦补偿器
    
    对静摩擦和动摩擦进行补偿，消除低速爬行现象。
    
    算法: Stribeck摩擦模型
    F_fric = Fc * sign(v) + Fs * exp(-|v|/vs) * sign(v) + Fv * v
    
    Attributes:
        static_friction: 静摩擦力 N
        coulomb_friction: 库仑摩擦力 N
        viscous_friction: 粘性摩擦系数 N·s/m
        stribeck_velocity: Stribeck速度 m/s
    """
    
    def __init__(
        self,
        static_friction: float = 1.0,
        coulomb_friction: float = 0.8,
        viscous_friction: float = 0.1,
        stribeck_velocity: float = 0.01,
    ):
        self.static_friction = static_friction
        self.coulomb_friction = coulomb_friction
        self.viscous_friction = viscous_friction
        self.stribeck_velocity = stribeck_velocity
        self._last_velocity = 0.0
    
    def compensate(self, velocity: float, mass: float = 1.0) -> float:
        """
        计算摩擦补偿力
        
        Args:
            velocity: 当前速度 m/s
            mass: 质量 kg
        
        Returns:
            补偿力 N (正值为推力方向)
        """
        v = velocity
        sign_v = np.sign(v) if abs(v) > 1e-9 else 0.0
        
        # Stribeck项: exp(-|v|/vs)
        stribeck = np.exp(-abs(v) / self.stribeck_velocity) if self.stribeck_velocity > 1e-9 else 0.0
        
        # 总摩擦力: 静摩擦 + Stribeck修正 + 粘性摩擦
        friction = (
            self.coulomb_friction * sign_v +
            (self.static_friction - self.coulomb_friction) * stribeck * sign_v +
            self.viscous_friction * v
        )
        
        # 静止时不补偿(静摩擦处理)
        if abs(v) < 1e-9 and abs(self._last_velocity) < 1e-9:
            return 0.0
        
        self._last_velocity = v
        return -friction * mass
    
    def reset(self):
        """重置内部状态"""
        self._last_velocity = 0.0


# ─── 速度PID控制器 ─────────────────────────────────────────────────────

class VelocityPIDController:
    """
    速度PID控制器
    
    带积分抗饱和和微分滤波的速度控制器。
    
    Attributes:
        kp: 比例增益
        ki: 积分增益
        kd: 微分增益
        output_limit: 输出限幅
        integral_limit: 积分限幅
        derivative_filter_alpha: 微分滤波器系数 [0,1]
    """
    
    def __init__(
        self,
        kp: float = 3.0,
        ki: float = 0.2,
        kd: float = 0.1,
        output_limit: float = 10.0,
        integral_limit: float = 5.0,
        derivative_filter_alpha: float = 0.3,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.alpha = derivative_filter_alpha
        
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_measurement = 0.0
        self._filtered_derivative = 0.0
    
    def update(
        self,
        setpoint: float,
        measurement: float,
        dt: float,
    ) -> float:
        """
        计算PID控制输出
        
        Args:
            setpoint: 目标速度 m/s
            measurement: 实测速度 m/s
            dt: 控制周期 s
        
        Returns:
            控制输出 (电压/PWM占空比)
        """
        error = setpoint - measurement
        
        # 积分项 (抗积分饱和)
        self._integral = np.clip(
            self._integral + error * dt,
            -self.integral_limit / self.ki if self.ki > 1e-9 else -self.integral_limit,
            self.integral_limit / self.ki if self.ki > 1e-9 else self.integral_limit,
        )
        
        # 微分项 (带低通滤波)
        raw_derivative = (error - self._prev_error) / dt if dt > 1e-9 else 0.0
        self._filtered_derivative = (
            self.alpha * raw_derivative +
            (1 - self.alpha) * self._filtered_derivative
        )
        
        # PID输出
        output = self.kp * error + self.ki * self._integral + self.kd * self._filtered_derivative
        
        # 输出限幅
        output = np.clip(output, -self.output_limit, self.output_limit)
        
        self._prev_error = error
        self._prev_measurement = measurement
        
        return output
    
    def reset(self):
        """重置PID状态"""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_measurement = 0.0
        self._filtered_derivative = 0.0


# ─── AGV速度控制器 ─────────────────────────────────────────────────────

class AGVVelocityController:
    """
    AGV速度控制器
    
    为差分驱动AGV提供完整的速度控制:
    - S曲线速度规划
    - 左右轮独立PID速度闭环
    - 可选摩擦补偿
    - 滑移检测与容错
    
    Attributes:
        grade: AGV等级 (S/M/L/XL/XXL)
        kinematics_type: 运动学类型 ("differential" | "mecanum")
    """
    
    def __init__(
        self,
        grade: str = "M",
        kinematics_type: str = "differential",
        enable_friction_comp: bool = True,
    ):
        if grade not in AGV_VELOCITY_CONTROL_GRADES:
            raise ValueError(f"未知AGV等级: {grade}")
        
        self.grade = grade
        self.kinematics_type = kinematics_type
        self.spec = AGV_VELOCITY_CONTROL_GRADES[grade]
        
        self.enable_friction_comp = (
            enable_friction_comp and self.spec["friction_compensation"]
        )
        
        # 控制器
        self.left_pid = VelocityPIDController(
            kp=self.spec["velocity_kp"],
            ki=self.spec["velocity_ki"],
            kd=self.spec["velocity_kd"],
            output_limit=self.spec["max_acceleration"],
        )
        self.right_pid = VelocityPIDController(
            kp=self.spec["velocity_kp"],
            ki=self.spec["velocity_ki"],
            kd=self.spec["velocity_kd"],
            output_limit=self.spec["max_acceleration"],
        )
        
        # 速度规划器
        self.profile_planner = VelocityProfile1D(
            max_velocity=self.spec["max_linear_velocity"],
            max_acceleration=self.spec["max_acceleration"],
            jerk_limit=self.spec["jerk_limit"],
        )
        
        # 摩擦补偿
        self.friction_comp = (
            FrictionCompensator() if self.enable_friction_comp else None
        )
        
        # 当前状态
        self._current_left_vel = 0.0
        self._current_right_vel = 0.0
        self._current_left_pos = 0.0
        self._current_right_pos = 0.0
        self._slip_threshold = self.spec["slip_ratio_limit"]
        self._last_timestamp = None
        
        # 轨迹缓冲
        self._trajectory: Optional[Dict] = None
        self._traj_idx = 0
    
    def plan_trajectory(
        self,
        start_pos: Tuple[float, float, float],
        end_pos: Tuple[float, float, float],
    ) -> Dict[str, np.ndarray]:
        """
        规划线速度S曲线轨迹
        
        Args:
            start_pos: 起始位姿 (x, y, theta) m, m, rad
            end_pos: 目标位姿 (x, y, theta) m, m, rad
        
        Returns:
            轨迹字典
        """
        # 计算直线距离
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = np.sqrt(dx * dx + dy * dy)
        
        # 方向
        direction = np.arctan2(dy, dx)
        heading = start_pos[2]
        
        return self.profile_planner.plan(
            start_pos=0.0,
            end_pos=distance,
            start_vel=0.0,
            end_vel=0.0,
        )
    
    def compute_wheel_velocities(
        self,
        linear_velocity: float,
        angular_velocity: float,
    ) -> Tuple[float, float]:
        """
        运动学逆解: 线速度+角速度 → 左右轮速度
        
        Args:
            linear_velocity: 目标线速度 m/s
            angular_velocity: 目标角速度 rad/s
        
        Returns:
            (left_wheel_vel, right_wheel_vel) m/s
        """
        wheelbase = self.spec["wheelbase"]
        
        if self.kinematics_type == "differential":
            # 差分驱动逆运动学
            left = linear_velocity - angular_velocity * wheelbase / 2.0
            right = linear_velocity + angular_velocity * wheelbase / 2.0
        elif self.kinematics_type == "mecanum":
            # 全向麦克纳姆轮
            gamma = np.radians(45)  # 滚轮角度
            factor = 1.0 / (np.cos(gamma) + np.sin(gamma))
            left = (linear_velocity - angular_velocity * wheelbase / 2.0) * factor
            right = (linear_velocity + angular_velocity * wheelbase / 2.0) * factor
        else:
            left = linear_velocity
            right = linear_velocity
        
        return left, right
    
    def update(
        self,
        cmd_linear: float,
        cmd_angular: float,
        meas_left_vel: float,
        meas_right_vel: float,
        dt: float,
    ) -> WheelVelocityCommand:
        """
        更新速度控制
        
        Args:
            cmd_linear: 目标线速度 m/s
            cmd_angular: 目标角速度 rad/s
            meas_left_vel: 实测左轮速度 m/s
            meas_right_vel: 实测右轮速度 m/s
            dt: 控制周期 s
        
        Returns:
            车轮速度指令
        """
        # 运动学逆解
        target_left, target_right = self.compute_wheel_velocities(
            cmd_linear, cmd_angular
        )
        
        # 摩擦补偿
        if self.friction_comp is not None:
            target_left += self.friction_comp.compensate(meas_left_vel) / self.spec.get("mass", 1.0)
            target_right += self.friction_comp.compensate(meas_right_vel) / self.spec.get("mass", 1.0)
        
        # PID控制
        left_output = self.left_pid.update(target_left, meas_left_vel, dt)
        right_output = self.right_pid.update(target_right, meas_right_vel, dt)
        
        # 滑移检测
        slip_left = abs(target_left - meas_left_vel) / (abs(target_left) + 1e-6)
        slip_right = abs(target_right - meas_right_vel) / (abs(target_right) + 1e-6)
        
        # 更新状态
        self._current_left_vel = meas_left_vel
        self._current_right_vel = meas_right_vel
        self._current_left_pos += meas_left_vel * dt
        self._current_right_pos += meas_right_vel * dt
        
        return WheelVelocityCommand(
            left_velocity=meas_left_vel + left_output,
            right_velocity=meas_right_vel + right_output,
            timestamp=0.0,
        )
    
    def get_state(self) -> VelocityControllerState:
        """获取控制器状态"""
        return VelocityControllerState(
            left_velocity_error=self.left_pid._prev_error,
            right_velocity_error=self.right_pid._prev_error,
            left_integral=self.left_pid._integral,
            right_integral=self.right_pid._integral,
            left_output=self.left_pid._prev_error * self.left_pid.kp,
            right_output=self.right_pid._prev_error * self.right_pid.kp,
            left_slip_detected=abs(self.left_pid._prev_error) > self._slip_threshold,
            right_slip_detected=abs(self.right_pid._prev_error) > self._slip_threshold,
            saturation=(
                abs(self.left_pid._integral) >= self.left_pid.integral_limit or
                abs(self.right_pid._integral) >= self.right_pid.integral_limit
            ),
        )
    
    def reset(self):
        """重置所有内部状态"""
        self.left_pid.reset()
        self.right_pid.reset()
        self._current_left_vel = 0.0
        self._current_right_vel = 0.0
        self._current_left_pos = 0.0
        self._current_right_pos = 0.0
        self._trajectory = None
        self._traj_idx = 0
        if self.friction_comp:
            self.friction_comp.reset()
