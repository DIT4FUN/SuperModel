"""
电机控制模块 (Motor Control)
支持直流电机、BLDC电机、步进电机、舵机的控制
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum


class MotorType(Enum):
    """电机类型"""
    DC = "dc"                   # 直流电机
    BLDC = "bldc"              # 无刷直流电机
    STEPPER = "stepper"        # 步进电机
    SERVO = "servo"            # 舵机/伺服电机
    LINEAR = "linear"          # 直线电机


class MotorControlMode(Enum):
    """控制模式"""
    POSITION = "position"      # 位置控制
    VELOCITY = "velocity"     # 速度控制
    TORQUE = "torque"         # 力矩控制
    TRAJECTORY = "trajectory" # 轨迹跟踪


@dataclass
class MotorState:
    """电机状态"""
    timestamp: float
    motor_id: str
    # 位置 (rad 或 m)
    position: float
    # 速度 (rad/s 或 m/s)
    velocity: float
    # 力矩/电流 (Nm 或 A)
    torque: float
    # 目标位置
    target_position: float = 0.0
    # 目标速度
    target_velocity: float = 0.0
    # 错误标志
    error: int = 0
    # 温度 (°C)
    temperature: float = 25.0


@dataclass
class PIDConfig:
    """PID配置"""
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    # 输出限幅
    output_limit: float = 100.0
    # 积分限幅
    integral_limit: float = 50.0
    # 微分滤波器系数
    derivative_filter: float = 0.1


class PIDController:
    """PID控制器"""

    def __init__(self, config: PIDConfig):
        self.config = config
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_derivative = 0.0

    def compute(self, setpoint: float, measured: float, dt: float) -> float:
        """计算PID输出"""
        error = setpoint - measured

        # 比例
        p_out = self.config.kp * error

        # 积分
        self._integral += error * dt
        self._integral = np.clip(self._integral, -self.config.integral_limit, self.config.integral_limit)
        i_out = self.config.ki * self._integral

        # 微分 (带滤波)
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        derivative_filtered = (self.config.derivative_filter * derivative +
                                (1 - self.config.derivative_filter) * self._prev_derivative)
        d_out = self.config.kd * derivative_filtered

        self._prev_error = error
        self._prev_derivative = derivative_filtered

        output = p_out + i_out + d_out
        return np.clip(output, -self.config.output_limit, self.config.output_limit)

    def reset(self):
        """重置积分项"""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_derivative = 0.0


class Motor:
    """电机基类"""

    def __init__(self, motor_id: str, motor_type: MotorType, config: Optional[Dict] = None):
        self.motor_id = motor_id
        self.motor_type = motor_type
        self.config = config or {}

        # 电机参数
        self.max_velocity = self.config.get("max_velocity", 10.0)
        self.max_torque = self.config.get("max_torque", 10.0)
        self.gear_ratio = self.config.get("gear_ratio", 1.0)

        # 状态
        self._position = 0.0
        self._velocity = 0.0
        self._torque = 0.0
        self._target_position = 0.0
        self._target_velocity = 0.0
        self._control_mode = MotorControlMode.POSITION
        self._enabled = True
        self._error = 0

        # PID控制器
        self._pid = PIDController(PIDConfig(
            kp=self.config.get("kp", 10.0),
            ki=self.config.get("ki", 0.1),
            kd=self.config.get("kd", 1.0)
        ))

    def enable(self):
        """使能电机"""
        self._enabled = True

    def disable(self):
        """禁用电机"""
        self._enabled = False

    def set_target(self, target: float, mode: Optional[MotorControlMode] = None):
        """设置目标值"""
        if mode:
            self._control_mode = mode
        if mode == MotorControlMode.POSITION or self._control_mode == MotorControlMode.POSITION:
            self._target_position = target
        elif mode == MotorControlMode.VELOCITY or self._control_mode == MotorControlMode.VELOCITY:
            self._target_velocity = target

    def step(self, dt: float) -> MotorState:
        """执行一个控制周期"""
        if not self._enabled:
            return self.get_state()

        torque_output = self._pid.compute(self._target_position, self._position, dt)
        self._apply_torque(np.clip(torque_output, -self.max_torque, self.max_torque))

        # 积分更新位置
        self._position += self._velocity * dt
        self._velocity += self._torque * dt  # 简化
        self._velocity = np.clip(self._velocity, -self.max_velocity, self.max_velocity)

        return self.get_state()

    def _apply_torque(self, torque: float):
        """施加力矩"""
        self._torque = torque

    def get_state(self) -> MotorState:
        """获取当前状态"""
        return MotorState(
            timestamp=np.datetime64('now').astype(float) / 1e9,
            motor_id=self.motor_id,
            position=self._position,
            velocity=self._velocity,
            torque=self._torque,
            target_position=self._target_position,
            target_velocity=self._target_velocity,
            error=self._error
        )

    def get_position_rad(self) -> float:
        return self._position

    def get_velocity_rpm(self) -> float:
        return self._velocity * 60 / (2 * np.pi)


class DCMotor(Motor):
    """直流电机"""

    def __init__(self, motor_id: str, config: Optional[Dict] = None):
        super().__init__(motor_id, MotorType.DC, config)
        self.motor_constant = self.config.get("motor_constant", 0.01)  # Nm/A
        self.armature_resistance = self.config.get("r", 1.0)  # Ohm
        self._voltage = 0.0

    def _apply_torque(self, torque: float):
        """计算并施加电压"""
        current = torque / self.motor_constant
        voltage = current * self.armature_resistance
        self._voltage = np.clip(voltage, -24, 24)
        self._torque = torque

    def get_power_consumption(self) -> float:
        """计算功耗 (W)"""
        return abs(self._voltage * self._torque / self.motor_constant)


class BLDCmotor(Motor):
    """无刷直流电机"""

    def __init__(self, motor_id: str, config: Optional[Dict] = None):
        super().__init__(motor_id, MotorType.BLDC, config)
        self.pole_pairs = self.config.get("pole_pairs", 4)
        self._commutation_angle = 0.0

    def step(self, dt: float) -> MotorState:
        """BLDC控制周期"""
        if not self._enabled:
            return self.get_state()

        # FOC控制
        torque_output = self._pid.compute(self._target_position, self._position, dt)
        self._torque = np.clip(torque_output, -self.max_torque, self.max_torque)

        # 更新位置
        self._position += (self._velocity / self.pole_pairs) * dt
        # 更新速度
        self._velocity += self._torque * dt

        return self.get_state()


class ServoMotor(Motor):
    """伺服舵机"""

    def __init__(self, motor_id: str, config: Optional[Dict] = None):
        super().__init__(motor_id, MotorType.SERVO, config)
        self.min_angle = self.config.get("min_angle", -3.14)
        self.max_angle = self.config.get("max_angle", 3.14)
        self.max_speed = self.config.get("max_speed", 10.0)  # rad/s

    def set_target(self, target: float, mode: Optional[MotorControlMode] = None):
        """设置目标角度"""
        target = np.clip(target, self.min_angle, self.max_angle)
        super().set_target(target, mode)

    def step(self, dt: float) -> MotorState:
        """舵机控制周期"""
        if not self._enabled:
            return self.get_state()

        # 限幅目标速度
        position_error = self._target_position - self._position
        max_step = self.max_speed * dt
        clamped_target = self._position + np.clip(position_error, -max_step, max_step)

        torque_output = self._pid.compute(clamped_target, self._position, dt)
        self._apply_torque(np.clip(torque_output, -self.max_torque, self.max_torque))

        # 更新
        self._velocity = (clamped_target - self._position) / dt if dt > 0 else 0
        self._position = clamped_target

        return self.get_state()


class StepperMotor(Motor):
    """步进电机"""

    def __init__(self, motor_id: str, config: Optional[Dict] = None):
        super().__init__(motor_id, MotorType.STEPPER, config)
        self.steps_per_rev = self.config.get("steps_per_rev", 200)
        self._step_angle = 2 * np.pi / self.steps_per_rev
        self._current_step = 0
        self._microstep = self.config.get("microstep", 16)

    def step(self, dt: float) -> MotorState:
        """步进电机控制"""
        if not self._enabled:
            return self.get_state()

        # 计算所需步数
        steps_needed = (self._target_position - self._position) / self._step_angle
        step_cmd = int(round(steps_needed))

        if step_cmd != 0:
            self._current_step += step_cmd
            self._position = self._current_step * self._step_angle

        self._velocity = step_cmd / dt if dt > 0 else 0

        return self.get_state()


class MotorController:
    """多电机控制器"""

    def __init__(self):
        self.motors: Dict[str, Motor] = {}
        self._control_period = 0.001  # 1ms

    def add_motor(self, motor: Motor):
        """添加电机"""
        self.motors[motor.motor_id] = motor

    def get_motor(self, motor_id: str) -> Optional[Motor]:
        return self.motors.get(motor_id)

    def set_all_targets(self, targets: Dict[str, float], mode: Optional[MotorControlMode] = None):
        """设置所有电机目标"""
        for motor_id, target in targets.items():
            if motor_id in self.motors:
                self.motors[motor_id].set_target(target, mode)

    def step_all(self, dt: Optional[float] = None) -> Dict[str, MotorState]:
        """执行所有电机控制周期"""
        dt = dt or self._control_period
        return {mid: motor.step(dt) for mid, motor in self.motors.items()}

    def get_all_states(self) -> Dict[str, MotorState]:
        """获取所有电机状态"""
        return {mid: motor.get_state() for mid, motor in self.motors.items()}

    def enable_all(self):
        """使能所有电机"""
        for motor in self.motors.values():
            motor.enable()

    def disable_all(self):
        """禁用所有电机"""
        for motor in self.motors.values():
            motor.disable()

    def get_positions(self) -> Dict[str, float]:
        """获取所有电机位置"""
        return {mid: m.get_position_rad() for mid, m in self.motors.items()}

    def get_velocities(self) -> Dict[str, float]:
        """获取所有电机速度 (rpm)"""
        return {mid: m.get_velocity_rpm() for mid, m in self.motors.items()}

    def emergency_stop(self):
        """紧急停止"""
        self.disable_all()
        for motor in self.motors.values():
            motor._pid.reset()
