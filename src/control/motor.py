"""
电机控制模块 (Motor Control)
支持DC电机、BLDC、伺服电机、步进电机
包含PID控制器
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum


class MotorControlMode(Enum):
    """电机控制模式"""
    POSITION = "position"      # 位置控制
    VELOCITY = "velocity"      # 速度控制
    TORQUE = "torque"          # 力矩控制
    PWM = "pwm"                # PWM开环控制


@dataclass
class MotorState:
    """电机状态"""
    motor_id: str
    timestamp: float

    # 位置 (rad)
    position: float = 0.0
    # 速度 (rad/s)
    velocity: float = 0.0
    # 电流 (A)
    current: float = 0.0
    # 温度 (°C)
    temperature: float = 25.0
    # PWM占空比 (0-1)
    pwm_duty: float = 0.0
    # 使能状态
    enabled: bool = False
    # 错误状态
    error: str = ""
    # 目标位置/速度
    target_position: float = 0.0
    target_velocity: float = 0.0

    def to_vector(self) -> np.ndarray:
        """
        返回特征向量
        格式: [position_norm, velocity_norm, current_norm, temp_norm, enabled_flag]
        """
        return np.array([
            self.position / (2 * np.pi),  # 归一化到圈数
            self.velocity / 100.0,  # 假设最大100 rad/s
            self.current / 10.0,    # 假设最大10A
            self.temperature / 100.0,
            1.0 if self.enabled else 0.0
        ])

    def is_valid(self) -> bool:
        """检查状态是否有效"""
        if self.error:
            return False
        if self.temperature > 80:  # 过温保护
            return False
        return True


class Motor(ABC):
    """电机基类"""

    def __init__(
        self,
        motor_id: str,
        name: str = "Motor",
        reduction_ratio: float = 1.0,  # 减速比
        max_velocity: float = 100.0,  # rad/s (输出轴)
        max_torque: float = 10.0,  # Nm
        max_current: float = 10.0  # A
    ):
        self.motor_id = motor_id
        self.name = name
        self.reduction_ratio = reduction_ratio
        self.max_velocity = max_velocity
        self.max_torque = max_torque
        self.max_current = max_current

        # 当前状态
        self._state = MotorState(motor_id=motor_id, timestamp=0.0)
        self._enabled = False

        # 控制模式
        self._control_mode = MotorControlMode.VELOCITY
        self._target = 0.0

        # 电机参数 (电机轴侧)
        self._motor_Kv = 1000.0  # RPM/V (无刷电机)
        self._motor_R = 10.0     # Ohm
        self._motor_L = 0.01     # H

    @abstractmethod
    def enable(self):
        """使能电机"""
        pass

    @abstractmethod
    def disable(self):
        """禁用电机"""
        pass

    @abstractmethod
    def set_target(self, target: float, mode: MotorControlMode):
        """设置目标值"""
        pass

    @abstractmethod
    def step(self, dt: float) -> MotorState:
        """步进控制 (dt: 时间步长秒)"""
        pass

    def get_state(self) -> MotorState:
        """获取当前状态"""
        return self._state

    def get_position_rad(self) -> float:
        """获取当前位置 (rad, 输出轴)"""
        return self._state.position

    def get_velocity_rpm(self) -> float:
        """获取当前速度 (RPM, 输出轴)"""
        return self._state.velocity * 60.0 / (2 * np.pi)

    def get_position_revs(self) -> float:
        """获取位置 (圈数, 输出轴)"""
        return self._state.position / (2 * np.pi)

    def set_control_mode(self, mode: MotorControlMode):
        """设置控制模式"""
        self._control_mode = mode

    def is_enabled(self) -> bool:
        """检查是否使能"""
        return self._enabled


class DCMotor(Motor):
    """
    直流电机 (DC Motor)
    适用于: 简单AGV、输送线、仓储机器人
    """

    def __init__(
        self,
        motor_id: str,
        name: str = "DCMotor",
        voltage: float = 24.0,  # V
        reduction_ratio: float = 30.0,
        max_velocity: float = 100.0,
        max_torque: float = 10.0,
        armature_resistance: float = 5.0,  # Ohm
        torque_constant: float = 0.1  # Nm/A
    ):
        super().__init__(motor_id, name, reduction_ratio, max_velocity, max_torque)

        self.voltage = voltage
        self.armature_resistance = armature_resistance
        self.torque_constant = torque_constant

        # 内部状态
        self._armature_voltage = 0.0
        self._back_emf = 0.0

    def enable(self):
        """使能电机"""
        self._enabled = True
        self._state.enabled = True

    def disable(self):
        """禁用电机"""
        self._enabled = False
        self._state.enabled = False
        self._armature_voltage = 0.0
        self._state.pwm_duty = 0.0

    def set_target(self, target: float, mode: MotorControlMode):
        """设置目标值"""
        self._target = target
        self._control_mode = mode

    def step(self, dt: float) -> MotorState:
        """步进控制"""
        if not self._enabled:
            return self._state

        # 电机轴速度
        motor_velocity = self._state.velocity * self.reduction_ratio

        # 根据控制模式计算电压
        if self._control_mode == MotorControlMode.PWM:
            self._armature_voltage = self._target * self.voltage
        elif self._control_mode == MotorControlMode.VELOCITY:
            # 速度环: PID输出电压
            self._armature_voltage = self._compute_voltage_for_velocity(self._target, motor_velocity, dt)
        elif self._control_mode == MotorControlMode.POSITION:
            # 位置环: PID输出电压
            self._armature_voltage = self._compute_voltage_for_position(self._target, motor_velocity, dt)
        elif self._control_mode == MotorControlMode.TORQUE:
            # 力矩环: Kt * I = torque
            target_current = self._target / self.torque_constant
            self._armature_voltage = target_current * self.armature_resistance

        # PWM限幅
        self._armature_voltage = np.clip(self._armature_voltage, -self.voltage, self.voltage)
        self._state.pwm_duty = abs(self._armature_voltage) / self.voltage

        # 反电动势
        self._back_emf = motor_velocity / (self._motor_Kv * 60 / (2 * np.pi))

        # 计算电流
        applied_voltage = self._armature_voltage - self._back_emf
        self._state.current = applied_voltage / self.armature_resistance

        # 计算力矩
        torque = self.torque_constant * self._state.current

        # 更新位置和速度 (输出轴)
        motor_accel = torque / (self.max_torque + 1e-6) * 1000  # 简化加速度
        motor_velocity += motor_accel * dt
        motor_velocity = np.clip(motor_velocity, -self.max_velocity * self.reduction_ratio,
                                  self.max_velocity * self.reduction_ratio)

        # 限电流
        self._state.current = np.clip(self._state.current, -self.max_current, self.max_current)

        # 更新状态
        self._state.velocity = motor_velocity / self.reduction_ratio
        self._state.position += self._state.velocity * dt
        self._state.target_velocity = self._target if self._control_mode == MotorControlMode.VELOCITY else 0
        self._state.target_position = self._target if self._control_mode == MotorControlMode.POSITION else 0

        # 温度模拟
        self._state.temperature += 0.01 * abs(self._state.current)
        self._state.temperature = np.clip(self._state.temperature, 25, 80)

        self._state.timestamp += dt
        return self._state

    def _compute_voltage_for_velocity(self, target_vel: float, current_vel: float, dt: float) -> float:
        """计算速度环电压"""
        error = target_vel - current_vel / self.reduction_ratio
        # 简化的PI控制
        return error * 0.5

    def _compute_voltage_for_position(self, target_pos: float, current_vel: float, dt: float) -> float:
        """计算位置环电压"""
        error = target_pos - self._state.position
        # 简化的PID
        return error * 2.0


class BLDCmotor(Motor):
    """
    无刷直流电机 (BLDC Motor)
    适用于: 高性能AGV、服务机器人
    """

    def __init__(
        self,
        motor_id: str,
        name: str = "BLDCmotor",
        poles: int = 4,  # 极对数
        kv: int = 1000,  # RPM/V
        reduction_ratio: float = 20.0,
        max_velocity: float = 150.0,
        max_torque: float = 5.0,
        phase_resistance: float = 0.1,  # Ohm
        phase_inductance: float = 0.001  # H
    ):
        super().__init__(motor_id, name, reduction_ratio, max_velocity, max_torque)

        self.poles = poles
        self.kv = kv  # RPM/V
        self.phase_resistance = phase_resistance
        self.phase_inductance = phase_inductance

        # 内部状态
        self._phase_voltage = np.zeros(3)
        self._phase_current = np.zeros(3)
        self._electrical_angle = 0.0

    def enable(self):
        """使能电机"""
        self._enabled = True
        self._state.enabled = True

    def disable(self):
        """禁用电机"""
        self._enabled = False
        self._state.enabled = False

    def set_target(self, target: float, mode: MotorControlMode):
        """设置目标值"""
        self._target = target
        self._control_mode = mode

    def step(self, dt: float) -> MotorState:
        """步进控制"""
        if not self._enabled:
            return self._state

        # 电机轴角速度
        motor_velocity = self._state.velocity * self.reduction_ratio

        # FOC控制 (简化)
        if self._control_mode == MotorControlMode.VELOCITY:
            # 速度环
            voltage = self._compute_foc_voltage(self._target, motor_velocity, dt)
        elif self._control_mode == MotorControlMode.POSITION:
            # 位置环
            voltage = self._compute_position_voltage(self._target, dt)
        else:
            voltage = 0.0

        voltage = np.clip(voltage, -48, 48)  # 48V max
        self._state.pwm_duty = abs(voltage) / 48.0

        # 计算电流 (简化)
        self._state.current = voltage / self.phase_resistance

        # 计算力矩
        torque = self._state.current * 0.1

        # 更新状态
        motor_accel = torque / (self.max_torque + 1e-6) * 1000
        motor_velocity += motor_accel * dt
        motor_velocity = np.clip(motor_velocity, -self.max_velocity * self.reduction_ratio,
                                 self.max_velocity * self.reduction_ratio)

        self._state.velocity = motor_velocity / self.reduction_ratio
        self._state.position += self._state.velocity * dt

        # 温度
        self._state.temperature += 0.005 * abs(self._state.current)
        self._state.temperature = np.clip(self._state.temperature, 25, 85)

        # 更新电角度
        self._electrical_angle = (self.poles / 2) * self._state.position
        self._electrical_angle = self._electrical_angle % (2 * np.pi)

        self._state.timestamp += dt
        return self._state

    def _compute_foc_voltage(self, target_vel: float, current_vel: float, dt: float) -> float:
        """计算FOC电压"""
        error = target_vel - current_vel / self.reduction_ratio
        return error * 1.0

    def _compute_position_voltage(self, target_pos: float, dt: float) -> float:
        """计算位置环电压"""
        error = target_pos - self._state.position
        return error * 3.0


class ServoMotor(Motor):
    """
    伺服舵机 (Servo Motor)
    适用于: 机械臂关节、精密定位
    """

    def __init__(
        self,
        motor_id: str,
        name: str = "ServoMotor",
        angle_range: float = 360.0,  # 度
        reduction_ratio: float = 100.0,
        max_velocity: float = 50.0,
        max_torque: float = 2.0,
        position_resolution: float = 0.01  # 度
    ):
        super().__init__(motor_id, name, reduction_ratio, max_velocity, max_torque)

        self.angle_range = angle_range  # 度
        self.position_resolution = position_resolution

        # 位置PID参数
        self._pos_kp = 10.0
        self._pos_ki = 0.1
        self._pos_kd = 0.5
        self._pos_integral = 0.0
        self._pos_last_error = 0.0

    def enable(self):
        """使能舵机"""
        self._enabled = True
        self._state.enabled = True

    def disable(self):
        """禁用舵机"""
        self._enabled = False
        self._state.enabled = False
        self._state.pwm_duty = 0.0

    def set_target(self, target: float, mode: MotorControlMode):
        """设置目标角度 (度)"""
        if mode == MotorControlMode.POSITION:
            self._target = np.clip(target, 0, self.angle_range)
        else:
            self._target = target
        self._control_mode = mode

    def step(self, dt: float) -> MotorState:
        """步进控制"""
        if not self._enabled:
            return self._state

        if self._control_mode == MotorControlMode.POSITION:
            # 位置PID控制
            error = self._target - self._state.position * 180.0 / np.pi

            # 积分
            self._pos_integral += error * dt
            self._pos_integral = np.clip(self._pos_integral, -100, 100)

            # 微分
            derivative = (error - self._pos_last_error) / dt if dt > 0 else 0
            self._pos_last_error = error

            # PID输出
            pwm = self._pos_kp * error + self._pos_ki * self._pos_integral + self._pos_kd * derivative
            self._state.pwm_duty = np.clip(abs(pwm), 0, 1)

        elif self._control_mode == MotorControlMode.VELOCITY:
            self._state.pwm_duty = np.clip(abs(self._target) / self.max_velocity, 0, 1)

        # 更新状态 (简化)
        if self._control_mode == MotorControlMode.POSITION:
            direction = 1 if self._target > self._state.position * 180.0 / np.pi else -1
            self._state.velocity = direction * self.max_velocity * self._state.pwm_duty
        else:
            self._state.velocity = self._target

        self._state.velocity = np.clip(self._state.velocity, -self.max_velocity, self.max_velocity)
        self._state.position += self._state.velocity * dt

        # 温度
        self._state.temperature += 0.002 * self._state.pwm_duty
        self._state.temperature = np.clip(self._state.temperature, 25, 75)

        self._state.timestamp += dt
        return self._state


class StepperMotor(Motor):
    """
    步进电机 (Stepper Motor)
    适用于: 3D打印机、 CNC、精密传动
    """

    def __init__(
        self,
        motor_id: str,
        name: str = "StepperMotor",
        steps_per_rev: int = 200,  # 每转步数
        microsteps: int = 16,  # 细分
        reduction_ratio: float = 10.0,
        max_velocity: float = 20.0,
        holding_torque: float = 1.0  # Nm
    ):
        super().__init__(motor_id, name, reduction_ratio, max_velocity, holding_torque)

        self.steps_per_rev = steps_per_rev
        self.microsteps = microsteps
        self._steps_per_output_rev = steps_per_rev * microsteps * reduction_ratio

        # 内部状态
        self._step_position = 0  # 步数
        self._target_step = 0

    def enable(self):
        """使能电机"""
        self._enabled = True
        self._state.enabled = True

    def disable(self):
        """禁用电机"""
        self._enabled = False
        self._state.enabled = False

    def set_target(self, target: float, mode: MotorControlMode):
        """设置目标位置 (输出轴角度, 度)"""
        if mode == MotorControlMode.POSITION:
            # 转换为步数
            target_deg = target * 180.0 / np.pi  # 转换为度
            self._target_step = int(target_deg / 360.0 * self._steps_per_output_rev)
        self._control_mode = mode

    def step(self, dt: float) -> MotorState:
        """步进控制"""
        if not self._enabled:
            return self._state

        # 计算目标速度
        error = self._target_step - self._step_position
        steps_to_move = np.clip(error, -1000, 1000)

        # 更新步位置
        self._step_position += steps_to_move

        # 更新输出轴位置
        self._state.position = self._step_position / self._steps_per_output_rev * 2 * np.pi

        # 更新速度
        self._state.velocity = steps_to_move / (self._steps_per_output_rev * dt) * 2 * np.pi
        self._state.velocity = np.clip(self._state.velocity, -self.max_velocity, self.max_velocity)

        # PWM占空比表示负载
        self._state.pwm_duty = min(abs(error) / 100, 1.0)

        # 温度
        self._state.temperature = 25.0 + 0.001 * self._state.pwm_duty * 50

        self._state.timestamp += dt
        return self._state


class PIDController:
    """
    PID控制器
    通用PID实现，支持位置式和增量式
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        output_limit: float = None,
        integral_limit: float = None,
        derivative_filter: float = 0.0
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.derivative_filter = derivative_filter

        # 内部状态
        self._integral = 0.0
        self._last_error = 0.0
        self._last_output = 0.0
        self._filtered_derivative = 0.0

    def compute(self, error: float, dt: float) -> float:
        """
        计算PID输出

        Args:
            error: 误差 (setpoint - measured)
            dt: 时间步长

        Returns:
            控制输出
        """
        # 比例
        p = self.kp * error

        # 积分
        self._integral += error * dt
        if self.integral_limit is not None:
            self._integral = np.clip(self._integral, -self.integral_limit, self.integral_limit)
        i = self.ki * self._integral

        # 微分 (带滤波)
        if dt > 0:
            raw_derivative = (error - self._last_error) / dt
            alpha = self.derivative_filter
            self._filtered_derivative = alpha * self._filtered_derivative + (1 - alpha) * raw_derivative
        d = self.kd * self._filtered_derivative

        # 总输出
        output = p + i + d

        # 输出限幅
        if self.output_limit is not None:
            output = np.clip(output, -self.output_limit, self.output_limit)

        self._last_error = error
        self._last_output = output

        return output

    def reset(self):
        """重置PID状态"""
        self._integral = 0.0
        self._last_error = 0.0
        self._last_output = 0.0
        self._filtered_derivative = 0.0


class MotorController:
    """
    多电机控制器
    管理多个电机，支持同步控制
    """

    def __init__(self, name: str = "MotorController"):
        self.name = name
        self._motors: Dict[str, Motor] = {}

    def add_motor(self, motor: Motor) -> bool:
        """
        添加电机

        Args:
            motor: Motor 实例

        Returns:
            添加是否成功
        """
        if motor.motor_id in self._motors:
            return False
        self._motors[motor.motor_id] = motor
        return True

    def remove_motor(self, motor_id: str) -> bool:
        """移除电机"""
        if motor_id in self._motors:
            del self._motors[motor_id]
            return True
        return False

    def enable_all(self):
        """使能所有电机"""
        for motor in self._motors.values():
            motor.enable()

    def disable_all(self):
        """禁用所有电机"""
        for motor in self._motors.values():
            motor.disable()

    def set_all_targets(self, targets: Dict[str, float], mode: MotorControlMode):
        """
        设置所有电机目标

        Args:
            targets: 电机ID到目标值的字典
            mode: 控制模式
        """
        for motor_id, target in targets.items():
            if motor_id in self._motors:
                self._motors[motor_id].set_target(target, mode)

    def step_all(self, dt: float) -> Dict[str, MotorState]:
        """
        所有电机步进

        Args:
            dt: 时间步长 (s)

        Returns:
            电机ID到状态的字典
        """
        states = {}
        for motor_id, motor in self._motors.items():
            states[motor_id] = motor.step(dt)
        return states

    def get_all_states(self) -> Dict[str, MotorState]:
        """获取所有电机状态"""
        return {motor_id: motor.get_state() for motor_id, motor in self._motors.items()}

    def get_motor(self, motor_id: str) -> Optional[Motor]:
        """获取指定电机"""
        return self._motors.get(motor_id)

    def emergency_stop(self):
        """紧急停止"""
        for motor in self._motors.values():
            motor.disable()
            motor.set_target(0.0, MotorControlMode.PWM)

    def __len__(self) -> int:
        return len(self._motors)

    def __repr__(self) -> str:
        return f"MotorController(motors={list(self._motors.keys())})"
