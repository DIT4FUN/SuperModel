"""
安全监控模块 (Safety Monitoring)
提供AGV安全监控、紧急停止、碰撞检测和状态监控功能
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time


class SafetyLevel(Enum):
    """安全等级"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY_STOP = "emergency_stop"


class StopReason(Enum):
    """停止原因"""
    NONE = "none"
    COLLISION_DETECTED = "collision_detected"
    FORCE_THRESHOLD_EXCEEDED = "force_threshold_exceeded"
    VELOCITY_LIMIT_EXCEEDED = "velocity_limit_exceeded"
    BOUNDARY_VIOLATION = "boundaryViolation"
    EMERGENCY_BUTTON = "emergency_button"
    SENSOR_FAILURE = "sensor_failure"
    TIMEOUT = "timeout"
    MANUAL_STOP = "manual_stop"


@dataclass
class SafetyStatus:
    """安全状态"""
    level: SafetyLevel = SafetyLevel.NORMAL
    stop_reason: StopReason = StopReason.NONE
    message: str = ""
    timestamp: float = 0.0
    velocity: float = 0.0
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    force_magnitude: float = 0.0
    is_estop_triggered: bool = False
    sensors_healthy: Dict[str, bool] = field(default_factory=dict)


class SafetyMonitor:
    """
    安全监控器

    监控:
    - 速度限制
    - 力/力矩限制
    - 位置边界
    - 碰撞检测
    - 传感器健康状态
    - 紧急停止触发

    Args:
        max_velocity: 最大允许速度 (m/s)
        max_acceleration: 最大允许加速度 (m/s²)
        boundary_min: 最小边界 [x_min, y_min, theta_min]
        boundary_max: 最大边界 [x_max, y_max, theta_max]
        force_threshold: 力阈值 (N)
        torque_threshold: 力矩阈值 (Nm)
    """

    def __init__(
        self,
        max_velocity: float = 2.0,
        max_acceleration: float = 1.0,
        boundary_min: Optional[np.ndarray] = None,
        boundary_max: Optional[np.ndarray] = None,
        force_threshold: float = 100.0,
        torque_threshold: float = 2.0,
        reaction_time: float = 0.05  # s
    ):
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.force_threshold = force_threshold
        self.torque_threshold = torque_threshold
        self.reaction_time = reaction_time

        # 边界 (默认无限制)
        self.boundary_min = boundary_min if boundary_min is not None else np.array([-100, -100, -np.pi])
        self.boundary_max = boundary_max if boundary_max is not None else np.array([100, 100, np.pi])

        # 紧急停止回调
        self._estop_callbacks: List[Callable[[], None]] = []

        # 状态
        self._status = SafetyStatus()
        self._status.timestamp = time.time()
        self._last_check_time = time.time()
        self._velocity_history: List[float] = []
        self._estop_triggered = False

    def add_estop_callback(self, callback: Callable[[], None]):
        """添加紧急停止回调"""
        self._estop_callbacks.append(callback)

    def check_velocity(
        self,
        velocity: float,
        dt: float,
        timestamp: Optional[float] = None
    ) -> SafetyStatus:
        """
        检查速度是否超限

        Args:
            velocity: 当前速度 (m/s)
            dt: 时间步长 (s)
            timestamp: 时间戳

        Returns:
            SafetyStatus
        """
        if timestamp is None:
            timestamp = time.time()

        self._status.timestamp = timestamp
        self._status.velocity = velocity

        # 速度检查
        if abs(velocity) > self.max_velocity:
            self._status.level = SafetyLevel.CRITICAL
            self._status.stop_reason = StopReason.VELOCITY_LIMIT_EXCEEDED
            self._status.message = f"速度 {abs(velocity):.2f}m/s 超过限制 {self.max_velocity}m/s"
            return self._status

        # 加速度检查
        self._velocity_history.append(velocity)
        if len(self._velocity_history) > 2:
            self._velocity_history.pop(0)

        if len(self._velocity_history) == 2:
            accel = abs(velocity - self._velocity_history[0]) / dt
            if accel > self.max_acceleration:
                self._status.level = SafetyLevel.WARNING
                self._status.message = f"加速度 {accel:.2f}m/s² 超过限制 {self.max_acceleration}m/s²"

        return self._status

    def check_boundary(
        self,
        position: Tuple[float, float, float],
        timestamp: Optional[float] = None
    ) -> SafetyStatus:
        """
        检查位置边界

        Args:
            position: 位置 (x, y, theta)
            timestamp: 时间戳

        Returns:
            SafetyStatus
        """
        if timestamp is None:
            timestamp = time.time()

        self._status.timestamp = timestamp
        self._status.position = position

        pos = np.array(position)

        # 检查是否超出边界
        if np.any(pos < self.boundary_min) or np.any(pos > self.boundary_max):
            self._status.level = SafetyLevel.CRITICAL
            self._status.stop_reason = StopReason.BOUNDARY_VIOLATION
            self._status.message = f"位置 {position} 超出边界"
            return self._status

        return self._status

    def check_force(
        self,
        force_magnitude: float,
        torque_magnitude: float = 0.0,
        timestamp: Optional[float] = None
    ) -> SafetyStatus:
        """
        检查力和力矩

        Args:
            force_magnitude: 合力大小 (N)
            torque_magnitude: 合力矩大小 (Nm)
            timestamp: 时间戳

        Returns:
            SafetyStatus
        """
        if timestamp is None:
            timestamp = time.time()

        self._status.timestamp = timestamp
        self._status.force_magnitude = force_magnitude

        if force_magnitude > self.force_threshold:
            self._status.level = SafetyLevel.CRITICAL
            self._status.stop_reason = StopReason.FORCE_THRESHOLD_EXCEEDED
            self._status.message = f"力 {force_magnitude:.2f}N 超过阈值 {self.force_threshold}N"
            return self._status

        if torque_magnitude > self.torque_threshold:
            self._status.level = SafetyLevel.CRITICAL
            self._status.stop_reason = StopReason.FORCE_THRESHOLD_EXCEEDED
            self._status.message = f"力矩 {torque_magnitude:.3f}Nm 超过阈值 {self.torque_threshold}Nm"
            return self._status

        return self._status

    def check_collision(
        self,
        collision_detected: bool,
        timestamp: Optional[float] = None
    ) -> SafetyStatus:
        """
        检查碰撞

        Args:
            collision_detected: 是否检测到碰撞
            timestamp: 时间戳

        Returns:
            SafetyStatus
        """
        if timestamp is None:
            timestamp = time.time()

        self._status.timestamp = timestamp

        if collision_detected:
            self._status.level = SafetyLevel.CRITICAL
            self._status.stop_reason = StopReason.COLLISION_DETECTED
            self._status.message = "检测到碰撞"
            return self._status

        return self._status

    def check_sensors(
        self,
        sensor_health: Dict[str, bool],
        timestamp: Optional[float] = None
    ) -> SafetyStatus:
        """
        检查传感器健康状态

        Args:
            sensor_health: 传感器健康状态 {sensor_id: is_healthy}
            timestamp: 时间戳

        Returns:
            SafetyStatus
        """
        if timestamp is None:
            timestamp = time.time()

        self._status.timestamp = timestamp
        self._status.sensors_healthy = sensor_health

        failed_sensors = [sid for sid, healthy in sensor_health.items() if not healthy]
        if failed_sensors:
            self._status.level = SafetyLevel.WARNING
            self._status.message = f"传感器故障: {', '.join(failed_sensors)}"

        return self._status

    def check_all(
        self,
        velocity: float,
        position: Tuple[float, float, float],
        force_magnitude: float,
        torque_magnitude: float,
        collision_detected: bool,
        sensor_health: Dict[str, bool],
        dt: float,
        timestamp: Optional[float] = None
    ) -> SafetyStatus:
        """
        综合安全检查

        Args:
            velocity: 当前速度 (m/s)
            position: 当前位置 (x, y, theta)
            force_magnitude: 合力大小 (N)
            torque_magnitude: 合力矩大小 (Nm)
            collision_detected: 是否检测到碰撞
            sensor_health: 传感器健康状态
            dt: 时间步长 (s)
            timestamp: 时间戳

        Returns:
            SafetyStatus
        """
        if timestamp is None:
            timestamp = time.time()

        # 如果已经触发紧急停止，直接返回
        if self._estop_triggered:
            self._status.is_estop_triggered = True
            return self._status

        # 逐项检查，每项检查会更新status
        # 按优先级检查：碰撞 > 力 > 速度 > 边界 > 传感器
        self._status = SafetyStatus()
        self._status.timestamp = timestamp

        # 碰撞检查 (最高优先级)
        self.check_collision(collision_detected, timestamp)
        if self._status.level == SafetyLevel.CRITICAL:
            return self._trigger_estop()

        # 力检查
        self.check_force(force_magnitude, torque_magnitude, timestamp)
        if self._status.level == SafetyLevel.CRITICAL:
            return self._trigger_estop()

        # 速度检查
        self.check_velocity(velocity, dt, timestamp)
        if self._status.level == SafetyLevel.CRITICAL:
            return self._trigger_estop()

        # 边界检查
        self.check_boundary(position, timestamp)
        if self._status.level == SafetyLevel.CRITICAL:
            return self._trigger_estop()

        # 传感器检查
        self.check_sensors(sensor_health, timestamp)

        return self._status

    def _trigger_estop(self) -> SafetyStatus:
        """触发紧急停止"""
        self._estop_triggered = True
        self._status.is_estop_triggered = True
        self._status.level = SafetyLevel.EMERGENCY_STOP

        # 调用所有紧急停止回调
        for callback in self._estop_callbacks:
            try:
                callback()
            except Exception:
                pass

        return self._status

    def emergency_stop(self, reason: str = "manual"):
        """
        触发紧急停止

        Args:
            reason: 停止原因
        """
        self._status.level = SafetyLevel.EMERGENCY_STOP
        self._status.stop_reason = StopReason.EMERGENCY_BUTTON
        self._status.message = f"紧急停止: {reason}"
        self._status.timestamp = time.time()
        self._status.is_estop_triggered = True
        self._estop_triggered = True

        for callback in self._estop_callbacks:
            try:
                callback()
            except Exception:
                pass

    def reset_estop(self):
        """重置紧急停止状态"""
        self._estop_triggered = False
        self._status = SafetyStatus()
        self._status.timestamp = time.time()
        self._velocity_history.clear()

    def get_status(self) -> SafetyStatus:
        """获取当前安全状态"""
        return self._status

    def is_safe(self) -> bool:
        """检查是否处于安全状态"""
        return (
            self._status.level == SafetyLevel.NORMAL or
            self._status.level == SafetyLevel.WARNING
        ) and not self._estop_triggered

    def set_limits(
        self,
        max_velocity: Optional[float] = None,
        max_acceleration: Optional[float] = None,
        force_threshold: Optional[float] = None,
        torque_threshold: Optional[float] = None
    ):
        """在线修改安全限制"""
        if max_velocity is not None:
            self.max_velocity = max_velocity
        if max_acceleration is not None:
            self.max_acceleration = max_acceleration
        if force_threshold is not None:
            self.force_threshold = force_threshold
        if torque_threshold is not None:
            self.torque_threshold = torque_threshold


class EmergencyStopController:
    """
    紧急停止控制器

    支持:
    - 硬件紧急停止 (GPIO/继电器)
    - 软件紧急停止
    - 紧急停止记录
    - 恢复锁定
    """

    def __init__(self, name: str = "EStop"):
        self.name = name
        self._is_estopped = False
        self._estop_time: Optional[float] = None
        self._lockout_duration = 5.0  # s, 锁定时间
        self._estop_reasons: List[Tuple[str, float]] = []  # [(reason, timestamp), ...]
        self._hardware_estop_pin: Optional[int] = None

    def trigger(self, reason: str = "unknown") -> bool:
        """
        触发紧急停止

        Args:
            reason: 触发原因

        Returns:
            是否成功触发
        """
        if self._is_estopped:
            return False

        self._is_estopped = True
        self._estop_time = time.time()
        self._estop_reasons.append((reason, self._estop_time))
        return True

    def reset(self, require_lockout: bool = True) -> Tuple[bool, str]:
        """
        重置紧急停止

        Args:
            require_lockout: 是否需要等待锁定时间

        Returns:
            (是否成功, 消息)
        """
        if not self._is_estopped:
            return True, "未处于紧急停止状态"

        if require_lockout and self._estop_time is not None:
            elapsed = time.time() - self._estop_time
            if elapsed < self._lockout_duration:
                remaining = self._lockout_duration - elapsed
                return False, f"锁定中，请等待 {remaining:.1f}s"

        self._is_estopped = False
        self._estop_time = None
        return True, "已重置紧急停止"

    def is_active(self) -> bool:
        """检查是否处于紧急停止状态"""
        return self._is_estopped

    def get_last_reason(self) -> Tuple[Optional[str], Optional[float]]:
        """获取最近一次紧急停止原因"""
        if self._estop_reasons:
            return self._estop_reasons[-1]
        return None, None

    def get_history(self) -> List[Tuple[str, float]]:
        """获取紧急停止历史"""
        return self._estop_reasons.copy()

    def set_lockout_duration(self, duration: float):
        """设置锁定时间"""
        self._lockout_duration = max(0, duration)
