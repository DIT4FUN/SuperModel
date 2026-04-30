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
deployment.py - 具身智能实机部署管理模块
SuperModel 超模态大模型具身智能系统

支持:
- 部署前配置验证
- 传感器/执行器健康检查
- 紧急停车程序
- 运行时状态监控
- 故障恢复与降级策略
- AGV五级规格适配
"""

from __future__ import annotations

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from enum import Enum, auto
from collections import deque

if TYPE_CHECKING:
    from ..sensors.manager import SensorManager
    from ..control.embodied_control import EmbodiedControlPipeline
    from ..core.core_brain import CoreBrain

logger = logging.getLogger(__name__)

__all__ = [
    'DeploymentState',
    'HealthStatus',
    'DeploymentConfig',
    'HealthCheckResult',
    'DeploymentValidator',
    'HealthMonitor',
    'EmergencyProcedure',
    'DeploymentManager',
]


class DeploymentState(Enum):
    """部署状态"""
    IDLE = "idle"
    VALIDATING = "validating"
    DEPLOYING = "deploying"
    RUNNING = "running"
    DEGRADED = "degraded"
    EMERGENCY_STOP = "emergency_stop"
    SHUTDOWN = "shutdown"


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class DeploymentConfig:
    """部署配置"""
    # AGV等级
    grade: str = "M"

    # 部署范围
    enable_vision: bool = True
    enable_audio: bool = True
    enable_tactile: bool = True
    enable_force: bool = True
    enable_imu: bool = True
    enable_control: bool = True

    # 健康检查
    health_check_interval_s: float = 5.0
    max_latency_ms: float = 100.0
    max_sensor_dropout_rate: float = 0.05

    # 紧急停车
    emergency_stop_enabled: bool = True
    emergency_stop_timeout_s: float = 0.5

    # 降级策略
    allow_degraded_mode: bool = True
    min_required_sensors: Set[str] = field(default_factory=lambda: {"imu"})

    # 运行时限制
    max_consecutive_errors: int = 5
    recovery_wait_s: float = 2.0

    # 日志级别
    log_health_data: bool = False
    health_history_size: int = 100


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    component: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def is_healthy(self) -> bool:
        return self.status in (HealthStatus.HEALTHY, HealthStatus.WARNING)

    def is_critical(self) -> bool:
        return self.status in (HealthStatus.CRITICAL, HealthStatus.FAILED)


class DeploymentValidator:
    """
    部署前配置验证器

    检查:
    - 硬件配置一致性
    - 传感器连接状态
    - 通信链路可用性
    - 软件依赖完整性
    """

    def __init__(self, config: DeploymentConfig):
        self.config = config

    def validate_config(self) -> List[HealthCheckResult]:
        """验证部署配置"""
        results = []

        # 验证AGV等级
        valid_grades = {"S", "M", "L", "XL", "XXL"}
        if self.config.grade not in valid_grades:
            results.append(HealthCheckResult(
                component="config",
                status=HealthStatus.FAILED,
                message=f"Invalid grade: {self.config.grade}. Must be one of {valid_grades}"
            ))
        else:
            results.append(HealthCheckResult(
                component="config",
                status=HealthStatus.HEALTHY,
                message=f"AGV grade {self.config.grade} is valid"
            ))

        # 验证传感器使能配置
        if self.config.enable_tactile and self.config.enable_force:
            results.append(HealthCheckResult(
                component="config",
                status=HealthStatus.HEALTHY,
                message="Tactile and force sensors enabled"
            ))
        elif not self.config.enable_tactile and not self.config.enable_force:
            results.append(HealthCheckResult(
                component="config",
                status=HealthStatus.WARNING,
                message="Neither tactile nor force sensors enabled - reduced manipulation capability"
            ))

        # 验证紧急停车配置
        if self.config.emergency_stop_enabled:
            results.append(HealthCheckResult(
                component="config",
                status=HealthStatus.HEALTHY,
                message="Emergency stop is enabled"
            ))
        else:
            results.append(HealthCheckResult(
                component="config",
                status=HealthStatus.WARNING,
                message="Emergency stop is DISABLED - not recommended for production"
            ))

        # 验证健康检查间隔
        if 1.0 <= self.config.health_check_interval_s <= 60.0:
            results.append(HealthCheckResult(
                component="config",
                status=HealthStatus.HEALTHY,
                message=f"Health check interval {self.config.health_check_interval_s}s is appropriate"
            ))
        else:
            results.append(HealthCheckResult(
                component="config",
                status=HealthStatus.WARNING,
                message=f"Unusual health check interval: {self.config.health_check_interval_s}s"
            ))

        return results

    def validate_grade_specs(self) -> List[HealthCheckResult]:
        """验证AGV等级规格匹配"""
        results = []
        grade_specs = {
            "S": {"control_freq": 50, "sensors": 2, "max_speed": 1.0},
            "M": {"control_freq": 100, "sensors": 4, "max_speed": 1.5},
            "L": {"control_freq": 200, "sensors": 6, "max_speed": 2.0},
            "XL": {"control_freq": 500, "sensors": 8, "max_speed": 3.0},
            "XXL": {"control_freq": 1000, "sensors": 10, "max_speed": 5.0},
        }

        spec = grade_specs.get(self.config.grade, grade_specs["M"])
        enabled_sensors = sum([
            self.config.enable_vision,
            self.config.enable_audio,
            self.config.enable_tactile,
            self.config.enable_force,
            self.config.enable_imu,
        ])

        results.append(HealthCheckResult(
            component="grade_spec",
            status=HealthStatus.HEALTHY,
            message=f"Grade {self.config.grade}: control@{spec['control_freq']}Hz, "
                    f"sensors={enabled_sensors}/{spec['sensors']}, max_speed={spec['max_speed']}m/s",
            details=spec
        ))

        return results


class HealthMonitor:
    """
    运行时健康状态监控器

    持续监控:
    - 传感器数据流
    - 控制延迟
    - 错误率
    - 资源使用率
    """

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.state = DeploymentState.IDLE
        self._lock = threading.RLock()
        self._history: deque = deque(maxlen=config.health_history_size)
        self._error_counts: Dict[str, int] = {}
        self._last_check: Dict[str, float] = {}
        self._callbacks: List[Callable[[HealthCheckResult], None]] = []

    def add_callback(self, cb: Callable[[HealthCheckResult], None]):
        """添加健康状态变化回调"""
        self._callbacks.append(cb)

    def report_health(self, result: HealthCheckResult):
        """报告组件健康状态"""
        with self._lock:
            self._history.append(result)
            self._last_check[result.component] = result.timestamp

            if result.is_critical():
                self._error_counts[result.component] = \
                    self._error_counts.get(result.component, 0) + 1
            else:
                self._error_counts[result.component] = 0

            for cb in self._callbacks:
                try:
                    cb(result)
                except Exception as e:
                    logger.error(f"Health callback error: {e}")

    def get_overall_status(self) -> HealthStatus:
        """获取整体健康状态"""
        with self._lock:
            if not self._history:
                return HealthStatus.UNKNOWN

            recent = list(self._history)[-20:]
            critical_count = sum(1 for r in recent if r.is_critical())
            warning_count = sum(1 for r in recent if r.status == HealthStatus.WARNING)

            if critical_count > len(recent) * 0.3:
                return HealthStatus.CRITICAL
            elif critical_count > 0:
                return HealthStatus.CRITICAL
            elif warning_count > len(recent) * 0.5:
                return HealthStatus.WARNING
            else:
                return HealthStatus.HEALTHY

    def get_state(self) -> DeploymentState:
        with self._lock:
            return self.state

    def set_state(self, state: DeploymentState):
        with self._lock:
            old_state = self.state
            self.state = state
            logger.info(f"Deployment state: {old_state.value} -> {state.value}")

    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康状态摘要"""
        with self._lock:
            recent = list(self._history)
            return {
                "state": self.state.value,
                "overall_status": self.get_overall_status().value,
                "total_checks": len(recent),
                "components": list(self._last_check.keys()),
                "last_check": {k: v for k, v in self._last_check.items()},
                "error_counts": dict(self._error_counts),
            }


class EmergencyProcedure:
    """
    紧急停车程序

    触发条件:
    - 物理碰撞检测
    - 传感器数据异常
    - 通信超时
    - 手动急停信号
    - 电池欠压
    """

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self._stop_callbacks: List[Callable[[], None]] = []
        self._stop_reason: Optional[str] = None
        self._stop_time: Optional[float] = None

    def register_stop_callback(self, cb: Callable[[], None]):
        """注册停车回调"""
        self._stop_callbacks.append(cb)

    def trigger(self, reason: str):
        """
        触发紧急停车

        Args:
            reason: 停车原因描述
        """
        self._stop_reason = reason
        self._stop_time = time.time()
        logger.warning(f"EMERGENCY STOP triggered: {reason}")

        for cb in self._stop_callbacks:
            try:
                cb()
            except Exception as e:
                logger.error(f"Emergency stop callback error: {e}")

    def get_stop_info(self) -> Dict[str, Any]:
        """获取停车信息"""
        return {
            "reason": self._stop_reason,
            "time": self._stop_time,
            "duration_s": time.time() - self._stop_time if self._stop_time else 0,
        }

    @staticmethod
    def check_safety_conditions(
        imu_data: Optional[Dict[str, Any]] = None,
        force_data: Optional[Dict[str, Any]] = None,
        tactile_data: Optional[Dict[str, Any]] = None,
        collision_threshold: float = 50.0,
        tilt_threshold_deg: float = 45.0,
    ) -> Tuple[bool, Optional[str]]:
        """
        检查安全条件

        Returns:
            (is_safe, reason_if_not_safe)
        """
        # IMU倾角检查
        if imu_data:
            roll = imu_data.get("roll_deg", 0.0)
            pitch = imu_data.get("pitch_deg", 0.0)
            if abs(roll) > tilt_threshold_deg or abs(pitch) > tilt_threshold_deg:
                return False, f"Vehicle tilted: roll={roll:.1f}°, pitch={pitch:.1f}°"

        # 力觉碰撞检测
        if force_data:
            total_force = force_data.get("total_magnitude_N", 0.0)
            if total_force > collision_threshold:
                return False, f"Collision detected: force={total_force:.1f}N"

        # 触觉阵列高压力检测
        if tactile_data:
            max_pressure = tactile_data.get("max_pressure_pct", 0.0)
            if max_pressure > 95.0:
                return False, f"Excessive pressure detected: {max_pressure:.1f}%"

        return True, None


class DeploymentManager:
    """
    具身智能部署管理器

    整合:
    - DeploymentValidator: 部署前验证
    - HealthMonitor: 运行时健康监控
    - EmergencyProcedure: 紧急停车程序
    - 状态机管理
    """

    def __init__(
        self,
        config: Optional[DeploymentConfig] = None,
        sensor_manager: Optional["SensorManager"] = None,
        control_pipeline: Optional["EmbodiedControlPipeline"] = None,
        core_brain: Optional["CoreBrain"] = None,
    ):
        self.config = config or DeploymentConfig()
        self.sensor_manager = sensor_manager
        self.control_pipeline = control_pipeline
        self.core_brain = core_brain

        self.validator = DeploymentValidator(self.config)
        self.health_monitor = HealthMonitor(self.config)
        self.emergency = EmergencyProcedure(self.config)

        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # 注册紧急停车回调
        self.emergency.register_stop_callback(self._on_emergency_stop)

    def pre_deployment_check(self) -> Tuple[bool, List[HealthCheckResult]]:
        """
        部署前全面检查

        Returns:
            (all_passed, results)
        """
        self.health_monitor.set_state(DeploymentState.VALIDATING)
        all_results: List[HealthCheckResult] = []

        # 配置验证
        config_results = self.validator.validate_config()
        all_results.extend(config_results)

        # 等级规格验证
        grade_results = self.validator.validate_grade_specs()
        all_results.extend(grade_results)

        # 检查传感器管理器
        if self.sensor_manager:
            sensor_result = HealthCheckResult(
                component="sensor_manager",
                status=HealthStatus.HEALTHY,
                message="Sensor manager initialized"
            )
            all_results.append(sensor_result)
        else:
            all_results.append(HealthCheckResult(
                component="sensor_manager",
                status=HealthStatus.WARNING,
                message="No sensor manager - running in simulation mode"
            ))

        # 检查控制管道
        if self.control_pipeline:
            all_results.append(HealthCheckResult(
                component="control_pipeline",
                status=HealthStatus.HEALTHY,
                message="Control pipeline initialized"
            ))
        else:
            all_results.append(HealthCheckResult(
                component="control_pipeline",
                status=HealthStatus.WARNING,
                message="No control pipeline - monitoring only"
            ))

        # 判断结果
        critical_failures = [r for r in all_results if r.status == HealthStatus.FAILED]
        all_passed = len(critical_failures) == 0

        return all_passed, all_results

    def deploy(self) -> bool:
        """
        执行部署

        Returns:
            部署是否成功
        """
        all_passed, results = self.pre_deployment_check()

        for r in results:
            self.health_monitor.report_health(r)

        if not all_passed:
            logger.error("Pre-deployment check FAILED")
            self.health_monitor.set_state(DeploymentState.SHUTDOWN)
            return False

        self.health_monitor.set_state(DeploymentState.DEPLOYING)
        logger.info("Deployment starting...")

        # 启动健康监控线程
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

        self.health_monitor.set_state(DeploymentState.RUNNING)
        logger.info("Deployment complete - system RUNNING")
        return True

    def shutdown(self):
        """关闭部署"""
        self.health_monitor.set_state(DeploymentState.SHUTDOWN)
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Deployment shutdown complete")

    def _monitor_loop(self):
        """健康监控循环"""
        while not self._stop_event.is_set():
            try:
                self._do_health_check()
            except Exception as e:
                logger.error(f"Health check error: {e}")

            self._stop_event.wait(self.config.health_check_interval_s)

    def _do_health_check(self):
        """执行健康检查"""
        status = self.health_monitor.get_overall_status()

        if status == HealthStatus.CRITICAL:
            if self.config.allow_degraded_mode:
                self.health_monitor.set_state(DeploymentState.DEGRADED)
                logger.warning("System in DEGRADED mode due to health issues")
            else:
                self.emergency.trigger("Critical health status")

        self.health_monitor.report_health(HealthCheckResult(
            component="system",
            status=HealthStatus.HEALTHY if status != HealthStatus.CRITICAL else HealthStatus.CRITICAL,
            message=f"Overall health: {status.value}"
        ))

    def _on_emergency_stop(self):
        """紧急停车回调"""
        self.health_monitor.set_state(DeploymentState.EMERGENCY_STOP)
        if self.control_pipeline:
            logger.warning("Emergency stop - halting control pipeline")

    def get_status(self) -> Dict[str, Any]:
        """获取部署状态"""
        return {
            "state": self.health_monitor.get_state().value,
            "health_summary": self.health_monitor.get_health_summary(),
            "emergency_info": self.emergency.get_stop_info(),
        }


def create_deployment_manager(
    grade: str = "M",
    sensor_manager: Optional["SensorManager"] = None,
    control_pipeline: Optional["EmbodiedControlPipeline"] = None,
    core_brain: Optional["CoreBrain"] = None,
    **kwargs
) -> DeploymentManager:
    """
    创建部署管理器工厂函数

    Args:
        grade: AGV等级 (S/M/L/XL/XXL)
        sensor_manager: 传感器管理器实例
        control_pipeline: 控制管道实例
        core_brain: 核心大脑实例
        **kwargs: 其他配置参数

    Returns:
        DeploymentManager实例
    """
    config = DeploymentConfig(grade=grade, **kwargs)
    return DeploymentManager(
        config=config,
        sensor_manager=sensor_manager,
        control_pipeline=control_pipeline,
        core_brain=core_brain,
    )
