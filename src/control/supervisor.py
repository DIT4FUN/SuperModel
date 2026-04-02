"""
Control Supervisor - 控制子系统监管模块
======================================

控制器的生命周期管理、模式切换与故障恢复
- 多控制器协调调度
- 控制模式自动切换
- 故障检测与安全降级
- 状态监控与日志

支持AGV等级: S / M / L / XL / XXL
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
import time


class ControlMode(Enum):
    """控制模式"""
    IDLE = "idle"
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    JOINT_TORQUE = "joint_torque"
    CARTESIAN_VELOCITY = "cartesian_velocity"
    CARTESIAN_POSITION = "cartesian_position"
    IMPEDANCE = "impedance"
    FORCE = "force"
    ADMITTANCE = "admittance"
    TELEOP = "teleop"
    AUTONOMOUS = "autonomous"
    EMERGENCY_STOP = "emergency_stop"


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAULT = "fault"
    EMERGENCY = "emergency"


@dataclass
class ControllerMetrics:
    """控制器性能指标"""
    name: str
    latency_ms: float = 0.0          # 控制周期延迟
    tracking_error: float = 0.0      # 跟踪误差 RMS
    success_rate: float = 1.0         # 成功率
    cpu_percent: float = 0.0         # CPU占用
    memory_mb: float = 0.0           # 内存占用
    last_update: float = field(default_factory=lambda: time.time())  # 上次更新时间戳


@dataclass
class SupervisorConfig:
    """监管器配置"""
    # 切换策略
    mode_switch_timeout_s: float = 2.0   # 模式切换超时
    controller_heartbeat_s: float = 0.5   # 心跳间隔

    # 故障检测
    max_latency_ms: float = 50.0          # 最大允许延迟
    max_tracking_error: float = 0.5       # 最大跟踪误差 (rad/m)
    fault_count_threshold: int = 3         # 故障判定连续次数

    # 安全降级
    enable_fault_recovery: bool = True    # 自动故障恢复
    graceful_degradation: bool = True     # 优雅降级
    emergency_stop_enabled: bool = True    # 急停使能

    # AGV等级对应的性能目标
    target_latency_ms: float = 10.0        # 目标延迟
    target_rate_hz: float = 200.0         # 目标控制频率


@dataclass
class ControlState:
    """控制子系统状态"""
    mode: ControlMode = ControlMode.IDLE
    health: HealthStatus = HealthStatus.HEALTHY
    active_controllers: List[str] = field(default_factory=list)
    last_control_time: float = 0.0
    uptime_s: float = 0.0
    fault_history: List[Tuple[float, str]] = field(default_factory=list)
    metrics: Dict[str, ControllerMetrics] = field(default_factory=dict)


class ControllerInterface(ABC):
    """
    控制器标准接口

    所有控制器必须实现此接口
    """

    def __init__(self, name: str, controller_type: str):
        self.name = name
        self.controller_type = controller_type
        self.is_active = False
        self._metrics = ControllerMetrics(name=name)

    @abstractmethod
    def start(self) -> bool:
        """启动控制器"""
        pass

    @abstractmethod
    def stop(self) -> bool:
        """停止控制器"""
        pass

    @abstractmethod
    def reset(self):
        """重置控制器状态"""
        pass

    @abstractmethod
    def compute(self, state: Dict, target: Dict) -> Dict:
        """计算控制输出

        Args:
            state: 当前系统状态
            target: 目标状态

        Returns:
            control_output: 控制输出字典
        """
        pass

    def get_metrics(self) -> ControllerMetrics:
        """获取性能指标"""
        return self._metrics

    def health_check(self) -> Tuple[bool, str]:
        """健康检查

        Returns:
            (is_healthy, message)
        """
        return True, "OK"


class ControlSupervisor:
    """
    控制子系统监管器

    职责:
    1. 管理多个控制器的生命周期
    2. 处理控制器之间的模式切换
    3. 监控控制器健康状态和性能指标
    4. 故障检测、安全降级和紧急停止
    5. 记录日志和生成诊断报告
    """

    def __init__(
        self,
        config: Optional[SupervisorConfig] = None,
        supervisor_id: str = "supervisor_0"
    ):
        self.config = config or SupervisorConfig()
        self.supervisor_id = supervisor_id
        self._controllers: Dict[str, ControllerInterface] = {}
        self._state = ControlState()
        self._mode_switch_time: Optional[float] = None
        self._fault_counts: Dict[str, int] = {}
        self._start_time = time.time()
        self._log: List[Dict] = []

    # ── 控制器注册与管理 ─────────────────────────────────

    def register_controller(self, controller: ControllerInterface) -> bool:
        """
        注册控制器

        Args:
            controller: 实现 ControllerInterface 的控制器实例

        Returns:
            bool: 注册是否成功
        """
        if controller.name in self._controllers:
            print(f"[ControlSupervisor] WARNING: Controller '{controller.name}' already registered")
            return False

        self._controllers[controller.name] = controller
        self._state.metrics[controller.name] = ControllerMetrics(name=controller.name)
        self._fault_counts[controller.name] = 0

        self._log_event("controller_registered", {
            "controller": controller.name,
            "type": controller.controller_type
        })

        print(f"[ControlSupervisor] Registered controller: {controller.name} ({controller.controller_type})")
        return True

    def unregister_controller(self, name: str) -> bool:
        """注销控制器"""
        if name not in self._controllers:
            return False

        controller = self._controllers[name]
        if controller.is_active:
            self._deactivate_controller(name)

        del self._controllers[name]
        del self._state.metrics[name]
        del self._fault_counts[name]

        self._log_event("controller_unregistered", {"controller": name})
        print(f"[ControlSupervisor] Unregistered controller: {name}")
        return True

    def get_controller(self, name: str) -> Optional[ControllerInterface]:
        """获取控制器实例"""
        return self._controllers.get(name)

    def list_controllers(self) -> List[str]:
        """列出所有已注册控制器"""
        return list(self._controllers.keys())

    # ── 模式切换 ─────────────────────────────────────────

    def switch_mode(self, target_mode: ControlMode) -> bool:
        """
        切换控制模式

        流程:
        1. 检查目标模式是否有效
        2. 保存当前控制器状态
        3. 停用当前控制器
        4. 激活目标控制器
        5. 验证切换成功

        Args:
            target_mode: 目标控制模式

        Returns:
            bool: 切换是否成功
        """
        if self._state.mode == ControlMode.EMERGENCY_STOP:
            print(f"[ControlSupervisor] Cannot switch mode: EMERGENCY_STOP active")
            return False

        if target_mode == self._state.mode:
            return True

        print(f"[ControlSupervisor] Switching mode: {self._state.mode.value} -> {target_mode.value}")
        self._mode_switch_time = time.time()

        # 查找对应模式的控制器
        target_controller = self._find_controller_for_mode(target_mode)
        if target_controller is None and target_mode != ControlMode.IDLE:
            print(f"[ControlSupervisor] No controller available for mode: {target_mode.value}")
            return False

        # 停用当前控制器
        for name in self._state.active_controllers[:]:
            self._deactivate_controller(name)

        # 激活目标控制器
        if target_mode != ControlMode.IDLE:
            success = self._activate_controller(target_controller.name)
            if not success:
                return False

        # 更新状态
        old_mode = self._state.mode
        self._state.mode = target_mode

        self._log_event("mode_switch", {
            "from": old_mode.value,
            "to": target_mode.value,
            "success": True
        })

        return True

    def _find_controller_for_mode(self, mode: ControlMode) -> Optional[ControllerInterface]:
        """查找支持指定模式的控制器"""
        mode_to_type = {
            ControlMode.JOINT_POSITION: "joint_position",
            ControlMode.JOINT_VELOCITY: "joint_velocity",
            ControlMode.JOINT_TORQUE: "joint_torque",
            ControlMode.CARTESIAN_VELOCITY: "cartesian_velocity",
            ControlMode.CARTESIAN_POSITION: "cartesian_position",
            ControlMode.IMPEDANCE: "impedance",
            ControlMode.FORCE: "force",
            ControlMode.ADMITTANCE: "admittance",
            ControlMode.TELEOP: "teleop",
            ControlMode.AUTONOMOUS: "autonomous",
        }

        target_type = mode_to_type.get(mode)
        if target_type is None:
            return None

        for controller in self._controllers.values():
            if controller.controller_type == target_type and controller.is_active is False:
                return controller

        # 如果没有非活跃的，返回任意匹配类型的
        for name, controller in self._controllers.items():
            if controller.controller_type == target_type:
                return controller

        return None

    def _activate_controller(self, name: str) -> bool:
        """激活控制器"""
        controller = self._controllers.get(name)
        if controller is None:
            return False

        try:
            success = controller.start()
            if success:
                controller.is_active = True
                if name not in self._state.active_controllers:
                    self._state.active_controllers.append(name)
                print(f"[ControlSupervisor] Activated controller: {name}")
            return success
        except Exception as e:
            self._log_event("controller_activation_failed", {
                "controller": name,
                "error": str(e)
            })
            return False

    def _deactivate_controller(self, name: str) -> bool:
        """停用控制器"""
        controller = self._controllers.get(name)
        if controller is None:
            return False

        try:
            controller.stop()
            controller.is_active = False
            if name in self._state.active_controllers:
                self._state.active_controllers.remove(name)
            print(f"[ControlSupervisor] Deactivated controller: {name}")
            return True
        except Exception as e:
            self._log_event("controller_deactivation_failed", {
                "controller": name,
                "error": str(e)
            })
            return False

    # ── 控制循环 ─────────────────────────────────────────

    def control_cycle(self, state: Dict, target: Dict) -> Tuple[Dict, bool]:
        """
        执行一个控制周期

        流程:
        1. 检查系统健康状态
        2. 如果处于急停模式，直接返回急停输出
        3. 执行当前活动控制器
        4. 更新性能指标
        5. 检测故障

        Args:
            state: 当前系统状态
            target: 目标状态

        Returns:
            (control_output, success)
        """
        cycle_start = time.time()
        output = {}

        # 紧急停止检查
        if self._state.mode == ControlMode.EMERGENCY_STOP:
            self._state.health = HealthStatus.EMERGENCY
            return self._emergency_stop_output(), True

        # 健康状态检查
        healthy, msg = self._check_system_health()
        if not healthy:
            self._handle_health_issue(msg)
            return {}, False

        # 执行活动控制器
        for name in self._state.active_controllers:
            controller = self._controllers.get(name)
            if controller is None or not controller.is_active:
                continue

            try:
                cycle_controller_start = time.time()
                ctrl_output = controller.compute(state, target)
                cycle_controller_time = (time.time() - cycle_controller_start) * 1000

                # 更新指标
                metrics = controller.get_metrics()
                metrics.latency_ms = cycle_controller_time
                metrics.last_update = time.time()

                # 合并输出
                if ctrl_output:
                    output = {**output, **ctrl_output}

            except Exception as e:
                self._handle_controller_error(name, str(e))
                return {}, False

        # 检查切换超时
        if self._mode_switch_time is not None:
            elapsed = time.time() - self._mode_switch_time
            if elapsed > self.config.mode_switch_timeout_s:
                print(f"[ControlSupervisor] WARNING: Mode switch timeout ({elapsed:.2f}s)")

        self._state.last_control_time = time.time()
        self._state.uptime_s = time.time() - self._start_time

        return output, True

    def _check_system_health(self) -> Tuple[bool, str]:
        """检查系统整体健康状态"""
        # 检查控制器心跳
        for name, metrics in self._state.metrics.items():
            if time.time() - metrics.last_update > self.config.controller_heartbeat_s * 10:
                return False, f"Controller {name} heartbeat timeout"

        # 检查延迟
        for name, metrics in self._state.metrics.items():
            if metrics.latency_ms > self.config.max_latency_ms:
                return False, f"Controller {name} latency {metrics.latency_ms:.1f}ms exceeds max"

        # 检查跟踪误差
        for name, metrics in self._state.metrics.items():
            if metrics.tracking_error > self.config.max_tracking_error:
                self._fault_counts[name] = self._fault_counts.get(name, 0) + 1
                if self._fault_counts[name] >= self.config.fault_count_threshold:
                    return False, f"Controller {name} tracking error fault"

        return True, "OK"

    def _handle_health_issue(self, message: str):
        """处理健康问题"""
        self._state.health = HealthStatus.DEGRADED

        if self.config.enable_fault_recovery:
            # 尝试自动恢复
            print(f"[ControlSupervisor] Attempting recovery: {message}")
            self._attempt_recovery(message)
        else:
            print(f"[ControlSupervisor] Health issue (no auto-recovery): {message}")

        self._log_event("health_issue", {"message": message})

    def _handle_controller_error(self, controller_name: str, error: str):
        """处理控制器错误"""
        self._fault_counts[controller_name] = self._fault_counts.get(controller_name, 0) + 1
        count = self._fault_counts[controller_name]

        self._state.fault_history.append((time.time(), f"{controller_name}: {error}"))
        self._state.health = HealthStatus.FAULT

        print(f"[ControlSupervisor] Controller error [{controller_name}] (fault #{count}): {error}")

        if count >= self.config.fault_count_threshold:
            if self.config.graceful_degradation:
                self._degrade_controller(controller_name)
            else:
                self.trigger_emergency_stop(f"Controller {controller_name} fault threshold reached")

        self._log_event("controller_error", {
            "controller": controller_name,
            "error": error,
            "fault_count": count
        })

    def _attempt_recovery(self, message: str) -> bool:
        """尝试故障恢复"""
        # 尝试重置所有控制器
        for name, controller in self._controllers.items():
            try:
                controller.reset()
                self._fault_counts[name] = 0
            except Exception:
                pass

        # 如果当前模式有问题，切换到空闲
        if "latency" in message.lower() or "timeout" in message.lower():
            print(f"[ControlSupervisor] Recovery: attempting mode reset")
            self.switch_mode(ControlMode.IDLE)

        return True

    def _degrade_controller(self, name: str):
        """优雅降级: 停用故障控制器"""
        if name in self._state.active_controllers:
            self._deactivate_controller(name)
            print(f"[ControlSupervisor] Degraded controller: {name}")

            # 尝试切换到备用模式
            if len(self._state.active_controllers) == 0:
                fallback = self._find_fallback_mode()
                if fallback:
                    print(f"[ControlSupervisor] Falling back to mode: {fallback.value}")
                    self.switch_mode(fallback)

    def _find_fallback_mode(self) -> Optional[ControlMode]:
        """查找备用控制模式"""
        # 优先级: IDLE -> JOINT_VELOCITY -> JOINT_POSITION -> 任何可用的
        fallback_order = [
            ControlMode.IDLE,
            ControlMode.JOINT_VELOCITY,
            ControlMode.JOINT_POSITION,
            ControlMode.CARTESIAN_VELOCITY,
        ]

        for mode in fallback_order:
            if self._find_controller_for_mode(mode) is not None:
                return mode

        return None

    # ── 紧急停止 ─────────────────────────────────────────

    def trigger_emergency_stop(self, reason: str = ""):
        """
        触发紧急停止

        立即停用所有控制器，切换到急停模式
        """
        print(f"[ControlSupervisor] !!! EMERGENCY STOP !!! Reason: {reason}")

        # 停用所有控制器
        for name in self._state.active_controllers[:]:
            self._deactivate_controller(name)

        self._state.mode = ControlMode.EMERGENCY_STOP
        self._state.health = HealthStatus.EMERGENCY

        self._state.fault_history.append((time.time(), f"EMERGENCY_STOP: {reason}"))
        self._log_event("emergency_stop", {"reason": reason})

    def release_emergency_stop(self) -> bool:
        """
        释放紧急停止

        只有在故障排除后才能释放
        """
        if self._state.health != HealthStatus.EMERGENCY:
            return True

        print(f"[ControlSupervisor] Releasing EMERGENCY_STOP")
        self._state.mode = ControlMode.IDLE
        self._state.health = HealthStatus.HEALTHY
        self._state.active_controllers = []

        self._log_event("emergency_stop_released", {})
        return True

    def _emergency_stop_output(self) -> Dict:
        """生成急停输出 (零速度/零力矩)"""
        return {
            "joint_velocity": np.zeros(6),
            "joint_torque": np.zeros(6),
            "cartesian_velocity": np.zeros(6),
            "emergency_stop": True
        }

    # ── 状态查询 ─────────────────────────────────────────

    def get_state(self) -> ControlState:
        """获取当前控制子系统状态"""
        return self._state

    def get_diagnostics(self) -> Dict[str, Any]:
        """获取诊断报告"""
        return {
            "supervisor_id": self.supervisor_id,
            "uptime_s": self._state.uptime_s,
            "mode": self._state.mode.value,
            "health": self._state.health.value,
            "active_controllers": self._state.active_controllers,
            "registered_controllers": self.list_controllers(),
            "metrics": {
                name: {
                    "latency_ms": m.latency_ms,
                    "tracking_error": m.tracking_error,
                    "success_rate": m.success_rate,
                    "last_update": m.last_update
                }
                for name, m in self._state.metrics.items()
            },
            "fault_history": self._state.fault_history[-10:],
            "log": self._log[-20:]
        }

    def print_diagnostics(self):
        """打印诊断信息"""
        diag = self.get_diagnostics()
        print(f"\n{'='*60}")
        print(f"ControlSupervisor Diagnostics: {diag['supervisor_id']}")
        print(f"{'='*60}")
        print(f"  Mode: {diag['mode']} | Health: {diag['health']}")
        print(f"  Uptime: {diag['uptime_s']:.1f}s")
        print(f"  Active Controllers: {diag['active_controllers']}")
        print(f"  Registered Controllers: {diag['registered_controllers']}")
        print(f"  Metrics:")

        for name, m in diag['metrics'].items():
            print(f"    {name}: latency={m['latency_ms']:.2f}ms, "
                  f"error={m['tracking_error']:.4f}, "
                  f"rate={m['success_rate']:.2%}")

        if diag['fault_history']:
            print(f"  Recent Faults:")
            for ts, msg in diag['fault_history'][-5:]:
                print(f"    [{ts:.1f}] {msg}")

        print(f"{'='*60}\n")

    # ── 日志 ─────────────────────────────────────────────

    def _log_event(self, event_type: str, data: Dict):
        """记录事件"""
        self._log.append({
            "timestamp": time.time(),
            "type": event_type,
            "data": data
        })

        # 限制日志长度
        if len(self._log) > 1000:
            self._log = self._log[-500:]

    def get_log(self, max_entries: int = 100) -> List[Dict]:
        """获取事件日志"""
        return self._log[-max_entries:]

    def clear_log(self):
        """清空日志"""
        self._log = []

    # ── 上下文管理 ───────────────────────────────────────

    def __enter__(self):
        self._start_time = time.time()
        print(f"[ControlSupervisor] {self.supervisor_id} started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 停用所有控制器
        for name in list(self._controllers.keys()):
            self.unregister_controller(name)

        print(f"[ControlSupervisor] {self.supervisor_id} stopped (uptime: {self._state.uptime_s:.1f}s)")


# ── Mock控制器 (用于测试) ──────────────────────────────────────────────

class MockJointController(ControllerInterface):
    """模拟关节控制器 (用于测试Supervisor)"""

    def __init__(self, name: str = "mock_joint"):
        super().__init__(name, "joint_position")
        self._joints = 6

    def start(self) -> bool:
        self.is_active = True
        self._metrics.last_update = time.time()
        print(f"[MockJointController] {self.name} started")
        return True

    def stop(self) -> bool:
        self.is_active = False
        print(f"[MockJointController] {self.name} stopped")
        return True

    def reset(self):
        self._metrics = ControllerMetrics(name=self.name)

    def compute(self, state: Dict, target: Dict) -> Dict:
        # 模拟控制计算延迟
        import time
        time.sleep(0.001)

        return {
            "joint_velocity": np.zeros(self._joints),
            "joint_torque": np.zeros(self._joints)
        }


class MockCartesianController(ControllerInterface):
    """模拟笛卡尔控制器 (用于测试Supervisor)"""

    def __init__(self, name: str = "mock_cartesian"):
        super().__init__(name, "cartesian_velocity")
        self._dof = 6

    def start(self) -> bool:
        self.is_active = True
        self._metrics.last_update = time.time()
        print(f"[MockCartesianController] {self.name} started")
        return True

    def stop(self) -> bool:
        self.is_active = False
        print(f"[MockCartesianController] {self.name} stopped")
        return True

    def reset(self):
        self._metrics = ControllerMetrics(name=self.name)

    def compute(self, state: Dict, target: Dict) -> Dict:
        return {
            "cartesian_velocity": np.zeros(self._dof)
        }


class MockImpedanceController(ControllerInterface):
    """模拟阻抗控制器 (用于测试Supervisor)"""

    def __init__(self, name: str = "mock_impedance"):
        super().__init__(name, "impedance")

    def start(self) -> bool:
        self.is_active = True
        self._metrics.last_update = time.time()
        return True

    def stop(self) -> bool:
        self.is_active = False
        return True

    def reset(self):
        self._metrics = ControllerMetrics(name=self.name)

    def compute(self, state: Dict, target: Dict) -> Dict:
        return {
            "cartesian_velocity": np.zeros(6),
            "joint_torque": np.zeros(6)
        }


# ── AGV五级监管器规格表 ──────────────────────────────────────────────────

class SupervisorGrade(Enum):
    """监管器等级 (与AGV五级对应)"""
    S = "S"
    M = "M"
    L = "L"
    XL = "XL"
    XXL = "XXL"


@dataclass
class SupervisorGradeSpec:
    """AGV五级监管器规格"""
    grade: SupervisorGrade
    target_latency_ms: float
    target_rate_hz: float
    max_latency_ms: float
    fault_detection: bool
    fault_prediction: bool
    fault_isolation: bool
    fault_recovery: bool
    safety_level: str
    emergency_stop_level: int
    graceful_degradation: bool
    controller_redundancy: int
    hot_swap: bool
    watchdog_enabled: bool
    watchdog_timeout_ms: float
    log_depth: int
    diagnostics_level: str
    mode_switch_timeout_s: float
    auto_mode_switch: bool


def get_supervisor_spec(grade: SupervisorGrade) -> SupervisorGradeSpec:
    specs = {
        SupervisorGrade.S: SupervisorGradeSpec(
            grade=SupervisorGrade.S, target_latency_ms=20.0, target_rate_hz=50.0,
            max_latency_ms=100.0, fault_detection=True, fault_prediction=False,
            fault_isolation=False, fault_recovery=False, safety_level="basic",
            emergency_stop_level=1, graceful_degradation=False, controller_redundancy=0,
            hot_swap=False, watchdog_enabled=False, watchdog_timeout_ms=0.0,
            log_depth=100, diagnostics_level="basic", mode_switch_timeout_s=5.0,
            auto_mode_switch=False,
        ),
        SupervisorGrade.M: SupervisorGradeSpec(
            grade=SupervisorGrade.M, target_latency_ms=10.0, target_rate_hz=100.0,
            max_latency_ms=50.0, fault_detection=True, fault_prediction=False,
            fault_isolation=True, fault_recovery=True, safety_level="standard",
            emergency_stop_level=2, graceful_degradation=True, controller_redundancy=0,
            hot_swap=False, watchdog_enabled=False, watchdog_timeout_ms=0.0,
            log_depth=500, diagnostics_level="basic", mode_switch_timeout_s=2.0,
            auto_mode_switch=True,
        ),
        SupervisorGrade.L: SupervisorGradeSpec(
            grade=SupervisorGrade.L, target_latency_ms=5.0, target_rate_hz=200.0,
            max_latency_ms=25.0, fault_detection=True, fault_prediction=True,
            fault_isolation=True, fault_recovery=True, safety_level="enhanced",
            emergency_stop_level=3, graceful_degradation=True, controller_redundancy=1,
            hot_swap=False, watchdog_enabled=False, watchdog_timeout_ms=0.0,
            log_depth=1000, diagnostics_level="detailed", mode_switch_timeout_s=1.0,
            auto_mode_switch=True,
        ),
        SupervisorGrade.XL: SupervisorGradeSpec(
            grade=SupervisorGrade.XL, target_latency_ms=2.0, target_rate_hz=500.0,
            max_latency_ms=10.0, fault_detection=True, fault_prediction=True,
            fault_isolation=True, fault_recovery=True, safety_level="high",
            emergency_stop_level=4, graceful_degradation=True, controller_redundancy=2,
            hot_swap=True, watchdog_enabled=True, watchdog_timeout_ms=5.0,
            log_depth=5000, diagnostics_level="detailed", mode_switch_timeout_s=0.5,
            auto_mode_switch=True,
        ),
        SupervisorGrade.XXL: SupervisorGradeSpec(
            grade=SupervisorGrade.XXL, target_latency_ms=1.0, target_rate_hz=1000.0,
            max_latency_ms=5.0, fault_detection=True, fault_prediction=True,
            fault_isolation=True, fault_recovery=True, safety_level="critical",
            emergency_stop_level=5, graceful_degradation=True, controller_redundancy=3,
            hot_swap=True, watchdog_enabled=True, watchdog_timeout_ms=2.0,
            log_depth=20000, diagnostics_level="comprehensive", mode_switch_timeout_s=0.2,
            auto_mode_switch=True,
        ),
    }
    return specs.get(grade, specs[SupervisorGrade.M])


def get_supervisor_config(grade: SupervisorGrade) -> SupervisorConfig:
    spec = get_supervisor_spec(grade)
    return SupervisorConfig(
        mode_switch_timeout_s=spec.mode_switch_timeout_s,
        controller_heartbeat_s=1.0 / spec.target_rate_hz,
        max_latency_ms=spec.max_latency_ms,
        max_tracking_error=0.5 if grade in [SupervisorGrade.S, SupervisorGrade.M] else 0.1,
        fault_count_threshold=5 if grade == SupervisorGrade.S else (3 if grade == SupervisorGrade.M else 1),
        enable_fault_recovery=spec.fault_recovery,
        graceful_degradation=spec.graceful_degradation,
        emergency_stop_enabled=True,
        target_latency_ms=spec.target_latency_ms,
        target_rate_hz=spec.target_rate_hz,
    )


class GradeAwareSupervisor(ControlSupervisor):
    """
    AGV五级感知控制监管器
    
    根据AGV等级自动配置:
    - S级: 基础监控，50Hz，5s切换超时，无冗余
    - M级: 标准监控，100Hz，2s切换超时，优雅降级
    - L级: 专业监控，200Hz，1s切换超时，单冗余
    - XL级: 高性能监控，500Hz，0.5s切换超时，双冗余+看门狗
    - XXL级: 旗舰监控，1000Hz，0.2s切换超时，三冗余+看门狗+自愈
    """

    def __init__(
        self,
        grade: SupervisorGrade = SupervisorGrade.M,
        supervisor_id: str = "grade_aware_supervisor"
    ):
        self.grade = grade
        self.grade_spec = get_supervisor_spec(grade)
        config = get_supervisor_config(grade)
        super().__init__(config=config, supervisor_id=supervisor_id)
        self._primary_controllers: Dict[str, str] = {}
        self._backup_controllers: Dict[str, List[str]] = {}
        self._hot_standby: bool = grade.value in ["XL", "XXL"]
        self._watchdog_enabled = self.grade_spec.watchdog_enabled
        self._watchdog_timers: Dict[str, float] = {}
        self._last_heartbeat: float = time.time()
        self._fault_tolerance_enabled = (grade == SupervisorGrade.XXL)
        self._consecutive_faults: int = 0
        self._max_consecutive_faults: int = 10
        print(f"[GradeAwareSupervisor] grade={grade.value}, rate={self.grade_spec.target_rate_hz}Hz, "
              f"latency={self.grade_spec.target_latency_ms}ms, "
              f"redundancy={self.grade_spec.controller_redundancy}, watchdog={self._watchdog_enabled}")

    def register_with_redundancy(self, controller: ControllerInterface, modes: List[ControlMode],
                                 is_primary: bool = True) -> bool:
        success = self.register_controller(controller)
        if not success:
            return False
        for mode in modes:
            mode_key = mode.value
            if is_primary:
                self._primary_controllers[mode_key] = controller.name
                self._backup_controllers.setdefault(mode_key, [])
            else:
                self._backup_controllers.setdefault(mode_key, [])
                self._backup_controllers[mode_key].append(controller.name)
        return True

    def kick_watchdog(self, controller_name: str):
        if not self._watchdog_enabled:
            return
        self._watchdog_timers[controller_name] = time.time()
        self._last_heartbeat = time.time()

    def _check_watchdog(self) -> Tuple[bool, str]:
        if not self._watchdog_enabled:
            return True, "OK"
        current_time = time.time()
        timeout = self.grade_spec.watchdog_timeout_ms / 1000.0
        for name, last_time in list(self._watchdog_timers.items()):
            if current_time - last_time > timeout:
                return False, f"Watchdog timeout for controller: {name}"
        return True, "OK"

    def get_grade_capabilities(self) -> Dict[str, Any]:
        spec = self.grade_spec
        return {
            "grade": spec.grade.value,
            "performance": {
                "target_latency_ms": spec.target_latency_ms,
                "target_rate_hz": spec.target_rate_hz,
                "max_latency_ms": spec.max_latency_ms,
            },
            "fault_handling": {
                "detection": spec.fault_detection,
                "prediction": spec.fault_prediction,
                "isolation": spec.fault_isolation,
                "recovery": spec.fault_recovery,
            },
            "safety": {
                "level": spec.safety_level,
                "emergency_stop_level": spec.emergency_stop_level,
                "graceful_degradation": spec.graceful_degradation,
            },
            "redundancy": {
                "controller_count": spec.controller_redundancy,
                "hot_swap": spec.hot_swap,
                "hot_standby": self._hot_standby,
            },
            "watchdog": {"enabled": spec.watchdog_enabled, "timeout_ms": spec.watchdog_timeout_ms},
            "diagnostics": {"level": spec.diagnostics_level, "log_depth": spec.log_depth},
        }

    def step_watchdog(self) -> bool:
        if not self._watchdog_enabled:
            return True
        healthy, msg = self._check_watchdog()
        if not healthy:
            print(f"[GradeAwareSupervisor] WATCHDOG TRIGGERED: {msg}")
            self.trigger_emergency_stop(f"watchdog: {msg}")
            return False
        return True

    def step_fault_tolerance(self, fault_detected: bool) -> bool:
        if not self._fault_tolerance_enabled:
            return not fault_detected
        if fault_detected:
            self._consecutive_faults += 1
            if self._consecutive_faults > self._max_consecutive_faults:
                print(f"[GradeAwareSupervisor] CRITICAL: Max consecutive faults exceeded")
                self.trigger_emergency_stop("fault_tolerance_exceeded")
                return False
        else:
            self._consecutive_faults = max(0, self._consecutive_faults - 1)
        return True
