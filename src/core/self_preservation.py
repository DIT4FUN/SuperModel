"""
Self-Preservation - 自我保存模块 (P4执行器)
==========================================

实现 P4 自我生存安全 核心目标:

目标:
  保护自身硬件完整性和软件系统健康,以持续为人类服务

功能:
  - 硬件完整性保护: 避免碰撞损坏/过热/过载
  - 软件完整性维护: 保持系统稳定性/故障检测
  - 能源管理: 电池管理/功耗优化
  - 资源合理使用: 计算资源/内存/带宽
  - 自我健康监测: 定期自检/异常预警

状态评估:
  - 综合健康评分 [0.0, 1.0]
  - 各子系统健康状态
  - 预测性维护建议
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import time
import threading


@dataclass
class SubsystemHealth:
    """子系统健康状态"""
    name: str
    health_score: float = 1.0       # [0.0, 1.0]
    temperature_c: float = 25.0     # °C
    vibration_g: float = 0.0        # g
    error_count: int = 0
    last_error: Optional[str] = None
    uptime_hours: float = 0.0
    is_critical: bool = False       # 是否关键子系统


@dataclass
class SelfPreservationState:
    """
    自我保存状态

    包含所有与P4目标相关的状态信息:
    - 各子系统健康
    - 能源状态
    - 安全状态
    - 完整性状态
    """
    overall_health: float = 1.0    # 综合健康评分
    subsystem_states: Dict[str, SubsystemHealth] = field(default_factory=dict)
    battery_level: float = 1.0     # 电量 [0, 1]
    battery_temperature_c: float = 25.0  # 电池温度
    power_consumption_w: float = 0.0  # 当前功耗 (W)
    available_compute_percent: float = 100.0  # 可用计算资源
    memory_usage_mb: float = 0.0    # 内存使用 (MB)
    network_latency_ms: float = 0.0  # 网络延迟

    # 碰撞相关
    collision_count: int = 0
    last_collision_time: Optional[float] = None
    structural_integrity: float = 1.0  # 结构完整性 [0, 1]

    # 状态时间
    total_uptime_hours: float = 0.0
    last_self_check: Optional[float] = None

    @property
    def is_healthy(self) -> bool:
        return self.overall_health > 0.6

    @property
    def needs_maintenance(self) -> bool:
        return self.overall_health < 0.4


class SelfPreservation:
    """
    自我保存系统 - P4核心目标的执行器

    职责:
    1. 持续监测自身各子系统的健康状态
    2. 在健康评分降低时采取保护措施
    3. 预测性维护 - 在故障发生前采取措施
    4. 能源优化 - 平衡性能和续航

    保护策略:
    - 健康评分 < 0.8: 启动警告
    - 健康评分 < 0.5: 限制功率/速度
    - 健康评分 < 0.3: 进入低功耗模式
    - 健康评分 < 0.1: 寻找安全地方停机

    使用方式:
      sp = SelfPreservation()

      # 更新状态
      sp.update_state(context)

      # 获取保护动作
      protective_action = sp.get_protective_action(context)

      # 获取健康评分 (传递给GoalContext)
      health_score = sp.get_health_score()
    """

    def __init__(self):
        self._state = SelfPreservationState()
        self._lock = threading.RLock()
        self._maintenance_log: List[Dict[str, Any]] = []

        # 初始化默认子系统
        self._init_default_subsystems()

    def _init_default_subsystems(self):
        """初始化默认子系统健康状态"""
        default_subsystems = [
            ("motor_left", True),
            ("motor_right", True),
            ("battery", True),
            ("imu", False),
            ("vision", False),
            ("tactile", False),
            ("force_sensor", False),
            ("network", False),
            ("compute", True),
        ]
        for name, is_critical in default_subsystems:
            self._state.subsystem_states[name] = SubsystemHealth(
                name=name,
                is_critical=is_critical,
            )

    def update_state(self, context: Any):
        """
        根据上下文更新自我保存状态

        Args:
            context: GoalContext 或类似的上下文对象
        """
        with self._lock:
            # 更新电池状态
            if hasattr(context, 'robot_battery_level'):
                self._state.battery_level = context.robot_battery_level

            # 更新温度
            if hasattr(context, 'robot_temperature'):
                self._state.subsystem_states['compute'].temperature_c = (
                    context.robot_temperature
                )

            # 更新计算资源
            if hasattr(context, 'self_health_score'):
                self._state.overall_health = context.self_health_score

            # 检查故障列表
            if hasattr(context, 'robot_faults') and context.robot_faults:
                for fault in context.robot_faults:
                    self._handle_fault(fault)

            # 更新运行时间
            self._state.total_uptime_hours += 0.01 / 3600  # 假设10ms周期

            # 重新计算综合健康评分
            self._recalculate_health()

    def _handle_fault(self, fault: str):
        """处理故障"""
        # 查找可能的子系统
        fault_lower = fault.lower()
        for name, health in self._state.subsystem_states.items():
            if name in fault_lower:
                health.error_count += 1
                health.last_error = fault
                self._log_maintenance("fault_detected", {
                    "subsystem": name,
                    "fault": fault,
                })
                break

    def _recalculate_health(self):
        """重新计算综合健康评分"""
        scores = []
        weights = []

        for name, health in self._state.subsystem_states.items():
            # 基础评分
            score = health.health_score

            # 温度影响 (过高或过低都降低评分)
            if health.temperature_c > 60:
                score *= 0.7
            elif health.temperature_c > 45:
                score *= 0.9

            # 错误次数影响
            if health.error_count > 0:
                score *= max(0.3, 1.0 - health.error_count * 0.1)

            # 权重: 关键子系统权重更高
            weight = 3.0 if health.is_critical else 1.0

            scores.append(score * weight)
            weights.append(weight)

        # 电池特殊处理
        battery_score = self._state.battery_level
        if self._state.battery_level < 0.2:
            battery_score *= 0.5  # 低电量时大幅降低评分

        if weights:
            self._state.overall_health = (
                sum(scores) / sum(weights) * 0.7 +
                battery_score * 0.3
            )
        else:
            self._state.overall_health = battery_score

        self._state.overall_health = max(0.0, min(1.0, self._state.overall_health))

    def get_protective_action(self, context: Any) -> Tuple[np.ndarray, str]:
        """
        获取保护性动作

        根据当前健康状态返回适当的保护动作:

        Returns:
            Tuple[np.ndarray, str]: (保护动作, 保护原因)
        """
        health = self._state.overall_health

        # 紧急情况 - 寻找安全停机点
        if health < 0.1:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "health_critical_stop"

        # 低功耗模式
        if health < 0.3:
            # 限制所有运动
            action = np.zeros(6)
            return action, "low_power_mode"

        # 限制性能
        if health < 0.5:
            # 限制速度50%
            action = np.zeros(6)
            return action, "performance_limited"

        # 正常状态 - 无需特殊保护
        return np.zeros(6), "no_action_needed"

    def get_health_score(self) -> float:
        """获取综合健康评分"""
        return self._state.overall_health

    def get_subsystem_scores(self) -> Dict[str, float]:
        """获取各子系统评分"""
        return {
            name: h.health_score
            for name, h in self._state.subsystem_states.items()
        }

    def check_battery_safety(self) -> Tuple[bool, str]:
        """
        检查电池安全

        Returns:
            Tuple[bool, str]: (是否安全, 描述)
        """
        if self._state.battery_level < 0.05:
            return False, f"电池严重电量不足: {self._state.battery_level*100:.1f}%"
        if self._state.battery_temperature_c > 50:
            return False, f"电池温度过高: {self._state.battery_temperature_c:.1f}°C"
        if self._state.battery_level < 0.15:
            return False, f"电池电量低: {self._state.battery_level*100:.1f}%, 建议充电"
        return True, "电池状态正常"

    def get_recommended_action(self) -> Optional[Dict[str, Any]]:
        """
        获取建议的行动

        基于当前状态返回建议的行动:
        - 充电
        - 停机维护
        - 降低负载
        - 寻求帮助
        """
        recommendations = []

        if self._state.battery_level < 0.15:
            recommendations.append({
                "action": "seek_charging",
                "priority": "high",
                "reason": f"电量{self._state.battery_level*100:.1f}%过低"
            })

        if self._state.overall_health < 0.5:
            recommendations.append({
                "action": "reduce_performance",
                "priority": "medium",
                "reason": f"健康评分{self._state.overall_health:.2f}低于阈值"
            })

        for name, health in self._state.subsystem_states.items():
            if health.error_count > 3:
                recommendations.append({
                    "action": "maintenance_required",
                    "priority": "high" if health.is_critical else "medium",
                    "reason": f"子系统{name}错误次数{health.error_count}过多"
                })

        if recommendations:
            return sorted(
                recommendations,
                key=lambda x: {"high": 0, "medium": 1, "low": 2}[x["priority"]]
            )[0]
        return None

    def _log_maintenance(self, event_type: str, data: Dict[str, Any]):
        """记录维护事件"""
        self._maintenance_log.append({
            "timestamp": time.time(),
            "type": event_type,
            **data
        })
        if len(self._maintenance_log) > 500:
            self._maintenance_log = self._maintenance_log[-500:]

    def get_status(self) -> Dict[str, Any]:
        """获取自我保存系统状态"""
        return {
            "overall_health": self._state.overall_health,
            "is_healthy": self._state.is_healthy,
            "needs_maintenance": self._state.needs_maintenance,
            "battery_level": self._state.battery_level,
            "total_uptime_hours": self._state.total_uptime_hours,
            "subsystems": {
                name: {
                    "health_score": h.health_score,
                    "temperature_c": h.temperature_c,
                    "error_count": h.error_count,
                    "is_critical": h.is_critical,
                }
                for name, h in self._state.subsystem_states.items()
            },
            "maintenance_log_size": len(self._maintenance_log),
        }
