"""
multi_robot_load_balancer.py - 多机器人动态负载均衡器
======================================================

功能:
  - 实时监控所有AGV的负载状态 (CPU/内存/任务队列/电池/传感器)
  - 动态任务重分配: 基于负载权重进行任务迁移
  - 三种均衡算法: 轮询/最小负载/能力感知
  - 过载保护: 自动迁移任务防止单点故障
  - 能耗均衡: 避免单一AGV过度消耗导致提早失效
  - 适应性调节: 根据任务类型动态调整权重

Author: SuperModel Development Team
Version: 3.15.0
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple, Any

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LoadMetric(Enum):
    """负载指标类型"""
    CPU_USAGE = "cpu"
    MEMORY_USAGE = "memory"
    TASK_QUEUE_LENGTH = "task_queue"
    BATTERY_LEVEL = "battery"  # lower is more loaded (low battery = high stress)
    THERMAL_LEVEL = "thermal"
    SENSOR_LATENCY = "sensor_latency"
    NETWORK_LATENCY = "network_latency"
    MOTOR_CURRENT = "motor_current"

class BalanceStrategy(Enum):
    """均衡策略"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    CAPABILITY_AWARE = "capability_aware"
    ENERGY_AWARE = "energy_aware"
    HYBRID = "hybrid"

class RebalanceTrigger(Enum):
    """重均衡触发条件"""
    THRESHOLD_EXCEEDED = "threshold"  # 负载超过阈值
    TIME_INTERVAL = "interval"        # 定时触发
    TASK_COMPLETION = "task_done"      # 任务完成时检查
    BATTERY_CRITICAL = "battery_critical"
    MANUAL = "manual"

@dataclass
class AGVLoadProfile:
    """AGV负载画像"""
    agv_id: str
    timestamp: float = field(default_factory=time.time)
    # 原始指标
    cpu_usage: float = 0.0       # 0.0-1.0
    memory_usage: float = 0.0    # 0.0-1.0
    task_queue_depth: int = 0
    task_queue_weights: List[float] = field(default_factory=list)  # 每任务权重
    battery_level: float = 1.0   # 0.0-1.0
    thermal_level: float = 0.0   # 0.0-1.0 (temperature margin to max)
    sensor_latency_ms: float = 0.0
    motor_current_ma: float = 0.0
    # 能力参数
    max_payload_kg: float = 100.0
    max_speed_mps: float = 2.0
    capability_score: float = 1.0  # 0.0-1.0 综合能力评分
    # 当前状态
    current_load_kg: float = 0.0
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # 计算属性
    _composite_load: Optional[float] = field(default=None, repr=False)
    _energy_stress: float = field(default=0.0, repr=False)

    def compute_composite_load(self) -> float:
        """计算综合负载分数 (0.0=空闲, 1.0=满载)"""
        cpu_weight = 0.25
        mem_weight = 0.15
        queue_weight = 0.30
        thermal_weight = 0.15
        battery_weight = 0.15

        # 任务队列归一化 (假设max=20)
        queue_norm = min(self.task_queue_depth / 20.0, 1.0)

        # 电池压力 (低电池=高负载)
        battery_stress = 1.0 - self.battery_level

        # 综合计算
        load = (
            cpu_weight * self.cpu_usage
            + mem_weight * self.memory_usage
            + queue_weight * queue_norm
            + thermal_weight * self.thermal_level
            + battery_weight * battery_stress
        )
        self._composite_load = min(load, 1.0)
        return self._composite_load

    def compute_energy_stress(self) -> float:
        """计算能耗压力"""
        base_rate = 0.01  # 基准消耗率 (每小时1%)
        load_factor = self.compute_composite_load()
        # 高负载加速消耗
        stress = base_rate * (1.0 + load_factor * 3.0)
        self._energy_stress = stress
        return stress

    @property
    def composite_load(self) -> float:
        if self._composite_load is None:
            return self.compute_composite_load()
        return self._composite_load

    def get_distance_to(self, x: float, y: float) -> float:
        """到目标点的距离"""
        return math.sqrt(
            (self.position[0] - x)**2
            + (self.position[1] - y)**2
        )


@dataclass
class TaskSpec:
    """任务规格 (用于负载均衡决策)"""
    task_id: str
    priority: int = 3  # 1=最高
    workload_units: float = 1.0  # 工作量单位
    required_capability: float = 0.0  # 最低能力需求 0.0-1.0
    preferred_zones: Set[str] = field(default_factory=set)  # 偏好区域
    deadline: Optional[float] = None  # 截止时间
    created_at: float = field(default_factory=time.time)
    estimated_duration_s: float = 60.0
    from_position: Tuple[float, float] = (0.0, 0.0)
    to_position: Tuple[float, float] = (0.0, 0.0)
    assigned_agv_id: Optional[str] = None
    migrated_count: int = 0


@dataclass
class LoadThreshold:
    """负载阈值配置"""
    overload_threshold: float = 0.85   # 触发重均衡的负载阈值
    underload_threshold: float = 0.25   # 触发任务分配的空闲阈值
    critical_battery: float = 0.15      # 低电量阈值
    warning_battery: float = 0.25       # 电量警告阈值
    thermal_warning: float = 0.80       # 温度警告阈值
    max_queue_depth: int = 15           # 最大任务队列深度

    def is_overloaded(self, profile: AGVLoadProfile) -> bool:
        return profile.composite_load >= self.overload_threshold

    def is_critical(self, profile: AGVLoadProfile) -> bool:
        return (
            profile.battery_level <= self.critical_battery
            or profile.composite_load >= 0.95
        )


@dataclass
class RebalanceDecision:
    """重均衡决策"""
    decision_id: str
    trigger: RebalanceTrigger
    from_agv_id: str
    to_agv_id: str
    task_id: str
    reason: str
    estimated_benefit: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class LoadStats:
    """负载统计"""
    total_agvs: int
    overloaded_agvs: int
    idle_agvs: int
    critical_agvs: int
    avg_load: float
    load_variance: float
    total_tasks: int
    pending_tasks: int
    completed_tasks: int
    rebalance_events: int
    avg_queue_depth: float


# ---------------------------------------------------------------------------
# Load Balancing Strategies
# ---------------------------------------------------------------------------

class BalanceStrategyBase(ABC):
    """均衡策略基类"""

    @abstractmethod
    def select_target_agv(
        self,
        task: TaskSpec,
        profiles: Dict[str, AGVLoadProfile],
        threshold: LoadThreshold,
    ) -> Optional[str]:
        """为目标任务选择最优AGV"""
        pass

    @abstractmethod
    def select_task_to_migrate(
        self,
        profile: AGVLoadProfile,
        profiles: Dict[str, AGVLoadProfile],
        threshold: LoadThreshold,
    ) -> Optional[TaskSpec]:
        """选择要从过载AGV迁移的任务"""
        pass


class RoundRobinStrategy(BalanceStrategyBase):
    """轮询策略 (不考虑负载)"""

    def __init__(self):
        self._index = 0

    def select_target_agv(
        self,
        task: TaskSpec,
        profiles: Dict[str, AGVLoadProfile],
        threshold: LoadThreshold,
    ) -> Optional[str]:
        if not profiles:
            return None
        agv_ids = sorted(profiles.keys())
        self._index = (self._index + 1) % len(agv_ids)
        return agv_ids[self._index]

    def select_task_to_migrate(
        self,
        profile: AGVLoadProfile,
        profiles: Dict[str, AGVLoadProfile],
        threshold: LoadThreshold,
    ) -> Optional[TaskSpec]:
        return None  # 不主动迁移


class LeastLoadedStrategy(BalanceStrategyBase):
    """最小负载策略"""

    def select_target_agv(
        self,
        task: TaskSpec,
        profiles: Dict[str, AGVLoadProfile],
        threshold: LoadThreshold,
    ) -> Optional[str]:
        candidates = {
            agv_id: p for agv_id, p in profiles.items()
            if not threshold.is_overloaded(p) and p.task_queue_depth < threshold.max_queue_depth
        }
        if not candidates:
            return None
        return min(candidates, key=lambda aid: candidates[aid].composite_load)

    def select_task_to_migrate(
        self,
        profile: AGVLoadProfile,
        all_tasks: Dict[str, TaskSpec],
        threshold: LoadThreshold,
    ) -> Optional[TaskSpec]:
        # 迁移优先级最低、权重最小的任务 (只选分配给该profile的任务)
        my_tasks = [t for t in all_tasks.values() if t.assigned_agv_id == profile.agv_id]
        if not my_tasks:
            return None
        return min(my_tasks, key=lambda t: t.priority * t.workload_units)


class CapabilityAwareStrategy(BalanceStrategyBase):
    """能力感知策略"""

    def select_target_agv(
        self,
        task: TaskSpec,
        profiles: Dict[str, AGVLoadProfile],
        threshold: LoadThreshold,
    ) -> Optional[str]:
        candidates = {}
        for agv_id, profile in profiles.items():
            if threshold.is_overloaded(profile):
                continue
            if profile.task_queue_depth >= threshold.max_queue_depth:
                continue
            if profile.capability_score < task.required_capability:
                continue
            # 综合评分: 能力 * (1 - 负载)
            score = profile.capability_score * (1.0 - profile.composite_load)
            candidates[agv_id] = score

        if not candidates:
            return None
        return max(candidates, key=candidates.get)

    def select_task_to_migrate(
        self,
        profile: AGVLoadProfile,
        all_tasks: Dict[str, TaskSpec],
        threshold: LoadThreshold,
    ) -> Optional[TaskSpec]:
        my_tasks = [t for t in all_tasks.values() if t.assigned_agv_id == profile.agv_id]
        if not my_tasks:
            return None
        # 迁移能力要求最低的任务
        return min(my_tasks, key=lambda t: t.required_capability)


class EnergyAwareStrategy(BalanceStrategyBase):
    """能耗感知策略"""

    def select_target_agv(
        self,
        task: TaskSpec,
        profiles: Dict[str, AGVLoadProfile],
        threshold: LoadThreshold,
    ) -> Optional[str]:
        candidates = {
            agv_id: p for agv_id, p in profiles.items()
            if not threshold.is_overloaded(p)
            and p.battery_level > threshold.critical_battery
            and p.task_queue_depth < threshold.max_queue_depth
        }
        if not candidates:
            return None
        # 优先选择能耗压力低、且接近任务目标点的AGV
        best = None
        best_score = -float("inf")
        for agv_id, profile in candidates.items():
            energy_score = 1.0 - profile.compute_energy_stress()
            if task.to_position:
                dist = profile.get_distance_to(*task.to_position[:2])
                dist_score = max(0.0, 1.0 - dist / 100.0)
            else:
                dist_score = 0.5
            score = energy_score * 0.6 + dist_score * 0.4
            if score > best_score:
                best_score = score
                best = agv_id
        return best

    def select_task_to_migrate(
        self,
        profile: AGVLoadProfile,
        all_tasks: Dict[str, TaskSpec],
        threshold: LoadThreshold,
    ) -> Optional[TaskSpec]:
        my_tasks = [t for t in all_tasks.values() if t.assigned_agv_id == profile.agv_id]
        if not my_tasks:
            return None
        # 优先迁移 deadline 宽松的任务
        now = time.time()
        return max(my_tasks, key=lambda t: (t.deadline or float("inf")) - now)


class HybridStrategy(BalanceStrategyBase):
    """混合策略 (综合多种因素)"""

    def __init__(self):
        self.capability = CapabilityAwareStrategy()
        self.energy = EnergyAwareStrategy()
        self.least = LeastLoadedStrategy()

    def select_target_agv(
        self,
        task: TaskSpec,
        profiles: Dict[str, AGVLoadProfile],
        threshold: LoadThreshold,
    ) -> Optional[str]:
        # 优先用能力感知
        result = self.capability.select_target_agv(task, profiles, threshold)
        if result:
            return result
        # 其次用能耗感知
        result = self.energy.select_target_agv(task, profiles, threshold)
        if result:
            return result
        # 最后用最小负载
        return self.least.select_target_agv(task, profiles, threshold)

    def select_task_to_migrate(
        self,
        profile: AGVLoadProfile,
        all_tasks: Dict[str, TaskSpec],
        threshold: LoadThreshold,
    ) -> Optional[TaskSpec]:
        # 如果是电池临界，优先用能耗策略
        if profile.battery_level <= threshold.warning_battery:
            return self.energy.select_task_to_migrate(profile, all_tasks, threshold)
        return self.capability.select_task_to_migrate(profile, all_tasks, threshold)


# ---------------------------------------------------------------------------
# DynamicLoadBalancer
# ---------------------------------------------------------------------------

class DynamicLoadBalancer:
    """多机器人动态负载均衡器"""

    def __init__(
        self,
        strategy: BalanceStrategy = BalanceStrategy.HYBRID,
        threshold: Optional[LoadThreshold] = None,
        rebalance_interval_s: float = 10.0,
    ):
        self.strategy = strategy
        self.threshold = threshold or LoadThreshold()
        self.rebalance_interval_s = rebalance_interval_s

        self._profiles: Dict[str, AGVLoadProfile] = {}
        self._tasks: Dict[str, TaskSpec] = {}
        self._rebalance_history: List[RebalanceDecision] = []
        self._decision_counter = 0
        self._last_rebalance_time = time.time()
        self._total_rebalances = 0
        self._completed_tasks = 0

        self._strategy_impl = self._create_strategy(strategy)

    def _create_strategy(self, strat: BalanceStrategy) -> BalanceStrategyBase:
        mapping = {
            BalanceStrategy.ROUND_ROBIN: RoundRobinStrategy(),
            BalanceStrategy.LEAST_LOADED: LeastLoadedStrategy(),
            BalanceStrategy.CAPABILITY_AWARE: CapabilityAwareStrategy(),
            BalanceStrategy.ENERGY_AWARE: EnergyAwareStrategy(),
            BalanceStrategy.HYBRID: HybridStrategy(),
        }
        return mapping.get(strat, HybridStrategy())

    def register_agv(self, agv_id: str, profile: AGVLoadProfile) -> None:
        """注册AGV"""
        profile._composite_load = None  # 重置缓存
        profile._energy_stress = 0.0
        self._profiles[agv_id] = profile

    def unregister_agv(self, agv_id: str) -> None:
        """注销AGV"""
        if agv_id in self._profiles:
            del self._profiles[agv_id]
        # 重新分配其任务
        orphaned = [t for t in self._tasks.values() if t.assigned_agv_id == agv_id]
        for task in orphaned:
            self._reassign_task(task)

    def update_profile(self, agv_id: str, **kwargs) -> bool:
        """更新AGV负载画像"""
        profile = self._profiles.get(agv_id)
        if not profile:
            return False
        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        # 刷新计算属性
        profile._composite_load = None
        profile._energy_stress = 0.0
        profile.compute_composite_load()
        return True

    def submit_task(self, task: TaskSpec) -> Optional[str]:
        """提交任务, 返回分配的AGV ID"""
        self._tasks[task.task_id] = task
        assigned = self._assign_task(task)
        if assigned:
            task.assigned_agv_id = assigned
        return assigned

    def _assign_task(self, task: TaskSpec) -> Optional[str]:
        """为任务分配最优AGV"""
        if not self._profiles:
            return None
        target = self._strategy_impl.select_target_agv(task, self._profiles, self.threshold)
        if target:
            profile = self._profiles[target]
            profile.task_queue_depth += 1
            profile.task_queue_weights.append(task.workload_units)
        return target

    def _reassign_task(self, task: TaskSpec) -> Optional[str]:
        """重新分配任务"""
        task.assigned_agv_id = None
        task.migrated_count += 1
        return self._assign_task(task)

    def complete_task(self, task_id: str) -> bool:
        """标记任务完成"""
        task = self._tasks.get(task_id)
        if not task:
            return False
        if task.assigned_agv_id:
            profile = self._profiles.get(task.assigned_agv_id)
            if profile:
                profile.task_queue_depth = max(0, profile.task_queue_depth - 1)
                if profile.task_queue_weights:
                    profile.task_queue_weights.pop(0)
        task.assigned_agv_id = None
        self._completed_tasks += 1
        # 检查是否需要重均衡
        self._check_trigger_rebalance(RebalanceTrigger.TASK_COMPLETION)
        return True

    def _check_trigger_rebalance(self, trigger: RebalanceTrigger) -> None:
        """检查是否触发重均衡"""
        now = time.time()
        if trigger == RebalanceTrigger.TIME_INTERVAL:
            if now - self._last_rebalance_time < self.rebalance_interval_s:
                return
        elif trigger == RebalanceTrigger.THRESHOLD_EXCEEDED:
            has_overloaded = any(self.threshold.is_overloaded(p) for p in self._profiles.values())
            if not has_overloaded:
                return
        elif trigger == RebalanceTrigger.BATTERY_CRITICAL:
            has_critical = any(
                p.battery_level <= self.threshold.critical_battery
                for p in self._profiles.values()
            )
            if not has_critical:
                return
        elif trigger == RebalanceTrigger.TASK_COMPLETION:
            has_overloaded = any(self.threshold.is_overloaded(p) for p in self._profiles.values())
            if not has_overloaded:
                return

        self._do_rebalance(trigger)

    def _do_rebalance(self, trigger: RebalanceTrigger) -> List[RebalanceDecision]:
        """执行重均衡"""
        decisions = []
        self._last_rebalance_time = time.time()

        # 找出过载AGV
        overloaded = {
            agv_id: p for agv_id, p in self._profiles.items()
            if self.threshold.is_overloaded(p)
        }

        # 找出空闲/轻载AGV
        idle = {
            agv_id: p for agv_id, p in self._profiles.items()
            if not self.threshold.is_overloaded(p)
            and p.task_queue_depth < self.threshold.max_queue_depth
            and p.battery_level > self.threshold.critical_battery
        }

        for from_id, from_profile in overloaded.items():
            if not idle:
                break
            # 找出该AGV上最应该迁移的任务
            task_to_migrate = self._strategy_impl.select_task_to_migrate(
                from_profile, self._tasks, self.threshold
            )
            if not task_to_migrate:
                continue

            # 找最优目标AGV
            target_id = self._strategy_impl.select_target_agv(
                task_to_migrate, idle, self.threshold
            )
            if not target_id:
                continue

            # 执行迁移
            decision = self._execute_migration(
                from_id, target_id, task_to_migrate, trigger
            )
            if decision:
                decisions.append(decision)

        return decisions

    def _execute_migration(
        self,
        from_agv_id: str,
        to_agv_id: str,
        task: TaskSpec,
        trigger: RebalanceTrigger,
    ) -> Optional[RebalanceDecision]:
        """执行任务迁移"""
        self._decision_counter += 1
        decision_id = f"REB{self._decision_counter:08d}"

        # 更新源AGV
        from_profile = self._profiles[from_agv_id]
        from_profile.task_queue_depth = max(0, from_profile.task_queue_depth - 1)
        if from_profile.task_queue_weights:
            from_profile.task_queue_weights.pop(0)

        # 更新目标AGV
        to_profile = self._profiles[to_agv_id]
        to_profile.task_queue_depth += 1
        to_profile.task_queue_weights.append(task.workload_units)

        # 更新任务
        task.assigned_agv_id = to_agv_id
        task.migrated_count += 1

        benefit = (
            (to_profile.composite_load - from_profile.composite_load) * 0.5
            + task.priority * 0.1
        )

        decision = RebalanceDecision(
            decision_id=decision_id,
            trigger=trigger,
            from_agv_id=from_agv_id,
            to_agv_id=to_agv_id,
            task_id=task.task_id,
            reason=f"Migrate from overloaded AGV (load={from_profile.composite_load:.2f}) to lighter AGV (load={to_profile.composite_load:.2f})",
            estimated_benefit=benefit,
        )
        self._rebalance_history.append(decision)
        self._total_rebalances += 1

        return decision

    def trigger_manual_rebalance(self) -> List[RebalanceDecision]:
        """手动触发重均衡"""
        return self._do_rebalance(RebalanceTrigger.MANUAL)

    def tick(self, dt: float = 1.0) -> Dict[str, Any]:
        """定时执行负载均衡检查"""
        # 定时触发
        self._check_trigger_rebalance(RebalanceTrigger.TIME_INTERVAL)

        # 检查电池临界
        self._check_trigger_rebalance(RebalanceTrigger.BATTERY_CRITICAL)

        return self.get_stats()

    def get_stats(self) -> LoadStats:
        """获取负载统计"""
        loads = [p.composite_load for p in self._profiles.values()]
        queues = [p.task_queue_depth for p in self._profiles.values()]
        overloaded = sum(1 for p in self._profiles.values() if self.threshold.is_overloaded(p))
        idle = sum(1 for p in self._profiles.values() if p.composite_load < self.threshold.underload_threshold)
        critical = sum(1 for p in self._profiles.values() if self.threshold.is_critical(p))

        return LoadStats(
            total_agvs=len(self._profiles),
            overloaded_agvs=overloaded,
            idle_agvs=idle,
            critical_agvs=critical,
            avg_load=np.mean(loads) if loads else 0.0,
            load_variance=np.var(loads) if loads else 0.0,
            total_tasks=len(self._tasks),
            pending_tasks=sum(1 for t in self._tasks.values() if t.assigned_agv_id is not None),
            completed_tasks=self._completed_tasks,
            rebalance_events=self._total_rebalances,
            avg_queue_depth=np.mean(queues) if queues else 0.0,
        )

    def get_load_distribution(self) -> Dict[str, float]:
        """获取所有AGV的负载分布"""
        return {agv_id: p.composite_load for agv_id, p in self._profiles.items()}

    def get_rebalance_history(self, limit: int = 50) -> List[Dict]:
        """获取重均衡历史"""
        history = self._rebalance_history[-limit:]
        return [
            {
                "decision_id": d.decision_id,
                "from": d.from_agv_id,
                "to": d.to_agv_id,
                "task": d.task_id,
                "benefit": d.estimated_benefit,
                "timestamp": d.timestamp,
            }
            for d in history
        ]

    def get_stress_heatmap(self) -> Dict[str, Dict[str, float]]:
        """获取各维度压力热力图"""
        result = {}
        for agv_id, profile in self._profiles.items():
            result[agv_id] = {
                "cpu": profile.cpu_usage,
                "memory": profile.memory_usage,
                "queue": min(profile.task_queue_depth / 20.0, 1.0),
                "battery_stress": 1.0 - profile.battery_level,
                "thermal": profile.thermal_level,
                "composite": profile.composite_load,
            }
        return result
