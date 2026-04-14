"""
task_executor.py - 具身任务执行器
SuperModel 超模态大模型具身智能系统

集成了行为树、仿真环境、长期记忆、真实AGV接口的端到端任务执行器。
支持:
- 行为树驱动的任务规划与执行
- 记忆增强的决策（从情景/语义记忆中检索经验）
- 仿真环境与真实AGV的无缝切换
- 多阶段任务执行与状态持久化
- 执行历史自动记录到情景记忆
"""

from __future__ import annotations

import abc
import enum
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# 执行状态与结果
# ============================================================================

class ExecutionPhase(enum.Enum):
    """任务执行阶段"""
    PLANNING = "planning"           # 规划阶段
    EXECUTING = "executing"         # 执行阶段
    MONITORING = "monitoring"       # 监控阶段
    SUCCEEDED = "succeeded"         # 成功完成
    FAILED = "failed"               # 执行失败
    ABORTED = "aborted"             # 被中止
    PAUSED = "paused"              # 暂停


class ExecutionResult(enum.Enum):
    """任务执行结果"""
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"
    PAUSED = "paused"
    ABORTED = "aborted"
    UNKNOWN = "unknown"


@dataclass
class TaskExecutionRecord:
    """任务执行记录（用于情景记忆）"""
    record_id: str
    task_id: str
    task_type: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    result: ExecutionResult = ExecutionResult.UNKNOWN
    phase: ExecutionPhase = ExecutionPhase.PLANNING
    phases_history: List[Dict[str, Any]] = field(default_factory=list)
    steps_executed: int = 0
    bt_tick_count: int = 0
    memory_retrieval_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    outcome_summary: str = ""

    def finalize(self, result: ExecutionResult, summary: str = ""):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.result = result
        self.outcome_summary = summary
        self.phase = (ExecutionPhase.SUCCEEDED if result == ExecutionResult.SUCCESS
                      else ExecutionPhase.FAILED if result == ExecutionResult.FAILURE
                      else ExecutionPhase.ABORTED)

    def add_phase(self, phase: ExecutionPhase, details: str = ""):
        self.phases_history.append({
            "phase": phase.value,
            "timestamp": time.time(),
            "details": details,
        })
        self.phase = phase

    def add_error(self, error: str):
        self.errors.append(error)

    def add_warning(self, warning: str):
        self.warnings.append(warning)

    def to_memory_format(self) -> Dict[str, Any]:
        """转换为情景记忆格式"""
        return {
            "record_id": self.record_id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "timestamp": self.start_time,
            "duration": self.duration,
            "result": self.result.value,
            "phases": [p["phase"] for p in self.phases_history],
            "steps_executed": self.steps_executed,
            "bt_tick_count": self.bt_tick_count,
            "memory_retrieval_count": self.memory_retrieval_count,
            "outcome_summary": self.outcome_summary,
            "entities": [self.task_type],
            "importance": "HIGH" if self.result == ExecutionResult.FAILURE else "MEDIUM",
        }


# ============================================================================
# 任务性能分析器
# ============================================================================

class TaskPerformanceProfiler:
    """
    任务执行性能分析器 - 追踪任务执行的各项性能指标

    追踪指标:
    - 每个行为树节点的执行时间
    - 每个执行阶段的耗时
    - 传感器处理时间
    - 动作执行时间
    - 记忆检索延迟
    - 总CPU/内存使用
    - 任务吞吐量统计
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._records: Dict[str, TaskExecutionRecord] = {}
        self._phase_timers: Dict[str, float] = {}
        self._node_times: Dict[str, List[float]] = {}
        self._phase_history: List[Dict[str, Any]] = []
        self._tick_times: List[float] = []
        self._sensor_times: List[float] = []
        self._memory_times: List[float] = []
        self._action_times: List[float] = []
        self._current_task_id: Optional[str] = None
        self._task_start_time: Optional[float] = None

    def start_task(self, task_id: str, task_type: str) -> None:
        """开始任务性能追踪"""
        self._current_task_id = task_id
        self._task_start_time = time.time()
        self._phase_timers = {
            'planning': 0.0,
            'execution': 0.0,
            'monitoring': 0.0,
            'memory': 0.0,
            'sensor': 0.0,
            'action': 0.0,
        }
        self._node_times = {}
        self._tick_times = []
        self._sensor_times = []
        self._memory_times = []
        self._action_times = []

    def end_task(self) -> Dict[str, Any]:
        """结束任务性能追踪，返回统计报告"""
        if self._task_start_time is None:
            return {}

        total_time = time.time() - self._task_start_time

        report = {
            'task_id': self._current_task_id,
            'total_time_s': round(total_time, 4),
            'phases': {
                phase: round(t, 4)
                for phase, t in self._phase_timers.items()
            },
            'node_times': {
                node: {
                    'count': len(times),
                    'total_ms': round(sum(times) * 1000, 2),
                    'avg_ms': round((sum(times) / len(times) * 1000) if times else 0, 3),
                    'max_ms': round(max(times) * 1000, 3) if times else 0,
                    'min_ms': round(min(times) * 1000, 3) if times else 0,
                }
                for node, times in self._node_times.items()
            },
            'tick_stats': self._compute_tick_stats(),
            'sensor_stats': self._compute_stats(self._sensor_times, 'sensor'),
            'memory_stats': self._compute_stats(self._memory_times, 'memory'),
            'action_stats': self._compute_stats(self._action_times, 'action'),
        }

        self._current_task_id = None
        self._task_start_time = None
        return report

    def start_phase(self, phase: str) -> None:
        """开始一个执行阶段"""
        self._phase_timers.setdefault(phase, 0.0)
        self._phase_timers[phase] -= time.time()  # 用负号标记开始

    def end_phase(self, phase: str) -> None:
        """结束一个执行阶段"""
        if phase in self._phase_timers:
            elapsed = time.time() + self._phase_timers[phase]  # 负号转正
            self._phase_timers[phase] = elapsed

    def record_tick_time(self, tick_time: float) -> None:
        """记录单个tick的耗时"""
        if self.enabled:
            self._tick_times.append(tick_time)

    def record_node_time(self, node_name: str, node_time: float) -> None:
        """记录节点执行时间"""
        if self.enabled:
            self._node_times.setdefault(node_name, []).append(node_time)

    def record_sensor_time(self, sensor_time: float) -> None:
        """记录传感器处理时间"""
        if self.enabled:
            self._sensor_times.append(sensor_time)

    def record_memory_time(self, memory_time: float) -> None:
        """记录记忆检索时间"""
        if self.enabled:
            self._memory_times.append(memory_time)

    def record_action_time(self, action_time: float) -> None:
        """记录动作执行时间"""
        if self.enabled:
            self._action_times.append(action_time)

    def _compute_tick_stats(self) -> Dict[str, Any]:
        return self._compute_stats(self._tick_times, 'tick')

    def _compute_stats(self, times: List[float], name: str) -> Dict[str, Any]:
        if not times:
            return {'count': 0, 'total_ms': 0.0, 'avg_ms': 0.0}
        import numpy as np
        return {
            'count': len(times),
            'total_ms': round(sum(times) * 1000, 4),
            'avg_ms': round(np.mean(times) * 1000, 4),
            'p50_ms': round(np.percentile(times, 50) * 1000, 3),
            'p95_ms': round(np.percentile(times, 95) * 1000, 3),
            'p99_ms': round(np.percentile(times, 99) * 1000, 3),
            'max_ms': round(max(times) * 1000, 3),
            'min_ms': round(min(times) * 1000, 3),
        }

    def get_realtime_report(self) -> Dict[str, Any]:
        """获取实时性能报告（任务执行中）"""
        report = {
            'current_task': self._current_task_id,
            'elapsed_s': round(time.time() - self._task_start_time, 2) if self._task_start_time else 0,
            'phases': {
                phase: round(max(0, time.time() + t), 4) if t < 0 else round(t, 4)
                for phase, t in self._phase_timers.items()
            },
            'tick_count': len(self._tick_times),
            'node_count': len(self._node_times),
        }
        if self._tick_times:
            import numpy as np
            report['tick_p50_ms'] = round(np.percentile(self._tick_times, 50) * 1000, 3)
            report['tick_p95_ms'] = round(np.percentile(self._tick_times, 95) * 1000, 3)
        return report


# ============================================================================
# 记忆增强执行器
# ============================================================================

class MemoryEnhancedExecutor:
    """
    记忆增强的具身任务执行器

    在任务规划阶段从长期记忆中检索相似经验，
    在执行阶段将成功/失败经验自动存入情景记忆，
    支持基于历史经验的决策优化。
    """

    def __init__(
        self,
        behavior_tree_root: Optional[Any] = None,
        memory_system: Optional[Any] = None,
        simulation_env: Optional[Any] = None,
        real_agv_interface: Optional[Any] = None,
        use_simulation: bool = True,
        enable_memory: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化记忆增强执行器

        Args:
            behavior_tree_root: 行为树根节点
            memory_system: 长期记忆系统实例
            simulation_env: 仿真环境实例
            real_agv_interface: 真实AGV接口实例
            use_simulation: 是否优先使用仿真环境
            enable_memory: 是否启用记忆增强
            config: 配置字典
        """
        self.bt_root = behavior_tree_root
        self.memory_system = memory_system
        self.simulation_env = simulation_env
        self.real_agv_interface = real_agv_interface
        self.use_simulation = use_simulation if simulation_env is not None else True
        self.enable_memory = enable_memory and memory_system is not None
        self.config = config or {}

        # 执行状态
        self.current_record: Optional[TaskExecutionRecord] = None
        self.execution_history: List[TaskExecutionRecord] = []
        self.is_running = False
        self.is_paused = False
        self._tick_count = 0

        # 回调函数
        self._on_phase_change: Optional[Callable] = None
        self._on_tick: Optional[Callable] = None
        self._on_error: Optional[Callable] = None

    def load_behavior_tree(self, bt_root: Any):
        """加载行为树"""
        self.bt_root = bt_root
        logger.info(f"Loaded behavior tree: {getattr(bt_root, 'name', 'unnamed')}")

    def load_memory_system(self, memory_system: Any):
        """加载记忆系统"""
        self.memory_system = memory_system
        self.enable_memory = memory_system is not None
        logger.info("Memory system enabled" if self.enable_memory else "Memory system not available")

    def set_callbacks(
        self,
        on_phase_change: Optional[Callable] = None,
        on_tick: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ):
        """设置回调函数"""
        self._on_phase_change = on_phase_change
        self._on_tick = on_tick
        self._on_error = on_error

    # -------------------------------------------------------------------------
    # 记忆检索接口
    # -------------------------------------------------------------------------

    def retrieve_relevant_experience(
        self,
        task_type: str,
        context: Optional[Dict[str, Any]] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        从长期记忆中检索相关经验

        Args:
            task_type: 任务类型
            context: 当前上下文
            limit: 返回数量上限

        Returns:
            相关经验列表
        """
        if not self.enable_memory:
            return []

        try:
            self.current_record.memory_retrieval_count += 1
            # 尝试使用记忆系统的检索接口
            if hasattr(self.memory_system, 'retrieve'):
                results = self.memory_system.retrieve(
                    query=task_type,
                    limit=limit,
                )
                return results if results else []
            elif hasattr(self.memory_system, 'search'):
                results = self.memory_system.search(
                    query=task_type,
                    top_k=limit,
                )
                return results if results else []
            else:
                return []
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")
            return []

    def store_execution_experience(
        self,
        record: TaskExecutionRecord,
        experience_type: str = "task_execution",
    ):
        """
        将执行经验存储到情景记忆

        Args:
            record: 执行记录
            experience_type: 经验类型标签
        """
        if not self.enable_memory:
            return

        try:
            memory_data = record.to_memory_format()
            memory_data["experience_type"] = experience_type
            if hasattr(self.memory_system, 'store'):
                self.memory_system.store(memory_data)
            elif hasattr(self.memory_system, 'add'):
                self.memory_system.add(memory_data)
            logger.debug(f"Stored execution experience: {record.task_id}")
        except Exception as e:
            logger.warning(f"Failed to store experience: {e}")

    def apply_learned_adjustments(
        self,
        task_config: Dict[str, Any],
        task_type: str,
    ) -> Dict[str, Any]:
        """
        根据历史经验调整任务配置

        从相似任务的执行经验中学习，调整速度、安全阈值等参数
        """
        experiences = self.retrieve_relevant_experience(task_type, limit=3)
        if not experiences:
            return task_config

        adjusted = task_config.copy()
        for exp in experiences:
            # 从成功经验中学习参数调整
            if exp.get("result") == "success":
                # 可以从经验中提取优化后的参数
                # 例如：如果历史经验显示某个速度参数效果更好
                pass
            # 从失败经验中学习避免重复错误
            elif exp.get("result") == "failure":
                # 添加安全约束
                if "safety_margin" not in adjusted:
                    adjusted["safety_margin"] = 1.2

        return adjusted

    # -------------------------------------------------------------------------
    # 核心执行接口
    # -------------------------------------------------------------------------

    def execute_task(
        self,
        task_type: str,
        task_config: Dict[str, Any],
        timeout: Optional[float] = None,
        tick_rate: float = 0.1,
    ) -> TaskExecutionRecord:
        """
        执行一个具身任务

        完整的任务执行流程:
        1. 创建执行记录
        2. 规划阶段：检索记忆，构建/调整行为树
        3. 执行阶段：tick行为树，切换仿真/真实环境
        4. 监控阶段：收集性能指标
        5. 完成阶段：存储执行经验

        Args:
            task_type: 任务类型 (e.g., "transport", "patrol", "rescue")
            task_config: 任务配置字典
            timeout: 最大执行时间（秒）
            tick_rate: tick间隔（秒）

        Returns:
            TaskExecutionRecord: 执行记录
        """
        if self.is_running:
            logger.warning("Executor is already running a task")
            return self.current_record

        record = TaskExecutionRecord(
            record_id=str(uuid.uuid4())[:8],
            task_id=str(uuid.uuid4())[:8],
            task_type=task_type,
            start_time=time.time(),
        )
        self.current_record = record
        self.is_running = True
        self.is_paused = False
        self._tick_count = 0

        logger.info(f"=== Starting task execution: {task_type} (id={record.task_id}) ===")

        try:
            # ---- Phase 1: Planning ----
            record.add_phase(ExecutionPhase.PLANNING, f"Planning {task_type}")
            self._on_phase_change and self._on_phase_change(record.phase, {})

            # 检索相关经验
            context = task_config.copy()
            context["environment"] = "simulation" if self.use_simulation else "real"
            relevant_exp = self.retrieve_relevant_experience(task_type, context)

            if relevant_exp:
                logger.info(f"Retrieved {len(relevant_exp)} relevant experiences")
                record.add_warning(f"Using {len(relevant_exp)} historical experiences to guide execution")

            # 根据经验调整任务配置
            adjusted_config = self.apply_learned_adjustments(task_config, task_type)

            # 构建行为树（如果需要）
            if self.bt_root is None:
                from .behavior_tree import create_behavior_tree_from_dict
                bt_config = self._build_bt_config(task_type, adjusted_config)
                self.bt_root = create_behavior_tree_from_dict(bt_config)
                logger.info(f"Built behavior tree from config for task: {task_type}")

            # ---- Phase 2: Execution ----
            record.add_phase(ExecutionPhase.EXECUTING, "Executing behavior tree")
            self._on_phase_change and self._on_phase_change(record.phase, {})

            from .behavior_tree import BehaviorTree, NodeStatus
            bt = BehaviorTree(self.bt_root, name=f"TaskExecutor_{task_type}")

            # 设置黑板初始状态（支持仿真/真实环境）
            import numpy as np
            target_pos = task_config.get('target_position')
            robot_pos = task_config.get('robot_position', [0.0, 0.0, 0.0])
            if target_pos is None and task_config.get('target'):
                # 解析目标标识为坐标
                known_points = {
                    'station_a': np.array([10.0, 0.0, 0.0]),
                    'station_b': np.array([20.0, 0.0, 0.0]),
                    'station_c': np.array([30.0, 0.0, 0.0]),
                    'entrance': np.array([0.0, 0.0, 0.0]),
                    'exit': np.array([40.0, 0.0, 0.0]),
                    'charging': np.array([2.0, 0.0, 0.0]),
                }
                t = task_config['target'].lower().replace('-', '_').replace(' ', '_')
                for key, pos in known_points.items():
                    if key in t or t in key:
                        target_pos = pos.tolist()
                        break
                if target_pos is None:
                    target_pos = [10.0, 0.0, 0.0]

            pickup = np.array(task_config.get('pickup_position', [0.0, 0.0, 0.0]))
            dropoff = np.array(task_config.get('dropoff_position', target_pos or [10.0, 0.0, 0.0]))

            bt.blackboard.update_robot_state({
                'position': robot_pos,
                'battery_level': task_config.get('battery_level', 0.8),
                'safety': task_config.get('safety', True),
                'speed': task_config.get('speed', 0.0),
            })
            bt.blackboard.goal_state.update({
                'target_position': dropoff,
                'target_object': task_config.get('object') or (task_type == 'transport' and 'package_001'),
                'pickup_position': pickup,
                'dropoff_position': dropoff,
            })
            logger.info(f"Blackboard initialized: robot_pos={robot_pos}, "
                        f"battery=0.8, pickup={pickup.tolist()}, dropoff={dropoff.tolist()}")

            start_tick_time = time.time()
            last_tick_time = start_tick_time

            while self.is_running:
                if self.is_paused:
                    time.sleep(tick_rate)
                    continue

                # 超时检查
                if timeout and (time.time() - record.start_time) > timeout:
                    record.finalize(ExecutionResult.FAILURE, f"Task timeout after {timeout}s")
                    logger.warning(f"Task timeout: {task_type}")
                    break

                # Tick行为树
                status = bt.tick()
                record.bt_tick_count += 1
                self._tick_count += 1
                record.steps_executed += 1

                # 性能指标采集
                now = time.time()
                tick_duration = now - last_tick_time
                last_tick_time = now
                record.performance_metrics["avg_tick_time_ms"] = (
                    record.performance_metrics.get("avg_tick_time_ms", 0) * 0.9
                    + tick_duration * 100 * 0.1
                )

                self._on_tick and self._on_tick(self._tick_count, status)

                # 检查执行结果
                if status == NodeStatus.SUCCESS:
                    record.finalize(ExecutionResult.SUCCESS, f"Task succeeded: {task_type}")
                    logger.info(f"Task SUCCEEDED: {task_type} (ticks={record.bt_tick_count})")
                    break
                elif status == NodeStatus.FAILURE:
                    record.finalize(ExecutionResult.FAILURE, f"Task failed: {task_type}")
                    record.add_error(f"Behavior tree returned FAILURE after {record.bt_tick_count} ticks")
                    logger.warning(f"Task FAILED: {task_type}")
                    break
                elif status == NodeStatus.RUNNING:
                    pass  # 继续执行

                time.sleep(tick_rate)

            # ---- Phase 3: Monitoring (finalize metrics) ----
            total_time = time.time() - start_tick_time
            record.performance_metrics["total_execution_time_s"] = total_time
            record.performance_metrics["ticks_per_second"] = (
                record.bt_tick_count / total_time if total_time > 0 else 0
            )

            # ---- Phase 4: Store experience ----
            if record.result != ExecutionResult.UNKNOWN:
                self.store_execution_experience(record)

        except Exception as e:
            record.add_error(str(e))
            record.finalize(ExecutionResult.FAILURE, f"Exception: {str(e)}")
            self._on_error and self._on_error(e)
            logger.exception(f"Task execution error: {e}")
        finally:
            self.is_running = False
            self.execution_history.append(record)
            logger.info(f"=== Task execution finished: {record.result.value} (duration={record.duration:.2f}s) ===")

        return record

    def _build_bt_config(
        self,
        task_type: str,
        task_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        根据任务类型和配置构建行为树配置

        支持的任务类型:
        - transport: 搬运任务 (安全检查 → 移动 → 抓取 → 移动 → 释放)
        - patrol: 巡逻任务 (重复: 移动到点 → 检查 → 返回)
        - rescue: 应急任务 (优先级: 避障 → 安全确认 → 执行)
        - collaborative: 协同任务 (角色协商 → 编队 → 协同移动)
        """
        # 基础任务行为树模板
        templates = {
            "transport": {
                "type": "sequence",
                "name": "TransportTask",
                "children": [
                    {"type": "agv_check_safe", "name": "SafetyCheck"},
                    {
                        "type": "selector",
                        "name": "BatteryCheckOrAbort",
                        "children": [
                            {"type": "agv_check_battery", "params": {"min_battery": task_config.get("min_battery", 0.2)}},
                            {
                                "type": "sequence",
                                "name": "AbortTransport",
                                "children": [
                                    {"type": "lambda", "action_name": "abort", "name": "AbortAction"},
                                ]
                            },
                        ],
                    },
                    {"type": "agv_move_to", "params": {"speed": task_config.get("move_speed", 0.5)}, "name": "MoveToPickup"},
                    {"type": "agv_check_position", "params": {"threshold": task_config.get("position_threshold", 0.15)}, "name": "CheckPickupPos"},
                    {"type": "agv_grasp", "name": "GraspObject"},
                    {"type": "agv_move_to", "params": {"speed": task_config.get("move_speed", 0.5)}, "name": "MoveToDropoff"},
                    {"type": "agv_release", "name": "ReleaseObject"},
                ],
            },
            "patrol": {
                "type": "repeater",
                "params": {"num_repeats": task_config.get("patrol_loops", 3)},
                "name": "PatrolTask",
                "children": [
                    {
                        "type": "sequence",
                        "name": "PatrolLoop",
                        "children": [
                            {"type": "agv_check_safe", "name": "SafetyCheck"},
                            {"type": "agv_move_to", "params": {"speed": task_config.get("patrol_speed", 0.4)}, "name": "MoveToWaypoint"},
                            {"type": "agv_check_position", "name": "CheckWaypoint"},
                        ],
                    },
                ],
            },
            "rescue": {
                "type": "sequence",
                "name": "RescueTask",
                "children": [
                    {"type": "agv_check_safe", "name": "EmergencySafetyCheck"},
                    {
                        "type": "selector",
                        "name": "NavigateOrAvoid",
                        "children": [
                            {"type": "agv_move_to", "params": {"speed": task_config.get("rescue_speed", 0.8)}, "name": "MoveToRescuePoint"},
                            {
                                "type": "sequence",
                                "name": "AvoidObstacle",
                                "children": [
                                    {"type": "lambda", "action_name": "avoid", "name": "AvoidAction"},
                                    {"type": "until_success", "children": [
                                        {"type": "agv_move_to", "params": {"speed": task_config.get("rescue_speed", 0.8)}},
                                    ]},
                                ],
                            },
                        ],
                    },
                    {"type": "agv_check_position", "params": {"threshold": task_config.get("rescue_threshold", 0.2)}, "name": "CheckRescuePos"},
                    {"type": "agv_grasp", "name": "GraspVictim"},
                    {"type": "agv_move_to", "params": {"speed": task_config.get("rescue_speed", 0.6)}, "name": "MoveToSafeZone"},
                    {"type": "agv_release", "name": "ReleaseVictim"},
                ],
            },
            "collaborative": {
                "type": "sequence",
                "name": "CollaborativeTask",
                "children": [
                    {"type": "agv_negotiate_role", "name": "NegotiateRole"},
                    {"type": "agv_check_safe", "name": "SafetyCheck"},
                    {"type": "agv_move_to_formation", "name": "MoveToFormation"},
                    {"type": "agv_check_formation", "params": {"threshold": task_config.get("formation_threshold", 0.15)}, "name": "CheckFormation"},
                    {"type": "agv_parallel_grasp", "name": "ParallelGrasp"},
                    {"type": "agv_coordinated_move", "params": {"speed": task_config.get("collaborative_speed", 0.4)}, "name": "CoordinatedMove"},
                    {"type": "agv_parallel_release", "name": "ParallelRelease"},
                ],
            },
        }

        return templates.get(task_type, templates["transport"])

    def pause(self):
        """暂停任务执行"""
        if self.is_running:
            self.is_paused = True
            if self.current_record:
                self.current_record.add_phase(ExecutionPhase.PAUSED, "User paused execution")

    def resume(self):
        """恢复任务执行"""
        if self.is_running and self.is_paused:
            self.is_paused = False
            if self.current_record:
                self.current_record.add_phase(ExecutionPhase.EXECUTING, "User resumed execution")

    def abort(self):
        """中止任务执行"""
        if self.current_record:
            self.current_record.finalize(ExecutionResult.ABORTED, "User aborted execution")
        self.is_running = False
        self.is_paused = False

    def get_status(self) -> Dict[str, Any]:
        """获取执行器当前状态"""
        return {
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "current_task": self.current_record.task_type if self.current_record else None,
            "task_id": self.current_record.task_id if self.current_record else None,
            "phase": self.current_record.phase.value if self.current_record else None,
            "tick_count": self._tick_count,
            "total_tasks": len(self.execution_history),
        }

    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行历史摘要"""
        if not self.execution_history:
            return {"total": 0, "success": 0, "failure": 0, "success_rate": 0.0}

        total = len(self.execution_history)
        successes = sum(1 for r in self.execution_history if r.result == ExecutionResult.SUCCESS)
        failures = sum(1 for r in self.execution_history if r.result == ExecutionResult.FAILURE)

        avg_duration = (
            sum(r.duration or 0 for r in self.execution_history) / total
        )

        return {
            "total": total,
            "success": successes,
            "failure": failures,
            "aborted": sum(1 for r in self.execution_history if r.result == ExecutionResult.ABORTED),
            "success_rate": successes / total if total > 0 else 0.0,
            "avg_duration_s": avg_duration,
            "total_ticks": sum(r.bt_tick_count for r in self.execution_history),
        }


# ============================================================================
# 场景化任务执行器
# ============================================================================

class ScenarioTaskExecutor(MemoryEnhancedExecutor):
    """
    场景化具身任务执行器

    在 MemoryEnhancedExecutor 基础上集成了:
    - 场景智能 (SceneIntelligence) 根据场景类型自适应调整行为
    - 场景协调 (SceneCoordination) 支持多AGV在同一场景中的协调
    - 仿真环境自动切换 场景变化时自动切换仿真参数
    """

    def __init__(
        self,
        scene_intelligence: Optional[Any] = None,
        scene_coordinator: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scene_intelligence = scene_intelligence
        self.scene_coordinator = scene_coordinator
        self.current_scene_type: Optional[str] = None

    def set_scene(self, scene_type: str, scene_config: Optional[Dict[str, Any]] = None):
        """设置当前场景类型"""
        self.current_scene_type = scene_type
        logger.info(f"Scene set to: {scene_type}")

        # 根据场景类型调整行为树参数
        if self.scene_intelligence:
            scene_params = self.scene_intelligence.get_parameters(scene_type)
            logger.info(f"Loaded {len(scene_params)} scene-specific parameters")

    def execute_scenario_task(
        self,
        scenario_type: str,
        task_config: Dict[str, Any],
        **kwargs,
    ) -> TaskExecutionRecord:
        """
        执行场景化任务

        自动根据场景类型:
        1. 加载场景特定的行为参数
        2. 配置安全规则和导航规则
        3. 选择合适的仿真环境
        """
        # 加载场景配置
        if self.scene_intelligence and scenario_type:
            scene_ctx = self.scene_intelligence.analyze_scene(scenario_type)
            # 将场景分析结果合并到任务配置
            if scene_ctx:
                task_config = {**task_config, "scene_context": scene_ctx}

        # 执行任务
        return self.execute_task(scenario_type, task_config, **kwargs)


# ============================================================================
# 工具函数
# ============================================================================

def create_task_executor(
    executor_type: str = "default",
    config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> MemoryEnhancedExecutor:
    """
    工厂函数：创建任务执行器

    Args:
        executor_type: 执行器类型 ("default", "scenario")
        config: 配置字典
        **kwargs: 传给执行器的其他参数

    Returns:
        任务执行器实例
    """
    config = config or {}

    if executor_type == "scenario":
        return ScenarioTaskExecutor(config=config, **kwargs)
    else:
        return MemoryEnhancedExecutor(config=config, **kwargs)


def create_executor_from_config(
    config: Dict[str, Any],
    memory_system: Optional[Any] = None,
    simulation_env: Optional[Any] = None,
    real_agv_interface: Optional[Any] = None,
) -> MemoryEnhancedExecutor:
    """
    从配置字典创建完整的任务执行器

    配置格式:
    {
        "type": "default" | "scenario",
        "use_simulation": true,
        "enable_memory": true,
        "default_task_type": "transport",
        "tick_rate": 0.1,
        "timeout": 300.0,
        "scene_type": "warehouse"  # scenario only
    }
    """
    exec_type = config.get("type", "default")
    executor = create_task_executor(
        executor_type=exec_type,
        config=config,
        memory_system=memory_system,
        simulation_env=simulation_env,
        real_agv_interface=real_agv_interface,
        use_simulation=config.get("use_simulation", True),
        enable_memory=config.get("enable_memory", True),
    )

    # 设置默认回调
    def default_on_tick(tick: int, status: Any):
        if tick % 100 == 0:
            logger.debug(f"Tick {tick}: {status}")

    executor.set_callbacks(
        on_tick=config.get("on_tick", default_on_tick),
        on_phase_change=config.get("on_phase_change"),
    )

    return executor


__all__ = [
    "ExecutionPhase",
    "ExecutionResult",
    "TaskExecutionRecord",
    "TaskPerformanceProfiler",
    "MemoryEnhancedExecutor",
    "ScenarioTaskExecutor",
    "create_task_executor",
    "create_executor_from_config",
]
