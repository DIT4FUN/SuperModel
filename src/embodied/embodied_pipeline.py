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
embodied_pipeline.py - SuperModel 具身智能统一Pipeline
=======================================================

一体化具身智能执行管道，集成所有具身模块：
- 行为树任务规划
- 场景感知与自适应
- 技能生命周期管理
- 记忆系统集成
- 任务执行与监控
- 硬件接口抽象
- 五级AGV规格适配

快速使用:
    pipeline = EmbodiedPipeline(grade="M", scene=SceneType.WAREHOUSE)
    pipeline.start()
    result = pipeline.execute_task("pick_and_stow", target="station_A")
    pipeline.stop()
"""

from __future__ import annotations

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from enum import Enum, auto
from collections import deque
from collections.abc import Sequence as ABCSequence
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    'PipelineMode',
    'PipelineState',
    'PipelineConfig',
    'TaskRequest',
    'TaskResult',
    'EmbodiedPipeline',
    'create_embodied_pipeline',
    'create_pipeline_from_config',
    'DegradationManager',
    'DegradationLevel',
    'DegradedCapability',
    'ErrorRecoveryPolicy',
    'DiagnosticCollector',
]


# ============================================================
# Pipeline 枚举与配置
# ============================================================

class PipelineMode(Enum):
    """Pipeline 运行模式"""
    SIMULATION = "simulation"     # 纯仿真模式
    HARDWARE_IN_LOOP = "hil"      # 硬件在环模式
    FULL_PHYSICAL = "full_physical"  # 全实体模式


class PipelineState(Enum):
    """Pipeline 状态"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class PipelineConfig:
    """Pipeline 全局配置"""
    # AGV 等级
    grade: str = "M"

    # 运行模式
    mode: PipelineMode = PipelineMode.SIMULATION

    # 场景
    scene_type: str = "WAREHOUSE"

    # 模块开关
    enable_skill_registry: bool = True
    enable_memory: bool = True
    enable_scene_intelligence: bool = True
    enable_hil: bool = False
    enable_federated_learning: bool = False
    enable_swarm_coordination: bool = True

    # 传感器配置
    enable_vision: bool = True
    enable_audio: bool = False
    enable_tactile: bool = True
    enable_force: bool = True
    enable_imu: bool = True

    # 执行配置
    max_concurrent_tasks: int = 4
    task_timeout_s: float = 600.0
    health_check_interval_s: float = 5.0

    # 仿真配置
    simulation_timestep: float = 0.01
    enable_physics_simulation: bool = True

    # 联邦学习配置
    fl_num_clients: int = 4
    fl_local_epochs: int = 2
    fl_rounds: int = 10
    fl_aggregation: str = "fedavg"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'grade': self.grade,
            'mode': self.mode.value,
            'scene_type': self.scene_type,
            'enable_skill_registry': self.enable_skill_registry,
            'enable_memory': self.enable_memory,
            'enable_scene_intelligence': self.enable_scene_intelligence,
            'enable_hil': self.enable_hil,
            'enable_federated_learning': self.enable_federated_learning,
            'enable_swarm_coordination': self.enable_swarm_coordination,
        }


# ============================================================
# 任务请求与结果
# ============================================================

@dataclass
class TaskRequest:
    """任务请求"""
    task_type: str
    task_id: str = ""
    target: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 2
    deadline: Optional[float] = None
    parent_task_id: Optional[str] = None
    collaborative_agv_ids: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.task_id:
            import uuid
            self.task_id = str(uuid.uuid4())[:8]


@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: str
    success: bool
    phase: str
    duration_ms: float
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    skill_used: Optional[str] = None
    scene_type: Optional[str] = None


# ============================================================
# 主Pipeline 类
# ============================================================

class EmbodiedPipeline:
    """
    SuperModel 具身智能统一Pipeline

    集成以下模块的统一入口:
    - BehaviorTree: 任务行为树规划
    - SceneIntelligence: 场景感知与自适应
    - EmbodiedSkillRegistry: 技能生命周期管理
    - EmbodiedMemoryManager: 记忆系统集成
    - TaskExecutor: 任务执行器
    - RealAGVController: 真实AGV硬件接口 (可选)

    示例:
        pipeline = EmbodiedPipeline(grade="L", mode=PipelineMode.SIMULATION)
        pipeline.start()
        result = pipeline.execute_task("transport", target="station_B")
        assert result.success
        pipeline.stop()
    """

    def __init__(
        self,
        grade: str = "M",
        mode: PipelineMode = PipelineMode.SIMULATION,
        scene_type: str = "WAREHOUSE",
        config: Optional[PipelineConfig] = None,
    ):
        self.config = config or PipelineConfig(
            grade=grade,
            mode=mode,
            scene_type=scene_type.upper() if isinstance(scene_type, str) else scene_type,
        )
        self._state = PipelineState.IDLE
        self._lock = threading.Lock()
        self._task_queue: deque[TaskRequest] = deque()
        self._active_tasks: Dict[str, TaskRequest] = {}
        self._completed_tasks: deque[TaskResult] = deque(maxlen=100)
        self._subscribers: Dict[str, Callable] = {}
        self._start_time: Optional[float] = None
        self._error_message: Optional[str] = None

        # 延迟初始化各模块 (lazy init)
        self._bt_planner: Optional[Any] = None
        self._scene_intel: Optional[Any] = None
        self._skill_registry: Optional[Any] = None
        self._memory_mgr: Optional[Any] = None
        self._task_executor: Optional[Any] = None
        self._hil_runner: Optional[Any] = None
        self._sim_enhancer: Optional[Any] = None
        self._fl_coordinator: Optional[Any] = None
        self._swarm_coord: Optional[Any] = None

        # FL round tracking
        self._fl_round_count: int = 0
        self._fl_last_result: Optional[Any] = None

        # 优雅降级管理器
        self._degradation_manager: Optional[DegradationManager] = None

    # ----------------------------------------------------------
    # 状态管理
    # ----------------------------------------------------------

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state == PipelineState.RUNNING

    @property
    def uptime_s(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def _set_state(self, new_state: PipelineState) -> None:
        old = self._state
        self._state = new_state
        logger.info(f"Pipeline state: {old.value} -> {new_state.value}")
        self._notify("state_changed", {"old": old, "new": new_state})

    # ----------------------------------------------------------
    # 模块初始化
    # ----------------------------------------------------------

    def _init_behavior_tree(self) -> None:
        """初始化行为树规划器"""
        try:
            from .behavior_tree import AGVTaskPlanner
            self._bt_planner = AGVTaskPlanner(grade=self.config.grade)
            logger.info("BehaviorTree planner initialized")
        except Exception as e:
            logger.warning(f"BehaviorTree init failed: {e}, using fallback")
            self._bt_planner = None

    def _init_scene_intelligence(self) -> None:
        """初始化场景智能"""
        try:
            from .scene_intelligence import get_scene_intelligence
            self._scene_intel = get_scene_intelligence(self.config.scene_type)
            logger.info(f"SceneIntelligence initialized: {self.config.scene_type}")
        except Exception as e:
            logger.warning(f"SceneIntelligence init failed: {e}")
            self._scene_intel = None

    def _init_skill_registry(self) -> None:
        """初始化技能注册表"""
        if not self.config.enable_skill_registry:
            return
        try:
            from .embodied_skill import get_global_skill_registry
            self._skill_registry = get_global_skill_registry()
            # 注册当前场景适用的技能
            self._register_scene_skills()
            logger.info("SkillRegistry initialized")
        except Exception as e:
            logger.warning(f"SkillRegistry init failed: {e}")
            self._skill_registry = None

    def _init_memory(self) -> None:
        """初始化记忆系统"""
        if not self.config.enable_memory:
            return
        try:
            from .memory_integration import create_embodied_memory_manager
            self._memory_mgr = create_embodied_memory_manager(
                config={},
            )
            logger.info("MemoryManager initialized")
        except Exception as e:
            logger.warning(f"MemoryManager init failed: {e}")
            self._memory_mgr = None

    def _init_task_executor(self) -> None:
        """初始化任务执行器"""
        try:
            from .task_executor import create_task_executor
            self._task_executor = create_task_executor(
                config={},
                enable_memory=self.config.enable_memory,
            )
            logger.info("TaskExecutor initialized")
        except Exception as e:
            logger.warning(f"TaskExecutor init failed: {e}")
            self._task_executor = None

    def _init_hil(self) -> None:
        """初始化HIL测试框架"""
        if not self.config.enable_hil:
            return
        try:
            from .hil_testing import HILTestRunner
            self._hil_runner = HILTestRunner()
            logger.info("HIL runner initialized")
        except Exception as e:
            logger.warning(f"HIL init failed: {e}")
            self._hil_runner = None

    def _init_simulation(self) -> None:
        """初始化仿真增强"""
        if self.config.mode != PipelineMode.SIMULATION:
            return
        try:
            from .simulation_enhancement import EmbodiedSimulationEnhancer
            self._sim_enhancer = EmbodiedSimulationEnhancer(
                agv_grade=self.config.grade,
                enable_noise=True,
                enable_delay=True,
                enable_enhanced_collision=True,
            )
            logger.info("SimulationEnhancer initialized")
        except Exception as e:
            logger.warning(f"SimulationEnhancer init failed: {e}")
            self._sim_enhancer = None

    def _init_federated_learning(self) -> None:
        """初始化联邦学习协调器"""
        if not self.config.enable_federated_learning:
            return
        try:
            from .federated_learning import FederatedLearningCoordinator
            self._fl_coordinator = FederatedLearningCoordinator(
                model_config={
                    'input_dim': 128,
                    'hidden_dim': 64,
                    'output_dim': 32,
                    'grade': self.config.grade,
                },
                grade=self.config.grade,
            )
            logger.info(
                f"FederatedLearning initialized: {self.config.fl_num_clients} clients, "
                f"{self.config.fl_rounds} rounds"
            )
        except Exception as e:
            logger.warning(f"FederatedLearning init failed: {e}")
            self._fl_coordinator = None

    def _init_swarm_coordination(self) -> None:
        """初始化蜂群协调器"""
        if not self.config.enable_swarm_coordination:
            return
        try:
            from .agv_swarm_coordinator import AGVSwarmCoordinator

            # Build a minimal scene with required navigation graph
            class MinimalSwarmScene:
                """Minimal scene object providing required AGVSwarmCoordinator interface"""
                def __init__(self):
                    self.warehouse_id = "pipeline_default"
                    self.width = 50.0
                    self.length = 50.0
                    # Required by _build_global_map: navigation_points and path_segments
                    self.navigation_points = {
                        'entrance': np.array([0.0, 0.0, 0.0]),
                        'charging_station': np.array([2.0, 0.0, 0.0]),
                        'station_A': np.array([10.0, 0.0, 0.0]),
                        'station_B': np.array([20.0, 0.0, 0.0]),
                        'station_C': np.array([30.0, 0.0, 0.0]),
                        'exit': np.array([40.0, 0.0, 0.0]),
                    }
                    self.path_segments = {
                        ('entrance', 'charging_station'): 2.0,
                        ('charging_station', 'station_A'): 8.0,
                        ('station_A', 'station_B'): 10.0,
                        ('station_B', 'station_C'): 10.0,
                        ('station_C', 'exit'): 10.0,
                    }
                    self.resources = {
                        'charger_1': {'position': (2.0, 0.0, 0.0)},
                        'station_A': {'position': (10.0, 0.0, 0.0)},
                        'station_B': {'position': (20.0, 0.0, 0.0)},
                    }

            scene = MinimalSwarmScene()
            self._swarm_coord = AGVSwarmCoordinator(
                scene=scene,
                max_workers=self.config.fl_num_clients,
            )
            logger.info(
                f"SwarmCoordinator initialized: {self.config.fl_num_clients} workers"
            )
        except Exception as e:
            logger.warning(f"SwarmCoordinator init failed: {e}")
            self._swarm_coord = None

    def _register_scene_skills(self) -> None:
        """根据场景类型注册适用的技能"""
        if self._skill_registry is None:
            return
        scene_type = self.config.scene_type.upper()
        # 注册场景相关技能 (skill_registry.add_skill 已存在)
        try:
            # 尝试注册该场景的典型技能
            for skill_name in self._get_scene_skill_names():
                if not self._skill_registry.has_skill(skill_name):
                    self._skill_registry.register_skill(skill_name, skill_name)
            logger.info(f"Registered scene skills for {scene_type}")
        except Exception as e:
            logger.debug(f"Scene skill registration: {e}")

    def _get_scene_skill_names(self) -> List[str]:
        """获取场景对应的技能名称列表"""
        scene_skills = {
            "WAREHOUSE": ["navigate_freight", "grasp_item", "stow_shelf", "patrol_aisle"],
            "HOSPITAL": ["navigate_clean", "transport_medicine", "sterile_delivery", "emergency_evacuate"],
            "FACTORY": ["navigate_manufacturing", "assembly_handover", "quality_scan", "maintenance_check"],
            "RESTAURANT": ["navigate_dining", "serve_table", "collect_dish", "clean_surface"],
            "OUTDOOR": ["navigate_outdoor", "terrain_traverse", "weather_adapt", "delivery_door"],
        }
        return scene_skills.get(self.config.scene_type.upper(), ["navigate_standard"])

    def _initialize_all_modules(self) -> None:
        """按正确顺序初始化所有模块"""
        self._init_behavior_tree()
        self._init_scene_intelligence()
        self._init_skill_registry()
        self._init_memory()
        self._init_task_executor()
        self._init_hil()
        self._init_simulation()
        self._init_federated_learning()
        self._init_swarm_coordination()
        self._init_degradation_manager()

    # ----------------------------------------------------------
    # 降级管理
    # ----------------------------------------------------------

    def _init_degradation_manager(self) -> None:
        """初始化优雅降级管理器"""
        try:
            self._degradation_manager = DegradationManager(
                pipeline=self,
                auto_recover=True,
                recovery_interval_s=30.0,
            )
            logger.info("DegradationManager initialized")
        except Exception as e:
            logger.warning(f"DegradationManager init failed: {e}")
            self._degradation_manager = None

    def get_degradation_report(self) -> Dict[str, Any]:
        """
        获取降级状态报告

        Returns:
            包含当前降级等级、降级模块、激活的降级策略等
        """
        if self._degradation_manager is None:
            return {'error': 'DegradationManager not initialized'}
        return self._degradation_manager.get_degradation_report()

    def check_degradation(self) -> str:
        """
        检查并更新降级等级

        Returns:
            当前降级等级字符串
        """
        if self._degradation_manager is None:
            return "unknown"
        level = self._degradation_manager.check_and_update()
        return level.value

    def can_use_capability(self, capability: str) -> bool:
        """
        检查某项能力是否可用

        Args:
            capability: 能力名称 (DegradedCapability枚举值)

        Returns:
            True if available
        """
        if self._degradation_manager is None:
            return True  # 假设可用如果未初始化
        try:
            cap = DegradedCapability(capability)
            return self._degradation_manager.can_use_capability(cap)
        except ValueError:
            return True  # 未知能力假设可用

    # ----------------------------------------------------------
    # Pipeline 生命周期
    # ----------------------------------------------------------

    def start(self) -> bool:
        """
        启动 Pipeline

        初始化所有模块并将状态设为 READY

        Returns:
            True if startup successful
        """
        with self._lock:
            if self._state not in (PipelineState.IDLE, PipelineState.STOPPED):
                logger.warning(f"Cannot start from state {self._state.value}")
                return False

            self._set_state(PipelineState.INITIALIZING)
            self._error_message = None

            try:
                self._initialize_all_modules()
                self._start_time = time.time()
                self._set_state(PipelineState.READY)
                logger.info(
                    f"EmbodiedPipeline started: grade={self.config.grade}, "
                    f"mode={self.config.mode.value}, scene={self.config.scene_type}"
                )
                return True
            except Exception as e:
                self._error_message = str(e)
                self._set_state(PipelineState.ERROR)
                logger.error(f"Pipeline startup failed: {e}")
                return False

    def pause(self) -> bool:
        """暂停 Pipeline"""
        with self._lock:
            if self._state != PipelineState.RUNNING:
                return False
            self._set_state(PipelineState.PAUSED)
            return True

    def resume(self) -> bool:
        """恢复 Pipeline"""
        with self._lock:
            if self._state != PipelineState.PAUSED:
                return False
            self._set_state(PipelineState.RUNNING)
            return True

    def stop(self) -> None:
        """停止 Pipeline"""
        with self._lock:
            self._set_state(PipelineState.STOPPED)
            self._active_tasks.clear()
            self._task_queue.clear()
            self._start_time = None
            logger.info("EmbodiedPipeline stopped")

    # ----------------------------------------------------------
    # 任务执行
    # ----------------------------------------------------------

    def execute_task(
        self,
        task_type: str,
        target: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        priority: int = 2,
        timeout_s: Optional[float] = None,
    ) -> TaskResult:
        """
        执行单个具身任务

        Args:
            task_type: 任务类型 (如 "transport", "patrol", "rescue")
            target: 目标位置或对象
            payload: 附加参数
            priority: 优先级 (0=最高)
            timeout_s: 超时时间

        Returns:
            TaskResult: 包含执行结果
        """
        if self._state not in (PipelineState.READY, PipelineState.RUNNING):
            return TaskResult(
                task_id="",
                success=False,
                phase="submit",
                duration_ms=0.0,
                error=f"Pipeline not ready, state={self._state.value}",
            )

        request = TaskRequest(
            task_id="",  # auto-generated in __post_init__
            task_type=task_type,
            target=target,
            payload=payload or {},
            priority=priority,
        )

        start = time.time()
        result = self._execute_request(request, timeout_s)
        result.duration_ms = (time.time() - start) * 1000.0
        self._completed_tasks.append(result)
        return result

    def submit_task(self, request: TaskRequest) -> bool:
        """
        提交任务到队列 (异步)

        Args:
            request: TaskRequest 对象

        Returns:
            True if queued successfully
        """
        if self._state not in (PipelineState.READY, PipelineState.RUNNING):
            return False
        with self._lock:
            self._task_queue.append(request)
            self._active_tasks[request.task_id] = request
        return True

    def _execute_request(
        self,
        request: TaskRequest,
        timeout_s: Optional[float] = None,
    ) -> TaskResult:
        """执行单个任务请求"""
        timeout = timeout_s or self.config.task_timeout_s
        deadline = time.time() + timeout

        # 1. 场景感知与技能匹配
        skill_used = self._match_skill(request)

        # 2. 行为树规划
        bt_config = self._plan_behavior(request)

        # 3. 任务执行 (使用 TaskExecutor 如果可用)
        if self._task_executor is not None:
            result = self._execute_with_executor(request, deadline)
        else:
            result = self._execute_fallback(request, deadline)

        result.skill_used = skill_used
        result.scene_type = self.config.scene_type
        return result

    def _match_skill(self, request: TaskRequest) -> Optional[str]:
        """根据任务请求匹配最佳技能"""
        if self._skill_registry is None:
            return None
        try:
            matched = self._skill_registry.get_best_skill_for_task(
                task_type=request.task_type,
                scene_type=self.config.scene_type.upper(),
            )
            return matched.name if matched else None
        except Exception as e:
            logger.debug(f"Skill matching failed: {e}")
            return None

    def _plan_behavior(self, request: TaskRequest) -> Optional[Dict[str, Any]]:
        """使用行为树规划任务"""
        if self._bt_planner is None:
            return None
        try:
            return self._bt_planner.plan_task(
                task_type=request.task_type,
                target=request.target,
                grade=self.config.grade,
            )
        except Exception as e:
            logger.debug(f"BT planning failed: {e}")
            return None

    def _execute_with_executor(
        self,
        request: TaskRequest,
        deadline: float,
    ) -> TaskResult:
        """使用任务执行器执行"""
        try:
            from .task_executor import ExecutionResult
            exec_result = self._task_executor.execute_task(
                task_type=request.task_type,
                task_config={
                    'target': request.target,
                    **request.payload,
                },
            )
            success = getattr(exec_result, 'result', None) == ExecutionResult.SUCCESS
            phase_val = getattr(exec_result, 'phase', None)
            phase_str = phase_val.value if phase_val else "completed"
            return TaskResult(
                task_id=request.task_id,
                success=success,
                phase=phase_str,
                duration_ms=getattr(exec_result, 'duration', 0.0) or 0.0,
                output={
                    "record_id": getattr(exec_result, 'record_id', ''),
                    "steps_executed": getattr(exec_result, 'steps_executed', 0),
                },
            )
        except Exception as e:
            return TaskResult(
                task_id=request.task_id,
                success=False,
                phase="execution",
                duration_ms=0.0,
                error=str(e),
            )

    def _execute_fallback(
        self,
        request: TaskRequest,
        deadline: float,
    ) -> TaskResult:
        """回退执行路径 (无 TaskExecutor)"""
        # 模拟任务执行
        elapsed = 0.0
        while elapsed < (deadline - time.time()):
            time.sleep(0.05)
            elapsed += 0.05
            # 检查超时
            if time.time() >= deadline:
                return TaskResult(
                    task_id=request.task_id,
                    success=False,
                    phase="timeout",
                    duration_ms=0.0,
                    error="Task execution timeout",
                )

        return TaskResult(
            task_id=request.task_id,
            success=True,
            phase="completed",
            duration_ms=elapsed * 1000.0,
            output={
                "task_type": request.task_type,
                "target": request.target,
                "grade": self.config.grade,
            },
        )

    # ----------------------------------------------------------
    # 仿真支持
    # ----------------------------------------------------------

    def run_simulation_step(self, dt: Optional[float] = None) -> Dict[str, Any]:
        """
        运行一个仿真步骤

        Args:
            dt: 时间步长 (秒)

        Returns:
            仿真状态字典
        """
        if self._state not in (PipelineState.READY, PipelineState.RUNNING):
            return {"error": f"Cannot step in state {self._state.value}"}

        if self._sim_enhancer is None:
            return {"error": "SimulationEnhancer not initialized"}

        timestep = dt or self.config.simulation_timestep
        try:
            state = self._sim_enhancer.step(timestep)
            return {"success": True, "state": state, "dt": timestep}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_scene_state(self) -> Dict[str, Any]:
        """获取当前场景状态"""
        if self._scene_intel is None:
            return {"scene_type": self.config.scene_type, "grade": self.config.grade}
        try:
            features = self._scene_intel.get_current_features()
            return {
                "scene_type": self.config.scene_type,
                "grade": self.config.grade,
                "features": features,
            }
        except Exception as e:
            return {"scene_type": self.config.scene_type, "error": str(e)}

    # ----------------------------------------------------------
    # 状态查询
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # 联邦学习接口
    # ----------------------------------------------------------

    def register_agv_to_fl(self, agv_id: str, agv_grade: str = "M") -> bool:
        """
        将 AGV 注册为联邦学习客户端

        Args:
            agv_id: AGV 唯一标识
            agv_grade: AGV 等级 (S/M/L/XL/XXL)

        Returns:
            True if registration successful
        """
        if self._fl_coordinator is None:
            logger.warning("Federated learning not enabled")
            return False
        try:
            self._fl_coordinator.register_agv(agv_id, agv_grade)
            logger.info(f"AGV {agv_id} registered to FL system")
            return True
        except Exception as e:
            logger.error(f"FL AGV registration failed: {e}")
            return False

    def start_fl_round(self) -> Optional[Dict[str, Any]]:
        """
        启动一轮联邦学习

        Returns:
            轮次结果字典，失败返回 None
        """
        if self._fl_coordinator is None:
            logger.warning("Federated learning not enabled")
            return None
        try:
            result = self._fl_coordinator.start_training_round()
            if result is not None:
                self._fl_round_count += 1
                self._fl_last_result = result
            return result
        except Exception as e:
            logger.error(f"FL round failed: {e}")
            return None

    def get_fl_status(self) -> Dict[str, Any]:
        """
        获取联邦学习状态

        Returns:
            FL 状态字典
        """
        if self._fl_coordinator is None:
            return {"enabled": False, "message": "Federated learning not enabled"}
        try:
            status = self._fl_coordinator.get_system_status()
            status["round_count"] = self._fl_round_count
            status["last_result"] = {
                "round": getattr(self._fl_last_result, 'round', None),
                "loss": getattr(self._fl_last_result, 'loss', None),
                "accuracy": getattr(self._fl_last_result, 'accuracy', None),
            } if self._fl_last_result else None
            status["enabled"] = True
            return status
        except Exception as e:
            return {"enabled": True, "error": str(e)}

    def trigger_swarm_task(self, task_type: str, target_agvs: List[str],
                           task_config: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        触发蜂群任务

        Args:
            task_type: 任务类型 ("transport"/"patrol"/"inspection"/"assembly")
            target_agvs: 目标 AGV ID 列表 (这些AGV需先注册到蜂群)
            task_config: 任务配置，包含 source_point, target_point 等

        Returns:
            任务 ID，失败返回 None
        """
        if self._swarm_coord is None:
            logger.warning("Swarm coordination not enabled")
            return None
        try:
            from .agv_swarm_coordinator import SwarmTask, TaskPriority
            cfg = task_config or {}
            task = SwarmTask(
                task_type=task_type,
                priority=TaskPriority.P2_MEDIUM,
                source_point=tuple(cfg.get('source', [0.0, 0.0, 0.0])),
                target_point=tuple(cfg.get('dest', [10.0, 0.0, 0.0])),
                payload=cfg.get('payload', 0.0),
                required_agv_spec=cfg.get('agv_spec', 'M'),
                deadline=cfg.get('deadline', 3600.0),
            )
            task_id = self._swarm_coord.add_task(task)
            logger.info(f"Swarm task triggered: {task_id} ({task_type})")
            return task_id
        except Exception as e:
            logger.error(f"Swarm task trigger failed: {e}")
            return None

    def get_swarm_status(self) -> Dict[str, Any]:
        """
        获取蜂群协调状态

        Returns:
            蜂群状态字典
        """
        if self._swarm_coord is None:
            return {"enabled": False, "message": "Swarm coordination not enabled"}
        try:
            return {
                "enabled": True,
                "num_agvs": len(getattr(self._swarm_coord, '_agvs', {})),
                "active_tasks": len(getattr(self._swarm_coord, '_active_tasks', [])),
            }
        except Exception as e:
            return {"enabled": True, "error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """获取 Pipeline 完整状态"""
        return {
            "state": self._state.value,
            "grade": self.config.grade,
            "mode": self.config.mode.value,
            "scene_type": self.config.scene_type,
            "uptime_s": round(self.uptime_s, 1),
            "queue_size": len(self._task_queue),
            "active_tasks": len(self._active_tasks),
            "completed_tasks": len(self._completed_tasks),
            "modules": {
                "behavior_tree": self._bt_planner is not None,
                "scene_intelligence": self._scene_intel is not None,
                "skill_registry": self._skill_registry is not None,
                "memory": self._memory_mgr is not None,
                "task_executor": self._task_executor is not None,
                "hil": self._hil_runner is not None,
                "simulation": self._sim_enhancer is not None,
                "federated_learning": self._fl_coordinator is not None,
                "swarm_coordination": self._swarm_coord is not None,
            },
            "error": self._error_message,
        }

    def get_memory_summary(self) -> Dict[str, Any]:
        """获取记忆系统摘要"""
        if self._memory_mgr is None:
            return {"enabled": False}
        try:
            return {
                "enabled": True,
                "episodic_count": getattr(self._memory_mgr, 'episodic_count', 0),
                "semantic_count": getattr(self._memory_mgr, 'semantic_count', 0),
                "procedural_count": getattr(self._memory_mgr, 'procedural_count', 0),
            }
        except Exception:
            return {"enabled": True, "error": "summary unavailable"}

    def get_skill_summary(self) -> Dict[str, Any]:
        """获取技能注册表摘要"""
        if self._skill_registry is None:
            return {"enabled": False}
        try:
            all_skills = self._skill_registry.list_all_skills()
            active = [s for s in all_skills if self._skill_registry.get_skill(s).status.value == "active"]
            return {
                "enabled": True,
                "total_skills": len(all_skills),
                "active_skills": len(active),
                "scene_type": self.config.scene_type,
            }
        except Exception:
            return {"enabled": True, "error": "summary unavailable"}

    # ----------------------------------------------------------
    # 订阅/通知
    # ----------------------------------------------------------

    def subscribe(self, event: str, callback: Callable) -> None:
        """订阅 Pipeline 事件"""
        self._subscribers.setdefault(event, []).append(callback)

    def _notify(self, event: str, data: Dict[str, Any]) -> None:
        for cb in self._subscribers.get(event, []):
            try:
                cb(data)
            except Exception as e:
                logger.warning(f"Subscriber callback failed: {e}")

    def __repr__(self) -> str:
        return (
            f"EmbodiedPipeline(grade={self.config.grade}, "
            f"mode={self.config.mode.value}, "
            f"scene={self.config.scene_type}, "
            f"state={self._state.value})"
        )

    # ----------------------------------------------------------
    # 状态持久化与恢复
    # ----------------------------------------------------------

    def save_state(self) -> Dict[str, Any]:
        """
        保存 Pipeline 完整状态（用于故障恢复和检查点）

        Returns:
            包含所有可序列化状态的字典
        """
        with self._lock:
            state = {
                'version': '1.0',
                'timestamp': time.time(),
                'pipeline': {
                    'grade': self.config.grade,
                    'mode': self.config.mode.value,
                    'scene_type': self.config.scene_type,
                    'state': self._state.value,
                    'uptime_s': self.uptime_s,
                    'error_message': self._error_message,
                    'fl_round_count': self._fl_round_count,
                },
                'task_queue': [
                    {
                        'task_id': r.task_id,
                        'task_type': r.task_type,
                        'target': r.target,
                        'priority': r.priority,
                        'deadline': r.deadline,
                    }
                    for r in self._task_queue
                ],
                'active_tasks': [
                    {
                        'task_id': r.task_id,
                        'task_type': r.task_type,
                        'target': r.target,
                        'priority': r.priority,
                    }
                    for r in self._active_tasks.values()
                ],
                'completed_tasks': [
                    {
                        'task_id': r.task_id,
                        'success': r.success,
                        'duration_ms': r.duration_ms,
                        'phase': r.phase,
                    }
                    for r in self._completed_tasks
                ],
            }
            return state

    def restore_state(self, state: Dict[str, Any]) -> bool:
        """
        从保存的状态恢复 Pipeline

        Args:
            state: save_state() 返回的状态字典

        Returns:
            是否恢复成功
        """
        try:
            with self._lock:
                self._task_queue.clear()
                for item in state.get('task_queue', []):
                    req = TaskRequest(
                        task_type=item['task_type'],
                        task_id=item.get('task_id', ''),
                        target=item.get('target'),
                        priority=item.get('priority', 2),
                        deadline=item.get('deadline'),
                    )
                    self._task_queue.append(req)
                self._completed_tasks.clear()
                for item in state.get('completed_tasks', [])[-100:]:
                    result = TaskResult(
                        task_id=item['task_id'],
                        success=item['success'],
                        duration_ms=item['duration_ms'],
                        phase=item['phase'],
                    )
                    self._completed_tasks.append(result)
                logger.info(
                    f"Pipeline state restored: "
                    f"{len(self._task_queue)} queued, "
                    f"{len(self._completed_tasks)} completed"
                )
                return True
        except Exception as e:
            logger.error(f"Failed to restore pipeline state: {e}")
            return False

    def export_checkpoint(self, path: str) -> bool:
        """导出检查点到文件 (.json)"""
        import json
        try:
            state = self.save_state()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, default=str)
            logger.info(f"Checkpoint exported to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export checkpoint: {e}")
            return False

    @classmethod
    def import_checkpoint(cls, path: str, **kwargs) -> Optional['EmbodiedPipeline']:
        """从检查点文件恢复并重建 Pipeline"""
        import json
        try:
            with open(path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            p_state = state.get('pipeline', {})
            pipeline = cls(
                grade=p_state.get('grade', 'M'),
                mode=PipelineMode(p_state.get('mode', 'simulation')),
                scene_type=p_state.get('scene_type', 'WAREHOUSE'),
                **kwargs,
            )
            pipeline.restore_state(state)
            logger.info(f"Checkpoint imported from {path}")
            return pipeline
        except Exception as e:
            logger.error(f"Failed to import checkpoint: {e}")
            return None

    def reset_health(self) -> None:
        """重置错误状态，恢复到 READY"""
        with self._lock:
            self._error_message = None
            if self._state == PipelineState.ERROR:
                self._set_state(PipelineState.READY)
            logger.info("Pipeline health reset")

    def get_health_report(self) -> Dict[str, Any]:
        """获取 Pipeline 健康状态报告"""
        report = {
            'timestamp': time.time(),
            'pipeline_state': self._state.value,
            'uptime_s': round(self.uptime_s, 1),
            'error': self._error_message,
            'modules': {},
            'tasks': {
                'queued': len(self._task_queue),
                'active': len(self._active_tasks),
                'completed': len(self._completed_tasks),
                'success_rate': self._calc_success_rate(),
            },
            'performance': {
                'avg_task_duration_ms': self._calc_avg_task_duration(),
            },
        }
        module_checks = [
            ('behavior_tree', self._bt_planner),
            ('scene_intelligence', self._scene_intel),
            ('skill_registry', self._skill_registry),
            ('memory', self._memory_mgr),
            ('task_executor', self._task_executor),
            ('simulation', self._sim_enhancer),
        ]
        for name, module in module_checks:
            report['modules'][name] = 'available' if module is not None else 'unavailable'
        return report

    def _calc_success_rate(self) -> float:
        if not self._completed_tasks:
            return 0.0
        successes = sum(1 for t in self._completed_tasks if t.success)
        return successes / len(self._completed_tasks)

    def _calc_avg_task_duration(self) -> float:
        if not self._completed_tasks:
            return 0.0
        return sum(t.duration_ms for t in self._completed_tasks) / len(self._completed_tasks)


# ============================================================
# 工厂函数
# ============================================================

def create_embodied_pipeline(
    grade: str = "M",
    mode: str = "simulation",
    scene_type: str = "WAREHOUSE",
    **kwargs,
) -> EmbodiedPipeline:
    """
    创建具身智能 Pipeline 的快捷工厂函数

    Args:
        grade: AGV 等级 (S/M/L/XL/XXL)
        mode: 运行模式 ("simulation"/"hardware_in_loop"/"full_physical")
        scene_type: 场景类型
        **kwargs: 传递给 PipelineConfig 的额外参数

    Returns:
        配置好的 EmbodiedPipeline 实例
    """
    mode_enum = {
        "simulation": PipelineMode.SIMULATION,
        "hardware_in_loop": PipelineMode.HARDWARE_IN_LOOP,
        "hil": PipelineMode.HARDWARE_IN_LOOP,
        "full_physical": PipelineMode.FULL_PHYSICAL,
    }.get(mode.lower(), PipelineMode.SIMULATION)

    config = PipelineConfig(
        grade=grade.upper(),
        mode=mode_enum,
        scene_type=scene_type.upper(),
        **kwargs,
    )
    return EmbodiedPipeline(config=config)


def create_pipeline_from_config(config_dict: Dict[str, Any]) -> EmbodiedPipeline:
    """
    从配置字典创建 Pipeline

    Args:
        config_dict: 包含 grade, mode, scene_type 等字段的字典

    Returns:
        配置好的 EmbodiedPipeline 实例
    """
    return create_embodied_pipeline(**config_dict)


# ============================================================
# 错误恢复与诊断增强 (v3.9.4)
# ============================================================

class DegradationLevel(Enum):
    """降级等级"""
    FULLY_OPERATIONAL = "fully_operational"   # 完全正常运行
    DEGRADED_MINOR = "degraded_minor"          # 轻微降级 (部分非关键模块不可用)
    DEGRADED_MODERATE = "degraded_moderate"    # 中度降级 (部分关键模块不可用)
    DEGRADED_SEVERE = "degraded_severe"         # 严重降级 (仅保留核心功能)
    EMERGENCY_ONLY = "emergency_only"          # 仅紧急模式
    OFFLINE = "offline"                        # 完全离线


class DegradedCapability(Enum):
    """可降级的能力"""
    FEDERATED_LEARNING = "federated_learning"
    SWARM_COORDINATION = "swarm_coordination"
    VLA_INFERENCE = "vla_inference"
    BEHAVIOR_TREE_PLANNING = "behavior_tree_planning"
    SCENE_UNDERSTANDING = "scene_understanding"
    LONG_TERM_MEMORY = "long_term_memory"
    HIL_TESTING = "hil_testing"
    MULTI_AGV_COORDINATION = "multi_agv_coordination"
    TERRAIN_MODELING = "terrain_modeling"
    FEDERATED_AGGREGATION = "federated_aggregation"


class DegradationManager:
    """
    优雅降级管理器

    监控各模块健康状态，自动触发降级策略:
    - 模块健康检查与故障检测
    - 降级等级评估
    - 自动降级/恢复
    - 降级能力映射表
    """

    # 模块 -> 降级能力的映射
    MODULE_CAPABILITY_MAP = {
        '_fl_coordinator': {
            'capability': DegradedCapability.FEDERATED_LEARNING,
            'critical': False,
            'fallback': 'local_training_only',
        },
        '_swarm_coord': {
            'capability': DegradedCapability.SWARM_COORDINATION,
            'critical': False,
            'fallback': 'single_agv_mode',
        },
        '_vla_model': {
            'capability': DegradedCapability.VLA_INFERENCE,
            'critical': False,
            'fallback': 'behavior_tree_only',
        },
        '_bt_planner': {
            'capability': DegradedCapability.BEHAVIOR_TREE_PLANNING,
            'critical': True,
            'fallback': 'simple_rule_based',
        },
        '_scene_intel': {
            'capability': DegradedCapability.SCENE_UNDERSTANDING,
            'critical': True,
            'fallback': 'basic_scene_model',
        },
        '_memory_mgr': {
            'capability': DegradedCapability.LONG_TERM_MEMORY,
            'critical': False,
            'fallback': 'episodic_only',
        },
        '_hil_runner': {
            'capability': DegradedCapability.HIL_TESTING,
            'critical': False,
            'fallback': 'simulation_only',
        },
        '_sim_enhancer': {
            'capability': DegradedCapability.TERRAIN_MODELING,
            'critical': False,
            'fallback': 'basic_physics',
        },
    }

    # 降级等级阈值
    DEGRADATION_THRESHOLDS = {
        DegradationLevel.FULLY_OPERATIONAL: 0,
        DegradationLevel.DEGRADED_MINOR: 2,       # 2个非关键模块不可用
        DegradationLevel.DEGRADED_MODERATE: 4,   # 4个模块不可用 或 1个关键模块不可用
        DegradationLevel.DEGRADED_SEVERE: 6,      # 6个模块不可用 或 2个关键模块不可用
        DegradationLevel.EMERGENCY_ONLY: 8,      # 大部分模块不可用
        DegradationLevel.OFFLINE: 10,            # 所有模块不可用
    }

    def __init__(
        self,
        pipeline: EmbodiedPipeline,
        auto_recover: bool = True,
        recovery_interval_s: float = 30.0,
    ):
        self._pipeline = pipeline
        self._auto_recover = auto_recover
        self._recovery_interval_s = recovery_interval_s
        self._degraded_modules: Dict[str, float] = {}  # module_name -> failure_time
        self._degraded_capabilities: Set[DegradedCapability] = set()
        self._active_fallbacks: Dict[DegradedCapability, str] = {}
        self._degradation_history: List[Dict] = []
        self._last_recovery_check: float = time.time()
        self._current_level = DegradationLevel.FULLY_OPERATIONAL

    @property
    def current_level(self) -> DegradationLevel:
        return self._current_level

    @property
    def degraded_capabilities(self) -> Set[DegradedCapability]:
        return self._degraded_capabilities.copy()

    def check_and_update(self) -> DegradationLevel:
        """
        检查所有模块健康状态，更新降级等级

        Returns:
            当前降级等级
        """
        unavailable = []
        critical_unavailable = []

        for module_name, info in self.MODULE_CAPABILITY_MAP.items():
            module = getattr(self._pipeline, module_name, None)
            if module is None:
                unavailable.append(module_name)
                if info['critical']:
                    critical_unavailable.append(module_name)
                self._degraded_modules[module_name] = time.time()
                self._degraded_capabilities.add(info['capability'])
                if info['capability'] not in self._active_fallbacks:
                    self._active_fallbacks[info['capability']] = info['fallback']

        # 计算降级分数
        non_critical = len([m for m in unavailable if m not in critical_unavailable])
        critical_count = len(critical_unavailable)

        # 综合评分
        degradation_score = non_critical + critical_count * 2

        # 确定降级等级
        if degradation_score == 0:
            new_level = DegradationLevel.FULLY_OPERATIONAL
        elif degradation_score >= 8:
            new_level = DegradationLevel.EMERGENCY_ONLY
        elif degradation_score >= 6:
            new_level = DegradationLevel.DEGRADED_SEVERE
        elif degradation_score >= 4 or critical_count >= 1:
            new_level = DegradationLevel.DEGRADED_MODERATE
        elif degradation_score >= 2:
            new_level = DegradationLevel.DEGRADED_MINOR
        else:
            new_level = DegradationLevel.FULLY_OPERATIONAL

        old_level = self._current_level
        if old_level != new_level:
            self._degradation_history.append({
                'timestamp': time.time(),
                'old_level': old_level.value,
                'new_level': new_level.value,
                'degraded_modules': unavailable,
                'critical_unavailable': critical_unavailable,
                'score': degradation_score,
            })
            self._current_level = new_level
            logger.warning(
                f"Degradation level changed: {old_level.value} -> {new_level.value} "
                f"(modules unavailable: {len(unavailable)}, score: {degradation_score})"
            )

        # 触发自动恢复检查
        if self._auto_recover and time.time() - self._last_recovery_check > self._recovery_interval_s:
            self._try_recover_modules()
            self._last_recovery_check = time.time()

        return new_level

    def _try_recover_modules(self) -> None:
        """尝试恢复已降级的模块"""
        recovery_count = 0
        for module_name in list(self._degraded_modules.keys()):
            module = getattr(self._pipeline, module_name, None)
            if module is not None:
                # 模块已恢复
                failure_time = self._degraded_modules.pop(module_name, None)
                if failure_time:
                    info = self.MODULE_CAPABILITY_MAP.get(module_name, {})
                    cap = info.get('capability')
                    if cap:
                        self._degraded_capabilities.discard(cap)
                        self._active_fallbacks.pop(cap, None)
                    logger.info(f"Module {module_name} recovered after {time.time() - failure_time:.1f}s")
                    recovery_count += 1
        if recovery_count > 0:
            self.check_and_update()

    def get_degradation_report(self) -> Dict[str, Any]:
        """获取完整的降级状态报告"""
        score = 0
        for m in self._degraded_modules:
            info = self.MODULE_CAPABILITY_MAP.get(m, {})
            score += 2 if info.get('critical', False) else 1

        return {
            'level': self._current_level.value,
            'level_score': score,
            'degraded_modules': {
                name: {
                    'failed_at': t,
                    'downtime_s': time.time() - t,
                    'capability': self.MODULE_CAPABILITY_MAP.get(name, {}).get('capability', None),
                    'critical': self.MODULE_CAPABILITY_MAP.get(name, {}).get('critical', False),
                    'fallback': self.MODULE_CAPABILITY_MAP.get(name, {}).get('fallback', None),
                }
                for name, t in self._degraded_modules.items()
            },
            'active_fallbacks': {
                cap.value: fallback
                for cap, fallback in self._active_fallbacks.items()
            },
            'degraded_capabilities': [cap.value for cap in self._degraded_capabilities],
            'history': self._degradation_history[-10:],  # 最近10条
            'auto_recover_enabled': self._auto_recover,
            'next_recovery_check_s': max(0, self._recovery_interval_s - (time.time() - self._last_recovery_check)),
        }

    def get_allowed_capabilities(self) -> Set[DegradedCapability]:
        """获取当前允许使用的能力集 (排除已降级的)"""
        all_caps = set(DegradedCapability)
        return all_caps - self._degraded_capabilities

    def can_use_capability(self, capability: DegradedCapability) -> bool:
        """检查某项能力是否可用"""
        return capability not in self._degraded_capabilities

    def get_fallback_for(self, capability: DegradedCapability) -> Optional[str]:
        """获取某能力的降级替代方案"""
        return self._active_fallbacks.get(capability)


class ErrorRecoveryPolicy(Enum):
    """错误恢复策略"""
    MANUAL = "manual"
    RETRY = "retry"
    FALLBACK = "fallback"
    RESTART_MODULE = "restart_module"
    FULL_RESET = "full_reset"


class DiagnosticCollector:
    """Pipeline诊断收集器"""

    def __init__(self, max_history: int = 10000):
        self._max_history = max_history
        self._snapshots: deque = deque(maxlen=max_history)
        self._error_log: deque = deque(maxlen=1000)
        self._metric_summaries: Dict[str, List[float]] = {}
        self._start_time = time.time()

    def record_tick(
        self,
        pipeline_state: str,
        active_tasks: int,
        queue_size: int,
        additional_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        snapshot = {
            'timestamp': time.time(),
            'pipeline_state': pipeline_state,
            'active_tasks': active_tasks,
            'queue_size': queue_size,
            'metrics': additional_metrics or {},
        }
        self._snapshots.append(snapshot)
        if additional_metrics:
            for k, v in additional_metrics.items():
                self._metric_summaries.setdefault(k, []).append(v)

    def record_error(
        self,
        error_code: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._error_log.append({
            'timestamp': time.time(),
            'error_code': error_code,
            'message': error_message,
            'context': context or {},
        })

    def generate_report(self) -> Dict[str, Any]:
        now = time.time()
        elapsed = now - self._start_time
        metric_stats = {}
        for name, values in self._metric_summaries.items():
            if values:
                metric_stats[name] = {
                    'min': round(float(np.min(values)), 4),
                    'max': round(float(np.max(values)), 4),
                    'mean': round(float(np.mean(values)), 4),
                    'count': len(values),
                }
        state_counts: Dict[str, int] = {}
        for snap in self._snapshots:
            s = snap['pipeline_state']
            state_counts[s] = state_counts.get(s, 0) + 1
        return {
            'report_time': now,
            'elapsed_s': round(elapsed, 1),
            'snapshots_collected': len(self._snapshots),
            'errors_logged': len(self._error_log),
            'state_distribution': state_counts,
            'metric_summaries': metric_stats,
            'recent_errors': list(self._error_log)[-20:],
        }

    def export(self, path: str) -> bool:
        import json
        try:
            report = self.generate_report()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            return True
        except Exception:
            return False

    def get_metric_trend(
        self,
        metric_name: str,
        window_size: int = 100,
    ) -> Optional[Dict[str, float]]:
        values = self._metric_summaries.get(metric_name, [])
        if len(values) < 2:
            return None
        recent = values[-window_size:]
        return {
            'first': round(recent[0], 4),
            'last': round(recent[-1], 4),
            'delta': round(recent[-1] - recent[0], 4),
            'slope': round((recent[-1] - recent[0]) / max(1, len(recent) - 1), 4),
        }


# ── EmbodiedPipeline 增强方法 ──────────────────────────────────────────────

# 以下方法通过 monkey-patch 风格添加到 EmbodiedPipeline 类
# 以避免修改原有类结构（向后兼容）

def _get_error_recovery_suggestions(self, error_code=None):
    suggestions: List[Dict[str, Any]] = []
    if self._state == PipelineState.ERROR:
        suggestions.append({
            'action': 'reset_health',
            'description': '重置Pipeline健康状态',
            'reason': f'Pipeline处于ERROR状态: {self._error_message}',
            'priority': 1,
        })
        suggestions.append({
            'action': 'stop + start',
            'description': '完全重启Pipeline',
            'reason': 'ERROR状态需要完整重启才能恢复',
            'priority': 2,
        })
    if len(self._task_queue) > 10:
        suggestions.append({
            'action': 'increase_workers',
            'description': '增加并发任务数或优化任务处理',
            'reason': f'任务队列积压 {len(self._task_queue)} 项',
            'priority': 3,
        })
    if self._skill_registry is None and getattr(self.config, 'enable_skill_registry', True):
        suggestions.append({
            'action': 'check_skill_registry',
            'description': '技能注册表未加载',
            'reason': 'enable_skill_registry=True 但模块未初始化',
            'priority': 2,
        })
    recent = list(self._completed_tasks)[-50:] if self._completed_tasks else []
    if recent:
        failures = [t for t in recent if not t.success]
        if len(failures) / len(recent) > 0.3:
            suggestions.append({
                'action': 'analyze_failure_pattern',
                'description': '任务失败率过高({:.0f}%)'.format(len(failures) / len(recent) * 100),
                'reason': f'{len(failures)}/{len(recent)} 最近任务失败',
                'priority': 2,
            })
    suggestions.sort(key=lambda x: x['priority'])
    return suggestions


def _attempt_auto_recovery(self, max_attempts=3):
    attempts: List[Dict[str, Any]] = []
    if self._state == PipelineState.ERROR:
        try:
            self.reset_health()
            attempts.append({'strategy': 'reset_health', 'success': True, 'message': 'ERROR状态已重置'})
        except Exception as e:
            attempts.append({'strategy': 'reset_health', 'success': False, 'message': str(e)})
    if len(self._task_queue) > 5:
        cleared = len(self._task_queue)
        with self._lock:
            self._task_queue.clear()
        attempts.append({'strategy': 'clear_queue', 'success': True, 'message': f'已清空 {cleared} 项任务'})
    if (self.config.mode == PipelineMode.SIMULATION and self._sim_enhancer is not None):
        try:
            reset_fn = getattr(self._sim_enhancer, 'reset', None)
            if callable(reset_fn):
                reset_fn()
            attempts.append({'strategy': 'reset_simulation', 'success': True, 'message': '仿真环境已重置'})
        except Exception as e:
            attempts.append({'strategy': 'reset_simulation', 'success': False, 'message': str(e)})
    recovered = self._state in (PipelineState.READY, PipelineState.RUNNING) and len(self._task_queue) < 5
    return {
        'recovered': recovered,
        'attempts': attempts,
        'final_state': self._state.value,
        'queue_size': len(self._task_queue),
    }


def _get_diagnostics(self):
    now = time.time()
    recent_100 = list(self._completed_tasks)[-100:] if self._completed_tasks else []
    durations = [t.duration_ms for t in recent_100] if recent_100 else [0]
    p50 = float(np.percentile(durations, 50)) if durations else 0.0
    p95 = float(np.percentile(durations, 95)) if durations else 0.0
    p99 = float(np.percentile(durations, 99)) if durations else 0.0

    def module_health(name, module):
        if module is None:
            return {'status': 'not_loaded', 'available': False}
        try:
            health = getattr(module, 'get_health_status', None)
            if callable(health):
                return {'status': 'healthy', 'available': True, 'detail': health()}
            return {'status': 'healthy', 'available': True}
        except Exception as e:
            return {'status': 'error', 'available': True, 'error': str(e)}

    return {
        'generated_at': now,
        'pipeline': {
            'version': '3.9.4',
            'state': self._state.value,
            'mode': self.config.mode.value,
            'grade': self.config.grade,
            'scene_type': self.config.scene_type,
            'uptime_s': round(self.uptime_s, 1),
        },
        'health': {
            'error_message': self._error_message,
            'success_rate_100': self._calc_success_rate() if recent_100 else 0.0,
        },
        'performance': {
            'p50_duration_ms': round(p50, 2),
            'p95_duration_ms': round(p95, 2),
            'p99_duration_ms': round(p99, 2),
            'avg_duration_ms': round(self._calc_avg_task_duration(), 2),
            'total_completed': len(self._completed_tasks),
        },
        'modules': {
            'behavior_tree': module_health('bt', self._bt_planner),
            'scene_intelligence': module_health('scene', self._scene_intel),
            'skill_registry': module_health('skill', self._skill_registry),
            'memory': module_health('memory', self._memory_mgr),
            'task_executor': module_health('executor', self._task_executor),
            'simulation': module_health('sim', self._sim_enhancer),
            'hil': module_health('hil', self._hil_runner),
            'federated_learning': module_health('fl', self._fl_coordinator),
            'swarm': module_health('swarm', self._swarm_coord),
        },
        'tasks': {
            'queued': len(self._task_queue),
            'active': len(self._active_tasks),
            'completed': len(self._completed_tasks),
            'recent_success_rate': round(self._calc_success_rate(), 4),
        },
        'recovery_suggestions': _get_error_recovery_suggestions(self),
    }


# 动态绑定增强方法到 EmbodiedPipeline
EmbodiedPipeline.get_error_recovery_suggestions = _get_error_recovery_suggestions
EmbodiedPipeline.attempt_auto_recovery = _attempt_auto_recovery
EmbodiedPipeline.get_diagnostics = _get_diagnostics

