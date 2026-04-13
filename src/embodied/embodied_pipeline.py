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

    def to_dict(self) -> Dict[str, Any]:
        return {
            'grade': self.grade,
            'mode': self.mode.value,
            'scene_type': self.scene_type,
            'enable_skill_registry': self.enable_skill_registry,
            'enable_memory': self.enable_memory,
            'enable_scene_intelligence': self.enable_scene_intelligence,
            'enable_hil': self.enable_hil,
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
            scene_type=scene_type,
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
