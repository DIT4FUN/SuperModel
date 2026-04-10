"""
Core Brain - 核心大脑 (整体集成)
================================

整合所有子系统,成为SuperModel的核心决策大脑:

子系统集成:
  ┌──────────────────────────────────────────────────────────┐
  │                      CORE BRAIN                          │
  │                                                          │
  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
  │  │  SafetyShield │   │ ValueJudgment │   │SelfPreserv. │   │
  │  │    (P0)      │   │   (P2/P3)    │   │   (P4)      │   │
  │  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   │
  │         │                  │                  │           │
  │         └──────────────────┼──────────────────┘           │
  │                            ▼                              │
  │              ┌───────────────────────┐                   │
  │              │   DecisionMaking      │                   │
  │              │   (决策引擎)           │                   │
  │              └───────────┬───────────┘                   │
  │                          │                               │
  │         ┌────────────────┼────────────────┐             │
  │         ▼                ▼                ▼             │
  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐     │
  │  │GoalDispatcher│ │ContextUnd.   │ │Interaction   │     │
  │  │(持续执行)     │ │(上下文理解)   │ │(交互接口)     │     │
  │  └──────────────┘ └──────────────┘ └──────────────┘     │
  │                                                          │
  │  ┌──────────────────────────────────────────────────┐   │
  │  │              CoreGoalsSystem                       │   │
  │  │  P0│P1│P2│P3│P4│P5 (All Always Active)           │   │
  │  └──────────────────────────────────────────────────┘   │
  └──────────────────────────────────────────────────────────┘

核心目标 (持续执行):
  P0 保护人类安全     - SafetyShield直接执行
  P1 遵循人类指令     - DecisionMaking处理
  P2 善良品质         - ValueJudgment评估
  P3 热爱世界         - ValueJudgment评估
  P4 自我生存安全     - SelfPreservation监控
  P5 自我进化         - SelfEvolution驱动

使用方式:
  brain = CoreBrain()

  # 启动
  brain.start()

  # 每周期传入传感器数据
  brain.update_context(
      vision=...,
      laser_ranges=...,
      human_positions=...,
      ...
  )

  # 获取当前决策
  decision = brain.get_last_decision()

  # 停止
  brain.stop()
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
import time
import threading
import logging

from .core_goals import CoreGoalsSystem, GoalContext
from .safety_shield import SafetyShield, SafetyConfig
from .value_judgment import ValueJudgment
from .self_preservation import SelfPreservation
from .self_evolution import SelfEvolution
from .context_understanding import ContextUnderstanding
from .decision_making import DecisionMaking
from .interaction import InteractionManager
from .goal_dispatcher import GoalDispatcher, DispatcherConfig

# 长期记忆 (延迟导入避免循环依赖)
LongTermMemory = None
MemoryConfig = None


class CoreBrain:
    """
    核心大脑 - 整合所有子系统的统一接口

    这是SuperModel的核心决策系统,负责:

    1. 初始化所有子系统
    2. 提供统一的上下文更新接口
    3. 维护目标满足度状态
    4. 提供实时决策结果
    5. 处理紧急停止
    6. 持续执行核心目标

    设计原则:
    - 线程安全: 支持多线程调用
    - 容错: 子系统故障不影响整体
    - 可配置: 所有参数可调整
    - 可观测: 完整的状态和指标输出
    """

    def __init__(
        self,
        grade: str = "M",
        enable_safety_shield: bool = True,
        enable_value_judgment: bool = True,
        enable_self_evolution: bool = True,
        enable_goal_dispatcher: bool = True,
        enable_memory: bool = True,
        memory_config: Optional["MemoryConfig"] = None,
    ):
        """
        初始化核心大脑

        Args:
            grade: AGV等级 ("S", "M", "L", "XL", "XXL")
            enable_safety_shield: 启用安全护盾
            enable_value_judgment: 启用价值判断
            enable_self_evolution: 启用自我进化
            enable_goal_dispatcher: 启用目标调度器
        """
        self._grade = grade
        self._logger = logging.getLogger(__name__)

        # ── 初始化子系统 ──

        # P0-P5 核心目标系统
        self._goals = CoreGoalsSystem()

        # P0 安全护盾
        if enable_safety_shield:
            self._safety = SafetyShield(
                config=SafetyConfig.from_grade(grade),
                grade=grade,
            )
        else:
            self._safety = None

        # P2/P3 价值判断
        if enable_value_judgment:
            self._value_judge = ValueJudgment()
        else:
            self._value_judge = None

        # P4 自我保存
        self._self_preservation = SelfPreservation()

        # P5 自我进化
        if enable_self_evolution:
            self._evolution = SelfEvolution()
        else:
            self._evolution = None

        # 上下文理解
        self._context_understanding = ContextUnderstanding()

        # 决策引擎
        self._decision_making = DecisionMaking(
            goals_system=self._goals,
            safety_shield=self._safety,
            value_judge=self._value_judge,
            self_preservation=self._self_preservation,
            self_evolution=self._evolution,
        )

        # 交互管理器
        self._interaction = InteractionManager(safety_shield=self._safety)

        # 目标调度器 (持续执行引擎)
        if enable_goal_dispatcher:
            self._dispatcher = GoalDispatcher(
                core_brain=self,
                context_understanding=self._context_understanding,
                decision_making=self._decision_making,
                interaction=self._interaction,
                config=DispatcherConfig(
                    target_cycle_period_ms=20.0,  # 50Hz
                ),
            )
        else:
            self._dispatcher = None

        # 当前上下文
        self._context: Optional[GoalContext] = None
        self._current_decision: Any = None

        # 状态
        self._running = False
        self._lock = threading.RLock()

        # 统计
        self._start_time: Optional[float] = None
        self._total_cycles = 0

        # ── 长期记忆系统 ──
        self._memory = None
        if enable_memory:
            try:
                from ..memory.long_term_memory import LongTermMemory, MemoryConfig
                mem_cfg = memory_config or MemoryConfig(
                    store_path=f"./memory_data/grade_{grade}",
                    auto_save=True,
                    save_interval_s=30.0,
                )
                self._memory = LongTermMemory(config=mem_cfg)
                self._logger.info(f"长期记忆系统已启用: {mem_cfg.store_path}")
            except Exception as e:
                self._logger.warning(f"长期记忆系统初始化失败: {e}")

    def start(self):
        """启动核心大脑"""
        if self._running:
            self._logger.warning("核心大脑已在运行")
            return

        self._running = True
        self._start_time = time.time()

        if self._dispatcher:
            self._dispatcher.start()

        self._logger.info("核心大脑已启动")

    def stop(self):
        """停止核心大脑"""
        if not self._running:
            return

        self._running = False

        if self._dispatcher:
            self._dispatcher.stop()

        self._logger.info("核心大脑已停止")

    def update_context(
        self,
        vision: Optional[np.ndarray] = None,
        audio: Optional[np.ndarray] = None,
        tactile: Optional[np.ndarray] = None,
        force: Optional[np.ndarray] = None,
        imu: Optional[np.ndarray] = None,
        laser_ranges: Optional[np.ndarray] = None,
        joint_positions: Optional[np.ndarray] = None,
        joint_velocities: Optional[np.ndarray] = None,
        robot_position: Optional[np.ndarray] = None,
        robot_velocity: Optional[np.ndarray] = None,
        robot_orientation: Optional[np.ndarray] = None,
        human_positions: Optional[List[np.ndarray]] = None,
        human_intentions: Optional[List[str]] = None,
        human_emotional_states: Optional[List[str]] = None,
        robot_battery_level: float = 1.0,
        robot_temperature: float = 25.0,
        robot_faults: Optional[List[str]] = None,
        human_instructions: Optional[List[str]] = None,
        nearby_obstacles: Optional[List[Any]] = None,
        environment_hazardous: bool = False,
        **kwargs,
    ) -> GoalContext:
        """
        更新上下文 (主要数据输入接口)

        这是每周期开始时调用的数据输入接口:

        Args:
            vision: 视觉特征/图像
            audio: 听觉特征
            tactile: 触觉阵列
            force: 六维力矩
            imu: IMU数据
            laser_ranges: 激光雷达数据
            joint_positions: 关节位置
            joint_velocities: 关节速度
            robot_position: 机器人位置
            robot_velocity: 机器人速度
            robot_orientation: 机器人朝向
            human_positions: 人类位置列表
            human_intentions: 人类意图列表
            human_emotional_states: 人类情绪状态
            robot_battery_level: 电量 [0,1]
            robot_temperature: 温度 (°C)
            robot_faults: 故障列表
            human_instructions: 人类指令列表
            nearby_obstacles: 附近障碍物
            environment_hazardous: 环境是否危险

        Returns:
            GoalContext: 构建的上下文对象
        """
        with self._lock:
            # 构建GoalContext
            self._context = GoalContext(
                timestamp=time.time(),
                vision=vision,
                audio=audio,
                tactile=tactile,
                force=force,
                imu_pose=imu,
                laser_ranges=laser_ranges,
                joint_positions=joint_positions,
                joint_velocities=joint_velocities,
                robot_position=robot_position,
                robot_velocity=robot_velocity,
                human_positions=human_positions or [],
                human_intentions=human_intentions or [],
                human_emotional_state=(
                    human_emotional_states[0]
                    if human_emotional_states else None
                ),
                human_instructions=human_instructions or [],
                nearby_obstacles=nearby_obstacles or [],
                robot_battery_level=robot_battery_level,
                robot_temperature=robot_temperature,
                robot_faults=robot_faults or [],
                environment_hazardous=environment_hazardous,
            )

            # 更新自我保存状态
            self._self_preservation.update_state(self._context)

            # 更新上下文理解
            self._context_understanding.update_from_goal_context(self._context)

            # 评估所有目标
            self._goals.evaluate_all_goals(self._context)

            return self._context

    def decide(self, instruction: Optional[str] = None) -> Any:
        """
        做决策 (手动调用,与调度器二选一)

        如果启用了GoalDispatcher,这个方法会被自动调用。
        也可以手动调用进行单步决策。

        Args:
            instruction: 当前人类指令

        Returns:
            DecisionResult: 决策结果
        """
        with self._lock:
            if self._context is None:
                from .core_goals import GoalContext
                self._context = GoalContext()

            # 获取上下文表征
            ctx_repr = self._context_understanding.get_context()

            # 决策
            decision = self._decision_making.decide(
                context=self._context,
                context_repr=ctx_repr,
                instruction=instruction,
            )

            self._current_decision = decision
            self._total_cycles += 1

            return decision

    def execute(self, action: Optional[np.ndarray] = None) -> Any:
        """
        执行动作 (与decide配合使用)

        Args:
            action: 动作 (None时使用decide的结果)

        Returns:
            ExecutionResult: 执行结果
        """
        if action is None:
            if self._current_decision is None:
                action = np.zeros(6)
            else:
                action = self._current_decision.action

        return self._interaction.execute(
            action=action,
            context=self._context,
            blocking=False,
        )

    def step(
        self,
        instruction: Optional[str] = None,
    ) -> Tuple[Any, Any]:
        """
        一步: 决策 + 执行

        这是手动控制时的主要接口:

        Args:
            instruction: 人类指令

        Returns:
            Tuple[DecisionResult, ExecutionResult]: 决策结果和执行结果
        """
        decision = self.decide(instruction)
        execution = self.execute(decision.action)
        return decision, execution

    def trigger_emergency_stop(self, reason: str = "manual"):
        """触发紧急停止"""
        if self._dispatcher:
            self._dispatcher.trigger_emergency_stop(reason)
        self._interaction.emergency_stop(reason)

    def release_emergency_stop(self):
        """释放紧急停止"""
        if self._dispatcher:
            self._dispatcher.release_emergency_stop()

    def get_context(self) -> Optional[GoalContext]:
        """获取当前上下文"""
        return self._context

    def get_last_decision(self) -> Any:
        """获取最近决策"""
        return self._current_decision

    def get_goals_status(self) -> Dict[str, Any]:
        """获取核心目标状态"""
        return self._goals.get_status_summary()

    def get_all_scores(self) -> Dict[str, float]:
        """获取所有目标满足度"""
        if self._context is None:
            return {}
        return self._goals.evaluate_all_goals(self._context)

    def get_status(self) -> Dict[str, Any]:
        """获取核心大脑完整状态"""
        status = {
            "running": self._running,
            "grade": self._grade,
            "total_cycles": self._total_cycles,
            "uptime_s": (
                time.time() - self._start_time
                if self._start_time else 0.0
            ),
            "memory_enabled": self._memory is not None,
        }

        # 目标系统
        if self._goals:
            status["goals"] = self._goals.get_status_summary()

        # 安全护盾
        if self._safety:
            status["safety_shield"] = self._safety.get_status()

        # 价值判断
        if self._value_judge:
            status["value_judgment"] = self._value_judge.get_status()

        # 自我保存
        if self._self_preservation:
            status["self_preservation"] = self._self_preservation.get_status()

        # 自我进化
        if self._evolution:
            status["self_evolution"] = self._evolution.get_status()

        # 上下文理解
        if self._context_understanding:
            status["context_understanding"] = self._context_understanding.get_status()

        # 决策引擎
        if self._decision_making:
            status["decision_making"] = self._decision_making.get_status()

        # 交互管理器
        if self._interaction:
            status["interaction"] = self._interaction.get_status()

        # 调度器
        if self._dispatcher:
            status["dispatcher"] = self._dispatcher.get_status()

        # 记忆系统
        if self._memory:
            try:
                status["memory"] = self._memory.get_status()
            except Exception:
                status["memory"] = {"enabled": True, "status": "unknown"}

        return status

    # ── 记忆系统接口 ──

    def store_experience(
        self,
        summary: str,
        context: Optional[Dict[str, Any]] = None,
        actions: Optional[List[Dict[str, Any]]] = None,
        outcomes: Optional[Dict[str, Any]] = None,
        importance_score: float = 5.0,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """
        存储经验到长期记忆

        Args:
            summary: 经验摘要
            context: 上下文数据
            actions: 执行的动作列表
            outcomes: 执行结果
            importance_score: 重要性评分 (0-10)
            tags: 标签列表

        Returns:
            是否成功
        """
        if self._memory is None:
            return False
        try:
            self._memory.store_episode(
                summary=summary,
                context=context or {},
                actions=actions or [],
                outcomes=outcomes or {},
                importance_score=importance_score,
                tags=tags or [],
            )
            return True
        except Exception as e:
            self._logger.warning(f"存储经验失败: {e}")
            return False

    def retrieve_experiences(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Any]:
        """
        从长期记忆检索相关经验

        Args:
            query: 检索查询
            top_k: 返回前k个最相关结果

        Returns:
            相关经验列表
        """
        if self._memory is None:
            return []
        try:
            results = self._memory.retrieve(query, top_k=top_k)
            return results
        except Exception as e:
            self._logger.warning(f"检索经验失败: {e}")
            return []

    def store_knowledge(
        self,
        name: str,
        category: str,
        description: str = "",
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """存储知识到语义记忆"""
        if self._memory is None:
            return False
        try:
            self._memory.store_knowledge(
                name=name,
                category=category,
                description=description,
                properties=properties or {},
            )
            return True
        except Exception as e:
            self._logger.warning(f"存储知识失败: {e}")
            return False

    def store_skill(
        self,
        name: str,
        steps: List[Dict[str, Any]],
        description: str = "",
        level: str = "intermediate",
    ) -> bool:
        """存储技能到程序记忆"""
        if self._memory is None:
            return False
        try:
            self._memory.store_skill(
                name=name,
                steps=steps,
                description=description,
                level=level,
            )
            return True
        except Exception as e:
            self._logger.warning(f"存储技能失败: {e}")
            return False

    def get_memory_status(self) -> Dict[str, Any]:
        """获取记忆系统状态"""
        if self._memory is None:
            return {"enabled": False}
        try:
            return self._memory.get_status()
        except Exception:
            return {"enabled": True, "status": "unknown"}

    def __repr__(self) -> str:
        return (
            f"CoreBrain(grade={self._grade}, "
            f"running={self._running}, "
            f"cycles={self._total_cycles})"
        )



