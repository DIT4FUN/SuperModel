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
Goal Dispatcher - 目标调度器
============================

持续执行引擎 - 确保核心目标始终在运行:

功能:
  - 主循环管理: 持续调用决策-执行周期
  - 目标状态维护: 保持always_active目标持续活跃
  - 异常恢复: 检测并从异常中恢复
  - 性能监控: 监控决策/执行延迟
  - 模式切换: 支持不同运行模式切换

运行模式:
  - REAL_TIME: 实时控制 (严格周期)
  - SIMULATION: 仿真模式 (可变速率)
  - STEP: 单步模式 (调试用)
  - PAUSED: 暂停

周期流程:
  ┌─────────────────────────────────────────────────────────┐
  │                    GOAL DISPATCHER                       │
  │  ┌─────────────┐   ┌─────────────┐   ┌──────────────┐  │
  │  │   SENSE     │──►│   THINK     │──►│    ACT       │  │
  │  │ (传感器读取)  │   │ (决策引擎)   │   │ (动作执行)   │  │
  │  └─────────────┘   └─────────────┘   └──────────────┘  │
  │         │                                                │
  │         │ GoalContext                                     │
  │         ▼                                                │
  │  ┌─────────────────────────────────────────────────────┐ │
  │  │              CORE GOALS (Always Active)              │ │
  │  │  P0 安全 │ P1 指令 │ P2 善良 │ P3 热爱 │ P4 自我 │ P5 进化 │
  │  └─────────────────────────────────────────────────────┘ │
  └─────────────────────────────────────────────────────────┘

Safety Integration:
  - SafetyShield在THINK阶段执行,确保P0目标
  - 任何周期都可能触发紧急停止
  - EMERGENCY_STOP模式下只执行P0
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import time
import threading
import logging


class DispatcherMode(Enum):
    """调度器运行模式"""
    REAL_TIME = "real_time"     # 实时模式
    SIMULATION = "simulation"   # 仿真模式
    STEP = "step"             # 单步模式
    PAUSED = "paused"         # 暂停


class DispatcherState(Enum):
    """调度器状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class CycleMetrics:
    """单周期指标"""
    cycle_id: int
    timestamp: float
    sense_time_ms: float = 0.0
    think_time_ms: float = 0.0
    act_time_ms: float = 0.0
    total_time_ms: float = 0.0
    decision_type: str = ""
    safety_overrides: int = 0
    emergency_stops: int = 0


@dataclass
class DispatcherConfig:
    """调度器配置"""
    target_cycle_period_ms: float = 20.0   # 目标周期 (50Hz)
    max_cycle_period_ms: float = 50.0     # 最大允许周期
    warn_on_overrun_ms: float = 25.0      # 周期超时的警告阈值
    max_consecutive_overruns: int = 3      # 最大连续超时次数
    enable_cycle_monitoring: bool = True   # 启用周期监控
    enable_metrics: bool = True            # 启用指标收集


class GoalDispatcher:
    """
    目标调度器 - 持续执行引擎

    这是SuperModel持续运行的核心管理器:

    启动:
      dispatcher = GoalDispatcher(
          core_brain=core_brain,
          context_understanding=ctx,
          decision_making=dm,
          interaction=interaction,
      )
      dispatcher.start()

    停止:
      dispatcher.stop()

    运行中会自动:
    1. 以固定周期调用 sense → think → act
    2. 维护所有always_active目标
    3. 监控性能和异常
    4. 处理紧急停止

    可通过set_mode切换运行模式:
    - REAL_TIME: 严格50Hz周期
    - SIMULATION: 仿真速率
    - STEP: 单步调试
    - PAUSED: 暂停
    """

    def __init__(
        self,
        core_brain: Any = None,
        context_understanding: Any = None,
        decision_making: Any = None,
        interaction: Any = None,
        config: Optional[DispatcherConfig] = None,
    ):
        self._config = config or DispatcherConfig()

        # 子系统引用
        self._brain = core_brain
        self._ctx = context_understanding
        self._dm = decision_making
        self._interaction = interaction

        # 状态
        self._state = DispatcherState.IDLE
        self._mode = DispatcherMode.REAL_TIME
        self._running = False
        self._paused = False

        # 线程
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # 周期管理
        self._cycle_id = 0
        self._cycle_metrics: List[CycleMetrics] = []
        self._max_metrics = 1000

        # 连续超时计数
        self._consecutive_overruns = 0

        # 紧急停止标志
        self._emergency_stop_active = False
        self._emergency_reason: Optional[str] = None

        # 回调
        self._on_cycle_complete: Optional[Callable] = None
        self._on_overrun: Optional[Callable] = None
        self._on_emergency_stop: Optional[Callable] = None

        # 日志
        self._logger = logging.getLogger(__name__)

    def start(self):
        """启动调度器"""
        if self._running:
            self._logger.warning("调度器已在运行")
            return

        self._running = True
        self._stop_event.clear()
        self._state = DispatcherState.RUNNING

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        self._logger.info("目标调度器已启动")

    def stop(self):
        """停止调度器"""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=2.0)

        self._state = DispatcherState.STOPPED
        self._logger.info("目标调度器已停止")

    def pause(self):
        """暂停调度器"""
        self._paused = True
        self._state = DispatcherState.PAUSED
        self._logger.info("调度器已暂停")

    def resume(self):
        """恢复调度器"""
        self._paused = False
        self._state = DispatcherState.RUNNING if self._running else DispatcherState.STOPPED
        self._logger.info("调度器已恢复")

    def set_mode(self, mode: DispatcherMode):
        """设置运行模式"""
        self._mode = mode
        self._logger.info(f"运行模式切换为: {mode.value}")

    def trigger_emergency_stop(self, reason: str):
        """
        触发紧急停止

        进入紧急停止模式后:
        - 只执行P0安全目标
        - 所有其他目标暂停
        - 需要手动调用release_emergency_stop恢复
        """
        self._emergency_stop_active = True
        self._emergency_reason = reason
        self._state = DispatcherState.EMERGENCY_STOP

        if self._interaction:
            self._interaction.emergency_stop(reason)

        if self._on_emergency_stop:
            self._on_emergency_stop(reason)

        self._logger.critical(f"紧急停止触发: {reason}")

    def release_emergency_stop(self):
        """释放紧急停止 (需确认安全)"""
        self._emergency_stop_active = False
        self._emergency_reason = None
        self._state = DispatcherState.RUNNING
        self._logger.info("紧急停止已释放")

    def _run_loop(self):
        """主循环"""
        period_s = self._config.target_cycle_period_ms / 1000.0
        next_cycle_time = time.time()

        while self._running:
            # 检查停止事件
            if self._stop_event.is_set():
                break

            # 暂停处理
            if self._paused:
                time.sleep(0.01)
                next_cycle_time = time.time()
                continue

            # 紧急停止模式
            if self._emergency_stop_active:
                self._run_emergency_cycle()
                next_cycle_time += period_s
                sleep_time = next_cycle_time - time.time()
                if sleep_time > 0:
                    time.sleep(max(0, sleep_time))
                continue

            # 正常周期
            cycle_start = time.time()

            try:
                self._run_cycle()
            except Exception as e:
                self._logger.error(f"周期执行错误: {e}")
                self._state = DispatcherState.ERROR

            cycle_end = time.time()
            cycle_duration_ms = (cycle_end - cycle_start) * 1000

            # 周期监控
            self._check_cycle_timing(cycle_duration_ms)

            # 调度下一周期
            next_cycle_time += period_s
            sleep_time = next_cycle_time - time.time()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # 周期超时
                if self._config.enable_cycle_monitoring:
                    self._consecutive_overruns += 1

    def _run_cycle(self):
        """执行单个周期: Sense → Think → Act"""
        self._cycle_id += 1
        cycle_start = time.time()

        # ── SENSE: 读取传感器,更新上下文 ──
        sense_start = time.time()
        context = self._sense()
        sense_time_ms = (time.time() - sense_start) * 1000

        # ── THINK: 决策 ──
        think_start = time.time()
        decision = self._think(context)
        think_time_ms = (time.time() - think_start) * 1000

        # ── ACT: 执行 ──
        act_start = time.time()
        self._act(decision, context)
        act_time_ms = (time.time() - act_start) * 1000

        # ── 记录指标 ──
        if self._config.enable_metrics:
            metrics = CycleMetrics(
                cycle_id=self._cycle_id,
                timestamp=time.time(),
                sense_time_ms=sense_time_ms,
                think_time_ms=think_time_ms,
                act_time_ms=act_time_ms,
                total_time_ms=(time.time() - cycle_start) * 1000,
                decision_type=decision.decision_type.value if decision else "",
                safety_overrides=1 if decision and not decision.safety_passed else 0,
            )
            self._cycle_metrics.append(metrics)
            if len(self._cycle_metrics) > self._max_metrics:
                self._cycle_metrics.pop(0)

        # ── 回调 ──
        if self._on_cycle_complete:
            self._on_cycle_complete(self._cycle_id, context, decision)

    def _run_emergency_cycle(self):
        """紧急停止模式下的最小周期"""
        self._cycle_id += 1

        # 只执行P0: 读取上下文,获取安全动作,执行
        context = self._sense() if self._ctx else None

        if self._interaction:
            # 紧急停止: 零速度
            self._interaction.execute(np.zeros(6), context, blocking=False)

    def _sense(self) -> Any:
        """
        SENSE阶段: 收集传感器数据,构建上下文

        在真实系统上会:
        - 读取视觉/听觉/触觉/力觉/IMU
        - 读取激光雷达
        - 读取关节状态

        Returns:
            GoalContext 或 ContextRepresentation
        """
        if self._ctx:
            # 调用上下文理解更新
            self._ctx.update_from_goal_context(
                self._brain._context if hasattr(self._brain, '_context') else None
            )
            return self._ctx.get_context()

        # 默认返回空上下文
        return None

    def _think(self, context: Any) -> Any:
        """
        THINK阶段: 做决策

        Args:
            context: 当前上下文

        Returns:
            DecisionResult
        """
        if self._dm:
            return self._dm.decide(context)

        return None

    def _act(self, decision: Any, context: Any):
        """
        ACT阶段: 执行决策

        Args:
            decision: DecisionResult
            context: 当前上下文
        """
        if self._interaction and decision:
            self._interaction.execute(decision.action, context, blocking=False)

    def _check_cycle_timing(self, duration_ms: float):
        """检查周期时间"""
        if duration_ms > self._config.max_cycle_period_ms:
            self._consecutive_overruns += 1

            if self._consecutive_overruns >= self._config.max_consecutive_overruns:
                self._logger.error(
                    f"连续{self._consecutive_overruns}次周期超时, "
                    f"最近一次: {duration_ms:.2f}ms"
                )
                if self._on_overrun:
                    self._on_overrun(duration_ms)
        else:
            self._consecutive_overruns = 0

        if duration_ms > self._config.warn_on_overrun_ms:
            self._logger.warning(
                f"周期超时: {duration_ms:.2f}ms (目标: {self._config.target_cycle_period_ms}ms)"
            )

    def get_cycle_frequency(self, last_n: int = 100) -> float:
        """获取实际运行频率"""
        if len(self._cycle_metrics) < 2:
            return 0.0

        recent = self._cycle_metrics[-last_n:]
        total_time = (
            recent[-1].timestamp - recent[0].timestamp
        ) + recent[0].total_time_ms / 1000.0

        if total_time <= 0:
            return 0.0

        return (len(recent) - 1) / total_time

    def get_average_cycle_time_ms(self, last_n: int = 100) -> float:
        """获取平均周期时间"""
        if not self._cycle_metrics:
            return 0.0

        recent = self._cycle_metrics[-last_n:]
        return sum(m.total_time_ms for m in recent) / len(recent)

    def get_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        return {
            "state": self._state.value,
            "mode": self._mode.value,
            "running": self._running,
            "paused": self._paused,
            "emergency_stop_active": self._emergency_stop_active,
            "emergency_reason": self._emergency_reason,
            "cycle_id": self._cycle_id,
            "avg_cycle_time_ms": self.get_average_cycle_time_ms(),
            "actual_frequency_hz": self.get_cycle_frequency(),
            "consecutive_overruns": self._consecutive_overruns,
        }

    def set_callbacks(
        self,
        on_cycle_complete: Optional[Callable] = None,
        on_overrun: Optional[Callable] = None,
        on_emergency_stop: Optional[Callable] = None,
    ):
        """设置回调"""
        self._on_cycle_complete = on_cycle_complete
        self._on_overrun = on_overrun
        self._on_emergency_stop = on_emergency_stop
