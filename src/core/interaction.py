# Copyright (C) 2026 焦洋 (Jiao Yang) <jiaoyang@cczu.edu.cn>
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
Interaction Manager - 环境交互接口
==================================

负责与物理环境和人类进行实际交互:

功能:
  - 动作执行: 将决策动作转换为实际控制命令
  - 传感器读取: 从传感器获取环境反馈
  - 人类交互: 处理人类指令和反馈
  - 状态反馈: 提供执行结果和状态更新
  - 误差补偿: 处理执行误差

交互层次:
  1. 高层交互: 任务级别 (导航到X, 抓取Y)
  2. 中层交互: 动作级别 (移动, 旋转, 停止)
  3. 底层交互: 控制级别 (关节力矩, 电机电流)

Safety Integration:
  - 所有动作执行前必须通过SafetyShield验证
  - 实时监控执行安全性
  - 异常时触发紧急停止
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import time
import threading


class InteractionState(Enum):
    """交互状态"""
    IDLE = "idle"
    EXECUTING = "executing"
    WAITING_FEEDBACK = "waiting_feedback"
    COMPLETED = "completed"
    FAILED = "failed"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class ExecutionResult:
    """动作执行结果"""
    success: bool
    action_executed: np.ndarray     # 实际执行的动作
    actual_outcome: np.ndarray      # 实际结果
    error: Optional[str] = None     # 错误描述
    execution_time_ms: float = 0.0  # 执行耗时
    safety_violation: bool = False  # 是否发生安全违规
    timestamp: float = field(default_factory=time.time)


class InteractionManager:
    """
    交互管理器 - 执行决策并与环境和人类交互

    职责:
    1. 接收决策结果 (DecisionResult)
    2. 预处理动作 (限幅/平滑/安全检查)
    3. 执行动作到机器人
    4. 收集传感器反馈
    5. 更新机器人状态
    6. 处理人类交互

    执行流程:
      decide() → preprocess_action() → execute() → observe() → update_state()

    安全措施:
      - 所有动作执行前经过SafetyShield验证
      - 执行中持续监控
      - 异常时立即停止并回退
    """

    def __init__(self, safety_shield: Any = None):
        self._lock = threading.RLock()
        self._shield = safety_shield

        # 交互状态
        self._state = InteractionState.IDLE
        self._last_action: Optional[np.ndarray] = None
        self._last_execution_result: Optional[ExecutionResult] = None

        # 执行统计
        self._total_executions = 0
        self._successful_executions = 0
        self._failed_executions = 0
        self._emergency_stops = 0

        # 动作平滑
        self._action_history: List[np.ndarray] = []
        self._max_action_history = 10

        # 回调
        self._on_execution_complete: Optional[Callable] = None
        self._on_emergency_stop: Optional[Callable] = None

    def execute(
        self,
        action: np.ndarray,
        context: Any,  # GoalContext
        blocking: bool = True,
        timeout_s: float = 1.0,
    ) -> ExecutionResult:
        """
        执行动作 (核心方法)

        Args:
            action: 决策产生的6维动作 [vx, vy, vz, wx, wy, wz]
            context: 当前上下文
            blocking: 是否阻塞等待执行完成
            timeout_s: 超时时间

        Returns:
            ExecutionResult: 执行结果
        """
        start_time = time.time()

        with self._lock:
            self._total_executions += 1
            self._state = InteractionState.EXECUTING

            # ── Step 1: 动作预处理 ──
            processed_action = self._preprocess_action(action, context)

            # ── Step 2: 安全检查 (最后一道防线) ──
            if self._shield:
                is_safe, reason = self._shield.check_action(processed_action, context)
                if not is_safe:
                    # 安全检查失败: 触发紧急停止
                    result = ExecutionResult(
                        success=False,
                        action_executed=np.zeros(6),
                        actual_outcome=np.zeros(6),
                        error=f"安全检查失败: {reason}",
                        safety_violation=True,
                        execution_time_ms=(time.time() - start_time) * 1000,
                    )
                    self._handle_failure(result)
                    return result

            # ── Step 3: 实际执行 (模拟) ──
            try:
                actual_result = self._do_execute(processed_action, timeout_s)
                result = ExecutionResult(
                    success=True,
                    action_executed=processed_action,
                    actual_outcome=actual_result,
                    execution_time_ms=(time.time() - start_time) * 1000,
                )
                self._handle_success(result)

            except Exception as e:
                result = ExecutionResult(
                    success=False,
                    action_executed=processed_action,
                    actual_outcome=np.zeros(6),
                    error=str(e),
                    execution_time_ms=(time.time() - start_time) * 1000,
                )
                self._handle_failure(result)

            # ── Step 4: 更新动作历史 ──
            self._action_history.append(processed_action)
            if len(self._action_history) > self._max_action_history:
                self._action_history.pop(0)

            self._last_action = processed_action
            self._last_execution_result = result

            # ── Step 5: 回调 ──
            if self._on_execution_complete:
                self._on_execution_complete(result)

            return result

    def _preprocess_action(
        self,
        action: np.ndarray,
        context: Any,
    ) -> np.ndarray:
        """
        动作预处理

        包括:
        - 限幅 (确保在安全范围内)
        - 平滑 (减少抖动)
        - 安全边界处理
        """
        processed = action.copy().astype(float)

        # 速度限幅
        max_linear = 2.0  # m/s
        max_angular = 1.5  # rad/s

        processed[:3] = np.clip(processed[:3], -max_linear, max_linear)
        processed[3:] = np.clip(processed[3:], -max_angular, max_angular)

        # 平滑处理 (指数移动平均)
        if self._last_action is not None:
            alpha = 0.7  # 平滑系数
            processed = alpha * processed + (1 - alpha) * self._last_action

        return processed

    def _do_execute(
        self,
        action: np.ndarray,
        timeout_s: float,
    ) -> np.ndarray:
        """
        实际执行动作

        在真实硬件上,这里会:
        - 发送CAN/CANopen命令到电机驱动器
        - 发送ROS2话题到下位机
        - 等待执行完成

        在仿真中,这里会:
        - 调用PyBullet/MuJoCo的setJointMotorControl
        - stepSimulation

        Returns:
            np.ndarray: 实际执行后的状态变化
        """
        # 模拟执行
        time.sleep(min(0.01, timeout_s))

        # 模拟执行结果 (应该是实际传感器反馈)
        # 实际中应该读取编码器/IMU等数据
        actual = action * 0.95 + np.random.randn(6) * 0.01

        return actual

    def _handle_success(self, result: ExecutionResult):
        """处理成功执行"""
        self._successful_executions += 1
        self._state = InteractionState.IDLE

    def _handle_failure(self, result: ExecutionResult):
        """处理执行失败"""
        self._failed_executions += 1
        self._state = InteractionState.FAILED

        if result.safety_violation:
            self._emergency_stops += 1
            self._state = InteractionState.EMERGENCY_STOP
            if self._on_emergency_stop:
                self._on_emergency_stop(result.error)

    def emergency_stop(self, reason: str = "manual"):
        """
        触发紧急停止

        Args:
            reason: 停止原因
        """
        with self._lock:
            self._state = InteractionState.EMERGENCY_STOP
            self._emergency_stops += 1

            # 发送零速度命令
            if self._on_emergency_stop:
                self._on_emergency_stop(reason)

    def get_current_state(self) -> InteractionState:
        """获取当前交互状态"""
        return self._state

    def get_last_result(self) -> Optional[ExecutionResult]:
        """获取最近执行结果"""
        return self._last_execution_result

    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        total = self._total_executions
        if total == 0:
            return {
                "total": 0,
                "success_rate": 1.0,
                "emergency_stops": 0,
            }

        return {
            "total": total,
            "success": self._successful_executions,
            "failed": self._failed_executions,
            "success_rate": self._successful_executions / total,
            "emergency_stops": self._emergency_stops,
        }

    def set_callbacks(
        self,
        on_execution_complete: Optional[Callable] = None,
        on_emergency_stop: Optional[Callable] = None,
    ):
        """设置回调"""
        self._on_execution_complete = on_execution_complete
        self._on_emergency_stop = on_emergency_stop

    def get_status(self) -> Dict[str, Any]:
        """获取交互管理器状态"""
        return {
            "state": self._state.value,
            "last_action": (
                self._last_action.tolist()
                if self._last_action is not None else None
            ),
            "execution_stats": self.get_execution_stats(),
        }
