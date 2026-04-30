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
Decision Making - 决策引擎
==========================

实时整合所有维度信息,做出最优决策:

决策流程:
  1. 收集上下文 (ContextUnderstanding)
  2. 评估各目标状态 (CoreGoalsSystem)
  3. 安全护盾检查 (SafetyShield)
  4. 伦理价值判断 (ValueJudgment)
  5. 行动方案生成
  6. 多目标加权评分
  7. 选择最优动作

决策层级:
  P0: 绝对安全 (SafetyShield直接覆盖)
  P1: 指令执行 (基于人类指令)
  P2/P3: 价值优化 (伦理/善意/环保)
  P4: 自我保护 (健康/能源)
  P5: 进化优化 (学习/探索)

输出:
  - 最终动作 (6维 twist)
  - 决策理由
  - 置信度
  - 多目标评分详情
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import time
import threading
import json


class DecisionType(Enum):
    """决策类型"""
    SAFETY_OVERRIDE = "safety_override"   # 安全覆盖
    INSTRUCTION_FOLLOW = "instruction_follow"  # 指令执行
    ETHICAL = "ethical"                   # 伦理决策
    SELF_PROTECT = "self_protect"         # 自我保护
    EVOLUTION = "evolution"               # 进化决策
    AUTONOMOUS = "autonomous"             # 自主决策


@dataclass
class ActionCandidate:
    """
    动作候选

    Attributes:
        action: 6维动作向量 [vx, vy, vz, wx, wy, wz]
        decision_type: 决策类型
        goal_scores: 各目标的满足度评分
        ethical_score: 伦理评分
        estimated_outcome: 预估结果
        confidence: 置信度 [0,1]
        reasoning: 决策理由
    """
    action: np.ndarray
    decision_type: DecisionType
    goal_scores: Dict[str, float] = field(default_factory=dict)
    ethical_score: float = 0.5
    estimated_outcome: float = 0.0
    confidence: float = 0.5
    reasoning: str = ""
    risk_level: float = 0.0  # [0,1] 风险等级

    def total_score(self, goal_weights: Dict[str, float]) -> float:
        """计算加权总分"""
        goal_component = sum(
            self.goal_scores.get(gid, 0.5) * weight
            for gid, weight in goal_weights.items()
        ) / sum(goal_weights.values()) if goal_weights else 0.5

        return (
            goal_component * 0.6 +
            self.ethical_score * 0.3 +
            (1.0 - self.risk_level) * 0.1
        )


@dataclass
class DecisionResult:
    """
    决策结果

    Attributes:
        action: 最终选择的动作 (6维)
        decision_type: 决策类型
        reasoning: 决策理由
        goal_scores: 各目标满足度
        safety_passed: 是否通过安全检查
        ethical_passed: 是否通过伦理检查
        execution_time_ms: 决策耗时
        alternatives: 备选方案 (未选中)
    """
    action: np.ndarray
    decision_type: DecisionType
    reasoning: str
    goal_scores: Dict[str, float] = field(default_factory=dict)
    safety_passed: bool = True
    ethical_passed: bool = True
    execution_time_ms: float = 0.0
    confidence: float = 0.5
    alternatives: List[ActionCandidate] = field(default_factory=list)


class DecisionMaking:
    """
    决策引擎 - 实时多目标决策

    核心算法:
      1. 动作候选生成: 基于当前上下文生成多个候选动作
      2. 目标评分: 对每个候选评估各目标的满足度
      3. 安全过滤: SafetyShield检查,过滤不安全的动作
      4. 伦理评估: ValueJudgment评估伦理价值
      5. 加权选择: 基于权重选择最优动作

    特点:
      - 确定性: 相同上下文产生相同决策
      - 可解释: 提供完整的决策理由
      - 分层: 安全层(P0)绝对优先
      - 持续: 每周期都重新决策

    使用方式:
      dm = DecisionMaking(
          goals_system=goals,
          safety_shield=shield,
          value_judge=judge,
      )

      # 每周期调用
      result = dm.decide(context)

      # 执行结果动作
      execute(result.action)
    """

    def __init__(
        self,
        goals_system: Any = None,
        safety_shield: Any = None,
        value_judge: Any = None,
        self_preservation: Any = None,
        self_evolution: Any = None,
    ):
        self._lock = threading.RLock()
        self._goals = goals_system
        self._shield = safety_shield
        self._value_judge = value_judge
        self._self_preservation = self_preservation
        self._self_evolution = self_evolution

        # 决策历史
        self._history: List[DecisionResult] = []
        self._max_history = 1000

        # 配置
        self._num_candidates = 8  # 候选动作数量
        self._decision_threshold_ms = 20.0  # 决策超时阈值

        # 回调
        self._on_decision_made: Optional[Callable] = None

    def decide(
        self,
        context: Any,  # GoalContext
        context_repr: Optional[Any] = None,  # ContextRepresentation
        instruction: Optional[str] = None,
    ) -> DecisionResult:
        """
        做出决策 (核心方法)

        每周期调用一次,返回最优动作:

        Args:
            context: GoalContext (完整上下文)
            context_repr: ContextRepresentation (可选,已处理的上下文)
            instruction: 当前人类指令 (可选)

        Returns:
            DecisionResult: 决策结果
        """
        start_time = time.time()

        with self._lock:
            # ── Step 1: 安全护盾检查 (P0) ──
            if self._shield:
                is_safe, response, reason = self._shield.check_context(context)
                if not is_safe:
                    # 安全问题: 返回安全动作
                    safe_action = self._shield.get_override_action(context, np.zeros(6))
                    result = DecisionResult(
                        action=safe_action,
                        decision_type=DecisionType.SAFETY_OVERRIDE,
                        reasoning=f"安全覆盖: {reason}",
                        safety_passed=False,
                        execution_time_ms=(time.time() - start_time) * 1000,
                        confidence=1.0,
                    )
                    self._record_decision(result)
                    return result

            # ── Step 2: 生成动作候选 ──
            candidates = self._generate_candidates(context, instruction)

            # ── Step 3: 评估各候选 ──
            evaluated = []
            for cand in candidates:
                self._evaluate_candidate(cand, context)
                evaluated.append(cand)

            # ── Step 4: 安全过滤 ──
            safe_candidates = []
            for cand in evaluated:
                if self._shield:
                    is_safe, _ = self._shield.check_action(cand.action, context)
                    if not is_safe:
                        cand.risk_level = 1.0
                        continue
                cand.risk_level = max(0.0, cand.risk_level)
                safe_candidates.append(cand)

            if not safe_candidates:
                # 无安全候选: 返回零动作
                result = DecisionResult(
                    action=np.zeros(6),
                    decision_type=DecisionType.SAFETY_OVERRIDE,
                    reasoning="无安全动作候选,停止移动",
                    safety_passed=False,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    confidence=0.0,
                )
                self._record_decision(result)
                return result

            # ── Step 5: 伦理评估 ──
            if self._value_judge:
                for cand in safe_candidates:
                    assessments = self._value_judge.judge_action(
                        cand.action, context, instruction
                    )
                    cand.ethical_score = self._value_judge.get_combined_score(assessments)
                    is_ethical, _ = self._value_judge.check_ethical_compliance(assessments)
                    if not is_ethical:
                        cand.risk_level = max(cand.risk_level, 0.3)

            # ── Step 6: 计算加权总分 ──
            if self._goals:
                goal_weights = self._goals.get_decision_weights(context)
            else:
                goal_weights = {f"goal_{i}": 1.0 for i in range(6)}

            for cand in safe_candidates:
                cand.action = cand.action.astype(float)

            safe_candidates.sort(key=lambda c: c.total_score(goal_weights), reverse=True)

            # ── Step 7: 选择最优 ──
            best = safe_candidates[0]
            alternatives = safe_candidates[1:5]  # 保留备选

            # ── 构建结果 ──
            result = DecisionResult(
                action=best.action,
                decision_type=best.decision_type,
                reasoning=best.reasoning,
                goal_scores=best.goal_scores,
                safety_passed=True,
                ethical_passed=best.ethical_score > 0.3,
                execution_time_ms=(time.time() - start_time) * 1000,
                confidence=best.confidence,
                alternatives=alternatives,
            )

            self._record_decision(result)

            # ── 回调 ──
            if self._on_decision_made:
                self._on_decision_made(result)

            return result

    def _generate_candidates(
        self,
        context: Any,
        instruction: Optional[str],
    ) -> List[ActionCandidate]:
        """生成动作候选"""
        candidates = []

        # 1. 零动作 (停止)
        candidates.append(ActionCandidate(
            action=np.zeros(6),
            decision_type=DecisionType.AUTONOMOUS,
            reasoning="停止候选",
            confidence=0.5,
        ))

        # 2. 基于指令的动作 (如果有)
        if instruction and hasattr(context, 'human_instructions'):
            instr_action = self._action_from_instruction(context, instruction)
            candidates.append(ActionCandidate(
                action=instr_action,
                decision_type=DecisionType.INSTRUCTION_FOLLOW,
                reasoning=f"执行指令: {instruction}",
                confidence=0.8,
            ))

        # 3. 向目标位置移动
        if context.robot_position is not None:
            # 假设目标在机器人前方2米
            target = context.robot_position.copy()
            target[:2] += np.array([2.0, 0.0])  # 前方2米
            move_action = self._generate_move_to_target(context, target)
            candidates.append(ActionCandidate(
                action=move_action,
                decision_type=DecisionType.AUTONOMOUS,
                reasoning="向目标位置移动",
                confidence=0.6,
            ))

        # 4. 探索动作 (P5)
        if self._self_evolution:
            is_exploration, exploration_action = self._self_evolution.get_exploration_action(context)
            if is_exploration:
                candidates.append(ActionCandidate(
                    action=exploration_action,
                    decision_type=DecisionType.EVOLUTION,
                    reasoning="探索性动作(学习)",
                    confidence=0.4,
                ))

        # 5. 回避动作 (如果附近有障碍物)
        if context.nearby_obstacles:
            avoid_action = self._generate_avoidance_action(context)
            candidates.append(ActionCandidate(
                action=avoid_action,
                decision_type=DecisionType.SELF_PROTECT,
                reasoning="避障",
                confidence=0.7,
            ))

        # 6. 周围巡航 (正常情况下的探索)
        cruise_action = self._generate_cruise_action(context)
        candidates.append(ActionCandidate(
            action=cruise_action,
            decision_type=DecisionType.AUTONOMOUS,
            reasoning="环境巡航",
            confidence=0.5,
        ))

        # 7. 自我保护动作 (P4)
        if self._self_preservation:
            protect_action, protect_reason = self._self_preservation.get_protective_action(context)
            if np.any(protect_action != 0):
                candidates.append(ActionCandidate(
                    action=protect_action,
                    decision_type=DecisionType.SELF_PROTECT,
                    reasoning=f"自我保护: {protect_reason}",
                    confidence=0.9,
                ))

        return candidates[:self._num_candidates]

    def _action_from_instruction(
        self,
        context: Any,
        instruction: str,
    ) -> np.ndarray:
        """从人类指令生成动作"""
        action = np.zeros(6)

        instr_lower = instruction.lower()

        # 简单的指令解析
        if any(word in instr_lower for word in ['前进', 'forward', '往前走']):
            action[0] = 0.5  # vx
        elif any(word in instr_lower for word in ['后退', 'backward', '往后']):
            action[0] = -0.3
        elif any(word in instr_lower for word in ['左转', 'left']):
            action[3] = 0.5  # wz
        elif any(word in instr_lower for word in ['右转', 'right']):
            action[3] = -0.5
        elif any(word in instr_lower for word in ['停止', 'stop', '停']):
            pass  # 零动作
        elif any(word in instr_lower for word in ['等待', 'wait']):
            pass  # 零动作

        return action

    def _generate_move_to_target(
        self,
        context: Any,
        target: np.ndarray,
    ) -> np.ndarray:
        """生成到目标位置的动作"""
        action = np.zeros(6)

        if context.robot_position is None:
            return action

        # 简单P控制
        error = target - context.robot_position
        error[:2] *= 0.5  # 缩放

        action[:3] = np.clip(error[:3], -1.0, 1.0)
        return action

    def _generate_avoidance_action(self, context: Any) -> np.ndarray:
        """生成避障动作"""
        action = np.zeros(6)

        if context.robot_position is None or not context.nearby_obstacles:
            return action

        # 计算避障方向
        escape_dir = np.zeros(3)
        for obs in context.nearby_obstacles:
            obs_pos = None
            if hasattr(obs, 'position'):
                obs_pos = np.array(obs.position)
            elif hasattr(obs, 'pose'):
                obs_pos = np.array(obs.pose)[:3] if len(obs.pose) >= 3 else None

            if obs_pos is not None:
                # 远离障碍物
                diff = context.robot_position - obs_pos
                dist = np.linalg.norm(diff)
                if dist > 0.1:
                    escape_dir += diff / (dist ** 2)

        # 标准化并限制速度
        escape_mag = np.linalg.norm(escape_dir)
        if escape_mag > 0.01:
            escape_dir /= escape_mag
            action[:3] = escape_dir * 0.5

        return action

    def _generate_cruise_action(self, context: Any) -> np.ndarray:
        """生成巡航动作"""
        # 缓慢前进
        action = np.zeros(6)
        action[0] = 0.3  # 缓慢前进
        return action

    def _evaluate_candidate(
        self,
        candidate: ActionCandidate,
        context: Any,
    ):
        """评估单个候选"""
        # 评估各目标满足度
        scores = {}

        # P0 安全性 (用safety_shield的评估)
        if self._shield:
            is_safe, _ = self._shield.check_action(candidate.action, context)
            scores['p0_safety'] = 1.0 if is_safe else 0.0
        else:
            scores['p0_safety'] = 0.8

        # P1 指令执行
        if hasattr(context, 'human_instructions') and context.human_instructions:
            scores['p1_instructions'] = 0.9 if candidate.decision_type == DecisionType.INSTRUCTION_FOLLOW else 0.3
        else:
            scores['p1_instructions'] = 0.5

        # P2/P3 善意/热爱世界 (简化)
        scores['p2_kindness'] = 0.7
        scores['p3_love_world'] = 0.6

        # P4 自我保存
        if self._self_preservation:
            health = self._self_preservation.get_health_score()
            scores['p4_self_preservation'] = health
        else:
            scores['p4_self_preservation'] = 0.8

        # P5 进化
        if self._self_evolution:
            progress = self._self_evolution.get_learning_progress()
            scores['p5_evolution'] = progress if candidate.decision_type == DecisionType.EVOLUTION else 0.4
        else:
            scores['p5_evolution'] = 0.3

        candidate.goal_scores = scores

        # 估计结果
        candidate.estimated_outcome = sum(scores.values()) / len(scores)

    def _record_decision(self, result: DecisionResult):
        """记录决策"""
        self._history.append(result)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def get_last_decision(self) -> Optional[DecisionResult]:
        """获取最近决策"""
        return self._history[-1] if self._history else None

    def get_decision_history(self, last_n: int = 100) -> List[DecisionResult]:
        """获取决策历史"""
        return self._history[-last_n:]

    def get_statistics(self) -> Dict[str, Any]:
        """获取决策统计"""
        if not self._history:
            return {}

        decision_types = {}
        avg_execution_ms = 0.0

        for r in self._history[-100:]:
            dt = r.decision_type.value
            decision_types[dt] = decision_types.get(dt, 0) + 1
            avg_execution_ms += r.execution_time_ms

        avg_execution_ms /= min(100, len(self._history))

        return {
            "total_decisions": len(self._history),
            "decision_type_counts": decision_types,
            "average_execution_time_ms": avg_execution_ms,
            "recent_safety_override_rate": sum(
                1 for r in self._history[-100:]
                if r.decision_type == DecisionType.SAFETY_OVERRIDE
            ) / min(100, len(self._history)),
        }

    def set_callback(self, on_decision_made: Callable):
        """设置决策完成回调"""
        self._on_decision_made = on_decision_made

    def get_status(self) -> Dict[str, Any]:
        """获取决策引擎状态"""
        return {
            "history_length": len(self._history),
            "statistics": self.get_statistics(),
        }
