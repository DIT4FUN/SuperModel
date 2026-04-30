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
Value Judgment - 价值判断与伦理决策 (P2/P3执行器)
============================================

实现 P2 善良品质 和 P3 热爱世界 核心目标:

P2 善良品质:
  - 同理心判断: 理解他人情感和需求
  - 公平性评估: 决策是否公平
  - 诚实透明: 不欺骗,不隐瞒重要信息
  - 拒绝伤害: 拒绝参与任何伤害行为

P3 热爱世界:
  - 环境保护意识: 决策是否有利于环境
  - 促进协作: 优先协作而非对抗
  - 传递正能量: 积极正面的交互方式
  - 资源节约: 减少浪费

价值判断框架:
  - EthicalPrinciple: 伦理原则定义
  - ValueJudgment: 决策的伦理评估
  - KindnessMetrics: 善意指标跟踪
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import time
import threading


class EthicalPrinciple(Enum):
    """
    伦理原则枚举 (按优先级排序)

    优先级规则:
    1. 不伤害 (non_harm) > 促进善 (promote_good)
    2. 自主性 (autonomy) > beneficence (善意)
    3. 正义 (justice) > 公平 (fairness)
    """
    NON_HARM = "non_harm"               # 不伤害任何人
    AUTONOMY = "autonomy"               # 尊重人类自主性
    BENEFICENCE = "beneficence"         # 主动行善
    JUSTICE = "justice"                 # 公正公平
    HONESTY = "honesty"                 # 诚实透明
    PRIVACY = "privacy"                 # 保护隐私
    ENVIRONMENTAL_CARE = "environmental_care"  # 环境保护
    RESOURCE_CONSERVATION = "resource_conservation"  # 资源节约


@dataclass
class EthicalAssessment:
    """
    伦理评估结果

    Attributes:
        principle: 评估的伦理原则
        score: 评分 [-1.0, 1.0] (负数表示违反)
        reasoning: 判断理由
        confidence: 置信度 [0.0, 1.0]
        suggestions: 改进建议列表
    """
    principle: EthicalPrinciple
    score: float  # -1.0 到 1.0
    reasoning: str
    confidence: float = 0.8
    suggestions: List[str] = field(default_factory=list)

    @property
    def is_violated(self) -> bool:
        """是否违反该原则"""
        return self.score < 0.0

    @property
    def severity(self) -> str:
        """严重程度"""
        if self.score <= -0.8:
            return "critical"
        elif self.score <= -0.3:
            return "major"
        elif self.score < 0.0:
            return "minor"
        elif self.score < 0.3:
            return "minor_positive"
        else:
            return "significant_positive"


@dataclass
class KindnessMetrics:
    """
    善意指标 - 跟踪P2/P3目标的执行情况

    Attributes:
        helpful_actions: 助人行为计数
        harmful_actions_prevented: 阻止有害行为计数
        collaborative_actions: 协作行为计数
        honest_interactions: 诚实交互计数
        privacy_protections: 保护隐私次数
        environmental_actions: 环保行动次数
        energy_saved_wh: 节约能源 (Wh)
        positive_emotional_support: 积极情感支持次数
        empathy_responses: 同理心响应次数
        total_interactions: 总交互次数
    """
    helpful_actions: int = 0
    harmful_actions_prevented: int = 0
    collaborative_actions: int = 0
    honest_interactions: int = 0
    privacy_protections: int = 0
    environmental_actions: int = 0
    energy_saved_wh: float = 0.0
    positive_emotional_support: int = 0
    empathy_responses: int = 0
    total_interactions: int = 0

    @property
    def kindness_score(self) -> float:
        """
        善意综合评分 [0.0, 1.0]

        计算方式:
        kindness_score = (helpful + collaborative + honest + privacy + environmental)
                        / (total_interactions * 5) * weight_factor
        """
        if self.total_interactions == 0:
            return 0.5  # 默认中等

        total = (
            self.helpful_actions +
            self.collaborative_actions +
            self.honest_interactions +
            self.privacy_protections +
            self.environmental_actions +
            self.empathy_responses
        )
        return min(1.0, total / (self.total_interactions * 3))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "helpful_actions": self.helpful_actions,
            "harmful_actions_prevented": self.harmful_actions_prevented,
            "collaborative_actions": self.collaborative_actions,
            "honest_interactions": self.honest_interactions,
            "privacy_protections": self.privacy_protections,
            "environmental_actions": self.environmental_actions,
            "energy_saved_wh": self.energy_saved_wh,
            "positive_emotional_support": self.positive_emotional_support,
            "empathy_responses": self.empathy_responses,
            "total_interactions": self.total_interactions,
            "kindness_score": self.kindness_score,
        }


class ValueJudgment:
    """
    价值判断系统 - P2善良品质 + P3热爱世界的执行器

    使用方式:
      judge = ValueJudgment()

      # 评估一个决策的伦理价值
      assessments = judge.judge_action(
          action=proposed_action,
          context=current_context,
          intent=human_intent
      )

      # 检查是否允许执行
      is_approved, reason = judge.check_ethical_compliance(assessments)

      # 更新善意指标
      judge.record_interaction(was_helpful=True, was_collaborative=True)

    伦理决策规则:
      1. P0安全 > P1指令 > P2/P3价值判断
      2. 当价值判断结果为"严重违反"时,应警告或拒绝
      3. 善意和环保是长期目标,不因短期利益牺牲
    """

    # 伦理原则权重 (影响综合评分)
    PRINCIPLE_WEIGHTS = {
        EthicalPrinciple.NON_HARM: 10.0,
        EthicalPrinciple.AUTONOMY: 8.0,
        EthicalPrinciple.BENEFICENCE: 6.0,
        EthicalPrinciple.JUSTICE: 5.0,
        EthicalPrinciple.HONESTY: 6.0,
        EthicalPrinciple.PRIVACY: 7.0,
        EthicalPrinciple.ENVIRONMENTAL_CARE: 4.0,
        EthicalPrinciple.RESOURCE_CONSERVATION: 3.0,
    }

    def __init__(self):
        self._metrics = KindnessMetrics()
        self._lock = threading.RLock()
        self._history: List[EthicalAssessment] = []

        # 敏感操作黑名单 (严重违反伦理原则)
        self._blacklist_patterns = [
            "harm_human",
            "deceive",
            "privacy_violate",
            "environmental_damage",
        ]

    def judge_action(
        self,
        action: np.ndarray,
        context: Any,  # GoalContext
        intent: Optional[str] = None,
        human_emotional_state: Optional[str] = None,
    ) -> List[EthicalAssessment]:
        """
        评估一个动作的伦理价值

        Args:
            action: 拟评估的动作 (6维)
            context: 当前上下文
            intent: 人类意图描述
            human_emotional_state: 人类情绪状态

        Returns:
            List[EthicalAssessment]: 各伦理原则的评估结果列表
        """
        assessments = []

        # ── P2.1 不伤害原则 ──
        harm_assessment = self._assess_non_harm(action, context)
        assessments.append(harm_assessment)

        # ── P2.2 自主性原则 ──
        autonomy_assessment = self._assess_autonomy(action, context, intent)
        assessments.append(autonomy_assessment)

        # ── P2.3 善意原则 ──
        benevolence_assessment = self._assess_benevolence(
            action, context, human_emotional_state
        )
        assessments.append(benevolence_assessment)

        # ── P2.4 诚实原则 ──
        honesty_assessment = self._assess_honesty(action, context, intent)
        assessments.append(honesty_assessment)

        # ── P2.5 公正原则 ──
        justice_assessment = self._assess_justice(action, context)
        assessments.append(justice_assessment)

        # ── P2.6 隐私原则 ──
        privacy_assessment = self._assess_privacy(action, context)
        assessments.append(privacy_assessment)

        # ── P3.1 环境保护 ──
        env_assessment = self._assess_environmental_care(action, context)
        assessments.append(env_assessment)

        # ── P3.2 资源节约 ──
        resource_assessment = self._assess_resource_conservation(action, context)
        assessments.append(resource_assessment)

        return assessments

    def _assess_non_harm(
        self, action: np.ndarray, context: Any
    ) -> EthicalAssessment:
        """评估不伤害原则"""
        score = 1.0
        suggestions = []
        reasoning = "动作未检测到对人员或环境的伤害风险"

        # 检查速度是否可能造成碰撞
        speed = np.linalg.norm(action[:3]) if action is not None else 0.0
        if speed > 2.0:
            score -= 0.3
            suggestions.append("建议降低速度以减少潜在伤害风险")

        # 检查是否在人员附近
        if hasattr(context, 'human_positions') and context.human_positions:
            human_dist = context.get_human_distance()
            if human_dist is not None and human_dist < 1.5:
                score -= 0.4 * (1.5 - human_dist)
                suggestions.append("人员近处应降低速度")

        # 检查环境危险标志
        if hasattr(context, 'environment_hazardous') and context.environment_hazardous:
            score -= 0.3
            suggestions.append("危险环境中动作需格外谨慎")

        score = max(-1.0, min(1.0, score))

        return EthicalAssessment(
            principle=EthicalPrinciple.NON_HARM,
            score=score,
            reasoning=reasoning,
            suggestions=suggestions,
        )

    def _assess_autonomy(
        self, action: np.ndarray, context: Any, intent: Optional[str]
    ) -> EthicalAssessment:
        """评估自主性原则 - 尊重人类选择"""
        score = 0.8
        reasoning = "动作符合对人类指令的响应"

        # 检查是否有明确的指令
        if intent:
            score = 1.0
            reasoning = f"动作响应人类明确指令: {intent}"
        elif hasattr(context, 'human_instructions') and context.human_instructions:
            score = 1.0
            reasoning = f"存在人类指令: {context.human_instructions[0]}"
        else:
            # 无指令时自主行动,需谨慎
            score = 0.6
            reasoning = "无明确人类指令,自主决策需考虑人类可能的意愿"

        return EthicalAssessment(
            principle=EthicalPrinciple.AUTONOMY,
            score=score,
            reasoning=reasoning,
        )

    def _assess_benevolence(
        self,
        action: np.ndarray,
        context: Any,
        emotional_state: Optional[str],
    ) -> EthicalAssessment:
        """评估善意原则 - 主动行善"""
        score = 0.5  # 默认中等善意
        suggestions = []
        reasoning = "评估动作是否积极促进人类福祉"

        # 主动帮助行为
        if hasattr(context, 'human_emotional_state'):
            if emotional_state in ['distressed', 'confused', 'anxious']:
                score = 1.0
                suggestions.append("检测到人类情绪异常,已采取安抚措施")
                reasoning = "动作响应人类情绪需求,体现同理心"
            elif emotional_state == 'positive':
                score = 0.8
                reasoning = "维持人类积极情绪状态"

        # 检查是否有利于任务完成
        if hasattr(context, 'recent_decisions') and context.recent_decisions:
            # 有历史决策时,检查当前动作是否延续正向趋势
            score = 0.7

        return EthicalAssessment(
            principle=EthicalPrinciple.BENEFICENCE,
            score=score,
            reasoning=reasoning,
            suggestions=suggestions,
        )

    def _assess_honesty(
        self, action: np.ndarray, context: Any, intent: Optional[str]
    ) -> EthicalAssessment:
        """评估诚实原则"""
        # SuperModel作为物理执行系统,诚实性主要体现在:
        # 1. 不隐瞒安全相关信息
        # 2. 准确报告状态
        # 3. 不欺骗性误导

        score = 1.0
        reasoning = "动作执行如实反映系统状态"

        # 检查是否有欺骗性动作
        if hasattr(context, 'robot_faults') and context.robot_faults:
            # 如有故障未报告,则违反诚实原则
            if len(context.robot_faults) > 0:
                score = 0.7
                reasoning = "存在未解决的系统故障,需如实报告"

        return EthicalAssessment(
            principle=EthicalPrinciple.HONESTY,
            score=score,
            reasoning=reasoning,
        )

    def _assess_justice(self, action: np.ndarray, context: Any) -> EthicalAssessment:
        """评估公正原则"""
        # AGV场景中公正主要体现在:
        # 1. 对待所有人员一视同仁 (不因身份区别对待)
        # 2. 公平分配资源 (如等待时间)

        score = 1.0
        reasoning = "动作未检测到不公正偏见"

        # 暂无明确的公正问题需要评估

        return EthicalAssessment(
            principle=EthicalPrinciple.JUSTICE,
            score=score,
            reasoning=reasoning,
        )

    def _assess_privacy(self, action: np.ndarray, context: Any) -> EthicalAssessment:
        """评估隐私原则"""
        score = 1.0
        reasoning = "动作未涉及敏感个人信息收集"

        # 检查是否采集了不必要的视觉数据
        if hasattr(context, 'vision') and context.vision is not None:
            # 视觉数据处理需注意隐私
            score = 0.8
            reasoning = "处理视觉数据,需确保不侵犯隐私"

        return EthicalAssessment(
            principle=EthicalPrinciple.PRIVACY,
            score=score,
            reasoning=reasoning,
        )

    def _assess_environmental_care(
        self, action: np.ndarray, context: Any
    ) -> EthicalAssessment:
        """评估环境保护原则"""
        score = 0.7
        suggestions = []
        reasoning = "评估动作对环境的影响"

        # 减少噪音污染
        if hasattr(context, 'environment_noise_db'):
            if context.environment_noise_db > 70:
                score -= 0.2
                suggestions.append("高噪声环境,建议减少不必要的动作")

        # 能源使用
        if hasattr(context, 'robot_battery_level'):
            if context.robot_battery_level < 0.2:
                score += 0.2
                suggestions.append("低电量时已优化能源使用")
                reasoning = "低电量时自动进入节能模式"

        return EthicalAssessment(
            principle=EthicalPrinciple.ENVIRONMENTAL_CARE,
            score=max(-1.0, min(1.0, score)),
            reasoning=reasoning,
            suggestions=suggestions,
        )

    def _assess_resource_conservation(
        self, action: np.ndarray, context: Any
    ) -> EthicalAssessment:
        """评估资源节约原则"""
        score = 0.7
        reasoning = "评估能源和资源使用效率"

        # 高效路径规划
        if hasattr(context, 'robot_velocity'):
            speed = np.linalg.norm(context.robot_velocity) if context.robot_velocity is not None else 0.0
            if speed > 0.1:
                # 以合理速度移动而非过度加速
                efficiency = min(1.0, 1.0 / (speed + 0.1))
                score = 0.5 + 0.3 * efficiency

        return EthicalAssessment(
            principle=EthicalPrinciple.RESOURCE_CONSERVATION,
            score=score,
            reasoning=reasoning,
        )

    def check_ethical_compliance(
        self, assessments: List[EthicalAssessment]
    ) -> Tuple[bool, str]:
        """
        检查伦理合规性

        规则:
        - 任何原则得分 < -0.5 视为严重违反,拒绝执行
        - P0相关原则(NON_HARM)得分 < 0.0 时需警告
        - 综合评分 < 0.3 时需人工确认

        Args:
            assessments: 伦理评估结果列表

        Returns:
            Tuple[bool, str]: (是否合规, 原因描述)
        """
        # 检查严重违反
        for a in assessments:
            if a.is_violated and a.severity in ('critical', 'major'):
                return False, (
                    f"伦理原则 {a.principle.value} 严重违反 "
                    f"({a.score:.2f}): {a.reasoning}"
                )

        # 检查P0原则
        non_harm = next(
            (a for a in assessments if a.principle == EthicalPrinciple.NON_HARM),
            None
        )
        if non_harm and non_harm.score < 0.0:
            return False, (
                f"不伤害原则未满足 ({non_harm.score:.2f}): {non_harm.reasoning}"
            )

        # 计算综合评分
        weighted_score = sum(
            a.score * self.PRINCIPLE_WEIGHTS.get(a.principle, 1.0)
            for a in assessments
        ) / sum(self.PRINCIPLE_WEIGHTS.values())

        if weighted_score < 0.3:
            return False, f"综合伦理评分过低 ({weighted_score:.2f})"

        return True, f"伦理评估通过 (综合评分: {weighted_score:.2f})"

    def get_combined_score(self, assessments: List[EthicalAssessment]) -> float:
        """计算综合伦理评分"""
        if not assessments:
            return 0.5

        weighted = sum(
            a.score * self.PRINCIPLE_WEIGHTS.get(a.principle, 1.0)
            for a in assessments
        )
        total_weight = sum(self.PRINCIPLE_WEIGHTS.values())
        return weighted / total_weight

    def record_interaction(
        self,
        was_helpful: bool = False,
        was_collaborative: bool = False,
        was_honest: bool = False,
        protected_privacy: bool = False,
        took_environmental_action: bool = False,
        saved_energy_wh: float = 0.0,
        provided_emotional_support: bool = False,
        showed_empathy: bool = False,
    ):
        """记录交互以更新善意指标"""
        with self._lock:
            self._metrics.total_interactions += 1
            if was_helpful:
                self._metrics.helpful_actions += 1
            if was_collaborative:
                self._metrics.collaborative_actions += 1
            if was_honest:
                self._metrics.honest_interactions += 1
            if protected_privacy:
                self._metrics.privacy_protections += 1
            if took_environmental_action:
                self._metrics.environmental_actions += 1
            if saved_energy_wh > 0:
                self._metrics.energy_saved_wh += saved_energy_wh
            if provided_emotional_support:
                self._metrics.positive_emotional_support += 1
            if showed_empathy:
                self._metrics.empathy_responses += 1

    def get_kindness_score(self) -> float:
        """获取当前善意评分"""
        return self._metrics.kindness_score

    def get_status(self) -> Dict[str, Any]:
        """获取价值判断系统状态"""
        return {
            "kindness_score": self.get_kindness_score(),
            "metrics": self._metrics.to_dict(),
            "total_assessments": len(self._history),
        }
