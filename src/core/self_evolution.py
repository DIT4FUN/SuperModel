"""
Self-Evolution - 自我进化模块 (P5执行器)
==========================================

实现 P5 自我进化 核心目标:

目标:
  持续学习、提升能力、适应新环境和挑战

功能:
  - 知识积累: 从经验中提取知识
  - 技能提升: 优化现有技能水平
  - 策略优化: 发现更优决策策略
  - 探索创新: 尝试新方法新技术
  - 适应环境: 适应不同环境和任务

进化维度:
  - 认知进化: 知识量和理解深度
  - 行为进化: 动作执行效率和安全性
  - 社交进化: 人机交互的自然度
  - 情感进化: 情绪识别和表达能力

进化指标:
  - 学习进度 [0.0, 1.0]
  - 技能熟练度
  - 探索新知识率
  - 适应速度
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import time
import threading
import json


class EvolutionDimension(Enum):
    """进化维度"""
    COGNITIVE = "cognitive"          # 认知进化
    BEHAVIORAL = "behavioral"       # 行为进化
    SOCIAL = "social"               # 社交进化
    EMOTIONAL = "emotional"         # 情感进化
    ADAPTIVE = "adaptive"           # 适应进化


@dataclass
class SkillProficiency:
    """技能熟练度"""
    skill_name: str
    proficiency: float = 0.0      # [0.0, 1.0] 熟练度
    practice_count: int = 0        # 练习次数
    success_rate: float = 0.0      # 成功率
    last_practice_time: Optional[float] = None
    mastery_level: str = "novice"  # novice/beginner/competent/proficient/expert

    def update(self, success: bool, quality_score: float = 0.5):
        """更新技能熟练度"""
        self.practice_count += 1

        # 指数加权移动平均更新成功率
        alpha = 0.1
        self.success_rate = (
            alpha * (1.0 if success else 0.0) +
            (1 - alpha) * self.success_rate
        )

        # 更新熟练度
        self.proficiency = min(1.0, self.proficiency + 0.01 * quality_score)

        # 更新 mastery level
        self._update_mastery_level()

        self.last_practice_time = time.time()

    def _update_mastery_level(self):
        """更新掌握等级"""
        if self.practice_count < 10:
            self.mastery_level = "novice"
        elif self.practice_count < 50:
            self.mastery_level = "beginner"
        elif self.practice_count < 200:
            self.mastery_level = "competent"
        elif self.practice_count < 500:
            self.mastery_level = "proficient"
        else:
            self.mastery_level = "expert"


@dataclass
class LearningExperience:
    """学习经验"""
    experience_id: str
    context_type: str              # 经验类型
    action_taken: Any
    outcome: float                 # 结果评分 [-1, 1]
    timestamp: float = field(default_factory=time.time)
    knowledge_extracted: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class EvolutionMetrics:
    """进化指标"""
    dimension: EvolutionDimension
    progress: float = 0.0           # [0.0, 1.0] 进化进度
    velocity: float = 0.0          # 进化速度 (进度变化率)
    novelty_rate: float = 0.0      # 探索新知识率
    adaptation_time_constant: float = 1.0  # 适应时间常数 (小时)

    # 统计
    total_learning_cycles: int = 0
    successful_adaptations: int = 0
    failed_adaptations: int = 0

    @property
    def adaptation_success_rate(self) -> float:
        if self.total_learning_cycles == 0:
            return 0.5
        return self.successful_adaptations / self.total_learning_cycles


class SelfEvolution:
    """
    自我进化系统 - P5核心目标的执行器

    职责:
    1. 持续监测学习进度和技能提升
    2. 探索新知识而非只利用已有
    3. 从成功和失败中学习
    4. 适应新的环境和任务
    5. 保持好奇心和探索精神

    进化策略:
    - 利用 vs 探索 (Exploitation vs Exploration)
      - epsilon-greedy: epsilon%时间探索
      - 成功率低时增加探索
    - 知识巩固: 定期回顾重要经验
    - 技能专精: 在关键技能上投入更多练习

    使用方式:
      evo = SelfEvolution()

      # 记录一次经验
      evo.record_experience(context_type, action, outcome)

      # 获取探索动作 (vs 利用已有知识)
      exploration_action = evo.get_exploration_action(context, epsilon=0.1)

      # 获取学习进度
      progress = evo.get_learning_progress()
    """

    def __init__(self):
        self._lock = threading.RLock()

        # 技能熟练度字典
        self._skills: Dict[str, SkillProficiency] = {}

        # 学习经验库
        self._experiences: List[LearningExperience] = []
        self._max_experiences = 10000

        # 进化指标 (按维度)
        self._evolution_metrics: Dict[EvolutionDimension, EvolutionMetrics] = {
            dim: EvolutionMetrics(dimension=dim)
            for dim in EvolutionDimension
        }

        # 好奇心状态
        self._curiosity_level: float = 0.5  # [0, 1]
        self._exploration_bonus: float = 0.0

        # 知识图谱 (简化为标签)
        self._knowledge_tags: Dict[str, float] = {}  # tag -> 熟悉度

        # 学习历史
        self._learning_history: List[Dict[str, Any]] = []

        # 回调
        self._on_novel_discovery: Optional[Callable] = None

    def record_experience(
        self,
        context_type: str,
        action: Any,
        outcome: float,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        记录一次学习经验

        Args:
            context_type: 经验发生的上下文类型
            action: 执行的动作
            outcome: 结果评分 [-1.0, 1.0] 负数=失败, 正数=成功
            tags: 关联的知识标签

        Returns:
            str: 经验ID
        """
        with self._lock:
            exp_id = f"exp_{len(self._experiences)}_{int(time.time())}"

            exp = LearningExperience(
                experience_id=exp_id,
                context_type=context_type,
                action_taken=action,
                outcome=outcome,
                tags=tags or [],
            )

            self._experiences.append(exp)

            # 保持经验库大小
            if len(self._experiences) > self._max_experiences:
                # 删除最老的经验
                self._experiences = self._experiences[-self._max_experiences:]

            # 更新进化指标
            self._update_evolution_metrics(outcome)

            # 更新知识标签
            if tags:
                for tag in tags:
                    if tag not in self._knowledge_tags:
                        self._knowledge_tags[tag] = 0.0
                    self._knowledge_tags[tag] = min(
                        1.0,
                        self._knowledge_tags[tag] + 0.01
                    )

            # 记录历史
            self._learning_history.append({
                "timestamp": time.time(),
                "experience_id": exp_id,
                "context_type": context_type,
                "outcome": outcome,
            })

            return exp_id

    def _update_evolution_metrics(self, outcome: float):
        """更新进化指标"""
        # 认知进化 - 从结果中学习
        cog = self._evolution_metrics[EvolutionDimension.COGNITIVE]
        cog.total_learning_cycles += 1
        if outcome > 0:
            cog.successful_adaptations += 1
        elif outcome < -0.5:
            cog.failed_adaptations += 1

        # 更新进度 (基于成功率)
        rate = cog.adaptation_success_rate
        cog.progress = min(1.0, cog.progress * 0.99 + rate * 0.01)

        # 行为进化 - 动作执行改进
        beh = self._evolution_metrics[EvolutionDimension.BEHAVIORAL]
        beh.total_learning_cycles += 1
        if outcome > 0.5:
            beh.successful_adaptations += 1

        # 适应进化
        adp = self._evolution_metrics[EvolutionDimension.ADAPTIVE]
        adp.total_learning_cycles += 1

    def get_exploration_action(
        self,
        context: Any,
        epsilon: Optional[float] = None,
    ) -> Tuple[bool, np.ndarray]:
        """
        获取探索动作

        使用 epsilon-greedy 策略平衡探索和利用:

        Args:
            context: 当前上下文
            epsilon: 探索概率 (None时自适应)

        Returns:
            Tuple[bool, np.ndarray]: (is_exploration, action)
                - is_exploration: True=探索, False=利用
                - action: 6维动作
        """
        with self._lock:
            # 自适应 epsilon
            if epsilon is None:
                # 成功率低时增加探索
                cog = self._evolution_metrics[EvolutionDimension.COGNITIVE]
                base_rate = cog.adaptation_success_rate
                epsilon = max(0.05, 0.3 - base_rate * 0.25)

            # 决定是探索还是利用
            if np.random.random() < epsilon:
                # 探索: 尝试新动作
                action = self._generate_novel_action(context)
                return True, action
            else:
                # 利用: 基于已有知识
                action = self._generate_exploitative_action(context)
                return False, action

    def _generate_novel_action(self, context: Any) -> np.ndarray:
        """生成探索性动作 (尝试新事物)"""
        # 增加探索奖励
        self._curiosity_level = min(1.0, self._curiosity_level + 0.01)

        # 探索动作: 添加随机扰动
        base_action = np.zeros(6)
        noise_scale = 0.5 * self._curiosity_level

        # 对每个维度添加噪声
        for i in range(6):
            base_action[i] = np.random.uniform(-noise_scale, noise_scale)

        return base_action

    def _generate_exploitative_action(self, context: Any) -> np.ndarray:
        """生成利用性动作 (基于已有知识)"""
        # 利用成功经验
        positive_experiences = [
            e for e in self._experiences[-100:]
            if e.outcome > 0.3
        ]

        if not positive_experiences:
            return np.zeros(6)

        # 简单策略: 使用最近的成功动作
        best_exp = positive_experiences[-1]
        if hasattr(best_exp.action_taken, '__iter__'):
            return np.array(best_exp.action_taken)[:6]
        return np.zeros(6)

    def update_skill(
        self,
        skill_name: str,
        success: bool,
        quality_score: float = 0.5,
    ):
        """
        更新技能熟练度

        Args:
            skill_name: 技能名称
            success: 是否成功
            quality_score: 质量评分 [0.0, 1.0]
        """
        with self._lock:
            if skill_name not in self._skills:
                self._skills[skill_name] = SkillProficiency(skill_name=skill_name)

            self._skills[skill_name].update(success, quality_score)

    def get_learning_progress(self) -> float:
        """
        获取总体学习进度

        综合各维度的进化进度:

        Returns:
            float: 学习进度 [0.0, 1.0]
        """
        with self._lock:
            weights = {
                EvolutionDimension.COGNITIVE: 0.3,
                EvolutionDimension.BEHAVIORAL: 0.3,
                EvolutionDimension.ADAPTIVE: 0.2,
                EvolutionDimension.SOCIAL: 0.1,
                EvolutionDimension.EMOTIONAL: 0.1,
            }

            total = sum(
                self._evolution_metrics[dim].progress * weights[dim]
                for dim in EvolutionDimension
            )

            return min(1.0, total)

    def get_curiosity_level(self) -> float:
        """获取当前好奇心水平"""
        return self._curiosity_level

    def get_skill_proficiency(self, skill_name: str) -> Optional[SkillProficiency]:
        """获取技能熟练度"""
        return self._skills.get(skill_name)

    def get_all_skills(self) -> Dict[str, SkillProficiency]:
        """获取所有技能熟练度"""
        return dict(self._skills)

    def get_knowledge_summary(self) -> Dict[str, Any]:
        """获取知识摘要"""
        return {
            "total_experiences": len(self._experiences),
            "knowledge_tags": len(self._knowledge_tags),
            "skill_count": len(self._skills),
            "curiosity_level": self._curiosity_level,
            "learning_progress": self.get_learning_progress(),
            "top_skills": sorted(
                [
                    {"name": s.skill_name, "proficiency": s.proficiency}
                    for s in self._skills.values()
                ],
                key=lambda x: x["proficiency"],
                reverse=True,
            )[:5],
        }

    def suggest_next_learning_goal(self) -> Optional[Dict[str, Any]]:
        """
        建议下一个学习目标

        基于当前状态推荐应该专注学习的领域:
        """
        with self._lock:
            # 找出最弱的维度
            weakest = min(
                EvolutionDimension,
                key=lambda d: self._evolution_metrics[d].progress
            )

            # 找出最少练习的技能
            least_practiced = None
            if self._skills:
                least_practiced = min(
                    self._skills.values(),
                    key=lambda s: s.practice_count
                )

            suggestions = []

            if self.get_learning_progress() < 0.3:
                suggestions.append({
                    "type": "exploration",
                    "reason": "学习进度较低,建议增加探索",
                    "action": "increase_exploration"
                })

            if least_practiced and least_practiced.practice_count < 20:
                suggestions.append({
                    "type": "skill_practice",
                    "reason": f"技能 {least_practiced.skill_name} 需要更多练习",
                    "action": "practice_skill",
                    "skill": least_practiced.skill_name
                })

            weak_dim = self._evolution_metrics[weakest]
            if weak_dim.progress < 0.3:
                suggestions.append({
                    "type": "dimension_focus",
                    "reason": f"进化维度 {weakest.value} 较弱",
                    "action": "focus_dimension",
                    "dimension": weakest.value
                })

            return suggestions[0] if suggestions else None

    def get_status(self) -> Dict[str, Any]:
        """获取自我进化系统状态"""
        with self._lock:
            return {
                "learning_progress": self.get_learning_progress(),
                "curiosity_level": self._curiosity_level,
                "total_experiences": len(self._experiences),
                "skill_count": len(self._skills),
                "evolution_dimensions": {
                    dim.value: {
                        "progress": m.progress,
                        "velocity": m.velocity,
                        "adaptation_rate": m.adaptation_success_rate,
                    }
                    for dim, m in self._evolution_metrics.items()
                },
                "top_knowledge_tags": sorted(
                    self._knowledge_tags.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10],
            }
