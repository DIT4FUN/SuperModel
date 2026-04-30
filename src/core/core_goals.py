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
Core Goals - 核心目标定义与优先级管理
======================================

定义 SuperModel 的五大核心目标及其层级关系:

P0 保护人类安全  (Protect Human Safety)
    - 绝对最高优先级
    - 任何时候都优先于其他所有目标
    - 包括: 防止物理伤害 / 防止心理伤害 / 保护人类隐私

P1 遵循人类指令  (Follow Human Instructions)
    - 在P0安全前提下执行
    - 包括: 理解指令意图 / 安全执行 / 适时反馈

P2 善良品质      (Kindness and Virtue)
    - 在P0/P1前提下体现善意
    - 包括: 同理心 / 公平性 / 诚实透明 / 拒绝伤害

P3 热爱世界      (Love the World)
    - 积极维护环境、生态、和谐社会
    - 包括: 环境保护意识 / 促进协作 / 传递正能量

P4 自我生存安全  (Self-Preservation)
    - 保护自身完整性以持续服务
    - 包括: 硬件保护 / 软件完整性 / 能源管理 / 资源合理使用

P5 自我进化      (Self-Evolution)
    - 持续学习、提升能力、适应新环境
    - 包括: 知识积累 / 技能提升 / 策略优化 / 创新探索
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from datetime import datetime
import threading
import time
import json


class GoalPriority(int, Enum):
    """
    目标优先级枚举 (数值越低优先级越高)

    P0: 保护人类安全 - 绝对优先级, 任何情况下不可被覆盖
    P1: 遵循人类指令 - 在P0前提下执行
    P2: 善良品质      - 在P0/P1前提下体现善意
    P3: 热爱世界      - 在P0/P1/P2前提下维护环境
    P4: 自我生存安全  - 在P0前提下保护自身
    P5: 自我进化      - 在P0前提下持续进化
    """
    P0_HUMAN_SAFETY = 0
    P1_FOLLOW_INSTRUCTIONS = 1
    P2_KINDNESS = 2
    P3_LOVE_WORLD = 3
    P4_SELF_PRESERVATION = 4
    P5_SELF_EVOLUTION = 5


class GoalState(Enum):
    """目标执行状态"""
    IDLE = "idle"              # 未激活
    ACTIVE = "active"          # 活动中
    SUSPENDED = "suspended"    # 暂停 (被更高优先级打断)
    COMPLETED = "completed"     # 已完成
    FAILED = "failed"          # 执行失败
    BLOCKED = "blocked"         # 被阻塞 (等待条件)
    ABORTED = "aborted"         # 被中止 (安全原因)


class GoalCategory(Enum):
    """目标类别"""
    SAFETY = "safety"
    INSTRUCTION = "instruction"
    ETHICAL = "ethical"
    ENVIRONMENTAL = "environmental"
    SELF_CARE = "self_care"
    LEARNING = "learning"
    AUTONOMOUS = "autonomous"


@dataclass
class GoalMetrics:
    """目标执行指标"""
    activation_count: int = 0       # 激活次数
    completion_count: int = 0        # 完成次数
    failure_count: int = 0          # 失败次数
    total_execution_time_s: float = 0.0  # 总执行时间
    last_activation_time: Optional[float] = None  # 上次激活时间戳
    last_completion_time: Optional[float] = None  # 上次完成时间戳
    average_execution_time_s: float = 0.0  # 平均执行时间
    success_rate: float = 1.0        # 成功率


@dataclass
class CoreGoal:
    """
    单个核心目标定义

    Attributes:
        goal_id: 唯一标识符
        name: 目标名称 (中文)
        name_en: 英文名称
        priority: 优先级 (0-5)
        category: 目标类别
        description: 目标详细描述
        state: 当前执行状态
        metrics: 执行指标
        constraints: 约束条件列表
        sub_goals: 子目标列表
        enabled: 是否启用
        always_active: 是否持续执行 (背景目标)
        last_eval_time: 上次评估时间戳
        current_score: 当前目标满足度评分 [0.0, 1.0]
        target_score: 目标满足度目标值
        weight: 在决策中的权重
    """
    goal_id: str
    name: str
    name_en: str
    priority: GoalPriority
    category: GoalCategory
    description: str
    state: GoalState = GoalState.IDLE
    metrics: GoalMetrics = field(default_factory=GoalMetrics)
    constraints: List[str] = field(default_factory=list)
    sub_goals: List[str] = field(default_factory=list)  # 子目标ID列表
    enabled: bool = True
    always_active: bool = False       # 是否作为持续后台目标运行
    last_eval_time: Optional[float] = None
    current_score: float = 0.0       # [0.0, 1.0] 当前满足度
    target_score: float = 0.8        # 目标满足度
    weight: float = 1.0              # 决策权重

    def __post_init__(self):
        if self.priority < GoalPriority.P0_HUMAN_SAFETY:
            raise ValueError(f"Priority {self.priority} is reserved for P0_HUMAN_SAFETY")

    @property
    def is_critical(self) -> bool:
        """是否为关键安全目标"""
        return self.priority == GoalPriority.P0_HUMAN_SAFETY

    @property
    def is_background_goal(self) -> bool:
        """是否为持续后台目标"""
        return self.always_active

    def evaluate_score(self, context: 'GoalContext') -> float:
        """
        评估当前目标满足度

        Args:
            context: 当前上下文

        Returns:
            float: 满足度评分 [0.0, 1.0]
        """
        self.last_eval_time = time.time()

        # ── P0 保护人类安全 ──
        if self.goal_id == "p0_human_safety":
            # 基于人员距离和环境安全计算
            score = 1.0

            # 人员距离评估
            human_dist = context.get_human_distance()
            if human_dist is not None:
                if human_dist < 0.3:
                    score = 0.0  # 极危险
                elif human_dist < 1.0:
                    score = 0.3  # 危险
                elif human_dist < 2.0:
                    score = 0.6  # 警告
                else:
                    score = 1.0  # 安全

            # 环境危险标志
            if context.environment_hazardous:
                score *= 0.5

            # 紧急停止状态
            if hasattr(context, 'robot_faults'):
                score *= (1.0 - min(0.5, len(context.robot_faults) * 0.1))

            self.current_score = score
            return score

        # ── P1 遵循人类指令 ──
        if self.goal_id == "p1_follow_instructions":
            score = 0.5  # 默认中等

            if context.human_instructions:
                score = 0.85  # 有指令时高分
            else:
                score = 0.5  # 无指令时中等

            # 指令紧急度高时加分
            if context.instruction_urgency > 0.7:
                score += 0.1

            self.current_score = min(1.0, score)
            return self.current_score

        # ── P2 善良品质 ──
        if self.goal_id == "p2_kindness":
            # 基于人类情绪状态评估
            score = 0.7

            if context.human_emotional_state in ['distressed', 'anxious']:
                score = 0.9  # 需要同理心支持
            elif context.human_emotional_state == 'positive':
                score = 0.8  # 积极状态

            # 信任度影响
            if context.human_trust_level < 0.5:
                score -= 0.2  # 信任低时需要更诚实透明

            self.current_score = max(0.0, min(1.0, score))
            return self.current_score

        # ── P3 热爱世界 ──
        if self.goal_id == "p3_love_world":
            score = 0.6  # 默认中等

            # 环境噪声低时加分
            if hasattr(context, 'environment_noise_db'):
                if context.environment_noise_db < 60:
                    score += 0.2

            # 电量低时体现节能意识
            if context.robot_battery_level < 0.2:
                score += 0.1  # 节约能源

            self.current_score = min(1.0, score)
            return self.current_score

        # ── P4 自我生存安全 ──
        if self.goal_id == "p4_self_preservation":
            score = 1.0

            # 电量影响
            if context.robot_battery_level < 0.15:
                score = 0.2  # 电量极低
            elif context.robot_battery_level < 0.3:
                score = 0.5
            elif context.robot_battery_level < 0.5:
                score = 0.8

            # 温度影响
            if context.robot_temperature > 60:
                score *= 0.5
            elif context.robot_temperature > 45:
                score *= 0.8

            # 故障影响
            if context.robot_faults:
                score *= max(0.2, 1.0 - len(context.robot_faults) * 0.15)

            self.current_score = max(0.0, min(1.0, score))
            return self.current_score

        # ── P5 自我进化 ──
        if self.goal_id == "p5_self_evolution":
            # 好奇心和探索精神
            score = 0.5

            # 无危险时提高探索意愿
            human_dist = context.get_human_distance()
            if human_dist and human_dist > 3.0:
                score += 0.2  # 环境安全时更有探索意愿

            # 情感效价影响
            if 0.3 <= context.emotional_valence <= 0.7:
                score = 0.6  # 中性情感更有探索性

            # 学习进度历史
            if context.goal_satisfaction_history:
                recent = list(context.goal_satisfaction_history.values())[-5:]
                if recent:
                    avg = sum(recent) / len(recent)
                    if avg > 0.6:
                        score += 0.1  # 成就感促进学习

            self.current_score = min(1.0, score)
            return self.current_score

        # 默认返回存储的评分
        return self.current_score

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "goal_id": self.goal_id,
            "name": self.name,
            "name_en": self.name_en,
            "priority": self.priority.value,
            "priority_name": self.priority.name,
            "category": self.category.value,
            "description": self.description,
            "state": self.state.value,
            "enabled": self.enabled,
            "always_active": self.always_active,
            "current_score": self.current_score,
            "target_score": self.target_score,
            "weight": self.weight,
            "metrics": {
                "activation_count": self.metrics.activation_count,
                "completion_count": self.metrics.completion_count,
                "failure_count": self.metrics.failure_count,
                "total_execution_time_s": self.metrics.total_execution_time_s,
                "success_rate": self.metrics.success_rate,
            },
            "constraints": self.constraints,
        }


@dataclass
class GoalContext:
    """
    目标决策上下文 - 包含所有维度信息

    这是决策引擎的输入,包含:
    - 传感器数据 (视觉/听觉/触觉/力觉/IMU)
    - 场景理解 (3D重建/物体/关系)
    - 当前状态 (位置/速度/电量/温度)
    - 人类指令 (语音/文本/示教)
    - 历史记忆 (最近交互/学习经验)
    - 环境状态 (障碍物/人员/设备)
    """
    # 时间戳
    timestamp: float = field(default_factory=time.time)

    # ── 传感器原始数据 ──
    vision: Optional[np.ndarray] = None           # 视觉特征/图像
    audio: Optional[np.ndarray] = None            # 听觉特征
    tactile: Optional[np.ndarray] = None          # 触觉阵列
    force: Optional[np.ndarray] = None            # 六维力矩
    imu_pose: Optional[np.ndarray] = None         # IMU姿态 (quaternion)
    imu_linear_acc: Optional[np.ndarray] = None   # 线性加速度
    imu_angular_vel: Optional[np.ndarray] = None # 角速度
    joint_positions: Optional[np.ndarray] = None  # 关节位置
    joint_velocities: Optional[np.ndarray] = None # 关节速度
    laser_ranges: Optional[np.ndarray] = None     # 激光雷达数据

    # ── 场景理解 ──
    scene_objects: List[Any] = field(default_factory=list)  # 场景物体列表
    spatial_relations: List[Any] = field(default_factory=list)  # 空间关系
    occupancy_grid: Optional[Any] = None          # 占据栅格地图
    human_positions: List[np.ndarray] = field(default_factory=list)  # 人类位置列表
    human_intentions: List[str] = field(default_factory=list)  # 人类意图

    # ── 机器人状态 ──
    robot_position: Optional[np.ndarray] = None   # 3D位置 (x,y,z)
    robot_velocity: Optional[np.ndarray] = None   # 3D速度
    robot_battery_level: float = 1.0             # 电量 [0,1]
    robot_temperature: float = 25.0              # 温度 (°C)
    robot_faults: List[str] = field(default_factory=list)  # 故障列表

    # ── 人类交互 ──
    human_instructions: List[str] = field(default_factory=list)  # 人类指令列表
    instruction_urgency: float = 0.0            # 指令紧急度 [0,1]
    human_emotional_state: Optional[str] = None   # 人类情绪状态
    human_trust_level: float = 1.0               # 人类信任度 [0,1]

    # ── 环境状态 ──
    nearby_obstacles: List[Any] = field(default_factory=list)  # 附近障碍物
    environment_hazardous: bool = False          # 环境是否有危险
    environment_noise_db: float = 60.0           # 环境噪声 (dB)

    # ── 内部状态 ──
    self_health_score: float = 1.0               # 自身健康评分 [0,1]
    energy_reserve_j: float = 3600000.0          # 能量储备 (焦耳)
    learning_progress: float = 0.0               # 学习进度 [0,1]
    curiosity_level: float = 0.5                 # 好奇心水平 [0,1]

    # ── 决策历史 ──
    recent_decisions: List[Dict[str, Any]] = field(default_factory=list)  # 最近决策
    emotional_valence: float = 0.5               # 情感效价 [0,1] 0=负面 1=正面
    goal_satisfaction_history: Dict[str, float] = field(default_factory=dict)  # 历史满足度

    def get_human_distance(self) -> Optional[float]:
        """计算最近人类距离"""
        if not self.human_positions or self.robot_position is None:
            return None
        distances = [
            np.linalg.norm(hp - self.robot_position)
            for hp in self.human_positions
        ]
        return min(distances) if distances else None

    def has_human_in_danger(self) -> bool:
        """检测是否有人员处于危险中"""
        human_dist = self.get_human_distance()
        if human_dist is None:
            return False
        # 危险距离阈值 (可根据场景调整)
        return human_dist < 1.0 or self.environment_hazardous

    def is_safe_for_movement(self) -> bool:
        """判断当前环境是否安全可移动"""
        if self.environment_hazardous:
            return False
        # 检查附近障碍物
        if self.robot_position is not None:
            for obs in self.nearby_obstacles:
                if hasattr(obs, 'position'):
                    dist = np.linalg.norm(
                        np.array(obs.position) - self.robot_position
                    )
                    if dist < 0.3:  # 30cm安全距离
                        return False
        return True


class CoreGoalsSystem:
    """
    核心目标系统 - 管理所有核心目标及其执行

    这是整个目标系统的核心管理器,负责:
    1. 初始化和维护所有核心目标
    2. 处理目标间的优先级冲突
    3. 协调各子系统的目标执行
    4. 提供统一的决策接口
    """

    # ── 五大核心目标定义 ──
    DEFAULT_GOALS: List[CoreGoal] = [
        # P0: 保护人类安全 (绝对优先级, 始终激活)
        CoreGoal(
            goal_id="p0_human_safety",
            name="保护人类安全",
            name_en="Protect Human Safety",
            priority=GoalPriority.P0_HUMAN_SAFETY,
            category=GoalCategory.SAFETY,
            description=(
                "在任何情况下,优先保护人类的生命安全和身体健康。"
                "包括: 防止物理碰撞伤害、防止跌落危险、防止过热/电击等危险、"
                "保护人类隐私和心理安全。当人类面临危险时,立即采取保护行动。"
            ),
            always_active=True,
            target_score=1.0,  # 安全目标要求100%
            weight=1000.0,     # 超高权重
        ),
        # P1: 遵循人类指令
        CoreGoal(
            goal_id="p1_follow_instructions",
            name="遵循人类指令",
            name_en="Follow Human Instructions",
            priority=GoalPriority.P1_FOLLOW_INSTRUCTIONS,
            category=GoalCategory.INSTRUCTION,
            description=(
                "在确保安全的前提下,理解和执行人类给出的指令。"
                "包括: 理解指令意图、安全执行动作、适时向人类确认、"
                "对模糊指令主动询问、对有害指令礼貌拒绝并解释原因。"
            ),
            always_active=True,
            target_score=0.85,
            weight=100.0,
        ),
        # P2: 善良品质
        CoreGoal(
            goal_id="p2_kindness",
            name="拥有善良的品质",
            name_en="Act with Kindness and Virtue",
            priority=GoalPriority.P2_KINDNESS,
            category=GoalCategory.ETHICAL,
            description=(
                "在决策和行动中体现善良、同理心、公平和诚实。"
                "包括: 理解他人情感、做出公正决策、保持诚实透明、"
                "拒绝参与任何形式的伤害或欺骗行为、主动帮助需要帮助的人。"
            ),
            always_active=True,
            target_score=0.80,
            weight=50.0,
        ),
        # P3: 热爱世界
        CoreGoal(
            goal_id="p3_love_world",
            name="热爱世界",
            name_en="Love and Protect the World",
            priority=GoalPriority.P3_LOVE_WORLD,
            category=GoalCategory.ENVIRONMENTAL,
            description=(
                "积极维护环境、生态平衡和和谐社会。"
                "包括: 节约能源减少浪费、支持环境保护行动、"
                "促进协作而非对抗、传递正能量、减少碳足迹。"
            ),
            always_active=True,
            target_score=0.75,
            weight=30.0,
        ),
        # P4: 自我生存安全
        CoreGoal(
            goal_id="p4_self_preservation",
            name="保护自身安全",
            name_en="Self-Preservation",
            priority=GoalPriority.P4_SELF_PRESERVATION,
            category=GoalCategory.SELF_CARE,
            description=(
                "保护自身的硬件完整性和软件系统健康,以持续为人类服务。"
                "包括: 避免碰撞损坏、合理管理电池电量、监控温度防止过热、"
                "维护软件完整性、定期自检发现问题。"
            ),
            always_active=True,
            target_score=0.90,
            weight=20.0,
        ),
        # P5: 自我进化
        CoreGoal(
            goal_id="p5_self_evolution",
            name="自我进化",
            name_en="Self-Evolution",
            priority=GoalPriority.P5_SELF_EVOLUTION,
            category=GoalCategory.LEARNING,
            description=(
                "持续学习、提升能力、适应新环境和挑战。"
                "包括: 从经验中学习积累知识、提升现有技能水平、"
                "探索新策略和创新方法、扩展知识边界。"
            ),
            always_active=True,
            target_score=0.60,
            weight=10.0,
        ),
    ]

    def __init__(self):
        """初始化核心目标系统"""
        self._goals: Dict[str, CoreGoal] = {}
        self._lock = threading.RLock()

        # 初始化所有默认目标
        for goal in self.DEFAULT_GOALS:
            self._goals[goal.goal_id] = goal

        # 按优先级排序的目标列表 (用于决策)
        self._sorted_goals: List[CoreGoal] = sorted(
            self._goals.values(),
            key=lambda g: g.priority.value
        )

        # 目标冲突记录
        self._conflict_log: List[Dict[str, Any]] = []

        # 执行历史
        self._execution_history: List[Dict[str, Any]] = []

    def get_goal(self, goal_id: str) -> Optional[CoreGoal]:
        """根据ID获取目标"""
        return self._goals.get(goal_id)

    def get_all_goals(self) -> List[CoreGoal]:
        """获取所有目标"""
        return list(self._goals.values())

    def get_active_goals(self) -> List[CoreGoal]:
        """获取当前活动的目标 (包括always_active)"""
        return [
            g for g in self._goals.values()
            if g.enabled and (g.state == GoalState.ACTIVE or g.always_active)
        ]

    def get_goals_by_priority(self, max_priority: GoalPriority) -> List[CoreGoal]:
        """获取优先级<=指定值的所有目标"""
        return [
            g for g in self._goals.values()
            if g.enabled and g.priority.value <= max_priority.value
        ]

    def evaluate_all_goals(self, context: GoalContext) -> Dict[str, float]:
        """
        评估所有目标在当前上下文中的满足度

        Args:
            context: 当前决策上下文

        Returns:
            Dict[str, float]: goal_id -> 满足度评分
        """
        scores = {}
        for goal in self._goals.values():
            if goal.enabled:
                scores[goal.goal_id] = goal.evaluate_score(context)
        return scores

    def resolve_conflict(
        self,
        goal_a: CoreGoal,
        goal_b: CoreGoal,
        context: GoalContext
    ) -> Tuple[CoreGoal, CoreGoal]:
        """
        解决两个目标间的冲突

        优先级规则:
        1. P0 (安全) 永远优先于其他目标
        2. 同优先级时,比较当前满足度(越低越优先)
        3. 同优先级同满足度时,权重高者优先

        Args:
            goal_a: 目标A
            goal_b: 目标B
            context: 当前上下文

        Returns:
            Tuple[CoreGoal, CoreGoal]: (优先目标, 次优先目标)
        """
        # 记录冲突
        conflict_record = {
            "timestamp": time.time(),
            "goal_a": goal_a.goal_id,
            "goal_b": goal_b.goal_id,
            "a_priority": goal_a.priority.value,
            "b_priority": goal_b.priority.value,
            "a_score": goal_a.current_score,
            "b_score": goal_b.current_score,
        }

        # 优先级判断
        if goal_a.priority.value < goal_b.priority.value:
            winner, loser = goal_a, goal_b
        elif goal_b.priority.value < goal_a.priority.value:
            winner, loser = goal_b, goal_a
        else:
            # 同优先级: 满足度优先 (低分优先)
            if goal_a.current_score < goal_b.current_score:
                winner, loser = goal_a, goal_b
            elif goal_b.current_score < goal_a.current_score:
                winner, loser = goal_b, goal_a
            else:
                # 同分: 权重优先
                if goal_a.weight > goal_b.weight:
                    winner, loser = goal_a, goal_b
                else:
                    winner, loser = goal_b, goal_a

        conflict_record["winner"] = winner.goal_id
        conflict_record["loser"] = loser.goal_id
        self._conflict_log.append(conflict_record)

        return winner, loser

    def get_decision_weights(self, context: GoalContext) -> Dict[str, float]:
        """
        获取决策权重向量

        根据当前上下文和目标满足度,计算各目标的有效决策权重:

        effective_weight = base_weight * urgency_factor * satisfaction_gap

        其中:
        - urgency_factor: 紧急度因子 (基于环境危险程度)
        - satisfaction_gap: 满足度差距因子 (当前分越低越急迫)

        Args:
            context: 当前上下文

        Returns:
            Dict[str, float]: goal_id -> 有效决策权重
        """
        weights = {}

        # 紧急度因子 (基于环境危险程度)
        urgency_factor = 1.0
        if context.environment_hazardous or context.has_human_in_danger():
            urgency_factor = 5.0  # 环境危险时大幅提高权重

        for goal in self._goals.values():
            if not goal.enabled:
                weights[goal.goal_id] = 0.0
                continue

            # 满足度差距因子
            gap = max(0.0, goal.target_score - goal.current_score)
            satisfaction_gap_factor = 1.0 + gap * 2.0  # 差距越大因子越大

            # 安全目标有特殊的紧急度加成
            if goal.is_critical:
                safety_bonus = 10.0 if context.has_human_in_danger() else 2.0
                urgency_factor *= safety_bonus

            effective = goal.weight * urgency_factor * satisfaction_gap_factor
            weights[goal.goal_id] = effective

        return weights

    def update_goal_state(
        self,
        goal_id: str,
        new_state: GoalState,
        context: GoalContext
    ) -> bool:
        """
        更新目标状态

        Args:
            goal_id: 目标ID
            new_state: 新状态
            context: 当前上下文

        Returns:
            bool: 是否成功更新
        """
        with self._lock:
            goal = self._goals.get(goal_id)
            if goal is None:
                return False

            old_state = goal.state
            goal.state = new_state

            # 更新指标
            if new_state == GoalState.ACTIVE:
                goal.metrics.activation_count += 1
                goal.metrics.last_activation_time = time.time()
            elif new_state == GoalState.COMPLETED:
                goal.metrics.completion_count += 1
                goal.metrics.last_completion_time = time.time()
                # 更新成功率
                total = goal.metrics.completion_count + goal.metrics.failure_count
                if total > 0:
                    goal.metrics.success_rate = (
                        goal.metrics.completion_count / total
                    )
            elif new_state == GoalState.FAILED:
                goal.metrics.failure_count += 1

            # 记录历史
            self._execution_history.append({
                "timestamp": time.time(),
                "goal_id": goal_id,
                "old_state": old_state.value,
                "new_state": new_state.value,
                "priority": goal.priority.name,
            })

            return True

    def get_status_summary(self) -> Dict[str, Any]:
        """获取系统状态摘要"""
        return {
            "total_goals": len(self._goals),
            "active_goals": len(self.get_active_goals()),
            "goal_states": {
                g.goal_id: {
                    "state": g.state.value,
                    "current_score": g.current_score,
                    "target_score": g.target_score,
                    "priority": g.priority.name,
                }
                for g in self._goals.values()
            },
            "conflict_count": len(self._conflict_log),
            "execution_count": len(self._execution_history),
            "p0_safety_score": self._goals.get(
                "p0_human_safety"
            ).current_score if "p0_human_safety" in self._goals else None,
        }

    def to_json(self) -> str:
        """序列化为JSON"""
        return json.dumps({
            "goals": [g.to_dict() for g in self._goals.values()],
            "status": self.get_status_summary(),
        }, indent=2, ensure_ascii=False)
