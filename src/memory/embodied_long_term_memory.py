"""
embodied_long_term_memory.py - 具身智能长期记忆系统
SuperModel 超模态大模型具身智能系统

具身记忆增强:
- 具身经验记忆 (Embodied Experience Memory)
- 场景-记忆关联索引 (Scene-Memory Association)
- 技能记忆与AGV等级感知 (Skill Memory with Grade Awareness)
- 基于记忆的任务预测 (Memory-based Task Prediction)
- 经验压缩与知识蒸馏 (Experience Compression)
- 多模态记忆检索 (Multimodal Memory Retrieval)
"""

from __future__ import annotations

import time
import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from enum import Enum
from collections import defaultdict, deque
import numpy as np

if TYPE_CHECKING:
    from ..embodied.scene_intelligence import SceneType
    from ..embodied.embodied_skill import EmbodiedSkill, EmbodiedSkillRegistry

logger = logging.getLogger(__name__)

__all__ = [
    'EmbodiedExperienceType',
    'EmbodiedMemoryTag',
    'EmbodiedExperience',
    'SceneMemoryIndex',
    'SkillMemoryRecord',
    'AGVGradeAwareMemory',
    'EmbodiedLongTermMemory',
    'ExperienceCompressor',
    'MemoryBasedTaskPredictor',
    'create_embodied_long_term_memory',
]


class EmbodiedExperienceType(Enum):
    """具身经验类型"""
    NAVIGATION = "navigation"           # 导航经验
    GRASP = "grasp"                     # 抓取经验
    MANIPULATION = "manipulation"       # 操作经验
    COLLABORATION = "collaboration"      # 协同经验
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"  # 避障经验
    EMERGENCY = "emergency"              # 紧急处理经验
    SCENE_ADAPTATION = "scene_adaptation"  # 场景适应经验
    SKILL_LEARNING = "skill_learning"    # 技能学习经验
    FAILURE_RECOVERY = "failure_recovery"  # 故障恢复经验


class EmbodiedMemoryTag(Enum):
    """具身记忆标签"""
    # 场景标签
    WAREHOUSE = "warehouse"
    FACTORY = "factory"
    HOSPITAL = "hospital"
    RESTAURANT = "restaurant"
    OUTDOOR = "outdoor"
    LABORATORY = "laboratory"
    OFFICE = "office"
    HOME = "home"
    # 行为标签
    NAVIGATE = "navigate"
    GRASP = "grasp"
    PLACE = "place"
    LIFT = "lift"
    CARRY = "carry"
    COLLABORATE = "collaborate"
    AVOID = "avoid"
    RECOVER = "recover"
    # 结果标签
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    ABORTED = "aborted"


@dataclass
class EmbodiedExperience:
    """具身经验条目
    
    存储一次完整的具身智能经验，包括:
    - 场景上下文
    - 传感器数据摘要
    - 执行的动作序列
    - 最终结果
    - 学习到的知识
    """
    experience_id: str
    experience_type: EmbodiedExperienceType
    scene_type: str
    
    # 时空上下文
    start_timestamp: float
    end_timestamp: float
    duration_seconds: float
    
    # 环境状态摘要
    initial_state: Dict[str, Any]    # 初始状态 (位置/负载/电量/传感器读数)
    final_state: Dict[str, Any]     # 最终状态
    
    # 动作序列 (压缩表示)
    action_sequence: List[Dict[str, Any]]  # [{"action": str, "params": dict, "outcome": str}, ...]
    
    # 结果评估
    outcome: str                      # "success" / "failure" / "partial" / "aborted"
    outcome_score: float             # 0.0-1.0, 任务完成质量
    efficiency_score: float          # 0.0-1.0, 资源利用效率
    safety_score: float              # 0.0-1.0, 安全程度
    
    # AGV配置
    agv_grade: str                   # "S" / "M" / "L" / "XL" / "XXL"
    sensor_config: Dict[str, Any]   # 使用的传感器配置
    
    # 学习到的知识
    learned_patterns: List[str]      # 习得的模式/规则
    failure_reasons: List[str]       # 失败原因分析
    improvement_hints: List[str]     # 改进建议
    
    # 标签
    tags: List[str] = field(default_factory=list)
    
    # 访问统计
    access_count: int = 0
    last_access_time: float = field(default_factory=time.time)
    importance_score: float = 0.5    # 0.0-1.0, 重要性
    
    # 向量嵌入 (用于相似度检索)
    embedding: Optional[np.ndarray] = None
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 关联经验ID列表
    related_experience_ids: List[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        return self.outcome == "success" and self.outcome_score >= 0.8
    
    @property
    def total_reward(self) -> float:
        """综合奖励分数"""
        return (self.outcome_score * 0.5 + 
                self.efficiency_score * 0.3 + 
                self.safety_score * 0.2)
    
    @property
    def age_days(self) -> float:
        """经验年龄(天)"""
        return (time.time() - self.start_timestamp) / 86400.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experience_id": self.experience_id,
            "experience_type": self.experience_type.value,
            "scene_type": self.scene_type,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "duration_seconds": self.duration_seconds,
            "initial_state": self.initial_state,
            "final_state": self.final_state,
            "action_sequence": self.action_sequence,
            "outcome": self.outcome,
            "outcome_score": self.outcome_score,
            "efficiency_score": self.efficiency_score,
            "safety_score": self.safety_score,
            "agv_grade": self.agv_grade,
            "sensor_config": self.sensor_config,
            "learned_patterns": self.learned_patterns,
            "failure_reasons": self.failure_reasons,
            "improvement_hints": self.improvement_hints,
            "tags": self.tags,
            "access_count": self.access_count,
            "last_access_time": self.last_access_time,
            "importance_score": self.importance_score,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "EmbodiedExperience":
        data = data.copy()
        data["experience_type"] = EmbodiedExperienceType(data["experience_type"])
        data["embedding"] = (np.array(data["embedding"]) 
                             if data.get("embedding") is not None else None)
        return cls(**data)


class SceneMemoryIndex:
    """场景-记忆关联索引
    
    为每个场景类型维护一个记忆索引，
    支持按场景快速检索相关经验。
    """
    
    def __init__(self):
        # 场景类型 -> 经验ID列表 (按时间排序, 最新在前)
        self._scene_experience_index: Dict[str, deque] = defaultdict(deque)
        # 场景类型 -> 标签 -> 经验ID集合
        self._scene_tag_index: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        # 场景类型 -> 经验类型 -> 经验ID集合
        self._scene_type_index: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        # 经验ID -> 场景类型
        self._experience_scene: Dict[str, str] = {}
        # 互斥锁
        self._lock = threading.RLock()
        # 最大索引容量
        self._max_per_scene = 1000
    
    def add_experience(self, experience: EmbodiedExperience) -> None:
        """添加经验到索引"""
        with self._lock:
            scene = experience.scene_type
            exp_id = experience.experience_id
            
            # 添加到场景索引
            if exp_id not in self._scene_experience_index[scene]:
                self._scene_experience_index[scene].appendleft(exp_id)
                # 限制大小
                if len(self._scene_experience_index[scene]) > self._max_per_scene:
                    old_id = self._scene_experience_index[scene].pop()
                    self._remove_from_indexes(old_id)
            
            # 添加到标签索引
            for tag in experience.tags:
                self._scene_tag_index[scene][tag].add(exp_id)
            
            # 添加到经验类型索引
            self._scene_type_index[scene][experience.experience_type.value].add(exp_id)
            
            # 记录场景映射
            self._experience_scene[exp_id] = scene
    
    def _remove_from_indexes(self, exp_id: str) -> None:
        """从所有索引中移除经验"""
        scene = self._experience_scene.pop(exp_id, None)
        if scene is None:
            return
        
        # 从标签索引移除
        for tag_index in self._scene_tag_index[scene].values():
            tag_index.discard(exp_id)
        
        # 从类型索引移除
        for type_index in self._scene_type_index[scene].values():
            type_index.discard(exp_id)
    
    def get_by_scene(
        self, 
        scene_type: str, 
        limit: int = 50,
        experience_types: Optional[List[EmbodiedExperienceType]] = None,
        tags: Optional[List[str]] = None,
    ) -> List[str]:
        """获取指定场景的经验ID列表"""
        with self._lock:
            exp_ids = list(self._scene_experience_index.get(scene_type, deque()))
            
            # 按标签过滤
            if tags:
                tag_filtered: Set[str] = set(exp_ids)
                for tag in tags:
                    tag_filtered &= self._scene_tag_index[scene_type].get(tag, set())
                exp_ids = list(tag_filtered)
            
            # 按经验类型过滤
            if experience_types:
                type_filtered: Set[str] = set(exp_ids)
                for et in experience_types:
                    type_filtered |= self._scene_type_index[scene_type].get(et.value, set())
                exp_ids = list(type_filtered)
            
            return exp_ids[:limit]
    
    def get_scene_stats(self, scene_type: str) -> Dict[str, Any]:
        """获取场景记忆统计"""
        with self._lock:
            return {
                "total_experiences": len(self._scene_experience_index.get(scene_type, [])),
                "tag_count": sum(1 for s in self._scene_tag_index[scene_type].values() for _ in s),
                "experience_types": {
                    str(et): len(ids) 
                    for et, ids in self._scene_type_index[scene_type].items()
                },
            }


@dataclass
class SkillMemoryRecord:
    """技能记忆记录
    
    记录某个技能在特定场景/AGV配置下的执行历史，
    用于技能可靠性和适应性评估。
    """
    skill_id: str
    skill_name: str
    scene_type: str
    agv_grade: str
    
    # 执行统计
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    avg_execution_time: float = 0.0
    
    # 质量指标
    avg_success_rate: float = 0.0
    avg_quality_score: float = 0.0
    
    # 场景适配度
    scene_adaptation_score: float = 0.0   # 0.0-1.0, 该场景下的适配程度
    
    # 学习进度
    learning_progress: float = 0.0         # 0.0-1.0, 学习曲线上当前位置
    learning_curve_points: List[Tuple[int, float]] = field(default_factory=list)
    
    # 时间戳
    first_execution_time: float = 0.0
    last_execution_time: float = 0.0
    last_success_time: float = 0.0
    
    # 经验ID列表
    related_experience_ids: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions
    
    @property
    def is_mastered(self) -> bool:
        """技能是否已掌握 (成功率 >= 95%)"""
        return self.success_rate >= 0.95 and self.total_executions >= 10
    
    @property
    def needs_relearning(self) -> bool:
        """技能是否需要重新学习 (30天未成功执行)"""
        if self.last_success_time == 0:
            return self.total_executions > 0
        return (time.time() - self.last_success_time) > 30 * 86400
    
    def update_from_experience(self, experience: EmbodiedExperience) -> None:
        """从具身经验更新技能记录"""
        self.total_executions += 1
        self.last_execution_time = experience.end_timestamp
        
        if self.first_execution_time == 0:
            self.first_execution_time = experience.start_timestamp
        
        if experience.success:
            self.successful_executions += 1
            self.last_success_time = experience.end_timestamp
        else:
            self.failed_executions += 1
        
        # 更新学习曲线
        self.learning_curve_points.append((self.total_executions, self.success_rate))
        if len(self.learning_curve_points) > 100:
            self.learning_curve_points = self.learning_curve_points[-100:]
        
        # 更新学习进度 (简单的线性进度模型)
        if self.total_executions >= 10:
            self.learning_progress = min(1.0, self.success_rate)
        
        # 更新平均值
        if self.total_executions > 0:
            self.avg_success_rate = self.success_rate
            self.avg_quality_score = (
                self.avg_quality_score * (self.total_executions - 1) + 
                experience.outcome_score
            ) / self.total_executions
        
        if experience.related_experience_ids:
            self.related_experience_ids.extend(experience.related_experience_ids)
            self.related_experience_ids = list(set(self.related_experience_ids))[-50:]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "skill_name": self.skill_name,
            "scene_type": self.scene_type,
            "agv_grade": self.agv_grade,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "avg_execution_time": self.avg_execution_time,
            "avg_success_rate": self.avg_success_rate,
            "avg_quality_score": self.avg_quality_score,
            "scene_adaptation_score": self.scene_adaptation_score,
            "learning_progress": self.learning_progress,
            "learning_curve_points": self.learning_curve_points,
            "first_execution_time": self.first_execution_time,
            "last_execution_time": self.last_execution_time,
            "last_success_time": self.last_success_time,
            "related_experience_ids": self.related_experience_ids,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "SkillMemoryRecord":
        return cls(**data)


class AGVGradeAwareMemory:
    """AGV等级感知记忆
    
    为不同AGV等级维护独立的记忆空间，
    支持跨等级经验迁移。
    """
    
    # AGV等级列表 (从小到大)
    GRADE_ORDER = ["S", "M", "L", "XL", "XXL"]
    GRADE_INDEX = {g: i for i, g in enumerate(GRADE_ORDER)}
    
    def __init__(self):
        # 等级 -> 场景 -> 经验类型 -> 经验列表
        self._grade_scene_exp: Dict[str, Dict[str, Dict[str, List[str]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        # 经验ID -> 等级
        self._exp_grade: Dict[str, str] = {}
        # 等级迁移映射表 (低等级经验可被高等级AGV使用)
        self._grade_transfer: Dict[str, Set[str]] = defaultdict(set)
        self._init_grade_transfer()
        self._lock = threading.RLock()
    
    def _init_grade_transfer(self) -> None:
        """初始化等级迁移映射"""
        for i, grade in enumerate(self.GRADE_ORDER):
            # 当前等级及以下等级的经验都可迁移
            for j in range(i + 1):
                self._grade_transfer[grade].add(self.GRADE_ORDER[j])
    
    def _can_transfer(self, exp_grade: str, target_grade: str) -> bool:
        """检查经验是否可以迁移到目标等级"""
        return exp_grade in self._grade_transfer[target_grade]
    
    def add_experience(self, exp_id: str, grade: str, scene: str, 
                       exp_type: EmbodiedExperienceType) -> None:
        """添加经验到等级感知存储"""
        with self._lock:
            self._grade_scene_exp[grade][scene][exp_type.value].append(exp_id)
            self._exp_grade[exp_id] = grade
    
    def get_for_grade(
        self, 
        target_grade: str, 
        scene: Optional[str] = None,
        exp_type: Optional[EmbodiedExperienceType] = None,
    ) -> List[str]:
        """获取适用于目标AGV等级的经验"""
        with self._lock:
            result = []
            for grade in self.GRADE_ORDER:
                if not self._can_transfer(grade, target_grade):
                    continue
                
                if scene:
                    if exp_type:
                        result.extend(
                            self._grade_scene_exp[grade][scene].get(exp_type.value, [])
                        )
                    else:
                        for et_list in self._grade_scene_exp[grade][scene].values():
                            result.extend(et_list)
                else:
                    for scene_data in self._grade_scene_exp[grade].values():
                        if exp_type:
                            result.extend(scene_data.get(exp_type.value, []))
                        else:
                            for et_list in scene_data.values():
                                result.extend(et_list)
            
            return result
    
    def get_transfer_benefit(self, from_grade: str, to_grade: str) -> float:
        """估算从源等级迁移到目标等级的效益 (0.0-1.0)"""
        from_idx = self.GRADE_INDEX.get(from_grade, 0)
        to_idx = self.GRADE_INDEX.get(to_grade, 0)
        
        if from_idx == to_idx:
            return 1.0
        
        # 等级差距越大，迁移效益越低
        gap = to_idx - from_idx
        return max(0.1, 1.0 - gap * 0.15)


class ExperienceCompressor:
    """经验压缩器
    
    将多个相似经验压缩为原型经验，
    减少存储开销同时保留核心知识。
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
    
    def compute_similarity(
        self, 
        exp1: EmbodiedExperience, 
        exp2: EmbodiedExperience,
    ) -> float:
        """计算两个经验的相似度"""
        if exp1.scene_type != exp2.scene_type:
            return 0.0
        if exp1.experience_type != exp2.experience_type:
            return 0.0
        
        score = 0.0
        # 场景匹配 (30%)
        score += 0.3
        # 经验类型匹配 (20%)
        score += 0.2
        # 结果相似度 (25%)
        if exp1.outcome == exp2.outcome:
            score += 0.25
        elif abs(exp1.outcome_score - exp2.outcome_score) < 0.2:
            score += 0.15
        # AGV等级接近度 (25%)
        g1 = AGVGradeAwareMemory.GRADE_INDEX.get(exp1.agv_grade, 0)
        g2 = AGVGradeAwareMemory.GRADE_INDEX.get(exp2.agv_grade, 0)
        grade_sim = 1.0 - abs(g1 - g2) / 4.0
        score += 0.25 * grade_sim
        
        return score
    
    def compress_experiences(
        self, 
        experiences: List[EmbodiedExperience],
    ) -> List[EmbodiedExperience]:
        """压缩相似经验列表,返回原型经验"""
        if len(experiences) <= 1:
            return experiences
        
        # 按类型分组
        by_type: Dict[Tuple[str, str], List[EmbodiedExperience]] = defaultdict(list)
        for exp in experiences:
            key = (exp.scene_type, exp.experience_type.value)
            by_type[key].append(exp)
        
        prototypes = []
        for exps in by_type.values():
            # 简单聚类: 按结果分组
            by_outcome: Dict[str, List[EmbodiedExperience]] = defaultdict(list)
            for exp in exps:
                by_outcome[exp.outcome].append(exp)
            
            for outcome, outcome_exps in by_outcome.items():
                if len(outcome_exps) == 1:
                    prototypes.append(outcome_exps[0])
                else:
                    # 合并为原型 (保留最高分经验)
                    best = max(outcome_exps, key=lambda e: e.total_reward)
                    # 合并标签
                    all_tags = set()
                    for exp in outcome_exps:
                        all_tags.update(exp.tags)
                    best.tags = list(all_tags)
                    # 增加重要性
                    best.importance_score = min(1.0, best.importance_score + 0.1)
                    prototypes.append(best)
        
        return prototypes


class MemoryBasedTaskPredictor:
    """基于记忆的任务结果预测器
    
    利用历史经验预测新任务的执行结果，
    辅助决策和任务规划。
    """
    
    def __init__(self):
        self._experience_base: Dict[str, EmbodiedExperience] = {}
        self._lock = threading.RLock()
    
    def register_experience(self, experience: EmbodiedExperience) -> None:
        """注册经验到预测库"""
        with self._lock:
            self._experience_base[experience.experience_id] = experience
    
    def predict_success(
        self,
        scene_type: str,
        exp_type: EmbodiedExperienceType,
        agv_grade: str,
        initial_state: Dict[str, Any],
    ) -> Tuple[float, float, List[str]]:
        """
        预测任务成功率
        
        Returns:
            (predicted_success_rate, confidence, similar_experience_ids)
        """
        with self._lock:
            # 查找相似经验
            similar = []
            for exp in self._experience_base.values():
                if (exp.scene_type == scene_type and 
                    exp.experience_type == exp_type and
                    self._matches_grade(exp.agv_grade, agv_grade)):
                    similar.append(exp)
            
            if not similar:
                return 0.5, 0.0, []
            
            # 加权平均预测
            total_weight = 0.0
            weighted_sum = 0.0
            for exp in similar:
                # 权重 = 重要性 * 时效衰减
                recency = max(0.1, 1.0 - exp.age_days / 90.0)  # 90天衰减
                weight = exp.importance_score * recency
                weighted_sum += exp.success * weight
                total_weight += weight
            
            if total_weight == 0:
                return 0.5, 0.0, []
            
            predicted_rate = weighted_sum / total_weight
            confidence = min(1.0, len(similar) / 10.0)  # 经验越多置信度越高
            similar_ids = [e.experience_id for e in similar[:5]]
            
            return predicted_rate, confidence, similar_ids
    
    def _matches_grade(self, exp_grade: str, target_grade: str) -> bool:
        """检查经验等级是否适用于目标AGV"""
        ei = AGVGradeAwareMemory.GRADE_INDEX.get(exp_grade, 0)
        ti = AGVGradeAwareMemory.GRADE_INDEX.get(target_grade, 0)
        return ei <= ti
    
    def get_failure_reasons(
        self, 
        scene_type: str, 
        exp_type: EmbodiedExperienceType,
    ) -> List[Tuple[str, int]]:
        """获取指定场景/任务类型的主要失败原因 (reason, count)"""
        with self._lock:
            reasons: Dict[str, int] = defaultdict(int)
            for exp in self._experience_base.values():
                if (exp.scene_type == scene_type and 
                    exp.experience_type == exp_type and
                    not exp.success):
                    for reason in exp.failure_reasons:
                        reasons[reason] += 1
            
            return sorted(reasons.items(), key=lambda x: -x[1])[:5]


class EmbodiedLongTermMemory:
    """
    具身智能长期记忆系统
    
    整合所有具身记忆功能:
    - 具身经验存储与检索
    - 场景-记忆关联索引
    - AGV等级感知记忆
    - 经验压缩与知识蒸馏
    - 任务结果预测
    """
    
    def __init__(
        self,
        storage_path: str = "memory_data/embodied",
        enable_compression: bool = True,
        enable_prediction: bool = True,
        max_experiences: int = 10000,
    ):
        self.storage_path = storage_path
        self.enable_compression = enable_compression
        self.enable_prediction = enable_prediction
        self.max_experiences = max_experiences
        
        # 核心组件
        self._scene_index = SceneMemoryIndex()
        self._grade_memory = AGVGradeAwareMemory()
        self._compressor = ExperienceCompressor()
        self._predictor = MemoryBasedTaskPredictor()
        
        # 经验存储
        self._experiences: Dict[str, EmbodiedExperience] = {}
        
        # 技能记忆存储
        self._skill_memories: Dict[str, SkillMemoryRecord] = {}
        
        # 统计
        self._stats = {
            "total_experiences": 0,
            "total_store_operations": 0,
            "compression_count": 0,
            "prediction_requests": 0,
        }
        
        # 互斥锁
        self._lock = threading.RLock()
        
        # 尝试加载持久化数据
        self._load()
    
    def store_experience(self, experience: EmbodiedExperience) -> str:
        """存储具身经验"""
        with self._lock:
            exp_id = experience.experience_id
            
            # 检查是否需要压缩
            if (self.enable_compression and 
                len(self._experiences) >= self.max_experiences):
                self._compress_old_experiences()
            
            # 存储经验
            self._experiences[exp_id] = experience
            
            # 更新索引
            self._scene_index.add_experience(experience)
            self._grade_memory.add_experience(
                exp_id,
                experience.agv_grade,
                experience.scene_type,
                experience.experience_type,
            )
            
            # 更新预测器
            if self.enable_prediction:
                self._predictor.register_experience(experience)
            
            self._stats["total_experiences"] = len(self._experiences)
            self._stats["total_store_operations"] += 1
            
            # 异步持久化
            threading.Thread(target=self._persist_experience, args=(experience,), daemon=True).start()
            
            return exp_id
    
    def _compress_old_experiences(self) -> None:
        """压缩旧经验"""
        if len(self._experiences) < self.max_experiences * 0.8:
            return
        
        # 找出低重要性、低访问的老经验
        now = time.time()
        candidates = []
        for exp in self._experiences.values():
            age_days = (now - exp.last_access_time) / 86400
            if age_days > 7 and exp.access_count < 3:
                candidates.append(exp)
        
        if not candidates:
            # 如果没有低价值经验,删除最老的20%
            sorted_exps = sorted(
                self._experiences.values(), 
                key=lambda e: e.start_timestamp
            )
            candidates = sorted_exps[:int(len(sorted_exps) * 0.2)]
        
        # 压缩
        compressed = self._compressor.compress_experiences(candidates)
        
        # 移除原始经验,保留原型
        for exp in candidates:
            if exp.experience_id not in [e.experience_id for e in compressed]:
                self._experiences.pop(exp.experience_id, None)
        
        self._stats["compression_count"] += 1
    
    def retrieve_experiences(
        self,
        scene_type: Optional[str] = None,
        exp_type: Optional[EmbodiedExperienceType] = None,
        agv_grade: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20,
        only_successful: bool = False,
    ) -> List[EmbodiedExperience]:
        """检索具身经验"""
        with self._lock:
            exp_ids: Set[str] = set()
            
            if scene_type:
                ids = self._scene_index.get_by_scene(
                    scene_type, limit=limit * 2,
                    experience_types=[exp_type] if exp_type else None,
                    tags=tags,
                )
                exp_ids.update(ids)
            
            if agv_grade:
                grade_ids = self._grade_memory.get_for_grade(
                    agv_grade, scene=scene_type, exp_type=exp_type,
                )
                exp_ids.update(grade_ids)
            
            if not exp_ids:
                exp_ids = set(self._experiences.keys())
            
            # 获取经验
            results = []
            for eid in list(exp_ids)[:limit * 2]:
                exp = self._experiences.get(eid)
                if exp is None:
                    continue
                if only_successful and not exp.success:
                    continue
                exp.access_count += 1
                exp.last_access_time = time.time()
                results.append(exp)
            
            # 按重要性排序
            results.sort(key=lambda e: (-e.importance_score, -e.access_count))
            return results[:limit]
    
    def update_skill_memory(
        self,
        skill_id: str,
        skill_name: str,
        scene_type: str,
        agv_grade: str,
        experience: Optional[EmbodiedExperience] = None,
    ) -> SkillMemoryRecord:
        """更新技能记忆"""
        with self._lock:
            key = f"{skill_id}:{scene_type}:{agv_grade}"
            
            if key not in self._skill_memories:
                self._skill_memories[key] = SkillMemoryRecord(
                    skill_id=skill_id,
                    skill_name=skill_name,
                    scene_type=scene_type,
                    agv_grade=agv_grade,
                )
            
            record = self._skill_memories[key]
            
            if experience:
                record.update_from_experience(experience)
            
            return record
    
    def get_skill_memory(
        self,
        skill_id: str,
        scene_type: str,
        agv_grade: str,
    ) -> Optional[SkillMemoryRecord]:
        """获取技能记忆"""
        key = f"{skill_id}:{scene_type}:{agv_grade}"
        return self._skill_memories.get(key)
    
    def predict_task_outcome(
        self,
        scene_type: str,
        exp_type: EmbodiedExperienceType,
        agv_grade: str,
        initial_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """预测任务执行结果"""
        with self._lock:
            self._stats["prediction_requests"] += 1
            
            success_rate, confidence, similar_ids = self._predictor.predict_success(
                scene_type, exp_type, agv_grade, initial_state
            )
            
            failure_reasons = self._predictor.get_failure_reasons(scene_type, exp_type)
            
            # 获取最相关的成功经验作为参考
            reference_experiences = []
            for eid in similar_ids[:3]:
                exp = self._experiences.get(eid)
                if exp:
                    reference_experiences.append({
                        "experience_id": exp.experience_id,
                        "outcome_score": exp.outcome_score,
                        "efficiency_score": exp.efficiency_score,
                        "learned_patterns": exp.learned_patterns,
                    })
            
            return {
                "predicted_success_rate": success_rate,
                "confidence": confidence,
                "similar_experience_count": len(similar_ids),
                "failure_reasons": failure_reasons,
                "reference_experiences": reference_experiences,
                "recommendation": self._get_recommendation(success_rate, confidence),
            }
    
    def _get_recommendation(self, success_rate: float, confidence: float) -> str:
        """根据预测结果给出建议"""
        if confidence < 0.3:
            return "低置信度,建议先进行小规模测试"
        if success_rate >= 0.9:
            return "高成功率预测,可以正常执行"
        elif success_rate >= 0.7:
            return "中等成功率,建议准备备选方案"
        elif success_rate >= 0.5:
            return "不确定结果,建议降低速度/负载进行试探"
        else:
            return "高失败风险,强烈建议重新规划或获取更多经验"
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取记忆系统摘要"""
        with self._lock:
            # 按场景统计
            scene_stats = {}
            for scene_type in ["warehouse", "factory", "hospital", "restaurant", "outdoor", "laboratory", "office", "home"]:
                stats = self._scene_index.get_scene_stats(scene_type)
                if stats["total_experiences"] > 0:
                    scene_stats[scene_type] = stats
            
            # 按经验类型统计
            type_stats: Dict[str, int] = defaultdict(int)
            for exp in self._experiences.values():
                type_stats[exp.experience_type.value] += 1
            
            # 技能记忆统计
            skill_count = len(self._skill_memories)
            mastered_skills = sum(1 for r in self._skill_memories.values() if r.is_mastered)
            
            return {
                "total_experiences": len(self._experiences),
                "scene_stats": scene_stats,
                "experience_type_stats": dict(type_stats),
                "skill_memory_count": skill_count,
                "mastered_skills": mastered_skills,
                "stats": self._stats.copy(),
            }
    
    def _persist_experience(self, experience: EmbodiedExperience) -> None:
        """持久化单个经验到磁盘"""
        try:
            import os
            path = os.path.join(self.storage_path, "experiences")
            os.makedirs(path, exist_ok=True)
            
            filepath = os.path.join(path, f"{experience.experience_id}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(experience.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist experience {experience.experience_id}: {e}")
    
    def _load(self) -> None:
        """从磁盘加载记忆数据"""
        try:
            import os
            path = os.path.join(self.storage_path, "experiences")
            if not os.path.exists(path):
                return
            
            for filename in os.listdir(path):
                if filename.endswith(".json"):
                    filepath = os.path.join(path, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        exp = EmbodiedExperience.from_dict(data)
                        self._experiences[exp.experience_id] = exp
                        self._scene_index.add_experience(exp)
                        self._grade_memory.add_experience(
                            exp.experience_id, exp.agv_grade,
                            exp.scene_type, exp.experience_type,
                        )
                        if self.enable_prediction:
                            self._predictor.register_experience(exp)
                    except Exception:
                        continue
            
            logger.info(f"Loaded {len(self._experiences)} experiences from disk")
        except Exception as e:
            logger.warning(f"Failed to load memory data: {e}")
    
    def export_knowledge(self) -> Dict[str, Any]:
        """导出具身知识(用于联邦学习)"""
        with self._lock:
            knowledge = {
                "export_time": time.time(),
                "experience_count": len(self._experiences),
                "scene_stats": {},
                "skill_stats": {},
                "learned_patterns": [],
                "failure_reasons": [],
            }
            
            # 收集所有习得的模式
            patterns: Set[str] = set()
            failures: Dict[str, int] = defaultdict(int)
            
            for exp in self._experiences.values():
                patterns.update(exp.learned_patterns)
                for f in exp.failure_reasons:
                    failures[f] += 1
            
            knowledge["learned_patterns"] = list(patterns)
            knowledge["failure_reasons"] = sorted(
                failures.items(), key=lambda x: -x[1]
            )[:20]
            
            return knowledge


# 全局实例
_global_instance: Optional[EmbodiedLongTermMemory] = None
_global_lock = threading.Lock()


def create_embodied_long_term_memory(
    storage_path: str = "memory_data/embodied",
    **kwargs,
) -> EmbodiedLongTermMemory:
    """创建具身长期记忆系统全局实例"""
    global _global_instance
    with _global_lock:
        if _global_instance is None:
            _global_instance = EmbodiedLongTermMemory(
                storage_path=storage_path, **kwargs
            )
        return _global_instance
