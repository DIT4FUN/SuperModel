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
cross_scene_transfer.py - 跨场景迁移学习模块
=============================================

功能:
  - SceneKnowledgeGraph: 场景知识图谱, 建模场景之间的技能/知识迁移关系
  - TransferabilityAnalyzer: 迁移性分析器, 评估技能从一个场景迁移到另一个场景的适用度
  - SceneAdapter: 场景适配器, 对迁移的技能进行微调以适应新场景
  - KnowledgeDistillation: 知识蒸馏, 从多场景经验中提取可迁移的通用策略
  - SceneCurriculum: 场景课程学习, 按难度顺序暴露于不同场景
  - CrossSceneSkillLibrary: 跨场景技能库

适用场景:
  - 仓库场景 → 工厂场景 (货架操作技能迁移)
  - 餐厅场景 → 医院场景 (导航避障技能迁移)
  - 户外场景 → 仓库场景 (地形适应技能迁移)

Author: SuperModel Development Team
Version: 3.15.0
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SceneType(Enum):
    """场景类型"""
    WAREHOUSE = "warehouse"
    RESTAURANT = "restaurant"
    HOSPITAL = "hospital"
    INDUSTRIAL = "industrial"
    FACTORY = "factory"
    OUTDOOR = "outdoor"
    OFFICE = "office"
    HOME = "home"
    UNKNOWN = "unknown"


class SkillDomain(Enum):
    """技能领域"""
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    OBJECT_DETECTION = "object_detection"
    PATH_PLANNING = "path_planning"
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"
    GRASPING = "grasping"
    SOCIAL = "social"
    SAFETY = "safety"
    ENERGY_MANAGEMENT = "energy_management"
    COMMUNICATION = "communication"


class TransferMode(Enum):
    """迁移模式"""
    DIRECT = "direct"            # 直接迁移 (零样本)
    FINE_TUNED = "fine_tuned"   # 微调迁移
    FEW_SHOT = "few_shot"       # 少样本迁移
    CURRICULUM = "curriculum"   # 课程学习迁移
    DISTILLATION = "distillation"  # 知识蒸馏迁移


@dataclass
class SkillSpec:
    """技能规格"""
    skill_id: str
    name: str
    domain: SkillDomain
    source_scene: SceneType
    proficiency: float = 0.0    # 0.0-1.0
    sample_count: int = 0
    success_rate: float = 0.0
    avg_completion_time_s: float = 0.0
    energy_cost: float = 0.0
    prerequisites: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    parameters: Dict[str, Any] = field(default_factory=dict)
    performance_history: List[float] = field(default_factory=list)  # 最近成功率历史
    trained_at: float = field(default_factory=time.time)


@dataclass
class SceneProfile:
    """场景画像"""
    scene_type: SceneType
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    # 物理属性
    space_scale: float = 100.0       # 空间规模 (m^2)
    obstacle_density: float = 0.2    # 障碍物密度 0.0-1.0
    terrain_variability: float = 0.0  # 地形变化 0.0-1.0
    dynamic_objects: float = 0.0    # 动态物体比例 0.0-1.0
    human_density: float = 0.0      # 人员密度 0.0-1.0
    # 环境属性
    is_outdoor: bool = False
    is_structured: bool = True      # 结构化环境 vs 非结构化
    lighting_conditions: str = "natural"  # natural/artificial/mixed
    connectivity: float = 1.0       # 空间连通性 0.0-1.0
    # 任务属性
    avg_task_duration_s: float = 300.0
    task_complexity: float = 0.5    # 0.0-1.0
    cooperation_required: float = 0.0  # 协作需求 0.0-1.0

    def distance_to(self, other: SceneProfile) -> float:
        """计算两场景间的特征距离"""
        attrs = [
            ("space_scale", 1000.0),
            ("obstacle_density", 1.0),
            ("terrain_variability", 1.0),
            ("dynamic_objects", 1.0),
            ("human_density", 1.0),
            ("task_complexity", 1.0),
            ("cooperation_required", 1.0),
            ("is_outdoor", 1.0),
            ("is_structured", 1.0),
            ("connectivity", 1.0),
        ]
        dist = 0.0
        for attr, scale in attrs:
            v1 = getattr(self, attr, 0.0) if attr in dir(self) else 0.0
            v2 = getattr(other, attr, 0.0) if attr in dir(other) else 0.0
            if isinstance(v1, bool):
                v1 = 1.0 if v1 else 0.0
                v2 = 1.0 if v2 else 0.0
            dist += ((v1 - v2) / scale) ** 2
        return math.sqrt(dist)


@dataclass
class TransferCandidate:
    """迁移候选"""
    skill: SkillSpec
    target_scene: SceneType
    transferability_score: float     # 0.0-1.0
    mode: TransferMode
    estimated_adaptation_samples: int
    risk_level: float               # 0.0=安全, 1.0=高风险
    adaptation_cost: float           # 预计适配工作量
    confidence: float               # 0.0-1.0
    reasoning: str = ""


@dataclass
class TransferRecord:
    """迁移记录"""
    transfer_id: str
    skill_id: str
    from_scene: SceneType
    to_scene: SceneType
    mode: TransferMode
    initial_performance: float
    final_performance: float
    adaptation_samples: int
    duration_s: float
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# SceneKnowledgeGraph
# ---------------------------------------------------------------------------

class SceneKnowledgeGraph:
    """场景知识图谱 — 管理场景、技能、迁移关系"""

    def __init__(self):
        self.scene_profiles: Dict[SceneType, SceneProfile] = {}
        self.skills: Dict[str, SkillSpec] = {}
        self.scene_skills: Dict[SceneType, Set[str]] = {}  # scene -> skill_ids
        self.skill_scenes: Dict[str, Set[SceneType]] = {}  # skill_id -> scenes
        self.transfer_history: List[TransferRecord] = []
        self._skill_counter = 0
        self._transfer_counter = 0

    def register_scene(self, profile: SceneProfile) -> None:
        """注册场景"""
        self.scene_profiles[profile.scene_type] = profile
        if profile.scene_type not in self.scene_skills:
            self.scene_skills[profile.scene_type] = set()

    def register_skill(self, skill: SkillSpec) -> str:
        """注册技能"""
        skill_id = skill.skill_id or f"SKILL{self._skill_counter:05d}"
        self._skill_counter += 1
        skill.skill_id = skill_id
        self.skills[skill_id] = skill
        self.scene_skills[skill.source_scene].add(skill_id)
        if skill_id not in self.skill_scenes:
            self.skill_scenes[skill_id] = set()
        self.skill_scenes[skill_id].add(skill.source_scene)
        return skill_id

    def get_skills_for_scene(self, scene: SceneType) -> List[SkillSpec]:
        """获取某场景的可用技能"""
        skill_ids = self.scene_skills.get(scene, set())
        return [self.skills[sid] for sid in skill_ids if sid in self.skills]

    def get_transferable_skills(
        self,
        from_scene: SceneType,
        to_scene: SceneType,
        domain_filter: Optional[SkillDomain] = None,
    ) -> List[Tuple[SkillSpec, float]]:
        """获取可从源场景迁移到目标场景的技能"""
        source_skills = self.get_skills_for_scene(from_scene)
        if domain_filter:
            source_skills = [s for s in source_skills if s.domain == domain_filter]

        if to_scene not in self.scene_profiles:
            return [(s, 0.0) for s in source_skills]

        target_profile = self.scene_profiles[to_scene]
        results = []
        for skill in source_skills:
            score = self._compute_transferability(skill, target_profile)
            results.append((skill, score))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def _compute_transferability(self, skill: SkillSpec, target: SceneProfile) -> float:
        """计算技能对目标场景的迁移性"""
        from_profile = self.scene_profiles.get(skill.source_scene)
        if not from_profile:
            return 0.1  # 未知源场景

        scene_distance = from_profile.distance_to(target)
        # 距离归一化 (假设最大距离约10)
        distance_factor = max(0.0, 1.0 - scene_distance / 10.0)

        # 技能成熟度
        maturity = min(skill.proficiency, 1.0)

        # 样本充足度
        sample_factor = min(skill.sample_count / 100, 1.0)

        # 历史成功率
        success_factor = skill.success_rate

        # 综合迁移性
        score = (
            distance_factor * 0.35
            + maturity * 0.25
            + sample_factor * 0.20
            + success_factor * 0.20
        )
        return min(score, 1.0)

    def record_transfer(self, record: TransferRecord) -> None:
        """记录迁移"""
        self._transfer_counter += 1
        record.transfer_id = f"TF{self._transfer_counter:08d}"
        self.transfer_history.append(record)

    def get_scene_graph(self) -> Dict[str, Any]:
        """获取场景图谱可视化数据"""
        nodes = []
        edges = []
        for scene_type, profile in self.scene_profiles.items():
            nodes.append({
                "id": scene_type.value,
                "type": "scene",
                "size": profile.space_scale,
                "complexity": profile.task_complexity,
            })
        # 基于迁移历史构建边
        for record in self.transfer_history:
            edges.append({
                "from": record.from_scene.value,
                "to": record.to_scene.value,
                "skill": record.skill_id,
                "performance_gain": record.final_performance - record.initial_performance,
            })
        return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# TransferabilityAnalyzer
# ---------------------------------------------------------------------------

class TransferabilityAnalyzer:
    """迁移性分析器"""

    def __init__(self, knowledge_graph: SceneKnowledgeGraph):
        self.kg = knowledge_graph

    def analyze(
        self,
        skill_id: str,
        target_scene: SceneType,
    ) -> TransferCandidate:
        """分析技能的迁移可行性"""
        skill = self.kg.skills.get(skill_id)
        if not skill:
            raise ValueError(f"Skill {skill_id} not found")

        target_profile = self.kg.scene_profiles.get(target_scene)
        if not target_profile:
            raise ValueError(f"Scene {target_scene} not found")

        score = self.kg._compute_transferability(skill, target_profile)

        # 选择迁移模式
        if score >= 0.75:
            mode = TransferMode.DIRECT
            samples = 0
        elif score >= 0.5:
            mode = TransferMode.FEW_SHOT
            samples = int((0.75 - score) * 100) + 5
        elif score >= 0.25:
            mode = TransferMode.FINE_TUNED
            samples = int((0.5 - score) * 200) + 20
        else:
            mode = TransferMode.CURRICULUM
            samples = max(50, int((0.25 - score) * 500) + 50)

        # 风险评估
        risk = self._assess_risk(skill, target_profile)
        adaptation_cost = samples * skill.avg_completion_time_s * 0.1

        return TransferCandidate(
            skill=skill,
            target_scene=target_scene,
            transferability_score=score,
            mode=mode,
            estimated_adaptation_samples=samples,
            risk_level=risk,
            adaptation_cost=adaptation_cost,
            confidence=min(score * 1.2, 1.0),
            reasoning=self._generate_reasoning(score, risk, mode),
        )

    def batch_analyze(
        self,
        skills: List[str],
        target_scene: SceneType,
    ) -> List[TransferCandidate]:
        """批量分析多个技能的迁移性"""
        return [self.analyze(sid, target_scene) for sid in skills]

    def find_best_transfer_path(
        self,
        from_scene: SceneType,
        to_scene: SceneType,
    ) -> List[SceneType]:
        """找到最优迁移路径 (通过中间场景)"""
        if from_scene == to_scene:
            return [from_scene]
        if from_scene not in self.kg.scene_profiles or to_scene not in self.kg.scene_profiles:
            return [from_scene, to_scene]

        # 简单BFS找中间节点
        all_scenes = set(self.kg.scene_profiles.keys())
        # 按场景距离排序, 找到中间过渡场景
        from_profile = self.kg.scene_profiles[from_scene]
        to_profile = self.kg.scene_profiles[to_scene]

        intermediates = sorted(
            all_scenes - {from_scene, to_scene},
            key=lambda s: min(
                self.kg.scene_profiles[s].distance_to(from_profile),
                self.kg.scene_profiles[s].distance_to(to_profile),
            )
        )
        if intermediates:
            return [from_scene, intermediates[0], to_scene]
        return [from_scene, to_scene]

    def _assess_risk(self, skill: SkillSpec, target: SceneProfile) -> float:
        """评估迁移风险"""
        from_profile = self.kg.scene_profiles.get(skill.source_scene)
        if not from_profile:
            return 0.8

        # 物理环境差异风险
        env_risk = abs(from_profile.obstacle_density - target.obstacle_density) * 0.3
        env_risk += abs(from_profile.terrain_variability - target.terrain_variability) * 0.3
        env_risk += abs(from_profile.dynamic_objects - target.dynamic_objects) * 0.2

        # 安全关键性 (safety domain 高风险)
        safety_risk = 0.3 if skill.domain == SkillDomain.SAFETY else 0.0

        # 协作需求差异
        coop_risk = abs(from_profile.cooperation_required - target.cooperation_required) * 0.2

        return min(env_risk + safety_risk + coop_risk, 1.0)

    def _generate_reasoning(self, score: float, risk: float, mode: TransferMode) -> str:
        """生成推理说明"""
        if score >= 0.75 and risk < 0.2:
            return f"High transferability ({score:.2f}) with low risk. Direct deployment recommended."
        elif score >= 0.5:
            return f"Moderate transferability ({score:.2f}). {mode.value} adaptation suggested."
        elif score >= 0.25:
            return f"Low transferability ({score:.2f}). Fine-tuning with {mode.value} approach needed."
        else:
            return f"Very low transferability ({score:.2f}). Curriculum learning or full retraining recommended."


# ---------------------------------------------------------------------------
# SceneAdapter
# ---------------------------------------------------------------------------

class SceneAdapter:
    """场景适配器 — 对迁移技能进行微调"""

    def __init__(self):
        self.adaptation_state: Dict[str, Dict] = {}  # skill_id -> adaptation progress
        self._adapter_counter = 0

    def start_adaptation(
        self,
        skill_id: str,
        target_scene: SceneType,
        candidate: TransferCandidate,
    ) -> str:
        """开始技能适配"""
        self._adapter_counter += 1
        adapter_id = f"ADP{self._adapter_counter:06d}"
        self.adaptation_state[skill_id] = {
            "adapter_id": adapter_id,
            "target_scene": target_scene,
            "mode": candidate.mode,
            "samples_collected": 0,
            "target_samples": candidate.estimated_adaptation_samples,
            "performance": candidate.initial_performance if hasattr(candidate, 'initial_performance') else 0.0,
            "start_time": time.time(),
            "losses": [],
            "completed": False,
        }
        return adapter_id

    def record_sample(
        self,
        skill_id: str,
        success: bool,
        completion_time_s: float,
    ) -> Dict[str, Any]:
        """记录一个适配样本"""
        state = self.adaptation_state.get(skill_id)
        if not state:
            return {}
        state["samples_collected"] += 1
        # 更新性能估计
        alpha = 0.1
        perf = 1.0 if success else 0.0
        state["performance"] = (1 - alpha) * state["performance"] + alpha * perf
        state["is_complete"] = state["samples_collected"] >= state["target_samples"]
        return {
            "progress": state["samples_collected"] / max(state["target_samples"], 1),
            "estimated_performance": state["performance"],
            "is_complete": state["is_complete"],
        }

    def get_adaptation_report(self, skill_id: str) -> Dict[str, Any]:
        """获取适配报告"""
        state = self.adaptation_state.get(skill_id)
        if not state:
            return {}
        elapsed = time.time() - state["start_time"]
        return {
            "skill_id": skill_id,
            "adapter_id": state["adapter_id"],
            "target_scene": state["target_scene"].value,
            "progress_pct": state["samples_collected"] / max(state["target_samples"], 1) * 100,
            "samples": f"{state['samples_collected']}/{state['target_samples']}",
            "estimated_performance": state["performance"],
            "elapsed_time_s": elapsed,
            "completed": state["completed"],
        }


# ---------------------------------------------------------------------------
# KnowledgeDistillation
# ---------------------------------------------------------------------------

class KnowledgeDistillation:
    """知识蒸馏 — 从多场景经验中提取通用策略"""

    def __init__(self):
        self.teacher_policies: Dict[str, np.ndarray] = {}  # scene_type -> policy logits
        self.distilled_knowledge: Optional[np.ndarray] = None
        self.temperature = 2.0
        self.alpha = 0.5  # distillation weight

    def register_teacher(
        self,
        scene_type: SceneType,
        policy_logits: np.ndarray,
    ) -> None:
        """注册教师策略"""
        self.teacher_policies[scene_type.value] = np.array(policy_logits)

    def distill(self) -> np.ndarray:
        """执行知识蒸馏"""
        if not self.teacher_policies:
            raise ValueError("No teacher policies registered")

        policies = list(self.teacher_policies.values())
        if len(policies) == 1:
            self.distilled_knowledge = policies[0]
            return self.distilled_knowledge

        # 加权平均, 样本多的场景权重更高
        logits_stack = np.stack(policies, axis=0)
        self.distilled_knowledge = np.mean(logits_stack, axis=0)
        return self.distilled_knowledge

    def get_generalized_policy(self) -> np.ndarray:
        """获取泛化策略"""
        if self.distilled_knowledge is None:
            self.distill()
        return self.distilled_knowledge

    def adapt_to_scene(
        self,
        target_scene: SceneType,
        teacher_logits: np.ndarray,
    ) -> np.ndarray:
        """使用蒸馏知识适配特定场景"""
        if self.distilled_knowledge is None:
            self.distill()
        # 插值: 蒸馏知识 + 场景特定知识
        return self.alpha * self.distilled_knowledge + (1 - self.alpha) * teacher_logits


# ---------------------------------------------------------------------------
# SceneCurriculum
# ---------------------------------------------------------------------------

class SceneCurriculum:
    """场景课程学习 — 按难度顺序训练"""

    def __init__(self, knowledge_graph: SceneKnowledgeGraph):
        self.kg = knowledge_graph
        self.curriculum_stages: List[Dict] = []
        self.current_stage = 0
        self.graduation_threshold = 0.75

    def build_curriculum(
        self,
        target_scene: SceneType,
        max_stages: int = 5,
    ) -> List[Dict]:
        """为目标场景构建课程"""
        if target_scene not in self.kg.scene_profiles:
            return []

        target = self.kg.scene_profiles[target_scene]
        all_scenes = list(self.kg.scene_profiles.keys())

        # 按与目标场景的距离排序
        sorted_scenes = sorted(
            all_scenes,
            key=lambda s: target.distance_to(self.kg.scene_profiles[s])
        )

        self.curriculum_stages = []
        # 每N个场景一个阶段
        chunk_size = max(1, len(sorted_scenes) // max_stages)
        for i in range(0, len(sorted_scenes), chunk_size):
            chunk = sorted_scenes[i:i+chunk_size]
            if not chunk:
                continue
            stage = {
                "stage_id": len(self.curriculum_stages),
                "scenes": [s.value for s in chunk],
                "target_scene": target_scene.value,
                "skills": [],
            }
            # 收集该阶段可用的技能
            for scene in chunk:
                skills = self.kg.get_skills_for_scene(scene)
                stage["skills"].extend([
                    {
                        "skill_id": s.skill_id,
                        "domain": s.domain.value,
                        "proficiency": s.proficiency,
                    }
                    for s in skills if s.proficiency >= 0.5
                ])
            self.curriculum_stages.append(stage)

        return self.curriculum_stages

    def get_current_stage(self) -> Optional[Dict]:
        """获取当前课程阶段"""
        if 0 <= self.current_stage < len(self.curriculum_stages):
            return self.curriculum_stages[self.current_stage]
        return None

    def advance_stage(self) -> bool:
        """进入下一阶段"""
        if self.current_stage < len(self.curriculum_stages) - 1:
            self.current_stage += 1
            return True
        return False

    def check_graduation(
        self,
        skill_performances: Dict[str, float],
    ) -> bool:
        """检查是否可以毕业"""
        current = self.get_current_stage()
        if not current:
            return True
        stage_skills = {s["skill_id"] for s in current["skills"]}
        if not stage_skills:
            return True
        scores = [skill_performances.get(sid, 0.0) for sid in stage_skills]
        avg = sum(scores) / len(scores) if scores else 0.0
        return avg >= self.graduation_threshold


# ---------------------------------------------------------------------------
# CrossSceneSkillLibrary
# ---------------------------------------------------------------------------

class CrossSceneSkillLibrary:
    """跨场景技能库 — 统一管理所有场景的技能迁移"""

    def __init__(self):
        self.kg = SceneKnowledgeGraph()
        self.analyzer = TransferabilityAnalyzer(self.kg)
        self.adapter = SceneAdapter()
        self.distiller = KnowledgeDistillation()
        self.curriculum: Optional[SceneCurriculum] = None
        self._initialize_default_scenes()

    def _initialize_default_scenes(self) -> None:
        """初始化默认场景画像"""
        defaults = [
            SceneProfile(SceneType.WAREHOUSE, "Warehouse",
                space_scale=5000.0, obstacle_density=0.4,
                terrain_variability=0.1, dynamic_objects=0.2,
                human_density=0.15, is_outdoor=False,
                is_structured=True, task_complexity=0.6,
                cooperation_required=0.4),
            SceneProfile(SceneType.RESTAURANT, "Restaurant",
                space_scale=500.0, obstacle_density=0.5,
                terrain_variability=0.0, dynamic_objects=0.5,
                human_density=0.8, is_outdoor=False,
                is_structured=True, task_complexity=0.5,
                cooperation_required=0.3),
            SceneProfile(SceneType.HOSPITAL, "Hospital",
                space_scale=2000.0, obstacle_density=0.3,
                terrain_variability=0.0, dynamic_objects=0.3,
                human_density=0.6, is_outdoor=False,
                is_structured=True, task_complexity=0.7,
                cooperation_required=0.6),
            SceneProfile(SceneType.INDUSTRIAL, "Industrial",
                space_scale=10000.0, obstacle_density=0.6,
                terrain_variability=0.3, dynamic_objects=0.4,
                human_density=0.2, is_outdoor=True,
                is_structured=False, task_complexity=0.8,
                cooperation_required=0.5),
            SceneProfile(SceneType.OUTDOOR, "Outdoor",
                space_scale=50000.0, obstacle_density=0.2,
                terrain_variability=0.8, dynamic_objects=0.3,
                human_density=0.05, is_outdoor=True,
                is_structured=False, task_complexity=0.6,
                cooperation_required=0.1),
        ]
        for profile in defaults:
            self.kg.register_scene(profile)

    def register_skill(
        self,
        name: str,
        domain: SkillDomain,
        source_scene: SceneType,
        **kwargs,
    ) -> str:
        """注册技能"""
        skill = SkillSpec(
            skill_id="",
            name=name,
            domain=domain,
            source_scene=source_scene,
            **kwargs,
        )
        return self.kg.register_skill(skill)

    def query_transfer(
        self,
        from_scene: SceneType,
        to_scene: SceneType,
        domain: Optional[SkillDomain] = None,
    ) -> List[TransferCandidate]:
        """查询迁移候选"""
        candidates = self.kg.get_transferable_skills(from_scene, to_scene, domain)
        results = []
        for skill, score in candidates:
            if score < 0.1:
                continue
            target_profile = self.kg.scene_profiles.get(to_scene)
            if not target_profile:
                continue
            # 重新分析获取详细信息
            candidate = self.analyzer.analyze(skill.skill_id, to_scene)
            results.append(candidate)
        return sorted(results, key=lambda c: c.transferability_score, reverse=True)

    def execute_transfer(
        self,
        skill_id: str,
        to_scene: SceneType,
    ) -> Dict[str, Any]:
        """执行技能迁移"""
        candidate = self.analyzer.analyze(skill_id, to_scene)
        if candidate.transferability_score < 0.1:
            return {"status": "rejected", "reason": "Too low transferability"}

        adapter_id = self.adapter.start_adaptation(skill_id, to_scene, candidate)

        # 记录迁移
        record = TransferRecord(
            transfer_id="",
            skill_id=skill_id,
            from_scene=self.kg.skills[skill_id].source_scene,
            to_scene=to_scene,
            mode=candidate.mode,
            initial_performance=candidate.transferability_score * 0.5,
            final_performance=0.0,
            adaptation_samples=0,
            duration_s=0.0,
        )
        self.kg.record_transfer(record)

        return {
            "status": "started",
            "adapter_id": adapter_id,
            "candidate": {
                "score": candidate.transferability_score,
                "mode": candidate.mode.value,
                "samples_needed": candidate.estimated_adaptation_samples,
                "risk": candidate.risk_level,
            },
        }

    def distill_knowledge(self) -> np.ndarray:
        """蒸馏所有场景知识"""
        return self.distiller.distill()

    def build_curriculum(self, target_scene: SceneType) -> List[Dict]:
        """为目标场景构建课程"""
        self.curriculum = SceneCurriculum(self.kg)
        return self.curriculum.build_curriculum(target_scene)
