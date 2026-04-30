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
embodied_skill - 具身技能生命周期管理
======================================

具身技能注册、版本管理、场景匹配、生命周期追踪
支持5种AGV场景（仓库/医院/工厂/餐厅/户外）的技能库管理
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class SkillStatus(Enum):
    """技能状态"""
    EXPERIMENTAL = "experimental"   # 实验性，待验证
    ACTIVE = "active"               # 活跃，可使用
    LEARNING = "learning"           # 学习中，成功率待提升
    DEPRECATED = "deprecated"       # 已废弃，不推荐使用
    MASTERED = "mastered"          # 已掌握，高可靠性


class SkillCategory(Enum):
    """技能类别"""
    NAVIGATION = "navigation"       # 导航移动
    MANIPULATION = "manipulation"   # 物品操作
    COLLABORATION = "collaboration" # 协同作业
    SAFETY = "safety"               # 安全监控
    MAINTENANCE = "maintenance"    # 维护检修
    PERCEPTION = "perception"       # 感知识别
    PLANNING = "planning"          # 任务规划


@dataclass
class SkillMetrics:
    """技能执行指标"""
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_duration_ms: float = 0.0
    last_used_timestamp: float = 0.0
    last_success_timestamp: float = 0.0
    last_failure_timestamp: float = 0.0
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    max_consecutive_successes: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.successful_attempts / self.total_attempts

    @property
    def average_duration_ms(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.total_duration_ms / self.total_attempts

    @property
    def reliability(self) -> float:
        """可靠性评分（考虑连续失败）"""
        if self.total_attempts == 0:
            return 0.5
        base = self.success_rate
        penalty = min(0.3, self.consecutive_failures * 0.1)
        return max(0.0, base - penalty)

    def record_success(self, duration_ms: float) -> None:
        self.successful_attempts += 1
        self.total_attempts += 1
        self.total_duration_ms += duration_ms
        self.last_used_timestamp = time.time()
        self.last_success_timestamp = time.time()
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.max_consecutive_successes = max(
            self.max_consecutive_successes, self.consecutive_successes
        )

    def record_failure(self, duration_ms: float) -> None:
        self.failed_attempts += 1
        self.total_attempts += 1
        self.total_duration_ms += duration_ms
        self.last_used_timestamp = time.time()
        self.last_failure_timestamp = time.time()
        self.consecutive_failures += 1
        self.consecutive_successes = 0


@dataclass
class SkillVersion:
    """技能版本"""
    version: str
    created_at: float
    changelog: str = ""
    metrics: SkillMetrics = field(default_factory=SkillMetrics)


@dataclass
class EmbodiedSkillDefinition:
    """具身技能定义"""
    skill_id: str
    name: str
    description: str
    category: SkillCategory
    scene_types: list[str]  # ["warehouse", "hospital", "factory", "restaurant", "outdoor"]
    behavior_tree_xml: str
    required_sensors: list[str] = field(default_factory=list)
    required_actuators: list[str] = field(default_factory=list)
    estimated_duration_ms: float = 5000.0
    difficulty: int = 1  # 1-5
    tags: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)  # skill_ids
    metadata: dict = field(default_factory=dict)


class EmbodiedSkill:
    """具身技能（实例化后的技能对象，含运行时状态）"""

    def __init__(
        self,
        definition: EmbodiedSkillDefinition,
        status: SkillStatus = SkillStatus.EXPERIMENTAL,
        version: str = "1.0.0",
    ):
        self.skill_id = definition.skill_id
        self.definition = definition
        self.status = status
        self.current_version = version
        self.versions: dict[str, SkillVersion] = {
            version: SkillVersion(version=version, created_at=time.time())
        }
        self.metrics: SkillMetrics = SkillMetrics()
        self.enabled: bool = True
        self.min_success_rate: float = 0.7  # 最低成功率阈值

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def success_rate(self) -> float:
        return self.metrics.success_rate

    @property
    def reliability(self) -> float:
        return self.metrics.reliability

    def record_execution(self, success: bool, duration_ms: float) -> None:
        """记录技能执行结果"""
        if success:
            self.metrics.record_success(duration_ms)
        else:
            self.metrics.record_failure(duration_ms)

        # 自动状态转换
        self._update_status()

    def _update_status(self) -> None:
        """基于指标自动更新技能状态"""
        sr = self.success_rate
        rl = self.reliability
        att = self.metrics.total_attempts

        if self.status == SkillStatus.EXPERIMENTAL:
            if att >= 10 and sr >= 0.6:
                self.status = SkillStatus.LEARNING
        elif self.status == SkillStatus.LEARNING:
            if sr >= 0.85 and rl >= 0.8:
                self.status = SkillStatus.ACTIVE
            elif self.metrics.consecutive_failures >= 3:
                self.status = SkillStatus.EXPERIMENTAL
        elif self.status == SkillStatus.ACTIVE:
            if sr >= 0.95 and self.metrics.max_consecutive_successes >= 20:
                self.status = SkillStatus.MASTERED
            elif sr < self.min_success_rate:
                self.status = SkillStatus.LEARNING
        elif self.status == SkillStatus.MASTERED:
            if sr < 0.8:
                self.status = SkillStatus.ACTIVE

    def get_status_summary(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "status": self.status.value,
            "version": self.current_version,
            "success_rate": round(self.success_rate, 3),
            "reliability": round(self.reliability, 3),
            "total_attempts": self.metrics.total_attempts,
            "avg_duration_ms": round(self.metrics.average_duration_ms, 1),
            "consecutive_successes": self.metrics.consecutive_successes,
            "consecutive_failures": self.metrics.consecutive_failures,
        }


class EmbodiedSkillRegistry:
    """具身技能注册表"""

    def __init__(self):
        self._skills: dict[str, EmbodiedSkill] = {}
        self._skill_definitions: dict[str, EmbodiedSkillDefinition] = {}
        self._scene_index: dict[str, set[str]] = {
            "warehouse": set(),
            "hospital": set(),
            "factory": set(),
            "restaurant": set(),
            "outdoor": set(),
        }
        self._category_index: dict[SkillCategory, set[str]] = {
            cat: set() for cat in SkillCategory
        }

    def register_definition(
        self,
        name: str,
        description: str,
        category: SkillCategory,
        scene_types: list[str],
        behavior_tree_xml: str,
        required_sensors: Optional[list[str]] = None,
        required_actuators: Optional[list[str]] = None,
        estimated_duration_ms: float = 5000.0,
        difficulty: int = 1,
        tags: Optional[list[str]] = None,
        prerequisites: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> EmbodiedSkillDefinition:
        """注册新技能定义"""
        skill_id = f"skill_{name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
        definition = EmbodiedSkillDefinition(
            skill_id=skill_id,
            name=name,
            description=description,
            category=category,
            scene_types=scene_types,
            behavior_tree_xml=behavior_tree_xml,
            required_sensors=required_sensors or [],
            required_actuators=required_actuators or [],
            estimated_duration_ms=estimated_duration_ms,
            difficulty=difficulty,
            tags=tags or [],
            prerequisites=prerequisites or [],
            metadata=metadata or {},
        )
        self._skill_definitions[skill_id] = definition

        # 索引
        for scene in scene_types:
            if scene in self._scene_index:
                self._scene_index[scene].add(skill_id)
        self._category_index[category].add(skill_id)

        return definition

    def create_skill(
        self,
        skill_id: str,
        status: SkillStatus = SkillStatus.EXPERIMENTAL,
        version: str = "1.0.0",
    ) -> Optional[EmbodiedSkill]:
        """从定义创建技能实例"""
        if skill_id not in self._skill_definitions:
            return None
        skill = EmbodiedSkill(
            definition=self._skill_definitions[skill_id],
            status=status,
            version=version,
        )
        self._skills[skill_id] = skill
        return skill

    def get_skill(self, skill_id: str) -> Optional[EmbodiedSkill]:
        return self._skills.get(skill_id)

    def get_skills_by_scene(self, scene_type: str) -> list[EmbodiedSkill]:
        """获取指定场景的所有技能"""
        skill_ids = self._scene_index.get(scene_type, set())
        return [self._skills[sid] for sid in skill_ids if sid in self._skills]

    def get_skills_by_category(self, category: SkillCategory) -> list[EmbodiedSkill]:
        """获取指定类别的所有技能"""
        skill_ids = self._category_index.get(category, set())
        return [self._skills[sid] for sid in skill_ids if sid in self._skills]

    def get_active_skills(self) -> list[EmbodiedSkill]:
        """获取所有活跃技能（ACTIVE + MASTERED）"""
        return [
            s for s in self._skills.values()
            if s.enabled and s.status in (SkillStatus.ACTIVE, SkillStatus.MASTERED)
        ]

    def get_best_skill_for_task(
        self,
        scene_type: str,
        category: SkillCategory,
        required_sensors: Optional[list[str]] = None,
    ) -> Optional[EmbodiedSkill]:
        """为任务找到最佳匹配技能"""
        candidates = [
            s for s in self.get_skills_by_scene(scene_type)
            if s.definition.category == category
            and s.enabled
            and s.status not in (SkillStatus.DEPRECATED,)
        ]
        if not candidates:
            return None

        # 按可靠性排序
        candidates.sort(key=lambda s: s.reliability, reverse=True)
        return candidates[0]

    def register_standard_agv_skills(self) -> None:
        """注册标准AGV技能库（仓库/医院/工厂/餐厅/户外）"""
        skills_defs = [
            # === 导航类 ===
            (
                "WarehouseNavigation",
                "仓库导航移动",
                SkillCategory.NAVIGATION,
                ["warehouse", "factory"],
                "<root><Sequence><AGVCheckSafeCondition/><AGVMoveToAction/></Sequence></root>",
                ["lidar", "encoder"],
                ["motor"],
                3000.0, 1, ["navigation", "warehouse", "basic"],
            ),
            (
                "HospitalNavigation",
                "医院环境导航",
                SkillCategory.NAVIGATION,
                ["hospital"],
                "<root><Sequence><AGVCheckSafeCondition/><AGVMoveToAction/></Sequence></root>",
                ["lidar", "imu", "camera"],
                ["motor"],
                4000.0, 2, ["navigation", "hospital", "precision"],
                [],
                {"max_speed_mps": 0.8, "quiet_mode": True},
            ),
            (
                "OutdoorNavigation",
                "户外环境导航",
                SkillCategory.NAVIGATION,
                ["outdoor"],
                "<root><Sequence><AGVCheckSafeCondition/><AGVMoveToAction/></Sequence></root>",
                ["lidar", "gps", "imu"],
                ["motor"],
                5000.0, 3, ["navigation", "outdoor", "gps"],
            ),
            # === 物品操作类 ===
            (
                "StandardGrasp",
                "标准抓取",
                SkillCategory.MANIPULATION,
                ["warehouse", "factory", "restaurant"],
                "<root><Sequence><AGVCheckSafeCondition/><AGVGraspAction/></Sequence></root>",
                ["tactile", "force"],
                ["gripper"],
                2000.0, 2, ["manipulation", "grasp", "basic"],
            ),
            (
                "HospitalGrasp",
                "医院无菌抓取",
                SkillCategory.MANIPULATION,
                ["hospital"],
                "<root><Sequence><AGVCheckSafeCondition/><AGVGraspAction/></Sequence></root>",
                ["tactile", "force", "camera"],
                ["gripper"],
                3000.0, 3, ["manipulation", "hospital", "sterile"],
                [],
                {"sterile_check": True, "force_limit_N": 5.0},
            ),
            (
                "StandardRelease",
                "标准释放",
                SkillCategory.MANIPULATION,
                ["warehouse", "factory", "hospital", "restaurant"],
                "<root><Sequence><AGVCheckSafeCondition/><AGVReleaseAction/></Sequence></root>",
                ["tactile", "force"],
                ["gripper"],
                1500.0, 1, ["manipulation", "release", "basic"],
            ),
            # === 协同作业类 ===
            (
                "FormationMove",
                "编队移动",
                SkillCategory.COLLABORATION,
                ["warehouse", "factory", "outdoor"],
                "<root><Sequence><AGVCheckFormationReachedCondition/><AGVCoordinatedMoveToAction/></Sequence></root>",
                ["lidar", "encoder"],
                ["motor"],
                4000.0, 3, ["collaboration", "formation", "swarm"],
            ),
            (
                "CollaborativeLift",
                "协同举升",
                SkillCategory.COLLABORATION,
                ["warehouse", "factory"],
                "<root><Parallel success_threshold='1'><AGVParallelGraspAction/><AGVNegotiateRoleAction/></Parallel></root>",
                ["force", "tactile"],
                ["gripper"],
                6000.0, 4, ["collaboration", "heavy_load", "swarm"],
            ),
            (
                "CollaborativeTransport",
                "协同运输",
                SkillCategory.COLLABORATION,
                ["warehouse", "factory"],
                "<root><Sequence><AGVNegotiateRoleAction/><AGVParallelGraspAction/><AGVCoordinatedMoveToAction/><AGVParallelReleaseAction/></Sequence></root>",
                ["lidar", "force", "tactile"],
                ["motor", "gripper"],
                15000.0, 4, ["collaboration", "transport", "swarm"],
            ),
            # === 安全监控类 ===
            (
                "CollisionAvoidance",
                "碰撞规避",
                SkillCategory.SAFETY,
                ["warehouse", "hospital", "factory", "restaurant", "outdoor"],
                "<root><Sequence><AGVCheckSafeCondition/><Fallback><AGVMoveToAction/><Inverter><AGVCheckSafeCondition/></Inverter></Fallback></Sequence></root>",
                ["lidar", "camera"],
                ["motor"],
                500.0, 2, ["safety", "collision", "emergency"],
            ),
            (
                "BatteryCheck",
                "电池状态检查",
                SkillCategory.SAFETY,
                ["warehouse", "hospital", "factory", "restaurant", "outdoor"],
                "<root><Sequence><AGVCheckBatteryCondition/></Sequence></root>",
                ["battery_sensor"],
                [],
                200.0, 1, ["safety", "battery", "monitoring"],
            ),
            # === 维护检修类 ===
            (
                "SelfDiagnostic",
                "自诊断检查",
                SkillCategory.MAINTENANCE,
                ["warehouse", "hospital", "factory", "restaurant", "outdoor"],
                "<root><Sequence><BatteryCheck/><SafetyCheck/><SensorCheck/></Sequence></root>",
                ["battery_sensor", "current_sensor"],
                [],
                3000.0, 1, ["maintenance", "diagnostic", "health"],
            ),
            # === 感知识别类 ===
            (
                "ObjectRecognition",
                "目标物体识别",
                SkillCategory.PERCEPTION,
                ["warehouse", "factory", "restaurant"],
                "<root><Sequence><CameraCapture/><ObjectDetect/></Sequence></root>",
                ["camera"],
                [],
                2000.0, 2, ["perception", "vision", "detection"],
            ),
            (
                "QrCodeLocalization",
                "二维码定位",
                SkillCategory.PERCEPTION,
                ["warehouse", "hospital", "factory"],
                "<root><Sequence><CameraCapture/><QrDetect/></Sequence></root>",
                ["camera"],
                [],
                1000.0, 1, ["perception", "localization", "qr"],
            ),
        ]

        for entry in skills_defs:
            if len(entry) == 10:
                (name, desc, cat, scenes, bt_xml, sensors, actuators, dur, diff, tags) = entry
                prereq, meta = [], {}
            elif len(entry) == 12:
                (name, desc, cat, scenes, bt_xml, sensors, actuators, dur, diff, tags, prereq, meta) = entry
            else:
                continue
            definition = self.register_definition(
                name=name,
                description=desc,
                category=cat,
                scene_types=scenes,
                behavior_tree_xml=bt_xml,
                required_sensors=sensors,
                required_actuators=actuators,
                estimated_duration_ms=dur,
                difficulty=diff,
                tags=tags,
                prerequisites=prereq or [],
                metadata=meta or {},
            )
            # Also create the skill instance so it can be used immediately
            self.create_skill(definition.skill_id, status=SkillStatus.ACTIVE)

    def list_all_skills(self) -> list[dict]:
        """列出所有已注册技能（带指标）"""
        return [s.get_status_summary() for s in self._skills.values()]

    def get_registry_stats(self) -> dict:
        """获取注册表统计"""
        by_status: dict[str, int] = {}
        by_category: dict[str, int] = {}
        for s in self._skills.values():
            by_status[s.status.value] = by_status.get(s.status.value, 0) + 1
            by_category[s.definition.category.value] = (
                by_category.get(s.definition.category.value, 0) + 1
            )
        return {
            "total_skills": len(self._skills),
            "total_definitions": len(self._skill_definitions),
            "by_status": by_status,
            "by_category": by_category,
            "scene_types_covered": list(self._scene_index.keys()),
        }


# === 全局注册表实例 ===
_global_registry: Optional[EmbodiedSkillRegistry] = None


def get_global_skill_registry() -> EmbodiedSkillRegistry:
    """获取全局技能注册表（单例）"""
    global _global_registry
    if _global_registry is None:
        _global_registry = EmbodiedSkillRegistry()
        _global_registry.register_standard_agv_skills()
    return _global_registry


def create_skill_registry() -> EmbodiedSkillRegistry:
    """创建新的技能注册表（隔离）"""
    registry = EmbodiedSkillRegistry()
    registry.register_standard_agv_skills()
    return registry
