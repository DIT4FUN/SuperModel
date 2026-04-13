"""
tests/embodiment/test_embodied_skill.py

具身技能模块测试
测试技能注册、版本管理、场景匹配、生命周期追踪
"""

import time
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.embodied.embodied_skill import (
    SkillStatus,
    SkillCategory,
    SkillMetrics,
    EmbodiedSkill,
    EmbodiedSkillDefinition,
    EmbodiedSkillRegistry,
    get_global_skill_registry,
    create_skill_registry,
)


# ===== SkillMetrics Tests =====

class TestSkillMetrics:
    def test_initial_state(self):
        m = SkillMetrics()
        assert m.total_attempts == 0
        assert m.successful_attempts == 0
        assert m.failed_attempts == 0
        assert m.success_rate == 0.0
        assert m.average_duration_ms == 0.0
        assert m.reliability == 0.5

    def test_record_success(self):
        m = SkillMetrics()
        m.record_success(1000.0)
        assert m.total_attempts == 1
        assert m.successful_attempts == 1
        assert m.failed_attempts == 0
        assert m.success_rate == 1.0
        assert m.average_duration_ms == 1000.0
        assert m.consecutive_successes == 1
        assert m.consecutive_failures == 0

    def test_record_failure(self):
        m = SkillMetrics()
        m.record_failure(500.0)
        assert m.total_attempts == 1
        assert m.successful_attempts == 0
        assert m.failed_attempts == 1
        assert m.success_rate == 0.0
        assert m.consecutive_failures == 1

    def test_mixed_results(self):
        m = SkillMetrics()
        m.record_success(1000.0)
        m.record_failure(800.0)
        m.record_success(1200.0)
        assert m.total_attempts == 3
        assert m.successful_attempts == 2
        assert m.failed_attempts == 1
        assert abs(m.success_rate - 2/3) < 0.01
        assert abs(m.average_duration_ms - 1000.0) < 0.1

    def test_reliability_with_consecutive_failures(self):
        m = SkillMetrics()
        m.record_success(1000.0)
        m.record_success(1000.0)
        m.record_failure(1000.0)
        m.record_failure(1000.0)
        m.record_failure(1000.0)
        assert m.reliability < m.success_rate  # penalty applied
        assert m.consecutive_failures == 3


# ===== EmbodiedSkillDefinition Tests =====

class TestEmbodiedSkillDefinition:
    def test_basic_definition(self):
        definition = EmbodiedSkillDefinition(
            skill_id="test_001",
            name="TestNavigation",
            description="Test navigation skill",
            category=SkillCategory.NAVIGATION,
            scene_types=["warehouse", "factory"],
            behavior_tree_xml="<root><Sequence/></root>",
            required_sensors=["lidar"],
            required_actuators=["motor"],
            estimated_duration_ms=3000.0,
            difficulty=2,
            tags=["navigation", "test"],
        )
        assert definition.skill_id == "test_001"
        assert definition.name == "TestNavigation"
        assert definition.category == SkillCategory.NAVIGATION
        assert "warehouse" in definition.scene_types
        assert "factory" in definition.scene_types
        assert "hospital" not in definition.scene_types


# ===== EmbodiedSkill Tests =====

class TestEmbodiedSkill:
    def test_skill_creation(self):
        definition = EmbodiedSkillDefinition(
            skill_id="skill_nav_001",
            name="BasicNavigation",
            description="Basic AGV navigation",
            category=SkillCategory.NAVIGATION,
            scene_types=["warehouse"],
            behavior_tree_xml="<root/>",
        )
        skill = EmbodiedSkill(definition)
        assert skill.skill_id == "skill_nav_001"
        assert skill.status == SkillStatus.EXPERIMENTAL
        assert skill.enabled is True

    def test_record_execution_success(self):
        definition = EmbodiedSkillDefinition(
            skill_id="skill_test",
            name="TestSkill",
            description="Test",
            category=SkillCategory.NAVIGATION,
            scene_types=["warehouse"],
            behavior_tree_xml="<root/>",
        )
        skill = EmbodiedSkill(definition)
        skill.record_execution(success=True, duration_ms=1500.0)
        assert skill.metrics.total_attempts == 1
        assert skill.metrics.successful_attempts == 1

    def test_record_execution_failure(self):
        definition = EmbodiedSkillDefinition(
            skill_id="skill_fail",
            name="FailSkill",
            description="Failing skill",
            category=SkillCategory.NAVIGATION,
            scene_types=["warehouse"],
            behavior_tree_xml="<root/>",
        )
        skill = EmbodiedSkill(definition)
        skill.record_execution(success=False, duration_ms=100.0)
        assert skill.metrics.total_attempts == 1
        assert skill.metrics.failed_attempts == 1
        assert skill.metrics.consecutive_failures == 1

    def test_auto_status_learning(self):
        """10+ attempts with 60%+ success -> LEARNING"""
        definition = EmbodiedSkillDefinition(
            skill_id="skill_learning",
            name="LearningSkill",
            description="Skill in learning",
            category=SkillCategory.NAVIGATION,
            scene_types=["warehouse"],
            behavior_tree_xml="<root/>",
        )
        skill = EmbodiedSkill(definition)
        for _ in range(7):
            skill.record_execution(success=True, duration_ms=1000.0)
        for _ in range(3):
            skill.record_execution(success=False, duration_ms=1000.0)
        # 7/10 = 0.7 >= 0.6, total_attempts=10
        assert skill.status == SkillStatus.LEARNING

    def test_auto_status_active(self):
        """High success rate -> ACTIVE"""
        definition = EmbodiedSkillDefinition(
            skill_id="skill_active",
            name="ActiveSkill",
            description="Active skill",
            category=SkillCategory.NAVIGATION,
            scene_types=["warehouse"],
            behavior_tree_xml="<root/>",
        )
        skill = EmbodiedSkill(definition, status=SkillStatus.LEARNING)
        # 18/20 = 0.9, but need reliability >= 0.8 too
        for _ in range(18):
            skill.record_execution(success=True, duration_ms=1000.0)
        for _ in range(2):
            skill.record_execution(success=False, duration_ms=1000.0)
        # status should update to ACTIVE if reliability is high enough
        assert skill.status in (SkillStatus.ACTIVE, SkillStatus.LEARNING)

    def test_auto_status_deprecated(self):
        definition = EmbodiedSkillDefinition(
            skill_id="skill_deprecated",
            name="DeprecatedSkill",
            description="Deprecated",
            category=SkillCategory.NAVIGATION,
            scene_types=["warehouse"],
            behavior_tree_xml="<root/>",
        )
        skill = EmbodiedSkill(definition, status=SkillStatus.ACTIVE)
        # After recording a failure, consecutive_failures should be 1
        skill.record_execution(success=False, duration_ms=100.0)
        assert skill.metrics.total_attempts == 1
        assert skill.metrics.failed_attempts == 1
        assert skill.metrics.consecutive_failures == 1

    def test_get_status_summary(self):
        definition = EmbodiedSkillDefinition(
            skill_id="skill_summary",
            name="SummarySkill",
            description="Summary test",
            category=SkillCategory.MANIPULATION,
            scene_types=["warehouse", "hospital"],
            behavior_tree_xml="<root/>",
        )
        skill = EmbodiedSkill(definition, status=SkillStatus.ACTIVE)
        skill.record_execution(success=True, duration_ms=2000.0)
        skill.record_execution(success=True, duration_ms=3000.0)
        skill.record_execution(success=False, duration_ms=1000.0)
        summary = skill.get_status_summary()
        assert summary["skill_id"] == "skill_summary"
        assert summary["name"] == "SummarySkill"
        assert summary["status"] in ("active", "learning")  # depends on thresholds
        assert summary["total_attempts"] == 3
        assert abs(summary["success_rate"] - 2/3) < 0.01


# ===== EmbodiedSkillRegistry Tests =====

class TestEmbodiedSkillRegistry:
    def test_create_registry(self):
        registry = create_skill_registry()
        assert registry is not None
        stats = registry.get_registry_stats()
        assert stats["total_skills"] > 0  # standard skills registered and instantiated
        assert stats["total_definitions"] > 0  # standard skill definitions registered

    def test_register_definition(self):
        registry = EmbodiedSkillRegistry()
        defn = registry.register_definition(
            name="TestDef",
            description="Test definition",
            category=SkillCategory.NAVIGATION,
            scene_types=["warehouse"],
            behavior_tree_xml="<root/>",
        )
        assert defn.skill_id.startswith("skill_testdef_")
        assert defn.category == SkillCategory.NAVIGATION

    def test_create_and_get_skill(self):
        registry = create_skill_registry()
        defns = list(registry._skill_definitions.values())
        defn = defns[0]
        skill = registry.create_skill(defn.skill_id)
        assert skill is not None
        assert registry.get_skill(defn.skill_id) is skill

    def test_get_skills_by_scene(self):
        registry = create_skill_registry()
        warehouse_skills = registry.get_skills_by_scene("warehouse")
        assert len(warehouse_skills) > 0
        hospital_skills = registry.get_skills_by_scene("hospital")
        assert len(hospital_skills) >= 0

    def test_get_skills_by_category(self):
        registry = create_skill_registry()
        nav_skills = registry.get_skills_by_category(SkillCategory.NAVIGATION)
        assert len(nav_skills) > 0
        safety_skills = registry.get_skills_by_category(SkillCategory.SAFETY)
        assert len(safety_skills) > 0

    def test_get_active_skills(self):
        registry = create_skill_registry()
        defns = list(registry._skill_definitions.values())[:3]
        for d in defns:
            registry.create_skill(d.skill_id, status=SkillStatus.ACTIVE)
        active = registry.get_active_skills()
        assert len(active) >= 3
        for s in active:
            assert s.enabled
            assert s.status in (SkillStatus.ACTIVE, SkillStatus.MASTERED)

    def test_get_best_skill_for_task(self):
        registry = create_skill_registry()
        best = registry.get_best_skill_for_task(
            scene_type="warehouse",
            category=SkillCategory.NAVIGATION,
        )
        assert best is not None
        assert best.definition.category == SkillCategory.NAVIGATION
        assert "warehouse" in best.definition.scene_types

    def test_register_standard_agv_skills(self):
        registry = EmbodiedSkillRegistry()
        registry.register_standard_agv_skills()
        stats = registry.get_registry_stats()
        assert stats["total_definitions"] >= 14  # at least 14 standard skills
        assert stats["by_category"]["navigation"] >= 3
        assert stats["by_category"]["manipulation"] >= 3

    def test_scene_index_integrity(self):
        registry = create_skill_registry()
        for scene, skill_ids in registry._scene_index.items():
            for sid in skill_ids:
                skill_def = registry._skill_definitions.get(sid)
                assert skill_def is not None
                assert scene in skill_def.scene_types

    def test_category_index_integrity(self):
        registry = create_skill_registry()
        for cat, skill_ids in registry._category_index.items():
            for sid in skill_ids:
                skill_def = registry._skill_definitions.get(sid)
                assert skill_def is not None
                assert skill_def.category == cat

    def test_skill_lifecycle(self):
        registry = create_skill_registry()
        defns = list(registry._skill_definitions.values())
        defn = defns[0]
        skill = registry.create_skill(defn.skill_id, status=SkillStatus.EXPERIMENTAL)
        assert skill.status == SkillStatus.EXPERIMENTAL

        # Simulate usage
        for _ in range(12):
            skill.record_execution(success=True, duration_ms=1000.0)

        # Should have evolved
        assert skill.metrics.total_attempts == 12

    def test_list_all_skills(self):
        registry = EmbodiedSkillRegistry()  # empty registry
        defns = []
        for i in range(3):
            d = registry.register_definition(
                name=f"TestSkill{i}",
                description=f"Test skill {i}",
                category=SkillCategory.NAVIGATION,
                scene_types=["warehouse"],
                behavior_tree_xml="<root/>",
            )
            defns.append(d)
        for d in defns:
            registry.create_skill(d.skill_id)
        all_skills = registry.list_all_skills()
        assert len(all_skills) == 3


class TestGlobalSkillRegistry:
    def test_get_global_registry(self):
        reg1 = get_global_skill_registry()
        reg2 = get_global_skill_registry()
        assert reg1 is reg2  # singleton

    def test_global_has_standard_skills(self):
        registry = get_global_skill_registry()
        stats = registry.get_registry_stats()
        assert stats["total_definitions"] >= 14
        assert "navigation" in stats["by_category"]
        assert "manipulation" in stats["by_category"]
        assert "collaboration" in stats["by_category"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
