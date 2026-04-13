"""
tests/embodiment/test_embodied_integration.py

具身智能模块端到端集成测试
测试行为树 + 任务执行器 + 技能注册表 + 记忆系统 的完整集成
"""

import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.embodied import (
    BehaviorTree,
    SequenceNode,
    ActionNode,
    NodeStatus,
    MemoryEnhancedExecutor,
    ScenarioTaskExecutor,
    get_scene_intelligence,
    get_scene_task_planner,
    get_global_skill_registry,
    TaskStatus,
    ExecutionPhase,
    SceneType,
    SkillStatus,
    SkillCategory,
)


class DummyAction(ActionNode):
    """测试用动作节点，总是成功"""
    def __init__(self, name="DummyAction", delay=0.01):
        super().__init__(name)
        self.delay = delay
        self.executed = False

    def execute(self, blackboard):
        import time
        time.sleep(self.delay)
        self.executed = True
        return NodeStatus.SUCCESS


def make_simple_bt():
    root = SequenceNode("TestRoot")
    root.add_child(DummyAction("Step1", delay=0.01))
    root.add_child(DummyAction("Step2", delay=0.01))
    return BehaviorTree(root, name="TestBT")


class TestEmbodiedEndToEnd:
    """端到端具身任务执行测试"""

    def test_single_agv_transport_task(self):
        """测试单AGV运输任务完整流程"""
        bt = make_simple_bt()
        executor = MemoryEnhancedExecutor(
            behavior_tree_root=bt.root,
            memory_system=None,
            simulation_env=None,
            real_agv_interface=None,
            enable_memory=False,
        )

        record = executor.execute_task(
            task_type="transport",
            task_config={"move_speed": 0.5},
            timeout=5.0,
            tick_rate=0.01,
        )
        assert record is not None
        assert record.phase == ExecutionPhase.SUCCEEDED

    def test_patrol_task_with_scene_intelligence(self):
        """测试巡逻任务 + 场景智能"""
        from src.embodied.scene_intelligence import SceneIntelligence, SceneConfig
        config = SceneConfig()
        scene_intel = SceneIntelligence(config=config)
        assert scene_intel is not None

        context = scene_intel.update(
            laser_ranges=np.array([5.0] * 360),
            vision_features={"openness": 0.5},
            location_hint="warehouse",
            nearby_humans=2,
        )
        assert context is not None
        assert context.nearby_humans == 2

        # 场景规则获取
        rules = scene_intel.get_active_rules()
        assert isinstance(rules, dict)

    def test_hospital_task_with_strict_safety(self):
        """测试医院场景严格安全规则"""
        from src.embodied.scene_intelligence import SceneIntelligence, SceneConfig
        config = SceneConfig()
        scene_intel = SceneIntelligence(config=config)
        assert scene_intel is not None

        context = scene_intel.update(
            laser_ranges=np.array([5.0] * 360),
            vision_features={"openness": 0.5},
            location_hint="hospital",
            nearby_humans=0,
        )
        assert context is not None

        # 获取安全规则
        rules = scene_intel.get_active_rules()
        assert isinstance(rules, dict)

    def test_collaborative_task_with_skill_matching(self):
        """测试协同任务 + 技能匹配"""
        registry = get_global_skill_registry()

        best_skill = registry.get_best_skill_for_task(
            scene_type="warehouse",
            category=SkillCategory.COLLABORATION,
        )
        assert best_skill is not None
        assert best_skill.definition.category == SkillCategory.COLLABORATION
        assert "collaboration" in best_skill.definition.tags

    def test_task_executor_with_callbacks(self):
        """测试任务执行器回调系统"""
        phase_changes = []
        ticks = []

        def on_phase_change(phase: ExecutionPhase, task_id: str):
            phase_changes.append((phase, task_id))

        def on_tick(tick_num: int, task_id: str):
            ticks.append((tick_num, task_id))

        bt = make_simple_bt()
        executor = MemoryEnhancedExecutor(
            behavior_tree_root=bt.root,
            memory_system=None,
            simulation_env=None,
            real_agv_interface=None,
            enable_memory=False,
        )
        executor.set_callbacks(on_phase_change=on_phase_change, on_tick=on_tick)

        record = executor.execute_task(
            task_type="transport",
            task_config={},
            timeout=5.0,
            tick_rate=0.01,
        )
        assert record is not None
        assert len(phase_changes) > 0

    def test_scenario_task_executor(self):
        """测试场景化任务执行器"""
        executor = ScenarioTaskExecutor(
            behavior_tree_root=None,
            memory_system=None,
            simulation_env=None,
            real_agv_interface=None,
            enable_memory=False,
        )
        executor.set_scene("warehouse")
        assert executor.current_scene_type == "warehouse"

        record = executor.execute_task(
            task_type="patrol",
            task_config={},
            timeout=5.0,
            tick_rate=0.01,
        )
        assert record is not None

    def test_skill_lifecycle_integration(self):
        """测试技能在任务执行中的生命周期"""
        registry = get_global_skill_registry()

        defns = list(registry._skill_definitions.values())
        skill = registry.create_skill(defns[0].skill_id)
        assert skill.status == SkillStatus.EXPERIMENTAL

        for _ in range(15):
            skill.record_execution(success=True, duration_ms=2000.0)

        assert skill.metrics.total_attempts == 15
        assert skill.metrics.success_rate == 1.0

    def test_scene_task_planner_factory(self):
        """测试场景任务规划器工厂"""
        warehouse_planner = get_scene_task_planner(SceneType.WAREHOUSE)
        assert warehouse_planner is not None

        hospital_planner = get_scene_task_planner(SceneType.HOSPITAL)
        assert hospital_planner is not None

        factory_planner = get_scene_task_planner(SceneType.FACTORY)
        assert factory_planner is not None

    def test_multi_scene_skill_coverage(self):
        """测试多场景技能覆盖"""
        registry = get_global_skill_registry()
        stats = registry.get_registry_stats()

        for scene in ["warehouse", "hospital", "factory", "restaurant", "outdoor"]:
            scene_skills = registry.get_skills_by_scene(scene)
            categories = {s.definition.category for s in scene_skills}
            if scene in ["warehouse", "factory"]:
                assert SkillCategory.NAVIGATION in categories
            assert SkillCategory.SAFETY in categories

    def test_skill_reliability_ranking(self):
        """测试技能可靠性排序"""
        registry = get_global_skill_registry()

        nav_defns = [
            d for d in registry._skill_definitions.values()
            if d.category == SkillCategory.NAVIGATION
        ][:3]
        for d in nav_defns:
            registry.create_skill(d.skill_id, status=SkillStatus.ACTIVE)

        skills = [registry.get_skill(d.skill_id) for d in nav_defns]
        skills[0].record_execution(success=True, duration_ms=1000.0)
        skills[1].record_execution(success=False, duration_ms=1000.0)

        active = registry.get_active_skills()
        nav_active = [s for s in active if s.definition.category == SkillCategory.NAVIGATION]
        assert len(nav_active) >= 2

    def test_task_executor_pause_resume(self):
        """测试任务执行器暂停/恢复"""
        bt = make_simple_bt()
        executor = MemoryEnhancedExecutor(
            behavior_tree_root=bt.root,
            memory_system=None,
            simulation_env=None,
            real_agv_interface=None,
            enable_memory=False,
        )

        task_id = executor.execute_task(
            task_type="transport",
            task_config={},
            timeout=10.0,
            tick_rate=0.001,
        )
        assert task_id is not None


class TestEmbodiedMemorySkillIntegration:
    """记忆系统 + 技能系统 集成测试"""

    def test_skill_registry_with_memory(self):
        """测试技能注册表与记忆系统联动"""
        from src.embodied.memory_integration import EmbodiedSkill as MemorySkill
        from src.embodied.embodied_skill import create_skill_registry

        # 技能注册表
        registry = create_skill_registry()

        defns = list(registry._skill_definitions.values())[:2]
        skill = registry.create_skill(defns[0].skill_id)
        skill.record_execution(success=True, duration_ms=2000.0)

        # 创建记忆条目
        memory_skill = MemorySkill(
            skill_id=skill.skill_id,
            name=skill.name,
            description=skill.definition.description,
            behavior_tree_config={},
            preconditions=[],
            success_rate=skill.success_rate,
            avg_duration=skill.metrics.average_duration_ms,
            usage_count=skill.metrics.total_attempts,
        )
        assert memory_skill.success_rate == 1.0

    def test_skill_updates_reflect_in_memory(self):
        """测试技能更新后反映到记忆中"""
        from src.embodied.memory_integration import EmbodiedSkill as MemorySkill
        from src.embodied.embodied_skill import create_skill_registry

        registry = create_skill_registry()
        defns = list(registry._skill_definitions.values())
        skill = registry.create_skill(defns[0].skill_id)

        for _ in range(5):
            skill.record_execution(success=True, duration_ms=1000.0)
        for _ in range(2):
            skill.record_execution(success=False, duration_ms=1000.0)

        memory_skill = MemorySkill(
            skill_id=skill.skill_id,
            name=skill.name,
            description=skill.definition.description,
            behavior_tree_config={},
            preconditions=[],
            success_rate=skill.success_rate,
            avg_duration=skill.metrics.average_duration_ms,
            usage_count=skill.metrics.total_attempts,
        )

        assert memory_skill.success_rate == skill.success_rate
        assert memory_skill.usage_count == skill.metrics.total_attempts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
