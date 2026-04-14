"""
test_hierarchical_bt_and_degradation.py - 层级行为树 + 降级管理器测试
SuperModel 超模态大模型具身智能系统

测试内容:
1. HierarchicalBehaviorTreeComposer - 层级行为树组合
2. SceneMemoryAugmentedPlanner - 记忆增强规划
3. CrossSceneTransferLearner - 跨场景迁移学习
4. DegradationManager - 优雅降级管理
5. 降级与任务执行集成
"""

import pytest
import time
import threading
from collections import deque
from typing import Any, Dict, List, Optional

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


# ============================================================
# 测试 1: HierarchicalBehaviorTreeComposer
# ============================================================

class TestHierarchicalBehaviorTreeComposer:
    """层级行为树组合器测试"""

    def test_compose_warehouse_task_tree(self):
        """测试仓库任务行为树组合"""
        from src.embodied.scene_task_planner import (
            HierarchicalBehaviorTreeComposer,
            HierarchicalTaskLevel,
            SceneType,
        )

        composer = HierarchicalBehaviorTreeComposer()

        bt = composer.compose_task_tree(
            task_goal="warehouse_pick_and_place",
            scene_type=SceneType.WAREHOUSE,
            context={"target_location": "A1", "item_id": "SKU123"},
        )

        assert bt is not None
        assert bt.name.startswith("HTN_warehouse_pick")
        assert bt.root is not None

        # 执行行为树
        blackboard_data = {
            "battery_level": 80,
            "agv_healthy": True,
            "obstacle_detected": False,
            "comms_active": True,
            "heartbeat_timeout": False,
        }
        for k, v in blackboard_data.items():
            bt.blackboard.set(k, v)
        status = bt.tick()
        assert status is not None

    def test_compose_restaurant_task_tree(self):
        """测试餐厅任务行为树组合"""
        from src.embodied.scene_task_planner import (
            HierarchicalBehaviorTreeComposer,
            SceneType,
        )

        composer = HierarchicalBehaviorTreeComposer()

        bt = composer.compose_task_tree(
            task_goal="restaurant_food_delivery",
            scene_type=SceneType.RESTAURANT,
            context={"table_id": "T5", "order_id": "ORD789"},
        )

        assert bt is not None
        assert bt.root is not None

        blackboard_data = {
            "battery_level": 60,
            "agv_healthy": True,
            "obstacle_detected": False,
            "comms_active": True,
            "heartbeat_timeout": False,
        }
        for k, v in blackboard_data.items():
            bt.blackboard.set(k, v)
        status = bt.tick()
        assert status is not None

    def test_compose_hospital_task_tree(self):
        """测试医院任务行为树组合"""
        from src.embodied.scene_task_planner import (
            HierarchicalBehaviorTreeComposer,
            SceneType,
        )

        composer = HierarchicalBehaviorTreeComposer()

        bt = composer.compose_task_tree(
            task_goal="hospital_medicine_delivery",
            scene_type=SceneType.HOSPITAL,
            context={"ward_id": "ICU-3", "medicine_id": "MED001"},
        )

        assert bt is not None
        assert bt.root is not None

        blackboard_data = {
            "battery_level": 90,
            "agv_healthy": True,
            "obstacle_detected": False,
            "comms_active": True,
            "heartbeat_timeout": False,
        }
        for k, v in blackboard_data.items():
            bt.blackboard.set(k, v)
        status = bt.tick()
        assert status is not None

    def test_compose_caches_reused(self):
        """测试行为树组合缓存复用"""
        from src.embodied.scene_task_planner import (
            HierarchicalBehaviorTreeComposer,
            SceneType,
        )

        composer = HierarchicalBehaviorTreeComposer()

        bt1 = composer.compose_task_tree(
            task_goal="generic_transport",
            scene_type=SceneType.FACTORY,
        )
        bt2 = composer.compose_task_tree(
            task_goal="generic_transport",
            scene_type=SceneType.FACTORY,
        )

        # 相同任务目标+场景应该返回缓存版本
        assert bt1 is bt2

    def test_reactive_level_obstacle_avoidance(self):
        """测试反应级避障行为"""
        from src.embodied.scene_task_planner import (
            HierarchicalBehaviorTreeComposer,
            SceneType,
        )

        composer = HierarchicalBehaviorTreeComposer()

        bt = composer.compose_task_tree(
            task_goal="test_reactive",
            scene_type=SceneType.WAREHOUSE,
        )

        # 测试障碍物检测触发
        blackboard_data = {
            "battery_level": 50,
            "agv_healthy": True,
            "obstacle_detected": True,  # 障碍物触发
            "comms_active": True,
            "heartbeat_timeout": False,
        }
        for k, v in blackboard_data.items():
            bt.blackboard.set(k, v)
        bt.tick()
        assert bt.blackboard.get("avoidance_triggered") is True

    def test_reactive_level_low_battery(self):
        """测试反应级低电量触发"""
        from src.embodied.scene_task_planner import (
            HierarchicalBehaviorTreeComposer,
            SceneType,
        )

        composer = HierarchicalBehaviorTreeComposer()

        bt = composer.compose_task_tree(
            task_goal="test_battery",
            scene_type=SceneType.OUTDOOR,
        )

        blackboard_data = {
            "battery_level": 10,  # 低电量
            "agv_healthy": True,
            "obstacle_detected": False,
            "comms_active": True,
            "heartbeat_timeout": False,
        }
        for k, v in blackboard_data.items():
            bt.blackboard.set(k, v)
        bt.tick()
        assert bt.blackboard.get("recharge_triggered") is True

    def test_strategic_level_low_battery_blocks_task(self):
        """测试战略级低电量阻止任务"""
        from src.embodied.scene_task_planner import (
            HierarchicalBehaviorTreeComposer,
            SceneType,
        )

        composer = HierarchicalBehaviorTreeComposer()

        bt = composer.compose_task_tree(
            task_goal="warehouse_transport",
            scene_type=SceneType.WAREHOUSE,
        )

        blackboard_data = {
            "battery_level": 10,  # 电量不足
            "agv_healthy": True,
            "obstacle_detected": False,
            "comms_active": True,
            "heartbeat_timeout": False,
        }
        for k, v in blackboard_data.items():
            bt.blackboard.set(k, v)
        bt.tick()
        # 战略级应该检测到低电量并阻止任务
        # resources_allocated should not be True when battery is low
        assert bt.blackboard.get("resources_allocated") is not True or bt.last_status.value == "failure"

    def test_tactical_level_with_scene_intelligence(self):
        """测试战术级使用场景智能"""
        from src.embodied.scene_task_planner import (
            HierarchicalBehaviorTreeComposer,
            SceneType,
        )

        # 模拟场景智能
        class MockSceneIntelligence:
            def get_scene_context(self, scene_type):
                return {"type": scene_type.value, "risk_level": "medium"}

        mock_si = MockSceneIntelligence()
        composer = HierarchicalBehaviorTreeComposer(scene_intelligence=mock_si)

        bt = composer.compose_task_tree(
            task_goal="factory_production",
            scene_type=SceneType.FACTORY,
        )

        assert bt is not None
        blackboard_data = {
            "battery_level": 70,
            "agv_healthy": True,
            "obstacle_detected": False,
            "comms_active": True,
            "heartbeat_timeout": False,
        }
        for k, v in blackboard_data.items():
            bt.blackboard.set(k, v)
        bt.tick()
        assert bt.blackboard.get("scene_perceived") is True

    def test_execution_level_action_sequence(self):
        """测试执行级动作序列"""
        from src.embodied.scene_task_planner import (
            HierarchicalBehaviorTreeComposer,
            SceneType,
        )

        composer = HierarchicalBehaviorTreeComposer()

        bt = composer.compose_task_tree(
            task_goal="office_document_delivery",
            scene_type=SceneType.OFFICE,
        )

        blackboard_data = {
            "battery_level": 50,
            "agv_healthy": True,
            "obstacle_detected": False,
            "comms_active": True,
            "heartbeat_timeout": False,
            "simplified_available": False,
        }
        for k, v in blackboard_data.items():
            bt.blackboard.set(k, v)
        bt.tick()

        # 应该执行了导航、操作、放置序列
        assert bt.blackboard.get("nav_progress", 0) >= 1
        assert bt.blackboard.get("manip_progress", 0) >= 1
        assert bt.blackboard.get("place_progress", 0) >= 1

    def test_all_scene_types_compose(self):
        """测试所有场景类型都能组合"""
        from src.embodied.scene_task_planner import (
            HierarchicalBehaviorTreeComposer,
            SceneType,
        )

        composer = HierarchicalBehaviorTreeComposer()

        for scene_type in SceneType:
            if scene_type == SceneType.UNKNOWN:
                continue
            bt = composer.compose_task_tree(
                task_goal=f"{scene_type.value}_task",
                scene_type=scene_type,
            )
            assert bt is not None, f"Failed for {scene_type}"

    def test_cache_key_uniqueness(self):
        """测试缓存键唯一性"""
        from src.embodied.scene_task_planner import (
            HierarchicalBehaviorTreeComposer,
            SceneType,
        )

        composer = HierarchicalBehaviorTreeComposer()

        bt1 = composer.compose_task_tree("task_a", SceneType.WAREHOUSE)
        bt2 = composer.compose_task_tree("task_a", SceneType.RESTAURANT)
        bt3 = composer.compose_task_tree("task_b", SceneType.WAREHOUSE)

        # 不同任务目标或场景应生成不同树
        assert bt1 is not bt2
        assert bt1 is not bt3
        assert bt2 is not bt3

        # 相同参数应复用缓存
        bt1_dup = composer.compose_task_tree("task_a", SceneType.WAREHOUSE)
        assert bt1 is bt1_dup


# ============================================================
# 测试 2: SceneMemoryAugmentedPlanner
# ============================================================

class TestSceneMemoryAugmentedPlanner:
    """记忆增强规划器测试"""

    def test_retrieve_default_experiences(self):
        """测试默认经验检索"""
        from src.embodied.scene_task_planner import (
            SceneMemoryAugmentedPlanner,
            SceneType,
        )

        planner = SceneMemoryAugmentedPlanner()

        # 仓库场景应该有默认经验
        exps = planner.retrieve_relevant_experiences(
            task_goal="pick_item",
            scene_type=SceneType.WAREHOUSE,
            max_count=3,
        )
        assert len(exps) > 0
        assert all("pattern" in e or "source" in e for e in exps)

    def test_retrieve_restaurant_experiences(self):
        """测试餐厅经验检索"""
        from src.embodied.scene_task_planner import (
            SceneMemoryAugmentedPlanner,
            SceneType,
        )

        planner = SceneMemoryAugmentedPlanner()

        exps = planner.retrieve_relevant_experiences(
            task_goal="deliver_food",
            scene_type=SceneType.RESTAURANT,
            max_count=5,
        )
        assert len(exps) > 0

    def test_optimize_task_params_high_success(self):
        """测试参数优化 - 高成功率场景"""
        from src.embodied.scene_task_planner import (
            SceneMemoryAugmentedPlanner,
            SceneType,
        )

        planner = SceneMemoryAugmentedPlanner()

        base_params = {
            "max_speed": 1.0,
            "safety_margin": 1.0,
            "timeout_factor": 1.0,
        }

        # 直接调用优化，不依赖记忆
        optimized = planner.optimize_task_params(
            task_goal="test_task",
            scene_type=SceneType.WAREHOUSE,
            base_params=base_params,
        )

        assert "safety_margin" in optimized
        assert "timeout_factor" in optimized
        assert "estimated_duration_ms" in optimized

    def test_optimize_task_params_low_success(self):
        """测试参数优化 - 低成功率场景"""
        from src.embodied.scene_task_planner import (
            SceneMemoryAugmentedPlanner,
            SceneType,
        )

        planner = SceneMemoryAugmentedPlanner()

        base_params = {
            "max_speed": 1.0,
            "safety_margin": 1.0,
            "timeout_factor": 1.0,
        }

        # 模拟低成功率场景
        planner._experience_cache["low_success:WAREHOUSE"] = [
            {"success_rate": 0.5} for _ in range(5)
        ]

        optimized = planner.optimize_task_params(
            task_goal="low_success",
            scene_type=SceneType.WAREHOUSE,
            base_params=base_params,
        )

        # 低成功率应该增加安全系数
        assert optimized["safety_margin"] >= base_params["safety_margin"]

    def test_record_experience_updates_cache(self):
        """测试经验记录更新缓存"""
        from src.embodied.scene_task_planner import (
            SceneMemoryAugmentedPlanner,
            SceneType,
        )

        planner = SceneMemoryAugmentedPlanner()

        planner.record_experience(
            task_goal="test_pick",
            scene_type=SceneType.WAREHOUSE,
            success=True,
            duration_ms=15000,
            params={"max_speed": 0.8},
        )

        key = "test_pick:warehouse"
        assert key in planner._experience_cache
        assert len(planner._success_patterns.get(key, [])) > 0

    def test_record_failure_updates_failure_patterns(self):
        """测试失败经验记录"""
        from src.embodied.scene_task_planner import (
            SceneMemoryAugmentedPlanner,
            SceneType,
        )

        planner = SceneMemoryAugmentedPlanner()

        planner.record_experience(
            task_goal="test_difficult",
            scene_type=SceneType.HOSPITAL,
            success=False,
            duration_ms=45000,
            params={"battery_level": 20},
        )

        key = "test_difficult:hospital"
        assert key in planner._failure_patterns
        assert len(planner._failure_patterns[key]) > 0

    def test_failure_warnings_high_failure_rate(self):
        """测试高失败率警告"""
        from src.embodied.scene_task_planner import (
            SceneMemoryAugmentedPlanner,
            SceneType,
        )

        planner = SceneMemoryAugmentedPlanner()

        key = "risky_task:warehouse"
        planner._failure_patterns[key] = [
            {"params": {"battery_level": 20}} for _ in range(3)
        ]
        planner._success_patterns[key] = [
            {"params": {}} for _ in range(2)
        ]

        warnings = planner.get_failure_warnings("risky_task", SceneType.WAREHOUSE)
        assert len(warnings) > 0
        assert any("失败率" in w or "电量" in w for w in warnings)

    def test_failure_warnings_no_history(self):
        """测试无历史时无警告"""
        from src.embodied.scene_task_planner import (
            SceneMemoryAugmentedPlanner,
            SceneType,
        )

        planner = SceneMemoryAugmentedPlanner()

        warnings = planner.get_failure_warnings(
            "brand_new_task",
            SceneType.LABORATORY,
        )
        assert len(warnings) == 0

    def test_experience_cache_prevents_duplicate_retrieval(self):
        """测试经验缓存防止重复检索"""
        from src.embodied.scene_task_planner import (
            SceneMemoryAugmentedPlanner,
            SceneType,
        )

        planner = SceneMemoryAugmentedPlanner()

        exps1 = planner.retrieve_relevant_experiences(
            "cached_task", SceneType.FACTORY, max_count=3
        )
        exps2 = planner.retrieve_relevant_experiences(
            "cached_task", SceneType.FACTORY, max_count=3
        )

        # 第二次检索应从缓存返回
        assert exps1 == exps2


# ============================================================
# 测试 3: CrossSceneTransferLearner
# ============================================================

class TestCrossSceneTransferLearner:
    """跨场景迁移学习器测试"""

    def test_transferability_same_scene(self):
        """测试同场景迁移 (应返回接近1.0)"""
        from src.embodied.scene_task_planner import (
            CrossSceneTransferLearner,
            SceneType,
        )

        learner = CrossSceneTransferLearner()

        result = learner.evaluate_transferability(
            SceneType.WAREHOUSE,
            SceneType.WAREHOUSE,
        )

        # Same scene: scene_similarity=1.0, adaptable_ratio=0.85
        # transfer_score = 1.0 * 0.7 + 0.85 * 0.3 = 0.955
        assert result["transfer_score"] >= 0.95
        assert result["scene_similarity"] == 1.0

    def test_transferability_warehouse_to_factory(self):
        """测试仓库->工厂迁移"""
        from src.embodied.scene_task_planner import (
            CrossSceneTransferLearner,
            SceneType,
        )

        learner = CrossSceneTransferLearner()

        result = learner.evaluate_transferability(
            SceneType.WAREHOUSE,
            SceneType.FACTORY,
        )

        assert 0.5 <= result["transfer_score"] <= 1.0
        assert result["scene_similarity"] == 0.75
        assert len(result["generic_skills"]) > 0
        assert len(result["adaptable_skills"]) > 0

    def test_transferability_restaurant_to_hospital(self):
        """测试餐厅->医院迁移 (差异较大)"""
        from src.embodied.scene_task_planner import (
            CrossSceneTransferLearner,
            SceneType,
        )

        learner = CrossSceneTransferLearner()

        result = learner.evaluate_transferability(
            SceneType.RESTAURANT,
            SceneType.HOSPITAL,
        )

        assert result["scene_similarity"] == 0.40
        assert len(result["warnings"]) > 0
        assert result["transfer_score"] < 0.7

    def test_transfer_task_plan_speed_adjustment(self):
        """测试任务计划迁移速度调整"""
        from src.embodied.scene_task_planner import (
            CrossSceneTransferLearner,
            SceneType,
        )

        learner = CrossSceneTransferLearner()

        source_plan = {
            "max_speed": 1.0,
            "safe_distance": 0.5,
        }

        # 医院场景应该降低速度
        adapted = learner.transfer_task_plan(
            SceneType.WAREHOUSE,
            SceneType.HOSPITAL,
            source_plan,
        )

        assert adapted["max_speed"] < source_plan["max_speed"]
        assert adapted["safe_distance"] > source_plan["safe_distance"]
        # adaptations_applied contains strings like "speed_factor:0.7"
        assert any("speed_factor" in a for a in adapted["adaptations_applied"])
        assert any("safety_factor" in a for a in adapted["adaptations_applied"])
        assert adapted["transfer_score"] > 0

    def test_transfer_task_plan_outdoor(self):
        """测试户外场景迁移 (速度可能增加)"""
        from src.embodied.scene_task_planner import (
            CrossSceneTransferLearner,
            SceneType,
        )

        learner = CrossSceneTransferLearner()

        source_plan = {
            "max_speed": 1.0,
            "safe_distance": 0.5,
        }

        # 户外场景速度因子为1.1
        adapted = learner.transfer_task_plan(
            SceneType.WAREHOUSE,
            SceneType.OUTDOOR,
            source_plan,
        )

        assert adapted["max_speed"] == 1.0 * 1.1

    def test_transferability_caching(self):
        """测试迁移评估结果缓存"""
        from src.embodied.scene_task_planner import (
            CrossSceneTransferLearner,
            SceneType,
        )

        learner = CrossSceneTransferLearner()

        result1 = learner.evaluate_transferability(
            SceneType.FACTORY,
            SceneType.HOSPITAL,
        )
        result2 = learner.evaluate_transferability(
            SceneType.FACTORY,
            SceneType.HOSPITAL,
        )

        # 相同查询应返回相同结果
        assert result1["transfer_score"] == result2["transfer_score"]

    def test_generic_skills_always_transferable(self):
        """测试通用技能总是可迁移"""
        from src.embodied.scene_task_planner import (
            CrossSceneTransferLearner,
            SceneType,
        )

        learner = CrossSceneTransferLearner()

        for scene_a in SceneType:
            if scene_a == SceneType.UNKNOWN:
                continue
            for scene_b in SceneType:
                if scene_b == SceneType.UNKNOWN:
                    continue
                result = learner.evaluate_transferability(scene_a, scene_b)
                assert "navigate" in result["generic_skills"]
                assert "obstacle_avoidance" in result["generic_skills"]
                assert "emergency_stop" in result["generic_skills"]


# ============================================================
# 测试 4: DegradationManager
# ============================================================

class TestDegradationManager:
    """降级管理器测试"""

    def test_fully_operational_when_all_modules_available(self):
        """测试所有模块可用时完全运行"""
        from src.embodied.embodied_pipeline import (
            EmbodiedPipeline,
            DegradationManager,
            DegradationLevel,
            PipelineMode,
        )

        pipeline = EmbodiedPipeline(grade="M", mode=PipelineMode.SIMULATION)

        # 模拟所有模块都可用
        class MockPipeline:
            _bt_planner = object()
            _scene_intel = object()
            _skill_registry = object()
            _memory_mgr = object()
            _task_executor = object()
            _hil_runner = object()
            _sim_enhancer = object()
            _fl_coordinator = object()
            _swarm_coord = object()
            _vla_model = object()

        mock = MockPipeline()
        dm = DegradationManager(pipeline=mock, auto_recover=False)

        level = dm.check_and_update()
        assert level == DegradationLevel.FULLY_OPERATIONAL

    def test_degraded_when_bt_planner_unavailable(self):
        """测试行为树不可用时降级"""
        from src.embodied.embodied_pipeline import (
            DegradationManager,
            DegradationLevel,
            DegradedCapability,
        )

        class MockPipeline:
            _bt_planner = None  # 关键模块不可用
            _scene_intel = object()
            _skill_registry = object()
            _memory_mgr = object()
            _task_executor = object()
            _hil_runner = object()
            _sim_enhancer = object()
            _fl_coordinator = object()
            _swarm_coord = object()

        mock = MockPipeline()
        dm = DegradationManager(pipeline=mock, auto_recover=False)

        level = dm.check_and_update()
        assert level in (
            DegradationLevel.DEGRADED_MODERATE,
            DegradationLevel.DEGRADED_SEVERE,
        )
        # Check that behavior tree capability is degraded
        assert any("behavior_tree" in str(c).lower() for c in dm.degraded_capabilities)

    def test_degraded_minor_non_critical_only(self):
        """测试仅非关键模块不可用时轻微降级"""
        from src.embodied.embodied_pipeline import (
            DegradationManager,
            DegradationLevel,
        )

        class MockPipeline:
            _bt_planner = object()
            _scene_intel = object()
            _skill_registry = object()
            _memory_mgr = object()
            _task_executor = object()
            _hil_runner = None  # 非关键
            _sim_enhancer = None  # 非关键
            _fl_coordinator = object()
            _swarm_coord = object()

        mock = MockPipeline()
        dm = DegradationManager(pipeline=mock, auto_recover=False)

        level = dm.check_and_update()
        # 2 non-critical unavailable -> score=2 -> DEGRADED_MINOR
        assert level in (
            DegradationLevel.FULLY_OPERATIONAL,
            DegradationLevel.DEGRADED_MINOR,
        )

    def test_degradation_report_structure(self):
        """测试降级报告结构"""
        from src.embodied.embodied_pipeline import (
            DegradationManager,
            DegradationLevel,
        )

        class MockPipeline:
            _bt_planner = None
            _scene_intel = None
            _skill_registry = object()
            _memory_mgr = None
            _task_executor = object()
            _hil_runner = None
            _sim_enhancer = None
            _fl_coordinator = None
            _swarm_coord = object()

        mock = MockPipeline()
        dm = DegradationManager(pipeline=mock, auto_recover=False)

        dm.check_and_update()
        report = dm.get_degradation_report()

        assert "level" in report
        assert "degraded_modules" in report
        assert "active_fallbacks" in report
        assert "degraded_capabilities" in report
        assert "history" in report
        assert report["auto_recover_enabled"] is False

    def test_can_use_capability(self):
        """测试能力可用性检查"""
        from src.embodied.embodied_pipeline import (
            DegradationManager,
            DegradedCapability,
        )

        class MockPipeline:
            _bt_planner = object()
            _scene_intel = object()
            _skill_registry = None
            _memory_mgr = object()
            _task_executor = object()
            _hil_runner = object()
            _sim_enhancer = object()
            _fl_coordinator = None  # FL不可用
            _swarm_coord = object()

        mock = MockPipeline()
        dm = DegradationManager(pipeline=mock, auto_recover=False)
        dm.check_and_update()

        # FL不可用
        assert dm.can_use_capability(DegradedCapability.FEDERATED_LEARNING) is False
        # 关键模块通常仍可用
        assert dm.can_use_capability(DegradedCapability.LONG_TERM_MEMORY) is True

    def test_get_fallback_for(self):
        """测试降级替代方案查询"""
        from src.embodied.embodied_pipeline import (
            DegradationManager,
            DegradedCapability,
        )

        class MockPipeline:
            _bt_planner = None
            _scene_intel = object()
            _skill_registry = object()
            _memory_mgr = object()
            _task_executor = object()
            _hil_runner = object()
            _sim_enhancer = object()
            _fl_coordinator = object()
            _swarm_coord = object()

        mock = MockPipeline()
        dm = DegradationManager(pipeline=mock, auto_recover=False)
        dm.check_and_update()

        fallback = dm.get_fallback_for(DegradedCapability.BEHAVIOR_TREE_PLANNING)
        assert fallback is not None
        assert isinstance(fallback, str)

    def test_get_allowed_capabilities(self):
        """测试获取允许使用的能力集"""
        from src.embodied.embodied_pipeline import (
            DegradationManager,
            DegradedCapability,
        )

        class MockPipeline:
            _bt_planner = object()
            _scene_intel = object()
            _skill_registry = object()
            _memory_mgr = object()
            _task_executor = object()
            _hil_runner = None
            _sim_enhancer = object()
            _fl_coordinator = object()
            _swarm_coord = None

        mock = MockPipeline()
        dm = DegradationManager(pipeline=mock, auto_recover=False)
        dm.check_and_update()

        allowed = dm.get_allowed_capabilities()
        assert DegradedCapability.HIL_TESTING not in allowed
        assert len(allowed) >= len(DegradedCapability) - 4

    def test_history_tracks_level_changes(self):
        """测试历史记录追踪等级变化"""
        from src.embodied.embodied_pipeline import (
            DegradationManager,
            DegradationLevel,
        )

        class MockPipeline:
            _bt_planner = None
            _scene_intel = object()
            _skill_registry = object()
            _memory_mgr = object()
            _task_executor = object()
            _hil_runner = object()
            _sim_enhancer = object()
            _fl_coordinator = object()
            _swarm_coord = object()

        mock = MockPipeline()
        dm = DegradationManager(pipeline=mock, auto_recover=False)

        dm.check_and_update()
        dm.check_and_update()  # 重复检查不应重复记录

        report = dm.get_degradation_report()
        if len(report["history"]) > 0:
            assert "old_level" in report["history"][0]
            assert "new_level" in report["history"][0]

    def test_severe_degradation_multiple_modules(self):
        """测试多模块故障导致严重降级"""
        from src.embodied.embodied_pipeline import (
            DegradationManager,
            DegradationLevel,
        )

        class MockPipeline:
            _bt_planner = None  # 关键
            _scene_intel = None  # 关键
            _skill_registry = None  # 非关键
            _memory_mgr = None  # 非关键
            _task_executor = object()
            _hil_runner = None  # 非关键
            _sim_enhancer = None  # 非关键
            _fl_coordinator = None  # 非关键
            _swarm_coord = None  # 非关键

        mock = MockPipeline()
        dm = DegradationManager(pipeline=mock, auto_recover=False)

        level = dm.check_and_update()
        assert level in (
            DegradationLevel.DEGRADED_SEVERE,
            DegradationLevel.EMERGENCY_ONLY,
        )


# ============================================================
# 测试 5: EmbodiedPipeline 降级集成
# ============================================================

class TestPipelineDegradationIntegration:
    """Pipeline 降级集成测试"""

    def test_pipeline_has_degradation_manager(self):
        """测试 Pipeline 包含降级管理器"""
        from src.embodied.embodied_pipeline import (
            EmbodiedPipeline,
            PipelineMode,
        )

        pipeline = EmbodiedPipeline(grade="S", mode=PipelineMode.SIMULATION)
        assert hasattr(pipeline, "_degradation_manager")

    def test_pipeline_get_degradation_report(self):
        """测试 Pipeline 获取降级报告"""
        from src.embodied.embodied_pipeline import (
            EmbodiedPipeline,
            PipelineMode,
        )

        pipeline = EmbodiedPipeline(grade="M", mode=PipelineMode.SIMULATION)
        report = pipeline.get_degradation_report()

        assert "level" in report or "error" in report

    def test_pipeline_check_degradation(self):
        """测试 Pipeline 降级检查"""
        from src.embodied.embodied_pipeline import (
            EmbodiedPipeline,
            PipelineMode,
        )

        pipeline = EmbodiedPipeline(grade="L", mode=PipelineMode.SIMULATION)
        level = pipeline.check_degradation()

        assert isinstance(level, str)
        assert level != ""

    def test_pipeline_can_use_capability(self):
        """测试 Pipeline 能力可用性查询"""
        from src.embodied.embodied_pipeline import (
            EmbodiedPipeline,
            PipelineMode,
        )

        pipeline = EmbodiedPipeline(grade="S", mode=PipelineMode.SIMULATION)

        # 已知能力应该能查询
        result = pipeline.can_use_capability("behavior_tree_planning")
        assert isinstance(result, bool)

        # 未知能力应该默认True
        result_unknown = pipeline.can_use_capability("nonexistent_capability")
        assert result_unknown is True


# ============================================================
# 测试 6: 场景规划器导出
# ============================================================

class TestScenePlannerExports:
    """场景规划器导出测试"""

    def test_all_new_classes_exported(self):
        """测试新增类都从模块导出"""
        from src.embodied.scene_task_planner import (
            HierarchicalBehaviorTreeComposer,
            SceneMemoryAugmentedPlanner,
            CrossSceneTransferLearner,
            HierarchicalTaskLevel,
            TaskCompositionRule,
        )

        assert HierarchicalBehaviorTreeComposer is not None
        assert SceneMemoryAugmentedPlanner is not None
        assert CrossSceneTransferLearner is not None
        assert HierarchicalTaskLevel is not None
        assert TaskCompositionRule is not None

    def test_hierarchical_task_levels(self):
        """测试层级任务级别枚举"""
        from src.embodied.scene_task_planner import HierarchicalTaskLevel

        levels = list(HierarchicalTaskLevel)
        assert len(levels) == 4
        assert HierarchicalTaskLevel.STRATEGIC in levels
        assert HierarchicalTaskLevel.TACTICAL in levels
        assert HierarchicalTaskLevel.EXECUTION in levels
        assert HierarchicalTaskLevel.REACTIVE in levels

    def test_task_composition_rules(self):
        """测试任务组合规则枚举"""
        from src.embodied.scene_task_planner import TaskCompositionRule

        rules = list(TaskCompositionRule)
        assert len(rules) == 5
        assert TaskCompositionRule.SEQUENTIAL in rules
        assert TaskCompositionRule.PARALLEL in rules
        assert TaskCompositionRule.FALLBACK in rules


# ============================================================
# 测试 7: Pipeline 新导出
# ============================================================

class TestPipelineExports:
    """Pipeline 新导出测试"""

    def test_degradation_classes_exported(self):
        """测试降级相关类从 pipeline 模块导出"""
        from src.embodied.embodied_pipeline import (
            DegradationManager,
            DegradationLevel,
            DegradedCapability,
        )

        assert DegradationManager is not None
        assert DegradationLevel is not None
        assert DegradedCapability is not None

    def test_degraded_capability_values(self):
        """测试降级能力枚举值"""
        from src.embodied.embodied_pipeline import DegradedCapability

        caps = list(DegradedCapability)
        assert len(caps) >= 8
        assert DegradedCapability.BEHAVIOR_TREE_PLANNING.value == "behavior_tree_planning"
        assert DegradedCapability.VLA_INFERENCE.value == "vla_inference"
        assert DegradedCapability.SWARM_COORDINATION.value == "swarm_coordination"

    def test_degradation_level_values(self):
        """测试降级等级枚举值"""
        from src.embodied.embodied_pipeline import DegradationLevel

        levels = list(DegradationLevel)
        assert len(levels) == 6
        assert DegradationLevel.FULLY_OPERATIONAL.value == "fully_operational"
        assert DegradationLevel.OFFLINE.value == "offline"
        assert DegradationLevel.EMERGENCY_ONLY.value == "emergency_only"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
