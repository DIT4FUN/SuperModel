"""
embodied_scene_task_tests.py - 场景任务规划器测试
tests/embodied_scene_task_tests.py

测试覆盖:
- SceneTaskLibrary 任务模板库
- SceneTaskPlanner 场景任务规划器
- WarehouseTaskPlanner 仓库任务规划
- HospitalTaskPlanner 医院任务规划
- FactoryTaskPlanner 工厂任务规划
- RestaurantTaskPlanner 餐厅任务规划
- OutdoorTaskPlanner 户外任务规划
- SceneAdaptationEngine 场景适应引擎
"""

import pytest
import time
import numpy as np
from typing import Any, Dict, List

from src.embodied.scene_task_planner import (
    SceneTaskConfig,
    SceneTaskTemplate,
    SceneTaskLibrary,
    SceneTaskPlanner,
    WarehouseTaskPlanner,
    HospitalTaskPlanner,
    FactoryTaskPlanner,
    RestaurantTaskPlanner,
    OutdoorTaskPlanner,
    SceneAdaptationEngine,
    get_scene_task_planner,
)
from src.embodied.scene_intelligence import (
    SceneType,
    SceneIntelligence,
    SceneConfig,
)
from src.embodied.behavior_tree import NodeStatus


# ============================================================
# SceneTaskLibrary Tests
# ============================================================

class TestSceneTaskLibrary:
    """SceneTaskLibrary 任务模板库测试"""
    
    def test_library_initialization(self):
        """库初始化"""
        library = SceneTaskLibrary()
        assert SceneType.WAREHOUSE in library._templates
        assert SceneType.HOSPITAL in library._templates
        assert SceneType.FACTORY in library._templates
    
    def test_get_warehouse_templates(self):
        """获取仓库模板"""
        library = SceneTaskLibrary()
        templates = library.get_templates(SceneType.WAREHOUSE)
        assert len(templates) >= 3
        task_types = [t.task_type for t in templates]
        assert "pick_and_stow" in task_types
        assert "inventory_patrol" in task_types
    
    def test_get_hospital_templates(self):
        """获取医院模板"""
        library = SceneTaskLibrary()
        templates = library.get_templates(SceneType.HOSPITAL)
        assert len(templates) >= 2
        task_types = [t.task_type for t in templates]
        assert "medication_delivery" in task_types
        assert "specimen_transport" in task_types
    
    def test_get_factory_templates(self):
        """获取工厂模板"""
        library = SceneTaskLibrary()
        templates = library.get_templates(SceneType.FACTORY)
        assert len(templates) >= 1
        task_types = [t.task_type for t in templates]
        assert "production_line_feed" in task_types
    
    def test_get_restaurant_templates(self):
        """获取餐厅模板"""
        library = SceneTaskLibrary()
        templates = library.get_templates(SceneType.RESTAURANT)
        assert len(templates) >= 1
        task_types = [t.task_type for t in templates]
        assert "food_delivery" in task_types
    
    def test_get_outdoor_templates(self):
        """获取户外模板"""
        library = SceneTaskLibrary()
        templates = library.get_templates(SceneType.OUTDOOR)
        assert len(templates) >= 1
        task_types = [t.task_type for t in templates]
        assert "outdoor_delivery" in task_types
    
    def test_get_unknown_templates(self):
        """获取未知场景模板"""
        library = SceneTaskLibrary()
        templates = library.get_templates(SceneType.UNKNOWN)
        assert len(templates) >= 1
    
    def test_get_specific_template(self):
        """获取特定模板"""
        library = SceneTaskLibrary()
        tmpl = library.get_template(SceneType.WAREHOUSE, "pick_and_stow")
        assert tmpl is not None
        assert tmpl.priority == 1
        assert tmpl.safety_critical is False
        assert tmpl.collaborative is True
    
    def test_template_fields_complete(self):
        """模板字段完整性"""
        library = SceneTaskLibrary()
        templates = library.get_templates(SceneType.WAREHOUSE)
        for tmpl in templates:
            assert tmpl.task_type
            assert tmpl.scene_types
            assert tmpl.priority >= 0
            assert tmpl.bt_config
            assert tmpl.required_capabilities
            assert tmpl.typical_duration_s > 0
    
    def test_priority_ordering(self):
        """优先级排序"""
        library = SceneTaskLibrary()
        templates = library.get_templates(SceneType.HOSPITAL)
        # 安全关键任务应该有最高优先级
        safety_critical = [t for t in templates if t.safety_critical]
        non_critical = [t for t in templates if not t.safety_critical]
        assert all(sc.priority <= nc.priority for sc in safety_critical for nc in non_critical if nc.priority > 0)


# ============================================================
# SceneTaskPlanner Tests
# ============================================================

class TestSceneTaskPlanner:
    """SceneTaskPlanner 场景任务规划器测试"""
    
    def test_planner_initialization(self):
        """规划器初始化"""
        planner = SceneTaskPlanner()
        assert planner._config is not None
        assert planner._library is not None
    
    def test_plan_warehouse_task(self):
        """规划仓库任务"""
        planner = SceneTaskPlanner()
        bt, task = planner.plan_task(
            task_description="pick_and_stow",
            scene_type=SceneType.WAREHOUSE,
        )
        assert bt is not None
        assert task is not None
        assert "warehouse" in task.goal_description
        assert task.priority == 1
    
    def test_plan_hospital_task(self):
        """规划医院任务"""
        planner = SceneTaskPlanner()
        bt, task = planner.plan_task(
            task_description="medication_delivery",
            scene_type=SceneType.HOSPITAL,
        )
        assert bt is not None
        assert "hospital" in task.goal_description
    
    def test_plan_factory_task(self):
        """规划工厂任务"""
        planner = SceneTaskPlanner()
        bt, task = planner.plan_task(
            task_description="production_line_feed",
            scene_type=SceneType.FACTORY,
        )
        assert bt is not None
        assert "factory" in task.goal_description
    
    def test_plan_restaurant_task(self):
        """规划餐厅任务"""
        planner = SceneTaskPlanner()
        bt, task = planner.plan_task(
            task_description="food_delivery",
            scene_type=SceneType.RESTAURANT,
        )
        assert bt is not None
        assert "restaurant" in task.goal_description
    
    def test_plan_outdoor_task(self):
        """规划户外任务"""
        planner = SceneTaskPlanner()
        bt, task = planner.plan_task(
            task_description="outdoor_delivery",
            scene_type=SceneType.OUTDOOR,
        )
        assert bt is not None
        assert "outdoor" in task.goal_description
    
    def test_plan_unknown_scene(self):
        """规划未知场景"""
        planner = SceneTaskPlanner()
        bt, task = planner.plan_task(
            task_description="goto_location",
            scene_type=SceneType.UNKNOWN,
        )
        assert bt is not None
        assert task is not None
    
    def test_task_id_unique(self):
        """任务ID唯一性"""
        planner = SceneTaskPlanner()
        bt1, task1 = planner.plan_task("task1", SceneType.WAREHOUSE)
        time.sleep(0.01)
        bt2, task2 = planner.plan_task("task2", SceneType.WAREHOUSE)
        assert task1.task_id != task2.task_id
    
    def test_active_plan_tracking(self):
        """活动计划跟踪"""
        planner = SceneTaskPlanner()
        bt, task = planner.plan_task("test_task", SceneType.WAREHOUSE)
        active_bt, active_task = planner.get_active_plan()
        assert active_bt is bt
        assert active_task is task
    
    def test_plan_count_increments(self):
        """规划计数递增"""
        planner = SceneTaskPlanner()
        initial_count = planner._plan_count
        planner.plan_task("task1", SceneType.WAREHOUSE)
        assert planner._plan_count == initial_count + 1
    
    def test_plan_with_context(self):
        """带上下文的规划"""
        planner = SceneTaskPlanner()
        bt, task = planner.plan_task(
            task_description="pick_and_stow",
            scene_type=SceneType.WAREHOUSE,
            context={"target_item": "box_123", "stow_location": "shelf_A1"},
        )
        assert bt is not None
    
    def test_config_override(self):
        """配置覆盖"""
        config = SceneTaskConfig(
            task_timeout=600.0,
            max_retries=5,
            grade="XL",
        )
        planner = SceneTaskPlanner(config=config)
        assert planner._config.task_timeout == 600.0
        assert planner._config.max_retries == 5
        assert planner._config.grade == "XL"


# ============================================================
# WarehouseTaskPlanner Tests
# ============================================================

class TestWarehouseTaskPlanner:
    """WarehouseTaskPlanner 仓库专用规划器测试"""
    
    def test_initialization(self):
        """初始化"""
        planner = WarehouseTaskPlanner()
        assert planner is not None
    
    def test_plan_zone_patrol_single_zone(self):
        """单区域巡检规划"""
        planner = WarehouseTaskPlanner()
        bt = planner.plan_zone_patrol(["zone_A"])
        assert bt is not None
    
    def test_plan_zone_patrol_multiple_zones(self):
        """多区域巡检规划"""
        planner = WarehouseTaskPlanner()
        bt = planner.plan_zone_patrol(["zone_A", "zone_B", "zone_C"])
        assert bt is not None


# ============================================================
# HospitalTaskPlanner Tests
# ============================================================

class TestHospitalTaskPlanner:
    """HospitalTaskPlanner 医院专用规划器测试"""
    
    def test_initialization(self):
        """初始化"""
        planner = HospitalTaskPlanner()
        assert planner is not None
    
    def test_plan_verified_delivery(self):
        """规划需验证的配送"""
        planner = HospitalTaskPlanner()
        bt, task = planner.plan_verified_delivery(
            delivery_type="medication",
            destination="ward_301",
        )
        assert bt is not None
        assert "hospital" in task.goal_description
        assert "medication" in task.goal_description
        assert "ward_301" in task.goal_description
    
    def test_plan_specimen_transport(self):
        """规划样本运输"""
        planner = HospitalTaskPlanner()
        bt, task = planner.plan_verified_delivery(
            delivery_type="specimen",
            destination="laboratory_2",
        )
        assert bt is not None
        assert "hospital" in task.goal_description


# ============================================================
# FactoryTaskPlanner Tests
# ============================================================

class TestFactoryTaskPlanner:
    """FactoryTaskPlanner 工厂专用规划器测试"""
    
    def test_initialization(self):
        """初始化"""
        planner = FactoryTaskPlanner()
        assert planner is not None
    
    def test_plan_production_task(self):
        """规划生产任务"""
        planner = FactoryTaskPlanner()
        bt, task = planner.plan_production_task(
            task_type="assembly",
            station="station_5",
        )
        assert bt is not None
        assert "factory" in task.goal_description
        assert "assembly" in task.goal_description
        assert "station_5" in task.goal_description


# ============================================================
# RestaurantTaskPlanner Tests
# ============================================================

class TestRestaurantTaskPlanner:
    """RestaurantTaskPlanner 餐厅专用规划器测试"""
    
    def test_initialization(self):
        """初始化"""
        planner = RestaurantTaskPlanner()
        assert planner is not None
    
    def test_plan_food_delivery(self):
        """规划食物配送"""
        planner = RestaurantTaskPlanner()
        bt, task = planner.plan_food_delivery(
            table_id="table_12",
            order_type="main_course",
        )
        assert bt is not None
        assert "restaurant" in task.goal_description
        assert "table_12" in task.goal_description
        assert "main_course" in task.goal_description


# ============================================================
# OutdoorTaskPlanner Tests
# ============================================================

class TestOutdoorTaskPlanner:
    """OutdoorTaskPlanner 户外专用规划器测试"""
    
    def test_initialization(self):
        """初始化"""
        planner = OutdoorTaskPlanner()
        assert planner is not None
    
    def test_plan_outdoor_delivery(self):
        """规划户外配送"""
        planner = OutdoorTaskPlanner()
        bt, task = planner.plan_outdoor_delivery(
            pickup="hub_north",
            destination="customer_address",
            package_type="package",
        )
        assert bt is not None
        assert "outdoor" in task.goal_description
        assert "hub_north" in task.goal_description


# ============================================================
# SceneAdaptationEngine Tests
# ============================================================

class TestSceneAdaptationEngine:
    """SceneAdaptationEngine 场景适应引擎测试"""
    
    def test_initialization(self):
        """初始化"""
        engine = SceneAdaptationEngine()
        assert engine is not None
        assert len(engine._scene_params) > 0
    
    def test_record_successful_outcome(self):
        """记录成功结果"""
        engine = SceneAdaptationEngine()
        engine.record_outcome(
            scene_type=SceneType.WAREHOUSE,
            task_type="pick_and_stow",
            success=True,
            duration_s=100.0,
            parameters={"max_speed": 1.5, "safe_distance": 0.5},
        )
        assert SceneType.WAREHOUSE in engine._scene_params
    
    def test_record_failed_outcome(self):
        """记录失败结果"""
        engine = SceneAdaptationEngine()
        initial_rate = engine._scene_params.get(SceneType.HOSPITAL, {}).get("success_rate", 0.8)
        engine.record_outcome(
            scene_type=SceneType.HOSPITAL,
            task_type="medication_delivery",
            success=False,
            duration_s=300.0,
            parameters={"max_speed": 0.8, "safe_distance": 0.8},
        )
        # 成功率应该下降
        new_rate = engine._scene_params[SceneType.HOSPITAL].get("success_rate", 0.8)
        assert new_rate < initial_rate
    
    def test_get_adaptive_params_no_adjustment(self):
        """无调整时的参数获取"""
        engine = SceneAdaptationEngine()
        base_params = {"max_speed": 1.5, "safe_distance": 0.5}
        adapted = engine.get_adaptive_params(SceneType.UNKNOWN, base_params)
        assert "max_speed" in adapted
        assert "safe_distance" in adapted
    
    def test_get_adaptive_params_speed_adjustment(self):
        """速度调整"""
        engine = SceneAdaptationEngine()
        # 设置低成功率
        engine._scene_params[SceneType.FACTORY] = {"success_rate": 0.6}
        
        base_params = {"max_speed": 1.5, "safe_distance": 0.5}
        adapted = engine.get_adaptive_params(SceneType.FACTORY, base_params)
        # 低成功率应该导致速度降低
        assert adapted["max_speed"] < base_params["max_speed"]
    
    def test_get_adaptive_params_caution_adjustment(self):
        """安全距离调整"""
        engine = SceneAdaptationEngine()
        # 设置低成功率
        engine._scene_params[SceneType.HOSPITAL] = {"success_rate": 0.65}
        
        base_params = {"max_speed": 1.5, "safe_distance": 0.5}
        adapted = engine.get_adaptive_params(SceneType.HOSPITAL, base_params)
        # 低成功率应该导致安全距离增大
        assert adapted["safe_distance"] > base_params["safe_distance"]
    
    def test_get_adaptive_params_high_success(self):
        """高成功率优化"""
        engine = SceneAdaptationEngine()
        # 设置高成功率
        engine._scene_params[SceneType.WAREHOUSE] = {"success_rate": 0.95}
        
        base_params = {"max_speed": 1.5, "safe_distance": 0.5}
        adapted = engine.get_adaptive_params(SceneType.WAREHOUSE, base_params)
        # 高成功率应该允许稍微提高速度
        assert adapted["max_speed"] >= base_params["max_speed"]


# ============================================================
# Integration Tests
# ============================================================

class TestSceneTaskPlannerIntegration:
    """场景任务规划器集成测试"""
    
    def test_global_singleton(self):
        """全局单例"""
        planner1 = get_scene_task_planner()
        planner2 = get_scene_task_planner()
        assert planner1 is planner2
    
    def test_scene_type_coverage(self):
        """所有场景类型覆盖"""
        planner = SceneTaskPlanner()
        for scene_type in SceneType:
            if scene_type == SceneType.UNKNOWN:
                continue
            bt, task = planner.plan_task(
                task_description="generic_task",
                scene_type=scene_type,
            )
            assert bt is not None
            assert task is not None
    
    def test_bt_execution_smoke(self):
        """行为树执行冒烟测试"""
        planner = SceneTaskPlanner()
        bt, task = planner.plan_task(
            task_description="pick_and_stow",
            scene_type=SceneType.WAREHOUSE,
        )
        # 创建黑板
        from src.embodied.behavior_tree import Blackboard
        blackboard = Blackboard()
        blackboard.set("battery_ok", True)
        blackboard.set("path_clear", True)
        
        # 执行根节点
        status = bt.tick(blackboard)
        assert status in [NodeStatus.SUCCESS, NodeStatus.RUNNING, NodeStatus.FAILURE]
    
    def test_multiple_plans_sequence(self):
        """连续多次规划"""
        planner = SceneTaskPlanner()
        scenes = [SceneType.WAREHOUSE, SceneType.HOSPITAL, SceneType.FACTORY]
        tasks = []
        for scene in scenes:
            bt, task = planner.plan_task("test_task", scene)
            tasks.append(task)
        
        # 所有任务应该有不同ID
        task_ids = [t.task_id for t in tasks]
        assert len(task_ids) == len(set(task_ids))


# ============================================================
# Performance Tests
# ============================================================

class TestSceneTaskPlannerPerformance:
    """场景任务规划器性能测试"""
    
    def test_planning_is_fast(self):
        """规划速度测试"""
        planner = SceneTaskPlanner()
        start = time.time()
        for _ in range(100):
            planner.plan_task("pick_and_stow", SceneType.WAREHOUSE)
        elapsed = time.time() - start
        assert elapsed < 5.0  # 100次规划应在5秒内完成
    
    def test_library_lookup_performance(self):
        """库查询性能"""
        library = SceneTaskLibrary()
        start = time.time()
        for _ in range(1000):
            library.get_templates(SceneType.WAREHOUSE)
            library.get_template(SceneType.HOSPITAL, "medication_delivery")
        elapsed = time.time() - start
        assert elapsed < 1.0  # 1000次查询应在1秒内完成


# ============================================================
# Error Handling Tests
# ============================================================

class TestSceneTaskPlannerErrorHandling:
    """场景任务规划器错误处理测试"""
    
    def test_unknown_task_falls_back(self):
        """未知任务回退到通用模板"""
        planner = SceneTaskPlanner()
        bt, task = planner.plan_task(
            task_description="completely_unknown_task_type_xyz",
            scene_type=SceneType.WAREHOUSE,
        )
        assert bt is not None
        assert task is not None
    
    def test_memory_integration_graceful(self):
        """记忆集成容错"""
        # 传入无效记忆对象
        class BadMemory:
            pass
        
        planner = SceneTaskPlanner(memory=BadMemory())
        # 应该不抛异常
        bt, task = planner.plan_task("test", SceneType.WAREHOUSE)
        assert bt is not None
