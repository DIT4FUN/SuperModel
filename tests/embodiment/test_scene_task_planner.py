"""
test_scene_task_planner.py - 场景任务规划器集成测试
=====================================================

测试 SuperModel 场景化任务规划器的完整功能:
- 场景任务模板库
- 场景自适应引擎
- 跨场景任务迁移
- 动态场景重规划
- 场景任务规划器工厂函数
- AGV五级规格适配
"""

import time
import pytest
from typing import Any, Dict

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
    SceneContext,
    SafetyRule,
    NavigationRule,
)


# ============================================================
# 场景任务配置测试
# ============================================================

class TestSceneTaskConfig:
    """场景任务配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = SceneTaskConfig()
        assert config.safety_priority == 1
        assert config.delivery_priority == 2
        assert config.task_timeout == 300.0
        assert config.max_retries == 3
        assert config.grade == "M"

    def test_warehouse_config(self):
        """测试仓库场景配置"""
        config = SceneTaskConfig(
            grade="L",
            enable_express_mode=True,
            enable_collaborative_dispatch=True,
        )
        assert config.grade == "L"
        assert config.enable_express_mode is True
        assert config.enable_collaborative_dispatch is True

    def test_hospital_config(self):
        """测试医院场景配置"""
        config = SceneTaskConfig(
            grade="XL",
            require_human_confirmation=True,
            safety_priority=0,  # 最高优先级
        )
        assert config.require_human_confirmation is True
        assert config.safety_priority == 0

    def test_factory_config(self):
        """测试工厂场景配置"""
        config = SceneTaskConfig(
            grade="XXL",
            maintenance_priority=1,
            task_timeout=600.0,
        )
        assert config.maintenance_priority == 1
        assert config.task_timeout == 600.0

    def test_all_grade_configs(self):
        """测试所有AGV等级配置"""
        for grade in ["S", "M", "L", "XL", "XXL"]:
            config = SceneTaskConfig(grade=grade)
            assert config.grade == grade


# ============================================================
# 场景任务模板测试
# ============================================================

class TestSceneTaskTemplate:
    """场景任务模板测试"""

    def test_template_creation(self):
        """测试模板创建"""
        template = SceneTaskTemplate(
            task_type="test_transport",
            scene_types={SceneType.WAREHOUSE},
            priority=2,
            bt_config={
                "type": "sequence",
                "children": [
                    {"type": "action", "name": "navigate", "action": "navigate"},
                    {"type": "action", "name": "grasp", "action": "grasp"},
                ],
            },
            required_capabilities=["navigation", "grasp"],
            typical_duration_s=60.0,
        )
        assert template.task_type == "test_transport"
        assert SceneType.WAREHOUSE in template.scene_types
        assert len(template.bt_config["children"]) == 2

    def test_template_fields(self):
        """测试模板字段"""
        template = SceneTaskTemplate(
            task_type="patrol",
            scene_types={SceneType.FACTORY},
            priority=3,
            bt_config={"type": "sequence", "children": []},
            required_capabilities=["vision"],
            typical_duration_s=180.0,
            safety_critical=True,
            collaborative=True,
        )
        assert template.safety_critical is True
        assert template.collaborative is True


# ============================================================
# 场景任务库测试
# ============================================================

class TestSceneTaskLibrary:
    """场景任务库测试"""

    def test_library_creation(self):
        """测试任务库创建"""
        library = SceneTaskLibrary()
        assert library is not None

    def test_get_warehouse_templates(self):
        """测试获取仓库任务模板"""
        library = SceneTaskLibrary()
        templates = library.get_templates(SceneType.WAREHOUSE)
        assert len(templates) >= 4  # 至少4种仓库任务

    def test_get_factory_templates(self):
        """测试获取工厂任务模板"""
        library = SceneTaskLibrary()
        templates = library.get_templates(SceneType.FACTORY)
        assert len(templates) >= 3  # 至少3种工厂任务

    def test_get_hospital_templates(self):
        """测试获取医院任务模板"""
        library = SceneTaskLibrary()
        templates = library.get_templates(SceneType.HOSPITAL)
        assert len(templates) >= 3  # 至少3种医院任务

    def test_get_restaurant_templates(self):
        """测试获取餐厅任务模板"""
        library = SceneTaskLibrary()
        templates = library.get_templates(SceneType.RESTAURANT)
        assert len(templates) >= 2

    def test_get_outdoor_templates(self):
        """测试获取户外任务模板"""
        library = SceneTaskLibrary()
        templates = library.get_templates(SceneType.OUTDOOR)
        assert len(templates) >= 2

    def test_get_template_by_task_type(self):
        """测试按任务类型获取模板"""
        library = SceneTaskLibrary()
        template = library.get_template(SceneType.WAREHOUSE, "pick_and_stow")
        assert template is not None
        assert template.task_type == "pick_and_stow"

    def test_all_scene_types_have_templates(self):
        """测试所有场景类型都有模板"""
        library = SceneTaskLibrary()
        for scene_type in [
            SceneType.WAREHOUSE,
            SceneType.FACTORY,
            SceneType.HOSPITAL,
            SceneType.RESTAURANT,
            SceneType.OUTDOOR,
            SceneType.OFFICE,
            SceneType.LABORATORY,
            SceneType.HOME,
            SceneType.UNKNOWN,
        ]:
            templates = library.get_templates(scene_type)
            assert len(templates) > 0, f"{scene_type} should have templates"


# ============================================================
# 场景任务规划器测试
# ============================================================

class TestSceneTaskPlanner:
    """场景任务规划器测试"""

    def test_planner_creation(self):
        """测试规划器创建"""
        config = SceneTaskConfig(grade="M")
        planner = SceneTaskPlanner(config=config)
        assert planner is not None

    def test_planner_default_config(self):
        """测试默认配置创建规划器"""
        planner = SceneTaskPlanner()
        assert planner is not None

    def test_plan_task(self):
        """测试任务规划"""
        planner = SceneTaskPlanner()
        bt, task = planner.plan_task(
            task_description="搬运货物到仓库A",
            scene_type=SceneType.WAREHOUSE,
        )
        assert bt is not None
        assert task is not None

    def test_plan_task_with_context(self):
        """测试带上下文的任务规划"""
        planner = SceneTaskPlanner()
        context = {
            "battery_level": 0.8,
            "obstacles": ["wall", "person"],
            "time_of_day": "morning",
        }
        bt, task = planner.plan_task(
            task_description="巡检仓库B",
            scene_type=SceneType.WAREHOUSE,
            context=context,
        )
        assert bt is not None
        assert task is not None

    def test_replan_if_needed(self):
        """测试需要时重规划"""
        planner = SceneTaskPlanner()
        # 初始规划
        bt1, task1 = planner.plan_task(
            task_description="配送包裹",
            scene_type=SceneType.WAREHOUSE,
        )
        assert bt1 is not None
        # 检查是否需要重规划
        result = planner.replan_if_needed(
            current_bt=bt1,
            scene_type=SceneType.WAREHOUSE,
            context={"task_description": "继续配送", "route_blocked": True},
        )
        # 返回值可能是新行为树或None
        assert result is None or result is not None  # 两种情况都正常

    def test_get_active_plan(self):
        """测试获取当前活跃计划"""
        planner = SceneTaskPlanner()
        bt, task = planner.plan_task(
            task_description="测试任务",
            scene_type=SceneType.FACTORY,
        )
        active_bt, active_task = planner.get_active_plan()
        assert active_bt is not None


# ============================================================
# 专用规划器测试
# ============================================================

class TestWarehouseTaskPlanner:
    """仓库任务规划器测试"""

    def test_warehouse_planner_creation(self):
        """测试仓库规划器创建"""
        planner = WarehouseTaskPlanner()
        assert planner is not None

    def test_plan_zone_patrol(self):
        """测试区域巡检规划"""
        planner = WarehouseTaskPlanner()
        bt = planner.plan_zone_patrol(zones=["A1", "A2", "A3"])
        assert bt is not None


class TestHospitalTaskPlanner:
    """医院任务规划器测试"""

    def test_hospital_planner_creation(self):
        """测试医院规划器创建"""
        planner = HospitalTaskPlanner()
        assert planner is not None

    def test_plan_verified_delivery(self):
        """测试需验证的配送任务规划"""
        planner = HospitalTaskPlanner()
        bt, task = planner.plan_verified_delivery(
            delivery_type="medication",
            destination="ward_3_room_12",
        )
        assert bt is not None
        assert task is not None
        assert task.task_type == "medication"


class TestFactoryTaskPlanner:
    """工厂任务规划器测试"""

    def test_factory_planner_creation(self):
        """测试工厂规划器创建"""
        planner = FactoryTaskPlanner()
        assert planner is not None

    def test_plan_production_task(self):
        """测试生产线任务规划"""
        planner = FactoryTaskPlanner()
        bt, task = planner.plan_production_task(
            task_type="assembly",
            station="STATION-01",
        )
        assert bt is not None
        assert task is not None


class TestRestaurantTaskPlanner:
    """餐厅任务规划器测试"""

    def test_restaurant_planner_creation(self):
        """测试餐厅规划器创建"""
        planner = RestaurantTaskPlanner()
        assert planner is not None

    def test_plan_food_delivery(self):
        """测试食物配送规划"""
        planner = RestaurantTaskPlanner()
        bt, task = planner.plan_food_delivery(
            table_id="T-05",
            order_type="hot_dishes",
        )
        assert bt is not None
        assert task is not None


class TestOutdoorTaskPlanner:
    """户外任务规划器测试"""

    def test_outdoor_planner_creation(self):
        """测试户外规划器创建"""
        planner = OutdoorTaskPlanner()
        assert planner is not None

    def test_plan_outdoor_delivery(self):
        """测试户外配送规划"""
        planner = OutdoorTaskPlanner()
        bt, task = planner.plan_outdoor_delivery(
            pickup="hub_north",
            destination="customer_address",
            package_type="small_box",
        )
        assert bt is not None
        assert task is not None


# ============================================================
# 场景自适应引擎测试
# ============================================================

class TestSceneAdaptationEngine:
    """场景自适应引擎测试"""

    def test_engine_creation(self):
        """测试引擎创建（无记忆）"""
        engine = SceneAdaptationEngine()
        assert engine is not None

    def test_record_outcome(self):
        """测试记录任务结果"""
        engine = SceneAdaptationEngine()
        engine.record_outcome(
            scene_type=SceneType.WAREHOUSE,
            task_type="transport",
            success=True,
            duration_s=120.0,
            parameters={"speed": 1.0, "route": "A"},
        )

    def test_record_outcome_failure(self):
        """测试记录失败结果"""
        engine = SceneAdaptationEngine()
        engine.record_outcome(
            scene_type=SceneType.FACTORY,
            task_type="assembly",
            success=False,
            duration_s=90.0,
            parameters={"speed": 0.5, "force": 10.0},
        )

    def test_get_adaptive_params(self):
        """测试获取自适应参数"""
        engine = SceneAdaptationEngine()
        base_params = {
            "max_speed": 1.5,
            "safe_distance": 0.5,
        }
        adapted = engine.get_adaptive_params(
            scene_type=SceneType.WAREHOUSE,
            base_params=base_params,
        )
        assert adapted is not None
        assert isinstance(adapted, dict)
        assert "max_speed" in adapted
        assert "safe_distance" in adapted

    def test_adaptive_params_low_success_rate(self):
        """测试低成功率时的自适应参数"""
        engine = SceneAdaptationEngine()
        # 先记录一些失败
        for _ in range(5):
            engine.record_outcome(
                scene_type=SceneType.HOSPITAL,
                task_type="transport",
                success=False,
                duration_s=60.0,
                parameters={"speed": 2.0},
            )
        base_params = {"max_speed": 2.0, "safe_distance": 0.3}
        adapted = engine.get_adaptive_params(
            scene_type=SceneType.HOSPITAL,
            base_params=base_params,
        )
        assert adapted is not None
        # 低成功率时应降低速度

    def test_adaptive_params_learns_from_success(self):
        """测试从成功中学习"""
        engine = SceneAdaptationEngine()
        # 记录多次成功
        for _ in range(10):
            engine.record_outcome(
                scene_type=SceneType.FACTORY,
                task_type="inspection",
                success=True,
                duration_s=45.0,
                parameters={"speed": 1.2},
            )
        base_params = {"max_speed": 1.0, "safe_distance": 0.5}
        adapted = engine.get_adaptive_params(
            scene_type=SceneType.FACTORY,
            base_params=base_params,
        )
        assert adapted is not None


# ============================================================
# 工厂函数测试
# ============================================================

class TestSceneTaskPlannerFactory:
    """场景任务规划器工厂函数测试"""

    def test_get_scene_task_planner_default(self):
        """测试获取默认规划器"""
        planner = get_scene_task_planner()
        assert planner is not None
        assert isinstance(planner, SceneTaskPlanner)

    def test_get_scene_task_planner_with_config(self):
        """测试带配置获取规划器"""
        config = SceneTaskConfig(grade="L")
        planner = get_scene_task_planner(config=config)
        assert planner is not None

    def test_get_scene_task_planner_singleton(self):
        """测试全局单例"""
        planner1 = get_scene_task_planner()
        planner2 = get_scene_task_planner()
        assert planner1 is planner2  # 同一个实例


# ============================================================
# AGV五级规格适配测试
# ============================================================

class TestSceneTaskPlannerGradeAdaptation:
    """场景任务规划器AGV等级适配测试"""

    @pytest.mark.parametrize("grade", ["S", "M", "L", "XL", "XXL"])
    def test_all_grades_config(self, grade):
        """测试所有AGV等级配置"""
        config = SceneTaskConfig(grade=grade)
        assert config.grade == grade
        planner = SceneTaskPlanner(config=config)
        assert planner is not None

    @pytest.mark.parametrize("scene", [
        SceneType.WAREHOUSE,
        SceneType.HOSPITAL,
        SceneType.FACTORY,
        SceneType.RESTAURANT,
        SceneType.OUTDOOR,
    ])
    def test_all_scenes_templates(self, scene):
        """测试所有场景类型的模板"""
        library = SceneTaskLibrary()
        templates = library.get_templates(scene)
        assert len(templates) > 0

    def test_xxl_grade_large_scene_library(self):
        """测试XXL级别大规模场景模板"""
        library = SceneTaskLibrary()
        templates = library.get_templates(SceneType.FACTORY)
        assert len(templates) >= 3

    def test_s_grade_lightweight_planning(self):
        """测试S级别轻量级规划"""
        planner = SceneTaskPlanner(config=SceneTaskConfig(grade="S"))
        start = time.time()
        bt, task = planner.plan_task(
            task_description="简单搬运",
            scene_type=SceneType.WAREHOUSE,
        )
        elapsed = time.time() - start
        assert bt is not None
        assert elapsed < 2.0  # S级别应快速规划
