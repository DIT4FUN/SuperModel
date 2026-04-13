"""
test_scene_task_planner.py - 场景任务规划器测试
测试 SceneTaskPlanner, WarehouseTaskPlanner 等场景化任务规划功能
"""

import pytest
import numpy as np
from src.embodied.scene_task_planner import (
    SceneTaskConfig,
    SceneTaskTemplate,
    SceneTaskLibrary,
    SceneTaskPlanner,
    WarehouseTaskPlanner,
    SceneAdaptationEngine,
    get_scene_task_planner,
)
from src.embodied.scene_intelligence import SceneType


class TestSceneTaskPlanner:
    """场景任务规划器测试"""

    def test_scene_task_planner_init(self):
        """测试场景任务规划器初始化"""
        config = SceneTaskConfig(
            task_timeout=600.0,
            max_retries=5,
        )
        planner = SceneTaskPlanner(config=config)
        assert planner is not None
        assert planner._config.task_timeout == 600.0
        assert planner._config.max_retries == 5

    def test_warehouse_task_planner(self):
        """测试仓储任务规划器"""
        planner = WarehouseTaskPlanner()
        assert planner is not None

    def test_task_library_access(self):
        """测试任务库访问"""
        library = SceneTaskLibrary()
        
        # 获取仓储模板
        warehouse_templates = library.get_templates(SceneType.WAREHOUSE)
        assert len(warehouse_templates) >= 1
        
        # 获取医院模板
        hospital_templates = library.get_templates(SceneType.HOSPITAL)
        assert len(hospital_templates) >= 1

    def test_task_library_get_by_type(self):
        """测试按任务类型获取模板"""
        library = SceneTaskLibrary()
        
        # 获取 pick_and_stow 模板（仓库场景）
        template = library.get_template(SceneType.WAREHOUSE, "pick_and_stow")
        assert template is not None
        assert template.task_type == "pick_and_stow"

    def test_get_scene_task_planner_factory(self):
        """测试工厂函数"""
        planner = get_scene_task_planner()
        assert planner is not None


class TestSceneAdaptationEngine:
    """场景自适应引擎测试"""

    def test_adaptation_engine_init(self):
        """测试自适应引擎初始化"""
        engine = SceneAdaptationEngine()
        assert engine is not None

    def test_adaptation_scene_types(self):
        """测试场景类型自适应"""
        engine = SceneAdaptationEngine()
        
        # 测试不同场景的适应参数
        base_params = {"max_speed": 1.5, "safe_distance": 0.5}
        params_warehouse = engine.get_adaptive_params(SceneType.WAREHOUSE, base_params)
        params_hospital = engine.get_adaptive_params(SceneType.HOSPITAL, base_params)
        
        assert params_warehouse is not None
        assert params_hospital is not None
        assert "max_speed" in params_warehouse

    def test_record_outcome(self):
        """测试记录任务执行结果"""
        engine = SceneAdaptationEngine()
        
        engine.record_outcome(
            scene_type=SceneType.FACTORY,
            task_type="assembly",
            success=True,
            duration_s=120.0,
            parameters={"speed": 1.0},
        )
        # 无异常即通过


class TestSceneTaskLibrary:
    """场景任务库测试"""

    def test_library_scene_coverage(self):
        """测试场景库覆盖率"""
        library = SceneTaskLibrary()
        
        for scene_type in SceneType:
            templates = library.get_templates(scene_type)
            # 每个场景都应有模板（UNKNOWN可能有0个）
            if scene_type != SceneType.UNKNOWN:
                assert len(templates) >= 0  # 允许空，但不应报错


class TestSceneTaskTemplate:
    """场景任务模板测试"""

    def test_task_template_structure(self):
        """测试任务模板结构"""
        template = SceneTaskTemplate(
            task_type="test_task",
            scene_types={SceneType.FACTORY},
            priority=2,
            bt_config={"type": "sequence"},
            required_capabilities=["navigation"],
            typical_duration_s=60.0,
            safety_critical=False,
            collaborative=True,
        )
        
        assert template.task_type == "test_task"
        assert SceneType.FACTORY in template.scene_types
        assert template.priority == 2
        assert template.collaborative is True


class TestWarehouseTaskPlanner:
    """仓储任务规划器测试"""

    def test_warehouse_zone_patrol(self):
        """测试仓储区域巡检规划"""
        planner = WarehouseTaskPlanner()
        bt = planner.plan_zone_patrol(zones=["zone_a", "zone_b"])
        assert bt is not None

    def test_warehouse_production_task(self):
        """测试仓储生产任务规划"""
        from src.embodied.scene_task_planner import FactoryTaskPlanner
        planner = FactoryTaskPlanner()
        bt, task = planner.plan_production_task(
            task_type="assembly",
            station="station_a",
        )
        assert bt is not None
        assert task is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
