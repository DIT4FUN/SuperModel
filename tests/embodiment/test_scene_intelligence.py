"""
test_scene_intelligence.py - 场景智能测试
测试 SceneIntelligence, SceneType, SafetyRule 等场景智能功能
"""

import pytest
import numpy as np
from src.embodied.scene_intelligence import (
    SceneType,
    SceneContext,
    SceneConfig,
    SceneIntelligence,
    SafetyRule,
    NavigationRule,
    InteractionRule,
    SceneFeatures,
    get_scene_intelligence,
)


class TestSceneIntelligence:
    """场景智能测试"""

    def test_scene_intelligence_init(self):
        """测试场景智能初始化"""
        config = SceneConfig(
            detection_interval=1.0,
        )
        si = SceneIntelligence(config=config)
        assert si is not None
        assert si.config.detection_interval == 1.0

    def test_scene_context_update(self):
        """测试场景上下文更新"""
        config = SceneConfig()
        si = SceneIntelligence(config=config)
        
        context = si.update(
            laser_ranges=np.array([5.0] * 360),
            vision_features={"openness": 0.5},
            location_hint="warehouse",
            nearby_humans=2,
        )
        assert context is not None
        assert context.nearby_humans == 2

    def test_get_scene_intelligence_factory(self):
        """测试场景智能工厂函数"""
        si = get_scene_intelligence()
        assert si is not None
        assert isinstance(si, SceneIntelligence)

    def test_get_safe_distance(self):
        """测试安全距离获取"""
        config = SceneConfig()
        si = SceneIntelligence(config=config)
        
        distance = si.get_safe_distance()
        assert isinstance(distance, float)
        assert distance > 0.0

    def test_get_adaptive_speed_limit(self):
        """测试自适应速度限制"""
        config = SceneConfig()
        si = SceneIntelligence(config=config)
        
        speed = si.get_adaptive_speed_limit(base_speed=1.5)
        assert isinstance(speed, float)
        assert speed > 0.0

    def test_active_rules(self):
        """测试活动规则获取"""
        config = SceneConfig()
        si = SceneIntelligence(config=config)
        
        rules = si.get_active_rules()
        assert isinstance(rules, dict)

    def test_scene_context(self):
        """测试场景上下文"""
        config = SceneConfig()
        si = SceneIntelligence(config=config)
        
        context = si.get_scene_context()
        assert isinstance(context, SceneContext)


class TestSceneType:
    """场景类型枚举测试"""

    def test_all_scene_types(self):
        """测试所有场景类型"""
        expected_types = ["warehouse", "hospital", "factory", "restaurant", "outdoor"]
        actual = [st.value for st in SceneType]
        for expected in expected_types:
            assert expected in actual


class TestSafetyRule:
    """安全规则测试"""

    def test_safety_rule_creation(self):
        """测试安全规则创建"""
        rule = SafetyRule(
            rule_id="test_rule",
            scene_types={SceneType.WAREHOUSE},
            priority=1,
            enabled=True,
        )
        assert rule.rule_id == "test_rule"
        assert rule.priority == 1
        assert rule.enabled is True

    def test_safety_rule_defaults(self):
        """测试安全规则默认值"""
        rule = SafetyRule(
            rule_id="default_test",
            scene_types={SceneType.HOSPITAL},
        )
        assert rule.min_clearance == 0.3
        assert rule.max_speed == 1.5


class TestSceneContext:
    """场景上下文测试"""

    def test_scene_context_creation(self):
        """测试场景上下文创建"""
        ctx = SceneContext(
            current_location="warehouse_a1",
            floor_level=1,
            current_task="transport",
            nearby_humans=3,
        )
        assert ctx.current_location == "warehouse_a1"
        assert ctx.floor_level == 1
        assert ctx.nearby_humans == 3

    def test_scene_context_features(self):
        """测试场景上下文特征"""
        features = SceneFeatures(
            scene_type=SceneType.FACTORY,
            obstacle_density=0.2,
            human_density=0.1,
            floor_friction=0.7,
        )
        ctx = SceneContext(features=features)
        assert ctx.features.scene_type == SceneType.FACTORY
        assert ctx.features.human_density == 0.1

    def test_scene_features_safe_for_high_speed(self):
        """测试场景特征高速安全性判断"""
        features = SceneFeatures(
            obstacle_density=0.1,
            human_density=0.05,
            floor_friction=0.8,
        )
        assert features.is_safe_for_high_speed() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
