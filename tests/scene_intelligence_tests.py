"""
scene_intelligence_tests.py - 场景化具身智能测试
===============================================

测试:
- 场景分类识别
- 场景规则引擎
- 场景自适应行为
- 场景上下文更新
- 多AGV场景协同
"""

import pytest
import time
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embodied.scene_intelligence import (
    SceneType,
    SceneContext,
    SceneRule,
    SafetyRule,
    NavigationRule,
    InteractionRule,
    SceneFeatures,
    SceneIntelligence,
    SceneConfig,
    SceneClassifier,
    SceneRuleEngine,
    get_scene_intelligence,
)


# ============================================================
# 场景分类器测试
# ============================================================

class TestSceneClassifier:
    """场景分类器测试"""

    def test_warehouse_classification(self):
        """测试仓库场景分类"""
        clf = SceneClassifier()
        scene, conf, features = clf.classify(
            laser_ranges=np.array([3.0] * 360),
            vision_features={'open_space': 0.8, 'shelves_detected': 10},
            audio_activity=0.1,
            location_hint="warehouse",
        )
        assert scene == SceneType.WAREHOUSE
        assert conf >= 0.9

    def test_hospital_classification(self):
        """测试医院场景分类"""
        clf = SceneClassifier()
        scene, conf, features = clf.classify(
            laser_ranges=np.array([2.0] * 360),
            audio_activity=0.1,
            location_hint="hospital",
        )
        assert scene == SceneType.HOSPITAL
        assert conf >= 0.9

    def test_factory_classification(self):
        """测试工厂场景分类"""
        clf = SceneClassifier()
        scene, conf, features = clf.classify(
            laser_ranges=np.array([1.0] * 360),
            audio_activity=0.1,
            location_hint="factory",
        )
        assert scene == SceneType.FACTORY
        assert conf >= 0.9

    def test_restaurant_classification(self):
        """测试餐厅场景分类"""
        clf = SceneClassifier()
        scene, conf, features = clf.classify(
            laser_ranges=np.array([2.0] * 360),
            audio_activity=0.8,
            location_hint="restaurant",
        )
        assert scene == SceneType.RESTAURANT
        assert conf >= 0.9

    def test_outdoor_classification(self):
        """测试户外场景分类"""
        clf = SceneClassifier()
        scene, conf, features = clf.classify(
            location_hint="outdoor",
        )
        assert scene == SceneType.OUTDOOR
        assert conf >= 0.9

    def test_sensor_based_classification(self):
        """测试基于传感器特征的分类"""
        clf = SceneClassifier()
        scene, conf, features = clf.classify(
            laser_ranges=np.array([4.0] * 360),  # 高开放空间
            audio_activity=0.05,  # 低音频
            location_hint="",  # 无提示
        )
        # 高开放空间 + 低音频 = 仓库
        assert scene in (SceneType.WAREHOUSE, SceneType.UNKNOWN)

    def test_chinese_location_hint(self):
        """测试中文位置提示"""
        clf = SceneClassifier()
        scene, conf, features = clf.classify(
            location_hint="医院物流",
        )
        assert scene == SceneType.HOSPITAL
        assert conf >= 0.9

    def test_scene_features_building(self):
        """测试场景特征构建"""
        clf = SceneClassifier()
        scene, conf, features = clf.classify(
            location_hint="warehouse",
        )
        assert features.scene_type == SceneType.WAREHOUSE
        assert features.confidence >= 0.9
        assert features.max_speed_safe >= 1.5


# ============================================================
# 场景规则引擎测试
# ============================================================

class TestSceneRuleEngine:
    """场景规则引擎测试"""

    def test_default_rules_initialized(self):
        """测试默认规则初始化"""
        engine = SceneRuleEngine()
        # 验证各场景都有安全规则
        for scene in [SceneType.WAREHOUSE, SceneType.HOSPITAL, SceneType.FACTORY]:
            rules = engine.get_rule_by_type(SafetyRule, scene)
            assert len(rules) >= 1

    def test_warehouse_safety_rules(self):
        """测试仓库安全规则"""
        engine = SceneRuleEngine()
        rules = engine.get_applicable_safety_rules(SceneType.WAREHOUSE)
        assert len(rules) >= 1
        warehouse_safety = next((r for r in rules if r.rule_id == "warehouse_safety"), None)
        assert warehouse_safety is not None
        assert warehouse_safety.max_speed >= 1.5

    def test_hospital_safety_rules(self):
        """测试医院安全规则 - 最高安全等级"""
        engine = SceneRuleEngine()
        rules = engine.get_applicable_safety_rules(SceneType.HOSPITAL)
        hospital_safety = next((r for r in rules if r.rule_id == "hospital_safety"), None)
        assert hospital_safety is not None
        assert hospital_safety.max_speed <= 1.0
        assert hospital_safety.min_clearance >= 0.4
        assert hospital_safety.emergency_exit_required is True

    def test_factory_safety_rules(self):
        """测试工厂安全规则"""
        engine = SceneRuleEngine()
        rules = engine.get_applicable_safety_rules(SceneType.FACTORY)
        factory_safety = next((r for r in rules if r.rule_id == "factory_safety"), None)
        assert factory_safety is not None
        assert factory_safety.fire_safety_required is True

    def test_effective_max_speed(self):
        """测试有效最大速度"""
        engine = SceneRuleEngine()
        assert engine.get_effective_max_speed(SceneType.HOSPITAL) <= 1.0
        assert engine.get_effective_max_speed(SceneType.WAREHOUSE) >= 1.5
        assert engine.get_effective_max_speed(SceneType.OUTDOOR) >= 2.0

    def test_navigation_rules(self):
        """测试导航规则"""
        engine = SceneRuleEngine()
        nav_rules = engine.get_rule_by_type(NavigationRule, SceneType.HOSPITAL)
        assert len(nav_rules) >= 1
        hospital_nav = nav_rules[0]
        assert hospital_nav.path_replan_interval <= 1.0  # 医院需要快速重规划

    def test_interaction_rules(self):
        """测试交互规则"""
        engine = SceneRuleEngine()
        # 餐厅需要情感表达
        restaurant_rules = engine.get_rule_by_type(InteractionRule, SceneType.RESTAURANT)
        assert len(restaurant_rules) >= 1
        assert restaurant_rules[0].express_emotion is True


# ============================================================
# 场景智能主类测试
# ============================================================

class TestSceneIntelligence:
    """场景智能系统测试"""

    def test_scene_context_update(self):
        """测试场景上下文更新"""
        si = SceneIntelligence()
        ctx = si.update(
            laser_ranges=np.array([3.0] * 360),
            audio_activity=0.1,
            location_hint="warehouse",
            nearby_humans=2,
        )
        assert ctx.features.scene_type == SceneType.WAREHOUSE
        assert ctx.features.confidence >= 0.9
        assert ctx.nearby_humans == 2

    def test_adaptive_speed_limit(self):
        """测试自适应速度限制"""
        si = SceneIntelligence()
        # 无场景信息时返回基础速度
        limit = si.get_adaptive_speed_limit(2.0)
        assert limit <= 2.0

        # 设置医院场景
        si.update(location_hint="hospital")
        limit = si.get_adaptive_speed_limit(2.0)
        assert limit <= 1.0  # 医院最高0.8

    def test_safe_distance(self):
        """测试安全距离获取"""
        si = SceneIntelligence()
        si.update(location_hint="hospital")
        dist = si.get_safe_distance()
        assert dist >= 0.4  # 医院最小0.5

        si.update(location_hint="warehouse")
        dist = si.get_safe_distance()
        assert dist >= 0.3  # 仓库最小0.3

    def test_scene_history(self):
        """测试场景历史记录"""
        si = SceneIntelligence(config=SceneConfig(detection_interval=0.05))
        si.update(location_hint="warehouse")
        time.sleep(0.1)
        si.update(location_hint="hospital")
        history = si.get_scene_history(last_n=2)
        assert len(history) == 2
        assert history[0][0] == SceneType.WAREHOUSE
        assert history[1][0] == SceneType.HOSPITAL

    def test_scene_transition_detection(self):
        """测试场景转换检测"""
        si = SceneIntelligence(config=SceneConfig(detection_interval=0.05))
        si.update(location_hint="warehouse")
        time.sleep(0.1)
        si.update(location_hint="hospital")
        assert si.recognize_scene_transition() is True

    def test_multiple_human_speed_reduction(self):
        """测试多人类环境降速"""
        si = SceneIntelligence()
        si.update(location_hint="warehouse", nearby_humans=5)
        limit = si.get_adaptive_speed_limit(2.0)
        assert limit <= 1.0  # 有人时最高1.0

    def test_scene_context_with_agvs(self):
        """测试多AGV场景上下文"""
        si = SceneIntelligence()
        ctx = si.update(
            location_hint="warehouse",
            nearby_agvs=['AGV_01', 'AGV_02', 'AGV_03'],
        )
        assert len(ctx.nearby_agvs) == 3
        assert 'AGV_01' in ctx.nearby_agvs

    def test_scene_rule_update_suppression(self):
        """测试场景更新频率限制"""
        si = SceneIntelligence(config=SceneConfig(detection_interval=10.0))
        si.update(location_hint="warehouse")
        # 快速再次更新应该被抑制
        ctx2 = si.update(location_hint="hospital")
        assert ctx2.features.scene_type == SceneType.WAREHOUSE  # 还是旧的

    def test_global_singleton(self):
        """测试全局单例"""
        si1 = get_scene_intelligence()
        si2 = get_scene_intelligence()
        assert si1 is si2


# ============================================================
# 场景特征测试
# ============================================================

class TestSceneFeatures:
    """场景特征测试"""

    def test_warehouse_features(self):
        """测试仓库场景特征"""
        features = SceneFeatures(
            scene_type=SceneType.WAREHOUSE,
            confidence=0.95,
            floor_type="concrete",
            floor_friction=0.9,
            aisle_width=2.5,
            max_speed_safe=2.0,
        )
        assert features.scene_type == SceneType.WAREHOUSE
        assert features.max_speed_safe >= 1.5
        assert features.aisle_width >= 2.0

    def test_is_safe_for_high_speed(self):
        """测试高速安全判断"""
        features = SceneFeatures(
            scene_type=SceneType.OUTDOOR,
            confidence=0.9,
            obstacle_density=0.05,
            human_density=0.02,
            floor_friction=0.8,
        )
        assert features.is_safe_for_high_speed() is True

        features.obstacle_density = 0.5  # 高密度障碍
        assert features.is_safe_for_high_speed() is False


# ============================================================
# 场景上下文测试
# ============================================================

class TestSceneContext:
    """场景上下文测试"""

    def test_default_context(self):
        """测试默认上下文"""
        ctx = SceneContext()
        assert ctx.features.scene_type == SceneType.UNKNOWN
        assert ctx.nearby_humans == 0
        assert ctx.floor_level == 1

    def test_scene_context_with_features(self):
        """测试带特征的上下文"""
        features = SceneFeatures(scene_type=SceneType.FACTORY, confidence=0.85)
        ctx = SceneContext(
            features=features,
            current_location="zone_A",
            floor_level=2,
            current_task="assembly",
            nearby_humans=3,
        )
        assert ctx.get_scene_type() == SceneType.FACTORY
        assert ctx.current_location == "zone_A"
        assert ctx.current_task == "assembly"


# ============================================================
# 集成测试
# ============================================================

class TestSceneIntelligenceIntegration:
    """场景智能集成测试"""

    def test_scene_to_decision_flow(self):
        """测试场景到决策的流程"""
        si = SceneIntelligence()

        # 1. 识别仓库场景
        ctx = si.update(
            laser_ranges=np.array([3.5] * 360, dtype=float),
            audio_activity=0.1,
            location_hint="warehouse",
            nearby_humans=0,
        )
        assert ctx.features.scene_type == SceneType.WAREHOUSE

        # 2. 获取自适应速度限制
        max_speed = si.get_adaptive_speed_limit(2.0)
        assert max_speed >= 1.5

        # 3. 获取安全距离
        safe_dist = si.get_safe_distance()
        assert safe_dist >= 0.3

        # 4. 获取活跃规则
        rules = si.get_active_rules()
        assert SafetyRule in rules

    def test_hospital_high_safety_flow(self):
        """测试医院高安全流程"""
        si = SceneIntelligence()

        ctx = si.update(
            location_hint="hospital",
            nearby_humans=5,
        )
        assert ctx.features.scene_type == SceneType.HOSPITAL

        # 医院严格速度限制
        max_speed = si.get_adaptive_speed_limit(2.0)
        assert max_speed <= 1.0

        # 医院严格安全距离
        safe_dist = si.get_safe_distance()
        assert safe_dist >= 0.4

    def test_scene_adaptive_behavior(self):
        """测试场景自适应行为"""
        si = SceneIntelligence()

        # 场景切换
        scenes = ["warehouse", "hospital", "factory", "outdoor"]
        speeds = []
        distances = []

        for scene_hint in scenes:
            si.update(location_hint=scene_hint)
            speeds.append(si.get_adaptive_speed_limit(3.0))
            distances.append(si.get_safe_distance())

        # 医院最慢最安全
        assert speeds[1] <= speeds[0]  # 医院 <= 仓库
        assert distances[1] >= distances[0]  # 医院安全距离 >= 仓库

        # 户外最快
        assert speeds[3] >= speeds[2]  # 户外 >= 工厂


# ============================================================
# AGV五级规格适配测试
# ============================================================

class TestSceneIntelligenceGradeAdaptation:
    """场景智能AGV等级适配测试"""

    def test_config_grade_adaptation(self):
        """测试配置中的等级适应性"""
        config_s = SceneConfig(grade="S")
        config_xxl = SceneConfig(grade="XXL")

        assert config_s.warehouse_max_speed <= config_xxl.warehouse_max_speed

    def test_scene_intelligence_with_config(self):
        """测试带配置的实例化"""
        config = SceneConfig(
            grade="XL",
            detection_interval=1.0,
            min_confidence_threshold=0.8,
            enable_memory=False,
        )
        si = SceneIntelligence(config=config)
        assert si.config.grade == "XL"
        assert si.config.detection_interval == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
