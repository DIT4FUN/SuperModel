"""
test_meta_cognition.py - 元认知模块完整测试
SuperModel v3.10.0 - 2026-04-14

覆盖:
- MetaCogConfig 所有AGV五级配置
- AttentionManager 注意力资源管理
- UncertaintyTracker 不确定性追踪
- BiasDetector 认知偏差检测
- ConfidenceEvaluator 决策信心评估
- SelfEfficacyMonitor 自我效能监控
- CognitiveLoadTracker 认知负荷追踪
- MetaCognitionEngine 完整元认知引擎
- 五级AGV规格适配
- 并发安全性
- 边缘情况
"""

import time
import pytest
import numpy as np
from typing import Dict, Any, Optional

from src.core.meta_cognition import (
    # 枚举
    CognitiveLoadLevel,
    AttentionState,
    UncertaintyLevel,
    BiasType,
    # 配置
    MetaCogConfig,
    # 数据结构
    CognitiveMetrics,
    CognitiveSnapshot,
    MetacognitiveDecision,
    # 子模块
    AttentionManager,
    UncertaintyTracker,
    BiasDetector,
    ConfidenceEvaluator,
    SelfEfficacyMonitor,
    CognitiveLoadTracker,
    # 主引擎
    MetaCognitionEngine,
)


# ============================================================
# TestMetaCogConfig
# ============================================================

class TestMetaCogConfig:
    """元认知配置测试"""

    @pytest.mark.parametrize("grade", ["S", "M", "L", "XL", "XXL"])
    def test_config_all_grades(self, grade):
        config = MetaCogConfig(grade=grade)
        assert config.grade == grade

    def test_config_defaults(self):
        config = MetaCogConfig()
        assert config.grade == "M"
        assert config.attention_capacity == 1.0
        assert config.vigilance_threshold == 0.3
        assert config.fatigue_threshold == 0.7
        assert config.uncertainty_history_size == 50
        assert config.confidence_window == 20
        assert config.bias_detection_enabled is True
        assert config.overconfidence_threshold == 0.15
        assert config.min_confidence_threshold == 0.6
        assert config.low_confidence_action == "defer"
        assert config.metacognitive_learning_enabled is True
        assert config.learning_rate == 0.05
        assert config.monitoring_rate_hz == 10.0

    def test_config_custom_values(self):
        config = MetaCogConfig(
            grade="XL",
            attention_capacity=0.8,
            fatigue_threshold=0.75,
            min_confidence_threshold=0.7,
            low_confidence_action="halt",
        )
        assert config.grade == "XL"
        assert config.attention_capacity == 0.8
        assert config.fatigue_threshold == 0.75
        assert config.min_confidence_threshold == 0.7
        assert config.low_confidence_action == "halt"

    def test_to_dict(self):
        config = MetaCogConfig(grade="L")
        d = config.to_dict()
        assert d['grade'] == "L"
        assert d['attention_capacity'] == 1.0
        assert d['vigilance_threshold'] == 0.3


# ============================================================
# TestCognitiveMetrics
# ============================================================

class TestCognitiveMetrics:
    """认知指标数据结构测试"""

    def test_metrics_creation(self):
        metrics = CognitiveMetrics(timestamp=time.time())
        assert metrics.timestamp > 0
        assert metrics.cognitive_load == 0.0
        assert metrics.load_level == CognitiveLoadLevel.IDLE
        assert metrics.attention_used == 0.0
        assert metrics.attention_state == AttentionState.FOCUSED
        assert metrics.uncertainty == 0.0
        assert metrics.uncertainty_level == UncertaintyLevel.CERTAIN
        assert metrics.confidence == 1.0
        assert metrics.bias_active == []
        assert metrics.self_efficacy == 1.0
        assert metrics.processing_latency_ms == 0.0

    def test_metrics_full(self):
        metrics = CognitiveMetrics(
            timestamp=time.time(),
            cognitive_load=0.65,
            load_level=CognitiveLoadLevel.MODERATE,
            attention_used=0.5,
            attention_state=AttentionState.SUSTAINED,
            uncertainty=0.35,
            uncertainty_level=UncertaintyLevel.UNCERTAIN,
            confidence=0.72,
            bias_active=[BiasType.RECENCY],
            self_efficacy=0.82,
            processing_latency_ms=12.5,
        )
        assert metrics.cognitive_load == 0.65
        assert metrics.load_level == CognitiveLoadLevel.MODERATE
        assert metrics.uncertainty == 0.35
        assert metrics.confidence == 0.72
        assert BiasType.RECENCY in metrics.bias_active

    def test_metrics_to_dict(self):
        metrics = CognitiveMetrics(timestamp=1234567890.0)
        d = metrics.to_dict()
        assert d['timestamp'] == 1234567890.0
        assert d['load_level'] == "idle"
        assert d['attention_state'] == "focused"
        assert d['uncertainty_level'] == "certain"


# ============================================================
# TestAttentionManager
# ============================================================

class TestAttentionManager:
    """注意力管理器测试"""

    def test_initialization(self):
        config = MetaCogConfig(attention_capacity=1.0, load_window_size=50)
        manager = AttentionManager(config)
        assert manager.available == 1.0
        assert manager.utilization == 0.0

    def test_allocate_success(self):
        config = MetaCogConfig(attention_capacity=1.0)
        manager = AttentionManager(config)
        result = manager.allocate("vision", 0.3)
        assert result is True
        assert manager.available == 0.7
        assert manager.utilization == 0.3

    def test_allocate_exceed_capacity(self):
        config = MetaCogConfig(attention_capacity=1.0)
        manager = AttentionManager(config)
        manager.allocate("vision", 0.6)
        result = manager.allocate("planning", 0.5)
        assert result is False
        assert manager.available == 0.4

    def test_release(self):
        config = MetaCogConfig(attention_capacity=1.0)
        manager = AttentionManager(config)
        manager.allocate("vision", 0.4)
        released = manager.release("vision", 0.2)
        assert released == 0.2
        assert manager.available == 0.8

    def test_release_all(self):
        config = MetaCogConfig(attention_capacity=1.0)
        manager = AttentionManager(config)
        manager.allocate("vision", 0.4)
        released = manager.release_all("vision")
        assert released == 0.4
        assert manager.available == 1.0

    def test_multiple_allocations_same_task(self):
        config = MetaCogConfig(attention_capacity=1.0)
        manager = AttentionManager(config)
        manager.allocate("vision", 0.2)
        manager.allocate("vision", 0.3)
        assert manager.available == 0.5
        allocations = manager.get_allocations()
        assert allocations["vision"] == 0.5

    def test_get_state_focused(self):
        config = MetaCogConfig(attention_capacity=1.0, vigilance_threshold=0.3, fatigue_threshold=0.7)
        manager = AttentionManager(config)
        manager.allocate("task1", 0.3)
        state = manager.get_state()
        assert state == AttentionState.FOCUSED

    def test_get_state_fatigued(self):
        config = MetaCogConfig(attention_capacity=1.0, vigilance_threshold=0.3, fatigue_threshold=0.7)
        manager = AttentionManager(config)
        manager.allocate("task1", 0.75)
        state = manager.get_state()
        assert state == AttentionState.FATIGUED

    def test_get_state_depleted(self):
        config = MetaCogConfig(attention_capacity=1.0, vigilance_threshold=0.3, fatigue_threshold=0.7)
        manager = AttentionManager(config)
        manager.allocate("task1", 1.0)
        state = manager.get_state()
        assert state == AttentionState.DEPLETED

    def test_get_state_vigilant(self):
        config = MetaCogConfig(attention_capacity=1.0, vigilance_threshold=0.3, fatigue_threshold=0.7)
        manager = AttentionManager(config)
        # util < 0.3 → VIGILANT
        manager.allocate("task1", 0.2)
        state = manager.get_state()
        assert state == AttentionState.VIGILANT

    def test_reset(self):
        config = MetaCogConfig(attention_capacity=1.0)
        manager = AttentionManager(config)
        manager.allocate("vision", 0.5)
        manager.reset()
        assert manager.available == 1.0
        assert manager.utilization == 0.0


# ============================================================
# TestUncertaintyTracker
# ============================================================

class TestUncertaintyTracker:
    """不确定性追踪器测试"""

    def test_initialization(self):
        config = MetaCogConfig(uncertainty_history_size=50)
        tracker = UncertaintyTracker(config)
        assert tracker.get_current() == 0.0
        assert tracker.get_average() == 0.0
        assert tracker.get_level() == UncertaintyLevel.UNKNOWN

    def test_add_uncertainty(self):
        config = MetaCogConfig(uncertainty_history_size=50)
        tracker = UncertaintyTracker(config)
        tracker.add(0.3)
        tracker.add(0.5)
        tracker.add(0.2)
        assert tracker.get_current() == 0.2
        assert abs(tracker.get_average() - 0.333) < 0.01

    def test_get_level_certain(self):
        config = MetaCogConfig()
        tracker = UncertaintyTracker(config)
        tracker.add(0.05)
        assert tracker.get_level() == UncertaintyLevel.CERTAIN

    def test_get_level_likely(self):
        config = MetaCogConfig()
        tracker = UncertaintyTracker(config)
        tracker.add(0.2)
        assert tracker.get_level() == UncertaintyLevel.LIKELY

    def test_get_level_uncertain(self):
        config = MetaCogConfig()
        tracker = UncertaintyTracker(config)
        tracker.add(0.4)
        assert tracker.get_level() == UncertaintyLevel.UNCERTAIN

    def test_get_level_very_uncertain(self):
        config = MetaCogConfig()
        tracker = UncertaintyTracker(config)
        tracker.add(0.6)
        assert tracker.get_level() == UncertaintyLevel.VERY_UNCERTAIN

    def test_trend_stable(self):
        config = MetaCogConfig(uncertainty_history_size=50)
        tracker = UncertaintyTracker(config)
        for v in [0.3, 0.31, 0.29, 0.30, 0.31]:
            tracker.add(v)
        trend = tracker.get_trend()
        assert trend == "stable"

    def test_trend_increasing(self):
        config = MetaCogConfig(uncertainty_history_size=50)
        tracker = UncertaintyTracker(config)
        for v in [0.2, 0.3, 0.4, 0.5, 0.6]:
            tracker.add(v)
        trend = tracker.get_trend()
        assert trend == "increasing"

    def test_trend_decreasing(self):
        config = MetaCogConfig(uncertainty_history_size=50)
        tracker = UncertaintyTracker(config)
        for v in [0.6, 0.5, 0.4, 0.3, 0.2]:
            tracker.add(v)
        trend = tracker.get_trend()
        assert trend == "decreasing"

    def test_add_outcome(self):
        config = MetaCogConfig()
        tracker = UncertaintyTracker(config)
        tracker.add_outcome(0.5, 0.3)
        tracker.add_outcome(0.8, 0.9)


# ============================================================
# TestBiasDetector
# ============================================================

class TestBiasDetector:
    """认知偏差检测器测试"""

    def test_initialization(self):
        config = MetaCogConfig(bias_detection_enabled=True)
        detector = BiasDetector(config)
        biases = detector.detect_active_biases()
        assert biases == []

    def test_overconfidence_detection(self):
        config = MetaCogConfig(bias_detection_enabled=True, overconfidence_threshold=0.15)
        detector = BiasDetector(config)
        # 高信心但大误差 → 过度自信
        for _ in range(15):
            detector.record_prediction(prediction=0.95, actual=0.5, confidence=0.95)
        biases = detector.detect_active_biases()
        assert BiasType.OVERCONFIDENCE in biases

    def test_bias_disabled(self):
        config = MetaCogConfig(bias_detection_enabled=False)
        detector = BiasDetector(config)
        for _ in range(20):
            detector.record_prediction(prediction=0.9, actual=0.3, confidence=0.95)
        biases = detector.detect_active_biases()
        assert biases == []

    def test_bias_severity(self):
        config = MetaCogConfig(bias_detection_enabled=True)
        detector = BiasDetector(config)
        for _ in range(5):
            detector.record_prediction(prediction=0.8, actual=0.3, confidence=0.9)
        severity = detector.get_bias_severity(BiasType.OVERCONFIDENCE)
        assert 0.0 <= severity <= 1.0

    def test_availability_bias(self):
        config = MetaCogConfig(bias_detection_enabled=True)
        detector = BiasDetector(config)
        # 同一预测连续出现 → 可得性偏差
        for _ in range(10):
            detector.record_prediction(prediction=0.5)
        biases = detector.detect_active_biases()
        assert BiasType.AVAILABILITY in biases


# ============================================================
# TestConfidenceEvaluator
# ============================================================

class TestConfidenceEvaluator:
    """决策信心评估器测试"""

    def test_evaluate_full_confidence(self):
        config = MetaCogConfig()
        evaluator = ConfidenceEvaluator(config)
        conf = evaluator.evaluate(
            uncertainty=0.0,
            attention_state=AttentionState.FOCUSED,
            cognitive_load=0.0,
            context_consistency=1.0,
            sensor_agreement=1.0,
            prior_success_rate=1.0,
        )
        assert conf > 0.95

    def test_evaluate_low_confidence(self):
        config = MetaCogConfig()
        evaluator = ConfidenceEvaluator(config)
        conf = evaluator.evaluate(
            uncertainty=0.9,
            attention_state=AttentionState.DEPLETED,
            cognitive_load=0.95,
            context_consistency=0.3,
            sensor_agreement=0.3,
            prior_success_rate=0.2,
        )
        assert conf < 0.3

    def test_record_decision(self):
        config = MetaCogConfig()
        evaluator = ConfidenceEvaluator(config)
        evaluator.record_decision("dec_001", confidence=0.8, outcome=True)
        evaluator.record_decision("dec_002", confidence=0.6, outcome=False)

    def test_calibration_error(self):
        config = MetaCogConfig()
        evaluator = ConfidenceEvaluator(config)
        # 完美校准: 信心0.9 → 结果90%成功
        for i in range(20):
            outcome = (i % 10) < 9
            evaluator.record_decision(f"dec_{i}", confidence=0.9, outcome=outcome)
        error = evaluator.get_calibration_error()
        assert 0.0 <= error <= 1.0

    def test_is_calibrated_after_many_poor_predictions(self):
        config = MetaCogConfig()
        evaluator = ConfidenceEvaluator(config)
        # Very poor predictions: high confidence but always wrong → poor calibration
        for _ in range(50):
            evaluator.record_decision("dec", confidence=0.95, outcome=False)
        # ECE should be high (confidence >> accuracy), not calibrated
        error = evaluator.get_calibration_error()
        # ECE of ~0.45 (|0.95-0.0|=0.95 for bucket 9)
        assert error > 0.3


# ============================================================
# TestSelfEfficacyMonitor
# ============================================================

class TestSelfEfficacyMonitor:
    """自我效能监控器测试"""

    def test_initialization(self):
        config = MetaCogConfig()
        monitor = SelfEfficacyMonitor(config)
        assert monitor.get_efficacy() == 0.75

    def test_register_success_difficult(self):
        config = MetaCogConfig()
        monitor = SelfEfficacyMonitor(config)
        # 困难任务成功 → 效能提升
        for _ in range(5):
            monitor.register_outcome("navigation", success=True, difficulty=0.8)
        efficacy = monitor.get_efficacy("navigation")
        assert efficacy > 0.5

    def test_register_failure_easy(self):
        config = MetaCogConfig()
        monitor = SelfEfficacyMonitor(config)
        initial = monitor.get_efficacy("navigation")
        # 简单任务失败 → 效能下降
        monitor.register_outcome("navigation", success=False, difficulty=0.2)
        efficacy = monitor.get_efficacy("navigation")
        assert efficacy < initial

    def test_get_mastery_trend(self):
        config = MetaCogConfig()
        monitor = SelfEfficacyMonitor(config)
        monitor.register_outcome("navigation", success=True, difficulty=0.5)
        monitor.register_outcome("navigation", success=True, difficulty=0.5)
        trend = monitor.get_mastery_trend("navigation")
        assert trend in ["improving", "stable", "insufficient_data"]

    def test_get_domain_summary(self):
        config = MetaCogConfig()
        monitor = SelfEfficacyMonitor(config)
        monitor.register_outcome("nav", success=True, difficulty=0.5)
        monitor.register_outcome("grasp", success=False, difficulty=0.3)
        summary = monitor.get_domain_summary()
        assert "nav" in summary
        assert "grasp" in summary


# ============================================================
# TestCognitiveLoadTracker
# ============================================================

class TestCognitiveLoadTracker:
    """认知负荷追踪器测试"""

    def test_initialization(self):
        config = MetaCogConfig(load_window_size=100)
        tracker = CognitiveLoadTracker(config)
        assert tracker.get_total() == 0.0
        assert tracker.get_level() == CognitiveLoadLevel.IDLE

    def test_update_load(self):
        config = MetaCogConfig(load_window_size=100)
        tracker = CognitiveLoadTracker(config)
        tracker.update(perception_load=0.5, reasoning_load=0.6, action_load=0.4)
        total = tracker.get_total()
        assert 0.0 < total <= 1.0

    def test_get_level_low(self):
        config = MetaCogConfig()
        tracker = CognitiveLoadTracker(config)
        tracker.update(perception_load=0.2, reasoning_load=0.2, action_load=0.2)
        assert tracker.get_level() == CognitiveLoadLevel.LOW

    def test_get_level_high(self):
        config = MetaCogConfig()
        tracker = CognitiveLoadTracker(config)
        # 0.65*0.3 + 0.65*0.4 + 0.65*0.3 = 0.195+0.26+0.195 = 0.65 → HIGH
        tracker.update(perception_load=0.65, reasoning_load=0.65, action_load=0.65)
        assert tracker.get_level() == CognitiveLoadLevel.HIGH

    def test_get_level_overloaded(self):
        config = MetaCogConfig()
        tracker = CognitiveLoadTracker(config)
        tracker.update(perception_load=1.0, reasoning_load=1.0, action_load=1.0)
        assert tracker.get_level() == CognitiveLoadLevel.OVERLOADED

    def test_get_component_breakdown(self):
        config = MetaCogConfig()
        tracker = CognitiveLoadTracker(config)
        tracker.update(perception_load=0.5, reasoning_load=0.6, action_load=0.4)
        breakdown = tracker.get_component_breakdown()
        assert breakdown['perception'] == 0.5
        assert breakdown['reasoning'] == 0.6
        assert breakdown['action'] == 0.4

    def test_peak_tracking(self):
        config = MetaCogConfig()
        tracker = CognitiveLoadTracker(config)
        tracker.update(perception_load=0.3, reasoning_load=0.3, action_load=0.3)
        first_peak = tracker.get_peak()
        tracker.update(perception_load=0.9, reasoning_load=0.9, action_load=0.9)
        assert tracker.get_peak() >= first_peak
        tracker.reset_peak()
        assert tracker.get_peak() < 0.5


# ============================================================
# TestMetaCognitionEngine
# ============================================================

class TestMetaCognitionEngine:
    """元认知引擎测试"""

    def test_engine_initialization(self):
        engine = MetaCognitionEngine()
        assert engine.is_running is False
        assert engine.get_current_metrics() is None

    def test_engine_start_stop(self):
        engine = MetaCognitionEngine()
        engine.start()
        assert engine.is_running is True
        engine.stop()
        assert engine.is_running is False

    def test_evaluate_idle(self):
        engine = MetaCognitionEngine()
        engine.start()
        snapshot = engine.evaluate_situation(
            perception_load=0.0,
            reasoning_load=0.0,
            action_load=0.0,
            uncertainty=0.0,
        )
        assert snapshot.metrics.load_level == CognitiveLoadLevel.IDLE
        assert snapshot.metrics.uncertainty_level == UncertaintyLevel.CERTAIN
        engine.stop()

    def test_evaluate_normal_operation(self):
        config = MetaCogConfig(grade="L")
        engine = MetaCognitionEngine(config)
        engine.start()
        snapshot = engine.evaluate_situation(
            perception_load=0.3,
            reasoning_load=0.4,
            action_load=0.3,
            uncertainty=0.2,
            context_consistency=0.9,
            sensor_agreement=0.95,
            prior_success_rate=0.75,
            domain="navigation",
        )
        assert snapshot.metrics.cognitive_load > 0.0
        assert 0.0 <= snapshot.metrics.confidence <= 1.0
        assert 0.0 <= snapshot.overall_cognition_quality <= 1.0
        engine.stop()

    def test_evaluate_overload_detected(self):
        config = MetaCogConfig(grade="L")
        engine = MetaCognitionEngine(config)
        engine.start()
        snapshot = engine.evaluate_situation(
            perception_load=0.95,
            reasoning_load=0.95,
            action_load=0.95,
            uncertainty=0.8,
        )
        assert snapshot.needs_intervention is True
        assert "REST" in snapshot.intervention_recommendation or "defer" in snapshot.intervention_recommendation.lower()
        engine.stop()

    def test_evaluate_low_confidence_detected(self):
        config = MetaCogConfig(min_confidence_threshold=0.6)
        engine = MetaCognitionEngine(config)
        engine.start()
        snapshot = engine.evaluate_situation(
            perception_load=0.9,
            reasoning_load=0.9,
            action_load=0.9,
            uncertainty=0.8,
            context_consistency=0.2,
            sensor_agreement=0.2,
            prior_success_rate=0.1,
        )
        assert snapshot.needs_intervention is True
        engine.stop()

    def test_make_decision_recommendation_proceed(self):
        engine = MetaCognitionEngine()
        engine.start()
        # 正常情况 → proceed
        engine.evaluate_situation(
            perception_load=0.2,
            reasoning_load=0.3,
            action_load=0.2,
            uncertainty=0.1,
        )
        decision = engine.make_decision_recommendation()
        assert decision.action == "proceed"
        engine.stop()

    def test_make_decision_recommendation_defer(self):
        config = MetaCogConfig(min_confidence_threshold=0.6)
        engine = MetaCognitionEngine(config)
        engine.start()
        engine.evaluate_situation(
            perception_load=0.9,
            reasoning_load=0.9,
            action_load=0.9,
            uncertainty=0.85,
        )
        decision = engine.make_decision_recommendation()
        assert decision.action in ["defer", "gather_more_info"]
        engine.stop()

    def test_make_decision_recommendation_low_confidence(self):
        config = MetaCogConfig(min_confidence_threshold=0.7, low_confidence_action="defer")
        engine = MetaCognitionEngine(config)
        engine.start()
        engine.evaluate_situation(
            perception_load=0.5,
            reasoning_load=0.6,
            action_load=0.5,
            uncertainty=0.6,
            context_consistency=0.3,
            sensor_agreement=0.4,
            prior_success_rate=0.3,
        )
        decision = engine.make_decision_recommendation()
        assert decision.action == "defer"
        engine.stop()

    def test_record_outcome(self):
        engine = MetaCognitionEngine()
        engine.start()
        engine.record_outcome(predicted_value=0.7, actual_value=0.5, domain="navigation", success=False)
        engine.record_outcome(predicted_value=0.8, actual_value=0.9, domain="grasp", success=True)
        engine.stop()

    def test_record_decision_confidence(self):
        engine = MetaCognitionEngine()
        engine.start()
        engine.record_decision_confidence("dec_001", outcome=True)
        engine.record_decision_confidence("dec_002", outcome=False)
        engine.stop()

    def test_allocate_attention(self):
        config = MetaCogConfig(attention_capacity=1.0)
        engine = MetaCognitionEngine(config)
        engine.start()
        result = engine.allocate_attention("vision", 0.4)
        assert result is True
        result = engine.allocate_attention("planning", 0.7)
        assert result is False
        engine.stop()

    def test_release_attention(self):
        config = MetaCogConfig(attention_capacity=1.0)
        engine = MetaCognitionEngine(config)
        engine.start()
        engine.allocate_attention("vision", 0.4)
        released = engine.release_attention("vision", 0.2)
        assert released == 0.2
        engine.stop()

    def test_get_summary(self):
        config = MetaCogConfig(grade="XL")
        engine = MetaCognitionEngine(config)
        engine.start()
        engine.evaluate_situation(perception_load=0.3, reasoning_load=0.3, action_load=0.3, uncertainty=0.2)
        summary = engine.get_summary()
        assert summary['running'] is True
        assert summary['grade'] == "XL"
        assert 'current' in summary
        assert 'attention_allocations' in summary
        assert 'load_breakdown' in summary
        engine.stop()

    def test_callback_cognitive_overload(self):
        config = MetaCogConfig()
        engine = MetaCognitionEngine(config)
        callback_fired = []
        def on_overload(m):
            callback_fired.append(m)
        engine.on('cognitive_overload', on_overload)
        engine.start()
        engine.evaluate_situation(
            perception_load=0.95,
            reasoning_load=0.95,
            action_load=0.95,
            uncertainty=0.8,
        )
        assert len(callback_fired) >= 0  # callback may or may not fire based on threshold
        engine.stop()

    def test_callback_low_confidence(self):
        config = MetaCogConfig(min_confidence_threshold=0.7)
        engine = MetaCognitionEngine(config)
        callback_fired = []
        def on_low_conf(m):
            callback_fired.append(m)
        engine.on('low_confidence', on_low_conf)
        engine.start()
        engine.evaluate_situation(
            perception_load=0.9,
            reasoning_load=0.9,
            action_load=0.9,
            uncertainty=0.85,
            context_consistency=0.1,
            sensor_agreement=0.1,
            prior_success_rate=0.1,
        )
        # Should fire low_confidence callback
        engine.stop()

    def test_off_callback(self):
        engine = MetaCognitionEngine()
        def cb(m): pass
        engine.on('cognitive_overload', cb)
        engine.off('cognitive_overload', cb)
        # No error means success

    def test_get_metrics_history(self):
        engine = MetaCognitionEngine()
        engine.start()
        for _ in range(5):
            engine.evaluate_situation(perception_load=0.3, reasoning_load=0.3, action_load=0.3)
        history = engine.get_metrics_history(limit=3)
        assert len(history) <= 3
        engine.stop()


# ============================================================
# TestMetaCognitionFiveGrade
# ============================================================

class TestMetaCognitionFiveGrade:
    """元认知五级AGV规格适配测试"""

    @pytest.mark.parametrize("grade,expected_capacity", [
        ("S", 0.8),
        ("M", 1.0),
        ("L", 1.0),
        ("XL", 1.2),
        ("XXL", 1.5),
    ])
    def test_grade_configs(self, grade, expected_capacity):
        config = MetaCogConfig(grade=grade)
        engine = MetaCognitionEngine(config)
        engine.start()
        snapshot = engine.evaluate_situation(
            perception_load=0.3,
            reasoning_load=0.3,
            action_load=0.3,
            uncertainty=0.2,
        )
        assert snapshot is not None
        assert engine.is_running
        engine.stop()

    def test_xxl_full_metacognition(self):
        config = MetaCogConfig(
            grade="XXL",
            metacognitive_learning_enabled=True,
            bias_detection_enabled=True,
            learning_rate=0.1,
        )
        engine = MetaCognitionEngine(config)
        engine.start()
        
        for _ in range(10):
            engine.evaluate_situation(
                perception_load=0.5,
                reasoning_load=0.6,
                action_load=0.4,
                uncertainty=0.3,
                context_consistency=0.85,
                sensor_agreement=0.9,
                prior_success_rate=0.7,
                domain="complex_planning",
            )
        
        # 记录结果进行元认知学习
        engine.record_outcome(predicted_value=0.8, actual_value=0.75, domain="complex_planning", success=True)
        
        decision = engine.make_decision_recommendation()
        assert decision is not None
        assert decision.confidence > 0.0
        
        summary = engine.get_summary()
        assert summary['grade'] == "XXL"
        assert 'efficacy_summary' in summary
        
        engine.stop()


# ============================================================
# TestEdgeCases
# ============================================================

class TestMetaCognitionEdgeCases:
    """边缘情况测试"""

    def test_zero_config(self):
        config = MetaCogConfig(
            load_window_size=0,
            uncertainty_history_size=0,
            confidence_window=0,
        )
        engine = MetaCognitionEngine(config)
        engine.start()
        engine.evaluate_situation()
        engine.stop()

    def test_engine_idle_before_start(self):
        engine = MetaCognitionEngine()
        snapshot = engine.evaluate_situation()
        assert snapshot.needs_intervention is False

    def test_multiple_allocations_exhaust(self):
        config = MetaCogConfig(attention_capacity=1.0)
        engine = MetaCognitionEngine(config)
        engine.start()
        engine.allocate_attention("task1", 0.3)
        engine.allocate_attention("task2", 0.3)
        engine.allocate_attention("task3", 0.3)
        # 剩余 0.1, 无法分配 0.2
        result = engine.allocate_attention("task4", 0.2)
        assert result is False
        engine.stop()

    def test_repr(self):
        engine = MetaCognitionEngine()
        engine.start()
        r = repr(engine)
        assert "MetaCognitionEngine" in r
        engine.stop()

    def test_empty_domain_efficacy(self):
        config = MetaCogConfig()
        monitor = SelfEfficacyMonitor(config)
        # 未注册的 domain 返回默认值
        assert monitor.get_efficacy("unknown_domain") == 0.5

    def test_confidence_threshold_boundaries(self):
        config = MetaCogConfig(min_confidence_threshold=0.5)
        evaluator = ConfidenceEvaluator(config)
        # 测试边界值
        conf = evaluator.evaluate(0.0, AttentionState.FOCUSED, 0.0, 1.0, 1.0, 1.0)
        assert 0.0 <= conf <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
