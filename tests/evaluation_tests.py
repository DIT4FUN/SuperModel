"""
评估模块测试
============

测试基准测试套件和评估指标
"""

import pytest
import torch
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.benchmark import (
    BenchmarkSuite, BenchmarkConfig, BenchmarkResult,
    SensorBenchmark, FusionBenchmark, ControlBenchmark, EmbodiedBenchmark,
    AGVGrade, AGV_LATENCY_SPEC, AGV_MEMORY_SPEC
)
from src.evaluation.metrics import (
    LatencyMetrics, MultimodalMetrics, ControlMetrics,
    compute_multimodal_f1, compute_control_accuracy, LatencyTracker
)
from src.evaluation.reporter import EvaluationReporter


class TestBenchmarkConfig:
    """测试基准配置"""
    
    def test_default_config(self):
        cfg = BenchmarkConfig()
        assert cfg.grade == AGVGrade.M
        assert cfg.num_iterations == 100
        assert cfg.warmup_iterations == 10
        assert cfg.enable_profiling is False
    
    def test_custom_config(self):
        cfg = BenchmarkConfig(grade=AGVGrade.L, num_iterations=50, warmup_iterations=5)
        assert cfg.grade == AGVGrade.L
        assert cfg.num_iterations == 50
        assert cfg.warmup_iterations == 5
    
    def test_agv_grade_enum(self):
        assert AGVGrade.S.value == "S"
        assert AGVGrade.M.value == "M"
        assert AGVGrade.L.value == "L"
        assert AGVGrade.XL.value == "XL"
        assert AGVGrade.XXL.value == "XXL"


class TestAGVSpecTables:
    """测试 AGV 五级规格表"""
    
    def test_latency_spec_keys(self):
        assert set(AGV_LATENCY_SPEC.keys()) == {"S", "M", "L", "XL", "XXL"}
        for grade, spec in AGV_LATENCY_SPEC.items():
            assert "sensor" in spec
            assert "fusion" in spec
            assert "control" in spec
            assert "total" in spec
            assert spec["sensor"] >= spec["fusion"] >= spec["control"]
    
    def test_latency_scaling(self):
        """验证延迟随等级提升而降低"""
        for key in ["sensor", "fusion", "control", "total"]:
            s_latency = AGV_LATENCY_SPEC["S"][key]
            m_latency = AGV_LATENCY_SPEC["M"][key]
            l_latency = AGV_LATENCY_SPEC["L"][key]
            xl_latency = AGV_LATENCY_SPEC["XL"][key]
            xxl_latency = AGV_LATENCY_SPEC["XXL"][key]
            assert s_latency > m_latency > l_latency > xl_latency > xxl_latency
    
    def test_memory_spec_keys(self):
        assert set(AGV_MEMORY_SPEC.keys()) == {"S", "M", "L", "XL", "XXL"}
        for grade, spec in AGV_MEMORY_SPEC.items():
            assert "ram" in spec
            assert "flash" in spec
            assert spec["ram"] < spec["flash"]
    
    def test_memory_scaling(self):
        """验证内存随等级提升而增加"""
        grades = ["S", "M", "L", "XL", "XXL"]
        rams = [AGV_MEMORY_SPEC[g]["ram"] for g in grades]
        assert rams == sorted(rams)


class TestBenchmarkResult:
    """测试基准结果"""
    
    def test_result_creation(self):
        result = BenchmarkResult(
            name="Test",
            grade="M",
            passed=True,
            latency_ms=10.0,
            latency_spec_ms=20.0,
            memory_mb=50.0,
            memory_spec_mb=200.0,
            throughput_fps=100.0,
            accuracy=0.95,
        )
        assert result.name == "Test"
        assert result.grade == "M"
        assert result.passed is True
        assert result.latency_ms == 10.0
    
    def test_result_summary(self):
        result = BenchmarkResult(
            name="Vision",
            grade="M",
            passed=True,
            latency_ms=8.0,
            latency_spec_ms=10.0,
            memory_mb=30.0,
            memory_spec_mb=100.0,
            throughput_fps=120.0,
            accuracy=0.98,
        )
        summary = result.summary()
        assert "PASS" in summary
        assert "Vision" in summary
        assert "8.00" in summary
    
    def test_result_to_dict(self):
        result = BenchmarkResult(
            name="Test",
            grade="S",
            passed=False,
            latency_ms=200.0,
            latency_spec_ms=100.0,
            memory_mb=500.0,
            memory_spec_mb=300.0,
            throughput_fps=5.0,
            accuracy=0.7,
        )
        d = result.to_dict()
        assert d["name"] == "Test"
        assert d["grade"] == "S"
        assert d["passed"] is False


class TestSensorBenchmark:
    """测试传感器基准"""
    
    @pytest.fixture
    def config(self):
        return BenchmarkConfig(grade=AGVGrade.S, num_iterations=20, warmup_iterations=5)
    
    @pytest.fixture
    def bench(self, config):
        return SensorBenchmark(config)
    
    def test_benchmark_vision(self, bench):
        result = bench.benchmark_vision()
        assert result.name == "Vision"
        assert result.grade == "S"
        assert result.latency_ms > 0
        assert result.throughput_fps > 0
    
    def test_benchmark_tactile(self, bench):
        result = bench.benchmark_tactile()
        assert result.name == "Tactile"
        assert result.grade == "S"
        assert result.latency_ms > 0
    
    def test_benchmark_force(self, bench):
        result = bench.benchmark_force()
        assert result.name == "Force"
        assert result.grade == "S"
        assert result.latency_ms > 0
    
    def test_benchmark_imu(self, bench):
        result = bench.benchmark_imu()
        assert result.name == "IMU"
        assert result.grade == "S"
        assert result.latency_ms > 0
    
    def test_run_all(self, bench):
        results = bench.run_all()
        assert len(results) == 4
        names = {r.name for r in results}
        assert names == {"Vision", "Tactile", "Force", "IMU"}


class TestFusionBenchmark:
    """测试融合基准"""
    
    @pytest.fixture
    def config(self):
        return BenchmarkConfig(grade=AGVGrade.S, num_iterations=20, warmup_iterations=5)
    
    @pytest.fixture
    def bench(self, config):
        return FusionBenchmark(config)
    
    def test_benchmark_fusion(self, bench):
        result = bench.benchmark_fusion()
        assert result.name == "CrossModalFusion"
        assert result.grade == "S"
        assert result.latency_ms > 0
    
    def test_run_all(self, bench):
        results = bench.run_all()
        assert len(results) >= 1


class TestControlBenchmark:
    """测试控制基准"""
    
    @pytest.fixture
    def config(self):
        return BenchmarkConfig(grade=AGVGrade.M, num_iterations=50, warmup_iterations=10)
    
    @pytest.fixture
    def bench(self, config):
        return ControlBenchmark(config)
    
    def test_benchmark_control_loop(self, bench):
        result = bench.benchmark_control_loop()
        assert result.name == "ControlLoop"
        assert result.grade == "M"
        assert result.latency_ms >= 0
        assert result.throughput_fps > 0


class TestEmbodiedBenchmark:
    """测试具身智能基准"""
    
    @pytest.fixture
    def config(self):
        return BenchmarkConfig(grade=AGVGrade.S, num_iterations=10, warmup_iterations=3)
    
    @pytest.fixture
    def bench(self, config):
        return EmbodiedBenchmark(config)
    
    def test_benchmark_end_to_end(self, bench):
        result = bench.benchmark_end_to_end()
        assert result.name == "EndToEnd"
        assert result.grade == "S"
        assert result.latency_ms >= 0


class TestBenchmarkSuite:
    """测试基准测试套件"""
    
    @pytest.fixture
    def config(self):
        return BenchmarkConfig(grade=AGVGrade.S, num_iterations=10, warmup_iterations=3)
    
    @pytest.fixture
    def suite(self, config):
        return BenchmarkSuite(config)
    
    def test_run_all(self, suite):
        results = suite.run_all()
        assert "sensor" in results
        assert "fusion" in results
        assert "control" in results
        assert "embodied" in results
    
    def test_summary(self, suite):
        suite.run_all()
        summary = suite.summary()
        assert "AGV" in summary
        assert "通过率" in summary


class TestLatencyMetrics:
    """测试延迟指标"""
    
    def test_compute_empty(self):
        m = LatencyMetrics.compute([])
        assert m.min_ms == 0
        assert m.mean_ms == 0
        assert m.fps == 0
    
    def test_compute_with_data(self):
        latencies = [10.0, 12.0, 11.0, 13.0, 15.0, 9.0, 14.0, 10.5, 11.5, 12.5]
        m = LatencyMetrics.compute(latencies)
        assert m.min_ms == 9.0
        assert m.max_ms == 15.0
        assert 10 < m.mean_ms < 12
        assert m.p95_ms > m.median_ms
        assert m.fps > 0
    
    def test_is_within_spec(self):
        m = LatencyMetrics.compute([8.0, 9.0, 10.0])
        assert m.is_within_spec(10.0) is True
        assert m.is_within_spec(5.0) is False


class TestMultimodalMetrics:
    """测试多模态指标"""
    
    def test_is_acceptable(self):
        good = MultimodalMetrics(accuracy=0.9, f1_score=0.88, precision=0.9, recall=0.86, auroc=0.9)
        bad = MultimodalMetrics(accuracy=0.7, f1_score=0.65, precision=0.7, recall=0.6, auroc=0.7)
        assert good.is_acceptable(0.85) is True
        assert bad.is_acceptable(0.85) is False


class TestControlMetrics:
    """测试控制指标"""
    
    def test_compute(self):
        reference = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([0.95, 2.05, 2.9, 4.1, 5.0])
        metrics = ControlMetrics.compute(reference, actual)
        assert metrics.tracking_error_mean < 0.1
        assert metrics.tracking_error_max < 0.15


class TestComputeMetrics:
    """测试指标计算函数"""
    
    def test_compute_multimodal_f1(self):
        predictions = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        targets = np.array([0, 1, 2, 0, 1, 2, 0, 1, 1, 0])
        metrics = compute_multimodal_f1(predictions, targets, num_classes=3)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.f1_score <= 1
        assert metrics.conf_matrix is not None
    
    def test_compute_control_accuracy(self):
        reference = np.array([1.0, 2.0, 3.0])
        actual = np.array([1.0, 2.0, 3.0])
        accuracy, metrics = compute_control_accuracy(reference, actual, tolerance=0.01)
        assert accuracy > 90


class TestLatencyTracker:
    """测试延迟跟踪器"""
    
    def test_start_end(self):
        tracker = LatencyTracker(window_size=10)
        tracker.start()
        time.sleep(0.01)
        latency = tracker.end()
        assert latency >= 10.0
        assert tracker.get_metrics().mean_ms >= 10.0
    
    def test_window_overflow(self):
        tracker = LatencyTracker(window_size=5)
        for _ in range(10):
            tracker.start()
            tracker.end()
        assert len(tracker.latencies) == 5
    
    def test_reset(self):
        tracker = LatencyTracker(window_size=10)
        tracker.start()
        tracker.end()
        tracker.reset()
        assert len(tracker.latencies) == 0
        assert tracker._start is None


class TestEvaluationReporter:
    """测试评估报告生成器"""
    
    @pytest.fixture
    def reporter(self, tmp_path):
        return EvaluationReporter(grade="M", output_dir=str(tmp_path))
    
    @pytest.fixture
    def sample_results(self):
        return [
            BenchmarkResult("Vision", "M", True, 8.0, 10.0, 30.0, 100.0, 120.0, 0.95),
            BenchmarkResult("Fusion", "M", True, 5.0, 8.0, 20.0, 80.0, 200.0, 0.93),
            BenchmarkResult("Control", "M", False, 15.0, 10.0, 10.0, 50.0, 66.0, 0.85),
        ]
    
    def test_add_results(self, reporter, sample_results):
        reporter.add_results(sample_results)
        assert len(reporter.results) == 3
    
    def test_generate_json(self, reporter, sample_results, tmp_path):
        reporter.add_results(sample_results)
        path = reporter.generate_json()
        assert Path(path).exists()
        assert path.endswith(".json")
    
    def test_generate_markdown(self, reporter, sample_results, tmp_path):
        reporter.add_results(sample_results)
        path = reporter.generate_markdown()
        assert Path(path).exists()
        assert path.endswith(".md")
        content = Path(path).read_text()
        assert "AGV" in content
        assert "Vision" in content
    
    def test_generate_html(self, reporter, sample_results, tmp_path):
        reporter.add_results(sample_results)
        path = reporter.generate_html()
        assert Path(path).exists()
        assert path.endswith(".html")
    
    def test_generate_all(self, reporter, sample_results, tmp_path):
        reporter.add_results(sample_results)
        paths = reporter.generate_all()
        assert "json" in paths
        assert "markdown" in paths
        assert "html" in paths


class TestAllGrades:
    """测试所有 AGV 等级的基准"""
    
    @pytest.mark.parametrize("grade", [AGVGrade.S, AGVGrade.M, AGVGrade.L, AGVGrade.XL, AGVGrade.XXL])
    def test_all_grade_specs(self, grade):
        """验证所有等级的规格一致性"""
        g = grade.value
        assert g in AGV_LATENCY_SPEC
        assert g in AGV_MEMORY_SPEC
        
        latency = AGV_LATENCY_SPEC[g]
        memory = AGV_MEMORY_SPEC[g]
        
        assert latency["sensor"] > 0
        assert latency["fusion"] > 0
        assert latency["control"] > 0
        assert latency["total"] > 0
        assert memory["ram"] > 0
        assert memory["flash"] > 0
