"""
评估模块
========

SuperModel 超模态机器人具身智能大脑的性能评估与基准测试

主要功能:
- 传感器性能评估
- 融合网络性能评估
- 控制延迟基准测试
- 具身智能综合评估
- AGV 五级合规性评估
"""

from .benchmark import (
    BenchmarkSuite,
    BenchmarkConfig,
    BenchmarkResult,
    SensorBenchmark,
    FusionBenchmark,
    ControlBenchmark,
    EmbodiedBenchmark,
)
from .metrics import (
    MultimodalMetrics,
    ControlMetrics,
    LatencyMetrics,
    compute_multimodal_f1,
    compute_control_accuracy,
)
from .reporter import EvaluationReporter

__all__ = [
    "BenchmarkSuite",
    "BenchmarkConfig", 
    "BenchmarkResult",
    "SensorBenchmark",
    "FusionBenchmark",
    "ControlBenchmark",
    "EmbodiedBenchmark",
    "MultimodalMetrics",
    "ControlMetrics", 
    "LatencyMetrics",
    "compute_multimodal_f1",
    "compute_control_accuracy",
    "EvaluationReporter",
]
