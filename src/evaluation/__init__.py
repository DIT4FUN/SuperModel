# Copyright (C) 2026 焦洋 (Jiao Yang) <jiaoyang@cczu.edu.cn>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
