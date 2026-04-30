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
评估指标模块
============

提供多模态感知、控制、延迟等性能指标的计算
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque


@dataclass
class LatencyMetrics:
    """延迟指标"""
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    std_ms: float
    fps: float
    
    @classmethod
    def compute(cls, latencies_ms: List[float]) -> "LatencyMetrics":
        if not latencies_ms:
            return cls(0, 0, 0, 0, 0, 0, 0, 0)
        arr = np.array(latencies_ms)
        mean = np.mean(arr)
        return cls(
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            mean_ms=float(mean),
            median_ms=float(np.median(arr)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            std_ms=float(np.std(arr)),
            fps=float(1000.0 / mean) if mean > 0 else 0,
        )
    
    def is_within_spec(self, spec_ms: float, tolerance: float = 1.2) -> bool:
        return self.p99_ms <= spec_ms * tolerance


@dataclass
class MultimodalMetrics:
    """多模态融合指标"""
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    auroc: float
    conf_matrix: Optional[np.ndarray] = None
    
    def is_acceptable(self, threshold: float = 0.85) -> bool:
        return self.f1_score >= threshold and self.accuracy >= threshold


@dataclass
class ControlMetrics:
    """控制性能指标"""
    tracking_error_mean: float
    tracking_error_max: float
    tracking_error_rmse: float
    overshoot: float
    rise_time_ms: float
    settling_time_ms: float
    steady_state_error: float
    
    @classmethod
    def compute(cls, reference: np.ndarray, actual: np.ndarray) -> "ControlMetrics":
        error = reference - actual
        abs_error = np.abs(error)
        
        return cls(
            tracking_error_mean=float(np.mean(abs_error)),
            tracking_error_max=float(np.max(abs_error)),
            tracking_error_rmse=float(np.sqrt(np.mean(error ** 2))),
            overshoot=float(np.max(abs_error) / (np.max(np.abs(reference)) + 1e-8) * 100),
            rise_time_ms=0.0,
            settling_time_ms=0.0,
            steady_state_error=float(np.mean(np.abs(error[-100:]))),
        )


def _compute_confusion_matrix(predictions: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    """计算混淆矩阵 (无 sklearn 依赖)"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for pred, target in zip(predictions, targets):
        if 0 <= pred < num_classes and 0 <= target < num_classes:
            cm[target, pred] += 1
    return cm


def _compute_precision_recall_f1(cm: np.ndarray) -> Tuple[float, float, float]:
    """从混淆矩阵计算 precision/recall/f1"""
    num_classes = cm.shape[0]
    precisions, recalls, f1s = [], [], []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    return (float(np.mean(precisions)), 
            float(np.mean(recalls)), 
            float(np.mean(f1s)))


def compute_multimodal_f1(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 10,
) -> MultimodalMetrics:
    """计算多模态分类指标 (纯 numpy 实现，无 sklearn 依赖)"""
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()
    
    # 准确率
    accuracy = float(np.mean(predictions == targets))
    
    # 混淆矩阵
    cm = _compute_confusion_matrix(predictions, targets, num_classes)
    
    # 从混淆矩阵计算 precision/recall/f1
    precision, recall, f1 = _compute_precision_recall_f1(cm)
    
    # AUROC (简化为准确率的副本，因为没有 sklearn)
    auroc = accuracy
    
    return MultimodalMetrics(
        accuracy=accuracy,
        f1_score=f1,
        precision=precision,
        recall=recall,
        auroc=auroc,
        conf_matrix=cm,
    )


def compute_control_accuracy(
    reference: np.ndarray,
    actual: np.ndarray,
    tolerance: float = 0.05,
) -> Tuple[float, ControlMetrics]:
    """计算控制准确率"""
    reference = np.asarray(reference)
    actual = np.asarray(actual)
    
    errors = np.abs(reference - actual)
    max_vals = np.maximum(np.abs(reference), 1e-8)
    normalized_errors = errors / max_vals
    
    accuracy = float(np.mean(normalized_errors <= tolerance) * 100)
    metrics = ControlMetrics.compute(reference, actual)
    
    return accuracy, metrics


class LatencyTracker:
    """滑动窗口延迟跟踪器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.latencies: deque = deque(maxlen=window_size)
        self.timestamps: deque = deque(maxlen=window_size)
        self._start: Optional[float] = None
    
    def start(self):
        self._start = time.perf_counter()
    
    def end(self) -> float:
        if self._start is None:
            return 0.0
        latency_ms = (time.perf_counter() - self._start) * 1000
        self.latencies.append(latency_ms)
        self.timestamps.append(time.time())
        self._start = None
        return latency_ms
    
    def get_metrics(self) -> LatencyMetrics:
        return LatencyMetrics.compute(list(self.latencies))
    
    def reset(self):
        self.latencies.clear()
        self.timestamps.clear()
        self._start = None
