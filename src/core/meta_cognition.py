# Copyright (C) 2024-2026 赵元请 (DIT4FUN)
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
meta_cognition.py - 元认知模块 (Meta-Cognition)
================================================

SuperModel 超模态大模型具身智能系统

元认知是"对认知的认知"——对自己思维过程的觉察、监控和调节。
本模块为AGV具身智能系统提供：

功能:
  - 认知负荷追踪 (Cognitive Load Tracking)
  - 注意力资源管理 (Attention Resource Management)
  - 推理不确定性量化 (Uncertainty Quantification)
  - 认知偏差检测 (Cognitive Bias Detection)
  - 决策信心评估 (Decision Confidence Assessment)
  - 自我效能监控 (Self-Efficacy Monitoring)
  - 元认知学习 (Metacognitive Learning)

架构位置:
  感知层 → 认知层 → 元认知层 → 执行层
              ↑
         [监控/调节/反馈]

与现有模块的关系:
  - CoreBrain: 元认知监控CoreBrain的推理质量
  - DecisionMaking: 提供不确定性输入，增强决策
  - ContextUnderstanding: 评估上下文理解的置信度
  - SafetyShield: 元认知安全底线，配合SafetyShield工作

AGV五级规格适配:
  S:  基础认知监控
  M:  注意力管理 + 不确定性量化
  L:  认知偏差检测 + 信心评估
  XL: 元认知学习 + 自适应调节
  XXL: 完整元认知系统 + 跨任务迁移
"""

from __future__ import annotations

import time
import math
import threading
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union
from enum import Enum
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    'CognitiveLoadLevel',
    'AttentionState',
    'UncertaintyLevel',
    'BiasType',
    'MetacognitiveDecision',
    'CognitiveMetrics',
    'AttentionManager',
    'UncertaintyTracker',
    'BiasDetector',
    'ConfidenceEvaluator',
    'SelfEfficacyMonitor',
    'MetaCognitionEngine',
    'MetaCogConfig',
    'CognitiveSnapshot',
]


# ============================================================
# 枚举定义
# ============================================================

class CognitiveLoadLevel(Enum):
    """认知负荷等级"""
    IDLE = "idle"           # 空闲
    LOW = "low"             # 低负荷
    MODERATE = "moderate"   # 中等负荷
    HIGH = "high"           # 高负荷
    OVERLOADED = "overloaded"  # 过载


class AttentionState(Enum):
    """注意力状态"""
    FOCUSED = "focused"           # 集中
    DIVIDED = "divided"           # 分散
    SUSTAINED = "sustained"       # 持续
    VIGILANT = "vigilant"         # 警觉
    FATIGUED = "fatigued"         # 疲劳
    DEPLETED = "depleted"         # 耗尽


class UncertaintyLevel(Enum):
    """不确定性等级"""
    CERTAIN = "certain"           # 确定
    LIKELY = "likely"             # 大概率
    UNCERTAIN = "uncertain"       # 不确定
    VERY_UNCERTAIN = "very_uncertain"  # 非常不确定
    UNKNOWN = "unknown"           # 完全未知


class BiasType(Enum):
    """认知偏差类型"""
    CONFIRMATION = "confirmation"       # 确认偏差
    ANCHORING = "anchoring"             # 锚定偏差
    AVAILABILITY = "availability"      # 可得性偏差
    OVERCONFIDENCE = "overconfidence"   # 过度自信
    RECENCY = "recency"                 # 近因偏差
    PRIMACY = "primacy"                 # 首因偏差
    GAMBLER_FALLACY = "gambler_fallacy"  # 赌徒谬误


# ============================================================
# 配置与数据结构
# ============================================================

@dataclass
class MetaCogConfig:
    """元认知配置"""
    # AGV等级
    grade: str = "M"

    # 认知负荷窗口大小 (滚动窗口)
    load_window_size: int = 100

    # 注意力参数
    attention_capacity: float = 1.0    # 最大注意力容量 [0, 1]
    vigilance_threshold: float = 0.3   # 警觉阈值
    fatigue_threshold: float = 0.7    # 疲劳阈值

    # 不确定性追踪
    uncertainty_history_size: int = 50
    confidence_window: int = 20

    # 偏差检测阈值
    bias_detection_enabled: bool = True
    overconfidence_threshold: float = 0.15  # 超过此误差视为过度自信

    # 决策信心
    min_confidence_threshold: float = 0.6   # 决策最低信心阈值
    low_confidence_action: str = "defer"    # 低信心时动作: defer/halt/random

    # 元认知学习
    metacognitive_learning_enabled: bool = True
    learning_rate: float = 0.05

    # 采样率
    monitoring_rate_hz: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'grade': self.grade,
            'load_window_size': self.load_window_size,
            'attention_capacity': self.attention_capacity,
            'vigilance_threshold': self.vigilance_threshold,
            'fatigue_threshold': self.fatigue_threshold,
            'uncertainty_history_size': self.uncertainty_history_size,
            'confidence_window': self.confidence_window,
            'bias_detection_enabled': self.bias_detection_enabled,
            'overconfidence_threshold': self.overconfidence_threshold,
            'min_confidence_threshold': self.min_confidence_threshold,
            'low_confidence_action': self.low_confidence_action,
            'metacognitive_learning_enabled': self.metacognitive_learning_enabled,
            'learning_rate': self.learning_rate,
            'monitoring_rate_hz': self.monitoring_rate_hz,
        }


@dataclass
class CognitiveMetrics:
    """认知指标快照"""
    timestamp: float
    cognitive_load: float = 0.0          # [0, 1] 认知负荷
    load_level: CognitiveLoadLevel = CognitiveLoadLevel.IDLE
    attention_used: float = 0.0           # [0, 1] 已用注意力
    attention_state: AttentionState = AttentionState.FOCUSED
    uncertainty: float = 0.0              # [0, 1] 不确定性
    uncertainty_level: UncertaintyLevel = UncertaintyLevel.CERTAIN
    confidence: float = 1.0               # [0, 1] 决策信心
    bias_active: List[BiasType] = field(default_factory=list)
    self_efficacy: float = 1.0            # [0, 1] 自我效能
    processing_latency_ms: float = 0.0   # 处理延迟

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'cognitive_load': self.cognitive_load,
            'load_level': self.load_level.value,
            'attention_used': self.attention_used,
            'attention_state': self.attention_state.value,
            'uncertainty': self.uncertainty,
            'uncertainty_level': self.uncertainty_level.value,
            'confidence': self.confidence,
            'bias_active': [b.value for b in self.bias_active],
            'self_efficacy': self.self_efficacy,
            'processing_latency_ms': self.processing_latency_ms,
        }


@dataclass
class CognitiveSnapshot:
    """完整认知状态快照"""
    metrics: CognitiveMetrics
    perception_quality: float = 1.0      # 感知质量
    reasoning_quality: float = 1.0       # 推理质量
    decision_quality: float = 1.0         # 决策质量
    action_quality: float = 1.0          # 行动质量
    overall_cognition_quality: float = 1.0  # 总体认知质量
    needs_intervention: bool = False
    intervention_recommendation: str = ""


# ============================================================
# 注意力管理器 (Attention Manager)
# ============================================================

class AttentionManager:
    """注意力资源管理器"""

    def __init__(self, config: MetaCogConfig):
        self._config = config
        self._capacity = config.attention_capacity
        self._used: float = 0.0
        self._allocations: Dict[str, float] = {}
        self._history: Deque[Tuple[float, float]] = deque(maxlen=config.load_window_size)
        self._lock = threading.Lock()

    @property
    def available(self) -> float:
        """可用注意力容量"""
        return max(0.0, self._capacity - self._used)

    @property
    def utilization(self) -> float:
        """注意力利用率"""
        if self._capacity <= 0:
            return 1.0
        return self._used / self._capacity

    def allocate(self, task_name: str, amount: float) -> bool:
        """分配注意力资源"""
        with self._lock:
            if self.available >= amount:
                self._used += amount
                self._allocations[task_name] = self._allocations.get(task_name, 0.0) + amount
                self._history.append((time.time(), self._used))
                return True
            return False

    def release(self, task_name: str, amount: Optional[float] = None) -> float:
        """释放注意力资源"""
        with self._lock:
            released = 0.0
            current = self._allocations.get(task_name, 0.0)
            if amount is None:
                released = current
            else:
                released = min(current, amount)
            
            self._allocations[task_name] = current - released
            self._used = max(0.0, self._used - released)
            return released

    def release_all(self, task_name: str) -> float:
        """释放任务所有注意力"""
        return self.release(task_name, None)

    def get_state(self) -> AttentionState:
        """获取当前注意力状态"""
        util = self.utilization
        if util < self._config.vigilance_threshold:  # < 0.3
            return AttentionState.VIGILANT
        elif util < 0.5:  # 0.3 - 0.5
            return AttentionState.FOCUSED
        elif util < 0.7:  # 0.5 - 0.7
            return AttentionState.SUSTAINED
        elif util < 0.8:  # 0.7 - 0.8
            return AttentionState.FATIGUED
        elif util < 0.95:  # 0.8 - 0.95
            return AttentionState.DIVIDED
        else:  # >= 0.95
            return AttentionState.DEPLETED

    def get_allocations(self) -> Dict[str, float]:
        """获取注意力分配详情"""
        with self._lock:
            return dict(self._allocations)

    def reset(self) -> None:
        """重置注意力管理器"""
        with self._lock:
            self._used = 0.0
            self._allocations.clear()


# ============================================================
# 不确定性追踪器 (Uncertainty Tracker)
# ============================================================

class UncertaintyTracker:
    """推理不确定性量化追踪器"""

    def __init__(self, config: MetaCogConfig):
        self._config = config
        self._history: Deque[Tuple[float, float, float]] = deque(maxlen=config.uncertainty_history_size)
        self._decision_uncertainties: Deque[float] = deque(maxlen=config.confidence_window)
        self._outcome_log: Deque[Tuple[float, float]] = deque(maxlen=config.confidence_window)  # (pred, actual)
        self._total_uncertainty: float = 0.0
        self._count: int = 0
        self._lock = threading.Lock()

    def add(self, uncertainty: float, context: Optional[str] = None) -> None:
        """添加不确定性观测"""
        with self._lock:
            self._history.append((time.time(), uncertainty, hash(context) if context else 0))
            self._decision_uncertainties.append(uncertainty)
            self._total_uncertainty += uncertainty
            self._count += 1

    def add_outcome(self, predicted: float, actual: float) -> None:
        """添加决策结果用于校准"""
        with self._lock:
            error = abs(predicted - actual)
            self._outcome_log.append((predicted, actual))
            # 更新校准因子
            self._update_calibration()

    def _update_calibration(self) -> None:
        """更新不确定性校准"""
        if len(self._outcome_log) < 3:
            return
        errors = [abs(p - a) for p, a in self._outcome_log]
        mean_error = sum(errors) / len(errors)
        # 如果平均误差大，说明不确定性被低估

    def get_level(self) -> UncertaintyLevel:
        """获取当前不确定性等级"""
        if not self._decision_uncertainties:
            return UncertaintyLevel.UNKNOWN
        current = self._decision_uncertainties[-1]
        if current < 0.1:
            return UncertaintyLevel.CERTAIN
        elif current < 0.3:
            return UncertaintyLevel.LIKELY
        elif current < 0.5:
            return UncertaintyLevel.UNCERTAIN
        elif current < 0.7:
            return UncertaintyLevel.VERY_UNCERTAIN
        else:
            return UncertaintyLevel.UNKNOWN

    def get_trend(self) -> str:
        """获取不确定性趋势"""
        if len(self._decision_uncertainties) < 5:
            return "insufficient_data"
        recent = list(self._decision_uncertainties)[-5:]
        if all(recent[i] <= recent[i+1] for i in range(len(recent)-1)):
            return "increasing"
        elif all(recent[i] >= recent[i+1] for i in range(len(recent)-1)):
            return "decreasing"
        return "stable"

    def get_current(self) -> float:
        """获取当前不确定性值"""
        if not self._decision_uncertainties:
            return 0.0
        return self._decision_uncertainties[-1]

    def get_average(self) -> float:
        """获取平均不确定性"""
        if not self._decision_uncertainties:
            return 0.0
        return sum(self._decision_uncertainties) / len(self._decision_uncertainties)


# ============================================================
# 认知偏差检测器 (Bias Detector)
# ============================================================

class BiasDetector:
    """认知偏差检测器"""

    def __init__(self, config: MetaCogConfig):
        self._config = config
        self._bias_history: Dict[BiasType, Deque[float]] = {
            b: deque(maxlen=50) for b in BiasType
        }
        self._prediction_errors: Deque[float] = deque(maxlen=100)
        self._recent_predictions: Deque[float] = deque(maxlen=100)
        self._recent_actuals: Deque[float] = deque(maxlen=100)
        self._confidences: Deque[float] = deque(maxlen=100)
        self._lock = threading.Lock()

    def record_prediction(self, prediction: float, actual: Optional[float] = None,
                         confidence: float = 1.0) -> None:
        """记录预测用于偏差检测"""
        with self._lock:
            self._recent_predictions.append(prediction)
            self._confidences.append(confidence)
            if actual is not None:
                self._recent_actuals.append(actual)
                error = abs(prediction - actual)
                self._prediction_errors.append(error)
                # 检测过度自信
                self._bias_history[BiasType.OVERCONFIDENCE].append(
                    1.0 if confidence > 0.9 and error > self._config.overconfidence_threshold else 0.0
                )
                # 检测近因偏差 (最近的结果影响更大)
                if len(self._recent_actuals) >= 10:
                    recent_outcomes = list(self._recent_actuals)[-5:]
                    older_outcomes = list(self._recent_actuals)[-10:-5] if len(self._recent_actuals) >= 10 else []
                    if recent_outcomes and older_outcomes:
                        self._bias_history[BiasType.RECENCY].append(
                            float(abs(sum(recent_outcomes)/len(recent_outcomes) -
                                    sum(older_outcomes)/len(older_outcomes))) > 0.1
                        )

    def detect_active_biases(self) -> List[BiasType]:
        """检测当前活跃的认知偏差"""
        if not self._config.bias_detection_enabled:
            return []
        
        active = []
        with self._lock:
            # 过度自信检测
            oc_history = self._bias_history[BiasType.OVERCONFIDENCE]
            if len(oc_history) >= 10:
                oc_rate = sum(oc_history) / len(oc_history)
                if oc_rate > 0.3:
                    active.append(BiasType.OVERCONFIDENCE)

            # 近因偏差检测
            rec_history = self._bias_history[BiasType.RECENCY]
            if len(rec_history) >= 5:
                rec_rate = sum(rec_history) / len(rec_history)
                if rec_rate > 0.4:
                    active.append(BiasType.RECENCY)

            # 可得性偏差 (频繁事件被高估)
            # 如果某事件连续出现，系统会高估其概率
            if len(self._recent_predictions) >= 5:
                recent_vals = list(self._recent_predictions)[-5:]
                if len(set(recent_vals)) == 1:
                    active.append(BiasType.AVAILABILITY)

        return active

    def get_bias_severity(self, bias: BiasType) -> float:
        """获取偏差严重程度 [0, 1]"""
        with self._lock:
            history = self._bias_history.get(bias)
            if not history or len(history) < 3:
                return 0.0
            return sum(history) / len(history)


# ============================================================
# 决策信心评估器 (Confidence Evaluator)
# ============================================================

class ConfidenceEvaluator:
    """决策信心评估器"""

    def __init__(self, config: MetaCogConfig):
        self._config = config
        self._decision_history: Deque[Dict[str, float]] = deque(maxlen=200)
        self._calibration_data: List[Tuple[float, float]] = []  # (confidence, accuracy)
        self._calibration_buckets: Dict[int, List[float]] = {i: [] for i in range(10)}
        self._lock = threading.Lock()

    def evaluate(self, 
                 uncertainty: float,
                 attention_state: AttentionState,
                 cognitive_load: float,
                 context_consistency: float = 1.0,
                 sensor_agreement: float = 1.0,
                 prior_success_rate: float = 0.5) -> float:
        """综合评估决策信心 [0, 1]"""
        
        # 各维度权重 (可学习)
        w_uncertainty = 0.30
        w_attention = 0.15
        w_load = 0.15
        w_consistency = 0.20
        w_sensors = 0.10
        w_prior = 0.10

        # 不确定性贡献 (越低越好)
        uncertainty_score = 1.0 - min(uncertainty, 1.0)

        # 注意力贡献
        attention_scores = {
            AttentionState.FOCUSED: 1.0,
            AttentionState.SUSTAINED: 0.9,
            AttentionState.VIGILANT: 0.85,
            AttentionState.DIVIDED: 0.6,
            AttentionState.FATIGUED: 0.4,
            AttentionState.DEPLETED: 0.1,
        }
        attention_score = attention_scores.get(attention_state, 0.5)

        # 负荷贡献 (低负荷好)
        load_score = 1.0 - min(cognitive_load, 1.0)

        # 综合评分
        confidence = (
            w_uncertainty * uncertainty_score +
            w_attention * attention_score +
            w_load * load_score +
            w_consistency * context_consistency +
            w_sensors * sensor_agreement +
            w_prior * prior_success_rate
        )

        return min(1.0, max(0.0, confidence))

    def record_decision(self, decision_id: str, confidence: float, 
                       outcome: Optional[bool] = None) -> None:
        """记录决策及其结果用于校准"""
        with self._lock:
            bucket_idx = min(9, int(confidence * 10))
            self._decision_history.append({
                'id': decision_id,
                'confidence': confidence,
                'outcome': outcome,
                'timestamp': time.time(),
            })
            if outcome is not None:
                self._calibration_buckets[bucket_idx].append(1.0 if outcome else 0.0)
                self._calibration_data.append((confidence, 1.0 if outcome else 0.0))

    def get_calibration_error(self) -> float:
        """计算信心校准误差 (ECE)"""
        with self._lock:
            if not self._calibration_data:
                return 0.0
            total_error = 0.0
            total_count = 0
            for bucket_idx, outcomes in self._calibration_buckets.items():
                if outcomes:
                    avg_confidence = (bucket_idx + 0.5) / 10.0
                    avg_accuracy = sum(outcomes) / len(outcomes)
                    total_error += len(outcomes) * abs(avg_confidence - avg_accuracy)
                    total_count += len(outcomes)
            if total_count == 0:
                return 0.0
            return total_error / total_count

    def is_calibrated(self, threshold: float = 0.1) -> bool:
        """检查是否已良好校准"""
        return self.get_calibration_error() < threshold


# ============================================================
# 自我效能监控器 (Self-Efficacy Monitor)
# ============================================================

class SelfEfficacyMonitor:
    """自我效能监控器 (Bandura自我效能理论实现)"""

    def __init__(self, config: MetaCogConfig):
        self._config = config
        self._efficacy_by_domain: Dict[str, Deque[float]] = {}
        self._success_log: Deque[Tuple[str, bool, float]] = deque(maxlen=500)  # (domain, success, difficulty)
        self._domain_mastery: Dict[str, float] = {}
        self._overall_efficacy: float = 0.75
        self._lock = threading.Lock()

    def register_outcome(self, domain: str, success: bool, difficulty: float = 0.5) -> None:
        """注册任务结果影响自我效能"""
        with self._lock:
            self._success_log.append((domain, success, difficulty))
            if domain not in self._efficacy_by_domain:
                self._efficacy_by_domain[domain] = deque(maxlen=50)
            
            # 难度调节的效能更新
            # 困难任务成功 → 大效能提升; 简单任务失败 → 大效能下降
            impact = (2.0 * difficulty - 1.0) * 0.1 if success else -(2.0 * difficulty) * 0.1
            current = self._domain_mastery.get(domain, 0.5)
            new_efficacy = min(1.0, max(0.0, current + impact))
            self._domain_mastery[domain] = new_efficacy
            self._efficacy_by_domain[domain].append(new_efficacy)
            
            # 更新总体效能
            self._update_overall()

    def _update_overall(self) -> None:
        """更新总体自我效能"""
        if not self._domain_mastery:
            return
        self._overall_efficacy = sum(self._domain_mastery.values()) / len(self._domain_mastery)

    def get_efficacy(self, domain: Optional[str] = None) -> float:
        """获取自我效能"""
        with self._lock:
            if domain:
                return self._domain_mastery.get(domain, 0.5)
            return self._overall_efficacy

    def get_mastery_trend(self, domain: str) -> str:
        """获取领域技能掌握趋势"""
        with self._lock:
            history = self._efficacy_by_domain.get(domain)
            if not history or len(history) < 5:
                return "insufficient_data"
            recent = list(history)[-5:]
            if all(recent[i] <= recent[i+1] for i in range(len(recent)-1)):
                return "improving"
            elif all(recent[i] >= recent[i+1] for i in range(len(recent)-1)):
                return "declining"
            return "stable"

    def get_domain_summary(self) -> Dict[str, float]:
        """获取各领域自我效能"""
        with self._lock:
            return dict(self._domain_mastery)


# ============================================================
# 认知负荷追踪器 (Cognitive Load Tracker)
# ============================================================

class CognitiveLoadTracker:
    """认知负荷实时追踪器"""

    def __init__(self, config: MetaCogConfig):
        self._config = config
        self._load_history: Deque[Tuple[float, float]] = deque(maxlen=config.load_window_size)
        self._perception_load: float = 0.0
        self._reasoning_load: float = 0.0
        self._action_load: float = 0.0
        self._peak_load: float = 0.0
        self._lock = threading.Lock()

    def update(self, perception_load: float = 0.0, reasoning_load: float = 0.0,
               action_load: float = 0.0) -> None:
        """更新认知负荷分量"""
        with self._lock:
            self._perception_load = max(0.0, min(1.0, perception_load))
            self._reasoning_load = max(0.0, min(1.0, reasoning_load))
            self._action_load = max(0.0, min(1.0, action_load))
            
            total = (self._perception_load * 0.3 + 
                     self._reasoning_load * 0.4 + 
                     self._action_load * 0.3)
            self._load_history.append((time.time(), total))
            self._peak_load = max(self._peak_load, total)

    def get_total(self) -> float:
        """获取总认知负荷 [0, 1]"""
        with self._lock:
            if self._load_history:
                return self._load_history[-1][1]
            return 0.0

    def get_level(self) -> CognitiveLoadLevel:
        """获取认知负荷等级"""
        load = self.get_total()
        if load < 0.2:
            return CognitiveLoadLevel.IDLE
        elif load < 0.4:
            return CognitiveLoadLevel.LOW
        elif load < 0.6:
            return CognitiveLoadLevel.MODERATE
        elif load < 0.8:
            return CognitiveLoadLevel.HIGH
        else:
            return CognitiveLoadLevel.OVERLOADED

    def get_component_breakdown(self) -> Dict[str, float]:
        """获取认知负荷分量分解"""
        with self._lock:
            return {
                'perception': self._perception_load,
                'reasoning': self._reasoning_load,
                'action': self._action_load,
                'total': self.get_total(),
            }

    def get_trend(self, window: int = 10) -> str:
        """获取认知负荷趋势"""
        with self._lock:
            if len(self._load_history) < window:
                return "insufficient_data"
            recent = [load for _, load in list(self._load_history)[-window:]]
            if all(recent[i] <= recent[i+1] for i in range(len(recent)-1)):
                return "increasing"
            elif all(recent[i] >= recent[i+1] for i in range(len(recent)-1)):
                return "decreasing"
            return "stable"

    def get_peak(self) -> float:
        """获取历史峰值负荷"""
        with self._lock:
            return self._peak_load

    def reset_peak(self) -> None:
        """重置峰值记录"""
        with self._lock:
            self._peak_load = self.get_total()


# ============================================================
# 元认知决策建议 (Metacognitive Decision)
# ============================================================

@dataclass
class MetacognitiveDecision:
    """元认知决策建议"""
    action: str                          # 建议动作
    confidence: float                    # 建议信心 [0, 1]
    reasoning: str                       # 推理过程
    urgency: str = "normal"             # 紧急程度: low/normal/high/critical
    alternative_actions: List[str] = field(default_factory=list)
    safety_override: bool = False        # 是否需要安全覆盖


# ============================================================
# 元认知引擎 (Meta-Cognition Engine) - 主入口
# ============================================================

class MetaCognitionEngine:
    """
    元认知引擎 - 整合所有元认知子模块的主引擎

    使用示例:
        config = MetaCogConfig(grade="L")
        engine = MetaCognitionEngine(config)
        engine.start()

        # 每个决策周期调用
        decision = engine.evaluate_situation(
            perception_data={...},
            reasoning_data={...},
            action_data={...},
        )

        if decision.needs_intervention():
            engine.apply_compensation()

        engine.stop()
    """

    def __init__(self, config: Optional[MetaCogConfig] = None):
        self._config = config or MetaCogConfig()
        self._running = False
        self._lock = threading.Lock()

        # 子模块初始化
        self.attention = AttentionManager(self._config)
        self.uncertainty = UncertaintyTracker(self._config)
        self.bias_detector = BiasDetector(self._config)
        self.confidence = ConfidenceEvaluator(self._config)
        self.efficacy = SelfEfficacyMonitor(self._config)
        self.load_tracker = CognitiveLoadTracker(self._config)

        # 指标历史
        self._metrics_history: Deque[CognitiveMetrics] = deque(maxlen=1000)

        # 事件回调
        self._callbacks: Dict[str, List[Callable]] = {
            'cognitive_overload': [],
            'low_confidence': [],
            'bias_detected': [],
            'efficacy_change': [],
            'intervention': [],
        }

    # ─────────────────────────────────────────────────────────
    # 生命周期
    # ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """启动元认知引擎"""
        with self._lock:
            self._running = True
            logger.info(f"MetaCognitionEngine started (grade={self._config.grade})")

    def stop(self) -> None:
        """停止元认知引擎"""
        with self._lock:
            self._running = False
            self.attention.reset()
            logger.info("MetaCognitionEngine stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ─────────────────────────────────────────────────────────
    # 核心评估接口
    # ─────────────────────────────────────────────────────────

    def evaluate_situation(self,
                           perception_load: float = 0.0,
                           reasoning_load: float = 0.0,
                           action_load: float = 0.0,
                           uncertainty: float = 0.0,
                           context_consistency: float = 1.0,
                           sensor_agreement: float = 1.0,
                           prior_success_rate: float = 0.5,
                           domain: str = "general") -> CognitiveSnapshot:
        """综合评估当前认知状态并返回快照"""
        if not self._running:
            return self._create_idle_snapshot()

        # 更新负荷追踪
        self.load_tracker.update(perception_load, reasoning_load, action_load)

        # 添加不确定性
        self.uncertainty.add(uncertainty)

        # 获取注意力状态
        attention_state = self.attention.get_state()

        # 检测偏差
        active_biases = self.bias_detector.detect_active_biases()

        # 计算决策信心
        confidence = self.confidence.evaluate(
            uncertainty=uncertainty,
            attention_state=attention_state,
            cognitive_load=self.load_tracker.get_total(),
            context_consistency=context_consistency,
            sensor_agreement=sensor_agreement,
            prior_success_rate=prior_success_rate,
        )

        # 获取自我效能
        self_efficacy = self.efficacy.get_efficacy(domain)

        # 构建指标
        metrics = CognitiveMetrics(
            timestamp=time.time(),
            cognitive_load=self.load_tracker.get_total(),
            load_level=self.load_tracker.get_level(),
            attention_used=self.attention.utilization,
            attention_state=attention_state,
            uncertainty=uncertainty,
            uncertainty_level=self.uncertainty.get_level(),
            confidence=confidence,
            bias_active=active_biases,
            self_efficacy=self_efficacy,
        )

        # 触发回调
        self._check_and_fire_callbacks(metrics)

        # 记录历史
        self._metrics_history.append(metrics)

        return CognitiveSnapshot(
            metrics=metrics,
            overall_cognition_quality=self._compute_overall_quality(metrics),
            needs_intervention=self._needs_intervention(metrics),
            intervention_recommendation=self._get_intervention(metrics),
        )

    def make_decision_recommendation(self,
                                      confidence_threshold: Optional[float] = None) -> MetacognitiveDecision:
        """基于当前元认知状态给出决策建议"""
        threshold = confidence_threshold or self._config.min_confidence_threshold
        current_metrics = self.get_current_metrics()
        
        if not current_metrics:
            return MetacognitiveDecision(
                action="proceed",
                confidence=0.5,
                reasoning="no metrics available",
            )

        confidence = current_metrics.confidence
        load = current_metrics.cognitive_load
        uncertainty = current_metrics.uncertainty
        attention = current_metrics.attention_state
        biases = current_metrics.bias_active

        # 决策逻辑
        if load >= 0.9:
            action = "defer"
            reasoning = f"认知过载 (load={load:.2f}), 推迟非紧急决策"
            urgency = "high"
        elif confidence < threshold:
            action = self._config.low_confidence_action
            reasoning = f"决策信心不足 (confidence={confidence:.2f} < {threshold})"
            urgency = "normal"
        elif UncertaintyLevel.VERY_UNCERTAIN in [current_metrics.uncertainty_level]:
            action = "gather_more_info"
            reasoning = f"高度不确定 (uncertainty={uncertainty:.2f}), 建议收集更多信息"
            urgency = "normal"
        elif BiasType.OVERCONFIDENCE in biases:
            action = "reconsider"
            reasoning = "检测到过度自信偏差, 建议重新审视决策"
            urgency = "high"
        elif attention in (AttentionState.FATIGUED, AttentionState.DEPLETED):
            action = "rest"
            reasoning = f"注意力状态不佳 ({attention.value}), 建议短暂休息"
            urgency = "normal"
        else:
            action = "proceed"
            reasoning = "认知状态正常, 可以继续执行"
            urgency = "low"

        # 检查安全阈值
        safety_override = load > 0.95 or confidence < 0.3

        alternatives = self._get_alternative_actions(action, current_metrics)

        return MetacognitiveDecision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            urgency=urgency,
            alternative_actions=alternatives,
            safety_override=safety_override,
        )

    # ─────────────────────────────────────────────────────────
    # 结果记录接口 (用于学习与校准)
    # ─────────────────────────────────────────────────────────

    def record_outcome(self, predicted_value: float, actual_value: float,
                       domain: str = "general", success: bool = True) -> None:
        """记录决策结果用于元认知学习"""
        error = abs(predicted_value - actual_value) if not success else 0.0
        
        # 更新偏差检测
        self.bias_detector.record_prediction(
            prediction=predicted_value,
            actual=actual_value if success else None,
            confidence=self.get_current_metrics().confidence if self.get_current_metrics() else 0.5,
        )

        # 更新不确定性校准
        self.uncertainty.add_outcome(predicted_value, actual_value)

        # 更新自我效能
        self.efficacy.register_outcome(
            domain=domain,
            success=success,
            difficulty=min(1.0, error * 2),
        )

    def record_decision_confidence(self, decision_id: str, outcome: Optional[bool] = None) -> None:
        """记录决策信心及结果"""
        metrics = self.get_current_metrics()
        if metrics:
            self.confidence.record_decision(
                decision_id=decision_id,
                confidence=metrics.confidence,
                outcome=outcome,
            )

    # ─────────────────────────────────────────────────────────
    # 注意力资源接口
    # ─────────────────────────────────────────────────────────

    def allocate_attention(self, task: str, amount: float) -> bool:
        """为任务分配注意力资源"""
        return self.attention.allocate(task, amount)

    def release_attention(self, task: str, amount: Optional[float] = None) -> float:
        """释放注意力资源"""
        return self.attention.release(task, amount)

    # ─────────────────────────────────────────────────────────
    # 查询接口
    # ─────────────────────────────────────────────────────────

    def get_current_metrics(self) -> Optional[CognitiveMetrics]:
        """获取当前认知指标"""
        if self._metrics_history:
            return self._metrics_history[-1]
        return None

    def get_metrics_history(self, limit: int = 100) -> List[CognitiveMetrics]:
        """获取历史指标"""
        return list(self._metrics_history)[-limit:]

    def get_summary(self) -> Dict[str, Any]:
        """获取元认知系统摘要"""
        metrics = self.get_current_metrics()
        return {
            'running': self._running,
            'grade': self._config.grade,
            'current': metrics.to_dict() if metrics else None,
            'attention_allocations': self.attention.get_allocations(),
            'uncertainty_trend': self.uncertainty.get_trend(),
            'efficacy_summary': self.efficacy.get_domain_summary(),
            'load_breakdown': self.load_tracker.get_component_breakdown(),
            'load_trend': self.load_tracker.get_trend(),
            'biases_detected': [b.value for b in (metrics.bias_active if metrics else [])],
            'calibration_error': self.confidence.get_calibration_error(),
            'is_calibrated': self.confidence.is_calibrated(),
        }

    # ─────────────────────────────────────────────────────────
    # 事件回调
    # ─────────────────────────────────────────────────────────

    def on(self, event: str, callback: Callable[[CognitiveMetrics], None]) -> None:
        """注册事件回调"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def off(self, event: str, callback: Callable) -> None:
        """取消事件回调"""
        if event in self._callbacks:
            self._callbacks[event] = [c for c in self._callbacks[event] if c != callback]

    # ─────────────────────────────────────────────────────────
    # 私有方法
    # ─────────────────────────────────────────────────────────

    def _create_idle_snapshot(self) -> CognitiveSnapshot:
        """创建空闲状态快照"""
        return CognitiveSnapshot(
            metrics=CognitiveMetrics(
                timestamp=time.time(),
                cognitive_load=0.0,
                load_level=CognitiveLoadLevel.IDLE,
                attention_used=0.0,
                attention_state=AttentionState.FOCUSED,
                uncertainty=0.0,
                uncertainty_level=UncertaintyLevel.CERTAIN,
                confidence=1.0,
                bias_active=[],
                self_efficacy=0.75,
            ),
            needs_intervention=False,
        )

    def _compute_overall_quality(self, metrics: CognitiveMetrics) -> float:
        """计算总体认知质量"""
        q_confidence = metrics.confidence
        q_load = 1.0 - metrics.cognitive_load
        q_attention = 1.0 if metrics.attention_state in (
            AttentionState.FOCUSED, AttentionState.SUSTAINED, AttentionState.VIGILANT
        ) else 0.5
        q_efficacy = metrics.self_efficacy
        q_bias_penalty = 1.0 - (len(metrics.bias_active) * 0.1)

        return min(1.0, max(0.0, 
            q_confidence * 0.35 + q_load * 0.25 + q_attention * 0.15 + q_efficacy * 0.15 + q_bias_penalty * 0.10
        ))

    def _needs_intervention(self, metrics: CognitiveMetrics) -> bool:
        """判断是否需要干预"""
        return (
            metrics.cognitive_load >= 0.85 or
            metrics.confidence < self._config.min_confidence_threshold or
            metrics.attention_state in (AttentionState.DEPLETED,) or
            metrics.uncertainty >= 0.7 or
            BiasType.OVERCONFIDENCE in metrics.bias_active
        )

    def _get_intervention(self, metrics: CognitiveMetrics) -> str:
        """获取干预建议"""
        if metrics.cognitive_load >= 0.9:
            return "COGNITIVE_REST: 降低负荷,暂停非必要推理"
        if metrics.confidence < self._config.min_confidence_threshold:
            return "SEEK_ADVICE: 信心不足,建议寻求外部输入或延迟决策"
        if BiasType.OVERCONFIDENCE in metrics.bias_active:
            return "RETHINK: 检测过度自信,重新审视假设"
        if metrics.uncertainty >= 0.7:
            return "GATHER_INFO: 不确定性高,建议收集更多感知信息"
        if metrics.attention_state == AttentionState.FATIGUED:
            return "REST_ATTENTION: 注意力疲劳,建议短暂休息"
        return "MONITOR: 继续监控"

    def _get_alternative_actions(self, primary: str, 
                                  metrics: CognitiveMetrics) -> List[str]:
        """获取备选动作"""
        alternatives = {
            "proceed": ["proceed_with_caution", "request_human_oversight"],
            "defer": ["delay_5min", "delay_until_load_reduces", "seek_advice"],
            "gather_more_info": ["increase_sensors", "request_clarification", "wait_for_more_data"],
            "reconsider": ["review_assumptions", "consult_history", "request_second_opinion"],
            "rest": ["short_pause", "switch_task", "reduce_attention_allocation"],
        }
        return alternatives.get(primary, ["proceed"])

    def _check_and_fire_callbacks(self, metrics: CognitiveMetrics) -> None:
        """检查条件并触发回调"""
        if metrics.cognitive_load >= 0.85:
            for cb in self._callbacks['cognitive_overload']:
                try: cb(metrics)
                except Exception: pass
        
        if metrics.confidence < self._config.min_confidence_threshold:
            for cb in self._callbacks['low_confidence']:
                try: cb(metrics)
                except Exception: pass
        
        if metrics.bias_active:
            for cb in self._callbacks['bias_detected']:
                try: cb(metrics)
                except Exception: pass

    def __repr__(self) -> str:
        m = self.get_current_metrics()
        status = f"running={self._running}" if True else "stopped"
        if m:
            return f"MetaCognitionEngine({status}, load={m.cognitive_load:.2f}, conf={m.confidence:.2f})"
        return f"MetaCognitionEngine({status})"
