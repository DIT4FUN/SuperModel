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
federated_learning.py - 联邦学习多AGV协同模块
SuperModel 超模态大模型具身智能系统

联邦学习协同:
- 多AGV本地模型训练
- 梯度聚合与模型同步
- 差分隐私保护
- 通信效率优化
- 拜占庭容错
- 自适应聚合权重
"""

from __future__ import annotations

import time
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    'FLClientState',
    'FLRoundResult',
    'LocalTrainingResult',
    'FederatedClient',
    'FederatedServer',
    'DifferentialPrivacy',
    'ByzantineFilter',
    'AdaptiveAggregator',
    'FederatedLearningCoordinator',
    'create_federated_learning_system',
]


class FLClientState(Enum):
    """联邦学习客户端状态"""
    IDLE = "idle"
    TRAINING = "training"
    UPLOADING = "uploading"
    EVALUATING = "evaluating"
    FAILED = "failed"


@dataclass
class LocalTrainingResult:
    """本地训练结果"""
    client_id: str
    round_number: int
    num_samples: int
    training_loss: float
    validation_accuracy: float
    gradients: Dict[str, np.ndarray]       # 模型梯度
    model_update_hash: str                 # 更新哈希 (完整性验证)
    training_time_seconds: float
    communication_bytes: int
    client_state: FLClientState
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)


@dataclass
class FLRoundResult:
    """联邦学习轮次结果"""
    round_number: int
    num_participants: int
    aggregated_model: Dict[str, np.ndarray]
    global_loss: float
    global_accuracy: float
    client_losses: Dict[str, float]
    aggregation_time_seconds: float
    total_communication_bytes: int
    dropouts: List[str]                    # 本轮掉线的客户端
    byzantine_filtered: List[str]          # 拜占庭过滤掉的客户端
    differential_privacy_applied: bool
    epsilon: Optional[float]               # DP隐私预算
    timestamp: float = field(default_factory=time.time)


class DifferentialPrivacy:
    """差分隐私模块"""

    def __init__(self, epsilon: float = 3.0, delta: float = 1e-5, sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.noise_multiplier = self._compute_noise_multiplier()

    def _compute_noise_multiplier(self) -> float:
        """计算噪声乘数 (基于隐私预算)"""
        # 简化的高斯噪声机制
        return self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

    def add_noise_to_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """向梯度添加高斯噪声"""
        noise = np.random.normal(0, self.noise_multiplier, gradient.shape).astype(gradient.dtype)
        return gradient + noise

    def add_noise_to_gradients(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """对所有梯度添加噪声"""
        return {
            name: self.add_noise_to_gradient(arr)
            for name, arr in gradients.items()
        }

    def compute_privacy_spent(self, num_rounds: int) -> Tuple[float, float]:
        """计算已消耗的隐私预算 (基于顺序组合)"""
        # 顺序组合定理
        spent_epsilon = self.epsilon * num_rounds
        spent_delta = num_rounds * self.delta
        return spent_epsilon, spent_delta


class ByzantineFilter:
    """拜占庭容错过滤器"""

    def __init__(self, f: int = 1, n: int = 10):
        """
        Args:
            f: 拜占庭节点数量上限
            n: 总节点数
        """
        self.f = f
        self.n = n
        self._client_metrics: Dict[str, List[float]] = {}

    def register_metric(self, client_id: str, metric: float) -> None:
        """注册客户端指标"""
        if client_id not in self._client_metrics:
            self._client_metrics[client_id] = []
        self._client_metrics[client_id].append(metric)

    def filter_byzantine_clients(self, results: List[LocalTrainingResult]) -> List[str]:
        """
        过滤拜占庭客户端
        Returns: 被过滤的客户端ID列表
        """
        if len(results) < 3:
            return []  # 无法执行过滤
        
        # 方法1: 统计检验 (简化版本)
        losses = [(r.client_id, r.training_loss) for r in results]
        losses_sorted = sorted(losses, key=lambda x: x[1])
        
        # Trimmed mean: 去掉最高和最低的k个
        k = self.f
        if len(losses_sorted) > 2 * k:
            trimmed = losses_sorted[k:-k]
        else:
            trimmed = losses_sorted
        
        mean_loss = np.mean([l for _, l in trimmed])
        std_loss = np.std([l for _, l in trimmed])
        
        filtered = []
        for client_id, loss in losses:
            z_score = abs(loss - mean_loss) / (std_loss + 1e-8)
            if z_score > 3.0:  # 3-sigma原则
                filtered.append(client_id)
                logger.warning(f"ByzantineFilter: 过滤异常客户端 {client_id} (loss={loss:.4f}, z={z_score:.2f})")
        
        return filtered

    def compute_robust_aggregate(
        self,
        updates: Dict[str, np.ndarray],
        weights: Dict[str, float],
    ) -> np.ndarray:
        """计算鲁棒聚合 (Krum/Multi-Krum)"""
        # 简化版本: 加权裁剪平均
        # 实际实现应使用MultiKrum算法
        filtered_updates = {
            cid: up for cid, up in updates.items()
            if cid not in self.filter_byzantine_clients([
                LocalTrainingResult(client_id=cid, round_number=0, num_samples=1,
                                   training_loss=0, validation_accuracy=0,
                                   gradients={}, model_update_hash="", training_time_seconds=0,
                                   communication_bytes=0, client_state=FLClientState.IDLE)
                for cid in updates
            ])
        }
        
        if not filtered_updates:
            return np.zeros_like(next(iter(updates.values())))
        
        total_weight = sum(weights.get(cid, 1.0) for cid in filtered_updates)
        
        result = None
        for cid, update in filtered_updates.items():
            w = weights.get(cid, 1.0) / total_weight
            if result is None:
                result = w * update
            else:
                result = result + w * update
        
        return result if result is not None else np.zeros_like(next(iter(updates.values())))


class AdaptiveAggregator:
    """自适应聚合器 - 根据客户端质量动态调整权重"""

    def __init__(self):
        self._historical_accuracy: Dict[str, List[float]] = {}
        self._communication_costs: Dict[str, List[int]] = {}
        self._reliability_scores: Dict[str, float] = {}

    def update_client_history(
        self,
        client_id: str,
        accuracy: float,
        communication_bytes: int,
    ) -> None:
        """更新客户端历史记录"""
        if client_id not in self._historical_accuracy:
            self._historical_accuracy[client_id] = []
            self._communication_costs[client_id] = []
        
        self._historical_accuracy[client_id].append(accuracy)
        self._communication_costs[client_id].append(communication_bytes)
        
        # 保持最近10轮记录
        for hist in (self._historical_accuracy, self._communication_costs):
            if len(hist[client_id]) > 10:
                hist[client_id] = hist[client_id][-10:]

    def compute_adaptive_weights(
        self,
        results: List[LocalTrainingResult],
    ) -> Dict[str, float]:
        """计算自适应聚合权重"""
        weights = {}
        
        for result in results:
            cid = result.client_id
            
            # 更新可靠性评分
            acc_trend = self._compute_trend(self._historical_accuracy.get(cid, [result.validation_accuracy]))
            comm_cost = np.mean(self._communication_costs.get(cid, [result.communication_bytes]))
            
            # 综合评分
            base_score = result.validation_accuracy
            trend_bonus = acc_trend * 0.2
            efficiency_bonus = 1.0 / (1.0 + comm_cost / 1e6)  # 通信效率
            
            reliability = base_score + trend_bonus + efficiency_bonus
            self._reliability_scores[cid] = reliability
            weights[cid] = reliability
        
        # 归一化权重
        total = sum(weights.values())
        if total > 0:
            weights = {cid: w / total for cid, w in weights.items()}
        
        return weights

    def _compute_trend(self, values: List[float]) -> float:
        """计算趋势斜率"""
        if len(values) < 2:
            return 0.0
        n = len(values)
        x = np.arange(n)
        y = np.array(values)
        
        # 简单线性回归斜率
        mean_x, mean_y = np.mean(x), np.mean(y)
        slope = np.sum((x - mean_x) * (y - mean_y)) / (np.sum((x - mean_x)**2) + 1e-8)
        
        return slope


class FederatedClient:
    """联邦学习客户端"""

    def __init__(
        self,
        client_id: str,
        agv_id: str,
        model_config: Dict,
        local_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.01,
    ):
        self.client_id = client_id
        self.agv_id = agv_id
        self.model_config = model_config
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self._state = FLClientState.IDLE
        self._local_model: Optional[Dict[str, np.ndarray]] = None
        self._training_data_size = 0
        self._round_history: List[Dict] = []

    @property
    def state(self) -> FLClientState:
        return self._state

    def receive_global_model(self, global_model: Dict[str, np.ndarray]) -> None:
        """接收全局模型更新"""
        self._local_model = {k: v.copy() for k, v in global_model.items()}

    def local_train(
        self,
        train_data: Dict,
        validation_data: Optional[Dict] = None,
    ) -> LocalTrainingResult:
        """执行本地训练"""
        self._state = FLClientState.TRAINING
        start_time = time.time()
        
        # 模拟本地训练
        # 实际实现中这里会调用真实的训练循环
        num_samples = train_data.get('size', self.batch_size * 10)
        self._training_data_size = num_samples
        
        # 模拟梯度计算 (简化)
        gradient_shape = self.model_config.get('gradient_shape', (128,))
        gradients = {
            f'layer_{i}': np.random.randn(*gradient_shape).astype(np.float32) * 0.01
            for i in range(self.model_config.get('num_layers', 4))
        }
        
        # 模拟训练损失
        training_loss = np.random.uniform(0.3, 1.5)
        validation_accuracy = np.random.uniform(0.7, 0.98)
        
        # 计算梯度哈希
        update_hash = self._compute_update_hash(gradients)
        
        self._state = FLClientState.UPLOADING
        
        result = LocalTrainingResult(
            client_id=self.client_id,
            round_number=len(self._round_history) + 1,
            num_samples=num_samples,
            training_loss=training_loss,
            validation_accuracy=validation_accuracy,
            gradients=gradients,
            model_update_hash=update_hash,
            training_time_seconds=time.time() - start_time,
            communication_bytes=self._estimate_communication(gradients),
            client_state=self._state,
            metadata={
                'agv_id': self.agv_id,
                'local_epochs': self.local_epochs,
                'batch_size': self.batch_size,
            },
        )
        
        self._round_history.append({
            'round': result.round_number,
            'loss': result.training_loss,
            'accuracy': result.validation_accuracy,
            'timestamp': result.timestamp,
        })
        
        self._state = FLClientState.IDLE
        return result

    def _compute_update_hash(self, gradients: Dict[str, np.ndarray]) -> str:
        """计算梯度更新哈希"""
        hash_parts = []
        for name, arr in sorted(gradients.items()):
            hash_parts.append(name)
            hash_parts.append(hashlib.md5(arr.tobytes()).hexdigest()[:8])
        combined = ''.join(hash_parts)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _estimate_communication(self, gradients: Dict[str, np.ndarray]) -> int:
        """估算通信量 (字节)"""
        total = 0
        for arr in gradients.values():
            total += arr.nbytes
        return total

    def evaluate(self, test_data: Dict) -> Dict:
        """评估本地模型"""
        self._state = FLClientState.EVALUATING
        
        accuracy = np.random.uniform(0.7, 0.98)
        loss = np.random.uniform(0.2, 1.0)
        
        self._state = FLClientState.IDLE
        
        return {
            'client_id': self.client_id,
            'agv_id': self.agv_id,
            'test_accuracy': accuracy,
            'test_loss': loss,
            'num_samples': test_data.get('size', 0),
        }


class FederatedServer:
    """联邦学习服务器"""

    def __init__(
        self,
        model_config: Dict,
        num_rounds: int = 100,
        min_clients_per_round: int = 3,
        aggregation_strategy: str = "fedavg",
        use_differential_privacy: bool = False,
        dp_epsilon: float = 3.0,
    ):
        self.model_config = model_config
        self.num_rounds = num_rounds
        self.min_clients_per_round = min_clients_per_round
        self.aggregation_strategy = aggregation_strategy
        
        # 初始化全局模型
        self._global_model = self._initialize_model()
        self._round_number = 0
        
        # 聚合器
        self._byzantine_filter = ByzantineFilter()
        self._adaptive_aggregator = AdaptiveAggregator()
        
        # 差分隐私
        self._use_dp = use_differential_privacy
        self._dp = DifferentialPrivacy(epsilon=dp_epsilon) if use_differential_privacy else None
        
        # 注册的客户端
        self._clients: Dict[str, FederatedClient] = {}
        
        # 训练历史
        self._round_results: List[FLRoundResult] = []

    def _initialize_model(self) -> Dict[str, np.ndarray]:
        """初始化全局模型"""
        gradient_shape = self.model_config.get('gradient_shape', (128,))
        return {
            f'layer_{i}': np.zeros(gradient_shape, dtype=np.float32)
            for i in range(self.model_config.get('num_layers', 4))
        }

    @property
    def global_model(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self._global_model.items()}

    def register_client(self, client: FederatedClient) -> None:
        """注册联邦学习客户端"""
        self._clients[client.client_id] = client
        logger.info(f"FLServer: 注册客户端 {client.client_id} (AGV: {client.agv_id})")

    def select_clients(self, min_count: Optional[int] = None) -> List[str]:
        """选择参与本轮的客户端"""
        count = min_count or self.min_clients_per_round
        available = [
            cid for cid, client in self._clients.items()
            if client.state == FLClientState.IDLE
        ]
        
        # 优先选择历史表现好的
        if self._adaptive_aggregator._reliability_scores:
            available.sort(
                key=lambda cid: self._adaptive_aggregator._reliability_scores.get(cid, 0),
                reverse=True,
            )
        
        return available[:count]

    def execute_round(
        self,
        selected_client_ids: List[str],
    ) -> FLRoundResult:
        """执行一轮联邦学习"""
        self._round_number += 1
        start_time = time.time()
        
        # 分发全局模型
        for cid in selected_client_ids:
            if cid in self._clients:
                self._clients[cid].receive_global_model(self.global_model)
        
        # 收集本地训练结果
        results: List[LocalTrainingResult] = []
        for cid in selected_client_ids:
            if cid in self._clients:
                try:
                    result = self._clients[cid].local_train(
                        train_data={'size': np.random.randint(100, 500)}
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"FLClient {cid} 训练失败: {e}")
        
        # 拜占庭过滤
        byzantine_filtered = self._byzantine_filter.filter_byzantine_clients(results)
        valid_results = [r for r in results if r.client_id not in byzantine_filtered]
        
        # 计算自适应权重
        weights = self._adaptive_aggregator.compute_adaptive_weights(valid_results)
        
        # 聚合梯度
        aggregated = self._aggregate_updates(valid_results, weights)
        
        # 应用差分隐私
        if self._use_dp and self._dp:
            aggregated = self._dp.add_noise_to_gradients(aggregated)
        
        # 更新全局模型
        for key in self._global_model:
            if key in aggregated:
                self._global_model[key] = aggregated[key]
        
        # 计算全局指标
        global_loss = np.mean([r.training_loss for r in valid_results]) if valid_results else float('inf')
        global_accuracy = np.mean([r.validation_accuracy for r in valid_results]) if valid_results else 0
        
        total_comm = sum(r.communication_bytes for r in valid_results)
        
        # 更新客户端历史
        for r in valid_results:
            self._adaptive_aggregator.update_client_history(
                r.client_id, r.validation_accuracy, r.communication_bytes
            )
        
        round_result = FLRoundResult(
            round_number=self._round_number,
            num_participants=len(valid_results),
            aggregated_model=self.global_model,
            global_loss=global_loss,
            global_accuracy=global_accuracy,
            client_losses={r.client_id: r.training_loss for r in valid_results},
            aggregation_time_seconds=time.time() - start_time,
            total_communication_bytes=total_comm,
            dropouts=[cid for cid in selected_client_ids if cid not in {r.client_id for r in results}],
            byzantine_filtered=byzantine_filtered,
            differential_privacy_applied=self._use_dp,
            epsilon=self._dp.epsilon if self._dp else None,
        )
        
        self._round_results.append(round_result)
        
        return round_result

    def _aggregate_updates(
        self,
        results: List[LocalTrainingResult],
        weights: Dict[str, float],
    ) -> Dict[str, np.ndarray]:
        """聚合客户端更新"""
        if not results:
            return {}
        
        # 计算总样本数
        total_samples = sum(r.num_samples for r in results)
        
        # 加权聚合 (FedAvg)
        aggregated: Dict[str, np.ndarray] = {}
        
        for key in results[0].gradients:
            weighted_sum = None
            for r in results:
                if key not in r.gradients:
                    continue
                weight = weights.get(r.client_id, r.num_samples / total_samples)
                if weighted_sum is None:
                    weighted_sum = weight * r.gradients[key]
                else:
                    weighted_sum = weighted_sum + weight * r.gradients[key]
            
            aggregated[key] = weighted_sum if weighted_sum is not None else np.zeros((128,), dtype=np.float32)
        
        return aggregated

    def get_training_summary(self) -> Dict:
        """获取训练摘要"""
        if not self._round_results:
            return {'status': 'not_started'}
        
        accuracies = [r.global_accuracy for r in self._round_results]
        losses = [r.global_loss for r in self._round_results]
        
        return {
            'current_round': self._round_number,
            'total_rounds': self.num_rounds,
            'participation_rate': np.mean([r.num_participants / len(self._clients) for r in self._round_results[-10:]]),
            'avg_accuracy': np.mean(accuracies[-10:]),
            'best_accuracy': max(accuracies),
            'avg_loss': np.mean(losses[-10:]),
            'total_communication_mb': sum(r.total_communication_bytes for r in self._round_results) / 1e6,
            'dp_epsilon_spent': self._dp.compute_privacy_spent(self._round_number)[0] if self._dp else None,
        }


class FederatedLearningCoordinator:
    """联邦学习协调器 - 管理整个FL系统"""

    def __init__(self, model_config: Dict, grade: str = "L"):
        self.grade = grade
        self._server = FederatedServer(
            model_config=model_config,
            num_rounds=50,
            min_clients_per_round=3,
            use_differential_privacy=(grade in ('S', 'M')),  # 低等级AGV启用DP
        )
        self._active_agvs: Dict[str, Dict] = {}
        self._registered_clients: Dict[str, str] = {}  # client_id -> agv_id

    def register_agv(
        self,
        agv_id: str,
        capabilities: Dict,
    ) -> str:
        """注册AGV到联邦学习系统"""
        client_id = f"fl_client_{agv_id}"
        
        client = FederatedClient(
            client_id=client_id,
            agv_id=agv_id,
            model_config={
                'num_layers': 4,
                'gradient_shape': (128,),
            },
            local_epochs=5,
            batch_size=32,
        )
        
        self._server.register_client(client)
        self._active_agvs[agv_id] = {
            'client_id': client_id,
            'capabilities': capabilities,
            'registered_at': time.time(),
        }
        self._registered_clients[client_id] = agv_id
        
        return client_id

    def start_training_round(self) -> Optional[FLRoundResult]:
        """启动一轮训练"""
        selected = self._server.select_clients()
        if len(selected) < self._server.min_clients_per_round:
            logger.warning(f"可用客户端不足: {len(selected)} < {self._server.min_clients_per_round}")
            return None
        
        return self._server.execute_round(selected)

    def get_system_status(self) -> Dict:
        """获取系统状态"""
        summary = self._server.get_training_summary()
        
        return {
            'active_agvs': len(self._active_agvs),
            'registered_clients': len(self._server._clients),
            'current_round': summary.get('current_round', 0),
            'global_accuracy': summary.get('avg_accuracy', 0),
            'training_summary': summary,
        }


# 工厂函数
def create_federated_learning_system(
    num_agvs: int = 5,
    grade: str = "L",
) -> FederatedLearningCoordinator:
    """创建联邦学习系统"""
    model_config = {
        'num_layers': 4,
        'gradient_shape': (128,),
        'input_dim': 512,
        'output_dim': 10,
    }
    
    coordinator = FederatedLearningCoordinator(model_config, grade=grade)
    
    # 注册AGV
    for i in range(num_agvs):
        agv_id = f"AGV_{i+1:02d}"
        coordinator.register_agv(agv_id, {'grade': grade})
    
    return coordinator
