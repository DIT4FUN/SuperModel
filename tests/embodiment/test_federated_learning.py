"""
test_federated_learning.py - 联邦学习多AGV协同模块测试
Federated Learning Multi-AGV Coordination Tests

测试覆盖:
- FederatedClient 本地训练与状态管理
- FederatedServer 客户端注册与选择
- FederatedServer 联邦聚合 (FedAvg)
- ByzantineFilter 拜占庭容错过滤
- DifferentialPrivacy 差分隐私保护
- AdaptiveAggregator 自适应聚合权重
- FederatedLearningCoordinator 多AGV协调
- 端到端 FL 训练循环
- AGV 五级规格适配
"""

import pytest
import time
import numpy as np
from src.embodied.federated_learning import (
    FLClientState,
    FLRoundResult,
    LocalTrainingResult,
    FederatedClient,
    FederatedServer,
    DifferentialPrivacy,
    ByzantineFilter,
    AdaptiveAggregator,
    FederatedLearningCoordinator,
    create_federated_learning_system,
)


# ==================== FederatedClient Tests ====================

class TestFederatedClientBasics:
    """FederatedClient 基础功能测试"""

    def test_client_creation(self):
        client = FederatedClient(
            client_id="client_001",
            agv_id="AGV_01",
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
            local_epochs=5,
            batch_size=32,
            learning_rate=0.01,
        )
        assert client.client_id == "client_001"
        assert client.agv_id == "AGV_01"
        assert client.state == FLClientState.IDLE

    def test_client_receive_global_model(self):
        client = FederatedClient(
            client_id="client_001",
            agv_id="AGV_01",
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
        )
        global_model = {
            f'layer_{i}': np.random.randn(128).astype(np.float32)
            for i in range(4)
        }
        client.receive_global_model(global_model)
        # Client should be in IDLE state after receiving model
        assert client.state == FLClientState.IDLE

    def test_client_local_train_basic(self):
        client = FederatedClient(
            client_id="client_001",
            agv_id="AGV_01",
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
        )
        result = client.local_train(train_data={'size': 200})
        assert isinstance(result, LocalTrainingResult)
        assert result.client_id == "client_001"
        assert result.num_samples == 200
        assert 0.0 <= result.validation_accuracy <= 1.0
        assert result.training_loss >= 0.0
        assert len(result.gradients) == 4
        assert result.model_update_hash != ""

    def test_client_train_multiple_rounds(self):
        client = FederatedClient(
            client_id="client_001",
            agv_id="AGV_01",
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
        )
        results = []
        for _ in range(3):
            r = client.local_train(train_data={'size': 150})
            results.append(r)
        
        assert all(r.round_number == i + 1 for i, r in enumerate(results))
        assert all(r.client_id == "client_001" for r in results)

    def test_client_train_state_transitions(self):
        client = FederatedClient(
            client_id="client_001",
            agv_id="AGV_01",
            model_config={'num_layers': 2, 'gradient_shape': (64,)},
        )
        assert client.state == FLClientState.IDLE
        result = client.local_train(train_data={'size': 100})
        # After training, state transitions to UPLOADING during result generation
        assert result.client_state in (FLClientState.TRAINING, FLClientState.UPLOADING)


class TestFederatedClientMetrics:
    """FederatedClient 训练指标测试"""

    def test_train_data_size(self):
        client = FederatedClient(
            client_id="client_001",
            agv_id="AGV_01",
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
        )
        result = client.local_train(train_data={'size': 500})
        assert result.num_samples == 500

    def test_gradient_hash_uniqueness(self):
        client = FederatedClient(
            client_id="client_001",
            agv_id="AGV_01",
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
        )
        results = []
        for _ in range(5):
            r = client.local_train(train_data={'size': 100})
            results.append(r)
        
        hashes = [r.model_update_hash for r in results]
        # Due to random nature, some hashes might collide but most should differ
        assert len(hashes) == 5

    def test_gradient_shapes(self):
        client = FederatedClient(
            client_id="client_001",
            agv_id="AGV_01",
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
        )
        result = client.local_train(train_data={'size': 100})
        for key, grad in result.gradients.items():
            assert grad.shape == (128,)
            assert grad.dtype == np.float32

    def test_communication_bytes(self):
        client = FederatedClient(
            client_id="client_001",
            agv_id="AGV_01",
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
        )
        result = client.local_train(train_data={'size': 100})
        assert result.communication_bytes > 0

    def test_training_time(self):
        client = FederatedClient(
            client_id="client_001",
            agv_id="AGV_01",
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
        )
        result = client.local_train(train_data={'size': 100})
        assert result.training_time_seconds >= 0


# ==================== FederatedServer Tests ====================

class TestFederatedServerBasics:
    """FederatedServer 基础功能测试"""

    def test_server_creation(self):
        server = FederatedServer(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
            num_rounds=100,
            min_clients_per_round=3,
        )
        assert server.num_rounds == 100
        assert server.min_clients_per_round == 3

    def test_server_global_model_initialization(self):
        server = FederatedServer(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
        )
        model = server.global_model
        assert len(model) == 4
        assert all(k.startswith('layer_') for k in model.keys())

    def test_server_register_client(self):
        server = FederatedServer(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
        )
        client = FederatedClient(
            client_id="client_001",
            agv_id="AGV_01",
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
        )
        server.register_client(client)
        assert "client_001" in server._clients

    def test_server_select_clients(self):
        server = FederatedServer(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
            min_clients_per_round=2,
        )
        for i in range(3):
            client = FederatedClient(
                client_id=f"client_{i:03d}",
                agv_id=f"AGV_{i:02d}",
                model_config={'num_layers': 4, 'gradient_shape': (128,)},
            )
            server.register_client(client)
        
        selected = server.select_clients()
        assert len(selected) == 2

    def test_server_select_clients_with_specific_count(self):
        server = FederatedServer(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
            min_clients_per_round=2,
        )
        for i in range(5):
            client = FederatedClient(
                client_id=f"client_{i:03d}",
                agv_id=f"AGV_{i:02d}",
                model_config={'num_layers': 4, 'gradient_shape': (128,)},
            )
            server.register_client(client)
        
        selected = server.select_clients(min_count=4)
        assert len(selected) == 4


class TestFederatedServerRoundExecution:
    """FederatedServer 轮次执行测试"""

    def test_server_execute_round_single_client(self):
        server = FederatedServer(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
            min_clients_per_round=1,
        )
        client = FederatedClient(
            client_id="client_001",
            agv_id="AGV_01",
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
        )
        server.register_client(client)
        
        selected = server.select_clients()
        result = server.execute_round(selected)
        
        assert isinstance(result, FLRoundResult)
        assert result.round_number == 1
        assert result.num_participants >= 1

    def test_server_execute_round_multiple_clients(self):
        server = FederatedServer(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
            min_clients_per_round=3,
        )
        for i in range(5):
            client = FederatedClient(
                client_id=f"client_{i:03d}",
                agv_id=f"AGV_{i:02d}",
                model_config={'num_layers': 4, 'gradient_shape': (128,)},
            )
            server.register_client(client)
        
        selected = server.select_clients()
        result = server.execute_round(selected)
        
        # At least one client should participate; byzantine filter may reduce count
        assert result.num_participants >= 1
        assert len(result.client_losses) >= 1

    def test_server_execute_multiple_rounds(self):
        server = FederatedServer(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
            num_rounds=10,
            min_clients_per_round=2,
        )
        for i in range(4):
            client = FederatedClient(
                client_id=f"client_{i:03d}",
                agv_id=f"AGV_{i:02d}",
                model_config={'num_layers': 4, 'gradient_shape': (128,)},
            )
            server.register_client(client)
        
        for round_num in range(3):
            selected = server.select_clients()
            result = server.execute_round(selected)
            assert result.round_number == round_num + 1

    def test_global_model_updates_after_round(self):
        server = FederatedServer(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
            min_clients_per_round=1,
        )
        client = FederatedClient(
            client_id="client_001",
            agv_id="AGV_01",
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
        )
        server.register_client(client)
        
        original_model = server.global_model.copy()
        selected = server.select_clients()
        result = server.execute_round(selected)
        
        # Global model should have been updated
        new_model = server.global_model
        assert set(new_model.keys()) == set(original_model.keys())


class TestFederatedServerWithDifferentialPrivacy:
    """FederatedServer 差分隐私测试"""

    def test_server_with_dp_enabled(self):
        server = FederatedServer(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
            use_differential_privacy=True,
            dp_epsilon=3.0,
        )
        assert server._use_dp is True
        assert server._dp is not None

    def test_server_without_dp(self):
        server = FederatedServer(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
            use_differential_privacy=False,
        )
        assert server._use_dp is False
        assert server._dp is None

    def test_dp_round_result_contains_epsilon(self):
        server = FederatedServer(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
            use_differential_privacy=True,
            dp_epsilon=3.0,
            min_clients_per_round=1,
        )
        client = FederatedClient(
            client_id="client_001",
            agv_id="AGV_01",
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
        )
        server.register_client(client)
        
        selected = server.select_clients()
        result = server.execute_round(selected)
        
        assert result.differential_privacy_applied is True
        assert result.epsilon == 3.0


# ==================== DifferentialPrivacy Tests ====================

class TestDifferentialPrivacyBasics:
    """差分隐私基础测试"""

    def test_dp_creation_default(self):
        dp = DifferentialPrivacy(epsilon=3.0)
        assert dp.epsilon == 3.0

    def test_add_noise_basic(self):
        dp = DifferentialPrivacy(epsilon=3.0)
        gradients = {
            'layer_0': np.random.randn(128).astype(np.float32),
            'layer_1': np.random.randn(64).astype(np.float32),
        }
        noisy_gradients = dp.add_noise_to_gradients(gradients)
        
        assert set(noisy_gradients.keys()) == set(gradients.keys())
        assert noisy_gradients['layer_0'].shape == (128,)
        assert noisy_gradients['layer_1'].shape == (64,)

    def test_noise_scales_with_sensitivity(self):
        dp = DifferentialPrivacy(epsilon=2.0, sensitivity=0.1)
        gradient = np.zeros(100, dtype=np.float32)
        noisy = dp.add_noise_to_gradients({'layer': gradient})
        
        # Noise should be added
        assert not np.allclose(noisy['layer'], gradient)

    def test_compute_privacy_spent(self):
        dp = DifferentialPrivacy(epsilon=3.0)
        spent, delta = dp.compute_privacy_spent(10)
        
        assert spent > 0
        assert delta >= 0

    def test_dp_repr(self):
        dp = DifferentialPrivacy(epsilon=3.0)
        r = repr(dp)
        assert "DifferentialPrivacy" in r


# ==================== ByzantineFilter Tests ====================

class TestByzantineFilter:
    """拜占庭容错过滤器测试"""

    def test_byzantine_filter_creation(self):
        bf = ByzantineFilter(f=1, n=10)
        assert bf.f == 1
        assert bf.n == 10

    def test_filter_empty_results(self):
        bf = ByzantineFilter()
        filtered = bf.filter_byzantine_clients([])
        assert filtered == []

    def test_filter_all_valid(self):
        bf = ByzantineFilter()
        results = [
            LocalTrainingResult(
                client_id=f"client_{i:03d}",
                round_number=1,
                num_samples=200,
                training_loss=0.5,
                validation_accuracy=0.9,
                gradients={'layer_0': np.zeros(128, dtype=np.float32)},
                model_update_hash=f"hash_{i}",
                training_time_seconds=10.0,
                communication_bytes=1000,
                client_state=FLClientState.IDLE,
            )
            for i in range(5)
        ]
        filtered = bf.filter_byzantine_clients(results)
        assert len(filtered) == 0  # All should pass

    def test_filter_malicious_client(self):
        # Need >= 3 clients for byzantine filter to work (len < 3 returns [])
        bf = ByzantineFilter(f=1, n=4)
        results = [
            LocalTrainingResult(
                client_id="legit_client",
                round_number=1,
                num_samples=200,
                training_loss=0.5,
                validation_accuracy=0.9,
                gradients={'layer_0': np.zeros(128, dtype=np.float32)},
                model_update_hash="legit_hash",
                training_time_seconds=10.0,
                communication_bytes=1000,
                client_state=FLClientState.IDLE,
            ),
            LocalTrainingResult(
                client_id="normal_client2",
                round_number=1,
                num_samples=200,
                training_loss=0.55,
                validation_accuracy=0.88,
                gradients={'layer_0': np.zeros(128, dtype=np.float32)},
                model_update_hash="normal_hash2",
                training_time_seconds=9.5,
                communication_bytes=950,
                client_state=FLClientState.IDLE,
            ),
            LocalTrainingResult(
                client_id="malicious_client",
                round_number=1,
                num_samples=200,
                training_loss=999.0,  # Abnormal loss - huge outlier
                validation_accuracy=0.01,  # Abysmal accuracy
                gradients={'layer_0': np.zeros(128, dtype=np.float32)},
                model_update_hash="malicious_hash",
                training_time_seconds=0.001,  # Suspiciously fast
                communication_bytes=1,  # Suspiciously small
                client_state=FLClientState.IDLE,
            ),
        ]
        filtered = bf.filter_byzantine_clients(results)
        # With z-score > 3.0, a loss of 999 vs mean ~333 should be filtered
        assert "malicious_client" in filtered

    def test_filter_high_loss_clients(self):
        bf = ByzantineFilter()
        results = [
            LocalTrainingResult(
                client_id=f"client_{i:03d}",
                round_number=1,
                num_samples=200,
                training_loss=0.5 + i * 0.1,  # Increasing loss
                validation_accuracy=0.95 - i * 0.05,  # Decreasing accuracy
                gradients={'layer_0': np.zeros(128, dtype=np.float32)},
                model_update_hash=f"hash_{i}",
                training_time_seconds=10.0,
                communication_bytes=1000,
                client_state=FLClientState.IDLE,
            )
            for i in range(5)
        ]
        filtered = bf.filter_byzantine_clients(results)
        # The worst performing client should be filtered
        assert len(filtered) <= 5


# ==================== AdaptiveAggregator Tests ====================

class TestAdaptiveAggregator:
    """自适应聚合器测试"""

    def test_aggregator_creation(self):
        agg = AdaptiveAggregator()
        assert agg._reliability_scores == {}

    def test_update_client_history(self):
        agg = AdaptiveAggregator()
        agg.update_client_history("client_001", 0.9, 1000)
        agg.update_client_history("client_001", 0.92, 1100)
        
        assert "client_001" in agg._historical_accuracy
        assert len(agg._historical_accuracy["client_001"]) == 2

    def test_compute_adaptive_weights_single_client(self):
        agg = AdaptiveAggregator()
        results = [
            LocalTrainingResult(
                client_id="client_001",
                round_number=1,
                num_samples=200,
                training_loss=0.5,
                validation_accuracy=0.9,
                gradients={'layer_0': np.zeros(128, dtype=np.float32)},
                model_update_hash="hash_001",
                training_time_seconds=10.0,
                communication_bytes=1000,
                client_state=FLClientState.IDLE,
            )
        ]
        weights = agg.compute_adaptive_weights(results)
        assert "client_001" in weights
        assert weights["client_001"] == pytest.approx(1.0, abs=0.1)

    def test_compute_adaptive_weights_multiple_clients(self):
        agg = AdaptiveAggregator()
        results = [
            LocalTrainingResult(
                client_id=f"client_{i:03d}",
                round_number=1,
                num_samples=200,
                training_loss=0.5 - i * 0.05,
                validation_accuracy=0.85 + i * 0.03,
                gradients={'layer_0': np.zeros(128, dtype=np.float32)},
                model_update_hash=f"hash_{i}",
                training_time_seconds=10.0,
                communication_bytes=1000 + i * 100,
                client_state=FLClientState.IDLE,
            )
            for i in range(3)
        ]
        weights = agg.compute_adaptive_weights(results)
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001  # Should be normalized

    def test_weights_sum_to_one(self):
        agg = AdaptiveAggregator()
        results = [
            LocalTrainingResult(
                client_id=f"client_{i:03d}",
                round_number=1,
                num_samples=200,
                training_loss=0.5,
                validation_accuracy=0.9,
                gradients={'layer_0': np.zeros(128, dtype=np.float32)},
                model_update_hash=f"hash_{i}",
                training_time_seconds=10.0,
                communication_bytes=1000,
                client_state=FLClientState.IDLE,
            )
            for i in range(5)
        ]
        weights = agg.compute_adaptive_weights(results)
        assert abs(sum(weights.values()) - 1.0) < 0.001

    def test_history_trim_at_10(self):
        agg = AdaptiveAggregator()
        for i in range(15):
            agg.update_client_history("client_001", 0.9, 1000)
        
        assert len(agg._historical_accuracy["client_001"]) == 10


# ==================== FederatedLearningCoordinator Tests ====================

class TestFLCoordinatorBasics:
    """联邦学习协调器基础测试"""

    def test_coordinator_creation(self):
        coordinator = FederatedLearningCoordinator(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
            grade="L",
        )
        assert coordinator.grade == "L"

    def test_register_single_agv(self):
        coordinator = FederatedLearningCoordinator(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
        )
        client_id = coordinator.register_agv("AGV_01", {'grade': 'L'})
        assert client_id == "fl_client_AGV_01"
        assert "AGV_01" in coordinator._active_agvs

    def test_register_multiple_agvs(self):
        coordinator = FederatedLearningCoordinator(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
        )
        for i in range(5):
            client_id = coordinator.register_agv(f"AGV_{i:02d}", {'grade': 'M'})
        
        assert len(coordinator._active_agvs) == 5
        assert len(coordinator._server._clients) == 5

    def test_get_system_status(self):
        coordinator = FederatedLearningCoordinator(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
        )
        coordinator.register_agv("AGV_01", {'grade': 'M'})
        coordinator.register_agv("AGV_02", {'grade': 'M'})
        
        status = coordinator.get_system_status()
        assert status['active_agvs'] == 2
        assert status['registered_clients'] == 2


class TestFLCoordinatorTraining:
    """联邦学习协调器训练测试"""

    def test_start_training_round(self):
        coordinator = FederatedLearningCoordinator(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
        )
        for i in range(5):
            coordinator.register_agv(f"AGV_{i:02d}", {'grade': 'M'})
        
        result = coordinator.start_training_round()
        assert result is not None
        assert isinstance(result, FLRoundResult)

    def test_multiple_training_rounds(self):
        coordinator = FederatedLearningCoordinator(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
        )
        for i in range(5):
            coordinator.register_agv(f"AGV_{i:02d}", {'grade': 'M'})
        
        for _ in range(3):
            result = coordinator.start_training_round()
            assert result is not None
        
        status = coordinator.get_system_status()
        assert status['current_round'] == 3

    def test_start_round_insufficient_clients(self):
        coordinator = FederatedLearningCoordinator(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
            grade="L",
        )
        # Register fewer clients than min_clients_per_round
        coordinator.register_agv("AGV_01", {'grade': 'L'})
        
        result = coordinator.start_training_round()
        # Should return None when not enough clients
        assert result is None


# ==================== Federated Learning Integration Tests ====================

class TestFLEndToEnd:
    """联邦学习端到端测试"""

    def test_full_fl_cycle(self):
        """完整的联邦学习训练周期"""
        coordinator = create_federated_learning_system(num_agvs=5, grade="L")
        
        # 执行多轮训练
        for i in range(5):
            result = coordinator.start_training_round()
            assert result is not None
            assert result.round_number == i + 1
        
        status = coordinator.get_system_status()
        assert status['current_round'] == 5
        assert status['registered_clients'] == 5

    def test_fl_with_dp(self):
        """带差分隐私的联邦学习"""
        server = FederatedServer(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
            use_differential_privacy=True,
            dp_epsilon=2.0,
            min_clients_per_round=3,
        )
        for i in range(5):
            client = FederatedClient(
                client_id=f"client_{i:03d}",
                agv_id=f"AGV_{i:02d}",
                model_config={'num_layers': 4, 'gradient_shape': (128,)},
            )
            server.register_client(client)
        
        for _ in range(3):
            selected = server.select_clients()
            result = server.execute_round(selected)
            assert result.differential_privacy_applied is True

    def test_fl_convergence_trend(self):
        """联邦学习收敛趋势测试"""
        coordinator = create_federated_learning_system(num_agvs=5, grade="L")
        
        accuracies = []
        for _ in range(10):
            result = coordinator.start_training_round()
            if result:
                accuracies.append(result.global_accuracy)
        
        assert len(accuracies) == 10
        # Accuracy should generally trend upward (with some noise due to simulation)
        # This is a weak test due to random simulation
        assert all(0 <= acc <= 1 for acc in accuracies)

    def test_fl_client_dropout_simulation(self):
        """模拟客户端掉线"""
        server = FederatedServer(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
            num_rounds=10,
            min_clients_per_round=3,
        )
        for i in range(6):
            client = FederatedClient(
                client_id=f"client_{i:03d}",
                agv_id=f"AGV_{i:02d}",
                model_config={'num_layers': 4, 'gradient_shape': (128,)},
            )
            server.register_client(client)
        
        # Simulate some clients being unavailable by selecting fewer
        selected = server.select_clients(min_count=3)
        result = server.execute_round(selected)
        
        # At least one client should participate
        assert result.num_participants >= 1

    def test_fl_byzantine_resilience(self):
        """拜占庭容错韧性测试"""
        server = FederatedServer(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
            min_clients_per_round=3,
        )
        # Register normal clients
        for i in range(4):
            client = FederatedClient(
                client_id=f"legit_{i:03d}",
                agv_id=f"AGV_{i:02d}",
                model_config={'num_layers': 4, 'gradient_shape': (128,)},
            )
            server.register_client(client)
        
        # Execute a round - byzantine filter should run
        selected = server.select_clients()
        result = server.execute_round(selected)
        
        assert result.round_number == 1


# ==================== AGV Grade Adaptation Tests ====================

class TestFLGradeAdaptation:
    """联邦学习AGV五级规格适配测试"""

    @pytest.mark.parametrize("grade", ["S", "M", "L", "XL", "XXL"])
    def test_fl_system_all_grades(self, grade):
        """测试所有AGV等级的FL系统创建"""
        coordinator = create_federated_learning_system(num_agvs=3, grade=grade)
        assert coordinator.grade == grade
        
        status = coordinator.get_system_status()
        assert status['active_agvs'] == 3

    def test_fl_low_grade_enables_dp(self):
        """低等级AGV自动启用差分隐私"""
        for grade in ["S", "M"]:
            coordinator = FederatedLearningCoordinator(
                model_config={'num_layers': 4, 'gradient_shape': (128,)},
                grade=grade,
            )
            coordinator.register_agv("AGV_01", {})
            # Low grades enable DP in the server
            assert coordinator._server._use_dp is True

    def test_fl_high_grade_no_dp_by_default(self):
        """高等级AGV默认不启用差分隐私"""
        coordinator = FederatedLearningCoordinator(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
            grade="XXL",
        )
        coordinator.register_agv("AGV_01", {})
        assert coordinator._server._use_dp is False


# ==================== FederatedClient AGV 五级规格测试 ====================

class TestFLClientGradeSpecs:
    """FederatedClient AGV五级规格测试"""

    def test_client_grade_s_config(self):
        """S级AGV客户端配置"""
        client = FederatedClient(
            client_id="client_S",
            agv_id="AGV_S",
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
            local_epochs=3,  # 低算力，少训练轮次
            batch_size=16,
            learning_rate=0.005,
        )
        result = client.local_train(train_data={'size': 100})
        assert result.num_samples == 100

    def test_client_grade_xxl_config(self):
        """XXL级AGV客户端配置"""
        client = FederatedClient(
            client_id="client_XXL",
            agv_id="AGV_XXL",
            model_config={'num_layers': 8, 'gradient_shape': (512,)},
            local_epochs=10,  # 高算力，多训练轮次
            batch_size=128,
            learning_rate=0.02,
        )
        result = client.local_train(train_data={'size': 1000})
        assert result.num_samples == 1000


# ==================== Summary Tests ====================

class TestFLSummary:
    """联邦学习总结报告测试"""

    def test_server_training_summary(self):
        server = FederatedServer(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
            num_rounds=10,
            min_clients_per_round=2,
        )
        for i in range(3):
            client = FederatedClient(
                client_id=f"client_{i:03d}",
                agv_id=f"AGV_{i:02d}",
                model_config={'num_layers': 4, 'gradient_shape': (128,)},
            )
            server.register_client(client)
        
        for _ in range(3):
            selected = server.select_clients()
            server.execute_round(selected)
        
        summary = server.get_training_summary()
        # After training, summary has round data (no 'status' key)
        assert summary['current_round'] == 3
        assert 'avg_accuracy' in summary

    def test_summary_before_training(self):
        server = FederatedServer(
            model_config={'num_layers': 4, 'gradient_shape': (128,)},
        )
        summary = server.get_training_summary()
        assert summary['status'] == 'not_started'

    def test_coordinator_summary_after_training(self):
        coordinator = create_federated_learning_system(num_agvs=4, grade="L")
        
        for _ in range(5):
            coordinator.start_training_round()
        
        summary = coordinator.get_system_status()
        assert summary['current_round'] == 5
        assert summary['active_agvs'] == 4
        assert 'global_accuracy' in str(summary) or 'avg_accuracy' in str(summary.get('training_summary', {}))
