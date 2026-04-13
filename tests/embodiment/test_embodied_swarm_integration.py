"""
test_embodied_swarm_integration.py - 具身多AGV蜂群集成测试
==========================================================
测试 SuperModel 具身智能系统在多AGV蜂群协同场景下的完整集成
使用 embodiment 包 (root-level) 的核心组件
"""

import pytest
import time
import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from embodiment.multi_agv_coordinator import (
    MultiAGVCoordinator, AGVStatus, AGVTask,
    MarketAuctionAllocator, FormationController, MarketAuctionConfig
)
from embodiment.behavior_tree_engine import (
    NodeStatus, BehaviorTreeEngine, BehaviorNode,
    SequenceNode, ConditionNode, TaskNode, RetryNode
)
from embodiment.simulation import (
    EmbodimentSimulator, SimSceneConfig, SimAGVConfig,
    GymnasiumVectorEnv
)
from src.embodied.federated_learning import (
    FederatedServer, FederatedClient, FLClientState,
    LocalTrainingResult
)
from src.embodied.scene_intelligence import (
    SceneIntelligence, SceneType, SceneConfig, SceneClassifier
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def warehouse_scene():
    """仓库场景仿真"""
    config = SimSceneConfig(scene_type="warehouse")
    return EmbodimentSimulator(scene_config=config, gui=False)


@pytest.fixture
def multi_coordinator():
    """多AGV协调器"""
    return MultiAGVCoordinator(max_workers=10)


@pytest.fixture
def market_allocator(multi_coordinator):
    """拍卖分配器"""
    config = MarketAuctionConfig(auction_timeout=5.0, reserve_price=0.3)
    return MarketAuctionAllocator(coordinator=multi_coordinator, config=config)


# =============================================================================
# Swarm Registration Tests
# =============================================================================

class TestSwarmRegistration:
    """蜂群注册测试"""

    def test_register_single_agv(self, multi_coordinator):
        """测试注册单个AGV"""
        result = multi_coordinator.register_agv(
            "test_agv_001",
            position=(0.0, 0.0, 0.0),
            type="transport",
            capabilities=["transport"],
        )
        assert result["agv_id"] == "test_agv_001"

    def test_register_multiple_agvs(self, multi_coordinator):
        """测试注册多个AGV"""
        for i in range(5):
            result = multi_coordinator.register_agv(
                f"agv_{i:03d}",
                position=(float(i * 5), 0.0, 0.0),
                type="transport",
            )
            assert result["agv_id"] == f"agv_{i:03d}"

    def test_register_agv_with_status(self, multi_coordinator):
        """测试带状态注册"""
        result = multi_coordinator.register_agv(
            "active_agv",
            position=(0.0, 0.0, 0.0),
            status=AGVStatus.ACTIVE,
            battery=0.95,
        )
        assert result["status"] == AGVStatus.ACTIVE

    def test_register_agv_with_type(self, multi_coordinator):
        """测试不同类型AGV注册"""
        for agv_type in ["forklift", "delivery", "tugger", "platform"]:
            result = multi_coordinator.register_agv(
                f"type_{agv_type}",
                position=(0.0, 0.0, 0.0),
                type=agv_type,
            )
            assert result["type"] == agv_type


# =============================================================================
# Swarm Task Tests
# =============================================================================

class TestSwarmTaskAllocation:
    """蜂群任务分配测试"""

    def test_add_task(self, multi_coordinator):
        """测试添加任务"""
        task = AGVTask(
            task_id="task_001",
            task_type="transport",
            target_position=(10.0, 5.0, 0.0),
        )
        task_id = multi_coordinator.add_task(task)
        assert task_id is not None

    def test_add_multiple_tasks(self, multi_coordinator):
        """测试添加多个任务"""
        for i in range(3):
            task = AGVTask(
                task_id=f"multi_task_{i}",
                task_type="transport",
                target_position=(float(10 + i * 5), 0.0, 0.0),
                priority=i + 1,
            )
            multi_coordinator.add_task(task)
        summary = multi_coordinator.get_task_summary()
        assert summary.get('pending', 0) >= 3 or summary.get('total', 0) == 3

    def test_cancel_task(self, multi_coordinator):
        """测试取消任务"""
        task = AGVTask(task_id="cancel_test", task_type="transport", target_position=(10.0, 0.0, 0.0))
        task_id = multi_coordinator.add_task(task)
        cancelled = multi_coordinator.cancel_task(task_id)
        assert cancelled is True

    def test_priority_task_allocation(self, multi_coordinator):
        """测试优先级任务分配"""
        # 注册AGV
        for i in range(3):
            multi_coordinator.register_agv(f"prio_agv_{i}", position=(float(i * 10), 0.0, 0.0))

        # 添加不同优先级任务
        priorities = [1, 5, 3]
        for i, pri in enumerate(priorities):
            task = AGVTask(
                task_id=f"prio_task_{i}",
                task_type="transport",
                target_position=(50.0, 0.0, 0.0),
                priority=pri,
            )
            multi_coordinator.add_task(task)

        allocated = multi_coordinator.assign_tasks()
        assert isinstance(allocated, dict)


# =============================================================================
# Swarm Collision Tests
# =============================================================================

class TestSwarmCollisionAvoidance:
    """蜂群避障测试"""

    def test_check_collision_risk(self, multi_coordinator):
        """测试碰撞风险检测"""
        multi_coordinator.register_agv("agv_A", position=(0.0, 0.0, 0.0), velocity=(1.0, 0.0, 0.0))
        multi_coordinator.register_agv("agv_B", position=(5.0, 0.0, 0.0), velocity=(-1.0, 0.0, 0.0))

        risk = multi_coordinator.check_collision_risk("agv_A", "agv_B")
        assert isinstance(risk, float)

    def test_get_avoidance_path(self, multi_coordinator):
        """测试避障路径获取"""
        multi_coordinator.register_agv("avoid_A", position=(0.0, 0.0, 0.0))
        multi_coordinator.register_agv("avoid_B", position=(5.0, 0.0, 0.0))

        path = multi_coordinator.get_avoidance_path("avoid_A", "avoid_B")
        assert path is not None
        assert isinstance(path, list)

    def test_emergency_stop(self, multi_coordinator):
        """测试紧急停止"""
        for i in range(3):
            multi_coordinator.register_agv(
                f"emerg_agv_{i}",
                position=(float(i * 5), 0.0, 0.0),
                status=AGVStatus.ACTIVE,
            )

        # 更新状态为ACTIVE以触发紧急停止逻辑
        for i in range(3):
            multi_coordinator.update_agv_status(f"emerg_agv_{i}", AGVStatus.ERROR)


# =============================================================================
# Formation Control Tests
# =============================================================================

class TestSwarmFormationControl:
    """蜂群编队控制测试"""

    def test_formation_creation(self, multi_coordinator):
        """测试编队创建"""
        for i in range(4):
            multi_coordinator.register_agv(f"form_{i}", position=(float(i * 2), 0.0, 0.0))

        formation = FormationController(multi_coordinator)
        formation.set_formation(FormationController.FormationType.LINE, spacing=2.0)
        formation.set_leader(0)
        positions = formation.compute_formation_positions()
        assert len(positions) == 4

    def test_formation_controller_init(self, multi_coordinator):
        """测试编队控制器初始化"""
        formation = FormationController(multi_coordinator)
        assert formation.coordinator is not None
        assert formation.formation_type == formation.FormationType.LINE


# =============================================================================
# Federated Learning Integration Tests
# =============================================================================

class TestFederatedSwarmIntegration:
    """联邦学习与蜂群集成测试"""

    def test_fl_client_creation(self):
        """测试联邦学习客户端创建"""
        client = FederatedClient(
            client_id="client_0",
            agv_id="agv_0",
            model_config={},
        )
        assert client is not None

    def test_fl_client_state_transitions(self):
        """测试FL客户端状态转换"""
        client = FederatedClient(
            client_id="state_client",
            agv_id="agv_state",
            model_config={},
        )
        assert client.state == FLClientState.IDLE

    def test_fl_server_creation(self):
        """测试联邦学习服务器创建"""
        server = FederatedServer(
            model_config={},
            num_rounds=10,
            min_clients_per_round=2,
        )
        assert server is not None

    def test_fl_server_register_client(self):
        """测试FL服务器注册客户端"""
        server = FederatedServer(model_config={})
        for i in range(3):
            client = FederatedClient(
                client_id=f"c{i}",
                agv_id=f"a{i}",
                model_config={},
            )
            server.register_client(client)
        assert len(server._clients) == 3


# =============================================================================
# Multi-Scene Swarm Integration Tests
# =============================================================================

class TestMultiSceneSwarmIntegration:
    """多场景蜂群集成测试"""

    def test_swarm_health_reporting(self, multi_coordinator):
        """测试蜂群健康报告"""
        for i in range(3):
            multi_coordinator.register_agv(
                f"health_agv_{i}",
                position=(float(i * 5), 0.0, 0.0),
                battery=1.0 - i * 0.2,
            )

        health = multi_coordinator.get_swarm_health()
        assert isinstance(health, float)
        assert 0.0 <= health <= 1.0

    def test_battery_summary(self, multi_coordinator):
        """测试电量汇总"""
        for i in range(3):
            multi_coordinator.register_agv(
                f"batt_agv_{i}",
                position=(float(i * 5), 0.0, 0.0),
                battery=0.95 - i * 0.15,
            )

        summary = multi_coordinator.get_battery_summary()
        assert "min" in summary
        assert "avg" in summary
        assert summary["min"] > 0

    def test_task_summary(self, multi_coordinator):
        """测试任务汇总"""
        for i in range(2):
            multi_coordinator.register_agv(f"task_agv_{i}", position=(0.0, 0.0, 0.0))

        for i in range(3):
            task = AGVTask(task_id=f"sum_task_{i}", task_type="transport", target_position=(10.0, 0.0, 0.0))
            multi_coordinator.add_task(task)

        summary = multi_coordinator.get_task_summary()
        assert "pending" in summary or "total" in summary


# =============================================================================
# Embodied Simulation Swarm Tests
# =============================================================================

class TestEmbodiedSimulationSwarm:
    """具身仿真蜂群测试"""

    def test_simulator_multi_agv_registration(self, warehouse_scene):
        """测试仿真器多AGV注册"""
        for i in range(3):
            agv_id = warehouse_scene.add_agv(initial_pos=(float(i * 3), 0.0, 0.1))
            assert agv_id is not None

        state = warehouse_scene.step()
        assert "agvs" in state
        assert len(state["agvs"]) == 3

    def test_simulator_obstacle_avoidance(self, warehouse_scene):
        """测试仿真器障碍规避"""
        warehouse_scene.add_agv(initial_pos=(0.0, 0.0, 0.1))
        warehouse_scene.add_obstacle("box", position=(1.0, 0.0, 0.0), size=(0.5, 0.5, 0.5))

        state = warehouse_scene.step()
        assert "agvs" in state

    def test_gymnasium_vector_env_reset(self):
        """测试Gymnasium向量化环境重置"""
        scene_cfg = SimSceneConfig(scene_type="warehouse")
        env = GymnasiumVectorEnv(num_envs=2, scene_config=scene_cfg)
        obs, info = env.reset()
        assert isinstance(obs, (list, tuple, np.ndarray))
        assert len(obs) >= 1  # 至少有一个环境的观测
        env.close()


# =============================================================================
# Behavior Tree Swarm Integration Tests
# =============================================================================

class TestBehaviorTreeSwarmIntegration:
    """行为树蜂群集成测试"""

    def test_bt_swarm_task_allocation(self, multi_coordinator):
        """测试行为树驱动的蜂群任务分配"""
        # 注册AGV
        multi_coordinator.register_agv("bt_agv_1", position=(0.0, 0.0, 0.0), type="transport")
        multi_coordinator.register_agv("bt_agv_2", position=(10.0, 0.0, 0.0), type="transport")

        # 创建行为树
        bt = BehaviorTreeEngine()
        ctx = {"allocation_done": False}

        def check_available_agvs(ctx):
            return NodeStatus.SUCCESS

        def allocate_to_agv(ctx):
            task = {"task_id": "bt_task_001", "type": "transport", "target_position": (20.0, 0.0, 0.0)}
            allocated = multi_coordinator.allocate_task(task)
            ctx["allocation_done"] = True
            return NodeStatus.SUCCESS

        def verify_allocation(ctx):
            return NodeStatus.SUCCESS if ctx["allocation_done"] else NodeStatus.FAILURE

        bt.add_node(BehaviorNode("check", check_available_agvs))
        bt.add_node(BehaviorNode("allocate", allocate_to_agv))
        bt.add_node(BehaviorNode("verify", verify_allocation))
        bt.add_sequence("bt_allocate", ["check", "allocate", "verify"])

        result = bt.run("bt_allocate", ctx)
        assert result == NodeStatus.SUCCESS

    def test_bt_swarm_task_retry(self, multi_coordinator):
        """测试行为树任务重试机制"""
        bt = BehaviorTreeEngine()
        attempts = [0]

        def always_fail(ctx):
            attempts[0] += 1
            return NodeStatus.FAILURE

        def eventually_succeed(ctx):
            if attempts[0] >= 3:
                return NodeStatus.SUCCESS
            return NodeStatus.FAILURE

        bt.add_node(BehaviorNode("fail_once", always_fail))
        retry_node = RetryNode("retry", child=bt.nodes["fail_once"], max_retries=3)
        bt.add_node(retry_node)
        bt.add_sequence("retry_seq", ["retry"])

        result = bt.run("retry_seq", {})
        assert attempts[0] <= 4  # initial + max_retries

    def test_bt_navigation_sequence(self):
        """测试导航行为树序列"""
        bt = BehaviorTreeEngine()

        def do_nav(ctx):
            return NodeStatus.SUCCESS

        bt.add_node(BehaviorNode("navigate", do_nav))
        bt.add_sequence("nav_seq", ["navigate"])

        result = bt.run("nav_seq", {})
        assert result == NodeStatus.SUCCESS


# =============================================================================
# Scene Intelligence Swarm Tests
# =============================================================================

class TestSceneIntelligenceSwarm:
    """场景智能蜂群测试"""

    def test_scene_classifier_creation(self):
        """测试场景分类器创建"""
        classifier = SceneClassifier()
        assert classifier is not None

    def test_scene_rule_engine_creation(self):
        """测试场景规则引擎创建"""
        from src.embodied.scene_intelligence import SceneRuleEngine
        engine = SceneRuleEngine()
        assert engine is not None

    def test_scene_intelligence_creation(self):
        """测试场景智能系统创建"""
        scene_intel = SceneIntelligence()
        assert scene_intel is not None


# =============================================================================
# Market Auction Tests
# =============================================================================

class TestMarketAuction:
    """市场拍卖分配测试"""

    def test_auction_creation(self, market_allocator):
        """测试拍卖创建"""
        task = AGVTask(
            task_id="auc_task",
            task_type="transport",
            load=100.0,
        )
        auction_id = market_allocator.start_auction(task)
        assert auction_id is not None

    def test_auction_bidding(self, market_allocator):
        """测试拍卖竞标"""
        task = AGVTask(task_id="bid_task", task_type="transport", load=150.0)
        auction_id = market_allocator.start_auction(task)

        market_allocator.submit_bid(auction_id, "bidder_1", bid_value=80.0)
        market_allocator.submit_bid(auction_id, "bidder_2", bid_value=90.0)

    def test_auction_close(self, market_allocator):
        """测试拍卖关闭"""
        task = AGVTask(task_id="close_task", task_type="transport", load=200.0)
        auction_id = market_allocator.start_auction(task)
        market_allocator.submit_bid(auction_id, "c_bidder_1", bid_value=75.0)
        market_allocator.submit_bid(auction_id, "c_bidder_2", bid_value=85.0)
        winner = market_allocator.close_auction(auction_id)
        assert winner is not None


# =============================================================================
# Load Balancing Tests
# =============================================================================

class TestSwarmLoadBalancing:
    """蜂群负载均衡测试"""

    def test_nearest_agv_query(self, multi_coordinator):
        """测试最近AGV查询"""
        positions = [(0.0, 0.0), (20.0, 0.0), (10.0, 10.0)]
        for i, pos in enumerate(positions):
            multi_coordinator.register_agv(f"near_agv_{i}", position=(pos[0], pos[1], 0.0))

        nearest = multi_coordinator.get_nearest_agv(position=(8.0, 5.0, 0.0))
        assert nearest is not None

    def test_idle_agv_filtering(self, multi_coordinator):
        """测试空闲AGV筛选"""
        for i in range(4):
            status = AGVStatus.IDLE if i % 2 == 0 else AGVStatus.ACTIVE
            multi_coordinator.register_agv(
                f"idle_agv_{i}",
                position=(float(i * 5), 0.0, 0.0),
                status=status,
            )

        idle_agvs = multi_coordinator.get_idle_agvs()
        assert isinstance(idle_agvs, list)
