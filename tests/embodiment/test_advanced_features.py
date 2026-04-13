"""
Advanced Embodiment Tests - 高级具身智能测试
测试扩展行为树节点、市场拍卖、编队控制、Gymnasium集成
"""

import pytest
import time
import math
import numpy as np
from embodiment.behavior_tree_engine import (
    NodeStatus, SequenceNode, ConditionNode, TaskNode,
    ParallelNode, StateMachineNode, RetryNode, TimeoutNode,
    InverterNode, AlwaysSuccessNode, AlwaysFailureNode,
    AGVTaskTrees, Node
)
from embodiment.multi_agv_coordinator import (
    MultiAGVCoordinator, AGVStatus, AGVTask,
    MarketAuctionAllocator, MarketAuctionConfig,
    FormationController
)
from embodiment.simulation import (
    EmbodimentSimulator, SimSceneConfig, SimAGVConfig,
    GymnasiumAGVEnv, SimulationScene
)


# =============================================================================
# Behavior Tree Extended Node Tests
# =============================================================================

class TestParallelNode:
    """测试并行节点"""
    
    def test_parallel_require_all_success(self):
        """REQUIRE_ALL策略：全部成功才成功"""
        seq1 = SequenceNode("seq1")
        seq1.add_child(TaskNode("s1_t1", lambda ctx: {"success": True}))
        seq1.add_child(TaskNode("s1_t2", lambda ctx: {"success": True}))
        
        seq2 = SequenceNode("seq2")
        seq2.add_child(TaskNode("s2_t1", lambda ctx: {"success": True}))
        seq2.add_child(TaskNode("s2_t2", lambda ctx: {"success": True}))
        
        parallel = ParallelNode("parallel", ParallelNode.Policy.REQUIRE_ALL)
        parallel.add_child(seq1)
        parallel.add_child(seq2)
        
        ctx = {}
        status = parallel.tick(ctx)
        assert status == NodeStatus.SUCCESS
    
    def test_parallel_require_all_failure(self):
        """REQUIRE_ALL策略：一个失败则整体失败"""
        seq1 = SequenceNode("seq1")
        seq1.add_child(TaskNode("s1_t1", lambda ctx: {"success": True}))
        seq1.add_child(TaskNode("s1_t2", lambda ctx: {"success": False}))
        
        seq2 = SequenceNode("seq2")
        seq2.add_child(TaskNode("s2_t1", lambda ctx: {"success": True}))
        seq2.add_child(TaskNode("s2_t2", lambda ctx: {"success": True}))
        
        parallel = ParallelNode("parallel", ParallelNode.Policy.REQUIRE_ALL)
        parallel.add_child(seq1)
        parallel.add_child(seq2)
        
        ctx = {}
        status = parallel.tick(ctx)
        assert status == NodeStatus.FAILURE
    
    def test_parallel_require_one_success(self):
        """REQUIRE_ONE策略：一个成功即成功（直接子节点中至少一个成功）"""
        # 使用独立的TaskNode，确保一个成功一个失败
        fail_node = TaskNode("always_fail", lambda ctx: {"success": False})
        success_node = TaskNode("always_success", lambda ctx: {"success": True})
        
        parallel = ParallelNode("parallel", ParallelNode.Policy.REQUIRE_ONE)
        parallel.add_child(fail_node)
        parallel.add_child(success_node)
        
        ctx = {}
        status = parallel.tick(ctx)
        assert status == NodeStatus.SUCCESS
        
        # 反过来也成立
        parallel2 = ParallelNode("parallel2", ParallelNode.Policy.REQUIRE_ONE)
        parallel2.add_child(TaskNode("t1", lambda ctx: {"success": True}))
        parallel2.add_child(TaskNode("t2", lambda ctx: {"success": True}))
        status2 = parallel2.tick(ctx)
        assert status2 == NodeStatus.SUCCESS
        
        # 全部失败应该返回FAILURE
        parallel3 = ParallelNode("parallel3", ParallelNode.Policy.REQUIRE_ONE)
        parallel3.add_child(TaskNode("t1", lambda ctx: {"success": False}))
        parallel3.add_child(TaskNode("t2", lambda ctx: {"success": False}))
        status3 = parallel3.tick(ctx)
        assert status3 == NodeStatus.FAILURE
    
    def test_parallel_require_majority(self):
        """REQUIRE_MAJORITY策略：多数成功才成功"""
        seq1 = SequenceNode("seq1")
        seq1.add_child(TaskNode("s1_t1", lambda ctx: {"success": True}))
        
        seq2 = SequenceNode("seq2")
        seq2.add_child(TaskNode("s2_t1", lambda ctx: {"success": False}))
        
        seq3 = SequenceNode("seq3")
        seq3.add_child(TaskNode("s3_t1", lambda ctx: {"success": True}))
        
        parallel = ParallelNode("parallel", ParallelNode.Policy.REQUIRE_MAJORITY)
        parallel.add_child(seq1)
        parallel.add_child(seq2)
        parallel.add_child(seq3)
        
        ctx = {}
        status = parallel.tick(ctx)
        # 2/3 成功，超过多数
        assert status == NodeStatus.SUCCESS


class TestStateMachineNode:
    """测试状态机节点"""
    
    def test_state_machine_basic(self):
        """基本状态转换"""
        sm = StateMachineNode("AGVStateMachine")
        
        # 定义状态
        idle_behavior = SequenceNode("IdleBehavior")
        idle_behavior.add_child(TaskNode("CheckBattery", lambda ctx: {"success": True}))
        idle_behavior.add_child(TaskNode("Wait", lambda ctx: {"success": True}))
        
        moving_behavior = SequenceNode("MovingBehavior")
        moving_behavior.add_child(TaskNode("Navigate", lambda ctx: {"success": True}))
        moving_behavior.add_child(TaskNode("Move", lambda ctx: {"success": True}))
        
        # 添加状态
        sm.add_state("IDLE", idle_behavior)
        sm.add_state("MOVING", moving_behavior)
        
        # 初始状态
        assert sm.get_current_state() == "IDLE"
        
        # 添加转换规则
        sm.add_transition("IDLE", "MOVING", lambda ctx: ctx.get("should_move", False))
        
        # 测试IDLE状态执行成功
        ctx = {"should_move": False}
        status = sm.tick(ctx)
        assert status == NodeStatus.SUCCESS
        assert sm.get_current_state() == "IDLE"
        
        # 测试状态转换
        ctx = {"should_move": True}
        sm.tick(ctx)
        assert sm.get_current_state() == "MOVING"
    
    def test_state_machine_force_set_state(self):
        """强制状态切换"""
        sm = StateMachineNode("TestSM")
        
        idle = TaskNode("IdleTask", lambda ctx: {"success": True})
        emergency = TaskNode("EmergencyTask", lambda ctx: {"success": True})
        
        sm.add_state("IDLE", idle)
        sm.add_state("EMERGENCY", emergency)
        
        assert sm.get_current_state() == "IDLE"
        
        sm.set_state("EMERGENCY")
        assert sm.get_current_state() == "EMERGENCY"


class TestRetryNode:
    """测试重试节点"""
    
    def test_retry_success_on_third_try(self):
        """第三次尝试才成功"""
        attempts = {"count": 0}
        
        def unreliable_task(ctx):
            attempts["count"] += 1
            return {"success": attempts["count"] >= 3}
        
        task = TaskNode("Unreliable", unreliable_task)
        retry = RetryNode("Retry3", task, max_retries=3)
        
        ctx = {}
        
        # 第一次：失败
        status = retry.tick(ctx)
        assert status == NodeStatus.RUNNING
        
        # 第二次：失败
        status = retry.tick(ctx)
        assert status == NodeStatus.RUNNING
        
        # 第三次：成功
        status = retry.tick(ctx)
        assert status == NodeStatus.SUCCESS
    
    def test_retry_success_after_retries(self):
        """重试节点在成功前经历多次失败"""
        attempts = {"count": 0}
        
        def eventually_succeeds(ctx):
            attempts["count"] += 1
            return {"success": attempts["count"] >= 2}
        
        task = TaskNode("EventuallySucceeds", eventually_succeeds)
        retry = RetryNode("Retry5", task, max_retries=5)
        
        ctx = {}
        
        # 第一次：失败但重试（RUNNING）
        status1 = retry.tick(ctx)
        assert status1 == NodeStatus.RUNNING
        assert attempts["count"] == 1
        
        # 第二次：成功
        status2 = retry.tick(ctx)
        assert status2 == NodeStatus.SUCCESS
        assert attempts["count"] == 2
        
        # 成功后重置计数器，再次执行会再次成功（因为task返回success）
        status3 = retry.tick(ctx)
        assert status3 == NodeStatus.SUCCESS
    
    def test_retry_failure_and_reset_cycle(self):
        """重试节点失败后重置计数器的行为"""
        attempts = {"count": 0}
        
        def always_fail(ctx):
            attempts["count"] += 1
            return {"success": False}
        
        task = TaskNode("AlwaysFail", always_fail)
        retry = RetryNode("Retry2", task, max_retries=2)
        
        ctx = {}
        
        # 连续失败直到FAILURE，然后重置继续重试
        # max_retries=2:
        # tick1: FAILURE->retry_count=1<2, return RUNNING
        # tick2: FAILURE->retry_count=2>=2, return FAILURE, reset retry_count=0
        # tick3: FAILURE->retry_count=1<2, return RUNNING
        statuses = []
        for _ in range(5):
            status = retry.tick(ctx)
            statuses.append(status)
        
        assert statuses[0] == NodeStatus.RUNNING   # tick1: retry_count=1
        assert statuses[1] == NodeStatus.FAILURE   # tick2: retry_count=2>=2, FAILURE+reset
        assert statuses[2] == NodeStatus.RUNNING   # tick3: new cycle, retry_count=1
        assert statuses[3] == NodeStatus.FAILURE   # tick4: retry_count=2>=2, FAILURE+reset
        assert statuses[4] == NodeStatus.RUNNING   # tick5: new cycle, retry_count=1


class TestModifierNodes:
    """测试修饰器节点"""
    
    def test_inverter_success_to_failure(self):
        """反转器：成功变失败"""
        child = TaskNode("AlwaysSuccess", lambda ctx: {"success": True})
        inverter = InverterNode("Inverter", child)
        
        ctx = {}
        status = inverter.tick(ctx)
        assert status == NodeStatus.FAILURE
    
    def test_inverter_failure_to_success(self):
        """反转器：失败变成功"""
        child = TaskNode("AlwaysFail", lambda ctx: {"success": False})
        inverter = InverterNode("Inverter", child)
        
        ctx = {}
        status = inverter.tick(ctx)
        assert status == NodeStatus.SUCCESS
    
    def test_always_success_node(self):
        """总是成功节点"""
        child = TaskNode("Whatever", lambda ctx: {"success": False})
        always_ok = AlwaysSuccessNode("AlwaysOK", child)
        
        ctx = {}
        status = always_ok.tick(ctx)
        assert status == NodeStatus.SUCCESS
    
    def test_always_failure_node(self):
        """总是失败节点"""
        child = TaskNode("Whatever", lambda ctx: {"success": True})
        always_fail = AlwaysFailureNode("AlwaysFail", child)
        
        ctx = {}
        status = always_fail.tick(ctx)
        assert status == NodeStatus.FAILURE


class TestAGVTaskTrees:
    """测试预建AGV任务树"""
    
    def test_build_patrol_tree(self):
        """巡逻任务树构建"""
        tree = AGVTaskTrees.build_patrol_tree()
        assert tree is not None
        assert tree.name == "PatrolLoop"
    
    def test_build_transport_tree(self):
        """运输任务树构建"""
        tree = AGVTaskTrees.build_transport_tree()
        assert tree is not None
        assert tree.name == "Transport"
    
    def test_build_emergency_tree(self):
        """应急任务树构建"""
        tree = AGVTaskTrees.build_emergency_tree()
        assert tree is not None
        assert tree.name == "EmergencyRoot"
    
    def test_patrol_tree_execution(self):
        """巡逻树执行"""
        tree = AGVTaskTrees.build_patrol_tree()
        ctx = {"battery": 0.5, "obstacles": [], "arrived": False}
        status = tree.tick(ctx)
        assert status == NodeStatus.SUCCESS


# =============================================================================
# Market Auction Tests
# =============================================================================

class TestMarketAuction:
    """测试市场拍卖分配器"""
    
    def test_auction_start_and_bid(self):
        """启动拍卖并出价"""
        coordinator = MultiAGVCoordinator()
        allocator = MarketAuctionAllocator(coordinator)
        
        task = AGVTask(task_id="task_001", task_type="transport")
        auction_id = allocator.start_auction(task)
        
        assert auction_id.startswith("auction_task_001")
        
        # 提交出价
        bid1 = allocator.submit_bid(auction_id, "agv_001", 10.0)
        assert bid1 == True
        
        bid2 = allocator.submit_bid(auction_id, "agv_002", 8.0)
        assert bid2 == True
    
    def test_auction_winner_lowest_bid(self):
        """拍卖winner是最低出价"""
        coordinator = MultiAGVCoordinator()
        allocator = MarketAuctionAllocator(coordinator)
        
        task = AGVTask(task_id="task_002", task_type="transport")
        auction_id = allocator.start_auction(task)
        
        allocator.submit_bid(auction_id, "agv_001", 15.0)
        allocator.submit_bid(auction_id, "agv_002", 7.0)
        allocator.submit_bid(auction_id, "agv_003", 12.0)
        
        winner = allocator.close_auction(auction_id)
        assert winner == "agv_002"  # 出价最低
    
    def test_auction_no_bids(self):
        """无出价时关闭拍卖"""
        coordinator = MultiAGVCoordinator()
        allocator = MarketAuctionAllocator(coordinator)
        
        task = AGVTask(task_id="task_003", task_type="transport")
        auction_id = allocator.start_auction(task)
        
        winner = allocator.close_auction(auction_id)
        assert winner is None
    
    def test_auction_timeout(self):
        """拍卖超时"""
        coordinator = MultiAGVCoordinator()
        config = MarketAuctionConfig(auction_timeout=0.01)  # 10ms超时
        allocator = MarketAuctionAllocator(coordinator, config)
        
        task = AGVTask(task_id="task_004", task_type="transport")
        auction_id = allocator.start_auction(task)
        
        # 等待超时
        time.sleep(0.05)
        
        bid_result = allocator.submit_bid(auction_id, "agv_001", 10.0)
        assert bid_result == False  # 超时后无法出价
    
    def test_auction_statistics(self):
        """拍卖统计"""
        coordinator = MultiAGVCoordinator()
        allocator = MarketAuctionAllocator(coordinator)
        
        stats = allocator.get_statistics()
        assert stats["total_auctions"] == 0
        
        # 创建一些拍卖
        for i in range(3):
            task = AGVTask(task_id=f"task_{i}", task_type="transport")
            auction_id = allocator.start_auction(task)
            allocator.submit_bid(auction_id, "agv_001", float(10 + i))
            allocator.close_auction(auction_id)
        
        stats = allocator.get_statistics()
        assert stats["completed_auctions"] == 3


# =============================================================================
# Formation Controller Tests
# =============================================================================

class TestFormationController:
    """测试编队控制器"""
    
    def test_formation_line(self):
        """直线编队"""
        coordinator = MultiAGVCoordinator()
        
        # 添加3个AGV
        coordinator.add_agv(0, position=(0.0, 0.0))
        coordinator.add_agv(1, position=(1.0, 0.0))
        coordinator.add_agv(2, position=(2.0, 0.0))
        
        controller = FormationController(coordinator)
        controller.set_formation(FormationController.FormationType.LINE, spacing=1.0)
        controller.set_leader(0)
        
        positions = controller.compute_formation_positions()
        
        assert 0 in positions
        assert 1 in positions
        assert 2 in positions
        
        # AGV1和AGV2应该在领队后方（y < 0）
        assert positions[1][1] < positions[0][1] or positions[2][1] < positions[0][1]
    
    def test_formation_rectangle(self):
        """矩形编队"""
        coordinator = MultiAGVCoordinator()
        
        for i in range(4):
            coordinator.add_agv(i, position=(float(i), 0.0))
        
        controller = FormationController(coordinator)
        controller.set_formation(FormationController.FormationType.RECTANGLE, spacing=1.5)
        controller.set_leader(0)
        
        positions = controller.compute_formation_positions()
        
        assert len(positions) == 4
        # 检查位置分布在不同象限
        x_coords = [p[0] for p in positions.values()]
        y_coords = [p[1] for p in positions.values()]
        assert max(x_coords) != min(x_coords)  # 有水平分布
        assert max(y_coords) != min(y_coords)  # 有垂直分布
    
    def test_formation_wedge(self):
        """楔形编队"""
        coordinator = MultiAGVCoordinator()
        
        for i in range(3):
            coordinator.add_agv(i, position=(float(i), 0.0))
        
        controller = FormationController(coordinator)
        controller.set_formation(FormationController.FormationType.WEDGE, spacing=1.0)
        controller.set_leader(0)
        
        positions = controller.compute_formation_positions()
        
        assert len(positions) == 3
        # 楔形编队：follower应该在领队后方，且有左右分布
    
    def test_formation_control_output(self):
        """编队控制输出"""
        coordinator = MultiAGVCoordinator()
        coordinator.add_agv(0, position=(0.0, 0.0))
        coordinator.add_agv(1, position=(2.0, 0.0))
        
        controller = FormationController(coordinator)
        controller.set_formation(FormationController.FormationType.LINE, spacing=1.0)
        controller.set_leader(0)
        
        controls = controller.maintain_formation()
        
        # 领队不应该有控制输出（由上层控制）
        assert 0 not in controls
        # follower应该有控制输出
        assert 1 in controls


# =============================================================================
# Gymnasium Integration Tests
# =============================================================================

class TestGymnasiumIntegration:
    """测试Gymnasium集成"""
    
    def test_gymnasium_env_creation(self):
        """Gymnasium环境创建"""
        try:
            env = GymnasiumAGVEnv(
                scene_config=SimSceneConfig(scene_type="warehouse"),
                num_agvs=1,
                max_steps=100
            )
            
            assert env is not None
            assert env.num_agvs == 1
            assert env.max_steps == 100
            assert env.observation_space is not None
            assert env.action_space is not None
            
            env.close()
        except RuntimeError as e:
            if "Gymnasium is not installed" in str(e):
                pytest.skip("Gymnasium not available")
            raise
    
    def test_gymnasium_env_reset(self):
        """Gymnasium环境重置"""
        try:
            env = GymnasiumAGVEnv(num_agvs=1, max_steps=100)
            
            obs, info = env.reset()
            
            assert obs is not None
            assert isinstance(obs, np.ndarray)
            assert len(obs) == 5  # x, y, theta, v, battery
            
            env.close()
        except RuntimeError as e:
            if "Gymnasium is not installed" in str(e):
                pytest.skip("Gymnasium not available")
            raise
    
    def test_gymnasium_env_step(self):
        """Gymnasium环境单步"""
        try:
            env = GymnasiumAGVEnv(num_agvs=1, max_steps=100)
            
            obs, info = env.reset()
            
            # 执行一步：零动作
            action = np.zeros(2, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            
            env.close()
        except RuntimeError as e:
            if "Gymnasium is not installed" in str(e):
                pytest.skip("Gymnasium not available")
            raise
    
    def test_gymnasium_env_observation_shape(self):
        """Gymnasium观测空间形状"""
        try:
            # 单AGV
            env1 = GymnasiumAGVEnv(num_agvs=1)
            assert env1.observation_space.shape == (5,)
            env1.close()
            
            # 多AGV
            env4 = GymnasiumAGVEnv(num_agvs=4)
            assert env4.observation_space.shape == (20,)  # 4 * 5
            env4.close()
        except RuntimeError as e:
            if "Gymnasium is not installed" in str(e):
                pytest.skip("Gymnasium not available")
            raise
    
    def test_gymnasium_env_max_steps(self):
        """Gymnasium环境最大步数限制"""
        try:
            env = GymnasiumAGVEnv(num_agvs=1, max_steps=10)
            
            obs, info = env.reset()
            terminated = False
            truncated = False
            steps = 0
            
            while not (terminated or truncated) and steps < 20:
                action = np.zeros(2, dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
            
            assert steps <= 10  # 应该在max_steps内终止
            assert truncated == True  # 被截断
            
            env.close()
        except RuntimeError as e:
            if "Gymnasium is not installed" in str(e):
                pytest.skip("Gymnasium not available")
            raise


# =============================================================================
# Embodied Simulation Enhanced Tests
# =============================================================================

class TestEmbodiedSimulationEnhanced:
    """测试增强仿真功能"""
    
    def test_scene_obstacle_detection(self):
        """场景障碍物检测"""
        scene_config = SimSceneConfig(
            scene_type="warehouse",
            obstacles=[(3.0, 3.0, 0.5), (5.0, -2.0, 0.3)]
        )
        sim = EmbodimentSimulator(scene_config=scene_config, gui=False)
        
        agv_id = sim.add_agv(initial_pos=(0.0, 0.0, 0.1))
        
        # 移动AGV向障碍物方向
        sim.set_agv_command(agv_id, v=0.5, omega=0.0)
        state = sim.step(duration=2.0)
        
        agv_state = state["agvs"][agv_id]["state"]
        
        # AGV应该记录了障碍物
        assert "obstacles" in agv_state
        
        sim.close()
    
    def test_multi_agv_sensor_simulation(self):
        """多AGV传感器仿真"""
        scene_config = SimSceneConfig(scene_type="logistics")
        sim = EmbodimentSimulator(scene_config=scene_config, gui=False)
        
        # 添加多个AGV
        agv1_id = sim.add_agv(initial_pos=(0.0, 0.0, 0.1))
        agv2_id = sim.add_agv(initial_pos=(5.0, 0.0, 0.1))
        
        state = sim.step(duration=1.0)
        
        # 两个AGV都应该有状态
        assert agv1_id in state["agvs"]
        assert agv2_id in state["agvs"]
        
        # 验证IMU传感器数据
        agv1_state = state["agvs"][agv1_id]
        assert "sensors" in agv1_state
        
        sim.close()
    
    def test_simulation_battery_consumption(self):
        """仿真电池消耗"""
        scene_config = SimSceneConfig(scene_type="warehouse")
        sim = EmbodimentSimulator(scene_config=scene_config, gui=False)
        
        agv_id = sim.add_agv(initial_pos=(0.0, 0.0, 0.1))
        
        initial_state = sim.step(duration=0.1)
        initial_battery = initial_state["agvs"][agv_id]["state"]["battery_level"]
        
        # 移动一段时间
        sim.set_agv_command(agv_id, v=1.0, omega=0.0)
        moved_state = sim.step(duration=5.0)
        moved_battery = moved_state["agvs"][agv_id]["state"]["battery_level"]
        
        # 电池应该消耗了
        assert moved_battery < initial_battery
        
        sim.close()
    
    def test_gripper_command(self):
        """夹爪指令"""
        scene_config = SimSceneConfig(scene_type="warehouse")
        sim = EmbodimentSimulator(scene_config=scene_config, gui=False)
        
        agv_id = sim.add_agv(initial_pos=(0.0, 0.0, 0.1))
        
        # 测试开夹爪
        sim.set_gripper_command(agv_id, "open")
        state1 = sim.step(duration=0.1)
        assert state1["agvs"][agv_id]["state"]["gripper_state"] == "open"
        
        # 测试闭夹爪
        sim.set_gripper_command(agv_id, "close")
        state2 = sim.step(duration=0.1)
        assert state2["agvs"][agv_id]["state"]["gripper_state"] == "close"
        
        sim.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
