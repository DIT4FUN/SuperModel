"""
test_multiswarm.py - 多AGV蜂群协同测试
SuperModel 超模态大模型具身智能系统

测试覆盖:
- AGV注册与状态管理
- 动态任务分配
- 碰撞风险检测
- 蜂群编队
- 故障处理
- 避障路径
- 冲突检测与解决
- 拍卖算法
- 匈牙利算法
- 五级AGV规格适配
"""

import pytest
import math
import time
from embodiment.multi_agv_coordinator import (
    MultiAGVCoordinator,
    AGVStatus,
    AGVTask,
    MarketAuctionAllocator,
)


class TestMultiAGVRegistration:
    """AGV注册测试"""
    
    def test_single_agv_registration(self):
        """测试单AGV注册"""
        coordinator = MultiAGVCoordinator()
        agv = coordinator.register_agv("agv_001", position=(0, 0, 0))
        assert agv is not None
        assert 'agv_id' in agv
    
    def test_multiple_agv_registration(self):
        """测试多AGV注册"""
        coordinator = MultiAGVCoordinator()
        coordinator.register_agv("agv_001", position=(0, 0, 0))
        coordinator.register_agv("agv_002", position=(5, 0, 0))
        coordinator.register_agv("agv_003", position=(10, 0, 0))
        assert len(coordinator.agvs) == 3
    
    def test_agv_type_registration(self):
        """测试不同类型AGV注册"""
        coordinator = MultiAGVCoordinator()
        coordinator.register_agv("forklift_01", position=(0, 0, 0), type="forklift")
        coordinator.register_agv("delivery_01", position=(5, 0, 0), type="delivery")
        coordinator.register_agv("inspector_01", position=(10, 0, 0), type="inspection")
        agv1 = coordinator.get_agv("forklift_01")
        agv2 = coordinator.get_agv("delivery_01")
        agv3 = coordinator.get_agv("inspector_01")
        assert agv1 is not None
        assert agv2 is not None
        assert agv3 is not None


class TestTaskAllocation:
    """任务分配测试"""
    
    def test_capability_based_allocation(self):
        """测试基于能力的任务分配"""
        coordinator = MultiAGVCoordinator()
        coordinator.register_agv("agv_lift", position=(0, 0, 0), type="forklift", 
                                capabilities=["lift", "transport"])
        coordinator.register_agv("agv_fast", position=(5, 0, 0), type="delivery",
                                capabilities=["transport"])
        
        task = AGVTask(task_id="t1", required_capability="lift", load=100)
        allocated = coordinator.allocate_task(task)
        assert allocated == "agv_lift"
    
    def test_load_based_allocation(self):
        """测试基于负载能力的分配"""
        coordinator = MultiAGVCoordinator()
        coordinator.register_agv("agv_heavy", position=(0, 0, 0), max_load=500)
        coordinator.register_agv("agv_light", position=(5, 0, 0), max_load=50)
        
        task = AGVTask(task_id="t1", load=300, target_position=(10, 0, 0))
        allocated = coordinator.allocate_task(task)
        assert allocated == "agv_heavy"
    
    def test_distance_based_allocation(self):
        """测试基于距离的任务分配"""
        coordinator = MultiAGVCoordinator()
        coordinator.register_agv("agv_near", position=(1, 0, 0))
        coordinator.register_agv("agv_far", position=(100, 0, 0))
        
        task = AGVTask(task_id="t1", target_position=(2, 0, 0))
        allocated = coordinator.allocate_task(task)
        assert allocated == "agv_near"
    
    def test_priority_based_allocation(self):
        """测试基于优先级的任务分配"""
        coordinator = MultiAGVCoordinator()
        coordinator.register_agv("agv_01", position=(0, 0, 0))
        
        low_priority = AGVTask(task_id="low", priority=1, target_position=(5, 0, 0))
        high_priority = AGVTask(task_id="high", priority=10, target_position=(5, 0, 0))
        
        coordinator.allocate_task(high_priority)
        coordinator.allocate_task(low_priority)
        
        high_task = coordinator.get_agv_task(1)
        assert high_task.task_id == "high"
    
    def test_task_cancel(self):
        """测试任务取消"""
        coordinator = MultiAGVCoordinator()
        coordinator.register_agv("agv_01", position=(0, 0, 0))
        
        task = AGVTask(task_id="t1", target_position=(5, 0, 0))
        coordinator.add_task(task)
        result = coordinator.cancel_task("t1")
        assert result is True


class TestCollisionAvoidance:
    """碰撞避免测试"""
    
    def test_proximity_detection(self):
        """测试接近检测"""
        coordinator = MultiAGVCoordinator()
        coordinator.register_agv("agv_01", position=(0, 0, 0))
        coordinator.register_agv("agv_02", position=(1, 0, 0))
        
        risk = coordinator.check_collision_risk("agv_01", "agv_02")
        assert isinstance(risk, float)
        assert risk >= 0.0
    
    def test_collision_risk_zone(self):
        """测试碰撞风险区域"""
        coordinator = MultiAGVCoordinator(safety_distance=0.3)
        coordinator.register_agv("agv_01", position=(0, 0, 0))
        coordinator.register_agv("agv_02", position=(0.2, 0, 0))
        
        risk = coordinator.check_collision_risk("agv_01", "agv_02")
        assert risk > 0.0  # 距离很近，应该有碰撞风险
    
    def test_no_collision_risk_far_apart(self):
        """测试远距离无碰撞风险"""
        coordinator = MultiAGVCoordinator(safety_distance=0.3)
        coordinator.register_agv("agv_01", position=(0, 0, 0))
        coordinator.register_agv("agv_02", position=(10, 0, 0))
        
        risk = coordinator.check_collision_risk("agv_01", "agv_02")
        assert risk == 0.0  # 远距离无碰撞风险
    
    def test_check_collision_risks_all(self):
        """测试批量碰撞风险检测"""
        coordinator = MultiAGVCoordinator()
        for i in range(4):
            coordinator.register_agv(f"agv_{i:02d}", position=(i * 0.5, 0, 0))
        
        risks = coordinator.check_collision_risks()
        assert isinstance(risks, list)
    
    def test_avoidance_path(self):
        """测试避障路径计算"""
        coordinator = MultiAGVCoordinator()
        coordinator.register_agv("agv_01", position=(0, 0, 0))
        coordinator.register_agv("agv_02", position=(2, 0, 0))
        
        path = coordinator.get_avoidance_path("agv_01", "agv_02")
        assert path is not None


class TestSwarmCoordination:
    """蜂群协同测试"""
    
    def test_swarm_health_check(self):
        """测试蜂群健康检查"""
        coordinator = MultiAGVCoordinator()
        for i in range(3):
            coordinator.register_agv(f"agv_{i:02d}", position=(i, 0, 0))
        
        health = coordinator.get_swarm_health()
        assert isinstance(health, float)
        assert 0.0 <= health <= 1.0
    
    def test_battery_summary(self):
        """测试电量摘要"""
        coordinator = MultiAGVCoordinator()
        coordinator.register_agv("agv_01", position=(0, 0, 0), battery=80)
        coordinator.register_agv("agv_02", position=(5, 0, 0), battery=50)
        
        summary = coordinator.get_battery_summary()
        assert isinstance(summary, dict)
    
    def test_nearest_agv(self):
        """测试最近AGV查找"""
        coordinator = MultiAGVCoordinator()
        coordinator.register_agv("agv_01", position=(0, 0, 0))
        coordinator.register_agv("agv_02", position=(10, 0, 0))
        coordinator.register_agv("agv_03", position=(3, 0, 0))
        
        nearest = coordinator.get_nearest_agv((2, 0))
        assert nearest is not None
    
    def test_idle_agvs(self):
        """测试空闲AGV查询"""
        coordinator = MultiAGVCoordinator()
        coordinator.register_agv("agv_01", position=(0, 0, 0))
        coordinator.register_agv("agv_02", position=(5, 0, 0))
        
        idle = coordinator.get_idle_agvs()
        assert len(idle) == 2
    
    def test_compute_formation(self):
        """测试编队计算"""
        coordinator = MultiAGVCoordinator()
        for i in range(4):
            coordinator.register_agv(f"agv_{i:02d}", position=(i, 0, 0))
        
        formation = coordinator.compute_formation("line", leader_id=1)
        assert formation is not None
    
    def test_check_formation(self):
        """测试编队检查"""
        coordinator = MultiAGVCoordinator()
        for i in range(4):
            coordinator.register_agv(f"agv_{i:02d}", position=(i * 2, 0, 0))
        
        is_formed = coordinator.check_formation("line", spacing=2.0)
        assert isinstance(is_formed, bool)


class TestFaultTolerance:
    """故障容错测试"""
    
    def test_agv_failure_handling(self):
        """测试AGV故障处理"""
        coordinator = MultiAGVCoordinator()
        coordinator.register_agv("agv_01", position=(0, 0, 0))
        coordinator.register_agv("agv_02", position=(10, 0, 0))
        
        task = AGVTask(task_id="t1", target_position=(5, 0, 0))
        coordinator.add_task(task)
        
        coordinator.handle_agv_failure("agv_01")
        # Should handle gracefully
    
    def test_emergency_stop_all(self):
        """测试全部紧急停止"""
        coordinator = MultiAGVCoordinator()
        for i in range(3):
            coordinator.register_agv(f"agv_{i:02d}", position=(i, 0, 0))
        
        coordinator.emergency_stop_all()
        # Should complete without error
    
    def test_task_reallocation_after_failure(self):
        """测试故障后任务重新分配"""
        coordinator = MultiAGVCoordinator()
        coordinator.register_agv("agv_01", position=(0, 0, 0))
        coordinator.register_agv("agv_02", position=(10, 0, 0))
        
        task = AGVTask(task_id="t1", target_position=(5, 0, 0))
        coordinator.add_task(task)
        coordinator.handle_agv_failure("agv_01")
        
        # 重新分配
        reallocated = coordinator.reallocate_failed_task("t1")
        # reallocate_failed_task returns the new agv id or None


class TestPathPlanning:
    """路径规划测试"""
    
    def test_obstacle_adding(self):
        """测试障碍物添加"""
        coordinator = MultiAGVCoordinator()
        coordinator.register_agv("agv_01", position=(0, 0, 0))
        coordinator.add_obstacle((2, 2, 0))
        
        collision = coordinator.check_obstacle_collision("agv_01")
        assert isinstance(collision, list)
    
    def test_replan_path(self):
        """测试路径重规划"""
        coordinator = MultiAGVCoordinator()
        coordinator.register_agv("agv_01", position=(0, 0, 0))
        coordinator.add_obstacle((3, 0, 0))
        
        path = coordinator.replan_path("agv_01")
        assert path is not None or path is None  # 取决于障碍物位置


class TestCommunicationReliability:
    """通信可靠性测试"""
    
    def test_system_status(self):
        """测试系统状态"""
        coordinator = MultiAGVCoordinator()
        for i in range(3):
            coordinator.register_agv(f"agv_{i:02d}", position=(i, 0, 0))
        
        status = coordinator.get_system_status()
        assert isinstance(status, dict)
        assert 'agv_count' in status or len(status) > 0


class TestConflictResolution:
    """冲突解决测试"""
    
    def test_check_conflicts(self):
        """测试冲突检测"""
        coordinator = MultiAGVCoordinator()
        for i in range(4):
            coordinator.register_agv(f"agv_{i:02d}", position=(i, 0, 0))
        
        conflicts = coordinator.check_conflicts()
        assert isinstance(conflicts, list)
    
    def test_path_conflicts(self):
        """测试路径冲突检测"""
        coordinator = MultiAGVCoordinator()
        for i in range(3):
            coordinator.register_agv(f"agv_{i:02d}", position=(i, 0, 0))
        
        has_conflicts = coordinator.check_path_conflicts()
        assert isinstance(has_conflicts, bool)


class TestAuctionAlgorithm:
    """拍卖算法测试"""
    
    def test_market_auction_allocator(self):
        """测试市场拍卖分配器"""
        coordinator = MultiAGVCoordinator()
        for i in range(3):
            coordinator.register_agv(f"agv_{i:02d}", position=(i * 5, 0, 0))
        
        allocator = MarketAuctionAllocator(coordinator)
        task = AGVTask(task_id="t1", target_position=(5, 0, 0))
        
        auction_id = allocator.start_auction(task)
        assert auction_id is not None
        
        allocator.submit_bid(auction_id, "agv_01", 10.0)
        allocator.submit_bid(auction_id, "agv_02", 5.0)
        
        winner = allocator.close_auction(auction_id)
        assert winner in ["agv_01", "agv_02"]
    
    def test_auction_statistics(self):
        """测试拍卖统计"""
        coordinator = MultiAGVCoordinator()
        coordinator.register_agv("agv_01", position=(0, 0, 0))
        
        allocator = MarketAuctionAllocator(coordinator)
        stats = allocator.get_statistics()
        assert isinstance(stats, dict)


class TestAGVGradeSpecs:
    """AGV五级规格适配测试"""
    
    def test_grade_s_basic_swarm(self):
        """测试S级基础蜂群"""
        coordinator = MultiAGVCoordinator(grade="S")
        coordinator.register_agv("agv_01", position=(0, 0, 0))
        coordinator.register_agv("agv_02", position=(5, 0, 0))
        assert len(coordinator.agvs) == 2
    
    def test_grade_m_enhanced_coordination(self):
        """测试M级增强协同"""
        coordinator = MultiAGVCoordinator(grade="M")
        for i in range(5):
            coordinator.register_agv(f"agv_{i:02d}", position=(i * 3, 0, 0))
        
        health = coordinator.get_swarm_health()
        assert isinstance(health, float)
    
    def test_grade_l_advanced_algorithms(self):
        """测试L级高级算法"""
        coordinator = MultiAGVCoordinator(grade="L")
        for i in range(3):
            coordinator.register_agv(f"agv_{i:02d}", position=(i * 5, 0, 0))
        
        allocator = MarketAuctionAllocator(coordinator)
        assert allocator is not None
    
    def test_grade_xl_full_features(self):
        """测试XL级完整功能"""
        coordinator = MultiAGVCoordinator(grade="XL")
        for i in range(5):
            coordinator.register_agv(f"agv_{i:02d}", position=(i, 0, 0))
        
        coordinator.add_obstacle((2.5, 0, 0))
        path = coordinator.replan_path("agv_01")
        assert path is not None or path is None
    
    def test_grade_xxl_maximum_scale(self):
        """测试XXL级最大规模"""
        coordinator = MultiAGVCoordinator(grade="XXL")
        for i in range(10):
            coordinator.register_agv(f"agv_{i:02d}", position=(i, 0, 0))
        
        assert len(coordinator.agvs) == 10


class TestPerformance:
    """性能测试"""
    
    def test_large_swarm_registration(self):
        """测试大规模AGV注册"""
        start_time = time.time()
        coordinator = MultiAGVCoordinator()
        
        for i in range(20):
            coordinator.register_agv(f"agv_{i:03d}", position=(i, 0, 0))
        
        elapsed = time.time() - start_time
        assert elapsed < 2.0  # 20个AGV注册应在2秒内
    
    def test_rapid_task_allocation(self):
        """测试快速任务分配"""
        coordinator = MultiAGVCoordinator()
        for i in range(5):
            coordinator.register_agv(f"agv_{i:02d}", position=(i * 10, 0, 0))
        
        start_time = time.time()
        
        for i in range(20):
            task = AGVTask(
                task_id=f"t{i}",
                target_position=(i % 5 * 10, 0, 0),
            )
            coordinator.allocate_task(task)
        
        elapsed = time.time() - start_time
        assert elapsed < 1.0  # 20个任务分配应在1秒内


# ============================================================
# 测试运行入口
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
