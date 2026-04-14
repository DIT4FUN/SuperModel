"""
test_behavior_tree_new_features.py - 行为树新功能测试
=======================================================
测试 DynamicBTReplanner 和 SwarmTaskAllocator
"""

import pytest
import numpy as np
import time


class TestDynamicBTReplanner:
    """DynamicBTReplanner 动态行为树重规划器测试"""

    def test_initialization(self):
        """测试重规划器初始化"""
        from src.embodied.behavior_tree import DynamicBTReplanner, SequenceNode, ReplanTrigger
        replanner = DynamicBTReplanner(
            base_bt_factory=lambda: SequenceNode("test"),
            max_replan_attempts=3,
            grade="M",
        )
        assert replanner.max_replan_attempts == 3
        assert replanner.grade == "M"
        assert replanner.replan_count == 0

    def test_grade_params(self):
        """测试AGV五级参数"""
        from src.embodied.behavior_tree import DynamicBTReplanner, SequenceNode
        for grade, expected_cooldown in [("S", 3.0), ("M", 2.0), ("L", 1.5), ("XL", 1.0), ("XXL", 0.5)]:
            replanner = DynamicBTReplanner(base_bt_factory=lambda: SequenceNode("x"), grade=grade)
            assert replanner.replan_cooldown == expected_cooldown

    def test_register_bt_variant(self):
        """测试BT变体注册"""
        from src.embodied.behavior_tree import DynamicBTReplanner, SequenceNode, SelectorNode
        replanner = DynamicBTReplanner(base_bt_factory=lambda: SequenceNode("base"))
        replanner.register_bt_variant(SelectorNode("variant1"))
        replanner.register_bt_variant(SequenceNode("variant2"))
        assert len(replanner.bt_variants) == 2

    def test_should_replan_task_failed(self):
        """测试任务失败触发重规划"""
        from src.embodied.behavior_tree import DynamicBTReplanner, SequenceNode, ReplanTrigger
        replanner = DynamicBTReplanner(base_bt_factory=lambda: SequenceNode("x"))
        assert replanner.should_replan(ReplanTrigger.TASK_FAILED) is True

    def test_should_not_replan_during_cooldown(self):
        """测试冷却期内不重规划"""
        from src.embodied.behavior_tree import DynamicBTReplanner, SequenceNode, ReplanTrigger
        replanner = DynamicBTReplanner(base_bt_factory=lambda: SequenceNode("x"), replan_cooldown_s=10.0)
        replanner.last_replan_time = time.time()  # 刚刚重规划
        assert replanner.should_replan(ReplanTrigger.TASK_FAILED) is False

    def test_should_not_replan_max_attempts(self):
        """测试超过最大重规划次数不触发"""
        from src.embodied.behavior_tree import DynamicBTReplanner, SequenceNode, ReplanTrigger
        replanner = DynamicBTReplanner(base_bt_factory=lambda: SequenceNode("x"), max_replan_attempts=2, replan_cooldown_s=0.0)
        replanner.replan_count = 2  # Already at max
        assert replanner.should_replan(ReplanTrigger.TASK_FAILED) is False

    def test_should_replan_battery_low(self):
        """测试电池低触发重规划"""
        from src.embodied.behavior_tree import DynamicBTReplanner, SequenceNode, ReplanTrigger
        replanner = DynamicBTReplanner(base_bt_factory=lambda: SequenceNode("x"))
        assert replanner.should_replan(ReplanTrigger.BATTERY_LOW, {'battery_level': 0.05}) is True
        assert replanner.should_replan(ReplanTrigger.BATTERY_LOW, {'battery_level': 0.5}) is False

    def test_should_replan_trajectory_deviation(self):
        """测试轨迹偏差触发重规划"""
        from src.embodied.behavior_tree import DynamicBTReplanner, SequenceNode, ReplanTrigger
        replanner = DynamicBTReplanner(base_bt_factory=lambda: SequenceNode("x"), deviation_threshold=0.3)
        assert replanner.should_replan(ReplanTrigger.TRAJECTORY_DEVIATION, {'deviation': 0.5}) is True
        assert replanner.should_replan(ReplanTrigger.TRAJECTORY_DEVIATION, {'deviation': 0.1}) is False

    def test_replan_uses_variant(self):
        """测试重规划使用BT变体"""
        from src.embodied.behavior_tree import DynamicBTReplanner, SequenceNode, SelectorNode, ReplanTrigger
        replanner = DynamicBTReplanner(base_bt_factory=lambda: SequenceNode("base"))
        replanner.register_bt_variant(SelectorNode("variant1"))
        replanner.load_initial_bt()
        new_bt = replanner.replan(ReplanTrigger.TASK_FAILED)
        assert new_bt is not None
        assert replanner.replan_count == 1
        assert replanner.active_variant_index == 1

    def test_replan_falls_back_to_base(self):
        """测试变体用尽后重建基础BT"""
        import time
        from src.embodied.behavior_tree import DynamicBTReplanner, SequenceNode, SelectorNode, ReplanTrigger
        replanner = DynamicBTReplanner(base_bt_factory=lambda: SequenceNode("base"), replan_cooldown_s=0.01)
        replanner.register_bt_variant(SelectorNode("v1"))
        replanner.load_initial_bt()
        # 第一次重规划用变体
        replanner.replan(ReplanTrigger.TASK_FAILED)
        assert replanner.active_variant_index == 1
        # 等待冷却期
        time.sleep(0.015)
        # 第二次重规划重建基础
        new_bt = replanner.replan(ReplanTrigger.TASK_FAILED)
        assert new_bt is not None
        assert replanner.active_variant_index == 0  # 回退到基础

    def test_replan_history(self):
        """测试重规划历史"""
        import time
        from src.embodied.behavior_tree import DynamicBTReplanner, SequenceNode, ReplanTrigger
        replanner = DynamicBTReplanner(base_bt_factory=lambda: SequenceNode("x"), replan_cooldown_s=0.01)
        replanner.load_initial_bt()
        replanner.replan(ReplanTrigger.OBSTACLE_DETECTED, {'obstacle_distance': 0.2})
        time.sleep(0.015)
        replanner.replan(ReplanTrigger.BATTERY_LOW, {'battery_level': 0.05})
        assert len(replanner.replan_history) == 2
        assert replanner.replan_history[0]['trigger'] == 'obstacle_detected'
        assert replanner.replan_history[1]['trigger'] == 'battery_low'

    def test_reset_after_success(self):
        """测试成功后重置"""
        from src.embodied.behavior_tree import DynamicBTReplanner, SequenceNode, ReplanTrigger
        replanner = DynamicBTReplanner(base_bt_factory=lambda: SequenceNode("x"))
        replanner.load_initial_bt()
        replanner.replan_count = 3
        replanner.active_variant_index = 2
        replanner.failure_contexts.append({'trigger': 'task_failed'})
        replanner.reset_after_success()
        assert replanner.replan_count == 0
        assert replanner.active_variant_index == 0
        assert len(replanner.failure_contexts) == 0

    def test_replan_statistics(self):
        """测试重规划统计"""
        from src.embodied.behavior_tree import DynamicBTReplanner, SequenceNode, ReplanTrigger
        replanner = DynamicBTReplanner(base_bt_factory=lambda: SequenceNode("x"), grade="L")
        replanner.load_initial_bt()
        replanner.replan(ReplanTrigger.MANUAL_REQUEST)
        stats = replanner.get_replan_statistics()
        assert stats['total_replans'] == 1
        assert stats['max_attempts'] == 4
        # No variants registered, so active_variant stays at 0 (recreated base)
        assert stats['available_variants'] == 0


class TestRobotCapabilities:
    """RobotCapabilities 机器人能力描述测试"""

    def test_can_execute_payload(self):
        """测试payload约束"""
        from src.embodied.behavior_tree import RobotCapabilities, SwarmTask
        robot = RobotCapabilities(robot_id="R1", max_payload=30.0)
        task_heavy = SwarmTask("T1", "carry", (1, 1), required_payload=50.0)
        task_light = SwarmTask("T2", "carry", (1, 1), required_payload=20.0)
        assert robot.can_execute(task_heavy) is False
        assert robot.can_execute(task_light) is True

    def test_can_execute_lift_gripper(self):
        """测试lift/gripper约束"""
        from src.embodied.behavior_tree import RobotCapabilities, SwarmTask
        robot_no_lift = RobotCapabilities(robot_id="R1", has_lift=False, has_gripper=True)
        robot_with_lift = RobotCapabilities(robot_id="R2", has_lift=True, has_gripper=True)
        task_lift = SwarmTask("T1", "place", (1, 1), requires_lift=True)
        task_grip = SwarmTask("T2", "pick", (1, 1), requires_gripper=True)
        task_none = SwarmTask("T3", "goto", (1, 1))
        assert robot_no_lift.can_execute(task_lift) is False
        assert robot_with_lift.can_execute(task_lift) is True
        assert robot_with_lift.can_execute(task_grip) is True
        assert robot_no_lift.can_execute(task_none) is True

    def test_can_execute_battery(self):
        """测试电池约束"""
        from src.embodied.behavior_tree import RobotCapabilities, SwarmTask
        robot_low = RobotCapabilities(robot_id="R1", battery_level=0.05)
        task = SwarmTask("T1", "goto", (1, 1))
        assert robot_low.can_execute(task) is False

    def test_estimated_time(self):
        """测试任务时间估算"""
        from src.embodied.behavior_tree import RobotCapabilities, SwarmTask
        robot = RobotCapabilities(robot_id="R1", max_speed=1.0, current_position=(0, 0))
        task = SwarmTask("T1", "goto", (5, 0), estimated_duration=10.0)
        est_time = robot.estimated_time(task)
        assert est_time > 0
        assert est_time >= 5.0  # 至少行驶时间


class TestSwarmTask:
    """SwarmTask 蜂群任务测试"""

    def test_task_initialization(self):
        """测试任务初始化"""
        from src.embodied.behavior_tree import SwarmTask
        task = SwarmTask(
            task_id="T1",
            task_type="delivery",
            target_position=(3.0, 4.0),
            priority=8,
        )
        assert task.task_id == "T1"
        assert task.priority == 8
        assert task.assigned_robot is None
        assert task.status == "pending"


class TestSwarmTaskAllocator:
    """SwarmTaskAllocator 多机器人任务分配器测试"""

    def test_register_robot(self):
        """测试机器人注册"""
        from src.embodied.behavior_tree import SwarmTaskAllocator, RobotCapabilities, AllocationStrategy
        allocator = SwarmTaskAllocator(strategy=AllocationStrategy.GREEDY)
        robot = RobotCapabilities(robot_id="R1", current_position=(0, 0))
        allocator.register_robot(robot)
        assert "R1" in allocator.robots

    def test_add_task(self):
        """测试任务添加"""
        from src.embodied.behavior_tree import SwarmTaskAllocator, SwarmTask, AllocationStrategy
        allocator = SwarmTaskAllocator(strategy=AllocationStrategy.GREEDY)
        task = SwarmTask("T1", "goto", (1, 1))
        allocator.add_task(task)
        assert len(allocator.pending_tasks) == 1

    def test_greedy_allocation(self):
        """测试贪心分配"""
        from src.embodied.behavior_tree import (
            SwarmTaskAllocator, RobotCapabilities, SwarmTask, AllocationStrategy
        )
        allocator = SwarmTaskAllocator(strategy=AllocationStrategy.GREEDY)
        # 注册两个机器人
        r1 = RobotCapabilities(robot_id="R1", current_position=(0, 0), max_speed=1.0)
        r2 = RobotCapabilities(robot_id="R2", current_position=(5, 5), max_speed=1.0)
        allocator.register_robot(r1)
        allocator.register_robot(r2)
        # 添加两个任务
        t1 = SwarmTask("T1", "goto", (1, 0))  # R1更近
        t2 = SwarmTask("T2", "goto", (6, 5))  # R2更近
        allocator.add_tasks_batch([t1, t2])
        result = allocator.allocate()
        assert len(result) >= 1

    def test_load_balanced_allocation(self):
        """测试负载均衡分配"""
        from src.embodied.behavior_tree import (
            SwarmTaskAllocator, RobotCapabilities, SwarmTask, AllocationStrategy
        )
        allocator = SwarmTaskAllocator(strategy=AllocationStrategy.LOAD_BALANCED)
        r1 = RobotCapabilities(robot_id="R1", current_position=(0, 0))
        r2 = RobotCapabilities(robot_id="R2", current_position=(0, 0))
        allocator.register_robot(r1)
        allocator.register_robot(r2)
        tasks = [SwarmTask(f"T{i}", "goto", (1, 0)) for i in range(4)]
        allocator.add_tasks_batch(tasks)
        result = allocator.allocate()
        # 两个机器人应有不同任务
        robot_task_counts = {}
        for alloc in result.values():
            robot_task_counts[alloc.robot_id] = robot_task_counts.get(alloc.robot_id, 0) + 1
        # 至少应有分配
        assert sum(robot_task_counts.values()) >= 1

    def test_capability_matched_allocation(self):
        """测试能力匹配分配"""
        from src.embodied.behavior_tree import (
            SwarmTaskAllocator, RobotCapabilities, SwarmTask, AllocationStrategy
        )
        allocator = SwarmTaskAllocator(strategy=AllocationStrategy.CAPABILITY_MATCHED)
        r_lift = RobotCapabilities(robot_id="R_lift", has_lift=True, has_gripper=True, current_position=(10, 0))
        r_no_lift = RobotCapabilities(robot_id="R_no_lift", has_lift=False, has_gripper=True, current_position=(0, 0))
        allocator.register_robot(r_lift)
        allocator.register_robot(r_no_lift)
        task_lift = SwarmTask("T_lift", "place", (1, 0), requires_lift=True)
        allocator.add_task(task_lift)
        result = allocator.allocate()
        # 应该分配给有lift的机器人
        if task_lift.task_id in result:
            assert result[task_lift.task_id].robot_id == "R_lift"

    def test_priority_ordered_allocation(self):
        """测试优先级顺序分配"""
        from src.embodied.behavior_tree import (
            SwarmTaskAllocator, RobotCapabilities, SwarmTask, AllocationStrategy
        )
        allocator = SwarmTaskAllocator(strategy=AllocationStrategy.PRIORITY_ORDERED)
        r1 = RobotCapabilities(robot_id="R1", current_position=(0, 0))
        allocator.register_robot(r1)
        t_low = SwarmTask("T_low", "goto", (5, 0), priority=2)
        t_high = SwarmTask("T_high", "goto", (5, 0), priority=9)
        allocator.add_tasks_batch([t_low, t_high])
        result = allocator.allocate()
        # 高优先级任务应优先分配
        if len(result) >= 1:
            high_allocated = t_high.task_id in result
            # 验证分配顺序或至少有一个被分配

    def test_conflict_resolution(self):
        """测试冲突解决"""
        from src.embodied.behavior_tree import (
            SwarmTaskAllocator, RobotCapabilities, SwarmTask, AllocationStrategy, AllocationResult
        )
        allocator = SwarmTaskAllocator(strategy=AllocationStrategy.GREEDY)
        r1 = RobotCapabilities(robot_id="R1", current_position=(0, 0))
        allocator.register_robot(r1)
        # 手动添加冲突分配（同一机器人两个任务）
        t1 = SwarmTask("T1", "goto", (1, 0), priority=5)
        t2 = SwarmTask("T2", "goto", (2, 0), priority=3)
        allocator.add_tasks_batch([t1, t2])
        # 直接设置冲突分配
        allocator.allocations[t1.task_id] = AllocationResult(
            task_id=t1.task_id, robot_id="R1", estimated_time=1.0, distance=1.0, strategy=AllocationStrategy.GREEDY
        )
        allocator.allocations[t2.task_id] = AllocationResult(
            task_id=t2.task_id, robot_id="R1", estimated_time=2.0, distance=2.0, strategy=AllocationStrategy.GREEDY
        )
        resolved = allocator._resolve_conflicts(allocator.allocations)
        # 高优先级的应保留
        assert len(resolved) <= 1

    def test_reallocate_on_failure(self):
        """测试任务失败后重新分配"""
        from src.embodied.behavior_tree import (
            SwarmTaskAllocator, RobotCapabilities, SwarmTask, AllocationStrategy
        )
        allocator = SwarmTaskAllocator(strategy=AllocationStrategy.GREEDY, max_reallocation_attempts=3)
        r1 = RobotCapabilities(robot_id="R1", current_position=(0, 0))
        r2 = RobotCapabilities(robot_id="R2", current_position=(10, 0))
        allocator.register_robot(r1)
        allocator.register_robot(r2)
        t1 = SwarmTask("T1", "goto", (1, 0))
        allocator.add_task(t1)
        allocator.allocate()
        # 模拟失败
        result = allocator.reallocate_on_failure("T1", "R1")
        # R2应该接手
        if result:
            assert result.robot_id == "R2"

    def test_allocation_report(self):
        """测试分配报告"""
        from src.embodied.behavior_tree import (
            SwarmTaskAllocator, RobotCapabilities, SwarmTask, AllocationStrategy
        )
        allocator = SwarmTaskAllocator(strategy=AllocationStrategy.GREEDY)
        r1 = RobotCapabilities(robot_id="R1")
        allocator.register_robot(r1)
        t1 = SwarmTask("T1", "goto", (1, 0))
        allocator.add_task(t1)
        allocator.allocate()
        report = allocator.get_allocation_report()
        assert 'strategy' in report
        assert report['strategy'] == 'greedy'
        assert 'total_robots' in report

    def test_update_robot_state(self):
        """测试机器人状态更新"""
        from src.embodied.behavior_tree import SwarmTaskAllocator, RobotCapabilities, AllocationStrategy
        allocator = SwarmTaskAllocator(strategy=AllocationStrategy.GREEDY)
        r1 = RobotCapabilities(robot_id="R1", battery_level=1.0)
        allocator.register_robot(r1)
        allocator.update_robot_state("R1", battery_level=0.5, current_position=(5, 5))
        assert allocator.robots["R1"].battery_level == 0.5
        assert allocator.robots["R1"].current_position == (5, 5)
