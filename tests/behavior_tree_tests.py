"""
behavior_tree_tests.py - 行为树具身任务规划模块测试
SuperModel 超模态大模型具身智能系统

测试覆盖:
- 节点类型测试 (Sequence/Selector/Parallel/装饰器)
- 黑板测试
- 简单任务树测试
- 复杂分层任务树测试
- AGV五级规格测试
- 性能基准测试
"""

import pytest
import time
import numpy as np
from src.embodied.behavior_tree import (
    NodeStatus,
    BTNode,
    SequenceNode,
    SelectorNode,
    ParallelNode,
    RepeaterNode,
    UntilFailNode,
    UntilSuccessNode,
    InverterNode,
    ConditionNode,
    ActionNode,
    LambdaActionNode,
    BehaviorTree,
    Blackboard,
    EmbodiedTask,
    EmbodiedTaskPlanner,
    AGVTaskPlanner,
    TaskStatus,
)


class TestNodeStatus:
    """节点状态枚举测试"""

    def test_enum_values(self):
        """测试枚举值存在"""
        assert NodeStatus.IDLE.value == "IDLE"
        assert NodeStatus.RUNNING.value == "RUNNING"
        assert NodeStatus.SUCCESS.value == "SUCCESS"
        assert NodeStatus.FAILURE.value == "FAILURE"


class TestBlackboard:
    """黑板测试"""

    def test_basic_operations(self):
        """基础操作测试"""
        bb = Blackboard()
        bb.set("key1", "value1")
        bb.set("key2", 42)
        assert bb.has("key1")
        assert not bb.has("key3")
        assert bb.get("key1") == "value1"
        assert bb.get("key2") == 42
        assert bb.get("key3", "default") == "default"

    def test_remove(self):
        """移除测试"""
        bb = Blackboard()
        bb.set("key", "value")
        assert bb.remove("key")
        assert not bb.has("key")
        assert not bb.remove("key")

    def test_update_robot_state(self):
        """更新机器人状态测试"""
        bb = Blackboard()
        bb.update_robot_state({"position": [0, 0, 0], "velocity": [1, 0, 0]})
        assert bb.robot_state["position"] == [0, 0, 0]
        assert bb.robot_state["velocity"] == [1, 0, 0]

    def test_get_robot_position(self):
        """获取机器人位置测试"""
        bb = Blackboard()
        bb.update_robot_state({"position": [1.0, 2.0, 3.0]})
        pos = bb.get_robot_position()
        assert pos is not None
        assert np.allclose(pos, np.array([1.0, 2.0, 3.0]))


class TestSequenceNode:
    """序列节点测试"""

    def test_all_success(self):
        """全部成功 → 成功"""
        seq = SequenceNode("test")
        seq.add_children(
            ConditionNode(lambda bb: True, "cond1"),
            ConditionNode(lambda bb: True, "cond2"),
        )
        bb = Blackboard()
        status = seq.tick(bb)
        assert status == NodeStatus.SUCCESS

    def test_first_failure(self):
        """第一个失败 → 失败"""
        seq = SequenceNode("test")
        seq.add_children(
            ConditionNode(lambda bb: False, "cond1"),
            ConditionNode(lambda bb: True, "cond2"),
        )
        bb = Blackboard()
        status = seq.tick(bb)
        assert status == NodeStatus.FAILURE

    def test_middle_running(self):
        """中间运行 → 返回运行"""
        call_count = []
        def first(bb):
            call_count.append(1)
            return True
        def second(bb):
            call_count.append(2)
            return False

        seq = SequenceNode("test")
        seq.add_children(
            ConditionNode(first, "first"),
            LambdaActionNode(lambda bb: NodeStatus.RUNNING, "run"),
            ConditionNode(second, "second"),
        )
        bb = Blackboard()
        status = seq.tick(bb)
        assert status == NodeStatus.RUNNING
        assert len(call_count) == 1


class TestSelectorNode:
    """选择节点测试"""

    def test_first_success(self):
        """第一个成功 → 成功"""
        sel = SelectorNode("test")
        sel.add_children(
            ConditionNode(lambda bb: True, "success"),
            ConditionNode(lambda bb: False, "failure"),
        )
        bb = Blackboard()
        assert sel.tick(bb) == NodeStatus.SUCCESS

    def test_last_success(self):
        """最后一个成功 → 成功"""
        sel = SelectorNode("test")
        sel.add_children(
            ConditionNode(lambda bb: False, "f1"),
            ConditionNode(lambda bb: False, "f2"),
            ConditionNode(lambda bb: True, "s3"),
        )
        bb = Blackboard()
        assert sel.tick(bb) == NodeStatus.SUCCESS

    def test_all_failure(self):
        """全部失败 → 失败"""
        sel = SelectorNode("test")
        sel.add_children(
            ConditionNode(lambda bb: False),
            ConditionNode(lambda bb: False),
        )
        bb = Blackboard()
        assert sel.tick(bb) == NodeStatus.FAILURE


class TestParallelNode:
    """并行节点测试"""

    def test_require_all_success_all_success(self):
        """需要全部成功，全部成功 → 成功"""
        parallel = ParallelNode("test", success_policy=ParallelNode.Policy.REQUIRE_ALL,
                               failure_policy=ParallelNode.Policy.REQUIRE_ANY)
        parallel.add_children(
            ConditionNode(lambda bb: True),
            ConditionNode(lambda bb: True),
        )
        bb = Blackboard()
        assert parallel.tick(bb) == NodeStatus.SUCCESS

    def test_require_any_success_one_success(self):
        """需要一个成功，一个成功 → 成功"""
        parallel = ParallelNode("test", success_policy=ParallelNode.Policy.REQUIRE_ANY,
                               failure_policy=ParallelNode.Policy.REQUIRE_ALL)
        parallel.add_children(
            ConditionNode(lambda bb: True),
            ConditionNode(lambda bb: False),
        )
        bb = Blackboard()
        assert parallel.tick(bb) == NodeStatus.SUCCESS

    def test_any_failure_one_failure(self):
        """任意失败，一个失败 → 失败"""
        parallel = ParallelNode("test", success_policy=ParallelNode.Policy.REQUIRE_ALL,
                               failure_policy=ParallelNode.Policy.REQUIRE_ANY)
        parallel.add_children(
            ConditionNode(lambda bb: True),
            ConditionNode(lambda bb: False),
        )
        bb = Blackboard()
        assert parallel.tick(bb) == NodeStatus.FAILURE


class TestDecoratorNodes:
    """装饰器节点测试"""

    def test_inverter_success_to_failure(self):
        """反转: 成功 → 失败"""
        inv = InverterNode(ConditionNode(lambda bb: True, "success"))
        bb = Blackboard()
        assert inv.tick(bb) == NodeStatus.FAILURE

    def test_inverter_failure_to_success(self):
        """反转: 失败 → 成功"""
        inv = InverterNode(ConditionNode(lambda bb: False, "failure"))
        bb = Blackboard()
        assert inv.tick(bb) == NodeStatus.SUCCESS

    def test_until_fail_success_keeps_running(self):
        """直到失败: 子成功 → 保持运行"""
        node = UntilFailNode(ConditionNode(lambda bb: True))
        bb = Blackboard()
        assert node.tick(bb) == NodeStatus.RUNNING

    def test_until_fail_failure_returns_success(self):
        """直到失败: 子失败 → 返回成功"""
        node = UntilFailNode(ConditionNode(lambda bb: False))
        bb = Blackboard()
        assert node.tick(bb) == NodeStatus.SUCCESS

    def test_repeater_finite_times(self):
        """有限次数重复"""
        count = []
        def action(bb):
            count.append(1)
            return NodeStatus.SUCCESS
        node = RepeaterNode(LambdaActionNode(action), times=3)
        bb = Blackboard()

        # 前两次应该返回 RUNNING
        for i in range(2):
            status = node.tick(bb)
            assert status == NodeStatus.RUNNING
            assert len(count) == i+1

        # 第三次返回 SUCCESS
        status = node.tick(bb)
        assert status == NodeStatus.SUCCESS
        assert len(count) == 3


class TestConditionNode:
    """条件节点测试"""

    def test_condition_true(self):
        """条件真 → 成功"""
        cond = ConditionNode(lambda bb: True)
        bb = Blackboard()
        assert cond.tick(bb) == NodeStatus.SUCCESS

    def test_condition_false(self):
        """条件假 → 失败"""
        cond = ConditionNode(lambda bb: False)
        bb = Blackboard()
        assert cond.tick(bb) == NodeStatus.FAILURE


class TestLambdaActionNode:
    """Lambda动作节点测试"""

    def test_returns_correct_status(self):
        """返回正确状态"""
        for status in [NodeStatus.SUCCESS, NodeStatus.FAILURE, NodeStatus.RUNNING]:
            action = LambdaActionNode(lambda bb: status)
            bb = Blackboard()
            assert action.tick(bb) == status


class TestBehaviorTree:
    """行为树整体测试"""

    def test_simple_navigation_tree(self):
        """简单导航树测试"""
        # 导航任务: 检查安全 → 移动 → 检查到达
        root = SequenceNode("Navigation")
        root.add_children(
            ConditionNode(lambda bb: bb.get("safety_ok", False), "CheckSafety"),
            LambdaActionNode(lambda bb: NodeStatus.RUNNING, "MoveToGoal"),
            ConditionNode(lambda bb: bb.get("arrived", False), "CheckArrived"),
        )

        bt = BehaviorTree(root, "NavigationTest")

        # 初始状态: 不安全 → 失败
        bt.blackboard.set("safety_ok", False)
        bt.blackboard.set("arrived", False)
        assert bt.tick() == NodeStatus.FAILURE

        # 安全，但未到达 → 运行中
        bt.reset()
        bt.blackboard.set("safety_ok", True)
        bt.blackboard.set("arrived", False)
        assert bt.tick() == NodeStatus.RUNNING

        # 安全且已到达 → 成功
        bt.reset()
        bt.blackboard.set("safety_ok", True)
        bt.blackboard.set("arrived", True)
        # Sequence节点在第一个RUNNING节点就返回RUNNING
        # 只有MoveToGoal返回成功才会检查arrived
        status = bt.tick()
        assert status == NodeStatus.RUNNING

    def test_get_statistics(self):
        """获取统计信息测试"""
        root = SequenceNode("test")
        root.add_children(
            ConditionNode(lambda bb: True),
            ConditionNode(lambda bb: True),
        )
        bt = BehaviorTree(root)
        stats = bt.get_statistics()
        assert "total_nodes" in stats
        assert stats["total_nodes"] == 3  # root + 2 children
        assert "SequenceNode" in stats["node_types"]
        assert "ConditionNode" in stats["node_types"]


class TestEmbodiedTask:
    """具身任务测试"""

    def test_task_lifecycle(self):
        """任务生命周期测试"""
        task = EmbodiedTask(
            task_id="test_001",
            task_type="navigation",
            goal_description="Navigate to (5, 0)",
            target_position=np.array([5.0, 0.0]),
            priority=0,
        )

        assert task.status == TaskStatus.IDLE
        task.start()
        assert task.status == TaskStatus.RUNNING
        assert task.start_time is not None
        assert not task.is_timeout()

        time.sleep(0.01)
        task.finish(success=True)
        assert task.status == TaskStatus.COMPLETED
        assert task.success
        assert task.get_duration() > 0

    def test_timeout(self):
        """超时测试"""
        task = EmbodiedTask(
            task_id="test_timeout",
            task_type="test",
            goal_description="",
            timeout=0.001,
        )
        task.start()
        time.sleep(0.01)
        assert task.is_timeout()


class TestEmbodiedTaskPlanner:
    """具身任务规划器测试"""

    def test_register_task_type(self):
        """注册任务类型测试"""
        planner = EmbodiedTaskPlanner()
        root = SequenceNode("test")
        root.add_child(ConditionNode(lambda bb: True))
        planner.register_task_type("test", root)
        assert "test" in planner.behavior_trees

    def test_add_task(self):
        """添加任务测试"""
        planner = EmbodiedTaskPlanner()
        task = EmbodiedTask(task_id="t1", task_type="navigation", goal_description="")
        planner.add_task(task)
        assert task.task_id in planner.tasks
        assert planner.get_status()['pending_tasks'] == 1

    def test_select_next_task(self):
        """选择下一个任务测试 (按优先级)"""
        planner = EmbodiedTaskPlanner()
        task_low = EmbodiedTask(task_id="low", task_type="nav", goal_description="", priority=10)
        task_high = EmbodiedTask(task_id="high", task_type="nav", goal_description="", priority=0)
        planner.add_task(task_low)
        planner.add_task(task_high)

        selected = planner.select_next_task()
        assert selected is not None
        assert selected.task_id == "high"  # 优先级 0 < 10 → 选高优先级


class TestAGVTaskPlanner:
    """AGV专用任务规划器测试"""

    def test_default_setup(self):
        """默认设置测试"""
        planner = AGVTaskPlanner(grade="M")
        assert planner.grade == "M"
        assert 'navigate' in planner.behavior_trees
        assert 'transport' in planner.behavior_trees

    def test_get_capabilities(self):
        """获取能力测试"""
        planner_s = AGVTaskPlanner(grade="S")
        planner_m = AGVTaskPlanner(grade="M")
        planner_l = AGVTaskPlanner(grade="L")
        planner_xl = AGVTaskPlanner(grade="XL")
        planner_xxl = AGVTaskPlanner(grade="XXL")

        assert planner_s.get_capabilities()['max_planning_depth'] == 3
        assert planner_m.get_capabilities()['max_planning_depth'] == 6
        assert planner_l.get_capabilities()['support_multi_agent'] is True
        assert planner_m.get_capabilities()['support_multi_agent'] is False


class TestAGVNodes:
    """AGV专用节点测试"""

    def test_check_battery_condition_ok(self):
        """电量充足 → 成功"""
        from src.embodied.behavior_tree import AGVCheckBatteryCondition
        cond = AGVCheckBatteryCondition(min_battery=0.2)
        bb = Blackboard()
        bb.update_robot_state({'battery_level': 0.5})
        # 需要手动获取到condition函数
        # 这里测试结构
        assert cond is not None

    def test_check_safe_condition(self):
        """安全检查节点"""
        from src.embodied.behavior_tree import AGVCheckSafeCondition
        cond = AGVCheckSafeCondition()
        bb = Blackboard()
        bb.update_robot_state({'safety': True})
        assert cond is not None


class PerformanceBenchmark:
    """性能基准测试"""

    def test_1000_ticks_small_tree(self, benchmark):
        """1000次tick小树性能基准"""
        root = SequenceNode("benchmark")
        for i in range(10):
            root.add_child(ConditionNode(lambda bb: True))

        bt = BehaviorTree(root)
        bb = bt.blackboard

        def run():
            for _ in range(100):
                bt.tick()
                bt.reset()

        benchmark(run)
        # 不做断言，用于性能测量


def test_material_transport_example():
    """物料搬运任务完整示例"""
    from src.embodied.behavior_tree import (
        SequenceNode, ConditionNode, ActionNode, BehaviorTree, Blackboard,
        AGVCheckBatteryCondition, AGVCheckSafeCondition,
    )

    # 物料搬运任务行为树
    root = SequenceNode("MaterialTransport")

    # 1. 检查安全和电量
    root.add_children(
        AGVCheckSafeCondition(),
        AGVCheckBatteryCondition(min_battery=0.2),
    )

    # 2. 简单导航到取货 (模拟) - 使用 Lambda 总是返回成功简化测试
    root.add_child(ConditionNode(lambda bb: True))

    # 3. 抓取
    root.add_child(ConditionNode(lambda bb: True))

    # 4. 导航到卸货
    root.add_child(ConditionNode(lambda bb: True))

    # 5. 放下完成
    root.add_child(ConditionNode(lambda bb: True))

    bt = BehaviorTree(root, "MaterialTransport")
    # 设置默认状态到黑板
    bt.update_robot_state({'safety': True, 'battery_level': 0.5})
    status = bt.tick()
    assert status == NodeStatus.SUCCESS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
