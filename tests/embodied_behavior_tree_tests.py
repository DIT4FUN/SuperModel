"""
embodied_behavior_tree_tests.py - 行为树具身任务规划测试
SuperModel 超模态大模型具身智能系统

测试内容:
- 基础行为树节点功能
- AGV任务规划
- 层级任务测试
- 动态重规划测试
- 多任务调度测试
"""

import pytest
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
    AGVCheckBatteryCondition,
    AGVCheckSafeCondition,
    AGVCheckPositionReached,
    AGVMoveToAction,
)

class TestBehaviorTreeNodes:
    """行为树基础节点测试"""

    def test_sequence_node_all_success(self):
        """测试序列节点 - 全部成功"""
        seq = SequenceNode("TestSequence")
        seq.add_children(
            LambdaActionNode(lambda bb: NodeStatus.SUCCESS),
            LambdaActionNode(lambda bb: NodeStatus.SUCCESS),
            LambdaActionNode(lambda bb: NodeStatus.SUCCESS),
        )
        bb = Blackboard()
        status = seq.tick(bb)
        assert status == NodeStatus.SUCCESS

    def test_sequence_node_middle_failure(self):
        """测试序列节点 - 中间失败"""
        seq = SequenceNode("TestSequence")
        seq.add_children(
            LambdaActionNode(lambda bb: NodeStatus.SUCCESS),
            LambdaActionNode(lambda bb: NodeStatus.FAILURE),
            LambdaActionNode(lambda bb: NodeStatus.SUCCESS),
        )
        bb = Blackboard()
        status = seq.tick(bb)
        assert status == NodeStatus.FAILURE

    def test_selector_node_first_success(self):
        """测试选择节点 - 第一个成功"""
        sel = SelectorNode("TestSelector")
        sel.add_children(
            LambdaActionNode(lambda bb: NodeStatus.SUCCESS),
            LambdaActionNode(lambda bb: NodeStatus.FAILURE),
        )
        bb = Blackboard()
        status = sel.tick(bb)
        assert status == NodeStatus.SUCCESS

    def test_selector_node_all_fail(self):
        """测试选择节点 - 全部失败"""
        sel = SelectorNode("TestSelector")
        sel.add_children(
            LambdaActionNode(lambda bb: NodeStatus.FAILURE),
            LambdaActionNode(lambda bb: NodeStatus.FAILURE),
        )
        bb = Blackboard()
        status = sel.tick(bb)
        assert status == NodeStatus.FAILURE

    def test_selector_node_last_success(self):
        """测试选择节点 - 最后一个成功"""
        sel = SelectorNode("TestSelector")
        sel.add_children(
            LambdaActionNode(lambda bb: NodeStatus.FAILURE),
            LambdaActionNode(lambda bb: NodeStatus.FAILURE),
            LambdaActionNode(lambda bb: NodeStatus.SUCCESS),
        )
        bb = Blackboard()
        status = sel.tick(bb)
        assert status == NodeStatus.SUCCESS

    def test_inverter_node(self):
        """测试反转节点"""
        action_success = LambdaActionNode(lambda bb: NodeStatus.SUCCESS)
        inverter = InverterNode(action_success)
        bb = Blackboard()
        status = inverter.tick(bb)
        assert status == NodeStatus.FAILURE

        action_fail = LambdaActionNode(lambda bb: NodeStatus.FAILURE)
        inverter2 = InverterNode(action_fail)
        status2 = inverter2.tick(bb)
        assert status2 == NodeStatus.SUCCESS

    def test_until_success_node(self):
        """测试直到成功节点"""
        attempts = [NodeStatus.FAILURE, NodeStatus.FAILURE, NodeStatus.SUCCESS]
        current = [0]
        def action(bb):
            result = attempts[current[0]]
            current[0] += 1
            return result

        node = UntilSuccessNode(LambdaActionNode(action))
        bb = Blackboard()

        # 第一次尝试失败
        status1 = node.tick(bb)
        assert status1 == NodeStatus.RUNNING

        # 第二次尝试失败
        status2 = node.tick(bb)
        assert status2 == NodeStatus.RUNNING

        # 第三次尝试成功
        status3 = node.tick(bb)
        assert status3 == NodeStatus.SUCCESS

    def test_repeater_node_finite_times(self):
        """测试有限重复节点"""
        count = [0]
        def action(bb):
            count[0] += 1
            return NodeStatus.SUCCESS

        node = RepeaterNode(LambdaActionNode(action), times=3)
        bb = Blackboard()

        node.tick(bb)  # 1
        node.tick(bb)  # 2
        status = node.tick(bb)  # 3
        assert count[0] == 3
        assert status == NodeStatus.SUCCESS

    def test_parallel_node_require_all_success(self):
        """测试并行节点 - 需要全部成功"""
        parallel = ParallelNode(
            "TestParallel",
            success_policy=ParallelNode.Policy.REQUIRE_ALL,
            failure_policy=ParallelNode.Policy.REQUIRE_ANY
        )
        parallel.add_children(
            LambdaActionNode(lambda bb: NodeStatus.SUCCESS),
            LambdaActionNode(lambda bb: NodeStatus.SUCCESS),
        )
        bb = Blackboard()
        status = parallel.tick(bb)
        assert status == NodeStatus.SUCCESS

    def test_parallel_node_one_failure(self):
        """测试并行节点 - 一个失败即整体失败"""
        parallel = ParallelNode(
            "TestParallel",
            success_policy=ParallelNode.Policy.REQUIRE_ALL,
            failure_policy=ParallelNode.Policy.REQUIRE_ANY
        )
        parallel.add_children(
            LambdaActionNode(lambda bb: NodeStatus.SUCCESS),
            LambdaActionNode(lambda bb: NodeStatus.FAILURE),
        )
        bb = Blackboard()
        status = parallel.tick(bb)
        assert status == NodeStatus.FAILURE

    def test_condition_node_true(self):
        """条件节点测试 - 真"""
        cond = ConditionNode(lambda bb: True)
        bb = Blackboard()
        status = cond.tick(bb)
        assert status == NodeStatus.SUCCESS

    def test_condition_node_false(self):
        """条件节点测试 - 假"""
        cond = ConditionNode(lambda bb: False)
        bb = Blackboard()
        status = cond.tick(bb)
        assert status == NodeStatus.FAILURE

    def test_reset_node(self):
        """测试节点重置"""
        seq = SequenceNode("Test")
        seq.add_children(
            LambdaActionNode(lambda bb: NodeStatus.RUNNING),
        )
        bb = Blackboard()
        status1 = seq.tick(bb)
        assert status1 == NodeStatus.RUNNING
        seq.reset()
        assert seq.status == NodeStatus.IDLE


class TestBlackboard:
    """黑板测试"""

    def test_get_set_has(self):
        """测试基础存取"""
        bb = Blackboard()
        bb.set("key1", "value1")
        assert bb.has("key1")
        assert bb.get("key1") == "value1"
        assert bb.remove("key1")
        assert not bb.has("key1")

    def test_update_robot_state(self):
        """测试机器人状态更新"""
        bb = Blackboard()
        bb.update_robot_state({
            'position': [1.0, 2.0, 0.0],
            'battery_level': 0.8,
            'safety': True,
        })
        assert np.array_equal(bb.get_robot_position(), np.array([1.0, 2.0, 0.0]))
        assert bb.get_battery_level() == 0.8
        assert bb.is_safe() == True

    def test_get_robot_velocity(self):
        """测试速度获取"""
        bb = Blackboard()
        bb.update_robot_state({'velocity': [0.5, 0.0, 0.0]})
        assert np.array_equal(bb.get_robot_velocity(), np.array([0.5, 0.0, 0.0]))


class TestBehaviorTree:
    """整棵行为树测试"""

    def test_behavior_tree_navigation(self):
        """测试导航行为树"""
        from src.embodied.behavior_tree import AGVTaskPlanner
        planner = AGVTaskPlanner(grade="M")
        assert 'navigate' in planner.behavior_trees
        bt = planner.behavior_trees['navigate']
        assert bt.root is not None
        stats = bt.get_statistics()
        assert stats['total_nodes'] > 0

    def test_behavior_tree_statistics(self):
        """测试行为树统计"""
        root = SequenceNode("Root")
        root.add_children(
            ConditionNode(lambda bb: True),
            LambdaActionNode(lambda bb: NodeStatus.SUCCESS),
        )
        bt = BehaviorTree(root)
        stats = bt.get_statistics()
        assert stats['total_nodes'] == 3
        assert 'SequenceNode' in stats['node_types']
        assert 'ConditionNode' in stats['node_types']

    def test_behavior_tree_update_state(self):
        """测试行为树状态更新"""
        root = SequenceNode("Root")
        root.add_children(
            AGVCheckBatteryCondition(min_battery=0.2),
            AGVMoveToAction(),
        )
        bt = BehaviorTree(root)
        bt.update_robot_state({'position': [0.0, 0.0, 0.0], 'battery_level': 0.5})
        bt.set_goal({'target_position': [1.0, 0.0, 0.0]})
        status = bt.tick()
        # 应该还在运行，因为还没到达
        assert status == NodeStatus.RUNNING

    def test_behavior_tree_reset(self):
        """测试行为树重置"""
        root = SequenceNode("Root")
        root.add_children(
            AGVMoveToAction(),
        )
        bt = BehaviorTree(root)
        bt.update_robot_state({'position': [0, 0, 0]})
        bt.set_goal({'target_position': [1, 0, 0]})
        bt.tick()
        assert bt.is_running()
        bt.reset()
        assert not bt.is_running()
        assert bt.last_status == NodeStatus.IDLE


class TestEmbodiedTask:
    """具身任务测试"""

    def test_task_lifecycle(self):
        """测试任务生命周期"""
        task = EmbodiedTask(
            task_id="test_001",
            task_type="navigate",
            goal_description="Navigate to (10, 0)",
            target_position=np.array([10.0, 0.0, 0.0]),
            priority=0,
        )
        assert task.status == TaskStatus.IDLE
        task.start()
        assert task.status == TaskStatus.RUNNING
        assert task.start_time is not None
        assert not task.is_timeout()
        task.finish(success=True)
        assert task.status == TaskStatus.COMPLETED
        assert task.success == True
        assert task.get_duration() > 0

    def test_task_timeout(self):
        """测试任务超时"""
        import time
        task = EmbodiedTask(
            task_id="test_timeout",
            task_type="navigate",
            goal_description="Slow task",
            timeout=0.1,
        )
        task.start()
        time.sleep(0.2)
        assert task.is_timeout()


class TestEmbodiedTaskPlanner:
    """具身任务规划器测试"""

    def test_register_task_type(self):
        """测试注册任务类型"""
        planner = EmbodiedTaskPlanner()
        root = SequenceNode("TestTask")
        root.add_children(
            LambdaActionNode(lambda bb: NodeStatus.SUCCESS),
        )
        planner.register_task_type("test_task", root)
        assert "test_task" in planner.behavior_trees

    def test_add_and_select_task(self):
        """测试添加和选择任务"""
        planner = EmbodiedTaskPlanner()
        root = SequenceNode("Root")
        root.add_children(LambdaActionNode(lambda bb: NodeStatus.SUCCESS))
        planner.register_task_type("high_prio", root)

        task1 = EmbodiedTask(
            task_id="task1",
            task_type="high_prio",
            goal_description="High priority",
            priority=0,
        )
        task2 = EmbodiedTask(
            task_id="task2",
            task_type="high_prio",
            goal_description="Low priority",
            priority=1,
        )
        planner.add_task(task1)
        planner.add_task(task2)
        selected = planner.select_next_task()
        assert selected is not None
        assert selected.task_id == "task1"

    def test_task_execution_tick(self):
        """测试任务执行"""
        planner = EmbodiedTaskPlanner()
        # 创建一个总是成功的简单任务
        root = SequenceNode("SimpleTask")
        root.add_children(
            AGVCheckSafeCondition(),
            LambdaActionNode(lambda bb: NodeStatus.SUCCESS),
        )
        planner.register_task_type("simple", root)
        task = EmbodiedTask(
            task_id="simple_001",
            task_type="simple",
            goal_description="Simple test",
            priority=0,
        )
        planner.add_task(task)
        status = planner.tick(
            robot_state={'safety': True},
            world_state={}
        )
        assert status == NodeStatus.SUCCESS
        assert planner.status == TaskStatus.COMPLETED
        assert task.status == TaskStatus.COMPLETED

    def test_abort_current_task(self):
        """测试中止当前任务"""
        planner = EmbodiedTaskPlanner()
        root = SequenceNode("RunningTask")
        root.add_children(
            LambdaActionNode(lambda bb: NodeStatus.RUNNING),
        )
        planner.register_task_type("running", root)
        task = EmbodiedTask(
            task_id="running_001",
            task_type="running",
            goal_description="Running forever",
            priority=0,
        )
        planner.add_task(task)
        planner.tick({'safety': True}, {})
        assert planner.current_task is not None
        planner.abort_current()
        assert planner.current_task is None
        assert task.status == TaskStatus.ABORTED

    def test_get_status(self):
        """测试获取状态"""
        planner = AGVTaskPlanner(grade="M")
        status = planner.get_status()
        assert 'status' in status
        assert 'current_task' in status
        assert 'pending_tasks' in status
        assert 'plan_version' in status
        assert 'registered_types' in status
        assert 'navigate' in status['registered_types']
        assert 'transport' in status['registered_types']


class TestAGVSpecificNodes:
    """AGV特定节点测试"""

    def test_check_battery_condition_ok(self):
        """电量检查 - 充足"""
        cond = AGVCheckBatteryCondition(min_battery=0.2)
        bb = Blackboard()
        bb.update_robot_state({'battery_level': 0.5})
        status = cond.tick(bb)
        assert status == NodeStatus.SUCCESS

    def test_check_battery_condition_low(self):
        """电量检查 - 过低"""
        cond = AGVCheckBatteryCondition(min_battery=0.2)
        bb = Blackboard()
        bb.update_robot_state({'battery_level': 0.1})
        status = cond.tick(bb)
        assert status == NodeStatus.FAILURE

    def test_check_safe_condition(self):
        """安全检查"""
        cond = AGVCheckSafeCondition()
        bb = Blackboard()
        bb.update_robot_state({'safety': True})
        status = cond.tick(bb)
        assert status == NodeStatus.SUCCESS
        bb.update_robot_state({'safety': False})
        status2 = cond.tick(bb)
        assert status2 == NodeStatus.FAILURE

    def test_check_position_reached_yes(self):
        """位置检查 - 已到达"""
        cond = AGVCheckPositionReached(threshold=0.1)
        bb = Blackboard()
        bb.update_robot_state({'position': [1.0, 0.0, 0.0]})
        bb.goal_state['target_position'] = [1.05, 0.0, 0.0]
        status = cond.tick(bb)
        assert status == NodeStatus.SUCCESS

    def test_check_position_reached_no(self):
        """位置检查 - 未到达"""
        cond = AGVCheckPositionReached(threshold=0.1)
        bb = Blackboard()
        bb.update_robot_state({'position': [0.0, 0.0, 0.0]})
        bb.goal_state['target_position'] = [1.0, 0.0, 0.0]
        status = cond.tick(bb)
        assert status == NodeStatus.FAILURE

    def test_move_to_action_running(self):
        """移动动作 - 运行中"""
        action = AGVMoveToAction()
        bb = Blackboard()
        bb.update_robot_state({'position': [0.0, 0.0, 0.0]})
        bb.goal_state['target_position'] = [1.0, 0.0, 0.0]
        status = action.tick(bb)
        assert status == NodeStatus.RUNNING

    def test_agv_task_planner_capabilities(self):
        """测试AGV任务规划器能力检查"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            planner = AGVTaskPlanner(grade=grade)
            caps = planner.get_capabilities()
            assert caps['grade'] == grade
            assert 'max_planning_depth' in caps
            assert 'max_concurrent_tasks' in caps
            assert 'support_behavior_tree' in caps
        # XXL应该支持多AGV
        planner_xxl = AGVTaskPlanner(grade='XXL')
        assert planner_xxl.get_capabilities()['support_multi_agent'] == True

    def test_agv_transport_task(self):
        """测试AGV搬运任务"""
        planner = AGVTaskPlanner(grade='M')
        assert 'transport' in planner.behavior_trees
        bt = planner.behavior_trees['transport']
        # 检查行为树结构正确
        stats = bt.get_statistics()
        assert stats['total_nodes'] > 5
        assert 'SequenceNode' in stats['node_types']


class TestMultiAgentBehaviorTree:
    """多AGV协同行为树测试"""

    def test_multi_agent_coordination_sequence(self):
        """多AGV协调序列测试"""
        # 创建多AGV协调行为树
        # 序列: 检查安全 → 等待另一个AGV到达 → 移动 → 协同操作

        from src.embodied.behavior_tree import AGVCheckSafeCondition, AGVCheckPositionReached, AGVMoveToAction

        multi_seq = SequenceNode("MultiAGVCoordination")
        multi_seq.add_children(
            AGVCheckSafeCondition(),
            AGVCheckBatteryCondition(0.3),
            # 等待队友到达会合点
            # 然后移动到目标位置
            AGVMoveToAction(),
            AGVCheckPositionReached(0.15),
        )

        bt = BehaviorTree(multi_seq)
        bt.update_robot_state({
            'position': [0.0, 0.0, 0.0],
            'battery_level': 0.8,
            'safety': True,
        })
        bt.set_goal({'target_position': [5.0, 0.0, 0.0]})

        # 第一次tick 应该还在运行
        status1 = bt.tick()
        assert status1 == NodeStatus.RUNNING

        # 更新位置到目标附近
        bt.update_robot_state({'position': [5.05, 0.01, 0.0]})
        status2 = bt.tick()
        assert status2 == NodeStatus.SUCCESS


# 应该完成了


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
