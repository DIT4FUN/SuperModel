"""
behavior_tree_tests.py - 行为树具身任务规划模块测试
SuperModel 超模态大模型具身智能系统

测试覆盖:
- 基础节点功能 (Sequence/Selector/Parallel/Decorator)
- 条件节点和动作节点
- 行为树完整执行流程
- AGV专用任务规划器测试
- 具身任务调度测试
"""

import pytest
import numpy as np
import time
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
    AGVGraspAction,
    AGVReleaseAction,
)


class TestBlackboard:
    """黑板测试"""

    def test_get_set(self):
        """测试基本存取"""
        bb = Blackboard()
        bb.set('test_key', 'test_value')
        assert bb.get('test_key') == 'test_value'
        assert bb.has('test_key')
        assert bb.remove('test_key')
        assert not bb.has('test_key')

    def test_update_robot_state(self):
        """测试机器人状态更新"""
        bb = Blackboard()
        bb.update_robot_state({
            'position': [1.0, 2.0, 0.0],
            'velocity': [0.5, 0.0, 0.0],
            'battery_level': 0.8,
        })
        assert np.array_equal(bb.get_robot_position(), np.array([1.0, 2.0, 0.0]))
        assert np.array_equal(bb.get_robot_velocity(), np.array([0.5, 0.0, 0.0]))
        assert bb.get_battery_level() == 0.8

    def test_is_safe_default(self):
        """测试默认安全状态"""
        bb = Blackboard()
        assert bb.is_safe() is True


class TestSequenceNode:
    """序列节点测试"""

    def test_all_success(self):
        """所有子节点成功"""
        seq = SequenceNode("Test")
        seq.add_children(
            ConditionNode(lambda bb: True, "Cond1"),
            ConditionNode(lambda bb: True, "Cond2"),
        )
        bb = Blackboard()
        status = seq.tick(bb)
        assert status == NodeStatus.SUCCESS

    def test_first_failure(self):
        """第一个节点失败"""
        seq = SequenceNode("Test")
        seq.add_children(
            ConditionNode(lambda bb: False, "Cond1"),
            ConditionNode(lambda bb: True, "Cond2"),
        )
        bb = Blackboard()
        status = seq.tick(bb)
        assert status == NodeStatus.FAILURE

    def test_running_keeps_running(self):
        """有节点运行中保持运行状态"""
        class RunningAction(ActionNode):
            def execute(self, bb):
                return NodeStatus.RUNNING

        seq = SequenceNode("Test")
        seq.add_children(
            ConditionNode(lambda bb: True, "Cond1"),
            RunningAction("Run"),
            ConditionNode(lambda bb: True, "Cond2"),
        )
        bb = Blackboard()
        status = seq.tick(bb)
        assert status == NodeStatus.RUNNING


class TestSelectorNode:
    """选择节点测试"""

    def test_first_success(self):
        """第一个成功则整体成功"""
        sel = SelectorNode("Test")
        sel.add_children(
            ConditionNode(lambda bb: True, "Cond1"),
            ConditionNode(lambda bb: False, "Cond2"),
        )
        bb = Blackboard()
        status = sel.tick(bb)
        assert status == NodeStatus.SUCCESS

    def test_all_fail(self):
        """全部失败则整体失败"""
        sel = SelectorNode("Test")
        sel.add_children(
            ConditionNode(lambda bb: False, "Cond1"),
            ConditionNode(lambda bb: False, "Cond2"),
        )
        bb = Blackboard()
        status = sel.tick(bb)
        assert status == NodeStatus.FAILURE

    def test_one_running(self):
        """有节点运行中保持运行"""
        class RunningAction(ActionNode):
            def execute(self, bb):
                return NodeStatus.RUNNING

        sel = SelectorNode("Test")
        sel.add_children(
            ConditionNode(lambda bb: False, "Cond1"),
            RunningAction("Run"),
        )
        bb = Blackboard()
        status = sel.tick(bb)
        assert status == NodeStatus.RUNNING


class TestParallelNode:
    """并行节点测试"""

    def test_require_all_success_all_succeed(self):
        """所有成功要求下全部成功"""
        parallel = ParallelNode(
            "Test",
            success_policy=ParallelNode.Policy.REQUIRE_ALL,
            failure_policy=ParallelNode.Policy.REQUIRE_ANY
        )
        parallel.add_children(
            ConditionNode(lambda bb: True, "C1"),
            ConditionNode(lambda bb: True, "C2"),
        )
        bb = Blackboard()
        status = parallel.tick(bb)
        assert status == NodeStatus.SUCCESS

    def test_require_all_success_one_fail(self):
        """所有成功要求下一个失败立即失败"""
        parallel = ParallelNode(
            "Test",
            success_policy=ParallelNode.Policy.REQUIRE_ALL,
            failure_policy=ParallelNode.Policy.REQUIRE_ANY
        )
        parallel.add_children(
            ConditionNode(lambda bb: True, "C1"),
            ConditionNode(lambda bb: False, "C2"),
        )
        bb = Blackboard()
        status = parallel.tick(bb)
        assert status == NodeStatus.FAILURE


class TestDecoratorNodes:
    """装饰器节点测试"""

    def test_inverter(self):
        """反转装饰器测试"""
        cond_true = ConditionNode(lambda bb: True, "True")
        inv = InverterNode(cond_true)
        bb = Blackboard()
        assert inv.tick(bb) == NodeStatus.FAILURE

        cond_false = ConditionNode(lambda bb: False, "False")
        inv2 = InverterNode(cond_false)
        assert inv2.tick(bb) == NodeStatus.SUCCESS

    def test_until_success(self):
        """直到成功装饰器测试"""
        attempts = []
        def cond(bb):
            attempts.append(1)
            return len(attempts) >= 3

        node = ConditionNode(cond, "Cond")
        until = UntilSuccessNode(node)
        bb = Blackboard()

        # 前两次失败，返回运行中
        assert until.tick(bb) == NodeStatus.RUNNING
        assert until.tick(bb) == NodeStatus.RUNNING
        # 第三次成功
        assert until.tick(bb) == NodeStatus.SUCCESS

    def test_repeater有限次数(self):
        """重复装饰器有限次数测试"""
        count = 0
        def action(bb):
            nonlocal count
            count += 1
            return NodeStatus.SUCCESS

        node = LambdaActionNode(action)
        repeater = RepeaterNode(node, times=3)
        bb = Blackboard()

        for _ in range(2):
            assert repeater.tick(bb) == NodeStatus.RUNNING
        assert repeater.tick(bb) == NodeStatus.SUCCESS
        assert count == 3


class TestBehaviorTree:
    """行为树整体测试"""

    def test_simple_navigation_tree(self):
        """简单导航树测试"""
        # 导航序列: 检查安全 -> 检查电量 -> 移动 -> 检查到达
        root = SequenceNode("Navigation")
        root.add_children(
            AGVCheckSafeCondition(),
            AGVCheckBatteryCondition(min_battery=0.2),
            AGVMoveToAction(),
            AGVCheckPositionReached(threshold=0.1),
        )

        bt = BehaviorTree(root)
        bt.update_robot_state({
            'position': [0.0, 0.0, 0.0],
            'battery_level': 0.8,
            'safety': True,
        })
        bt.set_goal({'target_position': [1.0, 0.0, 0.0]})

        # 还没到达，保持运行
        status = bt.tick()
        assert status == NodeStatus.RUNNING

        # 更新位置到目标附近
        bt.update_robot_state({'position': [0.95, 0.0, 0.0]})
        status = bt.tick()
        assert status == NodeStatus.SUCCESS

    def test_statistics(self):
        """测试统计信息"""
        root = SequenceNode("Root")
        root.add_children(
            ConditionNode(lambda bb: True),
            SelectorNode("Sel").add_children(
                ConditionNode(lambda bb: False),
                ConditionNode(lambda bb: True),
            )
        )
        bt = BehaviorTree(root)
        stats = bt.get_statistics()
        assert stats['total_nodes'] == 5
        assert 'SequenceNode' in stats['node_types']
        assert 'SelectorNode' in stats['node_types']
        assert 'ConditionNode' in stats['node_types']


class TestEmbodiedTask:
    """具身任务定义测试"""

    def test_task_lifecycle(self):
        """任务生命周期测试"""
        task = EmbodiedTask(
            task_id="task_001",
            task_type="navigate",
            goal_description="Navigate to position (10, 5)",
            target_position=np.array([10.0, 5.0, 0.0]),
            priority=0,
            timeout=300.0,
        )

        assert task.status == TaskStatus.IDLE
        assert not task.is_timeout()

        task.start()
        assert task.status == TaskStatus.RUNNING
        assert task.start_time is not None

        task.finish(success=True)
        assert task.status == TaskStatus.COMPLETED
        assert task.success is True
        assert task.end_time is not None
        assert task.get_duration() > 0


class TestEmbodiedTaskPlanner:
    """具身任务规划器测试"""

    def test_register_task_type(self):
        """测试注册任务类型"""
        planner = EmbodiedTaskPlanner()
        root = SequenceNode("TestTask")
        root.add_children(
            AGVCheckSafeCondition(),
            AGVCheckBatteryCondition(),
        )
        planner.register_task_type("test", root)
        assert "test" in planner.behavior_trees

    def test_add_remove_task(self):
        """测试添加移除任务"""
        planner = EmbodiedTaskPlanner()
        task = EmbodiedTask(
            task_id="t1",
            task_type="navigate",
            goal_description="Test"
        )
        planner.add_task(task)
        assert "t1" in planner.tasks
        assert planner.remove_task("t1")
        assert "t1" not in planner.tasks

    def test_priority_selection(self):
        """测试优先级选择"""
        planner = EmbodiedTaskPlanner()

        # 注册一个简单任务
        root = SequenceNode("Test")
        root.add_children(ConditionNode(lambda bb: True))
        planner.register_task_type("navigate", root)

        # 添加不同优先级任务
        task1 = EmbodiedTask("t1", "navigate", "Low priority", priority=10)
        task2 = EmbodiedTask("t2", "navigate", "High priority", priority=0)
        task3 = EmbodiedTask("t3", "navigate", "Medium priority", priority=5)

        planner.add_task(task1)
        planner.add_task(task2)
        planner.add_task(task3)

        # 应该选优先级最高的 (数字最小)
        selected = planner.select_next_task()
        assert selected.task_id == "t2"

    def test_tick_workflow(self):
        """测试完整tick工作流"""
        planner = EmbodiedTaskPlanner()

        # 创建导航任务树
        nav_sequence = SequenceNode("Navigate")
        nav_sequence.add_children(
            AGVCheckSafeCondition(),
            AGVCheckBatteryCondition(0.2),
            AGVMoveToAction(),
            AGVCheckPositionReached(0.1),
        )
        planner.register_task_type('navigate', nav_sequence)

        # 添加任务
        task = EmbodiedTask(
            task_id='nav_001',
            task_type='navigate',
            goal_description='Go to (2, 0)',
            target_position=np.array([2.0, 0.0, 0.0]),
            priority=0,
        )
        planner.add_task(task)

        # 第一次tick应该初始化并开始执行
        status = planner.tick(
            robot_state={'position': [0.0, 0.0, 0.0], 'battery_level': 0.8, 'safety': True},
            world_state={}
        )
        assert status == NodeStatus.RUNNING
        assert planner.current_task is not None
        assert planner.current_task.task_id == 'nav_001'

        # 更新位置接近目标
        status = planner.tick(
            robot_state={'position': [1.95, 0.0, 0.0], 'battery_level': 0.8, 'safety': True},
            world_state={}
        )
        # 应该完成
        assert status == NodeStatus.SUCCESS
        assert planner.status == TaskStatus.COMPLETED
        assert planner.current_task.status == TaskStatus.COMPLETED


class TestAGVTaskPlanner:
    """AGV专用任务规划器测试"""

    @pytest.mark.parametrize("grade", ["S", "M", "L", "XL", "XXL"])
    def test_capabilities_by_grade(self, grade):
        """测试不同等级AGV的规划能力"""
        planner = AGVTaskPlanner(grade=grade)
        caps = planner.get_capabilities()
        assert caps['grade'] == grade
        assert 'max_planning_depth' in caps
        assert 'max_concurrent_tasks' in caps
        assert caps['support_behavior_tree'] is True

        # 等级越高规划能力越强
        capabilities = AGVTaskPlanner.AGV_PLANNING_CAPABILITIES[grade]
        assert caps['max_planning_depth'] == capabilities['max_planning_depth']
        assert caps['max_concurrent_tasks'] == capabilities['max_concurrent_tasks']

    def test_default_tasks_registered(self):
        """测试默认任务已注册"""
        planner = AGVTaskPlanner(grade="M")
        registered = planner.get_status()['registered_types']
        assert 'navigate' in registered
        assert 'transport' in registered
        assert 'patrol' in registered

    def test_navigate_task_execution(self):
        """测试导航任务执行"""
        planner = AGVTaskPlanner(grade="M")
        task = EmbodiedTask(
            task_id='nav_test',
            task_type='navigate',
            goal_description='Navigate to (5, 0)',
            target_position=np.array([5.0, 0.0, 0.0]),
            priority=0,
        )
        planner.add_task(task)

        # 初始位置在原点，电量充足，安全
        status = planner.tick(
            robot_state={'position': [0.0, 0.0, 0.0], 'battery_level': 0.9, 'safety': True},
            world_state={}
        )
        assert status == NodeStatus.RUNNING

    def test_transport_task_structure(self):
        """测试搬运任务结构"""
        planner = AGVTaskPlanner(grade="M")
        assert 'transport' in planner.behavior_trees
        bt = planner.behavior_trees['transport']
        stats = bt.get_statistics()
        # 搬运任务有多个节点
        assert stats['total_nodes'] > 5

    def test_transport_task_with_pickup_dropoff(self):
        """测试完整搬运任务"""
        planner = AGVTaskPlanner(grade="M")
        task = EmbodiedTask(
            task_id='transport_001',
            task_type='transport',
            goal_description='Transport from A to B',
            priority=1,
        )
        planner.add_task(task)

        # 设置拾取和放置位置到目标
        planner.initialize_task(task)
        bt = planner.behavior_trees['transport']
        bt.set_goal({
            'pickup_position': np.array([1.0, 0.0, 0.0]),
            'dropoff_position': np.array([5.0, 0.0, 0.0]),
        })

        # 初始状态检查安全和电量
        bt.update_robot_state({
            'position': [0.0, 0.0, 0.0],
            'battery_level': 0.8,
            'safety': True,
        })
        # 第一个tick会走到这里，设置拾取目标，然后开始移动
        status = bt.tick()
        # 还在移动中
        assert status == NodeStatus.RUNNING


class TestAGVSpecificNodes:
    """AGV专用节点测试"""

    def test_check_battery_condition_ok(self):
        """电量检查 - 电量充足"""
        cond = AGVCheckBatteryCondition(min_battery=0.2)
        bb = Blackboard()
        bb.update_robot_state({'battery_level': 0.5})
        assert cond.tick(bb) == NodeStatus.SUCCESS

    def test_check_battery_condition_low(self):
        """电量检查 - 电量不足"""
        cond = AGVCheckBatteryCondition(min_battery=0.2)
        bb = Blackboard()
        bb.update_robot_state({'battery_level': 0.1})
        assert cond.tick(bb) == NodeStatus.FAILURE

    def test_check_safe_condition(self):
        """安全检查测试"""
        cond = AGVCheckSafeCondition()
        bb = Blackboard()
        bb.update_robot_state({'safety': True})
        assert cond.tick(bb) == NodeStatus.SUCCESS
        bb.update_robot_state({'safety': False})
        assert cond.tick(bb) == NodeStatus.FAILURE

    def test_check_position_reached(self):
        """位置到达检查"""
        cond = AGVCheckPositionReached(threshold=0.1)
        bb = Blackboard()
        bb.update_robot_state({'position': [1.05, 0.0, 0.0]})
        bb.goal_state['target_position'] = np.array([1.0, 0.0, 0.0])
        # 距离0.05 < 0.1 → 到达
        assert cond.tick(bb) == NodeStatus.SUCCESS

        bb.update_robot_state({'position': [2.0, 0.0, 0.0]})
        # 距离1.0 > 0.1 → 未到达
        assert cond.tick(bb) == NodeStatus.FAILURE

    def test_grasp_action_timing(self):
        """抓取动作计时"""
        grasp = AGVGraspAction()
        bb = Blackboard()
        bb.goal_state['target_object'] = 'box_001'

        # 第一次tick开始抓取
        status = grasp.execute(bb)
        assert status == NodeStatus.RUNNING

    def test_release_action_timing(self):
        """释放动作计时"""
        release = AGVReleaseAction()
        bb = Blackboard()
        status = release.execute(bb)
        assert status == NodeStatus.RUNNING


class TestBehaviorTreeReset:
    """行为树重置测试"""

    def test_reset_whole_tree(self):
        """整树重置测试"""
        root = SequenceNode("Root")
        root.add_children(
            RepeaterNode(ConditionNode(lambda bb: True, "Cond"), times=2),
        )
        bt = BehaviorTree(root)
        bt.tick()
        assert bt.root.status != NodeStatus.IDLE
        assert bt.root.children[0].status != NodeStatus.IDLE

        bt.reset()
        assert bt.last_status == NodeStatus.IDLE
        assert bt.root.status == NodeStatus.IDLE
        assert bt.root.children[0].status == NodeStatus.IDLE


def run_all_tests():
    """运行所有测试"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_all_tests()
