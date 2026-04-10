"""
behavior_tree_tests.py - 行为树具身任务规划测试
SuperModel 超模态大模型具身智能系统

测试内容:
- 行为树基本节点功能测试
- 序列/选择/并行节点测试
- AGV任务行为树测试
- 具身任务规划器测试
- 多任务调度测试
- 五级AGV规划能力适配测试
"""

import unittest
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embodied.behavior_tree import (
    NodeStatus,
    TaskStatus,
    SequenceNode,
    SelectorNode,
    ParallelNode,
    RepeaterNode,
    UntilFailNode,
    UntilSuccessNode,
    InverterNode,
    ConditionNode,
    LambdaActionNode,
    BehaviorTree,
    Blackboard,
    EmbodiedTask,
    EmbodiedTaskPlanner,
    AGVTaskPlanner,
    AGVCheckBatteryCondition,
    AGVCheckSafeCondition,
    AGVCheckPositionReached,
    AGVMoveToAction,
)


class TestBasicNodes(unittest.TestCase):
    """测试基本节点功能"""

    def setUp(self):
        self.blackboard = Blackboard()

    def test_sequence_node_all_success(self):
        """序列节点 - 全部成功"""
        seq = SequenceNode("TestSequence")
        seq.add_children(
            LambdaActionNode(lambda bb: NodeStatus.SUCCESS, "A"),
            LambdaActionNode(lambda bb: NodeStatus.SUCCESS, "B"),
            LambdaActionNode(lambda bb: NodeStatus.SUCCESS, "C"),
        )
        status = seq.tick(self.blackboard)
        self.assertEqual(status, NodeStatus.SUCCESS)

    def test_sequence_node_first_failure(self):
        """序列节点 - 第一个失败"""
        seq = SequenceNode("TestSequence")
        seq.add_children(
            LambdaActionNode(lambda bb: NodeStatus.FAILURE, "A"),
            LambdaActionNode(lambda bb: NodeStatus.SUCCESS, "B"),
        )
        status = seq.tick(self.blackboard)
        self.assertEqual(status, NodeStatus.FAILURE)

    def test_selector_node_first_success(self):
        """选择节点 - 第一个成功"""
        sel = SelectorNode("TestSelector")
        sel.add_children(
            LambdaActionNode(lambda bb: NodeStatus.SUCCESS, "A"),
            LambdaActionNode(lambda bb: NodeStatus.FAILURE, "B"),
        )
        status = sel.tick(self.blackboard)
        self.assertEqual(status, NodeStatus.SUCCESS)

    def test_selector_node_all_fail(self):
        """选择节点 - 全部失败"""
        sel = SelectorNode("TestSelector")
        sel.add_children(
            LambdaActionNode(lambda bb: NodeStatus.FAILURE, "A"),
            LambdaActionNode(lambda bb: NodeStatus.FAILURE, "B"),
        )
        status = sel.tick(self.blackboard)
        self.assertEqual(status, NodeStatus.FAILURE)

    def test_inverter_node(self):
        """反转节点"""
        # 成功 → 失败
        inv = InverterNode(LambdaActionNode(lambda bb: NodeStatus.SUCCESS))
        status = inv.tick(self.blackboard)
        self.assertEqual(status, NodeStatus.FAILURE)

        # 失败 → 成功
        inv = InverterNode(LambdaActionNode(lambda bb: NodeStatus.FAILURE))
        status = inv.tick(self.blackboard)
        self.assertEqual(status, NodeStatus.SUCCESS)

    def test_condition_node_true(self):
        """条件节点 - 真"""
        cond = ConditionNode(lambda bb: True)
        status = cond.tick(self.blackboard)
        self.assertEqual(status, NodeStatus.SUCCESS)

    def test_condition_node_false(self):
        """条件节点 - 假"""
        cond = ConditionNode(lambda bb: False)
        status = cond.tick(self.blackboard)
        self.assertEqual(status, NodeStatus.FAILURE)

    def test_repeater_node_limited(self):
        """重复节点 - 有限次数"""
        count = 0
        def action(bb):
            nonlocal count
            count += 1
            return NodeStatus.SUCCESS

        node = RepeaterNode(LambdaActionNode(action), times=3)
        status = node.tick(self.blackboard)
        # 第一次tick，还没完成三次
        self.assertEqual(status, NodeStatus.RUNNING)
        self.assertEqual(count, 1)

        # 继续tick直到完成
        for _ in range(2):
            status = node.tick(self.blackboard)
        self.assertEqual(status, NodeStatus.SUCCESS)
        self.assertEqual(count, 3)

    def test_until_fail_node(self):
        """直到失败节点"""
        count = 0
        def action(bb):
            nonlocal count
            count += 1
            return NodeStatus.SUCCESS if count < 3 else NodeStatus.FAILURE

        node = UntilFailNode(LambdaActionNode(action))
        # 前两次都应该返回 RUNNING
        for _ in range(2):
            status = node.tick(self.blackboard)
            self.assertEqual(status, NodeStatus.RUNNING)
        # 第三次失败，返回 SUCCESS
        status = node.tick(self.blackboard)
        self.assertEqual(status, NodeStatus.SUCCESS)
        self.assertEqual(count, 3)

    def test_until_success_node(self):
        """直到成功节点"""
        count = 0
        def action(bb):
            nonlocal count
            count += 1
            return NodeStatus.FAILURE if count < 3 else NodeStatus.SUCCESS

        node = UntilSuccessNode(LambdaActionNode(action))
        for _ in range(2):
            status = node.tick(self.blackboard)
            self.assertEqual(status, NodeStatus.RUNNING)
        status = node.tick(self.blackboard)
        self.assertEqual(status, NodeStatus.SUCCESS)
        self.assertEqual(count, 3)


class TestParallelNode(unittest.TestCase):
    """测试并行节点"""

    def setUp(self):
        self.blackboard = Blackboard()

    def test_parallel_require_all_success_all_success(self):
        """并行节点 - 全部成功要求，全部成功"""
        parallel = ParallelNode(
            "Test",
            success_policy=ParallelNode.Policy.REQUIRE_ALL,
            failure_policy=ParallelNode.Policy.REQUIRE_ANY
        )
        parallel.add_children(
            LambdaActionNode(lambda bb: NodeStatus.SUCCESS),
            LambdaActionNode(lambda bb: NodeStatus.SUCCESS),
        )
        status = parallel.tick(self.blackboard)
        self.assertEqual(status, NodeStatus.SUCCESS)

    def test_parallel_require_all_success_one_failure(self):
        """并行节点 - 全部成功要求，一个失败"""
        parallel = ParallelNode(
            "Test",
            success_policy=ParallelNode.Policy.REQUIRE_ALL,
            failure_policy=ParallelNode.Policy.REQUIRE_ANY
        )
        parallel.add_children(
            LambdaActionNode(lambda bb: NodeStatus.SUCCESS),
            LambdaActionNode(lambda bb: NodeStatus.FAILURE),
        )
        status = parallel.tick(self.blackboard)
        self.assertEqual(status, NodeStatus.FAILURE)


class TestBehaviorTree(unittest.TestCase):
    """测试行为树整体"""

    def setUp(self):
        # 创建一个简单的导航行为树
        root = SequenceNode("Navigation")
        root.add_children(
            AGVCheckSafeCondition(),
            AGVCheckBatteryCondition(min_battery=0.2),
            AGVMoveToAction(),
            AGVCheckPositionReached(threshold=0.1),
        )
        self.bt = BehaviorTree(root)

    def test_bt_statistics(self):
        """行为树统计"""
        stats = self.bt.get_statistics()
        self.assertGreater(stats['total_nodes'], 0)
        self.assertIn('SequenceNode', stats['node_types'])
        self.assertIn('AGVCheckSafeCondition', stats['node_types'])

    def test_bt_update_state(self):
        """更新机器人状态"""
        self.bt.update_robot_state({
            'position': [0.0, 0.0, 0.0],
            'safety': True,
            'battery_level': 0.8,
        })
        self.bt.update_world_state({
            'obstacles': [],
        })
        self.bt.set_goal({'target_position': [1.0, 0.0, 0.0]})

    def test_bt_reset(self):
        """重置行为树"""
        self.bt.tick()
        self.bt.reset()
        self.assertEqual(self.bt.last_status, NodeStatus.IDLE)


class TestAGVNodes(unittest.TestCase):
    """测试AGV专用节点"""

    def setUp(self):
        self.bb = Blackboard()
        self.bb.update_robot_state({
            'safety': True,
            'battery_level': 0.5,
            'position': [0.0, 0.0, 0.0],
        })
        self.bb.goal_state['target_position'] = [1.0, 0.0, 0.0]

    def test_check_battery_ok(self):
        """电量检查 - 电量充足"""
        node = AGVCheckBatteryCondition(min_battery=0.2)
        status = node.tick(self.bb)
        self.assertEqual(status, NodeStatus.SUCCESS)

    def test_check_battery_low(self):
        """电量检查 - 电量不足"""
        self.bb.update_robot_state({'battery_level': 0.1})
        node = AGVCheckBatteryCondition(min_battery=0.2)
        status = node.tick(self.bb)
        self.assertEqual(status, NodeStatus.FAILURE)

    def test_check_safe_ok(self):
        """安全检查 - 安全"""
        node = AGVCheckSafeCondition()
        status = node.tick(self.bb)
        self.assertEqual(status, NodeStatus.SUCCESS)

    def test_check_safe_not_ok(self):
        """安全检查 - 不安全"""
        self.bb.update_robot_state({'safety': False})
        node = AGVCheckSafeCondition()
        status = node.tick(self.bb)
        self.assertEqual(status, NodeStatus.FAILURE)

    def test_check_position_reached_yes(self):
        """位置检查 - 已到达"""
        self.bb.update_robot_state({'position': [0.99, 0.01, 0.0]})
        node = AGVCheckPositionReached(threshold=0.1)
        status = node.tick(self.bb)
        self.assertEqual(status, NodeStatus.SUCCESS)

    def test_check_position_reached_no(self):
        """位置检查 - 未到达"""
        self.bb.update_robot_state({'position': [0.5, 0.0, 0.0]})
        node = AGVCheckPositionReached(threshold=0.1)
        status = node.tick(self.bb)
        self.assertEqual(status, NodeStatus.FAILURE)


class TestEmbodiedTaskPlanner(unittest.TestCase):
    """测试具身任务规划器"""

    def setUp(self):
        self.planner = EmbodiedTaskPlanner()

        # 创建一个简单的导航任务
        nav_root = SequenceNode("NavigateRoot")
        nav_root.add_children(
            AGVCheckSafeCondition(),
            AGVCheckBatteryCondition(),
            AGVMoveToAction(),
        )
        self.planner.register_task_type('navigate', nav_root)

    def test_add_task(self):
        """添加任务"""
        task = EmbodiedTask(
            task_id='test_001',
            task_type='navigate',
            goal_description='Navigate to origin',
            target_position=np.array([0.0, 0.0, 0.0]),
            priority=0,
        )
        self.planner.add_task(task)
        self.assertIn('test_001', self.planner.tasks)

    def test_select_next_task_by_priority(self):
        """按优先级选择下一个任务"""
        task1 = EmbodiedTask(task_id='low', task_type='navigate', goal_description='Low', priority=10)
        task2 = EmbodiedTask(task_id='high', task_type='navigate', goal_description='High', priority=0)
        self.planner.add_task(task1)
        self.planner.add_task(task2)

        next_task = self.planner.select_next_task()
        self.assertEqual(next_task.task_id, 'high')

    def test_tick_with_no_tasks(self):
        """没有任务时tick"""
        status = self.planner.tick({}, {})
        self.assertEqual(status, NodeStatus.IDLE)

    def test_initialize_task(self):
        """初始化任务"""
        task = EmbodiedTask(
            task_id='nav_001',
            task_type='navigate',
            goal_description='Test navigation',
            target_position=np.array([1.0, 0.0, 0.0]),
        )
        self.planner.add_task(task)
        bt = self.planner.initialize_task(task)
        self.assertIsNotNone(bt)
        self.assertEqual(task.status, TaskStatus.RUNNING)
        self.assertEqual(self.planner.status, TaskStatus.RUNNING)

    def test_abort_current(self):
        """中止当前任务"""
        task = EmbodiedTask(task_id='test', task_type='navigate', goal_description='Test')
        self.planner.add_task(task)
        self.planner.initialize_task(task)
        self.planner.abort_current()
        self.assertIsNone(self.planner.current_task)
        self.assertEqual(self.planner.status, TaskStatus.ABORTED)

    def test_get_status(self):
        """获取规划器状态"""
        status = self.planner.get_status()
        self.assertIn('current_task', status)
        self.assertIn('pending_tasks', status)
        self.assertIn('registered_types', status)
        self.assertIn('navigate', status['registered_types'])


class TestAGVTaskPlanner(unittest.TestCase):
    """测试AGV专用任务规划器"""

    def test_all_grades_capabilities(self):
        """测试所有等级的规划能力"""
        grades = ['S', 'M', 'L', 'XL', 'XXL']
        for grade in grades:
            planner = AGVTaskPlanner(grade=grade)
            caps = planner.get_capabilities()
            self.assertEqual(caps['grade'], grade)
            self.assertIn('max_planning_depth', caps)
            self.assertIn('max_concurrent_tasks', caps)
            self.assertTrue(caps['support_behavior_tree'])

    def test_default_tasks_registered(self):
        """默认任务已注册"""
        planner = AGVTaskPlanner(grade='M')
        status = planner.get_status()
        registered = status['registered_types']
        self.assertIn('navigate', registered)
        self.assertIn('transport', registered)
        self.assertIn('patrol', registered)

    def test_planning_depth_increases(self):
        """规划深度随等级增加"""
        depths = {}
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            planner = AGVTaskPlanner(grade=grade)
            depths[grade] = planner.get_capabilities()['max_planning_depth']

        self.assertLess(depths['S'], depths['M'])
        self.assertLess(depths['M'], depths['L'])
        self.assertLess(depths['L'], depths['XL'])
        self.assertLess(depths['XL'], depths['XXL'])

    def test_replan_interval_decreases(self):
        """重规划间隔随等级减少"""
        intervals = {}
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            planner = AGVTaskPlanner(grade=grade)
            intervals[grade] = planner.get_capabilities()['replan_interval']

        self.assertGreater(intervals['S'], intervals['M'])
        self.assertGreater(intervals['M'], intervals['L'])
        self.assertGreater(intervals['L'], intervals['XL'])
        self.assertGreater(intervals['XL'], intervals['XXL'])


class TestEmbodiedTask(unittest.TestCase):
    """测试具身任务定义"""

    def test_task_start(self):
        """任务开始"""
        task = EmbodiedTask(
            task_id='test',
            task_type='navigate',
            goal_description='Test'
        )
        self.assertEqual(task.status, TaskStatus.IDLE)
        task.start()
        self.assertEqual(task.status, TaskStatus.RUNNING)
        self.assertIsNotNone(task.start_time)

    def test_task_finish_success(self):
        """任务成功完成"""
        task = EmbodiedTask(
            task_id='test',
            task_type='navigate',
            goal_description='Test'
        )
        task.start()
        time.sleep(0.01)
        task.finish(success=True)
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertTrue(task.success)
        self.assertGreater(task.get_duration(), 0)

    def test_task_timeout(self):
        """任务超时"""
        task = EmbodiedTask(
            task_id='test',
            task_type='navigate',
            goal_description='Test',
            timeout=0.01
        )
        task.start()
        time.sleep(0.02)
        self.assertTrue(task.is_timeout())


class TestBlackboard(unittest.TestCase):
    """测试黑板"""

    def setUp(self):
        self.bb = Blackboard()

    def test_get_set_has_remove(self):
        """基本存取"""
        self.bb.set('key', 'value')
        self.assertTrue(self.bb.has('key'))
        self.assertEqual(self.bb.get('key'), 'value')
        self.assertTrue(self.bb.remove('key'))
        self.assertFalse(self.bb.has('key'))

    def test_update_robot_state(self):
        """更新机器人状态"""
        self.bb.update_robot_state({'position': [1, 2, 3], 'battery': 0.8})
        self.assertEqual(self.bb.robot_state['position'], [1, 2, 3])
        self.assertEqual(self.bb.robot_state['battery'], 0.8)

    def test_get_robot_position_none(self):
        """获取位置 - 无数据"""
        pos = self.bb.get_robot_position()
        self.assertIsNone(pos)

    def test_get_robot_position_convert(self):
        """获取位置 - 转换为numpy"""
        self.bb.update_robot_state({'position': [1.0, 2.0, 3.0]})
        pos = self.bb.get_robot_position()
        self.assertIsInstance(pos, np.ndarray)
        np.testing.assert_array_equal(pos, np.array([1.0, 2.0, 3.0]))


if __name__ == '__main__':
    unittest.main()
