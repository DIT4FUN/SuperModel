"""
tests/embodiment/test_plan_task_and_blackboard.py

测试 AGVTaskPlanner.plan_task 方法和 TaskExecutor 黑板初始化
- AGVTaskPlanner.plan_task 接口
- 目标位置解析
- 黑板状态初始化
- 完整行为树执行
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'src')

from embodied.behavior_tree import (
    AGVTaskPlanner,
    BehaviorTree,
    NodeStatus,
)


class TestAGVTaskPlannerPlanTask:
    """AGVTaskPlanner.plan_task 方法测试"""

    def test_plan_task_returns_dict(self):
        """plan_task 返回正确的字典结构"""
        planner = AGVTaskPlanner(grade='M')
        result = planner.plan_task('transport', 'station_A', 'M')
        assert isinstance(result, dict)
        assert 'task_id' in result
        assert 'task_type' in result
        assert result['task_type'] == 'transport'
        assert 'behavior_tree' in result
        assert 'target_position' in result

    def test_plan_task_resolves_known_target(self):
        """plan_task 解析已知目标为坐标"""
        planner = AGVTaskPlanner(grade='M')
        result = planner.plan_task('transport', 'station_A', 'M')
        # station_A 应该解析为 [10.0, 0.0, 0.0]
        assert result['target_position'] is not None
        pos = np.array(result['target_position'])
        assert pos.shape == (3,)
        assert abs(pos[0] - 10.0) < 0.01

    def test_plan_task_resolves_station_b(self):
        """plan_task 解析 station_B 为正确坐标"""
        planner = AGVTaskPlanner(grade='M')
        result = planner.plan_task('transport', 'station_B', 'M')
        pos = np.array(result['target_position'])
        assert abs(pos[0] - 20.0) < 0.01

    def test_plan_task_resolves_entrance(self):
        """plan_task 解析 entrance 为 [0,0,0]"""
        planner = AGVTaskPlanner(grade='M')
        result = planner.plan_task('navigate', 'entrance', 'M')
        pos = np.array(result['target_position'])
        assert abs(pos[0]) < 0.01

    def test_plan_task_unknown_target_defaults(self):
        """plan_task 对未知目标使用默认坐标"""
        planner = AGVTaskPlanner(grade='M')
        result = planner.plan_task('transport', 'random_unknown_place', 'M')
        # 应该返回默认 [10, 0, 0]
        assert result['target_position'] is not None
        pos = np.array(result['target_position'])
        assert abs(pos[0] - 10.0) < 0.01

    def test_plan_task_with_kwargs(self):
        """plan_task 接受额外参数"""
        planner = AGVTaskPlanner(grade='M')
        result = planner.plan_task('transport', 'station_A', priority=1, timeout=600.0)
        assert result['task_type'] == 'transport'
        assert result['grade'] == 'M'

    def test_plan_task_returns_behavior_tree(self):
        """plan_task 返回可执行的 BehaviorTree"""
        planner = AGVTaskPlanner(grade='M')
        result = planner.plan_task('transport', 'station_A')
        bt = result['behavior_tree']
        assert isinstance(bt, BehaviorTree)
        assert bt.root is not None

    def test_plan_task_all_grades(self):
        """plan_task 支持所有 AGV 等级"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            planner = AGVTaskPlanner(grade=grade)
            result = planner.plan_task('transport', 'station_A')
            assert result['grade'] == grade
            assert result['behavior_tree'] is not None


class TestAGVTaskPlannerExecute:
    """AGVTaskPlanner 行为树执行测试"""

    def test_transport_task_completes(self):
        """搬运任务完整执行流程"""
        planner = AGVTaskPlanner(grade='M')
        result = planner.plan_task('transport', 'station_A', 'M')
        bt = result['behavior_tree']

        # 初始化黑板
        bt.blackboard.update_robot_state({
            'position': [0.0, 0.0, 0.0],
            'battery_level': 0.8,
            'safety': True,
        })
        bt.blackboard.goal_state.update({
            'target_position': np.array(result['target_position']),
            'target_object': 'package_001',
            'pickup_position': np.array([0.0, 0.0, 0.0]),
            'dropoff_position': np.array(result['target_position']),
        })

        # 快速执行（5秒超时）
        start = __import__('time').time()
        max_ticks = 60
        tick = 0
        while tick < max_ticks:
            status = bt.tick()
            tick += 1
            if status in (NodeStatus.SUCCESS, NodeStatus.FAILURE):
                break
            if __import__('time').time() - start > 5.0:
                break

        assert status == NodeStatus.SUCCESS, f"BT returned {status} after {tick} ticks"
        assert tick < max_ticks

    def test_navigate_task_completes(self):
        """导航任务执行 (station_A距0m约10m,50ticks足够)"""
        planner = AGVTaskPlanner(grade='M')
        result = planner.plan_task('navigate', 'station_A', 'M')  # 10m距离, 0.25m/tick → 40ticks
        bt = result['behavior_tree']

        bt.blackboard.update_robot_state({
            'position': [0.0, 0.0, 0.0],
            'battery_level': 0.8,
            'safety': True,
        })
        bt.blackboard.goal_state['target_position'] = np.array(result['target_position'])

        start = __import__('time').time()
        max_ticks = 60
        tick = 0
        while tick < max_ticks:
            status = bt.tick()
            tick += 1
            if status in (NodeStatus.SUCCESS, NodeStatus.FAILURE):
                break
            if __import__('time').time() - start > 10.0:
                break

        assert status == NodeStatus.SUCCESS, f"BT returned {status} after {tick} ticks"


class TestBlackboardSetup:
    """黑板状态初始化测试"""

    def test_battery_level_defaults(self):
        """电池电量默认值测试"""
        planner = AGVTaskPlanner(grade='M')
        result = planner.plan_task('transport', 'station_A')
        bt = result['behavior_tree']
        # 默认电池 0.8
        assert bt.blackboard.get_battery_level() is None  # 未设置时返回 None
        bt.blackboard.update_robot_state({'battery_level': 0.8})
        assert bt.blackboard.get_battery_level() == 0.8

    def test_is_safe_default(self):
        """安全状态默认值测试"""
        planner = AGVTaskPlanner(grade='M')
        result = planner.plan_task('transport', 'station_A')
        bt = result['behavior_tree']
        # 默认安全
        assert bt.blackboard.is_safe() is True
        # 不安全状态
        bt.blackboard.update_robot_state({'safety': False})
        assert bt.blackboard.is_safe() is False

    def test_position_tracking(self):
        """位置跟踪测试"""
        planner = AGVTaskPlanner(grade='M')
        result = planner.plan_task('transport', 'station_A')
        bt = result['behavior_tree']

        bt.blackboard.update_robot_state({'position': [0.0, 0.0, 0.0]})
        pos = bt.blackboard.get_robot_position()
        assert pos is not None
        assert pos[0] == 0.0

        # 位置更新
        bt.blackboard.update_robot_state({'position': [5.0, 0.0, 0.0]})
        pos = bt.blackboard.get_robot_position()
        assert abs(pos[0] - 5.0) < 0.01


class TestAGVTaskPlannerCapabilities:
    """AGVTaskPlanner 能力测试"""

    def test_capabilities_include_grade(self):
        """能力包含等级信息"""
        planner = AGVTaskPlanner(grade='M')
        caps = planner.get_capabilities()
        assert caps['grade'] == 'M'
        assert 'max_planning_depth' in caps
        assert 'max_concurrent_tasks' in caps

    def test_all_grades_capabilities(self):
        """所有等级能力正确"""
        expected = {
            'S': {'max_planning_depth': 3, 'max_concurrent_tasks': 1},
            'M': {'max_planning_depth': 6, 'max_concurrent_tasks': 2},
            'L': {'max_planning_depth': 10, 'max_concurrent_tasks': 3},
            'XL': {'max_planning_depth': 15, 'max_concurrent_tasks': 4},
            'XXL': {'max_planning_depth': 20, 'max_concurrent_tasks': 8},
        }
        for grade, exp in expected.items():
            planner = AGVTaskPlanner(grade=grade)
            caps = planner.get_capabilities()
            assert caps['max_planning_depth'] == exp['max_planning_depth']
            assert caps['max_concurrent_tasks'] == exp['max_concurrent_tasks']
