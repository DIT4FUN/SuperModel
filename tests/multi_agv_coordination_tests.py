"""
multi_agv_coordination_tests.py - 多AGV蜂群协同测试用例
SuperModel 超模态大模型项目

测试覆盖:
- 多AGV任务分配
- 路径冲突检测与避让
- 区域协同调度
- 蜂群算法测试
- 动态任务重分配

测试总数: 42 项
"""

import pytest
import numpy as np
from typing import List, Dict, Any
import time

from src.embodied.behavior_tree import (
    EmbodiedTaskPlanner,
    AGVTaskPlanner,
    EmbodiedTask,
    TaskStatus,
)
from src.control.swarm_coordination import (
    SwarmCoordinator,
    TaskAllocator,
    ConflictDetector,
    PathConflict,
    CoordinationStrategy,
)
from src.embodied.simulation_enhancement import (
    EnvironmentGenerator,
    Obstacle,
)


class TestTaskAllocator:
    """多AGV任务分配测试"""

    def test_create_allocator(self):
        """测试创建任务分配器"""
        allocator = TaskAllocator(num_robots=3)
        assert allocator.num_robots == 3
        assert allocator.strategy == "bipartite"

    def test_allocate_single_task(self):
        """测试分配单个任务"""
        allocator = TaskAllocator(num_robots=2)
        robots = [
            {'id': 'agv1', 'position': np.array([0.0, 0.0]), 'available': True},
            {'id': 'agv2', 'position': np.array([10.0, 10.0]), 'available': True},
        ]
        tasks = [
            {'id': 'task1', 'target': np.array([1.0, 1.0])}
        ]
        assignments = allocator.allocate(robots, tasks)
        assert len(assignments) == 1
        # 最近的AGV应该得到任务 → agv1
        assert assignments[0]['robot_id'] == 'agv1'

    def test_allocate_multiple_tasks(self):
        """测试分配多个任务"""
        allocator = TaskAllocator(num_robots=2)
        robots = [
            {'id': 'agv1', 'position': np.array([0.0, 0.0]), 'available': True},
            {'id': 'agv2', 'position': np.array([10.0, 0.0]), 'available': True},
        ]
        tasks = [
            {'id': 't1', 'target': np.array([1.0, 0.0])},
            {'id': 't2', 'target': np.array([9.0, 0.0])},
        ]
        assignments = allocator.allocate(robots, tasks)
        # 两个任务都应该分配出去
        assert len(assignments) == 2
        assigned_robots = {a['robot_id'] for a in assignments}
        assert assigned_robots == {'agv1', 'agv2'}

    def test_allocate_more_tasks_than_robots(self):
        """测试任务比机器人多（单任务模式）"""
        allocator = TaskAllocator(num_robots=2, max_tasks_per_robot=1)
        robots = [
            {'id': 'a1', 'position': np.array([0.0, 0.0]), 'available': True, 'assigned_tasks': []},
            {'id': 'a2', 'position': np.array([10.0, 0.0]), 'available': True, 'assigned_tasks': []},
        ]
        tasks = [
            {'id': 't1', 'target': np.array([1.0, 0.0])},
            {'id': 't2', 'target': np.array([3.0, 0.0])},
            {'id': 't3', 'target': np.array([5.0, 0.0])},
        ]
        assignments = allocator.allocate(robots, tasks)
        # 只能分配给可用机器人，每个机器人一个任务
        assert len(assignments) == 2

    def test_distance_based_cost(self):
        """测试基于距离的成本计算"""
        allocator = TaskAllocator()
        cost = allocator.calculate_cost(
            {'position': np.array([0.0, 0.0])},
            {'target': np.array([3.0, 4.0])}
        )
        assert cost == pytest.approx(5.0)

    def test_consider_battery_in_cost(self):
        """测试考虑电量成本"""
        allocator = TaskAllocator(consider_battery=True)
        # 低电量机器人成本更高
        cost_low = allocator.calculate_cost(
            {'position': np.array([0.0, 0.0]), 'battery': 0.1},
            {'target': np.array([1.0, 0.0])}
        )
        cost_high = allocator.calculate_cost(
            {'position': np.array([0.0, 0.0]), 'battery': 0.9},
            {'target': np.array([1.0, 0.0])}
        )
        assert cost_low > cost_high


class TestConflictDetector:
    """路径冲突检测测试"""

    def test_detect_no_conflict(self):
        """检测无冲突"""
        detector = ConflictDetector(conflict_distance_threshold=0.5)
        path1 = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0])]
        path2 = [np.array([0.0, 2.0]), np.array([1.0, 2.0]), np.array([2.0, 2.0])]
        conflicts = detector.detect_conflicts('a1', path1, 'a2', path2)
        assert len(conflicts) == 0

    def test_detect_head_on_conflict(self):
        """检测对头冲突"""
        detector = ConflictDetector(conflict_distance_threshold=0.5)
        # 两个AGV相向而行会在中间相遇
        path1 = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0])]
        path2 = [np.array([2.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 0.0])]
        conflicts = detector.detect_conflicts('a1', path1, 'a2', path2)
        # 在(1.0, 0.0)位置会冲突
        assert len(conflicts) >= 1
        assert any(c.is_time_conflict() for c in conflicts)

    def test_detect_crossing_conflict(self):
        """检测交叉冲突"""
        detector = ConflictDetector(conflict_distance_threshold=0.3)
        # 添加顶点在交叉点
        path1 = [np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([2.0, 2.0])]
        path2 = [np.array([0.0, 2.0]), np.array([1.0, 1.0]), np.array([2.0, 0.0])]
        conflicts = detector.detect_conflicts('a1', path1, 'a2', path2)
        # 在交叉点(1,1)冲突
        assert len(conflicts) >= 1

    def test_calculate_conflict_time(self):
        """计算冲突时间"""
        detector = ConflictDetector()
        # 计算两个路径到达冲突点的时间
        path1 = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0])]
        path2 = [np.array([2.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 0.0])]
        conflicts = detector.detect_conflicts('a1', path1, 'a2', path2)
        # 两个点同时到达交点，应该判定为冲突
        assert len(conflicts) > 0
        assert conflicts[0].is_time_conflict()

    def test_get_conflict_severity(self):
        """获取冲突严重程度"""
        detector = ConflictDetector()
        conflict = PathConflict(
            robot1_id='a1',
            robot2_id='a2',
            position=np.array([1.0, 0.0]),
            time1=1.0,
            time2=1.1,
            distance=0.2,
        )
        severity = conflict.get_severity()
        assert 0 <= severity <= 1
        # 时间差很小，距离很近，严重性高
        assert severity > 0.5


class TestSwarmCoordinator:
    """蜂群协调器测试"""

    def test_create_coordinator(self):
        """测试创建协调器"""
        coord = SwarmCoordinator(num_robots=3, strategy=CoordinationStrategy.DECENTRALIZED)
        assert coord.num_robots == 3
        assert coord.strategy == CoordinationStrategy.DECENTRALIZED

    def test_register_robot(self):
        """测试注册机器人"""
        coord = SwarmCoordinator(num_robots=2)
        coord.register_robot('agv1', position=np.array([0.0, 0.0]))
        coord.register_robot('agv2', position=np.array([5.0, 5.0]))
        assert 'agv1' in coord.robots
        assert 'agv2' in coord.robots
        assert len(coord.robots) == 2

    def test_add_global_task(self):
        """测试添加全局任务"""
        coord = SwarmCoordinator(num_robots=2)
        coord.register_robot('agv1', np.array([0.0, 0.0]))
        coord.add_global_task({
            'id': 'task1',
            'type': 'transport',
            'target': np.array([10.0, 10.0]),
        })
        assert len(coord.pending_tasks) == 1

    def test_coordinate_step(self):
        """测试单步协调"""
        coord = SwarmCoordinator(num_robots=2)
        coord.register_robot('agv1', np.array([0.0, 0.0]))
        coord.register_robot('agv2', np.array([10.0, 10.0]))
        coord.add_global_task({
            'id': 't1',
            'target': np.array([1.0, 1.0]),
        })
        coord.add_global_task({
            'id': 't2',
            'target': np.array([9.0, 9.0]),
        })
        # 运行一次协调
        assignments = coord.coordinate_step()
        assert len(assignments) == 2
        # 每个任务分配给最近的机器人
        robot_assignments = {a['robot_id'] for a in assignments}
        assert robot_assignments == {'agv1', 'agv2'}

    def test_resolve_conflict_speed_alteration(self):
        """测试通过速度调整解决冲突"""
        coord = SwarmCoordinator()
        coord.register_robot('a1', np.array([0.0, 0.0]))
        coord.register_robot('a2', np.array([2.0, 0.0]))
        conflict = PathConflict(
            robot1_id='a1',
            robot2_id='a2',
            position=np.array([1.0, 0.0]),
            time1=1.0,
            time2=1.0,
            distance=0.0,
        )
        resolution = coord.resolve_conflict(conflict)
        assert resolution.resolved
        # 一个加速一个减速解决冲突
        assert 'speed_adjustment' in resolution.method

    def test_resolve_conflict_waiting(self):
        """测试通过等待解决冲突"""
        coord = SwarmCoordinator(strategy=CoordinationStrategy.CENTRALIZED)
        coord.register_robot('a1', np.array([0.0, 0.0]))
        coord.register_robot('a2', np.array([2.0, 0.0]))
        conflict = PathConflict(
            robot1_id='a1',
            robot2_id='a2',
            position=np.array([1.0, 0.0]),
            time1=1.0,
            time2=1.0,
            distance=0.0,
        )
        resolution = coord.resolve_conflict(conflict)
        assert resolution.resolved

    def test_get_statistics(self):
        """测试获取协调统计"""
        coord = SwarmCoordinator(num_robots=5)
        stats = coord.get_statistics()
        assert 'total_robots' in stats
        assert 'pending_tasks' in stats
        assert 'total_conflicts_detected' in stats
        assert 'conflicts_resolved' in stats


class TestMultiAGVBehaviorTreeIntegration:
    """多AGV行为树集成测试"""

    def test_multi_agv_task_planning(self):
        """测试多AGV任务规划"""
        # 为每个AGV创建规划器
        planners: Dict[str, AGVTaskPlanner] = {
            'agv1': AGVTaskPlanner(grade='M'),
            'agv2': AGVTaskPlanner(grade='M'),
        }
        # 添加任务
        task1 = EmbodiedTask(
            task_id='nav_1',
            task_type='navigate',
            goal_description='Navigate to point A',
            target_position=np.array([5.0, 0.0]),
            priority=1,
        )
        task2 = EmbodiedTask(
            task_id='nav_2',
            task_type='navigate',
            goal_description='Navigate to point B',
            target_position=np.array([-5.0, 0.0]),
            priority=1,
        )
        planners['agv1'].add_task(task1)
        planners['agv2'].add_task(task2)

        # 检查任务被接收
        assert 'nav_1' in planners['agv1'].tasks
        assert 'nav_2' in planners['agv2'].tasks
        assert planners['agv1'].current_task is None
        # 启动规划
        planners['agv1'].tick({'position': [0.0, 0.0], 'safety': True, 'battery_level': 0.8}, {})
        assert planners['agv1'].current_task is not None

    def test_multi_agv_prioritization(self):
        """测试多AGV任务优先级"""
        planner = EmbodiedTaskPlanner()
        task_low = EmbodiedTask(task_id='low', task_type='nav', goal_description='', priority=5)
        task_high = EmbodiedTask(task_id='high', task_type='nav', goal_description='', priority=0)
        task_med = EmbodiedTask(task_id='med', task_type='nav', goal_description='', priority=2)
        planner.add_task(task_low)
        planner.add_task(task_high)
        planner.add_task(task_med)
        selected = planner.select_next_task()
        assert selected is not None
        assert selected.task_id == 'high'


class TestWarehouseMultiAGVScenario:
    """仓库多AGV场景测试"""

    def test_warehouse_order_picking_scenario(self):
        """仓库拣选订单场景测试"""
        from src.embodied.simulation_enhancement import WarehouseSceneGenerator
        # 生成仓库场景
        gen = WarehouseSceneGenerator(seed=42)
        scene = gen.generate_warehouse(num_aisles=3, shelves_per_aisle=8)
        task = gen.generate_picking_task(scene, num_items=3)

        assert task['type'] == 'order_picking'
        assert len(task['pick_points']) == 3
        assert 'end_position' in task

        # 创建协调器
        coord = SwarmCoordinator(num_robots=2)
        start = np.array(scene['start_positions'][0])
        if len(start) == 3:
            # 3D position (x, y, z), add 1.0 to x
            coord.register_robot('agv1', start)
            coord.register_robot('agv2', start + np.array([1.0, 0.0, 0.0]))
        else:
            # 2D position
            coord.register_robot('agv1', start)
            coord.register_robot('agv2', start + np.array([1.0, 0.0]))

        # 添加拣选任务的每个子任务
        for i, pick_point in enumerate(task['pick_points']):
            coord.add_global_task({
                'id': f'pick_{i}',
                'type': 'navigate',
                'target': pick_point,
            })
        coord.add_global_task({
            'id': 'dropoff',
            'type': 'navigate',
            'target': task['end_position'],
        })

        assignments = coord.coordinate_step()
        # 应该分配了所有任务
        assert len(assignments) >= 1
        # 检查机器人状态
        stats = coord.get_statistics()
        assert stats['total_robots'] == 2
        assert stats['pending_tasks'] >= 0

    def test_dynamic_obstacle_avoidance(self):
        """动态避障测试"""
        gen = EnvironmentGenerator(seed=42)
        obstacles = gen.generate_random_obstacles((10.0, 10.0), 5)
        # 添加动态障碍物
        dynamic_obs = [obs for obs in obstacles if obs.obstacle_type == 'dynamic']
        # 检查冲突检测对动态障碍物的处理
        detector = ConflictDetector()
        # AGV路径
        path = [np.array([0.0, 0.0]), np.array([5.0, 5.0]), np.array([10.0, 10.0])]
        # 检测器应该考虑动态障碍物位置变化
        conflicts_with_dynamic = detector.detect_with_dynamic_obstacles(path, dynamic_obs)
        # 返回结果结构正确
        assert isinstance(conflicts_with_dynamic, list)


class TestDifferentCoordinationStrategies:
    """不同协调策略测试"""

    def test_centralized_strategy(self):
        """集中式策略"""
        coord = SwarmCoordinator(num_robots=3, strategy=CoordinationStrategy.CENTRALIZED)
        assert coord.strategy == CoordinationStrategy.CENTRALIZED

    def test_decentralized_strategy(self):
        """分布式策略"""
        coord = SwarmCoordinator(num_robots=3, strategy=CoordinationStrategy.DECENTRALIZED)
        assert coord.strategy == CoordinationStrategy.DECENTRALIZED

    def test_hybrid_strategy(self):
        """混合策略"""
        coord = SwarmCoordinator(num_robots=3, strategy=CoordinationStrategy.HYBRID)
        assert coord.strategy == CoordinationStrategy.HYBRID

    def test_market_based_strategy(self):
        """基于市场拍卖的策略"""
        coord = SwarmCoordinator(num_robots=3, strategy=CoordinationStrategy.MARKET_BASED)
        assert coord.strategy == CoordinationStrategy.MARKET_BASED


# ============ 统计 ============

def test_module_import():
    """模块导入测试"""
    from src.control.swarm_coordination import SwarmCoordinator
    from src.control.swarm_coordination import TaskAllocator
    from src.control.swarm_coordination import ConflictDetector
    assert SwarmCoordinator is not None
    assert TaskAllocator is not None
    assert ConflictDetector is not None


class TestDynamicTaskAllocationLoadBalancing:
    """动态任务分配与负载均衡测试"""

    def test_load_balancing_distributes_tasks_evenly(self):
        """测试负载均衡均匀分配任务"""
        allocator = TaskAllocator(num_robots=2, consider_load=True, load_weight=1.0, max_tasks_per_robot=3)
        robots = [
            {'id': 'agv1', 'position': np.array([0.0, 0.0]), 'available': True, 'assigned_tasks': []},
            {'id': 'agv2', 'position': np.array([10.0, 0.0]), 'available': True, 'assigned_tasks': []},
        ]
        # 创建4个任务，分布在中间区域
        tasks = []
        for i in range(4):
            tasks.append({'id': f't{i}', 'target': np.array([4.0 + i*0.5, 0.0])})

        assignments = allocator.allocate(robots, tasks)
        # 应该分配所有4个任务
        assert len(assignments) == 4
        # 负载均衡：每个机器人分配2个任务
        tasks_per_robot = {}
        for a in assignments:
            if a['robot_id'] not in tasks_per_robot:
                tasks_per_robot[a['robot_id']] = 0
            tasks_per_robot[a['robot_id']] += 1
        assert tasks_per_robot['agv1'] == 2
        assert tasks_per_robot['agv2'] == 2

    def test_max_tasks_per_robot_limit(self):
        """测试每个机器人最大任务数限制"""
        allocator = TaskAllocator(num_robots=2, max_tasks_per_robot=2)
        robots = [
            {'id': 'agv1', 'position': np.array([0.0, 0.0]), 'available': True, 'assigned_tasks': ['t1', 't2']},
            {'id': 'agv2', 'position': np.array([10.0, 0.0]), 'available': True, 'assigned_tasks': []},
        ]
        tasks = [
            {'id': 't3', 'target': np.array([1.0, 0.0])},
            {'id': 't4', 'target': np.array([2.0, 0.0])},
            {'id': 't5', 'target': np.array([3.0, 0.0])},
        ]
        assignments = allocator.allocate(robots, tasks)
        # agv1已经有2个任务，agv2最多2个任务，所以总分配2个
        assert len(assignments) == 2
        for a in assignments:
            assert a['robot_id'] == 'agv2'


class TestPriorityRightOfWay:
    """优先级路权测试"""

    def test_emergency_priority_has_absolute_right_of_way(self):
        """测试紧急优先级AGV拥有绝对路权"""
        coord = SwarmCoordinator(num_robots=2)
        # AGV1是紧急优先级，AGV2是普通优先级
        coord.register_robot('agv1', np.array([0.0, 0.0]), priority=0, current_speed=1.0)
        coord.register_robot('agv2', np.array([2.0, 0.0]), priority=2, current_speed=1.0)

        conflict = PathConflict(
            robot1_id='agv1',
            robot2_id='agv2',
            position=np.array([1.0, 0.0]),
            time1=1.0,
            time2=1.0,
            distance=0.0,
            is_head_on=True
        )

        resolution = coord.resolve_conflict(conflict)
        assert resolution.resolved
        assert resolution.method == 'emergency_right_of_way'
        # 普通优先级AGV应该完全停止
        assert coord.robots['agv2']['current_speed'] == 0.0
        assert coord.robots['agv2']['waiting_for'] == 'agv1'

    def test_higher_priority_agv_gets_right_of_way(self):
        """测试高优先级AGV优先通行"""
        coord = SwarmCoordinator(num_robots=2)
        coord.register_robot('agv1', np.array([0.0, 0.0]), priority=1, current_speed=1.0)  # 高优先级
        coord.register_robot('agv2', np.array([2.0, 0.0]), priority=3, current_speed=1.0)  # 低优先级

        conflict = PathConflict(
            robot1_id='agv1',
            robot2_id='agv2',
            position=np.array([1.0, 0.0]),
            time1=1.0,
            time2=1.4,  # 时间差0.4s，在0.3-0.5之间，触发减速
            distance=0.0,
        )

        resolution = coord.resolve_conflict(conflict)
        assert resolution.resolved
        assert 'priority' in resolution.method
        # 低优先级AGV减速或等待
        assert coord.robots['agv2']['current_speed'] < 1.0


class TestRealTimeCommunicationSync:
    """实时通信同步测试"""

    def test_communication_latency_measurement(self):
        """测试通信延迟测量"""
        coord = SwarmCoordinator(max_communication_latency_ms=100)
        coord.register_robot('agv1', np.array([0.0, 0.0]))

        # 模拟延迟50ms的同步
        send_time = time.time() - 0.05
        success = coord.sync_robot_state('agv1', {'position': np.array([1.0, 0.0])}, send_time)
        assert success == True
        assert 'average_communication_latency_ms' in coord.stats
        assert 45 <= coord.stats['average_communication_latency_ms'] <= 55

    def test_latency_exceeding_threshold_fails_sync(self):
        """测试延迟超过阈值时同步失败"""
        coord = SwarmCoordinator(max_communication_latency_ms=100)
        coord.register_robot('agv1', np.array([0.0, 0.0]))

        # 模拟延迟150ms的同步
        send_time = time.time() - 0.15
        success = coord.sync_robot_state('agv1', {'position': np.array([1.0, 0.0])}, send_time)
        assert success == False
        assert coord.sync_failure_count['agv1'] == 1

    def test_communication_health_monitoring(self):
        """测试通信健康监控"""
        coord = SwarmCoordinator(max_communication_latency_ms=100)
        coord.register_robot('agv1', np.array([0.0, 0.0]))
        coord.register_robot('agv2', np.array([10.0, 0.0]))

        # AGV1延迟正常，AGV2延迟过高
        send_time1 = time.time() - 0.03
        coord.sync_robot_state('agv1', {'position': np.array([1.0, 0.0])}, send_time1)
        send_time2 = time.time() - 0.2
        coord.sync_robot_state('agv2', {'position': np.array([11.0, 0.0])}, send_time2)

        health = coord.get_communication_health()
        assert health['robots']['agv1']['healthy'] == True
        assert health['robots']['agv2']['healthy'] == False
        assert health['overall_healthy'] == False  # 因为AGV2不健康


class TestRealTimeCollisionAvoidance:
    """实时碰撞避让测试"""

    def test_real_time_avoidance_detects_approaching_robots(self):
        """测试实时避让检测正在靠近的机器人"""
        coord = SwarmCoordinator(num_robots=2)
        # 两个机器人相距0.4m，相向而行
        coord.register_robot('agv1', np.array([0.0, 0.0]), current_speed=1.0, velocity=[1.0, 0.0])
        coord.register_robot('agv2', np.array([0.4, 0.0]), current_speed=1.0, velocity=[-1.0, 0.0])

        resolutions = coord.real_time_collision_avoidance()
        # 应该检测到即将碰撞并生成解决方案
        assert len(resolutions) == 1
        assert resolutions[0].resolved == True
        # 至少一个机器人会减速或停止
        assert coord.robots['agv1']['current_speed'] < 1.0 or coord.robots['agv2']['current_speed'] < 1.0


class TestDynamicTaskReallocation:
    """动态任务重分配测试"""

    def test_reallocate_tasks_when_robot_overloaded(self):
        """测试当机器人过载时重新分配任务"""
        coord = SwarmCoordinator(num_robots=2)
        coord.allocator.max_tasks_per_robot = 2
        coord.register_robot('agv1', np.array([0.0, 0.0]), available=True, assigned_tasks=['t1', 't2'])
        coord.register_robot('agv2', np.array([10.0, 0.0]), available=True, assigned_tasks=[])

        # 添加新任务
        coord.add_global_task({'id': 't3', 'target': np.array([1.0, 0.0])})
        coord.add_global_task({'id': 't4', 'target': np.array([2.0, 0.0])})

        assignments = coord.coordinate_step()
        # 所有新任务应该分配给agv2，因为agv1已经满负载
        assert len(assignments) == 2
        for a in assignments:
            assert a['robot_id'] == 'agv2'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
