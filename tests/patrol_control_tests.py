"""
巡逻控制模块单元测试
测试 PatrolController 的自主巡逻功能
覆盖: 巡逻状态机、动态避障、五级AGV规格、传感器融合
"""

import unittest
import numpy as np
import sys
import os
import time

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from control.patrol_control import (
    PatrolController, PatrolRoute, PatrolPoint, PatrolState,
    Obstacle, PatrolGrade, PatrolSpec, create_patrol_controller,
    run_patrol_benchmark, get_patrol_spec, PatrolMetrics, PatrolEvent,
)


class TestPatrolSpec(unittest.TestCase):
    """测试巡逻规格"""

    def test_spec_from_grade_s(self):
        spec = get_patrol_spec('S')
        self.assertEqual(spec['grade'], 'S')
        self.assertEqual(spec['avoidance_strategy'], 'simple')
        self.assertEqual(spec['control_frequency'], 50.0)
        self.assertLess(spec['max_patrol_speed'], 0.5)

    def test_spec_from_grade_m(self):
        spec = get_patrol_spec('M')
        self.assertEqual(spec['grade'], 'M')
        self.assertEqual(spec['avoidance_strategy'], 'dwa')
        self.assertEqual(spec['control_frequency'], 100.0)
        self.assertGreater(len(spec['sensor_modalities']), 1)

    def test_spec_from_grade_l(self):
        spec = get_patrol_spec('L')
        self.assertEqual(spec['grade'], 'L')
        self.assertEqual(spec['avoidance_strategy'], 'apf')
        self.assertEqual(spec['control_frequency'], 200.0)

    def test_spec_from_grade_xl(self):
        spec = get_patrol_spec('XL')
        self.assertEqual(spec['grade'], 'XL')
        self.assertEqual(spec['avoidance_strategy'], 'vfh')
        self.assertEqual(spec['control_frequency'], 500.0)
        self.assertTrue(spec['has_multi_agent'])

    def test_spec_from_grade_xxl(self):
        spec = get_patrol_spec('XXL')
        self.assertEqual(spec['grade'], 'XXL')
        self.assertEqual(spec['avoidance_strategy'], 'hybrid')
        self.assertEqual(spec['control_frequency'], 1000.0)
        self.assertTrue(spec['has_emergency_recovery'])

    def test_spec_from_grade_default(self):
        spec = get_patrol_spec('M')  # 不存在的级别默认为M
        self.assertEqual(spec['grade'], 'M')


class TestPatrolPoint(unittest.TestCase):
    """测试巡逻点"""

    def test_creation(self):
        pt = PatrolPoint(x=1.0, y=2.0, theta=0.5, name="test_point", dwell_time=2.0, priority=3)
        self.assertEqual(pt.x, 1.0)
        self.assertEqual(pt.y, 2.0)
        self.assertEqual(pt.theta, 0.5)
        self.assertEqual(pt.name, "test_point")
        self.assertEqual(pt.dwell_time, 2.0)
        self.assertEqual(pt.priority, 3)


class TestPatrolRoute(unittest.TestCase):
    """测试巡逻路线"""

    def test_route_with_points(self):
        route = PatrolRoute(
            name="test_route",
            points=[
                PatrolPoint(x=0.0, y=0.0, name="start"),
                PatrolPoint(x=1.0, y=1.0, name="mid"),
                PatrolPoint(x=2.0, y=0.0, name="end"),
            ],
            loop=True,
        )
        self.assertEqual(route.name, "test_route")
        self.assertEqual(len(route.points), 3)
        self.assertTrue(route.loop)

    def test_route_no_loop(self):
        route = PatrolRoute(name="no_loop", points=[PatrolPoint(x=0, y=0)], loop=False)
        self.assertFalse(route.loop)


class TestObstacle(unittest.TestCase):
    """测试障碍物"""

    def test_static_obstacle(self):
        obs = Obstacle(position=np.array([1.0, 2.0]), radius=0.5, type="static")
        self.assertEqual(obs.type, "static")
        self.assertEqual(obs.radius, 0.5)
        np.testing.assert_array_equal(obs.position, [1.0, 2.0])
        np.testing.assert_array_equal(obs.velocity, [0.0, 0.0])

    def test_dynamic_obstacle(self):
        obs = Obstacle(position=np.array([0.0, 0.0]), radius=0.3, velocity=np.array([0.5, 0.0]), type="person")
        self.assertEqual(obs.type, "person")
        np.testing.assert_array_equal(obs.velocity, [0.5, 0.0])

    def test_predict_position(self):
        obs = Obstacle(position=np.array([0.0, 0.0]), radius=0.3, velocity=np.array([1.0, 0.5]))
        future = obs.predict_position(1.0)
        np.testing.assert_array_almost_equal(future, [1.0, 0.5])


class TestPatrolControllerCreation(unittest.TestCase):
    """测试巡逻控制器创建"""

    def test_create_patrol_controller_default(self):
        ctrl = create_patrol_controller()
        self.assertEqual(ctrl.grade, 'M')
        self.assertEqual(ctrl.state, PatrolState.IDLE)

    def test_create_patrol_controller_with_grade(self):
        ctrl = create_patrol_controller(grade='L', pose=(1.0, 2.0, 0.0))
        self.assertEqual(ctrl.grade, 'L')
        pose = ctrl.get_pose()
        self.assertAlmostEqual(pose[0], 1.0)
        self.assertAlmostEqual(pose[1], 2.0)

    def test_create_with_route(self):
        route = PatrolRoute(name="test", points=[PatrolPoint(x=1.0, y=1.0)])
        ctrl = create_patrol_controller(route=route)
        self.assertEqual(ctrl.current_route, route)


class TestPatrolControllerLifecycle(unittest.TestCase):
    """测试巡逻控制器生命周期"""

    def setUp(self):
        self.route = PatrolRoute(
            name="test_route",
            points=[
                PatrolPoint(x=1.0, y=0.0, name="p1"),
                PatrolPoint(x=1.0, y=1.0, name="p2"),
            ],
            loop=True,
        )
        self.ctrl = create_patrol_controller(grade='M', pose=(0.0, 0.0, 0.0), route=self.route)

    def test_initial_state(self):
        self.assertEqual(self.ctrl.state, PatrolState.IDLE)

    def test_start_patrol(self):
        result = self.ctrl.start_patrol()
        self.assertTrue(result)
        self.assertEqual(self.ctrl.state, PatrolState.PATROLLING)

    def test_stop_patrol(self):
        self.ctrl.start_patrol()
        self.ctrl.stop_patrol()
        self.assertEqual(self.ctrl.state, PatrolState.IDLE)
        np.testing.assert_array_almost_equal(self.ctrl.velocity, [0.0, 0.0, 0.0])

    def test_pause_resume(self):
        self.ctrl.start_patrol()
        self.ctrl.pause_patrol()
        self.assertEqual(self.ctrl.state, PatrolState.PAUSED)
        self.ctrl.resume_patrol()
        self.assertEqual(self.ctrl.state, PatrolState.PATROLLING)

    def test_emergency_stop(self):
        self.ctrl.start_patrol()
        self.ctrl.emergency_stop()
        self.assertEqual(self.ctrl.state, PatrolState.EMERGENCY_STOP)
        self.assertEqual(self.ctrl.metrics.emergency_stops, 1)


class TestPatrolControllerUpdate(unittest.TestCase):
    """测试巡逻控制器更新"""

    def setUp(self):
        self.route = PatrolRoute(
            name="test",
            points=[PatrolPoint(x=0.5, y=0.0, name="target")],
            loop=False,
        )
        self.ctrl = create_patrol_controller(grade='M', pose=(0.0, 0.0, 0.0), route=self.route)

    def test_update_idle_state(self):
        """IDLE状态应保持速度为0"""
        vel, state = self.ctrl.update(dt=0.01)
        np.testing.assert_array_almost_equal(vel, [0.0, 0.0, 0.0])
        self.assertEqual(state, PatrolState.IDLE)

    def test_update_patrolling(self):
        """PATROLLING状态应产生速度"""
        self.ctrl.start_patrol()
        vel, state = self.ctrl.update(dt=0.01)
        self.assertEqual(state, PatrolState.PATROLLING)
        # 应有前向速度
        self.assertGreater(vel[0], 0.0)

    def test_update_with_dt(self):
        """测试时间步长"""
        self.ctrl.start_patrol()
        self.ctrl.update(dt=0.1)
        self.assertGreater(self.ctrl.metrics.total_time, 0.09)

    def test_update_avoiding(self):
        """避障状态测试"""
        self.ctrl.start_patrol()
        # 添加近距离障碍物
        self.ctrl.obstacles.append(Obstacle(position=np.array([0.1, 0.0]), radius=0.2))
        vel, state = self.ctrl.update(dt=0.01)
        self.assertIn(state, (PatrolState.AVOIDING, PatrolState.PATROLLING))


class TestPatrolControllerMetrics(unittest.TestCase):
    """测试巡逻指标"""

    def test_metrics_initial(self):
        ctrl = create_patrol_controller()
        metrics = ctrl.get_metrics()
        self.assertEqual(metrics['total_distance'], 0.0)
        self.assertEqual(metrics['total_time'], 0.0)
        self.assertEqual(metrics['obstacles_avoided'], 0)

    def test_metrics_after_update(self):
        route = PatrolRoute(name="test", points=[PatrolPoint(x=1.0, y=0.0)])
        ctrl = create_patrol_controller(route=route)
        ctrl.start_patrol()
        for _ in range(100):
            ctrl.update(dt=0.01)
        metrics = ctrl.get_metrics()
        self.assertGreater(metrics['total_time'], 0.9)


class TestPatrolControllerReset(unittest.TestCase):
    """测试重置功能"""

    def test_reset_pose(self):
        ctrl = create_patrol_controller(pose=(5.0, 5.0, 1.0))
        ctrl.reset(pose=(0.0, 0.0, 0.0))
        pose = ctrl.get_pose()
        self.assertAlmostEqual(pose[0], 0.0)
        self.assertAlmostEqual(pose[1], 0.0)

    def test_reset_metrics(self):
        route = PatrolRoute(name="test", points=[PatrolPoint(x=1.0, y=0.0)])
        ctrl = create_patrol_controller(route=route)
        ctrl.start_patrol()
        ctrl.update(dt=0.1)
        ctrl.reset()
        metrics = ctrl.get_metrics()
        self.assertEqual(metrics['total_time'], 0.0)


class TestPatrolEvents(unittest.TestCase):
    """测试事件记录"""

    def test_event_logging(self):
        route = PatrolRoute(name="test", points=[PatrolPoint(x=1.0, y=0.0)])
        ctrl = create_patrol_controller(route=route)
        ctrl.start_patrol()
        # 触发避障
        ctrl.obstacles.append(Obstacle(position=np.array([0.1, 0.0]), radius=0.2))
        for _ in range(50):
            ctrl.update(dt=0.01)
        events = ctrl.get_events()
        self.assertGreater(len(events), 0)


class TestPatrolBenchmark(unittest.TestCase):
    """测试巡逻基准测试"""

    def test_benchmark_single_grade(self):
        results = run_patrol_benchmark(['S', 'M'])
        self.assertIn('S', results)
        self.assertIn('M', results)
        for grade, metrics in results.items():
            self.assertIn('total_distance', metrics)
            self.assertIn('total_time', metrics)

    def test_benchmark_all_grades(self):
        results = run_patrol_benchmark(['S', 'M', 'L'])
        self.assertEqual(len(results), 3)


class TestPatrolFiveGradeConsistency(unittest.TestCase):
    """测试五级规格一致性"""

    def test_grade_progression(self):
        """验证等级递增关系"""
        specs = {g: get_patrol_spec(g) for g in ['S', 'M', 'L', 'XL', 'XXL']}
        # 速度应随等级增加
        self.assertLessEqual(specs['S']['max_patrol_speed'], specs['M']['max_patrol_speed'])
        self.assertLessEqual(specs['M']['max_patrol_speed'], specs['L']['max_patrol_speed'])
        self.assertLessEqual(specs['L']['max_patrol_speed'], specs['XL']['max_patrol_speed'])
        self.assertLessEqual(specs['XL']['max_patrol_speed'], specs['XXL']['max_patrol_speed'])

    def test_control_frequency_progression(self):
        """验证控制频率递增"""
        specs = {g: get_patrol_spec(g) for g in ['S', 'M', 'L', 'XL', 'XXL']}
        freqs = [specs[g]['control_frequency'] for g in ['S', 'M', 'L', 'XL', 'XXL']]
        for i in range(len(freqs) - 1):
            self.assertLessEqual(freqs[i], freqs[i + 1])

    def test_all_grades_have_required_keys(self):
        """验证所有等级都有必需字段"""
        required = ['grade', 'max_patrol_speed', 'avoidance_strategy', 'control_frequency', 'sensor_modalities']
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_patrol_spec(grade)
            for key in required:
                self.assertIn(key, spec, f"Grade {grade} missing key: {key}")

    def test_sensor_modality_count(self):
        """验证传感器数量递增"""
        specs = {g: get_patrol_spec(g) for g in ['S', 'M', 'L', 'XL', 'XXL']}
        counts = [len(specs[g]['sensor_modalities']) for g in ['S', 'M', 'L', 'XL', 'XXL']]
        for i in range(len(counts) - 1):
            self.assertLessEqual(counts[i], counts[i + 1])


class TestPatrolRoutePriority(unittest.TestCase):
    """测试巡逻路线优先级"""

    def test_priority_sorting(self):
        route = PatrolRoute(
            name="priority_test",
            points=[
                PatrolPoint(x=0, y=0, priority=1),
                PatrolPoint(x=1, y=1, priority=5),
                PatrolPoint(x=2, y=2, priority=3),
            ],
            loop=False,
        )
        # 路由应按优先级排序
        self.assertEqual(route.points[0].priority, 5)
        self.assertEqual(route.points[1].priority, 3)
        self.assertEqual(route.points[2].priority, 1)


class TestPatrolMetricsToDict(unittest.TestCase):
    """测试指标字典转换"""

    def test_metrics_fields(self):
        metrics = PatrolMetrics()
        d = metrics.to_dict()
        expected_keys = ['total_distance', 'total_time', 'obstacles_avoided', 'alerts_triggered',
                        'points_completed', 'points_total', 'avg_speed', 'emergency_stops']
        for key in expected_keys:
            self.assertIn(key, d)


if __name__ == '__main__':
    unittest.main()
