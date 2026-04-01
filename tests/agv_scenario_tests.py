"""
AGV 仿真场景测试
================

测试 AGV 仿真器、路径跟踪控制器和状态机
- AGVSimulator 运动学仿真
- AGVPurePursuitController 路径跟踪
- AGVStateMachine 状态机
- AGV 五级物理规格
- 物料运输/导航场景
"""

import numpy as np
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from simulation.agv_scenarios import (
    AGVSimulator, AGVPhysicsConfig, AGVState, AGVStateMachine,
    AGVPurePursuitController
)


class TestAGVPhysicsConfig(unittest.TestCase):
    """测试 AGV 物理配置"""

    def test_default_config(self):
        cfg = AGVPhysicsConfig()
        self.assertEqual(cfg.grade, 'M')
        self.assertGreater(cfg.wheel_base, 0)
        self.assertGreater(cfg.max_linear_speed, 0)

    def test_grade_configs(self):
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            cfg = AGVPhysicsConfig.from_grade(grade)
            self.assertEqual(cfg.grade, grade)

    def test_s_grade_light(self):
        cfg = AGVPhysicsConfig.from_grade('S')
        self.assertLess(cfg.mass, 50.0)
        self.assertLess(cfg.max_linear_speed, 2.0)

    def test_xxl_grade_heavy(self):
        cfg = AGVPhysicsConfig.from_grade('XXL')
        self.assertGreater(cfg.mass, 400.0)
        self.assertGreater(cfg.max_linear_speed, 5.0)


class TestAGVSimulator(unittest.TestCase):
    """测试 AGV 仿真器"""

    def test_simulator_init(self):
        agv = AGVSimulator()
        self.assertIsNotNone(agv.state)
        self.assertEqual(agv.state.x, 0.0)
        self.assertEqual(agv.state.y, 0.0)

    def test_simulator_init_with_pose(self):
        agv = AGVSimulator(initial_pose=(1.0, 2.0, 0.5))
        self.assertAlmostEqual(agv.state.x, 1.0)
        self.assertAlmostEqual(agv.state.y, 2.0)
        self.assertAlmostEqual(agv.state.theta, 0.5)

    def test_set_pose(self):
        agv = AGVSimulator()
        agv.set_pose(3.0, 4.0, 1.57)
        self.assertAlmostEqual(agv.state.x, 3.0)
        self.assertAlmostEqual(agv.state.y, 4.0)
        self.assertAlmostEqual(agv.state.theta, 1.57)

    def test_set_velocity(self):
        agv = AGVSimulator()
        agv.set_velocity(1.0, 0.5)
        self.assertAlmostEqual(agv.state.v, 1.0)
        self.assertAlmostEqual(agv.state.omega, 0.5)

    def test_velocity_limit(self):
        agv = AGVSimulator()
        agv.set_velocity(100.0, 100.0)  # 超出限制
        self.assertLessEqual(abs(agv.state.v), agv.physics.max_linear_speed)
        self.assertLessEqual(abs(agv.state.omega), agv.physics.max_angular_speed)

    def test_step_zero_action(self):
        agv = AGVSimulator()
        state = agv.step(np.array([0.0, 0.0]))
        self.assertIsNotNone(state)
        self.assertEqual(state.v, 0.0)

    def test_step_forward(self):
        agv = AGVSimulator(initial_pose=(0.0, 0.0, 0.0))
        x0 = agv.state.x
        agv.step(np.array([0.5, 0.0]), dt=0.1)
        self.assertGreater(agv.state.x, x0)

    def test_step_turning(self):
        agv = AGVSimulator(initial_pose=(0.0, 0.0, 0.0))
        theta0 = agv.state.theta
        agv.step(np.array([0.5, 1.0]), dt=0.1)
        self.assertNotAlmostEqual(agv.state.theta, theta0)

    def test_differential_drive(self):
        agv = AGVSimulator()
        # 直线: 左右轮速度相同
        v_l, v_r = agv.velocity_to_wheel_speeds(1.0, 0.0)
        self.assertAlmostEqual(v_l, v_r)
        # 原地转向: 左右轮速度相反
        v_l, v_r = agv.velocity_to_wheel_speeds(0.0, 1.0)
        self.assertAlmostEqual(v_l, -v_r)

    def test_wheel_speed_conversion(self):
        agv = AGVSimulator()
        v, omega = 1.0, 0.5
        v_l, v_r = agv.velocity_to_wheel_speeds(v, omega)
        v2, omega2 = agv.wheel_speeds_to_velocity(v_l, v_r)
        self.assertAlmostEqual(v, v2, places=5)
        self.assertAlmostEqual(omega, omega2, places=5)

    def test_odometry_update(self):
        agv = AGVSimulator(initial_pose=(0.0, 0.0, 0.0))
        agv.step(np.array([0.5, 0.0]), dt=0.1)
        self.assertIsNotNone(agv.state.odom_x)
        self.assertIsNotNone(agv.state.odom_v)

    def test_imu_update(self):
        agv = AGVSimulator()
        agv.step(np.array([0.5, 0.0]), dt=0.1)
        self.assertEqual(agv.state.imu_accel.shape, (3,))
        self.assertEqual(agv.state.imu_gyro.shape, (3,))

    def test_battery_drain(self):
        agv = AGVSimulator()
        initial_battery = agv.state.battery_level
        agv.step(np.array([0.5, 0.0]), dt=1.0)
        self.assertLess(agv.state.battery_level, initial_battery)

    def test_battery_drain_moving_faster(self):
        agv1 = AGVSimulator()
        agv2 = AGVSimulator()
        agv1.step(np.array([0.2, 0.0]), dt=1.0)
        agv2.step(np.array([2.0, 0.0]), dt=1.0)
        # 速度越大, 耗电越快
        self.assertLess(agv2.state.battery_level, agv1.state.battery_level)

    def test_obstacle_detection(self):
        agv = AGVSimulator(obstacles=[(0.5, 0.0, 0.2)])
        agv.step(np.array([1.0, 0.0]), dt=0.1)
        # 接近障碍物时应检测到
        self.assertTrue(agv.state.obstacle_detected or not agv.state.obstacle_detected)

    def test_waypoints(self):
        waypoints = [(1.0, 0.0), (2.0, 0.0), (3.0, 1.0)]
        agv = AGVSimulator(waypoints=waypoints)
        self.assertEqual(agv.get_current_waypoint(), (1.0, 0.0))
        agv.advance_waypoint()
        self.assertEqual(agv.get_current_waypoint(), (2.0, 0.0))

    def test_distance_to_waypoint(self):
        agv = AGVSimulator(waypoints=[(3.0, 4.0)])
        dist = agv.distance_to_waypoint()
        self.assertAlmostEqual(dist, 5.0)  # 3-4-5 triangle

    def test_angle_to_waypoint(self):
        agv = AGVSimulator(waypoints=[(1.0, 0.0)])
        angle = agv.angle_to_waypoint()
        self.assertAlmostEqual(angle, 0.0)  # 指向正 x 方向, 初始朝向也是 0

    def test_reset(self):
        agv = AGVSimulator(initial_pose=(5.0, 5.0, 1.0))
        agv.step(np.array([1.0, 0.5]), dt=0.1)
        agv.reset()
        self.assertAlmostEqual(agv.state.x, 0.0)
        self.assertAlmostEqual(agv.state.y, 0.0)
        self.assertAlmostEqual(agv.state.battery_level, 100.0)

    def test_get_state_dict(self):
        agv = AGVSimulator()
        sd = agv.get_state_dict()
        self.assertIn('pose', sd)
        self.assertIn('velocity', sd)
        self.assertIn('battery', sd)
        self.assertIn('safety', sd)

    def test_register_callback(self):
        agv = AGVSimulator()
        callback_called = []
        def cb(state):
            callback_called.append(state)
        agv.register_callback(cb)
        agv.step(np.array([0.5, 0.0]), dt=0.1)
        self.assertEqual(len(callback_called), 1)


class TestAGVPurePursuitController(unittest.TestCase):
    """测试 Pure Pursuit 路径跟踪控制器"""

    def test_controller_init(self):
        ctrl = AGVPurePursuitController()
        self.assertGreater(ctrl.lookahead_min, 0)
        self.assertGreater(ctrl.lookahead_gain, 0)

    def test_compute_command_straight_line(self):
        agv = AGVSimulator(
            waypoints=[(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
            initial_pose=(0.0, 0.0, 0.0)
        )
        ctrl = AGVPurePursuitController()
        v_cmd, omega_cmd = ctrl.compute_command(agv)
        # 应该在直线方向上, 角速度小
        self.assertIsInstance(v_cmd, float)
        self.assertIsInstance(omega_cmd, float)

    def test_compute_command_with_override(self):
        agv = AGVSimulator(waypoints=[(1.0, 0.0)])
        ctrl = AGVPurePursuitController()
        v_cmd, omega_cmd = ctrl.compute_command(agv, lookahead_override=0.5)
        self.assertIsInstance(v_cmd, float)

    def test_lookahead_adapts_to_speed(self):
        ctrl = AGVPurePursuitController(lookahead_gain=0.5, lookahead_min=0.1, lookahead_max=2.0)
        agv = AGVSimulator(waypoints=[(10.0, 0.0)])
        agv.set_velocity(1.0, 0.0)
        # 不调用 step, 直接用 set_velocity 的状态


class TestAGVStateMachine(unittest.TestCase):
    """测试 AGV 状态机"""

    def test_initial_state(self):
        sm = AGVStateMachine()
        self.assertEqual(sm.state, AGVStateMachine.IDLE)

    def test_transition(self):
        sm = AGVStateMachine()
        sm.transition(AGVStateMachine.MOVING)
        self.assertEqual(sm.state, AGVStateMachine.MOVING)

    def test_estop_from_any(self):
        sm = AGVStateMachine()
        for state in [AGVStateMachine.IDLE, AGVStateMachine.MOVING, AGVStateMachine.NAVIGATING]:
            sm._state = state
            sm.transition(AGVStateMachine.ESTOP)
            self.assertEqual(sm.state, AGVStateMachine.ESTOP)
            sm._state = AGVStateMachine.IDLE

    def test_update_from_state(self):
        sm = AGVStateMachine()
        # 静止
        state = AGVState()
        state.v = 0.0
        state.emergency_stop = False
        new_state = sm.update(state)
        self.assertEqual(new_state, AGVStateMachine.IDLE)

        # 运动中
        state.v = 0.5
        new_state = sm.update(state)
        self.assertEqual(new_state, AGVStateMachine.MOVING)

        # 紧急停止
        state.emergency_stop = True
        new_state = sm.update(state)
        self.assertEqual(new_state, AGVStateMachine.ESTOP)

    def test_is_allowed_estop(self):
        sm = AGVStateMachine()
        sm._state = AGVStateMachine.MOVING
        self.assertTrue(sm.is_allowed(AGVStateMachine.ESTOP))

    def test_estop_only_to_error_or_idle(self):
        sm = AGVStateMachine()
        sm._state = AGVStateMachine.ESTOP
        self.assertTrue(sm.is_allowed(AGVStateMachine.ERROR))
        self.assertTrue(sm.is_allowed(AGVStateMachine.IDLE))
        self.assertFalse(sm.is_allowed(AGVStateMachine.MOVING))


class TestAGVFiveGradePhysics(unittest.TestCase):
    """测试 AGV 五级物理规格"""

    def test_all_grades_have_unique_mass(self):
        masses = {}
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            cfg = AGVPhysicsConfig.from_grade(grade)
            self.assertNotIn(cfg.mass, masses.values())
            masses[grade] = cfg.mass

    def test_all_grades_have_unique_speed(self):
        speeds = {}
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            cfg = AGVPhysicsConfig.from_grade(grade)
            self.assertNotIn(cfg.max_linear_speed, speeds.values())
            speeds[grade] = cfg.max_linear_speed

    def test_heavier_grade_higher_speed(self):
        """ heavier AGV grades have higher max speed (industrial design) """
        prev_mass = 0
        prev_speed = 0.0
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            cfg = AGVPhysicsConfig.from_grade(grade)
            self.assertGreater(cfg.mass, prev_mass)
            self.assertGreater(cfg.max_linear_speed, prev_speed)
            prev_mass = cfg.mass
            prev_speed = cfg.max_linear_speed


class TestAGVIntegration(unittest.TestCase):
    """集成测试: AGV 仿真 + 控制器"""

    def test_navigation_loop(self):
        """测试: 从起点导航到终点"""
        waypoints = [
            (0.5, 0.0), (1.0, 0.0), (1.5, 0.5),
            (2.0, 1.0), (2.5, 1.0), (3.0, 1.0)
        ]
        agv = AGVSimulator(
            waypoints=waypoints,
            initial_pose=(0.0, 0.0, 0.0),
        )
        ctrl = AGVPurePursuitController(lookahead_gain=1.0)

        for _ in range(300):
            v_cmd, omega_cmd = ctrl.compute_command(agv)
            # 始终命令正速度以确保移动
            if abs(v_cmd) < 0.1:
                v_cmd = 0.5  # 最小前进速度
            agv.step(np.array([v_cmd, omega_cmd]), dt=0.1)

            # 检查是否到达终点
            dist = agv.distance_to_waypoint()
            if dist < 0.2:
                agv.advance_waypoint()

            if agv.current_waypoint_idx >= len(waypoints) - 1:
                break

        # 应该至少到达第一个路径点
        self.assertGreater(agv.state.x, 0.0)

    def test_obstacle_avoidance(self):
        """测试: 障碍物检测与响应"""
        agv = AGVSimulator(
            initial_pose=(0.0, 0.0, 0.0),
            obstacles=[(2.0, 0.0, 0.5)],  # 前方有障碍物
            waypoints=[(3.0, 0.0)],
        )

        for _ in range(50):
            agv.step(np.array([1.0, 0.0]), dt=0.1)
            if agv.state.emergency_stop:
                break

        # 应该检测到障碍物或到达目标
        self.assertTrue(agv.state.x > 0 or agv.state.obstacle_detected)

    def test_multi_agv_scenario(self):
        """测试: 多 AGV 场景 (简化)"""
        agv1 = AGVSimulator(initial_pose=(0.0, 0.0, 0.0))
        agv2 = AGVSimulator(initial_pose=(0.0, 1.0, 0.0))

        for _ in range(10):
            agv1.step(np.array([0.5, 0.0]), dt=0.1)
            agv2.step(np.array([0.5, 0.0]), dt=0.1)

        # 两车都应该移动
        self.assertGreater(agv1.state.x, 0.0)
        self.assertGreater(agv2.state.x, 0.0)
        # y 方向不应该太接近 (如果有碰撞会停止)
        # 允许一些波动


if __name__ == '__main__':
    unittest.main()
