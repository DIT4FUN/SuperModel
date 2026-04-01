"""
Gazebo 仿真模块测试
===================

测试 ROS2-Gazebo 联合仿真接口:
- GazeboROS2Bridge 话题桥接
- GazeboSimulator 仿真器
- AGVGazeboSimulator AGV专用仿真
- GazeboROS2Simulator 完整联合仿真器
- AGV五级等级配置
"""

import numpy as np
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from simulation.gazebo_sim import (
    GazeboROS2Bridge, GazeboSimulator, AGVGazeboSimulator,
    GazeboROS2Simulator, GazeboROS2Config, GazeboAGVSpec,
    GazeboWorld, HAS_ROS2, HAS_GZSIM
)


class TestGazeboROS2Config(unittest.TestCase):
    """测试配置类"""

    def test_default_config(self):
        cfg = GazeboROS2Config()
        self.assertEqual(cfg.namespace, "/supermodel")
        self.assertTrue(cfg.use_sim_time)
        self.assertEqual(cfg.world, GazeboWorld.EMPTY)

    def test_warehouse_config(self):
        cfg = GazeboROS2Config(
            namespace="/agv_01",
            world=GazeboWorld.WAREHOUSE,
            camera_topic="/front_camera/image_raw",
            imu_topic="/imu/data",
            lidar_topic="/scan",
        )
        self.assertEqual(cfg.namespace, "/agv_01")
        self.assertEqual(cfg.world, GazeboWorld.WAREHOUSE)
        self.assertEqual(cfg.camera_topic, "/front_camera/image_raw")


class TestGazeboAGVSpec(unittest.TestCase):
    """测试 AGV 规格"""

    def test_default_spec(self):
        spec = GazeboAGVSpec()
        self.assertEqual(spec.mass, 50.0)
        self.assertEqual(spec.wheelbase, 0.5)
        self.assertEqual(spec.max_linear, 2.0)
        self.assertEqual(spec.grade, "M")

    def test_spec_grade_s(self):
        spec = GazeboAGVSpec(grade="S", mass=30)
        self.assertEqual(spec.mass, 30)
        self.assertEqual(spec.grade, "S")

    def test_spec_grade_xxl(self):
        spec = GazeboAGVSpec(grade="XXL", mass=200, max_linear=5.0)
        self.assertEqual(spec.mass, 200)
        self.assertEqual(spec.max_linear, 5.0)


class TestGazeboROS2Bridge(unittest.TestCase):
    """测试 ROS2-Gazebo 桥接器"""

    def setUp(self):
        self.bridge = GazeboROS2Bridge()

    def test_init_no_ros2(self):
        bridge = GazeboROS2Bridge()
        self.assertIsNone(bridge._node)
        self.assertFalse(bridge._running)

    def test_initialize_without_ros2(self):
        bridge = GazeboROS2Bridge()
        result = bridge.initialize()
        # 在没有 ROS2 的环境中返回 False
        self.assertFalse(result)

    def test_shutdown(self):
        bridge = GazeboROS2Bridge()
        bridge.initialize()
        bridge.shutdown()
        self.assertFalse(bridge._running)

    def test_get_camera_image_no_data(self):
        img = self.bridge.get_camera_image()
        self.assertIsNone(img)

    def test_get_imu_data_no_data(self):
        imu = self.bridge.get_imu_data()
        self.assertIsNone(imu)

    def test_get_lidar_scan_no_data(self):
        scan = self.bridge.get_lidar_scan()
        self.assertIsNone(scan)

    def test_get_odometry_no_data(self):
        odom = self.bridge.get_odometry()
        self.assertIsNone(odom)

    def test_get_joint_states_no_data(self):
        joints = self.bridge.get_joint_states()
        self.assertIsNone(joints)

    def test_config(self):
        cfg = GazeboROS2Config(namespace="/test")
        bridge = GazeboROS2Bridge(cfg)
        self.assertEqual(bridge.config.namespace, "/test")


class TestGazeboSimulator(unittest.TestCase):
    """测试 Gazebo 仿真器"""

    def test_init_default(self):
        sim = GazeboSimulator()
        self.assertIsNotNone(sim.spec)
        self.assertIsNotNone(sim.config)
        self.assertIsNotNone(sim.bridge)

    def test_init_with_spec(self):
        spec = GazeboAGVSpec(mass=100, grade="XL")
        sim = GazeboSimulator(spec=spec)
        self.assertEqual(sim.spec.mass, 100)
        self.assertEqual(sim.spec.grade, "XL")

    def test_spawn_already_spawned(self):
        sim = GazeboSimulator()
        sim._spawned = True  # 模拟已生成
        result = sim.spawn()
        self.assertTrue(result)
        sim.kill()

    def test_spawn_initializes_bridge(self):
        sim = GazeboSimulator()
        sim.bridge.initialize = MagicMock(return_value=True)
        sim._subscribe_camera = MagicMock()
        sim._subscribe_imu = MagicMock()
        sim._subscribe_lidar = MagicMock()
        sim._subscribe_odom = MagicMock()
        sim._subscribe_joints = MagicMock()
        result = sim.spawn()
        self.assertTrue(result)
        self.assertTrue(sim._spawned)

    def test_kill(self):
        sim = GazeboSimulator()
        sim.spawn()
        sim.kill()
        self.assertFalse(sim._spawned)

    def test_set_velocity(self):
        sim = GazeboSimulator()
        sim.bridge.publish_cmd_vel = MagicMock()
        sim.set_velocity((0.5, 0.0, 0.0))
        sim.bridge.publish_cmd_vel.assert_called_once_with(0.5, 0.0, 0.0)


class TestAGVGazeboSimulator(unittest.TestCase):
    """测试 AGV 专用 Gazebo 仿真器"""

    def test_init(self):
        sim = AGVGazeboSimulator()
        self.assertEqual(sim._wb, sim.spec.wheelbase)
        self.assertEqual(sim._tw, sim.spec.track_width)
        self.assertEqual(sim._wr, sim.spec.wheel_radius)

    def test_differential_to_wheel_straight(self):
        sim = AGVGazeboSimulator()
        v_left, v_right = sim.differential_to_wheel(vx=1.0, omega=0.0)
        self.assertAlmostEqual(v_left, 1.0)
        self.assertAlmostEqual(v_right, 1.0)

    def test_differential_to_wheel_turn_left(self):
        sim = AGVGazeboSimulator()
        # track_width = 0.4, omega = 2.0 -> 左右轮速差 = 2*0.4 = 0.8
        v_left, v_right = sim.differential_to_wheel(vx=1.0, omega=2.0)
        self.assertAlmostEqual(v_right - v_left, 0.8)

    def test_wheel_to_differential_straight(self):
        sim = AGVGazeboSimulator()
        vx, omega = sim.wheel_to_differential(v_left=1.0, v_right=1.0)
        self.assertAlmostEqual(vx, 1.0)
        self.assertAlmostEqual(omega, 0.0)

    def test_wheel_to_differential_turn(self):
        sim = AGVGazeboSimulator()
        vx, omega = sim.wheel_to_differential(v_left=0.6, v_right=1.4)
        self.assertAlmostEqual(vx, 1.0)
        self.assertAlmostEqual(omega, 2.0)  # (1.4 - 0.6) / 0.4 = 2.0

    def test_differential_roundtrip(self):
        sim = AGVGazeboSimulator()
        vx, omega = 1.5, 0.5
        v_left, v_right = sim.differential_to_wheel(vx, omega)
        vx2, omega2 = sim.wheel_to_differential(v_left, v_right)
        self.assertAlmostEqual(vx, vx2)
        self.assertAlmostEqual(omega, omega2)

    def test_step_kinematic(self):
        sim = AGVGazeboSimulator()
        sim.get_twist = MagicMock(return_value=(1.0, 0.0, 0.0))
        sim._odom_x = 0.0
        sim._odom_y = 0.0
        sim._odom_yaw = 0.0
        sim.step(dt=0.1)
        self.assertAlmostEqual(sim._odom_x, 0.1, places=5)
        self.assertAlmostEqual(sim._odom_y, 0.0, places=5)

    def test_step_turn(self):
        sim = AGVGazeboSimulator()
        sim.get_twist = MagicMock(return_value=(0.0, 0.0, 1.0))
        sim._odom_x = 0.0
        sim._odom_y = 0.0
        sim._odom_yaw = 0.0
        sim.step(dt=0.1)
        self.assertAlmostEqual(sim._odom_yaw, 0.1, places=5)

    def test_step_no_twist(self):
        sim = AGVGazeboSimulator()
        sim.get_twist = MagicMock(return_value=None)
        sim._odom_x = 1.0
        sim._odom_y = 1.0
        sim._odom_yaw = 0.5
        sim.step(dt=0.1)
        # No motion when twist is None
        self.assertAlmostEqual(sim._odom_x, 1.0)

    def test_get_pose_no_odom(self):
        sim = AGVGazeboSimulator()
        sim.bridge.get_odometry = MagicMock(return_value=None)
        sim._odom_x = 1.0
        sim._odom_y = 2.0
        sim._odom_yaw = 0.5
        pose = sim.get_pose()
        np.testing.assert_array_almost_equal(pose, [1.0, 2.0, 0.5])

    def test_for_grade_s(self):
        sim = AGVGazeboSimulator.for_grade('S')
        self.assertEqual(sim.spec.mass, 30)
        self.assertEqual(sim.spec.max_linear, 1.0)
        self.assertFalse(sim.spec.lidar_enabled)

    def test_for_grade_m(self):
        sim = AGVGazeboSimulator.for_grade('M')
        self.assertEqual(sim.spec.mass, 50)
        self.assertTrue(sim.spec.lidar_enabled)

    def test_for_grade_xl(self):
        sim = AGVGazeboSimulator.for_grade('XL')
        self.assertEqual(sim.spec.mass, 120)
        self.assertEqual(sim.spec.max_linear, 4.0)

    def test_for_grade_xxl(self):
        sim = AGVGazeboSimulator.for_grade('XXL')
        self.assertEqual(sim.spec.mass, 200)
        self.assertEqual(sim.spec.max_linear, 5.0)

    def test_context_manager(self):
        sim = AGVGazeboSimulator()
        sim.bridge.initialize = MagicMock(return_value=True)
        with sim as s:
            self.assertTrue(sim._spawned)
        self.assertFalse(sim._spawned)


class TestGazeboROS2Simulator(unittest.TestCase):
    """测试完整 ROS2+Gazebo 仿真器"""

    def test_init(self):
        sim = GazeboROS2Simulator()
        self.assertIsInstance(sim, GazeboSimulator)
        self.assertEqual(len(sim._action_clients), 0)

    def test_navigate_to(self):
        sim = GazeboROS2Simulator()
        # 异步测试 (run_in_executor 不实际执行)
        result = sim.navigate_to(1.0, 2.0, 0.5)
        # No ROS2 so it returns True by default
        self.assertTrue(result)

    def test_grasp_at(self):
        sim = GazeboROS2Simulator()
        result = sim.grasp_at(0.5, 0.0, 0.1)
        self.assertTrue(result)

    def test_reset(self):
        sim = GazeboROS2Simulator()
        sim.set_velocity = MagicMock()
        sim.reset()
        sim.set_velocity.assert_called_once_with((0.0, 0.0, 0.0))

    def test_reset_with_pose(self):
        sim = GazeboROS2Simulator()
        sim.reset(pose=(1.0, 2.0, 0.5))


class TestGazeboWorld(unittest.TestCase):
    """测试 Gazebo 世界枚举"""

    def test_world_values(self):
        self.assertEqual(GazeboWorld.EMPTY.value, "empty")
        self.assertEqual(GazeboWorld.WAREHOUSE.value, "warehouse")
        self.assertEqual(GazeboWorld.OFFICE.value, "office")
        self.assertEqual(GazeboWorld.INDUSTRIAL.value, "industrial")

    def test_world_in_config(self):
        cfg = GazeboROS2Config(world=GazeboWorld.INDUSTRIAL)
        self.assertEqual(cfg.world, GazeboWorld.INDUSTRIAL)


if __name__ == '__main__':
    unittest.main()
