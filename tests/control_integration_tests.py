"""
控制模块集成测试
================
测试传感器 → 融合 → 控制 → 执行器的完整闭环

覆盖:
- TactileServoController + TactileArray
- ForceController + ForceTorqueSensor  
- IMU-based pose estimation + trajectory tracking
- Cross-modal fusion with control pipeline
"""

import unittest
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sensors.tactile import (
    TactileArray, TactileFrame, TactileSensorType,
    VirtualTactileSensor, TactileContact
)
from src.sensors.force import (
    ForceTorqueSensor, Wrench, ForceSensorType,
    VirtualForceSensor
)
from src.sensors.imu import (
    IMUSensor, IMUFrame, Pose, PoseEstimator, IMUSensorType,
    VirtualIMUSensor
)
from src.control.tactile_control import (
    TactileServoController, TactileServoParams, GraspQualityController
)
from src.control.force_control import (
    ForceController, ForceControlParams, HybridForcePositionController
)
from src.control.imu_control import (
    AttitudeStabilizer, IMUControlParams, MotionEstimator
)
from src.control.agv import (
    AGVMotionController, AGVPose, AGVTwist, DriveType, AGVGrade, AGVSpec
)


class TestTactileServoController(unittest.TestCase):
    """触觉伺服控制器测试"""

    def setUp(self):
        self.tactile = TactileArray(
            array_size=(16, 16),
            sensor_type=TactileSensorType.CAPACITIVE,
            sensor_id="test_tactile_ctrl"
        )
        self.tactile.open()
        self.params = TactileServoParams(control_rate=50, grade='M')
        self.controller = TactileServoController(self.tactile, self.params)

    def tearDown(self):
        self.tactile.close()

    def test_controller_initialization(self):
        """测试控制器初始化"""
        self.assertIsInstance(self.controller.params, TactileServoParams)
        self.assertFalse(self.controller._is_grasping)

    def test_compute_control_signal_no_contact(self):
        """测试无接触时控制信号为零"""
        signal = self.controller.compute_control_signal(target_force=5.0)
        self.assertEqual(signal.shape, (3,))
        np.testing.assert_allclose(signal, np.zeros(3), atol=1e-5)

    def test_compute_control_signal_sequence(self):
        """测试连续帧控制信号"""
        signals = []
        for _ in range(5):
            sig = self.controller.compute_control_signal(target_force=10.0)
            signals.append(sig)
        self.assertEqual(len(signals), 5)
        for s in signals:
            self.assertEqual(s.shape, (3,))

    def test_detect_and_react_slip(self):
        """测试滑移检测与响应"""
        self.tactile.capture()
        self.tactile.capture()
        self.tactile.capture()
        reactive = self.controller.detect_and_react_slip()
        self.assertEqual(reactive.shape, (3,))

    def test_monitor_grasp_quality(self):
        """测试抓取质量监控"""
        for _ in range(10):
            self.controller.compute_control_signal(target_force=5.0)
        quality = self.controller.monitor_grasp_quality()
        self.assertIn('current', quality)
        self.assertIn('average', quality)
        self.assertIn('trend', quality)
        # stable may or may not be present depending on history
        if 'stable' in quality:
            self.assertIsInstance(quality['stable'], (bool, np.bool_))

    def test_is_contact_detection(self):
        """测试接触检测"""
        is_contact = self.controller.is_contact()
        self.assertIsInstance(is_contact, bool)

    def test_grade_configs(self):
        """测试AGV五级配置"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            params = TactileServoParams.from_grade(grade)
            self.assertEqual(params.grade, grade)


class TestGraspQualityController(unittest.TestCase):
    """抓取质量控制器测试"""

    def setUp(self):
        self.tactile = TactileArray((8, 8), sensor_id="gqc_test")
        self.tactile.open()
        self.controller = GraspQualityController(self.tactile, target_quality=0.7)

    def tearDown(self):
        self.tactile.close()

    def test_evaluation_returns_metrics(self):
        """测试评估返回质量指标"""
        for _ in range(3):
            self.tactile.capture()
        quality, adjustment = self.controller.evaluate_and_regulate()
        self.assertIn('overall', quality)
        self.assertEqual(adjustment.shape, (3,))

    def test_target_quality_configurable(self):
        """测试目标质量可配置"""
        controller = GraspQualityController(self.tactile, target_quality=0.9)
        self.assertEqual(controller.target_quality, 0.9)


class TestForceController(unittest.TestCase):
    """力觉控制器测试"""

    def setUp(self):
        self.force = ForceTorqueSensor(
            sensor_type=ForceSensorType.SIX_AXIS,
            sensor_id="test_force_ctrl"
        )
        self.force.open()
        self.params = ForceControlParams(control_rate=100, grade='M')
        self.controller = ForceController(self.force, self.params)

    def tearDown(self):
        self.force.close()

    def test_admittance_control_basic(self):
        """测试导纳控制基础输出"""
        desired = np.array([0.0, 0.0, -10.0])
        adj = self.controller.compute_admittance(desired)
        self.assertEqual(adj.shape, (3,))

    def test_admittance_sequence(self):
        """测试连续导纳控制"""
        desired = np.array([0.0, 0.0, -5.0])
        adjustments = []
        for _ in range(10):
            adj = self.controller.compute_admittance(desired, dt=0.01)
            adjustments.append(adj)
        self.assertEqual(len(adjustments), 10)

    def test_collision_detection(self):
        """测试碰撞检测"""
        # Lower threshold to detect collision in simulation
        self.controller.params.collision_threshold = 2.0
        self.force.capture()
        self.force.capture()
        self.force.capture()
        is_collision, magnitude = self.controller.detect_collision()
        self.assertIsInstance(is_collision, (bool, np.bool_))
        self.assertGreaterEqual(magnitude, 0.0)

    def test_collision_response(self):
        """测试碰撞响应"""
        direction = np.array([1.0, 0.0, 0.0])
        response = self.controller.compute_collision_response(direction)
        self.assertEqual(response.shape, (3,))

    def test_grade_configs(self):
        """测试AGV五级配置"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            params = ForceControlParams.from_grade(grade)
            self.assertEqual(params.grade, grade)


class TestHybridForcePositionController(unittest.TestCase):
    """力位混合控制器测试"""

    def setUp(self):
        self.force = ForceTorqueSensor(
            sensor_type=ForceSensorType.SIX_AXIS,
            sensor_id="test_hybrid"
        )
        self.force.open()
        self.params = ForceControlParams(control_rate=100, grade='M')
        self.controller = HybridForcePositionController(self.force, self.params)

    def tearDown(self):
        self.force.close()

    def test_hybrid_control_output(self):
        """测试混合控制输出"""
        target_force = np.array([0.0, 0.0, -10.0])
        target_pos = np.array([0.5, 0.5, 0.0])
        pos_out, force_out = self.controller.compute_control(target_force, target_pos)
        self.assertEqual(pos_out.shape, (3,))
        self.assertEqual(force_out.shape, (3,))

    def test_force_control_axes_config(self):
        """测试力控轴配置"""
        self.controller.force_control_axes = np.array([True, False, True])
        np.testing.assert_array_equal(
            self.controller.force_control_axes,
            np.array([True, False, True])
        )


class TestIMUBasedControl(unittest.TestCase):
    """IMU姿态控制测试"""

    def setUp(self):
        self.imu = IMUSensor(
            sensor_type=IMUSensorType.BMI088,
            sensor_id="test_imu_ctrl"
        )
        self.imu.open()
        self.pose_estimator = PoseEstimator(algorithm='madgwick', sample_rate=200)
        self.params = IMUControlParams.from_grade('M')
        self.attitude_stab = AttitudeStabilizer(
            self.imu, self.params
        )

    def tearDown(self):
        self.imu.close()

    def test_pose_estimation_sequence(self):
        """测试姿态估计序列"""
        for _ in range(10):
            frame = self.imu.capture()
            pose = self.pose_estimator.update(
                frame.accel, frame.gyro, frame.mag
            )
            self.assertIsInstance(pose, Pose)
        euler = self.pose_estimator.get_euler()
        self.assertEqual(euler.shape, (3,))

    def test_attitude_stabilizer(self):
        """测试姿态稳定器"""
        self.imu.calibrate_gyro_bias(num_samples=50)
        self.attitude_stab.set_target_attitude(0.0, 0.0, 0.0)
        for _ in range(20):
            frame = self.imu.capture()
            self.attitude_stab.update(frame)
        tilt = self.attitude_stab.get_tilt_status()
        self.assertIn('tilt_magnitude', tilt)
        self.assertIn('is_stable', tilt)

    def test_motion_estimator(self):
        """测试运动估计器"""
        estimator = MotionEstimator(self.imu, remove_gravity=True)
        for _ in range(10):
            frame = self.imu.capture()
            vel, pos = estimator.update(frame, dt=0.01)
        self.assertEqual(vel.shape, (3,))
        self.assertEqual(pos.shape, (3,))


class TestAGVMotionController(unittest.TestCase):
    """AGV运动控制器测试"""

    def _make_spec(self, grade_str):
        grade_map = {
            'S': (AGVGrade.S, 0.5, 1.0),
            'M': (AGVGrade.M, 1.5, 2.0),
            'L': (AGVGrade.L, 2.0, 2.5),
            'XL': (AGVGrade.XL, 2.5, 3.0),
            'XXL': (AGVGrade.XXL, 3.0, 3.5),
        }
        grade, max_lin, max_ang = grade_map[grade_str]
        return AGVSpec(
            grade=grade,
            max_linear_speed=max_lin,
            max_angular_speed=max_ang,
            max_linear_accel=2.0,
            max_angular_accel=3.0,
            wheelbase=0.5,
            track_width=0.4,
            wheel_radius=0.07,
            drive_type=DriveType.DIFFERENTIAL,
            control_frequency=100.0
        )

    def test_agv_initialization(self):
        """测试AGV控制器初始化"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = self._make_spec(grade)
            controller = AGVMotionController(spec=spec)
            self.assertIsNotNone(controller)

    def test_inverse_kinematics(self):
        """测试逆运动学"""
        spec = self._make_spec('M')
        controller = AGVMotionController(spec=spec)
        twist = AGVTwist(vx=0.5, vy=0.0, omega=0.2)
        wheel_vel = controller.inverse_kinematics(twist)
        self.assertIsInstance(wheel_vel, np.ndarray)
        self.assertEqual(len(wheel_vel), 2)

    def test_forward_kinematics(self):
        """测试正运动学"""
        spec = self._make_spec('M')
        controller = AGVMotionController(spec=spec)
        wheel_vel = np.array([1.0, 1.0])
        twist = controller.forward_kinematics(wheel_vel)
        self.assertIsInstance(twist, AGVTwist)

    def test_compute_wheel_commands(self):
        """测试轮速命令计算"""
        spec = self._make_spec('M')
        controller = AGVMotionController(spec=spec)
        target = AGVPose(x=1.0, y=0.0, theta=0.0)
        controller.update_pose(target)
        wheel_cmd = controller.compute_wheel_commands(target, dt=0.01)
        self.assertIsInstance(wheel_cmd, np.ndarray)
        self.assertEqual(len(wheel_cmd), 2)


class TestControlPipelineIntegration(unittest.TestCase):
    """控制流水线集成测试: 传感器→融合→控制→执行"""

    def test_tactile_pipeline(self):
        """测试触觉控制流水线"""
        tactile = TactileArray((16, 16), sensor_id="pipeline_tactile")
        tactile.open()
        controller = TactileServoController(tactile, TactileServoParams.from_grade('M'))
        
        for i in range(10):
            frame = tactile.capture()
            contacts = tactile.detect_contacts(frame)
            control_sig = controller.compute_control_signal(target_force=10.0)
            slip_reactive = controller.detect_and_react_slip(frame)
        
        tactile.close()
        self.assertTrue(True)  # 无异常即通过

    def test_force_pipeline(self):
        """测试力控流水线"""
        force = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS, sensor_id="pipeline_force")
        force.open()
        controller = ForceController(force, ForceControlParams.from_grade('M'))
        
        for i in range(10):
            wrench = force.capture()
            desired = np.array([0.0, 0.0, -10.0])
            adm_adj = controller.compute_admittance(desired)
            is_col, mag = controller.detect_collision()
        
        force.close()
        self.assertTrue(True)

    def test_imu_pipeline(self):
        """测试IMU控制流水线"""
        imu = IMUSensor(sensor_type=IMUSensorType.BMI088, sensor_id="pipeline_imu")
        imu.open()
        pose_est = PoseEstimator(algorithm='madgwick', sample_rate=200)
        
        for i in range(20):
            frame = imu.capture()
            pose = pose_est.update(frame.accel, frame.gyro, frame.mag)
            euler = pose_est.get_euler()
            R = pose_est.get_rotation_matrix()
        
        imu.close()
        self.assertTrue(True)

    def test_agv_kinematics_pipeline(self):
        """测试AGV运动学流水线"""
        spec = AGVSpec(
            grade=AGVGrade.M,
            max_linear_speed=1.5,
            max_angular_speed=2.0,
            max_linear_accel=2.0,
            max_angular_accel=3.0,
            wheelbase=0.5,
            track_width=0.4,
            wheel_radius=0.07,
            drive_type=DriveType.DIFFERENTIAL,
            control_frequency=100.0
        )
        agv = AGVMotionController(spec=spec)
        
        for step in range(50):
            target_twist = AGVTwist(vx=0.5, vy=0.0, omega=0.0)
            agv.update_twist(target_twist)
            wheel_vel = agv.inverse_kinematics(agv.twist)
            self.assertEqual(wheel_vel.shape, (2,))
            # Forward kinematics back
            twist_recovered = agv.forward_kinematics(wheel_vel)
            self.assertIsInstance(twist_recovered, AGVTwist)


if __name__ == '__main__':
    unittest.main()
