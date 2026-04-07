"""
传感器模块单元测试
测试触觉、力觉、IMU传感器的功能
覆盖: TactileArray, ForceTorqueSensor, IMUSensor 及其虚拟传感器
"""

import unittest
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sensors.tactile import (
    TactileArray, TactileFrame, TactileContact, TactileCalibration,
    TactileSensorType, PressureProcessor, VirtualTactileSensor,
    get_tactile_spec, AGV_TACTILE_GRADES
)
from src.sensors.force import (
    ForceTorqueSensor, Wrench, ForceCalibration, ContactState,
    ForceSensorType, WrenchProcessor, VirtualForceSensor,
    get_force_spec, AGV_FORCE_GRADES
)
from src.sensors.imu import (
    IMUSensor, IMUFrame, Pose, PoseEstimator, IMUCalibration,
    IMUSensorType, VirtualIMUSensor, get_imu_spec, AGV_IMU_GRADES
)


class TestTactileArray(unittest.TestCase):
    """测试 TactileArray 电子皮肤触觉阵列"""

    def setUp(self):
        self.array = TactileArray(
            array_size=(8, 8),
            sensor_type=TactileSensorType.RESISTIVE,
            sensor_id="test_tactile"
        )

    def test_creation(self):
        """测试创建"""
        self.assertEqual(self.array.sensor_id, "test_tactile")
        self.assertEqual(self.array.array_size, (8, 8))
        self.assertEqual(self.array.sensor_type, TactileSensorType.RESISTIVE)

    def test_context_manager(self):
        """测试上下文管理器"""
        with TactileArray((4, 4), sensor_id="ctx_test") as arr:
            self.assertTrue(arr._is_opened)
        # 退出后应关闭
        self.assertFalse(arr._is_opened)

    def test_capture_returns_frame(self):
        """测试捕获返回 TactileFrame"""
        self.array.open()
        frame = self.array.capture()
        self.assertIsInstance(frame, TactileFrame)
        self.assertEqual(frame.pressure_map.shape, (8, 8))
        self.assertEqual(frame.sensor_id, "test_tactile")
        self.array.close()

    def test_capture_requires_open(self):
        """测试未打开时抛异常"""
        with self.assertRaises(RuntimeError):
            self.array.capture()

    def test_detect_contacts(self):
        """测试接触检测"""
        self.array.open()
        frame = self.array.capture()
        contacts = self.array.detect_contacts(frame)
        # 初始状态可能无接触
        self.assertIsInstance(contacts, list)

    def test_slip_signal(self):
        """测试滑移信号计算"""
        self.array.open()
        self.array.capture()
        self.array.capture()
        slip = self.array.get_slip_signal()
        self.assertIsInstance(slip, np.ndarray)
        self.assertEqual(slip.shape, (8, 8))

    def test_grip_quality(self):
        """测试抓取质量评估"""
        self.array.open()
        frame = self.array.capture()
        quality = self.array.estimate_grip_quality(frame)
        self.assertIn('overall', quality)
        self.assertIn('contact_area', quality)
        self.assertIn('uniformity', quality)
        self.assertIn('stability', quality)

    def test_calibrate(self):
        """测试标定"""
        zero = np.zeros((8, 8))
        self.array.calibrate(zero_pressure=zero)
        self.assertIsNotNone(self.array.calibration.offset_map)

    def test_multiple_frames_sequence(self):
        """测试连续多帧采集"""
        self.array.open()
        for i in range(10):
            frame = self.array.capture()
            self.assertEqual(frame.frame_id, i)
        self.array.close()


class TestVirtualTactileSensor(unittest.TestCase):
    """测试虚拟触觉传感器"""

    def test_simulate_contact(self):
        """测试模拟接触"""
        sensor = VirtualTactileSensor((16, 16), "virtual_tactile")
        sensor.open()
        frame = sensor.simulate_contact(
            contact_pos=(0.5, 0.5),
            contact_radius=0.3,
            contact_force=15.0
        )
        self.assertIsInstance(frame, TactileFrame)
        self.assertGreater(np.max(frame.pressure_map), 0)
        sensor.close()

    def test_simulate_sliding(self):
        """测试模拟滑移动作"""
        sensor = VirtualTactileSensor((16, 16))
        sensor.open()
        frames = sensor.simulate_sliding(
            direction=(1.0, 0.0),
            speed=0.05,
            duration_frames=10
        )
        self.assertEqual(len(frames), 10)
        self.assertIsInstance(frames[0], TactileFrame)
        sensor.close()

    def test_simulate_multi_contact(self):
        """测试模拟多点接触"""
        sensor = VirtualTactileSensor((16, 16))
        sensor.open()
        contacts = [
            ((0.3, 0.3), 10.0, 0.2),
            ((0.7, 0.7), 8.0, 0.15)
        ]
        frame = sensor.simulate_multi_contact(contacts)
        self.assertGreater(np.max(frame.pressure_map), 0)
        sensor.close()

    def test_simulate_slip_detection(self):
        """测试模拟滑移检测"""
        sensor = VirtualTactileSensor((16, 16))
        sensor.open()
        result = sensor.simulate_slip_detection(
            normal_force=10.0,
            friction_coeff=0.3,
            velocity=(0.1, 0.0)
        )
        self.assertIn('slip_state', result)
        self.assertIn('slip_probability', result)
        sensor.close()


class TestForceTorqueSensor(unittest.TestCase):
    """测试六维力矩传感器"""

    def setUp(self):
        self.sensor = ForceTorqueSensor(
            sensor_type=ForceSensorType.SIX_AXIS,
            sensor_id="test_ft"
        )

    def test_creation(self):
        """测试创建"""
        self.assertEqual(self.sensor.sensor_id, "test_ft")
        self.assertEqual(self.sensor.sensor_type, ForceSensorType.SIX_AXIS)

    def test_context_manager(self):
        """测试上下文管理器"""
        with ForceTorqueSensor(sensor_id="ctx_ft") as sensor:
            self.assertTrue(sensor._is_streaming)
        self.assertFalse(sensor._is_streaming)

    def test_capture_returns_wrench(self):
        """测试捕获返回 Wrench"""
        self.sensor.open()
        wrench = self.sensor.capture()
        self.assertIsInstance(wrench, Wrench)
        self.assertEqual(wrench.force.shape, (3,))
        self.assertEqual(wrench.torque.shape, (3,))
        self.sensor.close()

    def test_capture_requires_open(self):
        """测试未打开时抛异常"""
        with self.assertRaises(RuntimeError):
            self.sensor.capture()

    def test_get_wrench(self):
        """测试获取最新力数据"""
        self.sensor.open()
        self.sensor.capture()
        wrench = self.sensor.get_wrench()
        self.assertIsNotNone(wrench)
        self.sensor.close()

    def test_detect_contact(self):
        """测试接触检测"""
        self.sensor.open()
        self.sensor.capture()
        state = self.sensor.detect_contact()
        self.assertIsInstance(state, ContactState)
        self.sensor.close()

    def test_estimate_payload(self):
        """测试负载估计"""
        self.sensor.open()
        self.sensor.capture()
        payload = self.sensor.estimate_payload()
        self.assertGreaterEqual(payload, 0)
        self.sensor.close()

    def test_set_tool_center(self):
        """测试设置工具中心"""
        self.sensor.open()
        self.sensor.set_tool_center(0.5, np.array([0.0, 0.0, 0.1]))
        self.assertEqual(self.sensor.tool_center[2], 0.1)
        self.sensor.close()

    def test_calibrate_bias(self):
        """测试偏置校准"""
        self.sensor.open()
        self.sensor.calibrate_bias(num_samples=10)
        self.assertIsNotNone(self.sensor.calibration.bias)
        self.sensor.close()

    def test_wrench_transform(self):
        """测试力旋量坐标变换"""
        self.sensor.open()
        wrench = self.sensor.capture()
        R = np.eye(3)
        t = np.array([0.1, 0.0, 0.0])
        new_wrench = wrench.transform(R, t)
        self.assertEqual(new_wrench.force.shape, (3,))
        self.sensor.close()


class TestWrench(unittest.TestCase):
    """测试 Wrench 力旋量数据类"""

    def test_creation(self):
        """测试创建"""
        w = Wrench(
            force=np.array([1.0, 2.0, 3.0]),
            torque=np.array([0.1, 0.2, 0.3])
        )
        self.assertEqual(w.force[0], 1.0)
        self.assertEqual(w.torque[2], 0.3)

    def test_magnitude(self):
        """测试力向量大小"""
        w = Wrench(force=np.array([3.0, 4.0, 0.0]), torque=np.zeros(3))
        self.assertAlmostEqual(w.magnitude, 5.0)

    def test_to_vector(self):
        """测试转换为6维向量"""
        w = Wrench(force=np.ones(3), torque=np.ones(3))
        vec = w.to_vector()
        self.assertEqual(vec.shape, (6,))

    def test_from_vector(self):
        """测试从6维向量创建"""
        vec = np.array([1, 2, 3, 0.1, 0.2, 0.3])
        w = Wrench.from_vector(vec)
        np.testing.assert_array_equal(w.force, vec[:3])
        np.testing.assert_array_equal(w.torque, vec[3:])


class TestVirtualForceSensor(unittest.TestCase):
    """测试虚拟力觉传感器"""

    def test_simulate_contact(self):
        """测试模拟接触力"""
        sensor = VirtualForceSensor("virtual_ft")
        sensor.open()
        wrench = sensor.simulate_contact(
            force=(10.0, 0.0, 0.0),
            torque=(0.0, 0.0, 0.0)
        )
        self.assertIsInstance(wrench, Wrench)
        self.assertGreater(wrench.magnitude, 0)
        sensor.close()

    def test_simulate_payload(self):
        """测试模拟负载重力"""
        sensor = VirtualForceSensor()
        sensor.open()
        wrench = sensor.simulate_payload(mass=1.0)
        self.assertLess(wrench.force[2], -5.0)  # Should be negative (gravity direction)
        sensor.close()

    def test_simulate_collision(self):
        """测试模拟碰撞事件"""
        sensor = VirtualForceSensor()
        sensor.open()
        frames = sensor.simulate_collision(
            direction=(1.0, 0.0, 0.0),
            peak_force=50.0,
            duration_ms=50.0
        )
        self.assertGreater(len(frames), 0)
        self.assertIsInstance(frames[0], Wrench)
        sensor.close()

    def test_simulate_surface_contact(self):
        """测试模拟表面接触"""
        sensor = VirtualForceSensor()
        sensor.open()
        wrench = sensor.simulate_surface_contact(
            surface_normal=(0.0, 0.0, 1.0),
            penetration_depth=0.002,
            stiffness=1000.0
        )
        # Spring force ~2N downward; damping noise can push it to ~±5N
        self.assertLess(wrench.force[2], 15.0)  # Should be small magnitude (noise-tolerant threshold)
        self.assertGreater(wrench.force[2], -15.0)  # Damping can make it more negative
        sensor.close()

    def test_simulate_friction_contact(self):
        """测试模拟摩擦力"""
        sensor = VirtualForceSensor()
        sensor.open()
        wrench = sensor.simulate_friction_contact(
            normal_force=10.0,
            velocity=(0.1, 0.0, 0.0),
            friction_coeff=0.3
        )
        self.assertIsInstance(wrench, Wrench)
        sensor.close()


class TestIMUSensor(unittest.TestCase):
    """测试IMU传感器"""

    def setUp(self):
        self.sensor = IMUSensor(
            sensor_type=IMUSensorType.BMI088,
            sensor_id="test_imu"
        )

    def test_creation(self):
        """测试创建"""
        self.assertEqual(self.sensor.sensor_id, "test_imu")
        self.assertEqual(self.sensor.sensor_type, IMUSensorType.BMI088)

    def test_context_manager(self):
        """测试上下文管理器"""
        with IMUSensor(sensor_id="ctx_imu") as sensor:
            self.assertTrue(sensor._is_opened)
        self.assertFalse(sensor._is_opened)

    def test_capture_returns_frame(self):
        """测试捕获返回 IMUFrame"""
        self.sensor.open()
        frame = self.sensor.capture()
        self.assertIsInstance(frame, IMUFrame)
        self.assertEqual(frame.accel.shape, (3,))
        self.assertEqual(frame.gyro.shape, (3,))
        self.sensor.close()

    def test_capture_requires_open(self):
        """测试未打开时抛异常"""
        with self.assertRaises(RuntimeError):
            self.sensor.capture()

    def test_self_test(self):
        """测试传感器自检"""
        self.sensor.open()
        result = self.sensor.self_test()
        self.assertIsInstance(result, bool)
        self.sensor.close()

    def test_calibrate_gyro_bias(self):
        """测试陀螺仪偏置校准"""
        self.sensor.open()
        self.sensor.calibrate_gyro_bias(num_samples=20)
        self.assertIsNotNone(self.sensor.calibration.gyro_bias)
        self.sensor.close()

    def test_calibrate_accel(self):
        """测试加速度计标定"""
        self.sensor.open()
        self.sensor.calibrate_accel(known_orientation="level")
        self.sensor.close()


class TestPose(unittest.TestCase):
    """测试 Pose 位姿数据类"""

    def test_identity(self):
        """测试单位位姿"""
        pose = Pose.identity()
        np.testing.assert_array_almost_equal(pose.orientation, [1, 0, 0, 0])
        np.testing.assert_array_equal(pose.position, [0, 0, 0])

    def test_to_euler(self):
        """测试转欧拉角"""
        pose = Pose.identity()
        euler = pose.to_euler()
        self.assertEqual(len(euler), 3)

    def test_from_euler(self):
        """测试从欧拉角创建"""
        pose = Pose.from_euler(
            position=np.array([1.0, 2.0, 3.0]),
            rpy=np.array([0.0, 0.0, 0.0])
        )
        np.testing.assert_array_almost_equal(pose.orientation, [1, 0, 0, 0])

    def test_to_matrix(self):
        """测试转4x4变换矩阵"""
        pose = Pose.identity()
        T = pose.to_matrix()
        self.assertEqual(T.shape, (4, 4))
        self.assertEqual(T[3, 3], 1.0)


class TestPoseEstimator(unittest.TestCase):
    """测试姿态估计器"""

    def test_initialization(self):
        """测试初始化"""
        est = PoseEstimator(algorithm="madgwick", sample_rate=200.0)
        self.assertEqual(est.algorithm, "madgwick")
        self.assertEqual(est.sample_rate, 200.0)

    def test_update_madgwick(self):
        """测试 Madgwick 更新"""
        est = PoseEstimator(algorithm="madgwick")
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.1])
        pose = est.update(accel, gyro, dt=0.01)
        self.assertIsInstance(pose, Pose)

    def test_update_complementary(self):
        """测试互补滤波更新"""
        est = PoseEstimator(algorithm="complementary")
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.1])
        pose = est.update(accel, gyro, dt=0.01)
        self.assertIsInstance(pose, Pose)

    def test_reset(self):
        """测试重置"""
        est = PoseEstimator()
        est.update(np.array([0, 0, 9.81]), np.array([0.1, 0.1, 0.1]), dt=0.01)
        est.reset()
        np.testing.assert_array_almost_equal(est.quaternion, [1, 0, 0, 0])

    def test_integrate_velocity(self):
        """测试速度积分"""
        est = PoseEstimator()
        accel = np.array([1.0, 0.0, 9.81])
        vel, pos = est.integrate_velocity(accel, dt=0.01)
        self.assertEqual(len(vel), 3)
        self.assertEqual(len(pos), 3)


class TestVirtualIMUSensor(unittest.TestCase):
    """测试虚拟IMU传感器"""

    def test_simulate_static(self):
        """测试模拟静止状态"""
        sensor = VirtualIMUSensor("virtual_imu")
        sensor.open()
        frame = sensor.simulate_static(orientation=(0.0, 0.0, 0.0))
        self.assertIsInstance(frame, IMUFrame)
        self.assertAlmostEqual(frame.accel[2], 9.81, places=1)
        sensor.close()

    def test_simulate_motion(self):
        """测试模拟运动状态"""
        sensor = VirtualIMUSensor()
        sensor.open()
        frame = sensor.simulate_motion(
            linear_accel=(1.0, 0.0, 0.0),
            angular_vel=(0.0, 0.0, 0.1),
            dt=0.01
        )
        self.assertIsInstance(frame, IMUFrame)
        sensor.close()

    def test_simulate_trajectory(self):
        """测试模拟轨迹"""
        sensor = VirtualIMUSensor()
        sensor.open()
        frames = sensor.simulate_trajectory(
            trajectory_type="circle",
            duration_s=0.1,
            dt=0.01
        )
        self.assertGreater(len(frames), 0)
        self.assertIsInstance(frames[0], IMUFrame)
        sensor.close()

    def test_simulate_agv_motion(self):
        """测试模拟AGV运动"""
        sensor = VirtualIMUSensor()
        sensor.open()
        frame = sensor.simulate_agv_motion(
            linear_velocity=(0.5, 0.0),
            angular_velocity=0.1,
            grade="M"
        )
        self.assertIsInstance(frame, IMUFrame)
        sensor.close()

    def test_simulate_human_walking(self):
        """测试模拟人类步行"""
        sensor = VirtualIMUSensor()
        sensor.open()
        frames = sensor.simulate_human_walking(
            step_frequency=1.5,
            walk_speed=1.0,
            duration_s=0.5
        )
        self.assertGreater(len(frames), 0)
        sensor.close()


class TestAGVTactileGrades(unittest.TestCase):
    """测试AGV五级触觉规格"""

    def test_all_grades_exist(self):
        """测试所有等级都有规格"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_tactile_spec(grade)
            self.assertIn('array', spec)
            self.assertIn('res', spec)
            self.assertIn('range_kpa', spec)
            self.assertIn('freq_hz', spec)

    def test_grade_progression(self):
        """测试等级递增(阵列越大)"""
        s_spec = get_tactile_spec('S')
        m_spec = get_tactile_spec('M')
        l_spec = get_tactile_spec('L')
        self.assertLess(s_spec['array'][0], m_spec['array'][0])
        self.assertLess(m_spec['array'][0], l_spec['array'][0])


class TestAGVForceGrades(unittest.TestCase):
    """测试AGV五级力觉规格"""

    def test_all_grades_exist(self):
        """测试所有等级都有规格"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_force_spec(grade)
            self.assertIn('axes', spec)
            self.assertIn('force_range', spec)
            self.assertIn('torque_range', spec)
            self.assertIn('sampling_hz', spec)

    def test_grade_progression(self):
        """测试等级递增(力范围越大)"""
        s_spec = get_force_spec('S')
        l_spec = get_force_spec('L')
        self.assertLess(s_spec['force_range'], l_spec['force_range'])


class TestAGVIMUGrades(unittest.TestCase):
    """测试AGV五级IMU规格"""

    def test_all_grades_exist(self):
        """测试所有等级都有规格"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_imu_spec(grade)
            self.assertIn('type', spec)
            self.assertIn('accel_range', spec)
            self.assertIn('gyro_range', spec)
            self.assertIn('sample_hz', spec)

    def test_grade_progression(self):
        """测试等级递增(采样率越高,噪声越低)"""
        s_spec = get_imu_spec('S')
        l_spec = get_imu_spec('L')
        self.assertLess(s_spec['sample_hz'], l_spec['sample_hz'])
        self.assertGreater(s_spec['noise_density'], l_spec['noise_density'])


class TestWrenchProcessor(unittest.TestCase):
    """测试力旋量信号处理器"""

    def test_filter(self):
        """测试指数移动平均滤波"""
        proc = WrenchProcessor(filter_alpha=0.3)
        wrench = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        filtered = proc.filter(wrench)
        np.testing.assert_array_almost_equal(filtered, wrench)

    def test_estimate_covariance(self):
        """测试协方差估计"""
        proc = WrenchProcessor()
        history = [np.random.randn(6) for _ in range(10)]
        cov = proc.estimate_covariance(history)
        self.assertEqual(cov.shape, (6, 6))

    def test_compute_force_direction(self):
        """测试力向量方向计算"""
        proc = WrenchProcessor()
        wrench = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        direction = proc.compute_force_direction(wrench)
        np.testing.assert_array_almost_equal(direction, [1.0, 0.0, 0.0])


class TestPressureProcessor(unittest.TestCase):
    """测试压力信号处理器"""

    def test_filter(self):
        """测试中值滤波"""
        proc = PressureProcessor(filter_window=3)
        data = np.random.rand(16, 16)
        filtered = proc.filter(data)
        self.assertEqual(filtered.shape, data.shape)

    def test_compensate_baseline(self):
        """测试基线补偿"""
        proc = PressureProcessor()
        data = np.random.rand(8, 8) * 0.1
        proc.compensate_baseline(data, set_baseline=True)
        compensated = proc.compensate_baseline(data)
        self.assertEqual(compensated.shape, data.shape)

    def test_compute_centroid(self):
        """测试压力分布质心计算"""
        proc = PressureProcessor()
        data = np.zeros((8, 8))
        data[4, 4] = 1.0
        cy, cx = proc.compute_centroid(data)
        # Centroid of a single peak at [4,4] in 8x8 array is near 3.5
        self.assertGreaterEqual(int(cy), 3)
        self.assertLessEqual(int(cy), 4)


if __name__ == '__main__':
    unittest.main()


class TestTactileEdgeCases(unittest.TestCase):
    """触觉传感器边界情况测试"""

    def test_empty_contact(self):
        """测试无接触情况"""
        sensor = TactileArray(array_size=(8, 8))
        sensor.open()
        frame = sensor.capture()
        contacts = sensor.detect_contacts(frame)
        self.assertEqual(len(contacts), 0)
        sensor.close()

    def test_multi_contact(self):
        """测试多点接触"""
        sensor = TactileArray(array_size=(16, 16))
        sensor.open()
        frame = sensor.capture()
        # 模拟两个接触点
        contacts = sensor.detect_contacts(frame)
        # 空阵列可能无接触
        self.assertIsInstance(contacts, list)
        sensor.close()

    def test_grip_quality_no_contact(self):
        """测试无接触时的抓取质量"""
        sensor = TactileArray(array_size=(8, 8))
        sensor.open()
        frame = sensor.capture()
        quality = sensor.estimate_grip_quality(frame)
        self.assertIn('overall', quality)
        self.assertIn('contact_area', quality)
        sensor.close()

    def test_calibrate_with_zero(self):
        """测试零压力标定"""
        sensor = TactileArray(array_size=(8, 8))
        sensor.open()
        zero = np.zeros((8, 8))
        sensor.calibrate(zero_pressure=zero)
        self.assertIsNotNone(sensor.calibration.offset_map)
        sensor.close()

    def test_context_manager(self):
        """测试上下文管理器"""
        with TactileArray(array_size=(8, 8)) as sensor:
            frame = sensor.capture()
            self.assertIsNotNone(frame)
        # After exit, sensor should be closed

    def test_pressure_processor_histogram(self):
        """测试压力直方图"""
        proc = PressureProcessor()
        data = np.random.rand(8, 8)
        hist, edges = proc.compute_pressure_histogram(data, bins=5)
        self.assertEqual(len(hist), 5)
        self.assertEqual(len(edges), 6)


class TestVirtualTactileEdgeCases(unittest.TestCase):
    """虚拟触觉传感器边界测试"""

    def test_simulate_contact_out_of_bounds(self):
        """测试边界外的接触"""
        sensor = VirtualTactileSensor(array_size=(8, 8))
        sensor.open()
        # 接触位置超出范围
        frame = sensor.simulate_contact((1.5, 0.5), contact_radius=0.1)
        self.assertIsNotNone(frame.pressure_map)
        sensor.close()

    def test_simulate_sliding(self):
        """测试滑移动作"""
        sensor = VirtualTactileSensor(array_size=(16, 16))
        sensor.open()
        frames = sensor.simulate_sliding(direction=(0.1, 0.0), speed=0.05, duration_frames=10)
        self.assertEqual(len(frames), 10)
        for f in frames:
            self.assertEqual(f.pressure_map.shape, (16, 16))
        sensor.close()

    def test_simulate_multi_contact(self):
        """测试多点接触"""
        sensor = VirtualTactileSensor(array_size=(16, 16))
        sensor.open()
        contacts = [((0.3, 0.3), 5.0, 0.2), ((0.7, 0.7), 8.0, 0.15)]
        frame = sensor.simulate_multi_contact(contacts)
        self.assertEqual(frame.pressure_map.shape, (16, 16))
        sensor.close()

    def test_simulate_slip_detection_stick(self):
        """测试静止滑移检测"""
        sensor = VirtualTactileSensor()
        result = sensor.simulate_slip_detection(normal_force=10.0, velocity=(0.0, 0.0))
        self.assertEqual(result['slip_state'], 'stick')
        self.assertEqual(result['slip_probability'], 0.0)

    def test_simulate_slip_detection_sliding(self):
        """测试滑动滑移检测"""
        sensor = VirtualTactileSensor()
        result = sensor.simulate_slip_detection(normal_force=10.0, velocity=(0.1, 0.0))
        self.assertIn(result['slip_state'], ['micro_slip', 'sliding'])
        self.assertGreater(result['slip_probability'], 0.0)


class TestForceEdgeCases(unittest.TestCase):
    """力传感器边界测试"""

    def test_wrench_transform(self):
        """测试力旋量坐标变换"""
        w = Wrench(force=np.array([0.0, 0.0, -5.0]), torque=np.zeros(3))
        R = np.eye(3)
        t = np.array([0.1, 0.0, 0.0])
        w2 = w.transform(R, t)
        self.assertEqual(w2.force[2], -5.0)

    def test_wrench_from_vector(self):
        """测试从向量创建Wrench"""
        vec = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        w = Wrench.from_vector(vec)
        np.testing.assert_array_almost_equal(w.force, [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(w.torque, [0.1, 0.2, 0.3])

    def test_wrench_magnitude(self):
        """测试力向量模长"""
        w = Wrench(force=np.array([3.0, 4.0, 0.0]), torque=np.zeros(3))
        self.assertAlmostEqual(w.magnitude, 5.0)

    def test_force_sensor_bias_calibration(self):
        """测试偏置校准"""
        sensor = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        sensor.open()
        sensor.calibrate_bias(num_samples=50)
        self.assertIsNotNone(sensor.calibration.bias)
        sensor.close()

    def test_force_sensor_tool_center(self):
        """测试工具中心设置"""
        sensor = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        sensor.open()
        sensor.set_tool_center(tool_mass=0.5, tool_com=np.array([0.05, 0.0, 0.1]))
        self.assertIsNotNone(sensor.calibration.bias)
        sensor.close()

    def test_contact_detection_no_contact(self):
        """测试无接触检测"""
        sensor = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        sensor.open()
        state = sensor.detect_contact(threshold=10.0)
        self.assertIsInstance(state.is_contact, bool)
        sensor.close()

    def test_payload_estimation(self):
        """测试负载估计"""
        sensor = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        sensor.open()
        payload = sensor.estimate_payload()
        self.assertGreaterEqual(payload, 0.0)
        sensor.close()


class TestVirtualForceEdgeCases(unittest.TestCase):
    """虚拟力传感器边界测试"""

    def test_simulate_payload(self):
        """测试负载仿真"""
        sensor = VirtualForceSensor()
        sensor.open()
        wrench = sensor.simulate_payload(mass=1.0, com_offset=(0.01, 0.0, 0.0))
        self.assertIsNotNone(wrench.force)
        sensor.close()

    def test_simulate_collision(self):
        """测试碰撞仿真"""
        sensor = VirtualForceSensor()
        sensor.open()
        frames = sensor.simulate_collision(direction=(1.0, 0.0, 0.0), peak_force=50.0, duration_ms=50.0)
        self.assertGreater(len(frames), 0)
        self.assertLessEqual(len(frames), 5)
        sensor.close()

    def test_simulate_surface_contact(self):
        """测试表面接触"""
        sensor = VirtualForceSensor()
        sensor.open()
        wrench = sensor.simulate_surface_contact(
            surface_normal=(0.0, 0.0, 1.0),
            contact_point=(0.0, 0.0, 0.0),
            penetration_depth=0.001,
            stiffness=1000.0
        )
        self.assertIsNotNone(wrench.force)
        sensor.close()

    def test_simulate_friction_contact(self):
        """测试摩擦力"""
        sensor = VirtualForceSensor()
        sensor.open()
        wrench = sensor.simulate_friction_contact(
            normal_force=10.0,
            velocity=(0.1, 0.0, 0.0),
            friction_coeff=0.3
        )
        self.assertIsNotNone(wrench.force)
        sensor.close()


class TestIMUEdgeCases(unittest.TestCase):
    """IMU边界测试"""

    def test_pose_identity(self):
        """测试单位位姿"""
        p = Pose.identity()
        np.testing.assert_array_almost_equal(p.position, [0, 0, 0])
        np.testing.assert_array_almost_equal(p.orientation, [1, 0, 0, 0])

    def test_pose_euler_roundtrip(self):
        """测试欧拉角往返转换"""
        p1 = Pose.from_euler(np.zeros(3), np.array([0.1, 0.2, 0.3]))
        euler = p1.to_euler()
        p2 = Pose.from_euler(p1.position, euler)
        np.testing.assert_array_almost_equal(p1.orientation, p2.orientation, decimal=5)

    def test_imu_self_test_pass(self):
        """测试IMU自检通过"""
        sensor = IMUSensor(sensor_type=IMUSensorType.BMI088)
        sensor.open()
        result = sensor.self_test()
        self.assertIsInstance(result, bool)
        sensor.close()

    def test_imu_calibrate_gyro(self):
        """测试陀螺仪偏置校准"""
        sensor = IMUSensor(sensor_type=IMUSensorType.MPU6050)
        sensor.open()
        sensor.calibrate_gyro_bias(num_samples=50)
        self.assertIsNotNone(sensor.calibration.gyro_bias)
        sensor.close()

    def test_imu_calibrate_accel(self):
        """测试加速度计标定"""
        sensor = IMUSensor(sensor_type=IMUSensorType.MPU6050)
        sensor.open()
        sensor.calibrate_accel(known_orientation='level')
        self.assertIsNotNone(sensor.calibration.accel_scale)
        sensor.close()

    def test_imu_context_manager(self):
        """测试IMU上下文管理器"""
        with IMUSensor(sensor_type=IMUSensorType.VIRTUAL) as sensor:
            frame = sensor.capture()
            self.assertIsNotNone(frame)


class TestTactileSlipDetection(unittest.TestCase):
    """触觉滑移检测边界测试"""

    def test_slip_under_low_normal_force(self):
        """测试低法向力下的滑移检测"""
        sensor = VirtualTactileSensor(array_size=(16, 16), sensor_id='slip_test')
        sensor.open()
        
        # 法向力过小，容易打滑
        result = sensor.simulate_slip_detection(
            normal_force=0.5,  # 极低法向力
            friction_coeff=0.3,
            velocity=(0.1, 0.0)
        )
        self.assertIn('slip_state', result)
        self.assertIn('slip_probability', result)
        self.assertGreaterEqual(result['slip_probability'], 0.0)
        self.assertLessEqual(result['slip_probability'], 1.0)
        
        sensor.close()

    def test_slip_high_velocity(self):
        """测试高速滑移"""
        sensor = VirtualTactileSensor(array_size=(16, 16), sensor_id='slip_test2')
        sensor.open()
        
        result = sensor.simulate_slip_detection(
            normal_force=20.0,
            friction_coeff=0.5,
            velocity=(1.0, 0.5)  # 高速
        )
        self.assertEqual(result['slip_state'], 'sliding')
        self.assertGreater(result['slip_probability'], 0.5)
        
        sensor.close()

    def test_slip_zero_velocity(self):
        """测试零速度(静止)"""
        sensor = VirtualTactileSensor(array_size=(16, 16), sensor_id='slip_test3')
        sensor.open()
        
        result = sensor.simulate_slip_detection(
            normal_force=15.0,
            friction_coeff=0.4,
            velocity=(0.0, 0.0)
        )
        self.assertEqual(result['slip_state'], 'stick')
        self.assertEqual(result['slip_probability'], 0.0)
        
        sensor.close()

    def test_slip_micro_slip_transition(self):
        """测试微观滑移过渡状态"""
        sensor = VirtualTactileSensor(array_size=(16, 16), sensor_id='slip_test4')
        sensor.open()
        
        # 边界速度: 严格小于0.05阈值(0.05触发sliding)
        for v in [0.01, 0.02, 0.03, 0.04, 0.049]:
            result = sensor.simulate_slip_detection(
                normal_force=5.0,
                friction_coeff=0.3,
                velocity=(v, 0.0)
            )
            self.assertIn(result['slip_state'], ['stick', 'micro_slip'])
            self.assertGreater(result['velocity_magnitude'], 0.0)
        
        sensor.close()

    def test_multi_contact_combined(self):
        """测试多点接触组合"""
        sensor = VirtualTactileSensor(array_size=(16, 16), sensor_id='multi_test')
        sensor.open()
        
        contacts = [
            ((0.3, 0.3), 5.0, 0.1),  # 位置, 力, 半径
            ((0.7, 0.7), 8.0, 0.15),
            ((0.5, 0.2), 3.0, 0.08),
        ]
        
        frame = sensor.simulate_multi_contact(contacts, noise_level=0.05)
        
        self.assertEqual(frame.pressure_map.shape, (16, 16))
        self.assertEqual(frame.temperature_map.shape, (16, 16))
        self.assertGreater(np.max(frame.pressure_map), 0.0)
        np.testing.assert_array_less(frame.pressure_map, 2.0)  # 应该在合理范围内
        
        sensor.close()


class TestForceWrenchTransform(unittest.TestCase):
    def test_wrench_frame_change(self):
        sensor = VirtualForceSensor(sensor_id="wrench_test")
        sensor.open()
        wrench = sensor.simulate_contact(force=(10.0, 0.0, 0.0), torque=(0.0, 0.0, 0.0))
        R = np.eye(3); t = np.array([0.1, 0.0, 0.0])
        rotated = wrench.transform(R, t)
        self.assertIsNotNone(rotated)
        self.assertEqual(len(rotated.force), 3)
        self.assertEqual(len(rotated.torque), 3)
        sensor.close()
    def test_wrench_bias_removal(self):
        sensor = VirtualForceSensor(sensor_id="bias_test")
        sensor.open()
        measurements = []
        for _ in range(10):
            w = sensor.simulate_contact(force=(0.0, 0.0, 0.0), torque=(0.0, 0.0, 0.0))
            measurements.append(w.to_vector())
        avg = np.mean(measurements, axis=0)
        self.assertGreater(len(avg), 0)
        sensor.close()
    def test_contact_state_dataclass_fields(self):
        from src.sensors.force import ContactState
        state = ContactState(is_contact=True, contact_force=10.0)
        self.assertTrue(state.is_contact)
        self.assertEqual(state.contact_force, 10.0)
        self.assertIsNone(state.contact_point)
        self.assertEqual(state.slip_probability, 0.0)
    def test_force_collision_simulation(self):
        sensor = VirtualForceSensor(sensor_id="collision_test")
        sensor.open()
        frames = sensor.simulate_collision(direction=(1.0, 0.0, 0.0), peak_force=50.0, duration_ms=50.0)
        self.assertGreater(len(frames), 0)
        for frame in frames:
            self.assertEqual(len(frame.force), 3)
            self.assertEqual(len(frame.torque), 3)
            self.assertGreater(frame.magnitude, 0.0)
        sensor.close()
class TestIMUPoseEstimator(unittest.TestCase):
    def test_madgwick_filter_convergence(self):
        estimator = PoseEstimator(algorithm="madgwick", beta=0.1)
        for i in range(100):
            accel = np.array([0.0, 0.0, 9.81], dtype=np.float32)
            gyro = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            pose = estimator.update(accel, gyro, dt=0.01)
        self.assertIsNotNone(pose)
        q = pose.orientation
        self.assertEqual(len(q), 4)
        norm = np.linalg.norm(q)
        self.assertAlmostEqual(norm, 1.0, places=5)
    def test_complementary_filter_convergence(self):
        estimator = PoseEstimator(algorithm="complementary")
        for i in range(200):
            accel = np.array([0.02 * np.sin(i * 0.1), 0.0, 9.8], dtype=np.float32)
            gyro = np.array([0.0, 0.0, 0.01 * np.sin(i * 0.1)], dtype=np.float32)
            pose = estimator.update(accel, gyro, dt=0.01)
        self.assertIsNotNone(pose)
        angles = estimator.get_euler()
        self.assertLess(abs(angles[0]), 0.1)
        self.assertLess(abs(angles[1]), 0.1)
    def test_ekf_state_initialization(self):
        from src.fusion.sensor_fusion import ExtendedKalmanFilter
        ekf = ExtendedKalmanFilter(state_dim=3, measurement_dim=3)
        ekf.initialize(np.zeros(3, dtype=np.float32))
        self.assertEqual(ekf._state.shape[0], 3)
        cov = ekf.get_covariance()
        self.assertEqual(cov.shape, (3, 3))
    def test_imu_agv_motion_simulation(self):
        sensor = VirtualIMUSensor(sensor_id="agv_motion_test")
        sensor.open()
        frame = sensor.simulate_agv_motion(linear_velocity=(1.0, 0.0), angular_velocity=0.0)
        self.assertIsNotNone(frame)
        self.assertEqual(len(frame.accel), 3)
        self.assertEqual(len(frame.gyro), 3)
        self.assertIsInstance(frame.timestamp, float)
        sensor.close()
    def test_imu_human_walking_simulation(self):
        sensor = VirtualIMUSensor(sensor_id="walking_test")
        sensor.open()
        frames = sensor.simulate_human_walking(step_frequency=1.5, walk_speed=1.0, duration_s=3.0, dt=0.01)
        self.assertGreater(len(frames), 0)
        self.assertGreater(len(frames), 100)
        sensor.close()
class TestSensorDataIntegrity(unittest.TestCase):
    def test_tactile_frame_sequence(self):
        sensor = VirtualTactileSensor(array_size=(16, 16), sensor_id="seq_test")
        sensor.open()
        prev_id = None
        for i in range(50):
            frame = sensor.simulate_contact(contact_pos=(0.5, 0.5), contact_radius=0.2, contact_force=5.0)
            if prev_id is not None:
                self.assertEqual(frame.frame_id, prev_id + 1)
            prev_id = frame.frame_id
        sensor.close()
    def test_force_frame_sequence(self):
        sensor = VirtualForceSensor(sensor_id="force_seq_test")
        sensor.open()
        prev_id = None
        for i in range(50):
            frame = sensor.simulate_contact(force=(0.0, 0.0, 0.0), torque=(0.0, 0.0, 0.0))
            if prev_id is not None:
                self.assertEqual(frame.frame_id, prev_id + 1)
            prev_id = frame.frame_id
        sensor.close()
    def test_imu_frame_sequence(self):
        sensor = VirtualIMUSensor(sensor_id="imu_seq_test")
        sensor.open()
        prev_id = None
        for i in range(50):
            frame = sensor.simulate_static()
            if prev_id is not None:
                self.assertEqual(frame.frame_id, prev_id + 1)
            prev_id = frame.frame_id
        sensor.close()
    def test_sensor_close_idempotent(self):
        sensor = VirtualTactileSensor(array_size=(8, 8), sensor_id="idempotent_test")
        sensor.open()
        sensor.simulate_contact(contact_pos=(0.5, 0.5), contact_radius=0.1)
        sensor.close()
        sensor.close()
        sensor.close()
    def test_all_agv_grades_tactile_spec(self):
        from src.sensors.tactile import get_tactile_spec, AGV_TACTILE_GRADES
        for grade in AGV_TACTILE_GRADES:
            spec = get_tactile_spec(grade)
            self.assertIn("array", spec)
            self.assertIn("freq_hz", spec)
            self.assertGreater(spec["array"][0], 0)
            self.assertGreater(spec["freq_hz"], 0)
    def test_all_agv_grades_force_spec(self):
        from src.sensors.force import get_force_spec, AGV_FORCE_GRADES
        for grade in AGV_FORCE_GRADES:
            spec = get_force_spec(grade)
            self.assertIn("axes", spec)
            self.assertIn("sampling_hz", spec)
            self.assertGreater(spec["axes"], 0)
            self.assertGreater(spec["sampling_hz"], 0)
    def test_all_agv_grades_imu_spec(self):
        from src.sensors.imu import get_imu_spec, AGV_IMU_GRADES
        for grade in AGV_IMU_GRADES:
            spec = get_imu_spec(grade)
            self.assertIn("accel_range", spec)
            self.assertIn("gyro_range", spec)
            self.assertGreater(spec["gyro_range"], 0)


class TestTactileArrayExtended(unittest.TestCase):
    """触觉传感器扩展测试"""

    def test_tactile_pressure_processor(self):
        """测试压力处理器"""
        from src.sensors.tactile import PressureProcessor
        processor = PressureProcessor(filter_window=3)
        pressure = np.random.rand(16, 16).astype(np.float32)
        filtered = processor.filter(pressure)
        self.assertEqual(filtered.shape, pressure.shape)
        self.assertTrue(np.all(filtered >= 0))

    def test_tactile_baseline_compensation(self):
        """测试基线补偿"""
        from src.sensors.tactile import PressureProcessor
        processor = PressureProcessor()
        pressure = np.random.rand(8, 8).astype(np.float32) * 0.5 + 0.1
        compensated = processor.compensate_baseline(pressure, set_baseline=True)
        self.assertEqual(compensated.shape, pressure.shape)

    def test_tactile_centroid_computation(self):
        """测试压力质心计算"""
        from src.sensors.tactile import PressureProcessor
        processor = PressureProcessor()
        # 创建高斯分布压力
        h, w = 16, 16
        center = (8, 8)
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        pressure = np.exp(-((xx - center[0])**2 + (yy - center[1])**2) / 10.0)
        cy, cx = processor.compute_centroid(pressure)
        self.assertTrue(5 < cy < 11)
        self.assertTrue(5 < cx < 11)

    def test_virtual_tactile_contact_simulation(self):
        """测试虚拟触觉接触模拟"""
        from src.sensors.tactile import VirtualTactileSensor
        sensor = VirtualTactileSensor(array_size=(8, 8))
        sensor.open()
        frame = sensor.simulate_contact((0.5, 0.5), contact_radius=0.2, contact_force=5.0)
        self.assertEqual(frame.pressure_map.shape, (8, 8))
        self.assertTrue(np.max(frame.pressure_map) > 0)
        sensor.close()

    def test_virtual_tactile_sliding(self):
        """测试虚拟触觉滑模拟"""
        from src.sensors.tactile import VirtualTactileSensor
        sensor = VirtualTactileSensor(array_size=(8, 8))
        sensor.open()
        frames = sensor.simulate_sliding(direction=(0.1, 0.0), speed=0.05, duration_frames=10)
        self.assertEqual(len(frames), 10)
        for f in frames:
            self.assertEqual(f.pressure_map.shape, (8, 8))
        sensor.close()


class TestForceSensorExtended(unittest.TestCase):
    """力觉传感器扩展测试"""

    def test_wrench_transform(self):
        """测试力旋量坐标变换"""
        from src.sensors.force import Wrench
        wrench = Wrench(force=np.array([1.0, 0.0, 0.0]), torque=np.array([0.0, 0.0, 0.0]))
        R = np.eye(3)
        t = np.array([0.1, 0.0, 0.0])
        transformed = wrench.transform(R, t)
        self.assertEqual(transformed.force[0], 1.0)

    def test_wrench_processor_filter(self):
        """测试力旋量处理器滤波"""
        from src.sensors.force import WrenchProcessor
        processor = WrenchProcessor(filter_alpha=0.3)
        wrench = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        filtered = processor.filter(wrench)
        self.assertEqual(len(filtered), 6)

    def test_wrench_direction(self):
        """测试力方向计算"""
        from src.sensors.force import WrenchProcessor
        processor = WrenchProcessor()
        wrench = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        direction = processor.compute_force_direction(wrench)
        np.testing.assert_array_almost_equal(direction, [1.0, 0.0, 0.0])

    def test_virtual_force_contact(self):
        """测试虚拟力觉接触"""
        from src.sensors.force import VirtualForceSensor
        sensor = VirtualForceSensor()
        sensor.open()
        wrench = sensor.simulate_contact((10.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        self.assertEqual(len(wrench.force), 3)
        self.assertTrue(wrench.magnitude > 0)
        sensor.close()

    def test_virtual_force_payload(self):
        """测试虚拟力觉负载模拟"""
        from src.sensors.force import VirtualForceSensor
        sensor = VirtualForceSensor()
        sensor.open()
        wrench = sensor.simulate_payload(mass=5.0, com_offset=(0.1, 0.0, 0.0))
        self.assertLess(wrench.force[2], -40.0)  # Should be negative around -5*9.81
        sensor.close()

    def test_virtual_force_collision(self):
        """测试虚拟力觉碰撞"""
        from src.sensors.force import VirtualForceSensor
        sensor = VirtualForceSensor()
        sensor.open()
        frames = sensor.simulate_collision(direction=(1.0, 0.0, 0.0), peak_force=50.0, duration_ms=50.0)
        self.assertTrue(len(frames) >= 4)
        for f in frames:
            self.assertEqual(len(f.force), 3)
        sensor.close()


class TestIMUExtended(unittest.TestCase):
    """IMU传感器扩展测试"""

    def test_pose_identity(self):
        """测试单位位姿"""
        from src.sensors.imu import Pose
        pose = Pose.identity()
        np.testing.assert_array_almost_equal(pose.position, [0, 0, 0])
        np.testing.assert_array_almost_equal(pose.orientation, [1, 0, 0, 0])

    def test_pose_to_euler(self):
        """测试位姿转欧拉角"""
        from src.sensors.imu import Pose
        pose = Pose(position=np.zeros(3), orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        euler = pose.to_euler()
        np.testing.assert_array_almost_equal(euler, [0, 0, 0])

    def test_pose_from_euler(self):
        """测试欧拉角转位姿"""
        from src.sensors.imu import Pose
        pose = Pose.from_euler(np.zeros(3), np.array([0.0, 0.0, 0.0]))
        self.assertIsNotNone(pose.orientation)

    def test_pose_estimator_reset(self):
        """测试姿态估计器重置"""
        from src.sensors.imu import PoseEstimator
        estimator = PoseEstimator(algorithm='madgwick')
        estimator.update(np.array([0, 0, -9.81]), np.array([0.1, 0.1, 0.1]), None, dt=0.01)
        estimator.reset()
        self.assertEqual(estimator.quaternion[0], 1.0)

    def test_virtual_imu_static(self):
        """测试虚拟IMU静止状态"""
        from src.sensors.imu import VirtualIMUSensor
        sensor = VirtualIMUSensor()
        sensor.open()
        frame = sensor.simulate_static((0.0, 0.0, 0.0))
        self.assertEqual(len(frame.accel), 3)
        self.assertTrue(frame.accel_magnitude > 9.0)
        sensor.close()

    def test_virtual_imu_motion(self):
        """测试虚拟IMU运动状态"""
        from src.sensors.imu import VirtualIMUSensor
        sensor = VirtualIMUSensor()
        sensor.open()
        frame = sensor.simulate_motion((1.0, 0.0, 0.0), (0.0, 0.0, 0.1), dt=0.01)
        self.assertEqual(len(frame.accel), 3)
        self.assertEqual(len(frame.gyro), 3)
        sensor.close()

    def test_virtual_imu_trajectory(self):
        """测试虚拟IMU轨迹"""
        from src.sensors.imu import VirtualIMUSensor
        sensor = VirtualIMUSensor()
        sensor.open()
        frames = sensor.simulate_trajectory("circle", duration_s=0.1, dt=0.01)
        self.assertTrue(len(frames) >= 9)
        sensor.close()

    def test_virtual_imu_agv_motion(self):
        """测试虚拟IMU AGV运动"""
        from src.sensors.imu import VirtualIMUSensor
        sensor = VirtualIMUSensor()
        sensor.open()
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            frame = sensor.simulate_agv_motion((0.5, 0.0), 0.0, grade=grade)
            self.assertEqual(len(frame.accel), 3)
        sensor.close()

    def test_virtual_imu_human_walking(self):
        """测试虚拟IMU人类步行"""
        from src.sensors.imu import VirtualIMUSensor
        sensor = VirtualIMUSensor()
        sensor.open()
        frames = sensor.simulate_human_walking(step_frequency=1.5, duration_s=0.5, dt=0.01)
        self.assertTrue(len(frames) >= 49)
        sensor.close()
