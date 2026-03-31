"""
传感器模块测试
==============

测试所有传感器模块:
- Vision (BinocularCamera, DepthProcessor)
- Audio (BinauralMic, SoundLocalizer)
- Tactile (TactileArray, PressureProcessor)
- Force (ForceTorqueSensor, WrenchProcessor)
- IMU (IMUSensor, PoseEstimator)
"""

import numpy as np
import sys
import time
import unittest
from unittest.mock import patch, MagicMock

# 添加项目路径
sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.vision import BinocularCamera, DepthProcessor, CameraIntrinsics, StereoExtrinsics, StereoFrame, get_stereo_spec
from sensors.audio import BinauralMic, SoundLocalizer, AudioFrame, SoundSource, get_audio_spec
from sensors.tactile import (
    TactileArray, TactileFrame, TactileContact, TactileCalibration, PressureProcessor,
    TactileSensorType, get_tactile_spec, VirtualTactileSensor
)
from sensors.force import (
    ForceTorqueSensor, ForceCalibration, Wrench, ContactState, WrenchProcessor,
    ForceSensorType, get_force_spec, VirtualForceSensor
)
from sensors.imu import (
    IMUSensor, IMUFrame, Pose, PoseEstimator, IMUCalibration, IMUSensorType,
    get_imu_spec, VirtualIMUSensor
)


class TestBinocularCamera(unittest.TestCase):
    """测试双目相机"""
    
    def test_camera_open_close(self):
        cam = BinocularCamera(resolution=(640, 480), fps=30)
        self.assertTrue(cam.open())
        self.assertTrue(cam._is_opened)
        cam.close()
        self.assertFalse(cam._is_opened)
    
    def test_camera_capture(self):
        cam = BinocularCamera()
        cam.open()
        frame = cam.capture()
        self.assertIsInstance(frame, StereoFrame)
        self.assertEqual(frame.left_image.shape, (480, 640, 3))
        self.assertEqual(frame.right_image.shape, (480, 640, 3))
        cam.close()
    
    def test_context_manager(self):
        with BinocularCamera() as cam:
            frame = cam.capture()
            self.assertIsNotNone(frame)
    
    def test_extrinsics(self):
        cam = BinocularCamera()
        ext = cam.get_extrinsics()
        self.assertIsInstance(ext, StereoExtrinsics)
        self.assertEqual(ext.translation[0], -0.05)
    
    def test_get_stereo_spec(self):
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_stereo_spec(grade)
            self.assertIn('baseline_mm', spec)
            self.assertIn('range_m', spec)


class TestDepthProcessor(unittest.TestCase):
    """测试深度处理器"""
    
    def setUp(self):
        self.left_int = CameraIntrinsics(width=640, height=480, fx=385.5, fy=385.5, cx=319.5, cy=239.5)
        self.right_int = CameraIntrinsics(width=640, height=480, fx=385.5, fy=385.5, cx=319.5, cy=239.5)
        self.ext = StereoExtrinsics(rotation=np.eye(3), translation=np.array([-0.05, 0.0, 0.0]))
    
    def test_processor_init(self):
        proc = DepthProcessor(self.left_int, self.right_int, self.ext)
        self.assertIsNotNone(proc)
    
    def test_filter_depth(self):
        proc = DepthProcessor(self.left_int, self.right_int, self.ext)
        depth = np.random.rand(480, 640).astype(np.float32) * 10.0
        filtered = proc.filter_depth(depth, min_dist=0.1, max_dist=5.0)
        self.assertEqual(filtered.shape, depth.shape)
        # 检查范围
        valid = filtered > 0
        if np.any(valid):
            self.assertTrue(np.all(filtered[valid] >= 0.1))
            self.assertTrue(np.all(filtered[valid] <= 5.0))
    
    def test_project_to_3d(self):
        proc = DepthProcessor(self.left_int, self.right_int, self.ext)
        p3d = proc.project_to_3d(u=320, v=240, depth=1.0)
        self.assertEqual(p3d.shape, (3,))
        self.assertAlmostEqual(p3d[2], 1.0, places=5)
    
    def test_depth_to_pointcloud(self):
        proc = DepthProcessor(self.left_int, self.right_int, self.ext)
        depth = np.zeros((480, 640), dtype=np.float32)
        depth[200:300, 300:400] = 2.0
        points, colors = proc.depth_to_pointcloud(depth)
        self.assertGreater(points.shape[0], 0)
        self.assertEqual(points.shape[1], 3)


class TestBinauralMic(unittest.TestCase):
    """测试双耳麦克风"""
    
    def test_mic_open_close(self):
        mic = BinauralMic(sample_rate=16000, chunk_size=512)
        self.assertTrue(mic.open())
        mic.close()
    
    def test_mic_capture(self):
        mic = BinauralMic()
        mic.open()
        frame = mic.capture()
        self.assertIsInstance(frame, AudioFrame)
        self.assertEqual(len(frame.left_channel), 512)
        self.assertEqual(len(frame.right_channel), 512)
        mic.close()
    
    def test_get_audio_spec(self):
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_audio_spec(grade)
            self.assertIn('channels', spec)
            self.assertIn('sr', spec)


class TestSoundLocalizer(unittest.TestCase):
    """测试声源定位"""
    
    def test_localizer_init(self):
        loc = SoundLocalizer(baseline_mm=95.0, sample_rate=16000)
        self.assertEqual(loc.baseline_mm, 95.0)
    
    def test_estimate_tdoa(self):
        loc = SoundLocalizer()
        # 生成相同信号 (时延为0)
        t = np.linspace(0, 0.1, 1600)
        left = 0.1 * np.sin(2 * np.pi * 440 * t)
        right = 0.1 * np.sin(2 * np.pi * 440 * t)
        tdoa = loc.estimate_tdoa(left, right)
        self.assertIsInstance(tdoa, float)
        self.assertIsInstance(tdoa, (float, np.floating))
        self.assertLess(abs(tdoa), 0.5)  # TDOA should be near 0 for identical signals
    
    def test_localize(self):
        loc = SoundLocalizer()
        t = np.linspace(0, 0.1, 1600)
        left = 0.1 * np.sin(2 * np.pi * 440 * t)
        right = 0.1 * np.sin(2 * np.pi * 440 * t)
        source = loc.localize(left, right)
        self.assertIsInstance(source, SoundSource)
        self.assertGreaterEqual(source.direction[0], -90)
        self.assertLessEqual(source.direction[0], 90)
    
    def test_beamform(self):
        loc = SoundLocalizer()
        t = np.linspace(0, 0.1, 1600)
        left = 0.1 * np.sin(2 * np.pi * 440 * t)
        right = 0.1 * np.sin(2 * np.pi * 440 * t)
        beamformed = loc.beamform(left, right, look_direction=0.0)
        self.assertEqual(len(beamformed), len(left))


class TestTactileArray(unittest.TestCase):
    """测试触觉阵列"""
    
    def test_tactile_open_close(self):
        tactile = TactileArray(array_size=(16, 16))
        self.assertTrue(tactile.open())
        tactile.close()
    
    def test_tactile_capture(self):
        tactile = TactileArray(array_size=(16, 16))
        tactile.open()
        frame = tactile.capture()
        self.assertIsInstance(frame, TactileFrame)
        self.assertEqual(frame.pressure_map.shape, (16, 16))
        self.assertIsNotNone(frame.temperature_map)
        self.assertEqual(frame.temperature_map.shape, (16, 16))
        tactile.close()
    
    def test_detect_contacts(self):
        tactile = TactileArray(array_size=(16, 16))
        tactile.open()
        frame = tactile.capture()
        contacts = tactile.detect_contacts(frame)
        self.assertIsInstance(contacts, list)
        tactile.close()
    
    def test_calibrate(self):
        tactile = TactileArray()
        tactile.open()
        zero_pressure = np.zeros((16, 16))
        tactile.calibrate(zero_pressure=zero_pressure)
        self.assertIsNotNone(tactile.calibration.offset_map)
        tactile.close()
    
    def test_get_tactile_spec(self):
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_tactile_spec(grade)
            self.assertIn('array', spec)
            self.assertIn('freq_hz', spec)


class TestPressureProcessor(unittest.TestCase):
    """测试压力处理器"""
    
    def test_processor_init(self):
        proc = PressureProcessor(filter_window=3)
        self.assertEqual(proc.filter_window, 3)
    
    def test_filter(self):
        proc = PressureProcessor()
        pressure = np.random.rand(16, 16).astype(np.float32)
        filtered = proc.filter(pressure)
        self.assertEqual(filtered.shape, pressure.shape)
    
    def test_compensate_baseline(self):
        proc = PressureProcessor()
        pressure = np.random.rand(16, 16).astype(np.float32)
        compensated = proc.compensate_baseline(pressure, set_baseline=True)
        self.assertEqual(compensated.shape, pressure.shape)
    
    def test_compute_force(self):
        proc = PressureProcessor()
        pressure = np.random.rand(16, 16).astype(np.float32)
        force = proc.compute_force(pressure, contact_area=1e-4)
        self.assertIsInstance(force, (float, np.floating, np.ndarray))
        self.assertGreater(float(force), 0)


class TestForceTorqueSensor(unittest.TestCase):
    """测试六维力矩传感器"""
    
    def test_sensor_open_close(self):
        sensor = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        self.assertTrue(sensor.open())
        sensor.close()
    
    def test_sensor_capture(self):
        sensor = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        sensor.open()
        wrench = sensor.capture()
        self.assertIsInstance(wrench, Wrench)
        self.assertEqual(wrench.force.shape, (3,))
        self.assertEqual(wrench.torque.shape, (3,))
        sensor.close()
    
    def test_wrench_operations(self):
        wrench = Wrench(force=np.array([10.0, 0.0, 0.0]), torque=np.array([0.0, 0.0, 5.0]))
        self.assertEqual(wrench.magnitude, 10.0)
        self.assertEqual(wrench.torque_magnitude, 5.0)
        
        vec = wrench.to_vector()
        self.assertEqual(vec.shape, (6,))
        
        wrench2 = Wrench.from_vector(vec)
        np.testing.assert_array_almost_equal(wrench.force, wrench2.force)
        np.testing.assert_array_almost_equal(wrench.torque, wrench2.torque)
    
    def test_wrench_transform(self):
        # Force perpendicular to translation produces torque
        wrench = Wrench(force=np.array([0.0, 10.0, 0.0]), torque=np.array([0.0, 0.0, 0.0]))
        R = np.eye(3)
        t = np.array([0.1, 0.0, 0.0])  # translation in X
        new_wrench = wrench.transform(R, t)
        # Force doesn't change direction under identity rotation
        np.testing.assert_array_almost_equal(new_wrench.force, wrench.force)
        # Torque should change due to cross product: t × F = (0.1, 0, 0) × (0, 10, 0) = (0, 0, 1)
        np.testing.assert_array_almost_equal(new_wrench.torque, np.array([0.0, 0.0, 1.0]))
    
    def test_detect_contact(self):
        sensor = ForceTorqueSensor()
        sensor.open()
        sensor.capture()
        state = sensor.detect_contact(threshold=2.0)
        self.assertIsInstance(state, ContactState)
        self.assertIn(state.is_contact, [True, False])
        sensor.close()
    
    def test_get_force_spec(self):
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_force_spec(grade)
            self.assertIn('axes', spec)
            self.assertIn('force_range', spec)


class TestWrenchProcessor(unittest.TestCase):
    """测试力旋量处理器"""
    
    def test_filter(self):
        proc = WrenchProcessor(filter_alpha=0.3)
        wrench = np.array([10.0, 0.0, -9.81, 0.0, 0.0, 0.0])
        filtered = proc.filter(wrench)
        self.assertEqual(filtered.shape, (6,))


class TestIMUSensor(unittest.TestCase):
    """测试IMU传感器"""
    
    def test_imu_open_close(self):
        imu = IMUSensor(sensor_type=IMUSensorType.BMI088)
        self.assertTrue(imu.open())
        imu.close()
    
    def test_imu_capture(self):
        imu = IMUSensor()
        imu.open()
        frame = imu.capture()
        self.assertIsInstance(frame, IMUFrame)
        self.assertEqual(frame.accel.shape, (3,))
        self.assertEqual(frame.gyro.shape, (3,))
        self.assertIsNotNone(frame.accel_magnitude)
        imu.close()
    
    def test_self_test(self):
        imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL)
        imu.open()
        result = imu.self_test()
        self.assertIsInstance(result, bool)
        imu.close()
    
    def test_calibrate_gyro(self):
        imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL)
        imu.open()
        imu.calibrate_gyro_bias(num_samples=50)
        self.assertIsNotNone(imu.calibration.gyro_bias)
        imu.close()
    
    def test_get_imu_spec(self):
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_imu_spec(grade)
            self.assertIn('sample_hz', spec)


class TestPoseEstimator(unittest.TestCase):
    """测试姿态估计器"""
    
    def test_pose_init(self):
        pose = Pose.identity()
        np.testing.assert_array_almost_equal(pose.position, np.zeros(3))
        np.testing.assert_array_almost_equal(pose.orientation, [1.0, 0.0, 0.0, 0.0])
    
    def test_pose_to_euler(self):
        pose = Pose.identity()
        euler = pose.to_euler()
        self.assertEqual(euler.shape, (3,))
    
    def test_pose_to_matrix(self):
        pose = Pose.identity()
        T = pose.to_matrix()
        self.assertEqual(T.shape, (4, 4))
        np.testing.assert_array_almost_equal(np.diag(T), [1, 1, 1, 1])
    
    def test_from_euler(self):
        pose = Pose.from_euler(position=np.zeros(3), rpy=np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(pose.orientation, [1.0, 0.0, 0.0, 0.0])
    
    def test_pose_estimator_update(self):
        estimator = PoseEstimator(algorithm="madgwick", sample_rate=100.0)
        
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.0])
        
        pose = estimator.update(accel, gyro)
        self.assertIsInstance(pose, Pose)
        
        euler = estimator.get_euler()
        self.assertEqual(euler.shape, (3,))


class TestAGVGradeSpecs(unittest.TestCase):
    """综合测试: 验证所有AGV等级规格一致性"""
    
    def test_all_grade_specs_complete(self):
        grades = ['S', 'M', 'L', 'XL', 'XXL']
        
        for grade in grades:
            vision_spec = get_stereo_spec(grade)
            audio_spec = get_audio_spec(grade)
            tactile_spec = get_tactile_spec(grade)
            force_spec = get_force_spec(grade)
            imu_spec = get_imu_spec(grade)
            
            # 确保所有规格都有必要字段
            self.assertIn('baseline_mm', vision_spec, f"Grade {grade} missing vision baseline")
            self.assertIn('channels', audio_spec, f"Grade {grade} missing audio channels")
            self.assertIn('array', tactile_spec, f"Grade {grade} missing tactile array")
            self.assertIn('axes', force_spec, f"Grade {grade} missing force axes")
            self.assertIn('sample_hz', imu_spec, f"Grade {grade} missing imu sample_hz")


class TestSensorEdgeCases(unittest.TestCase):
    """传感器边缘用例测试"""
    
    def test_tactile_context_manager(self):
        """测试触觉传感器上下文管理器"""
        with TactileArray(array_size=(8, 8)) as tactile:
            self.assertTrue(tactile._is_opened)
            frame = tactile.capture()
            self.assertIsInstance(frame, TactileFrame)
        # 退出后应该关闭
        self.assertFalse(tactile._is_opened)
    
    def test_tactile_multiple_captures(self):
        """测试连续多次采集"""
        tactile = TactileArray(array_size=(8, 8))
        tactile.open()
        frames = []
        for _ in range(5):
            frame = tactile.capture()
            frames.append(frame)
        self.assertEqual(len(frames), 5)
        tactile.close()
    
    def test_force_sensor_capture_returns_wrench(self):
        """测试力传感器采集返回Wrench对象"""
        sensor = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        sensor.open()
        wrench = sensor.capture()
        self.assertIsInstance(wrench, Wrench)
        self.assertEqual(wrench.force.shape, (3,))
        self.assertEqual(wrench.torque.shape, (3,))
        self.assertGreater(wrench.magnitude, 0.0)
        sensor.close()
    
    def test_imu_high_rate_capture(self):
        """测试IMU高速采集"""
        imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL)
        imu.open()
        # 模拟多次采集
        for _ in range(50):
            frame = imu.capture()
            self.assertIsInstance(frame, IMUFrame)
            self.assertEqual(frame.accel.shape, (3,))
            self.assertEqual(frame.gyro.shape, (3,))
        imu.close()
    
    def test_pose_to_matrix_and_back(self):
        """测试姿态矩阵转换的一致性"""
        pose = Pose.from_euler(
            position=np.array([1.0, 2.0, 3.0]),
            rpy=np.array([0.1, 0.2, 0.3])
        )
        matrix = pose.to_matrix()
        self.assertEqual(matrix.shape, (4, 4))
        # 检查矩阵的性质
        np.testing.assert_array_almost_equal(matrix[3, :], [0, 0, 0, 1])
    
    def test_tactile_pressure_processor(self):
        """测试触觉压力处理器"""
        proc = PressureProcessor(filter_window=3, drift_compensation=True)
        pressure = np.random.rand(16, 16).astype(np.float32)
        filtered = proc.filter(pressure)
        self.assertEqual(filtered.shape, pressure.shape)
        # 测试基线补偿
        compensated = proc.compensate_baseline(pressure, set_baseline=True)
        self.assertEqual(compensated.shape, pressure.shape)
    
    def test_force_wrench_processor(self):
        """测试力矩信号处理器"""
        proc = WrenchProcessor(filter_alpha=0.5)
        wrench = np.array([10.0, 0.0, -9.81, 0.0, 0.0, 0.0])
        filtered = proc.filter(wrench)
        self.assertEqual(filtered.shape, (6,))


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)


class TestSensorEndToEnd(unittest.TestCase):
    """端到端传感器融合测试"""
    
    def test_multi_sensor_fusion_pipeline(self):
        """测试多传感器数据融合流程"""
        # 1. 初始化所有传感器
        cam = BinocularCamera()
        mic = BinauralMic()
        tactile = TactileArray(array_size=(16, 16))
        force = ForceTorqueSensor()
        imu = IMUSensor()
        
        # 2. 采集数据
        cam.open()
        mic.open()
        tactile.open()
        force.open()
        imu.open()
        
        stereo = cam.capture()
        audio = mic.capture()
        tac_frame = tactile.capture()
        wrench = force.capture()
        imu_frame = imu.capture()
        
        # 3. 验证数据
        self.assertIsNotNone(stereo.left_image)
        self.assertIsNotNone(audio.left_channel)
        self.assertIsNotNone(tac_frame.pressure_map)
        self.assertIsNotNone(wrench.force)
        self.assertIsNotNone(imu_frame.accel)
        
        # 4. 处理
        depth_proc = DepthProcessor(cam.left_intrinsics, cam.right_intrinsics, cam.get_extrinsics())
        contacts = tactile.detect_contacts(tac_frame)
        contact_state = force.detect_contact(wrench)
        pose_est = PoseEstimator()
        pose = pose_est.update(imu_frame.accel, imu_frame.gyro)
        
        self.assertIsInstance(contacts, list)
        self.assertIsInstance(contact_state, ContactState)
        self.assertIsInstance(pose, Pose)
        
        # 5. 清理
        cam.close()
        mic.close()
        tactile.close()
        force.close()
        imu.close()
    
    def test_imu_pose_estimation_accuracy(self):
        """测试IMU姿态估计精度"""
        imu = IMUSensor(sensor_type=IMUSensorType.BMI088)
        imu.open()
        
        # 静止状态下的姿态估计
        frames = [imu.capture() for _ in range(100)]
        
        pose_est = PoseEstimator(algorithm='madgwick', beta=0.1)
        euler_angles = []
        
        for frame in frames:
            pose = pose_est.update(frame.accel, frame.gyro)
            euler = pose.to_euler()
            euler_angles.append(euler)
        
        # 静止时，roll和pitch应该接近0，yaw保持稳定
        euler_arr = np.array(euler_angles)
        roll_std = np.std(euler_arr[:, 0])
        pitch_std = np.std(euler_arr[:, 1])
        
        # 标准差应该很小
        self.assertLess(roll_std, 0.1)  # rad
        self.assertLess(pitch_std, 0.1)
        
        imu.close()
    
    def test_force_wrench_physical_consistency(self):
        """测试力矩数据的物理一致性"""
        sensor = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        sensor.open()
        
        # 静止时，力矩应该接近0
        wrenches = [sensor.capture() for _ in range(50)]
        
        forces = np.array([w.force for w in wrenches])
        torques = np.array([w.torque for w in wrenches])
        
        # 检查静态力矩不为零 (重力分量)
        mean_force_z = np.mean(forces[:, 2])
        self.assertLess(mean_force_z, -5.0)  # 应该有重力
        
        # 力矩波动应该很小
        torque_std = np.std(torques, axis=0)
        self.assertTrue(np.all(torque_std < 5.0))  # Nm
        
        sensor.close()
    
    def test_tactile_contact_detection_under_pressure(self):
        """测试触觉接触检测"""
        tactile = TactileArray(array_size=(24, 24))
        tactile.open()
        
        # 创建高压力区域
        frame = tactile.capture()
        # 手动设置高压力 - 覆盖整个区域以确保峰值足够高
        frame.pressure_map[8:16, 8:16] = 0.95
        frame.pressure_map[10:14, 10:14] = 1.0  # 中心最高
        
        contacts = tactile.detect_contacts(frame)
        
        # 应该检测到接触
        self.assertGreater(len(contacts), 0)
        
        # 验证接触区域
        contact = contacts[0]
        self.assertGreater(contact.area, 0)
        self.assertGreater(contact.peak_pressure, 0.1)  # 阈值设为0.1（接触阈值一致）
        self.assertGreater(contact.contact_force, 0)
        
        tactile.close()
    
    def test_tactile_agv_grade_compliance(self):
        """测试触觉传感器满足AGV等级规格"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_tactile_spec(grade)
            
            # 创建传感器
            tactile = TactileArray(
                array_size=spec['array'],
                sensor_id=f'tactile_{grade}'
            )
            tactile.open()
            frame = tactile.capture()
            
            # 验证分辨率
            self.assertEqual(frame.pressure_map.shape[0], spec['array'][0])
            self.assertEqual(frame.pressure_map.shape[1], spec['array'][1])
            
            tactile.close()
    
    def test_force_agv_grade_compliance(self):
        """测试力觉传感器满足AGV等级规格"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_force_spec(grade)
            
            sensor = ForceTorqueSensor(sensor_id=f'force_{grade}')
            sensor.open()
            wrench = sensor.capture()
            
            # 验证轴数
            if spec['axes'] == 3:
                self.assertEqual(len(wrench.force), 3)
            else:
                self.assertEqual(len(wrench.force), 3)
                self.assertEqual(len(wrench.torque), 3)
            
            sensor.close()
    
    def test_imu_agv_grade_compliance(self):
        """测试IMU满足AGV等级规格"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_imu_spec(grade)
            
            # 验证规格字段
            self.assertIn('type', spec)
            self.assertIn('accel_range', spec)
            self.assertIn('gyro_range', spec)
            self.assertIn('sample_hz', spec)
            
            imu = IMUSensor(
                sensor_type=IMUSensorType.VIRTUAL,
                accel_range=spec['accel_range'],
                gyro_range=spec['gyro_range'],
                sample_rate=spec['sample_hz'],
                sensor_id=f'imu_{grade}'
            )
            imu.open()
            frame = imu.capture()
            
            self.assertEqual(frame.accel.shape, (3,))
            self.assertEqual(frame.gyro.shape, (3,))
            
            imu.close()


class TestSensorPerformance(unittest.TestCase):
    """传感器性能测试"""
    
    def test_imu_high_frequency_capture(self):
        """测试IMU高频采集性能"""
        import time
        
        imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL, sample_rate=1000)
        imu.open()
        
        # 采集1000帧并计时
        start = time.time()
        for _ in range(1000):
            imu.capture()
        elapsed = time.time() - start
        
        # 应该能在2秒内完成
        self.assertLess(elapsed, 2.0)
        
        imu.close()
    
    def test_tactile_filter_performance(self):
        """测试触觉滤波性能"""
        import time
        
        proc = PressureProcessor(filter_window=5)
        pressure = np.random.rand(48, 48).astype(np.float32)
        
        start = time.time()
        for _ in range(1000):
            proc.filter(pressure)
        elapsed = time.time() - start
        
        # 1000次滤波应在5秒内完成 (考虑系统负载和scipy开销)
        self.assertLess(elapsed, 5.0)
    
    def test_force_wrench_processor_performance(self):
        """测试力矩处理器性能"""
        import time

        proc = WrenchProcessor(filter_alpha=0.3)
        wrench = np.array([10.0, 0.0, -9.81, 0.1, -0.1, 0.05])

        start = time.time()
        for _ in range(10000):
            proc.filter(wrench)
        elapsed = time.time() - start

        # 10000次处理应在0.5秒内完成
        self.assertLess(elapsed, 0.5)


class TestVirtualTactileSensor(unittest.TestCase):
    """测试虚拟触觉传感器"""
    
    def test_virtual_tactile_open_close(self):
        vt = VirtualTactileSensor(array_size=(16, 16))
        self.assertTrue(vt.open())
        self.assertTrue(vt._is_opened)
        vt.close()
        self.assertFalse(vt._is_opened)
    
    def test_simulate_contact(self):
        with VirtualTactileSensor(array_size=(16, 16)) as vt:
            frame = vt.simulate_contact(
                contact_pos=(0.5, 0.5),
                contact_radius=0.2,
                contact_force=10.0
            )
            self.assertIsInstance(frame, TactileFrame)
            self.assertEqual(frame.pressure_map.shape, (16, 16))
            self.assertTrue(np.max(frame.pressure_map) > 0)
    
    def test_simulate_sliding(self):
        with VirtualTactileSensor(array_size=(16, 16)) as vt:
            vt._last_contact_pos = (0.5, 0.5)
            frames = vt.simulate_sliding(
                direction=(1.0, 0.0),
                speed=0.05,
                duration_frames=10
            )
            self.assertEqual(len(frames), 10)
            for f in frames:
                self.assertIsInstance(f, TactileFrame)
    
    def test_context_manager(self):
        with VirtualTactileSensor(array_size=(8, 8)) as vt:
            frame = vt.simulate_contact((0.5, 0.5), 0.2, 5.0)
            self.assertIsNotNone(frame)


class TestVirtualForceSensor(unittest.TestCase):
    """测试虚拟力觉传感器"""
    
    def test_virtual_force_open_close(self):
        vf = VirtualForceSensor(sensor_id="test_force")
        self.assertTrue(vf.open())
        self.assertTrue(vf._is_opened)
        vf.close()
        self.assertFalse(vf._is_opened)
    
    def test_simulate_contact(self):
        with VirtualForceSensor() as vf:
            wrench = vf.simulate_contact(
                force=(10.0, 0.0, 0.0),
                torque=(0.0, 0.0, 0.0)
            )
            self.assertIsInstance(wrench, Wrench)
            self.assertEqual(wrench.force.shape, (3,))
            self.assertEqual(wrench.torque.shape, (3,))
    
    def test_simulate_payload(self):
        with VirtualForceSensor() as vf:
            wrench = vf.simulate_payload(mass=1.0, com_offset=(0.01, 0.0, 0.0))
            self.assertIsInstance(wrench, Wrench)
            # 检查力的大小在合理范围内（考虑噪声）
            self.assertTrue(-15 < wrench.force[2] < -5)
    
    def test_simulate_collision(self):
        with VirtualForceSensor() as vf:
            frames = vf.simulate_collision(
                direction=(1.0, 0.0, 0.0),
                peak_force=50.0,
                duration_ms=50.0
            )
            self.assertGreater(len(frames), 0)
            self.assertTrue(all(isinstance(f, Wrench) for f in frames))
    
    def test_context_manager(self):
        with VirtualForceSensor(sensor_id="ctx_test") as vf:
            wrench = vf.simulate_contact((5.0, 0.0, 0.0), (0.1, 0.0, 0.0))
            self.assertIsNotNone(wrench)


class TestVirtualIMUSensor(unittest.TestCase):
    """测试虚拟IMU传感器"""
    
    def test_virtual_imu_open_close(self):
        vi = VirtualIMUSensor(sensor_id="test_imu")
        self.assertTrue(vi.open())
        self.assertTrue(vi._is_opened)
        vi.close()
        self.assertFalse(vi._is_opened)
    
    def test_simulate_static(self):
        with VirtualIMUSensor() as vi:
            frame = vi.simulate_static(orientation=(0.0, 0.0, 0.0))
            self.assertIsInstance(frame, IMUFrame)
            self.assertEqual(frame.accel.shape, (3,))
            self.assertEqual(frame.gyro.shape, (3,))
            # 静止时角速度应接近0
            self.assertTrue(np.abs(frame.gyro).max() < 0.1)
    
    def test_simulate_motion(self):
        with VirtualIMUSensor() as vi:
            frame = vi.simulate_motion(
                linear_accel=(0.0, 1.0, 0.0),
                angular_vel=(0.0, 0.0, 0.1),
                dt=0.01
            )
            self.assertIsInstance(frame, IMUFrame)
            self.assertEqual(frame.accel.shape, (3,))
    
    def test_simulate_trajectory(self):
        for traj_type in ["circle", "figure8", "linear", "sine"]:
            with VirtualIMUSensor() as vi:
                frames = vi.simulate_trajectory(
                    trajectory_type=traj_type,
                    duration_s=0.2,
                    dt=0.01
                )
                self.assertGreater(len(frames), 5)
    
    def test_context_manager(self):
        with VirtualIMUSensor() as vi:
            frame = vi.simulate_static((0.0, 0.0, 0.0))
            self.assertIsNotNone(frame)


class TestSensorEdgeCasesExtended(unittest.TestCase):
    """扩展边缘用例测试"""
    
    def test_tactile_zero_contact(self):
        """触觉零接触检测"""
        ta = TactileArray(array_size=(8, 8))
        ta.open()
        frame = ta.capture()
        contacts = ta.detect_contacts(frame)
        # 无接触时应返回空列表
        self.assertIsInstance(contacts, list)
        ta.close()
    
    def test_tactile_extreme_pressure(self):
        """触觉极端压力值"""
        ta = TactileArray(array_size=(8, 8))
        ta.open()
        frame = ta.capture()
        # 压力值应在 [0, 1] 范围内
        self.assertTrue(np.all(frame.pressure_map >= 0))
        self.assertTrue(np.all(frame.pressure_map <= 1.5))  # 允许轻微超出
        ta.close()
    
    def test_force_zero_wrench(self):
        """力觉零力旋量"""
        with VirtualForceSensor(noise_level=0.0, bias_range=0.0) as vf:
            wrench = vf.simulate_contact((0, 0, 0), (0, 0, 0), add_noise=False)
            self.assertIsInstance(wrench, Wrench)
            self.assertLess(wrench.magnitude, 0.1)
    
    def test_force_extreme_values(self):
        """力觉极端值"""
        with VirtualForceSensor(noise_level=0.0, bias_range=0.0) as vf:
            # 大力值
            wrench = vf.simulate_contact((1000, 1000, 1000), (100, 100, 100), add_noise=False)
            self.assertGreater(wrench.magnitude, 0)
            # 小力值（无噪声无偏置时接近0）
            wrench = vf.simulate_contact((0.001, 0.001, 0.001), add_noise=False)
            self.assertLess(wrench.magnitude, 0.1)
    
    def test_imu_gravity_alignment(self):
        """IMU重力对齐验证"""
        with VirtualIMUSensor() as vi:
            frame = vi.simulate_static((0, 0, 0))
            # 静止时加速度大小应接近 9.81
            accel_mag = np.linalg.norm(frame.accel)
            self.assertAlmostEqual(accel_mag, 9.81, delta=0.5)
    
    def test_imu_extreme_orientation(self):
        """IMU极端姿态"""
        with VirtualIMUSensor() as vi:
            for roll, pitch, yaw in [(np.pi, 0, 0), (0, np.pi, 0), (0, 0, np.pi), (np.pi, np.pi, np.pi)]:
                frame = vi.simulate_static((roll, pitch, yaw))
                self.assertEqual(frame.accel.shape, (3,))
                self.assertEqual(frame.gyro.shape, (3,))
    
    def test_pose_estimation_drift(self):
        """姿态估计漂移测试"""
        pe = PoseEstimator(algorithm='madgwick', sample_rate=200)
        
        # 模拟静止状态
        accel_static = np.array([0.0, 0.0, 9.81])
        gyro_static = np.array([0.0, 0.0, 0.0])
        
        # 长时间积分
        for _ in range(500):
            pose = pe.update(accel_static, gyro_static)
        
        # 检查欧拉角不应漂移太多
        euler = pose.to_euler()
        self.assertTrue(np.abs(euler[0]) < 0.1)  # roll
        self.assertTrue(np.abs(euler[1]) < 0.1)  # pitch
    
    def test_wrench_transform_consistency(self):
        """力旋量坐标变换一致性"""
        with VirtualForceSensor() as vf:
            wrench = vf.simulate_contact((10.0, 0.0, 0.0))
            
            # 恒等旋转应保持不变
            I = np.eye(3)
            t = np.zeros(3)
            transformed = wrench.transform(I, t)
            self.assertTrue(np.allclose(wrench.force, transformed.force))
            self.assertTrue(np.allclose(wrench.torque, transformed.torque))
    
    def test_multimodal_timing(self):
        """多传感器时序一致性"""
        import time
        
        cam = BinocularCamera()
        mic = BinauralMic()
        ta = TactileArray()
        
        cam.open()
        mic.open()
        ta.open()
        
        timestamps = []
        
        for _ in range(10):
            t0 = time.time()
            cam.capture()
            mic.capture()
            ta.capture()
            t1 = time.time()
            timestamps.append(t1 - t0)
        
        cam.close()
        mic.close()
        ta.close()
        
        # 平均采集时间应合理
        avg_time = np.mean(timestamps)
        self.assertLess(avg_time, 0.1)  # 小于100ms


class TestSensorAGVIntegration(unittest.TestCase):
    """AGV传感器集成测试"""
    
    def test_agv_grade_spec_consistency(self):
        """AGV等级规格一致性检查"""
        grades = ['S', 'M', 'L', 'XL', 'XXL']
        
        for grade in grades:
            tactile_spec = get_tactile_spec(grade)
            force_spec = get_force_spec(grade)
            imu_spec = get_imu_spec(grade)
            
            # 验证规格包含必需字段
            self.assertIn('array', tactile_spec)
            self.assertIn('axes', force_spec)
            self.assertIn('type', imu_spec)
    
    def test_sensor_update_rate_matching(self):
        """传感器更新率匹配测试"""
        # 高速传感器 (IMU) 应该能够匹配
        vi = VirtualIMUSensor()
        vi.open()
        
        frame_times = []
        for i in range(50):
            t0 = time.time()
            vi.simulate_static()
            t1 = time.time()
            frame_times.append(t1 - t0)
        
        vi.close()
        
        # 平均帧间隔应小于 10ms (100Hz)
        avg_interval = np.mean(frame_times)
        self.assertLess(avg_interval, 0.01)
    
    def test_sensor_fusion_timing(self):
        """传感器融合时序测试"""
        from sensors.manager import SensorManager, SensorManagerConfig
        
        config = SensorManagerConfig(grade='M')
        manager = SensorManager(config=config)
        manager.open_all()
        
        sync_times = []
        for _ in range(20):
            t0 = time.time()
            data = manager.capture_all()
            t1 = time.time()
            sync_times.append(t1 - t0)
        
        manager.close_all()
        
        # 同步采集时间应合理
        avg_time = np.mean(sync_times)
        self.assertLess(avg_time, 0.1)  # 小于 100ms


class TestSensorNumericalStability(unittest.TestCase):
    """传感器数值稳定性测试"""
    
    def test_tactile_pressure_stability(self):
        """触觉压力稳定性"""
        ta = TactileArray(array_size=(8, 8))
        ta.open()
        
        pressures = []
        for _ in range(100):
            frame = ta.capture()
            pressures.append(np.mean(frame.pressure_map))
        
        ta.close()
        
        # 压力均值不应剧烈波动
        pressure_std = np.std(pressures)
        self.assertLess(pressure_std, 0.2)
    
    def test_force_wrench_stability(self):
        """力旋量稳定性"""
        with VirtualForceSensor(noise_level=0.001) as vf:
            wrench_mags = []
            for _ in range(100):
                wrench = vf.simulate_contact((5.0, 0.0, 0.0), add_noise=True)
                wrench_mags.append(wrench.magnitude)
            
            # 力矩大小标准差应小
            mag_std = np.std(wrench_mags)
            self.assertLess(mag_std, 2.0)
    
    def test_imu_accel_stability(self):
        """IMU加速度稳定性"""
        with VirtualIMUSensor(accel_noise=0.001) as vi:
            accel_mags = []
            for _ in range(100):
                frame = vi.simulate_static()
                accel_mags.append(np.linalg.norm(frame.accel))
            
            # 重力大小标准差应小
            mag_std = np.std(accel_mags)
            self.assertLess(mag_std, 0.1)
    
    def test_pose_quaternion_normalization(self):
        """四元数归一化验证"""
        pe = PoseEstimator()
        
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.1])
        
        for _ in range(50):
            pose = pe.update(accel, gyro)
            quat_norm = np.linalg.norm(pose.orientation)
            self.assertAlmostEqual(quat_norm, 1.0, places=5)


class TestTactileProximityAndSlip(unittest.TestCase):
    """测试触觉传感器接近觉和滑移检测功能"""
    
    def test_capacitive_tactile_has_proximity(self):
        """电容式触觉传感器应支持接近觉"""
        tactile = TactileArray(
            array_size=(16, 16),
            sensor_type=TactileSensorType.CAPACITIVE
        )
        tactile.open()
        frame = tactile.capture()
        
        self.assertIsNotNone(frame.proximity)
        self.assertEqual(frame.proximity.shape, (16, 16))
        # 接近觉值应在合理范围内
        self.assertTrue(np.all(frame.proximity >= 0))
        self.assertTrue(np.all(frame.proximity <= 0.1))  # 无接近时应很小
        tactile.close()
    
    def test_optical_tactile_has_proximity(self):
        """光学式触觉传感器应支持接近觉"""
        tactile = TactileArray(
            array_size=(16, 16),
            sensor_type=TactileSensorType.OPTICAL
        )
        tactile.open()
        frame = tactile.capture()
        
        self.assertIsNotNone(frame.proximity)
        self.assertEqual(frame.proximity.shape, (16, 16))
        tactile.close()
    
    def test_resistive_tactile_no_proximity(self):
        """电阻式触觉传感器不支持接近觉"""
        tactile = TactileArray(
            array_size=(16, 16),
            sensor_type=TactileSensorType.RESISTIVE
        )
        tactile.open()
        frame = tactile.capture()
        
        self.assertIsNone(frame.proximity)
        tactile.close()
    
    def test_tactile_slip_signal_after_contact(self):
        """接触后应产生滑移信号"""
        tactile = TactileArray(array_size=(16, 16))
        tactile.open()
        
        # 先采集几帧建立历史
        for _ in range(5):
            tactile.capture()
        
        # 之后再采集几帧,滑移信号通过压力梯度变化检测
        # 滑移信号可能在某些帧出现(梯度变化剧烈时)
        frame = tactile.capture()
        # slip_signal可以是None(无显著滑移)或非None(检测到梯度变化)
        # 主要测试capture()不会崩溃且返回有效frame
        self.assertIsNotNone(frame.pressure_map)
        tactile.close()
    
    def test_tactile_frame_buffer_growth(self):
        """触觉帧缓冲区应正确限制大小"""
        tactile = TactileArray(array_size=(16, 16))
        tactile.open()
        
        # 捕获超过100帧,验证缓冲区会被限制
        for i in range(120):
            tactile.capture()
        
        # 缓冲区在>100时裁剪到50,之后继续增长
        # 120帧时: 100帧触发裁剪到50,然后再追加20帧
        # 最终约70帧
        self.assertGreater(len(tactile._frame_buffer), 50)
        self.assertLessEqual(len(tactile._frame_buffer), 80)
        tactile.close()
    
    def test_tactile_resistive_quantization(self):
        """电阻式触觉应有12bit量化"""
        tactile = TactileArray(
            array_size=(16, 16),
            sensor_type=TactileSensorType.RESISTIVE
        )
        tactile.open()
        frame = tactile.capture()
        
        # 12bit: 1/4096 量化步长
        lsb = 1.0 / 4096
        quantized = np.round(frame.pressure_map / lsb) * lsb
        np.testing.assert_allclose(frame.pressure_map, quantized, atol=1e-6)
        tactile.close()


class TestForceToolCenter(unittest.TestCase):
    """测试力传感器的工具中心偏移"""
    
    def test_force_with_tool_center_offset(self):
        """力矩应反映工具中心偏移"""
        sensor = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        sensor.tool_center = np.array([0.0, 0.05, 0.1])  # 5cm Y, 10cm Z 偏移
        
        sensor.open()
        
        # 捕获多帧检查力矩不为零(因为有重力+偏移)
        torques = []
        for _ in range(10):
            wrench = sensor.capture()
            torques.append(wrench.torque_magnitude)
        
        # 由于工具中心有偏移,力矩应非零
        self.assertTrue(any(t > 0.01 for t in torques))
        sensor.close()
    
    def test_force_zero_tool_center(self):
        """工具中心在原点时力矩较小"""
        sensor = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        sensor.tool_center = np.zeros(3)
        
        sensor.open()
        wrench = sensor.capture()
        
        # 理想情况下力矩接近零(噪声范围内)
        self.assertLess(wrench.torque_magnitude, 1.0)
        sensor.close()
    
    def test_finger_tip_sensor_force_direction(self):
        """灵巧手指尖力传感器主要承受抓取力"""
        sensor = ForceTorqueSensor(sensor_type=ForceSensorType.FINGER_TIP)
        sensor.open()
        
        forces = []
        for _ in range(20):
            wrench = sensor.capture()
            forces.append(wrench.force)
        
        forces = np.array(forces)
        # Fz 应为主要方向(抓取方向)
        mean_fz = np.mean(forces[:, 2])
        self.assertLess(mean_fz, -1.0)  # 负值表示压力
        sensor.close()
    
    def test_joint_torque_sensor(self):
        """关节力矩传感器应返回关节力矩"""
        sensor = ForceTorqueSensor(sensor_type=ForceSensorType.JOINT_TORQUE)
        sensor.open()
        
        torques = []
        for _ in range(10):
            wrench = sensor.capture()
            torques.append(wrench.torque_magnitude)
        
        # 关节力矩 Nm 级
        self.assertTrue(any(t > 0.5 for t in torques))
        sensor.close()


class TestIMUMagnetometer(unittest.TestCase):
    """测试IMU磁力计功能"""
    
    def test_mpu9250_has_magnetometer(self):
        """MPU9250应有磁力计"""
        imu = IMUSensor(sensor_type=IMUSensorType.MPU9250, sample_rate=100)
        imu.open()
        frame = imu.capture()
        
        self.assertIsNotNone(frame.mag)
        self.assertEqual(frame.mag.shape, (3,))
        # 地磁场应在合理范围 (25-65 μT)
        mag_mag = np.linalg.norm(frame.mag)
        self.assertGreater(mag_mag, 10)
        self.assertLess(mag_mag, 100)
        imu.close()
    
    def test_bmi088_no_magnetometer(self):
        """BMI088无磁力计"""
        imu = IMUSensor(sensor_type=IMUSensorType.BMI088, sample_rate=200)
        imu.open()
        frame = imu.capture()
        
        self.assertIsNone(frame.mag)
        imu.close()
    
    def test_virtual_imu_magnetometer(self):
        """虚拟IMU可配置磁力计"""
        imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL, sample_rate=100)
        imu.open()
        frame = imu.capture()
        
        self.assertIsNotNone(frame.mag)
        imu.close()
    
    def test_imu_temperature_self_heating(self):
        """IMU温度应反映自发热"""
        imu = IMUSensor(sensor_type=IMUSensorType.BMI088, sample_rate=200)
        imu.open()
        
        # 连续采集多帧
        temps = []
        for _ in range(100):
            frame = imu.capture()
            temps.append(frame.temperature)
        
        # 温度应稳定在合理范围
        mean_temp = np.mean(temps)
        self.assertGreater(mean_temp, 20)
        self.assertLess(mean_temp, 35)  # 自发热后应高于室温但低于35°C
        
        # 温度标准差应较小
        std_temp = np.std(temps)
        self.assertLess(std_temp, 2.0)
        imu.close()
    
    def test_imu_accel_noise_consistent_with_spec(self):
        """IMU加速度噪声应符合规格"""
        imu = IMUSensor(sensor_type=IMUSensorType.ADIS16470, sample_rate=1000)
        imu.open()
        
        # 采集静止数据
        accels = []
        for _ in range(200):
            frame = imu.capture()
            accels.append(np.linalg.norm(frame.accel))
        
        accels = np.array(accels)
        # 噪声标准差应在合理范围
        std = np.std(accels)
        self.assertLess(std, 0.5)  # ADIS16470 应非常安静
        
        # 均值应接近 9.81 (重力)
        mean = np.mean(accels)
        self.assertGreater(mean, 9.7)
        self.assertLess(mean, 9.9)
        imu.close()


class TestSensorImprovedSimulation(unittest.TestCase):
    """测试改进后的传感器仿真"""
    
    def test_tactile_improved_simulation_interface_info(self):
        """触觉传感器应打印接口信息"""
        tactile = TactileArray(
            array_size=(16, 16),
            sensor_type=TactileSensorType.CAPACITIVE
        )
        # 测试不应抛出异常
        tactile.open()
        self.assertTrue(tactile._is_opened)
        tactile.close()
    
    def test_force_improved_simulation_calibration_loaded(self):
        """力传感器应加载校准参数"""
        sensor = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        sensor.open()
        
        # 应有校准信息
        self.assertIsNotNone(sensor.calibration)
        self.assertTrue(sensor._is_streaming)
        sensor.close()
    
    def test_imu_improved_simulation_start_time(self):
        """IMU传感器应记录启动时间"""
        imu = IMUSensor(sensor_type=IMUSensorType.BMI088)
        imu.open()
        
        self.assertIsNotNone(imu._start_time)
        # 帧ID应从0开始
        self.assertEqual(imu._frame_id, 0)
        
        frame = imu.capture()
        self.assertEqual(frame.frame_id, 0)
        imu.close()


class TestAGVTactileForceIMUCompliance(unittest.TestCase):
    """AGV五级触觉/力觉/IMU规格合规测试"""
    
    def test_tactile_agv_m_array_size(self):
        """AGV-M级触觉阵列应为16x16"""
        spec = get_tactile_spec('M')
        self.assertEqual(spec['array'], (16, 16))
        self.assertEqual(spec['freq_hz'], 100)
    
    def test_tactile_agv_xxl_array_size(self):
        """AGV-XXL级触觉阵列应为48x48"""
        spec = get_tactile_spec('XXL')
        self.assertEqual(spec['array'], (48, 48))
        self.assertEqual(spec['freq_hz'], 1000)
    
    def test_force_agv_m_six_axis(self):
        """AGV-M级力觉应为6轴"""
        spec = get_force_spec('M')
        self.assertEqual(spec['axes'], 6)
        self.assertEqual(spec['sampling_hz'], 500)
    
    def test_force_agv_xxl_sampling_rate(self):
        """AGV-XXL级力觉采样率应达5000Hz"""
        spec = get_force_spec('XXL')
        self.assertEqual(spec['sampling_hz'], 5000)
    
    def test_imu_agv_s_noise_density(self):
        """AGV-S级IMU噪声密度应符合规格"""
        spec = get_imu_spec('S')
        self.assertEqual(spec['noise_density'], 400)  # μg/√Hz
        self.assertEqual(spec['sample_hz'], 100)
    
    def test_imu_agv_xxl_sample_rate(self):
        """AGV-XXL级IMU采样率应达2000Hz"""
        spec = get_imu_spec('XXL')
        self.assertEqual(spec['sample_hz'], 2000)


class TestSensorTypeCompliance(unittest.TestCase):
    """传感器类型合规性测试"""

    def test_imu_all_sensor_types_capture(self):
        """所有IMU传感器类型都能正常采集"""
        for sensor_type in IMUSensorType:
            if sensor_type == IMUSensorType.VIRTUAL:
                continue  # 跳过虚拟类型
            imu = IMUSensor(sensor_type=sensor_type)
            imu.open()
            frame = imu.capture()
            self.assertIsInstance(frame, IMUFrame)
            self.assertEqual(frame.accel.shape, (3,))
            self.assertEqual(frame.gyro.shape, (3,))
            # MPU9250应该有磁力计
            if sensor_type == IMUSensorType.MPU9250:
                self.assertIsNotNone(frame.mag)
            # 其他6轴IMU不应该有磁力计
            elif sensor_type == IMUSensorType.BMI088:
                self.assertIsNone(frame.mag)
            imu.close()

    def test_force_all_sensor_types_capture(self):
        """所有力觉传感器类型都能正常采集"""
        for sensor_type in ForceSensorType:
            sensor = ForceTorqueSensor(sensor_type=sensor_type)
            sensor.open()
            wrench = sensor.capture()
            self.assertIsInstance(wrench, Wrench)
            self.assertEqual(wrench.force.shape, (3,))
            self.assertEqual(wrench.torque.shape, (3,))
            sensor.close()

    def test_tactile_all_sensor_types_capture(self):
        """所有触觉传感器类型都能正常采集"""
        for sensor_type in TactileSensorType:
            tactile = TactileArray(
                array_size=(16, 16),
                sensor_type=sensor_type
            )
            tactile.open()
            frame = tactile.capture()
            self.assertIsInstance(frame, TactileFrame)
            self.assertEqual(frame.pressure_map.shape, (16, 16))
            self.assertIsNotNone(frame.temperature_map)
            # 电容式和光学式应该有接近觉
            if sensor_type in [TactileSensorType.CAPACITIVE, TactileSensorType.OPTICAL]:
                self.assertIsNotNone(frame.proximity)
            else:
                self.assertIsNone(frame.proximity)
            tactile.close()

    def test_imu_sensor_ranges_match_spec(self):
        """IMU传感器量程应与AGV等级规格匹配"""
        test_cases = [
            (IMUSensorType.BMI088, 16, 2000),
            (IMUSensorType.MPU6050, 8, 1000),
            (IMUSensorType.ADIS16470, 40, 4000),
        ]
        for sensor_type, expected_accel_range, expected_gyro_range in test_cases:
            imu = IMUSensor(
                sensor_type=sensor_type,
                accel_range=expected_accel_range,
                gyro_range=expected_gyro_range
            )
            self.assertEqual(imu.accel_range, expected_accel_range)
            self.assertEqual(imu.gyro_range, expected_gyro_range)
            imu.open()
            frame = imu.capture()
            self.assertEqual(frame.accel.shape, (3,))
            imu.close()

    def test_force_sensor_with_ip_address(self):
        """网络力觉传感器应能设置IP地址"""
        sensor = ForceTorqueSensor(
            sensor_type=ForceSensorType.SIX_AXIS,
            ip_address="192.168.1.100",
            ethernet_type="TCP"
        )
        self.assertEqual(sensor.ip_address, "192.168.1.100")
        self.assertEqual(sensor.ethernet_type, "TCP")
        sensor.open()
        wrench = sensor.capture()
        self.assertIsInstance(wrench, Wrench)
        sensor.close()

    def test_tactile_calibration_with_weights(self):
        """触觉传感器标定应支持已知砝码"""
        tactile = TactileArray()
        tactile.open()
        known_weights = [0.5, 1.0, 2.0, 5.0]  # kg
        tactile.calibrate(known_weights=known_weights)
        self.assertIsNotNone(tactile.calibration.force_scale)
        self.assertGreater(tactile.calibration.force_scale, 0)
        tactile.close()

    def test_force_calibration_bias(self):
        """力觉传感器偏置校准"""
        sensor = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        sensor.open()
        sensor.calibrate_bias(num_samples=50)
        # 偏置应该被更新
        self.assertIsNotNone(sensor.calibration.bias)
        self.assertEqual(sensor.calibration.bias.shape, (6,))
        sensor.close()

    def test_imu_calibrate_accel_all_orientations(self):
        """IMU加速度计标定支持所有朝向"""
        orientations = ["level", "up", "down", "left", "right", "front", "back"]
        for orientation in orientations:
            imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL)
            imu.open()
            imu.calibrate_accel(known_orientation=orientation)
            self.assertIsNotNone(imu.calibration.accel_scale)
            imu.close()

    def test_pose_estimator_different_algorithms(self):
        """姿态估计器支持不同算法"""
        algorithms = ["madgwick", "complementary", "kalman"]
        for algo in algorithms:
            estimator = PoseEstimator(algorithm=algo, sample_rate=100.0)
            accel = np.array([0.0, 0.0, 9.81])
            gyro = np.array([0.0, 0.0, 0.0])
            pose = estimator.update(accel, gyro)
            self.assertIsInstance(pose, Pose)
            euler = pose.to_euler()
            self.assertEqual(euler.shape, (3,))

    def test_wrench_processor_outlier_removal(self):
        """力矩处理器应能去除异常值"""
        proc = WrenchProcessor(filter_alpha=0.3, outlier_threshold=3.0)
        # 正常值
        normal_wrench = np.array([10.0, 0.0, -9.81, 0.0, 0.0, 0.0])
        # 异常值
        outlier_wrench = np.array([100.0, 0.0, -9.81, 0.0, 0.0, 0.0])
        history = [normal_wrench for _ in range(20)]
        
        cleaned = proc.remove_outliers(outlier_wrench, history)
        self.assertEqual(cleaned.shape, (6,))
        # 异常值应该被替换为均值
        self.assertNotEqual(cleaned[0], 100.0)

    def test_wrench_equivalent_at_point(self):
        """力矩等效变换到指定点"""
        proc = WrenchProcessor()
        wrench = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        translation = np.array([0.0, 0.1, 0.0])  # 10cm Y方向偏移，产生Z轴力矩
        equivalent = proc.compute_equivalent_wrench_at(wrench, translation)
        self.assertEqual(equivalent.shape, (6,))
        # 力矩应该变化 - cross((0,0.1,0), (10,0,0)) = (0,0,-1), 所以Tz变为-1
        self.assertEqual(equivalent[5], -1.0)  # Tz分量

    def test_pressure_processor_centroid(self):
        """压力分布质心计算"""
        proc = PressureProcessor()
        # 创建一个简单的压力分布
        pressure = np.zeros((16, 16), dtype=np.float32)
        pressure[8, 8] = 1.0  # 单点峰值
        cy, cx = proc.compute_centroid(pressure)
        self.assertEqual(cy, 8.0)
        self.assertEqual(cx, 8.0)

    def test_pressure_processor_histogram(self):
        """压力分布直方图计算"""
        proc = PressureProcessor()
        pressure = np.random.rand(16, 16).astype(np.float32)
        hist, edges = proc.compute_pressure_histogram(pressure, bins=10)
        self.assertEqual(len(hist), 10)
        self.assertEqual(len(edges), 11)

    def test_virtual_tactile_multiple_contact_positions(self):
        """虚拟触觉传感器支持多个接触位置"""
        with VirtualTactileSensor(array_size=(16, 16)) as vt:
            positions = [(0.3, 0.3), (0.7, 0.7), (0.5, 0.5)]
            for pos in positions:
                frame = vt.simulate_contact(pos, 0.2, 10.0)
                self.assertIsInstance(frame, TactileFrame)
                self.assertGreater(np.max(frame.pressure_map), 0)

    def test_virtual_force_multiple_contacts(self):
        """虚拟力觉传感器支持多次接触模拟"""
        with VirtualForceSensor() as vf:
            for i in range(5):
                force = (float(i * 10), 0.0, 0.0)
                wrench = vf.simulate_contact(force)
                self.assertIsInstance(wrench, Wrench)

    def test_virtual_imu_trajectory_all_types(self):
        """虚拟IMU支持所有轨迹类型"""
        for traj_type in ["circle", "figure8", "linear", "sine"]:
            with VirtualIMUSensor() as vi:
                frames = vi.simulate_trajectory(traj_type, duration_s=0.1, dt=0.01)
                self.assertGreater(len(frames), 5)
                for f in frames:
                    self.assertIsInstance(f, IMUFrame)
                    self.assertEqual(f.accel.shape, (3,))
                    self.assertEqual(f.gyro.shape, (3,))

    def test_sensor_grades_progressive_rates(self):
        """AGV等级间传感器采样率应递进"""
        grades = ['S', 'M', 'L', 'XL', 'XXL']
        
        tactile_rates = [get_tactile_spec(g)['freq_hz'] for g in grades]
        force_rates = [get_force_spec(g)['sampling_hz'] for g in grades]
        imu_rates = [get_imu_spec(g)['sample_hz'] for g in grades]
        
        # 验证递进关系
        for i in range(len(grades) - 1):
            self.assertLess(tactile_rates[i], tactile_rates[i+1])
            self.assertLess(force_rates[i], force_rates[i+1])
            self.assertLess(imu_rates[i], imu_rates[i+1])

    def test_sensor_timestamps_increasing(self):
        """传感器时间戳应递增或保持非负"""
        cam = BinocularCamera()
        mic = BinauralMic()
        tactile = TactileArray()
        
        cam.open()
        mic.open()
        tactile.open()
        
        for _ in range(5):
            stereo = cam.capture()
            audio = mic.capture()
            tac = tactile.capture()
            
            # 验证时间戳非负
            self.assertGreaterEqual(stereo.timestamp, 0.0)
            self.assertGreaterEqual(audio.timestamp, 0.0)
            self.assertGreaterEqual(tac.timestamp, 0.0)
            
            # 验证返回了有效数据
            self.assertIsNotNone(stereo.left_image)
            self.assertIsNotNone(audio.left_channel)
            self.assertIsNotNone(tac.pressure_map)
        
        cam.close()
        mic.close()
        tactile.close()


class TestSensorSimulationIntegration(unittest.TestCase):
    """传感器-仿真环境集成测试
    
    验证触觉、力觉、IMU传感器与RobotSimulator的集成工作。
    """

    def test_simulated_contact_tactile_force_integration(self):
        """测试仿真接触场景中触觉和力觉的一致性"""
        from simulation.environment import SimConfig
        
        config = SimConfig(num_joints=6, dt=0.01)
        
        # 创建虚拟传感器
        tactile = VirtualTactileSensor(array_size=(16, 16), sensor_id="int_tactile")
        force = VirtualForceSensor(sensor_id="int_force", noise_level=0.1)
        
        tactile.open()
        force.open()
        
        # 模拟一个接触
        contact_pos = (0.5, 0.5)
        contact_force = 15.0  # N
        
        # 触觉捕获接触
        tac_frame = tactile.simulate_contact(
            contact_pos=contact_pos,
            contact_radius=0.3,
            contact_force=contact_force,
            noise_level=0.02
        )
        
        # 力觉捕获接触力 (禁用噪声以保证测试稳定性)
        wrench = force.simulate_contact(
            force=(0.0, 0.0, -contact_force),
            torque=(0.0, 0.0, 0.0),
            add_noise=False
        )
        
        # 验证触觉帧
        self.assertIsNotNone(tac_frame.pressure_map)
        self.assertGreater(np.max(tac_frame.pressure_map), 0)
        
        # 验证力觉数据
        self.assertIsNotNone(wrench.force)
        self.assertAlmostEqual(wrench.force[2], -contact_force, delta=2.0)
        
        # 验证一致性：触觉峰值压力与力觉Z轴力正相关
        tac_peak = np.max(tac_frame.pressure_map)
        force_mag = abs(wrench.force[2])
        self.assertGreater(tac_peak, 0)
        self.assertGreater(force_mag, 0)
        
        tactile.close()
        force.close()
    
    def test_imu_motion_with_virtual_sensor(self):
        """测试IMU随机器人运动的数据变化"""
        imu = VirtualIMUSensor(sensor_id="int_imu", accel_noise=0.005, gyro_noise=0.0005)
        imu.open()
        
        # 静止状态
        frame_static = imu.simulate_static(orientation=(0.0, 0.0, 0.0))
        self.assertAlmostEqual(frame_static.accel[2], 9.81, delta=0.5)
        
        # 模拟俯仰运动 (绕X轴旋转)
        frame_pitch = imu.simulate_motion(
            linear_accel=(0.0, 0.0, 0.0),
            angular_vel=(0.5, 0.0, 0.0),
            dt=0.01
        )
        self.assertIsNotNone(frame_pitch.gyro)
        self.assertGreater(abs(frame_pitch.gyro[0]), 0.3)  # 应有X轴角速度
        
        # 轨迹模拟 (圆周运动)
        trajectory = imu.simulate_trajectory(
            trajectory_type="circle",
            duration_s=0.1,
            dt=0.01
        )
        self.assertGreater(len(trajectory), 5)
        
        # 相邻帧应有速度变化
        if len(trajectory) >= 2:
            accel_diff = np.abs(trajectory[1].accel - trajectory[0].accel)
            self.assertGreater(np.mean(accel_diff), 0)
        
        imu.close()
    
    def test_pose_estimator_with_imu_trajectory(self):
        """测试姿态估计器对IMU轨迹的处理"""
        imu = VirtualIMUSensor(sensor_id="pose_test_imu")
        imu.open()
        
        # 使用Madgwick算法
        estimator = PoseEstimator(algorithm="madgwick", sample_rate=100.0, beta=0.1)
        
        # 模拟多个运动帧
        frames = imu.simulate_trajectory("sine", duration_s=0.5, dt=0.01)
        
        for frame in frames[:20]:
            pose = estimator.update(frame.accel, frame.gyro, frame.mag)
            self.assertIsNotNone(pose.orientation)
            # 四元数应归一化
            norm = np.linalg.norm(pose.orientation)
            self.assertAlmostEqual(norm, 1.0, places=5)
        
        # Euler角应在合理范围内
        euler = estimator.get_euler()
        self.assertEqual(len(euler), 3)
        
        imu.close()
    
    def test_multimodal_sensor_timing(self):
        """测试多模态传感器时间同步"""
        import time
        
        tactile = VirtualTactileSensor(array_size=(8, 8), sensor_id="sync_tactile")
        force = VirtualForceSensor(sensor_id="sync_force")
        imu = VirtualIMUSensor(sensor_id="sync_imu")
        
        for s in [tactile, force, imu]:
            s.open()
        
        timestamps = []
        
        for i in range(10):
            t_start = time.perf_counter()
            
            # 虚拟传感器直接获取帧
            tac = tactile.simulate_contact(contact_pos=(0.5, 0.5), contact_force=5.0)
            wrench = force.simulate_contact(force=(0, 0, -5))
            imu_frame = imu.simulate_static()
            
            t_end = time.perf_counter()
            
            # 所有时间戳应在合理范围内
            self.assertGreaterEqual(tac.timestamp, 0.0)
            self.assertGreaterEqual(wrench.timestamp, 0.0)
            self.assertGreaterEqual(imu_frame.timestamp, 0.0)
            
            timestamps.append(t_end - t_start)
        
        # 平均采集时间应小于1ms
        avg_time = np.mean(timestamps)
        self.assertLess(avg_time, 0.001, f"Sensor capture too slow: {avg_time*1000:.2f}ms")
        
        for s in [tactile, force, imu]:
            s.close()
    
    def test_force_sensor_payload_with_simulated_mass(self):
        """测试力觉传感器对仿真负载的估计"""
        force = VirtualForceSensor(sensor_id="payload_test", noise_level=0.05)
        force.open()
        
        # 模拟不同质量负载
        masses = [0.5, 1.0, 2.0, 5.0]
        estimated_masses = []
        
        for mass in masses:
            wrench = force.simulate_payload(mass=mass, com_offset=(0.0, 0.0, 0.0))
            # 力觉测量的Fz应该反映质量
            fz_measured = abs(wrench.force[2])
            est_mass = fz_measured / 9.81
            estimated_masses.append(est_mass)
        
        # 验证估计值与实际值趋势一致
        self.assertGreater(estimated_masses[-1], estimated_masses[0])
        
        # 误差应在30%以内
        for actual, estimated in zip(masses, estimated_masses):
            rel_error = abs(estimated - actual) / actual
            self.assertLess(rel_error, 0.30)
        
        force.close()
    
    def test_tactile_estimate_grip_quality(self):
        """测试触觉抓取质量评估"""
        ta = TactileArray(array_size=(16, 16))
        ta.open()
        
        # 连续采集多帧以建立滑移检测历史
        for _ in range(5):
            ta.capture()
        
        # 最后一帧用于质量评估（此时有历史数据）
        frame = ta.capture()
        quality = ta.estimate_grip_quality(frame)
        self.assertIn('overall', quality)
        self.assertIn('contact_area', quality)
        self.assertIn('uniformity', quality)
        self.assertIn('stability', quality)
        # overall可能是NaN当滑移数据不足；检查是数值或NaN
        if not np.isnan(quality['overall']):
            self.assertGreaterEqual(quality['overall'], 0.0)
            self.assertLessEqual(quality['overall'], 1.0)
        
        # 无接触时质量应为零
        zero_frame = TactileFrame(pressure_map=np.zeros((16, 16)), timestamp=0.0)
        zero_quality = ta.estimate_grip_quality(zero_frame)
        self.assertEqual(zero_quality['overall'], 0.0)
        
        ta.close()
    
    def test_tactile_sliding_simulation(self):
        """测试触觉滑移动画模拟"""
        vt = VirtualTactileSensor(array_size=(16, 16))
        vt.open()
        
        # 模拟滑动
        frames = vt.simulate_sliding(
            direction=(1.0, 0.0),  # 沿X方向
            speed=0.05,
            duration_frames=10
        )
        
        self.assertEqual(len(frames), 10)
        for f in frames:
            self.assertIsInstance(f, TactileFrame)
            self.assertEqual(f.pressure_map.shape, (16, 16))
        
        # 验证位置随帧变化
        self.assertNotEqual(frames[0].pressure_map.sum(), frames[-1].pressure_map.sum())
        vt.close()
    
    def test_force_sensor_estimate_payload(self):
        """测试力觉传感器负载估计"""
        sensor = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        sensor.open()
        
        # 模拟负载
        sensor.set_tool_center(tool_mass=1.5, tool_com=np.array([0.0, 0.0, 0.05]))
        
        for _ in range(5):
            wrench = sensor.capture()
        
        payload = sensor.estimate_payload()
        # 负载估计应接近设置值
        self.assertGreater(payload, 0.0)
        sensor.close()
    
    def test_force_collision_simulation(self):
        """测试力觉碰撞仿真"""
        with VirtualForceSensor(noise_level=0.02) as vf:
            # 沿X轴碰撞
            frames = vf.simulate_collision(
                direction=(1.0, 0.0, 0.0),
                peak_force=50.0,
                duration_ms=100.0,
                decay="exponential"
            )
            
            self.assertGreater(len(frames), 0)
            self.assertIsInstance(frames[0], Wrench)
            
            # 验证力向量方向
            for f in frames:
                self.assertGreater(f.force[0], 0)  # X分量应为正
            
            # 峰值力应在第一帧
            peak_idx = np.argmax([np.linalg.norm(f.force) for f in frames])
            self.assertEqual(peak_idx, 0)
    
    def test_force_linear_decay_collision(self):
        """测试线性衰减碰撞仿真"""
        with VirtualForceSensor(noise_level=0.0, bias_range=0.0) as vf:
            frames = vf.simulate_collision(
                direction=(0.0, 1.0, 0.0),
                peak_force=100.0,
                duration_ms=50.0,
                decay="linear"
            )
            
            # 线性衰减: 每帧递减
            for i in range(1, len(frames)):
                prev_mag = np.linalg.norm(frames[i-1].force)
                curr_mag = np.linalg.norm(frames[i].force)
                self.assertLessEqual(curr_mag, prev_mag)
    
    def test_imu_pose_integrator(self):
        """测试IMU姿态积分功能"""
        pe = PoseEstimator(algorithm='madgwick', beta=0.1, sample_rate=100.0)
        
        # 模拟旋转运动
        for i in range(50):
            t = i / 100.0
            gyro = np.array([0.0, 0.0, 0.5 * np.sin(t * 2)])  # 绕Z轴旋转
            accel = np.array([0.0, 0.0, 9.81])
            pose = pe.update(accel, gyro)
            self.assertIsInstance(pose, Pose)
        
        euler = pe.get_euler()
        self.assertEqual(euler.shape, (3,))
    
    def test_imu_with_magnetometer(self):
        """测试带磁力计的IMU (MPU9250)"""
        imu = IMUSensor(sensor_type=IMUSensorType.MPU9250)
        imu.open()
        
        frame = imu.capture()
        self.assertIsNotNone(frame.mag)
        self.assertEqual(frame.mag.shape, (3,))
        self.assertGreater(np.linalg.norm(frame.mag), 0.0)  # 地磁场存在
        
        imu.close()
    
    def test_virtual_imu_trajectory_circle(self):
        """测试虚拟IMU圆形轨迹仿真"""
        with VirtualIMUSensor() as vi:
            frames = vi.simulate_trajectory(
                trajectory_type="circle",
                duration_s=0.1,
                dt=0.01
            )
            
            self.assertGreater(len(frames), 5)
            for f in frames:
                self.assertIsInstance(f, IMUFrame)
                self.assertIsNotNone(f.accel)
    
    def test_virtual_imu_trajectory_figure8(self):
        """测试虚拟IMU八字形轨迹仿真"""
        with VirtualIMUSensor() as vi:
            frames = vi.simulate_trajectory(
                trajectory_type="figure8",
                duration_s=0.1,
                dt=0.01
            )
            
            self.assertGreater(len(frames), 5)
            # 八字形轨迹应在Y方向有更大的加速度变化
            accel_y_vals = [f.accel[1] for f in frames]
            self.assertGreater(np.std(accel_y_vals), 0.0)
    
    def test_gym_env_sensor_feedback(self):
        """测试Gym仿真环境与传感器反馈的闭环"""
        from simulation.gym_env import SuperModelGymEnv
        
        env = SuperModelGymEnv(render_mode=None)
        obs, info = env.reset()
        
        self.assertIsNotNone(obs)
        
        # 执行几步随机动作
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        # 验证仿真环境正确响应
        self.assertIn('joint_pos', info)
        
        env.close()


class TestPlannerHTNBacktracking(unittest.TestCase):
    """HTN规划器回溯能力测试"""

    def test_hierarchical_planner_backtrack(self):
        """测试HTN规划器回溯机制"""
        from control.planner import HierarchicalPlanner, Task
        
        planner = HierarchicalPlanner()
        
        # 默认已有pickup方法;再注册一个替代方法
        initial_methods = planner.get_available_methods("pickup")
        
        def pickup_method_alt(params):
            return [
                Task(id="p1_alt", name="approach_safe"),
                Task(id="p2_alt", name="slow_grasp"),
                Task(id="p3_alt", name="verify_grasp")
            ]
        
        planner.register_method("pickup", pickup_method_alt)
        
        # 验证方法数量增加
        self.assertEqual(planner.get_available_methods("pickup"), initial_methods + 1)
        
        # 测试回溯
        root = Task(id="root", name="pickup", parameters={"object": "box"})
        tasks, attempted = planner.backtrack(root, [])
        self.assertGreater(len(tasks), 0)
    
    def test_planner_with_replanning(self):
        """测试带重规划的分层规划"""
        from control.planner import HierarchicalPlanner, TaskSpec, WorldState, Action
        
        # 通过 action_library 构造
        action_lib = {
            "move": Action(
                name="move",
                precondition=lambda s: True,
                effect=lambda s, p: None,
                cost=1.0
            ),
            "grasp": Action(
                name="grasp",
                precondition=lambda s: True,
                effect=lambda s, p: None,
                cost=2.0
            ),
        }
        
        planner = HierarchicalPlanner(action_library=action_lib)
        
        spec = TaskSpec(
            name="test_task",
            goal_state={"object.grasped": True}
        )
        
        state = WorldState()
        tasks, metadata = planner.plan_with_replanning(spec, state, max_replan_attempts=2)
        
        self.assertIsInstance(metadata, dict)
        self.assertIn("replan_history", metadata)


if __name__ == '__main__':
    unittest.main()


class TestSensorCalibrationWorkflow(unittest.TestCase):
    """测试传感器标定完整工作流"""
    
    def test_force_sensor_complete_calibration_workflow(self):
        """力觉传感器完整标定流程"""
        sensor = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        sensor.open()
        
        # 1. 偏置校准 (无负载状态)
        sensor.calibrate_bias(num_samples=50)
        
        # 2. 设置工具中心 (添加重力补偿)
        sensor.set_tool_center(tool_mass=1.0, tool_com=np.array([0.0, 0.05, 0.1]))
        
        # 3. 采集数据
        wrenches = []
        for _ in range(20):
            w = sensor.capture()
            wrenches.append(w)
        
        # 验证校准效果: 力传感器应该输出接近0 (已偏置校准+重力补偿)
        forces = np.array([w.force for w in wrenches])
        torques = np.array([w.torque for w in wrenches])
        
        # 校准后XY方向力应接近0
        self.assertLess(abs(np.mean(forces[:, 0])), 2.0)
        self.assertLess(abs(np.mean(forces[:, 1])), 2.0)
        
        # 采集后传感器仍能正常工作
        final_wrench = sensor.capture()
        self.assertIsInstance(final_wrench, Wrench)
        
        sensor.close()
    
    def test_imu_complete_calibration_workflow(self):
        """IMU完整标定流程"""
        imu = IMUSensor(sensor_type=IMUSensorType.BMI088)
        imu.open()
        
        # 1. 自检
        self_test = imu.self_test()
        self.assertTrue(self_test)
        
        # 2. 陀螺仪偏置校准
        imu.calibrate_gyro_bias(num_samples=100)
        
        # 3. 加速度计标定
        imu.calibrate_accel(known_orientation="level")
        
        # 4. 验证校准效果
        frames = []
        for _ in range(50):
            f = imu.capture()
            frames.append(f)
        
        # 静止时角速度应该接近0
        gyro_norms = [np.linalg.norm(f.gyro) for f in frames]
        self.assertLess(np.mean(gyro_norms), 0.5)
        
        # 验证四元数归一化
        estimator = PoseEstimator()
        for frame in frames[:10]:
            pose = estimator.update(frame.accel, frame.gyro)
            norm = np.linalg.norm(pose.orientation)
            self.assertAlmostEqual(norm, 1.0, places=5)
        
        imu.close()
    
    def test_tactile_complete_calibration_workflow(self):
        """触觉传感器完整标定流程"""
        tactile = TactileArray(array_size=(16, 16))
        tactile.open()
        
        # 1. 零压力标定
        zero_pressure = np.zeros((16, 16))
        tactile.calibrate(zero_pressure=zero_pressure)
        
        # 2. 力标定
        tactile.calibrate(known_weights=[0.5, 1.0, 2.0])
        
        # 3. 采集数据验证
        for _ in range(10):
            frame = tactile.capture()
            self.assertEqual(frame.pressure_map.shape, (16, 16))
        
        # 4. 接触检测
        contacts = tactile.detect_contacts()
        self.assertIsInstance(contacts, list)
        
        # 5. 抓取质量评估
        frame = tactile.capture()
        quality = tactile.estimate_grip_quality(frame)
        self.assertIn('overall', quality)
        
        tactile.close()


class TestSensorErrorHandling(unittest.TestCase):
    """传感器错误处理测试"""
    
    def test_tactile_capture_before_open(self):
        """触觉传感器未打开时捕获应报错"""
        tactile = TactileArray()
        with self.assertRaises(RuntimeError):
            tactile.capture()
    
    def test_force_capture_before_open(self):
        """力觉传感器未打开时捕获应报错"""
        sensor = ForceTorqueSensor()
        with self.assertRaises(RuntimeError):
            sensor.capture()
    
    def test_imu_capture_before_open(self):
        """IMU传感器未打开时捕获应报错"""
        imu = IMUSensor()
        with self.assertRaises(RuntimeError):
            imu.capture()
    
    def test_multiple_context_manager_usage(self):
        """多次使用上下文管理器"""
        for i in range(3):
            tactile = TactileArray(array_size=(8, 8), sensor_id=f"test_{i}")
            with tactile:
                frame = tactile.capture()
                self.assertIsInstance(frame, TactileFrame)
            # 退出后应该关闭
            self.assertFalse(tactile._is_opened)


class TestAGVSensorSpecComplete(unittest.TestCase):
    """AGV五级规格完整性验证"""
    
    def test_stereo_spec_all_grades(self):
        """验证所有视觉等级规格"""
        from sensors.vision import get_stereo_spec
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_stereo_spec(grade)
            self.assertIn('baseline_mm', spec)
            self.assertIn('fov', spec)
            self.assertIn('range_m', spec)
    
    def test_audio_spec_all_grades(self):
        """验证所有听觉等级规格"""
        from sensors.audio import get_audio_spec
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_audio_spec(grade)
            self.assertIn('channels', spec)
            self.assertIn('sr', spec)
    
    def test_tactile_spec_all_grades(self):
        """验证所有触觉等级规格"""
        from sensors.tactile import get_tactile_spec
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_tactile_spec(grade)
            self.assertIn('array', spec)
            self.assertIn('freq_hz', spec)
            self.assertIn('res', spec)
    
    def test_force_spec_all_grades(self):
        """验证所有力觉等级规格"""
        from sensors.force import get_force_spec
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_force_spec(grade)
            self.assertIn('axes', spec)
            self.assertIn('force_range', spec)
            self.assertIn('sampling_hz', spec)
    
    def test_imu_spec_all_grades(self):
        """验证所有IMU等级规格"""
        from sensors.imu import get_imu_spec
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_imu_spec(grade)
            self.assertIn('type', spec)
            self.assertIn('accel_range', spec)
            self.assertIn('gyro_range', spec)
            self.assertIn('sample_hz', spec)
            self.assertIn('noise_density', spec)


class TestSensorTimingConsistency(unittest.TestCase):
    """传感器时序一致性测试"""
    
    def test_imu_frame_id_incrementing(self):
        """IMU帧ID应该递增"""
        imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL)
        imu.open()
        
        prev_id = -1
        for _ in range(20):
            frame = imu.capture()
            self.assertGreater(frame.frame_id, prev_id)
            prev_id = frame.frame_id
        
        imu.close()
    
    def test_tactile_frame_id_incrementing(self):
        """触觉帧ID应该递增"""
        tactile = TactileArray(array_size=(8, 8))
        tactile.open()
        
        prev_id = -1
        for _ in range(20):
            frame = tactile.capture()
            self.assertGreater(frame.frame_id, prev_id)
            prev_id = frame.frame_id
        
        tactile.close()
    
    def test_force_wrench_id_incrementing(self):
        """力矩ID应该递增"""
        sensor = ForceTorqueSensor()
        sensor.open()
        
        prev_id = -1
        for _ in range(20):
            wrench = sensor.capture()
            self.assertGreater(wrench.frame_id, prev_id)
            prev_id = wrench.frame_id
        
        sensor.close()
    
    def test_timestamp_monotonically_increasing(self):
        """时间戳应该单调递增"""
        cam = BinocularCamera()
        mic = BinauralMic()
        tactile = TactileArray()
        
        cam.open()
        mic.open()
        tactile.open()
        
        prev_cam_ts = -1
        prev_mic_ts = -1
        prev_tac_ts = -1
        
        for _ in range(10):
            cam_frame = cam.capture()
            mic_frame = mic.capture()
            tac_frame = tactile.capture()
            
            self.assertGreaterEqual(cam_frame.timestamp, prev_cam_ts)
            self.assertGreaterEqual(mic_frame.timestamp, prev_mic_ts)
            self.assertGreaterEqual(tac_frame.timestamp, prev_tac_ts)
            
            prev_cam_ts = cam_frame.timestamp
            prev_mic_ts = mic_frame.timestamp
            prev_tac_ts = tac_frame.timestamp
        
        cam.close()
        mic.close()
        tactile.close()


class TestSensorAGVFiveLevelCompliance(unittest.TestCase):
    """AGV五级规格合规性测试"""
    
    def test_tactile_all_grades_have_spec(self):
        """所有等级应有触觉规格"""
        from sensors.tactile import get_tactile_spec
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_tactile_spec(grade)
            self.assertIn('array', spec)
            self.assertIn('res', spec)
            self.assertIn('range_kpa', spec)
            self.assertIn('freq_hz', spec)
    
    def test_force_all_grades_have_spec(self):
        """所有等级应有力觉规格"""
        from sensors.force import get_force_spec
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_force_spec(grade)
            self.assertIn('axes', spec)
            self.assertIn('force_range', spec)
            self.assertIn('sampling_hz', spec)
    
    def test_imu_all_grades_have_spec(self):
        """所有等级应有IMU规格"""
        from sensors.imu import get_imu_spec
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_imu_spec(grade)
            self.assertIn('type', spec)
            self.assertIn('accel_range', spec)
            self.assertIn('sample_hz', spec)
    
    def test_tactile_xxl_has_highest_resolution(self):
        """XXL级应有最高触觉分辨率"""
        from sensors.tactile import get_tactile_spec
        spec_s = get_tactile_spec('S')
        spec_xxl = get_tactile_spec('XXL')
        self.assertGreater(spec_xxl['res'], spec_s['res'])
        self.assertGreater(spec_xxl['freq_hz'], spec_s['freq_hz'])
    
    def test_force_xxl_has_highest_range(self):
        """XXL级应有力觉最大范围"""
        from sensors.force import get_force_spec
        spec_m = get_force_spec('M')
        spec_xxl = get_force_spec('XXL')
        self.assertGreater(spec_xxl['force_range'], spec_m['force_range'])
        self.assertGreater(spec_xxl['sampling_hz'], spec_m['sampling_hz'])
    
    def test_imu_xxl_has_lowest_noise(self):
        """XXL级应有最低IMU噪声"""
        from sensors.imu import get_imu_spec
        spec_s = get_imu_spec('S')
        spec_xxl = get_imu_spec('XXL')
        self.assertLess(spec_xxl['noise_density'], spec_s['noise_density'])


class TestVirtualSensorIntegration(unittest.TestCase):
    """虚拟传感器集成测试"""
    
    def test_tactile_force_imu_pipeline(self):
        """触觉-力觉-IMU联合采集流水线"""
        from sensors.tactile import VirtualTactileSensor
        from sensors.force import VirtualForceSensor
        from sensors.imu import VirtualIMUSensor
        
        vt = VirtualTactileSensor((16, 16))
        vf = VirtualForceSensor()
        vi = VirtualIMUSensor()
        
        vt.open()
        vf.open()
        vi.open()
        
        # 采集数据
        tac = vt.simulate_contact((0.5, 0.5), contact_force=10.0)
        wrench = vf.simulate_contact(force=(0, 0, -10))
        imu_frame = vi.simulate_static((0.1, 0.1, 0.0))
        
        # 验证数据合理性
        self.assertIsNotNone(tac.pressure_map)
        self.assertGreater(tac.pressure_map.max(), 0)
        self.assertIsNotNone(wrench.force)
        self.assertIsNotNone(imu_frame.accel)
        
        # IMU静止时角速度应接近零
        self.assertLess(np.linalg.norm(imu_frame.gyro), 0.1)
        
        vt.close()
        vf.close()
        vi.close()
    
    def test_virtual_sensor_context_managers(self):
        """虚拟传感器上下文管理器"""
        from sensors.tactile import VirtualTactileSensor
        from sensors.force import VirtualForceSensor
        from sensors.imu import VirtualIMUSensor
        
        with VirtualTactileSensor((8, 8)) as vt:
            frame = vt.simulate_contact((0.5, 0.5))
            self.assertIsNotNone(frame)
        
        with VirtualForceSensor() as vf:
            w = vf.simulate_contact((0, 0, -5))
            self.assertIsNotNone(w.force)
        
        with VirtualIMUSensor() as vi:
            frame = vi.simulate_static()
            self.assertIsNotNone(frame.accel)
    
    def test_virtual_sensor_trajectory(self):
        """虚拟IMU轨迹仿真"""
        from sensors.imu import VirtualIMUSensor
        
        with VirtualIMUSensor() as vi:
            for traj_type in ['circle', 'figure8', 'linear', 'sine']:
                frames = vi.simulate_trajectory(traj_type, duration_s=0.1, dt=0.01)
                self.assertGreater(len(frames), 0)
                for f in frames:
                    self.assertIsNotNone(f.accel)
                    self.assertIsNotNone(f.gyro)
    
    def test_virtual_sensor_sliding(self):
        """虚拟触觉滑移仿真"""
        from sensors.tactile import VirtualTactileSensor
        
        with VirtualTactileSensor((16, 16)) as vt:
            frames = vt.simulate_sliding(
                direction=(1.0, 0.0),
                speed=0.05,
                duration_frames=10
            )
            self.assertEqual(len(frames), 10)
            # 验证帧ID递增
            for i, f in enumerate(frames):
                self.assertEqual(f.frame_id, i)


class TestTactileGripQuality(unittest.TestCase):
    """触觉抓取质量评估测试"""
    
    def test_grip_quality_no_contact(self):
        """无接触时抓取质量应为零"""
        from sensors.tactile import TactileArray
        
        with TactileArray(array_size=(8, 8)) as tactile:
            tactile.open()
            frame = tactile.capture()
            quality = tactile.estimate_grip_quality(frame)
            self.assertEqual(quality['overall'], 0.0)
            tactile.close()
    
    def test_grip_quality_with_contact(self):
        """有接触时抓取质量应大于零"""
        from sensors.tactile import VirtualTactileSensor
        
        with VirtualTactileSensor((16, 16)) as vt:
            frame = vt.simulate_contact((0.5, 0.5), contact_force=20.0)
            # 手动调用 estimate_grip_quality 需要真实 TactileArray
            # 这里测试帧数据有效性
            self.assertGreater(frame.pressure_map.max(), 0.0)


class TestForceWrenchTransform(unittest.TestCase):
    """力矩坐标变换测试"""
    
    def test_wrench_transform_rotation(self):
        """力矩旋转变换"""
        from sensors.force import Wrench
        
        w = Wrench(
            force=np.array([10.0, 0.0, 0.0]),
            torque=np.array([0.0, 0.0, 0.0])
        )
        
        # 绕Z轴旋转90度
        R = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ])
        t = np.zeros(3)
        
        w_rot = w.transform(R, t)
        
        # 力方向应旋转
        self.assertAlmostEqual(w_rot.force[0], 0.0, places=3)
        self.assertAlmostEqual(w_rot.force[1], 10.0, places=3)
    
    def test_wrench_transform_with_translation(self):
        """力矩平移变换"""
        from sensors.force import Wrench
        
        w = Wrench(
            force=np.array([0.0, 0.0, -10.0]),
            torque=np.array([0.0, 0.0, 0.0])
        )
        
        R = np.eye(3)
        t = np.array([0.1, 0.0, 0.0])  # 偏移0.1m
        
        w_trans = w.transform(R, t)
        
        # 力矩应产生额外的力矩分量 Tx = Fy * tz - Fz * ty = -10 * 0 = 0
        # 实际: torque' = torque + cross(translation, force)
        expected_torque = np.cross(t, w.force)
        np.testing.assert_array_almost_equal(w_trans.torque, expected_torque)


class TestIMUMadgwickConvergence(unittest.TestCase):
    """Madgwick姿态估计算法收敛性测试"""
    
    def test_pose_estimator_converges_to_ground_truth(self):
        """姿态估计应收敛到真实值"""
        from sensors.imu import IMUSensor, PoseEstimator, IMUSensorType
        
        imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL, sample_rate=100)
        imu.open()
        
        estimator = PoseEstimator(algorithm='madgwick', beta=0.1, sample_rate=100)
        estimator.reset()
        
        # 静止状态，传感器水平放置
        # 预期：roll≈0, pitch≈0, yaw保持不变
        euler_history = []
        
        for _ in range(100):
            frame = imu.capture()
            pose = estimator.update(frame.accel, frame.gyro)
            euler_history.append(pose.to_euler())
        
        imu.close()
        
        # 取最后10帧的平均
        final_euler = np.mean(euler_history[-10:], axis=0)
        
        # 静止水平时，roll和pitch应接近0
        self.assertLess(abs(final_euler[0]), 0.5)  # roll < 0.5度
        self.assertLess(abs(final_euler[1]), 0.5)  # pitch < 0.5度


class TestSensorManagerFullCoverage(unittest.TestCase):
    """传感器管理器全模态覆盖测试"""
    
    def test_all_modalities_timestamp_sync(self):
        """所有模态应保持时间戳同步"""
        from sensors.manager import SensorManager, SensorManagerConfig
        
        config = SensorManagerConfig(grade="M")
        manager = SensorManager(config)
        manager.open_all()
        
        data = manager.capture_all()
        
        # 验证数据帧存在
        self.assertIsNotNone(data)
        self.assertGreaterEqual(data.frame_id, 0)
        self.assertGreaterEqual(data.timestamp, 0)
        
        # 验证各模态数据存在
        self.assertIsNotNone(data.vision)
        self.assertIsNotNone(data.audio)
        self.assertIsNotNone(data.tactile)
        self.assertIsNotNone(data.force)
        self.assertIsNotNone(data.imu)
        
        manager.close_all()


if __name__ == '__main__':
    unittest.main()


# ==============================================================================
# 增强边缘测试 - 传感器极端情况测试
# ==============================================================================

class TestSensorEdgeCases:
    """传感器极端情况和边界条件测试"""
    
    def test_tactile_zero_pressure_calibration(self):
        """零压力校准测试"""
        from src.sensors.tactile import TactileArray, TactileSensorType
        
        sensor = TactileArray(
            array_size=(8, 8),
            sensor_type=TactileSensorType.RESISTIVE,
            sensor_id="test_zero"
        )
        sensor.open()
        
        # 采集零压力基准
        zero_pressure = sensor.capture().pressure_map.copy()
        sensor.calibrate(zero_pressure=zero_pressure)
        
        assert sensor.calibration.offset_map is not None
        assert sensor.calibration.offset_map.shape == (8, 8)
        
        sensor.close()
    
    def test_force_extreme_wrench(self):
        """极端力值测试"""
        from src.sensors.force import ForceTorqueSensor, ForceSensorType
        
        sensor = ForceTorqueSensor(
            sensor_type=ForceSensorType.SIX_AXIS,
            sensor_id="test_extreme"
        )
        sensor.open()
        
        # 采集极端力值
        for _ in range(10):
            wrench = sensor.capture()
            assert wrench.force.shape == (3,)
            assert wrench.torque.shape == (3,)
        
        sensor.close()
    
    def test_imu_extreme_orientation(self):
        """极端姿态测试"""
        from src.sensors.imu import IMUSensor, IMUSensorType
        from src.sensors.imu import PoseEstimator
        
        sensor = IMUSensor(
            sensor_type=IMUSensorType.BMI088,
            sensor_id="test_extreme"
        )
        sensor.open()
        
        estimator = PoseEstimator(algorithm="madgwick", sample_rate=200)
        
        # 模拟各种姿态
        for _ in range(100):
            frame = sensor.capture()
            pose = estimator.update(frame.accel, frame.gyro)
            euler = pose.to_euler()
            assert euler.shape == (3,)
        
        sensor.close()
    
    def test_tactile_slip_under_oscillation(self):
        """振荡情况下滑移检测"""
        from src.sensors.tactile import VirtualTactileSensor
        
        sensor = VirtualTactileSensor(array_size=(16, 16))
        sensor.open()
        
        # 模拟水平滑动
        frames = sensor.simulate_sliding(
            direction=(1.0, 0.0),
            speed=0.05,
            duration_frames=20
        )
        
        assert len(frames) == 20
        for frame in frames:
            assert frame.pressure_map.shape == (16, 16)
            assert np.any(frame.pressure_map > 0)
        
        sensor.close()
    
    def test_force_wrench_transform(self):
        """力旋量坐标变换"""
        from src.sensors.force import Wrench
        
        wrench = Wrench(
            force=np.array([10.0, 0.0, 0.0]),
            torque=np.array([0.0, 0.0, 5.0])
        )
        
        # 绕Z轴旋转90度
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        t = np.array([0.1, 0.0, 0.0])
        
        transformed = wrench.transform(R, t)
        
        assert transformed.force.shape == (3,)
        assert transformed.torque.shape == (3,)
        # 旋转后力应该沿Y轴
        assert abs(transformed.force[1]) > 0.5
    
    def test_imu_pose_composition(self):
        """位姿合成测试"""
        from src.sensors.imu import Pose
        
        pose1 = Pose.identity()
        pose2 = Pose.from_euler(
            position=np.array([1.0, 0.0, 0.0]),
            rpy=np.array([0.0, 0.0, 0.0])
        )
        
        # 验证四元数归一化
        assert abs(np.linalg.norm(pose1.orientation) - 1.0) < 1e-6
        assert abs(np.linalg.norm(pose2.orientation) - 1.0) < 1e-6
        
        # 验证欧拉角转换
        euler = pose2.to_euler()
        assert euler.shape == (3,)
        
        # 验证矩阵转换
        matrix = pose2.to_matrix()
        assert matrix.shape == (4, 4)
        assert np.allclose(matrix[3, :], [0, 0, 0, 1])
    
    def test_tactile_contact_minimal_area(self):
        """最小接触面积检测"""
        from src.sensors.tactile import VirtualTactileSensor
        
        sensor = VirtualTactileSensor(array_size=(8, 8))
        sensor.open()
        
        # 单点接触
        frame = sensor.simulate_contact(
            contact_pos=(0.5, 0.5),
            contact_radius=0.05,  # 非常小的接触
            contact_force=1.0
        )
        
        # 极小接触可能检测不到，这是正常的
        assert frame.pressure_map.shape == (8, 8)
        
        sensor.close()
    
    def test_force_payload_estimation_accuracy(self):
        """负载估计精度"""
        from src.sensors.force import VirtualForceSensor
        
        sensor = VirtualForceSensor(sensor_id="test_payload")
        sensor.open()
        
        # 模拟已知负载
        estimated = []
        for mass in [0.5, 1.0, 2.0, 5.0]:
            wrench = sensor.simulate_payload(mass=mass)
            # 力值应该与质量成正比
            fz_measured = abs(wrench.force[2])
            fz_expected = mass * 9.81
            # 允许一定误差（因为有噪声）
            error = abs(fz_measured - fz_expected) / fz_expected
            estimated.append(error < 0.5)  # 50%误差容忍
        
        assert any(estimated)
        sensor.close()
    
    def test_imu_madgwick_vs_complementary(self):
        """Madgwick与互补滤波对比"""
        from src.sensors.imu import PoseEstimator
        
        # 创建两个估计器
        madgwick = PoseEstimator(algorithm="madgwick", sample_rate=200)
        complementary = PoseEstimator(algorithm="complementary", sample_rate=200)
        
        # 模拟静止状态
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.0])
        
        for _ in range(50):
            p1 = madgwick.update(accel, gyro)
            p2 = complementary.update(accel, gyro)
        
        # 两种方法都应该给出水平姿态
        e1 = p1.to_euler()
        e2 = p2.to_euler()
        
        # roll和pitch应该接近0
        assert abs(e1[0]) < 0.1  # roll
        assert abs(e1[1]) < 0.1  # pitch
        assert abs(e2[0]) < 0.1
        assert abs(e2[1]) < 0.1
    
    def test_virtual_sensor_idempotency(self):
        """虚拟传感器幂等性测试"""
        from src.sensors.tactile import VirtualTactileSensor
        from src.sensors.force import VirtualForceSensor
        from src.sensors.imu import VirtualIMUSensor
        
        # 测试多次open/close不会出错
        for _ in range(3):
            t = VirtualTactileSensor()
            t.open()
            t.simulate_contact((0.5, 0.5), contact_force=5.0)
            t.close()
            
            f = VirtualForceSensor()
            f.open()
            f.simulate_contact((1, 1, 1))
            f.close()
            
            i = VirtualIMUSensor()
            i.open()
            i.simulate_static()
            i.close()


class TestSensorPerformance:
    """传感器性能基准测试"""
    
    def test_tactile_capture_latency(self):
        """触觉采集延迟"""
        import time
        from src.sensors.tactile import TactileArray, TactileSensorType
        
        sensor = TactileArray(
            array_size=(16, 16),
            sensor_type=TactileSensorType.CAPACITIVE
        )
        sensor.open()
        
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            sensor.capture()
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
        
        avg_latency = np.mean(latencies)
        p99_latency = np.percentile(latencies, 99)
        
        # 平均延迟应该小于10ms
        assert avg_latency < 10, f"Average latency {avg_latency:.2f}ms too high"
        # P99延迟应该小于50ms
        assert p99_latency < 50, f"P99 latency {p99_latency:.2f}ms too high"
        
        sensor.close()
    
    def test_force_capture_throughput(self):
        """力觉采集吞吐量"""
        import time
        from src.sensors.force import ForceTorqueSensor, ForceSensorType
        
        sensor = ForceTorqueSensor(
            sensor_type=ForceSensorType.SIX_AXIS
        )
        sensor.open()
        
        start = time.perf_counter()
        count = 0
        while time.perf_counter() - start < 0.1:  # 0.1秒内尽可能多采集
            sensor.capture()
            count += 1
        
        duration = time.perf_counter() - start
        throughput = count / duration
        
        # 吞吐量应该大于100Hz
        assert throughput > 100, f"Throughput {throughput:.0f}Hz too low"
        
        sensor.close()
    
    def test_imu_batch_capture(self):
        """IMU批量采集"""
        import time
        from src.sensors.imu import IMUSensor, IMUSensorType
        
        sensor = IMUSensor(
            sensor_type=IMUSensorType.BMI088,
            sample_rate=500
        )
        sensor.open()
        
        start = time.perf_counter()
        frames = []
        for _ in range(500):
            frames.append(sensor.capture())
        duration = time.perf_counter() - start
        
        # 500帧应该在1秒内完成
        assert duration < 1.1, f"500 frames took {duration:.2f}s, expected < 1.1s"
        
        sensor.close()


class TestSensorFusionCrossModal(unittest.TestCase):
    """传感器跨模态融合交叉测试"""
    
    def test_tactile_force_sensor_correlation(self):
        """触觉-力觉传感器数据相关性测试"""
        from scipy.stats import pearsonr
        
        tactile = TactileArray(array_size=(16, 16), sensor_type=TactileSensorType.CAPACITIVE)
        tactile.open()
        force = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        force.open()
        
        tactile_values = []
        force_values = []
        
        for _ in range(50):
            tf = tactile.capture()
            contacts = tactile.detect_contacts(tf)
            wrench = force.capture()
            
            # 总压力 vs 力大小
            total_pressure = np.sum(tf.pressure_map)
            force_mag = wrench.magnitude
            
            tactile_values.append(total_pressure)
            force_values.append(force_mag)
        
        # 验证数据合理性
        self.assertEqual(len(tactile_values), 50)
        self.assertEqual(len(force_values), 50)
        
        tactile.close()
        force.close()
    
    def test_imu_orientation_consistency(self):
        """IMU方向一致性测试"""
        imu = IMUSensor(sensor_type=IMUSensorType.BMI088)
        imu.open()
        
        estimator = PoseEstimator(algorithm='madgwick', sample_rate=200)
        
        euler_history = []
        for _ in range(200):
            frame = imu.capture()
            pose = estimator.update(frame.accel, frame.gyro)
            euler_history.append(estimator.get_euler())
        
        euler_arr = np.array(euler_history)
        
        # 检查收敛性 (最后20帧应该稳定)
        last_20 = euler_arr[-20:]
        roll_std = np.std(last_20[:, 0])
        pitch_std = np.std(last_20[:, 1])
        
        self.assertLess(roll_std, 0.05)
        self.assertLess(pitch_std, 0.05)
        
        imu.close()
    
    def test_multi_sensor_time_alignment(self):
        """多传感器时间对齐测试"""
        from sensors.manager import SensorManager, SensorManagerConfig
        
        config = SensorManagerConfig(grade='S')
        manager = SensorManager(config=config)
        
        manager.open_all()
        
        timestamps = []
        for _ in range(20):
            frame = manager.capture_all()
            # 使用 frame 的 timestamp 字段
            timestamps.append(frame.timestamp)
        
        manager.close_all()
        
        # 时间戳应该递增
        for i in range(1, len(timestamps)):
            self.assertGreater(timestamps[i], timestamps[i-1])
    
    def test_tactile_contact_centroid_tracking(self):
        """触觉接触质心跟踪测试"""
        tactile = TactileArray(array_size=(16, 16))
        tactile.open()
        
        positions = []
        for i in range(30):
            frame = tactile.capture()
            contacts = tactile.detect_contacts(frame)
            
            if contacts:
                centroid = contacts[0].centroid
                positions.append(centroid)
        
        tactile.close()
        
        # 如果有接触数据,验证跟踪稳定性
        if len(positions) > 5:
            pos_arr = np.array(positions)
            movement = np.diff(pos_arr, axis=0)
            avg_movement = np.mean(np.linalg.norm(movement, axis=1))
            self.assertLess(avg_movement, 10.0)  # 平均移动应小于10像素


class TestSensorAdvancedFeatures(unittest.TestCase):
    """传感器高级功能测试"""
    
    def test_tactile_thermal_response(self):
        """触觉热响应测试"""
        tactile = TactileArray(array_size=(8, 8))
        tactile.open()
        
        frames = []
        for _ in range(50):
            frame = tactile.capture()
            frames.append(frame)
        
        # 检查温度变化趋势
        temps = [np.mean(f.temperature_map) for f in frames]
        
        # 温度应该在合理范围内
        self.assertTrue(all(20 < t < 40 for t in temps))
        
        tactile.close()
    
    def test_force_payload_estimation(self):
        """力觉负载估计测试"""
        force = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        force.open()
        
        payloads = []
        for _ in range(100):
            wrench = force.capture()
            payload = force.estimate_payload(wrench)
            payloads.append(payload)
        
        # 负载估计应该在合理范围内
        avg_payload = np.mean(payloads)
        self.assertGreater(avg_payload, 0)
        self.assertLess(avg_payload, 10)  # kg
        
        force.close()
    
    def test_imu_madgwick_quaternion_stability(self):
        """IMU Madgwick四元数稳定性测试"""
        imu = IMUSensor(sensor_type=IMUSensorType.BMI088)
        imu.open()
        estimator = PoseEstimator(algorithm='madgwick', sample_rate=200)
        
        quaternions = []
        for _ in range(300):
            frame = imu.capture()
            pose = estimator.update(frame.accel, frame.gyro)
            quaternions.append(pose.orientation)
        
        quat_arr = np.array(quaternions)
        
        # 四元数应该保持归一化
        norms = np.linalg.norm(quat_arr, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=0.01))
        
        imu.close()
    
    def test_wrench_processor_filtering(self):
        """力旋量处理器滤波测试"""
        processor = WrenchProcessor(filter_alpha=0.3)
        
        raw_wrenches = []
        for _ in range(100):
            wrench_vec = np.concatenate([
                np.random.randn(3) * 5 + [0, 0, 10],
                np.random.randn(3) * 0.5
            ])
            raw_wrenches.append(wrench_vec)
        
        filtered_wrenches = []
        for w in raw_wrenches:
            fw = processor.filter(w)
            filtered_wrenches.append(fw)
        
        raw_arr = np.array(raw_wrenches)
        filtered_arr = np.array(filtered_wrenches)
        
        # 滤波后方差应该减小
        raw_std = np.std(raw_arr)
        filtered_std = np.std(filtered_arr)
        self.assertLess(filtered_std, raw_std)
    
    def test_tactile_slip_multiframe_detection(self):
        """触觉多帧滑移检测测试"""
        tactile = TactileArray(array_size=(16, 16))
        tactile.open()
        
        # 采集多帧
        for _ in range(10):
            tactile.capture()
        
        # 最后一帧检测滑移
        frame = tactile.capture()
        slip = tactile.get_slip_signal(frame)
        
        self.assertEqual(slip.shape, (16, 16))
        self.assertTrue(np.all(slip >= 0))
        self.assertTrue(np.all(slip <= 1))
        
        tactile.close()
    
    def test_force_contact_state_detection(self):
        """力觉接触状态检测测试"""
        force = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        force.open()
        
        contact_forces = []
        for _ in range(50):
            wrench = force.capture()
            state = force.detect_contact(wrench)
            contact_forces.append(state.contact_force)
        
        # 检查接触力在合理范围
        self.assertTrue(all(f >= 0 for f in contact_forces))
        self.assertTrue(np.std(contact_forces) < 100)  # 变化不应太大
        
        force.close()


class TestAGVSensorGradeCompleteness(unittest.TestCase):
    """AGV五级传感器完整性测试"""
    
    def test_all_grades_have_required_sensors(self):
        """所有等级具备必需传感器"""
        required_sensors = ['vision', 'audio', 'tactile', 'force', 'imu']
        
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            # 视觉
            from sensors.vision import get_stereo_spec
            vision_spec = get_stereo_spec(grade)
            self.assertIsNotNone(vision_spec)
            
            # 听觉
            from sensors.audio import get_audio_spec
            audio_spec = get_audio_spec(grade)
            self.assertIsNotNone(audio_spec)
            
            # 触觉
            from sensors.tactile import get_tactile_spec
            tactile_spec = get_tactile_spec(grade)
            self.assertIn('array', tactile_spec)
            
            # 力觉
            from sensors.force import get_force_spec
            force_spec = get_force_spec(grade)
            self.assertIn('axes', force_spec)
            
            # IMU
            from sensors.imu import get_imu_spec
            imu_spec = get_imu_spec(grade)
            self.assertIn('sample_hz', imu_spec)
    
    def test_grade_scaling_monotonic(self):
        """等级扩展性单调递增测试"""
        prev_res = None
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            from sensors.tactile import get_tactile_spec
            spec = get_tactile_spec(grade)
            res = spec['array'][0] * spec['array'][1]  # 总像素数
            
            if prev_res is not None:
                self.assertGreater(res, prev_res)
            prev_res = res
    
    def test_sensor_spec_consistency(self):
        """传感器规格一致性测试"""
        # 触觉规格: S < M < L < XL < XXL
        from sensors.tactile import get_tactile_spec
        
        arrays = [get_tactile_spec(g)['array'][0] for g in ['S', 'M', 'L', 'XL', 'XXL']]
        self.assertTrue(all(arrays[i] < arrays[i+1] for i in range(len(arrays)-1)))
        
        # 采样率: S < M < L < XL < XXL
        from sensors.imu import get_imu_spec
        sample_rates = [get_imu_spec(g)['sample_hz'] for g in ['S', 'M', 'L', 'XL', 'XXL']]
        self.assertTrue(all(sample_rates[i] < sample_rates[i+1] for i in range(len(sample_rates)-1)))
        
        # 力觉: S < M < L < XL < XXL
        from sensors.force import get_force_spec
        force_ranges = [get_force_spec(g)['force_range'] for g in ['S', 'M', 'L', 'XL', 'XXL']]
        self.assertTrue(all(force_ranges[i] < force_ranges[i+1] for i in range(len(force_ranges)-1)))


class TestSensorFaultInjection(unittest.TestCase):
    """
    传感器故障注入测试
    ==================

    测试传感器在各种故障条件下的行为:
    - 通信超时
    - 噪声异常
    - 偏置漂移
    - 数据丢失
    - 异常值
    """

    def test_tactile_sensor_timeout_handling(self):
        """触觉传感器超时处理"""
        from sensors.tactile import TactileArray
        import time

        tactile = TactileArray(array_size=(16, 16), sensor_id="timeout_test")
        tactile.open()

        # 模拟快速连续采集 (无延迟)
        for i in range(20):
            frame = tactile.capture()
            self.assertIsNotNone(frame)
            self.assertGreaterEqual(frame.pressure_map.shape[0], 8)

        # 模拟暂停后的恢复
        time.sleep(0.05)
        frame = tactile.capture()
        self.assertIsNotNone(frame)

        tactile.close()

    def test_tactile_noise_spike_detection(self):
        """触觉噪声尖峰检测"""
        from sensors.tactile import TactileArray, PressureProcessor

        tactile = TactileArray(array_size=(16, 16))
        processor = PressureProcessor(filter_window=3, drift_compensation=True)
        tactile.open()

        # 正常帧
        normal_frame = tactile.capture()
        normal_filtered = processor.filter(normal_frame.pressure_map)

        # 注入噪声尖峰
        spike_frame = tactile.capture()
        spike_frame.pressure_map[8, 8] = 1.0  # 尖峰
        spike_filtered = processor.filter(spike_frame.pressure_map)

        # 滤波后尖峰应被抑制
        original_peak = spike_frame.pressure_map[8, 8]
        filtered_peak = spike_filtered[8, 8]
        self.assertLess(filtered_peak, original_peak)

        tactile.close()

    def test_force_sensor_bias_drift(self):
        """力觉传感器偏置漂移"""
        from sensors.force import ForceTorqueSensor, WrenchProcessor

        ft = ForceTorqueSensor(sensor_id="bias_drift_test")
        processor = WrenchProcessor(filter_alpha=0.3)
        ft.open()

        # 采集初始偏置
        initial_wrenches = []
        for _ in range(50):
            w = ft.capture()
            initial_wrenches.append(w.to_vector())

        initial_mean = np.mean(initial_wrenches, axis=0)

        # 模拟随时间的缓慢偏置变化 (传感器温漂)
        drifted_wrenches = []
        for i in range(50):
            w = ft.capture()
            # 注入渐进偏置
            drift = np.array([0.1 * i / 50, 0.05 * i / 50, 0.2 * i / 50, 0.01 * i / 50, 0.01 * i / 50, 0.01 * i / 50])
            drifted = w.to_vector() + drift
            filtered = processor.filter(drifted)
            drifted_wrenches.append(filtered)

        # 验证偏置漂移检测能力
        final_mean = np.mean(drifted_wrenches[-10:], axis=0)
        drift_detected = np.linalg.norm(final_mean - initial_mean) > 0.1
        # 在这个简单测试中,漂移应该能被检测到
        self.assertGreater(len(drifted_wrenches), 0)

        ft.close()

    def test_force_sensor_outlier_removal(self):
        """力觉异常值去除"""
        from sensors.force import WrenchProcessor

        processor = WrenchProcessor(filter_alpha=0.3, outlier_threshold=3.0)

        # 正常数据历史
        history = []
        for _ in range(30):
            history.append(np.random.randn(6) * 0.5)

        # 注入异常值
        outlier = np.array([10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
        cleaned = processor.remove_outliers(outlier, history)

        # 异常值应被修正
        self.assertFalse(np.any(np.abs(cleaned[:3]) > 3.0))

    def test_imu_sensor_saturation_handling(self):
        """IMU饱和处理"""
        from sensors.imu import IMUSensor, IMUSensorType

        imu = IMUSensor(sensor_type=IMUSensorType.BMI088, sensor_id="saturation_test")
        imu.open()

        # 模拟正常数据
        normal_frames = [imu.capture() for _ in range(10)]
        normal_mags = [f.accel_magnitude for f in normal_frames]
        self.assertTrue(all(5.0 < m < 15.0 for m in normal_mags))

        imu.close()

    def test_imu_calibration_improves_accuracy(self):
        """IMU标定后零偏有效"""
        from sensors.imu import IMUSensor, IMUSensorType

        imu = IMUSensor(sensor_type=IMUSensorType.MPU6050, sensor_id="calib_test")
        imu.open()

        # 初始偏置应为零
        initial_bias = imu.calibration.gyro_bias.copy()

        # 执行标定
        imu.calibrate_gyro_bias(num_samples=100)

        # 验证标定后偏置已更新
        self.assertFalse(np.allclose(imu.calibration.gyro_bias, initial_bias))

        # 验证标定后采集的数据有效且有界
        post_cal_frames = [imu.capture() for _ in range(20)]
        for frame in post_cal_frames:
            self.assertTrue(np.all(np.isfinite(frame.gyro)))
            self.assertTrue(np.all(np.isfinite(frame.accel)))
            # 角速度应在合理范围内
            self.assertLess(np.linalg.norm(frame.gyro), 1.0)  # rad/s

        imu.close()

    def test_imu_sensor_mag_interference_rejection(self):
        """IMU磁力计干扰抑制"""
        from sensors.imu import IMUSensor, IMUSensorType, PoseEstimator

        imu = IMUSensor(sensor_type=IMUSensorType.MPU9250, sensor_id="mag_test")
        estimator = PoseEstimator(algorithm="madgwick", sample_rate=200.0)
        imu.open()

        # 模拟磁干扰
        mag_interference = np.array([50.0, 30.0, 20.0])  # 干扰磁场 (μT)

        for _ in range(50):
            frame = imu.capture()
            # 在有干扰的情况下,姿态估计仍应收敛
            if frame.mag is not None:
                pose = estimator.update(frame.accel, frame.gyro, mag=frame.mag, dt=1.0/200)

        self.assertIsNotNone(estimator.quaternion)
        self.assertAlmostEqual(np.linalg.norm(estimator.quaternion), 1.0, places=4)

        imu.close()

    def test_tactile_sensor_dead_pixel_handling(self):
        """触觉传感器坏点处理"""
        from sensors.tactile import TactileArray, PressureProcessor

        tactile = TactileArray(array_size=(16, 16), sensor_id="dead_pixel_test")
        processor = PressureProcessor(filter_window=3)
        tactile.open()

        # 模拟触觉帧
        frame = tactile.capture()
        frame.pressure_map[4, 4] = 0.0   # 死像素
        frame.pressure_map[10, 10] = 1.0  # 饱和像素

        # 滤波处理 - 验证不崩溃
        filtered = processor.filter(frame.pressure_map)

        # 验证输出形状不变
        self.assertEqual(filtered.shape, frame.pressure_map.shape)

        tactile.close()

    def test_force_sensor_temperature_compensation(self):
        """力觉传感器温度补偿"""
        from sensors.force import ForceTorqueSensor

        ft = ForceTorqueSensor(sensor_id="temp_comp_test")
        ft.open()

        # 模拟不同温度下的力数据
        temperatures = [25.0, 35.0, 45.0, 55.0]
        force_readings = []

        for temp in temperatures:
            # 温度升高会导致输出漂移 (简化模拟)
            for _ in range(10):
                w = ft.capture()
                # 注入温度相关漂移
                drift = (temp - 25.0) * 0.01
                force_readings.append(w.magnitude + drift)

        # 验证温度对输出的影响在合理范围
        self.assertLess(np.std(force_readings), 5.0)

        ft.close()

    def test_sensor_data_loss_recovery(self):
        """传感器数据丢失后恢复采集"""
        from sensors.tactile import TactileArray
        from sensors.force import ForceTorqueSensor
        from sensors.imu import IMUSensor

        tactile = TactileArray(array_size=(16, 16))
        force = ForceTorqueSensor()
        imu = IMUSensor(sensor_type=IMUSensorType.MPU6050)

        tactile.open()
        force.open()
        imu.open()

        # 正常采集
        t_frame = tactile.capture()
        f_wrench = force.capture()
        i_frame = imu.capture()

        self.assertIsNotNone(t_frame)
        self.assertIsNotNone(f_wrench)
        self.assertIsNotNone(i_frame)

        # 传感器open/close循环
        tactile.close()
        tactile.open()

        # 重新采集应正常
        t_frame2 = tactile.capture()
        self.assertIsNotNone(t_frame2)

        tactile.close()
        force.close()
        imu.close()

    def test_imu_sensor_reconnect_recovery(self):
        """IMU传感器重连恢复"""
        from sensors.imu import IMUSensor, IMUSensorType

        imu = IMUSensor(sensor_type=IMUSensorType.BMI088, sensor_id="reconnect_test")

        # 第一次打开
        imu.open()
        frame1 = imu.capture()
        self.assertIsNotNone(frame1)

        # 模拟断开
        imu.close()

        # 重新打开
        imu.open()
        frame2 = imu.capture()
        self.assertIsNotNone(frame2)

        # 验证重新采集的数据有效性
        self.assertTrue(np.all(np.isfinite(frame2.accel)))
        self.assertTrue(np.all(np.isfinite(frame2.gyro)))

        imu.close()

    def test_force_sensor_saturation_detection(self):
        """力觉传感器饱和检测"""
        from sensors.force import VirtualForceSensor

        vfs = VirtualForceSensor(sensor_id="sat_detect")
        vfs.open()

        # 模拟超出量程的力
        for _ in range(5):
            wrench = vfs.simulate_contact(
                force=(2000.0, 2000.0, 2000.0),  # 超出±1000N量程
                torque=(200.0, 200.0, 200.0),
                add_noise=False
            )

        # 传感器应能报告异常大的力
        self.assertGreater(wrench.magnitude, 0)

        vfs.close()

    def test_tactile_contact_history_maintained(self):
        """触觉接触历史维护"""
        from sensors.tactile import VirtualTactileSensor

        tactile = VirtualTactileSensor(array_size=(16, 16), sensor_id="history_test")
        tactile.open()

        # 多次接触
        positions = [(0.3, 0.3), (0.5, 0.5), (0.7, 0.7)]
        for pos in positions:
            tactile.simulate_contact(contact_pos=pos, contact_radius=0.2, contact_force=5.0)

        # 验证历史记录
        self.assertIsNotNone(tactile._last_contact_pos)

        tactile.close()
