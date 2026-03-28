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
    TactileSensorType, get_tactile_spec
)
from sensors.force import (
    ForceTorqueSensor, ForceCalibration, Wrench, ContactState, WrenchProcessor,
    ForceSensorType, get_force_spec
)
from sensors.imu import (
    IMUSensor, IMUFrame, Pose, PoseEstimator, IMUCalibration, IMUSensorType,
    get_imu_spec
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
        self.assertAlmostEqual(tdoa, 0.0, places=2)
    
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
        self.assertIsInstance(force, float)
        self.assertGreater(force, 0)


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
        wrench = Wrench(force=np.array([10.0, 0.0, 0.0]), torque=np.array([0.0, 0.0, 0.0]))
        R = np.eye(3)
        t = np.array([0.1, 0.0, 0.0])
        new_wrench = wrench.transform(R, t)
        # 力不改变方向
        np.testing.assert_array_almost_equal(new_wrench.force, wrench.force)
        # 力矩应改变 (因为平移)
        self.assertNotEqual(new_wrench.torque_magnitude, 0)
    
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


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
