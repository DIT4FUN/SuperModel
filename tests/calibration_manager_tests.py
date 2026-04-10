"""
标定管理器测试
测试 CalibrationManager, IMUCalibrator, ForceCalibrator, TactileCalibrator
"""

import unittest
import numpy as np
import sys
import os
import time
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.control.calibration_manager import (
    CalibrationManager, IMUCalibrator, ForceCalibrator, TactileCalibrator,
    CalibrationConfig, CalibrationStatus,
    IMUCalibrationResult, ForceCalibrationResult,
    create_calibration_manager, get_calibration_spec,
    AGV_CALIBRATION_SPEC
)
from src.sensors.imu import VirtualIMUSensor
from src.sensors.force import VirtualForceSensor
from src.sensors.tactile import VirtualTactileSensor


class TestIMUCalibrator(unittest.TestCase):
    """IMU 标定器测试"""

    def setUp(self):
        self.sensor = VirtualIMUSensor(sensor_id="test_imu")
        self.sensor.open()
        self.config = CalibrationConfig(
            imu_sample_rate=50.0,
            imu_num_samples_per_pose=100,
            imu_num_poses=6,
            imu_collection_time_per_pose=2.0,
            agv_grade="M"
        )
        self.calibrator = IMUCalibrator(self.sensor, self.config)

    def tearDown(self):
        self.sensor.close()

    def test_imu_calibrator_init(self):
        """测试 IMU 标定器初始化"""
        self.assertEqual(self.calibrator.sensor.sensor_id, "test_imu")
        self.assertEqual(self.calibrator.config.imu_sample_rate, 50.0)
        self.assertIsNone(self.calibrator._result)

    def test_collect_data(self):
        """测试 IMU 数据采集"""
        num = self.calibrator.collect_data("x_up", duration=0.5)
        self.assertGreater(num, 0)
        self.assertEqual(len(self.calibrator._data.accel_samples), num)
        self.assertEqual(len(self.calibrator._data.gyro_samples), num)
        self.assertIn("x_up", self.calibrator._data.orientations)

    def test_collect_multiple_poses(self):
        """测试多姿态数据采集"""
        orientations = ['x_up', 'x_down', 'y_up', 'y_down', 'z_up', 'z_down']
        for orientation in orientations:
            self.calibrator.collect_data(orientation, duration=0.3)

        self.assertEqual(len(self.calibrator._data.orientations), 6)
        self.assertGreater(self.calibrator._data.num_samples, 0)

    def test_calibrate_accel_six_facing(self):
        """测试六面法加速度计标定"""
        # 采集六面数据
        orientations = ['x_up', 'x_down', 'y_up', 'y_down', 'z_up', 'z_down']
        for orientation in orientations:
            self.calibrator.collect_data(orientation, duration=0.5)

        result = self.calibrator.calibrate_accel_six_facing()

        self.assertEqual(result.status, CalibrationStatus.COMPLETED)
        self.assertEqual(len(result.accel_bias), 3)
        self.assertEqual(len(result.accel_scale), 3)
        self.assertGreater(result.noise_density_accel, 0)

    def test_calibrate_gyro_rotation(self):
        """测试陀螺仪标定"""
        # 先做加速度计标定
        orientations = ['x_up', 'x_down', 'y_up', 'y_down', 'z_up', 'z_down']
        for orientation in orientations:
            self.calibrator.collect_data(orientation, duration=0.5)

        self.calibrator.calibrate_accel_six_facing()
        result = self.calibrator.calibrate_gyro_rotation()

        self.assertEqual(result.status, CalibrationStatus.COMPLETED)
        self.assertEqual(len(result.gyro_bias), 3)
        self.assertGreater(result.noise_density_gyro, 0)

    def test_save_load_calibration(self):
        """测试标定结果保存和加载"""
        orientations = ['x_up', 'x_down', 'y_up', 'y_down', 'z_up', 'z_down']
        for orientation in orientations:
            self.calibrator.collect_data(orientation, duration=0.5)

        self.calibrator.calibrate_accel_six_facing()
        self.calibrator.calibrate_gyro_rotation()

        tmpdir = tempfile.mkdtemp()
        try:
            filepath = f"{tmpdir}/imu_cal.json"
            self.calibrator.save(filepath)

            # 重新创建标定器并加载
            new_calibrator = IMUCalibrator(self.sensor, self.config)
            loaded = new_calibrator.load(filepath)

            self.assertEqual(loaded.status, CalibrationStatus.COMPLETED)
            np.testing.assert_array_almost_equal(
                loaded.accel_bias, self.calibrator._result.accel_bias
            )
            np.testing.assert_array_almost_equal(
                loaded.gyro_bias, self.calibrator._result.gyro_bias
            )
        finally:
            shutil.rmtree(tmpdir)

    def test_get_result(self):
        """测试获取标定结果"""
        self.assertIsNone(self.calibrator.get_result())

        orientations = ['x_up', 'x_down', 'y_up', 'y_down', 'z_up', 'z_down']
        for orientation in orientations:
            self.calibrator.collect_data(orientation, duration=0.3)

        result = self.calibrator.calibrate_accel_six_facing()
        self.assertIsNotNone(self.calibrator.get_result())
        self.assertEqual(result.status, CalibrationStatus.COMPLETED)


class TestForceCalibrator(unittest.TestCase):
    """力传感器标定器测试"""

    def setUp(self):
        self.sensor = VirtualForceSensor(sensor_id="test_force")
        self.sensor.open()
        self.config = CalibrationConfig(
            force_sample_rate=50.0,
            force_num_samples=200,
            force_known_weights=[0.0, 1.0, 2.0, 5.0, 10.0],
            agv_grade="M"
        )
        self.calibrator = ForceCalibrator(self.sensor, self.config)

    def tearDown(self):
        self.sensor.close()

    def test_force_calibrator_init(self):
        """测试力传感器标定器初始化"""
        self.assertEqual(self.calibrator.sensor.sensor_id, "test_force")
        self.assertEqual(self.calibrator.config.force_num_samples, 200)
        self.assertIsNone(self.calibrator._result)

    def test_collect_zero_load(self):
        """测试零点数据采集"""
        zero_bias = self.calibrator.collect_zero_load(num_samples=50)
        self.assertEqual(len(zero_bias), 6)
        self.assertGreater(len(self.calibrator._raw_samples), 0)

    def test_calibrate_with_weights(self):
        """测试已知砝码标定"""
        # 采集零点
        zero_bias = self.calibrator.collect_zero_load(num_samples=100)

        # 标定
        result = self.calibrator.calibrate_with_weights(zero_bias)

        self.assertEqual(result.status, CalibrationStatus.COMPLETED)
        self.assertEqual(len(result.force_bias), 3)
        self.assertEqual(len(result.force_scale), 3)
        self.assertEqual(len(result.torque_bias), 3)
        self.assertEqual(len(result.torque_scale), 3)

    def test_get_result(self):
        """测试获取力标定结果"""
        self.assertIsNone(self.calibrator.get_result())

        zero_bias = self.calibrator.collect_zero_load(num_samples=200)
        result = self.calibrator.calibrate_with_weights(zero_bias)

        self.assertIsNotNone(self.calibrator.get_result())
        self.assertEqual(result.status, CalibrationStatus.COMPLETED)


class TestTactileCalibrator(unittest.TestCase):
    """触觉传感器标定器测试"""

    def setUp(self):
        self.sensor = VirtualTactileSensor(array_size=(8, 8), sensor_id="test_tactile")
        self.sensor.open()
        self.config = CalibrationConfig(
            tactile_num_samples=50,
            tactile_pressure_threshold=0.01,
            agv_grade="S"
        )
        self.calibrator = TactileCalibrator(self.sensor, self.config)

    def tearDown(self):
        self.sensor.close()

    def test_tactile_calibrator_init(self):
        """测试触觉标定器初始化"""
        self.assertEqual(self.calibrator.sensor.sensor_id, "test_tactile")
        self.assertIsNone(self.calibrator._zero_baseline)

    def test_collect_zero_baseline(self):
        """测试零压力基准采集"""
        baseline = self.calibrator.collect_zero_baseline(num_samples=30)
        self.assertEqual(baseline.shape, (8, 8))
        self.assertIsNotNone(self.calibrator._zero_baseline)
        np.testing.assert_array_equal(baseline, self.calibrator._zero_baseline)

    def test_calibrate(self):
        """测试触觉传感器标定"""
        baseline = self.calibrator.collect_zero_baseline(num_samples=30)
        params = self.calibrator.calibrate(baseline)

        self.assertIn('offset_map', params)
        self.assertIn('mean_offset', params)
        self.assertIn('std_offset', params)
        self.assertEqual(params['sensor_id'], 'test_tactile')
        self.assertEqual(params['array_size'], (8, 8))


class TestCalibrationManager(unittest.TestCase):
    """标定管理器测试"""

    def test_create_calibration_manager(self):
        """测试创建标定管理器"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            manager = create_calibration_manager(grade)
            self.assertEqual(manager.agv_grade, grade)
            self.assertEqual(manager._status, CalibrationStatus.IDLE)

    def test_calibration_manager_grade_configs(self):
        """测试各等级配置"""
        grades = ['S', 'M', 'L', 'XL', 'XXL']
        expected_rates = {'S': 50, 'M': 100, 'L': 200, 'XL': 500, 'XXL': 1000}

        for grade in grades:
            manager = create_calibration_manager(grade)
            self.assertEqual(manager.grade_config['imu_rate_hz'], expected_rates[grade])
            self.assertEqual(manager.grade_config['force_rate_hz'], expected_rates[grade])

    def test_setup_imu(self):
        """测试设置 IMU"""
        manager = create_calibration_manager("M")
        sensor = VirtualIMUSensor(sensor_id="manager_test_imu")
        manager.setup_imu(sensor)

        self.assertIsNotNone(manager.imu_sensor)
        self.assertIsNotNone(manager.imu_calibrator)
        self.assertEqual(manager.imu_calibrator.config.imu_sample_rate, 100.0)

    def test_setup_force(self):
        """测试设置力传感器"""
        manager = create_calibration_manager("M")
        sensor = VirtualForceSensor(sensor_id="manager_test_force")
        manager.setup_force(sensor)

        self.assertIsNotNone(manager.force_sensor)
        self.assertIsNotNone(manager.force_calibrator)
        self.assertEqual(manager.force_calibrator.config.force_sample_rate, 100.0)

    def test_setup_tactile(self):
        """测试设置触觉传感器"""
        manager = create_calibration_manager("M")
        sensor = VirtualTactileSensor(array_size=(16, 16), sensor_id="manager_test_tactile")
        manager.setup_tactile(sensor)

        self.assertIsNotNone(manager.tactile_sensor)
        self.assertIsNotNone(manager.tactile_calibrator)

    def test_calibrate_all(self):
        """测试完整标定流程"""
        manager = create_calibration_manager("M")

        # 设置传感器
        imu_sensor = VirtualIMUSensor(sensor_id="full_test_imu")
        force_sensor = VirtualForceSensor(sensor_id="full_test_force")
        tactile_sensor = VirtualTactileSensor(array_size=(8, 8), sensor_id="full_test_tactile")

        manager.setup_imu(imu_sensor)
        manager.setup_force(force_sensor)
        manager.setup_tactile(tactile_sensor)

        # 执行标定
        results = manager.calibrate_all()

        self.assertIn('imu', results)
        self.assertIn('force', results)
        self.assertIn('tactile', results)
        self.assertEqual(results['imu']['status'], 'completed')
        self.assertEqual(results['force']['status'], 'completed')
        self.assertEqual(manager._status, CalibrationStatus.COMPLETED)

    def test_calibrate_imu_only(self):
        """测试仅 IMU 标定"""
        manager = create_calibration_manager("M")
        imu_sensor = VirtualIMUSensor(sensor_id="imu_only_test")
        manager.setup_imu(imu_sensor)

        results = manager.calibrate_all()

        self.assertIn('imu', results)
        self.assertEqual(results['imu']['status'], 'completed')
        self.assertNotIn('force', results)

    def test_calibrate_force_only(self):
        """测试仅力传感器标定"""
        manager = create_calibration_manager("M")
        force_sensor = VirtualForceSensor(sensor_id="force_only_test")
        manager.setup_force(force_sensor)

        results = manager.calibrate_all()

        self.assertIn('force', results)
        self.assertEqual(results['force']['status'], 'completed')
        self.assertNotIn('imu', results)

    def test_calibrate_tactile_only(self):
        """测试仅触觉传感器标定"""
        manager = create_calibration_manager("M")
        tactile_sensor = VirtualTactileSensor(array_size=(8, 8), sensor_id="tactile_only_test")
        manager.setup_tactile(tactile_sensor)

        results = manager.calibrate_all()

        self.assertIn('tactile', results)
        self.assertNotIn('imu', results)
        self.assertNotIn('force', results)

    def test_save_all_calibration(self):
        """测试保存所有标定结果"""
        manager = create_calibration_manager("M")

        imu_sensor = VirtualIMUSensor(sensor_id="save_test_imu")
        force_sensor = VirtualForceSensor(sensor_id="save_test_force")
        tactile_sensor = VirtualTactileSensor(array_size=(8, 8), sensor_id="save_test_tactile")

        manager.setup_imu(imu_sensor)
        manager.setup_force(force_sensor)
        manager.setup_tactile(tactile_sensor)

        manager.calibrate_all()

        tmpdir = tempfile.mkdtemp()
        try:
            manager.save_all(tmpdir)

            import os
            self.assertTrue(os.path.exists(f"{tmpdir}/imu_calibration.json"))
            self.assertTrue(os.path.exists(f"{tmpdir}/force_calibration.json"))
            self.assertTrue(os.path.exists(f"{tmpdir}/tactile_calibration.json"))
        finally:
            shutil.rmtree(tmpdir)

    def test_get_status(self):
        """测试获取标定状态"""
        manager = create_calibration_manager("M")
        self.assertEqual(manager.get_status(), CalibrationStatus.IDLE)

        imu_sensor = VirtualIMUSensor(sensor_id="status_test_imu")
        manager.setup_imu(imu_sensor)
        manager.calibrate_all()

        self.assertEqual(manager.get_status(), CalibrationStatus.COMPLETED)

    def test_get_progress(self):
        """测试获取标定进度"""
        manager = create_calibration_manager("M")
        imu_sensor = VirtualIMUSensor(sensor_id="progress_test_imu")
        manager.setup_imu(imu_sensor)

        progress = manager.get_progress()
        self.assertEqual(progress, {})

        manager.calibrate_all()
        progress = manager.get_progress()

        self.assertIn('imu', progress)
        self.assertEqual(progress['imu'], 1.0)


class TestCalibrationSpecs(unittest.TestCase):
    """AGV 标定规格表测试"""

    def test_agv_calibration_spec_complete(self):
        """测试所有等级的标定规格"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_calibration_spec(grade)
            self.assertIn('imu_poses', spec)
            self.assertIn('imu_samples', spec)
            self.assertIn('imu_rate_hz', spec)
            self.assertIn('force_samples', spec)
            self.assertIn('force_rate_hz', spec)
            self.assertIn('tactile_samples', spec)
            self.assertIn('expected_bias_stability_mg', spec)
            self.assertIn('expected_noise_density_accel_mg_sqrt_hz', spec)
            self.assertIn('expected_noise_density_gyro_mdps_sqrt_hz', spec)

    def test_calibration_grade_imu_rate_increasing(self):
        """测试等级越高 IMU 采样率越高"""
        rates = []
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_calibration_spec(grade)
            rates.append(spec['imu_rate_hz'])

        for i in range(len(rates) - 1):
            self.assertGreater(rates[i + 1], rates[i])

    def test_calibration_grade_noise_improving(self):
        """测试等级越高噪声密度要求越严格"""
        noise_levels = []
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_calibration_spec(grade)
            noise_levels.append(spec['expected_noise_density_accel_mg_sqrt_hz'])

        for i in range(len(noise_levels) - 1):
            self.assertLess(noise_levels[i + 1], noise_levels[i])

    def test_agv_calibration_grades_manager_configs(self):
        """测试 CalibrationManager 的等级配置"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            manager = create_calibration_manager(grade)
            grade_cfg = manager.grade_config

            # 验证配置与规格表一致
            spec = get_calibration_spec(grade)
            self.assertEqual(grade_cfg['imu_poses'], spec['imu_poses'])
            self.assertEqual(grade_cfg['imu_rate_hz'], spec['imu_rate_hz'])
            self.assertEqual(grade_cfg['force_samples'], spec['force_samples'])
            self.assertEqual(grade_cfg['force_rate_hz'], spec['force_rate_hz'])
            self.assertEqual(grade_cfg['tactile_samples'], spec['tactile_samples'])

    def test_get_calibration_spec_invalid_grade(self):
        """测试无效等级返回默认配置"""
        spec = get_calibration_spec("INVALID")
        default = get_calibration_spec("M")
        self.assertEqual(spec, default)


class TestCalibrationConfig(unittest.TestCase):
    """标定配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = CalibrationConfig()
        self.assertEqual(config.imu_sample_rate, 100.0)
        self.assertEqual(config.imu_num_samples_per_pose, 500)
        self.assertEqual(config.imu_num_poses, 6)
        self.assertEqual(config.force_num_samples, 500)
        self.assertEqual(config.tactile_num_samples, 100)
        self.assertEqual(config.agv_grade, "M")

    def test_custom_config(self):
        """测试自定义配置"""
        config = CalibrationConfig(
            imu_sample_rate=200.0,
            imu_num_poses=8,
            force_known_weights=[0.0, 5.0, 10.0, 20.0],
            agv_grade="L"
        )
        self.assertEqual(config.imu_sample_rate, 200.0)
        self.assertEqual(config.imu_num_poses, 8)
        self.assertEqual(len(config.force_known_weights), 4)
        self.assertEqual(config.agv_grade, "L")


if __name__ == '__main__':
    unittest.main(verbosity=2)
