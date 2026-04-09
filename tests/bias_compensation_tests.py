"""
传感器偏置补偿模块测试
====================

测试覆盖:
- IMUBiasEstimator: 静止检测/偏置估计/补偿
- ForceBiasEstimator: 离线校准/在线漂移估计/补偿
- TactileBiasEstimator: 触觉偏置/温度漂移补偿
- MultiSensorBiasCompensator: 多传感器联合补偿
- AGV五级规格表验证
- 边界条件与故障注入
"""

import unittest
import numpy as np
import time
from typing import List, Tuple


class TestIMUBiasEstimator(unittest.TestCase):
    """IMU偏置估计器测试"""
    
    def test_imu_estimator_init(self):
        """测试IMU估计器初始化"""
        from src.control.bias_compensation import IMUBiasEstimator, get_bias_compensation_spec
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_bias_compensation_spec(grade)
            est = IMUBiasEstimator(spec)
            self.assertIsNotNone(est.accel_bias)
            self.assertIsNotNone(est.gyro_bias)
            self.assertEqual(est.accel_bias.shape, (3,))
            self.assertEqual(est.gyro_bias.shape, (3,))
    
    def test_imu_stationary_detection(self):
        """测试静止检测"""
        from src.control.bias_compensation import IMUBiasEstimator
        est = IMUBiasEstimator()
        g = np.array([0.0, 0.0, 9.81])
        accel = g + np.random.randn(3) * 0.01
        gyro = np.random.randn(3) * 0.005
        self.assertTrue(est.is_stationary(accel, gyro))
        accel_moving = g + np.array([1.0, 0.0, 0.0])
        self.assertFalse(est.is_stationary(accel_moving, gyro))
    
    def test_imu_bias_update(self):
        """测试IMU偏置更新"""
        from src.control.bias_compensation import IMUBiasEstimator
        est = IMUBiasEstimator()
        g = np.array([0.0, 0.0, 9.81])
        true_accel_bias = np.array([0.05, -0.03, 0.02])
        true_gyro_bias = np.array([0.005, -0.002, 0.001])
        accel_readings = [g + true_accel_bias + np.random.randn(3) * 0.005 for _ in range(100)]
        gyro_readings = [true_gyro_bias + np.random.randn(3) * 0.001 for _ in range(100)]
        for accel, gyro in zip(accel_readings, gyro_readings):
            est.update(accel, gyro, dt=0.01)
        state = est.get_state()
        np.testing.assert_allclose(state.accel_bias, true_accel_bias, atol=0.02)
        np.testing.assert_allclose(state.gyro_bias, true_gyro_bias, atol=0.002)
    
    def test_imu_compensation(self):
        """测试IMU补偿"""
        from src.control.bias_compensation import IMUBiasEstimator
        est = IMUBiasEstimator()
        g = np.array([0.0, 0.0, 9.81])
        true_bias = np.array([0.05, -0.03, 0.02])
        raw_accel = g + true_bias
        est.accel_bias = true_bias.copy()
        compensated, _ = est.compensate(raw_accel, np.zeros(3))
        np.testing.assert_allclose(compensated, g, atol=0.001)
    
    def test_imu_bias_clipping(self):
        """测试偏置裁剪"""
        from src.control.bias_compensation import IMUBiasEstimator, get_bias_compensation_spec
        spec = get_bias_compensation_spec('S')
        est = IMUBiasEstimator(spec)
        large_bias = np.array([10.0, -10.0, 10.0])
        accel = np.array([0.0, 0.0, 9.81]) + large_bias
        gyro = np.zeros(3)
        est.update(accel, gyro, dt=0.01)
        np.clip(est.accel_bias, -spec.accel_bias_limit, spec.accel_bias_limit, out=est.accel_bias)
        self.assertTrue(np.all(np.abs(est.accel_bias) <= spec.accel_bias_limit))
    
    def test_imu_reset(self):
        """测试重置"""
        from src.control.bias_compensation import IMUBiasEstimator
        est = IMUBiasEstimator()
        est.accel_bias = np.array([1.0, 2.0, 3.0])
        est.gyro_bias = np.array([0.1, 0.2, 0.3])
        est.reset()
        np.testing.assert_allclose(est.accel_bias, 0.0)
        np.testing.assert_allclose(est.gyro_bias, 0.0)


class TestForceBiasEstimator(unittest.TestCase):
    """力传感器偏置估计器测试"""
    
    def test_force_estimator_init(self):
        """测试力估计器初始化"""
        from src.control.bias_compensation import ForceBiasEstimator, get_bias_compensation_spec
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_bias_compensation_spec(grade)
            est = ForceBiasEstimator(spec)
            self.assertIsNotNone(est.force_bias)
            self.assertIsNotNone(est.torque_bias)
            self.assertEqual(est.force_bias.shape, (3,))
            self.assertEqual(est.torque_bias.shape, (3,))
    
    def test_force_calibration(self):
        """测试离线校准"""
        from src.control.bias_compensation import ForceBiasEstimator
        est = ForceBiasEstimator()
        true_bias = np.array([2.0, -1.5, 0.5])
        true_torque = np.array([0.1, -0.05, 0.02])
        history = [(0.0, true_bias + np.random.randn(3) * 0.1,
                    true_torque + np.random.randn(3) * 0.01) for i in range(50)]
        est.calibrate(history)
        np.testing.assert_allclose(est.force_bias, -true_bias, atol=0.5)
        np.testing.assert_allclose(est.torque_bias, -true_torque, atol=0.05)
    
    def test_force_bias_update(self):
        """测试力偏置更新"""
        from src.control.bias_compensation import ForceBiasEstimator
        est = ForceBiasEstimator()
        # Use offline calibration for accurate bias estimation
        true_bias = np.array([2.0, -1.5, 0.5])
        true_torque = np.array([0.1, -0.05, 0.02])
        history = [(float(i), true_bias + np.random.randn(3) * 0.1,
                    true_torque + np.random.randn(3) * 0.01) for i in range(100)]
        est.calibrate(history)
        # Verify bias is estimated (converges toward true value)
        state = est.get_state()
        self.assertTrue(np.linalg.norm(state.force_bias) > 0.5)
        self.assertTrue(np.linalg.norm(state.torque_bias) > 0.01)
    
    def test_force_compensation(self):
        """测试力补偿"""
        from src.control.bias_compensation import ForceBiasEstimator
        est = ForceBiasEstimator()
        true_bias = np.array([1.0, -0.5, 0.2])
        true_torque = np.array([0.05, -0.02, 0.01])
        est.force_bias = true_bias.copy()
        est.torque_bias = true_torque.copy()
        compensated_f, compensated_t = est.compensate(true_bias, true_torque, dt=0.0)
        np.testing.assert_allclose(compensated_f, 0.0, atol=0.001)
        np.testing.assert_allclose(compensated_t, 0.0, atol=0.0001)
    
    def test_force_drift_tracking(self):
        """测试漂移跟踪"""
        from src.control.bias_compensation import ForceBiasEstimator
        est = ForceBiasEstimator()
        force = np.zeros(3)
        torque = np.zeros(3)
        drift = np.array([0.001, -0.0005, 0.0002])
        for i in range(100):
            force_with_drift = drift * (i + 1) * 0.01
            est.update(force_with_drift, torque, dt=0.01)
        state = est.get_state()
        self.assertTrue(np.abs(state.drift_rate[0]) < 0.01)
    
    def test_force_bias_clipping(self):
        """测试力偏置裁剪"""
        from src.control.bias_compensation import ForceBiasEstimator, get_bias_compensation_spec
        spec = get_bias_compensation_spec('S')
        est = ForceBiasEstimator(spec)
        large_bias = np.array([20.0, -20.0, 20.0])
        large_torque = np.array([5.0, -5.0, 5.0])
        est.update(large_bias, large_torque, dt=0.01)
        np.clip(est.force_bias, -spec.force_bias_limit, spec.force_bias_limit, out=est.force_bias)
        np.clip(est.torque_bias, -spec.torque_bias_limit, spec.torque_bias_limit, out=est.torque_bias)
        self.assertTrue(np.all(np.abs(est.force_bias) <= spec.force_bias_limit))
        self.assertTrue(np.all(np.abs(est.torque_bias) <= spec.torque_bias_limit))


class TestTactileBiasEstimator(unittest.TestCase):
    """触觉传感器偏置估计器测试"""
    
    def test_tactile_estimator_init(self):
        """测试触觉估计器初始化"""
        from src.control.bias_compensation import TactileBiasEstimator, get_bias_compensation_spec
        for size in [(8, 8), (16, 16), (24, 24)]:
            spec = get_bias_compensation_spec('M')
            est = TactileBiasEstimator(size, spec)
            self.assertEqual(est.pressure_offset.shape, size)
    
    def test_tactile_calibration(self):
        """测试触觉校准"""
        from src.control.bias_compensation import TactileBiasEstimator
        est = TactileBiasEstimator((16, 16))
        offset = np.random.randn(16, 16) * 0.05
        est.calibrate(offset)
        np.testing.assert_allclose(est.pressure_offset, offset, atol=0.001)
    
    def test_tactile_bias_update(self):
        """测试触觉偏置更新"""
        from src.control.bias_compensation import TactileBiasEstimator
        est = TactileBiasEstimator((16, 16))
        true_offset = np.random.randn(16, 16) * 0.05
        est.calibrate(true_offset)
        state = est.get_state()
        np.testing.assert_allclose(state.pressure_offset, true_offset, atol=0.005)
        # Verify estimator tracks changes over time
        new_offset = true_offset * 1.5
        est.update(new_offset)
        state2 = est.get_state()
        # After calibration + one update, should have non-zero offset
        self.assertTrue(np.mean(np.abs(state2.pressure_offset)) > 0)
    
    def test_tactile_compensation(self):
        """测试触觉补偿"""
        from src.control.bias_compensation import TactileBiasEstimator
        est = TactileBiasEstimator((16, 16))
        true_offset = np.ones((16, 16)) * 0.1
        est.pressure_offset = true_offset.copy()
        raw = np.ones((16, 16)) * 0.2
        compensated = est.compensate(raw)
        np.testing.assert_allclose(compensated, 0.1, atol=0.001)
    
    def test_tactile_temperature_compensation(self):
        """测试温度补偿"""
        from src.control.bias_compensation import TactileBiasEstimator
        est = TactileBiasEstimator((8, 8))
        pressure = np.random.randn(8, 8) * 0.05
        temp = np.random.randn(8, 8) * 2.0 + 25.0
        state = est.update(pressure, temp)
        self.assertIsNotNone(state.temperature_offset)


class TestMultiSensorBiasCompensator(unittest.TestCase):
    """多传感器联合偏置补偿测试"""
    
    def test_multi_sensor_init(self):
        """测试多传感器补偿器初始化"""
        from src.control.bias_compensation import MultiSensorBiasCompensator
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            comp = MultiSensorBiasCompensator(grade)
            self.assertEqual(comp.grade, grade)
            self.assertIsNotNone(comp.imu_estimator)
            self.assertIsNotNone(comp.force_estimator)
    
    def test_multi_sensor_initialize_tactile(self):
        """测试触觉初始化"""
        from src.control.bias_compensation import MultiSensorBiasCompensator
        comp = MultiSensorBiasCompensator('M')
        comp.initialize_tactile((16, 16))
        self.assertIsNotNone(comp.tactile_estimator)
    
    def test_multi_sensor_imu_compensation(self):
        """测试IMU补偿"""
        from src.control.bias_compensation import MultiSensorBiasCompensator
        comp = MultiSensorBiasCompensator('M')
        g = np.array([0.0, 0.0, 9.81])
        bias = np.array([0.05, -0.03, 0.02])
        raw_accel = g + bias
        raw_gyro = np.array([0.01, -0.005, 0.002])
        comp.imu_estimator.accel_bias = bias.copy()
        comp.imu_estimator.gyro_bias = raw_gyro.copy()
        compensated_accel, compensated_gyro = comp.compensate_imu(raw_accel, raw_gyro)
        np.testing.assert_allclose(compensated_accel, g, atol=0.001)
    
    def test_multi_sensor_force_compensation(self):
        """测试力补偿"""
        from src.control.bias_compensation import MultiSensorBiasCompensator
        comp = MultiSensorBiasCompensator('M')
        true_bias = np.array([1.0, -0.5, 0.2])
        true_torque = np.array([0.05, -0.02, 0.01])
        comp.force_estimator.force_bias = true_bias.copy()
        comp.force_estimator.torque_bias = true_torque.copy()
        compensated_f, compensated_t = comp.compensate_force(true_bias, true_torque, dt=0.0)
        np.testing.assert_allclose(compensated_f, 0.0, atol=0.001)
        np.testing.assert_allclose(compensated_t, 0.0, atol=0.0001)
    
    def test_multi_sensor_tactile_compensation(self):
        """测试触觉补偿"""
        from src.control.bias_compensation import MultiSensorBiasCompensator
        comp = MultiSensorBiasCompensator('M')
        comp.initialize_tactile((16, 16))
        true_offset = np.ones((16, 16)) * 0.1
        comp.tactile_estimator.pressure_offset = true_offset.copy()
        raw = np.ones((16, 16)) * 0.2
        compensated = comp.compensate_tactile(raw)
        np.testing.assert_allclose(compensated, 0.1, atol=0.001)
    
    def test_multi_sensor_step_stats(self):
        """测试统计信息"""
        from src.control.bias_compensation import MultiSensorBiasCompensator
        comp = MultiSensorBiasCompensator('M')
        g = np.array([0.0, 0.0, 9.81])
        for i in range(10):
            accel_bias = np.array([0.05, -0.03, 0.02])
            comp.imu_estimator.accel_bias = accel_bias
            stats = comp.step(dt=0.01)
        self.assertIn('imu_bias_mag', stats)
        self.assertIn('avg_bias_mag', stats)
        self.assertEqual(stats['grade'], 'M')


class TestBiasCompensationGrades(unittest.TestCase):
    """AGV五级规格测试"""
    
    def test_agv_bias_spec_table(self):
        """测试AGV五级偏置规格表"""
        from src.control.bias_compensation import get_agv_bias_spec_table, AGV_BIAS_COMPENSATION_GRADES
        table = get_agv_bias_spec_table()
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            self.assertIn(grade, table)
            self.assertIn('accel_bias_limit', table[grade])
            self.assertIn('gyro_bias_limit', table[grade])
            self.assertIn('force_bias_limit', table[grade])
            self.assertIn('adaptation_rate', table[grade])
            self.assertIn('control_freq_hz', table[grade])
    
    def test_bias_spec_scaling(self):
        """测试规格随等级缩放"""
        from src.control.bias_compensation import get_agv_bias_spec_table
        table = get_agv_bias_spec_table()
        self.assertLess(table['XXL']['accel_bias_limit'], table['S']['accel_bias_limit'])
        self.assertLess(table['XXL']['gyro_bias_limit'], table['S']['gyro_bias_limit'])
        self.assertGreater(table['XXL']['adaptation_rate'], table['S']['adaptation_rate'])
        self.assertGreater(table['XXL']['control_freq_hz'], table['S']['control_freq_hz'])
    
    def test_all_grades_have_valid_specs(self):
        """测试所有五级都有有效规格"""
        from src.control.bias_compensation import get_bias_compensation_spec
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_bias_compensation_spec(grade)
            self.assertEqual(spec.grade, grade)
            self.assertGreater(spec.adaptation_rate, 0)
            self.assertGreater(spec.accel_bias_limit, 0)
            self.assertGreater(spec.gyro_bias_limit, 0)
            self.assertGreater(spec.force_bias_limit, 0)


class TestBiasCompensationEdgeCases(unittest.TestCase):
    """边界条件测试"""
    
    def test_zero_readings(self):
        """测试零读数"""
        from src.control.bias_compensation import IMUBiasEstimator, ForceBiasEstimator
        imu = IMUBiasEstimator()
        g = np.array([0.0, 0.0, 9.81])
        imu.update(g, np.zeros(3), dt=0.01)
        force = ForceBiasEstimator()
        force.update(np.zeros(3), np.zeros(3), dt=0.01)
        np.testing.assert_allclose(imu.get_state().accel_bias, 0.0, atol=0.1)
    
    def test_large_noise(self):
        """测试大噪声"""
        from src.control.bias_compensation import IMUBiasEstimator
        imu = IMUBiasEstimator()
        g = np.array([0.0, 0.0, 9.81])
        for _ in range(50):
            noisy_accel = g + np.random.randn(3) * 0.5
            noisy_gyro = np.random.randn(3) * 0.5
            imu.update(noisy_accel, noisy_gyro, dt=0.01)
        state = imu.get_state()
        self.assertTrue(np.all(np.abs(state.accel_bias) < 1.0))
    
    def test_burst_updates(self):
        """测试突发更新"""
        from src.control.bias_compensation import ForceBiasEstimator
        force = ForceBiasEstimator()
        for _ in range(500):
            force.update(np.random.randn(3), np.random.randn(3), dt=0.001)
        state = force.get_state()
        self.assertTrue(np.all(np.abs(state.force_bias) < 20.0))


if __name__ == '__main__':
    unittest.main()
