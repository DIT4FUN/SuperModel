"""
融合模块测试用例
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion.sensor_fusion import (
    SensorFusion, ComplementaryFilter, ExtendedKalmanFilter, MultiSensorFusion
)
from sensors.imu import BMI088, MPU9250, IMUData
from sensors.tactile import PressureSensor, TactileData
from sensors.force import SixAxisFTSensor, ForceData


class TestComplementaryFilter(unittest.TestCase):
    """互补滤波器测试"""

    def setUp(self):
        self.filter = ComplementaryFilter(alpha=0.96)

    def test_initialization(self):
        """测试初始化"""
        self.assertAlmostEqual(self.filter.alpha, 0.96)
        self.assertFalse(self.filter._initialized)

    def test_update_with_accel_gyro(self):
        """测试加速度+陀螺仪更新"""
        accel = np.array([0.0, 0.0, -9.81])
        gyro = np.array([0.0, 0.0, 0.1])
        
        state = self.filter.update({'accel': accel, 'gyro': gyro}, dt=0.01)
        self.assertEqual(len(state), 3)
        self.assertTrue(self.filter._initialized)

    def test_update_gyro_only(self):
        """测试仅陀螺仪更新"""
        gyro = np.array([0.1, 0.2, 0.3])
        state1 = self.filter.update({'gyro': gyro}, dt=0.01)
        state2 = self.filter.update({'gyro': gyro}, dt=0.01)
        # yaw应该累积
        self.assertNotEqual(state2[2], state1[2])

    def test_reset(self):
        """测试重置"""
        self.filter.update({'accel': np.array([0, 0, -9.81]), 'gyro': np.array([0.1, 0.2, 0.3])}, dt=0.01)
        self.filter.reset()
        self.assertEqual(self.filter._pitch, 0.0)
        self.assertEqual(self.filter._roll, 0.0)
        self.assertFalse(self.filter._initialized)

    def test_get_state(self):
        """测试获取状态"""
        self.filter.update({'accel': np.array([0, 0, -9.81]), 'gyro': np.zeros(3)}, dt=0.01)
        state = self.filter.get_state()
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(len(state), 3)

    def test_convergence(self):
        """测试收敛性"""
        # 多次更新后，pitch/roll应接近加速度计算的值
        accel = np.array([0.0, 0.0, -9.81])
        
        for _ in range(100):
            self.filter.update({'accel': accel, 'gyro': np.array([0.0, 0.0, 0.0])}, dt=0.01)
        
        state = self.filter.get_state()
        # pitch应该接近0 (水平), roll接近0
        self.assertAlmostEqual(state[0], 0.0, places=2)
        self.assertAlmostEqual(state[1], 0.0, places=2)


class TestExtendedKalmanFilter(unittest.TestCase):
    """扩展卡尔曼滤波器测试"""

    def test_initialization(self):
        """测试EKF初始化"""
        ekf = ExtendedKalmanFilter(state_dim=5, measurement_dim=3)
        self.assertEqual(ekf.state_dim, 5)
        self.assertEqual(ekf.measurement_dim, 3)
        self.assertEqual(len(ekf.get_state()), 5)

    def test_initialize_state(self):
        """测试状态初始化"""
        ekf = ExtendedKalmanFilter(state_dim=3, measurement_dim=3)
        initial_state = np.array([1.0, 2.0, 3.0])
        ekf.initialize(initial_state)
        state = ekf.get_state()
        np.testing.assert_array_almost_equal(state, initial_state)

    def test_predict(self):
        """测试预测步骤"""
        ekf = ExtendedKalmanFilter(state_dim=3, measurement_dim=3)
        ekf.initialize(np.array([0.0, 0.0, 0.0]))
        ekf.predict(dt=0.1)
        # 匀速模型下，状态不变
        state = ekf.get_state()
        np.testing.assert_array_almost_equal(state, np.zeros(3))

    def test_correct(self):
        """测试校正步骤"""
        ekf = ExtendedKalmanFilter(state_dim=3, measurement_dim=3)
        ekf.initialize(np.zeros(3))
        # 设置观测矩阵为单位阵
        ekf.H = np.eye(3)
        
        measurement = np.array([1.0, 2.0, 3.0])
        ekf.correct(measurement)
        
        state = ekf.get_state()
        # 校正后状态应接近测量值
        np.testing.assert_array_almost_equal(state, measurement, decimal=1)

    def test_full_update(self):
        """测试完整EKF更新"""
        ekf = ExtendedKalmanFilter(state_dim=2, measurement_dim=2)
        ekf.initialize(np.array([0.0, 0.0]))
        ekf.H = np.eye(2)
        
        measurements = {'sensor1': np.array([1.0, 2.0])}
        state = ekf.update(measurements, dt=0.01)
        
        self.assertEqual(len(state), 2)
        self.assertIsInstance(state, np.ndarray)

    def test_covariance(self):
        """测试协方差矩阵"""
        ekf = ExtendedKalmanFilter(state_dim=3, measurement_dim=3)
        P = ekf.get_covariance()
        self.assertEqual(P.shape, (3, 3))


class TestMultiSensorFusion(unittest.TestCase):
    """多传感器融合测试"""

    def setUp(self):
        self.fusion = MultiSensorFusion()
        self.fusion.add_fusion_method("imu1", ComplementaryFilter(alpha=0.96), weight=1.0)
        self.fusion.add_fusion_method("imu2", ComplementaryFilter(alpha=0.98), weight=0.5)

    def test_add_fusion_method(self):
        """测试添加融合方法"""
        self.assertEqual(len(self.fusion.fusion_methods), 2)
        self.assertIn("imu1", self.fusion.fusion_methods)

    def test_update_multiple_sensors(self):
        """测试多传感器更新"""
        sensor_data = {
            "imu1": {
                'accel': np.array([0.0, 0.0, -9.81]),
                'gyro': np.array([0.0, 0.0, 0.1])
            },
            "imu2": {
                'accel': np.array([0.1, 0.0, -9.81]),
                'gyro': np.array([0.0, 0.0, 0.1])
            }
        }
        
        results = self.fusion.update(sensor_data, dt=0.01)
        self.assertEqual(len(results), 2)

    def test_get_fused_state(self):
        """测试获取融合状态"""
        self.fusion.update({
            "imu1": {'accel': np.array([0, 0, -9.81]), 'gyro': np.zeros(3)},
            "imu2": {'accel': np.array([0, 0, -9.81]), 'gyro': np.zeros(3)},
        }, dt=0.01)
        
        fused = self.fusion.get_fused_state()
        self.assertIsInstance(fused, np.ndarray)
        self.assertGreater(len(fused), 0)


class TestFusionWithRealSensors(unittest.TestCase):
    """真实传感器数据融合测试"""

    def test_imu_complementary_fusion(self):
        """测试IMU互补滤波融合"""
        imu = BMI088("imu_test")
        fusion = ComplementaryFilter(alpha=0.96)

        for _ in range(50):
            data = imu.read()
            fusion.update({'accel': data.acceleration, 'gyro': data.angular_velocity}, dt=0.01)

        state = fusion.get_state()
        self.assertEqual(len(state), 3)
        # roll和pitch应该接近0 (静止状态)
        self.assertAlmostEqual(state[0], 0.0, places=1)
        self.assertAlmostEqual(state[1], 0.0, places=1)

    def test_ft_sensor_ekf(self):
        """测试力觉传感器EKF"""
        ft_sensor = SixAxisFTSensor("ft_test")
        ekf = ExtendedKalmanFilter(state_dim=6, measurement_dim=6)
        ekf.initialize(np.zeros(6))
        ekf.H = np.eye(6)

        for _ in range(10):
            data = ft_sensor.read()
            ekf.update({'ft': data.wrench}, dt=0.01)

        state = ekf.get_state()
        self.assertEqual(len(state), 6)

    def test_tactile_force_fusion(self):
        """测试触觉-力觉融合"""
        fusion = MultiSensorFusion()
        
        tactile = PressureSensor("tactile_test").read()
        force = SixAxisFTSensor("ft_test").read()
        
        fused = fusion.fuse_tactile_force(tactile, force)
        self.assertIsInstance(fused, np.ndarray)


class TestFusionStability(unittest.TestCase):
    """融合稳定性测试"""

    def test_ekf_stability(self):
        """测试EKF长时间运行稳定性"""
        ekf = ExtendedKalmanFilter(state_dim=3, measurement_dim=3)
        ekf.initialize(np.zeros(3))
        ekf.H = np.eye(3)
        ekf.Q = np.eye(3) * 0.001
        ekf.R = np.eye(3) * 0.1

        # 运行1000次迭代
        for i in range(1000):
            measurement = np.array([0.1, 0.2, 0.3]) + np.random.normal(0, 0.05, 3)
            ekf.update({'sensor': measurement}, dt=0.01)

        state = ekf.get_state()
        # 状态应该收敛到测量值附近
        self.assertTrue(np.all(np.abs(state) < 10.0))

    def test_complementary_filter_drift(self):
        """测试互补滤波漂移"""
        fusion = ComplementaryFilter(alpha=0.99)  # 高alpha减少漂移
        
        gyro_bias = np.array([0.001, 0.001, 0.001])  # 小陀螺仪偏置
        
        for _ in range(100):
            accel = np.array([0.0, 0.0, -9.81])
            gyro = gyro_bias
            fusion.update({'accel': accel, 'gyro': gyro}, dt=0.01)
        
        # 漂移应该很小
        state = fusion.get_state()
        self.assertLess(np.abs(state[2]), 0.5)  # yaw漂移应小于0.5rad


if __name__ == '__main__':
    unittest.main(verbosity=2)
