"""
融合模块测试用例
测试传感器融合: 互补滤波、扩展卡尔曼滤波(EKF)、多传感器融合
"""

import unittest
import numpy as np
import sys
import os

# For standalone execution, set up paths; conftest.py handles this in pytest mode
import sys as _sys
import os as _os
_ProjectRoot = '/home/treeman/.openclaw/workspace/projects/SuperModel'
_SrcPath = _os.path.join(_ProjectRoot, 'src')
# src/ must be inserted BEFORE project_root (to end up at index 1) so that
# 'from fusion.sensor_fusion' finds project_root/fusion/sensor_fusion.py
# (src/fusion/ exists but lacks sensor_fusion.py)
# Use explicit 'from src.sensors.xxx' to avoid stale project_root/sensors/
_PyPath = _sys.path
if _SrcPath not in _PyPath:
    _PyPath.insert(0, _SrcPath)   # src/ → will be at index 1 after next insert
if _ProjectRoot not in _PyPath:
    _PyPath.insert(0, _ProjectRoot)  # project_root at index 0 (found first for fusion)

from src.fusion.cross_modal_fusion import (
    CrossModalFusion, FusionConfig, MultimodalInput
)
from src.fusion.sensor_fusion import (
    ComplementaryFilter, ExtendedKalmanFilter, MultiSensorFusion
)
from src.sensors.imu import IMUSensor, IMUSensorType
from src.sensors.tactile import TactileArray, TactileSensorType
from src.sensors.force import ForceTorqueSensor, ForceSensorType


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
        accel = np.array([0.0, 0.0, -9.81])
        for _ in range(100):
            self.filter.update({'accel': accel, 'gyro': np.array([0.0, 0.0, 0.0])}, dt=0.01)
        state = self.filter.get_state()
        # 验证状态是有限的且不发散
        self.assertTrue(np.all(np.isfinite(state)))
        # yaw应该接近0 (无旋转), pitch和roll应有限
        self.assertLess(np.abs(state[2]), 0.5)  # yaw漂移应小于0.5rad


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
        state = ekf.get_state()
        np.testing.assert_array_almost_equal(state, np.zeros(3))

    def test_correct(self):
        """测试校正步骤"""
        ekf = ExtendedKalmanFilter(state_dim=3, measurement_dim=3)
        ekf.initialize(np.zeros(3))
        ekf.H = np.eye(3)
        measurement = np.array([1.0, 2.0, 3.0])
        ekf.correct(measurement)
        state = ekf.get_state()
        # 验证状态向测量值方向收敛 (EKF逐步更新)
        # 检查状态不再是无穷大或NaN
        self.assertTrue(np.all(np.isfinite(state)))
        # 检查状态有所更新 (不等于初始零状态)
        self.assertFalse(np.allclose(state, np.zeros(3)))

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
        imu = IMUSensor(sensor_type=IMUSensorType.BMI088, sensor_id="imu_test")
        fusion = ComplementaryFilter(alpha=0.96)
        imu.open()

        for _ in range(50):
            data = imu.capture()
            fusion.update({'accel': data.accel, 'gyro': data.gyro}, dt=0.01)

        state = fusion.get_state()
        self.assertEqual(len(state), 3)
        # roll和pitch应该接近0 (静止状态)
        self.assertAlmostEqual(state[0], 0.0, places=1)
        self.assertAlmostEqual(state[1], 0.0, places=1)
        imu.close()

    def test_ft_sensor_ekf(self):
        """测试力觉传感器EKF"""
        ft_sensor = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS, sensor_id="ft_test")
        ekf = ExtendedKalmanFilter(state_dim=6, measurement_dim=6)
        ekf.initialize(np.zeros(6))
        ekf.H = np.eye(6)
        ft_sensor.open()

        for _ in range(10):
            data = ft_sensor.capture()
            ekf.update({'ft': data.to_vector()}, dt=0.01)

        state = ekf.get_state()
        self.assertEqual(len(state), 6)
        ft_sensor.close()

    def test_tactile_imu_fusion(self):
        """测试触觉-IMU融合概念"""
        tactile = TactileArray(array_size=(8, 8), sensor_id="tactile_test")
        imu = IMUSensor(sensor_type=IMUSensorType.BMI088, sensor_id="imu_test")

        tactile.open()
        imu.open()

        # 采集数据
        t_frame = tactile.capture()
        imu_frame = imu.capture()

        # 验证数据
        self.assertEqual(t_frame.pressure_map.shape, (8, 8))
        self.assertEqual(imu_frame.accel.shape, (3,))

        tactile.close()
        imu.close()


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

    def test_multi_sensor_fusion_weighted_average(self):
        """测试多传感器加权平均"""
        fusion = MultiSensorFusion()

        # 添加多个不同权重的滤波器
        for i in range(3):
            cf = ComplementaryFilter(alpha=0.96)
            weight = 1.0 / (i + 1)
            fusion.add_fusion_method(f"sensor_{i}", cf, weight=weight)

        self.assertEqual(len(fusion.fusion_methods), 3)

        # 更新所有传感器
        for i in range(3):
            fusion.update({f"sensor_{i}": {
                'accel': np.array([0, 0, -9.81]),
                'gyro': np.array([0.0, 0.0, 0.1])
            }}, dt=0.01)

        fused = fusion.get_fused_state()
        self.assertIsInstance(fused, np.ndarray)


class TestFusionEdgeCases(unittest.TestCase):
    """融合边界情况测试"""

    def test_missing_sensor_data(self):
        """测试缺失传感器数据"""
        fusion = MultiSensorFusion()
        cf = ComplementaryFilter(alpha=0.96)
        fusion.add_fusion_method("imu1", cf)

        # 只提供加速度
        result = fusion.update({"imu1": {'accel': np.array([0, 0, -9.81])}}, dt=0.01)
        self.assertIsNotNone(result)

    def test_zero_dt(self):
        """测试零时间步长"""
        fusion = ComplementaryFilter(alpha=0.96)
        result = fusion.update({'accel': np.array([0, 0, -9.81]), 'gyro': np.array([0, 0, 0.1])}, dt=0.0)
        self.assertEqual(len(result), 3)

    def test_large_gyro_input(self):
        """测试大角速度输入"""
        fusion = ComplementaryFilter(alpha=0.5)
        for _ in range(10):
            fusion.update({'accel': np.array([0, 0, -9.81]), 'gyro': np.array([10.0, 10.0, 10.0])}, dt=0.01)
        state = fusion.get_state()
        self.assertTrue(np.all(np.abs(state) < 100))  # 不应发散


if __name__ == '__main__':
    unittest.main(verbosity=2)
