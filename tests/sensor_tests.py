"""
传感器模块测试用例
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensors.tactile import (
    TactileType, TactileData, TactileSensor,
    PressureSensor, TaxelArray, PiezoelectricSensor, TactileArray
)
from sensors.force import (
    ForceSensorType, ForceData, ForceSensor,
    SixAxisFTSensor, SingleAxisForceSensor, ForceSensorArray
)
from sensors.imu import (
    IMUModel, IMUData, IMUSensor,
    BMI088, MPU9250, IMUArray
)


class TestTactileSensors(unittest.TestCase):
    """触觉传感器测试"""

    def test_pressure_sensor_read(self):
        """测试压力传感器读取"""
        sensor = PressureSensor("p1", {"sensitivity": 0.01, "offset": 1000.0})
        data = sensor.read()
        self.assertIsInstance(data, TactileData)
        self.assertEqual(data.sensor_id, "p1")
        self.assertEqual(data.tactile_type, TactileType.PRESSURE)
        self.assertIsNotNone(data.pressure)
        self.assertGreaterEqual(data.pressure, 0)

    def test_pressure_to_vector(self):
        """测试压力数据转向量"""
        sensor = PressureSensor("p1")
        data = sensor.read()
        vec = data.to_vector()
        self.assertIsInstance(vec, np.ndarray)
        self.assertGreater(len(vec), 0)

    def test_taxel_array_read(self):
        """测试触感阵列读取"""
        sensor = TaxelArray("tax1", rows=8, cols=8)
        data = sensor.read()
        self.assertIsInstance(data, TactileData)
        self.assertIsNotNone(data.taxel_matrix)
        self.assertEqual(data.taxel_matrix.shape, (8, 8))

    def test_taxel_contact_detection(self):
        """测试接触检测"""
        sensor = TaxelArray("tax1", rows=16, cols=16)
        sensor.read()
        centroid = sensor.detect_contact_centroid()
        # 无接触时返回None
        self.assertIsNone(centroid)  # 模拟随机，可能无接触

    def test_contact_state(self):
        """测试接触状态判断"""
        sensor = PressureSensor("p1", {"offset": 2000.0})
        data = sensor.read()
        is_contact = data.get_contact_state(threshold=1000.0)
        self.assertIsInstance(is_contact, bool)

    def test_tactile_array_read_all(self):
        """测试触觉阵列读取所有传感器"""
        arr = TactileArray()
        arr.add_sensor(PressureSensor("p1"))
        arr.add_sensor(TaxelArray("tax1", rows=4, cols=4))
        
        all_data = arr.read_all()
        self.assertEqual(len(all_data), 2)

    def test_tactile_fusion_data(self):
        """测试触觉融合数据"""
        arr = TactileArray()
        arr.add_sensor(PressureSensor("p1"))
        arr.add_sensor(TaxelArray("tax1", rows=4, cols=4))
        
        fusion = arr.get_fusion_data()
        self.assertIsInstance(fusion, np.ndarray)
        self.assertGreater(len(fusion), 0)


class TestForceSensors(unittest.TestCase):
    """力觉传感器测试"""

    def test_six_axis_ft_read(self):
        """测试六维力传感器"""
        sensor = SixAxisFTSensor("ft1", model="mini40")
        data = sensor.read()
        self.assertIsInstance(data, ForceData)
        self.assertIsNotNone(data.wrench)
        self.assertEqual(len(data.wrench), 6)

    def test_six_axis_ft_components(self):
        """测试六维力分量"""
        sensor = SixAxisFTSensor("ft1")
        data = sensor.read()
        self.assertIsNotNone(data.force)
        self.assertEqual(len(data.force), 3)
        self.assertIsNotNone(data.torque)
        self.assertEqual(len(data.torque), 3)

    def test_force_magnitude(self):
        """测试力的大小计算"""
        sensor = SixAxisFTSensor("ft1")
        data = sensor.read()
        mag = data.get_magnitude()
        self.assertIsInstance(mag, float)
        self.assertGreaterEqual(mag, 0)

    def test_torque_magnitude(self):
        """测试力矩大小计算"""
        sensor = SixAxisFTSensor("ft1")
        data = sensor.read()
        mag = data.get_torque_magnitude()
        self.assertIsInstance(mag, float)
        self.assertGreaterEqual(mag, 0)

    def test_safety_check(self):
        """测试安全检测"""
        sensor = SixAxisFTSensor("ft1")
        data = sensor.read()
        safe = data.is_safe(force_threshold=100.0, torque_threshold=20.0)
        self.assertIsInstance(safe, bool)

    def test_single_axis_force(self):
        """测试单轴力传感器"""
        sensor = SingleAxisForceSensor("f1", axis="z", range_n=100.0)
        data = sensor.read()
        self.assertIsInstance(data, ForceData)
        self.assertEqual(data.sensor_type, ForceSensorType.SINGLE_AXIS)

    def test_bias_calibration(self):
        """测试偏置校准"""
        sensor = SixAxisFTSensor("ft1")
        initial = sensor.read()
        sensor.set_bias(initial)
        self.assertTrue(sensor.is_calibrated)

    def test_force_sensor_array(self):
        """测试力觉传感器阵列"""
        arr = ForceSensorArray()
        arr.add_sensor(SixAxisFTSensor("ft1"))
        arr.add_sensor(SingleAxisForceSensor("f1"))

        states = arr.read_all()
        self.assertEqual(len(states), 2)

    def test_net_wrench(self):
        """测试合成六维力"""
        arr = ForceSensorArray()
        arr.add_sensor(SixAxisFTSensor("ft1"))
        arr.add_sensor(SixAxisFTSensor("ft2"))

        net = arr.get_net_wrench()
        self.assertEqual(len(net), 6)

    def test_contact_detection(self):
        """测试接触检测"""
        arr = ForceSensorArray()
        arr.add_sensor(SixAxisFTSensor("ft1"))
        arr.read_all()
        contact = arr.detect_contact(threshold=1.0)
        self.assertIsInstance(contact, bool)


class TestIMUSensors(unittest.TestCase):
    """IMU传感器测试"""

    def test_bmi088_read(self):
        """测试BMI088读取"""
        sensor = BMI088("imu1")
        data = sensor.read()
        self.assertIsInstance(data, IMUData)
        self.assertEqual(data.sensor_id, "imu1")
        self.assertEqual(data.model, IMUModel.BMI088)
        self.assertEqual(len(data.acceleration), 3)
        self.assertEqual(len(data.angular_velocity), 3)

    def test_imu_vector(self):
        """测试IMU数据转向量"""
        sensor = BMI088("imu1")
        data = sensor.read()
        vec = data.to_vector()
        self.assertEqual(len(vec), 6)  # accel + gyro

    def test_imu_pose_change(self):
        """测试IMU姿态变化计算"""
        sensor = BMI088("imu1")
        data = sensor.read()
        change = data.get_imu_pose_change(dt=0.01)
        self.assertIn("delta_angle_x", change)
        self.assertIn("delta_vel_x", change)

    def test_mpu9250_read(self):
        """测试MPU9250读取"""
        sensor = MPU9250("imu2")
        data = sensor.read()
        self.assertEqual(data.model, IMUModel.MPU9250)
        self.assertIsNotNone(data.magnetic_field)
        self.assertEqual(len(data.magnetic_field), 3)

    def test_mpu9250_euler(self):
        """测试MPU9250欧拉角"""
        sensor = MPU9250("imu2")
        data = sensor.read()
        self.assertIsNotNone(data.euler)
        self.assertEqual(len(data.euler), 3)

    def test_imu_array(self):
        """测试IMU阵列"""
        arr = IMUArray()
        arr.add_sensor(BMI088("imu1"))
        arr.add_sensor(MPU9250("imu2"))

        all_data = arr.read_all()
        self.assertEqual(len(all_data), 2)

    def test_imu_fusion_data(self):
        """测试IMU融合数据"""
        arr = IMUArray()
        arr.add_sensor(BMI088("imu1"))
        arr.add_sensor(MPU9250("imu2"))

        fusion = arr.get_fusion_data()
        # 2 * 6 (accel+gyro for BMI088) + 2 * 9 (for MPU9250 with mag)
        self.assertGreater(len(fusion), 0)

    def test_imu_heading(self):
        """测试航向角计算"""
        arr = IMUArray()
        arr.add_sensor(BMI088("imu1"))
        arr.read_all()
        heading = arr.compute_heading()
        self.assertIsInstance(heading, float)


class TestSensorIntegration(unittest.TestCase):
    """传感器集成测试"""

    def test_all_sensor_types_read(self):
        """测试所有传感器类型都能正常读取"""
        sensors = [
            PressureSensor("p1"),
            TaxelArray("tax1", rows=4, cols=4),
            PiezoelectricSensor("piezo1"),
            SixAxisFTSensor("ft1"),
            SingleAxisForceSensor("f1"),
            BMI088("imu1"),
            MPU9250("imu2"),
        ]

        for sensor in sensors:
            data = sensor.read()
            self.assertIsNotNone(data.timestamp)

    def test_sampling_stability(self):
        """测试采样稳定性"""
        sensor = PressureSensor("p1", {"noise_std": 1.0})
        readings = [sensor.read().pressure for _ in range(100)]
        std = np.std(readings)
        self.assertLess(std, 10.0)  # 标准差应在合理范围


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
