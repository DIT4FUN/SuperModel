"""
传感器模块单元测试
测试触觉、力觉、IMU传感器的功能
"""

import unittest
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensors.tactile import (
    TactileSensor, TactileData, PressureSensor, TaxelArray,
    PiezoelectricSensor, TactileArray
)
from sensors.force import (
    ForceSensor, ForceData, SixAxisFTSensor, SingleAxisForceSensor,
    ForceSensorArray
)
from sensors.imu import (
    IMUSensor, IMUData, BMI088, MPU9250, IMUArray,
    quaternion_to_euler, euler_to_quaternion, quaternion_multiply, normalize_quaternion
)


class TestTactileData(unittest.TestCase):
    """测试 TactileData 数据类"""

    def test_tactile_data_creation(self):
        """测试创建 TactileData"""
        data = TactileData(
            sensor_id="test",
            timestamp=1.0,
            pressure=100.0
        )
        self.assertEqual(data.sensor_id, "test")
        self.assertEqual(data.pressure, 100.0)

    def test_to_vector(self):
        """测试特征向量转换"""
        data = TactileData(
            sensor_id="test",
            timestamp=1.0,
            pressure=500.0
        )
        vec = data.to_vector()
        self.assertIsInstance(vec, np.ndarray)
        self.assertGreater(len(vec), 0)
        # 检查归一化范围
        self.assertTrue(np.all(vec >= -1.5))  # 允许一点超限
        self.assertTrue(np.all(vec <= 1.5))

    def test_contact_center(self):
        """测试接触中心计算"""
        data = TactileData(
            sensor_id="test",
            timestamp=1.0,
            taxel_data=np.array([[0, 0], [0, 255]])
        )
        center = data.get_contact_center()
        self.assertEqual(len(center), 2)

    def test_contact_area(self):
        """测试接触面积计算"""
        data = TactileData(
            sensor_id="test",
            timestamp=1.0,
            taxel_data=np.ones((10, 10)) * 100
        )
        area = data.get_contact_area(threshold=50)
        self.assertGreaterEqual(area, 0)
        self.assertLessEqual(area, 1)


class TestPressureSensor(unittest.TestCase):
    """测试压力传感器"""

    def setUp(self):
        self.sensor = PressureSensor("p1", max_pressure=1000.0)

    def test_creation(self):
        """测试传感器创建"""
        self.assertEqual(self.sensor.sensor_id, "p1")
        self.assertFalse(self.sensor._calibrated)

    def test_read(self):
        """测试读取数据"""
        data = self.sensor.read(0.0)
        self.assertIsInstance(data, TactileData)
        self.assertEqual(data.sensor_id, "p1")
        self.assertGreaterEqual(data.pressure, 0)

    def test_set_pressure(self):
        """测试设置压力"""
        self.sensor.set_pressure(500.0)
        data = self.sensor.read(0.0)
        self.assertGreater(data.pressure, 400)  # 有噪声允许误差

    def test_calibrate(self):
        """测试校准"""
        ref_data = np.array([100, 100, 100, 100])
        result = self.sensor.calibrate(ref_data)
        self.assertTrue(result)
        self.assertTrue(self.sensor._calibrated)


class TestTaxelArray(unittest.TestCase):
    """测试触感阵列"""

    def setUp(self):
        self.sensor = TaxelArray("taxel1", rows=4, cols=4)

    def test_creation(self):
        """测试创建"""
        self.assertEqual(self.sensor.rows, 4)
        self.assertEqual(self.sensor.cols, 16)

    def test_read(self):
        """测试读取"""
        data = self.sensor.read(0.0)
        self.assertIsInstance(data, TactileData)
        self.assertIsNotNone(data.taxel_data)

    def test_apply_contact(self):
        """测试施加接触"""
        self.sensor.apply_contact(0.5, 0.5, 0.3, 500.0)
        data = self.sensor.read(0.0)
        self.assertGreater(data.pressure, 0)


class TestPiezoelectricSensor(unittest.TestCase):
    """测试压电传感器"""

    def setUp(self):
        self.sensor = PiezoelectricSensor("piezo1")

    def test_creation(self):
        """测试创建"""
        self.assertEqual(self.sensor.sensor_id, "piezo1")

    def test_read(self):
        """测试读取"""
        data = self.sensor.read(0.0)
        self.assertIsInstance(data, TactileData)

    def test_add_vibration(self):
        """测试添加振动"""
        self.sensor.add_vibration(100.0, 50.0)
        data = self.sensor.read(0.0)
        self.assertGreater(data.vibration, 0)


class TestTactileArray(unittest.TestCase):
    """测试触觉传感器阵列"""

    def setUp(self):
        self.array = TactileArray()
        self.array.add_sensor(PressureSensor("p1"))
        self.array.add_sensor(PiezoelectricSensor("piezo1"))

    def test_add_sensor(self):
        """测试添加传感器"""
        self.assertEqual(len(self.array), 2)

    def test_remove_sensor(self):
        """测试移除传感器"""
        self.array.remove_sensor("p1")
        self.assertEqual(len(self.array), 1)

    def test_read_all(self):
        """测试读取所有"""
        data = self.array.read_all(0.0)
        self.assertEqual(len(data), 2)

    def test_fusion_data(self):
        """测试融合数据"""
        vec = self.array.get_fusion_data(0.0)
        self.assertIsInstance(vec, np.ndarray)
        self.assertGreater(len(vec), 0)

    def test_detect_collision(self):
        """测试碰撞检测"""
        # 无碰撞
        self.assertFalse(self.array.detect_collision(threshold=0.001))


class TestForceData(unittest.TestCase):
    """测试力觉数据"""

    def test_creation(self):
        """测试创建"""
        wrench = np.array([10, 20, 30, 0.5, 0.6, 0.7])
        data = ForceData(
            sensor_id="ft1",
            timestamp=1.0,
            wrench=wrench
        )
        self.assertEqual(data.sensor_id, "ft1")
        np.testing.assert_array_equal(data.wrench, wrench)

    def test_to_vector(self):
        """测试特征向量"""
        wrench = np.array([10, 20, 30, 0.5, 0.6, 0.7])
        data = ForceData(sensor_id="ft1", timestamp=1.0, wrench=wrench)
        vec = data.to_vector()
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(len(vec), 7)  # 6维力 + 温度

    def test_force_magnitude(self):
        """测试力大小计算"""
        data = ForceData(sensor_id="ft1", timestamp=1.0, wrench=np.array([3, 4, 0, 0, 0, 0]))
        self.assertAlmostEqual(data.get_force_magnitude(), 5.0)


class TestSixAxisFTSensor(unittest.TestCase):
    """测试六维力传感器"""

    def setUp(self):
        self.sensor = SixAxisFTSensor("ft1", model="mini40")

    def test_creation(self):
        """测试创建"""
        self.assertEqual(self.sensor.sensor_id, "ft1")
        self.assertEqual(self.sensor.model, "mini40")

    def test_read(self):
        """测试读取"""
        data = self.sensor.read(0.0)
        self.assertIsInstance(data, ForceData)
        self.assertEqual(len(data.wrench), 6)

    def test_set_bias(self):
        """测试设置零偏"""
        current = self.sensor.read(0.0)
        self.sensor.set_bias(current)
        self.assertTrue(self.sensor._is_bias_set)

    def test_tcp_wrench(self):
        """测试TCP力矩计算"""
        self.sensor.set_wrench([10, 0, 0, 0, 0, 0])  # Fx=10N
        tcp_offset = np.array([0.1, 0, 0])  # 10cm偏移
        tcp_wrench = self.sensor.compute_tcp_wrench(tcp_offset)
        # Mx应该被补偿: Mx' = Mx - Fz*y + Fy*z = 0 - 0 + 0 = 0
        self.assertEqual(tcp_wrench[3], 0)


class TestSingleAxisForceSensor(unittest.TestCase):
    """测试单轴力传感器"""

    def setUp(self):
        self.sensor = SingleAxisForceSensor("f1", axis="z", force_range=100.0)

    def test_read(self):
        """测试读取"""
        data = self.sensor.read(0.0)
        self.assertIsInstance(data, ForceData)
        # 单轴力应该只在Z方向
        self.assertEqual(data.wrench[0], 0)  # Fx
        self.assertEqual(data.wrench[1], 0)  # Fy


class TestForceSensorArray(unittest.TestCase):
    """测试力觉传感器阵列"""

    def setUp(self):
        self.array = ForceSensorArray()
        self.array.add_sensor(SixAxisFTSensor("ft1"))
        self.array.add_sensor(SingleAxisForceSensor("f1"))

    def test_read_all(self):
        """测试读取所有"""
        data = self.array.read_all(0.0)
        self.assertEqual(len(data), 2)

    def test_net_wrench(self):
        """测试合成力"""
        wrench = self.array.get_net_wrench(0.0)
        self.assertIsInstance(wrench, np.ndarray)
        self.assertEqual(len(wrench), 6)

    def test_safety_check(self):
        """测试安全检查"""
        safety = self.array.check_safety(force_threshold=100.0)
        self.assertIn('is_safe', safety)
        self.assertIn('sensor_readings', safety)


class TestIMUData(unittest.TestCase):
    """测试IMU数据"""

    def test_creation(self):
        """测试创建"""
        data = IMUData(
            sensor_id="imu1",
            timestamp=1.0,
            acceleration=np.array([0, 0, 9.81]),
            angular_velocity=np.array([0, 0, 0])
        )
        self.assertEqual(data.sensor_id, "imu1")

    def test_to_vector(self):
        """测试特征向量"""
        data = IMUData(
            sensor_id="imu1",
            timestamp=1.0,
            acceleration=np.array([0, 0, 9.81])
        )
        vec = data.to_vector()
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(len(vec), 12)  # 3 accel + 3 gyro + 3 mag + 3 euler

    def test_heading(self):
        """测试航向角"""
        data = IMUData(
            sensor_id="imu1",
            timestamp=1.0,
            euler=np.array([0, 0, np.pi/2])  # 90度
        )
        self.assertAlmostEqual(data.get_heading(), 90.0)


class TestQuaternionConversions(unittest.TestCase):
    """测试四元数转换"""

    def test_euler_to_quaternion(self):
        """测试欧拉角到四元数"""
        euler = np.array([0, 0, 0])
        q = euler_to_quaternion(euler)
        np.testing.assert_array_almost_equal(q, [1, 0, 0, 0])

    def test_quaternion_to_euler(self):
        """测试四元数到欧拉角"""
        q = np.array([1, 0, 0, 0])
        euler = quaternion_to_euler(q)
        np.testing.assert_array_almost_equal(euler, [0, 0, 0])

    def test_roundtrip(self):
        """测试往返转换"""
        euler_orig = np.array([0.5, 0.3, 0.8])
        q = euler_to_quaternion(euler_orig)
        euler_back = quaternion_to_euler(q)
        np.testing.assert_array_almost_equal(euler_orig, euler_back, decimal=5)

    def test_quaternion_multiply(self):
        """测试四元数乘法"""
        q1 = np.array([1, 0, 0, 0])
        q2 = np.array([1, 0, 0, 0])
        result = quaternion_multiply(q1, q2)
        np.testing.assert_array_almost_equal(result, [1, 0, 0, 0])

    def test_normalize(self):
        """测试四元数归一化"""
        q = np.array([2, 0, 0, 0])
        q_norm = normalize_quaternion(q)
        self.assertAlmostEqual(np.linalg.norm(q_norm), 1.0)


class TestBMI088(unittest.TestCase):
    """测试BMI088传感器"""

    def setUp(self):
        self.sensor = BMI088("bmi1")

    def test_creation(self):
        """测试创建"""
        self.assertEqual(self.sensor.sensor_id, "bmi1")

    def test_read(self):
        """测试读取"""
        data = self.sensor.read(0.0)
        self.assertIsInstance(data, IMUData)
        self.assertEqual(len(data.acceleration), 3)
        self.assertEqual(len(data.angular_velocity), 3)
        self.assertEqual(len(data.euler), 3)
        self.assertEqual(len(data.quaternion), 4)

    def test_update_orientation(self):
        """测试姿态更新"""
        gyro = np.array([0, 0, 0.1])  # 绕Z轴角速度
        self.sensor.update_orientation(gyro, dt=0.01)
        euler = self.sensor.get_euler()
        self.assertGreater(euler[2], 0)  # yaw应该增加

    def test_calibrate_gyro_bias(self):
        """测试陀螺仪校准"""
        self.sensor.calibrate_gyro_bias(samples=10)
        self.assertTrue(self.sensor._is_calibrated)


class TestMPU9250(unittest.TestCase):
    """测试MPU9250传感器"""

    def setUp(self):
        self.sensor = MPU9250("mpu1")

    def test_creation(self):
        """测试创建"""
        self.assertEqual(self.sensor.sensor_id, "mpu1")

    def test_read(self):
        """测试读取"""
        data = self.sensor.read(0.0)
        self.assertIsInstance(data, IMUData)


class TestIMUArray(unittest.TestCase):
    """测试IMU阵列"""

    def setUp(self):
        self.array = IMUArray()
        self.array.add_sensor(BMI088("imu1"))
        self.array.add_sensor(MPU9250("imu2"))

    def test_read_all(self):
        """测试读取所有"""
        data = self.array.read_all(0.0)
        self.assertEqual(len(data), 2)

    def test_fusion_data(self):
        """测试融合数据"""
        vec = self.array.get_fusion_data(0.0)
        self.assertIsInstance(vec, np.ndarray)
        self.assertGreater(len(vec), 0)

    def test_heading(self):
        """测试航向角"""
        heading = self.array.compute_heading(0.0)
        self.assertIsInstance(heading, float)


if __name__ == '__main__':
    unittest.main(verbosity=2)
