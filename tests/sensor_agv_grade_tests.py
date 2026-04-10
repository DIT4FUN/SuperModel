"""
AGV五级传感器融合集成测试
=========================

验证不同AGV等级(S/M/L/XL/XXL)下传感器系统的正确配置和融合性能
测试触觉、力觉、IMU在各级别规格下的行为

AGV五级规格:
- S:  小型AGV (30kg负载, MPU6050 IMU, 100Hz)
- M:  中型AGV (100kg负载, BMI088 IMU, 200Hz, 16×16触觉)
- L:  大型AGV (300kg负载, ADIS16470 IMU, 500Hz, 24×24触觉)
- XL: 超大型AGV (600kg负载, ADIS16577 IMU, 1000Hz, 32×32触觉)
- XXL: 重型AGV (1200kg负载, 4×ADIS IMU, 2000Hz, 48×48触觉)
"""

import unittest
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sensors.tactile import (
    TactileArray, TactileFrame, TactileSensorType,
    VirtualTactileSensor, get_tactile_spec, AGV_TACTILE_GRADES
)
from src.sensors.force import (
    ForceTorqueSensor, Wrench, ForceSensorType,
    VirtualForceSensor, get_force_spec, AGV_FORCE_GRADES
)
from src.sensors.imu import (
    IMUSensor, IMUFrame, IMUSensorType,
    VirtualIMUSensor, get_imu_spec, AGV_IMU_GRADES
)
from src.sensors.manager import SensorManager


class TestAGVTactileGrades(unittest.TestCase):
    """测试AGV五级触觉规格"""

    GRADE_CONFIGS = {
        'S':  {'array': (8, 8),    'res': 12, 'range_kpa': (0, 500),   'freq_hz': 50,  'temp': False},
        'M':  {'array': (16, 16),  'res': 12, 'range_kpa': (0, 1000),  'freq_hz': 100, 'temp': True},
        'L':  {'array': (24, 24),  'res': 14, 'range_kpa': (0, 2000),  'freq_hz': 200, 'temp': True},
        'XL': {'array': (32, 32),  'res': 14, 'range_kpa': (0, 5000), 'freq_hz': 500, 'temp': True},
        'XXL': {'array': (48, 48), 'res': 16, 'range_kpa': (0, 10000), 'freq_hz': 1000, 'temp': True},
    }

    def test_all_grades_spec(self):
        """测试所有等级规格定义"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_tactile_spec(grade)
            self.assertIn('array', spec)
            self.assertIn('res', spec)
            self.assertIn('range_kpa', spec)
            self.assertIn('freq_hz', spec)
            self.assertIn('temp', spec)
            
            # 验证规格递增
            expected_size = self.GRADE_CONFIGS[grade]['array']
            self.assertEqual(spec['array'], expected_size,
                           f"Grade {grade} array mismatch")

    def test_grade_s_m(self):
        """测试S和M等级触觉配置"""
        for grade in ['S', 'M']:
            spec = get_tactile_spec(grade)
            # S/M等级: 小阵列, 低分辨率
            self.assertLessEqual(spec['array'][0], 16)
            self.assertLessEqual(spec['res'], 12)
            self.assertLessEqual(spec['freq_hz'], 100)

    def test_grade_l_xl(self):
        """测试L和XL等级触觉配置"""
        for grade in ['L', 'XL']:
            spec = get_tactile_spec(grade)
            # L/XL等级: 中等阵列, 中等分辨率
            self.assertGreaterEqual(spec['array'][0], 24)
            self.assertLessEqual(spec['res'], 14)
            self.assertLessEqual(spec['freq_hz'], 500)

    def test_grade_xxl(self):
        """测试XXL等级触觉配置"""
        spec = get_tactile_spec('XXL')
        # XXL等级: 最大阵列, 最高分辨率
        self.assertEqual(spec['array'], (48, 48))
        self.assertEqual(spec['res'], 16)
        self.assertEqual(spec['freq_hz'], 1000)
        self.assertTrue(spec['temp'])

    def test_grade_array_size_scaling(self):
        """测试阵列尺寸随等级缩放"""
        prev_size = 0
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_tactile_spec(grade)
            size = spec['array'][0]
            # 每个等级应该比上一等级大或相等
            self.assertGreaterEqual(size, prev_size)
            prev_size = size


class TestAGVForceGrades(unittest.TestCase):
    """测试AGV五级力觉规格"""

    def test_all_force_grades(self):
        """测试所有等级力觉规格"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_force_spec(grade)
            self.assertIn('axes', spec)
            self.assertIn('force_range', spec)
            self.assertIn('torque_range', spec)
            self.assertIn('resolution', spec)
            self.assertIn('sampling_hz', spec)
            
            # 验证规格递增
            if grade == 'S':
                self.assertEqual(spec['axes'], 3)  # S级只有3轴力传感器
            else:
                self.assertEqual(spec['axes'], 6)  # 其他等级6轴
            
            # 采样率递增
            prev_spec = get_force_spec('S') if grade != 'S' else None
            if prev_spec:
                self.assertGreaterEqual(spec['sampling_hz'], prev_spec['sampling_hz'])

    def test_force_range_scaling(self):
        """测试力觉量程随等级缩放"""
        prev_range = 0
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_force_spec(grade)
            self.assertGreaterEqual(spec['force_range'], prev_range)
            prev_range = spec['force_range']

    def test_force_resolution_scaling(self):
        """测试力觉分辨率随等级提高"""
        prev_res = 1.0  # 初始值较大
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_force_spec(grade)
            # 分辨率数值应该递减(越小越精确)
            self.assertLessEqual(spec['resolution'], prev_res)
            prev_res = spec['resolution']


class TestAGVIMUGrades(unittest.TestCase):
    """测试AGV五级IMU规格"""

    def test_all_imu_grades(self):
        """测试所有等级IMU规格"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_imu_spec(grade)
            self.assertIn('type', spec)
            self.assertIn('accel_range', spec)
            self.assertIn('gyro_range', spec)
            self.assertIn('sample_hz', spec)
            self.assertIn('noise_density', spec)
            
            # 采样率递增
            prev_spec = get_imu_spec('S') if grade != 'S' else None
            if prev_spec:
                self.assertGreaterEqual(spec['sample_hz'], prev_spec['sample_hz'])

    def test_imu_noise_scaling(self):
        """测试IMU噪声密度随等级降低(更精确)"""
        prev_noise = 1000  # 初始值较大
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_imu_spec(grade)
            self.assertLessEqual(spec['noise_density'], prev_noise)
            prev_noise = spec['noise_density']

    def test_imu_sample_rate_scaling(self):
        """测试IMU采样率随等级升高"""
        rates = []
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_imu_spec(grade)
            rates.append(spec['sample_hz'])
        
        # 采样率应该递增
        for i in range(1, len(rates)):
            self.assertGreaterEqual(rates[i], rates[i-1])


class TestAGVSensorManagerGrades(unittest.TestCase):
    """测试AGV五级传感器管理器配置"""

    def test_sensor_manager_creation(self):
        """测试传感器管理器创建"""
        manager = SensorManager()
        self.assertIsNotNone(manager)
        manager.close_all()

    def test_grade_spec_retrieval(self):
        """测试各等级规格检索"""
        # 触觉规格
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            tactile_spec = get_tactile_spec(grade)
            self.assertIsNotNone(tactile_spec)
            
        # 力觉规格
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            force_spec = get_force_spec(grade)
            self.assertIsNotNone(force_spec)
            
        # IMU规格
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            imu_spec = get_imu_spec(grade)
            self.assertIsNotNone(imu_spec)


class TestSensorLatencyBudget(unittest.TestCase):
    """测试传感器延迟预算(按AGV等级)"""

    LATENCY_BUDGETS = {
        'S':  {'tactile_ms': None, 'force_ms': None, 'imu_ms': 10.0, 'fusion_ms': 40.0},
        'M':  {'tactile_ms': 20.0, 'force_ms': 10.0, 'imu_ms': 5.0, 'fusion_ms': 15.0},
        'L':  {'tactile_ms': 10.0, 'force_ms': 5.0, 'imu_ms': 2.0, 'fusion_ms': 8.0},
        'XL': {'tactile_ms': 5.0, 'force_ms': 2.0, 'imu_ms': 1.0, 'fusion_ms': 4.0},
        'XXL': {'tactile_ms': 2.0, 'force_ms': 1.0, 'imu_ms': 0.5, 'fusion_ms': 2.0},
    }

    def test_tactile_latency_budget(self):
        """测试触觉延迟预算"""
        for grade, budget in self.LATENCY_BUDGETS.items():
            if budget['tactile_ms'] is None:
                continue  # S等级无触觉
            
            # 创建对应等级的触觉传感器
            tactile_spec = get_tactile_spec(grade)
            sensor = TactileArray(
                array_size=tactile_spec['array'],
                sensor_id=f"test_tactile_{grade}"
            )
            sensor.open()
            
            # 测量捕获延迟
            start = time.perf_counter()
            frame = sensor.capture()
            latency_ms = (time.perf_counter() - start) * 1000
            
            # 延迟应该小于预算
            self.assertLess(latency_ms, budget['tactile_ms'],
                           f"Grade {grade} tactile latency {latency_ms:.2f}ms exceeds budget {budget['tactile_ms']}ms")
            
            sensor.close()

    def test_imu_latency_budget(self):
        """测试IMU延迟预算"""
        for grade, budget in self.LATENCY_BUDGETS.items():
            imu_spec = get_imu_spec(grade)
            sensor = IMUSensor(
                sensor_type=IMUSensorType.VIRTUAL,
                sensor_id=f"test_imu_{grade}"
            )
            sensor.open()
            
            # 测量捕获延迟
            start = time.perf_counter()
            frame = sensor.capture()
            latency_ms = (time.perf_counter() - start) * 1000
            
            # 延迟应该小于预算
            self.assertLess(latency_ms, budget['imu_ms'],
                           f"Grade {grade} IMU latency {latency_ms:.2f}ms exceeds budget {budget['imu_ms']}ms")
            
            sensor.close()


class TestSensorFusionIntegration(unittest.TestCase):
    """测试传感器融合集成"""

    def test_tactile_force_imu_fusion(self):
        """测试触觉-力觉-IMU三传感器融合"""
        # 创建虚拟传感器
        tactile = VirtualTactileSensor((16, 16), "fusion_tactile")
        force = VirtualForceSensor("fusion_force")
        imu = VirtualIMUSensor("fusion_imu")
        
        tactile.open()
        force.open()
        imu.open()
        
        # 模拟接触场景
        tactile_frame = tactile.simulate_contact(
            contact_pos=(0.5, 0.5),
            contact_force=10.0
        )
        
        force_wrench = force.simulate_contact(
            force=(5.0, 3.0, -10.0),
            torque=(0.1, 0.2, 0.0)
        )
        
        imu_frame = imu.simulate_static(orientation=(0.0, 0.0, 0.0))
        
        # 验证数据有效性
        self.assertIsNotNone(tactile_frame.pressure_map)
        self.assertGreater(np.max(tactile_frame.pressure_map), 0)
        
        self.assertIsNotNone(force_wrench.force)
        self.assertEqual(force_wrench.force.shape, (3,))
        
        self.assertIsNotNone(imu_frame.accel)
        self.assertEqual(imu_frame.accel.shape, (3,))
        
        tactile.close()
        force.close()
        imu.close()

    def test_multi_modality_consistency(self):
        """测试多模态数据一致性"""
        # 当没有接触时,触觉和力觉应该一致
        tactile = VirtualTactileSensor((16, 16))
        force = VirtualForceSensor()
        
        tactile.open()
        force.open()
        
        # 模拟无接触 - 使用极小的接触力
        tactile_frame = tactile.simulate_contact(
            contact_pos=(0.5, 0.5),
            contact_force=0.1  # 极小接触力
        )
        
        force_wrench = force.simulate_contact(
            force=(0.1, 0.1, 0.1),
            torque=(0.01, 0.01, 0.01)
        )
        
        # 极小接触时压力应较小
        self.assertLess(np.max(tactile_frame.pressure_map), 0.5)
        
        # 极小接触时力应较小
        self.assertLess(np.linalg.norm(force_wrench.force), 1.0)
        
        tactile.close()
        force.close()


class TestSensorTimeSynchronization(unittest.TestCase):
    """测试传感器时间同步"""

    SYNC_TOLERANCES = {
        'S':  10.0,   # ms
        'M':  5.0,    # ms
        'L':  2.0,    # ms
        'XL': 1.0,    # ms
        'XXL': 0.5,   # ms
    }

    def test_sensor_timestamp_sync(self):
        """测试传感器时间戳同步"""
        tactile = VirtualTactileSensor((16, 16))
        force = VirtualForceSensor()
        imu = VirtualIMUSensor()
        
        tactile.open()
        force.open()
        imu.open()
        
        # 同时采集
        t_start = time.perf_counter()
        
        # VirtualTactileSensor 使用 simulate_contact, 其他用 capture
        t_frame = tactile.simulate_contact(contact_pos=(0.5, 0.5), contact_force=5.0)
        f_wrench = force.simulate_contact(force=(0.0, 0.0, -10.0))
        i_frame = imu.simulate_static(orientation=(0.0, 0.0, 0.0))
        
        t_end = time.perf_counter()
        
        # 计算时间差 (使用仿真时间戳)
        tf_diff = abs(t_frame.timestamp - f_wrench.timestamp) * 1000
        ti_diff = abs(t_frame.timestamp - i_frame.timestamp) * 1000
        fi_diff = abs(f_wrench.timestamp - i_frame.timestamp) * 1000
        
        max_diff = max(tf_diff, ti_diff, fi_diff)
        
        # S等级容忍10ms差异
        tolerance = self.SYNC_TOLERANCES['S']
        self.assertLess(max_diff, tolerance,
                        f"Sensor timestamp diff {max_diff:.2f}ms exceeds tolerance {tolerance}ms")
        
        tactile.close()
        force.close()
        imu.close()


if __name__ == '__main__':
    unittest.main()
