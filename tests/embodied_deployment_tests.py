"""
SuperModel 具身智能部署验证测试
================================

验证实机部署流程的完整性:
- 传感器初始化与连接
- 五级校准流程
- 融合控制五级配置
- 运行状态检查
- 端到端pipeline

v2.35.0
"""

import unittest
import numpy as np
import time
import sys
import os

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
    VirtualIMUSensor, PoseEstimator, get_imu_spec, AGV_IMU_GRADES
)
from src.control.tactile_control import (
    TactileServoController, TactileServoParams
)
from src.control.force_control import (
    ForceController, ForceControlParams
)
from src.control.imu_control import (
    AttitudeStabilizer, IMUControlParams
)
from src.control.sensor_fusion_control import (
    SensorFusionController, FusionControlConfig,
    FusionControlGrade, AGV_FUSION_CONTROL_GRADES, get_fusion_control_spec
)
from src.control.bias_compensation import (
    MultiSensorBiasCompensator,
    IMUBiasEstimator, ForceBiasEstimator, TactileBiasEstimator,
    get_bias_compensation_spec
)
from src.fusion.cross_modal_fusion import (
    CrossModalFusion, FusionConfig, MultimodalInput
)


GRADES = ['S', 'M', 'L', 'XL', 'XXL']


# ============================================================================
# 五级规格验证
# ============================================================================

class TestAGVGradeSpecifications(unittest.TestCase):
    """验证AGV五级规格表完整性"""

    def test_tactile_grade_specs_complete(self):
        """触觉五级规格完整性"""
        for grade in GRADES:
            spec = get_tactile_spec(grade)
            self.assertIn('array', spec)
            self.assertIn('res', spec)
            self.assertIn('range_kpa', spec)
            self.assertIn('freq_hz', spec)
            self.assertIn('temp', spec)
            
            # 规格递增验证
            sizes = {g: get_tactile_spec(g)['array'][0] for g in GRADES}
            self.assertGreater(sizes['XXL'], sizes['XL'])
            self.assertGreater(sizes['XL'], sizes['L'])
            self.assertGreater(sizes['L'], sizes['M'])
            self.assertGreater(sizes['M'], sizes['S'])

    def test_force_grade_specs_complete(self):
        """力觉五级规格完整性"""
        for grade in GRADES:
            spec = get_force_spec(grade)
            self.assertIn('axes', spec)
            self.assertIn('force_range', spec)
            self.assertIn('torque_range', spec)
            self.assertIn('resolution', spec)
            self.assertIn('sampling_hz', spec)
            
            # 规格递增验证
            ranges = {g: get_force_spec(g)['force_range'] for g in GRADES}
            self.assertGreater(ranges['XXL'], ranges['XL'])
            self.assertGreater(ranges['XL'], ranges['L'])
            self.assertGreater(ranges['L'], ranges['M'])

    def test_imu_grade_specs_complete(self):
        """IMU五级规格完整性"""
        for grade in GRADES:
            spec = get_imu_spec(grade)
            self.assertIn('type', spec)
            self.assertIn('accel_range', spec)
            self.assertIn('gyro_range', spec)
            self.assertIn('sample_hz', spec)
            self.assertIn('noise_density', spec)
            
            # 噪声递减 (精度递增)
            noises = {g: get_imu_spec(g)['noise_density'] for g in GRADES}
            self.assertLess(noises['XXL'], noises['XL'])
            self.assertLess(noises['XL'], noises['L'])
            self.assertLess(noises['L'], noises['M'])

    def test_fusion_control_grade_specs(self):
        """融合控制五级规格完整性"""
        for grade in GRADES:
            spec = get_fusion_control_spec(grade)
            self.assertIn('freq', spec)
            self.assertIn('algorithm', spec)
            self.assertIn('imu_rate', spec)
            self.assertIn('force_rate', spec)
            self.assertIn('tactile_rate', spec)
            self.assertIn('latency_ms', spec)


# ============================================================================
# 传感器初始化
# ============================================================================

class TestSensorInitialization(unittest.TestCase):
    """测试传感器初始化流程"""

    def test_tactile_initialization_all_grades(self):
        """五级触觉传感器初始化"""
        sizes = {'S': (8, 8), 'M': (16, 16), 'L': (24, 24), 'XL': (32, 32), 'XXL': (48, 48)}
        types = {
            'S': TactileSensorType.RESISTIVE,
            'M': TactileSensorType.CAPACITIVE,
            'L': TactileSensorType.CAPACITIVE,
            'XL': TactileSensorType.OPTICAL,
            'XXL': TactileSensorType.OPTICAL,
        }
        
        for grade in GRADES:
            sensor = TactileArray(
                array_size=sizes[grade],
                sensor_type=types[grade],
                sensor_id=f"tactile_{grade}"
            )
            self.assertTrue(sensor.open())
            self.assertTrue(sensor._is_opened)
            
            frame = sensor.capture()
            self.assertIsInstance(frame, TactileFrame)
            self.assertEqual(frame.pressure_map.shape, sizes[grade])
            self.assertEqual(frame.sensor_id, f"tactile_{grade}")
            
            sensor.close()

    def test_force_initialization_all_grades(self):
        """五级力觉传感器初始化"""
        types = {
            'S': ForceSensorType.THREE_AXIS,
            'M': ForceSensorType.SIX_AXIS,
            'L': ForceSensorType.SIX_AXIS,
            'XL': ForceSensorType.SIX_AXIS,
            'XXL': ForceSensorType.SIX_AXIS,
        }
        
        for grade in GRADES:
            sensor = ForceTorqueSensor(
                sensor_type=types[grade],
                sensor_id=f"force_{grade}"
            )
            self.assertTrue(sensor.open())
            self.assertTrue(sensor._is_streaming)
            
            wrench = sensor.capture()
            self.assertIsInstance(wrench, Wrench)
            self.assertEqual(wrench.force.shape, (3,))
            self.assertEqual(wrench.torque.shape, (3,))
            
            sensor.close()

    def test_imu_initialization_all_grades(self):
        """五级IMU传感器初始化"""
        types = {
            'S': IMUSensorType.MPU6050,
            'M': IMUSensorType.BMI088,
            'L': IMUSensorType.BMI088,
            'XL': IMUSensorType.ADIS16470,
            'XXL': IMUSensorType.ADIS16470,
        }
        
        for grade in GRADES:
            sensor = IMUSensor(
                sensor_type=types[grade],
                sensor_id=f"imu_{grade}"
            )
            self.assertTrue(sensor.open())
            self.assertTrue(sensor._is_opened)
            
            frame = sensor.capture()
            self.assertIsInstance(frame, IMUFrame)
            self.assertEqual(frame.accel.shape, (3,))
            self.assertEqual(frame.gyro.shape, (3,))
            
            sensor.close()

    def test_virtual_sensors_all_modalities(self):
        """虚拟传感器全模态初始化"""
        tactile = VirtualTactileSensor((16, 16), "virtual_tactile")
        self.assertTrue(tactile.open())
        frame = tactile.simulate_contact((0.5, 0.5), 0.3, 10.0)
        self.assertIsInstance(frame, TactileFrame)
        tactile.close()

        force = VirtualForceSensor("virtual_force")
        self.assertTrue(force.open())
        wrench = force.simulate_contact((0, 0, -10), (0, 0, 0))
        self.assertIsInstance(wrench, Wrench)
        force.close()

        imu = VirtualIMUSensor("virtual_imu")
        self.assertTrue(imu.open())
        imu_frame = imu.simulate_static((0.0, 0.0, 0.0))
        self.assertIsInstance(imu_frame, IMUFrame)
        imu.close()


# ============================================================================
# 传感器校准
# ============================================================================

class TestSensorCalibration(unittest.TestCase):
    """测试传感器校准流程"""

    def test_imu_gyro_bias_calibration(self):
        """IMU陀螺仪偏置校准"""
        imu = IMUSensor(IMUSensorType.BMI088, "imu_cal")
        imu.open()
        
        initial_bias = imu.calibration.gyro_bias.copy()
        
        # 校准
        imu.calibrate_gyro_bias(num_samples=100)
        
        # 校准后偏置应接近零
        self.assertTrue(np.allclose(imu.calibration.gyro_bias, 0, atol=0.5))
        
        imu.close()

    def test_imu_accel_calibration(self):
        """IMU加速度计标定"""
        imu = IMUSensor(IMUSensorType.BMI088, "imu_accel_cal")
        imu.open()
        
        imu.calibrate_accel(known_orientation="level")
        
        # 检查比例因子
        self.assertTrue(np.allclose(imu.calibration.accel_scale, 1.0, atol=0.1))
        
        imu.close()

    def test_force_bias_calibration(self):
        """力觉零点校准"""
        ft = ForceTorqueSensor(ForceSensorType.SIX_AXIS, "ft_cal")
        ft.open()
        
        initial_bias = ft.calibration.bias.copy()
        
        ft.calibrate_bias(num_samples=50)
        
        # 偏置应该被更新
        self.assertFalse(np.allclose(ft.calibration.bias, initial_bias))
        
        ft.close()

    def test_tactile_calibration(self):
        """触觉传感器标定"""
        tactile = TactileArray((8, 8), TactileSensorType.RESISTIVE, "tactile_cal")
        tactile.open()
        
        # 模拟零点采集
        zero_frames = [tactile.capture() for _ in range(20)]
        zero_pressure = np.mean([f.pressure_map for f in zero_frames], axis=0)
        
        tactile.calibrate(zero_pressure=zero_pressure)
        
        self.assertIsNotNone(tactile.calibration.offset_map)
        self.assertEqual(tactile.calibration.offset_map.shape, (8, 8))
        
        tactile.close()

    def test_bias_compensation_system(self):
        """偏置补偿系统"""
        for grade in GRADES:
            # MultiSensorBiasCompensator 五级配置
            system = MultiSensorBiasCompensator(grade=grade)
            self.assertEqual(system.grade, grade)
            self.assertIsNotNone(system.config)
            
            # IMU估计器
            self.assertIsNotNone(system.imu_estimator)
            
            # 力觉估计器
            self.assertIsNotNone(system.force_estimator)
            
            # 初始化触觉
            system.initialize_tactile((16, 16))
            self.assertIsNotNone(system.tactile_estimator)

    def test_bias_compensation_spec_by_grade(self):
        """五级偏置补偿规格"""
        for grade in GRADES:
            spec = get_bias_compensation_spec(grade)
            self.assertIsInstance(spec, type(get_bias_compensation_spec('M')))
            # 验证基本字段
            self.assertTrue(hasattr(spec, 'enable_imu'))
            self.assertTrue(hasattr(spec, 'enable_force'))
            self.assertTrue(hasattr(spec, 'enable_tactile'))
            self.assertTrue(hasattr(spec, 'adaptation_rate'))
            self.assertEqual(spec.grade, grade)


# ============================================================================
# 融合控制五级配置
# ============================================================================

class TestFusionControlGrades(unittest.TestCase):
    """测试融合控制五级配置"""

    def test_fusion_config_all_grades(self):
        """融合控制五级配置"""
        for grade in GRADES:
            cfg = FusionControlConfig(grade=FusionControlGrade[grade])
            self.assertEqual(cfg.grade.value, grade)
            
            spec = get_fusion_control_spec(grade)
            self.assertEqual(self._get_ctrl_rate(grade), spec['freq'])

    def _get_ctrl_rate(self, grade):
        ctrl = SensorFusionController(grade=FusionControlGrade[grade])
        rate = ctrl.fusion_frequency
        ctrl.stop()
        return rate

    def test_fusion_controller_all_grades(self):
        """融合控制器五级实例化"""
        for grade in GRADES:
            ctrl = SensorFusionController(grade=FusionControlGrade[grade])
            
            self.assertEqual(ctrl.grade.value, grade)
            self.assertEqual(ctrl.fusion_frequency, get_fusion_control_spec(grade)['freq'])
            
            ctrl.stop()

    def test_fusion_update_with_all_sensors(self):
        """融合更新全传感器输入"""
        ctrl = SensorFusionController(grade=FusionControlGrade.M)
        
        for i in range(10):
            tactile_pressure = np.random.rand(16, 16).astype(np.float32)
            wrench_vec = np.random.randn(6).astype(np.float32)
            accel = np.random.randn(3).astype(np.float32)
            gyro = np.random.randn(3).astype(np.float32)
            
            result = ctrl.update(
                imu_accel=accel,
                imu_gyro=gyro,
                force_wrench=wrench_vec,
                tactile_pressure=tactile_pressure,
                dt=0.01
            )
            self.assertIsNotNone(result)
        
        ctrl.stop()


# ============================================================================
# 端到端具身pipeline
# ============================================================================

class TestEmbodiedPipeline(unittest.TestCase):
    """测试端到端具身控制pipeline"""

    def test_full_pipeline_single_step(self):
        """单步完整pipeline"""
        grade = 'M'
        
        # 传感器
        tactile = TactileArray((16, 16), TactileSensorType.CAPACITIVE, "t0")
        force = ForceTorqueSensor(ForceSensorType.SIX_AXIS, "f0")
        imu = IMUSensor(IMUSensorType.BMI088, "i0")
        
        # 融合控制
        fusion_ctrl = SensorFusionController(grade=FusionControlGrade.M)
        
        # 打开传感器
        tactile.open()
        force.open()
        imu.open()
        
        # 单步执行
        t_frame = tactile.capture()
        f_wrench = force.capture()
        i_frame = imu.capture()
        
        # 融合 (使用正确的数组参数接口)
        fusion_result = fusion_ctrl.update(
            imu_accel=i_frame.accel,
            imu_gyro=i_frame.gyro,
            force_wrench=f_wrench.to_vector(),
            tactile_pressure=t_frame.pressure_map,
            dt=0.01
        )
        
        self.assertIsNotNone(fusion_result)
        
        # 清理
        tactile.close()
        force.close()
        imu.close()
        fusion_ctrl.stop()

    def test_full_pipeline_multi_step(self):
        """多步完整pipeline (时间序列)"""
        tactile = TactileArray((16, 16), TactileSensorType.CAPACITIVE, "t0")
        force = ForceTorqueSensor(ForceSensorType.SIX_AXIS, "f0")
        imu = IMUSensor(IMUSensorType.BMI088, "i0")
        fusion_ctrl = SensorFusionController(grade=FusionControlGrade.M)
        
        tactile.open()
        force.open()
        imu.open()
        
        for step in range(50):
            t_frame = tactile.capture()
            f_wrench = force.capture()
            i_frame = imu.capture()
            
            result = fusion_ctrl.update(
                imu_accel=i_frame.accel,
                imu_gyro=i_frame.gyro,
                force_wrench=f_wrench.to_vector(),
                tactile_pressure=t_frame.pressure_map,
                dt=0.01
            )
            
            self.assertIsNotNone(result)
            
            # 验证帧ID递增
            self.assertEqual(t_frame.frame_id, step)
            self.assertEqual(f_wrench.frame_id, step)
            self.assertEqual(i_frame.frame_id, step)
        
        tactile.close()
        force.close()
        imu.close()
        fusion_ctrl.stop()

    def test_all_grades_full_pipeline(self):
        """所有AGV等级完整pipeline"""
        for grade in GRADES:
            with self.subTest(grade=grade):
                sizes = {'S': (8, 8), 'M': (16, 16), 'L': (24, 24), 'XL': (32, 32), 'XXL': (48, 48)}
                
                tactile = TactileArray(
                    array_size=sizes[grade],
                    sensor_id=f"t_{grade}"
                )
                force = ForceTorqueSensor(ForceSensorType.SIX_AXIS, f"f_{grade}")
                imu = IMUSensor(IMUSensorType.BMI088, f"i_{grade}")
                fusion_ctrl = SensorFusionController(grade=FusionControlGrade[grade])
                
                tactile.open()
                force.open()
                imu.open()
                
                # 5步pipeline
                for step in range(5):
                    t_frame = tactile.capture()
                    f_wrench = force.capture()
                    i_frame = imu.capture()
                    
                    result = fusion_ctrl.update(
                        imu_accel=i_frame.accel,
                        imu_gyro=i_frame.gyro,
                        force_wrench=f_wrench.to_vector(),
                        tactile_pressure=t_frame.pressure_map,
                        dt=0.01
                    )
                    self.assertIsNotNone(result)
                    self.assertEqual(t_frame.frame_id, step)
                
                tactile.close()
                force.close()
                imu.close()
                fusion_ctrl.stop()

    def test_pipeline_tactile_contact_detection(self):
        """Pipeline触觉接触检测"""
        # 使用虚拟触觉传感器模拟接触
        tactile = VirtualTactileSensor((16, 16), "t_contact")
        tactile.open()
        
        # 模拟接触
        frame = tactile.simulate_contact(
            contact_pos=(0.5, 0.5),
            contact_radius=0.2,
            contact_force=15.0
        )
        
        # 验证触觉帧
        self.assertIsInstance(frame, TactileFrame)
        self.assertEqual(frame.pressure_map.shape, (16, 16))
        self.assertGreater(frame.pressure_map.max(), 0)
        
        # 抓取质量评估
        grip_quality = tactile.simulate_slip_detection(
            normal_force=15.0,
            friction_coeff=0.3,
            velocity=(0.01, 0.0)
        )
        self.assertIn('slip_probability', grip_quality)
        
        tactile.close()

    def test_pipeline_force_contact_detection(self):
        """Pipeline力觉接触检测"""
        force = VirtualForceSensor("f_contact")
        force.open()
        
        # 模拟接触力
        wrench = force.simulate_contact(
            force=(0, 0, -10),
            torque=(0.1, 0.1, 0)
        )
        
        # 验证力值合理
        self.assertGreater(wrench.magnitude, 0)
        self.assertLess(wrench.force[2], 0)  # Z轴负向 (接触力向下)
        
        force.close()

    def test_pipeline_imu_pose_estimation(self):
        """Pipeline IMU姿态估计"""
        imu = IMUSensor(IMUSensorType.BMI088, "i_pose")
        imu.open()
        
        pose_estimator = PoseEstimator(algorithm='madgwick', sample_rate=200.0)
        
        for _ in range(50):
            frame = imu.capture()
            pose = pose_estimator.update(frame.accel, frame.gyro)
            
            self.assertIsNotNone(pose)
            self.assertEqual(pose.orientation.shape, (4,))
        
        euler = pose_estimator.get_euler()
        self.assertEqual(euler.shape, (3,))
        
        imu.close()


# ============================================================================
# 偏置补偿集成
# ============================================================================

class TestBiasCompensationIntegration(unittest.TestCase):
    """测试偏置补偿与pipeline集成"""

    def test_imu_bias_compensation_in_loop(self):
        """IMU偏置补偿控制循环"""
        imu = IMUSensor(IMUSensorType.BMI088, "imu_bias")
        imu.open()
        imu.calibrate_gyro_bias(num_samples=50)
        
        config = get_bias_compensation_spec('M')
        estimator = IMUBiasEstimator(config)
        stabilizer = AttitudeStabilizer(IMUControlParams(grade='M'))
        
        for _ in range(30):
            frame = imu.capture()
            
            # 更新估计器
            estimator.update(frame.accel, frame.gyro, dt=0.01)
            
            # 补偿
            c_accel, c_gyro = estimator.compensate(frame.accel, frame.gyro)
            
            # 验证形状
            self.assertEqual(c_accel.shape, (3,))
            self.assertEqual(c_gyro.shape, (3,))
            
            # 更新稳定器
            compensated_frame = IMUFrame(
                accel=c_accel, gyro=c_gyro, mag=frame.mag,
                timestamp=frame.timestamp, frame_id=frame.frame_id,
                sensor_id=frame.sensor_id
            )
            stabilizer.update(compensated_frame)
        
        imu.close()

    def test_force_bias_compensation(self):
        """力觉偏置补偿"""
        ft = ForceTorqueSensor(ForceSensorType.SIX_AXIS, "ft_bias")
        ft.open()
        ft.calibrate_bias(num_samples=50)
        
        config = get_bias_compensation_spec('M')
        estimator = ForceBiasEstimator(config)
        
        for _ in range(20):
            wrench = ft.capture()
            
            # 更新估计器
            estimator.update(wrench.force, wrench.torque, dt=0.01)
            
            # 补偿
            c_force, c_torque = estimator.compensate(wrench.force, wrench.torque)
            
            # 验证形状
            self.assertEqual(c_force.shape, (3,))
            self.assertEqual(c_torque.shape, (3,))

    def test_tactile_bias_compensation(self):
        """触觉偏置补偿"""
        tactile = TactileArray((8, 8), TactileSensorType.RESISTIVE, "t_bias")
        tactile.open()
        
        config = get_bias_compensation_spec('M')
        estimator = TactileBiasEstimator(array_size=(8, 8), config=config)
        
        # 先校准
        frame = tactile.capture()
        estimator.calibrate(frame.pressure_map, frame.temperature_map)
        
        for _ in range(10):
            frame = tactile.capture()
            compensated = estimator.compensate(frame.pressure_map, frame.temperature_map)
            
            # 补偿后形状不变
            self.assertEqual(compensated.shape, (8, 8))



# ============================================================================
# 健康检查
# ============================================================================

class TestHealthCheck(unittest.TestCase):
    """测试运行时健康检查"""

    def test_sensor_health_basic(self):
        """基本传感器健康检查"""
        tactile = TactileArray((8, 8), TactileSensorType.RESISTIVE, "t_health")
        force = ForceTorqueSensor(ForceSensorType.SIX_AXIS, "f_health")
        imu = IMUSensor(IMUSensorType.BMI088, "i_health")
        
        tactile.open()
        force.open()
        imu.open()
        
        # 采集多帧验证数据有效性
        for _ in range(10):
            t_frame = tactile.capture()
            f_wrench = force.capture()
            i_frame = imu.capture()
            
            # 基本合理性检查
            self.assertTrue(0 <= t_frame.pressure_map.mean() <= 1)
            self.assertTrue(-100 < f_wrench.magnitude < 100)
            self.assertTrue(0 < i_frame.accel_magnitude < 20)
        
        tactile.close()
        force.close()
        imu.close()

    def test_tactile_contact_tracking(self):
        """触觉接触跟踪稳定性"""
        tactile = TactileArray((16, 16), TactileSensorType.CAPACITIVE, "t_track")
        tactile.open()
        
        # 连续制造接触压力图
        positions = []
        for i in range(20):
            # 创建人工接触压力图 (中心偏移)
            pressure_map = np.zeros((16, 16), dtype=np.float32)
            cx, cy = int(4 + i), 8  # 移动的接触中心
            radius = 3
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    nx, ny = cx+dx, cy+dy
                    if 0 <= nx < 16 and 0 <= ny < 16:
                        pressure_map[ny, nx] = max(0, 10.0 - np.sqrt(dx**2 + dy**2))
            
            frame = TactileFrame(
                pressure_map=pressure_map,
                timestamp=time.time(),
                frame_id=i,
                sensor_id="t_track"
            )
            contacts = tactile.detect_contacts(frame)
            if contacts:
                positions.append(contacts[0].centroid)
        
        # 跟踪应该稳定 (检测到接触)
        self.assertGreater(len(positions), 10)
        
        tactile.close()

    def test_imu_pose_estimator_convergence(self):
        """IMU姿态估计收敛性"""
        imu = VirtualIMUSensor("imu_conv")
        imu.open()
        
        estimator = PoseEstimator(algorithm='madgwick', sample_rate=200.0)
        
        # 静止状态，姿态应收敛
        initial_euler = None
        final_euler = None
        
        for i in range(100):
            frame = imu.simulate_static((0.1, 0.05, 0.0))  # 小角度倾斜
            pose = estimator.update(frame.accel, frame.gyro)
            
            if i == 10:
                initial_euler = pose.to_euler().copy()
            if i == 99:
                final_euler = pose.to_euler().copy()
        
        # 收敛后欧拉角应稳定
        self.assertIsNotNone(initial_euler)
        self.assertIsNotNone(final_euler)
        self.assertTrue(np.allclose(initial_euler, final_euler, atol=0.1))
        
        imu.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
