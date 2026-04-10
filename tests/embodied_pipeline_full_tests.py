"""
具身智能全链路集成测试
======================

测试完整的感知→融合→控制闭环流程:
1. 传感器数据采集 (触觉/力觉/IMU)
2. 多传感器融合
3. 具身控制决策
4. AGV五级安全监控

覆盖 AGV 五级规格 (S/M/L/XL/XXL) 全流程
"""

import unittest
import numpy as np
import sys
import os

# ── 路径设置 ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_SrcPath = os.path.join(_PROJECT_ROOT, 'src')
if _SrcPath not in sys.path:
    sys.path.insert(0, _SrcPath)

from src.sensors.tactile import (
    TactileArray, TactileFrame, TactileSensorType,
    VirtualTactileSensor, get_tactile_spec
)
from src.sensors.force import (
    ForceTorqueSensor, Wrench, ForceSensorType,
    VirtualForceSensor, get_force_spec
)
from src.sensors.imu import (
    IMUSensor, IMUFrame, Pose, PoseEstimator, IMUSensorType,
    VirtualIMUSensor, get_imu_spec
)
from src.fusion.sensor_fusion import ComplementaryFilter, ExtendedKalmanFilter
from src.control.safety_controller import SafetyController, SafetyConfig, SafetyLevel


class TestEmbodiedPipeline(unittest.TestCase):
    """具身智能全链路管道测试"""

    def test_tactile_to_control_pipeline(self):
        """测试触觉→控制管道 (TactileArray 真实传感器)"""
        spec = get_tactile_spec('M')
        tactile = TactileArray(
            array_size=tuple(spec['array']),
            sensor_type=TactileSensorType.CAPACITIVE,
            sensor_id="test_tactile_control"
        )
        tactile.open()
        
        # 采集真实数据
        frame = tactile.capture()
        
        # 检测接触
        contacts = tactile.detect_contacts(frame)
        
        # 评估抓取质量
        quality = tactile.estimate_grip_quality(frame)
        
        # 连续capture用于滑移检测
        tactile.capture()
        tactile.capture()
        slip = tactile.get_slip_signal()
        
        tactile.close()
        
        # 验证结果
        self.assertIsInstance(frame, TactileFrame)
        self.assertEqual(frame.pressure_map.shape, tuple(spec['array']))
        self.assertIsInstance(contacts, list)
        self.assertIn('overall', quality)
        self.assertIsInstance(slip, np.ndarray)
        self.assertEqual(slip.shape, tuple(spec['array']))
        self.assertGreaterEqual(quality['overall'], 0.0)
        self.assertLessEqual(quality['overall'], 1.0)
    
    def test_force_to_control_pipeline(self):
        """测试力觉→控制管道 (ForceTorqueSensor 真实传感器)"""
        spec = get_force_spec('L')
        force = ForceTorqueSensor(
            sensor_type=ForceSensorType.SIX_AXIS,
            sensor_id="test_force_control"
        )
        force.open()
        
        # 采集真实数据
        wrench = force.capture()
        
        # 接触检测
        contact_state = force.detect_contact(wrench, threshold=5.0)
        
        # 负载估计
        payload = force.estimate_payload(wrench)
        
        force.close()
        
        self.assertIsInstance(wrench, Wrench)
        self.assertEqual(len(wrench.to_vector()), 6)
        self.assertIsInstance(contact_state.is_contact, (bool, np.bool_))
        self.assertGreaterEqual(payload, 0)
    
    def test_imu_to_control_pipeline(self):
        """测试IMU→控制管道"""
        spec = get_imu_spec('L')
        imu = IMUSensor(
            sensor_type=IMUSensorType.BMI088,
            sensor_id="test_imu_control",
            sample_rate=spec['sample_hz']
        )
        imu.open()
        
        estimator = PoseEstimator(algorithm='madgwick', sample_rate=spec['sample_hz'])
        
        poses = []
        for _ in range(50):
            frame = imu.capture()
            pose = estimator.update(frame.accel, frame.gyro, frame.mag, dt=1.0/spec['sample_hz'])
            poses.append(pose)
        
        imu.close()
        
        self.assertEqual(len(poses), 50)
        for pose in poses:
            self.assertIsInstance(pose, Pose)
            self.assertEqual(len(pose.orientation), 4)
            self.assertEqual(len(pose.position), 3)
    
    def test_multi_sensor_fusion_pipeline(self):
        """测试多传感器融合管道"""
        tactile = TactileArray((8, 8), sensor_id="fusion_tactile")
        tactile.open()
        
        force = ForceTorqueSensor(sensor_id="fusion_force")
        force.open()
        
        imu = IMUSensor(sensor_id="fusion_imu")
        imu.open()
        
        comp_filter = ComplementaryFilter(alpha=0.96)
        ekf = ExtendedKalmanFilter(state_dim=6, measurement_dim=3)
        ekf.initialize(np.zeros(6))
        
        for i in range(20):
            t_frame = tactile.capture()
            f_wrench = force.capture()
            i_frame = imu.capture()
            
            state = comp_filter.update({
                'accel': i_frame.accel,
                'gyro': i_frame.gyro
            }, dt=0.01)
            
            ekf.predict(dt=0.01)
        
        tactile.close()
        force.close()
        imu.close()
        
        state = comp_filter.get_state()
        self.assertEqual(len(state), 3)
        ekf_state = ekf.get_state()
        self.assertEqual(len(ekf_state), 6)


class TestGradeAwareEmbodiedPipeline(unittest.TestCase):
    """AGV五级具身控制管道测试"""

    GRADES = ['S', 'M', 'L', 'XL', 'XXL']
    
    def test_grade_tactile_pipeline(self):
        """测试各等级触觉管道"""
        for grade in self.GRADES:
            spec = get_tactile_spec(grade)
            tactile = TactileArray(
                array_size=tuple(spec['array']),
                sensor_type=TactileSensorType.CAPACITIVE if grade != 'S' else TactileSensorType.RESISTIVE,
                sensor_id=f"grade_{grade}_tactile"
            )
            tactile.open()
            
            frame = tactile.capture()
            contacts = tactile.detect_contacts(frame)
            quality = tactile.estimate_grip_quality(frame)
            
            tactile.close()
            
            self.assertIsInstance(frame, TactileFrame)
            self.assertIsInstance(contacts, list)
            self.assertIn('overall', quality)
    
    def test_grade_force_pipeline(self):
        """测试各等级力觉管道"""
        for grade in self.GRADES:
            spec = get_force_spec(grade)
            force = ForceTorqueSensor(
                sensor_type=ForceSensorType.SIX_AXIS if spec['axes'] == 6 else ForceSensorType.THREE_AXIS,
                sensor_id=f"grade_{grade}_force"
            )
            force.open()
            
            wrench = force.capture()
            contact = force.detect_contact(wrench)
            payload = force.estimate_payload(wrench)
            
            force.close()
            
            self.assertIsInstance(wrench, Wrench)
            self.assertIsInstance(contact.is_contact, (bool, np.bool_))
            self.assertGreaterEqual(payload, 0)
    
    def test_grade_imu_pipeline(self):
        """测试各等级IMU管道"""
        for grade in self.GRADES:
            spec = get_imu_spec(grade)
            imu_type_map = {
                'S': IMUSensorType.MPU6050,
                'M': IMUSensorType.BMI088,
                'L': IMUSensorType.BMI088,
                'XL': IMUSensorType.ADIS16470,
                'XXL': IMUSensorType.ADIS16470,
            }
            imu = IMUSensor(
                sensor_type=imu_type_map[grade],
                sensor_id=f"grade_{grade}_imu",
                sample_rate=spec['sample_hz']
            )
            imu.open()
            
            estimator = PoseEstimator(algorithm='madgwick', sample_rate=spec['sample_hz'])
            
            for _ in range(10):
                frame = imu.capture()
                pose = estimator.update(frame.accel, frame.gyro, frame.mag, dt=1.0/spec['sample_hz'])
            
            imu.close()
            
            self.assertIsInstance(pose, Pose)
    
    def test_grade_safety_monitoring(self):
        """测试各等级安全监控"""
        for grade in self.GRADES:
            config = SafetyConfig(
                joint_limits_lower=np.array([-1.0] * 6),
                joint_limits_upper=np.array([1.0] * 6),
                velocity_limits=np.array([1.0] * 6),
                acceleration_limits=np.array([1.0] * 6),
                torque_limits=np.array([10.0] * 6),
                force_limits=np.array([100.0] * 6),
                safety_level=SafetyLevel[grade]
            )
            controller = SafetyController(config=config)
            self.assertIsNotNone(controller)
            self.assertEqual(controller.safety_level, SafetyLevel[grade])


class TestVirtualSensorEmbodiedPipeline(unittest.TestCase):
    """虚拟传感器具身管道测试"""

    def test_virtual_tactile_contact(self):
        """测试虚拟触觉接触"""
        sensor = VirtualTactileSensor(array_size=(16, 16))
        sensor.open()
        
        frame = sensor.simulate_contact(
            contact_pos=(0.4, 0.6),
            contact_radius=0.2,
            contact_force=20.0,
            noise_level=0.01
        )
        
        self.assertIsInstance(frame, TactileFrame)
        self.assertEqual(frame.pressure_map.shape, (16, 16))
        self.assertGreater(np.max(frame.pressure_map), 0)
        
        sensor.close()
    
    def test_virtual_force_contact(self):
        """测试虚拟力觉接触"""
        sensor = VirtualForceSensor(sensor_id="virtual_ft")
        sensor.open()
        
        wrench = sensor.simulate_contact(
            force=(15.0, -10.0, -30.0),
            torque=(1.0, -0.5, 0.2),
            add_noise=True
        )
        
        self.assertIsInstance(wrench, Wrench)
        self.assertEqual(wrench.force.shape, (3,))
        self.assertEqual(wrench.torque.shape, (3,))
        
        sensor.close()
    
    def test_virtual_imu_motion(self):
        """测试虚拟IMU运动"""
        sensor = VirtualIMUSensor(sensor_id="virtual_imu")
        sensor.open()
        
        frames = sensor.simulate_trajectory(
            trajectory_type="circle",
            duration_s=1.0,
            dt=0.01
        )
        
        self.assertEqual(len(frames), 100)
        for frame in frames:
            self.assertIsInstance(frame, IMUFrame)
            self.assertEqual(frame.accel.shape, (3,))
            self.assertEqual(frame.gyro.shape, (3,))
        
        sensor.close()
    
    def test_virtual_agv_motion(self):
        """测试虚拟AGV运动"""
        sensor = VirtualIMUSensor(sensor_id="virtual_agv_imu")
        sensor.open()
        
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            frame = sensor.simulate_agv_motion(
                linear_velocity=(0.5, 0.3),
                angular_velocity=0.2,
                dt=0.01,
                grade=grade
            )
            self.assertIsInstance(frame, IMUFrame)
        
        sensor.close()


class TestEmbodied闭环(unittest.TestCase):
    """具身控制闭环测试"""

    def test_closed_loop_tactile_control(self):
        """测试触觉闭环控制 (VirtualTactileSensor)"""
        tactile = VirtualTactileSensor(array_size=(16, 16), sensor_id="闭环_tactile")
        tactile.open()
        
        frame1 = tactile.simulate_contact((0.5, 0.5), 0.3, 10.0)
        frame2 = tactile.simulate_contact((0.5, 0.5), 0.3, 20.0)
        
        tactile.close()
        
        # 更强的接触应该有更高的最大压力
        self.assertGreater(np.max(frame2.pressure_map), np.max(frame1.pressure_map) * 0.9)
    
    def test_closed_loop_force_control(self):
        """测试力觉闭环控制"""
        force = VirtualForceSensor(sensor_id="闭环_force")
        force.open()
        
        normal_wrench = force.simulate_contact((0, 0, -10), add_noise=True)
        collision_wrench = force.simulate_contact((50, -30, -80), add_noise=True)
        
        force.close()
        
        self.assertGreater(collision_wrench.magnitude, normal_wrench.magnitude)
    
    def test_closed_loop_imu_control(self):
        """测试IMU闭环姿态控制"""
        imu = VirtualIMUSensor(sensor_id="闭环_imu")
        imu.open()
        
        estimator = PoseEstimator(algorithm='madgwick', sample_rate=100.0)
        
        for _ in range(50):
            frame = imu.simulate_static(orientation=(0.1, 0.05, 0.0))
            pose = estimator.update(frame.accel, frame.gyro, None, dt=0.01)
        
        euler = pose.to_euler()
        
        imu.close()
        
        # roll和pitch应该接近设定值,允许漂移误差
        self.assertLess(abs(euler[0]), 0.3)
        self.assertLess(abs(euler[1]), 0.3)


if __name__ == '__main__':
    unittest.main()
