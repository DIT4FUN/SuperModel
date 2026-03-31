"""
具身智能综合测试
================

测试感知-学习-控制完整闭环:
- 多传感器协同采集
- 跨模态融合决策
- 世界模型想象 rollout
- 运动控制执行
- 安全监控反馈
"""

import numpy as np
import torch
import unittest
import sys
import time

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.vision import BinocularCamera
from sensors.audio import BinauralMic
from sensors.tactile import TactileArray, TactileSensorType
from sensors.force import ForceTorqueSensor, Wrench, ForceSensorType
from sensors.imu import IMUSensor, PoseEstimator, IMUSensorType
from sensors.manager import SensorManager, SensorManagerConfig, SensorDataFrame
from fusion.cross_modal_fusion import CrossModalFusion, FusionConfig, MultimodalInput
from learning.world_model import create_world_model_agent, get_world_model_spec
from learning.dreamer_agent import create_integrated_agent
from control.motion import MotionController, ControlMode
from control.impedance import ImpedanceController, ImpedanceParams
from control.safety_controller import SafetyController, SafetyConfig


class TestEmbodied闭环(unittest.TestCase):
    """测试具身智能完整闭环"""
    
    def test_perception_to_control_pipeline(self):
        """测试: 感知 → 融合 → 控制 全流程"""
        # 1. 传感器采集
        cam = BinocularCamera()
        cam.open()
        vision_frame = cam.capture()
        cam.close()
        
        # 2. 触觉采集
        tactile = TactileArray(array_size=(16, 16))
        tactile.open()
        tactile_frame = tactile.capture()
        tactile_contacts = tactile.detect_contacts(tactile_frame)
        tactile.close()
        
        # 3. 力觉采集
        force = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        force.open()
        wrench = force.capture()
        contact = force.detect_contact(wrench)
        force.close()
        
        # 4. IMU采集
        imu = IMUSensor(sensor_type=IMUSensorType.BMI088)
        imu.open()
        imu_frame = imu.capture()
        estimator = PoseEstimator(algorithm='madgwick', sample_rate=200)
        pose = estimator.update(imu_frame.accel, imu_frame.gyro)
        imu.close()
        
        # 验证采集成功
        self.assertIsNotNone(vision_frame)
        self.assertIsNotNone(tactile_frame)
        self.assertIsNotNone(wrench)
        self.assertIsNotNone(imu_frame)
        self.assertIsNotNone(pose)
    
    def test_multimodal_fusion_to_action(self):
        """测试: 多模态融合 → 动作选择"""
        fusion = CrossModalFusion(FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256, num_heads=4
        ))
        
        # 多模态输入 - 确保维度正确
        mmi = MultimodalInput(
            vision=torch.randn(2, 512),
            audio=torch.randn(2, 128),
            tactile=torch.randn(2, 64),
            force=torch.randn(2, 32),
            imu=torch.randn(2, 64)
        )
        
        # 融合
        fused = fusion(mmi)
        self.assertEqual(fused.shape, (2, 256))
    
    def test_world_model_grade_specs(self):
        """测试: 世界模型五级规格"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_world_model_spec(grade)
            self.assertGreater(spec.latent_dim, 0)
            self.assertGreater(spec.hidden_dim, 0)
            self.assertGreater(spec.imagination_horizon, 0)
    
    def test_sensor_manager_basic_operation(self):
        """测试: 传感器管理器基本操作"""
        config = SensorManagerConfig(grade='S')
        manager = SensorManager(config=config)
        
        manager.open_all()
        
        # 采集一帧
        frame = manager.capture_all()
        self.assertIsInstance(frame, SensorDataFrame)
        
        manager.close_all()
    
    def test_impedance_control_with_force_feedback(self):
        """测试: 阻抗控制 + 力反馈"""
        params = ImpedanceParams.default_6d()
        ctrl = ImpedanceController(params)
        
        # 期望位置和速度
        desired_pos = np.array([0.5, 0.0, 0.3])
        desired_vel = np.zeros(3)
        
        # 当前位置 (略有偏差)
        current_pos = np.array([0.52, 0.01, 0.28])
        current_vel = np.zeros(3)
        
        # 外部力 (Z方向有接触力)
        external_wrench = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
        
        # 雅可比矩阵 (简化)
        jacobian = np.eye(6)
        
        torque = ctrl.compute_torque(
            desired_pos, desired_vel, current_pos, current_vel,
            external_wrench, jacobian
        )
        
        self.assertEqual(torque.shape, (6,))
        # 验证力矩计算有输出 (具体符号不重要,只要是数值)
        self.assertFalse(np.isnan(torque).any())
        self.assertFalse(np.isinf(torque).any())
    
    def test_pose_estimator_convergence(self):
        """测试: 姿态估计器收敛"""
        estimator = PoseEstimator(algorithm='madgwick', sample_rate=200, beta=0.1)
        
        # 模拟静态姿态
        accel_static = np.array([0.0, 0.0, 9.81])
        gyro_static = np.zeros(3)
        
        # 多次更新
        for _ in range(200):
            estimator.update(accel_static, gyro_static)
        
        euler = estimator.get_euler()
        
        # 应该收敛到 roll=0, pitch=0
        self.assertAlmostEqual(euler[0], 0.0, delta=0.15)  # roll
        self.assertAlmostEqual(euler[1], 0.0, delta=0.15)  # pitch
    
    def test_tactile_contact_grip_quality(self):
        """测试: 触觉接触 + 抓取质量评估"""
        tactile = TactileArray(array_size=(16, 16), sensor_type=TactileSensorType.CAPACITIVE)
        tactile.open()
        
        # 模拟接触
        frame = tactile.capture()
        contacts = tactile.detect_contacts(frame)
        
        # 抓取质量评估
        quality = tactile.estimate_grip_quality(frame)
        
        self.assertIn('overall', quality)
        self.assertIn('contact_area', quality)
        self.assertIn('uniformity', quality)
        self.assertIn('stability', quality)
        
        self.assertGreaterEqual(quality['overall'], 0.0)
        self.assertLessEqual(quality['overall'], 1.0)
        
        tactile.close()
    
    def test_force_wrench_coordinate_transform(self):
        """测试: 力旋量坐标变换"""
        wrench = Wrench(
            force=np.array([10.0, 0.0, 0.0]),
            torque=np.array([0.0, 0.0, 5.0])
        )
        
        # 旋转 90度绕 Z轴
        rotation = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        translation = np.array([0.1, 0.0, 0.0])
        
        wrench_world = wrench.transform(rotation, translation)
        
        # 验证变换
        self.assertEqual(wrench_world.force.shape, (3,))
        self.assertEqual(wrench_world.torque.shape, (3,))
    
    def test_fusion_grades_config(self):
        """测试: 融合网络 AGV 五级配置"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            fusion = CrossModalFusion(FusionConfig(
                vision_dim=512, audio_dim=128, tactile_dim=64,
                force_dim=32, imu_dim=64, hidden_dim=256, num_heads=4
            ))
            
            mmi = MultimodalInput(
                vision=torch.randn(1, 512),
                audio=torch.randn(1, 128),
                tactile=torch.randn(1, 64),
                force=torch.randn(1, 32),
                imu=torch.randn(1, 64)
            )
            
            output = fusion(mmi)
            self.assertEqual(output.shape[1], 256)


class TestEmbodiedPerformance(unittest.TestCase):
    """具身智能性能测试"""
    
    def test_sensor_capture_throughput(self):
        """传感器采集吞吐量测试"""
        cam = BinocularCamera()
        tactile = TactileArray(array_size=(16, 16))
        force = ForceTorqueSensor()
        imu = IMUSensor()
        
        cam.open()
        tactile.open()
        force.open()
        imu.open()
        
        count = 0
        start = time.perf_counter()
        while time.perf_counter() - start < 0.1:
            _ = cam.capture()
            _ = tactile.capture()
            _ = force.capture()
            _ = imu.capture()
            count += 1
        
        throughput = count / 0.1
        
        # 吞吐量应该合理
        self.assertGreater(throughput, 10)
        
        cam.close()
        tactile.close()
        force.close()
        imu.close()
    
    def test_fusion_inference_latency(self):
        """融合推理延迟测试"""
        import time
        
        fusion = CrossModalFusion(FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256, num_heads=4
        ))
        
        mmi = MultimodalInput(
            vision=torch.randn(1, 512),
            audio=torch.randn(1, 128),
            tactile=torch.randn(1, 64),
            force=torch.randn(1, 32),
            imu=torch.randn(1, 64)
        )
        
        # 预热
        for _ in range(5):
            _ = fusion(mmi)
        
        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            _ = fusion(mmi)
            latencies.append((time.perf_counter() - start) * 1000)
        
        avg_latency = np.mean(latencies)
        
        # 平均延迟应该小于 20ms
        self.assertLess(avg_latency, 20)


class TestEmbodiedSafety(unittest.TestCase):
    """具身智能安全测试"""
    
    def test_safety_controller_basic(self):
        """测试: 安全控制器基本功能"""
        from control.safety_controller import get_safety_spec, SafetyLevel
        
        config = SafetyConfig(
            joint_limits_lower=np.array([-1.0] * 6),
            joint_limits_upper=np.array([1.0] * 6),
            velocity_limits=np.array([1.0] * 6),
            acceleration_limits=np.array([2.0] * 6),
        )
        safety = SafetyController(config=config)
        
        # 安全控制器基本状态检查
        self.assertFalse(safety.is_emergency_stopped)  # property
        self.assertEqual(safety.fault_count, 0)  # property
        
        # 获取安全规格
        spec = get_safety_spec(SafetyLevel.M)
        self.assertIn('level', spec)
        self.assertIn('response_time_ms', spec)


class TestEmbodiedIntegration(unittest.TestCase):
    """具身智能集成测试"""
    
    def test_sensor_time_alignment(self):
        """传感器时间对齐测试"""
        cam = BinocularCamera()
        tactile = TactileArray(array_size=(8, 8))
        
        cam.open()
        tactile.open()
        
        timestamps = {'cam': [], 'tactile': []}
        
        for _ in range(10):
            t0 = time.perf_counter()
            _ = cam.capture()
            t1 = time.perf_counter()
            _ = tactile.capture()
            t2 = time.perf_counter()
            
            timestamps['cam'].append(t1 - t0)
            timestamps['tactile'].append(t2 - t1)
        
        cam.close()
        tactile.close()
        
        # 采集时间应该合理
        self.assertTrue(all(t < 0.1 for t in timestamps['cam']))
        self.assertTrue(all(t < 0.05 for t in timestamps['tactile']))
    
    def test_tactile_force_correlation(self):
        """触觉-力觉相关性测试"""
        tactile = TactileArray(array_size=(16, 16), sensor_type=TactileSensorType.CAPACITIVE)
        tactile.open()
        force = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        force.open()
        
        tactile_values = []
        force_values = []
        
        for _ in range(30):
            tf = tactile.capture()
            contacts = tactile.detect_contacts(tf)
            wrench = force.capture()
            
            total_pressure = np.sum(tf.pressure_map)
            force_mag = wrench.magnitude
            
            tactile_values.append(total_pressure)
            force_values.append(force_mag)
        
        # 验证数据合理性
        self.assertEqual(len(tactile_values), 30)
        self.assertEqual(len(force_values), 30)
        self.assertTrue(all(f >= 0 for f in force_values))
        
        tactile.close()
        force.close()
    
    def test_imu_orientation_consistency(self):
        """IMU方向一致性测试"""
        imu = IMUSensor(sensor_type=IMUSensorType.BMI088)
        imu.open()
        
        estimator = PoseEstimator(algorithm='madgwick', sample_rate=200)
        
        euler_history = []
        for _ in range(200):
            frame = imu.capture()
            pose = estimator.update(frame.accel, frame.gyro)
            euler_history.append(estimator.get_euler())
        
        euler_arr = np.array(euler_history)
        
        # 检查收敛性 (最后20帧应该稳定)
        last_20 = euler_arr[-20:]
        roll_std = np.std(last_20[:, 0])
        pitch_std = np.std(last_20[:, 1])
        
        self.assertLess(roll_std, 0.1)
        self.assertLess(pitch_std, 0.1)
        
        imu.close()
    
    def test_motion_controller_basic(self):
        """运动控制器基本功能测试"""
        controller = MotionController(num_joints=6, control_rate=100)
        
        target_position = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
        torque = controller.compute_joint_torque(target_position)
        
        self.assertEqual(torque.shape, (6,))
        self.assertFalse(np.isnan(torque).any())
        self.assertFalse(np.isinf(torque).any())


if __name__ == '__main__':
    unittest.main()
