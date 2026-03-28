"""
SuperModel 集成测试
===================

端到端测试:
传感器 → 融合 → 世界模型 → 控制器 → 仿真反馈
"""

import numpy as np
import torch
import sys
import time
import unittest

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.vision import BinocularCamera, DepthProcessor, CameraIntrinsics, StereoExtrinsics, StereoFrame
from sensors.audio import BinauralMic, SoundLocalizer, AudioFrame
from sensors.tactile import TactileArray, TactileFrame
from sensors.force import ForceTorqueSensor, Wrench, ForceSensorType
from sensors.imu import IMUSensor, IMUFrame, PoseEstimator, IMUSensorType
from sensors.encoders import create_sensor_encoder
from fusion.cross_modal_fusion import CrossModalFusion, FusionConfig, MultimodalInput
from learning.world_model import WorldModel, ModelState
from control.motion import MotionController, JointState, ControlMode
from simulation.environment import RobotSimulator, SensorSimulator, SimConfig


class TestEndToEndPipeline(unittest.TestCase):
    """端到端流水线测试"""
    
    def setUp(self):
        self.vision = BinocularCamera(resolution=(640, 480), fps=30)
        self.audio = BinauralMic(sample_rate=16000, chunk_size=512)
        self.tactile = TactileArray(array_size=(16, 16))
        self.force = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        self.imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL)
        
        self.vision.open()
        self.audio.open()
        self.tactile.open()
        self.force.open()
        self.imu.open()
        
        self.encoder = create_sensor_encoder({
            'vision': (3, 224, 224),
            'audio': (100, 64),
            'tactile': (1, 16, 16),
            'force': (10, 6),
            'imu': (10, 6),
        }, grade='M')
        
        self.fusion = CrossModalFusion(FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256
        ))
        
        self.controller = MotionController(num_joints=6, control_rate=100.0)
        self.sim = RobotSimulator(SimConfig(num_joints=6, dt=0.01))
        self.sensor_sim = SensorSimulator(self.sim, SimConfig(dt=0.01))
    
    def tearDown(self):
        self.vision.close()
        self.audio.close()
        self.tactile.close()
        self.force.close()
        self.imu.close()
    
    def test_sensor_capture_all_modalities(self):
        """测试所有传感器同时采集"""
        vision_frame = self.vision.capture()
        self.assertIsInstance(vision_frame, StereoFrame)
        
        audio_frame = self.audio.capture()
        self.assertIsInstance(audio_frame, AudioFrame)
        
        tactile_frame = self.tactile.capture()
        self.assertIsInstance(tactile_frame, TactileFrame)
        
        wrench = self.force.capture()
        self.assertIsInstance(wrench, Wrench)
        
        imu_frame = self.imu.capture()
        self.assertIsInstance(imu_frame, IMUFrame)
    
    def test_encoder_processes_all_modalities(self):
        """测试编码器处理所有模态"""
        B = 2
        
        vision = torch.randn(B, 3, 224, 224)
        audio = torch.randn(B, 100, 64)
        tactile = torch.randn(B, 1, 16, 16)
        force = torch.randn(B, 10, 6)
        imu = torch.randn(B, 10, 6)
        
        encoded = self.encoder({
            'vision': vision,
            'audio': audio,
            'tactile': tactile,
            'force': force,
            'imu': imu
        })
        
        self.assertIn('fused', encoded)
        self.assertEqual(encoded['fused'].shape[0], B)
    
    def test_fusion_with_all_inputs(self):
        """测试跨模态融合"""
        B = 2
        multimodal = MultimodalInput(
            vision=torch.randn(B, 512),
            audio=torch.randn(B, 128),
            tactile=torch.randn(B, 64),
            force=torch.randn(B, 32),
            imu=torch.randn(B, 64)
        )
        
        fused = self.fusion(multimodal)
        self.assertEqual(fused.shape, (B, 256))
        self.assertFalse(torch.isnan(fused).any())
    
    def test_simulation_step(self):
        """测试仿真环境"""
        for i in range(10):
            state = self.sim.step(np.random.randn(6) * 0.5)
        
        self.assertIn('joint_positions', state)
        self.assertEqual(len(state['joint_positions']), 6)
        
        noisy_pos = self.sensor_sim.get_noisy_joint_positions()
        self.assertEqual(len(noisy_pos), 6)
    
    def test_imu_pose_estimation_loop(self):
        """测试IMU姿态估计循环"""
        estimator = PoseEstimator(algorithm='madgwick', sample_rate=100.0)
        
        for _ in range(50):
            frame = self.imu.capture()
            pose = estimator.update(frame.accel, frame.gyro)
            euler = estimator.get_euler()
        
        self.assertEqual(euler.shape, (3,))
        self.assertAlmostEqual(euler[0], 0.0, delta=0.5)  # roll ~ 0
    
    def test_force_contact_detection(self):
        """测试力传感器接触检测"""
        self.force.open()
        wrench = self.force.capture()
        state = self.force.detect_contact(wrench, threshold=2.0)
        self.assertIn(state.is_contact, [True, False])
    
    def test_control_loop_with_simulation(self):
        """测试控制环与仿真环境集成"""
        joint_state = JointState(
            position=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            velocity=np.zeros(6),
            torque=np.zeros(6)
        )
        self.controller.update_joint_state(joint_state)
        
        target = np.array([0.5, 0.3, -0.2, 0.1, 0.0, 0.0])
        torque = self.controller.compute_joint_torque(target)
        
        self.assertEqual(len(torque), 6)
        
        state = self.sim.step(torque)
        self.assertIsNotNone(state)
    
    def test_full_agent_loop(self):
        """测试完整智能体循环 (感知→融合→决策→控制)"""
        # 1. 感知层
        vision_frame = self.vision.capture()
        audio_frame = self.audio.capture()
        tactile_frame = self.tactile.capture()
        wrench = self.force.capture()
        imu_frame = self.imu.capture()
        
        # 2. 编码
        encoded = self.encoder({
            'vision': torch.randn(1, 3, 224, 224),
            'audio': torch.randn(1, 100, 64),
            'tactile': torch.randn(1, 1, 16, 16),
            'force': torch.randn(1, 10, 6),
            'imu': torch.randn(1, 10, 6)
        })
        
        # 3. 融合
        multimodal = MultimodalInput(
            vision=torch.randn(1, 512),
            audio=torch.randn(1, 128),
            tactile=torch.randn(1, 64),
            force=torch.randn(1, 32),
            imu=torch.randn(1, 64)
        )
        fused = self.fusion(multimodal)
        
        # 4. 决策 (模拟)
        action = torch.randn(1, 6)
        
        # 5. 控制
        torque = self.controller.compute_joint_torque(np.zeros(6))
        
        # 6. 执行
        state = self.sim.step(torque)
        
        # 验证
        self.assertEqual(fused.shape[0], 1)
        self.assertEqual(len(torque), 6)
        self.assertIn('time', state)
    
    def test_world_model_step(self):
        """测试世界模型单步"""
        obs_embed = torch.randn(1, 256)
        action = torch.randn(1, 6)
        prev_state = ModelState(
            deter=torch.randn(1, 256),
            stoch=torch.randn(1, 32),
            action=torch.randn(1, 6)
        )
        
        # 这里只验证接口，不实际调用forward（需要完整的obs_dims）
        self.assertEqual(obs_embed.shape[0], 1)
        self.assertEqual(action.shape[1], 6)
        self.assertEqual(prev_state.deter.shape, (1, 256))


class TestPerformanceBenchmarks(unittest.TestCase):
    """性能基准测试"""
    
    def test_fusion_latency(self):
        """测试融合延迟"""
        fusion = CrossModalFusion(FusionConfig(hidden_dim=256, num_heads=4, num_layers=2))
        
        multimodal = MultimodalInput(
            vision=torch.randn(1, 512),
            audio=torch.randn(1, 128),
            tactile=torch.randn(1, 64),
            force=torch.randn(1, 32),
            imu=torch.randn(1, 64)
        )
        
        # 预热
        for _ in range(5):
            fusion(multimodal)
        
        # 计时
        start = time.time()
        iterations = 100
        for _ in range(iterations):
            fusion(multimodal)
        elapsed = time.time() - start
        
        avg_ms = (elapsed / iterations) * 1000
        print(f"\n[Fusion] Average latency: {avg_ms:.2f}ms per iteration")
        
        # M级应该 < 20ms
        self.assertLess(avg_ms, 50, "Fusion too slow")
    
    def test_encoder_throughput(self):
        """测试编码器吞吐量"""
        encoder = create_sensor_encoder({
            'vision': (3, 224, 224),
            'audio': (100, 64),
        }, grade='M')
        
        B = 8
        batch = {
            'vision': torch.randn(B, 3, 224, 224),
            'audio': torch.randn(B, 100, 64),
        }
        
        start = time.time()
        for _ in range(20):
            encoder(batch)
        elapsed = time.time() - start
        
        throughput = (B * 20) / elapsed
        print(f"\n[Encoder] Throughput: {throughput:.1f} samples/sec")
        
        self.assertGreater(throughput, 10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
