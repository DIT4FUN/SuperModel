"""
SuperModel 性能基准测试
=======================

测试各模块的延迟、吞吐量和内存使用情况
"""

import time
import gc
import sys
import tracemalloc
import unittest
from unittest.mock import MagicMock
import numpy as np
import torch

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.vision import BinocularCamera, DepthProcessor, CameraIntrinsics, StereoExtrinsics
from sensors.audio import BinauralMic, SoundLocalizer
from sensors.tactile import TactileArray
from sensors.force import ForceTorqueSensor
from sensors.imu import IMUSensor, PoseEstimator
from fusion.cross_modal_fusion import (
    CrossModalAttention, FusionConfig, MultimodalInput,
    CrossModalFusion, UnifiedRepresentation
)
from control.motion import MotionController
from control.impedance import ImpedanceController, ImpedanceParams
from control.safety_controller import SafetyController, SafetyConfig


def make_stereo_extrinsics():
    """创建测试用双目外参"""
    return StereoExtrinsics(
        rotation=np.eye(3),
        translation=np.array([-0.05, 0.0, 0.0])
    )


class TestSensorBenchmark(unittest.TestCase):
    """传感器性能基准测试"""

    def test_vision_capture_latency(self):
        """测试双目相机采集延迟"""
        cam = BinocularCamera()
        cam.open()

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            cam.capture()
            latencies.append((time.perf_counter() - start) * 1000)

        cam.close()
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        print(f"\n  双目相机采集延迟: avg={avg_latency:.2f}ms p95={p95_latency:.2f}ms")
        self.assertLess(avg_latency, 50, "平均延迟应小于50ms")

    def test_depth_processor_throughput(self):
        """测试深度处理吞吐量"""
        left_int = CameraIntrinsics(width=640, height=480, fx=385.5, fy=385.5, cx=319.5, cy=239.5)
        right_int = CameraIntrinsics(width=640, height=480, fx=385.5, fy=385.5, cx=319.5, cy=239.5)
        ext = make_stereo_extrinsics()
        processor = DepthProcessor(left_int, right_int, ext)
        depth = np.random.rand(480, 640).astype(np.float32) * 10.0

        start = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            processor.filter_depth(depth, min_dist=0.1, max_dist=5.0)
            processor.project_to_3d(u=320, v=240, depth=1.0)
        elapsed = time.perf_counter() - start

        fps = iterations / elapsed
        print(f"\n  深度处理吞吐量: {fps:.1f} fps")
        self.assertGreater(fps, 10, "吞吐量应大于10fps")

    def test_audio_capture_latency(self):
        """测试双耳麦克风采集延迟"""
        mic = BinauralMic()
        mic.open()

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            mic.capture()
            latencies.append((time.perf_counter() - start) * 1000)

        mic.close()
        avg_latency = np.mean(latencies)
        print(f"\n  双耳麦克风采集延迟: avg={avg_latency:.2f}ms")
        self.assertLess(avg_latency, 60, "平均延迟应小于60ms")

    def test_sound_localization_latency(self):
        """测试声源定位延迟"""
        loc = SoundLocalizer(baseline_mm=95.0, sample_rate=16000)
        left = np.random.randn(512)
        right = np.random.randn(512)

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            loc.localize(left, right)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = np.mean(latencies)
        print(f"\n  声源定位延迟: avg={avg_latency:.2f}ms")
        self.assertLess(avg_latency, 20, "定位延迟应小于20ms")

    def test_tactile_capture_latency(self):
        """测试触觉采集延迟"""
        tactile = TactileArray()
        tactile.open()

        latencies = []
        for _ in range(200):
            start = time.perf_counter()
            tactile.capture()
            latencies.append((time.perf_counter() - start) * 1000)

        tactile.close()
        avg_latency = np.mean(latencies)
        print(f"\n  触觉采集延迟: avg={avg_latency:.2f}ms")
        self.assertLess(avg_latency, 10, "平均延迟应小于10ms")

    def test_tactile_contact_detection_latency(self):
        """测试触觉接触检测延迟"""
        tactile = TactileArray()
        tactile.open()
        frame = tactile.capture()

        latencies = []
        for _ in range(200):
            start = time.perf_counter()
            tactile.detect_contacts(frame)
            latencies.append((time.perf_counter() - start) * 1000)

        tactile.close()
        avg_latency = np.mean(latencies)
        print(f"\n  触觉接触检测延迟: avg={avg_latency:.2f}ms")
        self.assertLess(avg_latency, 5, "检测延迟应小于5ms")

    def test_force_sensor_capture_latency(self):
        """测试力觉采集延迟"""
        sensor = ForceTorqueSensor()
        sensor.open()

        latencies = []
        for _ in range(500):
            start = time.perf_counter()
            sensor.capture()
            latencies.append((time.perf_counter() - start) * 1000)

        sensor.close()
        avg_latency = np.mean(latencies)
        print(f"\n  力觉采集延迟: avg={avg_latency:.2f}ms")
        self.assertLess(avg_latency, 5, "平均延迟应小于5ms")

    def test_imu_capture_latency(self):
        """测试IMU采集延迟"""
        imu = IMUSensor()
        imu.open()

        latencies = []
        for _ in range(500):
            start = time.perf_counter()
            imu.capture()
            latencies.append((time.perf_counter() - start) * 1000)

        imu.close()
        avg_latency = np.mean(latencies)
        print(f"\n  IMU采集延迟: avg={avg_latency:.2f}ms")
        self.assertLess(avg_latency, 5, "平均延迟应小于5ms")

    def test_pose_estimation_latency(self):
        """测试姿态估计延迟"""
        estimator = PoseEstimator(algorithm='madgwick', beta=0.1)
        accel = np.array([0.0, 0.0, 9.8])
        gyro = np.array([0.1, 0.05, 0.0])

        latencies = []
        for _ in range(1000):
            start = time.perf_counter()
            estimator.update(accel, gyro)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = np.mean(latencies)
        print(f"\n  姿态估计延迟: avg={avg_latency:.3f}ms")
        self.assertLess(avg_latency, 1, "姿态估计延迟应小于1ms")


class TestFusionBenchmark(unittest.TestCase):
    """融合网络性能基准测试"""

    def test_cross_attention_latency(self):
        """测试跨模态注意力延迟"""
        attn = CrossModalAttention(query_dim=128, key_dim=128, value_dim=128, num_heads=4)
        q = torch.randn(2, 10, 128)
        k = torch.randn(2, 10, 128)
        v = torch.randn(2, 10, 128)

        # warmup
        _ = attn(q, k, v)

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            attn(q, k, v)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = np.mean(latencies)
        print(f"\n  跨模态注意力延迟: avg={avg_latency:.2f}ms")
        self.assertLess(avg_latency, 50, "融合延迟应小于50ms")

    def test_fusion_throughput(self):
        """测试融合网络吞吐量"""
        config = FusionConfig(
            vision_dim=512, audio_dim=128,
            tactile_dim=64, force_dim=32, imu_dim=64,
            hidden_dim=256, num_heads=4
        )
        fusion = CrossModalFusion(config)

        multimodal = MultimodalInput(
            vision=torch.randn(4, 512),
            audio=torch.randn(4, 128),
            tactile=torch.randn(4, 64),
            force=torch.randn(4, 32),
            imu=torch.randn(4, 64),
            language=None
        )

        # warmup
        _ = fusion(multimodal)

        start = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            fusion(multimodal)
        elapsed = time.perf_counter() - start

        fps = iterations / elapsed
        print(f"\n  融合网络吞吐量: {fps:.1f} fps (batch=4)")
        self.assertGreater(fps, 20, "吞吐量应大于20fps")

    def test_unified_representation_latency(self):
        """测试统一表示层延迟"""
        ur = UnifiedRepresentation(input_dim=256, hidden_dim=512, output_dim=128)
        x = torch.randn(4, 256)

        # warmup
        _ = ur(x)

        latencies = []
        for _ in range(200):
            start = time.perf_counter()
            ur(x)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = np.mean(latencies)
        print(f"\n  统一表示层延迟: avg={avg_latency:.2f}ms")
        self.assertLess(avg_latency, 20, "统一表示层延迟应小于20ms")


class TestControlBenchmark(unittest.TestCase):
    """控制系统性能基准测试"""

    def test_motion_controller_latency(self):
        """测试运动控制器延迟"""
        controller = MotionController(num_joints=6)

        latencies = []
        for _ in range(1000):
            start = time.perf_counter()
            controller.compute_joint_torque(np.random.randn(6))
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = np.mean(latencies)
        print(f"\n  运动控制器延迟: avg={avg_latency:.3f}ms")
        self.assertLess(avg_latency, 1, "控制器延迟应小于1ms")

    def test_impedance_controller_latency(self):
        """测试阻抗控制器延迟"""
        controller = ImpedanceController(ImpedanceParams.default_6d())

        n_joints = 6
        # Cartesian position (3D) and velocity (3D)
        des_pos = np.zeros(3)
        des_vel = np.zeros(3)
        cur_pos = np.zeros(3)
        cur_vel = np.zeros(3)
        ext_wrench = np.zeros(6)
        jac = np.eye(6, n_joints)

        latencies = []
        for _ in range(1000):
            start = time.perf_counter()
            controller.compute_torque(des_pos, des_vel, cur_pos, cur_vel, ext_wrench, jac)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = np.mean(latencies)
        print(f"\n  阻抗控制器延迟: avg={avg_latency:.3f}ms")
        self.assertLess(avg_latency, 1, "阻抗控制器延迟应小于1ms")

    def test_safety_controller_latency(self):
        """测试安全控制器延迟"""
        config = SafetyConfig(
            joint_limits_lower=np.array([-3.14] * 6),
            joint_limits_upper=np.array([3.14] * 6),
            velocity_limits=np.array([2.0] * 6),
            acceleration_limits=np.array([10.0] * 6),
        )
        controller = SafetyController(config)

        from control.safety_controller import JointStateSnapshot

        latencies = []
        for _ in range(1000):
            state = JointStateSnapshot(
                positions=np.random.randn(6),
                velocities=np.random.randn(6)
            )
            start = time.perf_counter()
            controller.check(state)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = np.mean(latencies)
        print(f"\n  安全控制器延迟: avg={avg_latency:.3f}ms")
        self.assertLess(avg_latency, 1, "安全控制器延迟应小于1ms")


class TestMemoryBenchmark(unittest.TestCase):
    """内存使用基准测试"""

    def test_fusion_memory_usage(self):
        """测试融合网络内存占用"""
        tracemalloc.start()

        config = FusionConfig(
            vision_dim=512, audio_dim=128,
            tactile_dim=64, force_dim=32, imu_dim=64,
            hidden_dim=256, num_heads=4
        )
        fusion = CrossModalFusion(config)

        gc.collect()
        snapshot1 = tracemalloc.take_snapshot()

        for _ in range(10):
            multimodal = MultimodalInput(
                vision=torch.randn(4, 512),
                audio=torch.randn(4, 128),
                tactile=torch.randn(4, 64),
                force=torch.randn(4, 32),
                imu=torch.randn(4, 64),
                language=None
            )
            fusion(multimodal)

        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()

        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_diff = sum(stat.size_diff for stat in top_stats[:5])

        tracemalloc.stop()
        memory_mb = total_diff / 1024 / 1024
        print(f"\n  融合网络内存增量: {memory_mb:.2f} MB (10次前向传播)")
        self.assertLess(abs(memory_mb), 500, "内存增长应在合理范围内")


if __name__ == '__main__':
    unittest.main(verbosity=2)
