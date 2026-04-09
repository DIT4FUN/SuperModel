"""
SuperModel 具身智能大脑集成测试
================================

测试完整具身感知→融合→控制 pipeline:
- 触觉 + 力觉 + IMU → 跨模态融合 → 具身控制
- AGV五级规格验证
- 端到端时延分析

v2.34.0
"""

import unittest
import numpy as np
import time

from src.sensors.tactile import (
    TactileArray, TactileFrame, TactileSensorType,
    VirtualTactileSensor
)
from src.sensors.force import (
    ForceTorqueSensor, Wrench, ForceSensorType,
    VirtualForceSensor
)
from src.sensors.imu import (
    IMUSensor, IMUFrame, PoseEstimator, IMUSensorType,
    VirtualIMUSensor
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
from src.fusion.cross_modal_fusion import (
    CrossModalFusion, MultimodalInput, FusionConfig,
    create_multimodal_input
)


GRADES = ['S', 'M', 'L', 'XL', 'XXL']


class TestEmbodiedSensorCapture(unittest.TestCase):
    """测试具身传感器捕获能力"""
    
    def test_tactile_capture_all_grades(self):
        """五级触觉传感器捕获"""
        sizes = {'S': (8, 8), 'M': (16, 16), 'L': (24, 24), 'XL': (32, 32), 'XXL': (48, 48)}
        
        for grade in GRADES:
            sensor = TactileArray(
                array_size=sizes[grade],
                sensor_type=TactileSensorType.CAPACITIVE,
                sensor_id=f"tactile_{grade}"
            )
            sensor.open()
            
            for _ in range(10):
                frame = sensor.capture()
                self.assertEqual(frame.pressure_map.shape, sizes[grade])
                self.assertIsNotNone(frame.temperature_map)
            
            sensor.close()
    
    def test_force_capture_all_grades(self):
        """五级力觉传感器捕获"""
        for grade in GRADES:
            sensor = ForceTorqueSensor(
                sensor_type=ForceSensorType.SIX_AXIS,
                sensor_id=f"force_{grade}"
            )
            sensor.open()
            
            for _ in range(10):
                wrench = sensor.capture()
                self.assertEqual(wrench.force.shape, (3,))
                self.assertEqual(wrench.torque.shape, (3,))
            
            sensor.close()
    
    def test_imu_capture_all_grades(self):
        """五级IMU传感器捕获"""
        sample_rates = {'S': 100, 'M': 200, 'L': 500, 'XL': 1000, 'XXL': 2000}
        
        for grade in GRADES:
            sensor = IMUSensor(
                sensor_type=IMUSensorType.BMI088,
                sensor_id=f"imu_{grade}",
                sample_rate=sample_rates[grade]
            )
            sensor.open()
            
            for _ in range(20):
                frame = sensor.capture()
                self.assertEqual(frame.accel.shape, (3,))
                self.assertEqual(frame.gyro.shape, (3,))
                self.assertGreater(frame.accel_magnitude, 0)
            
            sensor.close()


class TestEmbodiedSensorFusion(unittest.TestCase):
    """测试具身传感器融合"""
    
    def test_fusion_all_grades(self):
        """五级融合网络"""
        hidden_dims = {'S': 128, 'M': 256, 'L': 512, 'XL': 768, 'XXL': 1024}
        
        for grade in GRADES:
            config = FusionConfig(hidden_dim=hidden_dims[grade], num_heads=4)
            fusion = CrossModalFusion(config)
            
            for _ in range(5):
                multimodal = MultimodalInput(
                    vision=np.random.randn(1, 256).astype(np.float32),
                    tactile=np.random.randn(1, 64).astype(np.float32),
                    force=np.random.randn(1, 32).astype(np.float32),
                    imu=np.random.randn(1, 32).astype(np.float32)
                )
                out = fusion.forward(multimodal)
                self.assertEqual(out.shape[1], hidden_dims[grade])
    
    def test_fusion_latency_grade_scaling(self):
        """融合延迟随等级变化"""
        latencies = {}
        
        for grade in GRADES:
            config = FusionConfig(hidden_dim=256, num_heads=4)
            fusion = CrossModalFusion(config)
            
            # warmup
            for _ in range(5):
                multimodal = MultimodalInput(
                    vision=np.random.randn(1, 256).astype(np.float32),
                    tactile=np.random.randn(1, 64).astype(np.float32),
                    force=np.random.randn(1, 32).astype(np.float32),
                    imu=np.random.randn(1, 32).astype(np.float32)
                )
                fusion.forward(multimodal)
            
            times = []
            for _ in range(20):
                multimodal = MultimodalInput(
                    vision=np.random.randn(1, 256).astype(np.float32),
                    tactile=np.random.randn(1, 64).astype(np.float32),
                    force=np.random.randn(1, 32).astype(np.float32),
                    imu=np.random.randn(1, 32).astype(np.float32)
                )
                t0 = time.perf_counter()
                fusion.forward(multimodal)
                times.append((time.perf_counter() - t0) * 1000)
            
            latencies[grade] = np.mean(times)
        
        for grade, lat in latencies.items():
            self.assertLess(lat, 100, f"{grade} latency {lat:.2f}ms too high")
    
    def test_fusion_missing_modality(self):
        """缺失模态时的融合"""
        config = FusionConfig(hidden_dim=256)
        fusion = CrossModalFusion(config)
        
        # 只有触觉
        multimodal = MultimodalInput(
            tactile=np.random.randn(1, 64).astype(np.float32)
        )
        out = fusion.forward(multimodal)
        self.assertEqual(out.shape[1], 256)
        
        # 只有IMU
        multimodal2 = MultimodalInput(
            imu=np.random.randn(1, 32).astype(np.float32)
        )
        out2 = fusion.forward(multimodal2)
        self.assertEqual(out2.shape[1], 256)
        
        # 空的
        multimodal3 = MultimodalInput()
        out3 = fusion.forward(multimodal3)
        self.assertEqual(out3.shape[1], 256)


class TestEmbodiedControlLoop(unittest.TestCase):
    """测试具身控制闭环"""
    
    def test_tactile_servo_all_grades(self):
        """五级触觉伺服"""
        for grade in GRADES:
            params = TactileServoParams(grade=grade)
            
            tactile = TactileArray(array_size=(16, 16), sensor_id=f"ts_{grade}")
            tactile.open()
            frame = tactile.capture()
            
            ctrl = TactileServoController(tactile, params)
            
            # 获取控制输出
            cmd = ctrl.compute_control_signal(frame)
            self.assertIsNotNone(cmd)
            
            tactile.close()
    
    def test_force_control_all_grades(self):
        """五级力控制"""
        for grade in GRADES:
            force_sensor = ForceTorqueSensor(sensor_id=f"fc_{grade}")
            force_sensor.open()
            params = ForceControlParams(grade=grade)
            ctrl = ForceController(force_sensor, params)
            
            wrench = force_sensor.capture()
            desired = np.array([0.0, 0.0, 10.0])
            
            # 导纳控制
            vel = ctrl.compute_admittance(desired, wrench)
            self.assertEqual(len(vel), 3)
            
            force_sensor.close()
    
    def test_imu_stabilizer_all_grades(self):
        """五级IMU姿态稳定"""
        for grade in GRADES:
            imu = IMUSensor(sample_rate=200, sensor_id=f"ims_{grade}")
            imu.open()
            params = IMUControlParams(grade=grade)
            stabilizer = AttitudeStabilizer(imu, params)
            
            frame = imu.capture()
            stabilizer.set_target_attitude(0.0, 0.0, 0.0)
            
            cmd = stabilizer.update(frame, dt=0.01)
            self.assertEqual(cmd.shape, (3,))
            
            imu.close()


class TestEmbodiedPipelineEndToEnd(unittest.TestCase):
    """端到端具身 pipeline 测试"""
    
    def test_full_pipeline_all_grades(self):
        """完整 pipeline: 感知→融合→控制"""
        hidden_dims = {'S': 128, 'M': 256, 'L': 512, 'XL': 768, 'XXL': 1024}
        
        for grade in GRADES:
            # 初始化传感器
            tactile = TactileArray(array_size=(16, 16), sensor_id=f"tp_{grade}")
            tactile.open()
            
            force = ForceTorqueSensor(sensor_id=f"fp_{grade}")
            force.open()
            
            imu = IMUSensor(sample_rate=200, sensor_id=f"ip_{grade}")
            imu.open()
            
            # 融合
            fusion_cfg = FusionConfig(hidden_dim=hidden_dims[grade])
            fusion = CrossModalFusion(fusion_cfg)
            
            # 控制器
            t_ctrl = TactileServoController(tactile, TactileServoParams(grade=grade))
            f_ctrl = ForceController(force, ForceControlParams(grade=grade))
            i_ctrl = AttitudeStabilizer(imu, IMUControlParams(grade=grade))
            
            # 采集
            t_frame = tactile.capture()
            f_wrench = force.capture()
            i_frame = imu.capture()
            
            # 融合 (使用编码特征)
            fused = fusion.forward(MultimodalInput(
                tactile=t_frame.pressure_map.flatten()[None,:].astype(np.float32) * 10,
                force=f_wrench.to_vector()[None,:].astype(np.float32),
                imu=np.concatenate([i_frame.accel, i_frame.gyro])[None,:].astype(np.float32)
            ))
            
            # 控制
            t_cmd = t_ctrl.compute_control_signal(t_frame)
            f_cmd = f_ctrl.compute_admittance(np.array([0, 0, 5]), f_wrench)
            i_cmd = i_ctrl.update(i_frame, dt=0.01)
            
            # 验证
            self.assertEqual(fused.shape[1], hidden_dims[grade])
            self.assertIsNotNone(t_cmd)
            self.assertEqual(len(f_cmd), 3)
            self.assertEqual(i_cmd.shape, (3,))
            
            tactile.close()
            force.close()
            imu.close()
    
    def test_pipeline_timing_all_grades(self):
        """Pipeline 时序分析"""
        timings = {}
        
        for grade in GRADES:
            tactile = TactileArray(array_size=(16, 16), sensor_id=f"tt_{grade}")
            tactile.open()
            force = ForceTorqueSensor(sensor_id=f"ft_{grade}")
            force.open()
            imu = IMUSensor(sample_rate=200, sensor_id=f"it_{grade}")
            imu.open()
            fusion = CrossModalFusion(FusionConfig(hidden_dim=256))
            
            times = {'capture': [], 'fusion': [], 'control': [], 'total': []}
            
            for _ in range(30):
                t0 = time.perf_counter()
                
                t1 = time.perf_counter()
                t_frame = tactile.capture()
                f_wrench = force.capture()
                i_frame = imu.capture()
                times['capture'].append((time.perf_counter() - t1) * 1000)
                
                t2 = time.perf_counter()
                inp = MultimodalInput(
                    tactile=t_frame.pressure_map.flatten()[None,:].astype(np.float32) * 10,
                    force=f_wrench.to_vector()[None,:].astype(np.float32),
                    imu=np.concatenate([i_frame.accel, i_frame.gyro])[None,:].astype(np.float32)
                )
                fusion.forward(inp)
                times['fusion'].append((time.perf_counter() - t2) * 1000)
                
                times['total'].append((time.perf_counter() - t0) * 1000)
            
            timings[grade] = {k: np.mean(v) for k, v in times.items()}
            
            tactile.close()
            force.close()
            imu.close()
        
        for grade, t in timings.items():
            self.assertLess(t['capture'], 20, f"{grade} capture too slow: {t['capture']:.2f}ms")
            self.assertLess(t['fusion'], 100, f"{grade} fusion too slow: {t['fusion']:.2f}ms")
            self.assertLess(t['total'], 150, f"{grade} total too slow: {t['total']:.2f}ms")


class TestEmbodiedFaultTolerance(unittest.TestCase):
    """容错测试"""
    
    def test_missing_modality_graceful_degradation(self):
        """缺失模态时优雅降级"""
        fusion = CrossModalFusion(FusionConfig(hidden_dim=256))
        
        # 只有触觉
        multimodal = MultimodalInput(
            tactile=np.random.randn(1, 64).astype(np.float32)
        )
        out = fusion.forward(multimodal)
        self.assertIsNotNone(out)
        self.assertEqual(out.shape[1], 256)
        
        # 只有IMU
        multimodal2 = MultimodalInput(
            imu=np.random.randn(1, 32).astype(np.float32)
        )
        out2 = fusion.forward(multimodal2)
        self.assertIsNotNone(out2)
        
        # 全空
        multimodal3 = MultimodalInput()
        out3 = fusion.forward(multimodal3)
        self.assertIsNotNone(out3)
    
    def test_sensor_noise_resilience(self):
        """噪声鲁棒性"""
        fusion = CrossModalFusion(FusionConfig(hidden_dim=256))
        
        outputs = []
        for _ in range(50):
            multimodal = MultimodalInput(
                tactile=np.random.randn(1, 64).astype(np.float32),
                force=np.random.randn(1, 32).astype(np.float32),
                imu=np.random.randn(1, 32).astype(np.float32)
            )
            out = fusion.forward(multimodal)
            outputs.append(out.mean())
        
        std = np.std(outputs)
        self.assertLess(std, 0.5, f"Output too noisy: std={std:.4f}")


if __name__ == '__main__':
    unittest.main()
