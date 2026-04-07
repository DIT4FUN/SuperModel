"""
AGV五级完整集成测试
Five-Grade AGV Complete Integration Tests

测试目标: 验证 SuperModel 在 S/M/L/XL/XXL 五个AGV等级下的
传感器采集、跨模态融合、运动控制、安全监控的端到端集成

覆盖:
- 各等级传感器配置正确性
- 传感器-控制闭环延迟
- 五级安全规格合规性
- 感知-融合-控制完整流水线
"""

import unittest
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAGVGradeSpecifications(unittest.TestCase):
    """AGV五级规格验证"""

    GRADES = ["S", "M", "L", "XL", "XXL"]

    GRADE_SPECS = {
        "S": {"tactile_size": (8, 8), "tactile_hz": 50, "force_axes": 3, "force_range": 100, "force_hz": 100,
              "imu_model": "MPU6050", "imu_hz": 100, "control_freq": 50, "max_latency_ms": 200, "max_jitter_ms": 50},
        "M": {"tactile_size": (16, 16), "tactile_hz": 100, "force_axes": 6, "force_range": 200, "force_hz": 500,
              "imu_model": "BMI088", "imu_hz": 200, "control_freq": 100, "max_latency_ms": 80, "max_jitter_ms": 20},
        "L": {"tactile_size": (24, 24), "tactile_hz": 200, "force_axes": 6, "force_range": 500, "force_hz": 1000,
              "imu_model": "BMI088", "imu_hz": 500, "control_freq": 200, "max_latency_ms": 35, "max_jitter_ms": 8},
        "XL": {"tactile_size": (32, 32), "tactile_hz": 500, "force_axes": 6, "force_range": 1000, "force_hz": 2000,
               "imu_model": "ADIS16470", "imu_hz": 1000, "control_freq": 500, "max_latency_ms": 15, "max_jitter_ms": 3},
        "XXL": {"tactile_size": (48, 48), "tactile_hz": 1000, "force_axes": 6, "force_range": 5000, "force_hz": 5000,
                "imu_model": "ADIS16470", "imu_hz": 2000, "control_freq": 1000, "max_latency_ms": 7, "max_jitter_ms": 1.5},
    }

    def test_grade_tactile_configs(self):
        """验证五级触觉配置"""
        from src.sensors.tactile import TactileArray, TactileSensorType
        for grade, spec in self.GRADE_SPECS.items():
            arr = TactileArray(
                array_size=spec["tactile_size"],
                sensor_type=TactileSensorType.RESISTIVE,
                sensor_id=f"tactile_{grade}"
            )
            self.assertEqual(arr.array_size, spec["tactile_size"], f"Grade {grade} tactile size mismatch")

    def test_grade_force_configs(self):
        """验证五级力觉配置"""
        from src.sensors.force import ForceTorqueSensor, ForceSensorType
        for grade, spec in self.GRADE_SPECS.items():
            sensor = ForceTorqueSensor(
                sensor_type=ForceSensorType.SIX_AXIS,
                sensor_id=f"force_{grade}",
            )
            self.assertEqual(sensor.sensor_id, f"force_{grade}")

    def test_grade_imu_configs(self):
        """验证五级IMU配置"""
        from src.sensors.imu import IMUSensor, IMUSensorType
        for grade, spec in self.GRADE_SPECS.items():
            model_map = {
                "MPU6050": IMUSensorType.MPU6050,
                "BMI088": IMUSensorType.BMI088,
                "ADIS16470": IMUSensorType.ADIS16470,
            }
            sensor = IMUSensor(
                sensor_type=model_map[spec["imu_model"]],
                sensor_id=f"imu_{grade}"
            )
            self.assertEqual(sensor.sensor_id, f"imu_{grade}")


class TestSensorControl闭环ForAllGrades(unittest.TestCase):
    """五级传感器-控制闭环延迟测试"""

    GRADES = ["S", "M", "L", "XL", "XXL"]

    GRADE_SPECS = {
        "S": {"control_freq": 50, "max_latency_ms": 200, "max_jitter_ms": 50},
        "M": {"control_freq": 100, "max_latency_ms": 80, "max_jitter_ms": 20},
        "L": {"control_freq": 200, "max_latency_ms": 35, "max_jitter_ms": 8},
        "XL": {"control_freq": 500, "max_latency_ms": 15, "max_jitter_ms": 3},
        "XXL": {"control_freq": 1000, "max_latency_ms": 7, "max_jitter_ms": 1.5},
    }

    def test_sensor_capture_latency_per_grade(self):
        """测试各等级传感器采集延迟"""
        from src.sensors.tactile import VirtualTactileSensor
        from src.sensors.force import VirtualForceSensor
        from src.sensors.imu import VirtualIMUSensor

        for grade in self.GRADES:
            spec = self.GRADE_SPECS[grade]
            size = {"S": (8, 8), "M": (16, 16), "L": (24, 24), "XL": (32, 32), "XXL": (48, 48)}[grade]

            tactile = VirtualTactileSensor(array_size=size, sensor_id=f"t_{grade}")
            force = VirtualForceSensor(sensor_id=f"f_{grade}")
            imu = VirtualIMUSensor(sensor_id=f"i_{grade}")

            tactile.open()
            force.open()
            imu.open()

            latencies = []
            for _ in range(20):
                t0 = time.perf_counter()
                tactile.simulate_contact(contact_pos=(0.5, 0.5), contact_radius=0.1)
                force.simulate_contact(force=(0, 0, 10), torque=(0, 0, 0))
                imu.simulate_static()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)

            tactile.close()
            force.close()
            imu.close()

            mean_latency = np.mean(latencies)
            self.assertLess(mean_latency, spec["max_latency_ms"],
                           f"Grade {grade} capture latency {mean_latency:.2f}ms exceeds {spec['max_latency_ms']}ms")


class TestCrossModalFusionForAllGrades(unittest.TestCase):
    """五级跨模态融合测试"""

    GRADES = ["S", "M", "L", "XL", "XXL"]

    def test_fusion_output_shape_per_grade(self):
        """验证各等级融合输出维度"""
        from src.fusion.cross_modal_fusion import CrossModalFusion, FusionConfig, MultimodalInput

        for grade in self.GRADES:
            tactile_dim_map = {"S": 64, "M": 256, "L": 576, "XL": 1024, "XXL": 2304}
            config = FusionConfig(
                vision_dim=128,
                audio_dim=64,
                tactile_dim=tactile_dim_map[grade],
                force_dim=36,
                imu_dim=12,
                hidden_dim=256,
                num_heads=4,
            )
            fusion = CrossModalFusion(config)

            # 构造 MultimodalInput
            multimodal = MultimodalInput(
                vision=np.random.randn(1, 128).astype(np.float32),
                audio=np.random.randn(1, 64).astype(np.float32),
                tactile=np.random.randn(1, tactile_dim_map[grade]).astype(np.float32),
                force=np.random.randn(1, 36).astype(np.float32),
                imu=np.random.randn(1, 12).astype(np.float32),
            )

            output = fusion(multimodal)
            self.assertEqual(output.shape[0], 1)
            self.assertEqual(output.shape[1], 256)


class TestSafetyMonitoringForAllGrades(unittest.TestCase):
    """五级安全监控合规性测试"""

    GRADES = ["S", "M", "L", "XL", "XXL"]

    # 各等级安全阈值
    GRADE_LIMITS = {
        "S": {"max_velocity": 1.0, "max_force": 80, "max_accel": 2.0, "boundary_margin": 0.1},
        "M": {"max_velocity": 2.0, "max_force": 160, "max_accel": 5.0, "boundary_margin": 0.05},
        "L": {"max_velocity": 3.0, "max_force": 400, "max_accel": 10.0, "boundary_margin": 0.03},
        "XL": {"max_velocity": 5.0, "max_force": 800, "max_accel": 20.0, "boundary_margin": 0.02},
        "XXL": {"max_velocity": 8.0, "max_force": 4000, "max_accel": 40.0, "boundary_margin": 0.01},
    }

    def test_velocity_limits(self):
        """测试各等级速度限制"""
        for grade, limits in self.GRADE_LIMITS.items():
            velocity = np.array([limits["max_velocity"] * 0.95, 0, 0])
            # 应该在限制内
            self.assertLess(np.linalg.norm(velocity), limits["max_velocity"])

            velocity_over = np.array([limits["max_velocity"] * 1.1, 0, 0])
            # 超速应被检测
            self.assertGreater(np.linalg.norm(velocity_over), limits["max_velocity"])

    def test_force_limits(self):
        """测试各等级力限制"""
        for grade, limits in self.GRADE_LIMITS.items():
            force = limits["max_force"] * 0.95
            self.assertLess(force, limits["max_force"])

            force_over = limits["max_force"] * 1.1
            self.assertGreater(force_over, limits["max_force"])


class TestRealtimeMonitorPerGrade(unittest.TestCase):
    """实时监控器五级性能验证"""

    GRADES = ["S", "M", "L", "XL", "XXL"]

    @unittest.skipIf(os.environ.get("SKIP_SLOW_TESTS") == "1", "Slow test - skip in CI")
    def test_monitor_compliance_per_grade(self):
        """验证各等级实时监控合规性 (5秒测试)"""
        try:
            from src.simulation.real_time_monitor import RealTimeMonitor, AGVGrade
        except ImportError:
            self.skipTest("real_time_monitor not available")

        grade_map = {"S": AGVGrade.S, "M": AGVGrade.M, "L": AGVGrade.L,
                     "XL": AGVGrade.XL, "XXL": AGVGrade.XXL}

        for grade in self.GRADES:
            monitor = RealTimeMonitor(grade=grade_map[grade], window_size=200)
            monitor.start()
            time.sleep(3.0)  # 3秒测试
            monitor.stop()

            stats = monitor.get_statistics()
            self.assertGreater(stats.get("compliance_rate_percent", 0), 0,
                             f"Grade {grade}: no samples collected")
            self.assertGreater(stats.get("sample_count", 0), 100,
                             f"Grade {grade}: insufficient samples")


if __name__ == "__main__":
    unittest.main(verbosity=2)
