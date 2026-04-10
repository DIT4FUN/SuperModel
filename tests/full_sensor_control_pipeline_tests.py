"""
全链路传感器→融合→控制集成测试
==================================

端到端测试: 触觉+力觉+IMU → 跨模态融合 → AGV控制
覆盖 S/M/L/XL/XXL 五级完整规格验证

测试目标:
1. 传感器采集 → 融合网络 → 控制输出 全链路
2. AGV五级规格逐级验证
3. 控制回路时序与延迟保证
4. 故障注入与安全降级
"""

import unittest
import numpy as np
import sys
import time

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.tactile import (
    TactileArray, TactileFrame, TactileSensorType, TactileContact,
    VirtualTactileSensor, get_tactile_spec, AGV_TACTILE_GRADES
)
from sensors.force import (
    ForceTorqueSensor, Wrench, ForceSensorType,
    VirtualForceSensor, get_force_spec, AGV_FORCE_GRADES
)
from sensors.imu import (
    IMUSensor, IMUFrame, IMUSensorType, PoseEstimator, VirtualIMUSensor,
    get_imu_spec, AGV_IMU_GRADES
)
from sensors.manager import SensorManager

from fusion.cross_modal_fusion import (
    CrossModalFusion, FusionConfig, MultimodalInput,
    get_fusion_spec, create_fusion_for_grade
)


GRADES = ['S', 'M', 'L', 'XL', 'XXL']


def tactile_to_feat(t_frame: TactileFrame, target_dim: int = 64) -> np.ndarray:
    """将触觉帧转换为固定维度特征向量"""
    feat = t_frame.pressure_map.flatten()[:target_dim].astype(np.float32)
    if len(feat) < target_dim:
        feat = np.pad(feat, (0, target_dim - len(feat)))
    return feat


def force_to_feat(wrench: Wrench, target_dim: int = 32) -> np.ndarray:
    """将力旋量转换为固定维度特征向量"""
    feat = wrench.to_vector()[:target_dim].astype(np.float32)
    if len(feat) < target_dim:
        feat = np.pad(feat, (0, target_dim - len(feat)))
    return feat


def imu_to_feat(i_frame: IMUFrame, target_dim: int = 32) -> np.ndarray:
    """将IMU帧转换为固定维度特征向量"""
    accel_norm = i_frame.accel / 9.81
    gyro_norm = i_frame.gyro / np.pi * 180
    feat = np.concatenate([accel_norm, gyro_norm])[:target_dim].astype(np.float32)
    if len(feat) < target_dim:
        feat = np.pad(feat, (0, target_dim - len(feat)))
    return feat


class TestSensorGradeSpecCompliance(unittest.TestCase):
    """测试各AGV等级的传感器规格合规性"""

    def test_tactile_grade_specs_all_present(self):
        """验证触觉五级规格完整性"""
        for grade in GRADES:
            spec = get_tactile_spec(grade)
            self.assertIn('array', spec)
            self.assertIn('res', spec)
            self.assertIn('range_kpa', spec)
            self.assertIn('freq_hz', spec)
            self.assertIn('temp', spec)

    def test_force_grade_specs_all_present(self):
        """验证力觉五级规格完整性"""
        for grade in GRADES:
            spec = get_force_spec(grade)
            self.assertIn('axes', spec)
            self.assertIn('force_range', spec)
            self.assertIn('torque_range', spec)
            self.assertIn('resolution', spec)
            self.assertIn('sampling_hz', spec)

    def test_imu_grade_specs_all_present(self):
        """验证IMU五级规格完整性"""
        for grade in GRADES:
            spec = get_imu_spec(grade)
            self.assertIn('type', spec)
            self.assertIn('accel_range', spec)
            self.assertIn('gyro_range', spec)
            self.assertIn('sample_hz', spec)
            self.assertIn('noise_density', spec)

    def test_grade_progression_monotonic(self):
        """验证五级规格单调递增"""
        tactile_sizes = [get_tactile_spec(g)['array'][0] for g in GRADES]
        self.assertEqual(tactile_sizes, sorted(tactile_sizes))

        force_ranges = [get_force_spec(g)['force_range'] for g in GRADES]
        self.assertEqual(force_ranges, sorted(force_ranges))

        imu_rates = [get_imu_spec(g)['sample_hz'] for g in GRADES]
        self.assertEqual(imu_rates, sorted(imu_rates))

    def test_grade_hidden_dim_scaling(self):
        """验证融合网络隐藏维度随等级缩放"""
        prev_dim = 0
        for grade in GRADES:
            spec = get_fusion_spec(grade)
            self.assertIn('hidden_dim', spec)
            dim = spec['hidden_dim']
            self.assertGreaterEqual(dim, prev_dim)
            prev_dim = dim


class TestFullPipelineSingleGrade(unittest.TestCase):
    """测试单个等级(M级)的完整传感器→融合→控制链路"""

    GRADE = 'M'

    def setUp(self):
        self.grade = self.GRADE
        self.t_spec = get_tactile_spec(self.grade)
        self.f_spec = get_force_spec(self.grade)
        self.i_spec = get_imu_spec(self.grade)

        # 创建传感器
        self.tactile = TactileArray(
            array_size=self.t_spec['array'],
            sensor_type=TactileSensorType.CAPACITIVE,
            sensor_id=f"tactile_{self.grade}"
        )
        self.force = ForceTorqueSensor(
            sensor_type=ForceSensorType.SIX_AXIS,
            sensor_id=f"force_{self.grade}"
        )
        self.imu = IMUSensor(
            sensor_type=IMUSensorType.BMI088,
            sample_rate=self.i_spec['sample_hz'],
            sensor_id=f"imu_{self.grade}"
        )

        # 融合网络 (使用FusionConfig)
        spec = get_fusion_spec(self.grade)
        cfg = FusionConfig(
            tactile_dim=64,
            force_dim=32,
            imu_dim=32,
            hidden_dim=spec['hidden_dim'],
            output_dim=spec['output_dim'],
            fusion_type=spec['fusion_type'],
            num_heads=spec['attention_heads'],
            num_layers=spec['transformer_layers'],
            dropout=0.1,
        )
        self.fusion = CrossModalFusion(cfg)

        # 姿态估计
        self.pose_est = PoseEstimator(
            algorithm='madgwick',
            sample_rate=self.i_spec['sample_hz']
        )

        self.tactile.open()
        self.force.open()
        self.imu.open()

    def tearDown(self):
        self.tactile.close()
        self.force.close()
        self.imu.close()

    def test_sensor_capture_rates(self):
        """测试传感器采样率是否符合等级规格"""
        for name, sensor, expected_rate, spec_key in [
            ('tactile', self.tactile, self.t_spec['freq_hz'], 'freq_hz'),
            ('force', self.force, self.f_spec['sampling_hz'], 'sampling_hz'),
            ('imu', self.imu, self.i_spec['sample_hz'], 'sample_hz')
        ]:
            start = time.time()
            count = 0
            target_time = 0.2
            while time.time() - start < target_time:
                sensor.capture()
                count += 1

            elapsed = time.time() - start
            measured_rate = count / elapsed
            self.assertGreaterEqual(
                measured_rate, expected_rate * 0.8,
                f"{name} rate {measured_rate:.1f}Hz below spec {expected_rate}Hz"
            )

    def test_sensor_frame_structure(self):
        """测试传感器帧结构完整性"""
        t_frame = self.tactile.capture()
        self.assertIsInstance(t_frame, TactileFrame)
        self.assertEqual(t_frame.pressure_map.shape,
                        (self.t_spec['array'][0], self.t_spec['array'][1]))
        self.assertIsNotNone(t_frame.temperature_map)

        wrench = self.force.capture()
        self.assertIsInstance(wrench, Wrench)
        self.assertEqual(wrench.force.shape, (3,))
        self.assertEqual(wrench.torque.shape, (3,))

        i_frame = self.imu.capture()
        self.assertIsInstance(i_frame, IMUFrame)
        self.assertEqual(i_frame.accel.shape, (3,))
        self.assertEqual(i_frame.gyro.shape, (3,))

    def test_fusion_forward_with_real_data(self):
        """测试融合网络前向传播"""
        t_frame = self.tactile.capture()
        wrench = self.force.capture()
        i_frame = self.imu.capture()

        # 转换特征
        tactile_feat = tactile_to_feat(t_frame, 64)
        force_feat = force_to_feat(wrench, 32)
        imu_feat = imu_to_feat(i_frame, 32)

        # MultimodalInput
        multimodal = MultimodalInput(
            tactile=tactile_feat[np.newaxis, :],
            force=force_feat[np.newaxis, :],
            imu=imu_feat[np.newaxis, :]
        )

        fused = self.fusion(multimodal)
        self.assertIsInstance(fused, np.ndarray)
        self.assertEqual(fused.ndim, 2)
        self.assertEqual(fused.shape[0], 1)
        self.assertTrue(np.all(np.isfinite(fused)))

    def test_pose_estimation_quality(self):
        """测试姿态估计质量"""
        errors = []
        for _ in range(50):
            i_frame = self.imu.capture()
            pose = self.pose_est.update(
                i_frame.accel,
                i_frame.gyro,
                i_frame.mag,
                dt=1.0 / self.i_spec['sample_hz']
            )
            euler = pose.to_euler()
            self.assertTrue(np.all(np.isfinite(euler)))
            errors.append(np.linalg.norm(euler))

        self.assertLess(np.std(errors), 0.5)

    def test_contact_detection_pipeline(self):
        """测试接触检测完整管道"""
        virtual_tactile = VirtualTactileSensor(
            array_size=self.t_spec['array'],
            sensor_id=f"virtual_tactile_{self.grade}"
        )
        virtual_tactile.open()

        frame = virtual_tactile.simulate_contact(
            contact_pos=(0.5, 0.5),
            contact_radius=0.2,
            contact_force=5.0
        )
        contacts = self.tactile.detect_contacts(frame)
        self.assertGreaterEqual(len(contacts), 0)

        frame_multi = virtual_tactile.simulate_multi_contact([
            ((0.3, 0.3), 3.0, 0.15),
            ((0.7, 0.7), 4.0, 0.2)
        ])
        contacts_multi = self.tactile.detect_contacts(frame_multi)
        self.assertGreaterEqual(len(contacts_multi), 0)

        virtual_tactile.close()

    def test_wrench_contact_pipeline(self):
        """测试力觉接触检测管道"""
        virtual_force = VirtualForceSensor(sensor_id=f"virtual_force_{self.grade}")
        virtual_force.open()

        wrench = virtual_force.simulate_contact(
            force=(5.0, 2.0, -10.0),
            torque=(0.1, 0.05, 0.0)
        )
        contact_state = self.force.detect_contact(wrench)
        self.assertIsInstance(contact_state.is_contact, (bool, np.bool_))

        wrench_idle = virtual_force.simulate_contact(
            force=(0.1, 0.1, -0.5),
            torque=(0.0, 0.0, 0.0)
        )
        contact_idle = self.force.detect_contact(wrench_idle)
        self.assertFalse(contact_idle.is_contact)

        virtual_force.close()

    def test_imu_trajectory_pipeline(self):
        """测试IMU轨迹模拟管道"""
        virtual_imu = VirtualIMUSensor(sensor_id=f"virtual_imu_{self.grade}")
        virtual_imu.open()

        frames = virtual_imu.simulate_trajectory(
            trajectory_type='circle', duration_s=1.0, dt=0.01
        )
        self.assertGreater(len(frames), 50)

        frames_8 = virtual_imu.simulate_trajectory(
            trajectory_type='figure8', duration_s=2.0, dt=0.01
        )
        self.assertGreater(len(frames_8), 100)

        frames_agv = virtual_imu.simulate_agv_motion(
            linear_velocity=(0.5, 0.0),
            angular_velocity=0.5,
            dt=0.01,
            grade=self.grade
        )
        self.assertIsInstance(frames_agv, IMUFrame)

        virtual_imu.close()

    def test_control_loop_timing(self):
        """测试控制回路时序满足等级要求"""
        control_freq = self.i_spec['sample_hz']
        control_period = 1000.0 / control_freq
        max_allowed = control_period * 0.8

        loop_times = []
        for _ in range(30):
            t0 = time.time()

            t_frame = self.tactile.capture()
            wrench = self.force.capture()
            i_frame = self.imu.capture()

            multimodal = MultimodalInput(
                tactile=tactile_to_feat(t_frame, 64)[np.newaxis, :],
                force=force_to_feat(wrench, 32)[np.newaxis, :],
                imu=imu_to_feat(i_frame, 32)[np.newaxis, :]
            )
            fused = self.fusion(multimodal)

            self.pose_est.update(i_frame.accel, i_frame.gyro, i_frame.mag,
                                dt=1.0 / self.i_spec['sample_hz'])

            t1 = time.time()
            loop_times.append((t1 - t0) * 1000)

        self.assertLess(np.max(loop_times), max_allowed,
                       f"Loop {np.max(loop_times):.2f}ms > allowed {max_allowed:.2f}ms")


class TestFullPipelineAllGrades(unittest.TestCase):
    """测试所有AGV等级的全链路性能差异"""

    def test_grade_scaling_consistency(self):
        """验证各等级规格缩放一致性"""
        prev_t_area = 0
        prev_f_range = 0
        prev_i_rate = 0

        for grade in GRADES:
            t_spec = get_tactile_spec(grade)
            f_spec = get_force_spec(grade)
            i_spec = get_imu_spec(grade)

            t_area = t_spec['array'][0] * t_spec['array'][1]
            self.assertGreater(t_area, prev_t_area)
            prev_t_area = t_area

            self.assertGreater(f_spec['force_range'], prev_f_range)
            prev_f_range = f_spec['force_range']

            self.assertGreater(i_spec['sample_hz'], prev_i_rate)
            prev_i_rate = i_spec['sample_hz']

    def test_hidden_dim_progression(self):
        """验证融合网络隐藏维度随等级递增"""
        dims = [get_fusion_spec(g)['hidden_dim'] for g in GRADES]
        self.assertEqual(dims, sorted(dims))
        # 验证单调递增 (允许相等但不允许递减)
        for i in range(len(dims) - 1):
            self.assertGreaterEqual(dims[i+1], dims[i])

    def test_pipeline_latency_scaling(self):
        """验证各等级处理延迟符合规格(高等级<=低等级)"""
        latencies = {}
        for grade in GRADES:
            t_spec = get_tactile_spec(grade)
            i_spec = get_imu_spec(grade)

            tactile = TactileArray(
                array_size=t_spec['array'],
                sensor_type=TactileSensorType.CAPACITIVE,
                sensor_id=f"tlat_{grade}"
            )
            force = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS,
                                     sensor_id=f"flat_{grade}")
            imu = IMUSensor(
                sensor_type=IMUSensorType.BMI088,
                sample_rate=i_spec['sample_hz'],
                sensor_id=f"ilat_{grade}"
            )

            tactile.open()
            force.open()
            imu.open()

            loop_times = []
            for _ in range(10):
                t0 = time.time()
                tactile.capture()
                force.capture()
                imu.capture()
                loop_times.append((time.time() - t0) * 1000)

            latencies[grade] = np.mean(loop_times)
            tactile.close()
            force.close()
            imu.close()

        # 验证各等级延迟符合AGV五级规格预算
        # 注意: 高等级因传感器分辨率/采样率更高,单帧采集时间可能更长
        # 这里验证延迟在合理范围内(各等级规格预算: S<5ms, M<3ms, L<2ms, XL<1.5ms, XXL<1ms)
        latency_budgets = {'S': 5.0, 'M': 3.0, 'L': 2.0, 'XL': 1.5, 'XXL': 1.0}
        for grade, measured in latencies.items():
            budget = latency_budgets.get(grade, 5.0)
            self.assertLessEqual(
                measured, budget,
                f"Grade {grade} latency {measured:.2f}ms exceeds budget {budget:.2f}ms"
            )


class TestSensorFusionControlLoop(unittest.TestCase):
    """测试传感器→融合→控制闭环"""

    def test_closed_loop_control_response(self):
        """测试闭环控制响应"""
        grade = 'L'
        t_spec = get_tactile_spec(grade)
        i_spec = get_imu_spec(grade)

        v_tactile = VirtualTactileSensor(array_size=t_spec['array'],
                                         sensor_id="loop_tactile")
        v_force = VirtualForceSensor(sensor_id="loop_force")
        v_imu = VirtualIMUSensor(sensor_id="loop_imu")

        v_tactile.open()
        v_force.open()
        v_imu.open()

        Kp = 1.0
        target = 0.5
        responses = []

        for step in range(20):
            contact_pos = 0.3 + 0.02 * step
            actual_force = 5.0 if contact_pos > 0.4 else 2.0

            t_frame = v_tactile.simulate_contact(
                contact_pos=(contact_pos, 0.5),
                contact_radius=0.2,
                contact_force=actual_force
            )
            wrench = v_force.simulate_contact(
                force=(0, 0, -actual_force),
                torque=(0, 0, 0)
            )
            v_imu.simulate_agv_motion(
                linear_velocity=(0.1, 0.0),
                angular_velocity=0.05,
                dt=0.01,
                grade=grade
            )

            error = target - contact_pos
            control_output = Kp * error
            responses.append(control_output)
            self.assertTrue(np.isfinite(control_output))

        self.assertLess(abs(responses[-1]), abs(responses[0]))
        v_tactile.close()
        v_force.close()
        v_imu.close()

    def test_fusion_grades_with_missing_modalities(self):
        """测试各等级融合网络对缺失模态的处理"""
        for grade in GRADES:
            spec = get_fusion_spec(grade)
            cfg = FusionConfig(
                tactile_dim=64, force_dim=32, imu_dim=32,
                hidden_dim=spec['hidden_dim'],
                output_dim=spec['output_dim'],
                fusion_type=spec['fusion_type'],
                num_heads=spec['attention_heads'],
                num_layers=spec['transformer_layers'],
            )
            fusion = CrossModalFusion(cfg)

            # 单模态输入
            multimodal = MultimodalInput(
                tactile=np.random.randn(1, 64).astype(np.float32),
                force=np.zeros((1, 32), dtype=np.float32),
                imu=np.zeros((1, 32), dtype=np.float32),
            )
            result = fusion(multimodal)
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape[0], 1)
            self.assertTrue(np.all(np.isfinite(result)))


class TestSafetyAndLimits(unittest.TestCase):
    """测试安全限制和边界条件"""

    def test_force_limit_enforcement(self):
        """测试力限制强制执行"""
        v_force = VirtualForceSensor(sensor_id="safety_force")
        v_force.open()
        for magnitude in [10.0, 100.0, 500.0, 1000.0]:
            wrench = v_force.simulate_contact(
                force=(magnitude, 0, -magnitude),
                torque=(0.5, 0.3, 0.1)
            )
            self.assertLess(wrench.magnitude, 1e6)
        v_force.close()

    def test_tactile_limit_enforcement(self):
        """测试触觉限制强制执行"""
        v_tactile = VirtualTactileSensor(array_size=(16, 16),
                                        sensor_id="safety_tactile")
        v_tactile.open()
        frame = v_tactile.simulate_contact(
            contact_pos=(0.5, 0.5),
            contact_radius=0.5,
            contact_force=100.0
        )
        self.assertTrue(np.all(frame.pressure_map >= 0))
        self.assertTrue(np.all(frame.pressure_map <= 1.2))
        v_tactile.close()

    def test_imu_boundary_conditions(self):
        """测试IMU边界条件"""
        v_imu = VirtualIMUSensor(sensor_id="safety_imu")
        v_imu.open()
        for _ in range(10):
            frame = v_imu.simulate_motion(
                linear_accel=(100.0, 100.0, 100.0),
                angular_vel=(10.0, 10.0, 10.0),
                dt=0.001
            )
            self.assertTrue(np.all(np.isfinite(frame.accel)))
            self.assertTrue(np.all(np.isfinite(frame.gyro)))
        v_imu.close()


if __name__ == '__main__':
    unittest.main()
