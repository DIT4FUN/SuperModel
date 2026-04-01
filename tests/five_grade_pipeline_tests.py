"""
AGV五级完整流水线测试
=====================

测试所有五级 (S/M/L/XL/XXL) 传感器-控制完整流水线:
- 触觉 + 控制 (TactileServoController)
- 力觉 + 控制 (ForceController)
- IMU + 控制 (AttitudeStabilizer)
- 跨模态融合 + 联合控制
- 多传感器协同 + 安全检查

覆盖:
- 各等级传感器初始化和采集
- 各等级控制器参数配置
- 传感器-控制器数据流
- 五级安全限幅检查
- 端到端感知-控制闭环
"""

import numpy as np
import sys
import unittest

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.tactile import (
    TactileArray, TactileFrame, TactileContact,
    TactileSensorType, get_tactile_spec
)
from sensors.force import (
    ForceTorqueSensor, Wrench,
    ForceSensorType, get_force_spec
)
from sensors.imu import (
    IMUSensor, IMUFrame, Pose, PoseEstimator,
    IMUSensorType, get_imu_spec
)
from control.tactile_control import (
    TactileServoController, TactileServoParams, GraspQualityController,
    get_tactile_control_spec
)
from control.force_control import (
    ForceController, ForceControlParams, HybridForcePositionController,
)
from control.imu_control import (
    AttitudeStabilizer, IMUControlParams, MotionEstimator,
)
from control.safety_controller import (
    SafetyController, SafetyConfig, SafetyLevel
)
from fusion.cross_modal_fusion import (
    CrossModalFusion, FusionConfig,
    MultimodalInput, create_multimodal_input
)
import torch


# =============================================================================
# Helper: Build SafetyConfig for a given grade
# =============================================================================
_GRADE_SAFETY_CONFIGS = {
    'S': {
        'joint_limits': [-3.14, -2.5, -3.14, -3.14, -3.14, -3.14],
        'velocity_limits': [1.0, 1.0, 1.0, 1.5, 1.5, 1.5],
        'acceleration_limits': [2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
        'level': SafetyLevel.S,
    },
    'M': {
        'joint_limits': [-3.14, -2.5, -3.14, -3.14, -3.14, -3.14],
        'velocity_limits': [2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
        'acceleration_limits': [5.0, 5.0, 5.0, 8.0, 8.0, 8.0],
        'level': SafetyLevel.M,
    },
    'L': {
        'joint_limits': [-3.14, -2.5, -3.14, -3.14, -3.14, -3.14],
        'velocity_limits': [3.0, 3.0, 3.0, 5.0, 5.0, 5.0],
        'acceleration_limits': [10.0, 10.0, 10.0, 15.0, 15.0, 15.0],
        'level': SafetyLevel.L,
    },
    'XL': {
        'joint_limits': [-3.14, -2.5, -3.14, -3.14, -3.14, -3.14],
        'velocity_limits': [5.0, 5.0, 5.0, 8.0, 8.0, 8.0],
        'acceleration_limits': [15.0, 15.0, 15.0, 20.0, 20.0, 20.0],
        'level': SafetyLevel.XL,
    },
    'XXL': {
        'joint_limits': [-3.14, -2.5, -3.14, -3.14, -3.14, -3.14],
        'velocity_limits': [8.0, 8.0, 8.0, 10.0, 10.0, 10.0],
        'acceleration_limits': [20.0, 20.0, 20.0, 30.0, 30.0, 30.0],
        'level': SafetyLevel.XXL,
    },
}

def make_safety_config(grade: str) -> SafetyConfig:
    """构建指定等级的安全配置"""
    spec = _GRADE_SAFETY_CONFIGS[grade]
    limits = spec['joint_limits']
    return SafetyConfig(
        joint_limits_lower=np.array(limits),
        joint_limits_upper=np.array([abs(x) for x in limits]),
        velocity_limits=np.array(spec['velocity_limits']),
        acceleration_limits=np.array(spec['acceleration_limits']),
        torque_limits=np.array([50.0]*6),
        safety_level=spec['level']
    )


# =============================================================================
# Test Tactile + Control Pipeline (All 5 Grades)
# =============================================================================
class TestTactileControlPipeline(unittest.TestCase):
    """触觉-控制流水线测试: TactileArray → TactileServoController 完整链路"""

    GRADES = ['S', 'M', 'L', 'XL', 'XXL']

    def test_tactile_servo_initialization_all_grades(self):
        """测试各等级 TactileServoController 初始化"""
        for grade in self.GRADES:
            with self.subTest(grade=grade):
                params = TactileServoParams.from_grade(grade)
                sensor = TactileArray(array_size=get_tactile_spec(grade)['array'])
                ctrl = TactileServoController(sensor, params=params)
                self.assertIsNotNone(ctrl)
                self.assertEqual(ctrl.params.grade, grade)

    def test_tactile_array_grade_specs(self):
        """测试各等级触觉阵列规格"""
        expected = {
            'S':  {'array': (8, 8),     'res': 12, 'range_kpa': (0, 500),   'freq_hz': 50,  'temp': False},
            'M':  {'array': (16, 16),   'res': 12, 'range_kpa': (0, 1000),  'freq_hz': 100, 'temp': True},
            'L':  {'array': (24, 24),   'res': 14, 'range_kpa': (0, 2000),  'freq_hz': 200, 'temp': True},
            'XL': {'array': (32, 32),   'res': 14, 'range_kpa': (0, 5000), 'freq_hz': 500, 'temp': True},
            'XXL':{'array': (48, 48),   'res': 16, 'range_kpa': (0, 10000),'freq_hz': 1000,'temp': True},
        }
        for grade in self.GRADES:
            spec = get_tactile_spec(grade)
            self.assertEqual(spec['array'], expected[grade]['array'])
            self.assertEqual(spec['res'], expected[grade]['res'])
            self.assertEqual(spec['freq_hz'], expected[grade]['freq_hz'])

    def test_tactile_servo_control_loop(self):
        """测试 TactileServoController 闭环控制"""
        for grade in self.GRADES:
            with self.subTest(grade=grade):
                sensor = TactileArray(
                    array_size=get_tactile_spec(grade)['array'],
                )
                params = TactileServoParams.from_grade(grade)
                ctrl = TactileServoController(sensor, params=params)
                sensor.open()

                # 采集触觉帧
                frame = sensor.capture()
                contacts = sensor.detect_contacts(frame)

                # 闭环控制计算 (位置增量)
                output = ctrl.compute_control_signal(target_force=5.0, current_frame=frame)
                self.assertIsInstance(output, np.ndarray)
                self.assertEqual(output.shape[0], 3)

                sensor.close()

    def test_tactile_slip_detection(self):
        """测试滑移检测响应"""
        sensor = TactileArray(array_size=(16, 16))
        params = TactileServoParams.from_grade('M')
        ctrl = TactileServoController(sensor, params=params)
        sensor.open()

        frame = sensor.capture()
        slip_output = ctrl.detect_and_react_slip(frame)
        self.assertIsInstance(slip_output, np.ndarray)
        self.assertEqual(slip_output.shape, (3,))

        sensor.close()

    def test_grasp_quality_assessment(self):
        """测试抓取质量评估"""
        sensor = TactileArray(array_size=(16, 16))
        params = TactileServoParams.from_grade('M')
        ctrl = TactileServoController(sensor, params=params)
        sensor.open()

        frame = sensor.capture()
        quality_metrics = ctrl.monitor_grasp_quality()
        self.assertIn('current', quality_metrics)
        self.assertIn('average', quality_metrics)

        sensor.close()


# =============================================================================
# Test Force + Control Pipeline (All 5 Grades)
# =============================================================================
class TestForceControlPipeline(unittest.TestCase):
    """力觉-控制流水线测试: ForceTorqueSensor → ForceController 完整链路"""

    GRADES = ['S', 'M', 'L', 'XL', 'XXL']

    def test_force_sensor_grade_specs(self):
        """测试各等级力觉传感器规格"""
        expected = {
            'S':  {'axes': 3, 'force_range': 100,  'sampling_hz': 100},
            'M':  {'axes': 6, 'force_range': 200,  'sampling_hz': 500},
            'L':  {'axes': 6, 'force_range': 500,  'sampling_hz': 1000},
            'XL': {'axes': 6, 'force_range': 1000, 'sampling_hz': 2000},
            'XXL':{'axes': 6, 'force_range': 5000, 'sampling_hz': 5000},
        }
        for grade in self.GRADES:
            spec = get_force_spec(grade)
            self.assertEqual(spec['axes'], expected[grade]['axes'])
            self.assertEqual(spec['force_range'], expected[grade]['force_range'])
            self.assertEqual(spec['sampling_hz'], expected[grade]['sampling_hz'])

    def test_force_controller_initialization_all_grades(self):
        """测试各等级 ForceController 初始化"""
        for grade in self.GRADES:
            with self.subTest(grade=grade):
                params = ForceControlParams.from_grade(grade)
                sensor = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS if grade != 'S' else ForceSensorType.THREE_AXIS)
                ctrl = ForceController(sensor, params=params)
                self.assertIsNotNone(ctrl)
                self.assertEqual(ctrl.params.grade, grade)

    def test_force_controller_admittance_control(self):
        """测试导纳控制"""
        for grade in self.GRADES:
            with self.subTest(grade=grade):
                params = ForceControlParams.from_grade(grade)
                sensor = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
                ctrl = ForceController(sensor, params=params)
                sensor.open()

                current_wrench = sensor.capture()
                desired_force = np.array([0, 0, -5.0])
                dt = 0.01

                output = ctrl.compute_admittance(desired_force, current_wrench, dt)
                self.assertIsInstance(output, np.ndarray)
                self.assertEqual(output.shape[0], 3)

                sensor.close()

    def test_force_controller_collision_detection(self):
        """测试碰撞检测"""
        for grade in self.GRADES:
            with self.subTest(grade=grade):
                params = ForceControlParams.from_grade(grade)
                sensor = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
                ctrl = ForceController(sensor, params=params)

                collision_wrench = Wrench(
                    force=np.array([50.0, 0, 0]),
                    torque=np.zeros(3), timestamp=0.0, frame_id=0
                )
                # Call detect_collision multiple times to build history
                for _ in range(5):
                    ctrl.detect_collision(collision_wrench)
                is_collision, magnitude = ctrl.detect_collision(collision_wrench)
                self.assertIsInstance(is_collision, (bool, np.bool_))
                self.assertIsInstance(magnitude, (float, np.floating))

    def test_hybrid_force_position_control(self):
        """测试力位混合控制"""
        params = ForceControlParams.from_grade('M')
        sensor = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        ctrl = HybridForcePositionController(sensor, params=params)
        self.assertIsNotNone(ctrl)


# =============================================================================
# Test IMU + Control Pipeline (All 5 Grades)
# =============================================================================
class TestIMUControlPipeline(unittest.TestCase):
    """IMU-控制流水线测试: IMUSensor → AttitudeStabilizer 完整链路"""

    GRADES = ['S', 'M', 'L', 'XL', 'XXL']

    def test_imu_sensor_grade_specs(self):
        """测试各等级IMU传感器规格"""
        expected = {
            'S':  {'type': 'MPU6050', 'sample_hz': 100,  'noise_density': 400},
            'M':  {'type': 'BMI088',  'sample_hz': 200,  'noise_density': 120},
            'L':  {'type': 'BMI088',  'sample_hz': 500,  'noise_density': 60},
            'XL': {'type': 'ADIS16470','sample_hz': 1000, 'noise_density': 20},
            'XXL':{'type': 'ADIS16470','sample_hz': 2000, 'noise_density': 10},
        }
        for grade in self.GRADES:
            spec = get_imu_spec(grade)
            self.assertEqual(spec['sample_hz'], expected[grade]['sample_hz'])
            self.assertEqual(spec['noise_density'], expected[grade]['noise_density'])

    def test_attitude_stabilizer_initialization_all_grades(self):
        """测试各等级 AttitudeStabilizer 初始化"""
        for grade in self.GRADES:
            with self.subTest(grade=grade):
                params = IMUControlParams.from_grade(grade)
                imu = IMUSensor(sensor_type=IMUSensorType.BMI088)
                ctrl = AttitudeStabilizer(imu, params=params)
                self.assertIsNotNone(ctrl)
                self.assertEqual(ctrl.params.grade, grade)

    def test_attitude_stabilizer_update(self):
        """测试姿态稳定控制器更新"""
        for grade in self.GRADES:
            with self.subTest(grade=grade):
                params = IMUControlParams.from_grade(grade)
                imu = IMUSensor(sensor_type=IMUSensorType.BMI088)
                ctrl = AttitudeStabilizer(imu, params=params)
                imu.open()

                ctrl.set_target_attitude(roll=0.0, pitch=0.0, yaw=0.0)
                torque_cmd = ctrl.update(dt=0.01)
                self.assertIsInstance(torque_cmd, np.ndarray)
                self.assertEqual(torque_cmd.shape, (3,))

                imu.close()

    def test_attitude_stabilizer_tilt_status(self):
        """测试倾角状态检测"""
        params = IMUControlParams.from_grade('M')
        imu = IMUSensor(sensor_type=IMUSensorType.BMI088)
        ctrl = AttitudeStabilizer(imu, params=params)
        imu.open()

        status = ctrl.get_tilt_status()
        self.assertIn('roll', status)
        self.assertIn('pitch', status)
        self.assertIn('is_stable', status)

        imu.close()

    def test_motion_estimator(self):
        """测试运动估计器"""
        imu = IMUSensor(sensor_type=IMUSensorType.BMI088)
        motion_est = MotionEstimator(imu_sensor=imu)
        imu.open()

        velocity, position = motion_est.update(dt=0.01)
        self.assertIsInstance(velocity, np.ndarray)
        self.assertEqual(velocity.shape, (3,))
        self.assertIsInstance(position, np.ndarray)
        self.assertEqual(position.shape, (3,))

        motion_est.reset()
        imu.close()


# =============================================================================
# Test Safety Integration Across All Grades
# =============================================================================
class TestSafetyFiveGrade(unittest.TestCase):
    """五级安全集成测试"""

    GRADES = ['S', 'M', 'L', 'XL', 'XXL']
    SAFETY_LEVELS = [SafetyLevel.S, SafetyLevel.M, SafetyLevel.L,
                     SafetyLevel.XL, SafetyLevel.XXL]

    def test_safety_controller_all_grades(self):
        """测试各等级安全控制器"""
        for grade, level in zip(self.GRADES, self.SAFETY_LEVELS):
            with self.subTest(grade=grade):
                config = make_safety_config(grade)
                safety = SafetyController(config)
                self.assertEqual(safety.safety_level, level)
                self.assertFalse(safety.is_emergency_stopped)

    def test_safety_velocity_limits_all_grades(self):
        """测试各等级速度限制"""
        for grade in self.GRADES:
            with self.subTest(grade=grade):
                config = make_safety_config(grade)
                safety = SafetyController(config)

                current_vel = np.array([2.0, 0.5, 0.0, 1.0, 0.5, 0.0])
                desired_vel = np.array([5.0, 1.0, 0.0, 8.0, 2.0, 0.0])

                safe_vel = safety.compute_safe_velocity(current_vel, desired_vel)
                self.assertIsInstance(safe_vel, np.ndarray)
                self.assertEqual(safe_vel.shape, current_vel.shape)

    def test_emergency_stop_all_grades(self):
        """测试各等级紧急停止"""
        for grade in self.GRADES:
            with self.subTest(grade=grade):
                config = make_safety_config(grade)
                safety = SafetyController(config)

                safety.emergency_stop()
                self.assertTrue(safety.is_emergency_stopped)

                safety.reset()
                self.assertFalse(safety.is_emergency_stopped)


# =============================================================================
# Test Complete End-to-End Pipeline
# =============================================================================
class TestEndToEndPipeline(unittest.TestCase):
    """端到端完整流水线测试: 传感器采集 → 融合 → 控制 → 安全检查"""

    def test_complete_tactile_pipeline(self):
        """测试完整触觉流水线: 采集→控制→安全"""
        sensor = TactileArray(array_size=(16, 16))
        params = TactileServoParams.from_grade('M')
        ctrl = TactileServoController(sensor, params=params)
        safety = SafetyController(make_safety_config('M'))

        sensor.open()

        # 采集
        frame = sensor.capture()
        contacts = sensor.detect_contacts(frame)

        # 控制
        ctrl_output = ctrl.compute_control_signal(target_force=5.0, current_frame=frame)

        # 安全 (safety controller operates on 6-DOF joint velocity)
        safe_output = safety.compute_safe_velocity(np.zeros(6), np.concatenate([ctrl_output, np.zeros(3)]))

        self.assertIsNotNone(ctrl_output)
        self.assertIsNotNone(safe_output)

        sensor.close()

    def test_complete_force_pipeline(self):
        """测试完整力觉流水线"""
        sensor = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        params = ForceControlParams.from_grade('M')
        ctrl = ForceController(sensor, params=params)
        safety = SafetyController(make_safety_config('M'))

        sensor.open()

        # 采集
        wrench = sensor.capture()

        # 控制
        ctrl_output = ctrl.compute_admittance(
            np.array([0, 0, -5.0]),
            wrench,
            dt=0.01
        )

        # 安全
        safe_output = safety.compute_safe_velocity(np.zeros(6), np.concatenate([ctrl_output, np.zeros(3)]))

        self.assertIsNotNone(safe_output)
        sensor.close()

    def test_complete_imu_pipeline(self):
        """测试完整IMU流水线"""
        imu = IMUSensor(sensor_type=IMUSensorType.BMI088)
        params = IMUControlParams.from_grade('M')
        ctrl = AttitudeStabilizer(imu, params=params)
        safety = SafetyController(make_safety_config('M'))

        imu.open()

        # 采集
        frame = imu.capture()

        # 控制
        ctrl.set_target_attitude(roll=0.0, pitch=0.0, yaw=0.0)
        torque_cmd = ctrl.update(dt=0.01)

        # 安全
        safe_output = safety.compute_safe_velocity(np.zeros(6), np.concatenate([torque_cmd, np.zeros(3)]))

        self.assertIsNotNone(safe_output)
        imu.close()

    def test_multimodal_fusion_to_control_pipeline(self):
        """测试多模态融合→控制流水线"""
        # 初始化各传感器
        tactile = TactileArray(array_size=(16, 16))
        force = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        imu = IMUSensor(sensor_type=IMUSensorType.BMI088)

        # 融合网络
        fusion = CrossModalFusion(FusionConfig(hidden_dim=256, num_heads=4, num_layers=2))
        fusion.eval()

        tactile.open()
        force.open()
        imu.open()

        # 采集多模态数据
        tac_frame = tactile.capture()
        wrench = force.capture()
        imu_frame = imu.capture()

        # 填充到编码器期望维度 (tactile=64, force=32, imu=64)
        tactile_feat = tac_frame.pressure_map.flatten()[:64]
        if len(tactile_feat) < 64:
            tactile_feat = np.pad(tactile_feat, (0, 64 - len(tactile_feat)))
        force_vec = wrench.to_vector()
        if len(force_vec) < 32:
            force_vec = np.pad(force_vec, (0, 32 - len(force_vec)))
        imu_vec = np.concatenate([imu_frame.accel, imu_frame.gyro])
        if len(imu_vec) < 64:
            imu_vec = np.pad(imu_vec, (0, 64 - len(imu_vec)))

        # 融合
        multimodal = create_multimodal_input(
            tactile=tactile_feat.reshape(1, -1).astype(np.float32),
            force=force_vec.reshape(1, -1).astype(np.float32),
            imu=imu_vec.reshape(1, -1).astype(np.float32),
        )

        with torch.no_grad():
            fused = fusion(multimodal)

        # 基于融合特征做决策
        fused_mean = fused.mean().item()

        # 模拟控制器
        ctrl_output = fused_mean * 0.1

        self.assertIsNotNone(fused)
        self.assertGreater(fused.shape[1], 0)
        self.assertIsInstance(ctrl_output, float)

        tactile.close()
        force.close()
        imu.close()


# =============================================================================
# Test Five Grade Parameter Monotonicity
# =============================================================================
class TestFiveGradeParameterMonotonicity(unittest.TestCase):
    """五级参数单调性测试: 确保高级规格参数优于低级规格"""

    def test_tactile_grade_monotonicity(self):
        """触觉五级参数单调递增"""
        grades = ['S', 'M', 'L', 'XL', 'XXL']
        for grade in grades:
            spec = get_tactile_spec(grade)
            idx = grades.index(grade)
            if idx > 0:
                prev_spec = get_tactile_spec(grades[idx - 1])
                # 高级规格数组更大/采样率更高
                self.assertGreaterEqual(spec['array'][0], prev_spec['array'][0])
                self.assertGreaterEqual(spec['freq_hz'], prev_spec['freq_hz'])

    def test_force_grade_monotonicity(self):
        """力觉五级参数单调递增"""
        grades = ['S', 'M', 'L', 'XL', 'XXL']
        for grade in grades:
            spec = get_force_spec(grade)
            idx = grades.index(grade)
            if idx > 0:
                prev_spec = get_force_spec(grades[idx - 1])
                self.assertGreaterEqual(spec['force_range'], prev_spec['force_range'])
                self.assertGreaterEqual(spec['sampling_hz'], prev_spec['sampling_hz'])

    def test_imu_grade_monotonicity(self):
        """IMU五级参数单调递增"""
        grades = ['S', 'M', 'L', 'XL', 'XXL']
        for grade in grades:
            spec = get_imu_spec(grade)
            idx = grades.index(grade)
            if idx > 0:
                prev_spec = get_imu_spec(grades[idx - 1])
                self.assertGreaterEqual(spec['sample_hz'], prev_spec['sample_hz'])
                # 噪声密度递减 (越好越低)
                self.assertLessEqual(spec['noise_density'], prev_spec['noise_density'])


# =============================================================================
# Test Fusion Network Five Grade Scalability
# =============================================================================
class TestFusionFiveGradeScalability(unittest.TestCase):
    """融合网络五级可扩展性测试"""

    GRADES = ['S', 'M', 'L', 'XL', 'XXL']

    def test_fusion_initialization_all_grades(self):
        """测试各等级融合网络初始化"""
        grades_hidden = {'S': 128, 'M': 256, 'L': 384, 'XL': 512, 'XXL': 768}
        grades_heads = {'S': 2, 'M': 4, 'L': 6, 'XL': 8, 'XXL': 12}

        for grade in self.GRADES:
            with self.subTest(grade=grade):
                config = FusionConfig(
                    hidden_dim=grades_hidden[grade],
                    num_heads=grades_heads[grade],
                    num_layers=2
                )
                fusion = CrossModalFusion(config)
                self.assertIsNotNone(fusion)

    def test_fusion_forward_partial_modalities(self):
        """测试部分模态输入融合"""
        grades_hidden = {'S': 128, 'M': 256, 'L': 384, 'XL': 512, 'XXL': 768}

        for grade in self.GRADES:
            with self.subTest(grade=grade):
                config = FusionConfig(hidden_dim=grades_hidden[grade], num_heads=4, num_layers=2)
                fusion = CrossModalFusion(config)
                fusion.eval()

                # 仅触觉 + IMU (填充到编码器期望维度)
                tactile = np.random.randn(2, 64).astype(np.float32)
                imu = np.random.randn(2, 64).astype(np.float32)

                multimodal = create_multimodal_input(
                    tactile=tactile,
                    imu=imu,
                )

                with torch.no_grad():
                    output = fusion(multimodal)

                self.assertEqual(output.shape[0], 2)
                self.assertEqual(output.shape[1], grades_hidden[grade])


if __name__ == '__main__':
    unittest.main(verbosity=2)
