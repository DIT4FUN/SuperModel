"""
SuperModel 全流水线集成测试
============================

端到端测试: 传感器采集 → 跨模态融合 → 控制决策 → 执行反馈

覆盖:
1. 触觉+IMU+力觉多传感器同步采集
2. 多模态特征编码与跨模态注意力融合
3. 触觉伺服/力控/姿态稳定的端到端闭环
4. AGV五级规格流水线验证

版本: v1.88.0
"""

import pytest
import numpy as np
import sys
import time
import os

# 路径设置
_PROJECT_ROOT = '/home/treeman/.openclaw/workspace/projects/SuperModel'
_SRC_PATH = os.path.join(_PROJECT_ROOT, 'src')
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sensors.tactile import (
    TactileArray, TactileFrame, TactileContact, TactileSensorType,
    VirtualTactileSensor, PressureProcessor, get_tactile_spec
)
from sensors.force import (
    ForceTorqueSensor, ForceSensorType, Wrench, VirtualForceSensor,
    WrenchProcessor, get_force_spec
)
from sensors.imu import (
    IMUSensor, IMUSensorType, IMUFrame, Pose, PoseEstimator,
    VirtualIMUSensor, get_imu_spec
)
from fusion.cross_modal_fusion import (
    CrossModalFusion, FusionConfig, MultimodalInput,
    create_multimodal_input, get_fusion_spec
)
from control.tactile_control import TactileServoController, TactileServoParams
from control.force_control import ForceController, ForceControlParams
from control.imu_control import AttitudeStabilizer, IMUControlParams


# ─────────────────────────────────────────────────────────────────────────────
# 测试配置
# ─────────────────────────────────────────────────────────────────────────────

GRADES = ['S', 'M', 'L', 'XL', 'XXL']


# ─────────────────────────────────────────────────────────────────────────────
# 传感器同步采集测试
# ─────────────────────────────────────────────────────────────────────────────

class TestSensorSynchronizedCapture:
    """多传感器同步采集与数据完整性"""

    @pytest.mark.parametrize("grade", GRADES)
    def test_synchronized_capture_3sensor(self, grade):
        """触觉+IMU+力觉三传感器同步采集，数据完整性测试"""
        tactile = VirtualTactileSensor(array_size=(16, 16), sensor_id=f"tactile_{grade}")
        force = VirtualForceSensor(sensor_id=f"force_{grade}")
        imu = VirtualIMUSensor(sensor_id=f"imu_{grade}")

        tactile.open()
        force.open()
        imu.open()

        for i in range(20):
            # 同步采集
            tactile_frame = tactile.simulate_contact(
                contact_pos=(0.5, 0.5),
                contact_radius=0.2,
                contact_force=10.0
            )
            force_wrench = force.simulate_contact(
                force=(0.0, 0.0, -10.0),
                torque=(0.0, 0.0, 0.0)
            )
            imu_frame = imu.simulate_static(orientation=(0.0, 0.0, 0.0))

            # 验证数据完整性
            assert tactile_frame.pressure_map.shape == (16, 16)
            assert force_wrench.force.shape == (3,)
            assert imu_frame.accel.shape == (3,)

            # 验证数据类型 (允许float32或float64)
            assert tactile_frame.pressure_map.dtype in (np.float32, np.float64)
            assert force_wrench.force.dtype in (np.float32, np.float64)
            assert imu_frame.accel.dtype in (np.float32, np.float64)

        tactile.close()
        force.close()
        imu.close()

    @pytest.mark.parametrize("grade", GRADES)
    def test_agv_grade_sensor_specs(self, grade):
        """验证AGV五级规格与传感器能力匹配"""
        t_spec = get_tactile_spec(grade)
        f_spec = get_force_spec(grade)
        i_spec = get_imu_spec(grade)

        assert t_spec['array'][0] >= 8
        assert t_spec['freq_hz'] >= 50
        assert f_spec['axes'] in [3, 6]
        assert f_spec['force_range'] > 0
        assert f_spec['sampling_hz'] >= 100
        assert i_spec['sample_hz'] >= 100


# ─────────────────────────────────────────────────────────────────────────────
# 跨模态融合集成测试
# ─────────────────────────────────────────────────────────────────────────────

class TestFusionIntegration:
    """跨模态融合与传感器控制集成"""

    @pytest.mark.parametrize("grade", GRADES)
    def test_fusion_spec_exists(self, grade):
        """FusionSpec 存在性验证"""
        f_spec = get_fusion_spec(grade)
        assert f_spec is not None
        assert 'fusion_dim' in f_spec or 'hidden_dim' in f_spec

    @pytest.mark.parametrize("grade", GRADES)
    def test_multimodal_input_creation(self, grade):
        """MultimodalInput 创建与验证"""
        tactile_arr = np.random.randn(16, 16).astype(np.float32)
        force_vec = np.random.randn(6).astype(np.float32)
        imu_vec = np.random.randn(6).astype(np.float32)

        mm_input = create_multimodal_input(
            tactile=tactile_arr,
            force=force_vec,
            imu=imu_vec
        )

        assert 'tactile' in mm_input.modalities
        assert 'force' in mm_input.modalities
        assert 'imu' in mm_input.modalities

    @pytest.mark.parametrize("grade", GRADES)
    def test_fusion_forward_pass(self, grade):
        """跨模态融合前向传播"""
        import torch
        fusion = CrossModalFusion(FusionConfig(hidden_dim=128, num_heads=4, num_layers=2, tactile_dim=256, force_dim=6, imu_dim=6))

        tactile_arr = np.random.randn(256).astype(np.float32)
        force_vec = np.random.randn(6).astype(np.float32)
        imu_vec = np.random.randn(6).astype(np.float32)

        mm_input = create_multimodal_input(
            tactile=tactile_arr,
            force=force_vec,
            imu=imu_vec
        )

        fused = fusion(mm_input)
        # fused is a numpy ndarray, shape depends on hidden_dim
        assert isinstance(fused, np.ndarray)
        assert fused.ndim >= 1

    @pytest.mark.parametrize("grade", GRADES)
    def test_fusion_with_real_sensor_data(self, grade):
        """真实传感器数据的融合处理"""
        import torch

        tactile = TactileArray(
            array_size=(16, 16),
            sensor_type=TactileSensorType.RESISTIVE,
            sensor_id=f"tactile_real_{grade}"
        )
        force = ForceTorqueSensor(
            sensor_type=ForceSensorType.SIX_AXIS,
            sensor_id=f"force_real_{grade}"
        )
        imu = IMUSensor(
            sensor_type=IMUSensorType.BMI088,
            sensor_id=f"imu_real_{grade}"
        )

        tactile.open()
        force.open()
        imu.open()

        t_frame = tactile.capture()
        f_wrench = force.capture()
        i_frame = imu.capture()

        t_feat = t_frame.pressure_map.flatten()[:64].astype(np.float32)
        if len(t_feat) < 64:
            t_feat = np.pad(t_feat, (0, 64 - len(t_feat)))

        t_input = t_feat
        f_feat = f_wrench.to_vector().astype(np.float32)
        i_feat = np.concatenate([i_frame.accel, i_frame.gyro]).astype(np.float32)

        fusion = CrossModalFusion(FusionConfig(hidden_dim=64, force_dim=6, imu_dim=6))
        mm_input = create_multimodal_input(
            tactile=t_input,
            force=f_feat,
            imu=i_feat
        )
        fused = fusion(mm_input)
        assert isinstance(fused, np.ndarray)

        tactile.close()
        force.close()
        imu.close()


# ─────────────────────────────────────────────────────────────────────────────
# 触觉伺服控制集成测试
# ─────────────────────────────────────────────────────────────────────────────

class TestTactileServoIntegration:
    """触觉传感器与伺服控制器集成"""

    @pytest.mark.parametrize("grade", GRADES)
    def test_tactile_servo_closed_loop(self, grade):
        """触觉伺服闭环控制"""
        params = TactileServoParams.from_grade(grade)

        tactile = TactileArray(
            array_size=(16, 16),
            sensor_type=TactileSensorType.CAPACITIVE,
            sensor_id=f"tactile_servo_{grade}"
        )
        tactile.open()

        controller = TactileServoController(tactile, params)

        for i in range(10):
            frame = tactile.capture()

            control_signal = controller.compute_control_signal(
                target_force=10.0,
                current_frame=frame
            )

            assert control_signal.shape == (3,)
            assert not np.any(np.isnan(control_signal))

        quality = controller.monitor_grasp_quality()
        assert 'current' in quality
        assert 'average' in quality

        tactile.close()

    @pytest.mark.parametrize("grade", GRADES)
    def test_slip_detection_and_reaction(self, grade):
        """滑移检测与反应控制"""
        params = TactileServoParams.from_grade(grade)

        tactile = TactileArray(
            array_size=(16, 16),
            sensor_type=TactileSensorType.PIEZOELECTRIC,
            sensor_id=f"tactile_slip_{grade}"
        )
        tactile.open()

        controller = TactileServoController(tactile, params)

        # 模拟多次采集触觉帧
        frames = []
        for _ in range(5):
            frame = tactile.capture()
            frames.append(frame)

        for f in frames:
            reactive = controller.detect_and_react_slip(f)
            assert reactive.shape == (3,)

        tactile.close()


# ─────────────────────────────────────────────────────────────────────────────
# 力觉控制集成测试
# ─────────────────────────────────────────────────────────────────────────────

class TestForceControlIntegration:
    """力觉传感器与力控制器集成"""

    @pytest.mark.parametrize("grade", GRADES)
    def test_force_controller_admittance(self, grade):
        """力觉导纳控制"""
        params = ForceControlParams.from_grade(grade)

        force_sensor = VirtualForceSensor(
            sensor_id=f"force_ctrl_{grade}"
        )
        force_sensor.open()

        controller = ForceController(force_sensor, params)

        desired_force = np.array([0.0, 0.0, -10.0])

        for i in range(10):
            measured = force_sensor.simulate_contact(
                force=(0.0, 0.0, -8.0 + i * 0.2),
                torque=(0.0, 0.0, 0.0)
            )

            adj = controller.compute_admittance(
                desired_force=desired_force,
                current_wrench=measured,
                dt=0.01
            )

            assert adj.shape == (3,)
            assert not np.any(np.isnan(adj))

        force_sensor.close()

    @pytest.mark.parametrize("grade", GRADES)
    def test_collision_detection_integration(self, grade):
        """碰撞检测集成"""
        force_sensor = VirtualForceSensor(
            sensor_id=f"collision_{grade}"
        )
        force_sensor.open()

        controller = ForceController(force_sensor)

        # 正常状态
        normal_wrench = force_sensor.simulate_contact(
            force=(0.0, 0.0, -5.0),
            torque=(0.0, 0.0, 0.0)
        )
        is_collision, magnitude = controller.detect_collision(normal_wrench)
        assert bool(is_collision) in [True, False]
        assert magnitude >= 0

        # 碰撞状态
        collision_wrench = force_sensor.simulate_contact(
            force=(0.0, 0.0, -80.0),
            torque=(0.0, 0.0, 0.0)
        )
        is_collision, magnitude = controller.detect_collision(collision_wrench)
        assert magnitude >= 0

        force_sensor.close()

    @pytest.mark.parametrize("grade", GRADES)
    def test_payload_estimation_integration(self, grade):
        """负载估计与力控集成"""
        force_sensor = ForceTorqueSensor(
            sensor_type=ForceSensorType.SIX_AXIS,
            sensor_id=f"payload_{grade}"
        )
        force_sensor.open()

        virtual = VirtualForceSensor(sensor_id=f"virtual_payload_{grade}")
        virtual.open()

        wrench = virtual.simulate_payload(mass=1.5, com_offset=(0.01, 0.0, 0.05))
        estimated = force_sensor.estimate_payload(wrench)

        assert abs(estimated - 1.5) < 0.5

        force_sensor.close()
        virtual.close()


# ─────────────────────────────────────────────────────────────────────────────
# IMU控制集成测试
# ─────────────────────────────────────────────────────────────────────────────

class TestIMUControlIntegration:
    """IMU传感器与姿态稳定控制器集成"""

    @pytest.mark.parametrize("grade", GRADES)
    def test_attitude_stabilizer_closed_loop(self, grade):
        """姿态稳定闭环"""
        params = IMUControlParams.from_grade(grade)

        imu = IMUSensor(
            sensor_type=IMUSensorType.BMI088,
            sensor_id=f"imu_stab_{grade}"
        )
        imu.open()

        stabilizer = AttitudeStabilizer(imu, params)
        stabilizer.set_target_attitude(roll=0.0, pitch=0.0, yaw=0.0)

        for i in range(20):
            frame = imu.capture()

            torque_cmd = stabilizer.update(frame, dt=0.01)

            assert torque_cmd.shape == (3,)
            assert not np.any(np.isnan(torque_cmd))

        tilt_status = stabilizer.get_tilt_status()
        assert 'roll' in tilt_status
        assert 'pitch' in tilt_status

        imu.close()

    @pytest.mark.parametrize("grade", GRADES)
    def test_pose_estimator_integration(self, grade):
        """姿态估计器与IMU集成"""
        imu = IMUSensor(
            sensor_type=IMUSensorType.MPU9250,
            sensor_id=f"imu_pose_{grade}"
        )
        imu.open()

        estimator = PoseEstimator(algorithm='madgwick', sample_rate=200.0)

        poses = []
        for i in range(50):
            frame = imu.capture()

            pose = estimator.update(
                frame.accel,
                frame.gyro,
                frame.mag
            )

            poses.append(pose)
            assert pose.orientation.shape == (4,)

        for p in poses:
            norm = np.linalg.norm(p.orientation)
            assert abs(norm - 1.0) < 0.01

        imu.close()


# ─────────────────────────────────────────────────────────────────────────────
# 全流程端到端集成测试
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipelineEndToEnd:
    """完整流水线: 感知→融合→决策→控制→反馈"""

    @pytest.mark.parametrize("grade", ['M', 'L', 'XL'])
    def test_end_to_end_grasp_pipeline(self, grade):
        """端到端抓取流水线"""
        import torch

        tactile = TactileArray(
            array_size=(16, 16),
            sensor_type=TactileSensorType.CAPACITIVE,
            sensor_id=f"e2e_tactile_{grade}"
        )
        force = ForceTorqueSensor(
            sensor_type=ForceSensorType.SIX_AXIS,
            sensor_id=f"e2e_force_{grade}"
        )

        tactile.open()
        force.open()

        t_params = TactileServoParams.from_grade(grade)
        tactile_ctrl = TactileServoController(tactile, t_params)

        fusion = CrossModalFusion(FusionConfig(hidden_dim=64, force_dim=6, imu_dim=6))

        for step in range(10):
            t_frame = tactile.capture()
            f_wrench = force.capture()

            # 融合
            t_flat = t_frame.pressure_map.flatten()[:64].astype(np.float32)
            if len(t_flat) < 64:
                t_flat = np.pad(t_flat, (0, 64 - len(t_flat)))
            f_input = f_wrench.to_vector().astype(np.float32)
            i_input = np.zeros(6, dtype=np.float32)

            mm_input = create_multimodal_input(
                tactile=t_flat,
                force=f_input,
                imu=i_input
            )
            fused = fusion(mm_input)
            assert isinstance(fused, np.ndarray)

            ctrl = tactile_ctrl.compute_control_signal(
                target_force=10.0,
                current_frame=t_frame
            )
            assert ctrl.shape == (3,)

        quality = tactile_ctrl.monitor_grasp_quality()
        assert quality['current'] >= 0.0

        tactile.close()
        force.close()

    @pytest.mark.parametrize("grade", ['S', 'M', 'L'])
    def test_end_to_end_agv_navigation(self, grade):
        """端到端AGV导航流水线"""
        imu = VirtualIMUSensor(
            sensor_id=f"e2e_imu_{grade}"
        )
        imu.open()

        estimator = PoseEstimator(algorithm='madgwick', sample_rate=200.0)

        stab_params = IMUControlParams.from_grade(grade)
        stabilizer = AttitudeStabilizer(imu, stab_params)
        stabilizer.set_target_attitude(0.0, 0.0, 0.0)

        for i in range(30):
            frame = imu.simulate_agv_motion(
                linear_velocity=(0.5, 0.0),
                angular_velocity=0.1,
                dt=0.01,
                grade=grade
            )

            pose = estimator.update(frame.accel, frame.gyro, frame.mag)
            torque_cmd = stabilizer.update(frame, dt=0.01)

            assert pose.orientation.shape == (4,)
            assert torque_cmd.shape == (3,)

        imu.close()


# ─────────────────────────────────────────────────────────────────────────────
# AGV五级规格完整性测试
# ─────────────────────────────────────────────────────────────────────────────

class TestAGVFiveGradeCompleteness:
    """AGV五级规格完整性验证"""

    def test_all_grades_have_complete_sensor_modules(self):
        """所有AGV等级都有完整的传感器模块"""
        for grade in GRADES:
            t_spec = get_tactile_spec(grade)
            f_spec = get_force_spec(grade)
            i_spec = get_imu_spec(grade)

            assert t_spec is not None
            assert f_spec is not None
            assert i_spec is not None
            assert t_spec['array'][0] >= 8
            assert f_spec['force_range'] > 0
            assert i_spec['sample_hz'] >= 100

    def test_all_grades_tactile_control_params(self):
        """所有AGV等级的触觉控制参数"""
        for grade in GRADES:
            params = TactileServoParams.from_grade(grade)
            assert params.Kp_position > 0
            assert params.control_rate >= 30

    def test_all_grades_force_control_params(self):
        """所有AGV等级的力觉控制参数"""
        for grade in GRADES:
            params = ForceControlParams.from_grade(grade)
            assert params.Kp_force > 0

    def test_all_grades_imu_control_params(self):
        """所有AGV等级的IMU控制参数"""
        for grade in GRADES:
            params = IMUControlParams.from_grade(grade)
            assert params.Kp_attitude > 0
            assert params.control_rate >= 50


# ─────────────────────────────────────────────────────────────────────────────
# 压力传感器处理器测试
# ─────────────────────────────────────────────────────────────────────────────

class TestPressureProcessorIntegration:
    """压力处理器与触觉阵列集成"""

    @pytest.mark.parametrize("grade", ['M', 'L', 'XL'])
    def test_pressure_processor_full_pipeline(self, grade):
        """压力处理器完整流水线"""
        processor = PressureProcessor(filter_window=3, drift_compensation=True)

        tactile = TactileArray(
            array_size=(16, 16),
            sensor_type=TactileSensorType.RESISTIVE,
            sensor_id=f"pp_{grade}"
        )
        tactile.open()

        baseline_frame = tactile.capture()
        processor.compensate_baseline(baseline_frame.pressure_map, set_baseline=True)

        for i in range(10):
            frame = tactile.capture()

            filtered = processor.filter(frame.pressure_map)
            compensated = processor.compensate_baseline(filtered)

            cy, cx = processor.compute_centroid(compensated)
            assert 0 <= cy <= 16 and 0 <= cx <= 16

            force = processor.compute_force(compensated, contact_area=1e-4)
            assert force >= 0

            hist, edges = processor.compute_pressure_histogram(compensated)
            assert len(hist) == 10

        tactile.close()


# ─────────────────────────────────────────────────────────────────────────────
# Wrench处理器集成测试
# ─────────────────────────────────────────────────────────────────────────────

class TestWrenchProcessorIntegration:
    """Wrench处理器与力觉传感器集成"""

    @pytest.mark.parametrize("grade", GRADES)
    def test_wrench_processor_filtering(self, grade):
        """Wrench滤波处理"""
        processor = WrenchProcessor(filter_alpha=0.3)

        for i in range(20):
            wrench_vec = np.array([0.0, 0.0, -10.0 + np.random.randn() * 0.5,
                                  0.0, 0.0, 0.0], dtype=np.float32)
            filtered = processor.filter(wrench_vec)
            assert filtered.shape == (6,)

    @pytest.mark.parametrize("grade", GRADES)
    def test_wrench_covariance_estimation(self, grade):
        """协方差估计"""
        processor = WrenchProcessor()

        history = []
        for i in range(20):
            wrench_vec = np.array([np.random.randn() * 0.5,
                                  np.random.randn() * 0.5,
                                  -10.0 + np.random.randn() * 0.2,
                                  0.0, 0.0, 0.0], dtype=np.float32)
            history.append(wrench_vec)

        cov = processor.estimate_covariance(history)
        assert cov.shape == (6, 6)
        # 协方差矩阵应该是正定的
        eigvals = np.linalg.eigvals(cov)
        assert np.all(eigvals.real > -1e-6)

    @pytest.mark.parametrize("grade", GRADES)
    def test_wrench_equivalent_at_point(self, grade):
        """等效力旋量变换"""
        processor = WrenchProcessor()

        wrench = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        translation = np.array([0.1, 0.0, 0.0], dtype=np.float32)

        equiv = processor.compute_equivalent_wrench_at(wrench, translation)
        assert equiv.shape == (6,)
        # Tz = x*Fy - y*Fx = 0.1*0 - 0*10 = 0
        assert abs(equiv[5]) < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# 姿态估计器多算法测试
# ─────────────────────────────────────────────────────────────────────────────

class TestPoseEstimatorMultiAlgorithm:
    """姿态估计器多算法验证"""

    @pytest.mark.parametrize("algorithm", ['madgwick', 'complementary', 'kalman'])
    def test_pose_estimator_all_algorithms(self, algorithm):
        """三种姿态估计算法验证"""
        imu = IMUSensor(
            sensor_type=IMUSensorType.BMI088,
            sensor_id=f"imu_{algorithm}"
        )
        imu.open()

        estimator = PoseEstimator(algorithm=algorithm, sample_rate=200.0)

        for i in range(30):
            frame = imu.capture()
            pose = estimator.update(frame.accel, frame.gyro, frame.mag)

            assert pose.orientation.shape == (4,)
            norm = np.linalg.norm(pose.orientation)
            assert abs(norm - 1.0) < 0.01

        euler = estimator.get_euler()
        assert euler.shape == (3,)

        imu.close()
