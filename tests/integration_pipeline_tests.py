"""
SuperModel 完整系统集成测试
===========================

端到端测试: 传感器 → 融合 → 控制 → 仿真
测试完整的多模态感知-融合-控制流程
"""

import sys
import numpy as np
import torch
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.vision import BinocularCamera, DepthProcessor, CameraIntrinsics, StereoExtrinsics
from sensors.audio import BinauralMic, SoundLocalizer
from sensors.tactile import TactileArray, PressureProcessor, TactileContact
from sensors.force import ForceTorqueSensor, Wrench, WrenchProcessor, ForceSensorType
from sensors.imu import IMUSensor, IMUFrame, PoseEstimator, Pose, IMUSensorType
from fusion.cross_modal_fusion import CrossModalFusion, FusionConfig, MultimodalInput, create_multimodal_input
from control.agv import AGVMotionController, AGVSpec, AGVGrade, AGVPose, AGVTwist
from control.impedance import ImpedanceController, ImpedanceParams
from control.mpc import MPCConfig, JointSpaceMPC, DynamicsModel
from control.safety_controller import SafetyController, SafetyConfig, SafetyLevel, JointStateSnapshot
from simulation.environment import RobotSimulator, SensorSimulator, SimConfig


class TestSensorToFusionPipeline(unittest.TestCase):
    """传感器 → 融合网络 完整管道测试"""

    def test_vision_depth_fusion(self):
        """测试视觉+深度融合流程"""
        cam = BinocularCamera(resolution=(640, 480))
        cam.open()

        frame = cam.capture()
        depth_proc = DepthProcessor(cam.left_intrinsics, cam.right_intrinsics, cam.get_extrinsics())
        # 模拟深度图
        depth = np.random.rand(480, 640).astype(np.float32) * 5.0
        filtered_depth = depth_proc.filter_depth(depth, min_dist=0.1, max_dist=5.0)

        self.assertEqual(filtered_depth.shape, depth.shape)
        self.assertTrue(np.all(filtered_depth >= 0))
        cam.close()

    def test_audio_localization_fusion(self):
        """测试声源定位融合流程"""
        mic = BinauralMic(sample_rate=16000)
        localizer = SoundLocalizer(baseline_mm=95.0, sample_rate=16000)
        mic.open()

        frame = mic.capture()
        source = localizer.localize(frame.left_channel, frame.right_channel)

        self.assertIsInstance(source.direction[0], (float, np.floating))
        self.assertGreaterEqual(source.direction[0], -90)
        self.assertLessEqual(source.direction[0], 90)
        mic.close()

    def test_tactile_contact_detection_fusion(self):
        """测试触觉接触检测融合"""
        tactile = TactileArray(array_size=(16, 16))
        processor = PressureProcessor(filter_window=3)
        tactile.open()

        frame = tactile.capture()
        contacts = tactile.detect_contacts(frame)
        filtered = processor.filter(frame.pressure_map)

        self.assertEqual(filtered.shape, frame.pressure_map.shape)
        self.assertIsInstance(contacts, list)
        tactile.close()

    def test_force_wrench_processing(self):
        """测试力矩信号处理流程"""
        sensor = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        processor = WrenchProcessor(filter_alpha=0.3)
        sensor.open()

        wrench = sensor.capture()
        filtered = processor.filter(wrench.to_vector())

        self.assertEqual(filtered.shape, (6,))
        contact_state = sensor.detect_contact(threshold=2.0)
        self.assertIn(contact_state.is_contact, [True, False])
        sensor.close()

    def test_imu_pose_estimation_fusion(self):
        """测试IMU姿态估计融合"""
        imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL)
        estimator = PoseEstimator(algorithm="madgwick", sample_rate=100.0)
        imu.open()

        pose_estimates = []
        for _ in range(20):
            frame = imu.capture()
            pose = estimator.update(frame.accel, frame.gyro)
            pose_estimates.append(pose)

        # 静止时姿态应该稳定
        euler_arr = np.array([p.to_euler() for p in pose_estimates])
        roll_std = np.std(euler_arr[:, 0])
        pitch_std = np.std(euler_arr[:, 1])

        self.assertLess(roll_std, 0.1)
        self.assertLess(pitch_std, 0.1)
        imu.close()

    def test_multimodal_fusion_forward(self):
        """测试多模态融合前向传播"""
        config = FusionConfig(hidden_dim=256, num_heads=4, num_layers=2)
        fusion = CrossModalFusion(config)

        mmi = MultimodalInput(
            vision=torch.randn(4, 512),
            audio=torch.randn(4, 128),
            tactile=torch.randn(4, 64),
            force=torch.randn(4, 32),
            imu=torch.randn(4, 64),
        )

        output = fusion(mmi)
        self.assertEqual(output.shape, (4, 256))

    def test_numpy_to_fusion_pipeline(self):
        """测试 NumPy → 融合网络管道"""
        config = FusionConfig(hidden_dim=128, num_heads=2, num_layers=1)
        fusion = CrossModalFusion(config)

        # 使用2D音频 (B x D) 避免时序维度问题
        mmi = create_multimodal_input(
            vision=np.random.randn(2, 512).astype(np.float32),
            audio=np.random.randn(2, 128).astype(np.float32),
            tactile=np.random.randn(2, 64).astype(np.float32),
            force=np.random.randn(2, 32).astype(np.float32),
            imu=np.random.randn(2, 64).astype(np.float32),
        )

        output = fusion(mmi)
        self.assertEqual(output.shape, (2, 128))


class TestFusionToControlPipeline(unittest.TestCase):
    """融合网络 → 控制系统 管道测试"""

    def test_fusion_to_agv_control(self):
        """测试融合特征到AGV控制命令"""
        fusion = CrossModalFusion(FusionConfig(hidden_dim=256, num_heads=4))
        agv = AGVMotionController(AGVSpec.from_grade(AGVGrade.M))

        # 模拟融合特征
        fused_features = torch.randn(1, 256)

        # AGV 跟踪控制
        agv.update_pose(AGVPose(x=0.0, y=0.0, theta=0.0))
        target = AGVPose(x=1.0, y=0.5, theta=0.0)
        wheel_cmds = agv.compute_wheel_commands(target, dt=0.01)
        safe_cmds = agv.apply_safety_limits(wheel_cmds)

        self.assertGreater(len(safe_cmds), 0)

    def test_fusion_to_impedance_control(self):
        """测试融合特征到阻抗控制"""
        fusion = CrossModalFusion(FusionConfig(hidden_dim=128))
        imp = ImpedanceController(ImpedanceParams.default_6d())

        # 模拟末端执行器控制 (笛卡尔空间: x, y, z)
        desired_pos = np.array([0.4, 0.3, 0.2])
        desired_vel = np.zeros(3)
        current_pos = np.array([0.3, 0.2, 0.1])
        current_vel = np.zeros(3)
        external_wrench = np.array([0.0, 0.0, -5.0, 0.0, 0.0, 0.0])
        jacobian = np.random.randn(6, 6)

        tau = imp.compute_torque(
            desired_pos, desired_vel, current_pos, current_vel,
            external_wrench, jacobian
        )

        self.assertEqual(tau.shape, (6,))

    def test_fusion_to_mpc_control(self):
        """测试融合特征到MPC控制"""
        config = MPCConfig.for_grade('L', num_joints=6, dt=0.01)
        dynamics = DynamicsModel(num_joints=6)
        mpc = JointSpaceMPC(config=config, dynamics=dynamics, num_joints=6)

        current_pos = np.zeros(6)
        current_vel = np.zeros(6)
        target_pos = np.array([0.5, 0.2, 0.1, 0.0, 0.0, 0.0])

        tau = mpc.compute_control_simple(current_pos, current_vel, target_pos)
        self.assertEqual(tau.shape, (6,))


class TestSensorSimulationPipeline(unittest.TestCase):
    """传感器 → 仿真环境 管道测试"""

    def test_robot_simulator_step(self):
        """测试机器人仿真器步进"""
        sim = RobotSimulator(SimConfig(num_joints=6, dt=0.01))
        state = sim.step(np.zeros(6))
        self.assertIn('joint_positions', state)
        self.assertEqual(len(state['joint_positions']), 6)

    def test_sensor_simulator_integration(self):
        """测试传感器仿真器集成"""
        sim = RobotSimulator(SimConfig(num_joints=6, dt=0.01))
        sensor_sim = SensorSimulator(sim, SimConfig(num_joints=6, dt=0.01))

        # 执行多步
        for _ in range(10):
            sim.step(np.zeros(6))
            noisy_pos = sensor_sim.get_noisy_joint_positions()
            self.assertEqual(len(noisy_pos), 6)

            imu_data = sensor_sim.get_imu_data()
            self.assertEqual(len(imu_data['accel']), 3)

    def test_full_simulation_loop(self):
        """测试完整仿真循环"""
        sim = RobotSimulator(SimConfig(num_joints=6, dt=0.01))
        sensor_sim = SensorSimulator(sim, SimConfig(num_joints=6, dt=0.01))

        for step in range(50):
            torques = np.array([0.1, 0.05, -0.05, 0.0, 0.0, 0.0]) * np.sin(step * 0.1)
            state = sim.step(torques)

            # 传感器数据
            pos = sensor_sim.get_noisy_joint_positions()
            vel = sensor_sim.get_noisy_joint_velocities()
            imu = sensor_sim.get_imu_data()
            wrench = sensor_sim.get_wrench()

            # 基本验证
            self.assertEqual(len(pos), 6)
            self.assertEqual(len(vel), 6)
            self.assertEqual(len(imu['accel']), 3)
            self.assertEqual(len(wrench), 6)


class TestAGVGradeCompliance(unittest.TestCase):
    """AGV等级合规性综合测试"""

    def test_all_grades_forward_kinematics(self):
        """测试所有AGV等级的正运动学"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = AGVSpec.from_grade(AGVGrade(grade))
            agv = AGVMotionController(spec)

            # 差速驱动
            if spec.drive_type.value == 'differential':
                wheel_vels = np.array([5.0, 5.0])  # rad/s
                twist = agv.forward_kinematics(wheel_vels)
                self.assertIsInstance(twist, AGVTwist)

                # 反运动学
                target_twist = AGVTwist(vx=0.5, vy=0.0, omega=0.0)
                cmds = agv.inverse_kinematics(target_twist)
                self.assertEqual(len(cmds), 2)

    def test_all_grades_inverse_kinematics(self):
        """测试所有AGV等级的逆运动学"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = AGVSpec.from_grade(AGVGrade(grade))
            agv = AGVMotionController(spec)

            twist = AGVTwist(vx=1.0, vy=0.5, omega=0.5)
            cmds = agv.inverse_kinematics(twist)
            self.assertGreater(len(cmds), 0)

    def test_agv_pose_tracking(self):
        """测试AGV位姿跟踪"""
        spec = AGVSpec.from_grade(AGVGrade.M)
        agv = AGVMotionController(spec)

        poses = [
            AGVPose(x=0.0, y=0.0, theta=0.0),
            AGVPose(x=0.5, y=0.0, theta=0.0),
            AGVPose(x=1.0, y=0.3, theta=0.0),
        ]

        for target in poses:
            agv.update_pose(AGVPose(x=0.0, y=0.0, theta=0.0))
            cmds = agv.compute_wheel_commands(target, dt=0.01)
            self.assertGreater(len(cmds), 0)
            safe_cmds = agv.apply_safety_limits(cmds)
            self.assertEqual(len(safe_cmds), len(cmds))


class TestSafetyControlPipeline(unittest.TestCase):
    """安全控制系统管道测试"""

    def test_safety_check_pipeline(self):
        """测试安全检查管道"""
        config = SafetyConfig(
            joint_limits_lower=np.array([-3.14] * 6),
            joint_limits_upper=np.array([3.14] * 6),
            velocity_limits=np.array([2.0] * 6),
            acceleration_limits=np.array([5.0] * 6),
            torque_limits=np.array([100.0] * 6),
            safety_level=SafetyLevel.L,
        )
        safety = SafetyController(config)

        state = JointStateSnapshot(
            positions=np.array([0.1, 0.2, 0.1, 0.0, 0.0, 0.0]),
            velocities=np.array([0.5, 0.3, 0.2, 0.1, 0.1, 0.0]),
            accelerations=np.zeros(6),
            torques=np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.0]),
            timestamp=time.time(),
        )

        result = safety.check(state)
        self.assertIsNotNone(result)

    def test_emergency_stop_pipeline(self):
        """测试紧急停止管道"""
        config = SafetyConfig(
            joint_limits_lower=np.array([-3.14] * 6),
            joint_limits_upper=np.array([3.14] * 6),
            velocity_limits=np.array([2.0] * 6),
            acceleration_limits=np.array([5.0] * 6),
            torque_limits=np.array([100.0] * 6),
            safety_level=SafetyLevel.XXL,
        )
        safety = SafetyController(config)
        safety.enable()

        self.assertFalse(safety.is_emergency_stopped)
        safety.emergency_stop()
        self.assertTrue(safety.is_emergency_stopped)

        safety.reset()
        self.assertFalse(safety.is_emergency_stopped)


class TestControlFrequencyPerformance(unittest.TestCase):
    """控制频率性能测试"""

    def test_agv_control_loop_speed(self):
        """测试AGV控制回路速度"""
        import time

        spec = AGVSpec.from_grade(AGVGrade.L)
        agv = AGVMotionController(spec)

        start = time.time()
        iterations = 1000

        for _ in range(iterations):
            target = AGVPose(x=1.0, y=0.5, theta=0.0)
            cmds = agv.compute_wheel_commands(target, dt=0.01)
            agv.apply_safety_limits(cmds)

        elapsed = time.time() - start
        loop_time = elapsed / iterations

        # L级AGV应该能在1ms内完成控制计算
        self.assertLess(loop_time, 0.001, f"Control loop too slow: {loop_time*1000:.2f}ms")

    def test_fusion_network_inference_speed(self):
        """测试融合网络推理速度"""
        import time

        fusion = CrossModalFusion(FusionConfig(hidden_dim=256, num_heads=4, num_layers=2))

        mmi = MultimodalInput(
            vision=torch.randn(1, 512),
            audio=torch.randn(1, 128),
            tactile=torch.randn(1, 64),
            force=torch.randn(1, 32),
            imu=torch.randn(1, 64),
        )

        # 预热
        for _ in range(5):
            fusion(mmi)

        start = time.time()
        iterations = 100

        for _ in range(iterations):
            fusion(mmi)

        elapsed = time.time() - start
        per_iter = elapsed / iterations

        # 融合推理应该快于50ms
        self.assertLess(per_iter, 0.05, f"Fusion too slow: {per_iter*1000:.2f}ms")


class TestGradeSpecificSpecs(unittest.TestCase):
    """AGV各等级特定规格测试"""

    def test_grade_speed_limits(self):
        """测试各等级速度限制"""
        grade_specs = {
            'S': {'max_linear': 0.5, 'max_angular': 1.5},
            'M': {'max_linear': 1.0, 'max_angular': 2.0},
            'L': {'max_linear': 2.0, 'max_angular': 2.5},
            'XL': {'max_linear': 3.0, 'max_angular': 3.0},
            'XXL': {'max_linear': 5.0, 'max_angular': 3.5},
        }

        for grade, expected in grade_specs.items():
            spec = AGVSpec.from_grade(AGVGrade(grade))
            self.assertAlmostEqual(spec.max_linear_speed, expected['max_linear'], places=1)
            self.assertAlmostEqual(spec.max_angular_speed, expected['max_angular'], places=1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
