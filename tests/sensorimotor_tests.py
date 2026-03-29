"""
SuperModel 传感器-执行器集成测试
=================================

端到端传感器-执行器融合测试:
- 多传感器时间同步采集
- 传感器数据 -> 融合网络 -> 运动控制
- AGV五级规格合规性验证

覆盖模块:
- sensors/ (vision, audio, tactile, force, imu)
- fusion/ (cross_modal_fusion)
- control/ (motion, agv, impedance)
- simulation/ (gym_env)
"""

import numpy as np
import torch
import sys
import time
import unittest
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.vision import BinocularCamera, StereoFrame
from sensors.audio import BinauralMic, AudioFrame
from sensors.tactile import TactileArray, TactileFrame, TactileContact, TactileSensorType
from sensors.force import ForceTorqueSensor, Wrench, ForceSensorType
from sensors.imu import IMUSensor, IMUFrame, PoseEstimator, IMUSensorType, Pose
from fusion.cross_modal_fusion import (
    CrossModalFusion, FusionConfig, MultimodalInput,
    create_multimodal_input
)
from control.motion import MotionController, JointState, ControlMode
from control.agv import AGVMotionController, AGVSpec, AGVGrade, AGVPose, DriveType
from control.impedance import ImpedanceController, ImpedanceParams
from control.safety_controller import SafetyController, SafetyConfig, SafetyLevel, JointStateSnapshot
from simulation.gym_env import SuperModelGymEnv, GymEnvConfig


class TestAGVSensorCompliance(unittest.TestCase):
    """AGV五级传感器规格合规性测试"""

    AGV_GRADE_SPECS = {
        'S': {
            'vision_resolution': (640, 480),
            'vision_fps': 30,
            'tactile_size': (8, 8),
            'tactile_sampling': 50,
            'force_axes': 3,
            'force_sampling': 100,
            'imu_type': IMUSensorType.MPU6050,
            'imu_sampling': 100,
        },
        'M': {
            'vision_resolution': (1280, 720),
            'vision_fps': 30,
            'tactile_size': (16, 16),
            'tactile_sampling': 100,
            'force_axes': 6,
            'force_sampling': 500,
            'imu_type': IMUSensorType.BMI088,
            'imu_sampling': 200,
        },
        'L': {
            'vision_resolution': (1280, 720),
            'vision_fps': 60,
            'tactile_size': (24, 24),
            'tactile_sampling': 200,
            'force_axes': 6,
            'force_sampling': 1000,
            'imu_type': IMUSensorType.BMI088,
            'imu_sampling': 500,
        },
        'XL': {
            'vision_resolution': (1920, 1080),
            'vision_fps': 90,
            'tactile_size': (32, 32),
            'tactile_sampling': 500,
            'force_axes': 6,
            'force_sampling': 2000,
            'imu_type': IMUSensorType.ADIS16470,
            'imu_sampling': 1000,
        },
        'XXL': {
            'vision_resolution': (1920, 1080),
            'vision_fps': 120,
            'tactile_size': (48, 48),
            'tactile_sampling': 1000,
            'force_axes': 6,
            'force_sampling': 5000,
            'imu_type': IMUSensorType.ADIS16470,
            'imu_sampling': 2000,
        },
    }

    def _create_grade_system(self, grade: str) -> Dict:
        """创建指定AGV等级的完整传感器系统"""
        spec = self.AGV_GRADE_SPECS[grade]

        # 视觉
        cam = BinocularCamera(
            resolution=spec['vision_resolution'],
            fps=spec['vision_fps']
        )

        # 听觉 (BinauralMic: 固定双耳)
        mic = BinauralMic(sample_rate=16000)

        # 触觉
        tactile = TactileArray(
            array_size=spec['tactile_size'],
            sensor_type=TactileSensorType.RESISTIVE
        )

        # 力觉
        sensor_type = ForceSensorType.SIX_AXIS if spec['force_axes'] == 6 else ForceSensorType.THREE_AXIS
        force = ForceTorqueSensor(sensor_type=sensor_type)

        # IMU (sampling_rate baked into constructor)
        imu = IMUSensor(sensor_type=spec['imu_type'], sample_rate=spec['imu_sampling'])

        return {
            'camera': cam,
            'mic': mic,
            'tactile': tactile,
            'force': force,
            'imu': imu,
        }

    def _test_grade_compliance(self, grade: str):
        """测试单个AGV等级的合规性"""
        spec = self.AGV_GRADE_SPECS[grade]
        sensors = self._create_grade_system(grade)

        try:
            # 打开所有传感器
            for s in sensors.values():
                self.assertTrue(s.open(), f"{grade}: 传感器打开失败")

            # 采集测试
            for i in range(3):
                # 视觉 (resolution = (width, height), image shape = (height, width))
                frame = sensors['camera'].capture()
                self.assertIsInstance(frame, StereoFrame)
                img_h, img_w = frame.left_image.shape[:2]
                expected_h, expected_w = spec['vision_resolution'][1], spec['vision_resolution'][0]
                self.assertEqual((img_h, img_w), (expected_h, expected_w))

                # 触觉
                tac = sensors['tactile'].capture()
                self.assertIsInstance(tac, TactileFrame)
                self.assertEqual(tac.pressure_map.shape, spec['tactile_size'])

                # 力觉
                wrench = sensors['force'].capture()
                self.assertIsInstance(wrench, Wrench)

                # IMU
                imu_frame = sensors['imu'].capture()
                self.assertIsInstance(imu_frame, IMUFrame)

        finally:
            for s in sensors.values():
                s.close()

    def test_grade_S_compliance(self):
        """S级传感器合规性测试"""
        self._test_grade_compliance('S')

    def test_grade_M_compliance(self):
        """M级传感器合规性测试"""
        self._test_grade_compliance('M')

    def test_grade_L_compliance(self):
        """L级传感器合规性测试"""
        self._test_grade_compliance('L')

    def test_grade_XL_compliance(self):
        """XL级传感器合规性测试"""
        self._test_grade_compliance('XL')

    def test_grade_XXL_compliance(self):
        """XXL级传感器合规性测试"""
        self._test_grade_compliance('XXL')


class TestSensorimotorFusion(unittest.TestCase):
    """传感器-执行器融合测试"""

    def setUp(self):
        self.fusion = CrossModalFusion(FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256,
            num_heads=4, num_layers=2
        ))
        self.controller = MotionController(num_joints=6, control_rate=100.0)

    def test_tactile_feedback_control(self):
        """触觉反馈控制: 接触检测 -> 力调节"""
        # 模拟触觉接触 (编码后特征: tactile_dim=64)
        tactile_data = torch.randn(1, 64)
        force_data = torch.randn(1, 32)
        imu_data = torch.randn(1, 64)

        # 融合特征
        mmi = MultimodalInput(
            vision=torch.randn(1, 512),
            audio=torch.randn(1, 128),  # 编码后特征
            tactile=tactile_data,
            force=force_data,
            imu=imu_data,
            language=None
        )
        fused = self.fusion(mmi)
        self.assertEqual(fused.shape, (1, 256))

        # 基于触觉调整控制
        ctrl_mode = ControlMode.JOINT_POSITION

        target = np.random.randn(6).astype(np.float64)
        ctrl_out = self.controller.step(target, ctrl_mode)
        self.assertEqual(ctrl_out.shape[0], 6)

    def test_force_based_impedance(self):
        """基于力觉的阻抗控制"""
        # 模拟外力
        wrench = Wrench(
            force=np.array([5.0, 0.0, 10.0]),
            torque=np.array([0.5, 0.0, 0.0])
        )

        # 阻抗参数 (6x6 矩阵)
        imp_params = ImpedanceParams.default_6d()
        imp_ctrl = ImpedanceController(imp_params, control_rate=100.0)

        # 计算补偿力 (阻抗控制只使用前3维: x,y,z)
        desired_pos = np.zeros(3)
        desired_vel = np.zeros(3)
        current_pos = np.array([0.01, 0.0, 0.0])
        current_vel = np.zeros(3)

        wrench_vec = wrench.to_vector()
        tau = imp_ctrl.compute_torque(
            desired_pos, desired_vel,
            current_pos, current_vel,
            wrench_vec, np.eye(6)
        )
        self.assertEqual(tau.shape[0], 6)

    def test_imu_pose_estimation_closed_loop(self):
        """IMU姿态估计闭环控制"""
        estimator = PoseEstimator(sample_rate=100.0, algorithm='complementary')
        estimator.reset()

        # 模拟姿态变化
        for t in range(50):
            accel = np.array([0.0, 0.0, 9.81]) + np.random.randn(3) * 0.1
            gyro = np.array([0.0, 0.0, 0.1]) + np.random.randn(3) * 0.05
            pose = estimator.update(accel, gyro)
            self.assertIsInstance(pose, Pose)
            self.assertEqual(len(pose.orientation), 4)

        # 验证姿态收敛
        euler = pose.to_euler()
        self.assertAlmostEqual(euler[2], 0.1 * 50 * (1/100), delta=0.3)

    def test_agv_motion_with_sensor_feedback(self):
        """AGV运动控制 + 传感器反馈"""
        spec = AGVSpec.from_grade(AGVGrade.M)
        agv = AGVMotionController(spec)

        # 初始位姿
        self.assertEqual(agv.pose.x, 0.0)
        self.assertEqual(agv.pose.y, 0.0)

        # 目标位姿
        target = AGVPose(x=0.5, y=0.3, theta=0.0)

        # 计算轮速命令
        wheel_cmds = agv.compute_wheel_commands(target, dt=0.01)
        self.assertGreater(wheel_cmds.size, 0)

        # 验证输出在合理范围内
        self.assertTrue(np.all(np.isfinite(wheel_cmds)))


class TestMultimodalTemporalFusion(unittest.TestCase):
    """时序多模态融合测试"""

    def setUp(self):
        self.fusion = CrossModalFusion(FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256
        ))
        self.sequence_length = 10

    def test_temporal_sequence_fusion(self):
        """时序序列融合"""
        fused_sequence = []

        for t in range(self.sequence_length):
            # 各模态使用编码后特征 (2D: B x feature_dim)
            mmi = MultimodalInput(
                vision=torch.randn(1, 512),
                audio=torch.randn(1, 128),
                tactile=torch.randn(1, 64),
                force=torch.randn(1, 32),
                imu=torch.randn(1, 64),
                language=None
            )
            fused = self.fusion(mmi)
            fused_sequence.append(fused)

        # 验证序列长度
        self.assertEqual(len(fused_sequence), self.sequence_length)

        # 验证时序平滑性 (连续帧差异应有限)
        for i in range(1, len(fused_sequence)):
            diff = torch.norm(fused_sequence[i] - fused_sequence[i-1])
            self.assertLess(diff.item(), 20.0)

    def test_modality_dropout_robustness(self):
        """模态缺失鲁棒性测试"""
        # 仅触觉+IMU
        mmi_partial = MultimodalInput(
            vision=None,
            audio=None,
            tactile=torch.randn(1, 64),
            force=torch.randn(1, 32),
            imu=torch.randn(1, 64),
            language=None
        )
        fused = self.fusion(mmi_partial)
        self.assertEqual(fused.shape[1], 256)

        # 全模态
        mmi_full = MultimodalInput(
            vision=torch.randn(1, 512),
            audio=torch.randn(1, 128),
            tactile=torch.randn(1, 64),
            force=torch.randn(1, 32),
            imu=torch.randn(1, 64),
            language=torch.randint(0, 10000, (1, 32))
        )
        fused_full = self.fusion(mmi_full)
        self.assertEqual(fused_full.shape[1], 256)

        # 全模态和部分模态输出维度应一致
        self.assertEqual(fused.shape, fused_full.shape)


class TestGymEnvSensorIntegration(unittest.TestCase):
    """Gym环境传感器集成测试"""

    def test_gym_env_with_grade_M(self):
        """M级Gym环境测试"""
        config = GymEnvConfig(grade='M', dt=0.01)
        env = SuperModelGymEnv(config=config, render_mode=None)

        obs, info = env.reset(seed=42)
        self.assertGreater(len(obs), 0)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertGreater(len(obs), 0)
        self.assertIsInstance(reward, float)

        env.close()

    def test_gym_env_action_space(self):
        """Gym环境动作空间测试"""
        for grade in ['S', 'M', 'L']:
            config = GymEnvConfig(grade=grade)
            env = SuperModelGymEnv(config=config)

            action = env.action_space.sample()
            self.assertEqual(action.shape, (6,))

            env.close()

    def test_gym_env_observation_space(self):
        """Gym环境观测空间测试"""
        config = GymEnvConfig(grade='M', dt=0.01)
        env = SuperModelGymEnv(config=config)

        obs_space = env.observation_space
        self.assertIsNotNone(obs_space)

        action_space = env.action_space
        self.assertIsNotNone(action_space)

        env.close()


class TestSafetyControllerGrades(unittest.TestCase):
    """安全控制器AGV等级测试"""

    def test_safety_level_M(self):
        """M级安全控制器"""
        config = SafetyConfig(
            joint_limits_lower=np.array([-3.14]*6),
            joint_limits_upper=np.array([3.14]*6),
            velocity_limits=np.array([2.0]*6),
            acceleration_limits=np.array([5.0]*6),
            torque_limits=np.array([100.0]*6),
            safety_level=SafetyLevel.M,
        )
        safety = SafetyController(config)
        self.assertEqual(safety.safety_level, SafetyLevel.M)

        # 模拟安全检查
        state = JointStateSnapshot(
            positions=np.array([0.1]*6),
            velocities=np.array([0.5]*6),
            accelerations=np.array([0.1]*6),
            torques=np.array([10.0]*6),
            timestamp=time.time()
        )
        result = safety.check(state)
        self.assertTrue(result.safe)

    def test_safety_level_XL(self):
        """XL级安全控制器 (碰撞预测)"""
        config = SafetyConfig(
            joint_limits_lower=np.array([-3.14]*6),
            joint_limits_upper=np.array([3.14]*6),
            velocity_limits=np.array([3.0]*6),
            acceleration_limits=np.array([10.0]*6),
            torque_limits=np.array([200.0]*6),
            collision_threshold=50.0,
            safety_level=SafetyLevel.XL,
        )
        safety = SafetyController(config)
        self.assertEqual(safety.safety_level, SafetyLevel.XL)

        # 高速状态
        state = JointStateSnapshot(
            positions=np.array([0.1]*6),
            velocities=np.array([2.5]*6),
            accelerations=np.array([1.0]*6),
            torques=np.array([150.0]*6),
            timestamp=time.time()
        )
        result = safety.check(state)
        self.assertTrue(result.safe)


class TestCrossModalFusionEdgeCases(unittest.TestCase):
    """跨模态融合边界情况测试"""

    def setUp(self):
        self.fusion = CrossModalFusion(FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256
        ))

    def test_single_modality_each(self):
        """单模态输入测试 (各模态独立)"""
        modalities = {
            'vision': torch.randn(2, 512),
            'audio': torch.randn(2, 128),
            'tactile': torch.randn(2, 64),
            'force': torch.randn(2, 32),
            'imu': torch.randn(2, 64),
        }

        for name, data in modalities.items():
            kwargs = {k: None for k in modalities}
            kwargs[name] = data
            mmi = MultimodalInput(**kwargs)
            fused = self.fusion(mmi)
            self.assertEqual(fused.shape[0], 2)  # batch size preserved

    def test_batch_size_consistency(self):
        """批大小一致性测试"""
        for batch_size in [1, 4, 16, 32]:
            mmi = MultimodalInput(
                vision=torch.randn(batch_size, 512),
                audio=torch.randn(batch_size, 128),
                tactile=torch.randn(batch_size, 64),
                force=torch.randn(batch_size, 32),
                imu=torch.randn(batch_size, 64),
                language=torch.randint(0, 10000, (batch_size, 32))
            )
            fused = self.fusion(mmi)
            self.assertEqual(fused.shape[0], batch_size)

    def test_numpy_input_factory(self):
        """NumPy输入工厂函数测试"""
        import numpy as np

        mmi = create_multimodal_input(
            vision=np.random.randn(4, 512).astype(np.float32),
            audio=np.random.randn(4, 128).astype(np.float32),
            tactile=np.random.randn(4, 64).astype(np.float32),
            force=np.random.randn(4, 32).astype(np.float32),
            imu=np.random.randn(4, 64).astype(np.float32),
            language=np.random.randint(0, 10000, (4, 32))
        )

        # 验证可融合
        fused = self.fusion(mmi)
        self.assertEqual(fused.shape[0], 4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
