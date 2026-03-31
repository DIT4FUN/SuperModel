"""
具身智能全链路测试
==================

端到端测试: 传感器采集 → 多模态融合 → 决策规划 → 运动控制 → 仿真反馈

测试覆盖:
- 全模态传感器同步采集
- 跨模态特征融合
- 自主学习与决策
- 运动控制执行
- 仿真环境反馈

版本: v1.15.0
"""

import numpy as np
import sys
import time
import unittest
import torch

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.vision import BinocularCamera, StereoFrame
from sensors.audio import BinauralMic, AudioFrame
from sensors.tactile import TactileArray, TactileFrame, TactileSensorType
from sensors.force import ForceTorqueSensor, Wrench, ForceSensorType
from sensors.imu import IMUSensor, IMUFrame, IMUSensorType
from sensors.manager import SensorManager, SensorManagerConfig

from fusion.cross_modal_fusion import (
    CrossModalFusion, FusionConfig, FusionStrategy,
    MultimodalInput
)

from control.agv import AGVMotionController, AGVGrade, AGVSpec, AGVPose, AGVTwist, DriveType
from control.safety_controller import SafetyController, SafetyConfig, SafetyLevel, JointStateSnapshot
from control.motion import MotionController, JointState, TwistCommand
from control.obstacle_avoidance import (
    ObstacleAvoider, AvoidanceStrategy, Obstacle,
    DWAConfig, AvoidanceConfig
)

from simulation.gym_env import SuperModelGymEnv, GymEnvConfig


class TestEmbodiedSensorPipeline(unittest.TestCase):
    """测试具身传感全链路"""

    def test_sensor_manager_full_init(self):
        """测试传感器管理器全模态初始化"""
        config = SensorManagerConfig(grade="M")
        manager = SensorManager(config)
        self.assertEqual(manager.config.grade.value, "M")
        self.assertTrue(manager.config.vision_enabled)
        self.assertTrue(manager.config.force_enabled)
        self.assertTrue(manager.config.imu_enabled)
        manager.close_all()

    def test_multimodal_sensor_simulation(self):
        """测试多模态传感器同步模拟"""
        # 视觉
        cam = BinocularCamera(resolution=(640, 480), fps=30)
        cam.open()
        frame = cam.capture()
        self.assertIsInstance(frame, StereoFrame)
        self.assertEqual(frame.left_image.shape[:2], (480, 640))
        cam.close()

        # 听觉 (自动回退到模拟模式)
        mic = BinauralMic(sample_rate=16000)
        mic.open()
        audio = mic.capture()
        self.assertIsInstance(audio, AudioFrame)
        self.assertEqual(audio.left_channel.shape[0], 512)
        mic.close()

        # 触觉
        tactile = TactileArray(array_size=(16, 16))
        tactile.open()
        tf = tactile.capture()
        self.assertIsInstance(tf, TactileFrame)
        self.assertEqual(tf.pressure_map.shape, (16, 16))
        tactile.close()

        # 力觉 (六轴力矩传感器)
        force = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        force.open()
        wrench = force.capture()
        self.assertIsInstance(wrench, Wrench)
        self.assertEqual(len(wrench.force), 3)
        self.assertEqual(len(wrench.torque), 3)
        force.close()

        # IMU
        imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL)
        imu.open()
        imu_frame = imu.capture()
        self.assertIsInstance(imu_frame, IMUFrame)
        self.assertEqual(imu_frame.accel.shape, (3,))
        self.assertEqual(imu_frame.gyro.shape, (3,))
        imu.close()

    def test_sensor_timing_consistency(self):
        """测试传感器时序一致性"""
        sensors = [
            TactileArray(array_size=(16, 16)),
            ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS),
            IMUSensor(sensor_type=IMUSensorType.VIRTUAL),
        ]
        timestamps = []
        for s in sensors:
            s.open()
            t0 = time.perf_counter()
            s.capture()
            timestamps.append(time.perf_counter() - t0)
            s.close()
        for dt in timestamps:
            self.assertLess(dt, 0.01)


class TestCrossModalFusionPipeline(unittest.TestCase):
    """测试跨模态融合链路"""

    def test_fusion_config_defaults(self):
        """测试融合配置默认值"""
        config = FusionConfig()
        self.assertEqual(config.vision_dim, 512)
        self.assertEqual(config.audio_dim, 128)
        self.assertEqual(config.hidden_dim, 256)

    def test_five_modality_fusion(self):
        """测试五模态融合"""
        config = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256,
            num_heads=4, num_layers=2, strategy=FusionStrategy.HYBRID
        )
        fusion = CrossModalFusion(config)

        batch_size = 2
        vision = torch.randn(batch_size, 512)
        audio = torch.randn(batch_size, 128)
        tactile = torch.randn(batch_size, 64)
        force = torch.randn(batch_size, 32)
        imu = torch.randn(batch_size, 64)

        mm_input = MultimodalInput(
            vision=vision, audio=audio, tactile=tactile,
            force=force, imu=imu
        )
        output = fusion(mm_input)
        # output is torch.Tensor of shape (batch, hidden_dim)
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], 256)

    def test_fusion_with_language(self):
        """测试带语言模态的融合"""
        config = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, language_dim=768,
            hidden_dim=256, num_heads=4, num_layers=2
        )
        fusion = CrossModalFusion(config)
        vision = torch.randn(2, 512)
        lang = torch.randint(0, 10000, (2, 32))  # language must be Long tensor for embedding
        mm_input = MultimodalInput(vision=vision, language=lang)
        output = fusion(mm_input)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 256)


class TestAGVControlPipeline(unittest.TestCase):
    """测试AGV控制链路"""

    def test_agv_all_grades(self):
        """测试AGV全等级规格"""
        for grade in AGVGrade:
            spec = AGVSpec.from_grade(grade)
            controller = AGVMotionController(spec)
            self.assertEqual(controller.spec.grade, grade)
            self.assertGreater(controller.spec.max_linear_speed, 0)

    def test_agv_forward_inverse_kinematics(self):
        """测试AGV正逆运动学"""
        spec = AGVSpec.from_grade(AGVGrade.M)
        controller = AGVMotionController(spec)
        twist = AGVTwist(vx=0.5, vy=0.0, omega=0.0)
        wheel_cmd = controller.inverse_kinematics(twist)
        self.assertGreater(len(wheel_cmd), 0)

    def test_agv_pose_update(self):
        """测试AGV位姿更新"""
        spec = AGVSpec.from_grade(AGVGrade.M)
        controller = AGVMotionController(spec)
        new_pose = AGVPose(x=1.0, y=2.0, theta=0.5)
        controller.update_pose(new_pose)
        self.assertEqual(controller.pose.x, 1.0)
        self.assertEqual(controller.pose.y, 2.0)

    def test_safety_controller_basic(self):
        """测试安全控制器基础功能"""
        config = SafetyConfig(
            joint_limits_lower=np.array([-3.14] * 6),
            joint_limits_upper=np.array([3.14] * 6),
            velocity_limits=np.array([2.0] * 6),
            acceleration_limits=np.array([5.0] * 6),
            torque_limits=np.array([100.0] * 6),
            safety_level=SafetyLevel.L,
        )
        safety = SafetyController(config)
        self.assertEqual(safety.safety_level, SafetyLevel.L)

    def test_obstacle_avoidance_dwa(self):
        """测试DWA避障"""
        cfg = AvoidanceConfig(strategy=AvoidanceStrategy.DWA)
        avoider = ObstacleAvoider(cfg)
        obstacles = [Obstacle(position=np.array([1.0, 0.5]), radius=0.3)]
        cmd = avoider.compute_command(
            robot_pose=np.array([0.0, 0.0, 0.0]),
            robot_velocity=np.array([0.0, 0.0, 0.0]),
            goal=np.array([3.0, 0.0]),
            obstacles=obstacles,
            dt=0.1
        )
        self.assertGreaterEqual(cmd.vx, -1.0)
        self.assertLessEqual(cmd.vx, 1.0)

    def test_obstacle_avoidance_apf(self):
        """测试APF避障"""
        cfg = AvoidanceConfig(strategy=AvoidanceStrategy.APF)
        avoider = ObstacleAvoider(cfg)
        obstacles = [Obstacle(position=np.array([1.5, 0.0]), radius=0.3)]
        cmd = avoider.compute_command(
            robot_pose=np.array([0.0, 0.0, 0.0]),
            robot_velocity=np.array([0.0, 0.0, 0.0]),
            goal=np.array([3.0, 0.0]),
            obstacles=obstacles,
            dt=0.1
        )
        self.assertIsInstance(cmd.vx, float)


class TestMotionControlPipeline(unittest.TestCase):
    """测试运动控制链路"""

    def test_motion_controller_basic(self):
        """测试运动控制器基础功能"""
        controller = MotionController(num_joints=6, control_rate=100.0)
        controller._current_joint_pos = np.zeros(6)
        controller._current_joint_vel = np.zeros(6)
        target = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
        torque = controller.compute_joint_torque(target)
        self.assertEqual(torque.shape, (6,))


class TestSimulationPipeline(unittest.TestCase):
    """测试仿真链路"""

    def test_gym_env_reset(self):
        """测试Gym环境重置"""
        config = GymEnvConfig(num_joints=6, grade="M")
        env = SuperModelGymEnv(config)
        obs, info = env.reset(seed=42)
        self.assertGreater(len(obs), 20)
        env.close()

    def test_gym_env_step(self):
        """测试Gym环境步进"""
        config = GymEnvConfig(num_joints=6, grade="M", episode_length=100)
        env = SuperModelGymEnv(config)
        obs, info = env.reset(seed=42)
        action = np.zeros(6)
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        env.close()


class TestEndToEndEmbodiedLoop(unittest.TestCase):
    """端到端具身智能闭环测试"""

    def test_full_sensorimotor_loop(self):
        """测试完整传感-运动闭环"""
        # 1. 初始化传感器
        sensors = [
            BinocularCamera(resolution=(640, 480), fps=30),
            IMUSensor(sensor_type=IMUSensorType.VIRTUAL),
            TactileArray(array_size=(16, 16)),
        ]
        for s in sensors:
            s.open()

        # 2. 采集多模态感知
        vision_frame = sensors[0].capture()
        imu_frame = sensors[1].capture()
        tactile_frame = sensors[2].capture()

        # 3. 跨模态融合 (需要torch tensor)
        config = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256,
            num_heads=4, num_layers=2
        )
        fusion = CrossModalFusion(config)
        mm_input = MultimodalInput(
            vision=torch.randn(1, 512),
            imu=torch.randn(1, 64),
            tactile=torch.randn(1, 64),
        )
        fused = fusion(mm_input)
        # output is torch.Tensor of shape (batch, hidden_dim)
        self.assertEqual(fused.shape[1], 256)

        # 4. 运动控制
        motion = MotionController(num_joints=6, control_rate=100.0)
        motion._current_joint_pos = np.zeros(6)
        motion._current_joint_vel = np.zeros(6)
        q_target = np.array([0.1] * 6)
        cmd = motion.compute_joint_torque(q_target)
        self.assertEqual(cmd.shape, (6,))

        # 5. 安全检查
        safety_config = SafetyConfig(
            joint_limits_lower=np.array([-3.14] * 6),
            joint_limits_upper=np.array([3.14] * 6),
            velocity_limits=np.array([2.0] * 6),
            acceleration_limits=np.array([5.0] * 6),
            torque_limits=np.array([100.0] * 6),
            safety_level=SafetyLevel.L,
        )
        safety = SafetyController(safety_config)
        snap = JointStateSnapshot(
            positions=np.zeros(6),
            velocities=np.zeros(6),
        )
        result = safety.check(snap)
        self.assertTrue(result.safe)

        for s in sensors:
            s.close()

    def test_agv_obstacle_avoidance_loop(self):
        """测试AGV避障闭环"""
        spec = AGVSpec.from_grade(AGVGrade.L)
        controller = AGVMotionController(spec)
        cfg = AvoidanceConfig(strategy=AvoidanceStrategy.DWA)
        avoider = ObstacleAvoider(cfg)

        current_pose = np.array([0.0, 0.0, 0.0])
        target = np.array([5.0, 0.0])
        obstacles = [
            Obstacle(position=np.array([2.0, 0.3]), radius=0.5),
            Obstacle(position=np.array([3.5, -0.4]), radius=0.4),
        ]

        for step in range(50):
            vel = avoider.compute_command(
                robot_pose=current_pose,
                robot_velocity=np.array([0.0, 0.0, 0.0]),
                goal=target,
                obstacles=obstacles,
                dt=0.1
            )
            vx = np.clip(vel.vx, -2.0, 2.0)
            current_pose[0] += vx * 0.1
            dist = np.linalg.norm(current_pose[:2] - target)
            if dist < 0.2:
                break

        # 验证避障后位置在合理范围内
        self.assertLess(current_pose[0], 6.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
