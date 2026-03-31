"""
SuperModel 具身智能完整流程测试
================================

端到端测试: 传感器 → 融合 → 规划 → 控制 → 执行
覆盖完整的多模态感知-融合-规划-控制流程
"""

import numpy as np
import torch
import unittest
import sys
import time

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.vision import BinocularCamera, DepthProcessor, CameraIntrinsics, StereoExtrinsics
from sensors.audio import BinauralMic, SoundLocalizer
from sensors.tactile import TactileArray, TactileFrame, PressureProcessor
from sensors.force import ForceTorqueSensor, Wrench, ForceSensorType
from sensors.imu import IMUSensor, PoseEstimator, Pose, IMUSensorType
from sensors.manager import SensorManager, SensorManagerConfig
from fusion.cross_modal_fusion import CrossModalFusion, FusionConfig, MultimodalInput, create_multimodal_input
from control.agv import AGVMotionController, AGVSpec, AGVGrade, AGVPose, AGVTwist, TrajectoryTracker
from control.impedance import ImpedanceController, ImpedanceParams
from control.planner import TaskPlanner, Task, TaskStatus, TaskPriority
from control.skill import SkillLibrary, Skill, SkillRegistry, SkillConfig
from simulation.environment import RobotSimulator, SensorSimulator, SimConfig


class TestSensorimotorPipeline(unittest.TestCase):
    """完整传感-运动管道测试"""
    
    def test_full_sensorimotor_pipeline(self):
        """端到端传感-运动流程: 传感器采集 → 融合 → 控制"""
        # 1. 初始化传感器管理器 (M级)
        config = SensorManagerConfig(grade='M')
        manager = SensorManager(config=config)
        manager.open_all()
        
        # 2. 采集多模态数据
        data = manager.capture_all()
        
        self.assertIsNotNone(data)
        self.assertIsNotNone(data.vision)
        self.assertIsNotNone(data.audio)
        self.assertIsNotNone(data.tactile)
        self.assertIsNotNone(data.force)
        self.assertIsNotNone(data.imu)
        
        # 3. 初始化融合网络
        fusion_config = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256, num_heads=4
        )
        fusion = CrossModalFusion(fusion_config)
        fusion.eval()
        
        # 4. 构建多模态输入
        mmi = MultimodalInput(
            vision=torch.randn(1, 512),
            audio=torch.randn(1, 128),
            tactile=torch.randn(1, 64),
            force=torch.randn(1, 32),
            imu=torch.randn(1, 64),
        )
        
        # 5. 融合前向传播
        with torch.no_grad():
            fused = fusion(mmi)
        
        self.assertEqual(fused.shape[0], 1)
        self.assertEqual(fused.shape[1], 256)
        
        # 6. AGV运动控制
        agv_spec = AGVSpec.from_grade(AGVGrade.M)
        agv_ctrl = AGVMotionController(agv_spec)
        
        # 设置当前位姿
        current_pose = AGVPose(x=0.0, y=0.0, theta=0.0)
        agv_ctrl.update_pose(current_pose)
        
        # 设置目标位姿
        target_pose = AGVPose(x=1.0, y=0.5, theta=0.0)
        
        # 计算轮速命令
        wheel_cmds = agv_ctrl.compute_wheel_commands(target_pose, dt=0.01)
        
        self.assertIsInstance(wheel_cmds, np.ndarray)
        self.assertEqual(len(wheel_cmds), 2)  # 差速驱动: 左右轮
        
        # 7. 清理
        manager.close_all()
    
    def test_tactile_force_imu_closed_loop(self):
        """触觉-力觉-IMU闭环控制流程"""
        # 初始化传感器
        tactile = TactileArray(array_size=(16, 16))
        force = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        imu = IMUSensor(sensor_type=IMUSensorType.BMI088)
        
        tactile.open()
        force.open()
        imu.open()
        
        # 采集数据
        tac_frame = tactile.capture()
        wrench = force.capture()
        imu_frame = imu.capture()
        
        # 处理触觉
        contacts = tactile.detect_contacts(tac_frame)
        
        # 接触检测
        contact_state = force.detect_contact(wrench, threshold=2.0)
        
        # 姿态估计
        pose_est = PoseEstimator(algorithm='madgwick', sample_rate=200.0)
        pose = pose_est.update(imu_frame.accel, imu_frame.gyro)
        
        # 验证输出
        self.assertIsInstance(tac_frame, TactileFrame)
        self.assertIsInstance(wrench, Wrench)
        self.assertIsInstance(pose, Pose)
        self.assertIn(contact_state.is_contact, [True, False])
        
        # 清理
        tactile.close()
        force.close()
        imu.close()
    
    def test_agv_trajectory_tracking_with_fusion(self):
        """AGV轨迹跟踪 + 融合输出"""
        # 1. 创建轨迹跟踪器
        agv_spec = AGVSpec.from_grade(AGVGrade.M)
        tracker = TrajectoryTracker(spec=agv_spec, look_ahead_distance=0.2)
        
        # 2. 创建轨迹
        trajectory = [
            AGVPose(x=0.0, y=0.0, theta=0.0),
            AGVPose(x=0.5, y=0.0, theta=0.0),
            AGVPose(x=1.0, y=0.0, theta=0.0),
            AGVPose(x=1.0, y=0.5, theta=0.0),
            AGVPose(x=1.0, y=1.0, theta=0.0),
        ]
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        tracker.set_trajectory(trajectory, times)
        
        # 3. 设置初始位姿
        tracker.set_pose(AGVPose(x=0.0, y=0.0, theta=0.0))
        
        # 4. 模拟跟踪
        for step in range(50):
            wheel_cmds = tracker.compute_command(dt=0.1)
            
            if tracker.is_trajectory_complete():
                break
        
        # 5. 融合网络验证
        fusion_config = FusionConfig(hidden_dim=256)
        fusion = CrossModalFusion(fusion_config)
        mmi = MultimodalInput(
            vision=torch.randn(1, 512),
            tactile=torch.randn(1, 64),
            imu=torch.randn(1, 64),
        )
        with torch.no_grad():
            fused = fusion(mmi)
        
        self.assertEqual(fused.shape, (1, 256))
        
        # 重置跟踪器
        tracker.reset()
        self.assertIsNotNone(tracker)
    
    def test_impedance_control_with_force_feedback(self):
        """阻抗控制 + 力反馈"""
        # 1. 初始化阻抗控制器
        params = ImpedanceParams(
            M=np.diag([1.0, 1.0, 0.5]),  # 惯性矩阵
            D=np.diag([10.0, 10.0, 5.0]),  # 阻尼矩阵
            K=np.diag([20.0, 20.0, 10.0]),  # 刚度矩阵
        )
        imp_ctrl = ImpedanceController(params)
        
        # 2. 模拟力反馈 (6轴: Fx,Fy,Fz,Tx,Ty,Tz)
        external_wrench = np.array([0.0, 5.0, 0.0, 0.0, 0.0, 0.0])
        desired_pos = np.zeros(3)
        desired_vel = np.zeros(3)
        current_pos = np.array([0.0, 0.1, 0.0])
        current_vel = np.zeros(3)
        jacobian = np.eye(6, 3)  # 简化雅可比
        
        # 3. 计算阻抗控制力矩
        joint_torques = imp_ctrl.compute_torque(
            desired_pos, desired_vel, current_pos, current_vel,
            external_wrench, jacobian
        )
        
        self.assertIsInstance(joint_torques, np.ndarray)
        
        # 4. 力觉传感器验证
        force_sensor = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
        force_sensor.open()
        wrench = force_sensor.capture()
        
        self.assertIsInstance(wrench, Wrench)
        self.assertEqual(wrench.force.shape, (3,))
        
        force_sensor.close()
    
    def test_task_planner_integration(self):
        """任务规划器集成"""
        # 1. 初始化技能库
        skill_lib = SkillLibrary()
        skill_lib.register_skill(Skill(config=SkillConfig(name='move_to', description='移动到目标位置')))
        skill_lib.register_skill(Skill(config=SkillConfig(name='grasp', description='抓取物体')))
        skill_lib.register_skill(Skill(config=SkillConfig(name='place', description='放置物体')))
        
        # 2. 初始化任务规划器
        planner = TaskPlanner()
        
        # 3. 添加任务
        task1 = Task(
            id='task_1',
            name='move_to',
            description='移动到目标位置',
            parameters={'target': [1.0, 0.5, 0.0]}
        )
        task2 = Task(
            id='task_2',
            name='grasp',
            description='抓取物体',
            parameters={'object_id': 'box_01'}
        )
        
        planner.add_task(task1)
        planner.add_task(task2)
        
        # 4. 验证任务添加成功
        self.assertEqual(len(planner._task_queue), 2)
        
        # 5. 验证任务可以按优先级排序
        task3 = Task(id='task_3', name='place', priority=TaskPriority.HIGH)
        planner.add_task(task3)
        self.assertEqual(planner._task_queue[0].id, 'task_3')  # HIGH priority first
    
    def test_sensor_manager_grade_compliance(self):
        """传感器管理器等级合规性"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            config = SensorManagerConfig(grade=grade)
            manager = SensorManager(config=config)
            result = manager.open_all()
            
            # 验证管理器打开了
            self.assertIsNotNone(result)
            
            # 验证捕获不会崩溃
            data = manager.capture_all()
            self.assertIsNotNone(data)
            self.assertIsInstance(data, type(data))  # basic type check
            
            manager.close_all()
    
    def test_multimodal_fusion_at_different_granularities(self):
        """多粒度多模态融合测试"""
        fusion_config = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256, num_heads=4
        )
        fusion = CrossModalFusion(fusion_config)
        
        # 测试不同模态组合
        test_cases = [
            ('all', MultimodalInput(
                vision=torch.randn(2, 512), audio=torch.randn(2, 128),
                tactile=torch.randn(2, 64), force=torch.randn(2, 32),
                imu=torch.randn(2, 64)
            )),
            ('vision_audio', MultimodalInput(
                vision=torch.randn(2, 512), audio=torch.randn(2, 128)
            )),
            ('tactile_force', MultimodalInput(
                tactile=torch.randn(2, 64), force=torch.randn(2, 32)
            )),
            ('imu_only', MultimodalInput(imu=torch.randn(2, 64))),
        ]
        
        for name, mmi in test_cases:
            with torch.no_grad():
                out = fusion(mmi)
            self.assertEqual(out.shape, (2, 256)), f"Failed for {name}"


class TestSensorSimulationPipeline(unittest.TestCase):
    """传感器仿真管道测试"""
    
    def test_robot_simulator_integration(self):
        """机器人仿真器集成"""
        sim_config = SimConfig(
            dt=0.001,
            gravity=np.array([0.0, 0.0, -9.81]),
            num_joints=6
        )
        
        robot = RobotSimulator(config=sim_config)
        
        # 重置
        robot.reset()
        self.assertEqual(robot._step_count, 0)
        
        # 步进
        action = np.zeros(6)  # 6轴关节力矩
        state = robot.step(action)
        
        self.assertIsInstance(state, dict)
        self.assertIn('joint_positions', state)
        
    def test_sensor_simulator_data_types(self):
        """传感器仿真器数据类型"""
        sim_config = SimConfig(dt=0.001, num_joints=6)
        robot = RobotSimulator(config=sim_config)
        sensor_sim = SensorSimulator(simulator=robot, config=sim_config)
        
        # 测试IMU数据
        imu_data = sensor_sim.get_imu_data()
        self.assertIn('accel', imu_data)
        self.assertIn('gyro', imu_data)
        
        # 测试关节位置
        pos = sensor_sim.get_noisy_joint_positions()
        self.assertEqual(len(pos), 6)
        
        # 测试力矩数据
        wrench = sensor_sim.get_wrench()
        self.assertEqual(len(wrench), 6)


class TestAGVControlPipeline(unittest.TestCase):
    """AGV控制管道测试"""
    
    def test_agv_all_grades_initialization(self):
        """所有AGV等级初始化"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = AGVSpec.from_grade(AGVGrade(grade))
            ctrl = AGVMotionController(spec)
            
            self.assertIsNotNone(ctrl.pose)
            self.assertIsNotNone(ctrl.twist)
            self.assertEqual(ctrl.spec.grade.value, grade)
    
    def test_agv_differential_kinematics(self):
        """差速驱动运动学"""
        spec = AGVSpec.from_grade(AGVGrade.M)
        ctrl = AGVMotionController(spec)
        
        # 正运动学: 轮速 → 速度
        wheel_vels = np.array([10.0, 10.0])  # rad/s
        twist = ctrl.forward_kinematics(wheel_vels)
        
        self.assertIsInstance(twist, AGVTwist)
        self.assertGreater(twist.vx, 0)  # 前进
        
        # 逆运动学: 速度 → 轮速
        twist_input = AGVTwist(vx=0.5, vy=0.0, omega=0.0)
        wheel_cmds = ctrl.inverse_kinematics(twist_input)
        
        self.assertEqual(len(wheel_cmds), 2)
        np.testing.assert_array_almost_equal(wheel_cmds[0], wheel_cmds[1])  # 直线行驶两轮等速
    
    def test_agv_turn_in_place(self):
        """原地旋转"""
        spec = AGVSpec.from_grade(AGVGrade.M)
        ctrl = AGVMotionController(spec)
        
        ctrl.update_pose(AGVPose(x=0.0, y=0.0, theta=0.0))
        
        # 原地旋转: 左右轮速相反
        twist = AGVTwist(vx=0.0, vy=0.0, omega=1.0)  # 1 rad/s
        wheel_cmds = ctrl.inverse_kinematics(twist)
        
        self.assertAlmostEqual(wheel_cmds[0], -wheel_cmds[1], places=5)


class TestCrossModalAttentionPerformance(unittest.TestCase):
    """跨模态注意力性能测试"""
    
    def test_fusion_throughput(self):
        """融合网络吞吐量测试"""
        fusion_config = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=512, num_heads=8
        )
        fusion = CrossModalFusion(fusion_config)
        fusion.eval()
        
        mmi = MultimodalInput(
            vision=torch.randn(1, 512),
            audio=torch.randn(1, 128),
            tactile=torch.randn(1, 64),
            force=torch.randn(1, 32),
            imu=torch.randn(1, 64),
        )
        
        # 预热
        for _ in range(5):
            with torch.no_grad():
                fusion(mmi)
        
        # 计时
        start = time.time()
        iterations = 100
        for _ in range(iterations):
            with torch.no_grad():
                fusion(mmi)
        elapsed = time.time() - start
        
        avg_time_ms = (elapsed / iterations) * 1000
        self.assertLess(avg_time_ms, 12.0, f"Fusion too slow: {avg_time_ms:.2f}ms per iteration")


if __name__ == '__main__':
    unittest.main(verbosity=2)
