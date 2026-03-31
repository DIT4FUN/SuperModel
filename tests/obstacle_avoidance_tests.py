"""
障碍物回避模块测试
=================

测试AGV避障算法:
- DynamicWindowApproach (DWA)
- ArtificialPotentialField (APF)
- VectorFieldHistogram (VFH)
- ObstacleAvoider 综合控制器
"""

import numpy as np
import unittest
import math

import sys
sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from control.obstacle_avoidance import (
    Obstacle, VelocityCommand, TrajectorySample,
    DWAConfig, APFConfig, VFHConfig, AvoidanceConfig, AvoidanceStrategy,
    DynamicWindowApproach, ArtificialPotentialField, VectorFieldHistogram,
    ObstacleAvoider, get_obstacle_avoidance_spec
)


class TestObstacle(unittest.TestCase):
    """测试障碍物类"""
    
    def test_obstacle_creation(self):
        obs = Obstacle(position=np.array([1.0, 2.0]), radius=0.5)
        self.assertEqual(obs.position[0], 1.0)
        self.assertEqual(obs.position[1], 2.0)
        self.assertEqual(obs.radius, 0.5)
        self.assertEqual(obs.type, "static")
    
    def test_dynamic_obstacle(self):
        obs = Obstacle(
            position=np.array([1.0, 2.0]),
            radius=0.5,
            velocity=np.array([0.5, 0.3]),
            type="dynamic"
        )
        self.assertEqual(obs.type, "dynamic")
        self.assertTrue(np.allclose(obs.velocity, [0.5, 0.3]))
    
    def test_predict_position(self):
        obs = Obstacle(
            position=np.array([1.0, 2.0]),
            radius=0.5,
            velocity=np.array([1.0, 0.0]),
            type="dynamic"
        )
        predicted = obs.predict_position(0.5)
        self.assertTrue(np.allclose(predicted, [1.5, 2.0]))


class TestVelocityCommand(unittest.TestCase):
    """测试速度指令类"""
    
    def test_velocity_command_creation(self):
        cmd = VelocityCommand(vx=0.5, vy=0.3, omega=1.0, score=0.8)
        self.assertEqual(cmd.vx, 0.5)
        self.assertEqual(cmd.vy, 0.3)
        self.assertEqual(cmd.omega, 1.0)
        self.assertEqual(cmd.score, 0.8)
    
    def test_to_array(self):
        cmd = VelocityCommand(vx=0.5, vy=0.3, omega=1.0)
        arr = cmd.to_array()
        self.assertTrue(np.allclose(arr, [0.5, 0.3, 1.0]))


class TestDWAConfig(unittest.TestCase):
    """测试DWA配置"""
    
    def test_default_config(self):
        cfg = DWAConfig()
        self.assertEqual(cfg.max_linear_speed, 1.0)
        self.assertEqual(cfg.max_angular_speed, 2.0)
        self.assertEqual(cfg.robot_radius, 0.3)
    
    def test_custom_config(self):
        cfg = DWAConfig(
            max_linear_speed=2.0,
            max_angular_speed=3.0,
            robot_radius=0.5
        )
        self.assertEqual(cfg.max_linear_speed, 2.0)
        self.assertEqual(cfg.robot_radius, 0.5)


class TestDynamicWindowApproach(unittest.TestCase):
    """测试动态窗口法"""
    
    def test_dwa_initialization(self):
        dwa = DynamicWindowApproach()
        self.assertIsInstance(dwa.config, DWAConfig)
    
    def test_dwa_custom_config(self):
        cfg = DWAConfig(max_linear_speed=2.0)
        dwa = DynamicWindowApproach(cfg)
        self.assertEqual(dwa.config.max_linear_speed, 2.0)
    
    def test_dwa_no_obstacles(self):
        """无障碍物时正常输出速度"""
        dwa = DynamicWindowApproach()
        pose = np.array([0.0, 0.0, 0.0])
        velocity = np.array([0.0, 0.0, 0.0])
        goal = np.array([2.0, 0.0])
        obstacles = []
        
        cmd = dwa.compute_velocities(pose, velocity, goal, obstacles, dt=0.1)
        
        self.assertIsInstance(cmd, VelocityCommand)
        self.assertGreaterEqual(cmd.vx, 0.0)
    
    def test_dwa_obstacle_ahead(self):
        """障碍物在前方时应减速或停止"""
        dwa = DynamicWindowApproach()
        pose = np.array([0.0, 0.0, 0.0])
        velocity = np.array([0.5, 0.0, 0.0])
        goal = np.array([5.0, 0.0])
        obstacles = [Obstacle(position=np.array([1.5, 0.0]), radius=0.5)]
        
        cmd = dwa.compute_velocities(pose, velocity, goal, obstacles, dt=0.1)
        
        # 速度应该有所降低或保持安全范围
        self.assertLessEqual(cmd.vx, 0.7)
    
    def test_dwa_lateral_obstacle(self):
        """侧向障碍物"""
        dwa = DynamicWindowApproach()
        pose = np.array([0.0, 0.0, 0.0])
        velocity = np.array([0.0, 0.0, 0.0])
        goal = np.array([2.0, 0.0])
        obstacles = [Obstacle(position=np.array([1.0, 1.0]), radius=0.5)]
        
        cmd = dwa.compute_velocities(pose, velocity, goal, obstacles, dt=0.1)
        self.assertIsInstance(cmd, VelocityCommand)
    
    def test_dwa_close_to_obstacle(self):
        """非常接近障碍物"""
        dwa = DynamicWindowApproach()
        pose = np.array([0.0, 0.0, 0.0])
        velocity = np.array([0.1, 0.0, 0.0])
        goal = np.array([2.0, 0.0])
        obstacles = [Obstacle(position=np.array([0.3, 0.0]), radius=0.3)]
        
        cmd = dwa.compute_velocities(pose, velocity, goal, obstacles, dt=0.1)
        # 非常近时应显著减速
        self.assertLessEqual(cmd.vx, 0.35)
    
    def test_dwa_multiple_obstacles(self):
        """多个障碍物"""
        dwa = DynamicWindowApproach()
        pose = np.array([0.0, 0.0, 0.0])
        velocity = np.array([0.0, 0.0, 0.0])
        goal = np.array([3.0, 0.0])
        obstacles = [
            Obstacle(position=np.array([1.0, 0.0]), radius=0.3),
            Obstacle(position=np.array([2.0, 0.5]), radius=0.3),
            Obstacle(position=np.array([2.5, -0.5]), radius=0.3),
        ]
        
        cmd = dwa.compute_velocities(pose, velocity, goal, obstacles, dt=0.1)
        self.assertIsInstance(cmd, VelocityCommand)
    
    def test_dwa_dynamic_obstacle(self):
        """移动障碍物"""
        dwa = DynamicWindowApproach()
        pose = np.array([0.0, 0.0, 0.0])
        velocity = np.array([0.5, 0.0, 0.0])
        goal = np.array([5.0, 0.0])
        obstacles = [
            Obstacle(
                position=np.array([2.0, 0.0]),
                radius=0.3,
                velocity=np.array([-0.5, 0.0]),
                type="dynamic"
            )
        ]
        
        cmd = dwa.compute_velocities(pose, velocity, goal, obstacles, dt=0.1)
        self.assertIsInstance(cmd, VelocityCommand)
    
    def test_dwa_at_goal(self):
        """已到达目标"""
        dwa = DynamicWindowApproach()
        pose = np.array([2.0, 0.0, 0.0])
        velocity = np.array([0.0, 0.0, 0.0])
        goal = np.array([2.0, 0.0])
        obstacles = []
        
        cmd = dwa.compute_velocities(pose, velocity, goal, obstacles, dt=0.1)
        self.assertEqual(cmd.vx, 0.0)


class TestAPFConfig(unittest.TestCase):
    """测试APF配置"""
    
    def test_default_config(self):
        cfg = APFConfig()
        self.assertEqual(cfg.attract_gain, 5.0)
        self.assertEqual(cfg.repel_gain, 100.0)
        self.assertEqual(cfg.robot_radius, 0.3)


class TestArtificialPotentialField(unittest.TestCase):
    """测试人工势场法"""
    
    def test_apf_initialization(self):
        apf = ArtificialPotentialField()
        self.assertIsInstance(apf.config, APFConfig)
    
    def test_apf_attractive_force(self):
        """测试吸引力"""
        apf = ArtificialPotentialField(APFConfig(attract_gain=1.0))
        robot_pose = np.array([0.0, 0.0])
        robot_velocity = np.array([0.0, 0.0])
        goal = np.array([5.0, 0.0])
        obstacles = []
        
        force = apf.compute_force(robot_pose, robot_velocity, goal, obstacles)
        
        # 方向应指向目标(正x方向)
        self.assertGreater(force[0], 0.0)
    
    def test_apf_repulsive_force(self):
        """测试排斥力"""
        apf = ArtificialPotentialField(APFConfig(repel_gain=100.0, repel_range=2.0))
        robot_pose = np.array([1.0, 0.0])
        robot_velocity = np.array([0.0, 0.0])
        goal = np.array([5.0, 0.0])
        obstacles = [Obstacle(position=np.array([1.5, 0.0]), radius=0.3)]
        
        force = apf.compute_force(robot_pose, robot_velocity, goal, obstacles)
        
        # 方向应远离障碍物
        self.assertLess(force[0], 0.0)
    
    def test_apf_no_obstacles(self):
        """无障碍物时只有吸引力"""
        apf = ArtificialPotentialField()
        robot_pose = np.array([0.0, 0.0])
        robot_velocity = np.array([0.0, 0.0])
        goal = np.array([3.0, 3.0])
        obstacles = []
        
        force = apf.compute_force(robot_pose, robot_velocity, goal, obstacles)
        
        # 合力应为吸引力
        self.assertGreater(force[0], 0.0)
        self.assertGreater(force[1], 0.0)
    
    def test_apf_at_goal(self):
        """到达目标时无吸引力"""
        apf = ArtificialPotentialField(APFConfig(goal_tolerance=0.1, escape_threshold=0.5))
        robot_pose = np.array([2.05, 2.05])  # 非常接近目标
        robot_velocity = np.array([0.6, 0.0])  # 快速移动避免escape force
        goal = np.array([2.0, 2.0])
        obstacles = []
        
        force = apf.compute_force(robot_pose, robot_velocity, goal, obstacles)
        
        # 吸引力在容差范围内应接近零
        self.assertLess(np.linalg.norm(force), 1.0)
    
    def test_apf_obstacle_far(self):
        """障碍物在排斥场范围外"""
        apf = ArtificialPotentialField(APFConfig(repel_range=2.0))
        robot_pose = np.array([0.0, 0.0])
        robot_velocity = np.array([0.0, 0.0])
        goal = np.array([5.0, 0.0])
        obstacles = [Obstacle(position=np.array([10.0, 0.0]), radius=0.5)]
        
        force = apf.compute_force(robot_pose, robot_velocity, goal, obstacles)
        
        # 合力应接近纯吸引力
        self.assertGreater(force[0], 0.0)
    
    def test_apf_compute_velocity(self):
        """测试速度计算"""
        apf = ArtificialPotentialField()
        robot_pose = np.array([0.0, 0.0])
        robot_velocity = np.array([0.0, 0.0])
        goal = np.array([2.0, 0.0])
        obstacles = []
        
        vel = apf.compute_velocity(robot_pose, robot_velocity, goal, obstacles)
        
        self.assertEqual(vel.shape, (2,))
        self.assertGreater(vel[0], 0.0)
    
    def test_apf_velocity_limit(self):
        """测试速度限制"""
        apf = ArtificialPotentialField(APFConfig(attract_gain=10.0))
        robot_pose = np.array([0.0, 0.0])
        robot_velocity = np.array([0.0, 0.0])
        goal = np.array([100.0, 0.0])
        obstacles = []
        
        vel = apf.compute_velocity(robot_pose, robot_velocity, goal, obstacles, max_speed=0.5)
        
        self.assertLessEqual(np.linalg.norm(vel), 0.5 + 1e-5)


class TestVFHConfig(unittest.TestCase):
    """测试VFH配置"""
    
    def test_default_config(self):
        cfg = VFHConfig()
        self.assertEqual(cfg.sector_angle, 5.0)
        self.assertEqual(cfg.detection_radius, 3.0)


class TestVectorFieldHistogram(unittest.TestCase):
    """测试向量场直方图"""
    
    def test_vfh_initialization(self):
        vfh = VectorFieldHistogram()
        self.assertIsInstance(vfh.config, VFHConfig)
        self.assertEqual(vfh.num_sectors, 72)  # 360 / 5
    
    def test_vfh_histogram_building(self):
        """测试直方图构建"""
        vfh = VectorFieldHistogram()
        pose = np.array([0.0, 0.0, 0.0])
        obstacles = [Obstacle(position=np.array([1.0, 0.0]), radius=0.3)]
        
        histogram = vfh._build_histogram(pose, obstacles)
        
        self.assertEqual(len(histogram), vfh.num_sectors)
        self.assertGreater(np.max(histogram), 0.0)
    
    def test_vfh_no_obstacles(self):
        """无障碍物"""
        vfh = VectorFieldHistogram()
        pose = np.array([0.0, 0.0, 0.0])
        obstacles = []
        
        histogram = vfh._build_histogram(pose, obstacles)
        
        self.assertTrue(np.allclose(histogram, 0.0))
    
    def test_vfh_threshold(self):
        """测试阈值化"""
        vfh = VectorFieldHistogram()
        histogram = np.array([0.0, 10.0, 100.0, 0.0, 60.0])
        
        binary = vfh._threshold_histogram(histogram)
        
        self.assertEqual(binary[0], 0.0)
        self.assertEqual(binary[1], 0.0)
        self.assertEqual(binary[2], 1.0)
    
    def test_vfh_direction_selection(self):
        """测试方向选择"""
        vfh = VectorFieldHistogram()
        histogram = np.zeros(vfh.num_sectors)
        histogram[0:5] = 1.0  # 阻塞前方
        histogram[60:72] = 1.0  # 阻塞后方
        
        steering = vfh._select_direction(histogram, 0.0)
        
        # 应选择非阻塞方向
        self.assertIsInstance(steering, float)
    
    def test_vfh_compute_direction(self):
        """测试完整方向计算"""
        vfh = VectorFieldHistogram()
        pose = np.array([0.0, 0.0, 0.0])
        velocity = np.array([0.0, 0.0, 0.0])
        goal = np.array([2.0, 0.0])
        obstacles = []
        
        angle, cmd = vfh.compute_direction(pose, velocity, goal, obstacles)
        
        self.assertIsInstance(angle, float)
        self.assertIsInstance(cmd, VelocityCommand)
    
    def test_vfh_normalize_angle(self):
        """测试角度归一化"""
        self.assertAlmostEqual(VectorFieldHistogram._normalize_angle(math.pi), math.pi)
        self.assertAlmostEqual(VectorFieldHistogram._normalize_angle(3*math.pi), math.pi)
        self.assertAlmostEqual(VectorFieldHistogram._normalize_angle(-3*math.pi), -math.pi)


class TestObstacleAvoider(unittest.TestCase):
    """测试障碍物回避主控制器"""
    
    def test_avoider_initialization(self):
        avoider = ObstacleAvoider()
        self.assertIsInstance(avoider.config, AvoidanceConfig)
    
    def test_avoider_dwa_strategy(self):
        """测试DWA策略"""
        cfg = AvoidanceConfig(strategy=AvoidanceStrategy.DWA)
        avoider = ObstacleAvoider(cfg)
        
        pose = np.array([0.0, 0.0, 0.0])
        velocity = np.array([0.0, 0.0, 0.0])
        goal = np.array([2.0, 0.0])
        obstacles = []
        
        cmd = avoider.compute_command(pose, velocity, goal, obstacles)
        
        self.assertIsInstance(cmd, VelocityCommand)
    
    def test_avoider_apf_strategy(self):
        """测试APF策略"""
        cfg = AvoidanceConfig(strategy=AvoidanceStrategy.APF)
        avoider = ObstacleAvoider(cfg)
        
        pose = np.array([0.0, 0.0, 0.0])
        velocity = np.array([0.0, 0.0, 0.0])
        goal = np.array([2.0, 0.0])
        obstacles = []
        
        cmd = avoider.compute_command(pose, velocity, goal, obstacles)
        
        self.assertIsInstance(cmd, VelocityCommand)
    
    def test_avoider_vfh_strategy(self):
        """测试VFH策略"""
        cfg = AvoidanceConfig(strategy=AvoidanceStrategy.VFH)
        avoider = ObstacleAvoider(cfg)
        
        pose = np.array([0.0, 0.0, 0.0])
        velocity = np.array([0.0, 0.0, 0.0])
        goal = np.array([2.0, 0.0])
        obstacles = []
        
        cmd = avoider.compute_command(pose, velocity, goal, obstacles)
        
        self.assertIsInstance(cmd, VelocityCommand)
    
    def test_avoider_hybrid_strategy(self):
        """测试混合策略"""
        cfg = AvoidanceConfig(strategy=AvoidanceStrategy.HYBRID)
        avoider = ObstacleAvoider(cfg)
        
        pose = np.array([0.0, 0.0, 0.0])
        velocity = np.array([0.0, 0.0, 0.0])
        goal = np.array([5.0, 0.0])
        obstacles = []
        
        cmd = avoider.compute_command(pose, velocity, goal, obstacles)
        
        self.assertIsInstance(cmd, VelocityCommand)
    
    def test_avoider_switch_strategy(self):
        """测试策略切换"""
        avoider = ObstacleAvoider(AvoidanceConfig(strategy=AvoidanceStrategy.DWA))
        avoider.set_strategy(AvoidanceStrategy.APF)
        self.assertEqual(avoider.current_strategy, AvoidanceStrategy.APF)
    
    def test_avoider_obstacle_blocked(self):
        """障碍物完全阻挡"""
        cfg = AvoidanceConfig(strategy=AvoidanceStrategy.DWA)
        avoider = ObstacleAvoider(cfg)
        
        pose = np.array([0.0, 0.0, 0.0])
        velocity = np.array([0.0, 0.0, 0.0])
        goal = np.array([5.0, 0.0])
        obstacles = [
            Obstacle(position=np.array([1.0, -0.5]), radius=1.0),
            Obstacle(position=np.array([1.0, 0.5]), radius=1.0),
        ]
        
        cmd = avoider.compute_command(pose, velocity, goal, obstacles)
        
        # 应选择绕行方向
        self.assertIsInstance(cmd, VelocityCommand)
    
    def test_avoider_create_from_grade(self):
        """测试从AGV等级创建"""
        for grade in ["S", "M", "L", "XL", "XXL"]:
            avoider = ObstacleAvoider.create_from_grade(grade)
            self.assertIsInstance(avoider, ObstacleAvoider)


class TestGetObstacleAvoidanceSpec(unittest.TestCase):
    """测试AGV等级避障规格"""
    
    def test_spec_all_grades(self):
        for grade in ["S", "M", "L", "XL", "XXL"]:
            spec = get_obstacle_avoidance_spec(grade)
            self.assertIn("strategy", spec)
            self.assertIn("max_obstacles", spec)
            self.assertIn("reaction_time", spec)
            self.assertIn("clearance", spec)
    
    def test_spec_grade_s(self):
        spec = get_obstacle_avoidance_spec("S")
        self.assertEqual(spec["strategy"], "none")
        self.assertEqual(spec["max_obstacles"], 0)
    
    def test_spec_grade_m(self):
        spec = get_obstacle_avoidance_spec("M")
        self.assertEqual(spec["strategy"], "DWA")
    
    def test_spec_grade_l(self):
        spec = get_obstacle_avoidance_spec("L")
        self.assertEqual(spec["strategy"], "HYBRID")
    
    def test_spec_default(self):
        spec = get_obstacle_avoidance_spec("UNKNOWN")
        self.assertEqual(spec["strategy"], "DWA")


class TestObstacleAvoiderIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_navigation_scenario(self):
        """完整导航场景"""
        avoider = ObstacleAvoider(AvoidanceConfig(strategy=AvoidanceStrategy.HYBRID))
        
        pose = np.array([0.0, 0.0, 0.0])
        velocity = np.array([0.0, 0.0, 0.0])
        goal = np.array([5.0, 0.0])
        obstacles = [
            Obstacle(position=np.array([2.0, 0.0]), radius=0.5),
            Obstacle(position=np.array([3.5, 0.8]), radius=0.4),
        ]
        
        # 模拟多步导航
        for step in range(20):
            cmd = avoider.compute_command(pose, velocity, goal, obstacles)
            
            # 更新位姿
            dt = 0.1
            pose[0] += cmd.vx * dt
            pose[1] += cmd.vy * dt
            pose[2] += cmd.omega * dt
            
            velocity = np.array([cmd.vx, cmd.vy, cmd.omega])
            
            # 检查是否到达
            if np.linalg.norm(goal - pose[:2]) < 0.2:
                break
        
        # 验证
        self.assertIsInstance(cmd, VelocityCommand)
    
    def test_narrow_corridor(self):
        """窄走廊场景"""
        avoider = ObstacleAvoider(AvoidanceConfig(strategy=AvoidanceStrategy.DWA))
        
        pose = np.array([0.0, 0.0, 0.0])
        velocity = np.array([0.3, 0.0, 0.0])
        goal = np.array([5.0, 0.0])
        # 走廊宽度约1.2m，机器人直径0.6m
        obstacles = [
            Obstacle(position=np.array([i*0.8, -0.7]), radius=0.3) for i in range(1, 7)
        ] + [
            Obstacle(position=np.array([i*0.8, 0.7]), radius=0.3) for i in range(1, 7)
        ]
        
        cmd = avoider.compute_command(pose, velocity, goal, obstacles)
        
        self.assertIsInstance(cmd, VelocityCommand)
    
    def test_dynamic_obstacle_avoidance(self):
        """动态障碍物避障"""
        avoider = ObstacleAvoider(AvoidanceConfig(strategy=AvoidanceStrategy.HYBRID))
        
        pose = np.array([0.0, 0.0, 0.0])
        velocity = np.array([0.5, 0.0, 0.0])
        goal = np.array([5.0, 0.0])
        obstacles = [
            Obstacle(
                position=np.array([2.0, 0.0]),
                radius=0.5,
                velocity=np.array([0.0, 0.5]),
                type="dynamic"
            )
        ]
        
        cmd = avoider.compute_command(pose, velocity, goal, obstacles)
        
        self.assertIsInstance(cmd, VelocityCommand)
    
    def test_all_strategies_compare(self):
        """对比所有策略"""
        pose = np.array([0.0, 0.0, 0.0])
        velocity = np.array([0.0, 0.0, 0.0])
        goal = np.array([3.0, 0.0])
        obstacles = [Obstacle(position=np.array([1.5, 0.5]), radius=0.4)]
        
        results = {}
        for strategy in AvoidanceStrategy:
            avoider = ObstacleAvoider(AvoidanceConfig(strategy=strategy))
            cmd = avoider.compute_command(pose, velocity, goal, obstacles)
            results[strategy.value] = cmd
        
        self.assertEqual(len(results), 4)


if __name__ == "__main__":
    unittest.main()
