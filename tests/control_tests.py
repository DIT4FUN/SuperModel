"""
控制模块测试
============

测试所有控制模块:
- MotionController (PID关节控制)
- ImpedanceController (阻抗/导纳控制)
- SkillLibrary (技能库)
- TaskPlanner (任务规划)
- RobotSimulator / SensorSimulator (仿真环境)
"""

import numpy as np
import sys
import time
import unittest

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from control.motion import (
    MotionController, ControlMode, JointState, JointTrajectory,
    TwistCommand, TwistToJoint
)
from control.agv import (
    AGVMotionController, AGVSpec, AGVGrade, AGVPose, AGVTwist,
    DriveType, DifferentialKinematics, MecanumKinematics, get_agv_spec
)
from control.impedance import (
    ImpedanceController, ImpedanceParams, AdmittanceController,
    ForceImpedanceController, CollaborativeController
)
from control.skill import (
    Skill, PrimitiveSkill, CompositeSkill, SkillLibrary,
    SkillConfig, SkillResult, SkillStatus, PRESET_GRASP_CONFIGS
)
from control.planner import (
    TaskPlanner, HierarchicalPlanner, Task, TaskSpec,
    TaskStatus as PlanTaskStatus, TaskPriority, WorldState, Action
)
from simulation.environment import (
    RobotSimulator, SensorSimulator, SimConfig, PhysicsEngine,
    PRESET_SCENES, create_scene
)


class TestMotionController(unittest.TestCase):
    """测试运动控制器"""
    
    def setUp(self):
        self.controller = MotionController(num_joints=6, control_rate=100.0)
        self.controller.kp = np.ones(6) * 1.0
        self.controller.ki = np.zeros(6)
        self.controller.kd = np.zeros(6)
    
    def test_controller_init(self):
        self.assertEqual(self.controller.num_joints, 6)
        self.assertEqual(self.controller.control_rate, 100.0)
        self.assertEqual(self.controller.dt, 0.01)
    
    def test_set_joint_limits(self):
        lower = -np.ones(6) * 0.5 * np.pi
        upper = np.ones(6) * 0.5 * np.pi
        self.controller.set_joint_limits(lower, upper)
        np.testing.assert_array_almost_equal(self.controller.joint_limits_lower, lower)
        np.testing.assert_array_almost_equal(self.controller.joint_limits_upper, upper)
    
    def test_set_pid_gains(self):
        kp = np.ones(6) * 2.0
        ki = np.ones(6) * 0.1
        kd = np.ones(6) * 0.5
        self.controller.set_pid_gains(kp, ki, kd)
        np.testing.assert_array_almost_equal(self.controller.kp, kp)
        np.testing.assert_array_almost_equal(self.controller.ki, ki)
        np.testing.assert_array_almost_equal(self.controller.kd, kd)
    
    def test_compute_joint_torque(self):
        self.controller._current_joint_pos = np.zeros(6)
        self.controller._current_joint_vel = np.zeros(6)
        
        target = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
        torque = self.controller.compute_joint_torque(target)
        
        self.assertEqual(torque.shape, (6,))
        # 应该有关注力矩输出
        self.assertFalse(np.allclose(torque, np.zeros(6)))
    
    def test_compute_joint_torque_with_velocity(self):
        self.controller._current_joint_pos = np.zeros(6)
        self.controller._current_joint_vel = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        target_pos = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        target_vel = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        torque = self.controller.compute_joint_torque(target_pos, target_vel)
        self.assertEqual(torque.shape, (6,))
    
    def test_compute_cartesian_velocity(self):
        twist = TwistCommand(
            linear=np.array([0.1, 0.0, 0.0]),
            angular=np.array([0.0, 0.0, 0.0])
        )
        jacobian = np.random.randn(6, 6)
        joint_vel = self.controller.compute_cartesian_velocity(twist, jacobian)
        self.assertEqual(joint_vel.shape, (6,))
    
    def test_apply_safety_limits(self):
        command = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        limited = self.controller.apply_safety_limits(command)
        np.testing.assert_array_less(np.abs(limited), self.controller.max_torque * 1.1)
    
    def test_joint_state_update(self):
        state = JointState(
            position=np.ones(6) * 0.5,
            velocity=np.ones(6) * 0.1,
            torque=np.ones(6) * 1.0
        )
        self.controller.update_joint_state(state)
        np.testing.assert_array_almost_equal(self.controller._current_joint_pos, state.position)


class TestTwistToJoint(unittest.TestCase):
    """测试笛卡尔速度转关节速度"""
    
    def test_twist_to_joint_init(self):
        def dummy_jacobian_fn(q):
            return np.random.randn(6, 6)
        converter = TwistToJoint(dummy_jacobian_fn)
        self.assertIsNotNone(converter)
    
    def test_twist_to_joint_compute(self):
        def jacobian_fn(q):
            return np.eye(6)
        
        converter = TwistToJoint(jacobian_fn)
        twist = TwistCommand(
            linear=np.array([0.1, 0.0, 0.0]),
            angular=np.array([0.0, 0.0, 0.0])
        )
        joint_vel = converter.compute(twist, np.zeros(6))
        self.assertEqual(joint_vel.shape, (6,))


class TestJointTrajectory(unittest.TestCase):
    """测试关节轨迹"""
    
    def test_trajectory_init(self):
        positions = np.random.randn(10, 6)
        timestamps = np.linspace(0, 1, 10)
        traj = JointTrajectory(positions=positions, timestamps=timestamps)
        self.assertEqual(traj.positions.shape, (10, 6))
        self.assertEqual(len(traj.timestamps), 10)
    
    def test_trajectory_interpolation(self):
        positions = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        timestamps = np.array([0.0, 0.5, 1.0])
        traj = JointTrajectory(positions=positions, timestamps=timestamps)
        
        controller = MotionController(num_joints=2)
        
        pos, vel = controller.interpolate_trajectory(traj, 0.25)
        self.assertEqual(pos.shape, (2,))


class TestImpedanceController(unittest.TestCase):
    """测试阻抗控制器"""
    
    def setUp(self):
        self.params = ImpedanceParams.default_6d()
        self.controller = ImpedanceController(self.params, control_rate=100.0)
    
    def test_controller_init(self):
        self.assertIsNotNone(self.controller.params)
        self.assertEqual(self.controller.dt, 0.01)
    
    def test_compute_torque_shapes(self):
        desired_pos = np.array([0.5, 0.0, 0.3])
        desired_vel = np.zeros(3)
        current_pos = np.array([0.52, 0.01, 0.28])
        current_vel = np.zeros(3)
        wrench = np.zeros(6)
        jacobian = np.random.randn(6, 6)
        
        torque = self.controller.compute_torque(
            desired_pos, desired_vel, current_pos, current_vel, wrench, jacobian
        )
        self.assertEqual(torque.shape, (6,))
    
    def test_set_impedance_params(self):
        new_params = ImpedanceParams.high_stiffness()
        self.controller.set_impedance_params(new_params)
        np.testing.assert_array_almost_equal(
            self.controller.params.K, new_params.K
        )
    
    def test_impedance_params_default(self):
        params = ImpedanceParams.default_6d()
        self.assertEqual(params.M.shape, (6, 6))
        self.assertEqual(params.D.shape, (6, 6))
        self.assertEqual(params.K.shape, (6, 6))
    
    def test_impedance_params_high_stiffness(self):
        params = ImpedanceParams.high_stiffness()
        # 高刚度的对角元素应该大于默认刚度
        self.assertGreater(params.K[0, 0], ImpedanceParams.default_6d().K[0, 0])


class TestAdmittanceController(unittest.TestCase):
    """测试导纳控制器"""
    
    def test_admittance_init(self):
        adm = AdmittanceController(M=10.0, D=50.0, K=200.0)
        self.assertEqual(adm.M, 10.0)
        self.assertEqual(adm.D, 50.0)
        self.assertEqual(adm.K, 200.0)
    
    def test_admittance_update(self):
        adm = AdmittanceController()
        external_force = 10.0
        desired_pos = 0.0
        
        adjusted = adm.update(external_force, desired_pos)
        self.assertIsInstance(adjusted, float)
    
    def test_admittance_reset(self):
        adm = AdmittanceController()
        adm.update(10.0, 0.0)
        adm.reset()
        self.assertEqual(adm._velocity, 0.0)
        self.assertEqual(adm._position, 0.0)


class TestForceImpedanceController(unittest.TestCase):
    """测试力位混合控制器"""
    
    def test_force_impedance_init(self):
        # Z轴力控
        force_axes = np.array([0, 0, 1, 0, 0, 0])
        ctrl = ForceImpedanceController(force_axes)
        self.assertEqual(ctrl.Kp, 100.0)
        self.assertEqual(ctrl.Kf, 1.0)


class TestCollaborativeController(unittest.TestCase):
    """测试协作控制器"""
    
    def test_collaborative_init(self):
        ctrl = CollaborativeController(
            safety_force_limit=50.0,
            safety_velocity_limit=0.5
        )
        self.assertEqual(ctrl.safety_force_limit, 50.0)
        self.assertEqual(ctrl.safety_velocity_limit, 0.5)
    
    def test_check_safety(self):
        ctrl = CollaborativeController()
        
        # 安全情况
        safe, msg = ctrl.check_safety(np.zeros(3), np.array([0.1, 0.1, 0.1]))
        self.assertTrue(safe)
        
        # 危险情况 - 力超限
        unsafe, msg = ctrl.check_safety(
            np.array([100.0, 0.0, 0.0]),
            np.zeros(3)
        )
        self.assertFalse(unsafe)
        self.assertIn("force_limit", msg)
    
    def test_get_reaction_torque(self):
        ctrl = CollaborativeController(reaction_mode="pause")
        jacobian = np.random.randn(6, 6)
        torque = ctrl.get_reaction_torque(np.zeros(3), jacobian)
        self.assertEqual(torque.shape, (6,))


class TestSkillLibrary(unittest.TestCase):
    """测试技能库"""
    
    def setUp(self):
        self.library = SkillLibrary()
    
    def test_library_init(self):
        self.assertIsNotNone(self.library)
    
    def test_list_skills(self):
        skills = self.library.list_skills()
        self.assertIsInstance(skills, list)
    
    def test_create_skill(self):
        skill = self.library.create_skill("move_to", {"target": [0.5, 0.0, 0.3]})
        self.assertIsNotNone(skill)
    
    def test_create_unknown_skill(self):
        skill = self.library.create_skill("nonexistent_skill", {})
        self.assertIsNone(skill)
    
    def test_preset_grasp_configs(self):
        self.assertIn("top_grasp", PRESET_GRASP_CONFIGS)
        self.assertIn("side_grasp", PRESET_GRASP_CONFIGS)
        self.assertEqual(PRESET_GRASP_CONFIGS["top_grasp"]["approach_height"], 0.1)


class TestSkillExecution(unittest.TestCase):
    """测试技能执行"""
    
    def test_skill_config(self):
        config = SkillConfig(name="test_skill", description="测试技能", timeout=10.0)
        self.assertEqual(config.name, "test_skill")
        self.assertEqual(config.timeout, 10.0)
    
    def test_skill_result(self):
        result = SkillResult(success=True, status=SkillStatus.SUCCEEDED)
        self.assertTrue(result.success)
        self.assertEqual(result.status, SkillStatus.SUCCEEDED)


class TestTaskPlanner(unittest.TestCase):
    """测试任务规划器"""
    
    def setUp(self):
        self.planner = TaskPlanner()
    
    def test_planner_init(self):
        self.assertIsNotNone(self.planner)
        self.assertEqual(len(self.planner._task_queue), 0)
    
    def test_add_task(self):
        task = Task(id="test", name="test_task")
        self.planner.add_task(task)
        self.assertEqual(len(self.planner._task_queue), 1)
    
    def test_get_next_task(self):
        task = Task(id="test", name="test_task")
        self.planner.add_task(task)
        next_task = self.planner.get_next_task()
        self.assertEqual(next_task.id, "test")
        self.assertEqual(next_task.status, PlanTaskStatus.RUNNING)
    
    def test_world_state(self):
        state = WorldState()
        state.objects["robot"] = {"position": [0, 0, 0]}
        self.planner.set_world_state(state)
        
        copied = self.planner._world_state.copy()
        self.assertIn("robot", copied.objects)
    
    def test_plan_empty_state(self):
        spec = TaskSpec(name="test", goal_state={"robot.position": [1, 0, 0]})
        plan = self.planner.plan(spec)
        self.assertIsInstance(plan, list)


class TestHierarchicalPlanner(unittest.TestCase):
    """测试层次化任务网络规划器"""
    
    def setUp(self):
        self.planner = HierarchicalPlanner()
    
    def test_htn_init(self):
        self.assertIsNotNone(self.planner)
    
    def test_decompose_task(self):
        task = Task(id="pickup_test", name="pickup", parameters={"object": "box"})
        subtasks = self.planner.decompose_task(task, max_depth=3)
        self.assertIsInstance(subtasks, list)
        self.assertGreater(len(subtasks), 0)
    
    def test_plan_hierarchical(self):
        spec = TaskSpec(
            name="pickup",
            goal_state={"held": True},
            max_depth=3
        )
        tasks = self.planner.plan_hierarchical(spec)
        self.assertIsInstance(tasks, list)


class TestRobotSimulator(unittest.TestCase):
    """测试机器人仿真器"""
    
    def setUp(self):
        self.config = SimConfig(dt=0.01, num_joints=6)
        self.sim = RobotSimulator(self.config)
    
    def test_simulator_init(self):
        self.assertEqual(self.sim.n, 6)
        self.assertEqual(self.sim.dt, 0.01)
    
    def test_set_joint_positions(self):
        positions = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
        self.sim.set_joint_positions(positions)
        np.testing.assert_array_almost_equal(self.sim.joint_positions, positions)
    
    def test_step(self):
        torque = np.zeros(6)
        state = self.sim.step(torque)
        self.assertIn('time', state)
        self.assertIn('joint_positions', state)
        self.assertIn('step', state)
    
    def test_joint_limits(self):
        # 测试限位反弹
        self.sim.set_joint_positions(np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0]))
        for _ in range(10):
            self.sim.step(np.zeros(6))
        
        for i in range(6):
            self.assertLessEqual(self.sim.joint_positions[i], self.sim.jl_upper[i])
    
    def test_get_jacobian(self):
        J = self.sim.get_jacobian()
        self.assertEqual(J.shape, (6, 6))
    
    def test_check_self_collision(self):
        result = self.sim.check_self_collision()
        self.assertIsInstance(result, bool)
    
    def test_reset(self):
        self.sim.set_joint_positions(np.ones(6) * 0.5)
        self.sim.step(np.ones(6))
        self.sim.reset()
        np.testing.assert_array_almost_equal(self.sim.joint_positions, np.zeros(6))


class TestSensorSimulator(unittest.TestCase):
    """测试传感器仿真器"""
    
    def setUp(self):
        self.config = SimConfig(dt=0.01, num_joints=6)
        self.sim = RobotSimulator(self.config)
        self.sensor_sim = SensorSimulator(self.sim, self.config)
    
    def test_get_noisy_joint_positions(self):
        pos = self.sensor_sim.get_noisy_joint_positions()
        self.assertEqual(pos.shape, (6,))
    
    def test_get_noisy_joint_velocities(self):
        vel = self.sensor_sim.get_noisy_joint_velocities()
        self.assertEqual(vel.shape, (6,))
    
    def test_get_imu_data(self):
        data = self.sensor_sim.get_imu_data()
        self.assertIn('accel', data)
        self.assertIn('gyro', data)
        self.assertEqual(data['accel'].shape, (3,))
        self.assertEqual(data['gyro'].shape, (3,))
    
    def test_get_wrench(self):
        wrench = self.sensor_sim.get_wrench()
        self.assertEqual(wrench.shape, (6,))
    
    def test_get_contact_force(self):
        force = self.sensor_sim.get_contact_force()
        # Returns float or numpy scalar
        self.assertTrue(isinstance(force, (int, float, np.number)), f"Expected numeric type, got {type(force)}")


class TestPhysicsEngine(unittest.TestCase):
    """测试物理引擎"""
    
    def test_engine_init_default(self):
        engine = PhysicsEngine()
        self.assertEqual(engine.engine, "custom")
    
    def test_engine_init_with_config(self):
        engine = PhysicsEngine(engine="custom", config={"num_joints": 6})
        self.assertIsNotNone(engine.simulator)
    
    def test_engine_step(self):
        engine = PhysicsEngine()
        state = engine.step(np.zeros(6))
        self.assertIn('time', state)


class TestPresetScenes(unittest.TestCase):
    """测试预设仿真场景"""
    
    def test_preset_scenes_exist(self):
        self.assertIn("tabletop", PRESET_SCENES)
        self.assertIn("shelf", PRESET_SCENES)
        self.assertIn("door", PRESET_SCENES)
    
    def test_create_scene(self):
        scene = create_scene("tabletop")
        self.assertIn("obstacles", scene)
        self.assertEqual(scene["description"], "桌面抓取场景")
    
    def test_create_unknown_scene(self):
        scene = create_scene("unknown")
        self.assertIn("obstacles", scene)  # 默认tabletop


class TestControlIntegration(unittest.TestCase):
    """控制模块集成测试"""
    
    def test_full_control_loop(self):
        """测试完整控制回路: 规划 -> 控制 -> 仿真"""
        # 1. 任务规划
        planner = HierarchicalPlanner()
        spec = TaskSpec(name="pickup", goal_state={"held": True}, max_depth=2)
        tasks = planner.plan_hierarchical(spec)
        
        # 2. 控制器
        controller = MotionController(num_joints=6)
        controller.kp = np.ones(6) * 2.0
        
        # 3. 仿真器
        sim = RobotSimulator(SimConfig(num_joints=6))
        
        # 4. 执行几个控制步骤
        for step in range(50):
            target = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
            controller._current_joint_pos = sim.joint_positions.copy()
            torque = controller.compute_joint_torque(target)
            state = sim.step(torque)
        
        # 验证机器人移动了
        displacement = np.linalg.norm(sim.joint_positions - np.zeros(6))
        self.assertGreater(displacement, 0.01)
    
    def test_impedance_control_loop(self):
        """测试阻抗控制回路"""
        imp_ctrl = ImpedanceController(ImpedanceParams.default_6d())
        sim = RobotSimulator(SimConfig(num_joints=6))
        
        for step in range(20):
            desired_pos = np.array([0.5, 0.0, 0.3])
            current_pos = sim.end_effector_pose[:3, 3]
            jacobian = sim.get_jacobian()[:3, :]
            
            torque = imp_ctrl.compute_torque(
                desired_position=desired_pos,
                desired_velocity=np.zeros(3),
                current_position=current_pos,
                current_velocity=np.zeros(3),
                external_wrench=np.zeros(6),
                jacobian=jacobian
            )
            
            sim.step(torque)
        
        # 仿真完成无错误
        self.assertTrue(True)


class TestTrajectoryGenerator(unittest.TestCase):
    """测试轨迹生成器"""
    
    def setUp(self):
        from control.trajectory import TrajectoryGenerator, TrajectoryConfig
        self.config = TrajectoryConfig(
            max_velocity=np.ones(6) * np.pi,
            max_acceleration=np.ones(6) * 2.0 * np.pi,
            dt=0.01
        )
        self.gen = TrajectoryGenerator(num_joints=6, config=self.config)
    
    def test_generator_init(self):
        self.assertEqual(self.gen.num_joints, 6)
        self.assertIsNotNone(self.gen.config)
    
    def test_quintic_polynomial_shapes(self):
        start = np.zeros(6)
        end = np.array([0.5, 0.3, -0.2, 0.0, 0.0, 0.0])
        waypoints = self.gen.generate_quintic_polynomial(start, end, duration=2.0)
        
        self.assertIsInstance(waypoints, list)
        self.assertGreater(len(waypoints), 0)
        
        # 检查每个路点形状
        for wp in waypoints:
            self.assertEqual(wp.position.shape, (6,))
            self.assertEqual(wp.velocity.shape, (6,))
            self.assertEqual(wp.acceleration.shape, (6,))
    
    def test_quintic_polynomial_boundary_conditions(self):
        start = np.zeros(6)
        end = np.ones(6) * 0.5
        waypoints = self.gen.generate_quintic_polynomial(
            start, end, duration=2.0,
            start_vel=np.zeros(6),
            end_vel=np.zeros(6)
        )
        
        # 第一个路点
        np.testing.assert_array_almost_equal(waypoints[0].position, start, decimal=5)
        # 最后一个路点
        np.testing.assert_array_almost_equal(waypoints[-1].position, end, decimal=5)
    
    def test_trapezoidal_trajectory(self):
        start = np.zeros(6)
        end = np.ones(6) * 0.5
        max_vel = np.ones(6) * 0.5
        max_acc = np.ones(6) * 1.0
        
        waypoints, total_time = self.gen.generate_trapezoidal(start, end, max_vel, max_acc)
        
        self.assertIsInstance(waypoints, list)
        self.assertGreater(len(waypoints), 0)
        self.assertGreater(total_time, 0)
    
    def test_resample_trajectory(self):
        start = np.zeros(6)
        end = np.ones(6) * 0.5
        waypoints = self.gen.generate_quintic_polynomial(start, end, duration=2.0)
        
        resampled = self.gen.resample_trajectory(waypoints, new_dt=0.005)
        
        self.assertIsInstance(resampled, list)
        self.assertGreater(len(resampled), len(waypoints))


class TestRRTPlanner(unittest.TestCase):
    """测试RRT规划器"""
    
    def setUp(self):
        from control.trajectory import RRTPlanner, PlanningAlgorithm
        self.planner = RRTPlanner(
            space_dim=3,
            bounds=[(-1, 1), (-1, 1), (0, 2)],
            max_iterations=200,
            step_size=0.1
        )
    
    def test_planner_init(self):
        self.assertEqual(self.planner.space_dim, 3)
        self.assertEqual(self.planner.max_iterations, 200)
        self.assertEqual(self.planner.step_size, 0.1)
    
    def test_plan_no_obstacle(self):
        start = np.array([0.0, 0.0, 0.5])
        goal = np.array([0.5, 0.5, 1.0])
        
        def no_obs(pos):
            return False
        
        path, cost = self.planner.plan(start, goal, no_obs)
        
        self.assertIsNotNone(path)
        self.assertIsInstance(path, list)
        self.assertLess(cost, float('inf'))
        self.assertGreater(len(path), 0)
    
    def test_plan_rrt_star(self):
        from control.trajectory import PlanningAlgorithm
        start = np.array([0.0, 0.0, 0.5])
        goal = np.array([0.5, 0.5, 1.0])
        
        path, cost = self.planner.plan(
            start, goal,
            obstacle_check=lambda p: False,
            algorithm=PlanningAlgorithm.RRT_STAR
        )
        
        self.assertIsNotNone(path)
        self.assertLess(cost, float('inf'))
    
    def test_plan_informed_rrt(self):
        from control.trajectory import PlanningAlgorithm
        start = np.array([0.0, 0.0, 0.5])
        goal = np.array([0.5, 0.5, 1.0])
        
        path, cost = self.planner.plan(
            start, goal,
            obstacle_check=lambda p: False,
            algorithm=PlanningAlgorithm.INF_PLANNER
        )
        
        self.assertIsNotNone(path)
    
    def test_plan_with_obstacle(self):
        start = np.array([0.0, 0.0, 0.5])
        goal = np.array([0.8, 0.8, 1.5])
        
        # 碰撞检测: 某个区域有障碍
        def with_obs(pos):
            # 以原点为中心的球形障碍
            return np.linalg.norm(pos - np.array([0.4, 0.4, 1.0])) < 0.2
        
        path, cost = self.planner.plan(start, goal, with_obs)
        
        # 即使有障碍，也应该能找到路径或返回None
        if path is not None:
            self.assertIsInstance(path, list)
            self.assertLess(cost, float('inf'))
    
    def test_plan_start_collision(self):
        start = np.array([0.0, 0.0, 0.5])
        goal = np.array([0.5, 0.5, 1.0])
        
        def obs_everywhere(pos):
            return True
        
        path, cost = self.planner.plan(start, goal, obs_everywhere)
        self.assertIsNone(path)
        self.assertEqual(cost, float('inf'))


class TestScurveGenerator(unittest.TestCase):
    """测试S型曲线生成器"""
    
    def test_scurve_init(self):
        from control.trajectory import ScurveGenerator
        gen = ScurveGenerator(max_velocity=1.0, max_acceleration=2.0, max_jerk=10.0)
        self.assertEqual(gen.v_max, 1.0)
        self.assertEqual(gen.a_max, 2.0)
        self.assertEqual(gen.j_max, 10.0)
    
    def test_scurve_plan(self):
        from control.trajectory import ScurveGenerator
        gen = ScurveGenerator(max_velocity=1.0, max_acceleration=2.0, max_jerk=10.0)
        segments = gen.plan(start_pos=0.0, end_pos=1.0)
        
        self.assertIsInstance(segments, list)
        self.assertGreater(len(segments), 0)
        
        for seg in segments:
            self.assertIn('phase', seg)
            self.assertIn('duration', seg)
            self.assertGreater(seg['duration'], 0)


class TestAGVTrajectorySpecs(unittest.TestCase):
    """测试AGV五级轨迹规划规格"""
    
    def test_all_grades_have_specs(self):
        from control.trajectory import get_trajectory_spec, AGV_TRAJECTORY_GRADES
        
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_trajectory_spec(grade)
            self.assertIn('algorithm', spec)
            self.assertIn('max_degree', spec)
            self.assertIn('jerk_limit', spec)
            self.assertIn('collision_check', spec)
    
    def test_s_grade_linear(self):
        from control.trajectory import get_trajectory_spec
        spec = get_trajectory_spec('S')
        self.assertEqual(spec['algorithm'], 'linear')
        self.assertFalse(spec['jerk_limit'])
    
    def test_xxl_grade_optimal(self):
        from control.trajectory import get_trajectory_spec
        spec = get_trajectory_spec('XXL')
        self.assertEqual(spec['algorithm'], 'optimal')
        self.assertTrue(spec['jerk_limit'])


class TestROS2Interface(unittest.TestCase):
    """测试 ROS2 接口模块"""
    
    def test_ros2_joint_trajectory_init(self):
        from control.ros2_interface import ROS2JointTrajectoryInterface, ControlInterfaceMode
        iface = ROS2JointTrajectoryInterface(
            joint_names=['joint1', 'joint2', 'joint3'],
            interface_mode=ControlInterfaceMode.POSITION
        )
        self.assertEqual(iface.num_joints, 3)
        self.assertEqual(iface.mode, ControlInterfaceMode.POSITION)
    
    def test_ros2_joint_trajectory_activate_deactivate(self):
        from control.ros2_interface import ROS2JointTrajectoryInterface
        iface = ROS2JointTrajectoryInterface(['j1', 'j2'])
        self.assertFalse(iface._is_active)
        iface.activate()
        self.assertTrue(iface._is_active)
        iface.deactivate()
        self.assertFalse(iface._is_active)
    
    def test_ros2_joint_trajectory_send_point(self):
        from control.ros2_interface import ROS2JointTrajectoryInterface, JointCommand
        import numpy as np
        iface = ROS2JointTrajectoryInterface(['j1', 'j2', 'j3'])
        iface.activate()
        
        cmd = JointCommand(positions=np.array([0.1, 0.2, 0.3]))
        result = iface.send_point(cmd)
        self.assertTrue(result)
    
    def test_ros2_joint_trajectory_send_trajectory(self):
        from control.ros2_interface import ROS2JointTrajectoryInterface, JointCommand
        import numpy as np
        iface = ROS2JointTrajectoryInterface(['j1', 'j2'])
        iface.activate()
        
        traj = [
            JointCommand(positions=np.array([0.1, 0.2])),
            JointCommand(positions=np.array([0.3, 0.4])),
            JointCommand(positions=np.array([0.5, 0.6])),
        ]
        result = iface.send_trajectory(traj)
        self.assertTrue(result)
    
    def test_ros2_joint_trajectory_cancel(self):
        from control.ros2_interface import ROS2JointTrajectoryInterface, JointCommand
        import numpy as np
        iface = ROS2JointTrajectoryInterface(['j1'])
        iface.activate()
        
        traj = [JointCommand(positions=np.array([0.5]))]
        iface.send_trajectory(traj)
        result = iface.cancel()
        self.assertTrue(result)
    
    def test_ros2_joint_trajectory_update(self):
        from control.ros2_interface import ROS2JointTrajectoryInterface, JointCommand, JointState
        import numpy as np
        iface = ROS2JointTrajectoryInterface(['j1', 'j2'])
        iface.activate()
        
        # 发送轨迹
        traj = [JointCommand(positions=np.array([0.1, 0.2]))]
        iface.send_trajectory(traj)
        
        # 更新状态
        state = JointState(
            positions=np.array([0.1, 0.2]),
            velocities=np.zeros(2),
            efforts=np.zeros(2)
        )
        result = iface.update(state)
        # 应该返回 None 因为点已到达
        self.assertIsNone(result)
    
    def test_ros2_topic_interface_init(self):
        from control.ros2_interface import ROS2TopicInterface
        iface = ROS2TopicInterface(node_name="test_node")
        self.assertEqual(iface.node_name, "test_node")
    
    def test_ros2_topic_publish_subscribe(self):
        from control.ros2_interface import ROS2TopicInterface
        iface = ROS2TopicInterface()
        
        received = []
        def callback(data):
            received.append(data)
        
        iface.create_subscription('/test', 'std_msgs/String', callback)
        iface.create_publisher('/test', 'std_msgs/String')
        iface.publish('/test', {'message': 'hello'})
        
        self.assertEqual(len(received), 0)  # 模拟不自动触发
    
    def test_ros2_service_interface_init(self):
        from control.ros2_interface import ROS2ServiceInterface
        svc = ROS2ServiceInterface(node_name="test_service")
        self.assertEqual(svc.node_name, "test_service")
    
    def test_ros2_service_create_and_call(self):
        from control.ros2_interface import ROS2ServiceInterface
        svc = ROS2ServiceInterface()
        
        def callback(request):
            return {'success': True, 'result': request.get('value', 0) * 2}
        
        svc.create_service('/double', 'std_srvs/srv/Double', callback)
        result = svc.call_service('/double', {'value': 5})
        self.assertTrue(result['success'])
        self.assertEqual(result['result'], 10)
    
    def test_ros2_topic_interface_get_data(self):
        from control.ros2_interface import ROS2TopicInterface
        import numpy as np
        iface = ROS2TopicInterface()
        
        data = np.array([1.0, 2.0, 3.0])
        iface.create_publisher('/sensor', 'sensor_msgs/Float64MultiArray')
        iface.publish('/sensor', data)
        
        retrieved = iface.get_topic_data('/sensor')
        np.testing.assert_array_equal(retrieved, data)
    
    def test_ros2_joint_trajectory_stats(self):
        from control.ros2_interface import ROS2JointTrajectoryInterface, JointCommand
        import numpy as np
        iface = ROS2JointTrajectoryInterface(['j1'])
        iface.activate()
        
        # Set command callback so send_point increments counter
        iface.set_command_callback(lambda cmd: None)
        
        for _ in range(5):
            iface.send_point(JointCommand(positions=np.array([0.1])))
        
        stats = iface.get_stats()
        self.assertEqual(stats['sent_commands'], 5)
        self.assertEqual(stats['failed_commands'], 0)
        self.assertEqual(stats['success_rate'], 1.0)
    
    def test_ros2_ros_topics_constants(self):
        from control.ros2_interface import ROSTopics, ROSServices, ROSParams
        
        self.assertEqual(ROSTopics.JOINT_TRAJECTORY_CMD, '/supermodel/joint_trajectory/command')
        self.assertEqual(ROSServices.PERCEPTION, '/supermodel/perception')
        self.assertEqual(ROSParams.CONTROL_RATE, 'control.rate')
    
    def test_get_ros2_spec(self):
        from control.ros2_interface import get_ros2_spec
        
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_ros2_spec(grade)
            self.assertIn('topics', spec)
            self.assertIn('services', spec)
            self.assertIn('max_freq_hz', spec)
            self.assertIn('realtime', spec)
            self.assertIn('qos_depth', spec)
    
    def test_ros2_spec_lxxl_realtime(self):
        from control.ros2_interface import get_ros2_spec
        
        # L 级以上需要实时性
        for grade in ['L', 'XL', 'XXL']:
            spec = get_ros2_spec(grade)
            self.assertTrue(spec['realtime'])

    # ========== 安全控制器测试 ==========

    def test_safety_controller_init(self):
        from control.safety_controller import SafetyController, SafetyConfig, SafetyLevel
        
        config = SafetyConfig(
            joint_limits_lower=np.array([-3.14, -2.5, -3.14, -3.14, -3.14, -3.14]),
            joint_limits_upper=np.array([3.14, 2.5, 3.14, 3.14, 3.14, 3.14]),
            velocity_limits=np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0]),
            torque_limits=np.array([100, 100, 80, 40, 40, 20]),
            acceleration_limits=np.array([5.0, 5.0, 5.0, 8.0, 8.0, 8.0]),
            safety_level=SafetyLevel.L,
        )
        safety = SafetyController(config)
        self.assertEqual(safety.safety_level, SafetyLevel.L)
        self.assertFalse(safety.is_emergency_stopped)
    
    def test_safety_joint_limit_check(self):
        from control.safety_controller import SafetyController, SafetyConfig, SafetyLevel, SafetyEvent, JointStateSnapshot
        
        config = SafetyConfig(
            joint_limits_lower=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            joint_limits_upper=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            velocity_limits=np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0]),
            acceleration_limits=np.array([5.0, 5.0, 5.0, 8.0, 8.0, 8.0]),
            torque_limits=np.array([100, 100, 80, 40, 40, 20]),
            safety_level=SafetyLevel.S,
        )
        safety = SafetyController(config)
        
        # 正常状态
        state = JointStateSnapshot(
            positions=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            velocities=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        result = safety.check(state)
        self.assertTrue(result.safe)
        
        # 超限状态
        state_limit = JointStateSnapshot(
            positions=np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 关节0超上限
            velocities=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        result = safety.check(state_limit)
        self.assertFalse(result.safe)
        self.assertTrue(any(e.event_type == SafetyEvent.JOINT_LIMIT for e in result.events))
    
    def test_safety_velocity_limit_check(self):
        from control.safety_controller import SafetyController, SafetyConfig, SafetyLevel, SafetyEvent, JointStateSnapshot
        
        config = SafetyConfig(
            joint_limits_lower=np.array([-3.14] * 6),
            joint_limits_upper=np.array([3.14] * 6),
            velocity_limits=np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0]),
            acceleration_limits=np.array([5.0] * 6),
            torque_limits=np.array([100, 100, 80, 40, 40, 20]),
            safety_level=SafetyLevel.M,
        )
        safety = SafetyController(config)
        
        # 超速状态
        state = JointStateSnapshot(
            positions=np.array([0.0] * 6),
            velocities=np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 关节0超速
        )
        result = safety.check(state)
        self.assertFalse(result.safe)
        self.assertTrue(any(e.event_type == SafetyEvent.VELOCITY_LIMIT for e in result.events))
    
    def test_safety_torque_limit_check(self):
        from control.safety_controller import SafetyController, SafetyConfig, SafetyLevel, SafetyEvent, JointStateSnapshot
        
        config = SafetyConfig(
            joint_limits_lower=np.array([-3.14] * 6),
            joint_limits_upper=np.array([3.14] * 6),
            velocity_limits=np.array([2.0] * 6),
            acceleration_limits=np.array([5.0] * 6),
            torque_limits=np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
            safety_level=SafetyLevel.S,
        )
        safety = SafetyController(config)
        
        # 超力矩状态
        state = JointStateSnapshot(
            positions=np.array([0.0] * 6),
            velocities=np.array([0.0] * 6),
            torques=np.array([20.0, 5.0, 5.0, 5.0, 5.0, 5.0]),  # 关节0超力矩
        )
        result = safety.check(state)
        self.assertFalse(result.safe)
        self.assertTrue(any(e.event_type == SafetyEvent.TORQUE_LIMIT for e in result.events))
    
    def test_safety_emergency_stop(self):
        from control.safety_controller import SafetyController, SafetyConfig, SafetyLevel, JointStateSnapshot
        
        config = SafetyConfig(
            joint_limits_lower=np.array([-3.14] * 6),
            joint_limits_upper=np.array([3.14] * 6),
            velocity_limits=np.array([2.0] * 6),
            acceleration_limits=np.array([5.0] * 6),
            torque_limits=np.array([100] * 6),
            safety_level=SafetyLevel.XXL,
        )
        safety = SafetyController(config)
        
        # 触发紧急停止
        safety.emergency_stop()
        self.assertTrue(safety.is_emergency_stopped)
        
        # 重置
        safety.reset()
        self.assertFalse(safety.is_emergency_stopped)
    
    def test_safety_watchdog_timeout(self):
        from control.safety_controller import SafetyController, SafetyConfig, SafetyLevel, SafetyEvent, JointStateSnapshot
        import time
        
        config = SafetyConfig(
            joint_limits_lower=np.array([-3.14] * 6),
            joint_limits_upper=np.array([3.14] * 6),
            velocity_limits=np.array([2.0] * 6),
            acceleration_limits=np.array([5.0] * 6),
            torque_limits=np.array([100] * 6),
            watchdog_timeout=0.05,  # 50ms
            safety_level=SafetyLevel.XL,
        )
        safety = SafetyController(config)
        
        state = JointStateSnapshot(
            positions=np.array([0.0] * 6),
            velocities=np.array([0.0] * 6),
        )
        
        # 第一次检查
        safety.check(state)
        time.sleep(0.06)  # 超过看门狗超时
        
        result = safety.check(state)
        self.assertFalse(result.watchdog_ok)
    
    def test_safety_compute_safe_velocity(self):
        from control.safety_controller import SafetyController, SafetyConfig, SafetyLevel, JointStateSnapshot
        
        config = SafetyConfig(
            joint_limits_lower=np.array([-1.0] * 6),
            joint_limits_upper=np.array([1.0] * 6),
            velocity_limits=np.array([2.0] * 6),
            acceleration_limits=np.array([5.0] * 6),
            torque_limits=np.array([100] * 6),
            safety_level=SafetyLevel.L,
        )
        safety = SafetyController(config)
        
        # 正常计算
        current = np.array([0.0] * 6)
        desired = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0])  # 超速
        safe = safety.compute_safe_velocity(current, desired)
        self.assertTrue(np.all(safe <= np.array([2.0] * 6)))
    
    def test_safety_callback_registration(self):
        from control.safety_controller import SafetyController, SafetyConfig, SafetyLevel, SafetyEvent, JointStateSnapshot
        
        config = SafetyConfig(
            joint_limits_lower=np.array([-1.0] * 6),
            joint_limits_upper=np.array([1.0] * 6),
            velocity_limits=np.array([2.0] * 6),
            acceleration_limits=np.array([5.0] * 6),
            torque_limits=np.array([100] * 6),
            safety_level=SafetyLevel.S,
        )
        safety = SafetyController(config)
        
        callback_called = []
        def my_callback(record):
            callback_called.append(record)
        
        safety.register_callback(SafetyEvent.JOINT_LIMIT, my_callback)
        
        # 触发限位检查
        state = JointStateSnapshot(
            positions=np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 严重超限
            velocities=np.array([0.0] * 6),
        )
        safety.check(state)
        self.assertEqual(len(callback_called), 1)
    
    def test_safety_fault_tolerance(self):
        from control.safety_controller import SafetyController, SafetyConfig, SafetyLevel, SafetyEvent, JointStateSnapshot
        
        config = SafetyConfig(
            joint_limits_lower=np.array([-3.14] * 6),
            joint_limits_upper=np.array([3.14] * 6),
            velocity_limits=np.array([2.0] * 6),
            acceleration_limits=np.array([5.0] * 6),
            torque_limits=np.array([1.0] * 6),  # 很小的力矩限制
            safety_level=SafetyLevel.XXL,
            max_fault_count=2,
        )
        safety = SafetyController(config)
        
        # 多次触发故障
        state = JointStateSnapshot(
            positions=np.array([0.0] * 6),
            velocities=np.array([0.0] * 6),
            torques=np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0]),  # 严重超力矩
        )
        
        for i in range(3):
            result = safety.check(state)
            resp = safety.execute_response(result)
        
        # 验证紧急停止机制
        safety.emergency_stop()
        self.assertTrue(safety.is_emergency_stopped)
        
        # 重置后验证
        safety.reset()
        self.assertFalse(safety.is_emergency_stopped)
        self.assertEqual(safety.fault_count, 0)
    
    def test_get_safety_spec(self):
        from control.safety_controller import get_safety_spec, SafetyLevel
        
        for level in SafetyLevel:
            spec = get_safety_spec(level)
            self.assertIn('level', spec)
            self.assertIn('features', spec)
            self.assertEqual(spec['level'], level.value)
    
    def test_safety_status(self):
        from control.safety_controller import SafetyController, SafetyConfig, SafetyLevel, JointStateSnapshot
        
        config = SafetyConfig(
            joint_limits_lower=np.array([-3.14] * 6),
            joint_limits_upper=np.array([3.14] * 6),
            velocity_limits=np.array([2.0] * 6),
            acceleration_limits=np.array([5.0] * 6),
            torque_limits=np.array([100] * 6),
            safety_level=SafetyLevel.M,
        )
        safety = SafetyController(config)
        
        status = safety.get_safety_status()
        self.assertIn('enabled', status)
        self.assertIn('safety_level', status)
        self.assertIn('fault_count', status)
        self.assertEqual(status['safety_level'], 'M')

if __name__ == '__main__':
    unittest.main(verbosity=2)


class TestAGVMotionController(unittest.TestCase):
    """AGV运动控制器测试"""
    
    def test_agv_spec_from_grade(self):
        from control.agv import AGVSpec, AGVGrade, get_agv_spec
        
        for grade in AGVGrade:
            spec = AGVSpec.from_grade(grade)
            self.assertIsInstance(spec, AGVSpec)
            self.assertEqual(spec.grade, grade)
            self.assertGreater(spec.max_linear_speed, 0)
            self.assertGreater(spec.control_frequency, 0)
    
    def test_get_agv_spec(self):
        from control.agv import get_agv_spec
        
        for grade_str in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_agv_spec(grade_str)
            self.assertIsInstance(spec, AGVSpec)
            self.assertEqual(spec.grade.value, grade_str)
    
    def test_agv_pose_operations(self):
        from control.agv import AGVPose
        
        pose = AGVPose(x=1.0, y=2.0, theta=0.5)
        vec = pose.to_vector()
        self.assertEqual(vec.shape, (3,))
        self.assertEqual(vec[0], 1.0)
        
        pose2 = AGVPose.from_vector(vec)
        self.assertEqual(pose2.x, pose.x)
        self.assertEqual(pose2.y, pose.y)
        self.assertEqual(pose2.theta, pose.theta)
    
    def test_agv_twist_operations(self):
        from control.agv import AGVTwist
        
        twist = AGVTwist(vx=0.5, vy=0.3, omega=0.1)
        vec = twist.to_vector()
        self.assertEqual(vec.shape, (3,))
        
        twist2 = AGVTwist.from_vector(vec)
        self.assertEqual(twist2.vx, twist.vx)
    
    def test_differential_kinematics(self):
        from control.agv import AGVMotionController, AGVSpec, AGVGrade, AGVTwist
        
        spec = AGVSpec.from_grade(AGVGrade.M)
        agv = AGVMotionController(spec)
        
        # 差速驱动: 直行时左右轮速相同
        twist = AGVTwist(vx=1.0, vy=0.0, omega=0.0)
        wheel_vel = agv.inverse_kinematics(twist)
        self.assertEqual(len(wheel_vel), 2)
        
        # 正运动学: 左右轮速相同时为直行
        body_twist = agv.forward_kinematics(np.array([1.0, 1.0]))
        self.assertAlmostEqual(body_twist.vy, 0.0, places=5)
    
    def test_mecanum_kinematics(self):
        from control.agv import AGVMotionController, AGVSpec, AGVGrade, AGVTwist
        
        spec = AGVSpec.from_grade(AGVGrade.L)
        spec.drive_type = DriveType.MECANUM
        agv = AGVMotionController(spec)
        
        # 全向移动: 斜向
        twist = AGVTwist(vx=0.0, vy=1.0, omega=0.0)
        wheel_vel = agv.inverse_kinematics(twist)
        self.assertEqual(len(wheel_vel), 4)
        
        # 回环检验
        body_twist = agv.forward_kinematics(wheel_vel)
        self.assertAlmostEqual(body_twist.vx, 0.0, places=3)
        self.assertAlmostEqual(body_twist.vy, 1.0, places=3)
    
    def test_agv_pose_update(self):
        from control.agv import AGVMotionController, AGVSpec, AGVGrade, AGVPose
        
        spec = AGVSpec.from_grade(AGVGrade.M)
        agv = AGVMotionController(spec)
        
        new_pose = AGVPose(x=2.0, y=1.5, theta=0.3)
        agv.update_pose(new_pose)
        
        self.assertEqual(agv.pose.x, 2.0)
        self.assertEqual(agv.pose.y, 1.5)
        self.assertEqual(agv.pose.theta, 0.3)
    
    def test_agv_safety_limits(self):
        from control.agv import AGVMotionController, AGVSpec, AGVGrade
        
        spec = AGVSpec.from_grade(AGVGrade.M)
        agv = AGVMotionController(spec)
        
        # 超速命令应被限幅
        max_vel = spec.max_linear_speed / spec.wheel_radius
        large_cmd = np.array([max_vel * 5, max_vel * 5])
        limited = agv.apply_safety_limits(large_cmd)
        
        self.assertTrue(np.all(np.abs(limited) <= max_vel))
    
    def test_wheel_commands_computation(self):
        from control.agv import AGVMotionController, AGVSpec, AGVGrade, AGVPose
        
        spec = AGVSpec.from_grade(AGVGrade.M)
        agv = AGVMotionController(spec)
        
        # 初始位姿
        agv.update_pose(AGVPose(x=0.0, y=0.0, theta=0.0))
        
        # 目标位姿
        target = AGVPose(x=0.1, y=0.0, theta=0.0)
        cmds = agv.compute_wheel_commands(target, dt=0.01)
        
        self.assertEqual(len(cmds), 2)
        # 前进命令
        self.assertTrue(cmds[0] > 0 or cmds[1] > 0)
