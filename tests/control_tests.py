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


if __name__ == '__main__':
    unittest.main(verbosity=2)
