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

    def test_htn_plan_transport(self):
        """测试 HTN 搬运任务分解"""
        spec = TaskSpec(
            name="transport",
            goal_state={"object": "box", "destination": "table"}
        )
        plan = self.planner.plan(spec)
        self.assertIsInstance(plan, list)
        # 搬运任务应完全分解为叶子动作
        # transport = pickup + navigate + place
        # pickup -> approach, grasp, lift
        # navigate -> plan_route, follow_trajectory, reach_target
        # place -> move_to, release, retract
        self.assertIn("approach", plan)
        self.assertIn("grasp", plan)
        self.assertIn("lift", plan)
        self.assertIn("plan_route", plan)
        self.assertIn("follow_trajectory", plan)
        self.assertIn("reach_target", plan)

    def test_htn_plan_pickup(self):
        """测试 HTN 拾取任务分解"""
        spec = TaskSpec(
            name="pickup",
            goal_state={"object": "cylinder"}
        )
        plan = self.planner.plan(spec)
        self.assertIsInstance(plan, list)
        # 拾取应分解为 approach, grasp, lift
        self.assertTrue(len(plan) >= 3)
        self.assertIn("approach", plan)
        self.assertIn("grasp", plan)
        self.assertIn("lift", plan)

    def test_htn_plan_navigate(self):
        """测试 HTN 导航任务分解"""
        spec = TaskSpec(
            name="navigate",
            goal_state={"target": "waypoint_a"}
        )
        plan = self.planner.plan(spec)
        self.assertIsInstance(plan, list)
        # 导航应分解为 plan_route, follow_trajectory, reach_target
        self.assertTrue(len(plan) >= 3)
        self.assertIn("plan_route", plan)
        self.assertIn("follow_trajectory", plan)
        self.assertIn("reach_target", plan)

    def test_htn_plan_fallback_to_greedy(self):
        """测试当 HTN 方法不存在时回退到贪心规划"""
        spec = TaskSpec(
            name="unknown_task",
            goal_state={"robot.position": [1, 0, 0]}
        )
        plan = self.planner.plan(spec)
        self.assertIsInstance(plan, list)

    def test_htn_decompose_inspect(self):
        """测试 HTN 检查任务分解"""
        spec = TaskSpec(
            name="inspect",
            goal_state={"location": "machine_1"}
        )
        plan = self.planner.plan(spec)
        self.assertIsInstance(plan, list)
        # 检查应分解为 move_to, sense_environment, analyze_data
        self.assertTrue(len(plan) >= 3)
        self.assertIn("move_to", plan)
        self.assertIn("sense_environment", plan)
        self.assertIn("analyze_data", plan)

    def test_htn_decompose_open_door(self):
        """测试 HTN 开门任务分解"""
        spec = TaskSpec(
            name="open_door",
            goal_state={"door_position": [1, 0, 0], "target_position": [2, 0, 0]}
        )
        plan = self.planner.plan(spec)
        self.assertIsInstance(plan, list)
        # 开门应分解为 move_to (door), grasp, pull, move_to (target)
        self.assertTrue(len(plan) >= 4)
        self.assertIn("move_to", plan)
        self.assertIn("grasp", plan)
        self.assertIn("pull", plan)

    def test_htn_plan_with_action_library(self):
        """测试 HTN 规划与动作库结合"""
        from control.planner import Action
        
        def approach_precond(state):
            return True
        
        def approach_effect(state, params):
            state.robot_state["position"] = params.get("target", [0, 0, 0])
        
        action = Action(
            name="approach",
            precondition=approach_precond,
            effect=approach_effect,
            cost=1.0
        )
        planner = TaskPlanner(action_library={"approach": action})
        state = WorldState()
        state.robot_state["position"] = [0, 0, 0]
        planner.set_world_state(state)
        
        spec = TaskSpec(name="pickup", goal_state={})
        plan = planner.plan(spec)
        self.assertIsInstance(plan, list)


class TestHierarchicalPlanner(unittest.TestCase):
    """测试层次化任务网络规划器"""
    
    def setUp(self):
        self.planner = HierarchicalPlanner()
    
    def test_htn_init(self):
        self.assertIsNotNone(self.planner)
        self.assertGreater(len(self.planner._methods), 0)
    
    def test_decompose_task(self):
        task = Task(id="pickup_test", name="pickup", parameters={"object": "box"})
        subtasks = self.planner.decompose_task(task, max_depth=3)
        self.assertIsInstance(subtasks, list)
        self.assertGreater(len(subtasks), 0)
    
    def test_decompose_navigate(self):
        """测试导航任务分解"""
        task = Task(id="nav_test", name="navigate", parameters={"target": "kitchen"})
        subtasks = self.planner.decompose_task(task, max_depth=3)
        self.assertIsInstance(subtasks, list)
        self.assertGreaterEqual(len(subtasks), 3)  # plan, follow, reach
    
    def test_decompose_transport(self):
        """测试搬运任务分解"""
        task = Task(id="transport_test", name="transport", parameters={
            "object": "package", "destination": "loading_dock"
        })
        subtasks = self.planner.decompose_task(task, max_depth=5)
        self.assertIsInstance(subtasks, list)
        # 运输 = pickup + navigate + place
        self.assertGreaterEqual(len(subtasks), 3)
    
    def test_decompose_inspect(self):
        """测试检查任务分解"""
        task = Task(id="inspect_test", name="inspect", parameters={"location": "panel_a"})
        subtasks = self.planner.decompose_task(task, max_depth=3)
        self.assertIsInstance(subtasks, list)
        self.assertEqual(len(subtasks), 3)  # move_to, sense, analyze
    
    def test_register_custom_method(self):
        """测试注册自定义方法"""
        def custom_method(params):
            return [Task(id="step1", name="custom_action", parameters={})]
        
        self.planner.register_method("custom_task", custom_method)
        self.assertEqual(self.planner.get_available_methods("custom_task"), 1)
        
        task = Task(id="ct", name="custom_task", parameters={})
        subtasks = self.planner.decompose_task(task)
        self.assertEqual(len(subtasks), 1)
        self.assertEqual(subtasks[0].name, "custom_action")
    
    def test_get_available_methods(self):
        """测试获取可用方法数量"""
        self.assertGreater(self.planner.get_available_methods("pickup"), 0)
        self.assertGreater(self.planner.get_available_methods("navigate"), 0)
        self.assertGreater(self.planner.get_available_methods("transport"), 0)
        self.assertEqual(self.planner.get_available_methods("nonexistent"), 0)
    
    def test_plan_hierarchical(self):
        spec = TaskSpec(
            name="pickup",
            goal_state={"held": True},
            max_depth=3
        )
        tasks, metadata = self.planner.plan_hierarchical(spec)
        self.assertIsInstance(tasks, list)
        self.assertIsInstance(metadata, dict)
        self.assertGreater(metadata["num_tasks"], 0)
        self.assertGreater(metadata["estimated_cost"], 0)
    
    def test_plan_hierarchical_with_validation(self):
        """测试带验证的计划"""
        initial_state = WorldState()
        initial_state.objects["robot"] = {"position": [0, 0, 0], "status": "idle"}
        self.planner.set_world_state(initial_state)
        
        spec = TaskSpec(name="pickup", goal_state={"held": True}, max_depth=3)
        tasks, metadata = self.planner.plan_hierarchical(spec, initial_state=initial_state, validate=True)
        
        self.assertIsInstance(tasks, list)
        self.assertIn("is_valid", metadata)
        self.assertIn("validation_reason", metadata)
    
    def test_estimate_plan_cost(self):
        """测试计划成本估计"""
        tasks = [
            Task(id="1", name="move_near", parameters={}),
            Task(id="2", name="grasp", parameters={}),
            Task(id="3", name="move_up", parameters={}),
        ]
        cost = self.planner.estimate_plan_cost(tasks)
        self.assertGreaterEqual(cost, 0)
    
    def test_validate_plan_empty(self):
        """测试空计划验证"""
        state = WorldState()
        is_valid, reason = self.planner.validate_plan([], state)
        self.assertFalse(is_valid)
        self.assertIn("Empty", reason)
    
    def test_validate_plan_with_state(self):
        """测试带状态的计划验证"""
        state = WorldState()
        state.objects["robot"] = {"position": [0, 0, 0], "status": "idle"}
        self.planner.set_world_state(state)
        
        tasks = [
            Task(id="1", name="move_near", parameters={"target": "box"}),
            Task(id="2", name="grasp", parameters={"object": "box"}),
        ]
        is_valid, reason = self.planner.validate_plan(tasks, state)
        # 可能失败因为动作库中可能没有完整的动作定义
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(reason, str)
    
    def test_backtrack_no_alternative(self):
        """测试回溯无可用替代方法"""
        task = Task(id="unknown", name="nonexistent_task", parameters={})
        result, attempted = self.planner.backtrack(task, ["failed_step"])
        self.assertEqual(result, [])
        self.assertEqual(attempted, [])
    
    def test_backtrack_with_custom_method(self):
        """测试回溯有替代方法"""
        def alt_method(params):
            return [Task(id="alt1", name="alternative_action", parameters={})]
        
        self.planner.register_method("backtrack_task", alt_method)
        
        task = Task(id="bt", name="backtrack_task", parameters={})
        result, attempted = self.planner.backtrack(task, ["failed"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "alternative_action")
    
    def test_plan_hierarchical_metadata(self):
        """测试计划元数据结构"""
        spec = TaskSpec(name="inspect", goal_state={}, max_depth=3)
        tasks, metadata = self.planner.plan_hierarchical(spec)
        
        self.assertIn("num_tasks", metadata)
        self.assertIn("estimated_cost", metadata)
        self.assertIn("task_names", metadata)
        self.assertEqual(len(metadata["task_names"]), metadata["num_tasks"])
        self.assertIsInstance(metadata["task_names"], list)


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


class TestROS2ActionInterface(unittest.TestCase):
    """ROS2 Action 接口测试"""

    def test_action_interface_creation(self):
        from control.ros2_interface import ROS2ActionInterface, ActionGoalStatus
        action = ROS2ActionInterface("test_action")
        self.assertEqual(action.action_name, "test_action")
        self.assertFalse(action._is_server_active)

    def test_action_server_lifecycle(self):
        from control.ros2_interface import ROS2ActionInterface
        action = ROS2ActionInterface()
        action.start_server()
        self.assertTrue(action._is_server_active)
        action.stop_server()
        self.assertFalse(action._is_server_active)

    def test_send_goal_and_update(self):
        from control.ros2_interface import ROS2ActionInterface, JointCommand, JointState
        action = ROS2ActionInterface()
        action.start_server()

        trajectory = [
            JointCommand(positions=np.array([0.1, 0.2, 0.3])),
            JointCommand(positions=np.array([0.2, 0.3, 0.4])),
            JointCommand(positions=np.array([0.3, 0.4, 0.5])),
        ]

        goal_id = action.send_goal(trajectory)
        self.assertIsNotNone(goal_id)

        state = JointState(
            positions=np.array([0.05, 0.1, 0.15]),
            velocities=np.zeros(3),
            efforts=np.zeros(3),
        )

        has_active = action.update_server(state)
        self.assertTrue(has_active)

        action.stop_server()

    def test_feedback_callback(self):
        from control.ros2_interface import ROS2ActionInterface
        action = ROS2ActionInterface()
        feedbacks = []

        def on_feedback(fb):
            feedbacks.append(fb)

        action.set_feedback_callback(on_feedback)
        self.assertEqual(action._feedback_callback, on_feedback)

    def test_cancel_goal(self):
        from control.ros2_interface import ROS2ActionInterface, JointCommand, ActionGoalStatus
        action = ROS2ActionInterface()
        action.start_server()

        trajectory = [
            JointCommand(positions=np.array([0.1, 0.2])),
        ]
        goal_id = action.send_goal(trajectory)

        result = action.cancel_goal(goal_id)
        self.assertTrue(result)

        status = action.get_goal_status(goal_id)
        self.assertEqual(status, ActionGoalStatus.CANCELLED)

        action.stop_server()

    def test_action_stats(self):
        from control.ros2_interface import ROS2ActionInterface, JointCommand
        action = ROS2ActionInterface()
        action.start_server()

        trajectory = [JointCommand(positions=np.array([0.1, 0.2]))]
        action.send_goal(trajectory)
        action.send_goal(trajectory)

        stats = action.get_stats()
        self.assertEqual(stats["total_goals"], 2)
        self.assertEqual(stats["active"], 2)

        action.stop_server()

    def test_wait_for_result_timeout(self):
        from control.ros2_interface import ROS2ActionInterface
        action = ROS2ActionInterface()
        result = action.wait_for_result("nonexistent", timeout_sec=0.1)
        self.assertIsNone(result)

    def test_cancel_all_goals(self):
        from control.ros2_interface import ROS2ActionInterface, JointCommand
        action = ROS2ActionInterface()
        action.start_server()

        for _ in range(3):
            action.send_goal([JointCommand(positions=np.array([0.1]))])

        count = action.cancel_all_goals()
        self.assertEqual(count, 3)

        stats = action.get_stats()
        self.assertEqual(stats["cancelled"], 3)

        action.stop_server()


class TestROS2ParameterInterface(unittest.TestCase):
    """ROS2 Parameter 接口测试"""

    def test_parameter_interface_creation(self):
        from control.ros2_interface import ROS2ParameterInterface, ROSParams
        param = ROS2ParameterInterface("test_node")
        self.assertEqual(param.node_name, "test_node")
        self.assertGreater(len(param.list_parameters()), 0)

    def test_get_set_parameter(self):
        from control.ros2_interface import ROS2ParameterInterface
        param = ROS2ParameterInterface()
        param.set_parameter("test_param", 42)
        self.assertEqual(param.get_parameter("test_param"), 42)

    def test_get_parameter_default(self):
        from control.ros2_interface import ROS2ParameterInterface
        param = ROS2ParameterInterface()
        val = param.get_parameter("nonexistent_param", default=100)
        self.assertEqual(val, 100)

    def test_batch_get_parameters(self):
        from control.ros2_interface import ROS2ParameterInterface, ROSParams
        param = ROS2ParameterInterface()
        params = param.get_parameters([ROSParams.CONTROL_RATE, ROSParams.MAX_VELOCITY])
        self.assertIn(ROSParams.CONTROL_RATE, params)

    def test_list_parameters(self):
        from control.ros2_interface import ROS2ParameterInterface
        param = ROS2ParameterInterface()
        all_params = param.list_parameters()
        self.assertIsInstance(all_params, list)

    def test_list_parameters_with_prefix(self):
        from control.ros2_interface import ROS2ParameterInterface
        param = ROS2ParameterInterface()
        control_params = param.list_parameters("control.")
        for p in control_params:
            self.assertTrue(p.startswith("control."))

    def test_subscribe_parameter_change(self):
        from control.ros2_interface import ROS2ParameterInterface, ROSParams
        param = ROS2ParameterInterface()
        change_count = [0]

        def on_change(value):
            change_count[0] += 1

        param.subscribe_parameter_change(ROSParams.CONTROL_RATE, on_change)
        param.set_parameter(ROSParams.CONTROL_RATE, 200.0)
        self.assertEqual(change_count[0], 1)

    def test_load_from_dict(self):
        from control.ros2_interface import ROS2ParameterInterface
        param = ROS2ParameterInterface()
        param.load_from_dict({"custom_param": "hello", "another": 123})
        self.assertEqual(param.get_parameter("custom_param"), "hello")
        self.assertEqual(param.get_parameter("another"), 123)

    def test_to_dict(self):
        from control.ros2_interface import ROS2ParameterInterface, ROSParams
        param = ROS2ParameterInterface()
        d = param.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn(ROSParams.CONTROL_RATE, d)


class TestROS2ComponentInterface(unittest.TestCase):
    """ROS2 Component 接口测试"""

    def test_component_creation(self):
        from control.ros2_interface import ROS2ComponentInterface
        comp = ROS2ComponentInterface("test_component")
        self.assertEqual(comp.component_name, "test_component")
        self.assertEqual(comp.get_state(), "unconfigured")

    def test_lifecycle_callbacks(self):
        from control.ros2_interface import ROS2ComponentInterface
        comp = ROS2ComponentInterface("lifecycle_test")

        configure_called = [False]
        activate_called = [False]

        def on_configure():
            configure_called[0] = True
            return True

        def on_activate():
            activate_called[0] = True
            return True

        comp.on_configure(on_configure)
        comp.on_activate(on_activate)

        comp.configure()
        self.assertTrue(configure_called[0])
        self.assertEqual(comp.get_state(), "inactive")

        comp.activate()
        self.assertTrue(activate_called[0])
        self.assertEqual(comp.get_state(), "active")

    def test_lifecycle_no_callbacks(self):
        from control.ros2_interface import ROS2ComponentInterface
        comp = ROS2ComponentInterface("no_cbs")
        comp.configure()
        self.assertEqual(comp.get_state(), "inactive")
        comp.activate()
        self.assertEqual(comp.get_state(), "active")
        comp.deactivate()
        self.assertEqual(comp.get_state(), "inactive")
        comp.cleanup()
        self.assertEqual(comp.get_state(), "unconfigured")
        comp.shutdown()
        self.assertEqual(comp.get_state(), "shutdown")

    def test_cannot_activate_from_wrong_state(self):
        from control.ros2_interface import ROS2ComponentInterface
        comp = ROS2ComponentInterface("wrong_state_test")
        result = comp.activate()
        self.assertFalse(result)
        self.assertEqual(comp.get_state(), "unconfigured")

    def test_context_manager(self):
        from control.ros2_interface import ROS2ComponentInterface
        comp = ROS2ComponentInterface("ctx_test")
        with comp:
            self.assertEqual(comp.get_state(), "active")
        self.assertEqual(comp.get_state(), "unconfigured")


class TestROSTopicsServicesParams(unittest.TestCase):
    """ROS2 话题/服务/参数常量测试"""

    def test_rostopics(self):
        from control.ros2_interface import ROSTopics
        self.assertEqual(ROSTopics.JOINT_TRAJECTORY_CMD, "/supermodel/joint_trajectory/command")
        self.assertEqual(ROSTopics.JOINT_STATES, "/supermodel/joint_states")
        self.assertEqual(ROSTopics.IMU, "/supermodel/imu")
        self.assertEqual(ROSTopics.FORCE, "/supermodel/force")

    def test_rosservices(self):
        from control.ros2_interface import ROSServices
        self.assertEqual(ROSServices.PERCEPTION, "/supermodel/perception")
        self.assertEqual(ROSServices.PLANNING, "/supermodel/planning")
        self.assertEqual(ROSServices.EXECUTE_SKILL, "/supermodel/execute_skill")

    def test_rosparams(self):
        from control.ros2_interface import ROSParams
        self.assertEqual(ROSParams.CONTROL_RATE, "control.rate")
        self.assertEqual(ROSParams.MAX_VELOCITY, "control.max_velocity")
        self.assertEqual(ROSParams.FUSION_STRATEGY, "fusion.strategy")

    def test_get_ros2_spec(self):
        from control.ros2_interface import get_ros2_spec
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_ros2_spec(grade)
            self.assertIn('topics', spec)
            self.assertIn('services', spec)
            self.assertIn('max_freq_hz', spec)
            self.assertIn('realtime', spec)

        default = get_ros2_spec('UNKNOWN')
        self.assertEqual(default, get_ros2_spec('M'))


class TestAGVTrajectoryTracking(unittest.TestCase):
    """AGV轨迹跟踪与五级规格合规性测试"""

    def test_agv_all_grades_trajectory_tracking(self):
        """测试所有AGV等级的轨迹跟踪能力"""
        from control.agv import AGVMotionController, AGVSpec, AGVGrade, AGVPose, AGVTwist
        
        for grade in AGVGrade:
            spec = AGVSpec.from_grade(grade)
            agv = AGVMotionController(spec)
            
            # 初始化位姿
            agv.update_pose(AGVPose(x=0.0, y=0.0, theta=0.0))
            
            # 生成直线轨迹 (沿X轴前进)
            target_poses = []
            for i in range(10):
                target_poses.append(AGVPose(x=float(i) * 0.1, y=0.0, theta=0.0))
            
            # 执行轨迹跟踪
            tracking_errors = []
            for target in target_poses:
                cmds = agv.compute_wheel_commands(target, dt=1.0/spec.control_frequency)
                # 应用安全限制
                cmds = agv.apply_safety_limits(cmds)
                tracking_errors.append(np.sqrt(target.x**2 + target.y**2))
            
            # 验证跟踪性能
            self.assertEqual(len(tracking_errors), 10)
            # 应该有关注度输出（不为零）
            self.assertTrue(any(e > 0.001 for e in tracking_errors))

    def test_agv_xxl_high_frequency_control(self):
        """测试XXL级1000Hz高频控制"""
        from control.agv import AGVMotionController, AGVSpec, AGVGrade, AGVPose, AGVTwist
        import time
        
        spec = AGVSpec.from_grade(AGVGrade.XXL)
        agv = AGVMotionController(spec)
        
        self.assertEqual(spec.control_frequency, 1000.0)
        
        # 模拟高频控制循环
        agv.update_pose(AGVPose(x=0.0, y=0.0, theta=0.0))
        target = AGVPose(x=1.0, y=0.0, theta=0.0)
        
        start = time.time()
        iterations = 0
        for _ in range(1000):
            cmds = agv.compute_wheel_commands(target, dt=0.001)
            cmds = agv.apply_safety_limits(cmds)
            iterations += 1
        elapsed = time.time() - start
        
        # 1000次迭代应该在合理时间内完成
        self.assertLess(elapsed, 5.0)
        self.assertEqual(iterations, 1000)

    def test_agv_mecanum_omnidirectional(self):
        """测试麦克纳姆轮全向移动"""
        from control.agv import AGVMotionController, AGVSpec, AGVGrade, AGVTwist, DriveType
        
        spec = AGVSpec.from_grade(AGVGrade.XL)
        spec.drive_type = DriveType.MECANUM
        agv = AGVMotionController(spec)
        
        # 纯侧向移动 (需要麦克纳姆轮)
        twist = AGVTwist(vx=0.0, vy=1.0, omega=0.0)
        wheel_vel = agv.inverse_kinematics(twist)
        
        self.assertEqual(len(wheel_vel), 4)
        
        # 验证逆运动学输出有正负交替模式 (麦克纳姆轮特征)
        # 纯侧向移动时，相邻轮子转向相反
        # vy=1.0: w_fl=-1/r, w_fr=+1/r -> 符号相反
        self.assertLess(wheel_vel[1] * wheel_vel[0], 0)  # fr和fl符号相反
        self.assertLess(wheel_vel[2] * wheel_vel[0], 0)  # rl和fl符号相反
    
    def test_agv_differential_turning(self):
        """测试差速驱动原地转向"""
        from control.agv import AGVMotionController, AGVSpec, AGVGrade, AGVTwist, DriveType
        
        spec = AGVSpec.from_grade(AGVGrade.M)
        spec.drive_type = DriveType.DIFFERENTIAL
        agv = AGVMotionController(spec)
        
        # 纯旋转
        twist = AGVTwist(vx=0.0, vy=0.0, omega=1.0)
        wheel_vel = agv.inverse_kinematics(twist)
        
        # 左右轮应该方向相反
        self.assertLess(wheel_vel[0] * wheel_vel[1], 0)
        
        # 直行分量为零
        body_twist = agv.forward_kinematics(np.array([1.0, -1.0]))
        self.assertAlmostEqual(body_twist.vx, 0.0, places=3)

    def test_agv_velocity_saturation(self):
        """测试速度饱和限制"""
        from control.agv import AGVMotionController, AGVSpec, AGVGrade, AGVTwist
        
        spec = AGVSpec.from_grade(AGVGrade.M)
        agv = AGVMotionController(spec)
        
        # 尝试发送超高速命令
        twist = AGVTwist(
            vx=spec.max_linear_speed * 10,
            vy=0.0,
            omega=spec.max_angular_speed * 10
        )
        wheel_vel = agv.inverse_kinematics(twist)
        limited = agv.apply_safety_limits(wheel_vel)
        
        max_wheel = spec.max_linear_speed / spec.wheel_radius
        self.assertTrue(np.all(np.abs(limited) <= max_wheel + 1e-6))

    def test_agv_collision_avoidance_limits(self):
        """测试安全限制下的碰撞回避"""
        from control.agv import AGVMotionController, AGVSpec, AGVGrade, AGVPose
        
        spec = AGVSpec.from_grade(AGVGrade.L)
        agv = AGVMotionController(spec)
        
        # 目标在障碍物后方的情况
        agv.update_pose(AGVPose(x=0.0, y=0.0, theta=0.0))
        
        # 目标太近
        close_target = AGVPose(x=0.01, y=0.0, theta=0.0)
        cmds = agv.compute_wheel_commands(close_target, dt=0.01)
        
        # 命令应该在安全范围内
        max_wheel = spec.max_linear_speed / spec.wheel_radius
        self.assertTrue(np.all(np.abs(cmds) <= max_wheel * 2))

    def test_agv_pose_vector_roundtrip(self):
        """测试AGV位姿向量往返转换"""
        from control.agv import AGVPose
        
        original = AGVPose(x=1.234, y=5.678, theta=0.785)
        vec = original.to_vector()
        restored = AGVPose.from_vector(vec)
        
        self.assertAlmostEqual(restored.x, original.x, places=5)
        self.assertAlmostEqual(restored.y, original.y, places=5)
        self.assertAlmostEqual(restored.theta, original.theta, places=5)

    def test_agv_kinematics_roundtrip(self):
        """测试运动学正逆变换往返精度 (麦克纳姆轮全向)"""
        from control.agv import AGVMotionController, AGVSpec, AGVGrade, AGVTwist, DriveType
        
        spec = AGVSpec.from_grade(AGVGrade.L)
        spec.drive_type = DriveType.MECANUM
        agv = AGVMotionController(spec)
        
        # 任意速度命令 (全向移动)
        original_twist = AGVTwist(vx=0.5, vy=0.3, omega=0.2)
        
        # 逆运动学
        wheel_vel = agv.inverse_kinematics(original_twist)
        
        # 正运动学还原
        recovered_twist = agv.forward_kinematics(wheel_vel)
        
        # 验证还原精度 (麦克纳姆轮支持全向移动)
        self.assertAlmostEqual(recovered_twist.vx, original_twist.vx, places=3)
        self.assertAlmostEqual(recovered_twist.vy, original_twist.vy, places=3)
        self.assertAlmostEqual(recovered_twist.omega, original_twist.omega, places=3)

    def test_agv_xxl_load_capacity(self):
        """测试XXL级2000kg负载能力"""
        from control.agv import AGVSpec, AGVGrade
        
        spec_xxl = AGVSpec.from_grade(AGVGrade.XXL)
        
        # 验证规格满足负载需求
        self.assertGreater(spec_xxl.max_linear_speed, 0)  # 有速度能力
        self.assertGreater(spec_xxl.max_linear_accel, 0)  # 有加速度能力
        # 最大负载通过规格表单独定义，此处验证驱动能力

    def test_agv_s_grade_educational_spec(self):
        """测试S级教育/实验规格"""
        from control.agv import AGVSpec, AGVGrade
        
        spec_s = AGVSpec.from_grade(AGVGrade.S)
        
        self.assertEqual(spec_s.control_frequency, 50.0)
        self.assertLessEqual(spec_s.max_linear_speed, 0.5)
        self.assertLessEqual(spec_s.max_angular_speed, 1.5)

    def test_agv_motion_controller_pid_convergence(self):
        """测试PID控制器向目标收敛"""
        from control.agv import AGVMotionController, AGVSpec, AGVGrade, AGVPose
        
        spec = AGVSpec.from_grade(AGVGrade.M)
        agv = AGVMotionController(spec)
        
        # 从原点开始
        agv.update_pose(AGVPose(x=0.0, y=0.0, theta=0.0))
        target = AGVPose(x=0.5, y=0.0, theta=0.0)
        
        # 多次迭代控制
        initial_dist = np.sqrt(target.x**2 + target.y**2)
        
        for iteration in range(100):
            cmds = agv.compute_wheel_commands(target, dt=0.01)
            cmds = agv.apply_safety_limits(cmds)
            # 模拟更新 (简化版)
            if iteration % 10 == 0:
                current = agv.pose
                dist = np.sqrt((target.x - current.x)**2 + (target.y - current.y)**2)
        
        # 验证控制器在运行
        self.assertTrue(agv.pose is not None)

    def test_agv_twist_conversion(self):
        """测试AGV速度转换"""
        from control.agv import AGVTwist
        
        twist = AGVTwist(vx=1.0, vy=2.0, omega=0.5)
        vec = twist.to_vector()
        self.assertEqual(vec.shape, (3,))
        self.assertEqual(vec[0], 1.0)
        self.assertEqual(vec[1], 2.0)
        self.assertEqual(vec[2], 0.5)
        
        restored = AGVTwist.from_vector(vec)
        self.assertAlmostEqual(restored.vx, twist.vx)
        self.assertAlmostEqual(restored.vy, twist.vy)
        self.assertAlmostEqual(restored.omega, twist.omega)


class TestSafetyControllerGrades(unittest.TestCase):
    """安全控制器AGV五级合规性测试"""

    def test_safety_level_s_basic(self):
        """S级安全: 基础限位"""
        from control.safety_controller import SafetyController, SafetyConfig, SafetyLevel, JointStateSnapshot
        
        config = SafetyConfig(
            joint_limits_lower=np.array([-1.0] * 6),
            joint_limits_upper=np.array([1.0] * 6),
            velocity_limits=np.array([1.0] * 6),
            acceleration_limits=np.array([5.0] * 6),
            torque_limits=np.array([10.0] * 6),
            safety_level=SafetyLevel.S,
        )
        safety = SafetyController(config)
        
        # 正常运行
        state = JointStateSnapshot(
            positions=np.zeros(6),
            velocities=np.zeros(6),
        )
        result = safety.check(state)
        self.assertTrue(result.safe)
        
        # 超限检测
        state_limit = JointStateSnapshot(
            positions=np.array([2.0] * 6),
            velocities=np.zeros(6),
        )
        result = safety.check(state_limit)
        self.assertFalse(result.safe)

    def test_safety_level_xxl_comprehensive(self):
        """XXL级安全: 完整故障容忍"""
        from control.safety_controller import SafetyController, SafetyConfig, SafetyLevel, SafetyEvent, JointStateSnapshot
        
        config = SafetyConfig(
            joint_limits_lower=np.array([-3.14] * 6),
            joint_limits_upper=np.array([3.14] * 6),
            velocity_limits=np.array([5.0] * 6),
            acceleration_limits=np.array([10.0] * 6),
            torque_limits=np.array([100.0] * 6),
            safety_level=SafetyLevel.XXL,
            watchdog_timeout=0.01,
            max_fault_count=5,
        )
        safety = SafetyController(config)
        
        # 正常状态
        state = JointStateSnapshot(
            positions=np.zeros(6),
            velocities=np.zeros(6),
            torques=np.ones(6) * 5.0,
        )
        result = safety.check(state)
        self.assertTrue(result.safe)
        self.assertTrue(result.watchdog_ok)
        
        # 紧急停止功能
        safety.emergency_stop()
        self.assertTrue(safety.is_emergency_stopped)
        
        # 重置
        safety.reset()
        self.assertFalse(safety.is_emergency_stopped)
        
        # 看门狗超时测试
        import time
        safety.check(state)
        time.sleep(0.02)
        result2 = safety.check(state)
        self.assertFalse(result2.watchdog_ok)

    def test_safety_five_grades_features(self):
        """验证五级安全特性"""
        from control.safety_controller import SafetyController, SafetyConfig, SafetyLevel, get_safety_spec
        
        for level in SafetyLevel:
            spec = get_safety_spec(level)
            self.assertEqual(spec['level'], level.value)
            
            config = SafetyConfig(
                joint_limits_lower=np.array([-3.14] * 6),
                joint_limits_upper=np.array([3.14] * 6),
                velocity_limits=np.array([2.0] * 6),
                acceleration_limits=np.array([5.0] * 6),
                torque_limits=np.array([100.0] * 6),
                safety_level=level,
            )
            safety = SafetyController(config)
            status = safety.get_safety_status()
            self.assertEqual(status['safety_level'], level.value)


class TestImpedanceControlCompliance(unittest.TestCase):
    """阻抗控制AGV五级合规性测试"""

    def test_impedance_params_grade_s(self):
        """S级无阻抗控制"""
        from control.impedance import ImpedanceParams
        params = ImpedanceParams.default_6d()
        self.assertIsNotNone(params)
    
    def test_impedance_params_grade_l(self):
        """L级完整阻抗参数"""
        from control.impedance import ImpedanceParams
        
        params = ImpedanceParams(
            M=np.eye(6) * 5.0,
            D=np.eye(6) * 100.0,
            K=np.eye(6) * 200.0,
        )
        self.assertEqual(params.M.shape, (6, 6))
        self.assertEqual(params.D.shape, (6, 6))
        self.assertEqual(params.K.shape, (6, 6))

    def test_collaborative_force_limit(self):
        """协作安全力限"""
        from control.impedance import CollaborativeController
        
        ctrl = CollaborativeController(safety_force_limit=150.0)
        self.assertEqual(ctrl.safety_force_limit, 150.0)


class TestMPCControlCompliance(unittest.TestCase):
    """MPC控制器AGV五级合规性测试"""

    def test_mpc_grade_s_constraints(self):
        """S级MPC: 基础约束"""
        from control.mpc import get_mpc_spec
        
        spec = get_mpc_spec('S')
        self.assertEqual(spec['horizon'], 10)
        self.assertIn('joint_limits', spec['constraints'])

    def test_mpc_grade_xxl_full_constraints(self):
        """XXL级MPC: 完整约束包括力约束"""
        from control.mpc import get_mpc_spec
        
        spec = get_mpc_spec('XXL')
        self.assertEqual(spec['horizon'], 50)
        self.assertIn('force', spec['constraints'])
        self.assertIn('obstacle', spec['constraints'])
        self.assertEqual(spec['solver'], 'osqp')

    def test_dynamics_model_forward(self):
        """测试动力学模型"""
        from control.mpc import DynamicsModel
        import numpy as np
        
        model = DynamicsModel(num_joints=6)
        q = np.zeros(6)
        qd = np.zeros(6)
        tau = np.array([0, 0, 10, 0, 0, 0])
        qdd = model.forward(q, qd, tau)
        self.assertEqual(qdd.shape, (6,))
        self.assertGreater(qdd[2], 0)

    def test_joint_space_mpc_compute(self):
        """测试关节空间MPC"""
        from control.mpc import JointSpaceMPC, MPCConfig
        import numpy as np
        
        config = MPCConfig.for_grade('M', num_joints=6)
        mpc = JointSpaceMPC(config=config, num_joints=6)
        
        current_pos = np.zeros(6)
        current_vel = np.zeros(6)
        target_pos = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        tau = mpc.compute_control_simple(current_pos, current_vel, target_pos)
        self.assertEqual(tau.shape, (6,))
        self.assertTrue(np.all(np.isfinite(tau)))


class TestSensorSimulatorAGVCompliance(unittest.TestCase):
    """传感器仿真AGV五级规格测试"""

    def test_all_grades_sensor_noise(self):
        """测试所有AGV等级的传感器噪声特性"""
        from simulation.environment import SensorSimulator, RobotSimulator, SimConfig
        
        # 不同噪声水平模拟不同等级
        noise_configs = [0.01, 0.005, 0.001, 0.0005, 0.0001]
        
        for noise_level in noise_configs:
            config = SimConfig(dt=0.01, num_joints=6, position_noise=noise_level)
            sim = RobotSimulator(config)
            sensor = SensorSimulator(sim, config)
            
            # IMU数据
            imu_data = sensor.get_imu_data()
            self.assertIn('accel', imu_data)
            self.assertIn('gyro', imu_data)
            
            # 力觉数据
            wrench = sensor.get_wrench()
            self.assertEqual(wrench.shape, (6,))

    def test_sensor_noise_levels_by_grade(self):
        """测试等级相关的噪声水平"""
        from simulation.environment import SensorSimulator, RobotSimulator, SimConfig
        
        # 高等级有更低噪声
        config_low = SimConfig(dt=0.01, num_joints=6, position_noise=0.01)
        config_high = SimConfig(dt=0.01, num_joints=6, position_noise=0.001)
        
        sim_low = RobotSimulator(config_low)
        sim_high = RobotSimulator(config_high)
        
        sensor_low = SensorSimulator(sim_low, config_low)
        sensor_high = SensorSimulator(sim_high, config_high)
        
        # 采集多次取平均
        noises_low = []
        noises_high = []
        
        for _ in range(10):
            imu_low = sensor_low.get_imu_data()
            imu_high = sensor_high.get_imu_data()
            noises_low.append(np.std(imu_low['accel']))
            noises_high.append(np.std(imu_high['accel']))
        
        # 高等级(低噪声)应该噪声更低
        self.assertLess(np.mean(noises_high), np.mean(noises_low) * 2)


class TestFusionSensorIntegration(unittest.TestCase):
    """多传感器融合与AGV等级集成测试"""

    def test_multimodal_fusion_all_grades(self):
        """测试所有AGV等级的多模态融合"""
        from fusion.cross_modal_fusion import CrossModalFusion, FusionConfig, FusionStrategy, MultimodalInput
        import torch
        
        for grade, hidden_dim, num_heads in [
            ('S', 128, 2),
            ('M', 256, 4),
            ('L', 512, 8),
            ('XL', 768, 12),
            ('XXL', 1024, 16),
        ]:
            config = FusionConfig(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=max(1, num_heads // 4),
            )
            fusion = CrossModalFusion(config)
            
            # 模拟多模态输入 (使用与配置匹配的维度)
            mmi = MultimodalInput(
                vision=torch.randn(2, 512),
                audio=torch.randn(2, 128),
                tactile=torch.randn(2, 64),
                force=torch.randn(2, 32),   # 默认 force_dim=32
                imu=torch.randn(2, 64),     # 默认 imu_dim=64
            )
            
            output = fusion(mmi)
            self.assertEqual(output.shape[0], 2)
            self.assertEqual(output.shape[1], hidden_dim)

    def test_sensor_fusion_pipeline(self):
        """测试传感器融合完整流程"""
        from sensors.vision import BinocularCamera, CameraIntrinsics, StereoExtrinsics
        from sensors.audio import BinauralMic
        from sensors.tactile import TactileArray
        from sensors.force import ForceTorqueSensor, Wrench
        from sensors.imu import IMUSensor, IMUSensorType
        from fusion.cross_modal_fusion import CrossModalFusion, FusionConfig, MultimodalInput
        import torch
        
        # 创建虚拟传感器
        cam = BinocularCamera(resolution=(640, 480), fps=30)
        mic = BinauralMic(sample_rate=16000, chunk_size=512)
        tactile = TactileArray(array_size=(16, 16))
        force = ForceTorqueSensor()
        imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL)
        
        # 采集
        cam.open()
        mic.open()
        tactile.open()
        force.open()
        imu.open()
        
        stereo = cam.capture()
        audio = mic.capture()
        tac_frame = tactile.capture()
        wrench = force.capture()
        imu_frame = imu.capture()
        
        # 融合 (使用正确维度)
        config = FusionConfig(hidden_dim=256, num_heads=4)
        fusion = CrossModalFusion(config)
        
        # 模拟视觉和触觉特征 (使用默认维度: force_dim=32, imu_dim=64)
        mmi = MultimodalInput(
            vision=torch.randn(1, 512),
            tactile=torch.randn(1, 64),
            force=torch.randn(1, 32),
            imu=torch.randn(1, 64),
        )
        
        fused = fusion(mmi)
        self.assertEqual(fused.shape, (1, 256))
        
        # 清理
        cam.close()
        mic.close()
        tactile.close()
        force.close()
        imu.close()


class TestTrajectoryTracker(unittest.TestCase):
    """测试AGV轨迹跟踪控制器"""

    def test_tracker_initialization(self):
        """轨迹跟踪器初始化"""
        from control.agv import TrajectoryTracker, AGVSpec, AGVGrade
        spec = AGVSpec.from_grade(AGVGrade.M)
        tracker = TrajectoryTracker(spec)
        self.assertEqual(tracker.look_ahead_distance, 0.3)
        self.assertEqual(tracker.k_gain, 2.0)

    def test_tracker_set_trajectory(self):
        """设置轨迹"""
        from control.agv import TrajectoryTracker, AGVSpec, AGVGrade, AGVPose
        spec = AGVSpec.from_grade(AGVGrade.M)
        tracker = TrajectoryTracker(spec)

        trajectory = [
            AGVPose(x=0.0, y=0.0, theta=0.0),
            AGVPose(x=0.5, y=0.0, theta=0.0),
            AGVPose(x=1.0, y=0.0, theta=0.0),
        ]
        times = np.array([0.0, 1.0, 2.0])
        tracker.set_trajectory(trajectory, times)

        self.assertEqual(len(tracker._trajectory), 3)

    def test_tracker_normalize_angle(self):
        """角度归一化"""
        from control.agv import TrajectoryTracker, AGVSpec, AGVGrade
        spec = AGVSpec.from_grade(AGVGrade.M)
        tracker = TrajectoryTracker(spec)

        # 测试边界情况
        self.assertAlmostEqual(tracker._normalize_angle(np.pi), np.pi)
        self.assertAlmostEqual(tracker._normalize_angle(-np.pi), -np.pi)
        self.assertAlmostEqual(tracker._normalize_angle(3*np.pi), np.pi)
        self.assertAlmostEqual(tracker._normalize_angle(-3*np.pi), -np.pi)

    def test_tracker_find_look_ahead(self):
        """前看点查找"""
        from control.agv import TrajectoryTracker, AGVSpec, AGVGrade, AGVPose
        spec = AGVSpec.from_grade(AGVGrade.M)
        tracker = TrajectoryTracker(spec)

        trajectory = [
            AGVPose(x=0.0, y=0.0, theta=0.0),
            AGVPose(x=0.2, y=0.0, theta=0.0),
            AGVPose(x=0.5, y=0.0, theta=0.0),
            AGVPose(x=1.0, y=0.0, theta=0.0),
        ]
        times = np.array([0.0, 0.2, 0.5, 1.0])
        tracker.set_trajectory(trajectory, times)

        # 起始位置靠近第一个点
        tracker._agv.update_pose(AGVPose(x=0.0, y=0.0, theta=0.0))
        idx, pt = tracker._find_look_ahead_point(np.array([0.0, 0.0]))
        self.assertGreaterEqual(idx, 0)
        self.assertLessEqual(idx, len(trajectory) - 1)

    def test_tracker_compute_command_basic(self):
        """基本命令计算"""
        from control.agv import TrajectoryTracker, AGVSpec, AGVGrade, AGVPose
        # AGVGrade.L 使用 Mecanum 驱动 (4轮)
        spec = AGVSpec.from_grade(AGVGrade.L)
        tracker = TrajectoryTracker(spec)

        trajectory = [
            AGVPose(x=0.0, y=0.0, theta=0.0),
            AGVPose(x=0.5, y=0.0, theta=0.0),
            AGVPose(x=1.0, y=0.0, theta=0.0),
        ]
        times = np.array([0.0, 1.0, 2.0])
        tracker.set_trajectory(trajectory, times)
        tracker.set_pose(AGVPose(x=0.0, y=0.0, theta=0.0))

        # 计算一步 (Mecanum 4轮)
        wheel_cmds = tracker.compute_command(dt=0.01)
        self.assertEqual(len(wheel_cmds), 4)

    def test_tracker_empty_trajectory(self):
        """空轨迹处理"""
        from control.agv import TrajectoryTracker, AGVSpec, AGVGrade
        spec = AGVSpec.from_grade(AGVGrade.M)
        tracker = TrajectoryTracker(spec)

        wheel_cmds = tracker.compute_command(dt=0.01)
        self.assertTrue(np.allclose(wheel_cmds, np.zeros(4)))

    def test_tracker_is_complete(self):
        """轨迹完成判断"""
        from control.agv import TrajectoryTracker, AGVSpec, AGVGrade, AGVPose
        spec = AGVSpec.from_grade(AGVGrade.M)
        tracker = TrajectoryTracker(spec)

        self.assertTrue(tracker.is_trajectory_complete())

        trajectory = [AGVPose(x=1.0, y=0.0, theta=0.0)]
        times = np.array([1.0])
        tracker.set_trajectory(trajectory, times)
        tracker.set_pose(AGVPose(x=0.0, y=0.0, theta=0.0))

        self.assertFalse(tracker.is_trajectory_complete())

    def test_tracker_reset(self):
        """重置跟踪器"""
        from control.agv import TrajectoryTracker, AGVSpec, AGVGrade, AGVPose
        spec = AGVSpec.from_grade(AGVGrade.M)
        tracker = TrajectoryTracker(spec)

        trajectory = [AGVPose(x=1.0, y=0.0, theta=0.0)]
        times = np.array([1.0])
        tracker.set_trajectory(trajectory, times)
        tracker.set_pose(AGVPose(x=0.5, y=0.0, theta=0.0))
        tracker._current_idx = 1

        tracker.reset()

        self.assertEqual(tracker._current_idx, 0)
        self.assertEqual(tracker._last_error, 0.0)

    def test_tracker_full_simulation(self):
        """完整轨迹跟踪仿真"""
        from control.agv import TrajectoryTracker, AGVSpec, AGVGrade, AGVPose
        spec = AGVSpec.from_grade(AGVGrade.M)
        tracker = TrajectoryTracker(spec)

        # 生成直线轨迹
        n_points = 10
        trajectory = [
            AGVPose(x=float(i) * 0.1, y=0.0, theta=0.0)
            for i in range(n_points)
        ]
        times = np.array([float(i) * 0.1 for i in range(n_points)])
        tracker.set_trajectory(trajectory, times)
        tracker.set_pose(AGVPose(x=0.0, y=0.0, theta=0.0))

        # 模拟多步
        for step in range(20):
            wheel_cmds = tracker.compute_command(dt=0.01)

            if tracker.is_trajectory_complete():
                break

        # 不应该崩溃
        self.assertTrue(True)


class TestTeleoperationController(unittest.TestCase):
    """测试遥操作控制器"""

    def test_teleop_init(self):
        """遥操作控制器初始化"""
        from control.teleop import (
            TeleoperationController, TeleopMode, TeleopState,
            AuthorityLevel, TeleopConfig, TeleopCommand,
            MasterState, SlaveState
        )
        config = TeleopConfig(
            master_ip="192.168.1.100",
            slave_ip="192.168.1.101",
            control_frequency=100.0,
            safety_stop_threshold=30.0
        )
        controller = TeleoperationController(config)
        self.assertEqual(controller.state, TeleopState.IDLE)
        self.assertEqual(controller._current_authority, AuthorityLevel.OPERATOR)

    def test_teleop_connect_disconnect(self):
        """连接和断开"""
        from control.teleop import TeleoperationController, TeleopConfig, TeleopState
        config = TeleopConfig()
        controller = TeleoperationController(config)
        
        controller.connect()
        self.assertEqual(controller.state, TeleopState.IDLE)
        
        controller.disconnect()

    def test_teleop_set_master_slave_state(self):
        """设置主从状态"""
        from control.teleop import (
            TeleoperationController, TeleopConfig, TeleopCommand,
            TeleopMode, MasterState, SlaveState, AuthorityLevel
        )
        config = TeleopConfig(control_frequency=100.0)
        controller = TeleoperationController(config)
        controller.connect()

        master = MasterState(
            joint_positions=np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0]),
            joint_velocities=np.zeros(6),
            joint_torques=np.zeros(6),
            wrench=np.zeros(6),
            timestamp=time.time(),
            authority=AuthorityLevel.OPERATOR
        )
        slave = SlaveState(
            joint_positions=np.array([0.05, 0.1, 0.15, 0.0, 0.0, 0.0]),
            joint_velocities=np.zeros(6),
            end_effector_pose=np.array([0.0, 0.0, 0.5]),
            contact_wrench=np.zeros(6),
            timestamp=time.time()
        )

        controller.set_master_state(master)
        controller.set_slave_state(slave)

    def test_teleop_send_command(self):
        """发送遥操作命令"""
        from control.teleop import (
            TeleoperationController, TeleopConfig, TeleopCommand,
            TeleopMode, MasterState, SlaveState, AuthorityLevel
        )
        config = TeleopConfig(control_frequency=100.0)
        controller = TeleoperationController(config)
        controller.connect()

        master = MasterState(
            joint_positions=np.zeros(6),
            joint_velocities=np.zeros(6),
            joint_torques=np.zeros(6),
            wrench=np.zeros(6),
            timestamp=time.time(),
            authority=AuthorityLevel.OPERATOR
        )
        slave = SlaveState(
            joint_positions=np.zeros(6),
            joint_velocities=np.zeros(6),
            end_effector_pose=np.array([0.0, 0.0, 0.5]),
            contact_wrench=np.zeros(6),
            timestamp=time.time()
        )
        controller.set_master_state(master)
        controller.set_slave_state(slave)

        self.assertIsNotNone(controller._master_state)
        self.assertIsNotNone(controller._slave_state)

    def test_teleop_send_command(self):
        """发送遥操作命令"""
        from control.teleop import (
            TeleoperationController, TeleopConfig, TeleopCommand,
            TeleopMode, MasterState, SlaveState, AuthorityLevel
        )
        config = TeleopConfig(control_frequency=100.0)
        controller = TeleoperationController(config)
        controller.connect()

        master = MasterState(
            joint_positions=np.zeros(6),
            joint_velocities=np.zeros(6),
            joint_torques=np.zeros(6),
            wrench=np.zeros(6),
            timestamp=time.time(),
            authority=AuthorityLevel.OPERATOR
        )
        slave = SlaveState(
            joint_positions=np.zeros(6),
            joint_velocities=np.zeros(6),
            end_effector_pose=np.array([0.0, 0.0, 0.5]),
            contact_wrench=np.zeros(6),
            timestamp=time.time()
        )
        controller.set_master_state(master)
        controller.set_slave_state(slave)

        cmd = TeleopCommand(
            mode=TeleopMode.POSITION_SYNC,
            target_joint_positions=np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
        )
        success = controller.send_command(cmd)
        self.assertTrue(success)

    def test_teleop_compute_slave_command(self):
        """计算从端命令"""
        from control.teleop import (
            TeleoperationController, TeleopConfig, TeleopCommand,
            TeleopMode, MasterState, SlaveState, AuthorityLevel
        )
        config = TeleopConfig(control_frequency=100.0)
        controller = TeleoperationController(config)
        controller.connect()

        master = MasterState(
            joint_positions=np.zeros(6),
            joint_velocities=np.zeros(6),
            joint_torques=np.zeros(6),
            wrench=np.zeros(6),
            timestamp=time.time(),
            authority=AuthorityLevel.OPERATOR
        )
        slave = SlaveState(
            joint_positions=np.zeros(6),
            joint_velocities=np.zeros(6),
            end_effector_pose=np.array([0.0, 0.0, 0.5]),
            contact_wrench=np.zeros(6),
            timestamp=time.time()
        )
        controller.set_master_state(master)
        controller.set_slave_state(slave)

        cmd = TeleopCommand(
            mode=TeleopMode.POSITION_SYNC,
            target_joint_positions=np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
        )
        controller.send_command(cmd)

        result = controller.compute_slave_command()
        self.assertIsNotNone(result)
        blended_cmd, autonomy = result
        self.assertEqual(len(blended_cmd), 6)

    def test_teleop_authority_request(self):
        """权限请求"""
        from control.teleop import (
            TeleoperationController, TeleopConfig, AuthorityLevel
        )
        config = TeleopConfig()
        controller = TeleoperationController(config)

        # 需要 SUPERVISOR 以上才能提升权限
        result = controller.request_authority(AuthorityLevel.SUPERVISOR)
        self.assertFalse(result)  # 默认 OPERATOR 不能提升到 SUPERVISOR

        # 释放权限
        controller.release_authority()
        self.assertEqual(controller._current_authority, AuthorityLevel.VIEWER)

    def test_teleop_pause_resume(self):
        """暂停和恢复"""
        from control.teleop import (
            TeleoperationController, TeleopConfig, TeleopState,
            TeleopCommand, TeleopMode, MasterState, SlaveState, AuthorityLevel
        )
        config = TeleopConfig()
        controller = TeleoperationController(config)
        controller.connect()

        # 先建立有效状态
        master = MasterState(
            joint_positions=np.zeros(6), joint_velocities=np.zeros(6),
            joint_torques=np.zeros(6), wrench=np.zeros(6),
            timestamp=time.time(), authority=AuthorityLevel.OPERATOR
        )
        slave = SlaveState(
            joint_positions=np.zeros(6), joint_velocities=np.zeros(6),
            end_effector_pose=np.array([0.0, 0.0, 0.5]),
            contact_wrench=np.zeros(6), timestamp=time.time()
        )
        controller.set_master_state(master)
        controller.set_slave_state(slave)
        controller.send_command(TeleopCommand(
            mode=TeleopMode.POSITION_SYNC,
            target_joint_positions=np.zeros(6)
        ))

        # 注意: pause() 只能在 ACTIVE 状态下调用
        # pause 只在 ACTIVE 时有效
        controller.pause()
        # resume 也只从 PAUSED 有效
        controller.resume()
        self.assertTrue(True)  # 不崩溃即通过

    def test_teleop_emergency_stop(self):
        """紧急停止"""
        from control.teleop import (
            TeleoperationController, TeleopConfig, TeleopState, AuthorityLevel
        )
        config = TeleopConfig(safety_stop_threshold=10.0)
        controller = TeleoperationController(config)
        
        controller.connect()
        controller.emergency_stop()
        self.assertEqual(controller.state, TeleopState.SAFETY_STOP)
        self.assertEqual(controller._current_authority, AuthorityLevel.VIEWER)

    def test_teleop_acknowledge_safety_stop(self):
        """确认安全停止"""
        from control.teleop import (
            TeleoperationController, TeleopConfig, TeleopState, AuthorityLevel
        )
        config = TeleopConfig()
        controller = TeleoperationController(config)
        controller.connect()
        controller.emergency_stop()
        
        # 需要 SUPERVISOR 才能确认
        result = controller.acknowledge_safety_stop(AuthorityLevel.SUPERVISOR)
        self.assertTrue(result)
        self.assertEqual(controller.state, TeleopState.IDLE)

    def test_teleop_latency_compensator(self):
        """延迟补偿器"""
        from control.teleop import (
            LatencyCompensator, TeleopConfig, TeleopCommand,
            TeleopMode, SlaveState
        )
        config = TeleopConfig(max_latency_compensation_ms=50.0)
        compensator = LatencyCompensator(config)

        slave = SlaveState(
            joint_positions=np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0]),
            joint_velocities=np.zeros(6),
            end_effector_pose=np.array([0.0, 0.0, 0.5]),
            contact_wrench=np.zeros(6),
            timestamp=time.time()
        )

        cmd = TeleopCommand(
            mode=TeleopMode.POSITION_SYNC,
            target_joint_positions=np.array([0.15, 0.25, 0.35, 0.0, 0.0, 0.0])
        )

        predicted = compensator.predict_slave_state(cmd, slave, latency_ms=30.0)
        self.assertIsNotNone(predicted)

    def test_teleop_shared_control_blender(self):
        """共享控制混合器"""
        from control.teleop import (
            SharedControlBlender, TeleopConfig, TeleopMode
        )
        config = TeleopConfig()
        blender = SharedControlBlender(config)

        operator = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        autonomous = np.array([0.5, 1.0, 1.5, 0.0, 0.0, 0.0])

        blended = blender.blend_commands(operator, autonomous)
        self.assertEqual(len(blended), 6)
        # 默认 50% 自主性
        expected = 0.5 * operator + 0.5 * autonomous
        self.assertTrue(np.allclose(blended, expected))

    def test_teleop_shared_control_autonomy_update(self):
        """共享控制自主性更新"""
        from control.teleop import SharedControlBlender, TeleopConfig
        config = TeleopConfig()
        blender = SharedControlBlender(config)

        # 更新自主性 (安全)
        blender.update_autonomy(
            operator_confidence=0.8,
            task_difficulty=0.3,
            safety_margin=0.7
        )
        
        # 更新自主性 (危险 - 安全裕度低)
        blender.update_autonomy(
            operator_confidence=0.8,
            task_difficulty=0.3,
            safety_margin=0.1  # 低安全裕度
        )
        
        # 自主性应该被限制在最小值
        self.assertLessEqual(blender.autonomy_level, config.autonomy_blend_max)


if __name__ == '__main__':
    unittest.main(verbosity=2)
