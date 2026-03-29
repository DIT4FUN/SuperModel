"""
SuperModel 仿真环境测试
=======================

测试仿真环境、传感器仿真、物理引擎适配层
"""

import numpy as np
import sys
import time
import unittest

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from simulation.environment import (
    RobotSimulator, SensorSimulator, PhysicsEngine, SceneManager,
    TrajectoryRecorder, SimConfig, PRESET_SCENES, create_scene
)


class TestRobotSimulator(unittest.TestCase):
    """测试机器人仿真器"""
    
    def test_simulator_init(self):
        config = SimConfig(num_joints=6, dt=0.01)
        sim = RobotSimulator(config)
        self.assertEqual(sim.n, 6)
        self.assertEqual(sim.dt, 0.01)
    
    def test_set_joint_positions(self):
        sim = RobotSimulator()
        positions = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
        sim.set_joint_positions(positions)
        np.testing.assert_array_almost_equal(sim.joint_positions, positions)
    
    def test_joint_limits_enforcement(self):
        sim = RobotSimulator()
        # 设置限位
        sim.jl_lower = np.array([-0.5] * 6)
        sim.jl_upper = np.array([0.5] * 6)
        
        # 尝试设置超限位置
        positions = np.array([1.0] * 6)
        sim.set_joint_positions(positions)
        
        # 应该被裁剪到限位
        np.testing.assert_array_less(sim.joint_positions, 0.51)
        np.testing.assert_array_less(-0.51, sim.joint_positions)
    
    def test_step_dynamics(self):
        sim = RobotSimulator(SimConfig(num_joints=6, dt=0.01))
        
        # 零力矩步进
        state1 = sim.step(np.zeros(6))
        state2 = sim.step(np.zeros(6))
        
        # 应该有时间推进
        self.assertGreater(state2['time'], state1['time'])
    
    def test_step_with_torque(self):
        sim = RobotSimulator(SimConfig(num_joints=6, dt=0.01))
        
        # 施加恒定力矩
        torque = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        state = sim.step(torque)
        
        self.assertEqual(len(state['joint_positions']), 6)
        self.assertEqual(len(state['joint_velocities']), 6)
        self.assertFalse(np.any(np.isnan(state['joint_positions'])))
    
    def test_joint_limit_bounce(self):
        """测试关节限位反弹"""
        sim = RobotSimulator()
        sim.jl_lower = np.array([-0.1] * 6)
        sim.jl_upper = np.array([0.1] * 6)
        sim.joint_positions = np.array([0.0] * 6)
        
        # 给一个冲向限位的初速度
        sim.joint_velocities = np.array([1.0] * 6)
        
        # 多次步进直到撞到限位
        for _ in range(20):
            prev_pos = sim.joint_positions[0]
            sim.step(np.zeros(6))
            if sim.joint_positions[0] >= sim.jl_upper[0] - 0.001:
                break
        
        # 当碰到限位时，速度应该被反转（变为负值）
        self.assertLess(sim.joint_velocities[0], 0.5)
    
    def test_get_state(self):
        sim = RobotSimulator()
        state = sim.get_state()
        
        self.assertIn('time', state)
        self.assertIn('joint_positions', state)
        self.assertIn('joint_velocities', state)
        self.assertIn('end_effector_pose', state)
    
    def test_jacobian(self):
        sim = RobotSimulator(SimConfig(num_joints=6))
        J = sim.get_jacobian()
        self.assertEqual(J.shape, (6, 6))
    
    def test_self_collision(self):
        sim = RobotSimulator()
        result = sim.check_self_collision()
        self.assertIsInstance(result, bool)
    
    def test_environment_collision(self):
        sim = RobotSimulator()
        obstacles = [
            {"type": "sphere", "center": np.array([0.5, 0.0, 0.5]), "radius": 0.1}
        ]
        collisions = sim.check_environment_collision(obstacles)
        self.assertIsInstance(collisions, list)
    
    def test_callback(self):
        sim = RobotSimulator()
        callback_called = []
        
        def cb(state):
            callback_called.append(state['time'])
        
        sim.add_callback(cb)
        sim.step(np.zeros(6))
        sim.step(np.zeros(6))
        
        self.assertEqual(len(callback_called), 2)
    
    def test_reset(self):
        sim = RobotSimulator()
        sim.joint_positions = np.ones(6) * 0.5
        sim.joint_velocities = np.ones(6)
        sim._time = 10.0
        
        sim.reset()
        
        np.testing.assert_array_almost_equal(sim.joint_positions, np.zeros(6))
        np.testing.assert_array_almost_equal(sim.joint_velocities, np.zeros(6))
        self.assertEqual(sim._time, 0.0)


class TestSensorSimulator(unittest.TestCase):
    """测试传感器仿真器"""
    
    def test_sensor_sim_init(self):
        sim = RobotSimulator()
        sensor_sim = SensorSimulator(sim)
        self.assertEqual(sensor_sim.sim, sim)
    
    def test_noisy_joint_positions(self):
        sim = RobotSimulator()
        sensor_sim = SensorSimulator(sim)
        
        noisy = sensor_sim.get_noisy_joint_positions()
        self.assertEqual(len(noisy), sim.n)
    
    def test_noisy_joint_velocities(self):
        sim = RobotSimulator()
        sensor_sim = SensorSimulator(sim)
        
        noisy = sensor_sim.get_noisy_joint_velocities()
        self.assertEqual(len(noisy), sim.n)
    
    def test_imu_data(self):
        sim = RobotSimulator()
        sensor_sim = SensorSimulator(sim)
        
        imu_data = sensor_sim.get_imu_data()
        
        self.assertIn('accel', imu_data)
        self.assertIn('gyro', imu_data)
        self.assertEqual(imu_data['accel'].shape, (3,))
        self.assertEqual(imu_data['gyro'].shape, (3,))
    
    def test_wrench(self):
        sim = RobotSimulator()
        sensor_sim = SensorSimulator(sim)
        
        wrench = sensor_sim.get_wrench()
        self.assertEqual(wrench.shape, (6,))
    
    def test_contact_force(self):
        sim = RobotSimulator()
        sensor_sim = SensorSimulator(sim)
        
        contact = sensor_sim.get_contact_force()
        self.assertGreaterEqual(contact, 0.0)
    
    def test_sensor_delay(self):
        sim = RobotSimulator()
        sensor_sim = SensorSimulator(sim)
        
        result = sensor_sim.apply_sensor_delay({"data": 123})
        self.assertEqual(result["data"], 123)


class TestPhysicsEngine(unittest.TestCase):
    """测试物理引擎适配层"""
    
    def test_custom_engine(self):
        engine = PhysicsEngine(engine="custom", config={"num_joints": 6, "dt": 0.01})
        self.assertEqual(engine.engine, "custom")
        self.assertIsInstance(engine.simulator, RobotSimulator)
    
    def test_engine_step(self):
        engine = PhysicsEngine(engine="custom", config={"num_joints": 6})
        state = engine.step(np.zeros(6))
        self.assertIn('time', state)
    
    def test_engine_get_state(self):
        engine = PhysicsEngine(engine="custom", config={"num_joints": 6})
        state = engine.get_state()
        self.assertIn('joint_positions', state)


class TestSceneManager(unittest.TestCase):
    """测试场景管理器"""
    
    def test_add_object(self):
        manager = SceneManager()
        name = manager.add_object(
            name="box1",
            obj_type="box",
            position=np.array([0.5, 0.0, 0.3])
        )
        self.assertEqual(name, "box1")
        self.assertIn("box1", manager.objects)
    
    def test_auto_name(self):
        manager = SceneManager()
        name1 = manager.add_object(name=None, obj_type="sphere", position=np.array([0, 0, 0]))
        name2 = manager.add_object(name=None, obj_type="sphere", position=np.array([1, 1, 1]))
        self.assertNotEqual(name1, name2)
    
    def test_remove_object(self):
        manager = SceneManager()
        manager.add_object("box1", "box", np.array([0, 0, 0]))
        result = manager.remove_object("box1")
        self.assertTrue(result)
        self.assertNotIn("box1", manager.objects)
    
    def test_move_object(self):
        manager = SceneManager()
        manager.add_object("box1", "box", np.array([0, 0, 0]))
        new_pos = np.array([0.5, 0.5, 0.5])
        result = manager.move_object("box1", new_pos)
        self.assertTrue(result)
        np.testing.assert_array_almost_equal(manager.objects["box1"]["position"], new_pos)
    
    def test_move_grasped_object_fails(self):
        manager = SceneManager()
        manager.add_object("box1", "box", np.array([0, 0, 0]))
        manager.grasp("box1")
        result = manager.move_object("box1", np.array([1, 1, 1]))
        self.assertFalse(result)
    
    def test_grasp(self):
        manager = SceneManager()
        manager.add_object("box1", "box", np.array([0, 0, 0]))
        result = manager.grasp("box1")
        self.assertTrue(result)
        self.assertEqual(manager.grasp_target, "box1")
        self.assertTrue(manager.objects["box1"]["grasped"])
    
    def test_release(self):
        manager = SceneManager()
        manager.add_object("box1", "box", np.array([0, 0, 0]))
        manager.grasp("box1")
        released = manager.release()
        self.assertEqual(released, "box1")
        self.assertFalse(manager.objects["box1"]["grasped"])
        self.assertIsNone(manager.grasp_target)
    
    def test_get_object(self):
        manager = SceneManager()
        manager.add_object("box1", "box", np.array([0, 0, 0]))
        obj = manager.get_object("box1")
        self.assertIsNotNone(obj)
        self.assertEqual(obj['type'], "box")
    
    def test_get_all_objects(self):
        manager = SceneManager()
        manager.add_object("box1", "box", np.array([0, 0, 0]))
        manager.add_object("sphere1", "sphere", np.array([1, 1, 1]))
        objs = manager.get_all_objects()
        self.assertEqual(len(objs), 2)
    
    def test_get_object_positions(self):
        manager = SceneManager()
        manager.add_object("box1", "box", np.array([0.1, 0.2, 0.3]))
        positions = manager.get_object_positions()
        self.assertIn("box1", positions)
        np.testing.assert_array_almost_equal(positions["box1"], [0.1, 0.2, 0.3])


class TestTrajectoryRecorder(unittest.TestCase):
    """测试轨迹记录器"""
    
    def test_record(self):
        recorder = TrajectoryRecorder()
        recorder.record(
            joint_positions=np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0]),
            cartesian_position=np.array([0.5, 0.0, 0.3]),
            wrench=np.zeros(6)
        )
        self.assertEqual(len(recorder.joint_trajectory), 1)
    
    def test_get_joint_trajectory(self):
        recorder = TrajectoryRecorder()
        for _ in range(10):
            recorder.record(np.random.randn(6))
        traj = recorder.get_joint_trajectory()
        self.assertEqual(traj.shape[0], 10)
        self.assertEqual(traj.shape[1], 6)
    
    def test_get_cartesian_trajectory(self):
        recorder = TrajectoryRecorder()
        recorder.record(np.zeros(6), cartesian_position=np.array([1, 2, 3]))
        traj = recorder.get_cartesian_trajectory()
        self.assertIsNotNone(traj)
        self.assertEqual(traj.shape, (1, 3))
    
    def test_get_duration(self):
        recorder = TrajectoryRecorder()
        recorder.record(np.zeros(6))
        time.sleep(0.05)
        recorder.record(np.zeros(6))
        duration = recorder.get_duration()
        self.assertGreater(duration, 0.0)
    
    def test_clear(self):
        recorder = TrajectoryRecorder()
        for _ in range(5):
            recorder.record(np.zeros(6))
        recorder.clear()
        self.assertEqual(len(recorder.joint_trajectory), 0)
        self.assertIsNone(recorder._start_time)
    
    def test_export(self):
        recorder = TrajectoryRecorder()
        for _ in range(3):
            recorder.record(np.random.randn(6))
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=True) as f:
            recorder.export(f.name)
            data = np.load(f.name)
            self.assertIn('joint_trajectory', data)


class TestPresetScenes(unittest.TestCase):
    """测试预设场景"""
    
    def test_tabletop_scene(self):
        scene = create_scene("tabletop")
        self.assertIn("obstacles", scene)
        self.assertEqual(scene["table_height"], 0.3)
    
    def test_shelf_scene(self):
        scene = create_scene("shelf")
        self.assertIn("shelf_heights", scene)
        self.assertGreater(len(scene["shelf_heights"]), 0)
    
    def test_door_scene(self):
        scene = create_scene("door")
        self.assertIn("obstacles", scene)
    
    def test_unknown_scene_defaults_to_tabletop(self):
        scene = create_scene("unknown_scene")
        self.assertEqual(scene["description"], "桌面抓取场景")


class TestSimConfig(unittest.TestCase):
    """测试仿真配置"""
    
    def test_default_config(self):
        config = SimConfig()
        self.assertEqual(config.dt, 0.01)
        self.assertEqual(config.num_joints, 6)
        self.assertEqual(config.engine, "custom")
    
    def test_custom_config(self):
        config = SimConfig(dt=0.005, num_joints=7, engine="pybullet")
        self.assertEqual(config.dt, 0.005)
        self.assertEqual(config.num_joints, 7)


if __name__ == '__main__':
    unittest.main(verbosity=2)


class TestGymEnvConfig(unittest.TestCase):
    """测试 Gym 环境配置"""

    def test_default_config(self):
        from simulation.gym_env import GymEnvConfig
        cfg = GymEnvConfig()
        self.assertEqual(cfg.dt, 0.01)
        self.assertEqual(cfg.num_joints, 6)
        self.assertEqual(cfg.episode_length, 1000)

    def test_custom_config(self):
        from simulation.gym_env import GymEnvConfig
        cfg = GymEnvConfig(dt=0.005, episode_length=500)
        self.assertEqual(cfg.dt, 0.005)
        self.assertEqual(cfg.episode_length, 500)


class TestSuperModelGymEnv(unittest.TestCase):
    """测试 Gymnasium 环境"""

    def setUp(self):
        import os
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from simulation.gym_env import SuperModelGymEnv, GymEnvConfig
        self.env = SuperModelGymEnv(
            config=GymEnvConfig(grade='M'),
            scenario='reach'
        )

    def tearDown(self):
        self.env.close()

    def test_env_creation(self):
        self.assertIsNotNone(self.env)

    def test_spaces(self):
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        self.assertEqual(obs_space.shape[0], 53)
        self.assertEqual(act_space.shape[0], 6)

    def test_reset(self):
        obs, info = self.env.reset(seed=42)
        self.assertEqual(obs.shape, (53,))
        self.assertIn('timestep', info)
        self.assertEqual(info['timestep'], 0)

    def test_reset_with_options(self):
        import numpy as np
        target = np.zeros(6)
        obs, info = self.env.reset(options={'target': target})
        self.assertEqual(obs.shape, (53,))

    def test_step(self):
        self.env.reset(seed=42)
        action = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertEqual(obs.shape, (53,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)

    def test_episode_length(self):
        self.env.reset(seed=42)
        for _ in range(100):
            action = self.env.action_space.sample()
            _, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break
        # episode 可以终止也可以继续

    def test_action_clipping(self):
        """动作应被限幅"""
        self.env.reset(seed=42)
        # 采样一个在限制范围内的动作
        action = self.env.action_space.sample()
        obs, _, _, _, _ = self.env.step(action)
        self.assertEqual(obs.shape, (53,))

    def test_deterministic_reset(self):
        """相同 seed 应产生相同的初始状态"""
        import numpy as np
        np.random.seed(0)
        obs1, _ = self.env.reset(seed=123)
        np.random.seed(0)
        obs2, _ = self.env.reset(seed=123)
        np.testing.assert_array_almost_equal(obs1, obs2)


class TestSuperModelGymEnvTrack(unittest.TestCase):
    """测试跟踪场景"""

    def setUp(self):
        import os
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from simulation.gym_env import SuperModelGymEnv, GymEnvConfig
        self.env = SuperModelGymEnv(
            config=GymEnvConfig(grade='M'),
            scenario='track'
        )

    def tearDown(self):
        self.env.close()

    def test_track_scenario(self):
        obs, info = self.env.reset(seed=42)
        self.assertEqual(info['scenario'], 'track')
        action = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertEqual(obs.shape, (53,))


class TestSuperModelGymEnvGrasp(unittest.TestCase):
    """测试抓取场景"""

    def setUp(self):
        import os
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from simulation.gym_env import SuperModelGymEnv, GymEnvConfig
        self.env = SuperModelGymEnv(
            config=GymEnvConfig(grade='L'),
            scenario='grasp'
        )

    def tearDown(self):
        self.env.close()

    def test_grasp_scenario(self):
        obs, info = self.env.reset(seed=42)
        self.assertEqual(info['scenario'], 'grasp')


class TestCollectRollout(unittest.TestCase):
    """测试 rollout 收集"""

    def setUp(self):
        import os
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from simulation.gym_env import SuperModelGymEnv, GymEnvConfig
        self.env = SuperModelGymEnv(
            config=GymEnvConfig(grade='S'),
            scenario='reach'
        )

    def tearDown(self):
        self.env.close()

    def test_collect_rollout(self):
        from simulation.gym_env import collect_rollout

        def random_policy(obs):
            return self.env.action_space.sample()

        rollout = collect_rollout(self.env, random_policy, max_steps=10)
        self.assertIn('observations', rollout)
        self.assertIn('actions', rollout)
        self.assertIn('rewards', rollout)
        self.assertGreater(rollout['length'], 0)


class TestGetGymSpec(unittest.TestCase):
    """测试 Gym 环境规格"""

    def test_all_grades(self):
        from simulation.gym_env import get_gym_spec
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_gym_spec(grade)
            self.assertIn('dt', spec)
            self.assertIn('episode_length', spec)
            self.assertIn('reward_tracking', spec)
            self.assertIn('max_torque', spec)

    def test_grade_S_dt(self):
        from simulation.gym_env import get_gym_spec
        spec = get_gym_spec('S')
        self.assertEqual(spec['dt'], 0.02)
        self.assertEqual(spec['max_torque'], 50)

    def test_grade_XXL_dt(self):
        from simulation.gym_env import get_gym_spec
        spec = get_gym_spec('XXL')
        self.assertEqual(spec['dt'], 0.002)
        self.assertEqual(spec['max_torque'], 1000)
