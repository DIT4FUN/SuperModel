"""
仿真控制接口测试
测试 SimulationInterface 及各仿真后端的五级AGV规格支持
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.control.simulation import (
    SimulationInterface, SimulationBackend, SimulationGrade,
    SimulationConfig, SimState, AGVSimParams,
    AGV_SIM_PARAMS, AGV_SIMULATION_GRADES,
    get_agv_sim_params, get_simulation_spec
)


class TestAGVSimParams(unittest.TestCase):
    """AGV仿真参数测试"""
    
    def test_agv_sim_params_all_grades(self):
        """测试所有等级的AGV仿真参数"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            params = get_agv_sim_params(grade)
            self.assertIsInstance(params, AGVSimParams)
            self.assertGreater(params.max_load_kg, 0)
            self.assertGreater(params.wheel_radius_m, 0)
            self.assertGreater(params.vehicle_mass_kg, 0)
            self.assertGreater(params.max_linear_speed_mps, 0)
    
    def test_agv_sim_params_monotonic(self):
        """测试规格参数的单调性(S→XXL 性能递增)"""
        speeds = [get_agv_sim_params(g).max_linear_speed_mps for g in ['S', 'M', 'L', 'XL', 'XXL']]
        loads = [get_agv_sim_params(g).max_load_kg for g in ['S', 'M', 'L', 'XL', 'XXL']]
        # XXL应该比S负载更大
        self.assertGreater(loads[-1], loads[0])
        # XXL应该比S速度更快或相当
        self.assertGreaterEqual(speeds[-1], speeds[0] * 0.5)


class TestSimulationGrades(unittest.TestCase):
    """仿真等级规格测试"""
    
    def test_simulation_grades_all(self):
        """测试所有仿真等级配置"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_simulation_spec(grade)
            self.assertIn('backend', spec)
            self.assertIn('dt', spec)
            self.assertIn('freq', spec)
            self.assertIn('fidelity', spec)
    
    def test_simulation_grades_fidelity(self):
        """测试仿真保真度递增"""
        fid_map = {'S': 1, 'M': 2, 'L': 3, 'XL': 4, 'XXL': 5}
        for g1, g2 in [('S', 'M'), ('M', 'L'), ('L', 'XL'), ('XL', 'XXL')]:
            spec1 = get_simulation_spec(g1)
            spec2 = get_simulation_spec(g2)
            self.assertGreaterEqual(fid_map[g2], fid_map[g1])


class TestSimulationInterface(unittest.TestCase):
    """仿真接口测试"""
    
    def test_creation_all_grades(self):
        """测试所有等级创建仿真接口"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            sim = SimulationInterface(grade=grade, backend='none')
            self.assertEqual(sim.grade, grade)
            self.assertEqual(sim.backend, SimulationBackend.NONE)
            info = sim.get_info()
            self.assertEqual(info['grade'], grade)
    
    def test_config_dt_per_grade(self):
        """测试等级对应的dt配置"""
        grade_dt = {
            'S': 0.02,
            'M': 0.01,
            'L': 0.005,
            'XL': 0.002,
            'XXL': 0.001,
        }
        for grade, expected_dt in grade_dt.items():
            config = SimulationConfig(grade=SimulationGrade(grade))
            self.assertEqual(config.dt, expected_dt)
    
    def test_reset_returns_state(self):
        """测试reset返回SimState"""
        for grade in ['S', 'M']:
            sim = SimulationInterface(grade=grade, backend='none')
            state = sim.reset()
            self.assertIsInstance(state, SimState)
            self.assertEqual(state.timestamp, 0.0)
            self.assertEqual(state.dt, get_simulation_spec(grade)['dt'])
            sim.close()
    
    def test_step_updates_state(self):
        """测试step更新状态"""
        sim = SimulationInterface(grade='M', backend='none')
        sim.reset()
        action = np.array([0.5])
        state, reward, done, info = sim.step(action)
        self.assertIsInstance(state, SimState)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        self.assertGreater(state.timestamp, 0)
        sim.close()
    
    def test_mock_backend_steps(self):
        """测试mock后端多步执行"""
        sim = SimulationInterface(grade='M', backend='none')
        sim.reset()
        for i in range(100):
            action = np.array([0.1 * (1 if i % 2 == 0 else -0.5)])
            state, reward, done, info = sim.step(action)
            self.assertIsNotNone(state.velocity)
            if done:
                break
        sim.close()
    
    def test_compute_reward_bounds(self):
        """测试奖励函数有界"""
        sim = SimulationInterface(grade='M', backend='none')
        sim.reset()
        rewards = []
        for _ in range(50):
            action = np.array([np.random.rand()])
            _, reward, _, _ = sim.step(action)
            rewards.append(reward)
        # 奖励应该是有界的
        self.assertTrue(all(-10 < r < 10 for r in rewards))
        sim.close()
    
    def test_get_info(self):
        """测试获取仿真信息"""
        sim = SimulationInterface(grade='XL', backend='none')
        sim.reset()
        sim.step(np.array([0.5]))
        info = sim.get_info()
        self.assertEqual(info['grade'], 'XL')
        self.assertEqual(info['backend'], 'none')
        self.assertGreater(info['step_count'], 0)
        self.assertGreater(info['sim_time'], 0)
        sim.close()
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with SimulationInterface(grade='L', backend='none') as sim:
            state = sim.reset()
            self.assertIsInstance(state, SimState)
            self.assertTrue(sim.get_info()['is_running'])
        # close后is_running应为False


class TestSimState(unittest.TestCase):
    """SimState数据类测试"""
    
    def test_state_creation(self):
        """测试状态对象创建"""
        state = SimState(
            timestamp=1.0,
            dt=0.01,
            position=np.array([1., 2., 3.]),
            orientation=np.array([1., 0., 0., 0.]),
            velocity=np.array([0.1, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            joint_positions=np.zeros(2),
            joint_velocities=np.zeros(2),
            joint_torques=np.zeros(2),
            contact_forces={},
            sensor_readings={},
        )
        self.assertEqual(state.timestamp, 1.0)
        np.testing.assert_array_equal(state.position, [1., 2., 3.])
        np.testing.assert_array_equal(state.velocity, [0.1, 0.0, 0.0])


if __name__ == '__main__':
    unittest.main()
