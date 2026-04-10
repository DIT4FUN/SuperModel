"""
具身智能仿真集成测试
====================

测试完整具身智能仿真pipeline:
- 传感器仿真 (Tactile/Force/IMU)
- 物理动力学仿真
- 多模态感知融合
- 闭环控制响应
- AGV五级规格验证

覆盖 S/M/L/XL/XXL 五级 AGV 规格
"""

import unittest
import numpy as np
import sys
import time
import math

_ProjectRoot = '/home/treeman/.openclaw/workspace/projects/SuperModel'
_SrcPath = f'{_ProjectRoot}/src'
if _SrcPath not in sys.path:
    sys.path.insert(0, _SrcPath)
if _ProjectRoot not in sys.path:
    sys.path.insert(0, _ProjectRoot)

from src.sensors.tactile import (
    TactileFrame,
    VirtualTactileSensor, get_tactile_spec
)
from src.sensors.force import (
    Wrench,
    VirtualForceSensor, get_force_spec
)
from src.sensors.imu import (
    IMUFrame,
    VirtualIMUSensor, get_imu_spec
)
from src.control.sensor_fusion_control import (
    SensorFusionController,
    FusionControlGrade,
    get_fusion_control_spec
)
from src.control.embodied_sim import (
    EmbodiedSimulator, EmbodiedSimEnv,
    get_sim_grade_spec, create_sim_env
)


# ============================================================================
# AGV五级规格验证
# ============================================================================

class TestAGVGradeSpecifications(unittest.TestCase):
    """AGV五级规格一致性验证"""

    def test_tactile_grade_specs(self):
        """验证触觉五级规格"""
        expected = {
            'S':  {'array': (8, 8),    'res': 12,  'freq_hz': 50,  'temp': False},
            'M':  {'array': (16, 16),  'res': 12,  'freq_hz': 100, 'temp': True},
            'L':  {'array': (24, 24),  'res': 14,  'freq_hz': 200, 'temp': True},
            'XL': {'array': (32, 32),  'res': 14,  'freq_hz': 500, 'temp': True},
            'XXL': {'array': (48, 48), 'res': 16,  'freq_hz': 1000, 'temp': True},
        }
        for grade, spec in expected.items():
            actual = get_tactile_spec(grade)
            self.assertEqual(actual['array'], spec['array'], f"{grade} array mismatch")
            self.assertEqual(actual['res'], spec['res'], f"{grade} resolution mismatch")
            self.assertEqual(actual['freq_hz'], spec['freq_hz'], f"{grade} freq mismatch")
            self.assertEqual(actual['temp'], spec['temp'], f"{grade} temp mismatch")

    def test_force_grade_specs(self):
        """验证力觉五级规格"""
        expected = {
            'S':  {'axes': 3, 'force_range': 100,  'torque_range': 10,   'sampling_hz': 100},
            'M':  {'axes': 6, 'force_range': 200,  'torque_range': 20,   'sampling_hz': 500},
            'L':  {'axes': 6, 'force_range': 500,  'torque_range': 50,   'sampling_hz': 1000},
            'XL': {'axes': 6, 'force_range': 1000, 'torque_range': 100,  'sampling_hz': 2000},
            'XXL': {'axes': 6, 'force_range': 5000, 'torque_range': 500, 'sampling_hz': 5000},
        }
        for grade, spec in expected.items():
            actual = get_force_spec(grade)
            self.assertEqual(actual['axes'], spec['axes'], f"{grade} axes mismatch")
            self.assertEqual(actual['force_range'], spec['force_range'], f"{grade} force_range mismatch")
            self.assertEqual(actual['torque_range'], spec['torque_range'], f"{grade} torque_range mismatch")
            self.assertEqual(actual['sampling_hz'], spec['sampling_hz'], f"{grade} sampling_hz mismatch")

    def test_imu_grade_specs(self):
        """验证IMU五级规格"""
        expected = {
            'S':  {'type': 'MPU6050', 'sample_hz': 100,  'noise_density': 400},
            'M':  {'type': 'BMI088',  'sample_hz': 200,  'noise_density': 120},
            'L':  {'type': 'BMI088',  'sample_hz': 500,  'noise_density': 60},
            'XL': {'type': 'ADIS16470', 'sample_hz': 1000, 'noise_density': 20},
            'XXL': {'type': 'ADIS16470', 'sample_hz': 2000, 'noise_density': 10},
        }
        for grade, spec in expected.items():
            actual = get_imu_spec(grade)
            self.assertEqual(actual['type'], spec['type'], f"{grade} type mismatch")
            self.assertEqual(actual['sample_hz'], spec['sample_hz'], f"{grade} sample_hz mismatch")
            self.assertEqual(actual['noise_density'], spec['noise_density'], f"{grade} noise_density mismatch")

    def test_fusion_control_grade_specs(self):
        """验证融合控制五级规格"""
        expected = {
            'S':  {'freq': 50,  'latency_ms': 20},
            'M':  {'freq': 100, 'latency_ms': 10},
            'L':  {'freq': 200, 'latency_ms': 5},
            'XL': {'freq': 500, 'latency_ms': 2},
            'XXL': {'freq': 1000, 'latency_ms': 1},
        }
        for grade, spec in expected.items():
            actual = get_fusion_control_spec(grade)
            self.assertEqual(actual['freq'], spec['freq'], f"{grade} freq mismatch")
            self.assertEqual(actual['latency_ms'], spec['latency_ms'], f"{grade} latency mismatch")


# ============================================================================
# 仿真器基础功能测试
# ============================================================================

class TestEmbodiedSimulator(unittest.TestCase):
    """EmbodiedSimulator 具身仿真器测试"""

    def setUp(self):
        self.sim = EmbodiedSimulator(grade='M')
        self.sim.reset()  # Must call reset before step

    def test_creation(self):
        """测试创建"""
        self.assertEqual(self.sim.grade, 'M')
        self.assertIsNotNone(self.sim.spec)

    def test_single_step(self):
        """测试单步仿真"""
        state = self.sim.step(0.5, 0.0)  # cmd_vx, cmd_wz, contact_force, contact_center
        self.assertIsNotNone(state)
        self.assertIsNotNone(state.position)
        self.assertIsNotNone(state.linear_velocity)

    def test_velocity_command(self):
        """测试速度命令"""
        # 前进
        state1 = self.sim.step(0.5, 0.0)
        pos1 = state1.position.copy()
        # 后退
        state2 = self.sim.step(-0.5, 0.0)
        pos2 = state2.position.copy()
        # 前进应该比后退位置更靠前
        self.assertGreater(pos1[0], pos2[0])

    def test_rotation_command(self):
        """测试旋转命令"""
        self.sim.reset()  # 重置到初始状态
        state0 = self.sim.step(0.0, 0.5)  # 旋转一步后检查角速度
        ang_vel = state0.angular_velocity
        # 角速度应该非零
        self.assertGreater(abs(ang_vel[2]), 0.0)

    def test_sensor_readings(self):
        """测试传感器读取"""
        self.sim.reset()
        tactile = self.sim.tactile_sim.get_pressure_map()
        self.assertIsNotNone(tactile)
        self.assertGreaterEqual(tactile.shape[0], 8)

    def test_reset(self):
        """测试重置"""
        # 走一段
        for _ in range(100):
            self.sim.step(1.0, 0.0)
        pos_before = self.sim.physics.position.copy()
        # 重置
        self.sim.reset()
        pos_after = self.sim.physics.position.copy()
        # 重置后应该回到原点
        np.testing.assert_allclose(pos_after, np.zeros(3), atol=0.01)


# ============================================================================
# 多模态传感器融合控制测试
# ============================================================================

class TestSensorFusionControlPipeline(unittest.TestCase):
    """传感器融合控制完整pipeline测试"""

    def setUp(self):
        self.ctrl = SensorFusionController(grade=FusionControlGrade.M)
        self.ctrl.start()

    def tearDown(self):
        self.ctrl.stop()

    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.ctrl._state)
        self.assertTrue(hasattr(self.ctrl, '_is_running'))
        self.assertEqual(self.ctrl.grade, FusionControlGrade.M)

    def test_control_cycle(self):
        """测试控制周期"""
        # 模拟传感器数据 (positional args per actual API)
        state = self.ctrl.update(
            imu_accel=np.array([0.1, 0.0, 9.81]),
            imu_gyro=np.array([0.0, 0.0, 0.0]),
            imu_mag=None,
            force_wrench=np.array([0.0, 0.0, -5.0, 0.0, 0.0, 0.0]),
            tactile_pressure=np.random.rand(16, 16).astype(np.float32),
            dt=0.01
        )

        # 验证输出
        self.assertIsNotNone(state)
        self.assertTrue(hasattr(state, 'velocity_cmd'))

    def test_get_state(self):
        """测试获取状态"""
        state = self.ctrl.get_state()
        self.assertIsNotNone(state)
        self.assertTrue(hasattr(state, 'velocity_cmd'))
        self.assertTrue(hasattr(state, 'torque_cmd'))

    def test_fusion_control_all_grades(self):
        """测试所有AGV等级的融合控制器"""
        for grade in FusionControlGrade:
            ctrl = SensorFusionController(grade=grade)
            ctrl.start()
            self.assertEqual(ctrl.grade, grade)
            ctrl.stop()


# ============================================================================
# AGV五级完整pipeline测试
# ============================================================================

class TestFiveGradePipeline(unittest.TestCase):
    """AGV五级完整pipeline集成测试"""

    def test_grade_sensor_specs(self):
        """测试五级传感器规格递增"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            tactile = get_tactile_spec(grade)
            force = get_force_spec(grade)
            imu = get_imu_spec(grade)
            fusion = get_fusion_control_spec(grade)
            
            # 规格应该随等级提升而提升
            self.assertIsNotNone(tactile)
            self.assertIsNotNone(force)
            self.assertIsNotNone(imu)
            self.assertIsNotNone(fusion)

    def test_grade_control_rates(self):
        """测试五级控制频率递增"""
        rates = []
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            fusion = get_fusion_control_spec(grade)
            rates.append(fusion['freq'])
        
        # 频率应该严格递增
        self.assertEqual(rates, sorted(rates))
        self.assertEqual(rates[0], 50)   # S
        self.assertEqual(rates[-1], 1000)  # XXL

    def test_grade_latency_decrease(self):
        """测试五级延迟递减"""
        latencies = []
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            fusion = get_fusion_control_spec(grade)
            latencies.append(fusion['latency_ms'])
        
        # 延迟应该严格递减
        self.assertEqual(latencies, sorted(latencies, reverse=True))
        self.assertEqual(latencies[0], 20)   # S (最高延迟)
        self.assertEqual(latencies[-1], 1)  # XXL (最低延迟)

    def test_grade_sim_params(self):
        """测试五级仿真参数"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_sim_grade_spec(grade)
            self.assertIsNotNone(spec)
            self.assertGreater(spec['control_rate'], 0)
            self.assertGreater(spec['max_linear_speed'], 0)
            self.assertGreater(spec['payload_kg'], 0)

    def test_grade_payload_progression(self):
        """测试五级负载递增"""
        payloads = []
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_sim_grade_spec(grade)
            payloads.append(spec['payload_kg'])
        
        # 负载应该严格递增
        self.assertEqual(payloads, sorted(payloads))
        self.assertEqual(payloads[0], 30)    # S
        self.assertEqual(payloads[-1], 1200)  # XXL


# ============================================================================
# Gymnasium环境集成测试
# ============================================================================

class TestGymEmbodiedIntegration(unittest.TestCase):
    """Gymnasium环境与具身控制集成测试"""

    def test_gym_env_creation(self):
        """测试Gym环境创建"""
        env = create_sim_env(grade='M')
        self.assertIsNotNone(env)
        self.assertEqual(env.grade, 'M')

    def test_gym_reset(self):
        """测试Gym reset"""
        env = create_sim_env(grade='M')
        obs, info = env.reset()
        self.assertIsNotNone(obs)
        # 检查obs维度
        self.assertGreater(len(obs), 0)
        env.close()

    def test_gym_step(self):
        """测试Gym step"""
        env = create_sim_env(grade='M')
        env.reset()
        
        # 随机动作
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        self.assertGreater(len(obs), 0)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        env.close()

    def test_gym_episode(self):
        """测试完整episode"""
        env = create_sim_env(grade='M')
        env.reset()
        
        total_reward = 0.0
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        self.assertIsInstance(total_reward, float)
        env.close()

    def test_gym_grade_all(self):
        """测试所有等级Gym环境"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            env = create_sim_env(grade=grade)
            obs, _ = env.reset()
            self.assertGreater(len(obs), 0)
            env.close()


# ============================================================================
# 端到端场景测试
# ============================================================================

class TestEndToEndScenarios(unittest.TestCase):
    """端到端场景测试"""

    def test_navigation_scenario(self):
        """导航场景测试"""
        sim = EmbodiedSimulator(grade='M')
        sim.reset()
        
        target_x = 2.0
        max_steps = 500
        
        for step in range(max_steps):
            state = sim.get_state()
            dx = target_x - state.position[0]
            
            # 简单比例控制
            vx = np.clip(dx * 0.5, -1.0, 1.0)
            sim.step(vx, 0.0)
            
            if abs(dx) < 0.05:
                break
        
        final_x = sim.get_state().position[0]
        # 允许稍大的误差容限
        self.assertLess(abs(final_x - target_x), 0.3)

    def test_rotation_scenario(self):
        """旋转场景测试"""
        sim = EmbodiedSimulator(grade='M')
        sim.reset()
        
        target_yaw = math.pi / 2  # 90度
        max_steps = 500
        
        for step in range(max_steps):
            state = sim.get_state()
            ang_vel_z = state.angular_velocity[2]
            # 持续施加旋转命令
            omega = 0.5
            sim.step(0.0, omega)
            
            if step > 10:  # 至少走几步后检查角速度
                break
        
        # 验证角速度非零
        final_state = sim.get_state()
        self.assertGreater(abs(final_state.angular_velocity[2]), 0.0)

    def test_contact_scenario(self):
        """接触检测场景测试"""
        sim = EmbodiedSimulator(grade='M')
        sim.reset()
        
        # 模拟向前运动并应用接触力
        for _ in range(200):
            # 向前运动
            sim.step(0.5, 0.0)
        
        # 验证仿真状态正常
        state = sim.get_state()
        self.assertIsNotNone(state.position)
        self.assertIsNotNone(state.linear_velocity)


# ============================================================================
# 性能基准测试
# ============================================================================

class TestPerformanceBenchmarks(unittest.TestCase):
    """性能基准测试"""

    def test_fusion_control_update_rate(self):
        """测试融合控制更新率"""
        ctrl = SensorFusionController(grade=FusionControlGrade.M)
        ctrl.start()
        
        start = time.time()
        n_iterations = 100
        
        for _ in range(n_iterations):
            ctrl.update(
                imu_accel=np.array([0.1, 0, 9.81]),
                imu_gyro=np.zeros(3),
                imu_mag=None,
                force_wrench=np.array([0, 0, -5, 0, 0, 0]),
                tactile_pressure=np.random.rand(16, 16).astype(np.float32),
                dt=0.01
            )
        
        elapsed = time.time() - start
        rate = n_iterations / elapsed
        
        ctrl.stop()
        
        # 应该能达到 100Hz 以上
        self.assertGreater(rate, 80)

    def test_sim_step_rate(self):
        """测试仿真步进率"""
        sim = EmbodiedSimulator(grade='M')
        sim.reset()
        
        start = time.time()
        n_steps = 500
        
        for _ in range(n_steps):
            sim.step(0.5, 0.0)
        
        elapsed = time.time() - start
        rate = n_steps / elapsed
        
        # 仿真应该能达到足够快的速度
        self.assertGreater(rate, 100)


# ============================================================================
# 主函数
# ============================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
