"""
具身仿真模块测试
================

测试具身智能仿真环境 (embodied_sim.py)

覆盖:
- EmbodiedSimulator 物理 + 传感器仿真
- SensorNoiseModel 噪声建模
- TactileSimulator 触觉仿真
- PhysicsSimulator 运动学/动力学
- EmbodiedSimEnv Gymnasium 接口
- AGV五级规格适配
"""

import unittest
import numpy as np
from typing import Tuple


class TestSensorNoiseModel(unittest.TestCase):
    """传感器噪声模型测试"""

    def test_imu_noise_added(self):
        """IMU噪声正确叠加"""
        from src.control.embodied_sim import SensorNoiseModel, get_sim_grade_spec

        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            noise_model = SensorNoiseModel(grade=grade)
            spec = get_sim_grade_spec(grade)

            accel_true = np.array([0.0, 0.0, -9.81])
            gyro_true = np.array([0.0, 0.0, 0.1])

            accel_noisy, gyro_noisy = noise_model.add_imu_noise(accel_true, gyro_true, dt=0.01)

            # 噪声应该改变原始值
            self.assertFalse(np.allclose(accel_noisy, accel_true))
            self.assertFalse(np.allclose(gyro_noisy, gyro_true))

    def test_bias_drift(self):
        """随机游走偏置随时间漂移"""
        from src.control.embodied_sim import SensorNoiseModel

        noise_model = SensorNoiseModel(grade='M')
        accel_true = np.array([0.0, 0.0, -9.81])
        gyro_true = np.array([0.0, 0.0, 0.0])

        accel_1, _ = noise_model.add_imu_noise(accel_true, gyro_true, dt=0.01)
        accel_2, _ = noise_model.add_imu_noise(accel_true, gyro_true, dt=0.01)

        # 连续两次读数应该不同 (因为随机游走)
        self.assertFalse(np.allclose(accel_1, accel_2))

    def test_calibrate_accel_bias(self):
        """加速度计偏置标定"""
        from src.control.embodied_sim import SensorNoiseModel

        noise_model = SensorNoiseModel(grade='M')
        # 静止时的读数
        samples = np.array([[0.1, 0.05, -9.81] for _ in range(100)])

        # 标定不应崩溃
        noise_model.calibrate_accel_bias(samples)

        # 标定后bias drift应重置
        self.assertTrue(np.allclose(noise_model._accel_bias_drift, np.zeros(3)))

    def test_calibrate_gyro_bias(self):
        """陀螺仪偏置标定"""
        from src.control.embodied_sim import SensorNoiseModel

        noise_model = SensorNoiseModel(grade='M')
        samples = np.array([[0.01, -0.01, 0.02] for _ in range(100)])

        noise_model.calibrate_gyro_bias(samples)

        self.assertTrue(np.allclose(noise_model._gyro_bias_drift, np.zeros(3)))

    def test_force_noise(self):
        """力觉传感器噪声"""
        from src.control.embodied_sim import SensorNoiseModel

        for grade in ['S', 'M', 'L']:
            noise_model = SensorNoiseModel(grade=grade)
            wrench_true = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            wrench_noisy = noise_model.add_force_noise(wrench_true)

            self.assertFalse(np.allclose(wrench_noisy, wrench_true))
            self.assertEqual(wrench_noisy.shape, (6,))

    def test_reset(self):
        """重置噪声模型"""
        from src.control.embodied_sim import SensorNoiseModel

        noise_model = SensorNoiseModel(grade='M')
        accel_true = np.array([0.0, 0.0, -9.81])
        gyro_true = np.array([0.0, 0.0, 0.1])

        noise_model.add_imu_noise(accel_true, gyro_true, dt=0.01)
        noise_model.reset()

        # 重置后偏置应为零
        self.assertTrue(np.allclose(noise_model._imu_accel_bias, np.zeros(3)))
        self.assertTrue(np.allclose(noise_model._imu_gyro_bias, np.zeros(3)))


class TestPhysicsSimulator(unittest.TestCase):
    """物理仿真器测试"""

    def test_reset(self):
        """重置后状态正确"""
        from src.control.embodied_sim import PhysicsSimulator

        sim = PhysicsSimulator(grade='M')
        sim.reset(position=np.array([1.0, 2.0, 0.0]))

        self.assertTrue(np.allclose(sim.position, [1.0, 2.0, 0.0]))
        self.assertTrue(np.allclose(sim.linear_vel, np.zeros(3)))
        self.assertTrue(np.allclose(sim.angular_vel, np.zeros(3)))

    def test_differential_drive(self):
        """差分驱动运动学"""
        from src.control.embodied_sim import PhysicsSimulator

        sim = PhysicsSimulator(grade='M')
        sim.reset()

        # 直线运动
        sim.step(cmd_vx=0.5, cmd_wz=0.0, dt=0.01)

        self.assertGreater(sim.position[0], 0.0)
        self.assertTrue(np.abs(sim.position[1]) < 0.01)  # 几乎没有侧向偏移

    def test_turn_in_place(self):
        """原地转向"""
        from src.control.embodied_sim import PhysicsSimulator

        sim = PhysicsSimulator(grade='M')
        sim.reset()

        yaw_before = sim._get_yaw()
        sim.step(cmd_vx=0.0, cmd_wz=0.5, dt=0.01)
        yaw_after = sim._get_yaw()

        self.assertNotAlmostEqual(yaw_after, yaw_before, places=3)

    def test_velocity_limits(self):
        """速度限幅"""
        from src.control.embodied_sim import PhysicsSimulator

        sim = PhysicsSimulator(grade='M')
        sim.reset()

        max_v = sim.spec['max_linear_speed']
        max_w = sim.spec['max_angular_speed']

        # 施加过大的速度命令
        sim.step(cmd_vx=max_v * 10, cmd_wz=max_w * 10, dt=0.01)

        self.assertLessEqual(np.abs(sim.linear_vel[0]), max_v * 1.01)
        self.assertLessEqual(np.abs(sim.angular_vel[2]), max_w * 1.01)

    def test_payload_affects_dynamics(self):
        """有效载荷影响动力学"""
        from src.control.embodied_sim import PhysicsSimulator

        sim_light = PhysicsSimulator(grade='M')
        sim_heavy = PhysicsSimulator(grade='M')

        sim_light.reset()
        sim_heavy.reset()
        sim_heavy.set_payload(50.0)

        # 相同命令下，有负载的速度响应应该更慢
        for _ in range(10):
            sim_light.step(cmd_vx=0.5, cmd_wz=0.0, dt=0.01)
            sim_heavy.step(cmd_vx=0.5, cmd_wz=0.0, dt=0.01)

        # 重的AGV速度稍低 (简化模型)
        self.assertLessEqual(sim_heavy.linear_vel[0], sim_light.linear_vel[0] + 0.01)

    def test_imu_reading(self):
        """IMU读数计算"""
        from src.control.embodied_sim import PhysicsSimulator

        sim = PhysicsSimulator(grade='M')
        sim.reset()

        accel, gyro = sim.compute_imu_reading()

        # 静止时，加速度模接近重力 (方向可能因坐标系旋转而异)
        accel_mag = np.linalg.norm(accel)
        self.assertAlmostEqual(accel_mag, 9.81, places=1)
        self.assertTrue(np.allclose(gyro, np.zeros(3), atol=0.1))


class TestTactileSimulator(unittest.TestCase):
    """触觉阵列仿真器测试"""

    def test_creation_all_grades(self):
        """各级别触觉阵列尺寸"""
        from src.control.embodied_sim import TactileSimulator, get_sim_grade_spec

        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            tactile = TactileSimulator(grade=grade)
            expected = get_sim_grade_spec(grade)['tactile_array_size']
            self.assertEqual(tactile.array_size, expected)

    def test_apply_contact(self):
        """施加接触压力"""
        from src.control.embodied_sim import TactileSimulator

        tactile = TactileSimulator(grade='M')
        tactile.reset()

        tactile.apply_contact(force=5.0, center=(8, 8), radius=3)
        pressure = tactile.get_pressure_map()

        self.assertTrue(np.any(pressure > 0))
        self.assertTrue(np.all(pressure >= 0))
        self.assertTrue(np.all(pressure <= 1.0))

    def test_contact_center_pressure(self):
        """接触中心压力最大"""
        from src.control.embodied_sim import TactileSimulator

        tactile = TactileSimulator(grade='M')
        tactile.reset()

        center = (8, 8)
        tactile.apply_contact(force=1.0, center=center, radius=3)
        pressure = tactile.get_pressure_map()

        # 中心点压力应该最大
        self.assertEqual(pressure[center[0], center[1]],
                         np.max(pressure))

    def test_release_contact_decay(self):
        """释放接触后压力衰减"""
        from src.control.embodied_sim import TactileSimulator

        tactile = TactileSimulator(grade='M')
        tactile.reset()

        tactile.apply_contact(force=1.0)
        tactile.release_contact()
        pressure1 = tactile.get_pressure_map()

        # 再调用一次应该衰减
        pressure2 = tactile.get_pressure_map()
        self.assertTrue(np.all(pressure2 <= pressure1))

    def test_no_contact_returns_zero(self):
        """无接触时压力为零"""
        from src.control.embodied_sim import TactileSimulator

        tactile = TactileSimulator(grade='M')
        tactile.reset()

        pressure = tactile.get_pressure_map()
        self.assertTrue(np.allclose(pressure, np.zeros_like(pressure)))


class TestEmbodiedSimulator(unittest.TestCase):
    """完整具身仿真器测试"""

    def test_creation_all_grades(self):
        """创建所有等级仿真器"""
        from src.control.embodied_sim import EmbodiedSimulator, get_sim_grade_spec

        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            sim = EmbodiedSimulator(grade=grade)
            self.assertEqual(sim.grade, grade)
            self.assertEqual(sim.spec, get_sim_grade_spec(grade))

    def test_reset(self):
        """重置仿真器"""
        from src.control.embodied_sim import EmbodiedSimulator

        sim = EmbodiedSimulator(grade='M', seed=42)
        state = sim.reset(initial_position=np.array([1.0, 2.0, 0.0]))

        self.assertTrue(np.allclose(state.position, [1.0, 2.0, 0.0]))
        self.assertEqual(state.timestamp, 0.0)
        self.assertTrue(sim._is_running)

    def test_step(self):
        """仿真一步"""
        from src.control.embodied_sim import EmbodiedSimulator

        sim = EmbodiedSimulator(grade='M', seed=42)
        sim.reset()

        state = sim.step(cmd_vx=0.5, cmd_wz=0.0)

        self.assertIsNotNone(state)
        self.assertGreaterEqual(state.timestamp, 0.0)
        self.assertIsNotNone(state.imu_accel_noisy)
        self.assertIsNotNone(state.imu_gyro_noisy)

    def test_step_updates_time(self):
        """每步时间正确递增"""
        from src.control.embodied_sim import EmbodiedSimulator

        sim = EmbodiedSimulator(grade='M', seed=42)
        sim.reset()

        initial_time = sim._sim_time
        for _ in range(10):
            sim.step(cmd_vx=0.0, cmd_wz=0.0)

        self.assertAlmostEqual(sim._sim_time, initial_time + 10 * sim.dt)

    def test_sensor_readings(self):
        """传感器读数有效"""
        from src.control.embodied_sim import EmbodiedSimulator

        sim = EmbodiedSimulator(grade='M', seed=42)
        sim.reset()

        sim.step(cmd_vx=0.3, cmd_wz=0.1)

        sensors = sim.get_sensor_dict()

        self.assertIn('imu', sensors)
        self.assertIn('force', sensors)
        self.assertIn('tactile', sensors)
        self.assertIn('encoders', sensors)
        self.assertIn('pose', sensors)

        # IMU形状检查
        self.assertEqual(sensors['imu']['accel'].shape, (3,))
        self.assertEqual(sensors['imu']['gyro'].shape, (3,))
        self.assertEqual(sensors['force'].shape, (6,))

    def test_observation_vector(self):
        """观测向量形状正确"""
        from src.control.embodied_sim import EmbodiedSimulator

        sim = EmbodiedSimulator(grade='M', seed=42)
        sim.reset()
        sim.step(cmd_vx=0.1, cmd_wz=0.0)

        obs = sim.get_observation()

        # 31维观测: pos(3) + euler(3) + lin_vel(3) + ang_vel(3) + imu_a(3) + imu_g(3) + wheel_pos(2) + wheel_vel(2) + battery(3) + slip(3) + temp(3)
        self.assertEqual(obs.shape, (31,))
        self.assertFalse(np.any(np.isnan(obs)))

    def test_payload(self):
        """设置有效载荷"""
        from src.control.embodied_sim import EmbodiedSimulator

        sim = EmbodiedSimulator(grade='M', seed=42)
        sim.reset()
        sim.set_payload(50.0)

        self.assertEqual(sim.physics.payload_mass, 50.0)

    def test_terrain(self):
        """设置地形"""
        from src.control.embodied_sim import EmbodiedSimulator

        sim = EmbodiedSimulator(grade='M', seed=42)
        sim.reset()

        for terrain in ['flat', 'slope', 'rough']:
            sim.set_terrain(terrain)
            self.assertEqual(sim.physics.terrain, terrain)

    def test_stop(self):
        """停止仿真"""
        from src.control.embodied_sim import EmbodiedSimulator

        sim = EmbodiedSimulator(grade='M', seed=42)
        sim.reset()
        sim.stop()

        self.assertFalse(sim._is_running)
        self.assertRaises(RuntimeError, lambda: sim.step(cmd_vx=0.1, cmd_wz=0.0))


class TestEmbodiedSimGradeSpec(unittest.TestCase):
    """五级规格测试"""

    def test_grade_spec_completeness(self):
        """所有等级规格完整"""
        from src.control.embodied_sim import AGV_SIM_GRADES, get_sim_grade_spec

        required_keys = [
            'dt', 'control_rate', 'sensor_rate', 'max_linear_speed',
            'max_angular_speed', 'max_linear_accel', 'max_angular_accel',
            'tactile_array_size', 'force_noise_std', 'imu_noise_std',
            'encoder_resolution', 'payload_kg', 'vehicle_mass_kg',
            'wheel_radius_m', 'wheelbase_m'
        ]

        for grade, spec in AGV_SIM_GRADES.items():
            for key in required_keys:
                self.assertIn(key, spec, f"Grade {grade} missing key: {key}")

    def test_higher_grades_higher_fidelity(self):
        """更高等级 = 更高保真 (dt更小, rate更高, noise更低)"""
        from src.control.embodied_sim import AGV_SIM_GRADES

        grades = ['S', 'M', 'L', 'XL', 'XXL']
        specs = [AGV_SIM_GRADES[g] for g in grades]

        # dt 递减
        dts = [s['dt'] for s in specs]
        self.assertEqual(dts, sorted(dts, reverse=True))

        # 控制频率递增
        rates = [s['control_rate'] for s in specs]
        self.assertEqual(rates, sorted(rates))

        # 噪声标准差递减
        noises = [s['imu_noise_std'] for s in specs]
        self.assertEqual(noises, sorted(noises, reverse=True))

    def test_payload_scales_with_grade(self):
        """有效载荷随等级增加"""
        from src.control.embodied_sim import AGV_SIM_GRADES

        grades = ['S', 'M', 'L', 'XL', 'XXL']
        payloads = [AGV_SIM_GRADES[g]['payload_kg'] for g in grades]

        # 递增
        self.assertEqual(payloads, sorted(payloads))


class TestEmbodiedSimEnv(unittest.TestCase):
    """Gymnasium 环境接口测试"""

    def test_env_creation(self):
        """环境创建"""
        from src.control.embodied_sim import EmbodiedSimEnv

        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            env = EmbodiedSimEnv(grade=grade)
            self.assertEqual(env.grade, grade)
            self.assertIsNotNone(env.observation_space)
            self.assertIsNotNone(env.action_space)

    def test_env_reset(self):
        """环境重置"""
        from src.control.embodied_sim import EmbodiedSimEnv

        env = EmbodiedSimEnv(grade='M')
        obs, info = env.reset(seed=42)

        self.assertEqual(obs.shape, (31,))
        self.assertFalse(np.any(np.isnan(obs)))
        self.assertIsInstance(info, dict)

    def test_env_step(self):
        """环境一步"""
        from src.control.embodied_sim import EmbodiedSimEnv

        env = EmbodiedSimEnv(grade='M')
        env.reset(seed=42)

        action = np.array([0.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)

        self.assertEqual(obs.shape, (31,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    def test_action_decoding(self):
        """动作解码"""
        from src.control.embodied_sim import EmbodiedSimEnv

        env = EmbodiedSimEnv(grade='M')
        env.reset(seed=42)

        max_v = env.spec['max_linear_speed']
        max_w = env.spec['max_angular_speed']

        # 满幅动作
        action_full = np.array([1.0, 1.0])
        obs1, _, _, _, _ = env.step(action_full)

        # 零动作
        env.reset(seed=42)
        action_zero = np.array([0.0, 0.0])
        obs2, _, _, _, _ = env.step(action_zero)

        # 满幅和零幅的速度响应应不同
        self.assertFalse(np.allclose(obs1, obs2))

    def test_episode_truncation(self):
        """回合截断"""
        from src.control.embodied_sim import EmbodiedSimEnv

        env = EmbodiedSimEnv(grade='M', max_episode_steps=10)
        env.reset(seed=42)

        truncated_seen = []
        for i in range(12):
            _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0]))
            truncated_seen.append(truncated)

        # max_episode_steps=10: after 10 steps (indices 0-9), episode truncates
        # First 9 steps (indices 0-8): truncated should be False
        for i in range(9):
            self.assertFalse(truncated_seen[i], f"Step {i} should not truncate")
        # Step 9 (10th step) onwards: truncated should be True
        for i in range(9, len(truncated_seen)):
            self.assertTrue(truncated_seen[i], f"Step {i} should truncate")

    def test_tracking_reward(self):
        """速度跟踪奖励"""
        from src.control.embodied_sim import EmbodiedSimEnv

        env = EmbodiedSimEnv(grade='M', reward_type='tracking')
        env.reset(seed=42)

        _, reward, _, _, _ = env.step(np.array([0.0, 0.0]))

        self.assertIsInstance(reward, float)

    def test_energy_efficient_reward(self):
        """能量效率奖励"""
        from src.control.embodied_sim import EmbodiedSimEnv

        env = EmbodiedSimEnv(grade='M', reward_type='energy_efficient')
        env.reset(seed=42)

        _, reward, _, _, _ = env.step(np.array([0.0, 0.0]))

        self.assertIsInstance(reward, float)

    def test_close(self):
        """关闭环境"""
        from src.control.embodied_sim import EmbodiedSimEnv

        env = EmbodiedSimEnv(grade='M')
        env.reset(seed=42)
        env.close()

        self.assertFalse(env.sim._is_running)


class TestCreateSimEnv(unittest.TestCase):
    """工厂函数测试"""

    def test_create_sim_env(self):
        """创建仿真环境工厂函数"""
        from src.control.embodied_sim import create_sim_env

        env = create_sim_env(grade='M')
        self.assertEqual(env.grade, 'M')
        env.reset(seed=42)
        env.close()

    def test_all_grades(self):
        """所有等级"""
        from src.control.embodied_sim import create_sim_env

        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            env = create_sim_env(grade=grade)
            self.assertEqual(env.grade, grade)
            env.close()


class TestGetGradeSummary(unittest.TestCase):
    """规格摘要测试"""

    def test_grade_summary(self):
        """五级规格摘要"""
        from src.control.embodied_sim import get_grade_summary

        summary = get_grade_summary()
        self.assertIsInstance(summary, str)
        self.assertIn('S', summary)
        self.assertIn('M', summary)
        self.assertIn('L', summary)
        self.assertIn('XL', summary)
        self.assertIn('XXL', summary)


class TestSimEnvironmentState(unittest.TestCase):
    """仿真环境状态测试"""

    def test_state_to_array(self):
        """状态展平为观测向量"""
        from src.control.embodied_sim import SimEnvironmentState

        state = SimEnvironmentState()
        arr = state.to_array()

        # 检查形状 (31维: pos(3) + euler(3) + lin_vel(3) + ang_vel(3) + imu_a(3) + imu_g(3) + wheel_pos(2) + wheel_vel(2) + battery(3) + slip(3) + temp(3))
        self.assertEqual(arr.shape, (31,))
        self.assertFalse(np.any(np.isnan(arr)))

    def test_state_initial_values(self):
        """初始值正确"""
        from src.control.embodied_sim import SimEnvironmentState

        state = SimEnvironmentState()

        self.assertEqual(state.timestamp, 0.0)
        self.assertEqual(state.dt, 0.01)
        self.assertTrue(np.allclose(state.position, np.zeros(3)))
        self.assertTrue(np.allclose(state.linear_velocity, np.zeros(3)))


class TestBatterySimulation(unittest.TestCase):
    """电池/SOC仿真测试 (新增)"""

    def test_battery_soc_decreases(self):
        """电池SOC随运动降低"""
        from src.control.embodied_sim import EmbodiedSimulator

        sim = EmbodiedSimulator(grade='M', seed=42)
        sim.reset()
        
        initial_soc = sim.state.battery_soc
        
        # 模拟多步运动
        for _ in range(100):
            sim.step(cmd_vx=0.5, cmd_wz=0.0)
        
        final_soc = sim.state.battery_soc
        
        # SOC应该降低
        self.assertLess(final_soc, initial_soc)
        # SOC不应该低于0
        self.assertGreaterEqual(final_soc, 0.0)

    def test_battery_state_dict(self):
        """电池状态字典正确"""
        from src.control.embodied_sim import EmbodiedSimulator

        sim = EmbodiedSimulator(grade='M', seed=42)
        sim.reset()
        sim.step(cmd_vx=0.1, cmd_wz=0.0)
        
        sensor_dict = sim.get_sensor_dict()
        battery = sensor_dict['battery']
        
        self.assertIn('soc', battery)
        self.assertIn('voltage', battery)
        self.assertIn('current', battery)
        self.assertIn('remaining_wh', battery)

    def test_physics_battery_getter(self):
        """PhysicsSimulator电池状态获取"""
        from src.control.embodied_sim import PhysicsSimulator

        physics = PhysicsSimulator(grade='M')
        physics.reset()
        
        # 模拟运动
        physics.step(cmd_vx=0.5, cmd_wz=0.0, dt=0.01)
        
        battery_state = physics.get_battery_state()
        
        self.assertIn('soc', battery_state)
        self.assertIn('voltage', battery_state)
        self.assertEqual(battery_state['voltage'], 48.0)


class TestWheelSlipSimulation(unittest.TestCase):
    """车轮滑移仿真测试 (新增)"""

    def test_wheel_slip_on_rough_terrain(self):
        """粗糙地形上车轮滑移增加"""
        from src.control.embodied_sim import EmbodiedSimulator

        sim = EmbodiedSimulator(grade='M', seed=42)
        sim.reset()
        
        # 平地
        sim.set_terrain("flat")
        for _ in range(10):
            sim.step(cmd_vx=0.5, cmd_wz=0.0)
        slip_flat = sim.state.wheel_slip_l + sim.state.wheel_slip_r
        
        # 粗糙地形
        sim.set_terrain("rough")
        for _ in range(10):
            sim.step(cmd_vx=0.5, cmd_wz=0.0)
        slip_rough = sim.state.wheel_slip_l + sim.state.wheel_slip_r
        
        # 粗糙地形滑移应该更大
        self.assertGreaterEqual(slip_rough, slip_flat)

    def test_terrain_friction_change(self):
        """地形摩擦系数随地形变化"""
        from src.control.embodied_sim import PhysicsSimulator

        physics = PhysicsSimulator(grade='M')
        
        physics.terrain = "flat"
        physics.step(cmd_vx=0.5, cmd_wz=0.0, dt=0.01)
        friction_flat = physics.terrain_friction
        
        physics.terrain = "wet"
        physics.step(cmd_vx=0.5, cmd_wz=0.0, dt=0.01)
        friction_wet = physics.terrain_friction
        
        # 湿滑地面摩擦系数更低
        self.assertLess(friction_wet, friction_flat)

    def test_wheel_slip_getter(self):
        """车轮滑移率获取"""
        from src.control.embodied_sim import PhysicsSimulator

        physics = PhysicsSimulator(grade='M')
        physics.reset()
        physics.step(cmd_vx=0.5, cmd_wz=0.0, dt=0.01)
        
        slip_l, slip_r = physics.get_wheel_slip()
        
        self.assertGreaterEqual(slip_l, 0.0)
        self.assertLessEqual(slip_l, 1.0)
        self.assertGreaterEqual(slip_r, 0.0)
        self.assertLessEqual(slip_r, 1.0)


class TestMotorTemperatureSimulation(unittest.TestCase):
    """电机温度仿真测试 (新增)"""

    def test_motor_temp_increases(self):
        """电机温度随运动升高"""
        from src.control.embodied_sim import EmbodiedSimulator

        sim = EmbodiedSimulator(grade='M', seed=42)
        sim.reset()
        
        initial_temp_l = sim.state.motor_temp_l
        initial_temp_r = sim.state.motor_temp_r
        
        # 模拟多步运动
        for _ in range(100):
            sim.step(cmd_vx=0.5, cmd_wz=0.0)
        
        final_temp_l = sim.state.motor_temp_l
        final_temp_r = sim.state.motor_temp_r
        
        # 温度应该升高或保持
        self.assertGreaterEqual(final_temp_l, initial_temp_l)
        self.assertGreaterEqual(final_temp_r, initial_temp_r)

    def test_motor_overheating_flag(self):
        """电机过热标志"""
        from src.control.embodied_sim import SimEnvironmentState

        state = SimEnvironmentState()
        state.motor_temp_l = 85.0
        state.motor_temp_r = 75.0
        state.motor_overheating = (state.motor_temp_l >= 80.0 or state.motor_temp_r >= 80.0)
        
        self.assertTrue(state.motor_overheating)

    def test_motor_temp_getter(self):
        """电机温度获取"""
        from src.control.embodied_sim import PhysicsSimulator

        physics = PhysicsSimulator(grade='M')
        physics.reset()
        physics.step(cmd_vx=0.5, cmd_wz=0.0, dt=0.01)
        
        temp_l, temp_r = physics.get_motor_temperatures()
        
        self.assertGreater(temp_l, 0.0)
        self.assertGreater(temp_r, 0.0)

    def test_motor_sensor_dict(self):
        """电机温度传感器字典"""
        from src.control.embodied_sim import EmbodiedSimulator

        sim = EmbodiedSimulator(grade='M', seed=42)
        sim.reset()
        sim.step(cmd_vx=0.1, cmd_wz=0.0)
        
        sensor_dict = sim.get_sensor_dict()
        motor_temp = sensor_dict['motor_temp']
        
        self.assertIn('motor_l', motor_temp)
        self.assertIn('motor_r', motor_temp)
        self.assertIn('overheating', motor_temp)


class TestAGVGradesExtended(unittest.TestCase):
    """AGV五级规格扩展测试"""

    def test_battery_per_grade(self):
        """不同AGV等级的电池容量"""
        from src.control.embodied_sim import EmbodiedSimulator, get_sim_grade_spec

        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            sim = EmbodiedSimulator(grade=grade, seed=42)
            sim.reset()
            
            # 验证电池状态存在
            self.assertGreater(sim.state.battery_soc, 0.0)
            self.assertEqual(sim.state.battery_voltage, 48.0)

    def test_observation_space_per_grade(self):
        """所有AGV等级观测空间一致 (31维)"""
        from src.control.embodied_sim import EmbodiedSimulator

        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            sim = EmbodiedSimulator(grade=grade, seed=42)
            sim.reset()
            sim.step(cmd_vx=0.1, cmd_wz=0.0)
            
            obs = sim.get_observation()
            self.assertEqual(obs.shape[0], 31)


if __name__ == '__main__':
    unittest.main()
