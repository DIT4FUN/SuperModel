"""
simulation_enhancement_tests.py - 具身仿真环境增强测试
SuperModel 超模态大模型具身智能系统

测试内容:
- 物理参数校准测试
- 传感器噪声模型测试
- 延迟/丢包仿真测试
- 碰撞检测增强测试
- 仓库场景生成测试
- 五级AGV物理参数适配测试
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embodied.simulation_enhancement import (
    PhysicsParameters,
    SensorNoiseModel,
    DelaySimulator,
    CollisionEnhancer,
    EnvironmentGenerator,
    WarehouseSceneGenerator,
    EmbodiedSimulationEnhancer,
    Obstacle,
)


class TestPhysicsParameters(unittest.TestCase):
    """测试物理参数"""

    def test_default_m_grade(self):
        """默认M级参数"""
        params = PhysicsParameters()
        self.assertEqual(params.mass_empty, 35.0)
        self.assertEqual(params.wheel_radius, 0.07)
        self.assertEqual(params.wheel_base, 0.45)

    def test_all_grades(self):
        """所有等级参数"""
        grades = ['S', 'M', 'L', 'XL', 'XXL']
        for grade in grades:
            params = PhysicsParameters.for_grade(grade)
            self.assertIsInstance(params, PhysicsParameters)
            self.assertGreater(params.mass_empty, 0)
            self.assertGreater(params.wheel_radius, 0)

    def test_mass_increases_with_grade(self):
        """质量随等级增加"""
        prev_mass = 0
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            params = PhysicsParameters.for_grade(grade)
            self.assertGreater(params.mass_empty, prev_mass)
            prev_mass = params.mass_empty

    def test_calculate_max_speed(self):
        """计算最大速度"""
        params = PhysicsParameters.for_grade('M')
        max_speed = params.calculate_max_speed()
        self.assertGreater(max_speed, 0)
        self.assertLess(max_speed, 10)  # 合理范围

    def test_calculate_max_acceleration(self):
        """计算最大加速度"""
        params = PhysicsParameters.for_grade('M')
        max_accel = params.calculate_max_acceleration(current_load=0)
        self.assertGreater(max_accel, 0)
        # 满载加速度降低
        max_accel_loaded = params.calculate_max_acceleration(current_load=100)
        self.assertLess(max_accel_loaded, max_accel)


class TestSensorNoiseModel(unittest.TestCase):
    """测试传感器噪声模型"""

    def setUp(self):
        self.model = SensorNoiseModel(seed=42)

    def test_add_noise_lidar(self):
        """添加激光雷达噪声"""
        ranges = np.full(360, 10.0)
        noisy = self.model.add_noise_lidar(ranges)
        # 均值还是接近10
        self.assertAlmostEqual(np.mean(noisy), 10, delta=1)
        # 有方差
        self.assertGreater(np.std(noisy), 0)

    def test_add_noise_imu(self):
        """添加IMU噪声"""
        accel = np.zeros(3)
        gyro = np.zeros(3)
        noisy_accel, noisy_gyro = self.model.add_noise_imu(accel, gyro)
        self.assertEqual(noisy_accel.shape, (3,))
        self.assertEqual(noisy_gyro.shape, (3,))
        # 有噪声
        self.assertGreater(np.std(noisy_accel), 0)

    def test_add_noise_tactile(self):
        """添加触觉噪声"""
        pressures = np.zeros((8, 8))
        pressures[2:4, 2:4] = 0.5
        noisy = self.model.add_noise_tactile(pressures)
        self.assertEqual(noisy.shape, pressures.shape)
        # 值被裁剪到 0-1
        self.assertTrue(np.all(noisy >= 0))
        self.assertTrue(np.all(noisy <= 1))

    def test_add_noise_force(self):
        """添加力传感器噪声"""
        wrench = np.zeros(6)
        noisy = self.model.add_noise_force(wrench)
        self.assertEqual(noisy.shape, (6,))

    def test_reset_drift(self):
        """重置漂移"""
        # 先产生漂移
        accel = np.zeros(3)
        gyro = np.zeros(3)
        self.model.add_noise_imu(accel, gyro)
        self.model.reset_drift()
        self.assertEqual(self.model.drift, {})


class TestDelaySimulator(unittest.TestCase):
    """测试延迟仿真"""

    def setUp(self):
        self.sim = DelaySimulator(packet_loss_rate=0, seed=42)

    def test_default_sensor_delays_exist(self):
        """默认传感器延迟存在"""
        self.assertIn('lidar', self.sim.sensor_delay_ms)
        self.assertIn('imu', self.sim.sensor_delay_ms)
        self.assertIn('camera', self.sim.sensor_delay_ms)

    def test_should_drop_with_high_loss(self):
        """高丢包率应该丢包"""
        sim = DelaySimulator(packet_loss_rate=1.0)
        self.assertTrue(sim.should_drop())

    def test_buffer_and_get_delayed(self):
        """缓存和获取延迟数据"""
        self.sim.buffer_data('lidar', 0, [1, 2, 3])
        data = self.sim.get_delayed_data('lidar')
        self.assertEqual(data, [1, 2, 3])

    def test_clear(self):
        """清空缓存"""
        self.sim.buffer_data('lidar', 0, [1, 2, 3])
        self.assertIn('lidar', self.sim.buffers)
        self.sim.clear()
        # 清空后所有缓存被删除
        self.assertNotIn('lidar', self.sim.buffers)


class TestCollisionEnhancer(unittest.TestCase):
    """测试碰撞检测增强"""

    def setUp(self):
        self.enhancer = CollisionEnhancer()

    def test_check_proximity_near(self):
        """检测近处障碍物"""
        robot_pos = np.array([0, 0, 0])
        obstacles = [np.array([0.2, 0, 0])]
        is_near, dist, closest = self.enhancer.check_proximity(
            robot_pos, obstacles, robot_radius=0.3
        )
        self.assertTrue(is_near)
        self.assertLess(dist, 0.3 + 0.3)

    def test_check_proximity_far(self):
        """检测远处障碍物"""
        robot_pos = np.array([0, 0, 0])
        obstacles = [np.array([10, 0, 0])]
        is_near, dist, _ = self.enhancer.check_proximity(
            robot_pos, obstacles, robot_radius=0.3
        )
        self.assertFalse(is_near)

    def test_estimate_collision_force(self):
        """估计碰撞力"""
        force = self.enhancer.estimate_collision_force(0.01, 35, 1.0)
        self.assertGreater(force, 0)


class TestEnvironmentGenerator(unittest.TestCase):
    """测试环境生成器"""

    def setUp(self):
        self.gen = EnvironmentGenerator(seed=42)

    def test_generate_random_obstacles(self):
        """生成随机障碍物"""
        obstacles = self.gen.generate_random_obstacles(
            area_size=(10, 10), num_obstacles=10
        )
        self.assertEqual(len(obstacles), 10)
        for obs in obstacles:
            self.assertIsInstance(obs, Obstacle)
            self.assertGreater(obs.size[0], 0)

    def test_obstacle_bounding_box_contains(self):
        """障碍物AABB包含测试"""
        obs = Obstacle(
            position=np.array([5, 5, 1]),
            size=np.array([2, 2, 2]),
            obstacle_type='static'
        )
        min_corner, max_corner = obs.get_bounding_box()
        self.assertEqual(min_corner.tolist(), [4, 4, 0])
        self.assertEqual(max_corner.tolist(), [6, 6, 2])
        self.assertTrue(obs.contains_point(np.array([5, 5, 1])))
        self.assertFalse(obs.contains_point(np.array([0, 0, 0])))

    def test_generate_cluttered(self):
        """生成杂乱环境"""
        obstacles = self.gen.generate_cluttered_environment(20, 20, density=0.1)
        self.assertGreater(len(obstacles), 0)


class TestWarehouseSceneGenerator(unittest.TestCase):
    """测试仓库场景生成器"""

    def setUp(self):
        self.gen = WarehouseSceneGenerator(seed=42)

    def test_generate_warehouse(self):
        """生成标准仓库"""
        scene = self.gen.generate_warehouse(
            num_aisles=5,
            aisle_length=20.0,
        )
        self.assertIn('obstacles', scene)
        self.assertIn('start_positions', scene)
        self.assertIn('goal_positions', scene)
        self.assertIn('picking_stations', scene)
        self.assertGreater(len(scene['obstacles']), 0)

    def test_generate_picking_task(self):
        """生成拣选任务"""
        warehouse = self.gen.generate_warehouse(num_aisles=3)
        task = self.gen.generate_picking_task(warehouse, num_items=3)
        self.assertEqual(task['type'], 'order_picking')
        self.assertEqual(len(task['pick_points']), 3)
        self.assertIsNotNone(task['end_position'])


class TestEmbodiedSimulationEnhancer(unittest.TestCase):
    """测试整合仿真增强器"""

    def test_init_all_grades(self):
        """所有等级初始化"""
        grades = ['S', 'M', 'L', 'XL', 'XXL']
        for grade in grades:
            enhancer = EmbodiedSimulationEnhancer(agv_grade=grade)
            self.assertEqual(enhancer.grade, grade)
            params = enhancer.get_physics_parameters()
            self.assertIsInstance(params, PhysicsParameters)

    def test_process_sensor_data(self):
        """处理传感器数据"""
        enhancer = EmbodiedSimulationEnhancer(seed=42)
        ranges = np.full(360, 5.0)
        processed = enhancer.process_sensor_data('lidar', ranges)
        # 添加了噪声，应该有变化
        self.assertFalse(np.array_equal(processed, ranges))

    def test_generate_warehouse_scene(self):
        """生成仓库场景"""
        enhancer = EmbodiedSimulationEnhancer()
        scene = enhancer.generate_warehouse_scene(num_aisles=4)
        self.assertIn('obstacles', scene)
        self.assertGreater(len(scene['obstacles']), 0)

    def test_set_load(self):
        """设置负载"""
        enhancer = EmbodiedSimulationEnhancer(agv_grade='M')
        enhancer.set_load(50)
        self.assertEqual(enhancer.physics.mass_load, 35 + 50)

    def test_reset(self):
        """重置"""
        enhancer = EmbodiedSimulationEnhancer()
        enhancer.reset()
        # 漂移已重置
        if enhancer.noise_model:
            self.assertEqual(enhancer.noise_model.drift, {})


if __name__ == '__main__':
    unittest.main()
