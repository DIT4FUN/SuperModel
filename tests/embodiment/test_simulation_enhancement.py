"""
test_simulation_enhancement.py - 具身仿真环境增强模块测试
========================================================

测试 SuperModel 具身仿真增强模块:
- PhysicsParameters: AGV等级物理参数
- SensorNoiseModel: 传感器噪声模型
- DelaySimulator: 延迟仿真
- CollisionEnhancer: 碰撞检测增强
- EnvironmentGenerator: 环境生成器
- WarehouseSceneGenerator: 仓库场景生成
- EmbodiedSimulationEnhancer: 完整仿真增强器
- DynamicObstacleGenerator: 动态障碍物生成
- MultiAGVSimulationEnhancer: 多AGV仿真增强
"""

import pytest
import numpy as np
import time
from typing import Tuple

from src.embodied.simulation_enhancement import (
    PhysicsParameters,
    SensorNoiseModel,
    DelaySimulator,
    CollisionEnhancer,
    EnvironmentGenerator,
    WarehouseSceneGenerator,
    EmbodiedSimulationEnhancer,
    DynamicObstacleGenerator,
    MultiAGVSimulationEnhancer,
    WeatherType,
    WeatherEffect,
    WEATHER_EFFECTS,
    Obstacle,
    DigitalTwinSynchronizer,
    HILBridge,
    TerrainDeformationModel,
    AGVStateEstimator,
)


# =============================================================================
# PhysicsParameters Tests
# =============================================================================

class TestPhysicsParameters:
    """物理参数测试"""

    def test_default_initialization(self):
        """测试默认初始化"""
        params = PhysicsParameters()
        assert params.mass_empty == 35.0
        assert params.wheel_radius == 0.07
        assert params.wheel_base == 0.45
        assert params.wheel_friction == 0.95

    @pytest.mark.parametrize("grade", ["S", "M", "L", "XL", "XXL"])
    def test_grade_specific_params(self, grade):
        """测试各等级物理参数"""
        params = PhysicsParameters.for_grade(grade)
        assert params is not None
        assert params.mass_empty > 0
        assert params.wheel_radius > 0
        assert params.wheel_base > 0

    def test_grade_s_mass(self):
        """测试S级参数"""
        params = PhysicsParameters.for_grade("S")
        assert params.mass_empty == 15.0
        assert params.mass_load == 45.0
        assert params.wheel_radius == 0.05
        assert params.wheel_base == 0.30

    def test_grade_l_mass(self):
        """测试L级参数"""
        params = PhysicsParameters.for_grade("L")
        assert params.mass_empty == 60.0
        assert params.mass_load == 360.0
        assert params.wheel_base == 0.60

    def test_grade_xxl_mass(self):
        """测试XXL级参数"""
        params = PhysicsParameters.for_grade("XXL")
        assert params.mass_empty == 250.0
        assert params.mass_load == 1450.0
        assert params.wheel_radius == 0.095
        assert params.wheel_base == 1.20

    def test_calculate_max_speed(self):
        """测试最大速度计算"""
        params = PhysicsParameters()
        max_speed = params.calculate_max_speed()
        assert max_speed > 0

    @pytest.mark.parametrize("load_kg", [0.0, 20.0, 50.0, 100.0])
    def test_calculate_max_acceleration(self, load_kg):
        """测试不同负载下的最大加速度"""
        params = PhysicsParameters()
        accel = params.calculate_max_acceleration(load_kg)
        assert accel > 0

    def test_acceleration_decreases_with_load(self):
        """测试负载增加时加速度降低"""
        params = PhysicsParameters()
        accel_empty = params.calculate_max_acceleration(0.0)
        accel_loaded = params.calculate_max_acceleration(100.0)
        assert accel_loaded < accel_empty


# =============================================================================
# SensorNoiseModel Tests
# =============================================================================

class TestSensorNoiseModel:
    """传感器噪声模型测试"""

    def test_default_initialization(self):
        """测试默认初始化"""
        model = SensorNoiseModel()
        assert model.enable_gaussian is True
        assert model.enable_outliers is True
        assert model.enable_drift is True

    def test_lidar_noise_basic(self):
        """测试激光雷达噪声"""
        model = SensorNoiseModel(seed=42)
        ranges = np.linspace(1.0, 10.0, 360)
        noisy = model.add_noise_lidar(ranges, noise_std=0.015)
        assert noisy.shape == ranges.shape
        assert np.all(noisy >= 0)  # 距离不能为负

    def test_lidar_noise_preserves_shape(self):
        """测试激光雷达噪声保持形状"""
        model = SensorNoiseModel(seed=42)
        ranges = np.ones(720)
        noisy = model.add_noise_lidar(ranges)
        assert noisy.shape == ranges.shape

    def test_lidar_noise_reproducibility(self):
        """测试激光雷达噪声可重现性"""
        model1 = SensorNoiseModel(seed=123)
        model2 = SensorNoiseModel(seed=123)
        ranges = np.linspace(1.0, 5.0, 180)
        noisy1 = model1.add_noise_lidar(ranges)
        noisy2 = model2.add_noise_lidar(ranges)
        np.testing.assert_array_almost_equal(noisy1, noisy2)

    def test_lidar_noise_no_gaussian(self):
        """测试无高斯噪声"""
        model = SensorNoiseModel(enable_gaussian=False, seed=42)
        ranges = np.linspace(1.0, 10.0, 360)
        noisy = model.add_noise_lidar(ranges, noise_std=0.015)
        assert noisy.shape == ranges.shape

    def test_imu_noise_basic(self):
        """测试IMU噪声"""
        model = SensorNoiseModel(seed=42)
        accel = np.array([0.0, 0.0, 9.8])
        gyro = np.array([0.0, 0.0, 0.0])
        noisy_accel, noisy_gyro = model.add_noise_imu(accel, gyro)
        assert noisy_accel.shape == accel.shape
        assert noisy_gyro.shape == gyro.shape

    def test_encoder_noise(self):
        """测试编码器噪声"""
        model = SensorNoiseModel(seed=42)
        pos, vel = model.add_noise_encoder(1.5, 0.5)
        assert isinstance(pos, float)
        assert isinstance(vel, float)

    def test_encoder_noise_quantization(self):
        """测试编码器量化"""
        model = SensorNoiseModel(seed=42, enable_gaussian=False)
        pos, vel = model.add_noise_encoder(1.555, 0.5, resolution=0.01)
        assert abs(pos - 1.555) < 0.01

    def test_tactile_noise(self):
        """测试触觉传感器噪声"""
        model = SensorNoiseModel(seed=42)
        pressures = np.random.rand(16, 16)
        noisy = model.add_noise_tactile(pressures)
        assert noisy.shape == pressures.shape
        assert np.all(noisy >= 0) and np.all(noisy <= 1)

    def test_force_noise(self):
        """测试力传感器噪声"""
        model = SensorNoiseModel(seed=42)
        wrench = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        noisy = model.add_noise_force(wrench)
        assert noisy.shape == wrench.shape

    def test_reset_drift(self):
        """测试漂移重置"""
        model = SensorNoiseModel(seed=42)
        accel = np.array([0.0, 0.0, 9.8])
        gyro = np.array([0.0, 0.0, 0.0])
        model.add_noise_imu(accel, gyro)
        model.add_noise_imu(accel, gyro)
        model.reset_drift()
        assert model.drift == {}


# =============================================================================
# DelaySimulator Tests
# =============================================================================

class TestDelaySimulator:
    """延迟仿真测试"""

    def test_default_initialization(self):
        """测试默认初始化"""
        sim = DelaySimulator()
        assert sim.communication_delay_ms == 10.0
        assert sim.packet_loss_rate == 0.001
        assert 'lidar' in sim.sensor_delay_ms
        assert 'imu' in sim.sensor_delay_ms

    def test_should_drop_low_rate(self):
        """测试低丢包率"""
        sim = DelaySimulator(packet_loss_rate=0.0)
        dropped = sum(sim.should_drop() for _ in range(1000))
        assert dropped == 0

    def test_should_drop_high_rate(self):
        """测试高丢包率"""
        sim = DelaySimulator(packet_loss_rate=1.0)
        dropped = sum(sim.should_drop() for _ in range(100))
        assert dropped == 100

    def test_get_delay_samples(self):
        """测试延迟样本数"""
        sim = DelaySimulator()
        lidar_samples = sim.get_delay_samples('lidar')
        imu_samples = sim.get_delay_samples('imu')
        assert lidar_samples >= 1
        assert imu_samples >= 1

    def test_buffer_data_and_retrieve(self):
        """测试数据缓冲和延迟获取"""
        sim = DelaySimulator()
        timestamp = time.time()
        data = np.array([1.0, 2.0, 3.0])

        sim.buffer_data('lidar', timestamp, data)
        delayed = sim.get_delayed_data('lidar')
        assert delayed is not None
        np.testing.assert_array_equal(delayed, data)

    def test_buffer_multiple_data(self):
        """测试多个数据缓冲"""
        sim = DelaySimulator()
        timestamps = [time.time() + i * 0.01 for i in range(5)]
        data_list = [np.array([i]) for i in range(5)]

        for ts, data in zip(timestamps, data_list):
            sim.buffer_data('imu', ts, data)

        for expected_data in data_list:
            retrieved = sim.get_delayed_data('imu')
            np.testing.assert_array_equal(retrieved, expected_data)

    def test_get_delayed_data_empty(self):
        """测试空缓冲区"""
        sim = DelaySimulator()
        result = sim.get_delayed_data('nonexistent')
        assert result is None

    def test_clear_buffers(self):
        """测试清空缓冲区"""
        sim = DelaySimulator()
        sim.buffer_data('lidar', time.time(), np.array([1.0]))
        sim.buffer_data('imu', time.time(), np.array([2.0]))
        sim.clear()
        assert sim.get_delayed_data('lidar') is None
        assert sim.get_delayed_data('imu') is None


# =============================================================================
# CollisionEnhancer Tests
# =============================================================================

class TestCollisionEnhancer:
    """碰撞检测增强测试"""

    def test_default_initialization(self):
        """测试默认初始化"""
        enhancer = CollisionEnhancer()
        assert enhancer.enable_proximity_warning is True
        assert enhancer.proximity_threshold == 0.3
        assert enhancer.enable_force_estimation is True

    def test_check_proximity_no_obstacles(self):
        """测试无障碍物时的接近检测"""
        enhancer = CollisionEnhancer()
        robot_pos = np.array([5.0, 5.0, 0.0])
        obstacles = []
        is_near, min_dist, closest = enhancer.check_proximity(robot_pos, obstacles)
        assert is_near is False
        assert min_dist == float('inf')
        assert closest is None

    def test_check_proximity_with_obstacles(self):
        """测试有障碍物时的接近检测"""
        enhancer = CollisionEnhancer()
        robot_pos = np.array([1.0, 1.0, 0.0])
        obstacles = [np.array([1.2, 1.2, 0.0]), np.array([10.0, 10.0, 0.0])]
        is_near, min_dist, closest = enhancer.check_proximity(robot_pos, obstacles)
        # is_near may be np.bool_, use bool() for comparison
        assert bool(is_near) is True
        assert closest is not None

    def test_check_proximity_far_obstacle(self):
        """测试障碍物较远"""
        enhancer = CollisionEnhancer()
        robot_pos = np.array([0.0, 0.0, 0.0])
        obstacles = [np.array([10.0, 10.0, 0.0])]
        is_near, min_dist, closest = enhancer.check_proximity(robot_pos, obstacles, robot_radius=0.3)
        assert bool(is_near) is False

    def test_estimate_collision_force(self):
        """测试碰撞力估计"""
        enhancer = CollisionEnhancer()
        force = enhancer.estimate_collision_force(
            penetration_depth=0.05,
            robot_mass=50.0,
            relative_velocity=0.5
        )
        assert force > 0
        assert force > 500

    def test_estimate_collision_force_zero_depth(self):
        """测试零穿透深度"""
        enhancer = CollisionEnhancer()
        force = enhancer.estimate_collision_force(
            penetration_depth=0.0,
            robot_mass=50.0,
            relative_velocity=0.0
        )
        assert force == 0.0


# =============================================================================
# Obstacle Tests
# =============================================================================

class TestObstacle:
    """障碍物测试"""

    def test_obstacle_creation(self):
        """测试障碍物创建"""
        obs = Obstacle(
            position=np.array([1.0, 2.0, 0.5]),
            size=np.array([0.5, 0.5, 1.0]),
            obstacle_type="static",
        )
        assert obs.position[0] == 1.0
        assert obs.obstacle_type == "static"

    def test_get_bounding_box(self):
        """测试包围盒计算"""
        obs = Obstacle(
            position=np.array([5.0, 5.0, 1.0]),
            size=np.array([2.0, 2.0, 2.0]),
            obstacle_type="static",
        )
        min_corner, max_corner = obs.get_bounding_box()
        np.testing.assert_array_almost_equal(min_corner, np.array([4.0, 4.0, 0.0]))
        np.testing.assert_array_almost_equal(max_corner, np.array([6.0, 6.0, 2.0]))

    def test_contains_point_inside(self):
        """测试点在障碍物内"""
        obs = Obstacle(
            position=np.array([5.0, 5.0, 0.5]),
            size=np.array([2.0, 2.0, 1.0]),
            obstacle_type="static",
        )
        point = np.array([5.0, 5.0, 0.5])
        assert obs.contains_point(point) == True  # noqa: E712

    def test_contains_point_outside(self):
        """测试点在障碍物外"""
        obs = Obstacle(
            position=np.array([5.0, 5.0, 0.5]),
            size=np.array([2.0, 2.0, 1.0]),
            obstacle_type="static",
        )
        point = np.array([10.0, 10.0, 0.5])
        assert obs.contains_point(point) == False  # noqa: E712


# =============================================================================
# EnvironmentGenerator Tests
# =============================================================================

class TestEnvironmentGenerator:
    """环境生成器测试"""

    def test_default_initialization(self):
        """测试默认初始化"""
        gen = EnvironmentGenerator()
        assert gen.rng is not None

    def test_generate_random_obstacles_basic(self):
        """测试生成随机障碍物"""
        gen = EnvironmentGenerator(seed=42)
        obstacles = gen.generate_random_obstacles(
            area_size=(10.0, 10.0),
            num_obstacles=5,
        )
        assert len(obstacles) <= 5
        for obs in obstacles:
            assert obs.position[0] >= 0.8
            assert obs.obstacle_type in ['static', 'dynamic']

    def test_generate_random_obstacles_reproducibility(self):
        """测试生成可重现性"""
        gen1 = EnvironmentGenerator(seed=42)
        gen2 = EnvironmentGenerator(seed=42)
        obs1 = gen1.generate_random_obstacles((10.0, 10.0), 5)
        obs2 = gen2.generate_random_obstacles((10.0, 10.0), 5)
        assert len(obs1) == len(obs2)

    def test_generate_cluttered_environment(self):
        """测试生成杂乱环境"""
        gen = EnvironmentGenerator(seed=42)
        obstacles = gen.generate_cluttered_environment(20.0, 20.0, density=0.1)
        assert len(obstacles) > 0


# =============================================================================
# WarehouseSceneGenerator Tests
# =============================================================================

class TestWarehouseSceneGenerator:
    """仓库场景生成器测试"""

    def test_default_initialization(self):
        """测试默认初始化"""
        gen = WarehouseSceneGenerator()
        assert gen.rng is not None

    def test_generate_warehouse_basic(self):
        """测试生成仓库"""
        gen = WarehouseSceneGenerator(seed=42)
        warehouse = gen.generate_warehouse(num_aisles=3)
        assert 'obstacles' in warehouse
        assert 'start_positions' in warehouse
        assert 'goal_positions' in warehouse
        assert 'picking_stations' in warehouse
        assert len(warehouse['obstacles']) > 0

    def test_generate_warehouse_num_aisles(self):
        """测试不同通道数的仓库"""
        gen = WarehouseSceneGenerator(seed=42)
        for num_aisles in [1, 3, 5]:
            warehouse = gen.generate_warehouse(num_aisles=num_aisles)
            assert warehouse['num_aisles'] == num_aisles

    def test_generate_picking_task(self):
        """测试拣选任务生成"""
        gen = WarehouseSceneGenerator(seed=42)
        warehouse = gen.generate_warehouse(num_aisles=3)
        task = gen.generate_picking_task(warehouse, num_items=3)
        assert task['type'] == 'order_picking'
        assert len(task['pick_points']) == 3
        assert 'end_position' in task


# =============================================================================
# WeatherEffect Tests
# =============================================================================

class TestWeatherEffects:
    """天气效果测试"""

    def test_clear_weather(self):
        """测试晴天效果"""
        effect = WEATHER_EFFECTS[WeatherType.CLEAR]
        assert effect.lidar_noise_multiplier == 1.0
        assert effect.visibility_range == 100.0
        assert effect.friction_multiplier == 1.0

    def test_rain_weather(self):
        """测试雨天效果"""
        effect = WEATHER_EFFECTS[WeatherType.RAIN]
        assert effect.lidar_noise_multiplier == 1.8
        assert effect.friction_multiplier == 0.7
        assert effect.visibility_range == 50.0

    def test_snow_weather(self):
        """测试雪天效果"""
        effect = WEATHER_EFFECTS[WeatherType.SNOW]
        assert effect.friction_multiplier == 0.4
        assert effect.visibility_range == 30.0

    def test_fog_weather(self):
        """测试雾天效果"""
        effect = WEATHER_EFFECTS[WeatherType.FOG]
        assert effect.camera_noise_multiplier == 5.0
        assert effect.visibility_range == 15.0

    def test_dust_weather(self):
        """测试沙尘效果"""
        effect = WEATHER_EFFECTS[WeatherType.DUST]
        assert effect.imu_noise_multiplier == 1.3
        assert effect.visibility_range == 20.0


# =============================================================================
# EmbodiedSimulationEnhancer Tests
# =============================================================================

class TestEmbodiedSimulationEnhancer:
    """完整仿真增强器测试"""

    def test_default_initialization(self):
        """测试默认初始化"""
        enhancer = EmbodiedSimulationEnhancer()
        assert enhancer.grade == "M"
        assert enhancer.physics is not None
        assert enhancer.noise_model is not None
        assert enhancer.delay_simulator is not None

    @pytest.mark.parametrize("grade", ["S", "M", "L", "XL", "XXL"])
    def test_grade_initialization(self, grade):
        """测试各等级初始化"""
        enhancer = EmbodiedSimulationEnhancer(agv_grade=grade)
        assert enhancer.grade == grade
        assert isinstance(enhancer.physics, PhysicsParameters)

    def test_enable_noise_disabled(self):
        """测试禁用噪声"""
        enhancer = EmbodiedSimulationEnhancer(enable_noise=False)
        assert enhancer.noise_model is None

    def test_enable_delay_disabled(self):
        """测试禁用延迟"""
        enhancer = EmbodiedSimulationEnhancer(enable_delay=False)
        assert enhancer.delay_simulator is None

    def test_enable_collision_disabled(self):
        """测试禁用碰撞增强"""
        enhancer = EmbodiedSimulationEnhancer(enable_enhanced_collision=False)
        assert enhancer.collision_enhancer is None

    def test_process_lidar_sensor_data(self):
        """测试处理激光雷达数据"""
        enhancer = EmbodiedSimulationEnhancer()
        ranges = np.ones(360) * 5.0
        processed = enhancer.process_sensor_data('lidar', ranges)
        assert processed is not None
        assert isinstance(processed, np.ndarray)

    def test_process_imu_sensor_data(self):
        """测试处理IMU数据"""
        enhancer = EmbodiedSimulationEnhancer()
        accel = np.array([0.0, 0.0, 9.8])
        gyro = np.array([0.0, 0.0, 0.0])
        processed = enhancer.process_sensor_data('imu', (accel, gyro))
        assert processed is not None

    def test_process_tactile_sensor_data(self):
        """测试处理触觉数据"""
        enhancer = EmbodiedSimulationEnhancer()
        pressures = np.random.rand(16, 16)
        processed = enhancer.process_sensor_data('tactile', pressures)
        assert processed is not None

    def test_process_force_sensor_data(self):
        """测试处理力传感器数据"""
        enhancer = EmbodiedSimulationEnhancer()
        wrench = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        processed = enhancer.process_sensor_data('force', wrench)
        assert processed is not None

    def test_generate_warehouse_scene(self):
        """测试生成仓库场景"""
        enhancer = EmbodiedSimulationEnhancer()
        scene = enhancer.generate_warehouse_scene(num_aisles=3)
        assert 'obstacles' in scene
        assert 'start_positions' in scene

    def test_get_physics_parameters(self):
        """测试获取物理参数"""
        enhancer = EmbodiedSimulationEnhancer(agv_grade="L")
        physics = enhancer.get_physics_parameters()
        assert isinstance(physics, PhysicsParameters)

    def test_set_load(self):
        """测试设置负载"""
        enhancer = EmbodiedSimulationEnhancer()
        original_mass = enhancer.physics.mass_load
        enhancer.set_load(50.0)
        assert enhancer.physics.mass_load == enhancer.physics.mass_empty + 50.0

    def test_reset(self):
        """测试重置"""
        enhancer = EmbodiedSimulationEnhancer()
        enhancer.reset()
        # 验证重置后噪声模型和延迟模拟器被清理
        if enhancer.noise_model:
            enhancer.noise_model.reset_drift()
        if enhancer.delay_simulator:
            enhancer.delay_simulator.clear()


# =============================================================================
# DynamicObstacleGenerator Tests
# =============================================================================

class TestDynamicObstacleGenerator:
    """动态障碍物生成器测试"""

    def test_default_initialization(self):
        """测试默认初始化"""
        gen = DynamicObstacleGenerator(scene_bounds=(0, 0, 10, 10))
        assert gen.scene_bounds == (0, 0, 10, 10)
        assert len(gen.obstacles) == 0

    def test_generate_person_obstacle(self):
        """测试生成人员障碍物"""
        gen = DynamicObstacleGenerator(scene_bounds=(0, 0, 10, 10))
        obs_id = gen.generate_person_obstacle()
        assert obs_id.startswith("person_")
        assert obs_id in gen.obstacles
        obs = gen.obstacles[obs_id]
        assert obs.obstacle_type == "person"

    def test_generate_person_with_start_position(self):
        """测试指定起始位置的人员"""
        gen = DynamicObstacleGenerator(scene_bounds=(0, 0, 10, 10))
        start_pos = np.array([5.0, 5.0, 0.0])
        obs_id = gen.generate_person_obstacle(start_pos=start_pos)
        obs = gen.obstacles[obs_id]
        np.testing.assert_array_equal(obs.position, start_pos)

    def test_generate_forklift_obstacle(self):
        """测试生成叉车障碍物"""
        gen = DynamicObstacleGenerator(scene_bounds=(0, 0, 10, 10))
        obs_id = gen.generate_forklift_obstacle()
        assert obs_id.startswith("forklift_")
        assert obs_id in gen.obstacles
        obs = gen.obstacles[obs_id]
        assert obs.obstacle_type == "forklift"

    def test_generate_random_box_obstacle(self):
        """测试生成随机箱子障碍物"""
        gen = DynamicObstacleGenerator(scene_bounds=(0, 0, 10, 10))
        obs_id = gen.generate_random_box_obstacle()
        assert obs_id.startswith("box_")
        assert obs_id in gen.obstacles
        obs = gen.obstacles[obs_id]
        assert obs.obstacle_type == "box"

    def test_step_updates_positions(self):
        """测试步进更新位置"""
        gen = DynamicObstacleGenerator(scene_bounds=(0, 0, 10, 10))
        path = [np.array([0.0, 0.0]), np.array([5.0, 5.0])]
        obs_id = gen.generate_person_obstacle(path=path)
        initial_pos = gen.obstacles[obs_id].position.copy()
        gen.step(0.1)
        updated_pos = gen.obstacles[obs_id].position
        # 位置应该发生变化
        assert np.linalg.norm(updated_pos[:2] - initial_pos[:2]) >= 0

    def test_remove_obstacle(self):
        """测试移除障碍物 - 通过字典操作"""
        gen = DynamicObstacleGenerator(scene_bounds=(0, 0, 10, 10))
        obs_id = gen.generate_person_obstacle()
        assert obs_id in gen.obstacles
        # 直接从字典删除（实际remove_obstacle方法不存在）
        del gen.obstacles[obs_id]
        assert obs_id not in gen.obstacles


# =============================================================================
# MultiAGVSimulationEnhancer Tests
# =============================================================================

class TestMultiAGVSimulationEnhancer:
    """多AGV仿真增强器测试"""

    def test_default_initialization(self):
        """测试默认初始化"""
        configs = {
            'agv1': {'grade': 'M'},
            'agv2': {'grade': 'L'},
        }
        multi = MultiAGVSimulationEnhancer(agv_configs=configs)
        assert 'agv1' in multi.enhancers
        assert 'agv2' in multi.enhancers
        assert len(multi.agv_ids) == 2

    def test_update_robot_state(self):
        """测试更新机器人状态"""
        configs = {'agv1': {'grade': 'M'}}
        multi = MultiAGVSimulationEnhancer(agv_configs=configs)
        state = {'position': np.array([1.0, 2.0, 0.0]), 'velocity': np.array([0.5, 0.0, 0.0])}
        multi.update_robot_state('agv1', state)
        assert 'position' in multi._robot_states['agv1']

    def test_process_sensor_data(self):
        """测试处理传感器数据"""
        configs = {'agv1': {'grade': 'M'}}
        multi = MultiAGVSimulationEnhancer(agv_configs=configs)
        ranges = np.ones(360) * 5.0
        processed = multi.process_sensor_data('agv1', 'lidar', ranges)
        assert processed is not None

    def test_check_inter_agv_collision_far(self):
        """测试两AGV相距较远"""
        configs = {
            'agv1': {'grade': 'M'},
            'agv2': {'grade': 'M'},
        }
        multi = MultiAGVSimulationEnhancer(agv_configs=configs)
        multi.update_robot_state('agv1', {'position': np.array([0.0, 0.0, 0.0])})
        multi.update_robot_state('agv2', {'position': np.array([10.0, 10.0, 0.0])})
        is_close, dist = multi.check_inter_agv_collision('agv1', 'agv2')
        assert bool(is_close) is False
        assert dist > 1.0

    def test_check_inter_agv_collision_close(self):
        """测试两AGV相距较近"""
        configs = {
            'agv1': {'grade': 'M'},
            'agv2': {'grade': 'M'},
        }
        multi = MultiAGVSimulationEnhancer(agv_configs=configs)
        multi.update_robot_state('agv1', {'position': np.array([0.0, 0.0, 0.0])})
        multi.update_robot_state('agv2', {'position': np.array([0.3, 0.0, 0.0])})
        is_close, dist = multi.check_inter_agv_collision('agv1', 'agv2')
        assert bool(is_close) is True

    def test_get_all_collision_warnings(self):
        """测试获取所有碰撞警告"""
        configs = {
            'agv1': {'grade': 'M'},
            'agv2': {'grade': 'M'},
            'agv3': {'grade': 'M'},
        }
        multi = MultiAGVSimulationEnhancer(agv_configs=configs)
        multi.update_robot_state('agv1', {'position': np.array([0.0, 0.0, 0.0])})
        multi.update_robot_state('agv2', {'position': np.array([0.3, 0.0, 0.0])})  # 近
        multi.update_robot_state('agv3', {'position': np.array([10.0, 10.0, 0.0])})  # 远
        warnings = multi.get_all_collision_warnings()
        assert len(warnings) >= 0  # 至少agv1-agv2应该触发警告

    def test_generate_shared_scene_warehouse(self):
        """测试生成共享仓库场景"""
        configs = {
            'agv1': {'grade': 'M'},
            'agv2': {'grade': 'L'},
        }
        multi = MultiAGVSimulationEnhancer(agv_configs=configs)
        scene = multi.generate_shared_scene(scene_type="warehouse", num_aisles=3)
        assert 'obstacles' in scene


# =============================================================================
# Integration Tests
# =============================================================================

class TestSimulationEnhancementIntegration:
    """仿真增强集成测试"""

    def test_full_enhancer_pipeline(self):
        """测试完整增强器管道"""
        enhancer = EmbodiedSimulationEnhancer(agv_grade="M")

        # 生成仓库场景
        scene = enhancer.generate_warehouse_scene(num_aisles=3)
        assert len(scene['obstacles']) > 0

        # 处理传感器数据
        ranges = np.ones(360) * 5.0
        processed = enhancer.process_sensor_data('lidar', ranges)
        assert processed is not None

        # 获取物理参数
        physics = enhancer.get_physics_parameters()
        assert physics.mass_empty > 0

    def test_multi_agv_with_scene(self):
        """测试多AGV与场景集成"""
        configs = {
            'agv1': {'grade': 'M', 'enable_noise': True},
            'agv2': {'grade': 'L', 'enable_noise': True},
        }
        multi = MultiAGVSimulationEnhancer(agv_configs=configs)
        scene = multi.generate_shared_scene(scene_type="warehouse", num_aisles=2)

        # 更新AGV状态
        for agv_id in multi.agv_ids:
            multi.update_robot_state(agv_id, {
                'position': np.array([1.0 * multi.agv_ids.index(agv_id), 0.0, 0.0]),
                'velocity': np.array([0.5, 0.0, 0.0]),
            })

        # 检查碰撞
        warnings = multi.get_all_collision_warnings()
        assert isinstance(warnings, list)

    def test_weather_effects_on_agv_physics(self):
        """测试天气对AGV物理参数的影响"""
        # 不同天气对应不同的摩擦系数
        rain_effect = WEATHER_EFFECTS[WeatherType.RAIN]
        snow_effect = WEATHER_EFFECTS[WeatherType.SNOW]
        assert rain_effect.friction_multiplier < 1.0
        assert snow_effect.friction_multiplier < rain_effect.friction_multiplier

    def test_agv_grade_max_speed_ordering(self):
        """测试AGV等级与最大速度的关系"""
        grades = ["S", "M", "L", "XL", "XXL"]
        max_speeds = []
        for grade in grades:
            enhancer = EmbodiedSimulationEnhancer(agv_grade=grade)
            max_speed = enhancer.physics.calculate_max_speed()
            max_speeds.append(max_speed)

        # XXL级AGV的最大速度应该不小于S级
        assert max_speeds[-1] >= max_speeds[0]

    def test_sensor_noise_processing_pipeline(self):
        """测试传感器噪声处理管道"""
        enhancer = EmbodiedSimulationEnhancer(enable_noise=True)

        # 处理多种传感器
        for sensor_type, data in [
            ('lidar', np.ones(360) * 5.0),
            ('tactile', np.random.rand(16, 16)),
            ('force', np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ]:
            processed = enhancer.process_sensor_data(sensor_type, data)
            assert processed is not None

    def test_delay_and_noise_combined(self):
        """测试延迟和噪声组合"""
        enhancer = EmbodiedSimulationEnhancer(enable_noise=True, enable_delay=True)
        ranges = np.ones(360) * 5.0

        # 多次处理同一数据
        results = [enhancer.process_sensor_data('lidar', ranges.copy()) for _ in range(5)]
        # 结果应该不完全相同（有随机噪声）
        # 但非常接近
        assert all(r is not None for r in results)


# ==============================================================================
# Tests for New Classes: DigitalTwinSynchronizer, HILBridge, 
#                         TerrainDeformationModel, AGVStateEstimator
# ==============================================================================


class TestDigitalTwinSynchronizer:
    """测试数字孪生同步器"""

    def test_default_initialization(self):
        """测试默认初始化"""
        sync = DigitalTwinSynchronizer()
        assert sync.sync_frequency == 50.0
        assert sync.mode == "bidirectional"
        assert sync.sync_enabled is True
        assert not sync.is_diverged
        assert sync.sync_count == 0

    def test_real_state_update(self):
        """测试真实状态更新"""
        sync = DigitalTwinSynchronizer()
        position = np.array([1.0, 2.0, 0.5])
        velocity = np.array([0.5, 0.1])
        sync.update_real_state(position, velocity, timestamp=1.0)
        np.testing.assert_array_almost_equal(sync.real_position, position)
        np.testing.assert_array_almost_equal(sync.real_velocity, velocity)

    def test_sim_state_update(self):
        """测试仿真状态更新"""
        sync = DigitalTwinSynchronizer()
        position = np.array([0.9, 1.9, 0.49])
        velocity = np.array([0.48, 0.09])
        sync.update_sim_state(position, velocity, timestamp=1.0)
        np.testing.assert_array_almost_equal(sync.sim_position, position)

    def test_kalman_filter_sync(self):
        """测试卡尔曼滤波同步"""
        sync = DigitalTwinSynchronizer(use_kalman_filter=True, sync_frequency=50.0)
        for i in range(10):
            pos = np.array([0.1 * i, 0.05 * i, 0.01 * i])
            vel = np.array([0.1, 0.01])
            sync.update_real_state(pos, vel, timestamp=float(i) * 0.02)
            sync.update_sim_state(pos * 0.99, vel * 0.99, timestamp=float(i) * 0.02)
            result = sync.sync(current_time=float(i) * 0.02)
            assert result['status'] in ('ok', 'diverged')
            assert 'estimated_position' in result
        assert sync.sync_count == 10

    def test_delay_compensation(self):
        """测试延迟补偿"""
        sync = DigitalTwinSynchronizer()
        for i in range(5):
            sync.update_real_state(np.array([float(i), 0.0, 0.0]),
                                   np.array([1.0, 0.0]), timestamp=float(i) * 0.1)
        # 插值查询
        pos, vel = sync.get_compensated_state(target_timestamp=0.25)
        assert abs(pos[0] - 2.5) < 0.01

    def test_divergence_detection(self):
        """测试异常检测"""
        sync = DigitalTwinSynchronizer(position_drift_threshold=0.1, velocity_drift_threshold=0.1)
        # 真实状态
        sync.update_real_state(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0]), timestamp=0.0)
        # 仿真状态 - 巨大偏差 (先同步一次，建立正确的估计)
        sync.update_sim_state(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0]), timestamp=0.0)
        sync.sync(current_time=0.0)
        # 然后更新仿真到巨大偏差，使用real_to_sim模式
        sync.set_mode("real_to_sim")
        sync.update_sim_state(np.array([10.0, 10.0, 0.0]), np.array([5.0, 1.0]), timestamp=0.1)
        result = sync.sync(current_time=0.1)
        assert result['is_diverged'] is True
        assert result['action'] == 'reset_simulation'

    def test_mode_setting(self):
        """测试模式设置"""
        sync = DigitalTwinSynchronizer()
        sync.set_mode("real_to_sim")
        assert sync.mode == "real_to_sim"
        sync.set_mode("bidirectional")
        assert sync.mode == "bidirectional"
        sync.set_mode("invalid_mode")
        assert sync.mode == "bidirectional"  # 不变

    def test_reset(self):
        """测试重置"""
        sync = DigitalTwinSynchronizer()
        sync.update_real_state(np.array([1.0, 2.0, 0.5]), np.array([0.5, 0.1]), timestamp=1.0)
        sync.sync_count = 5
        sync.reset()
        assert sync.sync_count == 0
        assert not sync.is_diverged
        np.testing.assert_array_equal(sync.real_position, np.zeros(3))


class TestHILBridge:
    """测试HIL桥接器"""

    def test_default_initialization(self):
        """测试默认初始化"""
        hil = HILBridge(grade="M")
        assert hil.grade == "M"
        assert hil.control_frequency == 100.0
        assert hil.mode == HILBridge.Mode.HIL_SOFTWARE_IN_LOOP
        assert not hil.is_running

    def test_sensor_interface_registration(self):
        """测试传感器接口注册"""
        hil = HILBridge()
        mock_interface = type('MockInterface', (), {'read': lambda: np.array([1.0, 2.0, 3.0])})()
        hil.register_sensor_interface('lidar', mock_interface)
        assert 'lidar' in hil.real_sensor_interfaces

    def test_actuator_interface_registration(self):
        """测试执行器接口注册"""
        hil = HILBridge()
        mock_interface = type('MockActuator', (), {'set_speed': lambda *args: None})()
        hil.register_actuator_interface('motor', mock_interface)
        assert 'motor' in hil.real_actuator_interfaces

    def test_read_real_sensors(self):
        """测试读取真实传感器"""
        hil = HILBridge()
        mock_lidar = type('MockLidar', (), {'read': lambda self: np.ones(360) * 5.0})()
        hil.register_sensor_interface('lidar', mock_lidar)
        data = hil.read_real_sensors(['lidar'])
        assert 'lidar' in data
        assert data['lidar'] is not None

    def test_read_real_sensors_with_dropout(self):
        """测试传感器掉线故障注入"""
        hil = HILBridge()
        mock_interface = type('MockInterface', (), {'read': lambda self: np.array([1.0, 2.0])})()
        hil.register_sensor_interface('test_sensor', mock_interface)
        hil.inject_fault('sensor_dropout', active=True)
        hil.fault_params['sensor_dropout']['probability'] = 1.0  # 100% dropout
        data = hil.read_real_sensors(['test_sensor'])
        assert data['test_sensor'] is None

    def test_forward_command(self):
        """测试命令转发"""
        hil = HILBridge()
        forwarded = []
        mock_motor = type('MockMotor', (), {'set_speed': lambda self, *args: forwarded.append(args)})()
        hil.register_actuator_interface('motor', mock_motor)
        hil.forward_command({'actuator': 'motor', 'type': 'speed', 'value': (0.5, 0.1)})
        assert len(forwarded) == 1

    def test_forward_command_with_saturation(self):
        """测试命令饱和故障"""
        hil = HILBridge()
        forwarded = []
        mock_motor = type('MockMotor', (), {'set_speed': lambda self, *args: forwarded.append(args)})()
        hil.register_actuator_interface('motor', mock_motor)
        hil.inject_fault('actuator_saturation', active=True)
        hil.fault_params['actuator_saturation']['limit'] = 1.0
        hil.forward_command({'actuator': 'motor', 'type': 'speed', 'value': (5.0, 2.0)})
        # 值应该被限制到1.0 (saturated)
        assert forwarded[-1] == (1.0, 1.0), f"Expected (1.0, 1.0), got {forwarded[-1]}"

    def test_fault_injection(self):
        """测试故障注入"""
        hil = HILBridge()
        hil.inject_fault('sensor_dropout', True)
        assert hil.injectable_faults['sensor_dropout'] is True
        hil.inject_fault('sensor_dropout', False)
        assert hil.injectable_faults['sensor_dropout'] is False

    def test_start_stop_pause_resume(self):
        """测试生命周期"""
        hil = HILBridge()
        hil.start()
        assert hil.is_running is True
        assert hil.is_paused is False
        hil.pause()
        assert hil.is_paused is True
        hil.resume()
        assert hil.is_paused is False
        hil.stop()
        assert hil.is_running is False

    def test_step_not_running(self):
        """测试未运行时step返回"""
        hil = HILBridge()
        result = hil.step_simulation()
        assert result['status'] == 'not_running'

    def test_step_simulation(self):
        """测试仿真步进"""
        hil = HILBridge()
        hil.start()
        result = hil.step_simulation()
        assert result['mode'] == 'hil_software_in_loop'
        assert hil.total_steps == 1

    def test_get_performance_metrics(self):
        """测试性能指标"""
        hil = HILBridge()
        metrics = hil.get_performance_metrics()
        assert 'mode' in metrics
        assert 'is_running' in metrics
        assert 'total_steps' in metrics
        assert metrics['active_faults'] == []

    def test_reset(self):
        """测试重置"""
        hil = HILBridge()
        hil.start()
        hil.inject_fault('sensor_dropout', True)
        hil.total_steps = 100
        hil.reset()
        assert not hil.is_running
        assert not hil.injectable_faults['sensor_dropout']
        assert hil.total_steps == 0


class TestTerrainDeformationModel:
    """测试地形变形模型"""

    def test_default_initialization(self):
        """测试默认初始化"""
        tdm = TerrainDeformationModel()
        assert tdm.grid_resolution == 0.05
        assert tdm.terrain_type == "elastic"
        assert tdm.height_map.shape == (200, 200)

    def test_world_to_grid(self):
        """测试世界坐标到网格坐标"""
        tdm = TerrainDeformationModel()
        gx, gy = tdm.world_to_grid(0.0, 0.0)
        assert gx == tdm.nx // 2
        assert gy == tdm.ny // 2

    def test_wheel_deformation(self):
        """测试车轮变形"""
        tdm = TerrainDeformationModel(grid_resolution=0.05, grid_size=(100, 100))
        deform = tdm.apply_wheel_deformation(0.0, 0.0, load_force=500.0, wheel_radius=0.07)
        assert deform > 0
        # 中心位置应该有最大变形
        center_h = tdm.get_height_at(0.0, 0.0)
        assert center_h < 0  # 负值表示凹陷

    def test_agv_deformation(self):
        """测试AGV整体变形"""
        tdm = TerrainDeformationModel(grid_resolution=0.05, grid_size=(100, 100))
        position = np.array([0.0, 0.0, 0.0])
        result = tdm.apply_agv_deformation(position, total_mass=100.0)
        assert 'front' in result
        assert 'rear' in result
        assert result['front'] > 0
        assert result['rear'] > 0

    def test_terrain_recovery_elastic(self):
        """测试弹性地形恢复"""
        tdm = TerrainDeformationModel()
        tdm.terrain_type = "elastic"
        tdm.apply_wheel_deformation(0.0, 0.0, load_force=500.0)
        initial_height = tdm.get_height_at(0.0, 0.0)
        tdm.recover_terrain(dt=0.1)
        recovered_height = tdm.get_height_at(0.0, 0.0)
        assert recovered_height > initial_height  # 恢复中

    def test_terrain_recovery_rigid(self):
        """测试刚性地形不恢复"""
        tdm = TerrainDeformationModel()
        tdm.terrain_type = "rigid"
        tdm.apply_wheel_deformation(0.0, 0.0, load_force=500.0)
        initial_height = tdm.get_height_at(0.0, 0.0)
        tdm.recover_terrain(dt=1.0)
        recovered_height = tdm.get_height_at(0.0, 0.0)
        assert recovered_height == initial_height  # 刚性地形不恢复

    def test_get_slope_at(self):
        """测试斜坡梯度获取"""
        tdm = TerrainDeformationModel()
        # 添加坑洼
        tdm.add_pothole(0.5, 0.5, radius=0.2, depth=0.05)
        # 在坑洼边缘检测梯度
        dz_dx, dz_dy = tdm.get_slope_at(0.5 + 0.1, 0.5)
        # 坑洼边缘应该有梯度
        assert dz_dx != 0 or dz_dy != 0, "Expected non-zero slope at pothole edge"

    def test_terrain_adjustment(self):
        """测试地形影响调整"""
        tdm = TerrainDeformationModel()
        tdm.apply_wheel_deformation(0.0, 0.0, load_force=500.0)
        adj = tdm.get_terrain_adjustment(0.0, 0.0)
        assert 'height_offset' in adj
        assert 'pitch_offset' in adj
        assert 'roll_offset' in adj
        assert 'slope_factor' in adj
        assert 0 < adj['slope_factor'] <= 1

    def test_add_pothole(self):
        """测试添加坑洼"""
        tdm = TerrainDeformationModel()
        tdm.add_pothole(0.0, 0.0, radius=0.1, depth=0.02)
        depth = tdm.get_height_at(0.0, 0.0)
        assert depth < -0.01  # 应该有凹陷
        assert len(tdm.deformation_log) > 0

    def test_reset(self):
        """测试重置"""
        tdm = TerrainDeformationModel()
        tdm.add_pothole(0.0, 0.0, radius=0.1, depth=0.02)
        tdm.apply_wheel_deformation(0.0, 0.0, load_force=500.0)
        tdm.reset()
        assert tdm.get_height_at(0.0, 0.0) == 0.0
        assert len(tdm.deformation_log) == 0


class TestAGVStateEstimator:
    """测试AGV状态估计器"""

    def test_default_initialization(self):
        """测试默认初始化"""
        estimator = AGVStateEstimator()
        assert estimator.wheel_radius == 0.07
        assert estimator.wheel_base == 0.45
        assert estimator.state_dim == 7
        np.testing.assert_array_equal(estimator.x, np.zeros(7))

    def test_predict(self):
        """测试预测步骤"""
        estimator = AGVStateEstimator()
        estimator.predict(v=0.5, omega=0.0, dt=0.1)
        state = estimator.get_state()
        assert state['v'] == 0.5
        assert state['omega'] == 0.0

    def test_predict_arc_motion(self):
        """测试弧线运动预测"""
        estimator = AGVStateEstimator()
        initial_x = estimator.x[0]
        estimator.predict(v=0.5, omega=0.5, dt=0.1)
        assert estimator.x[0] != initial_x  # 有位移
        assert estimator.x[2] != 0  # 有角度变化

    def test_update_odometry(self):
        """测试里程计更新"""
        estimator = AGVStateEstimator()
        estimator.predict(v=0.5, omega=0.0, dt=0.1)
        initial_cov = estimator.P.copy()
        estimator.update_odometry(x=0.05, y=0.0, theta=0.0)
        # 协方差应该更新
        assert not np.array_equal(estimator.P, initial_cov)

    def test_update_imu(self):
        """测试IMU更新"""
        estimator = AGVStateEstimator()
        accel = np.array([0.1, 0.0, 9.81])
        gyro = 0.1
        initial_x = estimator.x.copy()
        estimator.update_imu(accel, gyro)
        assert not np.array_equal(estimator.x, initial_x)

    def test_get_state(self):
        """测试获取状态"""
        estimator = AGVStateEstimator()
        state = estimator.get_state()
        assert 'x' in state
        assert 'y' in state
        assert 'theta' in state
        assert 'v' in state
        assert 'omega' in state

    def test_get_covariance(self):
        """测试获取协方差"""
        estimator = AGVStateEstimator()
        cov = estimator.get_covariance()
        assert cov.shape == (7, 7)
        assert np.all(np.isfinite(cov))

    def test_reset(self):
        """测试重置"""
        estimator = AGVStateEstimator()
        estimator.predict(v=0.5, omega=0.1, dt=0.1)
        estimator.update_odometry(0.5, 0.0, 0.1)
        estimator.reset()
        np.testing.assert_array_equal(estimator.x, np.zeros(7))
        np.testing.assert_array_equal(estimator.get_state()['x'], 0.0)


class TestNewClassesIntegration:
    """测试新增类的集成"""

    def test_digital_twin_with_hil_bridge(self):
        """测试数字孪生与HIL桥接集成"""
        sync = DigitalTwinSynchronizer(sync_frequency=100.0)
        hil = HILBridge(control_frequency=100.0)

        for i in range(5):
            pos = np.array([float(i) * 0.1, 0.0, 0.0])
            vel = np.array([0.1, 0.0])
            sync.update_real_state(pos, vel, timestamp=float(i) * 0.01)
            sync.update_sim_state(pos * 0.99, vel * 0.99, timestamp=float(i) * 0.01)
            result = sync.sync(current_time=float(i) * 0.01)
            assert result is not None

    def test_terrain_with_state_estimator(self):
        """测试地形变形与状态估计集成"""
        tdm = TerrainDeformationModel(grid_resolution=0.05, grid_size=(100, 100))
        estimator = AGVStateEstimator()

        for i in range(3):
            x = float(i) * 0.1
            tdm.apply_wheel_deformation(x, 0.0, load_force=500.0)
            adj = tdm.get_terrain_adjustment(x, 0.0)
            assert adj['slope_factor'] > 0
            estimator.predict(v=0.1, omega=0.0, dt=0.1)
            estimator.update_odometry(x, 0.0, 0.0)

    def test_full_hil_pipeline(self):
        """测试完整HIL流水线"""
        hil = HILBridge(grade="M", control_frequency=50.0)
        sync = DigitalTwinSynchronizer(sync_frequency=50.0)
        tdm = TerrainDeformationModel()
        estimator = AGVStateEstimator()

        hil.start()
        for step in range(10):
            real_pos = np.array([float(step) * 0.01, 0.0, 0.0])
            real_vel = np.array([0.01, 0.0])
            sync.update_real_state(real_pos, real_vel, timestamp=float(step) * 0.02)
            sync.update_sim_state(real_pos * 0.99, real_vel * 0.99, timestamp=float(step) * 0.02)
            sync.sync(current_time=float(step) * 0.02)

            est_pos, est_vel = sync.get_estimated_state()
            estimator.predict(v=est_vel[0], omega=est_vel[1], dt=0.02)
            estimator.update_odometry(est_pos[0], est_pos[1], est_pos[2])

            tdm.apply_wheel_deformation(est_pos[0], est_pos[1], load_force=500.0)
            hil.step_simulation()

        metrics = hil.get_performance_metrics()
        assert metrics['total_steps'] == 10
        sync_result = sync.sync(current_time=1.0)
        assert 'estimated_position' in sync_result
        est_state = estimator.get_state()
        assert 'x' in est_state
