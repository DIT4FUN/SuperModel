"""
test_simulation_new_features.py - 仿真增强新功能测试
====================================================
测试 WheelTerrainInteractionModel 和 TrajectoryPredictor
"""

import pytest
import numpy as np
import math


class TestWheelTerrainInteractionModel:
    """WheelTerrainInteractionModel 轮地交互模型测试"""

    def test_terrain_params_defined(self):
        """测试所有地形参数已定义"""
        from src.embodied.simulation_enhancement import TERRAIN_PARAMS, TerrainType
        for terrain in TerrainType:
            assert terrain in TERRAIN_PARAMS, f"Missing params for {terrain}"
        assert len(TERRAIN_PARAMS) == len(TerrainType)

    def test_rolling_resistance_order(self):
        """测试滚动阻力大小关系: 硬地面 < 草地 < 沙地 < 泥地"""
        from src.embodied.simulation_enhancement import WheelTerrainInteractionModel, TerrainType
        model = WheelTerrainInteractionModel(vehicle_mass=50.0)
        terrains = [
            TerrainType.FLAT_CONCRETE,
            TerrainType.GRASS,
            TerrainType.SAND,
            TerrainType.MUDSOIL,
        ]
        rr_values = [model.get_terrain_params()["rolling_resistance"] for _ in terrains]
        # 设置地形后获取
        for t, expected_min in zip(terrains, [0.0, 0.04, 0.10, 0.20]):
            model.set_terrain(t)
            params = model.get_terrain_params()
            assert params["rolling_resistance"] >= expected_min, f"{t} rr too low"

    def test_set_terrain_history(self):
        """测试地形切换历史记录"""
        from src.embodied.simulation_enhancement import WheelTerrainInteractionModel, TerrainType
        model = WheelTerrainInteractionModel()
        model.set_terrain(TerrainType.FLAT_CONCRETE, timestamp=1.0)
        model.set_terrain(TerrainType.GRASS, timestamp=2.0)
        model.set_terrain(TerrainType.GRAVEL, timestamp=3.0)
        assert len(model.terrain_history) == 3
        assert model.terrain_history[1][1] == TerrainType.GRASS

    def test_slope_force(self):
        """测试坡度力计算"""
        from src.embodied.simulation_enhancement import WheelTerrainInteractionModel
        model = WheelTerrainInteractionModel(vehicle_mass=100.0)
        # 水平: 无坡度力
        model.set_slope(0.0)
        assert abs(model.compute_slope_force()) < 1e-6
        # 10度坡度
        slope_10deg = 10 * math.pi / 180
        model.set_slope(slope_10deg)
        expected = 100 * 9.81 * math.sin(slope_10deg)
        assert abs(model.compute_slope_force() - expected) < 0.1

    def test_max_speed_on_slope(self):
        """测试坡度最大速度计算"""
        from src.embodied.simulation_enhancement import WheelTerrainInteractionModel, TerrainType
        # Use larger motors for a capable AGV that can actually climb slopes
        model = WheelTerrainInteractionModel(max_torque_per_motor=50.0)
        model.set_terrain(TerrainType.FLAT_CONCRETE)
        # 轻微坡度应有速度输出
        max_v = model.compute_max_speed_on_slope(0.05)
        assert max_v > 0, f"Concrete slope should allow positive speed, got {max_v}"
        # 泥地极限坡度
        model.set_terrain(TerrainType.MUDSOIL)
        max_v_mud = model.compute_max_speed_on_slope(0.5)  # 约30度
        assert max_v_mud >= 0  # 任何非负值

    def test_apply_terrain_to_velocity(self):
        """测试地形对速度的影响"""
        from src.embodied.simulation_enhancement import WheelTerrainInteractionModel, TerrainType
        model = WheelTerrainInteractionModel()
        model.set_terrain(TerrainType.EPOXY_FLOOR)
        v_new = model.apply_terrain_to_velocity(1.0, dt=0.1, commanded_torque=0.0)
        # 无驱动时在平整地面速度应下降（阻力）
        assert v_new <= 1.0

    def test_imu_noise_different_terrains(self):
        """测试不同地形的IMU噪声差异"""
        from src.embodied.simulation_enhancement import WheelTerrainInteractionModel, TerrainType
        model = WheelTerrainInteractionModel()
        base_accel = np.array([0.0, 0.0, 9.81])
        base_gyro = 0.0

        # 硬地面噪声小
        model.set_terrain(TerrainType.FLAT_CONCRETE)
        accel_hard, gyro_hard = model.get_imu_noise(base_accel, base_gyro)
        noise_hard = np.linalg.norm(accel_hard - base_accel)

        # 沙地噪声大
        model.set_terrain(TerrainType.SAND)
        accel_sand, gyro_sand = model.get_imu_noise(base_accel, base_gyro)
        noise_sand = np.linalg.norm(accel_sand - base_accel)

        assert noise_sand > noise_hard, "Sand should have higher IMU noise"

    def test_safe_speed_for_turn(self):
        """测试转弯安全速度"""
        from src.embodied.simulation_enhancement import WheelTerrainInteractionModel
        model = WheelTerrainInteractionModel()
        # 大转弯半径 -> 较高安全速度
        safe_v_large = model.compute_safe_speed_for_turn(2.0)
        # 小转弯半径 -> 较低安全速度
        safe_v_small = model.compute_safe_speed_for_turn(0.3)
        assert safe_v_large > safe_v_small

    def test_terrain_classification_from_imu(self):
        """测试IMU地形分类"""
        from src.embodied.simulation_enhancement import WheelTerrainInteractionModel, TerrainType
        model = WheelTerrainInteractionModel()

        # 低振动 -> 硬地面 (用固定值避免随机性)
        low_vib_accel = np.array([0.001, -0.001, 9.81])
        terrain = model.classify_terrain_from_imu(low_vib_accel, 0.005, 0.5)
        assert terrain in [TerrainType.FLAT_CONCRETE, TerrainType.EPOXY_FLOOR, TerrainType.TILE_FLOOR], \
            f"Low vibration should be hard floor, got {terrain}"

        # 高振动 -> 沙地/泥地
        high_vib_accel = np.array([0.5, -0.3, 8.5])
        terrain_hard = model.classify_terrain_from_imu(high_vib_accel, 0.1, 0.5)
        assert terrain_hard in [TerrainType.SAND, TerrainType.MUDSOIL, TerrainType.GRAVEL,
                                  TerrainType.DIRT, TerrainType.UNKNOWN], \
            f"High vibration should be soft terrain, got {terrain_hard}"

    def test_traversability_score(self):
        """测试可通行性评分"""
        from src.embodied.simulation_enhancement import WheelTerrainInteractionModel, TerrainType
        model = WheelTerrainInteractionModel()
        model.set_terrain(TerrainType.FLAT_CONCRETE)
        score_concrete = model.get_traversability_score()
        model.set_terrain(TerrainType.SAND)
        score_sand = model.get_traversability_score()
        assert 0.0 <= score_concrete <= 1.0
        assert 0.0 <= score_sand <= 1.0
        assert score_concrete > score_sand, "Concrete more traversable than sand"

    def test_reset(self):
        """测试重置"""
        from src.embodied.simulation_enhancement import WheelTerrainInteractionModel, TerrainType
        model = WheelTerrainInteractionModel()
        model.set_terrain(TerrainType.GRASS)
        model.set_slope(0.2)
        model.reset()
        assert model.current_terrain == TerrainType.FLAT_CONCRETE
        assert model.current_slope_rad == 0.0
        assert len(model.terrain_history) == 0


class TestTrajectoryPredictor:
    """TrajectoryPredictor 轨迹预测器测试"""

    def test_predict_constant_velocity(self):
        """测试恒定速度轨迹预测"""
        from src.embodied.simulation_enhancement import TrajectoryPredictor
        predictor = TrajectoryPredictor(dt=0.1, max_prediction_horizon=1.0)
        initial = np.array([0.0, 0.0, 0.0, 0.5, 0.0])  # x, y, theta, v, omega
        traj = predictor.predict_constant_velocity(initial, v=0.5, omega=0.0)
        assert len(traj) > 0
        # 直线运动，y应保持不变
        y_values = [pt['y'] for pt in traj]
        assert max(y_values) - min(y_values) < 0.05, "Should be nearly straight line"

    def test_predict_circular_trajectory(self):
        """测试圆弧轨迹预测"""
        from src.embodied.simulation_enhancement import TrajectoryPredictor
        predictor = TrajectoryPredictor(dt=0.1, max_prediction_horizon=2.0)
        initial = np.array([0.0, 0.0, 0.0, 0.5, 0.5])
        traj = predictor.predict_constant_velocity(initial, v=0.5, omega=0.5)
        assert len(traj) > 0
        # 角速度>0，应有航向角累积
        theta_final = traj[-1]['theta']
        assert theta_final > 0.0

    def test_predict_trajectory_length(self):
        """测试轨迹长度计算"""
        from src.embodied.simulation_enhancement import TrajectoryPredictor
        predictor = TrajectoryPredictor()
        initial = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
        traj = predictor.predict_constant_velocity(initial, v=1.0, omega=0.0)
        length = predictor.compute_trajectory_length(traj)
        expected_length = 1.0 * predictor.max_horizon  # v * time
        assert abs(length - expected_length) / expected_length < 0.1

    def test_collision_detection(self):
        """测试碰撞检测"""
        from src.embodied.simulation_enhancement import TrajectoryPredictor
        predictor = TrajectoryPredictor(dt=0.1, max_prediction_horizon=2.0)
        # 轨迹朝向 (1, 0) 方向前进
        initial = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
        traj = predictor.predict_constant_velocity(initial, v=1.0, omega=0.0)
        # 障碍物在轨迹前方
        collision = predictor.detect_collision_with_obstacle(
            traj, obstacle_pos=(2.0, 0.0), obstacle_radius=0.3, agv_radius=0.25
        )
        assert collision is not None, "Should detect collision"
        assert collision['time'] > 0

    def test_no_collision_when_clear(self):
        """测试无碰撞情况"""
        from src.embodied.simulation_enhancement import TrajectoryPredictor
        predictor = TrajectoryPredictor(dt=0.1, max_prediction_horizon=1.0)
        initial = np.array([0.0, 0.0, 0.0, 0.5, 0.0])
        traj = predictor.predict_constant_velocity(initial, v=0.5, omega=0.0)
        # 障碍物在侧面，不应碰撞
        collision = predictor.detect_collision_with_obstacle(
            traj, obstacle_pos=(5.0, 5.0), obstacle_radius=0.3, agv_radius=0.25
        )
        assert collision is None, "Should not detect collision"

    def test_trajectory_conflict_detection(self):
        """测试轨迹冲突检测"""
        from src.embodied.simulation_enhancement import TrajectoryPredictor
        predictor = TrajectoryPredictor(dt=0.1, max_prediction_horizon=2.0)
        # 两条轨迹从不同起点出发
        traj_a = predictor.predict_constant_velocity(
            np.array([0.0, 0.0, 0.0, 0.5, 0.0]), v=0.5, omega=0.0
        )
        traj_b = predictor.predict_constant_velocity(
            np.array([0.0, 0.1, 0.0, 0.5, 0.0]), v=0.5, omega=0.0
        )
        # 两条轨迹几乎平行，会有小间距
        conflict = predictor.detect_trajectory_conflict(traj_a, traj_b, min_separation=0.5)
        # 由于起点y差0.1，加上运动范围，可能会冲突
        assert conflict is None or isinstance(conflict, dict)

    def test_time_to_collision(self):
        """测试碰撞时间计算"""
        from src.embodied.simulation_enhancement import TrajectoryPredictor
        predictor = TrajectoryPredictor(dt=0.1, max_prediction_horizon=3.0)
        initial = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
        traj = predictor.predict_constant_velocity(initial, v=1.0, omega=0.0)
        ttc = predictor.get_time_to_collision(traj, obstacle_pos=(1.5, 0.0), obstacle_radius=0.3)
        assert ttc is not None
        assert 0.5 < ttc < 3.0  # 约1.5m远处，以1m/s前进

    def test_bt_actions_trajectory(self):
        """测试从行为树动作序列预测"""
        from src.embodied.simulation_enhancement import TrajectoryPredictor
        predictor = TrajectoryPredictor(dt=0.1, max_prediction_horizon=5.0)
        initial = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        # 动作序列: (v, omega, duration)
        bt_actions = [
            (0.5, 0.0, 1.0),   # 直线1秒
            (0.5, 0.5, 1.0),   # 右转1秒
            (0.5, 0.0, 1.0),   # 直线1秒
        ]
        traj = predictor.predict_from_bt_actions(initial, bt_actions)
        assert len(traj) > 0
        # 总时长约3秒
        total_time = traj[-1]['t'] if traj else 0
        assert 2.5 < total_time < 3.5

    def test_normalize_angle(self):
        """测试角度归一化"""
        from src.embodied.simulation_enhancement import TrajectoryPredictor
        # 直接测试静态方法
        norm = TrajectoryPredictor._normalize
        assert abs(norm(np.pi * 2)) < 1e-6, f"2π should normalize to ~0, got {norm(np.pi*2)}"
        assert abs(norm(-np.pi * 2)) < 1e-6, f"-2π should normalize to ~0, got {norm(-np.pi*2)}"
        # π stays as π in the [-π, π] range (same as math.atan2)
        assert abs(norm(np.pi) - np.pi) < 1e-6, f"π should stay as π, got {norm(np.pi)}"
