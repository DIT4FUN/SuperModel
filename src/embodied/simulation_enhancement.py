"""
simulation_enhancement.py - 具身仿真环境增强模块
SuperModel 超模态大模型具身智能系统

增强功能:
- 真实物理参数校准
- 摩擦力/惯性仿真
- 传感器噪声模型
- 延迟仿真
- 碰撞检测增强
- 多AGV协同场景生成
- 仓库/工厂场景模板
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import logging
import random
import math

class WeatherType(Enum):
    CLEAR = "clear"
    RAIN = "rain"
    SNOW = "snow"
    DUST = "dust"
    FOG = "fog"

@dataclass
class WeatherEffect:
    lidar_noise_multiplier: float = 1.0
    camera_noise_multiplier: float = 1.0
    tactile_noise_multiplier: float = 1.0
    imu_noise_multiplier: float = 1.0
    friction_multiplier: float = 1.0
    visibility_range: float = 100.0  # meters

WEATHER_EFFECTS: Dict[WeatherType, WeatherEffect] = {
    WeatherType.CLEAR: WeatherEffect(),
    WeatherType.RAIN: WeatherEffect(
        lidar_noise_multiplier=1.8,
        camera_noise_multiplier=2.5,
        tactile_noise_multiplier=1.2,
        friction_multiplier=0.7,
        visibility_range=50.0
    ),
    WeatherType.SNOW: WeatherEffect(
        lidar_noise_multiplier=2.2,
        camera_noise_multiplier=3.0,
        tactile_noise_multiplier=1.5,
        friction_multiplier=0.4,
        visibility_range=30.0
    ),
    WeatherType.DUST: WeatherEffect(
        lidar_noise_multiplier=2.0,
        camera_noise_multiplier=4.0,
        imu_noise_multiplier=1.3,
        visibility_range=20.0
    ),
    WeatherType.FOG: WeatherEffect(
        lidar_noise_multiplier=1.5,
        camera_noise_multiplier=5.0,
        visibility_range=15.0
    )
}

logger = logging.getLogger(__name__)
__all__ = [
    'PhysicsParameters',
    'SensorNoiseModel',
    'DelaySimulator',
    'CollisionEnhancer',
    'EnvironmentGenerator',
    'WarehouseSceneGenerator',
    'EmbodiedSimulationEnhancer',
]


@dataclass
class PhysicsParameters:
    """物理参数 - 真实AGV校准"""

    # 质量参数
    mass_empty: float = 35.0       # 空载质量 (kg) - M级AGV
    mass_load: float = 135.0       # 满载质量 (kg)
    wheel_radius: float = 0.07     # 轮子半径 (m) - 5.5寸轮毂直径140mm
    wheel_base: float = 0.45       # 轮距 (m)
    track_width: float = 0.35      # 轮轨宽度 (m)

    # 摩擦力参数
    wheel_friction: float = 0.95   # 轮子摩擦系数
    ground_friction: float = 0.8   # 地面摩擦系数
    rolling_resistance: float = 0.02 # 滚动阻力系数

    # 惯性参数
    moment_of_inertia: float = 1.2  # 转动惯量 (kg·m²)
    motor_inertia: float = 0.001    # 电机转动惯量

    # 电机参数
    motor_max_torque: float = 2.5   # 最大扭矩 (N·m)
    motor_max_speed: float = 120     # 最大转速 (rad/s)
    motor_kc: float = 0.08           # 扭矩常数 (N·m/A)
    motor_r: float = 1.2             # 相电阻 (Ω)

    # 传动参数
    gear_ratio: float = 1.0          # 减速比
    efficiency: float = 0.85         # 传动效率

    # 空气阻力
    drag_coefficient: float = 0.4    # 风阻系数
    frontal_area: float = 0.15       # 迎风面积 (m²)

    @classmethod
    def for_grade(cls, grade: str) -> 'PhysicsParameters':
        """根据AGV等级获取物理参数"""
        params = cls()
        if grade == 'S':
            params.mass_empty = 15.0
            params.mass_load = 45.0
            params.wheel_radius = 0.05
            params.wheel_base = 0.30
            params.motor_max_torque = 1.0
        elif grade == 'M':
            # 默认就是M级参数
            pass
        elif grade == 'L':
            params.mass_empty = 60.0
            params.mass_load = 360.0
            params.wheel_radius = 0.07
            params.wheel_base = 0.60
            params.motor_max_torque = 4.0
        elif grade == 'XL':
            params.mass_empty = 120.0
            params.mass_load = 720.0
            params.wheel_radius = 0.0825
            params.wheel_base = 0.80
            params.motor_max_torque = 6.0
        elif grade == 'XXL':
            params.mass_empty = 250.0
            params.mass_load = 1450.0
            params.wheel_radius = 0.095
            params.wheel_base = 1.20
            params.motor_max_torque = 10.0
        return params

    def calculate_max_speed(self) -> float:
        """计算最大线速度 (m/s)"""
        # motor_max_speed is RPM → convert to rad/s
        motor_max_speed_rad_s = self.motor_max_speed * 2 * np.pi / 60
        return motor_max_speed_rad_s * self.wheel_radius / self.gear_ratio * self.efficiency

    def calculate_max_acceleration(self, current_load: float = 0.0) -> float:
        """计算最大加速度 (m/s²)"""
        total_mass = self.mass_empty + current_load
        max_force = 2 * (self.motor_max_torque * self.gear_ratio * self.efficiency) / self.wheel_radius
        return max_force / total_mass


class SensorNoiseModel:
    """传感器噪声模型 - 真实仿真"""

    def __init__(
        self,
        enable_gaussian: bool = True,
        enable_outliers: bool = True,
        enable_drift: bool = True,
        seed: int = 42
    ):
        self.enable_gaussian = enable_gaussian
        self.enable_outliers = enable_outliers
        self.enable_drift = enable_drift
        self.rng = np.random.RandomState(seed)
        self.drift = {}

    def add_noise_lidar(self, ranges: np.ndarray, noise_std: float = 0.015) -> np.ndarray:
        """添加激光雷达噪声"""
        noisy_ranges = ranges.copy()

        if self.enable_gaussian:
            # 距离相关噪声
            noise = self.rng.normal(0, noise_std * (1 + ranges / 20), size=ranges.shape)
            noisy_ranges += noise

        if self.enable_outliers:
            # 0.5%概率出现离群点
            outlier_mask = self.rng.rand(*ranges.shape) < 0.005
            noisy_ranges[outlier_mask] = noisy_ranges[outlier_mask] * self.rng.uniform(1.2, 2.0)

        # 保证不出现负值
        noisy_ranges[noisy_ranges < 0] = 0
        return noisy_ranges

    def add_noise_imu(self, accel: np.ndarray, gyro: np.ndarray,
                      accel_std: float = 0.05, gyro_std: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """添加IMU噪声"""
        noisy_accel = accel.copy()
        noisy_gyro = gyro.copy()

        if self.enable_gaussian:
            noisy_accel += self.rng.normal(0, accel_std, size=accel.shape)
            noisy_gyro += self.rng.normal(0, gyro_std, size=gyro.shape)

        if self.enable_drift:
            # 温度漂移慢变化
            if 'imu' not in self.drift:
                self.drift['imu'] = self.rng.normal(0, 0.001, size=6)
            noisy_accel += self.drift['imu'][:3]
            noisy_gyro += self.drift['imu'][3:]
            # 漂移缓慢变化
            self.drift['imu'] += self.rng.normal(0, 0.0001, size=6)

        return noisy_accel, noisy_gyro

    def add_noise_encoder(self, position: float, velocity: float,
                          resolution: float = 0.001) -> Tuple[float, float]:
        """添加编码器噪声"""
        # 量化噪声
        quantized_pos = round(position / resolution) * resolution
        # 微小速度噪声
        noisy_vel = velocity + self.rng.normal(0, 0.01)
        return quantized_pos, noisy_vel

    def add_noise_tactile(self, pressures: np.ndarray, std: float = 0.02) -> np.ndarray:
        """添加触觉传感器噪声"""
        noisy = pressures.copy()
        if self.enable_gaussian:
            noisy += self.rng.normal(0, std, size=pressures.shape)
        return np.clip(noisy, 0, 1)

    def add_noise_force(self, wrench: np.ndarray, std: float = 0.05) -> np.ndarray:
        """添加力传感器噪声"""
        noisy = wrench.copy()
        if self.enable_gaussian:
            noisy += self.rng.normal(0, std, size=wrench.shape)
        return noisy

    def reset_drift(self) -> None:
        """重置漂移"""
        self.drift = {}


class DelaySimulator:
    """通信/传感器延迟仿真"""

    def __init__(
        self,
        sensor_delay_ms: Dict[str, float] = None,
        communication_delay_ms: float = 10.0,
        packet_loss_rate: float = 0.001,
        seed: int = 42
    ):
        self.sensor_delay_ms = sensor_delay_ms or {
            'lidar': 100.0,
            'imu': 10.0,
            'tactile': 20.0,
            'force': 20.0,
            'encoder': 5.0,
            'camera': 200.0,
        }
        self.communication_delay_ms = communication_delay_ms
        self.packet_loss_rate = packet_loss_rate
        self.rng = np.random.RandomState(seed)
        self.buffers: Dict[str, List[Tuple[float, Any]]] = {}

    def should_drop(self) -> bool:
        """是否丢包"""
        return self.rng.rand() < self.packet_loss_rate

    def get_delay_samples(self, sensor_type: str) -> int:
        """获取延迟样本数 (假设1kHz控制周期)"""
        delay_ms = self.sensor_delay_ms.get(sensor_type, 10.0)
        return max(1, int(delay_ms))  # 1ms per sample

    def buffer_data(self, sensor_type: str, timestamp: float, data: Any) -> None:
        """缓存数据模拟延迟"""
        if sensor_type not in self.buffers:
            self.buffers[sensor_type] = []
        self.buffers[sensor_type].append((timestamp, data))

    def get_delayed_data(self, sensor_type: str) -> Optional[Any]:
        """获取延迟后的数据"""
        if sensor_type not in self.buffers:
            return None
        if not self.buffers[sensor_type]:
            return None
        # 返回最老的数据
        return self.buffers[sensor_type].pop(0)[1]

    def clear(self) -> None:
        """清空缓存"""
        self.buffers.clear()


class CollisionEnhancer:
    """碰撞检测增强"""

    def __init__(
        self,
        enable_proximity_warning: bool = True,
        proximity_threshold: float = 0.3,
        enable_force_estimation: bool = True,
        enable_penetration_check: bool = True,
    ):
        self.enable_proximity_warning = enable_proximity_warning
        self.proximity_threshold = proximity_threshold
        self.enable_force_estimation = enable_force_estimation
        self.enable_penetration_check = enable_penetration_check

    def check_proximity(
        self,
        robot_position: np.ndarray,
        obstacles: List[np.ndarray],
        robot_radius: float = 0.3
    ) -> Tuple[bool, float, Optional[np.ndarray]]:
        """检查附近障碍物，提前预警"""
        min_dist = float('inf')
        closest_obstacle = None

        for obs_pos in obstacles:
            dist = np.linalg.norm(robot_position[:2] - obs_pos[:2])
            if dist < min_dist:
                min_dist = dist
                closest_obstacle = obs_pos

        is_near = min_dist < (robot_radius + self.proximity_threshold)
        return is_near, min_dist, closest_obstacle

    def estimate_collision_force(
        self,
        penetration_depth: float,
        robot_mass: float,
        relative_velocity: float
    ) -> float:
        """估计碰撞力"""
        # 简化的胡克定律模型
        stiffness = 10000  # N/m
        damping = 2000      # N·s/m
        elastic_force = stiffness * penetration_depth
        damping_force = damping * relative_velocity
        return elastic_force + damping_force

    def find_collision_contacts(
        self,
        robot_vertices: List[np.ndarray],
        obstacle_vertices: List[np.ndarray]
    ) -> List[np.ndarray]:
        """找到碰撞接触点 (SAT算法简化版)"""
        contacts = []
        # 这里简化实现，实际在PyBullet中已经有碰撞检测
        # 该方法用于额外的精细检查
        for rv in robot_vertices:
            min_dist = float('inf')
            closest_point = None
            for ov in obstacle_vertices:
                dist = np.linalg.norm(rv - ov)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = ov
            if min_dist < 0.02:  # 2cm容差
                contacts.append((rv + closest_point) / 2)
        return contacts


@dataclass
class Obstacle:
    """障碍物定义"""
    position: np.ndarray
    size: np.ndarray
    obstacle_type: str  # "static", "dynamic", "human"
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    id: str = ""

    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取AABB包围盒"""
        half = self.size / 2
        min_corner = self.position - half
        max_corner = self.position + half
        return min_corner, max_corner

    def contains_point(self, point: np.ndarray) -> bool:
        """检查点是否在障碍物内"""
        min_corner, max_corner = self.get_bounding_box()
        return np.all(point >= min_corner) and np.all(point <= max_corner)


class EnvironmentGenerator:
    """环境生成器 - 生成各种测试场景"""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_random_obstacles(
        self,
        area_size: Tuple[float, float],
        num_obstacles: int,
        min_size: Tuple[float, float] = (0.2, 0.2, 0.5),
        max_size: Tuple[float, float] = (1.0, 1.0, 2.0),
        margin: float = 0.8,
    ) -> List[Obstacle]:
        """生成随机障碍物"""
        obstacles = []
        width, height = area_size

        for i in range(num_obstacles):
            # 尝试找一个不重叠的位置
            for attempt in range(100):
                x = self.rng.uniform(margin, width - margin)
                y = self.rng.uniform(margin, height - margin)
                sx = self.rng.uniform(min_size[0], max_size[0])
                sy = self.rng.uniform(min_size[1], max_size[1])
                sz = self.rng.uniform(min_size[2], max_size[2])

                pos = np.array([x, y, sz/2])
                size = np.array([sx, sy, sz])

                # 检查重叠
                overlap = False
                for obs in obstacles:
                    dist = np.linalg.norm(pos[:2] - obs.position[:2])
                    min_dist = (size[0] + obs.size[0]) / 2 + margin
                    if dist < min_dist:
                        overlap = True
                        break

                if not overlap:
                    obs_type = self.rng.choice(['static', 'static', 'static', 'dynamic'], p=[0.7, 0.1, 0.1, 0.1])
                    vel = np.zeros(3)
                    if obs_type == 'dynamic':
                        vel[:2] = self.rng.uniform(-0.5, 0.5, size=2)
                    obstacles.append(Obstacle(
                        position=pos,
                        size=size,
                        obstacle_type=obs_type,
                        velocity=vel,
                        id=f"obs_{i}"
                    ))
                    break

        return obstacles

    def generate_cluttered_environment(
        self,
        width: float,
        height: float,
        density: float = 0.15
    ) -> List[Obstacle]:
        """生成杂乱环境"""
        area = width * height
        num_obstacles = int(area * density)
        return self.generate_random_obstacles((width, height), num_obstacles)


class WarehouseSceneGenerator(EnvironmentGenerator):
    """仓库场景生成器"""

    def __init__(self, seed: int = 42):
        super().__init__(seed)

    def generate_warehouse(
        self,
        num_aisles: int = 5,
        aisle_length: float = 20.0,
        shelf_width: float = 1.0,
        aisle_width: float = 3.0,
        shelves_per_aisle: int = 10,
    ) -> Dict[str, Any]:
        """生成标准仓库货架布局"""
        obstacles = []
        start_positions = []
        goal_positions = []
        picking_stations = []

        total_width = num_aisles * (shelf_width + aisle_width)

        # 生成货架通道
        for aisle_idx in range(num_aisles):
            aisle_x = aisle_idx * (shelf_width + aisle_width)

            # 左右两侧货架
            for side in [-shelf_width/2, aisle_width + shelf_width/2]:
                current_x = aisle_x + side + total_width/2

                for shelf_idx in range(shelves_per_aisle):
                    shelf_y = -aisle_length/2 + shelf_idx * (aisle_length / shelves_per_aisle) + aisle_length/(2 * shelves_per_aisle)
                    shelf_z = 1.6  # 货架高度中心
                    obstacles.append(Obstacle(
                        position=np.array([current_x, shelf_y, shelf_z]),
                        size=np.array([shelf_width, aisle_length/shelves_per_aisle * 0.9, 3.2]),
                        obstacle_type='static',
                        id=f"shelf_{aisle_idx}_{side}_{shelf_idx}"
                    ))

        # 起点区域（入口）
        start_x = -total_width/2 + 2
        start_y = 0
        start_positions.append(np.array([start_x, start_y, 0]))

        # 拣选站（出口）
        end_x = total_width/2 - 2
        end_y = 0
        goal_positions.append(np.array([end_x, end_y, 0]))
        picking_stations.append(np.array([end_x, end_y, 0]))

        return {
            'obstacles': obstacles,
            'start_positions': start_positions,
            'goal_positions': goal_positions,
            'picking_stations': picking_stations,
            'dimensions': (total_width, aisle_length),
            'num_aisles': num_aisles,
        }

    def generate_picking_task(
        self,
        warehouse: Dict[str, Any],
        num_items: int = 3
    ) -> Dict[str, Any]:
        """生成一个拣选任务"""
        # 从货架中随机选择位置作为拣选点
        obstacles = warehouse['obstacles']
        shelves = [obs for obs in obstacles if obs.id.startswith('shelf')]

        pick_points = []
        for _ in range(num_items):
            shelf = self.rng.choice(shelves)
            # 在货架前拣选位置
            pick_pos = shelf.position.copy()
            pick_pos[0] += shelf.size[0]/2 + 0.5  # 偏移到通道中
            pick_pos[2] = 0
            pick_points.append(pick_pos)

        # 终点在拣选站
        if warehouse['picking_stations']:
            end_pos = warehouse['picking_stations'][0]
        else:
            end_pos = warehouse['goal_positions'][0]

        return {
            'type': 'order_picking',
            'pick_points': pick_points,
            'end_position': end_pos,
            'num_items': num_items,
        }


class EmbodiedSimulationEnhancer:
    """具身仿真增强器 - 整合所有增强功能"""

    def __init__(
        self,
        agv_grade: str = "M",
        enable_noise: bool = True,
        enable_delay: bool = True,
        enable_enhanced_collision: bool = True,
        seed: int = 42
    ):
        self.physics = PhysicsParameters.for_grade(agv_grade)
        self.grade = agv_grade
        self.noise_model = SensorNoiseModel(seed=seed) if enable_noise else None
        self.delay_simulator = DelaySimulator(seed=seed) if enable_delay else None
        self.collision_enhancer = CollisionEnhancer(
            enable_proximity_warning=enable_enhanced_collision,
            proximity_threshold=0.3,
            enable_force_estimation=enable_enhanced_collision,
            enable_penetration_check=enable_enhanced_collision
        ) if enable_enhanced_collision else None
        self.environment_generator = EnvironmentGenerator(seed=seed)
        self.warehouse_generator = WarehouseSceneGenerator(seed=seed)

    def process_sensor_data(
        self,
        sensor_type: str,
        data: Any
    ) -> Any:
        """处理传感器数据，添加噪声和延迟"""
        # 先添加噪声
        if self.noise_model is not None:
            if sensor_type == 'lidar':
                data = self.noise_model.add_noise_lidar(data)
            elif sensor_type == 'imu':
                accel, gyro = data
                data = self.noise_model.add_noise_imu(accel, gyro)
            elif sensor_type == 'tactile':
                data = self.noise_model.add_noise_tactile(data)
            elif sensor_type == 'force':
                data = self.noise_model.add_noise_force(data)

        # 然后缓存模拟延迟（调用者需要后续获取延迟后的数据）
        if self.delay_simulator is not None:
            if not self.delay_simulator.should_drop():
                import time
                self.delay_simulator.buffer_data(sensor_type, time.time(), data)
                return self.delay_simulator.get_delayed_data(sensor_type)
            else:
                return None  # 丢包

        return data

    def generate_warehouse_scene(
        self,
        num_aisles: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """生成仓库场景"""
        return self.warehouse_generator.generate_warehouse(num_aisles, **kwargs)

    def get_physics_parameters(self) -> PhysicsParameters:
        """获取物理参数"""
        return self.physics

    def set_load(self, load_mass: float) -> None:
        """设置当前负载，更新物理参数"""
        self.physics.mass_load = self.physics.mass_empty + load_mass

    def reset(self) -> None:
        """重置仿真增强器"""
        if self.noise_model:
            self.noise_model.reset_drift()
        if self.delay_simulator:
            self.delay_simulator.clear()
# ============================================================================
# 多AGV仿真增强器
# ============================================================================


class MultiAGVSimulationEnhancer:
    """
    多AGV仿真增强器 - 管理多个AGV的仿真环境

    功能:
    - 为每个AGV创建独立的仿真增强器
    - 管理AGV间碰撞检测
    - 协调场景级仿真参数
    - 支持蜂群协同仿真
    """

    def __init__(
        self,
        agv_configs: Dict[str, Dict[str, Any]],
        shared_scene: Optional[Dict[str, Any]] = None,
        enable_inter_agv_collision: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            agv_configs: AGV ID -> 配置字典
                {
                    'agv1': {'grade': 'L', 'enable_noise': True},
                    'agv2': {'grade': 'M', 'enable_noise': True},
                }
            shared_scene: 共享场景信息 (障碍物/货架位置等)
            enable_inter_agv_collision: 是否启用AGV间碰撞检测
            seed: 随机种子
        """
        self.agv_ids = list(agv_configs.keys())
        self.shared_scene = shared_scene or {}
        self.enable_inter_agv_collision = enable_inter_agv_collision
        self.seed = seed

        # 为每个AGV创建独立的仿真增强器
        self.enhancers: Dict[str, EmbodiedSimulationEnhancer] = {}
        self._robot_states: Dict[str, Dict[str, Any]] = {}

        for agv_id, config in agv_configs.items():
            self.enhancers[agv_id] = EmbodiedSimulationEnhancer(
                agv_grade=config.get('grade', 'M'),
                enable_noise=config.get('enable_noise', True),
                enable_delay=config.get('enable_delay', True),
                enable_enhanced_collision=config.get('enable_collision', True),
                seed=seed + self.agv_ids.index(agv_id),
            )
            self._robot_states[agv_id] = {
                'position': np.zeros(3),
                'velocity': np.zeros(3),
                'heading': 0.0,
            }

    def update_robot_state(self, agv_id: str, state: Dict[str, Any]) -> None:
        """更新某个AGV的状态"""
        if agv_id in self._robot_states:
            self._robot_states[agv_id].update(state)

    def process_sensor_data(
        self,
        agv_id: str,
        sensor_type: str,
        data: Any
    ) -> Any:
        """为指定AGV处理传感器数据"""
        if agv_id not in self.enhancers:
            return data
        return self.enhancers[agv_id].process_sensor_data(sensor_type, data)

    def check_inter_agv_collision(
        self,
        agv1_id: str,
        agv2_id: str,
    ) -> Tuple[bool, float]:
        """
        检查两个AGV之间是否可能碰撞

        Returns:
            (is_close, min_distance)
        """
        if not self.enable_inter_agv_collision:
            return False, float('inf')

        if agv1_id not in self._robot_states or agv2_id not in self._robot_states:
            return False, float('inf')

        pos1 = np.array(self._robot_states[agv1_id].get('position', [0, 0, 0]))
        pos2 = np.array(self._robot_states[agv2_id].get('position', [0, 0, 0]))

        # 使用AGV的最小转向半径作为安全距离
        safe_distance = 0.6  # 米

        distance = np.linalg.norm(pos1[:2] - pos2[:2])
        is_close = distance < safe_distance

        return is_close, distance

    def get_all_collision_warnings(self) -> List[Dict[str, Any]]:
        """获取所有AGV对的碰撞警告"""
        warnings = []
        for i, agv1_id in enumerate(self.agv_ids):
            for agv2_id in self.agv_ids[i + 1:]:
                is_close, distance = self.check_inter_agv_collision(agv1_id, agv2_id)
                if is_close:
                    warnings.append({
                        'agv_pair': (agv1_id, agv2_id),
                        'min_distance': distance,
                        'severity': 'HIGH' if distance < 0.3 else 'MEDIUM',
                    })
        return warnings

    def generate_shared_scene(self, scene_type: str = "warehouse", **kwargs) -> Dict[str, Any]:
        """生成共享场景"""
        if scene_type == "warehouse":
            scene = self.enhancers[self.agv_ids[0]].generate_warehouse_scene(**kwargs)
        elif scene_type == "factory":
            scene = self.enhancers[self.agv_ids[0]].environment_generator.generate_cluttered_environment(**kwargs)
        else:
            scene = {}

        self.shared_scene = scene
        return scene

    def reset_all(self) -> None:
        """重置所有AGV的仿真增强器"""
        for enhancer in self.enhancers.values():
            enhancer.reset()

    def get_enhancer(self, agv_id: str) -> Optional[EmbodiedSimulationEnhancer]:
        """获取指定AGV的仿真增强器"""
        return self.enhancers.get(agv_id)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'num_agvs': len(self.agv_ids),
            'agv_ids': self.agv_ids,
            'grades': {agv_id: e.grade for agv_id, e in self.enhancers.items()},
            'inter_agv_collision_enabled': self.enable_inter_agv_collision,
            'active_collision_warnings': len(self.get_all_collision_warnings()),
        }


# ============================================================================
# 仿真指标收集器
# ============================================================================


class SimulationMetricsCollector:
    """
    仿真指标收集器 - 跟踪和记录仿真 KPIs

    跟踪指标:
    - 任务完成率 / 平均任务时间
    - 能耗统计 (电机电流/电压积分)
    - 碰撞次数
    - 路径效率 (实际路径 vs 最短路径)
    - 传感器数据质量 (延迟/丢包率)
    - AGV利用率
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """重置所有指标"""
        self.task_stats: List[Dict[str, Any]] = []
        self.energy_samples: List[float] = []
        self.collision_events: List[Dict[str, Any]] = []
        self.path_lengths: List[float] = []
        self.sensor_drop_events: List[Dict[str, Any]] = []
        self.active_time_per_agv: Dict[str, float] = {}
        self._task_start_times: Dict[str, float] = {}

    def start_task(self, task_id: str) -> None:
        """记录任务开始时间"""
        import time
        self._task_start_times[task_id] = time.time()

    def end_task(self, task_id: str, success: bool, task_type: str = "") -> None:
        """记录任务完成"""
        import time
        if task_id in self._task_start_times:
            duration = time.time() - self._task_start_times[task_id]
            self.task_stats.append({
                'task_id': task_id,
                'duration': duration,
                'success': success,
                'task_type': task_type,
            })
            del self._task_start_times[task_id]

    def record_energy(self, energy_joules: float) -> None:
        """记录能耗样本"""
        self.energy_samples.append(energy_joules)

    def record_collision(
        self,
        position: np.ndarray,
        severity: str = "LOW",
        agv_id: str = "",
    ) -> None:
        """记录碰撞事件"""
        import time
        self.collision_events.append({
            'timestamp': time.time(),
            'position': position.tolist() if isinstance(position, np.ndarray) else position,
            'severity': severity,
            'agv_id': agv_id,
        })

    def record_path_length(self, actual: float, optimal: float) -> None:
        """记录路径长度 (实际 vs 最优)"""
        efficiency = optimal / actual if actual > 0 else 1.0
        self.path_lengths.append({
            'actual': actual,
            'optimal': optimal,
            'efficiency': efficiency,
        })

    def record_sensor_drop(self, sensor_type: str, agv_id: str = "") -> None:
        """记录传感器丢包"""
        import time
        self.sensor_drop_events.append({
            'timestamp': time.time(),
            'sensor_type': sensor_type,
            'agv_id': agv_id,
        })

    def update_active_time(self, agv_id: str, delta_seconds: float) -> None:
        """更新AGV活跃时间"""
        self.active_time_per_agv[agv_id] = self.active_time_per_agv.get(agv_id, 0.0) + delta_seconds

    def get_summary(self) -> Dict[str, Any]:
        """获取指标汇总"""
        import time as time_module

        total_tasks = len(self.task_stats)
        successful_tasks = sum(1 for t in self.task_stats if t['success'])
        task_success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0

        avg_task_duration = (
            sum(t['duration'] for t in self.task_stats) / total_tasks
            if total_tasks > 0 else 0.0
        )

        total_energy = sum(self.energy_samples)
        avg_energy_per_sample = total_energy / len(self.energy_samples) if self.energy_samples else 0.0

        collision_count = len(self.collision_events)
        high_severity_collisions = sum(
            1 for c in self.collision_events if c['severity'] == 'HIGH'
        )

        path_efficiencies = [p['efficiency'] for p in self.path_lengths]
        avg_path_efficiency = (
            sum(path_efficiencies) / len(path_efficiencies)
            if path_efficiencies else 0.0
        )

        total_sensor_drops = len(self.sensor_drop_events)
        drop_rate = (
            total_sensor_drops / (total_sensor_drops + 1000)  # 假设基准1000次
            if total_sensor_drops > 0 else 0.0
        )

        return {
            'timestamp': time_module.time(),
            'tasks': {
                'total': total_tasks,
                'successful': successful_tasks,
                'success_rate': task_success_rate,
                'avg_duration_s': avg_task_duration,
            },
            'energy': {
                'total_joules': total_energy,
                'avg_per_sample_joules': avg_energy_per_sample,
            },
            'collisions': {
                'total': collision_count,
                'high_severity': high_severity_collisions,
            },
            'path_efficiency': {
                'average': avg_path_efficiency,
                'samples': len(path_efficiencies),
            },
            'sensor_drops': {
                'total': total_sensor_drops,
                'drop_rate': drop_rate,
            },
            'active_time_per_agv': dict(self.active_time_per_agv),
        }


# ============================================================================
# 动态障碍物生成器
# ============================================================================


class DynamicObstacleGenerator:
    """
    动态障碍物生成器 - 生成真实场景中的移动障碍物
    
    支持障碍物类型:
    - 行走的人员 (随机路径/固定路径)
    - 移动的叉车/其他AGV
    - 随机掉落的货物
    - 临时放置的障碍物
    """

    @dataclass
    class Obstacle:
        obstacle_id: str
        obstacle_type: str  # person, forklift, box, barrier
        position: np.ndarray
        velocity: np.ndarray
        size: Tuple[float, float, float]  # (width, height, depth)
        moving: bool = True
        path: Optional[List[np.ndarray]] = None
        current_path_index: int = 0

    def __init__(self, scene_bounds: Tuple[float, float, float, float]):
        """
        初始化动态障碍物生成器
        
        Args:
            scene_bounds: 场景边界 (x_min, y_min, x_max, y_max)
        """
        self.scene_bounds = scene_bounds
        self.obstacles: Dict[str, DynamicObstacleGenerator.Obstacle] = {}
        self.obstacle_counter = 0

    def generate_person_obstacle(self, start_pos: Optional[np.ndarray] = None, path: Optional[List[np.ndarray]] = None) -> str:
        """生成人员障碍物"""
        if start_pos is None:
            x = random.uniform(self.scene_bounds[0], self.scene_bounds[2])
            y = random.uniform(self.scene_bounds[1], self.scene_bounds[3])
            start_pos = np.array([x, y, 0.0])
        
        obstacle_id = f"person_{self.obstacle_counter}"
        self.obstacle_counter += 1
        
        velocity = np.array([random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), 0.0]) if path is None else np.zeros(3)
        
        obstacle = self.Obstacle(
            obstacle_id=obstacle_id,
            obstacle_type="person",
            position=start_pos,
            velocity=velocity,
            size=(0.4, 1.7, 0.3),
            path=path,
            current_path_index=0
        )
        
        self.obstacles[obstacle_id] = obstacle
        return obstacle_id

    def generate_forklift_obstacle(self, start_pos: Optional[np.ndarray] = None) -> str:
        """生成叉车障碍物"""
        if start_pos is None:
            x = random.uniform(self.scene_bounds[0], self.scene_bounds[2])
            y = random.uniform(self.scene_bounds[1], self.scene_bounds[3])
            start_pos = np.array([x, y, 0.0])
        
        obstacle_id = f"forklift_{self.obstacle_counter}"
        self.obstacle_counter += 1
        
        obstacle = self.Obstacle(
            obstacle_id=obstacle_id,
            obstacle_type="forklift",
            position=start_pos,
            velocity=np.array([random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), 0.0]),
            size=(1.2, 2.0, 0.8)
        )
        
        self.obstacles[obstacle_id] = obstacle
        return obstacle_id

    def generate_random_box_obstacle(self) -> str:
        """生成随机掉落的箱子障碍物"""
        x = random.uniform(self.scene_bounds[0], self.scene_bounds[2])
        y = random.uniform(self.scene_bounds[1], self.scene_bounds[3])
        position = np.array([x, y, random.uniform(0.0, 0.5)])
        
        obstacle_id = f"box_{self.obstacle_counter}"
        self.obstacle_counter += 1
        
        obstacle = self.Obstacle(
            obstacle_id=obstacle_id,
            obstacle_type="box",
            position=position,
            velocity=np.zeros(3),
            size=(random.uniform(0.2, 0.6), random.uniform(0.1, 0.4), random.uniform(0.2, 0.6)),
            moving=False
        )
        
        self.obstacles[obstacle_id] = obstacle
        return obstacle_id

    def step(self, delta_time: float = 0.01) -> None:
        """更新所有障碍物位置"""
        for obstacle in self.obstacles.values():
            if not obstacle.moving:
                continue
            
            if obstacle.path is not None and len(obstacle.path) > 0:
                # 沿路径移动
                target = obstacle.path[obstacle.current_path_index]
                direction = target - obstacle.position[:2]
                distance = np.linalg.norm(direction)
                
                if distance < 0.1:
                    obstacle.current_path_index = (obstacle.current_path_index + 1) % len(obstacle.path)
                    target = obstacle.path[obstacle.current_path_index]
                    direction = target - obstacle.position[:2]
                    distance = np.linalg.norm(direction)
                
                if distance > 0:
                    direction = direction / distance
                    speed = 0.8 if obstacle.obstacle_type == "person" else 1.5
                    obstacle.velocity[:2] = direction * speed
            
            # 更新位置
            obstacle.position += obstacle.velocity * delta_time
            
            # 边界检查，碰到边界反弹
            if obstacle.position[0] < self.scene_bounds[0] or obstacle.position[0] > self.scene_bounds[2]:
                obstacle.velocity[0] *= -1
            if obstacle.position[1] < self.scene_bounds[1] or obstacle.position[1] > self.scene_bounds[3]:
                obstacle.velocity[1] *= -1

    def get_all_obstacles(self) -> List[Dict[str, Any]]:
        """获取所有障碍物信息"""
        return [
            {
                'id': obs.obstacle_id,
                'type': obs.obstacle_type,
                'position': obs.position.tolist(),
                'velocity': obs.velocity.tolist(),
                'size': obs.size,
                'moving': obs.moving
            }
            for obs in self.obstacles.values()
        ]

    def check_collision(self, agv_position: np.ndarray, agv_size: Tuple[float, float, float]) -> List[str]:
        """检查AGV是否与障碍物碰撞"""
        collisions = []
        agv_half_size = np.array(agv_size) / 2
        agv_min = agv_position - agv_half_size
        agv_max = agv_position + agv_half_size
        
        for obs in self.obstacles.values():
            obs_half_size = np.array(obs.size) / 2
            obs_min = obs.position - obs_half_size
            obs_max = obs.position + obs_half_size
            
            # AABB碰撞检测
            if (agv_min[0] < obs_max[0] and agv_max[0] > obs_min[0] and
                agv_min[1] < obs_max[1] and agv_max[1] > obs_min[1] and
                agv_min[2] < obs_max[2] and agv_max[2] > obs_min[2]):
                collisions.append(obs.obstacle_id)
        
        return collisions


# ============================================================================
# 环境条件模拟器
# ============================================================================


class EnvironmentalConditionSimulator:
    """
    环境条件模拟器 - 模拟影响传感器性能的环境变化
    
    支持模拟:
    - 光照变化 (影响视觉传感器)
    - 粉尘/烟雾 (降低视觉清晰度)
    - 雨水/潮湿 (影响触觉/力觉传感器精度)
    - 电磁干扰 (影响IMU/无线通信)
    - 温度变化 (影响电机/传感器性能)
    """

    def __init__(self):
        self.lighting_intensity: float = 1000.0  # 光照强度 (lux), 正常室内500-2000
        self.dust_density: float = 0.0  # 粉尘密度 0-1
        self.rain_intensity: float = 0.0  # 降雨强度 0-1
        self.emi_level: float = 0.0  # 电磁干扰等级 0-1
        self.temperature: float = 25.0  # 温度 (°C)
        self.humidity: float = 50.0  # 湿度 (%)

    def set_normal_conditions(self) -> None:
        """设置正常室内环境条件"""
        self.lighting_intensity = 1000.0
        self.dust_density = 0.0
        self.rain_intensity = 0.0
        self.emi_level = 0.0
        self.temperature = 25.0
        self.humidity = 50.0

    def set_low_light(self) -> None:
        """设置低光照条件"""
        self.lighting_intensity = random.uniform(50.0, 200.0)

    def set_dusty_environment(self) -> None:
        """设置多粉尘环境 (工厂/仓库场景)"""
        self.dust_density = random.uniform(0.3, 0.7)

    def set_rainy_outdoor(self) -> None:
        """设置室外雨天条件"""
        self.rain_intensity = random.uniform(0.2, 0.8)
        self.humidity = random.uniform(80.0, 95.0)

    def set_high_emi(self) -> None:
        """设置高电磁干扰环境 (靠近电机/高压设备)"""
        self.emi_level = random.uniform(0.4, 0.9)

    def set_extreme_temperature(self, hot: bool = True) -> None:
        """设置极端温度条件"""
        if hot:
            self.temperature = random.uniform(40.0, 55.0)
        else:
            self.temperature = random.uniform(-10.0, 10.0)

    def get_camera_performance_factor(self) -> float:
        """获取摄像头性能系数 (0-1, 1=最佳性能)"""
        # 光照影响：过暗或过亮都会降低性能
        light_factor = 1.0
        if self.lighting_intensity < 100.0:
            light_factor = max(0.2, self.lighting_intensity / 100.0)
        elif self.lighting_intensity > 10000.0:
            light_factor = max(0.3, 10000.0 / self.lighting_intensity)
        
        # 粉尘影响：降低清晰度
        dust_factor = 1.0 - (self.dust_density * 0.6)
        
        # 雨水影响：镜头起雾/水滴
        rain_factor = 1.0 - (self.rain_intensity * 0.7)
        
        return min(light_factor, dust_factor, rain_factor)

    def get_tactile_sensor_performance_factor(self) -> float:
        """获取触觉传感器性能系数"""
        # 湿度影响：潮湿会降低触觉精度
        humidity_factor = 1.0
        if self.humidity > 70.0:
            humidity_factor = 1.0 - ((self.humidity - 70.0) / 30.0 * 0.4)
        
        # 温度影响：极端温度降低精度
        temp_factor = 1.0
        if abs(self.temperature - 25.0) > 20.0:
            temp_factor = 1.0 - (abs(self.temperature - 25.0) - 20.0) / 30.0 * 0.5
            temp_factor = max(0.5, temp_factor)
        
        return min(humidity_factor, temp_factor)

    def get_imu_performance_factor(self) -> float:
        """获取IMU性能系数"""
        # 电磁干扰影响
        emi_factor = 1.0 - (self.emi_level * 0.7)
        
        # 温度影响
        temp_factor = 1.0
        if abs(self.temperature - 25.0) > 15.0:
            temp_factor = 1.0 - (abs(self.temperature - 25.0) - 15.0) / 35.0 * 0.4
            temp_factor = max(0.6, temp_factor)
        
        return min(emi_factor, temp_factor)

    def get_communication_reliability(self) -> float:
        """获取通信可靠性系数 (0-1, 1=完全可靠)"""
        # 电磁干扰影响通信
        emi_factor = 1.0 - (self.emi_level * 0.8)
        
        # 雨水影响无线信号
        rain_factor = 1.0 - (self.rain_intensity * 0.3)
        
        return min(emi_factor, rain_factor)

    def step(self, delta_time: float = 1.0) -> None:
        """模拟环境条件的缓慢变化"""
        # 光照缓慢波动
        self.lighting_intensity += random.uniform(-50.0, 50.0) * delta_time
        self.lighting_intensity = max(50.0, min(20000.0, self.lighting_intensity))
        
        # 粉尘缓慢变化
        self.dust_density += random.uniform(-0.01, 0.01) * delta_time
        self.dust_density = max(0.0, min(1.0, self.dust_density))
        
        # 温度缓慢波动
        self.temperature += random.uniform(-0.1, 0.1) * delta_time
        self.temperature = max(-20.0, min(60.0, self.temperature))

    def get_environmental_status(self) -> Dict[str, Any]:
        """获取当前环境状态"""
        return {
            'lighting_intensity_lux': self.lighting_intensity,
            'dust_density': self.dust_density,
            'rain_intensity': self.rain_intensity,
            'emi_level': self.emi_level,
            'temperature_celsius': self.temperature,
            'humidity_percent': self.humidity,
            'camera_performance': self.get_camera_performance_factor(),
            'tactile_performance': self.get_tactile_sensor_performance_factor(),
            'imu_performance': self.get_imu_performance_factor(),
            'communication_reliability': self.get_communication_reliability(),
        }


# ============================================================================
# 地形建模系统
# ============================================================================


class FloorType(Enum):
    """地板类型枚举"""
    SMOOTH_CONCRETE = "smooth_concrete"     # 光滑混凝土 (最理想)
    ROUGH_CONCRETE = "rough_concrete"        # 粗糙混凝土 (仓库常用)
    EPOXY_COATING = "epoxy_coating"           # 环氧树脂涂层 (防滑)
    RUBBER_MAT = "rubber_mat"                # 橡胶垫 (静音)
    OUTDOOR_ASPHALT = "outdoor_asphalt"      # 室外沥青
    GRASS = "grass"                          # 草地 (户外)
    METAL_PLATE = "metal_plate"              # 金属板 (车间)
    UNKNOWN = "unknown"


class TerrainRegion:
    """地形区域 - 描述地图中的一个区域的地形特征"""
    
    def __init__(
        self,
        region_id: str,
        floor_type: FloorType,
        center: Tuple[float, float],
        radius: float = 2.0,
        slope_angle: float = 0.0,          # 斜坡角度 (度)
        slope_direction: float = 0.0,       # 斜坡方向 (度, 0=正北)
        unevenness: float = 0.0,            # 不平整度 0-1
        is_wet: bool = False,
    ):
        self.region_id = region_id
        self.floor_type = floor_type
        self.center = center  # (x, y)
        self.radius = radius
        self.slope_angle = slope_angle
        self.slope_direction = slope_direction  # degrees
        self.unevenness = unevenness
        self.is_wet = is_wet
    
    def contains_point(self, x: float, y: float) -> bool:
        """检查点是否在此区域内"""
        import math
        dx = x - self.center[0]
        dy = y - self.center[1]
        dist = math.sqrt(dx*dx + dy*dy)
        return dist <= self.radius
    
    def get_slope_vector(self) -> Tuple[float, float, float]:
        """获取斜坡方向向量 (vx, vy, vz)"""
        import math
        angle_rad = math.radians(self.slope_direction)
        # 斜坡在水平方向的投影
        vx = math.sin(math.radians(self.slope_angle)) * math.cos(angle_rad)
        vy = math.sin(math.radians(self.slope_angle)) * math.sin(angle_rad)
        vz = math.cos(math.radians(self.slope_angle))
        return (vx, vy, vz)


class TerrainModelingSystem:
    """
    地形建模系统 - 模拟不同地面材质、斜坡、不平整地对AGV运动的影响
    
    功能:
    - 地板类型识别与物理参数计算
    - 斜坡角度对导航的影响
    - 轮子打滑建模
    - 不同地形对速度/加速度的限制
    - 地形感知对导航路径规划的影响
    
    支持AGV五级规格:
    - M级: 基本室内地形
    - L级: 室内+轻量室外地形
    - XL级: 全地形支持
    - XXL级: 全地形+复杂地形
    """
    
    # 地板类型物理参数: (friction_factor, max_speed_factor, slip_factor)
    FLOOR_PARAMS = {
        FloorType.SMOOTH_CONCRETE:   (1.00, 1.00, 0.02),
        FloorType.ROUGH_CONCRETE:     (0.85, 0.90, 0.05),
        FloorType.EPOXY_COATING:      (0.90, 0.95, 0.03),
        FloorType.RUBBER_MAT:         (0.95, 0.80, 0.01),
        FloorType.OUTDOOR_ASPHALT:    (0.75, 0.85, 0.08),
        FloorType.GRASS:              (0.50, 0.50, 0.25),
        FloorType.METAL_PLATE:        (0.80, 0.90, 0.10),
        FloorType.UNKNOWN:            (0.70, 0.80, 0.10),
    }
    
    # 等级对应的地形支持列表
    GRADE_FLOOR_TYPES = {
        'S':  [FloorType.SMOOTH_CONCRETE],
        'M':  [FloorType.SMOOTH_CONCRETE, FloorType.ROUGH_CONCRETE, FloorType.EPOXY_COATING],
        'L':  [FloorType.SMOOTH_CONCRETE, FloorType.ROUGH_CONCRETE, FloorType.EPOXY_COATING,
               FloorType.RUBBER_MAT, FloorType.METAL_PLATE],
        'XL': [FloorType.SMOOTH_CONCRETE, FloorType.ROUGH_CONCRETE, FloorType.EPOXY_COATING,
               FloorType.RUBBER_MAT, FloorType.METAL_PLATE, FloorType.OUTDOOR_ASPHALT],
        'XXL': list(FloorType),
    }
    
    def __init__(
        self,
        grade: str = "M",
        enable_slope_modeling: bool = True,
        enable_slip_modeling: bool = True,
        enable_wet_surface: bool = True,
    ):
        self.grade = grade
        self.enable_slope_modeling = enable_slope_modeling
        self.enable_slip_modeling = enable_slip_modeling
        self.enable_wet_surface = enable_wet_surface
        
        self.regions: List[TerrainRegion] = []
        self.supported_floor_types = self.GRADE_FLOOR_TYPES.get(grade, [FloorType.SMOOTH_CONCRETE])
        
        # 默认参数 (在光滑混凝土上的理想参数)
        self.base_max_speed = 2.0        # m/s
        self.base_max_accel = 1.0        # m/s^2
        self.base_friction = 0.8
        
        # 当前地形状态
        self.current_region: Optional[TerrainRegion] = None
        self.last_update_position: Tuple[float, float] = (0.0, 0.0)
    
    def add_region(self, region: TerrainRegion) -> None:
        """添加地形区域"""
        # 检查AGV等级是否支持该地形
        if region.floor_type not in self.supported_floor_types:
            logger.warning(
                f"AGV grade {self.grade} does not support floor type "
                f"{region.floor_type.value}. Skipping region {region.region_id}"
            )
            return
        self.regions.append(region)
    
    def add_region_simple(
        self,
        region_id: str,
        floor_type: FloorType,
        center: Tuple[float, float],
        radius: float = 2.0,
        slope_angle: float = 0.0,
        slope_direction: float = 0.0,
    ) -> None:
        """简化接口：直接添加地形区域"""
        region = TerrainRegion(
            region_id=region_id,
            floor_type=floor_type,
            center=center,
            radius=radius,
            slope_angle=slope_angle,
            slope_direction=slope_direction,
        )
        self.add_region(region)
    
    def get_region_at(self, x: float, y: float) -> Optional[TerrainRegion]:
        """获取指定坐标的地形区域"""
        for region in self.regions:
            if region.contains_point(x, y):
                return region
        return None
    
    def update_position(self, x: float, y: float) -> None:
        """更新AGV位置，检测当前地形区域"""
        self.current_region = self.get_region_at(x, y)
        self.last_update_position = (x, y)
    
    def get_current_floor_type(self) -> FloorType:
        """获取当前地板类型"""
        if self.current_region:
            return self.current_region.floor_type
        return FloorType.SMOOTH_CONCRETE  # 假设默认地板
    
    def get_floor_parameters(self, floor_type: Optional[FloorType] = None) -> Dict[str, float]:
        """
        获取指定地板类型的物理参数
        
        Returns:
            dict with keys: friction_factor, max_speed_factor, slip_factor
        """
        if floor_type is None:
            floor_type = self.get_current_floor_type()
        
        params = self.FLOOR_PARAMS.get(floor_type, self.FLOOR_PARAMS[FloorType.UNKNOWN])
        return {
            'friction_factor': params[0],
            'max_speed_factor': params[1],
            'slip_factor': params[2],
        }
    
    def get_effective_max_speed(self) -> float:
        """获取考虑地形后的最大有效速度"""
        floor_params = self.get_floor_parameters()
        speed_factor = floor_params['max_speed_factor']
        
        # 斜坡影响：上坡降低速度
        if self.enable_slope_modeling and self.current_region:
            slope_angle = self.current_region.slope_angle
            if slope_angle > 0:
                # 上坡: 角度越大，速度折扣越大
                slope_factor = max(0.3, 1.0 - slope_angle / 45.0)
                speed_factor *= slope_factor
        
        return self.base_max_speed * speed_factor
    
    def get_effective_max_acceleration(self) -> float:
        """获取考虑地形后的最大有效加速度"""
        floor_params = self.get_floor_parameters()
        friction = floor_params['friction_factor']
        
        # 摩擦系数直接影响可用的加速度
        accel_factor = friction
        
        # 斜坡影响: 上坡降低有效加速度
        if self.enable_slope_modeling and self.current_region:
            slope_angle = self.current_region.slope_angle
            if slope_angle > 5.0:  # 超过5度才考虑
                slope_factor = max(0.2, 1.0 - (slope_angle - 5.0) / 40.0)
                accel_factor *= slope_factor
        
        return self.base_max_accel * accel_factor
    
    def get_slip_probability(
        self,
        velocity: Tuple[float, float],
        load_mass: float = 0.0,
    ) -> float:
        """
        计算当前地形下的打滑概率
        
        Args:
            velocity: 当前速度 (vx, vy) m/s
            load_mass: 负载质量 kg
            
        Returns:
            打滑概率 0-1
        """
        if not self.enable_slip_modeling:
            return 0.0
        
        floor_params = self.get_floor_parameters()
        base_slip = floor_params['slip_factor']
        
        # 速度越高，打滑概率越高 (非线性)
        speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
        speed_factor = min(2.0, speed / self.base_max_speed)
        
        # 负载增加打滑风险
        load_factor = 1.0 + (load_mass / 500.0) * 0.5  # 500kg负载增加50%风险
        
        # 湿滑地面大幅增加打滑
        wet_factor = 2.5 if (self.current_region and self.current_region.is_wet) else 1.0
        
        # 坡度影响
        slope_factor = 1.0
        if self.current_region and self.current_region.slope_angle > 3.0:
            slope_factor = 1.0 + (self.current_region.slope_angle - 3.0) / 30.0
        
        slip_prob = base_slip * speed_factor * load_factor * wet_factor * slope_factor
        return min(0.95, slip_prob)  # 最高95%
    
    def get_gravity_slope_force(self) -> Tuple[float, float]:
        """
        获取斜坡产生的重力分力 (沿坡面方向)
        
        Returns:
            (force_x, force_y) 沿坡面方向的分力
        """
        if not self.enable_slope_modeling or not self.current_region:
            return (0.0, 0.0)
        
        slope_vector = self.current_region.get_slope_vector()
        # 返回水平方向的分力
        return (slope_vector[0], slope_vector[1])
    
    def apply_slip_noise(
        self,
        intended_velocity: Tuple[float, float, float],
        slip_probability: float,
    ) -> Tuple[float, float, float]:
        """
        对速度应用打滑噪声
        
        Args:
            intended_velocity: 目标速度 (vx, vy, vz)
            slip_probability: 打滑概率
            
        Returns:
            带打滑噪声的实际速度
        """
        if random.random() > slip_probability:
            return intended_velocity
        
        # 打滑时，速度方向发生偏移
        vx, vy, vz = intended_velocity
        
        # 随机偏移角度 (0-45度)
        slip_angle = random.uniform(0, math.pi / 4)
        slip_direction = random.uniform(0, 2 * math.pi)
        
        # 偏移后的速度
        speed = math.sqrt(vx*vx + vy*vy + vz*vz)
        new_vx = speed * math.cos(slip_direction) * math.cos(slip_angle) - speed * math.sin(slip_direction) * math.sin(slip_angle)
        new_vy = speed * math.sin(slip_direction) * math.cos(slip_angle) + speed * math.cos(slip_direction) * math.sin(slip_angle)
        
        # 打滑时速度衰减
        slip_magnitude = random.uniform(0.3, 0.8)
        new_vx *= slip_magnitude
        new_vy *= slip_magnitude
        
        return (new_vx, new_vy, vz)
    
    def check_slope_safety(self, max_safe_angle: float = 15.0) -> Tuple[bool, str]:
        """
        检查当前位置斜坡是否在安全范围内
        
        Returns:
            (is_safe, message)
        """
        if not self.current_region:
            return (True, "No slope information available")
        
        angle = self.current_region.slope_angle
        if angle > max_safe_angle:
            return (False, f"Slope angle {angle:.1f}° exceeds safe limit {max_safe_angle}°")
        elif angle > 10.0:
            return (True, f"Slope warning: {angle:.1f}° (approaching limit)")
        
        return (True, f"Slope OK: {angle:.1f}°")
    
    def get_terrain_status(self, x: float, y: float) -> Dict[str, Any]:
        """
        获取指定位置的地形状态
        
        Returns:
            包含完整地形信息的字典
        """
        self.update_position(x, y)
        
        floor_type = self.get_current_floor_type()
        floor_params = self.get_floor_parameters(floor_type)
        
        result = {
            'floor_type': floor_type.value,
            'friction_factor': floor_params['friction_factor'],
            'max_speed_factor': floor_params['max_speed_factor'],
            'slip_factor': floor_params['slip_factor'],
            'effective_max_speed': self.get_effective_max_speed(),
            'effective_max_accel': self.get_effective_max_acceleration(),
        }
        
        if self.current_region:
            result.update({
                'slope_angle': self.current_region.slope_angle,
                'slope_direction': self.current_region.slope_direction,
                'unevenness': self.current_region.unevenness,
                'is_wet': self.current_region.is_wet,
            })
            is_safe, msg = self.check_slope_safety()
            result['slope_safety'] = {'is_safe': is_safe, 'message': msg}
        else:
            result.update({
                'slope_angle': 0.0,
                'slope_direction': 0.0,
                'unevenness': 0.0,
                'is_wet': False,
                'slope_safety': {'is_safe': True, 'message': 'Unknown terrain'},
            })
        
        return result
    
    def generate_warehouse_layout(
        self,
        seed: int = 42,
    ) -> None:
        """
        生成标准仓库地形布局
        
        布局:
        - 入库区: 光滑混凝土 + 卸货斜坡
        - 存储区: 环氧树脂涂层 (平整)
        - 出库区: 粗糙混凝土
        - 通道: 环氧树脂涂层
        """
        random.seed(seed)
        
        self.regions.clear()
        
        # 入库区
        self.add_region_simple("inbound_concrete", FloorType.SMOOTH_CONCRETE, (0, 0), radius=5.0)
        self.add_region_simple("inbound_ramp", FloorType.ROUGH_CONCRETE, (6, 0), radius=2.0, slope_angle=8.0, slope_direction=0.0)
        
        # 主通道
        self.add_region_simple("main_aisle", FloorType.EPOXY_COATING, (10, 0), radius=3.0)
        
        # 货架区
        for row in range(3):
            for col in range(5):
                region_id = f"storage_{row}_{col}"
                cx = 15 + col * 6
                cy = -20 + row * 20
                self.add_region_simple(region_id, FloorType.EPOXY_COATING, (cx, cy), radius=2.5)
        
        # 出库区
        self.add_region_simple("outbound_rough", FloorType.ROUGH_CONCRETE, (45, 0), radius=5.0)
    
    def get_supported_floor_types(self) -> List[str]:
        """获取AGV等级支持的地板类型列表"""
        return [ft.value for ft in self.supported_floor_types]


__all__ += [
    'MultiAGVSimulationEnhancer',
    'SimulationMetricsCollector',
    'DynamicObstacleGenerator',
    'EnvironmentalConditionSimulator',
    'TerrainModelingSystem',
    'TerrainRegion',
    'FloorType',
]
