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
import logging
import random
import math

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
