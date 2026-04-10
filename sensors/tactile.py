"""
触觉传感器模块 (Tactile Sensors)
支持压阻式压力传感器、触感阵列、压电式振动传感器
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class TactileData:
    """触觉数据"""
    sensor_id: str
    timestamp: float
    pressure: float = 0.0  # Pa (帕斯卡)
    taxel_data: Optional[np.ndarray] = None  # 触感阵列数据 [rows, cols]
    vibration: float = 0.0  # 振动幅度 (压电传感器)
    temperature: float = 25.0  # 温度 (°C)
    raw_values: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_vector(self) -> np.ndarray:
        """
        返回归一化特征向量
        格式: [pressure_norm, vibration_norm, temp_norm, ...taxel_flat...]
        """
        features = []

        # 压力归一化 (假设最大量程 1000 Pa)
        pressure_norm = np.clip(self.pressure / 1000.0, 0.0, 1.0)
        features.append(pressure_norm)

        # 振动归一化 (假设最大 1000 Hz)
        vibration_norm = np.clip(self.vibration / 1000.0, 0.0, 1.0)
        features.append(vibration_norm)

        # 温度归一化 (假设范围 0-50°C)
        temp_norm = np.clip(self.temperature / 50.0, 0.0, 1.0)
        features.append(temp_norm)

        # 触感阵列展平并归一化 (如果存在)
        if self.taxel_data is not None:
            taxel_norm = self.taxel_data.flatten() / 255.0 if self.taxel_data.max() > 1 else self.taxel_data.flatten()
            features.extend(taxel_norm.tolist())

        return np.array(features)

    def get_contact_center(self) -> Tuple[float, float]:
        """获取接触中心 (归一化坐标)"""
        if self.taxel_data is None or self.taxel_data.size == 0:
            return (0.0, 0.0)

        rows, cols = self.taxel_data.shape
        total = self.taxel_data.sum()
        if total < 1e-6:
            return (0.5, 0.5)

        # 质心计算
        row_indices, col_indices = np.indices(self.taxel_data.shape)
        cy = row_indices.flatten() @ self.taxel_data.flatten() / total / rows
        cx = col_indices.flatten() @ self.taxel_data.flatten() / total / cols

        return (float(cx), float(cy))

    def get_contact_area(self) -> float:
        """获取接触面积 (归一化)"""
        if self.taxel_data is None:
            return 0.0
        active = np.count_nonzero(self.taxel_data)
        return active / self.taxel_data.size


class TactileSensor(ABC):
    """触觉传感器基类"""

    def __init__(self, sensor_id: str):
        self.sensor_id = sensor_id
        self._bias = 0.0

    @abstractmethod
    def read(self, timestamp: float) -> TactileData:
        """读取传感器数据"""
        pass

    def set_bias(self, data: TactileData) -> None:
        """设置零偏"""
        self._bias = data.pressure

    def remove_bias(self) -> None:
        """移除零偏"""
        self._bias = 0.0


class PressureSensor(TactileSensor):
    """压阻式压力传感器"""

    def __init__(
        self,
        sensor_id: str,
        max_pressure: float = 1000.0,
        resolution: float = 0.1,
        sampling_rate: float = 100.0,
    ):
        super().__init__(sensor_id)
        self.max_pressure = max_pressure
        self.resolution = resolution
        self.sampling_rate = sampling_rate
        self._last_reading = 0.0

    def read(self, timestamp: float) -> TactileData:
        """读取压力数据 (带噪声和零偏)"""
        raw = np.random.normal(self._last_reading, self.resolution)
        raw = np.clip(raw, 0, self.max_pressure)
        self._last_reading = raw

        return TactileData(
            sensor_id=self.sensor_id,
            timestamp=timestamp,
            pressure=raw + self._bias,
            raw_values=np.array([raw]),
        )

    def set_resolution(self, resolution: float) -> None:
        """设置分辨率"""
        self.resolution = resolution


class TaxelArray(TactileSensor):
    """
    触感阵列 (Tactile Array)
    模拟电子皮肤多像素触觉感知
    """

    def __init__(
        self,
        name: str,
        rows: int = 16,
        cols: int = 16,
        resolution: float = 1.0,  # mm
        max_pressure: float = 1000.0,  # Pa
        sampling_rate: float = 100.0,
    ):
        super().__init__(name)
        self.name = name
        self.rows = rows
        self.cols = cols
        self.resolution = resolution
        self.max_pressure = max_pressure
        self.sampling_rate = sampling_rate
        self._taxels = np.zeros((rows, cols), dtype=np.float32)
        self._temperature = np.full((rows, cols), 25.0, dtype=np.float32)

    def simulate_contact(
        self,
        center: Tuple[int, int],
        radius: int = 2,
        pressure: float = 500.0,
    ) -> None:
        """
        模拟接触事件

        Args:
            center: 接触中心 (row, col)
            radius: 接触半径
            pressure: 压力值 (Pa)
        """
        row_c, col_c = center
        rows_i = np.arange(self.rows)
        cols_i = np.arange(self.cols)
        row_g, col_g = np.meshgrid(rows_i, cols_i, indexing='ij')

        dist = np.sqrt((row_g - row_c) ** 2 + (col_g - col_c) ** 2)
        contact = np.exp(-(dist ** 2) / (2 * radius ** 2)) * pressure
        self._taxels += contact.astype(np.float32)

    def simulate_proximity(self, distance_mm: float, object_pressure: float = 100.0) -> float:
        """
        模拟接近觉

        Args:
            distance_mm: 距离 (mm)
            object_pressure: 物体等效压力

        Returns:
            接近觉强度 (归一化 0-1)
        """
        if distance_mm <= 0:
            return 1.0
        # 线性衰减，10mm外无响应
        return np.clip(1.0 - distance_mm / 10.0, 0.0, 1.0) * (object_pressure / self.max_pressure)

    def simulate_slip(self, force_direction: Tuple[float, float], force_magnitude: float) -> bool:
        """
        模拟滑移检测

        Args:
            force_direction: 滑移力方向 (dx, dy)
            force_magnitude: 滑移力大小

        Returns:
            是否检测到滑移
        """
        # 库仑摩擦模型: 静摩擦阈值 = μ_s * F_n, 动摩擦 = μ_k * F_n
        mu_static = 0.6
        mu_dynamic = 0.4
        # 找最大正压力
        max_normal = self._taxels.max() if self._taxels.max() > 0 else 1.0
        slip_threshold = mu_static * max_normal
        return force_magnitude > slip_threshold

    def read(self, timestamp: float) -> TactileData:
        """读取触感阵列数据"""
        noise = np.random.normal(0, 0.5, self._taxels.shape).astype(np.float32)
        sensed = np.clip(self._taxels + noise, 0, self.max_pressure)

        return TactileData(
            sensor_id=self.sensor_id,
            timestamp=timestamp,
            pressure=float(sensed.mean()),
            taxel_data=sensed,
            temperature=float(self._temperature.mean()),
            raw_values=sensed.flatten(),
        )

    def apply_temperature(self, temp: float) -> None:
        """设置温度分布"""
        self._temperature.fill(temp)

    def clear(self) -> None:
        """清除所有触觉数据"""
        self._taxels.fill(0.0)

    def get_pressure_map(self) -> np.ndarray:
        """获取压力图 (带噪声)"""
        noise = np.random.normal(0, 0.5, self._taxels.shape).astype(np.float32)
        return np.clip(self._taxels + noise, 0, self.max_pressure)

    def get_grasp_quality(self) -> Dict[str, float]:
        """
        评估抓取质量

        Returns:
            抓取质量指标字典
        """
        if self._taxels.max() < 1.0:
            return {'quality': 0.0, 'stability': 0.0, 'contact_ratio': 0.0}

        # 接触比例
        active = np.count_nonzero(self._taxels)
        contact_ratio = active / self._taxels.size

        # 压力分布均匀性 (标准差越小越均匀)
        flat = self._taxels.flatten()
        non_zero = flat[flat > 0]
        uniformity = 1.0 / (1.0 + np.std(non_zero)) if len(non_zero) > 1 else 0.0

        # 综合质量
        quality = contact_ratio * uniformity * (min(1.0, self._taxels.max() / self.max_pressure))

        return {
            'quality': float(quality),
            'stability': float(uniformity),
            'contact_ratio': float(contact_ratio),
        }


class PiezoelectricSensor(TactileSensor):
    """压电式振动传感器"""

    def __init__(
        self,
        sensor_id: str,
        sensitivity: float = 50.0,  # mV/N
        freq_range: Tuple[float, float] = (1, 1000),  # Hz
        sampling_rate: float = 5000.0,
    ):
        super().__init__(sensor_id)
        self.sensitivity = sensitivity
        self.freq_range = freq_range
        self.sampling_rate = sampling_rate
        self._last_vibration = 0.0

    def read(self, timestamp: float) -> TactileData:
        """读取振动数据"""
        vibration = np.random.normal(self._last_vibration, 10.0)
        self._last_vibration = vibration * 0.9  # 衰减
        vibration = np.clip(vibration, 0, 1000)

        return TactileData(
            sensor_id=self.sensor_id,
            timestamp=timestamp,
            vibration=vibration,
            raw_values=np.array([vibration]),
        )


class TactileArray:
    """
    触觉传感器阵列管理器
    管理多个触觉传感器
    """

    def __init__(self, name: str = "TactileArray"):
        self.name = name
        self._sensors: Dict[str, TactileSensor] = {}

    def add_sensor(self, sensor: TactileSensor) -> bool:
        """
        添加传感器

        Args:
            sensor: TactileSensor 实例

        Returns:
            添加是否成功
        """
        if sensor.sensor_id in self._sensors:
            return False

        self._sensors[sensor.sensor_id] = sensor
        return True

    def remove_sensor(self, sensor_id: str) -> bool:
        """移除传感器"""
        if sensor_id in self._sensors:
            del self._sensors[sensor_id]
            return True
        return False

    def read_all(self, timestamp: float) -> List[TactileData]:
        """读取所有传感器数据"""
        return [sensor.read(timestamp) for sensor in self._sensors.values()]

    def get_fusion_data(self, timestamp: float) -> np.ndarray:
        """
        获取融合后的特征向量

        Returns:
            归一化特征向量
        """
        all_data = self.read_all(timestamp)

        if not all_data:
            return np.array([])

        vectors = [data.to_vector() for data in all_data]
        return np.concatenate(vectors)

    def detect_touch_distribution(self) -> Dict[str, float]:
        """检测触觉分布统计"""
        distribution = {}

        for sensor_id, sensor in self._sensors.items():
            if isinstance(sensor, (PressureSensor, TaxelArray)):
                data = sensor.read(0.0)  # 使用当前时间戳
                distribution[sensor_id] = {
                    'pressure': data.pressure,
                    'contact_center': data.get_contact_center(),
                    'contact_area': data.get_contact_area() if data.taxel_data is not None else 0.0
                }

        return distribution

    def get_total_pressure(self, timestamp: float) -> float:
        """获取总压力"""
        all_data = self.read_all(timestamp)
        return sum(data.pressure for data in all_data)

    def detect_collision(self, threshold: float = 100.0) -> bool:
        """检测碰撞 (任意传感器超过阈值)"""
        timestamp = 0.0  # 使用当前时间
        all_data = self.read_all(timestamp)
        return any(data.pressure > threshold for data in all_data)

    def __len__(self) -> int:
        return len(self._sensors)

    def __repr__(self) -> str:
        return f"TactileArray(sensors={list(self._sensors.keys())})"


# =============================================================================
# AGV五级触觉传感器规格表
# =============================================================================

AGV_TACTILE_GRADES = {
    'S': {
        'name': '小型AGV',
        'array_size': (8, 8),
        'resolution_mm': 16,
        'pressure_range_kpa': (0, 500),
        'sampling_hz': 50,
        'temperature': False,
        'proximity': False,
        'slip_detection': False,
        'noise_kpa': 5.0,
        'taxels': 64,
        'typical_use': '轻载仓库AGV，碰撞检测',
    },
    'M': {
        'name': '中型AGV',
        'array_size': (16, 16),
        'resolution_mm': 8,
        'pressure_range_kpa': (0, 2000),
        'sampling_hz': 100,
        'temperature': True,
        'proximity': True,
        'slip_detection': True,
        'noise_kpa': 2.0,
        'taxels': 256,
        'typical_use': '物流分拣AGV，物料搬运',
    },
    'L': {
        'name': '大型AGV',
        'array_size': (24, 24),
        'resolution_mm': 5,
        'pressure_range_kpa': (0, 5000),
        'sampling_hz': 200,
        'temperature': True,
        'proximity': True,
        'slip_detection': True,
        'noise_kpa': 1.0,
        'taxels': 576,
        'typical_use': '产线配送AGV，重载物料输送',
    },
    'XL': {
        'name': '特大型AGV',
        'array_size': (32, 32),
        'resolution_mm': 4,
        'pressure_range_kpa': (0, 8000),
        'sampling_hz': 500,
        'temperature': True,
        'proximity': True,
        'slip_detection': True,
        'noise_kpa': 0.5,
        'taxels': 1024,
        'typical_use': '重载车间AGV，精密装配',
    },
    'XXL': {
        'name': '超大型AGV',
        'array_size': (48, 48),
        'resolution_mm': 2.5,
        'pressure_range_kpa': (0, 10000),
        'sampling_hz': 1000,
        'temperature': True,
        'proximity': True,
        'slip_detection': True,
        'noise_kpa': 0.2,
        'taxels': 2304,
        'typical_use': '港口物流AGV，极限负载场景',
    },
}


def get_tactile_spec(grade: str) -> dict:
    """
    获取AGV指定等级的触觉传感器规格

    Args:
        grade: AGV等级 (S/M/L/XL/XXL)

    Returns:
        触觉传感器规格字典
    """
    return AGV_TACTILE_GRADES.get(grade, AGV_TACTILE_GRADES['M'])


def create_tactile_sensor_for_grade(grade: str, sensor_id: str = "tactile_0") -> TactileSensor:
    """
    创建指定AGV等级的触觉传感器

    Args:
        grade: AGV等级 (S/M/L/XL/XXL)
        sensor_id: 传感器ID

    Returns:
        TactileSensor 实例 (TaxelArray)
    """
    spec = get_tactile_spec(grade)
    rows, cols = spec['array_size']

    return TaxelArray(
        name=sensor_id,
        rows=rows,
        cols=cols,
        resolution=spec['resolution_mm'],
        max_pressure=spec['pressure_range_kpa'][1] * 1000.0,  # kPa → Pa
        sampling_rate=spec['sampling_hz'],
    )


def list_tactile_capabilities() -> str:
    """列出所有AGV等级的触觉传感器能力"""
    lines = ["AGV五级触觉传感器能力表:"]
    lines.append(f"{'等级':<6} {'阵列':<10} {'分辨率':<10} {'量程(kPa)':<16} {'采样率':<10} {'温度':<6} {'接近觉':<6} {'滑移检测'}")
    lines.append("-" * 85)
    for grade, spec in AGV_TACTILE_GRADES.items():
        array_str = f"{spec['array_size'][0]}x{spec['array_size'][1]}"
        range_str = f"{spec['pressure_range_kpa'][0]}-{spec['pressure_range_kpa'][1]}"
        lines.append(
            f"{grade:<6} {array_str:<10} {spec['resolution_mm']}mm{'':<4} "
            f"{range_str:<16} {spec['sampling_hz']}Hz{'':<4} "
            f"{'Yes' if spec['temperature'] else 'No':<6} "
            f"{'Yes' if spec['proximity'] else 'No':<6} "
            f"{'Yes' if spec['slip_detection'] else 'No'}"
        )
    return "\n".join(lines)
