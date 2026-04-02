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
            return (0.5, 0.5)  # 默认中心

        # 找到最大压力位置
        rows, cols = self.taxel_data.shape
        max_idx = np.unravel_index(np.argmax(self.taxel_data), self.taxel_data.shape)
        return (max_idx[1] / cols, max_idx[0] / rows)

    def get_contact_area(self, threshold: float = 0.1) -> float:
        """获取接触面积 (归一化)"""
        if self.taxel_data is None or self.taxel_data.size == 0:
            return 0.0

        normalized = self.taxel_data / (self.taxel_data.max() + 1e-6)
        contact_ratio = np.sum(normalized > threshold) / self.taxel_data.size
        return contact_ratio


class TactileSensor(ABC):
    """触觉传感器基类"""

    def __init__(self, sensor_id: str, name: str = "TactileSensor"):
        self.sensor_id = sensor_id
        self.name = name
        self._calibrated = False
        self._calibration_offset = 0.0
        self._sensitivity = 1.0  # V/Pa 或 counts/Pa

    @abstractmethod
    def read(self, timestamp: float) -> TactileData:
        """读取传感器数据"""
        pass

    def calibrate(self, reference_data: np.ndarray) -> bool:
        """
        校准传感器

        Args:
            reference_data: 参考压力值 (Pa)

        Returns:
            校准是否成功
        """
        if len(reference_data) < 10:
            return False

        self._calibration_offset = np.mean(reference_data)
        self._calibrated = True
        return True

    def get_sensitivity(self) -> float:
        """获取灵敏度"""
        return self._sensitivity

    def reset_calibration(self):
        """重置校准"""
        self._calibration_offset = 0.0
        self._calibrated = False


class PressureSensor(TactileSensor):
    """
    压阻式压力传感器
    适用于: 脚踏开关、碰撞检测、负载监测
    """

    def __init__(
        self,
        sensor_id: str,
        name: str = "PressureSensor",
        max_pressure: float = 1000.0,  # Pa
        resolution: float = 0.01,  # Pa/bit
        noise_floor: float = 0.1  # Pa RMS
    ):
        super().__init__(sensor_id, name)
        self.max_pressure = max_pressure
        self.resolution = resolution
        self.noise_floor = noise_floor
        self._sensitivity = 1.0 / resolution

        # 模拟内部状态
        self._current_pressure = 0.0

    def read(self, timestamp: float) -> TactileData:
        """读取压力数据"""
        # 模拟读取 (实际应用中从 ADC/I2C/SPI 读取)
        # 添加一些噪声
        noise = np.random.normal(0, self.noise_floor)
        raw_pressure = self._current_pressure + noise

        # 应用校准偏移
        if self._calibrated:
            raw_pressure -= self._calibration_offset

        # 限幅
        raw_pressure = np.clip(raw_pressure, 0, self.max_pressure)

        return TactileData(
            sensor_id=self.sensor_id,
            timestamp=timestamp,
            pressure=raw_pressure,
            raw_values=np.array([raw_pressure])
        )

    def set_pressure(self, pressure: float):
        """模拟设置压力值 (用于测试)"""
        self._current_pressure = np.clip(pressure, 0, self.max_pressure)

    def apply_force(self, force: float, area: float = 0.01):
        """施加力并转换为压力 (模拟)"""
        # P = F / A
        pressure = force / area if area > 0 else 0
        self.set_pressure(pressure)


class TaxelArray(TactileSensor):
    """
    触感阵列 (仿生皮肤)
    16x16 或可配置的 taxel 网格
    """

    def __init__(
        self,
        sensor_id: str,
        name: str = "TaxelArray",
        rows: int = 16,
        cols: int = 16,
        max_pressure: float = 1000.0,  # Pa
        response_time: float = 0.001  # s (<1ms)
    ):
        super().__init__(sensor_id, name)
        self.rows = rows
        self.cols = cols
        self.max_pressure = max_pressure
        self.response_time = response_time
        self._sensitivity = 255.0 / max_pressure

        # 初始化 taxel 矩阵 (存储压力值 Pa)
        self._taxel_matrix = np.zeros((rows, cols), dtype=np.float32)
        self._last_taxel_matrix = np.zeros((rows, cols), dtype=np.float32)

    def read(self, timestamp: float) -> TactileData:
        """读取触感阵列数据"""
        # 模拟读取 (实际应用中从传感器读取)
        # 添加热噪声
        noise = np.random.normal(0, 0.5, (self.rows, self.cols))
        self._taxel_matrix = np.clip(self._taxel_matrix + noise, 0, self.max_pressure)

        # 应用校准
        if self._calibrated:
            self._taxel_matrix -= self._calibration_offset

        # 转换为 8-bit 值用于存储
        normalized = (self._taxel_matrix / self.max_pressure * 255).astype(np.uint8)

        # 计算振动分量 (帧间差分)
        vibration = np.abs(self._taxel_matrix - self._last_taxel_matrix).mean()
        self._last_taxel_matrix = self._taxel_matrix.copy()

        # 平均压力
        avg_pressure = self._taxel_matrix.mean()

        return TactileData(
            sensor_id=self.sensor_id,
            timestamp=timestamp,
            pressure=avg_pressure,
            taxel_data=normalized,
            vibration=vibration,
            raw_values=self._taxel_matrix.flatten()
        )

    def set_taxel_pressure(self, row: int, col: int, pressure: float):
        """设置指定 taxel 的压力 (模拟, 用于测试)"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self._taxel_matrix[row, col] = np.clip(pressure, 0, self.max_pressure)

    def apply_contact(self, center_row: float, center_col: float,
                      radius: float, pressure: float):
        """
        模拟接触 (高斯分布压力)

        Args:
            center_row: 接触中心行 (归一化 0-1)
            center_col: 接触中心列 (归一化 0-1)
            radius: 接触半径 (归一化 0-1)
            pressure: 接触压力 (Pa)
        """
        # 创建高斯分布
        rows_idx, cols_idx = np.meshgrid(
            np.arange(self.rows), np.arange(self.cols), indexing='ij'
        )

        # 中心点
        cr = int(center_row * (self.rows - 1))
        cc = int(center_col * (self.cols - 1))

        # 计算距离
        distances = np.sqrt(
            ((rows_idx - cr) / (radius * self.rows))**2 +
            ((cols_idx - cc) / (radius * self.cols))**2
        )

        # 高斯权重
        gaussian = np.exp(-distances**2 / 2)

        # 应用压力
        self._taxel_matrix += pressure * gaussian

        # 限幅
        self._taxel_matrix = np.clip(self._taxel_matrix, 0, self.max_pressure)

    def clear(self):
        """清空所有 taxel 数据"""
        self._taxel_matrix.fill(0)
        self._last_taxel_matrix.fill(0)


class PiezoelectricSensor(TactileSensor):
    """
    压电式振动传感器
    适用于: 振动检测、纹理识别、动态触觉
    """

    def __init__(
        self,
        sensor_id: str,
        name: str = "PiezoelectricSensor",
        frequency_range: Tuple[float, float] = (0.1, 1000),  # Hz
        sampling_rate: float = 1000,  # Hz
        sensitivity: float = 1.0,  # V/(m/s) 或自定义
        output_impedance: float = 100e3  # Ohm
    ):
        super().__init__(sensor_id, name)
        self.frequency_range = frequency_range
        self.sampling_rate = sampling_rate
        self.output_impedance = output_impedance
        self._sensitivity = sensitivity

        # 模拟振动缓冲区
        self._vibration_buffer = np.zeros(int(sampling_rate))  # 1秒缓冲区
        self._buffer_index = 0
        self._current_vibration = 0.0

    def read(self, timestamp: float) -> TactileData:
        """读取振动数据"""
        # 模拟读取
        # 添加随机振动 (实际应用中从 ADC 读取)
        self._current_vibration = np.random.normal(0, 10)  # 模拟振动

        # 限幅到频率范围
        if self._current_vibration > self.frequency_range[1]:
            self._current_vibration = self.frequency_range[1]
        elif self._current_vibration < self.frequency_range[0]:
            self._current_vibration = 0

        # 存储到缓冲区
        self._vibration_buffer[self._buffer_index] = self._current_vibration
        self._buffer_index = (self._buffer_index + 1) % len(self._vibration_buffer)

        return TactileData(
            sensor_id=self.sensor_id,
            timestamp=timestamp,
            vibration=self._current_vibration,
            raw_values=self._vibration_buffer.copy()
        )

    def add_vibration(self, frequency: float, amplitude: float):
        """
        添加振动信号 (模拟, 用于测试)

        Args:
            frequency: 频率 (Hz)
            amplitude: 幅度
        """
        if self.frequency_range[0] <= frequency <= self.frequency_range[1]:
            t = np.arange(len(self._vibration_buffer)) / self.sampling_rate
            wave = amplitude * np.sin(2 * np.pi * frequency * t)
            self._vibration_buffer += wave
            self._current_vibration = amplitude

    def get_spectrum(self) -> np.ndarray:
        """获取振动频谱"""
        return np.abs(np.fft.rfft(self._vibration_buffer))

    def detect_frequency(self) -> float:
        """检测主频率"""
        spectrum = self.get_spectrum()
        peak_idx = np.argmax(spectrum[1:]) + 1  # 跳过 DC
        return peak_idx * self.sampling_rate / len(self._vibration_buffer)


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

        # 简单拼接所有传感器的特征向量
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
