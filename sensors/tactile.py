"""
触觉传感器模块 (Tactile Sensor Module)
支持多种触觉传感器：压阻式、电容式、触感阵列
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class TactileType(Enum):
    """触觉传感器类型"""
    PRESSURE = "pressure"           # 压阻式
    CAPACITIVE = "capacitive"        # 电容式
    PIEZOELECTRIC = "piezoelectric"  # 压电式
    TAXEL_ARRAY = "taxel_array"     # 触感阵列(仿生皮肤)


@dataclass
class TactileData:
    """触觉数据"""
    timestamp: float
    sensor_id: str
    tactile_type: TactileType
    # 压力值 (Pa)
    pressure: Optional[float] = None
    # 触感阵列数据 [rows, cols]
    taxel_matrix: Optional[np.ndarray] = None
    # 温度 (°C)
    temperature: Optional[float] = None
    # 湿度 (%)
    humidity: Optional[float] = None
    # 原始数据
    raw: Optional[np.ndarray] = None

    def to_vector(self) -> np.ndarray:
        """转换为特征向量用于融合"""
        vec = []
        if self.pressure is not None:
            vec.append(self.pressure)
        if self.taxel_matrix is not None:
            vec.extend(self.taxel_matrix.flatten())
        if self.temperature is not None:
            vec.append(self.temperature)
        return np.array(vec) if vec else np.array([0.0])

    def get_contact_state(self, threshold: float = 1000.0) -> bool:
        """判断是否接触"""
        if self.pressure is not None:
            return self.pressure > threshold
        if self.taxel_matrix is not None:
            return np.max(self.taxel_matrix) > threshold * 0.1
        return False


class TactileSensor:
    """触觉传感器基类"""

    def __init__(self, sensor_id: str, tactile_type: TactileType, config: Optional[Dict] = None):
        self.sensor_id = sensor_id
        self.tactile_type = tactile_type
        self.config = config or {}
        self.calibration_matrix = np.eye(4)
        self.is_calibrated = False
        self._last_reading: Optional[TactileData] = None

    def read(self, timestamp: Optional[float] = None) -> TactileData:
        """读取触觉数据"""
        raise NotImplementedError

    def calibrate(self, reference_data: np.ndarray) -> bool:
        """校准传感器"""
        if reference_data.shape == (4, 4):
            self.calibration_matrix = reference_data
            self.is_calibrated = True
            return True
        return False

    def get_sensitivity(self) -> float:
        """获取灵敏度 (Pa/bit)"""
        return self.config.get("sensitivity", 1.0)


class PressureSensor(TactileSensor):
    """压阻式压力传感器"""

    def __init__(self, sensor_id: str, config: Optional[Dict] = None):
        super().__init__(sensor_id, TactileType.PRESSURE, config)
        self.sensitivity = config.get("sensitivity", 0.01)  # Pa/bit
        self.offset = config.get("offset", 0.0)
        self.noise_std = config.get("noise_std", 10.0)  # Pa

    def read(self, timestamp: Optional[float] = None) -> TactileData:
        """读取压力数据"""
        # 模拟数据，实际应连接真实传感器API
        raw = np.random.normal(0, self.noise_std)
        pressure = max(0, raw * self.sensitivity + self.offset)

        self._last_reading = TactileData(
            timestamp=timestamp or np.datetime64('now').astype(float) / 1e9,
            sensor_id=self.sensor_id,
            tactile_type=self.tactile_type,
            pressure=pressure,
            raw=np.array([raw])
        )
        return self._last_reading


class TaxelArray(TactileSensor):
    """触感阵列 (仿生皮肤)"""

    def __init__(self, sensor_id: str, rows: int = 16, cols: int = 16, config: Optional[Dict] = None):
        super().__init__(sensor_id, TactileType.TAXEL_ARRAY, config)
        self.rows = rows
        self.cols = cols
        self.baseline = np.zeros((rows, cols))
        self.noise_std = config.get("noise_std", 5.0)
        self._initialize_baseline()

    def _initialize_baseline(self):
        """初始化基线"""
        self.baseline = np.zeros((self.rows, self.cols))

    def read(self, timestamp: Optional[float] = None) -> TactileData:
        """读取触感阵列数据"""
        # 模拟数据：带噪声的空白读数
        noise = np.random.normal(0, self.noise_std, (self.rows, self.cols))
        # 模拟一些接触区域
        contact = np.zeros((self.rows, self.cols))
        if np.random.rand() < 0.2:  # 20%概率有接触
            cx, cy = np.random.randint(4, self.rows - 4), np.random.randint(4, self.cols - 4)
            for i in range(-2, 3):
                for j in range(-2, 3):
                    contact[cx + i, cy + j] = np.random.uniform(500, 2000)

        taxel_data = self.baseline + noise + contact
        taxel_data = np.clip(taxel_data, 0, None)

        self._last_reading = TactileData(
            timestamp=timestamp or np.datetime64('now').astype(float) / 1e9,
            sensor_id=self.sensor_id,
            tactile_type=self.tactile_type,
            taxel_matrix=taxel_data,
            raw=taxel_data.flatten()
        )
        return self._last_reading

    def detect_contact_centroid(self) -> Optional[Tuple[int, int]]:
        """检测接触重心"""
        if self._last_reading and self._last_reading.taxel_matrix is not None:
            matrix = self._last_reading.taxel_matrix
            threshold = np.mean(matrix) + 2 * np.std(matrix)
            mask = matrix > threshold
            if np.any(mask):
                ys, xs = np.where(mask)
                return int(np.mean(xs)), int(np.mean(ys))
        return None


class PiezoelectricSensor(TactileSensor):
    """压电式振动传感器"""

    def __init__(self, sensor_id: str, config: Optional[Dict] = None):
        super().__init__(sensor_id, TactileType.PIEZOELECTRIC, config)
        self.sample_rate = config.get("sample_rate", 1000)  # Hz
        self.buffer_size = config.get("buffer_size", 256)
        self._buffer = np.zeros(self.buffer_size)
        self._buffer_idx = 0

    def read(self, timestamp: Optional[float] = None) -> TactileData:
        """读取振动数据"""
        # 模拟振动信号
        vibration = np.random.normal(0, 50)
        self._buffer[self._buffer_idx % self.buffer_size] = vibration
        self._buffer_idx += 1

        # 计算振动幅度
        amplitude = np.max(np.abs(self._buffer))

        self._last_reading = TactileData(
            timestamp=timestamp or np.datetime64('now').astype(float) / 1e9,
            sensor_id=self.sensor_id,
            tactile_type=self.tactile_type,
            pressure=amplitude,
            raw=self._buffer.copy()
        )
        return self._last_reading

    def get_vibration_spectrum(self) -> np.ndarray:
        """获取振动频谱"""
        if self._last_reading and self._last_reading.raw is not None:
            return np.abs(np.fft.rfft(self._last_reading.raw))
        return np.array([])


class TactileArray:
    """触觉传感器阵列管理"""

    def __init__(self):
        self.sensors: Dict[str, TactileSensor] = {}

    def add_sensor(self, sensor: TactileSensor):
        """添加传感器"""
        self.sensors[sensor.sensor_id] = sensor

    def read_all(self, timestamp: Optional[float] = None) -> List[TactileData]:
        """读取所有传感器数据"""
        return [sensor.read(timestamp) for sensor in self.sensors.values()]

    def get_fusion_data(self, timestamp: Optional[float] = None) -> np.ndarray:
        """获取融合数据"""
        all_data = self.read_all(timestamp)
        vectors = [d.to_vector() for d in all_data]
        return np.concatenate(vectors) if vectors else np.array([])

    def detect_touch_distribution(self) -> Dict[str, float]:
        """检测触摸分布"""
        results = {}
        for sensor_id, sensor in self.sensors.items():
            if isinstance(sensor, TaxelArray) and sensor._last_reading:
                matrix = sensor._last_reading.taxel_matrix
                if matrix is not None:
                    results[sensor_id] = {
                        "max_pressure": float(np.max(matrix)),
                        "mean_pressure": float(np.mean(matrix)),
                        "contact_area": int(np.sum(matrix > 100)),
                        "centroid": sensor.detect_contact_centroid()
                    }
        return results
