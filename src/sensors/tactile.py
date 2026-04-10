"""
电子皮肤触觉传感器模块
====================

支持阵列式电子皮肤触觉传感器
- 压力分布检测
- 接触位置识别
- 滑动检测
- 温度分布检测
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
from enum import Enum


class TactileSensorType(Enum):
    """触觉传感器类型"""
    RESISTIVE = "resistive"  # 压阻式
    CAPACITIVE = "capacitive"  # 电容式
    PIEZOELECTRIC = "piezoelectric"  # 压电式
    OPTICAL = "optical"  # 光学式


@dataclass
class TactileReading:
    """单个触觉传感器读数"""
    pressure: float  # 压力值 (kPa)
    temperature: Optional[float] = None  # 温度 (°C)
    shear_x: float = 0.0  # X方向剪力
    shear_y: float = 0.0  # Y方向剪力
    timestamp: float = 0.0


@dataclass
class TactileFrame:
    """触觉帧 (整个阵列)"""
    pressure_map: np.ndarray  # H x W, 压力分布 kPa
    temperature_map: Optional[np.ndarray] = None  # H x W, 温度分布 °C
    shear_map: Optional[np.ndarray] = None  # H x W x 2, 剪力分布
    contact_mask: Optional[np.ndarray] = None  # H x W, 接触区域掩码
    center_of_pressure: Optional[np.ndarray] = None  # [x, y], 压力中心 像素坐标
    total_force: float = 0.0  # 总压力 N
    timestamp: float = 0.0
    frame_id: int = 0


@dataclass
class ContactEvent:
    """接触事件"""
    contact_detected: bool
    contact_area: int  # 接触面积 (像素数)
    center_of_pressure: np.ndarray  # [x, y]
    total_force: float  # 总力 N
    sliding_detected: bool = False
    sliding_velocity: Optional[np.ndarray] = None  # [vx, vy] 像素/秒
    timestamp: float = 0.0


class TactileArray:
    """
    阵列式电子皮肤触觉传感器
    
    支持:
    - 多种尺寸阵列 (8x8, 16x16, 32x32)
    - 多种传感器类型 (压阻/电容/压电/光学)
    - I2C/SPI接口
    - 仿真模式
    """
    
    def __init__(
        self,
        rows: int = 16,
        cols: int = 16,
        sensor_type: TactileSensorType = TactileSensorType.CAPACITIVE,
        sample_rate: int = 100,
        i2c_address: Optional[int] = None,
        calibration: bool = True
    ):
        self.rows = rows
        self.cols = cols
        self.sensor_type = sensor_type
        self.sample_rate = sample_rate
        self.i2c_address = i2c_address
        
        # 校准参数
        self._baseline = np.zeros((rows, cols), dtype=np.float32)
        self._gain = np.ones((rows, cols), dtype=np.float32)
        self._is_calibrated = False
        
        # 配置参数
        self.min_pressure = 0.0  # kPa
        self.max_pressure = 100.0  # kPa
        self.pressure_threshold = 2.0  # kPa 接触阈值
        
        # 状态
        self._is_opened = False
        self._frame_counter = 0
        self._sim_time = 0.0
        
        # 历史数据用于滑动检测
        self._prev_cop = None  # 前一帧压力中心
        self._prev_pressure = None  # 前一帧压力图
        
        if calibration:
            self.calibrate()
    
    def open(self) -> bool:
        """打开传感器"""
        # 优先尝试硬件接口
        try:
            import smbus
            if self.i2c_address is not None:
                self._bus = smbus.SMBus(1)  # /dev/i2c-1
                # 这里添加具体的传感器初始化代码
                self._use_hardware = True
                print(f"[TactileArray] Opened on I2C address 0x{self.i2c_address:02x}: {self.rows}x{self.cols} {self.sensor_type.value}")
            else:
                raise ValueError("No I2C address provided")
            self._is_opened = True
            return True
        except (ImportError, Exception):
            # Fallback: 模拟模式
            self._use_hardware = False
            print(f"[TactileArray] Opened in SIMULATION mode: {self.rows}x{self.cols} {self.sensor_type.value}")
            self._is_opened = True
            return True
    
    def close(self):
        """关闭传感器"""
        if self._is_opened:
            self._is_opened = False
            print("[TactileArray] Closed")
    
    def calibrate(self, samples: int = 100) -> None:
        """
        校准传感器
        
        采集空载基线作为零点
        """
        if self._is_opened:
            # 采集多个样本取平均
            baseline = np.zeros((self.rows, self.cols), dtype=np.float32)
            for _ in range(samples):
                raw = self._read_raw()
                baseline += raw
            baseline /= samples
            self._baseline = baseline
            self._is_calibrated = True
            print(f"[TactileArray] Calibration completed: {samples} samples")
        else:
            # 未打开时使用默认校准
            self._baseline = np.zeros((self.rows, self.cols), dtype=np.float32)
            self._is_calibrated = True
    
    def set_calibration(self, baseline: np.ndarray, gain: Optional[np.ndarray] = None):
        """设置自定义校准参数"""
        assert baseline.shape == (self.rows, self.cols), f"Baseline shape mismatch: expected {self.rows}x{self.cols}"
        self._baseline = baseline.astype(np.float32)
        if gain is not None:
            assert gain.shape == (self.rows, self.cols)
            self._gain = gain.astype(np.float32)
        self._is_calibrated = True
    
    def _read_raw(self) -> np.ndarray:
        """读取原始数据 (内部方法)"""
        if getattr(self, '_use_hardware', False):
            # 这里添加实际I2C读取代码
            # 对于不同传感器实现不同协议
            return self._baseline + np.random.randn(self.rows, self.cols) * 0.1
        else:
            # 模拟模式: 返回随机噪声
            return self._baseline + np.random.randn(self.rows, self.cols) * 0.5
    
    def read(self) -> TactileFrame:
        """读取一帧触觉数据"""
        if not self._is_opened:
            raise RuntimeError("Sensor not opened")
        
        # 读取原始数据
        raw = self._read_raw()
        
        # 校准: 减去基线并应用增益
        pressure = (raw - self._baseline) * self._gain
        
        # 限制范围
        pressure = np.clip(pressure, self.min_pressure, self.max_pressure)
        
        # 检测接触区域
        contact_mask = pressure > self.pressure_threshold
        
        # 计算压力中心和总力
        if np.any(contact_mask):
            # 计算压力中心
            y_idx, x_idx = np.where(contact_mask)
            weighted_pressures = pressure[contact_mask]
            cop_x = np.sum(x_idx * weighted_pressures) / np.sum(weighted_pressures)
            cop_y = np.sum(y_idx * weighted_pressures) / np.sum(weighted_pressures)
            center_of_pressure = np.array([cop_x, cop_y])
            
            # 计算总力: 假设每个单元面积为 (cell_size_mm)^2
            # 默认 5mm 单元间距
            cell_area = (5e-3) ** 2  # m^2
            total_force = np.sum(pressure[contact_mask] * 1000 * cell_area)  # kPa -> Pa = N/m^2
        else:
            center_of_pressure = None
            total_force = 0.0
        
        # 模拟温度: 室温 + 小噪声
        temperature = 25.0 + np.random.randn(self.rows, self.cols) * 0.1
        
        # 计算滑动
        self._prev_pressure = pressure
        self._prev_cop = center_of_pressure
        
        dt = 1.0 / self.sample_rate
        self._sim_time += dt
        self._frame_counter += 1
        
        return TactileFrame(
            pressure_map=pressure,
            temperature_map=temperature,
            contact_mask=contact_mask,
            center_of_pressure=center_of_pressure,
            total_force=total_force,
            timestamp=self._sim_time,
            frame_id=self._frame_counter
        )
    
    def detect_contact_event(self, current_frame: TactileFrame, prev_frame: Optional[TactileFrame] = None) -> ContactEvent:
        """
        检测接触事件
        
        - 是否接触
        - 滑动检测
        """
        if prev_frame is None:
            prev_cop = self._prev_cop
        else:
            prev_cop = prev_frame.center_of_pressure
        
        contact_detected = current_frame.contact_area > 0
        sliding_detected = False
        sliding_velocity = None
        
        if contact_detected and prev_cop is not None and current_frame.center_of_pressure is not None:
            dt = current_frame.timestamp - (prev_frame.timestamp if prev_frame else self._sim_time - 1.0/self.sample_rate)
            if dt > 0:
                displacement = current_frame.center_of_pressure - prev_cop
                velocity = displacement / dt
                if np.linalg.norm(velocity) > 5.0:  # 像素/秒 阈值
                    sliding_detected = True
                    sliding_velocity = velocity
        
        return ContactEvent(
            contact_detected=contact_detected,
            contact_area=current_frame.contact_area if hasattr(current_frame, 'contact_area') else np.sum(current_frame.contact_mask),
            center_of_pressure=current_frame.center_of_pressure if current_frame.center_of_pressure is not None else np.array([0, 0]),
            total_force=current_frame.total_force,
            sliding_detected=sliding_detected,
            sliding_velocity=sliding_velocity,
            timestamp=current_frame.timestamp
        )
    
    def get_pressure_histogram(self, frame: TactileFrame) -> Tuple[np.ndarray, np.ndarray]:
        """获取压力分布直方图"""
        pressure = frame.pressure_map[frame.pressure_map > self.pressure_threshold]
        if len(pressure) == 0:
            return np.array([]), np.array([])
        hist, bins = np.histogram(pressure, bins=20, range=(0, self.max_pressure))
        return hist, bins
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TactileGlove:
    """
    触觉手套接口
    
    多指触觉传感
    - 五指各有独立传感阵列
    - 指尖密集采样
    """
    
    def __init__(
        self,
        fingers: int = 5,
        cells_per_finger: Tuple[int, int] = (4, 8),
        **kwargs
    ):
        self.fingers = fingers
        self.cells_per_finger = cells_per_finger
        self._arrays = []
        for _ in range(fingers):
            rows, cols = cells_per_finger
            self._arrays.append(TactileArray(rows=rows, cols=cols, **kwargs))
    
    def open(self):
        """打开所有指传感器"""
        for i, arr in enumerate(self._arrays):
            arr.open()
        return True
    
    def close(self):
        """关闭所有指传感器"""
        for arr in self._arrays:
            arr.close()
    
    def read_all(self) -> List[TactileFrame]:
        """读取所有手指"""
        return [arr.read() for arr in self._arrays]
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AGVTactileBumper:
    """
    AGV 触觉保险杠
    
    分布在AGV四周的触觉传感器阵列
    用于碰撞检测和障碍物触觉感知
    """
    
    def __init__(
        self,
        segments: int = 8,  # 四周分段
        pressure_threshold: float = 5.0,  # kPa
        sample_rate: int = 50
    ):
        self.segments = segments
        self.pressure_threshold = pressure_threshold
        self.sample_rate = sample_rate
        
        self._readings = np.zeros(segments, dtype=np.float32)
        self._is_opened = False
        self._frame_counter = 0
        self._sim_time = 0.0
    
    def open(self):
        """打开保险杠传感器"""
        try:
            # 尝试CAN总线接口
            import can
            self._use_can = True
            # 初始化CAN总线...
            print(f"[AGVTactileBumper] Opened with CAN: {self.segments} segments")
        except (ImportError, Exception):
            self._use_can = False
            print(f"[AGVTactileBumper] Opened in SIMULATION mode: {self.segments} segments")
        self._is_opened = True
        return True
    
    def close(self):
        """关闭"""
        self._is_opened = False
        print("[AGVTactileBumper] Closed")
    
    def read(self) -> np.ndarray:
        """读取所有分段压力"""
        if not self._is_opened:
            raise RuntimeError("Bumper not opened")
        
        if getattr(self, '_use_can', False):
            # 实际CAN读取...
            pass
        else:
            # 模拟模式
            self._readings = np.zeros(self.segments, dtype=np.float32)
            # 偶尔有随机接触
            if np.random.rand() < 0.05:
                seg = np.random.randint(0, self.segments)
                self._readings[seg] = 10.0 + np.random.rand() * 20.0
        
        self._sim_time += 1.0 / self.sample_rate
        self._frame_counter += 1
        
        return self._readings.copy()
    
    def detect_collision(self) -> Tuple[bool, List[int]]:
        """
        检测碰撞
        
        Returns:
            collision_detected: 是否发生碰撞
            hit_segments: 碰撞分段索引
        """
        readings = self.read()
        hit_segments = np.where(readings > self.pressure_threshold)[0].tolist()
        return len(hit_segments) > 0, hit_segments
    
    def get_centroid_direction(self) -> Optional[float]:
        """获取碰撞中心方向 (0~360度)"""
        readings = self.read()
        if np.all(readings <= self.pressure_threshold):
            return None
        
        # 计算加权方向
        segment_angle = 360.0 / self.segments
        angles = np.arange(self.segments) * segment_angle
        weights = np.maximum(readings - self.pressure_threshold, 0)
        weighted_angle = np.sum(angles * weights) / np.sum(weights)
        
        return weighted_angle
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# AGV五级触觉传感器规格
AGV_TACTILE_GRADES = {
    'S': {
        'array': (8, 8),
        'res': 12,
        'freq_hz': 50,
        'temp': False,
        'bumper_segments': 4,
        'has_skin': False,
        'resolution': 'coarse',
        'sample_rate': 50
    },
    'M': {
        'array': (16, 16),
        'res': 12,
        'freq_hz': 100,
        'temp': True,
        'bumper_segments': 8,
        'has_skin': True,
        'skin_size': (16, 16),
        'resolution': 'medium',
        'sample_rate': 100
    },
    'L': {
        'array': (24, 24),
        'res': 14,
        'freq_hz': 200,
        'temp': True,
        'bumper_segments': 12,
        'has_skin': True,
        'skin_size': (32, 32),
        'resolution': 'medium',
        'sample_rate': 100
    },
    'XL': {
        'array': (32, 32),
        'res': 14,
        'freq_hz': 500,
        'temp': True,
        'bumper_segments': 16,
        'has_skin': True,
        'skin_size': (32, 64),
        'resolution': 'high',
        'sample_rate': 200,
        'has_temperature': True
    },
    'XXL': {
        'array': (48, 48),
        'res': 16,
        'freq_hz': 1000,
        'temp': True,
        'bumper_segments': 24,
        'has_skin': True,
        'skin_size': (64, 128),
        'resolution': 'very_high',
        'sample_rate': 200,
        'has_temperature': True,
        'has_shear': True
    }
}


def get_tactile_spec(grade: str) -> dict:
    """获取AGV指定等级的触觉传感器规格"""
    return AGV_TACTILE_GRADES.get(grade, AGV_TACTILE_GRADES['M'])


# 兼容旧名称 (用于测试)
class PressureProcessor:
    """压力处理器 (兼容别名)"""
    def __init__(self, filter_window=3):
        self.filter_window = filter_window
    
    def process(self, frame):
        """处理压力帧"""
        return {
            'mean_pressure': np.mean(frame.pressure_map),
            'std_pressure': np.std(frame.pressure_map),
        }


class TactileContact:
    """接触 (兼容别名)"""
    def __init__(self, center, area, peak_pressure, mean_pressure=None, centroid=None, contact_force=None):
        self.center = center if centroid is None else centroid
        self.area = area
        self.peak_pressure = peak_pressure
        self.mean_pressure = mean_pressure
        self.contact_force = contact_force


class TactileCalibration:
    """标定 (兼容别名)"""
    @classmethod
    def create_default(cls, size):
        return cls()
    
    def apply(self, frame):
        return frame


class VirtualTactileSensor:
    """虚拟触觉传感器 (兼容别名)"""
    def __init__(self, array_size=(16, 16), sensor_id="virtual"):
        self.array_size = array_size
        self.sensor_id = sensor_id
    
    def open(self):
        return True
    
    def close(self):
        pass
    
    def simulate_contact(self, *args, **kwargs):
        """兼容方法 - 返回空接触"""
        return TactileContact(center=np.zeros(2), area=0, peak_pressure=0)
