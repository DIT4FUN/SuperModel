"""
触觉感知模块
============

电子皮肤触觉阵列接口
- 压力分布感知
- 温度梯度
- 接近觉
- 滑移检测

支持传感器:
- Digi Sensing 电子皮肤阵列
- 通用电阻式/电容式触觉传感器
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict
from enum import Enum


class TactileSensorType(Enum):
    """触觉传感器类型"""
    RESISTIVE = "resistive"   # 电阻式
    CAPACITIVE = "capacitive" # 电容式
    PIEZOELECTRIC = "piezoelectric"  # 压电式
    OPTICAL = "optical"       # 光学式


@dataclass
class TactileFrame:
    """触觉帧"""
    pressure_map: np.ndarray          # H x W, 压力值 (归一化 0-1)
    temperature_map: Optional[np.ndarray] = None  # H x W, 温度 (摄氏度)
    proximity: Optional[np.ndarray] = None  # H x W, 接近距离 (米)
    slip_signal: Optional[np.ndarray] = None  # H x W, 滑移信号
    timestamp: float = 0.0
    frame_id: int = 0
    sensor_id: str = "default"


@dataclass
class TactileContact:
    """接触事件"""
    center: Tuple[int, int]       # 接触中心 (row, col)
    area: int                     # 接触面积 (像素数)
    peak_pressure: float          # 峰值压力
    mean_pressure: float          # 平均压力
    centroid: Tuple[float, float] # 压力质心 (row, col)
    contact_force: float          # 估计接触力 (N)
    slip_probability: float = 0.0  # 滑移概率
    temperature: Optional[float] = None  # 接触区温度


@dataclass
class TactileCalibration:
    """触觉传感器标定参数"""
    pressure_min: float = 0.0
    pressure_max: float = 1.0
    temperature_range: Tuple[float, float] = (0.0, 50.0)  # 摄氏度
    force_scale: float = 100.0  # N (满量程)
    offset_map: Optional[np.ndarray] = None  # 偏置校正


class TactileArray:
    """
    电子皮肤触觉阵列接口
    
    支持:
    - 多区域触觉感知
    - 压力/温度/接近觉融合
    - 接触检测与跟踪
    """
    
    def __init__(
        self,
        array_size: Tuple[int, int] = (16, 16),
        sensor_type: TactileSensorType = TactileSensorType.RESISTIVE,
        sensor_id: str = "tactile_0",
        calibration: Optional[TactileCalibration] = None
    ):
        """
        Args:
            array_size: 触觉阵列尺寸 (rows, cols)
            sensor_type: 传感器类型
            sensor_id: 传感器标识
            calibration: 标定参数
        """
        self.array_size = array_size  # (rows, cols)
        self.sensor_type = sensor_type
        self.sensor_id = sensor_id
        self.calibration = calibration or TactileCalibration()
        
        # 传感器连接配置
        self.rows, self.cols = array_size
        
        # 内部状态
        self._is_opened = False
        self._frame_buffer = []
        self._contact_history: List[TactileContact] = []
        
        # 预分配的帧对象
        self._last_frame: Optional[TactileFrame] = None
        
    def open(self) -> bool:
        """打开传感器"""
        # TODO: 实现硬件接口
        # - I2C/SPI 接口读取
        # - USB 串口通信
        # - CAN 总线
        self._is_opened = True
        print(f"[TactileArray] Opened: {self.sensor_id}, Size={self.array_size}, Type={self.sensor_type.value}")
        return True
    
    def close(self):
        """关闭传感器"""
        if self._is_opened:
            self._is_opened = False
            print(f"[TactileArray] {self.sensor_id} Closed")
    
    def capture(self) -> TactileFrame:
        """捕获一帧触觉数据"""
        if not self._is_opened:
            raise RuntimeError("Tactile sensor not opened")
        
        # TODO: 实现实际数据采集
        # 这里返回模拟数据用于测试
        h, w = self.array_size
        
        # 模拟压力分布 (高斯分布)
        xx, yy = np.meshgrid(np.linspace(-2, 2, w), np.linspace(-2, 2, h))
        gaussian = np.exp(-(xx**2 + yy**2))
        
        # 添加一些噪声
        noise = np.random.randn(h, w) * 0.05
        pressure_map = np.clip(gaussian * 0.7 + noise, 0, 1).astype(np.float32)
        
        # 模拟温度分布
        temperature_map = 25.0 + np.random.randn(h, w) * 0.5
        
        frame = TactileFrame(
            pressure_map=pressure_map,
            temperature_map=temperature_map,
            timestamp=0.0,
            frame_id=0,
            sensor_id=self.sensor_id
        )
        
        self._last_frame = frame
        return frame
    
    def detect_contacts(self, frame: Optional[TactileFrame] = None) -> List[TactileContact]:
        """
        检测接触区域
        
        使用连通域分析提取接触团块
        """
        if frame is None:
            frame = self._last_frame
        if frame is None:
            return []
        
        pressure = frame.pressure_map.copy()
        threshold = 0.1  # 接触阈值
        
        # 二值化
        binary = (pressure > threshold).astype(np.uint8)
        
        # 简单团块检测 (使用scipy/numpy)
        contacts = []
        
        # 找峰值区域
        from scipy.ndimage import label, center_of_mass, find_objects
        labeled, num_features = label(binary)
        
        for i in range(1, num_features + 1):
            region = labeled == i
            area = np.sum(region)
            if area < 2:
                continue
            
            # 质心
            cy, cx = center_of_mass(region)
            
            # 压力统计
            region_pressure = pressure[region]
            peak = np.max(region_pressure)
            mean = np.mean(region_pressure)
            
            # 估计接触力
            contact_force = mean * self.calibration.force_scale
            
            # 滑移概率 (简化估计)
            slip_prob = 0.0
            
            contact = TactileContact(
                center=(int(cy), int(cx)),
                area=int(area),
                peak_pressure=float(peak),
                mean_pressure=float(mean),
                centroid=(float(cy), float(cx)),
                contact_force=contact_force,
                slip_probability=slip_prob,
                temperature=float(np.mean(frame.temperature_map[region])) if frame.temperature_map is not None else None
            )
            contacts.append(contact)
        
        self._contact_history = contacts
        return contacts
    
    def get_slip_signal(self, frame: Optional[TactileFrame] = None) -> np.ndarray:
        """
        计算滑移信号
        
        基于时序压力变化检测滑移趋势
        """
        if frame is None:
            frame = self._last_frame
        if frame is None:
            return np.zeros(self.array_size)
        
        # TODO: 实现滑移检测算法
        # - 压力梯度变化
        # - 纹理追踪
        # - 频域分析
        slip = np.zeros_like(frame.pressure_map)
        return slip
    
    def calibrate(
        self,
        zero_pressure: Optional[np.ndarray] = None,
        known_weights: Optional[List[float]] = None
    ):
        """
        传感器标定
        
        Args:
            zero_pressure: 零压力基准
            known_weights: 已知砝码列表, 用于力-电压标定
        """
        if zero_pressure is not None:
            self.calibration.offset_map = zero_pressure
            print(f"[TactileArray] Zero calibration done: shape={zero_pressure.shape}")
        
        if known_weights is not None:
            # 线性标定
            self.calibration.force_scale = np.mean(known_weights)
            print(f"[TactileArray] Force calibration done: scale={self.calibration.force_scale:.2f} N")
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PressureProcessor:
    """
    压力信号处理器
    
    功能:
    - 噪声滤波
    - 漂移补偿
    - 特征提取
    """
    
    def __init__(
        self,
        filter_window: int = 3,
        drift_compensation: bool = True
    ):
        self.filter_window = filter_window
        self.drift_compensation = drift_compensation
        self._baseline: Optional[np.ndarray] = None
        self._frame_count = 0
        
    def filter(self, pressure_map: np.ndarray) -> np.ndarray:
        """
        滤波处理
        
        使用中值滤波去除椒盐噪声
        """
        from scipy.ndimage import median_filter
        return median_filter(pressure_map, size=self.filter_window)
    
    def compensate_baseline(
        self,
        pressure_map: np.ndarray,
        set_baseline: bool = False
    ) -> np.ndarray:
        """
        基线漂移补偿
        
        Args:
            pressure_map: 原始压力图
            set_baseline: True=设置为新基线, False=补偿到现有基线
        """
        if set_baseline or self._baseline is None:
            self._baseline = pressure_map.copy()
            return pressure_map
        
        compensated = pressure_map - self._baseline
        compensated = np.clip(compensated, 0, 1)
        return compensated
    
    def compute_force(self, pressure_map: np.ndarray, contact_area: float) -> float:
        """
        计算总接触力
        
        Args:
            pressure_map: 压力分布图
            contact_area: 单个传感单元面积 (m^2)
            
        Returns:
            total_force: 总接触力 (N)
        """
        total_pressure = np.sum(pressure_map)
        return total_pressure * contact_area * 1e6  # 归一化系数


# AGV五级触觉规格
AGV_TACTILE_GRADES = {
    'S':  {'array': (8, 8),    'res': 12,  'range_kpa': (0, 500),   'freq_hz': 50,  'temp': False},
    'M':  {'array': (16, 16),  'res': 12,  'range_kpa': (0, 1000),  'freq_hz': 100, 'temp': True},
    'L':  {'array': (24, 24),  'res': 14,  'range_kpa': (0, 2000),  'freq_hz': 200, 'temp': True},
    'XL': {'array': (32, 32),  'res': 14,  'range_kpa': (0, 5000), 'freq_hz': 500, 'temp': True},
    'XXL': {'array': (48, 48), 'res': 16,  'range_kpa': (0, 10000), 'freq_hz': 1000, 'temp': True},
}


def get_tactile_spec(grade: str) -> dict:
    """获取AGV指定等级的触觉规格"""
    return AGV_TACTILE_GRADES.get(grade, AGV_TACTILE_GRADES['M'])
