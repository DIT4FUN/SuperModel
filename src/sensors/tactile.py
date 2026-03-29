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
        采用多尺度滑移检测算法:
        1. 压力梯度变化检测
        2. 高频振动成分检测
        3. 压力分布变化率分析
        """
        if frame is None:
            frame = self._last_frame
        if frame is None:
            return np.zeros(self.array_size)
        
        # 如果有历史帧，进行滑移检测
        if len(self._frame_buffer) >= 2:
            prev_frame = self._frame_buffer[-1]
            
            # 1. 压力梯度变化检测
            pressure_diff = frame.pressure_map - prev_frame.pressure_map
            
            # 2. 计算局部梯度 (使用Sobel算子模拟)
            from scipy.ndimage import convolve
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
            sobel_y = sobel_x.T
            
            grad_x = convolve(frame.pressure_map, sobel_x)
            grad_y = convolve(frame.pressure_map, sobel_y)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # 3. 高频成分检测 (滑移产生高频振动)
            # 使用Laplacian算子检测高频成分
            laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            laplacian = convolve(frame.pressure_map, laplacian_kernel)
            high_freq = np.abs(laplacian)
            
            # 4. 多帧历史分析 (如果有多帧历史)
            slip_signal = np.zeros_like(frame.pressure_map)
            if len(self._frame_buffer) >= 3:
                # 计算连续帧的变化率
                prev_prev_frame = self._frame_buffer[-2]
                diff2 = frame.pressure_map - 2 * prev_frame.pressure_map + prev_prev_frame.pressure_map
                # 加速度变化指示滑移趋势
                slip_signal += np.abs(diff2) * 0.5
            
            # 5. 综合滑移信号
            # 梯度变化贡献
            slip_signal += np.abs(pressure_diff) * (1 + grad_magnitude * 5.0)
            # 高频振动贡献 (滑移时振动加剧)
            slip_signal += high_freq * 3.0
            # 接触区域加权 (仅在接触区域计算)
            contact_mask = (frame.pressure_map > 0.1).astype(np.float32)
            slip_signal *= contact_mask
            
            # 归一化
            slip_signal = np.clip(slip_signal, 0, 1)
            
            # 更新帧缓冲区
            self._frame_buffer.append(frame)
            if len(self._frame_buffer) > 10:  # 保留最近10帧
                self._frame_buffer.pop(0)
            
            return slip_signal.astype(np.float32)
        
        # 第一帧，初始化缓冲区
        self._frame_buffer.append(frame)
        return np.zeros(self.array_size, dtype=np.float32)
    
    def estimate_grip_quality(self, frame: Optional[TactileFrame] = None) -> Dict[str, float]:
        """
        估计抓取质量
        
        综合评估:
        - 接触面积
        - 压力分布均匀性
        - 抓取稳定性
        
        Returns:
            grip_quality: 包含各项指标的字典
        """
        if frame is None:
            frame = self._last_frame
        if frame is None:
            return {'overall': 0.0, 'contact_area': 0.0, 'uniformity': 0.0, 'stability': 0.0}
        
        contacts = self.detect_contacts(frame)
        
        if not contacts:
            return {'overall': 0.0, 'contact_area': 0.0, 'uniformity': 0.0, 'stability': 0.0}
        
        # 接触面积评分 (相对于阵列大小)
        total_contact_area = sum(c.area for c in contacts)
        max_area = self.array_size[0] * self.array_size[1]
        contact_score = min(total_contact_area / max_area * 5.0, 1.0)
        
        # 均匀性评分 (基于压力方差)
        pressures = frame.pressure_map[frame.pressure_map > 0.1]
        if len(pressures) > 1:
            uniformity_score = 1.0 - min(np.std(pressures), 1.0)
        else:
            uniformity_score = 0.0
        
        # 稳定性评分 (基于历史滑移)
        slip = self.get_slip_signal(frame)
        avg_slip = np.mean(slip[slip > 0])
        stability_score = 1.0 - min(avg_slip * 5.0, 1.0)
        
        # 综合评分
        overall = 0.4 * contact_score + 0.3 * uniformity_score + 0.3 * stability_score
        
        return {
            'overall': float(overall),
            'contact_area': float(contact_score),
            'uniformity': float(uniformity_score),
            'stability': float(stability_score)
        }
    
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
    
    def compute_centroid(self, pressure_map: np.ndarray) -> Tuple[float, float]:
        """
        计算压力分布质心
        
        Args:
            pressure_map: 压力分布图
            
        Returns:
            centroid: (row, col) 质心坐标
        """
        h, w = pressure_map.shape
        rows = np.arange(h)[:, np.newaxis]
        cols = np.arange(w)[np.newaxis, :]
        
        total = np.sum(pressure_map) + 1e-10
        cy = np.sum(rows * pressure_map) / total
        cx = np.sum(cols * pressure_map) / total
        
        return float(cy), float(cx)
    
    def compute_pressure_histogram(
        self,
        pressure_map: np.ndarray,
        bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算压力分布直方图
        
        Args:
            pressure_map: 压力分布图
            bins: 直方图 bin 数量
            
        Returns:
            hist: 直方图值
            bin_edges: bin 边界
        """
        flat = pressure_map.flatten()
        hist, edges = np.histogram(flat, bins=bins, range=(0, 1))
        return hist.astype(np.float32), edges


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


class VirtualTactileSensor:
    """
    虚拟触觉传感器 (仿真环境使用)
    
    模拟真实触觉感知，用于:
    - 仿真环境中的触觉反馈
    - 算法验证和调试
    - 抓取/操作任务仿真
    """
    
    def __init__(
        self,
        array_size: Tuple[int, int] = (16, 16),
        sensor_id: str = "virtual_tactile"
    ):
        self.array_size = array_size
        self.sensor_id = sensor_id
        self.rows, self.cols = array_size
        self._is_opened = False
        self._frame_id = 0
        self._last_contact_pos: Optional[Tuple[float, float]] = None
    
    def open(self) -> bool:
        self._is_opened = True
        return True
    
    def close(self):
        self._is_opened = False
    
    def simulate_contact(
        self,
        contact_pos: Tuple[float, float],
        contact_radius: float = 0.3,
        contact_force: float = 10.0,
        noise_level: float = 0.05
    ) -> TactileFrame:
        """
        模拟接触事件
        
        Args:
            contact_pos: 接触中心位置 (归一化 0-1)
            contact_radius: 接触半径 (归一化)
            contact_force: 接触力 (N)
            noise_level: 噪声水平
            
        Returns:
            TactileFrame with simulated pressure map
        """
        h, w = self.array_size
        
        # 创建接触区域高斯分布
        xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        cx, cy = contact_pos
        
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        gaussian = np.exp(-dist**2 / (2 * contact_radius**2))
        
        # 压力值转换
        pressure_map = gaussian * (contact_force / 100.0)
        noise = np.random.randn(h, w) * noise_level
        pressure_map = np.clip(pressure_map + noise, 0, 1).astype(np.float32)
        
        # 温度模拟
        temperature_map = 25.0 + pressure_map * 5.0 + np.random.randn(h, w) * 0.3
        
        self._last_contact_pos = contact_pos
        
        frame = TactileFrame(
            pressure_map=pressure_map,
            temperature_map=temperature_map,
            timestamp=0.0,
            frame_id=self._frame_id,
            sensor_id=self.sensor_id
        )
        self._frame_id += 1
        return frame
    
    def simulate_sliding(
        self,
        direction: Tuple[float, float],
        speed: float = 0.1,
        duration_frames: int = 30
    ) -> List[TactileFrame]:
        """
        模拟滑移动作
        
        Args:
            direction: 滑动方向 (dx, dy) 归一化
            speed: 滑动速度 (归一化/帧)
            duration_frames: 持续帧数
            
        Returns:
            List of TactileFrame
        """
        frames = []
        current_pos = self._last_contact_pos or (0.5, 0.5)
        
        for i in range(duration_frames):
            new_pos = (
                current_pos[0] + direction[0] * speed,
                current_pos[1] + direction[1] * speed
            )
            # 边界约束
            new_pos = (
                max(0.05, min(0.95, new_pos[0])),
                max(0.05, min(0.95, new_pos[1]))
            )
            
            frame = self.simulate_contact(
                new_pos,
                contact_radius=0.25,
                contact_force=8.0 + np.random.randn() * 2.0,
                noise_level=0.08
            )
            frames.append(frame)
            current_pos = new_pos
        
        return frames
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, *args):
        self.close()
