# Copyright (C) 2026 焦洋 (Jiao Yang) <jiaoyang@cczu.edu.cn>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
    sensor_id: str = "default"  # 传感器ID
    
    @property
    def contact_area(self) -> int:
        """兼容接口 - 接触面积像素数"""
        if self.contact_mask is not None:
            return int(np.sum(self.contact_mask))
        # 如果 contact_mask 未提供，从 pressure_map 计算接触面积（压力大于0）
        if self.pressure_map is not None:
            return int(np.sum(self.pressure_map > 0))
        return 0


@dataclass
class ContactEvent:
    """接触事件"""
    contact_detected: bool
    contact_area: int  # 接触面积 (像素数)
    center_of_pressure: np.ndarray  # [x, y]
    total_force: float  # 总力 N
    peak_pressure: float = 0.0  # 峰值压力
    mean_pressure: float = 0.0  # 平均压力
    sliding_detected: bool = False
    sliding_velocity: Optional[np.ndarray] = None  # [vx, vy] 像素/秒
    timestamp: float = 0.0
    
    @property
    def centroid(self):
        """兼容别名：质心就是压力中心"""
        return self.center_of_pressure


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
        array_size: Optional[Tuple[int, int]] = None,  # 向后兼容旧接口
        sensor_type: TactileSensorType = TactileSensorType.CAPACITIVE,
        sample_rate: int = 100,
        i2c_address: Optional[int] = None,
        calibration: bool = True,
        sensor_id: Optional[str] = None,  # 向后兼容旧接口
    ):
        # 完整处理所有兼容情况:
        # 1. TactileArray((8, 8)) -> 第一个参数就是 tuple/list
        if isinstance(rows, (tuple, list, np.ndarray)) and len(rows) == 2:
            r, c = rows
            rows, cols = int(r), int(c)
        # 2. TactileArray(array_size=(8, 8)) -> keyword 参数
        elif array_size is not None and isinstance(array_size, (tuple, list, np.ndarray)) and len(array_size) == 2:
            r, c = array_size
            rows, cols = int(r), int(c)
        # 3. 正常参数
        self.rows = int(rows)
        self.cols = int(cols)
        self.sensor_type = sensor_type
        self.sample_rate = sample_rate
        self.i2c_address = i2c_address
        self.array_size = (rows, cols)  # 兼容属性
        self.sensor_id = sensor_id  # 兼容属性
        
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

        # 帧缓冲区 (限制最大100帧，防止溢出)
        self._frame_buffer: List[TactileFrame] = []

        # 兼容标定对象
        self.calibration = TactileCalibration.create_default((rows, cols))
        # 同步基线到标定对象
        self.calibration.offset_map = self._baseline

        if calibration:
            self.calibrate()

    @property
    def _frame_id(self) -> int:
        """兼容别名: 帧ID = 帧计数器"""
        return self._frame_counter
    
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
            self._frame_counter = 0  # 重置帧计数器
            self._sim_time = 0.0
            return True

    def close(self):
        """关闭传感器"""
        if self._is_opened:
            self._is_opened = False
            print("[TactileArray] Closed")
    
    def calibrate(self, samples: int = 100, zero_pressure: Optional[np.ndarray] = None, known_weights: Optional[List[float]] = None) -> None:
        """
        校准传感器
        
        采集空载基线作为零点
        """
        if zero_pressure is not None:
            # 使用提供的零点压力
            self._baseline = zero_pressure.astype(np.float32)
            self._is_calibrated = True
            return
        
        if self._is_opened:
            # 采集多个样本取平均
            baseline = np.zeros((self.rows, self.cols), dtype=np.float32)
            for _ in range(samples):
                raw = self._read_raw()
                baseline += raw
            baseline /= samples
            self._baseline = baseline
            if self.calibration is not None:
                self.calibration.offset_map = self._baseline.copy()
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
            # 模拟模式: 返回随机噪声，如果有接触位置添加高斯接触压力
            raw = self._baseline + np.random.randn(self.rows, self.cols) * 0.5
            
            # 如果有指定接触位置，添加高斯接触压力
            if hasattr(self, '_last_contact_pos') and self._last_contact_pos is not None:
                # 归一化坐标 -> 像素坐标
                x_norm, y_norm = self._last_contact_pos
                x = int(round(x_norm * (self.cols - 1)))
                y = int(round(y_norm * (self.rows - 1)))
                
                # 添加2D高斯压力分布
                sigma = 2.0  # 高斯半径
                yy, xx = np.mgrid[0:self.rows, 0:self.cols]
                gauss = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
                peak_pressure = 5.0  # kPa
                raw += peak_pressure * gauss
            
            return raw
    
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
        frame_id = self._frame_counter
        self._sim_time += dt
        self._frame_counter += 1
        
        frame = TactileFrame(
            pressure_map=pressure,
            temperature_map=temperature,
            contact_mask=contact_mask,
            center_of_pressure=center_of_pressure,
            total_force=total_force,
            timestamp=self._sim_time,
            frame_id=frame_id,
            sensor_id=self.sensor_id if hasattr(self, 'sensor_id') and self.sensor_id else 'default'
        )

        # 追加到缓冲区并限制最大长度
        self._frame_buffer.append(frame)
        if len(self._frame_buffer) > 100:
            self._frame_buffer.pop(0)

        return frame
    
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
        
        # 使用传感器的压力阈值判断接触
        contact_area = current_frame.contact_area
        # 如果接触面积大于0且有压力超过阈值，则检测到接触
        contact_detected = contact_area > 0
        # 如果没有mask但是pressure_map有压力，也认为接触
        if not contact_detected and current_frame.pressure_map is not None:
            contact_detected = np.any(current_frame.pressure_map > self.pressure_threshold)
            if contact_detected:
                contact_area = int(np.sum(current_frame.pressure_map > self.pressure_threshold))
        
        # 如果 center_of_pressure 未提供，自动计算
        if current_frame.center_of_pressure is None and current_frame.pressure_map is not None:
            # 计算压力中心
            ys, xs = np.where(current_frame.pressure_map > self.pressure_threshold)
            if len(ys) > 0:
                if len(ys) == 1:
                    cx, cy = xs[0], ys[0]
                else:
                    cx = np.mean(xs)
                    cy = np.mean(ys)
                center_of_pressure = np.array([cx, cy])
            else:
                center_of_pressure = None
        else:
            center_of_pressure = current_frame.center_of_pressure
            
        sliding_detected = False
        sliding_velocity = None
        
        if contact_detected and prev_cop is not None and center_of_pressure is not None:
            dt = current_frame.timestamp - (prev_frame.timestamp if prev_frame else self._sim_time - 1.0/self.sample_rate)
            if dt > 0:
                displacement = current_frame.center_of_pressure - prev_cop
                velocity = displacement / dt
                if np.linalg.norm(velocity) > 5.0:  # 像素/秒 阈值
                    sliding_detected = True
                    sliding_velocity = velocity
        
        # 计算峰值和平均压力
        if current_frame.pressure_map is not None and np.any(current_frame.pressure_map > 0):
            pressure_valid = current_frame.pressure_map[current_frame.pressure_map > 0]
            peak_p = np.max(pressure_valid) if len(pressure_valid) > 0 else 0.0
            mean_p = np.mean(pressure_valid) if len(pressure_valid) > 0 else 0.0
        else:
            peak_p = 0.0
            mean_p = 0.0
            
        return ContactEvent(
            contact_detected=contact_detected,
            contact_area=contact_area,
            center_of_pressure=center_of_pressure if center_of_pressure is not None else np.array([0, 0]),
            total_force=current_frame.total_force,
            peak_pressure=peak_p,
            mean_pressure=mean_p,
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
    
    def capture(self) -> TactileFrame:
        """兼容旧接口 - capture 别名"""
        return self.read()
    
    def detect_contacts(self, frame: TactileFrame):
        """兼容旧接口别名 - 返回接触列表保持向后兼容"""
        event = self.detect_contact_event(frame)
        if event.contact_detected:
            return [event]
        else:
            return []
    
    def estimate_grip_quality(self, frame: TactileFrame):
        """估算抓取质量 - 返回字典保持向后兼容"""
        if frame.contact_mask is None:
            score = 0.0
        else:
            contact_area = frame.contact_area
            total_force = frame.total_force
            if contact_area == 0 or total_force < 0.1:
                score = 0.0
            else:
                # 简单评分：接触面积适中，力合适 → 高质量
                area_score = min(1.0, contact_area / (frame.pressure_map.shape[0] * frame.pressure_map.shape[1] * 0.5))
                force_score = min(1.0, total_force / 20.0)
                score = (area_score + force_score) / 2.0
        ca = frame.contact_area if frame.contact_mask is not None else 0
        # 计算压力均匀性 (标准差越小越均匀)
        if frame.contact_mask is not None and ca > 0:
            active = frame.pressure_map[frame.contact_mask]
            uniformity = 1.0 / (1.0 + float(np.std(active)))
        else:
            uniformity = 0.0
        # 稳定性评分: 基于接触面积和力的综合稳定性
        stability = score  # 复用 overall 作为稳定性评分
        return {
            'overall': score,
            'contact_area': ca,
            'uniformity': uniformity,
            'stability': stability,
            'area_score': min(1.0, ca / (frame.pressure_map.shape[0] * frame.pressure_map.shape[1] * 0.5)),
            'force_score': score,
        }
    
    def get_slip_signal(self, frame: TactileFrame = None, prev_frame: Optional[TactileFrame] = None) -> Optional[np.ndarray]:
        """获取滑动信号 - 兼容无参数调用"""
        if frame is None:
            # 向后兼容：无参数调用返回零滑动数组
            return np.zeros((self.rows, self.cols), dtype=np.float32)
        event = self.detect_contact_event(frame, prev_frame)
        if event.sliding_detected and event.sliding_velocity is not None:
            # 返回完整数组，兼容旧接口期望
            slip = np.zeros((self.rows, self.cols), dtype=np.float32)
            if event.sliding_velocity is not None:
                # 滑动信号填充到对应位置
                if hasattr(frame, 'center_of_pressure') and frame.center_of_pressure is not None:
                    y, x = frame.center_of_pressure
                    if 0 <= int(y) < self.rows and 0 <= int(x) < self.cols:
                        slip[int(y), int(x)] = np.linalg.norm(event.sliding_velocity)
            return slip
        return np.zeros((self.rows, self.cols), dtype=np.float32)
    
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
    
    def capture(self) -> List[TactileFrame]:
        """兼容旧接口 - capture 别名"""
        return self.read_all()


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
        'sample_rate': 50,
        'range_kpa': 500,
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
        'sample_rate': 100,
        'range_kpa': 1000,
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
        'sample_rate': 100,
        'range_kpa': 2000,
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
        'has_temperature': True,
        'range_kpa': 5000,
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
        'has_shear': True,
        'range_kpa': 10000,
    }
}


def get_tactile_spec(grade: str) -> dict:
    """获取AGV指定等级的触觉传感器规格"""
    return AGV_TACTILE_GRADES.get(grade, AGV_TACTILE_GRADES['M'])


# 兼容旧名称 (用于测试)
class PressureProcessor:
    """压力处理器 (兼容别名)"""
    def __init__(self, filter_window=3, drift_compensation=False):
        self.filter_window = filter_window
        self.drift_compensation = drift_compensation
        self._baseline_pressure: Optional[np.ndarray] = None
        self._history: List[Dict[str, Any]] = []
        self._ema_map: Optional[np.ndarray] = None  # EMA 滤波状态

    def filter(self, pressure_map):
        """EMA 滤波压力图"""
        arr = np.asarray(pressure_map, dtype=np.float32)
        alpha = 2.0 / (self.filter_window + 1)
        if self._ema_map is None:
            self._ema_map = arr.copy()
        else:
            self._ema_map = alpha * arr + (1 - alpha) * self._ema_map
        return self._ema_map

    def compensate_baseline(self, pressure_map, set_baseline=False):
        """基线补偿"""
        arr = np.asarray(pressure_map, dtype=np.float32)
        if set_baseline or self._baseline_pressure is None:
            self._baseline_pressure = arr.copy()
            return arr
        return arr - self._baseline_pressure + np.mean(self._baseline_pressure)

    def compute_force(self, pressure_map, contact_area=1e-4):
        """估算接触力 (N)"""
        mean_p = float(np.mean(pressure_map))
        return mean_p * contact_area * 1000.0  # kPa -> Pa = N/m^2 * area

    def compute_centroid(self, pressure_map):
        """计算压力图质心"""
        arr = np.asarray(pressure_map, dtype=np.float32)
        total = np.sum(arr)
        if total < 1e-9:
            return np.array([0.5, 0.5])
        rows, cols = arr.shape
        y_idx, x_idx = np.indices(arr.shape)
        cx = np.sum(x_idx * arr) / total
        cy = np.sum(y_idx * arr) / total
        return np.array([cx / cols, cy / rows])

    def compute_pressure_histogram(self, pressure_map, bins=10):
        """计算压力分布直方图"""
        arr = np.asarray(pressure_map, dtype=np.float32).flatten()
        hist, edges = np.histogram(arr, bins=bins, range=(0, np.max(arr) + 1e-6))
        return hist, edges

    def process(self, frame):
        """处理压力帧"""
        result = {
            'mean_pressure': float(np.mean(frame.pressure_map)),
            'std_pressure': float(np.std(frame.pressure_map)),
            'max_pressure': float(np.max(frame.pressure_map)),
            'min_pressure': float(np.min(frame.pressure_map)),
        }

        # 漂移补偿
        if self.drift_compensation:
            if self._baseline_pressure is None:
                self._baseline_pressure = frame.pressure_map.copy()
            else:
                # 缓慢更新的基线
                self._baseline_pressure = 0.99 * self._baseline_pressure + 0.01 * frame.pressure_map
            compensated = frame.pressure_map - self._baseline_pressure
            result['compensated_mean'] = float(np.mean(compensated))

        # 滑动窗口滤波
        if self._history:
            self._history.append(result)
            if len(self._history) > self.filter_window:
                self._history.pop(0)

        return result


class TactileContact:
    """接触 (兼容别名)"""
    def __init__(self, center, area, peak_pressure, mean_pressure=None, centroid=None, contact_force=None, slip_probability=None):
        self.center = center if centroid is None else centroid
        self.area = area
        self.peak_pressure = peak_pressure
        self.mean_pressure = mean_pressure
        self.contact_force = contact_force
        self.slip_probability = slip_probability


@dataclass
class TactileCalibration:
    """标定 (兼容别名)"""
    offset_map: np.ndarray = None
    
    @classmethod
    def create_default(cls, size):
        obj = cls()
        rows, cols = size if isinstance(size, (tuple, list)) else (size, size)
        obj.offset_map = np.zeros((rows, cols), dtype=np.float32)
        return obj
    
    def apply(self, frame):
        if self.offset_map is not None and frame.pressure_map is not None:
            frame.pressure_map = frame.pressure_map - self.offset_map
        return frame


class VirtualTactileSensor:
    """虚拟触觉传感器 (兼容别名)"""
    def __init__(self, array_size=(16, 16), sensor_id="virtual", noise_level=None):
        self.array_size = array_size
        self.rows, self.cols = array_size
        self.sensor_id = sensor_id
        self.noise_level = noise_level
        self._is_opened = False
    
    def open(self):
        self._is_opened = True
        return True
    
    def close(self):
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def simulate_contact(self, center_xy=None, radius=None, peak_pressure=None, contact_pos=None, pressure=None, contact_radius=None, contact_force=None, noise_level=None):
        """模拟接触生成压力图，返回TactileFrame"""
        # 兼容多种参数形式
        if contact_pos is not None:
            # contact_pos=(cx, cy), pressure=peak_pressure/contact_force, contact_radius=radius
            cx, cy = contact_pos
            if contact_force is not None:
                # contact_force (N) 归一化到压力范围 (0-1.2), 以100N为基准
                peak_pressure = contact_force / 100.0
            else:
                peak_pressure = pressure if pressure is not None else 10.0
            radius = contact_radius if contact_radius is not None else (radius if radius is not None else 0.3)
        elif center_xy is not None:
            cx, cy = center_xy
            peak_pressure = peak_pressure if peak_pressure is not None else 10.0
            radius = radius if radius is not None else 0.3
        else:
            cx, cy = (0.5, 0.5)
            peak_pressure = 10.0
            radius = 0.3
            
        pressure_map = np.zeros((self.rows, self.cols), dtype=np.float32)
        
        # 生成二维高斯分布
        for r in range(self.rows):
            for c in range(self.cols):
                # 归一化坐标
                y_norm = r / (self.rows - 1) if self.rows > 1 else 0.5
                x_norm = c / (self.cols - 1) if self.cols > 1 else 0.5
                dist_sq = (x_norm - cx) ** 2 + (y_norm - cy) ** 2
                if dist_sq <= radius ** 2:
                    pressure_map[r, c] = peak_pressure * np.exp(-dist_sq / (2 * (radius/3)**2))
        
        # 检测接触
        contact_mask = pressure_map > 0.1 * peak_pressure
        
        # 计算压力中心
        if np.any(contact_mask):
            y_idx, x_idx = np.where(contact_mask)
            weighted_pressures = pressure_map[contact_mask]
            cop_x = np.sum(x_idx * weighted_pressures) / np.sum(weighted_pressures)
            cop_y = np.sum(y_idx * weighted_pressures) / np.sum(weighted_pressures)
            center_of_pressure = np.array([cop_x, cop_y])
            total_force = np.sum(pressure_map) * 0.01
        else:
            center_of_pressure = None
            total_force = 0.0
        
        # 添加噪声
        nl = noise_level if noise_level is not None else self.noise_level
        if nl is not None and nl > 0:
            pressure_map += np.random.randn(*pressure_map.shape) * nl
            pressure_map = np.clip(pressure_map, 0, None)

        return TactileFrame(
            pressure_map=pressure_map,
            temperature_map=25.0 + np.random.randn(self.rows, self.cols) * 0.1,
            contact_mask=contact_mask,
            center_of_pressure=center_of_pressure,
            total_force=total_force
        )
    
    def capture(self):
        """兼容capture接口"""
        return self.simulate_contact((0.5, 0.5), 0.3, 5.0)

    def simulate_multi_contact(self, contacts, noise_level=None):
        """模拟多点接触

        Args:
            contacts: list of (pos, force, radius) tuples
                      pos: (x, y) 归一化位置
                      force: 接触力 N
                      radius: 接触半径 (归一化)
            noise_level: 噪声标准差

        Returns:
            TactileFrame with combined pressure map
        """
        pressure_map = np.zeros((self.rows, self.cols), dtype=np.float32)

        for cx_norm, force, radius in contacts:
            cx, cy = cx_norm
            # 计算峰值压力 (kPa, 假设 1N -> 10kPa)
            peak_p = force * 10.0

            for r in range(self.rows):
                for c in range(self.cols):
                    y_norm = r / (self.rows - 1) if self.rows > 1 else 0.5
                    x_norm = c / (self.cols - 1) if self.cols > 1 else 0.5
                    dist_sq = (x_norm - cx) ** 2 + (y_norm - cy) ** 2
                    if dist_sq <= radius ** 2:
                        pressure_map[r, c] += peak_p * np.exp(-dist_sq / (2 * (radius / 3) ** 2))

        # 噪声
        nl = noise_level if noise_level is not None else self.noise_level
        if nl is not None and nl > 0:
            pressure_map += np.random.randn(*pressure_map.shape) * nl
            pressure_map = np.clip(pressure_map, 0, None)

        # 接触掩码
        contact_mask = pressure_map > 0.5

        # 压力中心
        if np.any(contact_mask):
            y_idx, x_idx = np.where(contact_mask)
            wp = pressure_map[contact_mask]
            cop_x = np.sum(x_idx * wp) / np.sum(wp)
            cop_y = np.sum(y_idx * wp) / np.sum(wp)
            center_of_pressure = np.array([cop_x, cop_y])
        else:
            center_of_pressure = np.array([self.cols / 2, self.rows / 2])

        # 记录最后接触位置
        self._last_contact_pos = center_of_pressure

        total_force = np.sum(pressure_map) * 0.01
        return TactileFrame(
            pressure_map=pressure_map,
            contact_mask=contact_mask,
            center_of_pressure=center_of_pressure,
            total_force=total_force
        )

    def simulate_sliding(self, direction=None, speed=None, duration_frames=None, start_position=None, end_position=None, peak_pressure=10.0):
        """模拟滑移动作

        Args:
            direction: (dx, dy) 滑动方向 (归一化)
            speed: 滑动速度 (归一化/秒)
            duration_frames: 帧数
            start_position: 起始位置 (x, y) - 归一化
            end_position: 结束位置 (x, y) - 归一化
            peak_pressure: 峰值压力

        Returns:
            list[TactileFrame]: 滑移帧序列
        """
        frames = []
        num_frames = duration_frames if duration_frames else 10

        if start_position is not None and end_position is not None:
            # 方式1: 起始-结束位置
            for i in range(num_frames):
                t = i / max(1, num_frames - 1)
                cx = start_position[0] + t * (end_position[0] - start_position[0])
                cy = start_position[1] + t * (end_position[1] - start_position[1])
                frame = self.simulate_contact((cx, cy), radius=0.2, peak_pressure=peak_pressure)
                frames.append(frame)
        else:
            # 方式2: 方向+速度
            direction = direction or (1.0, 0.0)
            speed = speed or 0.1
            cx, cy = 0.5, 0.5
            for i in range(num_frames):
                cx += direction[0] * speed / 100.0
                cy += direction[1] * speed / 100.0
                cx = np.clip(cx, 0.1, 0.9)
                cy = np.clip(cy, 0.1, 0.9)
                frame = self.simulate_contact((cx, cy), radius=0.2, peak_pressure=peak_pressure)
                frames.append(frame)

        return frames

    def simulate_slip_detection(self, start_position=None, end_position=None, slip_velocity=1.0, velocity=None, peak_pressure=10.0, normal_force=None, friction_coeff=None):
        """滑动检测模拟

        Args:
            start_position: 起始位置 (x, y) - 归一化
            end_position: 结束位置 (x, y) - 归一化
            slip_velocity: 滑移速度
            velocity: 别名 for slip_velocity
            peak_pressure: 峰值压力 (kPa)
            normal_force: 别名 for peak_pressure
            friction_coeff: 摩擦系数 (0-1)

        Returns:
            dict with slip_state, slip_probability, etc.
        """
        if velocity is not None:
            slip_velocity = velocity
        if normal_force is not None:
            peak_pressure = normal_force
        friction_coeff = friction_coeff if friction_coeff is not None else 0.3

        # 简化的库伦摩擦模型
        # slip_probability 基于法向力和摩擦系数的比值
        if friction_coeff <= 0:
            slip_state = "sliding"
            slip_prob = 1.0
        elif peak_pressure < 5.0:
            slip_state = "sliding"
            slip_prob = 0.9
        elif peak_pressure < 10.0:
            slip_state = "micro_slip"
            slip_prob = 0.4
        else:
            slip_state = "stick"
            slip_prob = 0.1

        return {
            'slip_state': slip_state,
            'slip_probability': slip_prob,
            'overall': 1.0 - slip_prob,
            'contact_stability': 1.0 - slip_prob * 0.5,
        }

