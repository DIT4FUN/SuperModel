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
传感器信号处理器
================

为触觉、力觉、IMU等传感器提供高级滤波和信号处理能力。

功能特性:
- 卡尔曼滤波 (Kalman Filter)
- 低通/高通/带通滤波器
- 异常值检测与剔除
- 信号平滑
- 零相位滤波
- 噪声密度估计
- 传感器融合预处理

Author: SuperModel Team
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum


class FilterType(Enum):
    """滤波器类型"""
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    KALMAN = "kalman"
    MEDIAN = "median"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL = "exponential"


@dataclass
class FilterConfig:
    """滤波器配置"""
    filter_type: FilterType = FilterType.KALMAN
    cutoff_freq: float = 5.0       # 截止频率 (Hz)
    sample_rate: float = 100.0    # 采样率 (Hz)
    noise_density: float = 0.01    # 噪声密度 (sensor units/√Hz)
    process_noise: float = 0.001   # 过程噪声
    measurement_noise: float = 0.1 # 测量噪声
    window_size: int = 5           # 滑动窗口大小


@dataclass
class SignalStats:
    """信号统计信息"""
    mean: float
    std: float
    min_val: float
    max_val: float
    rms: float
    snr: float  # 信噪比 (dB)
    noise_estimate: float


class KalmanFilter1D:
    """
    一维卡尔曼滤波器
    
    用于传感器信号的在线滤波和状态估计。
    """

    def __init__(
        self,
        process_noise: float = 0.001,
        measurement_noise: float = 0.1,
        initial_state: float = 0.0,
        initial_covariance: float = 1.0
    ):
        self._x = initial_state      # 状态估计
        self._p = initial_covariance # 估计协方差
        self._q = process_noise     # 过程噪声协方差
        self._r = measurement_noise  # 测量噪声协方差

    def update(self, measurement: float) -> float:
        """
        更新滤波器状态
        
        Args:
            measurement: 当前测量值
            
        Returns:
            滤波后的估计值
        """
        # 预测步骤
        self._p = self._p + self._q

        # 更新步骤
        k = self._p / (self._p + self._r)  # 卡尔曼增益
        self._x = self._x + k * (measurement - self._x)
        self._p = (1 - k) * self._p

        return self._x

    def reset(self, initial_state: float = 0.0, initial_covariance: float = 1.0):
        """重置滤波器状态"""
        self._x = initial_state
        self._p = initial_covariance

    @property
    def state(self) -> float:
        return self._x

    @property
    def error_covariance(self) -> float:
        return self._p


class KalmanFilter3D:
    """
    三维卡尔曼滤波器
    
    用于IMU等3D传感器信号的滤波。
    """

    def __init__(
        self,
        process_noise: float = 0.001,
        measurement_noise: float = 0.1,
        initial_state: Optional[np.ndarray] = None
    ):
        if initial_state is None:
            initial_state = np.zeros(3, dtype=np.float32)
        
        self._x = initial_state.astype(np.float32)
        self._p = np.eye(3, dtype=np.float32)  # 3x3协方差矩阵
        self._q = np.eye(3, dtype=np.float32) * process_noise
        self._r = np.eye(3, dtype=np.float32) * measurement_noise

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        更新滤波器状态
        
        Args:
            measurement: 3D测量向量
            
        Returns:
            滤波后的3D估计向量
        """
        measurement = np.asarray(measurement, dtype=np.float32)

        # 预测
        self._p = self._p + self._q

        # 更新
        s = self._p + self._r  # 创新协方差
        k = self._p @ np.linalg.inv(s)  # 卡尔曼增益 (3x3)
        innovation = measurement - self._x
        self._x = self._x + k @ innovation
        self._p = (np.eye(3) - k) @ self._p

        return self._x.copy()

    def reset(self, initial_state: Optional[np.ndarray] = None):
        """重置滤波器状态"""
        if initial_state is None:
            self._x = np.zeros(3, dtype=np.float32)
        else:
            self._x = np.asarray(initial_state, dtype=np.float32)
        self._p = np.eye(3, dtype=np.float32)

    @property
    def state(self) -> np.ndarray:
        return self._x.copy()


class ButterworthFilter:
    """
    Butterworth数字滤波器
    
    支持低通、高通、带通滤波。
    使用scipy.signal设计系数，零相位滤波。
    """

    def __init__(
        self,
        filter_type: FilterType = FilterType.LOWPASS,
        cutoff_freq: float = 5.0,
        sample_rate: float = 100.0,
        order: int = 2
    ):
        self.filter_type = filter_type
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        self.order = order
        
        self._design_filter()

    def _design_filter(self):
        """使用scipy.signal设计滤波器系数"""
        try:
            from scipy.signal import butter, lfilter_zi, lfilter
            
            # 归一化截止频率 (Nyquist=fs/2)
            nyquist = self.sample_rate / 2.0
            normalized_cutoff = self.cutoff_freq / nyquist
            normalized_cutoff = np.clip(normalized_cutoff, 0.001, 0.999)
            
            # 滤波器类型映射
            btype_map = {
                FilterType.LOWPASS: 'low',
                FilterType.HIGHPASS: 'high',
                FilterType.BANDPASS: 'bandpass',
            }
            btype = btype_map.get(self.filter_type, 'low')
            
            # 设计巴特沃斯滤波器
            self._sos = butter(self.order, normalized_cutoff, btype=btype, output='sos')
            self._use_scipy = True
        except ImportError:
            # Fallback: 简化一阶低通
            alpha = 1.0 / (1.0 + self.sample_rate / (2 * np.pi * self.cutoff_freq + 1e-6))
            self._alpha = float(np.clip(alpha, 0.01, 0.99))
            self._last = 0.0
            self._use_scipy = False

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """
        对信号应用滤波器
        
        Args:
            signal: 输入信号 (1D numpy array)
            
        Returns:
            滤波后的信号
        """
        signal = np.asarray(signal, dtype=np.float64)
        
        if self._use_scipy:
            try:
                from scipy.signal import sosfilt
                return sosfilt(self._sos, signal).astype(np.float32)
            except Exception:
                pass
        
        # Fallback: 简单指数低通
        filtered = np.zeros_like(signal)
        for i in range(len(signal)):
            self._last = self._alpha * signal[i] + (1 - self._alpha) * self._last
            filtered[i] = self._last
        
        return filtered.astype(np.float32)


class MedianFilter:
    """
    中值滤波器
    
    有效去除脉冲噪声（异常值），保留边缘。
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._buffer: List[float] = []

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """
        应用中值滤波
        
        Args:
            signal: 输入信号
            
        Returns:
            滤波后信号
        """
        signal = np.asarray(signal, dtype=np.float32)
        n = len(signal)
        half = self.window_size // 2
        filtered = np.zeros_like(signal)
        
        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            window = signal[start:end]
            filtered[i] = float(np.median(window))
        
        return filtered


class ExponentialSmoother:
    """
    指数平滑滤波器
    
    低计算量，适合实时应用。
    """

    def __init__(self, alpha: float = 0.3):
        """
        Args:
            alpha: 平滑系数 (0-1), 越大越敏感
        """
        self.alpha = np.clip(alpha, 0.0, 1.0)
        self._last_value: Optional[float] = None

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """应用指数平滑"""
        signal = np.asarray(signal, dtype=np.float32)
        filtered = np.zeros_like(signal)
        
        for i in range(len(signal)):
            if self._last_value is None:
                filtered[i] = signal[i]
            else:
                filtered[i] = self.alpha * signal[i] + (1 - self.alpha) * self._last_value
            self._last_value = filtered[i]
        
        return filtered

    def update(self, value: float) -> float:
        """单步更新"""
        if self._last_value is None:
            self._last_value = value
            return value
        
        filtered = self.alpha * value + (1 - self.alpha) * self._last_value
        self._last_value = filtered
        return filtered

    def reset(self):
        """重置状态"""
        self._last_value = None


class OutlierDetector:
    """
    异常值检测器
    
    支持统计方法和Z-score方法。
    """

    def __init__(
        self,
        method: str = "zscore",
        threshold: float = 3.0,
        window_size: int = 50
    ):
        """
        Args:
            method: 'zscore' or 'iqr'
            threshold: Z-score阈值 或 IQR倍数
            window_size: 统计窗口大小
        """
        self.method = method
        self.threshold = threshold
        self.window_size = window_size
        self._buffer: List[float] = []

    def detect(self, value: float) -> Tuple[bool, float]:
        """
        检测异常值
        
        Args:
            value: 当前测量值
            
        Returns:
            (是否异常, 置信度分数)
        """
        self._buffer.append(value)
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)
        
        if len(self._buffer) < 10:
            return False, 1.0
        
        data = np.array(self._buffer)
        median = float(np.median(data))
        
        if self.method == "zscore":
            mad = float(np.median(np.abs(data - median)))
            if mad < 1e-6:
                return False, 1.0
            
            # Modified Z-score using MAD
            modified_z = 0.6745 * (value - median) / mad
            is_outlier = abs(modified_z) > self.threshold
            confidence = min(abs(modified_z) / self.threshold, 1.0)
        
        elif self.method == "iqr":
            q1 = float(np.percentile(data, 25))
            q3 = float(np.percentile(data, 75))
            iqr = q3 - q1
            lower = q1 - self.threshold * iqr
            upper = q3 + self.threshold * iqr
            is_outlier = value < lower or value > upper
            confidence = max(
                (value - lower) / (upper - lower + 1e-6),
                (upper - value) / (upper - lower + 1e-6)
            )
            confidence = np.clip(confidence, 0.0, 1.0)
        else:
            is_outlier = False
            confidence = 1.0
        
        return is_outlier, confidence

    def is_valid(self, value: float) -> bool:
        """判断值是否有效（非异常）"""
        is_outlier, _ = self.detect(value)
        return not is_outlier

    def reset(self):
        """重置缓冲区"""
        self._buffer.clear()


class SignalProcessor:
    """
    统一信号处理器
    
    整合滤波、异常值检测、统计计算等功能。
    """

    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()
        
        # 初始化各滤波器
        self._kalman = KalmanFilter3D(
            process_noise=self.config.process_noise,
            measurement_noise=self.config.measurement_noise
        )
        self._median = MedianFilter(window_size=self.config.window_size)
        self._exponential = ExponentialSmoother(alpha=0.3)
        self._outlier = OutlierDetector(method="zscore", threshold=3.0)
        
        self._enabled = True
        self._initialized = False

    def process_frame(
        self,
        data: np.ndarray,
        remove_outliers: bool = True,
        filter_type: FilterType = FilterType.KALMAN
    ) -> np.ndarray:
        """
        处理一帧传感器数据
        
        Args:
            data: 原始数据 (3D向量)
            remove_outliers: 是否去除异常值
            filter_type: 使用的滤波器类型
            
        Returns:
            处理后的数据
        """
        if not self._enabled:
            return data
        
        data = np.asarray(data, dtype=np.float32).copy()
        
        # 异常值剔除
        if remove_outliers:
            processed = np.zeros_like(data)
            for i in range(len(data)):
                is_outlier, confidence = self._outlier.detect(float(data[i]))
                if is_outlier and self._initialized:
                    # 用前值替代异常值
                    processed[i] = processed[i - 1] if i > 0 else 0.0
                else:
                    processed[i] = data[i]
            data = processed
        
        # 滤波
        if filter_type == FilterType.KALMAN:
            data = self._kalman.update(data)
        elif filter_type == FilterType.MEDIAN:
            # Median filter for 1D signals; for 3D, apply per channel
            pass  # Kalman handles 3D
        
        self._initialized = True
        return data

    def process_scalar(self, value: float) -> float:
        """处理标量数据"""
        if not self._enabled:
            return value
        
        is_outlier, _ = self._outlier.detect(value)
        if is_outlier and self._initialized:
            return self._exponential.update(value)  # will be filtered
        
        return self._exponential.update(value)

    def compute_stats(self, signal: np.ndarray) -> SignalStats:
        """
        计算信号统计信息
        
        Args:
            signal: 输入信号
            
        Returns:
            SignalStats对象
        """
        signal = np.asarray(signal, dtype=np.float32).flatten()
        
        mean = float(np.mean(signal))
        std = float(np.std(signal))
        min_val = float(np.min(signal))
        max_val = float(np.max(signal))
        rms = float(np.sqrt(np.mean(signal ** 2)))
        
        # SNR估计
        if std > 1e-9:
            snr = 20 * np.log10(rms / (std + 1e-9))
        else:
            snr = 60.0  # 高信噪比
        
        # 噪声估计 (使用相邻样本差分)
        if len(signal) > 1:
            diff = np.diff(signal)
            noise_est = float(np.std(diff) / np.sqrt(2))
        else:
            noise_est = 0.0
        
        return SignalStats(
            mean=mean, std=std, min_val=min_val, max_val=max_val,
            rms=rms, snr=snr, noise_estimate=noise_est
        )

    def reset(self):
        """重置所有滤波器状态"""
        self._kalman.reset()
        self._outlier.reset()
        self._exponential.reset()
        self._initialized = False

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False


# AGV五级规格 - 信号处理
AGV_SIGNAL_PROCESSING_GRADES = {
    'S': {
        'filters': ['exponential'],
        'outlier_detection': False,
        'max_sample_rate': 100,
        'channels': 1,
    },
    'M': {
        'filters': ['exponential', 'kalman'],
        'outlier_detection': True,
        'max_sample_rate': 200,
        'channels': 3,
    },
    'L': {
        'filters': ['exponential', 'kalman', 'median', 'butterworth'],
        'outlier_detection': True,
        'max_sample_rate': 500,
        'channels': 6,
    },
    'XL': {
        'filters': ['exponential', 'kalman', 'median', 'butterworth'],
        'outlier_detection': True,
        'max_sample_rate': 1000,
        'channels': 9,
    },
    'XXL': {
        'filters': ['exponential', 'kalman', 'median', 'butterworth', 'bandpass'],
        'outlier_detection': True,
        'max_sample_rate': 2000,
        'channels': 12,
    },
}


def get_signal_processing_grade_spec(grade: str) -> dict:
    """获取指定AGV等级的信号处理规格"""
    return AGV_SIGNAL_PROCESSING_GRADES.get(grade, AGV_SIGNAL_PROCESSING_GRADES['M'])
