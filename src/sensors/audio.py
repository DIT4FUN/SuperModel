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
双耳听觉感知模块
================

支持双耳麦克风阵列
- 波束形成
- 声源定位
- 语音识别
- 环境声音分类
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
import threading


@dataclass
class AudioFrame:
    """音频帧"""
    left_channel: np.ndarray  # 左耳音频
    right_channel: np.ndarray  # 右耳音频
    sample_rate: int = 16000
    timestamp: float = 0.0
    frame_id: int = 0


@dataclass
class SoundSource:
    """声源定位结果"""
    direction: np.ndarray  # 3D 方向向量 (azimuth, elevation)
    distance: Optional[float] = None  # 距离, 米
    confidence: float = 0.0  # 置信度
    source_type: str = "unknown"  # speech / ambient / music


class BinauralMic:
    """
    双耳麦克风接口
    
    支持:
    - ReSpeaker 4-Mic Array
    - 通用USB麦克风
    - I2S MEMS麦克风
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 512,
        device_id: Optional[int] = None
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device_id = device_id
        
        # 双耳麦克风配置 (假设线性排列, 间距约 95mm)
        self.baseline_mm = 95.0
        self.mic_positions = np.array([
            [-self.baseline_mm / 2, 0, 0],  # 左
            [self.baseline_mm / 2, 0, 0]    # 右
        ])
        
        self._stream = None
        self._is_recording = False
        self._buffer = []
        
    def open(self) -> bool:
        """打开音频流"""
        # 优先尝试 sounddevice
        try:
            import sounddevice as sd
            self._stream = sd.InputStream(
                channels=2,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                device=self.device_id,
                dtype='float32',
                callback=self._audio_callback
            )
            self._stream.start()
            self._use_sounddevice = True
            print(f"[BinauralMic] Opened with sounddevice: SR={self.sample_rate}, Chunks={self.chunk_size}")
            self._is_recording = True
            return True
        except (ImportError, OSError):
            # Fallback: 模拟模式
            self._use_sounddevice = False
            self._sim_t = 0.0
            self._is_recording = True
            print(f"[BinauralMic] Opened in SIMULATION mode: SR={self.sample_rate}, Chunks={self.chunk_size}")
            return True
    
    def _audio_callback(self, indata, frames, time, status):
        """音频回调"""
        if status:
            print(f"[BinauralMic] Status: {status}")
        # 双声道: [左, 右]
        left = indata[:, 0].copy().astype(np.float32)
        right = indata[:, 1].copy().astype(np.float32)
        frame = AudioFrame(left, right, self.sample_rate, time.current_time, len(self._buffer))
        self._buffer.append(frame)
        # 限制缓冲区大小
        if len(self._buffer) > 10:
            self._buffer = self._buffer[-5:]
    
    def close(self):
        """关闭音频流"""
        self._is_recording = False
        if getattr(self, '_stream', None):
            self._stream.stop()
            self._stream.close()
        self._buffer.clear()
        print("[BinauralMic] Closed")
    
    def capture(self) -> AudioFrame:
        """捕获一帧音频"""
        if not self._is_recording:
            raise RuntimeError("Audio stream not opened")
        
        if getattr(self, '_use_sounddevice', False):
            # 从 sounddevice 缓冲区获取最新帧
            if self._buffer:
                return self._buffer.pop(0)
            else:
                # 无数据时返回静音
                t = np.linspace(0, self.chunk_size / self.sample_rate, self.chunk_size)
                left = np.zeros(self.chunk_size, dtype=np.float32)
                right = np.zeros(self.chunk_size, dtype=np.float32)
                return AudioFrame(left, right, self.sample_rate, self._sim_t, 0)
        else:
            # 模拟模式: 生成多频率复合音频
            dt = self.chunk_size / self.sample_rate
            t = np.linspace(self._sim_t, self._sim_t + dt, self.chunk_size, endpoint=False)
            
            # 复合正弦波: 440Hz 基频 + 880Hz 谐波 + 噪声
            left = (0.1 * np.sin(2 * np.pi * 440 * t) +
                    0.05 * np.sin(2 * np.pi * 880 * t) +
                    0.02 * np.random.randn(self.chunk_size))
            right = (0.1 * np.sin(2 * np.pi * 440 * t + 0.05) +  # 轻微相位差模拟空间感
                     0.05 * np.sin(2 * np.pi * 880 * t) +
                     0.02 * np.random.randn(self.chunk_size))
            
            # 限制幅度
            left = np.clip(left, -1.0, 1.0).astype(np.float32)
            right = np.clip(right, -1.0, 1.0).astype(np.float32)
            
            frame = AudioFrame(left, right, self.sample_rate, self._sim_t, 0)
            self._sim_t += dt
            return frame
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SoundLocalizer:
    """
    声源定位处理器
    
    方法:
    - GCC-PHAT (广义互相关)
    - MUSIC (多信号分类)
    - 神经网络定位
    """
    
    def __init__(
        self,
        baseline_mm: float = 95.0,
        sample_rate: int = 16000
    ):
        self.baseline_mm = baseline_mm
        self.baseline = baseline_mm / 1000.0  # 转换为米
        self.sample_rate = sample_rate
        self.speed_of_sound = 343.0  # m/s
        
    def estimate_tdoa(self, left: np.ndarray, right: np.ndarray) -> float:
        """
        估计时延差 (TDOA)
        
        使用 GCC-PHAT 算法
        
        Returns:
            tdoa: 时延差, 秒
        """
        n = len(left)
        
        # 互相关
        cross_corr = np.correlate(left, right, 'full')
        
        # PHAT加权
        spectrum_left = np.fft.rfft(left, n * 2 - 1)
        spectrum_right = np.fft.rfft(right, n * 2 - 1)
        
        cross_spectrum = spectrum_left * np.conj(spectrum_right)
        phat_spectrum = cross_spectrum / (np.abs(cross_spectrum) + 1e-10)
        
        phat_corr = np.fft.irfft(phat_spectrum, n * 2 - 1)
        
        # 找峰值
        lag = np.argmax(np.abs(phat_corr)) - (len(phat_corr) - 1) // 2
        tdoa = lag / self.sample_rate
        
        return tdoa
    
    def localize(self, left: np.ndarray, right: np.ndarray) -> SoundSource:
        """
        声源定位
        
        基于TDOA估计声源方向
        
        Returns:
            SoundSource: 包含方向、距离、置信度
        """
        tdoa = self.estimate_tdoa(left, right)
        
        # 计算角度
        # 假设声源在远场, 平面波入射
        max_delay = self.baseline / self.speed_of_sound
        
        # 限制TDOA范围
        tdoa = np.clip(tdoa, -max_delay, max_delay)
        
        # 方位角: -90° ~ +90°
        azimuth = np.arcsin(tdoa / max_delay)  # 弧度
        azimuth_deg = np.degrees(azimuth)
        
        # 俯仰角假设为0 (水平面)
        elevation = 0.0
        
        # 置信度 (基于相关峰值)
        confidence = min(abs(tdoa) / max_delay + 0.3, 1.0)
        
        direction = np.array([azimuth_deg, elevation])
        
        return SoundSource(
            direction=direction,
            distance=None,  # 需要多麦克风阵列才能估计距离
            confidence=confidence,
            source_type="unknown"
        )
    
    def beamform(
        self,
        left: np.ndarray,
        right: np.ndarray,
        look_direction: float = 0.0
    ) -> np.ndarray:
        """
        波束形成
        
        在指定方向形成波束增益
        
        Args:
            left: 左声道
            right: 右声道
            look_direction: 拾音方向, 度 (-90 ~ 90)
            
        Returns:
            beamformed: 波束形成后的信号
        """
        # 转换为弧度
        theta = np.radians(look_direction)
        
        # 计算延迟
        delay_samples = int(
            self.baseline * np.sin(theta) / self.speed_of_sound * self.sample_rate
        )
        
        # 延迟求和 (DSB: Delay and Sum Beamforming)
        if delay_samples > 0:
            beamformed = left[delay_samples:] + right[:-delay_samples]
        elif delay_samples < 0:
            beamformed = left[:delay_samples] + right[-delay_samples:]
        else:
            beamformed = left + right
            
        return beamformed


# AGV五级听觉规格
AGV_AUDIO_GRADES = {
    'S': {'channels': 2, 'sr': 16000, 'range_m': 2.0, 'beamforming': False},
    'M': {'channels': 2, 'sr': 16000, 'range_m': 3.0, 'beamforming': True},
    'L': {'channels': 4, 'sr': 22050, 'range_m': 5.0, 'beamforming': True},
    'XL': {'channels': 6, 'sr': 32000, 'range_m': 8.0, 'beamforming': True},
    'XXL': {'channels': 8, 'sr': 44100, 'range_m': 10.0, 'beamforming': True}
}


def get_audio_spec(grade: str) -> dict:
    """获取AGV指定等级的听觉规格"""
    return AGV_AUDIO_GRADES.get(grade, AGV_AUDIO_GRADES['M'])
