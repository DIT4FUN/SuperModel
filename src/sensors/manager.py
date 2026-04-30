# Copyright (C) 2024-2026 赵元请 (DIT4FUN)
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
统一传感器管理器
================

整合所有传感器，提供统一的采集接口
- 自动传感器初始化
- 同步/异步采集
- 数据预处理管道
- 传感器健康监控

支持 AGV 五级规格 (S/M/L/XL/XXL)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import threading
import time


class SensorGrade(Enum):
    """传感器等级"""
    S = "S"
    M = "M"
    L = "L"
    XL = "XL"
    XXL = "XXL"


@dataclass
class SensorDataFrame:
    """
    统一数据帧
    
    包含所有传感器数据的融合时间戳版本
    """
    timestamp: float
    frame_id: int
    
    # 视觉
    vision: Optional[Any] = None          # StereoFrame
    vision_encoded: Optional[np.ndarray] = None
    
    # 听觉
    audio: Optional[Any] = None          # AudioFrame
    audio_encoded: Optional[np.ndarray] = None
    
    # 触觉
    tactile: Optional[Any] = None        # TactileFrame
    tactile_encoded: Optional[np.ndarray] = None
    
    # 力觉
    force: Optional[Any] = None          # Wrench
    force_encoded: Optional[np.ndarray] = None
    
    # IMU
    imu: Optional[Any] = None           # IMUFrame
    imu_encoded: Optional[np.ndarray] = None
    
    # 健康状态
    healthy: Dict[str, bool] = field(default_factory=dict)
    latencies_ms: Dict[str, float] = field(default_factory=dict)
    
    def get_modalities(self) -> List[str]:
        """返回可用模态列表"""
        mods = []
        if self.vision is not None:
            mods.append("vision")
        if self.audio is not None:
            mods.append("audio")
        if self.tactile is not None:
            mods.append("tactile")
        if self.force is not None:
            mods.append("force")
        if self.imu is not None:
            mods.append("imu")
        return mods
    
    def is_healthy(self) -> bool:
        """检查整体健康状态"""
        if not self.healthy:
            return True
        return all(self.healthy.values())


class SensorManagerConfig:
    """传感器管理器配置"""
    
    def __init__(self, grade: str = "M"):
        self.grade = SensorGrade(grade)
        
        # 传感器配置
        self.vision_enabled = True
        self.audio_enabled = True
        self.tactile_enabled = True
        self.force_enabled = True
        self.imu_enabled = True
        
        # 采集参数
        self.capture_rate_hz = self._get_rate()
        self.timeout_seconds = 1.0 / self.capture_rate_hz * 2
        
        # 预处理
        self.denoise = True
        self.calibrate = True
        self.encode_on_capture = False  # 是否捕获时直接编码
    
    def _get_rate(self) -> float:
        rates = {
            SensorGrade.S: 10,
            SensorGrade.M: 30,
            SensorGrade.L: 60,
            SensorGrade.XL: 100,
            SensorGrade.XXL: 200,
        }
        return rates.get(self.grade, 30)


class SensorManager:
    """
    统一传感器管理器
    
    统一管理所有传感器，提供:
    - 同步采集所有传感器
    - 异步采集管道
    - 健康监控
    - 自动重连
    
    使用示例:
    ```python
    manager = SensorManager(grade="M")
    manager.open_all()
    
    # 同步采集
    frame = manager.capture_all()
    
    # 异步采集
    manager.start_async_capture()
    for frame in manager.frame_generator():
        process(frame)
    
    manager.close_all()
    ```
    """
    
    def __init__(self, config: Optional[SensorManagerConfig] = None):
        self.config = config or SensorManagerConfig()
        
        # 传感器实例
        self._vision_sensor = None
        self._audio_sensor = None
        self._tactile_sensor = None
        self._force_sensor = None
        self._imu_sensor = None
        
        # 编码器
        self._encoders = {}
        
        # 状态
        self._is_open = False
        self._is_async_running = False
        self._async_thread: Optional[threading.Thread] = None
        self._frame_queue: List[SensorDataFrame] = []
        self._queue_lock = threading.Lock()
        self._frame_id = 0
        
        # 健康监控
        self._sensor_health: Dict[str, bool] = {}
        self._sensor_last_ts: Dict[str, float] = {}
        self._error_counts: Dict[str, int] = {}
        
        # 回调
        self._callbacks: Dict[str, List[Callable]] = {}
    
    # ─── 初始化 ────────────────────────────────────────────────
    
    def _import_and_create_sensors(self):
        """延迟导入并创建传感器"""
        # 视觉
        if self.config.vision_enabled:
            try:
                from sensors.vision import BinocularCamera
                self._vision_sensor = BinocularCamera()
            except ImportError:
                self._vision_sensor = None
        
        # 听觉
        if self.config.audio_enabled:
            try:
                from sensors.audio import BinauralMic
                self._audio_sensor = BinauralMic()
            except ImportError:
                self._audio_sensor = None
        
        # 触觉
        if self.config.tactile_enabled:
            try:
                from sensors.tactile import TactileArray
                arr_spec = self._get_tactile_spec()
                self._tactile_sensor = TactileArray(array_size=arr_spec)
            except ImportError:
                self._tactile_sensor = None
        
        # 力觉
        if self.config.force_enabled:
            try:
                from sensors.force import ForceTorqueSensor
                self._force_sensor = ForceTorqueSensor()
            except ImportError:
                self._force_sensor = None
        
        # IMU
        if self.config.imu_enabled:
            try:
                from sensors.imu import IMUSensor
                self._imu_sensor = IMUSensor()
            except ImportError:
                self._imu_sensor = None
    
    def _get_tactile_spec(self) -> Tuple[int, int]:
        specs = {
            SensorGrade.S: (8, 8),
            SensorGrade.M: (16, 16),
            SensorGrade.L: (24, 24),
            SensorGrade.XL: (32, 32),
            SensorGrade.XXL: (48, 48),
        }
        return specs.get(self.config.grade, (16, 16))
    
    # ─── 生命周期 ──────────────────────────────────────────────
    
    def open_all(self) -> bool:
        """打开所有传感器"""
        if self._is_open:
            return True
        
        self._import_and_create_sensors()
        
        errors = []
        
        if self._vision_sensor:
            try:
                self._vision_sensor.open()
                self._sensor_health["vision"] = True
            except Exception as e:
                errors.append(f"vision: {e}")
                self._sensor_health["vision"] = False
        
        if self._audio_sensor:
            try:
                self._audio_sensor.open()
                self._sensor_health["audio"] = True
            except Exception as e:
                errors.append(f"audio: {e}")
                self._sensor_health["audio"] = False
        
        if self._tactile_sensor:
            try:
                self._tactile_sensor.open()
                self._sensor_health["tactile"] = True
            except Exception as e:
                errors.append(f"tactile: {e}")
                self._sensor_health["tactile"] = False
        
        if self._force_sensor:
            try:
                self._force_sensor.open()
                self._sensor_health["force"] = True
            except Exception as e:
                errors.append(f"force: {e}")
                self._sensor_health["force"] = False
        
        if self._imu_sensor:
            try:
                self._imu_sensor.open()
                self._sensor_health["imu"] = True
                # IMU 需要预热
                for _ in range(10):
                    self._imu_sensor.capture()
            except Exception as e:
                errors.append(f"imu: {e}")
                self._sensor_health["imu"] = False
        
        self._is_open = True
        
        if errors:
            print(f"[SensorManager] Warnings: {errors}")
        
        return True
    
    def close_all(self):
        """关闭所有传感器"""
        self.stop_async_capture()
        
        for sensor, name in [
            (self._vision_sensor, "vision"),
            (self._audio_sensor, "audio"),
            (self._tactile_sensor, "tactile"),
            (self._force_sensor, "force"),
            (self._imu_sensor, "imu"),
        ]:
            if sensor is not None:
                try:
                    sensor.close()
                except Exception:
                    pass
        
        self._is_open = False
    
    # ─── 采集 ──────────────────────────────────────────────────
    
    def capture_all(self) -> SensorDataFrame:
        """
        同步采集所有传感器数据
        
        Returns:
            SensorDataFrame: 包含所有模态数据的帧
        """
        now = time.time()
        frame_id = self._frame_id
        self._frame_id += 1
        
        healthy = {}
        latencies = {}
        frame = SensorDataFrame(
            timestamp=now,
            frame_id=frame_id,
            healthy=healthy,
            latencies_ms=latencies,
        )
        
        # 视觉
        if self._vision_sensor and self._sensor_health.get("vision"):
            t0 = time.time()
            try:
                frame.vision = self._vision_sensor.capture()
                healthy["vision"] = True
                latencies["vision"] = (time.time() - t0) * 1000
                self._sensor_last_ts["vision"] = now
            except Exception as e:
                healthy["vision"] = False
                self._error_counts["vision"] = self._error_counts.get("vision", 0) + 1
        
        # 听觉
        if self._audio_sensor and self._sensor_health.get("audio"):
            t0 = time.time()
            try:
                frame.audio = self._audio_sensor.capture()
                healthy["audio"] = True
                latencies["audio"] = (time.time() - t0) * 1000
                self._sensor_last_ts["audio"] = now
            except Exception as e:
                healthy["audio"] = False
                self._error_counts["audio"] = self._error_counts.get("audio", 0) + 1
        
        # 触觉
        if self._tactile_sensor and self._sensor_health.get("tactile"):
            t0 = time.time()
            try:
                frame.tactile = self._tactile_sensor.capture()
                healthy["tactile"] = True
                latencies["tactile"] = (time.time() - t0) * 1000
                self._sensor_last_ts["tactile"] = now
            except Exception as e:
                healthy["tactile"] = False
                self._error_counts["tactile"] = self._error_counts.get("tactile", 0) + 1
        
        # 力觉
        if self._force_sensor and self._sensor_health.get("force"):
            t0 = time.time()
            try:
                frame.force = self._force_sensor.capture()
                healthy["force"] = True
                latencies["force"] = (time.time() - t0) * 1000
                self._sensor_last_ts["force"] = now
            except Exception as e:
                healthy["force"] = False
                self._error_counts["force"] = self._error_counts.get("force", 0) + 1
        
        # IMU
        if self._imu_sensor and self._sensor_health.get("imu"):
            t0 = time.time()
            try:
                frame.imu = self._imu_sensor.capture()
                healthy["imu"] = True
                latencies["imu"] = (time.time() - t0) * 1000
                self._sensor_last_ts["imu"] = now
            except Exception as e:
                healthy["imu"] = False
                self._error_counts["imu"] = self._error_counts.get("imu", 0) + 1
        
        return frame
    
    def capture_single(self, modality: str) -> Any:
        """采集单个模态"""
        if modality == "vision" and self._vision_sensor:
            return self._vision_sensor.capture()
        elif modality == "audio" and self._audio_sensor:
            return self._audio_sensor.capture()
        elif modality == "tactile" and self._tactile_sensor:
            return self._tactile_sensor.capture()
        elif modality == "force" and self._force_sensor:
            return self._force_sensor.capture()
        elif modality == "imu" and self._imu_sensor:
            return self._imu_sensor.capture()
        return None
    
    # ─── 异步采集 ────────────────────────────────────────────────
    
    def start_async_capture(self, queue_size: int = 10):
        """启动异步采集线程"""
        if self._is_async_running:
            return
        
        self._is_async_running = True
        self._frame_queue.clear()
        
        def _capture_loop():
            target_interval = 1.0 / self.config.capture_rate_hz
            while self._is_async_running:
                t0 = time.time()
                frame = self.capture_all()
                
                with self._queue_lock:
                    self._frame_queue.append(frame)
                    if len(self._frame_queue) > queue_size:
                        self._frame_queue.pop(0)
                
                elapsed = time.time() - t0
                sleep_time = target_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        self._async_thread = threading.Thread(target=_capture_loop, daemon=True)
        self._async_thread.start()
    
    def stop_async_capture(self):
        """停止异步采集"""
        self._is_async_running = False
        if self._async_thread:
            self._async_thread.join(timeout=2.0)
            self._async_thread = None
    
    def frame_generator(self, max_queue: int = 5):
        """
        异步帧生成器
        
        Yields:
            SensorDataFrame
        """
        while self._is_async_running:
            with self._queue_lock:
                if self._frame_queue:
                    yield self._frame_queue.pop(0)
            time.sleep(0.001)
    
    def get_latest_frame(self) -> Optional[SensorDataFrame]:
        """获取最新帧 (非阻塞)"""
        with self._queue_lock:
            if self._frame_queue:
                return self._frame_queue[-1]
        return None
    
    # ─── 健康监控 ────────────────────────────────────────────────
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取传感器健康状态"""
        return {
            "vision": self._sensor_health.get("vision", False),
            "audio": self._sensor_health.get("audio", False),
            "tactile": self._sensor_health.get("tactile", False),
            "force": self._sensor_health.get("force", False),
            "imu": self._sensor_health.get("imu", False),
            "errors": dict(self._error_counts),
            "last_timestamps": dict(self._sensor_last_ts),
            "async_running": self._is_async_running,
        }
    
    def check_sensor_alive(self, modality: str, timeout: float = 5.0) -> bool:
        """检查传感器是否存活"""
        last_ts = self._sensor_last_ts.get(modality, 0)
        if last_ts == 0:
            return self._sensor_health.get(modality, False)
        return (time.time() - last_ts) < timeout
    
    def register_callback(self, modality: str, callback: Callable):
        """注册传感器回调"""
        if modality not in self._callbacks:
            self._callbacks[modality] = []
        self._callbacks[modality].append(callback)
    
    def __enter__(self):
        self.open_all()
        return self
    
    def __exit__(self, *args):
        self.close_all()
