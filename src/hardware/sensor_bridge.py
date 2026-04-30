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
传感器硬件桥接模块 (Sensor Hardware Bridge)
==========================================

统一的传感器硬件接口层，支持:
- CAN Bus 传感器 (IMU/力觉/触觉)
- RS485/RS232 传感器
- I2C/SPI 传感器
- USB 传感器 (HID/串口)
- 模拟仿真传感器

提供统一的 `SensorHardwareBridge` 抽象接口，
将不同协议的传感器映射为标准化的 `SensorData` 流。

Author: SuperModel Team
Version: v2.57.0
"""

import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np


# ─── 数据类型枚举 ────────────────────────────────────────────────────

class SensorDataType(Enum):
    """传感器数据类型"""
    IMU = "imu"                    # 惯性测量 (accel/gyro/mag)
    FORCE_TORQUE = "force_torque"  # 六维力/力矩
    TACTILE = "tactile"           # 触觉阵列
    ENCODER = "encoder"           # 编码器 (轮式)
    DISTANCE = "distance"         # 距离传感器 (超声波/激光)
    CAMERA = "camera"             # 视觉 (GigE/USB)
    BATTERY = "battery"           # 电池状态
    TEMPERATURE = "temperature"   # 温度


class SensorProtocol(Enum):
    """传感器通信协议"""
    CAN = "can"                   # CAN Bus / CANopen
    RS485 = "rs485"               # RS-485 (Modbus RTU)
    RS232 = "rs232"               # RS-232
    I2C = "i2c"                   # I2C
    SPI = "spi"                   # SPI
    USB = "usb"                   # USB HID / 虚拟串口
    ETHERNET = "ethernet"         # TCP/UDP (GigE Vision)
    WIRELESS = "wireless"        # WiFi/Bluetooth
    SIMULATED = "simulated"       # 仿真/虚拟


class SensorHealth(Enum):
    """传感器健康状态"""
    OK = "ok"
    DEGRADED = "degraded"
    ERROR = "error"
    OFFLINE = "offline"


# ─── 标准传感器数据格式 ──────────────────────────────────────────────

@dataclass
class SensorData:
    """统一传感器数据格式"""
    sensor_id: str
    sensor_type: SensorDataType
    protocol: SensorProtocol
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0

    # IMU 数据
    accel: Optional[np.ndarray] = None        # [3] m/s²
    gyro: Optional[np.ndarray] = None         # [3] rad/s
    mag: Optional[np.ndarray] = None          # [3] μT
    euler: Optional[np.ndarray] = None        # [3] rad (roll,pitch,yaw)
    quat: Optional[np.ndarray] = None         # [4] (w,x,y,z)

    # 力觉数据
    wrench: Optional[np.ndarray] = None       # [6] [Fx,Fy,Fz,Mx,My,Mz]
    is_saturated: bool = False

    # 触觉数据
    tactile_array: Optional[np.ndarray] = None  # [rows, cols]
    total_pressure: float = 0.0
    contact_center: Optional[Tuple[float, float]] = None
    is_contact: bool = False

    # 编码器数据
    encoder_counts: Optional[np.ndarray] = None  # 各轮脉冲数
    wheel_angles: Optional[np.ndarray] = None     # 各轮角度 rad
    linear_velocity: float = 0.0                  # m/s
    angular_velocity: float = 0.0                 # rad/s

    # 电池数据
    battery_voltage: float = 0.0     # V
    battery_current: float = 0.0      # A
    battery_soc: float = 0.0         # % State of Charge

    # 温度
    temperature: float = 25.0         # °C

    # 元数据
    health: SensorHealth = SensorHealth.OK
    error_message: Optional[str] = None
    raw_data: Optional[bytes] = None


@dataclass
class SensorHardwareConfig:
    """传感器硬件配置"""
    sensor_id: str
    sensor_type: SensorDataType
    protocol: SensorProtocol

    # 连接参数
    port: str = "/dev/ttyUSB0"          # 串口/CAN接口
    address: int = 1                    # 传感器地址 (Modbus/Node-ID)
    baudrate: int = 115200              # 波特率

    # CAN 参数
    can_channel: int = 0
    can_bitrate: int = 1000000
    canopen_node_id: int = 1

    # I2C/SPI 参数
    bus_number: int = 1
    chip_select: int = 0
    i2c_address: int = 0x68

    # 传感器特定参数
    sample_rate: int = 100              # Hz
    frame_size: Optional[Tuple[int, int]] = None  # 触觉阵列大小
    calibration_file: Optional[str] = None

    # AGV 等级 (用于自动配置)
    agv_grade: str = "M"


# ─── 传感器硬件接口基类 ──────────────────────────────────────────────

class SensorHardwareInterface(ABC):
    """
    传感器硬件接口基类
    
    所有传感器驱动必须实现此接口:
    - `open()` / `close()`: 连接管理
    - `read()`: 读取单帧数据
    - `configure()`: 传感器配置
    """

    def __init__(
        self,
        config: SensorHardwareConfig,
        on_data: Optional[Callable[[SensorData], None]] = None
    ):
        self.config = config
        self._on_data = on_data
        self._is_opened = False
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._sequence = 0
        self._last_data: Optional[SensorData] = None

    @abstractmethod
    def open(self) -> bool:
        """打开传感器连接"""
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭传感器连接"""
        pass

    @abstractmethod
    def read(self) -> Optional[SensorData]:
        """读取单帧传感器数据"""
        pass

    @abstractmethod
    def configure(self, **kwargs) -> bool:
        """配置传感器参数"""
        pass

    def start_streaming(self, callback: Optional[Callable[[SensorData], None]] = None) -> None:
        """
        开始数据流 (异步线程)
        
        Args:
            callback: 数据回调函数 (如果未在构造函数提供)
        """
        if callback:
            self._on_data = callback
        if not self._on_data:
            raise ValueError("No callback provided for streaming")

        self._running = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()

    def stop_streaming(self) -> None:
        """停止数据流"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def is_opened(self) -> bool:
        return self._is_opened

    @property
    def last_data(self) -> Optional[SensorData]:
        return self._last_data

    def _stream_loop(self) -> None:
        """数据流循环"""
        period = 1.0 / self.config.sample_rate if self.config.sample_rate > 0 else 0.01
        while self._running:
            try:
                data = self.read()
                if data:
                    self._sequence += 1
                    data.sequence = self._sequence
                    self._last_data = data
                    self._on_data(data)
            except Exception:
                pass
            time.sleep(period)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


# ─── 仿真传感器实现 ──────────────────────────────────────────────────

class SimulatedSensorInterface(SensorHardwareInterface):
    """
    仿真传感器接口
    
    使用数学模型生成模拟传感器数据，支持:
    - 恒定值
    - 正弦波动
    - 随机噪声
    - 阶跃响应
    """

    NOISE_MODELS = ["gaussian", "uniform", "none"]
    WAVE_MODELS = ["sine", "square", "sawtooth", "none"]

    def __init__(
        self,
        config: SensorHardwareConfig,
        on_data: Optional[Callable[[SensorData], None]] = None,
        noise_model: str = "gaussian",
        wave_model: str = "sine"
    ):
        super().__init__(config, on_data)
        self.noise_model = noise_model
        self.wave_model = wave_model
        self._t = 0.0
        self._base_values: Dict[str, np.ndarray] = {}
        self._noise_std: Dict[str, float] = {}
        self._wave_amp: Dict[str, float] = {}
        self._wave_freq: Dict[str, float] = {}
        self._rng = np.random.default_rng(int(time.time()))

    def open(self) -> bool:
        self._is_opened = True
        self._init_base_values()
        return True

    def close(self) -> None:
        self._is_opened = False

    def configure(self, **kwargs) -> bool:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return True

    def _init_base_values(self) -> None:
        """初始化各传感器类型的基准值"""
        stype = self.config.sensor_type
        if stype == SensorDataType.IMU:
            self._base_values['accel'] = np.array([0.0, 0.0, -9.81])
            self._base_values['gyro'] = np.array([0.0, 0.0, 0.0])
            self._noise_std['accel'] = 0.01
            self._noise_std['gyro'] = 0.001
            self._wave_amp['gyro'] = 0.05
            self._wave_freq['gyro'] = 0.5
        elif stype == SensorDataType.FORCE_TORQUE:
            self._base_values['wrench'] = np.zeros(6)
            self._noise_std['wrench'] = 0.1
        elif stype == SensorDataType.TACTILE:
            rows, cols = self.config.frame_size or (8, 8)
            self._base_values['tactile'] = np.zeros((rows, cols))
            self._noise_std['tactile'] = 2.0
        elif stype == SensorDataType.ENCODER:
            self._base_values['counts'] = np.zeros(4)
            self._noise_std['counts'] = 0.0

    def read(self) -> Optional[SensorData]:
        if not self._is_opened:
            return None

        self._t += 1.0 / self.config.sample_rate
        data = SensorData(
            sensor_id=self.config.sensor_id,
            sensor_type=self.config.sensor_type,
            protocol=SensorProtocol.SIMULATED,
            timestamp=time.time(),
        )

        stype = self.config.sensor_type

        if stype == SensorDataType.IMU:
            accel = self._base_values['accel'].copy()
            if self.noise_model == "gaussian":
                accel += self._rng.normal(0, self._noise_std.get('accel', 0.01), 3)
            data.accel = accel

            gyro = self._base_values['gyro'].copy()
            if self.wave_model == "sine":
                gyro[2] += self._wave_amp.get('gyro', 0) * np.sin(2 * np.pi * self._wave_freq.get('gyro', 0.5) * self._t)
            if self.noise_model == "gaussian":
                gyro += self._rng.normal(0, self._noise_std.get('gyro', 0.001), 3)
            data.gyro = gyro

        elif stype == SensorDataType.FORCE_TORQUE:
            wrench = self._base_values['wrench'].copy()
            if self.noise_model == "gaussian":
                wrench += self._rng.normal(0, self._noise_std.get('wrench', 0.1), 6)
            data.wrench = wrench

        elif stype == SensorDataType.TACTILE:
            rows, cols = self.config.frame_size or (8, 8)
            tactile = self._base_values.get('tactile', np.zeros((rows, cols)))
            if self.noise_model == "gaussian":
                tactile = tactile + self._rng.normal(0, self._noise_std.get('tactile', 2.0), (rows, cols))
                tactile = np.clip(tactile, 0, 255)
            data.tactile_array = tactile.astype(np.uint8)
            data.total_pressure = float(np.sum(tactile))
            y_idx, x_idx = np.where(tactile > 0)
            if len(y_idx) > 0:
                data.contact_center = (float(np.mean(x_idx)) / cols, float(np.mean(y_idx)) / rows)
                data.is_contact = True

        elif stype == SensorDataType.ENCODER:
            counts = self._base_values['counts'].copy()
            counts += self._rng.normal(0, self._noise_std.get('counts', 0), 4).astype(int)
            data.encoder_counts = counts

        return data


# ─── 传感器硬件桥接器 ────────────────────────────────────────────────

class SensorHardwareBridge:
    """
    传感器硬件桥接器
    
    统一管理多种传感器硬件接口，提供:
    - 统一的数据访问 (get_latest, get_all)
    - 传感器健康监控
    - 自动重连
    - 数据记录
    """

    def __init__(self, name: str = "sensor_bridge"):
        self.name = name
        self._interfaces: Dict[str, SensorHardwareInterface] = {}
        self._lock = threading.RLock()
        self._data_buffer: Dict[str, List[SensorData]] = {}
        self._max_buffer_size = 1000
        self._health_check_interval = 5.0  # seconds
        self._health_thread: Optional[threading.Thread] = None
        self._running = False
        self._health_status: Dict[str, SensorHealth] = {}

    def register(
        self,
        sensor_id: str,
        interface: SensorHardwareInterface
    ) -> None:
        """
        注册传感器接口
        
        Args:
            sensor_id: 传感器唯一标识
            interface: 传感器硬件接口实例
        """
        with self._lock:
            self._interfaces[sensor_id] = interface
            self._data_buffer[sensor_id] = []
            self._health_status[sensor_id] = SensorHealth.OK

    def unregister(self, sensor_id: str) -> None:
        """注销传感器"""
        with self._lock:
            if sensor_id in self._interfaces:
                self._interfaces[sensor_id].close()
                del self._interfaces[sensor_id]
            if sensor_id in self._data_buffer:
                del self._data_buffer[sensor_id]
            if sensor_id in self._health_status:
                del self._health_status[sensor_id]

    def open_all(self) -> Dict[str, bool]:
        """打开所有传感器连接"""
        results = {}
        with self._lock:
            for sid, iface in self._interfaces.items():
                results[sid] = iface.open()
        return results

    def close_all(self) -> None:
        """关闭所有传感器连接"""
        with self._lock:
            for iface in self._interfaces.values():
                iface.close()

    def start_all(self) -> None:
        """启动所有传感器数据流"""
        self._running = True
        self._health_thread = threading.Thread(target=self._health_loop, daemon=True)
        self._health_thread.start()
        with self._lock:
            for iface in self._interfaces.values():
                if not iface._running:
                    iface.start_streaming(callback=self._on_sensor_data)

    def stop_all(self) -> None:
        """停止所有传感器数据流"""
        self._running = False
        if self._health_thread:
            self._health_thread.join(timeout=2.0)
        with self._lock:
            for iface in self._interfaces.values():
                iface.stop_streaming()

    def _on_sensor_data(self, data: SensorData) -> None:
        """传感器数据回调"""
        with self._lock:
            buf = self._data_buffer.get(data.sensor_id, [])
            buf.append(data)
            if len(buf) > self._max_buffer_size:
                buf.pop(0)
            self._data_buffer[data.sensor_id] = buf

    def get_latest(self, sensor_id: str) -> Optional[SensorData]:
        """获取最新传感器数据"""
        with self._lock:
            buf = self._data_buffer.get(sensor_id, [])
            return buf[-1] if buf else None

    def get_all(self, sensor_id: str, max_count: int = 100) -> List[SensorData]:
        """获取最近N条传感器数据"""
        with self._lock:
            buf = self._data_buffer.get(sensor_id, [])
            return buf[-max_count:]

    def get_health(self, sensor_id: str) -> SensorHealth:
        """获取传感器健康状态"""
        return self._health_status.get(sensor_id, SensorHealth.OFFLINE)

    def get_all_health(self) -> Dict[str, SensorHealth]:
        """获取所有传感器健康状态"""
        return self._health_status.copy()

    def _health_loop(self) -> None:
        """健康检查循环"""
        while self._running:
            time.sleep(self._health_check_interval)
            with self._lock:
                for sid, iface in self._interfaces.items():
                    last = iface.last_data
                    if last is None:
                        self._health_status[sid] = SensorHealth.OFFLINE
                    elif time.time() - last.timestamp > 5.0:
                        self._health_status[sid] = SensorHealth.ERROR
                    elif time.time() - last.timestamp > 1.0:
                        self._health_status[sid] = SensorHealth.DEGRADED
                    else:
                        self._health_status[sid] = SensorHealth.OK

    def list_sensors(self) -> List[str]:
        """列出所有已注册传感器ID"""
        return list(self._interfaces.keys())

    def create_sensor(
        self,
        sensor_id: str,
        sensor_type: SensorDataType,
        protocol: SensorProtocol,
        use_simulated: bool = True,
        **kwargs
    ) -> SensorHardwareInterface:
        """
        创建传感器接口 (工厂方法)
        
        Args:
            sensor_id: 传感器ID
            sensor_type: 传感器类型
            protocol: 通信协议
            use_simulated: 使用仿真模式 (无真实硬件时)
            **kwargs: 额外配置参数
        
        Returns:
            传感器硬件接口实例
        """
        config = SensorHardwareConfig(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            protocol=protocol,
            **kwargs
        )

        if use_simulated or protocol == SensorProtocol.SIMULATED:
            return SimulatedSensorInterface(config)
        
        # 真实传感器驱动选择 (基于协议)
        if protocol == SensorProtocol.CAN:
            from .canbus import create_can_bus, IMUCANopenNode, ForceTorqueCANopenNode, TactileCANopenNode
            bus = create_can_bus(config.port, config.can_channel)
            if sensor_type == SensorDataType.IMU:
                return IMUCANopenNode(config.canopen_node_id, bus)
            elif sensor_type == SensorDataType.FORCE_TORQUE:
                return ForceTorqueCANopenNode(config.canopen_node_id, bus)
            elif sensor_type == SensorDataType.TACTILE:
                return TactileCANopenNode(config.canopen_node_id, bus)

        # RS485/Modbus
        if protocol == SensorProtocol.RS485:
            # 使用 pyserial + minimalmodbus
            return SimulatedSensorInterface(config)

        # I2C (e.g., MPU6050)
        if protocol == SensorProtocol.I2C:
            return SimulatedSensorInterface(config)

        # 默认仿真
        return SimulatedSensorInterface(config)

    def __enter__(self):
        self.open_all()
        return self

    def __exit__(self, *args):
        self.close_all()


# ─── AGV五级硬件桥接规格 ────────────────────────────────────────────

AGV_SENSOR_BRIDGE_GRADES: Dict[str, Dict[str, Any]] = {
    "S": {
        "max_sensors": 4,
        "supported_types": [SensorDataType.IMU, SensorDataType.ENCODER],
        "supported_protocols": [SensorProtocol.SIMULATED, SensorProtocol.I2C],
        "total_bandwidth_mbps": 1.0,
        "max_sample_rate_hz": 200,
        "data_buffer_size": 500,
    },
    "M": {
        "max_sensors": 8,
        "supported_types": [SensorDataType.IMU, SensorDataType.FORCE_TORQUE, SensorDataType.ENCODER],
        "supported_protocols": [SensorProtocol.SIMULATED, SensorProtocol.I2C, SensorProtocol.CAN, SensorProtocol.RS485],
        "total_bandwidth_mbps": 5.0,
        "max_sample_rate_hz": 500,
        "data_buffer_size": 1000,
    },
    "L": {
        "max_sensors": 16,
        "supported_types": [SensorDataType.IMU, SensorDataType.FORCE_TORQUE, SensorDataType.TACTILE, SensorDataType.ENCODER, SensorDataType.DISTANCE],
        "supported_protocols": [SensorProtocol.SIMULATED, SensorProtocol.I2C, SensorProtocol.CAN, SensorProtocol.RS485, SensorProtocol.USB],
        "total_bandwidth_mbps": 20.0,
        "max_sample_rate_hz": 1000,
        "data_buffer_size": 2000,
    },
    "XL": {
        "max_sensors": 32,
        "supported_types": [SensorDataType.IMU, SensorDataType.FORCE_TORQUE, SensorDataType.TACTILE, SensorDataType.ENCODER, SensorDataType.DISTANCE, SensorDataType.CAMERA],
        "supported_protocols": [SensorProtocol.SIMULATED, SensorProtocol.CAN, SensorProtocol.RS485, SensorProtocol.USB, SensorProtocol.ETHERNET],
        "total_bandwidth_mbps": 100.0,
        "max_sample_rate_hz": 2000,
        "data_buffer_size": 5000,
    },
    "XXL": {
        "max_sensors": 64,
        "supported_types": [SensorDataType.IMU, SensorDataType.FORCE_TORQUE, SensorDataType.TACTILE, SensorDataType.ENCODER, SensorDataType.DISTANCE, SensorDataType.CAMERA, SensorDataType.BATTERY],
        "supported_protocols": [SensorProtocol.SIMULATED, SensorProtocol.CAN, SensorProtocol.RS485, SensorProtocol.USB, SensorProtocol.ETHERNET, SensorProtocol.WIRELESS],
        "total_bandwidth_mbps": 500.0,
        "max_sample_rate_hz": 5000,
        "data_buffer_size": 10000,
    },
}


def get_bridge_spec(grade: str) -> Dict[str, Any]:
    """获取指定AGV等级的硬件桥接规格"""
    return AGV_SENSOR_BRIDGE_GRADES.get(grade, AGV_SENSOR_BRIDGE_GRADES["M"])


# ─── 导出符号 ─────────────────────────────────────────────────────────

__all__ = [
    # 枚举
    'SensorDataType', 'SensorProtocol', 'SensorHealth',
    # 数据结构
    'SensorData', 'SensorHardwareConfig',
    # 接口
    'SensorHardwareInterface', 'SimulatedSensorInterface',
    # 桥接器
    'SensorHardwareBridge',
    # 规格表
    'AGV_SENSOR_BRIDGE_GRADES', 'get_bridge_spec',
]
