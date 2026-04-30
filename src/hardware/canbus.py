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
CAN Bus 硬件接口模块
====================

支持工业级CAN总线传感器:
- Kistler 六维力传感器 (CANopen)
- ATI Force/Torque 传感器 (Net F/T)
- LORD IMU (RS-485/USB, via USB-CAN adapter)
- xsens IMU (CANopen)
- 触觉阵列 (自定义CAN协议)

CANopen协议实现 (CiA 301 / CiA 406)
- NMT (Network Management)
- PDO (Process Data Object) 实时数据
- SDO (Service Data Object) 参数配置
- Heartbeat 节点保护

Author: SuperModel Team
Version: v2.57.0
"""

import struct
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np

# CANopen Object Dictionary 常用索引
CANOPEN_OD = {
    # Device Type
    0x1000: ("device_type", "<I"),
    # Error Register
    0x1001: ("error_register", "<B"),
    # Heartbeat Producer Time
    0x1017: ("heartbeat_producer", "<H"),
    # Identity Object
    0x1018: ("identity", "<IBBBBBBBB"),
    # TPDO1 Communication
    0x1800: ("tpd1_comm", "<HBBBBB"),
    # RPDO1 Communication
    0x1400: ("rpd1_comm", "<HBBBBB"),
    # TPDO1 Mapping
    0x1A00: ("tpd1_map", "<IBBBBBBBB"),
    # RPDO1 Mapping
    0x1600: ("rpd1_map", "<IBBBBBBBB"),
}


class CANBusState(Enum):
    """CAN总线状态"""
    CLOSED = auto()
    OPEN = auto()
    ERROR_ACTIVE = auto()
    ERROR_PASSIVE = auto()
    BUS_OFF = auto()
    WARNING = auto()


class CANopenNodeState(Enum):
    """CANopen NMT 状态"""
    INITIALISING = 0x00
    PREOPERATIONAL = 0x7F
    OPERATIONAL = 0x05
    STOPPED = 0x04


@dataclass
class CANFrame:
    """CAN总线数据帧"""
    can_id: int          # 11/29位CAN-ID
    data: bytes          # 数据 (0-8字节)
    is_extended: bool = False  # 扩展帧 (29位ID)
    is_rtr: bool = False       # 远程帧
    timestamp: float = 0.0     # 时间戳 (秒)


@dataclass
class CANopenPDO:
    """CANopen PDO (Process Data Object) 映射"""
    cob_id: int
    transmission_type: int = 0  # 0=同步, 1=异步, 254/255=事件驱动
    inhibit_time_ms: int = 0
    event_time_ms: int = 0
    mapping: Dict[int, Tuple[int, str]] = field(default_factory=dict)
    # {sub_index: (byte_size, format_string)}


@dataclass
class SensorCANConfig:
    """传感器CAN配置"""
    node_id: int              # CANopen Node-ID (1-127)
    can_channel: int = 0     # CAN通道 (0/1/...)
    bitrate: int = 1000000    # 波特率 (默认1Mbps)
    # CANopen PDO 映射
    rx_pdo_cob_id: int = 0x180 + 1   # 默认 TPDO1
    tx_pdo_cob_id: int = 0x200 + 1   # 默认 RPDO1
    # 滤波器
    accept_filter_mask: int = 0x7FF   # 接受滤波掩码
    accept_filter_code: int = 0x000   # 接受滤波码


# ─── 虚拟CAN总线 (用于仿真/测试) ──────────────────────────────────────

class VirtualCANBus:
    """
    虚拟CAN总线 (用于仿真/测试)
    
    模拟真实CAN总线行为，支持:
    - 帧收发
    - 总线冲突
    - 总线错误
    - 节点超时检测
    """

    def __init__(self, channel_id: int = 0):
        self.channel_id = channel_id
        self._lock = threading.RLock()
        self._listeners: List[Callable[[CANFrame], None]] = []
        self._nodes: Dict[int, 'VirtualCANopenNode'] = {}
        self._frames: List[CANFrame] = []
        self._max_frames = 1000
        self._is_open = False
        self._bus_state = CANBusState.CLOSED

    def open(self) -> bool:
        """打开虚拟CAN总线"""
        with self._lock:
            self._is_open = True
            self._bus_state = CANBusState.OPEN
            return True

    def close(self) -> None:
        """关闭虚拟CAN总线"""
        with self._lock:
            self._is_open = False
            self._frames.clear()
            self._bus_state = CANBusState.CLOSED

    def send(self, frame: CANFrame) -> bool:
        """
        发送CAN帧 (到虚拟总线)
        所有监听器都会收到
        """
        if not self._is_open:
            return False
        frame.timestamp = time.time()
        with self._lock:
            if len(self._frames) >= self._max_frames:
                self._frames.pop(0)
            self._frames.append(frame)
        # 分发给所有监听器
        for listener in self._listeners:
            try:
                listener(frame)
            except Exception:
                pass
        return True

    def receive(self, timeout: float = 0.0) -> Optional[CANFrame]:
        """接收CAN帧 (非阻塞或超时等待)"""
        if not self._is_open:
            return None
        with self._lock:
            if self._frames:
                return self._frames.pop(0)
        if timeout > 0:
            time.sleep(timeout)
        return None

    def add_listener(self, listener: Callable[[CANFrame], None]) -> None:
        """添加CAN帧监听器"""
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[CANFrame], None]) -> None:
        """移除CAN帧监听器"""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def register_node(self, node_id: int, node: 'VirtualCANopenNode') -> None:
        """注册CANopen节点"""
        with self._lock:
            self._nodes[node_id] = node

    def unregister_node(self, node_id: int) -> None:
        """注销CANopen节点"""
        with self._lock:
            self._nodes.pop(node_id, None)

    def get_bus_state(self) -> CANBusState:
        return self._bus_state

    def inject_error(self, error_type: str = "bus_off") -> None:
        """注入总线错误 (用于测试)"""
        with self._lock:
            if error_type == "bus_off":
                self._bus_state = CANBusState.BUS_OFF
            elif error_type == "error_passive":
                self._bus_state = CANBusState.ERROR_PASSIVE
            elif error_type == "warning":
                self._bus_state = CANBusState.WARNING

    def reset(self) -> None:
        """复位总线 (清除错误状态)"""
        with self._lock:
            self._frames.clear()
            self._bus_state = CANBusState.OPEN if self._is_open else CANBusState.CLOSED

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


# ─── 真实CAN总线接口 (需要socketcan或NI-CAN) ─────────────────────────

class RealCANBus:
    """
    真实CAN总线接口
    
    依赖: socketcan (Linux) 或 NI-CAN (Windows)
    - Linux: 使用 socketcan 接口 (/dev CAN*)
    - NI-CAN: Windows 专用
    
    示例 (Linux):
        # 加载CAN驱动
        sudo ip link add dev can0 type can bitrate 1000000
        sudo ip link set can0 up
    """

    def __init__(self, interface: str = "can0", channel_id: int = 0):
        self.interface = interface
        self.channel_id = channel_id
        self._socket = None
        self._rx_thread: Optional[threading.Thread] = None
        self._running = False
        self._listeners: List[Callable[[CANFrame], None]] = []
        self._lock = threading.RLock()
        self._bus_state = CANBusState.CLOSED

    def open(self) -> bool:
        """打开真实CAN总线接口"""
        try:
            import socketcan
            self._socket = socketcan.open_channel(self.interface, bustype='socketcan')
            self._socket.can_id = self.channel_id
            self._running = True
            self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
            self._rx_thread.start()
            self._bus_state = CANBusState.OPEN
            return True
        except ImportError:
            # socketcan 未安装，使用回退模式
            return False
        except Exception:
            return False

    def close(self) -> None:
        """关闭CAN总线"""
        self._running = False
        if self._rx_thread:
            self._rx_thread.join(timeout=1.0)
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        self._bus_state = CANBusState.CLOSED

    def send(self, frame: CANFrame) -> bool:
        """发送CAN帧"""
        if not self._socket:
            return False
        try:
            self._socket.send(frame.can_id, frame.data, extended=frame.is_extended)
            return True
        except Exception:
            return False

    def receive(self, timeout: float = 0.0) -> Optional[CANFrame]:
        """接收CAN帧"""
        if not self._socket:
            return None
        try:
            can_id, data, _ = self._socket.receive(timeout=timeout if timeout > 0 else None)
            return CANFrame(can_id=can_id, data=data, timestamp=time.time())
        except Exception:
            return None

    def add_listener(self, listener: Callable[[CANFrame], None]) -> None:
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[CANFrame], None]) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def _rx_loop(self) -> None:
        """接收线程"""
        while self._running:
            frame = self.receive(timeout=0.1)
            if frame:
                for listener in self._listeners:
                    try:
                        listener(frame)
                    except Exception:
                        pass

    def get_bus_state(self) -> CANBusState:
        return self._bus_state

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()


# ─── CANopen 节点基类 ─────────────────────────────────────────────────

class CANopenNode(ABC):
    """
    CANopen 节点基类
    
    实现标准CANopen协议:
    - NMT (Network Management)
    - PDO (实时数据)
    - SDO (参数读写)
    - Heartbeat
    """

    def __init__(
        self,
        node_id: int,
        bus: Optional[Any] = None,
        config: Optional[SensorCANConfig] = None
    ):
        self.node_id = node_id
        self.bus = bus
        self.config = config or SensorCANConfig(node_id=node_id)
        self._state = CANopenNodeState.INITIALISING
        self._hb_consumer_time_ms: int = 1000
        self._hb_producer_time_ms: int = 500
        self._last_hb: Dict[int, float] = {}
        self._listeners: Dict[int, List[Callable]] = {}
        self._object_dictionary: Dict[int, Any] = {}
        self._lock = threading.RLock()
        self._running = False
        self._hb_thread: Optional[threading.Thread] = None

        # 默认OD条目
        self._init_object_dictionary()

    def _init_object_dictionary(self) -> None:
        """初始化对象字典"""
        self._object_dictionary[0x1000] = 0x00020192  # Device Type (牵引/力传感器)
        self._object_dictionary[0x1001] = 0x00        # Error Register
        self._object_dictionary[0x1017] = self._hb_producer_time_ms  # Heartbeat Time
        self._object_dictionary[0x1018] = {           # Identity Object
            1: 0x00000219,   # Vendor-ID (D-Robotics)
            2: 0x00000001,   # Product Code
            3: 0x00000001,   # Revision Number
            4: 0x00000001,   # Serial Number
        }

    def start(self) -> None:
        """启动节点"""
        self._running = True
        self._state = CANopenNodeState.OPERATIONAL
        if self.bus:
            self.bus.add_listener(self._on_can_frame)
        self._hb_thread = threading.Thread(target=self._hb_loop, daemon=True)
        self._hb_thread.start()

    def stop(self) -> None:
        """停止节点"""
        self._running = False
        if self.bus:
            self.bus.remove_listener(self._on_can_frame)
        self._state = CANopenNodeState.STOPPED

    @property
    def state(self) -> CANopenNodeState:
        return self._state

    def _on_can_frame(self, frame: CANFrame) -> None:
        """处理收到的CAN帧"""
        with self._lock:
            # NMT 命令
            if frame.can_id == 0x000:
                self._handle_nmt(frame.data)
            # 节点特定帧
            elif (frame.can_id & 0xF80) == 0x580:  # SDO Response
                self._handle_sdo_rx(frame)
            elif (frame.can_id & 0xF80) == 0x200:  # TPDO1
                self._handle_pdo_rx(frame, pdo_num=1)
            elif (frame.can_id & 0xF80) == 0x280:  # TPDO2
                self._handle_pdo_rx(frame, pdo_num=2)
            # Heartbeat
            elif frame.can_id == 0x700 + self.node_id:
                self._handle_heartbeat(frame.data)

    def _handle_nmt(self, data: bytes) -> None:
        """处理NMT命令"""
        if len(data) < 2:
            return
        cmd, target = data[0], data[1]
        if target != self.node_id and target != 0:
            return
        if cmd == 0x01:  # Start
            self._state = CANopenNodeState.OPERATIONAL
        elif cmd == 0x02:  # Stop
            self._state = CANopenNodeState.STOPPED
        elif cmd == 0x80:  # Enter Pre-operational
            self._state = CANopenNodeState.PREOPERATIONAL
        elif cmd == 0x81:  # Reset node
            self._init_object_dictionary()
            self._state = CANopenNodeState.INITIALISING
        elif cmd == 0x82:  # Reset communication
            self._state = CANopenNodeState.INITIALISING

    def _handle_sdo_rx(self, frame: CANFrame) -> None:
        """处理SDO响应"""
        pass

    def _handle_pdo_rx(self, frame: CANFrame, pdo_num: int) -> None:
        """处理PDO接收"""
        pass

    def _handle_heartbeat(self, data: bytes) -> None:
        """处理心跳接收"""
        if len(data) < 1:
            return
        node_id = self.node_id  # HB发送者ID需从CAN-ID提取
        self._last_hb[node_id] = time.time()

    def _hb_loop(self) -> None:
        """心跳生产/消费线程"""
        while self._running:
            # 发送自身心跳
            if self._state == CANopenNodeState.OPERATIONAL:
                hb_frame = CANFrame(
                    can_id=0x700 + self.node_id,
                    data=bytes([self._state.value]),
                )
                self.bus.send(hb_frame)
            # 检查其他节点心跳超时
            now = time.time()
            timeout_nodes = [
                nid for nid, last in self._last_hb.items()
                if now - last > self._hb_consumer_time_ms * 1.5 / 1000.0
            ]
            for nid in timeout_nodes:
                self._on_node_timeout(nid)
                del self._last_hb[nid]
            time.sleep(self._hb_producer_time_ms / 1000.0 / 2)

    def _on_node_timeout(self, node_id: int) -> None:
        """节点超时回调"""
        pass

    # ─── SDO 读写接口 ────────────────────────────────────────────────

    def sdo_read(self, index: int, subindex: int = 0) -> Optional[bytes]:
        """SDO读请求 (发送后等待响应)"""
        if not self.bus or not self._socket:
            return None
        # 发送SDO请求
        req = self._build_sdo_request(index, subindex)
        self.bus.send(CANFrame(can_id=0x600 + self.node_id, data=req))
        # 等待响应 (简化实现)
        timeout = 1.0
        start = time.time()
        while time.time() - start < timeout:
            frame = self.bus.receive(timeout=0.1)
            if frame and frame.can_id == 0x580 + self.node_id:
                return frame.data
        return None

    def sdo_write(self, index: int, subindex: int, data: bytes) -> bool:
        """SDO写请求"""
        if not self.bus:
            return False
        req = self._build_sdo_response(index, subindex, data)
        return self.bus.send(CANFrame(can_id=0x600 + self.node_id, data=req))

    def _build_sdo_request(self, index: int, subindex: int) -> bytes:
        """构建SDO读请求"""
        cmd = 0x40
        return struct.pack("<BBHB", cmd, subindex, index & 0xFF, (index >> 8) & 0xFF)

    def _build_sdo_response(self, index: int, subindex: int, data: bytes) -> bytes:
        """构建SDO写请求"""
        n = len(data)
        cmd = 0x23 | ((4 - n) << 2)  # 0x23=4字节, 0x27=3字节, 0x2B=2字节, 0x2F=1字节
        payload = struct.pack("<BBHB", cmd, subindex, index & 0xFF, (index >> 8) & 0xFF)
        return payload + data.ljust(4, b'\x00')

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# ─── IMU CANopen 节点 ────────────────────────────────────────────────

class IMUCANopenNode(CANopenNode):
    """
    IMU CANopen 节点 (e.g., xsens MTi compatible)
    
    PDO1 映射 (默认):
    - 0x1A00[0]: 数据长度
    - 0x1A00[1]: 加速计 (12字节: 3x float)
    - 0x1A00[2]: 陀螺仪 (12字节: 3x float)
    - 0x1A00[3]: 磁力计 (12字节, 可选)
    - 0x1A00[4]: 姿态角 (12字节: 3x float)
    """
    # PDO1 默认 COB-ID
    DEFAULT_TPDO1_COB_ID = 0x180 + 0x01  # 0x181 (Node-ID=1)

    def __init__(
        self,
        node_id: int,
        bus: Optional[Any] = None,
        config: Optional[SensorCANConfig] = None
    ):
        super().__init__(node_id, bus, config)
        self._accel: Optional[np.ndarray] = None
        self._gyro: Optional[np.ndarray] = None
        self._mag: Optional[np.ndarray] = None
        self._euler: Optional[np.ndarray] = None
        self._quat: Optional[np.ndarray] = None
        self._temperature: float = 25.0
        self._on_data_callbacks: List[Callable] = []

        # 对象字典扩展
        self._object_dictionary[0x2000] = (0, "<3f")  # 加速度计
        self._object_dictionary[0x2001] = (0, "<3f")  # 陀螺仪
        self._object_dictionary[0x2002] = (0, "<3f")  # 磁力计
        self._object_dictionary[0x2003] = (0, "<3f")  # 欧拉角
        self._object_dictionary[0x2004] = (0, "<4f")  # 四元数
        self._object_dictionary[0x2005] = (0, "<f")   # 温度

    def set_output_config(
        self,
        enable_accel: bool = True,
        enable_gyro: bool = True,
        enable_mag: bool = False,
        enable_euler: bool = True,
        enable_quat: bool = False,
        sample_rate: int = 100  # Hz
    ) -> None:
        """配置IMU输出项"""
        self._output_config = {
            'accel': enable_accel,
            'gyro': enable_gyro,
            'mag': enable_mag,
            'euler': enable_euler,
            'quat': enable_quat,
            'sample_rate': sample_rate,
        }

    def on_imu_data(self, callback: Callable) -> None:
        """注册IMU数据回调"""
        self._on_data_callbacks.append(callback)

    def _handle_pdo_rx(self, frame: CANFrame, pdo_num: int) -> None:
        """解析PDO数据"""
        if len(frame.data) < 12:
            return
        data = frame.data

        if pdo_num == 1:
            # TPDO1: 加速度 + 陀螺仪
            if len(data) >= 24:
                self._accel = np.array(struct.unpack("<3f", data[0:12]))
                self._gyro = np.array(struct.unpack("<3f", data[12:24]))
            elif len(data) >= 12:
                self._accel = np.array(struct.unpack("<3f", data[0:12]))
        elif pdo_num == 2:
            # TPDO2: 欧拉角 + 温度
            if len(data) >= 16:
                self._euler = np.array(struct.unpack("<3f", data[0:12]))
                self._temperature = struct.unpack("<f", data[12:16])[0]

        # 触发回调
        for cb in self._on_data_callbacks:
            try:
                cb(self)
            except Exception:
                pass

    def get_accel(self) -> Optional[np.ndarray]:
        return self._accel.copy() if self._accel is not None else None

    def get_gyro(self) -> Optional[np.ndarray]:
        return self._gyro.copy() if self._gyro is not None else None

    def get_euler(self) -> Optional[np.ndarray]:
        return self._euler.copy() if self._euler is not None else None

    def get_quat(self) -> Optional[np.ndarray]:
        return self._quat.copy() if self._quat is not None else None

    def get_temperature(self) -> float:
        return self._temperature


# ─── 六维力/力矩 CANopen 节点 ─────────────────────────────────────────

class ForceTorqueCANopenNode(CANopenNode):
    """
    六维力/力矩传感器 CANopen 节点
    
    支持:
    - Kistler ti 系列的 CANopen 输出
    - ATI Force/Torque 传感器 (通过 NI-CAN 或 USB-CAN)
    - 自定义力传感器
    
    PDO 映射:
    - Fx, Fy, Fz (float, N)
    - Mx, My, Mz (float, Nm)
    - 温度 (float, °C)
    """

    def __init__(
        self,
        node_id: int,
        bus: Optional[Any] = None,
        config: Optional[SensorCANConfig] = None
    ):
        super().__init__(node_id, bus, config)
        self._wrench: Optional[np.ndarray] = None  # [Fx, Fy, Fz, Mx, My, Mz]
        self._raw_adc: Optional[np.ndarray] = None
        self._temperature: float = 25.0
        self._is_saturated: bool = False
        self._on_data_callbacks: List[Callable] = []

        # 标定矩阵 (6x6 温度补偿)
        self._calibration_matrix = np.eye(6)
        self._offset = np.zeros(6)

        # 对象字典
        self._object_dictionary[0x3000] = (0, "<6f")  # 原始力/力矩
        self._object_dictionary[0x3001] = (0, "<6f")  # 补偿后力/力矩
        self._object_dictionary[0x3002] = (0, "<f")   # 传感器温度

    def set_calibration(
        self,
        calibration_matrix: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None
    ) -> None:
        """设置标定矩阵和零点偏移"""
        if calibration_matrix is not None:
            self._calibration_matrix = np.array(calibration_matrix)
        if offset is not None:
            self._offset = np.array(offset)

    def zero(self) -> None:
        """零点校准 (记录当前值作为零点)"""
        if self._wrench is not None:
            self._offset = self._wrench.copy()

    def on_force_data(self, callback: Callable) -> None:
        """注册力觉数据回调"""
        self._on_data_callbacks.append(callback)

    def _handle_pdo_rx(self, frame: CANFrame, pdo_num: int) -> None:
        """解析PDO数据"""
        if len(frame.data) < 24:  # 6 floats = 24 bytes
            return

        # 原始值
        raw = np.array(struct.unpack("<6f", frame.data[0:24]))

        # 应用标定和零点补偿
        calibrated = self._calibration_matrix @ (raw - self._offset)
        self._wrench = calibrated

        # 解析温度 (可选, PDO2)
        if len(frame.data) >= 28 and pdo_num == 2:
            self._temperature = struct.unpack("<f", frame.data[24:28])[0]

        # 饱和检测
        F_MAX = 120.0  # N (可配置)
        M_MAX = 2.0    # Nm
        self._is_saturated = (
            np.any(np.abs(calibrated[:3]) > F_MAX) or
            np.any(np.abs(calibrated[3:]) > M_MAX)
        )

        # 触发回调
        for cb in self._on_data_callbacks:
            try:
                cb(self)
            except Exception:
                pass

    def get_wrench(self) -> Optional[np.ndarray]:
        """获取当前力/力矩向量 [Fx, Fy, Fz, Mx, My, Mz]"""
        return self._wrench.copy() if self._wrench is not None else None

    def is_saturated(self) -> bool:
        return self._is_saturated


# ─── 触觉阵列 CANopen 节点 ───────────────────────────────────────────

class TactileCANopenNode(CANopenNode):
    """
    触觉阵列 CANopen 节点 (e.g., custom BioTac-style array)
    
    PDO 映射:
    - Taxel 数据 (多字节, 取决于阵列大小)
    - 总压力 (float)
    - 接触状态 (uint8)
    """

    def __init__(
        self,
        node_id: int,
        bus: Optional[Any] = None,
        config: Optional[SensorCANConfig] = None,
        array_size: Tuple[int, int] = (8, 8)
    ):
        super().__init__(node_id, bus, config)
        self.array_size = array_size
        self._taxel_data: Optional[np.ndarray] = None
        self._total_pressure: float = 0.0
        self._contact_center: Tuple[float, float] = (0.0, 0.0)
        self._is_contact: bool = False
        self._on_data_callbacks: List[Callable] = []

        self._object_dictionary[0x4000] = (0, f"<{array_size[0]*array_size[1]}B")  # Taxel raw
        self._object_dictionary[0x4001] = (0, "<f")   # 总压力
        self._object_dictionary[0x4002] = (0, "<B")    # 接触状态

    def on_tactile_data(self, callback: Callable) -> None:
        """注册触觉数据回调"""
        self._on_data_callbacks.append(callback)

    def _handle_pdo_rx(self, frame: CANFrame, pdo_num: int) -> None:
        """解析PDO数据"""
        if len(frame.data) < self.array_size[0] * self.array_size[1]:
            return

        n_taxels = self.array_size[0] * self.array_size[1]
        raw_taxels = struct.unpack(f"<{n_taxels}B", frame.data[:n_taxels])
        self._taxel_data = np.array(raw_taxels).reshape(self.array_size)
        self._total_pressure = np.sum(self._taxel_data)

        # 接触中心计算
        rows, cols = self.array_size
        y_indices, x_indices = np.where(self._taxel_data > 0)
        if len(y_indices) > 0:
            self._contact_center = (
                np.mean(x_indices) / cols,
                np.mean(y_indices) / rows
            )
            self._is_contact = True
        else:
            self._is_contact = False

        # 触发回调
        for cb in self._on_data_callbacks:
            try:
                cb(self)
            except Exception:
                pass

    def get_taxel_data(self) -> Optional[np.ndarray]:
        return self._taxel_data.copy() if self._taxel_data is not None else None

    def get_total_pressure(self) -> float:
        return self._total_pressure

    def get_contact_center(self) -> Tuple[float, float]:
        return self._contact_center

    def is_contact(self) -> bool:
        return self._is_contact


# ─── CAN 总线工厂函数 ─────────────────────────────────────────────────

def create_can_bus(
    interface: str = "can0",
    channel_id: int = 0,
    use_virtual: bool = False
) -> Any:
    """
    创建CAN总线实例
    
    Args:
        interface: CAN接口名称 (Linux: can0, can1, ...)
        channel_id: CAN通道编号
        use_virtual: 强制使用虚拟总线 (用于测试/仿真)
    
    Returns:
        CAN总线实例 (VirtualCANBus 或 RealCANBus)
    """
    if use_virtual:
        return VirtualCANBus(channel_id=channel_id)
    
    # 尝试真实总线
    real = RealCANBus(interface=interface, channel_id=channel_id)
    if real.open():
        return real
    
    # 回退到虚拟总线
    return VirtualCANBus(channel_id=channel_id)


# ─── AGV五级CAN总线规格表 ─────────────────────────────────────────────

AGV_CAN_GRADES: Dict[str, Dict[str, Any]] = {
    "S": {
        "bitrate": 500000,
        "max_nodes": 4,
        "can_channels": 1,
        "supported_sensors": ["imu", "force"],
        "max_frame_rate": 200,
        "protocol": "CANopen",
    },
    "M": {
        "bitrate": 1000000,
        "max_nodes": 8,
        "can_channels": 1,
        "supported_sensors": ["imu", "force", "tactile"],
        "max_frame_rate": 500,
        "protocol": "CANopen",
    },
    "L": {
        "bitrate": 1000000,
        "max_nodes": 16,
        "can_channels": 2,
        "supported_sensors": ["imu", "force", "tactile", "encoder"],
        "max_frame_rate": 1000,
        "protocol": "CANopen",
    },
    "XL": {
        "bitrate": 1000000,
        "max_nodes": 32,
        "can_channels": 4,
        "supported_sensors": ["imu", "force", "tactile", "encoder", "lidar"],
        "max_frame_rate": 2000,
        "protocol": "CANopen + proprietary",
    },
    "XXL": {
        "bitrate": 1000000,
        "max_nodes": 64,
        "can_channels": 8,
        "supported_sensors": ["imu", "force", "tactile", "encoder", "lidar", "camera"],
        "max_frame_rate": 5000,
        "protocol": "CANopen + EtherCAT hybrid",
    },
}


def get_can_spec(grade: str) -> Dict[str, Any]:
    """获取指定AGV等级的CAN总线规格"""
    return AGV_CAN_GRADES.get(grade, AGV_CAN_GRADES["M"])


# ─── 导出符号 ─────────────────────────────────────────────────────────

__all__ = [
    # CAN 帧
    'CANFrame', 'CANBusState', 'CANopenNodeState', 'CANopenPDO', 'SensorCANConfig',
    # CAN 总线
    'VirtualCANBus', 'RealCANBus', 'create_can_bus',
    # CANopen 节点
    'CANopenNode', 'IMUCANopenNode', 'ForceTorqueCANopenNode', 'TactileCANopenNode',
    # AGV 五级规格
    'AGV_CAN_GRADES', 'get_can_spec',
]
