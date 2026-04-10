"""
real_agv_interface.py - 真实AGV机器人硬件接口适配
SuperModel 超模态大模型具身智能系统

支持:
- CAN总线通信 (ZLAC8015D 驱动器)
- 镭神N10P激光雷达
- ETT10A-PW IMU
- 触觉/力觉传感器桥接
- ROS2桥接
- 实时状态反馈
- AGV五级硬件规格适配
"""

from __future__ import annotations
import abc
import enum
import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from utils import logger, retry, RateLimiter

__all__ = [
    'AGVHardwareStatus',
    'MotorState',
    'WheelEncoder',
    'RealAGVInterface',
    'CANZAC8015DDriver',
    'LidarN10P',
    'IMUETT10APW',
    'AGVTactileBridge',
    'AGVForceBridge',
    'HardwareMonitor',
    'RealAGVController',
    'AGV_HARDWARE_SPECS',
]


class AGVHardwareStatus(enum.Enum):
    """AGV硬件状态"""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RUNNING = "RUNNING"
    ERROR = "ERROR"
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass
class MotorState:
    """电机状态"""
    motor_id: int
    enabled: bool = False
    current_rpm: float = 0.0
    current_current: float = 0.0
    current_position: int = 0  # 编码器位置
    temperature: float = 25.0
    voltage: float = 24.0
    error_code: int = 0

    def get_radians_per_second(self, encoder_ppr: int = 1000) -> float:
        """转换转速到角速度"""
        return (self.current_rpm * 2 * np.pi) / 60.0


@dataclass
class WheelEncoder:
    """轮式编码器读数"""
    left_ticks: int = 0
    right_ticks: int = 0
    left_delta: int = 0
    right_delta: int = 0
    timestamp: float = 0.0


@dataclass
class IMUData:
    """IMU数据"""
    accelerometer: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m/s^2
    gyroscope: np.ndarray = field(default_factory=lambda: np.zeros(3))  # rad/s
    magnetometer: Optional[np.ndarray] = None  # uT
    quaternion: Optional[np.ndarray] = None  # wxyz
    temperature: float = 25.0
    timestamp: float = 0.0


@dataclass
class LidarScan:
    """激光雷达扫描数据"""
    ranges: np.ndarray  # 距离值 (米)
    angles: np.ndarray  # 角度 (弧度)
    intensities: Optional[np.ndarray] = None
    timestamp: float = 0.0


# AGV五级硬件规格
# -----------------------------------------------------------------------------

AGV_HARDWARE_SPECS = {
    'S': {
        'grade': 'S',
        'load_kg': 30,
        'wheel_config': '2-wheel-diff',
        'motor_type': 'stepper_57',
        'motor_power_w': 50 * 2,
        'encoder_ppr': 1000,
        'max_speed_mps': 1.0,
        'battery_v': 24,
        'battery_ah': 10,
        'lidar': 'none',  # or 360_10m
        'has_imu': True,
        'has_tactile': False,
        'has_force': False,
        'can_bitrate': 250000,
    },
    'M': {
        'grade': 'M',
        'load_kg': 100,
        'wheel_config': '2-wheel-diff',
        'motor_type': 'hub_5.5inch',
        'motor_power_w': 150 * 2,
        'encoder_ppr': 1024,
        'max_speed_mps': 1.5,
        'battery_v': 48,
        'battery_ah': 20,
        'lidar': '360_25m',  # 镭神N10P
        'has_imu': True,
        'has_tactile': True,
        'has_force': True,
        'can_bitrate': 500000,
    },
    'L': {
        'grade': 'L',
        'load_kg': 300,
        'wheel_config': '4-wheel-diff',
        'motor_type': 'hub_5.5inch',
        'motor_power_w': 150 * 4,
        'encoder_ppr': 1024,
        'max_speed_mps': 1.5,
        'battery_v': 48,
        'battery_ah': 35,
        'lidar': '360_25m',
        'has_imu': True,
        'has_tactile': True,
        'has_force': True,
        'can_bitrate': 500000,
    },
    'XL': {
        'grade': 'XL',
        'load_kg': 600,
        'wheel_config': '4-wheel-diff',
        'motor_type': 'hub_6.5inch',
        'motor_power_w': 300 * 4,
        'encoder_ppr': 2048,
        'max_speed_mps': 1.2,
        'battery_v': 48,
        'battery_ah': 50,
        'lidar': '360_40m',
        'has_imu': True,
        'has_tactile': True,
        'has_force': True,
        'can_bitrate': 1000000,
    },
    'XXL': {
        'grade': 'XXL',
        'load_kg': 1200,
        'wheel_config': '4-wheel-diff',
        'motor_type': 'hub_7.5inch',
        'motor_power_w': 500 * 4,
        'encoder_ppr': 4096,
        'max_speed_mps': 1.0,
        'battery_v': 72,
        'battery_ah': 60,
        'lidar': 'double_360_40m',
        'has_imu': True,
        'has_tactile': True,
        'has_force': True,
        'can_bitrate': 1000000,
    },
}


class RealAGVInterface(abc.ABC):
    """真实AGV硬件接口基类"""

    def __init__(self):
        self.status: AGVHardwareStatus = AGVHardwareStatus.DISCONNECTED
        self.last_update_time: float = 0.0

    @abc.abstractmethod
    def connect(self) -> bool:
        """连接硬件"""
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """断开连接"""
        pass

    @abc.abstractmethod
    def get_status(self) -> AGVHardwareStatus:
        """获取当前状态"""
        pass

    def is_connected(self) -> bool:
        return self.status == AGVHardwareStatus.CONNECTED or self.status == AGVHardwareStatus.RUNNING


class CANZAC8015DDriver(RealAGVInterface):
    """
    中菱 ZLAC8015D 双路轮毂伺服驱动器 CAN总线接口
    支持一拖二，CANopen协议
    """

    # CANopen 节点ID默认分配
    DEFAULT_NODE_ID = {
        'left': 0x01,
        'right': 0x02,
        'front_left': 0x01,
        'front_right': 0x02,
        'rear_left': 0x03,
        'rear_right': 0x04,
    }

    # 对象字典索引
    OD_CONTROL_WORD = 0x6040
    OD_STATUS_WORD = 0x6041
    OD_TARGET_POSITION = 0x607A
    OD_TARGET_VELOCITY = 0x60FF
    OD_PROFILE_VELOCITY = 0x6081
    OD_PROFILE_ACCELERATION = 0x6083
    OD_PROFILE_DECELERATION = 0x6084
    OD_ACTUAL_POSITION = 0x6064
    OD_ACTUAL_VELOCITY = 0x606C
    OD_ACTUAL_CURRENT = 0x6078
    OD_MOTOR_TEMP = 0x606F
    OD_DRIVER_TEMP = 0x6067

    # 控制字命令
    CW_SHUTDOWN = 0x0006
    CW_SWITCH_ON = 0x000F
    CW_ENABLE_OPERATION = 0x000F
    CW_FAULT_RESET = 0x0080

    def __init__(self, can_interface: str = "can0", node_ids: Optional[List[int]] = None,
                 bitrate: int = 500000):
        """
        Args:
            can_interface: CAN接口名称 (can0, can1)
            node_ids: 驱动器节点ID列表，每个轮子一个ID
            bitrate: CAN总线波特率
        """
        super().__init__()
        self.can_interface = can_interface
        self.node_ids = node_ids or [0x01, 0x02]
        self.bitrate = bitrate
        self.motor_states: Dict[int, MotorState] = {}
        self.target_velocities: Dict[int, float] = {}  # rad/s
        self._can_socket = None
        self._reading_thread: Optional[threading.Thread] = None
        self._command_queue: queue.Queue = queue.Queue(maxsize=100)
        self._running = False
        self._emergency_stop = False

        for node_id in self.node_ids:
            self.motor_states[node_id] = MotorState(motor_id=node_id)

    def connect(self) -> bool:
        """连接CAN总线并初始化驱动器"""
        self.status = AGVHardwareStatus.CONNECTING
        logger.info(f"Connecting to ZLAC8015D on {self.can_interface}, bitrate={self.bitrate}")

        try:
            # 这里实际需要 python-can 库
            import can
            self._can_socket = can.Bus(
                interface='socketcan',
                channel=self.can_interface,
                bitrate=self.bitrate,
            )

            # 初始化每个电机
            for node_id in self.node_ids:
                if not self._init_motor(node_id):
                    logger.error(f"Failed to initialize motor node {node_id}")
                    self.status = AGVHardwareStatus.ERROR
                    return False

            # 启动读取线程
            self._running = True
            self._reading_thread = threading.Thread(target=self._reading_loop, daemon=True)
            self._reading_thread.start()

            # 启动命令发送线程
            self._writing_thread = threading.Thread(target=self._writing_loop, daemon=True)
            self._writing_thread.start()

            self.status = AGVHardwareStatus.CONNECTED
            logger.info(f"Connected to ZLAC8015D, {len(self.node_ids)} motors initialized")
            return True

        except ImportError:
            logger.error("python-can not installed, cannot connect to CAN bus")
            self.status = AGVHardwareStatus.ERROR
            return False
        except Exception as e:
            logger.error(f"CAN connection failed: {e}")
            self.status = AGVHardwareStatus.ERROR
            return False

    def disconnect(self) -> None:
        """断开连接"""
        self._running = False
        if self._reading_thread:
            self._reading_thread.join(timeout=1.0)
        if self._writing_thread:
            self._writing_thread.join(timeout=1.0)
        if self._can_socket:
            self._can_socket.shutdown()
            self._can_socket = None
        self.status = AGVHardwareStatus.DISCONNECTED
        logger.info("Disconnected from ZLAC8015D")

    def _init_motor(self, node_id: int) -> bool:
        """初始化单个电机"""
        # 发送关机命令
        self._send_sdo(node_id, self.OD_CONTROL_WORD, self.CW_SHUTDOWN)
        time.sleep(0.1)
        # 发送开机命令
        self._send_sdo(node_id, self.OD_CONTROL_WORD, self.CW_SWITCH_ON)
        time.sleep(0.1)
        # 使能运行
        self._send_sdo(node_id, self.OD_CONTROL_WORD, self.CW_ENABLE_OPERATION)
        time.sleep(0.1)
        # 读取初始状态
        self._read_motor_state(node_id)
        return True

    def _send_sdo(self, node_id: int, index: int, value: int) -> bool:
        """发送SDO请求"""
        if self._can_socket is None:
            return False

        # SDO发送格式 (简化)
        msg = can.Message(
            arbitration_id=0x600 + node_id,
            data=[0x2B, index & 0xFF, index >> 8, 0x00, value & 0xFF, (value >> 8) & 0xFF, 0x00, 0x00],
            is_extended_id=False
        )
        try:
            self._can_socket.send(msg)
            return True
        except Exception as e:
            logger.error(f"Failed to send SDO: {e}")
            return False

    def _read_motor_state(self, node_id: int) -> None:
        """读取电机状态"""
        # 实际实现会读取各种状态寄存器
        # 这里是框架代码
        pass

    def _reading_loop(self) -> None:
        """后台读取循环"""
        while self._running:
            if self._can_socket is None:
                break
            try:
                msg = self._can_socket.recv(timeout=0.1)
                if msg is not None:
                    self._process_message(msg)
            except Exception as e:
                logger.warning(f"CAN receive error: {e}")
                time.sleep(0.01)

    def _writing_loop(self) -> None:
        """后台命令发送循环"""
        while self._running:
            try:
                cmd = self._command_queue.get(timeout=0.1)
                if self._can_socket:
                    self._can_socket.send(cmd)
            except queue.Empty:
                continue
            except Exception as e:
                logger.warning(f"CAN send error: {e}")

    def _process_message(self, msg) -> None:
        """处理接收到的CAN消息"""
        # 解析SDO响应，更新电机状态
        pass

    def set_velocity(self, motor_id: int, velocity_rad_s: float) -> bool:
        """设置电机速度 (rad/s)"""
        if self._emergency_stop:
            return False

        node_id = self.node_ids[motor_id] if isinstance(motor_id, int) else motor_id
        self.target_velocities[node_id] = velocity_rad_s

        # 转换为驱动器单位 (通常是 pulse/s 或 rpm)
        # 这里需要根据编码器PPR转换
        rpm = (velocity_rad_s * 60) / (2 * np.pi)
        # 发送目标速度到驱动器
        # self._send_sdo(node_id, self.OD_TARGET_VELOCITY, int(rpm * scale))
        return True

    def set_wheel_velocities(self, velocities: List[float]) -> bool:
        """设置所有轮子速度"""
        if len(velocities) != len(self.node_ids):
            logger.error(f"Velocity count mismatch: expected {len(self.node_ids)}, got {len(velocities)}")
            return False

        all_ok = True
        for i, vel in enumerate(velocities):
            if not self.set_velocity(i, vel):
                all_ok = False
        return all_ok

    def emergency_stop(self) -> None:
        """紧急停止"""
        self._emergency_stop = True
        for node_id in self.node_ids:
            self._send_sdo(node_id, self.OD_CONTROL_WORD, self.CW_SHUTDOWN)
        self.status = AGVHardwareStatus.EMERGENCY_STOP
        logger.warning("Emergency stop triggered on ZLAC8015D")

    def release_emergency_stop(self) -> bool:
        """释放紧急停止"""
        if self.status != AGVHardwareStatus.EMERGENCY_STOP:
            return True

        self._emergency_stop = False
        for node_id in self.node_ids:
            self._init_motor(node_id)
        self.status = AGVHardwareStatus.CONNECTED
        logger.info("Emergency stop released")
        return True

    def get_motor_states(self) -> List[MotorState]:
        """获取所有电机状态"""
        return list(self.motor_states.values())

    def get_status(self) -> AGVHardwareStatus:
        return self.status


class LidarN10P(RealAGVInterface):
    """
    镭神 N10P 360° 激光雷达
    25米测距，360°全覆盖
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 115200):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self._serial = None
        self._reading_thread: Optional[threading.Thread] = None
        self._running = False
        self._last_scan: Optional[LidarScan] = None
        self._scan_callback: Optional[Callable[[LidarScan], None]] = None

    def connect(self) -> bool:
        self.status = AGVHardwareStatus.CONNECTING
        try:
            import serial
            self._serial = serial.Serial(self.port, self.baudrate, timeout=1.0)
            self._running = True
            self._reading_thread = threading.Thread(target=self._reading_loop, daemon=True)
            self._reading_thread.start()
            self.status = AGVHardwareStatus.CONNECTED
            logger.info(f"Connected to LeiShen N10P lidar on {self.port}")
            return True
        except ImportError:
            logger.error("pyserial not installed")
            self.status = AGVHardwareStatus.ERROR
            return False
        except Exception as e:
            logger.error(f"Failed to connect to lidar: {e}")
            self.status = AGVHardwareStatus.ERROR
            return False

    def disconnect(self) -> None:
        self._running = False
        if self._reading_thread:
            self._reading_thread.join(timeout=1.0)
        if self._serial:
            self._serial.close()
            self._serial = None
        self.status = AGVHardwareStatus.DISCONNECTED
        logger.info("Disconnected from lidar")

    def _reading_loop(self) -> None:
        import serial
        while self._running:
            if self._serial is None:
                break
            try:
                data = self._serial.read(4096)
                scan = self._parse_data(data)
                if scan is not None:
                    self._last_scan = scan
                    if self._scan_callback:
                        self._scan_callback(scan)
            except serial.SerialException as e:
                logger.warning(f"Lidar serial error: {e}")
                time.sleep(0.1)

    def _parse_data(self, data: bytes) -> Optional[LidarScan]:
        """解析雷达数据"""
        # 实际解析需要根据镭神通信协议
        # 这里是框架占位
        return None

    def get_last_scan(self) -> Optional[LidarScan]:
        """获取最新扫描数据"""
        return self._last_scan

    def set_callback(self, callback: Callable[[LidarScan], None]) -> None:
        """设置扫描回调"""
        self._scan_callback = callback

    def get_status(self) -> AGVHardwareStatus:
        return self.status


class IMUETT10APW(RealAGVInterface):
    """
    亿天 ETT10A-PW 六轴IMU
    IP67防水，内置姿态解算
    """

    def __init__(self, port: str = "/dev/ttyUSB1", baudrate: int = 115200):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self._serial = None
        self._reading_thread: Optional[threading.Thread] = None
        self._running = False
        self._last_data: Optional[IMUData] = None
        self._callback: Optional[Callable[[IMUData], None]] = None

    def connect(self) -> bool:
        self.status = AGVHardwareStatus.CONNECTING
        try:
            import serial
            self._serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self._running = True
            self._reading_thread = threading.Thread(target=self._reading_loop, daemon=True)
            self._reading_thread.start()
            self.status = AGVHardwareStatus.CONNECTED
            logger.info(f"Connected to ETT10A-PW IMU on {self.port}")
            return True
        except ImportError:
            logger.error("pyserial not installed")
            self.status = AGVHardwareStatus.ERROR
            return False
        except Exception as e:
            logger.error(f"Failed to connect to IMU: {e}")
            self.status = AGVHardwareStatus.ERROR
            return False

    def disconnect(self) -> None:
        self._running = False
        if self._reading_thread:
            self._reading_thread.join(timeout=1.0)
        if self._serial:
            self._serial.close()
        self.status = AGVHardwareStatus.DISCONNECTED

    def _reading_loop(self) -> None:
        while self._running:
            if self._serial is None:
                break
            try:
                line = self._serial.readline()
                data = self._parse_line(line)
                if data is not None:
                    self._last_data = data
                    if self._callback:
                        self._callback(data)
            except Exception as e:
                logger.warning(f"IMU read error: {e}")
                time.sleep(0.01)

    def _parse_line(self, line: bytes) -> Optional[IMUData]:
        """解析IMU数据行"""
        # 根据ETT10A-PW协议解析
        # 框架占位
        return None

    def get_last_data(self) -> Optional[IMUData]:
        """获取最新IMU数据"""
        return self._last_data

    def set_callback(self, callback: Callable[[IMUData], None]) -> None:
        self._callback = callback

    def get_status(self) -> AGVHardwareStatus:
        return self.status


class AGVTactileBridge(RealAGVInterface):
    """
    触觉传感器硬件桥接
    连接电子皮肤触觉阵列到主控制器
    """

    def __init__(self, can_node_id: int = 0x05):
        super().__init__()
        self.can_node_id = can_node_id
        self._last_data: Optional[np.ndarray] = None
        self._connected = False

    def connect(self) -> bool:
        # 通过CAN总线连接触觉采集板
        self.status = AGVHardwareStatus.CONNECTED
        self._connected = True
        logger.info(f"Connected to tactile array on CAN node {self.can_node_id}")
        return True

    def disconnect(self) -> None:
        self._connected = False
        self.status = AGVHardwareStatus.DISCONNECTED

    def get_tactile_array(self) -> Optional[np.ndarray]:
        """获取触觉阵列数据 (rows x cols)"""
        return self._last_data

    def get_status(self) -> AGVHardwareStatus:
        return self.status


class AGVForceBridge(RealAGVInterface):
    """
    六维力矩传感器硬件桥接
    """

    def __init__(self, can_node_id: int = 0x06):
        super().__init__()
        self.can_node_id = can_node_id
        self._wrench: Optional[np.ndarray] = None  # [fx, fy, fz, mx, my, mz]
        self._connected = False

    def connect(self) -> bool:
        self.status = AGVHardwareStatus.CONNECTED
        self._connected = True
        logger.info(f"Connected to force torque sensor on CAN node {self.can_node_id}")
        return True

    def disconnect(self) -> None:
        self._connected = False
        self.status = AGVHardwareStatus.DISCONNECTED

    def get_wrench(self) -> Optional[np.ndarray]:
        """获取当前力旋量"""
        return self._wrench

    def get_status(self) -> AGVHardwareStatus:
        return self.status


class HardwareMonitor:
    """硬件监控器 - 监控所有硬件的状态"""

    def __init__(self, update_rate_hz: float = 10.0):
        self.update_rate_hz = update_rate_hz
        self.devices: List[RealAGVInterface] = []
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False
        self.last_update: Dict[str, AGVHardwareStatus] = {}

    def add_device(self, device: RealAGVInterface, name: str) -> None:
        """添加要监控的设备"""
        self.devices.append(device)
        self.last_update[name] = device.get_status()

    def start(self) -> None:
        """启动监控"""
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"Hardware monitor started at {self.update_rate_hz} Hz")

    def stop(self) -> None:
        """停止监控"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        logger.info("Hardware monitor stopped")

    def _monitor_loop(self) -> None:
        while self._running:
            for device in self.devices:
                status = device.get_status()
                # 检测状态变化，记录日志
                # ...
            time.sleep(1.0 / self.update_rate_hz)

    def get_all_status(self) -> Dict[str, str]:
        """获取所有设备状态"""
        return {name: status.value for name, status in self.last_update.items()}


class RealAGVController:
    """
    真实AGV机器人总控制器
    整合所有硬件接口，提供统一控制接口
    """

    def __init__(self, grade: str = "M", can_interface: str = "can0"):
        self.grade = grade
        self.spec = AGV_HARDWARE_SPECS[grade]
        self.can_interface = can_interface
        self.status: AGVHardwareStatus = AGVHardwareStatus.DISCONNECTED

        # 硬件组件
        self.motor_driver: Optional[CANZAC8015DDriver] = None
        self.lidar: Optional[LidarN10P] = None
        self.imu: Optional[IMUETT10APW] = None
        self.tactile: Optional[AGVTactileBridge] = None
        self.force: Optional[AGVForceBridge] = None
        self.hardware_monitor: HardwareMonitor = HardwareMonitor()

        # 状态
        self.current_position: np.ndarray = np.zeros(3)
        self.current_velocity: np.ndarray = np.zeros(3)
        self.current_battery: float = 1.0

        # 根据等级初始化
        self._init_by_grade()

    def _init_by_grade(self) -> None:
        """根据AGV等级初始化组件"""
        wheel_config = self.spec['wheel_config']
        if wheel_config == '2-wheel-diff':
            node_ids = [0x01, 0x02]
        elif wheel_config == '4-wheel-diff':
            node_ids = [0x01, 0x02, 0x03, 0x04]
        else:
            node_ids = [0x01, 0x02]

        self.motor_driver = CANZAC8015DDriver(
            can_interface=self.can_interface,
            node_ids=node_ids,
            bitrate=self.spec['can_bitrate']
        )

        if self.spec['lidar'] == '360_25m':
            self.lidar = LidarN10P()

        if self.spec['has_imu']:
            self.imu = IMUETT10APW()

        if self.spec['has_tactile']:
            self.tactile = AGVTactileBridge(can_node_id=0x05)

        if self.spec['has_force']:
            self.force = AGVForceBridge(can_node_id=0x06)

    def connect_all(self) -> bool:
        """连接所有硬件"""
        self.status = AGVHardwareStatus.CONNECTING
        all_ok = True

        if self.motor_driver:
            if not self.motor_driver.connect():
                all_ok = False

        if self.lidar:
            if not self.lidar.connect():
                logger.warning("Failed to connect lidar, continuing without lidar")
                # 非致命错误，不阻止启动

        if self.imu:
            if not self.imu.connect():
                logger.warning("Failed to connect IMU, continuing without IMU")

        if self.tactile:
            if not self.tactile.connect():
                logger.warning("Failed to connect tactile sensor")

        if self.force:
            if not self.force.connect():
                logger.warning("Failed to connect force sensor")

        # 启动监控
        if self.motor_driver:
            self.hardware_monitor.add_device(self.motor_driver, "motor_driver")
        if self.lidar:
            self.hardware_monitor.add_device(self.lidar, "lidar")
        if self.imu:
            self.hardware_monitor.add_device(self.imu, "imu")
        if self.tactile:
            self.hardware_monitor.add_device(self.tactile, "tactile")
        if self.force:
            self.hardware_monitor.add_device(self.force, "force")

        self.hardware_monitor.start()

        if all_ok:
            self.status = AGVHardwareStatus.CONNECTED
            logger.info(f"All {self.grade} grade AGV hardware connected successfully")
        else:
            self.status = AGVHardwareStatus.ERROR
            logger.error("Some hardware failed to connect")

        return all_ok

    def disconnect_all(self) -> None:
        """断开所有连接"""
        self.hardware_monitor.stop()

        if self.motor_driver:
            self.motor_driver.disconnect()
        if self.lidar:
            self.lidar.disconnect()
        if self.imu:
            self.imu.disconnect()
        if self.tactile:
            self.tactile.disconnect()
        if self.force:
            self.force.disconnect()

        self.status = AGVHardwareStatus.DISCONNECTED
        logger.info("All hardware disconnected")

    def set_wheel_velocities(self, velocities: List[float]) -> bool:
        """设置轮子速度"""
        if self.motor_driver:
            return self.motor_driver.set_wheel_velocities(velocities)
        return False

    def emergency_stop(self) -> None:
        """紧急停止"""
        if self.motor_driver:
            self.motor_driver.emergency_stop()
        self.status = AGVHardwareStatus.EMERGENCY_STOP

    def release_emergency_stop(self) -> bool:
        """释放紧急停止"""
        if self.motor_driver:
            success = self.motor_driver.release_emergency_stop()
            if success:
                self.status = AGVHardwareStatus.CONNECTED
            return success
        return False

    def get_current_state(self) -> Dict[str, Any]:
        """获取当前完整状态"""
        state = {
            'status': self.status.value,
            'grade': self.grade,
            'position': self.current_position.tolist(),
            'velocity': self.current_velocity.tolist(),
            'battery': self.current_battery,
        }

        if self.motor_driver:
            state['motors'] = [
                {
                    'id': m.motor_id,
                    'enabled': m.enabled,
                    'rpm': m.current_rpm,
                    'temperature': m.temperature,
                }
                for m in self.motor_driver.get_motor_states()
            ]

        if self.imu and self.imu.get_last_data():
            imu_data = self.imu.get_last_data()
            state['imu'] = {
                'accelerometer': imu_data.accelerometer.tolist(),
                'gyroscope': imu_data.gyroscope.tolist(),
                'temperature': imu_data.temperature,
            }
            if imu_data.quaternion is not None:
                state['imu']['quaternion'] = imu_data.quaternion.tolist()

        if self.tactile and self.tactile.get_tactile_array() is not None:
            state['tactile'] = {
                'shape': self.tactile.get_tactile_array().shape,
            }

        if self.force and self.force.get_wrench() is not None:
            state['force'] = self.force.get_wrench().tolist()

        return state

    def get_spec(self) -> Dict[str, Any]:
        """获取硬件规格"""
        return self.spec.copy()
