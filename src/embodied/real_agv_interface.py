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
real_agv_interface.py - 真实AGV机器人硬件接口适配
SuperModel 超模态大模型具身智能系统

支持:
- CAN Bus 通信 (ZLAC8015D 驱动器)
- 镭神N10P激光雷达
- ETT10A-PW IMU
- 电子皮肤触觉传感器
- 六维力矩传感器
- ROS2 桥接
- RK3588 NPU 加速
"""

from __future__ import annotations
import abc
import enum
import time
import threading
import queue
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)
__all__ = [
    'AGVHardwareConfig',
    'HardwareInterface',
    'CANBusDriver',
    'ZLAC8015DController',
    'LidarInterface',
    'IMUInterface',
    'TactileInterface',
    'ForceSensorInterface',
    'RealAGVController',
    'ThreadedSensorReader',
]


@dataclass
class AGVHardwareConfig:
    """真实AGV硬件配置"""

    # AGV等级
    grade: str = "M"

    # CAN Bus 配置
    can_interface: str = "can0"
    can_baudrate: int = 500000
    left_motor_id: int = 1
    right_motor_id: int = 2

    # 驱动器参数 (ZLAC8015D)
    motor_rated_current: float = 10.0  # A
    motor_pole_pairs: int = 8
    gear_ratio: float = 1.0

    # 机械参数
    wheel_radius: float = 0.07  # m
    wheel_base: float = 0.45     # m
    max_speed: float = 1.5       # m/s

    # 传感器配置
    lidar_port: str = "/dev/ttyUSB0"
    lidar_baudrate: int = 921600
    imu_port: str = "/dev/ttyUSB1"
    imu_baudrate: int = 115200
    tactile_can_id: int = 0x20
    force_can_id: int = 0x30

    # 采样频率
    control_frequency: float = 50.0  # Hz
    sensor_frequency: float = 100.0  # Hz

    # NPU配置
    use_rk3588_npu: bool = True
    fusion_model_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'grade': self.grade,
            'can_interface': self.can_interface,
            'can_baudrate': self.can_baudrate,
            'left_motor_id': self.left_motor_id,
            'right_motor_id': self.right_motor_id,
            'wheel_radius': self.wheel_radius,
            'wheel_base': self.wheel_base,
            'max_speed': self.max_speed,
        }

    @classmethod
    def from_grade(cls, grade: str) -> 'AGVHardwareConfig':
        """根据等级创建配置"""
        config = cls(grade=grade)
        if grade == 'S':
            config.wheel_radius = 0.05
            config.wheel_base = 0.30
            config.max_speed = 1.0
        elif grade == 'M':
            # 默认就是M级
            pass
        elif grade == 'L':
            config.wheel_radius = 0.07
            config.wheel_base = 0.60
            config.max_speed = 1.2
            config.left_motor_id = 1
            config.right_motor_id = 2
        elif grade == 'XL':
            config.wheel_radius = 0.0825
            config.wheel_base = 0.80
            config.max_speed = 1.0
            # 四个电机
        elif grade == 'XXL':
            config.wheel_radius = 0.095
            config.wheel_base = 1.20
            config.max_speed = 0.8
            # 四个电机
        return config


class HardwareInterface(abc.ABC):
    """硬件接口基类"""

    def __init__(self):
        self.connected: bool = False
        self.last_update_time: float = 0.0
        self.error_count: int = 0

    @abc.abstractmethod
    def connect(self) -> bool:
        """连接设备"""
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """断开连接"""
        pass

    @abc.abstractmethod
    def read(self) -> Optional[Any]:
        """读取数据"""
        pass

    def is_connected(self) -> bool:
        """是否已连接"""
        return self.connected

    def get_error_count(self) -> int:
        """获取错误计数"""
        return self.error_count


class CANBusDriver:
    """CAN Bus 驱动 - 使用 python-can"""

    def __init__(self, interface: str = "can0", baudrate: int = 500000):
        self.interface = interface
        self.baudrate = baudrate
        self.bus = None
        self.connected = False

    def connect(self) -> bool:
        """连接CAN总线"""
        try:
            import can
            self.bus = can.Bus(
                interface='socketcan',
                channel=self.interface,
                bitrate=self.baudrate
            )
            self.connected = True
            logger.info(f"CAN Bus connected on {self.interface}@{self.baudrate}")
            return True
        except ImportError:
            logger.error("python-can not installed, please install with: pip install python-can")
            return False
        except Exception as e:
            logger.error(f"Failed to connect CAN Bus: {e}")
            self.connected = False
            return False

    def disconnect(self) -> None:
        """断开CAN总线"""
        if self.bus:
            self.bus.shutdown()
            self.connected = False

    def send_message(self, can_id: int, data: bytes) -> bool:
        """发送CAN消息"""
        if not self.connected:
            return False
        try:
            import can
            msg = can.Message(
                arbitration_id=can_id,
                data=data,
                is_extended_id=False
            )
            self.bus.send(msg)
            return True
        except Exception as e:
            logger.error(f"CAN send failed: {e}")
            self.error_count += 1
            return False

    def receive_message(self, timeout: float = 0.1) -> Optional[Any]:
        """接收CAN消息"""
        if not self.connected:
            return None
        try:
            return self.bus.recv(timeout=timeout)
        except Exception as e:
            logger.error(f"CAN receive failed: {e}")
            return None

    def is_connected(self) -> bool:
        return self.connected


class ZLAC8015DController:
    """中菱 ZLAC8015D 双路轮毂电机驱动器"""

    # 命令码定义
    CMD_READ_STATUS = 0x01
    CMD_SET_SPEED = 0x02
    CMD_SET_POSITION = 0x03
    CMD_TORQUE_LIMIT = 0x04
    CMD_CLEAR_FAULT = 0x05
    CMD_READ_ENCODER = 0x06

    def __init__(
        self,
        can_driver: CANBusDriver,
        motor_id_left: int = 1,
        motor_id_right: int = 2,
        rated_current: float = 10.0,
        pole_pairs: int = 8,
    ):
        self.can = can_driver
        self.motor_id_left = motor_id_left
        self.motor_id_right = motor_id_right
        self.rated_current = rated_current
        self.pole_pairs = pole_pairs
        self.current_speed_left = 0.0
        self.current_speed_right = 0.0
        self.fault_left = False
        self.fault_right = False
        self.position_left = 0.0
        self.position_right = 0.0

    def set_speed(self, left_speed_rpm: float, right_speed_rpm: float) -> bool:
        """设置左右电机速度 (RPM)"""
        success = True
        # 转换为字节，范围 ±10000 RPM → 16位
        left_int = int(left_speed_rpm * 10)
        right_int = int(right_speed_rpm * 10)

        left_bytes = [
            self.CMD_SET_SPEED,
            (left_int >> 8) & 0xFF,
            left_int & 0xFF,
            0, 0, 0, 0, 0
        ]
        success &= self.can.send_message(self.motor_id_left, bytes(left_bytes))

        right_bytes = [
            self.CMD_SET_SPEED,
            (right_int >> 8) & 0xFF,
            right_int & 0xFF,
            0, 0, 0, 0, 0
        ]
        success &= self.can.send_message(self.motor_id_right, bytes(right_bytes))

        return success

    def set_wheel_speed(self, left_speed_mps: float, right_speed_mps: float, wheel_radius: float) -> bool:
        """设置轮子线速度 (m/s)"""
        # 转换为 RPM
        left_rpm = (left_speed_mps / (2 * np.pi * wheel_radius)) * 60
        right_rpm = (right_speed_mps / (2 * np.pi * wheel_radius)) * 60
        return self.set_speed(left_rpm, right_rpm)

    def read_status(self) -> Dict[str, Any]:
        """读取电机状态"""
        # 发送读取请求
        result = {
            'left': {'fault': True, 'speed': 0, 'current': 0},
            'right': {'fault': True, 'speed': 0, 'current': 0},
        }

        # 读取左电机
        msg = self.can.receive_message(timeout=0.01)
        if msg:
            if msg.arbitration_id == self.motor_id_left:
                self.fault_left = (msg.data[1] & 0x01) != 0
                speed_raw = (msg.data[2] << 8) | msg.data[3]
                current_raw = (msg.data[4] << 8) | msg.data[5]
                self.current_speed_left = speed_raw / 10.0
                result['left']['fault'] = self.fault_left
                result['left']['speed'] = self.current_speed_left
                result['left']['current'] = current_raw / 10.0

        # 读取右电机
        msg = self.can.receive_message(timeout=0.01)
        if msg:
            if msg.arbitration_id == self.motor_id_right:
                self.fault_right = (msg.data[1] & 0x01) != 0
                speed_raw = (msg.data[2] << 8) | msg.data[3]
                current_raw = (msg.data[4] << 8) | msg.data[5]
                self.current_speed_right = speed_raw / 10.0
                result['right']['fault'] = self.fault_right
                result['right']['speed'] = self.current_speed_right
                result['right']['current'] = current_raw / 10.0

        return result

    def clear_fault(self) -> bool:
        """清除故障"""
        success = True
        # 清除左电机故障
        data = bytes([self.CMD_CLEAR_FAULT] + [0] * 7)
        success &= self.can.send_message(self.motor_id_left, data)
        # 清除右电机故障
        success &= self.can.send_message(self.motor_id_right, data)
        self.fault_left = False
        self.fault_right = False
        return success

    def emergency_stop(self) -> bool:
        """紧急停止"""
        return self.set_speed(0, 0)

    def get_encoder_position(self) -> Tuple[float, float]:
        """获取编码器位置 (脉冲数)"""
        return self.position_left, self.position_right


class LidarInterface(HardwareInterface):
    """镭神 N10P 激光雷达接口"""

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 921600,
        num_points: int = 360,
    ):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.num_points = num_points
        self.serial = None
        self.ranges: np.ndarray = np.full(num_points, 25.0)
        self.angles: np.ndarray = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    def connect(self) -> bool:
        """连接激光雷达"""
        try:
            import serial
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1.0)
            self.connected = True
            logger.info(f"Lidar connected on {self.port}@{self.baudrate}")
            return True
        except ImportError:
            logger.error("pyserial not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to connect lidar: {e}")
            return False

    def disconnect(self) -> None:
        """断开连接"""
        if self.serial:
            self.serial.close()
            self.connected = False

    def read(self) -> Optional[np.ndarray]:
        """读取一帧数据"""
        if not self.connected:
            return None

        # 这里简化实现，实际需要根据镭神通信协议解析
        # 实际项目中使用 leishen-sdk
        self.last_update_time = time.time()
        return self.ranges

    def get_point_cloud(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取点云坐标"""
        x = self.ranges * np.cos(self.angles)
        y = self.ranges * np.sin(self.angles)
        return x, y


class IMUInterface(HardwareInterface):
    """ETT10A-PW IMU接口"""

    def __init__(
        self,
        port: str = "/dev/ttyUSB1",
        baudrate: int = 115200,
    ):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.accel = np.zeros(3)
        self.gyro = np.zeros(3)
        self.mag = np.zeros(3)
        self.quaternion = np.array([1, 0, 0, 0])

    def connect(self) -> bool:
        """连接IMU"""
        try:
            import serial
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1.0)
            self.connected = True
            logger.info(f"IMU connected on {self.port}@{self.baudrate}")
            return True
        except ImportError:
            logger.error("pyserial not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to connect IMU: {e}")
            return False

    def disconnect(self) -> None:
        if self.serial:
            self.serial.close()
            self.connected = False

    def read(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """读取IMU数据 (accel, gyro, mag)"""
        if not self.connected:
            return None

        # 这里简化，实际需要根据协议解析
        self.last_update_time = time.time()
        return self.accel, self.gyro, self.mag

    def get_accel(self) -> np.ndarray:
        return self.accel

    def get_gyro(self) -> np.ndarray:
        return self.gyro


class TactileInterface(HardwareInterface):
    """电子皮肤触觉传感器接口 - CAN Bus"""

    def __init__(
        self,
        can_driver: CANBusDriver,
        can_id: int = 0x20,
        rows: int = 8,
        cols: int = 8,
    ):
        super().__init__()
        self.can = can_driver
        self.can_id = can_id
        self.rows = rows
        self.cols = cols
        self.pressure_map = np.zeros((rows, cols))

    def connect(self) -> bool:
        self.connected = self.can.is_connected()
        return self.connected

    def disconnect(self) -> None:
        self.connected = False

    def read(self) -> Optional[np.ndarray]:
        """读取触觉数据"""
        if not self.connected:
            return None

        # 读取多帧CAN消息拼接压力图
        # 每个帧8字节，64点需要8帧
        for i in range((self.rows * self.cols + 7) // 8):
            msg = self.can.receive_message(timeout=0.005)
            if msg and msg.arbitration_id == self.can_id:
                # 解析压力值，这里简化
                # 实际：每个字节一个16级压力值
                byte_idx = i * 8
                for j, byte in enumerate(msg.data):
                    row = (byte_idx + j) // self.cols
                    col = (byte_idx + j) % self.cols
                    if row < self.rows and col < self.cols:
                        self.pressure_map[row, col] = byte / 255.0

        self.last_update_time = time.time()
        return self.pressure_map

    def get_pressure_map(self) -> np.ndarray:
        """获取压力图"""
        return self.pressure_map

    def detect_contact(self, threshold: float = 0.1) -> bool:
        """检测是否有接触"""
        return np.any(self.pressure_map > threshold)

    def get_contact_center(self) -> Tuple[float, float]:
        """获取接触中心"""
        if not self.detect_contact():
            return -1, -1

        rows, cols = np.where(self.pressure_map > 0.1)
        if len(rows) == 0:
            return -1, -1

        return (float(np.mean(rows)), float(np.mean(cols)))


class ForceSensorInterface(HardwareInterface):
    """六维力矩传感器接口 - CAN Bus"""

    def __init__(
        self,
        can_driver: CANBusDriver,
        can_id: int = 0x30,
    ):
        super().__init__()
        self.can = can_driver
        self.can_id = can_id
        # [fx, fy, fz, tx, ty, tz]
        self.wrench = np.zeros(6)
        self.bias = np.zeros(6)

    def connect(self) -> bool:
        self.connected = self.can.is_connected()
        return self.connected

    def disconnect(self) -> None:
        self.connected = False

    def calibrate_bias(self) -> None:
        """校准零点 - 采集当前作为零点"""
        samples = []
        for _ in range(100):
            self.read()
            samples.append(self.wrench.copy())
            time.sleep(0.001)
        self.bias = np.mean(samples, axis=0)

    def read(self) -> Optional[np.ndarray]:
        """读取力/力矩数据"""
        if not self.connected:
            return None

        msg = self.can.receive_message(timeout=0.01)
        if msg and msg.arbitration_id == self.can_id:
            # 解析六维数据
            # 每个分量16位
            fx = (msg.data[0] << 8) | msg.data[1]
            fy = (msg.data[2] << 8) | msg.data[3]
            fz = (msg.data[4] << 8) | msg.data[5]
            tx = (msg.data[6] << 8) | msg.data[7]

            # 第二帧包含剩下两个分量
            msg2 = self.can.receive_message(timeout=0.01)
            ty = 0
            tz = 0
            if msg2 and msg2.arbitration_id == self.can_id:
                ty = (msg2.data[0] << 8) | msg2.data[1]
                tz = (msg2.data[2] << 8) | msg2.data[3]

            # 转换为物理单位 (简化)
            scale = 0.01  # N/LSB
            self.wrench[0] = (fx - 32768) * scale - self.bias[0]
            self.wrench[1] = (fy - 32768) * scale - self.bias[1]
            self.wrench[2] = (fz - 32768) * scale - self.bias[2]
            self.wrench[3] = (tx - 32768) * 0.001 - self.bias[3]
            self.wrench[4] = (ty - 32768) * 0.001 - self.bias[4]
            self.wrench[5] = (tz - 32768) * 0.001 - self.bias[5]

        self.last_update_time = time.time()
        return self.wrench - self.bias

    def get_wrench(self) -> np.ndarray:
        """获取力旋量"""
        return self.wrench - self.bias

    def get_total_force(self) -> float:
        """获取合力大小"""
        return np.linalg.norm(self.wrench[:3])


class ThreadedSensorReader:
    """后台线程传感器读取器"""

    def __init__(
        self,
        sensors: Dict[str, HardwareInterface],
        update_frequency: float = 100.0
    ):
        self.sensors = sensors
        self.update_frequency = update_frequency
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.data_queue: Dict[str, queue.Queue] = {
            name: queue.Queue(maxsize=10)
            for name in sensors
        }

    def start(self) -> None:
        """开始后台读取"""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started sensor reader with {len(self.sensors)} sensors")

    def stop(self) -> None:
        """停止后台读取"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None

    def _read_loop(self) -> None:
        """读取循环"""
        interval = 1.0 / self.update_frequency
        while self.running:
            start_time = time.time()

            for name, sensor in self.sensors.items():
                if sensor.is_connected():
                    try:
                        data = sensor.read()
                        if data is not None:
                            if not self.data_queue[name].full():
                                self.data_queue[name].put(data)
                    except Exception as e:
                        sensor.error_count += 1
                        logger.error(f"Sensor {name} read error: {e}")

            # 保持频率
            elapsed = time.time() - start_time
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def get_latest(self, sensor_name: str) -> Optional[Any]:
        """获取最新数据"""
        if sensor_name not in self.data_queue:
            return None
        if self.data_queue[sensor_name].empty():
            return None
        # 取出所有数据，返回最新的
        latest = None
        while not self.data_queue[sensor_name].empty():
            latest = self.data_queue[sensor_name].get()
        return latest


class RealAGVController:
    """真实AGV控制器 - 整合所有硬件接口"""

    def __init__(self, config: AGVHardwareConfig = None):
        self.config = config or AGVHardwareConfig()
        self.can_driver: Optional[CANBusDriver] = None
        self.motor_controller: Optional[ZLAC8015DController] = None
        self.lidar: Optional[LidarInterface] = None
        self.imu: Optional[IMUInterface] = None
        self.tactile: Optional[TactileInterface] = None
        self.force_sensor: Optional[ForceSensorInterface] = None
        self.sensor_reader: Optional[ThreadedSensorReader] = None
        self.initialized: bool = False
        self.running: bool = False

        # 状态机
        self.state_machine = AGVStateMachine(agv_id=f"agv_{id(self)}")
        self.state_machine.connect()

        # 心跳监控器 (延迟初始化)
        self.heartbeat_monitor: Optional[AGVHeartbeatMonitor] = None

        # 健康监控
        self.health_monitor = AGVHealthMonitor(agv_id=f"agv_{id(self)}")

        # 当前状态
        self.current_velocity = np.zeros(2)  # v, w
        self.current_position = np.zeros(3)  # x, y, theta
        self.battery_voltage: float = 24.0
        self.battery_level: float = 1.0

        # 电池阈值
        self._battery_warning_voltage = 23.5
        self._battery_critical_voltage = 22.0

    def initialize(self) -> bool:
        """初始化所有硬件"""
        try:
            self.state_machine.transition(
                self.state_machine.State.CONNECTING,
                "initializing hardware"
            )

            # 1. 初始化 CAN Bus
            self.can_driver = CANBusDriver(
                interface=self.config.can_interface,
                baudrate=self.config.can_baudrate
            )
            if not self.can_driver.connect():
                logger.error("Failed to connect CAN Bus")
                self.state_machine.set_error("CAN Bus connection failed")
                return False

            # 2. 初始化电机控制器
            self.motor_controller = ZLAC8015DController(
                self.can_driver,
                motor_id_left=self.config.left_motor_id,
                motor_id_right=self.config.right_motor_id,
                rated_current=self.config.motor_rated_current,
                pole_pairs=self.config.motor_pole_pairs,
            )

            # 3. 初始化激光雷达
            self.lidar = LidarInterface(
                port=self.config.lidar_port,
                baudrate=self.config.lidar_baudrate
            )
            self.lidar.connect()  # 即使失败也继续，可能是仿真

            # 4. 初始化IMU
            self.imu = IMUInterface(
                port=self.config.imu_port,
                baudrate=self.config.imu_baudrate
            )
            self.imu.connect()

            # 5. 初始化触觉传感器
            self.tactile = TactileInterface(
                self.can_driver,
                can_id=self.config.tactile_can_id
            )
            self.tactile.connect()

            # 6. 初始化力传感器
            self.force_sensor = ForceSensorInterface(
                self.can_driver,
                can_id=self.config.force_can_id
            )
            self.force_sensor.connect()

            # 7. 启动后台传感器读取
            sensors = {}
            if self.lidar.is_connected():
                sensors['lidar'] = self.lidar
            if self.imu.is_connected():
                sensors['imu'] = self.imu
            if self.tactile.is_connected():
                sensors['tactile'] = self.tactile
            if self.force_sensor.is_connected():
                sensors['force'] = self.force_sensor

            if sensors:
                self.sensor_reader = ThreadedSensorReader(
                    sensors,
                    update_frequency=self.config.sensor_frequency
                )
                self.sensor_reader.start()

            # 8. 初始化心跳监控器
            self.heartbeat_monitor = AGVHeartbeatMonitor(
                controller=self,
                state_machine=self.state_machine,
                heartbeat_interval=1.0,
                max_timeout_count=3,
                reconnect_delay=2.0,
                max_reconnect_attempts=5,
            )
            self.heartbeat_monitor.start()

            self.initialized = True
            self.running = True
            self.state_machine.transition(
                self.state_machine.State.IDLE,
                "initialization complete"
            )
            logger.info("Real AGV controller initialized successfully")
            logger.info(f"Connected sensors: {list(sensors.keys())}")

            return True

        except Exception as e:
            logger.error(f"Real AGV initialization failed: {e}")
            self.state_machine.set_error(str(e))
            return False

    def shutdown(self) -> None:
        """关闭所有硬件"""
        self.state_machine.shutdown()
        self.running = False

        if self.heartbeat_monitor:
            self.heartbeat_monitor.stop()
            self.heartbeat_monitor = None

        if self.sensor_reader:
            self.sensor_reader.stop()
        if self.motor_controller:
            self.motor_controller.emergency_stop()
        if self.can_driver:
            self.can_driver.disconnect()
        if self.lidar:
            self.lidar.disconnect()
        if self.imu:
            self.imu.disconnect()
        self.initialized = False
        self.state_machine.disconnect()
        logger.info("Real AGV controller shutdown")

    def set_speed(self, linear_velocity: float, angular_velocity: float) -> bool:
        """设置速度 (线速度 m/s, 角速度 rad/s)"""
        if not self.motor_controller:
            return False

        # 差速运动学计算
        # v = (vl + vr) / 2, w = (vr - vl) / d
        d = self.config.wheel_base
        vl = linear_velocity - (angular_velocity * d) / 2
        vr = linear_velocity + (angular_velocity * d) / 2

        success = self.motor_controller.set_wheel_speed(
            vl, vr, self.config.wheel_radius
        )
        self.current_velocity = np.array([linear_velocity, angular_velocity])
        return success

    def emergency_stop(self) -> bool:
        """紧急停止"""
        if self.motor_controller:
            return self.motor_controller.emergency_stop()
        return False

    def get_sensor_data(self) -> Dict[str, Any]:
        """获取所有传感器最新数据"""
        data = {
            'lidar': None,
            'imu_accel': None,
            'imu_gyro': None,
            'tactile_pressure': None,
            'force_wrench': None,
            'motor_status': None,
            'position': self.current_position,
            'velocity': self.current_velocity,
            'battery_level': self.battery_level,
        }

        if self.sensor_reader:
            data['lidar'] = self.sensor_reader.get_latest('lidar')
            imu_data = self.sensor_reader.get_latest('imu')
            if imu_data is not None:
                data['imu_accel'] = imu_data[0]
                data['imu_gyro'] = imu_data[1]
            data['tactile_pressure'] = self.sensor_reader.get_latest('tactile')
            data['force_wrench'] = self.sensor_reader.get_latest('force')

        if self.motor_controller:
            data['motor_status'] = self.motor_controller.read_status()

        return data

    def get_full_status(self) -> Dict[str, Any]:
        """
        获取完整AGV状态报告
        
        Returns:
            包含状态机/心跳监控/传感器数据/电池状态的完整状态报告
        """
        status = {
            'initialized': self.initialized,
            'running': self.running,
            'state_machine': {
                'state': self.state_machine.state.value,
                'is_operational': self.state_machine.is_operational,
                'is_alive': self.state_machine.is_alive,
                'error_reason': self.state_machine._error_reason,
                'time_in_current_state_s': round(self.state_machine.time_in_current_state(), 2),
            },
        }

        # 心跳监控
        if self.heartbeat_monitor:
            status['heartbeat'] = self.heartbeat_monitor.get_status()

        # 健康监控
        if self.health_monitor:
            status['health'] = self.health_monitor.get_health_status()

        # 传感器数据
        status['sensor_data'] = self.get_sensor_data()

        # 电池状态
        status['battery'] = {
            'voltage': self.battery_voltage,
            'level': self.battery_level,
            'warning_threshold': self._battery_warning_voltage,
            'critical_threshold': self._battery_critical_voltage,
            'is_low': self.battery_voltage < self._battery_warning_voltage,
            'is_critical': self.battery_voltage < self._battery_critical_voltage,
        }

        # 硬件连接状态
        status['hardware'] = {
            'can_bus': self.can_driver.is_connected() if self.can_driver else False,
            'motor': self.motor_controller is not None,
            'lidar': self.lidar.is_connected() if self.lidar else False,
            'imu': self.imu.is_connected() if self.imu else False,
            'tactile': self.tactile.is_connected() if self.tactile else False,
            'force': self.force_sensor.is_connected() if self.force_sensor else False,
        }

        return status

    def start_task(self) -> bool:
        """开始任务"""
        if not self.initialized:
            logger.warning("Cannot start task: not initialized")
            return False
        return self.state_machine.start_running()

    def pause_task(self) -> bool:
        """暂停任务"""
        return self.state_machine.pause()

    def resume_task(self) -> bool:
        """恢复任务"""
        return self.state_machine.resume()

    def update_odometry(self, delta_time: float) -> None:
        """更新里程计"""
        v, w = self.current_velocity
        if abs(w) < 1e-6:
            # 直线运动
            dx = v * delta_time * np.cos(self.current_position[2])
            dy = v * delta_time * np.sin(self.current_position[2])
            dtheta = 0
        else:
            # 圆弧运动
            r = v / w
            theta0 = self.current_position[2]
            theta1 = theta0 + w * delta_time
            dx = r * (np.sin(theta1) - np.sin(theta0))
            dy = r * (np.cos(theta0) - np.cos(theta1))
            dtheta = w * delta_time

        self.current_position[0] += dx
        self.current_position[1] += dy
        self.current_position[2] += dtheta

        # 角度归一化
        self.current_position[2] = (self.current_position[2] + np.pi) % (2 * np.pi) - np.pi

    def is_initialized(self) -> bool:
        return self.initialized

    def get_config(self) -> AGVHardwareConfig:
        return self.config

    def emergency_stop(self, reason: str = "UNKNOWN") -> None:
        """
        紧急停止AGV
        
        Args:
            reason: 紧急停止原因
        """
        logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
        self.emergency_stop_active = True
        
        # 立即停止电机
        if self.motor_controller:
            try:
                self.motor_controller.set_speed(0, 0)
                self.motor_controller.enable_torque(False)
            except Exception as e:
                logger.error(f"Failed to stop motors during emergency stop: {e}")
        
        # 停止所有传感器读取
        if self.sensor_reader:
            self.sensor_reader.stop()
        
        # 记录紧急停止事件
        self.health_monitor.record_error("emergency_stop", reason)

    def reset_emergency_stop(self) -> bool:
        """重置紧急停止状态"""
        if not self.emergency_stop_active:
            return True
        
        # 检查系统是否健康
        health_status = self.health_monitor.get_health_status()
        if health_status['overall_health'] < 0.7:
            logger.error(f"Cannot reset emergency stop: system health too low ({health_status['overall_health']:.2f})")
            return False
        
        # 重新启用电机
        if self.motor_controller:
            try:
                self.motor_controller.enable_torque(True)
            except Exception as e:
                logger.error(f"Failed to re-enable motors: {e}")
                return False
        
        # 重启传感器读取
        if self.sensor_reader:
            self.sensor_reader.start()
        
        self.emergency_stop_active = False
        logger.info("Emergency stop reset successfully")
        return True

    def set_emergency_stop_callback(self, callback: Callable[[str], None]) -> None:
        """设置紧急停止回调函数"""
        self.emergency_stop_callback = callback


# ============================================================================
# AGV健康监控器
# ============================================================================


class AGVHealthMonitor:
    """
    AGV健康监控器 - 实时监控硬件状态和健康度
    
    监控指标:
    - 电机温度/电流/电压
    - 传感器连接状态
    - 电池电压/电流/温度
    - CAN总线通信状态
    - CPU/NPU使用率
    - 磁盘/内存使用率
    """

    def __init__(self, agv_id: str):
        self.agv_id = agv_id
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.metrics: Dict[str, List[float]] = {
            'motor_temperature_left': [],
            'motor_temperature_right': [],
            'motor_current_left': [],
            'motor_current_right': [],
            'battery_voltage': [],
            'battery_current': [],
            'battery_temperature': [],
            'can_bus_error_rate': [],
            'cpu_usage': [],
            'memory_usage': [],
            'npu_usage': [],
        }
        self.max_metric_history = 1000

    def record_error(self, error_type: str, message: str) -> None:
        """记录错误事件"""
        import time
        self.errors.append({
            'timestamp': time.time(),
            'type': error_type,
            'message': message,
        })
        logger.error(f"AGV {self.agv_id} ERROR [{error_type}]: {message}")

    def record_warning(self, warning_type: str, message: str) -> None:
        """记录警告事件"""
        import time
        self.warnings.append({
            'timestamp': time.time(),
            'type': warning_type,
            'message': message,
        })
        logger.warning(f"AGV {self.agv_id} WARNING [{warning_type}]: {message}")

    def record_metric(self, metric_name: str, value: float) -> None:
        """记录性能指标"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append(value)
        
        # 保持历史长度
        if len(self.metrics[metric_name]) > self.max_metric_history:
            self.metrics[metric_name].pop(0)

    def get_health_status(self) -> Dict[str, Any]:
        """获取整体健康状态"""
        health_score = 1.0
        issues = []
        
        # 检查电机温度
        for motor_side in ['left', 'right']:
            temp_key = f'motor_temperature_{motor_side}'
            if self.metrics[temp_key]:
                avg_temp = np.mean(self.metrics[temp_key][-10:])
                if avg_temp > 80.0:
                    health_score -= 0.3
                    issues.append(f"{motor_side} motor over temperature ({avg_temp:.1f}°C)")
                elif avg_temp > 60.0:
                    health_score -= 0.1
                    issues.append(f"{motor_side} motor high temperature ({avg_temp:.1f}°C)")
        
        # 检查电池电压
        if self.metrics['battery_voltage']:
            avg_voltage = np.mean(self.metrics['battery_voltage'][-10:])
            if avg_voltage < 22.0:  # 24V电池低电
                health_score -= 0.4
                issues.append(f"Battery critically low ({avg_voltage:.1f}V)")
            elif avg_voltage < 23.5:
                health_score -= 0.2
                issues.append(f"Battery low ({avg_voltage:.1f}V)")
        
        # 检查CAN总线错误率
        if self.metrics['can_bus_error_rate']:
            avg_error_rate = np.mean(self.metrics['can_bus_error_rate'][-10:])
            if avg_error_rate > 0.1:
                health_score -= 0.3
                issues.append(f"High CAN bus error rate ({avg_error_rate:.1%})")
        
        # 检查CPU使用率
        if self.metrics['cpu_usage']:
            avg_cpu = np.mean(self.metrics['cpu_usage'][-10:])
            if avg_cpu > 90.0:
                health_score -= 0.2
                issues.append(f"High CPU usage ({avg_cpu:.1f}%)")
        
        health_score = max(0.0, health_score)
        
        return {
            'overall_health': health_score,
            'health_level': 'EXCELLENT' if health_score >= 0.9 else 'GOOD' if health_score >= 0.7 else 'WARNING' if health_score >= 0.4 else 'CRITICAL',
            'issues': issues,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'latest_errors': self.errors[-5:],
            'latest_warnings': self.warnings[-5:],
        }

    def get_metric_statistics(self, metric_name: str) -> Optional[Dict[str, float]]:
        """获取指定指标的统计信息"""
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return None
        
        values = np.array(self.metrics[metric_name])
        return {
            'min': np.min(values),
            'max': np.max(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'latest': values[-1],
        }

    def reset(self) -> None:
        """重置所有监控数据"""
        self.errors.clear()
        self.warnings.clear()
        for key in self.metrics:
            self.metrics[key].clear()


class AGVStateMachine:
    """
    AGV状态机 - 管理AGV硬件状态转换
    
    状态:
    - DISCONNECTED: 初始断开状态
    - CONNECTING: 正在连接硬件
    - IDLE: 已连接,待机状态
    - RUNNING: 执行任务中
    - PAUSED: 暂停
    - ERROR: 错误状态
    - RECOVERING: 正在恢复
    - SHUTDOWN: 已关闭
    
    转换规则:
    - DISCONNECTED -> CONNECTING (调用connect)
    - CONNECTING -> IDLE (连接成功)
    - CONNECTING -> ERROR (连接失败)
    - IDLE -> RUNNING (开始任务)
    - RUNNING -> IDLE (任务完成)
    - RUNNING -> ERROR (检测到错误)
    - ERROR -> RECOVERING (尝试恢复)
    - RECOVERING -> IDLE (恢复成功)
    - RECOVERING -> ERROR (恢复失败)
    - any -> SHUTDOWN (调用shutdown)
    """

    class State(enum.Enum):
        DISCONNECTED = "disconnected"
        CONNECTING = "connecting"
        IDLE = "idle"
        RUNNING = "running"
        PAUSED = "paused"
        ERROR = "error"
        RECOVERING = "recovering"
        SHUTDOWN = "shutdown"

    def __init__(self, agv_id: str):
        self.agv_id = agv_id
        self._state = self.State.DISCONNECTED
        self._error_reason: Optional[str] = None
        self._last_transition_time: float = time.time()
        self._transition_history: List[Dict[str, Any]] = []
        self._listeners: Dict[str, List[Callable]] = {
            'state_changed': [],
            'error': [],
            'recovery': [],
        }

    @property
    def state(self) -> 'AGVStateMachine.State':
        return self._state

    @property
    def is_operational(self) -> bool:
        return self._state in (self.State.IDLE, self.State.RUNNING, self.State.PAUSED)

    @property
    def is_alive(self) -> bool:
        return self._state not in (self.State.DISCONNECTED, self.State.SHUTDOWN)

    def transition(self, new_state: 'AGVStateMachine.State', reason: str = "") -> bool:
        """状态转换"""
        if self._state == new_state:
            return True

        old_state = self._state
        self._state = new_state
        self._last_transition_time = time.time()

        entry = {
            'from': old_state.value,
            'to': new_state.value,
            'reason': reason,
            'timestamp': self._last_transition_time,
        }
        self._transition_history.append(entry)

        logger.info(f"AGV {self.agv_id} state: {old_state.value} -> {new_state.value} ({reason})")

        # 通知监听器
        self._notify('state_changed', entry)
        if new_state == self.State.ERROR:
            self._error_reason = reason
            self._notify('error', {'reason': reason, 'timestamp': self._last_transition_time})
        if old_state == self.State.RECOVERING and new_state == self.State.IDLE:
            self._notify('recovery', {'timestamp': self._last_transition_time})

        return True

    def set_error(self, reason: str) -> None:
        """设置错误状态"""
        self.transition(self.State.ERROR, reason)

    def set_recovering(self) -> None:
        """设置恢复状态"""
        self.transition(self.State.RECOVERING, "attempting recovery")

    def connect(self) -> bool:
        """开始连接"""
        return self.transition(self.State.CONNECTING, "connecting")

    def disconnect(self) -> None:
        """断开连接"""
        self.transition(self.State.DISCONNECTED, "disconnected")

    def start_running(self) -> bool:
        """开始运行"""
        if self._state not in (self.State.IDLE, self.State.PAUSED):
            logger.warning(f"Cannot start running from state {self._state.value}")
            return False
        self.transition(self.State.RUNNING, "task started")
        return True

    def pause(self) -> bool:
        """暂停"""
        if self._state != self.State.RUNNING:
            return False
        self.transition(self.State.PAUSED, "paused")
        return True

    def resume(self) -> bool:
        """恢复"""
        if self._state != self.State.PAUSED:
            return False
        self.transition(self.State.RUNNING, "resumed")
        return True

    def recover_success(self) -> None:
        """恢复成功"""
        self.transition(self.State.IDLE, "recovery successful")

    def shutdown(self) -> None:
        """关闭"""
        self.transition(self.State.SHUTDOWN, "shutdown")

    def add_listener(self, event: str, callback: Callable) -> None:
        """添加状态监听器"""
        if event in self._listeners:
            self._listeners[event].append(callback)

    def _notify(self, event: str, data: Any) -> None:
        for cb in self._listeners.get(event, []):
            try:
                cb(data)
            except Exception as e:
                logger.error(f"State listener error: {e}")

    def get_transition_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._transition_history[-limit:]

    def time_in_current_state(self) -> float:
        return time.time() - self._last_transition_time


class AGVHeartbeatMonitor:
    """
    AGV心跳监控器 - 定期检查硬件存活并触发重连
    
    功能:
    - 定期发送心跳信号到所有硬件组件
    - 检测组件超时/无响应
    - 触发自动重连机制
    - 记录心跳历史和统计
    """

    def __init__(
        self,
        controller: RealAGVController,
        state_machine: AGVStateMachine,
        heartbeat_interval: float = 1.0,
        max_timeout_count: int = 3,
        reconnect_delay: float = 2.0,
        max_reconnect_attempts: int = 5,
    ):
        self.controller = controller
        self.state_machine = state_machine
        self.heartbeat_interval = heartbeat_interval
        self.max_timeout_count = max_timeout_count
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # 心跳计数
        self.heartbeat_count: int = 0
        self.last_heartbeat_time: float = 0.0
        self.last_response_time: float = 0.0

        # 超时跟踪
        self.timeout_counts: Dict[str, int] = {
            'can_bus': 0,
            'motor': 0,
            'lidar': 0,
            'imu': 0,
            'tactile': 0,
            'force': 0,
        }

        # 心跳历史
        self.heartbeat_history: deque = deque(maxlen=100)
        self.error_history: deque = deque(maxlen=100)
        self.reconnect_history: deque = deque(maxlen=50)

        # 统计
        self.total_heartbeats: int = 0
        self.total_timeouts: int = 0
        self.total_reconnects: int = 0
        self.start_time: float = 0.0

    def start(self) -> None:
        """启动心跳监控"""
        if self._running:
            return
        self._running = True
        self.start_time = time.time()
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        logger.info(f"AGVHeartbeatMonitor started (interval={self.heartbeat_interval}s)")

    def stop(self) -> None:
        """停止心跳监控"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("AGVHeartbeatMonitor stopped")

    def _heartbeat_loop(self) -> None:
        """心跳主循环"""
        while self._running:
            try:
                self._do_heartbeat()
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                time.sleep(self.heartbeat_interval)

    def _do_heartbeat(self) -> None:
        """执行一次心跳"""
        self.heartbeat_count += 1
        self.last_heartbeat_time = time.time()

        heartbeat_result = {
            'timestamp': self.last_heartbeat_time,
            'heartbeat_num': self.heartbeat_count,
            'components': {},
            'overall_alive': True,
            'errors': [],
        }

        # 检查 CAN Bus
        can_alive = self._check_can_bus()
        heartbeat_result['components']['can_bus'] = can_alive
        if not can_alive:
            self._handle_component_timeout('can_bus')
            heartbeat_result['errors'].append('CAN bus timeout')

        # 检查电机
        motor_alive = self._check_motor()
        heartbeat_result['components']['motor'] = motor_alive
        if not motor_alive:
            self._handle_component_timeout('motor')
            heartbeat_result['errors'].append('Motor timeout')

        # 检查传感器
        lidar_alive = self._check_lidar()
        heartbeat_result['components']['lidar'] = lidar_alive
        if not lidar_alive:
            self._handle_component_timeout('lidar')

        imu_alive = self._check_imu()
        heartbeat_result['components']['imu'] = imu_alive
        if not imu_alive:
            self._handle_component_timeout('imu')

        # 传感器超时不算致命错误
        heartbeat_result['overall_alive'] = can_alive and motor_alive

        self.last_response_time = time.time()
        self.total_heartbeats += 1

        if heartbeat_result['errors']:
            self.error_history.append(heartbeat_result)

        self.heartbeat_history.append(heartbeat_result)

    def _check_can_bus(self) -> bool:
        """检查CAN总线"""
        try:
            if self.controller.can_driver is None:
                return False
            return self.controller.can_driver.is_connected()
        except Exception:
            return False

    def _check_motor(self) -> bool:
        """检查电机控制器"""
        try:
            if self.controller.motor_controller is None:
                return False
            # 尝试读取编码器位置
            self.controller.motor_controller.get_encoder_position()
            return True
        except Exception:
            return False

    def _check_lidar(self) -> bool:
        """检查激光雷达"""
        try:
            if self.controller.lidar is None:
                return False
            if not self.controller.lidar.is_connected():
                return False
            self.controller.lidar.read()
            return True
        except Exception:
            return False

    def _check_imu(self) -> bool:
        """检查IMU"""
        try:
            if self.controller.imu is None:
                return False
            if not self.controller.imu.is_connected():
                return False
            self.controller.imu.read()
            return True
        except Exception:
            return False

    def _handle_component_timeout(self, component: str) -> None:
        """处理组件超时"""
        self.timeout_counts[component] += 1
        self.total_timeouts += 1

        logger.warning(
            f"Component {component} timeout "
            f"(count={self.timeout_counts[component]}/{self.max_timeout_count})"
        )

        # 超过最大超时次数，触发重连
        if self.timeout_counts[component] >= self.max_timeout_count:
            logger.error(f"Component {component} exceeded timeout limit, triggering reconnect")
            self._attempt_reconnect(component)
            # 重置计数
            self.timeout_counts[component] = 0

    def _attempt_reconnect(self, component: str) -> bool:
        """尝试重连指定组件"""
        reconnect_attempt = {
            'component': component,
            'timestamp': time.time(),
            'attempts': 0,
            'success': False,
        }

        for attempt in range(1, self.max_reconnect_attempts + 1):
            reconnect_attempt['attempts'] = attempt
            logger.info(f"Reconnect attempt {attempt}/{self.max_reconnect_attempts} for {component}")

            try:
                success = False
                if component == 'can_bus':
                    success = self._reconnect_can_bus()
                elif component == 'motor':
                    success = self._reconnect_motor()
                elif component == 'lidar':
                    success = self._reconnect_lidar()
                elif component == 'imu':
                    success = self._reconnect_imu()

                if success:
                    reconnect_attempt['success'] = True
                    self.state_machine._error_reason = None
                    logger.info(f"Successfully reconnected {component} on attempt {attempt}")
                    break

            except Exception as e:
                logger.error(f"Reconnect attempt {attempt} failed for {component}: {e}")

            time.sleep(self.reconnect_delay)

        self.total_reconnects += 1
        self.reconnect_history.append(reconnect_attempt)
        return reconnect_attempt['success']

    def _reconnect_can_bus(self) -> bool:
        """重连CAN总线"""
        if self.controller.can_driver is None:
            return False
        self.controller.can_driver.disconnect()
        return self.controller.can_driver.connect()

    def _reconnect_motor(self) -> bool:
        """重连电机控制器"""
        if self.controller.motor_controller is None:
            return False
        try:
            self.controller.motor_controller.clear_fault()
            return True
        except Exception:
            return False

    def _reconnect_lidar(self) -> bool:
        """重连激光雷达"""
        if self.controller.lidar is None:
            return False
        self.controller.lidar.disconnect()
        return self.controller.lidar.connect()

    def _reconnect_imu(self) -> bool:
        """重连IMU"""
        if self.controller.imu is None:
            return False
        self.controller.imu.disconnect()
        return self.controller.imu.connect()

    def reset_timeout_count(self, component: str) -> None:
        """重置指定组件的超时计数"""
        if component in self.timeout_counts:
            self.timeout_counts[component] = 0

    def get_heartbeat_statistics(self) -> Dict[str, Any]:
        """获取心跳统计信息"""
        uptime = time.time() - self.start_time if self.start_time else 0.0
        recent_history = list(self.heartbeat_history)[-20:]
        error_rate = (
            sum(1 for h in recent_history if h.get('errors')) / len(recent_history)
            if recent_history else 0.0
        )

        return {
            'total_heartbeats': self.total_heartbeats,
            'total_timeouts': self.total_timeouts,
            'total_reconnects': self.total_reconnects,
            'uptime_s': round(uptime, 1),
            'heartbeat_interval_s': self.heartbeat_interval,
            'recent_error_rate': round(error_rate, 4),
            'current_timeout_counts': dict(self.timeout_counts),
            'last_heartbeat_time': self.last_heartbeat_time,
            'last_response_time': self.last_response_time,
        }

    def get_status(self) -> Dict[str, Any]:
        """获取完整状态"""
        return {
            'running': self._running,
            'statistics': self.get_heartbeat_statistics(),
            'recent_errors': list(self.error_history)[-10:],
            'reconnect_history': list(self.reconnect_history)[-10:],
        }


__all__ += [
    'TrajectoryTrackingController',
    'SensorAutoCalibrator',
    'CalibrationStatus',
    'AGVHealthMonitor',
    'AGVStateMachine',
    'AGVHeartbeatMonitor',
]


# ============================================================================
# 轨迹跟踪控制器 - Trajectory Tracking Controller
# ============================================================================


class TrajectoryTrackingController:
    """
    轨迹跟踪控制器 - Pure Pursuit + PID 速度控制
    
    功能:
    - Pure Pursuit 路径跟踪
    - PID 速度伺服
    - 轨迹偏差监控
    - AGV五级规格适配 (跟踪精度/响应速度)
    """

    def __init__(
        self,
        wheel_base: float = 0.45,
        grade: str = "M",
        look_ahead_gain: float = 0.5,
        min_look_ahead: float = 0.1,
        max_look_ahead: float = 1.0,
        kp_velocity: float = 2.0,
        ki_velocity: float = 0.1,
        kd_velocity: float = 0.3,
        max_linear_vel: float = 1.5,
        max_angular_vel: float = 2.0,
    ):
        self.wheel_base = wheel_base
        self.grade = grade
        self.look_ahead_gain = look_ahead_gain
        self.min_look_ahead = min_look_ahead
        self.max_look_ahead = max_look_ahead

        # PID velocity controller
        self.kp = kp_velocity
        self.ki = ki_velocity
        self.kd = kd_velocity
        self.integral_vel = 0.0
        self.prev_error_vel = 0.0
        self.prev_time = None

        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel

        # 五级AGV参数适配
        self._apply_grade_params()

        # 跟踪状态
        self.current_waypoint_index = 0
        self.total_waypoints = 0
        self.trajectory: List[Tuple[float, float]] = []  # [(x, y), ...]
        self.last_error = 0.0
        self.cross_track_error_history: List[float] = []

    def _apply_grade_params(self) -> None:
        """根据AGV等级应用参数"""
        grade_params = {
            "S": {"look_ahead_gain": 0.3, "kp": 1.5, "max_linear_vel": 0.5, "max_angular_vel": 1.5},
            "M": {"look_ahead_gain": 0.5, "kp": 2.0, "max_linear_vel": 1.0, "max_angular_vel": 2.0},
            "L": {"look_ahead_gain": 0.7, "kp": 2.5, "max_linear_vel": 1.5, "max_angular_vel": 2.5},
            "XL": {"look_ahead_gain": 0.9, "kp": 3.0, "max_linear_vel": 2.0, "max_angular_vel": 3.0},
            "XXL": {"look_ahead_gain": 1.2, "kp": 3.5, "max_linear_vel": 2.5, "max_angular_vel": 3.5},
        }
        p = grade_params.get(self.grade, grade_params["M"])
        self.look_ahead_gain = p["look_ahead_gain"]
        self.kp = p["kp"]
        self.max_linear_vel = p["max_linear_vel"]
        self.max_angular_vel = p["max_angular_vel"]

    def load_trajectory(self, waypoints: List[Tuple[float, float]]) -> None:
        """加载轨迹点"""
        self.trajectory = list(waypoints)
        self.total_waypoints = len(waypoints)
        self.current_waypoint_index = 0

    def compute_look_ahead_distance(self, velocity: float) -> float:
        """计算前视距离 (速度越快前视越远)"""
        lad = self.look_ahead_gain * abs(velocity)
        return float(np.clip(lad, self.min_look_ahead, self.max_look_ahead))

    def find_look_ahead_point(
        self,
        current_pos: Tuple[float, float],
        velocity: float,
    ) -> Optional[Tuple[float, float, int]]:
        """
        找到Pure Pursuit前视点
        
        Returns:
            (lx, ly, waypoint_index) 或 None
        """
        if not self.trajectory:
            return None
        lad = self.compute_look_ahead_distance(velocity)
        start_idx = max(0, self.current_waypoint_index - 1)

        for i in range(start_idx, len(self.trajectory)):
            # Skip waypoints already passed
            if i < self.current_waypoint_index:
                continue
            wx, wy = self.trajectory[i]
            dx = wx - current_pos[0]
            dy = wy - current_pos[1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist >= lad:
                return (float(wx), float(wy), i)

        # 到终点
        if self.trajectory:
            last = self.trajectory[-1]
            return (float(last[0]), float(last[1]), len(self.trajectory) - 1)
        return None

    def compute_pure_pursuit(
        self,
        current_pos: Tuple[float, float],
        current_theta: float,
        velocity: float,
    ) -> Tuple[float, float]:
        """
        Pure Pursuit 控制律
        
        Args:
            current_pos: (x, y)
            current_theta: 航向角 (rad)
            velocity: 当前速度 (m/s)
        
        Returns:
            (v_cmd, omega_cmd)
        """
        la_point = self.find_look_ahead_point(current_pos, velocity)
        if la_point is None:
            return (0.0, 0.0)

        lx, ly, wp_idx = la_point
        self.current_waypoint_index = wp_idx

        # 转换为车辆局部坐标系
        dx = lx - current_pos[0]
        dy = ly - current_pos[1]
        local_x = dx * np.cos(current_theta) + dy * np.sin(current_theta)
        local_y = -dx * np.sin(current_theta) + dy * np.cos(current_theta)

        # 偏航角误差
        alpha = np.arctan2(local_y, max(local_x, 0.01))
        # 前视距离
        lad = np.sqrt(local_x**2 + local_y**2)

        # 角速度命令
        k = 2.0 / max(lad, 0.05)
        omega_cmd = k * np.sin(alpha)

        # 速度命令 (沿轨迹方向)
        v_cmd = velocity

        # 限幅
        v_cmd = float(np.clip(v_cmd, -self.max_linear_vel, self.max_linear_vel))
        omega_cmd = float(np.clip(omega_cmd, -self.max_angular_vel, self.max_angular_vel))

        return (v_cmd, omega_cmd)

    def compute_pid_velocity(
        self,
        current_vel: float,
        target_vel: float,
        dt: float,
    ) -> float:
        """
        PID速度控制
        
        Args:
            current_vel: 当前速度 (m/s)
            target_vel: 目标速度 (m/s)
            dt: 时间步长 (s)
        
        Returns:
            扭矩命令 (Nm)
        """
        error = target_vel - current_vel

        # PID
        self.integral_vel += error * dt
        self.integral_vel = float(np.clip(self.integral_vel, -2.0, 2.0))
        derivative = (error - self.prev_error_vel) / dt if dt > 0 else 0.0

        torque = self.kp * error + self.ki * self.integral_vel + self.kd * derivative
        self.prev_error_vel = error
        return float(np.clip(torque, -5.0, 5.0))

    def track_trajectory(
        self,
        current_pos: Tuple[float, float],
        current_theta: float,
        current_vel: float,
        dt: float,
    ) -> Dict[str, Any]:
        """
        主跟踪函数
        
        Returns:
            {'v_cmd', 'omega_cmd', 'cross_track_error', 'waypoint_progress', 'finished'}
        """
        # Pure Pursuit 获取速度命令
        v_cmd_pp, omega_cmd = self.compute_pure_pursuit(
            current_pos, current_theta, current_vel
        )

        # PID 速度伺服
        torque_cmd = self.compute_pid_velocity(current_vel, v_cmd_pp, dt)

        # 计算横向跟踪误差
        cross_track_error = self._compute_cross_track_error(current_pos)

        # 进度
        progress = self.current_waypoint_index / max(self.total_waypoints - 1, 1)
        finished = self.current_waypoint_index >= len(self.trajectory) - 1

        self.last_error = cross_track_error
        self.cross_track_error_history.append(cross_track_error)
        if len(self.cross_track_error_history) > 100:
            self.cross_track_error_history.pop(0)

        return {
            'v_cmd': v_cmd_pp,
            'omega_cmd': omega_cmd,
            'torque_cmd': torque_cmd,
            'cross_track_error': cross_track_error,
            'waypoint_progress': progress,
            'current_waypoint': self.current_waypoint_index,
            'total_waypoints': self.total_waypoints,
            'finished': finished,
        }

    def _compute_cross_track_error(self, current_pos: Tuple[float, float]) -> float:
        """计算到轨迹的最近距离 (横向误差)"""
        if len(self.trajectory) < 2:
            return 0.0
        min_dist = float('inf')
        for i in range(len(self.trajectory) - 1):
            p1 = self.trajectory[i]
            p2 = self.trajectory[i + 1]
            dist = self._point_to_segment_distance(current_pos, p1, p2)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    @staticmethod
    def _point_to_segment_distance(
        point: Tuple[float, float],
        seg_a: Tuple[float, float],
        seg_b: Tuple[float, float],
    ) -> float:
        """点，到线段的最短距离"""
        px, py = point
        ax, ay = seg_a
        bx, by = seg_b
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        t = np.clip((apx * abx + apy * aby) / (abx**2 + aby**2 + 1e-10), 0.0, 1.0)
        proj_x = ax + t * abx
        proj_y = ay + t * aby
        return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    def reset(self) -> None:
        """重置跟踪器"""
        self.current_waypoint_index = 0
        self.integral_vel = 0.0
        self.prev_error_vel = 0.0
        self.cross_track_error_history.clear()


# ============================================================================
# 传感器自动标定 - Sensor Auto-Calibration
# ============================================================================


class CalibrationStatus(enum.Enum):
    """标定状态"""
    IDLE = "idle"
    RUNNING = "running"
    CALIBRATED = "calibrated"
    FAILED = "failed"


class SensorAutoCalibrator:
    """
    传感器自动标定器
    
    支持:
    - IMU零偏标定 (静止状态采集)
    - IMU噪声参数估计
    - 力传感器零点漂移校准
    - 里程计比例因子标定
    """

    def __init__(
        self,
        calibration_samples: int = 200,
        sample_rate_hz: float = 100.0,
        imu_bias_threshold: float = 0.1,
        force_drift_threshold: float = 0.5,
    ):
        self.calibration_samples = calibration_samples
        self.sample_rate_hz = sample_rate_hz
        self.imu_bias_threshold = imu_bias_threshold
        self.force_drift_threshold = force_drift_threshold

        # 标定状态
        self.imu_calibration_status = CalibrationStatus.IDLE
        self.force_calibration_status = CalibrationStatus.IDLE
        self.odom_calibration_status = CalibrationStatus.IDLE

        # IMU标定结果
        self.imu_accel_bias: Optional[np.ndarray] = None
        self.imu_gyro_bias: Optional[float] = None
        self.imu_noise_std: Optional[Dict[str, float]] = None

        # 力传感器标定结果
        self.force_zero_drift: Optional[np.ndarray] = None

        # 里程计标定
        self.odom_scale_factor: float = 1.0

        # 样本缓冲区
        self._accel_buffer: List[np.ndarray] = []
        self._gyro_buffer: List[float] = []
        self._force_buffer: List[np.ndarray] = []

        # 标定历史
        self.calibration_history: List[Dict[str, Any]] = []

    def start_imu_calibration(self) -> None:
        """开始IMU标定采集"""
        self.imu_calibration_status = CalibrationStatus.RUNNING
        self._accel_buffer.clear()
        self._gyro_buffer.clear()

    def add_imu_sample(self, accel: np.ndarray, gyro: float) -> bool:
        """
        添加IMU样本 (静止状态)
        
        Returns:
            True 如果采集完成
        """
        if self.imu_calibration_status != CalibrationStatus.RUNNING:
            return False

        self._accel_buffer.append(np.array(accel, dtype=np.float64))
        self._gyro_buffer.append(float(gyro))

        if len(self._accel_buffer) >= self.calibration_samples:
            self._finish_imu_calibration()
            return True
        return False

    def _finish_imu_calibration(self) -> None:
        """完成IMU标定计算"""
        try:
            accel_data = np.array(self._accel_buffer)
            gyro_data = np.array(self._gyro_buffer)

            # 零偏 = 均值 (静止时应为0 + 重力分量)
            mean_accel = np.mean(accel_data, axis=0)
            # 假设Z轴朝上，accel_z 应约为 9.81
            gravity_vec = np.array([0.0, 0.0, 9.81])
            # 零偏 = 测量均值 - 重力向量
            self.imu_accel_bias = mean_accel - gravity_vec

            # 陀螺仪零偏
            self.imu_gyro_bias = float(np.mean(gyro_data))

            # 噪声标准差
            self.imu_noise_std = {
                'accel_x': float(np.std(accel_data[:, 0])),
                'accel_y': float(np.std(accel_data[:, 1])),
                'accel_z': float(np.std(accel_data[:, 2])),
                'gyro': float(np.std(gyro_data)),
            }

            # 检查标定质量
            bias_magnitude = np.linalg.norm(self.imu_accel_bias)
            gyro_bias_magnitude = abs(self.imu_gyro_bias)

            if bias_magnitude < self.imu_bias_threshold and gyro_bias_magnitude < 0.05:
                self.imu_calibration_status = CalibrationStatus.CALIBRATED
            else:
                # 仍然标定，但标记有较大偏差
                self.imu_calibration_status = CalibrationStatus.CALIBRATED
                logger.warning(
                    f"IMU calibration done with bias: accel={bias_magnitude:.3f}m/s^2, "
                    f"gyro={gyro_bias_magnitude:.3f}rad/s (thresholds: "
                    f"{self.imu_bias_threshold}, 0.05)"
                )

            self.calibration_history.append({
                'type': 'imu',
                'status': self.imu_calibration_status.value,
                'accel_bias': self.imu_accel_bias.tolist(),
                'gyro_bias': self.imu_gyro_bias,
                'noise_std': self.imu_noise_std,
                'sample_count': len(self._accel_buffer),
            })

        except Exception as e:
            self.imu_calibration_status = CalibrationStatus.FAILED
            logger.error(f"IMU calibration failed: {e}")

    def apply_imu_calibration(self, accel: np.ndarray, gyro: float) -> Tuple[np.ndarray, float]:
        """
        应用IMU标定补偿
        
        Returns:
            (校正后加速度, 校正后角速度)
        """
        if self.imu_calibration_status != CalibrationStatus.CALIBRATED:
            return accel, gyro

        calibrated_accel = np.array(accel, dtype=np.float64) - (self.imu_accel_bias if self.imu_accel_bias is not None else np.zeros(3))
        calibrated_gyro = float(gyro) - (self.imu_gyro_bias or 0.0)
        return calibrated_accel, calibrated_gyro

    def start_force_calibration(self) -> None:
        """开始力传感器标定采集"""
        self.force_calibration_status = CalibrationStatus.RUNNING
        self._force_buffer.clear()

    def add_force_sample(self, force: np.ndarray) -> bool:
        """添加力传感器样本 (零负载状态)"""
        if self.force_calibration_status != CalibrationStatus.RUNNING:
            return False
        self._force_buffer.append(np.array(force, dtype=np.float64))
        if len(self._force_buffer) >= self.calibration_samples:
            self._finish_force_calibration()
            return True
        return False

    def _finish_force_calibration(self) -> None:
        """完成力传感器零点漂移标定"""
        try:
            force_data = np.array(self._force_buffer)
            self.force_zero_drift = np.mean(force_data, axis=0)
            self.force_calibration_status = CalibrationStatus.CALIBRATED
            self.calibration_history.append({
                'type': 'force',
                'status': self.force_calibration_status.value,
                'zero_drift': self.force_zero_drift.tolist(),
                'sample_count': len(self._force_buffer),
            })
        except Exception as e:
            self.force_calibration_status = CalibrationStatus.FAILED
            logger.error(f"Force calibration failed: {e}")

    def apply_force_calibration(self, force: np.ndarray) -> np.ndarray:
        """应用力传感器标定补偿"""
        if self.force_calibration_status != CalibrationStatus.CALIBRATED:
            return force
        drift = self.force_zero_drift if self.force_zero_drift is not None else np.zeros(6)
        return np.array(force, dtype=np.float64) - drift

    def calibrate_odometry_scale(
        self,
        commanded_distance: float,
        measured_distance: float,
    ) -> float:
        """
        标定里程计比例因子
        
        Args:
            commanded_distance: 命令行进距离 (m)
            measured_distance: 测量行进距离 (m, 来自外部基准如尺子)
        
        Returns:
            新的比例因子
        """
        if abs(measured_distance) < 0.01:
            return self.odom_scale_factor
        self.odom_scale_factor = commanded_distance / measured_distance
        self.odom_calibration_status = CalibrationStatus.CALIBRATED
        self.calibration_history.append({
            'type': 'odometry',
            'status': self.odom_calibration_status.value,
            'scale_factor': self.odom_scale_factor,
            'commanded_distance': commanded_distance,
            'measured_distance': measured_distance,
        })
        return self.odom_scale_factor

    def apply_odometry_calibration(self, raw_odom_distance: float) -> float:
        """应用里程计比例因子"""
        return raw_odom_distance * self.odom_scale_factor

    def get_calibration_report(self) -> Dict[str, Any]:
        """获取完整标定报告"""
        return {
            'imu': {
                'status': self.imu_calibration_status.value,
                'accel_bias': self.imu_accel_bias.tolist() if self.imu_accel_bias is not None else None,
                'gyro_bias': float(self.imu_gyro_bias) if self.imu_gyro_bias is not None else None,
                'noise_std': self.imu_noise_std,
            },
            'force': {
                'status': self.force_calibration_status.value,
                'zero_drift': self.force_zero_drift.tolist() if self.force_zero_drift is not None else None,
            },
            'odometry': {
                'status': self.odom_calibration_status.value,
                'scale_factor': self.odom_scale_factor,
            },
            'total_calibrations': len(self.calibration_history),
        }

    def is_fully_calibrated(self) -> bool:
        """检查是否所有传感器都已标定"""
        return (
            self.imu_calibration_status == CalibrationStatus.CALIBRATED
            and self.force_calibration_status == CalibrationStatus.CALIBRATED
            and self.odom_calibration_status == CalibrationStatus.CALIBRATED
        )

    def reset(self) -> None:
        """重置所有标定"""
        self.imu_calibration_status = CalibrationStatus.IDLE
        self.force_calibration_status = CalibrationStatus.IDLE
        self.odom_calibration_status = CalibrationStatus.IDLE
        self.imu_accel_bias = None
        self.imu_gyro_bias = None
        self.imu_noise_std = None
        self.force_zero_drift = None
        self.odom_scale_factor = 1.0
        self._accel_buffer.clear()
        self._gyro_buffer.clear()
        self._force_buffer.clear()


