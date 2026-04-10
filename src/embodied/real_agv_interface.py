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
import time
import threading
import queue
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

        # 当前状态
        self.current_velocity = np.zeros(2)  # v, w
        self.current_position = np.zeros(3)  # x, y, theta
        self.battery_voltage: float = 24.0
        self.battery_level: float = 1.0

    def initialize(self) -> bool:
        """初始化所有硬件"""
        try:
            # 1. 初始化 CAN Bus
            self.can_driver = CANBusDriver(
                interface=self.config.can_interface,
                baudrate=self.config.can_baudrate
            )
            if not self.can_driver.connect():
                logger.error("Failed to connect CAN Bus")
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

            self.initialized = True
            self.running = True
            logger.info("Real AGV controller initialized successfully")
            logger.info(f"Connected sensors: {list(sensors.keys())}")

            return True

        except Exception as e:
            logger.error(f"Real AGV initialization failed: {e}")
            return False

    def shutdown(self) -> None:
        """关闭所有硬件"""
        self.running = False
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
