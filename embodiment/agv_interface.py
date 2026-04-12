"""
AGV Hardware Interface - 真实AGV硬件抽象层接口
支持多种AGV硬件通信协议：CAN总线、Modbus、ROS、TCP/IP等
"""

import time
import math
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
try:
    import can
    CAN_AVAILABLE = True
except ImportError:
    CAN_AVAILABLE = False
    can = None


class AGVCommunicationType(Enum):
    """AGV通信类型"""
    CAN = "can"
    MODBUS = "modbus"
    ROS = "ros"
    TCP = "tcp"
    UDP = "udp"


class AGVStatus(Enum):
    """AGV运行状态"""
    IDLE = "idle"
    MOVING = "moving"
    PAUSED = "paused"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class AGVConfig:
    """AGV硬件配置"""
    agv_id: int
    communication_type: AGVCommunicationType = AGVCommunicationType.CAN
    can_interface: str = "can0"
    can_bitrate: int = 500000
    tcp_host: str = "192.168.1.100"
    tcp_port: int = 8080
    max_velocity: float = 1.5
    max_omega: float = 2.0
    wheel_radius: float = 0.076
    wheel_distance: float = 0.32
    battery_voltage_full: float = 24.0
    battery_voltage_empty: float = 20.0


@dataclass
class AGVState:
    """AGV状态数据"""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    v: float = 0.0
    omega: float = 0.0
    battery_level: float = 1.0
    battery_voltage: float = 24.0
    current_left: float = 0.0
    current_right: float = 0.0
    temperature_motor_left: float = 25.0
    temperature_motor_right: float = 25.0
    error_code: int = 0
    emergency_stop: bool = False
    gripper_state: str = "open"
    timestamp: float = 0.0


@dataclass
class AGVCommand:
    """AGV控制指令"""
    v: float = 0.0  # 线速度 (m/s)
    omega: float = 0.0  # 角速度 (rad/s)
    gripper_command: str = "idle"  # open/close/hold/idle
    led_color: Tuple[int, int, int] = (0, 255, 0)  # RGB
    buzzer: bool = False


class AGVHardwareInterface:
    """
    AGV硬件接口抽象层
    统一不同品牌AGV的控制接口，支持多种通信协议
    """

    def __init__(self, config: AGVConfig):
        self.config = config
        self.connected = False
        self.last_state = AGVState()
        self.can_bus: Optional[can.Bus] = None
        self.tcp_socket = None

    def connect(self) -> bool:
        """连接AGV硬件，返回是否成功"""
        try:
            if self.config.communication_type == AGVCommunicationType.CAN:
                if not CAN_AVAILABLE:
                    print(f"CAN module not available, cannot connect AGV {self.config.agv_id}")
                    self.connected = False
                    return False
                # 初始化CAN总线
                self.can_bus = can.interface.Bus(
                    channel=self.config.can_interface,
                    bustype='socketcan',
                    bitrate=self.config.can_bitrate
                )
                self.connected = True
            elif self.config.communication_type == AGVCommunicationType.TCP:
                # 初始化TCP连接
                import socket
                self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.tcp_socket.connect((self.config.tcp_host, self.config.tcp_port))
                self.connected = True
            # 其他通信类型实现类似逻辑

            # 发送心跳包验证连接
            self._send_heartbeat()
            return True
        except Exception as e:
            print(f"Failed to connect to AGV {self.config.agv_id}: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """断开AGV连接"""
        if self.can_bus:
            self.can_bus.shutdown()
            self.can_bus = None
        if self.tcp_socket:
            self.tcp_socket.close()
            self.tcp_socket = None
        self.connected = False

    def send_command(self, command: AGVCommand) -> bool:
        """发送控制指令到AGV，返回是否成功"""
        if not self.connected:
            return False

        try:
            # 速度限幅
            v = max(-self.config.max_velocity, min(self.config.max_velocity, command.v))
            omega = max(-self.config.max_omega, min(self.config.max_omega, command.omega))

            # 差速转换为左右轮速度
            v_left = v - omega * self.config.wheel_distance / 2
            v_right = v + omega * self.config.wheel_distance / 2
            w_left = v_left / self.config.wheel_radius  # rad/s
            w_right = v_right / self.config.wheel_radius

            # 根据通信类型发送指令
            if self.config.communication_type == AGVCommunicationType.CAN:
                # 构建CAN报文：速度指令
                data = bytearray(8)
                # 左轮速度 (int16, 单位: 0.01 rad/s)
                w_left_int = int(np.clip(w_left * 100, -32767, 32767))
                data[0] = (w_left_int >> 8) & 0xFF
                data[1] = w_left_int & 0xFF
                # 右轮速度
                w_right_int = int(np.clip(w_right * 100, -32767, 32767))
                data[2] = (w_right_int >> 8) & 0xFF
                data[3] = w_right_int & 0xFF
                # 夹爪指令
                if command.gripper_command == "open":
                    data[4] = 0x01
                elif command.gripper_command == "close":
                    data[4] = 0x02
                elif command.gripper_command == "hold":
                    data[4] = 0x03
                else:
                    data[4] = 0x00
                # LED颜色
                data[5] = command.led_color[0] // 2
                data[6] = command.led_color[1] // 2
                data[7] = command.led_color[2] // 2

                msg = can.Message(
                    arbitration_id=0x100 + self.config.agv_id,
                    data=data,
                    is_extended_id=False
                )
                self.can_bus.send(msg)
            elif self.config.communication_type == AGVCommunicationType.TCP:
                # 构建TCP数据包
                packet = f"CMD,{v:.2f},{omega:.2f},{command.gripper_command}\n".encode()
                self.tcp_socket.send(packet)

            return True
        except Exception as e:
            print(f"Failed to send command to AGV {self.config.agv_id}: {e}")
            self.connected = False
            return False

    def get_state(self) -> Optional[AGVState]:
        """读取AGV最新状态，返回None如果失败"""
        if not self.connected:
            return None

        try:
            if self.config.communication_type == AGVCommunicationType.CAN:
                # 读取CAN总线上的状态报文
                msg = self.can_bus.recv(timeout=0.01)
                if msg and msg.arbitration_id == 0x200 + self.config.agv_id:
                    data = msg.data
                    # 解析位置
                    x = int.from_bytes(data[0:2], byteorder='big', signed=True) / 100.0
                    y = int.from_bytes(data[2:4], byteorder='big', signed=True) / 100.0
                    theta = int.from_bytes(data[4:6], byteorder='big', signed=True) / 100.0
                    # 解析速度
                    v = int.from_bytes(data[6:7], byteorder='big', signed=True) / 10.0
                    omega = int.from_bytes(data[7:8], byteorder='big', signed=True) / 10.0

                    # 读取电池状态报文
                    msg_batt = self.can_bus.recv(timeout=0.01)
                    if msg_batt and msg_batt.arbitration_id == 0x210 + self.config.agv_id:
                        voltage = int.from_bytes(msg_batt.data[0:2], byteorder='big') / 100.0
                        battery_level = max(0.0, min(1.0,
                            (voltage - self.config.battery_voltage_empty) /
                            (self.config.battery_voltage_full - self.config.battery_voltage_empty)
                        ))
                        self.last_state.battery_voltage = voltage
                        self.last_state.battery_level = battery_level

                    self.last_state.x = x
                    self.last_state.y = y
                    self.last_state.theta = theta
                    self.last_state.v = v
                    self.last_state.omega = omega
                    self.last_state.timestamp = time.time()

            elif self.config.communication_type == AGVCommunicationType.TCP:
                # 读取TCP返回的状态数据
                data = self.tcp_socket.recv(1024).decode().strip()
                if data.startswith("STATE,"):
                    parts = data.split(",")
                    if len(parts) >= 7:
                        self.last_state.x = float(parts[1])
                        self.last_state.y = float(parts[2])
                        self.last_state.theta = float(parts[3])
                        self.last_state.v = float(parts[4])
                        self.last_state.omega = float(parts[5])
                        self.last_state.battery_level = float(parts[6])
                        self.last_state.timestamp = time.time()

            return self.last_state
        except Exception as e:
            print(f"Failed to read AGV {self.config.agv_id} state: {e}")
            self.connected = False
            return None

    def emergency_stop(self) -> bool:
        """发送紧急停止指令，返回是否成功"""
        if not self.connected:
            return False
        # 发送零速度指令
        return self.send_command(AGVCommand(v=0.0, omega=0.0, buzzer=True, led_color=(255, 0, 0)))

    def _send_heartbeat(self):
        """发送心跳包"""
        if self.config.communication_type == AGVCommunicationType.CAN:
            msg = can.Message(
                arbitration_id=0x001 + self.config.agv_id,
                data=[0x01],
                is_extended_id=False
            )
            self.can_bus.send(msg)
        elif self.config.communication_type == AGVCommunicationType.TCP:
            self.tcp_socket.send(b"HEARTBEAT\n")

    def is_connected(self) -> bool:
        """返回AGV是否在线"""
        return self.connected


class AGVInterface:
    """
    高层AGV接口，封装硬件通信、路径规划、任务执行等功能
    对外提供统一的AGV控制接口
    """

    def __init__(self, agv_id: str, agv_type: str = "AUTO_GUIDED_VEHICLE_LEVEL_5"):
        self.agv_id = agv_id
        self.agv_type = agv_type
        self.status = AGVStatus.IDLE
        # 解析AGV等级
        self.level = int(agv_type.split("_")[-1]) if "LEVEL_" in agv_type else 1
        # 初始化硬件配置
        self.config = AGVConfig(
            agv_id=int(agv_id.split("_")[-1]) if "_" in agv_id else 1,
            max_velocity=1.0 + 0.1 * self.level
        )
        # 硬件接口
        self.hw_interface = AGVHardwareInterface(self.config)
        # 当前位置
        self.current_pos = (0.0, 0.0, 0.0)
        # 目标位置
        self.target_pos = None
        # 传感器数据缓存
        self.sensor_cache = {}

    def connect(self) -> bool:
        """连接AGV硬件"""
        return self.hw_interface.connect()

    def disconnect(self):
        """断开AGV连接"""
        self.hw_interface.disconnect()

    def move_to(self, x: float, y: float, theta: float = 0.0, speed: float = None) -> Dict:
        """
        移动AGV到目标位置
        返回执行结果
        """
        if self.status == AGVStatus.EMERGENCY_STOP:
            return {"success": False, "error": "AGV is in emergency stop state"}
        
        self.target_pos = (x, y, theta)
        self.status = AGVStatus.MOVING
        
        # 速度设置
        if speed is None:
            speed = self.config.max_velocity
        speed = min(speed, self.config.max_velocity)

        # 简化实现：直接更新位置，实际应该路径规划+逐步移动
        self.current_pos = (x, y, theta)
        
        # 发送运动指令到底层硬件
        cmd = AGVCommand(v=speed, omega=0.0)
        self.hw_interface.send_command(cmd)
        
        return {"success": True, "target_pos": self.target_pos, "speed": speed}

    def stop(self) -> Dict:
        """停止AGV运动"""
        self.status = AGVStatus.IDLE
        self.target_pos = None
        # 发送停止指令
        self.hw_interface.send_command(AGVCommand(v=0.0, omega=0.0))
        return {"success": True}

    def emergency_stop(self) -> Dict:
        """紧急停止"""
        self.status = AGVStatus.EMERGENCY_STOP
        self.hw_interface.emergency_stop()
        return {"success": True}

    def get_sensor_data(self) -> Dict:
        """获取所有传感器数据"""
        # 读取硬件状态
        hw_state = self.hw_interface.get_state()
        if hw_state:
            self.current_pos = (hw_state.x, hw_state.y, hw_state.theta)
        
        # 组装传感器数据
        self.sensor_cache = {
            "position": self.current_pos,
            "status": self.status.value,
            "battery_level": hw_state.battery_level if hw_state else 1.0,
            "tactile": [],  # 后续接入触觉传感器数据
            "force": [],    # 后续接入力觉传感器数据
            "imu": {}       # 后续接入IMU传感器数据
        }
        
        return self.sensor_cache

    def is_connected(self) -> bool:
        """返回AGV是否连接"""
        return self.hw_interface.is_connected()

    def get_current_state(self) -> AGVState:
        """返回AGV当前状态"""
        return self.hw_interface.get_state() or AGVState()
