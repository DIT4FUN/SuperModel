"""
AGV Robot Hardware Interface Adaptation
=======================================
支持多种通信协议的AGV硬件适配层
- CAN Bus (SocketCAN)
- Modbus TCP/RTU
- TCP/IP 自定义协议
- 仿真接口 (用于测试)

Author: SuperModel Team
"""

import time
import math
from typing import Tuple, Optional, Dict, List
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

try:
    import can
    CAN_AVAILABLE = True
except ImportError:
    CAN_AVAILABLE = False

try:
    from pymodbus.client import ModbusTcpClient, ModbusRtuClient
    MODBUS_AVAILABLE = True
except ImportError:
    MODBUS_AVAILABLE = False


class AGVType(Enum):
    """AGV类型"""
    DIFFERENTIAL = "differential"  # 差速驱动
    ACKERMANN = "ackermann"        # 阿克曼转向
    MECANUM = "mecanum"            # 麦克纳姆轮全向
    OMNI = "omni"                  # 全向轮


class AGVStatus(Enum):
    """AGV运行状态"""
    IDLE = "idle"                  # 空闲
    RUNNING = "running"            # 运行中
    PAUSED = "paused"              # 暂停
    ERROR = "error"                # 错误
    EMERGENCY_STOP = "estop"       # 紧急停止


class AGVInterface(ABC):
    """AGV接口基类，所有硬件接口都继承此类"""

    def __init__(self, agv_type: AGVType = AGVType.DIFFERENTIAL):
        self.agv_type = agv_type
        self.connected = False
        self.status = AGVStatus.IDLE
        self.error_code = 0
        self.error_message = ""
        
        # 状态数据
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        self.current_v = 0.0
        self.current_omega = 0.0
        self.battery_level = 1.0
        self.temperature = 25.0
        self.load_weight = 0.0
        
        # 传感器数据
        self.obstacle_distances: List[float] = []  # 激光雷达/超声波距离
        self.tactile_data: List[float] = []        # 触觉传感器数据
        self.force_data: List[float] = []          # 力传感器数据
        self.imu_data: Dict[str, float] = {}       # IMU数据
        
        # 配置参数
        self.max_v = 1.5  # m/s
        self.max_omega = 2.0  # rad/s
        self.max_load = 50.0  # kg

    @abstractmethod
    def connect(self) -> bool:
        """连接AGV硬件"""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """断开连接"""
        pass

    @abstractmethod
    def set_velocity(self, v: float, omega: float) -> bool:
        """设置速度指令 (线速度 m/s, 角速度 rad/s)"""
        pass

    @abstractmethod
    def update_status(self) -> None:
        """更新AGV状态数据"""
        pass

    @abstractmethod
    def emergency_stop(self) -> bool:
        """触发紧急停止"""
        pass

    @abstractmethod
    def reset_error(self) -> bool:
        """重置错误状态"""
        pass

    def get_pose(self) -> Tuple[float, float, float]:
        """获取当前位姿 (x, y, theta)"""
        return self.current_x, self.current_y, self.current_theta

    def get_velocity(self) -> Tuple[float, float]:
        """获取当前速度 (v, omega)"""
        return self.current_v, self.current_omega

    def get_battery(self) -> float:
        """获取电池电量 0.0-1.0"""
        return self.battery_level

    def is_connected(self) -> bool:
        """检查是否连接"""
        return self.connected

    def get_status(self) -> AGVStatus:
        """获取运行状态"""
        return self.status


# =============================================================================
# 仿真AGV接口 (用于测试和模拟)
# =============================================================================

class SimulatedAGV(AGVInterface):
    """仿真AGV接口，完全模拟真实AGV行为，支持传感器噪声、物理效果"""

    def __init__(
        self,
        agv_type: AGVType = AGVType.DIFFERENTIAL,
        noise_level: float = 0.02,  # 传感器噪声等级
        friction_coeff: float = 0.1,  # 摩擦系数
        response_delay: float = 0.05  # 响应延迟 (s)
    ):
        super().__init__(agv_type)
        self.noise_level = noise_level
        self.friction_coeff = friction_coeff
        self.response_delay = response_delay
        
        # 仿真状态
        self.desired_v = 0.0
        self.desired_omega = 0.0
        self.last_command_time = 0.0
        self.sim_time = 0.0
        self.dt = 0.01

    def connect(self) -> bool:
        self.connected = True
        self.status = AGVStatus.IDLE
        self.sim_time = time.time()  # 初始化仿真时间
        return True

    def disconnect(self) -> None:
        self.connected = False

    def set_velocity(self, v: float, omega: float) -> bool:
        if not self.connected or self.status == AGVStatus.EMERGENCY_STOP:
            return False
        
        self.desired_v = np.clip(v, -self.max_v, self.max_v)
        self.desired_omega = np.clip(omega, -self.max_omega, self.max_omega)
        self.last_command_time = self.sim_time
        self.status = AGVStatus.RUNNING
        return True

    def update_status(self) -> None:
        if not self.connected:
            return
        
        # 更新仿真时间
        now = time.time()
        self.dt = now - self.sim_time
        self.sim_time = now

        # 模拟响应延迟
        if self.sim_time - self.last_command_time > self.response_delay:
            if self.response_delay == 0 and self.friction_coeff == 0:
                # 理想模式，无延迟无摩擦，速度立即切换
                self.current_v = self.desired_v
                self.current_omega = self.desired_omega
            else:
                # 速度平滑过渡 (模拟惯性)
                alpha = 1.0 - math.exp(-self.dt * 5.0)
                self.current_v = alpha * self.desired_v + (1 - alpha) * self.current_v
                self.current_omega = alpha * self.desired_omega + (1 - alpha) * self.current_omega

                # 摩擦减速
                if abs(self.desired_v) < 0.01:
                    self.current_v *= (1.0 - self.friction_coeff)
                    if abs(self.current_v) < 0.01:
                        self.current_v = 0.0

                if abs(self.desired_omega) < 0.01:
                    self.current_omega *= (1.0 - self.friction_coeff)
                    if abs(self.current_omega) < 0.01:
                        self.current_omega = 0.0

        # 更新位姿 (运动学模型)
        if self.agv_type == AGVType.DIFFERENTIAL:
            # 差速运动学
            v = self.current_v
            omega = self.current_omega

            if abs(omega) < 0.001:
                # 直线运动
                self.current_x += v * math.cos(self.current_theta) * self.dt
                self.current_y += v * math.sin(self.current_theta) * self.dt
            else:
                # 圆弧运动
                R = v / omega
                self.current_x += R * (math.sin(self.current_theta + omega * self.dt) - math.sin(self.current_theta))
                self.current_y -= R * (math.cos(self.current_theta + omega * self.dt) - math.cos(self.current_theta))
                self.current_theta += omega * self.dt

        elif self.agv_type == AGVType.MECANUM:
            # 麦克纳姆轮运动学 (简化)
            self.current_x += self.current_v * math.cos(self.current_theta) * self.dt
            self.current_y += self.current_v * math.sin(self.current_theta) * self.dt
            self.current_theta += self.current_omega * self.dt

        # 角度归一化
        while self.current_theta > math.pi:
            self.current_theta -= 2 * math.pi
        while self.current_theta < -math.pi:
            self.current_theta += 2 * math.pi

        # 添加传感器噪声
        if self.noise_level > 0:
            self.current_x += np.random.normal(0, self.noise_level * 0.1)
            self.current_y += np.random.normal(0, self.noise_level * 0.1)
            self.current_theta += np.random.normal(0, self.noise_level * 0.05)
            self.battery_level -= np.random.uniform(0, 0.0001)  # 模拟耗电
            self.battery_level = max(0.0, min(1.0, self.battery_level))

        # 如果长时间没有指令，切换到空闲
        if self.sim_time - self.last_command_time > 5.0 and self.current_v == 0 and self.current_omega == 0:
            self.status = AGVStatus.IDLE

    def emergency_stop(self) -> bool:
        self.desired_v = 0.0
        self.desired_omega = 0.0
        self.current_v = 0.0
        self.current_omega = 0.0
        self.status = AGVStatus.EMERGENCY_STOP
        return True

    def reset_error(self) -> bool:
        if self.status == AGVStatus.EMERGENCY_STOP or self.status == AGVStatus.ERROR:
            self.status = AGVStatus.IDLE
            self.error_code = 0
            self.error_message = ""
            return True
        return False


# =============================================================================
# CAN Bus AGV接口 (基于SocketCAN)
# =============================================================================

class CANAGV(AGVInterface):
    """CAN Bus接口适配，支持标准CAN 2.0A/B协议"""

    def __init__(
        self,
        agv_type: AGVType = AGVType.DIFFERENTIAL,
        channel: str = "can0",
        bitrate: int = 500000,
        can_ids: Dict[str, int] = None
    ):
        super().__init__(agv_type)
        self.channel = channel
        self.bitrate = bitrate
        self.bus = None
        
        # CAN ID映射
        self.can_ids = can_ids or {
            "velocity_cmd": 0x100,
            "status_report": 0x200,
            "sensor_report": 0x201,
            "estop_cmd": 0x300,
            "reset_cmd": 0x301
        }

    def connect(self) -> bool:
        if not CAN_AVAILABLE:
            self.error_code = -1
            self.error_message = "python-can library not installed"
            return False
        
        try:
            self.bus = can.interface.Bus(channel=self.channel, bustype='socketcan', bitrate=self.bitrate)
            self.connected = True
            self.status = AGVStatus.IDLE
            return True
        except Exception as e:
            self.error_code = -2
            self.error_message = f"CAN connect failed: {str(e)}"
            return False

    def disconnect(self) -> None:
        if self.bus:
            self.bus.shutdown()
            self.bus = None
        self.connected = False

    def set_velocity(self, v: float, omega: float) -> bool:
        if not self.connected or not self.bus or self.status == AGVStatus.EMERGENCY_STOP:
            return False
        
        try:
            # 速度转换为整数 (单位: mm/s, mrad/s)
            v_int = int(np.clip(v * 1000, -self.max_v * 1000, self.max_v * 1000))
            omega_int = int(np.clip(omega * 1000, -self.max_omega * 1000, self.max_omega * 1000))
            
            # 构造CAN消息 (8字节: v(2) + omega(2) + reserved(4))
            data = v_int.to_bytes(2, byteorder='little', signed=True) + \
                   omega_int.to_bytes(2, byteorder='little', signed=True) + \
                   b'\x00\x00\x00\x00'
            
            msg = can.Message(arbitration_id=self.can_ids["velocity_cmd"], data=data, is_extended_id=False)
            self.bus.send(msg)
            self.status = AGVStatus.RUNNING
            return True
        except Exception as e:
            self.error_code = -3
            self.error_message = f"Send velocity failed: {str(e)}"
            return False

    def update_status(self) -> None:
        if not self.connected or not self.bus:
            return
        
        try:
            # 接收CAN消息 (非阻塞)
            msg = self.bus.recv(timeout=0.001)
            while msg:
                if msg.arbitration_id == self.can_ids["status_report"]:
                    # 状态消息
                    if len(msg.data) >= 8:
                        self.current_x = int.from_bytes(msg.data[0:4], byteorder='little', signed=True) / 1000.0
                        self.current_y = int.from_bytes(msg.data[4:8], byteorder='little', signed=True) / 1000.0
                        if len(msg.data) >= 12:
                            self.current_theta = int.from_bytes(msg.data[8:12], byteorder='little', signed=True) / 1000.0
                        if len(msg.data) >= 14:
                            self.current_v = int.from_bytes(msg.data[12:14], byteorder='little', signed=True) / 1000.0
                        if len(msg.data) >= 16:
                            self.current_omega = int.from_bytes(msg.data[14:16], byteorder='little', signed=True) / 1000.0
                        if len(msg.data) >= 17:
                            self.battery_level = msg.data[16] / 100.0
                        if len(msg.data) >= 18:
                            status_byte = msg.data[17]
                            if status_byte & 0x01:
                                self.status = AGVStatus.ERROR
                            elif status_byte & 0x02:
                                self.status = AGVStatus.EMERGENCY_STOP
                            elif status_byte & 0x04:
                                self.status = AGVStatus.RUNNING
                            else:
                                self.status = AGVStatus.IDLE

                elif msg.arbitration_id == self.can_ids["sensor_report"]:
                    # 传感器数据
                    pass
                msg = self.bus.recv(timeout=0.001)
        except Exception as e:
            self.error_code = -4
            self.error_message = f"Receive status failed: {str(e)}"

    def emergency_stop(self) -> bool:
        if not self.connected or not self.bus:
            return False
        
        try:
            msg = can.Message(arbitration_id=self.can_ids["estop_cmd"], data=b'\x01', is_extended_id=False)
            self.bus.send(msg)
            self.status = AGVStatus.EMERGENCY_STOP
            return True
        except Exception as e:
            self.error_code = -5
            self.error_message = f"Send estop failed: {str(e)}"
            return False

    def reset_error(self) -> bool:
        if not self.connected or not self.bus:
            return False
        
        try:
            msg = can.Message(arbitration_id=self.can_ids["reset_cmd"], data=b'\x01', is_extended_id=False)
            self.bus.send(msg)
            self.status = AGVStatus.IDLE
            self.error_code = 0
            self.error_message = ""
            return True
        except Exception as e:
            self.error_code = -6
            self.error_message = f"Send reset failed: {str(e)}"
            return False


# =============================================================================
# Modbus TCP AGV接口
# =============================================================================

class ModbusAGV(AGVInterface):
    """Modbus TCP接口适配，支持标准Modbus协议"""

    def __init__(
        self,
        agv_type: AGVType = AGVType.DIFFERENTIAL,
        host: str = "192.168.1.100",
        port: int = 502,
        slave_id: int = 1,
        register_map: Dict[str, int] = None
    ):
        super().__init__(agv_type)
        self.host = host
        self.port = port
        self.slave_id = slave_id
        self.client = None
        
        # 寄存器映射
        self.register_map = register_map or {
            "v_cmd": 0x0000,  # 速度指令 (int16, mm/s)
            "omega_cmd": 0x0001,  # 角速度指令 (int16, mrad/s)
            "x_pos": 0x0010,  # X坐标 (int32, mm)
            "y_pos": 0x0012,  # Y坐标 (int32, mm)
            "theta_pos": 0x0014,  # 朝向 (int32, mrad)
            "v_fb": 0x0018,  # 速度反馈 (int16, mm/s)
            "omega_fb": 0x0019,  # 角速度反馈 (int16, mrad/s)
            "battery": 0x0020,  # 电量 (uint16, 0-100%)
            "status": 0x0021,  # 状态字
            "estop_cmd": 0x0030,  # 急停指令
            "reset_cmd": 0x0031  # 复位指令
        }

    def connect(self) -> bool:
        if not MODBUS_AVAILABLE:
            self.error_code = -1
            self.error_message = "pymodbus library not installed"
            return False
        
        try:
            self.client = ModbusTcpClient(host=self.host, port=self.port)
            if self.client.connect():
                self.connected = True
                self.status = AGVStatus.IDLE
                return True
            else:
                self.error_code = -2
                self.error_message = "Modbus TCP connect failed"
                return False
        except Exception as e:
            self.error_code = -3
            self.error_message = f"Modbus connect error: {str(e)}"
            return False

    def disconnect(self) -> None:
        if self.client:
            self.client.close()
            self.client = None
        self.connected = False

    def set_velocity(self, v: float, omega: float) -> bool:
        if not self.connected or not self.client or self.status == AGVStatus.EMERGENCY_STOP:
            return False
        
        try:
            v_int = int(np.clip(v * 1000, -self.max_v * 1000, self.max_v * 1000))
            omega_int = int(np.clip(omega * 1000, -self.max_omega * 1000, self.max_omega * 1000))
            
            # 写入保持寄存器
            self.client.write_register(self.register_map["v_cmd"], v_int, slave=self.slave_id)
            self.client.write_register(self.register_map["omega_cmd"], omega_int, slave=self.slave_id)
            self.status = AGVStatus.RUNNING
            return True
        except Exception as e:
            self.error_code = -4
            self.error_message = f"Write velocity failed: {str(e)}"
            return False

    def update_status(self) -> None:
        if not self.connected or not self.client:
            return
        
        try:
            # 读取状态寄存器组
            response = self.client.read_holding_registers(self.register_map["x_pos"], 10, slave=self.slave_id)
            if not response.isError():
                regs = response.registers
                self.current_x = int.from_bytes(regs[0:2], byteorder='big', signed=True) / 1000.0
                self.current_y = int.from_bytes(regs[2:4], byteorder='big', signed=True) / 1000.0
                self.current_theta = int.from_bytes(regs[4:6], byteorder='big', signed=True) / 1000.0
                self.current_v = int.from_bytes(regs[6:8], byteorder='big', signed=True) / 1000.0
                self.current_omega = int.from_bytes(regs[8:10], byteorder='big', signed=True) / 1000.0

            # 读取电池和状态
            response = self.client.read_holding_registers(self.register_map["battery"], 2, slave=self.slave_id)
            if not response.isError():
                self.battery_level = response.registers[0] / 100.0
                status = response.registers[1]
                if status & 0x0001:
                    self.status = AGVStatus.ERROR
                elif status & 0x0002:
                    self.status = AGVStatus.EMERGENCY_STOP
                elif status & 0x0004:
                    self.status = AGVStatus.RUNNING
                else:
                    self.status = AGVStatus.IDLE
        except Exception as e:
            self.error_code = -5
            self.error_message = f"Read status failed: {str(e)}"

    def emergency_stop(self) -> bool:
        if not self.connected or not self.client:
            return False
        
        try:
            self.client.write_register(self.register_map["estop_cmd"], 0x0001, slave=self.slave_id)
            self.status = AGVStatus.EMERGENCY_STOP
            return True
        except Exception as e:
            self.error_code = -6
            self.error_message = f"Send estop failed: {str(e)}"
            return False

    def reset_error(self) -> bool:
        if not self.connected or not self.client:
            return False
        
        try:
            self.client.write_register(self.register_map["reset_cmd"], 0x0001, slave=self.slave_id)
            self.status = AGVStatus.IDLE
            self.error_code = 0
            self.error_message = ""
            return True
        except Exception as e:
            self.error_code = -7
            self.error_message = f"Send reset failed: {str(e)}"
            return False


# =============================================================================
# AGV接口工厂
# =============================================================================

class AGVInterfaceFactory:
    """AGV接口工厂类，快速创建不同类型的AGV接口"""

    @staticmethod
    def create(interface_type: str, **kwargs) -> AGVInterface:
        """
        创建AGV接口
        
        Args:
            interface_type: 接口类型: "sim", "can", "modbus"
            **kwargs: 接口参数
        
        Returns:
            AGVInterface: AGV接口实例
        """
        if interface_type == "sim":
            return SimulatedAGV(**kwargs)
        elif interface_type == "can":
            return CANAGV(**kwargs)
        elif interface_type == "modbus":
            return ModbusAGV(**kwargs)
        else:
            raise ValueError(f"Unknown AGV interface type: {interface_type}")
