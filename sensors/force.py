"""
力觉传感器模块 (Force/Torque Sensors)
支持六维力传感器、单轴力传感器、TCP力矩计算
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class ForceData:
    """力觉数据"""
    sensor_id: str
    timestamp: float
    # 六维力/力矩向量 [Fx, Fy, Fz, Mx, My, Mz]
    # 力单位: N (牛顿)
    # 力矩单位: Nm (牛顿米)
    wrench: np.ndarray = field(default_factory=lambda: np.zeros(6))
    # 各轴原始值 (可能用于诊断)
    raw_forces: np.ndarray = field(default_factory=lambda: np.zeros(3))
    raw_torques: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # 温度 (用于温漂补偿)
    temperature: float = 25.0  # °C
    # 工作状态
    is_saturated: bool = False
    # 原始ADC值
    raw_adc: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        if isinstance(self.wrench, list):
            self.wrench = np.array(self.wrench)
        if isinstance(self.raw_forces, list):
            self.raw_forces = np.array(self.raw_forces)
        if isinstance(self.raw_torques, list):
            self.raw_torques = np.array(self.raw_torques)
        if isinstance(self.raw_adc, list):
            self.raw_adc = np.array(self.raw_adc)

    def to_vector(self) -> np.ndarray:
        """
        返回归一化特征向量
        格式: [Fx_norm, Fy_norm, Fz_norm, Mx_norm, My_norm, Mz_norm, temp_norm]
        """
        # 假设标准量程 (可根据传感器型号调整)
        F_MAX = 120.0  # N
        M_MAX = 2.0    # Nm

        features = [
            np.clip(self.wrench[0] / F_MAX, -1, 1),  # Fx
            np.clip(self.wrench[1] / F_MAX, -1, 1),  # Fy
            np.clip(self.wrench[2] / F_MAX, -1, 1),  # Fz
            np.clip(self.wrench[3] / M_MAX, -1, 1),  # Mx
            np.clip(self.wrench[4] / M_MAX, -1, 1),  # My
            np.clip(self.wrench[5] / M_MAX, -1, 1),  # Mz
            np.clip(self.temperature / 50.0, 0, 1)   # temp
        ]

        return np.array(features)

    def get_force_magnitude(self) -> float:
        """获取合力大小"""
        return np.linalg.norm(self.wrench[:3])

    def get_torque_magnitude(self) -> float:
        """获取合力矩大小"""
        return np.linalg.norm(self.wrench[3:])

    def is_valid(self, force_threshold: float = 200.0, torque_threshold: float = 10.0) -> bool:
        """检查数据是否有效 (未饱和)"""
        if self.is_saturated:
            return False
        if self.get_force_magnitude() > force_threshold:
            return False
        if self.get_torque_magnitude() > torque_threshold:
            return False
        return True

    def get_contact_point_estimate(self, calibration_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        估算接触点位置 (需要标定矩阵)

        Returns:
            接触点坐标 [x, y, z] (相对于传感器坐标系)
        """
        # 简化估算: 基于力矩和力计算接触偏移
        # M = r × F  =>  r ≈ M / F (简化)
        F = self.wrench[:3]
        M = self.wrench[3:]

        if np.linalg.norm(F) < 1e-6:
            return np.zeros(3)

        # 接触点估算 (需要标定矩阵进行精确计算)
        # 这里返回简单估算
        contact = np.cross(M, F) / (np.dot(F, F) + 1e-6)
        return contact


class ForceSensor(ABC):
    """力觉传感器基类"""

    def __init__(self, sensor_id: str, name: str = "ForceSensor"):
        self.sensor_id = sensor_id
        self.name = name

        # 校准相关
        self._bias = np.zeros(6)
        self._calibration_matrix = np.eye(6)
        self._is_bias_set = False

        # 量程
        self._force_range = np.array([100, 100, 100, 2, 2, 2])  # [N, N, N, Nm, Nm, Nm]

    @abstractmethod
    def read(self, timestamp: float) -> ForceData:
        """读取传感器数据"""
        pass

    def set_bias(self, current_reading: ForceData):
        """
        设置零偏 (当前读数作为零点)

        Args:
            current_reading: 当前 ForceData
        """
        self._bias = current_reading.wrench.copy()
        self._is_bias_set = True

    def apply_calibration(self, raw_data: np.ndarray) -> np.ndarray:
        """
        应用校准矩阵

        Args:
            raw_data: 原始数据 [Fx, Fy, Fz, Mx, My, Mz]

        Returns:
            校准后数据
        """
        # 去除零偏
        biased = raw_data - self._bias
        # 应用校准矩阵
        calibrated = self._calibration_matrix @ biased
        return calibrated

    def remove_bias(self) -> bool:
        """移除零偏"""
        if self._is_bias_set:
            self._bias.fill(0)
            self._is_bias_set = False
            return True
        return False

    def get_force_range(self) -> np.ndarray:
        """获取量程"""
        return self._force_range.copy()

    def set_calibration_matrix(self, matrix: np.ndarray):
        """设置校准矩阵"""
        if matrix.shape == (6, 6):
            self._calibration_matrix = matrix.copy()


class SixAxisFTSensor(ForceSensor):
    """
    六维力传感器 (ATI风格)
    典型型号: mini40, Gamma, SI-120-2.3
    """

    def __init__(
        self,
        sensor_id: str,
        name: str = "SixAxisFTSensor",
        model: str = "mini40",
        # 量程 (可根据型号调整)
        force_range: Tuple[float, float, float] = (120, 120, 120),  # N
        torque_range: Tuple[float, float, float] = (2, 2, 2)  # Nm
    ):
        super().__init__(sensor_id, name)
        self.model = model

        # 设置量程
        self._force_range = np.array([*force_range, *torque_range])

        # ATI mini40 典型规格
        if model.lower() == "mini40":
            self._force_range = np.array([120, 120, 120, 2, 2, 2])
        elif model.lower() == "gamma":
            self._force_range = np.array([200, 200, 200, 10, 10, 10])
        elif model.lower() == "si-120":
            self._force_range = np.array([120, 120, 120, 12, 12, 12])

        # 模拟内部状态
        self._current_wrench = np.zeros(6)
        self._temperature = 25.0

        # 标定矩阵 (6x6)
        # 实际产品出厂时提供
        self._calibration_matrix = np.eye(6)

        # 温度补偿系数
        self._temp_coefficient = np.zeros(6)  # N/°C, Nm/°C

    def read(self, timestamp: float) -> ForceData:
        """读取六维力数据"""
        # 模拟读取 (实际应用中从 EtherCAT/RS485/USB 读取)
        # 添加噪声
        noise = np.random.normal(0, 0.1, 6)  # 模拟噪声
        self._current_wrench += noise

        # 去除零偏
        if self._is_bias_set:
            self._current_wrench = self._current_wrench - self._bias

        # 应用校准矩阵
        calibrated = self._calibration_matrix @ self._current_wrench

        # 温度补偿 (简化)
        calibrated += self._temp_coefficient * (self._temperature - 25.0)

        # 检测饱和
        is_saturated = np.any(np.abs(calibrated) > self._force_range * 0.98)

        return ForceData(
            sensor_id=self.sensor_id,
            timestamp=timestamp,
            wrench=calibrated.copy(),
            raw_forces=calibrated[:3].copy(),
            raw_torques=calibrated[3:].copy(),
            temperature=self._temperature,
            is_saturated=is_saturated,
            raw_adc=self._current_wrench.copy()
        )

    def set_wrench(self, wrench: np.ndarray):
        """设置力值 (模拟, 用于测试)"""
        if len(wrench) == 6:
            self._current_wrench = np.array(wrench)

    def compute_tcp_wrench(self, tcp_offset: np.ndarray) -> np.ndarray:
        """
        计算TCP处的等效六维力

        Args:
            tcp_offset: TCP相对于力传感器坐标系的偏移 [x, y, z] (m)

        Returns:
            TCP处的六维力 [Fx, Fy, Fz, Mx, My, Mz]
        """
        # 当前读取的力和力矩
        F = self._current_wrench[:3]
        M = self._current_wrench[3:]

        # 力矩补偿: M_tcp = M - F × r
        # 其中 r 是 TCP 偏移向量
        M_tcp = M - np.cross(F, tcp_offset)

        # TCP处力保持不变
        F_tcp = F

        return np.concatenate([F_tcp, M_tcp])

    def apply_tcp_offset(self, wrench: np.ndarray, tcp_offset: np.ndarray) -> np.ndarray:
        """
        将TCP处的力转换到传感器坐标系

        Args:
            wrench: TCP处的六维力 [Fx, Fy, Fz, Mx, My, Mz]
            tcp_offset: TCP相对于力传感器坐标系的偏移 [x, y, z] (m)

        Returns:
            传感器坐标系下的六维力
        """
        F = wrench[:3]
        M_tcp = wrench[3:]

        # 传感器坐标系下的力矩: M = M_tcp + F × r
        M = M_tcp + np.cross(F, tcp_offset)

        return np.concatenate([F, M])

    def get_stiffness_matrix(self) -> np.ndarray:
        """
        获取刚度矩阵 (用于力控)

        返回:
            6x6 刚度矩阵
        """
        # 默认刚度 (实际根据传感器型号)
        # k[N/m], kt[Nm/rad]
        k = 1e6  # N/m (简化)
        kt = 100  # Nm/rad

        stiffness = np.diag([k, k, k, kt, kt, kt])
        return stiffness

    def set_temperature(self, temp: float):
        """设置温度 (用于温漂模拟)"""
        self._temperature = temp


class SingleAxisForceSensor(ForceSensor):
    """
    单轴力传感器
    适用于: 压力检测、负载测量、张力测量
    """

    def __init__(
        self,
        sensor_id: str,
        name: str = "SingleAxisForceSensor",
        axis: str = "z",  # x, y, z
        force_range: float = 100.0,  # N
        accuracy: float = 0.001  # % F.S.
    ):
        super().__init__(sensor_id, name)
        self.axis = axis
        self._force_range = np.array([force_range])
        self.accuracy = accuracy

        # 模拟内部状态
        self._current_force = 0.0
        self._temperature = 25.0

        # 灵敏度 (mV/V 或 count/N)
        self._sensitivity = 1.0

    def read(self, timestamp: float) -> ForceData:
        """读取单轴力数据"""
        # 模拟读取
        noise = np.random.normal(0, self._force_range[0] * self.accuracy)
        self._current_force += noise

        # 去除零偏
        if self._is_bias_set:
            self._current_force = self._current_force - self._bias[0]

        # 检测饱和
        is_saturated = abs(self._current_force) > self._force_range[0] * 0.98

        # 扩展为六维力格式
        wrench = np.zeros(6)
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[self.axis]
        wrench[axis_idx] = self._current_force

        return ForceData(
            sensor_id=self.sensor_id,
            timestamp=timestamp,
            wrench=wrench,
            raw_forces=np.array([self._current_force if self.axis == 'x' else 0,
                                 self._current_force if self.axis == 'y' else 0,
                                 self._current_force if self.axis == 'z' else 0]),
            raw_torques=np.zeros(3),
            temperature=self._temperature,
            is_saturated=is_saturated,
            raw_adc=np.array([self._current_force])
        )

    def set_force(self, force: float):
        """设置力值 (模拟, 用于测试)"""
        self._current_force = np.clip(force, -self._force_range[0], self._force_range[0])

    def get_axial_force(self) -> float:
        """获取轴向力"""
        return self._current_force


class ForceSensorArray:
    """
    多力觉传感器管理器
    管理多个六维力传感器或单轴力传感器
    """

    def __init__(self, name: str = "ForceSensorArray"):
        self.name = name
        self._sensors: Dict[str, ForceSensor] = {}

    def add_sensor(self, sensor: ForceSensor) -> bool:
        """
        添加传感器

        Args:
            sensor: ForceSensor 实例

        Returns:
            添加是否成功
        """
        if sensor.sensor_id in self._sensors:
            return False
        self._sensors[sensor.sensor_id] = sensor
        return True

    def remove_sensor(self, sensor_id: str) -> bool:
        """移除传感器"""
        if sensor_id in self._sensors:
            del self._sensors[sensor_id]
            return True
        return False

    def read_all(self, timestamp: float) -> List[ForceData]:
        """读取所有传感器数据"""
        return [sensor.read(timestamp) for sensor in self._sensors.values()]

    def get_net_wrench(self, timestamp: float) -> np.ndarray:
        """
        获取合成六维力 (所有传感器力/力矩矢量和)

        Returns:
            合成六维力 [Fx, Fy, Fz, Mx, My, Mz]
        """
        all_data = self.read_all(timestamp)
        net_wrench = np.zeros(6)

        for data in all_data:
            net_wrench += data.wrench

        return net_wrench

    def check_safety(
        self,
        force_threshold: float = 100.0,
        torque_threshold: float = 2.0,
        timestamp: float = 0.0
    ) -> Dict[str, any]:
        """
        安全检查

        Args:
            force_threshold: 力阈值 (N)
            torque_threshold: 力矩阈值 (Nm)
            timestamp: 时间戳

        Returns:
            安全状态字典
        """
        all_data = self.read_all(timestamp)

        safety_status = {
            'is_safe': True,
            'warnings': [],
            'violations': [],
            'sensor_readings': {}
        }

        for data in all_data:
            sensor_id = data.sensor_id
            safety_status['sensor_readings'][sensor_id] = {
                'force_mag': data.get_force_magnitude(),
                'torque_mag': data.get_torque_magnitude(),
                'is_saturated': data.is_saturated
            }

            # 检查饱和
            if data.is_saturated:
                safety_status['warnings'].append(
                    f"{sensor_id}: 传感器饱和"
                )

            # 检查阈值
            if data.get_force_magnitude() > force_threshold:
                safety_status['is_safe'] = False
                safety_status['violations'].append(
                    f"{sensor_id}: 力 {data.get_force_magnitude():.2f}N 超过阈值 {force_threshold}N"
                )

            if data.get_torque_magnitude() > torque_threshold:
                safety_status['is_safe'] = False
                safety_status['violations'].append(
                    f"{sensor_id}: 力矩 {data.get_torque_magnitude():.3f}Nm 超过阈值 {torque_threshold}Nm"
                )

        return safety_status

    def detect_contact(self, threshold: float = 1.0, timestamp: float = 0.0) -> bool:
        """
        检测接触 (任意传感器力超过阈值)

        Args:
            threshold: 力阈值 (N)
            timestamp: 时间戳

        Returns:
            是否检测到接触
        """
        all_data = self.read_all(timestamp)

        for data in all_data:
            if data.get_force_magnitude() > threshold:
                return True

        return False

    def get_fusion_data(self, timestamp: float) -> np.ndarray:
        """
        获取融合后的特征向量

        Returns:
            归一化特征向量
        """
        all_data = self.read_all(timestamp)

        if not all_data:
            return np.array([])

        # 拼接所有传感器特征向量
        vectors = [data.to_vector() for data in all_data]
        return np.concatenate(vectors)

    def bias_all_sensors(self):
        """对所有传感器设置当前值为零偏"""
        timestamp = 0.0
        for sensor in self._sensors.values():
            current_data = sensor.read(timestamp)
            sensor.set_bias(current_data)

    def remove_all_bias(self):
        """移除所有传感器零偏"""
        for sensor in self._sensors.values():
            sensor.remove_bias()

    def __len__(self) -> int:
        return len(self._sensors)

    def __repr__(self) -> str:
        return f"ForceSensorArray(sensors={list(self._sensors.keys())})"
