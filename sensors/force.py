"""
力觉传感器模块 (Force Sensor Module)
支持六维力传感器、力矩传感器、末端执行器力传感器
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ForceSensorType(Enum):
    """力觉传感器类型"""
    SINGLE_AXIS = "single_axis"       # 单轴力传感器
    THREE_AXIS = "three_axis"         # 三轴力传感器
    SIX_AXIS = "six_axis"             # 六维力传感器 (Fx, Fy, Fz, Mx, My, Mz)
    TORQUE = "torque"                 # 力矩传感器
    FT_SENSOR = "ft_sensor"           # ATI力觉传感器


@dataclass
class ForceData:
    """力觉数据"""
    timestamp: float
    sensor_id: str
    sensor_type: ForceSensorType
    # 力 (N)
    force: Optional[np.ndarray] = None        # [Fx, Fy, Fz] or [Fx]
    # 力矩 (Nm)
    torque: Optional[np.ndarray] = None      # [Mx, My, Mz] or [Mx, My, Mz]
    # 完整六维: [Fx, Fy, Fz, Mx, My, Mz]
    wrench: Optional[np.ndarray] = None
    # 原始数据
    raw: Optional[np.ndarray] = None
    # 信号质量 (0-1)
    quality: float = 1.0

    def __post_init__(self):
        if self.wrench is None and self.force is not None and self.torque is not None:
            if len(self.force) == 3 and len(self.torque) == 3:
                self.wrench = np.concatenate([self.force, self.torque])

    def to_vector(self) -> np.ndarray:
        """转换为特征向量"""
        if self.wrench is not None:
            return self.wrench
        vec = []
        if self.force is not None:
            vec.extend(self.force)
        if self.torque is not None:
            vec.extend(self.torque)
        return np.array(vec) if vec else np.array([0.0] * 6)

    def get_magnitude(self) -> float:
        """获取力的大小"""
        if self.force is not None:
            return float(np.linalg.norm(self.force))
        if self.wrench is not None:
            return float(np.linalg.norm(self.wrench[:3]))
        return 0.0

    def get_torque_magnitude(self) -> float:
        """获取力矩大小"""
        if self.torque is not None:
            return float(np.linalg.norm(self.torque))
        if self.wrench is not None:
            return float(np.linalg.norm(self.wrench[3:6]))
        return 0.0

    def is_safe(self, force_threshold: float = 50.0, torque_threshold: float = 10.0) -> bool:
        """判断是否在安全范围内"""
        return self.get_magnitude() < force_threshold and self.get_torque_magnitude() < torque_threshold


class ForceSensor:
    """力觉传感器基类"""

    def __init__(self, sensor_id: str, sensor_type: ForceSensorType, config: Optional[Dict] = None):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.config = config or {}
        self.calibration_matrix = np.eye(6)
        self.bias = np.zeros(6)
        self.is_calibrated = False
        self._last_reading: Optional[ForceData] = None

    def read(self, timestamp: Optional[float] = None) -> ForceData:
        """读取力觉数据"""
        raise NotImplementedError

    def set_bias(self, current_reading: ForceData):
        """设置零点偏移"""
        self.bias = current_reading.to_vector()
        self.is_calibrated = True

    def apply_calibration(self, raw_data: np.ndarray) -> np.ndarray:
        """应用校准和偏移"""
        calibrated = self.calibration_matrix @ raw_data - self.bias
        return calibrated


class SixAxisFTSensor(ForceSensor):
    """六维力传感器 (ATI风格)"""

    # 典型量程配置
    FORCE_RANGES = {
        "mini40": {"fx": 40, "fy": 40, "fz": 120, "tx": 2, "ty": 2, "tz": 2},
        "nano17": {"fx": 17, "fy": 17, "fz": 17, "tx": 1, "ty": 1, "tz": 1},
        "gamma": {"fx": 65, "fy": 65, "fz": 195, "tx": 5, "ty": 5, "tz": 5},
    }

    def __init__(self, sensor_id: str, model: str = "mini40", config: Optional[Dict] = None):
        super().__init__(sensor_id, ForceSensorType.SIX_AXIS, config)
        self.model = model
        self.ranges = self.FORCE_RANGES.get(model, self.FORCE_RANGES["mini40"])
        self.noise_levels = {k: v * 0.002 for k, v in self.ranges.items()}  # 0.2% F.S.
        self._filter_alpha = config.get("filter_alpha", 0.3)  # 低通滤波系数
        self._filtered_wrench = None

    def read(self, timestamp: Optional[float] = None) -> ForceData:
        """读取六维力数据"""
        # 模拟传感器数据
        raw = np.array([
            np.random.normal(0, self.noise_levels["fx"]),
            np.random.normal(0, self.noise_levels["fy"]),
            np.random.normal(0, self.noise_levels["fz"]),
            np.random.normal(0, self.noise_levels["tx"]),
            np.random.normal(0, self.noise_levels["ty"]),
            np.random.normal(0, self.noise_levels["tz"]),
        ])

        # 模拟一些接触力
        if np.random.rand() < 0.3:
            raw[0] += np.random.uniform(-5, 5)
            raw[2] += np.random.uniform(-20, -5)  # 通常Z轴承重

        # 低通滤波
        if self._filtered_wrench is None:
            self._filtered_wrench = raw
        else:
            self._filtered_wrench = (
                self._filter_alpha * raw +
                (1 - self._filter_alpha) * self._filtered_wrench
            )

        # 应用校准
        wrench = self.apply_calibration(self._filtered_wrench) if self.is_calibrated else self._filtered_wrench

        self._last_reading = ForceData(
            timestamp=timestamp or np.datetime64('now').astype(float) / 1e9,
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            wrench=wrench,
            force=wrench[:3],
            torque=wrench[3:6],
            raw=raw,
            quality=0.95 + np.random.uniform(0, 0.05)
        )
        return self._last_reading

    def compute_tcp_wrench(self, tcp_offset: np.ndarray) -> np.ndarray:
        """计算TCP坐标系下的等效六维力"""
        if self._last_reading is None or self._last_reading.wrench is None:
            return np.zeros(6)

        wrench = self._last_reading.wrench
        # 力矩补偿: M' = M + F × offset
        r = tcp_offset[:3]
        force = wrench[:3]
        torque_comp = wrench[3:6] + np.cross(r, force)
        return np.concatenate([force, torque_comp])


class SingleAxisForceSensor(ForceSensor):
    """单轴力传感器"""

    def __init__(self, sensor_id: str, axis: str = "z", range_n: float = 100.0, config: Optional[Dict] = None):
        super().__init__(sensor_id, ForceSensorType.SINGLE_AXIS, config)
        self.axis = axis
        self.range_n = range_n
        self.noise_std = range_n * 0.001

    def read(self, timestamp: Optional[float] = None) -> ForceData:
        """读取单轴力"""
        axis_idx = {"x": 0, "y": 1, "z": 2}[self.axis]
        raw = np.random.normal(0, self.noise_std)
        force = np.array([0.0, 0.0, 0.0])
        force[axis_idx] = raw

        self._last_reading = ForceData(
            timestamp=timestamp or np.datetime64('now').astype(float) / 1e9,
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            force=force,
            raw=np.array([raw])
        )
        return self._last_reading


class ForceSensorArray:
    """多力觉传感器管理"""

    def __init__(self):
        self.sensors: Dict[str, ForceSensor] = {}
        self._sampling_period = 0.001  # 1ms

    def add_sensor(self, sensor: ForceSensor):
        """添加传感器"""
        self.sensors[sensor.sensor_id] = sensor

    def read_all(self, timestamp: Optional[float] = None) -> List[ForceData]:
        """读取所有力觉数据"""
        return [sensor.read(timestamp) for sensor in self.sensors.values()]

    def get_net_wrench(self) -> np.ndarray:
        """获取合成六维力"""
        all_wrenches = []
        for data in self.read_all():
            if data.wrench is not None:
                all_wrenches.append(data.wrench)
        if all_wrenches:
            return np.mean(all_wrenches, axis=0)
        return np.zeros(6)

    def check_safety(self, force_thresh: float = 50.0, torque_thresh: float = 10.0) -> Dict:
        """安全检测"""
        results = {"safe": True, "warnings": []}
        for sensor_id, sensor in self.sensors.items():
            if sensor._last_reading:
                if not sensor._last_reading.is_safe(force_thresh, torque_thresh):
                    results["safe"] = False
                    results["warnings"].append(f"{sensor_id}: overload detected")
        return results

    def compute_grasp_force(self, contact_points: List[np.ndarray]) -> np.ndarray:
        """计算机器人抓取力 (简化模型)"""
        all_forces = []
        for data in self.read_all():
            if data.force is not None:
                all_forces.append(np.linalg.norm(data.force))

        if not all_forces:
            return np.array([0.0])

        # 简化：取最大力作为抓取力
        return np.array([max(all_forces)])

    def detect_contact(self, threshold: float = 2.0) -> bool:
        """检测是否有力接触"""
        for sensor in self.sensors.values():
            if sensor._last_reading:
                if sensor._last_reading.get_magnitude() > threshold:
                    return True
        return False
