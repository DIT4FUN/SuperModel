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
六维力/力矩传感器模块
====================

支持六维力传感器
- 三维力 (Fx, Fy, Fz) 测量
- 三维力矩 (Mx, My, Mz) 测量
- 重力补偿
- 温度漂移补偿
- 碰撞检测
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Union
from enum import Enum


class ForceSensorType(Enum):
    """力传感器类型"""
    SIX_AXIS = "six_axis"  # 六维力/力矩
    THREE_AXIS = "three_axis"  # 三维力
    ONE_AXIS = "one_axis"  # 单维力
    LOAD_CELL = "load_cell"  # 称重传感器


@dataclass
class ForceReading:
    """力/力矩读数"""
    fx: float  # X方向力 N
    fy: float  # Y方向力 N
    fz: float  # Z方向力 N
    mx: float  # X方向力矩 N·m
    my: float  # Y方向力矩 N·m
    mz: float  # Z方向力矩 N·m
    temperature: Optional[float] = None  # 温度 °C
    timestamp: float = 0.0
    frame_id: int = 0

    def force_vector(self) -> np.ndarray:
        """获取力向量 [fx, fy, fz]"""
        return np.array([self.fx, self.fy, self.fz])

    def torque_vector(self) -> np.ndarray:
        """获取力矩向量 [mx, my, mz]"""
        return np.array([self.mx, self.my, self.mz])

    def total_force(self) -> float:
        """合力大小"""
        return np.linalg.norm(self.force_vector())

    def total_torque(self) -> float:
        """合力矩大小"""
        return np.linalg.norm(self.torque_vector())


@dataclass
class Wrench:
    """旋量 (力旋量)"""
    force: np.ndarray  # 3D 力向量
    torque: np.ndarray  # 3D 力矩向量
    timestamp: float = 0.0
    frame_id: int = 0
    sensor_id: Optional[str] = None  # 传感器ID

    def to_array(self) -> np.ndarray:
        """转为6D数组 [fx, fy, fz, mx, my, mz]"""
        return np.concatenate([self.force, self.torque])

    def to_vector(self) -> np.ndarray:
        """转为6D向量 (兼容别名)"""
        return self.to_array()

    def transform(self, rotation: np.ndarray, translation: Optional[np.ndarray] = None) -> 'Wrench':
        """变换力旋量到新坐标系

        Args:
            rotation: 3x3 旋转矩阵
            translation: 平移向量 (新坐标系原点在旧坐标系中的位置)

        Returns:
            变换后的 Wrench
        """
        # 力旋量变换: 力只旋转, 力矩需要加上叉乘
        new_force = rotation @ self.force
        if translation is not None:
            new_torque = rotation @ (self.torque - np.cross(translation, self.force))
        else:
            new_torque = rotation @ self.torque
        return Wrench(force=new_force, torque=new_torque)

    @property
    def magnitude(self) -> float:
        """获取力的模长 (兼容旧接口)"""
        return np.linalg.norm(self.force)

    @property
    def torque_magnitude(self) -> float:
        """获取力矩模长 (兼容旧接口)"""
        return np.linalg.norm(self.torque)


class SixAxisForceTorque:
    """
    六维力/力矩传感器

    支持常见的六维力传感器:
    - ATI 系列
    - Robotiq FT 300
    - OnRobot
    - 国产六维力传感器
    """

    def __init__(
        self,
        can_id: Optional[int] = None,
        rs485_address: Optional[int] = None,
        calibration_matrix: Optional[np.ndarray] = None,
        sample_rate: int = 1000,
        gravity_compensation: bool = True,
        sensor_type: Optional[str] = None,  # 向后兼容旧接口
        sensor_id: Optional[str] = None,  # 向后兼容旧接口
    ):
        self._is_streaming = False
        self.calibration = ForceCalibration()

        self.can_id = can_id
        self.rs485_address = rs485_address
        self.sample_rate = sample_rate
        self.gravity_compensation = gravity_compensation
        self.sensor_type = sensor_type  # 兼容参数
        self.sensor_id = sensor_id  # 兼容参数

        # 标定矩阵 (6x6 用于原始数据转换)
        # 原始 -> 实际 = calibration_matrix @ 原始
        if calibration_matrix is None:
            self._calib = np.eye(6)
        else:
            assert calibration_matrix.shape == (6, 6), "Calibration matrix must be 6x6"
            self._calib = calibration_matrix

        # 偏置 (零点)
        self._bias = np.zeros(6, dtype=np.float32)

        # 重力补偿参数
        # 工具重力和重心位置
        self._tool_mass = 0.0  # kg
        self._tool_com = np.array([0.0, 0.0, 0.0])  # 重心在传感器坐标系中的位置

        # 范围限制: [fx_min, fx_max, fy_min, fy_max, fz_min, fz_max, mx_min, mx_max, my_min, my_max, mz_min, mz_max]
        self._force_limits = np.array([
            -500, 500,
            -500, 500,
            -500, 500,
            -50, 50,
            -50, 50,
            -50, 50
        ], dtype=np.float32)  # fx, fy, fz, mx, my, mz
        self._force_threshold = 5.0  # N 力检测阈值

        # 状态
        self._is_opened = False
        self._frame_counter = 0
        self._sim_time = 0.0

        # 低通滤波
        self._alpha = 0.1  # 滤波系数
        self._filtered = np.zeros(6, dtype=np.float32)

    def open(self) -> bool:
        """打开传感器"""
        # 尝试CAN总线接口
        can_opened = False
        if self.can_id is not None:
            try:
                import can
                self._bus = can.Bus(interface='socketcan', channel='can0', bitrate=250000)
                self._use_can = True
                print(f"[SixAxisForceTorque] Opened on CAN bus: ID=0x{self.can_id:02x}, SR={self.sample_rate}")
                can_opened = True
            except (ImportError, Exception):
                self._use_can = False

        # 尝试RS485接口
        if not can_opened and self.rs485_address is not None:
            try:
                import serial
                self._ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
                self._use_rs485 = True
                print(f"[SixAxisForceTorque] Opened on RS485: address={self.rs485_address}, SR={self.sample_rate}")
                can_opened = True
            except (ImportError, Exception):
                self._use_rs485 = False

        if not can_opened:
            # 模拟模式
            self._use_hardware = False
            print(f"[SixAxisForceTorque] Opened in SIMULATION mode: SR={self.sample_rate}")

        # 执行零点标定
        self.calibrate_zero()
        self._is_opened = True
        self._is_streaming = True
        return True

    def close(self):
        """关闭传感器"""
        if self._is_opened:
            if getattr(self, '_use_can', False) and hasattr(self, '_bus'):
                self._bus.shutdown()
            if getattr(self, '_use_rs485', False) and hasattr(self, '_ser'):
                self._ser.close()
            self._is_opened = False
            print("[SixAxisForceTorque] Closed")

    def calibrate_zero(self, samples: int = 1000) -> None:
        """
        零点标定

        在空载状态采集偏置
        """
        if not self._is_opened and not getattr(self, '_use_hardware', True):
            # 未打开时使用零偏置
            self._bias = np.zeros(6, dtype=np.float32)
            return

        print(f"[SixAxisForceTorque] Zero calibration: {samples} samples...")
        biases = np.zeros((samples, 6), dtype=np.float32)
        for i in range(samples):
            raw = self._read_raw()
            biases[i] = raw
        self._bias = np.mean(biases, axis=0)
        if self.calibration is not None:
            self.calibration.bias = self._bias.copy()
        print(f"[SixAxisForceTorque] Zero calibration done: bias={self._bias}")

    def calibrate_bias(self, samples: int = 1000, num_samples: Optional[int] = None) -> None:
        """偏置标定 (别名,兼容旧接口)"""
        if num_samples is not None:
            samples = num_samples
        self.calibrate_zero(samples)

    def set_gravity_compensation(self, mass: float, com: np.ndarray):
        """
        设置重力补偿

        Args:
            mass: 工具质量 (kg)
            com: 重心在传感器坐标系中的位置 (m), [x, y, z]
        """
        self._tool_mass = mass
        self._tool_com = np.array(com, dtype=np.float32)

    def set_tool_center(self, tool_mass: float, tool_com: np.ndarray) -> None:
        """设置工具中心点偏移 (重力补偿)"""
        self.set_gravity_compensation(tool_mass, tool_com)

    def _apply_gravity_compensation(self, reading: np.ndarray) -> np.ndarray:
        """应用重力补偿"""
        if self._tool_mass <= 0 or not self.gravity_compensation:
            return reading

        # 重力在传感器坐标系中 (假设z轴向上)
        g = 9.81  # m/s^2
        gravity_force = np.array([0, 0, -self._tool_mass * g])

        # 重力产生的力矩: r × F
        gravity_torque = np.cross(self._tool_com, gravity_force)

        # 从读数中减去重力影响
        compensated = reading.copy()
        compensated[0:3] -= gravity_force
        compensated[3:6] -= gravity_torque

        return compensated

    def _read_raw(self) -> np.ndarray:
        """读取原始数据 (内部方法)"""
        if getattr(self, '_use_can', False):
            # 这里添加具体的CAN消息解析
            # 不同传感器协议不同
            # 返回模拟值作为示例
            return self._bias + np.random.randn(6) * 0.5
        elif getattr(self, '_use_rs485', False):
            # 这里添加具体的RS485读取
            return self._bias + np.random.randn(6) * 0.5
        else:
            # 模拟模式
            return self._bias + np.random.randn(6) * 0.5

    def read(self) -> ForceReading:
        """读取一帧力/力矩数据"""
        if not self._is_opened:
            raise RuntimeError("Sensor not opened")

        # 读取原始数据
        raw = self._read_raw()

        # 应用标定矩阵和偏置
        calibrated = self._calib @ (raw - self._bias)

        # 重力补偿
        compensated = self._apply_gravity_compensation(calibrated)

        # 范围限制
        compensated = np.clip(
            compensated,
            self._force_limits[::2],
            self._force_limits[1::2]
        )

        # 低通滤波
        if self._frame_counter == 0:
            self._filtered = compensated
        else:
            self._filtered = (1 - self._alpha) * self._filtered + self._alpha * compensated

        fx, fy, fz, mx, my, mz = self._filtered

        # 模拟温度
        temperature = 25.0 + np.random.randn() * 0.5

        dt = 1.0 / self.sample_rate
        frame_id = self._frame_counter
        self._sim_time += dt
        self._frame_counter += 1

        return ForceReading(
            fx=float(fx),
            fy=float(fy),
            fz=float(fz),
            mx=float(mx),
            my=float(my),
            mz=float(mz),
            temperature=temperature,
            timestamp=self._sim_time,
            frame_id=frame_id
        )

    def read_wrench(self) -> Wrench:
        """读取为旋量格式"""
        reading = self.read()
        return Wrench(
            force=reading.force_vector(),
            torque=reading.torque_vector(),
            timestamp=reading.timestamp,
            frame_id=reading.frame_id
        )

    def get_wrench(self) -> Wrench:
        """获取当前力旋量 (兼容别名)"""
        return self.read_wrench()

    def detect_external_force(self, reading: ForceReading, threshold: Optional[float] = None) -> Tuple[bool, np.ndarray]:
        """
        检测是否有外力作用

        Returns:
            has_external: 是否有外力
            external_force: 外力向量
        """
        if threshold is None:
            threshold = self._force_threshold

        force = reading.force_vector()
        magnitude = np.linalg.norm(force)

        return magnitude > threshold, force

    def detect_collision(self, reading: ForceReading, force_threshold: float = 50.0, torque_threshold: float = 10.0) -> bool:
        """
        碰撞检测

        当力或力矩超过阈值时判定为碰撞
        """
        f_mag = reading.total_force()
        m_mag = reading.total_torque()
        return f_mag > force_threshold or m_mag > torque_threshold

    def set_force_limits(self, fx_min: float, fx_max: float,
                       fy_min: float, fy_max: float,
                       fz_min: float, fz_max: float,
                       mx_min: float, mx_max: float,
                       my_min: float, my_max: float,
                       mz_min: float, mz_max: float):
        """设置力/力矩范围限制"""
        self._force_limits = np.array([
            fx_min, fx_max,
            fy_min, fy_max,
            fz_min, fz_max,
            mx_min, mx_max,
            my_min, my_max,
            mz_min, mz_max
        ], dtype=np.float32)

    def get_bias(self) -> np.ndarray:
        """获取当前偏置"""
        return self._bias.copy()

    def detect_contact(self, wrench_or_threshold: Union[Wrench, float] = 5.0, threshold: float = None) -> 'ForceContactDetection':
        """检测是否有外力接触 - 兼容接口返回对象"""
        # 优先使用传入的Wrench
        if isinstance(wrench_or_threshold, Wrench):
            wrench = wrench_or_threshold
        else:
            wrench = self.read_wrench()

        # 确定阈值: keyword threshold优先,否则用位置参数,最后用默认值5.0
        effective_threshold = 5.0
        if threshold is not None:
            effective_threshold = threshold
        elif not isinstance(wrench_or_threshold, Wrench):
            effective_threshold = wrench_or_threshold

        force_magnitude = np.linalg.norm(wrench.force)
        is_contact = bool(force_magnitude > effective_threshold)

        # 返回对象保持向后兼容
        class ForceContactDetection:
            def __init__(self, is_contact, contact_force=0.0):
                self.is_contact = is_contact
                self.contact_force = contact_force

        return ForceContactDetection(is_contact, float(force_magnitude))

    def capture(self) -> Wrench:
        """兼容旧接口 - capture 别名"""
        return self.read_wrench()

    def estimate_payload(self, wrench: Optional[Wrench] = None) -> float:
        """估计负载重量 - 兼容接口"""
        if wrench is None:
            wrench = self.read_wrench()
        # 垂直力估计质量 mass = |fz| / g (取绝对值处理重力方向)
        mass = abs(wrench.force[2]) / 9.81
        return mass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class WheelForceSensor:
    """
    AGV 车轮力传感器

    每个车轮独立测量竖向负载和驱动力
    """

    def __init__(
        self,
        num_wheels: int = 2,
        sample_rate: int = 100
    ):
        self.num_wheels = num_wheels
        self.sample_rate = sample_rate

        # 每个车轮: [垂直负载(N), 驱动力(N)]
        self._biases = np.zeros((num_wheels, 2), dtype=np.float32)
        self._calibration = np.ones((num_wheels, 2), dtype=np.float32)

        self._is_opened = False
        self._frame_counter = 0
        self._sim_time = 0.0

    def open(self) -> bool:
        """打开车轮力传感器"""
        try:
            # 尝试CAN接口
            import can
            self._bus = can.Bus(interface='socketcan', channel='can0', bitrate=250000)
            self._use_can = True
            print(f"[WheelForceSensor] Opened: {self.num_wheels} wheels on CAN")
        except (ImportError, Exception):
            self._use_can = False
            print(f"[WheelForceSensor] Opened in SIMULATION mode: {self.num_wheels} wheels")

        self._is_opened = True
        return True

    def close(self):
        """关闭"""
        if getattr(self, '_use_can', False) and hasattr(self, '_bus'):
            self._bus.shutdown()
        self._is_opened = False
        print("[WheelForceSensor] Closed")

    def calibrate_zero(self):
        """零点标定"""
        # 在空载时标定
        pass

    def read(self) -> np.ndarray:
        """
        读取所有车轮力数据

        Returns:
            forces: [num_wheels, 2] - [垂直负载(N), 驱动力(N)]
        """
        if not self._is_opened:
            raise RuntimeError("Sensor not opened")

        if getattr(self, '_use_can', False):
            # CAN读取
            pass
        else:
            # 模拟: 均匀分布承重 + 噪声
            # 假设AGV总重50kg, 分在两个车轮
            base_load = (35 * 9.81) / self.num_wheels  # 自重 + 半负载
            forces = np.zeros((self.num_wheels, 2), dtype=np.float32)
            forces[:, 0] = base_load + np.random.randn(self.num_wheels) * 2.0
            # 驱动力与运动状态有关, 这里模拟
            if self._frame_counter % 100 < 50:
                forces[:, 1] = 20.0 + np.random.randn(self.num_wheels) * 2.0
            else:
                forces[:, 1] = 0.0 + np.random.randn(self.num_wheels) * 0.5

        dt = 1.0 / self.sample_rate
        self._sim_time += dt
        self._frame_counter += 1

        return (forces - self._biases) * self._calibration

    def get_total_weight(self) -> float:
        """获取AGV总重量 (包括负载)"""
        forces = self.read()
        return np.sum(forces[:, 0]) / 9.81  # N -> kg

    def detect_overload(self, max_load_per_wheel: float) -> Tuple[bool, List[int]]:
        """检测过载"""
        forces = self.read()
        overload_wheels = []
        for i in range(self.num_wheels):
            if forces[i, 0] > max_load_per_wheel * 9.81:
                overload_wheels.append(i)
        return len(overload_wheels) > 0, overload_wheels

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class LiftForceSensor:
    """
    升降机构力传感器

    用于叉举机构负载测量
    检测货物重量和过载保护
    """

    def __init__(
        self,
        can_id: Optional[int] = None,
        max_range: float = 2000.0  # N
    ):
        self.can_id = can_id
        self.max_range = max_range

        self._bias = 0.0
        self._calibration = 1.0
        self._is_opened = False
        self._frame_counter = 0

    def open(self) -> bool:
        self._is_opened = True
        if self.can_id is not None:
            print(f"[LiftForceSensor] Opened on CAN: ID=0x{self.can_id:02x}")
        else:
            print("[LiftForceSensor] Opened in SIMULATION mode")
        return True

    def close(self):
        self._is_opened = False
        print("[LiftForceSensor] Closed")

    def read_force(self) -> float:
        """读取升降机构力 (N)"""
        if not self._is_opened:
            raise RuntimeError("Sensor not opened")

        # 模拟
        base = 0.0 + np.random.randn() * 5.0
        return (base - self._bias) * self._calibration

    def read_weight(self) -> float:
        """读取负载重量 (kg)"""
        force = self.read_force()
        return max(0.0, force / 9.81)  # N -> kg, clamp to non-negative

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# AGV五级力传感器规格
AGV_FORCE_GRADES = {
    'S': {
        'axes': 3,
        'force_range': 100,
        'torque_range': 10,
        'sampling_hz': 100,
        'has_ft': False,
        'has_wheel_force': False,
        'has_lift_force': True,
        'max_lift': 50,  # kg
        'resolution': 0.1,
    },

    'M': {
        'axes': 6,
        'force_range': 200,
        'torque_range': 20,
        'sampling_hz': 500,
        'has_ft': False,
        'has_wheel_force': True,
        'has_lift_force': True,
        'max_lift': 100,  # kg
        'wheel_count': 2,
        'resolution': 0.05,
    },

    'L': {
        'axes': 6,
        'force_range': 500,
        'torque_range': 50,
        'sampling_hz': 1000,
        'has_ft': True,
        'has_wheel_force': True,
        'has_lift_force': True,
        'max_lift': 300,  # kg
        'wheel_count': 4,
        'ft_range': 1000,  # N
        'resolution': 0.02,
    },

    'XL': {
        'axes': 6,
        'force_range': 1000,
        'torque_range': 100,
        'sampling_hz': 2000,
        'has_ft': True,
        'has_wheel_force': True,
        'has_lift_force': True,
        'max_lift': 600,  # kg
        'wheel_count': 4,
        'ft_range': 2000,  # N
        'ft_sample_rate': 500,
        'resolution': 0.01,
    },

    'XXL': {
        'axes': 6,
        'force_range': 5000,
        'torque_range': 500,
        'sampling_hz': 5000,
        'has_ft': True,
        'has_wheel_force': True,
        'has_lift_force': True,
        'max_lift': 1200,  # kg
        'wheel_count': 4,
        'ft_range': 5000,  # N
        'ft_sample_rate': 1000,
        'temperature_compensation': True,
        'resolution': 0.005,
    }
}


def get_force_spec(grade: str) -> dict:
    """获取AGV指定等级的力传感器规格"""
    return AGV_FORCE_GRADES.get(grade, AGV_FORCE_GRADES['M'])


# 兼容旧名称 (用于测试)
class ForceTorqueSensor(SixAxisForceTorque):
    """兼容别名"""
    pass


class WrenchProcessor:
    """力矩处理器 (兼容别名)"""
    def __init__(self, window_size=5, filter_alpha: float = 0.3, outlier_threshold: float = 3.0):
        self.window_size = window_size
        self.filter_alpha = filter_alpha  # EMA滤波 alpha
        self.outlier_threshold = outlier_threshold  # 异常值阈值 (标准化残差)
        self._ema_force = None  # EMA滤波器状态
        self._ema_torque = None
        self._history: List[np.ndarray] = []  # 用于协方差估计

    def filter(self, wrench):
        """滤波处理 (支持Wrench对象或6D向量)"""
        # 统一转为6D数组
        if hasattr(wrench, 'to_array'):
            raw = wrench.to_array()
        else:
            raw = np.asarray(wrench, dtype=np.float64)

        # EMA滤波
        if self._ema_force is None:
            self._ema_force = raw.copy()
        else:
            self._ema_force = self.filter_alpha * raw + (1 - self.filter_alpha) * self._ema_force

        # 异常值检测 (基于标准化残差)
        if self._ema_force is not None and len(self._history) > 5:
            residuals = raw - self._ema_force
            # 用历史协方差标准化
            cov = self._estimate_covariance_internal()
            try:
                inv_cov = np.linalg.inv(cov + 1e-6 * np.eye(6))
                mahalanobis = np.sqrt(max(0, np.dot(residuals, np.dot(inv_cov, residuals))))
                if mahalanobis > self.outlier_threshold:
                    # 异常值,用滤波值替代
                    filtered = self._ema_force.copy()
                else:
                    filtered = self._ema_force.copy()
            except np.linalg.LinAlgError:
                filtered = self._ema_force.copy()
        else:
            filtered = self._ema_force.copy()

        # 更新历史
        self._history.append(raw.copy())
        if len(self._history) > self.window_size * 10:
            self._history.pop(0)

        return filtered

    def _estimate_covariance_internal(self) -> np.ndarray:
        """内部协方差估计"""
        if len(self._history) < 2:
            return np.eye(6)
        arr = np.array(self._history)
        return np.cov(arr, rowvar=False)

    def estimate_covariance(self, history: List[np.ndarray]) -> np.ndarray:
        """估计协方差矩阵"""
        if len(history) < 2:
            return np.eye(6)
        arr = np.array([np.asarray(h, dtype=np.float64).flatten() for h in history])
        return np.cov(arr, rowvar=False)

    def compute_equivalent_wrench_at(self, wrench, translation):
        """计算力旋量在另一点的等效力旋量

        Args:
            wrench: Wrench对象或6D数组 [fx,fy,fz,mx,my,mz]
            translation: 平移向量 [dx,dy,dz] - 新参考点到原参考点的向量

        Returns:
            np.ndarray: 等效力旋量在新参考点 [fx,fy,fz,mx,my,mz]
        """
        if hasattr(wrench, 'to_array'):
            arr = np.asarray(wrench.to_array(), dtype=np.float64).flatten()
        else:
            arr = np.asarray(wrench, dtype=np.float64).flatten()
        force = arr[:3]
        torque = arr[3:]
        # 力旋量变换: F' = R·F, M' = R·(M - r×F)
        rotation = np.eye(3)  # 默认单位旋转
        translation_arr = np.array(translation, dtype=np.float64)
        new_force = rotation @ force
        new_torque = rotation @ (torque - np.cross(translation_arr, force))
        return np.concatenate([new_force, new_torque])


@dataclass
class ForceCalibration:
    """标定 (兼容别名)"""
    bias: np.ndarray = None

    def __init__(self):
        self.bias = np.zeros(6, dtype=np.float32)


class ContactState:
    """接触状态 (兼容别名)"""
    pass


class VirtualForceSensor:
    """虚拟力传感器 (兼容别名)"""
    def __init__(self, sensor_id="virtual", noise_level=None, bias_range=None):
        self.sensor_id = sensor_id
        self.noise_level = noise_level
        self.bias_range = bias_range  # 偏置范围 e.g. 0.1 N
        self._bias = np.zeros(6)
        self._is_opened = False
        if bias_range is not None and bias_range > 0:
            self._bias = np.random.uniform(-bias_range, bias_range, 6)
    
    def open(self):
        self._is_opened = True
        return True

    def close(self):
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def capture(self):
        """兼容capture接口"""
        return self.simulate_contact(np.zeros(3))

    def simulate_collision(self, direction, peak_force, duration_ms=100.0, decay='exponential'):
        """模拟碰撞冲击,返回Wrench帧序列"""
        import math
        frames = []
        direction = np.array(direction, dtype=np.float64)
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        dt_ms = 10.0  # 10ms per frame
        n_steps = max(1, int(duration_ms / dt_ms))
        for i in range(n_steps):
            t_rel = i * dt_ms / 1000.0  # seconds
            if decay == 'exponential':
                factor = math.exp(-10.0 * t_rel)
            else:
                factor = max(0.0, 1.0 - t_rel / (duration_ms / 1000.0 + 1e-9))
            force = direction * peak_force * factor
            torque = np.zeros(3)
            frames.append(Wrench(force=force, torque=torque))
        return frames

    def simulate_contact(self, force_vector=None, torque_vector=None, add_noise=False, force=None, torque=None):
        """模拟接触力,返回Wrench"""
        # 兼容关键字参数
        if force is not None:
            force_vector = force
        if torque is not None:
            torque_vector = torque

        if force_vector is None:
            force_vector = np.zeros(3)
        if torque_vector is None:
            torque_vector = np.zeros(3)

        force = np.array(force_vector, dtype=np.float64)
        torque = np.array(torque_vector, dtype=np.float64)

        if add_noise and self.noise_level is not None:
            force += np.random.randn(3) * self.noise_level
            torque += np.random.randn(3) * (self.noise_level * 0.1)

        # 添加偏置
        force += self._bias[:3]
        torque += self._bias[3:]

        return Wrench(force=force, torque=torque)

    def simulate_payload(self, mass, com_offset=(0.0, 0.0, 0.0)):
        """模拟负载重力 (末端执行器负载)

        Args:
            mass: 负载质量 (kg)
            com_offset: 重心偏移 (x, y, z) in m

        Returns:
            Wrench with gravity force and moment
        """
        g = 9.81
        com = np.array(com_offset, dtype=np.float64)
        force = np.array([0.0, 0.0, -mass * g], dtype=np.float64)
        torque = np.cross(com, force)  # r × F
        return Wrench(force=force, torque=torque)

    def simulate_friction_contact(self, normal_force, velocity, friction_coeff, object_mass=1.0):
        """模拟摩擦接触

        Args:
            normal_force: 法向力 (N)
            velocity: 速度向量 (m/s) [vx, vy, vz]
            friction_coeff: 摩擦系数 (0-1)
            object_mass: 物体质量 (kg)

        Returns:
            Wrench with friction force
        """
        vel = np.array(velocity, dtype=np.float64)
        speed = np.linalg.norm(vel)
        if speed < 1e-9:
            return Wrench(force=np.zeros(3), torque=np.zeros(3))

        # 库伦摩擦: F_f = μ * N * direction
        friction_magnitude = friction_coeff * abs(normal_force)
        friction_force = -friction_magnitude * (vel / speed)

        # 切向力矩 (简化)
        friction_torque = np.zeros(3)

        return Wrench(force=friction_force, torque=friction_torque)

    def simulate_surface_contact(self, surface_normal, contact_point, penetration_depth, stiffness, damping):
        """模拟表面接触力 (弹簧-阻尼模型)"""
        normal = np.array(surface_normal, dtype=np.float64)
        point = np.array(contact_point, dtype=np.float64)

        # 法向 penetration_depth > 0 表示压入
        penetration = max(0.0, penetration_depth)

        # 弹簧阻尼力: F = k*x + b*v (这里简化: 静力学,只用弹簧)
        normal_force_magnitude = stiffness * penetration

        # 考虑阻尼
        if damping > 0:
            normal_force_magnitude += damping * penetration * 0.1

        # 力沿法向 (指向表面内部 = -normal)
        force = -normal_force_magnitude * normal

        # 力矩: r x F (接触点相对于传感器原点的力矩)
        torque = np.cross(point, force)

        return Wrench(force=force, torque=torque)
