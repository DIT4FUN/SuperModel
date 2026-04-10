"""
IMU传感器模块 (Inertial Measurement Unit)
支持BMI088、MPU9250等常见IMU芯片
包含四元数/欧拉角转换工具
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


# 四元数与欧拉角转换辅助函数
def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """
    四元数转欧拉角

    Args:
        q: 四元数 [w, x, y, z]

    Returns:
        欧拉角 [roll, pitch, yaw] (rad)
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """
    欧拉角转四元数 (ZYX顺序，即 yaw-pitch-roll)

    Args:
        euler: 欧拉角 [roll, pitch, yaw] (rad)

    Returns:
        四元数 [w, x, y, z]
    """
    roll, pitch, yaw = euler[0], euler[1], euler[2]

    # 半角
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    四元数乘法

    Args:
        q1: 四元数 [w, x, y, z]
        q2: 四元数 [w, x, y, z]

    Returns:
        乘积四元数 [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """
    四元数共轭

    Args:
        q: 四元数 [w, x, y, z]

    Returns:
        共轭四元数 [w, -x, -y, -z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """
    归一化四元数

    Args:
        q: 四元数 [w, x, y, z]

    Returns:
        归一化四元数
    """
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])  # 单位四元数
    return q / norm


@dataclass
class IMUData:
    """IMU数据"""
    sensor_id: str
    timestamp: float

    # 加速度 (m/s²)
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # 角速度 (rad/s)
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # 磁场 (μT)
    magnetic_field: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # 欧拉角 (rad) [roll, pitch, yaw]
    euler: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # 四元数 [w, x, y, z]
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    # 温度 (°C)
    temperature: float = 25.0
    # 原始数据
    raw_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    raw_gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    raw_mag: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        for name in ['acceleration', 'angular_velocity', 'magnetic_field',
                     'euler', 'quaternion', 'raw_accel', 'raw_gyro', 'raw_mag']:
            val = getattr(self, name)
            if isinstance(val, list):
                setattr(self, name, np.array(val))

    def to_vector(self) -> np.ndarray:
        """
        返回归一化特征向量
        格式: [ax_norm, ay_norm, az_norm, wx_norm, wy_norm, wz_norm,
               mx_norm, my_norm, mz_norm, roll_norm, pitch_norm, yaw_norm]
        """
        # 加速度归一化 (假设最大 ±24g = ±235.2 m/s²)
        accel_norm = self.acceleration / 235.2
        # 角速度归一化 (假设最大 ±2000°/s = ±34.9 rad/s)
        gyro_norm = self.angular_velocity / 34.9
        # 磁场归一化 (假设最大 ±100 μT)
        mag_norm = self.magnetic_field / 100.0
        # 欧拉角归一化 (±π, ±π/2)
        euler_norm = self.euler / np.array([np.pi, np.pi/2, np.pi])

        return np.concatenate([
            accel_norm, gyro_norm, mag_norm, euler_norm
        ])

    def get_roll(self) -> float:
        """获取翻滚角 (roll)"""
        return self.euler[0]

    def get_pitch(self) -> float:
        """获取俯仰角 (pitch)"""
        return self.euler[1]

    def get_yaw(self) -> float:
        """获取偏航角 (yaw)"""
        return self.euler[2]

    def get_heading(self) -> float:
        """获取航向角 (0-360°)"""
        yaw_deg = np.degrees(self.euler[2])
        if yaw_deg < 0:
            yaw_deg += 360
        return yaw_deg

    def is_data_valid(self) -> bool:
        """检查数据是否有效"""
        # 检查是否有 NaN
        if (np.isnan(self.acceleration).any() or
            np.isnan(self.angular_velocity).any() or
            np.isnan(self.euler).any()):
            return False

        # 检查量程
        if np.linalg.norm(self.acceleration) > 500:  # > 50g
            return False

        return True


class IMUSensor(ABC):
    """IMU传感器基类"""

    def __init__(self, sensor_id: str, name: str = "IMUSensor"):
        self.sensor_id = sensor_id
        self.name = name

        # 校准相关
        self._accel_bias = np.zeros(3)
        self._gyro_bias = np.zeros(3)
        self._mag_bias = np.zeros(3)
        self._is_calibrated = False

        # 方向 (四元数表示)
        self._quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self._euler = np.zeros(3)

        # 采样率
        self._sampling_rate = 1000.0  # Hz

    @abstractmethod
    def read(self, timestamp: float) -> IMUData:
        """读取IMU数据"""
        pass

    def update_orientation(self, gyro_data: np.ndarray, dt: float):
        """
        通过陀螺仪积分更新姿态 (四元数形式)

        Args:
            gyro_data: 角速度 [wx, wy, wz] (rad/s)
            dt: 时间步长 (s)
        """
        # 四元数导数 = 0.5 * q * omega
        # 其中 omega 是纯四元数形式的角速度

        # 归一化陀螺仪数据
        omega = np.array([0, gyro_data[0], gyro_data[1], gyro_data[2]])

        # 四元数乘法形式
        q = self._quaternion
        omega_quat = omega

        # q_dot = 0.5 * q ⊗ omega
        q_dot = 0.5 * quaternion_multiply(q, omega_quat)

        # 积分
        self._quaternion = normalize_quaternion(q + q_dot * dt)

        # 更新欧拉角
        self._euler = quaternion_to_euler(self._quaternion)

    def get_euler_from_quaternion(self, q: np.ndarray) -> np.ndarray:
        """从四元数获取欧拉角"""
        return quaternion_to_euler(q)

    def calibrate_gyro_bias(self, samples: int = 100):
        """
        校准陀螺仪零偏 (假设静止)

        Args:
            samples: 采样次数
        """
        gyro_samples = []
        for _ in range(samples):
            data = self.read(0.0)
            gyro_samples.append(data.angular_velocity.copy())

        self._gyro_bias = np.mean(gyro_samples, axis=0)
        self._is_calibrated = True

    def calibrate_accel_bias(self, samples: int = 100):
        """校准加速度计零偏 (假设静止水平)"""
        accel_samples = []
        for _ in range(samples):
            data = self.read(0.0)
            accel_samples.append(data.acceleration.copy())

        self._accel_bias = np.mean(accel_samples, axis=0)
        # 假设水平放置时 Z 轴应该为 g
        self._accel_bias[2] -= 9.81

    def reset_orientation(self):
        """重置姿态"""
        self._quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self._euler = np.zeros(3)

    def set_sampling_rate(self, rate: float):
        """设置采样率"""
        self._sampling_rate = rate

    def get_quaternion(self) -> np.ndarray:
        """获取当前四元数"""
        return self._quaternion.copy()

    def get_euler(self) -> np.ndarray:
        """获取当前欧拉角"""
        return self._euler.copy()


class BMI088(IMUSensor):
    """
    博世 BMI088 IMU
    广泛应用于AGV、无人机等场景
    特点: 高性能、低噪声、支持高温
    """

    def __init__(
        self,
        sensor_id: str,
        name: str = "BMI088",
        # 加速度计配置
        accel_range: str = "24g",  # ±24g, ±12g, ±6g, ±3g
        accel_bw: int = 3,  # 滤波带宽 0-7
        # 陀螺仪配置
        gyro_range: str = "2000dps",  # °/s
        gyro_bw: int = 3  # 滤波带宽 0-7
    ):
        super().__init__(sensor_id, name)

        # BMI088 规格
        self.accel_ranges = {'3g': 3, '6g': 6, '12g': 12, '24g': 24}
        self.gyro_ranges = {'250dps': 250, '500dps': 500, '1000dps': 1000,
                           '2000dps': 2000}

        self._accel_range = self.accel_ranges.get(accel_range, 24)  # g
        self._gyro_range = self.gyro_ranges.get(gyro_range, 2000)  # °/s

        # 噪声密度 (典型值)
        self._accel_noise_density = 150e-6  # μg/√Hz
        self._gyro_noise_density = 0.008  # °/s/√Hz

        # 模拟内部状态
        self._acceleration = np.zeros(3)
        self._angular_velocity = np.zeros(3)
        self._magnetic_field = np.array([25, 0, 45])  # μT (简化模拟)
        self._temperature = 25.0

    def read(self, timestamp: float) -> IMUData:
        """读取BMI088数据"""
        # 模拟读取 (实际应用中通过 SPI/I2C 读取)
        # 添加噪声
        noise_accel = np.random.normal(0, self._accel_noise_density * 1e-3 * 9.81, 3)
        noise_gyro = np.random.normal(0, self._gyro_noise_density * np.pi / 180, 3)

        self._acceleration += noise_accel
        self._angular_velocity += noise_gyro

        # 去除零偏
        if self._is_calibrated:
            self._acceleration -= self._accel_bias
            self._angular_velocity -= self._gyro_bias

        # 更新姿态
        self._angular_velocity = np.clip(
            self._angular_velocity,
            -np.radians(self._gyro_range),
            np.radians(self._gyro_range)
        )

        # 计算欧拉角 (从加速度计)
        accel_euler = np.zeros(3)
        g = self._acceleration
        if np.linalg.norm(g) > 1e-6:
            accel_euler[0] = np.arctan2(g[1], np.sqrt(g[0]**2 + g[2]**2))  # roll
            accel_euler[1] = np.arctan2(-g[0], g[2])  # pitch
            accel_euler[2] = self._euler[2]  # yaw (磁力计或积分)

        # 如果没有初始化，使用加速度计姿态
        if not self._is_calibrated or np.linalg.norm(self._euler) < 1e-6:
            self._euler = accel_euler
            self._quaternion = euler_to_quaternion(self._euler)

        return IMUData(
            sensor_id=self.sensor_id,
            timestamp=timestamp,
            acceleration=self._acceleration.copy(),
            angular_velocity=self._angular_velocity.copy(),
            magnetic_field=self._magnetic_field.copy(),
            euler=self._euler.copy(),
            quaternion=self._quaternion.copy(),
            temperature=self._temperature,
            raw_accel=self._acceleration.copy(),
            raw_gyro=self._angular_velocity.copy(),
            raw_mag=self._magnetic_field.copy()
        )

    def set_acceleration(self, accel: np.ndarray):
        """设置加速度 (模拟, 用于测试)"""
        self._acceleration = np.array(accel)

    def set_angular_velocity(self, gyro: np.ndarray):
        """设置角速度 (模拟, 用于测试)"""
        self._angular_velocity = np.array(gyro)

    def apply_accel_shock(self, magnitude: float, direction: np.ndarray):
        """施加加速度冲击 (模拟测试)"""
        dir_norm = direction / (np.linalg.norm(direction) + 1e-6)
        self._acceleration += magnitude * dir_norm

    def get_accel_resolution(self) -> float:
        """获取加速度分辨率 (m/s²/LSB)"""
        return 2 * self._accel_range * 9.81 / 65536

    def get_gyro_resolution(self) -> float:
        """获取陀螺仪分辨率 (rad/s/LSB)"""
        return 2 * self._gyro_range * np.pi / 180 / 65536


class MPU9250(IMUSensor):
    """
    MPU9250 9轴IMU
    内置三轴加速度计、三轴陀螺仪、三轴磁力计
    特点: 集成度高、成本低、支持I2C
    """

    def __init__(
        self,
        sensor_id: str,
        name: str = "MPU9250",
        # 加速度计配置
        accel_range: str = "2g",  # ±2g, ±4g, ±8g, ±16g
        # 陀螺仪配置
        gyro_range: str = "250dps",  # °/s
        # 磁力计配置
        mag_resolution: int = 14  # 14-bit
    ):
        super().__init__(sensor_id, name)

        # MPU9250 规格
        self.accel_ranges = {'2g': 2, '4g': 4, '8g': 8, '16g': 16}
        self.gyro_ranges = {'250dps': 250, '500dps': 500, '1000dps': 1000,
                           '2000dps': 2000}

        self._accel_range = self.accel_ranges.get(accel_range, 2)  # g
        self._gyro_range = self.gyro_ranges.get(gyro_range, 250)  # °/s

        # 噪声密度 (典型值)
        self._accel_noise_density = 400e-6  # μg/√Hz
        self._gyro_noise_density = 0.005  # °/s/√Hz

        # 模拟内部状态
        self._acceleration = np.zeros(3)
        self._angular_velocity = np.zeros(3)
        self._magnetic_field = np.array([25, 0, 45])  # μT
        self._temperature = 25.0

        # 磁力计偏置 (需要校准)
        self._mag_hard_iron = np.zeros(3)
        self._mag_soft_iron = np.eye(3)

    def read(self, timestamp: float) -> IMUData:
        """读取MPU9250数据"""
        # 模拟读取
        noise_accel = np.random.normal(0, self._accel_noise_density * 1e-3 * 9.81, 3)
        noise_gyro = np.random.normal(0, self._gyro_noise_density * np.pi / 180, 3)
        noise_mag = np.random.normal(0, 0.1, 3)  # μT

        self._acceleration += noise_accel
        self._angular_velocity += noise_gyro
        self._magnetic_field += noise_mag

        # 去除零偏
        if self._is_calibrated:
            self._acceleration -= self._accel_bias
            self._angular_velocity -= self._gyro_bias

        # 磁力计校准
        mag = self._magnetic_field - self._mag_hard_iron
        mag = self._mag_soft_iron @ mag

        # 限幅
        self._angular_velocity = np.clip(
            self._angular_velocity,
            -np.radians(self._gyro_range),
            np.radians(self._gyro_range)
        )

        # 计算姿态
        accel_euler = np.zeros(3)
        g = self._acceleration
        if np.linalg.norm(g) > 1e-6:
            accel_euler[0] = np.arctan2(g[1], np.sqrt(g[0]**2 + g[2]**2))
            accel_euler[1] = np.arctan2(-g[0], g[2])

        # 航向角从磁力计计算
        heading = np.arctan2(mag[1], mag[0])
        accel_euler[2] = heading

        if not self._is_calibrated or np.linalg.norm(self._euler) < 1e-6:
            self._euler = accel_euler
            self._quaternion = euler_to_quaternion(self._euler)

        return IMUData(
            sensor_id=self.sensor_id,
            timestamp=timestamp,
            acceleration=self._acceleration.copy(),
            angular_velocity=self._angular_velocity.copy(),
            magnetic_field=mag.copy(),
            euler=self._euler.copy(),
            quaternion=self._quaternion.copy(),
            temperature=self._temperature,
            raw_accel=self._acceleration.copy(),
            raw_gyro=self._angular_velocity.copy(),
            raw_mag=self._magnetic_field.copy()
        )

    def set_acceleration(self, accel: np.ndarray):
        """设置加速度 (模拟)"""
        self._acceleration = np.array(accel)

    def set_angular_velocity(self, gyro: np.ndarray):
        """设置角速度 (模拟)"""
        self._angular_velocity = np.array(gyro)

    def set_magnetic_field(self, mag: np.ndarray):
        """设置磁场 (模拟)"""
        self._magnetic_field = np.array(mag)

    def calibrate_magnetometer(self, samples: int = 500):
        """
        校准磁力计 (8字运动)

        简化实现: 计算硬软铁偏移
        """
        mag_samples = []
        for _ in range(samples):
            data = self.read(0.0)
            mag_samples.append(data.magnetic_field.copy())

        mag_samples = np.array(mag_samples)

        # 硬铁偏移 = 均值
        self._mag_hard_iron = np.mean(mag_samples, axis=0)

        # 软铁偏移 = 尺度因子 (简化)
        mag_centered = mag_samples - self._mag_hard_iron
        scale = np.max(np.abs(mag_centered), axis=0)
        self._mag_soft_iron = np.diag(1.0 / (scale + 1e-6))


class IMUArray:
    """
    多IMU管理器
    管理多个IMU传感器，支持冗余和融合
    """

    def __init__(self, name: str = "IMUArray"):
        self.name = name
        self._sensors: Dict[str, IMUSensor] = {}

    def add_sensor(self, sensor: IMUSensor) -> bool:
        """添加IMU传感器"""
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

    def read_all(self, timestamp: float) -> List[IMUData]:
        """读取所有IMU数据"""
        return [sensor.read(timestamp) for sensor in self._sensors.values()]

    def get_fusion_data(self, timestamp: float) -> np.ndarray:
        """
        获取融合后的特征向量

        Returns:
            所有IMU数据拼接的归一化向量
        """
        all_data = self.read_all(timestamp)
        if not all_data:
            return np.array([])

        vectors = [data.to_vector() for data in all_data]
        return np.concatenate(vectors)

    def estimate_pose_change(self, dt: float, timestamp: float = 0.0) -> Dict[str, float]:
        """
        估计姿态变化

        Args:
            dt: 时间步长 (s)
            timestamp: 时间戳

        Returns:
            姿态变化字典
        """
        all_data = self.read_all(timestamp)

        if not all_data:
            return {}

        # 简单平均
        avg_delta_euler = np.zeros(3)
        for data in all_data:
            avg_delta_euler += data.angular_velocity * dt

        avg_delta_euler /= len(all_data)

        return {
            'delta_roll': avg_delta_euler[0],
            'delta_pitch': avg_delta_euler[1],
            'delta_yaw': avg_delta_euler[2]
        }

    def compute_heading(self, timestamp: float = 0.0) -> float:
        """
        计算航向角 (平均)

        Returns:
            航向角 (rad)
        """
        all_data = self.read_all(timestamp)

        if not all_data:
            return 0.0

        heading_sum = 0.0
        for data in all_data:
            heading_sum += data.get_yaw()

        return heading_sum / len(all_data)

    def calibrate_gyro_bias(self, samples: int = 100):
        """校准所有陀螺仪"""
        for sensor in self._sensors.values():
            sensor.calibrate_gyro_bias(samples)

    def get_average_orientation(self, timestamp: float = 0.0) -> np.ndarray:
        """
        获取平均姿态 (四元数平均)

        Returns:
            平均四元数
        """
        all_data = self.read_all(timestamp)

        if not all_data:
            return np.array([1.0, 0.0, 0.0, 0.0])

        # 简单平均欧拉角后转换
        avg_euler = np.zeros(3)
        for data in all_data:
            avg_euler += data.euler

        avg_euler /= len(all_data)
        return euler_to_quaternion(avg_euler)

    def __len__(self) -> int:
        return len(self._sensors)

    def __repr__(self) -> str:
        return f"IMUArray(sensors={list(self._sensors.keys())})"


# =============================================================================
# AGV五级IMU传感器规格表
# =============================================================================

AGV_IMU_GRADES = {
    'S': {
        'name': '小型AGV',
        'type': 'MPU6050',
        'accel_range': 16,    # g (BMI088 equivalent spec)
        'gyro_range': 2000,   # deg/s
        'sample_hz': 100,
        'noise_density_accel': 400,   # ug/sqrt(Hz)
        'noise_density_gyro': 0.05,    # deg/s/sqrt(Hz)
        'magnetometer': False,
        'temperature_comp': False,
        'ahrs': False,
        'typical_use': '轻载仓库AGV，基础姿态检测',
    },
    'M': {
        'name': '中型AGV',
        'type': 'BMI088',
        'accel_range': 24,    # g (BMI088 supported: 3/6/12/24g)
        'gyro_range': 2000,   # deg/s
        'sample_hz': 200,
        'noise_density_accel': 120,   # ug/sqrt(Hz)
        'noise_density_gyro': 0.015,  # deg/s/sqrt(Hz)
        'magnetometer': False,
        'temperature_comp': True,
        'ahrs': True,
        'typical_use': '物流分拣AGV，姿态稳定',
    },
    'L': {
        'name': '大型AGV',
        'type': 'BMI088+Mag',
        'accel_range': 24,    # g (BMI088 supported: 3/6/12/24g)
        'gyro_range': 2000,   # deg/s
        'sample_hz': 500,
        'noise_density_accel': 100,   # ug/sqrt(Hz)
        'noise_density_gyro': 0.010,  # deg/s/sqrt(Hz)
        'magnetometer': True,
        'temperature_comp': True,
        'ahrs': True,
        'typical_use': '产线配送AGV，精密导航',
    },
    'XL': {
        'name': '特大型AGV',
        'type': 'ADIS16470',
        'accel_range': 40,    # g (ADI spec)
        'gyro_range': 4000,   # deg/s
        'sample_hz': 1000,
        'noise_density_accel': 31,    # ug/sqrt(Hz)
        'noise_density_gyro': 0.004,  # deg/s/sqrt(Hz)
        'magnetometer': True,
        'temperature_comp': True,
        'ahrs': True,
        'typical_use': '重载车间AGV，高精度定位',
    },
    'XXL': {
        'name': '超大型AGV',
        'type': 'Dual ADIS16470',
        'accel_range': 40,    # g (ADI spec, dual redundant)
        'gyro_range': 4000,   # deg/s
        'sample_hz': 2000,
        'noise_density_accel': 10,    # ug/sqrt(Hz) with dual fusion
        'noise_density_gyro': 0.002,  # deg/s/sqrt(Hz) with dual fusion
        'magnetometer': True,
        'temperature_comp': True,
        'ahrs': True,
        'typical_use': '港口物流AGV，极限精度要求',
    },
}


def get_imu_spec(grade: str) -> dict:
    """
    获取AGV指定等级的IMU传感器规格

    Args:
        grade: AGV等级 (S/M/L/XL/XXL)

    Returns:
        IMU传感器规格字典
    """
    return AGV_IMU_GRADES.get(grade, AGV_IMU_GRADES['M'])


def create_imu_sensor_for_grade(grade: str, sensor_id: str = "imu_0") -> IMUSensor:
    """
    创建指定AGV等级的IMU传感器

    Args:
        grade: AGV等级 (S/M/L/XL/XXL)
        sensor_id: 传感器ID

    Returns:
        IMUSensor 实例 (BMI088 或 MPU9250)
    """
    spec = get_imu_spec(grade)
    # Map numeric ranges to closest supported string values
    accel_map = {16: '16g', 24: '24g', 40: '24g'}
    gyro_map = {2000: '2000dps', 4000: '2000dps'}
    accel_str = accel_map.get(spec['accel_range'], '16g')
    gyro_str = gyro_map.get(spec['gyro_range'], '2000dps')

    if grade in ['S', 'M', 'L']:
        return BMI088(
            sensor_id=sensor_id,
            accel_range=accel_str,
            gyro_range=gyro_str,
        )
    else:
        # XL/XXL use MPU9250 clamped to max ranges
        return MPU9250(
            sensor_id=sensor_id,
            accel_range='16g',
            gyro_range='2000dps',
        )


def list_imu_capabilities() -> str:
    """列出所有AGV等级的IMU传感器能力"""
    lines = ["AGV五级IMU传感器能力表:"]
    header = f"{'等级':<6} {'型号':<16} {'加速度计范围':<14} {'陀螺仪范围':<14} "
    header += f"{'采样率':<8} {'噪声密度':<16} {'磁力计':<6} {'AHRS':<6} {'典型用途'}"
    lines.append(header)
    lines.append("-" * 115)
    for grade, spec in AGV_IMU_GRADES.items():
        accel_str = f"±{spec['accel_range']}g"
        gyro_str = f"±{spec['gyro_range']}deg/s"
        noise_str = f"{spec['noise_density_accel']}ug/sqrtHz"
        lines.append(
            f"{grade:<6} {spec['type']:<16} {accel_str:<14} {gyro_str:<14} "
            f"{spec['sample_hz']}Hz{'':<3} {noise_str:<16} "
            f"{'Yes' if spec['magnetometer'] else 'No':<6} "
            f"{'Yes' if spec['ahrs'] else 'No':<6} {spec['typical_use']}"
        )
    return "\n".join(lines)
