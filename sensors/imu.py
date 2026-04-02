"""
IMU传感器模块 (Inertial Measurement Unit)
支持加速度计、陀螺仪、磁力计的融合输出
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class IMUModel(Enum):
    """IMU型号"""
    BMI088 = "bmi088"           # 博世BMI088 (AGV常用)
    MPU6050 = "mpu6050"         # MPU6050 (入门级)
    MPU9250 = "mpu9250"         # MPU9250 (9轴)
    ADIS16465 = "adis16465"     # ADIS16465 (高精度工业级)
    XSENS_MTI = "xsens_mti"     # XSens MTI (室外AGV)


@dataclass
class IMUData:
    """IMU数据"""
    timestamp: float
    sensor_id: str
    model: IMUModel
    # 加速度 (m/s²) [ax, ay, az]
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # 角速度 (rad/s) [wx, wy, wz]
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # 磁场 (μT) [mx, my, mz]
    magnetic_field: Optional[np.ndarray] = None
    # 温度 (°C)
    temperature: float = 25.0
    # 欧拉角 (rad) [roll, pitch, yaw]
    euler: Optional[np.ndarray] = None
    # 四元数 [w, x, y, z]
    quaternion: Optional[np.ndarray] = None
    # 原始数据
    raw_accel: Optional[np.ndarray] = None
    raw_gyro: Optional[np.ndarray] = None
    # 信号质量指标
    signal_quality: Dict[str, float] = field(default_factory=lambda: {
        "accel": 1.0, "gyro": 1.0, "mag": 1.0
    })

    def to_vector(self) -> np.ndarray:
        """转换为特征向量"""
        vec = list(self.acceleration) + list(self.angular_velocity)
        if self.magnetic_field is not None:
            vec.extend(self.magnetic_field)
        return np.array(vec)

    def get_imu_pose_change(self, dt: float) -> Dict[str, float]:
        """从IMU数据计算姿态变化"""
        delta_angle = self.angular_velocity * dt
        delta_vel = self.acceleration * dt
        return {
            "delta_angle_x": delta_angle[0],
            "delta_angle_y": delta_angle[1],
            "delta_angle_z": delta_angle[2],
            "delta_vel_x": delta_vel[0],
            "delta_vel_y": delta_vel[1],
            "delta_vel_z": delta_vel[2],
        }


class IMUSensor:
    """IMU传感器基类"""

    GRAVITY = 9.81

    def __init__(self, sensor_id: str, model: IMUModel, config: Optional[Dict] = None):
        self.sensor_id = sensor_id
        self.model = model
        self.config = config or {}
        self._orientation = np.array([1.0, 0.0, 0.0, 0.0])  # 四元数 [w, x, y, z]
        self._last_timestamp: Optional[float] = None
        self._sample_count = 0
        self._drift_bias_accel = np.zeros(3)
        self._drift_bias_gyro = np.zeros(3)

    def read(self, timestamp: Optional[float] = None) -> IMUData:
        """读取IMU数据"""
        raise NotImplementedError

    def update_orientation(self, gyro_data: np.ndarray, dt: float):
        """更新四元数姿态 (简单积分)"""
        wx, wy, wz = gyro_data
        # 四元数导数
        q = self._orientation
        q_dot = 0.5 * np.array([
            -q[1]*wx - q[2]*wy - q[3]*wz,
             q[0]*wx + q[2]*wz - q[3]*wy,
             q[0]*wy - q[1]*wz + q[3]*wx,
             q[0]*wz + q[1]*wy - q[2]*wx
        ])
        self._orientation = q + q_dot * dt
        self._orientation /= np.linalg.norm(self._orientation)

    def get_euler_from_quaternion(self, q: np.ndarray) -> np.ndarray:
        """从四元数转欧拉角"""
        w, x, y, z = q
        # Roll (X轴旋转)
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        # Pitch (Y轴旋转)
        sinp = 2*(w*y - z*x)
        sinp = np.clip(sinp, -1, 1)
        pitch = np.arcsin(sinp)
        # Yaw (Z轴旋转)
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return np.array([roll, pitch, yaw])


class BMI088(IMUSensor):
    """博世BMI088 IMU (AGV常用)"""

    # 典型噪声密度
    NOISE_DENSITY_ACCEL = 150e-6  # μg/√Hz
    NOISE_DENSITY_GYRO = 3e-6     # °/s/√Hz

    def __init__(self, sensor_id: str, config: Optional[Dict] = None):
        super().__init__(sensor_id, IMUModel.BMI088, config)
        self.accel_range = config.get("accel_range", 24)  # ±24g
        self.gyro_range = config.get("gyro_range", 2000)  # ±2000°/s
        self._filter_bandwidth = config.get("filter_bw", 32)  # Hz

    def read(self, timestamp: Optional[float] = None) -> IMUData:
        """读取BMI088数据"""
        ts = timestamp or np.datetime64('now').astype(float) / 1e9

        # 模拟加速度计数据 (静止时应为[0, 0, g])
        accel_noise_std = self.NOISE_DENSITY_ACCEL * 1e-6 * self.GRAVITY * np.sqrt(100)
        accel = np.array([
            np.random.normal(0, accel_noise_std),
            np.random.normal(0, accel_noise_std),
            np.random.normal(-self.GRAVITY, accel_noise_std)
        ]) - self._drift_bias_accel

        # 模拟陀螺仪数据
        gyro_noise_std = self.NOISE_DENSITY_GYRO * np.pi / 180 * np.sqrt(100)
        gyro = np.array([
            np.random.normal(0, gyro_noise_std),
            np.random.normal(0, gyro_noise_std),
            np.random.normal(0, gyro_noise_std)
        ]) - self._drift_bias_gyro

        # 更新姿态
        dt = 0.01  # 假设10ms采样
        if self._last_timestamp:
            dt = ts - self._last_timestamp
        self._last_timestamp = ts
        self.update_orientation(gyro, dt)

        # 添加模拟运动
        if np.random.rand() < 0.1:
            accel += np.random.uniform(-0.5, 0.5, 3)
            gyro += np.random.uniform(-0.1, 0.1, 3)

        euler = self.get_euler_from_quaternion(self._orientation)

        self._sample_count += 1
        return IMUData(
            timestamp=ts,
            sensor_id=self.sensor_id,
            model=self.model,
            acceleration=accel,
            angular_velocity=gyro,
            euler=euler,
            quaternion=self._orientation.copy(),
            raw_accel=accel / self.GRAVITY,  # 归一化
            raw_gyro=gyro * 180 / np.pi,
            signal_quality={"accel": 0.98, "gyro": 0.97, "mag": 0.0}
        )


class MPU9250(IMUSensor):
    """MPU9250 9轴IMU"""

    def __init__(self, sensor_id: str, config: Optional[Dict] = None):
        super().__init__(sensor_id, IMUModel.MPU9250, config)
        self.accel_range = config.get("accel_range", 16)
        self.gyro_range = config.get("gyro_range", 2000)
        # 磁力计偏移
        self._mag_offset = np.array([0.0, 0.0, 0.0])

    def read(self, timestamp: Optional[float] = None) -> IMUData:
        """读取MPU9250数据"""
        ts = timestamp or np.datetime64('now').astype(float) / 1e9

        # 模拟加速度
        accel = np.array([
            np.random.normal(0, 0.05),
            np.random.normal(0, 0.05),
            np.random.normal(-self.GRAVITY, 0.05)
        ])

        # 模拟陀螺仪
        gyro = np.random.normal(0, 0.01, 3)

        # 模拟磁力计 (模拟地磁场 ~25-65μT)
        mag = np.array([
            np.random.normal(25, 2),
            np.random.normal(0, 2),
            np.random.normal(-25, 2)
        ]) + self._mag_offset

        # 更新姿态
        dt = 0.01
        if self._last_timestamp:
            dt = ts - self._last_timestamp
        self._last_timestamp = ts
        self.update_orientation(gyro, dt)
        euler = self.get_euler_from_quaternion(self._orientation)

        return IMUData(
            timestamp=ts,
            sensor_id=self.sensor_id,
            model=self.model,
            acceleration=accel,
            angular_velocity=gyro,
            magnetic_field=mag,
            euler=euler,
            quaternion=self._orientation.copy(),
            raw_accel=accel / self.GRAVITY,
            raw_gyro=gyro * 180 / np.pi,
            signal_quality={"accel": 0.95, "gyro": 0.95, "mag": 0.85}
        )


class IMUArray:
    """多IMU管理 (如AGV多节点)"""

    def __init__(self):
        self.sensors: Dict[str, IMUSensor] = {}

    def add_sensor(self, sensor: IMUSensor):
        """添加IMU"""
        self.sensors[sensor.sensor_id] = sensor

    def read_all(self, timestamp: Optional[float] = None) -> List[IMUData]:
        """读取所有IMU"""
        return [sensor.read(timestamp) for sensor in self.sensors.values()]

    def get_fusion_data(self) -> np.ndarray:
        """获取融合数据"""
        all_data = self.read_all()
        vectors = [d.to_vector() for d in all_data]
        return np.concatenate(vectors) if vectors else np.array([])

    def estimate_pose_change(self, dt: float) -> Dict[str, float]:
        """估计整体姿态变化"""
        all_changes = []
        for data in self.read_all():
            all_changes.append(data.get_imu_pose_change(dt))

        if not all_changes:
            return {}

        # 平均各IMU的估计
        avg = {}
        for key in all_changes[0]:
            avg[key] = np.mean([c[key] for c in all_changes])
        return avg

    def detect_motion(self, threshold: float = 0.5) -> bool:
        """检测运动状态"""
        for sensor in self.sensors.values():
            if sensor._last_timestamp:
                vel = np.linalg.norm(sensor._last_timestamp)
                if vel > threshold:
                    return True
        return False

    def compute_heading(self) -> float:
        """计算航向角 (yaw)"""
        headings = []
        for sensor in self.sensors.values():
            if hasattr(sensor, '_orientation'):
                q = sensor._orientation
                yaw = np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]*q[2] + q[3]*q[3]))
                headings.append(yaw)
        return np.mean(headings) if headings else 0.0

    def calibrate_gyro_bias(self, samples: int = 100):
        """校准陀螺仪偏置 (静止时)"""
        print(f"校准陀螺仪偏置，采集{samples}个样本...")
        gyro_readings = []
        for _ in range(samples):
            for sensor in self.sensors.values():
                data = sensor.read()
                if data.angular_velocity is not None:
                    gyro_readings.append(data.angular_velocity)
        if gyro_readings:
            bias = np.mean(gyro_readings, axis=0)
            for sensor in self.sensors.values():
                sensor._drift_bias_gyro = bias
            print(f"偏置校准完成: {bias}")
