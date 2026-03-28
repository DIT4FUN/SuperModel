"""
IMU感知模块
===========

惯性测量单元接口
- 三轴加速度计
- 三轴陀螺仪
- 三轴磁力计
- 姿态解算 (Euler / Quaternion)
- 线性速度/位置估计

支持传感器:
- BMI088
- MPU6050 / MPU9250
- ADIS16470
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict
from enum import Enum


class IMUSensorType(Enum):
    """IMU传感器类型"""
    BMI088 = "bmi088"          # 高性能6轴
    MPU6050 = "mpu6050"        # 消费级6轴
    MPU9250 = "mpu9250"        # 9轴 (含磁力计)
    ADIS16470 = "adis16470"    # 工业级
    VIRTUAL = "virtual"         # 仿真/融合输出


@dataclass
class IMUFrame:
    """IMU数据帧"""
    accel: np.ndarray          # 3, 加速度 (m/s^2)
    gyro: np.ndarray           # 3, 角速度 (rad/s)
    mag: Optional[np.ndarray]  # 3, 磁力计 (可选)
    temperature: float = 25.0  # 温度 (摄氏度)
    timestamp: float = 0.0
    frame_id: int = 0
    sensor_id: str = "imu_0"
    
    def __post_init__(self):
        if isinstance(self.accel, list):
            self.accel = np.array(self.accel, dtype=np.float32)
        if isinstance(self.gyro, list):
            self.gyro = np.array(self.gyro, dtype=np.float32)
        if isinstance(self.mag, list):
            self.mag = np.array(self.mag, dtype=np.float32)
    
    @property
    def accel_magnitude(self) -> float:
        """加速度向量模长"""
        return np.linalg.norm(self.accel)
    
    @property
    def gyro_magnitude(self) -> float:
        """角速度向量模长"""
        return np.linalg.norm(self.gyro)


@dataclass
class Pose:
    """位姿 (位置 + 姿态)"""
    position: np.ndarray        # 3, 位置 (m)
    orientation: np.ndarray      # 4, 四元数 (qw, qx, qy, qz)
    
    def __post_init__(self):
        if isinstance(self.position, list):
            self.position = np.array(self.position, dtype=np.float32)
        if isinstance(self.orientation, list):
            self.orientation = np.array(self.orientation, dtype=np.float32)
    
    @classmethod
    def identity(cls) -> 'Pose':
        """单位位姿"""
        return cls(
            position=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
    
    def to_euler(self) -> np.ndarray:
        """转欧拉角 [roll, pitch, yaw], rad"""
        q = self.orientation
        # 四元数转欧拉角
        sinr_cosp = 2 * (q[0] * q[1] + q[2] * q[3])
        cosr_cosp = 1 - 2 * (q[1]**2 + q[2]**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (q[0] * q[2] - q[3] * q[1])
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        
        siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
        cosy_cosp = 1 - 2 * (q[2]**2 + q[3]**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def to_matrix(self) -> np.ndarray:
        """转4x4变换矩阵"""
        q = self.orientation
        # 四元数转旋转矩阵
        R = np.array([
            [1-2*(q[2]**2+q[3]**2), 2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2])],
            [2*(q[1]*q[2]+q[0]*q[3]), 1-2*(q[1]**2+q[3]**2), 2*(q[2]*q[3]-q[0]*q[1])],
            [2*(q[1]*q[3]-q[0]*q[2]), 2*(q[2]*q[3]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2)]
        ])
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = self.position
        return T
    
    @classmethod
    def from_euler(cls, position: np.ndarray, rpy: np.ndarray) -> 'Pose':
        """从欧拉角创建"""
        roll, pitch, yaw = rpy
        
        # 欧拉角转四元数
        cr, sr = np.cos(roll/2), np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return cls(position=position, orientation=np.array([qw, qx, qy, qz]))


@dataclass
class IMUCalibration:
    """IMU标定参数"""
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accel_scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    gyro_scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    mag_hard_iron: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mag_soft_iron: np.ndarray = field(default_factory=lambda: np.eye(3))
    temperature_model: Optional[Dict] = None


class IMUSensor:
    """
    IMU传感器接口
    
    支持:
    - 原始数据采集
    - 温度补偿
    - 自检
    """
    
    def __init__(
        self,
        sensor_type: IMUSensorType = IMUSensorType.BMI088,
        sensor_id: str = "imu_0",
        calibration: Optional[IMUCalibration] = None,
        accel_range: int = 16,     # g
        gyro_range: int = 2000,    # deg/s
        sample_rate: int = 200     # Hz
    ):
        self.sensor_type = sensor_type
        self.sensor_id = sensor_id
        self.accel_range = accel_range
        self.gyro_range = gyro_range
        self.sample_rate = sample_rate
        self.calibration = calibration or IMUCalibration()
        
        # 硬件参数换算
        self.accel_lsb_per_g = 32768 / accel_range
        self.gyro_lsb_per_dps = 32768 / gyro_range
        
        self._is_opened = False
        self._last_frame: Optional[IMUFrame] = None
        self._frame_history: List[IMUFrame] = []
        
    def open(self) -> bool:
        """打开传感器"""
        # TODO: 实现硬件接口
        # - I2C/SPI: smbus2, spidev
        # - USB: hidapi
        # - ROS: rosserial
        self._is_opened = True
        print(f"[IMUSensor] Opened: {self.sensor_id}, Type={self.sensor_type.value}, "
              f"Accel={self.accel_range}g, Gyro={self.gyro_range}dps")
        return True
    
    def close(self):
        """关闭传感器"""
        if self._is_opened:
            self._is_opened = False
            print(f"[IMUSensor] {self.sensor_id} Closed")
    
    def capture(self) -> IMUFrame:
        """采集一帧IMU数据"""
        if not self._is_opened:
            raise RuntimeError("IMU sensor not opened")
        
        # TODO: 实现实际数据采集
        # 模拟数据
        t = len(self._frame_history) / self.sample_rate
        
        # 模拟加速度 (静止时接近重力)
        gravity = np.array([0.0, 0.0, 9.81])
        accel_noise = np.random.randn(3) * 0.01
        accel = gravity + accel_noise
        
        # 模拟角速度 (静止时接近零)
        gyro = np.random.randn(3) * 0.01
        
        # 磁力计 (如果有)
        mag = None
        if self.sensor_type == IMUSensorType.MPU9250:
            # 地磁场 ~ 25-65 uT
            mag = np.array([25.0, 0.0, 45.0]) + np.random.randn(3) * 0.5
        
        # 应用标定偏置
        accel = (accel - self.calibration.accel_bias) * self.calibration.accel_scale
        gyro = (gyro - self.calibration.gyro_bias) * self.calibration.gyro_scale
        
        frame = IMUFrame(
            accel=accel.astype(np.float32),
            gyro=gyro.astype(np.float32),
            mag=mag.astype(np.float32) if mag is not None else None,
            temperature=25.0,
            timestamp=t,
            frame_id=len(self._frame_history),
            sensor_id=self.sensor_id
        )
        
        self._last_frame = frame
        self._frame_history.append(frame)
        
        if len(self._frame_history) > 5000:
            self._frame_history = self._frame_history[-2000:]
        
        return frame
    
    def self_test(self) -> bool:
        """
        传感器自检
        
        返回 True 表示自检通过
        """
        print(f"[IMUSensor] Running self-test for {self.sensor_id}...")
        
        # 采集几帧检查合理性
        frames = [self.capture() for _ in range(10)]
        
        # 检查加速度范围 (应该接近1g)
        avg_accel_mag = np.mean([f.accel_magnitude for f in frames])
        if not (5.0 < avg_accel_mag < 15.0):
            print(f"[IMUSensor] Self-test FAILED: accel magnitude {avg_accel_mag:.2f} out of range")
            return False
        
        # 检查角速度范围
        avg_gyro_mag = np.mean([f.gyro_magnitude for f in frames])
        if avg_gyro_mag > 1.0:  # rad/s
            print(f"[IMUSensor] Self-test FAILED: gyro magnitude {avg_gyro_mag:.2f} too large")
            return False
        
        print(f"[IMUSensor] Self-test PASSED")
        return True
    
    def calibrate_gyro_bias(self, num_samples: int = 500, duration_sec: float = 5.0):
        """
        陀螺仪偏置校准
        
        在传感器静止状态下采集
        """
        print(f"[IMUSensor] Calibrating gyro bias ({num_samples} samples)...")
        
        biases = []
        for i in range(num_samples):
            frame = self.capture()
            biases.append(frame.gyro)
        
        bias = np.mean(biases, axis=0)
        std = np.std(biases, axis=0)
        
        print(f"[IMUSensor] Gyro bias: {bias}, std: {std}")
        
        if np.any(std > 0.1):
            print(f"[IMUSensor] WARNING: High gyro noise during calibration")
        
        self.calibration.gyro_bias = bias
    
    def calibrate_accel(self, known_orientation: str = "level"):
        """
        加速度计标定
        
        Args:
            known_orientation: 已知朝向 ("level", "up", "down", "left", "right", "front", "back")
        """
        print(f"[IMUSensor] Calibrating accel (orientation={known_orientation})...")
        
        frames = [self.capture() for _ in range(100)]
        accel_mean = np.mean([f.accel for f in frames], axis=0)
        
        # 理论重力向量
        gravity_map = {
            "level": np.array([0, 0, 9.81]),
            "up": np.array([0, 0, -9.81]),
            "down": np.array([0, 0, 9.81]),
            "left": np.array([0, 9.81, 0]),
            "right": np.array([0, -9.81, 0]),
            "front": np.array([9.81, 0, 0]),
            "back": np.array([-9.81, 0, 0]),
        }
        
        expected = gravity_map.get(known_orientation, gravity_map["level"])
        
        # 计算比例因子
        scale = np.linalg.norm(expected) / np.linalg.norm(accel_mean)
        self.calibration.accel_scale = np.ones(3) * scale
        
        print(f"[IMUSensor] Accel scale: {self.calibration.accel_scale}")
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PoseEstimator:
    """
    姿态估计器
    
    方法:
    - 互补滤波
    - 卡尔曼滤波
    - Mahony/Madgwick 算法
    """
    
    def __init__(
        self,
        algorithm: str = "madgwick",
        sample_rate: float = 200.0,
        beta: float = 0.1  # Madgwick 增益
    ):
        """
        Args:
            algorithm: "madgwick" / "complementary" / "kalman"
            sample_rate: 采样频率 Hz
            beta: Madgwick 滤波器增益
        """
        self.algorithm = algorithm
        self.sample_rate = sample_rate
        self.beta = beta
        
        # 状态
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # qw, qx, qy, qz
        
        # 互补滤波参数
        self.alpha = 0.98  # 陀螺仪权重
        
        # 速度/位置积分
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)
        self._initialized = False
        
    def update(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        mag: Optional[np.ndarray] = None,
        dt: Optional[float] = None
    ) -> Pose:
        """
        更新姿态估计
        
        Args:
            accel: 3D 加速度 (m/s^2)
            gyro: 3D 角速度 (rad/s)
            mag: 3D 磁力计 (可选, uT)
            dt: 时间步长 (可选, 默认用 sample_rate 计算)
        """
        if dt is None:
            dt = 1.0 / self.sample_rate
        
        if self.algorithm == "madgwick":
            self._update_madgwick(accel, gyro, mag, dt)
        elif self.algorithm == "complementary":
            self._update_complementary(accel, gyro, dt)
        elif self.algorithm == "kalman":
            self._update_kalman(accel, gyro, dt)
        else:
            self._update_madgwick(accel, gyro, mag, dt)
        
        return self.get_pose()
    
    def _update_madgwick(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        mag: Optional[np.ndarray],
        dt: float
    ):
        """Madgwick AHRS 算法"""
        q = self.quaternion
        
        # 归一化加速度
        accel_norm = np.linalg.norm(accel)
        if accel_norm < 1e-6:
            return
        a = accel / accel_norm
        
        # 梯度下降
        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - a[0],
            2*(q[0]*q[1] + q[2]*q[3]) - a[1],
            2*(0.5 - q[1]**2 - q[2]**2) - a[2]
        ])
        
        J = np.array([
            [-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
            [2*q[1], 2*q[0], 2*q[3], 2*q[2]],
            [0, -4*q[1], -4*q[2], 0]
        ])
        
        step = J.T @ f
        step_norm = np.linalg.norm(step)
        if step_norm > 0:
            step = step / step_norm
        
        # 四元数微分
        q_dot = 0.5 * np.array([
            -q[1]*gyro[0] - q[2]*gyro[1] - q[3]*gyro[2],
            q[0]*gyro[0] + q[2]*gyro[2] - q[3]*gyro[1],
            q[0]*gyro[1] - q[1]*gyro[2] + q[3]*gyro[0],
            q[0]*gyro[2] + q[1]*gyro[1] - q[2]*gyro[0]
        ])
        
        # 融合
        self.quaternion = q + (q_dot - self.beta * step) * dt
        
        # 归一化
        self.quaternion = self.quaternion / np.linalg.norm(self.quaternion)
    
    def _update_complementary(self, accel: np.ndarray, gyro: np.ndarray, dt: float):
        """互补滤波 (简化版)"""
        # 从加速度估算roll/pitch
        accel_norm = np.linalg.norm(accel)
        if accel_norm < 1e-6:
            return
        
        accel_n = accel / accel_norm
        
        # 加速度计估算
        accel_roll = np.arctan2(accel_n[1], accel_n[2])
        accel_pitch = np.arctan2(-accel_n[0], np.sqrt(accel_n[1]**2 + accel_n[2]**2))
        
        # 陀螺仪积分 (需要当前欧拉角)
        current_euler = self.get_euler()
        gyro_roll = current_euler[0] + gyro[0] * dt
        gyro_pitch = current_euler[1] + gyro[1] * dt
        
        # 互补融合
        roll = self.alpha * gyro_roll + (1 - self.alpha) * accel_roll
        pitch = self.alpha * gyro_pitch + (1 - self.alpha) * accel_pitch
        yaw = current_euler[2] + gyro[2] * dt
        
        # 更新四元数
        self.quaternion = self._euler_to_quaternion(np.array([roll, pitch, yaw]))
    
    def _update_kalman(self, accel: np.ndarray, gyro: np.ndarray, dt: float):
        """简单卡尔曼滤波 (简化版)"""
        # 使用Madgwick作为预测，加速度作为观测修正
        self._update_madgwick(accel, gyro, None, dt)
    
    def _euler_to_quaternion(self, euler: np.ndarray) -> np.ndarray:
        """欧拉角转四元数"""
        roll, pitch, yaw = euler
        
        cr, sr = np.cos(roll/2), np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)
        
        return np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        ])
    
    def get_pose(self) -> Pose:
        """获取当前姿态"""
        return Pose(
            position=self.position.copy(),
            orientation=self.quaternion.copy()
        )
    
    def get_euler(self) -> np.ndarray:
        """获取当前欧拉角 [roll, pitch, yaw], rad"""
        return self.get_pose().to_euler()
    
    def integrate_velocity(
        self,
        accel: np.ndarray,
        dt: float,
        remove_gravity: bool = True
    ):
        """
        积分加速度获得速度/位置
        
        Warning: 漂移严重，仅短时间有效
        """
        # 去除重力
        if remove_gravity:
            gravity = np.array([0, 0, 9.81])
            accel = accel - gravity
        
        self.velocity = self.velocity + accel * dt
        self.position = self.position + self.velocity * dt
        
        return self.velocity.copy(), self.position.copy()
    
    def reset(self):
        """重置积分状态"""
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)
        self._initialized = False


# AGV五级IMU规格
AGV_IMU_GRADES = {
    'S':  {'type': 'MPU6050', 'accel_range': 8,   'gyro_range': 1000,  'sample_hz': 100,  'noise_density': 400},
    'M':  {'type': 'BMI088',  'accel_range': 16,  'gyro_range': 2000,  'sample_hz': 200,  'noise_density': 120},
    'L':  {'type': 'BMI088',  'accel_range': 24,  'gyro_range': 4000,  'sample_hz': 500,  'noise_density': 60},
    'XL': {'type': 'ADIS16470', 'accel_range': 40, 'gyro_range': 4000, 'sample_hz': 1000, 'noise_density': 20},
    'XXL': {'type': 'ADIS16470', 'accel_range': 80, 'gyro_range': 8000, 'sample_hz': 2000, 'noise_density': 10},
}


def get_imu_spec(grade: str) -> dict:
    """获取AGV指定等级的IMU规格"""
    return AGV_IMU_GRADES.get(grade, AGV_IMU_GRADES['M'])
