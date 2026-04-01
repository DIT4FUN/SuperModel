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
        """打开传感器
        
        仿真模式: 初始化模拟 IMU 状态
        硬件模式: 建立 I2C/SPI/USB/ROS 串口连接
        
        Returns:
            bool: 是否成功打开
        """
        import time
        self._is_opened = True
        self._last_frame = None
        self._frame_history = []
        self._frame_id = 0
        self._start_time = time.time()
        
        # 接口信息
        if self.sensor_type == IMUSensorType.BMI088:
            interface = "SPI@20MHz / I2C@400kHz"
        elif self.sensor_type == IMUSensorType.MPU6050:
            interface = "I2C@100kHz"
        elif self.sensor_type == IMUSensorType.MPU9250:
            interface = "I2C@400kHz (9-axis)"
        elif self.sensor_type == IMUSensorType.ADIS16470:
            interface = "SPI@40MHz (工业级)"
        else:
            interface = "VIRTUAL"
        
        print(f"[IMUSensor] Opened ({interface}): {self.sensor_id}, "
              f"Accel=±{self.accel_range}g, Gyro=±{self.gyro_range}°/s, "
              f"SampleRate={self.sample_rate}Hz")
        
        # 仿真: 显示校准状态
        print(f"[IMUSensor] Calibration: accel_bias={self.calibration.accel_bias}, "
              f"gyro_bias={self.calibration.gyro_bias}")
        
        return True
    
    def close(self):
        """关闭传感器"""
        if self._is_opened:
            self._is_opened = False
            print(f"[IMUSensor] {self.sensor_id} Closed")
    
    def capture(self) -> IMUFrame:
        """采集一帧IMU数据
        
        仿真模式: 生成基于物理模型的IMU数据
        硬件模式: 从 I2C/SPI (BMI088/MPU6050) / USB HID / ROS 串口读取
        
        Returns:
            IMUFrame: 加速度、角速度、磁力计(可选)、温度、时间戳
        """
        if not self._is_opened:
            raise RuntimeError("IMU sensor not opened")
        
        import time
        t = len(self._frame_history) / self.sample_rate
        real_t = time.time() - self._start_time
        
        # --- 仿真模式: 基于物理模型的IMU数据生成 ---
        
        # 1. 重力向量 (假设传感器水平放置,重力沿+Z)
        gravity = np.array([0.0, 0.0, 9.81])
        
        # 2. 运动引起的比力 (简化: 假设静止或缓慢运动)
        motion_accel = np.array([0.0, 0.0, 0.0])
        
        # 3. 角度变化 (缓慢倾斜,模拟手臂运动)
        # 使用历史数据计算趋势
        if self._frame_history:
            prev_gyro = self._frame_history[-1].gyro
            # 添加小幅度的连续运动
            gyro_trend = prev_gyro * 0.95 + np.random.randn(3) * 0.002
        else:
            gyro_trend = np.zeros(3)
        
        # 4. 传感器噪声 (符合各型号规格)
        # 噪声密度: μg/√Hz, 转换为 RMS
        noise_bw = np.sqrt(self.sample_rate / 2)  # 等效噪声带宽
        if self.sensor_type == IMUSensorType.BMI088:
            accel_noise_density = 120e-6 * 9.81  # 120 μg/√Hz -> m/s²/√Hz
            gyro_noise_density = 3e-6 * np.pi / 180  # 3 mdps/√Hz -> rad/s/√Hz
        elif self.sensor_type == IMUSensorType.MPU6050:
            accel_noise_density = 400e-6 * 9.81
            gyro_noise_density = 5e-5 * np.pi / 180
        elif self.sensor_type == IMUSensorType.ADIS16470:
            accel_noise_density = 20e-6 * 9.81
            gyro_noise_density = 0.1e-6 * np.pi / 180
        else:
            accel_noise_density = 100e-6 * 9.81
            gyro_noise_density = 1e-5 * np.pi / 180
        
        accel_noise = np.random.randn(3) * accel_noise_density * np.sqrt(noise_bw)
        gyro_noise = np.random.randn(3) * gyro_noise_density * np.sqrt(noise_bw)
        
        # 5. 偏置稳定性 (慢漂移)
        bias_drift_time = (real_t % 3600) / 3600  # 小时级漂移周期
        accel_bias_drift = 0.001 * np.sin(bias_drift_time * 2 * np.pi)
        gyro_bias_drift = 0.0001 * np.sin(bias_drift_time * 2 * np.pi)
        
        # 组合加速度
        accel = gravity + motion_accel + accel_noise + accel_bias_drift
        
        # 组合角速度 (静止时应接近零)
        gyro = gyro_trend + gyro_noise + gyro_bias_drift
        
        # 6. 温度 (受环境 + 自身功耗影响)
        # 传感器自发热约 0.5-2°C
        self_heating = 0.5 + 0.5 * (accel_magnitude := np.linalg.norm(accel) / 9.81)
        temperature = 25.0 + self_heating + np.random.randn() * 0.1
        
        # 7. 磁力计 (MPU9250 / 9轴 IMU)
        mag = None
        if self.sensor_type in [IMUSensorType.MPU9250, IMUSensorType.VIRTUAL]:
            # 地磁场 (典型值 25-65 μT, 取决于位置)
            earth_field = np.array([25.0, 0.0, 45.0])  # μT, 典型值
            mag_noise = np.random.randn(3) * 0.5
            # 磁偏角随时间缓慢变化
            mag_drift = 0.1 * np.sin(real_t / 100)
            mag = (earth_field + mag_noise + mag_drift).astype(np.float32)
        
        # 应用校准偏置和比例因子
        accel = (accel - self.calibration.accel_bias) * self.calibration.accel_scale
        gyro = (gyro - self.calibration.gyro_bias) * self.calibration.gyro_scale
        
        frame = IMUFrame(
            accel=accel.astype(np.float32),
            gyro=gyro.astype(np.float32),
            mag=mag,
            temperature=float(temperature),
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
    
    def get_rotation_matrix(self) -> np.ndarray:
        """获取当前旋转矩阵 (3x3)"""
        return self.get_pose().to_matrix()[:3, :3]
    
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
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])


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


class VirtualIMUSensor:
    """
    虚拟IMU传感器 (仿真环境使用)
    
    模拟惯性测量单元，用于:
    - 仿真环境中的姿态反馈
    - 运动轨迹验证
    - 传感器融合算法测试
    """
    
    def __init__(
        self,
        sensor_id: str = "virtual_imu",
        accel_noise: float = 0.01,
        gyro_noise: float = 0.001,
        gyro_bias: float = 0.0005
    ):
        self.sensor_id = sensor_id
        self.accel_noise = accel_noise
        self.gyro_noise = gyro_noise
        self.gyro_bias = gyro_bias
        self._is_opened = False
        self._frame_id = 0
        self._current_pose = Pose(
            position=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        self._velocity = np.zeros(3)
        self._gyro_bias = np.zeros(3)
    
    def open(self) -> bool:
        self._is_opened = True
        self._gyro_bias = np.random.randn(3) * self.gyro_bias
        return True
    
    def close(self):
        self._is_opened = False
    
    def simulate_static(
        self,
        orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> IMUFrame:
        """
        模拟静止状态
        
        Args:
            orientation: 欧拉角 (roll, pitch, yaw) rad
            
        Returns:
            IMUFrame with gravity-aligned acceleration
        """
        roll, pitch, yaw = orientation
        gravity = np.array([0.0, 0.0, 9.81])
        
        # 旋转重力向量
        R = self._euler_to_rot(roll, pitch, yaw)
        accel = R.T @ gravity
        
        noise_accel = np.random.randn(3) * self.accel_noise
        noise_gyro = np.random.randn(3) * self.gyro_noise
        
        frame = IMUFrame(
            accel=(accel + noise_accel).astype(np.float32),
            gyro=(noise_gyro + self._gyro_bias).astype(np.float32),
            mag=None,
            temperature=25.0 + np.random.randn() * 0.5,
            timestamp=0.0,
            frame_id=self._frame_id,
            sensor_id=self.sensor_id
        )
        self._frame_id += 1
        return frame
    
    def simulate_motion(
        self,
        linear_accel: Tuple[float, float, float],
        angular_vel: Tuple[float, float, float],
        dt: float = 0.01
    ) -> IMUFrame:
        """
        模拟运动状态
        
        Args:
            linear_accel: 线性加速度 (m/s^2)
            angular_vel: 角速度 (rad/s)
            dt: 时间步长
            
        Returns:
            IMUFrame with simulated motion
        """
        accel = np.array(linear_accel, dtype=np.float32) + np.array([0, 0, 9.81])
        gyro = np.array(angular_vel, dtype=np.float32)
        
        noise_accel = np.random.randn(3) * self.accel_noise
        noise_gyro = np.random.randn(3) * self.gyro_noise
        
        frame = IMUFrame(
            accel=(accel + noise_accel).astype(np.float32),
            gyro=(gyro + noise_gyro + self._gyro_bias).astype(np.float32),
            mag=None,
            temperature=25.0 + np.random.randn() * 0.5,
            timestamp=dt * self._frame_id,
            frame_id=self._frame_id,
            sensor_id=self.sensor_id
        )
        
        # 更新速度和位置
        linear_accel_np = np.array(linear_accel, dtype=np.float32)
        self._velocity += linear_accel_np * dt
        self._current_pose.position += self._velocity * dt
        
        self._frame_id += 1
        return frame
    
    def simulate_trajectory(
        self,
        trajectory_type: str = "circle",
        duration_s: float = 2.0,
        dt: float = 0.01
    ) -> List[IMUFrame]:
        """
        模拟典型轨迹
        
        Args:
            trajectory_type: 'circle', 'figure8', 'linear', 'sine'
            duration_s: 持续时间
            dt: 时间步长
            
        Returns:
            List of IMUFrame
        """
        frames = []
        n_frames = int(duration_s / dt)
        t = np.linspace(0, duration_s, n_frames)
        
        for i, ti in enumerate(t):
            if trajectory_type == "circle":
                x = np.cos(2 * np.pi * ti)
                y = np.sin(2 * np.pi * ti)
                vx = -2 * np.pi * np.sin(2 * np.pi * ti)
                vy = 2 * np.pi * np.cos(2 * np.pi * ti)
                ax = -(2 * np.pi)**2 * np.cos(2 * np.pi * ti)
                ay = -(2 * np.pi)**2 * np.sin(2 * np.pi * ti)
            elif trajectory_type == "figure8":
                x = np.sin(2 * np.pi * ti)
                y = np.sin(4 * np.pi * ti) / 2
                vx = 2 * np.pi * np.cos(2 * np.pi * ti)
                vy = 2 * np.pi * np.cos(4 * np.pi * ti)
                ax = -(2 * np.pi)**2 * np.sin(2 * np.pi * ti)
                ay = -(2 * np.pi)**2 * np.sin(4 * np.pi * ti)
            elif trajectory_type == "linear":
                x = ti
                y = 0.0
                vx, vy = 1.0, 0.0
                ax, ay = 0.0, 0.0
            elif trajectory_type == "sine":
                x = ti
                y = np.sin(2 * np.pi * ti)
                vx = 1.0
                vy = 2 * np.pi * np.cos(2 * np.pi * ti)
                ax = 0.0
                ay = -(2 * np.pi)**2 * np.sin(2 * np.pi * ti)
            else:
                raise ValueError(f"Unknown trajectory: {trajectory_type}")
            
            # 在惯性坐标系下
            accel = np.array([ax, ay, 0.0])
            omega = np.array([0.0, 0.0, 0.0])  # 简化，不考虑旋转
            
            frame = self.simulate_motion(
                (ax, ay, 0.0),
                (0.0, 0.0, 0.0),
                dt
            )
            frames.append(frame)
        
        return frames
    
    def _euler_to_rot(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """欧拉角转旋转矩阵"""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]
        ])
        return R
    
    def simulate_agv_motion(
        self,
        linear_velocity: Tuple[float, float] = (0.0, 0.0),
        angular_velocity: float = 0.0,
        dt: float = 0.01,
        grade: str = "M"
    ) -> IMUFrame:
        """
        模拟AGV运动 (考虑不同等级IMU特性)
        
        Args:
            linear_velocity: 线速度 (vx, vy) m/s
            angular_velocity: 角速度 omega rad/s
            dt: 时间步长
            grade: AGV等级 (S/M/L/XL/XXL)
            
        Returns:
            IMUFrame with simulated AGV motion
        """
        import math
        
        vx, vy = linear_velocity
        omega = angular_velocity
        
        # 根据AGV等级设置噪声特性
        grade_noise = {
            'S': (0.01, 0.005),
            'M': (0.005, 0.002),
            'L': (0.002, 0.001),
            'XL': (0.001, 0.0005),
            'XXL': (0.0005, 0.0002),
        }
        accel_n, gyro_n = grade_noise.get(grade, grade_noise['M'])
        
        # 加速度 = 线加速度 + 重力分量
        # 假设AGV在水平面上运动
        linear_accel_x = 0.0  # 简化，假设匀速
        linear_accel_y = 0.0
        linear_accel_z = 0.0
        
        # 旋转引起的向心加速度
        if abs(omega) > 1e-6 and math.sqrt(vx**2 + vy**2) > 1e-6:
            # 向心加速度 a = v x omega (在2D情况下)
            centripetal = omega * math.sqrt(vx**2 + vy**2)
            # 方向指向旋转中心
            linear_accel_x += -vy * omega
            linear_accel_y += vx * omega
        
        # 完整的比力 (去除重力后的加速度)
        specific_force = np.array([linear_accel_x, linear_accel_y, linear_accel_z])
        
        # 添加噪声
        noise_accel = np.random.randn(3) * accel_n
        noise_gyro = np.random.randn(3) * gyro_n
        
        frame = IMUFrame(
            accel=(specific_force + noise_accel).astype(np.float32),
            gyro=(np.array([0.0, 0.0, omega]) + noise_gyro + self._gyro_bias).astype(np.float32),
            mag=None,
            temperature=25.0 + np.random.randn() * 0.3,
            timestamp=dt * self._frame_id,
            frame_id=self._frame_id,
            sensor_id=self.sensor_id
        )
        
        self._frame_id += 1
        return frame
    
    def simulate_human_walking(
        self,
        step_frequency: float = 1.5,
        walk_speed: float = 1.0,
        duration_s: float = 5.0,
        dt: float = 0.01
    ) -> List[IMUFrame]:
        """
        模拟人类步行运动
        
        Args:
            step_frequency: 步频 (Hz)
            walk_speed: 行走速度 (m/s)
            duration_s: 持续时间
            dt: 时间步长
            
        Returns:
            List of IMUFrame
        """
        import math
        
        frames = []
        n_frames = int(duration_s / dt)
        t = np.linspace(0, duration_s, n_frames)
        
        step_period = 1.0 / step_frequency
        
        for i, ti in enumerate(t):
            # 步态周期 (0-1)
            phase = (ti % step_period) / step_period
            
            # 髋关节摆动 (简化的正弦模型)
            hip_swing = math.sin(phase * 2 * math.pi) * 0.3  # 约30度
            knee_flexion = abs(math.sin(phase * 2 * math.pi)) * 0.4  # 膝关节屈伸
            
            # 加速度分量 (简化的行走加速度模型)
            ax = walk_speed * math.cos(phase * 2 * math.pi) * step_frequency * 0.5
            ay = 0.0
            az = -0.5 * math.sin(phase * 4 * math.pi) * step_frequency**2  # 垂直振动
            
            # 角速度 (身体摇摆)
            omega_roll = 0.05 * math.sin(phase * 2 * math.pi)
            omega_pitch = 0.1 * math.cos(phase * 2 * math.pi)
            omega_yaw = 0.02 * step_frequency
            
            frame = IMUFrame(
                accel=np.array([ax, ay, az], dtype=np.float32),
                gyro=np.array([omega_roll, omega_pitch, omega_yaw], dtype=np.float32),
                mag=None,
                temperature=25.0 + np.random.randn() * 0.2,
                timestamp=dt * i,
                frame_id=self._frame_id,
                sensor_id=self.sensor_id
            )
            frames.append(frame)
            self._frame_id += 1
        
        return frames
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, *args):
        self.close()
