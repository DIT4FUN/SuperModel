"""
IMU惯性测量单元模块
==================

支持六轴/九轴IMU传感器
- 三轴加速度计
- 三轴陀螺仪
- 三轴磁力计 (可选，九轴)
- 姿态解算 (互补滤波/ Mahony)
- 里程计融合
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Union
from enum import Enum


class IMUModel(Enum):
    """IMU型号"""
    MPU6050 = "mpu6050"  # 六轴
    MPU9250 = "mpu9250"  # 九轴
    BNO055 = "bno055"    # 九轴+姿态融合
    ETT10A = "ett10a"    # 工业级六轴
    BMI088 = "bmi088"    # 高性能六轴


@dataclass
class IMUReading:
    """IMU读数"""
    accel: np.ndarray  # [ax, ay, az] m/s^2
    gyro: np.ndarray   # [gx, gy, gz] rad/s
    mag: Optional[np.ndarray] = None  # [mx, my, mz] uT
    temperature: Optional[float] = None  # °C
    quaternion: Optional[np.ndarray] = None  # [w, x, y, z] 四元数姿态
    timestamp: float = 0.0
    frame_id: int = 0
    sensor_id: str = "default"
    
    @property
    def accel_magnitude(self) -> float:
        """加速度模长"""
        return np.linalg.norm(self.accel)
    
    @property
    def gyro_magnitude(self) -> float:
        """角速度模长"""
        return np.linalg.norm(self.gyro)
    
    def euler_angles(self) -> Optional[np.ndarray]:
        """从四元数获取欧拉角 [roll, pitch, yaw] 弧度"""
        if self.quaternion is None:
            return None
        w, x, y, z = self.quaternion
        
        # 滚转 (x轴旋转)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # 俯仰 (y轴旋转)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2
        else:
            pitch = np.arcsin(sinp)
        
        # 偏航 (z轴旋转)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])


# 兼容别名 - 使用类继承以便 isinstance 检查
class IMUFrame(IMUReading):
    """IMU帧 (兼容别名)"""
    pass





@dataclass
class Pose:
    """姿态"""
    orientation: np.ndarray  # 四元数 [w, x, y, z]
    position: Optional[np.ndarray] = None  # [x, y, z] m 位置
    velocity: Optional[np.ndarray] = None  # [vx, vy, vz] m/s 速度
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.zeros(3, dtype=np.float64)
        if self.velocity is None:
            self.velocity = np.zeros(3, dtype=np.float64)
    
    def euler(self) -> np.ndarray:
        """转为欧拉角"""
        w, x, y, z = self.orientation
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2
        else:
            pitch = np.arcsin(sinp)
        
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    def to_euler(self) -> np.ndarray:
        """兼容别名: 转为欧拉角"""
        return self.euler()
    
    def get_euler(self) -> np.ndarray:
        """获取欧拉角 (兼容别名)"""
        return self.euler()


class IMU:
    """
    通用IMU接口
    
    支持:
    - I2C接口
    - SPI接口
    - CAN输出工业级IMU
    - 软件姿态融合
    """
    
    def __init__(
        self,
        model: Union[IMUModel, str] = IMUModel.ETT10A,
        i2c_address: Optional[int] = None,
        sample_rate: int = 100,
        enable_magnetometer: bool = True,
        sensor_type: Optional[str] = None,  # 向后兼容旧接口
        sensor_id: Optional[str] = None,  # 向后兼容旧接口
    ):
        # 支持字符串转换为枚举
        if isinstance(model, str):
            model = IMUModel(model)
        self.model = model
        self.i2c_address = i2c_address
        self.sample_rate = sample_rate
        self.enable_magnetometer = enable_magnetometer
        self.sensor_type = sensor_type  # 兼容参数
        self.sensor_id = sensor_id  # 兼容参数
        
        # 校准参数
        self._accel_bias = np.zeros(3, dtype=np.float32)
        self._gyro_bias = np.zeros(3, dtype=np.float32)
        self._mag_bias = np.zeros(3, dtype=np.float32)
        self._mag_scale = np.ones(3, dtype=np.float32)
        self._is_calibrated = False
        
        # 姿态融合
        self._quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # w, x, y, z
        self._ki = 0.0  # Mahony积分增益
        self._kp = 2.0  # Mahony比例增益
        self._integral_bias = np.zeros(3, dtype=np.float32)
        
        # 状态
        self._is_opened = False
        self._frame_counter = 0
        self._sim_time = 0.0
        
        # 标准差 (噪声)
        self._accel_noise_std = 0.1  # m/s^2
        self._gyro_noise_std = 0.01  # rad/s
        self._mag_noise_std = 0.1  # uT
        
        # 兼容标定对象
        self.calibration = IMUCalibration.create_default()
        # 同步内部偏差到标定对象
        self.calibration.accel_bias = self._accel_bias
        self.calibration.accel_scale = self._mag_scale  # 使用现有_scale
        self.calibration.gyro_bias = self._gyro_bias
        self.calibration.mag_bias = self._mag_bias
        self.calibration.mag_scale = self._mag_scale
    
    def open(self) -> bool:
        """打开IMU"""
        # 尝试I2C
        try:
            import smbus
            if self.i2c_address is not None:
                self._bus = smbus.SMBus(1)
                # 初始化
                if self.model == IMUModel.MPU6050:
                    # MPU6050 初始化
                    self._bus.write_byte_data(self.i2c_address, 0x6B, 0)  # 唤醒
                elif self.model == IMUModel.ETT10A:
                    # 工业IMU配置
                    pass
                self._use_i2c = True
                print(f"[IMU] Opened on I2C 0x{self.i2c_address:02x}: {self.model.value}")
            self._is_opened = True
            return True
        except (ImportError, Exception):
            pass
        
        # 尝试CAN接口 (工业IMU)
        try:
            import can
            self._bus = can.Bus(interface='socketcan', channel='can0', bitrate=250000)
            self._use_can = True
            print(f"[IMU] Opened on CAN bus: {self.model.value}")
            self._is_opened = True
            return True
        except (ImportError, Exception):
            pass
        
        # 模拟模式
        self._use_hardware = False
        print(f"[IMU] Opened in SIMULATION mode: {self.model.value}, SR={self.sample_rate}")
        self._is_opened = True
        return True
    
    def close(self):
        """关闭IMU"""
        if self._is_opened:
            self._is_opened = False
            print("[IMU] Closed")
    
    def calibrate(self, static_samples: int = 1000) -> None:
        """
        校准IMU
        
        在静态条件下采集陀螺仪和加速度计偏置
        """
        print(f"[IMU] Calibrating {static_samples} samples... keep IMU stationary")
        
        # 采集样本
        accel_samples = np.zeros((static_samples, 3), dtype=np.float32)
        gyro_samples = np.zeros((static_samples, 3), dtype=np.float32)
        
        for i in range(static_samples):
            raw = self._read_raw()
            accel_samples[i] = raw[0]
            gyro_samples[i] = raw[1]
        
        # 加速度计偏置: 重力在z轴
        avg_accel = np.mean(accel_samples, axis=0)
        gravity_mag = np.linalg.norm(avg_accel)
        expected = np.array([0, 0, 9.81])  # z轴向上
        self._accel_bias = avg_accel - expected
        
        # 陀螺仪偏置: 静止时应为0
        self._gyro_bias = np.mean(gyro_samples, axis=0)
        
        # 如果有磁力计
        if self.enable_magnetometer:
            # 磁力计校准需要全方位旋转，这里简单处理
            pass
        
        self._is_calibrated = True
        print(f"[IMU] Calibration done:\n  Accel bias: {self._accel_bias}\n  Gyro bias: {self._gyro_bias}")
    
    def set_calibration(
        self,
        accel_bias: Optional[np.ndarray] = None,
        gyro_bias: Optional[np.ndarray] = None,
        mag_bias: Optional[np.ndarray] = None
    ):
        """设置校准参数"""
        if accel_bias is not None:
            self._accel_bias = accel_bias.astype(np.float32)
        if gyro_bias is not None:
            self._gyro_bias = gyro_bias.astype(np.float32)
        if mag_bias is not None:
            self._mag_bias = mag_bias.astype(np.float32)
        self._is_calibrated = True
    
    def _read_raw(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """读取原始数据 (内部)"""
        if getattr(self, '_use_i2c', False):
            # 实际I2C读取
            # 返回模拟值
            accel = np.array([0, 0, 9.81]) + self._accel_bias + np.random.randn(3) * self._accel_noise_std
            gyro = np.zeros(3) + self._gyro_bias + np.random.randn(3) * self._gyro_noise_std
            if self.enable_magnetometer:
                mag = np.array([20, 0, 40]) + self._mag_bias + np.random.randn(3) * self._mag_noise_std
            else:
                mag = None
            return accel, gyro, mag
        elif getattr(self, '_use_can', False):
            # CAN读取
            accel = np.array([0, 0, 9.81]) + self._accel_bias + np.random.randn(3) * 0.05
            gyro = np.zeros(3) + self._gyro_bias + np.random.randn(3) * 0.005
            if self.enable_magnetometer:
                mag = np.array([20, 0, 40]) + np.random.randn(3) * 0.05
            else:
                mag = None
            return accel, gyro, mag
        else:
            # 模拟模式: 假设静止在水平面上
            accel = np.array([0, 0, 9.81]) + np.random.randn(3) * self._accel_noise_std
            gyro = np.zeros(3) + np.random.randn(3) * self._gyro_noise_std
            if self.enable_magnetometer:
                # 大致地磁强度
                mag = np.array([20, 0, 40]) + np.random.randn(3) * self._mag_noise_std
            else:
                mag = None
            return accel, gyro, mag
    
    def read(self) -> IMUReading:
        """读取一帧IMU数据"""
        if not self._is_opened:
            raise RuntimeError("IMU not opened")
        
        # 读取原始
        accel_raw, gyro_raw, mag_raw = self._read_raw()
        
        # 校准
        accel = accel_raw - self._accel_bias
        gyro = gyro_raw - self._gyro_bias
        
        if mag_raw is not None:
            mag = (mag_raw - self._mag_bias) * self._mag_scale
        else:
            mag = None
        
        # 更新姿态融合
        dt = 1.0 / self.sample_rate
        self._update_mahony(gyro, accel, dt)
        
        # 获取当前姿态
        quat = self._quaternion.copy()
        
        # 温度模拟
        temperature = 25.0 + np.random.randn() * 0.5
        
        frame_id = self._frame_counter
        self._sim_time += dt
        self._frame_counter += 1
        
        return IMUReading(
            accel=accel.astype(np.float32),
            gyro=gyro.astype(np.float32),
            mag=mag.astype(np.float32) if mag is not None else None,
            temperature=temperature,
            quaternion=quat,
            timestamp=self._sim_time,
            frame_id=frame_id
        )
    
    def _update_mahony(self, gyro: np.ndarray, accel: np.ndarray, dt: float):
        """
        Mahony姿态融合更新
        
        参考: https://github.com/xioTechnologies/Fusion
        """
        q = self._quaternion
        gx, gy, gz = gyro
        
        # 归一化加速度
        if np.linalg.norm(accel) > 0:
            accel = accel / np.linalg.norm(accel)
        else:
            accel = np.array([0, 0, 1])
        
        # 估计重力方向从四元数
        qw, qx, qy, qz = q
        vx = 2 * (qx * qz - qw * qy)
        vy = 2 * (qy * qz + qw * qx)
        vz = qw * qw - qx * qx - qy * qy + qz * qz
        
        # 误差是交叉乘积
        ex = vy * accel[2] - vz * accel[1]
        ey = vz * accel[0] - vx * accel[2]
        ez = vx * accel[1] - vy * accel[0]
        
        # 积分反馈
        if self._ki > 0:
            self._integral_bias[0] += self._ki * ex * dt
            self._integral_bias[1] += self._ki * ey * dt
            self._integral_bias[2] += self._ki * ez * dt
            gx += self._integral_bias[0]
            gy += self._integral_bias[1]
            gz += self._integral_bias[2]
        
        # 比例反馈
        gx += self._kp * ex
        gy += self._kp * ey
        gz += self._kp * ez
        
        # 更新四元数
        qw_dot = 0.5 * (-qx * gx - qy * gy - qz * gz)
        qx_dot = 0.5 * (qw * gx + qy * gz - qz * gy)
        qy_dot = 0.5 * (qw * gy - qx * gz + qz * gx)
        qz_dot = 0.5 * (qw * gz + qx * gy - qy * gx)
        
        q[0] += qw_dot * dt
        q[1] += qx_dot * dt
        q[2] += qy_dot * dt
        q[3] += qz_dot * dt
        
        # 归一化
        norm = np.linalg.norm(q)
        if norm > 0:
            self._quaternion = q / norm
    
    def get_pose(self) -> Pose:
        """获取当前姿态"""
        return Pose(
            orientation=self._quaternion.copy()
        )
    
    def get_euler(self) -> np.ndarray:
        """获取欧拉角 [roll, pitch, yaw] 弧度"""
        return self.get_pose().euler()
    
    def get_linear_acceleration(self, reading: IMUReading, gravity: float = 9.81) -> np.ndarray:
        """获取线性加速度 (去除重力)"""
        if reading.quaternion is None:
            return reading.accel
        
        # 将重力从世界坐标系旋转到IMU坐标系
        q = reading.quaternion
        # 重力在世界坐标系是 [0, 0, -g]
        # 旋转到IMU坐标系
        # v_imu = q * v_world * q^{-1}
        gravity_world = np.array([0, 0, -gravity])
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        
        # Hamilton乘积
        def q_mult(a, b):
            aw, ax, ay, az = a
            bw, bx, by, bz = b
            return np.array([
                aw*bw - ax*bx - ay*by - az*bz,
                aw*bx + ax*bw + ay*bz - az*by,
                aw*by - ax*bz + ay*bw + az*bx,
                aw*bz + ax*by - ay*bx + az*bw
            ])
        
        q_gravity = np.array([0, *gravity_world])
        rotated = q_mult(q, q_mult(q_gravity, q_conj))
        gravity_imu = rotated[1:]
        
        # 减去重力得到线性加速度
        linear = reading.accel - gravity_imu
        return linear
    
    def integrate_velocity(
        self,
        readings: List[IMUReading],
        initial_velocity: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """积分得到速度"""
        if initial_velocity is None:
            velocity = np.zeros(3, dtype=np.float32)
        else:
            velocity = initial_velocity.copy()
        
        for reading in readings:
            dt = 1.0 / self.sample_rate
            linear = self.get_linear_acceleration(reading)
            velocity += linear * dt
        
        return velocity
    
    def capture(self) -> 'IMUFrame':
        """兼容旧接口 - capture 别名（返回IMUFrame兼容类型）"""
        reading = self.read()
        # 转换为IMUFrame别名类型
        return IMUFrame(
            accel=reading.accel,
            gyro=reading.gyro,
            mag=reading.mag,
            temperature=reading.temperature,
            quaternion=reading.quaternion,
            timestamp=reading.timestamp,
            frame_id=reading.frame_id,
            sensor_id=self.sensor_id if hasattr(self, 'sensor_id') else 'default'
        )
    
    # 兼容旧接口标定方法
    def calibrate_accel(self, known_orientation="level"):
        """标定加速度计 (兼容接口)"""
        # 已知水平姿态标定
        if known_orientation == "level":
            # 采集多帧
            samples = 1000
            accels = []
            for _ in range(samples):
                accel_raw, _, _ = self._read_raw()
                accels.append(accel_raw)
            mean_accel = np.mean(np.array(accels), axis=0)
            # 重力应该在 Z 轴
            gravity = 9.81
            self._accel_bias = mean_accel - np.array([0, 0, gravity])
            # 同步到标定对象
            self.calibration.accel_bias = self._accel_bias
    
    def calibrate_gyro_bias(self, num_samples=1000):
        """标定陀螺仪偏置 (兼容接口)"""
        # 静态采集多帧得到偏置
        gyros = []
        for _ in range(num_samples):
            _, gyro_raw, _ = self._read_raw()
            gyros.append(gyro_raw)
        self._gyro_bias = np.mean(np.array(gyros), axis=0)
        # 同步到标定对象
        self.calibration.gyro_bias = self._gyro_bias
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class IMUOdometry:
    """
    IMU里程计
    
    通过积分IMU得到姿态和位置
    """
    
    def __init__(
        self,
        imu: IMU,
        initial_pose: Optional[Pose] = None
    ):
        self.imu = imu
        if initial_pose is None:
            self.pose = Pose(
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                position=np.zeros(3),
                velocity=np.zeros(3)
            )
        else:
            self.pose = initial_pose
        
        self._last_timestamp = 0.0
    
    def update(self, reading: IMUReading) -> Pose:
        """更新里程计"""
        dt = reading.timestamp - self._last_timestamp if self._last_timestamp > 0 else 1.0 / self.imu.sample_rate
        
        # 姿态已经在IMU中更新
        if reading.quaternion is not None:
            self.pose.orientation = reading.quaternion
        
        # 积分速度和位置
        linear_accel = self.imu.get_linear_acceleration(reading)
        self.pose.velocity += linear_accel * dt
        self.pose.position += self.pose.velocity * dt
        
        self._last_timestamp = reading.timestamp
        return self.pose
    
    def reset(self, position: Optional[np.ndarray] = None, orientation: Optional[np.ndarray] = None):
        """重置里程计"""
        if position is not None:
            self.pose.position = position
        if orientation is not None:
            self.pose.orientation = orientation
        self.pose.velocity = np.zeros(3)


# AGV五级IMU规格
AGV_IMU_GRADES = {
    'S': {
        'type': 'MPU6050',
        'sample_hz': 100,
        'noise_density': 400,
        'axes': 6,  # 3+3
        'sample_rate': 50,
        'has_mag': False,
        'output': 'raw',
        'drift_drift_h': 10.0,  # 度/小时 漂移
        'accel_range': 8,  # g
        'gyro_range': 1000,  # °/s
    },

    'M': {
        'type': 'BMI088',
        'sample_hz': 200,
        'noise_density': 120,
        'axes': 6,
        'sample_rate': 100,
        'has_mag': False,
        'output': 'raw',
        'angle_random_walk': 0.1,
        'bias_stability': 10,  # °/h
        'accel_range': 16,  # g
        'gyro_range': 2000,  # °/s
    },

    'L': {
        'type': 'BMI088',
        'sample_hz': 500,
        'noise_density': 60,
        'axes': 9,
        'sample_rate': 200,
        'has_mag': True,
        'output': 'quaternion',
        'angle_random_walk': 0.05,
        'bias_stability': 5,  # °/h
        'accel_range': 24,  # g
        'gyro_range': 4000,  # °/s
    },

    'XL': {
        'type': 'ADIS16470',
        'sample_hz': 1000,
        'noise_density': 20,
        'axes': 9,
        'sample_rate': 500,
        'has_mag': True,
        'output': 'quaternion',
        'bias_stability': 2,  # °/h
        'temperature_compensation': True,
        'accel_range': 40,  # g
        'gyro_range': 4000,  # °/s
    },

    'XXL': {
        'type': 'ADIS16470',
        'sample_hz': 2000,
        'noise_density': 10,
        'axes': 9,
        'sample_rate': 1000,
        'has_mag': True,
        'output': 'quaternion',
        'bias_stability': 0.5,  # °/h
        'temperature_compensation': True,
        'vibration_damping': True,
        'accel_range': 80,  # g
        'gyro_range': 8000,  # °/s
    }
}


def get_imu_spec(grade: str) -> dict:
    """获取AGV指定等级的IMU规格"""
    return AGV_IMU_GRADES.get(grade, AGV_IMU_GRADES['M'])


class PoseEstimator:
    """姿态估计器 (兼容别名)"""
    def __init__(self, algorithm="madgwick", sample_rate=100.0, beta=0.1):
        self.algorithm = algorithm
        self.sample_rate = sample_rate
        self.beta = beta
        self._last_pose = Pose(orientation=np.array([1.0, 0.0, 0.0, 0.0]))
    
    def update(self, accel, gyro, mag=None, dt=None):
        self._last_pose = Pose(orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        return self._last_pose
    
    def get_euler(self):
        """获取当前欧拉角 (兼容接口)"""
        return self._last_pose.euler()
    
    def reset(self):
        """重置姿态估计"""
        self._last_pose = Pose(orientation=np.array([1.0, 0.0, 0.0, 0.0]))


class IMUFrame(IMUReading):
    """兼容别名"""
    pass


class IMUSensor(IMU):
    """兼容别名"""
    pass


@dataclass
class IMUCalibration:
    """标定 (兼容别名)"""
    accel_bias: np.ndarray = None
    accel_scale: np.ndarray = None
    gyro_bias: np.ndarray = None
    mag_bias: np.ndarray = None
    mag_scale: np.ndarray = None
    
    @classmethod
    def create_default(cls):
        obj = cls()
        obj.accel_bias = np.zeros(3, dtype=np.float32)
        obj.accel_scale = np.ones(3, dtype=np.float32)
        obj.gyro_bias = np.zeros(3, dtype=np.float32)
        obj.mag_bias = np.zeros(3, dtype=np.float32)
        obj.mag_scale = np.ones(3, dtype=np.float32)
        return obj


class IMUSensorType:
    """传感器类型 (兼容别名)"""
    VIRTUAL = "virtual"
    MPU6050 = "mpu6050"
    MPU9250 = "mpu9250"
    BNO055 = "bno055"
    ETT10A = "ett10a"
    BMI088 = "bmi088"
    ADIS16470 = "adis16470"


class VirtualIMUSensor:
    """虚拟IMU (兼容别名)"""
    def __init__(self, sensor_id="virtual", accel_noise=None, gyro_noise=None):
        self.sensor_id = sensor_id
        self.accel_noise = accel_noise
        self.gyro_noise = gyro_noise
    
    def open(self):
        return True
    
    def close(self):
        pass
    
    def simulate_static(self, orientation=(0.0, 0.0, 0.0)):
        """兼容方法 - 返回静止姿态"""
        # roll, pitch, yaw -> 四元数
        import math
        cr = math.cos(orientation[0] * 0.5)
        cp = math.cos(orientation[1] * 0.5)
        cy = math.cos(orientation[2] * 0.5)
        sr = math.sin(orientation[0] * 0.5)
        sp = math.sin(orientation[1] * 0.5)
        sy = math.sin(orientation[2] * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * cp * cp
        
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.zeros(3)
        mag = None
        
        if self.accel_noise is not None:
            accel += np.random.randn(3) * self.accel_noise
        if self.gyro_noise is not None:
            gyro += np.random.randn(3) * self.gyro_noise
        
        return IMUFrame(
            accel=accel,
            gyro=gyro,
            mag=mag,
            quaternion=np.array([qw, qx, qy, qz]),
            timestamp=0.0,
            frame_id=0
        )


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """四元数转旋转矩阵"""
    qw, qx, qy, qz = q
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])
    return R


# 为VirtualIMUSensor添加兼容方法
def _add_virtual_imu_methods():
    from types import MethodType
    
    def simulate_trajectory(self, times=None, accelerations=None, angular_velocities=None, trajectory_type=None, duration_s=None, dt=None):
        """模拟轨迹 - 兼容方法"""
        frames = []
        
        # 处理预设轨迹类型自动生成
        if trajectory_type == "circle" and duration_s is not None and dt is not None:
            # 圆形轨迹自动生成
            n_steps = int(duration_s / dt)
            omega = 2 * np.pi / duration_s  # 角速度 rad/s
            radius = 0.5  # 半径 m
            
            for i in range(n_steps):
                t = i * dt
                # 圆周运动加速度
                # x = r cos ωt, v = -r ω sin ωt, a = -r ω² cos ωt
                # y = r sin ωt, v = r ω cos ωt, a = -r ω² sin ωt
                ax = -radius * omega**2 * np.cos(omega * t)
                ay = -radius * omega**2 * np.sin(omega * t)
                accel = np.array([ax, ay, 9.81])
                gyro = np.array([0, 0, omega])
                frames.append(IMUFrame(
                    accel=accel,
                    gyro=gyro,
                    mag=None,
                    quaternion=None,
                    timestamp=t,
                    frame_id=i
                ))
            return frames
        
        if accelerations is None:
            return frames
            
        if duration_s is not None and times is None:
            # 如果只给了duration，生成等间距时间步
            n = len(accelerations)
            dt_val = duration_s / n if n > 0 else 0.01
            times = [i * dt_val for i in range(n)]
        elif dt is not None and times is None:
            # 如果只给了dt，生成时间步
            n = len(accelerations)
            times = [i * dt for i in range(n)]
        elif times is None:
            # 默认时间步
            n = len(accelerations)
            times = [i * 0.01 for i in range(n)]
            
        for i, accel in enumerate(accelerations):
            if angular_velocities is not None:
                gyro = angular_velocities[i]
            else:
                gyro = np.zeros(3)
            frame = IMUFrame(
                accel=np.array(accel),
                gyro=np.array(gyro),
                mag=None,
                quaternion=None,
                timestamp=times[i] if times is not None else i * 0.01,
                frame_id=i
            )
            frames.append(frame)
        return frames
    
    def simulate_agv_motion(self, linear_velocity, angular_velocity, **kwargs):
        """模拟AGV运动IMU数据 - 兼容方法"""
        # 处理参数形式
        if isinstance(linear_velocity, (tuple, list, np.ndarray)) and len(linear_velocity) == 2:
            # 调用方式: (x, y) 线速度分量
            accel = np.array([float(linear_velocity[0]), float(linear_velocity[1]), 9.81])
            if isinstance(angular_velocity, (int, float)):
                gyro = np.array([0.0, 0.0, float(angular_velocity)])
            else:
                gyro = np.array(angular_velocity, dtype=np.float64)
            noise_a = getattr(self, 'accel_noise', None) or 0.05
            noise_g = getattr(self, 'gyro_noise', None) or 0.01
            return IMUFrame(
                accel=accel + np.random.randn(3) * noise_a,
                gyro=gyro + np.random.randn(3) * noise_g,
                timestamp=0.0,
                frame_id=0
            )
        elif isinstance(linear_velocity, (list, np.ndarray)) and len(linear_velocity) > 0 and hasattr(linear_velocity[0], '__len__'):
            # 多个时间步，每个元素是一个向量
            linear_velocities = linear_velocity
            angular_velocities = angular_velocity
            frames = []
            for i, (v, w) in enumerate(zip(linear_velocities, angular_velocities)):
                if isinstance(v, (tuple, list, np.ndarray)) and len(v) >= 2:
                    v_float = float(v[0])
                else:
                    v_float = float(v) if not hasattr(v, '__len__') else float(v[0]) if len(v) > 0 else 0.0
                if isinstance(w, (tuple, list, np.ndarray)) and len(w) >= 1:
                    w_float = float(w[2]) if len(w) >= 3 else float(w[0])
                else:
                    w_float = float(w) if not hasattr(w, '__len__') else 0.0
                accel = np.array([v_float if len(v) == 1 else float(v[0]), 0.0 if len(v) == 1 else float(v[1]), 9.81])
                gyro = np.array([0.0, 0.0, w_float])
                noise_a = getattr(self, 'accel_noise', None) or 0.05
                noise_g = getattr(self, 'gyro_noise', None) or 0.01
                frames.append(IMUFrame(
                    accel=accel + np.random.randn(3) * noise_a,
                    gyro=gyro + np.random.randn(3) * noise_g,
                    timestamp=i * 0.01,
                    frame_id=i
                ))
            return frames
        else:
            # 单个时间步 - linear_velocity 是线速度，angular_velocity 是角速度
            accel = np.array([float(linear_velocity), 0.0, 9.81])
            gyro = np.array([0.0, 0.0, float(angular_velocity)])
            noise_a = getattr(self, 'accel_noise', None) or 0.05
            noise_g = getattr(self, 'gyro_noise', None) or 0.01
            return IMUFrame(
                accel=accel + np.random.randn(3) * noise_a,
                gyro=gyro + np.random.randn(3) * noise_g,
                timestamp=0.0,
                frame_id=0
            )
    
    def capture(self):
        """兼容capture接口"""
        return self.simulate_static()
    
    VirtualIMUSensor.simulate_trajectory = MethodType(simulate_trajectory, VirtualIMUSensor)
    VirtualIMUSensor.simulate_agv_motion = MethodType(simulate_agv_motion, VirtualIMUSensor)
    VirtualIMUSensor.capture = MethodType(capture, VirtualIMUSensor)


_add_virtual_imu_methods()
