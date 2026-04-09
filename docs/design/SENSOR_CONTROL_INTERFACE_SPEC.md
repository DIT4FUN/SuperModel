# SuperModel 传感器-控制模块详细接口规范
## Sensor & Control Module Interface Specification v2.05.0

---

## 1. 概述

本文档定义 SuperModel 超模态大模型机器人具身智能大脑的传感器-控制模块接口规范，覆盖触觉、力觉、IMU 三大具身感知传感器与运动控制器的完整接口设计。

### 1.1 文档目的
- 定义传感器模块的标准接口契约
- 定义控制器模块的标准接口契约
- 确保传感器-控制器集成的一致性
- 支持 AGV 五级规格 (S/M/L/XL/XXL) 的可配置扩展

### 1.2 模块层次结构

```
┌─────────────────────────────────────────────────────────────┐
│                    具身智能大脑 (Embodied Brain)             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ 视觉感知   │  │ 听觉感知   │  │ 具身感知            │ │
│  │ Vision     │  │ Audio      │  │ Embodied            │ │
│  │ (视觉)    │  │ (听觉)     │  │ (触觉/力觉/IMU)     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                            │                                │
│               ┌────────────┴────────────┐                   │
│               │     跨模态融合网络       │                   │
│               │  CrossModalFusion       │                   │
│               │  Transformer×6模态     │                   │
│               └────────────┬────────────┘                   │
│                            │                                │
│               ┌────────────┴────────────┐                   │
│               │     传感-运动融合        │                   │
│               │  SensorimotorIntegration│                   │
│               └────────────┬────────────┘                   │
│                            │                                │
│  ┌─────────────────────────┼─────────────────────────────┐  │
│  │                    运 动 控 制 器                      │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │  │
│  │  │ 电机控制 │ │ 运动控制 │ │ 轨迹规划 │ │ 安全监控 │ │  │
│  │  │ Motor    │ │ Motion   │ │ Trajectory│ │ Safety   │ │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 触觉传感器模块接口 (TactileSensor)

### 2.1 基类接口

```python
class TactileSensor(ABC):
    """触觉传感器抽象基类"""

    @abstractmethod
    def open(self) -> bool:
        """打开传感器连接"""
        pass

    @abstractmethod
    def close(self):
        """关闭传感器连接"""
        pass

    @abstractmethod
    def capture(self, timestamp: Optional[float] = None) -> TactileFrame:
        """采集一帧触觉数据"""
        pass

    @abstractmethod
    def calibrate(self, reference_data: np.ndarray) -> bool:
        """校准传感器"""
        pass

    @abstractmethod
    def get_spec(self) -> Dict[str, Any]:
        """获取传感器规格"""
        pass
```

### 2.2 TactileFrame 数据结构

```python
@dataclass
class TactileFrame:
    """触觉帧数据结构"""
    sensor_id: str                    # 传感器唯一标识
    timestamp: float                  # 时间戳 (秒)
    frame: np.ndarray                # 触感阵列数据 [rows, cols], dtype=uint8
    temperature: float                # 温度 (°C)
    pressure_scale: float            # 压力换算系数 (Pa/count)
    baseline: Optional[np.ndarray]    # 基线数据 (用于基线补偿)
    contact_count: int               # 检测到的接触点数
    slip_detected: bool              # 是否检测到滑移
    grasp_quality: float             # 抓取质量评分 [0, 1]
    sensor_health: SensorHealth      # 传感器健康状态
    extra_data: Dict[str, Any]       # 扩展数据

    def get_pressure_map(self) -> np.ndarray:
        """获取压力分布图 (Pa)"""
        return self.frame.astype(np.float32) * self.pressure_scale

    def get_contact_centroids(self) -> List[Tuple[float, float]]:
        """获取各接触区域质心 (归一化坐标)"""
        ...

    def get_total_force(self) -> float:
        """获取总压力 (N)"""
        ...
```

### 2.3 TactileArray 具体实现接口

```python
class TactileArray(TactileSensor):
    """
    电子皮肤触觉阵列
    适用于: 机械臂末端抓取器、AGV 车体表面
    """

    def __init__(
        self,
        array_size: Tuple[int, int] = (16, 16),
        sensor_type: TactileSensorType = TactileSensorType.RESISTIVE,
        sensor_id: str = "tactile_array_0",
        max_pressure: float = 1000.0,  # Pa
        sample_rate: float = 100.0,   # Hz
        communication: str = "i2c"     # i2c / spi / uart
    ):
        ...

    # ── 采集接口 ──────────────────────────────────────────────
    def capture(self, timestamp: Optional[float] = None) -> TactileFrame:
        """
        采集一帧触觉数据

        Returns:
            TactileFrame: 包含完整触觉信息的数据帧
        """
        ...

    def capture_async(self) -> TactileFrame:
        """异步采集 (不阻塞)"""
        ...

    # ── 接触检测接口 ──────────────────────────────────────────
    def detect_contacts(
        self,
        frame: TactileFrame,
        threshold: float = 0.1
    ) -> List[TactileContact]:
        """
        检测接触事件

        Args:
            frame: 触觉帧
            threshold: 接触检测阈值 (归一化 0-1)

        Returns:
            List[TactileContact]: 接触事件列表
        """
        ...

    def detect_slip(self, frame: TactileFrame) -> bool:
        """
        检测滑移

        基于库仑摩擦模型: F_t > μ * F_n 时发生滑移

        Returns:
            bool: 是否检测到滑移
        """
        ...

    def estimate_grasp_quality(self, frame: TactileFrame) -> float:
        """
        评估抓取质量

        综合评估:
        - 接触面积 (力封闭)
        - 重心偏移
        - 滑移风险
        - 稳定性评分

        Returns:
            float: 抓取质量评分 [0, 1]
        """
        ...

    # ── 数据处理接口 ──────────────────────────────────────────
    def filter_frame(
        self,
        frame: np.ndarray,
        method: str = "median"
    ) -> np.ndarray:
        """
        滤波处理

        Args:
            frame: 原始帧数据
            method: 滤波方法 (median / mean / gaussian / kalman)

        Returns:
            np.ndarray: 滤波后数据
        """
        ...

    def compensate_baseline(
        self,
        frame: np.ndarray,
        baseline: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """基线补偿"""
        ...

    def compute_centroid(self, frame: np.ndarray) -> Tuple[float, float]:
        """计算压力分布质心 (归一化坐标)"""
        ...
```

### 2.4 AGV 五级触觉规格

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **阵列尺寸** | 8×8 | 16×16 | 24×24 | 32×32 | 48×48 |
| **分辨率** | 8bit | 12bit | 14bit | 14bit | 16bit |
| **压力范围** | 0-500kPa | 0-1MPa | 0-2MPa | 0-5MPa | 0-10MPa |
| **采样率** | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| **通信接口** | UART | I2C | SPI | SPI | SPI+DMA |
| **温度感知** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **接近觉** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **滑移检测** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **抓取质量评估** | ✗ | ✓ | ✓ | ✓ | ✓ |

---

## 3. 力觉传感器模块接口 (ForceSensor)

### 3.1 基类接口

```python
class ForceSensor(ABC):
    """力觉传感器抽象基类"""

    @abstractmethod
    def open(self) -> bool:
        """打开传感器连接"""
        pass

    @abstractmethod
    def close(self):
        """关闭传感器连接"""
        pass

    @abstractmethod
    def capture(self, timestamp: Optional[float] = None) -> Wrench:
        """采集一帧力觉数据"""
        pass

    @abstractmethod
    def set_bias(self) -> bool:
        """设置零偏 (当前值作为零点)"""
        pass

    @abstractmethod
    def clear_bias(self) -> bool:
        """清除零偏"""
        pass

    @abstractmethod
    def get_spec(self) -> Dict[str, Any]:
        """获取传感器规格"""
        pass
```

### 3.2 Wrench 数据结构

```python
@dataclass
class Wrench:
    """
    六维力旋量 (Wrench)

    表示作用在刚体上的广义力:
    - 力 (Force): Fx, Fy, Fz [N]
    - 力矩 (Torque): Mx, My, Mz [Nm]
    """

    force: np.ndarray      # [Fx, Fy, Fz] 单位: N
    torque: np.ndarray    # [Mx, My, Mz] 单位: Nm
    timestamp: float      # 时间戳 (秒)
    sensor_id: str        # 传感器ID
    is_saturated: bool    # 是否饱和
    temperature: float     # 温度 (°C)

    @classmethod
    def zero(cls, sensor_id: str = "") -> "Wrench":
        """创建零力旋量"""
        return cls(
            force=np.zeros(3),
            torque=np.zeros(3),
            timestamp=0.0,
            sensor_id=sensor_id,
            is_saturated=False,
            temperature=25.0
        )

    @property
    def force_magnitude(self) -> float:
        """合力大小 (N)"""
        return np.linalg.norm(self.force)

    @property
    def torque_magnitude(self) -> float:
        """合力矩大小 (Nm)"""
        return np.linalg.norm(self.torque)

    @property
    def wrench_vector(self) -> np.ndarray:
        """六维向量 [Fx, Fy, Fz, Mx, My, Mz]"""
        return np.concatenate([self.force, self.torque])

    def transform(self, transform_matrix: np.ndarray) -> "Wrench":
        """
        坐标变换

        Args:
            transform_matrix: 4x4 齐次变换矩阵

        Returns:
            Wrench: 变换后的力旋量
        """
        ...

    def to_vector(self, norm: bool = True) -> np.ndarray:
        """
        转换为特征向量

        Args:
            norm: 是否归一化

        Returns:
            np.ndarray: [Fx, Fy, Fz, Mx, My, Mz] (归一化或原始)
        """
        if norm:
            # 归一化到 [-1, 1]
            force_norm = self.force / 200.0   # 假设最大200N
            torque_norm = self.torque / 5.0    # 假设最大5Nm
            return np.concatenate([force_norm, torque_norm])
        return self.wrench_vector
```

### 3.3 ForceTorqueSensor 具体实现接口

```python
class ForceTorqueSensor(ForceSensor):
    """
    六维力矩传感器

    典型型号: ATI mini40 / Gamma / SI-120-2.3
    适用于: 机械臂末端力控、AGV 碰撞检测
    """

    def __init__(
        self,
        sensor_id: str = "ft_sensor_0",
        model: str = "mini40",  # mini40 / gamma / si-120
        calibration: Optional[str] = None,  # 校准文件路径
        communication: str = "ethercat"  # ethercat / rs485 / can
    ):
        # ATI mini40 典型规格
        if model.lower() == "mini40":
            self.force_range = np.array([120, 120, 120])   # N
            self.torque_range = np.array([2, 2, 2])         # Nm
        elif model.lower() == "gamma":
            self.force_range = np.array([200, 200, 200])
            self.torque_range = np.array([10, 10, 10])
        ...

    # ── 采集接口 ──────────────────────────────────────────────
    def capture(self, timestamp: Optional[float] = None) -> Wrench:
        """
        采集一帧六维力数据

        Returns:
            Wrench: 包含完整力觉信息的数据帧
        """
        ...

    def capture_raw(self) -> np.ndarray:
        """
        采集原始 ADC 数据 (未校准)

        Returns:
            np.ndarray: 原始六维数据 [Fx, Fy, Fz, Mx, My, Mz]
        """
        ...

    # ── 校准接口 ──────────────────────────────────────────────
    def set_bias(self) -> bool:
        """
        设置零偏 (假设当前处于自由状态)

        存储当前读数作为零点，后续读数自动减去零偏
        """
        ...

    def clear_bias(self) -> bool:
        """清除零偏，恢复原始数据"""
        ...

    def calibrate(
        self,
        calibration_matrix: np.ndarray,
        temperature_compensation: bool = True
    ) -> bool:
        """
        应用校准矩阵

        Args:
            calibration_matrix: 6x6 校准矩阵
            temperature_compensation: 是否启用温度补偿

        Returns:
            bool: 校准是否成功
        """
        ...

    def compensate_temperature(self, current_temp: float) -> Wrench:
        """
        温度补偿

        传感器在校准温度(通常25°C)下精度最高。
        温度变化会导致零漂和灵敏度变化。

        Args:
            current_temp: 当前温度 (°C)

        Returns:
            Wrench: 温度补偿后的力旋量
        """
        ...
```

### 3.4 AGV 五级力觉规格

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **轴数** | 3轴 | 6轴 | 6轴 | 6轴 | 6轴 |
| **Fx/Fy 范围** | ±100N | ±200N | ±500N | ±1000N | ±5000N |
| **Fz 范围** | ±200N | ±400N | ±1000N | ±2000N | ±10000N |
| **Tx/Ty/Tz 范围** | ±5Nm | ±10Nm | ±25Nm | ±50Nm | ±250Nm |
| **分辨率** | 0.1N | 0.05N | 0.02N | 0.01N | 0.005N |
| **采样率** | 100Hz | 500Hz | 1000Hz | 2000Hz | 5000Hz |
| **通信接口** | UART | CAN | EtherCAT | EtherCAT | EtherCAT+光纤 |
| **温度补偿** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **碰撞检测** | 基础 | 基础 | 精确 | 精确 | 精确 |
| **负载估计** | ✗ | ✓ | ✓ | ✓ | ✓ |

---

## 4. IMU 传感器模块接口 (IMUSensor)

### 4.1 基类接口

```python
class IMUSensor(ABC):
    """IMU 传感器抽象基类"""

    @abstractmethod
    def open(self) -> bool:
        """打开传感器连接"""
        pass

    @abstractmethod
    def close(self):
        """关闭传感器连接"""
        pass

    @abstractmethod
    def capture(self, timestamp: Optional[float] = None) -> IMUFrame:
        """采集一帧 IMU 数据"""
        pass

    @abstractmethod
    def calibrate_gyro(self, samples: int = 100) -> bool:
        """校准陀螺仪零偏 (静止假设)"""
        pass

    @abstractmethod
    def calibrate_accel(self, samples: int = 100) -> bool:
        """校准加速度计零偏 (水平假设)"""
        pass

    @abstractmethod
    def get_spec(self) -> Dict[str, Any]:
        """获取传感器规格"""
        pass
```

### 4.2 IMUFrame 数据结构

```python
@dataclass
class IMUFrame:
    """IMU 数据帧"""

    sensor_id: str                    # 传感器 ID
    timestamp: float                  # 时间戳 (秒)

    # 原始测量值
    accel: np.ndarray                 # 加速度 [ax, ay, az] m/s²
    gyro: np.ndarray                  # 角速度 [wx, wy, wz] rad/s
    mag: np.ndarray                   # 磁场 [mx, my, mz] μT (可选)

    # 姿态表示
    euler: np.ndarray                 # 欧拉角 [roll, pitch, yaw] rad
    quaternion: np.ndarray            # 四元数 [w, x, y, z]
    rotation_matrix: np.ndarray       # 旋转矩阵 3x3

    # 估计值
    linear_acceleration: np.ndarray   # 去除重力后的线加速度 m/s²
    angular_velocity_world: np.ndarray # 世界坐标系角速度 rad/s

    # 状态信息
    temperature: float                 # 温度 (°C)
    sensor_health: SensorHealth       # 健康状态
    calibration_age: float            # 校准时长 (秒)

    # ── 属性 ─────────────────────────────────────────────────
    @property
    def roll(self) -> float:
        """翻滚角 (roll) rad"""
        return self.euler[0]

    @property
    def pitch(self) -> float:
        """俯仰角 (pitch) rad"""
        return self.euler[1]

    @property
    def yaw(self) -> float:
        """偏航角 (yaw) rad"""
        return self.euler[2]

    @property
    def heading(self) -> float:
        """航向角 (0-360°)"""
        yaw_deg = np.degrees(self.yaw)
        return (yaw_deg + 360) % 360

    # ── 方法 ─────────────────────────────────────────────────
    def to_vector(self) -> np.ndarray:
        """
        转换为归一化特征向量

        格式: [ax_norm, ay_norm, az_norm,
               wx_norm, wy_norm, wz_norm,
               roll_norm, pitch_norm, yaw_norm]
        """
        ...

    def is_stationary(self, threshold: float = 0.1) -> bool:
        """判断是否静止 (基于加速度方差)"""
        ...
```

### 4.3 IMUSensor 具体实现接口

```python
class IMUSensor(ABC):
    """IMU 传感器基类"""

    def __init__(
        self,
        sensor_id: str = "imu_0",
        sensor_type: IMUSensorType = IMUSensorType.BMI088,
        sample_rate: float = 200.0,  # Hz
        orientation: str = "default"  # default / rotated_90 / rotated_180
    ):
        ...

    # ── 采集接口 ──────────────────────────────────────────────
    def capture(self, timestamp: Optional[float] = None) -> IMUFrame:
        """
        采集一帧 IMU 数据

        Returns:
            IMUFrame: 包含完整 IMU 信息的数据帧
        """
        ...

    def capture_burst(self, count: int) -> List[IMUFrame]:
        """
        突发采集多帧

        用于高频数据采集或滤波预处理

        Args:
            count: 采集帧数

        Returns:
            List[IMUFrame]: IMU 帧列表
        """
        ...

    # ── 校准接口 ──────────────────────────────────────────────
    def calibrate_gyro(self, samples: int = 100) -> bool:
        """
        校准陀螺仪零偏

        假设传感器在静止状态下采集 samples 个样本，
        计算均值作为零偏。

        Args:
            samples: 采样次数

        Returns:
            bool: 校准是否成功
        """
        ...

    def calibrate_accel(self, samples: int = 100) -> bool:
        """
        校准加速度计零偏

        假设传感器水平放置，Z轴应指向正上方 (g=9.81)

        Args:
            samples: 采样次数

        Returns:
            bool: 校准是否成功
        """
        ...

    def calibrate_mag(self, samples: int = 500) -> bool:
        """
        校准磁力计 (8字运动)

        Args:
            samples: 采样次数

        Returns:
            bool: 校准是否成功
        """
        ...

    def auto_calibrate(self, duration: float = 5.0) -> Dict[str, bool]:
        """
        自动完整校准

        Args:
            duration: 校准时长 (秒)

        Returns:
            Dict[str, bool]: 各轴校准结果
        """
        ...

    # ── 姿态估计接口 ──────────────────────────────────────────
    def estimate_attitude(
        self,
        method: str = "madgwick"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        估计姿态

        Args:
            method: 估计方法 (madgwick / complementary / kalman)

        Returns:
            (euler, quaternion): 欧拉角和四元数
        """
        ...
```

### 4.4 AGV 五级 IMU 规格

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **型号** | MPU6050 | BMI088 | BMI088 | ADIS16470 | ADIS16470×2 |
| **采样率** | 100Hz | 200Hz | 500Hz | 1000Hz | 2000Hz |
| **加速度范围** | ±8g | ±16g | ±24g | ±40g | ±40g |
| **陀螺仪范围** | ±1000°/s | ±2000°/s | ±4000°/s | ±4000°/s | ±4000°/s |
| **噪声密度** | 400μg/√Hz | 120μg/√Hz | 60μg/√Hz | 20μg/√Hz | 10μg/√Hz |
| **零偏稳定性** | ±20°/hr | ±5°/hr | ±5°/hr | ±0.5°/hr | ±0.3°/hr |
| **温度范围** | -40~85°C | -40~85°C | -40~105°C | -40~125°C | -40~125°C |
| **磁力计** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **气压计** | ✗ | ✗ | ✗ | ✓ | ✓ |
| **姿态估计精度** | ±2° | ±1° | ±0.5° | ±0.2° | ±0.1° |
| **通信接口** | I2C | SPI | SPI | SPI | SPI |

---

## 5. 控制器模块接口 (Controller)

### 5.1 控制器基类

```python
class Controller(ABC):
    """控制器抽象基类"""

    @abstractmethod
    def reset(self):
        """重置控制器状态"""
        pass

    @abstractmethod
    def set_reference(self, reference: np.ndarray):
        """设置参考值 (设定点)"""
        pass

    @abstractmethod
    def compute(self, measurement: np.ndarray, dt: float) -> np.ndarray:
        """
        计算控制输出

        Args:
            measurement: 当前测量值
            dt: 时间步长 (秒)

        Returns:
            np.ndarray: 控制输出
        """
        pass

    @abstractmethod
    def get_state(self) -> ControllerState:
        """获取控制器内部状态"""
        pass


@dataclass
class ControllerState:
    """控制器状态"""
    name: str
    is_running: bool
    tracking_error: float
    output: np.ndarray
    timestamp: float
    cycle_count: int
```

### 5.2 电机控制器接口

```python
class MotorController(Controller):
    """电机控制器"""

    def __init__(
        self,
        motor: Motor,
        controller_type: str = "pid",  # pid / pwm / velocity / position
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0
    ):
        ...

    def set_motor(self, motor: Motor):
        """绑定电机"""
        ...

    def set_gains(self, kp: float, ki: float, kd: float):
        """设置 PID 增益"""
        ...

    def compute(self, measurement: np.ndarray, dt: float) -> np.ndarray:
        """计算电机控制输出"""
        ...

    def enable(self):
        """使能控制器"""
        ...

    def disable(self):
        """禁用控制器"""
        ...

    def get_motor_state(self) -> MotorState:
        """获取电机状态"""
        ...
```

### 5.3 运动控制器接口

```python
class MotionController(Controller):
    """运动控制器 (轨迹跟踪)"""

    def __init__(
        self,
        kinematics: KinematicsModel,
        trajectory_tracker: TrajectoryTracker,
        safety_monitor: Optional[SafetyMonitor] = None
    ):
        ...

    def set_trajectory(self, trajectory: Trajectory):
        """设置轨迹"""
        ...

    def compute(
        self,
        current_pose: Pose2D,
        dt: float
    ) -> Tuple[float, float, float]:
        """
        计算控制输出

        Args:
            current_pose: 当前位姿 (x, y, theta)
            dt: 时间步长 (秒)

        Returns:
            (vx, vy, omega): 线速度 (m/s) 和角速度 (rad/s)
        """
        ...

    def is_trajectory_complete(self) -> bool:
        """检查轨迹是否完成"""
        ...

    def get_progress(self) -> float:
        """获取轨迹完成进度 [0, 1]"""
        ...
```

### 5.4 触觉伺服控制器接口

```python
class TactileServoController(Controller):
    """
    触觉伺服控制器

    基于触觉传感器的闭环力控制:
    - 检测接触
    - 调整抓取力
    - 防止滑移
    - 评估抓取质量
    """

    def __init__(
        self,
        tactile_sensor: TactileArray,
        motor_controller: MotorController,
        target_force: float = 5.0,   # N
        slip_threshold: float = 0.3,
        control_mode: str = "force"  # force / position / hybrid
    ):
        ...

    def set_target_force(self, force: float):
        """设置目标抓取力 (N)"""
        ...

    def set_target_position(self, position: float):
        """设置目标位置 (rad)"""
        ...

    def compute(
        self,
        tactile_frame: TactileFrame,
        dt: float
    ) -> float:
        """
        计算控制输出

        Args:
            tactile_frame: 当前触觉帧
            dt: 时间步长 (秒)

        Returns:
            float: 电机控制输出 (位置或 PWM)
        """
        ...

    def detect_and_respond(self, tactile_frame: TactileFrame) -> ControlResponse:
        """
        检测触觉事件并响应

        Returns:
            ControlResponse: 包含响应动作的数据类
        """
        ...
```

### 5.5 AGV 五级控制器规格

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **控制频率** | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| **控制模式** | 位置 | 位置+速度 | 位置+速度+阻抗 | 全模态+MPC | 全模态+MPC+自适应 |
| **关节位置精度** | ±0.5° | ±0.2° | ±0.1° | ±0.05° | ±0.02° |
| **力控精度** | ✗ | ±1N | ±0.5N | ±0.2N | ±0.1N |
| **碰撞响应时间** | >100ms | <50ms | <20ms | <10ms | <5ms |
| **姿态稳定时间** | <500ms | <200ms | <100ms | <50ms | <20ms |
| **实时性** | ✗ | ✗ | ✓ Xenomai | ✓ RT-PREEMPT | ✓ Xenomai+FPGA |
| **安全等级** | 基础 | 基础 | 增强 | 高级 | 最高 |
| **多机协同** | ✗ | ✗ | ✗ | ≤5台 | ≤20台 |

---

## 6. 传感器-控制器集成接口

### 6.1 SensorControllerBridge

```python
class SensorControllerBridge:
    """
    传感器-控制器桥接器

    负责:
    - 传感器数据预处理
    - 控制器输入组装
    - 控制输出分发
    - 时序同步
    """

    def __init__(
        self,
        sensors: List[Union[TactileSensor, ForceSensor, IMUSensor]],
        controllers: List[Controller],
        sync_period: float = 0.01  # 同步周期 (秒)
    ):
        self.sensors = sensors
        self.controllers = controllers
        self.sync_period = sync_period
        self._running = False
        ...

    def start(self):
        """启动桥接器 (开始同步循环)"""
        self._running = True
        ...

    def stop(self):
        """停止桥接器"""
        self._running = False
        ...

    def step(self, dt: float) -> BridgeOutput:
        """
        执行一步同步

        1. 采集所有传感器数据
        2. 预处理 (滤波、变换)
        3. 分发到各控制器
        4. 收集控制器输出

        Args:
            dt: 时间步长 (秒)

        Returns:
            BridgeOutput: 包含所有控制器输出的数据类
        """
        ...

    def get_fusion_data(self) -> np.ndarray:
        """
        获取融合后的多模态特征向量

        用于下游跨模态融合网络输入

        Returns:
            np.ndarray: 归一化特征向量
        """
        ...
```

### 6.2 集成流水线时序图

```
传感器采集 ──────────────────────────────────────────────────►
   │           │           │
   ▼           ▼           ▼
触觉帧      力旋量      IMU帧
   │           │           │
   ▼           ▼           ▼
预处理       预处理      预处理
(滤波)      (温度补偿)  (姿态估计)
   │           │           │
   └───────────┴───────────┘
                │
                ▼
         特征向量拼接
                │
                ▼
        ┌───────────────┐
        │ 跨模态融合网络 │
        │ CrossModalFusion │
        └───────────────┘
                │
                ▼
         融合特征向量
                │
       ┌────────┴────────┐
       ▼                 ▼
 触觉控制器          力觉控制器
 TactileServo     ForceController
       │                 │
       └────────┬────────┘
                ▼
         电机控制输出
                │
                ▼
           电机驱动
```

---

## 7. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v2.05.0 | 2026-04-09 | 新增完整传感器-控制器接口规范文档 |
| v2.04.0 | 2026-04-09 | 具身传感控制仿真测试完成, 1835项测试全通过 |
| v2.03.0 | 2026-04-09 | 新增具身智能实战演示脚本 |
| v2.02.1 | 2026-04-09 | 更新进度汇报脚本, 传感器融合集成测试完善 |
| v2.02.0 | 2026-04-08 | EmbodiedController.run() 仿真循环, AGV五级基准测试 |
| v1.53.0 | 2026-04-05 | 触觉/力觉/IMU模块终验完成 |
| v1.50.0 | 2026-04-03 | 传感器模块完善 |
| v1.39.0 | 2026-04-01 | 全模块完整验收确认 |

---

*文档版本: v2.05.0*
*生成时间: 2026-04-09*
*维护者: SuperModel Dev Team*
