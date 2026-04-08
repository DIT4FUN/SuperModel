# SuperModel 超模态大模型 - 架构设计文档

## 1. 项目概述

SuperModel 是一个超模态大模型具身智能系统，专注于 AGV（自动导引车）机器人的具身智能大脑开发。系统通过多模态感知融合实现对环境的全面理解，并具备自主学习和决策能力。

## 2. 系统架构

```
SuperModel
├── sensors/          # 传感器模块
│   ├── visual.py     # 视觉传感器 (已完成)
│   ├── audio.py      # 听觉传感器 (已完成)
│   ├── tactile.py    # 触觉传感器 (新增)
│   ├── force.py      # 力觉传感器 (新增)
│   └── imu.py        # IMU传感器 (新增)
├── fusion/           # 跨模态融合网络 (已完成)
├── control/          # 控制模块
│   ├── motor.py      # 电机控制 (新增)
│   └── motion.py     # 运动控制 (新增)
├── learning/         # 自主学习框架 (已完成)
├── simulation/       # 仿真环境
└── tests/           # 测试用例 (新增)
```

## 3. 传感器模块接口设计

### 3.1 触觉传感器 (tactile.py)

#### 类结构
```
TactileSensor (基类)
├── PressureSensor      # 压阻式压力传感器
├── TaxelArray          # 触感阵列(仿生皮肤)
└── PiezoelectricSensor # 压电式振动传感器

TactileArray            # 触觉传感器阵列管理
```

#### 接口规范
```python
class TactileSensor:
    def read(self, timestamp: float) -> TactileData
    def calibrate(self, reference_data: np.ndarray) -> bool
    def get_sensitivity(self) -> float

class TactileArray:
    def add_sensor(sensor: TactileSensor)
    def read_all(timestamp: float) -> List[TactileData]
    def get_fusion_data(timestamp: float) -> np.ndarray
    def detect_touch_distribution() -> Dict[str, float]
```

#### 数据格式
- `TactileData.to_vector()`: 返回归一化特征向量
- 压力单位: Pa (帕斯卡)
- 触感阵列: [rows, cols] numpy.ndarray

### 3.2 力觉传感器 (force.py)

#### 类结构
```
ForceSensor (基类)
├── SixAxisFTSensor    # 六维力传感器 (ATI风格)
└── SingleAxisForceSensor  # 单轴力传感器

ForceSensorArray        # 多力觉传感器管理
```

#### 接口规范
```python
class ForceSensor:
    def read(self, timestamp: float) -> ForceData
    def set_bias(current_reading: ForceData)
    def apply_calibration(raw_data: np.ndarray) -> np.ndarray

class SixAxisFTSensor:
    def compute_tcp_wrench(tcp_offset: np.ndarray) -> np.ndarray

class ForceSensorArray:
    def add_sensor(sensor: ForceSensor)
    def read_all(timestamp: float) -> List[ForceData]
    def get_net_wrench() -> np.ndarray  # 合成六维力
    def check_safety(force_thresh, torque_thresh) -> Dict
    def detect_contact(threshold: float) -> bool
```

#### 数据格式
- `ForceData.wrench`: [Fx, Fy, Fz, Mx, My, Mz] 单位: N, Nm
- 力矩补偿: M' = M + F × offset

### 3.3 IMU传感器 (imu.py)

#### 类结构
```
IMUSensor (基类)
├── BMI088      # 博世BMI088 (AGV常用)
└── MPU9250     # MPU9250 9轴IMU

IMUArray        # 多IMU管理
```

#### 接口规范
```python
class IMUSensor:
    def read(self, timestamp: float) -> IMUData
    def update_orientation(gyro_data: np.ndarray, dt: float)
    def get_euler_from_quaternion(q: np.ndarray) -> np.ndarray

class IMUArray:
    def add_sensor(sensor: IMUSensor)
    def read_all(timestamp: float) -> List[IMUData]
    def get_fusion_data() -> np.ndarray
    def estimate_pose_change(dt: float) -> Dict[str, float]
    def compute_heading() -> float
    def calibrate_gyro_bias(samples: int = 100)
```

#### 数据格式
- 加速度: m/s² [ax, ay, az]
- 角速度: rad/s [wx, wy, wz]
- 磁场: μT [mx, my, mz]
- 欧拉角: rad [roll, pitch, yaw]
- 四元数: [w, x, y, z]

## 4. 控制模块接口设计

### 4.1 电机控制 (motor.py)

#### 类结构
```
Motor (基类)
├── DCMotor        # 直流电机
├── BLDCmotor      # 无刷直流电机
├── ServoMotor     # 伺服舵机
└── StepperMotor   # 步进电机

MotorController     # 多电机控制器
```

#### 接口规范
```python
class Motor:
    def enable()
    def disable()
    def set_target(target: float, mode: MotorControlMode)
    def step(dt: float) -> MotorState
    def get_state() -> MotorState
    def get_position_rad() -> float
    def get_velocity_rpm() -> float

class MotorController:
    def add_motor(motor: Motor)
    def set_all_targets(targets: Dict[str, float], mode: MotorControlMode)
    def step_all(dt: float) -> Dict[str, MotorState]
    def emergency_stop()
```

### 4.2 运动控制 (motion.py)

#### 类结构
```
KinematicsModel (基类)
├── DifferentialDrive  # 差速驱动
└── MecanumDrive       # Mecanum全向驱动

TrajectoryPlanner      # 轨迹规划器
MotionController       # 运动控制器
AGVController          # AGV专用控制器
```

#### 接口规范
```python
class KinematicsModel:
    def forward(wheel_velocities: np.ndarray) -> Twist2D
    def inverse(twist: Twist2D) -> np.ndarray
    def integrate(pose: Pose2D, twist: Twist2D, dt: float) -> Pose2D

class AGVController:
    def set_target_pose(pose: Pose2D)
    def set_target_twist(twist: Twist2D)
    def step(dt: float) -> np.ndarray  # 返回轮速
    def move_to(x, y, theta, dt) -> np.ndarray
    def stop() -> np.ndarray
```

## 5. AGV五级规格表

| 等级 | 负载能力 | 导航方式 | 定位精度 | 典型场景 | 代表型号 |
|------|---------|---------|---------|---------|---------|
| **L1** | ≤500kg | 磁条/二维码 | ±10mm | 仓储拣选 | 潜伏式AGV |
| **L2** | 500-1500kg | 激光导航 | ±5mm | 产线配送 | 叉式AGV |
| **L3** | 1500-3000kg | SLAM视觉 | ±3mm | 柔性制造 | 复合AGV |
| **L4** | 3000-5000kg | 多传感器融合 | ±1mm | 重载车间 | 重载AGV |
| **L5** | >5000kg | 具身智能超模态 | <±0.5mm | 无人化工厂 | 超级AGV |

### L5级SuperModel核心规格

| 参数 | 规格 |
|------|------|
| **处理器** | NVIDIA Jetson AGX Orin / Tesla T4 |
| **AI算力** | ≥275 TOPS (INT8) |
| **传感器** | 深度相机 + 激光雷达 + IMU + 力觉 + 触觉 |
| **定位精度** | <±0.5mm (融合定位) |
| **导航速度** | 0-3m/s (自适应) |
| **负载能力** | 100-5000kg (模块化) |
| **安全标准** | ISO 3691-4, IEC 61508 SIL2 |
| **通讯协议** | WiFi 6E, 5G, MQTT |
| **续航能力** | 8-24h (视配置) |
| **多模态输入** | 视觉/听觉/触觉/力觉/IMU/位置 |
| **具身智能** | 超模态大模型 + RL自主学习 |

## 6. 仿真环境接口

### PyBullet 仿真

```python
from simulation.pybullet_sim import PyBulletSimulator
from simulation.agv_model_generator import generate_agv_urdf_detailed

# 创建仿真器
sim = PyBulletSimulator(gui=True)

# 生成AGV URDF
urdf = generate_agv_urdf_detailed('M', '2轮')

# 加载AGV
agv_id = sim.load_agv_model(urdf)

# 运行仿真
for i in range(1000):
    sim.step()
```

### 支持的AGV等级

| 等级 | 轮子配置 | 自重 | 负载 |
|------|----------|------|------|
| S | 2轮 | 15kg | 30kg |
| M | 2轮 | 35kg | 100kg |
| L | 4轮 | 80kg | 300kg |
| XL | 4轮 | 150kg | 600kg |
| XXL | 4轮 | 300kg | 1200kg |

## 7. 测试框架

### 运行测试
```bash
cd SuperModel
python -m pytest tests/ -v
python -m pytest tests/pybullet_sim_tests.py -v
```

### 测试覆盖
- 传感器模块: 单元测试 (114+项)
- 融合算法: 功能测试 + 稳定性测试 (185+项)
- 控制模块: 集成测试 (174+项)
- PyBullet仿真: URDF生成 + 物理模拟 (41项)

## 8. PID控制器详细接口 (pid.py)

### 类结构
```
PIDController          # 通用PID
├── compute()          # 位置式PID
├── compute_incremental()  # 增量式PID
├── set_tunings()      # 在线调参
└── reset()            # 状态重置

PIDController2D        # 二维PID（XY平面）
PIDAutotuner          # 自动整定（Ziegler-Nichols）
```

### PIDController 接口
```python
class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0,
                 output_limit=None, integral_limit=None,
                 derivative_filter=0.0, setpoint=0.0)
    def compute(error, dt) -> float      # 位置式
    def compute_incremental(error, dt) -> float  # 增量式
    def set_setpoint(setpoint)           # 设置目标
    def set_tunings(kp, ki, kd)           # 在线调参
    def reset()                           # 重置
    def get_state() -> Dict               # 获取内部状态
```

## 9. 安全监控详细接口 (safety.py)

### 类结构
```
SafetyLevel           # 安全等级枚举
StopReason            # 停止原因枚举
SafetyStatus          # 安全状态数据类
SafetyMonitor         # 安全监控器
EmergencyStopController  # 紧急停止控制器
```

### SafetyMonitor 接口
```python
class SafetyMonitor:
    def __init__(self, max_velocity=2.0, max_acceleration=1.0,
                 boundary_min=None, boundary_max=None,
                 force_threshold=100.0, torque_threshold=2.0)
    def check_velocity(velocity, dt, timestamp) -> SafetyStatus
    def check_boundary(position, timestamp) -> SafetyStatus
    def check_force(force_magnitude, torque_magnitude, timestamp) -> SafetyStatus
    def check_collision(collision_detected, timestamp) -> SafetyStatus
    def check_sensors(sensor_health, timestamp) -> SafetyStatus
    def check_all(velocity, position, force_magnitude, torque_magnitude,
                  collision_detected, sensor_health, dt, timestamp) -> SafetyStatus
    def emergency_stop(reason)             # 手动急停
    def reset_estop()                      # 重置急停
    def add_estop_callback(callback)        # 添加急停回调
    def is_safe() -> bool

class EmergencyStopController:
    def trigger(reason) -> bool            # 触发急停
    def reset(require_lockout=True) -> (bool, str)  # 重置
    def is_active() -> bool
    def get_history() -> List[(reason, timestamp)]
```

## 10. AGV五级规格表 (详细版)

| 等级 | 负载能力 | 导航方式 | 定位精度 | 通讯 | 安全标准 | 典型场景 |
|------|---------|---------|---------|------|---------|---------|
| **L1** | ≤500kg | 磁条/二维码 | ±10mm | 有线/低频WiFi | 基本 | 仓储拣选、电商物流 |
| **L2** | 500-1500kg | 激光导航 | ±5mm | WiFi/5GHz | ISO 3691-2 | 产线配送、工厂物流 |
| **L3** | 1500-3000kg | SLAM视觉 | ±3mm | WiFi 6/5G | ISO 3691-4 | 柔性制造、医药车间 |
| **L4** | 3000-5000kg | 多传感器融合 | ±1mm | 5G/MQTT | SIL2 | 重载车间、钢铁冶金 |
| **L5** | >5000kg | 超模态具身智能 | <±0.5mm | 5G/WiFi 6E | SIL2/功能安全 | 无人化工厂、智慧物流 |

### L5级SuperModel完整规格

| 参数 | 规格 | 备注 |
|------|------|------|
| **AI处理器** | NVIDIA Jetson AGX Orin | ≥275 TOPS INT8 |
| **AI算力** | ≥275 TOPS | 支持INT4/INT8/FP16 |
| **内存** | 32GB LPDDR5 | 支持更大模型 |
| **存储** | 64GB eMMC + NVMe | 高速数据存储 |
| **视觉** | 深度相机 + 4K广角 | RGB-D融合 |
| **激光雷达** | 360° 30m | 抗干扰 |
| **IMU** | BMI088 6轴 | 1000Hz采样 |
| **力觉** | ATI mini40 六维力 | 1000Hz |
| **触觉** | 16x16 触感阵列 | 1kHz |
| **麦克风** | 4麦克风阵列 | 远场拾音 |
| **定位精度** | <±0.5mm | 视觉+激光融合 |
| **导航速度** | 0-3m/s | 自适应 |
| **负载能力** | 100-5000kg | 模块化 |
| **安全标准** | ISO 3691-4, IEC 61508 | SIL2 |
| **通讯** | WiFi 6E, 5G, MQTT | 双链路备份 |
| **续航** | 8-24h | 视电池配置 |
| **充电** | 自动充电/换电 | 24h运行 |
| **防护** | IP54 | 工业级 |
| **工作温度** | -20°C to 50°C | 宽温 |
| **多模态** | 视觉/听觉/触觉/力觉/IMU | 全模态融合 |
| **具身智能** | 超模态大模型 + RL | 自主学习 |

---

## 附录F: AGV五级规格总表 (七大子系统全覆盖)

> 本附录汇总了 SuperModel 超模态大模型支持的全部 AGV 五级规格，为系统配置选型提供一键查询参考。

### F.1 基础规格总表

| 等级 | 负载 | 导航方式 | 定位精度 | 安全标准 | 控制频率 | 典型场景 | 代表型号 |
|------|------|---------|---------|---------|---------|---------|---------|
| **S** | ≤500kg | 磁条/二维码 | ±10mm | 基础软件限位 | 50Hz | 教育/实验室 | 潜伏式AGV |
| **M** | 500-1500kg | 激光导航 | ±5mm | ISO 3691-2 | 100Hz | 服务机器人 | 叉式AGV |
| **L** | 1500-3000kg | SLAM视觉 | ±3mm | ISO 3691-4 | 200Hz | 柔性制造 | 复合AGV |
| **XL** | 3000-5000kg | 多传感器融合 | ±1mm | IEC 61508 SIL2 | 500Hz | 重载车间 | 重载AGV |
| **XXL** | >5000kg | 超模态具身智能 | <±0.5mm | IEC 61508 SIL3 | 1000Hz | 无人化工厂 | 超级AGV |

### F.2 传感器子系统规格

| 等级 | 视觉 | 听觉 | 触觉 | 力觉 | IMU | 控制频率 |
|------|------|------|------|------|-----|---------|
| **S** | RealSense D435i | 单mic | 8x8, 50Hz | 3轴±100N | MPU6050 100Hz | 50Hz |
| **M** | RealSense D455 | 双耳阵列 | 16x16, 100Hz | 6轴±200N | BMI088 200Hz | 100Hz |
| **L** | 双目+深度 | 4-mic阵列 | 24x24, 200Hz | 6轴±500N | BMI088 500Hz | 200Hz |
| **XL** | 多目+LiDAR | 6-mic阵列 | 32x32, 500Hz | 6轴±1000N | ADIS16470 1000Hz | 500Hz |
| **XXL** | 多目+3D LiDAR | 8-mic阵列 | 48x48, 1000Hz | 6轴±5000N | ADIS16470 2000Hz | 1000Hz |

### F.3 运动控制子系统规格

| 等级 | 驱动方式 | 最大线速度 | 最大角速度 | 最大加速度 | 控制周期 |
|------|---------|---------|---------|---------|---------|
| **S** | 差速 | 0.5m/s | 1.5rad/s | 0.5m/s² | 20ms |
| **M** | 差速 | 1.0m/s | 2.0rad/s | 1.0m/s² | 10ms |
| **L** | 麦克纳姆 | 2.0m/s | 2.5rad/s | 2.0m/s² | 5ms |
| **XL** | 麦克纳姆 | 3.0m/s | 3.0rad/s | 3.0m/s² | 2ms |
| **XXL** | 麦克纳姆 | 5.0m/s | 3.5rad/s | 5.0m/s² | 1ms |

### F.4 安全子系统规格

| 等级 | 安全等级 | 碰撞检测 | 看门狗 | 故障容忍 | 急停响应 |
|------|---------|---------|--------|---------|---------|
| **S** | S | 软件限位 | 无 | 无 | 200ms |
| **M** | M | 速度监控 | 软看门狗 | 降级运行 | 100ms |
| **L** | L | 碰撞检测 | 实时看门狗 | 故障恢复 | 50ms |
| **XL** | XL | 融合检测 | 双冗余看门狗 | 故障容忍 | 20ms |
| **XXL** | XXL | 超模态感知 | 三冗余看门狗 | 功能安全SIL3 | 5ms |

### F.5 通信接口子系统规格

| 等级 | 有线接口 | 无线标准 | 工业总线 | 控制接口 | 远程控制 |
|------|--------|---------|---------|---------|---------|
| **S** | USB3.0, UART | WiFi 4 | CAN | ROS1 | 无 |
| **M** | USB3.0, ETH | WiFi 5 | CAN FD | ROS2 Humble | 基础 |
| **L** | ETH x2 | WiFi 6 | EtherCAT | ROS2 + MQTT | 云端 |
| **XL** | ETH x4 | 5G Sub-6 | EtherCAT+TSN | 双链路ROS2 | 云边协同 |
| **XXL** | ETH x8,光纤 | 5G mmWave | TSN+OPC-UA | 多主ROS2 | 全远程L4 |

### F.6 仿真与测试规格

| 等级 | 仿真平台 | 仿真精度 | 测试用例数 | 覆盖率目标 |
|------|---------|---------|-----------|-----------|
| **S** | Gazebo | 基础 | 200+ | 60% |
| **M** | Gazebo+MuJoCo | 标准 | 400+ | 70% |
| **L** | MuJoCo+Gym | 高精度 | 700+ | 80% |
| **XL** | MuJoCo+Isaac Gym | 物理精确 | 900+ | 85% |
| **XXL** | 全栈仿真+数字孪生 | 99%+物理精度 | 1100+ | 90% |

### F.7 AGV五级快速选型指南

```
选型流程:
1. 确定负载需求 (≤500kg / 500-1500 / 1500-3000 / 3000-5000 / >5000)
2. 确定精度需求 (±10mm / ±5mm / ±3mm / ±1mm / <±0.5mm)
3. 确定场景复杂度 (简单固定路线 / 动态障碍 / 多AGV协同 / 具身智能)
4. 确定安全等级要求

推荐配置:
- 教育实验 → S级 (RK3588 + RealSense + MPU6050)
- 服务机器人 → M级 (RK3588 + RealSense D455 + BMI088)
- 工业搬运 → L级 (RK3588x2 + 双目 + ADIS16470)
- 重载车间 → XL级 (RK3588集群 + 多目LiDAR + ADIS16470)
- 无人化工厂 → XXL级 (集群+GPU + 超模态感知 + 具身智能)
```

---

## 附录G: 详细模块接口设计规范 (v1.91.0)

### G.1 触觉传感器模块 (tactile.py) 完整接口

#### 核心类

```python
# --- 数据结构 ---
class TactileSensorType(Enum):
    RESISTIVE   # 电阻式 - I2C@0x18
    CAPACITIVE  # 电容式 - SPI@50MHz  
    PIEZOELECTRIC  # 压电式 - USB HID
    OPTICAL     # 光学式 - USB3.0

class TactileFrame:
    pressure_map: np.ndarray          # H×W, float32, 归一化 0-1
    temperature_map: np.ndarray       # H×W, float32, 摄氏度
    proximity: np.ndarray             # H×W, float32, 米
    slip_signal: np.ndarray           # H×W, float32, 0-1
    timestamp: float
    frame_id: int
    sensor_id: str

class TactileContact:
    center: Tuple[int, int]          # (row, col)
    area: int                         # 像素数
    peak_pressure: float
    mean_pressure: float
    centroid: Tuple[float, float]     # (row, col)
    contact_force: float              # N
    slip_probability: float           # 0-1
    temperature: float                # 摄氏度

class TactileCalibration:
    pressure_min/max: float
    temperature_range: Tuple[float, float]
    force_scale: float                # N 满量程
    offset_map: np.ndarray            # 偏置校正

class TactileArray:
    # 生命周期
    def open() -> bool
    def close()
    
    # 数据采集
    def capture() -> TactileFrame
    def detect_contacts(frame=None) -> List[TactileContact]
    def get_slip_signal(frame=None) -> np.ndarray
    
    # 质量评估
    def estimate_grip_quality(frame=None) -> Dict[str, float]
    
    # 标定
    def calibrate(zero_pressure=None, known_weights=None)
    
    # 虚拟传感器
class VirtualTactileSensor:
    def simulate_contact(pos, radius, force, noise) -> TactileFrame
    def simulate_sliding(direction, speed, frames) -> List[TactileFrame]
    def simulate_multi_contact(contacts) -> TactileFrame
    def simulate_slip_detection(normal_force, friction, velocity) -> Dict
```

#### 接口调用时序

```
TactileArray 生命周期:
  open() → capture() × N → detect_contacts() → estimate_grip_quality() → close()

物理层:
  仿真: generate_pressure_map() → apply_sensor_nonlinearity() → add_noise()
  硬件: I2C_read()/SPI_read() → raw2calibrated() → validate_range()
```

#### AGV五级触觉规格

| 等级 | 阵列 | 分辨率 | 压力范围 | 采样率 | 温度感知 |
|------|------|--------|---------|--------|---------|
| S | 8×8 | 12bit | 0-500kPa | 50Hz | ✗ |
| M | 16×16 | 12bit | 0-1000kPa | 100Hz | ✓ |
| L | 24×24 | 14bit | 0-2000kPa | 200Hz | ✓ |
| XL | 32×32 | 14bit | 0-5000kPa | 500Hz | ✓ |
| XXL | 48×48 | 16bit | 0-10000kPa | 1000Hz | ✓ |

### G.2 力觉传感器模块 (force.py) 完整接口

#### 核心类

```python
class ForceSensorType(Enum):
    SIX_AXIS       # 六维力矩 - ATI风格
    THREE_AXIS     # 三维力
    JOINT_TORQUE   # 关节力矩 - CAN/EtherCAT
    FINGER_TIP     # 手指尖力 - SPI/USB

class Wrench:
    force: np.ndarray      # 3, N
    torque: np.ndarray     # 3, N·m
    timestamp: float
    frame_id: int
    sensor_id: str
    
    @property def magnitude() -> float      # ||F||
    @property def torque_magnitude() -> float  # ||T||
    def to_vector() -> np.ndarray           # [Fx,Fy,Fz,Tx,Ty,Tz]
    def transform(R, t) -> Wrench            # 坐标变换

class ForceCalibration:
    bias: np.ndarray        # 6维偏置
    scale: np.ndarray       # 6维比例因子
    rotation_matrix: np.ndarray  # 3x3 旋转
    translation_offset: np.ndarray  # 3 平移
    force_range: Tuple[float, float]
    torque_range: Tuple[float, float]
    temp_coefficient: np.ndarray  # 温漂补偿

class ContactState:
    is_contact: bool
    contact_force: float
    contact_point: np.ndarray    # 3D
    normal_vector: np.ndarray     # 3D
    slip_probability: float

class ForceTorqueSensor:
    def open() -> bool
    def close()
    def capture() -> Wrench        # 6维力旋量
    def get_wrench() -> Wrench    # 最新数据
    def detect_contact(threshold=2.0) -> ContactState
    def estimate_payload() -> float  # kg
    def set_tool_center(mass, com)  # 重力补偿
    def calibrate_bias(samples=100)   # 零点校准

class WrenchProcessor:
    def filter(wrench) -> np.ndarray    # 指数移动平均
    def remove_outliers(wrench, history) -> np.ndarray
    def estimate_covariance(history) -> np.ndarray  # 6×6协方差
    def compute_force_direction(wrench) -> np.ndarray  # 归一化方向
    def compute_equivalent_wrench_at(wrench, translation) -> np.ndarray  # 等效变换

class VirtualForceSensor:
    def simulate_contact(force, torque, add_noise=True) -> Wrench
    def simulate_payload(mass, com_offset, gravity=9.81) -> Wrench
    def simulate_collision(direction, peak_force, duration_ms, decay) -> List[Wrench]
    def simulate_surface_contact(normal, point, penetration, stiffness, damping) -> Wrench
    def simulate_friction_contact(normal_force, velocity, coeff, mass) -> Wrench
```

#### 接口调用时序

```
ForceTorqueSensor 生命周期:
  open() → set_tool_center() → calibrate_bias() → capture() × N → close()

重力补偿:
  set_tool_center(mass, com)
    → 更新 calibration.bias = -torque_compensation
    → 下次 capture() 自动应用偏置补偿

接触检测:
  capture() → detect_contact(threshold=2.0N)
    → Wrench.magnitude > threshold → ContactState(is_contact=True)
```

### G.3 IMU传感器模块 (imu.py) 完整接口

#### 核心类

```python
class IMUSensorType(Enum):
    BMI088        # SPI@20MHz / I2C@400kHz, 高性能6轴
    MPU6050       # I2C@100kHz, 消费级6轴
    MPU9250       # I2C@400kHz, 9轴(含磁力计)
    ADIS16470     # SPI@40MHz, 工业级
    VIRTUAL       # 仿真/融合输出

class IMUFrame:
    accel: np.ndarray          # 3, m/s²
    gyro: np.ndarray            # 3, rad/s
    mag: np.ndarray             # 3, μT (可选)
    temperature: float          # 摄氏度
    timestamp: float
    frame_id: int
    sensor_id: str
    
    @property def accel_magnitude() -> float
    @property def gyro_magnitude() -> float

class Pose:
    position: np.ndarray   # 3, m
    orientation: np.ndarray  # 4, 四元数 [qw,qx,qy,qz]
    
    def to_euler() -> np.ndarray   # [roll, pitch, yaw] rad
    def to_matrix() -> np.ndarray   # 4×4变换矩阵
    @classmethod def identity() -> Pose
    @classmethod def from_euler(position, rpy) -> Pose

class IMUCalibration:
    accel_bias: np.ndarray   # 3
    gyro_bias: np.ndarray    # 3
    accel_scale: np.ndarray  # 3
    gyro_scale: np.ndarray   # 3
    mag_hard_iron: np.ndarray  # 3
    mag_soft_iron: np.ndarray  # 3×3

class IMUSensor:
    def open() -> bool
    def close()
    def capture() -> IMUFrame
    def self_test() -> bool
    def calibrate_gyro_bias(samples=500, duration_sec=5.0)
    def calibrate_accel(known_orientation="level")

class PoseEstimator:
    def __init__(algorithm="madgwick", sample_rate=200.0, beta=0.1)
    def update(accel, gyro, mag=None, dt=None) -> Pose
    def get_pose() -> Pose
    def get_euler() -> np.ndarray
    def get_rotation_matrix() -> np.ndarray
    def integrate_velocity(accel, dt, remove_gravity=True)
    def reset()

class VirtualIMUSensor:
    def open() -> bool
    def close()
    def simulate_static(orientation) -> IMUFrame
    def simulate_motion(linear_accel, angular_vel, dt) -> IMUFrame
    def simulate_trajectory(type, duration_s, dt) -> List[IMUFrame]
    def simulate_agv_motion(v_linear, omega, dt, grade) -> IMUFrame
    def simulate_human_walking(freq, speed, duration_s, dt) -> List[IMUFrame]
```

#### 姿态估计算法

```
Madgwick AHRS (默认):
  输入: accel[3], gyro[3], mag[3] (可选), dt
  输出: quaternion[4]
  原理: 梯度下降法融合加速度计和陀螺仪
  参数: beta=0.1 (收敛速度/精度权衡)
  
互补滤波:
  输入: accel[3], gyro[3], dt
  输出: quaternion[4]
  原理: alpha*gyro积分 + (1-alpha)*accel估算, alpha=0.98
  
卡尔曼滤波:
  输入: accel[3], gyro[3], dt
  输出: quaternion[4]
  原理: gyro积分预测 + accel修正
```

### G.4 控制模块 (control/) 接口概览

```
control/
├── motor.py          MotorController: 多电机统一管理
│   ├── add_motor(motor) / remove_motor(id)
│   ├── set_all_targets(targets, mode)
│   ├── step_all(dt) -> Dict[str, MotorState]
│   └── emergency_stop()
│
├── motion.py         MotionController: 运动学/动力学控制
│   ├── forward(wheel_velocities) -> Twist2D
│   ├── inverse(twist) -> np.ndarray
│   ├── integrate(pose, twist, dt) -> Pose2D
│   └── step(target_twist, dt) -> wheel_velocities
│
├── trajectory.py     轨迹规划: RRT / 样条 / S曲线
│   ├── plan(start, goal, algo) -> Trajectory
│   ├── smooth(trajectory) -> Trajectory
│   └── track(trajectory, state) -> wheel_commands
│
├── safety_controller.py  安全监控
│   ├── check_all(velocity, position, force, collision, sensors, dt)
│   ├── emergency_stop(reason)
│   └── is_safe() -> bool
│
├── agv.py           AGV专用控制
│   ├── AGVMotionController: 运动学/轨迹跟踪
│   ├── PurePursuitTracker / StanleyTracker / PIDTracker
│   └── get_agv_spec(grade) -> dict
│
├── supervisor.py    控制器监管 (生命周期/模式切换/故障恢复)
│   ├── GradeAwareSupervisor: 五级感知控制器
│   ├── SupervisorGrade: S/M/L/XL/XXL
│   └── get_supervisor_spec(grade)
│
├── embodied_control.py  具身智能控制 (最新)
│   ├── EmbodiedController: 感知-决策-控制闭环
│   ├── EmbodiedState: 具身状态 (触觉+力觉+IMU+视觉)
│   ├── EmbodiedCommand: 具身命令
│   ├── EmbodiedTaskExecutor: 任务执行器
│   └── EmbodiedGrade: S/M/L/XL/XXL 五级规格
│
├── mpc.py           模型预测控制
│   ├── JointSpaceMPC / CartesianMPC
│   └── get_mpc_spec(grade)
│
├── ros2_interface.py  ROS2 Humble 接口
│   ├── JointTrajectoryInterface
│   ├── TopicInterface / ServiceInterface / ActionInterface
│   └── ParameterInterface / ComponentInterface
│
└── obstacle_avoidance.py  避障算法
    ├── DynamicWindowApproach (DWA)
    ├── ArtificialPotentialField (APF)
    ├── VectorFieldHistogram (VFH)
    └── AvoidanceStrategy: STOP/HOLD/逃逸/减速
```

### G.5 具身智能控制模块 (embodied_control.py) 接口规范

```python
class EmbodiedState:
    """具身状态 - 融合所有感知模态"""
    tactile_contacts: List[TactileContact]  # 触觉接触
    wrench: Wrench                          # 六维力旋量
    imu_frame: IMUFrame                     # IMU数据
    pose: Pose                              # 姿态
    twist: Twist                            # 速度
    vision_features: np.ndarray            # 视觉特征
    audio_state: Dict                       # 听觉状态
    timestamp: float
    grade: EmbodiedGrade                    # 五级等级

class EmbodiedCommand:
    """具身命令"""
    target_velocity: Twist           # 目标速度
    target_wrench: Wrench           # 目标力 (力控时)
    control_mode: ControlMode        # 速度/位置/力/阻抗/混合
    safety_level: SafetyLevel
    skill_id: Optional[str]          # 技能ID

class EmbodiedController:
    """具身智能控制器 - 感知-决策-控制闭环"""
    
    def __init__(grade: EmbodiedGrade, config: EmbodiedControlParams)
    
    # 生命周期
    def open() -> bool
    def close()
    
    # 主循环
    def step(state: EmbodiedState, dt: float) -> EmbodiedCommand
    
    # 传感器融合
    def fuse_tactile_force(contacts, wrench) -> ContactQuality
    def fuse_imu_vision(imu, vision) -> PoseEstimate
    def predict_object_pose(tactile, vision) -> Pose
    
    # 任务执行
    def execute_grasp(target_pose, approach) -> GraspResult
    def execute_place(target_pose) -> PlaceResult
    def execute_push(direction, force) -> PushResult
    def execute_adjust(feedback) -> AdjustResult
    
    # 安全
    def check_safety(state) -> SafetyCheckResult
    def emergency_stop()

class EmbodiedControlParams:
    """具身控制参数"""
    fusion_algorithm: str           # "ekf" / "attention" / "late_fusion"
    control_frequency: int          # Hz
    tactile_threshold: float        # N
    force_threshold: float          # N
    imu_stability_window: int       # 帧数
    vision_confidence_threshold: float
    learning_rate: float           # 在线学习率
    grade: EmbodiedGrade

# 五级具身规格
AGV_EMBODIED_GRADES = {
    'S':  {'fusion': 'simple', 'freq': 50,  'grades': 3},
    'M':  {'fusion': 'ekf',   'freq': 100, 'grades': 6},
    'L':  {'fusion': 'attention', 'freq': 200, 'grades': 10},
    'XL': {'fusion': 'transformer', 'freq': 500, 'grades': 15},
    'XXL': {'fusion': 'foundation', 'freq': 1000, 'grades': 20},
}
```

### G.6 五级具身智能系统配置速查

```
EmbodiedGrade 定义:
  S  - 基础触觉+力觉反馈, 3级感知
  M  - 六维力矩+IMU融合, 6级感知  
  L  - 触觉+力觉+IMU+视觉协同, 10级感知
  XL - 超模态感知融合 + 在线学习, 15级感知
  XXL - 具身智能大脑 + 自主学习框架, 20级感知

每级核心差异:
  S:  触觉接触检测 → 抓取力限制
  M:  + IMU姿态稳定 + 力矩反馈
  L:  + 视觉引导抓取 + 多传感器协同
  XL: + 跨模态注意力融合 + 在线自适应
  XXL: + DreamerAgent自主学习 + 世界模型预测
```

