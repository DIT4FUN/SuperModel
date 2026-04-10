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

---

## 附录H: AGV五级规格总表 (v1.92.0 新增)

> 本附录提供 SuperModel 超模态大模型 AGV 五大等级（教育S / 标准M / 专业L / 高性能XL / 旗舰XXL）的完整规格对照，涵盖传感器、控制、融合、学习、仿真、硬件六大子系统的全部关键参数。

### H.1 快速选型对照表

| 维度 | S 教育级 | M 标准级 | L 专业级 | XL 高性能 | XXL 旗舰 |
|------|---------|---------|---------|----------|---------|
| **定位** | 教学/科研 | 室内服务 | 工业装配 | 高精度场景 | 全功能旗舰 |
| **控制频率** | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| **传感器** | 单目+IMU | 双目+力觉 | 多目+力觉+IMU | 多目+全传感器 | 多目+3D LiDAR+全传感器 |
| **触觉** | 8×8@50Hz | 16×16@100Hz | 24×24@200Hz | 32×32@500Hz | 48×48@1000Hz |
| **力觉** | 3轴@100Hz | 6轴@500Hz | 6轴@1000Hz | 6轴@2000Hz | 6轴@5000Hz |
| **IMU** | MPU6050@100Hz | BMI088@200Hz | BMI088@500Hz | ADIS16470@1000Hz | ADIS×2@2000Hz |
| **融合策略** | LATE | HYBRID | HYBRID | EARLY+HYBRID | EARLY+HYBRID+LATE |
| **NPU算力** | <5 TOPS | 5-20 TOPS | 20-100 TOPS | 100-300 TOPS | >300 TOPS |
| **典型价格** | ¥5-15K | ¥15-50K | ¥50-150K | ¥150-500K | >¥500K |

### H.2 感知子系统完整规格

#### H.2.1 触觉感知规格

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| **阵列尺寸** | 8×8 | 16×16 | 24×24 | 32×32 | 48×48 |
| **分辨率** | 12bit | 12bit | 14bit | 14bit | 16bit |
| **压力范围** | 0-500 kPa | 0-1000 kPa | 0-2000 kPa | 0-5000 kPa | 0-10000 kPa |
| **采样频率** | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| **温度感知** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **接近觉** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **滑移检测** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **编码器维度** | 32 | 64 | 64 | 128 | 128 |
| **模块类** | TactileArray | TactileArray | TactileArray | TactileArray | TactileArray |
| **虚拟传感器** | VirtualTactileSensor | VirtualTactileSensor | VirtualTactileSensor | VirtualTactileSensor | VirtualTactileSensor |

#### H.2.2 力觉感知规格

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| **轴数** | 3 | 6 | 6 | 6 | 6 |
| **力范围** | ±100N | ±200N | ±500N | ±1000N | ±5000N |
| **力矩范围** | ±10N·m | ±20N·m | ±50N·m | ±100N·m | ±500N·m |
| **分辨率** | 0.1N | 0.05N | 0.02N | 0.01N | 0.005N |
| **采样频率** | 100Hz | 500Hz | 1000Hz | 2000Hz | 5000Hz |
| **编码器维度** | 16 | 32 | 32 | 64 | 64 |
| **模块类** | ForceTorqueSensor | ForceTorqueSensor | ForceTorqueSensor | ForceTorqueSensor | ForceTorqueSensor |
| **虚拟传感器** | VirtualForceSensor | VirtualForceSensor | VirtualForceSensor | VirtualForceSensor | VirtualForceSensor |

#### H.2.3 IMU感知规格

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| **型号** | MPU6050 | BMI088 | BMI088 | ADIS16470 | ADIS16470×2 |
| **加速度量程** | ±8g | ±16g | ±24g | ±40g | ±80g |
| **陀螺量程** | ±1000°/s | ±2000°/s | ±4000°/s | ±4000°/s | ±8000°/s |
| **采样频率** | 100Hz | 200Hz | 500Hz | 1000Hz | 2000Hz |
| **噪声密度** | 400μg/√Hz | 120μg/√Hz | 60μg/√Hz | 20μg/√Hz | 10μg/√Hz |
| **磁力计** | ✗ | ✗/✓ | ✗ | ✓ | ✓ |
| **姿态算法** | 互补滤波 | Madgwick | Madgwick | Madgwick/KF | 扩展KF |
| **编码器维度** | 32 | 32 | 64 | 64 | 128 |
| **模块类** | IMUSensor | IMUSensor | IMUSensor | IMUSensor | IMUSensor |
| **虚拟传感器** | VirtualIMUSensor | VirtualIMUSensor | VirtualIMUSensor | VirtualIMUSensor | VirtualIMUSensor |

### H.3 控制子系统完整规格

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| **驱动类型** | 差速驱动 | 差速驱动 | Mecanum | Mecanum | Mecanum |
| **最大线速度** | 0.5m/s | 1.0m/s | 2.0m/s | 3.0m/s | 5.0m/s |
| **最大角速度** | 1.5rad/s | 2.0rad/s | 2.5rad/s | 3.0rad/s | 3.5rad/s |
| **控制频率** | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| **安全等级** | 软件限位 | ISO 3691-2 | ISO 3691-4 | IEC 61508 SIL2 | IEC 61508 SIL3 |
| **避障策略** | 传感器检测 | DWA | DWA+APF | VFH+DWA+APF | 混合+学习 |
| **控制模块** | MotorController | AGVController | AGVMotionController | AdaptiveMPCController | MultiAgentCoordinator |
| **力控模块** | — | ForceController | HybridForcePositionController | ImpedanceController | ForceImpedanceController |
| **触控模块** | TactileServoController | TactileServoController | TactileServoController | TactileServoController | GraspQualityController |
| **IMU控制** | AttitudeStabilizer | AttitudeStabilizer | AttitudeStabilizer | AttitudeStabilizer | MotionEstimator |
| **轨迹规划** | 线性插值 | 梯形速度 |五次多项式 | MINCO |学习型MINCO |

### H.4 仿真子系统规格

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| **仿真引擎** | NumPy | MuJoCo | PyBullet | PyBullet+Gazebo | 全栈仿真 |
| **物理步长** | 10ms | 5ms | 2ms | 1ms | 0.5ms |
| **渲染模式** | 无 | 软件渲染 | 硬件渲染 | 硬件渲染+ROS | 数字孪生 |
| **传感器仿真** | 简化模型 | 噪声模型 | 物理模型 | 物理+标定 | 物理+标定+漂移 |

### H.5 传感器-控制集成流水线路径

```
S级流水线:
  TactileArray → TactileServoController → MotorController → Motor

M级流水线:
  TactileArray + ForceTorqueSensor → TactileServoController + ForceController
  → HybridForcePositionController → MotorController → Motor

L级流水线:
  TactileArray + ForceTorqueSensor + IMUSensor
  → TactileServoController + ForceController + AttitudeStabilizer
  → HybridForcePositionController + AGVMotionController → MotorController → Motor

XL级流水线:
  TactileArray + ForceTorqueSensor + IMUSensor + BinocularCamera + LiDAR
  → TactileServoController + ForceController + AttitudeStabilizer + ObstacleAvoider
  → ImpedanceController + AdaptiveMPCController + AGVMotionController
  → MotorController → Motor

XXL级流水线:
  TactileArray + ForceTorqueSensor + IMUSensor + MultiViewCamera + 3D LiDAR + Audio
  → CrossModalFusion → EmbodiedController (DreamerAgent + WorldModel)
  → TactileServoController + ForceImpedanceController + MotionEstimator + Planner
  → MultiAgentCoordinator + Supervisor → MotorController → Motor
```

### H.6 AGV五级快速选型指南

| 需求场景 | 推荐等级 | 理由 |
|---------|---------|------|
| 教育/实验 | S | 成本低, 功能完整, 够用即可 |
| 服务机器人 | M | 性价比最高, 具备完整感知能力 |
| 工业装配 | L | 高精度, 强实时性, 多传感器协同 |
| 重载车间 | XL | 高性能, 大负载, 复杂避障 |
| 无人化工厂 | XXL | 全功能旗舰, 具身智能, 自主学习 |

选型步骤:
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



---

## 附录H: 模块接口规范 (Interface Specification)

### H.1 具身控制器接口 (EmbodiedController)

```python
from src.control.embodied_control import (
    EmbodiedController,
    EmbodiedControlParams,
    EmbodiedCommand,
    EmbodiedState,
    EmbodiedGrade,
    get_embodied_spec,
)

# ── 工厂方法创建 ──────────────────────────────
ctrl = EmbodiedController.create_for_grade(
    grade='M',           # AGV等级: 'S' | 'M' | 'L' | 'XL' | 'XXL'
    use_virtual=True,    # 使用虚拟传感器 (仿真模式)
    use_tactile=True,    # 启用触觉模态
    use_force=True,      # 启用力觉模态
    use_imu=True,        # 启用IMU模态
)

# ── 仿真循环 ─────────────────────────────────
result = ctrl.run(
    num_steps=100,       # 仿真步数
    cmd=EmbodiedCommand(mode='admittance')
)
# result = {
#     'states': List[EmbodiedState],   # 每步状态
#     'outputs': List[Dict],           # 每步控制输出
#     'slip_events': int,              # 滑移事件数
#     'contact_events': int,            # 接触事件数
#     'safety_stops': int,              # 安全停止次数
#     'avg_cycle_time_ms': float,      # 平均控制周期 (ms)
# }

# ── 五级基准测试 ─────────────────────────────
results = EmbodiedController.run_five_grade_benchmark(steps_per_grade=50)
# results = Dict[grade, result]  # 5个等级的仿真结果
```

### H.2 触觉传感器接口 (TactileArray)

```python
from src.sensors.tactile import (
    TactileArray, TactileSensorType, TactileFrame,
    TactileContact, VirtualTactileSensor
)

# ── 创建传感器 ───────────────────────────────
sensor = TactileArray(array_size=(16, 16), sensor_id="tactile_01")
sensor.open()

# ── 捕获触觉帧 ───────────────────────────────
frame: TactileFrame = sensor.capture()
# frame.pressure_map    # 压力分布图 [rows, cols], Pa
# frame.total_force     # 总压力, N
# frame.contact_center  # 接触中心 (row, col)
# frame.timestamp       # 时间戳

# ── 接触检测 ─────────────────────────────────
contacts: List[TactileContact] = sensor.detect_contacts(threshold=2.0)
# contact.position      # 接触位置 (row, col)
# contact.force         # 接触力, N
# contact.area          # 接触面积, m²

# ── 抓取质量评估 ─────────────────────────────
quality = sensor.assess_grasp_quality()
# quality.grasp_quality # 抓取质量 0~1
# quality.stability     # 稳定性指标 0~1
# quality.slip_risk      # 滑移风险 0~1

sensor.close()
```

### H.3 力觉传感器接口 (ForceTorqueSensor)

```python
from src.sensors.force import (
    ForceTorqueSensor, ForceSensorType, Wrench,
    VirtualForceSensor, ContactState
)

# ── 创建力传感器 ─────────────────────────────
sensor = ForceTorqueSensor(sensor_id="force_01")
sensor.open()

# ── 捕获六维力/力矩 ──────────────────────────
wrench: Wrench = sensor.capture()
# wrench.forces   # [Fx, Fy, Fz], N
# wrench.torques  # [Mx, My, Mz], N·m
# wrench.magnitude # 合力大小, N

# ── TCP力矩计算 ─────────────────────────────
tcp_wrench = sensor.compute_tcp_wrench(
    tcp_offset=np.array([0, 0, 0.05]),  # TCP偏移 [x, y, z], m
    measured_wrench=wrench
)

# ── 接触状态检测 ─────────────────────────────
contact: ContactState = sensor.detect_contact(force_threshold=2.0)
# contact.is_in_contact  # 是否接触
# contact.contact_force  # 接触力, N
# contact.sliding        # 是否滑移

sensor.close()
```

### H.4 IMU传感器接口 (IMUSensor)

```python
from src.sensors.imu import (
    IMUSensor, IMUSensorType, IMUFrame, Pose,
    VirtualIMUSensor, PoseEstimator
)

# ── 创建IMU ─────────────────────────────────
sensor = IMUSensor(sensor_type=IMUSensorType.VIRTUAL, sensor_id="imu_01")
sensor.open()

# ── 捕获IMU帧 ────────────────────────────────
frame: IMUFrame = sensor.capture()
# frame.accel   # 加速度 [ax, ay, az], m/s²
# frame.gyro    # 角速度 [wx, wy, wz], rad/s
# frame.quat    # 四元数 [w, x, y, z]

# ── 姿态估计 ────────────────────────────────
estimator = PoseEstimator(algorithm='madgwick', beta=0.1)
pose: Pose = estimator.update(accel, gyro, dt=0.01)
# pose.roll   # 横滚角, rad
# pose.pitch  # 俯仰角, rad
# pose.yaw    # 偏航角, rad

sensor.close()
```

### H.5 AGV运动控制器接口 (AGVMotionController)

```python
from src.control.agv import (
    AGVMotionController, AGVSpec, AGVGrade,
    AGVPose, AGVTwist, DifferentialDrive, MecanumDrive
)

# ── 创建AGV控制器 ───────────────────────────
spec = AGVSpec.from_grade(AGVGrade.M)
agv = AGVMotionController(spec)

# ── 更新位姿 ────────────────────────────────
agv.update_pose(AGVPose(x=0.0, y=0.0, theta=0.0))

# ── 计算轮速指令 ───────────────────────────
target = AGVPose(x=1.0, y=0.5, theta=0.0)
wheel_cmds = agv.compute_wheel_commands(target, dt=0.01)
# wheel_cmds: np.ndarray  # 轮子速度指令 [rad/s]

# ── 安全限幅 ───────────────────────────────
cmds_safe = agv.apply_safety_limits(wheel_cmds)
```

### H.6 安全控制器接口 (SafetyController)

```python
from src.control.safety_controller import (
    SafetyController, SafetyConfig, SafetyLevel, JointStateSnapshot
)

# ── 创建安全控制器 ─────────────────────────
config = SafetyConfig(
    joint_limits_lower=-np.ones(6) * np.pi,
    joint_limits_upper=np.ones(6) * np.pi,
    velocity_limits=np.ones(6) * 3.0,
    acceleration_limits=np.ones(6) * 10.0,
    torque_limits=np.ones(6) * 100.0,
    safety_level=SafetyLevel.M,
)
safety = SafetyController(config)

# ── 安全检查 ───────────────────────────────
state = JointStateSnapshot(
    positions=np.zeros(6),
    velocities=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    accelerations=np.zeros(6),
    torques=np.zeros(6),
    timestamp=time.time(),
)
result = safety.check(state)
# result.safe              # 是否安全
# result.violated_limits   # 违反的限制列表

# ── 安全速度计算 ───────────────────────────
safe_vel = safety.compute_safe_velocity(
    current_vel=np.array([0.5]*6),
    desired_vel=np.array([10.0]*6),
)
```

### H.7 轨迹规划接口 (TrajectoryPlanner)

```python
from src.control.trajectory import (
    TrajectoryPlanner, PlanningAlgorithm, VelocityProfile,
    VelocityProfiler
)

# ── 创建规划器 ─────────────────────────────
planner = TrajectoryPlanner(
    algorithm=PlanningAlgorithm.RRT_STAR,
    bounds=((0, 5), (0, 5)),
)

# ── 路径规划 ───────────────────────────────
path = planner.plan(
    start=(0.0, 0.0),
    goal=(4.0, 4.0),
)
# path: List[Tuple[x, y]]  # 规划路径点

# ── 速度规划 ───────────────────────────────
profiler = VelocityProfiler(
    max_v=1.0, max_a=0.5, max_j=2.0,
    profile_type=VelocityProfile.TRAPEZOIDAL,
)
positions, velocities = profiler.plan(distance=1.0, v0=0.0, v1=0.0)
```

---

## 附录I: 传感器模块完整接口规范 (v2.16.0)

> 本附录详细定义触觉、力觉、IMU三大新增传感器模块的完整接口，
> 包括类结构、核心方法、输入输出格式、AGV五级规格对照。

---

### I.1 触觉传感器模块 (tactile.py)

#### I.1.1 类层次结构

```
TactileArray                    # 主类：电子皮肤触觉阵列
├── TactileFrame                # 数据结构：触觉帧
├── TactileContact              # 数据结构：接触事件
├── TactileCalibration          # 数据结构：标定参数
├── PressureProcessor           # 信号处理器
├── VirtualTactileSensor       # 仿真传感器
└── AGV_TACTILE_GRADES         # 五级规格表
```

#### I.1.2 TactileArray 主类

```python
class TactileArray:
    """
    电子皮肤触觉阵列接口
    
    参数:
        array_size: (rows, cols) - 触觉阵列尺寸
        sensor_type: TactileSensorType (RESISTIVE/CAPACITIVE/PIEZOELECTRIC/OPTICAL)
        sensor_id: 传感器标识符
        calibration: TactileCalibration 标定参数
    """
    
    # === 生命周期 ===
    def open() -> bool:
        """打开传感器，建立连接（I2C/SPI/USB/CAN）"""
        # 仿真模式：初始化模拟状态
        # 硬件模式：建立真实传感器连接
        return True
    
    def close():
        """关闭传感器连接"""
    
    def __enter__() -> "TactileArray": ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...
    
    # === 数据采集 ===
    def capture() -> TactileFrame:
        """
        捕获一帧触觉数据
        Returns: TactileFrame
            - pressure_map: np.ndarray (H×W), 归一化压力 0-1
            - temperature_map: Optional[np.ndarray] (H×W), 摄氏度
            - proximity: Optional[np.ndarray] (H×W), 接近距离(m)
            - slip_signal: Optional[np.ndarray] (H×W), 滑移信号
            - timestamp: float
            - frame_id: int
            - sensor_id: str
        """
    
    # === 接触检测 ===
    def detect_contacts(frame: Optional[TactileFrame] = None) -> List[TactileContact]:
        """
        检测接触区域
        Returns: List[TactileContact]
            - center: (row, col) 接触中心
            - area: int 接触面积(像素数)
            - peak_pressure: float 峰值压力
            - mean_pressure: float 平均压力
            - centroid: (row, col) 压力质心
            - contact_force: float 估计接触力(N)
            - slip_probability: float 滑移概率
            - temperature: Optional[float] 接触区温度
        """
    
    # === 滑移检测 ===
    def get_slip_signal(frame: Optional[TactileFrame] = None) -> np.ndarray:
        """
        计算滑移信号（多尺度滑移检测）
        - 压力梯度变化检测
        - 高频振动成分检测
        - 多帧历史分析
        Returns: np.ndarray (H×W) 归一化滑移信号 0-1
        """
    
    # === 抓取质量评估 ===
    def estimate_grip_quality(frame: Optional[TactileFrame] = None) -> Dict[str, float]:
        """
        估计抓取质量
        Returns: {
            'overall': float,      # 综合评分 0-1
            'contact_area': float, # 接触面积评分
            'uniformity': float,   # 均匀性评分
            'stability': float     # 稳定性评分
        }
        """
    
    # === 标定 ===
    def calibrate(
        zero_pressure: Optional[np.ndarray] = None,
        known_weights: Optional[List[float]] = None
    ):
        """传感器标定（零压力基准 + 力-电压标定）"""
```

#### I.1.3 TactileSensorType 枚举

```python
class TactileSensorType(Enum):
    RESISTIVE    = "resistive"    # 电阻式 - I2C@0x18, 12bit ADC
    CAPACITIVE   = "capacitive"   # 电容式 - SPI@50MHz, 14bit ADC, 支持接近觉
    PIEZOELECTRIC = "piezoelectric" # 压电式 - USB HID, 14bit, 高频振动
    OPTICAL      = "optical"      # 光学式 - USB3.0, 16bit, 最高精度
```

#### I.1.4 VirtualTactileSensor 仿真类

```python
class VirtualTactileSensor:
    """仿真触觉传感器，用于仿真环境和算法验证"""
    
    def open() -> bool: ...
    def close(): ...
    
    def simulate_contact(
        contact_pos: Tuple[float, float],  # 归一化 (0-1)
        contact_radius: float = 0.3,       # 归一化
        contact_force: float = 10.0,        # N
        noise_level: float = 0.05
    ) -> TactileFrame:
        """模拟接触事件（高斯压力分布）"""
    
    def simulate_sliding(
        direction: Tuple[float, float],   # (dx, dy) 归一化
        speed: float = 0.1,               # 归一化/帧
        duration_frames: int = 30
    ) -> List[TactileFrame]:
        """模拟滑移动作"""
    
    def simulate_multi_contact(
        contacts: List[Tuple[position, force, radius]],
        noise_level: float = 0.05
    ) -> TactileFrame:
        """模拟多点接触"""
    
    def simulate_slip_detection(
        normal_force: float,              # N
        friction_coeff: float,             # 摩擦系数
        velocity: Tuple[float, float]      # 滑移速度
    ) -> Dict[str, float]:
        """
        模拟滑移检测
        Returns: {
            'slip_state': 'stick' | 'micro_slip' | 'sliding',
            'slip_probability': float,
            'friction_force': float,
            'velocity_magnitude': float,
            'max_static_friction': float
        }
        """
```

#### I.1.5 触觉传感器 AGV 五级规格表

```python
AGV_TACTILE_GRADES = {
    'S':   {'array': (8, 8),    'res': 12, 'range_kpa': (0, 500),    'freq_hz': 50,   'temp': False},
    'M':   {'array': (16, 16),  'res': 12, 'range_kpa': (0, 1000),   'freq_hz': 100,  'temp': True},
    'L':   {'array': (24, 24),  'res': 14, 'range_kpa': (0, 2000),   'freq_hz': 200,  'temp': True},
    'XL':  {'array': (32, 32),  'res': 14, 'range_kpa': (0, 5000),   'freq_hz': 500,  'temp': True},
    'XXL': {'array': (48, 48),  'res': 16, 'range_kpa': (0, 10000),  'freq_hz': 1000, 'temp': True},
}
# 用法: spec = get_tactile_spec('M') → {'array': (16, 16), 'res': 12, ...}
```

---

### I.2 力觉传感器模块 (force.py)

#### I.2.1 类层次结构

```
ForceTorqueSensor               # 主类：六维力矩传感器
├── Wrench                      # 数据结构：力旋量 (F, T)
├── ForceCalibration            # 数据结构：标定参数
├── ContactState                # 数据结构：接触状态
├── WrenchProcessor             # 信号处理器
├── VirtualForceSensor          # 仿真传感器
└── AGV_FORCE_GRADES           # 五级规格表
```

#### I.2.2 Wrench 力旋量数据结构

```python
@dataclass
class Wrench:
    """
    力旋量 (力与力矩的组合)
    
    属性:
        force: np.ndarray (3,) - 力向量 [Fx, Fy, Fz] 单位: N
        torque: np.ndarray (3,) - 力矩向量 [Tx, Ty, Tz] 单位: N·m
        timestamp: float
        frame_id: int
        sensor_id: str
    """
    
    @property
    def magnitude(self) -> float:
        """力向量大小 ||F|| = sqrt(Fx²+Fy²+Fz²)"""
    
    @property
    def torque_magnitude(self) -> float:
        """力矩大小 ||T||"""
    
    def to_vector(self) -> np.ndarray:
        """转换为6维向量 [Fx, Fy, Fz, Tx, Ty, Tz]"""
    
    @classmethod
    def from_vector(cls, vec: np.ndarray, **kwargs) -> "Wrench":
        """从6维向量创建"""
    
    def transform(self, rotation: np.ndarray, translation: np.ndarray) -> "Wrench":
        """
        坐标变换（传感器坐标系 → 世界坐标系）
        - new_force = R @ force
        - new_torque = R @ torque + translation × new_force
        """
```

#### I.2.3 ForceTorqueSensor 主类

```python
class ForceTorqueSensor:
    """
    六维力矩传感器接口（ATI风格）
    
    参数:
        sensor_type: ForceSensorType (SIX_AXIS/THREE_AXIS/JOINT_TORQUE/FINGER_TIP)
        sensor_id: 传感器标识符
        calibration: ForceCalibration 标定参数
        ip_address: 网口传感器IP (如 ATI Net F/T)
        ethernet_type: "UDP" | "TCP"
    """
    
    # === 生命周期 ===
    def open() -> bool:
        """打开传感器（网络/USB HID/CAN总线）"""
        # 仿真模式：初始化模拟状态
        # 硬件模式：建立真实传感器连接
        return True
    
    def close(): ...
    
    # === 数据采集 ===
    def capture() -> Wrench:
        """
        采集一帧力数据（基于物理模型仿真）
        - 重力补偿
        - 工具中心偏移力矩
        - 环境扰动
        - 传感器噪声
        - 温漂
        """
    
    def get_wrench() -> Optional[Wrench]:
        """获取最新力数据"""
    
    # === 接触检测 ===
    def detect_contact(
        wrench: Optional[Wrench] = None,
        threshold: Optional[float] = None  # 默认 2.0 N
    ) -> ContactState:
        """
        Returns: ContactState
            - is_contact: bool
            - contact_force: float
            - contact_point: Optional[np.ndarray] (3D)
            - normal_vector: Optional[np.ndarray]
            - slip_probability: float
        """
    
    # === 负载估计 ===
    def estimate_payload(wrench: Optional[Wrench] = None) -> float:
        """基于静止状态重力分量估计负载重量 (kg)"""
    
    # === 工具中心设置 ===
    def set_tool_center(tool_mass: float, tool_com: np.ndarray):
        """
        设置工具中心参数（用于重力补偿）
        tool_mass: 工具质量 kg
        tool_com: 工具质心在传感器坐标系中的位置 m
        """
    
    # === 标定 ===
    def calibrate_bias(num_samples: int = 100):
        """偏置校准（无负载状态采集零点）"""
```

#### I.2.4 VirtualForceSensor 仿真类

```python
class VirtualForceSensor:
    """仿真力觉传感器"""
    
    def open() -> bool: ...
    def close(): ...
    
    def simulate_contact(
        force: Tuple[float, float, float],      # (Fx, Fy, Fz) N
        torque: Tuple[float, float, float] = (0,0,0), # (Tx, Ty, Tz) N·m
        add_noise: bool = True
    ) -> Wrench:
        """模拟接触力"""
    
    def simulate_payload(
        mass: float = 1.0,                          # kg
        com_offset: Tuple[float, float, float] = (0,0,0), # m
        gravity: float = 9.81                        # m/s²
    ) -> Wrench:
        """模拟负载重力: Fz=-mg, Tx=mg*dy, Ty=-mg*dx"""
    
    def simulate_collision(
        direction: Tuple[float, float, float],  # 碰撞方向(归一化)
        peak_force: float = 50.0,               # N
        duration_ms: float = 100.0,              # 毫秒
        decay: str = "exponential"               # 'exponential' | 'linear'
    ) -> List[Wrench]:
        """模拟碰撞事件（峰值→衰减过程）"""
    
    def simulate_surface_contact(
        surface_normal: Tuple[float, float, float] = (0,0,1),
        contact_point: Tuple[float, float, float] = (0,0,0),
        penetration_depth: float = 0.001,        # m
        stiffness: float = 1000.0,               # N/m
        damping: float = 50.0                   # N·s/m
    ) -> Wrench:
        """模拟表面接触（弹簧阻尼模型）"""
    
    def simulate_friction_contact(
        normal_force: float,
        velocity: Tuple[float, float, float],
        friction_coeff: float = 0.3,
        object_mass: float = 1.0
    ) -> Wrench:
        """模拟摩擦力（库仑摩擦模型）"""
```

#### I.2.5 WrenchProcessor 信号处理器

```python
class WrenchProcessor:
    """力旋量信号处理器"""
    
    def __init__(self, filter_alpha: float = 0.3, outlier_threshold: float = 3.0):
        """
        filter_alpha: 指数移动平均系数
        outlier_threshold: 异常值倍数(标准差)
        """
    
    def filter(wrench: np.ndarray, return_wrench: bool = False):
        """指数移动平均滤波"""
    
    def remove_outliers(wrench: np.ndarray, history: List[np.ndarray]) -> np.ndarray:
        """基于历史数据去除异常值"""
    
    def estimate_covariance(history: List[np.ndarray]) -> np.ndarray:
        """估计测量协方差矩阵 (6×6)"""
    
    def compute_force_direction(wrench: np.ndarray) -> np.ndarray:
        """计算归一化力向量方向"""
    
    def compute_equivalent_wrench_at(wrench: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """计算等效到指定点的力旋量: T' = T + r × F"""
```

#### I.2.6 力觉传感器 AGV 五级规格表

```python
AGV_FORCE_GRADES = {
    'S':   {'axes': 3, 'force_range': 100,   'torque_range': 10,    'resolution': 0.1,   'sampling_hz': 100},
    'M':   {'axes': 6, 'force_range': 200,   'torque_range': 20,    'resolution': 0.05,  'sampling_hz': 500},
    'L':   {'axes': 6, 'force_range': 500,   'torque_range': 50,    'resolution': 0.02,  'sampling_hz': 1000},
    'XL':  {'axes': 6, 'force_range': 1000,  'torque_range': 100,  'resolution': 0.01,  'sampling_hz': 2000},
    'XXL': {'axes': 6, 'force_range': 5000,  'torque_range': 500,  'resolution': 0.005, 'sampling_hz': 5000},
}
```

---

### I.3 IMU传感器模块 (imu.py)

#### I.3.1 类层次结构

```
IMUSensor                        # 主类：惯性测量单元
├── IMUFrame                     # 数据结构：IMU数据帧
├── IMUCalibration               # 数据结构：标定参数
├── Pose                         # 数据结构：位姿
├── PoseEstimator                # 姿态估计器
├── VirtualIMUSensor             # 仿真传感器
└── AGV_IMU_GRADES              # 五级规格表
```

#### I.3.2 IMUFrame 数据帧

```python
@dataclass
class IMUFrame:
    """
    IMU数据帧
    
    属性:
        accel: np.ndarray (3,) - 加速度 m/s²
        gyro: np.ndarray (3,)  - 角速度 rad/s
        mag: Optional[np.ndarray] (3,) - 磁力计 μT
        temperature: float - 摄氏度
        timestamp: float
        frame_id: int
        sensor_id: str
    """
    
    @property
    def accel_magnitude(self) -> float:
        """加速度向量模长 ||accel||"""
    
    @property
    def gyro_magnitude(self) -> float:
        """角速度向量模长 ||gyro||"""
```

#### I.3.3 Pose 位姿数据结构

```python
@dataclass
class Pose:
    position: np.ndarray    # (3,) 位置 m
    orientation: np.ndarray  # (4,) 四元数 [qw, qx, qy, qz]
    
    @classmethod
    def identity(cls) -> "Pose":
        """单位位姿: position=(0,0,0), orientation=(1,0,0,0)"""
    
    def to_euler(self) -> np.ndarray:
        """转欧拉角 [roll, pitch, yaw] rad"""
    
    def to_matrix(self) -> np.ndarray:
        """转4×4变换矩阵"""
    
    @classmethod
    def from_euler(cls, position: np.ndarray, rpy: np.ndarray) -> "Pose":
        """从欧拉角创建姿态"""
```

#### I.3.4 IMUSensor 主类

```python
class IMUSensor:
    """
    IMU传感器接口（BMI088/MPU6050/MPU9250/ADIS16470）
    
    参数:
        sensor_type: IMUSensorType
        sensor_id: 传感器标识符
        calibration: IMUCalibration 标定参数
        accel_range: int 加速度量程 (g)
        gyro_range: int 陀螺仪量程 (deg/s)
        sample_rate: int 采样频率 Hz
    """
    
    # === 生命周期 ===
    def open() -> bool:
        """打开传感器（I2C/SPI/USB/ROS串口）"""
        return True
    
    def close(): ...
    
    # === 数据采集 ===
    def capture() -> IMUFrame:
        """
        采集一帧IMU数据（基于物理模型仿真）
        - 重力向量（传感器坐标系）
        - 运动引起的比力
        - 角度变化趋势
        - 传感器噪声（符合各型号规格）
        - 偏置稳定性（慢漂移）
        - 温度（环境+自发热）
        - 磁力计（MPU9250/9轴IMU）
        """
    
    # === 自检 ===
    def self_test() -> bool:
        """
        传感器自检
        - 检查加速度范围（应接近1g）
        - 检查角速度范围（应接近0）
        """
    
    # === 标定 ===
    def calibrate_gyro_bias(num_samples: int = 500, duration_sec: float = 5.0):
        """陀螺仪偏置校准（静止状态采集）"""
    
    def calibrate_accel(known_orientation: str = "level"):
        """
        加速度计标定
        known_orientation: "level"|"up"|"down"|"left"|"right"|"front"|"back"
        """
```

#### I.3.5 PoseEstimator 姿态估计器

```python
class PoseEstimator:
    """
    姿态估计器
    
    参数:
        algorithm: "madgwick" | "complementary" | "kalman"
        sample_rate: float Hz
        beta: float Madgwick滤波器增益
    """
    
    def update(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        mag: Optional[np.ndarray] = None,
        dt: Optional[float] = None
    ) -> Pose:
        """更新姿态估计"""
    
    def get_pose(self) -> Pose:
        """获取当前姿态"""
    
    def get_euler(self) -> np.ndarray:
        """获取当前欧拉角 [roll, pitch, yaw] rad"""
    
    def get_rotation_matrix(self) -> np.ndarray:
        """获取当前旋转矩阵 (3×3)"""
    
    def integrate_velocity(
        self, accel: np.ndarray, dt: float, remove_gravity: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        积分加速度获得速度/位置
        Warning: 漂移严重，仅短时间有效
        Returns: (velocity, position)
        """
    
    def reset():
        """重置积分状态（速度/位置/四元数）"""
```

#### I.3.6 VirtualIMUSensor 仿真类

```python
class VirtualIMUSensor:
    """仿真IMU传感器"""
    
    def open() -> bool: ...
    def close(): ...
    
    def simulate_static(
        orientation: Tuple[float, float, float] = (0,0,0)  # roll, pitch, yaw rad
    ) -> IMUFrame:
        """模拟静止状态（重力对齐加速度）"""
    
    def simulate_motion(
        linear_accel: Tuple[float, float, float],
        angular_vel: Tuple[float, float, float],
        dt: float = 0.01
    ) -> IMUFrame:
        """模拟运动状态"""
    
    def simulate_trajectory(
        trajectory_type: str = "circle",  # "circle"|"figure8"|"linear"|"sine"
        duration_s: float = 2.0,
        dt: float = 0.01
    ) -> List[IMUFrame]:
        """模拟典型轨迹"""
    
    def simulate_agv_motion(
        linear_velocity: Tuple[float, float] = (0,0),  # (vx, vy) m/s
        angular_velocity: float = 0.0,               # rad/s
        dt: float = 0.01,
        grade: str = "M"                            # "S"|"M"|"L"|"XL"|"XXL"
    ) -> IMUFrame:
        """模拟AGV运动（考虑不同等级IMU噪声特性）"""
    
    def simulate_human_walking(
        step_frequency: float = 1.5,  # Hz
        walk_speed: float = 1.0,       # m/s
        duration_s: float = 5.0,
        dt: float = 0.01
    ) -> List[IMUFrame]:
        """模拟人类步行运动"""
```

#### I.3.7 IMU AGV 五级规格表

```python
AGV_IMU_GRADES = {
    'S':   {'type': 'MPU6050',   'accel_range': 8,    'gyro_range': 1000,  'sample_hz': 100,  'noise_density': 400},
    'M':   {'type': 'BMI088',    'accel_range': 16,   'gyro_range': 2000,  'sample_hz': 200,  'noise_density': 120},
    'L':   {'type': 'BMI088',    'accel_range': 24,   'gyro_range': 4000,  'sample_hz': 500,  'noise_density': 60},
    'XL':  {'type': 'ADIS16470', 'accel_range': 40,   'gyro_range': 4000,  'sample_hz': 1000, 'noise_density': 20},
    'XXL': {'type': 'ADIS16470', 'accel_range': 80,   'gyro_range': 8000,  'sample_hz': 2000, 'noise_density': 10},
}
```

---

### I.4 触觉/力觉/IMU 五级综合规格速查表

| 规格项 | S级 | M级 | L级 | XL级 | XXL级 |
|--------|:---:|:---:|:---:|:---:|:---:|
| **触觉阵列** | 8×8 | 16×16 | 24×24 | 32×32 | 48×48 |
| **触觉ADC** | 12bit | 12bit | 14bit | 14bit | 16bit |
| **触觉采样** | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| **力觉轴数** | 3轴 | 6轴 | 6轴 | 6轴 | 6轴 |
| **力觉范围** | ±100N | ±200N | ±500N | ±1000N | ±5000N |
| **力觉采样** | 100Hz | 500Hz | 1000Hz | 2000Hz | 5000Hz |
| **IMU型号** | MPU6050 | BMI088 | BMI088 | ADIS16470 | ADIS16470 |
| **IMU采样** | 100Hz | 200Hz | 500Hz | 1000Hz | 2000Hz |
| **IMU噪声** | 400μg/√Hz | 120μg/√Hz | 60μg/√Hz | 20μg/√Hz | 10μg/√Hz |
| **融合编码器** | 128d | 256d | 512d | 768d | 1024d |
| **控制频率** | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| **闭环延迟** | <200ms | <80ms | <35ms | <15ms | <7ms |

---

### I.5 典型使用流程

```python
# === 触觉 + 力觉 + IMU 联合使用 ===
from src.sensors.tactile import TactileArray, TactileSensorType, get_tactile_spec
from src.sensors.force import ForceTorqueSensor, ForceSensorType, get_force_spec
from src.sensors.imu import IMUSensor, IMUSensorType, get_imu_spec, PoseEstimator

AGV_GRADE = 'M'

# 1. 读取五级规格
t_spec = get_tactile_spec(AGV_GRADE)  # {'array': (16,16), 'freq_hz': 100, ...}
f_spec = get_force_spec(AGV_GRADE)
i_spec = get_imu_spec(AGV_GRADE)

# 2. 创建传感器
tactile = TactileArray(array_size=t_spec['array'], sensor_type=TactileSensorType.CAPACITIVE)
force   = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
imu     = IMUSensor(sensor_type=IMUSensorType.BMI088, sample_rate=i_spec['sample_hz'])

# 3. 打开传感器
tactile.open()
force.open()
imu.open()

# 4. 标定
imu.calibrate_gyro_bias(num_samples=100)
imu.calibrate_accel(known_orientation="level")
force.calibrate_bias(num_samples=50)
force.set_tool_center(tool_mass=0.5, tool_com=np.array([0, 0, 0.1]))

# 5. 创建姿态估计器
pose_est = PoseEstimator(algorithm='madgwick', sample_rate=i_spec['sample_hz'])

# 6. 主循环
import time
while True:
    t = time.time()
    
    # 触觉数据
    t_frame = tactile.capture()
    contacts = tactile.detect_contacts(t_frame)
    slip = tactile.get_slip_signal(t_frame)
    grip_quality = tactile.estimate_grip_quality(t_frame)
    
    # 力觉数据
    wrench = force.capture()
    contact_state = force.detect_contact(wrench)
    
    # IMU数据
    i_frame = imu.capture()
    pose = pose_est.update(i_frame.accel, i_frame.gyro, i_frame.mag)
    euler = pose.to_euler()
    
    print(f"t={t:.3f} | 触觉:{len(contacts)}接触 | "
          f"力觉:{wrench.magnitude:.1f}N | "
          f"姿态:r={euler[0]:.2f}p={euler[1]:.2f}y={euler[2]:.2f}")
    
    time.sleep(1.0 / t_spec['freq_hz'])

# 7. 清理
tactile.close()
force.close()
imu.close()
```

---

### I.6 传感器模块对外导出接口

```python
# src/sensors/__init__.py 导出清单
from .tactile import (
    TactileArray,           # 触觉阵列主类
    TactileFrame,           # 触觉帧数据结构
    TactileContact,         # 接触事件数据结构
    TactileCalibration,     # 标定参数数据结构
    TactileSensorType,      # 传感器类型枚举
    PressureProcessor,       # 压力信号处理器
    VirtualTactileSensor,   # 仿真传感器
    get_tactile_spec,       # AGV五级规格查询
    AGV_TACTILE_GRADES,     # 五级规格表常量
)

from .force import (
    ForceTorqueSensor,      # 六维力矩传感器主类
    Wrench,                 # 力旋量数据结构
    ForceCalibration,       # 标定参数数据结构
    ContactState,           # 接触状态数据结构
    ForceSensorType,        # 传感器类型枚举
    WrenchProcessor,        # 力旋量信号处理器
    VirtualForceSensor,     # 仿真传感器
    get_force_spec,         # AGV五级规格查询
    AGV_FORCE_GRADES,       # 五级规格表常量
)

from .imu import (
    IMUSensor,              # IMU传感器主类
    IMUFrame,               # IMU数据帧数据结构
    Pose,                   # 位姿数据结构
    PoseEstimator,          # 姿态估计器
    IMUCalibration,         # 标定参数数据结构
    IMUSensorType,          # 传感器类型枚举
    VirtualIMUSensor,       # 仿真传感器
    get_imu_spec,           # AGV五级规格查询
    AGV_IMU_GRADES,         # 五级规格表常量
)
```


---

## 附录J: 物理仿真与跨模态标定 (v2.19.0)

> **版本**: v2.19.0  
> **更新**: 2026-04-09  
> **模块**: `src/simulation/physics_sim.py`, `src/simulation/cross_modal_calibration.py`

---

### J.1 模块概述

物理仿真与跨模态联合标定是 SuperModel 具身智能系统的两大支撑模块:

| 模块 | 文件 | 职责 |
|------|------|------|
| **PhysicsSim** | `physics_sim.py` | 刚体动力学仿真、接触力学、AGV五级物理规格 |
| **CrossModalCalibrator** | `cross_modal_calibration.py` | 触觉-力觉/IMU-姿态联合标定、标定质量评估 |

---

### J.2 物理仿真引擎 (PhysicsSim)

#### J.2.1 核心类

```
PhysicsSimulator
├── add_body(RigidBody) → body_id
├── get_body(name) → RigidBody
├── step(dt?) → None
├── simulate_drop(body_name, drop_height, duration) → trajectory
├── simulate_collision(body1_name, body2_name, impact_velocity, duration) → collision_data
└── config: PhysicsSimConfig

RigidBody
├── position: np.ndarray (3,)      # 世界坐标系位置 (m)
├── orientation: np.ndarray (4,)    # 四元数 (qw, qx, qy, qz)
├── linear_velocity: np.ndarray (3,) # 线速度 (m/s)
├── angular_velocity: np.ndarray (3,) # 角速度 (rad/s)
├── mass: float                     # 质量 (kg)
├── inertia: np.ndarray (3,)        # 主惯性矩
├── kinetic_energy → float          # 动能 (J)
└── to_pose_matrix() → np.ndarray (4,4)

PhysicsSimConfig
├── gravity: np.ndarray (3,)    # 重力加速度 (m/s²)
├── dt: float                   # 时间步长 (s)
├── substeps: int               # 子步数
├── restitution: float          # 恢复系数
├── friction_static/dynamic     # 摩擦系数
├── contact_stiffness/damping   # 接触弹簧阻尼
└── grade: str                  # AGV等级
```

#### J.2.2 接触力学模型

PhysicsSim 使用弹簧-阻尼模型计算接触力:

```
F_normal = k · penetration + c · relative_velocity_normal
F_friction ≤ μ · F_normal
```

接触检测采用球体近似 ( `_estimate_radius` )，支持:
- AGV 底盘刚体
- 轮子刚体
- 夹爪/灵巧手刚体
- 货物/障碍物刚体

#### J.2.3 AGV五级物理规格

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| **质量范围 (kg)** | 10-50 | 10-100 | 20-200 | 50-500 | 100-2000 |
| **尺寸范围 (m)** | 0.3-0.6 | 0.3-0.6 | 0.4-0.8 | 0.5-1.0 | 0.6-1.5 |
| **最大线速度 (m/s)** | 0.5 | 1.0 | 2.0 | 3.0 | 5.0 |
| **最大角速度 (rad/s)** | 1.0 | 2.0 | 3.0 | 5.0 | 10.0 |
| **接触刚度 (N/m)** | 5,000 | 10,000 | 20,000 | 50,000 | 100,000 |
| **阻尼 (N·s/m)** | 50 | 100 | 200 | 500 | 1000 |
| **静摩擦系数** | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
| **动摩擦系数** | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
| **仿真步长 (ms)** | 2.0 | 1.0 | 0.5 | 0.2 | 0.1 |
| **控制频率 (Hz)** | 50 | 100 | 200 | 500 | 1000 |

**快速创建:**
```python
from src.simulation.physics_sim import create_physics_sim_for_grade, create_agv_body

sim = create_physics_sim_for_grade('M')      # 创建M级物理仿真器
body = create_agv_body("chassis", grade='XL') # 创建XL级AGV刚体

sim.add_body(body)
result = sim.simulate_drop("chassis", drop_height=1.0, duration=2.0)
```

---

### J.3 跨模态联合标定 (CrossModalCalibration)

#### J.3.1 标定问题建模

跨模态联合标定解决三个核心问题:

**问题1: 触觉 → 力觉映射**
```
F_wrench = M_t2f @ tactile_features
其中: tactile_features = pressure_map.flatten()  # (N,)
      M_t2f: (6, N) 转换矩阵
      F_wrench: (6,) 力旋量 [Fx, Fy, Fz, Tx, Ty, Tz]
方法: 最小二乘法 + L2正则化 (λ = 1e-6)
```

**问题2: IMU → 姿态角映射**
```
euler = M_imu2e @ imu_features + bias_accel
其中: imu_features = [accel_x, accel_y, accel_z, roll, pitch估计]
方法: 多位置线性回归
```

**问题3: 力传感器零偏估计**
```
bias = mean(force_wrench_static)  # 静止数据均值
方法: 静止采集 → 均值估计 → 在线补偿
```

#### J.3.2 标定流程

```
1. 静止标定 (零偏估计)
   ↓
   add_static_calibration(force_wrench, accel, gyro)
   ↓
   calibrate_force_bias() → force_bias (6,)

2. 姿态标定 (IMU→姿态)
   ↓
   add_oriented_calibration(tactile_pressure, force_wrench, imu_euler, imu_accel)
   ↓
   calibrate_imu_orientation() → (imu_to_euler_matrix, accel_bias)

3. 触觉→力觉标定
   ↓
   calibrate_tactile_to_force() → tactile_to_force_matrix (6, N)

4. 温度系数标定 (可选)
   ↓
   _calibrate_temp_coefficients() → temp_coefficient_force, temp_coefficient_accel

5. 完整标定
   ↓
   calibrate_full() → CalibrationResult
```

#### J.3.3 标定质量评估

`evaluate_quality()` 返回综合评分:

| 指标 | 计算方法 | 评分标准 |
|------|---------|---------|
| **overall_score** | 0.4×force + 0.3×orient + 0.3×r2 | 越高越好 |
| **residual_force** | RMS(force_pred - force_true) | <0.1N 优秀 |
| **residual_orientation** | RMS(euler_pred - euler_true) | <0.01rad 优秀 |
| **r_squared_force** | R² 决定系数 | >0.95 优秀 |
| **num_samples** | 有效标定样本数 | >100 充足 |

#### J.3.4 AGV五级标定规格

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| **最少静止样本** | 50 | 100 | 200 | 500 | 1000 |
| **最少姿态样本** | 20 | 50 | 100 | 200 | 500 |
| **力精度要求 (N)** | 1.0 | 0.5 | 0.2 | 0.1 | 0.05 |
| **姿态精度要求 (rad)** | 0.1 | 0.05 | 0.02 | 0.01 | 0.005 |
| **温度范围 (°C)** | 10 | 15 | 20 | 25 | 30 |
| **标定时间 (min)** | 10 | 15 | 20 | 30 | 60 |

**使用示例:**
```python
from src.simulation.cross_modal_calibration import CrossModalCalibrator, get_calibration_spec

calibrator = CrossModalCalibrator(tactile_size=(16,16), grade='M')
spec = get_calibration_spec('M')  # {'min_static_samples': 100, ...}

# 采集静止数据
for _ in range(100):
    calibrator.add_static_calibration(
        force_wrench=np.array([0.0, 0.0, -9.81, 0.0, 0.0, 0.0]),
        accel=np.array([0.0, 0.0, 9.81]),
        gyro=np.array([0.0, 0.0, 0.0]),
    )

calibrator.calibrate_full()
quality = calibrator.evaluate_quality()
print(f"标定质量: {quality['overall_score']:.2%}")
calibrator.save('calibration.npz')
```

---

### J.4 仿真与标定集成

物理仿真与跨模态标定联合使用，实现仿真到真实的迁移 (Sim2Real):

```
物理仿真 (PhysicsSim)
    ├── 仿真触觉响应 (触觉阵列 + 接触力学)
    ├── 仿真力觉响应 (六维力矩传感器)
    ├── 仿真IMU数据 (加速度/角速度/姿态)
    ↓
跨模态标定 (CrossModalCalibrator)
    ├── 学习触觉→力觉映射
    ├── 学习IMU→姿态映射
    ├── 标定传感器零偏和比例因子
    ↓
真实机器人部署
    ├── 应用标定参数到真实传感器
    └── 补偿系统误差
```

---

### J.5 测试覆盖

| 测试文件 | 覆盖模块 | 测试数 |
|---------|---------|-------|
| `tests/simulation_tests.py` | physics_sim + cross_modal_calibration | 30 |

**主要测试用例:**
- 刚体添加、获取、动能计算
- AGV五级物理规格完整性
- 下落仿真轨迹生成
- 碰撞仿真能量守恒
- 静止标定数据采集
- 姿态标定数据采集
- 力觉零偏标定
- 触觉→力觉转换矩阵标定
- 完整标定流程
- 标定结果持久化
- 接触点/接触力数据结构

---

## 附录K: AGV卡死检测与自主恢复系统 (v2.20.0)

> 本附录定义 SuperModel 超模态大模型 AGV 的卡死检测与自主恢复系统，
> 为 L/XL/XXL 级 AGV 提供智能化故障恢复能力。

### K.1 系统架构

```
PatrolController (巡逻控制器)
    ├── StuckDetector (卡死检测器)
    │       ├── 机械卡死检测 (命令大但位置不变)
    │       ├── 振荡死锁检测 (位置方差过小)
    │       └── 轮胎打滑检测 (IMU vs 里程计不一致)
    │
    └── AutonomousRecoveryManager (自主恢复管理器)
            ├── RecoveryStrategy (恢复策略)
            │       ├── RETRY: 重试当前动作
            │       ├── BACKUP: 后退尝试
            │       ├── ROTATE: 原地旋转后重试
            │       ├── SIDESTEP: 侧向横移 (Mecanum)
            │       ├── REPLAN: 重新规划路径
            │       ├── ABORT: 放弃当前任务点
            │       └── ESCALATE: 升级处理 (人工干预)
            │
            ├── 策略降级 (等级不足时)
            ├── 策略升级 (多次失败时)
            └── 恢复历史记录
```

### K.2 卡死检测算法

#### K.2.1 机械卡死检测

检测条件:
- 电机指令幅度 > 0.05 m/s
- 实际运动效率 < 20%
- 有指令时间占比 > 50%

```
运动效率 = 实际位移 / 预期位移
预期位移 = 平均指令速度 × 有指令时间
```

#### K.2.2 振荡死锁检测

检测条件:
- 位置标准差 < 0.02m (卡死阈值)
- 持续时间 > 3.0s

说明: AGV 在小范围内振荡但无法前进，常见于复杂障碍物环境。

#### K.2.3 轮胎打滑检测

检测条件:
- 滑移率 > 50%
- 滑移率 = 1 - (里程计位移 / IMU位移)

说明: IMU 检测到加速但里程计无对应位移，常见于光滑地面。

### K.3 恢复策略分级

| 策略 | S级 | M级 | L级 | XL级 | XXL级 | 描述 |
|------|-----|-----|-----|------|-------|------|
| RETRY | ✅ | ✅ | ✅ | ✅ | ✅ | 重试当前动作 |
| BACKUP | ❌ | ✅ | ✅ | ✅ | ✅ | 后退后重试 |
| ROTATE | ❌ | ✅ | ✅ | ✅ | ✅ | 原地旋转后重试 |
| SIDESTEP | ❌ | ❌ | ✅ | ✅ | ✅ | Mecanum 横移 |
| REPLAN | ❌ | ❌ | ✅ | ✅ | ✅ | 重新规划路径 |
| ABORT | ❌ | ✅ | ✅ | ✅ | ✅ | 放弃当前目标点 |
| ESCALATE | ✅ | ❌ | ❌ | ✅ | ✅ | 请求人工干预 |

### K.4 策略升级机制

```
RETRY → BACKUP → ROTATE → SIDESTEP → REPLAN → ABORT → ESCALATE
```

每次恢复失败后自动升级策略，最多尝试 3 次后升级。

### K.5 AGV五级恢复能力规格

| 参数 | S | M | L | XL | XXL |
|------|-----|-----|-----|------|------|
| 卡死检测 | ❌ | ❌ | ✅ | ✅ | ✅ |
| 自主恢复 | ❌ | 基础 | 完整 | 完整+日志 | MPC预测+云端 |
| 最大恢复次数 | 0 | 3 | 3 | 5 | 无限制 |
| 恢复冷却时间(s) | N/A | 2.0 | 2.0 | 1.0 | 0.5 |
| 传感器降级 | ❌ | ❌ | ✅ | ✅ | ✅ |
| 故障日志 | ❌ | ❌ | ❌ | ✅ | ✅ |
| 人工干预接口 | ❌ | ❌ | ❌ | ✅ | ✅ |

### K.6 核心接口

#### StuckDetector

```python
class StuckDetector:
    def update(position, command, imu_frame, timestamp) -> StuckDetectionResult
    def reset()
```

#### AutonomousRecoveryManager

```python
class AutonomousRecoveryManager:
    def request_recovery(stuck_result, current_pose, target_pose, available_sensors, timestamp) -> Optional[Dict]
    def check_recovery_complete(strategy, elapsed_time, current_pose, start_pose) -> Tuple[bool, bool]
    def get_diagnostics() -> Dict
    def reset()
```

### K.7 集成到巡逻控制器

L/XL/XXL 级 PatrolController 自动启用卡死检测与恢复:

```python
controller = PatrolController(grade='XL', initial_pose=(0, 0, 0))
controller.start_patrol()

for _ in range(1000):
    vel, state = controller.update(dt=0.02)
    # 内部自动进行:
    # 1. 卡死检测 (每0.5秒)
    # 2. 策略选择与执行
    # 3. 恢复完成检查
```

### K.8 测试覆盖

| 测试文件 | 覆盖模块 | 测试数 |
|---------|---------|-------|
| `tests/patrol_control_tests.py` | StuckDetector + RecoveryManager | 37 |

**主要测试用例:**
- 五级巡逻控制器创建
- 巡逻启动/停止/暂停/恢复
- 单点/多点巡逻
- 障碍物检测与避障
- 到达判定
- 指标统计
- 事件记录
- 恢复策略选择 (机械卡死/振荡死锁/轮胎打滑)
- 策略降级与升级
- 恢复完成判定


---

## 附录L: AGV五级规格总表 (v2.60.0)

> 完整的 SuperModel AGV 五级 (S/M/L/XL/XXL) 系统规格参考手册
> 版本: v2.60.0 | 更新: 2026-04-10

### L.1 整车规格总表

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **负载能力** | 30kg | 100kg | 300kg | 600kg | 1200kg |
| **自重** | 15kg | 35kg | 80kg | 150kg | 300kg |
| **最大总重** | 45kg | 135kg | 380kg | 750kg | 1500kg |
| **车体尺寸** | 0.4×0.3×0.12m | 0.6×0.4×0.15m | 0.8×0.6×0.2m | 1.0×0.7×0.25m | 1.2×0.9×0.3m |
| **轮子配置** | 2轮驱动 | 2轮驱动 | 4轮驱动 | 4轮驱动 | 4轮驱动 |
| **轮子直径** | 100mm | 140mm | 140mm | 165mm | 200mm |
| **电机类型** | 57步进 | 5.5寸轮毂150W | 5.5寸轮毂150W×2 | 6.5寸轮毂200W×2 | 7.5寸轮毂300W×4 |
| **最高速度** | 0.5m/s | 1.5m/s | 2.0m/s | 2.5m/s | 3.0m/s |
| **最大扭矩** | 5Nm | 15Nm | 30Nm | 60Nm | 120Nm |
| **定位精度** | ±10mm | ±5mm | ±3mm | ±1mm | ±0.5mm |
| **防护等级** | IP20 | IP30 | IP54 | IP65 | IP67 |
| **典型价格** | ¥5-15K | ¥15-50K | ¥50-150K | ¥150-500K | >¥500K |

### L.2 感知子系统规格总表

| 模态 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **相机** | 单目640×480 | 双目D435i 720p | 双目D455 60fps | 双目+事件相机 | 多目+3D LiDAR |
| **麦克风** | 1ch | 2ch阵列 | 4ch阵列 | 6ch阵列 | 8ch阵列 |
| **触觉阵列** | 8×8 | 16×16 | 24×24 | 32×32 | 48×48 |
| **触觉分辨率** | 12bit | 12bit | 14bit | 14bit | 16bit |
| **触觉压力范围** | 0-500kPa | 0-1000kPa | 0-2000kPa | 0-5000kPa | 0-10000kPa |
| **触觉采样率** | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| **力觉轴数** | 3轴 | 6轴 | 6轴 | 6轴 | 6轴 |
| **力觉力范围** | ±100N | ±200N | ±500N | ±1000N | ±5000N |
| **力觉力矩范围** | ±10Nm | ±20Nm | ±50Nm | ±100Nm | ±500Nm |
| **力觉分辨率** | 0.1N | 0.05N | 0.02N | 0.01N | 0.005N |
| **力觉采样率** | 100Hz | 500Hz | 1000Hz | 2000Hz | 5000Hz |
| **IMU型号** | MPU6050 | BMI088 | BMI088 | ADIS16470 | ADIS16470 |
| **IMU采样率** | 100Hz | 200Hz | 500Hz | 1000Hz | 2000Hz |
| **IMU噪声密度** | 400μg/√Hz | 120μg/√Hz | 60μg/√Hz | 20μg/√Hz | 10μg/√Hz |
| **融合编码器** | 128CPR | 256CPR | 512CPR | 768CPR | 1024CPR |

### L.3 控制子系统规格总表

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **控制频率** | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| **控制架构** | 位置环 | 位置+速度环 | 位置+速度+阻抗 | 全模态闭环 | 全模态+MPC |
| **核心算法** | PID | PID+前馈 | 阻抗+前馈 | 阻抗+MPC | MPC+自适应 |
| **实时性** | 非实时 | 非实时 | Xenomai | RT-PREEMPT | Xenomai+FPGA |
| **力控能力** | 无 | 碰撞检测 | 5Hz力控 | 20Hz力控 | 50Hz力控 |
| **避障算法** | 人工势场 | DWA | DWA+VFH+APF | 混合 | 多层融合 |
| **轨迹规划** | 直线 | RRT | RRT*+样条 | MPC+RRT* | MPC+多次RRT* |
| **碰撞响应** | >100ms | <50ms | <20ms | <10ms | <5ms |
| **姿态稳定** | <500ms | <200ms | <100ms | <50ms | <20ms |
| **卡死检测** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **自主恢复** | ❌ | 基础 | 完整 | 完整+日志 | MPC预测+云端 |
| **多机协同** | ❌ | ❌ | ❌ | 5台 | 20台+ |

### L.4 计算与通信规格总表

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **处理器** | RPi 4B | RK3588/Nano | Orin NX | Orin AGX | Orin AGX×2+GPU |
| **AI算力** | <5 TOPS | 5-20 TOPS | 20-100 TOPS | 100-300 TOPS | >300 TOPS |
| **内存** | 4GB | 8GB | 16-32GB | 64-128GB | 256+GB |
| **功耗** | <10W | 15-30W | 30-80W | 80-150W | 150-500W |
| **实时控制** | ✗ | ✗ | ✓ Xenomai | ✓ RT-PREEMPT | ✓ Xenomai+FPGA |
| **有线通信** | USB | USB/ETH | Ethernet | EtherCAT | EtherCAT+光纤 |
| **无线通信** | WiFi | WiFi | WiFi+5G | 5G+LoRa | 5G+卫星 |
| **CAN总线** | ❌ | CANopen | CANopen×2 | EtherCAT | EtherCAT+双网 |
| **传感器接口** | USB | USB/CAN | CAN/ETH | EtherCAT | EtherCAT+光纤 |

### L.5 感知→控制闭环延迟规格总表

| 阶段 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **传感器采样** | 20ms | 10ms | 5ms | 2ms | 1ms |
| **特征提取** | 80ms | 30ms | 15ms | 5ms | 2ms |
| **融合推理** | 30ms | 10ms | 5ms | 2ms | 1ms |
| **决策规划** | 20ms | 10ms | 5ms | 2ms | 1ms |
| **控制计算** | 10ms | 5ms | 2ms | 1ms | 0.5ms |
| **电机响应** | 40ms | 15ms | 5ms | 2ms | 1ms |
| **总延迟** | <200ms | <80ms | <35ms | <15ms | <7ms |

### L.6 传感器模块接口速查表

#### TactileArray (触觉传感器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `open()` | - | bool | 打开传感器连接 |
| `close()` | - | None | 关闭传感器 |
| `capture()` | - | TactileFrame | 捕获触觉数据帧 |
| `detect_contacts(frame)` | TactileFrame | List[TactileContact] | 检测接触区域 |
| `get_slip_signal(frame)` | TactileFrame | np.ndarray | 获取滑移信号 |
| `estimate_grip_quality(frame)` | TactileFrame | Dict | 估计抓取质量 |
| `calibrate(zero_pressure, known_weights)` | np.ndarray, List[float] | None | 传感器标定 |

#### ForceTorqueSensor (力觉传感器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `open()` | - | bool | 打开传感器连接 |
| `close()` | - | None | 关闭传感器 |
| `capture()` | - | Wrench | 捕获六维力旋量 |
| `get_wrench()` | - | Wrench | 获取最新力数据 |
| `detect_contact(wrench, threshold)` | Wrench, float | ContactState | 接触检测 |
| `estimate_payload(wrench)` | Wrench | float | 估计负载重量 |
| `calibrate_bias(num_samples)` | int | None | 偏置校准 |
| `set_tool_center(tool_mass, tool_com)` | float, np.ndarray | None | 设置工具中心 |

#### IMUSensor (IMU传感器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `open()` | - | bool | 打开传感器连接 |
| `close()` | - | None | 关闭传感器 |
| `capture()` | - | IMUFrame | 捕获IMU数据帧 |
| `self_test()` | - | bool | 传感器自检 |
| `calibrate_gyro_bias(num_samples)` | int | None | 陀螺仪偏置校准 |
| `calibrate_accel(known_orientation)` | str | None | 加速度计标定 |

#### PoseEstimator (姿态估计器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `update(accel, gyro, mag, dt)` | np.ndarray, np.ndarray, np.ndarray, float | Pose | 更新姿态估计 |
| `get_pose()` | - | Pose | 获取当前姿态 |
| `get_euler()` | - | np.ndarray | 获取当前欧拉角 |
| `get_rotation_matrix()` | - | np.ndarray | 获取旋转矩阵 |
| `integrate_velocity(accel, dt, remove_gravity)` | np.ndarray, float, bool | Tuple | 积分速度/位置 |
| `reset()` | - | None | 重置积分状态 |

### L.7 已完成模块清单

| 模块 | 文件 | 状态 | 测试数 |
|------|------|:----:|:------:|
| 触觉传感器 | `sensors/tactile.py` | ✅ 完成 | 45 |
| 力觉传感器 | `sensors/force.py` | ✅ 完成 | 52 |
| IMU传感器 | `sensors/imu.py` | ✅ 完成 | 48 |
| 传感器管理器 | `sensors/manager.py` | ✅ 完成 | 35 |
| 信号处理器 | `sensors/signal_processor.py` | ✅ 完成 | 30 |
| 编码器 | `sensors/encoders.py` | ✅ 完成 | 28 |
| 控制模块 | `control/*.py` | ✅ 完成 | 400+ |
| 传感器测试 | `tests/sensor_tests.py` | ✅ 完成 | 341 |
| 融合测试 | `tests/fusion_tests.py` | ✅ 完成 | 73 |
| 设计文档 | `docs/DESIGN.md` | ✅ 完成 | - |
| 规格文档 | `docs/AGV_SPEC.md` | ✅ 完成 | - |
| 控制参数文档 | `docs/AGV_CONTROL_PARAMS.md` | ✅ 完成 | - |
