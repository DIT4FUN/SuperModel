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

待实现：
- Unity / Webots / Isaac Gym 仿真接口
- ROS2 集成
- 数字孪生支持

## 7. 测试框架

### 运行测试
```bash
cd SuperModel
python -m pytest tests/ -v
python -m unittest tests.sensor_tests -v
python -m unittest tests.fusion_tests -v
```

### 测试覆盖
- 传感器模块: 单元测试
- 融合算法: 功能测试 + 稳定性测试
- 控制模块: 集成测试 (PID、安全监控)

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
