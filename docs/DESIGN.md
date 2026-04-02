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
- 控制模块: 集成测试 (待实现)
