# SuperModel 超模态大模型

> AGV具身智能大脑 | 多模态感知融合 | 自主学习

## 项目简介

SuperModel 是一个超模态大模型具身智能系统，专注于 AGV（自动导引车）机器人的智能控制。系统通过融合视觉、听觉、触觉、力觉、IMU等多模态感知数据，结合超模态大模型实现对复杂环境的理解和自主决策。

## 核心特性

- **超模态感知**: 支持视觉、听觉、触觉、力觉、IMU等多模态传感器融合
- **具身智能**: 基于超模态大模型的决策和规划能力
- **自主学习**: 强化学习框架支持持续优化
- **实时控制**: 高性能运动控制（PID、轨迹规划、安全监控）
- **模块化设计**: 传感器、控制、融合模块解耦，易于扩展

## 项目结构

```
SuperModel/
├── src/
│   ├── sensors/          # 多模态传感器接口
│   │   ├── vision.py    # 双目RGBD相机 (RealSense)
│   │   ├── audio.py     # 双耳麦克风阵列
│   │   ├── tactile.py   # 电子皮肤触觉阵列
│   │   ├── force.py     # 六维力矩传感器 (ATI)
│   │   ├── imu.py        # IMU传感器 (BMI088/MPU9250)
│   │   ├── encoders.py  # 特征编码器
│   │   └── manager.py    # 传感器管理器
│   ├── fusion/           # 跨模态融合网络
│   │   ├── cross_modal_fusion.py  # 注意力融合Transformer
│   │   └── sensor_fusion.py        # 互补滤波/EKF/多传感器融合
│   ├── perception/       # 感知与场景理解
│   │   └── scene_understanding.py
│   ├── learning/         # 自主学习框架
│   │   ├── world_model.py
│   │   ├── dreamer_agent.py
│   │   └── autonomous_learning.py
│   ├── control/          # 动作控制模块
│   │   ├── motion.py     # 运动控制
│   │   ├── trajectory.py # 轨迹规划
│   │   ├── impedance.py  # 阻抗控制
│   │   ├── mpc.py        # 模型预测控制
│   │   ├── agv.py        # AGV运动学
│   │   ├── supervisor.py # 控制监管
│   │   ├── safety_controller.py
│   │   ├── ros2_interface.py
│   │   ├── teleop.py     # 遥操作
│   │   ├── multi_agent.py
│   │   ├── obstacle_avoidance.py
│   │   ├── planner.py    # 任务规划
│   │   ├── skill.py      # 技能库
│   │   ├── tactile_control.py
│   │   ├── force_control.py
│   │   └── imu_control.py
│   ├── simulation/       # 仿真环境
│   │   ├── mujoco_sim.py # MuJoCo仿真
│   │   ├── gazebo_sim.py # Gazebo/ROS2仿真
│   │   ├── gym_env.py    # Gymnasium环境
│   │   └── environment.py
│   └── utils.py          # 工具函数
├── examples/             # 示例脚本
│   ├── complete_embodied_pipeline_demo.py
│   ├── agv_five_level_demo.py
│   ├── sensorimotor_integration_demo.py
│   └── ...
├── tests/                # 测试用例 (1094项通过)
│   ├── sensor_tests.py
│   ├── fusion_tests.py
│   ├── control_tests.py
│   └── ...
├── docs/                 # 设计文档
│   ├── SPEC.md           # 技术规格
│   ├── DESIGN.md         # 架构设计
│   └── MODULE_INDEX.md   # 模块索引
└── README.md
```

## 快速开始

### 安装依赖

```bash
pip install numpy
pip install pytest  # 用于测试
```

### 运行测试

```bash
# 运行传感器和融合测试
python -m pytest tests/sensor_tests.py tests/fusion_tests.py -v

# 运行所有测试
python -m pytest tests/ -v

# 运行指定测试类
python -m pytest tests/sensor_tests.py::TestTactileData -v
```

### 示例代码

#### 传感器读取

```python
from sensors.tactile import PressureSensor, TaxelArray, TactileArray
from sensors.force import SixAxisFTSensor, ForceSensorArray
from sensors.imu import BMI088, IMUArray

# 触觉传感器
tactile = TactileArray()
tactile.add_sensor(PressureSensor("p1"))
tactile.add_sensor(TaxelArray("taxel1", rows=16, cols=16))

# 力觉传感器
force = ForceSensorArray()
force.add_sensor(SixAxisFTSensor("ft1", model="mini40"))

# IMU传感器
imu = IMUArray()
imu.add_sensor(BMI088("imu1"))

# 读取数据
t_data = tactile.read_all(0.0)
f_data = force.read_all(0.0)
i_data = imu.read_all(0.0)
```

#### 传感器融合

```python
from fusion.sensor_fusion import ComplementaryFilter, ExtendedKalmanFilter, MultiSensorFusion

# 互补滤波（适合IMU）
fusion = ComplementaryFilter(alpha=0.96)
state = fusion.update({
    'accel': imu_data.acceleration,
    'gyro': imu_data.angular_velocity
}, dt=0.01)

# 多传感器融合
msf = MultiSensorFusion()
msf.add_fusion_method("imu1", ComplementaryFilter(alpha=0.96), weight=1.0)
results = msf.update({"imu1": {...}}, dt=0.01)
```

#### 电机控制与PID

```python
from control.motor import DCMotor, MotorController
from control.pid import PIDController, PIDAutotuner

# 电机控制
motor = DCMotor("m1", reduction_ratio=20.0)
controller = MotorController()
controller.add_motor(motor)
motor.enable()

# PID位置控制
pid = PIDController(kp=1.0, ki=0.1, kd=0.05, output_limit=10.0)
for _ in range(100):
    error = target_position - motor.get_state().position
    control = pid.compute(error, dt=0.01)
    motor.set_target(control, MotorControlMode.PWM)
```

#### AGV运动控制

```python
from control.motion import AGVController, DifferentialDrive, Pose2D, TrajectoryPlanner

# AGV控制器
agv = AGVController(wheel_separation=0.5, wheel_radius=0.1)
agv.move_to(1.0, 2.0, 0.0, dt=0.01)

# 轨迹规划
planner = TrajectoryPlanner()
trajectory = planner.plan_trajectory(
    start=Pose2D(0, 0, 0),
    end=Pose2D(5, 5, 0),
    max_velocity=1.0,
    max_acceleration=0.5
)
```

#### 安全监控

```python
from control.safety import SafetyMonitor, EmergencyStopController, SafetyLevel

# 安全监控器
safety = SafetyMonitor(
    max_velocity=2.0,
    force_threshold=100.0,
    boundary_min=np.array([-10, -10, -np.pi]),
    boundary_max=np.array([10, 10, np.pi])
)

# 综合安全检查
status = safety.check_all(
    velocity=0.5,
    position=(1.0, 2.0, 0.1),
    force_magnitude=10.0,
    torque_magnitude=0.5,
    collision_detected=False,
    sensor_health={"imu1": True, "ft1": True},
    dt=0.01
)

if status.level == SafetyLevel.CRITICAL:
    print(f"安全警告: {status.message}")

# 紧急停止
estop = EmergencyStopController()
estop.trigger("manual")
```

## AGV五级规格表

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

## 传感器规格

### 触觉传感器 (tactile.py)

| 类型 | 型号 | 量程 | 精度 | 特点 |
|------|------|------|------|------|
| 压阻式 | - | 0-1000Pa | 0.01 Pa/bit | 成本低、耐久 |
| 触感阵列 | 16x16 | 0-1000Pa | <1ms响应 | 高密度、仿生皮肤 |
| 压电式 | - | 0-1000Hz | 1kHz采样 | 振动检测、纹理识别 |

### 力觉传感器 (force.py)

| 类型 | 型号 | Fz量程 | Mxy量程 | 噪声 |
|------|------|--------|--------|------|
| 六维力 | ATI mini40 | 120N | 2Nm | 0.2% F.S. |
| 六维力 | ATI Gamma | 200N | 10Nm | 0.2% F.S. |
| 六维力 | ATI SI-120 | 120N | 12Nm | 0.2% F.S. |
| 单轴力 | - | 0-100N | - | 0.001% F.S. |

### IMU传感器 (imu.py)

| 类型 | 型号 | 加速度 | 陀螺仪 | 噪声密度 |
|------|------|--------|--------|---------|
| 6轴IMU | BMI088 | ±24g | ±2000°/s | 150μg/√Hz |
| 9轴IMU | MPU9250 | ±16g | ±2000°/s | 400μg/√Hz |

## 控制模块规格

### PID控制器 (control/pid.py)

- **通用PID**: 位置式/增量式、微分滤波、抗积分饱和、在线调参
- **二维PID**: XY平面运动控制
- **自动整定**: Ziegler-Nichols法

### 安全监控 (control/safety.py)

- **速度限制**: 可配置最大速度和加速度
- **边界检查**: 位置边界保护
- **力矩限制**: 力/力矩阈值监控
- **碰撞检测**: 传感器融合碰撞判断
- **紧急停止**: 软件/硬件急停、恢复锁定

## 通讯协议

- **MQTT**: 发布/订阅式消息传递
- **WebSocket**: 实时数据传输
- **REST API**: 配置和状态查询

## 许可证

MIT License

## 贡献者

DIT4FUN Team

## GitHub

https://github.com/DIT4FUN/SuperModel
