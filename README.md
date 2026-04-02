# SuperModel 超模态大模型

> AGV具身智能大脑 | 多模态感知融合 | 自主学习

## 项目简介

SuperModel 是一个超模态大模型具身智能系统，专注于 AGV（自动导引车）机器人的智能控制。系统通过融合视觉、听觉、触觉、力觉、IMU等多模态感知数据，结合超模态大模型实现对复杂环境的理解和自主决策。

## 核心特性

- **超模态感知**: 支持视觉、听觉、触觉、力觉、IMU等多模态传感器融合
- **具身智能**: 基于超模态大模型的决策和规划能力
- **自主学习**: 强化学习框架支持持续优化
- **实时控制**: 高性能运动控制和电机驱动
- **模块化设计**: 传感器、控制、融合模块解耦

## 项目结构

```
SuperModel/
├── sensors/              # 传感器模块
│   ├── visual.py         # 视觉传感器
│   ├── audio.py           # 听觉传感器
│   ├── tactile.py         # 触觉传感器 ⭐新增
│   ├── force.py          # 力觉传感器 ⭐新增
│   └── imu.py            # IMU传感器 ⭐新增
├── fusion/               # 跨模态融合网络
│   └── sensor_fusion.py  # 传感器融合算法
├── control/              # 控制模块 ⭐新增
│   ├── motor.py          # 电机控制
│   └── motion.py         # 运动控制
├── learning/             # 自主学习框架
├── simulation/           # 仿真环境 (待实现)
├── tests/                # 测试用例 ⭐新增
│   ├── sensor_tests.py   # 传感器测试
│   └── fusion_tests.py   # 融合测试
├── docs/
│   └── DESIGN.md         # 架构设计文档
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
# 运行所有测试
python -m pytest tests/ -v

# 传感器测试
python -m unittest tests.sensor_tests -v

# 融合测试
python -m unittest tests.fusion_tests -v
```

### 示例代码

```python
from sensors.tactile import PressureSensor, TactileArray
from sensors.force import SixAxisFTSensor, ForceSensorArray
from sensors.imu import BMI088, IMUArray
from control.motor import MotorController, ServoMotor
from control.motion import AGVController, DifferentialDrive, Pose2D
from fusion.sensor_fusion import ComplementaryFilter

# 创建传感器
tactile = TactileArray()
tactile.add_sensor(PressureSensor("p1"))

force = ForceSensorArray()
force.add_sensor(SixAxisFTSensor("ft1", model="mini40"))

imu = IMUArray()
imu.add_sensor(BMI088("imu1"))

# 读取数据
for _ in range(10):
    t_data = tactile.read_all()
    f_data = force.read_all()
    i_data = imu.read_all()
    
    # IMU融合
    fusion = ComplementaryFilter(alpha=0.96)
    state = fusion.update({
        'accel': i_data[0].acceleration,
        'gyro': i_data[0].angular_velocity
    }, dt=0.01)
    print(f"融合姿态: {state}")

# AGV控制
agv = AGVController(wheel_separation=0.5, wheel_radius=0.1)
agv.move_to(1.0, 2.0, 0.0, dt=0.01)
print(f"AGV状态: {agv.get_state()}")
```

## AGV等级规格

| 等级 | 负载 | 定位精度 | 导航方式 |
|------|------|---------|---------|
| L1 | ≤500kg | ±10mm | 磁条/二维码 |
| L2 | 500-1500kg | ±5mm | 激光导航 |
| L3 | 1500-3000kg | ±3mm | SLAM视觉 |
| L4 | 3000-5000kg | ±1mm | 多传感器融合 |
| **L5** | >5000kg | <±0.5mm | **超模态具身智能** |

## 传感器规格

### 触觉传感器
- 压阻式压力传感器: 0-1000Pa, 灵敏度 0.01 Pa/bit
- 触感阵列: 16x16 taxel, 响应时间 <1ms
- 压电振动传感器: 0-1000Hz, 采样率 1kHz

### 力觉传感器
- 六维力传感器 (ATI mini40): Fz=120N, Mxy=2Nm
- 噪声等级: 0.2% F.S.
- 采样率: 1000Hz

### IMU
- BMI088: 加速度±24g, 陀螺仪±2000°/s, 噪声密度 150μg/√Hz
- MPU9250: 9轴, 内置磁力计

## 通讯协议

- MQTT: 发布/订阅式消息传递
- WebSocket: 实时数据传输
- REST API: 配置和状态查询

## 许可证

MIT License

## 贡献者

DIT4FUN Team

## GitHub

https://github.com/DIT4FUN/SuperModel
