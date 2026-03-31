# SuperModel 模块索引 / Module Index

> **版本**: v1.21.0
> **更新**: 2026-04-01
> **项目**: SuperModel 超模态机器人具身智能大脑

本文档是 SuperModel 项目的完整模块索引，提供所有源代码模块、设计文档和测试用例的快速导航。

---

## 源代码模块 (`src/`)

```
src/
├── sensors/          # 感知层 - 传感器接口
│   ├── __init__.py
│   ├── vision.py     # 双目相机 + 深度处理
│   ├── audio.py      # 双耳麦克风 + 声源定位
│   ├── tactile.py    # 触觉阵列 (电子皮肤)
│   ├── force.py      # 六维力矩传感器
│   ├── imu.py        # IMU惯性测量 + 姿态估计
│   ├── encoders.py   # 神经网络编码器 (CNN/RNN/Attention/Language)
│   └── manager.py    # 传感器管理器
│
├── fusion/           # 融合层 - 跨模态融合
│   ├── __init__.py
│   └── cross_modal_fusion.py  # 跨模态注意力融合网络
│
├── learning/         # 认知层 - 自主学习
│   ├── __init__.py
│   ├── dreamer_agent.py      # Dreamer 具身智能体
│   ├── world_model.py        # 世界模型
│   ├── self_supervised.py    # 自监督学习
│   └── autonomous_learning.py # 自主学习框架
│
├── perception/       # 感知层 - 场景理解
│   ├── __init__.py
│   └── scene_understanding.py  # 场景图 + 占据栅格
│
├── control/          # 执行层 - 运动控制 (18个子模块)
│   ├── __init__.py
│   ├── motion.py          # 关节运动控制器
│   ├── trajectory.py      # 轨迹生成 + RRT规划
│   ├── mpc.py             # 模型预测控制 (关节/笛卡尔空间)
│   ├── impedance.py       # 阻抗/导纳控制
│   ├── force_control.py   # 力觉控制 (碰撞检测/力位混合)
│   ├── imu_control.py     # IMU姿态稳定/运动估计
│   ├── tactile_control.py # 触觉伺服/抓取控制
│   ├── safety_controller.py # 安全监控 (五级)
│   ├── obstacle_avoidance.py # 避障 (DWA/APF/VFH)
│   ├── planner.py         # 任务规划 (HTN层次化)
│   ├── skill.py           # 技能库
│   ├── teleop.py          # 遥操作
│   ├── ros2_interface.py  # ROS2接口
│   ├── multi_agent.py     # 多AGV协调
│   └── agv.py             # AGV运动学/轨迹跟踪
│
├── hardware/         # 硬件抽象层
│   ├── __init__.py
│   ├── base.py          # 硬件基类
│   ├── rk3588.py        # RK3588 NPU接口
│   ├── nnpu.py          # NPU加速接口
│   ├── gpio.py          # GPIO控制
│   └── digu_robot.py    # 谛沽机器人接口
│
├── simulation/       # 仿真层
│   ├── __init__.py
│   ├── environment.py   # 物理仿真环境
│   └── gym_env.py      # Gymnasium RL环境
│
└── config_loader.py   # 配置加载器
```

---

## 感知层模块详情 (`src/sensors/`)

### 视觉感知 - `vision.py`
- **类**: `BinocularCamera`, `StereoFrame`, `CameraIntrinsics`
- **功能**: 双目校正、深度估计、点云生成
- **支持等级**: S (单目) → XXL (多目+LiDAR)

### 听觉感知 - `audio.py`
- **类**: `BinauralMic`, `AudioFrame`, `SoundLocalizer`
- **功能**: 双耳听觉、声源定位、波束形成
- **支持等级**: S (单mic) → XXL (8-mic阵列)

### 触觉感知 - `tactile.py`
- **类**: `TactileArray`, `TactileFrame`, `TactileContact`, `VirtualTactileSensor`, `PressureProcessor`
- **功能**: 压力分布、温度感知、接近觉、滑移检测、抓取质量评估
- **AGV规格**: S(8×8,50Hz) → XXL(48×48,1000Hz)
- **接口**: I2C/SPI/USB/EtherCAT

### 力觉感知 - `force.py`
- **类**: `ForceTorqueSensor`, `Wrench`, `ContactState`, `VirtualForceSensor`, `WrenchProcessor`
- **功能**: 六维力/力矩测量、负载估计、碰撞检测、工具坐标系标定
- **AGV规格**: S(3轴,100Hz) → XXL(6轴,5000Hz)
- **接口**: USB HID/CAN/EtherCAT/UDP-Force

### IMU感知 - `imu.py`
- **类**: `IMUSensor`, `IMUFrame`, `Pose`, `PoseEstimator`, `VirtualIMUSensor`
- **功能**: 三轴加速度/陀螺仪/磁力计、Madgwick/Madgwick/卡尔曼姿态估计、速度/位置积分
- **AGV规格**: S(MPU6050,100Hz) → XXL(ADIS16470×2,2000Hz)
- **接口**: I2C/SPI/USB

### 编码器 - `encoders.py`
- **类**: `SensorEncoderWrapper`, `VisionEncoder`, `AudioEncoder`, `TactileEncoder`, `ForceEncoder`, `IMUEncoder`, `MultimodalEncoder`
- **功能**: 多模态特征编码、CNN/RNN/Attention机制、语言编码

### 传感器管理器 - `manager.py`
- **类**: `SensorManager`, `SensorManagerConfig`
- **功能**: 统一传感器初始化/采集/同步管理

---

## 融合层模块详情 (`src/fusion/`)

### 跨模态融合 - `cross_modal_fusion.py`
- **类**: `CrossModalFusion`, `FusionConfig`, `MultimodalInput`, `UnifiedRepresentation`, `LanguageEncoder`, `CrossModalAttention`
- **融合策略**: EARLY / LATE / MIDDLE / HYBRID
- **模态交互**: 9种注意力对 (vision↔audio/force/imu/tactile/language等)
- **输出头**: 状态表示 + 动作策略 + 世界模型预测
- **支持等级**: S(128d) → XXL(1024d)

---

## 控制层模块详情 (`src/control/`)

| 模块 | 文件 | 核心类 | 功能 |
|------|------|--------|------|
| 关节运动 | `motion.py` | `MotionController` | PID位置/速度/力矩控制 |
| 轨迹生成 | `trajectory.py` | `TrajectoryGenerator`, `RRTPlanner`, `ScurveGenerator` | 多项式/RRT/S曲线轨迹 |
| MPC控制 | `mpc.py` | `JointSpaceMPC`, `CartesianMPC`, `DynamicsModel` | 模型预测控制 |
| 阻抗控制 | `impedance.py` | `ImpedanceController`, `AdmittanceController` | 柔顺控制 |
| 力觉控制 | `force_control.py` | `ForceController`, `HybridForcePositionController`, `CollisionDetector` | 导纳/碰撞响应 |
| IMU控制 | `imu_control.py` | `AttitudeStabilizer`, `MotionEstimator` | 姿态稳定/运动估计 |
| 触觉控制 | `tactile_control.py` | `TactileServoController`, `GraspQualityController` | 触觉伺服/抓取 |
| 安全控制 | `safety_controller.py` | `SafetyController` | 五级安全监控 |
| 避障 | `obstacle_avoidance.py` | `DynamicWindowApproach`, `ArtificialPotentialField`, `ObstacleAvoider` | DWA/APF/VFH |
| 任务规划 | `planner.py` | `TaskPlanner`, `HierarchicalPlanner` | HTN层次化规划 |
| 技能库 | `skill.py` | `SkillLibrary`, `Skill` | 技能注册/执行 |
| ROS2接口 | `ros2_interface.py` | `ROS2JointTrajectoryInterface`, `ROS2ActionInterface` | ROS2通信 |
| 多AGV协调 | `multi_agent.py` | `MultiAgentCoordinator` | 编队/碰撞检测/任务分配 |
| AGV运动 | `agv.py` | `AGVMotionController`, `KinematicsBase` | 差速/麦克纳姆轮运动学 |

---

## 测试套件 (`tests/`)

| 文件 | 测试数 | 覆盖范围 |
|------|--------|---------|
| `sensor_tests.py` | 197 | 触觉/力觉/IMU/Vision/Audio 全模块 |
| `fusion_tests.py` | 104 | 跨模态融合网络 |
| `control_tests.py` | 220 | 运动/MPC/阻抗/安全控制器 |
| `embodied_intelligence_tests.py` | 42 | 具身智能完整闭环 |
| `embodied_pipeline_tests.py` | 32 | 端到端流水线 |
| `obstacle_avoidance_tests.py` | 38 | 避障算法 |
| `sensor_control_integration_tests.py` | 31 | 传感器-控制集成 |
| `multi_agent_tests.py` | 25 | 多AGV协调 |
| `ros2_interface_tests.py` | 28 | ROS2通信 |
| `autonomous_learning_tests.py` | 22 | 自主学习框架 |
| 其他 | ~280 | 编码器/仿真/场景/MPC |
| **总计** | **1019** | **全部通过** |

---

## 设计文档 (`docs/`)

```
docs/
├── QUICKSTART.md          # 快速入门指南
├── architecture/
│   ├── SUPER_MODEL_ARCHITECTURE.md    # 系统架构
│   ├── SENSOR_CONTROL_PRACTICAL_GUIDE.md  # 传感器-控制实战指南
│   └── AGV_COMPLETE_SPEC_REFERENCE.md  # AGV完整规格参考
└── design/
    ├── MODULE_INTERFACE.md          # 模块接口设计 (Sections 1-36)
    ├── AGV_FIVE_LEVEL_CONSOLIDATED_SPEC.md  # AGV五级规格总表
    ├── AGV_GRADE_SPEC.md            # AGV等级规格
    ├── CONTROL_GRADE_SPEC.md        # 控制等级规格
    └── SYSTEM_ARCHITECTURE.md       # 系统架构文档
```

### MODULE_INTERFACE.md 内容导航 (Sections 1-36)

| Section | 内容 |
|---------|------|
| 1-3 | 概述 + 感知层接口 (视觉/听觉/触觉/力觉/IMU) |
| 4-5 | 融合层接口 + 认知层接口 |
| 6-12 | 执行层接口 (运动/阻抗/技能/AGV/安全) |
| 13-17 | 仿真层 + ROS2 + 数据格式规范 |
| 18-22 | 触觉/力觉/IMU详细接口 + 虚拟传感器 |
| 23-28 | ROS2接口 + 控制模块 + 多智能体协调 |
| 29-36 | 完整流水线 + 避障 + 触觉/力觉/IMU控制集成 |

---

## AGV五级规格速查

| 等级 | 定位 | 控制频率 | 触觉 | 力觉 | IMU | 融合策略 |
|------|------|---------|------|------|-----|---------|
| **S** | 教育/实验 | 50Hz | 8×8,50Hz | 3轴,100Hz | MPU6050 | LATE |
| **M** | 标准助手 | 100Hz | 16×16,100Hz | 6轴,500Hz | BMI088 | HYBRID |
| **L** | 专业工业 | 200Hz | 24×24,200Hz | 6轴,1000Hz | BMI088+Mag | HYBRID |
| **XL** | 高性能 | 500Hz | 32×32,500Hz | 6轴,2000Hz | ADIS16470 | EARLY |
| **XXL** | 旗舰全功能 | 1000Hz | 48×48,1000Hz | 6轴,5000Hz | ADIS×2 | EARLY |

---

*本文档版本: v1.21.0*
*最后更新: 2026-04-01*
