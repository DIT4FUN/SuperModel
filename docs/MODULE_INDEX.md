# SuperModel 模块索引 / Module Index
- **v2.09.0** (2026-04-09): 具身任务执行器扩展 (push/pull/surface_trace/insert/polish); 新增embodied_task_tests.py (41项); 修复SensorHealthMonitor bug; 421项测试全通过
- **v2.08.0** (2026-04-09): 新增AGV_CONTROL_PARAMS.md五级控制参数完整指南(PID/阻抗/MPC/安全监控); 新增agv_five_grade_demo.py五级完整对比演示; 1845项测试全通过
- **v2.07.0** (2026-04-09): 触觉/力觉/IMU模块完善 + 测试扩展 + 接口规范文档; 378项测试全通过
- **v2.06.0** (2026-04-09): 新增INTEGRATION_GUIDE完整集成指南; 传感器→融合→控制全链路接口规范; 368项传感器+融合测试全通过

> **版本**: v2.09.0
> **更新**: 2026-04-09
> **项目**: SuperModel 超模态机器人具身智能大脑

本文档是 SuperModel 项目的完整模块索引，提供所有源代码模块、设计文档和测试用例的快速导航。

---

## 更新日志

- **v1.76.0** (2026-04-08 19:55): 1456项测试全通过 + 轨迹规划模块全面升级(PurePursuit/Stanley/PID/RRT*/最小Snap) + 设计文档附录G完整接口规范
- **v1.75.0** (2026-04-08 19:15): 1434项测试全通过 + 触觉/力觉/IMU完整模块 + AGV五级规格表完善 + 控制模块深化 + 模块接口设计文档完整
- **v1.71.0** (2026-04-08 16:30): 传感器扩展测试(+36项) + 控制模块完善 + 文档更新
- **v1.65.0** (2026-04-08 14:40): 1423项测试全通过 + 触觉/力觉/IMU模块完善 + 控制模块深化 + 设计文档细化
- **v1.64.0** (2026-04-07 14:41): 持续完善控制模块 + 设计文档细化 + 1409项测试全通过
- **v1.61.0** (2026-04-07 12:36): 新增控制集成测试(control_integration_tests.py, 27项) + 传感器融合仿真演示(run_sensor_fusion.py, S/M/L三级) + 触觉/力觉/IMU流水线测试
- **v1.53.0** (2026-04-05 17:23): 更新所有文档、添加硬件规格(镭神N10P/ETT10A-PW/ESUN万向轮/Astra/C100/ZLAC8015D)
- **v1.52.0** (2026-04-05 12:48): 修复sensorimotor抓取仿真阶段判断bug、pybullet_sim.py物理引擎兼容修复、SPEC.md扩展接口设计文档、1311项测试通过
- **v1.51.0** (2026-04-03): 补充AGV五级控制子系统规格表、PyBullet可视化仿真脚本、1277项测试通过
- **v1.46.0** (2026-04-02 21:00): 新增AGV五级规格速查卡AGV_SPEC_QUICKREF.md、一图对比+选型指南、1135项测试持续通过

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
├── control/          # 执行层 - 运动控制 (19个子模块)
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
│   ├── agv.py             # AGV运动学/轨迹跟踪
│   ├── embodied_control.py # 具身传感控制 (触觉+力觉+IMU融合)
│   ├── sensorimotor.py    # 传感器-运动整合
│   └── navigation.py      # AGV导航 (全局路径规划+局部避障+轨迹跟踪)
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
│   ├── pybullet_sim.py  # PyBullet仿真环境
│   ├── agv_model_generator.py  # AGV URDF模型生成
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
| 控制监管 | `supervisor.py` | `ControlSupervisor`, `GradeAwareSupervisor` | 控制器生命周期/模式切换/故障恢复/AGV五级感知 |
| 避障 | `obstacle_avoidance.py` | `DynamicWindowApproach`, `ArtificialPotentialField`, `ObstacleAvoider` | DWA/APF/VFH |
| 任务规划 | `planner.py` | `TaskPlanner`, `HierarchicalPlanner` | HTN层次化规划 |
| 技能库 | `skill.py` | `SkillLibrary`, `Skill` | 技能注册/执行 |
| ROS2接口 | `ros2_interface.py` | `ROS2JointTrajectoryInterface`, `ROS2ActionInterface` | ROS2通信 |
| 多AGV协调 | `multi_agent.py` | `MultiAgentCoordinator` | 编队/碰撞检测/任务分配 |
| AGV运动 | `agv.py` | `AGVMotionController`, `KinematicsBase` | 差速/麦克纳姆轮运动学 |
| 导航控制 | `navigation.py` | `NavigationController`, `OccupancyGrid`, `AStarPlanner`, `DijkstraPlanner` | 全局路径规划+轨迹跟踪 |

---

## 测试套件 (`tests/`)

| 文件 | 测试数 | 覆盖范围 |
|------|--------|---------|
| `sensor_tests.py` | 163 | 触觉/力觉/IMU/Vision/Audio |
| `fusion_tests.py` | 44 | 跨模态融合网络 |
| `control_integration_tests.py` | 27 | 传感器-融合-控制-执行器完整闭环 |
| `control_tests.py` | 174+ | 运动/MPC/阻抗/安全控制器 |
| `pybullet_sim_tests.py` | 41 | PyBullet仿真 |
| `ros2_interface_tests.py` | 44+ | ROS2通信 |
| `five_grade_integration_tests.py` | 50+ | 五级AGV完整集成 |
| 其他 | 732+ | 编码器/仿真/场景/边界/鲁棒性 |
| **总计** | **1456** | **全部通过** |

---

## 设计文档 (`docs/`)

```
docs/
├── QUICKSTART.md               # 快速入门指南
├── HARDWARE_SPEC.md           # 硬件规格说明 (镭神N10P/ETT10A-PW/Astra/C100/ZLAC8015D)
├── AGV_SPEC.md                # AGV规格说明
├── SPEC.md                   # 技术规格文档
├── MODULE_INDEX.md           # 模块索引
├── DESIGN.md                # 架构设计文档
├── PRACTICAL_DEPLOYMENT.md   # AGV五级部署实战指南 (选型→配置→仿真→实机)
├── AGV_CONTROL_PARAMS.md     # AGV五级控制参数完整指南 (PID/阻抗/MPC/安全监控)
└── architecture/
│   ├── SUPER_MODEL_ARCHITECTURE.md    # 系统架构
│   ├── MODEL_STRUCTURE.md             # 模型内部结构层次图
│   ├── SENSOR_CONTROL_PRACTICAL_GUIDE.md  # 传感器-控制实战指南
│   └── AGV_COMPLETE_SPEC_REFERENCE.md  # AGV完整规格参考
└── design/
    ├── MODULE_INTERFACE.md          # 模块接口设计
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
| **M** | 标准助手 | 100Hz | 16×16,100Hz | 6轴,500Hz | BMI088/ETT10A-PW | HYBRID |
| **L** | 专业工业 | 200Hz | 24×24,200Hz | 6轴,1000Hz | BMI088+Mag | HYBRID |
| **XL** | 高性能 | 500Hz | 32×32,500Hz | 6轴,2000Hz | ADIS16470 | EARLY |
| **XXL** | 旗舰全功能 | 1000Hz | 48×48,1000Hz | 6轴,5000Hz | ADIS×2 | EARLY |

---

*本文档版本: v1.29.0*
*最后更新: 2026-04-01*

---

## 附录D: World Model 世界模型规格 (v1.23.0)

### 世界模型核心架构

World Model 位于 `src/learning/world_model.py`，提供环境动力学预测能力。

| 组件 | 类 | 功能 |
|------|-----|------|
| 环境模型 | `WorldModel` | 观测预测、动作建模 |
| 表征学习 | `RSSM` (Recurrent State Space Model) | 序列潜变量建模 |
| 奖励预测 | `RewardPredictor` | 隐空间奖励预测 |
| 观测重构 | `ObservationDecoder` | 图像/触觉/力觉重构 |
| 控制器 | `Controller` | MPC/PEIRL策略学习 |

### World Model AGV五级规格

| 规格 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| **潜空间维度** | 32 | 64 | 128 | 256 | 512 |
| **隐变量维度** | 32 | 32 | 64 | 128 | 256 |
| **序列长度** | 50 | 100 | 200 | 400 | 1000 |
| **动作空间** | 7DoF | 7DoF | 12DoF | 18DoF | 自由度 |
| **预测频率** | 10Hz | 20Hz | 50Hz | 100Hz | 200Hz |
| **观测重构** | 图像 | 图像+深度 | 多模态 | 多模态 | 全模态 |
| **NPU加速** | ✗ | RK3588 | RK3588 | RK3588×2 | RK3588×4 |
| **模型参数量** | 2M | 8M | 30M | 100M | 500M |

### World Model 训练模式

| 模式 | 描述 | 适用场景 |
|------|------|---------|
| `dreamer` | DreamerV3 自由运行训练 | 离线强化学习 |
| `cvae` | Conditional VAE 动作条件生成 | 模仿学习 |
| `contrastive` | 对比学习表征 | 跨模态对齐 |
| `masked` | Masked 自编码器 | 传感器融合 |

### World Model 集成接口

```python
from src.learning.world_model import WorldModel, WorldModelConfig

config = WorldModelConfig(
    hidden_dim=256,
    stochastic_dim=32,
    deterministic_dim=512,
    action_dim=7,
    obs_encoder_dim=1024,
    training_mode="dreamer",
    device="cuda" if torch.cuda.is_available() else "cpu"
)
model = WorldModel(config)

# 观测预测
observation_embed = encoder(obs)
posterior_z = model.rssm.encode(observation_embed, action, hidden)
reconstruction = model.decoder(posterior_z, hidden)
predicted_reward = model.reward_predictor(posterior_z, hidden)
```

### World Model 性能指标

| 指标 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| **观测预测MSE** | <0.05 | <0.02 | <0.01 | <0.005 | <0.001 |
| **奖励预测Acc** | >75% | >82% | >88% | >92% | >96% |
| **推理延迟(ms)** | 8 | 5 | 3 | 1.5 | 0.5 |
| **训练步数/秒** | 50 | 120 | 300 | 800 | 2000 |
| **内存占用(MB)** | 256 | 512 | 1024 | 2048 | 8192 |

### World Model 在具身智能中的作用

```
感知输入 → [编码器] → 观测嵌入
                           ↓
动作输入 → [动作编码] → concat → [RSSM] → 潜状态序列
                                              ↓
                    [奖励预测器] ← 奖励信号
                           ↓
                    [控制器] → 策略动作
                           ↓
                    [Decoder] → 下一帧预测
                           ↓
                         环境
```

World Model 使机器人能够：
1. **想象** 未执行动作的结果 (免费 rollout)
2. **梦学习** - 在潜空间探索不安全的动作
3. **跨模态预测** - 用视觉预测触觉/力觉反馈
4. **终身学习** - 持续更新环境模型

---

## 附录E: AGV五级完整规格总表 (v1.29.0)

> **目的**: 提供 SuperModel 全部七大子系统的 AGV 五级 (S/M/L/XL/XXL) 规格一站式总览。
> **覆盖**: 感知层、融合层、认知层、执行层、学习层、通信层、安全层
> **适用**: 快速选型 / 规格对比 / 项目报价 / 学术对标

---

### E.1 感知子系统 (Perception) — 传感器规格

| 模态 | 参数 | S | M | L | XL | XXL |
|------|------|-----|-----|-----|------|------|
| **视觉** | 相机类型 | 单目USB | 双目D435i | 双目D455 | 双目+事件相机 | 多目+3D LiDAR |
| | 分辨率 | 640×480 | 1280×720 | 1280×720 | 1920×1080 | 1920×1080×4 |
| | 帧率 (fps) | 30 | 30 | 60 | 90 | 120 |
| | 基线 (mm) | — | 50 | 50 | 75 | 100-200 |
| | 深度范围 (m) | — | 0.2-5 | 0.2-8 | 0.1-10 | 0.1-30 |
| | 编码器维度 | — | 256 | 512 | 768 | 1024 |
| **听觉** | 麦克风数量 | 1 | 2 | 4阵列 | 6阵列 | 8阵列 |
| | 采样率 (Hz) | 16000 | 16000 | 22050 | 32000 | 44100 |
| | 拾音范围 (m) | 1.0 | 3.0 | 5.0 | 8.0 | 10.0 |
| | 声源定位精度 | — | ±15° | ±5° | ±2° | ±1° |
| | 编码器维度 | 64 | 128 | 128 | 256 | 256 |
| **触觉** | 阵列尺寸 | 8×8 | 16×16 | 24×24 | 32×32 | 48×48 |
| | 分辨率 (bit) | 12 | 12 | 14 | 14 | 16 |
| | 压力范围 (kPa) | 0-500 | 0-1000 | 0-2000 | 0-5000 | 0-10000 |
| | 采样频率 (Hz) | 50 | 100 | 200 | 500 | 1000 |
| | 温度感知 | ✗ | ✓ | ✓ | ✓ | ✓ |
| | 接近觉 | ✗ | ✗ | ✓ | ✓ | ✓ |
| | 滑移检测 | ✗ | ✓ | ✓ | ✓ | ✓ |
| | 编码器维度 | 32 | 64 | 64 | 128 | 128 |
| **力觉** | 轴数 | 3 | 6 | 6 | 6 | 6 |
| | 力范围 (N) | ±100 | ±200 | ±500 | ±1000 | ±5000 |
| | 力矩范围 (N·m) | ±10 | ±20 | ±50 | ±100 | ±500 |
| | 分辨率 | 0.1N | 0.05N | 0.02N | 0.01N | 0.005N |
| | 采样频率 (Hz) | 100 | 500 | 1000 | 2000 | 5000 |
| | 编码器维度 | 16 | 32 | 32 | 64 | 64 |
| **IMU** | 传感器型号 | MPU6050 | BMI088 | BMI088 | ADIS16470 | ADIS16470 |
| | 加速度量程 (g) | 8 | 16 | 24 | 40 | 80 |
| | 陀螺量程 (°/s) | 1000 | 2000 | 4000 | 4000 | 8000 |
| | 采样频率 (Hz) | 100 | 200 | 500 | 1000 | 2000 |
| | 噪声密度 | 400μg/√Hz | 120μg/√Hz | 60μg/√Hz | 20μg/√Hz | 10μg/√Hz |
| | 磁力计 | ✗ | ✗/✓ | ✗ | ✓ | ✓ |
| | 编码器维度 | 32 | 32 | 64 | 64 | 128 |
| **编码器** | 融合隐层维度 | 128 | 256 | 512 | 768 | 1024 |
| | 输出表示维度 | 64 | 128 | 256 | 512 | 1024 |

---

### E.2 融合子系统 (Fusion) — 跨模态融合规格

| 参数 | S | M | L | XL | XXL |
|------|-----|-----|-----|------|------|
| **融合策略** | LATE | HYBRID | HYBRID | EARLY+HYBRID | EARLY+HYBRID+LATE |
| **注意力头数** | 4 | 4 | 6 | 8 | 12 |
| **融合层数** | 2 | 2 | 3 | 4 | 6 |
| **融合延迟** | <20ms | <10ms | <5ms | <2ms | <1ms |
| **内存占用** | <200MB | <500MB | <1GB | <2GB | <4GB |
| **吞吐量** | 50 fps | 100 fps | 200 fps | 500 fps | 1000 fps |
| **功耗** | <2W | <5W | <10W | <20W | <40W |
| **Cross-Attention** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **模态 dropout** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **渐进式融合** | ✗ | ✗ | ✓ | ✓ | ✓ |

---

### E.3 认知子系统 (Perception/Cognition) — 场景理解规格

| 参数 | S | M | L | XL | XXL |
|------|-----|-----|-----|------|------|
| **场景图构建** | ✗ | ✓ (规则) | ✓ (CNN) | ✓ (Transformer) | ✓ (多模态) |
| **占据栅格分辨率** | — | 0.1m | 0.05m | 0.02m | 0.01m |
| **动态物体跟踪** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **意图识别** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **建图更新频率** | — | 1Hz | 5Hz | 10Hz | 20Hz |

---

### E.4 执行子系统 (Control) — 运动控制规格

| 参数 | S | M | L | XL | XXL |
|------|-----|-----|-----|------|------|
| **控制频率** | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| **控制模式** | 位置 | 位置+速度 | 位置+速度+阻抗 | 全模态 | 全模态+MPC |
| **关节数** | 3 | 6 | 7 | 7-12 | 12+ |
| **位置环带宽** | 2Hz | 5Hz | 10Hz | 20Hz | 50Hz |
| **力控带宽** | — | 3Hz | 8Hz | 15Hz | 30Hz |
| **姿态稳定时间** | <500ms | <200ms | <100ms | <50ms | <20ms |
| **碰撞响应时间** | >100ms | <50ms | <20ms | <10ms | <5ms |
| **避障算法** | 人工势场 | DWA | DWA+VFH+APF | 混合 | 多层融合 |
| **轨迹规划** | 直线 | RRT | RRT*+样条 | MPC+RRT* | MPC+多次RRT* |
| **AGV驱动方式** | 双轮差分 | 双轮差分 | 麦克纳姆 | 麦克纳姆+舵机 | 全向+多连杆 |

---

### E.5 学习子系统 (Learning) — 自主学习规格

| 参数 | S | M | L | XL | XXL |
|------|-----|-----|-----|------|------|
| **学习范式** | 模仿学习 | 模仿+强化 | 强化+自监督 | 强化+自监督+迁移 | 全范式+持续学习 |
| **训练频率** | 离线 | 离线+在线 | 在线 | 在线+元学习 | 在线+元学习+联邦 |
| **数据效率** | 大样本 | 中样本 | 小样本 | 少样本 | 零样本 |
| **Dreamer 具身智能** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **世界模型** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **自主探索** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **持续学习** | ✗ | ✗ | ✗ | ✓ | ✓ |
| **分布式训练** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **SIM到REAL迁移** | ✗ | ✗ | ✓ | ✓ | ✓ |

---

### E.6 通信子系统 (Communication) — 接口与协议规格

| 参数 | S | M | L | XL | XXL |
|------|-----|-----|-----|------|------|
| **ROS2版本** | ROS2 Humble | Humble | Humble+Jazzy | Jazzy | Jazzy+自研 |
| **实时控制** | ✗ | ✗ | ✓ ( Xenomai) | ✓ (Xenomai/RT-PREEMPT) | ✓ (Xenomai+专用FPGA) |
| **有线通信** | USB | USB/Ethernet | Ethernet | EtherCAT | EtherCAT+光纤 |
| **无线通信** | WiFi | WiFi | WiFi+5G | 5G+LoRa | 5G+卫星+光纤 |
| **多机协同** | ✗ | ✗ | ✗ | ✓ (5台) | ✓ (20台+) |
| **边缘计算** | 本地 | 本地+边缘 | 边缘+云 | 边缘+云+雾 | 分布式计算集群 |
| **延迟要求** | <100ms | <50ms | <25ms | <10ms | <5ms |

---

### E.7 安全子系统 (Safety) — 安全与监控规格

| 参数 | S | M | L | XL | XXL |
|------|-----|-----|-----|------|------|
| **安全标准** | 基础 | ISO10218 | ISO10218+ISO15066 | ISO10218+ISO15066+CE | 全标准+功能安全(SIL3) |
| **碰撞检测** | 速度限制 | 力限幅 | 六维力矩检测 | 六维+触觉检测 | 六维+触觉+视觉 |
| **急停等级** | 基础 | PLd | PLe | PLe+监控 | PLe+冗余+监控 |
| **故障容忍** | ✗ | ✓ | ✓ | ✓ | ✓ (N=3冗余) |
| **安全监控频率** | 10Hz | 50Hz | 100Hz | 200Hz | 500Hz |
| **预测性维护** | ✗ | ✗ | ✓ | ✓ | ✓ (AI预测) |
| **安全日志** | 本地 | 本地+远程 | 本地+远程+审计 | 本地+远程+审计+区块链 | 全量+实时分析 |

---

### E.8 硬件平台规格

| 参数 | S | M | L | XL | XXL |
|------|-----|-----|-----|------|------|
| **计算平台** | Raspberry Pi 4B | RK3588/Jetson Nano | Jetson Orin NX | Jetson Orin AGX | Orin AGX×2+GPU |
| **算力 (TOPS)** | < 5 | 5-20 | 20-100 | 100-300 | > 300 |
| **内存** | 4 GB | 8 GB | 16-32 GB | 64-128 GB | 256+ GB |
| **存储** | 32 GB | 128 GB | 512 GB | 2 TB | 4+ TB (NVMe) |
| **典型功耗** | < 10W | 15-30W | 30-80W | 80-150W | 150-500W |
| **防护等级** | IP30 | IP54 | IP65 | IP67 | IP67+防腐 |
| **工作温度** | 0-40°C | -10-50°C | -20-60°C | -30-70°C | -40-80°C |
| **价格区间** | ¥5-15K | ¥15-50K | ¥50-150K | ¥150-500K | > ¥500K |

---

### E.9 端到端系统集成指标

| 指标 | S | M | L | XL | XXL |
|------|-----|-----|-----|------|------|
| **感知→决策延迟** | <150ms | <50ms | <25ms | <10ms | <5ms |
| **感知→控制总延迟** | <200ms | <70ms | <35ms | <15ms | <7ms |
| **抓取成功率** | — | >80% | >90% | >95% | >99% |
| **运动定位精度** | ±5cm | ±2cm | ±5mm | ±1mm | ±0.1mm |
| **电池续航 (小时)** | 2-3 | 4-6 | 6-8 | 8-12 | 12-24 |
| **MTBF (小时)** | 1000 | 5000 | 15000 | 30000 | 50000+ |

---

### E.10 快速选型指南

```
需求场景                      推荐等级   关键理由
─────────────────────────────────────────────────────
高校教学/科研实验              S-M        低成本、快速上手、ROS2兼容
室内服务机器人 (送物/导览)    M-L        标准SLAM+语音+轻量抓取
工业装配/精密操作             L-XL       力控+视觉+实时控制+安全
多机协同/物流仓储             XL-XXL     高速+多机+大规模仿真
户外全地形/特种作业           XXL        全模态+高可靠+强实时
科研发表/算法验证             M-L        平衡真实感与仿真便利性
产品原型/概念验证             S-M        快速迭代、成本可控
规模化商用部署                 L-XL       可靠+可维护+性价比
```

> **附录E版本**: v1.29.0 | **最后更新**: 2026-04-01 | **维护者**: SuperModel开发团队


---

## 附录D: 传感器模块完整接口规格

### D.1 触觉传感器接口 (tactile.py)

#### 核心类及方法签名

```python
# === TactileArray ===
class TactileArray:
    def __init__(
        self,
        array_size: Tuple[int, int],      # (rows, cols)
        sensor_type: TactileSensorType,     # RESISTIVE/CAPACITIVE/PIEZOELECTRIC/OPTICAL
        sensor_id: str = "default",
        calibration: Optional[TactileCalibration] = None
    ) -> None

    def open(self) -> bool                  # 打开传感器
    def close(self) -> None                 # 关闭传感器
    def capture(self) -> TactileFrame       # 采集一帧触觉数据
    def detect_contacts(self, frame: Optional[TactileFrame] = None) -> List[TactileContact]
    def get_slip_signal(self, frame: Optional[TactileFrame] = None) -> np.ndarray  # HxW
    def estimate_grip_quality(self, frame: Optional[TactileFrame] = None) -> Dict[str, float]
    def calibrate(self, zero_pressure: Optional[np.ndarray] = None,
                  known_weights: Optional[List[float]] = None) -> None

# === TactileFrame 数据结构 ===
@dataclass
class TactileFrame:
    pressure_map: np.ndarray              # HxW float32, 归一化 0-1
    temperature_map: Optional[np.ndarray]   # HxW float32, 摄氏度
    proximity: Optional[np.ndarray]        # HxW float32, 米
    slip_signal: Optional[np.ndarray]      # HxW float32, 0-1
    timestamp: float                      # 秒
    frame_id: int                         # 帧序号
    sensor_id: str                        # 传感器ID

# === TactileContact 接触事件 ===
@dataclass
class TactileContact:
    center: Tuple[int, int]              # 接触中心 (row, col)
    area: int                            # 接触面积 (像素)
    peak_pressure: float                  # 峰值压力
    mean_pressure: float                  # 平均压力
    centroid: Tuple[float, float]         # 压力质心
    contact_force: float                  # 估计接触力 (N)
    slip_probability: float              # 滑移概率
    temperature: Optional[float]          # 接触区温度

# === VirtualTactileSensor 仿真接口 ===
class VirtualTactileSensor:
    def simulate_contact(
        self,
        contact_pos: Tuple[float, float],   # 归一化 (0-1)
        contact_radius: float = 0.3,
        contact_force: float = 10.0,
        noise_level: float = 0.05
    ) -> TactileFrame

    def simulate_sliding(
        self,
        direction: Tuple[float, float],
        speed: float = 0.1,
        duration_frames: int = 30
    ) -> List[TactileFrame]

    def simulate_multi_contact(
        self,
        contacts: List[Tuple[Tuple[float, float], float, float]],
        noise_level: float = 0.05
    ) -> TactileFrame

    def simulate_slip_detection(
        self,
        normal_force: float = 10.0,
        friction_coeff: float = 0.3,
        velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> Dict[str, float]

# === AGV五级触觉规格 ===
AGV_TACTILE_GRADES = {
    'S':   {'array': (8,8),    'res': 12, 'range_kpa': (0,500),    'freq_hz': 50,   'temp': False},
    'M':   {'array': (16,16),  'res': 12, 'range_kpa': (0,1000),   'freq_hz': 100,  'temp': True},
    'L':   {'array': (24,24),  'res': 14, 'range_kpa': (0,2000),   'freq_hz': 200,  'temp': True},
    'XL':  {'array': (32,32),  'res': 14, 'range_kpa': (0,5000),   'freq_hz': 500,  'temp': True},
    'XXL': {'array': (48,48),  'res': 16, 'range_kpa': (0,10000),  'freq_hz': 1000, 'temp': True},
}
```

### D.2 力觉传感器接口 (force.py)

```python
# === ForceTorqueSensor ===
class ForceTorqueSensor:
    def __init__(
        self,
        sensor_type: ForceSensorType,        # SIX_AXIS/THREE_AXIS/JOINT_TORQUE/FINGER_TIP
        sensor_id: str = "ft_0",
        calibration: Optional[ForceCalibration] = None,
        ip_address: Optional[str] = None,
        ethernet_type: str = "UDP"
    ) -> None

    def open(self) -> bool                  # 打开传感器
    def close(self) -> None                 # 关闭传感器
    def capture(self) -> Wrench              # 采集六维力旋量
    def get_wrench(self) -> Optional[Wrench]
    def detect_contact(self, wrench: Optional[Wrench] = None,
                       threshold: Optional[float] = None) -> ContactState
    def estimate_payload(self, wrench: Optional[Wrench] = None) -> float
    def set_tool_center(self, tool_mass: float, tool_com: np.ndarray) -> None
    def calibrate_bias(self, num_samples: int = 100) -> None

# === Wrench 力旋量 ===
@dataclass
class Wrench:
    force: np.ndarray    # (3,) Fx,Fy,Fz 单位:N
    torque: np.ndarray   # (3,) Tx,Ty,Tz 单位:N·m
    timestamp: float
    frame_id: int
    sensor_id: str

    def magnitude(self) -> float           # ||F||
    def torque_magnitude(self) -> float    # ||T||
    def to_vector(self) -> np.ndarray      # (6,) [Fx,Fy,Fz,Tx,Ty,Tz]
    def transform(self, rotation: np.ndarray, translation: np.ndarray) -> 'Wrench'

# === VirtualForceSensor 仿真接口 ===
class VirtualForceSensor:
    def simulate_contact(
        self,
        force: Tuple[float, float, float],
        torque: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        add_noise: bool = True
    ) -> Wrench

    def simulate_payload(
        self,
        mass: float = 1.0,
        com_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        gravity: float = 9.81
    ) -> Wrench

    def simulate_collision(
        self,
        direction: Tuple[float, float, float],
        peak_force: float = 50.0,
        duration_ms: float = 100.0,
        decay: str = "exponential"
    ) -> List[Wrench]

    def simulate_surface_contact(
        self,
        surface_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        contact_point: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        penetration_depth: float = 0.001,
        stiffness: float = 1000.0,
        damping: float = 50.0
    ) -> Wrench

    def simulate_friction_contact(
        self,
        normal_force: float = 10.0,
        velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        friction_coeff: float = 0.3,
        object_mass: float = 1.0
    ) -> Wrench

# === AGV五级力觉规格 ===
AGV_FORCE_GRADES = {
    'S':   {'axes': 3,  'force_range': 100,   'torque_range': 10,    'resolution': 0.1,  'sampling_hz': 100},
    'M':   {'axes': 6,  'force_range': 200,   'torque_range': 20,    'resolution': 0.05, 'sampling_hz': 500},
    'L':   {'axes': 6,  'force_range': 500,   'torque_range': 50,    'resolution': 0.02, 'sampling_hz': 1000},
    'XL':  {'axes': 6,  'force_range': 1000,  'torque_range': 100,   'resolution': 0.01, 'sampling_hz': 2000},
    'XXL': {'axes': 6,  'force_range': 5000,  'torque_range': 500,   'resolution': 0.005,'sampling_hz': 5000},
}
```

### D.3 IMU传感器接口 (imu.py)

```python
# === IMUSensor ===
class IMUSensor:
    def __init__(
        self,
        sensor_type: IMUSensorType,          # BMI088/MPU6050/MPU9250/ADIS16470/VIRTUAL
        sensor_id: str = "imu_0",
        calibration: Optional[IMUCalibration] = None,
        accel_range: int = 16,               # g
        gyro_range: int = 2000,              # deg/s
        sample_rate: int = 200               # Hz
    ) -> None

    def open(self) -> bool                  # 打开传感器
    def close(self) -> None                 # 关闭传感器
    def capture(self) -> IMUFrame            # 采集IMU数据帧
    def self_test(self) -> bool              # 自检
    def calibrate_gyro_bias(self, num_samples: int = 500, duration_sec: float = 5.0) -> None
    def calibrate_accel(self, known_orientation: str = "level") -> None

# === IMUFrame 数据帧 ===
@dataclass
class IMUFrame:
    accel: np.ndarray             # (3,) m/s²
    gyro: np.ndarray              # (3,) rad/s
    mag: Optional[np.ndarray]     # (3,) μT (9轴IMU)
    temperature: float            # 摄氏度
    timestamp: float              # 秒
    frame_id: int
    sensor_id: str

    @property def accel_magnitude(self) -> float
    @property def gyro_magnitude(self) -> float

# === Pose 位姿 ===
@dataclass
class Pose:
    position: np.ndarray          # (3,) m
    orientation: np.ndarray       # (4,) qw,qx,qy,qz

    def to_euler(self) -> np.ndarray     # [roll, pitch, yaw] rad
    def to_matrix(self) -> np.ndarray    # 4x4变换矩阵
    @classmethod def identity(cls) -> 'Pose'
    @classmethod def from_euler(cls, position: np.ndarray, rpy: np.ndarray) -> 'Pose'

# === PoseEstimator 姿态估计 ===
class PoseEstimator:
    def __init__(
        self,
        algorithm: str = "madgwick",   # "madgwick" / "complementary" / "kalman"
        sample_rate: float = 200.0,
        beta: float = 0.1
    ) -> None

    def update(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        mag: Optional[np.ndarray] = None,
        dt: Optional[float] = None
    ) -> Pose

    def get_euler(self) -> np.ndarray
    def get_rotation_matrix(self) -> np.ndarray
    def integrate_velocity(self, accel: np.ndarray, dt: float,
                           remove_gravity: bool = True) -> Tuple[np.ndarray, np.ndarray]
    def reset(self) -> None

# === VirtualIMUSensor 仿真接口 ===
class VirtualIMUSensor:
    def simulate_static(self, orientation: Tuple[float, float, float] = (0,0,0)) -> IMUFrame
    def simulate_motion(self, linear_accel: Tuple, angular_vel: Tuple, dt: float = 0.01) -> IMUFrame
    def simulate_trajectory(self, trajectory_type: str = "circle",
                            duration_s: float = 2.0, dt: float = 0.01) -> List[IMUFrame]
    def simulate_agv_motion(self, linear_velocity: Tuple = (0,0),
                              angular_velocity: float = 0.0,
                              dt: float = 0.01, grade: str = "M") -> IMUFrame
    def simulate_human_walking(self, step_frequency: float = 1.5,
                               walk_speed: float = 1.0,
                               duration_s: float = 5.0,
                               dt: float = 0.01) -> List[IMUFrame]

# === AGV五级IMU规格 ===
AGV_IMU_GRADES = {
    'S':   {'type': 'MPU6050',    'accel_range': 8,   'gyro_range': 1000,  'sample_hz': 100,  'noise_density': 400},
    'M':   {'type': 'BMI088',     'accel_range': 16,  'gyro_range': 2000,  'sample_hz': 200,  'noise_density': 120},
    'L':   {'type': 'BMI088',      'accel_range': 24,  'gyro_range': 4000,  'sample_hz': 500,  'noise_density': 60},
    'XL':  {'type': 'ADIS16470',   'accel_range': 40,  'gyro_range': 4000,  'sample_hz': 1000, 'noise_density': 20},
    'XXL': {'type': 'ADIS16470',   'accel_range': 80,  'gyro_range': 8000,  'sample_hz': 2000, 'noise_density': 10},
}
```

### D.4 传感器融合接口 (sensor_fusion.py / cross_modal_fusion.py)

```python
# === sensor_fusion.py ===

class ComplementaryFilter(SensorFusion):
    def __init__(self, alpha: float = 0.98) -> None
    def update(self, measurements: Dict[str, np.ndarray], dt: float) -> np.ndarray  # [roll, pitch, yaw]
    def get_state(self) -> np.ndarray
    def reset(self) -> None

class ExtendedKalmanFilter:
    def __init__(self, state_dim: int, measurement_dim: int) -> None
    def initialize(self, initial_state: np.ndarray) -> None
    def predict(self, dt: float) -> None
    def correct(self, measurement: np.ndarray) -> None
    def update(self, measurements: Dict[str, np.ndarray], dt: float) -> np.ndarray
    def get_state(self) -> np.ndarray
    def get_covariance(self) -> np.ndarray

class MultiSensorFusion:
    def add_fusion_method(self, name: str, method: SensorFusion, weight: float) -> None
    def update(self, sensor_data: Dict[str, Any], dt: float) -> Dict[str, np.ndarray]
    def get_fused_state(self) -> np.ndarray

# === cross_modal_fusion.py ===

class CrossModalFusion(nn.Module):
    def __init__(self, config: FusionConfig) -> None
    def forward(self, multimodal: MultimodalInput) -> Dict[str, torch.Tensor]
    def set_fusion_weights(self, weights: Dict[str, float]) -> None
    def get_fusion_weights(self) -> Dict[str, float]

@dataclass
class FusionConfig:
    vision_dim: int = 256
    tactile_dim: int = 64
    force_dim: int = 32
    imu_dim: int = 32
    hidden_dim: int = 256
    num_heads: int = 8
    dropout: float = 0.1

@dataclass
class MultimodalInput:
    vision: Optional[torch.Tensor]   # BxCxHxW
    audio: Optional[torch.Tensor]    # BxTxF
    tactile: Optional[torch.Tensor] # BxN
    force: Optional[torch.Tensor]   # Bx6
    imu: Optional[torch.Tensor]     # Bx9
    language: Optional[torch.Tensor] # BxL
```

### D.5 传感-运动融合接口 (sensorimotor.py)

```python
class SensorimotorIntegration:
    def __init__(self, config: SensorimotorConfig) -> None
    def open(self) -> bool
    def close(self) -> None
    def step(self, dt: float) -> SensorimotorState
    def update_weights(self, tactile: float, force: float, imu: float) -> None
    def emergency_stop(self) -> None
    def get_state(self) -> SensorimotorState

@dataclass
class SensorimotorConfig:
    tactile_weight: float = 0.3
    force_weight: float = 0.4
    imu_weight: float = 0.3
    fusion_strategy: str = "weighted"
    control_rate: float = 100.0
    grade: str = 'M'
    tactile_enabled: bool = True
    force_enabled: bool = True
    imu_enabled: bool = True

    @classmethod
    def from_grade(cls, grade: str) -> 'SensorimotorConfig'

AGV_SENSORIMOTOR_GRADES = {
    'S':   {'control_rate': 50,  'fusion': 'simple',    'tactile_weight': 0.2},
    'M':   {'control_rate': 100, 'fusion': 'weighted',   'tactile_weight': 0.3},
    'L':   {'control_rate': 200, 'fusion': 'adaptive',   'tactile_weight': 0.35},
    'XL':  {'control_rate': 500, 'fusion': 'hierarchical','tactile_weight': 0.4},
    'XXL': {'control_rate': 1000,'fusion': 'full',       'tactile_weight': 0.4},
}
```

---

## 附录E: 控制子系统五级规格详解

### E.1 控制子系统 AGV 五级规格表

| 控制参数 | S | M | L | XL | XXL |
|---------|:--:|:--:|:--:|:--:|:--:|
| **控制频率** | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| **控制模式** | 位置 | 位置+速度 | 位置+速度+阻抗 | 全模态 | 全模态+MPC |
| **避障算法** | 人工势场 | DWA | DWA+VFH+APF | 混合 | 多层融合 |
| **轨迹规划** | 直线 | RRT | RRT*+样条 | MPC+RRT* | MPC+多次RRT* |
| **碰撞响应** | >100ms | <50ms | <20ms | <10ms | <5ms |
| **姿态稳定** | <500ms | <200ms | <100ms | <50ms | <20ms |
| **力控制** | 无 | 碰撞检测 | 力位混合 | 阻抗+导纳 | 全阻抗控制 |
| **触觉伺服** | 无 | 开环 | 闭环 | 自适应 | 智能预测 |
| **IMU融合** | 互补滤波 | Madgwick | EKF | EKF+磁力计 | 多传感器EKF |
| **多机协同** | 无 | 无 | 无 | 5台 | 20台+ |

### E.2 感知→控制闭环延迟

| 阶段 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| 传感器采样 | 20ms | 10ms | 5ms | 2ms | 1ms |
| 特征提取 | 80ms | 30ms | 15ms | 5ms | 2ms |
| 融合推理 | 30ms | 10ms | 5ms | 2ms | 1ms |
| 决策规划 | 20ms | 10ms | 5ms | 2ms | 1ms |
| 控制计算 | 10ms | 5ms | 2ms | 1ms | 0.5ms |
| 电机响应 | 40ms | 15ms | 5ms | 2ms | 1ms |
| **总延迟** | **<200ms** | **<80ms** | **<35ms** | **<15ms** | **<7ms** |

### E.3 计算与通信五级规格

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **处理器** | RPi 4B | RK3588/Nano | Orin NX | Orin AGX | Orin AGX×2+GPU |
| **AI算力** | <5 TOPS | 5-20 TOPS | 20-100 TOPS | 100-300 TOPS | >300 TOPS |
| **内存** | 4GB | 8GB | 16-32GB | 64-128GB | 256+GB |
| **功耗** | <10W | 15-30W | 30-80W | 80-150W | 150-500W |
| **实时控制** | ✗ | ✗ | ✓ Xenomai | ✓ RT-PREEMPT | ✓ Xenomai+FPGA |
| **有线通信** | USB | USB/ETH | Ethernet | EtherCAT | EtherCAT+光纤 |
| **无线通信** | WiFi | WiFi | WiFi+5G | 5G+LoRa | 5G+卫星 |
| **多机协同** | ✗ | ✗ | ✗ | ✓ 5台 | ✓ 20台+ |

### E.4 控制子系统详细接口

```python
# === 关节运动控制 ===
class MotionController:
    def __init__(self, num_joints: int, config: dict) -> None
    def set_target(self, position: np.ndarray, velocity: float = 0.0) -> None
    def step(self, dt: float) -> np.ndarray        # 返回控制量
    def stop(self) -> None

class AdaptivePIDController:
    def __init__(self, kp: float, ki: float, kd: float) -> None
    def compute(self, error: float, error_derivative: float, dt: float) -> float
    def auto_tune(self, plant: 'SimulatedPlant') -> TunerResult

# === 轨迹规划 ===
class TrajectoryGenerator:
    def generate(self, waypoints: List[JointWaypoint], dt: float) -> JointTrajectory
    def generate_scurve(self, start: float, end: float, duration: float, dt: float) -> np.ndarray

class RRTPlanner:
    def __init__(self, space_bounds: dict, max_iter: int = 1000) -> None
    def plan(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]

# === MPC控制 ===
class JointSpaceMPC:
    def __init__(self, config: MPCConfig) -> None
    def solve(self, state: np.ndarray, ref_traj: np.ndarray) -> np.ndarray
    def set_weights(self, Q: np.ndarray, R: np.ndarray) -> None

# === 阻抗控制 ===
class ImpedanceController:
    def __init__(self, M: np.ndarray, B: np.ndarray, K: np.ndarray) -> None
    def compute(self, error: np.ndarray, error_dot: np.ndarray) -> np.ndarray

# === 安全监控 ===
class SafetyController:
    def __init__(self, config: SafetyConfig) -> None
    def check(self, state: JointStateSnapshot) -> SafetyCheckResult
    def log_event(self, event: SafetyEvent) -> None

# === ROS2接口 ===
class ROS2JointTrajectoryInterface:
    def __init__(self, node_name: str) -> None
    def send_trajectory(self, trajectory: JointTrajectory) -> bool
    def get_state(self) -> JointState

# === AGV运动控制 ===
class AGVMotionController:
    def __init__(self, grade: AGVGrade, drive_type: DriveType) -> None
    def set_twist(self, twist: AGVTwist) -> None
    def follow_trajectory(self, trajectory: List[AGVPose], dt: float) -> None
    def stop(self) -> None

# === 多AGV协调 ===
class MultiAgentCoordinator:
    def __init__(self, formation: FormationType) -> None
    def add_agent(self, agent_id: str, state: AgentState) -> None
    def coordinate(self, states: Dict[str, AgentState]) -> Dict[str, AGVTwist]

# === 遥操作 ===
class TeleoperationController:
    def __init__(self, mode: TeleopMode, config: TeleopConfig) -> None
    def step(self, master: MasterState, latency_ms: float) -> SlaveCommand

# === 控制监管 ===
class ControlSupervisor:
    def __init__(self, config: SupervisorConfig) -> None
    def switch_mode(self, new_mode: ControlMode) -> bool
    def get_health(self) -> HealthStatus
    def recover_from_fault(self) -> bool
```

---

## 附录F: SuperModel 版本路线图

| 版本 | 阶段 | 主要目标 | 状态 |
|------|------|---------|------|
| v1.0 | 基础框架 | 传感器+融合+控制骨架 | ✅ |
| v1.5 | 核心实现 | 触觉+力觉+IMU完整实现 | ✅ |
| v1.7 | 仿真验证 | PyBullet+MuJoCo仿真 | ✅ |
| v1.8 | 具身集成 | 传感器-运动融合+仿真 | ✅ |
| **v1.71** | **测试完善** | **全面测试+接口文档** | **🔄进行中** |
| v2.0 | 具身智能 | 真实AGV部署+视觉-语言-动作 | 计划 |
| v3.0 | 超模态大脑 | 完整超模态LLM+具身RL | 计划 |

