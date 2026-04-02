# SuperModel 模块索引 / Module Index

> **版本**: v1.46.0
> **更新**: 2026-04-02 21:00
> **项目**: SuperModel 超模态机器人具身智能大脑

本文档是 SuperModel 项目的完整模块索引，提供所有源代码模块、设计文档和测试用例的快速导航。

---

## 更新日志

- **v1.46.0** (2026-04-02 21:00): 新增AGV五级规格速查卡AGV_SPEC_QUICKREF.md、一图对比+选型指南、1135项测试持续通过
- **v1.44.0** (2026-04-02): 新增41项边界/鲁棒性测试 (NaN/Inf/饱和/集成融合)，总计1135项测试通过
- **v1.31.0** (2026-04-01): 新增 AGV五级快速参考卡 AGV_SPEC_QUICKREF.md (一键速查S→XXL规格差异)
- **v1.30.0** (2026-04-01): 控制模块测试全部完成 (1112项测试通过)
- **v1.29.0** (2026-04-01): 新增附录E - AGV五级完整规格总表 (七大子系统全覆盖)
- **v1.23.0** (2026-04-01): 新增附录D - AGV World Model世界模型规格
- **v1.22.0** (2026-04-01): 新增MODULE_INDEX.md模块索引文档
- **v1.21.0** (2026-04-01): 全模块完成确认 (触觉/力觉/IMU + 控制 + 测试1019项通过)

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

