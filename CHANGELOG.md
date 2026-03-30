# Changelog

All notable changes to SuperModel will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.2] - 2026-03-30

### Fixed
- **测试套件**: 修复 `TestHierarchicalPlanner.backtrack` 相关测试用例，返回值解包错误（Tuple[List, List] → result, attempted）
- **全部 729 项测试通过**

### Verified
- 全量测试套件 **729 项**测试全部通过

## [1.4.1] - 2026-03-30

### Completed
- **触觉传感器模块** (`sensors/tactile.py`): 827行，支持电阻/电容/压电/光学式，电子皮肤阵列、压力/温度/接近觉/滑移检测
- **力觉传感器模块** (`sensors/force.py`): 700行，六维力矩传感器接口、偏置校准、负载估计、接触检测
- **IMU传感器模块** (`sensors/imu.py`): 支持BMI088/MPU6050/MPU9250/ADIS16470，Madgwick/互补滤波姿态估计
- **控制模块完善**: AGV运动学/动力学、Pure Pursuit轨迹跟踪、阻抗控制、MPC、ROS2接口、安全控制器
- **仿真环境**: Gymnasium仿真环境、物理引擎集成、传感器仿真
- **测试用例**: 721项测试全部通过

### Verified
- 全量测试套件 **721 项**测试全部通过

## [1.4.0] - 2026-03-30

### Added
- **完整系统演示脚本** (`examples/complete_system_demo.py`): 新增端到端演示脚本，涵盖所有核心模块
- **示例文档** (`examples/README.md`): 新增示例目录文档，包含快速开始和AGV五级配置示例
- **README.md**: 测试数量更新至 721 项

### Verified
- 全量测试套件 **721 项**测试全部通过

## [1.3.0] - 2026-03-30

### Added
- **Pure Pursuit轨迹跟踪器** (`control/agv.py`): 新增 `TrajectoryTracker` 类，实现:
  - 前看距离控制 (look_ahead_distance)
  - Pure Pursuit 转向算法
  - 速度前瞻控制
  - 轨迹完成检测与重置
- **AGV五级规格增强** (`docs/design/AGV_GRADE_SPEC.md`): 新增 Pure Pursuit 轨迹跟踪规格对照表
- **轨迹跟踪测试套件** (`tests/control_tests.py`): 新增 10 项测试覆盖 TrajectoryTracker:
  - 初始化、轨迹设置、角度归一化、前看点查找
  - 基本命令计算、空轨迹处理、完成判断、重置
  - 完整轨迹跟踪仿真
- **Bug Fix**: `simulation/gym_env.py` 修复重复导入 `Optional` 问题

### Verified
- 全量测试套件 **663 项**测试全部通过（新增 10 项轨迹跟踪测试）

## [1.2.0] - 2026-03-30

### Added
- **ROS2接口模块测试** (`tests/ros2_interface_tests.py`): 新增 44 项测试，覆盖:
  - `ROS2JointTrajectoryInterface`: 激活/停用、轨迹发送、状态更新、取消、统计、多种控制模式
  - `ROS2ActionInterface`: 服务器启停、目标管理(发送/取消/状态)、异步目标、结果等待、统计
  - `ROS2ParameterInterface`: 参数读写、批量操作、命名空间、前缀过滤、订阅变更、字典导入导出
  - `ROS2ComponentInterface`: 生命周期状态机(配置/激活/停用/清理/关闭)、回调注册
  - 关节命令/状态数据类、Action反馈/结果数据类

### Verified
- 全量测试套件 **653 项**测试全部通过（sensor_tests.py 90项 + fusion_tests.py 63项 + control_tests.py 174项 + ros2_interface_tests.py 44项 + 其他282项）

## [1.1.0] - 2026-03-30

### Added
- **扩展传感器测试**: 新增 `TestSensorEdgeCasesExtended` (触觉零接触/极端压力、力觉零力/极端值、IMU重力对齐/极端姿态、姿态估计漂移、力旋量变换一致性、多传感器时序一致性) 共 10 项
- **AGV传感器集成测试**: `TestSensorAGVIntegration` (等级规格一致性、传感器更新率匹配、传感器融合时序) 共 3 项
- **传感器数值稳定性测试**: `TestSensorNumericalStability` (压力/力矩/加速度稳定性、四元数归一化) 共 4 项
- **融合模块边缘用例测试**: `TestFusionEdgeCases` (极端值处理、缺失模态、语言模态、大批量处理、重复一致性) 共 6 项
- **融合模块集成测试**: `TestFusionIntegration` (完整模态融合、梯度流、跨注意力形状、语言编码器集成) 共 4 项
- **模态编码器测试**: `TestModalityEncoder` (vision/audio/tactile/force/imu) 共 5 项

### Verified
- 全量测试套件 **609 项**测试全部通过（sensor_tests.py 90项 + fusion_tests.py 63项 + control_tests.py 174项 + 其他282项）
- 触觉/力觉/IMU 模块功能完整：接触检测、滑移检测、抓取质量评估、六维力矩、姿态估计、轨迹仿真
- 控制模块完整：AGV运动控制、MPC、阻抗控制、安全监控、ROS2接口
- 仿真环境完整：RobotSimulator、SuperModelGymEnv、虚拟传感器全套
- 设计文档完整：AGV五级规格表（AGV_GRADE_SPEC.md）、模块接口设计（MODULE_INTERFACE.md）

## [1.0.3] - 2026-03-30

### Added
- **场景理解模块** (`perception/scene_understanding.py`): 占据栅格、物体检测与跟踪(Euclidean聚类)、场景图谱构建、动态/静态分离、触觉-视觉融合、完整场景状态输出
- **场景理解测试** (`tests/scene_tests.py`): OccupancyGrid/SceneObject/SceneGraph/SceneUnderstanding 全功能测试 26项
- **AGV五级场景理解规格**: AGV_GRADE_SPEC.md 新增 C.5 节，覆盖 S~XXL 五级的占据栅格、物体跟踪、语义分割规格
- **MODULE_INTERFACE.md 第24.11节**: 新增 SceneUnderstanding 完整接口文档，包括核心数据结构、AGV五级规格表和使用示例

### Verified
- 全量测试套件 **574 项**测试全部通过（新增 scene_tests.py 26项）
- 触觉/力觉/IMU 模块功能完整：接触检测、滑移检测、抓取质量评估、六维力矩、姿态估计、轨迹仿真
- 控制模块完整：AGV运动控制、MPC、阻抗控制、安全监控、ROS2接口
- 仿真环境完整：RobotSimulator、SuperModelGymEnv、虚拟传感器全套
- 设计文档完整：AGV五级规格表（AGV_GRADE_SPEC.md）、模块接口设计（MODULE_INTERFACE.md）

## [1.0.2] - 2026-03-30

### Added
- **虚拟传感器接口文档**: 新增 MODULE_INTERFACE.md 第22节，涵盖 VirtualTactileSensor、VirtualForceSensor、VirtualIMUSensor 完整API、集成示例和规格对照表
- 触觉仿真：接触事件模拟、滑移动作模拟、高斯压力分布
- 力觉仿真：接触力模拟、负载重力模拟、碰撞事件力曲线
- IMU仿真：静止/运动状态模拟、典型轨迹仿真（圆/8字/正弦/线性）

## [1.0.1] - 2026-03-29

### Fixed
- 更新 README 测试数量从 424 → 443（新增 sensorimotor_tests.py 19项用例）
- 更新 CHANGELOG 测试数量

## [1.0.0] - 2026-03-29

### Added
- **完整具身感知系统**: 双耳声觉 + 双目视觉 + 电子皮肤 + 六维力觉 + IMU 全套传感器模块
- **触觉感知模块** (`tactile.py`): 电子皮肤阵列，含压力分布、接触检测、滑移检测、抓取质量评估、温度感知、接近觉
- **力觉感知模块** (`force.py`): 六维力矩传感器，含负载估计、协作安全监控、偏置校准、工具中心点设置
- **IMU感知模块** (`imu.py`): 惯性测量单元，含 Madgwick/AHRS 姿态估计、线速度/位置估计、AGV五级规格
- **跨模态融合网络** (`fusion/cross_modal_fusion.py`): 跨模态注意力融合，含 Language 模态，统一表示学习
- **神经网络编码器** (`sensors/encoders.py`): CNN/RNN/注意力编码器，支持 Vision/Audio/Tactile/Force/IMU/Language
- **世界模型** (`learning/world_model.py`): Dreamer-style RSSM世界模型，支持 AGV S/M/L/XL/XXL 五级配置
- **Dreamer Agent** (`learning/dreamer_agent.py`): 基于想象轨迹的 Actor-Critic 强化学习智能体
- **自主学习框架** (`learning/self_supervised.py`): 对比学习、好奇心驱动、自监督适应
- **运动控制** (`control/`): PID、阻抗控制、导纳控制、协作控制、MPC 模型预测控制
- **AGV运动控制** (`control/agv.py`): 差速驱动、麦克纳姆轮、全向移动，AGV五级规格
- **轨迹规划** (`control/trajectory.py`): RRT 路径规划、多项式轨迹、S曲线加减速
- **技能库** (`control/skill.py`): 原子技能、时序组合、并行执行、参数化技能
- **任务规划** (`control/planner.py`): HTN 层级任务网络、状态机执行
- **ROS2集成** (`control/ros2_interface.py`): ROS2 Humble 关节轨迹接口、话题/服务接口
- **安全监控** (`control/safety_controller.py`): 碰撞检测、关节限位、速度/加速度限制、容错处理
- **仿真环境** (`simulation/environment.py`): Gymnasium 物理仿真环境，支持传感器仿真
- **完整测试套件**: 443 项测试全部通过，覆盖传感器、融合、控制、世界模型、仿真、传感器-执行器集成
- **AGV五级规格文档**: S/M/L/XL/XXL 全套规格表（感知/融合/认知/执行/硬件）
- **模块接口设计文档**: 完整的 API 接口定义和数据结构规范

### Features
- 支持 AGV 五级规格体系（教育级 → 旗舰全功能）
- 虚拟传感器支持，离线算法验证
- 全链路数据流：感知 → 编码 → 融合 → 认知 → 规划 → 控制
- 构建式学习：无需人工标注，自主构建知识
- 渐进式演化：实时采集、持续学习、自我进化

## [0.9.0] - 2026-03-28

### Added
- MPC 控制器 (`control/mpc.py`)：关节空间 MPC、笛卡尔空间 MPC
- 性能基准测试 (`tests/benchmark_tests.py`)：16 项性能测试
- AGV五级配置文件 (`configs/agv_*.yaml`)
- 多传感器数据采集脚本 (`scripts/multi_sensor_data_collection.py`)

## [0.8.0] - 2026-03-28

### Added
- 编码器测试套件 (`tests/encoder_tests.py`)
- AGV控制等级规格文档 (`docs/design/CONTROL_GRADE_SPEC.md`)
- AGV运动控制模块 (`control/agv.py`)

## [0.7.0] - 2026-03-28

### Added
- 增强触觉滑移检测算法（多尺度 + 高频振动分析）
- 抓取质量评估
- 仿真引擎真实初始化
- 技能库并行执行
- 规划器动作效果评估

## [0.6.0] - 2026-03-27

### Added
- 模块接口设计文档 (`docs/design/MODULE_INTERFACE.md`)
- AGV五级规格表 (`docs/design/AGV_GRADE_SPEC.md`)
- 系统架构设计 (`docs/design/SYSTEM_ARCHITECTURE.md`)

## [0.5.0] - 2026-03-27

### Added
- AGV五级配置（S/M/L/XL/XXL）
- 配置加载器
- 完整流水线演示脚本 (`scripts/demo_full_pipeline.py`)

## [0.4.0] - 2026-03-26

### Added
- MPC控制器（41项测试，新增至358项）
- Gymnasium RL环境

## [0.3.0] - 2026-03-26

### Added
- 性能基准测试（16项 benchmark_tests.py，扩至317项）

## [0.2.0] - 2026-03-25

### Added
- 触觉接触检测测试阈值修正
- pytest顺序依赖问题修复

## [0.1.0] - 2026-03-25

### Added
- 初始项目框架
- 传感器基础模块（视觉、听觉）
- 跨模态融合网络
- 自主学习框架
- 运动控制基础模块
- 仿真环境基础
