# Changelog

All notable changes to SuperModel will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **完整测试套件**: 424 项测试全部通过，覆盖传感器、融合、控制、世界模型、仿真
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
