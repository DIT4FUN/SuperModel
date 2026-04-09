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
- **PyBullet仿真**: 支持多AGV协同仿真和避障测试

## 项目结构

```
SuperModel/
├── src/
│   ├── sensors/          # 多模态传感器接口
│   │   ├── vision.py     # 深度相机 (RealSense/Astra)
│   │   ├── audio.py      # 双耳麦克风阵列
│   │   ├── tactile.py    # 电子皮肤触觉阵列
│   │   ├── force.py      # 六维力矩传感器
│   │   ├── imu.py        # IMU传感器
│   │   ├── encoders.py   # 特征编码器
│   │   └── manager.py    # 传感器管理器
│   ├── fusion/           # 跨模态融合网络
│   │   ├── cross_modal_fusion.py  # 注意力融合Transformer
│   │   └── sensor_fusion.py        # 互补滤波/EKF/多传感器融合
│   ├── perception/        # 感知与场景理解
│   │   └── scene_understanding.py
│   ├── learning/         # 自主学习框架
│   │   ├── world_model.py
│   │   ├── dreamer_agent.py
│   │   └── autonomous_learning.py
│   ├── control/          # 动作控制模块
│   │   ├── motion.py     # 运动控制
│   │   ├── trajectory.py  # 轨迹规划
│   │   ├── impedance.py   # 阻抗控制
│   │   ├── mpc.py        # 模型预测控制
│   │   ├── agv.py        # AGV运动学
│   │   ├── supervisor.py # 控制监管
│   │   ├── safety_controller.py
│   │   ├── ros2_interface.py
│   │   └── ...
│   ├── simulation/       # 仿真环境
│   │   ├── pybullet_sim.py      # PyBullet仿真
│   │   ├── agv_model_generator.py # AGV URDF模型生成
│   │   ├── mujoco_sim.py       # MuJoCo仿真
│   │   └── gym_env.py          # Gymnasium环境
│   └── utils.py          # 工具函数
├── sim_demos/            # PyBullet仿真演示
│   ├── run_gui.py        # S形路径避障
│   ├── run_warehouse.py   # 仓库物流仿真
│   ├── run_multi_agv.py  # 多AGV协同
│   ├── run_agv_showcase.py # AGV等级展示
│   └── base_sim.py       # 仿真基类
├── examples/             # 示例脚本
├── tests/               # 测试用例 (2160+项通过)
│   ├── sensor_tests.py   # 触觉/力觉/IMU传感器测试
│   ├── fusion_tests.py   # 跨模态融合测试
│   ├── control_integration_tests.py  # 控制集成测试
│   ├── five_grade_integration_tests.py # AGV五级集成测试
│   ├── pybullet_sim_tests.py
│   └── ...
├── docs/                # 设计文档
│   ├── HARDWARE_SPEC.md  # 硬件规格说明
│   ├── AGV_SPEC.md      # AGV技术规格
│   ├── AGV_SPEC_QUICKREF.md  # AGV五级速查卡
│   ├── SPEC.md          # 技术规格（含模块接口设计）
│   ├── DESIGN.md        # 架构设计
│   ├── MODULE_INDEX.md   # 模块索引（v1.61.0, 5334行）
│   ├── SENSOR_API_GUIDE.md  # 传感器API实用指南
│   └── design/
│       ├── MODULE_INTERFACE.md  # 详细接口设计（36章节, 5334行）
│       ├── AGV_FIVE_LEVEL_CONSOLIDATED_SPEC.md  # 五级规格总表
│       ├── CONTROL_GRADE_SPEC.md  # 控制等级规格
│       └── SYSTEM_ARCHITECTURE.md  # 系统架构
└── README.md
```

## 快速开始

### 安装依赖

```bash
pip install numpy pybullet pytest
```

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行PyBullet仿真测试
python -m pytest tests/pybullet_sim_tests.py -v
```

### 运行仿真演示

```bash
cd sim_demos

# S形路径避障仿真
python run_gui.py

# 仓库物流仿真
python run_warehouse.py

# 多AGV协同仿真
python run_multi_agv.py

# AGV等级展示
python run_agv_showcase.py
```

## AGV五级规格表

| 等级 | 负载 | 轮子配置 | 电机 | 典型场景 |
|------|------|----------|------|---------|
| **S** | 30kg | 2轮差速 | 57步进 | 小型仓库 |
| **M** | 100kg | 2轮差速 | 5.5寸轮毂150W | 物流分拣 |
| **L** | 300kg | 4轮差速 | 5.5寸轮毂×2 | 产线配送 |
| **XL** | 600kg | 4轮差速 | 6.5寸轮毂×2 | 重载车间 |
| **XXL** | 1200kg | 4轮差速 | 7.5寸轮毂×4 | 港口物流 |

### M级AGV详细规格

| 参数 | 规格 |
|------|------|
| 自重 | 35kg |
| 负载 | 100kg |
| 轮子直径 | 140mm (5.5寸) |
| 驱动电机 | 5.5寸轮毂电机 150W × 2 |
| 从动轮 | ESUN 2.5寸静音避震万向轮 |
| 最大速度 | 1.5m/s |

### 具身智能传感器综合展示

```bash
cd examples

# 运行M级AGV完整展示
python embodied_sensor_showcase.py --grade M

# 运行所有等级测试
python embodied_sensor_showcase.py --all-grades

# 运行特定等级
python embodied_sensor_showcase.py --grade L --duration 5
```

## 硬件配置

### 传感器

| 传感器 | 型号 | 参数 |
|--------|------|------|
| 激光雷达 | 镭神 N10P | 360°, 25m, TOF测距 |
| IMU | ETT10A-PW | 6轴, IP67防水 |
| RGB相机 | 奥比中光 C100 | 1080P, FOV 112° |
| 深度相机 | 奥比中光 Astra Pro Plus | 640×480, 0.4-8m |

### 电机驱动器

| 型号 | 类型 | 控制方式 | 输出电流 |
|------|------|----------|---------|
| 中菱 ZLAC8015D | 一拖二轮毂伺服 | CANopen/RS485 | 15A/30A峰值 |

### 从动轮

| 型号 | 轮径 | 材质 | 承重 |
|------|------|------|------|
| ESUN JQR25310-80A | 2.5寸 | 聚氨酯80A | 135kg/轮 |

## PyBullet仿真

### AGV模型生成

```python
from simulation.agv_model_generator import generate_agv_urdf_detailed, GRADE_CONFIGS

# 生成M级AGV URDF
urdf_path = generate_agv_urdf_detailed('M', '2轮')

# 查看配置
print(GRADE_CONFIGS['M'])
```

### 仿真控制

```python
import pybullet as p
from simulation.agv_model_generator import generate_agv_urdf_detailed

# 连接仿真
client = p.connect(p.DIRECT)
p.setGravity(0, 0, -9.81)

# 加载AGV
urdf = generate_agv_urdf_detailed('M', '2轮')
agv_id = p.loadURDF(urdf, basePosition=[0, 0, 0.15])

# 仿真控制
for i in range(1000):
    p.stepSimulation()

p.disconnect()
```

## 购买链接

| 组件 | 链接 |
|------|------|
| 镭神N10P激光雷达 | https://detail.tmall.com/item.htm?id=661907723595 |
| ETT10A-PW IMU | https://item.taobao.com/item.htm?id=622844097690 |
| ESUN 2.5寸万向轮 | https://detail.tmall.com/item.htm?id=591810849491 |
| 奥比中光 Astra Pro Plus | https://item.taobao.com/item.htm?id=646073233035 |
| 奥比中光 C100/C70 | https://item.taobao.com/item.htm?id=641692244195 |
| 中菱 ZLAC8015D 驱动器 | https://item.taobao.com/item.htm?id=677349695836 |

## 示例脚本

| 脚本 | 描述 |
|------|------|
| `examples/embodied_sensor_showcase.py` | 触觉+力觉+IMU→融合→AGV控制 完整闭环展示 |
| `examples/agv_five_level_demo.py` | AGV五级规格对比演示 |
| `examples/multimodal_sensor_fusion_demo.py` | 多模态传感器融合演示 |
| `sim_demos/run_sensor_fusion.py` | 传感器融合控制仿真 (PyBullet) |
| `sim_demos/run_agv_showcase.py` | AGV等级展示 |
| `sim_demos/run_warehouse.py` | 仓库物流仿真 |
| `sim_demos/run_multi_agv.py` | 多AGV协同仿真 |

## 许可证

MIT License

## GitHub

https://github.com/DIT4FUN/SuperModel
