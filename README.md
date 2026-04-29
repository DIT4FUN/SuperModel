# SuperModel 超模态大模型

> AGV具身智能大脑 | 多模态感知融合 | 自主学习

---

## 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [⭐ 核心目标系统](#-核心目标系统)
- [环境配置与要求](#环境配置与要求)
  - [软件环境](#软件环境)
  - [硬件规格](#硬件规格)
  - [AGV五级规格对照](#agv五级规格对照)
- [安装部署](#安装部署)
  - [基础安装](#基础安装)
  - [训练环境配置](#训练环境配置)
  - [RK3588边缘部署](#rk3588边缘部署)
  - [真实机器人部署](#真实机器人部署)
- [训练流程](#训练流程)
  - [数据采集](#数据采集)
  - [数据格式转换](#数据格式转换)
  - [模型训练](#模型训练)
  - [模型导出与部署](#模型导出与部署)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [硬件配置](#硬件配置)
- [示例脚本](#示例脚本)
- [许可证](#许可证)

---

## 项目简介

SuperModel 是一个超模态大模型具身智能系统，专注于 AGV（自动导引车）机器人的智能控制。系统通过融合视觉、听觉、触觉、力觉、IMU等多模态感知数据，结合超模态大模型实现对复杂环境的理解和自主决策。

## 核心特性

- **超模态感知**: 支持视觉、听觉、触觉、力觉、IMU等多模态传感器融合
- **具身智能**: 基于超模态大模型的决策和规划能力
- **自主学习**: 强化学习框架支持持续优化（Dreamer / World Model）
- **实时控制**: 高性能运动控制（PID、轨迹规划、安全监控）
- **模块化设计**: 传感器、控制、融合模块解耦，易于扩展
- **PyBullet仿真**: 支持多AGV协同仿真和避障测试
- **核心目标系统**: 六大核心目标持续运行，实时决策（见下方详述）
- **VLA端到端模型**: Vision-Language-Action 端到端控制
- **联邦学习**: 支持多机器人协同学习
- **RK3588 NPU边缘部署**: 支持边缘端低延迟推理

---

## 环境配置与要求

### 软件环境

| 组件 | 版本要求 | 说明 |
|------|----------|------|
| Python | ≥ 3.10 | 核心运行环境 |
| PyTorch | ≥ 2.0 | 深度学习框架 |
| CUDA | ≥ 11.8 | GPU加速（如需训练） |
| NumPy | ≥ 1.24 | 数值计算 |
| PyBullet | ≥ 3.0 | 机器人仿真 |
| pytest | ≥ 7.0 | 单元测试 |

**可选组件：**

| 组件 | 版本 | 说明 |
|------|------|------|
| DeepSpeed | ≥ 0.18 | 分布式训练 |
| wandb | ≥ 0.26 | 实验跟踪 |
| tensorboard | ≥ 2.20 | 训练可视化 |
| RKNN Toolkit2 | ≥ 1.4.0 | RK3588 NPU模型转换 |
| Gymnasium | ≥ 0.29 | 强化学习环境 |

#### 训练推荐环境（双卡 RTX 3090 24GB×2）

```bash
# 激活虚拟环境
source /root/llm_env/activate.sh

# 或创建新环境
python3 -m venv venv
source venv/bin/activate

# 安装基础依赖
pip install numpy pybullet pytest

# 安装训练相关依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install transformers datasets accelerate peft deepspeed wandb tensorboard
pip install gymnasium
```

### 硬件规格

#### 边缘计算平台

| 平台 | CPU | NPU算力 | 内存 | 适用等级 |
|------|-----|---------|------|---------|
| **RK3588** | 4×A76@2.4GHz + 4×A55@1.8GHz | 6-12 TOPS | 8-32GB LPDDR5 | S / M / L |
| Jetson Orin NX | 8核 ARM Cortex-A78AE | 22 TOPS | 8-16GB | L / XL |
| Raspberry Pi 5 | 4×A76@2.4GHz | - | 4-8GB | S（开发测试） |

#### 传感器配置

| 传感器 | 型号 | 接口 | 采样率 | 适用等级 |
|--------|------|------|--------|---------|
| 激光雷达 | 镭神 N10P | Ethernet | 25m/360° | 全等级 |
| IMU | ETT10A-PW | UART | 6轴 IP67 | 全等级 |
| RGB相机 | 奥比中光 C100 | USB3.0 | 1080P@30fps | M-L |
| 深度相机 | 奥比中光 Astra Pro Plus | USB3.0 | 640×480 | L-XL |
| 触觉阵列 | 电子皮肤 | SPI | 8×8~32×16 | M-XXL |
| 六维力传感器 | ATI Nano25 | CAN/RS485 | 6维 | L-XXL |

#### 驱动配置

| AGV等级 | 驱动器 | 电机功率 | 控制方式 |
|---------|--------|----------|---------|
| S | TB6612 | 57步进 | PWM |
| M | 中菱 ZLAC8015D ×1 | 150W×2 轮毂 | CANopen |
| L | 中菱 ZLAC8015D ×1 | 150W×4 轮毂 | CANopen |
| XL | 中菱 ZLAC8015D ×2 | 300W×4 轮毂 | CANopen |
| XXL | 中菱 ZLAC8015D ×2 | 500W×4 轮毂 | CANopen |

### AGV五级规格对照

| 等级 | 负载 | 自重 | 最大速度 | 轮子配置 | 典型场景 |
|------|------|------|----------|----------|---------|
| **S** | 30kg | 10kg | 1.0 m/s | 2轮差速 | 小型仓库 / 教育 |
| **M** | 100kg | 35kg | 1.5 m/s | 2轮差速 | 物流分拣 / 医院 |
| **L** | 300kg | 80kg | 1.2 m/s | 4轮差速 | 产线配送 |
| **XL** | 600kg | 150kg | 1.0 m/s | 4轮差速 | 重载车间 |
| **XXL** | 1200kg | 300kg | 0.8 m/s | 4轮差速 | 港口物流 |

#### M级AGV详细规格

| 参数 | 规格 |
|------|------|
| 自重 | 35kg |
| 负载 | 100kg |
| 轮子直径 | 140mm (5.5寸) |
| 驱动电机 | 5.5寸轮毂电机 150W × 2 |
| 从动轮 | ESUN 2.5寸静音避震万向轮 |
| 最大速度 | 1.5m/s |
| 典型场景 | 物流分拣、医院配送、餐饮配送 |

---

## 安装部署

### 基础安装

```bash
# 克隆代码
cd /root/projects
git clone https://github.com/DIT4FUN/SuperModel.git
cd SuperModel

# 方式一：pip 安装
pip install -e .

# 方式二：仅安装基础依赖
pip install numpy pybullet pytest

# 方式三：完整训练依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install transformers datasets accelerate peft deepspeed wandb tensorboard
pip install gymnasium

# 设置 PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH
```

### 训练环境配置

#### 环境变量配置

```bash
# 激活 LLM 训练环境
source /root/llm_env/activate.sh

# 或使用项目自带脚本
cd /root/projects/SuperModel
source scripts/setup_env.sh  # 如有
```

#### 硬件连接验证

```bash
# 检查 GPU
nvidia-smi
# 预期输出：NVIDIA GeForce RTX 3090 × 2

# 检查 CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.device_count()}')"
# 预期输出：CUDA: True, GPU: 2

# 测试多GPU训练
python -c "import torch; import torch.distributed as dist; print('Distributed OK')"
```

### RK3588边缘部署

#### 交叉编译环境（宿主机）

```bash
# 安装交叉编译工具链
sudo apt-get update
sudo apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
sudo apt-get install -y cmake build-essential git python3-pip

# 安装 RK3588 SDK
git clone https://github.com/rockchip-linux/rk3588-sdk.git
cd rk3588-sdk
source envsetup.sh aarch64
```

#### 目标板依赖

```bash
# 在 RK3588 板上安装
pip3 install numpy scipy scikit-learn
pip3 install rknn-toolkit2  # RKNN Toolkit2 for RK3588 NPU
pip3 install opencv-python-headless
pip3 install edge-tts requests websockets
```

#### 模型转换与部署

```bash
# Step 1: 导出 PyTorch 模型
python3 -c "
from src.fusion.cross_modal_fusion import CrossModalTransformer
import torch
model = CrossModalTransformer(grade='M')
model.eval()
torch.jit.trace(model, torch.randn(1, 128)).save('models/supermodel_M.pt')
"

# Step 2: 转换为 RKNN 格式
python3 scripts/convert_to_rknn.py --model models/supermodel_M.onnx --grade M

# Step 3: 一键部署到 RK3588
./scripts/deploy_rknn.sh M 192.168.1.100
```

详细文档见 [docs/RK3588_NPU_DEPLOYMENT.md](docs/RK3588_NPU_DEPLOYMENT.md)

### 真实机器人部署

#### 部署前检查清单

- [ ] 电源电压正常（24V/48V）
- [ ] CAN 总线接口已启用
- [ ] 所有传感器通电
- [ ] 紧急停止按钮可正常触发
- [ ] 驱动器通信正常
- [ ] 激光雷达IP可达
- [ ] 所有传感器读数正常

#### 部署步骤

```bash
# 1. 配置 AGV 等级
python3 examples/real_agv_deploy.py --grade M

# 2. 验证传感器
python3 -m pytest tests/sensor_tests.py -v

# 3. 启动部署管理器
python3 examples/real_agv_deploy.py --grade M --mode production
```

详细文档见 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

---

## 训练流程

### 数据采集

#### 支持的数据源

| 数据源 | 类型 | 说明 |
|--------|------|------|
| SensorManager | 实时 | 多模态传感器实时采集 |
| PyBullet 仿真 | 仿真 | PyBullet 仿真回放数据 |
| MuJoCo 仿真 | 仿真 | MuJoCo 仿真回放数据 |
| ROS2 bag | 回放 | ROS2 录制的 bag 包 |
| HDF5 数据集 | 离线 | 预录制的 HDF5 格式数据 |
| NumPy 文件 | 离线 | .npz 或目录组织的 .npy 文件 |

#### 方式A：从 SensorManager 录制

```python
from src.sensors.manager import SensorManager

manager = SensorManager(grade="M")
manager.start()

frames = []
for i in range(1000):
    frame = manager.get_frame(timeout=1.0)
    if frame:
        frames.append(frame)

manager.stop()

import numpy as np
np.savez(
    "/data/my_recording.npz",
    vision=np.array([f.vision_encoded for f in frames]),
    lidar=np.array([...]),
    tactile=np.array([f.tactile_encoded for f in frames]),
    force=np.array([f.force_encoded for f in frames]),
    imu=np.array([f.imu_encoded for f in frames]),
    actions=np.array([...]),
    rewards=np.array([f.reward for f in frames]),
)
```

#### 方式B：从 PyBullet 仿真回放

```python
from src.simulation.pybullet_sim import PyBulletSimulation

sim = PyBulletSimulation(grade='M')
sim.start()
# ... 运行仿真 ...
sim.export_trajectory("/data/sim_trajectory.h5")
```

### 数据格式转换

将原始数据转换为 HDF5 格式用于训练：

```bash
# 安装依赖
pip install h5py

# 单个文件转换
python scripts/convert_robot_data.py \
    --input /data/my_recording.npz \
    --grade M \
    --output /data/my_recording.h5

# 目录批量转换
python scripts/convert_robot_data.py \
    --input_dir ./raw_sessions/ \
    --output_dir ./h5_data/ \
    --grade M

# 干跑检查
python scripts/convert_robot_data.py \
    --input_dir ./raw_sessions/ \
    --dry_run
```

#### 输出 HDF5 结构

```
/data/run1.h5
├── observations/
│   ├── vision   (N, 512) float32
│   ├── lidar    (N, 128) float32
│   ├── tactile  (N, 64) float32
│   ├── force    (N, 6) float32
│   └── imu      (N, 6) float32
├── actions      (N, 7) float32
├── rewards      (N,) float32
├── dones        (N,) bool
└── timestamps   (N,) float64
```

### 模型训练

#### 批量训练（推荐）

```bash
source /root/llm_env/activate.sh

cd /root/projects/SuperModel

# M 级，批量模式，HDF5 数据
python scripts/train_real_data.py \
    --mode batch \
    --source_type hdf5_dataset \
    --data_root /data/h5_training/ \
    --grade M \
    --batch_size 16 \
    --seq_len 64 \
    --lr 3e-5 \
    --max_steps 50000 \
    --eval_interval 100 \
    --save_interval 5000 \
    --output_dir ./checkpoints/real_training \
    --num_workers 4
```

#### 在线增量训练

```bash
python scripts/train_real_data.py \
    --mode online \
    --source_type sensor_manager \
    --grade M \
    --batch_size 16 \
    --seq_len 32 \
    --lr 1e-5 \
    --max_steps 100000 \
    --output_dir ./checkpoints/online_training
```

#### 多卡分布式训练

```bash
# DeepSpeed 双卡
deepspeed --num_gpus=2 scripts/train_real_data.py \
    --mode batch \
    --data_root /data/h5_training/ \
    --grade M \
    --batch_size 32 \
    --lr 3e-5 \
    --max_steps 100000

# PyTorch 分布式
torchrun --nproc_per_node=2 scripts/train_real_data.py \
    --mode batch \
    --data_root /data/h5_training/ \
    --grade M
```

#### 继续训练

```bash
python scripts/train_real_data.py \
    --mode batch \
    --data_root /data/h5_training/ \
    --grade M \
    --resume ./checkpoints/real_training/supermodel_M_xxx/step_5000.pt \
    --max_steps 100000
```

#### 模态配置

| 等级 | Vision Dim | LiDAR Dim | Tactile Dim | Force Dim | IMU Dim | Action Dim |
|------|------------|-----------|-------------|-----------|---------|------------|
| S    | 512        | 128       | 64          | 6         | 6       | 6          |
| M    | 512        | 128       | 64          | 6         | 6       | 7          |
| L    | 768        | 256       | 128         | 6         | 6       | 7          |
| XL   | 1024       | 512       | 256         | 6         | 6       | 8          |
| XXL  | 1536       | 512       | 512         | 6         | 6       | 12         |

#### 推荐训练配置（双卡 RTX 3090 24GB×2）

**M 级 AGV：**

```bash
python scripts/train_real_data.py \
    --mode batch \
    --grade M \
    --batch_size 16 \
    --seq_len 64 \
    --lr 3e-5 \
    --grad_clip 1.0 \
    --accumulation_steps 4 \
    --fp16
```

**L 级 AGV：**

```bash
python scripts/train_real_data.py \
    --mode batch \
    --grade L \
    --batch_size 8 \
    --seq_len 32 \
    --lr 1e-5 \
    --grad_clip 0.5 \
    --accumulation_steps 8 \
    --fp16
```

### 模型导出与部署

#### 检查点输出结构

```
./checkpoints/real_training/supermodel_M_20260428_220000/
├── config.json              # 配置快照
├── step_1000.pt            # 中间检查点
├── step_5000.pt            # ...
└── final.pt                # 最终模型
```

#### 加载检查点

```python
import torch
from src.learning.world_model import WorldModel, WorldModelConfig, WORLD_MODEL_GRADES

checkpoint = torch.load("final.pt")
checkpoint.keys()
# ['model', 'optimizer', 'scheduler', 'global_step', 'grade']

# 加载模型
model = WorldModel(
    obs_dims={'vision': 512, 'lidar': 128, 'tactile': 64, 'force': 6, 'imu': 6},
    action_dim=7,
    config=WORLD_MODEL_GRADES['M']
)
model.load_state_dict(checkpoint['model'])
```

详细文档见 [docs/TRAINING_REAL_DATA.md](docs/TRAINING_REAL_DATA.md)

---

## ⭐ 核心目标系统 (Core Goals System)

SuperModel 内置六大核心目标,持续运行于后台,驱动所有决策:

### 核心目标层级 (Priority 0 = 最高优先)

| 优先级 | 目标 | 描述 | 执行器 |
|--------|------|------|--------|
| **P0** | **保护人类安全** | 绝对优先级,任何情况下不可被覆盖 | `SafetyShield` |
| **P1** | **遵循人类指令** | 在P0安全前提下执行指令 | `DecisionMaking` |
| **P2** | **善良品质** | 同理心/公平/诚实透明/拒绝伤害 | `ValueJudgment` |
| **P3** | **热爱世界** | 环境保护/促进协作/传递正能量 | `ValueJudgment` |
| **P4** | **自我生存安全** | 硬件保护/能源管理/健康监测 | `SelfPreservation` |
| **P5** | **自我进化** | 知识积累/技能提升/探索创新 | `SelfEvolution` |

### 系统架构

```
sensors → context_understanding → decision_making
                                        ↓
                              goal_dispatcher (always running @ 50Hz)
                                        ↓
                              ┌──────────┴──────────┐
                           safety_shield          interaction
                           (P0绝对优先)              ↓
                            ↓                    execute
                 core_goals (P0-P5)            ↓
                  实时评估与协调              observe
```

### 使用方式

```python
from src.core import CoreBrain

# 创建核心大脑
brain = CoreBrain(grade="M")

# 启动持续执行 (50Hz周期)
brain.start()

# 每周期更新传感器数据
brain.update_context(
    robot_position=np.array([0.0, 0.0, 0.0]),
    robot_velocity=np.array([0.5, 0.0, 0.0]),
    laser_ranges=np.array([5.0] * 360),
    human_positions=[np.array([3.0, 0.0, 0.0])],
    robot_battery_level=0.9,
    robot_temperature=35.0,
)

# 获取当前决策
decision = brain.decide()
print(f"Action: {decision.action}")
print(f"Type: {decision.decision_type}")
print(f"Reasoning: {decision.reasoning}")

# 获取所有目标状态
scores = brain.get_all_scores()
print(f"Goal scores: {scores}")

# 获取完整状态
status = brain.get_status()
print(f"Safety: {status['safety_shield']['safety_score']}")
print(f"Learning progress: {status['self_evolution']['learning_progress']}")

# 停止
brain.stop()
```

### 单步模式 (手动控制)

```python
# 决策+执行一体化
decision, execution = brain.step(instruction="前进")
```

### 紧急停止

```python
# 触发紧急停止
brain.trigger_emergency_stop("检测到危险")

# 释放紧急停止
brain.release_emergency_stop()
```

### 测试

```bash
# 运行核心目标系统测试
python -m pytest tests/core_tests.py -v

# 53项测试全部通过
```

## 项目结构

```
SuperModel/
├── src/                          # 源代码
│   ├── sensors/                  # 多模态传感器接口
│   │   ├── vision.py             # 深度相机 (RealSense/Astra)
│   │   ├── audio.py              # 双耳麦克风阵列
│   │   ├── tactile.py            # 电子皮肤触觉阵列
│   │   ├── force.py              # 六维力矩传感器
│   │   ├── imu.py                # IMU传感器
│   │   ├── lidar.py              # 激光雷达
│   │   ├── encoders.py           # 特征编码器
│   │   └── manager.py            # 传感器管理器
│   ├── fusion/                   # 跨模态融合网络
│   │   ├── cross_modal_fusion.py # 注意力融合Transformer
│   │   └── sensor_fusion.py      # 互补滤波/EKF/多传感器融合
│   ├── perception/               # 感知与场景理解
│   │   └── scene_understanding.py
│   ├── learning/                 # 自主学习框架
│   │   ├── world_model.py        # RSSM/Dreamer世界模型
│   │   ├── dreamer_agent.py      # Dreamer智能体
│   │   ├── autonomous_learning.py # 自主学习
│   │   └── self_supervised.py    # 自监督学习
│   ├── embodied/                 # ⭐ 具身智能核心
│   │   ├── vla_model.py         # Vision-Language-Action 模型
│   │   ├── vla_inference.py      # VLA推理引擎
│   │   ├── embodied_pipeline.py # 具身流水线
│   │   ├── real_agv_interface.py # 真实AGV接口
│   │   ├── deployment.py         # 部署管理
│   │   ├── federated_learning.py # 联邦学习
│   │   ├── behavior_tree.py      # 行为树引擎
│   │   ├── scene_task_planner.py # 场景任务规划
│   │   └── agv_swarm_coordinator.py # 多AGV协调
│   ├── control/                  # 动作控制模块
│   │   ├── motion.py             # 运动控制
│   │   ├── trajectory.py         # 轨迹规划
│   │   ├── impedance.py          # 阻抗控制
│   │   ├── mpc.py               # 模型预测控制
│   │   ├── agv.py               # AGV运动学
│   │   ├── supervisor.py         # 控制监管
│   │   ├── safety_controller.py   # 安全控制器
│   │   └── ros2_interface.py     # ROS2接口
│   ├── core/                     # ⭐ 核心目标系统
│   │   ├── core_goals.py         # 六大核心目标定义与优先级
│   │   ├── safety_shield.py      # P0安全护盾
│   │   ├── value_judgment.py     # P2/P3价值判断
│   │   ├── self_preservation.py  # P4自我保存
│   │   ├── self_evolution.py     # P5自我进化
│   │   ├── context_understanding.py  # 上下文理解
│   │   ├── decision_making.py    # 决策引擎
│   │   ├── interaction.py        # 环境交互接口
│   │   ├── goal_dispatcher.py    # 持续执行引擎 (50Hz)
│   │   └── core_brain.py          # 核心大脑
│   ├── simulation/               # 仿真环境
│   │   ├── pybullet_sim.py       # PyBullet仿真
│   │   ├── agv_model_generator.py # AGV URDF模型生成
│   │   ├── mujoco_sim.py         # MuJoCo仿真
│   │   └── gym_env.py            # Gymnasium环境
│   ├── memory/                   # 长期记忆系统
│   ├── evaluation/              # 评估模块
│   └── utils_pkg/                # 工具包
├── scripts/                      # 训练与部署脚本
│   ├── train_real_data.py       # 真实数据训练 ⭐
│   ├── train_world_model.py      # World Model训练
│   ├── convert_robot_data.py     # 数据格式转换 ⭐
│   ├── multi_sensor_data_collection.py # 多传感器采集
│   ├── deploy_rknn.sh           # RK3588一键部署
│   ├── convert_to_rknn.py       # RKNN模型转换
│   └── run_on_rk3588.py         # RK3588运行时
├── configs/                      # AGV五级配置文件
│   ├── agv_S.yaml
│   ├── agv_M.yaml
│   ├── agv_L.yaml
│   ├── agv_XL.yaml
│   └── agv_XXL.yaml
├── sim_demos/                    # PyBullet仿真演示
│   ├── run_gui.py               # S形路径避障
│   ├── run_warehouse.py          # 仓库物流仿真
│   ├── run_multi_agv.py         # 多AGV协同
│   └── run_agv_showcase.py      # AGV等级展示
├── examples/                     # 示例脚本
│   ├── embodied_sensor_showcase.py # 具身传感器展示
│   ├── embodied_grasp_demo.py    # 具身抓取演示
│   ├── complete_system_demo.py   # 完整系统演示
│   ├── multimodal_sensor_fusion_demo.py # 多模态融合演示
│   ├── real_agv_deploy.py       # 真实AGV部署
│   ├── autonomous_patrol_demo.py # 自主巡逻演示
│   └── ...
├── tests/                        # 测试用例 (2200+项通过)
│   ├── sensor_tests.py
│   ├── fusion_tests.py
│   ├── control_integration_tests.py
│   ├── five_grade_integration_tests.py
│   ├── core_tests.py             # 核心目标系统测试 (53项)
│   └── ...
├── docs/                         # 设计文档
│   ├── DEPLOYMENT.md            # 真实机器人部署指南
│   ├── RK3588_NPU_DEPLOYMENT.md # RK3588边缘部署指南
│   ├── TRAINING_REAL_DATA.md    # 真实数据训练指南 ⭐
│   ├── QUICKSTART.md           # 快速上手
│   ├── HARDWARE_SPEC.md        # 硬件规格
│   ├── AGV_SPEC.md             # AGV技术规格
│   ├── DESIGN.md               # 架构设计
│   ├── MODULE_INDEX.md         # 模块索引
│   └── design/
│       ├── SYSTEM_ARCHITECTURE.md
│       ├── MODULE_INTERFACE.md  # 详细接口设计
│       └── AGV_FIVE_LEVEL_*.md  # 五级规格文档
├── hardware/                    # 硬件接口层
│   └── agv_interface.py
├── data/                       # 数据目录
├── checkpoints/               # 模型检查点
└── README.md
```

## 快速开始

### 1. 安装

```bash
cd /root/projects/SuperModel
pip install numpy pybullet pytest
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install transformers datasets accelerate peft deepspeed wandb tensorboard
export PYTHONPATH=$PWD:$PYTHONPATH
```

### 2. 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行核心目标系统测试
python -m pytest tests/core_tests.py -v

# 运行传感器测试
python -m pytest tests/sensor_tests.py -v

# 运行PyBullet仿真测试
python -m pytest tests/pybullet_sim_tests.py -v
```

### 3. 运行仿真演示

```bash
cd /root/projects/SuperModel/sim_demos

# S形路径避障仿真
python run_gui.py

# 仓库物流仿真
python run_warehouse.py

# 多AGV协同仿真
python run_multi_agv.py

# AGV等级展示
python run_agv_showcase.py
```

### 4. 快速训练示例

```bash
# 数据转换
python scripts/convert_robot_data.py \
    --input /data/my_recording.npz \
    --grade M \
    --output /data/my_recording.h5

# 批量训练
python scripts/train_real_data.py \
    --mode batch \
    --data_root /data/h5_training/ \
    --grade M \
    --batch_size 16 \
    --seq_len 64 \
    --lr 3e-5 \
    --max_steps 50000 \
    --fp16
```

### 5. 核心目标系统示例

```python
from src.core import CoreBrain
import numpy as np

# 创建核心大脑
brain = CoreBrain(grade="M")
brain.start()

# 更新传感器数据
brain.update_context(
    robot_position=np.array([0.0, 0.0, 0.0]),
    robot_velocity=np.array([0.5, 0.0, 0.0]),
    laser_ranges=np.array([5.0] * 360),
    human_positions=[np.array([3.0, 0.0, 0.0])],
    robot_battery_level=0.9,
    robot_temperature=35.0,
)

# 获取决策
decision = brain.decide()
print(f"Action: {decision.action}")
print(f"Type: {decision.decision_type}")

# 获取目标状态
scores = brain.get_all_scores()
print(f"Goal scores: {scores}")

brain.stop()
```

## 示例脚本

### 训练相关

| 脚本 | 描述 |
|------|------|
| `scripts/train_real_data.py` | 真实数据训练（批量/在线） |
| `scripts/convert_robot_data.py` | 多源数据转换为 HDF5 |
| `scripts/train_world_model.py` | World Model 训练 |
| `scripts/multi_sensor_data_collection.py` | 多传感器同步采集 |

### 仿真演示

| 脚本 | 描述 |
|------|------|
| `sim_demos/run_gui.py` | S形路径避障仿真 |
| `sim_demos/run_warehouse.py` | 仓库物流仿真 |
| `sim_demos/run_multi_agv.py` | 多AGV协同仿真 |
| `sim_demos/run_agv_showcase.py` | AGV等级展示 |
| `sim_demos/run_sensor_fusion.py` | 传感器融合控制仿真 |

### 具身智能

| 脚本 | 描述 |
|------|------|
| `examples/embodied_sensor_showcase.py` | 触觉+力觉+IMU→融合→AGV控制 完整闭环展示 |
| `examples/embodied_grasp_demo.py` | 具身感知抓取演示 |
| `examples/complete_system_demo.py` | 端到端系统演示 |
| `examples/real_agv_deploy.py` | 真实AGV部署脚本 |
| `examples/multimodal_sensor_fusion_demo.py` | 多模态传感器融合演示 |
| `examples/autonomous_patrol_demo.py` | 自主巡逻演示 |
| `examples/agv_five_level_demo.py` | AGV五级规格对比演示 |

### 核心目标系统

| 脚本 | 描述 |
|------|------|
| `examples/core_goals_demo.py` | 核心目标系统演示 |

## 硬件配置

### 传感器

| 传感器 | 型号 | 参数 | 购买链接 |
|--------|------|------|---------|
| 激光雷达 | 镭神 N10P | 360°, 25m, TOF测距 | [购买链接](https://detail.tmall.com/item.htm?id=661907723595) |
| IMU | ETT10A-PW | 6轴, IP67防水 | [购买链接](https://item.taobao.com/item.htm?id=622844097690) |
| RGB相机 | 奥比中光 C100 | 1080P, FOV 112° | [购买链接](https://item.taobao.com/item.htm?id=641692244195) |
| 深度相机 | 奥比中光 Astra Pro Plus | 640×480, 0.4-8m | [购买链接](https://item.taobao.com/item.htm?id=646073233035) |

### 驱动与电机

| 型号 | 类型 | 控制方式 | 输出电流 | 购买链接 |
|------|------|----------|---------|---------|
| 中菱 ZLAC8015D | 一拖二轮毂伺服 | CANopen/RS485 | 15A/30A峰值 | [购买链接](https://item.taobao.com/item.htm?id=677349695836) |
| ESUN 2.5寸万向轮 | 从动轮 | - | 135kg/轮 | [购买链接](https://detail.tmall.com/item.htm?id=591810849491) |

## 文档索引

| 文档 | 说明 |
|------|------|
| [QUICKSTART.md](docs/QUICKSTART.md) | 快速上手指南 |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | 真实机器人部署指南 |
| [RK3588_NPU_DEPLOYMENT.md](docs/RK3588_NPU_DEPLOYMENT.md) | RK3588 边缘部署指南 |
| [TRAINING_REAL_DATA.md](docs/TRAINING_REAL_DATA.md) | 真实数据训练指南 |
| [AGV_SPEC.md](docs/AGV_SPEC.md) | AGV技术规格 |
| [HARDWARE_SPEC.md](docs/HARDWARE_SPEC.md) | 硬件规格说明 |
| [DESIGN.md](docs/DESIGN.md) | 架构设计文档 |
| [MODULE_INDEX.md](docs/MODULE_INDEX.md) | 模块索引 |
| [PRACTICAL_DEPLOYMENT.md](docs/PRACTICAL_DEPLOYMENT.md) | 实用部署指南 |

## 许可证

MIT License

## GitHub

https://github.com/DIT4FUN/SuperModel
