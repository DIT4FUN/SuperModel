# SuperModel - 超模态机器人具身智能大脑

> 🤖 融合双耳声觉 + 双目视觉 + 触觉 + 力觉 + IMU 的具身智能大脑

## 项目目标

构建超模态大模型机器人具身智能大脑，融合多模态感知、自主学习与实时控制能力。

## 核心创新范式

- **构建式学习**: 不依赖人工标注，自主构建知识
- **渐进式演化**: 实时采集、持续学习、自我进化
- **具身感知**: 双耳声觉 + 双目视觉 + 电子皮肤 + 力矩传感 + IMU
- **时空约束**: 有限任务目标下的小型高效模型设计

## 核心模块

| 模块 | 描述 | 状态 |
|------|------|------|
| `sensors/` | 多模态传感器接口 (视觉/听觉/触觉/力觉/IMU) | ✅ 完成 |
| `fusion/` | 跨模态注意力融合网络 | ✅ 完成 |
| `perception/` | 多模态特征提取与统一表示 | ✅ 完成 |
| `learning/` | 自主学习框架 (对比学习/世界模型/好奇心) | ✅ 完成 |
| `learning/world_model.py` | **Dreamer-style World Model** (RSSM/想象训练) | ✅ 完成 |
| `control/` | 运动控制 (PID/阻抗/技能库/规划) | ✅ 完成 |
| `simulation/` | 基础物理仿真环境 | ✅ 完成 |
| `docs/` | 架构设计与接口文档 | ✅ 完成 |
| `tests/` | 全套单元测试 (126项 + World Model专项8项全部通过) | ✅ 完成 |

## 🌟 World Model (世界模型)

实现 Dreamer 风格的世界模型，包含 RSSM (Recurrent State Space Model)：

```python
from learning.world_model import create_world_model_agent, get_world_model_spec

# 创建 M 级世界模型智能体
agent = create_world_model_agent('M', obs_dims, action_dim)

# 选择动作
action = agent.select_action(observations, deterministic=True)

# 存储经验
agent.store_transition(obs, action, reward, next_obs, done)

# 训练
losses = agent.train_step(batch_size=64)
```

### World Model 架构

```
观测 o_t → [Encoder] → obs_embed
                          ↓
动作 a_{t-1} + 隐状态 h_{t-1} + z_{t-1} → [RSSM] → h_t, z_t
                          ↓                              ↓
                     [先验 p]                      [后验 q]
                          ↓                              ↓
                     z_t 的分布 ←————— obs_embed ——→ z_t 的分布
                          ↓
                       [Decoder] → 预测观测
                          ↓
                       [Reward] → 预测奖励
```

### AGV 五级 World Model 配置

| 等级 | 隐状态维度 | 隐藏维度 | 想象步数 | 参数量 |
|------|-----------|----------|----------|--------|
| S | 128 | 256 | 10 | ~1M |
| M | 256 | 512 | 15 | ~5M |
| L | 512 | 1024 | 20 | ~20M |
| XL | 768 | 1536 | 25 | ~50M |
| XXL | 1024 | 2048 | 30 | ~100M |

## AGV五级规格体系

从 S 级（教育）到 XXL 级（旗舰），覆盖不同算力/成本需求：

| 等级 | 定位 | 算力 | 触觉阵列 | 力觉轴数 | IMU采样 |
|------|------|------|----------|----------|---------|
| S | 教育/实验 | < 5 TOPS | 8×8 @ 50Hz | 3轴 | 100Hz |
| M | 标准助手 | 5-20 TOPS | 16×16 @ 100Hz | 6轴 | 200Hz |
| L | 专业工业 | 20-100 TOPS | 24×24 @ 200Hz | 6轴 | 500Hz |
| XL | 高性能 | 100-300 TOPS | 32×32 @ 500Hz | 6轴 | 1000Hz |
| XXL | 旗舰全功能 | > 300 TOPS | 48×48 @ 1000Hz | 6轴 | 2000Hz |

详细规格见 [AGV_GRADE_SPEC.md](docs/design/AGV_GRADE_SPEC.md)

## 快速开始

```bash
# 克隆项目
git clone https://github.com/DIT4FUN/SuperModel.git
cd SuperModel

# 安装依赖
pip install torch numpy scipy

# 运行传感器测试
python3 -m pytest tests/sensor_tests.py -v

# 运行融合测试
python3 -m pytest tests/fusion_tests.py -v
```

## 使用示例

### 传感器数据采集

```python
from src.sensors.vision import BinocularCamera
from src.sensors.audio import BinauralMic
from src.sensors.tactile import TactileArray
from src.sensors.force import ForceTorqueSensor
from src.sensors.imu import IMUSensor

# 双目视觉
with BinocularCamera() as cam:
    frame = cam.capture()
    depth = cam.get_depth_map(frame)
    points = cam.get_point_cloud(frame)

# 双耳听觉
with BinauralMic() as mic:
    audio = mic.capture()
    direction = mic.get_sound_direction(audio)

# 触觉感知
tactile = TactileArray(array_size=(16, 16))
tactile.open()
frame = tactile.capture()
contacts = tactile.detect_contacts(frame)

# 六维力矩
force_sensor = ForceTorqueSensor()
force_sensor.open()
wrench = force_sensor.capture()

# IMU + 姿态估计
imu = IMUSensor()
imu.open()
frame = imu.capture()

from src.sensors.imu import PoseEstimator
estimator = PoseEstimator(algorithm='madgwick', sample_rate=200)
pose = estimator.update(frame.accel, frame.gyro)
```

### 跨模态融合

```python
import torch
from src.perception import CrossModalFusion, MultimodalInput, FusionConfig

config = FusionConfig(
    vision_dim=512, audio_dim=128, tactile_dim=64,
    force_dim=32, imu_dim=64, hidden_dim=256, num_heads=4
)
fusion = CrossModalFusion(config)

multimodal = MultimodalInput(
    vision=torch.randn(2, 512),
    audio=torch.randn(2, 128),
    tactile=torch.randn(2, 64),
    force=torch.randn(2, 32),
    imu=torch.randn(2, 64)
)

unified_features = fusion(multimodal)
```

### 运动控制

```python
from src.control.motion import MotionController, ControlMode
from src.control.impedance import ImpedanceController, ImpedanceParams
from src.control.skill import SkillLibrary

# PID 关节位置控制
controller = MotionController(num_joints=6, control_rate=100)
torque = controller.compute_joint_torque(target_position=np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0]))

# 阻抗控制
imp_params = ImpedanceParams.default_6d()
imp_ctrl = ImpedanceController(imp_params)
torque = imp_ctrl.compute_torque(
    desired_position=np.array([0.5, 0.0, 0.3]),
    desired_velocity=np.zeros(3),
    current_position=np.array([0.52, 0.01, 0.28]),
    current_velocity=np.zeros(3),
    external_wrench=np.zeros(6),
    jacobian=np.random.randn(6, 6)
)

# 技能库
lib = SkillLibrary()
skill = lib.create_skill("move_to", {"target": [0.5, 0.0, 0.3]})
```

### 世界模型 (Dreamer-style)

```python
from src.learning.world_model import (
    create_world_model_agent, get_world_model_spec,
    WORLD_MODEL_GRADES, ReplayBuffer
)

# 创建 AGV-M 级世界模型智能体
obs_dims = {
    'vision': 512, 'audio': 128,
    'tactile': 64, 'force': 32, 'imu': 64
}
agent = create_world_model_agent('M', obs_dims, action_dim=6)

# 与环境交互
obs = {
    'vision': np.random.randn(512),
    'audio': np.random.randn(128),
    'tactile': np.random.randn(64),
    'force': np.random.randn(32),
    'imu': np.random.randn(64)
}
action = agent.select_action(obs)
agent.store_transition(obs, action, reward=0.5, next_obs=obs, done=False)

# 训练步骤
losses = agent.train_step(batch_size=32)

# AGV 五级配置预览
for grade in ['S', 'M', 'L', 'XL', 'XXL']:
    spec = get_world_model_spec(grade)
    print(f"{grade}: latent={spec.latent_dim}, hidden={spec.hidden_dim}")
```

### 仿真环境

```python
from src.simulation.environment import RobotSimulator, SensorSimulator, SimConfig

config = SimConfig(dt=0.01, num_joints=6, position_noise=0.001)
sim = RobotSimulator(config)
sensor_sim = SensorSimulator(sim, config)

# 仿真控制循环
for _ in range(1000):
    torque = controller.step(target_position, mode=ControlMode.JOINT_POSITION)
    state = sim.step(torque)
    imu_data = sensor_sim.get_imu_data()
```

## 项目结构

```
SuperModel/
├── src/
│   ├── sensors/          # 传感器接口
│   │   ├── vision.py    # 双目相机 + 深度处理
│   │   ├── audio.py     # 双耳麦克风 + 声源定位
│   │   ├── tactile.py   # 电子皮肤 + 压力处理
│   │   ├── force.py     # 六维力矩传感器
│   │   └── imu.py       # IMU + 姿态估计
│   ├── perception/      # 感知融合
│   │   └── cross_modal_fusion.py
│   ├── fusion/          # 跨模态融合网络
│   ├── learning/        # 自主学习框架
│   │   ├── self_supervised.py  # 对比学习/好奇心/自主学习
│   │   └── world_model.py      # Dreamer-style 世界模型
│   ├── control/         # 运动控制
│   │   ├── motion.py    # PID + 轨迹控制
│   │   ├── impedance.py # 阻抗/导纳/协作控制
│   │   ├── skill.py     # 技能库
│   │   └── planner.py   # HTN 任务规划
│   └── simulation/      # 仿真环境
├── tests/
│   ├── sensor_tests.py  # 传感器单元测试 (43 tests)
│   ├── fusion_tests.py  # 融合网络测试 (24 tests)
│   ├── control_tests.py # 控制模块测试 (59 tests)
│   └── test_world_model.py # 世界模型专项测试 (8 tests)
├── configs/
│   └── default.yaml     # 项目配置
└── docs/
    ├── architecture/    # 架构设计
    └── design/          # 详细设计文档
        ├── AGV_GRADE_SPEC.md   # AGV五级规格表
        └── MODULE_INTERFACE.md # 模块接口设计
```

## 接口文档

- [模块接口设计](docs/design/MODULE_INTERFACE.md) — 完整的API接口定义
- [AGV五级规格表](docs/design/AGV_GRADE_SPEC.md) — S/ML/XL/XXL各等级参数
- [架构设计](docs/architecture/SUPER_MODEL_ARCHITECTURE.md) — 整体架构图

## 测试状态

| 测试套件 | 测试数 | 状态 |
|----------|--------|------|
| sensor_tests.py | 43 | ✅ 全部通过 |
| fusion_tests.py | 24 | ✅ 全部通过 |

## 技术栈

- **深度学习**: PyTorch
- **科学计算**: NumPy, SciPy
- **机器人框架**: ROS2 Humble (规划)
- **硬件**: NVIDIA Jetson / Intel RealSense / ATI Force
- **部署**: ONNX / TensorRT

## License

MIT

## GitHub

https://github.com/DIT4FUN/SuperModel
