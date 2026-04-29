# SuperModel 真实数据训练指南

## 概述

本指南说明如何使用真实传感器数据训练 SuperModel 的 World Model。

---

## 数据流程总览

```
真实传感器/仿真器
        ↓
数据录制 (SensorManager / PyBullet / RealSense)
        ↓
数据格式转换 (convert_robot_data.py)
        ↓
HDF5 训练数据集
        ↓
训练脚本 (train_real_data.py)
        ↓
模型检查点 (.pt)
```

---

## 支持的数据源

| 数据源 | 类型 | 说明 |
|--------|------|------|
| SensorManager | 实时 | 多模态传感器实时采集 |
| PyBullet 仿真 | 仿真 | PyBullet 仿真回放数据 |
| MuJoCo 仿真 | 仿真 | MuJoCo 仿真回放数据 |
| ROS2 bag | 回放 | ROS2 录制的 bag 包 |
| HDF5 数据集 | 离线 | 预录制的 HDF5 格式数据 |
| NumPy 文件 | 离线 | .npz 或目录组织的 .npy 文件 |
| CSV 轨迹 | 离线 | 仅动作-奖励轨迹（需填充零） |

---

## 第一步：准备数据

### 方式 A：从 SensorManager 录制

```python
from src.sensors.manager import SensorManager

manager = SensorManager(grade="M")
manager.start()

# 录制 1000 帧
frames = []
for i in range(1000):
    frame = manager.get_frame(timeout=1.0)
    if frame:
        frames.append(frame)

manager.stop()

# 保存为 npz
import numpy as np
np.savez(
    "/data/my_recording.npz",
    vision=np.array([f.vision_encoded for f in frames]),
    lidar=np.array([...]),  # 如果有
    tactile=np.array([f.tactile_encoded for f in frames]),
    force=np.array([f.force_encoded for f in frames]),
    imu=np.array([f.imu_encoded for f in frames]),
    actions=np.array([...]),
    rewards=np.array([f.reward for f in frames]),
    dones=np.array([f.done for f in frames]),
    timestamps=np.array([f.timestamp for f in frames]),
)
```

### 方式 B：从 PyBullet 仿真回放

```python
# 在仿真环境中保存数据
from src.simulation.pybullet_sim import PyBulletSimulation

sim = PyBulletSimulation(grade="M")
sim.start()

# ... 运行仿真 ...

# 仿真结束后导出
sim.export_trajectory("/data/sim_trajectory.h5")
```

---

## 第二步：转换为 HDF5 格式

### 方式 A：SuperModel 原生数据格式

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

# 干跑（不实际转换）
python scripts/convert_robot_data.py \
    --input_dir ./raw_sessions/ \
    --output_dir ./h5_data/ \
    --dry_run
```

### 方式 B：从 ModelScope/LeRobot 数据集转换

ModelScope 上的 LeRobot 数据集（如机械臂操控任务）可以转换为 SuperModel 格式：

```bash
# 安装依赖
pip install h5py pandas pyarrow

# 转换 lerobot 数据集
python scripts/convert_lerobot_data.py \
    --input /path/to/lerobot_parquet_file \
    --dataset berkeley_fanuc_manipulation \
    --grade M \
    --output /data/h5_training/fanuc.h5
```

**已测试可转换的 LeRobot 数据集：**

| 数据集 | 样本数 | 原始 action_dim | 适用等级 | 描述 |
|--------|--------|-----------------|----------|------|
| berkeley_fanuc_manipulation | 62,613 | 7 | AGV-M | Fanuc机械臂操控 |
| pusht_keypoints | 25,650 | 2 | AGV-S | PushT任务 |
| koch_pick_place_1_lego_raph | 32,951 | 6 | AGV-S | 拾取放置任务 |

**AGV 等级与 action_dim 对照：**

| 等级 | action_dim | 说明 |
|------|-----------|------|
| S | 6 | 小型AGV |
| M | 7 | 中型AGV (含夹爪) |
| L | 7 | 大型AGV |
| XL | 8 | 超大型AGV |
| XXL | 12 | 超重载AGV |

### 输出 HDF5 结构

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

---

## 第三步：训练

### 批量训练（推荐）

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

### 在线增量训练（实时传感器）

```bash
python scripts/train_real_data.py \
    --mode online \
    --source_type sensor_manager \
    --grade M \
    --batch_size 16 \
    --seq_len 32 \
    --lr 1e-5 \
    --max_steps 100000 \
    --output_dir ./checkpoints/online_training \
    --num_workers 0
```

### 多卡分布式

```bash
# 双卡 DeepSpeed
deepspeed --num_gpus=2 scripts/train_real_data.py \
    --mode batch \
    --data_root /data/h5_training/ \
    --grade M \
    --batch_size 32 \
    --lr 3e-5 \
    --max_steps 100000

# 或 PyTorch 分布式
torchrun --nproc_per_node=2 scripts/train_real_data.py \
    --mode batch \
    --data_root /data/h5_training/ \
    --grade M
```

### 继续训练

```bash
python scripts/train_real_data.py \
    --mode batch \
    --data_root /data/h5_training/ \
    --grade M \
    --resume ./checkpoints/real_training/supermodel_M_xxx/step_5000.pt \
    --max_steps 100000
```

---

## 模态配置

### 启用/禁用特定模态

```bash
# 仅视觉 + IMU（适用于无触觉/力觉传感器的情况）
python scripts/train_real_data.py \
    --mode batch \
    --data_root /data/vision_only/ \
    --grade M \
    --no_tactile \
    --no_force \
    --no_lidar
```

### AGV 等级与维度

| 等级 | Vision Dim | LiDAR Dim | Tactile Dim | Force Dim | IMU Dim | Action Dim |
|------|------------|-----------|-------------|-----------|---------|------------|
| S    | 512        | 128       | 64          | 6         | 6       | 6          |
| M    | 512        | 128       | 64          | 6         | 6       | 7          |
| L    | 768        | 256       | 128         | 6         | 6       | 7          |
| XL   | 1024       | 512       | 256         | 6         | 6       | 8          |
| XXL  | 1536       | 512       | 512         | 6         | 6       | 12         |

---

## 输出结构

```
./checkpoints/real_training/supermodel_M_20260428_220000/
├── config.json              # 配置快照
├── step_1000.pt            # 中间检查点
├── step_5000.pt            # ...
└── final.pt                # 最终模型
```

### 检查点内容

```python
checkpoint = torch.load("final.pt")
checkpoint.keys()
# ['model', 'optimizer', 'scheduler', 'global_step', 'grade']

# 加载模型
from src.learning.world_model import WorldModel, WorldModelConfig, WORLD_MODEL_GRADES
model = WorldModel(obs_dims={...}, action_dim=7, config=WORLD_MODEL_GRADES['M'])
model.load_state_dict(checkpoint['model'])
```

---

## 动态增量学习系统

SuperModel 支持三种增量学习模式，让模型在使用过程中持续进化：

### 三种学习模式

| 模式 | 触发条件 | 训练步数 | 学习率 | 用途 |
|------|---------|---------|--------|------|
| **Quick** | 发现有价值数据 | 50步 | 5e-5 | 即时知识更新 |
| **Idle** | 模型空闲5分钟+ | 200步 | 3e-5 | 中时增量学习 |
| **Deep** | 凌晨2-6点 | 5000步 | 1e-5 | 夜间大量数据训练 |

### 使用动态学习器

```bash
# 查看学习器状态
python scripts/dynamic_learner.py status

# 提交快速学习任务
python scripts/dynamic_learner.py submit \
    --mode quick \
    --data /data/h5_fanuc/fanuc_manipulation.h5 \
    --grade M \
    --steps 50

# 提交空闲学习任务
python scripts/dynamic_learner.py idle \
    --data /data/h5_training/ \
    --grade M

# 提交深度学习任务（夜间长时训练）
python scripts/dynamic_learner.py deep \
    --data /data/h5_fanuc/fanuc_manipulation.h5 \
    --grade M \
    --hours 8

# 查看任务历史
python scripts/dynamic_learner.py history

# 启动自动调度器（管理三种模式自动切换）
python scripts/dynamic_learner.py start
```

### Python API

```python
from scripts.dynamic_learner import DynamicLearner, LearningScheduler

# 创建学习器
learner = DynamicLearner()

# 快速学习
learner.submit_quick_learning(
    data_path="/data/h5_fanuc/fanuc_manipulation.h5",
    grade="M"
)

# 启动调度器（自动管理三种模式）
scheduler = LearningScheduler(learner)
scheduler.start()

# 查看状态
status = learner.get_status()
print(f"Model version: {status['model_version']}")
print(f"Total training steps: {status['total_training_steps']}")
```

### 学习状态持久化

- Checkpoint目录: `checkpoints/dynamic/`
- 状态文件: `checkpoints/dynamic/learner_state.json`
- 任务历史自动保存

---

## 常见问题

### Q: 训练 loss 不下降

1. 数据归一化问题 — 检查 HDF5 中数据范围
2. 学习率不合适 — 尝试 1e-5 到 1e-4 之间
3. batch size 太小 — 增加到 32 或 64
4. 模态数据全零 — 确认数据文件加载正确

### Q: 显存不足 (OOM)

1. 减小 `seq_len`（从 128 → 64 → 32）
2. 减小 `batch_size`（从 32 → 16 → 8）
3. 降级 AGV 等级（S/M → 更小）
4. 使用 `fp16` 混合精度（默认开启）

### Q: 数据格式不支持

使用 `convert_robot_data.py --dry_run` 查看支持格式，或提交 Issue。

### Q: KeyError: 'vision'

当使用 `--no_vision` 等参数禁用某些模态时，world_model.py 已修复此问题。
系统会自动从第一个可用的观测维度获取 device，并支持无 vision 的训练模式。

---

## 更新日志

### 2026-04-29

**Bug 修复：**
- `world_model.py`: 修复维度顺序处理，支持 (B,T,...) → (T,B,...) 自动转换
- `world_model.py`: 修复 vision 缺失时的 device 获取问题
- `train_real_data.py`: 根据 AGV 等级自动设置正确的 action_dim

**新增功能：**
- `scripts/convert_lerobot_data.py`: LeRobot 数据集转换器
- `scripts/dynamic_learner.py`: 动态增量学习系统（Quick/Idle/Deep三种模式）

**文档更新：**
- 添加 LeRobot 数据集转换说明
- 添加动态增量学习系统使用指南
- 添加已测试数据集对照表

---

## 参考配置

### M 级 AGV 推荐配置（双卡 RTX 3090 24GB×2）

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

### L 级 AGV 推荐配置

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
