#!/usr/bin/env python3
"""
train_real_data.py - SuperModel 真实数据训练流程
================================================

支持从多种数据源训练 World Model:
1. SensorManager 实时采集数据
2. PyBullet/MuJoCo 仿真回放数据
3. 预录制的数据集 (HDF5/Parquet 格式)
4. ROS2 bag 包回放

关键组件:
- RealDataLoader: 多源数据加载器
- DataAugmenter: 多模态数据增强
- OnlineTrainer: 在线增量训练
- BatchTrainer: 批量离线训练

环境变量:
    CUDA_VISIBLE_DEVICES=0,1  # 指定 GPU
    SUPERMODEL_DATA_ROOT=/path/to/data  # 数据根目录

用法:
    # 单卡批量训练
    python scripts/train_real_data.py --mode batch --data_root /data/agv_logs

    # 在线增量训练 (实时传感器)
    python scripts/train_real_data.py --mode online --grade M

    # 多卡分布式
    deepspeed --num_gpus=2 scripts/train_real_data.py --mode batch --data_root /data
"""

import argparse
import os
import sys
import time
import json
import glob
import threading
import queue
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable, Iterator
from dataclasses import dataclass, field, asdict
from collections import deque
import pickle
import struct

import numpy as np

# ========================
# 核心依赖
# ========================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ========================
# 项目路径
# ========================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.sensors.manager import SensorManager, SensorDataFrame
from src.sensors.vision import BinocularCamera, StereoFrame
from src.sensors.imu import IMUFrame
from src.sensors.force import Wrench
from src.sensors.tactile import TactileFrame
from src.sensors.audio import AudioFrame
from src.learning.world_model import (
    WorldModel, WorldModelConfig, WORLD_MODEL_GRADES,
    ReplayBuffer, create_world_model_agent
)
from src.embodied.vla_model import VLAAction


# ========================
# 日志配置
# ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("train_real_data")


# ========================
# 数据源类型
# ========================
class DataSourceType:
    """支持的数据源类型"""
    SENSOR_MANAGER = "sensor_manager"      # 实时传感器
    PYBULLET_SIM = "pybullet_sim"          # PyBullet 仿真
    MUJOCO_SIM = "mujoco_sim"              # MuJoCo 仿真
    ROS2_BAG = "ros2_bag"                 # ROS2 bag 回放
    HDF5_DATASET = "hdf5_dataset"          # HDF5 预录数据集
    PARQUET_DATASET = "parquet_dataset"    # Parquet 数据集
    DUMMY = "dummy"                        # 测试用 dummy 源


# ========================
# 配置
# ========================

@dataclass
class DataLoaderConfig:
    """数据加载器配置"""
    # 数据源
    source_type: str = DataSourceType.DUMMY

    # 数据根目录
    data_root: str = "./data"

    # AGV 等级
    grade: str = "M"

    # 模态开关
    use_vision: bool = True
    use_lidar: bool = True
    use_tactile: bool = True
    use_force: bool = True
    use_imu: bool = True

    # 数据维度 (由 grade 决定)
    vision_dim: int = 512
    lidar_dim: int = 128
    audio_dim: int = 128
    tactile_dim: int = 64
    force_dim: int = 6
    imu_dim: int = 6

    # 动作维度
    action_dim: int = 7  # COMBINED: vx, vy, vz, rx, ry, rz, gripper

    # 序列长度
    seq_len: int = 128

    # 批处理
    batch_size: int = 32
    num_workers: int = 4

    # 在线训练
    buffer_capacity: int = 100000
    sample_interval: int = 3  # 每隔N帧取一帧

    # 数据增强
    augmentation_enabled: bool = True
    noise_level: float = 0.01
    dropout_modality: float = 0.05  # 模态随机丢弃概率

    # 仿真相关
    sim_config_path: str = ""

    # ROS2 bag 路径
    ros2_bag_path: str = ""

    def get_obs_dims(self) -> Dict[str, int]:
        dims = {}
        if self.use_vision:
            dims['vision'] = self.vision_dim
        if self.use_lidar:
            dims['lidar'] = self.lidar_dim
        if self.use_tactile:
            dims['tactile'] = self.tactile_dim
        if self.use_force:
            dims['force'] = self.force_dim
        if self.use_imu:
            dims['imu'] = self.imu_dim
        return dims

    def get_modality_list(self) -> List[str]:
        mods = []
        if self.use_vision:
            mods.append('vision')
        if self.use_lidar:
            mods.append('lidar')
        if self.use_tactile:
            mods.append('tactile')
        if self.use_force:
            mods.append('force')
        if self.use_imu:
            mods.append('imu')
        return mods


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础
    grade: str = "M"
    device: str = "cuda"
    output_dir: str = "./checkpoints/real_training"
    experiment_name: str = ""

    # 优化器
    lr: float = 3e-5
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    warmup_steps: int = 500

    # 训练策略
    accumulation_steps: int = 4  # 梯度累积
    eval_interval: int = 1000
    save_interval: int = 5000
    max_steps: int = 100000

    # 在线训练
    online_training: bool = False
    update_interval: int = 64  # 隔多少帧更新一次

    # 分布式
    distributed: bool = False
    local_rank: int = 0

    # DeepSpeed
    use_deepspeed: bool = False
    deepspeed_config: str = ""

    # 混合精度
    fp16: bool = True

    # 继续训练
    resume_path: str = ""


# ========================
# 数据加载器
# ========================

class SensorFrame:
    """
    标准化传感器帧

    World Model 使用的标准观测格式。
    与 SensorDataFrame 一一对应。
    """

    def __init__(
        self,
        timestamp: float,
        vision: Optional[np.ndarray] = None,      # (512,) float32
        lidar: Optional[np.ndarray] = None,       # (128,) float32
        audio: Optional[np.ndarray] = None,       # (128,) float32
        tactile: Optional[np.ndarray] = None,    # (64,) float32
        force: Optional[np.ndarray] = None,       # (6,) float32
        imu: Optional[np.ndarray] = None,         # (6,) float32
        action: Optional[np.ndarray] = None,      # (action_dim,) float32
        reward: float = 0.0,
        done: bool = False,
        info: Optional[Dict] = None,
    ):
        self.timestamp = timestamp
        self.vision = vision
        self.lidar = lidar
        self.audio = audio
        self.tactile = tactile
        self.force = force
        self.imu = imu
        self.action = action
        self.reward = reward
        self.done = done
        self.info = info or {}

    def get_obs(self, obs_dims: Dict[str, int]) -> Dict[str, np.ndarray]:
        """转换为 World Model 观测格式"""
        obs = {}
        if 'vision' in obs_dims and self.vision is not None:
            obs['vision'] = self.vision.astype(np.float32)
        if 'lidar' in obs_dims and self.lidar is not None:
            obs['lidar'] = self.lidar.astype(np.float32)
        if 'audio' in obs_dims and self.audio is not None:
            obs['audio'] = self.audio.astype(np.float32)
        if 'tactile' in obs_dims and self.tactile is not None:
            obs['tactile'] = self.tactile.astype(np.float32)
        if 'force' in obs_dims and self.force is not None:
            obs['force'] = self.force.astype(np.float32)
        if 'imu' in obs_dims and self.imu is not None:
            obs['imu'] = self.imu.astype(np.float32)
        return obs

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'vision': self.vision,
            'lidar': self.lidar,
            'audio': self.audio,
            'tactile': self.tactile,
            'force': self.force,
            'imu': self.imu,
            'action': self.action,
            'reward': self.reward,
            'done': self.done,
        }

    @classmethod
    def from_sensor_data_frame(cls, frame: SensorDataFrame, action: np.ndarray, reward: float, done: bool) -> 'SensorFrame':
        """从 SensorDataFrame 转换"""
        return cls(
            timestamp=frame.timestamp,
            vision=frame.vision_encoded,
            lidar=None,  # 需要 LIDAR 传感器管理器
            audio=frame.audio_encoded,
            tactile=frame.tactile_encoded,
            force=frame.force_encoded,
            imu=frame.imu_encoded,
            action=action,
            reward=reward,
            done=done,
        )


class HDF5Dataset(Dataset):
    """
    HDF5 预录数据集加载器

    数据格式:
    /observations
      /vision  (N, 512) float32
      /lidar   (N, 128) float32
      /tactile (N, 64) float32
      /force   (N, 6) float32
      /imu     (N, 6) float32
    /actions    (N, action_dim) float32
    /rewards    (N,) float32
    /dones      (N,) bool
    /timestamps (N,) float64
    """

    def __init__(
        self,
        data_paths: List[str],
        obs_dims: Dict[str, int],
        action_dim: int = 7,
        seq_len: int = 128,
        augmentation: bool = True,
        noise_level: float = 0.01,
        modality_dropout: float = 0.05,
        normalize: bool = True,
        reward_scale: float = 1.0,
    ):
        import h5py

        self.obs_dims = obs_dims
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.augmentation = augmentation
        self.noise_level = noise_level
        self.modality_dropout = modality_dropout
        self.normalize = normalize
        self.reward_scale = reward_scale

        # 加载所有文件
        self.observations = {k: [] for k in obs_dims.keys()}
        self.actions = []
        self.rewards = []
        self.dones = []
        self.lengths = []

        for path in data_paths:
            logger.info(f"Loading: {path}")
            with h5py.File(path, 'r') as f:
                for modality in obs_dims.keys():
                    if modality in f['observations']:
                        self.observations[modality].append(
                            f[f'observations/{modality}'][:]
                        )
                    else:
                        # 填充零
                        n = f['actions'][:].shape[0]
                        self.observations[modality].append(
                            np.zeros((n, obs_dims[modality]), dtype=np.float32)
                        )

                self.actions.append(f['actions'][:])
                self.rewards.append(f['rewards'][:])
                self.dones.append(f['dones'][:])
                self.lengths.append(f['actions'][:].shape[0])

        # 拼接
        self.observations = {k: np.concatenate(v, axis=0) for k, v in self.observations.items()}
        self.actions = np.concatenate(self.actions, axis=0)
        self.rewards = np.concatenate(self.rewards, axis=0).astype(np.float32)
        self.dones = np.concatenate(self.dones, axis=0)
        self.total_length = sum(self.lengths)

        logger.info(f"Loaded {self.total_length} frames from {len(data_paths)} files")
        logger.info(f"Observation shapes: { {k: v.shape for k, v in self.observations.items()} }")

        # 统计信息
        if normalize:
            self.obs_mean = {k: v.mean(axis=0) for k, v in self.observations.items()}
            self.obs_std = {k: v.std(axis=0) + 1e-6 for k, v in self.observations.items()}
        else:
            self.obs_mean = {k: np.zeros(obs_dims[k]) for k in obs_dims.keys()}
            self.obs_std = {k: np.ones(obs_dims[k]) for k in obs_dims.keys()}

        # 动作统计
        self.action_mean = self.actions.mean(axis=0)
        self.action_std = self.actions.std(axis=0) + 1e-6

    def __len__(self):
        return max(0, self.total_length - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_batch = {}
        for modality in self.obs_dims.keys():
            obs = self.observations[modality][idx:idx + self.seq_len].copy()

            # 数据增强
            if self.augmentation:
                # 高斯噪声
                noise = np.random.randn(*obs.shape).astype(np.float32) * self.noise_level
                obs = obs + noise

                # 模态随机丢弃
                if np.random.rand() < self.modality_dropout:
                    obs = np.zeros_like(obs)

            # 归一化
            obs = (obs - self.obs_mean[modality]) / self.obs_std[modality]

            obs_batch[modality] = torch.from_numpy(obs)

        actions = self.actions[idx:idx + self.seq_len].copy()
        actions = (actions - self.action_mean) / self.action_std
        actions = torch.from_numpy(actions).float()

        rewards = torch.from_numpy(
            np.array(self.rewards[idx:idx + self.seq_len] * self.reward_scale, dtype=np.float32)
        )
        dones = torch.from_numpy(
            np.array(self.dones[idx:idx + self.seq_len], dtype=np.float32)
        )

        return obs_batch, actions, rewards, dones


class OnlineDataBuffer:
    """
    在线数据缓冲区

    实时接收传感器数据，用于在线增量训练。
    """

    def __init__(
        self,
        obs_dims: Dict[str, int],
        action_dim: int = 7,
        capacity: int = 100000,
        sample_interval: int = 3,
    ):
        self.capacity = capacity
        self.obs_dims = obs_dims
        self.action_dim = action_dim
        self.sample_interval = sample_interval

        self.buffer = deque(maxlen=capacity)
        self.frame_count = 0
        self.lock = threading.Lock()

    def push(self, frame: SensorFrame):
        """线程安全地添加一帧"""
        with self.lock:
            if self.frame_count % self.sample_interval == 0:
                self.buffer.append(frame)
            self.frame_count += 1

    def push_batch(self, frames: List[SensorFrame]):
        """批量添加"""
        for frame in frames:
            self.push(frame)

    def sample(self, batch_size: int, seq_len: int) -> Optional[Tuple]:
        """采样一个 batch"""
        with self.lock:
            if len(self.buffer) < seq_len + 1:
                return None

            # 随机起点
            start_idx = np.random.randint(0, len(self.buffer) - seq_len)
            seq_frames = [self.buffer[start_idx + i] for i in range(seq_len)]

        obs_batch = {k: [] for k in self.obs_dims.keys()}
        actions = []
        rewards = []
        dones = []

        for frame in seq_frames:
            obs = frame.get_obs(self.obs_dims)
            for k in self.obs_dims.keys():
                obs_batch[k].append(obs.get(k, np.zeros(self.obs_dims[k], dtype=np.float32)))

            if frame.action is not None:
                actions.append(frame.action)
            else:
                actions.append(np.zeros(self.action_dim, dtype=np.float32))

            rewards.append(frame.reward)
            dones.append(1.0 if frame.done else 0.0)

        obs_batch = {k: np.stack(v) for k, v in obs_batch.items()}
        actions = np.stack(actions)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        obs_batch = {k: torch.from_numpy(v) for k, v in obs_batch.items()}
        actions = torch.from_numpy(actions).float()
        rewards = torch.from_numpy(rewards).float()
        dones = torch.from_numpy(dones).float()

        return obs_batch, actions, rewards, dones

    def __len__(self):
        with self.lock:
            return len(self.buffer)


# ========================
# 数据源基类
# ========================

class BaseDataSource:
    """数据源抽象基类"""

    def __init__(self, config: DataLoaderConfig):
        self.config = config
        self.running = False

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def iterator(self) -> Iterator[SensorFrame]:
        raise NotImplementedError

    def __iter__(self):
        return self.iterator()


class DummyDataSource(BaseDataSource):
    """测试用 Dummy 数据源"""

    def __init__(self, config: DataLoaderConfig):
        super().__init__(config)
        self.t = 0.0
        self.obs_dims = config.get_obs_dims()
        self.action_dim = config.action_dim

    def iterator(self) -> Iterator[SensorFrame]:
        np.random.seed(42)
        while self.running:
            self.t += 0.1

            # 模拟 AGV 运动
            vx = 0.5 * np.sin(self.t * 0.5)
            vz = 0.1 * np.sin(self.t * 0.3)
            rx = 0.0
            ry = 0.0
            rz = 0.2 * np.sin(self.t * 0.4)
            gripper = 0.5 + 0.3 * np.sin(self.t * 0.2)
            action = np.array([vx, 0, vz, rx, ry, rz, gripper], dtype=np.float32)

            # 模拟奖励
            reward = np.sin(self.t * 0.1).item()
            done = False

            frame = SensorFrame(
                timestamp=self.t,
                vision=np.random.randn(self.obs_dims.get('vision', 512)).astype(np.float32) * 0.3,
                lidar=np.random.rand(self.obs_dims.get('lidar', 128)).astype(np.float32) * 10.0,
                audio=np.random.randn(self.obs_dims.get('audio', 128)).astype(np.float32) * 0.1,
                tactile=np.random.randn(self.obs_dims.get('tactile', 64)).astype(np.float32) * 0.1,
                force=np.random.randn(6).astype(np.float32) * 0.05,
                imu=np.array([
                    vx * 0.5 + np.random.randn() * 0.01,
                    rz * 0.3 + np.random.randn() * 0.01,
                    9.81 + np.random.randn() * 0.01,
                    np.random.randn() * 0.01,
                    np.random.randn() * 0.01,
                    rz + np.random.randn() * 0.01,
                ], dtype=np.float32),
                action=action,
                reward=reward,
                done=done,
            )

            yield frame
            time.sleep(0.01)  # ~100Hz


class HDF5DataSource(BaseDataSource):
    """HDF5 预录数据集源"""

    def __init__(self, config: DataLoaderConfig):
        super().__init__(config)
        self.config: DataLoaderConfig
        self.obs_dims = config.get_obs_dims()
        self.action_dim = config.action_dim
        self.seq_len = config.seq_len

        # 查找数据文件
        data_root = Path(config.data_root)
        if data_root.is_file():
            self.data_files = [str(data_root)]
        else:
            self.data_files = sorted(glob.glob(str(data_root / "**/*.h5"), recursive=True))
            if not self.data_files:
                self.data_files = sorted(glob.glob(str(data_root / "**/*.hdf5"), recursive=True))
            if not self.data_files:
                self.data_files = sorted(glob.glob(str(data_root / "**/*.h5py"), recursive=True))

        if not self.data_files:
            raise FileNotFoundError(f"No HDF5 files found in {data_root}")

        logger.info(f"Found {len(self.data_files)} HDF5 files")

        # 创建数据集
        self.dataset = HDF5Dataset(
            data_paths=self.data_files,
            obs_dims=self.obs_dims,
            action_dim=self.action_dim,
            seq_len=self.seq_len,
            augmentation=config.augmentation_enabled,
            noise_level=config.noise_level,
            modality_dropout=config.dropout_modality,
            normalize=True,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.iterator = iter(self.dataloader)

    def iterator(self) -> Iterator:
        while self.running:
            try:
                yield next(self.iterator)
            except StopIteration:
                # 重置迭代器
                self.iterator = iter(self.dataloader)


# ========================
# 数据采集器 (SensorManager 封装)
# ========================

class SensorDataCollector:
    """
    传感器数据采集器

    封装 SensorManager，提供实时数据流。
    支持在线训练模式。
    """

    def __init__(
        self,
        config: DataLoaderConfig,
        buffer: Optional[OnlineDataBuffer] = None,
    ):
        self.config = config
        self.buffer = buffer
        self.running = False
        self.sensor_manager: Optional[SensorManager] = None
        self.collection_thread: Optional[threading.Thread] = None

    def start(self):
        """启动采集"""
        if self.running:
            return

        self.running = True

        # 初始化传感器
        try:
            self.sensor_manager = SensorManager(grade=self.config.grade)
            self.sensor_manager.start()
            logger.info("SensorManager started")
        except Exception as e:
            logger.warning(f"SensorManager init failed: {e}, using dummy source")
            self.data_source = DummyDataSource(self.config)
            self.data_source.start()
            self.collection_thread = threading.Thread(target=self._collect_dummy)
        else:
            self.collection_thread = threading.Thread(target=self._collect_real)

        self.collection_thread.start()

    def stop(self):
        """停止采集"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join()
        if self.sensor_manager:
            self.sensor_manager.stop()

    def _collect_real(self):
        """从真实传感器采集"""
        last_action = np.zeros(self.config.action_dim, dtype=np.float32)
        last_reward = 0.0
        frame_count = 0

        while self.running:
            try:
                # 获取传感器数据
                frame = self.sensor_manager.get_frame(timeout=0.1)
                if frame is None:
                    continue

                # 转换
                sensor_frame = SensorFrame.from_sensor_data_frame(
                    frame, last_action, last_reward, False
                )

                # 存入 buffer
                if self.buffer:
                    self.buffer.push(sensor_frame)

                last_action = sensor_frame.action or last_action
                last_reward = sensor_frame.reward
                frame_count += 1

            except Exception as e:
                logger.error(f"Collection error: {e}")
                continue

    def _collect_dummy(self):
        """Dummy 数据采集"""
        for frame in self.data_source:
            if not self.running:
                break
            if self.buffer:
                self.buffer.push(frame)


# ========================
# 训练器
# ========================

class WorldModelTrainer:
    """World Model 训练器"""

    def __init__(
        self,
        data_config: DataLoaderConfig,
        train_config: TrainingConfig,
    ):
        self.data_config = data_config
        self.train_config = train_config

        self.grade = train_config.grade.upper()
        self.device = torch.device(train_config.device)
        self.output_dir = Path(train_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建模型
        self.model_config = WORLD_MODEL_GRADES.get(self.grade, WORLD_MODEL_GRADES['M'])
        self.model_config.action_dim = data_config.action_dim

        obs_dims = data_config.get_obs_dims()
        self.model = WorldModel(
            obs_dims=obs_dims,
            action_dim=data_config.action_dim,
            config=self.model_config,
        )
        self.model.to(self.device)

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )

        # 学习率调度
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=train_config.max_steps, eta_min=1e-6
        )

        # 梯度缩放器 (FP16)
        self.scaler = torch.amp.GradScaler('cuda') if train_config.fp16 else None

        # 在线 buffer
        if train_config.online_training:
            self.buffer = OnlineDataBuffer(
                obs_dims=obs_dims,
                action_dim=data_config.action_dim,
                capacity=data_config.buffer_capacity,
                sample_interval=data_config.sample_interval,
            )
        else:
            self.buffer = None

        # 步骤计数
        self.global_step = 0
        self.accumulation_counter = 0

        # 保存配置
        self._save_configs()

        logger.info(f"Trainer initialized: grade={self.grade}, params={sum(p.numel() for p in self.model.parameters())/1e6:.1f}M")

    def _save_configs(self):
        """保存配置到输出目录"""
        config = {
            'data_config': asdict(self.data_config),
            'train_config': asdict(self.train_config),
            'model_config': {
                'grade': self.grade,
                'latent_dim': self.model_config.latent_dim,
                'hidden_dim': self.model_config.hidden_dim,
                'deter_dim': self.model_config.deter_dim,
                'stoch_dim': self.model_config.stoch_dim,
                'num_classes': self.model_config.num_classes,
            }
        }
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)

    def train_step(self, batch: Tuple, accumulation: bool = True) -> Dict[str, float]:
        """单步训练"""
        obs_batch, actions, rewards, dones = batch

        # 移到 GPU
        obs_batch = {k: v.to(self.device) for k, v in obs_batch.items()}
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # FP16 前向
        with torch.amp.autocast('cuda', enabled=self.train_config.fp16):
            losses = self.model.compute_loss(
                observations=obs_batch,
                actions=actions.unsqueeze(0),  # (1, T, B, action_dim)
                rewards=rewards,
                dones=dones,
            )
            loss = losses['total'] / (self.train_config.accumulation_steps if accumulation else 1)

        # 反向
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if not accumulation:
            # 梯度裁剪
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip)

            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.scheduler.step()
            self.global_step += 1

        return {k: v.item() for k, v in losses.items()}

    def train_batch(
        self,
        dataloader: DataLoader,
        num_steps: Optional[int] = None,
    ):
        """批量离线训练"""
        self.model.train()

        t_start = time.time()
        step = 0
        loss_accum = {}

        for batch in dataloader:
            # 训练
            is_accum_step = (step + 1) % self.train_config.accumulation_steps != 0
            losses = self.train_step(batch, accumulation=is_accum_step)

            # 统计
            for k, v in losses.items():
                loss_accum[k] = loss_accum.get(k, 0) + v

            step += 1

            # 日志
            if step % self.train_config.eval_interval == 0:
                elapsed = time.time() - t_start
                avg_loss = {k: v / self.train_config.eval_interval for k, v in loss_accum.items()}
                lr = self.scheduler.get_last_lr()[0]

                logger.info(
                    f"[Step {self.global_step}] "
                    f"Loss: {avg_loss.get('total', 0):.4f} | "
                    f"KL: {avg_loss.get('kl', 0):.4f} | "
                    f"Reward: {avg_loss.get('reward', 0):.4f} | "
                    f"Decoder: {avg_loss.get('decoder', 0):.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Speed: {step / elapsed:.1f} it/s"
                )
                loss_accum = {}

            # 保存
            if step % self.train_config.save_interval == 0:
                self.save_checkpoint(f"step_{self.global_step}.pt")
                logger.info(f"Checkpoint saved: step_{self.global_step}.pt")

            if num_steps and step >= num_steps:
                break

        self.save_checkpoint("final.pt")
        logger.info(f"Training complete! Final model saved.")

    def train_online(
        self,
        data_source: BaseDataSource,
        num_steps: Optional[int] = None,
    ):
        """在线增量训练"""
        self.model.train()
        t_start = time.time()
        step = 0
        loss_accum = {}

        # 启动数据源
        data_source.start()

        while True:
            # 从 buffer 采样
            batch = self.buffer.sample(
                batch_size=self.data_config.batch_size,
                seq_len=self.data_config.seq_len,
            )

            if batch is None:
                time.sleep(0.1)
                continue

            # 训练
            is_accum_step = (step + 1) % self.train_config.accumulation_steps != 0
            losses = self.train_step(batch, accumulation=is_accum_step)

            for k, v in losses.items():
                loss_accum[k] = loss_accum.get(k, 0) + v

            step += 1

            # 日志
            if step % self.train_config.eval_interval == 0:
                elapsed = time.time() - t_start
                avg_loss = {k: v / self.train_config.eval_interval for k, v in loss_accum.items()}
                lr = self.scheduler.get_last_lr()[0]
                buffer_size = len(self.buffer)

                logger.info(
                    f"[Step {self.global_step}] "
                    f"Loss: {avg_loss.get('total', 0):.4f} | "
                    f"Buffer: {buffer_size} | "
                    f"LR: {lr:.2e} | "
                    f"Elapsed: {elapsed:.1f}s"
                )
                loss_accum = {}

            # 保存
            if step % self.train_config.save_interval == 0:
                self.save_checkpoint(f"step_{self.global_step}.pt")

            if num_steps and step >= num_steps:
                break

        data_source.stop()
        self.save_checkpoint("final.pt")

    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'grade': self.grade,
        }
        torch.save(checkpoint, self.output_dir / filename)

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.global_step = checkpoint['global_step']
        logger.info(f"Loaded checkpoint: {path} (step {self.global_step})")


# ========================
# 主函数
# ========================

def create_data_source(config: DataLoaderConfig) -> BaseDataSource:
    """根据配置创建数据源"""
    if config.source_type == DataSourceType.DUMMY:
        return DummyDataSource(config)
    elif config.source_type == DataSourceType.HDF5_DATASET:
        return HDF5DataSource(config)
    else:
        raise ValueError(f"Unknown source type: {config.source_type}")


def main():
    parser = argparse.ArgumentParser(description="SuperModel Real Data Training")

    # 数据配置
    parser.add_argument("--data_root", type=str, default="./data", help="Data root directory")
    parser.add_argument("--source_type", type=str, default="dummy",
                        choices=["dummy", "hdf5_dataset", "sensor_manager", "pybullet_sim", "mujoco_sim"])
    parser.add_argument("--grade", type=str, default="M", choices=["S", "M", "L", "XL", "XXL"])

    # 模态开关
    parser.add_argument("--no_vision", action="store_true")
    parser.add_argument("--no_lidar", action="store_true")
    parser.add_argument("--no_tactile", action="store_true")
    parser.add_argument("--no_force", action="store_true")
    parser.add_argument("--no_imu", action="store_true")

    # 训练配置
    parser.add_argument("--mode", type=str, default="batch", choices=["batch", "online"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/real_training")
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--resume", type=str, default="")

    # 高级
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--experiment_name", type=str, default="")

    args = parser.parse_args()

    # 根据AGV等级设置action_dim
    ACTION_DIMS_BY_GRADE = {'S': 6, 'M': 7, 'L': 7, 'XL': 8, 'XXL': 12}
    action_dim = ACTION_DIMS_BY_GRADE.get(args.grade.upper(), 7)
    
    # 数据配置
    data_config = DataLoaderConfig(
        source_type=args.source_type,
        data_root=args.data_root,
        grade=args.grade,
        action_dim=action_dim,
        use_vision=not args.no_vision,
        use_lidar=not args.no_lidar,
        use_tactile=not args.no_tactile,
        use_force=not args.no_force,
        use_imu=not args.no_imu,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
    )

    # 训练配置
    experiment_name = args.experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"supermodel_{args.grade}_{experiment_name}")

    train_config = TrainingConfig(
        grade=args.grade,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_dir=output_dir,
        lr=args.lr,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        fp16=args.fp16,
        grad_clip=args.grad_clip,
        accumulation_steps=args.accumulation_steps,
        resume_path=args.resume,
    )

    # 创建训练器
    trainer = WorldModelTrainer(data_config, train_config)

    # 继续训练
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 根据模式选择训练方式
    if args.mode == "online":
        # 在线增量训练
        data_source = DummyDataSource(data_config)  # 或 SensorDataCollector
        buffer = OnlineDataBuffer(
            obs_dims=data_config.get_obs_dims(),
            action_dim=data_config.action_dim,
            capacity=data_config.buffer_capacity,
            sample_interval=data_config.sample_interval,
        )
        trainer.buffer = buffer
        trainer.train_online(data_source, num_steps=args.max_steps)
    else:
        # 批量离线训练
        try:
            data_source = create_data_source(data_config)
        except FileNotFoundError as e:
            logger.warning(f"Data source not found: {e}")
            logger.info("Falling back to dummy data source for testing")
            data_source = DummyDataSource(data_config)

        if hasattr(data_source, 'dataloader'):
            trainer.train_batch(data_source.dataloader, num_steps=args.max_steps)
        else:
            # Dummy source without dataloader - use online mode
            buffer = OnlineDataBuffer(
                obs_dims=data_config.get_obs_dims(),
                action_dim=data_config.action_dim,
                capacity=data_config.buffer_capacity,
                sample_interval=data_config.sample_interval,
            )
            data_source.start()
            trainer.buffer = buffer
            trainer.train_online(data_source, num_steps=args.max_steps)
            data_source.stop()

    logger.info(f"Training complete! Output: {output_dir}")


if __name__ == "__main__":
    main()