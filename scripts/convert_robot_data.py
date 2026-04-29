#!/usr/bin/env python3
"""
convert_robot_data.py - 真实机器人数据转换为 HDF5 训练格式
=========================================================

将各种来源的机器人数据转换为 World Model 可读的 HDF5 格式。

支持的数据格式:
1. NumPy .npz 文件（由 SensorManager 录制）
2. ROS bag 包（通过 rosbag API）
3. 手动组织的目录（/vision/*.npy, /lidar/*.npy, ...）
4. CSV 格式的动作-奖励序列

输出 HDF5 结构:
    /observations
      /vision  (N, 512) float32
      /lidar   (N, 128) float32
      /tactile (N, 64) float32
      /force   (N, 6) float32
      /imu     (N, 6) float32
    /actions    (N, 7) float32
    /rewards    (N,) float32
    /dones      (N,) bool
    /timestamps (N,) float64

用法:
    # 从 npz 目录转换
    python scripts/convert_robot_data.py --input /data/robot_run1.npz --output ./data/run1.h5

    # 从目录批量转换
    python scripts/convert_robot_data.py --input_dir ./raw_data/ --output ./h5_data/

    # 指定 AGV 等级（决定维度）
    python scripts/convert_robot_data.py --input /data/run.npz --grade L --output ./run.h5
"""

import argparse
import os
import sys
import glob
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

logger = logging.getLogger("convert_robot_data")


# ========================
# 配置
# ========================

GRADE_OBS_DIMS = {
    'S':  {'vision': 512, 'lidar': 128, 'tactile': 64, 'force': 6, 'imu': 6, 'audio': 128},
    'M':  {'vision': 512, 'lidar': 128, 'tactile': 64, 'force': 6, 'imu': 6, 'audio': 128},
    'L':  {'vision': 768, 'lidar': 256, 'tactile': 128, 'force': 6, 'imu': 6, 'audio': 256},
    'XL': {'vision': 1024, 'lidar': 512, 'tactile': 256, 'force': 6, 'imu': 6, 'audio': 512},
    'XXL':{'vision': 1536, 'lidar': 512, 'tactile': 512, 'force': 6, 'imu': 6, 'audio': 512},
}

ACTION_DIMS = {
    'S':  6,   # Twist only
    'M':  7,   # Twist + gripper
    'L':  7,   # Twist + gripper
    'XL': 8,   # Twist + gripper + impedance
    'XXL': 12, # Full
}


# ========================
# 数据加载器基类
# ========================

class BaseDataLoader:
    """数据加载器抽象基类"""

    def __init__(self, grade: str = 'M'):
        self.grade = grade.upper()
        self.obs_dims = GRADE_OBS_DIMS.get(self.grade, GRADE_OBS_DIMS['M'])
        self.action_dim = ACTION_DIMS.get(self.grade, 7)

    def load(self) -> Tuple[
        Dict[str, np.ndarray],
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """返回 (observations, actions, rewards, dones, timestamps)"""
        raise NotImplementedError


class NumpyDirLoader(BaseDataLoader):
    """
    从目录加载 NumPy 录制数据

    预期目录结构:
        /data_root/
            vision.npy     (N, D_vision)
            lidar.npy     (N, D_lidar)
            tactile.npy    (N, D_tactile)
            force.npy     (N, D_force)
            imu.npy       (N, D_imu)
            audio.npy     (N, D_audio)
            actions.npy   (N, action_dim)
            rewards.npy   (N,)
            dones.npy     (N,)
            timestamps.npy (N,)
    """

    def __init__(self, data_root: str, grade: str = 'M'):
        super().__init__(grade)
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {data_root}")

    def load(self) -> Tuple[
        Dict[str, np.ndarray],
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        observations = {}

        for modality, dim in self.obs_dims.items():
            path = self.data_root / f"{modality}.npy"
            if path.exists():
                data = np.load(path)
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                # 验证维度
                if data.shape[1] != dim:
                    logger.warning(
                        f"{modality} dim mismatch: expected {dim}, got {data.shape[1]}. "
                        f"Padding/truncating."
                    )
                    if data.shape[1] < dim:
                        data = np.pad(data, ((0, 0), (0, dim - data.shape[1])))
                    else:
                        data = data[:, :dim]
                observations[modality] = data.astype(np.float32)
            else:
                logger.warning(f"{modality}.npy not found, using zeros")
                observations[modality] = np.zeros((self._get_length(), dim), np.float32)

        actions_path = self.data_root / "actions.npy"
        if actions_path.exists():
            actions = np.load(actions_path).astype(np.float32)
        else:
            actions = np.zeros((self._get_length(), self.action_dim), np.float32)

        rewards_path = self.data_root / "rewards.npy"
        rewards = np.load(rewards_path).astype(np.float32) if rewards_path.exists() else np.zeros(self._get_length(), np.float32)

        dones_path = self.data_root / "dones.npy"
        dones = np.load(dones_path) if dones_path.exists() else np.zeros(self._get_length(), dtype=bool)

        timestamps_path = self.data_root / "timestamps.npy"
        timestamps = np.load(timestamps_path) if timestamps_path.exists() else np.arange(self._get_length())

        return observations, actions, rewards, dones, timestamps

    def _get_length(self) -> int:
        """获取序列长度（从第一个存在的模态数据）"""
        for modality in self.obs_dims.keys():
            path = self.data_root / f"{modality}.npy"
            if path.exists():
                return np.load(path).shape[0]
        raise ValueError("No modality data found")


class NPZLoader(BaseDataLoader):
    """
    从 .npz 文件加载（SensorManager 录制格式）

    .npz 文件可能包含:
        vision: (N, D) or object array
        lidar: (N, D)
        tactile: (N, D)
        force: (N, 6)
        imu: (N, 6)
        audio: (N, D)
        actions: (N, action_dim)
        rewards: (N,)
        dones: (N,)
        timestamps: (N,)
    """

    def __init__(self, npz_path: str, grade: str = 'M'):
        super().__init__(grade)
        self.npz_path = npz_path
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    def load(self) -> Tuple[
        Dict[str, np.ndarray],
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        data = np.load(self.npz_path, allow_pickle=True)

        observations = {}
        keys = data.files

        for modality, dim in self.obs_dims.items():
            key = modality
            if key in keys:
                arr = data[key]
                if arr.dtype == object:
                    # 可能是编码后的特征，直接使用
                    if isinstance(arr.item(), np.ndarray):
                        arr = arr.item()
                    else:
                        arr = np.zeros((self._get_length(keys), dim), np.float32)
                arr = arr.astype(np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                if arr.shape[1] != dim:
                    if arr.shape[1] < dim:
                        arr = np.pad(arr, ((0, 0), (0, dim - arr.shape[1])))
                    else:
                        arr = arr[:, :dim]
                observations[modality] = arr
            else:
                observations[modality] = np.zeros((self._get_length(keys), dim), np.float32)

        # Actions
        if 'actions' in keys:
            actions = data['actions'].astype(np.float32)
        elif 'action' in keys:
            actions = data['action'].astype(np.float32)
        else:
            actions = np.zeros((self._get_length(keys), self.action_dim), np.float32)

        # Rewards
        rewards = data['rewards'].astype(np.float32) if 'rewards' in keys else np.zeros(self._get_length(keys), np.float32)

        # Dones
        dones = data['dones'] if 'dones' in keys else np.zeros(self._get_length(keys), dtype=bool)

        # Timestamps
        timestamps = data['timestamps'] if 'timestamps' in keys else np.arange(self._get_length(keys))

        return observations, actions, rewards, dones, timestamps

    def _get_length(self, keys: List[str] = None) -> int:
        if keys is None:
            data = np.load(self.npz_path, allow_pickle=True)
            keys = data.files
        for k in ['vision', 'actions', 'rewards', 'lidar']:
            if k in keys:
                arr = data[k]
                if arr.dtype == object:
                    return len(arr)
                return arr.shape[0]
        raise ValueError("Cannot determine data length")


class CSVTrajectoryLoader(BaseDataLoader):
    """
    从 CSV 文件加载动作-奖励轨迹

    适用于只有动作和奖励的数据（如 imitation learning 轨迹）。
    其他模态数据用零填充。
    """

    def __init__(self, csv_path: str, grade: str = 'M'):
        super().__init__(grade)
        self.csv_path = csv_path
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

    def load(self) -> Tuple[
        Dict[str, np.ndarray],
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        # 简单 CSV 加载（需要 pandas 或手动解析）
        try:
            import pandas as pd
            df = pd.read_csv(self.csv_path)
        except ImportError:
            logger.error("pandas required for CSV loading. Install with: pip install pandas")
            raise

        n = len(df)

        # 动作
        action_cols = [c for c in df.columns if 'action' in c.lower() or c in ['vx', 'vy', 'vz', 'rx', 'ry', 'rz', 'gripper']]
        if action_cols:
            actions = df[action_cols].values.astype(np.float32)
        else:
            actions = np.zeros((n, self.action_dim), np.float32)

        # 奖励
        reward_cols = [c for c in df.columns if 'reward' in c.lower() or c == 'r']
        if reward_cols:
            rewards = df[reward_cols].values.astype(np.float32).flatten()
        else:
            rewards = np.zeros(n, np.float32)

        # Done（episode 结束标志）
        done_cols = [c for c in df.columns if 'done' in c.lower() or c == 'd' or c == 'terminal']
        if done_cols:
            dones = df[done_cols].values.flatten().astype(bool)
        else:
            dones = np.zeros(n, dtype=bool)

        # 时间戳
        ts_cols = [c for c in df.columns if 'time' in c.lower() or c == 't' or c == 'timestamp']
        if ts_cols:
            timestamps = df[ts_cols].values.flatten()
        else:
            timestamps = np.arange(n)

        # 观测用零填充
        observations = {modality: np.zeros((n, dim), np.float32) for modality, dim in self.obs_dims.items()}

        return observations, actions, rewards, dones, timestamps


# ========================
# HDF5 写入器
# ========================

def save_hdf5(
    output_path: str,
    observations: Dict[str, np.ndarray],
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    timestamps: np.ndarray,
    compression: str = 'gzip',
    compression_level: int = 6,
):
    """保存为 HDF5 格式"""
    if not HAS_H5PY:
        raise ImportError("h5py required. Install with: pip install h5py")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # 观测
        obs_group = f.create_group('observations')
        for modality, data in observations.items():
            obs_group.create_dataset(
                modality,
                data=data,
                dtype=np.float32,
                compression=compression,
                compression_opts=compression_level,
            )

        # 动作
        f.create_dataset(
            'actions',
            data=actions,
            dtype=np.float32,
            compression=compression,
            compression_opts=compression_level,
        )

        # 奖励
        f.create_dataset(
            'rewards',
            data=rewards,
            dtype=np.float32,
            compression=compression,
            compression_opts=compression_level,
        )

        # Done
        f.create_dataset(
            'dones',
            data=dones,
            dtype=bool,
            compression=compression,
            compression_opts=compression_level,
        )

        # 时间戳
        f.create_dataset(
            'timestamps',
            data=timestamps,
            dtype=np.float64,
            compression=compression,
            compression_opts=compression_level,
        )

        # 元数据
        f.attrs['num_frames'] = len(actions)
        f.attrs['action_dim'] = actions.shape[1] if actions.ndim > 1 else 1
        f.attrs['modalities'] = list(observations.keys())

    logger.info(f"Saved: {output_path} ({len(actions)} frames)")


# ========================
# 主函数
# ========================

def detect_format_and_load(path: str, grade: str) -> Tuple[
    Dict[str, np.ndarray],
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """自动检测数据格式并加载"""
    path = Path(path)

    if path.is_file():
        if path.suffix.lower() == '.npz':
            loader = NPZLoader(str(path), grade=grade)
            return loader.load()
        elif path.suffix.lower() == '.csv':
            loader = CSVTrajectoryLoader(str(path), grade=grade)
            return loader.load()
        else:
            raise ValueError(f"Unknown file format: {path.suffix}")
    elif path.is_dir():
        # 检查是 NumPy 目录还是包含多个 npz
        npz_files = list(path.glob("*.npz"))
        if npz_files:
            # 多个 npz 文件，按时间顺序合并
            if len(npz_files) == 1:
                loader = NPZLoader(str(npz_files[0]), grade=grade)
            else:
                # 合并多个 npz
                loaders = [NPZLoader(str(f), grade=grade) for f in sorted(npz_files)]
                all_obs = [{k: [] for k in loader.obs_dims} for loader in loaders]
                all_actions, all_rewards, all_dones, all_ts = [], [], [], []

                for i, loader in enumerate(loaders):
                    obs, actions, rewards, dones, ts = loader.load()
                    for k in obs:
                        all_obs[i][k].append(obs[k])
                    all_actions.append(actions)
                    all_rewards.append(rewards)
                    all_dones.append(dones)
                    all_ts.append(ts)

                # 拼接
                concat_obs = {k: np.concatenate([a[k] for a in all_obs], axis=0) for k in all_obs[0].keys()}
                observations = concat_obs
                actions = np.concatenate(all_actions, axis=0)
                rewards = np.concatenate(all_rewards, axis=0)
                dones = np.concatenate(all_dones, axis=0)
                timestamps = np.concatenate(all_ts, axis=0)

                return observations, actions, rewards, dones, timestamps
        else:
            # NumPy 目录
            loader = NumpyDirLoader(str(path), grade=grade)
            return loader.load()
    else:
        raise FileNotFoundError(f"Path not found: {path}")


def convert_single(
    input_path: str,
    output_path: str,
    grade: str = 'M',
    compression: str = 'gzip',
):
    """转换单个文件/目录"""
    logger.info(f"Converting: {input_path} -> {output_path}")

    observations, actions, rewards, dones, timestamps = detect_format_and_load(input_path, grade)

    # 验证
    n = len(actions)
    for modality, data in observations.items():
        if len(data) != n:
            raise ValueError(f"Modality {modality} length {len(data)} != actions length {n}")

    logger.info(f"Data shape: {n} frames")
    logger.info(f"  observations: { {k: v.shape for k, v in observations.items()} }")
    logger.info(f"  actions: {actions.shape}")
    logger.info(f"  rewards: {rewards.shape}")
    logger.info(f"  dones: {dones.shape}")

    save_hdf5(output_path, observations, actions, rewards, dones, timestamps, compression)
    logger.info(f"Done: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert robot data to HDF5 training format")

    parser.add_argument("--input", type=str, default="",
                        help="Single file or directory to convert")
    parser.add_argument("--input_dir", type=str, default="",
                        help="Directory containing multiple files to convert")
    parser.add_argument("--output", type=str, default="",
                        help="Output HDF5 file (for single input)")
    parser.add_argument("--output_dir", type=str, default="./h5_data",
                        help="Output directory (for batch conversion)")
    parser.add_argument("--grade", type=str, default="M",
                        choices=["S", "M", "L", "XL", "XXL"],
                        help="AGV grade (determines observation dimensions)")
    parser.add_argument("--compression", type=str, default="gzip",
                        choices=["gzip", "lzf", "none"],
                        help="HDF5 compression")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be done without converting")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    if not HAS_H5PY:
        logger.error("h5py required. Install with: pip install h5py")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.input:
        # 单文件转换
        if not args.output:
            # 自动生成输出名
            input_name = Path(args.input).stem
            args.output = os.path.join(args.output_dir, f"{input_name}.h5")

        if args.dry_run:
            logger.info(f"Would convert: {args.input} -> {args.output}")
        else:
            convert_single(args.input, args.output, args.grade, args.compression)

    elif args.input_dir:
        # 批量转换
        input_dir = Path(args.input_dir)
        if input_dir.is_file():
            files = [input_dir]
        else:
            files = sorted(input_dir.glob("*"))

        for f in files:
            if f.is_file() and f.suffix.lower() in ['.npz', '.csv']:
                output_name = f.stem + ".h5"
                output_path = os.path.join(args.output_dir, output_name)

                if args.dry_run:
                    logger.info(f"Would convert: {f} -> {output_path}")
                else:
                    try:
                        convert_single(str(f), output_path, args.grade, args.compression)
                    except Exception as e:
                        logger.error(f"Failed to convert {f}: {e}")
            elif f.is_dir():
                output_name = f.name + ".h5"
                output_path = os.path.join(args.output_dir, output_name)

                if args.dry_run:
                    logger.info(f"Would convert dir: {f} -> {output_path}")
                else:
                    try:
                        convert_single(str(f), output_path, args.grade, args.compression)
                    except Exception as e:
                        logger.error(f"Failed to convert dir {f}: {e}")

    else:
        parser.print_help()
        logger.info("\n=== Quick Usage ===")
        logger.info("# Convert a single npz file:")
        logger.info(f"  python {sys.argv[0]} --input /data/recording.npz --grade M --output ./data.h5")
        logger.info("")
        logger.info("# Convert a directory of NumPy files:")
        logger.info(f"  python {sys.argv[0]} --input /data/session_001/ --grade M --output ./session_001.h5")
        logger.info("")
        logger.info("# Batch convert a directory:")
        logger.info(f"  python {sys.argv[0]} --input_dir ./raw_data/ --output_dir ./h5_data/ --grade M")


if __name__ == "__main__":
    main()
