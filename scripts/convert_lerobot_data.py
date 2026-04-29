#!/usr/bin/env python3
"""
convert_lerobot_data.py - 将LeRobot数据集转换为SuperModel HDF5格式
================================================================

将ModelScope上下载的lerobot数据集（如berkeley_fanuc_manipulation）
转换为SuperModel可读的HDF5格式。

用法:
    python scripts/convert_lerobot_data.py \
        --input /root/.cache/modelscope/hub/datasets/downloads/42ecbaebcec... \
        --dataset berkeley_fanuc_manipulation \
        --grade M \
        --output ./lerobot_fanuc.h5

输出HDF5结构:
    /observations
      /vision  (N, 512) float32  <- 用零向量占位（原始数据无视觉）
      /lidar   (N, 128) float32  <- 用零向量占位
      /tactile (N, 64) float32   <- 用零向量占位
      /force   (N, 6) float32    <- 用零向量占位
      /imu     (N, 6) float32     <- 用零向量占位
      /state   (N, 8) float32    <- 来自observation.state
    /actions    (N, 7) float32
    /rewards    (N,) float32
    /dones      (N,) bool
    /timestamps (N,) float64
"""

import argparse
import os
import sys
import glob
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("警告: h5py未安装，将无法写入HDF5文件")
    print("安装: pip install h5py")

logger = logging.getLogger("convert_lerobot")


# ========================
# AGV等级配置
# ========================

GRADE_OBS_DIMS = {
    'S':  {'vision': 512, 'lidar': 128, 'tactile': 64, 'force': 6, 'imu': 6, 'audio': 128, 'state': 8},
    'M':  {'vision': 512, 'lidar': 128, 'tactile': 64, 'force': 6, 'imu': 6, 'audio': 128, 'state': 8},
    'L':  {'vision': 768, 'lidar': 256, 'tactile': 128, 'force': 6, 'imu': 6, 'audio': 256, 'state': 12},
    'XL': {'vision': 1024, 'lidar': 512, 'tactile': 256, 'force': 6, 'imu': 6, 'audio': 512, 'state': 14},
    'XXL':{'vision': 1536, 'lidar': 512, 'tactile': 512, 'force': 6, 'imu': 6, 'audio': 512, 'state': 16},
}

ACTION_DIMS = {
    'S':  6,
    'M':  7,
    'L':  7,
    'XL': 8,
    'XXL': 12,
}


# ========================
# 数据转换
# ========================

def convert_lerobot_to_hdf5(
    parquet_path: str,
    output_path: str,
    grade: str = 'M',
   填补视觉占位符: bool = True,
) -> bool:
    """
    将lerobot parquet数据转换为SuperModel HDF5格式

    Args:
        parquet_path: lerobot parquet文件路径
        output_path: 输出HDF5文件路径
        grade: AGV等级 (S/M/L/XL/XXL)
        填补视觉占位符: 是否用零向量填充缺失的视觉/传感器数据

    Returns:
        bool: 转换是否成功
    """
    if not HAS_H5PY:
        logger.error("h5py未安装，无法写入HDF5文件")
        return False

    grade = grade.upper()
    obs_dims = GRADE_OBS_DIMS.get(grade, GRADE_OBS_DIMS['M'])
    action_dim = ACTION_DIMS.get(grade, 7)

    logger.info(f"读取数据: {parquet_path}")
    try:
        import pandas as pd
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        logger.error(f"读取parquet文件失败: {e}")
        return False

    n_samples = len(df)
    logger.info(f"样本数: {n_samples}")
    logger.info(f"列名: {list(df.columns)}")

    # 检查必要的列
    required_cols = ['observation.state', 'action']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"缺少必要列: {col}")
            return False

    # 获取原始数据维度
    obs_state = df['observation.state'].iloc[0]
    if hasattr(obs_state, 'shape'):
        state_dim = len(obs_state) if isinstance(obs_state, (list, np.ndarray)) else obs_state.shape[0]
    else:
        state_dim = len(obs_state)
    action_sample = df['action'].iloc[0]
    if hasattr(action_sample, 'shape'):
        action_dim_orig = len(action_sample) if isinstance(action_sample, (list, np.ndarray)) else action_sample.shape[0]
    else:
        action_dim_orig = len(action_sample)

    logger.info(f"原始state维度: {state_dim}")
    logger.info(f"原始action维度: {action_dim_orig}")

    # 分配HDF5数据集
    logger.info(f"创建HDF5文件: {output_path}")
    with h5py.File(output_path, 'w') as f:
        # Observations
        grp_obs = f.create_group('observations')
        if 填补视觉占位符:
            grp_obs.create_dataset('vision', (n_samples, obs_dims['vision']), dtype='f4')
            grp_obs.create_dataset('lidar', (n_samples, obs_dims['lidar']), dtype='f4')
            grp_obs.create_dataset('tactile', (n_samples, obs_dims['tactile']), dtype='f4')
            grp_obs.create_dataset('force', (n_samples, obs_dims['force']), dtype='f4')
            grp_obs.create_dataset('imu', (n_samples, obs_dims['imu']), dtype='f4')
        grp_obs.create_dataset('state', (n_samples, state_dim), dtype='f4')

        # Actions
        f.create_dataset('actions', (n_samples, action_dim), dtype='f4')

        # Rewards
        f.create_dataset('rewards', (n_samples,), dtype='f4')

        # Dones
        f.create_dataset('dones', (n_samples,), dtype='bool')

        # Timestamps
        f.create_dataset('timestamps', (n_samples,), dtype='f8')

        # 填充数据
        logger.info("填充数据...")
        for i in range(n_samples):
            if i % 10000 == 0:
                logger.info(f"  进度: {i}/{n_samples}")

            row = df.iloc[i]

            # 填充观测数据
            if 填补视觉占位符:
                # 视觉/传感器数据用零填充（占位）
                grp_obs['vision'][i] = np.zeros(obs_dims['vision'], dtype='f4')
                grp_obs['lidar'][i] = np.zeros(obs_dims['lidar'], dtype='f4')
                grp_obs['tactile'][i] = np.zeros(obs_dims['tactile'], dtype='f4')
                grp_obs['force'][i] = np.zeros(obs_dims['force'], dtype='f4')
                grp_obs['imu'][i] = np.zeros(obs_dims['imu'], dtype='f4')

            # State数据
            state_val = row['observation.state']
            if isinstance(state_val, np.ndarray):
                grp_obs['state'][i] = state_val.astype('f4')
            else:
                grp_obs['state'][i] = np.array(state_val, dtype='f4')

            # Action
            action_val = row['action']
            if isinstance(action_val, np.ndarray):
                action_arr = action_val.astype('f4')
            else:
                action_arr = np.array(action_val, dtype='f4')
            # 填充或截断action
            if len(action_arr) < action_dim:
                action_padded = np.zeros(action_dim, dtype='f4')
                action_padded[:len(action_arr)] = action_arr
                f['actions'][i] = action_padded
            else:
                f['actions'][i] = action_arr[:action_dim]

            # Reward
            if 'next.reward' in row and not pd.isna(row['next.reward']):
                f['rewards'][i] = float(row['next.reward'])
            else:
                f['rewards'][i] = 0.0

            # Done
            if 'next.done' in row and not pd.isna(row['next.done']):
                f['dones'][i] = bool(row['next.done'])
            else:
                f['dones'][i] = False

            # Timestamp
            if 'timestamp' in row and not pd.isna(row['timestamp']):
                f['timestamps'][i] = float(row['timestamp'])
            else:
                f['timestamps'][i] = i * 0.1  # 默认100ms间隔

    logger.info(f"转换完成: {output_path}")
    return True


def find_parquet_files(cache_dir: str, dataset_name: str) -> list:
    """在ModelScope缓存目录中查找parquet文件"""
    import json

    parquet_files = []
    json_files = glob.glob(os.path.join(cache_dir, "*.json"))

    for json_file in json_files:
        try:
            with open(json_file) as f:
                content = f.read()
                if dataset_name in content and 'parquet' in content.lower():
                    # 找到匹配的json，获取对应的parquet文件
                    hash_name = os.path.basename(json_file).replace('.json', '')
                    parquet_path = os.path.join(cache_dir, hash_name)
                    if os.path.exists(parquet_path):
                        parquet_files.append(parquet_path)
        except:
            pass

    return parquet_files


# ========================
# 命令行接口
# ========================

def main():
    parser = argparse.ArgumentParser(
        description='将LeRobot数据集转换为SuperModel HDF5格式'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='lerobot parquet文件路径，或包含parquet文件的目录'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='输出HDF5文件路径'
    )
    parser.add_argument(
        '--grade', '-g',
        default='M',
        choices=['S', 'M', 'L', 'XL', 'XXL'],
        help='AGV等级 (default: M)'
    )
    parser.add_argument(
        '--dataset',
        default='berkeley_fanuc_manipulation',
        help='数据集名称，用于查找缓存文件'
    )
    parser.add_argument(
        '--no-placeholder',
        action='store_true',
        help='不为缺失的视觉/传感器数据生成零向量占位符'
    )
    parser.add_argument(
        '--cache-dir',
        default='/root/.cache/modelscope/hub/datasets/downloads',
        help='ModelScope缓存目录'
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    input_path = args.input
    output_path = args.output
    grade = args.grade.upper()

    # 如果input是目录，在缓存目录中查找parquet文件
    if os.path.isdir(input_path):
        parquet_files = glob.glob(os.path.join(input_path, "*.parquet"))
        if not parquet_files:
            logger.error(f"目录中未找到parquet文件: {input_path}")
            sys.exit(1)
        input_path = parquet_files[0]
        logger.info(f"使用parquet文件: {input_path}")

    # 如果指定了dataset名称，尝试自动查找
    if not os.path.exists(input_path) and args.dataset:
        parquet_files = find_parquet_files(args.cache_dir, args.dataset)
        if parquet_files:
            input_path = parquet_files[0]
            logger.info(f"自动找到parquet文件: {input_path}")

    if not os.path.exists(input_path):
        logger.error(f"文件不存在: {input_path}")
        sys.exit(1)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 执行转换
    success = convert_lerobot_to_hdf5(
        parquet_path=input_path,
        output_path=output_path,
        grade=grade,
        填补视觉占位符=not args.no_placeholder,
    )

    if success:
        logger.info("转换成功!")
        sys.exit(0)
    else:
        logger.error("转换失败")
        sys.exit(1)


if __name__ == '__main__':
    main()
