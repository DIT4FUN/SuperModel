#!/usr/bin/env python3
"""
train_world_model.py - World Model 训练脚本
SuperModel 超模态大模型具身智能系统

基于 Dreamer 风格的世界模型训练:
1. RSSM 状态空间模型
2. 跨模态感知融合 (视觉/听觉/触觉/力觉/IMU)
3. 想象轨迹 rollouts
4. Actor-Critic 策略学习

用法:
    # 单 GPU
    python scripts/train_world_model.py --grade M --episodes 1000
    
    # 多 GPU (DeepSpeed ZeRO)
    deepspeed --num_gpus=2 scripts/train_world_model.py --grade M --episodes 1000
"""

import argparse
import os
import sys
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.learning.world_model import (
    WorldModel,
    WorldModelConfig,
    WorldModelAgent,
    create_world_model_agent,
    WORLD_MODEL_GRADES,
    ReplayBuffer,
)
from src.embodied.vla_model import VLAPerceptionFrame, VLAInput, VLAAction


# ============================================================
# 数据集生成 - 模拟多模态传感器数据
# ============================================================

class SyntheticAGVDataset(Dataset):
    """
    合成 AGV 训练数据集
    
    生成模拟的多模态传感器数据用于训练世界模型:
    - 视觉: 伪随机图像特征
    - 激光雷达: 距离扫描
    - 触觉: 接触力
    - 力觉: 六维力矩
    - IMU: 加速度/角速度
    - 奖励: 任务完成度
    """
    
    def __init__(
        self,
        num_episodes: int = 1000,
        episode_length: int = 128,
        grade: str = "M",
        obs_dims: Optional[Dict[str, int]] = None,
        action_dim: int = 7,
        seed: int = 42,
    ):
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.grade = grade
        self.action_dim = action_dim
        
        if obs_dims is None:
            self.obs_dims = {
                'vision': 512,    # 视觉特征
                'lidar': 128,     # 激光雷达
                'tactile': 64,    # 触觉
                'force': 6,       # 六维力矩
                'imu': 6,         # IMU
            }
        else:
            self.obs_dims = obs_dims
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 预生成所有数据
        self._generate_data()
    
    def _generate_data(self):
        """预生成所有 episodes 的数据"""
        self.episodes = []
        
        for ep in range(self.num_episodes):
            episode = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'dones': [],
            }
            
            # 初始化状态
            pos = np.random.rand(2) * 10  # x, y 位置
            heading = np.random.rand() * 2 * np.pi  # 朝向
            linear_vel = 0.0
            angular_vel = 0.0
            
            for t in range(self.episode_length):
                # 生成随机指令
                instruction = np.random.rand()  # 目标方向
                
                # 动作 (AGV 差速控制)
                v_linear = np.clip(linear_vel + np.random.randn() * 0.1, -1.5, 1.5)
                v_angular = np.clip(angular_vel + np.random.randn() * 0.2, -np.pi, np.pi)
                action = np.array([v_linear, v_angular])
                if self.action_dim > 2:
                    action = np.append(action, [0.0] * (self.action_dim - 2))
                
                # 更新状态
                heading += v_angular * 0.1
                pos[0] += v_linear * np.cos(heading) * 0.1
                pos[1] += v_linear * np.sin(heading) * 0.1
                
                # 激光雷达模拟
                lidar = np.random.rand(128).astype(np.float32) * 10.0  # 0-10m 距离
                if t > 0:
                    # 障碍物模拟
                    if np.random.rand() < 0.1:
                        obstacle_angle = np.random.rand() * 2 * np.pi
                        obstacle_dist = np.random.rand() * 3 + 0.5
                        obstacle_idx = int((obstacle_angle + np.pi) / (2 * np.pi) * 128)
                        if 0 <= obstacle_idx < 128:
                            lidar[obstacle_idx] = obstacle_dist
                
                # 视觉特征 (512维随机投影)
                vision = np.random.randn(self.obs_dims['vision']).astype(np.float32) * 0.3
                
                # 触觉
                tactile = np.random.randn(self.obs_dims['tactile']).astype(np.float32) * 0.1
                
                # 力觉
                force = np.random.randn(6).astype(np.float32) * 0.05
                
                # IMU
                imu = np.array([
                    v_linear * 0.5 + np.random.randn() * 0.01,  # ax
                    v_angular * 0.3 + np.random.randn() * 0.01,  # ay
                    9.81 + np.random.randn() * 0.01,  # az
                    np.random.randn() * 0.01,  # wx
                    np.random.randn() * 0.01,  # wy
                    heading + np.random.randn() * 0.01,  # wz (yaw rate)
                ], dtype=np.float32)
                
                # 观测
                obs = {
                    'vision': vision,
                    'lidar': lidar.astype(np.float32),
                    'tactile': tactile,
                    'force': force,
                    'imu': imu,
                }
                
                # 奖励 (基于接近目标)
                reward = float(np.random.randn() * 0.1 - 0.05)
                # 简单任务奖励
                if t % 20 == 0:
                    reward += np.random.rand() * 0.5
                
                # done
                done = (t == self.episode_length - 1)
                
                episode['observations'].append({
                    'vision': vision,
                    'lidar': lidar,
                    'tactile': tactile,
                    'force': force,
                    'imu': imu,
                })
                episode['actions'].append(action.astype(np.float32))
                episode['rewards'].append(reward)
                episode['dones'].append(done)
            
            self.episodes.append(episode)
    
    def __len__(self):
        return self.num_episodes * self.episode_length
    
    def __getitem__(self, idx):
        ep_idx = idx // self.episode_length
        step_idx = idx % self.episode_length
        
        ep = self.episodes[ep_idx]
        
        obs = ep['observations'][step_idx].copy()
        action = ep['actions'][step_idx].copy()
        reward = ep['rewards'][step_idx]
        done = ep['dones'][step_idx]
        
        return obs, action, reward, done
    
    def collate_fn(self, batch):
        """自定义批处理"""
        obs_list, actions, rewards, dones = zip(*batch)
        
        # 堆叠观测
        obs_batch = {}
        for modality in obs_list[0].keys():
            obs_batch[modality] = torch.from_numpy(
                np.stack([o[modality] for o in obs_list])
            ).float()
        
        actions_batch = torch.from_numpy(np.stack(actions)).float()
        rewards_batch = torch.from_numpy(np.stack(rewards)).float()
        dones_batch = torch.from_numpy(np.stack(dones)).float()
        
        return obs_batch, actions_batch, rewards_batch, dones_batch


# ============================================================
# 训练器
# ============================================================

class WorldModelTrainer:
    """World Model 训练器"""
    
    def __init__(
        self,
        grade: str = "M",
        device: str = "cuda",
        output_dir: str = "./checkpoints",
        log_interval: int = 10,
        save_interval: int = 100,
        batch_size: int = 64,
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        grad_clip: float = 100.0,
        use_distributed: bool = False,
        local_rank: int = 0,
    ):
        self.grade = grade.upper()
        self.device = device
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.use_distributed = use_distributed
        self.local_rank = local_rank
        self.global_step = 0
        self.episode = 0
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取模型配置
        self.config = WORLD_MODEL_GRADES.get(self.grade, WORLD_MODEL_GRADES['M'])
        self.config.action_dim = 7  # COMBINED action space
        
        # 观测维度
        obs_dims = {
            'vision': 512,
            'lidar': 128,
            'tactile': 64,
            'force': 6,
            'imu': 6,
        }
        
        # 创建模型
        self.model = WorldModel(
            obs_dims=obs_dims,
            action_dim=7,
            config=self.config,
        )
        self.model.to(device)
        
        # 分布式
        if use_distributed:
            self.model = DDP(self.model, device_ids=[local_rank])
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        # 学习率调度
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
        # 统计
        self.losses_history = []
        
        print(f"[Rank {local_rank}] Trainer initialized: grade={self.grade}, "
              f"device={device}, params={sum(p.numel() for p in self.model.parameters())/1e6:.1f}M")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步训练"""
        if isinstance(self.model, DDP):
            model = self.model.module
        else:
            model = self.model
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        losses = model.compute_loss(
            observations=batch['observations'],
            actions=batch['actions'],
            rewards=batch['rewards'],
            dones=batch['dones'],
        )
        
        # 反向传播
        losses['total'].backward()
        
        # 梯度裁剪
        if isinstance(self.model, DDP):
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.module.parameters(), max_norm=self.grad_clip
            )
        else:
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip
            )
        
        self.optimizer.step()
        self.scheduler.step()
        
        self.global_step += 1
        
        return {k: v.item() for k, v in losses.items()} | {'grad_norm': grad_norm.item()}
    
    def train(
        self,
        num_episodes: int = 1000,
        num_workers: int = 4,
    ):
        """完整训练流程"""
        print(f"\n{'='*60}")
        print(f"World Model Training - Grade: {self.grade}")
        print(f"{'='*60}")
        print(f"Episodes: {num_episodes}")
        print(f"Batch size: {self.batch_size}")
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")
        
        # 创建数据集
        dataset = SyntheticAGVDataset(
            num_episodes=num_episodes,
            episode_length=128,
            grade=self.grade,
        )
        
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(not self.use_distributed),
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            pin_memory=True,
        )
        
        # 训练循环
        epoch = 0
        t_start = time.time()
        
        for batch in loader:
            obs, actions, rewards, dones = batch
            
            # 移到 GPU
            obs = {k: v.to(self.device) for k, v in obs.items()}
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            dones = dones.to(self.device)
            
            # 构造成序列格式 (T, B, ...)
            T = 1  # 单步训练
            batch_seq = {
                'observations': {k: v.unsqueeze(0) for k, v in obs.items()},
                'actions': actions.unsqueeze(0),
                'rewards': rewards.unsqueeze(0),
                'dones': dones.unsqueeze(0),
            }
            
            # 训练一步
            loss_dict = self.train_step(batch_seq)
            self.losses_history.append(loss_dict)
            
            # 日志
            if self.global_step % self.log_interval == 0 and self.local_rank == 0:
                elapsed = time.time() - t_start
                lr = self.scheduler.get_last_lr()[0]
                
                print(f"[Step {self.global_step}] "
                      f"Loss: {loss_dict['total']:.4f} | "
                      f"KL: {loss_dict['kl']:.4f} | "
                      f"Reward: {loss_dict['reward']:.4f} | "
                      f"Decoder: {loss_dict['decoder']:.4f} | "
                      f"Grad: {loss_dict['grad_norm']:.2f} | "
                      f"LR: {lr:.2e} | "
                      f"Elapsed: {elapsed:.1f}s")
            
            # 保存
            if self.global_step % self.save_interval == 0 and self.local_rank == 0:
                self.save_checkpoint(f"step_{self.global_step}.pt")
                print(f"[Rank {self.local_rank}] Checkpoint saved: step_{self.global_step}.pt")
            
            # 安全检查
            if loss_dict['total'] > 100:
                print(f"[WARNING] Loss exploded: {loss_dict['total']:.2f}, skipping...")
                break
        
        # 最终保存
        if self.local_rank == 0:
            self.save_checkpoint("final.pt")
            print(f"\nTraining complete! Final model saved to {self.output_dir}/final.pt")
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        if isinstance(self.model, DDP):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        checkpoint = {
            'model': model_state,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'grade': self.grade,
        }
        
        torch.save(checkpoint, os.path.join(self.output_dir, filename))
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.global_step = checkpoint['global_step']
        
        print(f"Loaded checkpoint from {path} (step {self.global_step})")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="World Model Training")
    parser.add_argument("--grade", type=str, default="M", choices=["S", "M", "L", "XL", "XXL"])
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=100.0)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume")
    
    # DeepSpeed / 分布式
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--distributed", action="store_true")
    
    args = parser.parse_args()
    
    # 分布式训练设置
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        device = f"cuda:{args.local_rank}"
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        args.local_rank = 0
    
    # 创建时间戳输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"supermodel_{args.grade}_{timestamp}")
    
    # 创建训练器
    trainer = WorldModelTrainer(
        grade=args.grade,
        device=device,
        output_dir=output_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_clip=args.grad_clip,
        use_distributed=args.distributed,
        local_rank=args.local_rank,
    )
    
    # 恢复检查点
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train(
        num_episodes=args.episodes,
        num_workers=args.num_workers,
    )
    
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
