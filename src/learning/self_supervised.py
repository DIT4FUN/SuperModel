# Copyright (C) 2026 焦洋 (Jiao Yang) <jiaoyang@cczu.edu.cn>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
自主学习模块
============

自监督学习与在线强化学习
- 对比学习 (跨模态对齐)
- 世界模型预测
- 好奇心驱动探索
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LearningConfig:
    """学习配置"""
    # 对比学习
    temperature: float = 0.1
    contrastive_weight: float = 1.0
    
    # 世界模型
    world_model_weight: float = 0.5
    imagination_horizon: int = 5
    
    # 好奇心
    curiosity_weight: float = 0.2
    intrinsic_reward_scale: float = 1.0
    
    # 优化器
    lr: float = 1e-4
    weight_decay: float = 1e-5


class ContrastiveLoss(nn.Module):
    """
    对比损失
    
    InfoNCE 损失，用于跨模态对齐
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self, 
        anchor: torch.Tensor, 
        positive: torch.Tensor, 
        negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            anchor: B x D, 锚点特征
            positive: B x D, 正样本
            negatives: B x K x D, 负样本
            
        Returns:
            loss: 标量
        """
        B = anchor.shape[0]
        
        # 归一化
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)
        
        # 正样本相似度
        pos_sim = (anchor * positive).sum(dim=-1) / self.temperature  # B
        
        # 负样本相似度
        neg_sim = (anchor.unsqueeze(1) * negatives).sum(dim=-1) / self.temperature  # B x K
        
        # InfoNCE
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # B x (1+K)
        labels = torch.zeros(B, dtype=torch.long, device=anchor.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


class WorldModel(nn.Module):
    """
    世界模型
    
    预测下一个状态和奖励
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        reward_dim: int = 1
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 状态转移网络: s' = f(s, a)
        self.transition = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # 奖励网络: r = g(s, a)
        self.reward = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, reward_dim)
        )
        
        # 观测解码器: o = h(s)
        self.decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim)  # 重构观测
        )
        
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测下一个状态和奖励
        
        Args:
            state: B x state_dim
            action: B x action_dim
            
        Returns:
            next_state: B x state_dim
            reward: B x 1
        """
        sa = torch.cat([state, action], dim=-1)
        next_state = self.transition(sa)
        reward = self.reward(sa)
        return next_state, reward
    
    def imagine(
        self,
        state: torch.Tensor,
        policy: nn.Module,
        horizon: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        想象 rollout
        
        Args:
            state: 当前状态
            policy: 动作策略网络
            horizon: 预测步数
            
        Returns:
            states: horizon x B x state_dim
            rewards: horizon x B
        """
        T, B = horizon, state.shape[0]
        states = [state]
        rewards = []
        
        for t in range(T):
            with torch.no_grad():
                action = policy(state)  # B x action_dim
            next_state, reward = self.forward(state, action)
            states.append(next_state)
            rewards.append(reward.squeeze(-1))
            state = next_state
            
        return torch.stack(states), torch.stack(rewards)


class IntrinsicCuriosity(nn.Module):
    """
    内在好奇心驱动
    
    基于预测误差的内在奖励
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        # 逆动力学模型: a' = f(s, s')
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 正向预测模型: s' = g(s, a)
        self.forward_dynamics = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        计算内在奖励
        
        Args:
            state: B x state_dim
            next_state: B x state_dim
            action: B x action_dim
            
        Returns:
            intrinsic_reward: B
        """
        # 正向预测误差作为内在奖励
        pred_next = self.forward_dynamics(torch.cat([state, action], dim=-1))
        error = (pred_next - next_state).pow(2).mean(dim=-1)
        
        return error


class AutonomousLearning:
    """
    自主学习框架
    
    整合:
    - 对比学习 (跨模态对齐)
    - 世界模型
    - 好奇心驱动探索
    """
    
    def __init__(self, config: LearningConfig):
        self.config = config
        
        # 损失函数
        self.contrastive_loss = ContrastiveLoss(config.temperature)
        
        # 优化器
        self.optimizers = {}
        
    def train_step(
        self,
        models: Dict[str, nn.Module],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        训练一步
        
        Args:
            models: 包含各模型的字典
                - fusion: 融合网络
                - world_model: 世界模型
                - curiosity: 好奇心模块
                - policy: 策略网络
            batch: 包含以下键的数据批次
                - vision, audio, tactile, force, imu: 各模态特征
                - action: 动作
                - reward: 奖励
                - next_state: 下一状态
                
        Returns:
            metrics: 损失字典
        """
        metrics = {}
        
        # 1. 跨模态对比学习
        if 'fusion' in models and 'cross_modal' in batch:
            cross = batch['cross_modal']
            loss_contrastive = self.contrastive_loss(
                cross['anchor'],
                cross['positive'],
                cross['negatives']
            )
            metrics['contrastive'] = loss_contrastive.item()
            
        # 2. 世界模型学习
        if 'world_model' in models and 'state' in batch and 'action' in batch:
            wm = models['world_model']
            state = batch['state']
            action = batch['action']
            next_state = batch.get('next_state')
            reward = batch.get('reward')
            
            # 预测
            pred_next, pred_reward = wm(state, action)
            
            # 世界模型损失
            if next_state is not None:
                loss_transition = F.mse_loss(pred_next, next_state)
                metrics['transition'] = loss_transition.item()
            else:
                loss_transition = torch.tensor(0.0)
                
            if reward is not None:
                loss_reward = F.mse_loss(pred_reward.squeeze(), reward)
                metrics['reward'] = loss_reward.item()
            else:
                loss_reward = torch.tensor(0.0)
            
            loss_world = loss_transition + loss_reward
            
        # 3. 好奇心损失
        if 'curiosity' in models and 'state' in batch and 'next_state' in batch:
            cur = models['curiosity']
            intrinsic = cur(
                batch['state'],
                batch['next_state'],
                batch['action']
            )
            loss_curiosity = intrinsic.mean()
            metrics['curiosity'] = loss_curiosity.item()
            
        return metrics
    
    def compute_intrinsic_reward(
        self,
        curiosity_model: nn.Module,
        state: np.ndarray,
        next_state: np.ndarray,
        action: np.ndarray
    ) -> np.ndarray:
        """计算内在奖励"""
        state_t = torch.from_numpy(state).float()
        next_state_t = torch.from_numpy(next_state).float()
        action_t = torch.from_numpy(action).float()
        
        with torch.no_grad():
            intrinsic = curiosity_model(state_t, next_state_t, action_t)
            
        return intrinsic.cpu().numpy()
    
    def update(
        self,
        models: Dict[str, nn.Module],
        batch: Dict[str, torch.Tensor],
        optimizers: Dict[str, torch.optim.Optimizer]
    ):
        """更新模型"""
        total_loss = torch.tensor(0.0)
        
        # 合并所有损失
        metrics = self.train_step(models, batch)
        
        # 反向传播
        for name, opt in optimizers.items():
            opt.zero_grad()
        # ... (需要完整的损失计算)
        # optimizer.step()
        

# AGV五级学习规格
AGV_LEARNING_GRADES = {
    'S': {'batch_size': 16, 'lr': 1e-3, 'hidden_dim': 128},
    'M': {'batch_size': 32, 'lr': 5e-4, 'hidden_dim': 256},
    'L': {'batch_size': 64, 'lr': 1e-4, 'hidden_dim': 512},
    'XL': {'batch_size': 128, 'lr': 5e-5, 'hidden_dim': 768},
    'XXL': {'batch_size': 256, 'lr': 1e-5, 'hidden_dim': 1024}
}


def get_learning_spec(grade: str) -> dict:
    """获取AGV指定等级的学习规格"""
    return AGV_LEARNING_GRADES.get(grade, AGV_LEARNING_GRADES['M'])
