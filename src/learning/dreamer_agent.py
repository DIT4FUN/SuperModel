# Copyright (C) 2024-2026 赵元请 (DIT4FUN)
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
Dreamer Agent - 基于想象轨迹的强化学习
=========================================

实现 Dreamer 风格的学习算法:
1. 学习世界模型
2. 从世界模型中想象轨迹
3. 使用想象轨迹更新策略和价值函数

参考: Dreamer: Learning to Imagine and Plan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

# Import sensors encoders
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'sensors'))
from sensors.encoders import SensorEncoderWrapper, EncoderConfig


@dataclass
class DreamerConfig:
    """Dreamer 配置"""
    # World Model
    latent_dim: int = 256
    hidden_dim: int = 512
    deter_dim: int = 256
    stoch_dim: int = 32
    num_classes: int = 32
    
    # Actor-Critic
    actor_dim: int = 256
    critic_dim: int = 256
    action_dim: int = 6
    
    # 想象训练
    imagination_horizon: int = 15
    gamma: float = 0.99
    lambda_: float = 0.95
    
    # 损失权重
    actor_weight: float = 1.0
    critic_weight: float = 1.0
    kl_weight: float = 0.1
    
    # 优化器
    world_model_lr: float = 1e-4
    actor_lr: float = 3e-5
    critic_lr: float = 1e-4


class Actor(nn.Module):
    """Dreamer Actor - 随机策略网络"""
    
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        fixed_std: Optional[float] = None
    ):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.fixed_std = fixed_std
        
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        if fixed_std is None:
            self.log_std = nn.Linear(hidden_dim, action_dim)
            
    def forward(
        self, 
        latent: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent: (B, latent_dim) 隐状态
            deterministic: 是否使用确定性策略
            
        Returns:
            action: (B, action_dim)
            log_prob: (B, action_dim) 每个维度的 log 概率
        """
        x = self.network(latent)
        mean = torch.tanh(self.mean(x))
        
        if deterministic:
            # 确定性策略：直接返回均值
            return mean, torch.zeros_like(mean).sum(dim=-1, keepdim=True)
            
        if self.fixed_std is not None:
            std = self.fixed_std
        else:
            log_std = self.log_std(x)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            std = torch.exp(log_std)
            
        # 采样
        dist = Normal(mean, std)
        action_raw = dist.rsample()
        action = torch.tanh(action_raw)
        
        # log_prob (考虑 tanh 变换)
        log_prob = dist.log_prob(action_raw)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # (B, 1)
        
        return action, log_prob
    
    def get_action(
        self, 
        latent: torch.Tensor,
        exploration_noise: float = 0.0
    ) -> np.ndarray:
        """获取动作 (用于推理)"""
        with torch.no_grad():
            action, _ = self.forward(latent, deterministic=False)
            action = action.cpu().numpy()
            
            if exploration_noise > 0:
                noise = np.random.randn(*action.shape) * exploration_noise
                action = np.clip(action + noise, -1, 1)
                
        return action


class Critic(nn.Module):
    """Dreamer Critic - 价值函数网络"""
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, latent_dim) or (T, B, latent_dim)
            
        Returns:
            value: (B, 1) or (T, B, 1)
        """
        return self.network(latent)


class DreamerAgent(nn.Module):
    """
    Dreamer 智能体
    
    整合:
    - 世界模型 (已学习或提供)
    - Actor (策略网络)
    - Critic (价值网络)
    - 想象轨迹训练
    """
    
    def __init__(
        self,
        world_model: nn.Module,  # 预训练的世界模型
        action_dim: int,
        config: Optional[DreamerConfig] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.device = device
        self.config = config or DreamerConfig(action_dim=action_dim)
        self.cfg = self.config
        self.action_dim = action_dim
        
        # 世界模型 (冻结，只用于想象)
        self.world_model = world_model
        for param in self.world_model.parameters():
            param.requires_grad = False
            
        # Actor
        self.actor = Actor(
            latent_dim=self.cfg.latent_dim,
            action_dim=action_dim,
            hidden_dim=self.cfg.actor_dim,
            fixed_std=0.3  # Dreamer 使用的固定标准差
        )
        
        # Critic
        self.critic = Critic(
            latent_dim=self.cfg.latent_dim,
            hidden_dim=self.cfg.critic_dim
        )
        
        # 目标 Critic (用于计算 tdlambda)
        self.target_critic = Critic(
            latent_dim=self.cfg.latent_dim,
            hidden_dim=self.cfg.critic_dim
        )
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.cfg.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.cfg.critic_lr
        )
        
    def imagine(
        self,
        initial_deter: torch.Tensor,
        initial_stoch: torch.Tensor,
        horizon: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        想象 rollout
        
        Args:
            initial_deter: (B, deter_dim) 初始 deterministic 状态
            initial_stoch: (B, stoch_dim * num_classes) 初始随机状态
            horizon: 想象步数
            
        Returns:
            latent_seq: (horizon, B, latent_dim) 隐状态序列
            action_seq: (horizon, B, action_dim) 动作序列
            reward_seq: (horizon, B) 奖励序列
            value_seq: (horizon, B) 价值序列
        """
        B = initial_deter.shape[0]
        horizon = horizon or self.cfg.imagination_horizon
        
        latent_list = []
        action_list = []
        reward_list = []
        value_list = []
        
        deter = initial_deter
        stoch = initial_stoch
        action = torch.zeros(B, self.action_dim, device=self.device)
        
        for t in range(horizon):
            # 构建隐状态
            latent = self.world_model.get_latent(deter, stoch)
            latent_list.append(latent)
            
            # 策略采样动作
            action, _ = self.actor(latent)
            action_list.append(action)
            
            # 世界模型前进一步
            deter, _, stoch = self.world_model.transition(
                deter, action, None, stoch
            )
            
            # 预测奖励和价值
            latent_next = self.world_model.get_latent(deter, stoch)
            reward = self.world_model.reward_model(latent_next)
            value = self.critic(latent_next)
            
            reward_list.append(reward.squeeze(-1))
            value_list.append(value.squeeze(-1))
            
        return (
            torch.stack(latent_list),      # (horizon, B, latent_dim)
            torch.stack(action_list),      # (horizon, B, action_dim)
            torch.stack(reward_list),     # (horizon, B)
            torch.stack(value_list)        # (horizon, B)
        )
    
    def compute_return(
        self,
        reward_seq: torch.Tensor,
        value_seq: torch.Tensor,
        gamma: float = 0.99,
        lambda_: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 λ-return
        
        Args:
            reward_seq: (T, B)
            value_seq: (T, B)
            
        Returns:
            returns: (T, B) 回报
            advantages: (T, B) 优势
        """
        T, B = reward_seq.shape
        
        # Bootstrap value
        bootstrap = value_seq[-1]  # (B,)
        
        returns = []
        advantages = []
        
        gae = torch.zeros(B, device=self.device)
        
        for t in reversed(range(T)):
            # TD(λ) 优势
            delta = reward_seq[t] + gamma * bootstrap - value_seq[t]
            gae = delta + gamma * lambda_ * gae
            bootstrap = value_seq[t]
            
            # 回报
            ret = gae + value_seq[t]
            
            returns.insert(0, ret)
            advantages.insert(0, gae)
            
        return torch.stack(returns), torch.stack(advantages)
    
    def update_actor(
        self,
        latent_seq: torch.Tensor,
        action_seq: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor
    ) -> Dict[str, float]:
        """
        更新 Actor
        
        Args:
            latent_seq: (T, B, latent_dim)
            action_seq: (T, B, action_dim)
            returns: (T, B)
            advantages: (T, B)
            
        Returns:
            metrics: 训练指标
        """
        T, B = action_seq.shape[:2]
        
        # 重塑
        latent_flat = latent_seq.reshape(T * B, -1)
        action_flat = action_seq.reshape(T * B, -1)
        returns_flat = returns.reshape(T * B)
        advantages_flat = advantages.reshape(T * B)
        
        # 标准化优势
        advantages_norm = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        
        # 策略损失 (最大化期望回报)
        action_pred, log_prob = self.actor(latent_flat)
        
        # 策略梯度损失
        actor_loss = -(log_prob.squeeze(-1) * advantages_norm).mean()
        
        # 熵正则化 (鼓励探索)
        entropy = -log_prob.mean()
        
        # 总损失
        total_loss = actor_loss - 0.01 * entropy
        
        self.actor_optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=100.0)
        
        self.actor_optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item(),
            'adv_mean': advantages_flat.mean().item(),
            'adv_std': advantages_flat.std().item()
        }
    
    def update_critic(
        self,
        latent_seq: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, float]:
        """
        更新 Critic
        
        Args:
            latent_seq: (T, B, latent_dim)
            returns: (T, B)
            
        Returns:
            metrics: 训练指标
        """
        T, B = returns.shape
        
        # 重塑
        latent_flat = latent_seq.reshape(T * B, -1)
        returns_flat = returns.reshape(T * B)
        
        # 价值损失
        value_pred = self.critic(latent_flat).squeeze(-1)
        critic_loss = F.mse_loss(value_pred, returns_flat)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=100.0)
        
        self.critic_optimizer.step()
        
        # 更新目标网络
        with torch.no_grad():
            for target_param, param in zip(
                self.target_critic.parameters(),
                self.critic.parameters()
            ):
                target_param.data.copy_(
                    0.005 * param.data + 0.995 * target_param.data
                )
                
        return {
            'critic_loss': critic_loss.item(),
            'value_pred_mean': value_pred.mean().item(),
            'value_target_mean': returns_flat.mean().item()
        }
    
    def train_step(
        self,
        initial_deter: torch.Tensor,
        initial_stoch: torch.Tensor
    ) -> Dict[str, float]:
        """
        一步训练
        
        Args:
            initial_deter: (B, deter_dim)
            initial_stoch: (B, stoch_dim * num_classes)
            
        Returns:
            metrics: 所有训练指标
        """
        # 想象 rollout
        latent_seq, action_seq, reward_seq, value_seq = self.imagine(
            initial_deter, initial_stoch, self.cfg.imagination_horizon
        )
        
        # 计算回报和优势
        returns, advantages = self.compute_return(
            reward_seq, value_seq,
            gamma=self.cfg.gamma,
            lambda_=self.cfg.lambda_
        )
        
        # 更新 Actor
        actor_metrics = self.update_actor(
            latent_seq, action_seq, returns, advantages
        )
        
        # 更新 Critic
        critic_metrics = self.update_critic(latent_seq, returns)
        
        return {**actor_metrics, **critic_metrics}


class IntegratedAgent(nn.Module):
    """
    集成智能体
    
    整合:
    - 传感器编码器
    - 世界模型
    - Dreamer Actor-Critic
    """
    
    def __init__(
        self,
        obs_dims: Dict[str, int],
        action_dim: int,
        encoder_config,
        world_model_config,
        dreamer_config: Optional[DreamerConfig] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        
        # 传感器编码器
        from sensors.encoders import SensorEncoderWrapper
        self.encoder = SensorEncoderWrapper(
            obs_dims=obs_dims,
            encoder_config=encoder_config,
            latent_dim=encoder_config.latent_dim
        )
        
        # 世界模型
        from .world_model import WorldModel
        self.world_model = WorldModel(
            obs_dims=obs_dims,
            action_dim=action_dim,
            config=world_model_config
        )
        
        # Dreamer
        self.dreamer = DreamerAgent(
            world_model=self.world_model,
            action_dim=action_dim,
            config=dreamer_config,
            device=device
        )
        
        # 经验回放
        self.replay_buffer = None  # 稍后初始化
        
    def encode_observation(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """编码观测"""
        return self.encoder(observations)
        
    def select_action(
        self,
        observations: Dict[str, torch.Tensor],
        deterministic: bool = False
    ) -> np.ndarray:
        """选择动作"""
        self.eval()
        
        with torch.no_grad():
            # 编码观测
            encoded = self.encode_observation(observations)
            
            # 获取隐状态
            # (这里需要维护隐状态，简化为使用编码特征)
            latent = encoded.get('fused', torch.stack(list(encoded.values())).mean(0))
            latent = latent.unsqueeze(0) if latent.dim() == 1 else latent
            
            # 策略
            action = self.dreamer.actor.get_action(latent)
            
        return action.squeeze(0) if action.shape[0] == 1 else action
    
    def update(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """更新智能体"""
        self.train()
        
        # 更新世界模型
        wm_losses = self.world_model.update(
            batch,
            torch.optim.Adam(self.world_model.parameters(), lr=1e-4)
        )
        
        # 更新 Dreamer
        # 需要初始隐状态
        B = batch['observations']['vision'].shape[0]
        initial_deter = torch.zeros(B, self.world_model.config.deter_dim, device=self.device)
        initial_stoch = torch.zeros(B, self.world_model.config.stoch_dim * self.world_model.config.num_classes, device=self.device)
        
        dreamer_metrics = self.dreamer.train_step(initial_deter, initial_stoch)
        
        return {**wm_losses, **dreamer_metrics}


# 创建工厂函数
def create_integrated_agent(
    obs_dims: Dict[str, int],
    action_dim: int,
    grade: str = 'M',
    device: Optional[str] = None
) -> IntegratedAgent:
    """创建集成智能体"""
    from sensors.encoders import get_encoder_config, EncoderConfig
    from .world_model import WorldModelConfig, create_world_model_agent
    
    encoder_config = get_encoder_config(grade)
    world_model_config = WorldModelConfig(
        latent_dim=encoder_config.latent_dim,
        hidden_dim=encoder_config.hidden_dim,
        action_dim=action_dim
    )
    dreamer_config = DreamerConfig(
        latent_dim=encoder_config.latent_dim,
        hidden_dim=encoder_config.hidden_dim,
        action_dim=action_dim
    )
    
    return IntegratedAgent(
        obs_dims=obs_dims,
        action_dim=action_dim,
        encoder_config=encoder_config,
        world_model_config=world_model_config,
        dreamer_config=dreamer_config,
        device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    )
