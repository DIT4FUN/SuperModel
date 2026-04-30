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
World Model - 世界模型
=====================

实现 RSSM (Recurrent State Space Model) 类似 Dreamer 的世界模型

核心组件:
1. ObservationEncoder - 将观测编码为隐状态
2. TransitionModel - 状态转移 (s' = f(s, a))
3. ObservationDecoder - 从隐状态重建观测
4. RewardModel - 奖励预测
5. ValueModel - 价值估计
6. Actor-Critic - 基于想象轨迹的策略学习

参考:
- Dreamer: Learning World Models for Imagined Trajectories
- World Models (Ha & Schmidhuber)
- PlaNet: Learning Planning-based RL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Distribution
import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass


@dataclass
class WorldModelConfig:
    """世界模型配置"""
    # 隐状态维度
    latent_dim: int = 256
    hidden_dim: int = 512
    rnn_hidden_dim: int = 256
    
    # 观测编码
    obs_encoder_dim: int = 256
    
    # 动作
    action_dim: int = 6
    
    # RSSM 参数
    deter_dim: int = 256
    stoch_dim: int = 32
    num_classes: int = 32
    
    # 想象 rollout
    imagination_horizon: int = 15
    gamma: float = 0.99
    lambda_: float = 0.95
    
    # 损失权重
    kl_weight: float = 1.0
    reward_weight: float = 1.0
    decoder_weight: float = 1.0
    continue_weight: float = 1.0
    
    # 探索
    action_noise: float = 0.3
    exploration_eps: float = 0.1


class ModelState(NamedTuple):
    """模型状态"""
    deter: torch.Tensor  # 确定性隐状态 h
    stoch: torch.Tensor  # 随机隐状态 z
    action: Optional[torch.Tensor] = None  # 上一步动作


class ModelOutput(NamedTuple):
    """模型输出"""
    latent_mean: torch.Tensor
    latent_std: torch.Tensor
    stoch: torch.Tensor
    deter: torch.Tensor
    reward: torch.Tensor
    pcont: torch.Tensor  # continue probability (episode continuation)
    observation: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class ObservationEncoder(nn.Module):
    """
    观测编码器
    
    将多模态观测 (视觉/听觉/触觉/力觉/IMU) 编码为隐表示
    """
    
    def __init__(
        self,
        obs_dims: Dict[str, int],
        hidden_dim: int = 256,
        latent_dim: int = 256
    ):
        super().__init__()
        self.obs_dims = obs_dims
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # 各模态编码器
        self.modality_encoders = nn.ModuleDict()
        for name, dim in obs_dims.items():
            self.modality_encoders[name] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU()
            )
        
        # 模态融合
        num_modalities = len(obs_dims)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim // 2 * num_modalities, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            observations: {modality_name: tensor of shape (B, obs_dim)}
            
        Returns:
            encoded: (B, latent_dim)
        """
        encoded_modalities = []
        
        for name, obs in observations.items():
            if name in self.modality_encoders:
                encoded = self.modality_encoders[name](obs)
                encoded_modalities.append(encoded)
                
        # 拼接所有模态
        fused = torch.cat(encoded_modalities, dim=-1)
        encoded = self.fusion(fused)
        
        return encoded


class ObservationDecoder(nn.Module):
    """
    观测解码器
    
    从隐状态重建各模态观测
    """
    
    def __init__(
        self,
        latent_dim: int,
        obs_dims: Dict[str, int],
        hidden_dim: int = 256
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dims = obs_dims
        
        # 共享特征提取
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 各模态解码器
        self.modality_decoders = nn.ModuleDict()
        for name, dim in obs_dims.items():
            self.modality_decoders[name] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, dim)
            )
            
    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            latent: (B, latent_dim) 隐状态
            
        Returns:
            reconstructions: {modality_name: tensor of shape (B, obs_dim)}
        """
        features = self.shared(latent)
        
        reconstructions = {}
        for name, decoder in self.modality_decoders.items():
            reconstructions[name] = decoder(features)
            
        return reconstructions


class TransitionModel(nn.Module):
    """
    转移模型 (RSSM 核心)
    
    包含:
    1. 先行模型 (prior): p(z'|h, a) - 给定上一步 deterministic 状态和动作，预测下一步随机状态
    2. 后行模型 (posterior): q(z'|h, z, o) - 给定状态、随机变量和观测，更新状态
    3. 递归模型 (recurrent): h' = g(h, z, a) - 给定上一步 h、z 和动作，输出新的 h
    """
    
    def __init__(
        self,
        action_dim: int,
        deter_dim: int = 256,
        stoch_dim: int = 32,
        num_classes: int = 32,
        hidden_dim: int = 512
    ):
        super().__init__()
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.num_classes = num_classes
        
        # 先验网络 (预测下一步 z)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 后验网络 (根据观测更新 z)
        # 输入: deter + action + [stoch] + obs_embed
        # stoch 维度可能没有，所以用 larger input
        posterior_input_dim = deter_dim + action_dim + stoch_dim * num_classes + 256  # 256 for obs_embed
        self.posterior_net = nn.Sequential(
            nn.Linear(posterior_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 递归网络 (更新 deterministic 状态 h)
        self.recurrent_net = nn.GRUCell(
            input_size=stoch_dim * num_classes + action_dim,
            hidden_size=deter_dim
        )
        
        # z 的分布参数 (先验和后验共用)
        self.fc_mean = nn.Linear(hidden_dim, stoch_dim * num_classes)
        self.fc_std = nn.Linear(hidden_dim, stoch_dim * num_classes)
        
    def get_stoch_from_logits(self, logits: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """从 logits 采样随机状态 z
        
        使用 Gumbel-Softmax 采样
        """
        # logits: (B, stoch_dim * num_classes)
        # 转换为 (B, stoch_dim, num_classes)
        B = logits.shape[0]
        logits = logits.view(B, self.stoch_dim, self.num_classes)
        
        # Gumbel-Softmax 采样
        # gumbel = -log(-log(uniform + eps))
        gumbel = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(gumbel + 1e-8) + 1e-8)
        
        # logit + gumbel，然后 softmax
        logits_with_gumbel = (logits + gumbel) / temperature
        probs = F.softmax(logits_with_gumbel, dim=-1)
        
        # 转换为 embedding
        # 使用概率加权求和而不是 one-hot
        embed = torch.eye(self.num_classes, device=logits.device)  # (num_classes, num_classes)
        z = torch.matmul(probs, embed)  # (B, stoch_dim, num_classes)
        z = z.flatten(start_dim=1)  # (B, stoch_dim * num_classes)
        
        return z
    
    def forward(
        self,
        deter: torch.Tensor,
        prev_action: torch.Tensor,
        obs_embed: Optional[torch.Tensor] = None,
        prev_stoch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            deter: (B, deter_dim) 上一步 deterministic 状态
            prev_action: (B, action_dim) 上一步动作
            obs_embed: (B, embed_dim) 观测嵌入 (用于后验)
            prev_stoch: (B, stoch_dim * num_classes) 上一步随机状态
            
        Returns:
            deter: (B, deter_dim) 更新后的 deterministic 状态
            stoch_logits: (B, stoch_dim * num_classes) z 的分布 logits
            stoch: (B, stoch_dim * num_classes) 采样或后验 z
        """
        # 更新 deterministic 状态 h
        if prev_stoch is not None:
            rnn_input = torch.cat([prev_stoch, prev_action], dim=-1)
        else:
            rnn_input = torch.cat([
                torch.zeros_like(deter[:, :self.stoch_dim * self.num_classes]),
                prev_action
            ], dim=-1)
            
        deter = self.recurrent_net(rnn_input, deter)
        
        # 先验分布
        prior_input = torch.cat([deter, prev_action], dim=-1)
        prior_hidden = self.prior_net(prior_input)
        prior_logits = self.fc_mean(prior_hidden)
        
        # 后验分布 (如果有观测)
        if obs_embed is not None:
            # 后验: q(z_t | h_{t-1}, z_{t-1}, a_{t-1}, o_t)
            if prev_stoch is not None:
                posterior_input = torch.cat([deter, prev_action, prev_stoch, obs_embed], dim=-1)
            else:
                posterior_input = torch.cat([deter, prev_action, obs_embed], dim=-1)
            posterior_hidden = self.posterior_net(posterior_input)
            logits = self.fc_mean(posterior_hidden)
        else:
            logits = prior_logits
            
        # 采样 z
        stoch = self.get_stoch_from_logits(logits)
        
        return deter, logits, stoch


class RewardModel(nn.Module):
    """奖励模型"""
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """预测奖励"""
        return self.net(latent)


class ValueModel(nn.Module):
    """价值模型"""
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """预测状态价值"""
        return self.net(latent)


class ActorModel(nn.Module):
    """
    动作策略模型
    
    基于隐状态输出动作
    """
    
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        action_noise: float = 0.3,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_noise = action_noise
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )
        
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(
        self, 
        latent: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent: (B, latent_dim)
            deterministic: 是否使用确定性策略
            
        Returns:
            action: (B, action_dim)
            log_std: (B, action_dim)
        """
        features = self.net(latent)
        mean = self.fc_mean(features)
        
        if deterministic:
            return torch.tanh(mean), None
            
        log_std = self.fc_std(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        # 重参数化采样
        dist = Normal(mean, std)
        action_raw = dist.rsample()
        action = torch.tanh(action_raw)
        
        # 修正 log_prob (tanh 变换)
        log_std -= torch.log(1 + torch.tanh(action_raw).pow(2) + 1e-6)
        
        return action, log_std
    
    def act(self, latent: torch.Tensor) -> np.ndarray:
        """贪心策略 (用于推理)"""
        with torch.no_grad():
            action, _ = self.forward(latent, deterministic=True)
            action = action.cpu().numpy()
            # 添加噪声
            noise = np.random.randn(*action.shape) * self.action_noise
            action = np.clip(action + noise, -1, 1)
        return action


class WorldModel(nn.Module):
    """
    完整世界模型
    
    整合:
    - 观测编码器
    - RSSM 转移模型
    - 观测解码器
    - 奖励模型
    - 价值模型
    - 动作策略
    """
    
    def __init__(
        self,
        obs_dims: Dict[str, int],
        action_dim: int,
        config: Optional[WorldModelConfig] = None
    ):
        super().__init__()
        
        self.config = config or WorldModelConfig(action_dim=action_dim)
        self.cfg = self.config
        
        self.obs_dims = obs_dims
        self.action_dim = action_dim
        
        # 观测编码器
        embed_dim = self.cfg.obs_encoder_dim
        self.obs_encoder = ObservationEncoder(
            obs_dims, hidden_dim=embed_dim, latent_dim=embed_dim
        )
        
        # RSSM 转移模型
        self.transition = TransitionModel(
            action_dim=action_dim,
            deter_dim=self.cfg.deter_dim,
            stoch_dim=self.cfg.stoch_dim,
            num_classes=self.cfg.num_classes,
            hidden_dim=self.cfg.hidden_dim
        )
        
        # 观测解码器
        self.obs_decoder = ObservationDecoder(
            latent_dim=self.cfg.latent_dim,
            obs_dims=obs_dims,
            hidden_dim=self.cfg.hidden_dim
        )
        
        # 奖励模型
        self.reward_model = RewardModel(
            latent_dim=self.cfg.latent_dim,
            hidden_dim=self.cfg.hidden_dim
        )
        
        # 继续概率模型 (判断 episode 是否继续)
        self.pcont_model = nn.Sequential(
            nn.Linear(self.cfg.latent_dim, self.cfg.hidden_dim),
            nn.ELU(),
            nn.Linear(self.cfg.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 价值模型
        self.value_model = ValueModel(
            latent_dim=self.cfg.latent_dim,
            hidden_dim=self.cfg.hidden_dim
        )
        
        # 动作策略
        self.actor = ActorModel(
            latent_dim=self.cfg.latent_dim,
            action_dim=action_dim,
            hidden_dim=self.cfg.hidden_dim,
            action_noise=self.cfg.action_noise
        )
        
        # RSSM 状态到 latent 的投影层
        rssm_state_dim = self.cfg.deter_dim + self.cfg.stoch_dim * self.cfg.num_classes
        self.latent_proj = nn.Sequential(
            nn.Linear(rssm_state_dim, self.cfg.hidden_dim),
            nn.ELU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.latent_dim)
        )
        
        # 隐状态维度
        self.latent_dim = self.cfg.latent_dim
        self.rssm_state_dim = rssm_state_dim
        
    def get_latent(self, deter: torch.Tensor, stoch: torch.Tensor) -> torch.Tensor:
        """从 deter 和 stoch 构建隐状态"""
        rssm_state = torch.cat([deter, stoch], dim=-1)
        return self.latent_proj(rssm_state)
        """从 deter 和 stoch 构建完整隐状态"""
        return torch.cat([deter, stoch], dim=-1)
        
    def encode(
        self,
        observations: Dict[str, torch.Tensor],
        deter: torch.Tensor,
        prev_action: torch.Tensor,
        prev_stoch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """编码观测到隐状态"""
        # 观测编码
        obs_embed = self.obs_encoder(observations)
        
        # RSSM 前向
        deter, logits, stoch = self.transition(
            deter, prev_action, obs_embed, prev_stoch
        )
        
        return deter, logits, stoch
    
    def imagine(
        self,
        initial_state: ModelState,
        horizon: int,
        actor: Optional[ActorModel] = None
    ) -> Tuple[List[ModelState], torch.Tensor, torch.Tensor]:
        """
        想象 rollout
        
        Args:
            initial_state: 初始状态
            horizon: 想象步数
            actor: 动作策略 (如果为 None，使用内部 actor)
            
        Returns:
            states: 状态列表
            rewards: 奖励列表
            values: 价值列表
        """
        if actor is None:
            actor = self.actor
            
        states = [initial_state]
        rewards = []
        values = []
        
        deter = initial_state.deter
        stoch = initial_state.stoch
        action = initial_state.action
        
        for t in range(horizon):
            # 预测下一步 (无观测，使用先验)
            deter, prior_logits, stoch = self.transition(
                deter, action, obs_embed=None, prev_stoch=stoch
            )
            
            # 获取隐状态
            latent = self.get_latent(deter, stoch)
            
            # 预测奖励和价值
            reward = self.reward_model(latent)
            value = self.value_model(latent)
            
            # 策略输出动作
            action, _ = actor(latent)
            
            states.append(ModelState(deter=deter, stoch=stoch, action=action))
            rewards.append(reward)
            values.append(value)
            
        return states, torch.stack(rewards), torch.stack(values)
    
    def compute_loss(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        initial_deter: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算世界模型损失
        
        Args:
            observations: 观测字典
            actions: 动作序列 (T, B, action_dim)
            rewards: 奖励序列 (T, B)
            dones:终止标志 (T, B)
            initial_deter: 初始 deterministic 状态
            
        Returns:
            losses: 损失字典
        """
        # 处理维度顺序：期望 (T, B, action_dim)，但可能收到 (B, T, action_dim) 或 (1, B, T, action_dim)
        if actions.dim() == 4:
            # (1, B, T, action_dim) -> (T, B, action_dim)
            actions = actions.squeeze(0).transpose(0, 1)
        elif actions.dim() == 3 and actions.shape[0] != actions.shape[1]:
            # (B, T, action_dim) -> (T, B, action_dim)
            # 检测是否是 (B, T, ...) 而不是 (T, B, ...)
            if actions.shape[0] < actions.shape[1]:
                actions = actions.transpose(0, 1)
        
        T, B = actions.shape[:2]
        
        # 初始化 hidden states
        if initial_deter is None:
            initial_deter = torch.zeros(B, self.cfg.deter_dim, device=actions.device)
        deter = initial_deter
        
        # 初始化 stoch
        stoch = torch.zeros(
            B, self.cfg.stoch_dim * self.cfg.num_classes,
            device=actions.device
        )
        
        # KL 损失 (使用 tensor)
        # 获取device：从vision或第一个可用的observation，或从actions
        if 'vision' in observations:
            device = observations['vision'].device
        elif observations:
            device = next(iter(observations.values())).device
        else:
            device = actions.device
        
        # 处理 observations 维度顺序：期望 (T, B, obs_dim)，但可能收到 (B, T, obs_dim)
        for k in list(observations.keys()):
            v = observations[k]
            if v.dim() == 3:
                # 检测是否是 (B, T, obs_dim) 而不是 (T, B, obs_dim)
                if v.shape[0] < v.shape[1]:
                    # (B, T, obs_dim) -> (T, B, obs_dim)
                    observations[k] = v.transpose(0, 1)
        
        # 处理 rewards 和 dones 维度顺序：期望 (T, B) 或 (T,)，但可能收到 (B, T)
        if rewards.dim() == 2 and rewards.shape[0] < rewards.shape[1]:
            # (B, T) -> (T, B)
            rewards = rewards.transpose(0, 1)
        if dones.dim() == 2 and dones.shape[0] < dones.shape[1]:
            # (B, T) -> (T, B)
            dones = dones.transpose(0, 1)
        kl_loss = torch.tensor(0.0, device=device)
        reward_loss = torch.tensor(0.0, device=device)
        decoder_loss = torch.tensor(0.0, device=device)
        
        for t in range(T):
            # 当前观测
            obs_t = {k: v[t] for k, v in observations.items()}
            action_t = actions[t]
            
            # 前一步动作 (如果是第一步，用零动作)
            prev_action = actions[t-1] if t > 0 else torch.zeros_like(action_t)
            
            # RSSM 前向
            deter, logits, stoch = self.transition(
                deter, prev_action, None, stoch
            )
            
            # 隐状态
            latent = self.get_latent(deter, stoch)
            
            # 奖励损失
            if t < len(rewards):
                reward_pred = self.reward_model(latent)
                reward_loss = reward_loss + F.mse_loss(reward_pred.squeeze(), rewards[t])
            
            # 先验和后验的 KL 散度
            # (这里简化处理，实际应该用 model_specific_kl)
            
            # 观测解码损失
            obs_pred = self.obs_decoder(latent)
            for name in obs_t:
                if name in obs_pred:
                    decoder_loss = decoder_loss + F.mse_loss(obs_pred[name], obs_t[name])
        
        # 平均
        num_steps = min(T, len(rewards)) if isinstance(rewards, torch.Tensor) else T
        reward_loss = reward_loss / max(num_steps, 1)
        decoder_loss = decoder_loss / max(T, 1)
        
        # 总损失
        total_loss = (
            kl_loss * self.cfg.kl_weight +
            reward_loss * self.cfg.reward_weight +
            decoder_loss * self.cfg.decoder_weight
        )
        
        return {
            'total': total_loss,
            'kl': kl_loss,
            'reward': reward_loss,
            'decoder': decoder_loss
        }
    
    def update(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """训练一步"""
        self.train()
        optimizer.zero_grad()
        
        # 观测字典 (从 observations 子字典获取)
        obs_dict = batch['observations']
        if isinstance(obs_dict, dict):
            observations = obs_dict
        else:
            observations = {
                'vision': batch['vision'],
                'audio': batch['audio'],
                'tactile': batch['tactile'],
                'force': batch['force'],
                'imu': batch['imu']
            }
        
        actions = batch['actions']  # (T, B, action_dim)
        rewards = batch['rewards']  # (T, B)
        dones = batch['dones']  # (T, B)
        
        losses = self.compute_loss(
            observations, actions, rewards, dones
        )
        
        losses['total'].backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
        
        optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}


class WorldModelAgent:
    """
    基于世界模型的智能体
    
    使用 Dreamer 风格的学习:
    1. 学习世界模型
    2. 从世界模型中想象轨迹
    3. 用想象轨迹更新策略和价值函数
    """
    
    def __init__(
        self,
        obs_dims: Dict[str, int],
        action_dim: int,
        config: Optional[WorldModelConfig] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.config = config or WorldModelConfig(action_dim=action_dim)
        
        # 创建世界模型
        self.world_model = WorldModel(obs_dims, action_dim, self.config)
        self.world_model.to(device)
        
        # 优化器
        self.world_model_optimizer = torch.optim.Adam(
            self.world_model.parameters(),
            lr=1e-4,
            weight_decay=1e-6
        )
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
        # 训练步数
        self.train_steps = 0
        
    def select_action(
        self,
        observations: Dict[str, np.ndarray],
        deterministic: bool = False
    ) -> np.ndarray:
        """选择动作"""
        self.world_model.eval()
        
        with torch.no_grad():
            # 编码观测
            obs_tensors = {
                k: torch.from_numpy(v).float().to(self.device).unsqueeze(0)
                for k, v in observations.items()
            }
            
            # 简单处理：直接用观测初始化 deter
            # (实际需要维护隐状态)
            deter = torch.zeros(1, self.config.deter_dim, device=self.device)
            stoch = torch.zeros(
                1, self.config.stoch_dim * self.config.num_classes,
                device=self.device
            )
            action = torch.zeros(1, self.world_model.action_dim, device=self.device)
            
            # 编码
            deter, _, stoch = self.world_model.encode(
                obs_tensors, deter, action, stoch
            )
            
            # 策略
            latent = self.world_model.get_latent(deter, stoch)
            action, _ = self.world_model.actor(latent, deterministic=deterministic)
            action = action.cpu().numpy().squeeze()
            
        return action
    
    def store_transition(
        self,
        observations: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_observations: Dict[str, np.ndarray],
        done: bool
    ):
        """存储经验"""
        self.replay_buffer.push(
            observations, action, reward, next_observations, done
        )
        
    def train_step(self, batch_size: int = 64) -> Dict[str, float]:
        """训练一步"""
        if len(self.replay_buffer) < batch_size:
            return {}
            
        self.world_model.train()
        
        # 采样 batch
        batch = self.replay_buffer.sample(batch_size, self.device)
        
        # 更新世界模型
        losses = self.world_model.update(batch, self.world_model_optimizer)
        
        # 如果有想象的奖励和价值估计，也可以更新 actor 和 critic
        # (Dreamer 的 actor-critic 学习)
        
        self.train_steps += 1
        
        return losses
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'world_model': self.world_model.state_dict(),
            'config': self.config,
            'train_steps': self.train_steps
        }, path)
        
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.world_model.load_state_dict(checkpoint['world_model'])
        self.train_steps = checkpoint.get('train_steps', 0)


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(
        self,
        observations: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_observations: Dict[str, np.ndarray],
        done: bool
    ):
        """添加经验"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            
        self.buffer[self.position] = (
            observations, action, reward, next_observations, done
        )
        self.position = (self.position + 1) % self.capacity
        
    def sample(
        self, 
        batch_size: int, 
        device: str
    ) -> Dict[str, torch.Tensor]:
        """采样 batch"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        # 收集所有样本
        obs_list = {k: [] for k in self.buffer[0][0].keys()}
        actions_list = []
        rewards_list = []
        next_obs_list = {k: [] for k in self.buffer[0][3].keys()}
        dones_list = []
        
        for idx in indices:
            obs, action, reward, next_obs, done = self.buffer[idx]
            for k, v in obs.items():
                obs_list[k].append(v)
            actions_list.append(action)
            rewards_list.append(reward)
            for k, v in next_obs.items():
                next_obs_list[k].append(v)
            dones_list.append(done)
            
        # 转换为 tensor
        batch = {
            'observations': {k: torch.from_numpy(np.stack(v)).float().to(device) 
                           for k, v in obs_list.items()},
            'actions': torch.from_numpy(np.stack(actions_list)).float().to(device),
            'rewards': torch.from_numpy(np.stack(rewards_list)).float().to(device),
            'next_observations': {k: torch.from_numpy(np.stack(v)).float().to(device) 
                              for k, v in next_obs_list.items()},
            'dones': torch.from_numpy(np.stack(dones_list)).float().to(device)
        }
        
        return batch
        
    def __len__(self):
        return len(self.buffer)


# AGV 五级世界模型配置
WORLD_MODEL_GRADES = {
    'S': WorldModelConfig(
        latent_dim=128, hidden_dim=256, rnn_hidden_dim=128,
        obs_encoder_dim=128, deter_dim=128, stoch_dim=16, num_classes=32,
        imagination_horizon=10
    ),
    'M': WorldModelConfig(
        latent_dim=256, hidden_dim=512, rnn_hidden_dim=256,
        obs_encoder_dim=256, deter_dim=256, stoch_dim=32, num_classes=32,
        imagination_horizon=15
    ),
    'L': WorldModelConfig(
        latent_dim=512, hidden_dim=1024, rnn_hidden_dim=512,
        obs_encoder_dim=512, deter_dim=512, stoch_dim=64, num_classes=32,
        imagination_horizon=20
    ),
    'XL': WorldModelConfig(
        latent_dim=768, hidden_dim=1536, rnn_hidden_dim=768,
        obs_encoder_dim=768, deter_dim=768, stoch_dim=64, num_classes=64,
        imagination_horizon=25
    ),
    'XXL': WorldModelConfig(
        latent_dim=1024, hidden_dim=2048, rnn_hidden_dim=1024,
        obs_encoder_dim=1024, deter_dim=1024, stoch_dim=128, num_classes=64,
        imagination_horizon=30
    )
}


def create_world_model_agent(
    grade: str,
    obs_dims: Dict[str, int],
    action_dim: int,
    device: Optional[str] = None
) -> WorldModelAgent:
    """为指定 AGV 等级创建世界模型智能体"""
    config = WORLD_MODEL_GRADES.get(grade, WORLD_MODEL_GRADES['M'])
    config.action_dim = action_dim
    
    return WorldModelAgent(
        obs_dims=obs_dims,
        action_dim=action_dim,
        config=config,
        device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    )


def get_world_model_spec(grade: str) -> WorldModelConfig:
    """获取指定等级的世界模型配置"""
    return WORLD_MODEL_GRADES.get(grade, WORLD_MODEL_GRADES['M'])
