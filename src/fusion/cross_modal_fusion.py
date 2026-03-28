"""
跨模态融合模块
==============

实现多模态感知数据的融合
- 特征级融合
- 注意力机制
- 统一表示学习
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class FusionStrategy(Enum):
    """融合策略"""
    EARLY = "early"      # 早期融合 (原始特征)
    LATE = "late"        # 晚期融合 (决策级)
    HYBRID = "hybrid"    # 混合融合


@dataclass
class MultimodalInput:
    """多模态输入"""
    vision: Optional[torch.Tensor] = None       # B x C x H x W
    audio: Optional[torch.Tensor] = None       # B x T x F
    tactile: Optional[torch.Tensor] = None     # B x N (触点)
    force: Optional[torch.Tensor] = None       # B x 6 (Fxyz, Txyz)
    imu: Optional[torch.Tensor] = None         # B x 9 (acc, gyro, mag)
    language: Optional[torch.Tensor] = None    # B x L (token ids)
    
    @property
    def modalities(self) -> List[str]:
        """返回可用模态列表"""
        mods = []
        if self.vision is not None: mods.append('vision')
        if self.audio is not None: mods.append('audio')
        if self.tactile is not None: mods.append('tactile')
        if self.force is not None: mods.append('force')
        if self.imu is not None: mods.append('imu')
        if self.language is not None: mods.append('language')
        return mods


@dataclass
class FusionConfig:
    """融合配置"""
    vision_dim: int = 512
    audio_dim: int = 128
    tactile_dim: int = 64
    force_dim: int = 32
    imu_dim: int = 64
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    strategy: FusionStrategy = FusionStrategy.HYBRID


class CrossModalAttention(nn.Module):
    """
    跨模态注意力
    
    实现不同模态之间的注意力交互
    """
    
    def __init__(self, query_dim: int, key_dim: int, value_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        
        self.query_proj = nn.Linear(query_dim, query_dim)
        self.key_proj = nn.Linear(key_dim, query_dim)
        self.value_proj = nn.Linear(value_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: B x N x Dq
            key: B x M x Dk
            value: B x M x Dv
            mask: B x N x M (可选)
            
        Returns:
            output: B x N x Dq
        """
        B, N, _ = query.shape
        _, M, _ = key.shape
        
        # 投影
        Q = self.query_proj(query)  # B x N x D
        K = self.key_proj(key)      # B x M x D
        V = self.value_proj(value)  # B x M x D
        
        # 分头
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # B x H x N x d
        K = K.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # B x H x M x d
        V = V.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # B x H x M x d
        
        # 注意力
        attn = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # B x H x N x M
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
            
        attn = torch.softmax(attn, dim=-1)
        
        # 输出
        out = attn @ V  # B x H x N x d
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.out_proj(out)
        
        return out


class ModalityEncoder(nn.Module):
    """
    单模态编码器
    
    将原始感知数据编码为特征向量
    """
    
    def __init__(self, modality: str, input_dim: int, output_dim: int):
        super().__init__()
        self.modality = modality
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class CrossModalFusion(nn.Module):
    """
    跨模态融合网络
    
    实现:
    - 多模态特征提取
    - 跨模态注意力交互
    - 统一表示生成
    """
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        
        # 模态特定编码器
        self.vision_encoder = ModalityEncoder('vision', 512, config.hidden_dim)
        self.audio_encoder = ModalityEncoder('audio', 128, config.hidden_dim)
        self.tactile_encoder = ModalityEncoder('tactile', 64, config.hidden_dim)
        self.force_encoder = ModalityEncoder('force', 32, config.hidden_dim)
        self.imu_encoder = ModalityEncoder('imu', 64, config.hidden_dim)
        
        # 跨模态注意力层
        self.cross_attn_layers = nn.ModuleList([
            CrossModalAttention(
                config.hidden_dim, config.hidden_dim, config.hidden_dim,
                num_heads=config.num_heads
            )
            for _ in range(config.num_layers)
        ])
        
        # 模态间注意力 (每次两两交互)
        self.vision_audio_attn = CrossModalAttention(
            config.hidden_dim, config.hidden_dim, config.hidden_dim, num_heads=config.num_heads
        )
        self.vision_tactile_attn = CrossModalAttention(
            config.hidden_dim, config.hidden_dim, config.hidden_dim, num_heads=config.num_heads
        )
        self.audio_tactile_attn = CrossModalAttention(
            config.hidden_dim, config.hidden_dim, config.hidden_dim, num_heads=config.num_heads
        )
        
        # 融合层
        self.fusion_proj = nn.Sequential(
            nn.Linear(config.hidden_dim * len(self._get_active_mods()), config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim)
        )
        
    def _get_active_mods(self) -> List[str]:
        return ['vision', 'audio', 'tactile', 'force', 'imu']
    
    def forward(self, multimodal: MultimodalInput) -> torch.Tensor:
        """
        前向传播
        
        Args:
            multimodal: MultimodalInput, 包含各模态数据
            
        Returns:
            fused: B x hidden_dim, 融合后的统一表示
        """
        features = {}
        
        # 编码各模态
        if multimodal.vision is not None:
            # 假设vision已经是(B, hidden_dim)的特征
            # 实际应用中会有更复杂的视觉编码器
            features['vision'] = self.vision_encoder(multimodal.vision)
            
        if multimodal.audio is not None:
            features['audio'] = self.audio_encoder(multimodal.audio)
            
        if multimodal.tactile is not None:
            features['tactile'] = self.tactile_encoder(multimodal.tactile)
            
        if multimodal.force is not None:
            features['force'] = self.force_encoder(multimodal.force)
            
        if multimodal.imu is not None:
            features['imu'] = self.imu_encoder(multimodal.imu)
        
        if not features:
            raise ValueError("No modalities available")
        
        # 跨模态交互
        if 'vision' in features and 'audio' in features:
            features['vision'] = features['vision'] + self.vision_audio_attn(
                features['vision'], features['audio'], features['audio']
            )
            
        if 'vision' in features and 'tactile' in features:
            features['vision'] = features['vision'] + self.vision_tactile_attn(
                features['vision'], features['tactile'], features['tactile']
            )
            
        if 'audio' in features and 'tactile' in features:
            features['audio'] = features['audio'] + self.audio_tactile_attn(
                features['audio'], features['tactile'], features['tactile']
            )
        
        # 串联融合
        fused = torch.cat(list(features.values()), dim=-1)
        fused = self.fusion_proj(fused)
        
        return fused


class UnifiedRepresentation(nn.Module):
    """
    统一表示学习
    
    将融合特征转换为任务无关的统一表示
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 任务头
        self.state_head = nn.Linear(hidden_dim, output_dim)      # 状态表示
        self.action_head = nn.Linear(hidden_dim, output_dim)      # 动作策略
        self.world_head = nn.Linear(hidden_dim, output_dim)       # 世界模型预测
        
    def forward(
        self, 
        fused: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            fused: B x input_dim, 融合特征
            
        Returns:
            state: B x output_dim, 状态表示
            action: B x output_dim, 动作策略
            world: B x output_dim, 世界模型预测
        """
        h = self.encoder(fused)
        return self.state_head(h), self.action_head(h), self.world_head(h)


# 工具函数
def create_multimodal_input(
    vision: Optional[np.ndarray] = None,
    audio: Optional[np.ndarray] = None,
    tactile: Optional[np.ndarray] = None,
    force: Optional[np.ndarray] = None,
    imu: Optional[np.ndarray] = None,
    language: Optional[np.ndarray] = None
) -> MultimodalInput:
    """创建多模态输入"""
    return MultimodalInput(
        vision=torch.from_numpy(vision).float() if vision is not None else None,
        audio=torch.from_numpy(audio).float() if audio is not None else None,
        tactile=torch.from_numpy(tactile).float() if tactile is not None else None,
        force=torch.from_numpy(force).float() if force is not None else None,
        imu=torch.from_numpy(imu).float() if imu is not None else None,
        language=torch.from_numpy(language).long() if language is not None else None
    )
