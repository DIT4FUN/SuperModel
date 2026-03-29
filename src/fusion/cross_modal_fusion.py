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
    language_dim: int = 128
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    strategy: FusionStrategy = FusionStrategy.HYBRID
    vocab_size: int = 10000
    language_max_len: int = 32


class LanguageEncoder(nn.Module):
    """
    语言编码器
    
    将 token 序列编码为特征向量，用于多模态融合
    支持: 词嵌入 + 位置编码 + Transformer 编码
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        max_len: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # 词嵌入 + 位置编码
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        
        # 投影到 hidden_dim
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        
        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 池化: 使用 [CLS] token 或 mean pooling
        self.use_cls = True
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: B x L (token indices)
            
        Returns:
            features: B x hidden_dim
        """
        B, L = token_ids.shape
        
        # 词嵌入
        x = self.token_embedding(token_ids)  # B x L x embed_dim
        
        # 位置编码
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.position_embedding(positions)
        x = x + pos_emb
        
        # 投影 + dropout
        x = self.input_proj(x)
        x = self.dropout(x)
        
        # 添加 [CLS] token
        if self.use_cls:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # B x (L+1) x hidden_dim
        
        # Transformer 编码
        x = self.transformer(x)  # B x (L+1) x hidden_dim
        
        if self.use_cls:
            # 取 [CLS] token
            return x[:, 0]  # B x hidden_dim
        else:
            # Mean pooling
            return x.mean(dim=1)  # B x hidden_dim


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
            # mask: B x N x M -> B x 1 x N x M for broadcast to B x H x N x M
            mask = mask.unsqueeze(1)
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
        self.language_encoder = LanguageEncoder(
            vocab_size=config.vocab_size,
            embed_dim=config.language_dim,
            hidden_dim=config.hidden_dim,
            max_len=config.language_max_len
        )
        
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
        
        # 融合层 (预建不同模态数量的投影，运行时选取)
        self.fusion_proj_1 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.fusion_proj_2 = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.fusion_proj_3 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.fusion_proj_4 = nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        self.fusion_proj_5 = nn.Linear(config.hidden_dim * 5, config.hidden_dim)
        self.fusion_proj_6 = nn.Linear(config.hidden_dim * 6, config.hidden_dim)
        
    def _get_active_mods(self) -> List[str]:
        return ['vision', 'audio', 'tactile', 'force', 'imu', 'language']
    
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
            
        if multimodal.language is not None:
            features['language'] = self.language_encoder(multimodal.language)
        
        if not features:
            raise ValueError("No modalities available")
        
        # 跨模态交互
        # 确保特征是3D: B x N x D (N=1 for single token features)
        def ensure_3d(t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 2:
                return t.unsqueeze(1)
            return t
        
        def squeeze_2d(t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 3 and t.shape[1] == 1:
                return t.squeeze(1)
            return t
        
        if 'vision' in features and 'audio' in features:
            q = ensure_3d(features['vision'])
            k = ensure_3d(features['audio'])
            v = ensure_3d(features['audio'])
            out = self.vision_audio_attn(q, k, v)
            features['vision'] = features['vision'] + squeeze_2d(out)
            
        if 'vision' in features and 'tactile' in features:
            q = ensure_3d(features['vision'])
            k = ensure_3d(features['tactile'])
            v = ensure_3d(features['tactile'])
            out = self.vision_tactile_attn(q, k, v)
            features['vision'] = features['vision'] + squeeze_2d(out)
            
        if 'audio' in features and 'tactile' in features:
            q = ensure_3d(features['audio'])
            k = ensure_3d(features['tactile'])
            v = ensure_3d(features['tactile'])
            out = self.audio_tactile_attn(q, k, v)
            features['audio'] = features['audio'] + squeeze_2d(out)
        
        # 视觉-语言跨模态注意力 (关键对齐)
        if 'vision' in features and 'language' in features:
            q = ensure_3d(features['vision'])
            k = ensure_3d(features['language'])
            v = ensure_3d(features['language'])
            out = self.vision_tactile_attn(q, k, v)  # 复用已有的视觉-触觉注意力层
            features['vision'] = features['vision'] + squeeze_2d(out)
        
        # 串联融合 (根据实际模态数量选择投影)
        num_modalities = len(features)
        fused = torch.cat(list(features.values()), dim=-1)
        proj_map = {
            1: self.fusion_proj_1,
            2: self.fusion_proj_2,
            3: self.fusion_proj_3,
            4: self.fusion_proj_4,
            5: self.fusion_proj_5,
            6: self.fusion_proj_6,
        }
        fused = proj_map[num_modalities](fused)
        
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


# AGV五级融合规格
AGV_FUSION_GRADES = {
    'S': {
        'strategy': 'late', 'hidden_dim': 128, 'num_heads': 2,
        'num_layers': 1, 'output_dim': 64, 'latency_ms': 50,
        'dropout': 0.1, 'attention_dropout': 0.1,
    },
    'M': {
        'strategy': 'middle', 'hidden_dim': 256, 'num_heads': 4,
        'num_layers': 2, 'output_dim': 128, 'latency_ms': 20,
        'dropout': 0.1, 'attention_dropout': 0.1,
    },
    'L': {
        'strategy': 'middle', 'hidden_dim': 512, 'num_heads': 8,
        'num_layers': 4, 'output_dim': 256, 'latency_ms': 10,
        'dropout': 0.1, 'attention_dropout': 0.1,
    },
    'XL': {
        'strategy': 'hybrid', 'hidden_dim': 768, 'num_heads': 12,
        'num_layers': 6, 'output_dim': 512, 'latency_ms': 5,
        'dropout': 0.1, 'attention_dropout': 0.1,
    },
    'XXL': {
        'strategy': 'hybrid', 'hidden_dim': 1024, 'num_heads': 16,
        'num_layers': 8, 'output_dim': 1024, 'latency_ms': 2,
        'dropout': 0.1, 'attention_dropout': 0.1,
    },
}


def get_fusion_spec(grade: str) -> dict:
    """获取指定AGV等级的融合规格"""
    return AGV_FUSION_GRADES.get(grade, AGV_FUSION_GRADES['M'])
