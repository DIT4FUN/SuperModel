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
Sensor Encoders - 传感器编码器
=============================

将原始传感器数据编码为特征向量，供世界模型使用

支持:
1. VisionEncoder - 双目视觉编码
2. AudioEncoder - 双耳听觉编码
3. TactileEncoder - 触觉编码
4. ForceEncoder - 力觉编码
5. IMUEncoder - IMU 编码
6. MultiModalEncoder - 多模态融合编码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class EncoderConfig:
    """编码器配置"""
    vision_dim: int = 512
    audio_dim: int = 128
    tactile_dim: int = 64
    force_dim: int = 32
    imu_dim: int = 64
    
    hidden_dim: int = 256
    latent_dim: int = 256
    
    # Vision specific
    image_size: Tuple[int, int] = (224, 224)
    patch_size: int = 16
    
    # Audio specific
    audio_length: int = 16000  # 1 second at 16kHz
    n_mels: int = 64
    
    # Tactile specific
    tactile_size: Tuple[int, int] = (16, 16)
    
    # IMU specific
    imu_window: int = 10  # 时间窗口


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, dim: int, use_conv: bool = False):
        super().__init__()
        self.use_conv = use_conv
        
        if use_conv:
            # 用于 CNN 特征的残差块
            self.block = nn.Sequential(
                nn.BatchNorm2d(dim),
                nn.Conv2d(dim, dim * 4, 1),
                nn.GELU(),
                nn.Conv2d(dim * 4, dim, 1)
            )
        else:
            # 用于全连接层的残差块
            self.block = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            return x + 0.1 * self.block(x)
        else:
            return x + 0.1 * self.block(x)


class VisionEncoder(nn.Module):
    """
    双目视觉编码器
    
    将双目图像对编码为特征向量
    支持:
    - 简单 CNN 编码
    - 轻量级 ViT 编码
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        stereo: bool = True
    ):
        super().__init__()
        self.stereo = stereo
        
        # 共享特征提取
        self.shared = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            
            ResidualBlock(32, use_conv=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            ResidualBlock(64, use_conv=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            
            ResidualBlock(128, use_conv=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 双目融合 (如果启用)
        if stereo:
            # 左眼和右眼分别编码，然后融合
            self.left_encoder = nn.Linear(128, latent_dim)
            self.right_encoder = nn.Linear(128, latent_dim)
            self.fusion = nn.Sequential(
                nn.Linear(latent_dim * 2, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim)
            )
        else:
            self.encoder = nn.Linear(128, latent_dim)
            
    def forward(
        self, 
        left_image: torch.Tensor,
        right_image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            left_image: (B, C, H, W) 左眼图像
            right_image: (B, C, H, W) 右眼图像 (可选)
            
        Returns:
            features: (B, latent_dim)
        """
        # 提取共享特征
        left_features = self.shared(left_image).flatten(1)  # (B, 128)
        
        if self.stereo and right_image is not None:
            right_features = self.shared(right_image).flatten(1)
            left_encoded = self.left_encoder(left_features)
            right_encoded = self.right_encoder(right_features)
            combined = torch.cat([left_encoded, right_encoded], dim=-1)
            return self.fusion(combined)
        else:
            return self.left_encoder(left_features)


class AudioEncoder(nn.Module):
    """
    双耳听觉编码器
    
    将双耳音频编码为特征向量
    支持:
    - 梅尔频谱图 + CNN
    - 时序 LSTM/GRU
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        n_mels: int = 64,
        n_fft: int = 1024
    ):
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        
        # 梅尔频谱图提取 (简化为线性投影)
        # 双耳所以输入是 n_mels * 2
        self.mel_proj = nn.Linear(n_mels * 2, hidden_dim)
        
        # 时序建模
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, left_audio: torch.Tensor, right_audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            left_audio: (B, T, n_mels) 左耳音频特征
            right_audio: (B, T, n_mels) 右耳音频特征
            
        Returns:
            features: (B, latent_dim)
        """
        B, T, _ = left_audio.shape
        
        # 双耳融合
        stereo = torch.cat([left_audio, right_audio], dim=-1)  # (B, T, n_mels*2)
        
        # 投影 (输入是 n_mels*2)
        x = self.mel_proj(stereo)  # (B, T, hidden_dim)
        
        # 时序建模
        x, _ = self.rnn(x)  # (B, T, hidden_dim*2)
        
        # 池化 (时间维度)
        x = x.mean(dim=1)  # (B, hidden_dim*2)
        
        return self.output_proj(x)


class TactileEncoder(nn.Module):
    """
    触觉编码器
    
    将电子皮肤压力图编码为特征向量
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        array_size: Tuple[int, int] = (16, 16)
    ):
        super().__init__()
        self.array_size = array_size
        
        # CNN 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),  # 16 -> 8
            
            ResidualBlock(32, use_conv=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),  # 8 -> 4
            
            ResidualBlock(64, use_conv=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tactile: (B, 1, H, W) 压力图
            
        Returns:
            features: (B, latent_dim)
        """
        x = self.encoder(tactile).flatten(1)  # (B, 128)
        return self.output_proj(x)


class ForceEncoder(nn.Module):
    """
    力觉编码器
    
    将六维力矩数据编码为特征向量
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        window_size: int = 10
    ):
        super().__init__()
        self.window_size = window_size
        
        # 时序建模
        self.rnn = nn.GRU(
            input_size=6,  # Fx, Fy, Fz, Mx, My, Mz
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, wrench: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wrench: (B, T, 6) or (B, 6) 力矩序列
            
        Returns:
            features: (B, latent_dim)
        """
        if wrench.dim() == 2:
            wrench = wrench.unsqueeze(1)  # (B, 1, 6)
            
        x, hidden = self.rnn(wrench)  # x: (B, T, hidden_dim)
        
        # 使用最后一个时间步
        x = x[:, -1, :]  # (B, hidden_dim)
        
        return self.output_proj(x)


class IMUEncoder(nn.Module):
    """
    IMU 编码器
    
    将 IMU 数据 (加速度计 + 陀螺仪) 编码为特征向量
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        window_size: int = 10
    ):
        super().__init__()
        self.window_size = window_size
        
        # IMU 数据: accel(3) + gyro(3) = 6
        self.rnn = nn.GRU(
            input_size=6,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # 姿态估计辅助输出
        self.pose_head = nn.Linear(hidden_dim * 2, 4)  # quaternion
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, imu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            imu: (B, T, 6) or (B, 6) IMU 数据
            
        Returns:
            features: (B, latent_dim)
            quaternion: (B, 4) 姿态四元数
        """
        if imu.dim() == 2:
            imu = imu.unsqueeze(1)  # (B, 1, 6)
            
        x, _ = self.rnn(imu)  # (B, T, hidden_dim*2)
        
        # 池化
        pooled = x.mean(dim=1)  # (B, hidden_dim*2)
        
        # 姿态估计
        quat = self.pose_head(pooled)
        quat = F.normalize(quat, dim=-1)  # 归一化为四元数
        
        return self.output_proj(pooled), quat


class MultiModalEncoder(nn.Module):
    """
    多模态编码器
    
    将所有传感器数据编码为统一特征向量
    """
    
    def __init__(
        self,
        encoder_config: Optional[EncoderConfig] = None,
        latent_dim: int = 256
    ):
        super().__init__()
        self.config = encoder_config or EncoderConfig()
        self.latent_dim = latent_dim
        
        # 各模态编码器
        self.vision_encoder = VisionEncoder(
            latent_dim=latent_dim,
            hidden_dim=self.config.hidden_dim
        )
        
        self.audio_encoder = AudioEncoder(
            latent_dim=latent_dim,
            hidden_dim=self.config.hidden_dim,
            n_mels=self.config.n_mels
        )
        
        self.tactile_encoder = TactileEncoder(
            latent_dim=latent_dim,
            hidden_dim=self.config.hidden_dim,
            array_size=self.config.tactile_size
        )
        
        self.force_encoder = ForceEncoder(
            latent_dim=latent_dim,
            hidden_dim=self.config.hidden_dim
        )
        
        self.imu_encoder = IMUEncoder(
            latent_dim=latent_dim,
            hidden_dim=self.config.hidden_dim
        )
        
        # 模态融合注意力
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=4,
            batch_first=True
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim * 5, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
    def forward(
        self,
        vision_left: Optional[torch.Tensor] = None,
        vision_right: Optional[torch.Tensor] = None,
        audio_left: Optional[torch.Tensor] = None,
        audio_right: Optional[torch.Tensor] = None,
        tactile: Optional[torch.Tensor] = None,
        force: Optional[torch.Tensor] = None,
        imu: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            vision_left: (B, C, H, W) 左眼图像
            vision_right: (B, C, H, W) 右眼图像
            audio_left: (B, T, n_mels) 左耳音频
            audio_right: (B, T, n_mels) 右耳音频
            tactile: (B, 1, H, W) 触觉压力图
            force: (B, T, 6) or (B, 6) 力矩
            imu: (B, T, 6) or (B, 6) IMU 数据
            
        Returns:
            features: {modality: tensor} 各模态特征
            fused: (B, latent_dim) 融合特征
        """
        features = {}
        
        # Vision
        if vision_left is not None:
            features['vision'] = self.vision_encoder(vision_left, vision_right)
            
        # Audio
        if audio_left is not None and audio_right is not None:
            features['audio'] = self.audio_encoder(audio_left, audio_right)
            
        # Tactile
        if tactile is not None:
            features['tactile'] = self.tactile_encoder(tactile)
            
        # Force
        if force is not None:
            features['force'] = self.force_encoder(force)
            
        # IMU
        if imu is not None:
            imu_feat, quat = self.imu_encoder(imu)
            features['imu'] = imu_feat
            
        # 融合
        if len(features) > 1:
            # 使用注意力融合
            feature_list = list(features.values())
            features_tensor = torch.stack(feature_list, dim=1)  # (B, N, latent_dim)
            
            # 自注意力融合
            fused, _ = self.fusion_attention(
                features_tensor, features_tensor, features_tensor
            )
            fused = fused.mean(dim=1)  # (B, latent_dim)
        elif len(features) == 1:
            fused = list(features.values())[0]
        else:
            fused = torch.zeros(1, self.latent_dim)
            
        features['fused'] = fused
        return features


class SensorEncoderWrapper(nn.Module):
    """
    传感器编码器封装
    
    用于与 World Model 集成
    """
    
    def __init__(
        self,
        obs_dims: Dict[str, int],
        encoder_config: Optional[EncoderConfig] = None,
        latent_dim: int = 256
    ):
        super().__init__()
        self.obs_dims = obs_dims
        self.latent_dim = latent_dim
        
        # 多模态编码器
        self.encoder = MultiModalEncoder(
            encoder_config=encoder_config,
            latent_dim=latent_dim
        )
        
        # 独立编码器映射 (用于批量处理)
        self.encoders = nn.ModuleDict()
        if 'vision' in obs_dims:
            self.encoders['vision'] = VisionEncoder(latent_dim, hidden_dim=encoder_config.hidden_dim if encoder_config else 256)
        if 'audio' in obs_dims:
            self.encoders['audio'] = AudioEncoder(latent_dim, hidden_dim=encoder_config.hidden_dim if encoder_config else 256)
        if 'tactile' in obs_dims:
            self.encoders['tactile'] = TactileEncoder(latent_dim, hidden_dim=encoder_config.hidden_dim if encoder_config else 256)
        if 'force' in obs_dims:
            self.encoders['force'] = ForceEncoder(latent_dim, hidden_dim=encoder_config.hidden_dim if encoder_config else 256)
        if 'imu' in obs_dims:
            self.encoders['imu'] = IMUEncoder(latent_dim, hidden_dim=encoder_config.hidden_dim if encoder_config else 256)
            
    def forward(self, observations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            observations: {modality: tensor} 原始观测
            
        Returns:
            encoded: {modality: tensor} 编码后的特征
        """
        encoded = {}
        
        for modality, obs in observations.items():
            if modality in self.encoders:
                if modality == 'vision':
                    # vision 可能是单目或双目
                    if isinstance(obs, dict):
                        encoded[modality] = self.encoders[modality](
                            obs.get('left'), obs.get('right')
                        )
                    else:
                        encoded[modality] = self.encoders[modality](obs)
                elif modality == 'audio':
                    # audio 可能是单耳或双耳
                    if isinstance(obs, dict):
                        encoded[modality] = self.encoders[modality](
                            obs.get('left'), obs.get('right')
                        )
                    else:
                        encoded[modality] = self.encoders[modality](
                            obs, obs  # 如果单耳，复用
                        )
                elif modality == 'imu':
                    encoded[modality], _ = self.encoders[modality](obs)
                else:
                    encoded[modality] = self.encoders[modality](obs)
                    
        # 添加融合特征
        if len(encoded) > 0:
            encoded['fused'] = torch.stack(list(encoded.values())).mean(0)
                    
        return encoded


# AGV 五级编码器配置
ENCODER_GRADES = {
    'S': EncoderConfig(
        vision_dim=256, audio_dim=64, tactile_dim=32, force_dim=16, imu_dim=32,
        hidden_dim=128, latent_dim=128
    ),
    'M': EncoderConfig(
        vision_dim=512, audio_dim=128, tactile_dim=64, force_dim=32, imu_dim=64,
        hidden_dim=256, latent_dim=256
    ),
    'L': EncoderConfig(
        vision_dim=768, audio_dim=256, tactile_dim=128, force_dim=64, imu_dim=128,
        hidden_dim=512, latent_dim=512
    ),
    'XL': EncoderConfig(
        vision_dim=1024, audio_dim=512, tactile_dim=256, force_dim=128, imu_dim=256,
        hidden_dim=768, latent_dim=768
    ),
    'XXL': EncoderConfig(
        vision_dim=2048, audio_dim=1024, tactile_dim=512, force_dim=256, imu_dim=512,
        hidden_dim=1024, latent_dim=1024
    )
}


class LanguageEncoder(nn.Module):
    """
    语言编码器
    
    将文本 token 序列编码为特征向量，支持:
    - 词嵌入 + 位置编码
    - Transformer 编码
    - [CLS] token 池化
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
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
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
        x = self.token_embedding(token_ids)
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        x = x + self.position_embedding(positions)
        x = self.input_proj(x)
        x = self.dropout(x)
        
        if self.use_cls:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        x = self.transformer(x)
        
        if self.use_cls:
            return x[:, 0]
        else:
            return x.mean(dim=1)


def create_sensor_encoder(
    obs_dims: Dict[str, int],
    grade: str = 'M'
) -> SensorEncoderWrapper:
    """创建指定等级的传感器编码器"""
    config = ENCODER_GRADES.get(grade, ENCODER_GRADES['M'])
    return SensorEncoderWrapper(obs_dims, config, latent_dim=config.latent_dim)


def get_encoder_config(grade: str) -> EncoderConfig:
    """获取指定等级的编码器配置"""
    return ENCODER_GRADES.get(grade, ENCODER_GRADES['M'])
