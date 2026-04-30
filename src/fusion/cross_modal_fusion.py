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
跨模态融合网络 (Cross-Modal Fusion)
=====================================

实现多模态感知数据的深度融合
- 视觉 / 听觉 / 触觉 / 力觉 / IMU / 语言
- Early Fusion (特征拼接 + MLP)
- Late Fusion (各模态独立编码 + 注意力融合)
- Hybrid Fusion (混合架构)
- Transformer Cross-Attention 融合

支持 AGV 五级规格 (S/M/L/XL/XXL)
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json


@dataclass
class MultimodalInput:
    """
    多模态输入容器
    
    支持的模态:
    - vision: 视觉特征 (B, vision_dim)
    - audio: 听觉特征 (B, audio_dim)
    - tactile: 触觉特征 (B, tactile_dim)
    - force: 力觉特征 (B, force_dim)
    - imu: IMU特征 (B, imu_dim)
    - language: 语言特征 (B, lang_dim)
    """
    vision: Optional[np.ndarray] = None
    audio: Optional[np.ndarray] = None
    tactile: Optional[np.ndarray] = None
    force: Optional[np.ndarray] = None
    imu: Optional[np.ndarray] = None
    language: Optional[np.ndarray] = None
    
    def get_available_modalities(self) -> List[str]:
        """获取当前可用的模态列表"""
        mods = []
        if self.vision is not None: mods.append('vision')
        if self.audio is not None: mods.append('audio')
        if self.tactile is not None: mods.append('tactile')
        if self.force is not None: mods.append('force')
        if self.imu is not None: mods.append('imu')
        if self.language is not None: mods.append('language')
        return mods
    
    def get_shape(self, modality: str) -> Optional[Tuple[int, ...]]:
        """获取指定模态的形状"""
        m = getattr(self, modality, None)
        return tuple(m.shape) if m is not None else None
    
    def num_modalities(self) -> int:
        """可用模态数量"""
        return len(self.get_available_modalities())

    @property
    def modalities(self) -> List[str]:
        """可用模态列表 (属性形式, 与 get_available_modalities 等价)"""
        return self.get_available_modalities()


class FusionStrategy(Enum):
    """融合策略枚举"""
    EARLY = "early"
    LATE = "late"
    HYBRID = "hybrid"
    TRANSFORMER = "transformer"


@dataclass
class FusionConfig:
    """
    融合配置
    
    Args:
        vision_dim: 视觉特征维度 (默认512)
        audio_dim: 听觉特征维度 (默认128)
        tactile_dim: 触觉特征维度 (默认64)
        force_dim: 力觉特征维度 (默认32)
        imu_dim: IMU特征维度 (默认32)
        lang_dim: 语言特征维度 (默认768)
        hidden_dim: 融合隐藏层维度 (默认256)
        output_dim: 输出特征维度 (默认256)
        fusion_type: 融合类型 ("early" / "late" / "hybrid" / "transformer")
        num_heads: Transformer多头数量 (默认8)
        num_layers: Transformer层数 (默认2)
        dropout: Dropout比率 (默认0.1)
        use_batch_norm: 是否使用BatchNorm (默认True)
    """
    vision_dim: int = 512
    audio_dim: int = 128
    tactile_dim: int = 64
    force_dim: int = 32
    imu_dim: int = 32
    lang_dim: int = 768
    hidden_dim: int = 256
    output_dim: int = 256  # 默认与 hidden_dim 相同（可覆盖）
    fusion_type: str = "hybrid"
    strategy: Optional[FusionStrategy] = None
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.1
    use_batch_norm: bool = True

    def __post_init__(self):
        # strategy 参数优先于 fusion_type
        if self.strategy is not None:
            self.fusion_type = self.strategy.value
        # output_dim 默认为 hidden_dim（未显式设置时保持维度一致性）
        if self.hidden_dim != 256 and self.output_dim == 256:
            self.output_dim = self.hidden_dim
        valid_types = ["early", "late", "hybrid", "transformer"]
        if self.fusion_type not in valid_types:
            raise ValueError(f"fusion_type must be one of {valid_types}, got {self.fusion_type}")


class CrossModalFusion:
    """
    跨模态融合网络
    
    支持四种融合架构:
    1. Early Fusion: 特征拼接 → MLP
    2. Late Fusion: 各模态独立编码 → 注意力加权
    3. Hybrid: Early + Late 混合
    4. Transformer: Cross-Attention 多头融合
    """
    
    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self.cfg = self.config
        
        # 模态编码器 (Late Fusion / Hybrid 模式)
        self.encoders: Dict[str, np.ndarray] = {}
        self._init_encoders()
        
        # Early Fusion 的 MLP 参数
        self._init_early_fusion_mlp()
        
        # Late Fusion 的注意力参数
        self._init_attention_weights()
        
        # Transformer 参数
        if self.cfg.fusion_type == "transformer":
            self._init_transformer_layers()
        
        # 融合状态
        self._last_attention_weights: Dict[str, float] = {}
        self._forward_count = 0
    
    def _init_encoders(self):
        """初始化模态编码器 (投影到统一维度)"""
        dim_map = {
            'vision': self.cfg.vision_dim,
            'audio': self.cfg.audio_dim,
            'tactile': self.cfg.tactile_dim,
            'force': self.cfg.force_dim,
            'imu': self.cfg.imu_dim,
            'language': self.cfg.lang_dim,
        }
        
        hidden = self.cfg.hidden_dim
        
        for modality, in_dim in dim_map.items():
            if in_dim > 0:
                # Xavier 初始化权重: W[in_dim, hidden]
                scale = np.sqrt(2.0 / (in_dim + hidden))
                W = np.random.randn(in_dim, hidden).astype(np.float32) * scale
                b = np.zeros(hidden, dtype=np.float32)
                self.encoders[modality] = {
                    'W': W, 'b': b, 'in_dim': in_dim, 'out_dim': hidden
                }
    
    def _init_early_fusion_mlp(self):
        """初始化 Early Fusion MLP"""
        # 计算总输入维度
        total_dim = 0
        for e in self.encoders.values():
            total_dim += e['out_dim']
        
        if total_dim == 0:
            total_dim = self.cfg.hidden_dim
        
        self._early_fc1_W = np.random.randn(total_dim, self.cfg.hidden_dim).astype(np.float32) * 0.01
        self._early_fc1_b = np.zeros(self.cfg.hidden_dim, dtype=np.float32)
        
        self._early_fc2_W = np.random.randn(self.cfg.hidden_dim, self.cfg.output_dim).astype(np.float32) * 0.01
        self._early_fc2_b = np.zeros(self.cfg.output_dim, dtype=np.float32)
        
        self._early_total_dim = total_dim
    
    def _init_attention_weights(self):
        """初始化注意力权重"""
        self._attention_W = np.random.randn(self.cfg.hidden_dim, self.cfg.hidden_dim).astype(np.float32) * 0.01
        self._attention_v = np.random.randn(self.cfg.hidden_dim, 1).astype(np.float32) * 0.01
    
    def _init_transformer_layers(self):
        """初始化 Transformer 层"""
        d_model = self.cfg.hidden_dim
        num_heads = self.cfg.num_heads
        d_k = d_model // num_heads
        d_v = d_model // num_heads
        
        self._transformer_layers = []
        for _ in range(self.cfg.num_layers):
            layer = {
                'W_q': np.random.randn(d_model, d_model).astype(np.float32) * 0.01,
                'W_k': np.random.randn(d_model, d_model).astype(np.float32) * 0.01,
                'W_v': np.random.randn(d_model, d_model).astype(np.float32) * 0.01,
                'W_o': np.random.randn(d_model, d_model).astype(np.float32) * 0.01,
                'W_ffn1': np.random.randn(d_model, d_model * 4).astype(np.float32) * 0.01,
                'W_ffn2': np.random.randn(d_model * 4, d_model).astype(np.float32) * 0.01,
                'num_heads': num_heads,
                'd_k': d_k,
            }
            self._transformer_layers.append(layer)
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_max = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - x_max)
        return e / (np.sum(e, axis=axis, keepdims=True) + 1e-8)
    
    def _layernorm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + eps)
    
    def _encode_modality(self, modality: str, features: np.ndarray) -> np.ndarray:
        """将模态特征编码到统一隐藏维度"""
        if modality not in self.encoders:
            return None
        
        enc = self.encoders[modality]
        
        # features: (B, in_dim) 或 (in_dim,)
        if features.ndim == 1:
            features = features[np.newaxis, :]

        actual_dim = features.shape[1]
        expected_dim = enc['W'].shape[0]

        # 维度对齐：自动填充截断
        if actual_dim != expected_dim:
            if actual_dim < expected_dim:
                # 填充到期望维度
                pad_width = [(0, 0), (0, expected_dim - actual_dim)]
                features = np.pad(features, pad_width, mode='constant', constant_values=0)
            else:
                # 截断到期望维度
                features = features[:, :expected_dim]

        # Linear: (B, in_dim) @ (in_dim, hidden) + b
        h = features @ enc['W'] + enc['b']
        
        # ReLU
        h = self._relu(h)
        
        # LayerNorm
        h = self._layernorm(h)
        
        return h  # (B, hidden)
    
    def _early_fusion_forward(self, encoded: Dict[str, np.ndarray], batch_size: int) -> np.ndarray:
        """Early Fusion: 拼接所有模态 → MLP"""
        # 收集所有编码后的模态
        all_features = []
        for m in ['vision', 'audio', 'tactile', 'force', 'imu', 'language']:
            if m in encoded and encoded[m] is not None:
                all_features.append(encoded[m])

        if not all_features:
            return np.zeros((batch_size, self.cfg.output_dim), dtype=np.float32)

        # 拼接: (B, total_dim)
        concat = np.concatenate(all_features, axis=1)
        total_dim = concat.shape[1]

        # 动态创建 FC 层 (处理实际模态数量 != 初始化模态数量的情况)
        hidden_dim = self.cfg.hidden_dim
        if total_dim != self._early_total_dim:
            # 动态创建适应实际输入大小的 FC 层
            fc1_W = np.random.randn(total_dim, hidden_dim).astype(np.float32) * 0.01
            fc1_b = np.zeros(hidden_dim, dtype=np.float32)
            fc2_W = np.random.randn(hidden_dim, self.cfg.output_dim).astype(np.float32) * 0.01
            fc2_b = np.zeros(self.cfg.output_dim, dtype=np.float32)
        else:
            fc1_W = self._early_fc1_W
            fc1_b = self._early_fc1_b
            fc2_W = self._early_fc2_W
            fc2_b = self._early_fc2_b

        # FC1: (B, total_dim) → (B, hidden)
        h = concat @ fc1_W + fc1_b
        h = self._relu(h)

        # FC2: (B, hidden) → (B, output)
        out = h @ fc2_W + fc2_b

        return out
    
    def _late_fusion_attention(self, encoded: Dict[str, np.ndarray], batch_size: int) -> Tuple[np.ndarray, Dict[str, float]]:
        """Late Fusion: 注意力加权融合"""
        # 将字典转换为有序特征列表
        modality_order = ['vision', 'audio', 'tactile', 'force', 'imu', 'language']
        features_list = []
        modalities_present = []
        
        for m in modality_order:
            if m in encoded and encoded[m] is not None:
                features_list.append(encoded[m])  # (B, hidden)
                modalities_present.append(m)
        
        if not features_list:
            return np.zeros((batch_size, self.cfg.output_dim), dtype=np.float32), {}
        
        # Stack: (B, num_modalities, hidden)
        stacked = np.stack(features_list, axis=1)
        
        # 计算注意力分数: (B, num_modalities, hidden) @ (hidden, hidden) → (B, num_modalities, hidden)
        # 然后 @ attention_v → (B, num_modalities, 1) → squeeze → (B, num_modalities)
        attention_scores = []
        for i in range(len(features_list)):
            f = features_list[i]  # (B, hidden)
            score = f @ self._attention_W @ self._attention_v  # (B, 1)
            attention_scores.append(score.squeeze(-1))  # (B,)
        
        attention_scores = np.stack(attention_scores, axis=1)  # (B, num_modalities)
        
        # Softmax: (B, num_modalities)
        attention_weights = self._softmax(attention_scores, axis=1)
        
        # 加权求和: (B, num_modalities, 1) * (B, num_modalities, hidden) → (B, hidden)
        # 更简单的实现: sum over modalities
        fused = np.zeros_like(features_list[0])
        for i, m in enumerate(modalities_present):
            fused += attention_weights[:, i:i+1] * features_list[i]
        
        # 投影到输出维度
        out_dim = self.cfg.output_dim
        hidden_dim = self.cfg.hidden_dim
        
        if out_dim != hidden_dim:
            W_proj = np.random.randn(hidden_dim, out_dim).astype(np.float32) * 0.01
            b_proj = np.zeros(out_dim, dtype=np.float32)
            out = fused @ W_proj + b_proj
        else:
            out = fused
        
        # 返回注意力权重 (标量，用于分析)
        att_dict = {}
        for i, m in enumerate(modalities_present):
            att_dict[m] = float(np.mean(attention_weights[:, i]))
        
        return out, att_dict
    
    def _transformer_forward(self, encoded: Dict[str, np.ndarray], batch_size: int) -> np.ndarray:
        """Transformer Cross-Attention 融合"""
        modality_order = ['vision', 'audio', 'tactile', 'force', 'imu', 'language']
        features_list = []
        modalities_present = []
        
        for m in modality_order:
            if m in encoded and encoded[m] is not None:
                features_list.append(encoded[m])
                modalities_present.append(m)
        
        if not features_list:
            return np.zeros((batch_size, self.cfg.output_dim), dtype=np.float32)
        
        num_mods = len(features_list)
        hidden = self.cfg.hidden_dim
        num_heads = self.cfg.num_heads
        d_k = self.cfg.hidden_dim // num_heads
        
        # 构建序列: (B, num_modalities, hidden)
        sequence = np.stack(features_list, axis=1)  # (B, M, hidden)
        
        # Reshape for multi-head: (B, M, num_heads, d_k) → (B, num_heads, M, d_k)
        # 为简化，这里用单头近似实现
        for layer in self._transformer_layers:
            # Self-attention (简化版)
            # Q = K = V = sequence
            W_q = layer['W_q']  # (hidden, hidden)
            W_k = layer['W_k']
            W_v = layer['W_v']
            W_o = layer['W_o']
            
            Q = sequence @ W_q  # (B, M, hidden)
            K = sequence @ W_k
            V = sequence @ W_v
            
            # Scaled dot-product attention
            scale = np.sqrt(d_k)
            # Q @ K^T: (B, M, hidden) @ (B, hidden, M) → (B, M, M)
            scores = np.matmul(Q, K.transpose(0, 2, 1)) / scale
            att_weights = self._softmax(scores, axis=-1)  # (B, M, M)
            
            # Attention output: (B, M, M) @ (B, M, hidden) → (B, M, hidden)
            att_out = np.matmul(att_weights, V)
            
            # Output projection
            att_out = att_out @ W_o  # (B, M, hidden)
            
            # Residual + LayerNorm
            sequence = self._layernorm(sequence + att_out)
            
            # FFN: (B, M, hidden) → (B, M, hidden*4) → (B, M, hidden)
            W_ffn1 = layer['W_ffn1']
            W_ffn2 = layer['W_ffn2']
            ffn_out = self._relu(sequence @ W_ffn1) @ W_ffn2
            sequence = self._layernorm(sequence + ffn_out)
        
        # 汇聚所有模态: (B, M, hidden) → mean → (B, hidden)
        fused = np.mean(sequence, axis=1)
        
        # 投影到输出维度
        out_dim = self.cfg.output_dim
        W_proj = np.random.randn(hidden, out_dim).astype(np.float32) * 0.01
        b_proj = np.zeros(out_dim, dtype=np.float32)
        out = fused @ W_proj + b_proj
        
        return out
    
    def __call__(self, multimodal: MultimodalInput) -> np.ndarray:
        """使对象可调用，调用 forward"""
        return self.forward(multimodal)

    def forward(self, multimodal: MultimodalInput) -> np.ndarray:
        """
        前向传播

        Args:
            multimodal: MultimodalInput, 包含各模态特征

        Returns:
            fused_features: (B, output_dim) 始终2D
        """
        self._forward_count += 1
        
        # 编码所有可用模态
        encoded = {}
        modalities = multimodal.get_available_modalities()
        batch_size = 1  # 默认 batch_size

        for m in modalities:
            feat = getattr(multimodal, m)
            if feat is not None:
                # 自动转换 torch tensor → numpy
                if isinstance(feat, torch.Tensor):
                    feat = feat.detach().cpu().numpy().astype(np.float32)
                encoded[m] = self._encode_modality(m, feat)

        if not encoded:
            return np.zeros((batch_size, self.cfg.output_dim), dtype=np.float32)

        # 确定实际的 batch_size
        first_mod = list(encoded.values())[0]
        batch_size = first_mod.shape[0] if first_mod.ndim > 1 else 1
        
        # 根据融合类型选择前向路径
        if self.cfg.fusion_type == "early":
            output = self._early_fusion_forward(encoded, batch_size)
            self._last_attention_weights = {}
            
        elif self.cfg.fusion_type == "late":
            output, att_dict = self._late_fusion_attention(encoded, batch_size)
            self._last_attention_weights = att_dict
            
        elif self.cfg.fusion_type == "hybrid":
            # Hybrid: Early + Late concat + MLP
            early_out = self._early_fusion_forward(encoded, batch_size)
            late_out, att_dict = self._late_fusion_attention(encoded, batch_size)
            self._last_attention_weights = att_dict
            
            # 拼接 early 和 late
            combined = np.concatenate([early_out, late_out], axis=1)  # (B, 2*output)
            
            # 投影回 output_dim
            W_h = np.random.randn(2 * self.cfg.output_dim, self.cfg.output_dim).astype(np.float32) * 0.01
            b_h = np.zeros(self.cfg.output_dim, dtype=np.float32)
            output = self._relu(combined @ W_h + b_h)
        
        elif self.cfg.fusion_type == "transformer":
            output = self._transformer_forward(encoded, batch_size)
            self._last_attention_weights = {}
        
        else:
            output = self._early_fusion_forward(encoded, batch_size)
        
        # Keep 2D shape (B, output_dim) even when B==1 for consistent API
        # (No squeeze — removes inconsistent 1D/2D behavior)
        return output
    
    def get_attention_weights(self) -> Dict[str, float]:
        """获取最近一次前向传播的注意力权重"""
        return self._last_attention_weights.copy()
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置"""
        return {
            'vision_dim': self.cfg.vision_dim,
            'audio_dim': self.cfg.audio_dim,
            'tactile_dim': self.cfg.tactile_dim,
            'force_dim': self.cfg.force_dim,
            'imu_dim': self.cfg.imu_dim,
            'lang_dim': self.cfg.lang_dim,
            'hidden_dim': self.cfg.hidden_dim,
            'output_dim': self.cfg.output_dim,
            'fusion_type': self.cfg.fusion_type,
            'num_heads': self.cfg.num_heads,
            'num_layers': self.cfg.num_layers,
            'forward_count': self._forward_count,
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self._forward_count = 0
        self._last_attention_weights = {}

    def train(self) -> "CrossModalFusion":
        """设置训练模式（numpy版本为无操作）"""
        return self

    def eval(self) -> "CrossModalFusion":
        """设置评估模式（numpy版本为无操作）"""
        return self

    def parameters(self):
        """返回可迭代参数（numpy版本返回空列表）"""
        return iter([])


# =============================================================================
# AGV 五级融合规格
# =============================================================================

FUSION_GRADES = {
    'S': {
        'fusion_type': 'early',
        'hidden_dim': 128,
        'output_dim': 128,
        'modalities': 3,   # vision, audio, imu
        'max_modalities': 6,
        'attention_heads': 1,
        'transformer_layers': 0,
        'descriptor_dim': 128,
    },
    'M': {
        'fusion_type': 'late',
        'hidden_dim': 256,
        'output_dim': 256,
        'modalities': 4,   # +tactile
        'max_modalities': 6,
        'attention_heads': 4,
        'transformer_layers': 0,
        'descriptor_dim': 256,
    },
    'L': {
        'fusion_type': 'hybrid',
        'hidden_dim': 512,
        'output_dim': 512,
        'modalities': 5,   # +force
        'max_modalities': 6,
        'attention_heads': 8,
        'transformer_layers': 1,
        'descriptor_dim': 512,
    },
    'XL': {
        'fusion_type': 'transformer',
        'hidden_dim': 768,
        'output_dim': 768,
        'modalities': 6,   # all
        'max_modalities': 8,
        'attention_heads': 12,
        'transformer_layers': 2,
        'descriptor_dim': 768,
    },
    'XXL': {
        'fusion_type': 'transformer',
        'hidden_dim': 1024,
        'output_dim': 1024,
        'modalities': 6,   # all + future
        'max_modalities': 10,
        'attention_heads': 16,
        'transformer_layers': 3,
        'descriptor_dim': 1024,
    },
}


def get_fusion_spec(grade: str) -> dict:
    """获取 AGV 指定等级的融合规格"""
    return FUSION_GRADES.get(grade, FUSION_GRADES['M'])


def create_fusion_for_grade(grade: str) -> CrossModalFusion:
    """为指定 AGV 等级创建融合网络"""
    spec = get_fusion_spec(grade)
    
    cfg = FusionConfig(
        hidden_dim=spec['hidden_dim'],
        output_dim=spec['output_dim'],
        fusion_type=spec['fusion_type'],
        num_heads=spec['attention_heads'],
        num_layers=spec['transformer_layers'],
    )
    
    return CrossModalFusion(cfg)


# =============================================================================
# 简化接口 (与现有测试兼容)
# =============================================================================

def create_multimodal_input(**kwargs) -> MultimodalInput:
    """
    创建多模态输入对象的辅助函数

    Args:
        **kwargs: 模态名称到 numpy 数组的映射
                   支持: vision, audio, tactile, force, imu, language

    Returns:
        MultimodalInput: 多模态输入对象
    """
    # 确保所有输入都是 numpy 数组
    processed = {}
    for key, val in kwargs.items():
        if val is None:
            continue
        if isinstance(val, np.ndarray):
            # 如果是1D向量，确保是2D (B, D)
            if val.ndim == 1:
                val = val.reshape(1, -1)
            processed[key] = val.astype(np.float32)
        elif isinstance(val, torch.Tensor):
            # 支持 torch tensor，自动转换
            if val.ndim == 1:
                val = val.unsqueeze(0)
            processed[key] = val.detach().cpu().numpy().astype(np.float32)

    return MultimodalInput(**processed)


def simple_fusion(
    vision: np.ndarray,
    tactile: Optional[np.ndarray] = None,
    force: Optional[np.ndarray] = None,
    imu: Optional[np.ndarray] = None,
    hidden_dim: int = 256
) -> np.ndarray:
    """
    简化多模态融合接口
    
    Args:
        vision: 视觉特征 (B, 512) 或 (512,)
        tactile: 触觉特征 (B, 64) 或 None
        force: 力觉特征 (B, 32) 或 None
        imu: IMU特征 (B, 32) 或 None
        hidden_dim: 融合维度
        
    Returns:
        fused: 融合特征 (B, hidden_dim) 或 (hidden_dim,)
    """
    multimodal = MultimodalInput(
        vision=vision,
        tactile=tactile,
        force=force,
        imu=imu,
    )
    
    cfg = FusionConfig(
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        fusion_type='hybrid',
    )
    
    fusion = CrossModalFusion(cfg)
    return fusion.forward(multimodal)


# =============================================================================
# CrossModalAttention (PyTorch) - 跨模态注意力模块
# =============================================================================

class CrossModalAttention(nn.Module):
    """
    跨模态注意力模块

    实现标准的 Multi-Head Cross-Attention:
    - query:来自目标模态
    - key/value:来自源模态

    Args:
        query_dim: Query 向量维度
        key_dim: Key 向量维度
        value_dim: Value 向量维度
        num_heads: 注意力头数量
        dropout: Dropout 比率
    """

    def __init__(
        self,
        query_dim: int = 128,
        key_dim: int = 128,
        value_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        assert key_dim % num_heads == 0, "key_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_k = query_dim // num_heads
        self.d_v = value_dim // num_heads

        self.W_q = nn.Linear(query_dim, query_dim)
        self.W_k = nn.Linear(key_dim, key_dim)
        self.W_v = nn.Linear(value_dim, value_dim)
        self.W_o = nn.Linear(value_dim, value_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, T_q, query_dim)
            key: (B, T_k, key_dim)
            value: (B, T_k, value_dim)
            mask: (B, T_q, T_k) 可选掩码

        Returns:
            output: (B, T_q, value_dim)
        """
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        # Linear projections
        Q = self.W_q(query)  # (B, T_q, query_dim)
        K = self.W_k(key)    # (B, T_k, key_dim)
        V = self.W_v(value)  # (B, T_k, value_dim)

        # Reshape for multi-head: (B, T, num_heads, d_k) -> (B, num_heads, T, d_k)
        Q = Q.view(B, T_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T_k, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T_k, self.num_heads, self.d_v).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        # scores: (B, num_heads, T_q, T_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # attn_output: (B, num_heads, T_q, d_v)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        # (B, T_q, num_heads, d_v)
        attn_output = attn_output.view(B, T_q, self.num_heads * self.d_v)
        # (B, T_q, value_dim)

        # Output projection
        output = self.W_o(attn_output)

        return output


# =============================================================================
# ModalityEncoder - 通用单模态编码器
# =============================================================================

class ModalityEncoder(nn.Module):
    """
    通用模态编码器

    将任意模态的原始数据编码为固定维度的特征向量:
    - vision: (B, C, H, W) → (B, hidden_dim)
    - audio: (B, T, D) → (B, hidden_dim)
    - tactile: (B, H, W) → (B, hidden_dim)
    - force: (B, 6) → (B, hidden_dim)
    - imu: (B, 6) → (B, hidden_dim)

    Args:
        modality: 模态类型 ("vision"/"audio"/"tactile"/"force"/"imu")
        input_dim: 输入维度
        hidden_dim: 隐藏层/输出维度
    """

    def __init__(
        self,
        modality: str = "force",
        input_dim: int = 6,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.modality = modality
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if modality == "vision":
            # Vision: MLP encoder for feature vectors (B, D) or Conv2D for images (B, C, H, W)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
            )
        elif modality == "audio":
            # Audio: 1D temporal encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
        elif modality == "tactile":
            # Tactile: 2D convolutional encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32), nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(32, hidden_dim),
            )
        elif modality in ("force", "imu"):
            # Force/IMU: MLP encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
        else:
            # Default: MLP
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状因模态而异

        Returns:
            features: (B, hidden_dim) 编码特征
        """
        return self.encoder(x)


# =============================================================================
# LanguageEncoder - 语言编码器
# =============================================================================

class LanguageEncoder(nn.Module):
    """
    语言编码器

    将文本 token 序列编码为特征向量:
    - 词嵌入 + 位置编码
    - Transformer 编码
    - [CLS] token 池化

    Args:
        vocab_size: 词表大小
        embed_dim: 嵌入维度
        hidden_dim: 隐藏层维度
        max_len: 最大序列长度
        num_heads: 注意力头数
        num_layers: Transformer 层数
        dropout: Dropout 比率
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        max_len: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
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
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.use_cls = True
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, L) token indices

        Returns:
            features: (B, hidden_dim)
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


# =============================================================================
# UnifiedRepresentation (PyTorch) - 三路分解表示层
# =============================================================================

class UnifiedRepresentation(nn.Module):
    """
    统一表示层 - 将融合特征分解为三路表示

    将统一的表示向量分解为:
    1. State Representation (状态表示): 环境感知状态
    2. Action Representation (动作表示): 候选动作编码
    3. World Representation (世界模型表示): 世界模型输入

    结构: LayerNorm → MLP → 三路输出

    Args:
        input_dim: 输入特征维度 (即 CrossModalFusion.output_dim)
        hidden_dim: 隐藏层维度
        output_dim: 各输出分支维度
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # LayerNorm + MLP
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # 三路输出头
        self.state_head = nn.Linear(hidden_dim, output_dim)
        self.action_head = nn.Linear(hidden_dim, output_dim)
        self.world_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入张量 (B, input_dim)

        Returns:
            state: 状态表示 (B, output_dim)
            action: 动作表示 (B, output_dim)
            world: 世界模型表示 (B, output_dim)
        """
        h = self.norm(x)
        h = self.mlp(h)

        state = self.state_head(h)
        action = self.action_head(h)
        world = self.world_head(h)

        return state, action, world

    def get_config(self) -> dict:
        """获取配置"""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
        }
