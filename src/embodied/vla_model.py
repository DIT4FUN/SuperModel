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
vla_model.py - Vision-Language-Action (VLA) 端到端模型
SuperModel 超模态大模型具身智能系统

VLA模型: 接收视觉+语言输入，输出机器人动作
支持AGV五级规格 (S/M/L/XL/XXL)

功能:
- Vision Encoder: 图像特征提取 (ResNet/ViT风格)
- Language Encoder: 指令编码 (BERT风格)
- Cross-Modal Fusion: 视觉-语言特征融合
- Action Decoder: 动作序列生成 (Transformer解码器)
- AGV五级动作规格适配
- 动作空间: 末端速度/关节角度/力矩/导航Twist
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    'VLAActionSpace',
    'VLAGrade',
    'VLAAction',
    'VLAPerceptionFrame',
    'VLAInput',
    'VLAOutput',
    'VisionEncoder',
    'LanguageEncoder',
    'ActionDecoder',
    'VLAModel',
    'create_vla_model',
    'load_vla_model',
]


# ============================================================
# 动作空间定义
# ============================================================

class VLAActionSpace(Enum):
    """VLA动作空间类型"""
    TWIST = "twist"                  # 导航Twist (vx, vy, vz, rx, ry, rz)
    JOINT_POSITION = "joint_position"  # 关节位置
    JOINT_VELOCITY = "joint_velocity"  # 关节速度
    END_EFFECTOR = "end_effector"    # 末端执行器位姿
    GRIPPER = "gripper"              # 夹爪开合
    COMBINED = "combined"             # 组合动作 (Twist + Gripper)


class VLAGrade(str, Enum):
    """VLA模型AGV五级规格"""
    S = "S"   # 基础导航Twist
    M = "M"   # 导航 + 简单夹爪
    L = "L"   # 导航 + 关节控制 + 夹爪
    XL = "XL" # 全动作空间 + 力控
    XXL = "XXL"  # 完整VLA + MPC预测


# ============================================================
# VLA 数据结构
# ============================================================

@dataclass
class VLAAction:
    """VLA输出的单步动作
    
    统一动作格式，支持不同AGV等级的动作空间。
    实际使用的字段由 action_space 参数决定。
    """
    # 元数据
    action_id: str = field(default_factory=lambda: str(uuid.uuid4().hex[:8]))
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0  # 动作置信度 [0, 1]
    action_space: VLAActionSpace = VLAActionSpace.TWIST
    
    # 导航Twist (linear + angular)
    vx: float = 0.0   # m/s
    vy: float = 0.0   # m/s
    vz: float = 0.0   # m/s
    rx: float = 0.0   # rad/s
    ry: float = 0.0   # rad/s
    rz: float = 0.0   # rad/s
    
    # 关节控制 (最多8个关节)
    joint_positions: Optional[List[float]] = None  # rad
    joint_velocities: Optional[List[float]] = None  # rad/s
    
    # 末端执行器
    ee_pose: Optional[List[float]] = None  # [x, y, z, roll, pitch, yaw]
    ee_force: Optional[List[float]] = None  # [fx, fy, fz, mx, my, mz]
    
    # 夹爪 (0=完全打开, 1=完全闭合)
    gripper_position: float = 0.0
    gripper_force: float = 0.0  # N
    
    # 元认知
    attention_weights: Optional[Dict[str, float]] = None  # 各模态注意力权重
    reasoning: Optional[str] = None  # 动作推理描述

    def to_twist(self) -> Tuple[float, float, float, float, float, float]:
        """转换为Twist格式"""
        return (self.vx, self.vy, self.vz, self.rx, self.ry, self.rz)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action_id': self.action_id,
            'timestamp': self.timestamp,
            'confidence': self.confidence,
            'action_space': self.action_space.value,
            'twist': self.to_twist(),
            'joint_positions': self.joint_positions,
            'ee_pose': self.ee_pose,
            'gripper_position': self.gripper_position,
            'attention_weights': self.attention_weights,
            'reasoning': self.reasoning,
        }


@dataclass
class VLAPerceptionFrame:
    """VLA感知帧 - 单帧多模态感知输入"""
    frame_id: str = field(default_factory=lambda: str(uuid.uuid4().hex[:8]))
    timestamp: float = field(default_factory=time.time)
    
    # 视觉 (H, W, C) uint8 -> 将由 VisionEncoder 处理
    rgb_image: Optional[np.ndarray] = None
    depth_image: Optional[np.ndarray] = None
    
    # 激光雷达
    lidar_scan: Optional[np.ndarray] = None  # (N,) range readings
    lidar_angles: Optional[np.ndarray] = None
    
    # 语言指令
    instruction: str = ""
    instruction_tokens: Optional[List[int]] = None
    
    # 状态
    joint_states: Optional[np.ndarray] = None  # 当前关节角度
    base_pose: Optional[np.ndarray] = None  # (x, y, theta)
    battery_level: float = 1.0
    
    def get_modalities(self) -> List[str]:
        mods = []
        if self.rgb_image is not None: mods.append('vision')
        if self.depth_image is not None: mods.append('depth')
        if self.lidar_scan is not None: mods.append('lidar')
        if self.instruction: mods.append('language')
        if self.joint_states is not None: mods.append('proprioception')
        return mods


@dataclass
class VLAInput:
    """VLA模型输入"""
    perception: VLAPerceptionFrame
    history_actions: Optional[List[VLAAction]] = None  # 历史动作 (用于时序建模)
    context: Optional[Dict[str, Any]] = None  # 额外上下文
    
    def get_feature_dim(self) -> int:
        """估算输入特征维度"""
        dim = 0
        if self.perception.rgb_image is not None: dim += 512
        if self.perception.lidar_scan is not None: dim += 128
        if self.perception.instruction: dim += 768
        if self.perception.joint_states is not None: dim += len(self.perception.joint_states) * 4
        return dim


@dataclass 
class VLAOutput:
    """VLA模型输出"""
    action: VLAAction
    action_sequence: List[VLAAction] = field(default_factory=list)  # 多步预测
    fused_features: Optional[np.ndarray] = None
    vision_features: Optional[np.ndarray] = None
    language_features: Optional[np.ndarray] = None
    action_logits: Optional[np.ndarray] = None  # 用于安全验证
    
    # 安全相关
    collision_risk: float = 0.0  # 碰撞风险评估
    safety_override: bool = False
    fallback_action: Optional[VLAAction] = None
    
    # 执行信息
    inference_time_ms: float = 0.0
    model_version: str = "v1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action.to_dict(),
            'action_sequence': [a.to_dict() for a in self.action_sequence],
            'collision_risk': self.collision_risk,
            'safety_override': self.safety_override,
            'inference_time_ms': self.inference_time_ms,
            'model_version': self.model_version,
        }


# ============================================================
# Vision Encoder
# ============================================================

class VisionEncoder:
    """
    视觉编码器 - 从图像提取特征
    
    使用简化版 CNN (ResNet-style):
    - 4个残差块
    - 全局平均池化
    - 特征投影层
    
    支持:
    - RGB图像 (H, W, 3) -> (512,) 特征
    - 深度图像 (H, W, 1) -> (128,) 特征
    - 激光雷达 (N,) -> (128,) 特征
    """
    
    def __init__(
        self,
        vision_dim: int = 512,
        lidar_dim: int = 128,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        self.vision_dim = vision_dim
        self.lidar_dim = lidar_dim
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout
        
        # 残差块权重 (简化初始化)
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        global_scope = self.vision_dim
        
        def _create_layer(in_ch, out_ch, stride=1):
            # Conv 权重
            w_conv = np.random.randn(out_ch, in_ch, 3, 3).astype(np.float32) * 0.01
            b_conv = np.zeros(out_ch, dtype=np.float32)
            # BN 权重 (gamma, beta, mean, var)
            if self.use_batch_norm:
                bn_gamma = np.ones(out_ch, dtype=np.float32)
                bn_beta = np.zeros(out_ch, dtype=np.float32)
                bn_mean = np.zeros(out_ch, dtype=np.float32)
                bn_var = np.ones(out_ch, dtype=np.float32)
            else:
                bn_gamma = bn_beta = bn_mean = bn_var = None
            return {
                'w_conv': w_conv, 'b_conv': b_conv,
                'bn_gamma': bn_gamma, 'bn_beta': bn_beta,
                'bn_mean': bn_mean, 'bn_var': bn_var,
                'in_ch': in_ch, 'out_ch': out_ch, 'stride': stride,
            }
        
        # ResNet-style: conv1 -> bn1 -> conv2 -> bn2 -> conv3 -> bn3 -> conv4 -> bn4
        self.conv1 = _create_layer(3, 64, stride=2)    # 112x112
        self.conv2 = _create_layer(64, 128, stride=2)  # 56x56
        self.conv3 = _create_layer(128, 256, stride=2) # 28x28
        self.conv4 = _create_layer(256, 512, stride=2) # 14x14
        
        # 投影层 (12 -> vision_dim)
        self.proj_w = np.random.randn(12, global_scope).astype(np.float32) * np.sqrt(2.0 / (12 + global_scope))
        self.proj_b = np.zeros(global_scope, dtype=np.float32)
        
        # Lidar编码
        self.lidar_proj_w = np.random.randn(self.lidar_dim * 2, self.lidar_dim).astype(np.float32) * 0.01
        self.lidar_proj_b = np.zeros(self.lidar_dim, dtype=np.float32)
        
        # Depth编码 (13 stats features -> lidar_dim)
        self.depth_proj_w = np.random.randn(13, self.lidar_dim).astype(np.float32) * np.sqrt(2.0 / (13 + self.lidar_dim))
        self.depth_proj_b = np.zeros(self.lidar_dim, dtype=np.float32)
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x).astype(np.float32)
    
    def _conv2d(self, x: np.ndarray, layer: Dict) -> np.ndarray:
        """简化2D卷积"""
        # x: (C, H, W) -> (out_ch,) 特征向量
        # 使用简化实现：全局平均池化 + 全连接
        in_ch = layer['in_ch']
        out_ch = layer['out_ch']
        
        # 确保x是3D
        if x.ndim == 1:
            x = x.reshape(1, 1)
        elif x.ndim == 2:
            x = x.reshape(x.shape[0], -1)
        
        # 全局平均 -> (in_ch,)
        x_pooled = x.reshape(x.shape[0], -1).mean(axis=1)[:in_ch]
        
        # 全连接: (in_ch,) -> (out_ch,)
        w = layer['w_conv'].reshape(layer['out_ch'], -1)[:, :in_ch]
        h_fc = np.dot(w, x_pooled) + layer['b_conv']
        return h_fc.astype(np.float32)
    
    def encode_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        编码RGB图像
        
        Args:
            image: (H, W, 3) uint8 或 (3, H, W) float32
            
        Returns:
            features: (vision_dim,) float32
        """
        # 预处理: 归一化到 [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # 转CHW
        if image.shape[0] == 3:
            x = image  # 已经是 (3, H, W)
        else:
            x = np.transpose(image, (2, 0, 1))  # (H, W, 3) -> (3, H, W)
        
        # 全局平均池化 -> 压缩为 (3,) 通道统计
        x_flat = x.reshape(x.shape[0], -1)
        feat_mean = x_flat.mean(axis=1)  # (3,)
        feat_std = x_flat.std(axis=1) + 1e-6  # (3,)
        feat_min = x_flat.min(axis=1)  # (3,)
        feat_max = x_flat.max(axis=1)  # (3,)
        
        # 拼接统计特征 -> (12,)
        stats = np.concatenate([feat_mean, feat_std, feat_min, feat_max]).astype(np.float32)
        
        # 投影到 (vision_dim,) - stats (12,) @ proj_w.T (512, 12) -> (512,)
        feat = np.dot(stats, self.proj_w) + self.proj_b
        return feat.astype(np.float32)
    
    def encode_depth(self, depth: np.ndarray) -> np.ndarray:
        """编码深度图像 - 返回 (lidar_dim,) 特征"""
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0  # mm -> m
        elif depth.dtype == np.uint8:
            depth = depth.astype(np.float32) / 255.0 * 10.0  # 假设最大10m
        
        # 确保是2D
        if depth.ndim == 3:
            depth = depth[:, :, 0]  # 取第一通道
        
        # 深度统计特征
        d_flat = depth.reshape(-1)
        d_mean = d_flat.mean()
        d_std = d_flat.std() + 1e-6
        d_min = d_flat.min()
        d_max = d_flat.max()
        d_median = np.median(d_flat)
        
        # 直方图特征 (简化)
        hist, _ = np.histogram(d_flat, bins=8, range=(0, d_max + 1e-6))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-6)
        
        # 拼接
        stats = np.array([d_mean, d_std, d_min, d_max, d_median] + list(hist), dtype=np.float32)
        
        # 投影: stats (13,) @ depth_proj_w (13, lidar_dim) -> (lidar_dim,)
        feat = stats @ self.depth_proj_w + self.depth_proj_b
        return feat.astype(np.float32)
    
    def encode_lidar(self, scan: np.ndarray, angles: Optional[np.ndarray] = None) -> np.ndarray:
        """
        编码激光雷达扫描
        
        Args:
            scan: (N,) range readings
            angles: (N,) angles in radians
            
        Returns:
            features: (lidar_dim,) float32
        """
        N = scan.shape[0]
        
        # 直角坐标投影
        if angles is None:
            angles = np.linspace(-np.pi, np.pi, N)
        
        x = scan * np.cos(angles)
        y = scan * np.sin(angles)
        
        # 双视图: range + cartesian
        features = np.concatenate([scan, x, y]).astype(np.float32)  # (3N,)
        
        # 降维投影: feat_in (2*lidar_dim,) @ lidar_proj_w (2*lidar_dim, lidar_dim) -> (lidar_dim,)
        feat_in = features[:self.lidar_dim * 2]  # (2*lidar_dim,)
        feat = np.dot(feat_in, self.lidar_proj_w) + self.lidar_proj_b
        return feat.astype(np.float32)
    
    def encode(self, perception: VLAPerceptionFrame) -> np.ndarray:
        """统一编码接口 - 返回视觉特征 (vision_dim,)
        
        注意: 深度和激光雷达信息请使用 encode_full() 获取。
        """
        if perception.rgb_image is not None:
            rgb_feat = self.encode_rgb(perception.rgb_image)  # (vision_dim,)
        else:
            rgb_feat = np.zeros(self.vision_dim, dtype=np.float32)
        return rgb_feat

    def encode_full(self, perception: VLAPerceptionFrame) -> np.ndarray:
        """完整编码 - 返回视觉+激光雷达融合特征 (vision_dim + lidar_dim,)
        
        用于需要同时使用视觉和激光雷达的场景。
        """
        # 视觉特征
        vision_feat = self.encode(perception)  # (vision_dim,)
        
        # 激光雷达特征
        if perception.lidar_scan is not None:
            lidar_feat = self.encode_lidar(perception.lidar_scan, perception.lidar_angles)  # (lidar_dim,)
        else:
            lidar_feat = np.zeros(self.lidar_dim, dtype=np.float32)
        
        return np.concatenate([vision_feat, lidar_feat]).astype(np.float32)


# ============================================================
# Language Encoder
# ============================================================

class LanguageEncoder:
    """
    语言编码器 - 将文本指令编码为特征
    
    使用简化的 Transformer Encoder:
    - BPE分词 (简化版: 空格分词)
    - 词嵌入层
    - 位置编码
    - Transformer层 (self-attention + FFN)
    - 池化输出
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 256,
        hidden_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        max_length: int = 128,
        dropout: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_length = max_length
        self.dropout = dropout
        
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._build_vocab()
        self._init_weights()
    
    def _build_vocab(self):
        """构建基础词表"""
        # 特殊token
        special_tokens = ['<PAD>', '<UNK>', '<CLS>', '<SEP>']
        base_words = [
            # 导航指令
            'go', 'move', 'navigate', 'turn', 'forward', 'backward', 'left', 'right',
            'stop', 'wait', 'pause', 'resume', 'speed', 'slow', 'fast',
            # 抓取操作
            'grasp', 'pick', 'pickup', 'release', 'place', 'drop', 'lift', 'lower',
            'open', 'close', 'grip', 'hold',
            # 目标
            'target', 'destination', 'location', 'position', 'station', 'zone',
            'object', 'item', 'package', 'box', 'tray', 'pallet',
            # 传感器
            'sensor', 'camera', 'lidar', 'detect', 'see', 'look', 'avoid', 'obstacle',
            # 状态
            'battery', 'low', 'charge', 'full', 'empty',
            # 协同
            'collaborate', 'team', 'together', 'wait_for', 'signal',
            # 场景
            'warehouse', 'factory', 'hospital', 'lab', 'office', 'outdoor',
            # 任务
            'task', 'deliver', 'transport', 'patrol', 'inspect', 'check',
            # 安全
            'safe', 'careful', 'carefully', 'cautious', 'emergency', 'stop',
            # 数字
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
            # 方位
            'north', 'south', 'east', 'west', 'front', 'back', 'side',
        ]
        
        for i, token in enumerate(special_tokens + base_words):
            self._token_to_id[token] = i
            self._id_to_token[i] = token
        
        # 填充词表到vocab_size
        for i in range(len(special_tokens) + len(base_words), self.vocab_size):
            self._token_to_id[f'<UNK_{i}>'] = i
            self._id_to_token[i] = f'<UNK_{i}>'
    
    def _tokenize(self, text: str) -> List[int]:
        """简化分词"""
        words = text.lower().strip().split()
        return [self._token_to_id.get(w, self._token_to_id['<UNK>']) for w in words]
    
    def _init_weights(self):
        """初始化Transformer权重"""
        global_scope = self.hidden_dim
        
        # 词嵌入
        self.embed_w = np.random.randn(self.vocab_size, self.embed_dim).astype(np.float32) * 0.02
        
        # 位置编码
        self.pos_enc = np.zeros((self.max_length, self.embed_dim), dtype=np.float32)
        for pos in range(self.max_length):
            for i in range(0, self.embed_dim, 2):
                if i < self.embed_dim:
                    self.pos_enc[pos, i] = math.sin(pos / 10000 ** (2 * i / self.embed_dim))
                if i + 1 < self.embed_dim:
                    self.pos_enc[pos, i + 1] = math.cos(pos / 10000 ** (2 * i / self.embed_dim))
        
        # 嵌入投影
        self.embed_proj_w = np.random.randn(self.embed_dim, self.embed_dim).astype(np.float32) * np.sqrt(2.0 / self.embed_dim)
        self.embed_proj_b = np.zeros(self.embed_dim, dtype=np.float32)
        
        # Self-attention 权重 (单层简化)
        self.attn_w = np.random.randn(self.embed_dim, self.num_heads, self.embed_dim // self.num_heads).astype(np.float32) * np.sqrt(2.0 / self.embed_dim)
        self.attn_b = np.zeros((self.num_heads, self.embed_dim // self.num_heads), dtype=np.float32)
        
        # FFN
        ffn_hidden = self.embed_dim * 4
        self.ffn_w1 = np.random.randn(self.embed_dim, ffn_hidden).astype(np.float32) * np.sqrt(2.0 / self.embed_dim)
        self.ffn_b1 = np.zeros(ffn_hidden, dtype=np.float32)
        self.ffn_w2 = np.random.randn(ffn_hidden, self.embed_dim).astype(np.float32) * np.sqrt(2.0 / ffn_hidden)
        self.ffn_b2 = np.zeros(self.embed_dim, dtype=np.float32)
        
        # 输出投影 (embed_dim -> hidden_dim)
        self.out_proj_w = np.random.randn(self.embed_dim, global_scope).astype(np.float32) * np.sqrt(2.0 / self.embed_dim)
        self.out_proj_b = np.zeros(global_scope, dtype=np.float32)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum(axis=-1, keepdims=True)
    
    def encode(self, text: str) -> np.ndarray:
        """
        编码文本指令
        
        Args:
            text: 自然语言指令
            
        Returns:
            features: (hidden_dim,) float32
        """
        tokens = self._tokenize(text)
        
        # Truncate/pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [self._token_to_id['<PAD>']] * (self.max_length - len(tokens))
        
        # 词嵌入 + 位置编码
        embedded = self.embed_w[tokens].astype(np.float32)  # (max_len, embed_dim)
        embedded = embedded + self.pos_enc[:len(tokens)].astype(np.float32)
        
        # 投影
        h = np.dot(embedded, self.embed_proj_w) + self.embed_proj_b
        h = np.maximum(0, h)  # ReLU
        
        # 简化为: 直接对序列做加权平均
        seq_len = min(len(tokens), self.max_length)
        h_seq = h[:seq_len]  # (seq, embed_dim)
        # 简单注意力: 使用单一投影向量计算权重
        key = np.dot(h_seq, self.attn_w.mean(axis=1))  # (seq, embed_dim // num_heads)
        attn_weights = self._softmax(key.sum(axis=1, keepdims=True))  # (seq, 1)
        h_attn = h_seq * attn_weights  # (seq, embed_dim) broadcast
        
        # FFN
        h_ffn = np.dot(h_attn, self.ffn_w1) + self.ffn_b1
        h_ffn = np.maximum(0, h_ffn)  # ReLU
        h_ffn = np.dot(h_ffn, self.ffn_w2) + self.ffn_b2
        
        # 池化 (CLS token 或 mean pooling)
        if tokens[0] == self._token_to_id['<CLS>']:
            pooled = h_ffn[0]
        else:
            pooled = h_ffn.mean(axis=0)
        
        # 输出投影
        out = np.dot(pooled, self.out_proj_w) + self.out_proj_b
        return out.astype(np.float32)
    
    def batch_encode(self, texts: List[str]) -> np.ndarray:
        """批量编码"""
        return np.stack([self.encode(t) for t in texts])


# ============================================================
# Action Decoder
# ============================================================

class ActionDecoder:
    """
    动作解码器 - 将融合特征解码为动作序列
    
    使用 Transformer Decoder 架构:
    - 动作嵌入层
    - 因果掩码自注意力
    - 交叉注意力 (融合特征作为context)
    - MLP输出头
    
    支持动作空间:
    - TWIST: 6维 (vx, vy, vz, rx, ry, rz)
    - JOINT: N维关节位置
    - GRIPPER: 1维夹爪
    - COMBINED: 多模态组合
    """
    
    def __init__(
        self,
        action_space: VLAActionSpace = VLAActionSpace.TWIST,
        hidden_dim: int = 512,
        num_joints: int = 6,
        num_heads: int = 8,
        num_layers: int = 3,
        action_dim: int = 7,  # twist(6) + gripper(1)
        max_seq_len: int = 16,
        dropout: float = 0.1,
    ):
        self.action_space = action_space
        self.hidden_dim = hidden_dim
        self.num_joints = num_joints
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.action_dim = action_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化解码器权重"""
        scope = self.hidden_dim
        
        # 动作嵌入
        self.action_embed_w = np.random.randn(self.action_dim, scope).astype(np.float32) * 0.02
        self.action_embed_b = np.zeros(scope, dtype=np.float32)
        
        # 位置编码 (用于解码器)
        self.pos_enc = np.zeros((self.max_seq_len, scope), dtype=np.float32)
        for pos in range(self.max_seq_len):
            for i in range(0, scope, 2):
                if i < scope:
                    self.pos_enc[pos, i] = math.sin(pos / 10000 ** (2 * i / scope))
                if i + 1 < scope:
                    self.pos_enc[pos, i + 1] = math.cos(pos / 10000 ** (2 * i / scope))
        
        # 交叉注意力权重 (简化)
        self.cross_w = np.random.randn(scope, scope).astype(np.float32) * np.sqrt(2.0 / scope)
        self.cross_b = np.zeros(scope, dtype=np.float32)
        
        # 动作头 MLP
        self.action_head_w1 = np.random.randn(scope, scope).astype(np.float32) * np.sqrt(2.0 / scope)
        self.action_head_b1 = np.zeros(scope, dtype=np.float32)
        self.action_head_w2 = np.random.randn(scope, self.action_dim).astype(np.float32) * 0.01
        self.action_head_b2 = np.zeros(self.action_dim, dtype=np.float32)
        
        # 置信度头
        self.conf_head_w = np.random.randn(scope, 1).astype(np.float32) * np.sqrt(2.0 / scope)
        self.conf_head_b = np.zeros(1, dtype=np.float32)
    
    def _tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x).astype(np.float32)
    
    def decode(
        self,
        fused_features: np.ndarray,
        history_actions: Optional[List[VLAAction]] = None,
        num_steps: int = 1,
    ) -> Tuple[List[VLAAction], np.ndarray]:
        """
        解码动作序列
        
        Args:
            fused_features: (hidden_dim,) 融合特征
            history_actions: 历史动作列表
            num_steps: 预测步数
            
        Returns:
            actions: VLAAction列表
            logits: (num_steps, action_dim) 动作logits
        """
        seq_len = min(num_steps, self.max_seq_len)
        logits_list = []
        actions = []
        
        # 初始化动作嵌入
        prev_action = np.zeros(self.action_dim, dtype=np.float32)
        
        for step in range(seq_len):
            # 动作嵌入 + 位置编码
            action_embed = np.dot(prev_action, self.action_embed_w) + self.action_embed_b
            action_embed = action_embed + self.pos_enc[step]
            
            # 交叉注意力 (融合特征作为K/V)
            context_attn = np.tanh(np.dot(action_embed, self.cross_w) + self.cross_b)
            context_attn = context_attn * fused_features  # 元素乘
            
            # 残差连接
            h = action_embed + context_attn
            
            # MLP
            h = self._tanh(np.dot(h, self.action_head_w1) + self.action_head_b1)
            h = self._tanh(np.dot(h, self.action_head_w2) + self.action_head_b2)
            
            # 动作头
            logits = h  # (action_dim,)
            logits_list.append(logits)
            
            # 转换为VLAAction
            action = self._logits_to_action(logits)
            actions.append(action)
            
            # 更新 prev_action (用于下一步)
            prev_action = self._action_to_vector(action)
        
        logits_arr = np.stack(logits_list) if logits_list else np.zeros((0, self.action_dim))
        return actions, logits_arr
    
    def _action_to_vector(self, action: VLAAction) -> np.ndarray:
        """将VLAAction转换为向量"""
        vec = np.zeros(self.action_dim, dtype=np.float32)
        vec[0] = action.vx
        vec[1] = action.vy
        vec[2] = action.vz
        vec[3] = action.rx
        vec[4] = action.ry
        vec[5] = action.rz
        if self.action_dim > 6:
            vec[6] = action.gripper_position
        return vec
    
    def _logits_to_action(self, logits: np.ndarray) -> VLAAction:
        """将logits转换为VLAAction (带激活)"""
        # 限制动作范围
        action = VLAAction()
        
        if self.action_space in (VLAActionSpace.TWIST, VLAActionSpace.COMBINED):
            action.vx = float(np.tanh(logits[0]) * 2.0)   # [-2, 2] m/s
            action.vy = float(np.tanh(logits[1]) * 2.0)
            action.vz = float(np.tanh(logits[2]) * 0.5)
            action.rx = float(np.tanh(logits[3]) * math.pi)  # [-π, π] rad/s
            action.ry = float(np.tanh(logits[4]) * math.pi)
            action.rz = float(np.tanh(logits[5]) * math.pi)
            
            if self.action_dim > 6:
                action.gripper_position = float(np.tanh(logits[6]))  # [-1, 1]
        
        action.action_space = self.action_space
        return action


# ============================================================
# VLA Model 主类
# ============================================================

@dataclass
class VLAConfig:
    """VLA模型配置"""
    # 编码器配置
    vision_dim: int = 512
    lidar_dim: int = 128
    lang_hidden_dim: int = 768
    lang_embed_dim: int = 256
    lang_vocab_size: int = 10000
    lang_max_length: int = 128
    
    # 融合配置
    fusion_hidden_dim: int = 512
    fusion_num_heads: int = 8
    fusion_num_layers: int = 2
    
    # 解码器配置
    action_space: VLAActionSpace = VLAActionSpace.TWIST
    num_joints: int = 6
    action_seq_len: int = 8
    max_decode_steps: int = 16
    
    # AGV等级
    grade: str = "M"
    
    def get_grade_action_space(self) -> VLAActionSpace:
        """根据AGV等级确定动作空间"""
        grade_map = {
            'S': VLAActionSpace.TWIST,
            'M': VLAActionSpace.COMBINED,
            'L': VLAActionSpace.COMBINED,
            'XL': VLAActionSpace.COMBINED,
            'XXL': VLAActionSpace.COMBINED,
        }
        return grade_map.get(self.grade.upper(), VLAActionSpace.TWIST)
    
    def get_action_dim(self) -> int:
        """获取动作维度"""
        if self.action_space == VLAActionSpace.TWIST:
            return 6
        elif self.action_space == VLAActionSpace.GRIPPER:
            return 1
        else:  # COMBINED
            return 7  # twist(6) + gripper(1)


class VLAModel:
    """
    Vision-Language-Action 端到端模型
    
    架构:
    ┌──────────────┐   ┌────────────────┐
    │ VisionEncoder │ + │ LanguageEncoder │
    └──────┬───────┘   └───────┬────────┘
           │                   │
           └────────┬──────────┘
                    ▼
           ┌────────────────┐
           │ Cross-Modal     │
           │ Fusion          │
           └────────┬───────┘
                    ▼
           ┌────────────────┐
           │ ActionDecoder  │
           └────────┬───────┘
                    ▼
           ┌────────────────┐
           │ VLAAction      │
           └────────────────┘
    
    使用示例:
        model = VLAModel(grade="M")
        model.start()
        
        # 单步推理
        perception = VLAPerceptionFrame(
            rgb_image=camera_frame,
            instruction="go to station A"
        )
        output = model.step(perception)
        twist = output.action.to_twist()
        
        model.stop()
    """
    
    def __init__(
        self,
        config: Optional[VLAConfig] = None,
        grade: str = "M",
    ):
        # Always derive action_space/action_dim from grade (they're grade-dependent)
        if config is None:
            self.config = VLAConfig(grade=grade.upper())
        else:
            self.config = config
        self.config.action_space = self.config.get_grade_action_space()
        self.config.action_dim = self.config.get_action_dim()
        
        self._is_running = False
        
        # 子模块
        self.vision_encoder = VisionEncoder(
            vision_dim=self.config.vision_dim,
            lidar_dim=self.config.lidar_dim,
        )
        self.lang_encoder = LanguageEncoder(
            embed_dim=self.config.lang_embed_dim,
            hidden_dim=self.config.lang_hidden_dim,
            vocab_size=self.config.lang_vocab_size,
            max_length=self.config.lang_max_length,
            num_heads=self.config.fusion_num_heads,
            num_layers=self.config.fusion_num_layers,
        )
        self.action_decoder = ActionDecoder(
            action_space=self.config.action_space,
            hidden_dim=self.config.fusion_hidden_dim,
            num_joints=self.config.num_joints,
            action_dim=self.config.action_dim,
            max_seq_len=self.config.max_decode_steps,
        )
        
        # 融合层
        self._init_fusion_weights()
        
        # 状态
        self._history: List[VLAAction] = []
        self._total_inferences = 0
        
        logger.info(f"VLAModel initialized: grade={self.config.grade}, "
                   f"action_space={self.config.action_space.value}, "
                   f"action_dim={self.config.action_dim}")
    
    def _init_fusion_weights(self):
        """初始化融合层"""
        vd = self.config.vision_dim
        lidar_d = self.vision_encoder.lidar_dim  # 128
        ld = self.config.lang_hidden_dim
        hd = self.config.fusion_hidden_dim
        
        # 视觉投影 (vision + lidar concat -> fusion hidden)
        self.vision_proj_w = np.random.randn(vd + lidar_d, hd).astype(np.float32) * np.sqrt(2.0 / (vd + lidar_d))
        self.vision_proj_b = np.zeros(hd, dtype=np.float32)
        
        # 语言投影
        self.lang_proj_w = np.random.randn(ld, hd).astype(np.float32) * np.sqrt(2.0 / ld)
        self.lang_proj_b = np.zeros(hd, dtype=np.float32)
        
        # 多模态注意力 (简化)
        self.fusion_attn_w = np.random.randn(hd, hd).astype(np.float32) * np.sqrt(2.0 / hd)
        self.fusion_attn_b = np.zeros(hd, dtype=np.float32)
        
        # 输出归一化
        self.fusion_norm_gamma = np.ones(hd, dtype=np.float32)
        self.fusion_norm_beta = np.zeros(hd, dtype=np.float32)
        self.fusion_norm_mean = np.zeros(hd, dtype=np.float32)
        self.fusion_norm_var = np.ones(hd, dtype=np.float32)
    
    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                    mean: np.ndarray, var: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Layer Normalization"""
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta
    
    def _fuse(self, vision_feat: np.ndarray, lang_feat: np.ndarray) -> np.ndarray:
        """
        融合视觉和语言特征
        
        Args:
            vision_feat: (vision_dim,)
            lang_feat: (lang_hidden_dim,)
            
        Returns:
            fused: (fusion_hidden_dim,)
        """
        # 投影到统一空间
        v_proj = np.dot(vision_feat, self.vision_proj_w) + self.vision_proj_b
        l_proj = np.dot(lang_feat, self.lang_proj_w) + self.lang_proj_b
        
        # 元素级交叉注意力 (简化)
        cross = v_proj * l_proj  # 视觉-语言交互
        attn = np.tanh(np.dot(cross, self.fusion_attn_w) + self.fusion_attn_b)
        
        # 融合: 视觉 + 语言 + 交互
        fused = v_proj + l_proj + attn
        
        # Layer Norm
        fused = self._layer_norm(
            fused,
            self.fusion_norm_gamma,
            self.fusion_norm_beta,
            self.fusion_norm_mean,
            self.fusion_norm_var,
        )
        
        return fused.astype(np.float32)
    
    def start(self):
        """启动VLA模型"""
        self._is_running = True
        self._history.clear()
        self._total_inferences = 0
        logger.info("VLAModel started")
    
    def stop(self):
        """停止VLA模型"""
        self._is_running = False
        logger.info(f"VLAModel stopped. Total inferences: {self._total_inferences}")
    
    def step(self, input_data: VLAInput) -> VLAOutput:
        """
        单步推理
        
        Args:
            input_data: VLAInput 包含感知帧和历史动作
            
        Returns:
            VLAOutput 包含预测的动作
        """
        t0 = time.time()
        
        perception = input_data.perception
        
        # 1. Vision Encoding (vision + lidar)
        vision_feat = self.vision_encoder.encode_full(perception)
        
        # 2. Language Encoding
        if perception.instruction:
            lang_feat = self.lang_encoder.encode(perception.instruction)
        else:
            lang_feat = np.zeros(self.config.lang_hidden_dim, dtype=np.float32)
        
        # 3. Cross-Modal Fusion
        fused = self._fuse(vision_feat, lang_feat)
        
        # 4. Action Decoding
        history = input_data.history_actions or self._history[-5:]
        actions, logits = self.action_decoder.decode(
            fused,
            history_actions=history,
            num_steps=1,
        )
        
        primary_action = actions[0] if actions else VLAAction()
        primary_action.confidence = float(np.tanh(logits.mean()) * 0.5 + 0.5) if len(logits) > 0 else 1.0
        primary_action.attention_weights = {
            'vision': float(np.linalg.norm(vision_feat) / (np.linalg.norm(vision_feat) + np.linalg.norm(lang_feat) + 1e-6)),
            'language': float(np.linalg.norm(lang_feat) / (np.linalg.norm(vision_feat) + np.linalg.norm(lang_feat) + 1e-6)),
        }
        
        # 5. 多步预测
        if self.config.max_decode_steps > 1:
            multi_actions, _ = self.action_decoder.decode(
                fused,
                history_actions=history,
                num_steps=min(self.config.action_seq_len, self.config.max_decode_steps),
            )
        else:
            multi_actions = [primary_action]
        
        # 安全检查: 碰撞风险评估
        collision_risk = self._assess_collision_risk(primary_action, perception)
        safety_override = collision_risk > 0.8
        
        if safety_override:
            primary_action = self._get_safe_fallback_action(primary_action, perception)
            primary_action.confidence = 0.0
            primary_action.reasoning = "Safety override: collision risk detected"
        
        # 更新历史
        self._history.append(primary_action)
        if len(self._history) > self.config.max_decode_steps * 2:
            self._history = self._history[-self.config.max_decode_steps * 2:]
        
        self._total_inferences += 1
        inference_time_ms = (time.time() - t0) * 1000.0
        
        return VLAOutput(
            action=primary_action,
            action_sequence=multi_actions,
            fused_features=fused,
            vision_features=vision_feat,
            language_features=lang_feat,
            action_logits=logits,
            collision_risk=collision_risk,
            safety_override=safety_override,
            fallback_action=primary_action if safety_override else None,
            inference_time_ms=inference_time_ms,
        )
    
    def _assess_collision_risk(self, action: VLAAction, perception: VLAPerceptionFrame) -> float:
        """评估动作的碰撞风险"""
        risk = 0.0
        
        # 基于速度评估
        speed = math.sqrt(action.vx**2 + action.vy**2 + action.vz**2)
        if speed > 1.5:
            risk += 0.3
        
        # 基于激光雷达
        if perception.lidar_scan is not None:
            min_range = perception.lidar_scan.min()
            if min_range < 0.3:
                risk += 0.6
            elif min_range < 0.5:
                risk += 0.3
            elif min_range < 1.0:
                risk += 0.1
        
        # 基于旋转
        rot_speed = math.sqrt(action.rx**2 + action.ry**2 + action.rz**2)
        if rot_speed > math.pi / 4:
            risk += 0.2
        
        return min(1.0, risk)
    
    def _get_safe_fallback_action(self, action: VLAAction, perception: VLAPerceptionFrame) -> VLAAction:
        """生成安全的回退动作"""
        safe = VLAAction()
        safe.vx = 0.0
        safe.vy = 0.0
        safe.vz = 0.0
        safe.rx = 0.0
        safe.ry = 0.0
        safe.rz = 0.0
        safe.gripper_position = action.gripper_position  # 保持夹爪状态
        safe.action_space = self.config.action_space
        safe.reasoning = "Safe fallback: reduced speed due to collision risk"
        
        # 如果 lidar 显示障碍物近，使用最小速度
        if perception.lidar_scan is not None:
            min_range = perception.lidar_scan.min()
            if min_range < 0.3:
                safe.confidence = 0.0  # 完全停止
            else:
                safe.vx = 0.1  # 缓慢移动
        else:
            safe.vx = 0.1  # 保守前进
        
        return safe
    
    def batch_step(self, inputs: List[VLAInput]) -> List[VLAOutput]:
        """批量推理"""
        return [self.step(inp) for inp in inputs]
    
    def reset_history(self):
        """重置历史动作"""
        self._history.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取模型统计"""
        return {
            'total_inferences': self._total_inferences,
            'history_len': len(self._history),
            'grade': self.config.grade,
            'action_space': self.config.action_space.value,
            'is_running': self._is_running,
        }


# ============================================================
# 工厂函数
# ============================================================

def create_vla_model(grade: str = "M", action_space: Optional[VLAActionSpace] = None) -> VLAModel:
    """创建VLA模型"""
    config = VLAConfig(grade=grade.upper())
    if action_space:
        config.action_space = action_space
    return VLAModel(config=config)


def load_vla_model(path: str, grade: Optional[str] = None) -> VLAModel:
    """从文件加载VLA模型 (占位)"""
    # 实际实现应包含模型权重加载逻辑
    g = grade or "M"
    model = create_vla_model(grade=g)
    logger.info(f"VLA model loaded from {path} (placeholder)")
    return model
