"""
融合模块测试
=============

测试跨模态融合网络:
- CrossModalAttention
- ModalityEncoder
- CrossModalFusion
- UnifiedRepresentation
"""

import numpy as np
import torch
import sys
import unittest

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from fusion.cross_modal_fusion import (
    CrossModalFusion, CrossModalAttention, ModalityEncoder,
    FusionConfig, FusionStrategy, MultimodalInput, UnifiedRepresentation,
    create_multimodal_input
)


class TestCrossModalAttention(unittest.TestCase):
    """测试跨模态注意力"""
    
    def setUp(self):
        self.batch_size = 2
        self.query_dim = 128
        self.key_dim = 128
        self.value_dim = 128
        self.num_heads = 4
        
        self.attn = CrossModalAttention(
            query_dim=self.query_dim,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            num_heads=self.num_heads
        )
    
    def test_attention_shapes(self):
        B = self.batch_size
        N = 10  # query序列长度
        M = 20  # key序列长度
        
        query = torch.randn(B, N, self.query_dim)
        key = torch.randn(B, M, self.key_dim)
        value = torch.randn(B, M, self.value_dim)
        
        out = self.attn(query, key, value)
        
        self.assertEqual(out.shape, (B, N, self.query_dim))
    
    def test_attention_with_mask(self):
        B = self.batch_size
        N, M = 10, 20
        
        query = torch.randn(B, N, self.query_dim)
        key = torch.randn(B, M, self.key_dim)
        value = torch.randn(B, M, self.value_dim)
        mask = torch.rand(B, N, M) > 0.5  # 随机掩码
        
        out = self.attn(query, key, value, mask=mask)
        self.assertEqual(out.shape, (B, N, self.query_dim))
    
    def test_attention_gradients(self):
        B = 2
        N, M = 5, 8
        
        query = torch.randn(B, N, self.query_dim, requires_grad=True)
        key = torch.randn(B, M, self.key_dim)
        value = torch.randn(B, M, self.value_dim)
        
        out = self.attn(query, key, value)
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(query.grad)
        self.assertFalse(torch.isnan(query.grad).any())


class TestModalityEncoder(unittest.TestCase):
    """测试单模态编码器"""
    
    def test_encoder_shapes(self):
        B = 4
        input_dim = 64
        output_dim = 128
        
        encoder = ModalityEncoder('test', input_dim, output_dim)
        x = torch.randn(B, input_dim)
        out = encoder(x)
        
        self.assertEqual(out.shape, (B, output_dim))
    
    def test_encoder_trainable(self):
        encoder = ModalityEncoder('test', 64, 128)
        x = torch.randn(2, 64)
        
        out1 = encoder(x)
        loss = out1.sum()
        loss.backward()
        
        # 检查参数有梯度
        for param in encoder.parameters():
            self.assertIsNotNone(param.grad)


class TestMultimodalInput(unittest.TestCase):
    """测试多模态输入数据类"""
    
    def test_multimodal_input_empty(self):
        inp = MultimodalInput()
        self.assertEqual(inp.modalities, [])
    
    def test_multimodal_input_partial(self):
        vision = torch.randn(2, 3, 224, 224)
        audio = torch.randn(2, 100, 128)
        
        inp = MultimodalInput(vision=vision, audio=audio)
        mods = inp.modalities
        
        self.assertIn('vision', mods)
        self.assertIn('audio', mods)
        self.assertNotIn('tactile', mods)
        self.assertNotIn('force', mods)
    
    def test_multimodal_input_all(self):
        inp = MultimodalInput(
            vision=torch.randn(2, 512),
            audio=torch.randn(2, 128),
            tactile=torch.randn(2, 64),
            force=torch.randn(2, 6),
            imu=torch.randn(2, 9)
        )
        
        mods = inp.modalities
        self.assertEqual(len(mods), 5)
        for m in ['vision', 'audio', 'tactile', 'force', 'imu']:
            self.assertIn(m, mods)


class TestFusionConfig(unittest.TestCase):
    """测试融合配置"""
    
    def test_default_config(self):
        config = FusionConfig()
        self.assertEqual(config.hidden_dim, 256)
        self.assertEqual(config.num_heads, 4)
        self.assertEqual(config.num_layers, 2)
        self.assertEqual(config.strategy, FusionStrategy.HYBRID)
    
    def test_custom_config(self):
        config = FusionConfig(
            hidden_dim=512,
            num_heads=8,
            strategy=FusionStrategy.EARLY
        )
        self.assertEqual(config.hidden_dim, 512)
        self.assertEqual(config.num_heads, 8)
        self.assertEqual(config.strategy, FusionStrategy.EARLY)


class TestCrossModalFusion(unittest.TestCase):
    """测试跨模态融合网络"""
    
    def setUp(self):
        self.config = FusionConfig(
            vision_dim=512,
            audio_dim=128,
            tactile_dim=64,
            force_dim=32,
            imu_dim=64,
            hidden_dim=256,
            num_heads=4,
            num_layers=2
        )
        self.fusion = CrossModalFusion(self.config)
    
    def test_fusion_with_vision_only(self):
        B = 2
        vision = torch.randn(B, 512)
        
        multimodal = MultimodalInput(vision=vision)
        out = self.fusion(multimodal)
        
        self.assertEqual(out.shape, (B, self.config.hidden_dim))
    
    def test_fusion_with_all_modalities(self):
        B = 2
        multimodal = MultimodalInput(
            vision=torch.randn(B, 512),
            audio=torch.randn(B, 128),
            tactile=torch.randn(B, 64),
            force=torch.randn(B, 32),
            imu=torch.randn(B, 64)
        )
        
        out = self.fusion(multimodal)
        self.assertEqual(out.shape, (B, self.config.hidden_dim))
    
    def test_fusion_gradients(self):
        B = 2
        multimodal = MultimodalInput(
            vision=torch.randn(B, 512, requires_grad=True),
            audio=torch.randn(B, 128, requires_grad=True)
        )
        
        out = self.fusion(multimodal)
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(multimodal.vision.grad)
        self.assertIsNotNone(multimodal.audio.grad)
    
    def test_fusion_with_missing_modalities(self):
        B = 2
        # 故意只提供部分模态
        multimodal = MultimodalInput(
            vision=torch.randn(B, 512),
            tactile=torch.randn(B, 64)
        )
        
        out = self.fusion(multimodal)
        self.assertEqual(out.shape, (B, self.config.hidden_dim))
    
    def test_fusion_empty_input_error(self):
        multimodal = MultimodalInput()
        
        with self.assertRaises(ValueError):
            self.fusion(multimodal)


class TestUnifiedRepresentation(unittest.TestCase):
    """测试统一表示学习"""
    
    def setUp(self):
        self.input_dim = 256
        self.hidden_dim = 256
        self.output_dim = 128
        
        self.unified = UnifiedRepresentation(
            self.input_dim,
            self.hidden_dim,
            self.output_dim
        )
    
    def test_unified_output_shapes(self):
        B = 4
        fused = torch.randn(B, self.input_dim)
        
        state, action, world = self.unified(fused)
        
        self.assertEqual(state.shape, (B, self.output_dim))
        self.assertEqual(action.shape, (B, self.output_dim))
        self.assertEqual(world.shape, (B, self.output_dim))
    
    def test_unified_gradients(self):
        B = 2
        fused = torch.randn(B, self.input_dim, requires_grad=True)
        
        state, action, world = self.unified(fused)
        loss = state.sum() + action.sum() + world.sum()
        loss.backward()
        
        self.assertIsNotNone(fused.grad)
    
    def test_unified_different_heads(self):
        B = 4
        fused = torch.randn(B, self.input_dim)
        
        state, action, world = self.unified(fused)
        
        # 三种表示应该不同
        self.assertFalse(torch.allclose(state, action))
        self.assertFalse(torch.allclose(state, world))
        self.assertFalse(torch.allclose(action, world))


class TestCreateMultimodalInput(unittest.TestCase):
    """测试创建多模态输入"""
    
    def test_numpy_to_tensor(self):
        vision = np.random.randn(2, 512).astype(np.float32)
        audio = np.random.randn(2, 128).astype(np.float32)
        
        multimodal = create_multimodal_input(vision=vision, audio=audio)
        
        self.assertIsInstance(multimodal.vision, torch.Tensor)
        self.assertIsInstance(multimodal.audio, torch.Tensor)
        self.assertEqual(multimodal.vision.shape, (2, 512))
    
    def test_none_handling(self):
        multimodal = create_multimodal_input(vision=None, audio=None)
        
        self.assertIsNone(multimodal.vision)
        self.assertIsNone(multimodal.audio)


class TestFusionIntegration(unittest.TestCase):
    """融合模块集成测试"""
    
    def test_full_pipeline(self):
        """测试完整融合流程"""
        B = 4
        
        # 创建融合网络
        config = FusionConfig(hidden_dim=256, num_heads=4, num_layers=2)
        fusion = CrossModalFusion(config)
        
        # 创建统一表示
        unified = UnifiedRepresentation(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            output_dim=128
        )
        
        # 模拟多模态输入
        multimodal = MultimodalInput(
            vision=torch.randn(B, 512),
            audio=torch.randn(B, 128),
            tactile=torch.randn(B, 64),
            force=torch.randn(B, 32),
            imu=torch.randn(B, 64)
        )
        
        # 前向传播
        fused = fusion(multimodal)
        state, action, world = unified(fused)
        
        # 验证输出
        self.assertEqual(state.shape, (B, 128))
        self.assertEqual(action.shape, (B, 128))
        self.assertEqual(world.shape, (B, 128))
        
        # 检查数值稳定性
        self.assertFalse(torch.isnan(state).any())
        self.assertFalse(torch.isnan(action).any())
        self.assertFalse(torch.isnan(world).any())
    
    def test_partial_modality_fusion(self):
        """测试部分模态融合"""
        B = 2
        config = FusionConfig(hidden_dim=256)
        fusion = CrossModalFusion(config)
        
        # 只用视觉和触觉
        multimodal = MultimodalInput(
            vision=torch.randn(B, 512),
            tactile=torch.randn(B, 64)
        )
        
        fused = fusion(multimodal)
        self.assertEqual(fused.shape, (B, config.hidden_dim))
        
        # 验证注意力交互发生 (输出不为零)
        self.assertFalse(torch.allclose(fused, torch.zeros_like(fused)))


class TestCrossModalAttentionVariants(unittest.TestCase):
    """测试不同配置的注意力"""
    
    def test_different_head_counts(self):
        B = 2
        for num_heads in [1, 2, 4, 8]:
            attn = CrossModalAttention(128, 128, 128, num_heads=num_heads)
            out = attn(torch.randn(B, 10, 128), torch.randn(B, 20, 128), torch.randn(B, 20, 128))
            self.assertEqual(out.shape[0], B)
            self.assertEqual(out.shape[1], 10)
            self.assertEqual(out.shape[2], 128)
    
    def test_different_dimensions(self):
        B = 2
        attn = CrossModalAttention(query_dim=256, key_dim=128, value_dim=64, num_heads=4)
        out = attn(
            torch.randn(B, 10, 256),
            torch.randn(B, 20, 128),
            torch.randn(B, 20, 64)
        )
        self.assertEqual(out.shape, (B, 10, 256))


if __name__ == '__main__':
    # 检查 CUDA 是否可用
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, running on CPU")
    
    # 运行测试
    unittest.main(verbosity=2)
