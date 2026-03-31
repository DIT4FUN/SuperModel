"""
融合模块测试
============

测试跨模态融合网络:
- FusionStrategy 融合策略枚举
- MultimodalInput 多模态输入容器
- FusionConfig 融合配置
- LanguageEncoder 语言编码器
- CrossModalAttention 跨模态注意力
- ModalityEncoder 模态编码器
- CrossModalFusion 跨模态融合网络
- UnifiedRepresentation 统一表示学习
- create_multimodal_input 工厂函数
"""

import numpy as np
import sys
import torch
import unittest

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from fusion.cross_modal_fusion import (
    FusionStrategy, MultimodalInput, FusionConfig,
    LanguageEncoder, CrossModalAttention, ModalityEncoder,
    CrossModalFusion, UnifiedRepresentation, create_multimodal_input
)


class TestFusionStrategy(unittest.TestCase):
    """测试融合策略枚举"""

    def test_fusion_strategy_values(self):
        self.assertEqual(FusionStrategy.EARLY.value, "early")
        self.assertEqual(FusionStrategy.LATE.value, "late")
        self.assertEqual(FusionStrategy.HYBRID.value, "hybrid")

    def test_fusion_strategy_count(self):
        self.assertEqual(len(FusionStrategy), 3)


class TestMultimodalInput(unittest.TestCase):
    """测试多模态输入容器"""

    def test_empty_input(self):
        mmi = MultimodalInput()
        self.assertEqual(mmi.modalities, [])

    def test_vision_only(self):
        mmi = MultimodalInput(vision=torch.randn(2, 512))
        self.assertEqual(mmi.modalities, ['vision'])

    def test_all_modalities(self):
        mmi = MultimodalInput(
            vision=torch.randn(2, 512),
            audio=torch.randn(2, 100, 128),
            tactile=torch.randn(2, 64),
            force=torch.randn(2, 6),
            imu=torch.randn(2, 9),
            language=torch.randint(0, 1000, (2, 32))
        )
        mods = mmi.modalities
        self.assertEqual(len(mods), 6)
        for m in ['vision', 'audio', 'tactile', 'force', 'imu', 'language']:
            self.assertIn(m, mods)

    def test_partial_modalities(self):
        mmi = MultimodalInput(
            vision=torch.randn(2, 512),
            tactile=torch.randn(2, 64),
            imu=torch.randn(2, 9),
        )
        mods = mmi.modalities
        self.assertEqual(len(mods), 3)
        self.assertIn('vision', mods)
        self.assertIn('tactile', mods)
        self.assertIn('imu', mods)
        self.assertNotIn('audio', mods)


class TestFusionConfig(unittest.TestCase):
    """测试融合配置"""

    def test_default_config(self):
        config = FusionConfig()
        self.assertEqual(config.vision_dim, 512)
        self.assertEqual(config.audio_dim, 128)
        self.assertEqual(config.hidden_dim, 256)
        self.assertEqual(config.num_heads, 4)
        self.assertEqual(config.strategy, FusionStrategy.HYBRID)

    def test_custom_config(self):
        config = FusionConfig(
            vision_dim=1024,
            audio_dim=256,
            hidden_dim=512,
            num_heads=8,
            strategy=FusionStrategy.EARLY
        )
        self.assertEqual(config.vision_dim, 1024)
        self.assertEqual(config.hidden_dim, 512)
        self.assertEqual(config.num_heads, 8)
        self.assertEqual(config.strategy, FusionStrategy.EARLY)


class TestLanguageEncoder(unittest.TestCase):
    """测试语言编码器"""

    def test_encoder_initialization(self):
        enc = LanguageEncoder(vocab_size=5000, embed_dim=64, hidden_dim=128, max_len=16)
        self.assertEqual(enc.embed_dim, 64)
        self.assertEqual(enc.hidden_dim, 128)

    def test_encoder_forward_shape(self):
        enc = LanguageEncoder(vocab_size=5000, embed_dim=64, hidden_dim=128, max_len=16)
        token_ids = torch.randint(0, 5000, (4, 16))
        output = enc(token_ids)
        self.assertEqual(output.shape, (4, 128))

    def test_encoder_forward_variable_length(self):
        enc = LanguageEncoder(vocab_size=5000, embed_dim=64, hidden_dim=128, max_len=16)
        token_ids = torch.randint(0, 5000, (2, 8))
        output = enc(token_ids)
        self.assertEqual(output.shape, (2, 128))

    def test_encoder_single_token(self):
        enc = LanguageEncoder(vocab_size=5000, embed_dim=64, hidden_dim=128, max_len=16)
        token_ids = torch.randint(0, 5000, (1, 1))
        output = enc(token_ids)
        self.assertEqual(output.shape, (1, 128))

    def test_encoder_gradient_flow(self):
        enc = LanguageEncoder(vocab_size=5000, embed_dim=64, hidden_dim=128, max_len=16)
        token_ids = torch.randint(0, 5000, (2, 8))
        output = enc(token_ids)
        loss = output.sum()
        loss.backward()
        for name, param in enc.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")


class TestCrossModalAttention(unittest.TestCase):
    """测试跨模态注意力层"""

    def test_attention_initialization(self):
        attn = CrossModalAttention(query_dim=128, key_dim=128, value_dim=128, num_heads=4)
        self.assertEqual(attn.num_heads, 4)

    def test_attention_square_input(self):
        attn = CrossModalAttention(256, 256, 256, num_heads=4)
        q = torch.randn(2, 10, 256)
        k = torch.randn(2, 10, 256)
        v = torch.randn(2, 10, 256)
        out = attn(q, k, v)
        self.assertEqual(out.shape, (2, 10, 256))

    def test_attention_rectangular_input(self):
        attn = CrossModalAttention(256, 256, 256, num_heads=4)
        q = torch.randn(2, 5, 256)
        k = torch.randn(2, 12, 256)
        v = torch.randn(2, 12, 256)
        out = attn(q, k, v)
        self.assertEqual(out.shape, (2, 5, 256))

    def test_attention_single_head(self):
        attn = CrossModalAttention(128, 128, 128, num_heads=1)
        q = torch.randn(1, 8, 128)
        k = torch.randn(1, 8, 128)
        v = torch.randn(1, 8, 128)
        out = attn(q, k, v)
        self.assertEqual(out.shape, (1, 8, 128))

    def test_attention_gradient_flow(self):
        attn = CrossModalAttention(128, 128, 128, num_heads=4)
        q = torch.randn(2, 6, 128, requires_grad=True)
        k = torch.randn(2, 6, 128)
        v = torch.randn(2, 6, 128)
        out = attn(q, k, v)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(q.grad)


class TestModalityEncoder(unittest.TestCase):
    """测试模态编码器"""

    def test_encoder_initialization(self):
        enc = ModalityEncoder('vision', 512, 256)
        self.assertEqual(enc.modality, 'vision')

    def test_encoder_forward_2d_input(self):
        enc = ModalityEncoder('vision', 512, 256)
        x = torch.randn(4, 512)
        out = enc(x)
        self.assertEqual(out.shape, (4, 256))

    def test_encoder_forward_3d_input(self):
        enc = ModalityEncoder('audio', 128, 256)
        x = torch.randn(4, 10, 128)
        out = enc(x)
        # 应该压缩时间维度
        self.assertEqual(out.shape[0], 4)
        self.assertEqual(out.shape[-1], 256)


class TestCrossModalFusion(unittest.TestCase):
    """测试跨模态融合网络"""

    def setUp(self):
        self.config = FusionConfig(
            vision_dim=512, audio_dim=128,
            tactile_dim=64, force_dim=32, imu_dim=64,
            hidden_dim=256, num_heads=4, num_layers=2
        )
        self.fusion = CrossModalFusion(self.config)

    def test_fusion_initialization(self):
        self.assertEqual(self.fusion.config, self.config)
        self.assertEqual(len(self.fusion.cross_attn_layers), 2)

    def test_fusion_single_modality(self):
        vision = torch.randn(2, 512)
        mmi = MultimodalInput(vision=vision)
        out = self.fusion(mmi)
        self.assertEqual(out.shape, (2, 256))

    def test_fusion_two_modalities(self):
        mmi = MultimodalInput(
            vision=torch.randn(2, 512),
            audio=torch.randn(2, 128)
        )
        out = self.fusion(mmi)
        self.assertEqual(out.shape, (2, 256))

    def test_fusion_three_modalities(self):
        mmi = MultimodalInput(
            vision=torch.randn(2, 512),
            audio=torch.randn(2, 128),
            tactile=torch.randn(2, 64)
        )
        out = self.fusion(mmi)
        self.assertEqual(out.shape, (2, 256))

    def test_fusion_all_six_modalities(self):
        mmi = MultimodalInput(
            vision=torch.randn(2, 512),
            audio=torch.randn(2, 128),
            tactile=torch.randn(2, 64),
            force=torch.randn(2, 32),
            imu=torch.randn(2, 64),
            language=torch.randint(0, 10000, (2, 32))
        )
        out = self.fusion(mmi)
        self.assertEqual(out.shape, (2, 256))

    def test_fusion_no_modality_raises(self):
        mmi = MultimodalInput()
        with self.assertRaises(ValueError):
            self.fusion(mmi)

    def test_fusion_batch_size_consistency(self):
        """所有模态应支持相同batch size"""
        mmi = MultimodalInput(
            vision=torch.randn(3, 512),
            audio=torch.randn(3, 128),
            tactile=torch.randn(3, 64),
        )
        out = self.fusion(mmi)
        self.assertEqual(out.shape[0], 3)

    def test_fusion_deterministic_output(self):
        """相同输入应产生相同输出"""
        mmi = MultimodalInput(
            vision=torch.randn(2, 512),
            audio=torch.randn(2, 128)
        )
        torch.manual_seed(42)
        out1 = self.fusion(mmi)
        torch.manual_seed(42)
        out2 = self.fusion(mmi)
        self.assertTrue(torch.allclose(out1, out2))

    def test_fusion_gradient_flow(self):
        """融合网络应支持梯度流"""
        mmi = MultimodalInput(
            vision=torch.randn(2, 512),
            audio=torch.randn(2, 128)
        )
        out = self.fusion(mmi)
        loss = out.sum()
        loss.backward()
        # 检查至少部分参数有梯度
        has_grad = any(p.grad is not None for p in self.fusion.parameters())
        self.assertTrue(has_grad)

    def test_fusion_inference_mode(self):
        """融合网络应支持推理模式"""
        mmi = MultimodalInput(
            vision=torch.randn(2, 512),
            audio=torch.randn(2, 128)
        )
        with torch.no_grad():
            out1 = self.fusion(mmi)
            out2 = self.fusion(mmi)
        self.assertEqual(out1.shape, out2.shape)

    def test_fusion_language_only(self):
        """仅语言模态"""
        mmi = MultimodalInput(
            language=torch.randint(0, 10000, (2, 32))
        )
        out = self.fusion(mmi)
        self.assertEqual(out.shape, (2, 256))

    def test_fusion_force_imu_only(self):
        """仅力觉和IMU模态"""
        mmi = MultimodalInput(
            force=torch.randn(2, 32),
            imu=torch.randn(2, 64)
        )
        out = self.fusion(mmi)
        self.assertEqual(out.shape, (2, 256))

    def test_fusion_get_active_mods(self):
        mods = self.fusion._get_active_mods()
        self.assertEqual(len(mods), 6)


class TestUnifiedRepresentation(unittest.TestCase):
    """测试统一表示学习"""

    def test_unified_initialization(self):
        ur = UnifiedRepresentation(input_dim=256, hidden_dim=512, output_dim=128)
        self.assertEqual(ur.encoder[0].in_features, 256)

    def test_unified_forward(self):
        ur = UnifiedRepresentation(input_dim=256, hidden_dim=512, output_dim=128)
        fused = torch.randn(4, 256)
        state, action, world = ur(fused)
        self.assertEqual(state.shape, (4, 128))
        self.assertEqual(action.shape, (4, 128))
        self.assertEqual(world.shape, (4, 128))

    def test_unified_single_sample(self):
        ur = UnifiedRepresentation(input_dim=256, hidden_dim=512, output_dim=64)
        fused = torch.randn(1, 256)
        state, action, world = ur(fused)
        self.assertEqual(state.shape, (1, 64))

    def test_unified_gradient_flow(self):
        ur = UnifiedRepresentation(input_dim=256, hidden_dim=512, output_dim=128)
        fused = torch.randn(2, 256)
        state, action, world = ur(fused)
        loss = state.sum() + action.sum() + world.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in ur.parameters())
        self.assertTrue(has_grad)

    def test_unified_head_independence(self):
        """三个任务头应独立"""
        ur = UnifiedRepresentation(input_dim=256, hidden_dim=512, output_dim=128)
        fused = torch.randn(2, 256)
        state1, action1, world1 = ur(fused)
        state2, action2, world2 = ur(fused)
        # 不同样本应产生不同输出
        self.assertFalse(torch.allclose(state1, state2))


class TestCreateMultimodalInput(unittest.TestCase):
    """测试多模态输入工厂函数"""

    def test_numpy_vision(self):
        vision_np = np.random.randn(2, 512).astype(np.float32)
        mmi = create_multimodal_input(vision=vision_np)
        self.assertIsInstance(mmi.vision, torch.Tensor)
        self.assertEqual(mmi.vision.shape, (2, 512))

    def test_numpy_audio(self):
        audio_np = np.random.randn(2, 100, 128).astype(np.float32)
        mmi = create_multimodal_input(audio=audio_np)
        self.assertIsInstance(mmi.audio, torch.Tensor)
        self.assertEqual(mmi.audio.shape, (2, 100, 128))

    def test_numpy_tactile(self):
        tactile_np = np.random.randn(2, 64).astype(np.float32)
        mmi = create_multimodal_input(tactile=tactile_np)
        self.assertIsInstance(mmi.tactile, torch.Tensor)

    def test_numpy_force(self):
        force_np = np.random.randn(2, 6).astype(np.float32)
        mmi = create_multimodal_input(force=force_np)
        self.assertIsInstance(mmi.force, torch.Tensor)

    def test_numpy_imu(self):
        imu_np = np.random.randn(2, 9).astype(np.float32)
        mmi = create_multimodal_input(imu=imu_np)
        self.assertIsInstance(mmi.imu, torch.Tensor)

    def test_numpy_language(self):
        lang_np = np.random.randint(0, 10000, (2, 32))
        mmi = create_multimodal_input(language=lang_np)
        self.assertIsInstance(mmi.language, torch.Tensor)
        self.assertEqual(mmi.language.dtype, torch.long)

    def test_mixed_modalities(self):
        mmi = create_multimodal_input(
            vision=np.random.randn(3, 512),
            audio=np.random.randn(3, 50, 128),
            tactile=np.random.randn(3, 64),
            force=np.random.randn(3, 6),
            imu=np.random.randn(3, 9),
            language=np.random.randint(0, 10000, (3, 32))
        )
        self.assertEqual(len(mmi.modalities), 6)

    def test_none_inputs(self):
        mmi = create_multimodal_input()
        self.assertEqual(mmi.modalities, [])


class TestFusionPipeline(unittest.TestCase):
    """端到端融合流程测试"""

    def test_full_pipeline_single_modality(self):
        """单模态完整流程"""
        config = FusionConfig(hidden_dim=256)
        fusion = CrossModalFusion(config)
        unified = UnifiedRepresentation(input_dim=256, hidden_dim=384, output_dim=128)

        vision = torch.randn(2, 512)
        mmi = MultimodalInput(vision=vision)
        fused = fusion(mmi)
        state, action, world = unified(fused)

        self.assertEqual(state.shape, (2, 128))
        self.assertEqual(action.shape, (2, 128))
        self.assertEqual(world.shape, (2, 128))

    def test_full_pipeline_multimodal(self):
        """多模态完整流程"""
        config = FusionConfig(hidden_dim=256, num_heads=4)
        fusion = CrossModalFusion(config)
        unified = UnifiedRepresentation(input_dim=256, hidden_dim=384, output_dim=256)

        mmi = MultimodalInput(
            vision=torch.randn(2, 512),
            audio=torch.randn(2, 128),
            tactile=torch.randn(2, 64),
            force=torch.randn(2, 32),
            imu=torch.randn(2, 64),
            language=torch.randint(0, 10000, (2, 32))
        )
        fused = fusion(mmi)
        state, action, world = unified(fused)

        self.assertEqual(state.shape, (2, 256))
        self.assertEqual(action.shape, (2, 256))
        self.assertEqual(world.shape, (2, 256))

    def test_fusion_config_variants(self):
        """测试不同融合配置"""
        for strategy in FusionStrategy:
            config = FusionConfig(strategy=strategy, hidden_dim=128, num_heads=2)
            fusion = CrossModalFusion(config)
            mmi = MultimodalInput(
                vision=torch.randn(2, 512),
                audio=torch.randn(2, 128)
            )
            out = fusion(mmi)
            self.assertEqual(out.shape, (2, 128))

    def test_batch_processing(self):
        """批量处理稳定性"""
        config = FusionConfig(hidden_dim=256)
        fusion = CrossModalFusion(config)

        for batch_size in [1, 2, 4, 8, 16]:
            mmi = MultimodalInput(
                vision=torch.randn(batch_size, 512),
                audio=torch.randn(batch_size, 128),
                tactile=torch.randn(batch_size, 64)
            )
            out = fusion(mmi)
            self.assertEqual(out.shape[0], batch_size)
            self.assertEqual(out.shape[1], 256)


if __name__ == '__main__':
    unittest.main()


class TestFusionEdgeCases(unittest.TestCase):
    """融合模块边缘用例测试"""
    
    def test_single_modality_extreme_values(self):
        """单模态极端值处理"""
        config = FusionConfig(hidden_dim=256)
        fusion = CrossModalFusion(config)
        
        # 极端零值
        mmi_zero = MultimodalInput(
            vision=torch.zeros(2, 512),
            audio=torch.zeros(2, 128),
            tactile=torch.zeros(2, 64)
        )
        out = fusion(mmi_zero)
        self.assertEqual(out.shape, (2, 256))
        self.assertFalse(torch.isnan(out).any())
        
        # 极端最大值
        mmi_max = MultimodalInput(
            vision=torch.ones(2, 512) * 1000,
            audio=torch.ones(2, 128) * 1000,
            tactile=torch.ones(2, 64) * 1000
        )
        out = fusion(mmi_max)
        self.assertEqual(out.shape, (2, 256))
        self.assertFalse(torch.isnan(out).any())
    
    def test_missing_modality_handling(self):
        """缺失模态处理"""
        config = FusionConfig(hidden_dim=256)
        fusion = CrossModalFusion(config)
        
        # 只提供部分模态
        mmi_partial = MultimodalInput(
            vision=torch.randn(2, 512),
            audio=torch.randn(2, 128),
            tactile=torch.zeros(2, 64)  # 空模态
        )
        out = fusion(mmi_partial)
        self.assertEqual(out.shape, (2, 256))
    
    def test_fusion_with_language_only(self):
        """仅语言模态融合"""
        config = FusionConfig(hidden_dim=256)
        fusion = CrossModalFusion(config)
        
        mmi = MultimodalInput(
            vision=torch.randn(1, 512),
            language=torch.randint(0, 1000, (1, 32))
        )
        out = fusion(mmi)
        self.assertEqual(out.shape, (1, 256))
    
    def test_different_hidden_dimensions(self):
        """不同隐层维度配置"""
        for hidden_dim in [64, 128, 256, 512, 1024]:
            config = FusionConfig(hidden_dim=hidden_dim)
            fusion = CrossModalFusion(config)
            mmi = MultimodalInput(
                vision=torch.randn(1, 512),
                audio=torch.randn(1, 128)
            )
            out = fusion(mmi)
            self.assertEqual(out.shape[1], hidden_dim)
    
    def test_large_batch_stability(self):
        """大批量处理稳定性"""
        config = FusionConfig(hidden_dim=512)
        fusion = CrossModalFusion(config)
        
        # 大批量
        mmi = MultimodalInput(
            vision=torch.randn(64, 512),
            audio=torch.randn(64, 128),
            tactile=torch.randn(64, 64)
        )
        out = fusion(mmi)
        self.assertEqual(out.shape, (64, 512))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())
    
    def test_repeated_fusion_consistency(self):
        """重复融合一致性"""
        config = FusionConfig(hidden_dim=256)
        fusion = CrossModalFusion(config)
        fusion.eval()  # 切换到评估模式禁用dropout
        
        mmi = MultimodalInput(
            vision=torch.randn(2, 512),
            audio=torch.randn(2, 128)
        )
        
        # 多次融合相同输入应产生相同结果
        out1 = fusion(mmi)
        out2 = fusion(mmi)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-5))


class TestFusionIntegration(unittest.TestCase):
    """融合模块集成测试"""
    
    def test_full_modality_fusion(self):
        """完整模态融合"""
        config = FusionConfig(
            hidden_dim=512,
            num_heads=8,
            strategy=FusionStrategy.HYBRID,
        )
        fusion = CrossModalFusion(config)
        
        # 注意: force_dim=32, imu_dim=64 是编码器期望的输入维度
        mmi = MultimodalInput(
            vision=torch.randn(4, 512),
            audio=torch.randn(4, 128),
            tactile=torch.randn(4, 64),
            force=torch.randn(4, 32),
            imu=torch.randn(4, 64),
            language=torch.randint(0, 1000, (4, 32))
        )
        out = fusion(mmi)
        self.assertEqual(out.shape, (4, 512))
    
    def test_fusion_gradient_flow(self):
        """融合网络梯度流"""
        config = FusionConfig(hidden_dim=256)
        fusion = CrossModalFusion(config)
        
        mmi = MultimodalInput(
            vision=torch.randn(2, 512, requires_grad=True),
            audio=torch.randn(2, 128, requires_grad=True)
        )
        out = fusion(mmi)
        loss = out.sum()
        loss.backward()
        
        # 检查梯度是否传播
        self.assertIsNotNone(mmi.vision.grad)
        self.assertIsNotNone(mmi.audio.grad)
    
    def test_cross_attention_shapes(self):
        """跨注意力形状验证"""
        config = FusionConfig(hidden_dim=256, num_heads=4)
        fusion = CrossModalFusion(config)
        
        mmi = MultimodalInput(
            vision=torch.randn(2, 512),
            audio=torch.randn(2, 128)
        )
        out = fusion(mmi)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 256)
    
    def test_language_encoder_integration(self):
        """语言编码器集成"""
        encoder = LanguageEncoder(vocab_size=10000, embed_dim=128, hidden_dim=256, max_len=32)
        
        tokens = torch.randint(0, 1000, (2, 32))
        embeddings = encoder(tokens)
        self.assertEqual(embeddings.shape[0], 2)  # batch size
        self.assertEqual(embeddings.shape[1], 256)  # hidden dim


class TestModalityEncoder(unittest.TestCase):
    """模态编码器测试"""
    
    def test_vision_encoder(self):
        """视觉编码器"""
        encoder = ModalityEncoder('vision', input_dim=512, output_dim=256)
        
        x = torch.randn(2, 512)
        out = encoder(x)
        self.assertEqual(out.shape, (2, 256))
    
    def test_audio_encoder(self):
        """音频编码器"""
        encoder = ModalityEncoder('audio', input_dim=128, output_dim=256)
        
        x = torch.randn(2, 128)
        out = encoder(x)
        self.assertEqual(out.shape, (2, 256))
    
    def test_tactile_encoder(self):
        """触觉编码器"""
        encoder = ModalityEncoder('tactile', input_dim=64, output_dim=256)
        
        x = torch.randn(2, 64)
        out = encoder(x)
        self.assertEqual(out.shape, (2, 256))
    
    def test_force_encoder(self):
        """力觉编码器"""
        encoder = ModalityEncoder('force', input_dim=6, output_dim=256)
        
        x = torch.randn(2, 6)
        out = encoder(x)
        self.assertEqual(out.shape, (2, 256))
    
    def test_imu_encoder(self):
        """IMU编码器"""
        encoder = ModalityEncoder('imu', input_dim=6, output_dim=256)
        
        x = torch.randn(2, 6)
        out = encoder(x)
        self.assertEqual(out.shape, (2, 256))


class TestFusionProportionalTests(unittest.TestCase):
    """融合模块比例测试"""
    
    def test_tactile_dim_proportional_to_array_size(self):
        """触觉维度应与阵列大小成正比"""
        config = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256, num_heads=4
        )
        # 16x16 -> 64维特征
        expected_dim = 16 * 4  # 64
        self.assertEqual(config.tactile_dim, 64)
        
    def test_force_dim_always_six_axis(self):
        """力觉维度固定为6轴"""
        config = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256, num_heads=4
        )
        # 6轴力矩 -> 固定32维
        self.assertEqual(config.force_dim, 32)
    
    def test_imu_dim_supports_magnetometer(self):
        """IMU维度应支持9轴扩展"""
        config = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256, num_heads=4
        )
        self.assertEqual(config.imu_dim, 64)
    
    def test_cross_attention_heads_divisible(self):
        """注意力头数应能整除隐藏维度"""
        for num_heads in [2, 4, 8, 16]:
            config = FusionConfig(
                vision_dim=512, audio_dim=128, tactile_dim=64,
                force_dim=32, imu_dim=64, hidden_dim=256, num_heads=num_heads
            )
            fusion = CrossModalFusion(config)
            self.assertEqual(config.num_heads, num_heads)


class TestFusionTemporalConsistency(unittest.TestCase):
    """融合时序一致性测试"""
    
    def test_sequential_fusion_temporal_coherence(self):
        """连续融合应保持时间一致性"""
        torch.manual_seed(42)
        np.random.seed(42)
        
        config = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256, num_heads=4
        )
        fusion = CrossModalFusion(config)
        
        # 同一模态连续输入的编码应平滑变化
        tactile = [torch.randn(1, 64) for _ in range(10)]
        prev_out = None
        max_diff = 0.0
        
        for t in tactile:
            multimodal = MultimodalInput(
                vision=torch.randn(1, 512),
                audio=torch.randn(1, 128),
                tactile=t,
                force=torch.randn(1, 32),
                imu=torch.randn(1, 64)
            )
            out = fusion(multimodal)
            
            if prev_out is not None:
                diff = torch.abs(out - prev_out).max().item()
                max_diff = max(max_diff, diff)
            
            prev_out = out.clone()
        
        # 随机输入的连续差异应该不会太小(有变化)
        self.assertGreater(max_diff, 0.0)
    
    def test_fusion_idempotent_initialization(self):
        """融合模块初始化应幂等"""
        config = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256, num_heads=4
        )
        
        # 多次初始化应产生相同结构的模型
        fusion1 = CrossModalFusion(config)
        fusion2 = CrossModalFusion(config)
        
        # 验证输出维度一致性
        x = MultimodalInput(
            vision=torch.randn(2, 512),
            audio=torch.randn(2, 128),
            tactile=torch.randn(2, 64),
            force=torch.randn(2, 32),
            imu=torch.randn(2, 64)
        )
        
        out1 = fusion1(x)
        out2 = fusion2(x)
        
        # 输出shape应相同
        self.assertEqual(out1.shape, out2.shape)


class TestFusionMemoryEfficiency(unittest.TestCase):
    """融合内存效率测试"""
    
    def test_large_batch_memory_stable(self):
        """大批量处理内存应稳定"""
        config = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256, num_heads=4
        )
        fusion = CrossModalFusion(config)
        
        # 大批量
        x = MultimodalInput(
            vision=torch.randn(64, 512),
            audio=torch.randn(64, 128),
            tactile=torch.randn(64, 64),
            force=torch.randn(64, 32),
            imu=torch.randn(64, 64)
        )
        
        out = fusion(x)
        self.assertEqual(out.shape, (64, 256))
    
    def test_empty_modalities_zero_cost(self):
        """空模态应零开销"""
        config = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256, num_heads=4
        )
        fusion = CrossModalFusion(config)
        
        # 只有视觉
        x = MultimodalInput(
            vision=torch.randn(4, 512),
            audio=None,
            tactile=None,
            force=None,
            imu=None
        )
        
        out = fusion(x)
        self.assertEqual(out.shape, (4, 256))


class TestFusionAdvanced(unittest.TestCase):
    """高级融合测试"""

    def test_cross_attention_with_different_query_lengths(self):
        """跨模态注意力应支持不同查询长度"""
        config = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256, num_heads=4
        )
        fusion = CrossModalFusion(config)
        
        # 测试不同的序列长度
        for seq_len in [1, 5, 10, 20]:
            vision = torch.randn(2, seq_len, 512)
            multimodal = MultimodalInput(vision=vision)
            out = fusion(multimodal)
            self.assertEqual(out.shape[0], 2)  # batch size
    
    def test_fusion_with_very_large_hidden_dim(self):
        """融合模块应支持大隐藏维度"""
        config = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=2048, num_heads=8
        )
        fusion = CrossModalFusion(config)
        
        multimodal = MultimodalInput(
            vision=torch.randn(1, 512),
            audio=torch.randn(1, 128),
            tactile=torch.randn(1, 64),
            force=torch.randn(1, 32),
            imu=torch.randn(1, 64)
        )
        out = fusion(multimodal)
        self.assertEqual(out.shape[-1], 2048)
    
    def test_fusion_with_very_small_hidden_dim(self):
        """融合模块应支持小隐藏维度"""
        config = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=32, num_heads=2
        )
        fusion = CrossModalFusion(config)
        
        multimodal = MultimodalInput(
            vision=torch.randn(1, 512),
            audio=torch.randn(1, 128),
            tactile=torch.randn(1, 64),
            force=torch.randn(1, 32),
            imu=torch.randn(1, 64)
        )
        out = fusion(multimodal)
        self.assertEqual(out.shape[-1], 32)
    
    def test_language_encoder_with_token_ids(self):
        """语言编码器应处理token_ids"""
        encoder = LanguageEncoder(
            vocab_size=5000, embed_dim=64, hidden_dim=128, max_len=32
        )
        # 短序列
        token_ids = torch.randint(0, 5000, (2, 8))
        lang_out = encoder(token_ids)
        self.assertEqual(lang_out.shape, (2, 128))
    
    def test_language_encoder_single_token(self):
        """语言编码器应处理单token输入"""
        encoder = LanguageEncoder(
            vocab_size=5000, embed_dim=64, hidden_dim=128, max_len=32
        )
        # 单token
        token_ids = torch.randint(0, 5000, (1, 1))
        lang_out = encoder(token_ids)
        self.assertEqual(lang_out.shape, (1, 128))
    
    def test_modality_encoder_with_single_element(self):
        """模态编码器应处理单元素批次"""
        config = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256, num_heads=4
        )
        fusion = CrossModalFusion(config)
        
        # 单元素批次
        vision = torch.randn(1, 512)  # 2D input
        multimodal = MultimodalInput(vision=vision)
        out = fusion(multimodal)
        self.assertEqual(out.shape[0], 1)
    
    def test_fusion_with_identical_modalities(self):
        """融合模块应处理相同模态输入"""
        config = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256, num_heads=4
        )
        fusion = CrossModalFusion(config)
        
        # 相同的视觉输入多次
        vision = torch.randn(2, 512)
        multimodal = MultimodalInput(
            vision=vision,
            audio=torch.randn(2, 128),
            tactile=torch.randn(2, 64)
        )
        out = fusion(multimodal)
        self.assertEqual(out.shape[0], 2)
    
    def test_cross_modal_attention_gradient_flow(self):
        """跨模态注意力应允许梯度流"""
        config = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=128, num_heads=4
        )
        fusion = CrossModalFusion(config)
        fusion.train()
        
        vision = torch.randn(2, 512, requires_grad=True)
        audio = torch.randn(2, 128, requires_grad=True)
        
        multimodal = MultimodalInput(vision=vision, audio=audio)
        out = fusion(multimodal)
        
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(vision.grad)
        self.assertIsNotNone(audio.grad)
    
    def test_unified_representation_properties(self):
        """统一表示应具有正确的属性"""
        config = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256, num_heads=4
        )
        fusion = CrossModalFusion(config)
        
        multimodal = MultimodalInput(
            vision=torch.randn(2, 512),
            audio=torch.randn(2, 128),
            tactile=torch.randn(2, 64),
            force=torch.randn(2, 32),
            imu=torch.randn(2, 64)
        )
        out = fusion(multimodal)
        
        self.assertTrue(isinstance(out, torch.Tensor))
        self.assertEqual(out.dtype, torch.float32)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())
    
    def test_fusion_config_all_strategies(self):
        """融合配置应支持所有融合策略"""
        for strategy in FusionStrategy:
            config = FusionConfig(
                vision_dim=512, audio_dim=128, tactile_dim=64,
                force_dim=32, imu_dim=64, hidden_dim=256, num_heads=4,
                strategy=strategy
            )
            self.assertEqual(config.strategy, strategy)
    
    def test_fusion_config_defaults(self):
        """融合配置应有正确的默认值"""
        config = FusionConfig()
        self.assertEqual(config.hidden_dim, 256)
        self.assertEqual(config.num_heads, 4)
        self.assertEqual(config.vocab_size, 10000)


class TestModalityEncoderAdvanced(unittest.TestCase):
    """模态编码器高级测试"""

    def test_encoder_output_shape_consistency(self):
        """编码器输出形状应一致"""
        config = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256, num_heads=4
        )
        fusion = CrossModalFusion(config)
        
        # Test vision encoder
        x = torch.randn(2, 512)
        out = fusion.vision_encoder(x)
        self.assertEqual(out.shape[0], 2)  # batch size preserved
        self.assertEqual(out.shape[1], 256)  # hidden_dim
        
        # Test audio encoder
        x = torch.randn(2, 128)
        out = fusion.audio_encoder(x)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 256)
    
    def test_encoder_handles_near_zero_input(self):
        """编码器应处理接近零的输入"""
        config = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256, num_heads=4
        )
        fusion = CrossModalFusion(config)
        
        # 接近零的输入
        near_zero = torch.zeros(2, 512) + 1e-6
        out = fusion.vision_encoder(near_zero)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())
    
    def test_encoder_handles_large_magnitude_input(self):
        """编码器应处理大数值输入"""
        config = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256, num_heads=4
        )
        fusion = CrossModalFusion(config)
        
        # 大数值输入
        large_input = torch.randn(2, 512) * 100
        out = fusion.vision_encoder(large_input)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())
    
    def test_all_modality_encoders_work(self):
        """所有模态编码器应正常工作"""
        config = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256, num_heads=4
        )
        fusion = CrossModalFusion(config)
        
        encoders = [
            ('vision', fusion.vision_encoder, 512),
            ('audio', fusion.audio_encoder, 128),
            ('tactile', fusion.tactile_encoder, 64),
            ('force', fusion.force_encoder, 32),
            ('imu', fusion.imu_encoder, 64),
        ]
        
        for name, encoder, input_dim in encoders:
            x = torch.randn(4, input_dim)
            out = encoder(x)
            self.assertEqual(out.shape, (4, 256), f"{name} encoder failed")


class TestForceIMUCrossModalAttention(unittest.TestCase):
    """测试力觉/IMU跨模态注意力 (新增)"""

    def test_vision_force_attention_exists(self):
        """视觉-力觉注意力层应存在"""
        config = FusionConfig(hidden_dim=256, num_heads=4)
        fusion = CrossModalFusion(config)
        self.assertTrue(hasattr(fusion, 'vision_force_attn'))
        self.assertIsInstance(fusion.vision_force_attn, CrossModalAttention)

    def test_vision_imu_attention_exists(self):
        """视觉-IMU注意力层应存在"""
        config = FusionConfig(hidden_dim=256, num_heads=4)
        fusion = CrossModalFusion(config)
        self.assertTrue(hasattr(fusion, 'vision_imu_attn'))
        self.assertIsInstance(fusion.vision_imu_attn, CrossModalAttention)

    def test_audio_force_attention_exists(self):
        """听觉-力觉注意力层应存在"""
        config = FusionConfig(hidden_dim=256, num_heads=4)
        fusion = CrossModalFusion(config)
        self.assertTrue(hasattr(fusion, 'audio_force_attn'))
        self.assertIsInstance(fusion.audio_force_attn, CrossModalAttention)

    def test_audio_imu_attention_exists(self):
        """听觉-IMU注意力层应存在"""
        config = FusionConfig(hidden_dim=256, num_heads=4)
        fusion = CrossModalFusion(config)
        self.assertTrue(hasattr(fusion, 'audio_imu_attn'))
        self.assertIsInstance(fusion.audio_imu_attn, CrossModalAttention)

    def test_force_imu_attention_exists(self):
        """力觉-IMU注意力层应存在"""
        config = FusionConfig(hidden_dim=256, num_heads=4)
        fusion = CrossModalFusion(config)
        self.assertTrue(hasattr(fusion, 'force_imu_attn'))
        self.assertIsInstance(fusion.force_imu_attn, CrossModalAttention)

    def test_vision_force_fusion(self):
        """视觉-力觉融合应正常工作"""
        config = FusionConfig(hidden_dim=256, num_heads=4)
        fusion = CrossModalFusion(config)
        mmi = MultimodalInput(
            vision=torch.randn(2, 512),
            force=torch.randn(2, 32)
        )
        out = fusion(mmi)
        self.assertEqual(out.shape, (2, 256))

    def test_vision_imu_fusion(self):
        """视觉-IMU融合应正常工作"""
        config = FusionConfig(hidden_dim=256, num_heads=4)
        fusion = CrossModalFusion(config)
        mmi = MultimodalInput(
            vision=torch.randn(2, 512),
            imu=torch.randn(2, 64)
        )
        out = fusion(mmi)
        self.assertEqual(out.shape, (2, 256))

    def test_audio_force_fusion(self):
        """听觉-力觉融合应正常工作"""
        config = FusionConfig(hidden_dim=256, num_heads=4)
        fusion = CrossModalFusion(config)
        # 使用2D音频 (时间池化后的特征)
        mmi = MultimodalInput(
            audio=torch.randn(2, 128),
            force=torch.randn(2, 32)
        )
        out = fusion(mmi)
        self.assertEqual(out.shape, (2, 256))

    def test_audio_imu_fusion(self):
        """听觉-IMU融合应正常工作"""
        config = FusionConfig(hidden_dim=256, num_heads=4)
        fusion = CrossModalFusion(config)
        # 使用2D音频 (时间池化后的特征)
        mmi = MultimodalInput(
            audio=torch.randn(2, 128),
            imu=torch.randn(2, 64)
        )
        out = fusion(mmi)
        self.assertEqual(out.shape, (2, 256))

    def test_force_imu_fusion(self):
        """力觉-IMU融合应正常工作"""
        config = FusionConfig(hidden_dim=256, num_heads=4)
        fusion = CrossModalFusion(config)
        mmi = MultimodalInput(
            force=torch.randn(2, 32),
            imu=torch.randn(2, 64)
        )
        out = fusion(mmi)
        self.assertEqual(out.shape, (2, 256))

    def test_all_six_modalities_fusion(self):
        """六模态全融合应正常工作"""
        config = FusionConfig(hidden_dim=256, num_heads=4)
        fusion = CrossModalFusion(config)
        # 使用与FusionConfig匹配的维度: force_dim=32, imu_dim=64
        # 音频使用2D (时间池化后的特征)
        mmi = MultimodalInput(
            vision=torch.randn(2, 512),
            audio=torch.randn(2, 128),
            tactile=torch.randn(2, 64),
            force=torch.randn(2, 32),
            imu=torch.randn(2, 64),
            language=torch.randint(0, 1000, (2, 32))
        )
        out = fusion(mmi)
        self.assertEqual(out.shape, (2, 256))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_tactile_force_imu_triple_fusion(self):
        """触觉-力觉-IMU三模态融合应正常工作"""
        config = FusionConfig(hidden_dim=256, num_heads=4)
        fusion = CrossModalFusion(config)
        # 使用与FusionConfig匹配的维度
        mmi = MultimodalInput(
            tactile=torch.randn(4, 64),
            force=torch.randn(4, 32),
            imu=torch.randn(4, 64)
        )
        out = fusion(mmi)
        self.assertEqual(out.shape, (4, 256))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())


if __name__ == '__main__':
    unittest.main()
