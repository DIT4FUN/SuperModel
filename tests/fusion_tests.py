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
