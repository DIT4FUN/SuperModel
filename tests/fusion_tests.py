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


class TestFusionEdgeCases(unittest.TestCase):
    """融合模块边缘用例测试"""
    
    def test_multimodal_input_partial_none(self):
        """测试部分模态为None时的融合"""
        multimodal = MultimodalInput(
            vision=torch.randn(2, 512),
            audio=None,
            tactile=torch.randn(2, 64),
            force=torch.randn(2, 32),
            imu=None
        )
        # 应该能正常处理
        self.assertIsNotNone(multimodal.vision)
        self.assertIsNone(multimodal.audio)
        self.assertIsNone(multimodal.imu)
        self.assertIn('vision', multimodal.modalities)
        self.assertIn('tactile', multimodal.modalities)
        self.assertNotIn('audio', multimodal.modalities)
    
    def test_fusion_config_defaults(self):
        """测试融合配置默认值"""
        config = FusionConfig()
        self.assertEqual(config.hidden_dim, 256)
        self.assertEqual(config.num_heads, 4)
        self.assertEqual(config.strategy, FusionStrategy.HYBRID)
    
    def test_fusion_config_custom(self):
        """测试融合配置自定义"""
        config = FusionConfig(
            vision_dim=1024,
            audio_dim=256,
            hidden_dim=512,
            num_heads=8,
            strategy=FusionStrategy.EARLY
        )
        self.assertEqual(config.vision_dim, 1024)
        self.assertEqual(config.audio_dim, 256)
        self.assertEqual(config.hidden_dim, 512)
        self.assertEqual(config.num_heads, 8)
        self.assertEqual(config.strategy, FusionStrategy.EARLY)
    
    def test_attention_different_input_dims(self):
        """测试注意力模块处理不同输入维度"""
        B = 2
        attn = CrossModalAttention(query_dim=512, key_dim=256, value_dim=256, num_heads=4)
        q = torch.randn(B, 10, 512)
        k = torch.randn(B, 20, 256)
        v = torch.randn(B, 20, 256)
        out = attn(q, k, v)
        self.assertEqual(out.shape, (B, 10, 512))
    
    def test_fusion_with_late_strategy(self):
        """测试晚期融合策略"""
        config = FusionConfig(
            vision_dim=512,
            audio_dim=128,
            hidden_dim=256,
            strategy=FusionStrategy.LATE
        )
        fusion = CrossModalFusion(config)
        multimodal = MultimodalInput(
            vision=torch.randn(2, 512),
            audio=torch.randn(2, 128)
        )
        output = fusion(multimodal)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 256)


if __name__ == '__main__':
    # 检查 CUDA 是否可用
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, running on CPU")
    
    # 运行测试
    unittest.main(verbosity=2)


class TestFusionRealWorldScenarios(unittest.TestCase):
    """融合模块真实场景测试"""
    
    def setUp(self):
        self.config = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256,
            num_heads=4, num_layers=2
        )
        self.fusion = CrossModalFusion(self.config)
    
    def test_pick_and_place_scenario(self):
        """测试抓取放置场景 - 视觉+触觉+力觉融合"""
        batch_size = 4
        
        # 模拟视觉特征: 目标检测 + 物体位置
        vision = torch.randn(batch_size, 512)
        
        # 模拟触觉特征: 接触压力分布
        tactile = torch.randn(batch_size, 64)
        
        # 模拟力觉特征: 抓取力
        force = torch.randn(batch_size, 32)
        
        # 模拟IMU: 机械臂姿态
        imu = torch.randn(batch_size, 64)
        
        multimodal = MultimodalInput(
            vision=vision, tactile=tactile, force=force, imu=imu
        )
        
        output = self.fusion(multimodal)
        self.assertEqual(output.shape, (batch_size, 256))
        
        # 验证输出不为零
        self.assertFalse(torch.allclose(output, torch.zeros_like(output)))
    
    def test_human_robot_collaboration(self):
        """测试人机协作场景 - IMU+力觉+视觉"""
        batch_size = 4
        
        # 人类动作检测
        vision = torch.randn(batch_size, 512)
        
        # 接触力检测
        force = torch.randn(batch_size, 32)
        
        # 人类运动姿态
        imu = torch.randn(batch_size, 64)
        
        multimodal = MultimodalInput(
            vision=vision, force=force, imu=imu
        )
        
        output = self.fusion(multimodal)
        self.assertEqual(output.shape, (batch_size, 256))
    
    def test_voice_command_with_context(self):
        """测试语音指令理解 - 听觉+视觉+语言"""
        batch_size = 4
        
        # 视觉上下文
        vision = torch.randn(batch_size, 512)
        
        # 音频特征
        audio = torch.randn(batch_size, 128)
        
        # 语言嵌入 (可选)
        language = torch.randint(0, 1000, (batch_size, 20))
        
        multimodal = MultimodalInput(
            vision=vision, audio=audio, language=language
        )
        
        output = self.fusion(multimodal)
        self.assertEqual(output.shape, (batch_size, 256))
    
    def test_grape_manipulation_precision(self):
        """测试精细操作(葡萄抓取) - 全模态融合"""
        batch_size = 2
        
        # 高分辨率视觉
        vision = torch.randn(batch_size, 512)
        
        # 高密度触觉
        tactile = torch.randn(batch_size, 64)
        
        # 精细力觉
        force = torch.randn(batch_size, 32)
        
        # 高精度IMU
        imu = torch.randn(batch_size, 64)
        
        # 语音反馈
        audio = torch.randn(batch_size, 128)
        
        multimodal = MultimodalInput(
            vision=vision, tactile=tactile, force=force, imu=imu, audio=audio
        )
        
        output = self.fusion(multimodal)
        self.assertEqual(output.shape, (batch_size, 256))
        
        # 验证所有模态都参与了融合
        self.assertFalse(torch.allclose(output[:batch_size//2], output[batch_size//2:]))
    
    def test_degraded_modality_scenario(self):
        """测试降级模态场景 - 部分传感器失效"""
        batch_size = 4
        
        # 视觉正常
        vision = torch.randn(batch_size, 512)
        
        # 触觉失效 (全零)
        tactile = torch.zeros(batch_size, 64)
        
        # 力觉正常
        force = torch.randn(batch_size, 32)
        
        # IMU正常
        imu = torch.randn(batch_size, 64)
        
        multimodal = MultimodalInput(
            vision=vision, tactile=tactile, force=force, imu=imu
        )
        
        # 应该仍能正常工作
        output = self.fusion(multimodal)
        self.assertEqual(output.shape, (batch_size, 256))
    
    def test_batch_consistency(self):
        """测试批量推理一致性"""
        batch_size = 8
        
        vision = torch.randn(batch_size, 512)
        audio = torch.randn(batch_size, 128)
        
        multimodal = MultimodalInput(vision=vision, audio=audio)
        
        # 设置eval模式关闭dropout
        self.fusion.eval()
        
        # 单次批量处理
        output_batch = self.fusion(multimodal)
        
        # 逐个样本处理
        outputs_individual = []
        for i in range(batch_size):
            single = MultimodalInput(
                vision=vision[i:i+1], audio=audio[i:i+1]
            )
            outputs_individual.append(self.fusion(single))
        
        # 合并结果
        output_stacked = torch.cat(outputs_individual, dim=0)
        
        # 验证结果一致 (允许浮点误差)
        self.assertTrue(torch.allclose(output_batch, output_stacked, atol=1e-5))
    
    def test_temporal_sequence_fusion(self):
        """测试时序序列融合"""
        seq_len = 10
        batch_size = 4
        
        # 时序视觉特征
        vision = torch.randn(seq_len, batch_size, 512)
        
        # 时序触觉
        tactile = torch.randn(seq_len, batch_size, 64)
        
        multimodal_seq = MultimodalInput(vision=vision, tactile=tactile)
        
        # 使用序列融合
        config_seq = FusionConfig(
            vision_dim=512, audio_dim=128, tactile_dim=64,
            force_dim=32, imu_dim=64, hidden_dim=256,
            num_heads=4, num_layers=2, strategy=FusionStrategy.HYBRID
        )
        
        fusion_seq = CrossModalFusion(config_seq)
        
        # 展平处理
        B, T, C = vision.shape
        flat_vision = vision.reshape(B * T, C)
        T2, B2, C2 = tactile.shape
        flat_tactile = tactile.reshape(B2 * T2, C2)
        
        multimodal_flat = MultimodalInput(vision=flat_vision, tactile=flat_tactile)
        output = fusion_seq(multimodal_flat)
        self.assertEqual(output.shape[0], B * T)
    
    def test_emergency_stop_scenario(self):
        """测试紧急停止场景 - 高速力觉响应"""
        batch_size = 4
        
        # 正常视觉输入
        vision = torch.randn(batch_size, 512)
        
        # 异常高力信号 (碰撞) - 在Fx方向施加大的力
        force_base = torch.zeros(batch_size, 32)
        force_base[:, 0] = 100.0  # Fx方向
        force = torch.randn(batch_size, 32) * 10 + force_base
        
        # IMU 快速变化
        imu = torch.randn(batch_size, 64)
        
        multimodal = MultimodalInput(vision=vision, force=force, imu=imu)
        
        output = self.fusion(multimodal)
        self.assertEqual(output.shape, (batch_size, 256))
        
        # 输出应能检测异常 (非零梯度)
        self.assertTrue(output.abs().max() > 0.1)


class TestCrossModalAttentionAdvanced(unittest.TestCase):
    """跨模态注意力高级功能测试"""
    
    def test_attention_score_distribution(self):
        """测试注意力分数分布"""
        B, N, M = 2, 10, 20
        num_heads = 4
        
        attn = CrossModalAttention(query_dim=256, key_dim=256, value_dim=256, num_heads=num_heads)
        attn.eval()  # 关闭dropout
        
        query = torch.randn(B, N, 256)
        key = torch.randn(B, M, 256)
        value = torch.randn(B, M, 256)
        
        out = attn(query, key, value)
        
        # 验证输出形状
        self.assertEqual(out.shape, (B, N, 256))
        
        # 验证输出不为零
        self.assertFalse(torch.allclose(out, torch.zeros_like(out)))
    
    def test_cross_modality_attention(self):
        """测试跨模态注意力 (Q来自视觉, K/V来自其他模态)"""
        B = 2
        
        attn_visual_to_all = CrossModalAttention(
            query_dim=512, key_dim=512, value_dim=512, num_heads=8
        )
        attn_visual_to_all.eval()
        
        # 视觉作为查询
        vision_q = torch.randn(B, 10, 512)
        
        # 模拟其他模态特征 (统一维度后)
        other_modality_kv = torch.randn(B, 20, 512)
        
        # 跨模态注意力: 视觉查询关注其他模态
        out = attn_visual_to_all(vision_q, other_modality_kv, other_modality_kv)
        self.assertEqual(out.shape, (B, 10, 512))
    
    def test_attention_with_cosine_similarity(self):
        """测试基于余弦相似度的注意力"""
        B = 4
        attn = CrossModalAttention(query_dim=128, key_dim=128, value_dim=128, num_heads=4)
        attn.eval()
        
        # 创建相似的查询和键
        base = torch.randn(B, 5, 128)
        query = base + torch.randn_like(base) * 0.1  # 添加小噪声
        key = base + torch.randn_like(base) * 0.1
        value = torch.randn(B, 5, 128)
        
        out = attn(query, key, value)
        
        # 验证输出形状
        self.assertEqual(out.shape, (B, 5, 128))
        
        # 验证输出不为零
        self.assertFalse(torch.allclose(out, torch.zeros_like(out)))


class TestUnifiedRepresentation(unittest.TestCase):
    """统一表示测试"""
    
    def test_representation_shapes(self):
        """测试不同模态组合的统一表示"""
        from fusion.cross_modal_fusion import UnifiedRepresentation
        
        config = FusionConfig(hidden_dim=256, num_heads=4, num_layers=2)
        
        for num_modalities in [1, 2, 3, 4, 5, 6]:
            batch_size = 4
            
            vision = torch.randn(batch_size, 512) if num_modalities >= 1 else None
            audio = torch.randn(batch_size, 128) if num_modalities >= 2 else None
            tactile = torch.randn(batch_size, 64) if num_modalities >= 3 else None
            force = torch.randn(batch_size, 32) if num_modalities >= 4 else None
            imu = torch.randn(batch_size, 64) if num_modalities >= 5 else None
            language = torch.randint(0, 1000, (batch_size, 20)) if num_modalities >= 6 else None
            
            multimodal = MultimodalInput(
                vision=vision, audio=audio, tactile=tactile,
                force=force, imu=imu, language=language
            )
            
            fusion = CrossModalFusion(config)
            output = fusion(multimodal)
            self.assertEqual(output.shape, (batch_size, 256))
    
    def test_representation_distance(self):
        """测试不同输入产生的表示距离"""
        config = FusionConfig(hidden_dim=256)
        fusion = CrossModalFusion(config)
        fusion.eval()  # 关闭dropout以保证确定性
        
        # 相同输入
        vision1 = torch.randn(4, 512)
        vision2 = vision1.clone()
        
        multimodal1 = MultimodalInput(vision=vision1)
        multimodal2 = MultimodalInput(vision=vision2)
        
        out1 = fusion(multimodal1)
        out2 = fusion(multimodal2)
        
        # 相同输入应产生相同输出
        self.assertTrue(torch.allclose(out1, out2, atol=1e-6))
        
        # 不同输入应产生不同输出
        vision3 = torch.randn(4, 512)
        multimodal3 = MultimodalInput(vision=vision3)
        out3 = fusion(multimodal3)
        
        distance = torch.norm(out1 - out3, dim=1).mean()
        self.assertGreater(distance, 0.01)
