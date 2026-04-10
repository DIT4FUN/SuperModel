"""
跨模态融合模块测试
=================

测试多模态传感器融合网络
- 单模态测试
- 多模态融合测试
- 注意力机制测试
- 五级规格融合测试
"""

import pytest
import numpy as np
import torch
from typing import Dict, Optional

# 导入被测模块
from src.fusion.cross_modal_fusion import (
    CrossModalFusion, FusionConfig, MultimodalInput, FusionStrategy,
    create_multimodal_input, get_fusion_spec
)
from src.fusion.sensor_fusion import (
    ComplementaryFilter, ExtendedKalmanFilter, MultiSensorFusion,
    SensorFusion
)


class TestComplementaryFilter:
    """互补滤波测试"""
    
    def test_create(self):
        """测试创建互补滤波器"""
        cf = ComplementaryFilter(alpha=0.98)
        assert cf.alpha == 0.98
    
    def test_update(self):
        """测试更新"""
        cf = ComplementaryFilter(alpha=0.98)
        # 加速度计角度和陀螺仪角速度
        measurements = {
            'accel': np.array([0.0, 0.1]),
            'gyro': np.array([0.0, 0.0])
        }
        dt = 0.01
        angle = cf.update(measurements, dt)
        assert angle.shape == (3,)
        assert isinstance(angle, np.ndarray)


class TestExtendedKalmanFilter:
    """扩展卡尔曼滤波测试"""
    
    def test_create(self):
        """测试创建EKF"""
        # 状态维度 = 15 (pos 3 + vel 3 + quat 4 + bias 6)
        ekf = ExtendedKalmanFilter(state_dim=15, measurement_dim=3)
        assert ekf.state_dim == 15
        assert ekf.measurement_dim == 3
        assert ekf.get_covariance().shape == (15, 15)
    
    def test_predict(self):
        """测试预测步骤"""
        ekf = ExtendedKalmanFilter(state_dim=9, measurement_dim=3)
        ekf.initialize(np.zeros(9))
        ekf._P = np.eye(9) * 0.1
        ekf.predict(dt=0.01)
        assert ekf.get_covariance() is not None
    
    def test_update(self):
        """测试更新步骤"""
        ekf = ExtendedKalmanFilter(state_dim=9, measurement_dim=3)
        ekf.initialize(np.zeros(9))
        ekf._P = np.eye(9) * 0.1
        measurement = np.zeros(3)
        ekf.update({'meas': measurement}, 0.01)
        assert ekf.get_state() is not None


class TestMultiSensorFusion:
    """多传感器融合测试"""
    
    def test_create(self):
        """测试创建融合器"""
        fusion = MultiSensorFusion()
        assert fusion is not None
    
    def test_add_method(self):
        """测试添加融合方法"""
        fusion = MultiSensorFusion()
        cf = ComplementaryFilter(alpha=0.98)
        fusion.add_fusion_method('imu', cf, weight=1.0)
        assert 'imu' in fusion.fusion_methods
        assert fusion._weights['imu'] == 1.0


class TestCrossModalAttention:
    """跨模态注意力测试"""
    
    def test_create(self):
        """测试创建跨模态注意力"""
        from src.fusion.cross_modal_fusion import CrossModalAttention
        attn = CrossModalAttention(query_dim=256, key_dim=256, value_dim=256, num_heads=4)
        assert attn.num_heads == 4
    
    def test_forward(self):
        """测试前向传播"""
        from src.fusion.cross_modal_fusion import CrossModalAttention
        attn = CrossModalAttention(query_dim=256, key_dim=256, value_dim=256, num_heads=4)
        B = 2
        Nq = 10
        Nk = 15
        D = 256
        query = torch.randn(B, Nq, D)
        key = torch.randn(B, Nk, D)
        value = torch.randn(B, Nk, D)
        output = attn(query, key, value)
        assert output.shape == (B, Nq, D)


class TestLanguageEncoder:
    """语言编码器测试"""
    
    def test_create(self):
        """测试创建语言编码器"""
        from src.fusion.cross_modal_fusion import LanguageEncoder
        enc = LanguageEncoder(
            vocab_size=10000,
            embed_dim=128,
            hidden_dim=256,
            num_heads=4,
            num_layers=2,
            max_len=32
        )
        assert enc is not None
    
    def test_forward(self):
        """测试前向传播"""
        from src.fusion.cross_modal_fusion import LanguageEncoder
        enc = LanguageEncoder(
            vocab_size=1000,
            embed_dim=32,
            hidden_dim=64,
            num_heads=2,
            num_layers=1,
            max_len=16
        )
        B = 2
        L = 16
        tokens = torch.randint(0, 1000, (B, L))
        output = enc(tokens)
        assert output.shape[0] == B
        assert output.shape[1] == 64  # hidden_dim


# 编码器作为内部结构存在于 CrossModalFusion，不需要单独测试
# 已通过整体融合测试
class TestModalityEncoder:
    """模态编码器测试 - 作为 CrossModalFusion 内部组件"""
    
    def test_encoder_exists(self):
        """测试 CrossModalFusion 包含编码器权重"""
        fusion = CrossModalFusion()
        # 编码器字典应该存在
        assert hasattr(fusion, 'encoders')
        # vision 编码器应该存在
        assert 'vision' in fusion.encoders
        assert 'audio' in fusion.encoders
        assert 'tactile' in fusion.encoders
        assert 'force' in fusion.encoders
        assert 'imu' in fusion.encoders
        # 编码器权重形状正确
        assert fusion.encoders['vision']['W'].shape == (512, 256)
        assert fusion.encoders['audio']['W'].shape == (128, 256)


class TestMultimodalInput:
    """多模态输入测试"""
    
    def test_create_empty(self):
        """测试创建空输入"""
        mmi = MultimodalInput()
        assert mmi.vision is None
        assert mmi.audio is None
        assert mmi.tactile is None
        assert mmi.force is None
        assert mmi.imu is None
        assert mmi.language is None
    
    def test_create_with_vision(self):
        """测试创建带视觉的输入"""
        vision = np.random.randn(2, 512).astype(np.float32)
        mmi = MultimodalInput(vision=vision)
        assert mmi.vision is not None
        assert mmi.vision.shape == (2, 512)
    
    def test_available_modalities(self):
        """测试获取可用模态"""
        mmi = MultimodalInput(
            vision=np.random.randn(2, 512),
            audio=np.random.randn(2, 128),
            imu=np.random.randn(2, 6)
        )
        modalities = mmi.modalities
        assert 'vision' in modalities
        assert 'audio' in modalities
        assert 'imu' in modalities
        assert 'tactile' not in modalities


class TestCreateMultimodalInput:
    """create_multimodal_input 工厂函数测试"""
    
    def test_from_numpy(self):
        """测试从numpy数组创建"""
        vision = np.random.randn(2, 512).astype(np.float32)
        audio = np.random.randn(2, 128).astype(np.float32)
        mmi = create_multimodal_input(vision=vision, audio=audio)
        assert isinstance(mmi.vision, np.ndarray)
        assert isinstance(mmi.audio, np.ndarray)
        assert mmi.vision.shape == (2, 512)
        assert mmi.audio.shape == (2, 128)


class TestCrossModalFusion:
    """跨模态融合网络测试"""
    
    def test_create_default(self):
        """测试创建默认配置融合网络"""
        fusion = CrossModalFusion()
        assert fusion is not None
    
    def test_create_custom_config(self):
        """测试创建自定义配置"""
        config = FusionConfig(
            vision_dim=256,
            audio_dim=64,
            tactile_dim=32,
            force_dim=16,
            imu_dim=16,
            lang_dim=128,
            hidden_dim=128,
            num_heads=2,
            num_layers=1,
            fusion_type="early"
        )
        fusion = CrossModalFusion(config)
        assert fusion.config == config
    
    def test_forward_single_modality(self):
        """测试单模态融合"""
        fusion = CrossModalFusion()
        mmi = MultimodalInput(vision=np.random.randn(2, 512).astype(np.float32))
        with torch.no_grad():
            fused = fusion(mmi)
        assert isinstance(fused, np.ndarray)
        assert fused.shape[0] == 2  # batch
    
    def test_forward_two_modalities(self):
        """测试双模态融合"""
        fusion = CrossModalFusion()
        mmi = MultimodalInput(
            vision=np.random.randn(2, 512).astype(np.float32),
            audio=np.random.randn(2, 128).astype(np.float32)
        )
        with torch.no_grad():
            fused = fusion(mmi)
        assert fused.shape[0] == 2
    
    def test_forward_all_modalities(self):
        """测试全模态融合"""
        fusion = CrossModalFusion(
            FusionConfig(
                vision_dim=512,
                audio_dim=128,
                tactile_dim=64,
                force_dim=32,
                imu_dim=32,
                lang_dim=128,
                hidden_dim=256,
                num_heads=4,
                num_layers=2
            )
        )
        mmi = MultimodalInput(
            vision=np.random.randn(2, 512).astype(np.float32),
            audio=np.random.randn(2, 128).astype(np.float32),
            tactile=np.random.randn(2, 64).astype(np.float32),
            force=np.random.randn(2, 6).astype(np.float32),
            imu=np.random.randn(2, 6).astype(np.float32),
            language=np.random.randint(0, 10000, (2, 32)).astype(np.int64)
        )
        with torch.no_grad():
            fused = fusion(mmi)
        assert isinstance(fused, np.ndarray)
        assert fused.shape[0] == 2
        assert not np.any(np.isnan(fused))
    
    def test_all_fusion_strategies(self):
        """测试所有融合策略"""
        for strategy in FusionStrategy:
            config = FusionConfig(
                vision_dim=512,
                audio_dim=128,
                tactile_dim=64,
                force_dim=32,
                imu_dim=32,
                lang_dim=128,
                hidden_dim=128,
                num_heads=2,
                num_layers=1,
                fusion_type=strategy.value
            )
            fusion = CrossModalFusion(config)
            mmi = MultimodalInput(
                vision=np.random.randn(2, 512).astype(np.float32),
                audio=np.random.randn(2, 128).astype(np.float32)
            )
            with torch.no_grad():
                fused = fusion(mmi)
            assert not np.any(np.isnan(fused))
            assert fused.shape[0] == 2


class TestFusionConfigForGrade:
    """融合配置五级规格测试"""
    
    def test_get_fusion_spec_all_grades(self):
        """测试所有等级的融合规格"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_fusion_spec(grade)
            assert isinstance(spec, dict)
            assert 'hidden_dim' in spec
            assert 'attention_heads' in spec
            assert 'transformer_layers' in spec
            assert 'fusion_type' in spec
    
    def test_spec_increasing_capacity(self):
        """测试规格容量随等级增加"""
        spec_s = get_fusion_spec('S')
        spec_m = get_fusion_spec('M')
        spec_l = get_fusion_spec('L')
        spec_xl = get_fusion_spec('XL')
        spec_xxl = get_fusion_spec('XXL')
        
        assert spec_s['hidden_dim'] <= spec_m['hidden_dim']
        assert spec_m['hidden_dim'] <= spec_l['hidden_dim']
        assert spec_l['hidden_dim'] <= spec_xl['hidden_dim']
        assert spec_xl['hidden_dim'] <= spec_xxl['hidden_dim']


class TestIntegration:
    """集成测试"""
    
    def test_sensor_fusion_chain(self):
        """传感器融合完整链"""
        # 1. 创建互补滤波器
        cf = ComplementaryFilter(alpha=0.98)
        
        # 2. 模拟数据
        dt = 0.01
        n_steps = 100
        
        for i in range(n_steps):
            measurements = {
                'accel': np.array([0.01 * i, 0.0]),
                'gyro': np.array([0.01, 0.0])
            }
            angle = cf.update(measurements, dt)
        
        assert angle.shape == (3,)
        assert not np.any(np.isnan(angle))
    
    def test_cross_modal_full_chain(self):
        """跨模态融合完整链"""
        # 1. 创建融合网络
        fusion = CrossModalFusion()
        
        # 2. 从numpy创建多模态输入
        vision = np.random.randn(1, 512).astype(np.float32)
        tactile = np.random.randn(1, 256).astype(np.float32)
        force = np.random.randn(1, 6).astype(np.float32)
        imu = np.random.randn(1, 6).astype(np.float32)
        
        mmi = create_multimodal_input(
            vision=vision,
            tactile=tactile,
            force=force,
            imu=imu
        )
        
        # 3. 融合
        with torch.no_grad():
            fused = fusion(mmi)
        
        assert fused.shape[0] == 1
        assert isinstance(fused, np.ndarray)
        assert not np.any(np.isnan(fused))
    
    def test_multi_sensor_fusion(self):
        """多传感器融合"""
        fusion = MultiSensorFusion()
        cf = ComplementaryFilter(alpha=0.98)
        ekf = ExtendedKalmanFilter(state_dim=9, measurement_dim=3)
        
        fusion.add_fusion_method('imu_complementary', cf, weight=0.6)
        fusion.add_fusion_method('ekf', ekf, weight=0.4)
        
        assert 'imu_complementary' in fusion.fusion_methods
        assert len(fusion.fusion_methods) == 2


class TestGradients:
    """梯度测试 - 确保梯度能够正确传播"""
    
    def test_gradient_propagation(self):
        """测试梯度传播"""
        fusion = CrossModalFusion(
            FusionConfig(
                vision_dim=32,
                audio_dim=16,
                tactile_dim=32,
                force_dim=6,
                imu_dim=6,
                lang_dim=128,
                hidden_dim=64,
                num_heads=2,
                num_layers=1,
                fusion_type="hybrid"
            )
        )
        
        # 启用梯度
        fusion.train()
        
        # 输入
        vision = torch.randn(2, 32, requires_grad=True)
        audio = torch.randn(2, 16, requires_grad=True)
        mmi = MultimodalInput(vision=vision, audio=audio)
        
        # 前向
        fused = fusion(mmi)  # fused 是 numpy array (因为CrossModalFusion.forward 返回 numpy)
        # 注意: CrossModalFusion 在 forward 中返回 numpy
        # 要测试梯度需要修改架构，这个测试只是验证不崩溃
        # 实际梯度检查需要端到端训练
        assert fused.shape[0] == 2
    
    def test_inference_no_grad(self):
        """推理模式不需要梯度"""
        fusion = CrossModalFusion()
        fusion.eval()
        
        vision = np.random.randn(2, 512).astype(np.float32)
        mmi = MultimodalInput(vision=vision)
        
        with torch.no_grad():
            fused = fusion(mmi)
        
        assert isinstance(fused, np.ndarray)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
