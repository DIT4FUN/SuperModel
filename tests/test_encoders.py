"""
传感器编码器测试
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.encoders import (
    VisionEncoder, AudioEncoder, TactileEncoder,
    ForceEncoder, IMUEncoder, LanguageEncoder, MultiModalEncoder,
    SensorEncoderWrapper, EncoderConfig,
    create_sensor_encoder, get_encoder_config,
    ENCODER_GRADES
)


def test_vision_encoder():
    """测试视觉编码器"""
    print("\n[1] 视觉编码器测试")
    
    encoder = VisionEncoder(latent_dim=256, hidden_dim=256)
    
    # 模拟图像
    left_image = torch.randn(4, 3, 224, 224)
    right_image = torch.randn(4, 3, 224, 224)
    
    # 单眼
    features = encoder(left_image)
    assert features.shape == (4, 256), f"Expected (4, 256), got {features.shape}"
    
    # 双目
    features_stereo = encoder(left_image, right_image)
    assert features_stereo.shape == (4, 256), f"Expected (4, 256), got {features_stereo.shape}"
    
    print(f"    左眼特征: {features.shape}")
    print(f"    双目特征: {features_stereo.shape}")
    print("    ✅ 视觉编码器测试通过")


def test_audio_encoder():
    """测试音频编码器"""
    print("\n[2] 音频编码器测试")
    
    encoder = AudioEncoder(latent_dim=256, hidden_dim=256)
    
    # 模拟音频 (B, T, n_mels)
    left_audio = torch.randn(4, 100, 64)
    right_audio = torch.randn(4, 100, 64)
    
    features = encoder(left_audio, right_audio)
    assert features.shape == (4, 256), f"Expected (4, 256), got {features.shape}"
    
    print(f"    音频特征: {features.shape}")
    print("    ✅ 音频编码器测试通过")


def test_tactile_encoder():
    """测试触觉编码器"""
    print("\n[3] 触觉编码器测试")
    
    encoder = TactileEncoder(latent_dim=256, hidden_dim=256)
    
    # 模拟触觉数据 (B, 1, H, W)
    tactile = torch.randn(4, 1, 16, 16)
    
    features = encoder(tactile)
    assert features.shape == (4, 256), f"Expected (4, 256), got {features.shape}"
    
    print(f"    触觉特征: {features.shape}")
    print("    ✅ 触觉编码器测试通过")


def test_force_encoder():
    """测试力觉编码器"""
    print("\n[4] 力觉编码器测试")
    
    encoder = ForceEncoder(latent_dim=256, hidden_dim=256)
    
    # 单帧
    force_single = torch.randn(4, 6)
    features_single = encoder(force_single)
    assert features_single.shape == (4, 256), f"Expected (4, 256), got {features_single.shape}"
    
    # 时序
    force_seq = torch.randn(4, 10, 6)
    features_seq = encoder(force_seq)
    assert features_seq.shape == (4, 256), f"Expected (4, 256), got {features_seq.shape}"
    
    print(f"    单帧特征: {features_single.shape}")
    print(f"    时序特征: {features_seq.shape}")
    print("    ✅ 力觉编码器测试通过")


def test_imu_encoder():
    """测试 IMU 编码器"""
    print("\n[5] IMU 编码器测试")
    
    encoder = IMUEncoder(latent_dim=256, hidden_dim=256)
    
    # 单帧
    imu_single = torch.randn(4, 6)
    features_single, quat_single = encoder(imu_single)
    assert features_single.shape == (4, 256), f"Expected (4, 256), got {features_single.shape}"
    assert quat_single.shape == (4, 4), f"Expected (4, 4), got {quat_single.shape}"
    
    # 时序
    imu_seq = torch.randn(4, 10, 6)
    features_seq, quat_seq = encoder(imu_seq)
    assert features_seq.shape == (4, 256), f"Expected (4, 256), got {features_seq.shape}"
    assert quat_seq.shape == (4, 4), f"Expected (4, 4), got {quat_seq.shape}"
    
    print(f"    单帧特征: {features_single.shape}, 四元数: {quat_single.shape}")
    print(f"    时序特征: {features_seq.shape}, 四元数: {quat_seq.shape}")
    print("    ✅ IMU 编码器测试通过")


def test_multimodal_encoder():
    """测试多模态编码器"""
    print("\n[6] 多模态编码器测试")
    
    encoder = MultiModalEncoder(latent_dim=256)
    
    # 模拟所有模态数据
    vision_left = torch.randn(2, 3, 224, 224)
    vision_right = torch.randn(2, 3, 224, 224)
    audio_left = torch.randn(2, 100, 64)
    audio_right = torch.randn(2, 100, 64)
    tactile = torch.randn(2, 1, 16, 16)
    force = torch.randn(2, 10, 6)
    imu = torch.randn(2, 10, 6)
    
    features = encoder(
        vision_left, vision_right,
        audio_left, audio_right,
        tactile, force, imu
    )
    
    assert 'vision' in features
    assert 'audio' in features
    assert 'tactile' in features
    assert 'force' in features
    assert 'imu' in features
    assert 'fused' in features
    
    for k, v in features.items():
        print(f"    {k}: {v.shape}")
        
    print("    ✅ 多模态编码器测试通过")


def test_sensor_encoder_wrapper():
    """测试传感器编码器封装"""
    print("\n[7] 传感器编码器封装测试")
    
    obs_dims = {
        'vision': 512,
        'audio': 128,
        'tactile': 64,
        'force': 32,
        'imu': 64
    }
    
    encoder = SensorEncoderWrapper(obs_dims, latent_dim=256)
    
    observations = {
        'vision': torch.randn(2, 3, 224, 224),
        'audio': torch.randn(2, 100, 64),
        'tactile': torch.randn(2, 1, 16, 16),
        'force': torch.randn(2, 10, 6),
        'imu': torch.randn(2, 10, 6)
    }
    
    encoded = encoder(observations)
    
    assert 'vision' in encoded
    assert 'fused' in encoded
    
    print(f"    编码后模态: {list(encoded.keys())}")
    print("    ✅ 传感器编码器封装测试通过")


def test_agv_grades():
    """测试 AGV 五级配置"""
    print("\n[8] AGV 五级编码器配置测试")
    
    for grade in ['S', 'M', 'L', 'XL', 'XXL']:
        config = get_encoder_config(grade)
        print(f"    {grade}: latent={config.latent_dim}, hidden={config.hidden_dim}")
        
    print("    ✅ AGV 五级配置测试通过")


def test_encoder_training():
    """测试编码器训练"""
    print("\n[9] 编码器训练测试")
    
    encoder = MultiModalEncoder(latent_dim=256)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    
    # 模拟数据
    for _ in range(5):
        vision_left = torch.randn(4, 3, 224, 224)
        vision_right = torch.randn(4, 3, 224, 224)
        audio_left = torch.randn(4, 100, 64)
        audio_right = torch.randn(4, 100, 64)
        
        optimizer.zero_grad()
        features = encoder(
            vision_left, vision_right,
            audio_left, audio_right
        )
        loss = features['vision'].sum() + features['audio'].sum()
        loss.backward()
        optimizer.step()
        
    print(f"    训练完成，损失: {loss.item():.4f}")
    print("    ✅ 编码器训练测试通过")


def test_language_encoder():
    """测试语言编码器"""
    print("\n[10] 语言编码器测试")
    
    encoder = LanguageEncoder(
        vocab_size=5000, embed_dim=128, hidden_dim=256,
        max_len=32, num_heads=4, num_layers=2
    )
    
    # 模拟 token 序列
    B, L = 4, 20
    token_ids = torch.randint(0, 5000, (B, L))
    
    # eval 模式测试确定性
    encoder.eval()
    features = encoder(token_ids)
    assert features.shape == (B, 256), f"Expected ({B}, 256), got {features.shape}"
    
    features2 = encoder(token_ids)
    assert torch.allclose(features, features2, atol=1e-6), "Same inputs should produce same outputs"
    
    # 不同输入应产生不同输出
    token_ids3 = torch.randint(0, 5000, (B, L))
    features3 = encoder(token_ids3)
    distance = torch.norm(features - features3, dim=1).mean()
    assert distance > 0.01, "Different inputs should produce different outputs"
    
    # 训练模式测试梯度
    encoder.train()
    token_ids4 = torch.randint(0, 5000, (B, L))
    features4 = encoder(token_ids4)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    loss = features4.sum()
    loss.backward()
    optimizer.step()
    
    # 检查有参数有梯度
    has_grad = any(p.grad is not None for p in encoder.parameters() if p.requires_grad)
    assert has_grad, "Parameters should have gradients after training step"
    
    print(f"    特征形状: {features.shape}")
    print(f"    ✅ 语言编码器测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("传感器编码器测试套件")
    print("=" * 60)
    
    try:
        test_vision_encoder()
        test_audio_encoder()
        test_tactile_encoder()
        test_force_encoder()
        test_imu_encoder()
        test_multimodal_encoder()
        test_sensor_encoder_wrapper()
        test_agv_grades()
        test_encoder_training()
        test_language_encoder()
        
        print("\n" + "=" * 60)
        print("🎉 所有编码器测试通过!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(run_all_tests())
