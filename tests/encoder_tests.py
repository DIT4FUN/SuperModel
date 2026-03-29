"""
编码器模块 pytest 测试
=====================
"""

import torch
import numpy as np
import sys
import unittest

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.encoders import (
    VisionEncoder, AudioEncoder, TactileEncoder,
    ForceEncoder, IMUEncoder, LanguageEncoder, MultiModalEncoder,
    SensorEncoderWrapper, EncoderConfig,
    create_sensor_encoder, get_encoder_config,
    ENCODER_GRADES
)


class TestVisionEncoder(unittest.TestCase):
    """测试视觉编码器"""

    def test_encoder_init(self):
        enc = VisionEncoder(latent_dim=256, hidden_dim=256)
        self.assertIsNotNone(enc)

    def test_monocular_forward(self):
        enc = VisionEncoder(latent_dim=256, hidden_dim=256)
        x = torch.randn(4, 3, 224, 224)
        features = enc(x)
        self.assertEqual(features.shape, (4, 256))

    def test_stereo_forward(self):
        enc = VisionEncoder(latent_dim=256, hidden_dim=256)
        left = torch.randn(4, 3, 224, 224)
        right = torch.randn(4, 3, 224, 224)
        features = enc(left, right)
        self.assertEqual(features.shape, (4, 256))

    def test_single_sample(self):
        enc = VisionEncoder(latent_dim=128, hidden_dim=128)
        x = torch.randn(1, 3, 112, 112)
        features = enc(x)
        self.assertEqual(features.shape, (1, 128))

    def test_gradient_flow(self):
        enc = VisionEncoder(latent_dim=128, hidden_dim=128)
        x = torch.randn(2, 3, 112, 112, requires_grad=True)
        features = enc(x)
        loss = features.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_eval_mode_deterministic(self):
        enc = VisionEncoder(latent_dim=256, hidden_dim=256)
        enc.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            f1 = enc(x)
            f2 = enc(x)
        self.assertTrue(torch.allclose(f1, f2, atol=1e-6))


class TestAudioEncoder(unittest.TestCase):
    """测试音频编码器"""

    def test_encoder_init(self):
        enc = AudioEncoder(latent_dim=256, hidden_dim=256)
        self.assertIsNotNone(enc)

    def test_binaural_forward(self):
        enc = AudioEncoder(latent_dim=256, hidden_dim=256, n_mels=64)
        left = torch.randn(4, 100, 64)
        right = torch.randn(4, 100, 64)
        features = enc(left, right)
        self.assertEqual(features.shape, (4, 256))

    def test_single_channel(self):
        enc = AudioEncoder(latent_dim=128, hidden_dim=128, n_mels=32)
        left = torch.randn(2, 50, 32)
        right = torch.randn(2, 50, 32)
        features = enc(left, right)
        self.assertEqual(features.shape, (2, 128))

    def test_gradient_flow(self):
        enc = AudioEncoder(latent_dim=128, hidden_dim=128, n_mels=64)
        left = torch.randn(2, 50, 64, requires_grad=True)
        right = torch.randn(2, 50, 64)
        features = enc(left, right)
        loss = features.sum()
        loss.backward()
        self.assertIsNotNone(left.grad)


class TestTactileEncoder(unittest.TestCase):
    """测试触觉编码器"""

    def test_encoder_init(self):
        enc = TactileEncoder(latent_dim=256, hidden_dim=256)
        self.assertIsNotNone(enc)

    def test_forward(self):
        enc = TactileEncoder(latent_dim=256, hidden_dim=256)
        x = torch.randn(4, 1, 16, 16)
        features = enc(x)
        self.assertEqual(features.shape, (4, 256))

    def test_different_sizes(self):
        enc = TactileEncoder(latent_dim=128, hidden_dim=128)
        for size in [(8, 8), (16, 16), (24, 24), (32, 32)]:
            x = torch.randn(2, 1, *size)
            features = enc(x)
            self.assertEqual(features.shape, (2, 128))

    def test_gradient_flow(self):
        enc = TactileEncoder(latent_dim=128, hidden_dim=128)
        x = torch.randn(2, 1, 16, 16, requires_grad=True)
        features = enc(x)
        loss = features.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)


class TestForceEncoder(unittest.TestCase):
    """测试力觉编码器"""

    def test_encoder_init(self):
        enc = ForceEncoder(latent_dim=256, hidden_dim=256)
        self.assertIsNotNone(enc)

    def test_single_frame(self):
        enc = ForceEncoder(latent_dim=256, hidden_dim=256)
        x = torch.randn(4, 6)
        features = enc(x)
        self.assertEqual(features.shape, (4, 256))

    def test_sequence(self):
        enc = ForceEncoder(latent_dim=256, hidden_dim=256)
        x = torch.randn(4, 10, 6)
        features = enc(x)
        self.assertEqual(features.shape, (4, 256))

    def test_gradient_flow(self):
        enc = ForceEncoder(latent_dim=128, hidden_dim=128)
        x = torch.randn(2, 10, 6, requires_grad=True)
        features = enc(x)
        loss = features.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)


class TestIMUEncoder(unittest.TestCase):
    """测试IMU编码器"""

    def test_encoder_init(self):
        enc = IMUEncoder(latent_dim=256, hidden_dim=256)
        self.assertIsNotNone(enc)

    def test_single_frame(self):
        enc = IMUEncoder(latent_dim=256, hidden_dim=256)
        x = torch.randn(4, 6)
        features, quat = enc(x)
        self.assertEqual(features.shape, (4, 256))
        self.assertEqual(quat.shape, (4, 4))

    def test_sequence(self):
        enc = IMUEncoder(latent_dim=256, hidden_dim=256)
        x = torch.randn(4, 10, 6)
        features, quat = enc(x)
        self.assertEqual(features.shape, (4, 256))
        self.assertEqual(quat.shape, (4, 4))

    def test_gradient_flow(self):
        enc = IMUEncoder(latent_dim=128, hidden_dim=128)
        x = torch.randn(2, 10, 6, requires_grad=True)
        features, quat = enc(x)
        loss = features.sum() + quat.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_quaternion_normalized(self):
        enc = IMUEncoder(latent_dim=256, hidden_dim=256)
        x = torch.randn(4, 6)
        _, quat = enc(x)
        norms = torch.norm(quat, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))


class TestLanguageEncoder(unittest.TestCase):
    """测试语言编码器"""

    def test_encoder_init(self):
        enc = LanguageEncoder(vocab_size=5000, embed_dim=128, hidden_dim=256, max_len=32)
        self.assertIsNotNone(enc)

    def test_forward(self):
        enc = LanguageEncoder(vocab_size=5000, embed_dim=128, hidden_dim=256, max_len=32)
        token_ids = torch.randint(0, 5000, (4, 20))
        features = enc(token_ids)
        self.assertEqual(features.shape, (4, 256))

    def test_variable_length(self):
        enc = LanguageEncoder(vocab_size=5000, embed_dim=64, hidden_dim=128, max_len=32)
        for length in [1, 8, 16, 32]:
            token_ids = torch.randint(0, 5000, (2, length))
            features = enc(token_ids)
            self.assertEqual(features.shape, (2, 128))

    def test_deterministic(self):
        enc = LanguageEncoder(vocab_size=5000, embed_dim=64, hidden_dim=128, max_len=32)
        enc.eval()
        token_ids = torch.randint(0, 5000, (4, 16))
        with torch.no_grad():
            f1 = enc(token_ids)
            f2 = enc(token_ids)
        self.assertTrue(torch.allclose(f1, f2, atol=1e-6))

    def test_different_inputs_different_outputs(self):
        enc = LanguageEncoder(vocab_size=5000, embed_dim=64, hidden_dim=128, max_len=32)
        enc.eval()
        t1 = torch.randint(0, 5000, (4, 16))
        t2 = torch.randint(0, 5000, (4, 16))
        with torch.no_grad():
            f1 = enc(t1)
            f2 = enc(t2)
        distance = torch.norm(f1 - f2, dim=1).mean()
        self.assertGreater(distance.item(), 0.01)

    def test_gradient_flow(self):
        enc = LanguageEncoder(vocab_size=5000, embed_dim=64, hidden_dim=128, max_len=32)
        token_ids = torch.randint(0, 5000, (2, 16))
        features = enc(token_ids)
        loss = features.sum()
        loss.backward()
        # 嵌入层应该有梯度
        has_grad = any(
            p.grad is not None
            for p in enc.parameters()
            if p.requires_grad
        )
        self.assertTrue(has_grad)


class TestMultiModalEncoder(unittest.TestCase):
    """测试多模态编码器"""

    def setUp(self):
        self.enc = MultiModalEncoder(latent_dim=256)

    def test_encoder_init(self):
        self.assertEqual(self.enc.latent_dim, 256)

    def test_all_modalities(self):
        features = self.enc(
            vision_left=torch.randn(2, 3, 224, 224),
            vision_right=torch.randn(2, 3, 224, 224),
            audio_left=torch.randn(2, 100, 64),
            audio_right=torch.randn(2, 100, 64),
            tactile=torch.randn(2, 1, 16, 16),
            force=torch.randn(2, 10, 6),
            imu=torch.randn(2, 10, 6)
        )
        self.assertIn('vision', features)
        self.assertIn('audio', features)
        self.assertIn('tactile', features)
        self.assertIn('force', features)
        self.assertIn('imu', features)
        self.assertIn('fused', features)
        self.assertEqual(features['fused'].shape, (2, 256))

    def test_partial_modalities(self):
        features = self.enc(
            vision_left=torch.randn(2, 3, 112, 112),
            vision_right=torch.randn(2, 3, 112, 112),
        )
        self.assertIn('vision', features)
        self.assertIn('fused', features)

    def test_no_modalities(self):
        features = self.enc()
        self.assertIn('fused', features)

    def test_batch_consistency(self):
        for bs in [1, 2, 4, 8]:
            features = self.enc(
                vision_left=torch.randn(bs, 3, 112, 112),
                vision_right=torch.randn(bs, 3, 112, 112),
            )
            self.assertEqual(features['fused'].shape[0], bs)


class TestSensorEncoderWrapper(unittest.TestCase):
    """测试传感器编码器封装"""

    def test_wrapper_init(self):
        obs_dims = {
            'vision': 512, 'audio': 128, 'tactile': 64,
            'force': 32, 'imu': 64
        }
        wrapper = SensorEncoderWrapper(obs_dims, latent_dim=256)
        self.assertEqual(wrapper.latent_dim, 256)

    def test_encode_observations(self):
        obs_dims = {
            'vision': 512, 'audio': 128, 'tactile': 64,
            'force': 32, 'imu': 64
        }
        wrapper = SensorEncoderWrapper(obs_dims, latent_dim=256)
        observations = {
            'vision': torch.randn(2, 3, 112, 112),
            'audio': torch.randn(2, 50, 64),
            'tactile': torch.randn(2, 1, 8, 8),
            'force': torch.randn(2, 5, 6),
            'imu': torch.randn(2, 5, 6)
        }
        encoded = wrapper(observations)
        self.assertIn('fused', encoded)
        self.assertEqual(encoded['fused'].shape[0], 2)


class TestEncoderConfig(unittest.TestCase):
    """测试编码器配置"""

    def test_encoder_config_init(self):
        config = EncoderConfig(
            vision_dim=512, audio_dim=128, latent_dim=256
        )
        self.assertEqual(config.vision_dim, 512)
        self.assertEqual(config.audio_dim, 128)

    def test_encoder_grades(self):
        self.assertIn('S', ENCODER_GRADES)
        self.assertIn('M', ENCODER_GRADES)
        self.assertIn('L', ENCODER_GRADES)
        self.assertIn('XL', ENCODER_GRADES)
        self.assertIn('XXL', ENCODER_GRADES)

    def test_get_encoder_config(self):
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            config = get_encoder_config(grade)
            self.assertGreater(config.latent_dim, 0)
            self.assertGreater(config.hidden_dim, 0)


class TestCreateSensorEncoder(unittest.TestCase):
    """测试传感器编码器工厂函数"""

    def test_create_wrapper_s_grade(self):
        obs_dims = {'vision': 512, 'audio': 64, 'force': 16, 'imu': 32}
        enc = create_sensor_encoder(obs_dims, grade='S')
        self.assertIsInstance(enc, SensorEncoderWrapper)

    def test_create_wrapper_m_grade(self):
        obs_dims = {'vision': 512, 'audio': 128, 'force': 32, 'imu': 64}
        enc = create_sensor_encoder(obs_dims, grade='M')
        self.assertIsInstance(enc, SensorEncoderWrapper)

    def test_create_wrapper_xxl_grade(self):
        obs_dims = {'vision': 2048, 'audio': 1024, 'force': 256, 'imu': 512}
        enc = create_sensor_encoder(obs_dims, grade='XXL')
        self.assertIsInstance(enc, SensorEncoderWrapper)


class TestEncoderTraining(unittest.TestCase):
    """测试编码器训练流程"""

    def test_multimodal_training_step(self):
        enc = MultiModalEncoder(latent_dim=256)
        optimizer = torch.optim.Adam(enc.parameters(), lr=1e-4)

        features = enc(
            vision_left=torch.randn(4, 3, 112, 112),
            vision_right=torch.randn(4, 3, 112, 112),
            audio_left=torch.randn(4, 50, 64),
            audio_right=torch.randn(4, 50, 64),
            tactile=torch.randn(4, 1, 8, 8),
            force=torch.randn(4, 5, 6),
            imu=torch.randn(4, 5, 6)
        )

        loss = features['fused'].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        has_grad = any(
            p.grad is not None
            for p in enc.parameters()
            if p.requires_grad
        )
        self.assertTrue(has_grad)


if __name__ == '__main__':
    unittest.main(verbosity=2)
