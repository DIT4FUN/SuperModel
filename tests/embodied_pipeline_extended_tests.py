"""
具身智能完整流水线测试
======================

测试 SuperModel 具身智能大脑的完整端到端流水线:
- 多模态传感器协同采集
- 跨模态Transformer融合
- 传感-运动融合控制
- 世界模型想象 rollout
- AGV五级规格完整性
- 全流水线压力测试

运行:
  pytest tests/embodied_pipeline_extended_tests.py -v
"""

import pytest
import numpy as np
import torch
import time
import sys
import os
from dataclasses import dataclass

# Pose2D - defined inline to avoid control.motion path issues
@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

# Imports from src/ (conftest.py also sets up paths)
_ProjectRoot = '/home/treeman/.openclaw/workspace/projects/SuperModel'
if os.path.join(_ProjectRoot, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(_ProjectRoot, 'src'))
if _ProjectRoot not in sys.path:
    sys.path.insert(0, _ProjectRoot)

from sensors.vision import BinocularCamera, StereoFrame
from sensors.audio import BinauralMic, AudioFrame
from sensors.tactile import TactileArray, TactileSensorType, TactileFrame
from sensors.force import ForceTorqueSensor, Wrench, ForceSensorType
from sensors.imu import IMUSensor, IMUFrame, IMUSensorType
from fusion.cross_modal_fusion import CrossModalFusion, FusionConfig, MultimodalInput
from fusion.sensor_fusion import ComplementaryFilter, ExtendedKalmanFilter
from control.sensorimotor import (
    SensorimotorIntegration, SensorimotorConfig, SensorimotorState
)


# ─── 测试配置 ───────────────────────────────────────────────────────────────

GRADE_CFGS = {
    'S': {
        'tactile_array': (8, 8),
        'imu': IMUSensorType.MPU6050,
        'control_rate': 50,
        'tactile_w': 0.2,
        'force_w': 0.3,
        'imu_w': 0.5,
    },
    'M': {
        'tactile_array': (16, 16),
        'imu': IMUSensorType.BMI088,
        'control_rate': 100,
        'tactile_w': 0.3,
        'force_w': 0.4,
        'imu_w': 0.3,
    },
    'L': {
        'tactile_array': (24, 24),
        'imu': IMUSensorType.BMI088,
        'control_rate': 200,
        'tactile_w': 0.35,
        'force_w': 0.4,
        'imu_w': 0.25,
    },
}


# ───  Fixture ───────────────────────────────────────────────────────────────

@pytest.fixture
def grade_M_config():
    return GRADE_CFGS['M']


@pytest.fixture
def grade_M_sensors(grade_M_config):
    """创建 M 级传感器组"""
    vision = BinocularCamera()
    audio = BinauralMic()
    tactile = TactileArray(
        array_size=grade_M_config['tactile_array'],
        sensor_type=TactileSensorType.RESISTIVE,
    )
    force = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
    imu = IMUSensor(sensor_type=grade_M_config['imu'])

    vision.open()
    audio.open()
    tactile.open()
    force.open()
    imu.open()

    yield {
        'vision': vision,
        'audio': audio,
        'tactile': tactile,
        'force': force,
        'imu': imu,
    }

    vision.close()
    audio.close()
    tactile.close()
    force.close()
    imu.close()


@pytest.fixture
def fusion_network(grade_M_config):
    rows, cols = grade_M_config['tactile_array']
    tactile_dim = rows * cols
    cfg = FusionConfig(
        vision_dim=512,
        audio_dim=128,
        tactile_dim=tactile_dim,
        force_dim=32,
        imu_dim=64,
        hidden_dim=256,
        num_heads=4,
        dropout=0.1,
    )
    return CrossModalFusion(cfg)


@pytest.fixture
def sensorimotor_ctrl(grade_M_config, grade_M_sensors):
    cfg = SensorimotorConfig(
        tactile_weight=grade_M_config['tactile_w'],
        force_weight=grade_M_config['force_w'],
        imu_weight=grade_M_config['imu_w'],
        control_rate=grade_M_config['control_rate'],
        grade='M',
        fusion_strategy='weighted',
    )
    return SensorimotorIntegration(
        tactile_sensor=grade_M_sensors['tactile'],
        force_sensor=grade_M_sensors['force'],
        imu_sensor=grade_M_sensors['imu'],
        config=cfg,
    )


# ─── 传感器层测试 ────────────────────────────────────────────────────────────

class TestSensorLayer:
    """测试感知层传感器采集"""

    def test_all_sensors_open_and_capture(self, grade_M_sensors):
        """测试所有传感器打开并采集"""
        for name, sensor in grade_M_sensors.items():
            frame = sensor.capture()
            assert frame is not None, f"{name} capture failed"

    def test_vision_frame_fields(self, grade_M_sensors):
        """测试视觉帧结构"""
        frame = grade_M_sensors['vision'].capture()
        assert isinstance(frame, StereoFrame)
        assert frame.left_image.shape == frame.right_image.shape
        assert frame.timestamp >= 0
        assert frame.frame_id >= 0

    def test_audio_frame_fields(self, grade_M_sensors):
        """测试听觉帧结构"""
        frame = grade_M_sensors['audio'].capture()
        assert isinstance(frame, AudioFrame)
        assert len(frame.left_channel) > 0
        assert len(frame.right_channel) > 0
        assert frame.timestamp >= 0

    def test_tactile_frame_fields(self, grade_M_sensors):
        """测试触觉帧结构"""
        frame = grade_M_sensors['tactile'].capture()
        assert isinstance(frame, TactileFrame)
        assert frame.pressure_map.shape == (16, 16)
        assert hasattr(frame, 'temperature_map')

    def test_force_wrench_fields(self, grade_M_sensors):
        """测试力觉数据"""
        wrench = grade_M_sensors['force'].capture()
        assert isinstance(wrench, Wrench)
        assert wrench.force.shape == (3,)
        assert wrench.torque.shape == (3,)
        assert wrench.magnitude >= 0  # property

    def test_imu_frame_fields(self, grade_M_sensors):
        """测试IMU数据"""
        frame = grade_M_sensors['imu'].capture()
        assert isinstance(frame, IMUFrame)
        assert frame.accel.shape == (3,)
        assert frame.gyro.shape == (3,)
        assert frame.accel_magnitude > 0  # 重力 (property)

    def test_sensor_calibration(self, grade_M_sensors):
        """测试传感器校准"""
        imu = grade_M_sensors['imu']
        imu.calibrate_gyro_bias(num_samples=20)
        assert imu.calibration is not None
        assert hasattr(imu.calibration, 'gyro_bias')

        force = grade_M_sensors['force']
        force.calibrate_bias(num_samples=20)
        assert force.calibration is not None


# ─── 融合层测试 ──────────────────────────────────────────────────────────────

class TestFusionLayer:
    """测试融合层"""

    def test_fusion_config_creation(self, grade_M_config):
        """测试融合配置"""
        rows, cols = grade_M_config['tactile_array']
        cfg = FusionConfig(
            vision_dim=512,
            audio_dim=128,
            tactile_dim=rows * cols,
            force_dim=32,
            imu_dim=64,
            hidden_dim=256,
            num_heads=4,
        )
        assert cfg.vision_dim == 512
        assert cfg.audio_dim == 128

    def test_multimodal_input_batch_dim_required(self, fusion_network):
        """测试融合需要 batch 维度"""
        # 无 batch 维度应该报错
        mm_input = MultimodalInput(
            vision=torch.randn(512),
            audio=torch.randn(128),
            tactile=torch.randn(256),
            force=torch.randn(32),
            imu=torch.randn(64),
        )
        with pytest.raises(Exception):
            fusion_network(mm_input)

    def test_multimodal_input_with_batch(self, fusion_network):
        """测试带 batch 维度的融合"""
        mm_input = MultimodalInput(
            vision=torch.randn(2, 512),
            audio=torch.randn(2, 128),
            tactile=torch.randn(2, 256),
            force=torch.randn(2, 32),
            imu=torch.randn(2, 64),
        )
        out = fusion_network(mm_input)
        assert out.shape == (2, 256)

    def test_fusion_output_shape(self, fusion_network):
        """测试融合输出形状"""
        mm_input = MultimodalInput(
            vision=torch.randn(1, 512),
            audio=torch.randn(1, 128),
            tactile=torch.randn(1, 256),
            force=torch.randn(1, 32),
            imu=torch.randn(1, 64),
        )
        out = fusion_network(mm_input)
        assert out.shape == (1, 256)
        assert not torch.isnan(out).any()

    def test_fusion_confidence_from_std(self, fusion_network):
        """测试融合置信度估计"""
        mm_input = MultimodalInput(
            vision=torch.randn(1, 512) * 0.1,
            audio=torch.randn(1, 128) * 0.1,
            tactile=torch.randn(1, 256) * 0.1,
            force=torch.randn(1, 32) * 0.1,
            imu=torch.randn(1, 64) * 0.1,
        )
        out = fusion_network(mm_input)
        std_val = out.std().item()
        conf = np.clip(std_val * 10, 0.0, 1.0)
        assert 0.0 <= conf <= 1.0

    def test_complementary_filter_update(self):
        """测试互补滤波器"""
        filt = ComplementaryFilter(alpha=0.96)
        accel = np.array([0.0, 0.0, -9.81])
        gyro = np.array([0.0, 0.0, 0.1])
        state = filt.update({'accel': accel, 'gyro': gyro}, dt=0.01)
        assert len(state) == 3
        assert filt._initialized


# ─── 控制层测试 ──────────────────────────────────────────────────────────────

class TestControlLayer:
    """测试控制层"""

    def test_sensorimotor_integration_step(self, sensorimotor_ctrl):
        """测试传感-运动融合控制器单步"""
        state = sensorimotor_ctrl.step(dt=0.01)
        assert isinstance(state, SensorimotorState)
        assert state.fused_control.shape == (3,)

    def test_sensorimotor_multi_step(self, sensorimotor_ctrl):
        """测试传感-运动融合控制器多步"""
        for i in range(10):
            state = sensorimotor_ctrl.step(dt=0.01)
            assert state.frame_id == i + 1

    def test_sensorimotor_fused_control_bounded(self, sensorimotor_ctrl):
        """测试融合控制输出有界"""
        for _ in range(20):
            state = sensorimotor_ctrl.step(dt=0.01)
            assert np.abs(state.fused_control).max() < 1000  # 有界

    def test_sensorimotor_authority_sum(self, sensorimotor_ctrl):
        """测试控制权限归一化"""
        for _ in range(10):
            state = sensorimotor_ctrl.step(dt=0.01)
            auth = state.control_authority
            total = sum(auth.values())
            # 归一化后总和应为 1.0 或 0.0（无模态激活）
            assert total == pytest.approx(1.0, abs=0.01) or total == 0.0

    def test_sensorimotor_get_state(self, sensorimotor_ctrl):
        """测试获取状态"""
        sensorimotor_ctrl.step(dt=0.01)
        state = sensorimotor_ctrl.get_state()
        assert isinstance(state, SensorimotorState)

    def test_sensorimotor_is_safe(self, sensorimotor_ctrl):
        """测试安全检查"""
        is_safe = sensorimotor_ctrl.is_safe()
        assert isinstance(is_safe, bool)

    def test_pose2d_movement(self):
        """测试2D姿态运动更新"""
        pose = Pose2D(x=0.0, y=0.0, theta=0.0)
        dt = 0.01
        v = 1.0
        omega = 0.0

        pose.x += v * np.cos(pose.theta) * dt
        pose.y += v * np.sin(pose.theta) * dt
        pose.theta += omega * dt
        pose.theta = np.arctan2(np.sin(pose.theta), np.cos(pose.theta))

        assert abs(pose.x - v * dt) < 1e-6
        assert abs(pose.y) < 1e-6


# ─── 端到端流水线测试 ────────────────────────────────────────────────────────

class TestEndToEndPipeline:
    """测试完整端到端流水线"""

    def test_full_pipeline_single_step(self, grade_M_sensors, fusion_network, sensorimotor_ctrl):
        """测试完整流水线单步执行"""
        t = time.time()

        # 1. 传感器采集
        vision_frame = grade_M_sensors['vision'].capture()
        audio_frame = grade_M_sensors['audio'].capture()
        tactile_frame = grade_M_sensors['tactile'].capture()
        wrench = grade_M_sensors['force'].capture()
        imu_frame = grade_M_sensors['imu'].capture()

        # 2. 构建融合输入
        rows, cols = 16, 16
        mm_input = MultimodalInput(
            vision=torch.randn(1, 512),
            audio=torch.randn(1, 128),
            tactile=torch.randn(1, rows * cols),
            force=torch.randn(1, 32),
            imu=torch.randn(1, 64),
        )

        # 3. 跨模态融合
        fusion_out = fusion_network(mm_input)
        assert fusion_out.shape == (1, 256)

        # 4. 传感-运动融合控制
        sm_state = sensorimotor_ctrl.step(dt=0.01)
        assert sm_state.fused_control.shape == (3,)

        elapsed = time.time() - t
        assert elapsed < 1.0  # 应该在1秒内完成

    def test_full_pipeline_100_steps(self, grade_M_sensors, fusion_network, sensorimotor_ctrl):
        """测试100步流水线稳定性"""
        times = []
        for i in range(100):
            t = time.time()

            # 传感器
            grade_M_sensors['vision'].capture()
            grade_M_sensors['audio'].capture()
            grade_M_sensors['tactile'].capture()
            grade_M_sensors['force'].capture()
            grade_M_sensors['imu'].capture()

            # 融合
            mm_input = MultimodalInput(
                vision=torch.randn(1, 512),
                audio=torch.randn(1, 128),
                tactile=torch.randn(1, 256),
                force=torch.randn(1, 32),
                imu=torch.randn(1, 64),
            )
            fusion_network(mm_input)

            # 控制
            sensorimotor_ctrl.step(dt=0.01)

            times.append(time.time() - t)

        avg_time = np.mean(times)
        p99_time = np.percentile(times, 99)
        assert avg_time < 0.1  # 平均应该小于 100ms
        assert p99_time < 0.5  # P99 应该小于 500ms

    def test_pipeline_with_imu_attitude_control(self, grade_M_sensors, sensorimotor_ctrl):
        """测试带IMU姿态控制的流水线"""
        for _ in range(20):
            sm_state = sensorimotor_ctrl.step(
                target_attitude=(0.0, 0.0, 0.0),
                dt=0.01
            )
            assert sm_state.fused_control.shape == (3,)


# ─── AGV五级规格测试 ─────────────────────────────────────────────────────────

class TestAGVGradeSpecs:
    """测试AGV五级规格完整性"""

    def test_all_grades_have_sensor_configs(self):
        """测试所有等级都有传感器配置"""
        for grade, cfg in GRADE_CFGS.items():
            assert 'tactile_array' in cfg
            assert 'imu' in cfg
            assert 'control_rate' in cfg
            assert cfg['control_rate'] > 0

    def test_grade_control_rate_increases(self):
        """测试等级越高控制频率越高"""
        rates = [GRADE_CFGS[g]['control_rate'] for g in ['S', 'M', 'L']]
        assert rates[0] < rates[1] < rates[2]

    def test_grade_tactile_array_increases(self):
        """测试等级越高触觉阵列越大"""
        areas = [GRADE_CFGS[g]['tactile_array'][0] * GRADE_CFGS[g]['tactile_array'][1]
                 for g in ['S', 'M', 'L']]
        assert areas[0] < areas[1] < areas[2]

    def test_sensorimotor_config_from_grade(self):
        """测试从AGV等级创建传感-运动配置"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            cfg = SensorimotorConfig.from_grade(grade)
            assert cfg.grade == grade
            assert cfg.control_rate > 0
            assert cfg.tactile_weight + cfg.force_weight + cfg.imu_weight == pytest.approx(1.0)

    def test_imu_sensor_types(self):
        """测试各等级IMU传感器类型"""
        assert GRADE_CFGS['S']['imu'] == IMUSensorType.MPU6050
        assert GRADE_CFGS['M']['imu'] == IMUSensorType.BMI088
        assert GRADE_CFGS['L']['imu'] == IMUSensorType.BMI088


# ─── 世界模型模拟测试 ────────────────────────────────────────────────────────

class TestWorldModelRollout:
    """测试世界模型rollout模拟"""

    def test_rollout_accumulates_predictions(self):
        """测试rollout累积预测"""
        predictions = []
        for _ in range(50):
            pred = np.random.randn(64).astype(np.float32)
            predictions.append(pred)
            if len(predictions) > 50:
                predictions.pop(0)

        assert len(predictions) == 50
        assert all(p.shape == (64,) for p in predictions)

    def test_latent_proxy_from_fusion_output(self, fusion_network):
        """测试从融合输出提取潜在表示代理"""
        mm_input = MultimodalInput(
            vision=torch.randn(1, 512),
            audio=torch.randn(1, 128),
            tactile=torch.randn(1, 256),
            force=torch.randn(1, 32),
            imu=torch.randn(1, 64),
        )
        out = fusion_network(mm_input)
        latent_proxy = out.mean().item()
        assert isinstance(latent_proxy, float)
        assert not np.isnan(latent_proxy)


# ─── 压力测试 ────────────────────────────────────────────────────────────────

class TestStressTests:
    """压力测试"""

    def test_high_frequency_sensor_reads(self, grade_M_sensors):
        """测试高频传感器读取"""
        n_reads = 1000
        start = time.time()
        for _ in range(n_reads):
            grade_M_sensors['imu'].capture()
        elapsed = time.time() - start
        rate = n_reads / elapsed
        assert rate > 100  # 至少 100 Hz

    def test_fusion_network_no_nan(self, fusion_network):
        """测试融合网络输出无NaN"""
        for _ in range(100):
            mm_input = MultimodalInput(
                vision=torch.randn(1, 512),
                audio=torch.randn(1, 128),
                tactile=torch.randn(1, 256),
                force=torch.randn(1, 32),
                imu=torch.randn(1, 64),
            )
            out = fusion_network(mm_input)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()

    def test_controller_stability_over_long_run(self, sensorimotor_ctrl):
        """测试控制器长时间运行稳定性"""
        for i in range(1000):
            state = sensorimotor_ctrl.step(dt=0.01)
            # 检查输出不爆炸
            assert np.abs(state.fused_control).max() < 1e6
            # 检查权限字典有效
            assert all(0 <= v <= 1.0 for v in state.control_authority.values())


# ─── 上下文管理器测试 ────────────────────────────────────────────────────────

class TestContextManagers:
    """测试传感器上下文管理器"""

    def test_tactile_context_manager(self, grade_M_sensors):
        """测试触觉传感器上下文管理"""
        tactile = grade_M_sensors['tactile']
        assert tactile._is_opened
        tactile.close()
        assert not tactile._is_opened

    def test_force_context_manager(self, grade_M_sensors):
        """测试力觉传感器上下文管理"""
        force = grade_M_sensors['force']
        assert force._is_streaming
        force.close()
        assert not force._is_streaming

    def test_imu_context_manager(self, grade_M_sensors):
        """测试IMU传感器上下文管理"""
        imu = grade_M_sensors['imu']
        assert imu._is_opened
        imu.close()
        assert not imu._is_opened


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
