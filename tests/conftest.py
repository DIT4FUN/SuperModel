"""
pytest 配置和共享 fixtures
========================

提供测试所需的共享 fixtures:
- 传感器实例
- 融合网络
- 控制器
- 仿真环境
"""

import pytest
import sys
import os
import numpy as np
import torch

_ProjectRoot = '/home/treeman/.openclaw/workspace/projects/SuperModel'
# src/ must be inserted BEFORE project_root so project_root ends up at index 0
# (Python searches from index 0, so project_root must be found first for fusion imports)
# src/fusion/ lacks sensor_fusion.py; project_root/fusion/ has it
# Use explicit 'from src.sensors.xxx' imports to avoid stale project_root/sensors/
sys.path.insert(0, _ProjectRoot)  # project_root at index 0 (for fusion imports)
sys.path.insert(0, os.path.join(_ProjectRoot, 'src'))  # src/ → will be pushed to index 1

from sensors.vision import BinocularCamera, CameraIntrinsics, StereoExtrinsics
from sensors.audio import BinauralMic
from sensors.tactile import TactileArray, PressureProcessor
from sensors.force import ForceTorqueSensor, WrenchProcessor, ForceSensorType
from sensors.imu import IMUSensor, PoseEstimator, IMUSensorType
from fusion.cross_modal_fusion import CrossModalFusion, FusionConfig, MultimodalInput
from control.agv import AGVMotionController, AGVSpec, AGVGrade, AGVPose, AGVTwist
from simulation.environment import RobotSimulator, SensorSimulator, SimConfig


# ─────────────────────────────────────────────────────────────────────────────
# Pytest Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def vision_sensor():
    """双目相机传感器 fixture"""
    cam = BinocularCamera(resolution=(640, 480), fps=30)
    cam.open()
    yield cam
    cam.close()


@pytest.fixture
def audio_sensor():
    """双耳麦克风 fixture"""
    mic = BinauralMic(sample_rate=16000, chunk_size=512)
    mic.open()
    yield mic
    mic.close()


@pytest.fixture
def tactile_sensor():
    """触觉传感器 fixture"""
    tactile = TactileArray(array_size=(16, 16), sensor_id="test_tactile")
    tactile.open()
    yield tactile
    tactile.close()


@pytest.fixture
def force_sensor():
    """六维力矩传感器 fixture"""
    sensor = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS, sensor_id="test_force")
    sensor.open()
    yield sensor
    sensor.close()


@pytest.fixture
def imu_sensor():
    """IMU传感器 fixture"""
    imu = IMUSensor(sensor_type=IMUSensorType.VIRTUAL, sensor_id="test_imu")
    imu.open()
    yield imu
    imu.close()


@pytest.fixture
def pressure_processor():
    """压力处理器 fixture"""
    return PressureProcessor(filter_window=3)


@pytest.fixture
def wrench_processor():
    """力矩处理器 fixture"""
    return WrenchProcessor(filter_alpha=0.3)


@pytest.fixture
def pose_estimator():
    """姿态估计器 fixture"""
    return PoseEstimator(algorithm="madgwick", sample_rate=100.0)


@pytest.fixture
def fusion_network():
    """跨模态融合网络 fixture"""
    config = FusionConfig(
        vision_dim=512, audio_dim=128,
        tactile_dim=64, force_dim=32, imu_dim=64,
        hidden_dim=256, num_heads=4, num_layers=2
    )
    return CrossModalFusion(config)


@pytest.fixture
def fusion_network_tiny():
    """小型融合网络 fixture (快速测试)"""
    config = FusionConfig(
        vision_dim=128, audio_dim=64,
        tactile_dim=32, force_dim=16, imu_dim=32,
        hidden_dim=128, num_heads=2, num_layers=1
    )
    return CrossModalFusion(config)


@pytest.fixture
def agv_controller_m():
    """AGV M级控制器 fixture"""
    spec = AGVSpec.from_grade(AGVGrade.M)
    return AGVMotionController(spec)


@pytest.fixture
def agv_controller_xxl():
    """AGV XXL级控制器 fixture"""
    spec = AGVSpec.from_grade(AGVGrade.XXL)
    return AGVMotionController(spec)


@pytest.fixture
def robot_simulator():
    """机器人仿真器 fixture"""
    config = SimConfig(num_joints=6, dt=0.01)
    return RobotSimulator(config)


@pytest.fixture
def sensor_simulator(robot_simulator):
    """传感器仿真器 fixture"""
    config = SimConfig(num_joints=6, dt=0.01)
    return SensorSimulator(robot_simulator, config)


@pytest.fixture
def all_sensors():
    """所有传感器实例 (快速创建/销毁)"""
    sensors = {
        'vision': BinocularCamera(resolution=(640, 480), fps=30),
        'audio': BinauralMic(sample_rate=16000, chunk_size=512),
        'tactile': TactileArray(array_size=(16, 16), sensor_id="all_test"),
        'force': ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS, sensor_id="all_test"),
        'imu': IMUSensor(sensor_type=IMUSensorType.VIRTUAL, sensor_id="all_test"),
    }
    for s in sensors.values():
        s.open()
    yield sensors
    for s in sensors.values():
        s.close()


@pytest.fixture
def sample_multimodal_input():
    """示例多模态输入 fixture"""
    return MultimodalInput(
        vision=torch.randn(2, 512),
        audio=torch.randn(2, 128),
        tactile=torch.randn(2, 64),
        force=torch.randn(2, 32),
        imu=torch.randn(2, 64),
        language=torch.randint(0, 10000, (2, 32))
    )


@pytest.fixture
def sample_agv_poses():
    """示例AGV位姿列表 fixture"""
    return [
        AGVPose(x=0.0, y=0.0, theta=0.0),
        AGVPose(x=0.5, y=0.0, theta=0.0),
        AGVPose(x=1.0, y=0.5, theta=np.pi/4),
        AGVPose(x=1.5, y=1.0, theta=np.pi/2),
        AGVPose(x=2.0, y=1.0, theta=np.pi),
    ]


@pytest.fixture
def camera_intrinsics():
    """相机内参 fixture"""
    return CameraIntrinsics(width=640, height=480, fx=385.5, fy=385.5, cx=319.5, cy=239.5)


@pytest.fixture
def stereo_extrinsics():
    """双目外参 fixture"""
    return StereoExtrinsics(rotation=np.eye(3), translation=np.array([-0.05, 0.0, 0.0]))


# ─────────────────────────────────────────────────────────────────────────────
# Parametrized fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(params=['S', 'M', 'L', 'XL', 'XXL'])
def agv_grade(request):
    """AGV所有等级参数化 fixture"""
    return request.param


@pytest.fixture(params=['S', 'M', 'L', 'XL', 'XXL'])
def fusion_grade(request):
    """融合网络等级参数化 fixture"""
    grade_specs = {
        'S': dict(vision_dim=256, audio_dim=64, hidden_dim=128, num_heads=2),
        'M': dict(vision_dim=512, audio_dim=128, hidden_dim=256, num_heads=4),
        'L': dict(vision_dim=512, audio_dim=128, hidden_dim=512, num_heads=8),
        'XL': dict(vision_dim=768, audio_dim=256, hidden_dim=768, num_heads=12),
        'XXL': dict(vision_dim=1024, audio_dim=512, hidden_dim=1024, num_heads=16),
    }
    kwargs = grade_specs[request.param]
    kwargs.update(dict(
        tactile_dim=64, force_dim=32, imu_dim=64,
        num_layers=2
    ))
    return request.param, FusionConfig(**kwargs)
