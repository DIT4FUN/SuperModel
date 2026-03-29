"""
SuperModel 超模态感知模块
=========================

多模态传感器接口与数据预处理

支持传感器:
- 双目RGBD相机 (RealSense D435i)
- 双耳麦克风阵列 (ReSpeaker)
- 电子皮肤触觉阵列 (Digi Sensing)
- 六维力矩传感器 (ATI)
- IMU/编码器 (惯性测量单元)
"""

from .vision import BinocularCamera, DepthProcessor
from .audio import BinauralMic, SoundLocalizer
from .tactile import TactileArray, PressureProcessor, VirtualTactileSensor
from .force import ForceTorqueSensor, WrenchProcessor, VirtualForceSensor
from .imu import IMUSensor, PoseEstimator, VirtualIMUSensor
from .encoders import (
    VisionEncoder, AudioEncoder, TactileEncoder, ForceEncoder,
    IMUEncoder, LanguageEncoder, SensorEncoderWrapper, EncoderConfig,
    ENCODER_GRADES, create_sensor_encoder, get_encoder_config
)
from .manager import SensorManager, SensorManagerConfig, SensorDataFrame, SensorGrade

__all__ = [
    'BinocularCamera', 'DepthProcessor',
    'BinauralMic', 'SoundLocalizer',
    'TactileArray', 'PressureProcessor', 'VirtualTactileSensor',
    'ForceTorqueSensor', 'WrenchProcessor', 'VirtualForceSensor',
    'IMUSensor', 'PoseEstimator', 'VirtualIMUSensor',
    'VisionEncoder', 'AudioEncoder', 'TactileEncoder', 'ForceEncoder',
    'IMUEncoder', 'LanguageEncoder', 'SensorEncoderWrapper', 'EncoderConfig',
    'ENCODER_GRADES', 'create_sensor_encoder', 'get_encoder_config',
    'SensorManager', 'SensorManagerConfig', 'SensorDataFrame', 'SensorGrade',
]
