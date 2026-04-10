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
from .tactile import (
    TactileArray, TactileFrame, TactileContact, TactileCalibration,
    TactileSensorType, PressureProcessor, VirtualTactileSensor,
    get_tactile_spec, AGV_TACTILE_GRADES
)
from .force import (
    ForceTorqueSensor, Wrench, ForceCalibration, ContactState,
    ForceSensorType, WrenchProcessor, VirtualForceSensor,
    get_force_spec, AGV_FORCE_GRADES
)
from .imu import (
    IMUSensor, IMUFrame, Pose, PoseEstimator, IMUCalibration,
    IMUSensorType, VirtualIMUSensor, get_imu_spec, AGV_IMU_GRADES
)
from .encoders import (
    VisionEncoder, AudioEncoder, TactileEncoder, ForceEncoder,
    IMUEncoder, LanguageEncoder, SensorEncoderWrapper, EncoderConfig,
    ENCODER_GRADES, create_sensor_encoder, get_encoder_config
)
from .manager import SensorManager, SensorManagerConfig, SensorDataFrame, SensorGrade
from .signal_processor import (
    SignalProcessor, KalmanFilter1D, KalmanFilter3D, ButterworthFilter,
    MedianFilter, ExponentialSmoother, OutlierDetector,
    FilterConfig, FilterType, SignalStats,
    AGV_SIGNAL_PROCESSING_GRADES, get_signal_processing_grade_spec
)

__all__ = [
    'BinocularCamera', 'DepthProcessor',
    'BinauralMic', 'SoundLocalizer',
    'TactileArray', 'TactileFrame', 'TactileContact', 'TactileCalibration',
    'TactileSensorType', 'PressureProcessor', 'VirtualTactileSensor',
    'get_tactile_spec', 'AGV_TACTILE_GRADES',
    'ForceTorqueSensor', 'Wrench', 'ForceCalibration', 'ContactState',
    'ForceSensorType', 'WrenchProcessor', 'VirtualForceSensor',
    'get_force_spec', 'AGV_FORCE_GRADES',
    'IMUSensor', 'IMUFrame', 'Pose', 'PoseEstimator', 'IMUCalibration',
    'IMUSensorType', 'VirtualIMUSensor', 'get_imu_spec', 'AGV_IMU_GRADES',
    'VisionEncoder', 'AudioEncoder', 'TactileEncoder', 'ForceEncoder',
    'IMUEncoder', 'LanguageEncoder', 'SensorEncoderWrapper', 'EncoderConfig',
    'ENCODER_GRADES', 'create_sensor_encoder', 'get_encoder_config',
    'SensorManager', 'SensorManagerConfig', 'SensorDataFrame', 'SensorGrade',
    'SignalProcessor', 'KalmanFilter1D', 'KalmanFilter3D', 'ButterworthFilter',
    'MedianFilter', 'ExponentialSmoother', 'OutlierDetector',
    'FilterConfig', 'FilterType', 'SignalStats',
    'AGV_SIGNAL_PROCESSING_GRADES', 'get_signal_processing_grade_spec',
]
