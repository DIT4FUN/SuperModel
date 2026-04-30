# Copyright (C) 2024-2026 赵元请 (DIT4FUN)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
    TactileArray, TactileFrame, TactileReading, ContactEvent,
    TactileSensorType, AGVTactileBumper, TactileGlove,
    TactileContact, TactileCalibration, PressureProcessor, VirtualTactileSensor,
    get_tactile_spec, AGV_TACTILE_GRADES
)
from .force import (
    SixAxisForceTorque, ForceReading, Wrench, WheelForceSensor, LiftForceSensor,
    ForceTorqueSensor, WrenchProcessor, ForceCalibration, ContactState, VirtualForceSensor,
    ForceSensorType, get_force_spec, AGV_FORCE_GRADES
)
from .imu import (
    IMU, IMUReading, Pose, IMUOdometry, IMUModel,
    IMUSensor, IMUFrame, PoseEstimator, IMUCalibration, IMUSensorType, VirtualIMUSensor,
    quaternion_to_rotation_matrix, get_imu_spec, AGV_IMU_GRADES
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
    'TactileArray', 'TactileFrame', 'TactileReading', 'ContactEvent',
    'TactileContact', 'TactileCalibration', 'PressureProcessor', 'VirtualTactileSensor',
    'TactileSensorType', 'AGVTactileBumper', 'TactileGlove',
    'get_tactile_spec', 'AGV_TACTILE_GRADES',
    'SixAxisForceTorque', 'ForceReading', 'Wrench', 'WheelForceSensor', 'LiftForceSensor',
    'ForceTorqueSensor', 'WrenchProcessor', 'ForceCalibration', 'ContactState', 'VirtualForceSensor',
    'ForceSensorType', 'get_force_spec', 'AGV_FORCE_GRADES',
    'IMU', 'IMUReading', 'Pose', 'IMUOdometry', 'IMUModel',
    'IMUSensor', 'IMUFrame', 'PoseEstimator', 'IMUCalibration', 'IMUSensorType', 'VirtualIMUSensor',
    'quaternion_to_rotation_matrix', 'get_imu_spec', 'AGV_IMU_GRADES',
    'VisionEncoder', 'AudioEncoder', 'TactileEncoder', 'ForceEncoder',
    'IMUEncoder', 'LanguageEncoder', 'SensorEncoderWrapper', 'EncoderConfig',
    'ENCODER_GRADES', 'create_sensor_encoder', 'get_encoder_config',
    'SensorManager', 'SensorManagerConfig', 'SensorDataFrame', 'SensorGrade',
    'SignalProcessor', 'KalmanFilter1D', 'KalmanFilter3D', 'ButterworthFilter',
    'MedianFilter', 'ExponentialSmoother', 'OutlierDetector',
    'FilterConfig', 'FilterType', 'SignalStats',
    'AGV_SIGNAL_PROCESSING_GRADES', 'get_signal_processing_grade_spec',
]
