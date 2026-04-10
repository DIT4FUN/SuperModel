"""
SuperModel 硬件支持模块
======================

支持多种机器人主板平台:
- RK3588 系列 (RK3588, RK3588S)
- 地瓜机器人 RDK 系列 (RDK X3, X5 Ultra, S100)

提供统一的硬件抽象接口，适配不同的计算平台和外设配置。
"""

from .base import (
    BoardBase, BoardInfo, BoardType,
    PeripheralType, ComputeCapability,
    create_board, detect_board
)
from .rk3588 import RK3588Platform, RK3588BoardInfo
from .digu_robot import (
    DiguRobotPlatform,
    RDKX3, RDKX5Ultra, RDKS100,
    RDKBoardInfo, DiguRobotSeries
)
from .gpio import GPIOController, PinMode, PinState
from .nnpu import NPUAccelerator, NPUPlugin, get_npu_context
from .predictive_maintenance import (
    PredictiveMaintenanceSystem, MotorHealthMonitor, BatterySOHEstimator, WheelHealthMonitor,
    HealthLevel, FaultType, AGVHealthReport,
    MotorHealthMetrics, BatteryHealthMetrics, WheelHealthMetrics,
    get_predictive_maintenance_spec, AGV_PREDICTIVE_MAINTENANCE_GRADES,
    create_predictive_maintenance_system
)
from .canbus import (
    CANFrame, CANBusState, CANopenNodeState, CANopenPDO, SensorCANConfig,
    VirtualCANBus, RealCANBus, create_can_bus,
    CANopenNode, IMUCANopenNode, ForceTorqueCANopenNode, TactileCANopenNode,
    AGV_CAN_GRADES, get_can_spec,
)
from .sensor_bridge import (
    SensorDataType, SensorProtocol, SensorHealth,
    SensorData, SensorHardwareConfig,
    SensorHardwareInterface, SimulatedSensorInterface,
    SensorHardwareBridge,
    AGV_SENSOR_BRIDGE_GRADES, get_bridge_spec,
)

__all__ = [
    # 基础抽象
    'BoardBase', 'BoardInfo', 'BoardType',
    'PeripheralType', 'ComputeCapability',
    'create_board', 'detect_board',
    # RK3588 平台
    'RK3588Platform', 'RK3588BoardInfo',
    # 地瓜机器人
    'DiguRobotPlatform', 'RDKX3', 'RDKX5Ultra', 'RDKS100',
    'RDKBoardInfo', 'DiguRobotSeries',
    # GPIO 控制
    'GPIOController', 'PinMode', 'PinState',
    # NPU 加速
    'NPUAccelerator', 'NPUPlugin', 'get_npu_context',
    # 预测性维护
    'PredictiveMaintenanceSystem', 'MotorHealthMonitor', 'BatterySOHEstimator', 'WheelHealthMonitor',
    'HealthLevel', 'FaultType', 'AGVHealthReport',
    'MotorHealthMetrics', 'BatteryHealthMetrics', 'WheelHealthMetrics',
    'get_predictive_maintenance_spec', 'AGV_PREDICTIVE_MAINTENANCE_GRADES',
    'create_predictive_maintenance_system',
    # CAN Bus 传感器接口
    'CANFrame', 'CANBusState', 'CANopenNodeState', 'CANopenPDO', 'SensorCANConfig',
    'VirtualCANBus', 'RealCANBus', 'create_can_bus',
    'CANopenNode', 'IMUCANopenNode', 'ForceTorqueCANopenNode', 'TactileCANopenNode',
    'AGV_CAN_GRADES', 'get_can_spec',
    # 传感器硬件桥接器
    'SensorDataType', 'SensorProtocol', 'SensorHealth',
    'SensorData', 'SensorHardwareConfig',
    'SensorHardwareInterface', 'SimulatedSensorInterface',
    'SensorHardwareBridge',
    'AGV_SENSOR_BRIDGE_GRADES', 'get_bridge_spec',
]
