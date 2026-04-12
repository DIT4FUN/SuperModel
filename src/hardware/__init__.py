"""
hardware - 真实AGV硬件接口模块
===============================

真实AGV机器人硬件适配，包含:
- CAN总线驱动 (ZLAC8015D)
- 镭神N10P激光雷达
- ETT10A-PW IMU
- 触觉/力觉传感器桥接
- 硬件监控
- AGV五级规格适配
"""

from .real_agv_interface import (
    AGVHardwareStatus,
    MotorState,
    WheelEncoder,
    IMUData,
    LidarScan,
    RealAGVInterface,
    RealAGVController,
    CANZAC8015DDriver,
    LidarN10P,
    IMUETT10APW,
    AGVTactileBridge,
    AGVForceBridge,
    HardwareMonitor,
    AGV_HARDWARE_SPECS,
)
from .agv_interface import (
    AGVType,
    AGVStatus,
    AGVInterfaceFactory,
    AGVInterface,
)

__all__ = [
    # 枚举
    'AGVHardwareStatus',
    'AGVType',
    'AGVStatus',
    # 数据结构
    'MotorState',
    'WheelEncoder',
    'IMUData',
    'LidarScan',
    # 接口类
    'RealAGVInterface',
    'RealAGVController',
    'CANZAC8015DDriver',
    'LidarN10P',
    'IMUETT10APW',
    'AGVTactileBridge',
    'AGVForceBridge',
    'HardwareMonitor',
    'AGVInterfaceFactory',
    'AGVInterface',
    # 规格表
    'AGV_HARDWARE_SPECS',
]

__version__ = "1.0.0"
