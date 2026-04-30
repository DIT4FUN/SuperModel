# Copyright (C) 2026 焦洋 (Jiao Yang) <jiaoyang@cczu.edu.cn>
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
