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
]
