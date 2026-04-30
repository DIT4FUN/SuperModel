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
硬件抽象层 - 基础接口
======================

定义所有机器人主板的抽象接口和通用类型。
"""

import os
import platform
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any


class BoardType(Enum):
    """主板类型枚举"""
    # RK3588 系列
    RK3588 = "rk3588"
    RK3588S = "rk3588s"
    
    # 地瓜机器人 RDK 系列
    RDK_X3 = "rdk_x3"
    RDK_X5_ULTRA = "rdk_x5_ultra"
    RDK_S100 = "rdk_s100"
    
    # 通用 x86_64 (开发/仿真)
    X86_64_GENERIC = "x86_64_generic"
    
    # 未知
    UNKNOWN = "unknown"


class PeripheralType(Enum):
    """外设类型"""
    GPIO = "gpio"
    I2C = "i2c"
    SPI = "spi"
    PWM = "pwm"
    UART = "uart"
    CAN = "can"
    ADC = "adc"


class ComputeCapability(Enum):
    """算力等级 (TOPS)"""
    S = 3       # ~3 TOPS (S100)
    M = 6       # ~6 TOPS (X3)
    L = 12      # ~12 TOPS (X5 Ultra)
    XL = 30     # ~30 TOPS (RK3588)
    UNKNOWN = 0


@dataclass
class BoardInfo:
    """主板信息"""
    board_type: BoardType
    name: str
    chip: str
    cpu_cores: int
    cpu_freq_mhz: int
    memory_mb: int
    npu_tops: float
    gpu_tops: Optional[float] = None
    arch: str = "aarch64"
    os: str = "linux"
    peripherals: List[PeripheralType] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_tops(self) -> float:
        """总算力 TOPS"""
        return self.npu_tops + (self.gpu_tops or 0.0)


class BoardBase(ABC):
    """
    主板抽象基类
    
    所有支持的主板必须继承此类并实现其接口。
    """
    
    # 类属性: 主板类型
    BOARD_TYPE: BoardType = BoardType.UNKNOWN
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化主板抽象
        
        Args:
            config: 主板配置参数
        """
        self.config = config or {}
        self._info: Optional[BoardInfo] = None
        self._initialized = False
        
    @property
    def info(self) -> BoardInfo:
        """获取主板信息"""
        if self._info is None:
            self._info = self._detect_info()
        return self._info
    
    @property
    def is_initialized(self) -> bool:
        """是否已初始化"""
        return self._initialized
    
    @abstractmethod
    def _detect_info(self) -> BoardInfo:
        """检测主板信息 (子类实现)"""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        初始化主板
        
        Returns:
            是否初始化成功
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """关闭主板，释放资源"""
        pass
    
    def get_npu_context(self) -> 'NPUContext':
        """获取 NPU 加速上下文"""
        from .nnpu import get_npu_context
        return get_npu_context(self)
    
    def get_gpio_controller(self) -> 'GPIOController':
        """获取 GPIO 控制器"""
        from .gpio import GPIOController
        return GPIOController(self)
    
    # ==================== 通用接口 ====================
    
    def get_cpu_usage(self) -> float:
        """获取 CPU 使用率 (0-1)"""
        try:
            with open('/proc/stat', 'r') as f:
                line = f.readline()
                fields = line.split()
                idle = int(fields[4])
                total = sum(int(x) for x in fields[1:8])
            return 1.0 - (idle / total) if total > 0 else 0.0
        except:
            return 0.0
    
    def get_memory_usage(self) -> Tuple[float, float]:
        """
        获取内存使用情况
        
        Returns:
            (used_mb, total_mb)
        """
        try:
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
            mem_info = {}
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0][:-1]  # 去掉冒号
                    value = int(parts[1])
                    mem_info[key] = value
            
            total = mem_info.get('MemTotal', 0) / 1024
            available = mem_info.get('MemAvailable', 0) / 1024
            used = total - available
            return used, total
        except:
            return 0.0, 0.0
    
    def get_temperature(self) -> Optional[float]:
        """获取 SoC 温度 (摄氏度)"""
        paths = [
            '/sys/class/thermal/thermal_zone0/temp',
            '/sys/devices/virtual/thermal/thermal_zone0/temp',
        ]
        for path in paths:
            try:
                with open(path, 'r') as f:
                    temp_milli = int(f.read().strip())
                    return temp_milli / 1000.0
            except:
                continue
        return None
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} type={self.BOARD_TYPE.value}>"


# ==================== 工厂函数 ====================

def detect_board() -> BoardBase:
    """
    自动检测并返回当前平台的主板实例
    
    Returns:
        BoardBase: 检测到的主板实例
    """
    # 尝试检测设备树
    machine = platform.machine()
    
    # 检测地瓜机器人 RDK 系列
    if os.path.exists('/proc/device-tree/model'):
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip().lower()
            if 'x5' in model or 'ultra' in model:
                from .digu_robot import RDKX5Ultra
                return RDKX5Ultra()
            elif 'x3' in model:
                from .digu_robot import RDKX3
                return RDKX3()
            elif 's100' in model or 's-100' in model:
                from .digu_robot import RDKS100
                return RDKS100()
    
    # 检测 RK3588
    if os.path.exists('/proc/device-tree/compatible'):
        with open('/proc/device-tree/compatible', 'r') as f:
            compatible = f.read().lower()
            if 'rk3588' in compatible:
                from .rk3588 import RK3588Platform
                return RK3588Platform()
    
    # x86_64 开发环境
    if machine == 'x86_64':
        from .digu_robot import DiguRobotPlatform
        return DiguRobotPlatform(board_type=BoardType.X86_64_GENERIC, name="x86_64 Generic")
    
    # 未知
    from .digu_robot import DiguRobotPlatform
    return DiguRobotPlatform()


def create_board(board_type: BoardType, **kwargs) -> BoardBase:
    """
    根据类型创建主板实例
    
    Args:
        board_type: 主板类型
        **kwargs: 额外参数
        
    Returns:
        BoardBase: 主板实例
    """
    if board_type == BoardType.RDK_X3:
        from .digu_robot import RDKX3
        return RDKX3(**kwargs)
    elif board_type == BoardType.RDK_X5_ULTRA:
        from .digu_robot import RDKX5Ultra
        return RDKX5Ultra(**kwargs)
    elif board_type == BoardType.RDK_S100:
        from .digu_robot import RDKS100
        return RDKS100(**kwargs)
    elif board_type in (BoardType.RK3588, BoardType.RK3588S):
        from .rk3588 import RK3588Platform
        return RK3588Platform(**kwargs)
    elif board_type == BoardType.X86_64_GENERIC:
        from .digu_robot import DiguRobotPlatform
        return DiguRobotPlatform(board_type=board_type, **kwargs)
    else:
        raise ValueError(f"Unsupported board type: {board_type}")


# 延迟导入类型别名
from .nnpu import NPUContext
from .gpio import GPIOController
