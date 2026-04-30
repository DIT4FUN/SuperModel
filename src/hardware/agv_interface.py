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
AGV Interface - AGV接口统一层
=============================
统一仿真和真实AGV的接口，对外提供一致的操作API。
"""

from __future__ import annotations
import enum
from typing import Optional, Dict, Any, Type
from dataclasses import dataclass

from .real_agv_interface import RealAGVInterface, AGVHardwareStatus
from simulation.agv_simulator import AGVSimulator


class AGVType(enum.Enum):
    """AGV类型"""
    SIMULATION = "simulation"
    REAL_DIFFERENTIAL = "real_differential"
    REAL_MECANUM = "real_mecanum"
    REAL_OMNI = "real_omni"
    VIRTUAL = "virtual"


class AGVStatus(enum.Enum):
    """AGV状态"""
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    IDLE = "idle"
    MOVING = "moving"
    RUNNING_TASK = "running_task"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"
    CHARGING = "charging"
    
    @classmethod
    def from_hardware_status(cls, hw_status: AGVHardwareStatus) -> 'AGVStatus':
        """从硬件状态转换"""
        mapping = {
            AGVHardwareStatus.DISCONNECTED: cls.DISCONNECTED,
            AGVHardwareStatus.CONNECTING: cls.DISCONNECTED,
            AGVHardwareStatus.CONNECTED: cls.CONNECTED,
            AGVHardwareStatus.RUNNING: cls.MOVING,
            AGVHardwareStatus.ERROR: cls.ERROR,
            AGVHardwareStatus.EMERGENCY_STOP: cls.EMERGENCY_STOP,
        }
        return mapping.get(hw_status, cls.ERROR)


@dataclass
class AGVInterfaceConfig:
    """AGV接口配置"""
    agv_type: AGVType = AGVType.SIMULATION
    agv_grade: str = "M"  # S/M/L/XL/XXL
    port: Optional[str] = None
    baud_rate: int = 115200
    sim_config: Optional[Dict[str, Any]] = None
    real_config: Optional[Dict[str, Any]] = None


class AGVInterfaceFactory:
    """AGV接口工厂类"""
    
    @staticmethod
    def create(config: AGVInterfaceConfig) -> Any:
        """创建AGV接口实例"""
        if config.agv_type == AGVType.SIMULATION:
            from simulation.agv_simulator import AGVSimulatorConfig
            sim_config = config.sim_config or {}
            return AGVSimulator(
                grade=config.agv_grade,
                **sim_config
            )
        elif config.agv_type in [AGVType.REAL_DIFFERENTIAL, AGVType.REAL_MECANUM, AGVType.REAL_OMNI]:
            real_config = config.real_config or {}
            return RealAGVInterface(
                agv_type=config.agv_type.value,
                grade=config.agv_grade,
                port=config.port,
                baud_rate=config.baud_rate,
                **real_config
            )
        elif config.agv_type == AGVType.VIRTUAL:
            from simulation.agv_simulator import VirtualAGV
            return VirtualAGV(grade=config.agv_grade)
        else:
            raise ValueError(f"Unsupported AGV type: {config.agv_type}")


__all__ = [
    'AGVType',
    'AGVStatus',
    'AGVInterfaceConfig',
    'AGVInterfaceFactory',
]
