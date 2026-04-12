"""
Embodiment Module - 具身智能模块
=================================
包含具身智能仿真、真实AGV硬件接口、行为树执行引擎、多AGV蜂群协同等功能

Submodules:
- simulation: 增强型具身仿真环境 (PyBullet/MuJoCo)
- agv_interface: 真实AGV硬件抽象层接口
- behavior_tree_engine: 行为树运行时执行引擎
- multi_agv_coordinator: 多AGV蜂群协同调度器
"""

from .simulation import EmbodimentSimulator
from .agv_interface import AGVHardwareInterface, AGVCommand, AGVState
from .behavior_tree_engine import BehaviorTreeEngine
from .multi_agv_coordinator import MultiAGVCoordinator, AGVAssignment

__all__ = [
    "EmbodimentSimulator",
    "AGVHardwareInterface",
    "AGVCommand",
    "AGVState",
    "BehaviorTreeEngine",
    "MultiAGVCoordinator",
    "AGVAssignment"
]

__version__ = "1.0.0"
