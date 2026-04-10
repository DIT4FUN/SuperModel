"""
simulation - 仿真环境模块
============================

PyBullet/MuJoCo/Gymnasium仿真 + 增强具身仿真
"""

from .pybullet_sim import *
from .mujoco_sim import *
from .gym_env import *
from .agv_model_generator import generate_agv_urdf_detailed, GRADE_CONFIGS
from .embodied_sim import (
    EmbodiedSimEnv,
    EmbodiedSensorSimulator,
    WarehouseScene,
    TaskMetrics,
    MultiAGVEmbodiedSim,
    create_embodied_sim,
    ContactPoint,
)

__all__ = [
    # 新增增强具身仿真
    'EmbodiedSimEnv',
    'EmbodiedSensorSimulator',
    'WarehouseScene',
    'TaskMetrics',
    'MultiAGVEmbodiedSim',
    'create_embodied_sim',
    'ContactPoint',
    # 原有导出
    'generate_agv_urdf_detailed',
    'GRADE_CONFIGS',
]

__version__ = "1.1.0"
