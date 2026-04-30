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
