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
SuperModel - 超模态机器人具身智能大脑
=====================================

多模态感知融合 + 自主学习 + 具身智能

主要模块:
- sensors: 多模态传感器接口
- perception: 感知融合
- fusion: 跨模态融合
- learning: 自主学习
- control: 动作控制
- memory: 长期记忆系统
"""

__version__ = "2.90.0"
__author__ = "DIT4FUN"

from .sensors import (
    BinocularCamera, DepthProcessor,
    BinauralMic, SoundLocalizer,
    TactileArray, PressureProcessor,
    ForceTorqueSensor, WrenchProcessor,
    IMUSensor, PoseEstimator
)

from .perception import (
    MultimodalInput,
    FusionConfig,
    CrossModalFusion,
    UnifiedRepresentation,
    create_multimodal_input
)

from .fusion import (
    CrossModalFusion,
    UnifiedRepresentation
)

from .learning import (
    LearningConfig,
    ContrastiveLoss,
    WorldModel,
    IntrinsicCuriosity,
    AutonomousLearning,
    get_learning_spec
)

from .memory import (
    LongTermMemory,
    MemoryConfig,
    EpisodicMemory,
    SemanticMemory,
    ProceduralMemory,
    WorkingMemory,
    MemoryStore,
    MemoryRetrieval,
    MemoryConsolidation,
)

from . import utils_pkg

__all__ = [
    # 版本
    '__version__',
    
    # 传感器
    'BinocularCamera', 'DepthProcessor',
    'BinauralMic', 'SoundLocalizer',
    'TactileArray', 'PressureProcessor',
    'ForceTorqueSensor', 'WrenchProcessor',
    'IMUSensor', 'PoseEstimator',
    
    # 感知
    'MultimodalInput',
    'FusionConfig',
    'CrossModalFusion',
    'UnifiedRepresentation',
    'create_multimodal_input',
    
    # 学习
    'LearningConfig',
    'ContrastiveLoss',
    'WorldModel',
    'IntrinsicCuriosity',
    'AutonomousLearning',
    'get_learning_spec',

    # 记忆系统
    'LongTermMemory',
    'MemoryConfig',
    'EpisodicMemory',
    'SemanticMemory',
    'ProceduralMemory',
    'WorkingMemory',
    'MemoryStore',
    'MemoryRetrieval',
    'MemoryConsolidation',

    # 工具
    'utils_pkg'
]
