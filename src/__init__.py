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

__version__ = "2.33.0"
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
