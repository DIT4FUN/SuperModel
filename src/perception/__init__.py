"""
SuperModel 感知模块
===================

多模态感知融合与特征提取:
- 跨模态融合网络 (CrossModalFusion)
- 场景理解 (SceneUnderstanding)
"""

from fusion.cross_modal_fusion import (
    MultimodalInput,
    FusionConfig,
    FusionStrategy,
    CrossModalFusion,
    CrossModalAttention,
    ModalityEncoder,
    UnifiedRepresentation,
    LanguageEncoder,
    create_multimodal_input
)
from .scene_understanding import (
    SceneUnderstanding,
    SceneObject,
    SceneGraph,
    SceneState,
    SpatialRelation,
    OccupancyGrid,
    DynamicState,
    ObjectClass,
    get_scene_spec,
    AGV_SCENE_UNDERSTANDING_GRADES
)

__all__ = [
    # 融合网络
    'MultimodalInput',
    'FusionConfig',
    'FusionStrategy',
    'CrossModalFusion',
    'CrossModalAttention',
    'ModalityEncoder',
    'UnifiedRepresentation',
    'LanguageEncoder',
    'create_multimodal_input',
    # 场景理解
    'SceneUnderstanding',
    'SceneObject',
    'SceneGraph',
    'SceneState',
    'SpatialRelation',
    'OccupancyGrid',
    'DynamicState',
    'ObjectClass',
    'get_scene_spec',
    'AGV_SCENE_UNDERSTANDING_GRADES',
]
