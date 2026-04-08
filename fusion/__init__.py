"""融合模块"""
from .sensor_fusion import SensorFusion, ExtendedKalmanFilter
from .cross_modal_fusion import (
    CrossModalFusion, FusionConfig, MultimodalInput,
    get_fusion_spec, create_fusion_for_grade, simple_fusion,
    FUSION_GRADES, UnifiedRepresentation, FusionStrategy,
    create_multimodal_input, CrossModalAttention,
)

__all__ = [
    "SensorFusion", "ExtendedKalmanFilter",
    "CrossModalFusion", "FusionConfig", "MultimodalInput",
    "get_fusion_spec", "create_fusion_for_grade", "simple_fusion",
    "FUSION_GRADES", "UnifiedRepresentation", "FusionStrategy",
    "create_multimodal_input", "CrossModalAttention",
]
