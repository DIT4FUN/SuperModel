"""
SuperModel 融合模块
==================

跨模态感知融合
"""

from .cross_modal_fusion import (
    MultimodalInput,
    FusionConfig,
    CrossModalFusion,
    UnifiedRepresentation,
    FusionStrategy,
    create_multimodal_input
)

from .sensor_fusion import (
    SensorFusion,
    ComplementaryFilter,
    ExtendedKalmanFilter,
    MultiSensorFusion
)

__all__ = [
    # cross-modal fusion
    'MultimodalInput',
    'FusionConfig',
    'CrossModalFusion',
    'UnifiedRepresentation',
    'FusionStrategy',
    'create_multimodal_input',
    # sensor fusion
    'SensorFusion',
    'ComplementaryFilter',
    'ExtendedKalmanFilter',
    'MultiSensorFusion'
]
