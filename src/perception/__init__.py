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
SuperModel 感知模块
===================

多模态感知融合与特征提取:
- 跨模态融合网络 (CrossModalFusion)
- 场景理解 (SceneUnderstanding)
"""

try:
    from ..fusion.cross_modal_fusion import (
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
except ImportError:
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
