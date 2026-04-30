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
