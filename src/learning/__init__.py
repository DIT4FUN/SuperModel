# Copyright (C) 2024-2026 赵元请 (DIT4FUN)
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
SuperModel 学习模块
==================

自监督学习与在线强化学习
"""

from .self_supervised import (
    LearningConfig,
    ContrastiveLoss,
    WorldModel as SimpleWorldModel,
    IntrinsicCuriosity,
    AutonomousLearning,
    get_learning_spec
)

from .world_model import (
    WorldModelConfig,
    WorldModel,
    ObservationEncoder,
    TransitionModel,
    RewardModel,
    ValueModel,
    ActorModel,
    ModelState,
    ModelOutput,
    WorldModelAgent,
    ReplayBuffer,
    create_world_model_agent,
    get_world_model_spec,
    WORLD_MODEL_GRADES
)

from .autonomous_learning import (
    AutonomousLearningConfig,
    Experience,
    PrioritizedReplayBuffer,
    SumTree,
    EWC,
    MetaLearner,
    CuriosityModule,
    SkillLibrary,
    AutonomousLearningAgent
)

__all__ = [
    # 自监督学习
    'LearningConfig',
    'ContrastiveLoss',
    'SimpleWorldModel',
    'IntrinsicCuriosity',
    'AutonomousLearning',
    'get_learning_spec',
    
    # World Model (Dreamer-style)
    'WorldModelConfig',
    'WorldModel',
    'ObservationEncoder',
    'TransitionModel',
    'RewardModel',
    'ValueModel',
    'ActorModel',
    'ModelState',
    'ModelOutput',
    'WorldModelAgent',
    'ReplayBuffer',
    'create_world_model_agent',
    'get_world_model_spec',
    'WORLD_MODEL_GRADES',
    
    # 自主学习框架 (RDK/持续学习)
    'AutonomousLearningConfig',
    'Experience',
    'PrioritizedReplayBuffer',
    'SumTree',
    'EWC',
    'MetaLearner',
    'CuriosityModule',
    'SkillLibrary',
    'AutonomousLearningAgent'
]
