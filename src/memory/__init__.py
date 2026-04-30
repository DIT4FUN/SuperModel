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
Long-Term Memory System - 长期记忆系统
=====================================

SuperModel 的持久化记忆系统，支持:
- 情景记忆 (Episodic): 经验、事件、场景记忆
- 语义记忆 (Semantic): 知识、概念、事实
- 程序记忆 (Procedural): 技能、流程、习惯
- 工作记忆 (Working): 当前上下文短期记忆

架构:
  sensors → context_understanding → working_memory
                                        ↓
                              memory_consolidation
                                        ↓
  episodic ←────────────────────── memory_store
  semantic  ←───────────────────── ↗
  procedural ←──────────────────── ↗
"""

from .long_term_memory import LongTermMemory, MemoryConfig
from .episodic_memory import EpisodicMemory
from .semantic_memory import SemanticMemory
from .procedural_memory import ProceduralMemory
from .working_memory import WorkingMemory
from .memory_store import MemoryStore
from .memory_retrieval import MemoryRetrieval
from .memory_consolidation import MemoryConsolidation
from .embodied_long_term_memory import (
    EmbodiedExperienceType,
    EmbodiedMemoryTag,
    EmbodiedExperience,
    SceneMemoryIndex,
    SkillMemoryRecord,
    AGVGradeAwareMemory,
    ExperienceCompressor,
    MemoryBasedTaskPredictor,
    EmbodiedLongTermMemory,
    create_embodied_long_term_memory,
)

__all__ = [
    'LongTermMemory',
    'MemoryConfig',
    'EpisodicMemory',
    'SemanticMemory', 
    'ProceduralMemory',
    'WorkingMemory',
    'MemoryStore',
    'MemoryRetrieval',
    'MemoryConsolidation',
    # 具身长期记忆
    'EmbodiedExperienceType',
    'EmbodiedMemoryTag',
    'EmbodiedExperience',
    'SceneMemoryIndex',
    'SkillMemoryRecord',
    'AGVGradeAwareMemory',
    'ExperienceCompressor',
    'MemoryBasedTaskPredictor',
    'EmbodiedLongTermMemory',
    'create_embodied_long_term_memory',
]
