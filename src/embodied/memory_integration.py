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
memory_integration.py - 具身记忆系统集成
SuperModel 超模态大模型具身智能系统

将长期记忆系统 (情景/语义/程序/工作记忆) 与具身智能系统深度集成:
- 情景记忆：记录任务执行经历，供后续决策参考
- 语义记忆：存储场景知识、物体属性、环境规则
- 程序记忆：存储技能/动作序列，支持技能复用
- 工作记忆：管理当前任务状态、注意力焦点

支持:
- 经验驱动的行为树参数自动调优
- 场景感知的情境化记忆检索
- 技能记忆的条件化激活
- 跨任务经验迁移学习
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ============================================================================
# 具身记忆条目
# ============================================================================

@dataclass
class EmbodiedMemoryEntry:
    """具身记忆条目"""
    entry_id: str
    entry_type: str  # "episode", "semantic", "skill", "context"
    timestamp: float
    content: Dict[str, Any]
    importance: float = 0.5  # 0.0 - 1.0
    accessibility: float = 1.0  # 可访问性 (Ebbinghaus遗忘曲线)
    retrieval_count: int = 0
    tags: Set[str] = field(default_factory=set)
    source_module: str = ""  # 来源模块: simulation, real_agv, behavior_tree, planner
    learned_from: Optional[str] = None  # 经验来源: success, failure, observation

    def touch(self):
        """被检索时调用，提升可访问性"""
        self.retrieval_count += 1
        self.accessibility = min(1.0, self.accessibility + 0.1)

    def decay(self, factor: float = 0.95):
        """时间衰减"""
        self.accessibility *= factor


@dataclass
class EmbodiedSkill:
    """具身技能（存储在程序记忆中）"""
    skill_id: str
    name: str
    description: str
    behavior_tree_config: Dict[str, Any]
    preconditions: List[Dict[str, Any]]
    success_rate: float = 0.0
    avg_duration: float = 0.0
    usage_count: int = 0
    last_used: Optional[float] = None
    scene_types: List[str] = field(default_factory=list)  # 适用场景类型
    agv_types: List[str] = field(default_factory=list)    # 适用AGV类型
    tags: Set[str] = field(default_factory=set)

    def activate(self) -> Dict[str, Any]:
        """激活技能，返回行为树配置"""
        self.usage_count += 1
        self.last_used = time.time()
        return self.behavior_tree_config

    def update_success(self, success: bool, duration: float):
        """更新技能统计"""
        self.avg_duration = (self.avg_duration * self.usage_count + duration) / (self.usage_count + 1)
        if success:
            self.success_rate = (self.success_rate * self.usage_count + 1.0) / (self.usage_count + 1)
        else:
            self.success_rate = (self.success_rate * self.usage_count) / (self.usage_count + 1)
        self.usage_count += 1


# ============================================================================
# 具身记忆管理器
# ============================================================================

class EmbodiedMemoryManager:
    """
    具身记忆管理器

    协调四种记忆类型的集成:
    - 情景记忆 (episodes): 任务执行经历
    - 语义记忆 (semantics): 场景/物体/规则知识
    - 程序记忆 (skills): 可复用的技能库
    - 工作记忆 (working): 当前状态和注意力
    """

    def __init__(
        self,
        episodic_memory: Optional[Any] = None,
        semantic_memory: Optional[Any] = None,
        procedural_memory: Optional[Any] = None,
        working_memory: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.procedural = procedural_memory
        self.working = working_memory
        self.config = config or {}
        self.enable_memory = (
            episodic_memory is not None
            or semantic_memory is not None
            or procedural_memory is not None
        )

        # 内存中的缓存（当外部记忆系统不可用时使用）
        self._episode_cache: List[EmbodiedMemoryEntry] = []
        self._skill_cache: Dict[str, EmbodiedSkill] = {}
        self._semantic_cache: List[EmbodiedMemoryEntry] = []
        self._working_state: Dict[str, Any] = {}

        # 遗忘曲线参数
        self._decay_factor = self.config.get("decay_factor", 0.95)
        self._decay_interval = self.config.get("decay_interval", 3600)  # 每小时衰减
        self._last_decay = time.time()

        logger.info("EmbodiedMemoryManager initialized")

    # -------------------------------------------------------------------------
    # 情景记忆操作
    # -------------------------------------------------------------------------

    def store_episode(
        self,
        episode_type: str,
        content: Dict[str, Any],
        importance: float = 0.5,
        tags: Optional[Set[str]] = None,
        source: str = "behavior_tree",
        outcome: Optional[str] = None,
    ) -> EmbodiedMemoryEntry:
        """
        存储情景记忆（任务执行经历）
        """
        entry = EmbodiedMemoryEntry(
            entry_id=str(uuid.uuid4())[:8],
            entry_type=episode_type,
            timestamp=time.time(),
            content=content,
            importance=importance,
            tags=tags or set(),
            source_module=source,
            learned_from=outcome,
        )

        # 存入外部记忆系统（如果可用）
        if self.episodic and hasattr(self.episodic, 'store'):
            self.episodic.store(entry.to_memory_format())
        elif self.episodic and hasattr(self.episodic, 'add'):
            self.episodic.add(entry.to_memory_format())

        # 同时存入本地缓存
        self._episode_cache.append(entry)
        self._apply_decay()

        logger.debug(f"Stored episode: {episode_type} (id={entry.entry_id}, outcome={outcome})")
        return entry

    def retrieve_episodes(
        self,
        query: str,
        max_results: int = 5,
        time_window: Optional[float] = None,
        outcome_filter: Optional[str] = None,
    ) -> List[EmbodiedMemoryEntry]:
        """
        检索情景记忆

        Args:
            query: 检索查询（任务类型/场景等）
            max_results: 最大返回数
            time_window: 只检索最近N秒内的记忆
            outcome_filter: 只检索特定结果的记忆 ("success", "failure")

        Returns:
            相关情景记忆列表
        """
        results = []

        # 外部记忆系统检索
        if self.episodic:
            try:
                if hasattr(self.episodic, 'retrieve'):
                    ext_results = self.episodic.retrieve(query=query, limit=max_results)
                    for r in ext_results:
                        entry = EmbodiedMemoryEntry(
                            entry_id=r.get("record_id", ""),
                            entry_type=r.get("task_type", ""),
                            timestamp=r.get("timestamp", 0),
                            content=r,
                            importance=r.get("importance", 0.5),
                            learned_from=r.get("result"),
                        )
                        results.append(entry)
                elif hasattr(self.episodic, 'search'):
                    ext_results = self.episodic.search(query=query, top_k=max_results)
                    for r in ext_results:
                        entry = EmbodiedMemoryEntry(
                            entry_id=r.get("record_id", ""),
                            entry_type=r.get("task_type", ""),
                            timestamp=r.get("timestamp", 0),
                            content=r,
                        )
                        results.append(entry)
            except Exception as e:
                logger.warning(f"External episodic memory retrieval failed: {e}")

        # 本地缓存检索
        now = time.time()
        for entry in reversed(self._episode_cache):
            # 时间窗口过滤
            if time_window and (now - entry.timestamp) > time_window:
                continue
            # 结果过滤
            if outcome_filter and entry.learned_from != outcome_filter:
                continue
            # 文本匹配
            entry_text = (entry.entry_type + " " + str(entry.content)).lower()
            if query.lower() in entry_text or not query:
                entry.touch()  # 提升可访问性
                results.append(entry)
                if len(results) >= max_results:
                    break

        return results[:max_results]

    def get_recent_episodes(self, count: int = 10) -> List[EmbodiedMemoryEntry]:
        """获取最近的N条情景记忆"""
        all_episodes = []
        if self.episodic:
            try:
                if hasattr(self.episodic, 'get_recent'):
                    ext = self.episodic.get_recent(count)
                    for r in ext:
                        entry = EmbodiedMemoryEntry(
                            entry_id=r.get("record_id", ""),
                            entry_type=r.get("task_type", ""),
                            timestamp=r.get("timestamp", 0),
                            content=r,
                        )
                        all_episodes.append(entry)
            except Exception:
                pass
        all_episodes.extend(self._episode_cache)
        all_episodes.sort(key=lambda e: e.timestamp, reverse=True)
        return all_episodes[:count]

    # -------------------------------------------------------------------------
    # 程序记忆（技能）操作
    # -------------------------------------------------------------------------

    def register_skill(
        self,
        name: str,
        behavior_tree_config: Dict[str, Any],
        description: str = "",
        preconditions: Optional[List[Dict[str, Any]]] = None,
        scene_types: Optional[List[str]] = None,
        tags: Optional[Set[str]] = None,
    ) -> EmbodiedSkill:
        """
        注册新技能到程序记忆
        """
        skill = EmbodiedSkill(
            skill_id=str(uuid.uuid4())[:8],
            name=name,
            description=description,
            behavior_tree_config=behavior_tree_config,
            preconditions=preconditions or [],
            scene_types=scene_types or [],
            tags=tags or set(),
        )

        self._skill_cache[skill.skill_id] = skill

        if self.procedural and hasattr(self.procedural, 'store'):
            self.procedural.store(skill.__dict__)

        logger.info(f"Registered skill: {name} (id={skill.skill_id})")
        return skill

    def retrieve_skills(
        self,
        query: Optional[str] = None,
        scene_type: Optional[str] = None,
        min_success_rate: float = 0.0,
        limit: int = 5,
    ) -> List[EmbodiedSkill]:
        """
        检索适合的技能

        Args:
            query: 技能名称/描述关键词
            scene_type: 目标场景类型
            min_success_rate: 最低成功率过滤
            limit: 返回数量上限

        Returns:
            适合的技能列表（按成功率排序）
        """
        candidates = []

        for skill in self._skill_cache.values():
            # 成功率过滤
            if skill.success_rate < min_success_rate:
                continue
            # 场景类型过滤
            if scene_type and scene_type not in skill.scene_types:
                continue
            # 查询过滤
            if query:
                query_lower = query.lower()
                if (query_lower not in skill.name.lower()
                        and query_lower not in skill.description.lower()
                        and not any(query_lower in t.lower() for t in skill.tags)):
                    continue

            candidates.append(skill)

        # 按成功率排序
        candidates.sort(key=lambda s: s.success_rate, reverse=True)
        return candidates[:limit]

    def update_skill_outcome(
        self,
        skill_id: str,
        success: bool,
        duration: float,
    ):
        """更新技能的执行结果统计"""
        if skill_id in self._skill_cache:
            skill = self._skill_cache[skill_id]
            skill.update_success(success, duration)
            logger.debug(f"Updated skill {skill.name}: success_rate={skill.success_rate:.2%}, "
                         f"usage={skill.usage_count}")

    # -------------------------------------------------------------------------
    # 语义记忆操作
    # -------------------------------------------------------------------------

    def store_semantic(
        self,
        concept_type: str,
        content: Dict[str, Any],
        importance: float = 0.5,
        tags: Optional[Set[str]] = None,
    ) -> EmbodiedMemoryEntry:
        """存储语义记忆（场景/物体/规则知识）"""
        entry = EmbodiedMemoryEntry(
            entry_id=str(uuid.uuid4())[:8],
            entry_type=concept_type,
            timestamp=time.time(),
            content=content,
            importance=importance,
            tags=tags or set(),
        )

        self._semantic_cache.append(entry)

        if self.semantic and hasattr(self.semantic, 'store'):
            self.semantic.store(entry.content)

        return entry

    def query_semantic(
        self,
        concept_type: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 5,
    ) -> List[EmbodiedMemoryEntry]:
        """检索语义记忆"""
        results = []

        for entry in reversed(self._semantic_cache):
            if concept_type and entry.entry_type != concept_type:
                continue
            if query:
                entry_text = str(entry.content).lower()
                if query.lower() not in entry_text:
                    continue
            results.append(entry)
            if len(results) >= limit:
                break

        return results

    # -------------------------------------------------------------------------
    # 工作记忆操作
    # -------------------------------------------------------------------------

    def set_working(self, key: str, value: Any):
        """设置工作记忆条目"""
        self._working_state[key] = value
        if self.working and hasattr(self.working, 'set'):
            self.working.set(key, value)

    def get_working(self, key: str, default: Any = None) -> Any:
        """获取工作记忆条目"""
        return self._working_state.get(key, default)

    def clear_working(self, key: Optional[str] = None):
        """清除工作记忆"""
        if key:
            self._working_state.pop(key, None)
        else:
            self._working_state.clear()

    def get_attention_focus(self) -> Dict[str, Any]:
        """获取当前注意力焦点（工作记忆的核心状态）"""
        return {
            "current_task": self._working_state.get("current_task"),
            "target_position": self._working_state.get("target_position"),
            "carried_object": self._working_state.get("carried_object"),
            "battery_level": self._working_state.get("battery_level"),
            "safety_status": self._working_state.get("safety_status"),
            "scene_type": self._working_state.get("scene_type"),
        }

    def set_attention_focus(self, focus: Dict[str, Any]):
        """设置当前注意力焦点"""
        for key, value in focus.items():
            self.set_working(key, value)

    # -------------------------------------------------------------------------
    # 内部维护
    # -------------------------------------------------------------------------

    def _apply_decay(self):
        """应用遗忘曲线衰减"""
        now = time.time()
        if now - self._last_decay < self._decay_interval:
            return

        for entry in self._episode_cache:
            entry.decay(self._decay_factor)

        # 删除完全遗忘的条目
        before = len(self._episode_cache)
        self._episode_cache = [
            e for e in self._episode_cache
            if e.accessibility > 0.01 and (now - e.timestamp) < 30 * 86400  # 30天TTL
        ]
        after = len(self._episode_cache)
        if before != after:
            logger.debug(f"Memory decay: removed {before - after} forgotten entries")

        self._last_decay = now

    def get_memory_summary(self) -> Dict[str, Any]:
        """获取记忆系统状态摘要"""
        return {
            "episodes_cached": len(self._episode_cache),
            "skills_registered": len(self._skill_cache),
            "semantic_entries": len(self._semantic_cache),
            "working_keys": len(self._working_state),
            "avg_skill_success_rate": (
                sum(s.success_rate for s in self._skill_cache.values()) / max(1, len(self._skill_cache))
            ),
            "total_skill_usage": sum(s.usage_count for s in self._skill_cache.values()),
            "recent_episodes": len(self.get_recent_episodes(5)),
        }


# ============================================================================
# 工厂函数
# ============================================================================

def create_embodied_memory_manager(
    episodic_memory: Optional[Any] = None,
    semantic_memory: Optional[Any] = None,
    procedural_memory: Optional[Any] = None,
    working_memory: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
) -> EmbodiedMemoryManager:
    """
    创建具身记忆管理器（自动连接长期记忆系统）

    如果外部记忆模块传入None，会创建轻量级内存版本
    """
    return EmbodiedMemoryManager(
        episodic_memory=episodic_memory,
        semantic_memory=semantic_memory,
        procedural_memory=procedural_memory,
        working_memory=working_memory,
        config=config or {},
    )


def connect_to_long_term_memory(
    memory_dir: str = "memory_data",
) -> Dict[str, Any]:
    """
    连接到长期记忆系统的各模块

    尝试从 memory_data 目录加载已初始化的记忆模块
    """
    modules = {}

    try:
        import sys
        import importlib.util

        module_map = {
            "episodic": "episodic_memory",
            "semantic": "semantic_memory",
            "procedural": "procedural_memory",
            "working": "working_memory",
        }

        for key, module_name in module_map.items():
            try:
                spec = importlib.util.find_module(module_name)
                if spec:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    modules[key] = mod
                    logger.info(f"Connected to {module_name}")
            except Exception as e:
                logger.debug(f"Could not load {module_name}: {e}")

    except Exception as e:
        logger.warning(f"Failed to connect to long-term memory: {e}")

    return modules


__all__ = [
    "EmbodiedMemoryEntry",
    "EmbodiedSkill",
    "EmbodiedMemoryManager",
    "create_embodied_memory_manager",
    "connect_to_long_term_memory",
]
