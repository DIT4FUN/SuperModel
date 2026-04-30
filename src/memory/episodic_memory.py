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
Episodic Memory - 情景记忆模块
=============================

存储和管理Agent的经历、事件和经验。

情景记忆特点:
- 时间标记: 每个记忆都有时间戳
- 事件关联: 与特定场景/任务关联
- 情感标记: 记忆的情感强度
- 可检索性: 基于时间和上下文的检索
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import json
import uuid
import time


class EmotionalTag(Enum):
    """情感标签"""
    VERY_POSITIVE = "very_positive"   # 非常积极
    POSITIVE = "positive"             # 积极
    NEUTRAL = "neutral"              # 中性
    NEGATIVE = "negative"            # 消极
    VERY_NEGATIVE = "very_negative"  # 非常消极


class ImportanceLevel(Enum):
    """重要性等级"""
    CRITICAL = 5   # 关键 - 必须记住
    HIGH = 4       # 高 - 重要经验
    MEDIUM = 3     # 中 - 一般经历
    LOW = 2        # 低 - 琐事
    MINIMAL = 1    # 最小 - 可遗忘


@dataclass
class Episode:
    """
    单个情景记忆单元
    
    Attributes:
        id: 唯一标识符
        timestamp: 时间戳
        duration_s: 持续时间(秒)
        summary: 摘要描述
        context: 场景上下文
        actions: 执行的动作列表
        outcomes: 结果/奖励
        emotional_tag: 情感标记
        importance: 重要性等级
        importance_score: 重要性分数 [0, 10]
        entities: 涉及的实体
        locations: 涉及的位置
        tags: 自定义标签
        lessons_learned: 经验教训
        accessibility: 可访问性分数 [0, 1] (用于遗忘模型)
        last_accessed: 上次访问时间
        access_count: 访问次数
        consolidated: 是否已整合到语义记忆
    """
    id: str
    timestamp: float
    summary: str
    context: Dict[str, Any] = field(default_factory=dict)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    outcomes: Dict[str, Any] = field(default_factory=dict)
    emotional_tag: EmotionalTag = EmotionalTag.NEUTRAL
    importance: ImportanceLevel = ImportanceLevel.MEDIUM
    importance_score: float = 5.0
    duration_s: float = 0.0
    entities: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    accessibility: float = 1.0
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    consolidated: bool = False
    parent_episode_id: Optional[str] = None  # 关联的原始情景
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'summary': self.summary,
            'context': self.context,
            'actions': self.actions,
            'outcomes': self.outcomes,
            'emotional_tag': self.emotional_tag.value,
            'importance': self.importance.value,
            'importance_score': self.importance_score,
            'duration_s': self.duration_s,
            'entities': self.entities,
            'locations': self.locations,
            'tags': self.tags,
            'lessons_learned': self.lessons_learned,
            'accessibility': self.accessibility,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count,
            'consolidated': self.consolidated,
            'parent_episode_id': self.parent_episode_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Episode:
        """从字典反序列化"""
        return cls(
            id=data['id'],
            timestamp=data['timestamp'],
            summary=data['summary'],
            context=data.get('context', {}),
            actions=data.get('actions', []),
            outcomes=data.get('outcomes', {}),
            emotional_tag=EmotionalTag(data.get('emotional_tag', 'neutral')),
            importance=ImportanceLevel(data.get('importance', 3)),
            importance_score=data.get('importance_score', 5.0),
            duration_s=data.get('duration_s', 0.0),
            entities=data.get('entities', []),
            locations=data.get('locations', []),
            tags=data.get('tags', []),
            lessons_learned=data.get('lessons_learned', []),
            accessibility=data.get('accessibility', 1.0),
            last_accessed=data.get('last_accessed', time.time()),
            access_count=data.get('access_count', 0),
            consolidated=data.get('consolidated', False),
            parent_episode_id=data.get('parent_episode_id'),
        )


class EpisodicMemory:
    """
    情景记忆管理器
    
    负责:
    - 存储和索引情景记忆
    - 基于时间的检索
    - 基于上下文的检索
    - 记忆衰减和遗忘
    - 记忆整合准备
    """
    
    def __init__(
        self,
        store_path: Optional[str] = None,
        max_episodes: int = 10000,
        decay_rate: float = 0.01,
    ):
        """
        Args:
            store_path: 存储路径
            max_episodes: 最大情景记忆数
            decay_rate: 访问衰减率
        """
        self.store_path = store_path
        self.max_episodes = max_episodes
        self.decay_rate = decay_rate
        
        # 记忆存储
        self._episodes: Dict[str, Episode] = {}
        self._episodes_by_time: List[str] = []  # 按时间排序的ID列表
        self._episodes_by_context: Dict[str, List[str]] = {}  # context_hash -> episode_ids
        
        # 统计
        self._total_episodes = 0
        self._last_consolidation = time.time()
        
        # 加载已有记忆
        if store_path:
            self._load()
    
    def store(
        self,
        summary: str,
        context: Optional[Dict[str, Any]] = None,
        actions: Optional[List[Dict[str, Any]]] = None,
        outcomes: Optional[Dict[str, Any]] = None,
        emotional_tag: EmotionalTag = EmotionalTag.NEUTRAL,
        importance_score: float = 5.0,
        duration_s: float = 0.0,
        entities: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        lessons_learned: Optional[List[str]] = None,
    ) -> Episode:
        """
        存储新的情景记忆
        
        Args:
            summary: 记忆摘要
            context: 场景上下文
            actions: 执行的动作
            outcomes: 结果
            emotional_tag: 情感标签
            importance_score: 重要性分数 [0, 10]
            duration_s: 持续时间
            entities: 涉及的实体
            locations: 涉及的位置
            tags: 标签
            lessons_learned: 经验教训
            
        Returns:
            创建的Episode对象
        """
        episode_id = str(uuid.uuid4())
        
        episode = Episode(
            id=episode_id,
            timestamp=time.time(),
            summary=summary,
            context=context or {},
            actions=actions or [],
            outcomes=outcomes or {},
            emotional_tag=emotional_tag,
            importance=self._score_to_level(importance_score),
            importance_score=importance_score,
            duration_s=duration_s,
            entities=entities or [],
            locations=locations or [],
            tags=tags or [],
            lessons_learned=lessons_learned or [],
        )
        
        self._add_episode(episode)
        
        return episode
    
    def _add_episode(self, episode: Episode) -> None:
        """添加记忆到存储"""
        self._episodes[episode.id] = episode
        self._episodes_by_time.append(episode.id)
        
        # 按上下文索引
        context_hash = self._context_hash(episode.context)
        if context_hash not in self._episodes_by_context:
            self._episodes_by_context[context_hash] = []
        self._episodes_by_context[context_hash].append(episode.id)
        
        self._total_episodes += 1
        
        # 容量管理 - 删除最不重要的记忆
        if len(self._episodes) > self.max_episodes:
            self._prune_low_value_episodes()
        
        # 持久化
        if self.store_path:
            self._save()
    
    def _context_hash(self, context: Dict[str, Any]) -> str:
        """计算上下文的哈希值"""
        # 简化: 使用上下文的部分关键字段
        key_parts = []
        for k in sorted(context.keys()):
            v = context[k]
            if v is not None:
                key_parts.append(f"{k}:{str(v)[:50]}")
        return hash("|".join(key_parts))
    
    def _score_to_level(self, score: float) -> ImportanceLevel:
        """分数转等级"""
        if score >= 9: return ImportanceLevel.CRITICAL
        elif score >= 7: return ImportanceLevel.HIGH
        elif score >= 5: return ImportanceLevel.MEDIUM
        elif score >= 3: return ImportanceLevel.LOW
        else: return ImportanceLevel.MINIMAL
    
    def _prune_low_value_episodes(self, num_to_remove: int = 10) -> None:
        """删除低价值记忆"""
        # 计算每个记忆的价值分数
        def value_score(ep: Episode) -> float:
            age = time.time() - ep.timestamp
            # 价值 = 重要性 * 衰减后的可访问性 * (1 / (1 + age_days))
            decay = np.exp(-self.decay_rate * age / 86400)  # 按天衰减
            return ep.importance_score * ep.accessibility * decay
        
        # 按价值排序
        sorted_episodes = sorted(
            self._episodes.values(),
            key=value_score
        )[:num_to_remove]
        
        for ep in sorted_episodes:
            self._remove_episode(ep.id)
    
    def _remove_episode(self, episode_id: str) -> None:
        """删除记忆"""
        if episode_id in self._episodes:
            ep = self._episodes[episode_id]
            
            # 从时间列表移除
            if episode_id in self._episodes_by_time:
                self._episodes_by_time.remove(episode_id)
            
            # 从上下文索引移除
            context_hash = self._context_hash(ep.context)
            if context_hash in self._episodes_by_context:
                if episode_id in self._episodes_by_context[context_hash]:
                    self._episodes_by_context[context_hash].remove(episode_id)
            
            del self._episodes[episode_id]
    
    def retrieve_by_time(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
    ) -> List[Episode]:
        """
        按时间检索记忆
        
        Args:
            start_time: 开始时间戳
            end_time: 结束时间戳
            limit: 返回数量限制
            
        Returns:
            匹配的记忆列表
        """
        results = []
        
        for ep_id in reversed(self._episodes_by_time):
            ep = self._episodes.get(ep_id)
            if ep is None:
                continue
                
            if start_time is not None and ep.timestamp < start_time:
                continue
            if end_time is not None and ep.timestamp > end_time:
                continue
                
            results.append(ep)
            
            if len(results) >= limit:
                break
        
        return results
    
    def retrieve_by_context(
        self,
        context: Dict[str, Any],
        similarity_threshold: float = 0.7,
        limit: int = 20,
    ) -> List[Tuple[Episode, float]]:
        """
        按上下文检索相似记忆
        
        Args:
            context: 查询上下文
            similarity_threshold: 相似度阈值
            limit: 返回数量限制
            
        Returns:
            (记忆, 相似度分数) 列表
        """
        target_hash = self._context_hash(context)
        similar_episodes = []
        
        # 精确匹配
        if target_hash in self._episodes_by_context:
            for ep_id in self._episodes_by_context[target_hash]:
                ep = self._episodes.get(ep_id)
                if ep:
                    similar_episodes.append((ep, 1.0))
        
        # 模糊匹配 - 基于上下文重叠
        for ep_id, ep in self._episodes.items():
            if ep_id in [e.id for e, _ in similar_episodes]:
                continue
                
            similarity = self._compute_context_similarity(context, ep.context)
            if similarity >= similarity_threshold:
                similar_episodes.append((ep, similarity))
        
        # 排序
        similar_episodes.sort(key=lambda x: (x[1], x[0].timestamp), reverse=True)
        
        return similar_episodes[:limit]
    
    def _compute_context_similarity(
        self,
        ctx1: Dict[str, Any],
        ctx2: Dict[str, Any],
    ) -> float:
        """计算上下文相似度"""
        if not ctx1 or not ctx2:
            return 0.0
            
        # 基于键的重叠
        keys1 = set(ctx1.keys())
        keys2 = set(ctx2.keys())
        
        if not keys1 or not keys2:
            return 0.0
        
        intersection = keys1 & keys2
        if not intersection:
            return 0.0
        
        # 计算重叠键的值相似度
        value_similarities = []
        for key in intersection:
            v1, v2 = ctx1[key], ctx2[key]
            if v1 == v2:
                value_similarities.append(1.0)
            elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                max_val = max(abs(v1), abs(v2), 1e-6)
                value_similarities.append(1.0 - abs(v1 - v2) / max_val)
            else:
                value_similarities.append(0.0)
        
        # Jaccard + 值相似度
        jaccard = len(intersection) / len(keys1 | keys2)
        avg_value_sim = np.mean(value_similarities) if value_similarities else 0.0
        
        return 0.5 * jaccard + 0.5 * avg_value_sim
    
    def retrieve_by_entities(
        self,
        entities: List[str],
        limit: int = 50,
    ) -> List[Episode]:
        """按实体检索记忆"""
        results = []
        entity_set = set(entities)
        
        for ep in self._episodes.values():
            if entity_set & set(ep.entities):
                results.append(ep)
        
        # 按时间和重要性排序
        results.sort(
            key=lambda e: (e.timestamp, e.importance_score),
            reverse=True
        )
        
        return results[:limit]
    
    def retrieve_by_tags(
        self,
        tags: List[str],
        match_all: bool = False,
        limit: int = 50,
    ) -> List[Episode]:
        """
        按标签检索记忆
        
        Args:
            tags: 标签列表
            match_all: True表示全部匹配，False表示任一匹配
            limit: 返回数量限制
        """
        results = []
        tag_set = set(tags)
        
        for ep in self._episodes.values():
            ep_tag_set = set(ep.tags)
            
            if match_all:
                if tag_set <= ep_tag_set:  # 全部包含
                    results.append(ep)
            else:
                if tag_set & ep_tag_set:  # 任一匹配
                    results.append(ep)
        
        results.sort(key=lambda e: (e.timestamp, e.importance_score), reverse=True)
        return results[:limit]
    
    def retrieve_recent(self, limit: int = 20) -> List[Episode]:
        """检索最近的记忆"""
        return [
            self._episodes[ep_id]
            for ep_id in self._episodes_by_time[-limit:]
            if ep_id in self._episodes
        ]
    
    def access_episode(self, episode_id: str) -> Optional[Episode]:
        """
        访问记忆 (更新访问统计)
        
        Returns:
            记忆对象或None
        """
        ep = self._episodes.get(episode_id)
        if ep:
            ep.access_count += 1
            ep.last_accessed = time.time()
            # 可访问性增强
            ep.accessibility = min(1.0, ep.accessibility + 0.1)
        return ep
    
    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """获取指定记忆"""
        return self._episodes.get(episode_id)
    
    def delete_episode(self, episode_id: str) -> bool:
        """删除指定记忆"""
        if episode_id in self._episodes:
            self._remove_episode(episode_id)
            return True
        return False
    
    def mark_consolidated(self, episode_id: str) -> bool:
        """标记记忆已整合"""
        ep = self._episodes.get(episode_id)
        if ep:
            ep.consolidated = True
            if self.store_path:
                self._save()
            return True
        return False
    
    def get_unconsolidated_episodes(
        self,
        min_importance: float = 7.0,
        max_age_days: float = 7.0,
    ) -> List[Episode]:
        """获取待整合的记忆"""
        cutoff_time = time.time() - max_age_days * 86400
        
        return [
            ep for ep in self._episodes.values()
            if not ep.consolidated
            and ep.importance_score >= min_importance
            and ep.timestamp < cutoff_time
        ]
    
    def apply_forgetting(self, forgetting_rate: float = 0.01) -> int:
        """
        应用遗忘模型
        
        Args:
            forgetting_rate: 遗忘率
            
        Returns:
            处理的记忆数量
        """
        now = time.time()
        pruned = 0
        
        for ep in list(self._episodes.values()):
            age = now - ep.timestamp
            
            # 时间衰减
            decay = np.exp(-forgetting_rate * age / 86400)
            
            # 访问衰减
            access_decay = np.exp(-0.1 * ep.access_count)
            
            # 综合可访问性
            new_accessibility = ep.importance_score / 10.0 * decay * access_decay
            
            # 可访问性过低则删除
            if new_accessibility < 0.05:
                self._remove_episode(ep.id)
                pruned += 1
        
        return pruned
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        episodes = list(self._episodes.values())
        
        if not episodes:
            return {
                'total_episodes': 0,
                'avg_importance': 0.0,
                'avg_access_count': 0.0,
                'by_importance': {},
                'by_emotional_tag': {},
            }
        
        importance_dist = {}
        emotional_dist = {}
        
        for ep in episodes:
            # 重要性分布
            imp_key = ep.importance.name
            importance_dist[imp_key] = importance_dist.get(imp_key, 0) + 1
            
            # 情感分布
            emo_key = ep.emotional_tag.value
            emotional_dist[emo_key] = emotional_dist.get(emo_key, 0) + 1
        
        return {
            'total_episodes': len(episodes),
            'avg_importance': np.mean([e.importance_score for e in episodes]),
            'avg_access_count': np.mean([e.access_count for e in episodes]),
            'consolidated_count': sum(1 for e in episodes if e.consolidated),
            'by_importance': importance_dist,
            'by_emotional_tag': emotional_dist,
            'oldest_episode_age_days': (time.time() - min(e.timestamp for e in episodes)) / 86400,
            'newest_episode_age_hours': (time.time() - max(e.timestamp for e in episodes)) / 3600,
        }
    
    def _save(self) -> None:
        """保存到磁盘"""
        try:
            data = {
                'episodes': [ep.to_dict() for ep in self._episodes.values()],
                'episodes_by_time': self._episodes_by_time,
                'total_episodes': self._total_episodes,
            }
            
            path = f"{self.store_path}/episodic_memory.json"
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save episodic memory: {e}")
    
    def _load(self) -> None:
        """从磁盘加载"""
        try:
            path = f"{self.store_path}/episodic_memory.json"
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._episodes = {
                ep['id']: Episode.from_dict(ep)
                for ep in data.get('episodes', [])
            }
            self._episodes_by_time = data.get('episodes_by_time', [])
            self._total_episodes = data.get('total_episodes', len(self._episodes))
            
            # 重建上下文索引
            self._episodes_by_context = {}
            for ep in self._episodes.values():
                context_hash = self._context_hash(ep.context)
                if context_hash not in self._episodes_by_context:
                    self._episodes_by_context[context_hash] = []
                self._episodes_by_context[context_hash].append(ep.id)
                
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Failed to load episodic memory: {e}")
    
    def __len__(self) -> int:
        return len(self._episodes)
