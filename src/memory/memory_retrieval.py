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
Memory Retrieval - 记忆检索系统
==============================

统一的检索接口，支持多种检索策略。

检索策略:
- 精确匹配
- 相似度检索
- 语义检索 (可选)
- 时间范围检索
- 关联检索
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
from collections import OrderedDict
import json

# 可选依赖
SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

OPENAI_AVAILABLE = False
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    pass


class RetrievalStrategy(Enum):
    """检索策略"""
    EXACT = "exact"              # 精确匹配
    SIMILARITY = "similarity"    # 相似度匹配
    SEMANTIC = "semantic"        # 语义检索 (需要嵌入)
    TEMPORAL = "temporal"        # 时间范围
    ASSOCIATIVE = "associative"  # 关联检索
    HYBRID = "hybrid"            # 混合策略


@dataclass
class RetrievalQuery:
    """检索查询"""
    content: Optional[str] = None           # 文本内容
    context: Optional[Dict[str, Any]] = None # 上下文
    entities: Optional[List[str]] = None      # 实体列表
    time_range: Optional[Tuple[float, float]] = None  # 时间范围
    tags: Optional[List[str]] = None          # 标签
    limit: int = 20                          # 返回数量
    min_score: float = 0.0                   # 最低分数
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    memory_type: Optional[str] = None         # 限定记忆类型


@dataclass
class RetrievalResult:
    """检索结果"""
    memory_id: str
    memory_type: str  # episodic / semantic / procedural
    score: float
    data: Dict[str, Any]
    match_reasons: List[str] = field(default_factory=list)
    age_hours: float = 0.0


class MemoryRetrieval:
    """
    统一记忆检索系统
    
    提供:
    - 多类型记忆的联合检索
    - 可配置的检索策略
    - 检索结果排序和过滤
    - 检索历史
    """
    
    def __init__(
        self,
        episodic_memory=None,
        semantic_memory=None,
        procedural_memory=None,
        memory_store=None,
        # 嵌入配置
        embedding_model: str = "all-MiniLM-L6-v2",  # 本地嵌入模型
        embedding_provider: str = "local",  # local/openai
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        embedding_dim: int = 1536,
    ):
        """
        Args:
            episodic_memory: 情景记忆实例
            semantic_memory: 语义记忆实例
            procedural_memory: 程序记忆实例
            memory_store: 存储层实例 (用于向量检索)
            embedding_model: 嵌入模型名称
            embedding_provider: 嵌入提供商 (local/openai)
            openai_api_key: OpenAI API密钥
            openai_base_url: OpenAI API端点
            embedding_dim: 嵌入维度
        """
        self._episodic = episodic_memory
        self._semantic = semantic_memory
        self._procedural = procedural_memory
        self._memory_store = memory_store
        
        # 嵌入配置
        self.embedding_provider = embedding_provider
        self.embedding_dim = embedding_dim
        self._embedding_model: Optional[Any] = None
        self._openai_client: Optional[Any] = None
        
        # 初始化嵌入模型
        if embedding_provider == "local" and SENTENCE_TRANSFORMERS_AVAILABLE:
            self._embedding_model = SentenceTransformer(embedding_model)
        elif embedding_provider == "openai" and OPENAI_AVAILABLE:
            self._openai_client = openai.OpenAI(api_key=openai_api_key, base_url=openai_base_url)
        
        # 检索历史
        self._history: List[RetrievalQuery] = []
        self._max_history = 100
        
        # 缓存 (LRU)
        self._cache: OrderedDict[str, Tuple[float, List[RetrievalResult]]] = OrderedDict()
        self._cache_lock = threading.RLock()
        self._default_cache_ttl = 60.0  # 默认缓存生存时间 (秒)
        self._max_cache_entries = 200  # 最大缓存条目数
        # 缓存统计
        self._cache_hits = 0
        self._cache_misses = 0
        self._query_access_count: Dict[str, int] = {}  # 查询访问次数，用于调整ttl
        
        # 经验回放统计
        self._replay_count = 0
        self._replay_success_count = 0
    
    # ==================== 主检索接口 ====================
    
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        执行检索
        
        Args:
            query: 检索查询
            
        Returns:
            检索结果列表
        """
        start_time = time.time()
        
        # 检查缓存
        cache_key = self._get_cache_key(query)
        with self._cache_lock:
            # 更新访问计数
            self._query_access_count[cache_key] = self._query_access_count.get(cache_key, 0) + 1
            
            if cache_key in self._cache:
                cached_time, results = self._cache[cache_key]
                # 热门查询延长ttl：访问次数>10次，ttl延长到5分钟；>50次，延长到30分钟
                access_count = self._query_access_count[cache_key]
                ttl = self._default_cache_ttl
                if access_count > 50:
                    ttl = 1800.0
                elif access_count > 10:
                    ttl = 300.0
                
                if time.time() - cached_time < ttl:
                    # LRU：移动到末尾
                    self._cache.move_to_end(cache_key)
                    self._cache_hits += 1
                    return results
                else:
                    # 过期删除
                    del self._cache[cache_key]
        
        self._cache_misses += 1
        # 执行检索
        results = self._execute_retrieval(query)
        
        # 过滤和排序
        results = self._filter_and_rank(results, query)
        
        # 缓存
        with self._cache_lock:
            self._cache[cache_key] = (time.time(), results)
            # LRU淘汰超过最大条目的
            if len(self._cache) > self._max_cache_entries:
                self._cache.popitem(last=False)
        
        # 记录历史
        self._history.append(query)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        
        return results
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """获取文本嵌入向量"""
        if not text:
            return None
        
        try:
            if self.embedding_provider == "local" and self._embedding_model:
                return self._embedding_model.encode(text, convert_to_numpy=True)
            elif self.embedding_provider == "openai" and self._openai_client:
                response = self._openai_client.embeddings.create(input=text, model="text-embedding-ada-002")
                return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Embedding generation failed: {e}")
        
        return None
    
    def _execute_retrieval(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """执行实际检索"""
        results = []
        
        # 确定检索的记忆类型
        memory_types = ['episodic', 'semantic', 'procedural']
        if query.memory_type:
            memory_types = [query.memory_type]
        
        # 语义向量检索 (优先)
        if query.strategy in [RetrievalStrategy.SEMANTIC, RetrievalStrategy.HYBRID] and self._memory_store and query.content:
            query_vector = self._get_embedding(query.content)
            if query_vector is not None:
                vector_results = self._memory_store.search_vectors(query_vector, top_k=query.limit * 2)
                for memory_id, score, metadata in vector_results:
                    memory_type = metadata.get('type', 'unknown')
                    if memory_type not in memory_types:
                        continue
                    
                    # 加载记忆数据
                    if memory_type == 'episodic' and self._episodic:
                        memory_data = self._episodic.get_episode(memory_id)
                    elif memory_type == 'semantic' and self._semantic:
                        memory_data = self._semantic.get_concept(memory_id)
                    elif memory_type == 'procedural' and self._procedural:
                        memory_data = self._procedural.get_skill(memory_id)
                    else:
                        continue
                    
                    if memory_data:
                        results.append(RetrievalResult(
                            memory_id=memory_id,
                            memory_type=memory_type,
                            score=score,
                            data=memory_data.to_dict(),
                            match_reasons=["语义匹配"],
                            age_hours=(time.time() - memory_data.timestamp) / 3600 if hasattr(memory_data, 'timestamp') else 0.0,
                        ))
        
        # 传统关键词检索
        if query.strategy != RetrievalStrategy.SEMANTIC or len(results) < query.limit:
            # 情景记忆检索
            if 'episodic' in memory_types and self._episodic:
                results.extend(self._retrieve_episodic(query))
            
            # 语义记忆检索
            if 'semantic' in memory_types and self._semantic:
                results.extend(self._retrieve_semantic(query))
            
            # 程序记忆检索
            if 'procedural' in memory_types and self._procedural:
                results.extend(self._retrieve_procedural(query))
        
        return results
    
    # ==================== 经验回放 ====================
    
    def experience_replay(
        self,
        num_samples: int = 10,
        priority: str = "importance",  # importance / recency / random
        task_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        经验回放 - 从历史记忆中采样经验用于训练和回忆
        
        Args:
            num_samples: 采样数量
            priority: 采样优先级
                - importance: 重要性优先
                - recency: 新近度优先
                - random: 随机采样
            task_filter: 可选的任务过滤
            
        Returns:
            经验列表
        """
        self._replay_count += 1
        
        # 获取所有记忆
        all_memories = []
        
        if self._episodic:
            for ep in self._episodic._episodes.values():
                if task_filter and task_filter not in ep.summary:
                    continue
                all_memories.append({
                    'type': 'episodic',
                    'data': ep,
                    'importance': ep.importance_score,
                    'timestamp': ep.timestamp,
                })
        
        if self._procedural:
            for skill in self._procedural._skills.values():
                if task_filter and task_filter not in skill.name:
                    continue
                all_memories.append({
                    'type': 'procedural',
                    'data': skill,
                    'importance': skill.success_rate * 10,
                    'timestamp': skill.created_at,
                })
        
        if not all_memories:
            return []
        
        # 采样
        if priority == "importance":
            all_memories.sort(key=lambda x: x['importance'], reverse=True)
        elif priority == "recency":
            all_memories.sort(key=lambda x: x['timestamp'], reverse=True)
        elif priority == "random":
            import random
            random.shuffle(all_memories)
        
        selected = all_memories[:num_samples]
        
        # 转换为结果格式
        results = []
        for mem in selected:
            results.append({
                'type': mem['type'],
                'content': mem['data'].to_dict(),
                'importance': mem['importance'],
                'age_hours': (time.time() - mem['timestamp']) / 3600,
            })
        
        self._replay_success_count += 1
        return results
    
    def get_replay_statistics(self) -> Dict[str, Any]:
        """获取经验回放统计"""
        return {
            'total_replays': self._replay_count,
            'successful_replays': self._replay_success_count,
            'success_rate': self._replay_success_count / self._replay_count if self._replay_count > 0 else 0.0,
        }
    
    def _retrieve_episodic(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """检索情景记忆"""
        results = []
        
        # 内容检索
        if query.content:
            # 简单关键词匹配
            keywords = query.content.lower().split()
            
            for ep in self._episodic._episodes.values():
                score = 0.0
                reasons = []
                
                # 摘要匹配
                summary_lower = ep.summary.lower()
                for kw in keywords:
                    if kw in summary_lower:
                        score += 0.3
                        reasons.append(f"关键词'{kw}'在摘要中")
                
                # 标签匹配
                for tag in ep.tags:
                    for kw in keywords:
                        if kw in tag.lower():
                            score += 0.2
                            reasons.append(f"匹配标签'{tag}'")
                
                # 实体匹配
                for entity in ep.entities:
                    for kw in keywords:
                        if kw in entity.lower():
                            score += 0.25
                            reasons.append(f"匹配实体'{entity}'")
                
                if score > 0:
                    results.append(RetrievalResult(
                        memory_id=ep.id,
                        memory_type='episodic',
                        score=min(1.0, score),
                        data=ep.to_dict(),
                        match_reasons=reasons,
                        age_hours=(time.time() - ep.timestamp) / 3600,
                    ))
        
        # 实体检索
        if query.entities:
            entity_eps = self._episodic.retrieve_by_entities(query.entities, limit=query.limit)
            for ep in entity_eps:
                results.append(RetrievalResult(
                    memory_id=ep.id,
                    memory_type='episodic',
                    score=0.8,
                    data=ep.to_dict(),
                    match_reasons=[f"实体匹配: {query.entities}"],
                    age_hours=(time.time() - ep.timestamp) / 3600,
                ))
        
        # 时间范围检索
        if query.time_range:
            start, end = query.time_range
            time_eps = self._episodic.retrieve_by_time(start, end, limit=query.limit)
            for ep in time_eps:
                results.append(RetrievalResult(
                    memory_id=ep.id,
                    memory_type='episodic',
                    score=0.7,
                    data=ep.to_dict(),
                    match_reasons=[f"时间范围匹配"],
                    age_hours=(time.time() - ep.timestamp) / 3600,
                ))
        
        # 标签检索
        if query.tags:
            tag_eps = self._episodic.retrieve_by_tags(query.tags, limit=query.limit)
            for ep in tag_eps:
                results.append(RetrievalResult(
                    memory_id=ep.id,
                    memory_type='episodic',
                    score=0.75,
                    data=ep.to_dict(),
                    match_reasons=[f"标签匹配: {query.tags}"],
                    age_hours=(time.time() - ep.timestamp) / 3600,
                ))
        
        return results
    
    def _retrieve_semantic(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """检索语义记忆"""
        results = []
        
        # 内容检索
        if query.content:
            concepts = self._semantic.search_concepts(query.content, limit=query.limit)
            for concept in concepts:
                results.append(RetrievalResult(
                    memory_id=concept.id,
                    memory_type='semantic',
                    score=concept.confidence,
                    data=concept.to_dict(),
                    match_reasons=[f"概念匹配: {concept.name}"],
                    age_hours=(time.time() - concept.created_at) / 3600,
                ))
        
        # 实体检索
        if query.entities:
            for entity_name in query.entities:
                concept = self._semantic.find_concept_by_name(entity_name)
                if concept:
                    results.append(RetrievalResult(
                        memory_id=concept.id,
                        memory_type='semantic',
                        score=concept.confidence,
                        data=concept.to_dict(),
                        match_reasons=[f"实体概念: {concept.name}"],
                        age_hours=(time.time() - concept.created_at) / 3600,
                    ))
        
        return results
    
    def _retrieve_procedural(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """检索程序记忆"""
        results = []
        
        # 内容检索
        if query.content:
            skills = self._procedural.search_skills(query.content, limit=query.limit)
            for skill in skills:
                results.append(RetrievalResult(
                    memory_id=skill.id,
                    memory_type='procedural',
                    score=skill.success_rate,
                    data=skill.to_dict(),
                    match_reasons=[f"技能匹配: {skill.name}"],
                    age_hours=(time.time() - skill.created_at) / 3600,
                ))
        
        return results
    
    # ==================== 辅助方法 ====================
    
    def _filter_and_rank(
        self,
        results: List[RetrievalResult],
        query: RetrievalQuery,
    ) -> List[RetrievalResult]:
        """过滤和排序结果"""
        # 过滤
        results = [r for r in results if r.score >= query.min_score]
        
        # 去重 (同一记忆只保留最高分)
        seen = {}
        for r in results:
            if r.memory_id not in seen or r.score > seen[r.memory_id].score:
                seen[r.memory_id] = r
        
        results = list(seen.values())
        
        # 排序
        # 综合考虑: 分数、新近度、匹配原因数
        def ranking_key(r: RetrievalResult):
            recency_boost = 1.0 / (1.0 + r.age_hours / 24)  # 越新越高
            match_boost = len(r.match_reasons) * 0.05
            
            return r.score * (1.0 + recency_boost * 0.3 + match_boost)
        
        results.sort(key=ranking_key, reverse=True)
        
        # 限制数量
        return results[:query.limit]
    
    def _get_cache_key(self, query: RetrievalQuery) -> str:
        """生成缓存键"""
        parts = [
            query.content or "",
            str(sorted(query.entities)) if query.entities else "",
            str(query.time_range) if query.time_range else "",
            str(sorted(query.tags)) if query.tags else "",
            query.memory_type or "",
            str(query.limit),
        ]
        return "|".join(parts)
    
    # ==================== 关联检索 ====================
    
    def retrieve_associative(
        self,
        memory_id: str,
        memory_type: str,
        depth: int = 2,
        limit: int = 20,
    ) -> List[RetrievalResult]:
        """
        关联检索 - 从给定记忆出发，检索关联的记忆
        
        Args:
            memory_id: 起始记忆ID
            memory_type: 记忆类型
            depth: 关联深度
            limit: 返回数量
            
        Returns:
            关联结果
        """
        results = []
        visited = {memory_id}
        current_ids = {(memory_id, memory_type)}
        
        for _ in range(depth):
            next_ids = set()
            
            for mid, mtype in current_ids:
                # 获取关联
                if mtype == 'episodic':
                    associated = self._get_associated_episodic(mid)
                elif mtype == 'semantic':
                    associated = self._get_associated_semantic(mid)
                else:
                    continue
                
                for aid, atype, score in associated:
                    if aid not in visited:
                        visited.add(aid)
                        next_ids.add((aid, atype))
                        
                        results.append(RetrievalResult(
                            memory_id=aid,
                            memory_type=atype,
                            score=score,
                            data={},  # 简化
                            match_reasons=[f"关联于 {mid}"],
                        ))
            
            current_ids = next_ids
        
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]
    
    def _get_associated_episodic(self, episode_id: str) -> List[Tuple[str, str, float]]:
        """获取情景记忆的关联"""
        if not self._episodic:
            return []
        
        ep = self._episodic.get_episode(episode_id)
        if not ep:
            return []
        
        associated = []
        
        # 通过实体关联
        for entity in ep.entities:
            related = self._episodic.retrieve_by_entities([entity], limit=5)
            for rel_ep in related:
                if rel_ep.id != episode_id:
                    associated.append((rel_ep.id, 'episodic', 0.7))
        
        # 通过标签关联
        for tag in ep.tags:
            related = self._episodic.retrieve_by_tags([tag], limit=5)
            for rel_ep in related:
                if rel_ep.id != episode_id:
                    associated.append((rel_ep.id, 'episodic', 0.6))
        
        return associated
    
    def _get_associated_semantic(self, concept_id: str) -> List[Tuple[str, str, float]]:
        """获取概念记忆的关联"""
        if not self._semantic:
            return []
        
        concept = self._semantic.get_concept(concept_id)
        if not concept:
            return []
        
        associated = []
        
        # 通过关系关联
        for rel_id, rel_type in concept.relations.items():
            associated.append((rel_id, 'semantic', 0.8))
        
        return associated
    
    # ==================== 检索建议 ====================
    
    def suggest_retrieval(self, context: Dict[str, Any]) -> List[str]:
        """
        根据上下文提供检索建议
        
        Returns:
            建议的检索内容列表
        """
        suggestions = []
        
        # 基于当前激活的实体
        if 'active_entities' in context:
            for entity in context['active_entities']:
                suggestions.append(f"关于 {entity} 的记忆")
        
        # 基于当前任务
        if 'task' in context:
            task = context['task']
            suggestions.append(f"完成 {task} 的经验")
            suggestions.append(f"{task} 相关的技能")
        
        # 基于情感状态
        if 'emotional_state' in context:
            emotion = context['emotional_state']
            if emotion == 'positive':
                suggestions.append("成功的经验")
            elif emotion == 'negative':
                suggestions.append("失败的教训")
        
        return suggestions[:5]
    
    # ==================== 历史 ====================
    
    def get_history(self, limit: int = 20) -> List[RetrievalQuery]:
        """获取检索历史"""
        return self._history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        with self._cache_lock:
            total_queries = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total_queries if total_queries > 0 else 0.0
            
            return {
                'total_queries': total_queries,
                'cache_hits': self._cache_hits,
                'cache_misses': self._cache_misses,
                'cache_hit_rate': hit_rate,
                'cache_entries': len(self._cache),
                'max_cache_entries': self._max_cache_entries,
            }
    
    def clear_history(self) -> None:
        """清除检索历史"""
        self._history.clear()
        with self._cache_lock:
            self._cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            self._query_access_count.clear()


# ==================== 便捷函数 ====================

def create_retrieval_query(
    content: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    entities: Optional[List[str]] = None,
    time_range: Optional[Tuple[float, float]] = None,
    tags: Optional[List[str]] = None,
    limit: int = 20,
    strategy: str = "hybrid",
    memory_type: Optional[str] = None,
) -> RetrievalQuery:
    """创建检索查询的便捷函数"""
    return RetrievalQuery(
        content=content,
        context=context,
        entities=entities,
        time_range=time_range,
        tags=tags,
        limit=limit,
        strategy=RetrievalStrategy(strategy),
        memory_type=memory_type,
    )
