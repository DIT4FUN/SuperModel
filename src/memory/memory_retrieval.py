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
    ):
        """
        Args:
            episodic_memory: 情景记忆实例
            semantic_memory: 语义记忆实例
            procedural_memory: 程序记忆实例
        """
        self._episodic = episodic_memory
        self._semantic = semantic_memory
        self._procedural = procedural_memory
        
        # 检索历史
        self._history: List[RetrievalQuery] = []
        self._max_history = 100
        
        # 缓存
        self._cache: Dict[str, List[RetrievalResult]] = {}
        self._cache_lock = threading.RLock()
        self._cache_ttl = 60.0  # 缓存生存时间 (秒)
    
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
            if cache_key in self._cache:
                cached_time, results = self._cache[cache_key]
                if time.time() - cached_time < self._cache_ttl:
                    return results
        
        # 执行检索
        results = self._execute_retrieval(query)
        
        # 过滤和排序
        results = self._filter_and_rank(results, query)
        
        # 缓存
        with self._cache_lock:
            self._cache[cache_key] = (time.time(), results)
        
        # 记录历史
        self._history.append(query)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        
        return results
    
    def _execute_retrieval(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """执行实际检索"""
        results = []
        
        # 确定检索的记忆类型
        memory_types = ['episodic', 'semantic', 'procedural']
        if query.memory_type:
            memory_types = [query.memory_type]
        
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
    
    def clear_history(self) -> None:
        """清除检索历史"""
        self._history.clear()
        with self._cache_lock:
            self._cache.clear()


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
