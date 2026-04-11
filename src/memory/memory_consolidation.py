"""
Memory Consolidation - 记忆整合模块
====================================

负责将工作记忆和情景记忆整合到长期记忆中。

整合策略:
- 重要性过滤: 只有重要的记忆进入长期存储
- 总结压缩: 将多个相似记忆整合
- 关联构建: 建立记忆之间的关联
- 遗忘模拟: 不重要的记忆逐渐遗忘
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import time
import threading
from collections import defaultdict


@dataclass
class ConsolidationConfig:
    """整合配置"""
    min_importance_threshold: float = 5.0    # 进入长期记忆的最低重要性
    consolidation_interval_s: float = 3600.0  # 整合间隔 (1小时)
    max_memories_per_consolidation: int = 50   # 每次最大整合数
    similarity_threshold: float = 0.7          # 相似度阈值
    decay_base_rate: float = 0.001            # 基础衰减率
    enable_auto_consolidation: bool = True    # 自动整合


@dataclass
class ConsolidationResult:
    """整合结果"""
    consolidated_count: int
    episodic_to_semantic: int   # 情景 -> 语义
    pruned_count: int            # 遗忘删除数
    new_associations: int        # 新关联数
    duration_ms: float
    errors: List[str] = field(default_factory=list)


class MemoryConsolidation:
    """
    记忆整合器
    
    职责:
    1. 将重要情景记忆整合为语义知识
    2. 建立记忆间的关联
    3. 遗忘不重要的记忆
    4. 定期整合
    """
    
    def __init__(
        self,
        episodic_memory,
        semantic_memory,
        procedural_memory,
        config: Optional[ConsolidationConfig] = None,
    ):
        """
        Args:
            episodic_memory: 情景记忆实例
            semantic_memory: 语义记忆实例
            procedural_memory: 程序记忆实例
            config: 整合配置
        """
        self._episodic = episodic_memory
        self._semantic = semantic_memory
        self._procedural = procedural_memory
        self.config = config or ConsolidationConfig()
        
        # 整合统计
        self._last_consolidation = time.time()
        self._total_consolidations = 0
        self._total_consolidated = 0
        
        # 相似记忆缓存
        self._similarity_cache: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # 自动整合线程
        self._consolidation_thread: Optional[threading.Thread] = None
        self._stop_thread = threading.Event()
        
        if self.config.enable_auto_consolidation:
            self._start_auto_consolidation()
    
    def _start_auto_consolidation(self) -> None:
        """启动自动整合线程"""
        def consolidation_loop():
            while not self._stop_thread.wait(self.config.consolidation_interval_s):
                self.consolidate()
        
        self._consolidation_thread = threading.Thread(
            target=consolidation_loop,
            daemon=True
        )
        self._consolidation_thread.start()
    
    def stop_auto_consolidation(self) -> None:
        """停止自动整合"""
        self._stop_thread.set()
        if self._consolidation_thread:
            self._consolidation_thread.join(timeout=5.0)
    
    # ==================== 主整合流程 ====================
    
    def consolidate(
        self,
        force: bool = False,
    ) -> ConsolidationResult:
        """
        执行整合
        
        Args:
            force: 是否强制整合
            
        Returns:
            整合结果
        """
        start_time = time.time()
        result = ConsolidationResult(
            consolidated_count=0,
            episodic_to_semantic=0,
            pruned_count=0,
            new_associations=0,
            duration_ms=0,
        )
        
        try:
            # 1. 情景记忆 -> 语义记忆
            semantic_new = self._consolidate_episodic_to_semantic()
            result.episodic_to_semantic = semantic_new
            result.consolidated_count += semantic_new
            
            # 2. 提取技能到程序记忆
            skill_new = self._extract_skills_from_episodes()
            result.consolidated_count += skill_new
            
            # 3. 建立关联
            associations = self._build_associations()
            result.new_associations = associations
            
            # 4. 遗忘低价值记忆
            pruned = self._apply_forgetting()
            result.pruned_count = pruned
            
            self._total_consolidations += 1
            self._total_consolidated += result.consolidated_count
            self._last_consolidation = time.time()
            
        except Exception as e:
            result.errors.append(str(e))
        
        result.duration_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _consolidate_episodic_to_semantic(self) -> int:
        """将情景记忆整合到语义记忆"""
        # 获取待整合的情景记忆
        episodes = self._episodic.get_unconsolidated_episodes(
            min_importance=self.config.min_importance_threshold,
            max_age_days=1.0,  # 超过1天的记忆才整合
        )
        
        if not episodes:
            return 0
        
        episodes = episodes[:self.config.max_memories_per_consolidation]
        new_concepts = 0
        
        for ep in episodes:
            # 1. 创建概念
            concept_name = self._generate_concept_name(ep)
            if concept_name:
                # 检查是否已存在
                existing = self._semantic.find_concept_by_name(concept_name)
                if not existing:
                    concept = self._semantic.add_concept(
                        name=concept_name,
                        category=self._infer_category(ep),
                        description=ep.summary,
                        confidence=min(1.0, ep.importance_score / 10.0),
                        source=KnowledgeSource.DIRECT_EXPERIENCE,
                        source_episode_id=ep.id,
                        tags=ep.tags,
                    )
                    new_concepts += 1
                    
                    # 2. 添加相关事实
                    if ep.entities:
                        for entity in ep.entities:
                            entity_concept = self._semantic.find_concept_by_name(entity)
                            if entity_concept:
                                self._semantic.add_fact(
                                    subject_id=entity_concept.id,
                                    predicate="参与",
                                    object_value=concept_name,
                                    confidence=0.7,
                                    source=KnowledgeSource.DIRECT_EXPERIENCE,
                                    source_episode_id=ep.id,
                                )
                    
                    # 3. 提取经验教训为规则
                    for lesson in ep.lessons_learned:
                        if lesson:
                            self._semantic.add_rule(
                                if_conditions=[concept_name],
                                then_conclusion=lesson,
                                confidence=ep.importance_score / 10.0,
                                source_episode_id=ep.id,
                            )
                    
                    # 标记已整合
                    self._episodic.mark_consolidated(ep.id)
        
        return new_concepts
    
    def _generate_concept_name(self, episode) -> Optional[str]:
        """生成概念名称"""
        # 简单策略: 使用摘要的前几个词
        summary = episode.summary.strip()
        if not summary:
            return None
        
        words = summary.split()
        if len(words) <= 3:
            name = summary
        else:
            name = " ".join(words[:3])
        
        # 清理
        name = name.title()
        return name
    
    def _infer_category(self, episode) -> str:
        """推断类别"""
        # 基于标签推断
        category_keywords = {
            'navigation': ['导航', '移动', '路径', '避障'],
            'manipulation': ['抓取', '放置', '操作', '装配'],
            'communication': ['对话', '指令', '交互'],
            'learning': ['学习', '训练', '探索'],
            'safety': ['安全', '紧急', '停止'],
        }
        
        tags_lower = [t.lower() for t in episode.tags]
        
        for cat, keywords in category_keywords.items():
            if any(kw in ' '.join(tags_lower) for kw in keywords):
                return cat
        
        return 'general'
    
    def _extract_skills_from_episodes(self) -> int:
        """从经验中提取技能"""
        # 获取高成功率的记忆
        recent_eps = self._episodic.retrieve_recent(limit=100)
        
        # 按相似性聚类
        clusters = self._cluster_episodes(recent_eps)
        
        new_skills = 0
        
        for cluster in clusters:
            if len(cluster) < 3:
                continue
            
            # 检查是否是成功的模式
            success_rate = sum(1 for ep in cluster if ep.outcomes.get('success', False)) / len(cluster)
            if success_rate < 0.7:
                continue
            
            # 创建技能
            first_ep = cluster[0]
            skill = self._procedural.add_skill(
                name=f"模式_{first_ep.summary[:20]}",
                description=f"从{len(cluster)}次经验中提取的技能",
                category=self._infer_category(first_ep),
                procedure_type="流程",
                steps=[{'action': ep.actions} for ep in cluster],
                applicable_contexts=first_ep.tags,
                tags=['extracted', 'consolidated'],
                source_episode_id=cluster[0].id if cluster else None,
            )
            
            new_skills += 1
        
        return new_skills
    
    def _cluster_episodes(
        self,
        episodes: List,
        similarity_threshold: float = 0.7,
    ) -> List[List]:
        """简单聚类"""
        clusters = []
        assigned = set()
        
        for i, ep1 in enumerate(episodes):
            if ep1.id in assigned:
                continue
            
            cluster = [ep1]
            assigned.add(ep1.id)
            
            for ep2 in episodes[i+1:]:
                if ep2.id in assigned:
                    continue
                
                # 计算相似度
                similarity = self._episodic._compute_context_similarity(
                    ep1.context,
                    ep2.context
                )
                
                if similarity >= similarity_threshold:
                    cluster.append(ep2)
                    assigned.add(ep2.id)
            
            clusters.append(cluster)
        
        return clusters
    
    def _build_associations(self) -> int:
        """建立记忆间关联"""
        associations = 0
        
        # 情景记忆间的关联
        recent_eps = self._episodic.retrieve_recent(limit=50)
        
        for ep in recent_eps:
            concept = self._semantic.find_concept_by_name(
                self._generate_concept_name(ep)
            )
            
            if not concept:
                continue
            
            # 关联相关概念
            for entity in ep.entities:
                entity_concept = self._semantic.find_concept_by_name(entity)
                if entity_concept and entity_concept.id != concept.id:
                    # 检查是否已有关联
                    if entity not in concept.relations:
                        self._semantic.update_concept(
                            concept.id,
                            add_relations={entity_concept.id: "相关"},
                        )
                        associations += 1
        
        return associations
    
    def _apply_forgetting(self) -> int:
        """应用遗忘算法 - 基于重要性和时间衰减的综合遗忘"""
        before = len(self._episodic)
        now = time.time()
        
        # 计算所有记忆的保留分数
        memory_scores = []
        for ep_id, ep in self._episodic._episodes.items():
            # 时间衰减: 越旧分数越低 (半衰期30天)
            age_days = (now - ep.timestamp) / (3600 * 24)
            time_decay = np.exp(-age_days / 30)  # 指数衰减
            
            # 重要性权重: 越重要分数越高
            importance_weight = ep.importance_score / 10.0
            
            # 使用频率加权: 被检索过的记忆更不容易遗忘
            usage_weight = min(1.0, 0.5 + ep.retrieval_count * 0.1) if hasattr(ep, 'retrieval_count') else 0.5
            
            # 综合保留分数
            retention_score = importance_weight * time_decay * usage_weight
            
            # 已整合的记忆保留分数加倍
            if ep.consolidated:
                retention_score *= 2.0
            
            memory_scores.append((ep_id, retention_score, ep.importance_score, age_days))
        
        # 按保留分数升序排序 (低分先遗忘)
        memory_scores.sort(key=lambda x: x[1])
        
        # 确定要遗忘的数量: 总容量的10%，最低0，最多100条
        total_memories = len(memory_scores)
        forget_count = min(100, max(0, int(total_memories * 0.1)))
        
        # 保留重要性>=8的记忆，无论分数多低
        pruned = 0
        for i in range(forget_count):
            ep_id, score, importance, age = memory_scores[i]
            if importance >= 8.0:
                continue  # 高重要性记忆不遗忘
            # 遗忘分数低于阈值的记忆
            if score < 0.2:
                self._episodic.delete_episode(ep_id)
                pruned += 1
        
        # 语义记忆遗忘: 低置信度、长期未使用的概念
        if self._semantic:
            for concept_id, concept in self._semantic._concepts.items():
                age_days = (now - concept.updated_at) / (3600 * 24) if hasattr(concept, 'updated_at') else 365
                if concept.confidence < 0.3 and age_days > 90:
                    self._semantic.delete_concept(concept_id)
                    pruned += 1
        
        # 程序记忆遗忘: 低成功率、长期未使用的技能
        if self._procedural:
            for skill_id, skill in self._procedural._skills.items():
                age_days = (now - skill.last_used) / (3600 * 24) if hasattr(skill, 'last_used') else 180
                if skill.success_rate < 0.3 and age_days > 60:
                    self._procedural.delete_skill(skill_id)
                    pruned += 1
        
        return pruned
    
    # ==================== 即时整合 ====================
    
    def integrate_episode_immediately(
        self,
        episode_id: str,
    ) -> bool:
        """
        立即整合单个记忆
        
        Args:
            episode_id: 记忆ID
            
        Returns:
            是否成功
        """
        ep = self._episodic.get_episode(episode_id)
        if not ep:
            return False
        
        # 立即整合到语义记忆
        concept_name = self._generate_concept_name(ep)
        if concept_name:
            existing = self._semantic.find_concept_by_name(concept_name)
            if not existing:
                self._semantic.add_concept(
                    name=concept_name,
                    category=self._infer_category(ep),
                    description=ep.summary,
                    confidence=ep.importance_score / 10.0,
                    source=KnowledgeSource.DIRECT_EXPERIENCE,
                    source_episode_id=ep.id,
                    tags=ep.tags,
                )
                
                # 标记已整合
                self._episodic.mark_consolidated(ep.id)
                return True
        
        return False
    
    # ==================== 统计和状态 ====================
    
    def get_consolidation_status(self) -> Dict[str, Any]:
        """获取整合状态"""
        return {
            'last_consolidation_age_s': time.time() - self._last_consolidation,
            'total_consolidations': self._total_consolidations,
            'total_consolidated': self._total_consolidated,
            'config': {
                'min_importance_threshold': self.config.min_importance_threshold,
                'consolidation_interval_s': self.config.consolidation_interval_s,
                'enable_auto': self.config.enable_auto_consolidation,
            },
            'pending_episodes': len(
                self._episodic.get_unconsolidated_episodes(
                    min_importance=self.config.min_importance_threshold,
                    max_age_days=1.0,
                )
            ),
        }
    
    def force_consolidation_now(self) -> ConsolidationResult:
        """强制立即整合"""
        return self.consolidate(force=True)


# 需要导入 KnowledgeSource
from .semantic_memory import KnowledgeSource
