"""
Long-Term Memory - 统一长期记忆接口
====================================

整合所有记忆模块的统一接口。

使用方式:
  ltm = LongTermMemory()
  
  # 存储记忆
  ltm.store_episode(summary="完成抓取任务", context={'object': 'box'})
  ltm.store_knowledge(name="物体", category="物理实体")
  ltm.store_skill(name="抓取", steps=[...])
  
  # 检索记忆
  results = ltm.retrieve("抓取经验")
  
  # 获取状态
  status = ltm.get_status()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
import time
import threading
import json

from .episodic_memory import EpisodicMemory, Episode, EmotionalTag, ImportanceLevel
from .semantic_memory import SemanticMemory, Concept, Fact, Rule, KnowledgeSource
from .procedural_memory import ProceduralMemory, Skill, SkillLevel
from .working_memory import WorkingMemory, WorkingMemoryConfig
from .memory_store import MemoryStore
from .memory_retrieval import MemoryRetrieval, RetrievalQuery, RetrievalResult, RetrievalStrategy
from .memory_consolidation import MemoryConsolidation, ConsolidationConfig, ConsolidationResult


@dataclass
class MemoryConfig:
    """记忆系统配置"""
    # 存储路径
    store_path: str = "./memory_data"
    
    # 容量限制
    max_episodes: int = 10000
    max_concepts: int = 5000
    max_skills: int = 1000
    
    # 整合设置
    consolidation_interval_s: float = 3600.0
    min_importance_threshold: float = 5.0
    
    # 工作记忆
    working_memory_config: WorkingMemoryConfig = None
    
    # 自动保存
    auto_save: bool = True
    save_interval_s: float = 60.0


class LongTermMemory:
    """
    统一长期记忆系统
    
    整合:
    - 情景记忆 (Episodic)
    - 语义记忆 (Semantic)
    - 程序记忆 (Procedural)
    - 工作记忆 (Working)
    - 记忆整合 (Consolidation)
    - 统一检索 (Retrieval)
    """
    
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
    ):
        """
        Args:
            config: 记忆系统配置
        """
        self.config = config or MemoryConfig()
        
        # 确保存储目录存在
        store_path = Path(self.config.store_path)
        store_path.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.RLock()
        
        # 初始化各模块
        # 1. 情景记忆
        self.episodic = EpisodicMemory(
            store_path=self.config.store_path,
            max_episodes=self.config.max_episodes,
        )
        
        # 2. 语义记忆
        self.semantic = SemanticMemory(
            store_path=self.config.store_path,
        )
        
        # 3. 程序记忆
        self.procedural = ProceduralMemory(
            store_path=self.config.store_path,
        )
        
        # 4. 工作记忆
        working_config = self.config.working_memory_config or WorkingMemoryConfig()
        self.working = WorkingMemory(config=working_config)
        
        # 5. 存储层
        self.store = MemoryStore(
            base_path=self.config.store_path,
            auto_save=self.config.auto_save,
            save_interval_s=self.config.save_interval_s,
        )
        
        # 6. 检索系统
        self.retrieval = MemoryRetrieval(
            episodic_memory=self.episodic,
            semantic_memory=self.semantic,
            procedural_memory=self.procedural,
        )
        
        # 7. 整合系统
        consolidation_config = ConsolidationConfig(
            min_importance_threshold=self.config.min_importance_threshold,
            consolidation_interval_s=self.config.consolidation_interval_s,
        )
        self.consolidation = MemoryConsolidation(
            episodic_memory=self.episodic,
            semantic_memory=self.semantic,
            procedural_memory=self.procedural,
            config=consolidation_config,
        )
        
        # 状态
        self._initialized = True
        self._start_time = time.time()
        
        # 加载元数据
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """加载元数据"""
        metadata_file = Path(self.config.store_path) / "memory_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self._start_time = metadata.get('start_time', time.time())
            except Exception:
                pass
    
    # ==================== 情景记忆操作 ====================
    
    def store_episode(
        self,
        summary: str,
        context: Optional[Dict[str, Any]] = None,
        actions: Optional[List[Dict[str, Any]]] = None,
        outcomes: Optional[Dict[str, Any]] = None,
        emotional_tag: str = "neutral",
        importance_score: float = 5.0,
        duration_s: float = 0.0,
        entities: Optional[List[str]] = None,
        locations: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        lessons_learned: Optional[List[str]] = None,
    ) -> Episode:
        """
        存储情景记忆
        
        Args:
            summary: 记忆摘要
            context: 场景上下文
            actions: 执行的动作
            outcomes: 结果
            emotional_tag: 情感标签
            importance_score: 重要性 [0, 10]
            duration_s: 持续时间
            entities: 涉及的实体
            locations: 涉及的位置
            tags: 标签
            lessons_learned: 经验教训
            
        Returns:
            创建的Episode对象
        """
        with self._lock:
            ep = self.episodic.store(
                summary=summary,
                context=context or {},
                actions=actions,
                outcomes=outcomes,
                emotional_tag=EmotionalTag(emotional_tag),
                importance_score=importance_score,
                duration_s=duration_s,
                entities=entities,
                locations=locations,
                tags=tags,
                lessons_learned=lessons_learned,
            )
            
            # 更新工作记忆中的激活模式
            if entities:
                for entity in entities:
                    self.working.activate(entity)
            
            return ep
    
    def retrieve_episodes(
        self,
        content: Optional[str] = None,
        entities: Optional[List[str]] = None,
        time_range: Optional[tuple] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[Episode]:
        """
        检索情景记忆
        """
        with self._lock:
            results = []
            
            if content:
                # 内容检索
                hits = self.episodic.retrieve_by_context(
                    context={'content': content},
                    limit=limit,
                )
                results.extend([ep for ep, _ in hits])
            
            if entities:
                entity_eps = self.episodic.retrieve_by_entities(entities, limit=limit)
                results.extend(entity_eps)
            
            if time_range:
                time_eps = self.episodic.retrieve_by_time(
                    time_range[0],
                    time_range[1],
                    limit=limit,
                )
                results.extend(time_eps)
            
            if tags:
                tag_eps = self.episodic.retrieve_by_tags(tags, limit=limit)
                results.extend(tag_eps)
            
            if not results:
                results = self.episodic.retrieve_recent(limit=limit)
            
            # 去重
            seen = {}
            for ep in results:
                if ep.id not in seen:
                    seen[ep.id] = ep
            
            return list(seen.values())[:limit]
    
    # ==================== 语义记忆操作 ====================
    
    def store_knowledge(
        self,
        name: str,
        category: str = "general",
        properties: Optional[Dict[str, Any]] = None,
        description: str = "",
        confidence: float = 0.8,
        source: str = "direct",
        tags: Optional[List[str]] = None,
    ) -> Concept:
        """
        存储知识/概念
        """
        with self._lock:
            return self.semantic.add_concept(
                name=name,
                category=category,
                properties=properties,
                description=description,
                confidence=confidence,
                source=KnowledgeSource(source),
                tags=tags,
            )
    
    def add_fact(
        self,
        subject: str,
        predicate: str,
        object_value: Optional[Any] = None,
        object_concept_id: Optional[str] = None,
        confidence: float = 0.8,
        context: str = "",
    ) -> Optional[Fact]:
        """
        添加事实
        """
        with self._lock:
            # 查找subject概念
            subject_concept = self.semantic.find_concept_by_name(subject)
            if not subject_concept:
                # 创建subject概念
                subject_concept = self.semantic.add_concept(name=subject, category="entity")
            
            return self.semantic.add_fact(
                subject_id=subject_concept.id,
                predicate=predicate,
                object_id=object_concept_id,
                object_value=object_value,
                confidence=confidence,
                context=context,
            )
    
    def add_rule(
        self,
        if_conditions: List[str],
        then_conclusion: str,
        confidence: float = 0.7,
    ) -> Rule:
        """
        添加规则
        """
        with self._lock:
            return self.semantic.add_rule(
                if_conditions=if_conditions,
                then_conclusion=then_conclusion,
                confidence=confidence,
            )
    
    def search_knowledge(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 20,
    ) -> List[Concept]:
        """
        搜索知识
        """
        with self._lock:
            return self.semantic.search_concepts(
                query=query,
                category=category,
                limit=limit,
            )
    
    # ==================== 程序记忆操作 ====================
    
    def store_skill(
        self,
        name: str,
        description: str = "",
        category: str = "general",
        steps: Optional[List[Dict[str, Any]]] = None,
        code_reference: Optional[str] = None,
        conditions: Optional[List[str]] = None,
        contexts: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> Skill:
        """
        存储技能
        """
        with self._lock:
            return self.procedural.add_skill(
                name=name,
                description=description,
                category=category,
                steps=steps,
                code_reference=code_reference,
                conditions=conditions,
                applicable_contexts=contexts,
                tags=tags,
            )
    
    def update_skill(
        self,
        skill_name: str,
        success: bool,
        duration_s: float,
        feedback: Optional[str] = None,
    ) -> Optional[Skill]:
        """
        更新技能熟练度
        """
        with self._lock:
            skill = self.procedural.find_skill_by_name(skill_name)
            if not skill:
                return None
            
            return self.procedural.update_skill(
                skill_id=skill.id,
                success=success,
                duration_s=duration_s,
                feedback=feedback,
            )
    
    def find_skill(
        self,
        task: str,
        context: Optional[str] = None,
    ) -> Optional[Skill]:
        """
        查找适合任务的技能
        """
        with self._lock:
            return self.procedural.get_best_skill_for_task(
                task_description=task,
                context=context,
            )
    
    # ==================== 统一检索 ====================
    
    def retrieve(
        self,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[RetrievalResult]:
        """
        统一检索接口
        
        Args:
            query: 查询内容
            memory_type: 限定记忆类型 ('episodic'/'semantic'/'procedural')
            limit: 返回数量
            
        Returns:
            检索结果列表
        """
        with self._lock:
            retrieval_query = RetrievalQuery(
                content=query,
                memory_type=memory_type,
                limit=limit,
            )
            
            return self.retrieval.retrieve(retrieval_query)
    
    # ==================== 工作记忆操作 ====================
    
    def focus(self, key: str, content: Any, importance: float = 5.0) -> None:
        """将信息放入焦点"""
        self.working.focus(key, content, importance=importance)
    
    def get_focused(self, key: str) -> Optional[Any]:
        """获取焦点信息"""
        return self.working.get_focused(key)
    
    def get_working_summary(self) -> Dict[str, Any]:
        """获取工作记忆摘要"""
        return self.working.get_focus_summary()
    
    # ==================== 整合操作 ====================
    
    def consolidate(self) -> ConsolidationResult:
        """执行记忆整合"""
        return self.consolidation.consolidate()
    
    def get_consolidation_status(self) -> Dict[str, Any]:
        """获取整合状态"""
        return self.consolidation.get_consolidation_status()
    
    # ==================== 状态和统计 ====================
    
    def get_status(self) -> Dict[str, Any]:
        """获取记忆系统完整状态"""
        with self._lock:
            return {
                'uptime_s': time.time() - self._start_time,
                'episodic': {
                    'count': len(self.episodic),
                    'stats': self.episodic.get_statistics(),
                },
                'semantic': {
                    'count': len(self.semantic),
                    'stats': self.semantic.get_statistics(),
                },
                'procedural': {
                    'count': len(self.procedural),
                    'stats': self.procedural.get_statistics(),
                },
                'working': self.working.get_status(),
                'retrieval': self.retrieval.get_statistics(),
                'consolidation': self.get_consolidation_status(),
                'storage': self.store.get_storage_info(),
            }
    
    def get_memory_summary(self) -> str:
        """获取可读的记忆摘要"""
        status = self.get_status()
        
        lines = [
            "=== SuperModel 长期记忆系统 ===",
            f"运行时长: {status['uptime_s']/3600:.1f} 小时",
            "",
            f"情景记忆: {status['episodic']['count']} 条",
            f"  - 总记忆数: {status['episodic']['stats'].get('total_episodes', 0)}",
            f"  - 已整合: {status['episodic']['stats'].get('consolidated_count', 0)}",
            "",
            f"语义记忆: {status['semantic']['count']} 个概念",
            f"  - 事实数: {status['semantic']['stats'].get('total_facts', 0)}",
            f"  - 规则数: {status['semantic']['stats'].get('total_rules', 0)}",
            "",
            f"程序记忆: {status['procedural']['count']} 个技能",
            f"  - 总执行次数: {status['procedural']['stats'].get('total_executions', 0)}",
            f"  - 平均成功率: {status['procedural']['stats'].get('avg_success_rate', 0):.1%}",
            "",
            f"工作记忆: {status['working']['focus_count']} 项焦点",
            "",
            f"整合状态: {status['consolidation']['total_consolidated']} 条已整合",
        ]
        
        return "\n".join(lines)
    
    # ==================== 持久化 ====================
    
    def save(self) -> bool:
        """保存所有记忆"""
        with self._lock:
            try:
                self.episodic._save()
                self.semantic._save()
                self.procedural._save()
                self.store.save_all()
                return True
            except Exception as e:
                print(f"Save failed: {e}")
                return False
    
    def create_backup(self, name: Optional[str] = None) -> str:
        """创建备份"""
        return self.store.create_backup(name)
    
    def close(self) -> None:
        """关闭记忆系统"""
        self.save()
        self.consolidation.stop_auto_consolidation()
        self.store.close()
    
    # ==================== 便捷方法 ====================
    
    def learn_from_interaction(
        self,
        interaction_type: str,
        summary: str,
        context: Dict[str, Any],
        actions: List[Dict[str, Any]],
        outcome: Dict[str, Any],
        success: bool,
        entities: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> Episode:
        """
        从交互中学习的便捷方法
        
        Args:
            interaction_type: 交互类型 (抓取/导航/对话/...)
            summary: 总结
            context: 上下文
            actions: 执行的动作
            outcome: 结果
            success: 是否成功
            entities: 涉及的实体
            tags: 标签
            
        Returns:
            创建的情景记忆
        """
        # 计算重要性
        importance = 5.0
        if success:
            importance = 7.0
        if not success and outcome.get('critical_failure'):
            importance = 9.0
        
        # 确定情感标签
        emotional = "neutral"
        if success and outcome.get('exceeded_expectations'):
            emotional = "very_positive"
        elif success:
            emotional = "positive"
        elif not success:
            emotional = "negative"
        
        # 添加标签
        all_tags = [interaction_type]
        if tags:
            all_tags.extend(tags)
        if success:
            all_tags.append("成功")
        else:
            all_tags.append("失败")
        
        # 存储记忆
        episode = self.store_episode(
            summary=summary,
            context=context,
            actions=actions,
            outcomes=outcome,
            emotional_tag=emotional,
            importance_score=importance,
            entities=entities,
            tags=all_tags,
            lessons_learned=outcome.get('lessons', []),
        )
        
        # 如果成功，提取技能
        if success and len(actions) >= 3:
            self.store_skill(
                name=f"{interaction_type}_模式",
                description=summary,
                category=interaction_type,
                steps=actions[:5],  # 最多5步
                contexts=all_tags,
                tags=["extracted_from_experience"],
            )
        
        return episode
    
    def __repr__(self) -> str:
        return f"LongTermMemory(episodes={len(self.episodic)}, concepts={len(self.semantic)}, skills={len(self.procedural)})"
