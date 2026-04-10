"""
Semantic Memory - 语义记忆模块
==============================

存储和组织知识、概念、事实。

语义记忆特点:
- 概念中心: 以概念/实体为中心组织
- 关系网络: 概念之间的关系
- 置信度: 知识的可信程度
- 来源追溯: 知识的来源
- 可更新性: 知识修正
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
from enum import Enum
import json
import uuid
import time


class KnowledgeSource(Enum):
    """知识来源"""
    DIRECT_EXPERIENCE = "direct"    # 直接经验
    OBSERVATION = "observation"     # 观察学习
    INSTRUCTION = "instruction"     # 指令/教导
    INFERENCE = "inference"          # 推理得出
    EXTERNAL = "external"           # 外部来源


class ConfidenceLevel(Enum):
    """置信度等级"""
    CERTAIN = 1.0      # 确定
    HIGH = 0.9         # 高置信
    MODERATE = 0.7     # 中等置信
    LOW = 0.5          # 低置信
    SPECULATIVE = 0.3  # 推测


@dataclass
class Concept:
    """
    概念/实体单元
    
    Attributes:
        id: 唯一标识符
        name: 概念名称
        category: 概念类别
        properties: 属性字典
        relations: 关系字典 {related_concept_id: relation_type}
        description: 描述
        examples: 示例
        confidence: 置信度 [0, 1]
        source: 知识来源
        source_episode_id: 来源记忆ID
        created_at: 创建时间
        updated_at: 更新时间
        access_count: 访问次数
        last_accessed: 上次访问时间
        aliases: 别名/同义词
        tags: 标签
        applicability_scope: 适用场景/条件
        exceptions: 例外情况
        verision: 版本号
    """
    id: str
    name: str
    category: str
    properties: Dict[str, Any] = field(default_factory=dict)
    relations: Dict[str, str] = field(default_factory=dict)  # concept_id -> relation_type
    description: str = ""
    examples: List[str] = field(default_factory=list)
    confidence: float = 0.8
    source: KnowledgeSource = KnowledgeSource.DIRECT_EXPERIENCE
    source_episode_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    aliases: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    applicability_scope: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category,
            'properties': self.properties,
            'relations': self.relations,
            'description': self.description,
            'examples': self.examples,
            'confidence': self.confidence,
            'source': self.source.value,
            'source_episode_id': self.source_episode_id,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'aliases': self.aliases,
            'tags': self.tags,
            'applicability_scope': self.applicability_scope,
            'exceptions': self.exceptions,
            'version': self.version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Concept:
        return cls(
            id=data['id'],
            name=data['name'],
            category=data['category'],
            properties=data.get('properties', {}),
            relations=data.get('relations', {}),
            description=data.get('description', ''),
            examples=data.get('examples', []),
            confidence=data.get('confidence', 0.8),
            source=KnowledgeSource(data.get('source', 'direct')),
            source_episode_id=data.get('source_episode_id'),
            created_at=data.get('created_at', time.time()),
            updated_at=data.get('updated_at', time.time()),
            access_count=data.get('access_count', 0),
            last_accessed=data.get('last_accessed', time.time()),
            aliases=data.get('aliases', []),
            tags=data.get('tags', []),
            applicability_scope=data.get('applicability_scope', []),
            exceptions=data.get('exceptions', []),
            version=data.get('version', 1),
        )


@dataclass
class Fact:
    """
    事实/陈述单元
    
    Attributes:
        id: 唯一标识符
        subject: 主语概念ID
        predicate: 谓词/关系
        object: 宾语概念ID (可选)
        object_value: 宾语值 (非概念时使用)
        confidence: 置信度
        context: 适用上下文
        source: 来源
        source_episode_id: 来源记忆ID
        created_at: 创建时间
        is_active: 是否激活
    """
    id: str
    subject_id: str
    predicate: str
    object_id: Optional[str] = None
    object_value: Optional[Any] = None
    confidence: float = 0.8
    context: str = ""  # 适用场景
    source: KnowledgeSource = KnowledgeSource.DIRECT_EXPERIENCE
    source_episode_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'subject_id': self.subject_id,
            'predicate': self.predicate,
            'object_id': self.object_id,
            'object_value': self.object_value,
            'confidence': self.confidence,
            'context': self.context,
            'source': self.source.value,
            'source_episode_id': self.source_episode_id,
            'created_at': self.created_at,
            'is_active': self.is_active,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Fact:
        return cls(
            id=data['id'],
            subject_id=data['subject_id'],
            predicate=data['predicate'],
            object_id=data.get('object_id'),
            object_value=data.get('object_value'),
            confidence=data.get('confidence', 0.8),
            context=data.get('context', ''),
            source=KnowledgeSource(data.get('source', 'direct')),
            source_episode_id=data.get('source_episode_id'),
            created_at=data.get('created_at', time.time()),
            is_active=data.get('is_active', True),
        )


@dataclass 
class Rule:
    """
    规则/因果关系
    
    Attributes:
        id: 唯一标识符
        if_conditions: IF条件列表
        then_conclusion: THEN结论
        confidence: 置信度
        source_episode_id: 来源记忆ID
        usage_count: 使用次数
        last_used: 上次使用时间
        applicability: 适用条件
        exceptions: 例外情况
    """
    id: str
    if_conditions: List[str] = field(default_factory=list)
    then_conclusion: str = ""
    confidence: float = 0.7
    source_episode_id: Optional[str] = None
    usage_count: int = 0
    last_used: float = field(default_factory=time.time)
    applicability: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'if_conditions': self.if_conditions,
            'then_conclusion': self.then_conclusion,
            'confidence': self.confidence,
            'source_episode_id': self.source_episode_id,
            'usage_count': self.usage_count,
            'last_used': self.last_used,
            'applicability': self.applicability,
            'exceptions': self.exceptions,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Rule:
        return cls(
            id=data['id'],
            if_conditions=data.get('if_conditions', []),
            then_conclusion=data.get('then_conclusion', ''),
            confidence=data.get('confidence', 0.7),
            source_episode_id=data.get('source_episode_id'),
            usage_count=data.get('usage_count', 0),
            last_used=data.get('last_used', time.time()),
            applicability=data.get('applicability', []),
            exceptions=data.get('exceptions', []),
        )


class SemanticMemory:
    """
    语义记忆管理器
    
    负责:
    - 概念存储和检索
    - 事实存储和验证
    - 规则学习和管理
    - 知识网络构建
    - 知识推理
    """
    
    def __init__(
        self,
        store_path: Optional[str] = None,
        embedding_dim: int = 128,
    ):
        """
        Args:
            store_path: 存储路径
            embedding_dim: 嵌入维度
        """
        self.store_path = store_path
        self.embedding_dim = embedding_dim
        
        # 存储
        self._concepts: Dict[str, Concept] = {}
        self._concepts_by_name: Dict[str, str] = {}  # name -> id (快速查找)
        self._concepts_by_category: Dict[str, List[str]] = {}  # category -> [concept_ids]
        self._facts: Dict[str, Fact] = {}
        self._rules: Dict[str, Rule] = {}
        
        # 索引
        self._concept_embeddings: Dict[str, np.ndarray] = {}
        
        # 统计
        self._total_concepts = 0
        self._total_facts = 0
        self._total_rules = 0
        
        if store_path:
            self._load()
    
    # ==================== 概念管理 ====================
    
    def add_concept(
        self,
        name: str,
        category: str,
        properties: Optional[Dict[str, Any]] = None,
        description: str = "",
        confidence: float = 0.8,
        source: KnowledgeSource = KnowledgeSource.DIRECT_EXPERIENCE,
        source_episode_id: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> Concept:
        """
        添加新概念
        
        Returns:
            创建的Concept对象
        """
        # 检查是否已存在
        if name in self._concepts_by_name:
            concept = self._concepts[self._concepts_by_name[name]]
            return self.update_concept(concept.id, properties=properties)
        
        concept_id = str(uuid.uuid4())
        
        concept = Concept(
            id=concept_id,
            name=name,
            category=category,
            properties=properties or {},
            description=description,
            confidence=confidence,
            source=source,
            source_episode_id=source_episode_id,
            aliases=aliases or [],
            tags=tags or [],
        )
        
        self._add_concept(concept)
        
        return concept
    
    def _add_concept(self, concept: Concept) -> None:
        """内部: 添加概念"""
        self._concepts[concept.id] = concept
        self._concepts_by_name[concept.name] = concept.id
        
        # 按类别索引
        if concept.category not in self._concepts_by_category:
            self._concepts_by_category[concept.category] = []
        self._concepts_by_category[concept.category].append(concept.id)
        
        # 别名索引
        for alias in concept.aliases:
            self._concepts_by_name[alias.lower()] = concept.id
        
        self._total_concepts += 1
        
        if self.store_path:
            self._save()
    
    def update_concept(
        self,
        concept_id: str,
        properties: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        confidence: Optional[float] = None,
        add_relations: Optional[Dict[str, str]] = None,
        add_tags: Optional[List[str]] = None,
        add_examples: Optional[List[str]] = None,
    ) -> Optional[Concept]:
        """更新概念"""
        concept = self._concepts.get(concept_id)
        if not concept:
            return None
        
        if properties:
            concept.properties.update(properties)
        if description is not None:
            concept.description = description
        if confidence is not None:
            concept.confidence = confidence
        if add_relations:
            concept.relations.update(add_relations)
        if add_tags:
            concept.tags.extend([t for t in add_tags if t not in concept.tags])
        if add_examples:
            concept.examples.extend([e for e in add_examples if e not in concept.examples])
        
        concept.updated_at = time.time()
        concept.version += 1
        
        if self.store_path:
            self._save()
        
        return concept
    
    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """获取概念"""
        concept = self._concepts.get(concept_id)
        if concept:
            concept.access_count += 1
            concept.last_accessed = time.time()
        return concept
    
    def find_concept_by_name(self, name: str) -> Optional[Concept]:
        """按名称查找概念"""
        concept_id = self._concepts_by_name.get(name.lower())
        if concept_id:
            return self.get_concept(concept_id)
        return None
    
    def get_concepts_by_category(self, category: str) -> List[Concept]:
        """获取类别的所有概念"""
        concept_ids = self._concepts_by_category.get(category, [])
        return [self.get_concept(cid) for cid in concept_ids if cid in self._concepts]
    
    def get_related_concepts(self, concept_id: str) -> List[Tuple[Concept, str]]:
        """获取相关概念"""
        concept = self._concepts.get(concept_id)
        if not concept:
            return []
        
        related = []
        for rel_id, rel_type in concept.relations.items():
            rel_concept = self._concepts.get(rel_id)
            if rel_concept:
                related.append((rel_concept, rel_type))
        
        return related
    
    # ==================== 事实管理 ====================
    
    def add_fact(
        self,
        subject_id: str,
        predicate: str,
        object_id: Optional[str] = None,
        object_value: Optional[Any] = None,
        confidence: float = 0.8,
        context: str = "",
        source: KnowledgeSource = KnowledgeSource.DIRECT_EXPERIENCE,
        source_episode_id: Optional[str] = None,
    ) -> Optional[Fact]:
        """添加事实"""
        # 验证subject存在
        if subject_id not in self._concepts:
            return None
        
        fact_id = str(uuid.uuid4())
        
        fact = Fact(
            id=fact_id,
            subject_id=subject_id,
            predicate=predicate,
            object_id=object_id,
            object_value=object_value,
            confidence=confidence,
            context=context,
            source=source,
            source_episode_id=source_episode_id,
        )
        
        self._facts[fact_id] = fact
        self._total_facts += 1
        
        if self.store_path:
            self._save()
        
        return fact
    
    def get_facts_about(self, subject_id: str) -> List[Fact]:
        """获取关于某概念的所有事实"""
        return [f for f in self._facts.values() if f.subject_id == subject_id and f.is_active]
    
    def update_fact_confidence(self, fact_id: str, delta: float) -> bool:
        """更新事实置信度"""
        fact = self._facts.get(fact_id)
        if fact:
            fact.confidence = max(0.0, min(1.0, fact.confidence + delta))
            return True
        return False
    
    def deactivate_fact(self, fact_id: str) -> bool:
        """停用事实 (软删除)"""
        fact = self._facts.get(fact_id)
        if fact:
            fact.is_active = False
            if self.store_path:
                self._save()
            return True
        return False
    
    # ==================== 规则管理 ====================
    
    def add_rule(
        self,
        if_conditions: List[str],
        then_conclusion: str,
        confidence: float = 0.7,
        source_episode_id: Optional[str] = None,
        applicability: Optional[List[str]] = None,
    ) -> Rule:
        """添加规则"""
        rule_id = str(uuid.uuid4())
        
        rule = Rule(
            id=rule_id,
            if_conditions=if_conditions,
            then_conclusion=then_conclusion,
            confidence=confidence,
            source_episode_id=source_episode_id,
            applicability=applicability or [],
        )
        
        self._rules[rule_id] = rule
        self._total_rules += 1
        
        return rule
    
    def apply_rules(
        self,
        conditions: Set[str],
        context: Optional[str] = None,
    ) -> List[Tuple[Rule, float]]:
        """
        应用规则进行推理
        
        Args:
            conditions: 当前条件集合
            context: 当前上下文
            
        Returns:
            [(匹配的规则, 置信度)] 列表
        """
        matches = []
        
        for rule in self._rules.values():
            # 检查条件是否满足
            if all(cond in conditions for cond in rule.if_conditions):
                # 检查适用性
                if context and rule.applicability:
                    if context not in rule.applicability:
                        continue
                
                rule.usage_count += 1
                rule.last_used = time.time()
                
                # 计算综合置信度
                effective_confidence = rule.confidence
                matches.append((rule, effective_confidence))
        
        # 按置信度排序
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    # ==================== 知识整合 ====================
    
    def integrate_from_episode(
        self,
        episode_id: str,
        summary: str,
        lessons: List[str],
        context: Dict[str, Any],
    ) -> List[str]:
        """
        从情景记忆整合知识到语义记忆
        
        Args:
            episode_id: 情景记忆ID
            summary: 经验摘要
            lessons: 经验教训
            context: 上下文
            
        Returns:
            创建的概念/规则ID列表
        """
        created_ids = []
        
        # 从上下文中提取实体
        entities = context.get('entities', [])
        for entity_name in entities:
            concept = self.add_concept(
                name=entity_name,
                category='extracted_entity',
                description=f"从经验 {episode_id} 中提取",
                source=KnowledgeSource.DIRECT_EXPERIENCE,
                source_episode_id=episode_id,
            )
            created_ids.append(concept.id)
        
        # 从教训中提取规则
        for lesson in lessons:
            # 简单处理: 假设 lessons 包含 "如果...那么..." 格式
            if '如果' in lesson and '那么' in lesson:
                parts = lesson.split('那么')
                if_parts = parts[0].replace('如果', '').split('和')
                if_parts = [p.strip() for p in if_parts if p.strip()]
                
                rule = self.add_rule(
                    if_conditions=if_parts,
                    then_conclusion=parts[1].strip() if len(parts) > 1 else '',
                    confidence=0.7,
                    source_episode_id=episode_id,
                )
                created_ids.append(rule.id)
        
        return created_ids
    
    # ==================== 检索 ====================
    
    def search_concepts(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 20,
    ) -> List[Concept]:
        """
        搜索概念
        
        Args:
            query: 查询字符串
            category: 限定类别
            limit: 返回数量
            
        Returns:
            匹配的概念列表
        """
        query_lower = query.lower()
        results = []
        
        for concept in self._concepts.values():
            if category and concept.category != category:
                continue
            
            # 名称匹配
            if query_lower in concept.name.lower():
                results.append(concept)
                continue
            
            # 别名匹配
            if any(query_lower in alias.lower() for alias in concept.aliases):
                results.append(concept)
                continue
            
            # 标签匹配
            if any(query_lower in tag.lower() for tag in concept.tags):
                results.append(concept)
                continue
        
        # 按访问次数和置信度排序
        results.sort(
            key=lambda c: (c.access_count * 0.3 + c.confidence * 0.7),
            reverse=True
        )
        
        return results[:limit]
    
    def get_knowledge_network(
        self,
        concept_id: str,
        depth: int = 2,
    ) -> Dict[str, Any]:
        """
        获取概念的知识网络
        
        Args:
            concept_id: 中心概念ID
            depth: 探索深度
            
        Returns:
            知识网络结构
        """
        network = {
            'center': None,
            'nodes': {},
            'edges': [],
        }
        
        center = self._concepts.get(concept_id)
        if not center:
            return network
        
        network['center'] = center.to_dict()
        visited = {concept_id}
        
        def explore(cid: str, current_depth: int):
            if current_depth >= depth:
                return
            
            concept = self._concepts.get(cid)
            if not concept:
                return
            
            # 遍历关系
            for rel_id, rel_type in concept.relations.items():
                if rel_id in visited:
                    continue
                    
                visited.add(rel_id)
                network['edges'].append({
                    'from': cid,
                    'to': rel_id,
                    'type': rel_type,
                })
                
                rel_concept = self._concepts.get(rel_id)
                if rel_concept:
                    network['nodes'][rel_id] = rel_concept.to_dict()
                
                explore(rel_id, current_depth + 1)
        
        network['nodes'][concept_id] = center.to_dict()
        explore(concept_id, 0)
        
        return network
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_concepts': len(self._concepts),
            'total_facts': len(self._facts),
            'total_rules': len(self._rules),
            'categories': {
                cat: len(ids)
                for cat, ids in self._concepts_by_category.items()
            },
            'avg_confidence': np.mean([c.confidence for c in self._concepts.values()]) if self._concepts else 0.0,
            'by_source': {
                src.value: sum(1 for c in self._concepts.values() if c.source.value == src.value)
                for src in KnowledgeSource
            },
        }
    
    # ==================== 持久化 ====================
    
    def _save(self) -> None:
        """保存到磁盘"""
        try:
            data = {
                'concepts': {cid: c.to_dict() for cid, c in self._concepts.items()},
                'facts': {fid: f.to_dict() for fid, f in self._facts.items()},
                'rules': {rid: r.to_dict() for rid, r in self._rules.items()},
                'total_concepts': self._total_concepts,
                'total_facts': self._total_facts,
                'total_rules': self._total_rules,
            }
            
            path = f"{self.store_path}/semantic_memory.json"
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save semantic memory: {e}")
    
    def _load(self) -> None:
        """从磁盘加载"""
        try:
            path = f"{self.store_path}/semantic_memory.json"
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 加载概念
            self._concepts = {
                cid: Concept.from_dict(cdata)
                for cid, cdata in data.get('concepts', {}).items()
            }
            
            # 重建索引
            for concept in self._concepts.values():
                self._concepts_by_name[concept.name] = concept.id
                for alias in concept.aliases:
                    self._concepts_by_name[alias.lower()] = concept.id
                
                if concept.category not in self._concepts_by_category:
                    self._concepts_by_category[concept.category] = []
                self._concepts_by_category[concept.category].append(concept.id)
            
            # 加载事实
            self._facts = {
                fid: Fact.from_dict(fdata)
                for fid, fdata in data.get('facts', {}).items()
            }
            
            # 加载规则
            self._rules = {
                rid: Rule.from_dict(rdata)
                for rid, rdata in data.get('rules', {}).items()
            }
            
            self._total_concepts = data.get('total_concepts', len(self._concepts))
            self._total_facts = data.get('total_facts', len(self._facts))
            self._total_rules = data.get('total_rules', len(self._rules))
            
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Failed to load semantic memory: {e}")
    
    def __len__(self) -> int:
        return len(self._concepts)
