"""
Long Term Memory System - 长期记忆系统
支持三种记忆类型：
1. 情景记忆 (Episodic Memory)：存储过往经历、事件、场景
2. 语义记忆 (Semantic Memory)：存储知识、事实、概念
3. 程序记忆 (Procedural Memory)：存储技能、操作流程、行为模式

特性：
- 向量存储 + 相似度检索
- 自动记忆重要性评分
- 记忆遗忘机制（低重要性记忆自动过期）
- 持久化到磁盘
- 多模态支持（文本/图像/传感器数据）
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import os
import numpy as np
from pathlib import Path


class MemoryType(Enum):
    """记忆类型"""
    EPISODIC = "episodic"    # 情景记忆
    SEMANTIC = "semantic"    # 语义记忆
    PROCEDURAL = "procedural"  # 程序记忆


@dataclass
class MemoryConfig:
    """记忆系统配置"""
    storage_path: str = "memory_data/long_term"
    vector_dim: int = 1536
    forget_threshold_days: int = 30
    default_importance: float = 0.5


@dataclass
class MemoryEntry:
    """记忆条目"""
    memory_id: str
    memory_type: MemoryType
    content: Any  # 记忆内容，可以是文本、图像、结构化数据等
    embedding: Optional[np.ndarray] = None  # 向量嵌入
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5  # 重要性评分 0-1，越高越不容易被遗忘
    access_count: int = 0  # 访问次数，访问越多越重要
    last_access_time: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)  # 标签，用于分类检索
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

    def to_dict(self) -> Dict:
        """转换为字典用于序列化"""
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_access_time": self.last_access_time,
            "tags": self.tags,
            "metadata": self.metadata
        }

    @property
    def summary(self) -> str:
        """获取记忆摘要（便捷属性）"""
        if isinstance(self.content, dict):
            return self.content.get("summary", "") or self.content.get("name", "")
        return str(self.content)[:200]

    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryEntry':
        """从字典反序列化"""
        embedding = np.array(data["embedding"]) if data["embedding"] is not None else None
        return cls(
            memory_id=data["memory_id"],
            memory_type=MemoryType(data["memory_type"]),
            content=data["content"],
            embedding=embedding,
            timestamp=data["timestamp"],
            importance=data["importance"],
            access_count=data["access_count"],
            last_access_time=data["last_access_time"],
            tags=data["tags"],
            metadata=data["metadata"]
        )


class LongTermMemory:
    """
    长期记忆系统
    """

    def __init__(self, storage_path: str = "memory_data/long_term", vector_dim: int = 1536, forget_threshold_days: int = 30):
        self.storage_path = Path(storage_path)
        self.vector_dim = vector_dim
        self.forget_threshold_days = forget_threshold_days  # 记忆过期时间，超过这个时间且重要性<0.3的记忆会被遗忘

        # 确保存储目录存在
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 内存中的记忆索引
        self.memories: Dict[str, MemoryEntry] = {}
        self.vector_index: Dict[str, np.ndarray] = {}  # memory_id -> embedding

        # 加载已有的记忆
        self._load_memories()

    def _load_memories(self):
        """从磁盘加载记忆"""
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    memory = MemoryEntry.from_dict(data)
                    self.memories[memory.memory_id] = memory
                    if memory.embedding is not None:
                        self.vector_index[memory.memory_id] = memory.embedding
            except Exception as e:
                print(f"加载记忆失败 {file_path}: {e}")

        # 执行遗忘机制
        self._forget_old_memories()

    def _save_memory(self, memory: MemoryEntry):
        """保存记忆到磁盘"""
        file_path = self.storage_path / f"{memory.memory_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(memory.to_dict(), f, ensure_ascii=False, indent=2)

    def _delete_memory(self, memory_id: str):
        """删除记忆"""
        if memory_id in self.memories:
            del self.memories[memory_id]
        if memory_id in self.vector_index:
            del self.vector_index[memory_id]
        file_path = self.storage_path / f"{memory_id}.json"
        if file_path.exists():
            file_path.unlink()

    def _forget_old_memories(self):
        """遗忘机制：删除过期且低重要性的记忆"""
        current_time = time.time()
        threshold_time = current_time - self.forget_threshold_days * 86400

        to_delete = []
        for memory_id, memory in self.memories.items():
            if memory.timestamp < threshold_time and memory.importance < 0.3:
                to_delete.append(memory_id)

        for memory_id in to_delete:
            self._delete_memory(memory_id)

    def add_memory(
        self,
        memory_type: MemoryType,
        content: Any,
        embedding: Optional[np.ndarray] = None,
        importance: float = 0.5,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        添加记忆
        返回记忆ID
        """
        memory_id = f"mem_{int(time.time() * 1000)}_{np.random.randint(0, 1000)}"

        # 验证embedding维度
        if embedding is not None:
            if embedding.shape[0] != self.vector_dim:
                raise ValueError(f"Embedding维度错误，期望{self.vector_dim}，实际{embedding.shape[0]}")

        memory = MemoryEntry(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            embedding=embedding,
            importance=max(0.0, min(1.0, importance)),  # 限制在0-1之间
            tags=tags or [],
            metadata=metadata or {}
        )

        self.memories[memory_id] = memory
        if embedding is not None:
            self.vector_index[memory_id] = embedding

        # 保存到磁盘
        self._save_memory(memory)

        return memory_id

    def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """根据ID获取记忆"""
        memory = self.memories.get(memory_id)
        if memory:
            # 更新访问统计
            memory.access_count += 1
            memory.last_access_time = time.time()
            self._save_memory(memory)
        return memory

    def search_by_similarity(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        memory_type: Optional[MemoryType] = None,
        tags: Optional[List[str]] = None,
        threshold: float = 0.7
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        按向量相似度检索记忆
        返回 (记忆条目, 相似度得分) 的列表，按得分降序排列
        """
        if not self.vector_index:
            return []

        # 验证查询向量维度
        if query_embedding.shape[0] != self.vector_dim:
            raise ValueError(f"查询向量维度错误，期望{self.vector_dim}，实际{query_embedding.shape[0]}")

        # 计算余弦相似度
        similarities = []
        for memory_id, embedding in self.vector_index.items():
            memory = self.memories[memory_id]

            # 过滤记忆类型
            if memory_type is not None and memory.memory_type != memory_type:
                continue

            # 过滤标签
            if tags is not None and not all(tag in memory.tags for tag in tags):
                continue

            # 计算余弦相似度
            dot_product = np.dot(query_embedding, embedding)
            norm_product = np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            if norm_product == 0:
                similarity = 0.0
            else:
                similarity = dot_product / norm_product

            if similarity >= threshold:
                similarities.append((memory, similarity))

        # 按相似度降序排列，取top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = similarities[:top_k]

        # 更新访问统计
        for memory, _ in results:
            memory.access_count += 1
            memory.last_access_time = time.time()
            self._save_memory(memory)

        return results

    def search_by_tags(self, tags: List[str], memory_type: Optional[MemoryType] = None, top_k: int = 100) -> List[MemoryEntry]:
        """按标签检索记忆"""
        results = []
        for memory in self.memories.values():
            if memory_type is not None and memory.memory_type != memory_type:
                continue
            if all(tag in memory.tags for tag in tags):
                results.append(memory)

        # 按重要性和访问次数排序
        results.sort(key=lambda x: (x.importance, x.access_count), reverse=True)
        return results[:top_k]

    def search_by_time_range(self, start_time: float, end_time: float, memory_type: Optional[MemoryType] = None) -> List[MemoryEntry]:
        """按时间范围检索记忆"""
        results = []
        for memory in self.memories.values():
            if memory_type is not None and memory.memory_type != memory_type:
                continue
            if start_time <= memory.timestamp <= end_time:
                results.append(memory)

        # 按时间降序排列
        results.sort(key=lambda x: x.timestamp, reverse=True)
        return results

    def update_memory_importance(self, memory_id: str, new_importance: float):
        """更新记忆的重要性评分"""
        if memory_id not in self.memories:
            return
        memory = self.memories[memory_id]
        memory.importance = max(0.0, min(1.0, new_importance))
        self._save_memory(memory)

    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        if memory_id in self.memories:
            self._delete_memory(memory_id)
            return True
        return False

    def get_memory_stats(self) -> Dict:
        """获取记忆统计信息"""
        total = len(self.memories)
        episodic = sum(1 for m in self.memories.values() if m.memory_type == MemoryType.EPISODIC)
        semantic = sum(1 for m in self.memories.values() if m.memory_type == MemoryType.SEMANTIC)
        procedural = sum(1 for m in self.memories.values() if m.memory_type == MemoryType.PROCEDURAL)
        has_embedding = len(self.vector_index)

        return {
            "total_memories": total,
            "by_type": {
                "episodic": episodic,
                "semantic": semantic,
                "procedural": procedural
            },
            "has_embedding": has_embedding,
            "storage_path": str(self.storage_path.absolute())
        }

    def clear_all_memories(self):
        """清空所有记忆（谨慎使用）"""
        for memory_id in list(self.memories.keys()):
            self._delete_memory(memory_id)

    # ==================== 便捷方法 ====================

    def store_episode(
        self,
        summary: str,
        context: Dict[str, Any] = None,
        importance_score: float = 0.5,
        tags: List[str] = None,
    ) -> MemoryEntry:
        """存储情景记忆（便捷方法）"""
        content = {
            "summary": summary,
            "context": context or {},
            "type": "episode",
        }
        memory_id = self.add_memory(
            memory_type=MemoryType.EPISODIC,
            content=content,
            importance=importance_score,
            tags=tags or [],
        )
        return self.get_memory(memory_id)

    def store_knowledge(
        self,
        name: str,
        category: str,
        description: str = "",
    ) -> MemoryEntry:
        """存储语义记忆/知识（便捷方法）"""
        content = {
            "name": name,
            "category": category,
            "description": description,
            "type": "knowledge",
        }
        memory_id = self.add_memory(
            memory_type=MemoryType.SEMANTIC,
            content=content,
            importance=0.6,
            tags=[category],
        )
        return self.get_memory(memory_id)

    def store_skill(
        self,
        name: str,
        description: str = "",
        category: str = "general",
    ) -> MemoryEntry:
        """存储程序记忆/技能（便捷方法）"""
        content = {
            "name": name,
            "description": description,
            "category": category,
            "type": "skill",
        }
        memory_id = self.add_memory(
            memory_type=MemoryType.PROCEDURAL,
            content=content,
            importance=0.7,
            tags=[category],
        )
        return self.get_memory(memory_id)

    def retrieve(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """统一检索接口（按关键词简单过滤）"""
        query_lower = query.lower()
        results = []
        for memory in self.memories.values():
            content_str = str(memory.content).lower()
            if query_lower in content_str:
                results.append(memory)
            if len(results) >= limit:
                break
        return results

    def learn_from_interaction(
        self,
        interaction_type: str,
        summary: str,
        context: Dict[str, Any] = None,
        actions: List[Dict[str, Any]] = None,
        outcome: Dict[str, Any] = None,
        success: bool = True,
        tags: List[str] = None,
    ) -> MemoryEntry:
        """从交互中学习（存储情景记忆的快捷方法）"""
        content = {
            "interaction_type": interaction_type,
            "summary": summary,
            "context": context or {},
            "actions": actions or [],
            "outcome": outcome or {},
            "success": success,
            "type": "interaction",
        }
        importance = 0.8 if success else 0.9
        memory_id = self.add_memory(
            memory_type=MemoryType.EPISODIC,
            content=content,
            importance=importance,
            tags=tags or [interaction_type],
        )
        return self.get_memory(memory_id)

    def get_status(self) -> Dict[str, Any]:
        """获取记忆系统状态（包含各子类型统计）"""
        stats = self.get_memory_stats()
        episodic_memories = [m for m in self.memories.values() if m.memory_type == MemoryType.EPISODIC]
        semantic_memories = [m for m in self.memories.values() if m.memory_type == MemoryType.SEMANTIC]
        procedural_memories = [m for m in self.memories.values() if m.memory_type == MemoryType.PROCEDURAL]
        return {
            "episodic": {
                "count": len(episodic_memories),
                "recent": [m.content.get("summary", str(m.content)) for m in list(episodic_memories)[-3:]],
            },
            "semantic": {
                "count": len(semantic_memories),
                "recent": [m.content.get("name", str(m.content)) for m in list(semantic_memories)[-3:]],
            },
            "procedural": {
                "count": len(procedural_memories),
                "recent": [m.content.get("name", str(m.content)) for m in list(procedural_memories)[-3:]],
            },
            "total": stats["total_memories"],
            "storage_path": stats["storage_path"],
        }

    def get_memory_summary(self) -> str:
        """获取记忆摘要（可读格式）"""
        episodic = sum(1 for m in self.memories.values() if m.memory_type == MemoryType.EPISODIC)
        semantic = sum(1 for m in self.memories.values() if m.memory_type == MemoryType.SEMANTIC)
        procedural = sum(1 for m in self.memories.values() if m.memory_type == MemoryType.PROCEDURAL)
        total = len(self.memories)
        return (
            f"记忆系统摘要: 共{total}条记忆\n"
            f"- 情景记忆: {episodic}条\n"
            f"- 语义记忆: {semantic}条\n"
            f"- 程序记忆: {procedural}条"
        )

    def get_working_summary(self) -> Dict[str, Any]:
        """获取工作记忆状态摘要（与长期记忆配合使用）"""
        return {
            "total_memories": len(self.memories),
            "recent_memories": [
                {"id": m.memory_id, "type": m.memory_type.value, "summary": m.summary}
                for m in list(self.memories.values())[-5:]
            ],
            "timestamp": time.time(),
        }

    def consolidate(self) -> Dict[str, Any]:
        """执行记忆整合/巩固（调用遗忘机制）"""
        before_count = len(self.memories)
        self._forget_old_memories()
        after_count = len(self.memories)
        return {
            "before_count": before_count,
            "after_count": after_count,
            "consolidated_count": before_count - after_count,
            "timestamp": time.time(),
        }

    def close(self):
        """关闭记忆系统（持久化所有内存中的记忆）"""
        for memory in self.memories.values():
            self._save_memory(memory)
