"""
Working Memory - 工作记忆模块
=============================

短期记忆模块，管理当前上下文和正在处理的信息。

工作记忆特点:
- 容量限制: 有限的注意力资源
- 快速访问: 当前焦点信息
- 层级组织: 主空间 + 子空间
- 自动衰减: 不活跃信息逐渐淡出
- 与长期记忆交互: 提取和存储
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import deque
import time
import threading


@dataclass
class FocusItem:
    """
    注意力焦点项
    
    Attributes:
        content: 内容
        attention_level: 注意力等级 [0, 1]
        created_at: 创建时间
        last_accessed: 上次访问时间
        source: 来源 (perception, long_term, reasoning)
        importance: 重要性 [0, 10]
        decay_rate: 衰减率
    """
    content: Any
    attention_level: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    source: str = "perception"
    importance: float = 5.0
    decay_rate: float = 0.01
    
    def update_access(self, boost: float = 0.1) -> None:
        """更新访问时间并增强注意力"""
        self.last_accessed = time.time()
        self.attention_level = min(1.0, self.attention_level + boost)


@dataclass
class WorkingMemoryConfig:
    """工作记忆配置"""
    max_focus_items: int = 7           # 最大焦点数量 (Miller's Law)
    max_buffer_items: int = 20        # 最大缓冲区数量
    attention_threshold: float = 0.1   # 注意力阈值，低于此值被移除
    decay_base_rate: float = 0.001     # 基础衰减率
    default_decay_rate: float = 0.01   # 默认衰减率


class WorkingMemory:
    """
    工作记忆管理器
    
    负责:
    - 当前信息的存储和访问
    - 注意力焦点管理
    - 信息的衰减和遗忘
    - 与长期记忆的交互
    - 推理过程的中间状态
    """
    
    def __init__(
        self,
        config: Optional[WorkingMemoryConfig] = None,
    ):
        self.config = config or WorkingMemoryConfig()
        
        self._lock = threading.RLock()
        
        # 主焦点区域 (当前最关注的信息)
        self._focus_items: Dict[str, FocusItem] = {}
        
        # 缓冲区 (次要关注的信息)
        self._buffer_items: deque = deque(maxlen=self.config.max_buffer_items)
        
        # 当前任务上下文
        self._current_task: Optional[Dict[str, Any]] = None
        
        # 推理栈
        self._reasoning_stack: List[Dict[str, Any]] = []
        
        # 临时绑定 (变量 -> 值)
        self._bindings: Dict[str, Any] = {}
        
        # 激活模式 (当前活跃的概念/实体)
        self._activation_pattern: Set[str] = set()
        
        # 统计
        self._total_items_processed = 0
        self._decay_operations = 0
    
    # ==================== 焦点管理 ====================
    
    def focus(
        self,
        key: str,
        content: Any,
        importance: float = 5.0,
        source: str = "perception",
        attention_level: float = 1.0,
    ) -> None:
        """
        将信息放入焦点区域
        
        Args:
            key: 唯一键
            content: 内容
            importance: 重要性 [0, 10]
            source: 来源
            attention_level: 初始注意力等级
        """
        with self._lock:
            # 如果已存在，更新
            if key in self._focus_items:
                item = self._focus_items[key]
                item.content = content
                item.importance = importance
                item.update_access(boost=0.2)
            else:
                # 添加新项
                item = FocusItem(
                    content=content,
                    importance=importance,
                    source=source,
                    attention_level=attention_level,
                )
                self._focus_items[key] = item
            
            # 容量管理
            self._manage_focus_capacity()
            
            self._total_items_processed += 1
    
    def get_focused(self, key: str) -> Optional[Any]:
        """
        获取焦点信息
        
        Returns:
            内容或None
        """
        with self._lock:
            item = self._focus_items.get(key)
            if item:
                item.update_access()
                return item.content
            return None
    
    def get_focus_summary(self) -> Dict[str, Any]:
        """获取焦点区域摘要"""
        with self._lock:
            items = []
            for key, item in self._focus_items.items():
                items.append({
                    'key': key,
                    'attention_level': item.attention_level,
                    'importance': item.importance,
                    'age_s': time.time() - item.created_at,
                    'source': item.source,
                })
            
            # 按注意力等级排序
            items.sort(key=lambda x: x['attention_level'] * x['importance'], reverse=True)
            
            return {
                'focus_count': len(items),
                'items': items,
                'current_task': self._current_task,
                'activation_count': len(self._activation_pattern),
            }
    
    def unfocus(self, key: str) -> bool:
        """移除焦点"""
        with self._lock:
            if key in self._focus_items:
                # 移入缓冲区
                item = self._focus_items[key]
                self._buffer_items.append({
                    'key': key,
                    'content': item.content,
                    'importance': item.importance,
                    'timestamp': time.time(),
                })
                del self._focus_items[key]
                return True
            return False
    
    def _manage_focus_capacity(self) -> None:
        """管理焦点容量"""
        if len(self._focus_items) <= self.config.max_focus_items:
            return
        
        # 计算每个项的优先级
        priorities = {}
        now = time.time()
        
        for key, item in self._focus_items.items():
            # 优先级 = 重要性 * 注意力 * (1 / (1 + age))
            age = now - item.created_at
            recency = 1.0 / (1.0 + age * 0.1)
            priority = item.importance * item.attention_level * recency
            priorities[key] = priority
        
        # 删除最低优先级的项
        sorted_keys = sorted(priorities.keys(), key=lambda k: priorities[k])
        items_to_remove = len(self._focus_items) - self.config.max_focus_items
        
        for key in sorted_keys[:items_to_remove]:
            self.unfocus(key)
    
    # ==================== 缓冲区管理 ====================
    
    def buffer(self, content: Any, key: Optional[str] = None) -> None:
        """
        将信息放入缓冲区
        
        Args:
            content: 内容
            key: 可选键
        """
        with self._lock:
            self._buffer_items.append({
                'key': key or f"buffer_{len(self._buffer_items)}",
                'content': content,
                'timestamp': time.time(),
            })
    
    def get_buffer(self, recent_n: int = 10) -> List[Dict[str, Any]]:
        """获取最近的缓冲区项"""
        with self._lock:
            items = list(self._buffer_items)[-recent_n:]
            return [
                {
                    'key': item['key'],
                    'content': item['content'],
                    'age_s': time.time() - item['timestamp'],
                }
                for item in items
            ]
    
    # ==================== 注意力衰减 ====================
    
    def apply_decay(self, dt: Optional[float] = None) -> int:
        """
        应用注意力衰减
        
        Args:
            dt: 时间增量 (秒)，默认使用实际经过时间
            
        Returns:
            衰减的项数量
        """
        with self._lock:
            now = time.time()
            if dt is None:
                # 计算经过的时间
                last_item_times = [
                    item.last_accessed
                    for item in self._focus_items.values()
                ]
                if last_item_times:
                    dt = now - min(last_item_times)
                else:
                    dt = 1.0
            
            decayed = 0
            to_remove = []
            
            for key, item in self._focus_items.items():
                # 计算衰减
                age = now - item.last_accessed
                
                # 基于时间和基础衰减率衰减
                item.attention_level *= (1.0 - self.config.decay_base_rate * dt)
                
                # 重要性衰减
                item.importance *= (1.0 - item.decay_rate * dt)
                
                # 检查是否低于阈值
                if item.attention_level < self.config.attention_threshold:
                    to_remove.append(key)
                    decayed += 1
            
            # 移除过期的项
            for key in to_remove:
                self.unfocus(key)
            
            self._decay_operations += 1
            
            return decayed
    
    # ==================== 任务上下文 ====================
    
    def set_task(self, task: Dict[str, Any]) -> None:
        """
        设置当前任务
        
        Args:
            task: 任务描述
        """
        with self._lock:
            self._current_task = {
                'task': task,
                'start_time': time.time(),
                'subgoals': [],
                'current_step': 0,
            }
    
    def update_task_progress(self, step: int, subgoal: Optional[str] = None) -> None:
        """更新任务进度"""
        with self._lock:
            if self._current_task:
                self._current_task['current_step'] = step
                if subgoal:
                    self._current_task['subgoals'].append({
                        'subgoal': subgoal,
                        'completed_at': time.time(),
                    })
    
    def get_current_task(self) -> Optional[Dict[str, Any]]:
        """获取当前任务"""
        with self._lock:
            return self._current_task
    
    def clear_task(self) -> None:
        """清除当前任务"""
        with self._lock:
            self._current_task = None
            self._reasoning_stack.clear()
    
    # ==================== 推理栈 ====================
    
    def push_reasoning(self, state: Dict[str, Any]) -> None:
        """
        推入推理状态
        
        Args:
            state: 推理状态
        """
        with self._lock:
            self._reasoning_stack.append({
                'state': state,
                'timestamp': time.time(),
            })
    
    def pop_reasoning(self) -> Optional[Dict[str, Any]]:
        """弹出推理状态"""
        with self._lock:
            if self._reasoning_stack:
                return self._reasoning_stack.pop()
            return None
    
    def get_reasoning_trace(self) -> List[Dict[str, Any]]:
        """获取推理痕迹"""
        with self._lock:
            return [
                {
                    'state': item['state'],
                    'age_s': time.time() - item['timestamp'],
                }
                for item in self._reasoning_stack
            ]
    
    def clear_reasoning(self) -> None:
        """清除推理栈"""
        with self._lock:
            self._reasoning_stack.clear()
    
    # ==================== 变量绑定 ====================
    
    def bind(self, variable: str, value: Any) -> None:
        """
        绑定变量
        
        Args:
            variable: 变量名
            value: 值
        """
        with self._lock:
            self._bindings[variable] = {
                'value': value,
                'timestamp': time.time(),
            }
    
    def get_binding(self, variable: str) -> Optional[Any]:
        """获取绑定值"""
        with self._lock:
            binding = self._bindings.get(variable)
            return binding['value'] if binding else None
    
    def get_all_bindings(self) -> Dict[str, Any]:
        """获取所有绑定"""
        with self._lock:
            return {
                var: binding['value']
                for var, binding in self._bindings.items()
            }
    
    def clear_bindings(self) -> None:
        """清除所有绑定"""
        with self._lock:
            self._bindings.clear()
    
    # ==================== 激活模式 ====================
    
    def activate(self, concept_id: str) -> None:
        """激活概念"""
        with self._lock:
            self._activation_pattern.add(concept_id)
    
    def deactivate(self, concept_id: str) -> None:
        """停用概念"""
        with self._lock:
            self._activation_pattern.discard(concept_id)
    
    def is_activated(self, concept_id: str) -> bool:
        """检查是否激活"""
        with self._lock:
            return concept_id in self._activation_pattern
    
    def get_activation_pattern(self) -> Set[str]:
        """获取当前激活模式"""
        with self._lock:
            return self._activation_pattern.copy()
    
    def clear_activation(self) -> None:
        """清除激活模式"""
        with self._lock:
            self._activation_pattern.clear()
    
    # ==================== 长期记忆交互 ====================
    
    def prepare_for_storage(self) -> Dict[str, Any]:
        """
        准备数据用于存储到长期记忆
        
        Returns:
            待存储的数据
        """
        with self._lock:
            return {
                'focus_summary': {
                    key: {
                        'attention_level': item.attention_level,
                        'importance': item.importance,
                        'source': item.source,
                    }
                    for key, item in self._focus_items.items()
                },
                'current_task': self._current_task,
                'reasoning_depth': len(self._reasoning_stack),
                'timestamp': time.time(),
            }
    
    def load_from_memory(
        self,
        focus_data: Dict[str, Dict[str, Any]],
        task_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        从长期记忆加载数据
        
        Args:
            focus_data: 焦点数据
            task_data: 任务数据
        """
        with self._lock:
            # 恢复焦点项
            for key, data in focus_data.items():
                self._focus_items[key] = FocusItem(
                    content=None,  # 需要从外部加载
                    attention_level=data.get('attention_level', 0.5),
                    importance=data.get('importance', 5.0),
                    source=data.get('source', 'memory'),
                )
            
            if task_data:
                self._current_task = task_data
    
    # ==================== 状态管理 ====================
    
    def clear(self) -> None:
        """清除所有工作记忆"""
        with self._lock:
            self._focus_items.clear()
            self._buffer_items.clear()
            self._current_task = None
            self._reasoning_stack.clear()
            self._bindings.clear()
            self._activation_pattern.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        with self._lock:
            return {
                'focus_count': len(self._focus_items),
                'buffer_count': len(self._buffer_items),
                'has_task': self._current_task is not None,
                'reasoning_depth': len(self._reasoning_stack),
                'binding_count': len(self._bindings),
                'activation_count': len(self._activation_pattern),
                'total_processed': self._total_items_processed,
                'decay_operations': self._decay_operations,
            }
    
    # ==================== 便捷方法 ====================
    
    def __contains__(self, key: str) -> bool:
        """检查键是否在焦点中"""
        with self._lock:
            return key in self._focus_items
    
    def __getitem__(self, key: str) -> Optional[Any]:
        """快速访问"""
        return self.get_focused(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """快速存储"""
        self.focus(key, value)
    
    def __len__(self) -> int:
        """焦点数量"""
        with self._lock:
            return len(self._focus_items)
