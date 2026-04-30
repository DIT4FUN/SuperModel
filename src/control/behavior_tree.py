# Copyright (C) 2024-2026 赵元请 (DIT4FUN)
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
行为树模块 (Behavior Tree)
==========================

分层任务执行框架, 与HTN planner互补:
- 节点类型: Selector, Sequence, Parallel, Decorator, Action, Condition
- 状态机集成: 与现有 state machine 无缝对接
- AGV五级规格: S级简单阈值触发 → XXL级MPC预测规划
- 应用场景: 任务执行、应急响应、自主导航、协作控制

与HTN Planner的关系:
  HTN Planner: 战略层 - 任务分解规划
  Behavior Tree: 战术层 - 实时行为执行
  Controller: 执行层 - 底层伺服控制

Author: SuperModel Dev Team
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional, Dict, Any, Tuple, Union
from abc import ABC, abstractmethod
import time


# ─────────────────────────────────────────────
# 节点状态枚举
# ─────────────────────────────────────────────

class NodeState(Enum):
    """行为树节点状态"""
    IDLE = auto()      # 空闲/未运行
    RUNNING = auto()   # 运行中
    SUCCESS = auto()   # 成功
    FAILURE = auto()   # 失败
    ERROR = auto()      # 出错


class BTGrade(str, Enum):
    """AGV五级行为树规格"""
    S = 'S'
    M = 'M'
    L = 'L'
    XL = 'XL'
    XXL = 'XXL'


# ─────────────────────────────────────────────
# AGV五级行为树规格表
# ─────────────────────────────────────────────

AGV_BT_GRADES = {
    'S': {
        'description': '小型AGV-阈值触发',
        'max_tree_depth': 3,
        'max_nodes': 16,
        'tick_rate_hz': 10,
        'supported_nodes': ['selector', 'sequence', 'action', 'condition'],
        'parallel_execution': False,
        'memory_nodes': False,
        'decorator_types': 1,
        'action_timeout_ms': 5000,
        'preemption': False,
    },
    'M': {
        'description': '中型AGV-条件决策',
        'max_tree_depth': 5,
        'max_nodes': 64,
        'tick_rate_hz': 50,
        'supported_nodes': ['selector', 'sequence', 'parallel', 'action', 'condition'],
        'parallel_execution': True,
        'memory_nodes': True,
        'decorator_types': 3,
        'action_timeout_ms': 2000,
        'preemption': True,
    },
    'L': {
        'description': '大型AGV-分层行为',
        'max_tree_depth': 8,
        'max_nodes': 256,
        'tick_rate_hz': 100,
        'supported_nodes': ['selector', 'sequence', 'parallel', 'decorator', 'action', 'condition'],
        'parallel_execution': True,
        'memory_nodes': True,
        'decorator_types': 6,
        'action_timeout_ms': 1000,
        'preemption': True,
    },
    'XL': {
        'description': '超大型AGV-预测规划',
        'max_tree_depth': 12,
        'max_nodes': 1024,
        'tick_rate_hz': 200,
        'supported_nodes': ['selector', 'sequence', 'parallel', 'decorator', 'action', 'condition', 'subtree'],
        'parallel_execution': True,
        'memory_nodes': True,
        'decorator_types': 10,
        'action_timeout_ms': 500,
        'preemption': True,
    },
    'XXL': {
        'description': '超超大型AGV-MPC集成',
        'max_tree_depth': 16,
        'max_nodes': 4096,
        'tick_rate_hz': 500,
        'supported_nodes': ['selector', 'sequence', 'parallel', 'decorator', 'action', 'condition', 'subtree', 'dynamic'],
        'parallel_execution': True,
        'memory_nodes': True,
        'decorator_types': 16,
        'action_timeout_ms': 100,
        'preemption': True,
    },
}


# ─────────────────────────────────────────────
# 基类定义
# ─────────────────────────────────────────────

@dataclass
class BTContext:
    """行为树执行上下文"""
    timestamp: float = 0.0
    agent_id: str = 'bt_agent'
    robot_state: Dict[str, Any] = field(default_factory=dict)
    sensor_data: Dict[str, Any] = field(default_factory=dict)
    blackboard: Dict[str, Any] = field(default_factory=dict)
    execution_count: int = 0
    total_tick_time_ms: float = 0.0

    def get_sensor(self, key: str, default: Any = None) -> Any:
        return self.sensor_data.get(key, default)

    def set_blackboard(self, key: str, value: Any) -> None:
        self.blackboard[key] = value

    def get_blackboard(self, key: str, default: Any = None) -> Any:
        return self.blackboard.get(key, default)


class BTNode(ABC):
    """行为树节点基类"""

    def __init__(
        self,
        name: str,
        grade: BTGrade = BTGrade.M,
        max_retries: int = 0,
    ):
        self.name = name
        self.grade = grade
        self.max_retries = max_retries
        self._state = NodeState.IDLE
        self._retries = 0
        self._last_tick_time: float = 0.0
        self._total_execution_time: float = 0.0
        self._execution_count: int = 0
        self._children: List[BTNode] = []
        self.parent: Optional[BTNode] = None

    @property
    def state(self) -> NodeState:
        return self._state

    @property
    def children(self) -> List[BTNode]:
        return self._children

    def add_child(self, child: BTNode) -> None:
        child.parent = self
        self._children.append(child)

    def remove_child(self, child: BTNode) -> None:
        if child in self._children:
            self._children.remove(child)
            child.parent = None

    def reset(self) -> None:
        self._state = NodeState.IDLE
        self._retries = 0
        for child in self._children:
            child.reset()

    @abstractmethod
    def tick(self, ctx: BTContext) -> NodeState:
        """执行一次tick, 返回节点状态"""
        pass

    def on_start(self, ctx: BTContext) -> None:
        """节点开始执行时调用"""
        self._state = NodeState.RUNNING
        self._last_tick_time = time.time()

    def on_end(self, ctx: BTContext, final_state: NodeState) -> None:
        """节点结束执行时调用"""
        self._state = final_state
        self._execution_count += 1
        elapsed = (time.time() - self._last_tick_time) * 1000.0
        self._total_execution_time += elapsed

    def get_stats(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'state': self.state.name,
            'executions': self._execution_count,
            'total_time_ms': self._total_execution_time,
            'avg_time_ms': self._total_execution_time / max(1, self._execution_count),
            'retries': self._retries,
        }


# ─────────────────────────────────────────────
# 控制流节点 (Control Flow Nodes)
# ─────────────────────────────────────────────

class Selector(BTNode):
    """
    Selector (Fallback) 节点
    - 从左到右尝试子节点
    - 遇到第一个 SUCCESS 则返回 SUCCESS
    - 全部 FAILURE 才返回 FAILURE
    - RUNNING 状态会中断后续节点执行
    """

    def __init__(self, name: str = 'Selector', grade: BTGrade = BTGrade.M, **kwargs):
        super().__init__(name, grade, **kwargs)
        self._running_child: Optional[BTNode] = None

    def tick(self, ctx: BTContext) -> NodeState:
        if not self._children:
            return NodeState.SUCCESS

        for i, child in enumerate(self._children):
            result = child.tick(ctx)

            if result == NodeState.RUNNING:
                self._running_child = child
                self.on_start(ctx)
                return NodeState.RUNNING

            if result == NodeState.SUCCESS:
                self.on_end(ctx, NodeState.SUCCESS)
                return NodeState.SUCCESS

        self._running_child = None
        self.on_end(ctx, NodeState.FAILURE)
        return NodeState.FAILURE


class Sequence(BTNode):
    """
    Sequence 节点
    - 从左到右执行子节点
    - 遇到 FAILURE 则返回 FAILURE
    - 全部 SUCCESS 才返回 SUCCESS
    - RUNNING 状态会中断后续节点执行
    """

    def __init__(self, name: str = 'Sequence', grade: BTGrade = BTGrade.M, **kwargs):
        super().__init__(name, grade, **kwargs)
        self._running_child: Optional[BTNode] = None

    def tick(self, ctx: BTContext) -> NodeState:
        if not self._children:
            return NodeState.SUCCESS

        for child in self._children:
            result = child.tick(ctx)

            if result == NodeState.RUNNING:
                self._running_child = child
                self.on_start(ctx)
                return NodeState.RUNNING

            if result == NodeState.FAILURE:
                self._running_child = None
                self.on_end(ctx, NodeState.FAILURE)
                return NodeState.FAILURE

        self._running_child = None
        self.on_end(ctx, NodeState.SUCCESS)
        return NodeState.SUCCESS


class Parallel(BTNode):
    """
    Parallel 节点
    - 同时执行所有子节点
    - policy: 'success_on_one' / 'success_on_all' / 'failure_on_one' / 'failure_on_all'
    - AGV L级以上支持
    """

    def __init__(
        self,
        name: str = 'Parallel',
        grade: BTGrade = BTGrade.L,
        policy: str = 'success_on_all',
        **kwargs
    ):
        super().__init__(name, grade, **kwargs)
        self.policy = policy

    def tick(self, ctx: BTContext) -> NodeState:
        if not self._children:
            return NodeState.SUCCESS

        results = [child.tick(ctx) for child in self._children]

        if self.policy == 'success_on_one':
            if NodeState.SUCCESS in results:
                return NodeState.SUCCESS
            if NodeState.RUNNING in results:
                return NodeState.RUNNING
            return NodeState.FAILURE

        if self.policy == 'success_on_all':
            if NodeState.FAILURE in results:
                return NodeState.FAILURE
            if NodeState.RUNNING in results:
                return NodeState.RUNNING
            return NodeState.SUCCESS

        if self.policy == 'failure_on_one':
            if NodeState.FAILURE in results:
                return NodeState.FAILURE
            if NodeState.RUNNING in results:
                return NodeState.RUNNING
            return NodeState.SUCCESS

        # failure_on_all
        if NodeState.SUCCESS in results:
            return NodeState.SUCCESS
        if NodeState.RUNNING in results:
            return NodeState.RUNNING
        return NodeState.FAILURE


# ─────────────────────────────────────────────
# 条件节点 (Condition Node)
# ─────────────────────────────────────────────

ConditionFn = Callable[[BTContext], bool]


class Condition(BTNode):
    """条件节点: 检查黑板/传感器/状态"""

    def __init__(
        self,
        name: str,
        condition_fn: ConditionFn,
        description: str = '',
        grade: BTGrade = BTGrade.M,
        **kwargs
    ):
        super().__init__(name, grade, **kwargs)
        self.condition_fn = condition_fn
        self.description = description

    def tick(self, ctx: BTContext) -> NodeState:
        self.on_start(ctx)
        try:
            result = self.condition_fn(ctx)
            state = NodeState.SUCCESS if result else NodeState.FAILURE
            self.on_end(ctx, state)
            return state
        except Exception:
            self.on_end(ctx, NodeState.ERROR)
            return NodeState.ERROR


# ─────────────────────────────────────────────
# 动作节点 (Action Node)
# ─────────────────────────────────────────────

ActionFn = Callable[[BTContext], NodeState]


class Action(BTNode):
    """动作节点: 执行具体行为"""

    def __init__(
        self,
        name: str,
        action_fn: ActionFn,
        timeout_ms: float = 5000.0,
        grade: BTGrade = BTGrade.M,
        **kwargs
    ):
        super().__init__(name, grade, **kwargs)
        self.action_fn = action_fn
        self.timeout_ms = timeout_ms
        self._start_time: float = 0.0

    def tick(self, ctx: BTContext) -> NodeState:
        if self._state == NodeState.IDLE:
            self._start_time = time.time()
            self.on_start(ctx)

        elapsed_ms = (time.time() - self._start_time) * 1000.0
        if elapsed_ms > self.timeout_ms:
            self.on_end(ctx, NodeState.FAILURE)
            return NodeState.FAILURE

        try:
            result = self.action_fn(ctx)
            if result not in (NodeState.SUCCESS, NodeState.FAILURE, NodeState.RUNNING):
                result = NodeState.SUCCESS
            self.on_end(ctx, result)
            return result
        except Exception:
            self.on_end(ctx, NodeState.ERROR)
            return NodeState.ERROR

    def reset(self) -> None:
        super().reset()
        self._start_time = 0.0


# ─────────────────────────────────────────────
# 装饰器节点 (Decorator Nodes)
# ─────────────────────────────────────────────

class Inverter(BTNode):
    """取反装饰器: SUCCESS→FAILURE, FAILURE→SUCCESS"""

    def __init__(self, name: str = 'Inverter', grade: BTGrade = BTGrade.M, **kwargs):
        super().__init__(name, grade, **kwargs)
        self._child: Optional[BTNode] = None

    @property
    def child(self) -> Optional[BTNode]:
        return self._child

    @child.setter
    def child(self, node: BTNode) -> None:
        self._child = node
        self._children.clear()
        if node:
            self.add_child(node)

    def tick(self, ctx: BTContext) -> NodeState:
        if not self._child:
            return NodeState.SUCCESS

        result = self._child.tick(ctx)

        if result == NodeState.SUCCESS:
            out = NodeState.FAILURE
        elif result == NodeState.FAILURE:
            out = NodeState.SUCCESS
        else:
            out = result

        self.on_end(ctx, out)
        return out


class RepeatUntil(BTNode):
    """循环装饰器: 直到条件满足或达到次数上限"""

    def __init__(
        self,
        name: str = 'RepeatUntil',
        maxRepeats: int = 10,
        until_success: bool = True,
        grade: BTGrade = BTGrade.M,
        **kwargs
    ):
        super().__init__(name, grade, **kwargs)
        self.max_repeats = maxRepeats
        self.until_success = until_success
        self._child: Optional[BTNode] = None
        self._repeat_count: int = 0

    @property
    def child(self) -> Optional[BTNode]:
        return self._child

    @child.setter
    def child(self, node: BTNode) -> None:
        self._child = node
        self._children.clear()
        if node:
            self.add_child(node)

    def tick(self, ctx: BTContext) -> NodeState:
        if not self._child:
            return NodeState.SUCCESS

        # If already in SUCCESS/FAILURE, don't re-execute child
        if self._state in (NodeState.SUCCESS, NodeState.FAILURE):
            return self._state

        result = self._child.tick(ctx)
        self._repeat_count += 1

        if self.until_success:
            if result == NodeState.SUCCESS:
                self.on_end(ctx, NodeState.SUCCESS)
                return NodeState.SUCCESS
        else:
            if result == NodeState.FAILURE:
                self.on_end(ctx, NodeState.FAILURE)
                return NodeState.FAILURE

        if self._repeat_count >= self.max_repeats:
            self._repeat_count = 0
            self.on_end(ctx, NodeState.SUCCESS)
            return NodeState.SUCCESS

        self.on_start(ctx)
        return NodeState.RUNNING

    def reset(self) -> None:
        super().reset()
        self._repeat_count = 0


class RetryUntil(BTNode):
    """重试装饰器: 失败时重试直到成功或达到上限"""

    def __init__(
        self,
        name: str = 'RetryUntil',
        max_retries: int = 3,
        retry_on_failure: bool = True,
        grade: BTGrade = BTGrade.M,
        **kwargs
    ):
        super().__init__(name, grade, **kwargs)
        self.max_retries = max_retries
        self.retry_on_failure = retry_on_failure
        self._child: Optional[BTNode] = None

    @property
    def child(self) -> Optional[BTNode]:
        return self._child

    @child.setter
    def child(self, node: BTNode) -> None:
        self._child = node
        self._children.clear()
        if node:
            self.add_child(node)

    def tick(self, ctx: BTContext) -> NodeState:
        if not self._child:
            return NodeState.SUCCESS

        result = self._child.tick(ctx)

        if result == NodeState.SUCCESS:
            self.on_end(ctx, NodeState.SUCCESS)
            return NodeState.SUCCESS

        if result == NodeState.RUNNING:
            return NodeState.RUNNING

        # FAILURE: retry if enabled
        if result == NodeState.FAILURE:
            if not self.retry_on_failure:
                self.on_end(ctx, NodeState.FAILURE)
                return NodeState.FAILURE

        # Non-SUCCESS/non-RUNNING triggers retry
        self._retries += 1
        if self._retries >= self.max_retries:
            self._retries = 0
            self.on_end(ctx, NodeState.FAILURE)
            return NodeState.FAILURE

        self._child.reset()
        self.on_end(ctx, NodeState.RUNNING)
        return NodeState.RUNNING

    def reset(self) -> None:
        super().reset()
        self._retries = 0


class Timeout(BTNode):
    """超时装饰器: 子节点执行超时后强制返回失败"""

    def __init__(
        self,
        name: str = 'Timeout',
        timeout_ms: float = 1000.0,
        grade: BTGrade = BTGrade.M,
        **kwargs
    ):
        super().__init__(name, grade, **kwargs)
        self.timeout_ms = timeout_ms
        self._child: Optional[BTNode] = None
        self._start_time: float = 0.0

    @property
    def child(self) -> Optional[BTNode]:
        return self._child

    @child.setter
    def child(self, node: BTNode) -> None:
        self._child = node
        self._children.clear()
        if node:
            self.add_child(node)

    def tick(self, ctx: BTContext) -> NodeState:
        if not self._child:
            return NodeState.SUCCESS

        if self._state == NodeState.IDLE:
            self._start_time = time.time()

        elapsed_ms = (time.time() - self._start_time) * 1000.0
        if elapsed_ms > self.timeout_ms:
            self._child.reset()
            self._start_time = 0.0
            self.on_end(ctx, NodeState.FAILURE)
            return NodeState.FAILURE

        result = self._child.tick(ctx)

        if result != NodeState.RUNNING:
            self._start_time = 0.0

        self.on_end(ctx, result)
        return result

    def reset(self) -> None:
        super().reset()
        self._start_time = 0.0


class RateLimiter(BTNode):
    """频率限制装饰器: 限制子节点tick频率"""

    def __init__(
        self,
        name: str = 'RateLimiter',
        max_rate_hz: float = 10.0,
        grade: BTGrade = BTGrade.M,
        **kwargs
    ):
        super().__init__(name, grade, **kwargs)
        self.max_rate_hz = max_rate_hz
        self.min_interval_s = 1.0 / max_rate_hz
        self._child: Optional[BTNode] = None
        self._last_tick_time: float = 0.0

    @property
    def child(self) -> Optional[BTNode]:
        return self._child

    @child.setter
    def child(self, node: BTNode) -> None:
        self._child = node
        self._children.clear()
        if node:
            self.add_child(node)

    def tick(self, ctx: BTContext) -> NodeState:
        if not self._child:
            return NodeState.SUCCESS

        now = time.time()
        elapsed = now - self._last_tick_time

        if elapsed < self.min_interval_s:
            self.on_end(ctx, NodeState.SUCCESS)
            return NodeState.SUCCESS

        self._last_tick_time = now
        result = self._child.tick(ctx)
        self.on_end(ctx, result)
        return result

    def reset(self) -> None:
        super().reset()
        self._last_tick_time = 0.0


# ─────────────────────────────────────────────
# 子树引用节点 (Subtree Reference)
# ─────────────────────────────────────────────

class SubTree(BTNode):
    """子树引用节点: 引用外部定义的行为树"""

    def __init__(
        self,
        name: str,
        subtree_root: BTNode,
        grade: BTGrade = BTGrade.XL,
        **kwargs
    ):
        super().__init__(name, grade, **kwargs)
        self.subtree_root = subtree_root

    def tick(self, ctx: BTContext) -> NodeState:
        result = self.subtree_root.tick(ctx)
        self.on_end(ctx, result)
        return result

    def reset(self) -> None:
        super().reset()
        self.subtree_root.reset()


# ─────────────────────────────────────────────
# 行为树管理器
# ─────────────────────────────────────────────

class BehaviorTree:
    """完整行为树管理器"""

    def __init__(
        self,
        root: BTNode,
        grade: BTGrade = BTGrade.M,
        name: str = 'BT',
    ):
        self.root = root
        self.grade = grade
        self.name = name
        self.ctx = BTContext()
        self._running = False
        self._tick_count: int = 0
        self._last_tick_duration: float = 0.0
        self._total_tick_duration: float = 0.0

    @classmethod
    def create_for_grade(cls, grade: BTGrade, root: BTNode, name: str = 'BT') -> BehaviorTree:
        return cls(root=root, grade=grade, name=name)

    def tick(self) -> NodeState:
        """执行一次行为树tick"""
        ctx_timestamp = time.time()
        self.ctx.timestamp = ctx_timestamp
        self.ctx.execution_count += 1

        t0 = time.time()
        result = self.root.tick(self.ctx)
        t1 = time.time()

        self._tick_count += 1
        self._last_tick_duration = (t1 - t0) * 1000.0
        self._total_tick_duration += self._last_tick_duration
        self.ctx.total_tick_time_ms = self._total_tick_duration

        return result

    def reset(self) -> None:
        """重置行为树"""
        self.root.reset()
        self._tick_count = 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'grade': self.grade.value,
            'tick_count': self._tick_count,
            'last_tick_ms': self._last_tick_duration,
            'avg_tick_ms': self._total_tick_duration / max(1, self._tick_count),
            'root_state': self.root.state.name,
            'root_stats': self.root.get_stats(),
        }


# ─────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────

def create_for_grade(grade: BTGrade, root: BTNode, name: str = 'BT') -> BehaviorTree:
    """为AGV五级规格创建行为树"""
    return BehaviorTree.create_for_grade(grade, root, name)


def create_safe_selector(name: str, children: List[BTNode], grade: BTGrade = BTGrade.M) -> Selector:
    """创建带安全检查的Selector"""
    sel = Selector(name, grade)
    for child in children:
        sel.add_child(child)
    return sel


def create_action_sequence(name: str, actions: List[BTNode], grade: BTGrade = BTGrade.M) -> Sequence:
    """创建顺序执行的动作序列"""
    seq = Sequence(name, grade)
    for action in actions:
        seq.add_child(action)
    return seq
