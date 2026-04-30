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
behavior_tree.py - 行为树具身任务规划模块
SuperModel 超模态大模型具身智能系统

支持:
- 序列节点 (Sequence)
- 选择节点 (Selector)
- 并行节点 (Parallel)
- 装饰器节点 (Decorator)
- 条件节点 (Condition)
- 动作节点 (Action)
- 行为树动态重规划
- AGV五级规格适配
- 具身任务层级规划
"""

from __future__ import annotations
import abc
import math
import enum
from enum import Enum  # noqa: F401, F403
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set
import numpy as np

# 简单Profiler定义
class Profiler:
    def __init__(self, enabled=False):
        self.enabled = enabled
    def profile(self, name):
        class context:
            def __enter__(self): pass
            def __exit__(self, *args): pass
        return context()

# 简单logger
import logging
logger = logging.getLogger(__name__)

__all__ = [
    'NodeStatus',
    'BTNode',
    'SequenceNode',
    'SelectorNode',
    'ParallelNode',
    'RepeaterNode',
    'UntilFailNode',
    'UntilSuccessNode',
    'InverterNode',
    'ConditionNode',
    'ActionNode',
    'BehaviorTree',
    'EmbodiedTaskPlanner',
    'AGVTaskPlanner',
    'TaskStatus',
    # AGV-specific nodes
    'IsAtTarget',
    'IsBatteryLow',
    'MoveTo',
    'Pickup',
    # Config-driven builder
    'create_behavior_tree_from_dict',
    'serialize_behavior_tree',
    'create_task_bt_from_config',
    'load_behavior_tree_from_yaml',
    'load_behavior_tree_from_json',
]


class NodeStatus(enum.Enum):
    """行为树节点状态"""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    SKIPPED = "SKIPPED"


class TaskStatus(enum.Enum):
    """具身任务整体状态"""
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


@dataclass
class Blackboard:
    """行为树黑板 - 共享数据存储"""
    data: Dict[str, Any] = field(default_factory=dict)
    robot_state: Dict[str, Any] = field(default_factory=dict)
    world_state: Dict[str, Any] = field(default_factory=dict)
    goal_state: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def has(self, key: str) -> bool:
        return key in self.data

    def remove(self, key: str) -> bool:
        if key in self.data:
            del self.data[key]
            return True
        return False

    def update_robot_state(self, updates: Dict[str, Any]) -> None:
        """更新机器人状态"""
        self.robot_state.update(updates)
        self.timestamp = time.time()

    def update_world_state(self, updates: Dict[str, Any]) -> None:
        """更新世界状态"""
        self.world_state.update(updates)
        self.timestamp = time.time()

    def get_robot_position(self) -> Optional[np.ndarray]:
        """获取机器人位置"""
        pos = self.robot_state.get('position')
        return np.array(pos) if pos is not None else None

    def get_robot_velocity(self) -> Optional[np.ndarray]:
        """获取机器人速度"""
        vel = self.robot_state.get('velocity')
        return np.array(vel) if vel is not None else None

    def get_battery_level(self) -> Optional[float]:
        """获取电池电量"""
        return self.robot_state.get('battery_level')

    def is_safe(self) -> bool:
        """检查是否安全"""
        return self.robot_state.get('safety', True)


class BTNode(abc.ABC):
    """行为树节点基类"""

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self.status: NodeStatus = NodeStatus.IDLE
        self.parent: Optional[BTNode] = None
        self.children: List[BTNode] = []
        self.start_time: Optional[float] = None
        self.last_tick: Optional[float] = None

    def add_child(self, child: BTNode) -> BTNode:
        child.parent = self
        self.children.append(child)
        return self

    def add_children(self, *children: BTNode) -> BTNode:
        for child in children:
            self.add_child(child)
        return self

    @abc.abstractmethod
    def tick(self, blackboard: Blackboard) -> NodeStatus:
        """执行一次节点 tick"""
        pass

    def reset(self) -> None:
        """重置节点状态"""
        self.status = NodeStatus.IDLE
        self.start_time = None
        self.last_tick = None
        for child in self.children:
            child.reset()

    def initialize(self) -> None:
        """节点初始化"""
        self.status = NodeStatus.RUNNING
        self.start_time = time.time()

    def terminate(self, new_status: NodeStatus) -> NodeStatus:
        """节点终止"""
        self.status = new_status
        return new_status

    def get_running_time(self) -> float:
        """获取运行时间"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', status={self.status})"


class CompositeNode(BTNode):
    """组合节点基类"""

    def __init__(self, name: str = "", children: Optional[List[BTNode]] = None):
        super().__init__(name)
        if children:
            for child in children:
                self.add_child(child)


class SequenceNode(CompositeNode):
    """
    序列节点 - 顺序执行所有子节点
    任意子节点失败 → 整体失败
    所有子节点成功 → 整体成功
    某个节点运行中 → 保持运行
    """
    _active_child: Optional['BTNode'] = None  # 当前活跃子节点

    def tick(self, blackboard: Blackboard) -> NodeStatus:
        if self.status != NodeStatus.RUNNING:
            self.initialize()
            self._active_child = None

        # 跟踪当前处理到的子节点索引
        start_from_active = self._active_child is not None
        active_idx = -1

        for i, child in enumerate(self.children):
            # 如果之前有活跃子节点，只从该节点继续（跳过已完成节点）
            if start_from_active:
                if child is not self._active_child:
                    continue  # 跳过已完成节点
                else:
                    start_from_active = False  # 找到活跃节点，继续处理后续节点

            status = child.tick(blackboard)

            if status == NodeStatus.RUNNING:
                self._active_child = child  # 记住当前活跃节点
                return self.terminate(NodeStatus.RUNNING)
            elif status == NodeStatus.FAILURE:
                self._active_child = None
                return self.terminate(NodeStatus.FAILURE)
            # SUCCESS: 继续下一个，_active_child 保持 None（表示无活跃节点）
            self._active_child = None

        self._active_child = None
        return self.terminate(NodeStatus.SUCCESS)


class SelectorNode(CompositeNode):
    """
    选择节点 - 按顺序尝试子节点
    任意子节点成功 → 整体成功
    所有子节点失败 → 整体失败
    某个节点运行中 → 保持运行
    """

    def tick(self, blackboard: Blackboard) -> NodeStatus:
        if self.status != NodeStatus.RUNNING:
            self.initialize()

        for child in self.children:
            status = child.tick(blackboard)

            if status == NodeStatus.RUNNING:
                return self.terminate(NodeStatus.RUNNING)
            elif status == NodeStatus.SUCCESS:
                return self.terminate(NodeStatus.SUCCESS)

        return self.terminate(NodeStatus.FAILURE)


class ParallelNode(CompositeNode):
    """
    并行节点 - 同时执行所有子节点
    策略:
    - require_all: 所有成功才成功, 任意失败即失败
    - require_any: 任意成功即成功, 所有失败才失败
    """

    class Policy(enum.Enum):
        REQUIRE_ALL = "REQUIRE_ALL"
        REQUIRE_ANY = "REQUIRE_ANY"

    def __init__(self, name: str = "", success_policy: Policy = Policy.REQUIRE_ALL,
                 failure_policy: Policy = Policy.REQUIRE_ANY, children: Optional[List[BTNode]] = None):
        super().__init__(name, children)
        self.success_policy = success_policy
        self.failure_policy = failure_policy

    def tick(self, blackboard: Blackboard) -> NodeStatus:
        if self.status != NodeStatus.RUNNING:
            self.initialize()

        success_count = 0
        failure_count = 0

        for child in self.children:
            status = child.tick(blackboard)

            if status == NodeStatus.SUCCESS:
                success_count += 1
            elif status == NodeStatus.FAILURE:
                failure_count += 1

        # 检查失败条件
        if self.failure_policy == self.Policy.REQUIRE_ALL:
            if failure_count == len(self.children):
                return self.terminate(NodeStatus.FAILURE)
        else:  # REQUIRE_ANY
            if failure_count > 0:
                return self.terminate(NodeStatus.FAILURE)

        # 检查成功条件
        if self.success_policy == self.Policy.REQUIRE_ALL:
            if success_count == len(self.children):
                return self.terminate(NodeStatus.SUCCESS)
        else:  # REQUIRE_ANY
            if success_count > 0:
                return self.terminate(NodeStatus.SUCCESS)

        # 仍有节点在运行
        return self.terminate(NodeStatus.RUNNING)


class DecoratorNode(BTNode):
    """装饰器节点基类"""

    def __init__(self, child: BTNode, name: str = ""):
        super().__init__(name)
        self.add_child(child)

    @property
    def child(self) -> BTNode:
        return self.children[0]


class RepeaterNode(DecoratorNode):
    """
    重复执行装饰器
    times = -1 表示无限重复
    """

    def __init__(self, child: BTNode, times: int = -1, name: str = "Repeater"):
        super().__init__(child, name)
        self.times = times
        self.current_count = 0

    def tick(self, blackboard: Blackboard) -> NodeStatus:
        if self.status != NodeStatus.RUNNING:
            self.initialize()
            self.current_count = 0

        status = self.child.tick(blackboard)

        if self.times < 0:
            # 无限重复，只要子节点不失败就一直运行
            if status == NodeStatus.FAILURE:
                return self.terminate(NodeStatus.FAILURE)
            return self.terminate(NodeStatus.RUNNING)

        if status == NodeStatus.SUCCESS or status == NodeStatus.FAILURE:
            self.current_count += 1
            if self.current_count >= self.times:
                return self.terminate(status)

        return self.terminate(NodeStatus.RUNNING)

    def reset(self) -> None:
        self.current_count = 0
        super().reset()


class UntilFailNode(DecoratorNode):
    """直到失败装饰器"""

    def tick(self, blackboard: Blackboard) -> NodeStatus:
        if self.status != NodeStatus.RUNNING:
            self.initialize()

        status = self.child.tick(blackboard)

        if status == NodeStatus.FAILURE:
            return self.terminate(NodeStatus.SUCCESS)

        return self.terminate(NodeStatus.RUNNING)


class UntilSuccessNode(DecoratorNode):
    """直到成功装饰器"""

    def tick(self, blackboard: Blackboard) -> NodeStatus:
        if self.status != NodeStatus.RUNNING:
            self.initialize()

        status = self.child.tick(blackboard)

        if status == NodeStatus.SUCCESS:
            return self.terminate(NodeStatus.SUCCESS)

        return self.terminate(NodeStatus.RUNNING)


class InverterNode(DecoratorNode):
    """反转结果装饰器"""

    def tick(self, blackboard: Blackboard) -> NodeStatus:
        if self.status != NodeStatus.RUNNING:
            self.initialize()

        status = self.child.tick(blackboard)

        if status == NodeStatus.SUCCESS:
            return self.terminate(NodeStatus.FAILURE)
        elif status == NodeStatus.FAILURE:
            return self.terminate(NodeStatus.SUCCESS)

        return self.terminate(status)


class ConditionNode(BTNode):
    """条件节点"""

    def __init__(self, condition: Callable[[Blackboard], bool], name: str = "Condition"):
        super().__init__(name)
        self.condition = condition

    def tick(self, blackboard: Blackboard) -> NodeStatus:
        self.initialize()
        result = self.condition(blackboard)
        return self.terminate(NodeStatus.SUCCESS if result else NodeStatus.FAILURE)


class ActionNode(BTNode):
    """动作节点基类"""

    def __init__(self, name: str = "Action"):
        super().__init__(name)

    @abc.abstractmethod
    def execute(self, blackboard: Blackboard) -> NodeStatus:
        """执行动作"""
        pass

    def tick(self, blackboard: Blackboard) -> NodeStatus:
        # 只在非RUNNING状态时初始化（避免SUCCESS后重复执行）
        if self.status == NodeStatus.IDLE:
            self.initialize()
        elif self.status == NodeStatus.SUCCESS:
            # 已完成的动作不再重复执行
            return self.status
        elif self.status == NodeStatus.FAILURE:
            # 已失败的动作不再重复执行
            return self.status
        return self.terminate(self.execute(blackboard))


class LambdaActionNode(ActionNode):
    """Lambda 动作节点 - 使用函数快速定义动作"""

    def __init__(self, action: Callable[[Blackboard], NodeStatus], name: str = "LambdaAction"):
        super().__init__(name)
        self.action = action

    def execute(self, blackboard: Blackboard) -> NodeStatus:
        return self.action(blackboard)


class BehaviorTree:
    """行为树"""

    def __init__(self, root: BTNode, name: str = "BehaviorTree"):
        self.root = root
        self.name = name
        self.blackboard = Blackboard()
        self.last_status: NodeStatus = NodeStatus.IDLE
        self.profiler = Profiler(enabled=False)

    def tick(self) -> NodeStatus:
        """执行一次 tick"""
        with self.profiler.profile("tick"):
            self.last_status = self.root.tick(self.blackboard)
            return self.last_status

    def reset(self) -> None:
        """重置整棵树"""
        self.root.reset()
        self.last_status = NodeStatus.IDLE

    def is_running(self) -> bool:
        return self.last_status == NodeStatus.RUNNING

    def is_complete(self) -> bool:
        return self.last_status in (NodeStatus.SUCCESS, NodeStatus.FAILURE)

    def update_robot_state(self, updates: Dict[str, Any]) -> None:
        """更新机器人状态到黑板"""
        self.blackboard.update_robot_state(updates)

    def update_world_state(self, updates: Dict[str, Any]) -> None:
        """更新世界状态到黑板"""
        self.blackboard.update_world_state(updates)

    def set_goal(self, goal: Dict[str, Any]) -> None:
        """设置目标"""
        self.blackboard.goal_state.update(goal)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        def count_nodes(node: BTNode) -> Dict[str, int]:
            counts = {node.__class__.__name__: 1}
            for child in node.children:
                child_counts = count_nodes(child)
                for cls, cnt in child_counts.items():
                    counts[cls] = counts.get(cls, 0) + cnt
            return counts

        return {
            'total_nodes': sum(count_nodes(self.root).values()),
            'node_types': count_nodes(self.root),
            'last_status': self.last_status,
        }


@dataclass
class EmbodiedTask:
    """具身任务定义"""

    task_id: str
    task_type: str
    goal_description: str
    target_position: Optional[np.ndarray] = None
    target_object: Optional[str] = None
    required_capabilities: Set[str] = field(default_factory=set)
    constraints: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # 优先级，数字越小优先级越高
    timeout: float = 300.0  # 超时时间(秒)

    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: TaskStatus = TaskStatus.IDLE
    success: bool = False

    def start(self) -> None:
        self.start_time = time.time()
        self.status = TaskStatus.RUNNING

    def finish(self, success: bool = True) -> None:
        self.end_time = time.time()
        self.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        self.success = success

    def is_timeout(self) -> bool:
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) > self.timeout

    def get_duration(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time


class EmbodiedTaskPlanner:
    """
    具身任务规划器
    基于行为树的层级任务规划
    支持动态重规划和多任务调度
    """

    def __init__(self, name: str = "EmbodiedTaskPlanner"):
        self.name = name
        self.tasks: Dict[str, EmbodiedTask] = {}
        self.current_task: Optional[EmbodiedTask] = None
        self.behavior_trees: Dict[str, BehaviorTree] = {}
        self.status: TaskStatus = TaskStatus.IDLE
        self.plan_version: int = 0

    def register_task_type(self, task_type: str, root_node: BTNode) -> None:
        """注册任务类型，创建对应的行为树"""
        bt = BehaviorTree(root_node, name=f"BT_{task_type}")
        self.behavior_trees[task_type] = bt

    def add_task(self, task: EmbodiedTask) -> None:
        """添加任务到队列"""
        self.tasks[task.task_id] = task

    def remove_task(self, task_id: str) -> bool:
        """移除任务"""
        if task_id in self.tasks:
            del self.tasks[task_id]
            if self.current_task and self.current_task.task_id == task_id:
                self.current_task = None
            return True
        return False

    def select_next_task(self) -> Optional[EmbodiedTask]:
        """选择下一个要执行的任务（按优先级）"""
        if not self.tasks:
            return None

        # 按优先级排序
        sorted_tasks = sorted(
            [t for t in self.tasks.values() if t.status == TaskStatus.IDLE],
            key=lambda t: t.priority
        )

        return sorted_tasks[0] if sorted_tasks else None

    def initialize_task(self, task: EmbodiedTask) -> Optional[BehaviorTree]:
        """初始化任务"""
        if task.task_type not in self.behavior_trees:
            logger.error(f"No behavior tree registered for task type: {task.task_type}")
            return None

        # 复制行为树（重置状态）
        bt = self.behavior_trees[task.task_type]
        bt.reset()

        # 设置目标到黑板
        if task.target_position is not None:
            bt.set_goal({'target_position': task.target_position})
        if task.target_object is not None:
            bt.set_goal({'target_object': task.target_object})

        task.start()
        self.current_task = task
        self.status = TaskStatus.RUNNING
        self.plan_version += 1

        return bt

    def tick(self, robot_state: Dict[str, Any], world_state: Dict[str, Any]) -> NodeStatus:
        """执行一次规划 tick"""
        if self.current_task is None:
            # 选择下一个任务
            next_task = self.select_next_task()
            if next_task is None:
                self.status = TaskStatus.IDLE
                return NodeStatus.IDLE
            self.initialize_task(next_task)

        if self.current_task is None:
            return NodeStatus.IDLE

        # 检查超时
        if self.current_task.is_timeout():
            logger.warning(f"Task {self.current_task.task_id} timed out")
            self.current_task.finish(success=False)
            self.status = TaskStatus.FAILED
            return NodeStatus.FAILURE

        # 获取当前行为树
        assert self.current_task.task_type in self.behavior_trees
        bt = self.behavior_trees[self.current_task.task_type]

        # 更新状态
        bt.update_robot_state(robot_state)
        bt.update_world_state(world_state)

        # tick
        status = bt.tick()

        # 检查任务完成
        if status in (NodeStatus.SUCCESS, NodeStatus.FAILURE):
            self.current_task.finish(success=(status == NodeStatus.SUCCESS))
            if status == NodeStatus.SUCCESS:
                self.status = TaskStatus.COMPLETED
            else:
                self.status = TaskStatus.FAILED

        return status

    def abort_current(self) -> None:
        """中止当前任务"""
        if self.current_task:
            self.current_task.status = TaskStatus.ABORTED
            bt = self.behavior_trees.get(self.current_task.task_type)
            if bt:
                bt.reset()
            self.current_task = None
            self.status = TaskStatus.ABORTED

    def get_status(self) -> Dict[str, Any]:
        """获取规划器状态"""
        return {
            'name': self.name,
            'status': self.status,
            'current_task': self.current_task.task_id if self.current_task else None,
            'pending_tasks': sum(1 for t in self.tasks.values() if t.status == TaskStatus.IDLE),
            'total_tasks': len(self.tasks),
            'plan_version': self.plan_version,
            'registered_types': list(self.behavior_trees.keys()),
        }


# AGV 特定行为树节点
# -----------------------------------------------------------------------------

class AGVCheckBatteryCondition(ConditionNode):
    """AGV电量检查条件节点"""

    def __init__(self, min_battery: float = 0.2, name: str = "CheckBattery"):
        super().__init__(lambda bb: (bb.get_battery_level() or 0.0) >= min_battery, name)
        self.min_battery = min_battery


class AGVCheckSafeCondition(ConditionNode):
    """AGV安全检查条件节点"""

    def __init__(self, name: str = "CheckSafe"):
        super().__init__(lambda bb: bb.is_safe(), name)


class AGVCheckPositionReached(ConditionNode):
    """检查是否到达目标位置"""

    def __init__(self, threshold: float = 0.1, name: str = "CheckPositionReached"):
        def check(bb: Blackboard) -> bool:
            current_pos = bb.get_robot_position()
            target_pos = bb.goal_state.get('target_position')
            if current_pos is None or target_pos is None:
                return False
            distance = np.linalg.norm(current_pos - np.array(target_pos))
            return distance < threshold
        super().__init__(check, name)
        self.threshold = threshold


class AGVMoveToAction(ActionNode):
    """AGV移动到目标动作节点"""

    def __init__(self, speed: float = 0.5, name: str = "MoveTo"):
        super().__init__(name)
        self.speed = speed
        self._started = False

    def execute(self, blackboard: Blackboard) -> NodeStatus:
        target_pos = blackboard.goal_state.get('target_position')
        current_pos = blackboard.get_robot_position()

        if target_pos is None:
            return NodeStatus.FAILURE

        if not self._started:
            logger.info(f"AGV moving to target: {target_pos} at speed {self.speed}")
            self._started = True

        # 模拟移动：逐步接近目标位置（仿真模式）
        if current_pos is not None:
            target_arr = np.array(target_pos)
            distance = np.linalg.norm(current_pos - target_arr)
            if distance < 0.1:
                self._started = False
                return NodeStatus.SUCCESS
            # 每tick前进一段距离（0.1s/tick，speed m/s → speed*0.1 m/tick）
            step = min(self.speed * 0.5, distance)  # 加速模拟
            direction = (target_arr - current_pos) / distance
            new_pos = current_pos + direction * step
            blackboard.update_robot_state({'position': new_pos.tolist()})
        else:
            # 初始位置未知，直接跳到目标附近
            blackboard.update_robot_state({'position': np.array(target_pos).tolist()})
            self._started = False
            return NodeStatus.SUCCESS

        return NodeStatus.RUNNING

    def reset(self) -> None:
        self._started = False
        super().reset()


class AGVGraspAction(ActionNode):
    """AGV抓取动作节点"""

    def __init__(self, name: str = "Grasp", grasp_duration: float = 0.5, grasp_ticks: int = 5):
        super().__init__(name)
        self.grasp_duration = grasp_duration
        self.grasp_start_time: Optional[float] = None
        self.grasp_ticks = grasp_ticks
        self._tick_count: Optional[int] = None

    def execute(self, blackboard: Blackboard) -> NodeStatus:
        target_object = blackboard.goal_state.get('target_object')
        if not target_object:
            return NodeStatus.FAILURE

        if self.grasp_start_time is None:
            logger.info(f"Starting grasp on object: {target_object}")
            self.grasp_start_time = time.time()
            self._tick_count = 0
            return NodeStatus.RUNNING

        # Tick-based timing for simulation (ticks advance without real time)
        if self._tick_count is not None:
            self._tick_count += 1
            if self._tick_count >= self.grasp_ticks:
                logger.info(f"Completed grasp on object: {target_object}")
                self.grasp_start_time = None
                self._tick_count = None
                return NodeStatus.SUCCESS
            return NodeStatus.RUNNING

        # Wall-clock fallback for production
        if time.time() - self.grasp_start_time > self.grasp_duration:
            logger.info(f"Completed grasp on object: {target_object}")
            self.grasp_start_time = None
            return NodeStatus.SUCCESS

        return NodeStatus.RUNNING

    def reset(self) -> None:
        self.grasp_start_time = None
        self._tick_count = None
        super().reset()


class AGVReleaseAction(ActionNode):
    """AGV释放动作节点"""

    def __init__(self, name: str = "Release", release_duration: float = 0.3, release_ticks: int = 3):
        super().__init__(name)
        self.release_duration = release_duration
        self.release_start_time: Optional[float] = None
        self.release_ticks = release_ticks
        self._tick_count: Optional[int] = None

    def execute(self, blackboard: Blackboard) -> NodeStatus:
        if self.release_start_time is None:
            logger.info("Starting release")
            self.release_start_time = time.time()
            self._tick_count = 0
            return NodeStatus.RUNNING

        # Tick-based timing for simulation (ticks advance without real time)
        if self._tick_count is not None:
            self._tick_count += 1
            if self._tick_count >= self.release_ticks:
                logger.info("Completed release")
                self.release_start_time = None
                self._tick_count = None
                return NodeStatus.SUCCESS
            return NodeStatus.RUNNING

        # Wall-clock fallback for production
        if time.time() - self.release_start_time > self.release_duration:
            logger.info("Completed release")
            self.release_start_time = None
            return NodeStatus.SUCCESS

        return NodeStatus.RUNNING

    def reset(self) -> None:
        self.release_start_time = None
        self._tick_count = None
        super().reset()


class AGVTaskPlanner(EmbodiedTaskPlanner):
    """
    AGV专用任务规划器
    预设了常见AGV任务的行为树
    """

    # AGV五级规格对应的规划能力
    AGV_PLANNING_CAPABILITIES = {
        'S': {
            'max_planning_depth': 3,
            'max_concurrent_tasks': 1,
            'support_behavior_tree': True,
            'support_multi_agent': False,
            'replan_interval': 5.0,
        },
        'M': {
            'max_planning_depth': 6,
            'max_concurrent_tasks': 2,
            'support_behavior_tree': True,
            'support_multi_agent': False,
            'replan_interval': 2.0,
        },
        'L': {
            'max_planning_depth': 10,
            'max_concurrent_tasks': 3,
            'support_behavior_tree': True,
            'support_multi_agent': True,
            'replan_interval': 1.0,
        },
        'XL': {
            'max_planning_depth': 15,
            'max_concurrent_tasks': 4,
            'support_behavior_tree': True,
            'support_multi_agent': True,
            'replan_interval': 0.5,
        },
        'XXL': {
            'max_planning_depth': 20,
            'max_concurrent_tasks': 8,
            'support_behavior_tree': True,
            'support_multi_agent': True,
            'replan_interval': 0.2,
        },
    }

    def __init__(self, grade: str = "M", name: str = "AGVTaskPlanner"):
        super().__init__(name)
        self.grade = grade
        self.capabilities = self.AGV_PLANNING_CAPABILITIES.get(grade, self.AGV_PLANNING_CAPABILITIES['M'])
        self._setup_default_tasks()

    def _setup_default_tasks(self) -> None:
        """设置默认AGV任务行为树"""

        # 1. 导航到目标点任务
        nav_sequence = SequenceNode("NavigateToSequence")
        nav_sequence.add_children(
            AGVCheckSafeCondition(),
            AGVCheckBatteryCondition(0.2),
            AGVMoveToAction(),
            AGVCheckPositionReached(),
        )
        self.register_task_type('navigate', nav_sequence)

        # 2. 搬运任务 (去A点 → 抓取 → 去B点 → 释放)
        transport_sequence = SequenceNode("TransportSequence")
        transport_sequence.add_children(
            AGVCheckSafeCondition(),
            AGVCheckBatteryCondition(0.3),
            # 移动到拾取位置
            LambdaActionNode(lambda bb: self._set_pickup_target(bb), "SetPickupTarget"),
            AGVMoveToAction(),
            AGVCheckPositionReached(),
            # 抓取
            AGVGraspAction(),
            # 移动到放置位置
            LambdaActionNode(lambda bb: self._set_dropoff_target(bb), "SetDropoffTarget"),
            AGVMoveToAction(),
            AGVCheckPositionReached(),
            # 释放
            AGVReleaseAction(),
        )
        self.register_task_type('transport', transport_sequence)

        # 3. 巡逻任务 (重复经过多个点)
        patrol_sequence = SequenceNode("PatrolSequence")
        patrol_sequence.add_children(
            AGVCheckSafeCondition(),
            AGVMoveToAction(),
            AGVCheckPositionReached(),
        )
        patrol_root = RepeaterNode(patrol_sequence, times=3, name="PatrolRoot")
        self.register_task_type('patrol', patrol_root)

        # 4. 救援任务 (紧急移动到救援点并带回)
        rescue_sequence = SequenceNode("RescueSequence")
        rescue_sequence.add_children(
            AGVCheckSafeCondition(),
            AGVCheckBatteryCondition(0.4),  # 救援需要更多电量
            LambdaActionNode(lambda bb: self._set_rescue_target(bb), "SetRescueTarget"),
            AGVMoveToAction(speed=0.8),  # 快速移动
            AGVCheckPositionReached(threshold=0.2),
            AGVGraspAction(),
            LambdaActionNode(lambda bb: self._set_safe_zone_target(bb), "SetSafeZoneTarget"),
            AGVMoveToAction(speed=0.6),  # 小心移动
            AGVCheckPositionReached(threshold=0.2),
            AGVReleaseAction(),
        )
        self.register_task_type('rescue', rescue_sequence)

        # 5. 巡检任务 (移动到检查点)
        inspection_sequence = SequenceNode("InspectionSequence")
        inspection_sequence.add_children(
            AGVCheckSafeCondition(),
            AGVCheckBatteryCondition(0.3),
            LambdaActionNode(lambda bb: self._set_inspection_target(bb), "SetInspectionTarget"),
            AGVMoveToAction(),
            AGVCheckPositionReached(),
            # 模拟检查动作
            LambdaActionNode(lambda bb: NodeStatus.SUCCESS, "InspectAction"),
        )
        self.register_task_type('inspection', inspection_sequence)

        # 6. 装配任务 (搬运部件到装配点)
        assembly_sequence = SequenceNode("AssemblySequence")
        assembly_sequence.add_children(
            AGVCheckSafeCondition(),
            AGVCheckBatteryCondition(0.3),
            LambdaActionNode(lambda bb: self._set_pickup_target(bb), "SetPickupTarget"),
            AGVMoveToAction(),
            AGVCheckPositionReached(),
            AGVGraspAction(),
            LambdaActionNode(lambda bb: self._set_assembly_target(bb), "SetAssemblyTarget"),
            AGVMoveToAction(),
            AGVCheckPositionReached(),
            AGVReleaseAction(),
        )
        self.register_task_type('assembly', assembly_sequence)

        # 4. 多AGV协同任务
        if self.capabilities.get('support_multi_agent', False):
            self._setup_swarm_tasks()

    def _setup_swarm_tasks(self) -> None:
        """设置蜂群协同任务"""
        # 协同搬运任务
        swarm_transport = SequenceNode("SwarmTransportSequence")
        swarm_transport.add_children(
            AGVCheckSafeCondition(),
            AGVCheckBatteryCondition(0.4),
            # 协商分工
            AGVNegotiateRoleAction(),
            # 移动到协同位置
            AGVMoveToFormation(),
            AGVCheckFormationReached(),
            # 协同抓取
            AGVParallelGraspAction(),
            # 协同移动
            AGVCoordinatedMoveTo(),
            AGVCheckPositionReached(),
            # 协同释放
            AGVParallelReleaseAction(),
        )
        self.register_task_type('swarm_transport', swarm_transport)

        # 区域搜索任务
        area_search_sequence = ParallelNode(
            "AreaSearchParallel",
            success_policy=ParallelNode.Policy.REQUIRE_ALL,
            failure_policy=ParallelNode.Policy.REQUIRE_ANY
        )
        # 每个AGV负责子区域并行搜索
        # 实际由动态添加子节点完成
        self.register_task_type('area_search', area_search_sequence)

    def _set_pickup_target(self, blackboard: Blackboard) -> NodeStatus:
        """设置拾取目标点到黑板"""
        pickup_pos = blackboard.goal_state.get('pickup_position')
        if pickup_pos is not None:
            blackboard.set('current_target', pickup_pos)
            blackboard.goal_state['target_position'] = pickup_pos
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE

    def _set_dropoff_target(self, blackboard: Blackboard) -> NodeStatus:
        """设置放置目标点到黑板"""
        dropoff_pos = blackboard.goal_state.get('dropoff_position')
        if dropoff_pos is not None:
            blackboard.set('current_target', dropoff_pos)
            blackboard.goal_state['target_position'] = dropoff_pos
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE

    def _set_rescue_target(self, blackboard: Blackboard) -> NodeStatus:
        """设置救援目标点到黑板"""
        rescue_pos = blackboard.goal_state.get('rescue_position', blackboard.goal_state.get('target_position'))
        if rescue_pos is not None:
            blackboard.set('current_target', rescue_pos)
            blackboard.goal_state['target_position'] = rescue_pos
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE

    def _set_safe_zone_target(self, blackboard: Blackboard) -> NodeStatus:
        """设置安全区目标点到黑板"""
        safe_zone_pos = blackboard.goal_state.get('safe_zone_position', [5.0, 5.0, 0.0])
        blackboard.set('current_target', safe_zone_pos)
        blackboard.goal_state['target_position'] = safe_zone_pos
        return NodeStatus.SUCCESS

    def _set_inspection_target(self, blackboard: Blackboard) -> NodeStatus:
        """设置巡检目标点到黑板"""
        inspection_pos = blackboard.goal_state.get('inspection_position', blackboard.goal_state.get('target_position'))
        if inspection_pos is not None:
            blackboard.set('current_target', inspection_pos)
            blackboard.goal_state['target_position'] = inspection_pos
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE

    def _set_assembly_target(self, blackboard: Blackboard) -> NodeStatus:
        """设置装配目标点到黑板"""
        assembly_pos = blackboard.goal_state.get('assembly_position', blackboard.goal_state.get('dropoff_position'))
        if assembly_pos is not None:
            blackboard.set('current_target', assembly_pos)
            blackboard.goal_state['target_position'] = assembly_pos
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE

    def get_capabilities(self) -> Dict[str, Any]:
        """获取当前AGV等级的规划能力"""
        return {
            'grade': self.grade,
            **self.capabilities,
        }

    def plan_task(
        self,
        task_type: str,
        target: Optional[str] = None,
        grade: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        规划任务的接口方法（供 EmbodiedPipeline 调用）

        Args:
            task_type: 任务类型 (transport/patrol/rescue/navigate)
            target: 目标位置描述或标识
            grade: AGV等级（兼容参数，此处忽略）
            **kwargs: 额外参数 (payload等)

        Returns:
            任务规划结果字典，包含 behavior_tree 和 task_config
        """
        # 构建目标位置
        target_position = None
        if target is not None:
            # 尝试解析目标标识为坐标
            target_position = self._resolve_target_position(target)

        # 创建具身任务
        task = EmbodiedTask(
            task_id=f"pipeline_{int(time.time() * 1000)}",
            task_type=task_type,
            goal_description=f"{task_type} task to {target}",
            target_position=target_position,
            target_object=kwargs.get('payload', {}).get('object'),
            priority=kwargs.get('priority', 2),
            timeout=kwargs.get('timeout', 300.0),
        )

        # 添加到规划器
        self.add_task(task)

        # 初始化任务获取行为树
        bt = self.initialize_task(task)

        # 构建规划结果
        result = {
            'task_id': task.task_id,
            'task_type': task_type,
            'target': target,
            'target_position': target_position.tolist() if target_position is not None else None,
            'behavior_tree': bt,
            'grade': grade or self.grade,
            'status': task.status.value if hasattr(task.status, 'value') else str(task.status),
        }

        logger.info(f"Planned task {task.task_id} ({task_type}) -> {target}")
        return result

    def _resolve_target_position(self, target: str) -> Optional[np.ndarray]:
        """将目标标识解析为位置坐标"""
        # 已知的导航点预定义坐标
        known_points = {
            'station_A': np.array([10.0, 0.0, 0.0]),
            'station_B': np.array([20.0, 0.0, 0.0]),
            'station_C': np.array([30.0, 0.0, 0.0]),
            'entrance': np.array([0.0, 0.0, 0.0]),
            'exit': np.array([40.0, 0.0, 0.0]),
            'charging': np.array([2.0, 0.0, 0.0]),
            'warehouse_entrance': np.array([0.0, 0.0, 0.0]),
            'loading_dock': np.array([5.0, 0.0, 0.0]),
            'unloading_dock': np.array([15.0, 0.0, 0.0]),
        }
        # 大小写不敏感匹配
        target_lower = target.lower().replace('-', '_').replace(' ', '_')
        for key, pos in known_points.items():
            if key.lower() == target_lower or target_lower in key.lower():
                return pos.copy()
        # 未知目标，返回默认坐标
        logger.debug(f"Unknown target '{target}', using default position")
        return np.array([10.0, 0.0, 0.0])


# 多AGV蜂群协同特定节点
# -----------------------------------------------------------------------------

class AGVNegotiateRoleAction(ActionNode):
    """AGV角色协商动作节点 - 蜂群协同"""

    def __init__(self, name: str = "NegotiateRole"):
        super().__init__(name)
        self._negotiation_started = False

    def execute(self, blackboard: Blackboard) -> NodeStatus:
        """协商角色分配"""
        if not self._negotiation_started:
            logger.info("Starting role negotiation for swarm task")
            self._negotiation_started = True
            # 在实际系统中，这里会和其他AGV通信协商
            # 这里简化为直接成功
            return NodeStatus.RUNNING

        # 模拟协商完成
        self._negotiation_started = False
        roles = blackboard.goal_state.get('swarm_roles', {})
        logger.info(f"Role negotiation completed, assigned roles: {roles}")
        return NodeStatus.SUCCESS

    def reset(self) -> None:
        self._negotiation_started = False
        super().reset()


class AGVMoveToFormationAction(ActionNode):
    """移动到协同阵位动作节点"""

    def __init__(self, name: str = "MoveToFormation"):
        super().__init__(name)
        self._started = False

    def execute(self, blackboard: Blackboard) -> NodeStatus:
        formation_pos = blackboard.goal_state.get('formation_position')
        if formation_pos is None:
            return NodeStatus.FAILURE

        if not self._started:
            logger.info(f"Moving to formation position: {formation_pos}")
            self._started = True

        current_pos = blackboard.get_robot_position()
        if current_pos is not None:
            distance = np.linalg.norm(current_pos - np.array(formation_pos))
            if distance < 0.15:  # 阵位容差稍大
                self._started = False
                return NodeStatus.SUCCESS

        return NodeStatus.RUNNING

    def reset(self) -> None:
        self._started = False
        super().reset()


class AGVCheckFormationReachedCondition(ConditionNode):
    """检查是否到达协同阵位"""

    def __init__(self, threshold: float = 0.15, name: str = "CheckFormationReached"):
        def check(bb: Blackboard) -> bool:
            current_pos = bb.get_robot_position()
            formation_pos = bb.goal_state.get('formation_position')
            if current_pos is None or formation_pos is None:
                return False
            distance = np.linalg.norm(current_pos - np.array(formation_pos))
            return distance < threshold
        super().__init__(check, name)


class AGVParallelGraspAction(ActionNode):
    """并行抓取动作 - 多AGV协同抓取重物"""

    def __init__(self, name: str = "ParallelGrasp"):
        super().__init__(name)
        self._start_time: Optional[float] = None

    def execute(self, blackboard: Blackboard) -> NodeStatus:
        if self._start_time is None:
            logger.info("Starting parallel coordinated grasp")
            self._start_time = time.time()
            return NodeStatus.RUNNING

        # 同步抓取需要协调时间
        if time.time() - self._start_time > 3.0:
            logger.info("Parallel grasp completed")
            self._start_time = None
            return NodeStatus.SUCCESS

        return NodeStatus.RUNNING

    def reset(self) -> None:
        self._start_time = None
        super().reset()


class AGVCoordinatedMoveToAction(ActionNode):
    """协同移动动作 - 保持阵位移动"""

    def __init__(self, name: str = "CoordinatedMoveTo"):
        super().__init__(name)
        self._started = False

    def execute(self, blackboard: Blackboard) -> NodeStatus:
        target = blackboard.goal_state.get('target_position')
        if target is None:
            return NodeStatus.FAILURE

        if not self._started:
            logger.info(f"Starting coordinated movement to: {target}")
            self._started = True

        current_pos = blackboard.get_robot_position()
        if current_pos is not None:
            distance = np.linalg.norm(current_pos - np.array(target))
            if distance < 0.2:
                self._started = False
                return NodeStatus.SUCCESS

        return NodeStatus.RUNNING

    def reset(self) -> None:
        self._started = False
        super().reset()


class AGVParallelReleaseAction(ActionNode):
    """并行释放动作"""

    def __init__(self, name: str = "ParallelRelease"):
        super().__init__(name)
        self._start_time: Optional[float] = None

    def execute(self, blackboard: Blackboard) -> NodeStatus:
        if self._start_time is None:
            logger.info("Starting parallel coordinated release")
            self._start_time = time.time()
            return NodeStatus.RUNNING

        if time.time() - self._start_time > 1.5:
            logger.info("Parallel release completed")
            self._start_time = None
            return NodeStatus.SUCCESS

        return NodeStatus.RUNNING

    def reset(self) -> None:
        self._start_time = None
        super().reset()


# 别名简化注册
AGVMoveToFormation = AGVMoveToFormationAction
AGVCheckFormationReached = AGVCheckFormationReachedCondition
AGVNegotiateRole = AGVNegotiateRoleAction
AGVParallelGrasp = AGVParallelGraspAction
AGVCoordinatedMoveTo = AGVCoordinatedMoveToAction
AGVParallelRelease = AGVParallelReleaseAction


class MultiAGVBehaviorTreePlanner:
    """多AGV行为树规划器 - 蜂群协同"""

    def __init__(self, num_agvs: int, grade: str = "L"):
        self.num_agvs = num_agvs
        self.grade = grade
        self.planners: Dict[str, AGVTaskPlanner] = {}
        self.shared_blackboard = Blackboard()

    def register_agv(self, agv_id: str, planner: AGVTaskPlanner) -> None:
        """注册一个AGV规划器"""
        self.planners[agv_id] = planner

    def coordinate_swarm_task(self, task_type: str, goal: Dict[str, Any]) -> None:
        """协调蜂群任务"""
        # 设置共享目标
        for agv_id, planner in self.planners.items():
            # 每个AGV获得子任务
            if task_type == 'swarm_transport':
                # 分配不同的阵位
                formation_positions = goal.get('formation_positions', {})
                if agv_id in formation_positions:
                    self.shared_blackboard.goal_state['formation_position'] = formation_positions[agv_id]

        logger.info(f"Coordinated swarm task {task_type} for {self.num_agvs} AGVs")


# 配置驱动的行为树构建器 (实现见下方 _build_node_from_config + create_behavior_tree_from_dict)

# 节点类型注册表
_NODE_TYPE_REGISTRY: Dict[str, type] = {
    # Composite
    'sequence': SequenceNode,
    'selector': SelectorNode,
    'parallel': ParallelNode,
    # Decorator
    'repeater': RepeaterNode,
    'until_fail': UntilFailNode,
    'until_success': UntilSuccessNode,
    'inverter': InverterNode,
    # Condition
    'condition': ConditionNode,
    # Action
    'action': ActionNode,
    'lambda': LambdaActionNode,
    # AGV专用
    'agv_check_battery': AGVCheckBatteryCondition,
    'agv_check_safe': AGVCheckSafeCondition,
    'agv_check_position': AGVCheckPositionReached,
    'agv_check_formation': AGVCheckFormationReachedCondition,
    'agv_move_to': AGVMoveToAction,
    'agv_grasp': AGVGraspAction,
    'agv_release': AGVReleaseAction,
    'agv_negotiate_role': AGVNegotiateRoleAction,
    'agv_move_to_formation': AGVMoveToFormationAction,
    'agv_parallel_grasp': AGVParallelGraspAction,
    'agv_coordinated_move': AGVCoordinatedMoveToAction,
    'agv_parallel_release': AGVParallelReleaseAction,
}


def _get_node_name(config: Dict[str, Any], index: int) -> str:
    """从配置中获取节点名称"""
    if 'name' in config:
        return str(config['name'])
    if 'type' in config:
        return f"{config['type']}_{index}"
    return f"node_{index}"


def _build_node_from_config(config: Dict[str, Any], index: int = 0) -> BTNode:
    """从配置字典递归构建行为树节点"""
    node_type = config.get('type', '').lower()
    node_name = _get_node_name(config, index)

    # 处理复合节点（序列/选择/并行）
    if node_type in ('sequence', 'selector', 'parallel'):
        children_configs = config.get('children', config.get('childs', []))
        node_class = _NODE_TYPE_REGISTRY.get(node_type, SequenceNode)
        if node_type == 'parallel':
            # ParallelNode 使用 success_policy/failure_policy (enum Policy)
            success_threshold = config.get('success_threshold', len(children_configs))
            if success_threshold >= len(children_configs):
                policy = ParallelNode.Policy.REQUIRE_ALL
            else:
                policy = ParallelNode.Policy.REQUIRE_ANY
            node = node_class(name=node_name, success_policy=policy)
        else:
            node = node_class(name=node_name)
        for i, child_config in enumerate(children_configs):
            child_node = _build_node_from_config(child_config, i)
            node.add_child(child_node)
        return node

    # 处理装饰器节点
    if node_type in ('repeater', 'until_fail', 'until_success', 'inverter'):
        children = config.get('children', config.get('childs', []))
        # 先构建子节点，因为装饰器需要child作为构造参数
        child_node = _build_node_from_config(children[0], 0) if children else LambdaActionNode(lambda bb: NodeStatus.SUCCESS)
        decorator_class = _NODE_TYPE_REGISTRY.get(node_type, RepeaterNode)
        if node_type == 'repeater':
            # Check params first, then top-level config
            params = config.get('params', {})
            times = params.get('num_repeats', config.get('num_repeats', config.get('times', -1)))
            node = decorator_class(child=child_node, times=times, name=node_name)
        elif node_type == 'until_fail':
            node = decorator_class(child=child_node, name=node_name)
        elif node_type == 'until_success':
            node = decorator_class(child=child_node, name=node_name)
        elif node_type == 'inverter':
            node = decorator_class(child=child_node, name=node_name)
        else:
            node = decorator_class(child=child_node, name=node_name)
        return node

    # 处理通用条件节点
    if node_type == 'condition':
        condition_lambda = config.get('condition')
        if callable(condition_lambda):
            return ConditionNode(condition_lambda, name=node_name)
        # 从参数构建简单条件
        condition_type = config.get('condition_type', '')
        if condition_type == 'battery':
            min_battery = config.get('min_battery', 0.2)
            return AGVCheckBatteryCondition(min_battery=min_battery, name=node_name)
        elif condition_type == 'safe':
            return AGVCheckSafeCondition(name=node_name)
        elif condition_type == 'position':
            threshold = config.get('threshold', 0.1)
            return AGVCheckPositionReached(threshold=threshold, name=node_name)
        elif condition_type == 'formation':
            threshold = config.get('threshold', 0.15)
            return AGVCheckFormationReachedCondition(threshold=threshold, name=node_name)
        # 默认：始终返回成功
        return ConditionNode(lambda bb: True, name=node_name)

    # 处理Lambda动作节点
    if node_type in ('lambda', 'action'):
        action_lambda = config.get('action')
        if callable(action_lambda):
            return LambdaActionNode(action_lambda, name=node_name)
        # 从参数构建AGV动作
        action_name = config.get('action_name', '')
        if action_name == 'move_to':
            speed = config.get('speed', 0.5)
            return AGVMoveToAction(speed=speed, name=node_name)
        elif action_name == 'grasp':
            return AGVGraspAction(name=node_name)
        elif action_name == 'release':
            return AGVReleaseAction(name=node_name)
        elif action_name == 'negotiate_role':
            return AGVNegotiateRoleAction(name=node_name)
        elif action_name == 'move_to_formation':
            return AGVMoveToFormationAction(name=node_name)
        elif action_name == 'parallel_grasp':
            return AGVParallelGraspAction(name=node_name)
        elif action_name == 'coordinated_move':
            return AGVCoordinatedMoveToAction(name=node_name)
        elif action_name == 'parallel_release':
            return AGVParallelReleaseAction(name=node_name)
        # 默认：返回成功（用于占位）
        return LambdaActionNode(lambda bb: NodeStatus.SUCCESS, name=node_name)

    # 处理AGV专用节点快捷方式
    agv_node_type = config.get('type', '').lower()
    if agv_node_type in _NODE_TYPE_REGISTRY:
        node_class = _NODE_TYPE_REGISTRY[agv_node_type]
        params = config.get('params', {})
        try:
            return node_class(name=node_name, **params)
        except TypeError:
            try:
                return node_class(**params)
            except TypeError:
                return node_class(name=node_name)

    # 未知类型，默认创建Lambda成功节点
    logger.warning(f"Unknown node type '{node_type}' in config, creating fallback SUCCESS node")
    return LambdaActionNode(lambda bb: NodeStatus.SUCCESS, name=node_name)


def create_behavior_tree_from_dict(config: Dict[str, Any]) -> BTNode:
    """
    从字典配置创建行为树 - 用于反序列化/配置文件驱动

    支持的节点类型:
        - composite: sequence, selector, parallel
        - decorator: repeater, until_fail, until_success, inverter
        - condition: condition (通用), agv_check_battery, agv_check_safe, agv_check_position, agv_check_formation
        - action: lambda, agv_move_to, agv_grasp, agv_release, agv_negotiate_role,
                  agv_move_to_formation, agv_parallel_grasp, agv_coordinated_move, agv_parallel_release

    配置格式:
        {
            "type": "sequence",           # 节点类型
            "name": "MySequence",         # 可选，节点名称
            "children": [                 # 子节点配置
                {"type": "agv_check_safe", "name": "SafetyCheck"},
                {
                    "type": "selector",
                    "name": "MoveOrWait",
                    "children": [
                        {"type": "agv_move_to", "params": {"speed": 0.5}},
                        {"type": "lambda", "action": lambda bb: NodeStatus.SUCCESS, "name": "Wait"},
                    ]
                },
            ],
            # parallel 节点专用:
            "success_threshold": 2,       # parallel 节点成功阈值
            # repeater 节点专用:
            "num_repeats": 3,             # 重复次数
        }

    示例 - 导航任务:
        config = {
            "type": "sequence",
            "name": "NavigateTask",
            "children": [
                {"type": "agv_check_safe", "name": "SafetyCheck"},
                {"type": "agv_check_battery", "params": {"min_battery": 0.2}},
                {"type": "agv_move_to", "params": {"speed": 0.8}},
                {"type": "agv_check_position", "params": {"threshold": 0.15}},
            ]
        }
        root = create_behavior_tree_from_dict(config)
        bt = BehaviorTree(root, name="Navigate")

    示例 - 搬运任务:
        config = {
            "type": "sequence",
            "name": "TransportTask",
            "children": [
                {"type": "agv_check_safe"},
                {"type": "agv_check_battery", "params": {"min_battery": 0.3}},
                {"type": "agv_move_to"},
                {"type": "agv_check_position"},
                {"type": "agv_grasp"},
                {"type": "agv_move_to"},
                {"type": "agv_release"},
            ]
        }
    """
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a dict, got {type(config).__name__}")
    if 'type' not in config:
        raise ValueError("Config must have a 'type' field")

    return _build_node_from_config(config, index=0)


def serialize_behavior_tree(node: BTNode) -> Dict[str, Any]:
    """
    将行为树节点序列化为字典（反向操作）

    注意: 不是所有节点都能完美序列化（如Lambda函数的源码），
    此函数主要用于可配置节点的序列化
    """
    result: Dict[str, Any] = {
        'type': node.__class__.__name__,
        'name': node.name,
    }

    if isinstance(node, CompositeNode):
        result['children'] = [serialize_behavior_tree(child) for child in node.children]
        if isinstance(node, ParallelNode):
            result['success_threshold'] = node.success_threshold

    if isinstance(node, DecoratorNode):
        if node.children:
            result['children'] = [serialize_behavior_tree(node.children[0])]
        if isinstance(node, RepeaterNode):
            result['num_repeats'] = node.num_repeats

    if isinstance(node, ConditionNode):
        if isinstance(node, AGVCheckBatteryCondition):
            result['type'] = 'agv_check_battery'
            result['params'] = {'min_battery': node.min_battery}
        elif isinstance(node, AGVCheckSafeCondition):
            result['type'] = 'agv_check_safe'
        elif isinstance(node, AGVCheckPositionReached):
            result['type'] = 'agv_check_position'
            result['params'] = {'threshold': node.threshold}
        elif isinstance(node, AGVCheckFormationReachedCondition):
            result['type'] = 'agv_check_formation'
            result['params'] = {'threshold': node.threshold}

    if isinstance(node, ActionNode):
        if isinstance(node, AGVMoveToAction):
            result['type'] = 'agv_move_to'
            result['params'] = {'speed': node.speed}
        elif isinstance(node, AGVGraspAction):
            result['type'] = 'agv_grasp'
        elif isinstance(node, AGVReleaseAction):
            result['type'] = 'agv_release'
        elif isinstance(node, LambdaActionNode):
            result['type'] = 'lambda'
            result['note'] = 'Lambda function - cannot be serialized'

    return result


def create_task_bt_from_config(task_config: Dict[str, Any]) -> BehaviorTree:
    """
    从任务配置创建完整行为树（快捷函数）

    任务配置格式:
        {
            "task_type": "navigate",
            "task_name": "MyNavigateTask",
            "tree": {
                "type": "sequence",
                "children": [...]
            }
        }
    """
    tree_config = task_config.get('tree', task_config)
    root = create_behavior_tree_from_dict(tree_config)
    task_name = task_config.get('task_name', task_config.get('task_type', 'Task'))
    return BehaviorTree(root, name=task_name)


# ============================================================================
# 配置文件加载器（支持YAML/JSON）
# ============================================================================

def load_behavior_tree_from_yaml(yaml_path: str) -> BTNode:
    """从YAML文件加载行为树配置并构建树"""
    try:
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return create_behavior_tree_from_dict(config)
    except ImportError:
        raise RuntimeError("PyYAML is required for YAML loading. Install with: pip install pyyaml")


def load_behavior_tree_from_json(json_path: str) -> BTNode:
    """从JSON文件加载行为树配置并构建树"""
    import json
    with open(json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return create_behavior_tree_from_dict(config)


# ============================================================================
# 多AGV协同行为树管理器
# ============================================================================


class MultiAGVBehaviorTreeManager:
    """
    多AGV协同行为树管理器 - 支持多AGV任务分配和协同执行
    
    功能:
    - 全局任务分解
    - AGV任务分配 (基于能力/负载/位置)
    - 协同行为树执行
    - 跨AGV任务同步
    - 冲突检测与解决
    """

    def __init__(self, agv_registry: Dict[str, Dict[str, Any]]):
        """
        初始化多AGV管理器
        
        Args:
            agv_registry: AGV注册信息，包含每个AGV的能力、位置、负载等
        """
        self.agv_registry = agv_registry
        self.agv_bt_instances: Dict[str, BehaviorTree] = {}
        self.global_task_queue: List[Dict[str, Any]] = []
        self.running_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: List[Dict[str, Any]] = []
        self.collision_avoidance_enabled = True
        self.task_sync_points: Dict[str, Set[str]] = {}

    def register_agv(self, agv_id: str, capabilities: List[str], position: np.ndarray, load_level: float = 0.0) -> None:
        """注册AGV到管理器"""
        self.agv_registry[agv_id] = {
            'capabilities': capabilities,
            'position': position,
            'load_level': load_level,
            'available': True,
            'current_task': None,
        }

    def add_global_task(self, task_config: Dict[str, Any]) -> str:
        """添加全局任务到队列"""
        import uuid
        task_id = str(uuid.uuid4())
        task_config['task_id'] = task_id
        self.global_task_queue.append(task_config)
        return task_id

    def allocate_tasks(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        任务分配算法 - 将全局任务分配给最合适的AGV
        
        分配优先级:
        1. AGV是否具备任务所需能力
        2. 距离任务起点的距离
        3. 当前负载水平
        4. 历史任务成功率
        
        Returns:
            列表 of (agv_id, task_id, task_config)
        """
        allocations = []
        available_agvs = [agv_id for agv_id, info in self.agv_registry.items() if info['available']]
        
        for task in self.global_task_queue[:]:
            required_capabilities = task.get('required_capabilities', [])
            task_position = task.get('start_position', np.zeros(3))
            
            # 筛选具备所需能力的AGV
            eligible_agvs = []
            for agv_id in available_agvs:
                agv_info = self.agv_registry[agv_id]
                if all(cap in agv_info['capabilities'] for cap in required_capabilities):
                    # 计算分配得分
                    distance = np.linalg.norm(agv_info['position'] - task_position)
                    score = (1.0 / (distance + 0.1)) * (1.0 - agv_info['load_level'] * 0.5)
                    eligible_agvs.append((score, agv_id))
            
            if eligible_agvs:
                # 选择得分最高的AGV
                eligible_agvs.sort(reverse=True, key=lambda x: x[0])
                best_agv_id = eligible_agvs[0][1]
                
                allocations.append((best_agv_id, task['task_id'], task))
                self.global_task_queue.remove(task)
                self.agv_registry[best_agv_id]['available'] = False
                self.agv_registry[best_agv_id]['current_task'] = task['task_id']
                self.running_tasks[task['task_id']] = {
                    'agv_id': best_agv_id,
                    'task_config': task,
                    'start_time': time.time(),
                    'status': 'RUNNING',
                }
        
        return allocations

    def create_agv_behavior_tree(self, agv_id: str, task_config: Dict[str, Any]) -> BehaviorTree:
        """为指定AGV创建任务行为树"""
        bt = create_task_bt_from_config(task_config)
        bt.set_blackboard_value('agv_id', agv_id)
        bt.set_blackboard_value('task_id', task_config['task_id'])
        self.agv_bt_instances[agv_id] = bt
        return bt

    def step_all(self, delta_time: float = 0.01) -> Dict[str, Any]:
        """执行所有AGV的行为树单步更新"""
        results = {}
        completed_task_ids = []
        
        # 先执行任务分配
        new_allocations = self.allocate_tasks()
        for agv_id, task_id, task_config in new_allocations:
            self.create_agv_behavior_tree(agv_id, task_config)
            logger.info(f"Allocated task {task_id} to AGV {agv_id}")
        
        # 执行所有AGV的行为树
        for agv_id, bt in self.agv_bt_instances.items():
            if not self.agv_registry[agv_id]['available']:
                status = bt.tick()
                results[agv_id] = {
                    'task_id': self.agv_registry[agv_id]['current_task'],
                    'status': status.name,
                    'blackboard': bt.get_blackboard_copy(),
                }
                
                # 检查任务是否完成
                if status in [NodeStatus.SUCCESS, NodeStatus.FAILURE]:
                    task_id = self.agv_registry[agv_id]['current_task']
                    if task_id in self.running_tasks:
                        self.running_tasks[task_id]['status'] = 'COMPLETED' if status == NodeStatus.SUCCESS else 'FAILED'
                        self.running_tasks[task_id]['end_time'] = time.time()
                        self.running_tasks[task_id]['duration'] = self.running_tasks[task_id]['end_time'] - self.running_tasks[task_id]['start_time']
                        self.completed_tasks.append(self.running_tasks[task_id])
                        completed_task_ids.append(task_id)
                    
                    # 标记AGV为可用
                    self.agv_registry[agv_id]['available'] = True
                    self.agv_registry[agv_id]['current_task'] = None
        
        # 清理已完成的任务
        for task_id in completed_task_ids:
            del self.running_tasks[task_id]
        
        # 碰撞检测与避免
        if self.collision_avoidance_enabled:
            self._run_collision_avoidance()
        
        return {
            'agv_results': results,
            'new_allocations': len(new_allocations),
            'running_tasks': len(self.running_tasks),
            'completed_tasks_total': len(self.completed_tasks),
            'queued_tasks': len(self.global_task_queue),
        }

    def _run_collision_avoidance(self) -> None:
        """运行AGV之间的碰撞避免算法"""
        # 收集所有AGV的位置和速度
        agv_states = []
        for agv_id, info in self.agv_registry.items():
            if not info['available'] and 'position' in info:
                bt = self.agv_bt_instances.get(agv_id)
                if bt:
                    velocity = bt.get_blackboard_value('current_velocity', np.zeros(2))
                    agv_states.append({
                        'agv_id': agv_id,
                        'position': info['position'],
                        'velocity': velocity,
                    })
        
        # 检查两两之间的距离
        for i in range(len(agv_states)):
            for j in range(i + 1, len(agv_states)):
                a = agv_states[i]
                b = agv_states[j]
                distance = np.linalg.norm(a['position'][:2] - b['position'][:2])
                
                if distance < 1.0:  # 安全距离阈值
                    # 距离过近，触发减速/停止
                    logger.warning(f"AGV {a['agv_id']} and AGV {b['agv_id']} are too close ({distance:.2f}m), triggering collision avoidance")
                    
                    for agv_state in [a, b]:
                        bt = self.agv_bt_instances.get(agv_state['agv_id'])
                        if bt:
                            # 降低目标速度
                            current_target_speed = bt.get_blackboard_value('target_speed', 1.0)
                            bt.set_blackboard_value('target_speed', current_target_speed * 0.5)
                            
                            if distance < 0.5:
                                # 紧急停止
                                bt.set_blackboard_value('emergency_stop', True)

    def get_global_status(self) -> Dict[str, Any]:
        """获取全局多AGV系统状态"""
        total_agvs = len(self.agv_registry)
        available_agvs = sum(1 for info in self.agv_registry.values() if info['available'])
        busy_agvs = total_agvs - available_agvs
        
        success_count = sum(1 for task in self.completed_tasks if task['status'] == 'COMPLETED')
        failure_count = len(self.completed_tasks) - success_count
        success_rate = success_count / len(self.completed_tasks) if self.completed_tasks else 0.0
        
        return {
            'total_agvs': total_agvs,
            'available_agvs': available_agvs,
            'busy_agvs': busy_agvs,
            'queued_tasks': len(self.global_task_queue),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'task_success_rate': success_rate,
            'running_task_details': self.running_tasks.copy(),
        }

    def cancel_task(self, task_id: str) -> bool:
        """取消指定任务"""
        if task_id in self.running_tasks:
            task_info = self.running_tasks[task_id]
            agv_id = task_info['agv_id']
            
            # 停止AGV的行为树
            if agv_id in self.agv_bt_instances:
                bt = self.agv_bt_instances[agv_id]
                bt.reset()
                bt.set_blackboard_value('emergency_stop', True)
            
            # 标记AGV为可用
            self.agv_registry[agv_id]['available'] = True
            self.agv_registry[agv_id]['current_task'] = None
            
            # 从运行任务中移除
            del self.running_tasks[task_id]
            logger.info(f"Cancelled task {task_id} on AGV {agv_id}")
            return True
        
        # 检查是否在队列中
        for i, task in enumerate(self.global_task_queue):
            if task['task_id'] == task_id:
                self.global_task_queue.pop(i)
                logger.info(f"Removed task {task_id} from queue")
                return True
        
        return False


__all__ += [
    'DynamicBTReplanner',
    'ReplanTrigger',
    'SwarmTaskAllocator',
    'MultiAGVBehaviorTreeManager',
    'BehaviorTreeBuilder',
]

# 行为树构建器别名 - 兼容旧代码
BehaviorTreeBuilder = create_behavior_tree_from_dict


# ============================== AGV-specific Condition & Action Nodes ==============================
class IsAtTarget(ConditionNode):
    """检查机器人是否到达目标位置"""
    def __init__(self, target: Tuple[float, float], tolerance: float = 0.1):
        self.target = target
        self.tolerance = tolerance
        # 构造condition函数
        def condition(blackboard: Blackboard) -> bool:
            # 兼容两种blackboard格式：旧版直接存current_x/current_y，新版存robot_state.position
            if hasattr(blackboard, 'robot_state'):
                current_pos = blackboard.robot_state.get('position', (0.0, 0.0))
            else:
                current_pos = (blackboard.get('current_x', 0.0), blackboard.get('current_y', 0.0))
            distance = math.hypot(current_pos[0] - self.target[0], current_pos[1] - self.target[1])
            return distance <= self.tolerance
        super().__init__(condition, name=f"IsAtTarget({target}, tol={tolerance})")


class IsBatteryLow(ConditionNode):
    """检查电池电量是否过低"""
    def __init__(self, threshold: float = 20.0):
        self.threshold = threshold
        # 构造condition函数
        def condition(blackboard: Blackboard) -> bool:
            # 兼容两种blackboard格式
            if hasattr(blackboard, 'robot_state'):
                battery_level = blackboard.robot_state.get('battery_level', 100.0)
            else:
                battery_level = blackboard.get('battery_level', 100.0)
            # 兼容百分比和0-1两种格式
            if battery_level <= 1.0:
                battery_level *= 100.0
            return battery_level <= self.threshold
        super().__init__(condition, name=f"IsBatteryLow(threshold={threshold}%)")


class MoveTo(ActionNode):
    """移动机器人到目标位置"""
    def __init__(self, target: Tuple[float, float], speed: float = 1.0):
        super().__init__(name=f"MoveTo({target}, speed={speed})")
        self.target = target
        self.speed = speed
        self.start_time = None
    
    def execute(self, blackboard: Blackboard) -> NodeStatus:
        if self.start_time is None:
            self.start_time = time.time()
            logger.info(f"开始移动到目标位置: {self.target}")
        
        # 兼容两种blackboard格式
        if hasattr(blackboard, 'robot_state'):
            current_pos = blackboard.robot_state.get('position', (0.0, 0.0))
        else:
            current_pos = (blackboard.get('current_x', 0.0), blackboard.get('current_y', 0.0))
        
        distance = math.hypot(current_pos[0] - self.target[0], current_pos[1] - self.target[1])
        
        # 仿真移动: 每秒移动speed距离
        time_elapsed = time.time() - self.start_time
        expected_distance_moved = self.speed * time_elapsed
        
        if expected_distance_moved >= distance:
            # 到达目标
            if hasattr(blackboard, 'robot_state'):
                blackboard.robot_state['position'] = self.target
            else:
                blackboard['current_x'] = self.target[0]
                blackboard['current_y'] = self.target[1]
            logger.info(f"到达目标位置: {self.target}")
            return NodeStatus.SUCCESS
        else:
            # 更新当前位置
            direction = ((self.target[0] - current_pos[0])/distance, (self.target[1] - current_pos[1])/distance) if distance > 0 else (0, 0)
            new_pos = (current_pos[0] + direction[0] * expected_distance_moved, current_pos[1] + direction[1] * expected_distance_moved)
            if hasattr(blackboard, 'robot_state'):
                blackboard.robot_state['position'] = new_pos
            else:
                blackboard['current_x'] = new_pos[0]
                blackboard['current_y'] = new_pos[1]
                # 设置速度供测试使用
                blackboard['desired_velocity'] = self.speed
                blackboard['desired_omega'] = 0.0
            return NodeStatus.RUNNING


class Pickup(ActionNode):
    """抓取指定位置的物体"""
    def __init__(self, object_position: Tuple[float, float]):
        super().__init__(name=f"Pickup({object_position})")
        self.object_position = object_position
        self.start_time = None
        self.pickup_duration = 2.0  # 抓取耗时2秒
    
    def execute(self, blackboard: Blackboard) -> NodeStatus:
        if self.start_time is None:
            # 先检查是否在物体位置附近
            if hasattr(blackboard, 'robot_state'):
                current_pos = blackboard.robot_state.get('position', (0.0, 0.0))
            else:
                current_pos = (blackboard.get('current_x', 0.0), blackboard.get('current_y', 0.0))
            distance = math.hypot(current_pos[0] - self.object_position[0], current_pos[1] - self.object_position[1])
            if distance > 0.1:
                logger.error(f"无法抓取: 距离物体位置{self.object_position}过远 ({distance}m)")
                return NodeStatus.FAILURE
            self.start_time = time.time()
            logger.info(f"开始抓取物体: {self.object_position}")
        
        if time.time() - self.start_time >= self.pickup_duration:
            logger.info(f"成功抓取物体: {self.object_position}")
            if hasattr(blackboard, 'robot_state'):
                blackboard.robot_state['carried_object'] = self.object_position
            else:
                blackboard['gripper_command'] = 'close'
            return NodeStatus.SUCCESS
        else:
            return NodeStatus.RUNNING


# ============================================================================
# 动态行为树重规划器 - Dynamic BT Replanner
# ============================================================================


class ReplanTrigger(Enum):
    """重规划触发原因"""
    TASK_FAILED = "task_failed"
    OBSTACLE_DETECTED = "obstacle_detected"
    BATTERY_LOW = "battery_low"
    COMMUNICATION_LOST = "communication_lost"
    TRAJECTORY_DEVIATION = "trajectory_deviation"
    MANUAL_REQUEST = "manual_request"
    ENVIRONMENT_CHANGED = "environment_changed"
    SWARM_CONFLICT = "swarm_conflict"


class DynamicBTReplanner:
    """
    动态行为树重规划器
    
    功能:
    - 监控BT执行失败并自动重规划
    - 基于环境变化触发重规划
    - 重规划历史与回退策略
    - AGV等级适配 (重规划频率/超时)
    """

    def __init__(
        self,
        base_bt_factory: Callable[[], BTNode],
        max_replan_attempts: Optional[int] = None,
        replan_cooldown_s: Optional[float] = None,
        deviation_threshold: Optional[float] = None,
        grade: str = "M",
    ):
        self.base_bt_factory = base_bt_factory
        self.grade = grade

        # 当前活跃的BT
        self.current_bt: Optional[BehaviorTree] = None
        self.bt_variants: List[BTNode] = []  # 替代方案BT变体
        self.active_variant_index = 0

        # 重规划状态
        self.replan_count = 0
        self.last_replan_time = 0.0
        self.replan_history: List[Dict[str, Any]] = []
        self.failure_contexts: List[Dict[str, Any]] = []

        # 监控阈值 - 先应用等级默认值，再覆盖用户显式设置的值
        self._apply_grade_thresholds()
        if max_replan_attempts is not None:
            self.max_replan_attempts = max_replan_attempts
        if replan_cooldown_s is not None:
            self.replan_cooldown = replan_cooldown_s
        if deviation_threshold is not None:
            self.deviation_threshold = deviation_threshold

    def _apply_grade_thresholds(self) -> None:
        """根据AGV等级设置重规划参数"""
        grade_params = {
            "S": {"max_replan": 2, "cooldown": 3.0, "dev_thresh": 0.2},
            "M": {"max_replan": 3, "cooldown": 2.0, "dev_thresh": 0.3},
            "L": {"max_replan": 4, "cooldown": 1.5, "dev_thresh": 0.4},
            "XL": {"max_replan": 5, "cooldown": 1.0, "dev_thresh": 0.5},
            "XXL": {"max_replan": 6, "cooldown": 0.5, "dev_thresh": 0.6},
        }
        p = grade_params.get(self.grade, grade_params["M"])
        self.max_replan_attempts = p["max_replan"]
        self.replan_cooldown = p["cooldown"]
        self.deviation_threshold = p["dev_thresh"]

    def register_bt_variant(self, bt_variant: BTNode) -> None:
        """注册一个BT变体作为重规划备选"""
        self.bt_variants.append(bt_variant)

    def load_initial_bt(self) -> BehaviorTree:
        """加载初始行为树"""
        root = self.base_bt_factory()
        self.current_bt = BehaviorTree(root)
        self.active_variant_index = 0
        logger.info(f"Loaded initial BT, {len(self.bt_variants)} variant(s) available")
        return self.current_bt

    def should_replan(
        self,
        trigger: ReplanTrigger,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        判断是否应该触发重规划
        
        Args:
            trigger: 触发原因
            context: 额外上下文
        
        Returns:
            True 如果应该重规划
        """
        now = time.time()

        # 重规划次数上限 (先检查次数，再检查冷却)
        if self.replan_count >= self.max_replan_attempts:
            return False

        # 冷却期检查
        if now - self.last_replan_time < self.replan_cooldown:
            return False

        # 基础判断
        _ctx = context or {}
        reasons = {
            ReplanTrigger.TASK_FAILED: True,
            ReplanTrigger.OBSTACLE_DETECTED: _ctx.get('obstacle_distance', 999) < 0.5,
            ReplanTrigger.BATTERY_LOW: _ctx.get('battery_level', 1.0) < 0.15,
            ReplanTrigger.TRAJECTORY_DEVIATION: _ctx.get('deviation', 0.0) > self.deviation_threshold,
            ReplanTrigger.MANUAL_REQUEST: True,
            ReplanTrigger.ENVIRONMENT_CHANGED: True,
        }
        should = reasons.get(trigger, False)

        if should:
            self.failure_contexts.append({
                'trigger': trigger.value,
                'context': context or {},
                'timestamp': now,
                'replan_count': self.replan_count,
            })

        return should

    def replan(
        self,
        trigger: ReplanTrigger,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[BehaviorTree]:
        """
        执行重规划
        
        Returns:
            新的 BehaviorTree，如果重规划失败返回 None
        """
        if not self.should_replan(trigger, context):
            return self.current_bt

        self.replan_count += 1
        self.last_replan_time = time.time()
        now = self.last_replan_time

        # 尝试使用BT变体
        if self.bt_variants and self.active_variant_index < len(self.bt_variants):
            self.active_variant_index += 1
            variant = self.bt_variants[self.active_variant_index - 1]
            self.current_bt = BehaviorTree(variant)
            new_bt = self.current_bt
            strategy = f"variant_{self.active_variant_index}"
        else:
            # 重新创建基础BT
            root = self.base_bt_factory()
            self.current_bt = BehaviorTree(root)
            new_bt = self.current_bt
            self.active_variant_index = 0
            strategy = "recreate_base"

        self.replan_history.append({
            'replan_id': self.replan_count,
            'trigger': trigger.value,
            'strategy': strategy,
            'timestamp': now,
            'context': context or {},
            'active_variant': self.active_variant_index,
            'total_variants': len(self.bt_variants),
        })

        logger.info(
            f"BT Replan #{self.replan_count}: trigger={trigger.value}, "
            f"strategy={strategy}, variant={self.active_variant_index}"
        )
        return new_bt

    def reset_after_success(self) -> None:
        """任务成功后重置重规划状态"""
        self.replan_count = 0
        self.active_variant_index = 0
        self.failure_contexts.clear()

    def get_replan_statistics(self) -> Dict[str, Any]:
        """获取重规划统计"""
        return {
            'total_replans': self.replan_count,
            'max_attempts': self.max_replan_attempts,
            'history': self.replan_history[-10:],  # 最近10次
            'failure_contexts': self.failure_contexts[-5:],  # 最近5次失败上下文
            'cooldown_s': self.replan_cooldown,
            'deviation_threshold': self.deviation_threshold,
            'available_variants': len(self.bt_variants),
            'active_variant': self.active_variant_index,
        }


# ============================================================================
# 多机器人任务分配器 - Swarm Task Allocator
# ============================================================================


class AllocationStrategy(Enum):
    """任务分配策略"""
    GREEDY = "greedy"           # 贪心 - 最近优先
    LOAD_BALANCED = "load_balanced"  # 负载均衡
    CAPABILITY_MATCHED = "capability_matched"  # 能力匹配
    DISTANCE_MINIMIZED = "distance_minimized"  # 距离最小化
    PRIORITY_ORDERED = "priority_ordered"  # 优先级顺序


class RobotCapabilities:
    """机器人能力描述"""

    def __init__(
        self,
        robot_id: str,
        grade: str = "M",
        max_speed: float = 1.5,
        max_payload: float = 50.0,
        has_lift: bool = False,
        has_gripper: bool = True,
        terrain_capability: float = 1.0,  # 0-1, 地形适应能力
        battery_level: float = 1.0,
        current_position: Tuple[float, float] = (0.0, 0.0),
    ):
        self.robot_id = robot_id
        self.grade = grade
        self.max_speed = max_speed
        self.max_payload = max_payload
        self.has_lift = has_lift
        self.has_gripper = has_gripper
        self.terrain_capability = terrain_capability
        self.battery_level = battery_level
        self.current_position = current_position

    def can_execute(self, task: 'SwarmTask') -> bool:
        """判断是否能执行任务"""
        if task.required_payload > self.max_payload:
            return False
        if task.requires_lift and not self.has_lift:
            return False
        if task.requires_gripper and not self.has_gripper:
            return False
        if self.battery_level < 0.1:
            return False
        return True

    def estimated_time(self, task: 'SwarmTask') -> float:
        """估算执行任务的时间 (s)"""
        dist = self._distance_to(task.target_position)
        travel_time = dist / max(self.max_speed, 0.1)
        return travel_time + task.estimated_duration

    def _distance_to(self, target: Tuple[float, float]) -> float:
        """计算到目标点的距离 (m)"""
        dx = target[0] - self.current_position[0]
        dy = target[1] - self.current_position[1]
        return math.sqrt(dx*dx + dy*dy)


class SwarmTask:
    """蜂群任务"""

    def __init__(
        self,
        task_id: str,
        task_type: str,
        target_position: Tuple[float, float],
        priority: int = 5,  # 1-10, 10最高
        required_payload: float = 0.0,
        requires_lift: bool = False,
        requires_gripper: bool = False,
        terrain_type: Optional[str] = None,
        estimated_duration: float = 30.0,
        deadline: Optional[float] = None,
    ):
        self.task_id = task_id
        self.task_type = task_type
        self.target_position = target_position
        self.priority = priority
        self.required_payload = required_payload
        self.requires_lift = requires_lift
        self.requires_gripper = requires_gripper
        self.terrain_type = terrain_type
        self.estimated_duration = estimated_duration
        self.deadline = deadline
        self.assigned_robot: Optional[str] = None
        self.status: str = "pending"


@dataclass
class AllocationResult:
    """分配结果"""
    task_id: str
    robot_id: str
    estimated_time: float
    distance: float
    strategy: AllocationStrategy


class SwarmTaskAllocator:
    """
    多机器人任务分配器
    
    功能:
    - 多种分配策略 (贪心/负载均衡/能力匹配/距离最小化/优先级)
    - 多约束任务分配 ( payload/地形/电池)
    - 冲突检测与解决
    - 实时重分配
    - AGV五级规格适配
    """

    def __init__(
        self,
        strategy: AllocationStrategy = AllocationStrategy.GREEDY,
        max_reallocation_attempts: int = 3,
        conflict_radius: float = 0.5,
        grade: str = "M",
    ):
        self.strategy = strategy
        self.max_reallocation = max_reallocation_attempts
        self.conflict_radius = conflict_radius
        self.grade = grade

        self.robots: Dict[str, RobotCapabilities] = {}
        self.pending_tasks: List[SwarmTask] = []
        self.allocations: Dict[str, AllocationResult] = {}
        self.allocation_history: List[Dict[str, Any]] = []

    def register_robot(self, robot: RobotCapabilities) -> None:
        """注册机器人"""
        self.robots[robot.robot_id] = robot

    def add_task(self, task: SwarmTask) -> None:
        """添加任务"""
        self.pending_tasks.append(task)

    def add_tasks_batch(self, tasks: List[SwarmTask]) -> None:
        """批量添加任务"""
        self.pending_tasks.extend(tasks)

    def allocate(self) -> Dict[str, AllocationResult]:
        """
        执行任务分配
        
        Returns:
            {task_id: AllocationResult}
        """
        if not self.pending_tasks or not self.robots:
            return self.allocations

        # 清理已完成任务
        self.pending_tasks = [t for t in self.pending_tasks if t.status == "pending"]

        if self.strategy == AllocationStrategy.GREEDY:
            allocations = self._allocate_greedy()
        elif self.strategy == AllocationStrategy.LOAD_BALANCED:
            allocations = self._allocate_load_balanced()
        elif self.strategy == AllocationStrategy.CAPABILITY_MATCHED:
            allocations = self._allocate_capability_matched()
        elif self.strategy == AllocationStrategy.DISTANCE_MINIMIZED:
            allocations = self._allocate_distance_minimized()
        elif self.strategy == AllocationStrategy.PRIORITY_ORDERED:
            allocations = self._allocate_priority_ordered()
        else:
            allocations = self._allocate_greedy()

        # 冲突检测
        allocations = self._resolve_conflicts(allocations)

        self.allocations = allocations
        self.allocation_history.append({
            'timestamp': time.time(),
            'strategy': self.strategy.value,
            'allocations': {k: {'task_id': v.task_id, 'robot_id': v.robot_id, 'time': v.estimated_time}
                            for k, v in allocations.items()},
            'pending_count': len(self.pending_tasks),
            'robot_count': len(self.robots),
        })

        return allocations

    def _allocate_greedy(self) -> Dict[str, AllocationResult]:
        """贪心分配 - 优先分配给最近的可用机器人"""
        allocations = {}
        available_robots = {rid: r for rid, r in self.robots.items()}

        for task in sorted(self.pending_tasks, key=lambda t: -t.priority):
            best_robot = None
            best_time = float('inf')

            for rid, robot in available_robots.items():
                if not robot.can_execute(task):
                    continue
                est_time = robot.estimated_time(task)
                if est_time < best_time:
                    best_time = est_time
                    best_robot = rid

            if best_robot:
                dist = robot._distance_to(task.target_position)
                allocations[task.task_id] = AllocationResult(
                    task_id=task.task_id,
                    robot_id=best_robot,
                    estimated_time=best_time,
                    distance=dist,
                    strategy=self.strategy,
                )
                task.assigned_robot = best_robot
                task.status = "assigned"
                del available_robots[best_robot]

        return allocations

    def _allocate_load_balanced(self) -> Dict[str, AllocationResult]:
        """负载均衡分配"""
        allocations = {}
        robot_workload: Dict[str, float] = {rid: 0.0 for rid in self.robots}

        for task in sorted(self.pending_tasks, key=lambda t: -t.priority):
            best_robot = None
            best_score = float('inf')

            for rid, robot in self.robots.items():
                if not robot.can_execute(task):
                    continue
                workload = robot_workload[rid]
                est_time = robot.estimated_time(task)
                # 分数 = 当前负载 + 预估时间
                score = workload + est_time
                if score < best_score:
                    best_score = score
                    best_robot = rid

            if best_robot:
                robot_workload[best_robot] += best_score
                dist = self.robots[best_robot]._distance_to(task.target_position)
                allocations[task.task_id] = AllocationResult(
                    task_id=task.task_id,
                    robot_id=best_robot,
                    estimated_time=best_score,
                    distance=dist,
                    strategy=self.strategy,
                )
                task.assigned_robot = best_robot
                task.status = "assigned"

        return allocations

    def _allocate_capability_matched(self) -> Dict[str, AllocationResult]:
        """能力匹配分配"""
        allocations = {}
        for task in sorted(self.pending_tasks, key=lambda t: -t.priority):
            best_robot = None
            best_match_score = -float('inf')

            for rid, robot in self.robots.items():
                if not robot.can_execute(task):
                    continue
                score = 0.0
                # payload 匹配度
                score += (1.0 - task.required_payload / max(robot.max_payload, 1.0)) * 2.0
                # 地形能力
                if task.terrain_type:
                    score += robot.terrain_capability * 1.0
                # 电池余量
                score += robot.battery_level * 1.0
                # 速度
                score += robot.max_speed / 2.0
                # 距离惩罚
                dist = robot._distance_to(task.target_position)
                score -= dist * 0.5

                if score > best_match_score:
                    best_match_score = score
                    best_robot = rid

            if best_robot:
                dist = self.robots[best_robot]._distance_to(task.target_position)
                est_time = self.robots[best_robot].estimated_time(task)
                allocations[task.task_id] = AllocationResult(
                    task_id=task.task_id,
                    robot_id=best_robot,
                    estimated_time=est_time,
                    distance=dist,
                    strategy=self.strategy,
                )
                task.assigned_robot = best_robot
                task.status = "assigned"

        return allocations

    def _allocate_distance_minimized(self) -> Dict[str, AllocationResult]:
        """总距离最小化分配 (简化为每个任务选最近)"""
        return self._allocate_greedy()  # 贪心 = 距离最近

    def _allocate_priority_ordered(self) -> Dict[str, AllocationResult]:
        """优先级顺序分配"""
        allocations = {}
        for task in sorted(self.pending_tasks, key=lambda t: (-t.priority, t.deadline or float('inf'))):
            best_robot = None
            best_time = float('inf')

            for rid, robot in self.robots.items():
                if not robot.can_execute(task):
                    continue
                est_time = robot.estimated_time(task)
                # 考虑截止时间紧迫性
                if task.deadline:
                    time_to_deadline = task.deadline - time.time()
                    if est_time > time_to_deadline:
                        est_time *= 2.0  # 惩罚
                if est_time < best_time:
                    best_time = est_time
                    best_robot = rid

            if best_robot:
                dist = self.robots[best_robot]._distance_to(task.target_position)
                allocations[task.task_id] = AllocationResult(
                    task_id=task.task_id,
                    robot_id=best_robot,
                    estimated_time=best_time,
                    distance=dist,
                    strategy=self.strategy,
                )
                task.assigned_robot = best_robot
                task.status = "assigned"

        return allocations

    def _resolve_conflicts(
        self,
        allocations: Dict[str, AllocationResult],
    ) -> Dict[str, AllocationResult]:
        """解决机器人任务冲突"""
        # 按机器人分组任务
        robot_tasks: Dict[str, List[str]] = {}
        for task_id, result in allocations.items():
            rid = result.robot_id
            if rid not in robot_tasks:
                robot_tasks[rid] = []
            robot_tasks[rid].append(task_id)

        # 检测同一机器人多个同时任务
        for rid, task_ids in robot_tasks.items():
            if len(task_ids) > 1:
                # 保留优先级最高的任务
                task_id_map = {t.task_id: t for t in self.pending_tasks if t.task_id in task_ids}
                sorted_tasks = sorted(
                    [task_id_map[tid] for tid in task_ids],
                    key=lambda t: -t.priority
                )
                for t in sorted_tasks[1:]:
                    del allocations[t.task_id]
                    t.assigned_robot = None
                    t.status = "pending"

        return allocations

    def reallocate_on_failure(
        self,
        failed_task_id: str,
        failed_robot_id: str,
    ) -> Optional[AllocationResult]:
        """任务失败后重新分配"""
        task = next((t for t in self.pending_tasks if t.task_id == failed_task_id), None)
        if not task or task.status == "completed":
            return None

        # 移除原分配中失败的机器人
        if failed_task_id in self.allocations:
            del self.allocations[failed_task_id]

        # 重分配
        for attempt in range(self.max_reallocation):
            best_robot = None
            best_time = float('inf')

            for rid, robot in self.robots.items():
                if rid == failed_robot_id:
                    continue
                if not robot.can_execute(task):
                    continue
                est_time = robot.estimated_time(task)
                if est_time < best_time:
                    best_time = est_time
                    best_robot = rid

            if best_robot:
                dist = self.robots[best_robot]._distance_to(task.target_position)
                result = AllocationResult(
                    task_id=task.task_id,
                    robot_id=best_robot,
                    estimated_time=best_time,
                    distance=dist,
                    strategy=self.strategy,
                )
                self.allocations[task.task_id] = result
                task.assigned_robot = best_robot
                task.status = "assigned"
                return result

        return None

    def get_allocation_report(self) -> Dict[str, Any]:
        """获取分配报告"""
        return {
            'strategy': self.strategy.value,
            'total_robots': len(self.robots),
            'total_pending_tasks': len([t for t in self.pending_tasks if t.status == "pending"]),
            'total_allocated': len(self.allocations),
            'allocations': {
                tid: {
                    'robot_id': r.robot_id,
                    'estimated_time_s': r.estimated_time,
                    'distance_m': r.distance,
                }
                for tid, r in self.allocations.items()
            },
            'history_count': len(self.allocation_history),
        }

    def update_robot_state(self, robot_id: str, **kwargs) -> None:
        """更新机器人状态"""
        if robot_id in self.robots:
            for key, value in kwargs.items():
                if hasattr(self.robots[robot_id], key):
                    setattr(self.robots[robot_id], key, value)


