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
import enum
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

    def tick(self, blackboard: Blackboard) -> NodeStatus:
        if self.status != NodeStatus.RUNNING:
            self.initialize()

        for child in self.children:
            status = child.tick(blackboard)

            if status == NodeStatus.RUNNING:
                return self.terminate(NodeStatus.RUNNING)
            elif status == NodeStatus.FAILURE:
                return self.terminate(NodeStatus.FAILURE)

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
        if self.status != NodeStatus.RUNNING:
            self.initialize()
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

        # 在实际系统中，这里会调用运动控制器
        # 这里只是示例，返回 RUNNING 模拟持续执行
        if not self._started:
            logger.info(f"AGV moving to target: {target_pos} at speed {self.speed}")
            self._started = True

        # 检查是否到达
        if current_pos is not None:
            distance = np.linalg.norm(current_pos - np.array(target_pos))
            if distance < 0.1:
                self._started = False
                return NodeStatus.SUCCESS

        return NodeStatus.RUNNING

    def reset(self) -> None:
        self._started = False
        super().reset()


class AGVGraspAction(ActionNode):
    """AGV抓取动作节点"""

    def __init__(self, name: str = "Grasp"):
        super().__init__(name)
        self.grasp_start_time: Optional[float] = None

    def execute(self, blackboard: Blackboard) -> NodeStatus:
        target_object = blackboard.goal_state.get('target_object')
        if not target_object:
            return NodeStatus.FAILURE

        if self.grasp_start_time is None:
            logger.info(f"Starting grasp on object: {target_object}")
            self.grasp_start_time = time.time()
            # 模拟抓取过程需要时间
            return NodeStatus.RUNNING

        # 模拟抓取完成
        if time.time() - self.grasp_start_time > 2.0:
            logger.info(f"Completed grasp on object: {target_object}")
            self.grasp_start_time = None
            return NodeStatus.SUCCESS

        return NodeStatus.RUNNING

    def reset(self) -> None:
        self.grasp_start_time = None
        super().reset()


class AGVReleaseAction(ActionNode):
    """AGV释放动作节点"""

    def __init__(self, name: str = "Release"):
        super().__init__(name)
        self.release_start_time: Optional[float] = None

    def execute(self, blackboard: Blackboard) -> NodeStatus:
        if self.release_start_time is None:
            logger.info("Starting release")
            self.release_start_time = time.time()
            return NodeStatus.RUNNING

        if time.time() - self.release_start_time > 1.0:
            logger.info("Completed release")
            self.release_start_time = None
            return NodeStatus.SUCCESS

        return NodeStatus.RUNNING

    def reset(self) -> None:
        self.release_start_time = None
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
        patrol_root = UntilFailNode(
            SelectorNode("PatrolSelector"),
            "PatrolRoot"
        )
        self.register_task_type('patrol', patrol_root)

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

    def get_capabilities(self) -> Dict[str, Any]:
        """获取当前AGV等级的规划能力"""
        return {
            'grade': self.grade,
            **self.capabilities,
        }


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


def create_behavior_tree_from_dict(config: Dict[str, Any]) -> BTNode:
    """从字典配置创建行为树 - 用于反序列化/配置文件驱动"""
    # TODO: 实现配置驱动的行为树创建
    pass
# ============================================================================
# 配置驱动的行为树构建器
# ============================================================================

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
            times = config.get('num_repeats', config.get('times', -1))
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
