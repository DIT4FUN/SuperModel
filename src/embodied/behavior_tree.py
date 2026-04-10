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


def create_behavior_tree_from_dict(config: Dict[str, Any]) -> BTNode:
    """从字典配置创建行为树 - 用于反序列化/配置文件驱动"""
    # TODO: 实现配置驱动的行为树创建
    pass
