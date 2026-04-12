"""
Behavior Tree Engine - 行为树运行时执行引擎
集成行为树、AGV状态、传感器数据、控制指令输出
"""

import time
from enum import Enum
from typing import Dict, Optional, Callable, Tuple, List
from threading import Thread, Lock
import numpy as np

try:
    from control.planner import BehaviorNode, NodeStatus as ControlNodeStatus
except ImportError:
    BehaviorNode = None
    ControlNodeStatus = None

# 测试兼容：如果没有导入BehaviorNode，使用本地Node类作为别名
if BehaviorNode is None:
    BehaviorNode = Node


class NodeStatus(Enum):
    """节点执行状态"""
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


class Node:
    """行为树节点基类"""
    def __init__(self, name: str):
        self.name = name
        self.parent = None
        self.children = []
    
    def tick(self, context: Dict) -> NodeStatus:
        """执行节点逻辑，返回状态"""
        raise NotImplementedError
    
    def add_child(self, child: "Node"):
        """添加子节点"""
        child.parent = self
        self.children.append(child)


class ConditionNode(Node):
    """条件节点：返回SUCCESS或FAILURE"""
    def __init__(self, name: str, condition_func: Callable[[Dict], bool]):
        super().__init__(name)
        self.condition_func = condition_func
    
    def tick(self, context: Dict) -> NodeStatus:
        if self.condition_func(context):
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE


class TaskNode(Node):
    """任务节点：执行具体动作"""
    def __init__(self, name: str, task_func: Callable[[Dict], Dict]):
        super().__init__(name)
        self.task_func = task_func
    
    def tick(self, context: Dict) -> NodeStatus:
        result = self.task_func(context)
        if result.get("success", False):
            # 更新上下文
            context.update(result)
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE


class SequenceNode(Node):
    """顺序节点：依次执行子节点，全部成功返回成功，否则失败"""
    def tick(self, context: Dict) -> NodeStatus:
        for child in self.children:
            status = child.tick(context)
            if status != NodeStatus.SUCCESS:
                return status
        return NodeStatus.SUCCESS


class BehaviorTreeEngine:
    """
    行为树执行引擎
    负责加载行为树、同步状态到黑板、执行tick、输出控制指令
    支持两种模式：传统BehaviorNode模式，和简化的测试兼容模式
    """

    def __init__(
        self,
        *args,
        behavior_tree: Optional[BehaviorNode] = None,
        tree_name: Optional[str] = None,
        update_rate: float = 100.0,  # Hz
        name: str = "BT_Engine"
    ):
        # 兼容测试模式：如果第一个参数是字符串，就是tree_name
        if len(args) == 1 and isinstance(args[0], str):
            tree_name = args[0]
        
        # 兼容测试模式：通过tree_name初始化
        if tree_name is not None:
            self.name = tree_name
            self.nodes = {}
            self.root = None
            # 初始化空的黑板/上下文
            self.blackboard = {}
            self.running = False
            self.thread = None
            self.lock = Lock()
            self.stats = {
                "total_ticks": 0,
                "success_count": 0,
                "failure_count": 0,
                "running_count": 0,
                "avg_tick_time": 0.0
            }
            return
        
        # 原有初始化逻辑
        self.behavior_tree = behavior_tree or (args[0] if len(args) == 1 else None)
        self.update_rate = update_rate
        self.dt = 1.0 / update_rate
        self.name = name
        self.behavior_tree = behavior_tree
        self.update_rate = update_rate
        self.dt = 1.0 / update_rate
        self.name = name
        self.running = False
        self.thread: Optional[Thread] = None
        self.lock = Lock()

        # 状态黑板
        self.blackboard: Dict = {
            "current_x": 0.0,
            "current_y": 0.0,
            "current_theta": 0.0,
            "current_time": 0.0,
            "battery_level": 1.0,
            "obstacles": [],
            "desired_velocity": 0.0,
            "desired_omega": 0.0,
            "gripper_command": "idle",
            "task_completed": False,
            "held_object": None
        }
        if self.behavior_tree is not None:
            self.behavior_tree.blackboard = self.blackboard

        # 回调函数
        self.on_control_output: Optional[Callable[[float, float, str], None]] = None
        self.on_status_change: Optional[Callable[[NodeStatus], None]] = None

        # 统计信息
        self.stats = {
            "total_ticks": 0,
            "success_count": 0,
            "failure_count": 0,
            "running_count": 0,
            "avg_tick_time": 0.0
        }

    def set_state(
        self,
        x: float,
        y: float,
        theta: float,
        battery_level: float = 1.0,
        obstacles: Optional[list] = None,
        current_time: Optional[float] = None
    ):
        """更新AGV状态到黑板"""
        with self.lock:
            self.blackboard["current_x"] = x
            self.blackboard["current_y"] = y
            self.blackboard["current_theta"] = theta
            self.blackboard["battery_level"] = battery_level
            self.blackboard["obstacles"] = obstacles or []
            if current_time is not None:
                self.blackboard["current_time"] = current_time
            else:
                self.blackboard["current_time"] = time.time()

    def get_control_output(self) -> Tuple[float, float, str]:
        """获取最新的控制输出：(v, omega, gripper_command)"""
        with self.lock:
            return (
                self.blackboard.get("desired_velocity", 0.0),
                self.blackboard.get("desired_omega", 0.0),
                self.blackboard.get("gripper_command", "idle")
            )

    def reset(self):
        """重置行为树和引擎状态"""
        with self.lock:
            self.behavior_tree.reset()
            self.blackboard["task_completed"] = False
            self.blackboard["held_object"] = None
            self.blackboard["desired_velocity"] = 0.0
            self.blackboard["desired_omega"] = 0.0
            self.blackboard["gripper_command"] = "idle"
            self.stats = {
                "total_ticks": 0,
                "success_count": 0,
                "failure_count": 0,
                "running_count": 0,
                "avg_tick_time": 0.0
            }

    def tick(self) -> NodeStatus:
        """执行一次tick，返回行为树状态"""
        start_time = time.time()
        with self.lock:
            status = self.behavior_tree.tick()
            self.stats["total_ticks"] += 1
            if status == NodeStatus.SUCCESS:
                self.stats["success_count"] += 1
            elif status == NodeStatus.FAILURE:
                self.stats["failure_count"] += 1
            else:
                self.stats["running_count"] += 1

        tick_time = time.time() - start_time
        self.stats["avg_tick_time"] = 0.9 * self.stats["avg_tick_time"] + 0.1 * tick_time

        # 调用回调
        if self.on_control_output:
            v, omega, gripper = self.get_control_output()
            self.on_control_output(v, omega, gripper)
        if self.on_status_change:
            self.on_status_change(status)

        return status

    def start(self, background: bool = True):
        """启动引擎，background=True时在后台线程运行"""
        if self.running:
            return
        self.running = True
        if background:
            self.thread = Thread(target=self._run_loop, daemon=True)
            self.thread.start()
        else:
            self._run_loop()

    def stop(self):
        """停止引擎"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None

    def _run_loop(self):
        """后台执行循环"""
        last_time = time.time()
        while self.running:
            current_time = time.time()
            elapsed = current_time - last_time

            if elapsed >= self.dt:
                self.tick()
                last_time = current_time

            # 节省CPU
            sleep_time = max(0.0, self.dt - (time.time() - current_time))
            time.sleep(sleep_time)

    def get_stats(self) -> Dict:
        """获取引擎运行统计信息"""
        with self.lock:
            return self.stats.copy()

    def is_running(self) -> bool:
        """返回引擎是否在运行"""
        return self.running

    def set_blackboard_value(self, key: str, value):
        """设置黑板值"""
        with self.lock:
            self.blackboard[key] = value

    def get_blackboard_value(self, key: str, default=None):
        """获取黑板值"""
        with self.lock:
            return self.blackboard.get(key, default)

    def add_node(self, node: Node):
        """添加节点到节点库（测试兼容接口）"""
        if hasattr(self, "nodes"):
            self.nodes[node.name] = node
    
    def set_root_sequence(self, node_names: list):
        """设置根顺序节点（测试兼容接口）"""
        if hasattr(self, "nodes"):
            self.root = SequenceNode("root")
            for name in node_names:
                if name in self.nodes:
                    self.root.add_child(self.nodes[name])
    
    def add_sequence(self, name: str, node_names: List[str]):
        """添加顺序节点（测试兼容接口）"""
        if not hasattr(self, "nodes"):
            self.nodes = {}
        seq_node = SequenceNode(name)
        for node_name in node_names:
            if node_name in self.nodes:
                seq_node.add_child(self.nodes[node_name])
        self.nodes[name] = seq_node
        # 如果没有根节点，设置为根
        if not hasattr(self, "root") or self.root is None:
            self.root = seq_node

    def add_fallback(self, name: str, node_names: List[str]):
        """添加fallback/选择节点（测试兼容接口）"""
        if not hasattr(self, "nodes"):
            self.nodes = {}
        # 先创建Fallback节点类（如果不存在）
        if not hasattr(self, "_fallback_class"):
            class FallbackNode(Node):
                def tick(self, context: Dict) -> NodeStatus:
                    for child in self.children:
                        status = child.tick(context)
                        if status != NodeStatus.FAILURE:
                            return status
                    return NodeStatus.FAILURE
            self._fallback_class = FallbackNode
        fb_node = self._fallback_class(name)
        for node_name in node_names:
            if node_name in self.nodes:
                fb_node.add_child(self.nodes[node_name])
        self.nodes[name] = fb_node
        # 如果没有根节点，设置为根
        if not hasattr(self, "root") or self.root is None:
            self.root = fb_node

    def execute(self, context: Dict) -> Dict:
        """执行行为树，返回结果（测试兼容接口）"""
        if not hasattr(self, "root") or not self.root:
            return {"success": False, "error": "No root node set"}
        
        # 拷贝上下文避免修改原对象
        ctx = context.copy()
        # 测试兼容：如果没有current_pos，默认等于start_pos
        if "current_pos" not in ctx and "start_pos" in ctx:
            ctx["current_pos"] = ctx["start_pos"]
        status = self.root.tick(ctx)
        
        return {
            "success": status == NodeStatus.SUCCESS,
            "context": ctx,
            "status": status.value
        }
