"""
Behavior Tree Engine - 行为树运行时执行引擎
集成行为树、AGV状态、传感器数据、控制指令输出
"""

import time
from typing import Dict, Optional, Callable, Tuple
from threading import Thread, Lock
import numpy as np

from control.planner import BehaviorNode, NodeStatus


class BehaviorTreeEngine:
    """
    行为树执行引擎
    负责加载行为树、同步状态到黑板、执行tick、输出控制指令
    """

    def __init__(
        self,
        behavior_tree: BehaviorNode,
        update_rate: float = 100.0,  # Hz
        name: str = "BT_Engine"
    ):
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
