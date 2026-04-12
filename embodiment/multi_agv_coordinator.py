"""
Multi AGV Coordinator - 多AGV蜂群协同调度器
支持任务分配、路径规划避障、AGV间冲突协调、负载均衡
"""

import time
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

from control.planner import RRTStarPlanner, Waypoint, TrajectoryPlanner


class AGVStatus(Enum):
    """AGV运行状态"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    CHARGING = "charging"


@dataclass
class AGVInfo:
    """AGV信息"""
    agv_id: int
    status: AGVStatus = AGVStatus.IDLE
    current_position: Tuple[float, float] = (0.0, 0.0)
    current_theta: float = 0.0
    battery_level: float = 1.0
    current_task_id: Optional[str] = None
    current_trajectory: Optional[list] = None
    speed: float = 1.0  # 移动速度系数


@dataclass
class AGVTask:
    """AGV任务"""
    task_id: str
    task_type: str  # transfer, patrol, charge
    priority: int = 5  # 1-10，越高优先级越高
    pick_location: Optional[Tuple[float, float]] = None
    place_location: Optional[Tuple[float, float]] = None
    patrol_points: Optional[List[Tuple[float, float]]] = None
    deadline: Optional[float] = None  # 截止时间戳
    assigned_agv_id: Optional[int] = None
    status: str = "pending"  # pending, assigned, running, completed, failed


@dataclass
class AGVAssignment:
    """任务分配结果"""
    task_id: str
    agv_id: int
    estimated_time: float
    success: bool = True
    reason: str = ""


class MultiAGVCoordinator:
    """
    多AGV协同调度器
    功能：
    1. 任务分配：基于优先级、距离、负载均衡
    2. 路径协调：AGV间碰撞避免，路口调度
    3. 状态监控：所有AGV状态实时监控
    4. 异常处理：AGV故障时任务重分配
    """

    def __init__(
        self,
        bounds: Tuple[float, float, float, float],  # 工作区域边界 (xmin, xmax, ymin, ymax)
        obstacle_safety_distance: float = 0.5,
        agv_safety_distance: float = 1.0
    ):
        self.bounds = bounds
        self.obstacle_safety_distance = obstacle_safety_distance
        self.agv_safety_distance = agv_safety_distance

        self.agvs: Dict[int, AGVInfo] = {}
        self.tasks: Dict[str, AGVTask] = {}
        self.global_obstacles: List[Tuple[float, float, float]] = []  # 全局静态障碍物

        # 路径规划器
        self.rrt_planner = RRTStarPlanner(bounds, max_iter=300)
        self.traj_planner = TrajectoryPlanner()

        # 冲突检测表
        self.reserved_regions: Dict[Tuple[int, int], List[int]] = {}  # (grid_x, grid_y) -> [agv_ids]
        self.grid_size = 0.5  # 栅格大小 0.5m

    def add_agv(self, agv_id: int, start_position: Tuple[float, float] = (0.0, 0.0)):
        """添加AGV到调度器"""
        self.agvs[agv_id] = AGVInfo(
            agv_id=agv_id,
            current_position=start_position
        )

    def remove_agv(self, agv_id: int):
        """从调度器移除AGV"""
        if agv_id in self.agvs:
            del self.agvs[agv_id]

    def add_task(self, task: AGVTask) -> str:
        """添加任务，返回任务ID"""
        self.tasks[task.task_id] = task
        return task.task_id

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.assigned_agv_id and task.status == "running":
                # 重置AGV状态
                if task.assigned_agv_id in self.agvs:
                    self.agvs[task.assigned_agv_id].status = AGVStatus.IDLE
                    self.agvs[task.assigned_agv_id].current_task_id = None
            del self.tasks[task_id]
            return True
        return False

    def update_agv_state(
        self,
        agv_id: int,
        position: Tuple[float, float],
        theta: float,
        battery_level: float,
        status: Optional[AGVStatus] = None
    ):
        """更新AGV实时状态"""
        if agv_id not in self.agvs:
            return
        agv = self.agvs[agv_id]
        agv.current_position = position
        agv.current_theta = theta
        agv.battery_level = battery_level
        if status is not None:
            agv.status = status

    def update_global_obstacles(self, obstacles: List[Tuple[float, float, float]]):
        """更新全局障碍物列表"""
        self.global_obstacles = obstacles

    def assign_tasks(self) -> List[AGVAssignment]:
        """
        执行任务分配，将待分配任务分配给空闲AGV
        分配策略：
        1. 高优先级任务优先分配
        2. 距离任务起点最近的AGV优先
        3. 电池电量充足的AGV优先
        """
        assignments = []
        pending_tasks = [t for t in self.tasks.values() if t.status == "pending"]
        idle_agvs = [a for a in self.agvs.values() if a.status == AGVStatus.IDLE and a.battery_level > 0.2]

        # 按优先级排序任务
        pending_tasks.sort(key=lambda t: -t.priority)

        for task in pending_tasks:
            if not idle_agvs:
                break

            # 计算每个空闲AGV的评分
            best_agv = None
            best_score = float('inf')
            best_eta = 0.0

            for agv in idle_agvs:
                # 计算到任务起点的距离
                if task.task_type == "transfer":
                    start_pos = task.pick_location if task.pick_location else (0, 0)
                elif task.task_type == "patrol":
                    start_pos = task.patrol_points[0] if task.patrol_points else (0, 0)
                else:
                    start_pos = (0, 0)

                dist = math.hypot(
                    agv.current_position[0] - start_pos[0],
                    agv.current_position[1] - start_pos[1]
                )

                # 评分 = 距离 * (1 + (1 - battery_level)*2) - 电池低的评分更高(更差)
                score = dist * (1 + (1 - agv.battery_level) * 2)

                if score < best_score:
                    best_score = score
                    best_agv = agv
                    best_eta = dist / agv.speed

            if best_agv:
                # 分配任务
                task.assigned_agv_id = best_agv.agv_id
                task.status = "assigned"
                best_agv.status = AGVStatus.BUSY
                best_agv.current_task_id = task.task_id
                idle_agvs.remove(best_agv)

                assignments.append(AGVAssignment(
                    task_id=task.task_id,
                    agv_id=best_agv.agv_id,
                    estimated_time=best_eta,
                    success=True
                ))
            else:
                assignments.append(AGVAssignment(
                    task_id=task.task_id,
                    agv_id=-1,
                    estimated_time=0.0,
                    success=False,
                    reason="No available AGV"
                ))

        return assignments

    def plan_agv_path(
        self,
        agv_id: int,
        target_position: Tuple[float, float]
    ) -> Optional[list]:
        """
        为AGV规划到目标点的无冲突路径
        考虑全局障碍物和其他AGV的位置
        """
        if agv_id not in self.agvs:
            return None

        agv = self.agvs[agv_id]
        start_pos = agv.current_position

        # 构建动态障碍物：其他AGV
        dynamic_obstacles = []
        for other_agv_id, other_agv in self.agvs.items():
            if other_agv_id != agv_id:
                ox, oy = other_agv.current_position
                dynamic_obstacles.append((ox, oy, self.agv_safety_distance))

        all_obstacles = self.global_obstacles + dynamic_obstacles

        # 规划路径
        path = self.rrt_planner.plan(start_pos, target_position, all_obstacles)

        if not path or len(path) < 2:
            return None

        # 转换为Waypoint
        waypoints = [Waypoint(x=p[0], y=p[1]) for p in path]
        trajectory = self.traj_planner.plan_path(waypoints)

        # 保存路径到AGV信息
        agv.current_trajectory = trajectory.points

        return trajectory.points

    def check_conflicts(self) -> List[Tuple[int, int, str]]:
        """
        检查AGV之间的冲突
        返回冲突列表：(agv_id1, agv_id2, conflict_type)
        conflict_type: collision, deadlock, priority
        """
        conflicts = []
        agv_list = list(self.agvs.values())

        # 检查碰撞冲突
        for i in range(len(agv_list)):
            for j in range(i+1, len(agv_list)):
                a1 = agv_list[i]
                a2 = agv_list[j]
                dist = math.hypot(
                    a1.current_position[0] - a2.current_position[0],
                    a1.current_position[1] - a2.current_position[1]
                )
                if dist < self.agv_safety_distance:
                    conflicts.append((a1.agv_id, a2.agv_id, "collision"))

        # 检查路径冲突
        # ... (实现路径重叠检查逻辑)

        return conflicts

    def resolve_conflicts(self, conflicts: List[Tuple[int, int, str]]):
        """解决AGV冲突"""
        for (agv1, agv2, conflict_type) in conflicts:
            if conflict_type == "collision":
                # 简单策略：低优先级任务停车让行
                task1 = self.tasks.get(self.agvs[agv1].current_task_id, None)
                task2 = self.tasks.get(self.agvs[agv2].current_task_id, None)

                priority1 = task1.priority if task1 else 0
                priority2 = task2.priority if task2 else 0

                # 低优先级的AGV停车
                if priority1 < priority2:
                    self.agvs[agv1].status = AGVStatus.BUSY  # 临时停车
                else:
                    self.agvs[agv2].status = AGVStatus.BUSY

    def get_agv_task(self, agv_id: int) -> Optional[AGVTask]:
        """获取AGV当前分配的任务"""
        if agv_id not in self.agvs:
            return None
        agv = self.agvs[agv_id]
        return self.tasks.get(agv.current_task_id, None)

    def complete_task(self, task_id: str, success: bool = True):
        """标记任务完成，释放AGV资源"""
        if task_id not in self.tasks:
            return
        task = self.tasks[task_id]
        task.status = "completed" if success else "failed"
        if task.assigned_agv_id and task.assigned_agv_id in self.agvs:
            agv = self.agvs[task.assigned_agv_id]
            agv.status = AGVStatus.IDLE
            agv.current_task_id = None
            agv.current_trajectory = None

    def get_system_status(self) -> Dict:
        """获取整个系统的状态统计"""
        total_agvs = len(self.agvs)
        idle_agvs = sum(1 for a in self.agvs.values() if a.status == AGVStatus.IDLE)
        busy_agvs = sum(1 for a in self.agvs.values() if a.status == AGVStatus.BUSY)
        error_agvs = sum(1 for a in self.agvs.values() if a.status == AGVStatus.ERROR)
        charging_agvs = sum(1 for a in self.agvs.values() if a.status == AGVStatus.CHARGING)

        total_tasks = len(self.tasks)
        pending_tasks = sum(1 for t in self.tasks.values() if t.status == "pending")
        running_tasks = sum(1 for t in self.tasks.values() if t.status == "running")
        completed_tasks = sum(1 for t in self.tasks.values() if t.status == "completed")
        failed_tasks = sum(1 for t in self.tasks.values() if t.status == "failed")

        return {
            "agvs": {
                "total": total_agvs,
                "idle": idle_agvs,
                "busy": busy_agvs,
                "error": error_agvs,
                "charging": charging_agvs
            },
            "tasks": {
                "total": total_tasks,
                "pending": pending_tasks,
                "running": running_tasks,
                "completed": completed_tasks,
                "failed": failed_tasks
            },
            "obstacles": len(self.global_obstacles)
        }
