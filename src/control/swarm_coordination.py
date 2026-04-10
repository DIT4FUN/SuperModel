"""
swarm_coordination.py - 多AGV蜂群协同调度模块
SuperModel 超模态大模型具身智能系统

功能:
- 多AGV任务分配 (匈牙利算法/二分图匹配)
- 路径冲突检测
- 冲突消解策略
- 不同协同策略 (集中式/分布式/混合/基于市场)
- 动态任务重分配
- 区域协同调度
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import logging

logger = logging.getLogger(__name__)
__all__ = [
    'CoordinationStrategy',
    'PathConflict',
    'ConflictResolution',
    'TaskAllocator',
    'ConflictDetector',
    'SwarmCoordinator',
]


class CoordinationStrategy(Enum):
    """多AGV协同策略"""
    CENTRALIZED = "centralized"       # 集中式调度
    DECENTRALIZED = "decentralized"  # 分布式协商
    HYBRID = "hybrid"                 # 混合策略
    MARKET_BASED = "market_based"     # 基于市场拍卖


@dataclass
class PathConflict:
    """路径冲突描述"""
    robot1_id: str
    robot2_id: str
    position: np.ndarray
    time1: float          # 机器人1到达冲突点时间
    time2: float          # 机器人2到达冲突点时间
    distance: float       # 预测最小距离
    is_head_on: bool = False  # 是否对头冲突

    def is_time_conflict(self, time_threshold: float = 0.5) -> bool:
        """是否在时间上冲突"""
        return abs(self.time1 - self.time2) < time_threshold

    def get_severity(self) -> float:
        """计算冲突严重程度 (0-1)"""
        # 距离越小，时间差越小，越严重
        time_factor = 1.0 - min(abs(self.time1 - self.time2), 1.0)
        distance_factor = 1.0 - min(self.distance, 1.0)
        return (time_factor + distance_factor) / 2


@dataclass
class ConflictResolution:
    """冲突解决方案"""
    conflict: PathConflict
    resolved: bool
    method: str  # "speed_adjustment", "waiting", "reroute"
    new_speed1: Optional[float] = None
    new_speed2: Optional[float] = None
    wait_time1: float = 0.0
    wait_time2: float = 0.0
    new_path1: Optional[List[np.ndarray]] = None
    new_path2: Optional[List[np.ndarray]] = None

    def get_waiting_total(self) -> float:
        """获取总等待时间"""
        return self.wait_time1 + self.wait_time2


class TaskAllocator:
    """多AGV任务分配器
    基于成本矩阵的二分图最优匹配
    支持距离和电量成本
    """

    def __init__(
        self,
        num_robots: int = 1,
        strategy: str = "bipartite",
        consider_battery: bool = True,
        battery_weight: float = 0.3,
    ):
        self.num_robots = num_robots
        self.strategy = strategy
        self.consider_battery = consider_battery
        self.battery_weight = battery_weight

    def calculate_cost(
        self,
        robot: Dict[str, Any],
        task: Dict[str, Any]
    ) -> float:
        """计算分配成本"""
        # 基础距离成本
        robot_pos = np.array(robot['position'])
        task_target = np.array(task['target'])
        distance_cost = float(np.linalg.norm(robot_pos - task_target))

        # 电量成本
        if self.consider_battery and 'battery' in robot:
            battery = robot['battery']
            # 低电量 = 高成本
            battery_cost = (1.0 - battery) * self.battery_weight * 10.0
            distance_cost += battery_cost

        return distance_cost

    def allocate(
        self,
        robots: List[Dict[str, Any]],
        tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """分配任务给机器人

        Args:
            robots: 可用机器人列表，每项包含 'id', 'position', 'available'
            tasks: 待分配任务列表，每项包含 'id', 'target'

        Returns:
            分配结果列表，每项包含 'task_id', 'robot_id', 'cost'
        """
        # 过滤出可用机器人
        available_robots = [r for r in robots if r.get('available', True)]
        if not available_robots or not tasks:
            return []

        # 简化贪婪分配（完整匈牙利算法计算量较大）
        # 实际大规模使用可替换为scipy.optimize.linear_sum_assignment
        assignments: List[Dict[str, Any]] = []
        unassigned_tasks = tasks.copy()
        used_robot_ids: Set[str] = set()

        # 按成本贪婪分配
        while unassigned_tasks and len(used_robot_ids) < len(available_robots):
            best_assignment = None
            best_cost = float('inf')

            for task in unassigned_tasks:
                for robot in available_robots:
                    if robot['id'] in used_robot_ids:
                        continue
                    cost = self.calculate_cost(robot, task)
                    if cost < best_cost:
                        best_cost = cost
                        best_assignment = {
                            'task_id': task['id'],
                            'robot_id': robot['id'],
                            'cost': cost,
                        }

            if best_assignment:
                assignments.append(best_assignment)
                used_robot_ids.add(best_assignment['robot_id'])
                unassigned_tasks = [t for t in unassigned_tasks if t['id'] != best_assignment['task_id']]
            else:
                break

        return assignments


class ConflictDetector:
    """多AGV路径冲突检测器"""

    def __init__(
        self,
        conflict_distance_threshold: float = 0.5,
        time_threshold: float = 0.5,
    ):
        self.conflict_distance_threshold = conflict_distance_threshold
        self.time_threshold = time_threshold

    def _estimate_arrival_time(
        self,
        start_pos: np.ndarray,
        target_pos: np.ndarray,
        average_speed: float = 0.5
    ) -> float:
        """估计到达目标点的时间"""
        distance = np.linalg.norm(start_pos - target_pos)
        return distance / average_speed if average_speed > 0 else 0

    def detect_conflicts(
        self,
        robot1_id: str,
        path1: List[np.ndarray],
        robot2_id: str,
        path2: List[np.ndarray],
        avg_speed: float = 0.5
    ) -> List[PathConflict]:
        """检测两个路径之间的所有冲突"""
        conflicts: List[PathConflict] = []

        # 检查每个路径点对
        for i, p1 in enumerate(path1):
            t1 = self._estimate_arrival_time(path1[0], p1, avg_speed)
            for j, p2 in enumerate(path2):
                t2 = self._estimate_arrival_time(path2[0], p2, avg_speed)
                dist = float(np.linalg.norm(p1 - p2))

                if dist < self.conflict_distance_threshold:
                    # 距离足够近，检查时间
                    is_head_on = False
                    # 对头冲突: 方向相反
                    if i > 0 and j > 0 and len(path1) > i+1 and len(path2) > j+1:
                        dir1 = path1[i+1] - path1[i-1]
                        dir2 = path2[j+1] - path2[j-1]
                        dot = np.dot(dir1, dir2)
                        if dot < 0:
                            is_head_on = True

                    conflict = PathConflict(
                        robot1_id=robot1_id,
                        robot2_id=robot2_id,
                        position=(p1 + p2) / 2,
                        time1=t1,
                        time2=t2,
                        distance=dist,
                        is_head_on=is_head_on,
                    )
                    if conflict.is_time_conflict(self.time_threshold):
                        conflicts.append(conflict)

        return conflicts

    def detect_with_dynamic_obstacles(
        self,
        robot_path: List[np.ndarray],
        dynamic_obstacles: List[Obstacle],
        avg_speed: float = 0.5
    ) -> List[PathConflict]:
        """检测与动态障碍物的路径冲突"""
        conflicts: List[PathConflict] = []

        for i, p in enumerate(robot_path):
            t_robot = self._estimate_arrival_time(robot_path[0], p, avg_speed)
            for obs in dynamic_obstacles:
                # 假设障碍物匀速运动
                obs_pos_at_t = obs.position + obs.velocity * t_robot
                dist = float(np.linalg.norm(p - obs_pos_at_t[:2]) if len(p) == 2 else np.linalg.norm(p - obs_pos_at_t))
                if dist < self.conflict_distance_threshold:
                    conflict = PathConflict(
                        robot1_id="robot",
                        robot2_id=obs.id,
                        position=p,
                        time1=t_robot,
                        time2=t_robot,
                        distance=dist,
                    )
                    conflicts.append(conflict)

        return conflicts

    def detect_all_conflicts(
        self,
        robot_paths: Dict[str, List[np.ndarray]],
        avg_speed: float = 0.5
    ) -> List[PathConflict]:
        """检测所有机器人路径对之间的冲突"""
        all_conflicts: List[PathConflict] = []
        robot_ids = list(robot_paths.keys())

        for i, id1 in enumerate(robot_ids):
            for id2 in robot_ids[i+1:]:
                conflicts = self.detect_conflicts(
                    id1, robot_paths[id1], id2, robot_paths[id2], avg_speed
                )
                all_conflicts.extend(conflicts)

        return all_conflicts


class SwarmCoordinator:
    """多AGV蜂群协调器
    整合任务分配、冲突检测、冲突消解
    支持多种协调策略
    """

    def __init__(
        self,
        num_robots: int = 1,
        strategy: CoordinationStrategy = CoordinationStrategy.CENTRALIZED
    ):
        self.num_robots = num_robots
        self.strategy = strategy
        self.robots: Dict[str, Dict[str, Any]] = {}
        self.pending_tasks: List[Dict[str, Any]] = []
        self.allocations: List[Dict[str, Any]] = []
        self.detector = ConflictDetector()
        self.allocator = TaskAllocator(num_robots)
        self.total_conflicts_detected = 0
        self.conflicts_resolved = 0

        # 统计
        self.stats = {
            'total_tasks_allocated': 0,
            'total_conflicts': 0,
            'conflicts_resolved': 0,
            'conflicts_unresolved': 0,
        }

    def register_robot(self, robot_id: str, position: np.ndarray, **kwargs) -> None:
        """注册机器人"""
        self.robots[robot_id] = {
            'id': robot_id,
            'position': position.copy(),
            'available': True,
            'battery': kwargs.get('battery', 1.0),
            **kwargs
        }

    def add_global_task(self, task: Dict[str, Any]) -> None:
        """添加全局待分配任务"""
        assert 'id' in task and 'target' in task
        self.pending_tasks.append(task)

    def coordinate_step(self) -> List[Dict[str, Any]]:
        """执行一步协调

        Returns:
            新分配的任务列表
        """
        # 1. 分配待处理任务
        available_robots = list(self.robots.values())
        new_assignments = self.allocator.allocate(available_robots, self.pending_tasks)

        # 将分配的任务从待处理中移除，并标记机器人为忙碌
        assigned_task_ids = {a['task_id'] for a in new_assignments}
        self.pending_tasks = [t for t in self.pending_tasks if t['id'] not in assigned_task_ids]

        # 更新机器人状态
        for assign in new_assignments:
            robot = self.robots[assign['robot_id']]
            robot['available'] = False
            robot['current_task'] = assign['task_id']

        self.allocations.extend(new_assignments)
        self.stats['total_tasks_allocated'] += len(new_assignments)

        # 2. 检测已有路径中的冲突
        # 集中式协调会重新规划路径，分布式则由机器人协商
        if self.strategy == CoordinationStrategy.CENTRALIZED:
            conflicts = self._detect_all_current_conflicts()
            self.total_conflicts_detected += len(conflicts)
            self.stats['total_conflicts'] += len(conflicts)

            # 解决冲突
            for conflict in conflicts:
                resolution = self.resolve_conflict(conflict)
                if resolution.resolved:
                    self.conflicts_resolved += 1
                    self.stats['conflicts_resolved'] += 1
                else:
                    self.stats['conflicts_unresolved'] += 1

        return new_assignments

    def _detect_all_current_conflicts(self) -> List[PathConflict]:
        """检测当前所有已分配任务路径中的冲突"""
        robot_paths: Dict[str, List[np.ndarray]] = {}
        for robot_id, robot in self.robots.items():
            if 'current_path' in robot and robot['current_path']:
                robot_paths[robot_id] = robot['current_path']
        return self.detector.detect_all_conflicts(robot_paths)

    def resolve_conflict(self, conflict: PathConflict) -> ConflictResolution:
        """解决两个机器人之间的路径冲突

        策略:
        1. 尝试速度调整错开时间
        2. 如果不行，让一台等待
        3. 如果还是不行，重规划路径
        """
        # 策略1: 速度调整 - 一个加速，一个减速
        robot1 = self.robots[conflict.robot1_id]
        robot2 = self.robots[conflict.robot2_id]
        current_speed1 = robot1.get('current_speed', 0.5)
        current_speed2 = robot2.get('current_speed', 0.5)

        # 如果时间接近，调整速度错开
        time_diff = abs(conflict.time1 - conflict.time2)
        if time_diff < 0.5:
            # 一个加速，一个减速
            # 保持平均速度，只是错开时间
            if conflict.time1 < conflict.time2:
                new_speed1 = min(current_speed1 * 1.5, 1.5)
                new_speed2 = max(current_speed2 * 0.5, 0.1)
            else:
                new_speed1 = max(current_speed1 * 0.5, 0.1)
                new_speed2 = min(current_speed2 * 1.5, 1.5)

            # 更新机器人速度
            robot1['current_speed'] = new_speed1
            robot2['current_speed'] = new_speed2

            return ConflictResolution(
                conflict=conflict,
                resolved=True,
                method='speed_adjustment',
                new_speed1=new_speed1,
                new_speed2=new_speed2
            )

        # 策略2: 等待 - 优先级低的等待
        priority1 = robot1.get('priority', 5)
        priority2 = robot2.get('priority', 5)

        if priority1 > priority2:
            # robot2 等待
            wait_time = abs(conflict.time1 - conflict.time2) + 1.0
            return ConflictResolution(
                conflict=conflict,
                resolved=True,
                method='waiting',
                wait_time2=wait_time
            )
        else:
            wait_time = abs(conflict.time1 - conflict.time2) + 1.0
            return ConflictResolution(
                conflict=conflict,
                resolved=True,
                method='waiting',
                wait_time1=wait_time
            )

        # 如果以上都不行，返回未解决（需要路径重规划）
        return ConflictResolution(
            conflict=conflict,
            resolved=False,
            method='none'
        )

    def get_statistics(self) -> Dict[str, Any]:
        """获取协调统计信息"""
        return {
            'total_robots': len(self.robots),
            'available_robots': sum(1 for r in self.robots.values() if r['available']),
            'pending_tasks': len(self.pending_tasks),
            'total_tasks_allocated': self.stats['total_tasks_allocated'],
            'total_conflicts_detected': self.total_conflicts_detected,
            'conflicts_resolved': self.conflicts_resolved,
            'conflicts_unresolved': self.stats['conflicts_unresolved'],
            'strategy': self.strategy.value,
        }

    def complete_task(self, robot_id: str) -> None:
        """标记任务完成，释放机器人"""
        if robot_id in self.robots:
            self.robots[robot_id]['available'] = True
            self.robots[robot_id]['current_task'] = None


@dataclass
class Obstacle:
    """从 simulation_enhancement 导入的障碍物定义"""
    position: np.ndarray
    size: np.ndarray
    obstacle_type: str  # "static", "dynamic", "human"
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    id: str = ""

    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取AABB包围盒"""
        half = self.size / 2
        min_corner = self.position - half
        max_corner = self.position + half
        return min_corner, max_corner

    def contains_point(self, point: np.ndarray) -> bool:
        """检查点是否在障碍物内"""
        min_corner, max_corner = self.get_bounding_box()
        return np.all(point >= min_corner) and np.all(point <= max_corner)
