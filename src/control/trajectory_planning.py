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
轨迹规划与跟踪模块
================

提供测试所需的轨迹规划与跟踪类:
- TrajectoryPoint / Trajectory: 轨迹数据结构
- TrajectoryPlanner: 基础轨迹规划器
- Waypoint: 航点
- PurePursuitTracker / StanleyTracker / PIDTrajectoryTracker: 跟踪控制器
- RRTStarPlanner: RRT* 路径规划器
- MinimumSnapTrajectory: 最小Snap轨迹生成
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
import math

class TrackerCmd:
    """支持 .v 和 ['v'] 两种访问方式的命令对象"""
    def __init__(self, v, omega):
        self.v = v
        self.omega = omega
        self._data = {'v': v, 'omega': omega}

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def __repr__(self):
        return f"TrackerCmd(v={self.v}, omega={self.omega})"



@dataclass
class Waypoint:
    """航点"""
    x: float
    y: float
    theta: float = 0.0
    v: float = 0.0


@dataclass
class TrajectoryPoint:
    """轨迹点"""
    x: float
    y: float
    theta: float
    v: float
    t: float

    def dist_to(self, other: 'TrajectoryPoint') -> float:
        """到另一个轨迹点的距离"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class Trajectory:
    """轨迹"""
    points: List[TrajectoryPoint]
    total_time: float
    total_length: float

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __iter__(self):
        return iter(self.points)

    def at_time(self, t: float) -> TrajectoryPoint:
        """在指定时间获取轨迹点（线性插值）"""
        if not self.points:
            return TrajectoryPoint(x=0, y=0, theta=0, v=0, t=t)
        if t <= self.points[0].t:
            return self.points[0]
        if t >= self.points[-1].t:
            return self.points[-1]

        for i in range(len(self.points) - 1):
            p0, p1 = self.points[i], self.points[i + 1]
            if p0.t <= t <= p1.t:
                alpha = (t - p0.t) / (p1.t - p0.t + 1e-10)
                return TrajectoryPoint(
                    x=p0.x + alpha * (p1.x - p0.x),
                    y=p0.y + alpha * (p1.y - p0.y),
                    theta=self._interp_angle(p0.theta, p1.theta, alpha),
                    v=p0.v + alpha * (p1.v - p0.v),
                    t=t
                )
        return self.points[-1]

    def closest_point(self, x: float, y: float) -> Tuple[TrajectoryPoint, int]:
        """找到最近的轨迹点"""
        if not self.points:
            return TrajectoryPoint(x=0, y=0, theta=0, v=0, t=0), 0
        min_dist = float('inf')
        best_idx = 0
        for i, pt in enumerate(self.points):
            d = math.sqrt((pt.x - x) ** 2 + (pt.y - y) ** 2)
            if d < min_dist:
                min_dist = d
                best_idx = i
        return self.points[best_idx], best_idx

    @staticmethod
    def _interp_angle(a0: float, a1: float, alpha: float) -> float:
        """返回角度插值的增量（不是绝对值）"""
        diff = a1 - a0
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return alpha * diff


class TrajectoryPlanner:
    """基础轨迹规划器"""

    def __init__(
        self,
        max_v: float = 1.0,
        max_a: float = 0.5,
        max_omega: float = 2.0,
        dt: float = 0.1
    ):
        self.max_v = max_v
        self.max_a = max_a
        self.max_omega = max_omega
        self.dt = dt

    def plan_line(self, start: Waypoint, end: Waypoint) -> Trajectory:
        """直线轨迹规划"""
        dx = end.x - start.x
        dy = end.y - start.y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < 1e-6:
            return Trajectory(points=[TrajectoryPoint(
                x=start.x, y=start.y, theta=start.theta, v=start.v, t=0.0
            )], total_time=0.0, total_length=0.0)

        duration = max(distance / self.max_v, 0.1)
        n_points = max(int(duration / self.dt), 2)

        points = []
        for i in range(n_points + 1):
            alpha = i / n_points
            t = alpha * duration
            theta = math.atan2(dy, dx)
            v = min(self.max_v, distance / (duration + 1e-10))
            points.append(TrajectoryPoint(
                x=start.x + alpha * dx,
                y=start.y + alpha * dy,
                theta=theta,
                v=v,
                t=t
            ))

        return Trajectory(
            points=points,
            total_time=duration,
            total_length=distance
        )

    def plan_arc(self, start: Waypoint, end: Waypoint, curvature: float) -> Trajectory:
        """圆弧轨迹规划（简化）"""
        dx = end.x - start.x
        dy = end.y - start.y
        distance = math.sqrt(dx * dx + dy * dy)

        if curvature == 0:
            return self.plan_line(start, end)

        radius = 1.0 / abs(curvature)
        duration = distance / self.max_v
        n_points = max(int(duration / self.dt), 2)

        points = []
        for i in range(n_points + 1):
            alpha = i / n_points
            t = alpha * duration
            theta = start.theta + alpha * curvature * distance
            v = self.max_v
            points.append(TrajectoryPoint(
                x=start.x + alpha * dx,
                y=start.y + alpha * dy,
                theta=theta,
                v=v,
                t=t
            ))

        return Trajectory(
            points=points,
            total_time=duration,
            total_length=distance
        )

    def plan_path(self, waypoints: List[Waypoint]) -> Trajectory:
        """多航点轨迹规划"""
        if not waypoints:
            return Trajectory(points=[], total_time=0.0, total_length=0.0)
        if len(waypoints) == 1:
            return Trajectory(points=[], total_time=0.0, total_length=0.0)

        all_points = []
        total_time = 0.0
        total_length = 0.0

        for i in range(len(waypoints) - 1):
            segment = self.plan_line(waypoints[i], waypoints[i + 1])
            total_time += segment.total_time
            total_length += segment.total_length
            for pt in segment.points:
                pt.t += total_time - segment.total_time
            all_points.extend(segment.points)

        return Trajectory(
            points=all_points,
            total_time=total_time,
            total_length=total_length
        )


@dataclass
class TrackerCommand:
    """跟踪控制器命令"""
    v: float
    omega: float


class PurePursuitTracker:
    """Pure Pursuit 轨迹跟踪器（测试接口）"""

    def __init__(
        self,
        lookahead_distance: float = 0.5,
        kp: float = 1.0,
        min_lookahead: float = 0.1,
        max_lookahead: float = 2.0
    ):
        self.lookahead_distance = lookahead_distance
        self.kp = kp
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self._trajectory: List[TrajectoryPoint] = []

    def compute(
        self,
        x: float, y: float, theta: float,
        v: float,
        trajectory: List[TrajectoryPoint]
    ) -> TrackerCommand:
        """计算控制命令"""
        self._trajectory = trajectory
        if not trajectory:
            return TrackerCmd(v=0.0, omega=0.0)

        # 找到前看点
        current_pos = np.array([x, y])
        lad = min(self.max_lookahead, max(self.min_lookahead, self.lookahead_distance))

        best_idx = 0
        best_dist = float('inf')
        for i, pt in enumerate(trajectory):
            d = math.sqrt((pt.x - x) ** 2 + (pt.y - y) ** 2)
            if d >= lad and d < best_dist:
                best_dist = d
                best_idx = i

        la = trajectory[best_idx]

        # 计算Pure Pursuit转向
        dx = la.x - x
        dy = la.y - y
        alpha = math.atan2(dy, dx) - theta
        # 归一化
        while alpha > math.pi:
            alpha -= 2 * math.pi
        while alpha < -math.pi:
            alpha += 2 * math.pi

        omega = self.kp * 2 * math.sin(alpha) / (best_dist + 1e-6)

        return TrackerCmd(v=float(v), omega=float(omega))


class StanleyTracker:
    """Stanley 轨迹跟踪器（测试接口）"""

    def __init__(self, k: float = 1.0, k_soft: float = 1.0):
        self.k = k
        self.k_soft = k_soft
        self._trajectory: List[TrajectoryPoint] = []

    def compute(
        self,
        x: float, y: float, theta: float,
        v: float,
        trajectory: List[TrajectoryPoint]
    ) -> TrackerCommand:
        """计算控制命令"""
        self._trajectory = trajectory
        if not trajectory:
            return TrackerCmd(v=0.0, omega=0.0)

        # 找最近点
        min_dist = float('inf')
        nearest_idx = 0
        for i, pt in enumerate(trajectory):
            d = math.sqrt((pt.x - x) ** 2 + (pt.y - y) ** 2)
            if d < min_dist:
                min_dist = d
                nearest_idx = i

        nearest = trajectory[nearest_idx]

        # 横向误差
        cross_track_error = min_dist

        # 航向误差
        heading_error = nearest.theta - theta
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        # Stanley 控制率
        steering = heading_error + math.atan2(self.k * cross_track_error, self.k_soft + abs(v) + 1e-6)

        return TrackerCmd(v=float(v), omega=float(steering))


class PIDTrajectoryTracker:
    """PID 轨迹跟踪器（测试接口）"""

    def __init__(
        self,
        kp_v: float = 1.0, ki_v: float = 0.0, kd_v: float = 0.0,
        kp_omega: float = 1.0, ki_omega: float = 0.0, kd_omega: float = 0.0
    ):
        self.kp_v = kp_v
        self.ki_v = ki_v
        self.kd_v = kd_v
        self.kp_omega = kp_omega
        self.ki_omega = ki_omega
        self.kd_omega = kd_omega
        self._trajectory: List[TrajectoryPoint] = []
        self._prev_v_error = 0.0
        self._v_integral = 0.0
        self._prev_omega_error = 0.0
        self._omega_integral = 0.0

    def compute(
        self,
        x: float, y: float, theta: float,
        trajectory: List[TrajectoryPoint]
    ) -> TrackerCommand:
        """计算控制命令"""
        self._trajectory = trajectory
        if not trajectory:
            return TrackerCmd(v=0.0, omega=0.0)

        # 找最近点
        min_dist = float('inf')
        target_idx = 0
        for i, pt in enumerate(trajectory):
            d = math.sqrt((pt.x - x) ** 2 + (pt.y - y) ** 2)
            if d < min_dist:
                min_dist = d
                target_idx = i

        target = trajectory[min(target_idx + 1, len(trajectory) - 1)]

        # 位置误差
        pos_error = math.sqrt((target.x - x) ** 2 + (target.y - y) ** 2)

        # PID 速度
        self._v_integral += pos_error * 0.01
        self._v_integral = max(-10, min(10, self._v_integral))
        v_deriv = pos_error - self._prev_v_error
        self._prev_v_error = pos_error
        v = max(0, self.kp_v * pos_error + self.ki_v * self._v_integral + self.kd_v * v_deriv)

        # PID 航向
        heading_error = target.theta - theta
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        self._omega_integral += heading_error * 0.01
        self._omega_integral = max(-5, min(5, self._omega_integral))
        omega_deriv = heading_error - self._prev_omega_error
        self._prev_omega_error = heading_error
        omega = self.kp_omega * heading_error + self.ki_omega * self._omega_integral + self.kd_omega * omega_deriv

        return TrackerCmd(v=float(v), omega=float(omega))


class RRTStarPlanner:
    """RRT* 路径规划器（测试接口）"""

    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        max_nodes: int = 500,
        goal_sample_rate: float = 0.1,
        robot_radius: float = 0.2
    ):
        self.bounds = bounds  # [(-5, 5), (-5, 5)]
        self.max_nodes = max_nodes
        self.goal_sample_rate = goal_sample_rate
        self.robot_radius = robot_radius
        self._rng = np.random.default_rng()

    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacles: List[Tuple[Tuple[float, float, float], str]] = None
    ) -> Optional[List[np.ndarray]]:
        """
        路径规划

        Args:
            start: (x, y)
            goal: (x, y)
            obstacles: [((cx, cy, r), 'circle'), ...]

        Returns:
            List of (x, y) waypoints or None
        """
        obstacles = obstacles or []

        # 检查起点/终点是否在障碍物中
        if self._in_obstacle(start[0], start[1], obstacles):
            return None
        if self._in_obstacle(goal[0], goal[1], obstacles):
            return None

        # RRT* 简化实现
        nodes = [np.array(start)]
        parents = [-1]
        goal_idx = None

        for _ in range(self.max_nodes):
            # 采样
            if self._rng.random() < self.goal_sample_rate:
                sample = np.array(goal)
            else:
                sample = np.array([
                    self._rng.uniform(self.bounds[0][0], self.bounds[0][1]),
                    self._rng.uniform(self.bounds[1][0], self.bounds[1][1])
                ])

            # 找最近节点
            nearest_idx = 0
            nearest_dist = float('inf')
            for i, node in enumerate(nodes):
                d = np.linalg.norm(node - sample)
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_idx = i

            # 朝采样点扩展
            direction = sample - nodes[nearest_idx]
            if np.linalg.norm(direction) > 0.1:
                direction = direction / np.linalg.norm(direction) * 0.1

            new_node = nodes[nearest_idx] + direction

            # 碰撞检测
            if self._in_obstacle(new_node[0], new_node[1], obstacles):
                continue

            nodes.append(new_node)
            parents.append(nearest_idx)

            # 检查是否到达目标
            if np.linalg.norm(new_node - np.array(goal)) < 0.2:
                goal_idx = len(nodes) - 1
                break

        if goal_idx is None:
            return None

        # 回溯路径
        path = []
        idx = goal_idx
        while idx != -1:
            path.append(nodes[idx])
            idx = parents[idx]
        path.reverse()

        return path

    def _in_obstacle(
        self,
        x: float, y: float,
        obstacles: List[Tuple[Tuple[float, float, float], str]]
    ) -> bool:
        """检查点是否在障碍物中"""
        for (cx, cy, r), shape in obstacles:
            d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if d < r + self.robot_radius:
                return True
        return False


class MinimumSnapTrajectory:
    """最小Snap轨迹生成（测试接口）"""

    def __init__(
        self,
        waypoints: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        max_velocity: float = 1.0,
        max_acceleration: float = 0.5
    ):
        """
        Args:
            waypoints: (N, D) 路径点数组
            velocities: (N, D) 速度边界条件
            max_velocity: 最大速度
            max_acceleration: 最大加速度
        """
        self.waypoints = waypoints
        self.velocities = velocities
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self._n = len(waypoints)
        self._D = waypoints.shape[1] if waypoints.ndim > 1 else 1

    def compute(self) -> List[np.ndarray]:
        """计算轨迹（返回路径点列表）"""
        if self._n < 2:
            return [self.waypoints[0]] if self._n == 1 else []

        # 简化：等间距采样 + 线性插值
        path = []
        for i in range(self._n - 1):
            p0 = self.waypoints[i]
            p1 = self.waypoints[i + 1]
            steps = max(2, int(np.linalg.norm(p1 - p0) / 0.1))
            for j in range(steps):
                alpha = j / steps
                pt = p0 + alpha * (p1 - p0)
                path.append(pt)
        path.append(self.waypoints[-1])

        return path
