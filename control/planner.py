"""Motion Planner - 轨迹规划模块"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class Waypoint:
    """航点"""
    x: float
    y: float
    theta: float = 0.0
    v: float = 0.0
    t: float = 0.0


@dataclass
class TrajectoryPoint:
    """轨迹点"""
    x: float
    y: float
    theta: float
    v: float
    t: float
    ax: float = 0.0
    ay: float = 0.0
    omega: float = 0.0


class TrajectoryPlanner:
    """轨迹规划器 - 生成平滑运动轨迹"""

    def __init__(self, max_v: float = 1.0, max_a: float = 0.5, dt: float = 0.1):
        self.max_v = max_v
        self.max_a = max_a
        self.dt = dt

    def plan_line(self, start: Waypoint, end: Waypoint) -> List[TrajectoryPoint]:
        """直线轨迹规划"""
        dist = np.hypot(end.x - start.x, end.y - start.y)
        direction = np.arctan2(end.y - start.y, end.x - start.x)
        n = max(int(dist / (self.max_v * self.dt)) + 1, 2)
        traj = []
        for i in range(n + 1):
            s = i / n
            t = s * dist / self.max_v
            x = start.x + s * (end.x - start.x)
            y = start.y + s * (end.y - start.y)
            theta = direction if n > 1 else start.theta
            traj.append(TrajectoryPoint(x=x, y=y, theta=theta, v=self.max_v, t=t))
        return traj

    def plan_arc(self, start: Waypoint, end: Waypoint, curvature: float) -> List[TrajectoryPoint]:
        """圆弧轨迹规划"""
        traj = []
        angle = curvature * np.hypot(end.x - start.x, end.y - start.y)
        n = max(int(abs(angle) / (self.max_v * self.dt * 0.1)) + 1, 2)
        for i in range(n + 1):
            s = i / n
            t = s * abs(angle) / abs(curvature) / self.max_v
            theta = start.theta + s * angle
            x = start.x + (np.sin(theta) - np.sin(start.theta)) / curvature
            y = start.y - (np.cos(theta) - np.cos(start.theta)) / curvature
            traj.append(TrajectoryPoint(x=x, y=y, theta=theta, v=self.max_v, t=t))
        return traj

    def smooth_trajectory(self, traj: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """轨迹平滑 (简单匀速平滑)"""
        if len(traj) < 3:
            return traj
        smoothed = [traj[0]]
        for i in range(1, len(traj)):
            pt = traj[i]
            prev = smoothed[-1]
            v_avg = (prev.v + pt.v) / 2
            smoothed.append(TrajectoryPoint(
                x=pt.x, y=pt.y, theta=pt.theta, v=v_avg, t=pt.t,
                ax=0, ay=0, omega=pt.omega
            ))
        return smoothed

    def plan_path(self, waypoints: List[Waypoint]) -> List[TrajectoryPoint]:
        """多路点轨迹规划"""
        traj = []
        for i in range(len(waypoints) - 1):
            segment = self.plan_line(waypoints[i], waypoints[i + 1])
            if i > 0:
                segment = segment[1:]
            traj.extend(segment)
        return self.smooth_trajectory(traj)


class RRTStarPlanner:
    """RRT* 路径规划器"""

    def __init__(self, bounds: Tuple[float, float, float, float], max_iter: int = 500):
        self.bounds = bounds  # (xmin, xmax, ymin, ymax)
        self.max_iter = max_iter
        self.step_size = 0.3
        self.search_radius = 0.5

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """规划路径"""
        nodes = [start]
        parent = {0: -1}
        goal_idx = None

        for _ in range(self.max_iter):
            if np.random.random() > 0.1:
                x = np.random.uniform(self.bounds[0], self.bounds[1])
                y = np.random.uniform(self.bounds[2], self.bounds[3])
                rnd = (x, y)
            else:
                rnd = goal

            nearest_idx = self._nearest(nodes, rnd)
            new = self._steer(nodes[nearest_idx], rnd)

            if self._collision_free(nodes[nearest_idx], new):
                ids = self._near_ids(nodes, new)
                min_cost_idx = self._choose_parent(ids, nodes, parent, new)
                new_idx = len(nodes)
                nodes.append(new)
                parent[new_idx] = min_cost_idx
                self._rewire(new_idx, nodes, parent)

                if np.hypot(new[0] - goal[0], new[1] - goal[1]) < 0.5:
                    goal_idx = new_idx
                    break

        if goal_idx is None:
            return [start, goal]

        path = []
        idx = goal_idx
        while idx != -1:
            path.append(nodes[idx])
            idx = parent[idx]
        path.reverse()
        return path

    def _nearest(self, nodes: List, point: Tuple) -> int:
        return int(np.argmin([np.hypot(n[0] - point[0], n[1] - point[1]) for n in nodes]))

    def _steer(self, from_node: Tuple, to: Tuple) -> Tuple[float, float]:
        dx = to[0] - from_node[0]
        dy = to[1] - from_node[1]
        d = np.hypot(dx, dy)
        if d < self.step_size:
            return to
        return (from_node[0] + dx / d * self.step_size,
                from_node[1] + dy / d * self.step_size)

    def _collision_free(self, a: Tuple, b: Tuple) -> bool:
        return True

    def _near_ids(self, nodes: List, point: Tuple) -> List[int]:
        return [i for i, n in enumerate(nodes)
                if np.hypot(n[0] - point[0], n[1] - point[1]) < self.search_radius]

    def _choose_parent(self, ids: List[int], nodes: List, parent: Dict, new: Tuple) -> int:
        return ids[0] if ids else 0

    def _rewire(self, new_idx: int, nodes: List, parent: Dict):
        pass
