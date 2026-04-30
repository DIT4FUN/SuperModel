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
AGV导航控制模块
==============

集成式AGV导航系统，融合:
- 全局路径规划 (A* / Dijkstra / RRT)
- 局部避障 (DWA / APF / VFH)
- 运动控制 (PID / MPC)
- 地图管理与定位

支持AGV等级: S / M / L / XL / XXL
支持仿真/实机部署
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Callable
from enum import Enum
import heapq
import math


class PlannerType(Enum):
    """全局规划器类型"""
    DIJKSTRA = "dijkstra"
    A_STAR = "astar"
    RRT = "rrt"
    RRT_STAR = "rrt_star"


class NavigationState(Enum):
    """导航状态"""
    IDLE = "idle"
    PLANNING = "planning"
    NAVIGATING = "navigating"
    AVOIDING = "avoiding"
    ARRIVED = "arrived"
    FAILED = "failed"
    ESTOP = "emergency_stop"


@dataclass
class Waypoint:
    """路径航点"""
    x: float
    y: float
    theta: float = 0.0
    speed: float = 0.0
    is_via: bool = True  # 是否为途经点

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta, self.speed])


@dataclass
class Path:
    """全局路径"""
    waypoints: List[Waypoint]
    length: float
    planner: PlannerType
    cost: float
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = 0.0

    def get_position_array(self) -> np.ndarray:
        """获取位置数组 (N x 2)"""
        return np.array([[w.x, w.y] for w in self.waypoints])

    def get_heading_angle(self, index: int) -> float:
        """获取航向角"""
        if index < len(self.waypoints) - 1:
            dx = self.waypoints[index + 1].x - self.waypoints[index].x
            dy = self.waypoints[index + 1].y - self.waypoints[index].y
            return math.atan2(dy, dx)
        return self.waypoints[index].theta


@dataclass
class OccupancyGrid:
    """占用栅格地图"""
    width: float          # 地图宽度 m
    height: float         # 地图高度 m
    resolution: float      # 分辨率 m/cell
    origin_x: float = 0.0  # 原点 x
    origin_y: float = 0.0  # 地图原点 y

    def __post_init__(self):
        self.cols = int(self.width / self.resolution)
        self.rows = int(self.height / self.resolution)
        self.grid = np.zeros((self.rows, self.cols), dtype=np.float32)

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """世界坐标转栅格坐标"""
        gx = int((wx - self.origin_x) / self.resolution)
        gy = int((wy - self.origin_y) / self.resolution)
        gx = np.clip(gx, 0, self.cols - 1)
        gy = np.clip(gy, 0, self.rows - 1)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """栅格坐标转世界坐标"""
        wx = self.origin_x + (gx + 0.5) * self.resolution
        wy = self.origin_y + (gy + 0.5) * self.resolution
        return wx, wy

    def set_obstacle(self, wx: float, wy: float, radius: float = 0.0) -> None:
        """设置障碍物"""
        gx, gy = self.world_to_grid(wx, wy)
        r_cells = int(radius / self.resolution) + 1
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows:
                    self.grid[ny, nx] = 1.0

    def is_free(self, wx: float, wy: float) -> bool:
        """检查是否空闲"""
        gx, gy = self.world_to_grid(wx, wy)
        return self.grid[gy, gx] < 0.5

    def get_nearby_obstacles(self, wx: float, wy: float, radius: float) -> List[Tuple[int, int]]:
        """获取附近障碍物"""
        gx, gy = self.world_to_grid(wx, wy)
        r = int(radius / self.resolution) + 1
        obstacles = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows and self.grid[ny, nx] > 0.5:
                    obstacles.append((nx, ny))
        return obstacles


class DijkstraPlanner:
    """Dijkstra 全局路径规划器"""

    def __init__(self, grid: OccupancyGrid):
        self.grid = grid

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[Path]:
        """规划路径"""
        sx, sy = self.grid.world_to_grid(*start)
        gx, gy = self.grid.world_to_grid(*goal)

        if not (0 <= sx < self.grid.cols and 0 <= sy < self.grid.rows):
            return None
        if not (0 <= gx < self.grid.cols and 0 <= gy < self.grid.rows):
            return None
        if self.grid.grid[sy, sx] > 0.5 or self.grid.grid[gy, gx] > 0.5:
            return None

        # 4方向移动
        DIRS = [(0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
                (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]

        open_set = [(0.0, sx, sy)]
        came_from = {}
        cost_so_far = {(sx, sy): 0.0}

        while open_set:
            _, cx, cy = heapq.heappop(open_set)

            if (cx, cy) == (gx, gy):
                return self._reconstruct_path(came_from, (sx, sy), (gx, gy))

            for dx, dy, cost in DIRS:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.grid.cols and 0 <= ny < self.grid.rows:
                    if self.grid.grid[ny, nx] > 0.5:
                        continue
                    new_cost = cost_so_far[(cx, cy)] + cost
                    if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                        cost_so_far[(nx, ny)] = new_cost
                        heapq.heappush(open_set, (new_cost, nx, ny))
                        came_from[(nx, ny)] = (cx, cy)

        return None

    def _reconstruct_path(self, came_from: Dict, start: Tuple, goal: Tuple) -> Path:
        """重建路径"""
        current = goal
        path_cells = [current]
        while current != start:
            current = came_from[current]
            path_cells.append(current)
        path_cells.reverse()

        waypoints = []
        total_length = 0.0
        for i, (gx, gy) in enumerate(path_cells):
            wx, wy = self.grid.grid_to_world(gx, gy)
            theta = 0.0
            if i < len(path_cells) - 1:
                ngx, ngy = path_cells[i + 1]
                theta = math.atan2(ngy - gy, ngx - gx)
            waypoints.append(Waypoint(wx, wy, theta=theta, speed=1.0))

        for i in range(len(path_cells) - 1):
            x1, y1 = path_cells[i]
            x2, y2 = path_cells[i + 1]
            total_length += math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * self.grid.resolution

        return Path(
            waypoints=waypoints,
            length=total_length,
            planner=PlannerType.DIJKSTRA,
            cost=total_length
        )


class AStarPlanner(DijkstraPlanner):
    """A* 路径规划器 (继承Dijkstra,增加启发式)"""

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[Path]:
        sx, sy = self.grid.world_to_grid(*start)
        gx, gy = self.grid.world_to_grid(*goal)

        if not (0 <= sx < self.grid.cols and 0 <= sy < self.grid.rows):
            return None
        if not (0 <= gx < self.grid.cols and 0 <= gy < self.grid.rows):
            return None
        if self.grid.grid[sy, sx] > 0.5 or self.grid.grid[gy, gx] > 0.5:
            return None

        DIRS = [(0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
                (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        open_set = [(heuristic((sx, sy), (gx, gy)), 0.0, sx, sy)]
        came_from = {}
        g_cost = {(sx, sy): 0.0}

        while open_set:
            _, _, cx, cy = heapq.heappop(open_set)

            if (cx, cy) == (gx, gy):
                p = self._reconstruct_path(came_from, (sx, sy), (gx, gy))
                if p:
                    p.planner = PlannerType.A_STAR
                return p

            for dx, dy, cost in DIRS:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.grid.cols and 0 <= ny < self.grid.rows:
                    if self.grid.grid[ny, nx] > 0.5:
                        continue
                    new_g = g_cost[(cx, cy)] + cost
                    if (nx, ny) not in g_cost or new_g < g_cost[(nx, ny)]:
                        g_cost[(nx, ny)] = new_g
                        f = new_g + heuristic((nx, ny), (gx, gy))
                        heapq.heappush(open_set, (f, new_g, nx, ny))
                        came_from[(nx, ny)] = (cx, cy)

        return None


class NavigationController:
    """
    AGV导航控制器

    功能:
    - 全局路径规划 (A* / Dijkstra)
    - 局部避障 (DWA 集成)
    - 轨迹跟踪 (PID)
    - 地图管理与定位
    """

    def __init__(
        self,
        grid: OccupancyGrid,
        planner_type: PlannerType = PlannerType.A_STAR,
        max_speed: float = 1.0,
        max_accel: float = 1.0,
        goal_tolerance: float = 0.1,
        angle_tolerance: float = 0.1,
    ):
        self.grid = grid
        self.planner_type = planner_type
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.goal_tolerance = goal_tolerance
        self.angle_tolerance = angle_tolerance

        # 初始化规划器
        if planner_type == PlannerType.DIJKSTRA:
            self.planner = DijkstraPlanner(grid)
        elif planner_type == PlannerType.A_STAR:
            self.planner = AStarPlanner(grid)
        else:
            self.planner = AStarPlanner(grid)

        # 状态
        self.state = NavigationState.IDLE
        self.current_path: Optional[Path] = None
        self.current_pose: np.ndarray = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.current_vel: np.ndarray = np.array([0.0, 0.0, 0.0])   # vx, vy, omega
        self.goal_pose: Optional[np.ndarray] = None
        self.target_idx: int = 0

        # PID跟踪参数
        self.kp_dist = 2.0
        self.kp_angle = 3.0
        self.integral_limit = 0.5

    def set_global_path(self, path: Path) -> None:
        """设置全局路径"""
        self.current_path = path
        self.target_idx = 0
        self.state = NavigationState.NAVIGATING

    def plan_to_goal(self, start: np.ndarray, goal: np.ndarray) -> bool:
        """规划到目标"""
        self.state = NavigationState.PLANNING
        self.goal_pose = goal

        path = self.planner.plan(
            (start[0], start[1]),
            (goal[0], goal[1])
        )

        if path is None:
            self.state = NavigationState.FAILED
            return False

        self.current_path = path
        self.target_idx = 0
        self.state = NavigationState.NAVIGATING
        return True

    def update(self, current_pose: np.ndarray, dt: float) -> np.ndarray:
        """
        更新导航控制

        Args:
            current_pose: [x, y, theta] 当前位姿
            dt: 时间步长

        Returns:
            [vx, vy, omega] 速度指令
        """
        self.current_pose = current_pose.copy()

        if self.state not in [NavigationState.NAVIGATING, NavigationState.AVOIDING]:
            return np.array([0.0, 0.0, 0.0])

        if self.current_path is None or len(self.current_path.waypoints) == 0:
            self.state = NavigationState.ARRIVED
            return np.array([0.0, 0.0, 0.0])

        # 检查是否到达终点
        if self.target_idx >= len(self.current_path.waypoints) - 1:
            wp = self.current_path.waypoints[-1]
            dist = math.sqrt((current_pose[0] - wp.x)**2 + (current_pose[1] - wp.y)**2)
            if dist < self.goal_tolerance:
                self.state = NavigationState.ARRIVED
                return np.array([0.0, 0.0, 0.0])

        # 获取当前目标航点
        wp = self.current_path.waypoints[self.target_idx]
        target_x, target_y = wp.x, wp.y

        # 计算到航点距离
        dx = target_x - current_pose[0]
        dy = target_y - current_pose[1]
        dist_to_wp = math.sqrt(dx**2 + dy**2)

        # 前进到下一个航点
        if dist_to_wp < self.goal_tolerance * 2:
            self.target_idx = min(self.target_idx + 1, len(self.current_path.waypoints) - 1)
            if self.target_idx >= len(self.current_path.waypoints) - 1:
                self.state = NavigationState.ARRIVED
                return np.array([0.0, 0.0, 0.0])

        # 角度控制
        target_angle = math.atan2(dy, dx)
        angle_diff = target_angle - current_pose[2]
        # 归一化到 [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # 速度指令
        v_linear = np.clip(self.kp_dist * dist_to_wp, 0, self.max_speed)
        omega = np.clip(self.kp_angle * angle_diff, -3.0, 3.0)

        return np.array([v_linear * math.cos(current_pose[2]),
                         v_linear * math.sin(current_pose[2]),
                         omega])

    def reset(self) -> None:
        """重置导航状态"""
        self.state = NavigationState.IDLE
        self.current_path = None
        self.goal_pose = None
        self.target_idx = 0

    def emergency_stop(self) -> None:
        """紧急停止"""
        self.state = NavigationState.ESTOP

    def get_state(self) -> NavigationState:
        return self.state

    def get_progress(self) -> float:
        """获取导航进度 0.0 ~ 1.0"""
        if self.current_path is None:
            return 0.0
        total = len(self.current_path.waypoints)
        if total == 0:
            return 1.0
        return min(self.target_idx / max(total - 1, 1), 1.0)


def create_navigation_grid(
    width: float,
    height: float,
    resolution: float = 0.05,
    obstacles: Optional[List[Tuple[float, float, float]]] = None,
) -> OccupancyGrid:
    """
    创建导航栅格地图

    Args:
        width: 地图宽度 m
        height: 地图高度 m
        resolution: 分辨率 m/cell
        obstacles: [(x, y, radius), ...] 障碍物列表

    Returns:
        OccupancyGrid 栅格地图
    """
    grid = OccupancyGrid(width, height, resolution)
    if obstacles:
        for x, y, r in obstacles:
            grid.set_obstacle(x, y, r)
    return grid
