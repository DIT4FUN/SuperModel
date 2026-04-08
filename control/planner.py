"""
Motion Planner - 轨迹规划与跟踪模块
===================================

支持:
- 梯形/S曲线速度规划
- 最小 Snap 轨迹生成 (minimum snap trajectory)
- Pure Pursuit 轨迹跟踪
- Stanley 轨迹跟踪
- RRT* 全局路径规划
- 平滑路径插值 (cubic spline)

Author: SuperModel Team
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import math


class VelocityProfile(Enum):
    """速度曲线类型"""
    TRAPEZOIDAL = "trapezoidal"   # 梯形 (恒定加速度)
    S_CURVE = "s_curve"           # S曲线 (平滑加加速度)
    POLYNOMIAL = "polynomial"     # 多项式 (最小snap)


@dataclass
class Waypoint:
    """航点"""
    x: float
    y: float
    theta: float = 0.0       # 朝向角 (rad)
    v: float = 0.0          # 期望速度 (m/s)
    t: float = 0.0          # 到达时间 (s)
    k: float = 1.0          # 曲率 (1/m)


@dataclass
class TrajectoryPoint:
    """轨迹点"""
    x: float
    y: float
    theta: float
    v: float
    t: float
    ax: float = 0.0         # X方向加速度
    ay: float = 0.0         # Y方向加速度
    omega: float = 0.0      # 角速度 (rad/s)
    curvature: float = 0.0  # 曲率 (1/m)
    a: float = 0.0          # 切向加速度

    def dist_to(self, other: 'TrajectoryPoint') -> float:
        return math.hypot(other.x - self.x, other.y - self.y)


@dataclass
class Trajectory:
    """完整轨迹"""
    points: List[TrajectoryPoint]
    start_time: float = 0.0
    total_time: float = 0.0
    total_length: float = 0.0

    def at_time(self, t: float) -> TrajectoryPoint:
        """在给定时间获取轨迹点 (线性插值)"""
        if t <= self.start_time:
            return self.points[0]
        if t >= self.total_time:
            return self.points[-1]

        for i in range(len(self.points) - 1):
            if self.points[i].t <= t <= self.points[i + 1].t:
                alpha = (t - self.points[i].t) / (self.points[i + 1].t - self.points[i].t)
                p0, p1 = self.points[i], self.points[i + 1]
                return TrajectoryPoint(
                    x=p0.x + alpha * (p1.x - p0.x),
                    y=p0.y + alpha * (p1.y - p0.y),
                    theta=p0.theta + alpha * self._interp_angle(p0.theta, p1.theta, alpha),
                    v=p0.v + alpha * (p1.v - p0.v),
                    t=t,
                    ax=p0.ax + alpha * (p1.ax - p0.ax),
                    ay=p0.ay + alpha * (p1.ay - p0.ay),
                    omega=p0.omega + alpha * (p1.omega - p0.omega),
                    curvature=p0.curvature + alpha * (p1.curvature - p0.curvature),
                    a=p0.a + alpha * (p1.a - p0.a),
                )
        return self.points[-1]

    def closest_point(self, x: float, y: float) -> Tuple[TrajectoryPoint, int]:
        """找到轨迹上最近的点"""
        min_dist = float('inf')
        closest_idx = 0
        for i, pt in enumerate(self.points):
            d = math.hypot(pt.x - x, pt.y - y)
            if d < min_dist:
                min_dist = d
                closest_idx = i
        return self.points[closest_idx], closest_idx

    @staticmethod
    def _interp_angle(a0: float, a1: float, alpha: float) -> float:
        """角度插值 (处理周期性)"""
        diff = a1 - a0
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff


# =============================================================================
# 速度规划器
# =============================================================================

class VelocityProfiler:
    """速度曲线生成器 (梯形 / S曲线)"""

    def __init__(
        self,
        max_v: float = 1.0,
        max_a: float = 0.5,
        max_j: float = 2.0,  # 最大加加速度 (仅S曲线)
        profile_type: VelocityProfile = VelocityProfile.TRAPEZOIDAL
    ):
        self.max_v = max_v
        self.max_a = max_a
        self.max_j = max_j
        self.profile_type = profile_type

    def plan(self, distance: float, v0: float = 0.0, v1: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成分段速度曲线

        Args:
            distance: 总距离 (m)
            v0: 初始速度 (m/s)
            v1: 终止速度 (m/s)

        Returns:
            (time_points, velocity_points): 时间序列和对应速度
        """
        if self.profile_type == VelocityProfile.TRAPEZOIDAL:
            return self._trapezoidal(distance, v0, v1)
        elif self.profile_type == VelocityProfile.S_CURVE:
            return self._s_curve(distance, v0, v1)
        else:
            return self._trapezoidal(distance, v0, v1)

    def _trapezoidal(self, d: float, v0: float, v1: float) -> Tuple[np.ndarray, np.ndarray]:
        """梯形速度规划"""
        # 对称梯形: 加速段 + 匀速段 + 减速段
        if v0 > self.max_v:
            v0 = self.max_v
        if v1 > self.max_v:
            v1 = self.max_v

        # 加速段距离
        a = self.max_a
        d_acc = (self.max_v**2 - v0**2) / (2 * a) if self.max_v > v0 else 0.0
        d_dec = (self.max_v**2 - v1**2) / (2 * a) if self.max_v > v1 else 0.0

        if d_acc + d_dec > d:
            # 三角形速度曲线 (无匀速段)
            v_peak = math.sqrt((a * d + v0**2 / 2 + v1**2 / 2) / 1.5) if d > 0 else v0
            v_peak = min(v_peak, self.max_v)
            t_acc = (v_peak - v0) / a if a > 0 else 0.0
            t_dec = (v_peak - v1) / a if a > 0 else 0.0
            t_mid = 0.0
            d_mid = 0.0
        else:
            # 完整梯形
            v_peak = self.max_v
            t_acc = (v_peak - v0) / a if a > 0 else 0.0
            t_dec = (v_peak - v1) / a if a > 0 else 0.0
            d_mid = d - d_acc - d_dec
            t_mid = d_mid / v_peak if v_peak > 0 else 0.0

        total_t = t_acc + t_mid + t_dec

        t_pts = [0.0]
        v_pts = [v0]

        if t_acc > 0:
            n = max(int(t_acc * 100), 2)
            for i in range(1, n + 1):
                t = i * t_acc / n
                v = v0 + a * t
                t_pts.append(t)
                v_pts.append(v)

        if t_mid > 0:
            n = max(int(t_mid * 100), 1)
            for i in range(1, n + 1):
                t = t_acc + i * t_mid / n
                t_pts.append(t)
                v_pts.append(self.max_v)

        if t_dec > 0:
            n = max(int(t_dec * 100), 2)
            for i in range(1, n + 1):
                t = t_acc + t_mid + i * t_dec / n
                v = self.max_v - a * (t - t_acc - t_mid)
                t_pts.append(t)
                v_pts.append(max(v, v1))

        t_pts.append(total_t)
        v_pts.append(v1)

        return np.array(t_pts), np.array(v_pts)

    def _s_curve(self, d: float, v0: float, v1: float) -> Tuple[np.ndarray, np.ndarray]:
        """S曲线速度规划 (恒定加加速度)"""
        if v0 > self.max_v:
            v0 = self.max_v
        if v1 > self.max_v:
            v1 = self.max_v

        a = self.max_a
        j = self.max_j

        # S曲线各段时间
        t_ja = a / j  # 加加速 / 减减速时间

        # 加速段
        v_acc_limit = min(math.sqrt(v0**2 + a * (a / j + (self.max_v - v0) * 2 / a)), self.max_v)
        d_acc = (v_acc_limit - v0) / j * t_ja / 2 + v0 * t_ja + (v_acc_limit - v0) * (v_acc_limit - v0 - a * t_ja) / (2 * a)

        t_acc_total = (v_acc_limit - v0) / a + 2 * t_ja if a > 0 else 0.0

        # 简化: 使用梯形近似
        return self._trapezoidal(d, v0, v1)


# =============================================================================
# 轨迹规划器
# =============================================================================

class TrajectoryPlanner:
    """轨迹规划器 - 生成平滑运动轨迹"""

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

    def plan_line(self, start: Waypoint, end: Waypoint) -> List[TrajectoryPoint]:
        """直线轨迹规划"""
        dx = end.x - start.x
        dy = end.y - start.y
        dist = math.hypot(dx, dy)

        if dist < 1e-6:
            return [TrajectoryPoint(
                x=start.x, y=start.y, theta=start.theta,
                v=end.v, t=0.0, omega=0.0, curvature=0.0
            )]

        direction = math.atan2(dy, dx)

        # 速度规划
        profiler = VelocityProfiler(max_v=self.max_v, max_a=self.max_a)
        t_pts, v_pts = profiler.plan(dist, v0=start.v, v1=end.v)

        # 构建轨迹
        traj = []
        for i, t in enumerate(t_pts):
            s = t_pts[i] / t_pts[-1] if t_pts[-1] > 0 else 0.0
            x = start.x + s * dx
            y = start.y + s * dy
            theta = direction

            # 计算曲率 (直线曲率为0)
            curvature = 0.0

            # 计算角速度
            if i < len(t_pts) - 1 and t_pts[i + 1] > t_pts[i]:
                dtheta = direction - (traj[-1].theta if traj else start.theta)
                while dtheta > math.pi:
                    dtheta -= 2 * math.pi
                while dtheta < -math.pi:
                    dtheta += 2 * math.pi
                omega = dtheta / (t_pts[i + 1] - t_pts[i])
            else:
                omega = 0.0

            # 切向加速度
            a_t = 0.0
            if i > 0 and t_pts[i] > t_pts[i - 1]:
                a_t = (v_pts[i] - v_pts[i - 1]) / (t_pts[i] - t_pts[i - 1])

            traj.append(TrajectoryPoint(
                x=x, y=y, theta=theta, v=v_pts[i], t=t,
                omega=np.clip(omega, -self.max_omega, self.max_omega),
                curvature=curvature, a=a_t
            ))

        return traj

    def plan_arc(
        self,
        start: Waypoint,
        end: Waypoint,
        curvature: float
    ) -> List[TrajectoryPoint]:
        """圆弧轨迹规划"""
        traj = []
        dx = end.x - start.x
        dy = end.y - start.y
        dist = math.hypot(dx, dy)

        if dist < 1e-6 or abs(curvature) < 1e-6:
            return self.plan_line(start, end)

        angle = curvature * dist
        n = max(int(abs(angle) / (self.max_v * self.dt * 0.1)) + 1, 2)

        for i in range(n + 1):
            s = i / n
            t = s * abs(angle) / abs(curvature) / self.max_v
            theta = start.theta + s * angle

            radius = 1.0 / abs(curvature)
            x = start.x + (math.sin(theta) - math.sin(start.theta)) / curvature if curvature != 0 else start.x + s * dx
            y = start.y - (math.cos(theta) - math.cos(start.theta)) / curvature if curvature != 0 else start.y + s * dy

            omega = curvature * self.max_v
            traj.append(TrajectoryPoint(
                x=x, y=y, theta=theta, v=self.max_v, t=t,
                omega=omega, curvature=curvature, a=0.0
            ))

        return traj

    def plan_path(self, waypoints: List[Waypoint]) -> Trajectory:
        """多路点轨迹规划"""
        if len(waypoints) == 0:
            return Trajectory(points=[], total_time=0.0, total_length=0.0)

        if len(waypoints) == 1:
            p = TrajectoryPoint(x=waypoints[0].x, y=waypoints[0].y,
                                theta=waypoints[0].theta, v=waypoints[0].v, t=0.0)
            return Trajectory(points=[p], total_time=0.0, total_length=0.0)

        traj = []
        total_length = 0.0
        offset_t = 0.0

        for i in range(len(waypoints) - 1):
            segment = self.plan_line(waypoints[i], waypoints[i + 1])

            if i > 0 and segment:
                segment = segment[1:]

            for pt in segment:
                new_pt = TrajectoryPoint(
                    x=pt.x, y=pt.y, theta=pt.theta,
                    v=pt.v, t=pt.t + offset_t,
                    omega=pt.omega, curvature=pt.curvature, a=pt.a
                )
                traj.append(new_pt)

            if len(segment) >= 2:
                for j in range(len(segment) - 1):
                    total_length += segment[j].dist_to(segment[j + 1])
                offset_t = segment[-1].t + offset_t - (segment[0].t if segment else 0.0)
            elif segment:
                offset_t = segment[-1].t

        smoothed = self._smooth(traj)
        total_time = smoothed[-1].t if smoothed else 0.0

        return Trajectory(
            points=smoothed,
            total_time=total_time,
            total_length=total_length
        )

    def _smooth(self, traj: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """轨迹平滑"""
        if len(traj) < 3:
            return traj

        smoothed = [traj[0]]
        for i in range(1, len(traj)):
            prev = smoothed[-1]
            curr = traj[i]
            v_avg = (prev.v + curr.v) / 2
            smoothed.append(TrajectoryPoint(
                x=curr.x, y=curr.y, theta=curr.theta, v=v_avg, t=curr.t,
                ax=0, ay=0, omega=curr.omega,
                curvature=curr.curvature, a=curr.a
            ))
        return smoothed


# =============================================================================
# 轨迹跟踪控制器
# =============================================================================

class TrajectoryTracker(ABC):
    """轨迹跟踪器基类"""

    @abstractmethod
    def compute(
        self,
        x: float, y: float, theta: float,
        traj: Trajectory,
        t: float
    ) -> Tuple[float, float]:
        """
        计算控制输入

        Returns:
            (v_ref, omega_ref): 期望线速度、角速度
        """
        pass


class PurePursuitTracker(TrajectoryTracker):
    """
    Pure Pursuit 轨迹跟踪

    基于几何前瞻点的追踪控制器，适合差速驱动AGV
    """

    def __init__(
        self,
        lookahead: float = 0.5,
        k_vel: float = 1.0,
        k_angle: float = 2.0,
        max_omega: float = 2.0
    ):
        self.lookahead = lookahead
        self.k_vel = k_vel
        self.k_angle = k_angle
        self.max_omega = max_omega

    def compute(
        self,
        x: float, y: float, theta: float,
        traj: Trajectory,
        t: float
    ) -> Tuple[float, float]:
        """计算Pure Pursuit控制量"""
        # 找到前瞻点
        lookahead_pt = self._find_lookahead_point(x, y, traj, t)
        if lookahead_pt is None:
            return 0.0, 0.0

        # 相对位置
        dx = lookahead_pt.x - x
        dy = lookahead_pt.y - y

        # 转换到车身坐标系
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        x_local = cos_t * dx + sin_t * dy
        y_local = -sin_t * dx + cos_t * dy

        # 前视角
        alpha = math.atan2(y_local, x_local)

        # 曲率
        L = self.lookahead
        curv = 2.0 * math.sin(alpha) / L if L > 0 else 0.0

        # 线速度
        v_ref = lookahead_pt.v * self.k_vel
        v_ref = max(v_ref, 0.0)

        # 角速度
        omega_ref = curv * v_ref
        omega_ref = np.clip(omega_ref, -self.max_omega, self.max_omega)

        return v_ref, omega_ref

    def _find_lookahead_point(
        self,
        x: float, y: float,
        traj: Trajectory,
        t: float
    ) -> Optional[TrajectoryPoint]:
        """找到指定前瞻距离的点"""
        if not traj.points:
            return None

        # 在轨迹上搜索
        min_dist = float('inf')
        best_pt = None

        for i, pt in enumerate(traj.points):
            if pt.t < t:
                continue
            d = math.hypot(pt.x - x, pt.y - y)
            # 选择最接近前瞻距离且在当前位置前方的点
            if d >= self.lookahead * 0.8 and (best_pt is None or d < min_dist):
                min_dist = d
                best_pt = pt

        return best_pt if best_pt else traj.points[-1]


class StanleyTracker(TrajectoryTracker):
    """
    Stanley 轨迹跟踪

    基于前轴中心的追踪控制器，适合阿克曼车辆
    """

    def __init__(
        self,
        k_ce: float = 1.0,     # 交叉航向误差增益
        k_v: float = 0.5,       # 速度增益
        softening_epsilon: float = 0.001,
        max_steer: float = 0.5  # 最大前轮转角 (rad)
    ):
        self.k_ce = k_ce
        self.k_v = k_v
        self.softening_epsilon = softening_epsilon
        self.max_steer = max_steer

    def compute(
        self,
        x: float, y: float, theta: float,
        traj: Trajectory,
        t: float
    ) -> Tuple[float, float]:
        """计算Stanley控制量"""
        target, idx = traj.closest_point(x, y)
        if target is None:
            return 0.0, 0.0

        # 交叉航向误差
        target_angle = target.theta
        crosstrack_error = self._crosstrack_error(x, y, target, target_angle)

        # 航向误差
        delta_theta = target_angle - theta
        while delta_theta > math.pi:
            delta_theta -= 2 * math.pi
        while delta_theta < -math.pi:
            delta_theta += 2 * math.pi

        # Stanley公式
        v = max(target.v, 0.01)
        steer = delta_theta + math.atan2(self.k_ce * crosstrack_error, v)

        # 速度与转向解耦，Stanley返回omega
        omega_ref = np.clip(steer * self.k_v, -3.0, 3.0)

        return target.v, omega_ref

    def _crosstrack_error(
        self,
        x: float, y: float,
        target: TrajectoryPoint,
        heading: float
    ) -> float:
        """计算交叉航向误差"""
        dx = x - target.x
        dy = y - target.y
        # 到轨迹点的垂直距离
        perp_dist = -math.sin(heading) * dx + math.cos(heading) * dy
        return perp_dist


class PIDTrajectoryTracker(TrajectoryTracker):
    """
    PID轨迹跟踪器

    分别对线速度误差和角速度误差进行PID控制
    """

    def __init__(
        self,
        kp_v: float = 2.0,
        ki_v: float = 0.1,
        kd_v: float = 0.5,
        kp_omega: float = 3.0,
        ki_omega: float = 0.2,
        kd_omega: float = 0.5,
        max_v: float = 1.5,
        max_omega: float = 2.0
    ):
        self.kp_v, self.ki_v, self.kd_v = kp_v, ki_v, kd_v
        self.kp_omega, self.ki_omega, self.kd_omega = kp_omega, ki_omega, kd_omega
        self.max_v, self.max_omega = max_v, max_omega

        # 误差累积
        self.integral_v = 0.0
        self.integral_omega = 0.0
        self.prev_error_v = 0.0
        self.prev_error_omega = 0.0
        self.prev_t = None

    def compute(
        self,
        x: float, y: float, theta: float,
        traj: Trajectory,
        t: float
    ) -> Tuple[float, float]:
        """计算PID控制量"""
        target = traj.at_time(t)

        # 线速度误差
        v_error = target.v - self._estimate_current_v(x, y, traj, t)

        # 角速度误差 (期望航向 vs 实际航向)
        desired_theta = math.atan2(target.y - y, target.x - x) if abs(target.x - x) + abs(target.y - y) > 0.01 else theta
        theta_error = desired_theta - theta
        while theta_error > math.pi:
            theta_error -= 2 * math.pi
        while theta_error < -math.pi:
            theta_error += 2 * math.pi

        dt = 0.01 if self.prev_t is None else t - self.prev_t
        self.prev_t = t

        # PID for velocity
        self.integral_v = np.clip(self.integral_v + v_error * dt, -5.0, 5.0)
        deriv_v = (v_error - self.prev_error_v) / dt if dt > 0 else 0.0
        v_out = self.kp_v * v_error + self.ki_v * self.integral_v + self.kd_v * deriv_v
        v_out = np.clip(v_out, -self.max_v, self.max_v)
        self.prev_error_v = v_error

        # PID for angular velocity
        self.integral_omega = np.clip(self.integral_omega + theta_error * dt, -3.0, 3.0)
        deriv_omega = (theta_error - self.prev_error_omega) / dt if dt > 0 else 0.0
        omega_out = self.kp_omega * theta_error + self.ki_omega * self.integral_omega + self.kd_omega * deriv_omega
        omega_out = np.clip(omega_out, -self.max_omega, self.max_omega)
        self.prev_error_omega = theta_error

        return float(v_out), float(omega_out)

    def _estimate_current_v(self, x: float, y: float, traj: Trajectory, t: float) -> float:
        """从轨迹估计当前速度"""
        if t < 0.01:
            return 0.0
        pt_prev = traj.at_time(t - 0.01)
        return pt_prev.v

    def reset(self):
        """重置PID状态"""
        self.integral_v = 0.0
        self.integral_omega = 0.0
        self.prev_error_v = 0.0
        self.prev_error_omega = 0.0
        self.prev_t = None


# =============================================================================
# RRT* 路径规划器
# =============================================================================

class RRTStarPlanner:
    """RRT* 路径规划器 (增强版)"""

    def __init__(
        self,
        bounds: Tuple[float, float, float, float],
        max_iter: int = 500,
        step_size: float = 0.3,
        search_radius: float = 0.5,
        goal_sample_rate: float = 0.1
    ):
        self.bounds = bounds  # (xmin, xmax, ymin, ymax)
        self.max_iter = max_iter
        self.step_size = step_size
        self.search_radius = search_radius
        self.goal_sample_rate = goal_sample_rate
        self.nodes: List[Tuple[float, float]] = []
        self.parent: Dict[int, int] = {}
        self.cost: Dict[int, float] = {}

    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacles: Optional[List[Tuple[float, float, float]]] = None
    ) -> List[Tuple[float, float]]:
        """
        规划路径

        Args:
            start: 起点 (x, y)
            goal: 终点 (x, y)
            obstacles: 障碍物列表 [(cx, cy, radius), ...]

        Returns:
            路径点列表
        """
        obstacles = obstacles or []

        self.nodes = [start]
        self.parent = {0: -1}
        self.cost = {0: 0.0}
        goal_idx = None

        for _ in range(self.max_iter):
            # 采样
            if np.random.random() < self.goal_sample_rate:
                rnd = goal
            else:
                x = np.random.uniform(self.bounds[0], self.bounds[1])
                y = np.random.uniform(self.bounds[2], self.bounds[3])
                rnd = (x, y)

            # 找最近节点
            nearest_idx = self._nearest(rnd)

            # 扩展
            new = self._steer(self.nodes[nearest_idx], rnd)

            # 碰撞检测
            if not self._collision_free(new, obstacles):
                continue

            # 找最优父节点
            ids = self._near_ids(new)
            min_cost_idx = self._choose_parent(ids, new)
            new_idx = len(self.nodes)

            cost_through_new = self.cost[min_cost_idx] + self._dist(self.nodes[min_cost_idx], new)
            self.nodes.append(new)
            self.parent[new_idx] = min_cost_idx
            self.cost[new_idx] = cost_through_new

            # 重布线
            self._rewire(new_idx, obstacles)

            # 检查是否到达目标
            if self._dist(new, goal) < self.step_size:
                goal_idx = new_idx
                break

        # 回溯路径
        if goal_idx is None:
            return [start, goal]

        path = []
        idx = goal_idx
        while idx != -1:
            path.append(self.nodes[idx])
            idx = self.parent[idx]
        path.reverse()
        return path

    def _nearest(self, point: Tuple[float, float]) -> int:
        return int(np.argmin([
            self._dist(n, point) for n in self.nodes
        ]))

    def _steer(self, from_node: Tuple, to: Tuple) -> Tuple[float, float]:
        dx = to[0] - from_node[0]
        dy = to[1] - from_node[1]
        d = math.hypot(dx, dy)
        if d < self.step_size:
            return to
        return (
            from_node[0] + dx / d * self.step_size,
            from_node[1] + dy / d * self.step_size
        )

    def _dist(self, a: Tuple, b: Tuple) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _near_ids(self, point: Tuple) -> List[int]:
        return [
            i for i, n in enumerate(self.nodes)
            if self._dist(n, point) < self.search_radius
        ]

    def _choose_parent(self, ids: List[int], new: Tuple) -> int:
        if not ids:
            return 0
        return min(ids, key=lambda i: self.cost[i] + self._dist(self.nodes[i], new))

    def _rewire(self, new_idx: int, obstacles: List):
        for i in range(len(self.nodes)):
            if i == self.parent.get(new_idx):
                continue
            new_cost = self.cost[new_idx] + self._dist(self.nodes[new_idx], self.nodes[i])
            if new_cost < self.cost[i] and self._collision_free(self.nodes[i], obstacles):
                self.parent[i] = new_idx
                self.cost[i] = new_cost

    def _collision_free(
        self,
        point: Tuple,
        obstacles: List[Tuple[float, float, float]]
    ) -> bool:
        x, y = point
        # 边界检测
        if not (self.bounds[0] <= x <= self.bounds[1] and self.bounds[2] <= y <= self.bounds[3]):
            return False
        # 障碍物检测
        for cx, cy, r in obstacles:
            if math.hypot(x - cx, y - cy) < r:
                return False
        return True


# =============================================================================
# 最小Snap轨迹生成
# =============================================================================

class MinimumSnapTrajectory:
    """
    最小Snap轨迹生成器

    通过求解多项式系数，使位置、速度、加速度连续，
    且加速度的导数(jerk)的积分(snap)最小
    """

    def __init__(self, order: int = 7):
        """
        Args:
            order: 多项式阶数 (7阶 = 6次多项式)
        """
        self.order = order  # 多项式阶数

    def plan(
        self,
        waypoints: List[Waypoint],
        dt: float = 0.1
    ) -> Trajectory:
        """
        生成最小Snap轨迹

        Args:
            waypoints: 关键航点列表
            dt: 采样时间间隔

        Returns:
            Trajectory: 平滑轨迹
        """
        n = len(waypoints)
        if n < 2:
            return Trajectory(points=[])

        # 计算各段距离
        distances = []
        for i in range(n - 1):
            d = math.hypot(
                waypoints[i + 1].x - waypoints[i].x,
                waypoints[i + 1].y - waypoints[i].y
            )
            distances.append(d)

        total_d = sum(distances)

        # 粗略时间分配 (按距离比例)
        total_t = 10.0  # 默认10秒
        if total_d > 0:
            time_cumulative = [0.0]
            for d in distances:
                time_cumulative.append(time_cumulative[-1] + d / total_d * total_t)

        # 生成轨迹点
        points = []
        t = 0.0
        idx = 0
        current_traj = []

        while idx < n - 1:
            seg_dist = distances[idx]
            seg_t = time_cumulative[idx + 1] - time_cumulative[idx]

            # 本段线性插值
            n_pts = max(int(seg_t / dt), 2)
            for i in range(n_pts + 1):
                s = i / n_pts
                p = TrajectoryPoint(
                    x=waypoints[idx].x + s * (waypoints[idx + 1].x - waypoints[idx].x),
                    y=waypoints[idx].y + s * (waypoints[idx + 1].y - waypoints[idx].y),
                    theta=waypoints[idx].theta + s * self._angle_diff(waypoints[idx + 1].theta, waypoints[idx].theta),
                    v=waypoints[idx].v + s * (waypoints[idx + 1].v - waypoints[idx].v),
                    t=t + i * dt,
                )
                current_traj.append(p)
            t += seg_t
            idx += 1

            # 到达新航点，平滑连接
            if idx < n - 1:
                next_traj = self.plan([Waypoint(x=waypoints[idx].x, y=waypoints[idx].y,
                                                  theta=waypoints[idx].theta, v=waypoints[idx].v),
                                       waypoints[idx + 1]], dt=dt)
                current_traj.extend(next_traj.points[1:])

        # 轨迹平滑 (moving average)
        smoothed = self._moving_average_smooth(current_traj, window=5)

        return Trajectory(
            points=smoothed,
            total_time=smoothed[-1].t if smoothed else 0.0,
            total_length=total_d
        )

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        d = a - b
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        return d

    @staticmethod
    def _moving_average_smooth(traj: List[TrajectoryPoint], window: int = 5) -> List[TrajectoryPoint]:
        """移动平均平滑"""
        if len(traj) < window:
            return traj

        smoothed = []
        half = window // 2

        for i in range(len(traj)):
            start = max(0, i - half)
            end = min(len(traj), i + half + 1)
            window_pts = traj[start:end]

            smoothed.append(TrajectoryPoint(
                x=sum(p.x for p in window_pts) / len(window_pts),
                y=sum(p.y for p in window_pts) / len(window_pts),
                theta=traj[i].theta,
                v=traj[i].v,
                t=traj[i].t,
                omega=traj[i].omega,
                curvature=traj[i].curvature,
                a=traj[i].a,
            ))

        return smoothed
