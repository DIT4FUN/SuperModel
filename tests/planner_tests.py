"""
轨迹规划模块单元测试
测试 control/planner.py 中的轨迹规划与跟踪功能
覆盖: VelocityProfiler, PurePursuitTracker, StanleyTracker, PIDTrajectoryTracker, RRTStarPlanner, MinimumSnapTrajectory
"""

import unittest
import numpy as np
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.planner import (
    TaskPlanner, Task, TaskStatus, TaskPriority,
    HierarchicalPlanner, PlannerGrade, PlannerSpec,
)
from control.trajectory import (
    VelocityProfiler, VelocityProfile,
)
from control.agv import (
    AGVMotionController, AGVPose,
)
from control.trajectory_planning import (
    TrajectoryPlanner, TrajectoryPoint, Trajectory,
    Waypoint, PurePursuitTracker, StanleyTracker,
    PIDTrajectoryTracker, RRTStarPlanner, MinimumSnapTrajectory,
)


class TestVelocityProfiler(unittest.TestCase):
    """测试速度规划器"""

    def setUp(self):
        self.profiler = VelocityProfiler(
            max_v=1.0,
            max_a=0.5,
            max_j=2.0,
            profile_type=VelocityProfile.TRAPEZOIDAL
        )

    def test_trapezoidal_plan_full(self):
        """测试梯形速度规划 - 完整加速-匀速-减速"""
        t_pts, v_pts = self.profiler.plan(distance=10.0, v0=0.0, v1=0.0)
        self.assertEqual(len(t_pts), len(v_pts))
        self.assertGreater(t_pts[-1], 0)
        self.assertLessEqual(v_pts.max(), self.profiler.max_v * 1.01)
        self.assertAlmostEqual(v_pts[0], 0.0, places=1)
        self.assertAlmostEqual(v_pts[-1], 0.0, places=1)

    def test_trapezoidal_plan_no_mid(self):
        """测试梯形速度规划 - 无匀速段（短距离）"""
        t_pts, v_pts = self.profiler.plan(distance=0.1, v0=0.0, v1=0.0)
        self.assertEqual(len(t_pts), len(v_pts))
        # 短距离：只有加速和减速
        self.assertGreaterEqual(v_pts[0], 0)  # starts at v0=0

    def test_trapezoidal_plan_with_initial_velocity(self):
        """测试梯形速度规划 - 有初始速度"""
        t_pts, v_pts = self.profiler.plan(distance=5.0, v0=0.5, v1=0.0)
        self.assertEqual(len(t_pts), len(v_pts))
        self.assertGreater(t_pts[-1], 0)

    def test_trapezoidal_plan_with_final_velocity(self):
        """测试梯形速度规划 - 有末端速度"""
        t_pts, v_pts = self.profiler.plan(distance=5.0, v0=0.0, v1=0.3)
        self.assertEqual(len(t_pts), len(v_pts))
        self.assertAlmostEqual(v_pts[-1], 0.3, places=1)

    def test_trapezoidal_plan_v0_exceeds_max(self):
        """测试初始速度超过最大速度时的限制"""
        t_pts, v_pts = self.profiler.plan(distance=10.0, v0=2.0, v1=0.0)
        self.assertLessEqual(v_pts.max(), self.profiler.max_v * 1.01)

    def test_s_curve_profile(self):
        """测试S曲线速度规划"""
        profiler_s = VelocityProfiler(
            max_v=1.0, max_a=0.5, max_j=2.0,
            profile_type=VelocityProfile.S_CURVE
        )
        t_pts, v_pts = profiler_s.plan(distance=5.0, v0=0.0, v1=0.0)
        self.assertEqual(len(t_pts), len(v_pts))
        self.assertGreater(t_pts[-1], 0)


class TestTrajectoryPoint(unittest.TestCase):
    """测试轨迹点"""

    def test_dist_to(self):
        """测试轨迹点间距离计算"""
        p1 = TrajectoryPoint(x=0.0, y=0.0, theta=0.0, v=0.0, t=0.0)
        p2 = TrajectoryPoint(x=3.0, y=4.0, theta=0.0, v=0.0, t=1.0)
        self.assertAlmostEqual(p1.dist_to(p2), 5.0, places=5)


class TestTrajectory(unittest.TestCase):
    """测试轨迹类"""

    def test_at_time_at_start(self):
        """测试轨迹在起始时间"""
        points = [
            TrajectoryPoint(x=0.0, y=0.0, theta=0.0, v=0.0, t=0.0),
            TrajectoryPoint(x=1.0, y=1.0, theta=0.0, v=1.0, t=1.0),
            TrajectoryPoint(x=2.0, y=2.0, theta=0.0, v=0.0, t=2.0),
        ]
        traj = Trajectory(points=points, total_time=2.0, total_length=2.828)
        result = traj.at_time(0.0)
        self.assertAlmostEqual(result.x, 0.0, places=5)

    def test_at_time_at_end(self):
        """测试轨迹在结束时间"""
        points = [
            TrajectoryPoint(x=0.0, y=0.0, theta=0.0, v=0.0, t=0.0),
            TrajectoryPoint(x=1.0, y=1.0, theta=0.0, v=1.0, t=1.0),
            TrajectoryPoint(x=2.0, y=2.0, theta=0.0, v=0.0, t=2.0),
        ]
        traj = Trajectory(points=points, total_time=2.0, total_length=2.828)
        result = traj.at_time(3.0)
        self.assertAlmostEqual(result.x, 2.0, places=5)

    def test_at_time_in_middle(self):
        """测试轨迹在中间时间 - 线性插值"""
        points = [
            TrajectoryPoint(x=0.0, y=0.0, theta=0.0, v=1.0, t=0.0),
            TrajectoryPoint(x=2.0, y=2.0, theta=0.0, v=1.0, t=1.0),
        ]
        traj = Trajectory(points=points, total_time=1.0, total_length=2.828)
        result = traj.at_time(0.5)
        self.assertAlmostEqual(result.x, 1.0, places=5)
        self.assertAlmostEqual(result.y, 1.0, places=5)

    def test_closest_point(self):
        """测试最近点查找"""
        points = [
            TrajectoryPoint(x=0.0, y=0.0, theta=0.0, v=0.0, t=0.0),
            TrajectoryPoint(x=1.0, y=1.0, theta=0.0, v=0.0, t=1.0),
            TrajectoryPoint(x=2.0, y=0.0, theta=0.0, v=0.0, t=2.0),
        ]
        traj = Trajectory(points=points, total_time=2.0, total_length=2.828)
        closest, idx = traj.closest_point(1.5, 0.5)
        # (1.5, 0.5) 距离 (1,1)=sqrt(0.5) 和 (2,0)=sqrt(0.5) 应该差不多近
        self.assertIn(idx, [1, 2])

    def test_interp_angle_positive(self):
        """测试角度插值 - 正向"""
        diff = Trajectory._interp_angle(0.1, 0.2, 0.5)
        self.assertAlmostEqual(diff, 0.05, places=5)

    def test_interp_angle_wraps_positive(self):
        """测试角度插值 - 跨越π正向"""
        diff = Trajectory._interp_angle(math.pi - 0.1, -math.pi + 0.1, 0.5)
        # 应该正确处理周期性
        self.assertLess(abs(diff), math.pi)


class TestTrajectoryPlanner(unittest.TestCase):
    """测试轨迹规划器"""

    def setUp(self):
        self.planner = TrajectoryPlanner(max_v=1.0, max_a=0.5, max_omega=2.0, dt=0.1)

    def test_plan_line_basic(self):
        """测试直线轨迹规划"""
        start = Waypoint(x=0.0, y=0.0, theta=0.0, v=0.0)
        end = Waypoint(x=1.0, y=1.0, theta=0.0, v=0.0)
        traj = self.planner.plan_line(start, end)
        self.assertGreater(len(traj), 0)
        self.assertAlmostEqual(traj[0].x, 0.0, places=3)
        self.assertAlmostEqual(traj[-1].x, 1.0, places=3)

    def test_plan_line_zero_distance(self):
        """测试零距离轨迹"""
        start = Waypoint(x=0.0, y=0.0, theta=0.0, v=0.0)
        end = Waypoint(x=0.0, y=0.0, theta=0.0, v=0.0)
        traj = self.planner.plan_line(start, end)
        self.assertEqual(len(traj), 1)
        self.assertAlmostEqual(traj[0].x, 0.0)

    def test_plan_arc_basic(self):
        """测试圆弧轨迹规划"""
        start = Waypoint(x=0.0, y=0.0, theta=0.0, v=0.0)
        end = Waypoint(x=1.0, y=0.0, theta=0.0, v=0.0)
        traj = self.planner.plan_arc(start, end, curvature=0.5)
        self.assertGreater(len(traj), 0)

    def test_plan_path_single_waypoint(self):
        """测试单航点轨迹"""
        waypoints = [Waypoint(x=0.0, y=0.0, theta=0.0, v=0.0)]
        traj = self.planner.plan_path(waypoints)
        self.assertEqual(len(traj.points), 0)


class TestPurePursuitTracker(unittest.TestCase):
    """测试Pure Pursuit轨迹跟踪器"""

    def setUp(self):
        self.tracker = PurePursuitTracker(
            lookahead_distance=0.5,
            kp=1.0,
            min_lookahead=0.1,
            max_lookahead=2.0
        )

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.tracker.lookahead_distance, 0.5)
        self.assertEqual(self.tracker.kp, 1.0)

    def test_compute_no_trajectory(self):
        """测试无轨迹时的计算"""
        cmd = self.tracker.compute(0.0, 0.0, 0.0, 1.0, [])
        self.assertAlmostEqual(cmd.v, 0.0, places=5)
        self.assertAlmostEqual(cmd.omega, 0.0, places=5)

    def test_compute_with_trajectory(self):
        """测试有轨迹时的计算"""
        traj = [
            TrajectoryPoint(x=0.0, y=0.0, theta=0.0, v=1.0, t=0.0),
            TrajectoryPoint(x=1.0, y=0.0, theta=0.0, v=1.0, t=1.0),
            TrajectoryPoint(x=2.0, y=0.0, theta=0.0, v=1.0, t=2.0),
        ]
        cmd = self.tracker.compute(0.0, 0.0, 0.0, 1.0, traj)
        self.assertIsNotNone(cmd)
        self.assertIn('v', cmd)
        self.assertIn('omega', cmd)


class TestStanleyTracker(unittest.TestCase):
    """测试Stanley轨迹跟踪器"""

    def setUp(self):
        self.tracker = StanleyTracker(k=1.0, k_soft=1.0)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.tracker.k, 1.0)
        self.assertEqual(self.tracker.k_soft, 1.0)

    def test_compute_no_trajectory(self):
        """测试无轨迹时的计算"""
        cmd = self.tracker.compute(0.0, 0.0, 0.0, 1.0, [])
        self.assertAlmostEqual(cmd.v, 0.0, places=5)
        self.assertAlmostEqual(cmd.omega, 0.0, places=5)

    def test_compute_with_trajectory(self):
        """测试有轨迹时的计算"""
        traj = [
            TrajectoryPoint(x=0.0, y=0.0, theta=0.0, v=1.0, t=0.0),
            TrajectoryPoint(x=1.0, y=0.0, theta=0.0, v=1.0, t=1.0),
            TrajectoryPoint(x=2.0, y=0.0, theta=0.0, v=1.0, t=2.0),
        ]
        cmd = self.tracker.compute(0.0, 0.0, 0.0, 1.0, traj)
        self.assertIsNotNone(cmd)


class TestPIDTrajectoryTracker(unittest.TestCase):
    """测试PID轨迹跟踪器"""

    def setUp(self):
        self.tracker = PIDTrajectoryTracker(
            kp_v=1.0, ki_v=0.1, kd_v=0.2,
            kp_omega=1.5, ki_omega=0.0, kd_omega=0.3
        )

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.tracker.kp_v, 1.0)
        self.assertEqual(self.tracker.ki_v, 0.1)
        self.assertEqual(self.tracker.kd_v, 0.2)

    def test_compute_no_trajectory(self):
        """测试无轨迹时的计算"""
        cmd = self.tracker.compute(0.0, 0.0, 0.0, [])
        self.assertAlmostEqual(cmd.v, 0.0, places=3)
        self.assertAlmostEqual(cmd.omega, 0.0, places=3)

    def test_compute_with_trajectory(self):
        """测试有轨迹时的计算"""
        traj = [
            TrajectoryPoint(x=0.0, y=0.0, theta=0.0, v=1.0, t=0.0),
            TrajectoryPoint(x=1.0, y=0.0, theta=0.0, v=1.0, t=1.0),
        ]
        cmd = self.tracker.compute(0.5, 0.0, 0.0, traj)
        self.assertIsNotNone(cmd)
        self.assertIn('v', cmd)
        self.assertIn('omega', cmd)


class TestRRTStarPlanner(unittest.TestCase):
    """测试RRT*路径规划器"""

    def setUp(self):
        self.planner = RRTStarPlanner(
            bounds=((-5.0, 5.0), (-5.0, 5.0)),
            max_nodes=500,
            goal_sample_rate=0.1,
            robot_radius=0.2
        )

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.planner.max_nodes, 500)
        self.assertEqual(self.planner.robot_radius, 0.2)

    def test_plan_no_obstacles(self):
        """测试无障碍物时的规划"""
        path = self.planner.plan((0.0, 0.0), (2.0, 2.0), obstacles=[])
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 0)

    def test_plan_with_obstacles(self):
        """测试有障碍物时的规划"""
        obstacles = [
            ((1.0, 1.0, 0.3), 'circle'),
            ((-1.0, 2.0, 0.4), 'circle'),
        ]
        path = self.planner.plan((0.0, 0.0), (3.0, 3.0), obstacles=obstacles)
        # 可能找到路径也可能没找到，取决于随机性
        if path:
            self.assertGreater(len(path), 0)

    def test_plan_unreachable(self):
        """测试无法到达的情况 - 起点在障碍物中"""
        obstacles = [((0.0, 0.0, 1.0), 'circle')]
        path = self.planner.plan((0.0, 0.0), (3.0, 3.0), obstacles=obstacles)
        # 起点在障碍物中，应该返回None
        self.assertIsNone(path)


class TestMinimumSnapTrajectory(unittest.TestCase):
    """测试最小Snap平滑轨迹生成"""

    def test_initialization(self):
        """测试初始化"""
        traj = MinimumSnapTrajectory(
            waypoints=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]),
            velocities=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
            max_velocity=1.0,
            max_acceleration=0.5
        )
        self.assertEqual(traj.waypoints.shape, (3, 2))

    def test_compute_valid_waypoints(self):
        """测试有效航点的轨迹计算"""
        waypoints = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
        ])
        velocities = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ])
        traj = MinimumSnapTrajectory(
            waypoints=waypoints,
            velocities=velocities,
            max_velocity=1.0,
            max_acceleration=0.5
        )
        result = traj.compute()
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)


if __name__ == '__main__':
    unittest.main()
