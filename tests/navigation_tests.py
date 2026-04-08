"""
AGV导航控制模块测试
===================

测试导航模块:
- OccupancyGrid 栅格地图
- Dijkstra / A* 路径规划
- NavigationController 导航控制
- 全局路径 + 本地跟踪集成
"""

import unittest
import numpy as np
import sys
import math

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from control.navigation import (
    NavigationController, OccupancyGrid, Path, Waypoint,
    PlannerType, NavigationState,
    DijkstraPlanner, AStarPlanner,
    create_navigation_grid
)


class TestOccupancyGrid(unittest.TestCase):
    """测试占用栅格地图"""

    def setUp(self):
        self.grid = OccupancyGrid(width=10.0, height=10.0, resolution=0.1)

    def test_creation(self):
        """测试创建"""
        self.assertEqual(self.grid.width, 10.0)
        self.assertEqual(self.grid.height, 10.0)
        self.assertEqual(self.grid.resolution, 0.1)
        self.assertEqual(self.grid.cols, 100)
        self.assertEqual(self.grid.rows, 100)

    def test_world_to_grid(self):
        """测试坐标转换"""
        gx, gy = self.grid.world_to_grid(5.0, 5.0)
        self.assertEqual(gx, 50)
        self.assertEqual(gy, 50)

        gx, gy = self.grid.world_to_grid(0.0, 0.0)
        self.assertEqual(gx, 0)
        self.assertEqual(gy, 0)

        gx, gy = self.grid.world_to_grid(10.0, 10.0)
        self.assertEqual(gx, 99)
        self.assertEqual(gy, 99)

    def test_grid_to_world(self):
        """测试栅格到世界坐标"""
        wx, wy = self.grid.grid_to_world(50, 50)
        self.assertAlmostEqual(wx, 5.05, places=1)
        self.assertAlmostEqual(wy, 5.05, places=1)

    def test_obstacle_setting(self):
        """测试障碍物设置"""
        self.grid.set_obstacle(5.0, 5.0, radius=0.1)
        self.assertFalse(self.grid.is_free(5.0, 5.0))

    def test_is_free(self):
        """测试空闲检测"""
        self.assertTrue(self.grid.is_free(1.0, 1.0))
        self.grid.set_obstacle(1.0, 1.0, radius=0.0)
        self.assertFalse(self.grid.is_free(1.0, 1.0))

    def test_clipping(self):
        """测试边界裁剪"""
        gx, gy = self.grid.world_to_grid(-1.0, -1.0)
        self.assertEqual(gx, 0)
        self.assertEqual(gy, 0)

        gx, gy = self.grid.world_to_grid(11.0, 11.0)
        self.assertEqual(gx, 99)
        self.assertEqual(gy, 99)


class TestDijkstraPlanner(unittest.TestCase):
    """测试 Dijkstra 规划器"""

    def setUp(self):
        self.grid = OccupancyGrid(width=5.0, height=5.0, resolution=0.5)

    def test_simple_path(self):
        """测试简单路径"""
        planner = DijkstraPlanner(self.grid)
        path = planner.plan((0.25, 0.25), (4.75, 4.75))
        self.assertIsNotNone(path)
        self.assertGreater(len(path.waypoints), 0)
        self.assertEqual(path.planner, PlannerType.DIJKSTRA)

    def test_obstacle_blocking(self):
        """测试障碍物阻挡"""
        self.grid.set_obstacle(2.0, 2.0, radius=1.0)
        planner = DijkstraPlanner(self.grid)
        path = planner.plan((0.25, 0.25), (4.75, 4.75))
        # 狭窄通道应该还能找到路径
        self.assertIsNotNone(path)

    def test_no_path(self):
        """测试无路径情况"""
        self.grid.set_obstacle(2.5, 0.0, radius=2.4)
        self.grid.set_obstacle(2.5, 5.0, radius=2.4)
        planner = DijkstraPlanner(self.grid)
        path = planner.plan((0.25, 2.5), (4.75, 2.5))
        self.assertIsNone(path)


class TestAStarPlanner(unittest.TestCase):
    """测试 A* 规划器"""

    def setUp(self):
        self.grid = OccupancyGrid(width=5.0, height=5.0, resolution=0.5)

    def test_simple_path(self):
        """测试简单路径"""
        planner = AStarPlanner(self.grid)
        path = planner.plan((0.25, 0.25), (4.75, 4.75))
        self.assertIsNotNone(path)
        self.assertGreater(len(path.waypoints), 0)
        self.assertEqual(path.planner, PlannerType.A_STAR)

    def test_path_length(self):
        """测试路径长度"""
        planner = AStarPlanner(self.grid)
        path = planner.plan((0.25, 0.25), (4.75, 4.75))
        self.assertIsNotNone(path)
        self.assertGreater(path.length, 0)

    def test_waypoint_positions(self):
        """测试航点位置"""
        planner = AStarPlanner(self.grid)
        path = planner.plan((0.25, 0.25), (4.75, 4.75))
        self.assertIsNotNone(path)
        positions = path.get_position_array()
        self.assertEqual(positions.shape[1], 2)
        self.assertGreater(len(positions), 0)

    def test_obstacle_avoidance(self):
        """测试避障"""
        self.grid.set_obstacle(2.5, 2.5, radius=0.5)
        planner = AStarPlanner(self.grid)
        path = planner.plan((0.25, 0.25), (4.75, 4.75))
        self.assertIsNotNone(path)


class TestNavigationController(unittest.TestCase):
    """测试导航控制器"""

    def setUp(self):
        self.grid = create_navigation_grid(
            width=10.0, height=10.0, resolution=0.2
        )
        self.nav = NavigationController(
            grid=self.grid,
            planner_type=PlannerType.A_STAR,
            max_speed=1.0,
            goal_tolerance=0.3,
        )

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.nav.state, NavigationState.IDLE)
        self.assertIsNone(self.nav.current_path)
        self.assertEqual(self.nav.target_idx, 0)

    def test_reset(self):
        """测试重置"""
        self.nav.state = NavigationState.NAVIGATING
        self.nav.reset()
        self.assertEqual(self.nav.state, NavigationState.IDLE)

    def test_plan_to_goal(self):
        """测试路径规划"""
        start = np.array([1.0, 1.0, 0.0])
        goal = np.array([8.0, 8.0, 0.0])
        result = self.nav.plan_to_goal(start, goal)
        self.assertTrue(result)
        self.assertEqual(self.nav.state, NavigationState.NAVIGATING)
        self.assertIsNotNone(self.nav.current_path)

    def test_update_navigating(self):
        """测试导航更新"""
        start = np.array([1.0, 1.0, 0.0])
        goal = np.array([8.0, 8.0, 0.0])
        self.nav.plan_to_goal(start, goal)

        # 模拟更新
        vel = self.nav.update(np.array([1.0, 1.0, 0.0]), dt=0.1)
        self.assertEqual(len(vel), 3)
        self.assertTrue(vel[0] >= 0)

    def test_update_idle(self):
        """测试空闲状态"""
        vel = self.nav.update(np.array([1.0, 1.0, 0.0]), dt=0.1)
        np.testing.assert_array_almost_equal(vel, [0.0, 0.0, 0.0])

    def test_emergency_stop(self):
        """测试急停"""
        self.nav.state = NavigationState.NAVIGATING
        self.nav.emergency_stop()
        self.assertEqual(self.nav.state, NavigationState.ESTOP)

    def test_progress(self):
        """测试进度获取"""
        progress = self.nav.get_progress()
        self.assertEqual(progress, 0.0)

        self.nav.current_path = Path(
            waypoints=[Waypoint(0, 0), Waypoint(1, 1), Waypoint(2, 2)],
            length=3.0,
            planner=PlannerType.A_STAR,
            cost=3.0
        )
        self.nav.target_idx = 1
        progress = self.nav.get_progress()
        self.assertAlmostEqual(progress, 0.5)

    def test_arrived_state(self):
        """测试到达状态"""
        self.nav.state = NavigationState.NAVIGATING
        self.nav.current_path = Path(
            waypoints=[Waypoint(1.0, 1.0, 0.0, 1.0)],
            length=0.0,
            planner=PlannerType.A_STAR,
            cost=0.0
        )
        self.nav.target_idx = 0

        # 在目标点附近更新
        vel = self.nav.update(np.array([1.0, 1.0, 0.0]), dt=0.1)
        self.assertEqual(self.nav.state, NavigationState.ARRIVED)
        np.testing.assert_array_almost_equal(vel, [0.0, 0.0, 0.0])


class TestWaypoint(unittest.TestCase):
    """测试航点"""

    def test_to_array(self):
        """测试转换为数组"""
        wp = Waypoint(x=1.0, y=2.0, theta=0.5, speed=1.0)
        arr = wp.to_array()
        self.assertEqual(len(arr), 4)
        self.assertAlmostEqual(arr[0], 1.0)
        self.assertAlmostEqual(arr[1], 2.0)


class TestPath(unittest.TestCase):
    """测试路径"""

    def test_get_position_array(self):
        """测试获取位置数组"""
        path = Path(
            waypoints=[
                Waypoint(0.0, 0.0),
                Waypoint(1.0, 1.0),
                Waypoint(2.0, 0.0),
            ],
            length=3.0,
            planner=PlannerType.A_STAR,
            cost=3.0
        )
        positions = path.get_position_array()
        self.assertEqual(positions.shape, (3, 2))

    def test_get_heading_angle(self):
        """测试航向角"""
        path = Path(
            waypoints=[
                Waypoint(0.0, 0.0),
                Waypoint(1.0, 0.0),
            ],
            length=1.0,
            planner=PlannerType.A_STAR,
            cost=1.0
        )
        angle = path.get_heading_angle(0)
        self.assertAlmostEqual(angle, 0.0)


class TestCreateNavigationGrid(unittest.TestCase):
    """测试创建导航栅格"""

    def test_create_empty_grid(self):
        """测试创建空地图"""
        grid = create_navigation_grid(width=10.0, height=10.0, resolution=0.1)
        self.assertIsInstance(grid, OccupancyGrid)
        self.assertTrue(grid.is_free(5.0, 5.0))

    def test_create_with_obstacles(self):
        """测试创建带障碍物地图"""
        obstacles = [(5.0, 5.0, 0.5), (3.0, 3.0, 0.3)]
        grid = create_navigation_grid(
            width=10.0, height=10.0, resolution=0.1,
            obstacles=obstacles
        )
        self.assertFalse(grid.is_free(5.0, 5.0))
        self.assertTrue(grid.is_free(1.0, 1.0))


class TestNavigationIntegration(unittest.TestCase):
    """导航集成测试"""

    def test_full_navigation_cycle(self):
        """测试完整导航周期"""
        # 创建地图
        grid = create_navigation_grid(
            width=10.0, height=10.0, resolution=0.1
        )
        # 添加一些障碍物
        grid.set_obstacle(5.0, 5.0, radius=0.3)

        # 创建导航器
        nav = NavigationController(
            grid=grid,
            planner_type=PlannerType.A_STAR,
            max_speed=0.5,
            goal_tolerance=0.5,
        )

        # 规划路径
        start = np.array([1.0, 1.0, 0.0])
        goal = np.array([8.0, 8.0, 0.0])
        result = nav.plan_to_goal(start, goal)
        self.assertTrue(result)

        # 模拟导航更新
        pose = start.copy()
        for step in range(100):
            vel = nav.update(pose, dt=0.1)
            if nav.state == NavigationState.ARRIVED or nav.state == NavigationState.FAILED:
                break
            # 更新位姿
            pose[0] += vel[0] * 0.1
            pose[1] += vel[1] * 0.1
            pose[2] += vel[2] * 0.1

        # 验证结果
        self.assertIn(nav.state, [NavigationState.ARRIVED, NavigationState.NAVIGATING])

    def test_multi_waypoint_navigation(self):
        """测试多航点导航"""
        grid = create_navigation_grid(width=20.0, height=20.0, resolution=0.2)
        nav = NavigationController(grid=grid, max_speed=1.0, goal_tolerance=0.5)

        path = Path(
            waypoints=[
                Waypoint(2.0, 2.0, speed=0.5),
                Waypoint(5.0, 5.0, speed=0.5),
                Waypoint(8.0, 8.0, speed=0.5),
                Waypoint(15.0, 15.0, speed=0.3),
            ],
            length=25.0,
            planner=PlannerType.A_STAR,
            cost=25.0
        )
        nav.set_global_path(path)

        self.assertEqual(nav.state, NavigationState.NAVIGATING)
        self.assertEqual(nav.target_idx, 0)

        # 更新到第一个航点
        nav.update(np.array([2.0, 2.0, 0.0]), dt=0.1)
        self.assertEqual(nav.target_idx, 1)


if __name__ == "__main__":
    unittest.main()
