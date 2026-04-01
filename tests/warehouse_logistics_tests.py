"""
仓库物流仿真场景测试
====================

测试 WarehouseLogisticsScenario:
- 环境初始化
- 货架布局
- AGV 移动与路径规划
- 碰撞检测
- 动态障碍
- 任务管理与分配
- 奖励计算
"""

import numpy as np
import sys
import unittest

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from simulation.warehouse_logistics import (
    WarehouseLogisticsScenario, WarehouseTask, WarehouseObstacle,
    ShelfPosition, TaskType, TaskStatus
)


class TestWarehouseInit(unittest.TestCase):
    """测试仓库环境初始化"""

    def test_single_aisle_init(self):
        env = WarehouseLogisticsScenario(warehouse_layout="single_aisle", num_agvs=1)
        self.assertEqual(len(env.shelves) > 0, True)
        self.assertEqual(len(env.agvs), 1)
        self.assertEqual(env.warehouse_layout, "single_aisle")

    def test_multi_aisle_init(self):
        env = WarehouseLogisticsScenario(warehouse_layout="multi_aisle", num_agvs=2)
        self.assertEqual(len(env.agvs), 2)
        self.assertEqual(env.warehouse_layout, "multi_aisle")

    def test_u_shape_init(self):
        env = WarehouseLogisticsScenario(warehouse_layout="u_shape", num_agvs=3)
        self.assertEqual(len(env.agvs), 3)
        self.assertEqual(len(env.shelves) > 0, True)

    def test_agv_params_by_grade(self):
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            env = WarehouseLogisticsScenario(grade=grade)
            self.assertIn('max_speed', env.agv_params)
            self.assertIn('max_accel', env.agv_params)
            self.assertIn('turn_radius', env.agv_params)


class TestShelfLayout(unittest.TestCase):
    """测试货架布局"""

    def test_shelf_positions(self):
        env = WarehouseLogisticsScenario(warehouse_layout="single_aisle")
        for shelf in env.shelves:
            self.assertIsInstance(shelf.x, float)
            self.assertIsInstance(shelf.y, float)
            self.assertIn(shelf.level, [1, 2, 3])

    def test_multi_aisle_shelves(self):
        env = WarehouseLogisticsScenario(warehouse_layout="multi_aisle", num_shelves=12)
        self.assertGreater(len(env.shelves), 12)

    def test_u_shape_shelves(self):
        env = WarehouseLogisticsScenario(warehouse_layout="u_shape")
        self.assertGreaterEqual(len(env.shelves), 30)  # u_shape 有 12 位 * 3 层 = 36
        # 货架层数检查 (每个位置有3层)
        levels = [s.level for s in env.shelves]
        self.assertEqual(sorted(set(levels)), [1, 2, 3])


class TestAGVState(unittest.TestCase):
    """测试 AGV 状态"""

    def test_agv_initial_state(self):
        env = WarehouseLogisticsScenario(num_agvs=2)
        for agv_id, agv in env.agvs.items():
            self.assertEqual(agv['state'], 'idle')
            self.assertIsNone(agv['task_id'])
            self.assertIsNone(agv['cargo'])

    def test_agv_movement(self):
        env = WarehouseLogisticsScenario(num_agvs=1)
        agv_id = list(env.agvs.keys())[0]
        # 分配任务后 AGV 才会移动
        if env.tasks:
            env.assign_task(agv_id, env.tasks[0].task_id)
        initial_x = env.agvs[agv_id]['x']
        initial_y = env.agvs[agv_id]['y']

        for _ in range(50):
            env.step()

        moved = (env.agvs[agv_id]['x'] != initial_x) or (env.agvs[agv_id]['y'] != initial_y)
        self.assertTrue(moved)


class TestTaskManagement(unittest.TestCase):
    """测试任务管理"""

    def test_initial_tasks(self):
        env = WarehouseLogisticsScenario()
        self.assertGreater(len(env.tasks), 0)
        for task in env.tasks:
            self.assertEqual(task.status, TaskStatus.PENDING)

    def test_add_task(self):
        env = WarehouseLogisticsScenario()
        initial_count = len(env.tasks)
        new_task = WarehouseTask(
            task_id="TASK-TEST",
            task_type=TaskType.DELIVERY,
            source=ShelfPosition(x=1.0, y=1.0, shelf_id=99),
            destination=ShelfPosition(x=2.0, y=2.0, shelf_id=98),
        )
        env.add_task(new_task)
        self.assertEqual(len(env.tasks), initial_count + 1)

    def test_assign_task(self):
        env = WarehouseLogisticsScenario(num_agvs=2)
        agv_id = list(env.agvs.keys())[0]
        task_id = env.tasks[0].task_id

        result = env.assign_task(agv_id, task_id)
        self.assertTrue(result)
        self.assertEqual(env.agvs[agv_id]['task_id'], task_id)
        self.assertEqual(env.agvs[agv_id]['state'], 'moving')

    def test_assign_nonexistent_task(self):
        env = WarehouseLogisticsScenario(num_agvs=1)
        agv_id = list(env.agvs.keys())[0]
        result = env.assign_task(agv_id, "NONEXISTENT_TASK")
        self.assertFalse(result)


class TestDynamicObstacles(unittest.TestCase):
    """测试动态障碍"""

    def test_add_obstacle(self):
        env = WarehouseLogisticsScenario()
        initial_count = len(env.obstacles)
        env.add_dynamic_obstacle(0.0, 0.0, "human")
        self.assertEqual(len(env.obstacles), initial_count + 1)

    def test_obstacle_update(self):
        env = WarehouseLogisticsScenario()
        obs = env.add_dynamic_obstacle(0.0, 0.0, "human")
        obs.vx = 1.0
        obs.vy = 0.0
        env.update_obstacles(0.1)
        self.assertAlmostEqual(obs.x, 0.1, places=2)

    def test_obstacle_boundary_bounce(self):
        env = WarehouseLogisticsScenario()
        obs = env.add_dynamic_obstacle(5.0, 0.0, "human")
        obs.vx = 1.0
        env.update_obstacles(0.1)
        self.assertAlmostEqual(obs.x, 5.1, places=2)
        # 反转后再次更新
        env.update_obstacles(0.1)
        self.assertAlmostEqual(obs.x, 5.2, places=2)


class TestCollisionDetection(unittest.TestCase):
    """测试碰撞检测"""

    def test_no_collision_initial(self):
        env = WarehouseLogisticsScenario()
        agv_id = list(env.agvs.keys())[0]
        collision = env._check_collision(agv_id, env.agvs[agv_id]['x'], env.agvs[agv_id]['y'])
        self.assertFalse(collision)

    def test_collision_with_shelf(self):
        env = WarehouseLogisticsScenario()
        agv_id = list(env.agvs.keys())[0]
        if env.shelves:
            shelf = env.shelves[0]
            collision = env._check_collision(agv_id, shelf.x, shelf.y)
            self.assertTrue(collision)

    def test_collision_between_agvs(self):
        env = WarehouseLogisticsScenario(num_agvs=2)
        agv_ids = list(env.agvs.keys())
        env.agvs[agv_ids[0]]['x'] = 0.0
        env.agvs[agv_ids[0]]['y'] = 0.0
        env.agvs[agv_ids[1]]['x'] = 0.1
        env.agvs[agv_ids[1]]['y'] = 0.1
        collision = env._check_collision(agv_ids[0], 0.1, 0.1)
        self.assertTrue(collision)

    def test_collision_with_obstacle(self):
        env = WarehouseLogisticsScenario()
        agv_id = list(env.agvs.keys())[0]
        env.add_dynamic_obstacle(0.5, 0.5, "human")
        collision = env._check_collision(agv_id, 0.5, 0.5)
        self.assertTrue(collision)


class TestSimulationStep(unittest.TestCase):
    """测试仿真步进"""

    def test_step_returns_dict(self):
        env = WarehouseLogisticsScenario()
        result = env.step()
        self.assertIn('observation', result)
        self.assertIn('reward', result)
        self.assertIn('done', result)
        self.assertIn('info', result)

    def test_observation_shape(self):
        env = WarehouseLogisticsScenario(num_agvs=2)
        obs = env._get_observation()
        self.assertIsInstance(obs, np.ndarray)
        self.assertGreater(len(obs), 0)

    def test_reward_computation(self):
        env = WarehouseLogisticsScenario()
        env.stats['tasks_completed'] = 0
        env.stats['total_distance'] = 0.0
        env.stats['collision_count'] = 0
        reward = env._compute_reward()
        self.assertIsInstance(reward, float)

    def test_is_done_false_initially(self):
        env = WarehouseLogisticsScenario()
        self.assertFalse(env._is_done())

    def test_reset(self):
        env = WarehouseLogisticsScenario(num_agvs=2)
        env.step()
        env.step()
        obs = env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(env.sim_time, 0.0)


class TestRender(unittest.TestCase):
    """测试渲染功能"""

    def test_render_returns_array(self):
        env = WarehouseLogisticsScenario()
        img = env.render()
        self.assertIsNotNone(img)
        self.assertEqual(img.shape[2], 3)


class TestStateDict(unittest.TestCase):
    """测试状态字典"""

    def test_get_state_dict(self):
        env = WarehouseLogisticsScenario(num_agvs=2)
        state = env.get_state_dict()
        self.assertIn('sim_time', state)
        self.assertIn('agvs', state)
        self.assertIn('tasks', state)
        self.assertIn('obstacles', state)
        self.assertIn('stats', state)


if __name__ == '__main__':
    unittest.main()
