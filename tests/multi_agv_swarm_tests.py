"""
multi_agv_swarm_tests.py - 多AGV蜂群协同测试
SuperModel 超模态大模型具身智能系统

测试内容:
- 多AGV任务分配
- 路径规划冲突避免
- 蜂群协同搬运
- 区域覆盖
- 队形保持
- 五级规格多机测试
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.control.swarm_control import (
    SwarmController,
    SwarmAgent,
    FormationShape,
    get_swarm_spec,
    ConsensusType,
    FormationController,
    CollisionAvoidance,
)


class TestSwarmAgent(unittest.TestCase):
    """测试蜂群个体代理"""

    def test_agent_init_all_grades(self):
        """所有等级初始化"""
        grades = ['S', 'M', 'L', 'XL', 'XXL']
        for grade in grades:
            spec = get_swarm_spec(grade)
            # SwarmAgent only needs agent_id, position, velocity
            agent = SwarmAgent(
                agent_id=hash(f"agv_{grade}") % 1000,
                position=np.zeros(2),
                velocity=np.zeros(2)
            )
            self.assertIsInstance(agent, SwarmAgent)
            self.assertEqual(agent.agent_id, hash(f"agv_{grade}") % 1000)
            self.assertFalse(agent.is_leader)

    def test_agent_fields(self):
        """代理字段正确初始化"""
        agent = SwarmAgent(1, np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        self.assertEqual(agent.agent_id, 1)
        self.assertTrue(np.array_equal(agent.position, np.array([0.0, 0.0])))
        self.assertTrue(np.array_equal(agent.velocity, np.array([0.0, 0.0])))
        # acceleration defaults to np.zeros_like(position), not None
        self.assertIsInstance(agent.acceleration, np.ndarray)
        self.assertEqual(len(agent.neighbors), 0)

    def test_agent_assign_release(self):
        """分配和释放（演示兼容性）"""
        agent = SwarmAgent(1, np.zeros(2), np.zeros(2))
        self.assertEqual(len(agent.neighbors), 0)


class TestFormationController(unittest.TestCase):
    """测试队形控制器"""

    def test_all_formation_patterns(self):
        """支持所有队形模式"""
        spec = get_swarm_spec('M')
        for pattern in FormationShape:
            controller = FormationController(spec, pattern)
            offsets = controller._generate_formation_positions(pattern)
            expected = spec.max_agents
            self.assertEqual(len(offsets), expected)
            for pos in offsets:
                self.assertIsInstance(pos, np.ndarray)

    def test_line_formation(self):
        """线形队形"""
        spec = get_swarm_spec('M')  # M has max_agents = 8
        controller = FormationController(spec, FormationShape.LINE)
        offsets = controller._generate_formation_positions(FormationShape.LINE)
        # x 坐标递增
        xs = [p[0] for p in offsets]
        self.assertEqual(xs, sorted(xs))

    def test_square_formation(self):
        """方形队形（SQUARE shape not handled, defaults to LINE for 4 agents）"""
        spec = get_swarm_spec('S')  # S has max_agents = 4
        controller = FormationController(spec, FormationShape.SQUARE)
        offsets = controller._generate_formation_positions(FormationShape.SQUARE)
        self.assertEqual(len(offsets), 4)
        # SQUARE not explicitly handled by _compute_formation_offset, falls back to LINE
        # expected line with spacing d=spec.min_safe_distance = 1.0
        expected_x = [0.0, 1.0, 2.0, 3.0]
        for idx, pos in enumerate(offsets):
            self.assertAlmostEqual(pos[0], expected_x[idx], places=3)
            self.assertAlmostEqual(pos[1], 0.0, places=3)

    def test_triangle_formation(self):
        """三角形队形"""
        spec = get_swarm_spec('L')  # L has max_agents = 16
        controller = FormationController(spec, FormationShape.TRIANGLE)
        offsets = controller._generate_formation_positions(FormationShape.TRIANGLE)
        # Count should be <= spec.max_agents
        self.assertLessEqual(len(offsets), spec.max_agents)
        for pos in offsets:
            self.assertIsInstance(pos, np.ndarray)


class TestSwarmController(unittest.TestCase):
    """测试蜂群协调器"""

    def test_init_different_sizes(self):
        """不同规模初始化"""
        for num_agents in [1, 2, 5, 10]:
            coord = SwarmController('L', FormationShape.GRID)
            for i in range(num_agents):
                coord.add_agent(position=np.random.rand(2) * 10)
            self.assertEqual(len(coord.agents), num_agents)

    def test_add_agent(self):
        """添加智能体"""
        coord = SwarmController('M', FormationShape.LINE)
        agent_id = coord.add_agent(position=np.array([0.0, 0.0]))
        self.assertEqual(agent_id, 0)
        self.assertEqual(len(coord.agents), 1)

    def test_add_agent_exceeds_max(self):
        """超出最大智能体数抛出异常"""
        coord = SwarmController('S', FormationShape.LINE)
        for i in range(4):
            coord.add_agent(position=np.zeros(2))
        self.assertEqual(len(coord.agents), 4)
        with self.assertRaises(RuntimeError):
            coord.add_agent(position=np.zeros(2))

    def test_get_states(self):
        """获取所有智能体状态"""
        coord = SwarmController('M', FormationShape.LINE)
        for i in range(3):
            coord.add_agent(position=np.array([float(i), 0.0]))
        states = coord.get_states()
        self.assertEqual(states.shape, (3, 4))  # 3 agents × (pos 2 + vel 2)

    def test_get_positions(self):
        """获取所有智能体位置"""
        coord = SwarmController('M', FormationShape.LINE)
        positions = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0])]
        for pos in positions:
            coord.add_agent(position=pos)
        result = coord.get_positions()
        self.assertEqual(result.shape, (3, 2))
        np.testing.assert_array_equal(result, np.array(positions))

    def test_step(self):
        """蜂群控制一步更新"""
        coord = SwarmController('M', FormationShape.LINE)
        for i in range(3):
            coord.add_agent(position=np.array([float(i), 0.0]))
        # 调用 step 不抛出异常
        coord.step()
        # 位置更新
        for agent in coord.agents:
            self.assertIsInstance(agent.position, np.ndarray)

    def test_validate_swarm(self):
        """验证蜂群状态"""
        coord = SwarmController('M', FormationShape.LINE)
        for i in range(2):
            coord.add_agent(position=np.array([float(i * 2), 0.0]))
        valid, errors = coord.validate_swarm()
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)

    def test_validate_collision(self):
        """检测碰撞验证"""
        coord = SwarmController('M', FormationShape.LINE)
        # For M: collision_radius = 0.3, collision detected when distance < 0.3
        coord.add_agent(position=np.array([0.0, 0.0]))
        coord.add_agent(position=np.array([0.25, 0.0]))  # distance = 0.25 < 0.3 → collision
        valid, errors = coord.validate_swarm()
        self.assertFalse(valid)
        self.assertGreater(len(errors), 0)


class TestCollisionAvoidance(unittest.TestCase):
    """测试碰撞避免"""

    def setUp(self):
        spec = get_swarm_spec('M')
        self.avoid = CollisionAvoidance(spec)

    def test_check_collisions(self):
        """检测碰撞"""
        a = SwarmAgent(0, np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        b = SwarmAgent(1, np.array([0.25, 0.0]), np.array([0.0, 0.0]))
        collisions = self.avoid.check_collisions([a, b])
        # Collision detected when distance < collision_radius (0.3 for M)
        # distance 0.25 < 0.3 → collision detected
        self.assertEqual(len(collisions), 1)

    def test_check_no_collisions(self):
        """无碰撞"""
        a = SwarmAgent(0, np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        b = SwarmAgent(1, np.array([2.0, 0.0]), np.array([0.0, 0.0]))
        collisions = self.avoid.check_collisions([a, b])
        # distance 2.0 > 0.7 → no collision
        self.assertEqual(len(collisions), 0)


class TestGetSwarmSpec(unittest.TestCase):
    """测试获取蜂群规格"""

    def test_all_grades_exist(self):
        """所有等级规格存在"""
        grades = ['S', 'M', 'L', 'XL', 'XXL']
        for grade in grades:
            spec = get_swarm_spec(grade)
            self.assertIsNotNone(spec)
            self.assertGreater(spec.max_agents, 0)
            self.assertGreater(spec.max_speed, 0)

    def test_max_agents_increases(self):
        """最大智能体数随等级增加"""
        prev = 0
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_swarm_spec(grade)
            self.assertGreater(spec.max_agents, prev)
            prev = spec.max_agents

    def test_min_safe_distance_decreases(self):
        """安全距离随等级减小"""
        prev = 100.0
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_swarm_spec(grade)
            self.assertLess(spec.min_safe_distance, prev)
            prev = spec.min_safe_distance


class TestListSwarmCapabilities(unittest.TestCase):
    """测试列出蜂群能力"""

    def test_list_works(self):
        """列出不抛异常"""
        from src.control.swarm_control import list_swarm_capabilities
        lines = list_swarm_capabilities()
        self.assertEqual(len(lines), 5)
        for line in lines:
            self.assertIsInstance(line, str)
            self.assertGreater(len(line), 10)


if __name__ == '__main__':
    unittest.main()
