"""
多智能体协调控制测试
====================

测试 MultiAgentCoordinator 模块:
- 智能体注册与管理
- 编队形成与重构
- 碰撞检测与避障
- 任务分配
"""

import numpy as np
import sys
import unittest

sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from control.multi_agent import (
    MultiAgentCoordinator, FormationType, CoordinationState,
    AgentState, FormationSlot, CollisionRisk,
    get_coordination_spec
)


class TestMultiAgentCoordinator(unittest.TestCase):
    """测试多智能体协调控制器"""
    
    def setUp(self):
        self.coord = MultiAgentCoordinator(
            communication_range=10.0,
            safety_distance=0.5,
            max_agents=20
        )
    
    def test_coordinator_creation(self):
        self.assertEqual(len(self.coord), 0)
        self.assertEqual(self.coord.communication_range, 10.0)
        self.assertEqual(self.coord.safety_distance, 0.5)
    
    def test_agent_registration(self):
        result = self.coord.register_agent("agent_0", np.array([0.0, 0.0]))
        self.assertTrue(result)
        self.assertEqual(len(self.coord), 1)
        self.assertIn("agent_0", self.coord.agents)
        
        # 重复注册
        result2 = self.coord.register_agent("agent_0", np.array([1.0, 1.0]))
        self.assertFalse(result2)
    
    def test_agent_registration_with_leader(self):
        self.coord.register_agent("leader", np.array([0.0, 0.0]))
        self.coord.register_agent("follower", np.array([1.0, 0.0]), leader_id="leader")
        
        self.assertEqual(self.coord.agents["follower"].leader_id, "leader")
    
    def test_agent_unregistration(self):
        self.coord.register_agent("agent_0", np.array([0.0, 0.0]))
        self.coord.unregister_agent("agent_0")
        self.assertEqual(len(self.coord), 0)
        self.assertNotIn("agent_0", self.coord.agents)
    
    def test_max_agents_limit(self):
        small_coord = MultiAgentCoordinator(max_agents=2)
        small_coord.register_agent("a1", np.array([0.0, 0.0]))
        small_coord.register_agent("a2", np.array([1.0, 0.0]))
        result = small_coord.register_agent("a3", np.array([2.0, 0.0]))
        self.assertFalse(result)
    
    def test_formation_creation_line(self):
        # 注册多个智能体
        for i in range(4):
            self.coord.register_agent(f"agent_{i}", np.array([float(i), 0.0]))
        
        task = self.coord.create_formation(
            "line_1",
            FormationType.LINE,
            target_position=np.array([0.0, 0.0]),
            target_heading=0.0
        )
        
        self.assertEqual(len(self.coord.formations), 1)
        self.assertEqual(len(task.slots), 4)
        self.assertEqual(task.formation_type, FormationType.LINE)
    
    def test_formation_creation_triangle(self):
        for i in range(6):
            self.coord.register_agent(f"agent_{i}", np.array([float(i), 0.0]))
        
        task = self.coord.create_formation(
            "tri_1",
            FormationType.TRIANGLE,
            target_position=np.array([0.0, 0.0])
        )
        
        self.assertEqual(len(task.slots), 6)
    
    def test_formation_creation_circle(self):
        for i in range(8):
            self.coord.register_agent(f"agent_{i}", np.array([float(i), 0.0]))
        
        task = self.coord.create_formation(
            "circle_1",
            FormationType.CIRCLE,
            target_position=np.array([0.0, 0.0])
        )
        
        self.assertEqual(len(task.slots), 8)
    
    def test_formation_creation_v_shape(self):
        for i in range(5):
            self.coord.register_agent(f"agent_{i}", np.array([float(i), 0.0]))
        
        task = self.coord.create_formation(
            "v_1",
            FormationType.V_SHAPE,
            target_position=np.array([0.0, 0.0])
        )
        
        self.assertEqual(len(task.slots), 5)
    
    def test_formation_duplicate_id(self):
        self.coord.register_agent("a1", np.array([0.0, 0.0]))
        self.coord.create_formation("f1", FormationType.LINE, np.array([0.0, 0.0]))
        
        with self.assertRaises(ValueError):
            self.coord.create_formation("f1", FormationType.LINE, np.array([1.0, 1.0]))
    
    def test_formation_target_computation(self):
        # 注册智能体
        self.coord.register_agent("leader", np.array([0.0, 0.0]))
        self.coord.register_agent("f1", np.array([2.0, 0.0]))
        self.coord.register_agent("f2", np.array([3.0, 0.0]))
        
        # 创建编队
        self.coord.create_formation(
            "line_1",
            FormationType.LINE,
            target_position=np.array([0.0, 0.0]),
            target_heading=0.0
        )
        
        # 计算跟随者目标位置
        target = self.coord.compute_formation_target(
            "f1", np.array([0.0, 0.0]), 0.0
        )
        
        # f1 在 slot 1, 应该距离 leader 1m
        self.assertAlmostEqual(target[0], 1.0, places=1)
    
    def test_formation_target_rotation(self):
        self.coord.register_agent("leader", np.array([0.0, 0.0]))
        self.coord.register_agent("f1", np.array([1.0, 0.0]))
        
        self.coord.create_formation(
            "line_1",
            FormationType.LINE,
            target_position=np.array([0.0, 0.0])
        )
        
        # 旋转90度
        target = self.coord.compute_formation_target(
            "f1", np.array([0.0, 0.0]), np.pi / 2
        )
        
        # 应该朝 y 方向
        self.assertAlmostEqual(target[1], 1.0, places=1)
    
    def test_agent_leave_formation(self):
        self.coord.register_agent("a1", np.array([0.0, 0.0]))
        self.coord.register_agent("a2", np.array([1.0, 0.0]))
        
        self.coord.create_formation("f1", FormationType.LINE, np.array([0.0, 0.0]))
        
        self.assertTrue(self.coord.agents["a1"].in_formation)
        
        self.coord._leave_formation("a1")
        
        self.assertFalse(self.coord.agents["a1"].in_formation)
        self.assertNotIn("a1", self.coord.agent_formations)
    
    def test_collision_detection_no_collision(self):
        self.coord.register_agent("a1", np.array([0.0, 0.0]))
        self.coord.register_agent("a2", np.array([10.0, 10.0]))
        
        risks = self.coord.detect_collisions()
        self.assertEqual(len(risks), 0)
    
    def test_collision_detection_close_agents(self):
        self.coord.register_agent("a1", np.array([0.0, 0.0]))
        self.coord.register_agent("a2", np.array([0.5, 0.0]))  # 距离0.5m < 2*safety_distance
        # 设置相对速度，使 TTC < 5s
        self.coord.agents["a1"].velocity = np.array([1.0, 0.0])
        self.coord.agents["a2"].velocity = np.array([-1.0, 0.0])
        
        risks = self.coord.detect_collisions()
        
        self.assertEqual(len(risks), 1)
        self.assertIn(risks[0].agent_a, ["a1", "a2"])
        self.assertIn(risks[0].agent_b, ["a1", "a2"])
    
    def test_collision_detection_multiple_agents(self):
        for i in range(5):
            self.coord.register_agent(f"a{i}", np.array([float(i), 0.0]))
        
        # a0 和 a1 靠近 (距离0.3 < 2*safety_distance=1.0)
        self.coord.agents["a0"].position = np.array([0.0, 0.0])
        self.coord.agents["a1"].position = np.array([0.3, 0.0])
        # 设置相对速度使 TTC < 5s
        self.coord.agents["a0"].velocity = np.array([1.0, 0.0])
        self.coord.agents["a1"].velocity = np.array([-1.0, 0.0])
        
        risks = self.coord.detect_collisions()
        
        # 应该检测到至少一个风险
        self.assertGreaterEqual(len(risks), 1)
    
    def test_collision_resolution(self):
        self.coord.register_agent("a1", np.array([0.0, 0.0]))
        self.coord.register_agent("a2", np.array([0.3, 0.0]))
        
        self.coord.agents["a1"].velocity = np.array([1.0, 0.0])
        self.coord.agents["a2"].velocity = np.array([-1.0, 0.0])
        
        self.coord.detect_collisions()
        corrections = self.coord.resolve_collisions()
        
        self.assertIn("a1", corrections)
        self.assertIn("a2", corrections)
    
    def test_collision_risk_severity_levels(self):
        # 临界碰撞
        self.coord.register_agent("a1", np.array([0.0, 0.0]))
        self.coord.register_agent("a2", np.array([0.2, 0.0]))
        
        risks = self.coord.detect_collisions()
        
        if risks:
            self.assertEqual(risks[0].severity, "critical")
    
    def test_task_assignment(self):
        for i in range(4):
            self.coord.register_agent(f"a{i}", np.array([float(i), 0.0]))
        
        tasks = [
            ("t1", np.array([0.5, 0.5])),
            ("t2", np.array([2.5, 0.5])),
        ]
        
        self.coord.assign_tasks(tasks)
        
        # 检查任务分配
        assigned = [a.task_id for a in self.coord.agents.values() if a.task_id is not None]
        self.assertEqual(len(assigned), 2)
    
    def test_step_updates_trajectories(self):
        self.coord.register_agent("a1", np.array([0.0, 0.0]))
        self.coord.agents["a1"].velocity = np.array([0.1, 0.0])
        
        self.coord.step(0.01)
        
        self.assertEqual(len(self.coord.trajectories["a1"]), 1)
    
    def test_get_formation_center(self):
        for i in range(3):
            self.coord.register_agent(f"a{i}", np.array([float(i), 0.0]))
        
        self.coord.create_formation("f1", FormationType.LINE, np.array([1.0, 1.0]))
        
        center = self.coord.get_formation_center("f1")
        
        # 3个智能体在 x=0,1,2
        self.assertAlmostEqual(center[0], 1.0, places=1)
    
    def test_get_status(self):
        for i in range(3):
            self.coord.register_agent(f"a{i}", np.array([float(i), 0.0]))
        
        self.coord.create_formation("f1", FormationType.LINE, np.array([0.0, 0.0]))
        
        status = self.coord.get_status()
        
        self.assertEqual(status["total_agents"], 3)
        self.assertEqual(status["active_formations"], 1)
    
    def test_agents_with_3d_position(self):
        """测试3D位置支持"""
        self.coord.register_agent("a1", np.array([0.0, 0.0, 0.0]))  # x, y, theta
        self.assertEqual(len(self.coord.agents["a1"].position), 3)
    
    def test_coordinator_repr(self):
        self.coord.register_agent("a1", np.array([0.0, 0.0]))
        self.assertIn("MultiAgentCoordinator", repr(self.coord))
    
    def test_collision_detection_empty(self):
        risks = self.coord.detect_collisions()
        self.assertEqual(len(risks), 0)


class TestFormationSlotGeneration(unittest.TestCase):
    """测试编队槽位生成"""
    
    def test_line_formation_spacing(self):
        coord = MultiAgentCoordinator()
        for i in range(4):
            coord.register_agent(f"a{i}", np.array([float(i), 0.0]))
        
        task = coord.create_formation("f1", FormationType.LINE, np.array([0.0, 0.0]))
        
        # 检查槽位间距
        for i in range(1, len(task.slots)):
            prev_pos = task.slots[i-1].relative_position
            curr_pos = task.slots[i].relative_position
            dist = np.linalg.norm(curr_pos - prev_pos)
            self.assertAlmostEqual(dist, 1.0, places=1)
    
    def test_v_shape_formation(self):
        coord = MultiAgentCoordinator()
        for i in range(5):
            coord.register_agent(f"a{i}", np.array([float(i), 0.0]))
        
        task = coord.create_formation("f1", FormationType.V_SHAPE, np.array([0.0, 0.0]))
        
        self.assertEqual(len(task.slots), 5)
        
        # 槽位0应该在原点
        self.assertEqual(task.slots[0].relative_position[0], 0.0)
        self.assertEqual(task.slots[0].relative_position[1], 0.0)


class TestCollisionRisk(unittest.TestCase):
    """测试碰撞风险数据结构"""
    
    def test_collision_risk_creation(self):
        risk = CollisionRisk(
            agent_a="a1",
            agent_b="a2",
            distance=0.5,
            time_to_collision=2.0,
            severity="medium"
        )
        
        self.assertEqual(risk.agent_a, "a1")
        self.assertEqual(risk.agent_b, "a2")
        self.assertEqual(risk.distance, 0.5)
        self.assertEqual(risk.time_to_collision, 2.0)
        self.assertEqual(risk.severity, "medium")


class TestCoordinationGrades(unittest.TestCase):
    """测试AGV五级协调规格"""
    
    def test_all_grades(self):
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_coordination_spec(grade)
            self.assertIn('multi_agent', spec)
            self.assertIn('max_agents', spec)
            self.assertIn('formation', spec)
            self.assertIn('collision_avoidance', spec)
    
    def test_grade_s_no_multi_agent(self):
        spec = get_coordination_spec('S')
        self.assertFalse(spec['multi_agent'])
        self.assertEqual(spec['max_agents'], 1)
    
    def test_grade_l_multi_agent(self):
        spec = get_coordination_spec('L')
        self.assertTrue(spec['multi_agent'])
        self.assertEqual(spec['max_agents'], 4)
        self.assertTrue(spec['formation'])
    
    def test_grade_xxl_max_agents(self):
        spec = get_coordination_spec('XXL')
        self.assertEqual(spec['max_agents'], 20)
        self.assertEqual(spec['collision_avoidance'], "optimal")


class TestAgentState(unittest.TestCase):
    """测试智能体状态"""
    
    def test_agent_state_creation(self):
        state = AgentState(
            agent_id="test",
            position=np.array([1.0, 2.0]),
            velocity=np.array([0.1, 0.2])
        )
        
        self.assertEqual(state.agent_id, "test")
        self.assertEqual(state.position[0], 1.0)
        self.assertEqual(state.in_formation, False)
        self.assertEqual(state.battery_level, 1.0)
    
    def test_agent_state_with_list_input(self):
        state = AgentState(
            agent_id="test",
            position=[1.0, 2.0],
            velocity=[0.1, 0.2]
        )
        
        self.assertIsInstance(state.position, np.ndarray)
        self.assertIsInstance(state.velocity, np.ndarray)


if __name__ == '__main__':
    unittest.main()
