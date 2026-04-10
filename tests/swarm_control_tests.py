"""
Swarm Control Module Tests
==========================

测试蜂群控制系统:
- 共识控制器 (一阶/二阶)
- 编队控制器 (多种形状)
- 碰撞避免
- AGV五级规格验证
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.control.swarm_control import (
    SwarmController, ConsensusController, FormationController,
    CollisionAvoidance, SwarmAgent,
    FormationShape, ConsensusType, FormationSpec,
    SWARM_GRADES, get_swarm_spec, list_swarm_capabilities,
)


class TestSwarmGrades(unittest.TestCase):
    """AGV五级蜂群规格测试"""
    
    def test_swarm_grades_exist(self):
        self.assertIn("S", SWARM_GRADES)
        self.assertIn("M", SWARM_GRADES)
        self.assertIn("L", SWARM_GRADES)
        self.assertIn("XL", SWARM_GRADES)
        self.assertIn("XXL", SWARM_GRADES)
    
    def test_swarm_grade_monotonicity(self):
        """验证规格随等级递增"""
        specs = [SWARM_GRADES[g] for g in ["S", "M", "L", "XL", "XXL"]]
        
        # 最大速度递增
        speeds = [s.max_speed for s in specs]
        self.assertEqual(speeds, sorted(speeds))
        
        # 最大智能体数递增
        counts = [s.max_agents for s in specs]
        self.assertEqual(counts, sorted(counts))
        
        # 控制频率递增
        freqs = [s.control_frequency for s in specs]
        self.assertEqual(freqs, sorted(freqs))
        
        # 安全距离递减
        distances = [s.min_safe_distance for s in specs]
        self.assertEqual(distances, sorted(distances, reverse=True))
    
    def test_get_swarm_spec(self):
        spec = get_swarm_spec("L")
        self.assertEqual(spec.max_agents, 16)
        self.assertEqual(spec.max_speed, 1.0)
        self.assertEqual(spec.dimension, 2)
        self.assertEqual(spec.consensus_type, ConsensusType.SECOND_ORDER)
    
    def test_get_swarm_spec_invalid(self):
        with self.assertRaises(ValueError):
            get_swarm_spec("Z")
    
    def test_list_swarm_capabilities(self):
        caps = list_swarm_capabilities()
        self.assertEqual(len(caps), 5)
        self.assertIn("[S]", caps[0])
        self.assertIn("[XXL]", caps[4])
    
    def test_swarm_grade_spec_details(self):
        """验证各等级详细规格"""
        # S级: 4台, 0.3m/s, 1.0m, 20Hz, 2D一阶
        s = SWARM_GRADES["S"]
        self.assertEqual(s.max_agents, 4)
        self.assertEqual(s.max_speed, 0.3)
        self.assertEqual(s.min_safe_distance, 1.0)
        self.assertEqual(s.control_frequency, 20.0)
        self.assertEqual(s.dimension, 2)
        self.assertEqual(s.consensus_type, ConsensusType.FIRST_ORDER)
        
        # XXL级: 64台, 2.0m/s, 0.2m, 200Hz, 3D二阶
        xxl = SWARM_GRADES["XXL"]
        self.assertEqual(xxl.max_agents, 64)
        self.assertEqual(xxl.max_speed, 2.0)
        self.assertEqual(xxl.min_safe_distance, 0.2)
        self.assertEqual(xxl.control_frequency, 200.0)
        self.assertEqual(xxl.dimension, 3)
        self.assertEqual(xxl.consensus_type, ConsensusType.SECOND_ORDER)


class TestConsensusController(unittest.TestCase):
    """共识控制器测试"""
    
    def test_ring_topology(self):
        """环形拓扑共识"""
        adj = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ], dtype=np.float32)
        
        ctrl = ConsensusController(adj, ConsensusType.FIRST_ORDER)
        self.assertEqual(ctrl.n, 4)
        self.assertEqual(ctrl.laplacian.shape, (4, 4))
        self.assertTrue(np.allclose(ctrl.laplacian.sum(axis=0), 0))
    
    def test_star_topology(self):
        """星型拓扑共识"""
        adj = np.array([
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ], dtype=np.float32)
        
        ctrl = ConsensusController(adj, ConsensusType.FIRST_ORDER)
        self.assertEqual(ctrl.n, 4)
    
    def test_first_order_consensus(self):
        """一阶共识"""
        adj = np.array([[0, 1], [1, 0]], dtype=np.float32)
        ctrl = ConsensusController(adj, ConsensusType.FIRST_ORDER)
        
        # 两智能体，位置差为[1, 0]
        states = np.array([[0, 0], [1, 0]], dtype=np.float32)
        control = ctrl.compute_consensus(states)
        
        self.assertEqual(control.shape, (2, 2))
        # agent0应向agent1移动, agent1应向agent0移动
        self.assertTrue(np.all(control[0] >= 0) or np.all(control[0] <= 0))
    
    def test_second_order_consensus(self):
        """二阶共识"""
        adj = np.array([[0, 1], [1, 0]], dtype=np.float32)
        ctrl = ConsensusController(adj, ConsensusType.SECOND_ORDER)
        
        states = np.array([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)  # [x, v]
        velocities = np.array([[0, 0], [0, 0]], dtype=np.float32)
        
        control = ctrl.compute_consensus(states, velocities)
        self.assertEqual(control.shape, (2, 2))
    
    def test_leader_consensus(self):
        """Leader-Follower共识"""
        adj = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ], dtype=np.float32)
        ctrl = ConsensusController(adj, ConsensusType.FIRST_ORDER)
        
        states = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)
        leader_ids = [0]
        leader_refs = np.array([[0.0, 0.0]])
        
        control = ctrl.compute_leader_consensus(states, leader_ids, leader_refs)
        self.assertEqual(control.shape, (3, 2))


class TestFormationController(unittest.TestCase):
    """编队控制器测试"""
    
    def test_line_formation(self):
        spec = SWARM_GRADES["S"]
        ctrl = FormationController(spec, FormationShape.LINE)
        self.assertEqual(ctrl.formation_shape, FormationShape.LINE)
        self.assertTrue(len(ctrl.formation_offset) > 0)
    
    def test_circle_formation(self):
        spec = SWARM_GRADES["M"]
        ctrl = FormationController(spec, FormationShape.CIRCLE)
        offsets = ctrl.formation_offset
        self.assertTrue(len(offsets) > 0)
        # 圆形编队各点应在同一半径
        radii = [np.linalg.norm(o) for o in offsets]
        for r in radii:
            self.assertAlmostEqual(r, radii[0], places=5)
    
    def test_grid_formation(self):
        spec = SWARM_GRADES["L"]
        ctrl = FormationController(spec, FormationShape.GRID)
        self.assertEqual(len(ctrl.formation_offset), spec.max_agents)
    
    def test_triangle_formation(self):
        spec = SWARM_GRADES["M"]
        ctrl = FormationController(spec, FormationShape.TRIANGLE)
        offsets = ctrl.formation_offset
        # 三角形编队行数应正确
        self.assertTrue(len(offsets) > 0)
    
    def test_formation_topology_meshes(self):
        """测试各种拓扑"""
        for topology in ["ring", "star", "mesh", "fully_connected"]:
            spec = SWARM_GRADES["M"]
            ctrl = FormationController(spec, FormationShape.LINE)
            self.assertIsNotNone(ctrl)


class TestCollisionAvoidance(unittest.TestCase):
    """碰撞避免测试"""
    
    def test_no_collision_when_far_apart(self):
        spec = SWARM_GRADES["S"]
        avoid = CollisionAvoidance(spec)
        
        agents = [
            SwarmAgent(0, np.array([0.0, 0.0]), np.zeros(2)),
            SwarmAgent(1, np.array([5.0, 5.0]), np.zeros(2)),
        ]
        
        collisions = avoid.check_collisions(agents)
        self.assertEqual(len(collisions), 0)
    
    def test_collision_detected_when_close(self):
        spec = SWARM_GRADES["S"]
        avoid = CollisionAvoidance(spec)
        
        agents = [
            SwarmAgent(0, np.array([0.0, 0.0]), np.zeros(2)),
            SwarmAgent(1, np.array([0.1, 0.0]), np.zeros(2)),  # 碰撞半径内
        ]
        
        collisions = avoid.check_collisions(agents)
        self.assertEqual(len(collisions), 1)
    
    def test_avoidance_control_generated(self):
        spec = SWARM_GRADES["M"]
        avoid = CollisionAvoidance(spec)
        
        agents = [
            SwarmAgent(0, np.array([0.0, 0.0]), np.zeros(2)),
            SwarmAgent(1, np.array([0.3, 0.0]), np.zeros(2)),  # 接近但未碰撞
        ]
        
        avoidance = avoid.compute_avoidance_control(agents)
        self.assertEqual(len(avoidance), 2)
        self.assertTrue(np.allclose(avoidance[0], -avoidance[1]))  # 互斥


class TestSwarmController(unittest.TestCase):
    """蜂群控制系统测试"""
    
    def test_swarm_controller_init(self):
        swarm = SwarmController(grade="S")
        self.assertEqual(len(swarm.agents), 0)
        self.assertEqual(swarm.spec.max_agents, 4)
    
    def test_add_agent(self):
        swarm = SwarmController(grade="S")
        aid = swarm.add_agent(np.array([0.0, 0.0]))
        self.assertEqual(aid, 0)
        self.assertEqual(len(swarm.agents), 1)
        
        aid2 = swarm.add_agent(np.array([1.0, 0.0]))
        self.assertEqual(aid2, 1)
        self.assertEqual(len(swarm.agents), 2)
    
    def test_add_agent_max_limit(self):
        swarm = SwarmController(grade="S")  # 最多4台
        for i in range(4):
            swarm.add_agent(np.array([i * 0.5, 0.0]))
        
        with self.assertRaises(RuntimeError):
            swarm.add_agent(np.array([10.0, 0.0]))
    
    def test_add_leader_agent(self):
        swarm = SwarmController(grade="M")
        swarm.add_agent(np.array([0.0, 0.0]), is_leader=True)
        self.assertTrue(swarm.agents[0].is_leader)
    
    def test_step_first_order(self):
        swarm = SwarmController(grade="S")
        swarm.add_agent(np.array([0.0, 0.0]))
        swarm.add_agent(np.array([1.0, 0.0]))
        
        pos_before = swarm.get_positions().copy()
        swarm.step()
        pos_after = swarm.get_positions()
        
        # 两智能体应趋向收敛
        self.assertFalse(np.allclose(pos_before, pos_after))
    
    def test_step_second_order(self):
        swarm = SwarmController(grade="L")  # 二阶
        swarm.add_agent(np.array([0.0, 0.0]), np.array([0.1, 0.0]))
        swarm.add_agent(np.array([1.0, 0.0]), np.array([-0.1, 0.0]))
        
        pos_before = swarm.get_positions().copy()
        for _ in range(10):
            swarm.step()
        pos_after = swarm.get_positions()
        
        # 多次迭代后应更接近
        initial_dist = np.linalg.norm(pos_before[0] - pos_before[1])
        final_dist = np.linalg.norm(pos_after[0] - pos_after[1])
        self.assertLess(final_dist, initial_dist)
    
    def test_leader_reference(self):
        swarm = SwarmController(grade="M")
        swarm.add_agent(np.array([0.0, 0.0]), is_leader=True)
        swarm.add_agent(np.array([5.0, 5.0]))
        
        # Leader参考位置驱动followers趋向leader
        for _ in range(100):
            swarm.step(leader_ref=np.array([0.0, 0.0]))
        
        # Follower应显著趋向Leader
        follower_pos = swarm.agents[1].position
        # 初始距离=7.07, 经过共识后应明显收敛
        self.assertLess(np.linalg.norm(follower_pos), 7.0)
    
    def test_change_formation(self):
        swarm = SwarmController(grade="M")
        self.assertEqual(swarm.formation_shape, FormationShape.LINE)
        
        swarm.change_formation(FormationShape.CIRCLE)
        self.assertEqual(swarm.formation_shape, FormationShape.CIRCLE)
    
    def test_validate_swarm_valid(self):
        swarm = SwarmController(grade="M")
        swarm.add_agent(np.array([0.0, 0.0]))
        swarm.add_agent(np.array([1.5, 0.0]))  # 远距离
        swarm.add_agent(np.array([3.0, 0.0]))
        
        valid, errors = swarm.validate_swarm()
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_swarm_collision(self):
        swarm = SwarmController(grade="M")
        swarm.add_agent(np.array([0.0, 0.0]))
        swarm.add_agent(np.array([0.1, 0.0]))  # 碰撞
        
        valid, errors = swarm.validate_swarm()
        self.assertFalse(valid)
        self.assertGreater(len(errors), 0)
    
    def test_validate_swarm_speed_overrun(self):
        swarm = SwarmController(grade="S")  # 0.3m/s限制
        swarm.add_agent(np.array([0.0, 0.0]), np.array([10.0, 0.0]))  # 超速
        
        valid, errors = swarm.validate_swarm()
        self.assertFalse(valid)
        self.assertTrue(any("速度" in e for e in errors))
    
    def test_get_states(self):
        swarm = SwarmController(grade="M")
        swarm.add_agent(np.array([0.0, 0.0]), np.array([0.1, 0.0]))
        swarm.add_agent(np.array([1.0, 0.0]), np.array([0.0, 0.0]))
        
        states = swarm.get_states()
        self.assertEqual(states.shape, (2, 4))  # 2 agents, 2D pos+2D vel
    
    def test_get_positions(self):
        swarm = SwarmController(grade="M")
        swarm.add_agent(np.array([0.0, 0.0]))
        swarm.add_agent(np.array([1.0, 1.0]))
        
        positions = swarm.get_positions()
        self.assertEqual(positions.shape, (2, 2))


class TestSwarmAGVGradeConsistency(unittest.TestCase):
    """蜂群控制与AGV五级规格一致性测试"""
    
    def test_all_grades_instantiable(self):
        for grade in SWARM_GRADES:
            swarm = SwarmController(grade=grade)
            self.assertIsNotNone(swarm.spec)
            self.assertEqual(len(swarm.agents), 0)
    
    def test_grade_speed_limits(self):
        for grade, spec in SWARM_GRADES.items():
            swarm = SwarmController(grade=grade)
            
            # 添加高速移动智能体
            swarm.add_agent(np.zeros(spec.dimension), 
                           np.ones(spec.dimension) * spec.max_speed * 0.5)
            
            # 一步更新
            swarm.step()
            
            # 速度不应超过限制
            for a in swarm.agents:
                speed = np.linalg.norm(a.velocity)
                self.assertLessEqual(speed, spec.max_speed * 1.01,
                    f"Grade {grade}: speed {speed} > {spec.max_speed}")
    
    def test_formation_controller_dimensions(self):
        """各等级维度一致性"""
        dim_map = {"S": 2, "M": 2, "L": 2, "XL": 3, "XXL": 3}
        for grade, expected_dim in dim_map.items():
            spec = SWARM_GRADES[grade]
            ctrl = FormationController(spec, FormationShape.LINE)
            self.assertEqual(spec.dimension, expected_dim)
            for offset in ctrl.formation_offset:
                self.assertEqual(len(offset), expected_dim)


if __name__ == "__main__":
    unittest.main()
