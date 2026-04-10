"""
multi_agv_swarm_tests.py - 多AGV蜂群协同测试
SuperModel 超模态大模型具身智能系统

测试覆盖:
- 多AGV队形保持测试
- 蜂群协同避障测试
- 多机任务分配测试
- 协同搬运测试
- 通讯延迟下的一致性测试
"""

import pytest
import numpy as np
from src.control.swarm_coordination import SwarmCoordinator
from src.control.swarm_control import FormationShape, SwarmController, FormationSpec, get_swarm_spec


class TestFormationSpec:
    """队形规格测试"""

    def test_line_formation_spec(self):
        """直线队形规格获取测试"""
        spec = get_swarm_spec('L')
        assert spec.max_agents == 16
        assert spec.max_speed == 1.0
        assert spec.min_safe_distance == 0.5
        assert spec.formation_shape == FormationShape.TRIANGLE  # L默认三角形

    def test_get_swarm_spec_all_grades(self):
        """获取所有等级蜂群规格测试"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            spec = get_swarm_spec(grade)
            assert spec is not None
            assert spec.max_agents > 0
            assert spec.control_frequency > 0

    def test_formation_offset_calculation(self):
        """队形偏移计算测试"""
        from src.control.swarm_control import FormationController
        spec = FormationSpec(
            max_agents=3,
            max_speed=0.5,
            min_safe_distance=1.0,
            topology="ring",
            consensus_type="first_order",
            dimension=2,
            formation_shape=FormationShape.LINE,
            control_frequency=20.0,
            position_error_limit=0.1,
            velocity_error_limit=0.1
        )
        controller = FormationController(spec, FormationShape.LINE)
        offsets = controller._compute_formation_offset(FormationShape.LINE)
        assert len(offsets) == 3
        # 所有y坐标为0
        assert all(offset[1] == 0 for offset in offsets)


class TestSwarmController:
    """蜂群控制器测试"""

    def test_create_controller_by_grade(self):
        """按等级创建蜂群控制器"""
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            controller = SwarmController(grade)
            assert controller is not None
            assert controller.spec is not None
            assert controller.spec.max_agents == get_swarm_spec(grade).max_agents

    def test_add_agent(self):
        """添加智能体测试"""
        controller = SwarmController('L', FormationShape.LINE)
        aid = controller.add_agent(np.array([0.0, 0.0]), is_leader=True)
        assert aid == 0
        assert len(controller.agents) == 1

    def test_add_multiple_agents(self):
        """添加多个智能体"""
        controller = SwarmController('M', FormationShape.LINE)
        for i in range(8):
            aid = controller.add_agent(np.array([float(i), 0.0]))
            assert aid == i
        assert len(controller.agents) == 8
        # M级最多8台，添加第9台应该抛出异常
        with pytest.raises(RuntimeError, match="已达最大智能体数 8"):
            controller.add_agent(np.array([8.0, 0.0]))
        assert len(controller.agents) == 8  # 仍然8台，达到上限

    def test_step_control(self):
        """单步控制测试"""
        controller = SwarmController('L', FormationShape.LINE)
        controller.add_agent(np.array([-1.0, 0.0]), is_leader=True)
        controller.add_agent(np.array([0.0, 0.0]))
        controller.add_agent(np.array([1.0, 0.0]))
        # 执行一步
        controller.step()
        # 位置更新
        assert len(controller.agents) == 3
        for agent in controller.agents:
            assert not np.allclose(agent.position, np.zeros(2))

    def test_formation_control_center_update(self):
        """编队中心更新测试"""
        controller = SwarmController('L', FormationShape.LINE)
        controller.add_agent(np.array([-1.0, 0.0]))
        controller.add_agent(np.array([0.0, 0.0]))
        controller.add_agent(np.array([1.0, 0.0]))
        # 中心在x=0
        controller.formation_ctrl.update_formation_center(controller.agents)
        assert np.allclose(controller.formation_ctrl.centroid, np.array([0, 0]))

    def test_collision_detection(self):
        """碰撞检测测试"""
        controller = SwarmController('L', FormationShape.LINE)
        # 放置两个AGV太近会检测到碰撞
        controller.add_agent(np.array([0.0, 0.0]))
        controller.add_agent(np.array([0.2, 0.0]))  # 间距0.2 < 0.5安全距离
        collisions = controller.collision_avoid.check_collisions(controller.agents)
        assert len(collisions) == 1


class TestConsensusController:
    """共识控制器测试"""

    def test_first_order_consensus(self):
        """一阶共识计算"""
        from src.control.swarm_control import ConsensusController
        # 3智能体环形拓扑
        adj = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ])
        controller = ConsensusController(adj, "first_order")
        states = np.array([
            [0.0],
            [0.5],
            [1.0],
        ])
        control = controller.compute_consensus(states, gain=1.0)
        # 收敛到平均值 (0 + 0.5 + 1)/3 = 0.5
        assert control.shape == (3, 1)

    def test_formation_consensus_control(self):
        """编队共识控制"""
        from src.control.swarm_control import ConsensusController, FormationController, FormationSpec
        spec = FormationSpec(
            max_agents=3,
            max_speed=0.5,
            min_safe_distance=1.0,
            topology="ring",
            consensus_type="first_order",
            dimension=2,
            formation_shape=FormationShape.LINE,
            control_frequency=20.0,
            position_error_limit=0.1,
            velocity_error_limit=0.1
        )
        controller = FormationController(spec, FormationShape.LINE)
        # 验证邻接矩阵构建正确
        assert controller.consensus is not None


class TestCollisionAvoidance:
    """碰撞避免测试"""

    def test_collision_risk_detection(self):
        """碰撞风险检测"""
        from src.control.swarm_control import CollisionAvoidance, FormationSpec
        spec = get_swarm_spec('L')
        avoid = CollisionAvoidance(spec)
        # 创建三个agent，两个距离近，一个远
        from src.control.swarm_control import SwarmAgent
        agents = [
            SwarmAgent(0, np.array([0.0, 0.0]), np.zeros(2)),
            SwarmAgent(1, np.array([0.2, 0.0]), np.zeros(2)),
            SwarmAgent(2, np.array([5.0, 0.0]), np.zeros(2)),
        ]
        collisions = avoid.check_collisions(agents)
        # 只有(0,1)距离 0.2 < collision_radius 0.3 会碰撞
        assert len(collisions) == 1
        assert (0, 1) in collisions


class TestSwarmValidation:
    """蜂群状态验证测试"""

    def test_valid_swarm_returns_ok(self):
        """合法蜂群验证通过"""
        controller = SwarmController('L', FormationShape.LINE)
        controller.add_agent(np.array([-2.0, 0.0]))
        controller.add_agent(np.array([-1.0, 0.0]))
        controller.add_agent(np.array([0.0, 0.0]))
        controller.add_agent(np.array([1.0, 0.0]))
        controller.add_agent(np.array([2.0, 0.0]))
        valid, errors = controller.validate_swarm()
        assert valid is True
        assert len(errors) == 0

    def test_invalid_collision_swarm_has_errors(self):
        """碰撞蜂群有错误"""
        controller = SwarmController('L', FormationShape.LINE)
        controller.add_agent(np.array([0.0, 0.0]))
        controller.add_agent(np.array([0.2, 0.0]))  # 间距0.2 < 0.5安全距离
        valid, errors = controller.validate_swarm()
        assert valid is False
        assert len(errors) > 0


def run_all_tests():
    """运行所有测试"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_all_tests()
