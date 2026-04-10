"""
multi_agv_swarm_tests.py - 多AGV蜂群协同测试
============================================

测试内容:
- 多AGV任务分配
- 碰撞避免
- 路径协调
- 蜂群算法基本功能
- 协同搬运测试
"""

import pytest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from control.swarm_control import (
    SwarmController,
    FormationSpec,
    CollisionAvoidance,
    FormationShape,
    SwarmAgent,
    FormationController,
    ConsensusType,
    get_swarm_spec,
)


class TestSwarmController:
    """蜂群控制器测试"""

    def test_init_multiple_agv(self):
        """初始化多个AGV"""
        controller = SwarmController("M", FormationShape.LINE)
        spec = get_swarm_spec("M")
        assert controller.spec.max_agents >= 8
        assert len(controller.agents) == 0

    def test_add_agent(self):
        """添加智能体"""
        controller = SwarmController("M", FormationShape.LINE)
        agent_id = controller.add_agent(np.array([0.0, 0.0]))
        assert agent_id == 0
        assert len(controller.agents) == 1
        assert np.array_equal(controller.agents[0].position, np.array([0.0, 0.0]))

    def test_add_multiple_agents(self):
        """添加多个智能体"""
        controller = SwarmController("S", FormationShape.LINE)
        for i in range(4):
            controller.add_agent(np.array([float(i), 0.0]))
        assert len(controller.agents) == 4
        # S级最大4台，不能再添加了
        with pytest.raises(RuntimeError):
            controller.add_agent(np.array([4.0, 0.0]))

    def test_step_update(self):
        """单步更新"""
        controller = SwarmController("S", FormationShape.LINE)
        controller.add_agent(np.array([-1.0, 0.0]))
        controller.add_agent(np.array([1.0, 0.0]), is_leader=True)
        # 执行一步
        controller.step(leader_ref=np.array([2.0, 0.0]))
        # 位置应该更新（向目标移动，从 1.0 开始往 2.0 移动，一步之后应该大于等于原值）
        assert controller.agents[1].position[0] >= 1.0
        assert controller.time > 0

    def test_get_positions(self):
        """获取所有位置"""
        controller = SwarmController("M", FormationShape.GRID)
        positions = [np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([1.0, 1.0])]
        for pos in positions:
            controller.add_agent(pos)
        gotten = controller.get_positions()
        assert gotten.shape == (4, 2)
        assert np.array_equal(gotten[0], np.array([0.0, 0.0]))

    def test_change_formation(self):
        """切换编队"""
        controller = SwarmController("M", FormationShape.LINE)
        controller.change_formation(FormationShape.CIRCLE)
        assert controller.formation_shape == FormationShape.CIRCLE

    def test_validate_swarm(self):
        """验证蜂群"""
        controller = SwarmController("M", FormationShape.LINE)
        # 添加两个距离足够远的AGV
        controller.add_agent(np.array([0.0, 0.0]))
        controller.add_agent(np.array([10.0, 0.0]))
        valid, errors = controller.validate_swarm()
        assert valid
        assert len(errors) == 0

    def test_validate_collision_warning(self):
        """验证碰撞检测"""
        controller = SwarmController("M", FormationShape.LINE)
        # collision_radius 默认 0.3
        # 添加两个相距 0.25 小于 collision_radius → 碰撞
        controller.add_agent(np.array([0.0, 0.0]))
        controller.add_agent(np.array([0.25, 0.0]))
        valid, errors = controller.validate_swarm()
        assert not valid
        assert len(errors) > 0
        assert "碰撞风险" in errors[0]


class TestCollisionAvoidance:
    """碰撞避免测试"""

    def test_check_collisions(self):
        """检查碰撞"""
        spec = get_swarm_spec("M")
        spec.collision_radius = 0.6
        avoid = CollisionAvoidance(spec)
        # 创建两个距离很近的agent (0.5 < 0.6 → collision)
        agent1 = SwarmAgent(0, np.array([0.0, 0.0]), np.zeros(2))
        agent2 = SwarmAgent(1, np.array([0.5, 0.0]), np.zeros(2))
        collisions = avoid.check_collisions([agent1, agent2])
        assert len(collisions) == 1  # 有一对碰撞

    def test_check_no_collisions(self):
        """无碰撞"""
        spec = get_swarm_spec("M")
        avoid = CollisionAvoidance(spec)
        agent1 = SwarmAgent(0, np.array([0.0, 0.0]), np.zeros(2))
        agent2 = SwarmAgent(1, np.array([5.0, 0.0]), np.zeros(2))
        collisions = avoid.check_collisions([agent1, agent2])
        assert len(collisions) == 0

    def test_compute_avoidance(self):
        """计算避障"""
        spec = get_swarm_spec("M")
        avoid = CollisionAvoidance(spec)
        agent1 = SwarmAgent(0, np.array([0.0, 0.0]), np.zeros(2))
        agent2 = SwarmAgent(1, np.array([0.5, 0.0]), np.zeros(2))
        controls = avoid.compute_avoidance_control([agent1, agent2])
        assert len(controls) == 2
        # agent1应该获得向左的排斥力，agent2向右
        assert controls[0][0] < 0  # agent1 x方向负
        assert controls[1][0] > 0  # agent2 x方向正


class TestFormationSpec:
    """编队规格测试"""

    def test_for_grade_s(self):
        """S级规格"""
        spec = get_swarm_spec("S")
        assert spec.max_agents == 4
        assert spec.max_speed == 0.3
        assert spec.dimension == 2

    def test_for_grade_m(self):
        """M级规格"""
        spec = get_swarm_spec("M")
        assert spec.max_agents == 8
        assert spec.max_speed == 0.6

    def test_for_grade_xxl(self):
        """XXL级规格"""
        spec = get_swarm_spec("XXL")
        assert spec.max_agents > 32
        assert spec.dimension == 3
        assert spec.max_speed == 2.0

    def test_consensus_type(self):
        """共识类型"""
        spec = get_swarm_spec("M")
        # M级默认一阶共识
        assert spec.consensus_type in [ConsensusType.FIRST_ORDER, ConsensusType.SECOND_ORDER]


class TestFormationController:
    """编队控制器测试"""

    def test_compute_line_formation(self):
        """直线编队计算"""
        spec = get_swarm_spec("S")
        controller = FormationController(spec, FormationShape.LINE)
        agent1 = SwarmAgent(0, np.array([-0.5, 0.0]), np.zeros(2))
        agent2 = SwarmAgent(1, np.array([0.5, 0.0]), np.zeros(2))
        controls = controller.compute_formation_control([agent1, agent2])
        assert len(controls) == 2
        # 都应该朝着期望位置移动
        assert isinstance(controls[0], np.ndarray)

    def test_desired_positions_line(self):
        """直线期望位置"""
        spec = get_swarm_spec("S")
        controller = FormationController(spec, FormationShape.LINE)
        # target_positions 存储期望偏移
        desired = controller.target_positions
        # 对于S级max_agents=4
        assert len(desired) == 4
        # x应该间隔排列
        if spec.dimension == 2:
            assert desired[1][0] > desired[0][0]
            assert desired[2][0] > desired[1][0]
            # y都相同
            assert desired[0][1] == desired[1][1] == desired[2][1]

    def test_desired_positions_grid(self):
        """网格期望位置"""
        spec = get_swarm_spec("M")
        controller = FormationController(spec, FormationShape.GRID)
        desired = controller.target_positions
        # 对于M级max_agents=8
        assert len(desired) == 8
        # 应该有多行多列
        xs = [p[0] for p in desired]
        ys = [p[1] for p in desired]
        assert len(set(xs)) >= 2  # 至少两个不同x坐标
        assert len(set(ys)) >= 2  # 至少两个y坐标

    def test_desired_positions_circle(self):
        """圆形期望位置"""
        spec = get_swarm_spec("M")
        controller = FormationController(spec, FormationShape.CIRCLE)
        desired = controller.target_positions
        n = spec.max_agents
        assert len(desired) == n
        # 都应该在半径附近
        d = spec.min_safe_distance
        for p in desired:
            r = np.linalg.norm(p)
            assert 0.5 * d < r < 1.5 * d  # 在期望距离附近


class TestMultiAGVIntegration:
    """多AGV集成测试"""

    def test_full_formation_control_circle(self):
        """完整圆形编队控制"""
        controller = SwarmController("S", FormationShape.CIRCLE)
        # 添加4个agent在错误位置
        positions = [
            np.array([-2.0, 0.0]),
            np.array([-1.0, -1.0]),
            np.array([0.0, -2.0]),
            np.array([1.0, -1.0]),
        ]
        for pos in positions:
            controller.add_agent(pos)

        # 运行几步让它们形成圆形
        for _ in range(50):
            controller.step()

        # 验证最终位置接近圆形分布
        positions = controller.get_positions()
        for pos in positions:
            r = np.linalg.norm(pos)
            assert 0.2 < r < 3.0  # 在合理范围

        # 验证没有碰撞
        valid, errors = controller.validate_swarm()
        assert valid or len(errors) == 0

    def test_leader_follower_movement(self):
        """Leader-Follower移动"""
        controller = SwarmController("S", FormationShape.LINE)
        controller.add_agent(np.array([-2.0, 0.0]), is_leader=False)
        controller.add_agent(np.array([-1.0, 0.0]), is_leader=False)
        controller.add_agent(np.array([0.0, 0.0]), is_leader=True)  # leader在最后

        # leader向右移动
        for _ in range(100):
            controller.step(leader_ref=np.array([5.0, 0.0]))

        # 整个编队应该跟着leader向右移动
        positions = controller.get_positions()
        # 都应该向右移动了
        assert positions[2][0] > 1.0
        # 保持直线队形相对间距
        for i in range(2):
            spacing = positions[i+1][0] - positions[i][0]
            assert 0.5 < spacing < 1.2  # S级min_safe_distance = 1.0


    def test_different_formation_shapes(self):
        """测试不同编队形状"""
        controller = SwarmController("M", FormationShape.LINE)
        for i in range(4):
            controller.add_agent(np.array([float(i-2), 0.0]))

        # 检查每种形状都能切换
        for shape in [FormationShape.LINE, FormationShape.CIRCLE,
                     FormationShape.SQUARE, FormationShape.V_SHAPE]:
            controller.change_formation(shape)
            for _ in range(10):
                controller.step()
            # 不抛出异常就是成功
            assert True

    def test_3d_formation_xxl(self):
        """XXL级支持3D编队"""
        spec = get_swarm_spec("XXL")
        assert spec.dimension == 3
        controller = SwarmController("XXL", FormationShape.GRID)
        # 添加8个agent (2x2x2网格)
        for x in [-0.5, 0.5]:
            for y in [-0.5, 0.5]:
                for z in [-0.5, 0.5]:
                    controller.add_agent(np.array([float(x), float(y), float(z)]))
        assert len(controller.agents) == 8
        # 运行几步
        for _ in range(10):
            controller.step()
        # 验证3D位置保留
        positions = controller.get_positions()
        assert positions.shape == (8, 3)


class TestSwarmAgent:
    """SwarmAgent测试"""

    def test_agent_init(self):
        """初始化"""
        agent = SwarmAgent(0, np.array([1.0, 2.0]), np.array([0.5, 0.0]))
        assert agent.agent_id == 0
        assert np.array_equal(agent.position, np.array([1.0, 2.0]))
        assert np.array_equal(agent.velocity, np.array([0.5, 0.0]))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
