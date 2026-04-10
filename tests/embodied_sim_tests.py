"""
embodied_sim_tests.py - 增强具身仿真环境测试
SuperModel 超模态大模型具身智能系统

测试覆盖:
- EmbodiedSimEnv基础测试
- 传感器仿真测试
- 场景生成测试
- 任务指标评估测试
- 多AGV协同测试
"""

import pytest
import numpy as np
from src.simulation.embodied_sim import (
    EmbodiedSimEnv,
    EmbodiedSensorSimulator,
    WarehouseScene,
    TaskMetrics,
    MultiAGVEmbodiedSim,
    create_embodied_sim,
    ContactPoint,
)


class TestEmbodiedSimEnv:
    """单个AGV仿真环境测试"""

    def test_create_empty_env(self):
        """创建空环境测试"""
        env = EmbodiedSimEnv(gui=False)
        env.reset()
        assert env.client is not None
        assert env.plane_id is not None
        env.close()

    def test_load_agv_m_grade(self):
        """加载M级AGV测试"""
        env = EmbodiedSimEnv(gui=False)
        env.reset()
        agv_id = env.load_agv(grade="M", position=(0, 0, 0.15))
        assert agv_id is not None
        assert env.agv_id == agv_id
        pos = env.get_robot_position()
        assert pos.shape == (3,)
        # 实际z坐标由URDF模型决定，大约0.19-0.20m
        assert 0.15 <= pos[2] <= 0.25
        assert np.allclose(pos[:2], np.array([0, 0]))
        env.close()

    def test_add_box_obstacle(self):
        """添加方块障碍物测试"""
        env = EmbodiedSimEnv(gui=False)
        env.reset()
        box_id = env.add_box("obs1", (0.3, 0.3, 0.3), (2.0, 0.0, 0.3), mass=1.0)
        assert box_id is not None
        assert "obs1" in env.objects
        assert box_id in env.obstacles
        env.close()

    def test_set_goal_and_check_reached(self):
        """设置目标和检查到达测试"""
        env = EmbodiedSimEnv(gui=False)
        env.reset()
        env.load_agv(grade="M", position=(0, 0, 0.15))
        goal_pos = np.array([1.0, 0.0])
        env.set_goal(goal_pos, radius=0.5)
        assert env.goal_region is not None
        # AGV在原点，还没到达
        assert not env.is_goal_reached()
        env.close()

    def test_check_collision(self):
        """碰撞检测测试"""
        env = EmbodiedSimEnv(gui=False)
        env.reset()
        env.load_agv(grade="M", position=(0, 0, 0.15))
        # 添加障碍物远离AGV
        env.add_box("obs1", (0.3, 0.3, 0.3), (5.0, 0.0, 0.3), mass=1.0)
        assert not env.check_collision()
        env.close()

    def test_start_finish_task(self):
        """任务开始结束指标测试"""
        env = EmbodiedSimEnv(gui=False)
        env.reset()
        env.load_agv(grade="M")
        metrics = env.start_task("test_task")
        assert metrics.task_id == "test_task"
        assert metrics.start_time is not None
        assert metrics.end_time is None
        # 完成任务
        result = env.finish_current_task(success=True)
        assert result.success
        assert result.end_time is not None
        assert result.completion_time > 0
        env.close()

    def test_task_score_calculation(self):
        """任务得分计算测试"""
        metrics = TaskMetrics(task_id="score_test", start_time=0.0)
        metrics.end_time = 10.0
        metrics.completion_time = 10.0
        metrics.success = True
        metrics.total_distance = 5.0
        metrics.collisions = 0
        metrics.min_obstacle_distance = 1.0
        metrics.force_regulation_error = 1.0
        score = metrics.get_score()
        assert 0 < score < 100
        # 碰撞越多得分越低
        metrics2 = TaskMetrics(task_id="score_test2", start_time=0.0)
        metrics2.end_time = 10.0
        metrics2.success = True
        metrics2.collisions = 3
        score2 = metrics2.get_score()
        assert score2 < score


class TestEmbodiedSensorSimulator:
    """多模态传感器仿真测试"""

    def test_setup_all_sensors(self):
        """设置所有传感器测试"""
        env = EmbodiedSimEnv(gui=False)
        env.reset()
        agv_id = env.load_agv(grade="M")
        sim = EmbodiedSensorSimulator(env.client, agv_id)

        tactile = sim.setup_tactile(grid_size=(16, 16))
        assert tactile is not None
        assert tactile.array_size == (16, 16)

        force = sim.setup_force()
        assert force is not None

        imu = sim.setup_imu()
        assert imu is not None

        env.close()

    def test_simulate_imu(self):
        """IMU仿真测试"""
        env = EmbodiedSimEnv(gui=False)
        env.reset()
        agv_id = env.load_agv(grade="M")
        sim = EmbodiedSensorSimulator(env.client, agv_id)
        sim.setup_imu()
        acc, gyro = sim.simulate_imu()
        assert acc.shape == (3,)
        assert gyro.shape == (3,)
        env.close()


class TestWarehouseScene:
    """仓库场景生成测试"""

    def test_generate_warehouse(self):
        """生成仓库场景测试"""
        env = EmbodiedSimEnv(gui=False)
        scene = WarehouseScene(env)
        scene.generate()
        assert env.agv_id is not None
        assert env.goal_region is not None
        # 应该有多个障碍物
        assert len(env.obstacles) >= 5
        env.close()


class TestMultiAGVEmbodiedSim:
    """多AGV仿真测试"""

    def test_add_multiple_agvs(self):
        """添加多个AGV测试"""
        sim = MultiAGVEmbodiedSim(gui=False)
        id1 = sim.add_agv("agv1", "M", (0.0, 0.0, 0.15))
        id2 = sim.add_agv("agv2", "M", (2.0, 0.0, 0.15))
        assert "agv1" in sim.agv_ids
        assert "agv2" in sim.agv_ids
        assert len(sim.agv_ids) == 2
        sim.close()

    def test_detect_collisions_between_agvs_no_collision(self):
        """AGV分开放置，无碰撞测试"""
        sim = MultiAGVEmbodiedSim(gui=False)
        sim.add_agv("agv1", "M", (0.0, 0.0, 0.15))
        sim.add_agv("agv2", "M", (5.0, 0.0, 0.15))
        collisions = sim.check_collisions_between_agvs()
        assert len(collisions) == 0
        sim.close()


class TestCreateEmbodiedSimFactory:
    """工厂方法测试"""

    def test_create_empty(self):
        """创建空环境工厂测试"""
        env = create_embodied_sim(grade="M", scene_type="empty", gui=False)
        assert env is not None
        assert env.agv_id is not None
        env.close()

    def test_create_navigation(self):
        """创建导航场景工厂测试"""
        env = create_embodied_sim(grade="M", scene_type="navigation", gui=False)
        assert env is not None
        assert env.goal_region is not None
        assert len(env.obstacles) == 2
        env.close()


class TestContactPoint:
    """接触点数据结构测试"""

    def test_create_contact_point(self):
        """创建接触点测试"""
        cp = ContactPoint(
            position=np.array([0, 0, 0]),
            normal=np.array([0, 0, 1]),
            force=np.array([0, 0, 10]),
            distance=0.01,
            body_a=1,
            body_b=2,
        )
        assert cp.position.shape == (3,)
        assert cp.force[2] == 10


def test_full_navigation_task():
    """完整导航任务仿真测试"""
    # 创建环境
    env = create_embodied_sim(grade="M", scene_type="navigation", gui=False)

    # 开始任务
    metrics = env.start_task("navigation_test")

    # 简单控制: 直走
    for i in range(50):
        env.set_wheel_velocity(1.0, 1.0)
        env.step(num_steps=10)

    # 结束任务
    result = env.finish_current_task()
    assert result is not None
    score = result.get_score()
    assert 0 <= score <= 100

    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
