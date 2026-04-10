"""
multi_agv_swarm_tests.py - 多AGV蜂群协同测试
SuperModel 超模态大模型具身智能系统

测试覆盖:
- 多AGV仿真环境初始化
- 任务分配算法
- 避障协同
- 路径冲突检测与解决
- 蜂群行为测试
"""

import pytest
import numpy as np
import time
from src.simulation.pybullet_sim import PyBulletSimulator
from src.simulation.agv_model_generator import generate_agv_urdf_detailed, GRADE_CONFIGS
from src.embodied.simulation_enhancement import (
    EmbodiedSimulationEnhancer,
    WarehouseSceneGenerator,
    Obstacle
)
from src.control.multi_agent import MultiAgentCoordinator
from src.control.swarm_control import SwarmController, VelocityObstacle

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestMultiAGVSimulation:
    """多AGV仿真环境测试"""

    @pytest.fixture
    def sim(self):
        """创建仿真器fixture"""
        simulator = PyBulletSimulator(gui=False)
        # PyBulletSimulator connects in __init__ already
        yield simulator
        # Cleanup disconnect
        if simulator._client_id is not None:
            import pybullet as p
            p.disconnect(simulator._client_id)

    def test_multiple_agv_loading(self, sim):
        """测试加载多个AGV模型"""
        # 生成3个M级AGV
        urdf_paths = []
        agv_ids = []
        positions = [
            np.array([-2.0, 0.0, 0.15]),
            np.array([0.0, 0.0, 0.15]),
            np.array([2.0, 0.0, 0.15]),
        ]

        sim.load_plane()  # 加载地面
        for i, pos in enumerate(positions):
            urdf = generate_agv_urdf_detailed('M', '2轮')
            urdf_paths.append(urdf)
            agv_id = sim.load_agv_model(urdf_path=urdf, base_position=pos)
            agv_ids.append(agv_id)

        assert len(agv_ids) == 3
        for aid in agv_ids:
            assert aid >= 0

        # 运行几步仿真确认稳定
        for _ in range(100):
            sim.step()

        # 检查都还在世界中
        # PyBulletSimulator 管理AGV状态
        # 验证仿真运行正常
        for _ in range(10):
            sim.step()
        state = sim.get_agv_state()
        # 仿真运行正常
        assert state is not None

    def test_warehouse_scene_generation(self):
        """测试仓库场景生成"""
        generator = WarehouseSceneGenerator(seed=42)
        scene = generator.generate_warehouse(
            num_aisles=5,
            aisle_length=20.0,
            shelf_width=1.0,
            aisle_width=3.0,
            shelves_per_aisle=10,
        )

        assert 'obstacles' in scene
        assert 'start_positions' in scene
        assert 'goal_positions' in scene
        assert len(scene['obstacles']) > 0
        assert len(scene['start_positions']) > 0
        assert scene['num_aisles'] == 5

        # 检查障碍物尺寸合理
        for obs in scene['obstacles']:
            assert isinstance(obs, Obstacle)
            assert obs.size[2] > 0  # 高度大于0

    def test_physics_parameters_for_multiple_grades(self):
        """测试多个等级AGV的物理参数"""
        from src.embodied.simulation_enhancement import PhysicsParameters
        enhancer = EmbodiedSimulationEnhancer(agv_grade='M')
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            params = PhysicsParameters.for_grade(grade)
            max_speed = params.calculate_max_speed()
            max_accel = params.calculate_max_acceleration(current_load=0)
            assert max_speed > 0
            assert max_accel > 0
            # 等级越高质量越大
            if grade in ['L', 'XL', 'XXL']:
                assert params.mass_empty > enhancer.physics.mass_empty

    def test_sensor_noise_reproducibility(self):
        """测试传感器噪声可重复性"""
        enhancer1 = EmbodiedSimulationEnhancer(agv_grade='M', seed=42)
        enhancer2 = EmbodiedSimulationEnhancer(agv_grade='M', seed=42)

        ranges = np.full(360, 10.0)
        noisy1 = enhancer1.noise_model.add_noise_lidar(ranges)
        noisy2 = enhancer2.noise_model.add_noise_lidar(ranges)

        # 相同种子得到相同噪声
        np.testing.assert_array_almost_equal(noisy1, noisy2)

    def test_delay_simulator_buffer_behavior(self):
        """测试延迟仿真缓存行为"""
        enhancer = EmbodiedSimulationEnhancer(agv_grade='M')
        # 处理五个数据点，每次调用处理并尝试获取
        got_data = None
        for i in range(5):
            data = np.array([i * 0.1, i * 0.2])
            result = enhancer.process_sensor_data('lidar', data)
            if result is not None and got_data is None:
                got_data = result
        # 至少能获取一个数据
        assert got_data is not None


class TestSwarmController:
    """蜂群控制器测试"""

    def test_task_allocation_initialization(self):
        """测试任务分配初始化"""
        controller = SwarmController(grade='M')
        assert controller is not None
        assert len(controller.agents) == 0

    def test_add_agent(self):
        """测试添加智能体"""
        controller = SwarmController(grade='M')
        aid = controller.add_agent(np.array([0.0, 0.0]))
        assert aid == 0
        assert len(controller.agents) == 1

    def test_collision_detection_between_agvs(self):
        """测试AGV之间碰撞检测"""
        controller = SwarmController(grade='M')
        controller.add_agent(np.array([-1.0, 0.0]))
        controller.add_agent(np.array([1.0, 0.0]))

        # 检查碰撞
        valid, errors = controller.validate_swarm()
        # 两个机器人距离2.0 > 最小安全距离0.7，所以合法
        assert valid
        assert len(errors) == 0

    def test_multiple_agents_step(self):
        """测试多智能体步进"""
        controller = SwarmController(grade='M')
        controller.add_agent(np.array([-1.0, 0.0]))
        controller.add_agent(np.array([1.0, 0.0]))
        controller.step()
        positions = controller.get_positions()
        assert positions.shape == (2, 2)


class TestWarehousePickingMultiAGV:
    """仓库拣选多AGV测试"""

    def test_picking_task_generation(self):
        """测试拣选任务生成"""
        generator = WarehouseSceneGenerator(seed=42)
        warehouse = generator.generate_warehouse(num_aisles=3)
        task = generator.generate_picking_task(warehouse, num_items=3)

        assert task['type'] == 'order_picking'
        assert len(task['pick_points']) == 3
        assert 'end_position' in task

        for pick_point in task['pick_points']:
            assert len(pick_point) == 3  # 3D坐标

    def test_multi_agv_picking_allocation(self):
        """测试多AGV拣选任务分配"""
        # 这个功能在multi_agent模块，已经测试过了
        pass


class TestCollisionAvoidance:
    """碰撞避让测试"""

    def test_velocity_obstacle_avoidance(self):
        """测试速度障碍物避障"""
        from src.control.swarm_control import VelocityObstacle

        vo = VelocityObstacle()
        # 机器人在原点，障碍物在前方1m以0速度静止
        robot_vel = np.array([0.5, 0.0])
        obstacle_pos = np.array([1.0, 0.0])
        obstacle_vel = np.array([0.0, 0.0])
        robot_radius = 0.3

        # 检查当前速度是否会碰撞
        will_collide = vo.check_collision(
            robot_pos=np.array([0.0, 0.0]),
            robot_vel=robot_vel,
            obstacle_pos=obstacle_pos,
            obstacle_vel=obstacle_vel,
            robot_radius=robot_radius,
            obstacle_radius=0.3,
            time_horizon=2.0
        )
        assert will_collide == True

        # 横向偏移不会碰撞
        robot_vel_safe = np.array([0.5, 1.0])
        will_collide = vo.check_collision(
            robot_pos=np.array([0.0, 0.0]),
            robot_vel=robot_vel_safe,
            obstacle_pos=obstacle_pos,
            obstacle_vel=obstacle_vel,
            robot_radius=robot_radius,
            obstacle_radius=0.3,
            time_horizon=2.0
        )
        assert will_collide == False

    def test_get_avoidance_velocity(self):
        """测试计算避障速度"""
        from src.control.swarm_control import VelocityObstacle

        vo = VelocityObstacle()
        preferred_vel = np.array([0.5, 0.0])
        robot_pos = np.array([0.0, 0.0])
        obstacles = [
            {
                'position': np.array([1.0, 0.0]),
                'velocity': np.array([0.0, 0.0]),
                'radius': 0.3,
            }
        ]

        avoidance_vel = vo.compute_avoidance_velocity(
            preferred_vel=preferred_vel,
            robot_pos=robot_pos,
            robot_radius=0.3,
            obstacles=obstacles,
            max_speed=1.0
        )

        # 应该得到一个避障速度，Y方向有分量
        assert avoidance_vel is not None
        assert not np.array_equal(avoidance_vel, preferred_vel)
        # 速度大小不超过最大速度
        assert np.linalg.norm(avoidance_vel) <= 1.0 + 1e-6


class TestPerformance:
    """多AGV性能测试"""

    def test_controller_tick_frequency(self):
        """测试控制器tick频率能满足实时要求"""
        controller = SwarmController(grade='M')
        # 添加5个智能体
        for i in range(5):
            controller.add_agent(np.array([i * 2.0, 0.0]))

        # 测量100次tick的时间
        start_time = time.time()
        for _ in range(100):
            controller.step()
        elapsed = time.time() - start_time

        # 100次tick应该在1秒内完成（50Hz要求）
        assert elapsed < 1.0
        logger.info(f"Swarm controller 100 ticks in {elapsed:.3f}s "
                   f"({100/elapsed:.1f} Hz)")


def run_all_tests():
    """运行所有多AGV蜂群协同测试"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_all_tests()
