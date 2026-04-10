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
        simulator.initialize()
        yield simulator
        simulator.shutdown()

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

        for i, pos in enumerate(positions):
            urdf = generate_agv_urdf_detailed('M', '2轮')
            urdf_paths.append(urdf)
            agv_id = sim.load_urdf(urdf, basePosition=pos)
            agv_ids.append(agv_id)

        assert len(agv_ids) == 3
        for aid in agv_ids:
            assert aid >= 0

        # 运行几步仿真确认稳定
        for _ in range(100):
            sim.step()

        # 检查都还在世界中
        for aid in agv_ids:
            pos, _ = sim.get_base_transform(aid)
            # Z坐标应该接近地面（有重力）
            assert pos[2] > 0.1
            assert pos[2] < 0.5

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
        enhancer = EmbodiedSimulationEnhancer(agv_grade='M')
        for grade in ['S', 'M', 'L', 'XL', 'XXL']:
            params = enhancer.physics.__class__.for_grade(grade)
            max_speed = params.calculate_max_speed()
            max_accel = params.calculate_max_accel(0)
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
        # 添加几个数据点
        for i in range(5):
            data = np.array([i * 0.1, i * 0.2])
            enhancer.process_sensor_data('lidar', data)

        # 能取出数据
        first_data = enhancer.delay_simulator.get_delayed_data('lidar')
        assert first_data is not None


class TestSwarmCoordinator:
    """蜂群协调器测试"""

    def test_task_allocation_initialization(self):
        """测试任务分配初始化"""
        coordinator = SwarmCoordinator(num_robots=3)
        coordinator.initialize()
        assert coordinator.num_robots == 3
        assert len(coordinator.robots) == 3

    def test_simple_task_allocation(self):
        """测试简单任务分配"""
        coordinator = SwarmCoordinator(num_robots=2)
        coordinator.initialize()

        # 添加两个导航任务
        tasks = [
            {'id': 't1', 'type': 'navigate', 'position': np.array([10.0, 0.0, 0.0]), 'priority': 0},
            {'id': 't2', 'type': 'navigate', 'position': np.array([-10.0, 0.0, 0.0]), 'priority': 0},
        ]

        allocation = coordinator.allocate_tasks(tasks)
        assert isinstance(allocation, TaskAllocation)
        assert len(allocation.assignments) == min(2, len(tasks))

    def test_collision_detection_between_agvs(self):
        """测试AGV之间碰撞检测"""
        coordinator = SwarmCoordinator(num_robots=2)
        coordinator.initialize()

        # 设置AGV位置很近
        coordinator.set_robot_position(0, np.array([0.0, 0.0, 0.15]))
        coordinator.set_robot_position(1, np.array([0.2, 0.0, 0.15]))

        # AGV半径0.3，距离0.2 < 0.6 → 碰撞
        collision = coordinator.check_robot_collision(0, 1, robot_radius=0.3)
        assert collision is True

        # 距离拉开到1.0 → 无碰撞
        coordinator.set_robot_position(1, np.array([1.0, 0.0, 0.15]))
        collision = coordinator.check_robot_collision(0, 1, robot_radius=0.3)
        assert collision is False

    def test_path_conflict_detection(self):
        """测试路径冲突检测"""
        coordinator = SwarmCoordinator(num_robots=2)
        coordinator.initialize()

        # 机器人1从左到右，机器人2从右到左，路径交叉
        path1 = [np.array([-5.0, 0.0]), np.array([0.0, 0.0]), np.array([5.0, 0.0])]
        path2 = [np.array([5.0, 0.0]), np.array([0.0, 0.0]), np.array([-5.0, 0.0])]

        conflict = coordinator.detect_path_conflict(path1, path2, conflict_threshold=0.5)
        assert conflict is True

        # 平行路径不相交
        path3 = [np.array([-5.0, 2.0]), np.array([5.0, 2.0])]
        conflict = coordinator.detect_path_conflict(path1, path3, conflict_threshold=0.5)
        # 距离2.0 > 0.5 → 无冲突
        assert conflict is False

    def test_conflict_resolution_priority_based(self):
        """测试基于优先级的冲突解决"""
        coordinator = SwarmCoordinator(num_robots=2)
        coordinator.initialize()

        # 高优先级任务和低优先级任务冲突
        tasks = [
            {'id': 'high_priority', 'priority': 0, 'path': [np.array([0, 0]), np.array([10, 0])]},
            {'id': 'low_priority', 'priority': 5, 'path': [np.array([10, 0]), np.array([0, 0])]},
        ]

        resolution = coordinator.resolve_conflict_priority(tasks)
        # 高优先级优先通行
        assert resolution['yield_robot_id'] == 1
        assert resolution['priority_robot_id'] == 0


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
        generator = WarehouseSceneGenerator(seed=42)
        warehouse = generator.generate_warehouse(num_aisles=3)
        task = generator.generate_picking_task(warehouse, num_items=3)

        coordinator = SwarmCoordinator(num_robots=2)
        coordinator.initialize()

        # 将拣选点分配给不同AGV
        allocations = coordinator.allocate_picking_tasks(
            task['pick_points'],
            start_positions=warehouse['start_positions']
        )

        assert len(allocations) > 0
        # 所有拣选点都应该被分配
        assigned_count = sum(len(a['pick_points']) for a in allocations)
        assert assigned_count == len(task['pick_points'])


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
        assert will_collide is True

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
        assert will_collide is False

    def test_get_avoidance_velocity(self):
        """测试计算避障速度"""
        from src.control.swarm_coordinator import VelocityObstacle

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

    def test_coordinator_tick_frequency(self):
        """测试协调器tick频率能满足实时要求"""
        coordinator = SwarmCoordinator(num_robots=5)
        coordinator.initialize()

        # 添加一些任务
        for i in range(10):
            coordinator.add_task({
                'id': f'task_{i}',
                'type': 'navigate',
                'position': np.array([i * 2.0, 0.0, 0.0]),
                'priority': i % 3,
            })

        # 测量100次tick的时间
        start_time = time.time()
        for _ in range(100):
            coordinator.tick()
        elapsed = time.time() - start_time

        # 100次tick应该在1秒内完成（50Hz要求）
        assert elapsed < 1.0
        logger.info(f"Swarm coordinator 100 ticks in {elapsed:.3f}s "
                   f"({100/elapsed:.1f} Hz)")


def run_all_tests():
    """运行所有多AGV蜂群协同测试"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_all_tests()
