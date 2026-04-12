"""
Test cases for Embodiment Module - 具身智能模块测试用例
包含仿真环境、AGV接口、行为树引擎、多AGV协调器的单元测试和集成测试
"""

import pytest
import time
import math
from typing import Tuple

from embodiment.simulation import EmbodimentSimulator, SimAGVConfig, SimSceneConfig
from embodiment.agv_interface import AGVHardwareInterface, AGVConfig, AGVCommand, AGVCommunicationType
from embodiment.behavior_tree_engine import BehaviorTreeEngine
from embodiment.multi_agv_coordinator import MultiAGVCoordinator, AGVTask, AGVStatus
from control.planner import BehaviorTreeBuilder, NodeStatus


class TestEmbodimentSimulator:
    """仿真环境测试"""

    def test_simulator_initialization(self):
        """测试仿真环境初始化"""
        scene_config = SimSceneConfig(
            obstacles=[(2.0, 0.0, 0.3), (0.0, 2.0, 0.3)],
            charging_stations=[(-1.0, -1.0)]
        )
        sim = EmbodimentSimulator(scene_config=scene_config, gui=False)
        assert sim.client_id >= 0
        assert sim.current_time == 0.0
        sim.close()

    def test_add_agv(self):
        """测试添加AGV到仿真环境"""
        sim = EmbodimentSimulator(gui=False)
        config = SimAGVConfig(start_position=(1.0, 1.0, 0.1))
        agv_id = sim.add_agv(config)
        assert agv_id == 0
        assert agv_id in sim.agvs
        assert sim.agvs[agv_id]["config"].start_position == (1.0, 1.0, 0.1)
        sim.close()

    def test_agv_movement(self):
        """测试AGV运动控制"""
        sim = EmbodimentSimulator(gui=False)
        agv_id = sim.add_agv()
        # 发送前进指令
        sim.set_agv_command(agv_id, v=0.5, omega=0.0)
        # 运行1秒
        for _ in range(100):
            states = sim.step()
        # 检查AGV是否前进了约0.5米
        final_x = states[agv_id]["state"]["x"]
        assert final_x > 0.3 and final_x < 0.7
        sim.close()

    def test_sensor_readings(self):
        """测试传感器数据读取"""
        sim = EmbodimentSimulator(gui=False)
        agv_id = sim.add_agv()
        states = sim.step()
        assert "sensors" in states[agv_id]
        assert "imu" in states[agv_id]["sensors"]
        assert "force_torque" in states[agv_id]["sensors"]
        assert "tactile" in states[agv_id]["sensors"]
        assert "obstacles" in states[agv_id]["state"]
        sim.close()


class TestBehaviorTreeEngine:
    """行为树引擎测试"""

    def test_engine_initialization(self):
        """测试引擎初始化"""
        # 创建简单行为树：移动到目标点
        bt = BehaviorTreeBuilder.create_warehouse_transfer_task(
            pick_location=(1.0, 0.0),
            place_location=(2.0, 0.0),
            charge_station=(-1.0, 0.0)
        )
        engine = BehaviorTreeEngine(bt, update_rate=100.0)
        assert not engine.is_running()
        assert engine.get_blackboard_value("current_x") == 0.0

    def test_bt_execution(self):
        """测试行为树执行"""
        # 创建移动到目标点的行为树
        from control.planner import Sequence, MoveTo, IsAtTarget
        target = (1.0, 0.0)
        bt = Sequence([
            MoveTo(target),
            IsAtTarget(target)
        ])
        engine = BehaviorTreeEngine(bt)

        # 设置当前AGV位置在原点
        engine.set_state(x=0.0, y=0.0, theta=0.0, current_time=0.0)

        # 执行tick，状态应该是running
        status = engine.tick()
        assert status == NodeStatus.RUNNING
        v, omega, _ = engine.get_control_output()
        assert v > 0.0  # 应该前进

        # 模拟到达目标点
        engine.set_state(x=1.0, y=0.0, theta=0.0, current_time=1.0)
        status = engine.tick()
        assert status == NodeStatus.SUCCESS

    def test_engine_background_run(self):
        """测试后台运行模式"""
        bt = BehaviorTreeBuilder.create_patrol_task(patrol_points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        engine = BehaviorTreeEngine(bt)
        engine.start(background=True)
        assert engine.is_running()
        time.sleep(0.2)
        engine.stop()
        assert not engine.is_running()
        stats = engine.get_stats()
        assert stats["total_ticks"] > 0


class TestMultiAGVCoordinator:
    """多AGV协调器测试"""

    def test_coordinator_initialization(self):
        """测试协调器初始化"""
        bounds = (-5.0, 5.0, -5.0, 5.0)
        coordinator = MultiAGVCoordinator(bounds)
        assert coordinator.bounds == bounds
        assert len(coordinator.agvs) == 0
        assert len(coordinator.tasks) == 0

    def test_task_assignment(self):
        """测试任务分配"""
        bounds = (-5.0, 5.0, -5.0, 5.0)
        coordinator = MultiAGVCoordinator(bounds)
        # 添加2个AGV
        coordinator.add_agv(0, start_position=(0.0, 0.0))
        coordinator.add_agv(1, start_position=(3.0, 0.0))
        # 添加搬运任务，起点在(2.0, 0.0)
        task = AGVTask(
            task_id="task_001",
            task_type="transfer",
            priority=8,
            pick_location=(2.0, 0.0),
            place_location=(4.0, 0.0)
        )
        coordinator.add_task(task)
        # 执行分配
        assignments = coordinator.assign_tasks()
        assert len(assignments) == 1
        assert assignments[0].success
        # 距离更近的AGV 1应该获得任务
        assert assignments[0].agv_id == 1

    def test_path_planning(self):
        """测试路径规划"""
        bounds = (-5.0, 5.0, -5.0, 5.0)
        coordinator = MultiAGVCoordinator(bounds)
        coordinator.add_agv(0, start_position=(0.0, 0.0))
        # 添加障碍物
        coordinator.update_global_obstacles([(1.0, 0.0, 0.3)])
        # 规划到(2.0, 0.0)的路径
        path = coordinator.plan_agv_path(0, (2.0, 0.0))
        assert path is not None
        assert len(path) >= 2
        # 路径应该绕开障碍物
        for point in path:
            dist = math.hypot(point.x - 1.0, point.y - 0.0)
            assert dist > 0.3  # 大于障碍物半径

    def test_conflict_detection(self):
        """测试冲突检测"""
        bounds = (-5.0, 5.0, -5.0, 5.0)
        coordinator = MultiAGVCoordinator(bounds, agv_safety_distance=1.0)
        coordinator.add_agv(0, start_position=(0.0, 0.0))
        coordinator.add_agv(1, start_position=(0.5, 0.0))  # 距离小于安全距离
        conflicts = coordinator.check_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0][2] == "collision"


@pytest.mark.integration
class TestEmbodimentIntegration:
    """具身模块集成测试"""

    def test_sim_bt_integration(self):
        """仿真环境 + 行为树引擎集成测试"""
        # 初始化仿真
        sim = EmbodimentSimulator(gui=False)
        agv_id = sim.add_agv()
        # 创建行为树：移动到(1.0, 0.0)
        from control.planner import Sequence, MoveTo, IsAtTarget
        target = (1.0, 0.0)
        bt = Sequence([
            MoveTo(target),
            IsAtTarget(target)
        ])
        engine = BehaviorTreeEngine(bt)

        # 运行仿真，执行行为树
        success = False
        for i in range(200):  # 最多2秒
            states = sim.step()
            agv_state = states[agv_id]["state"]
            # 更新状态到行为树
            engine.set_state(
                x=agv_state["x"],
                y=agv_state["y"],
                theta=agv_state["theta"],
                battery_level=agv_state["battery_level"],
                obstacles=agv_state["obstacles"],
                current_time=states[agv_id]["current_time"]
            )
            # 执行tick
            status = engine.tick()
            # 获取控制指令
            v, omega, gripper = engine.get_control_output()
            # 发送到仿真
            sim.set_agv_command(agv_id, v, omega)
            sim.set_gripper_command(agv_id, gripper)

            if status == NodeStatus.SUCCESS:
                success = True
                break

        assert success
        # 验证到达目标点
        final_state = sim.step()[agv_id]["state"]
        dist = math.hypot(final_state["x"] - target[0], final_state["y"] - target[1])
        assert dist < 0.1
        sim.close()

    def test_multi_agv_simulation(self):
        """多AGV仿真协同测试"""
        scene_config = SimSceneConfig(
            obstacles=[(1.0, 1.0, 0.3)],
            charging_stations=[(-1.0, -1.0)]
        )
        sim = EmbodimentSimulator(scene_config=scene_config, gui=False)
        # 添加2个AGV
        agv1_id = sim.add_agv(SimAGVConfig(start_position=(0.0, 0.0, 0.1)))
        agv2_id = sim.add_agv(SimAGVConfig(start_position=(0.0, 2.0, 0.1)))

        # 初始化协调器
        bounds = (-5.0, 5.0, -5.0, 5.0)
        coordinator = MultiAGVCoordinator(bounds)
        coordinator.add_agv(agv1_id, start_position=(0.0, 0.0))
        coordinator.add_agv(agv2_id, start_position=(0.0, 2.0))

        # 添加2个任务
        task1 = AGVTask(
            task_id="task1",
            task_type="transfer",
            priority=8,
            pick_location=(2.0, 0.0),
            place_location=(3.0, 0.0)
        )
        task2 = AGVTask(
            task_id="task2",
            task_type="transfer",
            priority=7,
            pick_location=(2.0, 2.0),
            place_location=(3.0, 2.0)
        )
        coordinator.add_task(task1)
        coordinator.add_task(task2)

        # 分配任务
        assignments = coordinator.assign_tasks()
        assert len(assignments) == 2
        assert all(a.success for a in assignments)

        sim.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
