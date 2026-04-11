#!/usr/bin/env python3
"""
test_agv_swarm_coordinator.py - 多AGV蜂群协调器测试
2026-04-11
"""

import sys
import os
import time
import pytest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.embodied.agv_swarm_coordinator import (
    AGVSwarmCoordinator, SwarmTask, TaskPriority, TaskStatus, AGVSwarmMember
)
from src.control.agv_kinematics import AGVSpec, AGVState
from src.simulation.embodied_sim import WarehouseScene

class TestAGVSwarmCoordinator:
    @pytest.fixture
    def mock_scene(self):
        """创建模拟仓库场景"""
        scene = WarehouseScene()
        # 添加导航点
        scene.navigation_points = {
            'N0': (0, 0, 0),
            'N1': (2, 0, 0),
            'N2': (4, 0, 0),
            'N3': (6, 0, 0),
            'N4': (8, 0, 0),
            'N5': (10, 0, 0),
            'N6': (0, 2, 0),
            'N7': (2, 2, 0),
            'N8': (4, 2, 0),
            'N9': (6, 2, 0),
            'N10': (8, 2, 0),
            'N11': (10, 2, 0)
        }
        # 添加路径段
        for i in range(5):
            scene.path_segments[(f'N{i}', f'N{i+1}')] = 2.0
            scene.path_segments[(f'N{i+6}', f'N{i+7}')] = 2.0
        # 横连接
        for i in range(6):
            scene.path_segments[(f'N{i}', f'N{i+6}')] = 2.0
        # 添加资源
        scene.resources = {
            'loading_dock_1': {'position': (0, 1, 0)},
            'loading_dock_2': {'position': (10, 1, 0)},
            'charging_station_1': {'position': (5, 0, 0)}
        }
        return scene
    
    @pytest.fixture
    def coordinator(self, mock_scene):
        """创建协调器实例"""
        return AGVSwarmCoordinator(mock_scene)
    
    @pytest.fixture
    def agv_spec_m(self):
        """M级AGV规格"""
        return AGVSpec(
            size_class='M',
            max_payload=100,
            max_speed=2.0,
            battery_capacity=1000,
            power_consumption_rate=0.5
        )
    
    @pytest.fixture
    def agv_spec_l(self):
        """L级AGV规格"""
        return AGVSpec(
            size_class='L',
            max_payload=300,
            max_speed=1.5,
            battery_capacity=2000,
            power_consumption_rate=0.8
        )
    
    def test_register_unregister_agv(self, coordinator, agv_spec_m):
        """测试AGV注册注销"""
        assert len(coordinator.agvs) == 0
        
        # 注册AGV
        state = AGVState(pose=(1, 1, 0), speed=0.0)
        coordinator.register_agv("AGV_001", agv_spec_m, state)
        
        assert len(coordinator.agvs) == 1
        assert "AGV_001" in coordinator.agvs
        assert coordinator.agvs["AGV_001"].spec.size_class == "M"
        
        # 注销AGV
        coordinator.unregister_agv("AGV_001")
        assert len(coordinator.agvs) == 0
        assert "AGV_001" not in coordinator.agvs
    
    def test_add_cancel_task(self, coordinator):
        """测试任务添加取消"""
        assert len(coordinator.tasks) == 0
        
        # 添加任务
        task = SwarmTask(
            task_type="transport",
            priority=TaskPriority.P1_HIGH,
            source_point=(0, 0, 0),
            target_point=(10, 2, 0),
            payload=80,
            required_agv_spec="M"
        )
        task_id = coordinator.add_task(task)
        
        assert len(coordinator.tasks) == 1
        assert task_id in coordinator.tasks
        assert coordinator.tasks[task_id].status == TaskStatus.PENDING
        
        # 取消任务
        result = coordinator.cancel_task(task_id)
        assert result is True
        assert coordinator.tasks[task_id].status == TaskStatus.CANCELLED
    
    def test_calculate_shortest_path(self, coordinator):
        """测试最短路径计算"""
        start = (0, 0, 0)
        end = (10, 2, 0)
        
        length = coordinator.calculate_shortest_path_length(start, end)
        # 最优路径: N0 → N1 → N2 → N3 → N4 → N5 → N11，总长度 2*6=12
        assert abs(length - 12.0) < 0.1
    
    def test_task_allocation_score(self, coordinator, agv_spec_m):
        """测试任务分配得分计算"""
        # 注册AGV
        state = AGVState(pose=(0, 0, 0), speed=0.0)
        coordinator.register_agv("AGV_001", agv_spec_m, state)
        agv = coordinator.agvs["AGV_001"]
        
        # 合适的任务
        task1 = SwarmTask(
            source_point=(0, 0, 0),
            target_point=(2, 0, 0),
            payload=50,
            required_agv_spec="M",
            deadline=3600
        )
        score1 = coordinator.calculate_task_allocation_score(agv, task1)
        assert score1 < float('inf')
        
        # 载重不足的任务
        task2 = SwarmTask(
            source_point=(0, 0, 0),
            target_point=(2, 0, 0),
            payload=150,
            required_agv_spec="M",
            deadline=3600
        )
        score2 = coordinator.calculate_task_allocation_score(agv, task2)
        assert score2 == float('inf')
        
        # 规格不足的任务
        task3 = SwarmTask(
            source_point=(0, 0, 0),
            target_point=(2, 0, 0),
            payload=50,
            required_agv_spec="L",
            deadline=3600
        )
        score3 = coordinator.calculate_task_allocation_score(agv, task3)
        assert score3 == float('inf')
    
    def test_task_allocation(self, coordinator, agv_spec_m, agv_spec_l):
        """测试任务分配"""
        # 注册2台AGV
        coordinator.register_agv("AGV_M", agv_spec_m, AGVState(pose=(0, 0, 0), speed=0.0))
        coordinator.register_agv("AGV_L", agv_spec_l, AGVState(pose=(10, 0, 0), speed=0.0))
        
        # 添加2个任务
        task1 = SwarmTask(
            source_point=(0, 0, 0),
            target_point=(10, 0, 0),
            payload=80,
            required_agv_spec="M",
            priority=TaskPriority.P0_URGENT
        )
        task2 = SwarmTask(
            source_point=(10, 0, 0),
            target_point=(0, 0, 0),
            payload=250,
            required_agv_spec="L",
            priority=TaskPriority.P1_HIGH
        )
        coordinator.add_task(task1)
        coordinator.add_task(task2)
        
        # 执行分配
        assigned = coordinator.allocate_tasks()
        assert assigned == 2
        
        # 检查分配结果
        assert task1.assigned_agv_id == "AGV_M"
        assert task2.assigned_agv_id == "AGV_L"
        assert task1.status == TaskStatus.ASSIGNED
        assert task2.status == TaskStatus.ASSIGNED
    
    def test_collision_detection(self, coordinator, agv_spec_m):
        """测试碰撞检测"""
        # 注册两台距离很近的AGV
        coordinator.register_agv("AGV_1", agv_spec_m, AGVState(pose=(0, 0, 0), speed=1.0))
        coordinator.register_agv("AGV_2", agv_spec_m, AGVState(pose=(0.3, 0, 0), speed=1.0))
        
        # 检测冲突
        conflicts = coordinator.detect_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "collision"
        assert set(conflicts[0].involved_agvs) == {"AGV_1", "AGV_2"}
        assert conflicts[0].severity == 8  # 速度>0.5，严重级别高
    
    def test_collision_resolution(self, coordinator, agv_spec_m):
        """测试碰撞解决"""
        # 注册两台AGV
        coordinator.register_agv("AGV_1", agv_spec_m, AGVState(pose=(0, 0, 0), speed=1.0))
        coordinator.register_agv("AGV_2", agv_spec_m, AGVState(pose=(0.3, 0, 0), speed=1.0))
        
        # 添加任务，AGV1优先级更高
        task1 = SwarmTask(priority=TaskPriority.P0_URGENT)
        task2 = SwarmTask(priority=TaskPriority.P2_MEDIUM)
        coordinator.agvs["AGV_1"].current_task = task1
        coordinator.agvs["AGV_2"].current_task = task2
        
        # 检测并解决冲突
        conflicts = coordinator.detect_conflicts()
        resolved = coordinator.resolve_conflicts(conflicts)
        assert resolved == 1
        assert conflicts[0].resolved is True
        # AGV2应该停车让行
        assert coordinator.agvs["AGV_2"].current_state.target_speed == 0.0
        assert coordinator.agvs["AGV_2"].current_state.waiting_for == ["AGV_1"]
    
    def test_swarm_simulation(self, coordinator, agv_spec_m):
        """测试完整蜂群模拟"""
        # 注册4台AGV
        for i in range(4):
            state = AGVState(pose=(i*2, 0, 0), speed=0.0)
            coordinator.register_agv(f"AGV_{i}", agv_spec_m, state)
        
        # 添加10个任务
        for i in range(10):
            task = SwarmTask(
                source_point=(i%5 * 2, 0, 0),
                target_point=(10 - (i%5 * 2), 2, 0),
                payload=50 + i*5,
                priority=TaskPriority(i%4),
                deadline=1800
            )
            coordinator.add_task(task)
        
        # 运行模拟500步（50秒）
        completed_before = coordinator.swarm_metrics['tasks_completed']
        for _ in range(500):
            coordinator.step(0.1)
        
        # 检查结果
        status = coordinator.get_swarm_status()
        assert status['active_agvs'] == 4
        assert status['completed_tasks'] > completed_before
        assert status['total_distance'] > 0.0
        logger.info(f"模拟结果: 完成任务 {status['completed_tasks']}, 总行驶距离 {status['total_distance']:.1f}m, 冲突 {status['conflict_count']}次")
    
    def test_deadlock_detection_resolution(self, coordinator, agv_spec_m):
        """测试死锁检测与解决"""
        # 注册4台AGV形成环形等待
        agv1 = AGVSwarmMember(agv_id="AGV_1", spec=agv_spec_m, current_state=AGVState(pose=(0,0,0)))
        agv2 = AGVSwarmMember(agv_id="AGV_2", spec=agv_spec_m, current_state=AGVState(pose=(2,0,0)))
        agv3 = AGVSwarmMember(agv_id="AGV_3", spec=agv_spec_m, current_state=AGVState(pose=(2,2,0)))
        agv4 = AGVSwarmMember(agv_id="AGV_4", spec=agv_spec_m, current_state=AGVState(pose=(0,2,0)))
        
        # 设置等待关系形成环: 1→2→3→4→1
        agv1.current_state.waiting_for = ["AGV_2"]
        agv2.current_state.waiting_for = ["AGV_3"]
        agv3.current_state.waiting_for = ["AGV_4"]
        agv4.current_state.waiting_for = ["AGV_1"]
        
        coordinator.agvs = {a.agv_id: a for a in [agv1, agv2, agv3, agv4]}
        coordinator._update_kd_tree()
        
        # 检测死锁
        conflicts = coordinator.detect_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "path_deadlock"
        assert set(conflicts[0].involved_agvs) == {"AGV_1", "AGV_2", "AGV_3", "AGV_4"}
        
        # 解决死锁
        resolved = coordinator.resolve_conflicts(conflicts)
        assert resolved == 1
        assert conflicts[0].resolved is True
        # 应该有一台AGV被重新规划路径
        agvs_with_path = [a for a in coordinator.agvs.values() if a.current_state.path is not None]
        assert len(agvs_with_path) >= 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
