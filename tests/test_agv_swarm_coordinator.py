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
import logging
logger = logging.getLogger(__name__)
import tempfile
import json
from io import StringIO

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.embodied.agv_swarm_coordinator import (
    AGVSwarmCoordinator, SwarmTask, TaskPriority, TaskStatus, AGVSwarmMember
)
from src.control.agv import AGVSpec, AGVGrade
# 模拟AGVState，包含协调器需要的属性
class MockAGVState:
    def __init__(self, pose=(0,0,0), speed=0.0):
        self.pose = pose
        self.speed = speed
        self.waiting_for = []
        self.occupying_resource = None
        self.target_speed = speed
        self.path = []

AGVState = MockAGVState
from src.simulation.embodied_sim import WarehouseScene

class TestAGVSwarmCoordinator:
    @pytest.fixture
    def mock_scene(self):
        """创建模拟仓库场景"""
        # 创建模拟场景（不需要真实env）
        class MockWarehouseScene:
            def __init__(self):
                self.path_segments = {}
                self.navigation_points = {}
        scene = MockWarehouseScene()
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
        spec = AGVSpec.from_grade(AGVGrade.M)
        # 补充测试需要的额外属性
        spec.size_class = 'M'
        spec.max_payload = 100
        spec.max_speed = spec.max_linear_speed
        spec.battery_capacity = 1000
        spec.power_consumption_rate = 0.5
        return spec
    
    @pytest.fixture
    def agv_spec_l(self):
        """L级AGV规格"""
        spec = AGVSpec.from_grade(AGVGrade.L)
        # 补充测试需要的额外属性
        spec.size_class = 'L'
        spec.max_payload = 300
        spec.max_speed = spec.max_linear_speed
        spec.battery_capacity = 2000
        spec.power_consumption_rate = 0.8
        return spec
    
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
            state = AGVState(pose=(i*2, 0, 0), speed=1.0)
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
        in_progress_before = len([t for t in coordinator.tasks.values() if t.status == TaskStatus.IN_PROGRESS])
        for _ in range(500):
            coordinator.step(0.1)
        
        # 检查结果
        status = coordinator.get_swarm_status()
        assert status['active_agvs'] == 4
        # 任务要么完成要么在执行中
        assert (status['completed_tasks'] > completed_before) or (len([t for t in coordinator.tasks.values() if t.status == TaskStatus.IN_PROGRESS]) > in_progress_before)
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
        # 添加当前任务以便死锁解决可以重新规划路径
        agv1.current_task = SwarmTask(priority=TaskPriority.P0_URGENT, target_point=(10,0,0))
        agv2.current_task = SwarmTask(priority=TaskPriority.P1_HIGH, target_point=(10,2,0))
        agv3.current_task = SwarmTask(priority=TaskPriority.P2_MEDIUM, target_point=(0,2,0))
        agv4.current_task = SwarmTask(priority=TaskPriority.P3_LOW, target_point=(0,0,0))
        
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
    
    def test_auction_based_allocation(self, coordinator, agv_spec_m, agv_spec_l):
        """测试基于拍卖的任务分配"""
        # 注册3台AGV
        coordinator.register_agv("AGV_1", agv_spec_m, AGVState(pose=(0, 0, 0), speed=0.0))
        coordinator.register_agv("AGV_2", agv_spec_m, AGVState(pose=(5, 0, 0), speed=0.0))
        coordinator.register_agv("AGV_3", agv_spec_l, AGVState(pose=(10, 0, 0), speed=0.0))
        
        # 添加2个任务，M规格和L规格
        task_m = SwarmTask(
            source_point=(0, 0, 0),
            target_point=(10, 0, 0),
            payload=80,
            required_agv_spec="M",
            priority=TaskPriority.P0_URGENT
        )
        task_l = SwarmTask(
            source_point=(10, 0, 0),
            target_point=(0, 0, 0),
            payload=250,
            required_agv_spec="L",
            priority=TaskPriority.P1_HIGH
        )
        coordinator.add_task(task_m)
        coordinator.add_task(task_l)
        
        # 使用拍卖算法分配
        assigned = coordinator.allocate_tasks(algorithm="auction")
        assert assigned == 2
        
        # 检查分配结果：AGV1离M任务近，拍得M任务；AGV3拍得L任务
        assert task_m.assigned_agv_id == "AGV_1"
        assert task_l.assigned_agv_id == "AGV_3"
        assert task_m.status == TaskStatus.ASSIGNED
        assert task_l.status == TaskStatus.ASSIGNED
    
    def test_hungarian_allocation(self, coordinator, agv_spec_m):
        """测试匈牙利算法全局最优分配"""
        # 注册3台AGV
        for i in range(3):
            coordinator.register_agv(f"AGV_{i}", agv_spec_m, AGVState(pose=(i*3, 0, 0), speed=0.0))
        
        # 添加3个任务
        tasks = []
        for i in range(3):
            task = SwarmTask(
                source_point=( (2-i)*3, 0, 0 ),  # 位置与AGV反向，匈牙利会匹配最优
                target_point=(10, 0, 0),
                payload=50,
                required_agv_spec="M"
            )
            tasks.append(task)
            coordinator.add_task(task)
        
        # 使用匈牙利算法分配
        assigned = coordinator.allocate_tasks(algorithm="hungarian")
        assert assigned == 3
        
        # 检查所有任务都被分配
        for task in tasks:
            assert task.status == TaskStatus.ASSIGNED
            assert task.assigned_agv_id is not None
    
    def test_distributed_collision_avoidance(self, coordinator, agv_spec_m):
        """测试分布式碰撞避免（ORCA）"""
        # 注册两台相向行驶的AGV
        agv1 = AGVSwarmMember(
            agv_id="AGV_1", 
            spec=agv_spec_m, 
            current_state=AGVState(pose=(0, 0, 0), speed=1.0)
        )
        agv2 = AGVSwarmMember(
            agv_id="AGV_2", 
            spec=agv_spec_m, 
            current_state=AGVState(pose=(0.3, 0, 0), speed=1.0)  # 距离0.3m < 0.5m安全距离，相向行驶
        )
        # 添加速度属性
        agv1.current_state.velocity = [1.0, 0, 0]
        agv2.current_state.velocity = [-1.0, 0, 0]
        
        coordinator.agvs = {"AGV_1": agv1, "AGV_2": agv2}
        coordinator._update_kd_tree()
        
        # 检测冲突
        conflicts = coordinator.detect_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "collision"
        
        # 使用分布式模式解决冲突
        resolved = coordinator.resolve_conflicts(conflicts, mode="distributed")
        assert resolved == 1
        assert conflicts[0].resolved is True
        
        # 检查速度已被调整，避免碰撞
        assert agv1.current_state.target_speed < 1.0
        assert agv2.current_state.target_speed < 1.0
    
    def test_monitoring_dashboard_json(self, coordinator, agv_spec_m):
        """测试监控面板JSON格式输出"""
        # 注册2台AGV
        coordinator.register_agv("AGV_1", agv_spec_m, AGVState(pose=(0, 0, 0), speed=0.5))
        coordinator.register_agv("AGV_2", agv_spec_m, AGVState(pose=(2, 0, 0), speed=0.0))
        
        # 添加2个任务
        task1 = SwarmTask(priority=TaskPriority.P1_HIGH, status=TaskStatus.IN_PROGRESS, assigned_agv_id="AGV_1", progress=0.5)
        task2 = SwarmTask(priority=TaskPriority.P2_MEDIUM, status=TaskStatus.PENDING)
        coordinator.agvs["AGV_1"].current_task = task1
        coordinator.tasks["TASK_1"] = task1
        coordinator.tasks["TASK_2"] = task2
        
        # 获取JSON格式面板数据
        data = coordinator.get_monitoring_dashboard_data(format="json")
        
        # 验证数据完整性
        assert "timestamp" in data
        assert "metrics" in data
        assert "agvs" in data
        assert "tasks" in data
        assert "active_conflicts" in data
        assert data["metrics"]["version"] == "v2.88.0"
        assert len(data["agvs"]) == 2
        assert len(data["tasks"]) == 2
        assert data["agvs"][0]["agv_id"] in ["AGV_1", "AGV_2"]
        
        # 验证JSON可序列化
        json_str = json.dumps(data)
        assert len(json_str) > 0
    
    def test_monitoring_dashboard_html(self, coordinator, agv_spec_m):
        """测试监控面板HTML格式输出"""
        # 注册AGV和任务
        coordinator.register_agv("AGV_1", agv_spec_m, AGVState(pose=(0, 0, 0), speed=0.5))
        task = SwarmTask(priority=TaskPriority.P0_URGENT, status=TaskStatus.IN_PROGRESS, assigned_agv_id="AGV_1")
        coordinator.agvs["AGV_1"].current_task = task
        coordinator.tasks["TASK_1"] = task
        
        # 获取HTML格式面板
        data = coordinator.get_monitoring_dashboard_data(format="html")
        assert "html" in data
        html_content = data["html"]
        
        # 验证HTML内容完整性
        assert "<html>" in html_content
        assert "<title>AGV Swarm Monitoring Dashboard v2.88.0</title>" in html_content
        assert "AGV_1" in html_content
        assert "TASK_1" in html_content
        assert "P0_URGENT" in html_content
    
    def test_monitoring_dashboard_csv(self, coordinator, agv_spec_m):
        """测试监控面板CSV格式输出"""
        coordinator.register_agv("AGV_1", agv_spec_m, AGVState(pose=(0, 0, 0), speed=0.5))
        
        data = coordinator.get_monitoring_dashboard_data(format="csv")
        assert "csv" in data
        csv_content = data["csv"]
        
        # 验证CSV格式
        assert "Category,Key,Value" in csv_content
        assert "AGV,AGV_1_agv_id,AGV_1" in csv_content
    
    def test_export_dashboard_to_file(self, coordinator, agv_spec_m):
        """测试导出监控面板到文件"""
        coordinator.register_agv("AGV_1", agv_spec_m, AGVState(pose=(0, 0, 0), speed=0.0))
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_path = f.name
        
        try:
            # 导出HTML
            result = coordinator.export_dashboard(temp_path, format="html")
            assert result is True
            
            # 验证文件内容
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert "<html>" in content
                assert "AGV Swarm Monitoring Dashboard v2.88.0" in content
        finally:
            # 清理临时文件
            os.unlink(temp_path)
    
    def test_allocation_algorithm_comparison(self, coordinator, agv_spec_m):
        """对比三种分配算法的效果"""
        # 注册5台AGV
        for i in range(5):
            coordinator.register_agv(f"AGV_{i}", agv_spec_m, AGVState(pose=(i*2, 0, 0), speed=0.0))
        
        # 添加10个任务
        for i in range(10):
            task = SwarmTask(
                source_point=(i*1, 0, 0),
                target_point=(10 - i*1, 2, 0),
                payload=50,
                deadline=1800
            )
            coordinator.add_task(task)
        
        # 测试三种算法
        for algorithm in ["greedy", "auction", "hungarian"]:
            # 重置任务状态
            for task in coordinator.tasks.values():
                task.status = TaskStatus.PENDING
                task.assigned_agv_id = None
            
            # 分配
            assigned = coordinator.allocate_tasks(algorithm=algorithm)
            assert assigned >= 5  # 至少分配5个任务
            logger.info(f"算法 {algorithm} 分配了 {assigned} 个任务")
    
    def test_resource_contention_detection_resolution(self, coordinator, agv_spec_m):
        """测试资源竞争检测与解决"""
        # 注册2台AGV同时占用同一个资源
        agv1 = AGVSwarmMember(agv_id="AGV_1", spec=agv_spec_m, current_state=AGVState(pose=(0, 1, 0)))
        agv2 = AGVSwarmMember(agv_id="AGV_2", spec=agv_spec_m, current_state=AGVState(pose=(0.6, 1, 0)))  # 距离0.6m > 0.5m，无碰撞
        agv1.current_state.occupying_resource = "loading_dock_1"
        agv2.current_state.occupying_resource = "loading_dock_1"
        
        # AGV1任务优先级更高
        agv1.current_task = SwarmTask(priority=TaskPriority.P0_URGENT)
        agv2.current_task = SwarmTask(priority=TaskPriority.P2_MEDIUM)
        
        coordinator.agvs = {"AGV_1": agv1, "AGV_2": agv2}
        coordinator._update_kd_tree()
        
        # 检测资源竞争
        conflicts = coordinator.detect_conflicts()
        # 只有资源竞争冲突，无碰撞
        resource_conflicts = [c for c in conflicts if c.conflict_type == "resource_contention"]
        assert len(resource_conflicts) == 1
        
        # 解决冲突
        resolved = coordinator.resolve_conflicts(conflicts)
        assert resolved == 1
        assert conflicts[0].resolved is True
        # AGV2应该释放资源，等待AGV1
        assert agv2.current_state.occupying_resource is None
        assert agv2.current_state.waiting_for == ["AGV_1"]
    
    def test_swarm_status_version(self, coordinator):
        """测试版本号正确为v2.88.0"""
        status = coordinator.get_swarm_status()
        assert "version" in status
        assert status["version"] == "v2.88.0"

    def test_task_queue_handling(self, coordinator, agv_spec_m):
        """测试AGV任务队列处理"""
        # 注册1台AGV
        coordinator.register_agv("AGV_1", agv_spec_m, AGVState(pose=(0, 0, 0), speed=0.0))
        agv = coordinator.agvs["AGV_1"]

        # 添加3个任务
        task1 = SwarmTask(task_id="T1", source_point=(0,0,0), target_point=(2,0,0), required_agv_spec="M")
        task2 = SwarmTask(task_id="T2", source_point=(2,0,0), target_point=(4,0,0), required_agv_spec="M")
        task3 = SwarmTask(task_id="T3", source_point=(4,0,0), target_point=(6,0,0), required_agv_spec="M")

        coordinator.add_task(task1)
        coordinator.add_task(task2)
        coordinator.add_task(task3)

        # 分配任务
        assigned = coordinator.allocate_tasks()
        assert assigned == 3

        # 检查任务分配：task1是当前任务，task2和task3在队列
        assert agv.current_task == task1
        assert len(agv.task_queue) == 2
        assert agv.task_queue[0] == task2
        assert agv.task_queue[1] == task3

        # 模拟任务1完成
        task1.progress = 1.0
        coordinator.sync_agv_states()

        # 检查task2成为当前任务，task3在队列
        assert agv.current_task == task2
        assert len(agv.task_queue) == 1
        assert agv.task_queue[0] == task3
        assert task1.status == TaskStatus.COMPLETED

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
