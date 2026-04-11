#!/usr/bin/env python3
"""
多AGV蜂群协调模块测试用例
SuperModel v2.88.0
"""

import pytest
import time
import uuid
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass, field

# 导入待测试模块
from src.embodied.agv_swarm_coordinator import (
    AGVSwarmCoordinator, SwarmTask, AGVSwarmMember, SwarmConflict,
    TaskPriority, TaskStatus
)

# 模拟依赖类
@dataclass
class MockAGVSpec:
    size_class: str = "M"
    max_payload: float = 100.0
    max_speed: float = 2.0
    power_consumption_rate: float = 10.0
    battery_capacity: float = 1000.0

@dataclass
class MockAGVState:
    pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    speed: float = 0.0
    target_speed: float = 0.0
    waiting_for: list = field(default_factory=list)
    occupying_resource: str = None
    path: list = field(default_factory=list)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    battery_level: float = 100.0

class MockWarehouseScene:
    def __init__(self):
        self.navigation_points: Dict[str, Tuple[float, float, float]] = {
            "N0": (0.0, 0.0, 0.0),
            "N1": (2.0, 0.0, 0.0),
            "N2": (4.0, 0.0, 0.0),
            "N3": (6.0, 0.0, 0.0),
            "N4": (8.0, 0.0, 0.0),
            "N5": (10.0, 0.0, 0.0),
            "N6": (0.0, 2.0, 0.0),
            "N7": (2.0, 2.0, 0.0),
            "N8": (4.0, 2.0, 0.0),
            "N9": (6.0, 2.0, 0.0),
            "N10": (8.0, 2.0, 0.0),
            "N11": (10.0, 2.0, 0.0),
        }
        self.path_segments: Dict[Tuple[str, str], float] = {
            ("N0", "N1"): 2.0, ("N1", "N2"): 2.0, ("N2", "N3"): 2.0, ("N3", "N4"): 2.0, ("N4", "N5"): 2.0,
            ("N6", "N7"): 2.0, ("N7", "N8"): 2.0, ("N8", "N9"): 2.0, ("N9", "N10"): 2.0, ("N10", "N11"): 2.0,
            ("N0", "N6"): 2.0, ("N1", "N7"): 2.0, ("N2", "N8"): 2.0, ("N3", "N9"): 2.0, ("N4", "N10"): 2.0, ("N5", "N11"): 2.0,
        }
        self.resources: Dict[str, dict] = {
            "DOCK1": {"position": (0.0, 0.0, 0.0)},
            "DOCK2": {"position": (10.0, 0.0, 0.0)},
            "CHARGE1": {"position": (0.0, 2.0, 0.0)},
        }

@pytest.fixture
def mock_scene():
    return MockWarehouseScene()

@pytest.fixture
def coordinator(mock_scene):
    return AGVSwarmCoordinator(mock_scene, max_workers=5)

@pytest.fixture
def agv_spec_m():
    return MockAGVSpec(size_class='M', max_payload=100, max_speed=2.0)

@pytest.fixture
def agv_spec_l():
    return MockAGVSpec(size_class='L', max_payload=300, max_speed=1.5)

# 测试用例1: 测试AGV注册和注销
def test_agv_registration_unregistration(coordinator, agv_spec_m):
    initial_agv_count = len(coordinator.agvs)
    state = MockAGVState(pose=(0.0, 0.0, 0.0))
    coordinator.register_agv("AGV_001", agv_spec_m, state)
    assert len(coordinator.agvs) == initial_agv_count + 1
    assert "AGV_001" in coordinator.agvs
    
    coordinator.unregister_agv("AGV_001")
    assert len(coordinator.agvs) == initial_agv_count
    assert "AGV_001" not in coordinator.agvs

# 测试用例2: 测试任务添加和取消
def test_task_add_cancel(coordinator):
    initial_task_count = len(coordinator.tasks)
    task = SwarmTask(
        task_type="transport",
        priority=TaskPriority.P2_MEDIUM,
        source_point=(0.0, 0.0, 0.0),
        target_point=(10.0, 0.0, 0.0),
        payload=50.0
    )
    task_id = coordinator.add_task(task)
    assert len(coordinator.tasks) == initial_task_count + 1
    assert task_id in coordinator.tasks
    assert coordinator.tasks[task_id].status == TaskStatus.PENDING
    
    result = coordinator.cancel_task(task_id)
    assert result is True
    assert coordinator.tasks[task_id].status == TaskStatus.CANCELLED

# 测试用例3: 测试任务得分计算
def test_task_allocation_score(coordinator, agv_spec_m):
    state = MockAGVState(pose=(0.0, 0.0, 0.0))
    coordinator.register_agv("AGV_001", agv_spec_m, state)
    agv = coordinator.agvs["AGV_001"]
    agv.battery_level = 100.0
    
    # 匹配任务
    task1 = SwarmTask(source_point=(2.0, 0.0, 0.0), target_point=(10.0, 0.0, 0.0), payload=50.0, required_agv_spec='M', deadline=3600)
    score1 = coordinator.calculate_task_allocation_score(agv, task1)
    assert score1 != float('inf')
    
    # 不匹配任务（载重超出）
    task2 = SwarmTask(payload=150.0, required_agv_spec='M')
    score2 = coordinator.calculate_task_allocation_score(agv, task2)
    assert score2 == float('inf')
    
    # 不匹配任务（规格不足）
    task3 = SwarmTask(required_agv_spec='L')
    score3 = coordinator.calculate_task_allocation_score(agv, task3)
    assert score3 == float('inf')

# 测试用例4: 测试贪心任务分配
def test_greedy_task_allocation(coordinator, agv_spec_m, agv_spec_l):
    # 注册3台AGV
    for i in range(3):
        state = MockAGVState(pose=(i*2.0, 0.0, 0.0))
        coordinator.register_agv(f"AGV_{i}", agv_spec_m if i < 2 else agv_spec_l, state)
    
    # 添加3个任务
    tasks = []
    for i in range(3):
        task = SwarmTask(
            source_point=(i*3.0, 0.0, 0.0),
            target_point=(10.0, 0.0, 0.0),
            payload=50 + i*50,
            priority=TaskPriority(i)
        )
        tasks.append(coordinator.add_task(task))
    
    # 执行贪心分配
    assigned_count = coordinator.allocate_tasks(algorithm="greedy")
    assert assigned_count == 3
    for task_id in tasks:
        assert coordinator.tasks[task_id].status == TaskStatus.ASSIGNED
        assert coordinator.tasks[task_id].assigned_agv_id is not None

# 测试用例5: 测试拍卖任务分配
def test_auction_task_allocation(coordinator, agv_spec_m):
    # 注册2台AGV
    for i in range(2):
        state = MockAGVState(pose=(i*5.0, 0.0, 0.0))
        coordinator.register_agv(f"AGV_{i}", agv_spec_m, state)
    
    # 添加2个任务，一个近AGV0，一个近AGV1
    task1 = SwarmTask(source_point=(1.0, 0.0, 0.0), target_point=(10.0, 0.0, 0.0))
    task2 = SwarmTask(source_point=(6.0, 0.0, 0.0), target_point=(10.0, 0.0, 0.0))
    task1_id = coordinator.add_task(task1)
    task2_id = coordinator.add_task(task2)
    
    # 执行拍卖分配
    assigned_count = coordinator.allocate_tasks(algorithm="auction")
    assert assigned_count == 2
    assert coordinator.tasks[task1_id].assigned_agv_id == "AGV_0"  # 近的AGV中标
    assert coordinator.tasks[task2_id].assigned_agv_id == "AGV_1"

# 测试用例6: 测试匈牙利算法分配
def test_hungarian_task_allocation(coordinator, agv_spec_m):
    # 注册3台AGV
    for i in range(3):
        state = MockAGVState(pose=(i*2.0, 0.0, 0.0))
        coordinator.register_agv(f"AGV_{i}", agv_spec_m, state)
    
    # 添加3个任务
    for i in range(3):
        task = SwarmTask(
            source_point=(i*4.0, 0.0, 0.0),
            target_point=(10.0, 0.0, 0.0),
            payload=50.0
        )
        coordinator.add_task(task)
    
    # 执行匈牙利分配
    assigned_count = coordinator.allocate_tasks(algorithm="hungarian")
    assert assigned_count == 3

# 测试用例7: 测试碰撞检测
def test_collision_detection(coordinator, agv_spec_m):
    # 注册2台距离很近的AGV
    state1 = MockAGVState(pose=(0.0, 0.0, 0.0), speed=0.5)
    state2 = MockAGVState(pose=(0.3, 0.0, 0.0), speed=0.5)  # 距离30cm < 50cm安全距离
    coordinator.register_agv("AGV_0", agv_spec_m, state1)
    coordinator.register_agv("AGV_1", agv_spec_m, state2)
    
    conflicts = coordinator.detect_conflicts()
    assert len(conflicts) >= 1
    assert any(c.conflict_type == "collision" for c in conflicts)
    assert coordinator.swarm_metrics['conflict_count'] >= 1

# 测试用例8: 测试集中式碰撞解决
def test_centralized_collision_resolution(coordinator, agv_spec_m):
    # 注册2台碰撞的AGV，AGV0有紧急任务
    state1 = MockAGVState(pose=(0.0, 0.0, 0.0), speed=0.5)
    state2 = MockAGVState(pose=(0.3, 0.0, 0.0), speed=0.5)
    coordinator.register_agv("AGV_0", agv_spec_m, state1)
    coordinator.register_agv("AGV_1", agv_spec_m, state2)
    
    # 给AGV0分配紧急任务
    task = SwarmTask(priority=TaskPriority.P0_URGENT)
    task_id = coordinator.add_task(task)
    coordinator.allocate_tasks()
    assert coordinator.agvs["AGV_0"].current_task is not None
    
    conflicts = coordinator.detect_conflicts()
    resolved_count = coordinator.resolve_conflicts(conflicts, mode="centralized")
    assert resolved_count == 1
    assert conflicts[0].resolved is True
    # AGV1应该让行，速度为0
    assert coordinator.agvs["AGV_1"].current_state.target_speed == 0.0
    assert "AGV_1 停车让行AGV_0" in conflicts[0].resolution or "AGV AGV_1 停车让行AGV AGV_0" in conflicts[0].resolution

# 测试用例9: 测试分布式ORCA碰撞避免
def test_distributed_orca_collision_avoidance(coordinator, agv_spec_m):
    # 注册2台相向行驶的AGV
    state1 = MockAGVState(pose=(0.0, 0.0, 0.0), speed=1.0, velocity=(1.0, 0.0, 0.0))
    state2 = MockAGVState(pose=(1.0, 0.0, 0.0), speed=1.0, velocity=(-1.0, 0.0, 0.0))
    coordinator.register_agv("AGV_0", agv_spec_m, state1)
    coordinator.register_agv("AGV_1", agv_spec_m, state2)
    
    conflicts = coordinator.detect_conflicts()
    resolved_count = coordinator.resolve_conflicts(conflicts, mode="distributed")
    assert resolved_count >= 1
    assert conflicts[0].resolved is True
    # 速度应该被调整
    assert coordinator.agvs["AGV_0"].current_state.target_speed < 1.0
    assert coordinator.agvs["AGV_1"].current_state.target_speed < 1.0

# 测试用例10: 测试死锁检测
def test_deadlock_detection(coordinator, agv_spec_m):
    # 注册3台AGV形成死锁环: A等B, B等C, C等A
    state1 = MockAGVState(pose=(0.0, 0.0, 0.0), waiting_for=["AGV_1"])
    state2 = MockAGVState(pose=(2.0, 0.0, 0.0), waiting_for=["AGV_2"])
    state3 = MockAGVState(pose=(4.0, 0.0, 0.0), waiting_for=["AGV_0"])
    coordinator.register_agv("AGV_0", agv_spec_m, state1)
    coordinator.register_agv("AGV_1", agv_spec_m, state2)
    coordinator.register_agv("AGV_2", agv_spec_m, state3)
    
    conflicts = coordinator.detect_conflicts()
    assert any(c.conflict_type == "path_deadlock" for c in conflicts)
    deadlock_conflict = next(c for c in conflicts if c.conflict_type == "path_deadlock")
    assert set(deadlock_conflict.involved_agvs) == {"AGV_0", "AGV_1", "AGV_2"}

# 测试用例11: 测试死锁解决
def test_deadlock_resolution(coordinator, agv_spec_m):
    # 注册2台AGV形成死锁
    state1 = MockAGVState(pose=(0.0, 0.0, 0.0), waiting_for=["AGV_1"])
    state2 = MockAGVState(pose=(2.0, 0.0, 0.0), waiting_for=["AGV_0"])
    coordinator.register_agv("AGV_0", agv_spec_m, state1)
    coordinator.register_agv("AGV_1", agv_spec_m, state2)
    
    # 给AGV0分配低优先级任务，AGV1分配高优先级
    task0 = SwarmTask(priority=TaskPriority.P3_LOW)
    task1 = SwarmTask(priority=TaskPriority.P1_HIGH)
    coordinator.add_task(task0)
    coordinator.add_task(task1)
    coordinator.allocate_tasks()
    
    conflicts = coordinator.detect_conflicts()
    resolved_count = coordinator.resolve_conflicts(conflicts)
    assert resolved_count >= 1
    # 低优先级的AGV0应该重新规划路径
    assert "AGV_0 重新规划路径" in conflicts[0].resolution
    assert len(coordinator.agvs["AGV_0"].current_state.waiting_for) == 0

# 测试用例12: 测试资源竞争检测
def test_resource_contention_detection(coordinator, agv_spec_m):
    # 注册2台AGV同时占用同一个DOCK
    state1 = MockAGVState(pose=(0.0, 0.0, 0.0), occupying_resource="DOCK1")
    state2 = MockAGVState(pose=(0.2, 0.0, 0.0), occupying_resource="DOCK1")
    coordinator.register_agv("AGV_0", agv_spec_m, state1)
    coordinator.register_agv("AGV_1", agv_spec_m, state2)
    
    conflicts = coordinator.detect_conflicts()
    assert any(c.conflict_type == "resource_contention" for c in conflicts)
    resource_conflict = next(c for c in conflicts if c.conflict_type == "resource_contention")
    assert "DOCK1" in str(resource_conflict.resolution) or "DOCK1" in str(resource_conflict.location)
    assert set(resource_conflict.involved_agvs) == {"AGV_0", "AGV_1"}

# 测试用例13: 测试资源竞争解决
def test_resource_contention_resolution(coordinator, agv_spec_m):
    # 注册2台AGV竞争同一个DOCK
    state1 = MockAGVState(pose=(0.0, 0.0, 0.0), occupying_resource="DOCK1")
    state2 = MockAGVState(pose=(0.2, 0.0, 0.0), occupying_resource="DOCK1")
    coordinator.register_agv("AGV_0", agv_spec_m, state1)
    coordinator.register_agv("AGV_1", agv_spec_m, state2)
    
    # 给AGV0分配高优先级任务
    task0 = SwarmTask(priority=TaskPriority.P1_HIGH)
    task1 = SwarmTask(priority=TaskPriority.P2_MEDIUM)
    coordinator.add_task(task0)
    coordinator.add_task(task1)
    coordinator.allocate_tasks()
    
    conflicts = coordinator.detect_conflicts()
    # 找到资源冲突
    resource_conflict = next(c for c in conflicts if c.conflict_type == "resource_contention")
    resolved_count = coordinator.resolve_conflicts(conflicts)
    assert resolved_count >= 1
    # AGV0保留资源，AGV1等待
    assert coordinator.agvs["AGV_1"].current_state.occupying_resource is None
    assert "AGV_0" in resource_conflict.resolution
    assert "其他AGV等待" in resource_conflict.resolution

# 测试用例14: 测试路径规划
def test_path_planning(coordinator, agv_spec_m):
    state = MockAGVState(pose=(0.1, 0.1, 0.0))
    coordinator.register_agv("AGV_0", agv_spec_m, state)
    agv = coordinator.agvs["AGV_0"]
    
    path = coordinator.plan_path_for_agv(agv, (10.0, 0.0, 0.0))
    assert len(path) >= 2
    # 起点附近
    assert np.linalg.norm(np.array(path[0]) - np.array((0.1, 0.1, 0.0))) < 0.2
    # 终点附近
    assert np.linalg.norm(np.array(path[-1]) - np.array((10.0, 0.0, 0.0))) < 0.2

# 测试用例15: 测试AGV状态同步
def test_agv_state_sync(coordinator, agv_spec_m):
    state = MockAGVState(pose=(0.0, 0.0, 0.0))
    coordinator.register_agv("AGV_0", agv_spec_m, state)
    
    # 分配任务
    task = SwarmTask(source_point=(0.0, 0.0, 0.0), target_point=(2.0, 0.0, 0.0))
    task_id = coordinator.add_task(task)
    coordinator.allocate_tasks()
    assert coordinator.tasks[task_id].status == TaskStatus.ASSIGNED
    
    # 同步状态，任务应该变为执行中
    coordinator.sync_agv_states()
    assert coordinator.tasks[task_id].status == TaskStatus.IN_PROGRESS
    assert coordinator.agvs["AGV_0"].current_task.started_at is not None

# 测试用例16: 测试任务完成处理
def test_task_completion(coordinator, agv_spec_m):
    state = MockAGVState(pose=(2.0, 0.0, 0.0), path=[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)])
    coordinator.register_agv("AGV_0", agv_spec_m, state)
    
    # 添加并分配任务
    task = SwarmTask(target_point=(2.0, 0.0, 0.0))
    task_id = coordinator.add_task(task)
    coordinator.allocate_tasks()
    coordinator.sync_agv_states()
    
    # 模拟任务完成（进度100%）
    coordinator.agvs["AGV_0"].current_task.progress = 1.0
    coordinator.sync_agv_states()
    
    assert coordinator.tasks[task_id].status == TaskStatus.COMPLETED
    assert coordinator.tasks[task_id].completed_at is not None
    assert coordinator.swarm_metrics['tasks_completed'] == 1
    assert coordinator.agvs["AGV_0"].current_task is None  # 任务完成后清空

# 测试用例17: 测试蜂群状态获取API
def test_get_swarm_status(coordinator, agv_spec_m):
    # 注册2台AGV
    for i in range(2):
        state = MockAGVState(pose=(i*2.0, 0.0, 0.0))
        coordinator.register_agv(f"AGV_{i}", agv_spec_m, state)
    
    # 添加3个任务
    for i in range(3):
        task = SwarmTask()
        coordinator.add_task(task)
    
    status = coordinator.get_swarm_status()
    assert status['version'] == 'v2.88.0'
    assert status['total_agvs'] == 2
    assert status['active_agvs'] == 2
    assert status['pending_tasks'] == 3
    assert status['completed_tasks'] == 0
    assert status['conflict_count'] == 0

# 测试用例18: 测试监控面板JSON格式导出
def test_monitoring_dashboard_json(coordinator, agv_spec_m):
    state = MockAGVState(pose=(0.0, 0.0, 0.0))
    coordinator.register_agv("AGV_0", agv_spec_m, state)
    coordinator.add_task(SwarmTask())
    
    data = coordinator.get_monitoring_dashboard_data(format="json")
    assert 'timestamp' in data
    assert 'metrics' in data
    assert 'agvs' in data
    assert 'tasks' in data
    assert 'active_conflicts' in data
    assert len(data['agvs']) == 1
    assert len(data['tasks']) == 1

# 测试用例19: 测试监控面板CSV格式导出
def test_monitoring_dashboard_csv(coordinator, agv_spec_m):
    state = MockAGVState(pose=(0.0, 0.0, 0.0))
    coordinator.register_agv("AGV_0", agv_spec_m, state)
    
    data = coordinator.get_monitoring_dashboard_data(format="csv")
    assert 'csv' in data
    assert 'Metrics' in data['csv']
    assert 'AGV' in data['csv']
    assert 'AGV_0' in data['csv']

# 测试用例20: 测试监控面板HTML格式导出
def test_monitoring_dashboard_html(coordinator, agv_spec_m):
    state = MockAGVState(pose=(0.0, 0.0, 0.0))
    coordinator.register_agv("AGV_0", agv_spec_m, state)
    coordinator.add_task(SwarmTask(priority=TaskPriority.P0_URGENT))
    
    data = coordinator.get_monitoring_dashboard_data(format="html")
    assert 'html' in data
    assert 'AGV Swarm Monitoring Dashboard v2.88.0' in data['html']
    assert 'AGV_0' in data['html']
    assert 'P0_URGENT' in data['html']

# 测试用例21: 测试面板导出到文件
def test_export_dashboard_to_file(coordinator, tmp_path):
    output_file = tmp_path / "dashboard.html"
    result = coordinator.export_dashboard(str(output_file), format="html")
    assert result is True
    assert output_file.exists()
    content = output_file.read_text()
    assert 'AGV Swarm Monitoring Dashboard' in content

# 测试用例22: 测试蜂群控制周期step
def test_swarm_step(coordinator, agv_spec_m):
    # 注册AGV
    state = MockAGVState(pose=(0.0, 0.0, 0.0), speed=0.5)
    coordinator.register_agv("AGV_0", agv_spec_m, state)
    # 添加任务
    coordinator.add_task(SwarmTask())
    
    initial_sim_time = coordinator.simulation_time
    initial_distance = coordinator.swarm_metrics['total_distance_traveled']
    
    # 执行step
    coordinator.step(0.1)
    
    assert coordinator.simulation_time == initial_sim_time + 0.1
    assert coordinator.swarm_metrics['total_distance_traveled'] > initial_distance
    # 任务应该被分配
    assert len([t for t in coordinator.tasks.values() if t.status == TaskStatus.ASSIGNED]) >= 1

# 测试用例23: 测试任务队列处理
def test_task_queue_processing(coordinator, agv_spec_m):
    state = MockAGVState(pose=(0.0, 0.0, 0.0))
    coordinator.register_agv("AGV_0", agv_spec_m, state)
    
    # 添加3个任务
    task_ids = []
    for i in range(3):
        task = SwarmTask(target_point=(2.0 * (i+1), 0.0, 0.0))
        task_ids.append(coordinator.add_task(task))
    
    # 分配任务
    coordinator.allocate_tasks()
    # 第一个任务执行中，后面两个在队列
    assert coordinator.agvs["AGV_0"].current_task is not None
    assert len(coordinator.agvs["AGV_0"].task_queue) == 2
    
    # 完成第一个任务
    coordinator.agvs["AGV_0"].current_task.progress = 1.0
    coordinator.sync_agv_states()
    
    # 第二个任务应该开始执行
    assert coordinator.agvs["AGV_0"].current_task.task_id == task_ids[1]
    assert len(coordinator.agvs["AGV_0"].task_queue) == 1
    assert coordinator.tasks[task_ids[0]].status == TaskStatus.COMPLETED

# 测试用例24: 测试低电量AGV不分配任务
def test_low_battery_task_allocation(coordinator, agv_spec_m):
    # 低电量AGV
    state_low = MockAGVState(pose=(0.0, 0.0, 0.0))
    coordinator.register_agv("AGV_LOW", agv_spec_m, state_low)
    coordinator.agvs["AGV_LOW"].battery_level = 5.0  # 电量5%
    
    # 正常AGV
    state_normal = MockAGVState(pose=(1.0, 0.0, 0.0))
    coordinator.register_agv("AGV_NORMAL", agv_spec_m, state_normal)
    
    # 添加任务
    task = SwarmTask(deadline=600)
    task_id = coordinator.add_task(task)
    coordinator.allocate_tasks()
    
    # 任务应该分配给正常AGV，不是低电量的
    assert coordinator.tasks[task_id].assigned_agv_id == "AGV_NORMAL"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
