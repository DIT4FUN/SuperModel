#!/usr/bin/env python3
"""
Test cases for AGV Swarm Coordinator module
SuperModel v2.88.0 - 2026-04-12
"""

import os
import sys
import time
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

# Add parent path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.embodied.agv_swarm_coordinator import (
    AGVSwarmCoordinator, SwarmTask, TaskPriority, TaskStatus,
    AGVSwarmMember, SwarmConflict
)

@pytest.fixture
def mock_scene():
    """Mock warehouse scene for testing"""
    scene = Mock()
    # Navigation points: grid of 5x5 points
    scene.navigation_points = {
        f"node_{x}_{y}": (x*2.0, y*2.0, 0.0)
        for x in range(5)
        for y in range(5)
    }
    # Path segments: connect adjacent nodes
    scene.path_segments = {}
    for x in range(5):
        for y in range(5):
            if x < 4:
                scene.path_segments[(f"node_{x}_{y}", f"node_{x+1}_{y}")] = 2.0
            if y < 4:
                scene.path_segments[(f"node_{x}_{y}", f"node_{x}_{y+1}")] = 2.0
    # Resources
    scene.resources = {
        "charging_1": {"position": (0.0, 0.0, 0.0)},
        "loading_1": {"position": (8.0, 0.0, 0.0)},
        "unloading_1": {"position": (0.0, 8.0, 0.0)},
    }
    return scene

@pytest.fixture
def mock_agv_spec_m():
    """Mock M size AGV spec"""
    spec = Mock()
    spec.size_class = "M"
    spec.max_payload = 100.0
    spec.max_speed = 2.0
    spec.power_consumption_rate = 10.0  # W
    spec.battery_capacity = 1000.0  # Wh
    return spec

@pytest.fixture
def mock_agv_spec_l():
    """Mock L size AGV spec"""
    spec = Mock()
    spec.size_class = "L"
    spec.max_payload = 300.0
    spec.max_speed = 1.5
    spec.power_consumption_rate = 15.0
    spec.battery_capacity = 1500.0
    return spec

@pytest.fixture
def mock_agv_state():
    """Mock AGV state"""
    state = Mock()
    state.pose = (0.0, 0.0, 0.0)
    state.speed = 0.0
    state.velocity = [0.0, 0.0, 0.0]
    state.waiting_for = []
    state.occupying_resource = None
    state.path = []
    state.target_speed = 0.0
    return state

@pytest.fixture
def coordinator(mock_scene):
    """Create coordinator instance"""
    return AGVSwarmCoordinator(mock_scene)

# Test 1: Basic initialization
def test_coordinator_initialization(coordinator):
    assert coordinator is not None
    assert len(coordinator.agvs) == 0
    assert len(coordinator.tasks) == 0
    assert len(coordinator.conflicts) == 0
    assert coordinator.global_map is not None
    assert len(coordinator.global_map.nodes) == 25
    assert len(coordinator.global_map.edges) == 40

# Test 2: AGV registration and unregistration
def test_agv_registration(coordinator, mock_agv_spec_m, mock_agv_state):
    coordinator.register_agv("AGV_001", mock_agv_spec_m, mock_agv_state)
    assert len(coordinator.agvs) == 1
    assert "AGV_001" in coordinator.agvs
    agv = coordinator.agvs["AGV_001"]
    assert agv.agv_id == "AGV_001"
    assert agv.spec == mock_agv_spec_m
    assert agv.available == True
    
    # Test unregistration
    coordinator.unregister_agv("AGV_001")
    assert len(coordinator.agvs) == 0
    assert "AGV_001" not in coordinator.agvs

# Test 3: Task addition and cancellation
def test_task_operations(coordinator):
    task = SwarmTask(
        task_type="transport",
        priority=TaskPriority.P1_HIGH,
        source_point=(0.0, 0.0, 0.0),
        target_point=(8.0, 8.0, 0.0),
        payload=50.0,
        deadline=3600.0
    )
    
    task_id = coordinator.add_task(task)
    assert len(coordinator.tasks) == 1
    assert task_id in coordinator.tasks
    assert coordinator.tasks[task_id].status == TaskStatus.PENDING
    
    # Test cancellation
    result = coordinator.cancel_task(task_id)
    assert result == True
    assert coordinator.tasks[task_id].status == TaskStatus.CANCELLED

# Test 4: Task allocation score calculation
def test_task_allocation_score(coordinator, mock_agv_spec_m, mock_agv_state):
    agv = AGVSwarmMember(
        agv_id="AGV_001",
        spec=mock_agv_spec_m,
        current_state=mock_agv_state,
        battery_level=100.0,
        available=True
    )
    
    # Test valid task
    valid_task = SwarmTask(
        payload=50.0,
        required_agv_spec="M",
        source_point=(2.0, 0.0, 0.0),
        target_point=(6.0, 6.0, 0.0),
        deadline=3600.0
    )
    
    score = coordinator.calculate_task_allocation_score(agv, valid_task)
    assert score != float('inf')
    assert score > 0
    
    # Test spec mismatch (task requires L, AGV is M)
    spec_mismatch_task = SwarmTask(
        payload=50.0,
        required_agv_spec="L",
        source_point=(0.0, 0.0, 0.0),
        target_point=(1.0, 1.0, 0.0)
    )
    score = coordinator.calculate_task_allocation_score(agv, spec_mismatch_task)
    assert score == float('inf')
    
    # Test payload exceed
    heavy_task = SwarmTask(
        payload=150.0,
        required_agv_spec="M",
        source_point=(0.0, 0.0, 0.0),
        target_point=(1.0, 1.0, 0.0)
    )
    score = coordinator.calculate_task_allocation_score(agv, heavy_task)
    assert score == float('inf')
    
    # Test low battery
    agv.battery_level = 1.0
    score = coordinator.calculate_task_allocation_score(agv, valid_task)
    assert score == float('inf')

# Test 5: Greedy task allocation
def test_greedy_allocation(coordinator, mock_agv_spec_m, mock_agv_state):
    # Register 2 AGVs
    state1 = Mock()
    state1.pose = (0.0, 0.0, 0.0)
    state1.speed = 0.0
    state1.waiting_for = []
    state1.occupying_resource = None
    state1.path = []
    coordinator.register_agv("AGV_001", mock_agv_spec_m, state1)
    
    state2 = Mock()
    state2.pose = (8.0, 8.0, 0.0)
    state2.speed = 0.0
    state2.waiting_for = []
    state2.occupying_resource = None
    state2.path = []
    coordinator.register_agv("AGV_002", mock_agv_spec_m, state2)
    
    # Add 2 tasks
    task1 = SwarmTask(
        source_point=(0.0, 0.0, 0.0),
        target_point=(8.0, 8.0, 0.0),
        payload=50.0,
        priority=TaskPriority.P0_URGENT
    )
    task2 = SwarmTask(
        source_point=(8.0, 8.0, 0.0),
        target_point=(0.0, 0.0, 0.0),
        payload=50.0,
        priority=TaskPriority.P1_HIGH
    )
    coordinator.add_task(task1)
    coordinator.add_task(task2)
    
    # Run allocation
    assigned = coordinator.allocate_tasks(algorithm="greedy")
    assert assigned == 2
    
    # Check assignments: task1 should go to AGV_001 (closer), task2 to AGV_002
    assert task1.assigned_agv_id == "AGV_001"
    assert task2.assigned_agv_id == "AGV_002"
    assert task1.status == TaskStatus.ASSIGNED
    assert task2.status == TaskStatus.ASSIGNED

# Test 6: Auction-based task allocation
def test_auction_allocation(coordinator, mock_agv_spec_m, mock_agv_spec_l, mock_agv_state):
    # Register 2 AGVs: M and L
    state1 = Mock()
    state1.pose = (0.0, 0.0, 0.0)
    state1.speed = 0.0
    state1.waiting_for = []
    state1.occupying_resource = None
    state1.path = []
    coordinator.register_agv("AGV_M", mock_agv_spec_m, state1)
    
    state2 = Mock()
    state2.pose = (0.0, 0.0, 0.0)
    state2.speed = 0.0
    state2.waiting_for = []
    state2.occupying_resource = None
    state2.path = []
    coordinator.register_agv("AGV_L", mock_agv_spec_l, state2)
    
    # Add heavy task that only L can handle
    heavy_task = SwarmTask(
        payload=200.0,
        required_agv_spec="L",
        source_point=(0.0, 0.0, 0.0),
        target_point=(8.0, 8.0, 0.0),
        priority=TaskPriority.P0_URGENT
    )
    coordinator.add_task(heavy_task)
    
    # Run auction allocation
    assigned = coordinator.allocate_tasks(algorithm="auction")
    assert assigned == 1
    assert heavy_task.assigned_agv_id == "AGV_L"

# Test 7: Hungarian algorithm allocation
def test_hungarian_allocation(coordinator, mock_agv_spec_m, mock_agv_state):
    # Register 3 AGVs
    for i in range(3):
        state = Mock()
        state.pose = (i*2.0, 0.0, 0.0)
        state.speed = 0.0
        state.waiting_for = []
        state.occupying_resource = None
        state.path = []
        coordinator.register_agv(f"AGV_{i}", mock_agv_spec_m, state)
    
    # Add 3 tasks
    for i in range(3):
        task = SwarmTask(
            source_point=(i*3.0, 2.0, 0.0),
            target_point=(8.0 - i*2.0, 8.0, 0.0),
            payload=50.0,
            priority=TaskPriority.P2_MEDIUM
        )
        coordinator.add_task(task)
    
    # Run Hungarian allocation
    assigned = coordinator.allocate_tasks(algorithm="hungarian")
    assert assigned == 3
    # All tasks should be assigned
    for task in coordinator.tasks.values():
        assert task.assigned_agv_id is not None
        assert task.status == TaskStatus.ASSIGNED

# Test 8: Collision detection
def test_collision_detection(coordinator, mock_agv_spec_m, mock_agv_state):
    # Register 2 AGVs very close to each other (less than 0.5m safety distance)
    state1 = Mock()
    state1.pose = (0.0, 0.0, 0.0)
    state1.speed = 0.6
    state1.waiting_for = []
    state1.occupying_resource = None
    state1.path = []
    coordinator.register_agv("AGV_001", mock_agv_spec_m, state1)
    
    state2 = Mock()
    state2.pose = (0.3, 0.0, 0.0)  # Only 30cm apart
    state2.speed = 0.6
    state2.waiting_for = []
    state2.occupying_resource = None
    state2.path = []
    coordinator.register_agv("AGV_002", mock_agv_spec_m, state2)
    
    # Detect conflicts
    conflicts = coordinator.detect_conflicts()
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.conflict_type == "collision"
    assert set(conflict.involved_agvs) == {"AGV_001", "AGV_002"}
    assert conflict.severity == 8  # Moving AGVs have higher severity

# Test 9: Deadlock detection
def test_deadlock_detection(coordinator, mock_agv_spec_m, mock_agv_state):
    # Register 2 AGVs waiting for each other
    state1 = Mock()
    state1.pose = (0.0, 0.0, 0.0)
    state1.speed = 0.0
    state1.waiting_for = ["AGV_002"]
    state1.occupying_resource = None
    state1.path = []
    coordinator.register_agv("AGV_001", mock_agv_spec_m, state1)
    
    state2 = Mock()
    state2.pose = (2.0, 0.0, 0.0)
    state2.speed = 0.0
    state2.waiting_for = ["AGV_001"]
    state2.occupying_resource = None
    state2.path = []
    coordinator.register_agv("AGV_002", mock_agv_spec_m, state2)
    
    # Detect conflicts
    conflicts = coordinator.detect_conflicts()
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.conflict_type == "path_deadlock"
    assert set(conflict.involved_agvs) == {"AGV_001", "AGV_002"}
    assert conflict.severity == 9

# Test 10: Resource contention detection
def test_resource_contention_detection(coordinator, mock_agv_spec_m, mock_agv_state):
    # Register 2 AGVs occupying the same resource
    state1 = Mock()
    state1.pose = (0.0, 0.0, 0.0)
    state1.speed = 0.0
    state1.waiting_for = []
    state1.occupying_resource = "loading_1"
    state1.path = []
    coordinator.register_agv("AGV_001", mock_agv_spec_m, state1)
    
    state2 = Mock()
    state2.pose = (0.5, 0.0, 0.0)
    state2.speed = 0.0
    state2.waiting_for = []
    state2.occupying_resource = "loading_1"
    state2.path = []
    coordinator.register_agv("AGV_002", mock_agv_spec_m, state2)
    
    # Detect conflicts
    conflicts = coordinator.detect_conflicts()
    assert len(conflicts) >= 1  # At least resource contention, maybe collision too
    resource_conflicts = [c for c in conflicts if c.conflict_type == "resource_contention"]
    assert len(resource_conflicts) == 1
    conflict = resource_conflicts[0]
    assert set(conflict.involved_agvs) == {"AGV_001", "AGV_002"}
    assert conflict.severity == 6

# Test 11: Centralized collision resolution
def test_centralized_collision_resolution(coordinator, mock_agv_spec_m, mock_agv_state):
    # Create collision conflict
    state1 = Mock()
    state1.pose = (0.0, 0.0, 0.0)
    state1.speed = 1.0
    state1.waiting_for = []
    state1.occupying_resource = None
    state1.path = []
    state1.target_speed = 1.0
    coordinator.register_agv("AGV_001", mock_agv_spec_m, state1)
    
    state2 = Mock()
    state2.pose = (0.3, 0.0, 0.0)
    state2.speed = 1.0
    state2.waiting_for = []
    state2.occupying_resource = None
    state2.path = []
    state2.target_speed = 1.0
    coordinator.register_agv("AGV_002", mock_agv_spec_m, state2)
    
    # Assign tasks to set priority: AGV_001 has P0, AGV_002 has P2
    task1 = SwarmTask(priority=TaskPriority.P0_URGENT)
    task2 = SwarmTask(priority=TaskPriority.P2_MEDIUM)
    coordinator.agvs["AGV_001"].current_task = task1
    coordinator.agvs["AGV_002"].current_task = task2
    
    # Detect and resolve conflicts
    conflicts = coordinator.detect_conflicts()
    resolved = coordinator.resolve_conflicts(conflicts, mode="centralized")
    
    assert resolved == 1
    assert conflicts[0].resolved == True
    # Lower priority AGV (002) should stop
    assert state2.target_speed == 0.0
    assert "AGV_001" in state2.waiting_for

# Test 12: Distributed ORCA collision avoidance
def test_distributed_collision_avoidance(coordinator, mock_agv_spec_m, mock_agv_state):
    # Create collision conflict
    state1 = Mock()
    state1.pose = (0.0, 0.0, 0.0)
    state1.speed = 1.0
    state1.velocity = [1.0, 0.0, 0.0]
    state1.waiting_for = []
    state1.occupying_resource = None
    state1.path = []
    state1.target_speed = 1.0
    coordinator.register_agv("AGV_001", mock_agv_spec_m, state1)
    
    state2 = Mock()
    state2.pose = (0.3, 0.0, 0.0)  # 30cm apart (less than 0.5m collision detection threshold), moving towards each other
    state2.speed = 1.0
    state2.velocity = [-1.0, 0.0, 0.0]
    state2.waiting_for = []
    state2.occupying_resource = None
    state2.path = []
    state2.target_speed = 1.0
    coordinator.register_agv("AGV_002", mock_agv_spec_m, state2)
    
    # Detect and resolve conflicts in distributed mode
    conflicts = coordinator.detect_conflicts()
    resolved = coordinator.resolve_conflicts(conflicts, mode="distributed")
    
    assert resolved == 1
    assert conflicts[0].resolved == True
    # Both AGVs should have adjusted speeds
    assert state1.target_speed < 1.0
    assert state2.target_speed < 1.0

# Test 13: Path planning
def test_path_planning(coordinator, mock_agv_spec_m, mock_agv_state):
    agv = AGVSwarmMember(
        agv_id="AGV_001",
        spec=mock_agv_spec_m,
        current_state=mock_agv_state
    )
    
    # Plan path from (0,0,0) to (8,8,0)
    path = coordinator.plan_path_for_agv(agv, (8.0, 8.0, 0.0))
    assert len(path) > 0
    # First point should be near start
    assert np.linalg.norm(np.array(path[0]) - np.array([0.0, 0.0, 0.0])) < 0.1
    # Last point should be near target
    assert np.linalg.norm(np.array(path[-1]) - np.array([8.0, 8.0, 0.0])) < 0.1
    # Path length should be ~16m (8 right, 8 up)
    total_length = sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) for i in range(len(path)-1))
    assert abs(total_length - 16.0) < 1.0

# Test 14: AGV state synchronization
def test_agv_state_sync(coordinator, mock_agv_spec_m, mock_agv_state):
    # Register AGV with task
    state = Mock()
    state.pose = (0.0, 0.0, 0.0)
    state.speed = 0.0
    state.waiting_for = []
    state.occupying_resource = None
    state.path = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (4.0, 0.0, 0.0)]
    coordinator.register_agv("AGV_001", mock_agv_spec_m, state)
    
    task = SwarmTask(
        source_point=(0.0, 0.0, 0.0),
        target_point=(4.0, 0.0, 0.0),
        payload=50.0
    )
    coordinator.add_task(task)
    coordinator.allocate_tasks()
    
    # Simulate AGV moving to 2.0, 0.0 (halfway)
    state.pose = (2.0, 0.0, 0.0)
    coordinator.sync_agv_states()
    
    # Check task progress
    assert task.progress == 0.5
    assert task.status == TaskStatus.IN_PROGRESS
    
    # Simulate AGV reaching target
    state.pose = (4.0, 0.0, 0.0)
    coordinator.sync_agv_states()
    
    # Check task completed
    assert task.progress >= 1.0
    assert task.status == TaskStatus.COMPLETED
    assert coordinator.swarm_metrics['tasks_completed'] == 1

# Test 15: Swarm status API
def test_get_swarm_status(coordinator, mock_agv_spec_m, mock_agv_state):
    # Register AGV and add tasks
    coordinator.register_agv("AGV_001", mock_agv_spec_m, mock_agv_state)
    for _ in range(5):
        task = SwarmTask()
        coordinator.add_task(task)
    
    status = coordinator.get_swarm_status()
    assert status['version'] == 'v2.88.0'
    assert status['total_agvs'] == 1
    assert status['active_agvs'] == 1
    assert status['pending_tasks'] == 5
    assert status['in_progress_tasks'] == 0
    assert status['completed_tasks'] == 0
    assert status['conflict_count'] == 0

# Test 16: Monitoring dashboard data (JSON format)
def test_monitoring_dashboard_json(coordinator, mock_agv_spec_m, mock_agv_state):
    coordinator.register_agv("AGV_001", mock_agv_spec_m, mock_agv_state)
    task = SwarmTask()
    coordinator.add_task(task)
    
    data = coordinator.get_monitoring_dashboard_data(format="json")
    assert 'timestamp' in data
    assert 'metrics' in data
    assert 'agvs' in data
    assert 'tasks' in data
    assert 'active_conflicts' in data
    assert len(data['agvs']) == 1
    assert len(data['tasks']) == 1
    assert data['agvs'][0]['agv_id'] == 'AGV_001'

# Test 17: Monitoring dashboard data (CSV format)
def test_monitoring_dashboard_csv(coordinator):
    data = coordinator.get_monitoring_dashboard_data(format="csv")
    assert 'csv' in data
    assert isinstance(data['csv'], str)
    assert 'Metrics' in data['csv']

# Test 18: Monitoring dashboard data (HTML format)
def test_monitoring_dashboard_html(coordinator):
    data = coordinator.get_monitoring_dashboard_data(format="html")
    assert 'html' in data
    assert isinstance(data['html'], str)
    assert '<html>' in data['html']
    assert 'AGV Swarm Monitoring Dashboard' in data['html']

# Test 19: Dashboard export to file
def test_dashboard_export(coordinator, tmp_path):
    output_file = tmp_path / "dashboard.html"
    result = coordinator.export_dashboard(str(output_file), format="html")
    assert result == True
    assert output_file.exists()
    content = output_file.read_text()
    assert '<html>' in content

# Test 20: Full simulation step
def test_full_simulation_step(coordinator, mock_agv_spec_m, mock_agv_state):
    # Register 2 AGVs
    state1 = Mock()
    state1.pose = (0.0, 0.0, 0.0)
    state1.speed = 0.0
    state1.waiting_for = []
    state1.occupying_resource = None
    state1.path = []
    state1.target_speed = 0.0
    state1.velocity = [0.0, 0.0, 0.0]
    coordinator.register_agv("AGV_001", mock_agv_spec_m, state1)
    
    state2 = Mock()
    state2.pose = (8.0, 8.0, 0.0)
    state2.speed = 0.0
    state2.waiting_for = []
    state2.occupying_resource = None
    state2.path = []
    state2.target_speed = 0.0
    state2.velocity = [0.0, 0.0, 0.0]
    coordinator.register_agv("AGV_002", mock_agv_spec_m, state2)
    
    # Add 2 tasks
    task1 = SwarmTask(source_point=(0.0, 0.0, 0.0), target_point=(8.0, 8.0, 0.0), payload=50.0)
    task2 = SwarmTask(source_point=(8.0, 8.0, 0.0), target_point=(0.0, 0.0, 0.0), payload=50.0)
    coordinator.add_task(task1)
    coordinator.add_task(task2)
    
    # Allocate tasks first
    coordinator.allocate_tasks()
    # Set AGV speed to 1m/s
    state1.speed = 1.0
    state2.speed = 1.0
    
    # Run 10 steps (1 second)
    for i in range(10):
        coordinator.step(0.1)
    
    # Check status
    status = coordinator.get_swarm_status()
    assert abs(status['simulation_time'] - 1.0) < 1e-9
    assert status['total_distance'] > 0.0
    assert status['total_agvs'] == 2

# Test 21: Task queue management
def test_task_queue_management(coordinator, mock_agv_spec_m, mock_agv_state):
    # Register 1 AGV
    coordinator.register_agv("AGV_001", mock_agv_spec_m, mock_agv_state)
    
    # Add 3 tasks
    for i in range(3):
        task = SwarmTask(
            task_id=f"task_{i}",
            source_point=(i*2.0, 0.0, 0.0),
            target_point=(i*2.0 + 1.0, 0.0, 0.0),
            payload=50.0
        )
        coordinator.add_task(task)
    
    # Allocate tasks
    assigned = coordinator.allocate_tasks()
    assert assigned == 3
    
    # Check AGV has current task and 2 in queue
    agv = coordinator.agvs["AGV_001"]
    assert agv.current_task is not None
    assert len(agv.task_queue) == 2
    
    # Simulate completing current task
    agv.current_task.progress = 1.0
    coordinator.sync_agv_states()
    
    # Check next task is taken from queue
    assert agv.current_task is not None
    assert len(agv.task_queue) == 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
