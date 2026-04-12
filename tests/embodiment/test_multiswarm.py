import pytest
import time
from embodiment.multi_agv_coordinator import MultiAGVCoordinator, AGVStatus

def test_multi_agv_registration():
    """Test AGV registration with coordinator"""
    coordinator = MultiAGVCoordinator()
    
    # Register 3 AGVs
    agv1 = coordinator.register_agv("agv_001", position=(0, 0, 0), type="forklift")
    agv2 = coordinator.register_agv("agv_002", position=(10, 0, 0), type="delivery")
    agv3 = coordinator.register_agv("agv_003", position=(20, 0, 0), type="inspection")
    
    assert len(coordinator.registered_agvs) == 3
    assert coordinator.get_agv("agv_001")["type"] == "forklift"
    assert coordinator.get_agv("agv_002")["position"] == (10, 0, 0)

def test_multi_agv_task_allocation():
    """Test dynamic task allocation across AGV swarm"""
    coordinator = MultiAGVCoordinator()
    
    # Register AGVs with different capabilities
    coordinator.register_agv("agv_heavy", position=(0,0,0), type="heavy_duty", max_load=500, capabilities=["lift", "transport"])
    coordinator.register_agv("agv_fast", position=(5,0,0), type="fast_delivery", max_load=50, capabilities=["transport", "inspection"])
    coordinator.register_agv("agv_inspect", position=(10,0,0), type="inspection", max_load=10, capabilities=["inspection", "mapping"])
    
    # Allocate transport task (needs transport capability, load 100kg)
    task1 = {"task_id": "t1", "type": "transport", "required_capability": "transport", "load": 100, "target_position": (15, 0, 0)}
    allocated_agv = coordinator.allocate_task(task1)
    assert allocated_agv == "agv_heavy"  # Only one that can handle 100kg load
    
    # Allocate inspection task
    task2 = {"task_id": "t2", "type": "inspection", "required_capability": "inspection", "target_position": (20, 0, 0)}
    allocated_agv = coordinator.allocate_task(task2)
    assert allocated_agv in ["agv_fast", "agv_inspect"]  # Both have inspection capability

def test_multi_agv_collision_avoidance():
    """Test inter-AGV collision avoidance"""
    coordinator = MultiAGVCoordinator()
    
    # Register two AGVs on collision course
    agv1 = coordinator.register_agv("agv_a", position=(0, 0, 0), velocity=(1, 0, 0))
    agv2 = coordinator.register_agv("agv_b", position=(5, 0, 0), velocity=(-1, 0, 0))
    
    # Check collision risk
    risk = coordinator.check_collision_risk("agv_a", "agv_b")
    assert risk > 0.8  # High collision risk
    
    # Get avoidance path
    path = coordinator.get_avoidance_path("agv_a", "agv_b")
    assert len(path) > 0
    assert path[0][1] != 0  # Should deviate in y-axis to avoid collision

def test_multi_agv_swarm_coordination():
    """Test swarm coordination for area coverage task"""
    coordinator = MultiAGVCoordinator()
    
    # Register 4 inspection AGVs
    for i in range(4):
        coordinator.register_agv(f"agv_{i}", position=(i*5, 0, 0), type="inspection", capabilities=["mapping"])
    
    # Assign area mapping task (20x20 area)
    task = {"task_id": "area_map", "type": "area_coverage", "area": (0,0,20,20), "required_capability": "mapping"}
    
    # Split task across swarm
    subtasks = coordinator.split_swarm_task(task)
    assert len(subtasks) == 4  # One subtask per AGV
    
    # Verify each subtask covers a quadrant of the area
    quadrants = [st["sub_area"] for st in subtasks]
    assert (0,0,10,10) in quadrants
    assert (10,0,20,10) in quadrants
    assert (0,10,10,20) in quadrants
    assert (10,10,20,20) in quadrants

def test_multi_agv_fault_tolerance():
    """Test swarm fault tolerance when an AGV fails"""
    coordinator = MultiAGVCoordinator()
    
    # Register 3 AGVs
    coordinator.register_agv("agv1", position=(0,0,0), type="delivery", status=AGVStatus.ACTIVE)
    coordinator.register_agv("agv2", position=(5,0,0), type="delivery", status=AGVStatus.ACTIVE)
    coordinator.register_agv("agv3", position=(10,0,0), type="delivery", status=AGVStatus.ACTIVE)
    
    # Assign task to agv1
    task = {"task_id": "deliver_1", "type": "transport", "required_capability": "transport"}
    allocated = coordinator.allocate_task(task)
    assert allocated == "agv1"
    
    # Mark agv1 as failed
    coordinator.update_agv_status("agv1", AGVStatus.FAULT)
    
    # Reallocate task
    reallocated = coordinator.reallocate_failed_task("deliver_1")
    assert reallocated in ["agv2", "agv3"]  # Task should be reallocated to active AGV
