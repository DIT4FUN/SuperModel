import pytest
import numpy as np
from embodiment.agv_interface import AGVInterface, AGVStatus
from embodiment.behavior_tree_engine import BehaviorTreeEngine, TaskNode, ConditionNode
from embodiment.multi_agv_coordinator import MultiAGVCoordinator
from embodiment.simulation import EmbodiedSimulation, SimulationScene

def test_agv_interface_basic():
    """Test basic AGV interface functionality"""
    agv = AGVInterface(agv_id="agv_001", agv_type="AUTO_GUIDED_VEHICLE_LEVEL_5")
    
    # Test initialization
    assert agv.agv_id == "agv_001"
    assert agv.status == AGVStatus.IDLE
    
    # Test movement commands
    result = agv.move_to(x=1.0, y=2.0, theta=0.0, speed=0.5)
    assert result["success"] == True
    assert agv.status == AGVStatus.MOVING
    
    # Test stop command
    stop_result = agv.stop()
    assert stop_result["success"] == True
    assert agv.status == AGVStatus.IDLE
    
    # Test sensor data reading
    sensor_data = agv.get_sensor_data()
    assert "tactile" in sensor_data
    assert "force" in sensor_data
    assert "imu" in sensor_data
    assert "position" in sensor_data

def test_behavior_tree_engine():
    """Test behavior tree task planning engine"""
    bt = BehaviorTreeEngine(tree_name="transport_task")
    
    # Add condition nodes
    bt.add_node(ConditionNode("is_holding_object", lambda ctx: ctx.get("has_object", False)))
    bt.add_node(ConditionNode("at_destination", lambda ctx: ctx.get("current_pos", (0,0)) == ctx.get("target_pos", (0,0))))
    
    # Add task nodes
    bt.add_node(TaskNode("pick_up_object", lambda ctx: {"success": True, "has_object": True}))
    bt.add_node(TaskNode("move_to_destination", lambda ctx: {"success": True, "current_pos": ctx["target_pos"]}))
    bt.add_node(TaskNode("drop_object", lambda ctx: {"success": True, "has_object": False}))
    
    # Build tree structure
    bt.set_root_sequence([
        "is_holding_object",
        "move_to_destination",
        "at_destination",
        "drop_object"
    ])
    
    # Test execution
    context = {"target_pos": (5.0, 5.0), "current_pos": (0.0, 0.0), "has_object": True}
    result = bt.execute(context)
    
    assert result["success"] == True
    assert result["context"]["has_object"] == False
    assert result["context"]["current_pos"] == (5.0, 5.0)

def test_multi_agv_coordinator():
    """Test multi AGV swarm coordination"""
    coordinator = MultiAGVCoordinator(swarm_id="factory_swarm_01")
    
    # Add AGVs
    coordinator.add_agv("agv_001", level=5, position=(0.0, 0.0))
    coordinator.add_agv("agv_002", level=5, position=(10.0, 0.0))
    coordinator.add_agv("agv_003", level=5, position=(20.0, 0.0))
    
    assert len(coordinator.agv_list) == 3
    
    # Assign tasks
    tasks = [
        {"task_id": "t1", "type": "transport", "start": (0,0), "end": (10,10), "priority": 1},
        {"task_id": "t2", "type": "transport", "start": (10,0), "end": (20,10), "priority": 2},
        {"task_id": "t3", "type": "transport", "start": (20,0), "end": (0,10), "priority": 3}
    ]
    
    assignment = coordinator.assign_tasks(tasks)
    assert len(assignment) == 3
    assert all([agv_id in ["agv_001", "agv_002", "agv_003"] for agv_id in assignment.keys()])
    
    # Test collision avoidance
    path_conflict = coordinator.check_path_conflicts()
    assert path_conflict == False
    
    # Test new collision detection feature
    # Update AGV positions to be very close (less than safety distance 0.5m)
    coordinator.update_agv_state(1, (0.0, 0.0), 0.0, 1.0)
    coordinator.update_agv_state(2, (0.3, 0.0), 0.0, 1.0)
    
    conflicts = coordinator.check_conflicts()
    assert len(conflicts) == 1
    assert conflicts[0][2] == "collision"
    
    # Test obstacle collision detection
    coordinator.update_global_obstacles([(0.2, 0.0, 0.2)])  # 障碍物在(0.2, 0.0)，半径0.2m
    conflicts = coordinator.check_conflicts()
    assert len(conflicts) >= 2  # AGV1和AGV2碰撞 + AGV1/AGV2和障碍物碰撞
    assert any(c[2] == "obstacle_collision" for c in conflicts)

def test_embodied_simulation():
    """Test embodied simulation environment"""
    sim = EmbodiedSimulation(scene=SimulationScene.FACTORY_WAREHOUSE)
    
    # Add AGV to simulation
    agv_id = sim.add_agv(agv_type="LEVEL_5", initial_pos=(0.0, 0.0, 0.0))
    
    # Add obstacles
    sim.add_obstacle(type="box", position=(5.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
    sim.add_obstacle(type="cylinder", position=(10.0, 5.0, 0.0), size=(0.5, 2.0))
    
    # Run simulation step
    state = sim.step(duration=1.0)
    assert "time" in state
    assert agv_id in state["agvs"]
    assert "obstacles" in state
    
    # Test collision detection
    collision = sim.check_collision(agv_id)
    assert collision == False
    
    # Reset simulation
    sim.reset()
    state = sim.get_current_state()
    assert state["time"] == 0.0

def test_embodiment_integration():
    """Test full embodiment module integration"""
    # Initialize all components
    sim = EmbodiedSimulation(scene=SimulationScene.LOGISTICS_CENTER)
    coordinator = MultiAGVCoordinator(swarm_id="test_swarm")
    bt = BehaviorTreeEngine("full_transport")
    
    # Add AGVs
    agv1_id = sim.add_agv("LEVEL_5", (0.0, 0.0, 0.0))
    agv2_id = sim.add_agv("LEVEL_5", (15.0, 0.0, 0.0))
    
    coordinator.add_agv(agv1_id, 5, (0,0))
    coordinator.add_agv(agv2_id, 5, (15,0))
    
    # Create transport task
    task = {"task_id": "transport_1", "type": "transport", "start": (0,0), "end": (20,20), "load": 50.0}
    
    # Assign task
    assignment = coordinator.assign_tasks([task])
    assigned_agv = list(assignment.keys())[0]
    
    # Execute task via behavior tree
    context = {
        "agv_id": assigned_agv,
        "start_pos": (0.0, 0.0),
        "target_pos": (20.0, 20.0),
        "has_object": False
    }
    
    # Build execution tree
    bt.add_node(ConditionNode("at_start", lambda ctx: ctx["current_pos"] == ctx["start_pos"]))
    bt.add_node(TaskNode("pick_load", lambda ctx: {"success": True, "has_object": True}))
    bt.add_node(TaskNode("navigate", lambda ctx: {"success": True, "current_pos": ctx["target_pos"]}))
    bt.add_node(ConditionNode("at_end", lambda ctx: ctx["current_pos"] == ctx["target_pos"]))
    bt.add_node(TaskNode("drop_load", lambda ctx: {"success": True, "has_object": False}))
    
    bt.set_root_sequence(["at_start", "pick_load", "navigate", "at_end", "drop_load"])
    
    result = bt.execute(context)
    assert result["success"] == True
    assert result["context"]["has_object"] == False

def test_multi_agv_swarm_collaborative_transport():
    """Test multi AGV swarm collaborative transport of heavy loads (requires 2+ AGVs to carry)"""
    coordinator = MultiAGVCoordinator(swarm_id="collaborative_swarm_01")
    
    # Add 4 AGVs for collaborative task
    for i in range(4):
        coordinator.add_agv(f"agv_{i:03d}", level=5, position=(float(i*2), 0.0))
    
    assert len(coordinator.agv_list) == 4
    
    # Collaborative transport task: load 200kg, requires minimum 2 AGVs (each can carry 100kg max)
    collaborative_task = {
        "task_id": "collab_t1",
        "type": "collaborative_transport",
        "start": (0.0, 0.0),
        "end": (10.0, 10.0),
        "load_weight": 200.0,
        "min_agvs_required": 2,
        "priority": 1
    }
    
    # Assign collaborative task
    assignment = coordinator.assign_collaborative_task(collaborative_task)
    assert len(assignment) >= 2
    assert all([agv_id.startswith("agv_") for agv_id in assignment.keys()])
    
    # Test coordinated movement
    movement_result = coordinator.execute_collaborative_movement(
        task_id="collab_t1",
        target_pos=(10.0, 10.0),
        speed=0.3
    )
    
    assert movement_result["success"] == True
    assert movement_result["all_agvs_arrived"] == True
    assert abs(movement_result["position_error"] < 0.1)  # Position alignment error < 10cm
    
    # Arrange AGVs into rectangle formation for testing
    # Position AGVs with 1m spacing in a line
    for i, agv_id in enumerate(assignment.keys()):
        agv_num = int(agv_id.split('_')[1])
        coordinator.update_agv_state(agv_num, (float(i * 1.0), 0.0), 0.0, 0.3)
    
    # Test formation maintenance during movement
    formation_check = coordinator.check_formation(formation_type="rectangle", spacing=1.0)
    assert formation_check == True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
