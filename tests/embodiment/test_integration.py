import pytest
import numpy as np
from embodiment.simulation import EmbodiedSimulation
from embodiment.agv_interface import AGVHardwareInterface
from embodiment.behavior_tree_engine import BehaviorTreeEngine, NodeStatus, BehaviorNode
from sensors.tactile import TaxelArray as TactileSensor
from sensors.force import SixAxisFTSensor as ForceSensor
from sensors.imu import BMI088 as IMUSensor

def test_embodied_sensor_integration():
    """Test integration of embodiment modules with sensor stack"""
    sim = EmbodiedSimulation(environment="warehouse")
    agv = sim.spawn_agv(position=(0, 0, 0), model="AGV_FIVE_GRADE")
    
    # Attach sensors to simulated AGV
    tactile_sensor = TactileSensor(name="tactile_01", rows=16, cols=16, sampling_rate=100)
    force_sensor = ForceSensor(name="ft_01", sampling_rate=1000)
    imu_sensor = IMUSensor(name="imu_01", sampling_rate=200)
    
    agv.attach_sensor("tactile", tactile_sensor)
    agv.attach_sensor("force", force_sensor)
    agv.attach_sensor("imu", imu_sensor)
    
    # Run simulation step with contact
    sim.step(contact_force=np.array([10, 0, 0, 0, 0, 0]))
    
    # Read sensor data
    tactile_data = agv.read_sensor("tactile")
    force_data = agv.read_sensor("force")
    imu_data = agv.read_sensor("imu")
    
    assert tactile_data["contact_detected"] == True
    assert np.linalg.norm(force_data["wrench"][:3]) > 9  # Contact force detected
    assert imu_data["linear_acceleration"][0] > 0  # Acceleration from contact

def test_agv_hardware_simulation_integration():
    """Test integration between hardware interface and simulation"""
    sim = EmbodiedSimulation(environment="factory")
    hw_interface = AGVHardwareInterface(interface_type="simulation", sim_instance=sim)
    
    # Connect to simulated AGV
    hw_interface.connect(agv_id="sim_agv_01")
    
    # Send motion command
    hw_interface.set_velocity(linear=0.5, angular=0.0)
    
    # Run simulation for 2 seconds
    for _ in range(200):
        sim.step(dt=0.01)
    
    # Get position
    position = hw_interface.get_position()
    assert position[0] > 0.9  # Should have moved ~1 meter (0.5m/s * 2s)
    
    # Test emergency stop
    hw_interface.emergency_stop()
    sim.step(dt=0.01)
    velocity = hw_interface.get_velocity()
    assert np.linalg.norm(velocity[:3]) < 0.01  # Should be stopped

def test_behavior_tree_simulation_integration():
    """Test integration of behavior tree with simulated AGV"""
    sim = EmbodiedSimulation(environment="warehouse")
    agv = sim.spawn_agv(position=(0,0,0))
    bt = BehaviorTreeEngine()
    
    # Build navigation task tree
    bt.add_sequence("navigate_to_target", [
        "check_safety",
        "move_towards_target",
        "adjust_heading",
        "confirm_arrival"
    ])
    
    target = (5.0, 0.0, 0.0)
    context = {"target_position": target, "agv": agv, "sim": sim}
    
    def check_safety(ctx):
        obstacle_dist = ctx["sim"].get_nearest_obstacle_distance(ctx["agv"].id)
        return NodeStatus.SUCCESS if obstacle_dist > 0.3 else NodeStatus.FAILURE
    
    def move_towards_target(ctx):
        pos = ctx["agv"].get_position()
        dist = np.linalg.norm(np.array(ctx["target_position"]) - np.array(pos[:3]))
        if dist < 0.1:
            return NodeStatus.SUCCESS
        ctx["agv"].set_velocity(linear=0.3, angular=0.0)
        ctx["sim"].step(dt=0.1)
        return NodeStatus.RUNNING
    
    def adjust_heading(ctx): return NodeStatus.SUCCESS
    def confirm_arrival(ctx):
        pos = ctx["agv"].get_position()
        dist = np.linalg.norm(np.array(target) - np.array(pos[:3]))
        return NodeStatus.SUCCESS if dist < 0.1 else NodeStatus.FAILURE
    
    bt.add_node(BehaviorNode("check_safety", check_safety))
    bt.add_node(BehaviorNode("move_towards_target", move_towards_target))
    bt.add_node(BehaviorNode("adjust_heading", adjust_heading))
    bt.add_node(BehaviorNode("confirm_arrival", confirm_arrival))
    
    # Run behavior tree until completion
    status = NodeStatus.RUNNING
    while status == NodeStatus.RUNNING:
        status = bt.run("navigate_to_target", context)
    
    assert status == NodeStatus.SUCCESS
    final_pos = agv.get_position()
    assert np.linalg.norm(np.array(target) - np.array(final_pos[:3])) < 0.1

def test_full_embodied_pipeline():
    """Test complete embodied pipeline: sensors -> perception -> decision -> action"""
    sim = EmbodiedSimulation(environment="warehouse")
    agv = sim.spawn_agv(position=(0,0,0))
    
    # Attach sensors
    tactile = TactileSensor(name="tactile_02")
    force = ForceSensor(name="ft_02")
    imu = IMUSensor(name="imu_02")
    agv.attach_sensor("tactile", tactile)
    agv.attach_sensor("force", force)
    agv.attach_sensor("imu", imu)
    
    # Behavior tree for cargo handling task
    bt = BehaviorTreeEngine()
    bt.add_sequence("handle_cargo", [
        "approach_cargo",
        "grasp_cargo",
        "verify_grasp_quality",
        "lift_cargo",
        "transport_to_target",
        "place_cargo"
    ])
    
    context = {"agv": agv, "sim": sim, "cargo_position": (2.0, 0, 0), "target_position": (7.0, 0, 0)}
    
    # Mock node implementations (simplified for test)
    def approach_cargo(ctx):
        ctx["agv"].move_to(ctx["cargo_position"])
        ctx["sim"].run_for(2.0)
        return NodeStatus.SUCCESS
    
    def grasp_cargo(ctx):
        ctx["agv"].close_gripper()
        ctx["sim"].step(contact_force=np.array([5,0,0,0,0,0]))
        return NodeStatus.SUCCESS
    
    def verify_grasp_quality(ctx):
        tactile_data = ctx["agv"].read_sensor("tactile")
        force_data = ctx["agv"].read_sensor("force")
        return NodeStatus.SUCCESS if tactile_data["contact_coverage"] > 0.8 and np.linalg.norm(force_data["wrench"][:3]) > 3 else NodeStatus.FAILURE
    
    def lift_cargo(ctx):
        ctx["agv"].lift_gripper(height=0.2)
        ctx["sim"].run_for(0.5)
        return NodeStatus.SUCCESS
    
    def transport_to_target(ctx):
        ctx["agv"].move_to(ctx["target_position"])
        ctx["sim"].run_for(3.0)
        return NodeStatus.SUCCESS
    
    def place_cargo(ctx):
        ctx["agv"].open_gripper()
        ctx["sim"].run_for(0.5)
        return NodeStatus.SUCCESS
    
    bt.add_node(BehaviorNode("approach_cargo", approach_cargo))
    bt.add_node(BehaviorNode("grasp_cargo", grasp_cargo))
    bt.add_node(BehaviorNode("verify_grasp_quality", verify_grasp_quality))
    bt.add_node(BehaviorNode("lift_cargo", lift_cargo))
    bt.add_node(BehaviorNode("transport_to_target", transport_to_target))
    bt.add_node(BehaviorNode("place_cargo", place_cargo))
    
    # Run complete task
    status = bt.run("handle_cargo", context)
    assert status == NodeStatus.SUCCESS
    
    # Verify cargo was moved
    cargo_pos = sim.get_object_position("cargo_01")
    assert np.linalg.norm(np.array(context["target_position"]) - np.array(cargo_pos[:3])) < 0.2
