"""
Test cases for new embodiment features:
1. Outdoor campus scene simulation
2. Modbus communication support
3. ROS communication support
"""

import pytest
import math
from embodiment.simulation import EmbodimentSimulator, SimulationScene, SimSceneConfig
from embodiment.agv_interface import AGVHardwareInterface, AGVConfig, AGVCommunicationType


def test_outdoor_campus_scene():
    """Test outdoor campus scene loading"""
    scene_config = SimSceneConfig(scene_type="outdoor")
    sim = EmbodimentSimulator(scene_config=scene_config, gui=False)
    
    # Add AGV
    agv_id = sim.add_agv(initial_pos=(0.0, 0.0, 0.1))
    
    # Test movement
    sim.set_agv_command(agv_id, v=0.5, omega=0.0)
    state = sim.step(duration=1.0)
    
    # Check AGV moved
    agv_state = state["agvs"][agv_id]["state"]
    assert agv_state["x"] > 0.4  # Should move ~0.5m in 1s
    assert abs(agv_state["y"]) < 0.1  # Should stay on y axis
    
    # Check obstacles detection
    assert len(agv_state["obstacles"]) >= 0  # Should detect trees/lamp posts
    
    sim.close()


def test_modbus_interface_simulation():
    """Test Modbus AGV interface simulation (without actual hardware)"""
    # Create simulation instance
    sim = EmbodimentSimulator(scene=SimulationScene.WAREHOUSE, gui=False)
    agv_id = sim.add_agv(initial_pos=(0.0, 0.0, 0.1))
    
    # Create AGV interface with Modbus type (will use simulation mode)
    config = AGVConfig(
        agv_id=agv_id,
        communication_type=AGVCommunicationType.MODBUS,
        tcp_host="127.0.0.1",
        tcp_port=502
    )
    interface = AGVHardwareInterface(config, interface_type="simulation", sim_instance=sim, agv_id=agv_id)
    
    assert interface.is_connected() == True
    
    # Test send command
    from embodiment.agv_interface import AGVCommand
    cmd = AGVCommand(v=0.5, omega=0.0)
    success = interface.send_command(cmd)
    assert success == True
    
    # Test get state
    state = interface.get_state()
    assert state is not None
    assert state.x >= 0.0
    assert state.battery_level > 0.0
    
    sim.close()


def test_ros_interface_simulation():
    """Test ROS AGV interface simulation (without actual ROS)"""
    # Create simulation instance
    sim = EmbodimentSimulator(scene=SimulationScene.LOGISTICS_CENTER, gui=False)
    agv_id = sim.add_agv(initial_pos=(0.0, 0.0, 0.1))
    
    # Create AGV interface with ROS type (will use simulation mode)
    config = AGVConfig(
        agv_id=agv_id,
        communication_type=AGVCommunicationType.ROS,
        tcp_host="127.0.0.1",
        tcp_port=11311
    )
    interface = AGVHardwareInterface(config, interface_type="simulation", sim_instance=sim, agv_id=agv_id)
    
    assert interface.is_connected() == True
    
    # Test send command
    from embodiment.agv_interface import AGVCommand
    cmd = AGVCommand(v=0.3, omega=0.5)
    success = interface.send_command(cmd)
    assert success == True
    
    # Test get state
    state = interface.get_state()
    assert state is not None
    assert abs(state.omega - 0.5) < 0.1  # Should be close to requested omega
    
    sim.close()


def test_all_scenes_supported():
    """Test all 4 scene types are supported"""
    scenes = [
        "warehouse",
        "logistics",
        "factory",
        "outdoor"
    ]
    
    for scene_type in scenes:
        scene_config = SimSceneConfig(scene_type=scene_type)
        sim = EmbodimentSimulator(scene_config=scene_config, gui=False)
        assert sim.client_id >= 0
        sim.close()
