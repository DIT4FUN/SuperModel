"""
Test cases for Behavior Tree Engine
"""
import time
import pytest
from embodiment.behavior_tree_engine import BehaviorTreeEngine
from control.planner import BehaviorNode, NodeStatus


class TestNode(BehaviorNode):
    """Test behavior node that increments a counter each tick"""
    def __init__(self, name: str = "TestNode"):
        super().__init__(name)
        self.tick_count = 0

    def tick(self) -> NodeStatus:
        self.tick_count += 1
        self.blackboard["tick_count"] = self.tick_count
        if self.tick_count >= 5:
            return NodeStatus.SUCCESS
        return NodeStatus.RUNNING
        
    def reset(self):
        """Reset tick count"""
        super().reset()
        self.tick_count = 0


class TestSuccessNode(BehaviorNode):
    """Test node that always returns success"""
    def tick(self) -> NodeStatus:
        self.blackboard["test_success"] = True
        return NodeStatus.SUCCESS


class TestFailureNode(BehaviorNode):
    """Test node that always returns failure"""
    def tick(self) -> NodeStatus:
        self.blackboard["test_failure"] = True
        return NodeStatus.FAILURE


def test_behavior_tree_initialization():
    """Test engine initialization"""
    test_node = TestNode()
    engine = BehaviorTreeEngine(test_node, update_rate=10.0)
    
    assert engine.is_running() is False
    assert engine.get_blackboard_value("current_x") == 0.0
    assert engine.get_blackboard_value("battery_level") == 1.0
    assert engine.get_blackboard_value("task_completed") is False
    
    stats = engine.get_stats()
    assert stats["total_ticks"] == 0
    assert stats["success_count"] == 0


def test_set_state():
    """Test setting AGV state to blackboard"""
    test_node = TestNode()
    engine = BehaviorTreeEngine(test_node)
    
    engine.set_state(x=1.0, y=2.0, theta=0.5, battery_level=0.75, obstacles=[(3.0, 4.0)])
    
    assert engine.get_blackboard_value("current_x") == 1.0
    assert engine.get_blackboard_value("current_y") == 2.0
    assert engine.get_blackboard_value("current_theta") == 0.5
    assert engine.get_blackboard_value("battery_level") == 0.75
    assert engine.get_blackboard_value("obstacles") == [(3.0, 4.0)]


def test_single_tick():
    """Test single tick execution"""
    test_node = TestNode()
    engine = BehaviorTreeEngine(test_node)
    
    status = engine.tick()
    assert status == NodeStatus.RUNNING
    assert test_node.tick_count == 1
    assert engine.get_blackboard_value("tick_count") == 1
    
    stats = engine.get_stats()
    assert stats["total_ticks"] == 1
    assert stats["running_count"] == 1


def test_tick_until_success():
    """Test ticking until node returns success"""
    test_node = TestNode()
    engine = BehaviorTreeEngine(test_node)
    
    for i in range(4):
        status = engine.tick()
        assert status == NodeStatus.RUNNING
        assert test_node.tick_count == i + 1
    
    # 5th tick should return success
    status = engine.tick()
    assert status == NodeStatus.SUCCESS
    assert test_node.tick_count == 5
    
    stats = engine.get_stats()
    assert stats["total_ticks"] == 5
    assert stats["running_count"] == 4
    assert stats["success_count"] == 1


def test_reset():
    """Test engine reset functionality"""
    test_node = TestNode()
    engine = BehaviorTreeEngine(test_node)
    
    # Tick a few times
    for _ in range(3):
        engine.tick()
    
    assert test_node.tick_count == 3
    stats = engine.get_stats()
    assert stats["total_ticks"] == 3
    
    # Reset
    engine.reset()
    
    assert test_node.tick_count == 0  # Node reset
    stats = engine.get_stats()
    assert stats["total_ticks"] == 0
    assert engine.get_blackboard_value("task_completed") is False
    assert engine.get_blackboard_value("desired_velocity") == 0.0


def test_background_run():
    """Test running engine in background thread"""
    test_node = TestNode()
    engine = BehaviorTreeEngine(test_node, update_rate=50.0)  # 50Hz
    
    engine.start(background=True)
    assert engine.is_running() is True
    
    # Let it run for 0.2 seconds (~10 ticks)
    time.sleep(0.2)
    
    engine.stop()
    assert engine.is_running() is False
    
    stats = engine.get_stats()
    assert stats["total_ticks"] >= 5  # Should have at least 5 ticks
    assert test_node.tick_count >= 5


def test_control_output():
    """Test control output retrieval"""
    test_node = TestSuccessNode()
    engine = BehaviorTreeEngine(test_node)
    
    # Set desired control values
    engine.set_blackboard_value("desired_velocity", 0.5)
    engine.set_blackboard_value("desired_omega", 0.2)
    engine.set_blackboard_value("gripper_command", "close")
    
    v, omega, gripper = engine.get_control_output()
    assert v == 0.5
    assert omega == 0.2
    assert gripper == "close"


def test_success_failure_tracking():
    """Test success and failure count tracking"""
    success_node = TestSuccessNode()
    failure_node = TestFailureNode()
    
    # Test success
    engine1 = BehaviorTreeEngine(success_node)
    engine1.tick()
    stats1 = engine1.get_stats()
    assert stats1["success_count"] == 1
    assert stats1["failure_count"] == 0
    
    # Test failure
    engine2 = BehaviorTreeEngine(failure_node)
    engine2.tick()
    stats2 = engine2.get_stats()
    assert stats2["success_count"] == 0
    assert stats2["failure_count"] == 1


def test_blackboard_operations():
    """Test custom blackboard value operations"""
    test_node = TestNode()
    engine = BehaviorTreeEngine(test_node)
    
    # Set custom value
    engine.set_blackboard_value("custom_key", "custom_value")
    assert engine.get_blackboard_value("custom_key") == "custom_value"
    
    # Get non-existent key with default
    assert engine.get_blackboard_value("non_existent", "default") == "default"
    assert engine.get_blackboard_value("non_existent") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
