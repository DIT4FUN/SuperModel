import pytest
from embodiment.behavior_tree_engine import BehaviorTreeEngine, BehaviorNode, NodeStatus

def test_behavior_tree_conditional_nodes():
    """Test behavior tree conditional nodes for AGV tasks"""
    bt = BehaviorTreeEngine()
    
    # Add conditional nodes
    bt.add_node(BehaviorNode("battery_low", lambda ctx: ctx.get("battery", 100) < 20))
    bt.add_node(BehaviorNode("obstacle_detected", lambda ctx: ctx.get("obstacle_distance", 10) < 0.5))
    bt.add_node(BehaviorNode("target_reached", lambda ctx: ctx.get("distance_to_target", 5) < 0.1))
    
    # Test battery low condition
    ctx = {"battery": 15}
    assert bt.evaluate_node("battery_low", ctx) == NodeStatus.SUCCESS
    
    ctx = {"battery": 30}
    assert bt.evaluate_node("battery_low", ctx) == NodeStatus.FAILURE

def test_behavior_tree_sequence():
    """Test behavior tree sequence execution"""
    bt = BehaviorTreeEngine()
    
    steps = []
    def step1(ctx): steps.append(1); return NodeStatus.SUCCESS
    def step2(ctx): steps.append(2); return NodeStatus.SUCCESS
    def step3(ctx): steps.append(3); return NodeStatus.SUCCESS
    
    bt.add_sequence("navigate_sequence", ["check_safety", "move_to_target", "confirm_arrival"])
    bt.add_node(BehaviorNode("check_safety", step1))
    bt.add_node(BehaviorNode("move_to_target", step2))
    bt.add_node(BehaviorNode("confirm_arrival", step3))
    
    status = bt.run("navigate_sequence", {})
    assert status == NodeStatus.SUCCESS
    assert steps == [1, 2, 3]

def test_behavior_tree_fallback():
    """Test behavior tree fallback (selector) execution"""
    bt = BehaviorTreeEngine()
    
    tried = []
    def fail_action(ctx): tried.append("fail"); return NodeStatus.FAILURE
    def success_action(ctx): tried.append("success"); return NodeStatus.SUCCESS
    
    bt.add_fallback("navigation_fallback", ["primary_nav", "secondary_nav", "emergency_stop"])
    bt.add_node(BehaviorNode("primary_nav", fail_action))
    bt.add_node(BehaviorNode("secondary_nav", fail_action))
    bt.add_node(BehaviorNode("emergency_stop", success_action))
    
    status = bt.run("navigation_fallback", {})
    assert status == NodeStatus.SUCCESS
    assert tried == ["fail", "fail", "success"]

def test_behavior_tree_decorator():
    """Test behavior tree decorator nodes"""
    bt = BehaviorTreeEngine()
    
    call_count = 0
    def count_action(ctx):
        nonlocal call_count
        call_count +=1
        return NodeStatus.SUCCESS if call_count >=3 else NodeStatus.FAILURE
    
    bt.add_node(BehaviorNode("repeat_action", count_action))
    bt.add_decorator("retry_3_times", "repeat_action", max_retries=3)
    
    status = bt.run("retry_3_times", {})
    assert status == NodeStatus.SUCCESS
    assert call_count == 3

def test_behavior_tree_agv_task_scenarios():
    """Test complete AGV task behavior tree scenarios"""
    bt = BehaviorTreeEngine()
    
    # Build complete delivery task tree
    bt.add_sequence("delivery_task", [
        "check_battery",
        "navigate_to_pickup",
        "load_cargo",
        "navigate_to_delivery",
        "unload_cargo",
        "return_to_base"
    ])
    
    # Mock nodes
    ctx = {"battery": 80, "cargo_loaded": False, "delivery_complete": False}
    
    def check_battery(c): return NodeStatus.SUCCESS if c["battery"] > 20 else NodeStatus.FAILURE
    def nav_pickup(c): return NodeStatus.SUCCESS
    def load_cargo(c): c["cargo_loaded"] = True; return NodeStatus.SUCCESS
    def nav_delivery(c): return NodeStatus.SUCCESS
    def unload_cargo(c): c["cargo_loaded"] = False; c["delivery_complete"] = True; return NodeStatus.SUCCESS
    def return_base(c): return NodeStatus.SUCCESS
    
    bt.add_node(BehaviorNode("check_battery", check_battery))
    bt.add_node(BehaviorNode("navigate_to_pickup", nav_pickup))
    bt.add_node(BehaviorNode("load_cargo", load_cargo))
    bt.add_node(BehaviorNode("navigate_to_delivery", nav_delivery))
    bt.add_node(BehaviorNode("unload_cargo", unload_cargo))
    bt.add_node(BehaviorNode("return_to_base", return_base))
    
    status = bt.run("delivery_task", ctx)
    assert status == NodeStatus.SUCCESS
    assert ctx["delivery_complete"] == True
    assert ctx["cargo_loaded"] == False
