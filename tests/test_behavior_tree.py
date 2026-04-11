"""
Test cases for Behavior Tree Task Planning System
"""

import pytest
import math
import sys
sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel')
from src.embodied.behavior_tree import (
    NodeStatus, BehaviorTreeBuilder
)
from control.planner import (
    IsAtTarget, IsBatteryLow, MoveTo, Pickup
)


def test_condition_nodes():
    """测试条件节点"""
    # 测试IsAtTarget
    cond = IsAtTarget(target=(1.0, 1.0), tolerance=0.1)
    cond.blackboard = {"current_x": 1.05, "current_y": 1.0}
    assert cond.tick() == NodeStatus.SUCCESS
    
    cond.blackboard = {"current_x": 2.0, "current_y": 1.0}
    assert cond.tick() == NodeStatus.FAILURE
    
    # 测试IsBatteryLow
    cond = IsBatteryLow(threshold=0.2)
    cond.blackboard = {"battery_level": 0.15}
    assert cond.tick() == NodeStatus.SUCCESS
    
    cond.blackboard = {"battery_level": 0.3}
    assert cond.tick() == NodeStatus.FAILURE


def test_move_to_action():
    """测试MoveTo动作节点"""
    action = MoveTo(target=(1.0, 0.0))
    action.blackboard = {
        "current_x": 0.0,
        "current_y": 0.0,
        "current_theta": 0.0,
        "current_time": 0.0
    }
    
    # 第一次执行，生成轨迹
    status = action.tick()
    assert status == NodeStatus.RUNNING
    assert "current_trajectory" in action.blackboard
    assert action.blackboard["desired_velocity"] > 0.0
    
    # 模拟时间推进，移动到目标附近
    for t in range(0, 20):
        action.blackboard["current_time"] = t * 0.1
        # 模拟位置更新
        action.blackboard["current_x"] = min(1.0, action.blackboard["current_x"] + action.blackboard["desired_velocity"] * 0.1)
        status = action.tick()
        if status == NodeStatus.SUCCESS:
            break
    
    assert status == NodeStatus.SUCCESS
    assert abs(action.blackboard["current_x"] - 1.0) < 0.1


def test_warehouse_transfer_task():
    """测试仓库搬运任务行为树"""
    bt = BehaviorTreeBuilder.create_warehouse_transfer_task(
        pick_location=(2.0, 0.0),
        place_location=(5.0, 0.0),
        charge_station=(0.0, 0.0)
    )
    
    # 初始化黑板
    bt.blackboard = {
        "current_x": 0.0,
        "current_y": 0.0,
        "current_theta": 0.0,
        "current_time": 0.0,
        "battery_level": 0.8,
        "obstacles": []
    }
    
    # 执行前几步：从充电站出发去取货点
    success = False
    for step in range(0, 100):
        bt.blackboard["current_time"] = step * 0.1
        # 模拟位置更新
        v = bt.blackboard.get("desired_velocity", 0.0)
        omega = bt.blackboard.get("desired_omega", 0.0)
        
        dt = 0.1
        bt.blackboard["current_x"] += v * math.cos(bt.blackboard["current_theta"]) * dt
        bt.blackboard["current_y"] += v * math.sin(bt.blackboard["current_theta"]) * dt
        bt.blackboard["current_theta"] += omega * dt
        
        # 角度归一化
        while bt.blackboard["current_theta"] > math.pi:
            bt.blackboard["current_theta"] -= 2 * math.pi
        
        status = bt.tick()
        
        # 检查是否到达取货点
        if abs(bt.blackboard["current_x"] - 2.0) < 0.1:
            success = True
            break
    
    assert success == True
    assert bt.blackboard["gripper_command"] == "close"  # 开始抓取


def test_patrol_task():
    """测试巡逻任务行为树"""
    bt = BehaviorTreeBuilder.create_patrol_task(
        patrol_points=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    )
    
    bt.blackboard = {
        "current_x": 0.0,
        "current_y": 0.0,
        "current_theta": 0.0,
        "current_time": 0.0,
        "obstacles": []
    }
    
    visited_points = set()
    
    for step in range(0, 500):
        bt.blackboard["current_time"] = step * 0.1
        v = bt.blackboard.get("desired_velocity", 0.5)
        omega = bt.blackboard.get("desired_omega", 0.0)
        
        dt = 0.1
        bt.blackboard["current_x"] += v * math.cos(bt.blackboard["current_theta"]) * dt
        bt.blackboard["current_y"] += v * math.sin(bt.blackboard["current_theta"]) * dt
        bt.blackboard["current_theta"] += omega * dt
        
        status = bt.tick()
        
        # 记录访问过的点位
        x, y = bt.blackboard["current_x"], bt.blackboard["current_y"]
        for (px, py) in [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]:
            if abs(x - px) < 0.2 and abs(y - py) < 0.2:
                visited_points.add((px, py))
        
        if len(visited_points) >= 4:
            break
    
    assert len(visited_points) >= 3  # 至少访问3个巡逻点


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
