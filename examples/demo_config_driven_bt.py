#!/usr/bin/env python3
"""
demo_config_driven_bt.py - 配置驱动的行为树演示

展示如何从JSON/YAML配置文件加载行为树，
并使用AGV任务规划器执行具身任务。
"""

import json
import time
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embodied.behavior_tree import (
    create_behavior_tree_from_dict,
    create_task_bt_from_config,
    load_behavior_tree_from_json,
    BehaviorTree,
    AGVTaskPlanner,
    EmbodiedTask,
    NodeStatus,
)


def demo_from_json_file():
    """演示从JSON文件加载行为树"""
    print("\n" + "=" * 60)
    print("演示1: 从JSON文件加载行为树")
    print("=" * 60)

    config_path = os.path.join(
        os.path.dirname(__file__),
        "behavior_tree_config.json"
    )

    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = json.load(f)

    # 加载导航任务
    nav_config = full_config['task_configs'][0]
    bt = create_task_bt_from_config(nav_config)

    print(f"任务: {nav_config['task_name']}")
    print(f"行为树统计: {bt.get_statistics()}")

    # 模拟执行
    bt.update_robot_state({
        'safety': True,
        'battery_level': 0.8,
        'position': np.array([0.0, 0.0, 0.0]),
    })
    bt.set_goal({'target_position': np.array([3.0, 0.0, 0.0])})

    steps = 0
    while bt.tick() == NodeStatus.RUNNING and steps < 50:
        steps += 1

    print(f"执行完成: {steps} 步, 状态={bt.last_status}")


def demo_from_dict():
    """演示从字典配置直接构建"""
    print("\n" + "=" * 60)
    print("演示2: 从字典配置直接构建行为树")
    print("=" * 60)

    config = {
        "type": "sequence",
        "name": "WarehouseNav",
        "children": [
            {"type": "agv_check_safe", "name": "Safety"},
            {"type": "agv_check_battery", "params": {"min_battery": 0.2}, "name": "Battery"},
            {
                "type": "selector",
                "name": "ActionChoice",
                "children": [
                    {"type": "agv_move_to", "params": {"speed": 1.0}, "name": "FastMove"},
                    {"type": "lambda", "name": "Fallback", "action_name": "release"},
                ]
            },
        ]
    }

    root = create_behavior_tree_from_dict(config)
    bt = BehaviorTree(root, name="DictConfig")

    print(f"构建成功: {bt.get_statistics()}")
    print(f"根节点: {bt.root.name} ({bt.root.__class__.__name__})")
    print(f"第二层: {[c.name for c in bt.root.children]}")


def demo_agv_planner_with_config():
    """演示AGV规划器使用配置驱动"""
    print("\n" + "=" * 60)
    print("演示3: AGV规划器 + 配置驱动行为树")
    print("=" * 60)

    config_path = os.path.join(
        os.path.dirname(__file__),
        "behavior_tree_config.json"
    )

    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = json.load(f)

    # 创建规划器
    planner = AGVTaskPlanner(grade="L")

    # 从配置注册所有任务
    for task_cfg in full_config['task_configs']:
        root = create_behavior_tree_from_dict(task_cfg['tree'])
        planner.register_task_type(task_cfg['task_type'], root)
        print(f"  注册任务类型: {task_cfg['task_type']} ({task_cfg['task_name']})")

    print(f"\n规划器状态: {planner.get_status()}")

    # 添加导航任务
    task = EmbodiedTask(
        task_id='demo_nav_001',
        task_type='navigate',
        goal_description='导航到目标点',
        target_position=np.array([5.0, 0.0, 0.0]),
        priority=1,
    )
    planner.add_task(task)

    # 执行几个tick
    print("\n执行任务...")
    for i in range(5):
        status = planner.tick(
            robot_state={
                'safety': True,
                'battery_level': 0.8,
                'position': [float(i * 1.2), 0.0, 0.0],
            },
            world_state={}
        )
        print(f"  Tick {i+1}: {status.value}")
        if status in (NodeStatus.SUCCESS, NodeStatus.FAILURE):
            break


def demo_swarm_config():
    """演示蜂群任务配置"""
    print("\n" + "=" * 60)
    print("演示4: 蜂群协同搬运任务配置")
    print("=" * 60)

    config = {
        "type": "sequence",
        "name": "SwarmDemo",
        "children": [
            {"type": "agv_check_safe", "name": "SafetyCheck"},
            {"type": "agv_check_battery", "params": {"min_battery": 0.4}},
            {"type": "agv_negotiate_role", "name": "RoleNegotiation"},
            {
                "type": "parallel",
                "name": "FormationMove",
                "success_threshold": 2,
                "children": [
                    {"type": "agv_move_to_formation", "name": "AGV1_Formation"},
                    {"type": "agv_move_to_formation", "name": "AGV2_Formation"},
                ]
            },
            {"type": "agv_parallel_grasp"},
            {"type": "agv_coordinated_move"},
            {"type": "agv_parallel_release"},
        ]
    }

    root = create_behavior_tree_from_dict(config)
    bt = BehaviorTree(root, name="SwarmTask")

    stats = bt.get_statistics()
    print(f"蜂群任务统计: {stats}")
    print(f"节点类型分布: {stats['node_types']}")


if __name__ == '__main__':
    print("=" * 60)
    print("SuperModel 配置驱动行为树演示")
    print("=" * 60)

    demo_from_dict()
    demo_from_json_file()
    demo_agv_planner_with_config()
    demo_swarm_config()

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
