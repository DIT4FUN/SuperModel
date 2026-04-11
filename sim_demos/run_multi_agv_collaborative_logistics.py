#!/usr/bin/env python3
"""
SuperModel 多AGV蜂群协同物流场景演示
Multi-AGV Swarm Collaborative Logistics Scenario Demo
v2.86.0
"""
import time
import random
import numpy as np
from typing import List, Dict
from sim_demos.base_sim import BaseSimulation, AGVRobot
from control.swarm_control import SwarmController
from control.behavior_tree import BehaviorTreePlanner
from sensors.imu import IMUSensor
from sensors.tactile import TactileSensor
from sensors.force import ForceSensor

class CollaborativeLogisticsSim(BaseSimulation):
    def __init__(self, num_agvs: int = 6):
        super().__init__(env_name="warehouse_logistics", size=(100, 80))
        self.num_agvs = num_agvs
        self.agvs: List[AGVRobot] = []
        self.swarm_controller = SwarmController(communication_range=20.0, collision_avoidance_radius=1.5)
        self.task_planner = BehaviorTreePlanner(scenario="logistics")
        
        # 场景配置
        self.pickup_points = [(10, 10), (10, 30), (10, 50), (10, 70)]
        self.dropoff_points = [(90, 10), (90, 30), (90, 50), (90, 70)]
        self.obstacles = self._generate_obstacles()
        self.task_queue = self._generate_initial_tasks(20)
        
        # 统计指标
        self.stats = {
            "completed_tasks": 0,
            "total_delivery_time": 0.0,
            "collision_avoidance_triggered": 0,
            "avg_task_completion_time": 0.0
        }
        
    def _generate_obstacles(self) -> List[tuple]:
        """生成仓库货架障碍物"""
        obstacles = []
        for x in range(20, 90, 15):
            for y in range(10, 70, 10):
                if random.random() < 0.7:
                    obstacles.append((x, y, 3, 2))  # x, y, width, height
        return obstacles
    
    def _generate_initial_tasks(self, count: int) -> List[Dict]:
        """生成初始物流任务"""
        tasks = []
        for i in range(count):
            pickup = random.choice(self.pickup_points)
            dropoff = random.choice(self.dropoff_points)
            while pickup == dropoff:
                dropoff = random.choice(self.dropoff_points)
            tasks.append({
                "task_id": f"T{i:03d}",
                "type": "delivery",
                "priority": random.randint(1, 5),
                "pickup_point": pickup,
                "dropoff_point": dropoff,
                "payload_weight": random.uniform(0.5, 10.0),
                "deadline": time.time() + random.uniform(60, 300),
                "assigned_agv": None,
                "status": "pending"
            })
        # 按优先级排序
        tasks.sort(key=lambda x: -x["priority"])
        return tasks
    
    def initialize_agvs(self):
        """初始化AGV车队"""
        for i in range(self.num_agvs):
            start_pos = (random.randint(20, 80), random.randint(10, 70))
            agv = AGVRobot(
                robot_id=f"AGV{i+1:02d}",
                grade=random.choice([3,4,5]),  # 3-5级AGV混合编队
                position=np.array([start_pos[0], start_pos[1], 0.0]),
                sensors={
                    "imu": IMUSensor(),
                    "tactile": TactileSensor(array_size=8),
                    "force": ForceSensor()
                }
            )
            self.agvs.append(agv)
            self.swarm_controller.add_robot(agv)
        print(f"✅ 初始化 {self.num_agvs} 台AGV完成")
    
    def assign_tasks(self):
        """动态任务分配"""
        available_agvs = [agv for agv in self.agvs if agv.status == "idle"]
        pending_tasks = [t for t in self.task_queue if t["status"] == "pending"]
        
        if not available_agvs or not pending_tasks:
            return
        
        # 基于距离和优先级的任务分配
        for agv in available_agvs:
            if not pending_tasks:
                break
            # 优先分配高优先级任务
            task = pending_tasks.pop(0)
            task["assigned_agv"] = agv.robot_id
            task["status"] = "assigned"
            agv.assign_task(task)
            print(f"📋 分配任务 {task['task_id']} 给 {agv.robot_id} (优先级: {task['priority']})")
    
    def run_simulation(self, duration: int = 300):
        """运行仿真"""
        print("\n🚚 启动多AGV协同物流场景仿真")
        print(f"📦 初始任务数: {len(self.task_queue)} | AGV数量: {self.num_agvs}")
        print("="*60)
        
        start_time = time.time()
        last_task_assign_time = start_time
        
        while time.time() - start_time < duration:
            current_time = time.time()
            
            # 每5秒重新分配一次任务
            if current_time - last_task_assign_time > 5.0:
                self.assign_tasks()
                last_task_assign_time = current_time
            
            # 更新AGV状态
            collision_events = self.swarm_controller.update()
            if collision_events:
                self.stats["collision_avoidance_triggered"] += len(collision_events)
                for event in collision_events:
                    print(f"⚠️  避障触发: {event['robot1']} 与 {event['robot2']} 距离 {event['distance']:.2f}m")
            
            # 检查任务完成情况
            for agv in self.agvs:
                if agv.status == "task_completed":
                    task = agv.completed_task
                    self.stats["completed_tasks"] += 1
                    delivery_time = current_time - task["assigned_time"]
                    self.stats["total_delivery_time"] += delivery_time
                    self.stats["avg_task_completion_time"] = self.stats["total_delivery_time"] / self.stats["completed_tasks"]
                    print(f"✅ {agv.robot_id} 完成任务 {task['task_id']} | 耗时: {delivery_time:.1f}s")
                    agv.status = "idle"
            
            # 打印实时统计
            if int(current_time) % 10 == 0:
                pending = len([t for t in self.task_queue if t["status"] == "pending"])
                running = len([agv for agv in self.agvs if agv.status == "executing"])
                print(f"\n📊 实时状态: 已完成={self.stats['completed_tasks']} | 待分配={pending} | 执行中={running} | 避障次数={self.stats['collision_avoidance_triggered']}")
            
            # 检查是否所有任务完成
            if self.stats["completed_tasks"] == len(self.task_queue):
                print("\n🎉 所有任务已完成!")
                break
            
            time.sleep(0.1)
        
        # 最终统计
        print("\n" + "="*60)
        print("📈 仿真结束 统计报告:")
        print(f"总任务数: {len(self.task_queue)} | 完成数: {self.stats['completed_tasks']} | 完成率: {self.stats['completed_tasks']/len(self.task_queue)*100:.1f}%")
        print(f"平均任务完成时间: {self.stats['avg_task_completion_time']:.1f}s")
        print(f"避障触发次数: {self.stats['collision_avoidance_triggered']}")
        print(f"总运行时间: {time.time() - start_time:.1f}s")
        
        # 保存仿真结果
        self._save_simulation_results()
        return self.stats
    
    def _save_simulation_results(self):
        """保存仿真结果到文件"""
        import json
        result = {
            "timestamp": time.time(),
            "version": "v2.86.0",
            "scenario": "multi_agv_collaborative_logistics",
            "stats": self.stats,
            "num_agvs": self.num_agvs
        }
        with open(f"./data/sim_results/logistics_sim_{int(time.time())}.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # 运行演示
    sim = CollaborativeLogisticsSim(num_agvs=8)
    sim.initialize_agvs()
    sim.run_simulation(duration=120)
