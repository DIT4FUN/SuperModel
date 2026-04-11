#!/usr/bin/env python3
"""
多AGV蜂群协同物流场景测试用例
Test cases for multi-AGV swarm collaborative logistics scenario
v2.86.0
"""
import pytest
import time
import numpy as np
from sim_demos.run_multi_agv_collaborative_logistics import CollaborativeLogisticsSim

class TestMultiAGVCollaborativeLogistics:
    @pytest.fixture
    def sim(self):
        """初始化仿真环境fixture"""
        sim = CollaborativeLogisticsSim(num_agvs=4)
        sim.initialize_agvs()
        yield sim
    
    def test_simulation_initialization(self, sim):
        """测试仿真环境初始化"""
        assert len(sim.agvs) == 4
        assert len(sim.task_queue) == 20
        assert len(sim.pickup_points) == 4
        assert len(sim.dropoff_points) == 4
        assert len(sim.obstacles) > 0
    
    def test_task_generation(self, sim):
        """测试任务生成逻辑"""
        tasks = sim._generate_initial_tasks(10)
        assert len(tasks) == 10
        # 检查任务字段完整性
        for task in tasks:
            assert "task_id" in task
            assert "priority" in task
            assert "pickup_point" in task
            assert "dropoff_point" in task
            assert "payload_weight" in task
            assert task["pickup_point"] != task["dropoff_point"]
            assert 1 <= task["priority"] <= 5
    
    def test_task_assignment(self, sim):
        """测试动态任务分配"""
        # 初始状态下所有AGV都是空闲的
        idle_agvs = [agv for agv in sim.agvs if agv.status == "idle"]
        assert len(idle_agvs) == 4
        
        # 执行任务分配
        sim.assign_tasks()
        
        # 检查4个AGV都分配到了任务
        assigned_tasks = [t for t in sim.task_queue if t["status"] == "assigned"]
        assert len(assigned_tasks) == 4
        for task in assigned_tasks:
            assert task["assigned_agv"] is not None
        
        # 检查AGV状态变为已分配
        assigned_agvs = [agv for agv in sim.agvs if agv.status == "assigned"]
        assert len(assigned_agvs) == 4
    
    def test_collision_avoidance(self, sim):
        """测试碰撞避免机制"""
        # 将两个AGV放置到很近的位置
        sim.agvs[0].position = np.array([50.0, 50.0, 0.0])
        sim.agvs[1].position = np.array([50.5, 50.5, 0.0])
        
        # 执行 swarm controller 更新
        collision_events = sim.swarm_controller.update()
        
        # 应该检测到碰撞并触发避障
        assert len(collision_events) >= 1
        assert collision_events[0]["distance"] < 1.5
    
    def test_short_simulation_run(self, sim):
        """测试短时间仿真运行"""
        # 运行10秒仿真
        stats = sim.run_simulation(duration=10)
        
        # 检查统计数据正常
        assert stats["collision_avoidance_triggered"] >= 0
        assert stats["completed_tasks"] >= 0
        assert stats["total_delivery_time"] >= 0
    
    def test_priority_task_scheduling(self, sim):
        """测试优先级任务调度"""
        # 生成一个高优先级任务
        high_prio_task = {
            "task_id": "T_HIGH",
            "type": "delivery",
            "priority": 5,
            "pickup_point": (10, 10),
            "dropoff_point": (90, 90),
            "payload_weight": 2.0,
            "deadline": time.time() + 60,
            "assigned_agv": None,
            "status": "pending"
        }
        # 插入到任务队列最前面
        sim.task_queue.insert(0, high_prio_task)
        
        # 执行任务分配
        sim.assign_tasks()
        
        # 高优先级任务应该第一个被分配
        assigned_task = [t for t in sim.task_queue if t["task_id"] == "T_HIGH"][0]
        assert assigned_task["status"] == "assigned"
        assert assigned_task["assigned_agv"] is not None
    
    @pytest.mark.parametrize("num_agvs", [2, 4, 6, 8])
    def test_scalability(self, num_agvs):
        """测试不同AGV数量下的扩展性"""
        sim = CollaborativeLogisticsSim(num_agvs=num_agvs)
        sim.initialize_agvs()
        assert len(sim.agvs) == num_agvs
        
        # 运行5秒仿真
        stats = sim.run_simulation(duration=5)
        assert stats is not None
    
    def test_task_completion(self):
        """测试单个AGV任务完成流程"""
        sim = CollaborativeLogisticsSim(num_agvs=1)
        sim.initialize_agvs()
        agv = sim.agvs[0]
        
        # 分配一个简单任务
        test_task = {
            "task_id": "T_TEST",
            "type": "delivery",
            "priority": 3,
            "pickup_point": (10, 10),
            "dropoff_point": (15, 15),
            "payload_weight": 1.0,
            "deadline": time.time() + 30,
            "assigned_agv": agv.robot_id,
            "status": "assigned",
            "assigned_time": time.time()
        }
        agv.assign_task(test_task)
        agv.status = "executing"
        
        # 模拟AGV到达卸货点
        agv.position = np.array([15.0, 15.0, 0.0])
        agv._check_task_completion()
        
        assert agv.status == "task_completed"
        assert agv.completed_task["task_id"] == "T_TEST"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
