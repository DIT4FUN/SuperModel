#!/usr/bin/env python3
"""
agv_swarm_coordinator.py - Multi-AGV Swarm Coordination Module
SuperModel v2.66.0 - 2026-04-11
功能: 多AGV蜂群协同控制，包括任务分配、路径规划、冲突避免、状态同步
"""

import os
import sys
import time
import uuid
import heapq
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from scipy.spatial import KDTree
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.logger import logger
from control.agv_kinematics import AGVSpec, AGVState
from simulation.embodied_sim import WarehouseScene

class TaskPriority(Enum):
    """任务优先级"""
    P0_URGENT = 0
    P1_HIGH = 1
    P2_MEDIUM = 2
    P3_LOW = 3

class TaskStatus(Enum):
    """任务状态"""
    PENDING = 0
    ASSIGNED = 1
    IN_PROGRESS = 2
    COMPLETED = 3
    FAILED = 4
    CANCELLED = 5

@dataclass
class SwarmTask:
    """蜂群任务定义"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = "transport"  # transport/patrol/inspection/assembly
    priority: TaskPriority = TaskPriority.P2_MEDIUM
    source_point: Tuple[float, float, float] = (0, 0, 0)
    target_point: Tuple[float, float, float] = (0, 0, 0)
    payload: float = 0.0  # kg
    required_agv_spec: str = "M"  # S/M/L/XL/XXL
    deadline: float = 3600  # 秒，从现在开始算
    dependencies: List[str] = field(default_factory=list)  # 依赖任务ID
    status: TaskStatus = TaskStatus.PENDING
    assigned_agv_id: Optional[str] = None
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

@dataclass
class AGVSwarmMember:
    """蜂群AGV成员"""
    agv_id: str
    spec: AGVSpec
    current_state: AGVState
    current_task: Optional[SwarmTask] = None
    task_queue: List[SwarmTask] = field(default_factory=list)
    battery_level: float = 100.0  # %
    health_status: float = 100.0  # %
    last_heartbeat: float = field(default_factory=time.time)
    available: bool = True

@dataclass
class SwarmConflict:
    """蜂群冲突定义"""
    conflict_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conflict_type: str = "collision"  # collision/path_deadlock/resource_contention
    involved_agvs: List[str] = field(default_factory=list)
    location: Tuple[float, float, float] = (0, 0, 0)
    timestamp: float = field(default_factory=time.time)
    severity: int = 5  # 1-10
    resolved: bool = False
    resolution: Optional[str] = None

class AGVSwarmCoordinator:
    def __init__(self, scene: WarehouseScene, max_workers: int = 10):
        self.scene = scene
        self.max_workers = max_workers
        self.agvs: Dict[str, AGVSwarmMember] = {}
        self.tasks: Dict[str, SwarmTask] = {}
        self.conflicts: List[SwarmConflict] = []
        self.global_map: nx.Graph = self._build_global_map()
        self.kd_tree: Optional[KDTree] = None
        self.running = False
        self.simulation_time = 0.0
        self.swarm_metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_distance_traveled': 0.0,
            'total_energy_consumed': 0.0,
            'average_task_completion_time': 0.0,
            'conflict_count': 0
        }
    
    def _build_global_map(self) -> nx.Graph:
        """构建全局路径规划地图"""
        G = nx.Graph()
        # 添加节点（仓库的货架、出入口、充电位等）
        for node_id, point in self.scene.navigation_points.items():
            G.add_node(node_id, pos=point)
        # 添加边（可行走路径）
        for (u, v), distance in self.scene.path_segments.items():
            G.add_edge(u, v, weight=distance, speed_limit=1.5)  # 默认速度1.5m/s
        logger.info(f"全局地图构建完成: {len(G.nodes)}个节点, {len(G.edges)}条边")
        return G
    
    def _update_kd_tree(self) -> None:
        """更新AGV位置的KD树用于碰撞检测"""
        positions = []
        self.agv_id_to_index = {}
        for idx, (agv_id, agv) in enumerate(self.agvs.items()):
            pos = agv.current_state.pose[:3]
            positions.append(pos)
            self.agv_id_to_index[agv_id] = idx
        if positions:
            self.kd_tree = KDTree(np.array(positions))
        else:
            self.kd_tree = None
    
    def register_agv(self, agv_id: str, spec: AGVSpec, initial_state: AGVState) -> None:
        """注册AGV到蜂群"""
        self.agvs[agv_id] = AGVSwarmMember(
            agv_id=agv_id,
            spec=spec,
            current_state=initial_state
        )
        logger.info(f"AGV {agv_id} 已注册到蜂群，规格: {spec.size_class}")
        self._update_kd_tree()
    
    def unregister_agv(self, agv_id: str) -> None:
        """从蜂群注销AGV"""
        if agv_id in self.agvs:
            # 取消AGV当前任务
            agv = self.agvs[agv_id]
            if agv.current_task:
                agv.current_task.status = TaskStatus.PENDING
                agv.current_task.assigned_agv_id = None
            del self.agvs[agv_id]
            logger.info(f"AGV {agv_id} 已从蜂群注销")
            self._update_kd_tree()
    
    def add_task(self, task: SwarmTask) -> str:
        """添加任务到蜂群任务池"""
        self.tasks[task.task_id] = task
        logger.info(f"任务 {task.task_id} 已添加，类型: {task.task_type}, 优先级: {task.priority.name}")
        return task.task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id not in self.tasks:
            return False
        task = self.tasks[task_id]
        task.status = TaskStatus.CANCELLED
        # 如果任务已分配，回收
        if task.assigned_agv_id and task.assigned_agv_id in self.agvs:
            agv = self.agvs[task.assigned_agv_id]
            if agv.current_task and agv.current_task.task_id == task_id:
                agv.current_task = None
            agv.task_queue = [t for t in agv.task_queue if t.task_id != task_id]
        logger.info(f"任务 {task_id} 已取消")
        return True
    
    def calculate_task_allocation_score(self, agv: AGVSwarmMember, task: SwarmTask) -> float:
        """计算AGV分配任务的得分（越低越好）"""
        if not agv.available:
            return float('inf')
        
        # 检查规格是否匹配
        spec_order = ['S', 'M', 'L', 'XL', 'XXL']
        if spec_order.index(agv.spec.size_class) < spec_order.index(task.required_agv_spec):
            return float('inf')  # 规格不足
        
        # 检查载重
        if agv.spec.max_payload < task.payload:
            return float('inf')
        
        # 计算AGV到任务起点的距离
        agv_pos = agv.current_state.pose[:3]
        distance_to_source = np.linalg.norm(np.array(agv_pos) - np.array(task.source_point))
        
        # 计算任务预计耗时
        path_length = self.calculate_shortest_path_length(task.source_point, task.target_point)
        task_duration = path_length / agv.spec.max_speed + 60  # 加装卸货时间
        
        # 检查电池是否足够
        energy_required = task_duration * agv.spec.power_consumption_rate
        if agv.battery_level * agv.spec.battery_capacity * 0.01 < energy_required:
            return float('inf')
        
        # 计算等待时间（AGV当前队列的预计完成时间）
        queue_duration = 0.0
        if agv.current_task:
            remaining = (1 - agv.current_task.progress) * self._estimate_task_duration(agv.current_task)
            queue_duration += remaining
        for queued_task in agv.task_queue:
            queue_duration += self._estimate_task_duration(queued_task)
        
        # 检查是否能在截止时间前完成
        if queue_duration + task_duration > task.deadline:
            return float('inf')
        
        # 综合得分（加权）
        score = (
            distance_to_source * 0.4 +
            task_duration * 0.3 +
            queue_duration * 0.2 +
            task.priority.value * 10  # 优先级权重
        )
        
        return score
    
    def _estimate_task_duration(self, task: SwarmTask) -> float:
        """估算任务耗时"""
        path_length = self.calculate_shortest_path_length(task.source_point, task.target_point)
        return path_length / 1.0 + 60  # 默认速度1m/s，加1分钟装卸货
    
    def calculate_shortest_path_length(self, start: Tuple[float, float, float], end: Tuple[float, float, float]) -> float:
        """计算两点之间的最短路径长度"""
        # 找到最近的导航点
        start_node = self._find_nearest_navigation_point(start)
        end_node = self._find_nearest_navigation_point(end)
        
        try:
            return nx.shortest_path_length(self.global_map, start_node, end_node, weight='weight')
        except nx.NetworkXNoPath:
            # 如果没有路径，返回欧氏距离作为 fallback
            return np.linalg.norm(np.array(start) - np.array(end))
    
    def _find_nearest_navigation_point(self, point: Tuple[float, float, float]) -> str:
        """找到最近的导航点"""
        min_dist = float('inf')
        nearest_node = None
        for node_id, pos in self.scene.navigation_points.items():
            dist = np.linalg.norm(np.array(pos) - np.array(point))
            if dist < min_dist:
                min_dist = dist
                nearest_node = node_id
        return nearest_node
    
    def allocate_tasks(self) -> int:
        """全局任务分配（匈牙利算法变种）"""
        pending_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]
        available_agvs = [a for a in self.agvs.values() if a.available]
        
        if not pending_tasks or not available_agvs:
            return 0
        
        # 构建成本矩阵
        cost_matrix = []
        for task in pending_tasks:
            row = []
            for agv in available_agvs:
                score = self.calculate_task_allocation_score(agv, task)
                row.append(score)
            cost_matrix.append(row)
        
        # 贪心分配（优先分配高优先级任务）
        assigned_count = 0
        task_indices = sorted(range(len(pending_tasks)), key=lambda i: pending_tasks[i].priority.value)
        
        for task_idx in task_indices:
            task = pending_tasks[task_idx]
            # 找到得分最低的AGV
            min_score = float('inf')
            best_agv_idx = -1
            for agv_idx, agv in enumerate(available_agvs):
                if cost_matrix[task_idx][agv_idx] < min_score:
                    min_score = cost_matrix[task_idx][agv_idx]
                    best_agv_idx = agv_idx
            
            if best_agv_idx != -1 and min_score != float('inf'):
                agv = available_agvs[best_agv_idx]
                # 分配任务
                task.status = TaskStatus.ASSIGNED
                task.assigned_agv_id = agv.agv_id
                if not agv.current_task:
                    agv.current_task = task
                    task.started_at = time.time()
                else:
                    agv.task_queue.append(task)
                assigned_count += 1
                logger.info(f"任务 {task.task_id} 已分配给AGV {agv.agv_id}，得分: {min_score:.2f}")
        
        return assigned_count
    
    def detect_conflicts(self) -> List[SwarmConflict]:
        """检测蜂群冲突（碰撞、死锁、资源竞争）"""
        conflicts = []
        
        # 1. 碰撞检测
        if self.kd_tree and len(self.agvs) >= 2:
            # 查找距离小于安全距离的AGV对
            safety_distance = 0.5  # 50cm安全距离
            pairs = self.kd_tree.query_pairs(r=safety_distance)
            
            for (i, j) in pairs:
                agv_id1 = list(self.agvs.keys())[i]
                agv_id2 = list(self.agvs.keys())[j]
                pos1 = self.agvs[agv_id1].current_state.pose[:3]
                pos2 = self.agvs[agv_id2].current_state.pose[:3]
                center = ((pos1[0]+pos2[0])/2, (pos1[1]+pos2[1])/2, (pos1[2]+pos2[2])/2)
                
                conflict = SwarmConflict(
                    conflict_type="collision",
                    involved_agvs=[agv_id1, agv_id2],
                    location=center,
                    severity=8 if min(self.agvs[agv_id1].current_state.speed, self.agvs[agv_id2].current_state.speed) > 0.5 else 4
                )
                conflicts.append(conflict)
                self.swarm_metrics['conflict_count'] += 1
                logger.warning(f"检测到碰撞冲突: AGV {agv_id1} 和 {agv_id2}，距离: {np.linalg.norm(np.array(pos1)-np.array(pos2)):.2f}m")
        
        # 2. 死锁检测
        # 构建AGV等待图
        wait_graph = nx.DiGraph()
        for agv in self.agvs.values():
            if agv.current_state.waiting_for:
                for target_agv_id in agv.current_state.waiting_for:
                    wait_graph.add_edge(agv.agv_id, target_agv_id)
        
        # 检测环
        for cycle in nx.simple_cycles(wait_graph):
            if len(cycle) >= 2:
                conflict = SwarmConflict(
                    conflict_type="path_deadlock",
                    involved_agvs=cycle,
                    location=self.agvs[cycle[0]].current_state.pose[:3],
                    severity=9
                )
                conflicts.append(conflict)
                self.swarm_metrics['conflict_count'] += 1
                logger.warning(f"检测到路径死锁: 涉及AGV {cycle}")
        
        # 3. 资源竞争检测
        resource_usage: Dict[str, List[str]] = {}
        for agv in self.agvs.values():
            if agv.current_state.occupying_resource:
                res = agv.current_state.occupying_resource
                if res not in resource_usage:
                    resource_usage[res] = []
                resource_usage[res].append(agv.agv_id)
        
        for res, agv_ids in resource_usage.items():
            if len(agv_ids) > 1:
                conflict = SwarmConflict(
                    conflict_type="resource_contention",
                    involved_agvs=agv_ids,
                    location=self.scene.resources[res]['position'],
                    severity=6
                )
                conflicts.append(conflict)
                self.swarm_metrics['conflict_count'] += 1
                logger.warning(f"检测到资源竞争: {res} 被AGV {agv_ids} 同时占用")
        
        self.conflicts.extend(conflicts)
        return conflicts
    
    def resolve_conflicts(self, conflicts: List[SwarmConflict]) -> int:
        """解决冲突"""
        resolved_count = 0
        
        for conflict in conflicts:
            if conflict.resolved:
                continue
            
            if conflict.conflict_type == "collision":
                # 碰撞解决：优先级低的AGV停车让行
                agv1 = self.agvs[conflict.involved_agvs[0]]
                agv2 = self.agvs[conflict.involved_agvs[1]]
                # 优先级规则：有紧急任务 > 载重高 > ID小
                priority1 = agv1.current_task.priority.value if agv1.current_task else 99
                priority2 = agv2.current_task.priority.value if agv2.current_task else 99
                
                if priority1 < priority2 or (priority1 == priority2 and agv1.spec.max_payload > agv2.spec.max_payload):
                    # agv2让行
                    agv2.current_state.target_speed = 0.0
                    agv2.current_state.waiting_for = [agv1.agv_id]
                    conflict.resolution = f"AGV {agv2.agv_id} 停车让行AGV {agv1.agv_id}"
                else:
                    # agv1让行
                    agv1.current_state.target_speed = 0.0
                    agv1.current_state.waiting_for = [agv2.agv_id]
                    conflict.resolution = f"AGV {agv1.agv_id} 停车让行AGV {agv2.agv_id}"
                
                conflict.resolved = True
                resolved_count += 1
                logger.info(f"碰撞冲突已解决: {conflict.resolution}")
            
            elif conflict.conflict_type == "path_deadlock":
                # 死锁解决：最低优先级的AGV重新规划路径
                agvs = [self.agvs[agv_id] for agv_id in conflict.involved_agvs]
                # 找优先级最低的AGV
                lowest_prio_agv = max(agvs, key=lambda a: a.current_task.priority.value if a.current_task else 99)
                # 重新规划路径
                if lowest_prio_agv.current_task:
                    new_path = self.plan_path_for_agv(lowest_prio_agv, lowest_prio_agv.current_task.target_point)
                    lowest_prio_agv.current_state.path = new_path
                    lowest_prio_agv.current_state.waiting_for = []
                    conflict.resolution = f"AGV {lowest_prio_agv.agv_id} 重新规划路径解开死锁"
                    conflict.resolved = True
                    resolved_count += 1
                    logger.info(f"死锁冲突已解决: {conflict.resolution}")
            
            elif conflict.conflict_type == "resource_contention":
                # 资源竞争解决：按优先级顺序分配
                agvs = [self.agvs[agv_id] for agv_id in conflict.involved_agvs]
                # 按优先级排序
                agvs.sort(key=lambda a: a.current_task.priority.value if a.current_task else 99)
                # 最高优先级的保留资源，其他重新规划
                for agv in agvs[1:]:
                    agv.current_state.occupying_resource = None
                    agv.current_state.waiting_for = [agvs[0].agv_id]
                conflict.resolution = f"资源分配给AGV {agvs[0].agv_id}，其他AGV等待"
                conflict.resolved = True
                resolved_count += 1
                logger.info(f"资源竞争冲突已解决: {conflict.resolution}")
        
        return resolved_count
    
    def plan_path_for_agv(self, agv: AGVSwarmMember, target: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """为AGV规划路径（A*算法）"""
        start = agv.current_state.pose[:3]
        start_node = self._find_nearest_navigation_point(start)
        end_node = self._find_nearest_navigation_point(target)
        
        try:
            path_nodes = nx.astar_path(self.global_map, start_node, end_node, weight='weight')
            # 转换为坐标点
            path = [self.scene.navigation_points[node] for node in path_nodes]
            # 添加起点和终点
            if np.linalg.norm(np.array(path[0]) - np.array(start)) > 0.1:
                path.insert(0, start)
            if np.linalg.norm(np.array(path[-1]) - np.array(target)) > 0.1:
                path.append(target)
            return path
        except nx.NetworkXNoPath:
            logger.error(f"AGV {agv.agv_id} 无法找到到 {target} 的路径")
            return []
    
    def sync_agv_states(self) -> None:
        """同步所有AGV状态"""
        current_time = time.time()
        for agv in self.agvs.values():
            agv.last_heartbeat = current_time
            # 检查任务进度
            if agv.current_task:
                # 更新进度
                if agv.current_task.status == TaskStatus.ASSIGNED:
                    agv.current_task.status = TaskStatus.IN_PROGRESS
                    agv.current_task.started_at = current_time
                
                # 计算进度（基于已行驶距离/总距离）
                if agv.current_state.path:
                    traveled = np.linalg.norm(np.array(agv.current_state.pose[:3]) - np.array(agv.current_state.path[0]))
                    total = sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) for i in range(len(path)-1))
                    if total > 0:
                        agv.current_task.progress = min(traveled / total, 1.0)
                
                # 检查是否完成
                if agv.current_task.progress >= 1.0:
                    agv.current_task.status = TaskStatus.COMPLETED
                    agv.current_task.completed_at = current_time
                    self.swarm_metrics['tasks_completed'] += 1
                    completion_time = agv.current_task.completed_at - agv.current_task.started_at
                    # 更新平均完成时间
                    total = self.swarm_metrics['average_task_completion_time'] * (self.swarm_metrics['tasks_completed'] - 1)
                    self.swarm_metrics['average_task_completion_time'] = (total + completion_time) / self.swarm_metrics['tasks_completed']
                    logger.info(f"AGV {agv.agv_id} 完成任务 {agv.current_task.task_id}，耗时: {completion_time:.1f}s")
                    
                    # 取下一个任务
                    if agv.task_queue:
                        agv.current_task = agv.task_queue.pop(0)
                        agv.current_task.started_at = current_time
                        # 规划路径
                        agv.current_state.path = self.plan_path_for_agv(agv, agv.current_task.target_point)
                    else:
                        agv.current_task = None
        
        self._update_kd_tree()
    
    def step(self, dt: float) -> None:
        """执行一次蜂群控制周期"""
        self.simulation_time += dt
        
        # 1. 同步AGV状态
        self.sync_agv_states()
        
        # 2. 任务分配
        self.allocate_tasks()
        
        # 3. 冲突检测
        conflicts = self.detect_conflicts()
        
        # 4. 冲突解决
        self.resolve_conflicts(conflicts)
        
        # 5. 更新metrics
        for agv in self.agvs.values():
            self.swarm_metrics['total_distance_traveled'] += agv.current_state.speed * dt
            self.swarm_metrics['total_energy_consumed'] += agv.spec.power_consumption_rate * dt
    
    def start(self) -> None:
        """启动蜂群控制循环"""
        self.running = True
        logger.info("多AGV蜂群协调器已启动")
        while self.running:
            start_time = time.time()
            self.step(0.1)  # 100ms控制周期
            elapsed = time.time() - start_time
            if elapsed < 0.1:
                time.sleep(0.1 - elapsed)
    
    def stop(self) -> None:
        """停止蜂群控制循环"""
        self.running = False
        logger.info("多AGV蜂群协调器已停止")
        logger.info(f"蜂群运行统计: {self.swarm_metrics}")
    
    def get_swarm_status(self) -> dict:
        """获取蜂群整体状态"""
        active_agvs = len([a for a in self.agvs.values() if a.available])
        pending_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING])
        in_progress_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS])
        completed_tasks = self.swarm_metrics['tasks_completed']
        
        return {
            'active_agvs': active_agvs,
            'total_agvs': len(self.agvs),
            'pending_tasks': pending_tasks,
            'in_progress_tasks': in_progress_tasks,
            'completed_tasks': completed_tasks,
            'conflict_count': self.swarm_metrics['conflict_count'],
            'total_distance': self.swarm_metrics['total_distance_traveled'],
            'average_task_time': self.swarm_metrics['average_task_completion_time'],
            'simulation_time': self.simulation_time
        }

if __name__ == "__main__":
    # 测试用例
    scene = WarehouseScene()
    coordinator = AGVSwarmCoordinator(scene)
    
    # 注册3台AGV
    spec_m = AGVSpec(size_class='M', max_payload=100, max_speed=2.0)
    spec_l = AGVSpec(size_class='L', max_payload=300, max_speed=1.5)
    
    for i in range(3):
        state = AGVState(pose=(i*2.0, 0, 0), speed=0.0)
        coordinator.register_agv(f"AGV_{i}", spec_m if i < 2 else spec_l, state)
    
    # 添加5个运输任务
    for i in range(5):
        task = SwarmTask(
            task_type="transport",
            priority=TaskPriority(i % 4),
            source_point=(i*3.0, 1.0, 0),
            target_point=(10 + i*2.0, 5.0, 0),
            payload=50 + i*20,
            deadline=1800
        )
        coordinator.add_task(task)
    
    # 运行模拟
    try:
        for _ in range(100):  # 模拟10秒
            coordinator.step(0.1)
            if _ % 10 == 0:
                status = coordinator.get_swarm_status()
                print(f"模拟时间 {status['simulation_time']:.1f}s: 完成任务 {status['completed_tasks']}, 冲突 {status['conflict_count']}")
    finally:
        coordinator.stop()
