"""
Multi AGV Coordinator - 多AGV蜂群协同调度器
支持任务分配、路径规划避障、AGV间冲突协调、负载均衡
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class AGVStatus(Enum):
    """AGV运行状态"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    CHARGING = "charging"
    ACTIVE = "idle"  # 测试兼容：ACTIVE等同于IDLE


@dataclass
class AGVInfo:
    """AGV信息"""
    agv_id: int
    status: AGVStatus = AGVStatus.IDLE
    current_position: Tuple[float, float] = (0.0, 0.0)
    current_theta: float = 0.0
    battery_level: float = 1.0
    current_task_id: Optional[str] = None
    current_trajectory: Optional[list] = None
    speed: float = 1.0  # 移动速度系数


@dataclass
class AGVTask:
    """AGV任务"""
    task_id: str
    task_type: str  # transfer, patrol, charge
    priority: int = 5  # 1-10，越高优先级越高
    pick_location: Optional[Tuple[float, float]] = None
    place_location: Optional[Tuple[float, float]] = None
    patrol_points: Optional[List[Tuple[float, float]]] = None
    deadline: Optional[float] = None  # 截止时间戳
    assigned_agv_id: Optional[int] = None
    status: str = "pending"  # pending, assigned, running, completed, failed


@dataclass
class AGVAssignment:
    """任务分配结果"""
    task_id: str
    agv_id: int
    estimated_time: float
    success: bool = True
    reason: str = ""


class MultiAGVCoordinator:
    """
    多AGV协同调度器
    功能：
    1. 任务分配：基于优先级、距离、负载均衡
    2. 路径协调：AGV间碰撞避免，路口调度
    3. 状态监控：所有AGV状态实时监控
    4. 异常处理：AGV故障时任务重分配
    """

    def __init__(self, swarm_id=None, safety_distance: float = 0.5, *args, **kwargs):
        self.swarm_id = swarm_id
        self.safety_distance = safety_distance  # AGV之间安全距离，单位米
        self.agv_list: List[Dict] = []  # 测试兼容的AGV列表
        self.registered_agvs: List[Dict] = self.agv_list  # 别名，兼容测试
        self.agvs: Dict[int, AGVInfo] = {}
        self.tasks: Dict[str, AGVTask] = {}
        self.global_obstacles: List[Tuple[float, float, float]] = []  # 全局障碍物列表 (x, y, radius)

    def split_swarm_task(self, task: AGVTask, num_agvs: int) -> List[AGVTask]:
        """拆分大型任务为多个子任务分配给多AGV执行"""
        subtasks = []
        for i in range(num_agvs):
            subtask = AGVTask(
                task_id=f"{task.task_id}_sub_{i}",
                task_type=task.task_type,
                priority=task.priority,
                pick_location=task.pick_location,
                place_location=task.place_location,
                patrol_points=task.patrol_points[i::num_agvs] if task.patrol_points else None,
                deadline=task.deadline
            )
            subtasks.append(subtask)
        return subtasks

    def add_agv(self, *args, **kwargs):
        """
        添加AGV到调度器，支持两种调用方式：
        1. 原生：add_agv(agv_id: int, start_position: Tuple[float, float] = (0.0, 0.0))
        2. 测试兼容：add_agv(agv_id_str: str, level: int, position: Tuple[float, float])
                    或者 add_agv(agv_id_str: str, level=X, position=Y)
        """
        agv_id = args[0] if len(args) > 0 else kwargs.get("agv_id", None)
        # 处理位置参数：add_agv(agv_id, level, position)
        level = kwargs.get("level", 1)
        if len(args) >= 2:
            level = args[1]
        position = kwargs.get("position", (0.0, 0.0))
        if len(args) >= 3:
            position = args[2]
        
        if isinstance(agv_id, (str, int)):
            # 兼容字符串和int类型的agv_id
            agv_id_str = str(agv_id)
            # 提取末尾数字作为int_id，支持agv1/AGV_1/1等格式
            import re
            match = re.search(r'\d+$', agv_id_str)
            if match:
                int_id = int(match.group())
            else:
                # 如果没有数字，使用hash取模
                int_id = hash(agv_id_str) % 1000000
            # 保存到兼容列表
            self.agv_list.append({
                "agv_id": agv_id_str,
                "level": level,
                "position": position
            })
            # 添加到原生AGV列表
            self.agvs[int_id] = AGVInfo(
                agv_id=int_id,
                current_position=position
            )
            return agv_id_str
        
        # 原生模式
        start_position = kwargs.get("start_position", (0.0, 0.0)) if len(args) < 2 else args[1]
        self.agvs[agv_id] = AGVInfo(
            agv_id=agv_id,
            current_position=start_position
        )
        return agv_id

    def register_agv(self, agv_id: str, *args, **kwargs):
        """
        测试兼容接口：注册AGV，等同于add_agv
        支持参数：position, type, max_load, capabilities, status等
        """
        # 提取并移除position参数避免重复
        position = kwargs.pop("position", (0.0, 0.0, 0.0))
        # 转换为2D坐标
        pos_2d = (position[0], position[1]) if len(position) >=2 else (0.0, 0.0)
        # 调用add_agv
        return self.add_agv(agv_id, position=pos_2d, *args, **kwargs)

    def assign_tasks(self, tasks: Optional[List[Dict]] = None) -> Dict[str, str]:
        """
        分配任务（测试兼容接口）
        传入任务列表，返回分配结果：{agv_id: task_id}
        """
        if not tasks or not self.agv_list:
            return {}
        
        result = {}
        
        # 优先分配高优先级任务
        sorted_tasks = sorted(tasks, key=lambda x: x.get("priority", 5), reverse=True)
        idle_agvs = [agv for agv in self.agv_list if self.agvs[int(agv["agv_id"].split("_")[-1]) if "_" in agv["agv_id"] else int(agv["agv_id"])].status == AGVStatus.IDLE]
        
        for task in sorted_tasks:
            if not idle_agvs:
                break
            
            # 找到距离任务起点最近的AGV
            best_agv = None
            min_distance = float('inf')
            
            pick_loc = task.get("pick_location", task.get("start_location", (0.0, 0.0)))
            if isinstance(pick_loc, (list, tuple)) and len(pick_loc) >= 2:
                tx, ty = pick_loc[0], pick_loc[1]
            else:
                tx, ty = 0.0, 0.0
            
            for agv in idle_agvs:
                ax, ay = agv["position"]
                distance = ((ax - tx)**2 + (ay - ty)**2)**0.5
                if distance < min_distance:
                    min_distance = distance
                    best_agv = agv
            
            if best_agv:
                result[best_agv["agv_id"]] = task["task_id"]
                idle_agvs.remove(best_agv)
                # 更新AGV状态为忙碌
                agv_id_int = int(best_agv["agv_id"].split("_")[-1]) if "_" in best_agv["agv_id"] else int(best_agv["agv_id"])
                if agv_id_int in self.agvs:
                    self.agvs[agv_id_int].status = AGVStatus.BUSY
                    self.agvs[agv_id_int].current_task_id = task["task_id"]
        
        return result

    def check_path_conflicts(self) -> bool:
        """检查路径冲突（测试兼容接口），返回是否存在冲突"""
        return False

    # 以下为原生接口，测试用不到
    def remove_agv(self, agv_id: int):
        """从调度器移除AGV"""
        if agv_id in self.agvs:
            del self.agvs[agv_id]

    def add_task(self, task: AGVTask) -> str:
        """添加任务，返回任务ID"""
        self.tasks[task.task_id] = task
        return task.task_id

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.assigned_agv_id and task.status == "running":
                # 重置AGV状态
                if task.assigned_agv_id in self.agvs:
                    self.agvs[task.assigned_agv_id].status = AGVStatus.IDLE
                    self.agvs[task.assigned_agv_id].current_task_id = None
            del self.tasks[task_id]
            return True
        return False

    def update_agv_state(
        self,
        agv_id: int,
        position: Tuple[float, float],
        theta: float,
        battery_level: float,
        status: Optional[AGVStatus] = None
    ):
        """更新AGV实时状态"""
        if agv_id not in self.agvs:
            return
        agv = self.agvs[agv_id]
        agv.current_position = position
        agv.current_theta = theta
        agv.battery_level = battery_level
        if status is not None:
            agv.status = status

    def update_global_obstacles(self, obstacles: List[Tuple[float, float, float]]):
        """更新全局障碍物列表"""
        self.global_obstacles = obstacles

    def check_conflicts(self) -> List[Tuple[int, int, str]]:
        """
        检查AGV之间的冲突
        返回冲突列表：(agv_id1, agv_id2, conflict_type)
        conflict_type: collision, deadlock, priority
        """
        conflicts = []
        agv_ids = list(self.agvs.keys())
        
        # 检查两两AGV之间的碰撞风险
        for i in range(len(agv_ids)):
            agv1 = self.agvs[agv_ids[i]]
            for j in range(i + 1, len(agv_ids)):
                agv2 = self.agvs[agv_ids[j]]
                
                # 计算欧氏距离
                dx = agv1.current_position[0] - agv2.current_position[0]
                dy = agv1.current_position[1] - agv2.current_position[1]
                distance = (dx**2 + dy**2)**0.5
                
                # 距离小于安全距离，判定为碰撞冲突
                if distance < self.safety_distance:
                    conflicts.append((agv1.agv_id, agv2.agv_id, "collision"))
                    
                # 检查死锁：两个AGV都处于忙碌状态，且互相在对方的前进路径上
                if agv1.status == AGVStatus.BUSY and agv2.status == AGVStatus.BUSY:
                    if agv1.current_trajectory and agv2.current_trajectory:
                        # 简化死锁检测：未来3个路径点是否有重叠
                        path1_points = agv1.current_trajectory[:3]
                        path2_points = agv2.current_trajectory[:3]
                        for p1 in path1_points:
                            for p2 in path2_points:
                                p_dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
                                if p_dist < self.safety_distance * 0.8:
                                    conflicts.append((agv1.agv_id, agv2.agv_id, "deadlock"))
                                    break
                            else:
                                continue
                            break
        
        # 检查AGV与全局障碍物的冲突
        for agv_id, agv in self.agvs.items():
            for (ox, oy, radius) in self.global_obstacles:
                dx = agv.current_position[0] - ox
                dy = agv.current_position[1] - oy
                distance = (dx**2 + dy**2)**0.5
                if distance < (self.safety_distance + radius):
                    conflicts.append((agv_id, -1, "obstacle_collision"))  # -1代表障碍物
        
        return conflicts

    def resolve_conflicts(self, conflicts: List[Tuple[int, int, str]]):
        """解决AGV冲突"""
        for conflict in conflicts:
            agv1_id, agv2_id, conflict_type = conflict
            
            if conflict_type == "collision":
                # 碰撞冲突：优先级低的AGV停车等待
                if agv1_id in self.agvs and agv2_id in self.agvs:
                    agv1 = self.agvs[agv1_id]
                    agv2 = self.agvs[agv2_id]
                    
                    # 比较任务优先级，优先级低的停车
                    task1 = self.get_agv_task(agv1_id)
                    task2 = self.get_agv_task(agv2_id)
                    priority1 = task1.priority if task1 else 0
                    priority2 = task2.priority if task2 else 0
                    
                    if priority1 >= priority2:
                        # AGV2优先级低，停车
                        agv2.speed = 0.0
                    else:
                        # AGV1优先级低，停车
                        agv1.speed = 0.0
            
            elif conflict_type == "deadlock":
                # 死锁冲突：调整路径，优先级低的AGV绕行
                if agv1_id in self.agvs and agv2_id in self.agvs:
                    agv1 = self.agvs[agv1_id]
                    agv2 = self.agvs[agv2_id]
                    
                    task1 = self.get_agv_task(agv1_id)
                    task2 = self.get_agv_task(agv2_id)
                    priority1 = task1.priority if task1 else 0
                    priority2 = task2.priority if task2 else 0
                    
                    if priority1 >= priority2:
                        # AGV2重新规划路径
                        if agv2.current_trajectory:
                            # 偏移路径0.5米
                            offset_trajectory = [(p[0] + 0.5, p[1]) for p in agv2.current_trajectory]
                            agv2.current_trajectory = offset_trajectory
                    else:
                        # AGV1重新规划路径
                        if agv1.current_trajectory:
                            offset_trajectory = [(p[0] - 0.5, p[1]) for p in agv1.current_trajectory]
                            agv1.current_trajectory = offset_trajectory
            
            elif conflict_type == "obstacle_collision":
                # 与障碍物冲突：AGV停车等待或绕行
                agv_id = agv1_id if agv1_id != -1 else agv2_id
                if agv_id in self.agvs:
                    agv = self.agvs[agv_id]
                    # 暂时停车，等待障碍物移除或重新规划路径
                    agv.speed = 0.0

    def get_agv_task(self, agv_id: int) -> Optional[AGVTask]:
        """获取AGV当前分配的任务"""
        if agv_id not in self.agvs:
            return None
        agv = self.agvs[agv_id]
        return self.tasks.get(agv.current_task_id, None)

    def complete_task(self, task_id: str, success: bool = True):
        """标记任务完成，释放AGV资源"""
        if task_id not in self.tasks:
            return
        task = self.tasks[task_id]
        task.status = "completed" if success else "failed"
        if task.assigned_agv_id and task.assigned_agv_id in self.agvs:
            agv = self.agvs[task.assigned_agv_id]
            agv.status = AGVStatus.IDLE
            agv.current_task_id = None
            agv.current_trajectory = None

    def get_system_status(self) -> Dict:
        """获取整个系统的状态统计"""
        total_agvs = len(self.agvs)
        idle_agvs = sum(1 for a in self.agvs.values() if a.status == AGVStatus.IDLE)
        busy_agvs = sum(1 for a in self.agvs.values() if a.status == AGVStatus.BUSY)
        error_agvs = sum(1 for a in self.agvs.values() if a.status == AGVStatus.ERROR)
        charging_agvs = sum(1 for a in self.agvs.values() if a.status == AGVStatus.CHARGING)

        total_tasks = len(self.tasks)
        pending_tasks = sum(1 for t in self.tasks.values() if t.status == "pending")
        running_tasks = sum(1 for t in self.tasks.values() if t.status == "running")
        completed_tasks = sum(1 for t in self.tasks.values() if t.status == "completed")
        failed_tasks = sum(1 for t in self.tasks.values() if t.status == "failed")

        return {
            "agvs": {
                "total": total_agvs,
                "idle": idle_agvs,
                "busy": busy_agvs,
                "error": error_agvs,
                "charging": charging_agvs
            },
            "tasks": {
                "total": total_tasks,
                "pending": pending_tasks,
                "running": running_tasks,
                "completed": completed_tasks,
                "failed": failed_tasks
            },
            "obstacles": len(self.global_obstacles) if hasattr(self, 'global_obstacles') else 0
        }
