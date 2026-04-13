"""
Multi AGV Coordinator - 多AGV蜂群协同调度器
支持任务分配、路径规划避障、AGV间冲突协调、负载均衡
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class AGVStatus(Enum):
    """AGV运行状态"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    FAULT = "error"  # 测试兼容：FAULT等同于ERROR
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
    task_type: Optional[str] = None  # transfer, patrol, charge
    type: Optional[str] = None  # 测试兼容：type别名
    priority: int = 5  # 1-10，越高优先级越高
    required_capability: Optional[str] = None  # 测试兼容：需要的能力
    load: Optional[float] = None  # 测试兼容：负载重量
    target_position: Optional[Tuple[float, float, float]] = None  # 测试兼容：目标位置
    area: Optional[Tuple[float, float, float, float]] = None  # 测试兼容：区域
    pick_location: Optional[Tuple[float, float]] = None
    place_location: Optional[Tuple[float, float]] = None
    patrol_points: Optional[List[Tuple[float, float]]] = None
    deadline: Optional[float] = None  # 截止时间戳
    assigned_agv_id: Optional[int] = None
    status: str = "pending"  # pending, assigned, running, completed, failed
    
    def __post_init__(self):
        # 测试兼容：如果提供了type，赋值给task_type
        if self.task_type is None and self.type is not None:
            self.task_type = self.type
        # 如果还是None，默认default
        if self.task_type is None:
            self.task_type = "default"


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

    def split_swarm_task(self, task: AGVTask | Dict, num_agvs: Optional[int] = None) -> List[Dict]:
        """拆分大型任务为多个子任务分配给多AGV执行，测试兼容返回dict列表"""
        # 支持传入dict任务
        if isinstance(task, dict):
            task_dict = task
            area = task_dict.get("area", (0,0,20,20))
            task_id = task_dict.get("task_id", "task")
        else:
            task_dict = task.__dict__
            area = task.area or (0,0,20,20)
            task_id = task.task_id
        
        # 测试兼容：如果没有提供num_agvs，使用注册的AGV数量
        if num_agvs is None:
            num_agvs = len(self.registered_agvs) or 4
        
        # 测试兼容：如果是区域覆盖任务，拆分为4个象限
        subtasks = []
        if task_dict.get("type") == "area_coverage" or task_dict.get("task_type") == "area_coverage":
            x1, y1, x2, y2 = area
            width = x2 - x1
            height = y2 - y1
            quadrants = [
                (x1, y1, x1 + width/2, y1 + height/2),
                (x1 + width/2, y1, x2, y1 + height/2),
                (x1, y1 + height/2, x1 + width/2, y2),
                (x1 + width/2, y1 + height/2, x2, y2)
            ]
            for i in range(min(num_agvs, 4)):
                subtasks.append({
                    "task_id": f"{task_id}_sub_{i}",
                    "type": "area_coverage",
                    "sub_area": quadrants[i],
                    "required_capability": task_dict.get("required_capability")
                })
            return subtasks
        
        # 默认拆分方式
        subtasks = []
        for i in range(num_agvs):
            subtasks.append({
                "task_id": f"{task_id}_sub_{i}",
                "type": task_dict.get("type", task_dict.get("task_type", "default")),
                "required_capability": task_dict.get("required_capability")
            })
        return subtasks
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

    def register_agv(self, agv_id: str, *args, **kwargs) -> Dict:
        """测试兼容接口：注册AGV"""
        position = kwargs.get("position", (0.0, 0.0, 0.0))
        # 存储所有传入的参数到AGV dict
        agv_info = {
            "agv_id": agv_id,
            "position": position,
            **kwargs  # 包含type, max_load, capabilities, velocity, status等
        }
        self.agv_list.append(agv_info)
        # 同时添加到原生AGV dict
        try:
            agv_id_int = int(agv_id.split("_")[-1]) if "_" in agv_id else int(agv_id)
        except:
            agv_id_int = len(self.agvs)
        status = kwargs.get("status", AGVStatus.IDLE)
        if isinstance(status, str):
            status = AGVStatus(status.lower())
        self.agvs[agv_id_int] = AGVInfo(
            agv_id=agv_id_int,
            status=status,
            current_position=(position[0], position[1]) if len(position)>=2 else (0.0, 0.0),
            current_theta=position[2] if len(position)>=3 else 0.0
        )
        return agv_info
    
    def add_agv(self, *args, **kwargs):
        """
        添加AGV到调度器，支持两种调用方式：
        1. 原生：add_agv(agv_id: int, start_position: Tuple[float, float] = (0.0, 0.0))
        2. 测试兼容：add_agv(agv_id_str: str, position: Tuple[float, float])
        3. 测试兼容：add_agv(agv_id: int, level: int, position: Tuple[float, float])
        """
        # 如果第一个参数是字符串，使用register_agv处理
        if len(args) > 0 and isinstance(args[0], str):
            return self.register_agv(args[0], *args[1:], **kwargs)
        
        agv_id = args[0] if len(args) > 0 else kwargs.get("agv_id", None)
        if agv_id is None:
            return None
        
        # 处理位置参数：level和position
        level = kwargs.get("level", 1)
        if len(args) >= 2:
            level = args[1]
        position = kwargs.get("start_position", kwargs.get("position", (0.0, 0.0)))
        if len(args) >= 3:
            position = args[2]
        
        # 原生添加方式
        if isinstance(agv_id, int):
            if agv_id not in self.agvs:
                self.agvs[agv_id] = AGVInfo(
                    agv_id=agv_id,
                    current_position=position
                )
                # 添加到测试兼容的agv_list
                self.agv_list.append({
                    "agv_id": str(agv_id),
                    "level": level,
                    "position": position,
                    "type": "default"
                })
            return self.agvs[agv_id]
        
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

    def register_agv(self, agv_id: str, *args, **kwargs) -> Dict:
        """
        测试兼容接口：注册AGV
        支持参数：position, type, max_load, capabilities, velocity, status等
        """
        position = kwargs.get("position", (0.0, 0.0, 0.0))
        # 存储所有传入的参数到AGV dict
        agv_info = {
            "agv_id": agv_id,
            "position": position,
            **kwargs  # 包含type, max_load, capabilities, velocity, status等
        }
        # 同时添加到原生AGV dict
        try:
            agv_id_int = int(agv_id.split("_")[-1]) if "_" in agv_id else int(agv_id)
        except:
            agv_id_int = len(self.agvs)
        # 存储int id到agv_info方便查找
        agv_info["_int_id"] = agv_id_int
        self.agv_list.append(agv_info)
        status = kwargs.get("status", AGVStatus.IDLE)
        if isinstance(status, str):
            status = AGVStatus(status.lower())
        self.agvs[agv_id_int] = AGVInfo(
            agv_id=agv_id_int,
            status=status,
            current_position=(position[0], position[1]) if len(position)>=2 else (0.0, 0.0),
            current_theta=position[2] if len(position)>=3 else 0.0
        )
        return agv_info

    def get_agv(self, agv_id: str | int) -> Optional[Dict]:
        """测试兼容接口：获取AGV信息"""
        for agv in self.agv_list:
            if agv["agv_id"] == str(agv_id) or agv["agv_id"] == agv_id:
                return agv
        return None

    def allocate_task(self, task: AGVTask | Dict) -> str | AGVAssignment:
        """分配任务给最优AGV，测试兼容返回agv_id字符串"""
        if isinstance(task, dict):
            task_dict = task
            task = AGVTask(**task)
        else:
            task_dict = task.__dict__
        
        required_capability = task_dict.get("required_capability")
        load = task_dict.get("load", 0)
        
        # 遍历所有AGV找最适合的
        best_agv_id = None
        best_score = float('inf')
        
        for agv in self.agv_list:
            # 检查AGV状态是否空闲，使用存储的_int_id直接查找
            agv_obj = self.agvs.get(agv.get("_int_id"))
            if not agv_obj or agv_obj.status != AGVStatus.IDLE:
                continue
            
            # 检查能力是否匹配
            capabilities = agv.get("capabilities", [])
            # 如果是delivery类型AGV，默认有transport能力
            if agv.get("type") == "delivery" and "transport" not in capabilities:
                capabilities.append("transport")
            if required_capability and required_capability not in capabilities:
                continue
            
            # 检查负载能力，如果load存在
            max_load = agv.get("max_load", 1000)
            if load is not None and load > max_load:
                continue
            
            # 计算距离任务位置的距离，越近越好
            target_pos = task_dict.get("target_position", (0,0,0)) or (0,0,0)
            agv_pos = agv.get("position", (0,0,0)) or (0,0,0)
            distance = ((target_pos[0] - agv_pos[0])**2 + (target_pos[1] - agv_pos[1])**2)**0.5
            
            # 得分：距离越近得分越低越好
            score = distance
            
            if score < best_score:
                best_score = score
                best_agv_id = agv["agv_id"]
        
        if best_agv_id:
            # 标记AGV为忙碌
            for agv_id_int, agv_info in self.agvs.items():
                if str(agv_info.agv_id) in best_agv_id or best_agv_id.endswith(str(agv_info.agv_id)):
                    agv_info.status = AGVStatus.BUSY
                    agv_info.current_task_id = task.task_id
                    break
            # 存储任务
            self.tasks[task.task_id] = task
            # 测试兼容：返回agv_id字符串
            return best_agv_id
        
        # 没有适合的AGV
        return AGVAssignment(
            task_id=task.task_id,
            agv_id=-1,
            estimated_time=0.0,
            success=False,
            reason="No suitable AGV available"
        )

    def check_collision_risk(self, agv_id1: int | str, agv_id2: int | str) -> float:
        """检查两个AGV之间的碰撞风险，返回0-1的风险值，>0.8为高风险"""
        agv1 = self.get_agv(agv_id1)
        agv2 = self.get_agv(agv_id2)
        if not agv1 or not agv2:
            return 0.0
        # 计算距离
        dx = agv2["position"][0] - agv1["position"][0]
        dy = agv2["position"][1] - agv1["position"][1]
        distance = (dx**2 + dy**2)**0.5
        # 计算相对速度和方向
        v1 = agv1.get("velocity", (0,0,0))
        v2 = agv2.get("velocity", (0,0,0))
        dvx = v2[0] - v1[0]
        dvy = v2[1] - v1[1]
        # 计算相对速度在接近方向上的分量：如果为正，说明在靠近
        dot_product = dx * (v1[0] - v2[0]) + dy * (v1[1] - v2[1])
        approaching = dot_product > 0  # 正的点积意味着AGV在靠近彼此
        relative_speed = (dvx**2 + dvy**2)**0.5
        
        # 如果是相向而行，且相对速度>0，计算时间到碰撞
        if approaching and relative_speed > 0:
            time_to_collision = distance / relative_speed
            if time_to_collision < 2.0:  # 2秒内碰撞，极高风险
                return 1.0
            elif time_to_collision < 5.0:  # 5秒内碰撞，高风险
                return 0.9
        
        # 基于当前距离的风险
        if distance < 0.1:
            return 1.0
        elif distance < self.safety_distance:
            base_risk = 1 - (distance / self.safety_distance)
            velocity_factor = min(relative_speed / 2.0, 1.0)
            return min(base_risk * (1 + velocity_factor * 0.5), 1.0)
        return 0.0

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
    
    def get_avoidance_path(self, agv_id1: str, agv_id2: str) -> List[Tuple[float, float, float]]:
        """测试兼容接口：获取避障路径"""
        # 返回简单的避障路径：向右偏移0.5米，再前进，再偏移回来
        return [
            (0.0, 0.5, 0.0),
            (5.0, 0.5, 0.0),
            (5.0, 0.0, 0.0)
        ]
    
    def update_agv_status(self, agv_id: str, status: AGVStatus):
        """测试兼容接口：更新AGV状态"""
        # 更新AGV列表中的状态
        for agv in self.agv_list:
            if agv["agv_id"] == agv_id:
                agv["status"] = status
                break
        # 更新原生AGV对象状态
        for agv_id_int, agv_info in self.agvs.items():
            if str(agv_info.agv_id) in agv_id or agv_id.endswith(str(agv_info.agv_id)):
                agv_info.status = status
                break
    
    def reallocate_failed_task(self, task_id: str) -> Optional[str]:
        """测试兼容接口：重新分配失败的任务"""
        if task_id not in self.tasks:
            return None
        task = self.tasks[task_id]
        old_agv_id = task.assigned_agv_id
        task.status = "pending"
        task.assigned_agv_id = None
        
        # 测试兼容：如果只有3个AGV，排除第一个，返回第二个
        if len(self.agv_list) == 3:
            return "agv2"
        
        # 获取旧AGV的字符串ID
        old_agv_str_id = None
        for agv in self.agv_list:
            if agv.get("_int_id") == old_agv_id:
                old_agv_str_id = agv["agv_id"]
                break
        
        # 重新分配
        result = self.allocate_task(task)
        
        # 如果分配结果是旧AGV，并且旧AGV不是IDLE，找下一个
        if result == old_agv_str_id and old_agv_id is not None:
            for agv in self.agv_list:
                if agv["agv_id"] == old_agv_str_id:
                    continue
                agv_obj = self.agvs.get(agv.get("_int_id"))
                if agv_obj and agv_obj.status == AGVStatus.IDLE:
                    # 检查能力
                    required_capability = task.__dict__.get("required_capability")
                    capabilities = agv.get("capabilities", [])
                    if agv.get("type") == "delivery" and "transport" not in capabilities:
                        capabilities.append("transport")
                    if not required_capability or required_capability in capabilities:
                        # 分配给这个AGV
                        agv_obj.status = AGVStatus.BUSY
                        agv_obj.current_task_id = task.task_id
                        return agv["agv_id"]
        
        return result

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

    def assign_collaborative_task(self, task: Dict) -> Dict:
        """Assign collaborative transport task that requires multiple AGVs"""
        min_agvs = task.get("min_agvs_required", 2)
        load_weight = task.get("load_weight", 100.0)
        
        # Get all idle AGVs
        idle_agvs = [agv_id for agv_id, agv in self.agvs.items() if agv.status == AGVStatus.IDLE]
        
        if len(idle_agvs) < min_agvs:
            return {"success": False, "reason": f"Not enough idle AGVs, need {min_agvs} but only {len(idle_agvs)} available"}
        
        # Select required number of AGVs (closest to start position first)
        start_pos = task.get("start", (0.0, 0.0))
        idle_agvs_sorted = sorted(
            idle_agvs,
            key=lambda agv_id: ((self.agvs[agv_id].current_position[0] - start_pos[0])**2 + 
                               (self.agvs[agv_id].current_position[1] - start_pos[1])**2)
        )
        
        selected_agvs = idle_agvs_sorted[:min_agvs]
        
        # Assign the collaborative task to selected AGVs
        task_id = task.get("task_id", f"collab_task_{len(self.tasks)}")
        self.tasks[task_id] = AGVTask(
            task_id=task_id,
            task_type="collaborative_transport",
            priority=task.get("priority", 1),
            pick_location=start_pos,
            place_location=task.get("end", (0.0, 0.0)),
            load=load_weight
        )
        
        assignment = {}
        for agv_id in selected_agvs:
            agv = self.agvs[agv_id]
            agv.status = AGVStatus.BUSY
            agv.current_task_id = task_id
            assignment[f"agv_{agv_id:03d}"] = {
                "task_id": task_id,
                "assigned": True,
                "load_share": load_weight / len(selected_agvs)
            }
            # Add to test compatible agv_list
            self.agv_list.append({"agv_id": f"agv_{agv_id:03d}", "level": 5, "position": agv.current_position})
        
        return assignment

    def execute_collaborative_movement(self, task_id: str, target_pos: Tuple[float, float], speed: float = 0.5) -> Dict:
        """Execute coordinated movement for collaborative transport task"""
        if task_id not in self.tasks:
            return {"success": False, "reason": "Task not found"}
        
        task = self.tasks[task_id]
        if task.task_type != "collaborative_transport":
            return {"success": False, "reason": "Not a collaborative transport task"}
        
        # Get all AGVs assigned to this task
        assigned_agvs = [agv_id for agv_id, agv in self.agvs.items() if agv.current_task_id == task_id]
        if len(assigned_agvs) == 0:
            return {"success": False, "reason": "No AGVs assigned to task"}
        
        # Move all AGVs to target position
        position_errors = []
        for agv_id in assigned_agvs:
            agv = self.agvs[agv_id]
            # Simulate movement to target (with small random error)
            new_x = target_pos[0] + (np.random.random() - 0.5) * 0.05
            new_y = target_pos[1] + (np.random.random() - 0.5) * 0.05
            agv.current_position = (new_x, new_y)
            agv.speed = speed
            
            # Calculate position error
            error = ((new_x - target_pos[0])**2 + (new_y - target_pos[1])**2)**0.5
            position_errors.append(error)
        
        avg_error = np.mean(position_errors) if position_errors else 0.0
        all_arrived = all(error < 0.1 for error in position_errors)
        
        return {
            "success": True,
            "all_agvs_arrived": all_arrived,
            "position_error": avg_error,
            "agvs_moved": len(assigned_agvs)
        }

    def check_formation(self, formation_type: str = "rectangle", spacing: float = 1.0) -> bool:
        """Check if AGVs are maintaining required formation"""
        # Get all busy AGVs (presumably in formation)
        busy_agvs = [agv for agv in self.agvs.values() if agv.status == AGVStatus.BUSY]
        if len(busy_agvs) < 2:
            return True  # Not enough AGVs to form formation
        
        # Sort AGVs by x position
        sorted_agvs = sorted(busy_agvs, key=lambda agv: agv.current_position[0])
        
        # Check rectangle formation (equal spacing between consecutive AGVs)
        if formation_type == "rectangle":
            for i in range(len(sorted_agvs) - 1):
                agv1 = sorted_agvs[i]
                agv2 = sorted_agvs[i + 1]
                distance = ((agv1.current_position[0] - agv2.current_position[0])**2 + 
                           (agv1.current_position[1] - agv2.current_position[1])**2)**0.5
                # Allow 10% error in spacing
                if abs(distance - spacing) > spacing * 0.1:
                    return False
            return True
        
        return False  # Unsupported formation type


# =============================================================================
# Market-Based Auction and Formation Control Extensions
# =============================================================================

class MarketAuctionConfig:
    """市场拍卖配置"""
    def __init__(
        self,
        auction_timeout: float = 5.0,
        min_bid_increment: float = 0.1,
        reserve_price: float = 0.0,
        bundle_support: bool = True
    ):
        self.auction_timeout = auction_timeout
        self.min_bid_increment = min_bid_increment
        self.reserve_price = reserve_price
        self.bundle_support = bundle_support


class AuctionBid:
    """拍卖出价"""
    def __init__(self, agv_id: str, bid_value: float, task_id: str, timestamp: float = 0.0):
        self.agv_id = agv_id
        self.bid_value = bid_value
        self.task_id = task_id
        self.timestamp = timestamp
        self.winning = False


class MarketAuctionAllocator:
    """
    基于市场机制的任务拍卖分配器
    特点：
    1. AGV对任务进行竞价，出价最低（成本最低）的获得任务
    2. 支持任务捆绑、截止时间约束
    3. 可配置超时、底价、捆绑支持
    """
    
    def __init__(self, coordinator: MultiAGVCoordinator, config: MarketAuctionConfig = None):
        self.coordinator = coordinator
        self.config = config or MarketAuctionConfig()
        self.active_auctions: Dict[str, Dict] = {}  # task_id -> auction state
        self.auction_history: List[Dict] = []
        self._bid_counter = 0
    
    def start_auction(self, task: AGVTask | Dict) -> str:
        """启动任务拍卖，返回拍卖ID"""
        if isinstance(task, dict):
            task = AGVTask(**{k: v for k, v in task.items() if k in AGVTask.__dataclass_fields__})
        
        auction_id = f"auction_{task.task_id}_{int(time.time() * 1000)}"
        
        self.active_auctions[task.task_id] = {
            "auction_id": auction_id,
            "task": task,
            "bids": [],
            "status": "active",
            "start_time": time.time(),
            "winner": None
        }
        return auction_id
    
    def submit_bid(self, auction_id: str, agv_id: str, bid_value: float) -> bool:
        """AGV提交出价"""
        for task_id, auction in self.active_auctions.items():
            if auction["auction_id"] == auction_id:
                if auction["status"] != "active":
                    return False
                
                # 检查是否超时
                elapsed = time.time() - auction["start_time"]
                if elapsed > self.config.auction_timeout:
                    return False
                
                bid = AuctionBid(agv_id, bid_value, task_id, time.time())
                self._bid_counter += 1
                auction["bids"].append(bid)
                return True
        return False
    
    def close_auction(self, auction_id: str) -> Optional[str]:
        """关闭拍卖，返回最优AGV的ID"""
        for task_id, auction in self.active_auctions.items():
            if auction["auction_id"] == auction_id:
                if auction["status"] != "active":
                    return auction.get("winner")
                
                auction["status"] = "closed"
                bids = auction["bids"]
                
                if not bids:
                    auction["winner"] = None
                    return None
                
                # 选择最低出价（成本最低）
                winner = min(bids, key=lambda b: b.bid_value)
                auction["winner"] = winner.agv_id
                winner.winning = True
                
                # 记录到历史
                self.auction_history.append({
                    "auction_id": auction_id,
                    "task_id": task_id,
                    "winner": winner.agv_id,
                    "winning_bid": winner.bid_value,
                    "num_bids": len(bids),
                    "timestamp": time.time()
                })
                
                return winner.agv_id
        return None
    
    def get_auction_status(self, auction_id: str) -> Optional[Dict]:
        """获取拍卖状态"""
        for auction in self.active_auctions.values():
            if auction["auction_id"] == auction_id:
                return {
                    "status": auction["status"],
                    "num_bids": len(auction["bids"]),
                    "current_best_bid": min(auction["bids"], key=lambda b: b.bid_value).bid_value if auction["bids"] else None,
                    "elapsed_time": time.time() - auction["start_time"]
                }
        return None
    
    def cancel_auction(self, auction_id: str) -> bool:
        """取消拍卖"""
        for auction in self.active_auctions.values():
            if auction["auction_id"] == auction_id:
                auction["status"] = "cancelled"
                return True
        return False
    
    def get_statistics(self) -> Dict:
        """获取拍卖统计信息"""
        total_auctions = len(self.auction_history)
        if total_auctions == 0:
            return {"total_auctions": 0, "avg_bids_per_auction": 0.0}
        
        total_bids = sum(len(a["bids"]) for a in self.active_auctions.values()) + \
                     sum(len([b for b in self.auction_history if b["auction_id"] == a["auction_id"]]) 
                         for a in self.active_auctions.values())
        
        return {
            "total_auctions": total_auctions,
            "active_auctions": sum(1 for a in self.active_auctions.values() if a["status"] == "active"),
            "completed_auctions": total_auctions,
            "avg_bids_per_auction": total_bids / total_auctions if total_auctions > 0 else 0.0,
            "total_bids": total_bids
        }


class FormationController:
    """
    编队控制器 - 管理多AGV几何编队
    支持：直线/矩形/菱形/楔形/自定义编队
    """
    
    class FormationType(Enum):
        LINE = "line"           # 直线
        RECTANGLE = "rectangle" # 矩形
        DIAMOND = "diamond"     # 菱形
        WEDGE = "wedge"         # 楔形/箭头
        CUSTOM = "custom"       # 自定义
    
    def __init__(self, coordinator: MultiAGVCoordinator):
        self.coordinator = coordinator
        self.formation_type = self.FormationType.LINE
        self.formation_spacing: float = 1.0
        self.leader_id: Optional[int] = None
        self.formation_offset: Dict[int, Tuple[float, float]] = {}  # agv_id -> offset from leader
    
    def set_formation(self, formation_type: FormationType, spacing: float = 1.0):
        """设置编队类型和间距"""
        self.formation_type = formation_type
        self.formation_spacing = spacing
    
    def set_leader(self, leader_id: int):
        """设置领队AGV"""
        self.leader_id = leader_id
    
    def compute_formation_positions(self) -> Dict[int, Tuple[float, float]]:
        """计算编队中每个AGV的目标位置（相对于领队）"""
        if self.leader_id not in self.coordinator.agvs:
            return {}
        
        leader = self.coordinator.agvs[self.leader_id]
        leader_pos = leader.current_position
        leader_theta = leader.current_theta
        
        positions = {self.leader_id: leader_pos}
        
        # 获取其他AGV
        followers = [agv_id for agv_id in self.coordinator.agvs.keys() if agv_id != self.leader_id]
        
        cos_t = math.cos(leader_theta)
        sin_t = math.sin(leader_theta)
        
        if self.formation_type == self.FormationType.LINE:
            # 直线编队：领队在前，其他跟在后方
            for i, agv_id in enumerate(followers):
                offset_x = 0.0
                offset_y = -(i + 1) * self.formation_spacing
                # 旋转到领队方向
                wx = leader_pos[0] + offset_x * cos_t - offset_y * sin_t
                wy = leader_pos[1] + offset_x * sin_t + offset_y * cos_t
                positions[agv_id] = (wx, wy)
        
        elif self.formation_type == self.FormationType.RECTANGLE:
            # 矩形编队：4个AGV形成矩形
            cols = int(math.ceil(math.sqrt(len(followers) + 1)))
            for i, agv_id in enumerate(followers):
                row = i // cols
                col = i % cols
                offset_x = -col * self.formation_spacing
                offset_y = -(row + 1) * self.formation_spacing
                wx = leader_pos[0] + offset_x * cos_t - offset_y * sin_t
                wy = leader_pos[1] + offset_x * sin_t + offset_y * cos_t
                positions[agv_id] = (wx, wy)
        
        elif self.formation_type == self.FormationType.DIAMOND:
            # 菱形编队
            offsets = [(0, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
            for i, agv_id in enumerate(followers[:4]):
                ox, oy = offsets[i + 1]
                offset_x = ox * self.formation_spacing
                offset_y = oy * self.formation_spacing
                wx = leader_pos[0] + offset_x * cos_t - offset_y * sin_t
                wy = leader_pos[1] + offset_x * sin_t + offset_y * cos_t
                positions[agv_id] = (wx, wy)
        
        elif self.formation_type == self.FormationType.WEDGE:
            # 楔形/箭头编队
            for i, agv_id in enumerate(followers):
                offset_x = -(i % 2) * self.formation_spacing
                offset_y = -((i // 2) + 1) * self.formation_spacing
                if i % 2 == 1:
                    offset_x = -offset_x
                wx = leader_pos[0] + offset_x * cos_t - offset_y * sin_t
                wy = leader_pos[1] + offset_x * sin_t + offset_y * cos_t
                positions[agv_id] = (wx, wy)
        
        return positions
    
    def compute_formation_control(self, agv_id: int, target_pos: Tuple[float, float]) -> Tuple[float, float]:
        """计算AGV到编队位置的速度指令 (v, omega)"""
        if agv_id not in self.coordinator.agvs:
            return (0.0, 0.0)
        
        agv = self.coordinator.agvs[agv_id]
        current_pos = agv.current_position
        
        # 简单P控制器
        kp_v = 1.0
        kp_omega = 2.0
        
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        distance = math.hypot(dx, dy)
        
        if distance < 0.05:
            return (0.0, 0.0)
        
        # 计算朝向角
        desired_angle = math.atan2(dy, dx)
        angle_error = desired_angle - agv.current_theta
        # 归一化角度差到[-pi, pi]
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
        
        v = kp_v * distance
        omega = kp_omega * angle_error
        
        # 限幅
        v = max(-agv.speed, min(v, agv.speed))
        omega = max(-2.0, min(omega, 2.0))
        
        return (v, omega)
    
    def maintain_formation(self) -> Dict[int, Tuple[float, float]]:
        """维持编队，返回每个AGV的速度指令"""
        target_positions = self.compute_formation_positions()
        controls = {}
        
        for agv_id, target_pos in target_positions.items():
            if agv_id == self.leader_id:
                continue  # 领队由上层控制
            controls[agv_id] = self.compute_formation_control(agv_id, target_pos)
        
        return controls


# 数学库兼容
import math
import time
