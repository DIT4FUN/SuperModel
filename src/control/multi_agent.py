# Copyright (C) 2024-2026 赵元请 (DIT4FUN)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
多智能体协调控制模块
====================

多AGV协同控制与编队管理
- 多AGV编队控制
- 分布式协同决策
- 冲突检测与解决
- 任务分配与调度

支持AGV等级: L / XL / XXL
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import heapq


class FormationType(Enum):
    """编队类型"""
    LINE = "line"               # 线性队列
    TRIANGLE = "triangle"       # 三角阵型
    SQUARE = "square"           # 方形阵型
    CIRCLE = "circle"          # 圆形阵型
    V_SHAPE = "v_shape"         # V字形
    GRID = "grid"              # 网格阵型
    FREE = "free"              # 自由分布


class CoordinationState(Enum):
    """协调状态"""
    IDLE = "idle"
    FORMING = "forming"        # 形成编队中
    FORMING_COMPLETE = "formed"
    NAVIGATING = "navigating"   # 编队导航中
    REFORMING = "reforming"     # 重构编队中
    DISBANDING = "disbanding"  # 解散编队中
    DISBANDED = "disbanded"
    EMERGENCY = "emergency"      # 紧急避障


@dataclass
class AgentState:
    """智能体状态"""
    agent_id: str
    position: np.ndarray         # 2D: (x, y) or 3D: (x, y, theta)
    velocity: np.ndarray         # 2D/3D速度
    target: Optional[np.ndarray] = None
    leader_id: Optional[str] = None
    neighbors: List[str] = field(default_factory=list)
    in_formation: bool = False
    formation_slot: Optional[int] = None
    state: CoordinationState = CoordinationState.IDLE
    battery_level: float = 1.0   # 0-1
    task_id: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.position, list):
            self.position = np.array(self.position, dtype=np.float32)
        if isinstance(self.velocity, list):
            self.velocity = np.array(self.velocity, dtype=np.float32)


@dataclass
class FormationSlot:
    """编队位置槽"""
    slot_id: int
    relative_position: np.ndarray  # 相对于队长的位置
    tolerance: float = 0.1          # 到达容忍度 (m)
    assigned_agent: Optional[str] = None


@dataclass
class CoordinationTask:
    """协调任务"""
    task_id: str
    formation_type: FormationType
    target_position: np.ndarray     # 编队目标中心
    target_heading: float           # 编队目标朝向
    slots: List[FormationSlot]
    deadline: Optional[float] = None
    
    
@dataclass
class CollisionRisk:
    """碰撞风险"""
    agent_a: str
    agent_b: str
    distance: float
    time_to_collision: float        # s, 预计碰撞时间
    severity: str                   # "low" / "medium" / "high" / "critical"


class MultiAgentCoordinator:
    """
    多智能体协调控制器
    
    功能:
    - 编队形成与保持
    - 编队变换与重构
    - 分布式避障协调
    - 任务分配与负载均衡
    """
    
    def __init__(
        self,
        communication_range: float = 10.0,  # m
        safety_distance: float = 0.5,       # m
        max_agents: int = 20
    ):
        self.communication_range = communication_range
        self.safety_distance = safety_distance
        self.max_agents = max_agents
        
        # 智能体注册
        self.agents: Dict[str, AgentState] = {}
        
        # 编队管理
        self.formations: Dict[str, CoordinationTask] = {}  # formation_id -> task
        self.agent_formations: Dict[str, str] = {}          # agent_id -> formation_id
        
        # 冲突检测
        self.collision_risks: List[CollisionRisk] = []
        
        # 任务队列
        self.task_queue: List[Tuple[float, CoordinationTask]] = []  # (priority, task)
        
        # 历史轨迹
        self.trajectories: Dict[str, List[np.ndarray]] = {}
        
    def register_agent(
        self,
        agent_id: str,
        initial_position: np.ndarray,
        leader_id: Optional[str] = None
    ) -> bool:
        """
        注册智能体
        
        Args:
            agent_id: 智能体唯一标识
            initial_position: 初始位置 (x, y) or (x, y, theta)
            leader_id: 可选的上级智能体ID
            
        Returns:
            bool: 注册是否成功
        """
        if agent_id in self.agents:
            print(f"[MultiAgentCoordinator] Agent {agent_id} already registered")
            return False
        
        if len(self.agents) >= self.max_agents:
            print(f"[MultiAgentCoordinator] Max agents ({self.max_agents}) reached")
            return False
        
        state = AgentState(
            agent_id=agent_id,
            position=np.array(initial_position, dtype=np.float32),
            velocity=np.zeros_like(initial_position),
            leader_id=leader_id
        )
        
        self.agents[agent_id] = state
        self.trajectories[agent_id] = []
        
        print(f"[MultiAgentCoordinator] Registered agent {agent_id} at {initial_position}")
        return True
    
    def unregister_agent(self, agent_id: str):
        """注销智能体"""
        if agent_id not in self.agents:
            return
        
        # 从编队中移除
        if agent_id in self.agent_formations:
            self._leave_formation(agent_id)
        
        del self.agents[agent_id]
        if agent_id in self.trajectories:
            del self.trajectories[agent_id]
        
        print(f"[MultiAgentCoordinator] Unregistered agent {agent_id}")
    
    def create_formation(
        self,
        formation_id: str,
        formation_type: FormationType,
        target_position: np.ndarray,
        target_heading: float = 0.0,
        formation_size: Optional[int] = None
    ) -> CoordinationTask:
        """
        创建编队
        
        Args:
            formation_id: 编队ID
            formation_type: 编队类型
            target_position: 编队目标中心
            target_heading: 编队目标朝向 (rad)
            formation_size: 编队人数, None=自动
        """
        if formation_id in self.formations:
            raise ValueError(f"Formation {formation_id} already exists")
        
        # 计算编队槽位
        slots = self._generate_formation_slots(formation_type, formation_size or len(self.agents))
        
        task = CoordinationTask(
            task_id=formation_id,
            formation_type=formation_type,
            target_position=np.array(target_position, dtype=np.float32),
            target_heading=target_heading,
            slots=slots
        )
        
        self.formations[formation_id] = task
        
        # 自动分配可用智能体
        available = [aid for aid in self.agents if aid not in self.agent_formations]
        for slot, agent_id in zip(slots, available):
            slot.assigned_agent = agent_id
            self.agent_formations[agent_id] = formation_id
            self.agents[agent_id].in_formation = True
            self.agents[agent_id].formation_slot = slot.slot_id
            self.agents[agent_id].state = CoordinationState.NAVIGATING
        
        print(f"[MultiAgentCoordinator] Created formation {formation_id} with {len(slots)} slots")
        return task
    
    def _generate_formation_slots(
        self,
        formation_type: FormationType,
        size: int
    ) -> List[FormationSlot]:
        """生成编队槽位"""
        slots = []
        
        if formation_type == FormationType.LINE:
            # 线性队列: 沿 x 轴排列
            for i in range(size):
                slot = FormationSlot(
                    slot_id=i,
                    relative_position=np.array([i * 1.0, 0.0], dtype=np.float32),
                    tolerance=0.1
                )
                slots.append(slot)
        
        elif formation_type == FormationType.TRIANGLE:
            # 三角阵型
            positions = []
            row = 0
            count = 0
            while count < size:
                for j in range(row + 1):
                    if count >= size:
                        break
                    x = j * 1.0 - row / 2
                    y = -row * 0.866  # sqrt(3)/2
                    positions.append(np.array([x, y], dtype=np.float32))
                    count += 1
                row += 1
            for i, pos in enumerate(positions):
                slots.append(FormationSlot(slot_id=i, relative_position=pos, tolerance=0.1))
        
        elif formation_type == FormationType.SQUARE:
            # 方形阵型
            side = int(np.ceil(np.sqrt(size)))
            positions = []
            for i in range(size):
                row = i // side
                col = i % side
                x = col * 1.0 - (side - 1) / 2
                y = -row * 1.0
                positions.append(np.array([x, y], dtype=np.float32))
            for i, pos in enumerate(positions):
                slots.append(FormationSlot(slot_id=i, relative_position=pos, tolerance=0.1))
        
        elif formation_type == FormationType.CIRCLE:
            # 圆形阵型
            for i in range(size):
                angle = 2 * np.pi * i / size
                x = np.cos(angle) * (size / (2 * np.pi))
                y = np.sin(angle) * (size / (2 * np.pi))
                slots.append(FormationSlot(
                    slot_id=i,
                    relative_position=np.array([x, y], dtype=np.float32),
                    tolerance=0.1
                ))
        
        elif formation_type == FormationType.V_SHAPE:
            # V字形
            for i in range(size):
                if i == 0:
                    pos = np.array([0.0, 0.0], dtype=np.float32)
                else:
                    side = 1 if i % 2 == 1 else -1
                    row = (i + 1) // 2
                    x = row * 1.0
                    y = side * row * 0.866
                    pos = np.array([x, y], dtype=np.float32)
                slots.append(FormationSlot(slot_id=i, relative_position=pos, tolerance=0.1))
        
        else:  # FREE or default
            for i in range(size):
                slots.append(FormationSlot(
                    slot_id=i,
                    relative_position=np.array([0.0, 0.0], dtype=np.float32),
                    tolerance=0.5
                ))
        
        return slots
    
    def _leave_formation(self, agent_id: str):
        """智能体离开编队"""
        if agent_id not in self.agent_formations:
            return
        
        formation_id = self.agent_formations[agent_id]
        del self.agent_formations[agent_id]
        
        agent = self.agents[agent_id]
        agent.in_formation = False
        agent.formation_slot = None
        agent.state = CoordinationState.IDLE
        
        # 更新编队槽位
        if formation_id in self.formations:
            for slot in self.formations[formation_id].slots:
                if slot.assigned_agent == agent_id:
                    slot.assigned_agent = None
    
    def compute_formation_target(
        self,
        agent_id: str,
        leader_position: np.ndarray,
        leader_heading: float
    ) -> np.ndarray:
        """
        计算智能体在编队中的目标位置
        
        Args:
            agent_id: 智能体ID
            leader_position: 队长位置 (x, y)
            leader_heading: 队长朝向 (rad)
            
        Returns:
            target_position: 智能体目标位置
        """
        if agent_id not in self.agent_formations:
            return self.agents[agent_id].position
        
        formation_id = self.agent_formations[agent_id]
        formation = self.formations[formation_id]
        agent = self.agents[agent_id]
        
        # 获取分配槽位
        slot = None
        for s in formation.slots:
            if s.slot_id == agent.formation_slot:
                slot = s
                break
        
        if slot is None:
            return agent.position
        
        # 旋转变换: 将相对位置转换到世界坐标系
        rel_pos = slot.relative_position
        
        # 旋转矩阵 (绕 z 轴)
        cos_h = np.cos(leader_heading)
        sin_h = np.sin(leader_heading)
        rot = np.array([
            [cos_h, -sin_h],
            [sin_h, cos_h]
        ])
        
        world_rel = rot @ rel_pos
        
        # 平移
        target = leader_position + world_rel
        
        return target
    
    def detect_collisions(self) -> List[CollisionRisk]:
        """
        检测碰撞风险
        
        Returns:
            List of CollisionRisk
        """
        risks = []
        agent_ids = list(self.agents.keys())
        
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                a_id = agent_ids[i]
                b_id = agent_ids[j]
                
                a = self.agents[a_id]
                b = self.agents[b_id]
                
                # 距离检查
                dist = np.linalg.norm(a.position[:2] - b.position[:2])
                
                if dist < 2 * self.safety_distance:
                    # 计算相对速度
                    rel_vel = a.velocity[:2] - b.velocity[:2]
                    rel_pos = b.position[:2] - a.position[:2]
                    
                    # 预计碰撞时间
                    if np.linalg.norm(rel_vel) > 1e-6:
                        ttc = dist / np.linalg.norm(rel_vel)
                    else:
                        ttc = float('inf')
                    
                    # 严重程度
                    if dist < 0.3:
                        severity = "critical"
                    elif dist < self.safety_distance:
                        severity = "high"
                    elif dist < 1.5 * self.safety_distance:
                        severity = "medium"
                    else:
                        severity = "low"
                    
                    if ttc < 5.0:  # 5秒内可能碰撞
                        risk = CollisionRisk(
                            agent_a=a_id,
                            agent_b=b_id,
                            distance=dist,
                            time_to_collision=ttc,
                            severity=severity
                        )
                        risks.append(risk)
        
        self.collision_risks = risks
        return risks
    
    def resolve_collisions(self) -> Dict[str, np.ndarray]:
        """
        冲突解决, 计算避障速度修正
        
        Returns:
            Dict[agent_id, velocity_correction]
        """
        corrections = {}
        
        for risk in self.collision_risks:
            if risk.severity in ["low"]:
                continue
            
            a = self.agents[risk.agent_a]
            b = self.agents[risk.agent_b]
            
            # 相对位置向量 (从 a 指向 b)
            rel_pos = b.position[:2] - a.position[:2]
            dist = np.linalg.norm(rel_pos)
            
            if dist < 1e-6:
                continue
            
            # 避障方向 (沿连线方向推开)
            avoid_dir = rel_pos / dist
            
            # 速度修正量 (根据严重程度)
            if risk.severity == "critical":
                magnitude = 2.0
            elif risk.severity == "high":
                magnitude = 1.0
            else:  # medium
                magnitude = 0.5
            
            # 应用到两个智能体
            correction = avoid_dir * magnitude * (5.0 - risk.time_to_collision) / 5.0
            
            if risk.agent_a in corrections:
                corrections[risk.agent_a] -= correction
            else:
                corrections[risk.agent_a] = -correction
            
            if risk.agent_b in corrections:
                corrections[risk.agent_b] += correction
            else:
                corrections[risk.agent_b] = correction
        
        return corrections
    
    def assign_tasks(self, tasks: List[Tuple[str, np.ndarray]]):
        """
        分布式任务分配 (最近邻算法)
        
        Args:
            tasks: List of (task_id, task_position)
        """
        available_agents = [
            (aid, a.position) 
            for aid, a in self.agents.items() 
            if a.task_id is None
        ]
        
        for task_id, task_pos in tasks:
            if not available_agents:
                break
            
            # 找最近可用智能体
            min_dist = float('inf')
            best_agent = None
            best_idx = -1
            
            for idx, (aid, a_pos) in enumerate(available_agents):
                dist = np.linalg.norm(task_pos - a_pos[:2])
                if dist < min_dist:
                    min_dist = dist
                    best_agent = aid
                    best_idx = idx
            
            if best_agent:
                self.agents[best_agent].task_id = task_id
                self.agents[best_agent].target = task_pos
                del available_agents[best_idx]
    
    def step(self, dt: float):
        """
        协调器主循环更新
        
        Args:
            dt: 时间步长 (s)
        """
        # 记录轨迹
        for agent_id, agent in self.agents.items():
            self.trajectories[agent_id].append(agent.position.copy())
            # 限制轨迹长度
            if len(self.trajectories[agent_id]) > 10000:
                self.trajectories[agent_id] = self.trajectories[agent_id][-5000:]
        
        # 检测碰撞
        self.detect_collisions()
        
        # 解决冲突
        if self.collision_risks:
            self.resolve_collisions()
    
    def get_formation_center(self, formation_id: str) -> np.ndarray:
        """获取编队几何中心"""
        if formation_id not in self.formations:
            return np.zeros(2)
        
        formation = self.formations[formation_id]
        assigned = [
            self.agents[slot.assigned_agent].position
            for slot in formation.slots
            if slot.assigned_agent and slot.assigned_agent in self.agents
        ]
        
        if not assigned:
            return formation.target_position
        
        return np.mean(assigned, axis=0)
    
    def get_status(self) -> Dict:
        """获取协调器状态摘要"""
        return {
            "total_agents": len(self.agents),
            "active_formations": len(self.formations),
            "agents_in_formation": sum(1 for a in self.agents.values() if a.in_formation),
            "collision_risks": len(self.collision_risks),
            "pending_tasks": len(self.task_queue)
        }
    
    def __len__(self):
        return len(self.agents)
    
    def __repr__(self):
        return f"MultiAgentCoordinator(agents={len(self.agents)}, formations={len(self.formations)})"


# AGV五级协调能力规格
AGV_COORDINATION_GRADES = {
    'S':  {'multi_agent': False, 'max_agents': 1,   'formation': False,    'collision_avoidance': False},
    'M':  {'multi_agent': False, 'max_agents': 1,   'formation': False,    'collision_avoidance': False},
    'L':  {'multi_agent': True,  'max_agents': 4,    'formation': True,     'collision_avoidance': "reactive"},
    'XL': {'multi_agent': True,  'max_agents': 10,   'formation': True,     'collision_avoidance': "predictive"},
    'XXL': {'multi_agent': True,  'max_agents': 20,   'formation': True,     'collision_avoidance': "optimal"},
}


def get_coordination_spec(grade: str) -> dict:
    """获取AGV指定等级的协调规格"""
    return AGV_COORDINATION_GRADES.get(grade, AGV_COORDINATION_GRADES['L'])
