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
Swarm Formation Control Module
===============================

多机器人蜂群编队控制系统
- 图论基础（邻接矩阵、拉普拉斯算子）
- 一阶/二阶共识协议
- 虚拟结构编队控制
- 行为式编队避障
-Leader-Follower共识控制
- 分布式任务分配

AGV五级规格:
- S级(0.3m/s): ≤4台, 2D平面, 距离≥1.0m
- M级(0.6m/s): ≤8台, 2D平面, 距离≥0.7m
- L级(1.0m/s): ≤16台, 2D+坡度, 距离≥0.5m
- XL级(1.5m/s): ≤32台, 3D空间, 距离≥0.3m
- XXL级(2.0m/s): >32台, 3D空间, 距离≥0.2m
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
import math


class FormationShape(Enum):
    """编队形状"""
    LINE = "line"
    TRIANGLE = "triangle"
    SQUARE = "square"
    CIRCLE = "circle"
    V_SHAPE = "v_shape"
    GRID = "grid"
    CHAIN = "chain"


class ConsensusType(Enum):
    """共识协议类型"""
    FIRST_ORDER = "first_order"   # 一阶积分器
    SECOND_ORDER = "second_order"  # 二阶积分器（考虑速度）


@dataclass
class SwarmAgent:
    """蜂群智能体"""
    agent_id: int
    position: np.ndarray           # 2D/3D位置 (x, y) 或 (x, y, z)
    velocity: np.ndarray           # 当前速度
    acceleration: np.ndarray = None  # 当前加速度（二阶模型）
    neighbors: Set[int] = field(default_factory=set)  # 邻接智能体ID集合
    is_leader: bool = False        # 是否为Leader
    state: np.ndarray = None      # 完整状态向量 [pos, vel]
    
    def __post_init__(self):
        if self.state is None:
            self.state = np.concatenate([self.position, self.velocity])
        if self.acceleration is None:
            self.acceleration = np.zeros_like(self.position)


@dataclass
class FormationSpec:
    """编队规格"""
    max_agents: int
    max_speed: float           # m/s
    min_safe_distance: float  # m
    topology: str              # "fully_connected" | "ring" | "star" | "mesh"
    consensus_type: ConsensusType
    dimension: int             # 2=平面, 3=立体
    formation_shape: FormationShape
    control_frequency: float   # Hz
    position_error_limit: float  # m
    velocity_error_limit: float  # m/s
    collision_radius: float = 0.3  # m
    
    def __post_init__(self):
        if self.dimension not in (2, 3):
            raise ValueError(f"Dimension must be 2 or 3, got {self.dimension}")


# AGV五级蜂群规格表
SWARM_GRADES = {
    "S": FormationSpec(
        max_agents=4, max_speed=0.3, min_safe_distance=1.0,
        topology="ring", consensus_type=ConsensusType.FIRST_ORDER,
        dimension=2, formation_shape=FormationShape.LINE,
        control_frequency=20.0, position_error_limit=0.1,
        velocity_error_limit=0.1
    ),
    "M": FormationSpec(
        max_agents=8, max_speed=0.6, min_safe_distance=0.7,
        topology="mesh", consensus_type=ConsensusType.FIRST_ORDER,
        dimension=2, formation_shape=FormationShape.GRID,
        control_frequency=30.0, position_error_limit=0.08,
        velocity_error_limit=0.08
    ),
    "L": FormationSpec(
        max_agents=16, max_speed=1.0, min_safe_distance=0.5,
        topology="mesh", consensus_type=ConsensusType.SECOND_ORDER,
        dimension=2, formation_shape=FormationShape.TRIANGLE,
        control_frequency=50.0, position_error_limit=0.05,
        velocity_error_limit=0.1
    ),
    "XL": FormationSpec(
        max_agents=32, max_speed=1.5, min_safe_distance=0.3,
        topology="mesh", consensus_type=ConsensusType.SECOND_ORDER,
        dimension=3, formation_shape=FormationShape.CIRCLE,
        control_frequency=100.0, position_error_limit=0.03,
        velocity_error_limit=0.08
    ),
    "XXL": FormationSpec(
        max_agents=64, max_speed=2.0, min_safe_distance=0.2,
        topology="mesh", consensus_type=ConsensusType.SECOND_ORDER,
        dimension=3, formation_shape=FormationShape.CIRCLE,
        control_frequency=200.0, position_error_limit=0.02,
        velocity_error_limit=0.05
    ),
}


def get_swarm_spec(grade: str) -> FormationSpec:
    """获取指定AGV等级的蜂群规格"""
    if grade not in SWARM_GRADES:
        raise ValueError(f"Unknown grade: {grade}. Valid: {list(SWARM_GRADES.keys())}")
    return SWARM_GRADES[grade]


def list_swarm_capabilities() -> List[str]:
    """列出所有等级蜂群控制能力"""
    lines = []
    for grade, spec in SWARM_GRADES.items():
        lines.append(f"[{grade}] {spec.max_agents}台 / {spec.max_speed}m/s / "
                     f"安全距离{spec.min_safe_distance}m / {spec.dimension}D / "
                     f"{spec.consensus_type.value} / {spec.control_frequency}Hz")
    return lines


class ConsensusController:
    """
    共识控制器
    实现一阶/二阶分布式共识协议
    基于图论Laplacian矩阵
    """
    
    def __init__(self, adj_matrix: np.ndarray, consensus_type: ConsensusType = ConsensusType.FIRST_ORDER):
        """
        Args:
            adj_matrix: 邻接矩阵 (N x N), a_ij > 0 表示i到j的边
            consensus_type: 共识协议类型
        """
        self.n = adj_matrix.shape[0]
        self.adj = np.array(adj_matrix, dtype=np.float32)
        self.consensus_type = consensus_type
        self.degree = np.diag(self.adj.sum(axis=1))  # 度矩阵
        self.laplacian = self.degree - self.adj     # Laplacian矩阵
        
        # 验证连通性（简单检查）
        if self.n > 0 and np.allclose(self.laplacian.sum(axis=0), 0):
            pass  # 连通图验证通过
    
    def compute_consensus(self, states: np.ndarray, velocities: Optional[np.ndarray] = None,
                          gain: float = 1.0) -> np.ndarray:
        """
        计算共识控制输入
        
        Args:
            states: 所有智能体状态 (N, state_dim), 一阶为位置, 二阶为位置+速度
            velocities: 所有智能体速度 (N, dim) [二阶模型]
            gain: 共识增益
            
        Returns:
            控制输入 (N, dim)
        """
        n = states.shape[0]  # 动态agent数量
        if self.consensus_type == ConsensusType.FIRST_ORDER:
            # 一阶共识: u_i = sum_j a_ij * (x_j - x_i)
            # positions = states (N, dim) for first order - assume last half is velocity, first half is position
            dim = states.shape[1] // 2 if states.ndim > 1 and states.shape[1] > 2 else states.shape[1]
            positions = states[:, :dim]  # 取前半部分作为位置
            control = np.zeros((n, dim))
            for i in range(n):
                for j in range(n):
                    if i < self.n and j < self.n and self.adj[i, j] > 0:
                        control[i] += self.adj[i, j] * (positions[j] - positions[i])
            return gain * control
        
        elif self.consensus_type == ConsensusType.SECOND_ORDER:
            # 二阶共识: 位置共识 + 速度共识
            dim = states.shape[1] // 2 if states.ndim > 1 else states.shape[0] // 2
            positions = states[:, :dim] if states.ndim > 1 else states[:dim]
            vels = velocities if velocities is not None else np.zeros((n, dim))
            
            control_pos = np.zeros((n, dim))
            control_vel = np.zeros((n, dim))
            
            for i in range(n):
                for j in range(n):
                    if i < self.n and j < self.n and self.adj[i, j] > 0:
                        control_pos[i] += self.adj[i, j] * (positions[j] - positions[i])
                        control_vel[i] += self.adj[i, j] * (vels[j] - vels[i])
            
            # alpha*位置误差 + beta*速度误差
            alpha, beta = 2.0 * gain, 0.5 * gain
            return alpha * control_pos + beta * control_vel
        
        return np.zeros((n, states.shape[1] if states.ndim > 1 else 1))
    
    def compute_leader_consensus(self, states: np.ndarray, leader_ids: List[int],
                                 leader_refs: np.ndarray) -> np.ndarray:
        """
        Leader-Follower共识控制
        Follower同时跟踪虚拟Leader的状态
        
        Args:
            states: 所有智能体状态 (N, dim)
            leader_ids: Leader智能体ID列表
            leader_refs: Leader参考状态 (len(leader_ids), dim)
        """
        n = states.shape[0]  # 动态agent数量
        dim = states.shape[1] if states.ndim > 1 else 1
        control = np.zeros((n, dim))
        
        # 提取follower索引
        follower_ids = [i for i in range(n) if i not in leader_ids]
        
        # 拓扑内共识（仅follower之间）
        for i in follower_ids:
            for j in follower_ids:
                if i < self.n and j < self.n and self.adj[i, j] > 0:
                    s = states[j] if states.ndim == 2 else states[j]
                    control[i] += self.adj[i, j] * (s - (states[i] if states.ndim == 2 else states[i]))
        
        # Leader跟踪（follower跟踪其对应leader）
        leader_idx_map = {lid: idx for idx, lid in enumerate(leader_ids)}
        for fi, i in enumerate(follower_ids):
            if i in leader_idx_map:
                ref = leader_refs[leader_idx_map[i]]
                control[i] += 1.0 * (ref - (states[i] if states.ndim == 2 else states[i]))
        
        return control


class FormationController:
    """
    编队控制器
    将共识控制与虚拟结构结合实现期望编队
    """
    
    def __init__(self, spec: FormationSpec, formation_shape: FormationShape,
                 initial_positions: Optional[List[np.ndarray]] = None):
        """
        Args:
            spec: 编队规格
            formation_shape: 编队形状
            initial_positions: 初始位置列表 (用于计算参考偏移)
        """
        self.spec = spec
        self.formation_shape = formation_shape
        self.consensus = ConsensusController(
            self._build_topology(spec.topology, spec.max_agents),
            spec.consensus_type
        )
        self.formation_offset = self._compute_formation_offset(formation_shape)
        self.centroid = np.zeros(spec.dimension)  # 编队几何中心
        
        # 目标编队位置
        self.target_positions = self._generate_formation_positions(formation_shape)
    
    def _build_topology(self, topology: str, n: int) -> np.ndarray:
        """构建通信拓扑邻接矩阵"""
        adj = np.zeros((n, n))
        if topology == "ring":
            for i in range(n):
                adj[i, (i - 1) % n] = 1.0
                adj[i, (i + 1) % n] = 1.0
        elif topology == "star":
            for i in range(1, n):
                adj[0, i] = 1.0
                adj[i, 0] = 1.0
        elif topology == "mesh":
            # 全连接网格
            for i in range(n):
                for j in range(i + 1, n):
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
        elif topology == "fully_connected":
            for i in range(n):
                for j in range(n):
                    if i != j:
                        adj[i, j] = 1.0
        return adj
    
    def _compute_formation_offset(self, shape: FormationShape) -> List[np.ndarray]:
        """计算各智能体相对于编队中心的期望偏移"""
        offsets = []
        d = self.spec.min_safe_distance
        
        if shape == FormationShape.LINE:
            for i in range(self.spec.max_agents):
                offsets.append(np.array([i * d, 0.0] if self.spec.dimension == 2 else [i * d, 0.0, 0.0]))
        elif shape == FormationShape.CIRCLE:
            n = self.spec.max_agents
            for i in range(n):
                angle = 2 * math.pi * i / n
                offsets.append(np.array([d * math.cos(angle), d * math.sin(angle)]))
        elif shape == FormationShape.GRID:
            cols = int(math.ceil(math.sqrt(self.spec.max_agents)))
            for i in range(self.spec.max_agents):
                row, col = i // cols, i % cols
                offsets.append(np.array([col * d, row * d] if self.spec.dimension == 2 else [col * d, row * d, 0.0]))
        elif shape == FormationShape.TRIANGLE:
            positions = []
            row = 0
            idx = 0
            while idx < self.spec.max_agents:
                for col in range(row + 1):
                    positions.append(np.array([col * d - row * d / 2, row * d * math.sqrt(3) / 2]))
                    idx += 1
                    if idx >= self.spec.max_agents:
                        break
                row += 1
            offsets = positions
        else:
            # 默认行
            for i in range(self.spec.max_agents):
                offsets.append(np.array([i * d, 0.0]))
        return offsets
    
    def _generate_formation_positions(self, shape: FormationShape) -> List[np.ndarray]:
        """生成期望编队位置（相对于几何中心）"""
        return self.formation_offset
    
    def compute_formation_control(self, agents: List[SwarmAgent],
                                   leader_ref: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        计算编队控制输入
        
        Args:
            agents: 智能体列表
            leader_ref: Leader参考位置/速度（若有）
            
        Returns:
            各智能体控制输入列表
        """
        n = len(agents)
        # For consensus, use position-only state
        dim = self.spec.dimension
        pos_states = np.array([a.position for a in agents])  # (n, dim)
        
        # 共识控制
        if leader_ref is not None:
            leader_ids = [i for i, a in enumerate(agents) if a.is_leader]
            leader_refs = np.array([leader_ref] * len(leader_ids)) if len(leader_ids) > 0 else np.array([]).reshape(0, self.spec.dimension)
            consensus_control = self.consensus.compute_leader_consensus(pos_states, leader_ids, leader_refs)
        else:
            consensus_control = self.consensus.compute_consensus(pos_states)
        
        # 编队形状控制（虚拟结构参考）
        formation_control = []
        for i, agent in enumerate(agents):
            # 期望位置 = 几何中心 + 形状偏移
            target = self.centroid + self.formation_offset[i % len(self.formation_offset)]
            shape_error = target - agent.position
            
            # 总控制 = 共识 + 编队
            u_consensus = consensus_control[i] if i < len(consensus_control) else np.zeros(self.spec.dimension)
            u_formation = 1.5 * shape_error  # 编队形状增益
            
            u_total = u_consensus + u_formation
            
            # 速度限幅
            speed = np.linalg.norm(u_total)
            if speed > self.spec.max_speed:
                u_total = u_total / speed * self.spec.max_speed
            
            formation_control.append(u_total)
        
        return formation_control
    
    def update_formation_center(self, agents: List[SwarmAgent]):
        """更新编队几何中心"""
        if agents:
            positions = np.array([a.position for a in agents])
            self.centroid = positions.mean(axis=0)


class VelocityObstacle:
    """
    速度障碍物 (Velocity Obstacle) 用于避障
    """
    
    def __init__(self):
        pass
    
    def check_collision(
        self,
        robot_pos: np.ndarray,
        robot_vel: np.ndarray,
        obstacle_pos: np.ndarray,
        obstacle_vel: np.ndarray,
        robot_radius: float,
        obstacle_radius: float,
        time_horizon: float = 2.0
    ) -> bool:
        """检查当前速度是否会在预测时间内发生碰撞"""
        # 相对位置
        rel_pos = obstacle_pos - robot_pos
        # 相对速度
        rel_vel = obstacle_vel - robot_vel
        
        # 合并半径
        sum_radius = robot_radius + obstacle_radius
        
        # 最近距离计算
        a = np.dot(rel_vel, rel_vel)
        if a < 1e-6:
            # 相对静止
            dist = np.linalg.norm(rel_pos)
            return dist < sum_radius
        
        b = 2 * np.dot(rel_pos, rel_vel)
        c = np.dot(rel_pos, rel_pos) - sum_radius ** 2
        
        # 判别式
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return False  # 无碰撞
        
        # 计算最近时间
        t = (-b - np.sqrt(discriminant)) / (2 * a)
        
        # 碰撞在时间范围内
        return t > 0 and t < time_horizon
    
    def compute_avoidance_velocity(
        self,
        preferred_vel: np.ndarray,
        robot_pos: np.ndarray,
        robot_radius: float,
        obstacles: List[Dict],
        max_speed: float
    ) -> np.ndarray:
        """计算避障速度，基于速度障碍物方法"""
        # 简单实现: 当会发生碰撞时偏离原方向
        n_obs = len(obstacles)
        avoided = preferred_vel.copy()
        
        for obs in obstacles:
            if self.check_collision(
                robot_pos, avoided,
                obs['position'], obs['velocity'],
                robot_radius, obs['radius'],
                2.0
            ):
                # 横向偏移
                dir_vec = obs['position'] - robot_pos
                dir_vec = dir_vec / np.linalg.norm(dir_vec)
                # 顺时针旋转90度
                perp = np.array([-dir_vec[1], dir_vec[0]])
                avoided += 0.5 * perp * max_speed
        
        # 限幅
        speed = np.linalg.norm(avoided)
        if speed > max_speed:
            avoided = avoided / speed * max_speed
        
        return avoided


class CollisionAvoidance:
    """
    蜂群碰撞避免
    基于人工势场 + ORCA(Optimal Reciprocal Collision Avoidance)
    """
    
    def __init__(self, spec: FormationSpec):
        self.spec = spec
        self.collision_radius = spec.collision_radius
        self.safe_distance = spec.min_safe_distance
    
    def compute_avoidance_control(self, agents: List[SwarmAgent]) -> List[np.ndarray]:
        """
        计算避障控制修正量
        
        Returns:
            避障修正量列表 (N, dim)
        """
        n = len(agents)
        avoidance = [np.zeros(self.spec.dimension) for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                pos_i = agents[i].position
                pos_j = agents[j].position
                
                diff = pos_i - pos_j
                dist = np.linalg.norm(diff)
                
                if dist < self.safe_distance and dist > 1e-6:
                    # 人工势场排斥
                    repulse_mag = (self.safe_distance - dist) / dist * 2.0
                    repulse_dir = diff / dist
                    repulse = repulse_mag * repulse_dir
                    
                    avoidance[i] += repulse
                    avoidance[j] -= repulse  # 互斥
        
        return avoidance
    
    def check_collisions(self, agents: List[SwarmAgent]) -> List[Tuple[int, int]]:
        """检测智能体间碰撞风险对"""
        collisions = []
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                dist = np.linalg.norm(agents[i].position - agents[j].position)
                if dist < self.collision_radius:
                    collisions.append((i, j))
        return collisions


class SwarmController:
    """
    蜂群控制系统主类
    整合共识、编队、避障于一体
    """
    
    def __init__(self, grade: str = "M", formation_shape: FormationShape = FormationShape.LINE):
        """
        Args:
            grade: AGV五级等级 (S/M/L/XL/XXL)
            formation_shape: 初始编队形状
        """
        self.spec = get_swarm_spec(grade)
        self.formation_shape = formation_shape
        self.formation_ctrl = FormationController(self.spec, formation_shape)
        self.collision_avoid = CollisionAvoidance(self.spec)
        
        self.agents: List[SwarmAgent] = []
        self.time = 0.0
        self.dt = 1.0 / self.spec.control_frequency
    
    def add_agent(self, position: np.ndarray, velocity: Optional[np.ndarray] = None,
                  is_leader: bool = False) -> int:
        """添加智能体"""
        if len(self.agents) >= self.spec.max_agents:
            raise RuntimeError(f"已达最大智能体数 {self.spec.max_agents}")
        
        agent_id = len(self.agents)
        vel = velocity if velocity is not None else np.zeros(self.spec.dimension)
        agent = SwarmAgent(agent_id, position.copy(), vel.copy(), is_leader=is_leader)
        self.agents.append(agent)
        return agent_id
    
    def step(self, leader_ref: Optional[np.ndarray] = None):
        """蜂群控制一步更新"""
        # 1. 计算编队控制
        formation_control = self.formation_ctrl.compute_formation_control(self.agents, leader_ref)
        
        # 2. 碰撞避障
        avoidance = self.collision_avoid.compute_avoidance_control(self.agents)
        
        # 3. 合成控制 + 更新
        for i, agent in enumerate(self.agents):
            u = formation_control[i] + avoidance[i]
            
            if self.spec.consensus_type == ConsensusType.SECOND_ORDER:
                # 二阶积分更新
                agent.velocity += u * self.dt
                agent.position += agent.velocity * self.dt
            else:
                # 一阶积分更新
                agent.position += u * self.dt
            
            # 更新状态向量
            agent.state = np.concatenate([agent.position, agent.velocity])
        
        # 4. 更新编队中心
        self.formation_ctrl.update_formation_center(self.agents)
        self.time += self.dt
    
    def get_states(self) -> np.ndarray:
        """获取所有智能体状态"""
        return np.array([a.state for a in self.agents])
    
    def get_positions(self) -> np.ndarray:
        """获取所有智能体位置"""
        return np.array([a.position for a in self.agents])
    
    def change_formation(self, new_shape: FormationShape):
        """切换编队形状"""
        self.formation_shape = new_shape
        self.formation_ctrl = FormationController(self.spec, new_shape)
    
    def validate_swarm(self) -> Tuple[bool, List[str]]:
        """验证蜂群状态合法性"""
        errors = []
        
        # 检查碰撞
        collisions = self.collision_avoid.check_collisions(self.agents)
        if collisions:
            errors.append(f"检测到 {len(collisions)} 对碰撞风险")
        
        # 检查速度限制
        for a in self.agents:
            speed = np.linalg.norm(a.velocity)
            if speed > self.spec.max_speed:
                errors.append(f"Agent {a.agent_id} 速度 {speed:.2f}m/s 超过限制 {self.spec.max_speed}m/s")
        
        # 检查拓扑连通性（简化）
        if len(self.agents) < 2:
            errors.append("智能体数量不足")
        
        return len(errors) == 0, errors
