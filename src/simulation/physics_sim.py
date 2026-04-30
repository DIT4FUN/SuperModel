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
PhysicsSim: 物理仿真引擎
=========================

刚体动力学 + 接触力学仿真:
- 2D/3D 刚体姿态与速度
- 弹簧阻尼接触力模型
- 摩擦力学 (库仑 + 粘性)
- AGV五级物理规格

适用于:
- 仿真环境中的触觉/力觉反馈
- 控制算法验证
- 传感器物理特性建模
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict
from enum import Enum


class BodyType(Enum):
    """刚体类型"""
    AGV_BASE = "agv_base"
    WHEEL = "wheel"
    GRIPPER = "gripper"
    FINGER = "finger"
    OBJECT = "object"
    OBSTACLE = "obstacle"


@dataclass
class RigidBody:
    """刚体状态"""
    position: np.ndarray          # 3, 位置 (x, y, z), m
    orientation: np.ndarray       # 4, 四元数 (qw, qx, qy, qz)
    linear_velocity: np.ndarray    # 3, 线速度, m/s
    angular_velocity: np.ndarray # 3, 角速度, rad/s
    mass: float                   # kg
    inertia: np.ndarray           # 3, 主惯性矩 (Ix, Iy, Iz)
    body_type: BodyType = BodyType.OBJECT
    name: str = "body"

    def __post_init__(self):
        if isinstance(self.position, list):
            self.position = np.array(self.position, dtype=np.float32)
        if isinstance(self.orientation, list):
            self.orientation = np.array(self.orientation, dtype=np.float32)
        if isinstance(self.linear_velocity, list):
            self.linear_velocity = np.array(self.linear_velocity, dtype=np.float32)
        if isinstance(self.angular_velocity, list):
            self.angular_velocity = np.array(self.angular_velocity, dtype=np.float32)
        if isinstance(self.inertia, list):
            self.inertia = np.array(self.inertia, dtype=np.float32)

    @property
    def kinetic_energy(self) -> float:
        """动能 = 1/2*m*v² + 1/2*ωᵀ*I*ω"""
        T_linear = 0.5 * self.mass * np.dot(self.linear_velocity, self.linear_velocity)
        T_angular = 0.5 * np.dot(
            self.angular_velocity * self.inertia,
            self.angular_velocity
        )
        return T_linear + T_angular

    def to_pose_matrix(self) -> np.ndarray:
        """转4x4变换矩阵"""
        q = self.orientation
        # 四元数转旋转矩阵
        R = np.array([
            [1-2*(q[2]**2+q[3]**2), 2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2])],
            [2*(q[1]*q[2]+q[0]*q[3]), 1-2*(q[1]**2+q[3]**2), 2*(q[2]*q[3]-q[0]*q[1])],
            [2*(q[1]*q[3]-q[0]*q[2]), 2*(q[2]*q[3]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2)]
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = self.position
        return T


@dataclass
class ContactPoint:
    """接触点"""
    position: np.ndarray      # 3, 接触点位置 (世界坐标系)
    normal: np.ndarray       # 3, 法向量 (指向body1)
    penetration: float        # 穿透深度, m
    velocity: np.ndarray     # 3, 相对速度, m/s


@dataclass
class ContactForce:
    """接触力"""
    normal_force: float       # 法向力, N
    friction_force: np.ndarray  # 3, 摩擦力向量, N
    torque: np.ndarray        # 3, 接触力矩, N·m


@dataclass
class PhysicsSimConfig:
    """物理仿真配置"""
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))
    dt: float = 0.001         # 时间步长, s
    substeps: int = 10        # 子步数 (提高稳定性)
    
    # 接触参数
    restitution: float = 0.2   # 恢复系数
    friction_static: float = 0.6  # 静摩擦系数
    friction_dynamic: float = 0.4  # 动摩擦系数
    contact_stiffness: float = 10000.0  # N/m
    contact_damping: float = 100.0      # N·s/m
    
    # AGV等级
    grade: str = 'M'
    
    @classmethod
    def for_grade(cls, grade: str) -> 'PhysicsSimConfig':
        """根据AGV等级获取物理仿真配置"""
        configs = {
            'S':  cls(dt=0.002, substeps=5, contact_stiffness=5000, grade='S'),
            'M':  cls(dt=0.001, substeps=10, contact_stiffness=10000, grade='M'),
            'L':  cls(dt=0.0005, substeps=20, contact_stiffness=20000, grade='L'),
            'XL': cls(dt=0.0002, substeps=50, contact_stiffness=50000, grade='XL'),
            'XXL': cls(dt=0.0001, substeps=100, contact_stiffness=100000, grade='XXL'),
        }
        return configs.get(grade, cls())


class PhysicsSimulator:
    """
    物理仿真引擎
    
    功能:
    - 刚体动力学积分 (半隐式欧拉)
    - 接触力计算 (弹簧阻尼模型)
    - 摩擦力学
    - AGV五级物理规格
    """
    
    def __init__(self, config: Optional[PhysicsSimConfig] = None):
        self.config = config or PhysicsSimConfig()
        self.gravity = self.config.gravity
        
        # 刚体列表
        self.bodies: List[RigidBody] = []
        self._body_map: Dict[str, RigidBody] = {}
        
        # 接触对缓存
        self._contact_cache: List[Tuple[int, int]] = []
        
        # 仿真时间
        self.sim_time: float = 0.0
        self._step_count: int = 0
        
    def add_body(self, body: RigidBody) -> int:
        """添加刚体
        
        Returns:
            body_id: 刚体索引
        """
        body_id = len(self.bodies)
        self.bodies.append(body)
        self._body_map[body.name] = body
        return body_id
    
    def get_body(self, name: str) -> Optional[RigidBody]:
        """获取刚体"""
        return self._body_map.get(name)
    
    def step(self, dt: Optional[float] = None):
        """
        仿真一步
        
        Args:
            dt: 时间步长 (默认使用配置值)
        """
        if dt is None:
            dt = self.config.dt
        
        sub_dt = dt / self.config.substeps
        
        for _ in range(self.config.substeps):
            self._step_single(sub_dt)
        
        self.sim_time += dt
        self._step_count += 1
    
    def _step_single(self, dt: float):
        """单步积分"""
        # 1. 计算每个刚体的受力
        forces = {}
        torques = {}
        
        for i, body in enumerate(self.bodies):
            # 重力
            f_gravity = self.gravity * body.mass
            t_gravity = np.zeros(3)
            
            # 初始受力
            forces[i] = f_gravity
            torques[i] = t_gravity
        
        # 2. 接触力
        contacts = self._detect_contacts()
        for contact in contacts:
            self._apply_contact_forces(contact, forces, torques)
        
        # 3. 积分 (半隐式欧拉)
        for i, body in enumerate(self.bodies):
            # 线速度更新
            accel = forces[i] / body.mass
            body.linear_velocity = body.linear_velocity + accel * dt
            body.position = body.position + body.linear_velocity * dt
            
            # 角速度更新
            alpha = torques[i] / body.inertia
            body.angular_velocity = body.angular_velocity + alpha * dt
            body.orientation = self._integrate_orientation(
                body.orientation, body.angular_velocity, dt
            )
            
            # 阻尼 (简化)
            body.linear_velocity *= 0.999
            body.angular_velocity *= 0.999
    
    def _detect_contacts(self) -> List[ContactPoint]:
        """检测接触 (简化: 地面 + 简单AABB)"""
        contacts = []
        ground_z = 0.0
        
        for body in self.bodies:
            # 简化为球体近似
            radius = self._estimate_radius(body)
            pos = body.position
            
            # 与地面接触
            if pos[2] - radius < ground_z:
                penetration = ground_z - (pos[2] - radius)
                contact = ContactPoint(
                    position=np.array([pos[0], pos[1], ground_z]),
                    normal=np.array([0.0, 0.0, 1.0]),
                    penetration=penetration,
                    velocity=body.linear_velocity.copy()
                )
                contacts.append(contact)
        
        return contacts
    
    def _apply_contact_forces(
        self,
        contact: ContactPoint,
        forces: Dict[int, np.ndarray],
        torques: Dict[int, np.ndarray]
    ):
        """计算并施加接触力"""
        # 弹簧阻尼法向力
        fn = self.config.contact_stiffness * contact.penetration
        fn += self.config.contact_damping * np.dot(contact.normal, contact.velocity)
        fn = max(0, fn)  # 法向力不能为负
        
        # 摩擦力
        tangent_vel = contact.velocity - np.dot(contact.velocity, contact.normal) * contact.normal
        tangent_speed = np.linalg.norm(tangent_vel)
        
        if tangent_speed > 1e-6:
            tangent_dir = tangent_vel / tangent_speed
            max_friction = self.config.friction_static * fn
            
            # 静摩擦/动摩擦切换
            if tangent_speed < 0.01:
                ft_mag = min(max_friction, self.config.contact_damping * tangent_speed * 10)
            else:
                ft_mag = self.config.friction_dynamic * fn
            
            ft = -tangent_dir * ft_mag
        else:
            ft = np.zeros(3)
        
        # 查找受力的刚体 (假设地面是静态的)
        friction_torque = np.cross(contact.position, ft)
        
        for i, body in enumerate(self.bodies):
            # 简化: 只对最近的刚体施力
            dist = np.linalg.norm(body.position - contact.position)
            if dist < 1.0:  # 阈值
                forces[i] = forces[i] + contact.normal * fn + ft
                torques[i] = torques[i] + friction_torque
                break
    
    def _estimate_radius(self, body: RigidBody) -> float:
        """估算刚体等效半径"""
        # 基于惯性矩估算
        I_avg = np.mean(body.inertia)
        r = (5 * I_avg / (2 * body.mass)) ** 0.5
        return max(0.01, min(r, 2.0))
    
    def _integrate_orientation(
        self,
        q: np.ndarray,
        omega: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """四元数积分 (欧拉法)"""
        # 四元数导数: q̇ = 0.5 * Ω ⊗ q
        omega_quat = np.array([0, omega[0], omega[1], omega[2]])
        
        q_dot = 0.5 * self._quat_multiply(omega_quat, q)
        q_new = q + q_dot * dt
        
        return q_new / np.linalg.norm(q_new)
    
    @staticmethod
    def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """四元数乘法"""
        return np.array([
            a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
            a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
            a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
            a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
        ])
    
    def simulate_drop(
        self,
        body_name: str,
        drop_height: float = 1.0,
        duration: float = 2.0
    ) -> Dict[str, List]:
        """
        模拟自由落体
        
        Args:
            body_name: 刚体名称
            drop_height: 初始高度, m
            duration: 仿真时长, s
            
        Returns:
            仿真轨迹数据
        """
        body = self.get_body(body_name)
        if body is None:
            raise ValueError(f"Body {body_name} not found")
        
        # 重置到初始位置
        body.position = np.array([0.0, 0.0, drop_height])
        body.linear_velocity = np.zeros(3)
        body.angular_velocity = np.zeros(3)
        
        # 记录数据
        times = []
        positions = []
        velocities = []
        energies = []
        
        n_steps = int(duration / self.config.dt)
        n_record = max(1, n_steps // 100)  # 每100步记录一次
        
        for step in range(n_steps):
            self.step()
            
            if step % n_record == 0:
                times.append(self.sim_time)
                positions.append(body.position.copy())
                velocities.append(body.linear_velocity.copy())
                energies.append(body.kinetic_energy)
            
            # 接触地面后减速
            if body.position[2] <= 0.01 and body.linear_velocity[2] < 0:
                break
        
        return {
            'time': times,
            'position': positions,
            'velocity': velocities,
            'energy': energies
        }
    
    def simulate_collision(
        self,
        body1_name: str,
        body2_name: str,
        impact_velocity: Tuple[float, float, float] = (0.0, 0.0, -2.0),
        duration: float = 0.5
    ) -> Dict[str, List]:
        """
        模拟碰撞事件
        
        Args:
            body1_name: 物体1名称
            body2_name: 物体2名称
            impact_velocity: 碰撞速度, m/s
            duration: 仿真时长, s
            
        Returns:
            碰撞力/能量变化数据
        """
        body1 = self.get_body(body1_name)
        body2 = self.get_body(body2_name)
        
        if body1 is None or body2 is None:
            raise ValueError(f"Body not found")
        
        # 设置初始速度
        body1.linear_velocity = np.array(impact_velocity, dtype=np.float32)
        body2.linear_velocity = np.zeros(3)
        
        times = []
        forces = []
        energies = []
        ke1_before = body1.kinetic_energy
        
        n_steps = int(duration / self.config.dt)
        n_record = max(1, n_steps // 100)
        
        for step in range(n_steps):
            self.step()
            
            if step % n_record == 0:
                times.append(self.sim_time)
                # 估算碰撞力 (通过加速度)
                approx_force = body1.mass * np.linalg.norm(
                    (body1.linear_velocity - body2.linear_velocity) / self.config.dt
                )
                forces.append(approx_force)
                energies.append((body1.kinetic_energy, body2.kinetic_energy))
        
        return {
            'time': times,
            'force': forces,
            'energy': energies,
            'total_energy_before': ke1_before,
            'total_energy_after': sum(e[0] + e[1] for e in energies[-3:]) / 3
        }


# ─── AGV五级物理规格 ────────────────────────────────────────────────────────

AGV_PHYSICS_GRADES = {
    'S': {
        'mass_range': (10, 50),        # kg
        'size_range': (0.3, 0.6),      # m
        'max_velocity': 0.5,           # m/s
        'max_accel': 0.5,              # m/s^2
        'max_angular_vel': 1.0,        # rad/s
        'contact_stiffness': 5000,     # N/m
        'damping': 50,                 # N·s/m
        'friction_static': 0.5,
        'friction_dynamic': 0.3,
        'sim_dt': 0.002,
        'control_rate': 50,           # Hz
    },
    'M': {
        'mass_range': (30, 150),       # kg
        'size_range': (0.4, 0.8),      # m
        'max_velocity': 1.5,           # m/s
        'max_accel': 1.5,              # m/s^2
        'max_angular_vel': 2.0,        # rad/s
        'contact_stiffness': 10000,    # N/m
        'damping': 100,                # N·s/m
        'friction_static': 0.6,
        'friction_dynamic': 0.4,
        'sim_dt': 0.001,
        'control_rate': 100,
    },
    'L': {
        'mass_range': (80, 400),       # kg
        'size_range': (0.6, 1.2),       # m
        'max_velocity': 2.0,           # m/s
        'max_accel': 2.0,              # m/s^2
        'max_angular_vel': 3.0,        # rad/s
        'contact_stiffness': 20000,    # N/m
        'damping': 200,                # N·s/m
        'friction_static': 0.7,
        'friction_dynamic': 0.5,
        'sim_dt': 0.0005,
        'control_rate': 200,
    },
    'XL': {
        'mass_range': (150, 800),      # kg
        'size_range': (0.8, 1.5),      # m
        'max_velocity': 2.5,           # m/s
        'max_accel': 2.5,              # m/s^2
        'max_angular_vel': 5.0,        # rad/s
        'contact_stiffness': 50000,    # N/m
        'damping': 500,                # N·s/m
        'friction_static': 0.8,
        'friction_dynamic': 0.6,
        'sim_dt': 0.0002,
        'control_rate': 500,
    },
    'XXL': {
        'mass_range': (300, 1500),     # kg
        'size_range': (1.0, 2.0),      # m
        'max_velocity': 3.0,           # m/s
        'max_accel': 3.0,              # m/s^2
        'max_angular_vel': 10.0,      # rad/s
        'contact_stiffness': 100000,   # N/m
        'damping': 1000,               # N·s/m
        'friction_static': 0.9,
        'friction_dynamic': 0.7,
        'sim_dt': 0.0001,
        'control_rate': 1000,
    },
}


def get_physics_spec(grade: str) -> dict:
    """获取AGV指定等级的物理规格"""
    return AGV_PHYSICS_GRADES.get(grade, AGV_PHYSICS_GRADES['M'])


def create_physics_sim_for_grade(grade: str) -> PhysicsSimulator:
    """为指定AGV等级创建物理仿真器"""
    spec = get_physics_spec(grade)
    config = PhysicsSimConfig(
        dt=spec['sim_dt'],
        contact_stiffness=spec['contact_stiffness'],
        contact_damping=spec['damping'],
        friction_static=spec['friction_static'],
        friction_dynamic=spec['friction_dynamic'],
        grade=grade
    )
    return PhysicsSimulator(config)


def create_agv_body(
    name: str,
    grade: str = 'M',
    body_type: BodyType = BodyType.AGV_BASE
) -> RigidBody:
    """为指定AGV等级创建刚体"""
    spec = get_physics_spec(grade)
    mass = (spec['mass_range'][0] + spec['mass_range'][1]) / 2
    size = (spec['size_range'][0] + spec['size_range'][1]) / 2
    
    # 惯性矩 (球体近似 I = 2/5 * m * r²)
    r = size / 2
    I = np.array([0.4 * mass * r**2, 0.4 * mass * r**2, 0.4 * mass * r**2])
    
    return RigidBody(
        position=np.array([0.0, 0.0, size/2]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        mass=float(mass),
        inertia=I,
        body_type=body_type,
        name=name
    )
