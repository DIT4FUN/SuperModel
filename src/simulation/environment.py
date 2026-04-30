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
仿真环境模块
============

基础机器人仿真环境
- 简化运动学/动力学仿真
- 传感器噪声注入
- 场景管理
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import time


@dataclass
class SimConfig:
    """仿真配置"""
    dt: float = 0.01              # 时间步长 (s)
    num_joints: int = 6          # 关节数
    gravity: np.ndarray = field(default_factory=lambda: np.array([0, 0, -9.81]))
    # 噪声参数
    position_noise: float = 0.001   # m
    velocity_noise: float = 0.01    # m/s
    accel_noise: float = 0.1        # m/s^2
    # 延迟参数
    sensor_delay: float = 0.0       # s
    control_delay: float = 0.0      # s
    # 仿真引擎
    engine: str = "custom"         # "custom" / "pybullet" / "mujoco"


class RobotSimulator:
    """
    机器人仿真器
    
    简化仿真，不依赖外部物理引擎
    实现:
    - 关节空间运动学
    - 一阶动力学响应
    - 碰撞检测 (简化)
    """
    
    def __init__(
        self,
        config: Optional[SimConfig] = None,
        joint_limits_lower: Optional[np.ndarray] = None,
        joint_limits_upper: Optional[np.ndarray] = None
    ):
        self.config = config or SimConfig()
        self.dt = self.config.dt
        
        n = self.config.num_joints
        self.n = n
        
        # 关节状态
        self.joint_positions = np.zeros(n)
        self.joint_velocities = np.zeros(n)
        self.joint_accelerations = np.zeros(n)
        self.joint_torques = np.zeros(n)
        
        # 末端执行器
        self.end_effector_pose = np.eye(4)
        
        # 关节限位
        self.jl_lower = joint_limits_lower or -np.ones(n) * np.pi
        self.jl_upper = joint_limits_upper or np.ones(n) * np.pi
        
        # 质量矩阵 (简化)
        self.mass_matrix = np.eye(n) * 0.5
        
        # 阻尼
        self.damping = np.ones(n) * 2.0
        
        # 时间
        self._time = 0.0
        self._step_count = 0
        
        # 回调
        self._callbacks: List[callable] = []
        
    def set_joint_positions(self, positions: np.ndarray):
        """设置关节位置"""
        self.joint_positions = np.clip(positions, self.jl_lower, self.jl_upper)
        
    def step(self, torque_command: np.ndarray) -> Dict[str, Any]:
        """
        仿真一步
        
        Args:
            torque_command: 关节力矩命令 (n,)
            
        Returns:
            state: 当前状态字典
        """
        dt = self.dt
        
        # 简化动力学: tau = M*qdd + C*qd + K*(q_target - q)
        # 这里简化为: qdd = (torque - damping*qd) / mass
        
        tau = np.asarray(torque_command)
        if len(tau) != self.n:
            tau = np.zeros(self.n)
        
        # 加速度
        self.joint_accelerations = (tau - self.damping * self.joint_velocities) / np.diag(self.mass_matrix)
        
        # 速度积分
        self.joint_velocities += self.joint_accelerations * dt
        
        # 位置积分
        self.joint_positions += self.joint_velocities * dt
        
        # 限位反弹
        for i in range(self.n):
            if self.joint_positions[i] < self.jl_lower[i]:
                self.joint_positions[i] = self.jl_lower[i]
                self.joint_velocities[i] *= -0.5  # 反弹
            elif self.joint_positions[i] > self.jl_upper[i]:
                self.joint_positions[i] = self.jl_upper[i]
                self.joint_velocities[i] *= -0.5
        
        # 更新末端执行器 (简化正运动学)
        self._update_end_effector()
        
        # 时间
        self._time += dt
        self._step_count += 1
        
        # 回调
        for cb in self._callbacks:
            cb(self.get_state())
        
        return self.get_state()
    
    def _update_end_effector(self):
        """简化正运动学: 末端执行器位置"""
        # 简化模型: 假设关节是串联臂
        # 累积关节角度影响
        cumulative = np.zeros(3)
        for i, angle in enumerate(self.joint_positions):
            cumulative += np.array([
                np.cos(angle * (i + 1) * 0.3) * 0.1,
                np.sin(angle * (i + 1) * 0.3) * 0.1,
                0.3  # 基础高度
            ])
        
        # 简化: 位置 = 累积
        self.end_effector_pose = np.eye(4)
        self.end_effector_pose[:3, 3] = cumulative
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'time': self._time,
            'step': self._step_count,
            'joint_positions': self.joint_positions.copy(),
            'joint_velocities': self.joint_velocities.copy(),
            'joint_accelerations': self.joint_accelerations.copy(),
            'end_effector_pose': self.end_effector_pose.copy(),
            'end_effector_position': self.end_effector_pose[:3, 3].copy(),
        }
    
    def get_jacobian(self, joint_positions: Optional[np.ndarray] = None) -> np.ndarray:
        """
        获取雅可比矩阵 (简化)
        
        Returns: 6 x n_joints
        """
        if joint_positions is None:
            joint_positions = self.joint_positions
        
        # 简化雅可比 (数值估计)
        J = np.zeros((6, self.n))
        
        # 数值微分
        eps = 1e-6
        for i in range(self.n):
            q_plus = joint_positions.copy()
            q_plus[i] += eps
            
            # 简化: 假设末端位置与累积角度相关
            pos_plus = np.sum([np.cos(q_plus[j] * (j+1) * 0.3) * 0.1 for j in range(self.n)])
            J[:3, i] = (pos_plus / eps) * 0.1  # 简化梯度
        
        return J
    
    def check_self_collision(self) -> bool:
        """自碰撞检测 (简化)"""
        # 简化: 不检测
        return False
    
    def check_environment_collision(self, obstacles: List[Dict]) -> List[Dict]:
        """
        环境碰撞检测 (简化)
        
        Args:
            obstacles: [{"type": "sphere", "center": np.array, "radius": float}, ...]
        """
        ee_pos = self.end_effector_pose[:3, 3]
        collisions = []
        
        for obs in obstacles:
            if obs['type'] == 'sphere':
                dist = np.linalg.norm(ee_pos - obs['center'])
                if dist < obs.get('radius', 0.1):
                    collisions.append(obs)
        
        return collisions
    
    def add_callback(self, callback: callable):
        """添加状态回调"""
        self._callbacks.append(callback)
    
    def reset(self):
        """重置仿真"""
        self.joint_positions = np.zeros(self.n)
        self.joint_velocities = np.zeros(self.n)
        self.joint_accelerations = np.zeros(self.n)
        self.joint_torques = np.zeros(self.n)
        self._time = 0.0
        self._step_count = 0


class SensorSimulator:
    """
    传感器仿真器
    
    为 RobotSimulator 添加带噪声的传感器输出
    """
    
    def __init__(self, simulator: RobotSimulator, config: Optional[SimConfig] = None):
        self.sim = simulator
        self.config = config or SimConfig()
        
        # 传感器数据缓冲
        self._imu_buffer = []
        self._joint_state_buffer = []
        
    def get_noisy_joint_positions(self) -> np.ndarray:
        """获取带噪声的关节位置"""
        noise = np.random.randn(self.sim.n) * self.config.position_noise
        return self.sim.joint_positions + noise
    
    def get_noisy_joint_velocities(self) -> np.ndarray:
        """获取带噪声的关节速度"""
        noise = np.random.randn(self.sim.n) * self.config.velocity_noise
        return self.sim.joint_velocities + noise
    
    def get_imu_data(self) -> Dict[str, np.ndarray]:
        """
        仿真 IMU 数据
        
        Returns:
            {"accel": 3, "gyro": 3, "timestamp": float}
        """
        # 角速度来自关节速度
        gyro = self.sim.joint_velocities[:3] * 0.5 + np.random.randn(3) * 0.01
        
        # 加速度 = 重力 + 运动加速度
        accel_noise = np.random.randn(3) * self.config.accel_noise
        accel = np.array([0, 0, 9.81]) + self.sim.joint_accelerations[:3] * 0.1 + accel_noise
        
        return {
            'accel': accel.astype(np.float32),
            'gyro': gyro.astype(np.float32),
            'timestamp': self.sim._time
        }
    
    def get_wrench(self) -> np.ndarray:
        """
        仿真末端力矩传感器数据
        
        Returns:
            wrench: 6D [Fx, Fy, Fz, Tx, Ty, Tz]
        """
        # 简化: 力矩来自关节力
        wrench = np.zeros(6)
        wrench[:3] = self.sim.joint_torques[:3] * 0.1 + np.random.randn(3) * 0.5
        wrench[3:] = np.random.randn(3) * 0.1
        return wrench
    
    def get_contact_force(self) -> float:
        """
        仿真接触力
        
        Returns:
            contact_force: 标量接触力 (N)
        """
        # 简化: 基于关节加速度
        return max(0, -np.sum(self.sim.joint_accelerations) * 5.0)
    
    def apply_sensor_delay(self, data: Any) -> Any:
        """应用传感器延迟 (简化)"""
        # 简化: 不实现实际延迟
        return data


class PhysicsEngine:
    """
    物理引擎适配层
    
    支持不同仿真引擎的统一接口
    """
    
    def __init__(self, engine: str = "custom", config: Optional[Dict] = None):
        self.engine = engine
        self.config = config or {}
        
        if engine == "pybullet":
            self._init_pybullet()
        elif engine == "mujoco":
            self._init_mujoco()
        else:
            self.simulator = RobotSimulator(SimConfig(**self.config))
            self.sensor_sim = SensorSimulator(self.simulator)
    
    def _init_pybullet(self):
        """初始化 PyBullet"""
        try:
            import pybullet as p
            self.pybullet = p
            
            # 连接物理引擎
            cid = p.connect(p.SHARED_MEMORY_SERVER if self.config.get('shared_memory', False) else p.DIRECT)
            if cid < 0:
                cid = p.connect(p.DIRECT)
            
            # 设置重力
            gravity = self.config.get('gravity', [0, 0, -9.81])
            p.setGravity(*gravity, physicsClientId=cid)
            
            # 设置时间步
            dt = self.config.get('dt', 0.01)
            p.setTimeStep(dt, physicsClientId=cid)
            
            # 创建地面
            plane_id = p.loadURDF("plane.urdf", [0, 0, 0], physicsClientId=cid)
            
            # 创建机器人 (如果提供了 URDF 路径)
            robot_urdf = self.config.get('robot_urdf')
            if robot_urdf:
                base_pos = self.config.get('robot_base_pos', [0, 0, 0])
                base_orn = self.config.get('robot_base_orn', [0, 0, 0, 1])
                self.robot_id = p.loadURDF(robot_urdf, base_pos, base_orn, physicsClientId=cid)
            else:
                self.robot_id = -1
            
            self._client_id = cid
            self.simulator = None  # PyBullet 自己管理状态
            
            print(f"[PhysicsEngine] PyBullet initialized (client={cid}, robot={self.robot_id})")
            
        except ImportError:
            print("[PhysicsEngine] PyBullet not installed, falling back to custom")
            self.engine = "custom"
            self.simulator = RobotSimulator()
            self.sensor_sim = SensorSimulator(self.simulator)
    
    def _init_mujoco(self):
        """初始化 MuJoCo"""
        try:
            import mujoco
            self.mujoco = mujoco
            
            # 创建默认模型或加载 MJCF
            model_path = self.config.get('model_xml')
            if model_path:
                self._model = mujoco.MjModel.from_xml_path(model_path)
            else:
                # 创建简单的双连杆模型
                xml_string = """
                <mujoco model="simple_arm">
                    <option timestep="0.01" gravity="0 0 -9.81"/>
                    <worldbody>
                        <body name="base" pos="0 0 0">
                            <geom type="box" size="0.1 0.1 0.05"/>
                            <joint type="free"/>
                            <body name="link1" pos="0 0 0.1">
                                <geom type="cylinder" size="0.03 0.1"/>
                                <joint type="hinge" axis="0 1 0"/>
                                <body name="link2" pos="0.2 0 0">
                                    <geom type="cylinder" size="0.025 0.15"/>
                                    <joint type="hinge" axis="0 1 0"/>
                                </body>
                            </body>
                        </body>
                    </worldbody>
                    <actuator>
                        <motor joint="hinge" gear="100"/>
                    </actuator>
                </mujoco>
                """
                self._model = mujoco.MjModel.from_xml_string(xml_string)
            
            self._data = mujoco.MjData(self._model)
            
            # 设置控制
            self._num_joints = self._model.nu
            
            print(f"[PhysicsEngine] MuJoCo initialized ({self._num_joints} actuators)")
            
        except ImportError:
            print("[PhysicsEngine] MuJoCo not installed, falling back to custom")
            self.engine = "custom"
            self.simulator = RobotSimulator()
            self.sensor_sim = SensorSimulator(self.simulator)
    
    def step(self, torque: np.ndarray) -> Dict:
        """一步仿真"""
        return self.simulator.step(torque)
    
    def get_state(self) -> Dict:
        """获取状态"""
        return self.simulator.get_state()


class SceneManager:
    """
    仿真场景管理器
    
    管理多物体场景，支持:
    - 物体添加/删除/移动
    - 碰撞查询
    - 抓取/放置操作
    """
    
    def __init__(self):
        self.objects: Dict[str, Dict] = {}
        self.grasp_target: Optional[str] = None
        self._object_counter = 0
    
    def add_object(
        self,
        name: str,
        obj_type: str,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        size: Optional[np.ndarray] = None,
        mass: float = 0.1,
        friction: float = 0.5,
        color: Optional[np.ndarray] = None
    ) -> str:
        """添加物体到场景"""
        if name is None:
            name = f"object_{self._object_counter}"
            self._object_counter += 1
        
        self.objects[name] = {
            'type': obj_type,
            'position': position.copy() if isinstance(position, np.ndarray) else np.array(position),
            'orientation': orientation.copy() if orientation is not None else np.array([1, 0, 0, 0]),
            'size': size,
            'mass': mass,
            'friction': friction,
            'color': color,
            'grasped': False,
            'grasped_by': None
        }
        return name
    
    def remove_object(self, name: str) -> bool:
        """从场景移除物体"""
        if name in self.objects:
            del self.objects[name]
            return True
        return False
    
    def move_object(self, name: str, position: np.ndarray) -> bool:
        """移动场景中的物体"""
        if name in self.objects and not self.objects[name]['grasped']:
            self.objects[name]['position'] = np.array(position)
            return True
        return False
    
    def grasp(self, object_name: str) -> bool:
        """尝试抓取物体"""
        if object_name in self.objects and not self.objects[object_name]['grasped']:
            self.objects[object_name]['grasped'] = True
            self.grasp_target = object_name
            return True
        return False
    
    def release(self) -> Optional[str]:
        """释放当前抓取的物体"""
        if self.grasp_target:
            name = self.grasp_target
            self.objects[name]['grasped'] = False
            self.grasp_target = None
            return name
        return None
    
    def get_object(self, name: str) -> Optional[Dict]:
        """获取物体信息"""
        return self.objects.get(name)
    
    def get_all_objects(self) -> List[Dict]:
        """获取所有物体"""
        return list(self.objects.values())
    
    def get_object_positions(self) -> Dict[str, np.ndarray]:
        """获取所有物体位置"""
        return {name: obj['position'].copy() for name, obj in self.objects.items()}


class TrajectoryRecorder:
    """
    轨迹记录器
    
    记录机器人运动轨迹，支持回放和可视化
    """
    
    def __init__(self):
        self.joint_trajectory: List[np.ndarray] = []
        self.cartesian_trajectory: List[np.ndarray] = []
        self.wrench_history: List[np.ndarray] = []
        self.timestamps: List[float] = []
        self._start_time = None
    
    def record(
        self,
        joint_positions: np.ndarray,
        cartesian_position: Optional[np.ndarray] = None,
        wrench: Optional[np.ndarray] = None
    ):
        """记录一个数据点"""
        if self._start_time is None:
            self._start_time = time.time()
        
        self.joint_trajectory.append(joint_positions.copy())
        if cartesian_position is not None:
            self.cartesian_trajectory.append(np.array(cartesian_position))
        if wrench is not None:
            self.wrench_history.append(np.array(wrench))
        self.timestamps.append(time.time() - self._start_time)
    
    def get_joint_trajectory(self) -> np.ndarray:
        """获取关节轨迹"""
        if not self.joint_trajectory:
            return np.zeros((0, 6))
        return np.array(self.joint_trajectory)
    
    def get_cartesian_trajectory(self) -> Optional[np.ndarray]:
        """获取笛卡尔轨迹"""
        if not self.cartesian_trajectory:
            return None
        return np.array(self.cartesian_trajectory)
    
    def get_duration(self) -> float:
        """获取记录总时长"""
        if not self.timestamps:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]
    
    def clear(self):
        """清空记录"""
        self.joint_trajectory.clear()
        self.cartesian_trajectory.clear()
        self.wrench_history.clear()
        self.timestamps.clear()
        self._start_time = None
    
    def export(self, path: str):
        """导出为 numpy 文件"""
        data = {
            'joint_trajectory': self.get_joint_trajectory(),
            'cartesian_trajectory': self.get_cartesian_trajectory(),
            'wrench_history': np.array(self.wrench_history) if self.wrench_history else np.zeros((0, 6)),
            'timestamps': np.array(self.timestamps)
        }
        np.savez_compressed(path, **data)


# 仿真场景预设
PRESET_SCENES = {
    "tabletop": {
        "description": "桌面抓取场景",
        "obstacles": [
            {"type": "sphere", "center": np.array([0.5, 0.0, 0.3]), "radius": 0.05},
            {"type": "sphere", "center": np.array([0.6, 0.1, 0.3]), "radius": 0.04},
        ],
        "table_height": 0.3,
    },
    "shelf": {
        "description": "货架取放场景",
        "obstacles": [
            {"type": "box", "center": np.array([0.4, 0.0, 0.5]), "size": (0.3, 0.3, 0.02)},
        ],
        "shelf_heights": [0.3, 0.5, 0.7, 0.9],
    },
    "door": {
        "description": "开门场景",
        "obstacles": [
            {"type": "plane", "normal": np.array([0, 0, 1]), "distance": 0.0},
        ],
        "door_angle": 0.0,
    }
}


def create_scene(scene_name: str) -> Dict:
    """创建仿真场景"""
    return PRESET_SCENES.get(scene_name, PRESET_SCENES["tabletop"])


class ContactPhysicsModel:
    """
    物理-based 接触模型
    ==================

    精确建模机器人与环境交互的物理过程:
    - 库伦摩擦锥 (Coulomb Friction Cone)
    - 法向接触力 (Hertz/Spring-Damper)
    - 切向力与静/动摩擦转换
    - 接触柔顺性 (Compliance)
    - 力闭合检测 (Force Closure)

    适用于:
    - 抓取稳定性和 slip 检测
    - 力控装配 (螺栓拧紧/插销配合)
    - 触觉感知物理仿真
    """

    def __init__(
        self,
        static_friction_coeff: float = 0.4,
        dynamic_friction_coeff: float = 0.3,
        contact_stiffness: float = 10000.0,   # N/m
        contact_damping: float = 200.0,        # N·s/m
        restitution: float = 0.1,              # 恢复系数
        contact_area_radius: float = 0.01,     # m
    ):
        """
        Args:
            static_friction_coeff: 静摩擦系数 (μ_s)
            dynamic_friction_coeff: 动摩擦系数 (μ_d)
            contact_stiffness: 接触刚度 (N/m)
            contact_damping: 接触阻尼 (N·s/m)
            restitution: 法向恢复系数 (e ∈ [0,1])
            contact_area_radius: 接触区域半径 (m)
        """
        self.mu_s = static_friction_coeff
        self.mu_d = dynamic_friction_coeff
        self.k_n = contact_stiffness
        self.c_n = contact_damping
        self.restitution = restitution
        self.contact_radius = contact_area_radius

        # 接触状态
        self._penetration_history: List[float] = []
        self._tangential_slip_history: List[float] = []
        self._normal_force_history: List[float] = []

    def compute_normal_force(
        self,
        penetration: float,
        normal_velocity: float
    ) -> float:
        """
        计算法向接触力 (Spring-Damper 模型)

        F_n = k_n * δ + c_n * δ̇

        Args:
            penetration: 侵入深度 (m, 正值表示接触)
            normal_velocity: 法向相对速度 (m/s, 正值表示接近)

        Returns:
            F_n: 法向接触力 (N, 正值表示推力)
        """
        if penetration <= 0:
            return 0.0

        # 弹簧力 (Hertz 接触简化)
        spring_force = self.k_n * penetration

        # 阻尼力 (防止反弹震荡)
        damping_force = self.c_n * max(0, normal_velocity)

        F_n = spring_force + damping_force

        # 记录历史
        self._normal_force_history.append(F_n)
        if len(self._normal_force_history) > 100:
            self._normal_force_history.pop(0)

        return max(0.0, F_n)

    def compute_tangential_force(
        self,
        normal_force: float,
        tangential_velocity: np.ndarray,
        object_velocity: Optional[np.ndarray] = None,
        dt: float = 0.001
    ) -> np.ndarray:
        """
        计算切向摩擦力 (库伦摩擦 + 粘滞摩擦混合)

        Args:
            normal_force: 法向接触力 (N)
            tangential_velocity: 接触点切向速度 (m/s)
            object_velocity: 物体运动速度 (可选, 用于相对速度)
            dt: 时间步长

        Returns:
            F_t: 切向摩擦力向量 (N)
        """
        if normal_force <= 0 or np.linalg.norm(tangential_velocity) < 1e-9:
            return np.zeros(3)

        # 相对速度
        rel_vel = tangential_velocity
        if object_velocity is not None:
            rel_vel = tangential_velocity - object_velocity

        speed = np.linalg.norm(rel_vel)
        if speed < 1e-9:
            return np.zeros(3)

        direction = rel_vel / speed

        # 粘滞摩擦分量 (低速度时主导)
        viscous_coeff = self.mu_s * 0.5  # 粘滞摩擦系数
        F_viscous = viscous_coeff * normal_force * (1 - np.exp(-speed * 50)) * direction

        # 库伦摩擦分量 (高速度时主导)
        # 判断静/动摩擦临界速度
        slip_threshold = 0.05  # m/s

        if speed < slip_threshold:
            # 静摩擦区域: F_t ≤ μ_s * F_n
            max_static = self.mu_s * normal_force
            F_t_magnitude = np.linalg.norm(F_viscous)
            if F_t_magnitude > max_static:
                F_t_magnitude = max_static
                F_t = F_t_magnitude * direction
            else:
                F_t = F_viscous

            # 记录滑移量
            self._tangential_slip_history.append(0.0)
        else:
            # 动摩擦区域: F_t = μ_d * F_n
            F_t = self.mu_d * normal_force * direction
            self._tangential_slip_history.append(speed * dt)

        if len(self._tangential_slip_history) > 100:
            self._tangential_slip_history.pop(0)

        return F_t

    def compute_friction_cone_force(
        self,
        contact_force: np.ndarray,
        contact_normal: np.ndarray,
        tangential_velocity: np.ndarray,
        dt: float = 0.001
    ) -> np.ndarray:
        """
        完整摩擦锥力计算

        将接触力分解为法向和切向分量, 并在摩擦锥内计算平衡力

        Args:
            contact_force: 接触力向量 (N)
            contact_normal: 接触法向 (归一化)
            tangential_velocity: 切向速度 (m/s)

        Returns:
            F_total: 摩擦锥约束下的接触力 (N)
        """
        # 法向分量
        F_n_scalar = np.dot(contact_force, contact_normal)
        F_n_vec = F_n_scalar * contact_normal

        # 切向分量
        F_t_vec = contact_force - F_n_vec

        # 计算法向力
        normal_force = self.compute_normal_force(
            penetration=F_n_scalar / self.k_n if self.k_n > 0 else 0,
            normal_velocity=0.0
        )

        # 计算切向摩擦力
        F_t = self.compute_tangential_force(
            normal_force=normal_force,
            tangential_velocity=tangential_velocity,
            dt=dt
        )

        # 组合
        F_total = F_n_vec + F_t
        return F_total

    def detect_slip(
        self,
        normal_force: float,
        tangential_force_magnitude: float,
        object_mass: float = 0.1,
        gravity: float = 9.81
    ) -> Tuple[bool, float]:
        """
        滑移检测

        基于力闭合 (Force Closure) 和摩擦锥约束判断是否滑移

        Args:
            normal_force: 法向接触力 (N)
            tangential_force_magnitude: 切向力大小 (N)
            object_mass: 物体质量 (kg)
            gravity: 重力加速度 (m/s²)

        Returns:
            (is_slip, slip_probability): 是否滑移, 滑移概率
        """
        # 重力引起的切向力
        gravity_tangential = object_mass * gravity * self.mu_s

        # 摩擦锥极限
        max_friction = self.mu_s * normal_force

        # 安全裕度
        safety_margin = max_friction - tangential_force_magnitude
        slip_threshold = gravity_tangential * 0.1  # 10% 容差

        if safety_margin < 0:
            # 摩擦力不足,必然滑移
            return True, 1.0
        elif safety_margin < slip_threshold:
            # 接近临界,概率性滑移
            slip_prob = 1.0 - (safety_margin / slip_threshold)
            return np.random.rand() < slip_prob, slip_prob
        else:
            # 稳定抓取
            return False, 0.0

    def compute_grasp_quality(
        self,
        contact_points: List[np.ndarray],
        contact_normals: List[np.ndarray],
        object_center: np.ndarray,
        object_mass: float = 0.1
    ) -> Dict[str, float]:
        """
        计算抓取质量 (基于力闭合指标)

        Args:
            contact_points: 接触点列表 (每个3D坐标)
            contact_normals: 各接触点法向列表
            object_center: 物体几何中心
            object_mass: 物体质量 (kg)

        Returns:
            grasp_quality: 包含各项抓取质量指标的字典
        """
        n_contacts = len(contact_points)
        if n_contacts < 2:
            return {'overall': 0.0, 'force_closure': 0.0, 'stiffness': 0.0, 'min_margin': 0.0}

        # 重力向量
        gravity = np.array([0, 0, -object_mass * 9.81])

        # 力闭合检测 (简化版本)
        # 对于每个接触点, 计算在摩擦锥内的最大可平衡重力
        total_resistive_force = 0.0
        min_margin = float('inf')

        for i, (cp, cn) in enumerate(zip(contact_points, contact_normals)):
            # 计算接触点到物体中心的向量
            lever_arm = cp - object_center
            torque = np.cross(lever_arm, gravity)

            # 法向力方向与重力方向的夹角
            gravity_dir = gravity / (np.linalg.norm(gravity) + 1e-9)
            alignment = np.dot(-cn, gravity_dir)  # 法向朝向重力方向时为正

            if alignment > 0:
                # 该接触点可以提供法向支持
                max_friction = self.mu_s * alignment * 10.0  # 假设法向力上限 10N
                total_resistive_force += max_friction

                # 计算安全裕度
                margin = max_friction - abs(torque[i % 3]) if n_contacts > 2 else max_friction - np.linalg.norm(torque) * 0.1
                min_margin = min(min_margin, margin)

        # 力闭合率
        required_force = np.linalg.norm(gravity)
        force_closure = min(total_resistive_force / required_force, 1.0) if required_force > 0 else 0.0

        # 接触刚度指标 (接触数量越多,刚度越高)
        stiffness_score = min(n_contacts / 4.0, 1.0)

        # 综合质量
        overall = 0.5 * force_closure + 0.3 * stiffness_score + 0.2 * (min(min_margin, 1.0) if min_margin > 0 else 0.0)

        return {
            'overall': float(overall),
            'force_closure': float(force_closure),
            'stiffness': float(stiffness_score),
            'min_margin': float(min(min_margin, 1.0)) if min_margin > 0 else 0.0,
            'num_contacts': float(n_contacts)
        }

    def simulate_contact_event(
        self,
        initial_penetration: float = 0.002,
        impact_velocity: float = 0.1,
        object_mass: float = 0.5,
        duration: float = 0.1,
        dt: float = 0.001
    ) -> Dict[str, Any]:
        """
        模拟一次完整的接触事件

        包含:
        1. 初始碰撞 (法向冲击)
        2. 压陷阶段 (静力平衡)
        3. 滑移/稳定判断

        Args:
            initial_penetration: 初始侵入深度 (m)
            impact_velocity: 碰撞法向速度 (m/s)
            object_mass: 接触物体质量 (kg)
            duration: 模拟总时长 (s)
            dt: 仿真时间步长

        Returns:
            contact_event: 接触事件数据
        """
        n_steps = int(duration / dt)
        event_data = {
            'time': [],
            'penetration': [],
            'normal_force': [],
            'tangential_force_magnitude': [],
            'slip_detected': [],
            'slip_probability': []
        }

        penetration = initial_penetration
        velocity = impact_velocity

        for step in range(n_steps):
            t = step * dt

            # 法向接触力
            F_n = self.compute_normal_force(penetration, velocity)

            # 切向力 (假设切向速度从某值逐渐衰减)
            tangential_vel = np.array([0.0, velocity * 0.1 * np.sin(step * 0.1), 0.0])
            F_t_mag = np.linalg.norm(self.compute_tangential_force(F_n, tangential_vel, dt=dt))

            # 滑移检测
            is_slip, slip_prob = self.detect_slip(F_n, F_t_mag, object_mass)

            # 动力学更新
            # 加速度 = 力 / 质量 (仅考虑法向)
            acc = F_n / object_mass if object_mass > 0 else 0
            velocity = velocity - acc * dt  # 反弹导致减速
            penetration = max(0, penetration - abs(velocity) * dt)  # 压陷深度变化

            # 记录
            event_data['time'].append(t)
            event_data['penetration'].append(penetration)
            event_data['normal_force'].append(F_n)
            event_data['tangential_force_magnitude'].append(F_t_mag)
            event_data['slip_detected'].append(is_slip)
            event_data['slip_probability'].append(slip_prob)

            # 穿透太深或速度太低则停止
            if penetration < 1e-6 and abs(velocity) < 1e-4:
                break

        return event_data

    def get_contact_impedance(
        self,
        normal_force: float,
        frequency: float = 10.0
    ) -> Tuple[float, float]:
        """
        计算接触阻抗

        用于阻抗控制中的接触刚度/阻尼估计

        Args:
            normal_force: 稳态法向接触力 (N)
            frequency: 扰动频率 (Hz)

        Returns:
            (stiffness, damping): 等效刚度和阻尼
        """
        omega = 2 * np.pi * frequency

        # 动态接触刚度 (简化的 Hertz 模型)
        if normal_force > 0:
            dynamic_stiffness = self.k_n * (1 + np.sqrt(normal_force / self.k_n))
        else:
            dynamic_stiffness = self.k_n

        # 阻尼比
        zeta = self.c_n / (2 * np.sqrt(self.k_n * dynamic_stiffness))
        critical_damping = 2 * np.sqrt(self.k_n * dynamic_stiffness)
        equivalent_damping = zeta * critical_damping

        return float(dynamic_stiffness), float(equivalent_damping)


# AGV五级接触物理规格
AGV_CONTACT_PHYSICS_GRADES = {
    'S':  {'mu_s': 0.3,  'mu_d': 0.2,  'stiffness': 5000,  'damping': 100,  'contact_area_mm': 5},
    'M':  {'mu_s': 0.4,  'mu_d': 0.3,  'stiffness': 10000, 'damping': 200,  'contact_area_mm': 8},
    'L':  {'mu_s': 0.5,  'mu_d': 0.35, 'stiffness': 20000, 'damping': 400,  'contact_area_mm': 10},
    'XL': {'mu_s': 0.55, 'mu_d': 0.4,  'stiffness': 50000, 'damping': 800,  'contact_area_mm': 15},
    'XXL': {'mu_s': 0.6, 'mu_d': 0.45, 'stiffness': 100000,'damping': 1500, 'contact_area_mm': 20},
}


def get_contact_physics_spec(grade: str) -> ContactPhysicsModel:
    """获取AGV指定等级的接触物理模型"""
    spec = AGV_CONTACT_PHYSICS_GRADES.get(grade, AGV_CONTACT_PHYSICS_GRADES['M'])
    return ContactPhysicsModel(
        static_friction_coeff=spec['mu_s'],
        dynamic_friction_coeff=spec['mu_d'],
        contact_stiffness=spec['stiffness'],
        contact_damping=spec['damping'],
        contact_area_radius=spec['contact_area_mm'] / 1000.0
    )
