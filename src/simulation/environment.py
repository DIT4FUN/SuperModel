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
            # TODO: PyBullet 初始化
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
            # TODO: MuJoCo 初始化
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
