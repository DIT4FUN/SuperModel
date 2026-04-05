"""
PyBullet 物理引擎仿真模块
=========================

基于 Bullet Physics 的仿真环境，支持 AGV 差速驱动仿真。

功能:
- 刚体动力学仿真
- 接触力/碰撞检测
- 差速驱动 AGV 仿真
- 传感器模拟 (IMU, 里程计, 接触力)
- ROS2 话题桥接

支持:
- PyBullet 3.x
- URDF/SDF 模型格式
- AGV 五级规格 (S/M/L/XL/XXL)

使用示例:
    from simulation.pybullet_sim import PyBulletSimulator, PyBulletConfig

    # 创建仿真器
    sim = PyBulletSimulator(gui=True)
    agv = sim.load_agv_model(grade='M')

    # 仿真控制 (左/右轮速度)
    sim.set_motor_velocities([1.0, -1.0])

    for _ in range(1000):
        sim.step()
        state = sim.get_agv_state()
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum
import tempfile
import os

# PyBullet 可选导入 (graceful degradation)
try:
    import pybullet as p
    import pybullet_data
    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False
    p = None

pybullet_data = None if not HAS_PYBULLET else pybullet_data

# 从 AGV 场景模块导入物理配置
try:
    import sys
    _path = os.path.join(os.path.dirname(__file__), 'agv_scenarios.py')
    if os.path.exists(_path):
        from .agv_scenarios import AGVPhysicsConfig, AGVState
    else:
        from simulation.agv_scenarios import AGVPhysicsConfig, AGVState
except Exception:
    AGVPhysicsConfig = None
    AGVState = None

# 从AGV模型生成器导入
try:
    from .agv_model_generator import generate_agv_urdf_detailed, GRADE_CONFIGS, MOTOR_55_SPECS
except Exception:
    from simulation.agv_model_generator import generate_agv_urdf_detailed, GRADE_CONFIGS, MOTOR_55_SPECS


class PyBulletGUI(Enum):
    """PyBullet 可视化模式"""
    NONE = "none"           # 无可视化 (headless)
    GUI = "gui"             # 手动 GUI
    GUI_SHARED_MEMORY = "gui_shared_memory"  # 共享内存 GUI (多进程)
    DIRECT = "direct"       # 直接模式 (无窗口)
    SHARED_MEMORY_SERVER = "shared_memory_server"  # 服务器模式
    EGL = "egl"             # EGL 渲染 (无显示器)


@dataclass
class PyBulletConfig:
    """PyBullet 仿真配置"""
    # 仿真参数
    dt: float = 1.0 / 240.0       # 物理步长 (s) - PyBullet 推荐 1/240
    num_sub_steps: int = 1        # 每步子迭代
    gravity: float = -9.81        # 重力 (m/s^2)
    solver_iterations: int = 50   # 约束求解器迭代次数

    # 物理引擎
    physics_engine: str = "bullet"  # bullet / flex
    deterministic: bool = False    # 确定性仿真 (慢)
    real_time: bool = False        # 实时仿真

    # 可视化
    gui_mode: PyBulletGUI = PyBulletGUI.NONE  # 可视化模式
    width: int = 640              # 窗口宽度
    height: int = 480             # 窗口高度
    shadow: bool = True           # 阴影
    grid: bool = True             # 地面网格

    # 传感器噪声 (PyBullet 内置噪声参数)
    joint_position_noise: float = 0.0    # 关节位置噪声 (m)
    joint_velocity_noise: float = 0.0    # 关节速度噪声 (m/s)
    link_position_noise: float = 0.0      # 连杆位置噪声 (m)
    link_orientation_noise: float = 0.0   # 连杆方向噪声 (rad)
    sensor_noise: float = 0.0            # 通用传感器噪声

    # 接触力参数
    contact_stiffness: float = 10000.0   # 接触刚度
    contact_damping: float = 100.0       # 接触阻尼
    lateral_friction: float = 0.5        # 侧向摩擦
    spinning_friction: float = 0.01       # 旋转摩擦

    # AGV 等级
    grade: str = 'M'

    @classmethod
    def from_grade(cls, grade: str) -> 'PyBulletConfig':
        """从 AGV 等级创建配置"""
        configs = {
            'S': cls(dt=1.0/480.0, gravity=-9.81, grade='S'),
            'M': cls(dt=1.0/240.0, gravity=-9.81, grade='M'),
            'L': cls(dt=1.0/120.0, gravity=-9.81, grade='L'),
            'XL': cls(dt=1.0/120.0, gravity=-9.81, grade='XL'),
            'XXL': cls(dt=1.0/60.0, gravity=-9.81, grade='XXL'),
        }
        return configs.get(grade, cls())


# ============================================================================
# AGV URDF 模型生成
# ============================================================================

AGV_URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="agv_differential">

  <!-- AGV 主体 -->
  <link name="base_link">
    <inertial>
      <origin xyz="{com_x} {com_y} {com_z}" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="{ixx}" ixy="{ixy}" ixz="{ixz}" iyy="{iyy}" iyz="{iyz}" izz="{izz}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 {body_height/2}" rpy="0 0 0"/>
      <geometry>
        <box size="{body_length} {body_width} {body_height}"/>
      </geometry>
      <material name="grey">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 {body_height/2}" rpy="0 0 0"/>
      <geometry>
        <box size="{body_length} {body_width} {body_height}"/>
      </geometry>
    </collision>
  </link>

  <!-- 左轮 -->
  <link name="left_wheel">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{wheel_mass}"/>
      <inertia ixx="{wheel_ixx}" ixy="0" ixz="0" iyy="{wheel_ixx}" iyz="0" izz="{wheel_ixx}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="{wheel_roll} 0 0"/>
      <geometry>
        <cylinder length="{wheel_width}" radius="{wheel_radius}"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="{wheel_roll} 0 0"/>
      <geometry>
        <cylinder length="{wheel_width}" radius="{wheel_radius}"/>
      </geometry>
    </collision>
  </link>

  <!-- 右轮 -->
  <link name="right_wheel">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{wheel_mass}"/>
      <inertia ixx="{wheel_ixx}" ixy="0" ixz="0" iyy="{wheel_ixx}" iyz="0" izz="{wheel_ixx}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="{wheel_roll} 0 0"/>
      <geometry>
        <cylinder length="{wheel_width}" radius="{wheel_radius}"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="{wheel_roll} 0 0"/>
      <geometry>
        <cylinder length="{wheel_width}" radius="{wheel_radius}"/>
      </geometry>
    </collision>
  </link>

  <!-- 从动轮 (前/后) -->
  <link name="caster_front">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{caster_mass}"/>
      <inertia ixx="{caster_ixx}" ixy="0" ixz="0" iyy="{caster_ixx}" iyz="0" izz="{caster_ixx}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="{caster_radius}"/>
      </geometry>
      <material name="grey">
        <color rgba="0.3 0.3 0.3 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="{caster_radius}"/>
      </geometry>
    </collision>
  </link>

  <link name="caster_back">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{caster_mass}"/>
      <inertia ixx="{caster_ixx}" ixy="0" ixz="0" iyy="{caster_ixx}" iyz="0" izz="{caster_ixx}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="{caster_radius}"/>
      </geometry>
      <material name="grey">
        <color rgba="0.3 0.3 0.3 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="{caster_radius}"/>
      </geometry>
    </collision>
  </link>

  <!-- 关节 -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="{wheel_offset_x} -{track_width/2} -{body_height/2+wheel_radius-0.01}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics friction="{friction}" damping="0.1"/>
  </joint>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="{wheel_offset_x} {track_width/2} -{body_height/2+wheel_radius-0.01}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics friction="{friction}" damping="0.1"/>
  </joint>

  <joint name="caster_front_joint" type="continuous">
    <parent link="base_link"/>
    <child link="caster_front"/>
    <origin xyz="{caster_offset_x} 0 -{body_height/2+wheel_radius}" rpy="0 0 0"/>
  </joint>

  <joint name="caster_back_joint" type="continuous">
    <parent link="base_link"/>
    <child link="caster_back"/>
    <origin xyz="-{caster_offset_x} 0 -{body_height/2+wheel_radius}" rpy="0 0 0"/>
  </joint>

  <!-- IMU 传感器 (视觉传感器) -->
  <link name="imu_link">
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
    </inertial>
  </link>

  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/>
    <child link="imu_link"/>
    <origin xyz="0 0 {body_height/2}" rpy="0 0 0"/>
  </joint>

  <!-- 前视相机 (可选) -->
  <link name="camera_link">
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.05 0.08 0.03"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="{body_length/2} 0 {body_height/2-0.02}" rpy="0 0 0"/>
  </joint>

</robot>
"""


def generate_agv_urdf(grade: str = 'M', output_path: Optional[str] = None, wheel_config: str = '2轮') -> str:
    """生成 AGV URDF 文件

    使用新的详细AGV模型生成器，支持5.5寸轮毂电机参数。

    Args:
        grade: AGV 等级 (S/M/L/XL/XXL)
        output_path: 输出路径 (None = 临时文件)
        wheel_config: 轮子配置 ('2轮' 或 '4轮')

    Returns:
        URDF 文件路径
    """
    return generate_agv_urdf_detailed(grade=grade, output_path=output_path, wheel_config=wheel_config)


# ============================================================================
# PyBullet 仿真器
# ============================================================================

class PyBulletSimulator:
    """
    PyBullet 物理仿真器

    提供基于 Bullet Physics 的机器人仿真能力。

    支持:
    - 差速驱动 AGV 仿真
    - URDF 模型加载
    - 关节速度/力矩控制
    - 接触力获取
    - RGB-D 相机模拟
    - IMU 数据模拟

    示例:
        sim = PyBulletSimulator(gui=True, grade='M')
        agv_id = sim.load_agv_model()

        # 差速控制: 左轮速度, 右轮速度 (rad/s)
        sim.set_motor_velocities([5.0, 5.0])

        for _ in range(1000):
            sim.step()
            state = sim.get_agv_state()
            print(f"x={state[0]:.3f} y={state[1]:.3f} theta={state[2]:.3f}")
    """

    _instance_count = 0  # 跟踪实例数量

    def __init__(
        self,
        config: Optional[PyBulletConfig] = None,
        gui: bool = False,
        grade: str = 'M',
    ):
        """
        初始化 PyBullet 仿真器

        Args:
            config: 仿真配置
            gui: 是否启用 GUI
            grade: AGV 等级 (S/M/L/XL/XXL)
        """
        if not HAS_PYBULLET:
            raise ImportError(
                "PyBullet not installed. Install with: pip install pybullet"
            )

        self.config = config or PyBulletConfig.from_grade(grade)
        self.grade = grade

        # PyBullet 客户端 ID
        PyBulletSimulator._instance_count += 1
        self._client_id: Optional[int] = None

        # AGV 模型 ID
        self._agv_id: Optional[int] = None
        self._left_wheel_joint: Optional[int] = None
        self._right_wheel_joint: Optional[int] = None
        self._base_link_id: Optional[int] = None
        self._imu_link_id: Optional[int] = None
        self._camera_link_id: Optional[int] = None

        # 地面
        self._plane_id: Optional[int] = None

        # 里程计状态
        self._prev_base_pos: Optional[np.ndarray] = None
        self._prev_base_orn: Optional[np.ndarray] = None
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_theta = 0.0

        # 时间
        self._time = 0.0
        self._step_count = 0

        # 初始化 PyBullet
        self._init_pybullet(gui)

        # 加载数据路径 (用于加载 MJCF/URDF)
        self._data_path = pybullet_data.getDataPath()

        # URDF 临时文件清理列表
        self._temp_urdfs: List[str] = []

    def _init_pybullet(self, gui: bool):
        """初始化 PyBullet 客户端"""
        if gui:
            gui_mode = p.GUI
        else:
            gui_mode = p.DIRECT

        self._client_id = p.connect(gui_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # 设置仿真参数
        p.setGravity(0, 0, self.config.gravity, physicsClientId=self._client_id)
        p.setTimeStep(self.config.dt, physicsClientId=self._client_id)
        
        # 设置求解器迭代次数
        if hasattr(p, 'setPhysicsEngine'):
            p.setPhysicsEngine(
                numSolverIterations=self.config.solver_iterations,
                physicsClientId=self._client_id,
            )

        # 启用实时仿真 (可选)
        if self.config.real_time:
            p.setRealTimeSimulation(1, physicsClientId=self._client_id)

        # 配置接触力参数 (某些版本不支持)
        try:
            p.setPhysicsEngineParameter(
                contactStiffness=self.config.contact_stiffness,
                contactDamping=self.config.contact_damping,
                friction=self.config.lateral_friction,
                spinningFriction=self.config.spinning_friction,
                physicsClientId=self._client_id,
            )
        except TypeError:
            pass  # 某些版本不支持这些参数

    def load_agv_model(
        self,
        urdf_path: Optional[str] = None,
        base_position: Optional[Tuple[float, float, float]] = None,
        base_orientation: Tuple[float, float, float, float] = (0, 0, 0, 1),
        initial_pose: Optional[Tuple[float, float, float]] = None,  # 兼容旧API
    ) -> int:
        """
        加载 AGV 模型

        Args:
            urdf_path: URDF 文件路径 (None = 自动生成)
            base_position: 初始位置 (x, y, z)
            base_orientation: 初始方向 (四元数)

        Returns:
            AGV body ID
        """
        # 处理 initial_pose 兼容旧API
        if initial_pose is not None:
            if base_position is None:
                base_position = initial_pose
        if base_position is None:
            base_position = (0, 0, 0.1)

        if urdf_path is None:
            urdf_path = generate_agv_urdf(self.grade)
            self._temp_urdfs.append(urdf_path)

        self._agv_id = p.loadURDF(
            urdf_path,
            basePosition=base_position,
            baseOrientation=base_orientation,
            physicsClientId=self._client_id,
        )

        # 查找关节 ID
        num_joints = p.getNumJoints(self._agv_id, physicsClientId=self._client_id)
        self._joint_indices: Dict[str, int] = {}

        for i in range(num_joints):
            joint_info = p.getJointInfo(self._agv_id, i, physicsClientId=self._client_id)
            joint_name = joint_info[1].decode() if isinstance(joint_info[1], bytes) else joint_info[1]
            self._joint_indices[joint_name] = i

            # 识别驱动轮
            if 'left_wheel' in joint_name:
                self._left_wheel_joint = i
            elif 'right_wheel' in joint_name:
                self._right_wheel_joint = i

        # 识别 link ID
        self._base_link_id = -1  # base link 是 -1
        if 'imu_link' in self._joint_indices:
            self._imu_link_id = self._joint_indices['imu_link']
        if 'camera_link' in self._joint_indices:
            self._camera_link_id = self._joint_indices['camera_link']

        # 初始化里程计
        self._prev_base_pos = np.array(base_position)
        self._prev_base_orn = np.array(base_orientation)

        # 启用关节控制
        self._enable_motor_control()

        return self._agv_id

    def _enable_motor_control(self):
        """启用电机控制模式"""
        if self._left_wheel_joint is not None:
            p.setJointMotorControl2(
                self._agv_id,
                self._left_wheel_joint,
                p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0,
                physicsClientId=self._client_id,
            )
        if self._right_wheel_joint is not None:
            p.setJointMotorControl2(
                self._agv_id,
                self._right_wheel_joint,
                p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0,
                physicsClientId=self._client_id,
            )

    def load_plane(
        self,
        normal: Tuple[float, float, float] = (0, 0, 1),
        plane_size: float = 100.0,
    ) -> int:
        """加载地面

        Args:
            normal: 平面法向量
            plane_size: 平面大小 (m)

        Returns:
            地面 body ID
        """
        self._plane_id = p.loadURDF(
            "plane.urdf",
            basePosition=(0, 0, 0),
            baseOrientation=(0, 0, 0, 1),
            physicsClientId=self._client_id,
        )
        # 设置地面摩擦
        p.changeDynamics(
            self._plane_id,
            -1,
            lateralFriction=self.config.lateral_friction,
            spinningFriction=self.config.spinning_friction,
            physicsClientId=self._client_id,
        )
        return self._plane_id

    def load_box(
        self,
        half_extents: Tuple[float, float, float],
        position: Tuple[float, float, float],
        mass: float = 1.0,
        color: Tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1),
    ) -> int:
        """加载障碍物方块

        Args:
            half_extents: 半尺寸 (x, y, z)
            position: 位置
            mass: 质量 (kg)
            color: RGBA 颜色

        Returns:
            body ID
        """
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
            physicsClientId=self._client_id,
        )
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self._client_id,
        )
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            physicsClientId=self._client_id,
        )
        return body_id

    def load_cylinder(
        self,
        radius: float,
        height: float,
        position: Tuple[float, float, float],
        mass: float = 1.0,
        color: Tuple[float, float, float, float] = (0.2, 0.8, 0.2, 1),
    ) -> int:
        """加载圆柱障碍物

        Args:
            radius: 半径 (m)
            height: 高度 (m)
            position: 位置
            mass: 质量 (kg)
            color: RGBA 颜色

        Returns:
            body ID
        """
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=color,
            physicsClientId=self._client_id,
        )
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=radius,
            height=height,
            physicsClientId=self._client_id,
        )
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            physicsClientId=self._client_id,
        )
        return body_id

    def set_motor_velocities(self, velocities: List[float]):
        """
        设置电机速度 (差速驱动)

        Args:
            velocities: [左轮速度, 右轮速度] (rad/s)
        """
        if self._left_wheel_joint is not None and len(velocities) > 0:
            p.setJointMotorControl2(
                self._agv_id,
                self._left_wheel_joint,
                p.VELOCITY_CONTROL,
                targetVelocity=velocities[0],
                force=100,
                physicsClientId=self._client_id,
            )
        if self._right_wheel_joint is not None and len(velocities) > 1:
            p.setJointMotorControl2(
                self._agv_id,
                self._right_wheel_joint,
                p.VELOCITY_CONTROL,
                targetVelocity=velocities[1],
                force=100,
                physicsClientId=self._client_id,
            )

    def set_motor_torques(self, torques: List[float]):
        """
        设置电机力矩

        Args:
            torques: [左轮力矩, 右轮力矩] (Nm)
        """
        if self._left_wheel_joint is not None and len(torques) > 0:
            p.setJointMotorControl2(
                self._agv_id,
                self._left_wheel_joint,
                p.TORQUE_CONTROL,
                force=torques[0],
                physicsClientId=self._client_id,
            )
        if self._right_wheel_joint is not None and len(torques) > 1:
            p.setJointMotorControl2(
                self._agv_id,
                self._right_wheel_joint,
                p.TORQUE_CONTROL,
                force=torques[1],
                physicsClientId=self._client_id,
            )

    def step(self):
        """步进仿真"""
        p.stepSimulation(physicsClientId=self._client_id)
        self._step_count += 1
        self._time += self.config.dt

    def get_agv_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取 AGV 状态

        Returns:
            (position, orientation_euler, velocity) 元组
            - position: [x, y, z] (m)
            - orientation_euler: [roll, pitch, yaw] (rad)
            - velocity: [vx, vy, vz, wx, wy, wz]
        """
        if self._agv_id is None:
            raise RuntimeError("AGV model not loaded")

        state = p.getBasePositionAndOrientation(
            self._agv_id, physicsClientId=self._client_id
        )
        pos = np.array(state[0])
        orn = np.array(state[1])

        # 四元数转欧拉角
        euler = np.array(p.getEulerFromQuaternion(orn))

        # 线速度和角速度
        vel = p.getBaseVelocity(self._agv_id, physicsClientId=self._client_id)
        lin_vel = np.array(vel[0])
        ang_vel = np.array(vel[1])

        return pos, euler, np.concatenate([lin_vel, ang_vel])

    def get_odometry(self) -> Dict[str, float]:
        """
        获取里程计数据

        Returns:
            包含 x, y, theta, v, omega 的字典
        """
        pos, euler, vel = self.get_agv_state()
        vx, vy, vz = vel[:3]
        wx, wy, wz = vel[3:]

        # 更新里程计 (简化)
        if self._prev_base_pos is not None:
            dx = pos[0] - self._prev_base_pos[0]
            dy = pos[1] - self._prev_base_pos[1]
            self._odom_x += dx
            self._odom_y += dy

        # 角度处理
        dtheta = euler[2]
        self._odom_theta = dtheta

        self._prev_base_pos = pos.copy()
        self._prev_base_orn = euler.copy()

        # 计算线速度
        v = np.sqrt(vx**2 + vy**2)
        omega = wz

        return {
            'x': self._odom_x,
            'y': self._odom_y,
            'theta': self._odom_theta,
            'v': v,
            'omega': omega,
            'vx': vx,
            'vy': vy,
        }

    def get_imu_data(self) -> Dict[str, np.ndarray]:
        """
        获取 IMU 数据

        Returns:
            包含 accel, gyro, timestamp 的字典
        """
        if self._agv_id is None:
            raise RuntimeError("AGV model not loaded")

        # 获取基座线速度和角速度
        vel = p.getBaseVelocity(self._agv_id, physicsClientId=self._client_id)
        lin_vel = np.array(vel[0], dtype=np.float32)
        ang_vel = np.array(vel[1], dtype=np.float32)

        # 简化 IMU 模型 (假设机体坐标系)
        # 实际 IMU 应该考虑姿态，这里简化为速度的微分
        accel = np.array([0, 0, self.config.gravity]) + lin_vel * 0.0
        gyro = ang_vel.copy()

        # 添加噪声
        if self.config.sensor_noise > 0:
            noise_accel = np.random.randn(3) * self.config.sensor_noise
            noise_gyro = np.random.randn(3) * self.config.sensor_noise * 0.1
            accel = accel + noise_accel
            gyro = gyro + noise_gyro

        return {
            'accel': accel,
            'gyro': gyro,
            'timestamp': self._time,
        }

    def get_joint_states(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有关节状态

        Returns:
            {joint_name: {position, velocity, torque}}
        """
        if self._agv_id is None:
            raise RuntimeError("AGV model not loaded")

        joint_states = p.getJointStates(
            self._agv_id,
            list(self._joint_indices.values()),
            physicsClientId=self._client_id,
        )

        result = {}
        for name, joint_id in self._joint_indices.items():
            pos, vel, forces, torque = joint_states[joint_id]
            result[name] = {
                'position': float(pos),
                'velocity': float(vel),
                'force': float(forces[0]) if forces else 0.0,
                'torque': float(torque),
            }

        return result

    def get_wheel_states(self) -> Dict[str, Dict[str, float]]:
        """获取驱动轮状态"""
        states = self.get_joint_states()
        return {
            'left': states.get('left_wheel_joint', {}),
            'right': states.get('right_wheel_joint', {}),
        }

    def get_contact_forces(self) -> List[Dict[str, Any]]:
        """
        获取接触力

        Returns:
            接触信息列表
        """
        if self._agv_id is None:
            return []

        contacts = p.getContactPoints(
            bodyA=self._agv_id, physicsClientId=self._client_id
        )

        result = []
        for contact in contacts:
            result.append({
                'body_id': contact[2],
                'link_id': contact[3],
                'position': np.array(contact[5]),
                'normal': np.array(contact[7]),
                'force': float(contact[9]),
                'distance': float(contact[8]),
            })

        return result

    def get_rgbd_image(
        self,
        width: int = 640,
        height: int = 480,
        view_matrix: Optional[Any] = None,
        proj_matrix: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        获取 RGB-D 图像

        Args:
            width: 图像宽度
            height: 图像高度
            view_matrix: 视图矩阵 (None = 默认相机)
            proj_matrix: 投影矩阵 (None = 默认)

        Returns:
            {rgb, depth, timestamp}
        """
        if self._agv_id is None:
            raise RuntimeError("AGV model not loaded")

        # 计算相机位姿
        if self._camera_link_id is not None:
            cam_pos, cam_orn = p.getLinkState(
                self._agv_id,
                self._camera_link_id,
                physicsClientId=self._client_id,
            )[:2]
        else:
            cam_pos, cam_orn = p.getBasePositionAndOrientation(
                self._agv_id, physicsClientId=self._client_id
            )
            cam_pos = list(cam_pos)
            cam_pos[2] += 0.2  # 抬高相机

        # 默认视图矩阵
        if view_matrix is None:
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=cam_pos,
                distance=1.0,
                yaw=0,
                pitch=-45,
                roll=0,
                upAxisIndex=2,
            )

        # 默认投影矩阵
        if proj_matrix is None:
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=width / height,
                nearVal=0.1, farVal=100.0
            )

        img = p.getCameraImage(
            width, height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self._client_id,
        )

        rgb = np.array(img[2], dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        depth = np.array(img[3], dtype=np.float32).reshape(height, width)

        return {
            'rgb': rgb,
            'depth': depth,
            'timestamp': self._time,
        }

    def reset(
        self,
        base_position: Tuple[float, float, float] = (0, 0, 0.1),
        base_orientation: Tuple[float, float, float, float] = (0, 0, 0, 1),
    ):
        """重置仿真"""
        if self._agv_id is not None:
            p.resetBasePositionAndOrientation(
                self._agv_id,
                base_position,
                base_orientation,
                physicsClientId=self._client_id,
            )
            p.resetBaseVelocity(
                self._agv_id, [0, 0, 0], [0, 0, 0],
                physicsClientId=self._client_id,
            )

        # 重置里程计
        self._prev_base_pos = np.array(base_position)
        self._prev_base_orn = np.array(base_orientation)
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_theta = 0.0
        self._time = 0.0
        self._step_count = 0

        # 重置关节速度
        self._enable_motor_control()

    @property
    def time(self) -> float:
        """仿真时间 (s)"""
        return self._time

    @property
    def step_count(self) -> int:
        """仿真步数"""
        return self._step_count

    @property
    def client_id(self) -> Optional[int]:
        """PyBullet 客户端 ID"""
        return self._client_id

    def close(self):
        """关闭仿真"""
        if self._client_id is not None:
            p.disconnect(physicsClientId=self._client_id)
            self._client_id = None

        # 清理临时 URDF 文件
        for urdf in self._temp_urdfs:
            try:
                os.remove(urdf)
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ============================================================================
# PyBullet 仿真工厂函数
# ============================================================================

def create_pybullet_simulator(
    gui: bool = False,
    grade: str = 'M',
    load_agv: bool = True,
    load_plane: bool = True,
    config: Optional[PyBulletConfig] = None,
) -> PyBulletSimulator:
    """
    创建 PyBullet 仿真器

    Args:
        gui: 是否启用 GUI
        grade: AGV 等级
        load_agv: 是否自动加载 AGV 模型
        load_plane: 是否自动加载地面
        config: 仿真配置

    Returns:
        PyBulletSimulator 实例
    """
    sim = PyBulletSimulator(config=config, gui=gui, grade=grade)

    if load_plane:
        sim.load_plane()

    if load_agv:
        sim.load_agv_model()

    return sim


def get_pybullet_spec(grade: str) -> PyBulletConfig:
    """获取 AGV 五级 PyBullet 规格"""
    return PyBulletConfig.from_grade(grade)
