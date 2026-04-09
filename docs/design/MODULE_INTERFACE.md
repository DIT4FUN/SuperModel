# SuperModel 模块接口设计

> 📑 **目录导航** (点击章节号跳转)
> - [1. 概述](#1-概述) · [2. 感知层接口](#2-感知层接口-perception) · [3. 融合层接口](#3-融合层接口-fusion) · [4. 认知层接口](#4-认知层接口-cognition) · [5. 执行层接口](#5-执行层接口-execution) · [6. 仿真层接口](#6-仿真层接口-simulation) · [7. 数据流图](#7-数据流图) · [8. 编码器模块接口](#8-编码器模块接口) · [9. 仿真模块接口](#9-仿真模块接口-simulation) · [10. ROS2 集成接口](#10-ros2-集成接口) · [11. 数据格式规范](#11-数据格式规范) · [12. 安全控制器接口](#12-安全控制器接口-safety-controller) · [13. 错误处理规范](#13-错误处理规范) · [14. 触觉/力觉/IMU传感器详细接口](#14-触觉力觉imu传感器详细接口) · [15. 跨模态融合网络详细接口](#15-跨模态融合网络详细接口) · [16. AGV五级规格对照](#16-agv五级规格对照-增强) · [17. 执行控制系统 AGV 五级规格汇总](#17-执行控制系统-agv-五级规格汇总) · [18. 传感器 AGV 五级规格速查](#18-传感器-agv-五级规格速查) · [19. 模型预测控制 (MPC)](#19-模型预测控制-mpc) · [20. Gymnasium RL 环境](#20-gymnasium-rl-环境) · [21. 完整系统集成示例](#21-完整系统集成示例) · [22. 虚拟传感器接口](#22-虚拟传感器接口-virtual-sensors) · [23. ROS2 接口模块](#23-ros2-接口模块-ros2_interface) · [24. 控制模块接口](#24-控制模块接口-control) · [25. 触觉传感器模块接口](#25-触觉传感器模块接口--tactilearray--virtualtactilesensor) · [26. 力觉传感器模块接口](#26-力觉传感器模块接口--forcetorquesensor--virtualforcesensor) · [27. IMU传感器模块接口](#27-imu传感器模块接口--imusensor--virtualimusensor--poseestimator) · [28. 多智能体协调控制](#28-多智能体协调控制-multi-agent-coordination) · [29. 完整传感器-控制集成流水线](#29-完整传感器-控制集成流水线) · [30. 避障模块接口](#30-避障模块接口-obstacle-avoidance) · [31. 仿真-避障集成](#31-仿真-避障集成) · [32. AGV等级与避障能力对照](#32-agv等级与避障能力对照) · [33. 触觉-控制集成实战](#33-触觉-控制集成实战-tactile-control-integration) · [34. 力觉-控制集成实战](#34-力觉-控制集成实战-force-control-integration) · [35. IMU-控制集成实战](#35-imu-控制集成实战-imu-control-integration) · [36. 多传感器-控制联合集成](#36-多传感器-控制联合集成)

---

## Table of Contents

| # | Section | Lines | Description |
|---|---------|-------|-------------|
| 1 | [概述](#1-概述) | 1-8 | 项目概述与文档结构 |
| 2 | [感知层接口](#2-感知层接口-perception) | 9-182 | 视觉/听觉传感器接口 |
| 3 | [融合层接口](#3-融合层接口-fusion) | 183-249 | 跨模态融合网络接口 |
| 4 | [认知层接口](#4-认知层接口-cognition) | 250-277 | 自主学习/世界模型接口 |
| 5 | [执行层接口](#5-执行层接口-execution) | 278-669 | 运动控制/轨迹规划/技能库 |
| 6 | [仿真层接口](#6-仿真层接口-simulation) | 670-695 | 仿真环境接口 |
| 7 | [数据流图](#7-数据流图) | 696-729 | 系统数据流架构 |
| 8 | [编码器模块接口](#8-编码器模块接口) | 730-774 | 神经网络编码器接口 |
| 9 | [仿真模块接口](#9-仿真模块接口-simulation) | 775-804 | RobotSimulator/Gym接口 |
| 10 | [ROS2 集成接口](#10-ros2-集成接口) | 805-862 | ROS2 Humble接口 |
| 11 | [数据格式规范](#11-数据格式规范) | 863-895 | 统一数据格式约定 |
| 12 | [安全控制器接口](#12-安全控制器接口-safety-controller) | 896-1053 | SafetyController接口 |
| 13 | [错误处理规范](#13-错误处理规范) | 1054-1081 | 异常与错误码定义 |
| 14 | [触觉/力觉/IMU详细接口](#14-触觉力觉imu传感器详细接口) | 1082-1239 | 三大传感器详细API |
| 15 | [跨模态融合网络详细接口](#15-跨模态融合网络详细接口) | 1240-1383 | CrossModalFusion详解 |
| 16 | [AGV五级规格对照(增强)](#16-agv五级规格对照-增强) | 1384-1396 | 五级规格增强版 |
| 17 | [执行控制系统AGV五级规格](#17-执行控制系统-agv-五级规格汇总) | 1397-1512 | 控制子系统五级规格 |
| 18 | [传感器AGV五级规格速查](#18-传感器-agv-五级规格速查) | 1513-1659 | 传感器五级规格速查表 |
| 19 | [模型预测控制(MPC)](#19-模型预测控制-mpc) | 1660-1751 | MPC控制器接口 |
| 20 | [Gymnasium RL环境](#20-gymnasium-rl-环境) | 1752-1845 | Gym环境接口 |
| 21 | [完整系统集成示例](#21-完整系统集成示例) | 1846-1944 | 端到端集成示例 |
| 22 | [虚拟传感器接口](#22-虚拟传感器接口-virtual-sensors) | 1945-2037 | 虚拟传感器接口 |
| 23 | [ROS2接口模块](#23-ros2-接口模块-ros2_interface) | 2038-2576 | ROS2接口详解 |
| 24 | [控制模块接口](#24-控制模块接口-control) | 2577-3113 | 控制子系统完整接口 |
| 25 | [触觉传感器模块接口](#25-触觉传感器模块接口--tactilearray--virtualtactilesensor) | 3114-3205 | TactileArray API |
| 26 | [力觉传感器模块接口](#26-力觉传感器模块接口--forcetorquesensor--virtualforcesensor) | 3206-3272 | ForceTorqueSensor API |
| 27 | [IMU传感器模块接口](#27-imu传感器模块接口--imusensor--virtualimusensor--poseestimator) | 3273-3405 | IMUSensor API |
| 28 | [多智能体协调控制](#28-多智能体协调控制-multi-agent-coordination) | 3406-3559 | MultiAgentCoordinator |
| 29 | [完整传感器-控制集成流水线](#29-完整传感器-控制集成流水线) | 3560-3971 | 端到端流水线 |
| 30 | [避障模块接口](#30-避障模块接口-obstacle-avoidance) | 3972-4118 | 避障控制器接口 |
| 31 | [仿真-避障集成](#31-仿真-避障集成) | 4119-4155 | 仿真与避障联动 |
| 32 | [AGV等级与避障能力对照](#32-agv等级与避障能力对照) | 4156-4174 | 五级避障规格 |
| 33 | [触觉-控制集成实战](#33-触觉-控制集成实战-tactile-control-integration) | 4175-4370 | TactileServoController |
| 34 | [力觉-控制集成实战](#34-力觉-控制集成实战-force-control-integration) | 4371-4737 | ForceController |
| 35 | [IMU-控制集成实战](#35-imu-控制集成实战-imu-control-integration) | 4738-5030 | AttitudeStabilizer |
| 36 | [多传感器-控制联合集成](#36-多传感器-控制联合集成) | 5031-5240 | 联合集成实战 |

---

## 1. 概述

本文档详细描述 SuperModel 超模态机器人具身智能大脑的核心模块接口设计，涵盖感知层、融合层、认知层和执行层的主要接口定义、数据格式和调用约定。

---

## 2. 感知层接口 (Perception)

### 2.1 视觉传感器 — BinocularCamera

```python
class BinocularCamera:
    def open(self) -> bool
    def close(self)
    def capture(self) -> StereoFrame          # 返回双目帧
    def get_depth_map(self, frame) -> np.ndarray  # HxW 深度图
    def get_point_cloud(self, frame) -> np.ndarray  # Nx3 点云
    def set_exposure(self, value: float)
    def set_gain(self, value: float)
    def __enter__ / __exit__  # 上下文管理
```

**StereoFrame 数据结构:**
```python
@dataclass
class StereoFrame:
    left_image: np.ndarray     # HxWx3 uint8
    right_image: np.ndarray    # HxWx3 uint8
    timestamp: float
    frame_id: int
    left_cam_info: CameraIntrinsics
    right_cam_info: CameraIntrinsics
    stereo_extrinsics: Optional[StereoExtrinsics]
```

**使用示例:**
```python
with BinocularCamera(resolution=(1280, 720), fps=30) as cam:
    frame = cam.capture()
    depth = cam.get_depth_map(frame)
    points = cam.get_point_cloud(frame)
```

---

### 2.2 听觉传感器 — BinauralMic

```python
class BinauralMic:
    def open(self) -> bool
    def close()
    def capture(self, duration: float = 0.1) -> AudioFrame  # 采集音频帧
    def get_sound_direction(self, frame) -> Optional[float]  # 返回方位角 (rad)
    def localize_sources(self, frame) -> List[SoundSource]  # 声源定位
    def set_gain(gain: float)
    def enable_beamforming(self, enabled: bool)
```

**AudioFrame 数据结构:**
```python
@dataclass
class AudioFrame:
    left_channel: np.ndarray   # N samples
    right_channel: np.ndarray  # N samples
    sample_rate: int           # Hz
    timestamp: float
    frame_id: int
```

---

### 2.3 触觉传感器 — TactileArray

```python
class TactileArray:
    def open(self) -> bool
    def close()
    def capture(self) -> TactileFrame
    def detect_contacts(self, frame) -> List[TactileContact]  # 接触检测
    def get_slip_signal(self, frame) -> np.ndarray            # 滑移检测
    def calibrate(zero_pressure, known_weights)
    def __enter__ / __exit__
```

**TactileFrame / TactileContact:**
```python
@dataclass
class TactileFrame:
    pressure_map: np.ndarray       # HxW float32 (0-1)
    temperature_map: Optional[np.ndarray]  # HxW float32 (°C)
    proximity: Optional[np.ndarray]        # HxW float32 (m)
    slip_signal: Optional[np.ndarray]      # HxW float32
    timestamp: float
    frame_id: int

@dataclass
class TactileContact:
    center: Tuple[int, int]        # 接触中心
    area: int                      # 接触面积
    peak_pressure: float
    mean_pressure: float
    centroid: Tuple[float, float]
    contact_force: float           # N
    slip_probability: float
    temperature: Optional[float]
```

---

### 2.4 力觉传感器 — ForceTorqueSensor

```python
class ForceTorqueSensor:
    def open(self) -> bool
    def close()
    def capture(self) -> Wrench                    # 采集力旋量
    def get_wrench(self) -> Optional[Wrench]       # 获取最新数据
    def detect_contact(self, wrench) -> ContactState  # 接触检测
    def estimate_payload(self, wrench) -> float    # 负载估计
    def set_tool_center(self, mass: float, com: np.ndarray)  # TCP设置
    def calibrate_bias(self, num_samples: int = 100)
    def __enter__ / __exit__
```

**Wrench 数据结构:**
```python
@dataclass
class Wrench:
    force: np.ndarray   # 3, [Fx, Fy, Fz] N
    torque: np.ndarray  # 3, [Tx, Ty, Tz] N·m
    timestamp: float
    frame_id: int

    def magnitude(self) -> float          # 力向量模长
    def torque_magnitude(self) -> float    # 力矩模长
    def to_vector(self) -> np.ndarray      # 6维向量
    def transform(R, t) -> Wrench          # 坐标变换
```

---

### 2.5 IMU传感器 — IMUSensor

```python
class IMUSensor:
    def open(self) -> bool
    def close()
    def capture(self) -> IMUFrame
    def self_test(self) -> bool             # 自检
    def calibrate_gyro_bias(self, num_samples=500)
    def calibrate_accel(self, known_orientation="level")
    def __enter__ / __exit__
```

**IMUFrame / Pose:**
```python
@dataclass
class IMUFrame:
    accel: np.ndarray          # 3, m/s^2
    gyro: np.ndarray           # 3, rad/s
    mag: Optional[np.ndarray]  # 3, uT
    temperature: float
    timestamp: float
    frame_id: int

class PoseEstimator:
    def update(self, accel, gyro, mag=None, dt=None) -> Pose
    def get_pose(self) -> Pose
    def get_euler(self) -> np.ndarray  # [roll, pitch, yaw]

@dataclass
class Pose:
    position: np.ndarray       # 3, m
    orientation: np.ndarray   # 4, 四元数 [qw, qx, qy, qz]
    def to_euler(self) -> np.ndarray
    def to_matrix(self) -> np.ndarray  # 4x4
```

---

## 3. 融合层接口 (Fusion)

### 3.1 跨模态融合网络 — CrossModalFusion

```python
class CrossModalFusion:
    def __init__(self, config: FusionConfig)
    
    def forward(
        self,
        multimodal_input: MultimodalInput,
        return_attention: bool = False
    ) -> UnifiedRepresentation
    
    def encode_modality(
        self,
        modality: str,
        data: np.ndarray
    ) -> np.ndarray  # 返回特征向量
    
    def compute_cross_attention(
        self,
        query_modality: str,
        key_modality: str
    ) -> np.ndarray
    
    def get_fusion_weights(self) -> Dict[str, float]
```

**输入输出数据结构:**
```python
@dataclass
class MultimodalInput:
    vision: Optional[np.ndarray]    # BxCxHxW 或 BxNxd
    audio: Optional[np.ndarray]    # BxTxD
    tactile: Optional[np.ndarray]  # BxHxW
    force: Optional[np.ndarray]    # Bx6
    imu: Optional[np.ndarray]      # Bx6
    language: Optional[np.ndarray] # BxLxD

@dataclass
class UnifiedRepresentation:
    features: torch.Tensor         # BxD 统一特征
    modality_weights: Dict[str, float]
    attention_maps: Optional[Dict[str, torch.Tensor]]
    timestamp: float
```

**FusionConfig:**
```python
@dataclass
class FusionConfig:
    vision_dim: int = 512
    audio_dim: int = 128
    tactile_dim: int = 64
    force_dim: int = 32
    imu_dim: int = 32
    hidden_dim: int = 256
    output_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    fusion_strategy: FusionStrategy = FusionStrategy.MIDDLE
    dropout: float = 0.1
```

---

## 4. 认知层接口 (Cognition)

### 4.1 自主学习框架 — SelfSupervisedLearner

```python
class SelfSupervisedLearner:
    def __init__(self, fusion_model: CrossModalFusion, config: dict)
    
    def update(
        self,
        multimodal_input: MultimodalInput,
        reward_signal: float,
        done: bool = False
    ) -> Dict[str, float]  # 返回损失字典
    
    def select_action(
        self,
        state: UnifiedRepresentation,
        epsilon: float = 0.0
    ) -> np.ndarray
    
    def save(self, path: str)
    def load(self, path: str)
    def get_statistics(self) -> Dict[str, Any]
```

---

## 5. 执行层接口 (Execution)

### 5.1 运动控制 — MotionController

```python
class MotionController:
    def __init__(self, num_joints: int, control_rate: float = 100.0, ...)
    
    def set_joint_limits(self, lower: np.ndarray, upper: np.ndarray)
    def set_pid_gains(self, kp: np.ndarray, ki: np.ndarray, kd: np.ndarray)
    def set_torque_callback(self, callback: Callable)
    
    def update_joint_state(self, joint_state: JointState)
    def compute_joint_torque(
        self,
        target_position: np.ndarray,
        target_velocity: Optional[np.ndarray] = None
    ) -> np.ndarray
    
    def compute_cartesian_velocity(
        self,
        target_twist: TwistCommand,
        jacobian: np.ndarray
    ) -> np.ndarray
    
    def interpolate_trajectory(
        self,
        trajectory: JointTrajectory,
        current_time: float
    ) -> Tuple[np.ndarray, np.ndarray]
    
    def step(self, target: np.ndarray, mode: ControlMode) -> np.ndarray
```

**关键数据结构:**
```python
@dataclass
class JointState:
    position: np.ndarray  # 关节位置
    velocity: np.ndarray  # 关节速度
    torque: np.ndarray    # 关节力矩
    timestamp: float

@dataclass
class TwistCommand:
    linear: np.ndarray     # 3, m/s
    angular: np.ndarray    # 3, rad/s
    frame_id: str = "base_link"

class ControlMode(Enum):
    JOINT_POSITION
    JOINT_VELOCITY
    JOINT_TORQUE
    CARTESIAN_VELOCITY
    CARTESIAN_POSITION
```

### 5.2 轨迹规划 — TrajectoryGenerator / RRTPlanner

```python
class TrajectoryGenerator:
    def __init__(self, num_joints: int, config: Optional[TrajectoryConfig] = None)

    def generate_quintic_polynomial(
        self,
        start: np.ndarray,
        end: np.ndarray,
        duration: float,
        start_vel: np.ndarray = None,
        end_vel: np.ndarray = None,
        start_acc: np.ndarray = None,
        end_acc: np.ndarray = None
    ) -> List[JointWaypoint]

    def generate_trapezoidal(
        self,
        start: np.ndarray,
        end: np.ndarray,
        max_velocity: np.ndarray,
        max_acceleration: np.ndarray
    ) -> Tuple[List[JointWaypoint], float]

    def resample_trajectory(
        self,
        waypoints: List[JointWaypoint],
        new_dt: float
    ) -> List[JointWaypoint]

class RRTPlanner:
    def __init__(
        self,
        space_dim: int,
        bounds: List[Tuple[float, float]],
        max_iterations: int = 1000,
        step_size: float = 0.1,
        goal_bias: float = 0.05,
        search_radius: float = 0.5
    )

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacle_check: Callable[[np.ndarray], bool],
        algorithm: PlanningAlgorithm = PlanningAlgorithm.RRT_STAR
    ) -> Tuple[Optional[List[np.ndarray]], float]

class ScurveGenerator:
    def __init__(self, max_velocity: float, max_acceleration: float, max_jerk: float)
    def plan(self, start_pos: float, end_pos: float, start_vel=0.0, end_vel=0.0) -> List[Dict]

@dataclass
class JointWaypoint:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    time_from_start: float

@dataclass
class CartesianWaypoint:
    position: np.ndarray      # 3
    orientation: np.ndarray  # 4
    linear_velocity: np.ndarray   # 3
    angular_velocity: np.ndarray  # 3
    time_from_start: float

@dataclass
class TrajectoryConfig:
    max_velocity: np.ndarray
    max_acceleration: np.ndarray
    max_jerk: Optional[np.ndarray] = None
    dt: float = 0.01
    tolerance: float = 1e-6

class PlanningAlgorithm(Enum):
    RRT = "rrt"
    RRT_STAR = "rrt_star"
    PRM = "prm"
    RRT_CONNECT = "rrt_connect"
    INF_PLANNER = "informed_rrt_star"
```

**使用示例:**
```python
from control.trajectory import TrajectoryGenerator, RRTPlanner, PlanningAlgorithm

# 关节空间轨迹生成
gen = TrajectoryGenerator(num_joints=6)
waypoints = gen.generate_quintic_polynomial(
    start=np.zeros(6),
    end=np.array([0.5, 0.3, -0.2, 0.0, 0.0, 0.0]),
    duration=2.0
)

# 笛卡尔空间RRT规划
def no_collision(pos):
    # 自定义碰撞检测
    return False

planner = RRTPlanner(
    space_dim=3,
    bounds=[(-1, 1), (-1, 1), (0, 2)],
    max_iterations=500
)
path, cost = planner.plan(
    start=np.array([0.0, 0.0, 0.5]),
    goal=np.array([0.8, 0.3, 0.8]),
    obstacle_check=no_collision,
    algorithm=PlanningAlgorithm.RRT_STAR
)
```

---

### 5.2 阻抗控制 — ImpedanceController

```python
class ImpedanceController:
    def __init__(self, impedance_params: ImpedanceParams, control_rate: float = 100.0)
    
    def set_impedance_params(self, params: ImpedanceParams)
    def compute_torque(
        self,
        desired_position: np.ndarray,
        desired_velocity: np.ndarray,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        external_wrench: np.ndarray,
        jacobian: np.ndarray
    ) -> np.ndarray
    
    def compute_cartesian_force(
        self,
        desired_pose: np.ndarray,
        desired_velocity: np.ndarray,
        external_wrench: np.ndarray
    ) -> np.ndarray

class AdmittanceController:
    def update(self, external_force: float, desired_position: float, dt=None) -> float
    def reset(self)

class ForceImpedanceController:
    def compute_torque(...) -> np.ndarray  # 力位混合控制

class CollaborativeController:
    def check_safety(external_force, velocity) -> Tuple[bool, str]
    def get_reaction_torque(external_force, jacobian) -> np.ndarray
```

---

### 5.3 技能库 — SkillLibrary / Skill

```python
class Skill:
    def __init__(self, config: SkillConfig)
    def can_execute(self, context: Dict) -> bool
    def execute(self, context: Dict) -> SkillResult
    def cancel()
    def check_timeout() -> bool

class SkillLibrary:
    def __init__(self)
    def create_skill(self, name: str, config: Dict) -> Optional[Skill]
    def register_skill(self, skill: Skill)
    def get_skill(self, name: str) -> Optional[Skill]
    def list_skills(self) -> List[str]

class SkillRegistry:
    @classmethod
    def get_instance(cls) -> SkillRegistry

@dataclass
class SkillResult:
    success: bool
    status: SkillStatus
    output: Optional[Dict]
    error_message: Optional[str]
    duration: float
```

---

### 5.4 任务规划 — TaskPlanner

```python
class TaskPlanner:
    def __init__(self)
    def add_task(self, task: Task) -> str
    def remove_task(self, task_id: str)
    def update_task(self, task_id: str, updates: Dict)
    def get_next_action(self, task_id: str, state: Dict) -> Optional[Dict]
    def replan(self, task_id: str, new_state: Dict)
    def pause_task(self, task_id: str)
    def resume_task(self, task_id: str)
    def get_task_status(self, task_id: str) -> TaskStatus

@dataclass
class Task:
    id: str
    name: str
    description: str
    priority: TaskPriority
    parameters: Dict
    subtasks: List[Task]
    status: TaskStatus
    start_time: Optional[float]
    end_time: Optional[float]
    result: Optional[Dict]
```

### 5.5 AGV运动控制 — AGVMotionController

```python
class AGVMotionController:
    """
    AGV专用运动控制器
    
    支持:
    - 差速驱动 (Differential)
    - 全向移动 (Omnidirectional)
    - 麦克纳姆轮 (Mecanum)
    - 轨迹跟踪与PID纠正
    """
    
    def __init__(self, spec: AGVSpec)
    
    def forward_kinematics(self, wheel_velocities: np.ndarray) -> AGVTwist
    def inverse_kinematics(self, twist: AGVTwist) -> np.ndarray
    def update_pose(self, new_pose: AGVPose)
    def update_twist(self, new_twist: AGVTwist)
    def compute_wheel_commands(self, target_pose: AGVPose, dt: float) -> np.ndarray
    def apply_safety_limits(self, wheel_commands: np.ndarray) -> np.ndarray
    
    @property
    def pose(self) -> AGVPose
    @property
    def twist(self) -> AGVTwist
```

**关键数据结构:**
```python
class DriveType(Enum):
    DIFFERENTIAL = "differential"
    OMNIDIRECTIONAL = "omnidirectional"
    MECANUM = "mecanum"
    ACKERMANN = "ackermann"

class AGVGrade(Enum):
    S = "S"    # 教育/实验
    M = "M"    # 标准助手
    L = "L"    # 专业工业
    XL = "XL"  # 高性能
    XXL = "XXL"  # 旗舰全功能

@dataclass
class AGVSpec:
    grade: AGVGrade
    max_linear_speed: float      # m/s
    max_angular_speed: float     # rad/s
    max_linear_accel: float     # m/s^2
    max_angular_accel: float    # rad/s^2
    wheelbase: float            # m
    track_width: float          # m
    wheel_radius: float         # m
    drive_type: DriveType
    control_frequency: float     # Hz

@dataclass
class AGVPose:
    x: float      # 世界X (m)
    y: float      # 世界Y (m)
    theta: float  # 朝向角 (rad)

@dataclass
class AGVTwist:
    vx: float     # X线速度 (m/s)
    vy: float     # Y线速度 (m/s)
    omega: float  # 角速度 (rad/s)
```

**AGVSpec 工厂方法:**
```python
AGVSpec.from_grade(AGVGrade.M)  # 获取M级标准规格
get_agv_spec("L")               # 获取L级标准规格
```

**使用示例:**
```python
from control.agv import AGVMotionController, AGVSpec, AGVGrade, AGVPose

# 创建M级AGV控制器
spec = AGVSpec.from_grade(AGVGrade.M)
agv = AGVMotionController(spec)

# 跟踪目标位姿
target = AGVPose(x=1.0, y=0.5, theta=0.0)
wheel_cmds = agv.compute_wheel_commands(target, dt=0.01)

# 应用安全限制
safe_cmds = agv.apply_safety_limits(wheel_cmds)
```

**虚拟传感器接口:**
```python
class VirtualTactileSensor:
    """虚拟触觉传感器 (仿真)"""
    def open(self) -> bool
    def close()
    def simulate_contact(contact_pos, contact_radius, contact_force, noise_level) -> TactileFrame
    def simulate_sliding(direction, speed, duration_frames) -> List[TactileFrame]

class VirtualForceSensor:
    """虚拟力觉传感器 (仿真)"""
    def open(self) -> bool
    def close()
    def simulate_contact(force, torque, add_noise) -> Wrench
    def simulate_payload(mass, com_offset, gravity) -> Wrench
    def simulate_collision(direction, peak_force, duration_ms) -> List[Wrench]

class VirtualIMUSensor:
    """虚拟IMU传感器 (仿真)"""
    def open(self) -> bool
    def close()
    def simulate_static(orientation) -> IMUFrame
    def simulate_motion(linear_accel, angular_vel, dt) -> IMUFrame
    def simulate_trajectory(trajectory_type, duration_s, dt) -> List[IMUFrame]
```

---

## 6. 仿真层接口 (Simulation)

### 6.1 仿真环境 — RobotSimulator / SensorSimulator

```python
class RobotSimulator:
    def __init__(self, config: SimConfig, ...)
    def set_joint_positions(self, positions: np.ndarray)
    def step(self, torque_command: np.ndarray) -> Dict[str, Any]
    def get_state(self) -> Dict
    def get_jacobian(self, joint_positions=None) -> np.ndarray
    def check_self_collision(self) -> bool
    def check_environment_collision(self, obstacles) -> List[Dict]
    def reset()

class SensorSimulator:
    def __init__(self, simulator: RobotSimulator, config: SimConfig)
    def get_noisy_joint_positions(self) -> np.ndarray
    def get_noisy_joint_velocities(self) -> np.ndarray
    def get_imu_data(self) -> Dict
    def get_wrench(self) -> np.ndarray
    def get_contact_force(self) -> float
```

---

## 7. 数据流图

```
感知输入:
  相机  ──> BinocularCamera ──> StereoFrame ──┐
  麦克风 ──> BinauralMic ──> AudioFrame ───────┤
  电子皮肤 > TactileArray ──> TactileFrame ───┤
  力矩仪 ──> ForceTorqueSensor ──> Wrench ────┤
  IMU ──> IMUSensor ──> IMUFrame ──────────────┤
                                                v
                                    CrossModalFusion
                                                │
                                                v
                                    UnifiedRepresentation
                                                │
                                   ┌────────────┼────────────┐
                                   v            v            v
                               SelfSupervisedLearner  TaskPlanner  SkillLibrary
                                   │            │            │
                                   v            v            v
                               Action      Task Plan    Skill Execute
                                   │            │            │
                                   v            v            v
                               MotionController ──> 关节力矩命令 ──> RobotSimulator
                                   │            │
                                   v            v
                          ImpedanceController 物理仿真
                                   │
                                   v
                          SensorSimulator (反馈)
```

---

## 8. 编码器模块接口

### 8.1 SensorEncoderWrapper

```python
class SensorEncoderWrapper:
    def __init__(self, config: EncoderConfig)
    def encode(self, sensor_type: str, data: np.ndarray) -> np.ndarray
    def forward(self, sensor_type: str, data: np.ndarray) -> np.ndarray
    def get_output_dim(self) -> int
    def freeze()
    def unfreeze()
```

### 8.2 各模态编码器

```python
class VisionEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor  # Bx3xHxW → Bx512

class AudioEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor  # BxTx1 → Bx128

class TactileEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor  # BxHxW → Bx64

class ForceEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor  # Bx6 → Bx32

class IMUEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor  # Bx6 → Bx32

class MultimodalEncoder(nn.Module):
    def forward(
        self,
        vision: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        tactile: Optional[torch.Tensor] = None,
        force: Optional[torch.Tensor] = None,
        imu: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]
```

---

## 9. 仿真模块接口 (Simulation)

### 9.1 仿真环境 — RobotSimulator

```python
class RobotSimulator:
    def __init__(self, config: Optional[SimConfig] = None)
    def set_joint_positions(self, positions: np.ndarray)
    def step(self, torque_command: np.ndarray) -> Dict[str, Any]
    def get_state(self) -> Dict[str, Any]
    def reset(self)
    def add_callback(self, callback: Callable)
```

**SimConfig 配置:**
```python
@dataclass
class SimConfig:
    dt: float = 0.01              # 时间步长 (s)
    num_joints: int = 6          # 关节数
    gravity: np.ndarray = field(default_factory=lambda: np.array([0, 0, -9.81]))
    position_noise: float = 0.001   # m
    velocity_noise: float = 0.01    # m/s
    sensor_delay: float = 0.0       # s
    control_delay: float = 0.0      # s
    engine: str = "custom"         # "custom" / "pybullet" / "mujoco"
```

---

## 10. ROS2 集成接口

### 10.1 ROS2 通信配置

```python
class ROS2Interface:
    """ROS2 话题/服务接口"""
    
    def __init__(
        self,
        namespace: str = "/supermodel",
        sensor_topics: Dict[str, str] = None,
        control_topic: str = "/joint_commands"
    )
    
    # 传感器话题
    #   /supermodel/camera/left      → sensor_msgs/Image
    #   /supermodel/camera/right     → sensor_msgs/Image
    #   /supermodel/audio             → audio_common_msgs/AudioData
    #   /supermodel/tactile          → geometry_msgs/WrenchStamped
    #   /supermodel/imu               → sensor_msgs/Imu
    #   /supermodel/joint_states      → sensor_msgs/JointState
    
    # 控制话题
    #   /supermodel/joint_commands   → trajectory_msgs/JointTrajectory
    #   /supermodel/gripper_command   → std_msgs/Float64
    
    # 服务
    #   /supermodel/perception        → supermodel_interfaces/srv/Perceive
    #   /supermodel/planning          → supermodel_interfaces/srv/Plan
    #   /supermodel/execute_skill     → supermodel_interfaces/srv/ExecuteSkill
```

### 10.2 ROS2 消息类型

```yaml
# supermodel_interfaces/msg/PerceptionResult.msg
std_msgs/Header header
string[] detected_objects
geometry_msgs/Pose[] object_poses
float64[] confidences
sensor_msgs/PointCloud point_cloud

# supermodel_interfaces/srv/Perceive.srv
sensor_msgs/Image image
---
PerceptionResult result

# supermodel_interfaces/srv/ExecuteSkill.srv
string skill_name
string[] parameters
---
bool success
string message
```

---

## 11. 数据格式规范

### 11.1 时间同步

所有传感器数据必须带有时间戳，建议使用：
- 硬件时间戳 (传感器原生)
- NTP 同步 (网络传感器)
- PTP 精确时间协议 (工业级)

### 11.2 坐标系约定

| 坐标系 | 说明 |
|--------|------|
| base_link | 机器人基座中心 |
| tool0 / ee_link | 末端执行器 |
| camera_color_optical_frame | 相机光心 (Z向前) |
| camera_depth_optical_frame | 深度相机光心 |
| world | 地图原点 |

### 11.3 单位制

| 物理量 | 单位 | 说明 |
|--------|------|------|
| 长度 | m (米) | 所有位置/距离 |
| 角度 | rad (弧度) | 内部计算 |
| 角速度 | rad/s | 角速度 |
| 力 | N (牛顿) | 力/力矩 |
| 质量 | kg | 质量/负载 |
| 温度 | °C | 摄氏度 |
| 磁场 | μT | 磁力计 |

---

## 12. 安全控制器接口 (Safety Controller)

### 12.1 概述

安全控制器是 AGV 五级安全体系的核心实现，提供从 S 级（基础限位）到 XXL 级（故障容忍）的安全监控能力。

### 12.2 安全等级对照

| 等级 | 核心功能 | 响应时间 | 冗余度 |
|------|----------|----------|--------|
| S | 关节位置限位、软件速度限幅 | 100ms | 无 |
| M | + 速度监控、警告系统 | 50ms | 单通道 |
| L | + 碰撞检测、力矩监控、自动减速 | 20ms | 双通道 |
| XL | + 看门狗、实时故障诊断 | 5ms | 双通道+独立监控 |
| XXL | + 故障容忍、自动恢复、预测性维护 | 1ms | 全冗余 |

### 12.3 SafetyController 类

```python
class SafetyController:
    def __init__(self, config: SafetyConfig)
    
    # 状态属性
    @property
    def safety_level(self) -> SafetyLevel
    @property
    def is_emergency_stopped(self) -> bool
    @property
    def fault_count(self) -> int
    @property
    def event_history(self) -> List[SafetyEventRecord]
    
    # 控制方法
    def enable(self)
    def disable(self)
    def emergency_stop(self)        # 触发紧急停止
    def reset(self)                 # 重置控制器
    
    # 核心检查
    def check(self, state: JointStateSnapshot) -> SafetyCheckResult
    def execute_response(self, result: SafetyCheckResult) -> SafetyResponse
    def compute_safe_velocity(self, current_vel, desired_vel) -> np.ndarray
    
    # 辅助方法
    def register_callback(self, event: SafetyEvent, 
                         callback: Callable[[SafetyEventRecord], None])
    def get_safety_status(self) -> Dict[str, Any]
```

### 12.4 SafetyConfig 配置

```python
@dataclass
class SafetyConfig:
    joint_limits_lower: np.ndarray    # 关节下限 (rad)
    joint_limits_upper: np.ndarray    # 关节上限 (rad)
    velocity_limits: np.ndarray       # rad/s
    velocity_warning_ratio: float     # 警告阈值比例 (默认 0.8)
    acceleration_limits: np.ndarray   # rad/s^2
    torque_limits: np.ndarray         # Nm
    force_limits: np.ndarray          # N
    collision_threshold: float       # N, 碰撞力阈值
    collision_time_threshold: float   # s, 碰撞判定时间
    watchdog_timeout: float           # s, 看门狗超时
    temperature_limits: Tuple[float, float]  # 摄氏度
    safety_level: SafetyLevel         # 安全等级
    max_fault_count: int              # 最大容错次数
    recovery_timeout: float            # 恢复超时 (s)
```

### 12.5 安全事件类型

```python
class SafetyEvent(Enum):
    JOINT_LIMIT = "joint_limit"         # 关节限位
    VELOCITY_LIMIT = "velocity_limit"   # 速度超限
    ACCELERATION_LIMIT = "acceleration_limit"  # 加速度超限
    COLLISION_DETECTED = "collision_detected"  # 碰撞检测
    EMERGENCY_STOP = "emergency_stop"  # 紧急停止
    WATCHDOG_TIMEOUT = "watchdog_timeout"  # 看门狗超时
    TORQUE_LIMIT = "torque_limit"      # 力矩超限
    TEMPERATURE_HIGH = "temperature_high"  # 温度过高
    POWER_EXCEPTION = "power_exception"  # 电源异常
```

### 12.6 安全响应策略

```python
class SafetyResponse(Enum):
    WARNING = "warning"          # 仅警告
    SLOWDOWN = "slowdown"        # 减速
    STOP = "stop"               # 停止
    EMERGENCY_STOP = "emergency_stop"  # 紧急停止
    FAULT_TOLERANT = "fault_tolerant"  # 故障容忍
```

### 12.7 使用示例

```python
from control.safety_controller import (
    SafetyController, SafetyConfig, SafetyLevel, JointStateSnapshot
)
import numpy as np

# 创建配置 (L级安全)
config = SafetyConfig(
    joint_limits_lower=np.array([-3.14, -2.5, -3.14, -3.14, -3.14, -3.14]),
    joint_limits_upper=np.array([3.14, 2.5, 3.14, 3.14, 3.14, 3.14]),
    velocity_limits=np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0]),
    torque_limits=np.array([100, 100, 80, 40, 40, 20]),
    safety_level=SafetyLevel.L,
)

safety = SafetyController(config)

# 定期安全检查 (控制循环中)
state = JointStateSnapshot(
    positions=joint_positions,    # 当前关节位置
    velocities=joint_velocities,   # 当前关节速度
    accelerations=joint_accels,    # 当前关节加速度
    torques=joint_torques,         # 当前关节力矩
    timestamp=time.time()
)

result = safety.check(state)

if not result.safe:
    response = safety.execute_response(result)
    if response == SafetyResponse.EMERGENCY_STOP:
        trigger_estop()
    elif response == SafetyResponse.STOP:
        stop_motion()
    elif response == SafetyResponse.SLOWDOWN:
        reduce_speed()

# 获取安全状态
status = safety.get_safety_status()
print(f"安全等级: {status['safety_level']}")
print(f"总检查次数: {status['total_checks']}")
print(f"近期违规: {status['total_violations']}")
```

### 12.8 等级特征表

```python
SafetyController.LEVEL_FEATURES = {
    SafetyLevel.S:  {"joint_limits", "velocity_limits"},
    SafetyLevel.M: {"joint_limits", "velocity_limits", "velocity_monitoring"},
    SafetyLevel.L: {"joint_limits", "velocity_limits", "collision_detection"},
    SafetyLevel.XL: {"joint_limits", "velocity_limits", "collision_detection", 
                    "watchdog"},
    SafetyLevel.XXL: {"joint_limits", "velocity_limits", "collision_detection", 
                     "watchdog", "fault_tolerance", "recovery"},
}
```

---

## 13. 错误处理规范

所有模块应遵循以下错误处理约定：

```python
class SensorError(Exception):
    """传感器错误基类"""
    pass

class SensorTimeoutError(SensorError):
    """传感器超时"""
    pass

class SensorCalibrationError(SensorError):
    """传感器标定错误"""
    pass

class FusionError(Exception):
    """融合错误基类"""
    pass

class ControlError(Exception):
    """控制错误基类"""
    pass
```

---

## 14. 触觉/力觉/IMU传感器详细接口

### 12.1 触觉传感器 — TactileArray

```python
from sensors.tactile import (
    TactileArray, TactileFrame, TactileContact, TactileCalibration,
    TactileSensorType, PressureProcessor
)

# 初始化
sensor = TactileArray(
    sensor_type=TactileSensorType.PIEZORESISTIVE,
    array_size=(16, 16),
    resolution=12,  # ADC位数
    sampling_rate=100.0  # Hz
)

# 打开传感器
sensor.open()

# 捕获触觉帧
frame = sensor.capture()  # -> TactileFrame

# 帧数据结构
print(f"压力矩阵形状: {frame.pressure_matrix.shape}")  # (H, W)
print(f"温度矩阵形状: {frame.temperature_matrix.shape}")  # (H, W)
print(f"触点列表: {len(frame.contacts)} 个")
for contact in frame.contacts:
    print(f"  触点位置: {contact.centroid}, 力度: {contact.force:.2f}N")

# 接触检测
if sensor.detect_contact(frame):
    print("检测到接触事件")

# 滑移检测
if sensor.get_slip_signal(frame) > 0.5:
    print("检测到滑移!")

# 标定
calibration = TactileCalibration.create_default(16)
calibrated = calibration.apply(frame)

# 压力处理器
processor = PressureProcessor(array_size=(16, 16))
processed = processor.process(frame)
print(f"热力图均值: {processed['mean_pressure']:.2f}")

sensor.close()
```

### 12.2 力觉传感器 — ForceTorqueSensor

```python
from sensors.force import (
    ForceTorqueSensor, ForceCalibration, Wrench, ContactState,
    WrenchProcessor, ForceSensorType
)

# 初始化
sensor = ForceTorqueSensor(
    sensor_type=ForceSensorType.SIX_AXIS_STRAIN_GAUGE,
    force_range=np.array([200.0, 200.0, 200.0]),  # N (x, y, z)
    torque_range=np.array([20.0, 20.0, 20.0]),    # Nm (x, y, z)
    resolution=16
)

sensor.open()

# 捕获六维力/力矩
wrench = sensor.capture()  # -> Wrench

print(f"力: Fx={wrench.force[0]:.2f}N, Fy={wrench.force[1]:.2f}N, Fz={wrench.force[2]:.2f}N")
print(f"力矩: Tx={wrench.torque[0]:.2f}Nm, Ty={wrench.torque[1]:.2f}Nm, Tz={wrench.torque[2]:.2f}Nm")
print(f"合力: {wrench.force_magnitude:.2f}N, 合力矩: {wrench.torque_magnitude:.2f}Nm")
print(f"接触状态: {wrench.contact_state}")

# 坐标系变换
transformed = wrench.transform(np.eye(4), np.array([0, 0, 0.05]))

# 碰撞检测
if sensor.detect_collision(wrench, threshold=50.0):
    print("碰撞检测触发!")

# 负载估计
payload = sensor.estimate_payload(wrench, gravity_vector=np.array([0, 0, -9.81]))
print(f"估计负载: {payload:.2f}kg")

# 力处理器
processor = WrenchProcessor(window_size=5)
filtered = processor.filter(wrench)
print(f"滤波后合力: {filtered.force_magnitude:.2f}N")

sensor.close()
```

### 12.3 IMU传感器 — IMUSensor & PoseEstimator

```python
from sensors.imu import (
    IMUSensor, IMUFrame, Pose, PoseEstimator,
    IMUCalibration, IMUSensorType
)

# 初始化
imu = IMUSensor(
    sensor_type=IMUSensorType.MPU6050,
    accel_range=16.0,   # ±16g
    gyro_range=1000.0,  # ±1000°/s
    sample_rate=100.0   # Hz
)

imu.open()

# 自检
if imu.self_test():
    print("IMU自检通过")

# 校准
calibration = IMUCalibration.create_default()
imu.calibrate_gyro_bias(calibration)

# 捕获IMU帧
frame = imu.capture()  # -> IMUFrame

print(f"加速度: ax={frame.accel[0]:.3f}g, ay={frame.accel[1]:.3f}g, az={frame.accel[2]:.3f}g")
print(f"角速度: wx={frame.gyro[0]:.2f}°/s, wy={frame.gyro[1]:.2f}°/s, wz={frame.gyro[2]:.2f}°/s")
print(f"磁场: mx={frame.mag[0]:.2f}, my={frame.mag[1]:.2f}, mz={frame.mag[2]:.2f}")
print(f"温度: {frame.temperature:.1f}°C")

# 姿态估计
estimator = PoseEstimator(sample_rate=100.0, algorithm='complementary')
estimator.reset()

for _ in range(100):
    frame = imu.capture()
    pose = estimator.update(frame)  # -> Pose

print(f"Euler角: roll={pose.euler[0]:.2f}°, pitch={pose.euler[1]:.2f}°, yaw={pose.euler[2]:.2f}°")
print(f"四元数: {pose.quaternion}")  # [w, x, y, z]
print(f"旋转矩阵:\n{pose.rotation_matrix}")

imu.close()
```

### 12.4 触觉/力觉/IMU接口规格表

| 参数 | 触觉 TactileArray | 力觉 ForceTorqueSensor | IMU IMUSensor |
|------|-------------------|------------------------|---------------|
| 数据类型 | TactileFrame | Wrench | IMUFrame |
| 主要字段 | pressure_matrix, temperature_matrix, contacts | force(3), torque(3), contact_state | accel(3), gyro(3), mag(3) |
| 采样率 | 最高 500Hz | 最高 1000Hz | 最高 1000Hz |
| 关键方法 | capture(), detect_contact(), get_slip_signal() | capture(), detect_collision(), estimate_payload() | capture(), self_test(), calibrate_gyro_bias() |
| 处理器 | PressureProcessor | WrenchProcessor | PoseEstimator |
| 支持等级 | S ~ XXL | M ~ XXL | S ~ XXL |

---

## 15. 跨模态融合网络详细接口

### 14.1 融合配置与输入

```python
from fusion.cross_modal_fusion import (
    FusionStrategy, FusionConfig, MultimodalInput,
    LanguageEncoder, CrossModalAttention, ModalityEncoder,
    CrossModalFusion, UnifiedRepresentation, create_multimodal_input
)
import torch

# 创建融合配置
config = FusionConfig(
    vision_dim=512,
    audio_dim=128,
    tactile_dim=64,
    force_dim=32,
    imu_dim=64,
    language_dim=128,
    hidden_dim=256,
    num_heads=4,
    num_layers=2,
    dropout=0.1,
    strategy=FusionStrategy.HYBRID,
    vocab_size=10000,
    language_max_len=32
)
```

### 14.2 语言编码器

```python
# 语言编码器: Token序列 -> 特征向量
lang_enc = LanguageEncoder(
    vocab_size=10000,
    embed_dim=128,
    hidden_dim=256,
    max_len=32,
    num_heads=4,
    num_layers=2
)

token_ids = torch.randint(0, 10000, (4, 32))  # B=4, L=32
features = lang_enc(token_ids)  # -> torch.Size([4, 256])
```

### 14.3 跨模态注意力

```python
# 两模态之间的注意力交互
attn = CrossModalAttention(query_dim=256, key_dim=256, value_dim=256, num_heads=4)

query = torch.randn(2, 10, 256)   # B=2, N=10, D=256
key   = torch.randn(2, 15, 256)  # B=2, M=15, D=256
value = torch.randn(2, 15, 256)  # B=2, M=15, D=256

output = attn(query, key, value)  # -> torch.Size([2, 10, 256])
```

### 14.4 完整融合流程

```python
# 初始化融合网络
fusion = CrossModalFusion(config)
unified = UnifiedRepresentation(input_dim=256, hidden_dim=384, output_dim=128)

# 构建多模态输入 (所有模态均为可选)
multimodal = MultimodalInput(
    vision=torch.randn(2, 512),           # 视觉特征
    audio=torch.randn(2, 128),            # 音频特征
    tactile=torch.randn(2, 64),           # 触觉特征
    force=torch.randn(2, 32),             # 力觉特征 (6维)
    imu=torch.randn(2, 64),               # IMU特征 (9维)
    language=torch.randint(0, 10000, (2, 32))  # 语言token
)

# 融合前向传播
fused_features = fusion(multimodal)  # -> torch.Size([2, 256])

# 统一表示生成 (三任务头)
state_repr, action_repr, world_repr = unified(fused_features)
# state_repr:  -> torch.Size([2, 128])  状态表示
# action_repr: -> torch.Size([2, 128])  动作策略
# world_repr:  -> torch.Size([2, 128])  世界模型预测

print(f"融合特征: {fused_features.shape}")
print(f"可用模态: {multimodal.modalities}")
```

### 14.5 NumPy 工厂函数

```python
# 使用 NumPy 数组创建多模态输入 (自动转换torch)
mmi = create_multimodal_input(
    vision=np.random.randn(2, 512).astype(np.float32),
    audio=np.random.randn(2, 50, 128).astype(np.float32),
    tactile=np.random.randn(2, 64).astype(np.float32),
    force=np.random.randn(2, 6).astype(np.float32),
    imu=np.random.randn(2, 9).astype(np.float32),
    language=np.random.randint(0, 10000, (2, 32))
)
# 所有numpy数组自动转换为torch.Tensor
```

### 14.6 融合策略说明

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `EARLY` | 早期融合，原始特征拼接后联合编码 | 模态对齐良好时 |
| `LATE` | 晚期融合，各模态独立编码后在决策层融合 | 模态异构性强时 |
| `HYBRID` | 混合融合，结合早期和晚期优点 | 默认，推荐使用 |

### 14.7 融合网络架构参数

| 组件 | 参数 | 默认值 |
|------|------|--------|
| 视觉编码器 | input_dim, output_dim | 512 → 256 |
| 音频编码器 | input_dim, output_dim | 128 → 256 |
| 触觉编码器 | input_dim, output_dim | 64 → 256 |
| 力觉编码器 | input_dim, output_dim | 32 → 256 |
| IMU编码器 | input_dim, output_dim | 64 → 256 |
| 语言编码器 | vocab, embed, hidden, layers | 10000, 128, 256, 2 |
| 跨模态注意力 | num_heads | 4 |
| 融合投影 | hidden_dim * num_modalities → hidden_dim | 可变 |

### 14.8 跨模态注意力对 (Cross-Modal Attention Pairs)

融合网络实现了所有模态两两之间的跨模态注意力交互:

| 注意力对 | 说明 | 应用场景 |
|----------|------|----------|
| `vision_audio` | 视觉-听觉 | 视听联合感知 |
| `vision_tactile` | 视觉-触觉 | 触觉引导的视觉定位 |
| `vision_force` | 视觉-力觉 | 力反馈视觉伺服 |
| `vision_imu` | 视觉-IMU | 姿态辅助视觉里程计 |
| `audio_tactile` | 听觉-触觉 | 声触联合感知 |
| `audio_force` | 听觉-力觉 | 碰撞声音检测 |
| `audio_imu` | 听觉-IMU | 运动声音同步 |
| `force_imu` | 力觉-IMU | 运动力学融合 |
| `vision_language` | 视觉-语言 | 视觉语言对齐 |

---

## 16. AGV五级规格对照 (增强)

### 15.1 传感器等级配置

| 传感器 | S级 | M级 | L级 | XL级 | XXL级 |
|--------|-----|-----|-----|------|-------|
| 触觉阵列 | 8×8 Piezoresistive | 16×16 Piezoresistive | 24×24 Hybrid | 32×32 Capacitive | 48×48 Multi-modal |
| 力觉 | — | 6轴 ±200N | 6轴 ±500N | 6轴 ±1000N | 6轴 ±5000N |
| IMU | MPU6050 (6轴) | BMI088 (6轴) | BMI088 + Mag | ADIS16470 | 双 ADIS16470 |
| 控制频率 | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |

---

## 17. 执行控制系统 AGV 五级规格汇总

本文档汇总所有执行控制相关的 AGV 五级规格，提供快速参考。

### 16.1 控制频率与精度

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 控制频率 (Hz) | 50 | 100 | 200 | 500 | 1000 |
| 位置控制精度 (mm) | ±5.0 | ±1.0 | ±0.5 | ±0.1 | ±0.01 |
| 力控精度 (N) | ±1.0 | ±0.5 | ±0.2 | ±0.1 | ±0.05 |
| 轨迹重规划周期 (ms) | 100 | 50 | 20 | 10 | 5 |

### 16.2 运动控制模式支持

| 控制模式 | S | M | L | XL | XXL |
|----------|---|---|---|---|-----|
| 关节位置控制 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 关节速度控制 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 关节力矩控制 | — | ✓ | ✓ | ✓ | ✓ |
| 笛卡尔位置控制 | — | ✓ | ✓ | ✓ | ✓ |
| 笛卡尔速度控制 | — | ✓ | ✓ | ✓ | ✓ |
| 位置阻抗控制 | — | ✓ | ✓ | ✓ | ✓ |
| 力阻抗控制 | — | — | ✓ | ✓ | ✓ |
| 力位混合控制 | — | — | ✓ | ✓ | ✓ |
| 导纳控制 | — | — | ✓ | ✓ | ✓ |
| 自适应阻抗 | — | — | — | ✓ | ✓ |
| 预测控制 | — | — | — | ✓ | ✓ |

### 16.3 轨迹规划算法支持

| 算法 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 线性插值 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 三次多项式 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 五次多项式 | — | ✓ | ✓ | ✓ | ✓ |
| S 曲线 | — | ✓ | ✓ | ✓ | ✓ |
| 梯形速度曲线 | ✓ | ✓ | ✓ | ✓ | ✓ |
| RRT | ✓ | ✓ | ✓ | ✓ | ✓ |
| RRT* | — | ✓ | ✓ | ✓ | ✓ |
| PRM | — | — | ✓ | ✓ | ✓ |
| 混合 A* | — | — | ✓ | ✓ | ✓ |
| CHOMP | — | — | — | ✓ | ✓ |
| STOMP | — | — | — | ✓ | ✓ |

### 16.4 技能库等级配置

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 内置技能数量 | 5 | 15 | 30 | 50 | 100+ |
| LfD 演示学习 | — | ✓ | ✓ | ✓ | ✓ |
| 技能合成 | — | — | ✓ | ✓ | ✓ |
| 主动技能学习 | — | — | — | ✓ | ✓ |
| 技能迁移 | — | — | — | ✓ | ✓ |

### 16.5 安全等级配置

| 安全功能 | S | M | L | XL | XXL |
|----------|---|---|---|---|-----|
| 软限位 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 硬限位 | — | ✓ | ✓ | ✓ | ✓ |
| 碰撞检测 (力) | — | ✓ | ✓ | ✓ | ✓ |
| 碰撞检测 (视觉) | — | — | — | ✓ | ✓ |
| 碰撞预测 | — | — | — | ✓ | ✓ |
| 人机协作安全 | — | ✓ | ✓ | ✓ | ✓ |
| 预测性安全 | — | — | — | ✓ | ✓ |
| 多机协调安全 | — | — | — | ✓ | ✓ |
| 故障诊断 | — | ✓ | ✓ | ✓ | ✓ |
| 安全审计日志 | — | ✓ | ✓ | ✓ | ✓ |

### 16.6 ROS2 通信等级配置

| 功能 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 话题发布/订阅 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 关节轨迹 Action | — | ✓ | ✓ | ✓ | ✓ |
| 生命周期管理 | — | ✓ | ✓ | ✓ | ✓ |
| 组件组合 (Composition) | — | — | ✓ | ✓ | ✓ |
| 实时节点 (rclpy/rclcpp) | — | — | ✓ | ✓ | ✓ |
| 硬件加速 (GPU/DSP) | — | — | — | ✓ | ✓ |
| 多机器人协调 | — | — | — | ✓ | ✓ |
| 5G/远程操控 | — | — | ✓ | ✓ | ✓ |

### 16.7 接口快速查询

```python
# 控制模块导入
from control import (
    MotionController, TrajectoryGenerator, RRTPlanner,
    ImpedanceController, SkillLibrary, TaskPlanner,
    SafetyController, ROS2JointTrajectoryInterface
)

# 创建 S 级控制器 (低成本，低频率)
ctrl_s = MotionController(num_joints=6, control_rate=50)

# 创建 XXL 级控制器 (高精度，高频率)
ctrl_xxl = MotionController(num_joints=7, control_rate=1000)

# 从规格表获取参数
from control.trajectory import get_trajectory_spec
from control.safety_controller import get_safety_spec
from control.ros2_interface import get_ros2_spec

spec_traj = get_trajectory_spec('XL')
spec_safety = get_safety_spec('XXL')
spec_ros2 = get_ros2_spec('XL')
```

---

*文档版本: v0.7.0*
*最后更新: 2026-03-29*

**2026-03-29 v1.0.0**: 文档去重重构，整合为单一完整版本 (Sections 1-21)

## 18. 传感器 AGV 五级规格速查

### 17.1 触觉传感器 TactileArray 规格表

```python
from sensors.tactile import get_tactile_spec, TactileArray, TactileSensorType

# 获取各等级规格
spec_s = get_tactile_spec('S')   # 8x8, 12bit, 50Hz
spec_m = get_tactile_spec('M')   # 16x16, 12bit, 100Hz + 温度
spec_l = get_tactile_spec('L')   # 24x24, 14bit, 200Hz + 温度 + 接近觉
spec_xl = get_tactile_spec('XL') # 32x32, 14bit, 500Hz + 全功能
spec_xxl = get_tactile_spec('XXL') # 48x48, 16bit, 1000Hz + 全功能

# 创建传感器
tactile = TactileArray(
    array_size=spec_xl['array'],
    sensor_type=TactileSensorType.CAPACITIVE
)
tactile.open()
frame = tactile.capture()
contacts = tactile.detect_contacts(frame)
tactile.close()
```

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 阵列尺寸 | 8×8 | 16×16 | 24×24 | 32×32 | 48×48 |
| 分辨率 (bit) | 12 | 12 | 14 | 14 | 16 |
| 压力范围 (kPa) | 0-500 | 0-1000 | 0-2000 | 0-5000 | 0-10000 |
| 采样频率 (Hz) | 50 | 100 | 200 | 500 | 1000 |
| 温度感知 | ✗ | ✓ | ✓ | ✓ | ✓ |
| 接近觉 | ✗ | ✗ | ✓ | ✓ | ✓ |

### 17.2 力觉传感器 ForceTorqueSensor 规格表

```python
from sensors.force import get_force_spec, ForceTorqueSensor, ForceSensorType, Wrench, ContactState

# 获取各等级规格
spec_s = get_force_spec('S')   # 3轴, ±100N, 100Hz
spec_m = get_force_spec('M')   # 6轴, ±200N, 500Hz
spec_l = get_force_spec('L')   # 6轴, ±500N, 1000Hz
spec_xl = get_force_spec('XL') # 6轴, ±1000N, 2000Hz
spec_xxl = get_force_spec('XXL') # 6轴, ±5000N, 5000Hz

# 创建传感器
sensor = ForceTorqueSensor(
    sensor_type=ForceSensorType.SIX_AXIS,
    sensor_id='ft_wrist'
)
sensor.open()
wrench = sensor.capture()
contact = sensor.detect_contact(wrench)
payload = sensor.estimate_payload(wrench)
sensor.close()
```

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 轴数 | 3 | 6 | 6 | 6 | 6 |
| 力范围 (N) | ±100 | ±200 | ±500 | ±1000 | ±5000 |
| 力矩范围 (N·m) | ±10 | ±20 | ±50 | ±100 | ±500 |
| 分辨率 | 0.1N | 0.05N | 0.02N | 0.01N | 0.005N |
| 采样频率 (Hz) | 100 | 500 | 1000 | 2000 | 5000 |

### 17.3 IMU传感器 IMUSensor 规格表

```python
from sensors.imu import get_imu_spec, IMUSensor, IMUSensorType, PoseEstimator, Pose

# 获取各等级规格
spec_s = get_imu_spec('S')   # MPU6050, 8g, 1000dps, 100Hz
spec_m = get_imu_spec('M')   # BMI088, 16g, 2000dps, 200Hz
spec_l = get_imu_spec('L')   # BMI088, 24g, 4000dps, 500Hz
spec_xl = get_imu_spec('XL') # ADIS16470, 40g, 4000dps, 1000Hz
spec_xxl = get_imu_spec('XXL') # ADIS16470, 80g, 8000dps, 2000Hz

# 创建传感器
imu = IMUSensor(
    sensor_type=IMUSensorType.BMI088,
    accel_range=16,
    gyro_range=2000,
    sample_rate=500
)
imu.open()
imu.self_test()
frame = imu.capture()

# 姿态估计
estimator = PoseEstimator(algorithm='madgwick', beta=0.1)
pose = estimator.update(frame.accel, frame.gyro)
euler = pose.to_euler()  # [roll, pitch, yaw] in rad
imu.close()
```

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 传感器型号 | MPU6050 | BMI088 | BMI088 | ADIS16470 | ADIS16470 |
| 加速度量程 (g) | 8 | 16 | 24 | 40 | 80 |
| 陀螺量程 (°/s) | 1000 | 2000 | 4000 | 4000 | 8000 |
| 采样频率 (Hz) | 100 | 200 | 500 | 1000 | 2000 |
| 噪声密度 (μg/√Hz) | 400 | 120 | 60 | 20 | 10 |

### 17.4 传感器融合使用示例

```python
# 多传感器融合使用示例
from sensors.vision import BinocularCamera
from sensors.audio import BinauralMic
from sensors.tactile import TactileArray
from sensors.force import ForceTorqueSensor
from sensors.imu import IMUSensor, PoseEstimator

# 初始化
cam = BinocularCamera()
mic = BinauralMic()
tactile = TactileArray(array_size=(24, 24))
force = ForceTorqueSensor()
imu = IMUSensor(sensor_type=IMUSensorType.BMI088, sample_rate=500)

# 打开所有传感器
for s in [cam, mic, tactile, force, imu]:
    s.open()

# 采集数据
stereo = cam.capture()        # 双目视觉
audio = mic.capture()         # 双耳声觉
tac_frame = tactile.capture() # 触觉
wrench = force.capture()      # 六维力矩
imu_frame = imu.capture()     # IMU

# 姿态估计
pose_est = PoseEstimator(algorithm='madgwick', beta=0.1)
pose = pose_est.update(imu_frame.accel, imu_frame.gyro)

# 接触检测
contacts = tactile.detect_contacts(tac_frame)
contact_state = force.detect_contact(wrench)

# 清理
for s in [cam, mic, tactile, force, imu]:
    s.close()
```

---

## 19. 模型预测控制 (MPC)

### 18.1 关节空间 MPC — JointSpaceMPC

```python
from control.mpc import (
    MPCConfig, JointSpaceMPC, CartesianMPC,
    DynamicsModel, get_mpc_spec
)

# 获取各等级规格
spec = get_mpc_spec('M')  # horizon=20, dt=0.01, solver='qp'
spec_xxl = get_mpc_spec('XXL')  # horizon=50, dt=0.002, solver='osqp'

# 创建配置
config = MPCConfig.for_grade('L', num_joints=6, dt=0.01)

# 创建动力学模型
dynamics = DynamicsModel(
    num_joints=6,
    mass_matrix_diag=np.ones(6) * 0.5,
    damping=np.ones(6) * 2.0
)

# 创建关节空间 MPC
mpc = JointSpaceMPC(config=config, dynamics=dynamics, num_joints=6)

# 当前状态
current_pos = np.zeros(6)
current_vel = np.zeros(6)

# 期望轨迹 (horizon x 6)
target_traj = np.tile(np.array([0.5, 0.2, 0.1, 0, 0, 0]), (20, 1))
target_traj[:, 0] = np.linspace(0, 0.5, 20)

# 计算控制力矩
tau = mpc.compute_control(current_pos, target_traj, current_vel)
# tau: 关节力矩命令 (6,)

# 简化接口 (PD + 前馈)
tau_simple = mpc.compute_control_simple(current_pos, current_vel, target_pos=np.ones(6) * 0.3)
```

### 18.2 笛卡尔空间 MPC — CartesianMPC

```python
# 笛卡尔空间 MPC
cart_mpc = CartesianMPC(config=config, num_joints=6)

# 当前关节状态
current_joint_pos = np.zeros(6)
current_joint_vel = np.zeros(6)

# 目标末端执行器位姿 [x, y, z, roll, pitch, yaw]
target_pose = np.array([0.3, 0.1, 0.5, 0.1, 0.0, 0.0])
target_twist = np.zeros(6)  # 目标速度

tau = cart_mpc.compute_control(current_joint_pos, current_joint_vel, target_pose, target_twist)
```

### 18.3 动力学模型 — DynamicsModel

```python
model = DynamicsModel(num_joints=6)

# 前向动力学
q = np.zeros(6)
qd = np.zeros(6)
tau = np.array([0, 0, 10, 0, 0, 0])
qdd = model.forward(q, qd, tau)

# 线性化
A, B, G = model.linearize(q, qd)

# 离散化
Ad, Bd = model.discrete_matrices(q, qd, dt=0.01)
```

### 18.4 MPC 控制器规格表

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 预测步数 | 10 | 20 | 30 | 40 | 50 |
| 控制步数 | 5 | 10 | 15 | 20 | 25 |
| 采样周期 (s) | 0.02 | 0.01 | 0.01 | 0.005 | 0.002 |
| 最大力矩 (Nm) | 50 | 100 | 200 | 500 | 1000 |
| 求解器 | qp | qp | osqp | osqp | osqp |
| 约束 | 关节/速度 | 关节/速度/力矩 | +碰撞 | +障碍 | +力约束 |
| 描述 | 基础 MPC | 标准 QP | 增强 OSQP | 碰撞回避 | 旗舰多目标 |

---

## 20. Gymnasium RL 环境

### 19.1 Gymnasium 环境 — SuperModelGymEnv

```python
from simulation.gym_env import (
    SuperModelGymEnv, GymEnvConfig,
    make_env, collect_rollout, get_gym_spec, register_gym_envs
)

# 直接创建环境
config = GymEnvConfig(grade='M', scenario='reach', dt=0.01)
env = SuperModelGymEnv(config=config, scenario='reach')

# 重置
obs, info = env.reset(seed=42)
assert obs.shape == (53,)  # joint_pos(6) + joint_vel(6) + ee_pos(3) +
                            # ee_quat(4) + imu(6) + wrench(6) +
                            # tactile(16) + target(6)

# 一步
action = env.action_space.sample()  # 关节力矩 (6,)
obs, reward, terminated, truncated, info = env.step(action)

# 渲染
env.render()  # human 模式打印状态

env.close()
```

### 19.2 使用 make_env 快捷创建

```python
# 注册并创建环境
env = make_env(scenario='reach', grade='M', render_mode=None, seed=42)

obs, info = env.reset()
for _ in range(100):
    action = policy(obs)  # policy: obs -> action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

### 19.3 收集 Rollout

```python
from simulation.gym_env import collect_rollout

def random_policy(obs):
    return env.action_space.sample()

rollout = collect_rollout(env, random_policy, max_steps=500, render=False)

print(f"Episode length: {rollout['length']}")
print(f"Total reward: {rollout['total_reward']:.2f}")
# rollout['observations']: (T, 53)
# rollout['actions']: (T, 6)
# rollout['rewards']: (T,)
```

### 19.4 观测空间与动作空间

**观测向量 (53 维):**
```
joint_pos(6) + joint_vel(6) + ee_pos(3) + ee_quat(4) +
imu_accel(3) + imu_gyro(3) + wrench(6) + tactile(16) + target(6) = 53
```

**动作空间:** `Box(low=-100, high=100, shape=(6,))` — 关节力矩命令 (Nm)

### 19.5 场景类型

| 场景 | 描述 | 目标 |
|------|------|------|
| `reach` | 关节位置控制 | 到达随机关节位置 |
| `track` | 正弦轨迹跟踪 | 跟踪周期性轨迹 |
| `grasp` | 抓取任务 | 到达抓取姿态 |

### 19.6 Gym 环境规格表

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 控制周期 (s) | 0.02 | 0.01 | 0.01 | 0.005 | 0.002 |
| Episode 长度 | 500 | 1000 | 1000 | 2000 | 5000 |
| 观测噪声 | 0.01 | 0.005 | 0.002 | 0.001 | 0.0005 |
| 跟踪奖励权重 | 1.0 | 1.0 | 2.0 | 5.0 | 10.0 |
| 最大力矩 (Nm) | 50 | 100 | 200 | 500 | 1000 |
| 描述 | 教育级 | 标准级 | 专业级 | 高性能 | 旗舰级 |

---

## 21. 完整系统集成示例

```python
"""
SuperModel 完整系统集成
========================
从传感器采集到控制执行的完整流程
"""

import numpy as np
from sensors.vision import BinocularCamera
from sensors.audio import BinauralMic
from sensors.tactile import TactileArray
from sensors.force import ForceTorqueSensor
from sensors.imu import IMUSensor, PoseEstimator
from fusion.cross_modal_fusion import CrossModalFusion
from control.mpc import MPCConfig, JointSpaceMPC, CartesianMPC
from simulation.gym_env import make_env

# ===== 1. 初始化各子系统 =====

# 传感器初始化 (AGV L 级)
cam = BinocularCamera(resolution=(1280, 720), fps=60)
mic = BinauralMic(sample_rate=22050)
tactile = TactileArray(array_size=(24, 24), sample_rate=200)
force = ForceTorqueSensor(grade='L')
imu = IMUSensor(sensor_type='BMI088', sample_rate=500)

# 融合网络初始化
fusion = CrossModalFusion(config={
    'hidden_dim': 512,
    'num_heads': 8,
    'num_layers': 4,
    'dropout': 0.1
})

# MPC 控制器 (L 级)
mpc_config = MPCConfig.for_grade('L', num_joints=6, dt=0.01)
mpc = JointSpaceMPC(config=mpc_config, num_joints=6)

# 姿态估计
pose_est = PoseEstimator(algorithm='madgwick', beta=0.1)

# ===== 2. 打开传感器 =====
for s in [cam, mic, tactile, force, imu]:
    s.open()

# ===== 3. 主循环 =====
for step in range(1000):
    # --- 感知层 ---
    stereo_frame = cam.capture()
    audio_frame = mic.capture()
    tac_frame = tactile.capture()
    wrench = force.capture()
    imu_frame = imu.capture()

    # --- 感知融合 ---
    observations = {
        'vision': stereo_frame.left_image,
        'audio': audio_frame.left_channel,
        'tactile': tac_frame.pressure_map,
        'force': wrench,
        'imu': np.concatenate([imu_frame.accel, imu_frame.gyro]),
    }

    fused_features = fusion(observations)

    # --- 决策规划 ---
    pose = pose_est.update(imu_frame.accel, imu_frame.gyro)

    # --- MPC 控制 ---
    current_joint_pos = np.array([0.1, 0.05, 0.0, 0.0, 0.0, 0.0])
    current_joint_vel = np.zeros(6)

    # 目标: 跟踪期望轨迹
    target_pos = np.array([0.3, 0.1, 0.1, 0.0, 0.0, 0.0])
    tau = mpc.compute_control_simple(current_joint_pos, current_joint_vel, target_pos)

    # --- 触觉反馈 ---
    contacts = tactile.detect_contacts(tac_frame)
    if contacts:
        print(f"检测到 {len(contacts)} 个接触点")

    # --- 安全检查 ---
    if np.any(np.abs(tau) > 200):
        print("警告: 力矩超限!")
        tau = np.clip(tau, -200, 200)

# ===== 4. 清理 =====
for s in [cam, mic, tactile, force, imu]:
    s.close()

print("系统正常退出")
```

---

---

## 11. AGV 控制等级规格 (Control Grade Specifications)

### 11.1 控制模块等级速查

| 功能 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| PID关节控制 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 关节速度控制 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 关节力矩控制 | - | ✓ | ✓ | ✓ | ✓ |
| 笛卡尔速度控制 | - | ✓ | ✓ | ✓ | ✓ |
| 笛卡尔位置控制 | - | ✓ | ✓ | ✓ | ✓ |
| 位置阻抗控制 | - | ✓ | ✓ | ✓ | ✓ |
| 力阻抗控制 | - | - | ✓ | ✓ | ✓ |
| 力位混合控制 | - | - | ✓ | ✓ | ✓ |
| 导纳控制 | - | - | ✓ | ✓ | ✓ |
| 自适应阻抗 | - | - | - | ✓ | ✓ |
| MPC (模型预测) | - | ✓ | ✓ | ✓ | ✓ |
| 碰撞回避 MPC | - | - | - | ✓ | ✓ |
| 多目标优化 | - | - | - | - | ✓ |
| RRT 路径规划 | ✓ | ✓ | ✓ | ✓ | ✓ |
| RRT* 渐进最优 | - | ✓ | ✓ | ✓ | ✓ |
| Informed RRT | - | - | ✓ | ✓ | ✓ |
| S曲线插值 | - | ✓ | ✓ | ✓ | ✓ |
| 五次多项式轨迹 | - | ✓ | ✓ | ✓ | ✓ |
| 安全限位 | 软限 | 软+硬限 | 限+监控 | 预测 | 预测+协作 |
| 紧急停止 | ✓ | ✓ | ✓ | ✓ | ✓ |

### 11.2 运动控制等级规格

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 控制频率 (Hz) | 50 | 100 | 200 | 500 | 1000 |
| 关节数支持 | 3-6 | 3-6 | 6-7 | 6-7 | 7+ |
| 轨迹插值 | 线性 | 三次 | 五次 | 五次+S曲线 | 最优+约束 |
| 控制延迟 (ms) | < 20 | < 10 | < 5 | < 2 | < 1 |
| 位置精度 (mm) | ±5 | ±1 | ±0.5 | ±0.1 | ±0.01 |
| 力控精度 (N) | ±1 | ±0.5 | ±0.2 | ±0.1 | ±0.05 |
| 最大速度 (m/s) | 0.5 | 1.0 | 2.0 | 3.0 | 5.0 |
| 最大负载 (kg) | 2 | 5 | 20 | 50 | 200 |

### 11.3 MPC 控制器等级规格

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 预测步数 N | 10 | 20 | 30 | 40 | 50 |
| 控制步数 nu | 5 | 10 | 15 | 20 | 25 |
| 采样时间 (ms) | 20 | 10 | 10 | 5 | 2 |
| 求解器 | QP | QP | OSQP | OSQP | OSQP |
| 约束类型 | 位置+速度 | +力矩 | +碰撞 | +障碍 | +力+多目标 |
| 最大力矩 (Nm) | 50 | 100 | 200 | 500 | 1000 |
| 关节限位 | 软 | 软 | 硬 | 硬+裕度 | 硬+安全裕 |

### 11.4 安全控制器等级规格

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 安全等级 | S | M | L | XL | XXL |
| 关节限位检查 | 软 | 软+硬 | ✓ | ✓ | ✓ |
| 速度限制检查 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 力矩限制检查 | - | ✓ | ✓ | ✓ | ✓ |
| 加速度限制检查 | - | - | ✓ | ✓ | ✓ |
| 看门狗监控 | - | - | ✓ | ✓ | ✓ |
| 碰撞检测 (力) | - | ✓ | ✓ | ✓ | ✓ |
| 碰撞检测 (视觉) | - | - | - | ✓ | ✓ |
| 碰撞预测 | - | - | - | ✓ | ✓ |
| 故障容忍 | - | - | - | - | ✓ |
| 协作安全 | - | - | ✓ | ✓ | ✓ |

### 11.5 ROS2 通信等级规格

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 通信协议 | REST | ROS2 | ROS2 | ROS2 | ROS2+自定义 |
| 主题 QoS | BEST_EFFORT | BEST_EFFORT | RELIABLE | RELIABLE | RELIABLE |
| 实时性 | 非实时 | 软实时 | 硬实时 | 硬实时 | 双系统 |
| 最大频率 (Hz) | 10 | 100 | 200 | 500 | 1000 |
| 消息队列深度 | 1 | 5 | 10 | 20 | 50 |
| 安全加密 | - | - | ✓ | ✓ | ✓ |

### 11.6 技能库等级规格

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 内置技能数 | 5 | 15 | 30 | 50 | 100+ |
| 自主学习技能 | - | 3 | 10 | 20 | 50+ |
| 技能编排 | 顺序 | 顺序+并行 | HTN | HTN+条件 | HTN+概率 |
| 视觉引导 | - | ✓ | ✓ | ✓ | ✓ |
| 力控技能 | - | ✓ | ✓ | ✓ | ✓ |
| 双手协作 | - | - | ✓ | ✓ | ✓ |
| 自主规划 | - | - | ✓ | ✓ | ✓ |

---

## 22. 虚拟传感器接口 (Virtual Sensors)

虚拟传感器用于仿真环境中的离线算法验证和测试，无需硬件即可模拟完整感知流程。

### 22.1 虚拟触觉传感器 — VirtualTactileSensor

```python
from sensors.tactile import VirtualTactileSensor, TactileFrame

# 初始化
vt = VirtualTactileSensor(array_size=(16, 16), sensor_id="virtual_tactile")

# 模拟接触事件
frame = vt.simulate_contact(
    contact_pos=(0.5, 0.5),    # 归一化接触中心 (x, y)
    contact_radius=0.3,         # 接触半径
    contact_force=10.0,         # 接触力 (N)
    noise_level=0.05            # 噪声水平
)

# 模拟滑移动作 (返回多帧)
frames = vt.simulate_sliding(
    direction=(1.0, 0.0),     # 滑动方向 (dx, dy)
    speed=0.1,                  # 滑动速度
    duration_frames=30          # 持续帧数
)

# 上下文管理器用法
with VirtualTactileSensor((24, 24)) as vt:
    frame = vt.simulate_contact((0.5, 0.5), contact_force=5.0)
    print(f"触觉帧: {frame.frame_id}, 压力范围: [{frame.pressure_map.min():.3f}, {frame.pressure_map.max():.3f}]")
```

### 22.2 虚拟力觉传感器 — VirtualForceSensor

```python
from sensors.force import VirtualForceSensor, Wrench

# 初始化
vf = VirtualForceSensor(
    sensor_id="virtual_force",
    noise_level=0.02,
    bias_range=0.1
)

# 模拟接触力
wrench = vf.simulate_contact(
    force=(0.0, 0.0, -10.0),   # 力向量 (Fx, Fy, Fz) N
    torque=(0.0, 0.0, 0.0),     # 力矩向量 (Tx, Ty, Tz) N·m
    add_noise=True
)

# 模拟负载重力
wrench = vf.simulate_payload(
    mass=1.0,                   # 负载质量 (kg)
    com_offset=(0.0, 0.0, 0.1), # 重心偏移 (m)
    gravity=9.81
)

# 模拟碰撞事件 (返回力曲线)
collision_frames = vf.simulate_collision(
    direction=(1.0, 0.0, 0.0), # 碰撞方向
    peak_force=50.0,             # 峰值力 (N)
    duration_ms=100.0,           # 持续时间 (ms)
    decay="exponential"          # 衰减模式: exponential / linear
)

# 上下文管理器用法
with VirtualForceSensor() as vf:
    w = vf.simulate_contact(force=(0, 0, -5.0))
    print(f"力: {w.force}, 力矩: {w.torque}")
```

### 22.3 虚拟IMU传感器 — VirtualIMUSensor

```python
from sensors.imu import VirtualIMUSensor, IMUFrame

# 初始化
vi = VirtualIMUSensor(
    sensor_id="virtual_imu",
    accel_noise=0.01,
    gyro_noise=0.001,
    gyro_bias=0.0005
)

# 模拟静止状态 (指定朝向)
frame = vi.simulate_static(
    orientation=(0.0, 0.0, 0.0)  # Euler角 (roll, pitch, yaw) rad
)

# 模拟运动状态
frame = vi.simulate_motion(
    linear_accel=(0.0, 0.0, 0.0), # 线性加速度 (m/s^2)
    angular_vel=(0.0, 0.0, 1.0),   # 角速度 (rad/s)
    dt=0.01
)

# 模拟典型轨迹 (返回IMU帧序列)
frames = vi.simulate_trajectory(
    trajectory_type="circle",   # 轨迹类型: circle / figure8 / linear / sine
    duration_s=2.0,              # 持续时间
    dt=0.01                      # 时间步长
)

# 上下文管理器用法
with VirtualIMUSensor() as vi:
    frame = vi.simulate_static((0.1, 0.1, 0.0))
    print(f"加速度: {frame.accel}, 角速度: {frame.gyro}")
```

### 22.4 虚拟传感器在仿真环境中的集成

```python
# 完整的虚拟传感器集成示例
from sensors.tactile import VirtualTactileSensor
from sensors.force import VirtualForceSensor
from sensors.imu import VirtualIMUSensor
from simulation.environment import RobotSimulator, SimConfig

# 创建仿真环境
sim = RobotSimulator(SimConfig(num_joints=6, dt=0.01))

# 创建虚拟传感器
tactile = VirtualTactileSensor((16, 16))
force = VirtualForceSensor()
imu = VirtualIMUSensor()

# 仿真循环
for step in range(100):
    # 仿真器步进
    state = sim.step(torques=np.zeros(6))
    
    # 采集虚拟传感器数据
    tactile_frame = tactile.simulate_contact(
        contact_pos=(0.5 + 0.01 * step, 0.5),
        contact_force=5.0 + step * 0.1
    )
    
    force_wrench = force.simulate_contact(
        force=(0, 0, -state['joint_torques'][0])
    )
    
    imu_frame = imu.simulate_motion(
        linear_accel=[0, 0, -9.81],
        angular_vel=[0, 0, 0.1 * np.sin(step * 0.1)]
    )
    
    # 多模态融合
    multimodal_data = {
        'tactile': tactile_frame.pressure_map.flatten(),
        'force': force_wrench.to_vector(),
        'imu': np.concatenate([imu_frame.accel, imu_frame.gyro])
    }
    
    print(f"Step {step}: tactile={tactile_frame.pressure_map.mean():.3f}, "
          f"force={force_wrench.magnitude:.2f}N, "
          f"accel={imu_frame.accel_magnitude:.2f}m/s^2")
```

### 22.5 虚拟传感器规格表

| 参数 | VirtualTactileSensor | VirtualForceSensor | VirtualIMUSensor |
|------|---------------------|-------------------|-----------------|
| 模拟精度 | 高斯压力分布 | 噪声注入 | 噪声+偏置注入 |
| 支持阵列 | 8×8 ~ 64×64 | N/A (单点) | N/A |
| 轨迹仿真 | 接触/滑移 | 碰撞力曲线 | 圆/8字/正弦/线性 |
| 噪声控制 | noise_level参数 | noise_level/bias_range | accel_noise/gyro_noise/gyro_bias |
| 典型用例 | 抓取仿真 | 力控算法验证 | 姿态估计验证 |

---

*文档版本: v1.2.0*
*最后更新: 2026-03-30*

**2026-03-30 v1.2.0**: 新增第22节 虚拟传感器接口文档，涵盖 VirtualTactileSensor、VirtualForceSensor、VirtualIMUSensor 的完整API、集成示例和规格对照表，完善仿真层文档体系。


---

## 23. ROS2 接口模块 (ros2_interface)

### 23.1 ROS2JointTrajectoryInterface

```python
class ROS2JointTrajectoryInterface:
    def __init__(self, joint_names: List[str],
                 interface_mode: ControlInterfaceMode = ControlInterfaceMode.POSITION)
    def activate(self)
    def deactivate(self)
    def send_trajectory(self, trajectory: List[JointCommand]) -> bool
    def send_point(self, point: JointCommand) -> bool
    def update(self, current_state: JointState) -> Optional[JointCommand]
    def cancel(self) -> bool
    def get_stats(self) -> Dict[str, Any]
```

### 23.2 ROS2ActionInterface

```python
class ROS2ActionInterface:
    def __init__(self, action_name: str = "joint_trajectory_action")
    def start_server(self)
    def stop_server(self)
    def send_goal(self, trajectory: List[JointCommand], goal_id: Optional[str] = None) -> str
    def update_server(self, current_state: JointState) -> bool
    def cancel_goal(self, goal_id: str) -> bool
    def get_goal_status(self, goal_id: str) -> Optional[ActionGoalStatus]
    def send_goal_async(self, trajectory: List[JointCommand], timeout_sec: float = 300.0) -> str
    def wait_for_result(self, goal_id: str, timeout_sec: Optional[float] = None) -> Optional[ActionResult]
    def cancel_all_goals(self) -> int
    def get_stats(self) -> Dict[str, Any]
```

**ActionFeedback 数据结构:**
```python
@dataclass
class ActionFeedback:
    sequence: int
    percent_complete: float
    current_joint_positions: Optional[np.ndarray]
    error: Optional[np.ndarray]
    message: str
```

**ActionResult 数据结构:**
```python
@dataclass
class ActionResult:
    success: bool
    message: str
    final_positions: Optional[np.ndarray]
    execution_time: float
    trajectory_length: int
```

**使用示例 (Action Server):**
```python
from control.ros2_interface import ROS2ActionInterface, JointCommand, JointState

action = ROS2ActionInterface("follow_joint_trajectory")
action.start_server()

trajectory = [
    JointCommand(positions=np.array([0.1, 0.2, 0.3])),
    JointCommand(positions=np.array([0.5, 0.6, 0.7])),
]

goal_id = action.send_goal(trajectory)

# 主循环
while action.update_server(current_state):
    # 发送反馈 / 监控进度
    pass

action.stop_server()
```

### 23.3 ROS2ParameterInterface

```python
class ROS2ParameterInterface:
    def __init__(self, node_name: str = "supermodel_params")
    def get_parameter(self, name: str, default: Any = None) -> Any
    def set_parameter(self, name: str, value: Any) -> bool
    def get_parameters(self, names: List[str]) -> Dict[str, Any]
    def list_parameters(self, prefix: str = "") -> List[str]
    def declare_parameter(self, name: str, value: Any, descriptor: Optional[Dict] = None)
    def subscribe_parameter_change(self, name: str, callback: Callable[[Any], None])
    def load_from_dict(self, params: Dict[str, Any])
    def to_dict(self) -> Dict[str, Any]
```

**使用示例:**
```python
param = ROS2ParameterInterface("test_node")
param.set_parameter("control.Kp", 1.0)
param.set_parameter("control.Ki", 0.1)

# 订阅变化
param.subscribe_parameter_change("control.Kp", lambda v: print(f"Kp changed to {v}"))
param.set_parameter("control.Kp", 2.0)  # 触发回调

# 批量加载
param.load_from_dict({"fusion.hidden_dim": 512, "agv.max_velocity": 1.5})
```

### 23.4 ROS2ComponentInterface

```python
class ROS2ComponentInterface:
    def __init__(self, component_name: str)
    def on_configure(self, callback: Callable[[], bool])
    def on_activate(self, callback: Callable[[], bool])
    def on_deactivate(self, callback: Callable[[], bool])
    def on_cleanup(self, callback: Callable[[], bool])
    def on_shutdown(self, callback: Callable[[], bool])
    def configure(self) -> bool
    def activate(self) -> bool
    def deactivate(self) -> bool
    def cleanup(self) -> bool
    def shutdown(self) -> bool
    def get_state(self) -> str  # unconfigured/inactive/active/shutdown
```

**ROS2 Lifecycle State Machine:**
```
unconfigured --configure()--> inactive --activate()--> active
     ^                                    |
     |-------deactivate()-------|        |
     ^                                    |
     |--------cleanup()---------|---shutdown()--> shutdown
```

### 23.5 标准话题/服务/参数

**ROSTopics (话题):**
| 话题 | 路径 | 说明 |
|------|------|------|
| CAMERA_LEFT | /supermodel/camera/left | 左侧相机 |
| CAMERA_RIGHT | /supermodel/camera/right | 右侧相机 |
| AUDIO | /supermodel/audio | 音频流 |
| TACTILE | /supermodel/tactile | 触觉数据 |
| IMU | /supermodel/imu | IMU数据 |
| FORCE | /supermodel/force | 力觉数据 |
| JOINT_TRAJECTORY_CMD | /supermodel/joint_trajectory/command | 关节轨迹命令 |
| JOINT_STATES | /supermodel/joint_states | 关节状态 |

**ROSServices (服务):**
| 服务 | 路径 |
|------|------|
| PERCEPTION | /supermodel/perception |
| PLANNING | /supermodel/planning |
| EXECUTE_SKILL | /supermodel/execute_skill |

**ROSParams (参数):**
| 参数 | 说明 |
|------|------|
| control.rate | 控制频率 (Hz) |
| control.max_velocity | 最大速度 (m/s) |
| control.max_acceleration | 最大加速度 (m/s²) |
| camera.exposure | 相机曝光时间 (s) |
| imu.sample_rate | IMU采样率 (Hz) |
| fusion.strategy | 融合策略 |
| fusion.hidden_dim | 融合隐层维度 |

### 23.6 ROS2 AGV五级规格

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| topics | 5 | 10 | 20 | 30 | 50 |
| services | 3 | 5 | 10 | 15 | 25 |
| max_freq_hz | 50 | 100 | 200 | 500 | 1000 |
| realtime | ✗ | ✗ | ✓ | ✓ | ✓ |
| qos_depth | 10 | 10 | 5 | 3 | 1 |

---

*文档版本: v1.3.0*
---

## 23. ROS2 接口模块 (ros2_interface)

### 23.1 ROS2JointTrajectoryInterface

```python
class ROS2JointTrajectoryInterface:
    def __init__(self, joint_names: List[str],
                 interface_mode: ControlInterfaceMode = ControlInterfaceMode.POSITION)
    def activate(self)
    def deactivate(self)
    def send_trajectory(self, trajectory: List[JointCommand]) -> bool
    def send_point(self, point: JointCommand) -> bool
    def update(self, current_state: JointState) -> Optional[JointCommand]
    def cancel(self) -> bool
    def get_stats(self) -> Dict[str, Any]
```

### 23.2 ROS2ActionInterface

```python
class ROS2ActionInterface:
    def __init__(self, action_name: str = "joint_trajectory_action")
    def start_server(self)
    def stop_server(self)
    def send_goal(self, trajectory: List[JointCommand], goal_id: Optional[str] = None) -> str
    def update_server(self, current_state: JointState) -> bool
    def cancel_goal(self, goal_id: str) -> bool
    def get_goal_status(self, goal_id: str) -> Optional[ActionGoalStatus]
    def send_goal_async(self, trajectory: List[JointCommand], timeout_sec: float = 300.0) -> str
    def wait_for_result(self, goal_id: str, timeout_sec: Optional[float] = None) -> Optional[ActionResult]
    def cancel_all_goals(self) -> int
    def get_stats(self) -> Dict[str, Any]
```

**ActionFeedback 数据结构:**
```python
@dataclass
class ActionFeedback:
    sequence: int
    percent_complete: float
    current_joint_positions: Optional[np.ndarray]
    error: Optional[np.ndarray]
    message: str
```

**ActionResult 数据结构:**
```python
@dataclass
class ActionResult:
    success: bool
    message: str
    final_positions: Optional[np.ndarray]
    execution_time: float
    trajectory_length: int
```

**使用示例 (Action Server):**
```python
from control.ros2_interface import ROS2ActionInterface, JointCommand, JointState

action = ROS2ActionInterface("follow_joint_trajectory")
action.start_server()

trajectory = [
    JointCommand(positions=np.array([0.1, 0.2, 0.3])),
    JointCommand(positions=np.array([0.5, 0.6, 0.7])),
]

goal_id = action.send_goal(trajectory)

# 主循环
while action.update_server(current_state):
    # 发送反馈 / 监控进度
    pass

action.stop_server()
```

### 23.3 ROS2ParameterInterface

```python
class ROS2ParameterInterface:
    def __init__(self, node_name: str = "supermodel_params")
    def get_parameter(self, name: str, default: Any = None) -> Any
    def set_parameter(self, name: str, value: Any) -> bool
    def get_parameters(self, names: List[str]) -> Dict[str, Any]
    def list_parameters(self, prefix: str = "") -> List[str]
    def declare_parameter(self, name: str, value: Any, descriptor: Optional[Dict] = None)
    def subscribe_parameter_change(self, name: str, callback: Callable[[Any], None])
    def load_from_dict(self, params: Dict[str, Any])
    def to_dict(self) -> Dict[str, Any]
```

**使用示例:**
```python
param = ROS2ParameterInterface("test_node")
param.set_parameter("control.Kp", 1.0)
param.set_parameter("control.Ki", 0.1)

# 订阅变化
param.subscribe_parameter_change("control.Kp", lambda v: print(f"Kp changed to {v}"))
param.set_parameter("control.Kp", 2.0)  # 触发回调

# 批量加载
param.load_from_dict({"fusion.hidden_dim": 512, "agv.max_velocity": 1.5})
```

### 23.4 ROS2ComponentInterface

```python
class ROS2ComponentInterface:
    def __init__(self, component_name: str)
    def on_configure(self, callback: Callable[[], bool])
    def on_activate(self, callback: Callable[[], bool])
    def on_deactivate(self, callback: Callable[[], bool])
    def on_cleanup(self, callback: Callable[[], bool])
    def on_shutdown(self, callback: Callable[[], bool])
    def configure(self) -> bool
    def activate(self) -> bool
    def deactivate(self) -> bool
    def cleanup(self) -> bool
    def shutdown(self) -> bool
    def get_state(self) -> str  # unconfigured/inactive/active/shutdown
```

**ROS2 Lifecycle State Machine:**
```
unconfigured --configure()--> inactive --activate()--> active
     ^                                    |
     |-------deactivate()-------|        |
     ^                                    |
     |--------cleanup()---------|---shutdown()--> shutdown
```

### 23.5 标准话题/服务/参数

**ROSTopics (话题):**
| 话题 | 路径 | 说明 |
|------|------|------|
| CAMERA_LEFT | /supermodel/camera/left | 左侧相机 |
| CAMERA_RIGHT | /supermodel/camera/right | 右侧相机 |
| AUDIO | /supermodel/audio | 音频流 |
| TACTILE | /supermodel/tactile | 触觉数据 |
| IMU | /supermodel/imu | IMU数据 |
| FORCE | /supermodel/force | 力觉数据 |
| JOINT_TRAJECTORY_CMD | /supermodel/joint_trajectory/command | 关节轨迹命令 |
| JOINT_STATES | /supermodel/joint_states | 关节状态 |

**ROSServices (服务):**
| 服务 | 路径 |
|------|------|
| PERCEPTION | /supermodel/perception |
| PLANNING | /supermodel/planning |
| EXECUTE_SKILL | /supermodel/execute_skill |

**ROSParams (参数):**
| 参数 | 说明 |
|------|------|
| control.rate | 控制频率 (Hz) |
| control.max_velocity | 最大速度 (m/s) |
| control.max_acceleration | 最大加速度 (m/s²) |
| camera.exposure | 相机曝光时间 (s) |
| imu.sample_rate | IMU采样率 (Hz) |
| fusion.strategy | 融合策略 |
| fusion.hidden_dim | 融合隐层维度 |

### 23.6 ROS2 AGV五级规格

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| topics | 5 | 10 | 20 | 30 | 50 |
| services | 3 | 5 | 10 | 15 | 25 |
| max_freq_hz | 50 | 100 | 200 | 500 | 1000 |
| realtime | ✗ | ✗ | ✓ | ✓ | ✓ |
| qos_depth | 10 | 10 | 5 | 3 | 1 |

---

## 24. 控制模块接口 (Control)

### 24.1 AGV运动控制器 — AGVMotionController

```python
class AGVMotionController:
    def __init__(self, spec: AGVSpec)
    def forward_kinematics(self, wheel_velocities: np.ndarray) -> AGVTwist
    def inverse_kinematics(self, twist: AGVTwist) -> np.ndarray
    def compute_wheel_commands(self, target_pose: AGVPose, dt: float) -> np.ndarray
    def apply_safety_limits(self, wheel_commands: np.ndarray) -> np.ndarray
    def update_pose(self, new_pose: AGVPose)
    def update_twist(self, new_twist: AGVTwist)
    @property def pose(self) -> AGVPose
    @property def twist(self) -> AGVTwist
```

**AGVPose / AGVTwist 数据结构:**
```python
@dataclass
class AGVPose:
    x: float          # 世界坐标系X (m)
    y: float          # 世界坐标系Y (m)
    theta: float      # 朝向角 (rad)
    def to_vector(self) -> np.ndarray   # [x, y, theta]
    @classmethod def from_vector(cls, v) -> AGVPose

@dataclass
class AGVTwist:
    vx: float         # X方向线速度 (m/s)
    vy: float         # Y方向线速度 (m/s)
    omega: float      # 角速度 (rad/s)
    def to_vector(self) -> np.ndarray   # [vx, vy, omega]
```

**使用示例:**
```python
from control.agv import AGVMotionController, AGVSpec, AGVGrade, AGVPose, AGVTwist

spec = AGVSpec.from_grade(AGVGrade.M)
agv = AGVMotionController(spec)

agv.update_pose(AGVPose(x=0.0, y=0.0, theta=0.0))
target = AGVPose(x=1.0, y=0.5, theta=0.0)

wheel_cmds = agv.compute_wheel_commands(target, dt=0.01)
wheel_cmds = agv.apply_safety_limits(wheel_cmds)
```

### 24.2 运动学模型 — Kinematics

```python
class KinematicsBase:
    def wheel_to_body(self, wheel_velocities) -> AGVTwist
    def body_to_wheel(self, twist) -> np.ndarray

class DifferentialKinematics(KinematicsBase):
    # 差速驱动: wheel_velocities = [left, right] rad/s
    def wheel_to_body(self, [wL, wR]) -> AGVTwist
    def body_to_wheel(self, twist) -> [wL, wR]

class MecanumKinematics(KinematicsBase):
    # 麦克纳姆轮: wheel_velocities = [fl, fr, rl, rr] rad/s
    def wheel_to_body(self, [wFL, wFR, wRL, wRR]) -> AGVTwist
    def body_to_wheel(self, twist) -> [wFL, wFR, wRL, wRR]
```

### 24.3 安全控制器 — SafetyController

```python
class SafetyController:
    def __init__(self, config: SafetyConfig)
    def check(self, state: JointStateSnapshot) -> SafetyResult
    def execute_response(self, result: SafetyResult) -> SafetyResponse
    def emergency_stop(self)
    def reset()
    def compute_safe_velocity(self, current, desired) -> np.ndarray
    def register_callback(self, event_type, callback)
    def get_safety_status(self) -> Dict
    @property def is_emergency_stopped(self) -> bool
```

**SafetyConfig:**
```python
@dataclass
class SafetyConfig:
    joint_limits_lower: np.ndarray      # 关节下限
    joint_limits_upper: np.ndarray      # 关节上限
    velocity_limits: np.ndarray         # 速度限制
    acceleration_limits: np.ndarray     # 加速度限制
    torque_limits: np.ndarray           # 力矩限制
    watchdog_timeout: float             # 看门狗超时 (s)
    safety_level: SafetyLevel           # S/M/L/XL/XXL
    max_fault_count: int               # 最大故障容忍次数
```

**SafetyResult / SafetyEvent:**
```python
@dataclass
class SafetyResult:
    safe: bool
    watchdog_ok: bool
    events: List[SafetyEventRecord]
    corrected_command: Optional[np.ndarray]

class SafetyEvent(Enum):
    JOINT_LIMIT = "joint_limit"
    VELOCITY_LIMIT = "velocity_limit"
    ACCELERATION_LIMIT = "acceleration_limit"
    COLLISION_DETECTED = "collision_detected"
    EMERGENCY_STOP = "emergency_stop"
    WATCHDOG_TIMEOUT = "watchdog_timeout"
    TORQUE_LIMIT = "torque_limit"
```

### 24.4 MPC控制器 — JointSpaceMPC / CartesianMPC

```python
class MPCConfig:
    horizon: int              # 预测步数 N
    control_horizon: int     # 控制步数 nu
    dt: float               # 采样时间 (s)
    Q_pos: np.ndarray       # 位置跟踪权重
    Q_vel: np.ndarray       # 速度跟踪权重
    R_acc: np.ndarray       # 加速度/控制权重
    torque_limits: np.ndarray
    solver: str            # "osqp" | "qp" | "unconstraint"
    @classmethod def for_grade(cls, grade: str, num_joints=6, dt=0.01) -> MPCConfig

class JointSpaceMPC:
    def __init__(self, config: MPCConfig, dynamics: DynamicsModel, num_joints=6)
    def compute_control(self, current_state, desired_trajectory, current_velocity) -> np.ndarray
    def compute_control_simple(self, current_pos, current_vel, target_pos, target_vel=None) -> np.ndarray
    def reset()

class CartesianMPC:
    def __init__(self, config: MPCConfig, num_joints=6)
    def compute_control(self, current_joint_pos, current_joint_vel, target_pose, target_twist=None) -> np.ndarray
```

**使用示例:**
```python
from control.mpc import MPCConfig, JointSpaceMPC, DynamicsModel

config = MPCConfig.for_grade('L', num_joints=6)
dynamics = DynamicsModel(num_joints=6)
mpc = JointSpaceMPC(config=config, dynamics=dynamics, num_joints=6)

current_pos = np.zeros(6)
current_vel = np.zeros(6)
target_pos = np.array([0.5, 0.3, 0.1, 0, 0, 0])

tau = mpc.compute_control_simple(current_pos, current_vel, target_pos)
```

### 24.5 阻抗控制器 — ImpedanceController

```python
class ImpedanceController:
    def __init__(self, params: ImpedanceParams, control_rate=100.0)
    def compute_torque(self, desired_pos, desired_vel, current_pos, current_vel,
                       wrench: np.ndarray, jacobian: np.ndarray) -> np.ndarray
    def set_impedance_params(self, params: ImpedanceParams)

class ImpedanceParams:
    M: np.ndarray   # 6x6 惯性矩阵
    D: np.ndarray   # 6x6 阻尼矩阵
    K: np.ndarray   # 6x6 刚度矩阵
    @classmethod def default_6d() -> ImpedanceParams
    @classmethod def high_stiffness() -> ImpedanceParams
    @classmethod def low_stiffness() -> ImpedanceParams

class AdmittanceController:
    def __init__(self, M=10.0, D=50.0, K=200.0)
    def update(self, external_force: float, desired_pos: float) -> float
    def reset()

class CollaborativeController:
    def __init__(self, safety_force_limit=100.0, safety_velocity_limit=0.5)
    def check_safety(self, force, velocity) -> (bool, str)
    def get_reaction_torque(self, contact_force, jacobian) -> np.ndarray
```

### 24.6 技能库 — SkillLibrary

```python
class SkillLibrary:
    def list_skills(self) -> List[str]
    def create_skill(self, name: str, params: dict) -> Optional[Skill]
    def execute_skill(self, skill: Skill, context) -> SkillResult

class Skill:
    name: str
    config: SkillConfig
    status: SkillStatus
    def execute(self, context) -> SkillResult
    def cancel()

class SkillStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

### 24.7 任务规划器 — TaskPlanner / HierarchicalPlanner

```python
class TaskPlanner:
    def add_task(self, task: Task)
    def get_next_task(self) -> Optional[Task]
    def plan(self, spec: TaskSpec) -> List[str]
        # 使用 HTN 层次化规划，返回动作序列
    def _htn_plan(self, spec: TaskSpec) -> List[str]
        # HTN 规划核心：任务分解为叶子动作
    def _decompose_and_resolve(self, decompose_fn, goal_state, state,
                                depth, max_depth, primitive_name=None) -> List[str]
        # 递归分解任务为原子动作序列
    def set_world_state(self, state: WorldState)

    # HTN 分解方法 (注册到方法库)
    def _decompose_transport(self, goal_state) -> List[Task]
        # transport → pickup + navigate + place
    def _decompose_pickup(self, goal_state) -> List[Task]
        # pickup → approach + grasp + lift
    def _decompose_place(self, goal_state) -> List[Task]
        # place → move_to + release + retract
    def _decompose_navigate(self, goal_state) -> List[Task]
        # navigate → plan_route + follow_trajectory + reach_target
    def _decompose_inspect(self, goal_state) -> List[Task]
        # inspect → move_to + sense_environment + analyze_data
    def _decompose_open_door(self, goal_state) -> List[Task]
        # open_door → move_to + grasp + pull + move_to
    def _decompose_assemble(self, goal_state) -> List[Task]
        # assemble → fetch + position + fasten
    def _decompose_disassemble(self, goal_state) -> List[Task]
        # disassemble → unfasten + separate + remove

class HierarchicalPlanner(TaskPlanner):
    def decompose_task(self, task: Task, max_depth=3) -> List[Task]
    def plan_hierarchical(self, spec: TaskSpec) -> List[Task]
    def plan_with_replanning(self, spec, initial_state, max_replan_attempts=3)
    def backtrack(self, task, failed_subtasks, attempted_methods=None)
    def estimate_plan_cost(self, tasks) -> float
    def validate_plan(self, tasks, initial_state) -> (bool, str)

@dataclass
class Task:
    id: str
    name: str
    parameters: Dict[str, Any]
    status: TaskStatus
    priority: TaskPriority

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# HTN 规划策略
# 1. 根据任务名选择分解方法
# 2. 递归分解为叶子动作
# 3. 叶子动作不在方法库中则直接返回
# 4. 深度超限则回退到贪心规划
```

### 24.8 轨迹生成器 — TrajectoryGenerator

```python
class TrajectoryGenerator:
    def __init__(self, num_joints: int, config: TrajectoryConfig)
    def generate_quintic_polynomial(self, start, end, duration,
                                    start_vel=None, end_vel=None) -> List[TrajectoryPoint]
    def generate_trapezoidal(self, start, end, max_vel, max_acc) -> (List, float)
    def resample_trajectory(self, waypoints, new_dt) -> List

class RRTPlanner:
    def __init__(self, space_dim: int, bounds, max_iterations=500, step_size=0.1)
    def plan(self, start, goal, obstacle_check,
             algorithm=PlanningAlgorithm.RRT) -> (List, float)
```

### 24.9 ROS2控制接口

```python
class ROS2JointTrajectoryInterface:
    def __init__(self, joint_names: List[str], interface_mode=ControlInterfaceMode.POSITION)
    def activate() / deactivate()
    def send_point(cmd: JointCommand) -> bool
    def send_trajectory(traj: List[JointCommand]) -> bool
    def update(state: JointState) -> Optional[JointCommand]
    def cancel() -> bool
    def get_stats() -> Dict

class ROS2ActionInterface:
    def send_goal(trajectory: List[JointCommand]) -> str  # goal_id
    def update_server(state: JointState) -> bool
    def cancel_goal(goal_id: str) -> bool
    def get_goal_status(goal_id: str) -> ActionGoalStatus

class ROS2ParameterInterface:
    def set_parameter(name: str, value: Any)
    def get_parameter(name: str, default=None) -> Any
    def list_parameters(prefix: str = "") -> List[str]
    def subscribe_parameter_change(name: str, callback)

class ROS2ComponentInterface:
    def configure() -> bool
    def activate() -> bool
    def deactivate() -> bool
    def cleanup()
    def shutdown()
    def get_state() -> str  # "unconfigured"/"inactive"/"active"/"shutdown"
```

### 24.10 控制模块五级规格

| 功能 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 关节位置PID | ✓ | ✓ | ✓ | ✓ | ✓ |
| 关节速度PID | ✓ | ✓ | ✓ | ✓ | ✓ |
| 力矩控制 | - | ✓ | ✓ | ✓ | ✓ |
| 笛卡尔空间控制 | - | ✓ | ✓ | ✓ | ✓ |
| 轨迹跟踪 | - | ✓ | ✓ | ✓ | ✓ |
| 阻抗控制 | - | 基础 | 完整 | 完整+自适应 | 完整+自适应 |
| MPC | - | - | ✓ | ✓ | ✓ |
| 安全限幅 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 碰撞检测 | - | ✓ | ✓ | ✓ | ✓ |
| 紧急停止 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 看门狗监控 | - | - | ✓ | ✓ | ✓ |
| 故障容忍 | - | - | - | ✓ | ✓ |
| 控制频率 | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| ROS2接口 | REST | ROS2 | ROS2 | ROS2 | ROS2 |
| 实时性 | 非实时 | 软实时 | 硬实时 | 硬实时 | 双系统 |

### 24.11 场景理解接口 — SceneUnderstanding

```python
class SceneUnderstanding:
    def __init__(
        self,
        resolution: float = 0.05,           # 栅格分辨率 (m)
        grid_size: Tuple[int, int, int] = (100, 100, 20),
        origin: Optional[np.ndarray] = None,
        use_raycasting: bool = True,
        tracking_window: int = 30
    )
    
    # 占据栅格更新
    def update_from_depth(
        self,
        depth_map: np.ndarray,              # HxW, 深度图 (米)
        intrinsics: np.ndarray,             # 3x3 内参
        extrinsics: Optional[np.ndarray] = None,  # 4x4 外参
        depth_scale: float = 1000.0,       # 深度缩放
        max_depth: float = 10.0
    ) -> OccupancyGrid
    
    def update_from_pointcloud(self, pointcloud: np.ndarray) -> None  # Nx3
    
    # 物体检测与跟踪
    def detect_objects(
        self,
        pointcloud: Optional[np.ndarray] = None,
        use_euclidean_clustering: bool = True,
        cluster_tolerance: float = 0.05,
        min_cluster_size: int = 10
    ) -> List[SceneObject]
    
    def track_objects(
        self,
        detected_objects: List[SceneObject],
        max_distance: float = 0.3
    ) -> List[SceneObject]
    
    def build_scene_graph(
        self,
        objects: List[SceneObject],
        robot_position: np.ndarray
    ) -> SceneGraph
    
    def classify_dynamic_objects(
        self,
        objects: List[SceneObject],
        velocity_threshold: float = 0.05
    ) -> List[int]
    
    # 触觉集成
    def integrate_tactile_contact(
        self,
        objects: List[SceneObject],
        tactile_contact_point: np.ndarray,  # 3, 世界坐标
        contact_force: float,
        sensor_id: str = "default"
    ) -> List[SceneObject]
    
    # 完整状态
    def get_scene_state(
        self,
        robot_pose: np.ndarray,             # 4x4
        robot_velocity: np.ndarray,        # 3
        imu_data: Optional[np.ndarray] = None,
        tactile_data: Optional[Dict] = None,
        force_data: Optional[np.ndarray] = None
    ) -> SceneState
    
    def reset()
```

**核心数据结构:**

```python
@dataclass
class SceneObject:
    object_id: int
    class_id: ObjectClass                   # FLOOR/TABLE/CHAIR/HUMAN/...
    class_name: str
    bounding_box_3d: np.ndarray             # 8x3, 3D包围盒
    centroid_3d: np.ndarray                  # 3, 质心
    pose: np.ndarray                        # 4x4
    velocity: Optional[np.ndarray] = None   # 3, m/s
    confidence: float = 1.0
    tactile_contact: bool = False
    force_reading: Optional[np.ndarray] = None  # 3

@dataclass
class OccupancyGrid:
    resolution: float
    size: Tuple[int, int, int]
    origin: np.ndarray                      # 3
    data: np.ndarray                       # nx*ny*nz, 占据概率 [0,1]
    
    def world_to_grid(self, point: np.ndarray) -> Tuple[int, int, int]
    def grid_to_world(self, gx, gy, gz) -> np.ndarray
    def set_occupied(self, point: np.ndarray, prob: float = 1.0)
    def is_occupied(self, point: np.ndarray, threshold: float = 0.5) -> bool

@dataclass
class SceneGraph:
    objects: List[SceneObject]
    relations: List[SpatialRelation]
    timestamp: float
    frame_id: int
    
    def get_object(self, object_id: int) -> Optional[SceneObject]
    def get_relations(self, object_id: int) -> List[SpatialRelation]

@dataclass
class SceneState:
    scene_graph: SceneGraph
    occupancy: OccupancyGrid
    robot_pose: np.ndarray                  # 4x4
    robot_velocity: np.ndarray              # 3
    imu_data: Optional[np.ndarray]
    tactile_data: Optional[Dict]
    force_data: Optional[np.ndarray]
    dynamic_objects: List[int]
    timestamp: float
    frame_id: int
```

**AGV五级场景理解规格:**

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 分辨率 (m) | 0.10 | 0.05 | 0.02 | 0.01 | 0.005 |
| 感知范围 (m) | 3.0 | 5.0 | 8.0 | 10.0 | 15.0 |
| 最大物体数 | 10 | 30 | 50 | 100 | 200 |
| 物体跟踪 | ✗ | ✓ | ✓ | ✓ | ✓ |
| 语义分割 | ✗ | ✗ | ✓ | ✓ | ✓ |

**使用示例:**
```python
scene = SceneUnderstanding(resolution=0.05, grid_size=(40, 40, 10))

# 从深度图更新
occupancy = scene.update_from_depth(depth_map, intrinsics)

# 检测物体
objects = scene.detect_objects(pointcloud)

# 跟踪
tracked = scene.track_objects(objects)

# 构建场景图
graph = scene.build_scene_graph(tracked, robot_position)

# 获取完整状态
state = scene.get_scene_state(robot_pose, robot_velocity,
                              imu_data=imu, force_data=wrench)
```

---

### 24.12 仿真环境接口 — RobotSimulator / SensorSimulator

```python
class RobotSimulator:
    def __init__(self, config: SimConfig)
    def step(self, torque: np.ndarray) -> Dict  # {'time', 'joint_positions', 'step', ...}
    def set_joint_positions(self, positions: np.ndarray)
    def get_jacobian() -> np.ndarray
    def end_effector_pose -> np.ndarray  # 4x4
    def check_self_collision() -> bool
    def reset()

class SensorSimulator:
    def __init__(self, robot: RobotSimulator, config: SimConfig)
    def get_noisy_joint_positions() -> np.ndarray
    def get_noisy_joint_velocities() -> np.ndarray
    def get_imu_data() -> Dict  # {'accel': (3,), 'gyro': (3,)}
    def get_wrench() -> np.ndarray  # (6,)
    def get_contact_force() -> float
```

**SimConfig:**
```python
@dataclass
class SimConfig:
    dt: float = 0.01
    num_joints: int = 6
    grade: str = "M"          # S/M/L/XL/XXL
    physics_engine: str = "custom"  # "pybullet"/"mujoco"/"dart"/"custom"
    noise_level: float = 0.01
```

---

*文档版本: v1.4.1*
*最后更新: 2026-03-30*

**2026-03-30 v1.4.0**: 新增第24节控制模块接口文档，涵盖AGV运动控制、安全控制、MPC、阻抗控制、技能库、任务规划、轨迹生成、ROS2接口、仿真环境等9个子模块，完善执行控制层接口体系。对应AGV五级控制规格表。

---

## 25. 触觉传感器模块接口 — TactileArray / VirtualTactileSensor

### 25.1 TactileArray

```python
class TactileArray:
    def __init__(
        self,
        array_size: Tuple[int, int] = (16, 16),   # (rows, cols)
        sensor_type: TactileSensorType = TactileSensorType.RESISTIVE,
        sensor_id: str = "tactile_0",
        calibration: Optional[TactileCalibration] = None
    )

    def open(self) -> bool      # 打开传感器
    def close()                  # 关闭传感器
    def capture() -> TactileFrame  # 采集一帧触觉数据

    # 接触检测与分析
    def detect_contacts(self, frame: Optional[TactileFrame] = None) -> List[TactileContact]
    def get_slip_signal(self, frame: Optional[TactileFrame] = None) -> np.ndarray  # (H, W)
    def estimate_grip_quality(self, frame: Optional[TactileFrame] = None) -> Dict[str, float]

    # 标定
    def calibrate(
        self,
        zero_pressure: Optional[np.ndarray] = None,
        known_weights: Optional[List[float]] = None
    )
```

**TactileFrame:**
```python
@dataclass
class TactileFrame:
    pressure_map: np.ndarray              # H×W, 归一化 0-1
    temperature_map: Optional[np.ndarray] # H×W, 摄氏度
    proximity: Optional[np.ndarray]       # H×W, 米 (电容式/光学式)
    slip_signal: Optional[np.ndarray]     # H×W, 滑移信号
    timestamp: float
    frame_id: int
    sensor_id: str
```

**TactileContact:**
```python
@dataclass
class TactileContact:
    center: Tuple[int, int]        # 接触中心 (row, col)
    area: int                     # 接触面积 (像素数)
    peak_pressure: float          # 峰值压力
    mean_pressure: float          # 平均压力
    centroid: Tuple[float, float]  # 压力质心
    contact_force: float         # 估计接触力 (N)
    slip_probability: float       # 滑移概率
    temperature: Optional[float]   # 接触区温度
```

### 25.2 VirtualTactileSensor (仿真环境)

```python
class VirtualTactileSensor:
    def open(self) -> bool
    def close()
    def simulate_contact(
        self,
        contact_pos: Tuple[float, float],  # 归一化 (0-1)
        contact_radius: float = 0.3,
        contact_force: float = 10.0,
        noise_level: float = 0.05
    ) -> TactileFrame
    def simulate_sliding(
        self,
        direction: Tuple[float, float],
        speed: float = 0.1,
        duration_frames: int = 30
    ) -> List[TactileFrame]
```

**AGV五级触觉规格:**

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 阵列尺寸 | 8×8 | 16×16 | 24×24 | 32×32 | 48×48 |
| 分辨率 | 12bit | 12bit | 14bit | 14bit | 16bit |
| 压力范围 | 0-500kPa | 0-1000kPa | 0-2000kPa | 0-5000kPa | 0-10000kPa |
| 采样频率 | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| 温度感知 | ✗ | ✓ | ✓ | ✓ | ✓ |
| 接口类型 | I2C | SPI | USB | USB/ETH | EtherCAT |

---

## 26. 力觉传感器模块接口 — ForceTorqueSensor / VirtualForceSensor

### 26.1 ForceTorqueSensor

```python
class ForceTorqueSensor:
    def __init__(
        self,
        sensor_type: ForceSensorType = ForceSensorType.SIX_AXIS,
        sensor_id: str = "ft_0",
        calibration: Optional[ForceCalibration] = None,
        ip_address: Optional[str] = None,
        ethernet_type: str = "UDP"
    )

    def open(self) -> bool
    def close()
    def capture() -> Wrench             # 采集一帧力数据
    def get_wrench() -> Optional[Wrench] # 获取最新数据

    # 接触检测
    def detect_contact(
        self,
        wrench: Optional[Wrench] = None,
        threshold: Optional[float] = None
    ) -> ContactState

    # 负载估计
    def estimate_payload(self, wrench: Optional[Wrench] = None) -> float

    # 工具坐标系
    def set_tool_center(self, tool_mass: float, tool_com: np.ndarray)

    # 标定
    def calibrate_bias(self, num_samples: int = 100)
```

**Wrench (力旋量):**
```python
@dataclass
class Wrench:
    force: np.ndarray    # 3, (Fx, Fy, Fz), N
    torque: np.ndarray   # 3, (Tx, Ty, Tz), N·m
    timestamp: float
    frame_id: int
    sensor_id: str

    @property
    def magnitude(self) -> float       # 力向量大小
    @property
    def torque_magnitude(self) -> float  # 力矩大小
    def to_vector(self) -> np.ndarray  # [Fx, Fy, Fz, Tx, Ty, Tz]
    def transform(self, rotation, translation) -> 'Wrench'  # 坐标变换
```

**AGV五级力觉规格:**

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 轴数 | 3 | 6 | 6 | 6 | 6 |
| 力范围 | ±100N | ±200N | ±500N | ±1000N | ±5000N |
| 力矩范围 | ±10N·m | ±20N·m | ±50N·m | ±100N·m | ±500N·m |
| 分辨率 | 0.1N | 0.05N | 0.02N | 0.01N | 0.005N |
| 采样频率 | 100Hz | 500Hz | 1000Hz | 2000Hz | 5000Hz |

---

## 27. IMU传感器模块接口 — IMUSensor / VirtualIMUSensor / PoseEstimator

### 27.1 IMUSensor

```python
class IMUSensor:
    def __init__(
        self,
        sensor_type: IMUSensorType = IMUSensorType.BMI088,
        sensor_id: str = "imu_0",
        calibration: Optional[IMUCalibration] = None,
        accel_range: int = 16,     # g
        gyro_range: int = 2000,    # deg/s
        sample_rate: int = 200      # Hz
    )

    def open(self) -> bool
    def close()
    def capture() -> IMUFrame    # 采集一帧IMU数据
    def self_test() -> bool      # 传感器自检

    # 标定
    def calibrate_gyro_bias(self, num_samples: int = 500, duration_sec: float = 5.0)
    def calibrate_accel(self, known_orientation: str = "level")
```

**IMUFrame:**
```python
@dataclass
class IMUFrame:
    accel: np.ndarray          # 3, 加速度 (m/s²)
    gyro: np.ndarray           # 3, 角速度 (rad/s)
    mag: Optional[np.ndarray]  # 3, 磁力计 (μT, 可选)
    temperature: float         # 温度 (摄氏度)
    timestamp: float
    frame_id: int
    sensor_id: str

    @property
    def accel_magnitude(self) -> float
    @property
    def gyro_magnitude(self) -> float
```

### 27.2 PoseEstimator

```python
class PoseEstimator:
    def __init__(
        self,
        algorithm: str = "madgwick",  # "madgwick" / "complementary" / "kalman"
        sample_rate: float = 200.0,
        beta: float = 0.1
    )

    def update(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        mag: Optional[np.ndarray] = None,
        dt: Optional[float] = None
    ) -> Pose

    def get_pose(self) -> Pose
    def get_euler(self) -> np.ndarray    # [roll, pitch, yaw] rad
    def get_rotation_matrix() -> np.ndarray  # 3x3

    # 速度/位置积分 (漂移严重，仅短时有效)
    def integrate_velocity(self, accel, dt, remove_gravity=True)
    def reset()
```

**Pose:**
```python
@dataclass
class Pose:
    position: np.ndarray      # 3, 位置 (m)
    orientation: np.ndarray   # 4, 四元数 (qw, qx, qy, qz)

    def to_euler(self) -> np.ndarray    # [roll, pitch, yaw] rad
    def to_matrix(self) -> np.ndarray   # 4x4 变换矩阵
    @classmethod
    def identity(cls) -> 'Pose'
    @classmethod
    def from_euler(cls, position, rpy) -> 'Pose'
```

### 27.3 VirtualIMUSensor (仿真环境)

```python
class VirtualIMUSensor:
    def open(self) -> bool
    def close()

    def simulate_static(
        self,
        orientation: Tuple[float, float, float] = (0., 0., 0.)  # roll, pitch, yaw rad
    ) -> IMUFrame

    def simulate_motion(
        self,
        linear_accel: Tuple[float, float, float],
        angular_vel: Tuple[float, float, float],
        dt: float = 0.01
    ) -> IMUFrame

    def simulate_trajectory(
        self,
        trajectory_type: str = "circle",  # "circle" / "figure8" / "linear" / "sine"
        duration_s: float = 2.0,
        dt: float = 0.01
    ) -> List[IMUFrame]
```

**AGV五级IMU规格:**

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 型号 | MPU6050 | BMI088 | BMI088 | ADIS16470 | ADIS16470 |
| 加速度量程 | ±8g | ±16g | ±24g | ±40g | ±80g |
| 陀螺量程 | ±1000°/s | ±2000°/s | ±4000°/s | ±4000°/s | ±8000°/s |
| 采样频率 | 100Hz | 200Hz | 500Hz | 1000Hz | 2000Hz |
| 噪声密度 | 400μg/√Hz | 120μg/√Hz | 60μg/√Hz | 20μg/√Hz | 10μg/√Hz |

---

*文档版本: v1.5.0*
*最后更新: 2026-03-30*

**2026-03-30 v1.5.0**: 新增第25-27节触觉/力觉/IMU传感器模块完整接口文档，包含 TactileArray、ForceTorqueSensor、IMUSensor、PoseEstimator、VirtualTactileSensor、VirtualForceSensor、VirtualIMUSensor 的类接口、数据结构、AGV五级规格表。可直接用于接口对照和代码生成。

---

## 第28节: 多智能体协调控制 (Multi-Agent Coordination)

### 28.1 MultiAgentCoordinator

多AGV协同控制与编队管理，支持 L/XL/XXL 等级。

```python
class MultiAgentCoordinator:
    def __init__(
        self,
        communication_range: float = 10.0,  # m
        safety_distance: float = 0.5,        # m
        max_agents: int = 20
    )

    # 智能体管理
    def register_agent(
        self,
        agent_id: str,
        initial_position: np.ndarray,       # (x, y) or (x, y, theta)
        leader_id: Optional[str] = None
    ) -> bool

    def unregister_agent(self, agent_id: str)

    # 编队管理
    def create_formation(
        self,
        formation_id: str,
        formation_type: FormationType,      # LINE / TRIANGLE / SQUARE / CIRCLE / V_SHAPE / GRID / FREE
        target_position: np.ndarray,
        target_heading: float = 0.0,
        formation_size: Optional[int] = None
    ) -> CoordinationTask

    def compute_formation_target(
        self,
        agent_id: str,
        leader_position: np.ndarray,
        leader_heading: float
    ) -> np.ndarray

    def get_formation_center(self, formation_id: str) -> np.ndarray

    # 碰撞检测与避障
    def detect_collisions(self) -> List[CollisionRisk]
    def resolve_collisions(self) -> Dict[str, np.ndarray]  # agent_id -> velocity_correction

    # 任务分配
    def assign_tasks(self, tasks: List[Tuple[str, np.ndarray]])  # (task_id, task_position)

    # 主循环
    def step(self, dt: float)

    def get_status(self) -> Dict
```

**核心数据类型:**

```python
class FormationType(Enum):
    LINE = "line"           # 线性队列
    TRIANGLE = "triangle"   # 三角阵型
    SQUARE = "square"       # 方形阵型
    CIRCLE = "circle"       # 圆形阵型
    V_SHAPE = "v_shape"     # V字形
    GRID = "grid"           # 网格阵型
    FREE = "free"           # 自由分布

class CoordinationState(Enum):
    IDLE = "idle"
    FORMING = "forming"
    FORMING_COMPLETE = "formed"
    NAVIGATING = "navigating"
    REFORMING = "reforming"
    DISBANDING = "disbanding"
    DISBANDED = "disbanded"
    EMERGENCY = "emergency"

@dataclass
class AgentState:
    agent_id: str
    position: np.ndarray      # (x, y) or (x, y, theta)
    velocity: np.ndarray
    target: Optional[np.ndarray]
    leader_id: Optional[str]
    neighbors: List[str]
    in_formation: bool
    formation_slot: Optional[int]
    state: CoordinationState
    battery_level: float      # 0-1
    task_id: Optional[str]

@dataclass
class FormationSlot:
    slot_id: int
    relative_position: np.ndarray  # 相对于队长的位置
    tolerance: float = 0.1          # m, 到达容忍度
    assigned_agent: Optional[str]

@dataclass
class CollisionRisk:
    agent_a: str
    agent_b: str
    distance: float
    time_to_collision: float  # s
    severity: str            # "low" / "medium" / "high" / "critical"
```

**使用示例:**
```python
from control.multi_agent import MultiAgentCoordinator, FormationType

coord = MultiAgentCoordinator(communication_range=10.0, safety_distance=0.5)

# 注册多个AGV
coord.register_agent("agv_0", np.array([0.0, 0.0]))
coord.register_agent("agv_1", np.array([1.0, 0.0]))
coord.register_agent("agv_2", np.array([2.0, 0.0]))
coord.register_agent("agv_3", np.array([3.0, 0.0]))

# 创建V字形编队
coord.create_formation("v_formation", FormationType.V_SHAPE, np.array([0.0, 0.0]))

# 协调主循环
risks = coord.detect_collisions()
corrections = coord.resolve_collisions()
coord.step(dt=0.01)

# 任务分配
coord.assign_tasks([
    ("task_1", np.array([5.0, 0.0])),
    ("task_2", np.array([6.0, 1.0])),
])
```

**AGV五级协调能力规格:**

| 功能 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 多AGV协同 | ✗ | ✗ | ✓ (≤4台) | ✓ (≤10台) | ✓ (≤20台) |
| 编队控制 | ✗ | ✗ | ✓ | ✓ | ✓ |
| 碰撞检测 | ✗ | ✗ | 反应式 | 预测式 | 最优避障 |
| 编队类型 | - | - | LINE/CIRCLE | 全部 | 全部 |
| 分布式决策 | - | - | - | ✓ | ✓ |
| 任务分配 | - | - | 最近邻 | 拍卖算法 | 分布式优化 |

*文档版本: v1.6.0*
*最后更新: 2026-03-30*

**2026-03-30 v1.6.0**: 新增第28节多智能体协调控制模块 (multi_agent.py)，包含 MultiAgentCoordinator 编队控制、CollisionRisk 碰撞检测、FormationType 编队类型、分布式任务分配，支持 L/XL/XXL 三级协调能力。

---

## 第29节: 完整传感器-控制集成流水线

### 29.1 端到端传感器融合流水线

```python
"""
SuperModel 完整传感器-控制集成流水线
=====================================

本节描述从多模态传感数据采集到运动控制的完整闭环流程。
适用于 S/M/L/XL/XXL 所有AGV等级。
"""

from sensors.vision import BinocularCamera, DepthProcessor
from sensors.audio import BinauralMic, SoundLocalizer
from sensors.tactile import TactileArray, PressureProcessor
from sensors.force import ForceTorqueSensor, WrenchProcessor
from sensors.imu import IMUSensor, PoseEstimator
from sensors.manager import SensorManager, SensorManagerConfig
from perception.cross_modal_fusion import CrossModalFusion, FusionConfig, MultimodalInput
from control.motion import MotionController, ControlMode
from control.impedance import ImpedanceController, ImpedanceParams
from control.safety_controller import SafetyController, SafetyConfig, SafetyLevel
from simulation.environment import RobotSimulator, SimConfig


class SuperModelPipeline:
    """
    SuperModel 完整流水线
    
    整合所有传感器模块、融合网络和控制模块，
    实现从感知到控制的完整闭环。
    
    使用示例:
        pipeline = SuperModelPipeline(grade='M')
        pipeline.open_all()
        
        for step in range(1000):
            # 1. 采集多模态传感器数据
            sensor_data = pipeline.capture_all()
            
            # 2. 跨模态融合
            fused = pipeline.fuse(sensor_data)
            
            # 3. 认知决策
            action = pipeline.decide(fused)
            
            # 4. 安全检查
            safe_action = pipeline.safety_check(action, sensor_data)
            
            # 5. 执行控制
            result = pipeline.execute(safe_action)
            
            # 6. 仿真推进
            pipeline.step(dt=0.01)
        
        pipeline.close_all()
    """
    
    def __init__(
        self,
        grade: str = 'M',
        enable_vision: bool = True,
        enable_audio: bool = True,
        enable_tactile: bool = True,
        enable_force: bool = True,
        enable_imu: bool = True,
        fusion_strategy: str = 'hybrid',
        safety_level: str = 'standard'
    ):
        self.grade = grade
        self.enable_vision = enable_vision
        self.enable_audio = enable_audio
        self.enable_tactile = enable_tactile
        self.enable_force = enable_force
        self.enable_imu = enable_imu
        
        # 传感器管理器
        self.sensor_manager = SensorManager(
            config=SensorManagerConfig(grade=grade)
        )
        
        # 融合网络
        self.fusion_config = FusionConfig(
            strategy=FusionStrategy[fusion_strategy.upper()]
        )
        self.fusion = CrossModalFusion(self.fusion_config)
        
        # 控制模块
        self.motion_ctrl = MotionController(num_joints=6)
        self.impedance_ctrl = ImpedanceController(ImpedanceParams.default_6d())
        
        # 安全控制器
        self.safety = SafetyController(
            config=SafetyConfig(safety_level=SafetyLevel[safety_level.upper()])
        )
        
        # 仿真环境
        self.sim = RobotSimulator(SimConfig(num_joints=6, dt=0.01))
        
        self._is_opened = False
    
    def open_all(self):
        """打开所有模块"""
        self.sensor_manager.open_all()
        self.fusion.eval()  # 推理模式
        self._is_opened = True
    
    def close_all(self):
        """关闭所有模块"""
        self.sensor_manager.close_all()
        self._is_opened = False
    
    def capture_all(self) -> dict:
        """采集所有启用传感器的数据"""
        return self.sensor_manager.capture_all()
    
    def fuse(self, sensor_data: dict):
        """跨模态融合"""
        import torch
        
        # 构建 MultimodalInput
        multimodal = MultimodalInput()
        
        if self.enable_vision and 'stereo_frame' in sensor_data:
            frame = sensor_data['stereo_frame']
            # 简化: 使用深度图统计特征
            depth = self.sensor_manager.depth_proc.filter_depth(
                frame.disparity_map, min_dist=0.1, max_dist=10.0
            )
            feat = torch.from_numpy(depth.flatten()[:512]).float().unsqueeze(0)
            multimodal.vision = feat
        
        if self.enable_audio and 'audio_frame' in sensor_data:
            frame = sensor_data['audio_frame']
            feat = torch.from_numpy(frame.left_channel[:128]).float().unsqueeze(0)
            multimodal.audio = feat
        
        if self.enable_tactile and 'tactile_frame' in sensor_data:
            frame = sensor_data['tactile_frame']
            feat = torch.from_numpy(frame.pressure_map.flatten()[:64]).float().unsqueeze(0)
            multimodal.tactile = feat
        
        if self.enable_force and 'wrench' in sensor_data:
            wrench = sensor_data['wrench']
            feat = torch.from_numpy(wrench.to_vector()[:32]).float().unsqueeze(0)
            multimodal.force = feat
        
        if self.enable_imu and 'imu_frame' in sensor_data:
            frame = sensor_data['imu_frame']
            feat = torch.cat([
                torch.from_numpy(frame.accel),
                torch.from_numpy(frame.gyro)
            ]).float().unsqueeze(0)
            multimodal.imu = feat
        
        # 融合前向传播
        with torch.no_grad():
            fused = self.fusion(multimodal)
        
        return fused
    
    def decide(self, fused_features):
        """基于融合特征进行认知决策"""
        # 简化: 返回零动作
        return np.zeros(6)
    
    def safety_check(self, action, sensor_data):
        """安全检查"""
        from control.safety_controller import JointStateSnapshot
        
        # 构建关节状态快照
        if 'joint_pos' in sensor_data:
            snapshot = JointStateSnapshot(
                joint_positions=sensor_data['joint_pos'],
                joint_velocities=sensor_data.get('joint_vel', np.zeros(6)),
                joint_torques=sensor_data.get('joint_tor', np.zeros(6)),
                external_wrenches=sensor_data.get('wrench', None),
                timestamp=sensor_data.get('timestamp', 0.0)
            )
            result = self.safety.check(snapshot)
            if not result.is_safe:
                return np.zeros(6)  # 紧急停止
        
        return action
    
    def execute(self, action):
        """执行控制"""
        return self.motion_ctrl.compute_joint_torque(target_position=action)
    
    def step(self, dt: float = 0.01):
        """仿真步进"""
        return self.sim.step(np.zeros(6))
    
    def __enter__(self):
        self.open_all()
        return self
    
    def __exit__(self, *args):
        self.close_all()


class MultimodalDataLogger:
    """
    多模态数据记录器
    
    用于记录和回放传感器数据，
    支持离线分析和仿真。
    """
    
    def __init__(self, save_dir: str = "./logs"):
        self.save_dir = save_dir
        self.episodes = []
        self.current_episode = []
    
    def start_episode(self, episode_id: str):
        """开始记录一个episode"""
        self.current_episode = []
        self.episode_id = episode_id
    
    def log_frame(self, sensor_data: dict):
        """记录一帧传感器数据"""
        import time
        self.current_episode.append({
            'timestamp': time.time(),
            'data': sensor_data
        })
    
    def end_episode(self):
        """结束并保存episode"""
        self.episodes.append({
            'id': self.episode_id,
            'frames': self.current_episode,
            'num_frames': len(self.current_episode)
        })
        self.current_episode = []


class SensorImuFusionFilter:
    """
    传感器-IMU紧耦合滤波
    
    将视觉里程计、IMU、编码器数据进行紧耦合融合，
    提供高精度位姿估计。
    
    方法:
    - 扩展卡尔曼滤波 (EKF)
    - 滑窗优化 (Sliding Window Optimization)
    - 因子图优化 (Factor Graph)
    """
    
    def __init__(
        self,
        method: str = 'ekf',
        state_dim: int = 15,  # 位置(3) + 速度(3) + 姿态(4) + 偏置(6)
        process_noise: np.ndarray = None,
        measurement_noise: np.ndarray = None
    ):
        self.method = method
        self.state_dim = state_dim
        
        # 默认噪声参数 (AGV-M级)
        if process_noise is None:
            self.process_noise = np.diag([0.01]*3 + [0.01]*3 + [0.001]*4 + [0.0001]*6)
        if measurement_noise is None:
            self.measurement_noise = np.diag([0.05]*3 + [0.01]*4)
        
        self.state = np.zeros(state_dim)
        self.covariance = np.eye(state_dim)
        self._initialized = False
    
    def initialize(self, initial_pose: np.ndarray):
        """初始化滤波器"""
        self.state[:3] = initial_pose[:3]
        self.state[6:10] = initial_pose[3:7]  # 四元数
        self._initialized = True
    
    def predict(self, imu_data: np.ndarray, dt: float):
        """IMU预测步骤"""
        if not self._initialized:
            return
        
        # 状态预测 (简化的运动模型)
        self.state[0:3] += self.state[3:6] * dt  # 位置更新
        # ... 完整实现需要IMU运动学积分
    
    def update(self, measurement: np.ndarray, measurement_type: str):
        """测量更新步骤"""
        if not self._initialized:
            return
        
        if measurement_type == 'vision':
            # 视觉里程计测量
            pass
        elif measurement_type == 'encoder':
            # 编码器测量
            pass
        # ... 卡尔曼更新公式
    
    def get_pose(self) -> np.ndarray:
        """获取当前位姿"""
        return self.state.copy()


# AGV五级流水线配置模板

AGV_PIPELINE_CONFIGS = {
    'S': {
        'enable_vision': True,
        'enable_audio': True,
        'enable_tactile': False,
        'enable_force': False,
        'enable_imu': True,
        'fusion_strategy': 'late',
        'safety_level': 'standard',
        'control_rate': 50,  # Hz
    },
    'M': {
        'enable_vision': True,
        'enable_audio': True,
        'enable_tactile': True,
        'enable_force': True,
        'enable_imu': True,
        'fusion_strategy': 'hybrid',
        'safety_level': 'standard',
        'control_rate': 100,
    },
    'L': {
        'enable_vision': True,
        'enable_audio': True,
        'enable_tactile': True,
        'enable_force': True,
        'enable_imu': True,
        'fusion_strategy': 'hybrid',
        'safety_level': 'enhanced',
        'control_rate': 200,
    },
    'XL': {
        'enable_vision': True,
        'enable_audio': True,
        'enable_tactile': True,
        'enable_force': True,
        'enable_imu': True,
        'fusion_strategy': 'early',
        'safety_level': 'enhanced',
        'control_rate': 500,
    },
    'XXL': {
        'enable_vision': True,
        'enable_audio': True,
        'enable_tactile': True,
        'enable_force': True,
        'enable_imu': True,
        'fusion_strategy': 'early',
        'safety_level': 'maximum',
        'control_rate': 1000,
    },
}


def get_pipeline_config(grade: str) -> dict:
    """获取AGV指定等级的流水线配置"""
    return AGV_PIPELINE_CONFIGS.get(grade, AGV_PIPELINE_CONFIGS['M'])


# 使用示例

"""
# 1. 创建流水线
config = get_pipeline_config('M')
pipeline = SuperModelPipeline(**config)

# 2. 打开所有模块
pipeline.open_all()

# 3. 主循环
for episode in range(10):
    pipeline.start_episode(f"episode_{episode}")
    
    for step in range(1000):
        # 采集
        sensor_data = pipeline.capture_all()
        
        # 融合
        fused = pipeline.fuse(sensor_data)
        
        # 决策
        action = pipeline.decide(fused)
        
        # 安全检查
        safe_action = pipeline.safety_check(action, sensor_data)
        
        # 执行
        torques = pipeline.execute(safe_action)
        
        # 仿真步进
        state = pipeline.step(dt=0.01)
        
        # 记录
        pipeline.log_frame(sensor_data)
    
    pipeline.end_episode()

# 4. 关闭
pipeline.close_all()
"""

*文档版本: v1.10.0*
*最后更新: 2026-03-31*

---

## 30. 避障模块接口 (Obstacle Avoidance)

### 30.1 障碍物表示 — Obstacle

```python
@dataclass
class Obstacle:
    position: np.ndarray      # 2, 世界坐标系 (x, y) m
    radius: float             # 障碍物半径 m
    velocity: np.ndarray = None  # 2, 速度 m/s
    type: str = "static"     # "static" | "dynamic"
    
    def predict_position(self, dt: float) -> np.ndarray  # 预测位置
```

### 30.2 速度指令 — VelocityCommand

```python
@dataclass
class VelocityCommand:
    vx: float         # x方向线速度 m/s
    vy: float         # y方向线速度 m/s
    omega: float      # 角速度 rad/s
    score: float = 0.0  # 轨迹评分
    
    def to_array(self) -> np.ndarray  # [vx, vy, omega]
```

### 30.3 动态窗口法 — DynamicWindowApproach

```python
class DynamicWindowApproach:
    def __init__(self, config: Optional[DWAConfig] = None)
    
    def compute_velocities(
        self,
        robot_pose: np.ndarray,      # (x, y, theta) m, rad
        robot_velocity: np.ndarray,  # (vx, vy, omega) m/s, rad/s
        goal: np.ndarray,           # (gx, gy) m
        obstacles: List[Obstacle],
        dt: float = 0.1
    ) -> VelocityCommand
```

**DWAConfig 参数:**
```python
@dataclass
class DWAConfig:
    max_linear_speed: float = 1.0      # 最大线速度 m/s
    max_angular_speed: float = 2.0     # 最大角速度 rad/s
    max_linear_accel: float = 2.0      # 最大线加速度 m/s^2
    max_angular_accel: float = 3.0    # 最大角加速度 rad/s^2
    vx_resolution: float = 0.05        # 线速度分辨率 m/s
    omega_resolution: float = 0.1     # 角速度分辨率 rad/s
    prediction_horizon: float = 2.0   # 预测时间窗口 s
    robot_radius: float = 0.3          # 机器人半径 m
    obstacle_margin: float = 0.1       # 障碍物裕度 m
```

### 30.4 人工势场法 — ArtificialPotentialField

```python
class ArtificialPotentialField:
    def __init__(self, config: Optional[APFConfig] = None)
    
    def compute_force(
        self,
        robot_pose: np.ndarray,      # (x, y) m
        robot_velocity: np.ndarray,  # (vx, vy) m/s
        goal: np.ndarray,           # (gx, gy) m
        obstacles: List[Obstacle]
    ) -> np.ndarray  # 力向量 (fx, fy)
    
    def compute_velocity(
        self, robot_pose, robot_velocity, goal, obstacles, max_speed=1.0
    ) -> np.ndarray  # 速度向量
```

**APFConfig 参数:**
```python
@dataclass
class APFConfig:
    attract_gain: float = 5.0       # 吸引增益
    goal_tolerance: float = 0.1    # 目标容差 m
    repel_gain: float = 100.0       # 排斥增益
    repel_range: float = 2.0        # 排斥场作用范围 m
    robot_radius: float = 0.3       # 机器人半径 m
    escape_gain: float = 2.0        # 局部最小逃脱增益
```

### 30.5 向量场直方图 — VectorFieldHistogram

```python
class VectorFieldHistogram:
    def __init__(self, config: Optional[VFHConfig] = None)
    
    def compute_direction(
        self,
        robot_pose: np.ndarray,      # (x, y, theta)
        robot_velocity: np.ndarray,  # (vx, vy, omega)
        goal: np.ndarray,           # (gx, gy)
        obstacles: List[Obstacle],
        dt: float = 0.1
    ) -> Tuple[float, VelocityCommand]  # (steering_angle, cmd)
```

### 30.6 综合避障控制器 — ObstacleAvoider

```python
class ObstacleAvoider:
    def __init__(self, config: Optional[AvoidanceConfig] = None)
    
    def compute_command(
        self,
        robot_pose: np.ndarray,
        robot_velocity: np.ndarray,
        goal: np.ndarray,
        obstacles: List[Obstacle],
        dt: float = 0.1
    ) -> VelocityCommand
    
    def set_strategy(self, strategy: AvoidanceStrategy)
    
    @staticmethod
    def create_from_grade(grade: str) -> 'ObstacleAvoider'
```

**使用示例:**
```python
# 创建避障器
avoider = ObstacleAvoider.create_from_grade("L")

# 定义障碍物
obstacles = [
    Obstacle(position=np.array([2.0, 1.0]), radius=0.5),
    Obstacle(position=np.array([3.5, 0.0]), radius=0.3, type="dynamic",
             velocity=np.array([0.2, 0.0])),
]

# 主循环
while not reached_goal:
    cmd = avoider.compute_command(robot_pose, velocity, goal, obstacles)
    robot_pose = integrate(robot_pose, cmd, dt)
```

---

## 31. 仿真-避障集成

### 31.1 仿真环境集成

```python
from simulation.environment import RobotSimulator
from control.obstacle_avoidance import ObstacleAvoider

# 创建仿真和避障
sim = RobotSimulator()
avoider = ObstacleAvoider.create_from_grade("L")

# 主循环
for step in range(1000):
    # 获取感知
    obstacles = sim.get_obstacles()  # 障碍物列表
    pose = sim.get_pose()
    velocity = sim.get_velocity()
    
    # 避障规划
    cmd = avoider.compute_command(pose, velocity, goal, obstacles)
    
    # 执行
    sim.send_velocity(cmd.vx, cmd.vy, cmd.omega)
    sim.step(dt=0.01)
```

### 31.2 ROS2集成

```python
# 避障器发布速度指令到 /cmd_vel
# 激光雷达扫描发布到 /scan
# 避障器订阅 /scan 并发布 /cmd_vel
```

---

## 32. AGV等级与避障能力对照

| 等级 | 避障策略 | 最大障碍物数 | 反应时间 | 安全距离 |
|------|---------|------------|---------|---------|
| S | 无 | 0 | - | - |
| M | DWA | 3 | 0.2s | 0.6m |
| L | HYBRID | 10 | 0.1s | 0.5m |
| XL | HYBRID | 25 | 0.05s | 0.4m |
| XXL | HYBRID | 50 | 0.02s | 0.3m |

---

*文档版本: v1.10.0*
*最后更新: 2026-03-31*

**2026-03-31 v1.9.0**: 新增第29节完整传感器-控制集成流水线，包含 SuperModelPipeline 端到端闭环、MultimodalDataLogger 数据记录器、SensorImuFusionFilter 紧耦合滤波、AGV五级流水线配置模板 (S/M/L/XL/XXL)。

---

## 33. 触觉-控制集成实战 (Tactile-Control Integration)

### 33.1 触觉伺服控制

```python
from sensors.tactile import TactileArray, TactileSensorType, TactileFrame
from sensors.force import ForceTorqueSensor, Wrench, ForceSensorType
from control.impedance import ImpedanceController
import numpy as np

class TactileServoController:
    """
    触觉伺服控制器
    
    基于触觉反馈的实时力/位置控制:
    1. 触觉感知接触区域和压力分布
    2. 计算期望抓取力
    3. 阻抗控制维持稳定抓取
    4. 滑移检测和响应
    """
    
    def __init__(
        self,
        target_force: float = 5.0,    # 目标抓取力 (N)
        stiffness: float = 1000.0,    # 刚度 (N/m)
        damping: float = 50.0         # 阻尼 (N·s/m)
    ):
        self.target_force = target_force
        self.controller = ImpedanceController(
            dim=1,
            Kp=stiffness,
            Kd=damping
        )
        
    def compute_control(
        self,
        tactile_frame: TactileFrame,
        wrench: Wrench,
        tactile_contacts: list,
        dt: float = 0.01
    ) -> np.ndarray:
        """
        计算控制输出
        
        Args:
            tactile_frame: 当前触觉帧
            wrench: 当前力旋量
            tactile_contacts: 检测到的接触列表
            dt: 控制周期
            
        Returns:
            控制量 (位置增量或力增量)
        """
        # 当前总接触力
        current_force = wrench.magnitude if wrench else 0.0
        
        # 力误差
        force_error = self.target_force - current_force
        
        # 滑移检测
        slip_detected = False
        for contact in tactile_contacts:
            if contact.slip_probability > 0.5:
                slip_detected = True
                break
        
        # 滑移响应: 增加抓取力
        if slip_detected:
            force_error += 2.0  # 增加2N
        
        # 阻抗控制
        control_output = self.controller.compute(
            error=force_error,
            error_dot=0.0,
            dt=dt
        )
        
        return np.array([control_output])
```

### 33.2 触觉引导的抓取控制

```python
class TactileGuidedGraspController:
    """
    触觉引导抓取控制器
    
    使用触觉传感器引导物体抓取:
    1. 扫描物体表面获取形状信息
    2. 规划抓取点和抓取角度
    3. 渐进式闭合手指直到触觉触发
    4. 评估抓取质量并调整
    """
    
    def __init__(
        self,
        tactile_array: TactileArray,
        grasp_force: float = 8.0,
        approach_speed: float = 0.05
    ):
        self.tactile = tactile_array
        self.grasp_force = grasp_force
        self.approach_speed = approach_speed
        
    def scan_object(
        self,
        approach_direction: np.ndarray,
        num_scan_points: int = 20
    ) -> Dict:
        """
        扫描物体获取形状信息
        
        Returns:
            {'center': (x,y,z), 'normal': (nx,ny,nz), 'contact_area': float}
        """
        pressure_maps = []
        
        for i in range(num_scan_points):
            # 接近一步
            frame = self.tactile.capture()
            contacts = self.tactile.detect_contacts(frame)
            
            if contacts:
                pressure_maps.append(frame.pressure_map)
        
        # 从压力分布估算物体形状
        if pressure_maps:
            avg_pressure = np.mean(pressure_maps, axis=0)
            quality = self.tactile.estimate_grip_quality(pressure_maps[-1])
            return {
                'avg_pressure': avg_pressure,
                'quality': quality,
                'num_contacts': len(contacts)
            }
        return {}
    
    def execute_grasp(
        self,
        target_object_pose: np.ndarray,
        gripper_width: float
    ) -> Tuple[bool, float]:
        """
        执行抓取
        
        Args:
            target_object_pose: 目标物体位姿 (x, y, z, roll, pitch, yaw)
            gripper_width: 初始夹爪宽度
            
        Returns:
            (success, final_grip_quality)
        """
        # 1. 移动到物体上方
        above_pose = target_object_pose.copy()
        above_pose[2] += 0.05  # 抬升5cm
        
        # 2. 渐进式下降
        current_z = above_pose[2]
        while current_z > target_object_pose[2]:
            current_z -= self.approach_speed
            frame = self.tactile.capture()
            contacts = self.tactile.detect_contacts(frame)
            
            if contacts and any(c.peak_pressure > 0.5 for c in contacts):
                break  # 触觉触发
        
        # 3. 闭合夹爪直到达到目标力
        force_history = []
        while len(force_history) < 100:
            frame = self.tactile.capture()
            contacts = self.tactile.detect_contacts(frame)
            
            if contacts:
                total_force = sum(c.contact_force for c in contacts)
                force_history.append(total_force)
                
                if total_force >= self.grasp_force:
                    quality = self.tactile.estimate_grip_quality(frame)
                    return quality['overall'] > 0.6, quality['overall']
        
        return False, 0.0
```

### 33.3 触觉-控制集成AGV等级配置

| 功能 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 触觉阵列 | 8×8 | 16×16 | 24×24 | 32×32 | 48×48 |
| 采样频率 (Hz) | 50 | 100 | 200 | 500 | 1000 |
| 触觉伺服 | ✗ | ✓ | ✓ | ✓ | ✓ |
| 滑移检测响应 | ✗ | ✓ | ✓ | ✓ | ✓ |
| 抓取质量评估 | ✗ | ✓ | ✓ | ✓ | ✓ |
| 多指协调 | ✗ | ✗ | ✓ | ✓ | ✓ |
| 动态重抓取 | ✗ | ✗ | ✗ | ✓ | ✓ |

---

## 34. 力觉-控制集成实战 (Force-Control Integration)

### 34.1 力控运动基元

```python
from sensors.force import ForceTorqueSensor, Wrench, ForceSensorType
from control.impedance import ImpedanceController
from control.motion import MotionController
import numpy as np

class ForceMotionPrimitive:
    """
    力控运动基元库
    
    包含常用的力控运动模式:
    - 恒力跟踪
    - 力顺应
    - 零力导引
    - 碰撞响应
    """
    
    PRIMITIVES = [
        'constant_force',    # 恒力跟踪
        'force_compliance', # 力顺应
        'zero_force_guide',  # 零力导引
        'collision_response',  # 碰撞响应
        'surface_following',  # 表面跟踪
        'insertion'          # 插孔任务
    ]
    
    def __init__(
        self,
        sensor: ForceTorqueSensor,
        impedance_ctrl: ImpedanceController,
        motion_ctrl: MotionController
    ):
        self.sensor = sensor
        self.impedance = impedance_ctrl
        self.motion = motion_ctrl
        
    def constant_force_tracking(
        self,
        target_force: np.ndarray,
        motion_direction: np.ndarray,
        target_velocity: float = 0.01,
        Kp_force: float = 1.0
    ) -> np.ndarray:
        """
        恒力跟踪
        
        在指定方向上保持恒定接触力
        
        Args:
            target_force: 目标力向量 (Fx, Fy, Fz)
            motion_direction: 运动方向 (归一化)
            target_velocity: 目标运动速度
            Kp_force: 力控制增益
            
        Returns:
            velocity_command: 速度指令
        """
        wrench = self.sensor.capture()
        current_force = wrench.force.copy()
        
        # 力误差
        force_error = target_force - current_force
        
        # 速度修正
        velocity_correction = Kp_force * force_error
        
        # 沿运动方向的速度
        motion_velocity = motion_direction * target_velocity
        
        # 组合
        velocity_cmd = motion_velocity + velocity_correction * 0.1
        
        return velocity_cmd
    
    def surface_following(
        self,
        contact_normal: np.ndarray,
        target_normal_force: float,
        tangential_speed: float = 0.02
    ) -> Tuple[np.ndarray, bool]:
        """
        表面跟踪
        
        沿表面切向移动，保持法向力恒定
        
        Args:
            contact_normal: 表面法向量 (归一化)
            target_normal_force: 目标法向力
            tangential_speed: 切向速度
            
        Returns:
            (velocity_command, is_contact)
        """
        wrench = self.sensor.capture()
        
        # 检测接触
        is_contact = wrench.magnitude > 0.5
        
        # 法向力
        normal_force = np.dot(wrench.force, contact_normal)
        
        # 调整法向速度修正力
        normal_correction = (target_normal_force - normal_force) * 0.5
        
        # 切向速度
        tangent_velocity = np.cross(contact_normal, np.array([0, 0, 1]))
        if np.linalg.norm(tangent_velocity) < 1e-6:
            tangent_velocity = np.cross(contact_normal, np.array([1, 0, 0]))
        tangent_velocity = tangent_velocity / (np.linalg.norm(tangent_velocity) + 1e-6)
        tangent_velocity *= tangential_speed
        
        # 法向修正
        normal_velocity = contact_normal * normal_correction
        
        velocity_cmd = tangent_velocity + normal_velocity
        
        return velocity_cmd, is_contact
    
    def insertion_task(
        self,
        insertion_depth: float,
        target_force: float = 2.0,
        max_force: float = 10.0
    ) -> Tuple[np.ndarray, bool]:
        """
        插孔任务
        
        精密插入操作，力控优先
        
        Returns:
            (velocity_command, success)
        """
        wrench = self.sensor.capture()
        
        # 监测力
        if wrench.magnitude > max_force:
            return np.zeros(3), False  # 过力保护
        
        # 插入方向
        insertion_dir = np.array([0, 0, -1])  # 假设-Z插入
        
        # 速度控制
        if wrench.magnitude < target_force:
            velocity = insertion_dir * 0.005  # 慢速接近
        else:
            velocity = insertion_dir * 0.001  # 精细调整
        
        return velocity, True
```

### 34.2 碰撞检测与响应

```python
class CollisionDetector:
    """
    基于力矩传感器的碰撞检测
    
    方法:
    - 阈值法: 力/力矩超过阈值触发
    - 变化率法: 力矩突变检测
    - 统计法: 基于历史数据的异常检测
    """
    
    def __init__(
        self,
        force_sensor: ForceTorqueSensor,
        torque_threshold: float = 5.0,  # N·m
        force_threshold: float = 20.0    # N
    ):
        self.sensor = force_sensor
        self.torque_threshold = torque_threshold
        self.force_threshold = force_threshold
        self.history = []
        
    def detect(self) -> Tuple[bool, str]:
        """
        碰撞检测
        
        Returns:
            (collision_detected, collision_type)
            collision_type: 'none' / 'collision' / 'obstruction' / 'wall'
        """
        wrench = self.sensor.capture()
        
        # 1. 阈值检测
        if wrench.magnitude > self.force_threshold:
            if wrench.torque_magnitude > self.torque_threshold:
                return True, 'collision'
            return True, 'wall'
        
        # 2. 突变检测
        self.history.append(wrench.to_vector())
        if len(self.history) > 50:
            self.history.pop(0)
            
            # 计算变化率
            recent = np.array(self.history[-10:])
            mean = np.mean(recent, axis=0)
            std = np.std(recent, axis=0)
            
            current = wrench.to_vector()
            deviation = np.abs(current - mean) / (std + 1e-6)
            
            if np.any(deviation > 3.0):  # 3σ 检测
                return True, 'obstruction'
        
        return False, 'none'
    
    def get_collision_direction(self) -> np.ndarray:
        """
        获取碰撞方向向量
        
        用于碰撞后的避让运动
        """
        wrench = self.sensor.capture()
        if wrench.magnitude < 1e-6:
            return np.zeros(3)
        
        # 力方向的反方向
        return -wrench.force / wrench.magnitude

### 34.3 ForceController 力觉控制

```python
from control.force_control import ForceController, ForceControlParams, HybridForcePositionController
import numpy as np

class ForceController:
    """
    力觉控制器 (导纳控制 + 碰撞检测)
    
    基于六维力矩传感器的闭环力控制:
    1. 导纳控制: 将力误差转换为位置调整
    2. 碰撞检测: 检测异常接触力并触发响应
    3. 力限幅: 防止过大接触力损坏物体
    """
    
    def __init__(
        self,
        params: ForceControlParams,
        sensor_id: str = "ft_0"
    ):
        self.params = params
        self.sensor_id = sensor_id
        self._collision_callback = None
        
    def compute_admittance(
        self,
        wrench: np.ndarray,
        target_force: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        导纳控制计算
        
        Args:
            wrench: 当前六维力旋量 (Fx, Fy, Fz, Tx, Ty, Tz)
            target_force: 目标六维力旋量
            dt: 控制周期
            
        Returns:
            velocity_cmd: 速度指令 (vx, vy, vz, wx, wy, wz)
        """
        
    def detect_collision(
        self,
        wrench: np.ndarray,
        history: list
    ) -> Tuple[bool, str]:
        """
        碰撞检测
        
        Returns:
            (collision_detected, collision_type)
            collision_type: 'none' / 'collision' / 'obstruction' / 'wall'
        """
        
    def set_collision_callback(self, callback):
        """设置碰撞响应回调函数"""
        
    def compute_force_control(
        self,
        current_wrench: np.ndarray,
        target_wrench: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        力控计算 (PD控制)
        
        Args:
            current_wrench: 当前力旋量
            target_wrench: 目标力旋量
            dt: 控制周期
            
        Returns:
            force_error_derivative: 力误差变化率
        """

class HybridForcePositionController:
    """
    力位混合控制器
    
    同时控制力和位置 (力控自由度 + 位控自由度的组合):
    1. 在任务空间指定力和位置控制轴
    2. 串联或并联混合控制
    3. 重力补偿和摩擦补偿
    """
    
    def __init__(
        self,
        force_dims: list,      # 力控维度 [0,1,2] = Fx,Fy,Fz
        position_dims: list,   # 位控维度 [3,4,5] = Rx,Ry,Rz
        force_params: ForceControlParams,
        position_params: dict
    ):
        self.force_dims = force_dims
        self.position_dims = position_dims
        
    def compute(
        self,
        current_pose: np.ndarray,
        target_pose: np.ndarray,
        current_wrench: np.ndarray,
        target_wrench: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        力位混合控制计算
        
        Args:
            current_pose: 当前末端位姿 (6D)
            target_pose: 目标位姿
            current_wrench: 当前力旋量
            target_wrench: 目标力旋量
            dt: 控制周期
            
        Returns:
            control_output: 混合控制输出 (6D)
        """
        
    def set_impedance(
        self,
        force_dims_stiffness: dict,
        position_dims_stiffness: dict
    ):
        """设置各维度刚度"""
```

### 34.4 力控AGV等级配置

| 功能 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 力传感器 | 3轴 | 6轴 | 6轴 | 6轴 | 6轴 |
| 采样频率 (Hz) | 100 | 500 | 1000 | 2000 | 5000 |
| 恒力跟踪 | ✗ | ✓ | ✓ | ✓ | ✓ |
| 表面跟踪 | ✗ | ✓ | ✓ | ✓ | ✓ |
| 碰撞检测 | ✗ | ✓ | ✓ | ✓ | ✓ |
| 阻抗控制 | ✗ | ✗ | ✓ | ✓ | ✓ |
| 力反馈遥操作 | ✗ | ✗ | ✓ | ✓ | ✓ |
| 多传感器融合 | ✗ | ✗ | ✗ | ✓ | ✓ |

---

## 35. IMU-控制集成实战 (IMU-Control Integration)

### 35.0 核心接口定义

```python
from control.imu_control import AttitudeStabilizer, IMUControlParams, MotionEstimator
from sensors.imu import IMUSensor, PoseEstimator, IMUSensorType
import numpy as np

class AttitudeStabilizer:
    """
    IMU姿态稳定控制器
    
    使用IMU反馈实现:
    - 姿态保持 (Roll/Pitch/Yaw)
    - 平衡控制 (抗倾倒)
    - 颠簸/振动补偿
    - 抗干扰控制
    """
    
    def __init__(
        self,
        params: IMUControlParams,
        imu: IMUSensor,
        estimator: PoseEstimator
    ):
        self.params = params
        self.imu = imu
        self.estimator = estimator
        
        # 目标姿态
        self.target_euler = np.array([0.0, 0.0, 0.0])
        
    def stabilize(self, dt: float) -> np.ndarray:
        """
        姿态稳定控制计算
        
        Args:
            dt: 控制周期
            
        Returns:
            torque_command: 关节力矩指令 (6D)
        """
        
    def compute_balance_control(
        self,
        disturbance_force: np.ndarray
    ) -> np.ndarray:
        """
        抗干扰平衡控制
        
        Args:
            disturbance_force: 外部扰动力 (Fx, Fy)
            
        Returns:
            torque_command: 关节力矩指令
        """
        
    def compensate_tilt(
        self,
        platform_height: float
    ) -> np.ndarray:
        """
        倾角补偿 (颠簸路面)
        
        Args:
            platform_height: 平台高度 (m)
            
        Returns:
           补偿位移 (3D)
        """

class MotionEstimator:
    """
    基于IMU的运动估计器
    
    功能:
    - 零速更新 (ZUPT)
    - 速度/位置积分
    - IMU里程计
    - 漂移校正
    """
    
    def __init__(
        self,
        imu: IMUSensor,
        estimator: PoseEstimator,
        gravity: np.ndarray = np.array([0, 0, 9.81])
    ):
        self.imu = imu
        self.estimator = estimator
        self.gravity = gravity
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)
        self.is_stationary = False
        
    def update(self, dt: float) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        更新运动估计
        
        Args:
            dt: 时间步长
            
        Returns:
            (velocity, position, is_stationary)
        """
        
    def reset(self):
        """重置估计状态"""
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)
        self.estimator.reset()
```

### 35.1 姿态稳定控制

```python
from sensors.imu import IMUSensor, IMUFrame, Pose, PoseEstimator, IMUSensorType
from control.motion import MotionController
import numpy as np

class AttitudeStabilizer:
    """
    IMU姿态稳定控制器
    
    使用IMU反馈实现:
    - 姿态保持
    - 平衡控制
    - 颠簸补偿
    """
    
    def __init__(
        self,
        imu: IMUSensor,
        estimator: PoseEstimator,
        Kp_roll: float = 5.0,
        Kp_pitch: float = 5.0,
        Kd_gyro: float = 0.5
    ):
        self.imu = imu
        self.estimator = estimator
        self.Kp_roll = Kp_roll
        self.Kp_pitch = Kp_pitch
        self.Kd_gyro = Kd_gyro
        
        # 目标姿态
        self.target_euler = np.array([0.0, 0.0, 0.0])
        
    def compute_balance_control(
        self,
        disturbance_force: np.ndarray
    ) -> np.ndarray:
        """
        抗干扰平衡控制
        
        Args:
            disturbance_force: 外部扰动力 (Fx, Fy)
            
        Returns:
            torque_command: 关节力矩指令
        """
        frame = self.imu.capture()
        pose = self.estimator.update(frame.accel, frame.gyro)
        euler = pose.to_euler()
        
        # 姿态误差
        roll_error = euler[0] - self.target_euler[0]  # roll
        pitch_error = euler[1] - self.target_euler[1]  # pitch
        
        # PD控制
        roll_torque = self.Kp_roll * roll_error + self.Kd_gyro * frame.gyro[0]
        pitch_torque = self.Kp_pitch * pitch_error + self.Kd_gyro * frame.gyro[1]
        
        # 前馈补偿外部扰动
        disturbance_compensation = disturbance_force * 0.1
        
        return np.array([roll_torque, pitch_torque, 0.0]) + disturbance_compensation
    
    def compute_sway_compensation(
        self,
        target_position: np.ndarray,
        current_position: np.ndarray
    ) -> np.ndarray:
        """
        颠簸补偿
        
        在移动平台上补偿地面不平带来的颠簸
        
        Returns:
            height_adjustment: 高度调整量
        """
        frame = self.imu.capture()
        pose = self.estimator.get_pose()
        euler = pose.to_euler()
        
        # 计算姿态角
        roll, pitch = euler[0], euler[1]
        
        # 位置误差
        pos_error = target_position - current_position
        
        # 颠簸补偿: 基于倾斜角调整目标高度
        # 简化: 假设平台高1m, 倾斜角θ对应高度变化≈h*(1-cosθ)
        platform_height = 1.0
        roll_comp = platform_height * (1 - np.cos(roll))
        pitch_comp = platform_height * (1 - np.cos(pitch))
        
        return np.array([0, 0, roll_comp + pitch_comp])
```

### 35.2 运动估计

```python
class MotionEstimator:
    """
    基于IMU的运动估计
    
    功能:
    - 速度积分
    - 位置积分
    - 漂移校正
    """
    
    def __init__(
        self,
        imu: IMUSensor,
        estimator: PoseEstimator,
        gravity: np.ndarray = np.array([0, 0, 9.81])
    ):
        self.imu = imu
        self.estimator = estimator
        self.gravity = gravity
        
        # 状态
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)
        
        # 零速检测
        self.is_stationary = False
        self.stationary_threshold = 0.1  # m/s^2
        
    def update(self, dt: float) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        更新运动估计
        
        Args:
            dt: 时间步长
            
        Returns:
            (velocity, position, is_stationary)
        """
        frame = self.imu.capture()
        
        # 去除重力
        accel = frame.accel - self.gravity
        
        # 检测静止状态 (加速度接近零)
        if np.linalg.norm(frame.accel - self.gravity) < self.stationary_threshold:
            self.is_stationary = True
            # 零速更新
            self.velocity *= 0.9  # 衰减
        else:
            self.is_stationary = False
            # 积分速度
            self.velocity += accel * dt
        
        # 积分位置
        self.position += self.velocity * dt
        
        return self.velocity.copy(), self.position.copy(), self.is_stationary
    
    def reset(self):
        """重置估计"""
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)
        self.estimator.reset()
```

### 35.3 IMU-控制AGV等级配置

| 功能 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| IMU型号 | MPU6050 | BMI088 | BMI088 | ADIS16470 | ADIS16470 |
| 采样频率 (Hz) | 100 | 200 | 500 | 1000 | 2000 |
| 姿态估计 | 互补滤波 | Madgwick | Madgwick | 卡尔曼滤波 | 自适应 |
| 姿态稳定 | ✗ | ✓ | ✓ | ✓ | ✓ |
| 颠簸补偿 | ✗ | ✗ | ✓ | ✓ | ✓ |
| 速度估计 | ✗ | ✓ | ✓ | ✓ | ✓ |
| SLAM融合 | ✗ | ✗ | ✓ | ✓ | ✓ |
| 运动预测 | ✗ | ✗ | ✗ | ✓ | ✓ |

---

## 36. 多传感器-控制联合集成

### 36.1 传感器-控制联合标定

```python
class SensorControlCalibrator:
    """
    传感器-控制联合标定
    
    标定内容:
    1. 触觉传感器到末端执行器的TF
    2. 力传感器到关节坐标系的转换
    3. IMU相对于机器人本体的安装位置/角度
    4. 视觉与机器人坐标系的注册
    """
    
    def __init__(self):
        self.tactile_to_ee = np.eye(4)  # 触觉->末端执行器
        self.force_to_joint = np.eye(4)  # 力传感器->关节坐标系
        self.imu_to_base = np.eye(4)  # IMU->基座
        self.camera_to_base = np.eye(4)  # 相机->基座
        
    def calibrate_tactile_to_ee(
        self,
        tactile: TactileArray,
        ee_position: np.ndarray,  # 末端执行器位置
        contacts: list
    ) -> np.ndarray:
        """
        标定触觉传感器相对于末端执行器的TF
        
        通过已知位置触发触觉传感器，估算相对TF
        """
        if not contacts:
            return self.tactile_to_ee
        
        # 触觉接触位置 (阵列像素坐标)
        contact_pos = np.array([contacts[0].centroid[1], contacts[0].centroid[0], 0])
        
        # 归一化到米
        tactile_scale = 0.001  # 假设1像素=1mm
        contact_pos *= tactile_scale
        
        # 末端执行器位置
        ee_pos = ee_position[:3]
        
        # 估算相对TF (简化: 仅平移)
        self.tactile_to_ee[:3, 3] = ee_pos - contact_pos
        
        return self.tactile_to_ee
    
    def calibrate_imu_to_base(
        self,
        imu: IMUSensor,
        base_pose: Pose,
        num_samples: int = 100
    ) -> np.ndarray:
        """
        标定IMU相对于基座的TF
        
        在已知基座姿态的情况下，采集IMU数据估算安装角度
        """
        frames = [imu.capture() for _ in range(num_samples)]
        
        # 计算平均加速度方向
        avg_accel = np.mean([f.accel for f in frames], axis=0)
        avg_accel = avg_accel / np.linalg.norm(avg_accel)
        
        # 理论重力方向 (基座坐标系)
        base_gravity = np.array([0, 0, -1])  # 假设基座水平
        
        # 计算旋转矩阵
        # R @ avg_accel = base_gravity
        # R = outer(base_gravity, avg_accel) (简化)
        self.imu_to_base[:3, :3] = np.outer(base_gravity, avg_accel)
        
        # 正交化
        U, S, Vt = np.linalg.svd(self.imu_to_base[:3, :3])
        self.imu_to_base[:3, :3] = U @ Vt
        
        return self.imu_to_base
```

### 36.2 统一控制周期管理

```python
class UnifiedControlLoop:
    """
    统一控制循环
    
    多传感器-多控制器协调:
    1. 传感器同步采集
    2. 数据预处理与滤波
    3. 融合与状态估计
    4. 控制器计算
    5. 执行器指令下发
    
    控制周期配置:
    - 高速 (1-2kHz): 电流环/力矩控制
    - 中速 (100-500Hz): 阻抗/位置控制  
    - 低速 (20-50Hz): 视觉伺服/规划
    """
    
    def __init__(
        self,
        sensors: Dict[str, Any],
        controllers: Dict[str, Any],
        control_rates: Dict[str, float]
    ):
        """
        Args:
            sensors: {'vision': cam, 'tactile': tactile, 'force': force, 'imu': imu}
            controllers: {'motion': motion_ctrl, 'impedance': impedance_ctrl}
            control_rates: {'motion': 100, 'impedance': 500, 'vision': 30}
        """
        self.sensors = sensors
        self.controllers = controllers
        self.rates = control_rates
        
        # 相位偏移
        self.phase_offsets = {k: 0.0 for k in control_rates}
        
        # 数据缓冲
        self.latest_data = {}
        
    def spin(self, dt: float):
        """
        主循环迭代
        
        Args:
            dt: 基础控制周期
        """
        # 1. 传感器采集
        self._capture_sensors()
        
        # 2. 控制器计算
        commands = {}
        for name, rate in self.rates.items():
            if name in self.controllers:
                ctrl = self.controllers[name]
                if name == 'motion':
                    commands[name] = ctrl.compute(
                        self.latest_data.get('position'),
                        self.latest_data.get('velocity')
                    )
                elif name == 'impedance':
                    commands[name] = ctrl.compute(
                        self.latest_data.get('wrench'),
                        dt
                    )
        
        # 3. 指令融合/选择
        final_command = self._fuse_commands(commands)
        
        # 4. 下发执行
        self._send_command(final_command)
        
    def _capture_sensors(self):
        """采集所有传感器"""
        if 'vision' in self.sensors:
            self.latest_data['vision'] = self.sensors['vision'].capture()
        if 'tactile' in self.sensors:
            self.latest_data['tactile'] = self.sensors['tactile'].capture()
        if 'force' in self.sensors:
            self.latest_data['force'] = self.sensors['force'].capture()
        if 'imu' in self.sensors:
            self.latest_data['imu'] = self.sensors['imu'].capture()
            
    def _fuse_commands(self, commands: Dict) -> np.ndarray:
        """融合多控制器指令"""
        # 简单优先级融合
        if 'impedance' in commands:
            return commands['impedance']
        return commands.get('motion', np.zeros(6))
```

### 36.3 联合集成AGV等级配置

| 功能 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 控制频率 (Hz) | 50 | 100 | 200 | 500 | 1000 |
| 传感器数量 | 3 | 4 | 5 | 6 | 6+ |
| 联合标定 | ✗ | ✓ | ✓ | ✓ | ✓ |
| 异构周期管理 | ✗ | ✗ | ✓ | ✓ | ✓ |
| 主动感知 | ✗ | ✗ | ✓ | ✓ | ✓ |
| 预测控制 | ✗ | ✗ | ✗ | ✓ | ✓ |
| 自主重标定 | ✗ | ✗ | ✗ | ✗ | ✓ |

---

*文档版本: v1.18.1*
*最后更新: 2026-03-31*

**2026-03-31 v1.18.1**: 补充力控/IMU控制模块接口设计:
- ForceController: 导纳控制 + 碰撞检测接口
- HybridForcePositionController: 力位混合控制接口
- AGV五级控制规格表补充触觉控制(19.6)、力控(19.7)、IMU姿态控制(19.8)

**2026-03-31 v1.18.0**: 新增触觉/力觉/IMU控制模块:
- TactileServoController, GraspQualityController
- ForceController, HybridForcePositionController
- AttitudeStabilizer, MotionEstimator
- sensor_control_integration_tests.py (+23项测试)

**2026-03-31 v1.14.0**: 新增第33-36节传感器-控制集成实战指南，包含:
- 触觉伺服控制和抓取控制 (TactileServoController, TactileGuidedGraspController)
- 力控运动基元库 (ForceMotionPrimitive, CollisionDetector)
- IMU姿态稳定控制和运动估计 (AttitudeStabilizer, MotionEstimator)
- 多传感器-控制联合标定和统一控制周期管理 (SensorControlCalibrator, UnifiedControlLoop)
- 完整AGV五级配置对照表


# 附录：AGV五级控制子系统完整规格对照表 (v1.51.0)

> **补充时间**: 2026-04-05
> **补充内容**: 执行控制子系统 + 感知子系统五级完整对照

## A.1 控制子系统五级规格总表

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| **控制频率 (Hz)** | 50 | 100 | 200 | 500 | 1000 |
| **运动控制算法** | PID | PID+前馈 | 自适应PID | MPC | 鲁棒MPC |
| **轨迹规划** | 线性插值 | S曲线 | RRT | RRT* | 混合A*+RRT |
| **力控能力** | ✗ | 基础力控 | 阻抗控制 | 力位混合 | 自适应阻抗 |
| **触觉反馈** | ✗ | 接触检测 | 滑移检测 | 抓取质量 | 精细操作 |
| **IMU融合** | 基础 | 互补滤波 | EKF | 自适应EKF | 图优化 |
| **安全等级** | PL-a | PL-b | PL-c | PL-d | PL-e |
| **容错能力** | 单点 | 冗余传感 | 多模态 | 预测容错 | 自主重构 |
| **通信接口** | UART | CAN | EtherCAT | TSN | TSN+5G |
| **实时性** | 软实时 | 软实时 | 硬实时 | 硬实时 | 硬实时 |

## A.2 感知-控制闭环延迟规格

| 环节 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| **传感器采集延迟** | 20ms | 10ms | 5ms | 2ms | 1ms |
| **融合处理延迟** | 50ms | 20ms | 10ms | 5ms | 2ms |
| **控制器计算延迟** | 10ms | 5ms | 2ms | 1ms | 0.5ms |
| **通信延迟** | 10ms | 5ms | 2ms | 1ms | 0.5ms |
| **执行器响应** | 20ms | 10ms | 5ms | 2ms | 1ms |
| ****总闭环延迟**** | **110ms** | **50ms** | **24ms** | **11ms** | **5ms** |

## A.3 触觉/力觉/IMU五级配置对照

| 感知模态 | S | M | L | XL | XXL |
|---------|---|---|---|---|-----|
| **触觉阵列** | 8×8 | 16×16 | 24×24 | 32×32 | 48×48 |
| **力传感器** | 3轴 | 6轴基础 | 6轴高精度 | 6轴+指尖 | 多指阵列 |
| **IMU级别** | MPU6050 | MPU9250 | BMI088 | ADIS16470 | 定制光纤 |
| **姿态精度** | ±5° | ±1° | ±0.5° | ±0.1° | ±0.01° |
| **力控精度** | N/A | ±10%FS | ±5%FS | ±2%FS | ±0.5%FS |

---

*附录版本: v1.51.0 | 补充日期: 2026-04-05*

---

## 附录E: AGV五级性能基准表 v1.67.0

> **补充时间**: 2026-04-07
> **补充内容**: 完整端到端性能基准 (感知延迟、融合延迟、控制延迟、通信延迟)

### E.1 感知子系统延迟基准

| 传感器 | S | M | L | XL | XXL |
|--------|---|---|---|---|-----|
| **视觉 (双目)** | 33ms | 33ms | 16ms | 11ms | 8ms |
| **听觉 (双耳)** | 62ms | 62ms | 45ms | 31ms | 22ms |
| **触觉 (阵列)** | 20ms | 10ms | 5ms | 2ms | 1ms |
| **力觉 (六维)** | 10ms | 2ms | 1ms | 0.5ms | 0.2ms |
| **IMU** | 10ms | 5ms | 2ms | 1ms | 0.5ms |

### E.2 端到端闭环延迟

| 场景 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| **简单避障** | 200ms | 100ms | 50ms | 25ms | 10ms |
| **视觉伺服抓取** | 500ms | 250ms | 100ms | 50ms | 20ms |
| **力控插孔** | N/A | 200ms | 100ms | 50ms | 20ms |
| **IMU姿态稳定** | 100ms | 50ms | 25ms | 10ms | 5ms |

### E.3 AGV运动性能

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| **最大线速度** | 0.5m/s | 1.0m/s | 1.5m/s | 2.0m/s | 3.0m/s |
| **最大角速度** | 1.0rad/s | 2.0rad/s | 3.0rad/s | 4.0rad/s | 5.0rad/s |
| **定位精度** | ±50mm | ±20mm | ±10mm | ±5mm | ±1mm |
| **位置精度** | ±5mm | ±2mm | ±1mm | ±0.5mm | ±0.1mm |
| **姿态精度** | ±5° | ±1° | ±0.5° | ±0.1° | ±0.01° |
| **力控精度** | N/A | ±10%FS | ±5%FS | ±2%FS | ±0.5%FS |

### E.4 完整感知-控制闭环延迟 (传感器→融合→控制→执行)

| 环节 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| **传感器采集延迟** | 20ms | 10ms | 5ms | 2ms | 1ms |
| **融合处理延迟** | 50ms | 20ms | 10ms | 5ms | 2ms |
| **控制器计算延迟** | 10ms | 5ms | 2ms | 1ms | 0.5ms |
| **通信延迟** | 10ms | 5ms | 2ms | 1ms | 0.5ms |
| **执行器响应** | 20ms | 10ms | 5ms | 2ms | 1ms |
| **总闭环延迟** | **110ms** | **50ms** | **24ms** | **11ms** | **5ms** |

---

*附录版本: v1.67.0 | 补充日期: 2026-04-07*

*详细性能基准请参考: `docs/design/AGV_FIVE_LEVEL_PERFORMANCE_SPEC.md`*


---

## 附录G: 轨迹规划与跟踪模块接口规范 v1.76.0

> **补充时间**: 2026-04-08
> **补充内容**: 轨迹规划器 (TrajectoryPlanner)、轨迹跟踪器 (PurePursuit/Stanley/PID)、RRT*、最小Snap轨迹生成

### G.1 轨迹规划子系统类图

```
TrajectoryPlanner
├── plan_line(waypoints) → Trajectory
├── plan_arc(waypoints, curvature) → Trajectory
├── plan_path(waypoints) → Trajectory
└── VelocityProfiler
    ├── plan(distance, v0, v1) → (time_pts, vel_pts)
    ├── _trapezoidal(distance, v0, v1)
    └── _s_curve(distance, v0, v1)

TrajectoryTracker (ABC)
├── compute(x, y, theta, traj, t) → (v_ref, omega_ref)
├── PurePursuitTracker
├── StanleyTracker
└── PIDTrajectoryTracker

RRTStarPlanner
├── plan(start, goal, obstacles) → path
└── _nearest/_steer/_collision_free/_rewire

MinimumSnapTrajectory
└── plan(waypoints, dt) → Trajectory
```

### G.2 核心数据结构

#### TrajectoryPoint (轨迹点)
```python
@dataclass
class TrajectoryPoint:
    x: float           # X坐标 (m)
    y: float           # Y坐标 (m)
    theta: float       # 朝向角 (rad)
    v: float           # 切向速度 (m/s)
    t: float           # 时间戳 (s)
    ax: float = 0.0    # X方向加速度 (m/s²)
    ay: float = 0.0    # Y方向加速度 (m/s²)
    omega: float = 0.0 # 角速度 (rad/s)
    curvature: float = 0.0  # 曲率 (1/m)
    a: float = 0.0     # 切向加速度 (m/s²)
```

#### Trajectory (完整轨迹)
```python
@dataclass
class Trajectory:
    points: List[TrajectoryPoint]  # 轨迹点序列
    start_time: float = 0.0        # 开始时间
    total_time: float = 0.0        # 总时长 (s)
    total_length: float = 0.0      # 总长度 (m)

    # 方法
    at_time(t) → TrajectoryPoint    # 时间插值获取轨迹点
    closest_point(x, y) → (point, idx)  # 最近轨迹点
```

#### Waypoint (航点)
```python
@dataclass
class Waypoint:
    x: float       # X坐标 (m)
    y: float       # Y坐标 (m)
    theta: float   # 朝向角 (rad)
    v: float       # 期望速度 (m/s)
    t: float       # 到达时间 (s)
    k: float       # 曲率 (1/m)
```

### G.3 轨迹规划器接口

#### TrajectoryPlanner
| 方法 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `plan_line(start, end)` | Waypoint, Waypoint | List[TrajectoryPoint] | 直线轨迹 |
| `plan_arc(start, end, curvature)` | Waypoint, Waypoint, float | List[TrajectoryPoint] | 圆弧轨迹 |
| `plan_path(waypoints)` | List[Waypoint] | Trajectory | 多路点完整轨迹 |
| `VelocityProfiler.plan(d, v0, v1)` | float, float, float | (np.ndarray, np.ndarray) | 速度曲线 |

#### VelocityProfile 类型
| 枚举值 | 说明 | 适用场景 |
|--------|------|---------|
| `TRAPEZOIDAL` | 梯形速度曲线 | 简单运动、快速加速 |
| `S_CURVE` | S曲线 (恒定jeb) | 平滑运动、无冲击 |
| `POLYNOMIAL` | 多项式曲线 | 最小Snap轨迹 |

### G.4 轨迹跟踪器接口

#### TrajectoryTracker.compute()
```python
def compute(x, y, theta, traj, t) → Tuple[float, float]:
    """
    计算轨迹跟踪控制量

    Args:
        x, y, theta: 当前机器人位姿
        traj: Trajectory 对象
        t: 当前时间 (s)

    Returns:
        (v_ref, omega_ref): 期望线速度 (m/s), 期望角速度 (rad/s)
    """
```

#### PurePursuitTracker (几何前瞻跟踪)
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lookahead` | float | 0.5 | 前瞻距离 (m) |
| `k_vel` | float | 1.0 | 速度增益 |
| `k_angle` | float | 2.0 | 角度增益 |
| `max_omega` | float | 2.0 | 最大角速度 (rad/s) |

#### StanleyTracker (前轴中心跟踪)
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `k_ce` | float | 1.0 | 交叉航向误差增益 |
| `k_v` | float | 0.5 | 速度增益 |
| `softening_epsilon` | float | 0.001 | 软化系数 |
| `max_steer` | float | 0.5 | 最大前轮转角 (rad) |

#### PIDTrajectoryTracker (PID跟踪)
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `kp_v, ki_v, kd_v` | float | 2.0, 0.1, 0.5 | 速度PID参数 |
| `kp_omega, ki_omega, kd_omega` | float | 3.0, 0.2, 0.5 | 角速度PID参数 |
| `max_v` | float | 1.5 | 最大线速度 (m/s) |
| `max_omega` | float | 2.0 | 最大角速度 (rad/s) |

### G.5 RRT* 规划器接口

```python
class RRTStarPlanner:
    def __init__(
        bounds: Tuple[float, float, float, float],  # (xmin, xmax, ymin, ymax)
        max_iter: int = 500,
        step_size: float = 0.3,
        search_radius: float = 0.5,
        goal_sample_rate: float = 0.1
    )

    def plan(
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacles: List[Tuple[float, float, float]] = None  # [(cx, cy, radius), ...]
    ) → List[Tuple[float, float]]
```

### G.6 最小Snap轨迹接口

```python
class MinimumSnapTrajectory:
    def __init__(self, order: int = 7):
        """order: 多项式阶数"""

    def plan(
        waypoints: List[Waypoint],
        dt: float = 0.1
    ) → Trajectory
```

### G.7 使用示例

```python
from control.planner import (
    TrajectoryPlanner, PurePursuitTracker, RRTStarPlanner,
    Waypoint, VelocityProfile, VelocityProfiler
)

# 1. 速度规划
profiler = VelocityProfiler(max_v=1.0, max_a=0.5, profile_type=VelocityProfile.TRAPEZOIDAL)
t_pts, v_pts = profiler.plan(distance=2.0, v0=0.0, v1=0.0)

# 2. 轨迹规划
planner = TrajectoryPlanner(max_v=1.0, max_a=0.5)
waypoints = [
    Waypoint(x=0, y=0, theta=0, v=0),
    Waypoint(x=2, y=1, theta=0.5, v=0.5),
    Waypoint(x=4, y=0, theta=0, v=0),
]
traj = planner.plan_path(waypoints)

# 3. 轨迹跟踪
tracker = PurePursuitTracker(lookahead=0.5)
v_ref, omega_ref = tracker.compute(x=0.5, y=0.1, theta=0.1, traj=traj, t=1.0)

# 4. RRT* 全局规划
rrt = RRTStarPlanner(bounds=(-5, 5, -5, 5), max_iter=300)
path = rrt.plan(start=(0, 0), goal=(3, 4), obstacles=[(1.5, 2.0, 0.5)])
```

### G.8 五级AGV轨迹控制规格

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **轨迹规划算法** | 梯形速度 | 梯形速度 | S曲线 | 最小Snap | 最小Snap |
| **跟踪算法** | PID | PID | PurePursuit | PurePursuit+Stanley | 自适应切换 |
| **最大规划频率** | 10Hz | 20Hz | 50Hz | 100Hz | 200Hz |
| **跟踪周期** | 50ms | 20ms | 10ms | 5ms | 2ms |
| **前瞻距离** | 0.2m | 0.5m | 0.8m | 1.0m | 1.5m |
| **路径平滑度** | 基础 | 中等 | 平滑 | 高平滑 | 最平滑 |
| **避障响应** | 200ms | 100ms | 50ms | 20ms | 10ms |

---

*附录版本: v1.76.0 | 补充日期: 2026-04-08*

---

## 附录H: 导航控制模块接口规范 v1.97.0

> **补充时间**: 2026-04-09
> **补充内容**: NavigationController 全局路径规划 (A*/Dijkstra) + 轨迹跟踪完整接口

### H.1 导航子系统架构

```
NavigationController
├── OccupancyGrid          # 地图管理 (栅格地图 + 障碍物管理)
├── DijkstraPlanner        # Dijkstra最短路径规划
├── AStarPlanner           # A* 启发式路径规划
├── PID 轨迹跟踪           # 位姿误差 PID 控制
└── NavigationState        # 状态机 (IDLE / PLANNING / NAVIGATING / AVOIDING / ARRIVED / FAILED)
```

### H.2 核心数据结构

#### PlannerType (规划器类型枚举)
```python
class PlannerType(Enum):
    DIJKSTRA = "dijkstra"    # Dijkstra 广度优先搜索
    A_STAR = "astar"         # A* 启发式搜索 (默认)
```

#### NavigationState (导航状态枚举)
```python
class NavigationState(Enum):
    IDLE        = "idle"         # 空闲，无任务
    PLANNING    = "planning"     # 正在规划路径
    NAVIGATING  = "navigating"   # 正常导航中
    AVOIDING    = "avoiding"     # 局部避障中
    ARRIVED     = "arrived"      # 到达目标
    FAILED      = "failed"       # 规划失败
```

#### OccupancyGrid (占据栅格地图)
| 方法 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `world_to_grid(wx, wy)` | float, float | (int, int) | 世界坐标→栅格坐标 |
| `grid_to_world(gx, gy)` | int, int | (float, float) | 栅格坐标→世界坐标 |
| `set_obstacle(wx, wy, radius)` | float, float, float | None | 设置障碍物 (圆形) |
| `is_free(wx, wy)` | float, float | bool | 查询是否可通行 |
| `get_nearby_obstacles(wx, wy, radius)` | float, float, float | List[(int,int)] | 获取附近障碍物 |

### H.3 规划器接口

#### DijkstraPlanner (Dijkstra最短路径)
```python
class DijkstraPlanner:
    def __init__(self, grid: OccupancyGrid):
        """初始化 Dijkstra 规划器"""

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[Path]:
        """
        规划从起点到终点的最短路径

        Args:
            start: (x, y) 起点世界坐标
            goal:  (x, y) 终点世界坐标

        Returns:
            Path 对象，或 None (无法到达)
        """
```

#### AStarPlanner (A* 启发式路径规划)
```python
class AStarPlanner(DijkstraPlanner):
    def __init__(
        self,
        grid: OccupancyGrid,
        heuristic_type: str = "euclidean"  # "euclidean" | "manhattan"
    ):
        """初始化 A* 规划器"""

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[Path]:
        """与 DijkstraPlanner 相同接口，默认使用欧几里得启发式"""
```

### H.4 NavigationController 完整接口

```python
class NavigationController:
    """
    AGV 导航控制器

    功能:
    - 全局路径规划 (A* / Dijkstra)
    - PID 轨迹跟踪
    - 地图管理与定位
    - 状态机管理
    """

    def __init__(
        self,
        grid: OccupancyGrid,
        planner_type: PlannerType = PlannerType.A_STAR,
        max_speed: float = 1.0,        # 最大线速度 (m/s)
        max_accel: float = 1.0,       # 最大加速度 (m/s²)
        goal_tolerance: float = 0.1,   # 位置容差 (m)
        angle_tolerance: float = 0.1,  # 角度容差 (rad)
        kp_dist: float = 2.0,          # 距离 PID P 参数
        kp_angle: float = 3.0,          # 角度 PID P 参数
        integral_limit: float = 0.5,    # 积分限幅
    ):

    # === 路径设置 ===
    def set_global_path(self, path: Path) -> None:
        """手动设置全局路径 (绕过规划器)"""

    def plan_to_goal(self, start: np.ndarray, goal: np.ndarray) -> bool:
        """
        从起点规划到终点

        Args:
            start: [x, y, theta] 起始位姿
            goal:  [x, y, theta] 目标位姿

        Returns:
            bool: 规划是否成功
        """

    # === 轨迹跟踪 ===
    def update(self, current_pose: np.ndarray, dt: float) -> np.ndarray:
        """
        导航控制更新 (调用频率 ≥ 10Hz)

        Args:
            current_pose: [x, y, theta] 当前位姿
            dt: 时间步长 (s)

        Returns:
            np.ndarray: [vx, vy, omega] 速度指令
        """

    # === 状态查询 ===
    @property
    def state(self) -> NavigationState:
        """当前导航状态"""

    @property
    def current_path(self) -> Optional[Path]:
        """当前规划路径"""

    @property
    def progress(self) -> float:
        """导航进度 (0.0 ~ 1.0)"""

    # === 控制 ===
    def stop(self) -> None:
        """停止导航，切换到 IDLE"""

    def pause(self) -> None:
        """暂停导航"""

    def resume(self) -> None:
        """恢复导航"""

    def reset(self) -> None:
        """重置所有状态"""
```

### H.5 五级AGV导航规格

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **规划算法** | Dijkstra | A* | A* | A* + 动态重规划 | A* + 增量D* |
| **全局规划周期** | 1Hz | 2Hz | 5Hz | 10Hz | 20Hz |
| **局部跟踪周期** | 10Hz | 20Hz | 50Hz | 100Hz | 200Hz |
| **最大速度** | 0.5m/s | 1.0m/s | 1.5m/s | 2.0m/s | 3.0m/s |
| **定位精度** | ±50mm | ±20mm | ±10mm | ±5mm | ±1mm |
| **路径重规划** | 手动 | 按需 | 按需+定时 | 实时 | 预测式 |
| **地图分辨率** | 0.2m | 0.1m | 0.05m | 0.02m | 0.01m |
| **障碍物检测半径** | 0.5m | 0.3m | 0.2m | 0.15m | 0.1m |
| **安全停车距离** | 1.0m | 0.5m | 0.3m | 0.2m | 0.1m |

### H.6 使用示例

```python
import numpy as np
from control.navigation import (
    NavigationController, OccupancyGrid,
    PlannerType, NavigationState,
    DijkstraPlanner, AStarPlanner
)

# 1. 创建地图
grid = OccupancyGrid(
    width=20.0, height=20.0, resolution=0.1,
    origin=(0.0, 0.0)
)

# 添加静态障碍物
grid.set_obstacle(5.0, 5.0, radius=0.5)
grid.set_obstacle(8.0, 3.0, radius=0.3)
grid.set_obstacle(3.0, 10.0, radius=0.4)

# 2. 直接使用规划器
planner = AStarPlanner(grid)
path = planner.plan(start=(1.0, 1.0), goal=(15.0, 15.0))
if path:
    print(f"路径长度: {path.length:.2f}m, 航点数: {len(path.waypoints)}")

# 3. 使用导航控制器
nav = NavigationController(
    grid=grid,
    planner_type=PlannerType.A_STAR,
    max_speed=1.0,
    goal_tolerance=0.1
)

# 规划路径
start_pose = np.array([1.0, 1.0, 0.0])
goal_pose  = np.array([15.0, 15.0, 0.0])
success = nav.plan_to_goal(start_pose, goal_pose)

if success:
    # 仿真跟踪循环
    current_pose = start_pose.copy()
    dt = 0.1  # 100ms 控制周期

    for step in range(500):
        vel = nav.update(current_pose, dt)
        vx, vy, omega = vel

        # 简易运动学积分 (差速驱动)
        current_pose[0] += vx * np.cos(current_pose[2]) * dt
        current_pose[1] += vx * np.sin(current_pose[2]) * dt
        current_pose[2] += omega * dt

        if nav.state == NavigationState.ARRIVED:
            print(f"到达目标! 步数: {step}, 位置: {current_pose}")
            break
        elif nav.state == NavigationState.FAILED:
            print("路径规划失败!")
            break
```

### H.7 五级导航控制规格速查

| 功能 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| 全局规划器 | Dijkstra | A* | A* | A* | A*+D*Lite |
| 局部避障 | 被动检测 | DWA | DWA+APF | APF+TCM | 预测避障 |
| 轨迹跟踪 | PID | PID | PurePursuit | PurePursuit+Stanley | 自适应MPC |
| 重规划触发 | 手动 | 碰撞后 | 碰撞后 | 预测式 | 实时增量 |
| 多目标导航 | ✗ | ✓ | ✓ | ✓ | ✓ |
| 动态障碍跟踪 | ✗ | ✗ | ✓ | ✓ | ✓ |

---

*附录版本: v1.97.0 | 补充日期: 2026-04-09*
