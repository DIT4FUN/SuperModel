# SuperModel 模块接口设计

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
    def plan(self, spec: TaskSpec) -> List[Action]
    def set_world_state(self, state: WorldState)

class HierarchicalPlanner(TaskPlanner):
    def decompose_task(self, task: Task, max_depth=3) -> List[Task]
    def plan_hierarchical(self, spec: TaskSpec) -> List[Task]

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

### 24.11 仿真环境接口 — RobotSimulator / SensorSimulator

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

*文档版本: v1.4.0*
*最后更新: 2026-03-30*

**2026-03-30 v1.4.0**: 新增第24节控制模块接口文档，涵盖AGV运动控制、安全控制、MPC、阻抗控制、技能库、任务规划、轨迹生成、ROS2接口、仿真环境等9个子模块，完善执行控制层接口体系。对应AGV五级控制规格表。
