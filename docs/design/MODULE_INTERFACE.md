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

## 9. AGV五级规格对照

AGV五级规格体系定义于 [AGV_GRADE_SPEC.md](AGV_GRADE_SPEC.md)，与模块接口的主要对应关系：

| 等级 | 感知接口 | 控制频率 | 融合策略 | 典型平台 |
|------|----------|----------|----------|----------|
| **S** | 3模态 | 50Hz | 晚期融合 | Raspberry Pi 5 |
| **M** | 5模态 | 100Hz | 中期融合 | Jetson Nano |
| **L** | 5模态 | 200Hz | 中期融合 | Jetson Orin Nano |
| **XL** | 5模态+事件相机 | 500Hz | 混合融合 | Jetson AGX Orin |
| **XXL** | 多目+LiDAR | 1000Hz | 早期+中期+晚期 | NVIDIA DRIVE |

各等级对应的规格参数（分辨率、采样率、维度等）详见 `AGV_GRADE_SPEC.md`。

### 9.1 AGV五级功能矩阵

| 功能 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| **传感器驱动** |
| 双目视觉 | ○ | ✅ | ✅ | ✅ | ✅ |
| 双耳声学 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 触觉阵列 | ○ | ✅ | ✅ | ✅ | ✅ |
| 六维力矩 | ○ | ✅ | ✅ | ✅ | ✅ |
| IMU | ✅ | ✅ | ✅ | ✅ | ✅ |
| 接近觉 | ✗ | ○ | ✅ | ✅ | ✅ |
| 事件相机 | ✗ | ✗ | ✗ | ✅ | ✅ |
| **融合网络** |
| 晚期融合 | ✅ | ✅ | ○ | ○ | ○ |
| 中期融合 | ✗ | ✅ | ✅ | ✅ | ○ |
| 早期融合 | ✗ | ✗ | ○ | ✅ | ✅ |
| 混合融合 | ✗ | ○ | ✅ | ✅ | ✅ |
| CrossModalAttention | ✗ | ✅ | ✅ | ✅ | ✅ |
| 统一表示分离 | ✗ | ✅ | ✅ | ✅ | ✅ |
| **认知学习** |
| 世界模型 | ✗ | ○ | ✅ | ✅ | ✅ |
| Dreamer Agent | ✗ | ✗ | ✅ | ✅ | ✅ |
| 对比学习 | ✗ | ✅ | ✅ | ✅ | ✅ |
| 好奇心驱动 | ✗ | ✗ | ✅ | ✅ | ✅ |
| **控制执行** |
| 关节PID控制 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 笛卡尔速度控制 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 阻抗控制 | ✗ | ✅ | ✅ | ✅ | ✅ |
| 力位混合控制 | ✗ | ○ | ✅ | ✅ | ✅ |
| 协作安全控制 | ✗ | ✅ | ✅ | ✅ | ✅ |
| 技能库调度 | ✗ | ✅ | ✅ | ✅ | ✅ |
| HTN任务规划 | ✗ | ○ | ✅ | ✅ | ✅ |
| **仿真环境** |
| 自定义物理仿真 | ✅ | ✅ | ✅ | ✅ | ✅ |
| PyBullet引擎 | ○ | ○ | ✅ | ✅ | ✅ |
| MuJoCo引擎 | ○ | ○ | ✅ | ✅ | ✅ |
| 场景管理器 | ✗ | ✅ | ✅ | ✅ | ✅ |
| 轨迹记录器 | ✗ | ✅ | ✅ | ✅ | ✅ |
| **编码器** |
| CNN视觉编码器 | ✗ | ✅ | ✅ | ✅ | ✅ |
| RNN时序编码器 | ✗ | ✅ | ✅ | ✅ | ✅ |
| Transformer编码器 | ✗ | ○ | ✅ | ✅ | ✅ |
| 多模态统一编码 | ✗ | ✅ | ✅ | ✅ | ✅ |

**图例**: ✅ 支持　○ 可选/简化　✗ 不支持

### 9.2 模块接口依赖关系

```
sensors/
├── vision.py       → sensors/encoders.py → fusion/
├── audio.py       → sensors/encoders.py → fusion/
├── tactile.py     → sensors/encoders.py → fusion/
├── force.py       → sensors/encoders.py → fusion/
└── imu.py        → sensors/encoders.py → fusion/

fusion/
└── cross_modal_fusion.py → perception/ → learning/
                                        → control/planner.py
                                        → control/skill.py

control/
├── motion.py      → simulation/
├── impedance.py   ← sensors/force.py (external_wrench)
├── skill.py       → control/motion.py
└── planner.py    → control/skill.py
```

### 9.3 快速启动示例

```python
# === 完整流程示例 ===
import numpy as np
import torch
from sensors.vision import BinocularCamera
from sensors.audio import BinauralMic
from sensors.tactile import TactileArray
from sensors.force import ForceTorqueSensor
from sensors.imu import IMUSensor, PoseEstimator
from fusion.cross_modal_fusion import CrossModalFusion, FusionConfig, MultimodalInput
from control.motion import MotionController, JointState
from simulation.environment import RobotSimulator, SimConfig

# 1. 初始化传感器
camera = BinocularCamera(resolution=(1280, 720), fps=30)
mic = BinauralMic(sample_rate=16000)
tactile = TactileArray(array_size=(16, 16))
force = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
imu = IMUSensor(sensor_type=IMUSensorType.BMI088)

# 2. 初始化融合网络
config = FusionConfig(vision_dim=512, audio_dim=128, tactile_dim=64, force_dim=32, imu_dim=64)
fusion = CrossModalFusion(config)

# 3. 初始化控制器
controller = MotionController(num_joints=6, control_rate=100.0)

# 4. 主循环
for step in range(1000):
    # 感知
    stereo = camera.capture()
    audio = mic.capture()
    tactile_frame = tactile.capture()
    wrench = force.capture()
    imu_frame = imu.capture()

    # 融合
    multimodal = MultimodalInput(
        vision=torch.randn(1, 512),
        audio=torch.randn(1, 128),
        tactile=torch.randn(1, 64),
        force=torch.randn(1, 32),
        imu=torch.randn(1, 64)
    )
    unified = fusion(multimodal)

    # 控制
    state = JointState(
        position=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        velocity=np.zeros(6),
        torque=np.zeros(6)
    )
    controller.update_joint_state(state)
    torque = controller.compute_joint_torque(target_position=np.array([0.5, 0.3, -0.2, 0.1, 0.0, 0.0]))
```

### 9.4 数据流时序图

```
t=0ms      t=10ms     t=20ms     t=30ms     t=40ms     t=50ms
   │          │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼          ▼
┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
│Camera│  │Mic   │  │Tactil│  │Force │  │ IMU │  │Process│
│Capture│ │Capture│  │Capture│  │Capture│  │Capture│  │& Fuse│
└──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘
   │          │          │          │          │          │
   └──────────┴──────────┴──────────┴──────────┴──────────┘
                              │
                              ▼
                      CrossModalFusion
                         unified
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         World Model     Task Planner    Skill Library
              │               │               │
              ▼               ▼               ▼
         Action           Plan           Skill Execute
              │               │               │
              └───────────────┴───────────────┘
                              │
                              ▼
                    MotionController
                         torque
                              │
                              ▼
                    RobotSimulator.step()
```

---

## 10. 版本历史

| 版本 | 日期 | 描述 |
|------|------|------|
| v0.4.0 | 2026-03-29 | 新增轨迹规划接口、扩展阻抗/导纳控制接口 |
| v0.3.0 | 2026-03-28 | 新增AGV五级功能矩阵、模块依赖关系图、快速启动示例、数据流时序图 |
| v0.2.0 | 2026-03-28 | 新增编码器接口章节、AGV五级规格对照 |
| v0.1.0 | 2026-03-28 | 初始接口设计文档 |

*文档版本: v0.4.0*

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

## 12. 错误处理规范

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

*文档版本: v0.5.0*
*最后更新: 2026-03-29*
