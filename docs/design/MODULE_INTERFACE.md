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

---

## 10. 版本历史

| 版本 | 日期 | 描述 |
|------|------|------|
| v0.2.0 | 2026-03-28 | 新增编码器接口章节、AGV五级规格对照 |
| v0.1.0 | 2026-03-28 | 初始接口设计文档 |

*文档版本: v0.2.0*
