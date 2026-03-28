# SuperModel 系统架构设计

> 版本: v0.2.0 | 更新日期: 2026-03-28

---

## 1. 系统概述

SuperModel 是一个超模态大模型机器人具身智能大脑，采用「构建式学习」范式，融合视觉、听觉、触觉、力觉、IMU五种感知模态，通过跨模态注意力网络实现统一表示学习，并基于世界模型（World Model）实现自主决策与控制。

### 1.1 设计目标

| 目标 | 描述 |
|------|------|
| 多模态感知 | 5+ 感知模态实时融合 |
| 自主学习 | 无需人工标注，自主构建知识 |
| 实时控制 | ≤10ms 端到端延迟 |
| 渐进演化 | 持续学习、自我进化 |
| 可扩展性 | AGV 五级规格 (S→XXL) |

### 1.2 系统层次

```
┌─────────────────────────────────────────────────┐
│                  用户/任务层                       │
│         (任务指令、自然语言、演示学习)               │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│                  认知层                           │
│    (世界模型 / 任务规划 / 技能调度 / 决策)          │
│    WorldModel | TaskPlanner | SkillLibrary       │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│                  融合层                           │
│         (CrossModalFusion / UnifiedRep)          │
│      跨模态注意力网络 | 特征对齐 | 联合表示         │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│                  感知层                           │
│  BinocularCamera | BinauralMic | TactileArray   │
│  ForceTorqueSensor | IMUSensor | EncoderNetwork  │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│                  执行层                           │
│   MotionController | ImpedanceController        │
│   PID控制 | 轨迹规划 | 阻抗控制 | 碰撞检测          │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│                  仿真层                           │
│     RobotSimulator | SensorSimulator            │
│     物理仿真 | 传感器噪声 | 场景管理               │
└─────────────────────────────────────────────────┘
```

---

## 2. 模块详细设计

### 2.1 感知层 (Perception)

#### 2.1.1 视觉 — BinocularCamera

**功能:** 双目立体视觉，提供深度感知和 3D 环境理解

**核心方法:**
```python
class BinocularCamera:
    def capture(self) -> StereoFrame       # 同步采集左右图像
    def get_depth_map(self, frame)        # 深度图估计
    def get_point_cloud(self, frame)       # 点云生成
    def set_exposure(self, value: float)  # 曝光控制
    def get_extrinsics(self) -> StereoExtrinsics  # 基线/外参
```

**输出格式:**
```python
@dataclass
class StereoFrame:
    left_image: np.ndarray       # HxWx3 uint8
    right_image: np.ndarray      # HxWx3 uint8
    timestamp: float
    frame_id: int
    left_cam_info: CameraIntrinsics   # 内参
    stereo_extrinsics: StereoExtrinsics  # 基线
```

**处理流水线:**
```
左相机 Raw → ISP处理 → 校正 → 特征提取
右相机 Raw → ISP处理 → 校正 → 特征提取
                              ↓
                       立体匹配 → 深度图 → 点云
```

#### 2.1.2 听觉 — BinauralMic

**功能:** 双耳声源定位、语音识别、波束形成

**核心方法:**
```python
class BinauralMic:
    def capture(self, duration=0.1) -> AudioFrame
    def localize_sources(self, frame) -> List[SoundSource]
    def enable_beamforming(self, enabled: bool)
```

**输出格式:**
```python
@dataclass
class AudioFrame:
    left_channel: np.ndarray    # N samples
    right_channel: np.ndarray   # N samples
    sample_rate: int            # Hz
```

**处理流水线:**
```
MIC_L → ADC → 带通滤波 → AGC → 特征提取
MIC_R → ADC → 带通滤波 → AGC → 特征提取
                        ↓
              TDOA估计 → 方位角 → 声源定位
              波束形成 → 增强语音
```

#### 2.1.3 触觉 — TactileArray

**功能:** 压力分布感知、接触检测、滑移检测、温度感知

**核心方法:**
```python
class TactileArray:
    def capture(self) -> TactileFrame
    def detect_contacts(self, frame) -> List[TactileContact]
    def get_slip_signal(self, frame) -> np.ndarray
    def calibrate(zero_pressure, known_weights)
```

**输出格式:**
```python
@dataclass
class TactileFrame:
    pressure_map: np.ndarray       # HxW float32 (0-1 归一化)
    temperature_map: np.ndarray    # HxW float32 (°C)
    proximity: Optional[np.ndarray] # HxW float32 (m)
    slip_signal: Optional[np.ndarray]  # HxW float32

@dataclass
class TactileContact:
    center: Tuple[int, int]     # 接触中心像素坐标
    area: int                   # 接触面积 (像素数)
    peak_pressure: float        # 峰值压力 (0-1)
    mean_pressure: float        # 平均压力
    centroid: Tuple[float, float]  # 压力质心
    contact_force: float         # 估计接触力 (N)
    slip_probability: float     # 滑移概率 (0-1)
    temperature: Optional[float]  # 接触区温度
```

#### 2.1.4 力觉 — ForceTorqueSensor

**功能:** 六维力矩测量、接触检测、负载估计、TCP 工具补偿

**核心方法:**
```python
class ForceTorqueSensor:
    def capture(self) -> Wrench
    def detect_contact(self, wrench, threshold=2.0) -> ContactState
    def estimate_payload(self, wrench) -> float
    def set_tool_center(self, mass: float, com: np.ndarray)  # TCP标定
    def calibrate_bias(self, num_samples=100)
```

**Wrench 数据结构:**
```python
@dataclass
class Wrench:
    force: np.ndarray    # 3, [Fx, Fy, Fz] N
    torque: np.ndarray   # 3, [Tx, Ty, Tz] N·m
    timestamp: float
    frame_id: int

    @property def magnitude(self) -> float      # ||F||
    @property def torque_magnitude(self) -> float  # ||T||
    def to_vector(self) -> np.ndarray          # [Fx,Fy,Fz,Tx,Ty,Tz]
    def transform(R, t) -> 'Wrench'             # 坐标变换
```

#### 2.1.5 IMU — IMUSensor

**功能:** 三轴加速度/角速度/磁力计测量、姿态解算、自检标定

**核心方法:**
```python
class IMUSensor:
    def capture(self) -> IMUFrame
    def self_test(self) -> bool
    def calibrate_gyro_bias(self, num_samples=500)
    def calibrate_accel(self, known_orientation="level")

class PoseEstimator:
    def update(self, accel, gyro, mag=None, dt=None) -> Pose
    def get_pose(self) -> Pose
    def get_euler(self) -> np.ndarray  # [roll, pitch, yaw] rad
    def integrate_velocity(self, accel, dt)  # 速度积分 (漂移严重)
```

**Pose 数据结构:**
```python
@dataclass
class Pose:
    position: np.ndarray      # 3, m
    orientation: np.ndarray   # 4, 四元数 [qw, qx, qy, qz]

    def to_euler(self) -> np.ndarray   # [roll, pitch, yaw]
    def to_matrix(self) -> np.ndarray  # 4x4 SE(3)
```

---

### 2.2 融合层 (Fusion)

#### 2.2.1 跨模态注意力网络

**架构:**
```
输入各模态特征
    │
    ├── Vision Encoder ────→ Q_V, K_V, V_V ─┐
    │                                       │
    ├── Audio Encoder ────→ Q_A, K_A, V_A ──┼──→ Cross Attention ──→ 融合特征
    │                                       │
    ├── Tactile Encoder → Q_T, K_T, V_T ──┤
    │                                       │
    ├── Force Encoder ───→ Q_F, K_F, V_F ──┤
    │                                       │
    └── IMU Encoder ─────→ Q_I, K_I, V_I ──┘
```

**接口:**
```python
class CrossModalFusion:
    def __init__(self, config: FusionConfig)
    def forward(self, multimodal_input: MultimodalInput) -> torch.Tensor  # B x hidden_dim
    def encode_modality(self, modality: str, data) -> torch.Tensor
    def compute_cross_attention(self, query, key, value, mask=None) -> torch.Tensor
```

**FusionConfig:**
```python
@dataclass
class FusionConfig:
    vision_dim: int = 512
    audio_dim: int = 128
    tactile_dim: int = 64
    force_dim: int = 32
    imu_dim: int = 64
    hidden_dim: int = 256
    output_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1

class FusionStrategy(Enum):
    EARLY = "early"        # 拼接后统一编码
    MIDDLE = "middle"      # 跨模态注意力交互
    LATE = "late"          # 各模态独立决策后融合
    HYBRID = "hybrid"      # 早期+晚期混合
```

---

### 2.3 认知层 (Cognition)

#### 2.3.1 世界模型 (World Model)

基于 Dreamer 的 RSSM (Recurrent State Space Model):

```python
class WorldModel:
    """RSSM 世界模型"""
    def forward(self, obs_embed, action, prev_state) -> Tuple[RSSMState, Dict]
        # obs_embed: 观测嵌入
        # action: 上一步动作
        # prev_state: 上一步隐状态
        # 返回: (新隐状态, 预测字典)
    
    def imagine(self, policy, horizon: int) -> Tuple[trajectories, rewards]
        # 想象 rollout
        # 给定策略，想象 H 步未来轨迹
    
    def train(self, batch) -> Dict[str, float]
        # 训练: 优化 world model + policy
```

**RSSMState:**
```python
@dataclass
class RSSMState:
    deterministic: torch.Tensor  # 确定性隐状态 (LSTM hidden)
    stochastic: torch.Tensor     # 随机隐状态 (z_t 分布采样)
    log_prob: torch.Tensor       # z_t 的 log 概率
```

#### 2.3.2 任务规划器 (TaskPlanner)

```python
class TaskPlanner:
    def plan(self, task_spec: TaskSpec) -> List[str]
        # 输入任务规格，输出动作序列
    
    def monitor_and_replan(self, current_state, failed_action) -> List[str]
        # 执行失败时重规划

class HierarchicalPlanner(TaskPlanner):
    def plan_hierarchical(self, task_spec) -> List[Task]
        # HTN 层次化分解
    
    def decompose_task(self, task, depth=0) -> List[Task]
        # 递归分解为叶子任务
```

#### 2.3.3 技能库 (SkillLibrary)

```python
class Skill:
    def can_execute(self, context: Dict) -> bool
    def execute(self, context: Dict) -> SkillResult
    def cancel()
    def check_timeout() -> bool

class SkillLibrary:
    def create_skill(name: str, config: Dict) -> Optional[Skill]
    def register_skill(skill: Skill)
    def get_skill(name: str) -> Optional[Skill]
    def list_skills() -> List[str]
```

---

### 2.4 执行层 (Execution)

#### 2.4.1 运动控制 (MotionController)

```python
class MotionController:
    def compute_joint_torque(target_pos, target_vel=None) -> np.ndarray
        # PID 关节位置控制
    
    def compute_cartesian_velocity(target_twist, jacobian) -> np.ndarray
        # 笛卡尔速度 → 关节速度 (雅可比伪逆)
    
    def interpolate_trajectory(trajectory, current_time) -> Tuple[pos, vel]
        # 轨迹插值 (五次多项式)

class ControlMode(Enum):
    JOINT_POSITION    # 关节位置
    JOINT_VELOCITY    # 关节速度
    JOINT_TORQUE      # 关节力矩
    CARTESIAN_VELOCITY # 笛卡尔速度
    CARTESIAN_POSITION # 笛卡尔位置
```

#### 2.4.2 阻抗控制 (ImpedanceController)

**阻抗方程:** M·ẍ + D·ẋ + K·x = F

```python
class ImpedanceController:
    def compute_torque(
        desired_pos, desired_vel,
        current_pos, current_vel,
        external_wrench, jacobian
    ) -> np.ndarray
        # 笛卡尔空间阻抗控制

class AdmittanceController:
    # 导纳控制 (力 → 位置调整)
    def update(external_force, desired_position) -> float

class ForceImpedanceController:
    # 力位混合控制
    def compute_torque(...) -> np.ndarray

class CollaborativeController:
    # 协作安全控制
    def check_safety(external_force, velocity) -> Tuple[bool, str]
    def get_reaction_torque(external_force, jacobian) -> np.ndarray
```

---

### 2.5 仿真层 (Simulation)

```python
class RobotSimulator:
    def step(torque_command) -> Dict  # 物理仿真一步
    def get_jacobian(joint_positions) -> np.ndarray  # 数值雅可比
    def check_environment_collision(obstacles) -> List  # 碰撞检测

class SensorSimulator:
    def get_noisy_joint_positions() -> np.ndarray  # 带噪声位置
    def get_imu_data() -> Dict  # 仿真 IMU
    def get_wrench() -> np.ndarray  # 仿真力觉

PRESET_SCENES = {
    "tabletop":  # 桌面抓取
    "shelf":     # 货架取放
    "door":      # 开门
}
```

---

## 3. 数据流与时序

### 3.1 实时感知控制环 (10ms 周期)

```
T=0ms    传感器采集 (视觉/听觉/触觉/力觉/IMU)
         ↓
T=2ms    神经网络编码器 → 特征向量
         ↓
T=4ms    跨模态注意力融合 → 统一表示
         ↓
T=6ms    世界模型隐状态更新 → 决策
         ↓
T=8ms    运动控制器计算力矩
         ↓
T=10ms   发送到执行器 + 仿真环境更新
         ↓
         (下一个周期)
```

### 3.2 异步学习环

```
后台线程持续采集经验 (obs, action, reward, done)
         ↓
每 N 个周期:
  - 世界模型训练 (Imagine Rollout)
  - Actor-Critic 更新
  - 技能库在线学习
```

---

## 4. 扩展接口

### 4.1 新增传感器

```python
# 1. 继承 SensorBase
class SensorBase:
    def open(self) -> bool
    def close()
    def capture() -> Any
    def calibrate()

# 2. 在 encoder 中注册
class SensorEncoder:
    def register_modality(self, name: str, encoder: nn.Module)
    
# 3. 在 fusion 中添加注意力头
fusion.config.{name}_dim = dim
```

### 4.2 新增控制策略

```python
# 1. 继承 ControllerBase
class ControllerBase:
    def compute(command) -> np.ndarray
    
# 2. 在 SkillLibrary 中注册为技能
```

---

## 5. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1.0 | 2026-03-28 | 初始架构设计 |
| v0.2.0 | 2026-03-28 | 增加详细接口定义、数据格式、处理流水线 |
