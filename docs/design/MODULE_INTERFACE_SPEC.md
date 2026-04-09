# SuperModel 模块接口规范
> **文档版本**: v1.0.0  
> **更新**: 2026-04-10  
> **项目**: SuperModel 超模态机器人具身智能大脑

本文档定义 SuperModel 各模块之间的标准接口，确保模块间的无缝集成。

---

## 一、传感器 → 融合层接口

### 1.1 TactileFrame (触觉帧)

```python
@dataclass
class TactileFrame:
    pressure_map: np.ndarray       # H×W, float32, 归一化压力 0-1
    temperature_map: np.ndarray    # H×W, float32, 温度 °C (可选)
    proximity: np.ndarray          # H×W, float32, 接近距离 m (可选)
    slip_signal: np.ndarray        # H×W, float32, 滑移信号 0-1 (可选)
    timestamp: float               # Unix时间戳
    frame_id: int                  # 帧序号
    sensor_id: str                 # 传感器标识

# 编码接口
def tactile_to_encoding(frame: TactileFrame) -> torch.Tensor:
    # 转换为 (1, 1, H, W) 的张量
    return torch.from_numpy(frame.pressure_map).float().unsqueeze(0).unsqueeze(0)
```

### 1.2 Wrench (力旋量)

```python
@dataclass
class Wrench:
    force: np.ndarray   # (3,), Fx/Fy/Fz 单位: N
    torque: np.ndarray  # (3,), Tx/Ty/Tz 单位: N·m
    timestamp: float
    frame_id: int
    sensor_id: str

# 接口方法
def wrench_to_vector(w: Wrench) -> np.ndarray:
    """转换为6维向量 [Fx,Fy,Fz,Tx,Ty,Tz]"""
    return np.concatenate([w.force, w.torque])

def wrench_from_vector(v: np.ndarray, **kwargs) -> Wrench:
    """从6维向量创建"""
    return Wrench(force=v[:3], torque=v[3:], **kwargs)

def wrench_transform(w: Wrench, R: np.ndarray, t: np.ndarray) -> Wrench:
    """坐标变换: R×force + t×R×force → torque"""
    new_force = R @ w.force
    new_torque = R @ w.torque + np.cross(t, new_force)
    return Wrench(force=new_force, torque=new_torque)
```

### 1.3 IMUFrame (IMU帧)

```python
@dataclass
class IMUFrame:
    accel: np.ndarray          # (3,), 加速度 m/s²
    gyro: np.ndarray           # (3,), 角速度 rad/s
    mag: Optional[np.ndarray]  # (3,), 磁力计 μT (可选)
    temperature: float         # 温度 °C
    timestamp: float
    frame_id: int
    sensor_id: str

# 编码接口
def imu_to_vector(frame: IMUFrame) -> np.ndarray:
    """拼接加速度+角速度 → (6,)"""
    return np.concatenate([frame.accel, frame.gyro])

# 姿态估计接口
def estimate_pose(imu_frame: IMUFrame, estimator: PoseEstimator) -> Pose:
    return estimator.update(imu_frame.accel, imu_frame.gyro)
```

---

## 二、融合层 → 控制层接口

### 2.1 MultimodalInput (多模态输入)

```python
@dataclass
class MultimodalInput:
    vision: Optional[torch.Tensor]  # (B,3,H,W) RGB图像
    audio: Optional[torch.Tensor]    # (B, D) 音频特征
    tactile: Optional[torch.Tensor] # (B,1,H,W) 触觉图
    force: Optional[torch.Tensor]   # (B,6) 力旋量
    imu: Optional[torch.Tensor]     # (B,6) IMU向量
    text: Optional[torch.Tensor]    # (B, L) 文本token
    depth: Optional[torch.Tensor]   # (B,1,H,W) 深度图
    lidar: Optional[torch.Tensor]   # (B, N) 激光雷达点云

@dataclass 
class FusionConfig:
    hidden_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    grade: str = 'M'
```

### 2.2 融合输出接口

```python
class CrossModalFusion(nn.Module):
    def forward(self, multimodal: MultimodalInput) -> FusionOutput:
        """
        Returns:
            FusionOutput:
                .state: (B, hidden_dim) 全局状态
                .modality_weights: Dict[str, float] 各模态权重
                .attn_weights: Optional[torch.Tensor] 注意力权重
        """

# 控制层使用
fused = fusion(multimodal)
control_input = fused.state  # (B, hidden_dim)
```

---

## 三、控制层内部接口

### 3.1 MotionController 接口

```python
class MotionController(ABC):
    @abstractmethod
    def compute(self, state: JointState, desired: JointTrajectory) -> MotorCommand:
        """计算电机控制量"""
        pass

@dataclass
class JointState:
    position: np.ndarray    # (n_joints,) 关节位置 rad
    velocity: np.ndarray    # (n_joints,) 关节速度 rad/s
    torque: np.ndarray      # (n_joints,) 关节力矩 N·m
    timestamp: float

@dataclass
class JointTrajectory:
    positions: np.ndarray   # (T, n_joints) 轨迹位置
    velocities: np.ndarray  # (T, n_joints) 轨迹速度
    accelerations: np.ndarray  # (T, n_joints) 轨迹加速度
    timestamps: np.ndarray  # (T,) 时间戳

@dataclass
class MotorCommand:
    voltage: np.ndarray     # (n_joints,) 电压命令 V
    current: np.ndarray     # (n_joints,) 电流命令 A
    mode: MotorControlMode
```

### 3.2 Trajectory 接口

```python
@dataclass
class CartesianWaypoint:
    position: np.ndarray    # (3,) 位置 m
    orientation: np.ndarray # (4,) 四元数 (qw,qx,qy,qz)
    velocity: np.ndarray    # (3,) 速度 m/s (可选)
    timestamp: float

class TrajectoryGenerator(ABC):
    @abstractmethod
    def generate(self, waypoints: List[CartesianWaypoint]) -> Trajectory:
        """从路点生成轨迹"""
        pass

@dataclass
class Trajectory:
    positions: np.ndarray      # (T, 3)
    orientations: np.ndarray    # (T, 4)
    velocities: np.ndarray     # (T, 3)
    angular_velocities: np.ndarray  # (T, 3)
    accelerations: np.ndarray   # (T, 3)
    timestamps: np.ndarray     # (T,)
```

### 3.3 SafetyController 接口

```python
class SafetyController:
    def check(self, state: JointState, command: MotorCommand) -> SafetyResult:
        """安全检查"""

@dataclass
class SafetyResult:
    is_safe: bool
    violations: List[str]
    recommended_action: str
    severity: SafetyLevel  # GREEN/YELLOW/ORANGE/RED
```

---

## 四、传感器管理器接口

### 4.1 SensorManager

```python
class SensorManager:
    def __init__(self, config: SensorManagerConfig):
        """初始化所有传感器"""
    
    def start(self):
        """启动所有传感器采集"""
    
    def stop(self):
        """停止所有传感器"""
    
    def get_frame(self, sensor_id: str) -> Any:
        """获取指定传感器最新帧"""
    
    def get_all_frames(self) -> Dict[str, Any]:
        """获取所有传感器最新帧"""
    
    def fuse_all(self) -> MultimodalInput:
        """融合所有传感器数据为 MultimodalInput"""
    
    def calibrate_all(self):
        """标定所有传感器"""

@dataclass
class SensorManagerConfig:
    grade: str = 'M'
    sensors: List[str] = field(default_factory=list)  # ['tactile','force','imu','vision','audio']
    sample_rates: Dict[str, int] = field(default_factory=dict)
    enable_fusion: bool = True
```

---

## 五、五级规格快速查询

| 函数 | 说明 |
|------|------|
| `get_tactile_spec(grade)` | 获取触觉规格 |
| `get_force_spec(grade)` | 获取力觉规格 |
| `get_imu_spec(grade)` | 获取IMU规格 |
| `get_encoder_config(grade)` | 获取编码器规格 |
| `get_trajectory_spec(grade)` | 获取轨迹规划规格 |
| `get_mpc_spec(grade)` | 获取MPC规格 |
| `get_safety_spec(grade)` | 获取安全监控规格 |
| `get_obstacle_avoidance_spec(grade)` | 获取避障规格 |
| `get_supervisor_config(grade)` | 获取监管器配置 |
| `get_sensorimotor_spec(grade)` | 获取传感-运动规格 |
| `get_teleop_spec(grade)` | 获取遥操作规格 |

---

## 六、数据流时序

```
传感器采集 (并行):
  TactileArray.capture()     → TactileFrame
  ForceTorqueSensor.capture() → Wrench  
  IMUSensor.capture()        → IMUFrame
       ↓
传感器预处理:
  TactileEncoder(frame)      → (1,1,H,W) tensor
  ForceEncoder(wrench)       → (1,6) tensor
  IMUEncoder(frame)          → (1,6) tensor
       ↓
跨模态融合:
  CrossModalFusion(input)    → FusionOutput.state (B, hidden_dim)
       ↓
控制决策:
  SensorimotorIntegration    → 控制量
  TrajectoryTracker          → 电机命令
       ↓
电机执行:
  MotorController            → 电压/电流输出
```
