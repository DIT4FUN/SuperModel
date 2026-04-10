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

### 3.3 BehaviorTree 接口

```python
# === 行为树核心接口 ===
class BTNode(ABC):
    @abstractmethod
    def tick(self, ctx: BTContext) -> NodeState: ...

class BehaviorTree:
    def __init__(self, root: BTNode, grade: BTGrade = BTGrade.M):
        self.grade = grade
        self.root = root
        self._tick_count = 0
    
    def tick(self, ctx: BTContext) -> NodeState:
        """执行一次 tick，返回根节点最终状态"""
        return self.root.tick(ctx)
    
    @classmethod
    def create_for_grade(cls, grade: BTGrade, root: BTNode, name: str) -> 'BehaviorTree':
        """根据AGV五级规格创建行为树"""

# === 组合节点 ===
class Selector(BTNode):  # 或关系：任意子节点成功即成功
class Sequence(BTNode):  # 与关系：全部子节点成功才成功
class Parallel(BTNode): # 并行执行多个子节点

# === 叶子节点 ===
class Condition(BTNode):  # 条件判断（返回 SUCCESS/FAILURE）
class Action(BTNode):     # 动作执行（可返回 RUNNING/SUCCESS/FAILURE）
class SubTree(BTNode):    # 子树调用（行为树复用）

# === 装饰器节点 ===
class Inverter(BTNode):     # 反转子节点结果
class RepeatUntil(BTNode): # 重复执行直到满足条件
class RetryUntil(BTNode):  # 重试直到成功
class Timeout(BTNode):     # 超时装饰器
class RateLimiter(BTNode): # 频率限制装饰器

# === 上下文 ===
@dataclass
class BTContext:
    robot_state: Dict[str, Any]    # 机器人当前状态
    task_goal: Optional[str]       # 当前任务目标
    sensor_data: Dict[str, Any]   # 传感器数据
    control_output: Dict[str, Any] # 控制输出
    bt_data: Dict[str, Any]       # 行为树内部数据

# === AGV五级规格 ===
AGV_BT_GRADES = {
    'S':   {'max_nodes': 20,  'tick_rate': 10,  'parallel': 1},
    'M':   {'max_nodes': 50,  'tick_rate': 50,  'parallel': 3},
    'L':   {'max_nodes': 150, 'tick_rate': 100, 'parallel': 5},
    'XL':  {'max_nodes': 500, 'tick_rate': 200, 'parallel': 10},
    'XXL': {'max_nodes': 2000,'tick_rate': 500, 'parallel': 20},
}

# === 工厂函数 ===
def create_for_grade(grade: BTGrade, root: BTNode, name: str) -> BehaviorTree
def create_safe_selector(name: str, children: List[BTNode], grade: BTGrade) -> Selector
def create_action_sequence(name: str, actions: List[BTNode], grade: BTGrade) -> Sequence
```

### 3.4 SafetyController 接口

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

---

## 七、完整AGV五级规格总表

### 7.1 整车基础规格对照

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **负载能力** | 30kg | 100kg | 300kg | 600kg | 1200kg |
| **自重** | 15kg | 35kg | 80kg | 150kg | 300kg |
| **最大总重** | 45kg | 135kg | 380kg | 750kg | 1500kg |
| **车体尺寸** | 0.4×0.3×0.12m | 0.6×0.4×0.15m | 0.8×0.6×0.2m | 1.0×0.7×0.25m | 1.2×0.9×0.3m |
| **轮子配置** | 2轮驱动 | 2轮驱动 | 4轮驱动 | 4轮驱动 | 4轮驱动 |
| **电机类型** | 57步进 | 5.5寸150W | 5.5寸150W×2 | 6.5寸200W×2 | 7.5寸300W×4 |
| **最高速度** | 0.5m/s | 1.5m/s | 2.0m/s | 2.5m/s | 3.0m/s |
| **最大扭矩** | 5Nm | 15Nm | 30Nm | 60Nm | 120Nm |
| **定位精度** | ±10mm | ±5mm | ±3mm | ±1mm | ±0.5mm |
| **防护等级** | IP20 | IP30 | IP54 | IP65 | IP67 |
| **典型价格** | ¥5-15K | ¥15-50K | ¥50-150K | ¥150-500K | >¥500K |

### 7.2 传感器子系统规格对照

| 传感器 | 参数 | S | M | L | XL | XXL |
|--------|------|:--:|:--:|:--:|:--:|:--:|
| **视觉** | 配置 | 单目640×480 | 双目D435i 720p | 双目D455 60fps | 双目+事件相机 | 多目+3D LiDAR |
| | 分辨率 | 640×480 | 1280×720 | 1280×720 | 1920×1080 | 1920×1080×4 |
| | 帧率 | 30fps | 30fps | 60fps | 90fps | 120fps |
| | 编码维度 | — | 256 | 512 | 768 | 1024 |
| **听觉** | 麦克风 | 1ch | 2ch阵列 | 4ch阵列 | 6ch阵列 | 8ch阵列 |
| | 采样率 | 16000Hz | 16000Hz | 22050Hz | 32000Hz | 44100Hz |
| | 拾音范围 | 1.0m | 3.0m | 5.0m | 8.0m | 10.0m |
| | 声源定位精度 | — | ±15° | ±5° | ±2° | ±1° |
| | 编码维度 | 64 | 128 | 128 | 256 | 256 |
| **触觉** | 阵列尺寸 | 8×8 | 16×16 | 24×24 | 32×32 | 48×48 |
| | 分辨率 | 12bit | 12bit | 14bit | 14bit | 16bit |
| | 压力范围 | 0-500kPa | 0-1000kPa | 0-2000kPa | 0-5000kPa | 0-10000kPa |
| | 采样频率 | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| | 温度感知 | ✗ | ✓ | ✓ | ✓ | ✓ |
| | 接近觉 | ✗ | ✗ | ✓ | ✓ | ✓ |
| | 滑移检测 | ✗ | ✓ | ✓ | ✓ | ✓ |
| **力觉** | 轴数 | 3 | 6 | 6 | 6 | 6 |
| | 力范围 | ±100N | ±200N | ±500N | ±1000N | ±5000N |
| | 力矩范围 | ±10Nm | ±20Nm | ±50Nm | ±100Nm | ±500Nm |
| | 采样频率 | 100Hz | 500Hz | 1000Hz | 2000Hz | 5000Hz |
| | 分辨力 | 0.1N | 0.05N | 0.02N | 0.01N | 0.005N |
| **IMU** | 型号 | MPU6050 | BMI088 | BMI088 | ADIS16470 | ADIS16470 |
| | 加速度量程 | ±8g | ±16g | ±24g | ±40g | ±80g |
| | 陀螺音量程 | ±1000°/s | ±2000°/s | ±4000°/s | ±4000°/s | ±8000°/s |
| | 采样频率 | 100Hz | 200Hz | 500Hz | 1000Hz | 2000Hz |
| | 噪声密度 | 400μg/√Hz | 120μg/√Hz | 60μg/√Hz | 20μg/√Hz | 10μg/√Hz |
| **编码器** | 类型 | 增量式 | 绝对值 | 绝对值 | 多圈绝对 | 多圈绝对 |
| | 分辨率 | 12bit | 17bit | 17bit | 20bit | 20bit |
| | 接口 | 并行 | RS485 | EtherCAT | EtherCAT | EtherCAT |

### 7.3 融合与学习规格对照

| 模块 | 参数 | S | M | L | XL | XXL |
|------|------|:--:|:--:|:--:|:--:|:--:|
| **跨模态融合** | 融合维度 | 256 | 512 | 768 | 1024 | 1536 |
| | 注意力头数 | 4 | 8 | 12 | 16 | 16 |
| | 模态数量 | 3 | 5 | 6 | 7 | 8 |
| | 融合频率 | 10Hz | 30Hz | 100Hz | 200Hz | 500Hz |
| **世界模型** | 模型类型 | MLP | Transformer | Transformer+物理 | 物理先验 | 物理先验+因果 |
| | 预测窗口 | 0.5s | 1.0s | 2.0s | 5.0s | 10.0s |
| | 更新频率 | 1Hz | 5Hz | 10Hz | 20Hz | 50Hz |
| **规划控制** | 控制频率 | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| | 规划算法 | PID | MPC | MPC+RL | MPC+RL+预测 | MPC+RL+预测+博弈 |
| | 安全等级 | 基础 | 增强 | 高级 | 冗余 | 容错+自愈 |
| **导航避障** | 定位方式 | 码盘 | 视觉+码盘 | 激光+视觉 | 激光+视觉+IMU | 多传感器融合 |
| | 避障范围 | 0.5m | 1.0m | 2.0m | 3.0m | 5.0m |
| | 响应时间 | 200ms | 100ms | 50ms | 20ms | 10ms |

---

## 八、控制模块详细接口

### 8.1 触觉控制 TactileControl

```python
class TactileServoController:
    """触觉伺服控制器"""
    def __init__(self, grade: str = 'M', params: Optional[TactileServoParams] = None)
    def compute_grip_force(self, frame: TactileFrame, desired_pressure: float) -> float
    def detect_slip(self, frame: TactileFrame) -> Tuple[bool, float]
    def estimate_contact_geometry(self, frame: TactileFrame) -> ContactGeometry
    def step(self, frame: TactileFrame, desired: float, dt: float) -> float

class GraspQualityController:
    """抓取质量评估与控制器"""
    def evaluate(self, frame: TactileFrame) -> GraspQuality
    def optimize_grip(self, frame: TactileFrame, initial_force: float) -> float
```

| AGV等级 | 控制频率 | 力控精度 | 滑移检测 | 抓取质量评估 |
|:--:|:--:|:--:|:--:|:--:|
| S | 50Hz | ±0.5N | ✗ | 基础 |
| M | 100Hz | ±0.2N | ✓ | 完整 |
| L | 200Hz | ±0.1N | ✓ | 完整+优化 |
| XL | 500Hz | ±0.05N | ✓ | 完整+预测 |
| XXL | 1000Hz | ±0.02N | ✓ | 完整+预测+自学习 |

### 8.2 力觉控制 ForceControl

```python
class ForceController:
    """力觉控制器"""
    def __init__(self, grade: str = 'M', params: Optional[ForceControlParams] = None)
    def set_force_target(self, wrench: Wrench)
    def set_impedance(self, stiffness: np.ndarray, damping: np.ndarray)
    def step(self, current_wrench: Wrench, dt: float) -> MotorCommand

class HybridForcePositionController:
    """混合力位控制器"""
    def __init__(self, force_axis: List[int], position_axis: List[int])
    def compute(self, state: JointState, desired_force: Wrench, desired_pos: Pose) -> MotorCommand
```

| AGV等级 | 控制频率 | 力控精度 | 响应带宽 | 控制模式 |
|:--:|:--:|:--:|:--:|:--:|
| S | 100Hz | ±2N | 5Hz | 恒力 |
| M | 500Hz | ±0.5N | 20Hz | 力位混合 |
| L | 1000Hz | ±0.2N | 50Hz | 阻抗+力位 |
| XL | 2000Hz | ±0.1N | 100Hz | 自适应阻抗 |
| XXL | 5000Hz | ±0.05N | 200Hz | 学习型阻抗 |

### 8.3 IMU控制 IMUControl

```python
class AttitudeStabilizer:
    """姿态稳定器"""
    def __init__(self, grade: str = 'M', params: Optional[IMUControlParams] = None)
    def set_target_orientation(self, quat: np.ndarray)
    def step(self, imu_frame: IMUFrame, dt: float) -> MotorCommand
    def get_current_orientation(self) -> np.ndarray

class MotionEstimator:
    """运动状态估计器"""
    def update(self, imu_frame: IMUFrame, dt: float) -> Tuple[np.ndarray, np.ndarray]
    def get_velocity(self) -> np.ndarray
    def get_position(self) -> np.ndarray
    def reset(self)
```

| AGV等级 | 姿态精度 | 位置精度(积分1s) | 零偏估计 | 异常检测 |
|:--:|:--:|:--:|:--:|:--:|
| S | ±2° | ±5cm | ✗ | ✗ |
| M | ±0.5° | ±1cm | ✓ | ✗ |
| L | ±0.1° | ±2mm | ✓ | ✓ |
| XL | ±0.05° | ±0.5mm | ✓ | ✓+预测 |
| XXL | ±0.01° | ±0.1mm | ✓+在线补偿 | ✓+预测+容错 |

### 8.4 传感-运动融合 Sensorimotor

```python
class SensorimotorIntegration:
    """传感-运动整合模块"""
    def __init__(self, grade: str = 'M', config: Optional[SensorimotorConfig] = None)
    def update(self, tactile_frame, wrench, imu_frame, desired_force) -> SensorimotorState
    def get_control_output(self) -> Dict[str, Any]
    def reset(self)
    def get_health_status() -> HealthStatus

@dataclass
class SensorimotorState:
    control_effort: np.ndarray       # 控制量
    grip_force: float               # 抓取力
    slip_probability: float         # 滑移概率
    contact_quality: float          # 接触质量
    body_pose: Pose                 # 身体姿态
    body_velocity: np.ndarray       # 身体速度
    force_saturation: bool          # 力饱和标志
    sensor_health: Dict[str, bool]  # 各传感器健康状态
```

| AGV等级 | 控制闭环频率 | 触觉-运动延迟 | 力控-运动延迟 | 融合维度 |
|:--:|:--:|:--:|:--:|:--:|
| S | 50Hz | 20ms | 10ms | 64 |
| M | 100Hz | 10ms | 5ms | 128 |
| L | 200Hz | 5ms | 2ms | 256 |
| XL | 500Hz | 2ms | 1ms | 512 |
| XXL | 1000Hz | 1ms | 0.5ms | 1024 |

---

## 九、偏置补偿模块接口

```python
class IMUBiasEstimator:
    """IMU偏置估计器"""
    def __init__(self, estimator_type: str = 'kalman', grade: str = 'M')
    def update(self, frame: IMUFrame, dt: float) -> IMUFrame
    def get_bias(self) -> Tuple[np.ndarray, np.ndarray]  # (accel_bias, gyro_bias)
    def reset(self)

class ForceBiasEstimator:
    """力觉偏置估计器"""
    def __init__(self, method: str = 'moving_average', grade: str = 'M')
    def update(self, wrench: Wrench) -> Wrench
    def get_bias(self) -> np.ndarray
    def calibrate(self, samples: List[Wrench])
    def reset()

class TactileBiasEstimator:
    """触觉偏置估计器"""
    def __init__(self, grade: str = 'M')
    def update(self, frame: TactileFrame) -> TactileFrame
    def set_baseline(self, frame: TactileFrame)
    def get_baseline(self) -> TactileFrame

class MultiSensorBiasCompensator:
    """多传感器偏置补偿器"""
    def __init__(self, config: BiasCompensationConfig)
    def compensate(self, imu: IMUFrame, force: Wrench, tactile: TactileFrame) -> Tuple[IMUFrame, Wrench, TactileFrame]
    def calibrate_all(self, duration: float)
    def get_status(self) -> Dict[str, Any]
```

| AGV等级 | IMU偏置估计 | 力觉偏置估计 | 触觉偏置估计 | 在线补偿 |
|:--:|:--:|:--:|:--:|:--:|
| S | ✗ | ✗ | ✗ | ✗ |
| M | ✓静止 | ✓静止 | ✓ | ✗ |
| L | ✓静止+运动 | ✓在线 | ✓ | ✗ |
| XL | ✓在线+自适应 | ✓在线+温漂 | ✓+温漂 | ✓ |
| XXL | ✓在线+自适应+物理约束 | ✓在线+温漂+物理约束 | ✓+温漂+鲁棒估计 | ✓+自适应 |

---

## 十、版本历史

| 版本 | 日期 | 变更内容 |
|------|------|---------|
| v1.0.0 | 2026-04-10 | 初始版本，定义传感器→融合→控制全链路接口 |
