# SuperModel 技术规格文档 (SPEC.md)

## 1. 模块接口设计

### 1.1 传感器模块接口

#### TactileArray (触觉传感器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `open()` | - | bool | 打开传感器连接 |
| `close()` | - | None | 关闭传感器 |
| `capture()` | - | TactileFrame | 捕获触觉数据帧 |
| `detect_contacts(frame)` | TactileFrame | List[TactileContact] | 检测接触区域 |
| `get_slip_signal(frame)` | TactileFrame | np.ndarray | 获取滑移信号 |
| `estimate_grip_quality(frame)` | TactileFrame | Dict | 估计抓取质量 |
| `calibrate(zero_pressure, known_weights)` | np.ndarray, List[float] | None | 传感器标定 |

#### ForceTorqueSensor (力觉传感器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `open()` | - | bool | 打开传感器连接 |
| `close()` | - | None | 关闭传感器 |
| `capture()` | - | Wrench | 捕获六维力旋量 |
| `get_wrench()` | - | Wrench | 获取最新力数据 |
| `detect_contact(wrench, threshold)` | Wrench, float | ContactState | 接触检测 |
| `estimate_payload(wrench)` | Wrench | float | 估计负载重量 |
| `calibrate_bias(num_samples)` | int | None | 偏置校准 |
| `set_tool_center(tool_mass, tool_com)` | float, np.ndarray | None | 设置工具中心 |

#### IMUSensor (IMU传感器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `open()` | - | bool | 打开传感器连接 |
| `close()` | - | None | 关闭传感器 |
| `capture()` | - | IMUFrame | 捕获IMU数据帧 |
| `self_test()` | - | bool | 传感器自检 |
| `calibrate_gyro_bias(num_samples)` | int | None | 陀螺仪偏置校准 |
| `calibrate_accel(known_orientation)` | str | None | 加速度计标定 |

#### PoseEstimator (姿态估计器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `update(accel, gyro, mag, dt)` | np.ndarray, np.ndarray, np.ndarray, float | Pose | 更新姿态估计 |
| `get_pose()` | - | Pose | 获取当前姿态 |
| `get_euler()` | - | np.ndarray | 获取当前欧拉角 |
| `get_rotation_matrix()` | - | np.ndarray | 获取旋转矩阵 |
| `integrate_velocity(accel, dt, remove_gravity)` | np.ndarray, float, bool | Tuple | 积分速度/位置 |
| `reset()` | - | None | 重置积分状态 |

### 1.2 控制模块接口

#### AGVMotionController (AGV运动控制器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `set_target_pose(pose)` | AGVPose | None | 设置目标位姿 |
| `set_target_twist(twist)` | AGVTwist | None | 设置目标速度 |
| `step(dt)` | float | np.ndarray | 步进控制,返回轮速 |
| `move_to(x, y, theta, dt)` | float, float, float, float | np.ndarray | 移动到目标位置 |
| `stop()` | - | np.ndarray | 停止运动 |
| `emergency_stop()` | - | None | 紧急停止 |

#### SafetyMonitor (安全监控器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `check_velocity(velocity, dt, timestamp)` | np.ndarray, float, float | SafetyStatus | 检查速度限制 |
| `check_boundary(position, timestamp)` | np.ndarray, float | SafetyStatus | 检查边界限制 |
| `check_force(force_magnitude, torque_magnitude, timestamp)` | float, float, float | SafetyStatus | 检查力限制 |
| `check_collision(collision_detected, timestamp)` | bool, float | SafetyStatus | 检查碰撞 |
| `check_all(...)` | ... | SafetyStatus | 综合安全检查 |
| `emergency_stop(reason)` | str | None | 触发急停 |
| `reset_estop()` | - | None | 重置急停 |

#### GradeAwareSupervisor (AGV五级感知控制监管器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(grade, supervisor_id)` | SupervisorGrade, str | GradeAwareSupervisor | 按AGV等级初始化监管器 |
| `register_controller(controller)` | ControllerInterface | bool | 注册控制器 |
| `unregister_controller(name)` | str | bool | 注销控制器 |
| `list_controllers()` | - | List[str] | 列出所有控制器 |
| `get_controller(name)` | str | ControllerInterface | 按名称获取控制器 |
| `switch_mode(target_mode)` | ControlMode | bool | 切换控制模式 |
| `trigger_emergency_stop(reason)` | str | None | 触发紧急停止 |
| `release_emergency_stop()` | - | bool | 解除紧急停止 |
| `get_state()` | - | ControlState | 获取当前状态 |
| `get_diagnostics()` | - | Dict | 获取诊断数据 |
| `step_watchdog()` | - | bool | 看门狗心跳 (XL/XXL) |
| `step_fault_tolerance(fault_detected)` | bool | None | 故障容错处理 (XXL) |
| `get_grade_capabilities()` | - | Dict | 获取等级能力清单 |
| `register_with_redundancy(controller, modes, is_primary)` | ControllerInterface, List, bool | bool | 冗余注册 (L+) |

#### TrajectoryPlanner (轨迹规划器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `plan_line(start, end)` | Waypoint, Waypoint | List[TrajectoryPoint] | 直线规划 |
| `plan_arc(start, end, curv)` | Waypoint, Waypoint, float | List[TrajectoryPoint] | 圆弧规划 |
| `smooth_trajectory(traj)` | List[TrajectoryPoint] | List[TrajectoryPoint] | 轨迹平滑 |
| `plan_path(waypoints)` | List[Waypoint] | List[TrajectoryPoint] | 多路点规划 |

#### NavigationController (AGV导航控制器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `plan_to_goal(start, goal)` | np.ndarray, np.ndarray | bool | 规划到目标位姿 |
| `set_global_path(path)` | Path | None | 设置全局路径 |
| `update(current_pose, dt)` | np.ndarray, float | np.ndarray | 更新导航,返回速度指令 |
| `reset()` | - | None | 重置导航状态 |
| `emergency_stop()` | - | None | 紧急停止 |
| `get_state()` | - | NavigationState | 获取导航状态 |
| `get_progress()` | - | float | 获取导航进度 0.0~1.0 |

#### OccupancyGrid (占用栅格地图)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `world_to_grid(wx, wy)` | float, float | Tuple[int, int] | 世界坐标转栅格坐标 |
| `grid_to_world(gx, gy)` | int, int | Tuple[float, float] | 栅格坐标转世界坐标 |
| `set_obstacle(wx, wy, radius)` | float, float, float | None | 设置障碍物 |
| `is_free(wx, wy)` | float, float | bool | 检查是否空闲 |
| `get_nearby_obstacles(wx, wy, radius)` | float, float, float | List[Tuple[int, int]] | 获取附近障碍物 |

#### DijkstraPlanner / AStarPlanner (全局路径规划器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `plan(start, goal)` | Tuple[float, float], Tuple[float, float] | Optional[Path] | 规划全局路径 |

### 1.3 融合模块接口

#### CrossModalFusion (跨模态融合网络)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `forward(multimodal_input)` | MultimodalInput | UnifiedRepresentation | 前向传播 |
| `encode_vision(vision_feat)` | torch.Tensor | torch.Tensor | 视觉编码 |
| `encode_audio(audio_feat)` | torch.Tensor | torch.Tensor | 听觉编码 |
| `encode_tactile(tactile_feat)` | torch.Tensor | torch.Tensor | 触觉编码 |
| `encode_force(force_feat)` | torch.Tensor | torch.Tensor | 力觉编码 |
| `encode_imu(imu_feat)` | torch.Tensor | torch.Tensor | IMU编码 |

#### ComplementaryFilter (互补滤波器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `update(measurements, dt)` | Dict, float | np.ndarray | 更新融合状态 |
| `get_state()` | - | np.ndarray | 获取当前状态 |
| `reset()` | - | None | 重置滤波器 |

#### ExtendedKalmanFilter (扩展卡尔曼滤波)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `initialize(initial_state)` | np.ndarray | None | 初始化状态 |
| `predict(dt)` | float | None | 预测步骤 |
| `correct(measurement)` | np.ndarray | None | 校正步骤 |
| `update(measurements, dt)` | Dict, float | np.ndarray | 完整更新 |
| `get_state()` | - | np.ndarray | 获取当前状态 |
| `get_covariance()` | - | np.ndarray | 获取协方差矩阵 |

## 2. AGV五级规格表

### 2.1 基础规格对比

| 等级 | 负载能力 | 导航方式 | 定位精度 | 安全标准 | 典型场景 | 代表型号 |
|------|---------|---------|---------|---------|---------|---------|
| **L1** | ≤500kg | 磁条/二维码 | ±10mm | ISO 3691-2 | 仓储拣选 | 潜伏式AGV |
| **L2** | 500-1500kg | 激光导航 | ±5mm | ISO 3691-4 | 产线配送 | 叉式AGV |
| **L3** | 1500-3000kg | SLAM视觉 | ±3mm | ISO 3691-4 | 柔性制造 | 复合AGV |
| **L4** | 3000-5000kg | 多传感器融合 | ±1mm | IEC 61508 SIL2 | 重载车间 | 重载AGV |
| **L5** | >5000kg | 具身智能超模态 | <±0.5mm | IEC 61508 SIL3 | 无人化工厂 | 超级AGV |

### 2.2 触觉传感器等级规格

| 等级 | 阵列尺寸 | 分辨率 | 压力范围 | 采样频率 | 温度感知 |
|------|---------|--------|---------|---------|----------|
| **S** | 8x8 | 12bit | 0-500 kPa | 50 Hz | 否 |
| **M** | 16x16 | 12bit | 0-1000 kPa | 100 Hz | 是 |
| **L** | 24x24 | 14bit | 0-2000 kPa | 200 Hz | 是 |
| **XL** | 32x32 | 14bit | 0-5000 kPa | 500 Hz | 是 |
| **XXL** | 48x48 | 16bit | 0-10000 kPa | 1000 Hz | 是 |

### 2.3 力觉传感器等级规格

| 等级 | 轴数 | 力范围 | 力矩范围 | 分辨率 | 采样频率 |
|------|-----|--------|---------|--------|----------|
| **S** | 3 | ±100 N | ±10 N·m | 0.1 N | 100 Hz |
| **M** | 6 | ±200 N | ±20 N·m | 0.05 N | 500 Hz |
| **L** | 6 | ±500 N | ±50 N·m | 0.02 N | 1000 Hz |
| **XL** | 6 | ±1000 N | ±100 N·m | 0.01 N | 2000 Hz |
| **XXL** | 6 | ±5000 N | ±500 N·m | 0.005 N | 5000 Hz |

### 2.4 IMU传感器等级规格

| 等级 | 型号 | 加速度范围 | 陀螺仪范围 | 采样频率 | 噪声密度 |
|------|-----|-----------|-----------|---------|----------|
| **S** | MPU6050 | ±8g | ±1000°/s | 100 Hz | 400 μg/√Hz |
| **M** | BMI088 | ±16g | ±2000°/s | 200 Hz | 120 μg/√Hz |
| **L** | BMI088 | ±24g | ±4000°/s | 500 Hz | 60 μg/√Hz |
| **XL** | ADIS16470 | ±40g | ±4000°/s | 1000 Hz | 20 μg/√Hz |
| **XXL** | ADIS16470 | ±80g | ±8000°/s | 2000 Hz | 10 μg/√Hz |

### 2.5 L5级 SuperModel 核心规格

| 参数 | 规格值 |
|------|--------|
| 处理器 | NVIDIA Jetson AGX Orin / Tesla T4 |
| AI算力 | ≥275 TOPS (INT8) |
| 传感器配置 | 深度相机 + 激光雷达 + IMU + 力觉 + 触觉 |
| 定位精度 | <±0.5mm (融合定位) |
| 导航速度 | 0-3m/s (自适应调速) |
| 负载能力 | 100-5000kg (模块化设计) |
| 安全标准 | ISO 3691-4, IEC 61508 SIL2/SIL3 |
| 通讯协议 | WiFi 6E, 5G, MQTT, ROS2 |
| 续航能力 | 8-24h (视电池配置) |
| 多模态输入 | 视觉/听觉/触觉/力觉/IMU/位置 |
| 具身智能 | 超模态大模型 + 强化学习自主学习 |

## 3. 数据格式规范

### TactileFrame
```python
@dataclass
class TactileFrame:
    pressure_map: np.ndarray          # H x W, 压力值 (归一化 0-1)
    temperature_map: Optional[np.ndarray] = None  # H x W, 温度 (摄氏度)
    proximity: Optional[np.ndarray] = None  # H x W, 接近距离 (米)
    slip_signal: Optional[np.ndarray] = None  # H x W, 滑移信号
    timestamp: float = 0.0
    frame_id: int = 0
    sensor_id: str = "default"
```

### Wrench (六维力旋量)
```python
@dataclass
class Wrench:
    force: np.ndarray           # 3, 力向量 (Fx, Fy, Fz), N
    torque: np.ndarray           # 3, 力矩向量 (Tx, Ty, Tz), N·m
    timestamp: float = 0.0
    frame_id: int = 0
    sensor_id: str = "default"
```

### IMUFrame
```python
@dataclass
class IMUFrame:
    accel: np.ndarray          # 3, 加速度 (m/s^2)
    gyro: np.ndarray           # 3, 角速度 (rad/s)
    mag: Optional[np.ndarray]  # 3, 磁力计 (可选)
    temperature: float = 25.0  # 温度 (摄氏度)
    timestamp: float = 0.0
    frame_id: int = 0
    sensor_id: str = "imu_0"
```

### Pose
```python
@dataclass
class Pose:
    position: np.ndarray        # 3, 位置 (m)
    orientation: np.ndarray      # 4, 四元数 (qw, qx, qy, qz)
```

## 4. AGV五级完整规格表

### 4.1 综合规格对比

| 参数 | L1 | L2 | L3 | L4 | L5 |
|------|-----|-----|-----|-----|-----|
| **负载能力** | ≤500kg | 500-1500kg | 1500-3000kg | 3000-5000kg | >5000kg |
| **导航方式** | 磁条/二维码 | 激光导航 | SLAM视觉 | 多传感器融合 | 超模态具身智能 |
| **定位精度** | ±10mm | ±5mm | ±3mm | ±1mm | <±0.5mm |
| **安全标准** | ISO 3691-2 | ISO 3691-4 | ISO 3691-4 | IEC 61508 SIL2 | IEC 61508 SIL3 |
| **AI算力** | - | - | - | ≥100 TOPS | ≥275 TOPS |
| **处理器** | PLC | ARM | ARM+GPU | ARM+GPU | NVIDIA Jetson AGX Orin |
| **典型场景** | 仓储拣选 | 产线配送 | 柔性制造 | 重载车间 | 无人化工厂 |
| **通讯协议** | 有线/低频WiFi | WiFi/5GHz | WiFi 6/5G | 5G/MQTT | 5G/WiFi 6E |

### 4.2 触觉传感器详细规格 (AGV_TACTILE_GRADES)

| 等级 | 阵列尺寸 | ADC分辨率 | 压力范围(kPa) | 采样频率 | 温度感知 | 通信接口 |
|------|---------|----------|--------------|---------|---------|----------|
| **S** | 8×8 | 12bit | 0-500 | 50Hz | 否 | I2C@0x18 |
| **M** | 16×16 | 12bit | 0-1000 | 100Hz | 是 | SPI@50MHz |
| **L** | 24×24 | 14bit | 0-2000 | 200Hz | 是 | SPI@50MHz |
| **XL** | 32×32 | 14bit | 0-5000 | 500Hz | 是 | USB 3.0 |
| **XXL** | 48×48 | 16bit | 0-10000 | 1000Hz | 是 | USB 3.0 |

### 4.3 力觉传感器详细规格 (AGV_FORCE_GRADES)

| 等级 | 轴数 | 力范围(N) | 力矩范围(N·m) | 分辨率(N) | 采样频率 | 通信接口 |
|------|-----|----------|--------------|----------|---------|----------|
| **S** | 3 | ±100 | ±10 | 0.1 | 100Hz | USB HID |
| **M** | 6 | ±200 | ±20 | 0.05 | 500Hz | CAN/EtherCAT |
| **L** | 6 | ±500 | ±50 | 0.02 | 1000Hz | CAN/EtherCAT |
| **XL** | 6 | ±1000 | ±100 | 0.01 | 2000Hz | Ethernet UDP |
| **XXL** | 6 | ±5000 | ±500 | 0.005 | 5000Hz | Ethernet UDP |

### 4.4 IMU传感器详细规格 (AGV_IMU_GRADES)

| 等级 | 型号 | 加速度范围 | 陀螺仪范围 | 采样频率 | 噪声密度(μg/√Hz) | 通信接口 |
|------|-----|----------|-----------|---------|----------------|----------|
| **S** | MPU6050 | ±8g | ±1000°/s | 100Hz | 400 | I2C@100kHz |
| **M** | BMI088 | ±16g | ±2000°/s | 200Hz | 120 | SPI@20MHz |
| **L** | BMI088 | ±24g | ±4000°/s | 500Hz | 60 | SPI@20MHz |
| **XL** | ADIS16470 | ±40g | ±4000°/s | 1000Hz | 20 | SPI@40MHz |
| **XXL** | ADIS16470 | ±80g | ±8000°/s | 2000Hz | 10 | SPI@40MHz |

### 4.5 通信接口规格

| 接口类型 | 速率 | 典型用途 | 支持等级 |
|---------|------|---------|----------|
| I2C | 100-400kHz | 消费级IMU/触觉 | S, M |
| SPI | 20-50MHz | 工业级IMU/力觉 | M, L, XL, XXL |
| USB HID | 12-480Mbps | 传感器集线器 | S, M |
| USB 3.0 | 5Gbps | 高带宽传感器 | XL, XXL |
| CAN | 1Mbps | 工业控制 | M, L |
| EtherCAT | 100Mbps | 实时控制 | M, L |
| Ethernet UDP | 1Gbps | 力觉/视觉传输 | XL, XXL |
| WiFi 6E | 9.6Gbps | 云端通信 | L4, L5 |
| 5G | 10Gbps | 远程控制 | L4, L5 |

### 4.6 控制子系统规格 (AGV五级)

| 参数 | L1 | L2 | L3 | L4 | L5 |
|------|-----|-----|-----|-----|-----|
| **控制频率** | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| **控制模式** | 位置 | 位置+速度 | 位置+速度+阻抗 | 全模态 | 全模态+MPC |
| **避障算法** | 人工势场 | DWA | DWA+VFH+APF | 混合 | 多层融合 |
| **轨迹规划** | 直线 | RRT | RRT*+样条 | MPC+RRT* | MPC+多次RRT* |
| **碰撞响应** | >100ms | <50ms | <20ms | <10ms | <5ms |
| **姿态稳定** | <500ms | <200ms | <100ms | <50ms | <20ms |
| **容错机制** | 无 | 单点 | 多点冗余 | 多点+看门狗 | 多点+看门狗+故障迁移 |
| **实时内核** | 无 | 无 | Xenomai | RT-PREEMPT | Xenomai+FPGA |

### 4.7 感知→控制闭环延迟规格 (AGV五级)

| 阶段 | L1 | L2 | L3 | L4 | L5 |
|------|-----|-----|-----|-----|-----|
| **传感器采样** | 20ms | 10ms | 5ms | 2ms | 1ms |
| **特征提取** | 80ms | 30ms | 15ms | 5ms | 2ms |
| **融合推理** | 30ms | 10ms | 5ms | 2ms | 1ms |
| **决策规划** | 20ms | 10ms | 5ms | 2ms | 1ms |
| **控制计算** | 10ms | 5ms | 2ms | 1ms | 0.5ms |
| **电机响应** | 40ms | 15ms | 5ms | 2ms | 1ms |
| **总延迟** | <200ms | <80ms | <35ms | <15ms | <7ms |

## 5. 模块接口详细设计

### 5.1 传感器管理器接口 (SensorManager)

```python
class SensorManager:
    def __init__(self, grade: str = 'M')
    def open_all(self) -> Dict[str, bool]
    def close_all(self)
    def capture_all(self) -> SensorDataFrame
    def get_modalities(self) -> List[str]
    def is_healthy(self) -> bool
    def get_latencies_ms(self) -> Dict[str, float]
```

### 5.2 控制模块核心接口

```python
class AGVMotionController:
    def set_target_pose(pose: AGVPose) -> None
    def set_target_twist(twist: AGVTwist) -> None
    def step(dt: float) -> np.ndarray  # 返回轮速
    def move_to(x: float, y: float, theta: float, dt: float) -> np.ndarray
    def stop() -> np.ndarray
    def emergency_stop() -> None

class SafetyMonitor:
    def check_velocity(velocity: np.ndarray, dt: float, timestamp: float) -> SafetyStatus
    def check_boundary(position: np.ndarray, timestamp: float) -> SafetyStatus
    def check_force(force_magnitude: float, torque_magnitude: float, timestamp: float) -> SafetyStatus
    def check_collision(collision_detected: bool, timestamp: float) -> SafetyStatus
    def check_all(...) -> SafetyStatus
    def emergency_stop(reason: str) -> None
    def reset_estop() -> None

class TrajectoryPlanner:
    def plan_line(start: Waypoint, end: Waypoint) -> List[TrajectoryPoint]
    def plan_arc(start: Waypoint, end: Waypoint, curvature: float) -> List[TrajectoryPoint]
    def smooth_trajectory(traj: List[TrajectoryPoint]) -> List[TrajectoryPoint]
    def plan_path(waypoints: List[Waypoint]) -> List[TrajectoryPoint]
```

### 5.3 融合网络接口

```python
class CrossModalFusion:
    def forward(multimodal_input: MultimodalInput) -> UnifiedRepresentation
    def encode_vision(vision_feat: torch.Tensor) -> torch.Tensor
    def encode_audio(audio_feat: torch.Tensor) -> torch.Tensor
    def encode_tactile(tactile_feat: torch.Tensor) -> torch.Tensor
    def encode_force(force_feat: torch.Tensor) -> torch.Tensor
    def encode_imu(imu_feat: torch.Tensor) -> torch.Tensor

class ComplementaryFilter:
    def update(measurements: Dict, dt: float) -> np.ndarray
    def get_state() -> np.ndarray
    def reset() -> None

class ExtendedKalmanFilter:
    def initialize(initial_state: np.ndarray) -> None
    def predict(dt: float) -> None
    def correct(measurement: np.ndarray) -> None
    def update(measurements: Dict, dt: float) -> np.ndarray
    def get_state() -> np.ndarray
    def get_covariance() -> np.ndarray
```

## 6. 测试规范

```bash
# 运行所有测试
pytest tests/ -v

# 传感器模块测试 (65项)
pytest tests/sensor_tests.py -v

# 传感器融合测试 (24项)
pytest tests/fusion_tests.py -v

# 控制模块测试 (260项)
pytest tests/control_tests.py -v

# 运行关键测试子集
pytest tests/sensor_tests.py tests/fusion_tests.py tests/control_tests.py -q

# 集成测试
pytest tests/integration_pipeline_tests.py -v
pytest tests/five_grade_pipeline_tests.py -v
```

## 7. 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| v1.50.0 | 2026-04-03 | 完成触觉/力觉/IMU传感器模块及测试套件、1277项测试通过 |
| v1.0 | 2026-04-01 | 初始版本，基础架构完成 |
| v1.1 | 2026-04-02 | 触觉/力觉/IMU模块完成，测试用例完善 |
| v1.2 | 2026-04-02 | AGV五级规格表完善，接口文档更新 |
| v1.4 | 2026-04-03 | GradeAwareSupervisor传感器融合集成测试11项、motor.py语法修复、1183项测试通过 |
| v1.3 | 2026-04-02 | 扩展边界情况测试(NaN/Inf/饱和)，新增41项鲁棒性测试，总计1135项通过 |

---

## 8. 控制模块扩展接口详细设计

### 8.1 Supervisor 模块接口 (supervisor.py)

#### ControlSupervisor (控制器生命周期管理器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(config, grade)` | SupervisorConfig, SupervisorGrade | ControlSupervisor | 按AGV等级初始化 |
| `register_controller(controller)` | ControllerInterface | bool | 注册控制器 |
| `unregister_controller(name)` | str | bool | 注销控制器 |
| `list_controllers()` | - | List[str] | 列出所有控制器 |
| `get_controller(name)` | str | ControllerInterface | 获取控制器 |
| `switch_mode(target_mode)` | ControlMode | bool | 切换控制模式 |
| `trigger_emergency_stop(reason)` | str | None | 触发急停 |
| `release_emergency_stop()` | bool | 解除急停 |
| `get_state()` | - | ControlState | 获取状态 |
| `get_diagnostics()` | - | Dict | 诊断信息 |
| `step_watchdog()` | - | bool | 看门狗心跳 (XL/XXL) |
| `step_fault_tolerance(fault_detected)` | bool | None | 故障容错 (XXL) |
| `get_grade_capabilities()` | - | Dict | 获取等级能力 |
| `register_with_redundancy(controller, modes, is_primary)` | ... | bool | 冗余注册 (L+) |

#### GradeAwareSupervisor (AGV五级感知监管器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(grade, supervisor_id)` | SupervisorGrade, str | GradeAwareSupervisor | 按AGV等级初始化 |
| `register_controller(controller, modes, is_primary)` | ControllerInterface, List, bool | bool | 带冗余注册 |
| `get_active_controller()` | - | ControllerInterface | 获取当前控制器 |
| `get_standby_controller()` | - | ControllerInterface | 获取备用控制器 |
| `switch_with_handover()` | - | bool | 无缝切换 (XL/XXL) |
| `run_diagnostics()` | - | Dict | 运行诊断 |
| `get_health_score()` | - | float | 健康评分 0-1 |

### 8.2 Sensorimotor 模块接口 (sensorimotor.py)

#### SensorimotorIntegration (传感-运动融合)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(config, grade)` | SensorimotorConfig, str | SensorimotorIntegration | 按等级初始化 |
| `open()` | - | bool | 打开所有传感器 |
| `close()` | - | None | 关闭所有传感器 |
| `step(target_force, target_attitude, dt)` | float, Tuple, float | SensorimotorState | 单步融合控制 |
| `capture_all()` | - | Dict | 捕获所有传感器 |
| `get_fused_control()` | - | np.ndarray | 获取融合控制量 |
| `get_control_authority()` | - | Dict | 各模态控制权重 |
| `reset()` | - | None | 重置状态 |
| `is_healthy()` | - | bool | 健康检查 |

#### SensorimotorSimulator (仿真器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `simulate_grasp(object_pos, object_force, num_steps, dt)` | Tuple, float, int, float | List[SensorimotorState] | 仿真抓取任务 |
| `simulate_agv_navigation(trajectory_type, duration_s, dt)` | str, float, float | List[SensorimotorState] | 仿真AGV导航 |
| `get_integration()` | - | SensorimotorIntegration | 获取融合器 |

### 8.3 Multi-Agent 模块接口 (multi_agent.py)

#### MultiAgentCoordinator (多AGV协调器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `add_agent(agent_id, state)` | str, AgentState | bool | 添加AGV |
| `remove_agent(agent_id)` | str | bool | 移除AGV |
| `update_agent_state(agent_id, state)` | str, AgentState | bool | 更新状态 |
| `get_agent_state(agent_id)` | str | AgentState | 获取状态 |
| `form_formation(formation_type, leader_id)` | FormationType, str | bool | 编队形成 |
| `dissolve_formation()` | - | bool | 解散编队 |
| `plan_collision_free_paths()` | - | List[Dict] | 规划无碰撞路径 |
| `compute_collision_risk(agent1, agent2)` | str, str | CollisionRisk | 碰撞风险评估 |
| `coordinate_task(task)` | CoordinationTask | Dict | 协调任务执行 |
| `get_formation_state()` | - | Dict | 获取编队状态 |

### 8.4 Teleop 模块接口 (teleop.py)

#### TeleoperationController (遥操作控制器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(config)` | TeleopConfig | TeleoperationController | 初始化遥操作 |
| `set_master_state(state)` | MasterState | None | 设置主端状态 |
| `get_slave_command()` | - | TeleopCommand | 获取从端命令 |
| `set_slave_state(state)` | SlaveState | None | 设置从端状态 |
| `get_master_feedback()` | - | Dict | 获取主端反馈 |
| `switch_authority(level)` | AuthorityLevel | bool | 切换权限 |
| `engage_shared_control()` | - | bool | 启用共享控制 |
| `disengage_shared_control()` | - | bool | 退出共享控制 |
| `set_latency_compensation(enabled)` | bool | None | 设置延迟补偿 |
| `emergency_stop()` | - | None | 遥操作急停 |

### 8.5 Autotune 模块接口 (autotune.py)

#### AutoTuner (自动调参器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(config)` | TunerConfig | AutoTuner | 初始化调参器 |
| `tune(controller, plant)` | Any, SimulatedPlant | TunerResult | 调参主流程 |
| `set_tuning_method(method)` | TuningMethod | None | 设置调参方法 |
| `set_plant(plant)` | SimulatedPlant | None | 设置被控对象模型 |
| `run_ziegler_nichols()` | - | TunerResult | Z-N调参 |
| `run_chiikawa()` | - | TunerResult | 千川法调参 |
| `run_rms_tuning()` | - | TunerResult | RMS自动调参 |
| `validate_tuning(result)` | TunerResult | bool | 验证调参结果 |

### 8.6 Motor 模块接口 (motor.py)

#### Motor (电机基类)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `open()` | - | bool | 打开电机 |
| `close()` | - | None | 关闭电机 |
| `enable()` | - | bool | 使能电机 |
| `disable()` | - | bool | 禁用电机 |
| `set_mode(mode)` | MotorControlMode | bool | 设置控制模式 |
| `set_position(target)` | float | bool | 设置目标位置 |
| `set_velocity(target)` | float | bool | 设置目标速度 |
| `set_torque(target)` | float | bool | 设置目标力矩 |
| `get_state()` | - | MotorState | 获取电机状态 |
| `get_position()` | - | float | 获取当前位置 |
| `get_velocity()` | - | float | 获取当前速度 |
| `reset_encoder()` | - | bool | 重置编码器 |

#### MotorController (多电机控制器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `add_motor(motor)` | Motor | int | 添加电机 |
| `remove_motor(motor_id)` | bool | 移除电机 |
| `set_positions(targets)` | List[float] | bool | 批量设置位置 |
| `set_velocities(targets)` | List[float] | bool | 批量设置速度 |
| `get_positions()` | - | List[float] | 批量获取位置 |
| `get_velocities()` | - | List[float] | 批量获取速度 |
| `stop_all()` | - | None | 停止所有电机 |

### 8.7 Planner 模块接口 (planner.py)

#### TaskPlanner (任务规划器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `plan_task(task)` | Task | List[Task] | 规划任务分解 |
| `update_task_status(task_id, status)` | str, TaskStatus | bool | 更新任务状态 |
| `get_executable_tasks()` | - | List[Task] | 获取可执行任务 |
| `replan(failed_task_id)` | str | List[Task] | 重规划 |
| `estimate_duration(task)` | Task | float | 估计任务耗时 |

## 9. 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| v1.52.0 | 2026-04-05 | 修复sensorimotor抓取阶段测试、pybullet_sim.py兼容性修复、1311项测试通过 |
| v1.51.0 | 2026-04-03 | 补充AGV五级控制子系统规格表、PyBullet可视化仿真脚本、1277项测试通过 |
| v1.50.0 | 2026-04-03 | 完成触觉/力觉/IMU传感器模块及测试套件、1277项测试通过 |
| v1.0 | 2026-04-01 | 初始版本，基础架构完成 |

## 10. 仿真环境模块接口详细设计

### 10.1 基础仿真接口 (environment.py)

#### RobotSimulator (机器人仿真器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(config, joint_limits_lower, joint_limits_upper)` | SimConfig, np.ndarray, np.ndarray | RobotSimulator | 初始化仿真器 |
| `reset()` | - | np.ndarray | 重置到初始状态 |
| `step(control)` | np.ndarray | Tuple | 执行一步仿真 |
| `get_joint_state()` | - | Tuple[np.ndarray, np.ndarray] | 获取关节状态 |
| `get_end_effector_pose()` | - | np.ndarray | 获取末端位姿 |
| `apply_external_force(force, point)` | np.ndarray, np.ndarray | None | 施加外力 |
| `check_self_collision()` | - | bool | 自碰撞检测 |
| `check_environment_collision()` | - | bool | 环境碰撞检测 |

#### SensorSimulator (传感器仿真器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `add_sensor(sensor_type, config)` | str, Dict | int | 添加传感器 |
| `get_sensor_data(sensor_id)` | int | Dict | 获取传感器数据 |
| `inject_noise(data, noise_level)` | Any, float | Any | 注入噪声 |
| `inject_delay(data, delay)` | Any, float | Any | 注入延迟 |

#### SceneManager (场景管理器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `load_scene(scene_config)` | Dict | bool | 加载场景 |
| `add_object(object_config)` | Dict | str | 添加物体 |
| `remove_object(object_id)` | str | bool | 移除物体 |
| `get_objects()` | - | List[Dict] | 获取所有物体 |

### 10.2 Gymnasium 环境接口 (gym_env.py)

#### SuperModelGymEnv (Gymnasium RL 环境)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `reset(seed)` | int | Tuple | 重置环境 |
| `step(action)` | np.ndarray | Tuple | 执行动作 |
| `render()` | - | np.ndarray | 渲染画面 |
| `close()` | - | None | 关闭环境 |
| `get_observation()` | - | np.ndarray | 获取观测 |
| `compute_reward(state, action, next_state)` | ... | float | 计算奖励 |

#### GymEnvConfig (Gym 环境配置)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dt` | float | 0.01 | 控制周期 (s) |
| `sim_dt` | float | 0.001 | 物理步长 (s) |
| `episode_length` | int | 1000 | 最大episode长度 |
| `num_joints` | int | 6 | 关节数 |
| `obs_type` | str | "full" | 观测类型(full/partial/image) |
| `grade` | str | "M" | AGV等级 |

### 10.3 MuJoCo 仿真接口 (mujoco_sim.py)

#### MuJoCoSimulator (MuJoCo 仿真器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(config)` | MuJoCoConfig | MuJoCoSimulator | 初始化 |
| `load_model(xml_string)` | str | int | 加载模型 |
| `reset()` | - | np.ndarray | 重置 |
| `step(ctrl)` | np.ndarray | None | 仿真一步 |
| `get_state()` | - | Dict | 获取状态 |
| `set_state(data)` | Dict | None | 设置状态 |
| `render(mode)` | str | np.ndarray | 渲染 |

### 10.4 PyBullet 仿真接口 (pybullet_sim.py)

#### PyBulletSimulator (PyBullet 仿真器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(config, gui)` | PyBulletConfig, bool | PyBulletSimulator | 初始化 |
| `load_robot(urdf_path, base_pos)` | str, List | int | 加载机器人 |
| `reset()` | - | List | 重置 |
| `step(ctrl)` | np.ndarray | List | 仿真一步 |
| `get_joint_state()` | - | List | 获取关节状态 |
| `apply_force(force, link)` | np.ndarray, int | None | 施加力 |
| `check_collision()` | - | bool | 碰撞检测 |

### 10.5 AGV 仿真场景接口 (agv_scenarios.py)

#### AGVSimulator (AGV 仿真器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(config)` | AGVPhysicsConfig | AGVSimulator | 初始化 |
| `reset()` | - | AGVState | 重置 |
| `step(velocities, dt)` | np.ndarray, float | AGVState | 步进 |
| `get_state()` | - | AGVState | 获取状态 |
| `apply_disturbance(force, torque)` | float, float | None | 施加扰动 |
| `check_boundary(bounds)` | Tuple | bool | 边界检查 |

#### AGVPurePursuitController (纯追踪控制器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `__init__(lookahead_dist, wheelbase)` | float, float | AGVPurePursuitController | 初始化 |
| `compute_control(state, path)` | AGVState, List | Tuple[float, float] | 计算控制 |
| `update_lookahead(dist)` | float | None | 更新前瞻距离 |

## 11. 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| v1.55.0 | 2026-04-07 | 完善SPEC.md仿真模块接口设计，1332项测试全通过 |
| v1.54.0 | 2026-04-07 | 修复sensor_tests.py中15项API不匹配问题，新增RK3588_NPU_DEPLOYMENT.md |
| v1.53.0 | 2026-04-05 | 触觉/力觉/IMU模块完善，AGV五级规格完整 |
| v1.52.0 | 2026-04-05 | 修复sensorimotor抓取阶段测试、pybullet_sim.py兼容性修复、1311项测试通过 |
| v1.51.0 | 2026-04-03 | 补充AGV五级控制子系统规格表、PyBullet可视化仿真脚本、1277项测试通过 |
| v1.50.0 | 2026-04-03 | 完成触觉/力觉/IMU传感器模块及测试套件、1277项测试通过 |
| v1.0 | 2026-04-01 | 初始版本，基础架构完成 |


## 12. 接口使用示例

### 12.1 触觉传感器使用流程

```python
from src.sensors.tactile import TactileArray, TactileSensorType, TactileCalibration

# 1. 创建传感器 (按AGV等级选择规格)
from src.sensors.tactile import get_tactile_spec
spec = get_tactile_spec('M')  # 16x16阵列, 100Hz

# 2. 初始化
array = TactileArray(
    array_size=(spec['array'][0], spec['array'][1]),
    sensor_type=TactileSensorType.CAPACITIVE,
    sensor_id="gripper_tactile_0"
)

# 3. 打开连接
array.open()

# 4. 标定 (新传感器首次使用)
array.calibrate(zero_pressure=None, known_weights=[0.0, 5.0, 10.0])

# 5. 主循环
for _ in range(100):
    frame = array.capture()               # 捕获触觉帧
    contacts = array.detect_contacts(frame)  # 检测接触
    slip = array.get_slip_signal(frame)   # 计算滑移信号
    quality = array.estimate_grip_quality(frame)  # 抓取质量
    
    if quality['overall'] < 0.3:
        print("抓取不稳定，需要调整!")

# 6. 关闭
array.close()
```

### 12.2 力觉传感器使用流程

```python
from src.sensors.force import ForceTorqueSensor, ForceSensorType, Wrench, VirtualForceSensor

# 1. 创建六维力矩传感器
sensor = ForceTorqueSensor(
    sensor_type=ForceSensorType.SIX_AXIS,
    sensor_id="ft_0",
    ip_address="192.168.1.100",
    ethernet_type="UDP"
)

# 2. 设置工具中心参数 (重力补偿)
sensor.set_tool_center(tool_mass=0.55, tool_com=np.array([0.0, 0.0, 0.05]))

# 3. 打开连接
sensor.open()

# 4. 偏置校准 (无负载状态下)
sensor.calibrate_bias(num_samples=100)

# 5. 主循环
for _ in range(100):
    wrench = sensor.capture()  # 获取六维力旋量
    contact = sensor.detect_contact(wrench, threshold=5.0)  # 接触检测
    payload = sensor.estimate_payload(wrench)  # 负载估计
    
    # 力旋量变换 (传感器坐标 -> 世界坐标)
    R = np.eye(3)  # 旋转矩阵
    t = np.array([0.0, 0.0, 0.1])  # 平移
    world_wrench = wrench.transform(R, t)

sensor.close()
```

### 12.3 IMU传感器使用流程

```python
from src.sensors.imu import IMUSensor, IMUSensorType, PoseEstimator, VirtualIMUSensor

# 1. 创建IMU传感器
imu = IMUSensor(
    sensor_type=IMUSensorType.BMI088,
    sensor_id="imu_0",
    accel_range=16,
    gyro_range=2000,
    sample_rate=200
)

# 2. 打开连接
imu.open()

# 3. 自检
if not imu.self_test():
    print("IMU自检失败!")
    exit(1)

# 4. 偏置校准 (静止状态下)
imu.calibrate_gyro_bias(num_samples=500)

# 5. 创建姿态估计器
estimator = PoseEstimator(algorithm='madgwick', sample_rate=200.0, beta=0.1)

# 6. 主循环
for _ in range(100):
    frame = imu.capture()  # 获取IMU帧
    
    # 姿态更新
    pose = estimator.update(
        accel=frame.accel,
        gyro=frame.gyro,
        mag=frame.mag,
        dt=1.0/200.0
    )
    
    euler = pose.to_euler()  # 欧拉角 [roll, pitch, yaw]
    R = estimator.get_rotation_matrix()  # 旋转矩阵

imu.close()
```

### 12.4 传感运动融合使用流程

```python
from src.control.sensorimotor import SensorimotorIntegration, SensorimotorConfig
from src.sensors.tactile import TactileArray, TactileSensorType
from src.sensors.force import ForceTorqueSensor, ForceSensorType
from src.sensors.imu import IMUSensor, IMUSensorType

# 1. 创建配置
config = SensorimotorConfig(
    grade='M',
    control_freq=100.0,
    force_control_gain=1.0,
    imu_control_gain=0.5
)

# 2. 创建融合器
fusion = SensorimotorIntegration(config=config, grade='M')
fusion.open()

# 3. 主循环
for step in range(1000):
    dt = 0.01
    target_force = 10.0  # N
    target_attitude = (0.0, 0.0, 0.0)  # roll, pitch, yaw
    
    state = fusion.step(
        target_force=target_force,
        target_attitude=target_attitude,
        dt=dt
    )
    
    # 获取融合控制量
    ctrl = fusion.get_fused_control()
    authority = fusion.get_control_authority()
    
    # 健康检查
    if not fusion.is_healthy():
        print("融合器异常!")
        break

fusion.close()
```

### 12.5 AGV五级导航控制使用流程

```python
from src.control.agv import AGVMotionController, AGVConfig
from src.control.navigation import NavigationController, AStarPlanner, OccupancyGrid

# 1. 创建AGV配置 (按等级)
config = AGVConfig(grade='M')
controller = AGVMotionController(config)

# 2. 创建导航系统
grid = OccupancyGrid(
    width=20.0, height=20.0, resolution=0.1,
    origin=(-10.0, -10.0)
)
planner = AStarPlanner(occupancy_grid=grid)
nav = NavigationController(
    planner=planner,
    max_lin_vel=1.5,
    max_ang_vel=2.0
)

# 3. 设置障碍物
grid.set_obstacle(2.0, 3.0, radius=0.5)
grid.set_obstacle(5.0, 5.0, radius=0.3)

# 4. 规划路径
start = np.array([0.0, 0.0, 0.0])  # x, y, theta
goal = np.array([8.0, 8.0, 0.0])
nav.plan_to_goal(start, goal)

# 5. 导航主循环
for step in range(1000):
    dt = 0.01
    current_pose = controller.get_pose()
    
    velocity = nav.update(current_pose, dt)
    controller.set_target_twist(velocity)
    wheel_speeds = controller.step(dt)
    
    if nav.get_progress() > 0.99:
        print("到达目标!")
        break

controller.stop()
```

## 13. 数据流与状态机

### 13.1 传感器数据采集流程

```
┌─────────────┐     open()      ┌──────────────┐
│   用户代码   │ ──────────────▶│  SensorMgr   │
└─────────────┘                └──────┬───────┘
                                      │ open_all()
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
              ┌───────────┐    ┌───────────┐    ┌───────────┐
              │  IMU      │    │  Tactile  │    │  Force    │
              │  Sensor   │    │  Array    │    │  Torque   │
              └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
                    │ capture()       │ capture()      │ capture()
                    ▼                 ▼                 ▼
              ┌───────────┐    ┌───────────┐    ┌───────────┐
              │  IMUFrame │    │ Tactile   │    │  Wrench   │
              │  accel,   │    │  Frame     │    │  6-axis   │
              │  gyro,    │    │ pressure   │    │  force    │
              │  mag      │    │  map       │    │  torque   │
              └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      ▼
                           ┌──────────────────┐
                           │  capture_all()  │
                           │  SensorDataFrame│
                           └──────────────────┘
                                      │
                                      ▼
                           ┌──────────────────┐
                           │ CrossModalFusion│
                           │  跨模态融合网络  │
                           └────────┬─────────┘
                                    │ forward()
                                    ▼
                           ┌──────────────────┐
                           │ UnifiedRep        │
                           │ 融合表征向量      │
                           └────────┬─────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
           ┌────────────┐  ┌────────────┐  ┌────────────┐
           │  感知决策   │  │ 运动控制   │  │  自主学习   │
           └────────────┘  └────────────┘  └────────────┘
```

### 13.2 AGV五级状态机

```
                    ┌─────────────────┐
                    │   INITIALIZED   │
                    │   系统初始化     │
                    └────────┬────────┘
                             │ startup()
                             ▼
                    ┌─────────────────┐
         ┌──────────│   IDLE          │
         │          │   待机/就绪     │
         │          └────────┬────────┘
         │                   │ start_navigation() / start_patrol()
         │                   ▼
         │          ┌─────────────────┐
         │          │   NAVIGATING     │
         │          │   自主导航中     │
         │          └────────┬────────┘
         │                   │ obstacle_detected / emergency
         │                   ▼
         │          ┌─────────────────┐
         │          │   AVOIDING      │
         │          │   避障中        │
         │          └────────┬────────┘
         │                   │ obstacle_cleared
         │                   ▼
         │          ┌─────────────────┐          emergency
         │          │   PAUSED        │◀─────────────────┐
         │          │   暂停          │                   │
         │          └────────┬────────┘                   │
         │                   │ resume()                    │
         │                   └─────────────────────────────┘
         │
         │          ┌─────────────────┐
         └─────────▶│   ERROR          │
                    │   故障           │
                    └────────┬────────┘
                             │ reset() / recover()
                             ▼
                    ┌─────────────────┐
                    │   IDLE          │
                    └─────────────────┘

    XXL等级额外状态:
                    ┌─────────────────┐
                    │   FAULT_TOLERANT│
                    │   容错运行      │
                    └────────┬────────┘
                             │ primary_failed
                             ▼
                    ┌─────────────────┐
                    │   BACKUP_ACTIVE │
                    │   备用接管      │
                    └─────────────────┘
```

### 13.3 传感器融合数据流

```
IMU原始数据          触觉原始数据         力觉原始数据
    │                    │                   │
    ▼                    ▼                   ▼
┌────────┐          ┌────────┐         ┌────────┐
│ 校准   │          │ 校准   │         │ 校准   │
│ 去偏置 │          │ 去噪   │         │ 去偏置 │
└───┬────┘          └───┬────┘         └───┬────┘
    │                   │                   │
    ▼                   ▼                   ▼
┌────────┐          ┌────────┐         ┌────────┐
│Madgwick│          │ 接触   │         │ 工具   │
│ AHRS   │          │ 检测   │         │ 补偿   │
└───┬────┘          └───┬────┘         └───┬────┘
    │                   │                   │
    │    ┌──────────────┼───────────────────┘
    │    │              │
    ▼    ▼              ▼
┌──────────────────────────┐
│    互补滤波 / EKF        │
│    姿态 + 接触 融合      │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│    具身控制量输出        │
│  力控 + 位控 + 阻抗控    │
└──────────────────────────┘
```

## 14. 错误处理与异常规范

### 14.1 传感器异常分类

| 异常类型 | 异常类 | 触发条件 | 处理策略 |
|---------|--------|---------|---------|
| 连接失败 | `SensorConnectionError` | open()失败 | 重试3次，报告 |
| 数据超时 | `SensorTimeoutError` | capture()超时 | 使用上次数据/降级 |
| 数据饱和 | `SensorSaturationError` | 测量值超范围 | 切换量程/报警 |
| 校准失败 | `CalibrationError` | 标定参数异常 | 使用默认参数 |
| 通信错误 | `CommunicationError` | 总线通信失败 | 切换备通道 |
| 硬件故障 | `HardwareFaultError` | 自检失败 | 进入安全模式 |

### 14.2 控制异常处理

| 异常类型 | 触发条件 | 响应级别 |
|---------|---------|---------|
| 电机过流 | 电流 > 额定150% | 降低功率 |
| 电机过热 | 温度 > 80°C | 降频运行 |
| 碰撞检测 | 接触力 > 阈值 | 立即停止 |
| 位置超限 | 超出边界 | 触发急停 |
| 通信中断 | CAN/ETH断开 | 保持最后姿态 |
| 看门狗超时 | 100ms无心跳 | 故障转移 |

### 14.3 AGV等级与故障容忍

| 等级 | 单点故障处理 | 多点故障处理 | 安全状态 |
|------|------------|------------|---------|
| **S** | 报警停机 | 人工干预 | 抱闸锁定 |
| **M** | 本地恢复 | 报警停机 | 缓慢停止 |
| **L** | 自动切换备机 | 本地恢复 | 受控停止 |
| **XL** | 热备切换 | 自动切换备机 | 安全位置停止 |
| **XXL** | 热备+无损转移 | 故障隔离继续运行 | 零速悬停 |

## 16. 表面跟踪与装配控制器接口

### 16.1 SurfaceFollowingController (表面跟踪控制器)

触觉引导的表面跟踪控制，支持擦拭、打磨、抛光、扫描等任务。

**位置**: `src/control/embodied_control.py`

**构造函数参数**:
```python
SurfaceFollowingController(
    grade='M',              # AGV五级等级
    follow_mode='admittance',  # constant_force | admittance | impedance | adaptive
    nominal_force=5.0,      # N, 期望法向接触力
    nominal_velocity=0.05, # m/s, 标称跟踪速度
    force_deadband=1.0,   # N, 力控制死区
)
```

**核心方法**:
```python
# 主控制计算
result = ctrl.compute_control(pressure_map, current_force, dt)
# Returns: {velocity, normal_force_error, surface_normal, tangent_direction}

# 表面法向估计 (基于压力梯度)
normal = ctrl.estimate_surface_normal(pressure_map)

# 接触质量评估
quality = ctrl.compute_contact_quality(pressure_map)
# Returns: {contact_ratio, uniformity, quality, is_good_contact}

# 状态查询
status = ctrl.get_status()
# Returns: {is_following, total_distance_m, surface_normal, mode}
```

**控制模式对比**:

| 模式 | 适用场景 | 特点 |
|------|---------|------|
| `constant_force` | S级AGV | 开环速度+固定下压力 |
| `admittance` | M级AGV | 触觉反馈+导纳控制 |
| `impedance` | L级AGV | 实时力位混合+阻抗控制 |
| `adaptive` | XL/XXL级 | 自适应刚度+多模态融合 |

**AGV五级表面跟踪能力**:

| 等级 | 跟踪速度 | 力控制精度 | 表面适应性 |
|------|---------|-----------|----------|
| S | 0.02m/s | ±2N | 平面 |
| M | 0.05m/s | ±1N | 平面+轻度曲面 |
| L | 0.10m/s | ±0.5N | 曲面+不规则表面 |
| XL | 0.20m/s | ±0.2N | 复杂曲面+自适应 |
| XXL | 0.50m/s | ±0.1N | MPC预测+全适应性 |

---

### 16.2 AssemblyController (精密装配控制器)

孔轴配合/插入装配控制，支持peg-in-hole、螺纹连接、卡扣装配等。

**位置**: `src/control/embodied_control.py`

**构造函数参数**:
```python
AssemblyController(
    grade='M',                 # AGV五级等级
    hole_tolerance=1.0,       # mm, 孔径公差
    insertion_depth=10.0,      # mm, 插入深度
    max_insertion_force=20.0, # N, 最大插入压力
    search_force=3.0,          # N, 搜索时的法向力
    search_pattern='spiral',   # spiral | raster | random
)
```

**装配阶段** (AssemblyPhase枚举):
```
IDLE → APPROACH → SEARCH → INSERT → SEAT → VERIFY → COMPLETE
                                              ↘ FAILED
```

**核心方法**:
```python
# 开始装配任务
ctrl.start_assembly(target_position, phase='approach')

# 主更新 (在控制循环中调用)
result = ctrl.update(current_position, current_force, lateral_force, dt)
# Returns: {phase, velocity, progress, should_stop, message}

# 搜索运动计算
motion = ctrl.compute_search_motion(dt)

# 插入控制
velocity = ctrl.compute_insertion_control(current_force, lateral_force, dt)

# 压合控制
velocity = ctrl.compute_seating_control(contact_force, dt)

# 统计查询
stats = ctrl.get_stats()
# Returns: {total_assemblies, success_rate, current_phase, insertion_progress}
```

**AGV五级装配能力**:

| 等级 | 定位精度 | 插入速度 | 力控精度 | 最大负载 |
|------|---------|---------|---------|----------|
| S | ±5mm | 0.5mm/s | ±3N | 5kg |
| M | ±1mm | 1mm/s | ±2N | 20kg |
| L | ±0.3mm | 2mm/s | ±1N | 50kg |
| XL | ±0.1mm | 5mm/s | ±0.5N | 200kg |
| XXL | ±0.01mm | 10mm/s | ±0.2N | 1000kg |

**故障检测**:
- 卡阻检测: 连续高力(>80% max)超过30个周期
- 插入失败: 反复卡阻5次以上
- 偏斜检测: 侧向力 > 3×search_force

---

## 17. AGV五级具身控制完整规格表

### 17.1 具身感知规格

| 感知模态 | S | M | L | XL | XXL |
|---------|:--:|:--:|:--:|:--:|:--:|
| **触觉阵列** | 8×8 | 16×16 | 24×24 | 32×32 | 48×48 |
| **触觉帧率** | 30Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| **触觉延迟** | 50ms | 20ms | 10ms | 5ms | 2ms |
| **力觉轴数** | 0 | 6轴 | 6轴 | 6轴+ | 6轴+冗余 |
| **力觉量程** | — | ±200N | ±500N | ±1000N | ±5000N |
| **力觉精度** | — | ±0.5N | ±0.2N | ±0.1N | ±0.05N |
| **IMU等级** | MPU6050 | BMI088 | BMI088 | ADIS16470 | ADIS16470×2 |
| **IMU采样** | 100Hz | 200Hz | 500Hz | 1000Hz | 2000Hz |
| **姿态精度** | ±2° | ±0.5° | ±0.1° | ±0.05° | ±0.01° |

### 17.2 具身控制规格

| 控制特性 | S | M | L | XL | XXL |
|---------|:--:|:--:|:--:|:--:|:--:|
| **控制频率** | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| **控制延迟** | 50ms | 20ms | 10ms | 5ms | 2ms |
| **力控制模式** | 固定值 | 导纳 | 阻抗 | 自适应阻抗 | MPC |
| **表面跟踪** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **精密装配** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **抓取适应** | ✗ | 粗 | 中等 | 精细 | 超精细 |
| **姿态稳定** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **滑移恢复** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **碰撞响应** | 100ms | 50ms | 20ms | 10ms | 5ms |

### 17.3 具身任务执行规格

| 任务类型 | S | M | L | XL | XXL |
|---------|:--:|:--:|:--:|:--:|:--:|
| **抓取-放置** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **推-拉操作** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **表面追踪** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **精密插入** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **表面抛光** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **螺纹连接** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **多指灵巧操作** | ✗ | ✗ | ✗ | ✓ | ✓ |
| **MPC预测控制** | ✗ | ✗ | ✗ | ✗ | ✓ |

### 17.4 健康监控与降级策略

| 指标 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **触觉自检** | 启动时 | 周期 | 实时 | 实时 | 实时 |
| **力觉自检** | — | 启动时 | 周期 | 实时 | 实时 |
| **IMU自检** | 启动时 | 周期 | 实时 | 实时 | 实时 |
| **降级模式** | 无 | 2级 | 3级 | 4级 | 全冗余 |
| **故障隔离** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **热备切换** | ✗ | ✗ | ✗ | ✓ | ✓ |
| **预测维护** | ✗ | ✗ | ✗ | ✓ | ✓ |

### 17.5 融合编码维度 (跨模态)

| 融合阶段 | S | M | L | XL | XXL |
|---------|:--:|:--:|:--:|:--:|:--:|
| **Early Fusion** | 64d | 128d | 256d | 512d | 1024d |
| **Mid Fusion** | 32d | 128d | 256d | 512d | 1024d |
| **Late Fusion** | 16d | 64d | 128d | 256d | 512d |
| **总融合维度** | 128d | 512d | 1024d | 2048d | 4096d |

---

## 15. 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| v2.14.0 | 2026-04-09 | 新增SurfaceFollowingController(表面跟踪)与AssemblyController(精密装配)控制器；AGV五级具身控制完整规格表；76项新增测试全通过 |
| v2.13.0 | 2026-04-09 | 完善SPEC.md接口使用示例(12章)、数据流与状态机(13章)、错误处理规范(14章)；触觉/力觉/IMU模块完善，378项传感器+融合测试全通过 |
| v2.12.0 | 2026-04-09 | 版本同步，清理临时文件，1937项测试全通过 |
| v2.11.0 | 2026-04-08 | 新增自主巡逻控制模块，多点巡逻+动态避障+传感器融合 |
| v2.10.0 | 2026-04-08 | 触觉/力觉/IMU三传感器融合集成测试 |
| v2.09.0 | 2026-04-08 | 具身任务执行器扩展 |
| v2.08.0 | 2026-04-07 | 新增AGV五级控制参数完整指南 |
| v1.55.0 | 2026-04-07 | 完善SPEC.md仿真模块接口设计 |
| v1.0 | 2026-04-01 | 初始版本，基础架构完成 |

---

## 18. 系统集成架构与数据流

### 18.1 整体架构层次

```
┌─────────────────────────────────────────────────────────────┐
│                    SuperModel 系统架构                        │
├─────────────────────────────────────────────────────────────┤
│  具身智能层 (Embodied Intelligence)                          │
│  ┌─────────────┐ ┌──────────────┐ ┌─────────────────────┐  │
│  │ DreamerAgent │ │ WorldModel   │ │ AutonomousLearning  │  │
│  └─────────────┘ └──────────────┘ └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  认知融合层 (Cognitive Fusion)                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         CrossModalFusion (跨模态注意力融合网络)          │  │
│  │  Vision + Audio + Tactile + Force + IMU + Encoders  │  │
│  └──────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  感知层 (Perception)                                        │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐  │
│  │ Vision │ │ Audio  │ │Tactile │ │ Force  │ │  IMU   │  │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘  │
│  ┌────────────────────────────────────────────────────┐    │
│  │              SensorManager (传感器管理器)             │    │
│  └────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│  执行层 (Control - AGV五级)                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ SafetyController → Supervisor → Trajectory → Motor   │  │
│  │    ↓                                                    │  │
│  │ Navigation / Patrol / Teleop / MPC / Impedance       │  │
│  └──────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  硬件抽象层 (Hardware Abstraction)                          │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────────────┐      │
│  │ RK3588 │ │  NPU   │ │ GPIO   │ │ DiguRobot/ROS2 │      │
│  └────────┘ └────────┘ └────────┘ └────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 18.2 AGV五级数据流延迟预算

| 阶段 | S级 | M级 | L级 | XL级 | XXL级 |
|------|-----|-----|-----|------|-------|
| **传感器采样** | 20ms | 10ms | 5ms | 2ms | 1ms |
| **感知预处理** | 30ms | 15ms | 8ms | 3ms | 1ms |
| **特征提取(CNN)** | 50ms | 20ms | 10ms | 4ms | 2ms |
| **跨模态融合** | 30ms | 10ms | 5ms | 2ms | 1ms |
| **世界模型推理** | 40ms | 20ms | 10ms | 5ms | 2ms |
| **运动规划** | 20ms | 10ms | 5ms | 2ms | 1ms |
| **控制计算** | 10ms | 5ms | 2ms | 1ms | 0.5ms |
| **电机响应** | 40ms | 15ms | 5ms | 2ms | 1ms |
| **总延迟** | **240ms** | **95ms** | **45ms** | **18ms** | **7.5ms** |

### 18.3 传感器-控制接口映射 (AGV五级)

| 控制模式 | S级接口 | M级接口 | L级接口 | XL级接口 | XXL级接口 |
|---------|---------|---------|---------|---------|---------|
| **位置控制** | JointTrajectory | JointTrajectory | JointTrajectory | JointTrajectory | JointTrajectory |
| **速度控制** | TwistCommand | TwistCommand | TwistCommand | TwistCommand | TwistCommand |
| **力矩控制** | ✗ | Wrench | Wrench | Wrench | Wrench |
| **阻抗控制** | ✗ | ImpedanceParams | ImpedanceParams | ImpedanceParams | ImpedanceParams |
| **MPC** | ✗ | ✗ | JointSpaceMPC | JointSpaceMPC | CartesianMPC |
| **触觉伺服** | ✗ | TactileServo | TactileServo | TactileServo | TactileServo |
| **遥操作** | ✗ | MasterState | MasterState | MasterState | SharedControl |

### 18.4 AGV五级模块兼容性矩阵

| 模块 | S | M | L | XL | XXL | 依赖 |
|------|:--:|:--:|:--:|:--:|:--:|------|
| TactileArray | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| ForceTorqueSensor | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| IMUSensor | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| PoseEstimator | ✓ | ✓ | ✓ | ✓ | ✓ | IMU |
| ComplementaryFilter | ✓ | ✓ | ✓ | ✓ | ✓ | IMU |
| ExtendedKalmanFilter | ✗ | ✓ | ✓ | ✓ | ✓ | IMU+Encoder |
| CrossModalFusion | ✓ | ✓ | ✓ | ✓ | ✓ | All sensors |
| AttitudeStabilizer | ✗ | ✓ | ✓ | ✓ | ✓ | IMU |
| TactileServoController | ✗ | ✓ | ✓ | ✓ | ✓ | Tactile |
| ForceController | ✗ | ✓ | ✓ | ✓ | ✓ | Force |
| AGVMotionController | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| SafetyController | ✓ | ✓ | ✓ | ✓ | ✓ | All |
| GradeAwareSupervisor | ✗ | ✗ | ✓ | ✓ | ✓ | Controllers |
| JointSpaceMPC | ✗ | ✗ | ✓ | ✓ | ✓ | DynamicsModel |
| CartesianMPC | ✗ | ✗ | ✗ | ✗ | ✓ | DynamicsModel |
| TeleoperationController | ✗ | ✓ | ✓ | ✓ | ✓ | Network |
| MultiAgentCoordinator | ✗ | ✗ | ✗ | ✓ | ✓ | AGV |
| PatrolController | ✗ | ✓ | ✓ | ✓ | ✓ | Navigation |
| SensorimotorIntegration | ✗ | ✓ | ✓ | ✓ | ✓ | All sensors |
| EmbodiedController | ✗ | ✓ | ✓ | ✓ | ✓ | All above |

### 18.5 五级传感器配置与融合策略

```
S级 (简单触觉+力觉):
  传感器: TactileArray(8x8) + ForceTorqueSensor(3轴)
  融合: TactileFrame + Wrench → 简单加权融合
  控制: 位置环 + 触觉反馈

M级 (完整触觉+力觉+IMU):
  传感器: TactileArray(16x16) + ForceTorqueSensor(6轴) + IMUSensor(BMI088)
  融合: ComplementaryFilter(IMU) + 触觉-力觉加权融合
  控制: 位置+速度环 + 阻抗 + 姿态稳定

L级 (高精度传感+基础MPC):
  传感器: TactileArray(24x24) + ForceTorqueSensor(6轴) + IMUSensor(BMI088×2)
  融合: EKF(IMU) + 触觉-力觉-IMU联合融合
  控制: JointSpaceMPC + 阻抗 + 姿态稳定 + 触觉伺服

XL级 (高性能传感+高级MPC):
  传感器: TactileArray(32x32) + ForceTorqueSensor(6轴) + IMUSensor(ADIS16470×2)
  融合: EKF + CrossModalFusion(Transformer) + 预测融合
  控制: JointSpaceMPC + 全阻抗 + 多级安全 + 触觉伺服

XXL级 (超高性能传感+预测控制):
  传感器: TactileArray(48x48) + ForceTorqueSensor(6轴) + IMUSensor(ADIS16470×4)
  融合: 分布式EKF + CrossModalFusion(Large) + 在线持续学习
  控制: CartesianMPC + 全阻抗 + 全冗余安全 + 多机协同 + 触觉伺服
```

### 18.6 关键接口时序图 (M级抓取任务)

```
时间轴 ─────────────────────────────────────────────────────►

传感器采集:
  IMU:       ──[10ms采样]──[10ms采样]──[10ms采样]──►
  Force:     ──[2ms采样]──[2ms采样]──[2ms采样]──►
  Tactile:   ──[10ms采样]──[10ms采样]──[10ms采样]──►

数据融合:
  PoseEst:   ──[15ms/帧]────────────────────────────────►
  Contact:   ──[5ms/帧]────────────────────────────────►
  Fusion:    ──[20ms/帧]────────────────────────────────►

控制计算:
  Supervisor:──[2ms]────────────────────────────────────►
  Trajectory:──[5ms]────────────────────────────────────►
  MotorCmd:   ──[1ms]────────────────────────────────────►

执行反馈:
  Motor:     ─────────────────[15ms响应]────────────────►
```

### 18.7 AGV五级典型部署场景

| 场景 | 推荐等级 | 核心模块组合 | 典型配置 |
|------|---------|------------|---------|
| 实验室研究 | S/M | Vision+IMU+AGV | RPi4B/RK3588, 单目 |
| 仓储物流 | M/L | Vision+IMU+Navigation+Patrol | RK3588/OrinNX, 双目D455 |
| 柔性制造 | L/XL | Vision+Force+IMU+Assembly+MPC | OrinNX/OrinAGX, 力控+双目 |
| 重载车间 | XL/XXL | Vision+Force+Tactile+IMU+MPC+MultiAgent | OrinAGX×2+GPU, 全冗余 |
| 无人化工厂 | XXL | 全模态+CrossModalFusion+Dreamer+WorldModel | OrinAGX×2+GPU+NPU, 全冗余+5G |

---

## 19. 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| v2.23.0 | 2026-04-10 | 新增sensorimotor_integration_tests.py(30项); 新增AGV五级完整规格增强版文档; 2009项测试全通过 |
| v2.15.0 | 2026-04-09 | 补充SPEC.md第18章: 系统集成架构与数据流, AGV五级模块兼容性矩阵, 接口时序图; 378项传感器+融合测试全通过 |
| v2.14.0 | 2026-04-09 | 新增SurfaceFollowingController与AssemblyController; AGV五级具身控制完整规格表 |
| v2.13.0 | 2026-04-09 | 完善SPEC.md接口使用示例(12章)、数据流与状态机(13章)、错误处理规范(14章) |
| v2.12.0 | 2026-04-09 | 版本同步，1937项测试全通过 |
