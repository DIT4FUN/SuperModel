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

## 19. 技能调度器接口 (SkillDispatcher)

### 19.1 概述

技能调度器 (SkillDispatcher) 是跨模态技能协调执行器，负责管理、调度和执行机器人的高层技能。它支持多技能并发调度、资源冲突仲裁、技能依赖管理和执行监控，并按 AGV 五级 (S/M/L/XL/XXL) 规格适配。

### 19.2 核心数据类型

```python
class SkillPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

class SkillStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ResourceType(Enum):
    MOTOR = "motor"
    SENSOR_VISION = "sensor_vision"
    SENSOR_FORCE = "sensor_force"
    SENSOR_TACTILE = "sensor_tactile"
    SENSOR_IMU = "sensor_imu"
    POSITION = "position"
    GRIPPER = "gripper"

@dataclass
class SkillRequest:
    skill_name: str
    params: Dict[str, Any]
    priority: SkillPriority = SkillPriority.NORMAL
    timeout_sec: float = 30.0
    request_id: str = ""
    dependencies: List[str] = field(default_factory=list)

@dataclass
class SkillResult:
    request_id: str
    skill_name: str
    status: SkillStatus
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_sec: float = 0.0
    resources_used: Set[ResourceType] = field(default_factory=set)

@dataclass
class SkillDefinition:
    name: str
    execute_fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    required_resources: Set[ResourceType]
    estimated_duration_sec: float = 5.0
    max_retries: int = 3
    grade_requirement: str = "S"
```

### 19.3 SkillDispatcher 接口

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `register_skill(skill)` | SkillDefinition | bool | 注册技能 |
| `unregister_skill(skill_name)` | str | bool | 注销技能 |
| `dispatch(request)` | SkillRequest | SkillResult | 调度技能请求 |
| `cancel(request_id)` | str | bool | 取消技能请求 |
| `get_status(skill_name)` | str | SkillStatus | 获取技能状态 |
| `get_result(request_id)` | str | SkillResult | 获取执行结果 |
| `get_stats()` | - | Dict | 获取调度统计 |

### 19.4 AGV五级技能调度规格

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|---|
| **最大并发技能数** | 1 | 2 | 3 | 4 | 6 |
| **默认超时(s)** | 30 | 20 | 15 | 10 | 5 |
| **技能注册上限** | 5 | 10 | 20 | 50 | 100 |
| **执行监控** | 否 | 是 | 是 | 是 | 是 |

### 19.5 预定义技能工厂

| 工厂函数 | 技能名称 | 所需资源 | 预估时长 |
|---------|---------|---------|---------|
| `create_grasp_skill(force_ctrl, tactile_ctrl)` | grasp | MOTOR + SENSOR_FORCE + SENSOR_TACTILE | 3.0s |
| `create_navigate_skill(nav_ctrl, avoider)` | navigate | MOTOR + POSITION + SENSOR_VISION | 10.0s |
| `create_place_skill(motor_ctrl)` | place | MOTOR + GRIPPER | 2.0s |

### 19.6 资源锁定机制

调度器在技能执行期间锁定其所需资源 (`ResourceType`)，防止冲突技能并发执行。资源在技能完成 (COMPLETED/FAILED) 后自动释放。

### 19.7 使用示例

```python
from src.control.skill_dispatcher import (
    SkillDispatcher, SkillRequest, SkillPriority, ResourceType,
    create_skill_dispatcher, create_grasp_skill
)

# 按AGV等级创建调度器
dispatcher = create_skill_dispatcher('M')

# 注册预定义技能
dispatcher.register_skill(create_grasp_skill(force_ctrl, tactile_ctrl))

# 提交抓取请求
request = SkillRequest(
    skill_name='grasp',
    params={'target_position': (0.5, 0.3, 0.1), 'grasp_force': 10.0},
    priority=SkillPriority.HIGH
)
result = dispatcher.dispatch(request)
print(f"Grasp {result.status}: {result.output}")
```

---

## 20. AGV五级规格总表

### 20.1 整车系统规格

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **负载能力** | 30kg | 100kg | 300kg | 600kg | 1200kg |
| **自重** | 15kg | 35kg | 80kg | 150kg | 300kg |
| **最大总重** | 45kg | 135kg | 380kg | 750kg | 1500kg |
| **车体尺寸** | 0.4×0.3×0.12m | 0.6×0.4×0.15m | 0.8×0.6×0.2m | 1.0×0.7×0.25m | 1.2×0.9×0.3m |
| **轮子配置** | 2轮驱动 | 2轮驱动 | 4轮驱动 | 4轮驱动 | 4轮驱动 |
| **轮子直径** | 100mm | 140mm | 140mm | 165mm | 200mm |
| **电机类型** | 57步进 | 5.5寸轮毂150W | 5.5寸轮毂150W×2 | 6.5寸轮毂200W×2 | 7.5寸轮毂300W×4 |
| **最高速度** | 0.5m/s | 1.5m/s | 2.0m/s | 2.5m/s | 3.0m/s |
| **最大扭矩** | 5Nm | 15Nm | 30Nm | 60Nm | 120Nm |
| **定位精度** | ±10mm | ±5mm | ±3mm | ±1mm | ±0.5mm |
| **防护等级** | IP20 | IP30 | IP54 | IP65 | IP67 |
| **典型价格** | ¥5-15K | ¥15-50K | ¥50-150K | ¥150-500K | >¥500K |
| **典型场景** | 实验室/桌面 | 小型仓储 | 中型产线 | 工业车间 | 重载车间 |

### 20.2 感知子系统规格

| 模态 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **相机分辨率** | 640×480 | 1280×720 | 1920×1080 | 2K+事件相机 | 4K+3D LiDAR |
| **相机类型** | 单目 | 双目D435i | 双目D455 | 双目+事件 | 多目+深度 |
| **帧率** | 30fps | 60fps | 60fps | 120fps | 200fps |
| **麦克风** | 1ch | 2ch阵列 | 4ch阵列 | 6ch阵列 | 8ch阵列 |
| **触觉阵列** | 8×8 | 16×16 | 24×24 | 32×32 | 48×48 |
| **触觉分辨率** | 12bit | 12bit | 14bit | 14bit | 16bit |
| **触觉频率** | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| **力觉轴数** | 3轴 | 6轴 | 6轴 | 6轴 | 6轴 |
| **力觉量程** | ±100N | ±200N | ±500N | ±1000N | ±5000N |
| **力矩量程** | ±10Nm | ±20Nm | ±50Nm | ±100Nm | ±500Nm |
| **IMU型号** | MPU6050 | BMI088 | BMI088 | ADIS16470 | ADIS16470 |
| **IMU采样率** | 100Hz | 200Hz | 500Hz | 1000Hz | 2000Hz |
| **IMU噪声密度** | 400μg/√Hz | 120μg/√Hz | 60μg/√Hz | 20μg/√Hz | 10μg/√Hz |
| **融合编码器** | 128d | 256d | 512d | 768d | 1024d |

### 20.3 控制系统规格

| 控制器 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **控制周期** | 50ms | 20ms | 10ms | 5ms | 1ms |
| **PID调参** | 手动 | 半自动 | 自动(Z-N) | 自适应 | 自适应+MPC |
| **轨迹规划** | 直线插补 | 梯形速度 | S曲线 | RRT | RRT*+MinimumSnap |
| **避障算法** | 固定阈值 | VFH | APF | DWA | DWA+VFH融合 |
| **安全等级** | PLD | PLd | PLe | PLe+SIL2 | PLe+SIL3 |
| **力控带宽** | - | 5Hz | 20Hz | 50Hz | 100Hz |
| **阻抗控制** | - | 基础 | 标准 | 自适应 | 自适应+迭代学习 |
| **MPC预测步数** | - | 10步 | 20步 | 50步 | 100步 |
| **最大并发技能** | 1 | 2 | 3 | 4 | 6 |
| **技能超时** | 60s | 45s | 30s | 15s | 5s |

### 20.4 计算平台规格

| 平台 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **处理器** | Raspberry Pi 4 | RK3588 | Jetson Nano | Jetson Orin NX | Jetson AGX Orin |
| **AI算力** | 4 TOPS | 6 TOPS | 40 TOPS | 100 TOPS | 275 TOPS |
| **CPU** | 4×Cortex-A72 | 4×A76+4×A55 | 4×Cortex-A78 | 8×Cortex-A78AE | 12×Cortex-A78AE |
| **内存** | 4GB LPDDR4 | 8GB LPDDR4X | 8GB LPDDR5 | 16GB LPDDR5 | 64GB LPDDR5 |
| **存储** | 32GB eMMC | 64GB eMMC | 128GB NVMe | 256GB NVMe | 512GB NVMe |
| **通信** | WiFi5/BT5.0 | WiFi6/BT5.2 | WiFi6E | WiFi6E+5G | WiFi6E+5G+LoRa |
| **ROS版本** | ROS2 Humble | ROS2 Humble | ROS2 Iron | ROS2 Iron | ROS2 Jazzy |
| **实时内核** | - | PREEMPT_RT | PREEMPT_RT | Xenomai | Xenomai+Fusion |

### 20.5 电源系统规格

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **工作电压** | 12V DC | 24V DC | 24V DC | 48V DC | 48V DC |
| **电池容量** | 2Ah | 10Ah | 20Ah | 40Ah | 80Ah |
| **续航时间** | 2h | 4h | 6h | 8h | 12h |
| **充电方式** | 手动 | 自动回充 | 自动回充 | 自动换电 | 自动换电 |
| **充电时间** | 2h | 3h | 4h | 2h(换电) | 1h(换电) |
| **峰值功率** | 100W | 500W | 1000W | 2000W | 4000W |
| **功耗模式** | 常规 | 常规+节能 | 常规+节能+深度睡眠 | 多档 | 多档+能量回收 |

### 20.6 自主学习子系统规格

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **学习方法** | 模仿学习 | 模仿+RRL | RL+DAPER | 完整Dreamer | 完整Dreamer+HER |
| **世界模型** | - | MLP | Transformer | Transformer+Attention | Transformer+全球建模 |
| **探索策略** | 随机 | Gaussian | 广义 | SAC+自动适应 | SAC+HER+课程 |
| **收敛样本数** | 10K | 50K | 200K | 500K | 1M+ |
| **在线学习** | - | 离线批量 | 离线批量 | 持续学习 | 持续学习+迁移 |
| **技能库容量** | 5个 | 15个 | 50个 | 100个 | 200个+ |
| **多任务学习** | - | 2任务 | 5任务 | 10任务 | 20任务 |

### 20.7 接口与扩展规格

| 接口 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **CAN总线** | CAN2.0 | CAN FD | CAN FD | 双CAN FD | 双CAN FD+TT-CAN |
| **Ethernet** | 100M | 1G | 1G | 2.5G | 10G |
| **USB** | USB3.0×1 | USB3.0×2 | USB3.2×2 | USB3.2×4 | USB4×4 |
| **显示** | - | HDMI | HDMI+DP | 双DP | 双DP+8K |
| **GPIO** | 8ch | 16ch | 32ch | 64ch | 128ch |
| **模拟输入** | 2ch 12bit | 4ch 16bit | 8ch 16bit | 16ch 16bit | 32ch 18bit |
| **编码器接口** | 1ch ABZ | 2ch ABZ | 4ch ABZ | 8ch ABZ | 16ch SSI |
| **扩展槽** | - | 1×M.2 | 2×M.2 | 2×PCIe×4 | 4×PCIe×8 |

---

## 21. 速度控制模块规格 (Velocity Control)

### 21.1 概述

速度控制模块 (`src/control/velocity_control.py`) 为差速驱动AGV提供先进的速度控制能力，包括S曲线速度规划、摩擦补偿、双轮PID闭环和打滑检测。

### 21.2 核心组件

| 类 | 功能 |
|----|------|
| `AGVVelocityController` | AGV完整速度控制器，整合运动学/PID/规划 |
| `SVelocityProfilePlanner` | S曲线速度规划器（梯形/S曲线） |
| `FrictionCompensator` | 库伦+粘滞摩擦补偿器 |
| `WheelVelocitySynchronizer` | 轮速同步与打滑检测 |
| `VelocityPIDController` | 自适应PID控制器 |

### 21.3 AGV五级速度控制规格

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **控制频率** | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| **最大线速度** | 0.5m/s | 1.5m/s | 2.0m/s | 3.0m/s | 3.5m/s |
| **最大角速度** | 1.5rad/s | 3.0rad/s | 2.5rad/s | 2.0rad/s | 1.5rad/s |
| **速度PID Kp** | 2.0 | 3.0 | 4.0 | 5.0 | 6.0 |
| **速度PID Ki** | 0.1 | 0.2 | 0.3 | 0.5 | 0.8 |
| **速度PID Kd** | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 |
| **摩擦补偿** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **前馈控制** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **打滑检测** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **自适应增益** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **速度规划** | 梯形 | 梯形 | S曲线 | S曲线 | S曲线 |
| **加速度限制** | 0.5m/s² | 1.0m/s² | 1.5m/s² | 2.0m/s² | 2.5m/s² |
| **加加速度限制** | — | — | 5.0m/s³ | 10.0m/s³ | 15.0m/s³ |
| **实时内核** | ✗ | ✗ | Xenomai | RT-PREEMPT | Xenomai+FPGA |

### 21.4 接口方法

#### AGVVelocityController

| 方法 | 说明 |
|------|------|
| `compute(target_linear, target_angular, meas_l, meas_r)` | 闭环速度控制，返回(左力矩, 右力矩, 状态) |
| `compute_openloop(linear, angular)` | 开环轮速计算 |
| `plan_trajectory(start_pos, end_pos)` | 规划线速度+角速度轨迹 |
| `start_trajectory(lp, ap)` | 启动轨迹执行 |
| `reset()` | 重置控制器状态 |
| `get_state()` | 获取完整状态字典 |

#### SVelocityProfilePlanner

| 方法 | 说明 |
|------|------|
| `plan(start, end, max_v, max_a, max_j)` | 生成速度剖面 |

### 21.5 使用示例

```python
from src.control.velocity_control import AGVVelocityController

# 创建M级速度控制器
ctrl = AGVVelocityController(grade="M")

# 开环速度控制
cmd = ctrl.compute_openloop(linear_vel=1.0, angular_vel=0.0)
print(f"左轮: {cmd.left_velocity_rps:.2f} rps")
print(f"右轮: {cmd.right_velocity_rps:.2f} rps")

# 闭环PID控制
left_tau, right_tau, state = ctrl.compute(
    target_linear_vel=1.0,
    target_angular_vel=0.0,
    measurement_left_rps=14.0,
    measurement_right_rps=14.1,
)
print(f"左力矩: {left_tau:.3f} Nm, 误差: {state.left_error:.2f} rps")
```

---

## 22. AGV五极控制规格模块 (Grade-Aware Control)

### 22.1 概述

`control/grade_control.py` 为不同AGV等级(S/M/L/XL/XXL)提供自适应控制参数和控制策略,包括PID配置、安全监控、轨迹规划的五级规格映射。

### 22.2 核心组件

| 类 | 功能 |
|----|------|
| `AGVGrade` | 五极等级枚举 (S/M/L/XL/XXL) |
| `GradePIDConfig` | 五极PID参数配置 |
| `GradeControllerConfig` | 五极控制器配置 |
| `GradeAwarePID` | 五极感知PID控制器 |
| `GradeAwareSafetyMonitor` | 五极感知安全监控器 |
| `GradeAwareTrajectoryPlanner` | 五极感知轨迹规划器 |

### 22.3 五极控制规格详细表

#### 22.3.1 控制器频率与周期

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **控制频率** | 20Hz | 50Hz | 100Hz | 200Hz | 1000Hz |
| **控制周期** | 50ms | 20ms | 10ms | 5ms | 1ms |
| **实时内核** | ✗ | ✗ | PREEMPT_RT | Xenomai | Xenomai+FPGA |

#### 22.3.2 PID参数

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **Kp** | 2.0 | 3.0 | 4.0 | 5.0 | 6.0 |
| **Ki** | 0.1 | 0.2 | 0.3 | 0.5 | 0.8 |
| **Kd** | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 |
| **输出限幅** | 10 | 20 | 30 | 50 | 80 |
| **积分限幅** | 5 | 10 | 15 | 25 | 40 |
| **前馈控制** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **自适应增益** | ✗ | ✗ | ✓ | ✓ | ✓ |

#### 22.3.3 运动限制

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **最大线速度** | 0.5m/s | 1.5m/s | 2.0m/s | 2.5m/s | 3.0m/s |
| **最大角速度** | 1.5rad/s | 3.0rad/s | 2.5rad/s | 2.0rad/s | 1.5rad/s |
| **最大加速度** | 0.5m/s² | 1.0m/s² | 1.5m/s² | 2.0m/s² | 2.5m/s² |
| **摩擦补偿** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **打滑检测** | ✗ | ✓ | ✓ | ✓ | ✓ |

#### 22.3.4 轨迹规划

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **规划模式** | 直线插补 | 梯形 | S曲线 | S曲线 | Minimum Snap |
| **规划算法** | line | trapezoidal | s_curve | rrt | rrt_star |
| **速度曲线** | 直线 | 梯形 | S曲线 | S曲线 | S曲线 |
| **最大加加速度** | — | — | 5.0m/s³ | 10.0m/s³ | 15.0m/s³ |

#### 22.3.5 安全与容错

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **安全等级** | PLd | PLd | PLe | PLe+SIL2 | PLe+SIL3 |
| **故障容错** | ✗ | ✗ | ✗ | ✓ | ✓ |
| **冗余设计** | ✗ | ✗ | ✗ | ✗ | ✓ |

#### 22.3.6 技能调度

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **最大并发技能** | 1 | 2 | 3 | 4 | 6 |
| **技能超时** | 60s | 45s | 30s | 15s | 5s |
| **收敛样本数** | 10K | 50K | 200K | 500K | 1M+ |

### 22.4 接口方法

#### GradeAwarePID

| 方法 | 说明 |
|------|------|
| `compute(error, dt, measurement, feedforward)` | PID计算,返回控制输出 |
| `reset()` | 重置PID状态 |
| `set_setpoint(setpoint)` | 设置设定值 |
| `get_state()` | 获取PID状态字典 |

#### GradeAwareSafetyMonitor

| 方法 | 说明 |
|------|------|
| `check_velocity(velocity, dt, timestamp)` | 速度限制检查 |
| `check_boundary(position, ...)` | 位置边界检查 |
| `check_force(force_magnitude, torque_magnitude)` | 力/力矩检查 |
| `check_slip(left, right, expected)` | 打滑检测 (M+) |
| `trigger_estop(reason)` | 触发紧急停止 |
| `reset_estop()` | 重置紧急停止 |
| `get_capabilities()` | 获取该等级安全能力 |

#### GradeAwareTrajectoryPlanner

| 方法 | 说明 |
|------|------|
| `plan_line(start, end)` | 直线轨迹规划 |
| `plan_trapezoidal(start, end, max_v, max_a)` | 梯形速度规划 |
| `plan_s_curve(start, end, max_v, max_a, max_j)` | S曲线速度规划 |
| `get_current_trajectory()` | 获取当前轨迹 |

### 22.5 使用示例

```python
from control.grade_control import (
    AGVGrade, GradeAwarePID, GradeAwareSafetyMonitor,
    GradeAwareTrajectoryPlanner, list_grade_capabilities
)

# 按AGV等级获取控制器
grade = AGVGrade.M
caps = list_grade_capabilities(grade)
print(f"{grade}级: {caps['control_frequency']}Hz, 最大速度 {caps['max_velocity']}m/s")

# 五极PID控制器
pid = GradeAwarePID(AGVGrade.XXL)
output = pid.compute(error=0.5, dt=0.001, feedforward=2.0)
print(f"PID输出: {output:.3f}")

# 五极安全监控
mon = GradeAwareSafetyMonitor(AGVGrade.XL)
level, msg = mon.check_velocity(velocity=3.0, dt=0.005)
print(f"安全检查: {level} - {msg}")

# 五极轨迹规划
planner = GradeAwareTrajectoryPlanner(AGVGrade.XXL)
traj = planner.plan_s_curve(
    start=(0, 0, 0), end=(2.0, 0, 0),
    max_jerk=15.0
)
print(f"轨迹点数: {len(traj)}")
```

---

## 23. 传感器信号处理模块规格 (Signal Processing)

### 23.1 概述

`src/sensors/signal_processor.py` 为触觉、力觉、IMU等传感器提供高级滤波和信号处理能力，包括卡尔曼滤波、异常值检测、噪声估计等功能。

### 23.2 核心组件

| 类 | 功能 |
|----|------|
| `KalmanFilter1D` | 一维卡尔曼滤波器 |
| `KalmanFilter3D` | 三维卡尔曼滤波器 (IMU等3D传感器) |
| `ButterworthFilter` | Butterworth数字滤波器 (低通/高通/带通) |
| `MedianFilter` | 中值滤波器 (去除脉冲噪声) |
| `ExponentialSmoother` | 指数平滑器 (实时低计算量) |
| `OutlierDetector` | 异常值检测器 (Z-score/IQR方法) |
| `SignalProcessor` | 统一信号处理器 (整合所有功能) |

### 23.3 AGV五级信号处理规格

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **可用滤波器** | 指数 | 指数+卡尔曼 | 指数+卡尔曼+中值+巴特沃斯 | 同L | 同L+带通 |
| **异常值检测** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **最大采样率** | 100Hz | 200Hz | 500Hz | 1000Hz | 2000Hz |
| **通道数** | 1ch | 3ch | 6ch | 9ch | 12ch |

### 23.4 接口方法

#### KalmanFilter3D

| 方法 | 说明 |
|------|------|
| `update(measurement)` | 更新滤波器状态，返回滤波后3D向量 |
| `reset(initial_state)` | 重置滤波器状态 |
| `state` | 当前滤波状态 |
| `error_covariance` | 估计协方差矩阵对角元素 |

#### SignalProcessor

| 方法 | 说明 |
|------|------|
| `process_frame(data, remove_outliers, filter_type)` | 处理3D传感器帧 |
| `process_scalar(value)` | 处理标量数据 |
| `compute_stats(signal)` | 计算信号统计信息 |
| `reset()` | 重置所有滤波器 |
| `enable()` / `disable()` | 启用/禁用处理 |

#### OutlierDetector

| 方法 | 说明 |
|------|------|
| `detect(value)` | 检测异常值，返回(是否异常, 置信度) |
| `is_valid(value)` | 判断值是否有效 |
| `reset()` | 重置缓冲区 |

### 23.5 使用示例

```python
from src.sensors.signal_processor import (
    SignalProcessor, KalmanFilter3D, FilterConfig, FilterType
)

# 创建处理器 (M级)
proc = SignalProcessor(FilterConfig(
    filter_type=FilterType.KALMAN,
    process_noise=0.001,
    measurement_noise=0.1,
    window_size=5
))

# 处理IMU数据
raw_imu = np.array([0.1, -0.2, 9.81], dtype=np.float32)
filtered = proc.process_frame(raw_imu, remove_outliers=True)

# 计算统计
stats = proc.compute_stats(raw_imu)
print(f"RMS: {stats.rms:.4f}, SNR: {stats.snr:.2f}dB")

# 异常值检测
is_outlier, conf = proc._outlier.detect(100.0)  # via internal detector

# 单步卡尔曼滤波
kf = KalmanFilter3D(process_noise=0.001, measurement_noise=0.1)
measurement = np.array([0.1, -0.2, 9.81], dtype=np.float32)
filtered_3d = kf.update(measurement)
```

### 23.6 信号统计 SignalStats

```python
@dataclass
class SignalStats:
    mean: float       # 均值
    std: float        # 标准差
    min_val: float   # 最小值
    max_val: float   # 最大值
    rms: float       # 均方根值
    snr: float       # 信噪比 (dB)
    noise_estimate: float  # 噪声估计 (通过差分法)
```

---

## 24. 阻抗控制模块 (Impedance Control)

### 24.1 概述

阻抗控制实现柔顺控制，核心方程：
```
M·Xdd + D·Xd + K·X = F
```
其中 X 是位置误差，F 是外力，M/D/K 分别是惯性/阻尼/刚度矩阵。

**支持控制器：**
- `ImpedanceController` - 基础阻抗控制器
- `AdmittanceController` - 导纳控制器（输入力→输出位置）
- `ForceImpedanceController` - 力位混合控制器
- `CollaborativeController` - 协作/拖动示教控制器
- `AdaptiveImpedanceController` - 自适应阻抗控制器（MRAC + 李雅普诺夫）

### 24.2 AGV五级阻抗控制规格表

| 规格项 | S | M | L | XL | XXL |
|--------|---|---|---|---|-----|
| 控制频率 (Hz) | 50 | 100 | 200 | 500 | 1000 |
| 刚度范围 (N/m) | 50~500 | 100~1000 | 200~2000 | 300~3000 | 500~5000 |
| 阻尼范围 (Ns/m) | 10~100 | 20~200 | 50~500 | 70~700 | 100~1000 |
| 惯性范围 (kg) | 1~10 | 2~20 | 5~50 | 7~70 | 10~100 |
| 力限制 (N) | 50 | 100 | 200 | 350 | 500 |
| 位置误差限制 (m) | 0.05 | 0.02 | 0.01 | 0.005 | 0.001 |
| 自适应率 | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 |
| 收敛时间 (s) | 5.0 | 2.0 | 1.0 | 0.5 | 0.5 |
| 李雅普诺夫稳定 | ✗ | ✗ | ✓ | ✓ | ✓ |
| MRAC自适应 | ✗ | ✗ | ✗ | ✓ | ✓ |

### 24.3 核心接口

```python
# AGV五级规格查询
from src.control.impedance import AGV_IMPEDANCE_GRADES, get_impedance_spec
spec = get_impedance_spec("XXL")  # 获取XXL级规格

# 基础阻抗控制
from src.control.impedance import ImpedanceController, ImpedanceParams
params = ImpedanceParams.default_6d()
ctrl = ImpedanceController(params, control_rate=100.0)
jacobian = np.eye(6, 3)
torque, info = ctrl.compute_torque(
    desired_position=np.array([0.1, 0.0, 0.0]),
    desired_velocity=np.zeros(3),
    current_position=np.array([0.05, 0.0, 0.0]),
    current_velocity=np.zeros(3),
    external_wrench=np.zeros(6),
    jacobian=jacobian,
)

# 自适应阻抗控制（MRAC + 李雅普诺夫）
from src.control.impedance import AdaptiveImpedanceController
adapt_ctrl = AdaptiveImpedanceController(
    control_rate=500.0,
    adaptation_rate=0.05,
    env_stiffness_bounds=(100.0, 10000.0),
    use_lyapunov=True,
)
torque, info = adapt_ctrl.update(
    desired_position=np.array([0.1, 0.0, 0.0]),
    current_position=np.array([0.05, 0.0, 0.0]),
    current_velocity=np.zeros(3),
    external_wrench=np.zeros(6),
    jacobian=jacobian,
    contact_phase="contact",
    task_type="assembly",
)
print(f"估计环境刚度: {info['est_env_K']:.1f} N/m")
print(f"收敛性: {adapt_ctrl.get_convergence_metrics()}")

# 导纳控制
from src.control.impedance import AdmittanceController
adm_ctrl = AdmittanceController(M=10.0, D=50.0, K=200.0, control_rate=100.0)
pos = adm_ctrl.update(external_force=10.0, desired_position=0.0)
```

---

## 25. 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| v2.51.0 | 2026-04-10 | 新增AGV五级阻抗控制规格(AGV_IMPEDANCE_GRADES); ForceImpedanceController.compute_torque bug修复; impedance_control_tests.py 31项测试; SPEC.md第24章阻抗控制规格; 2636项测试全通过 |
| v2.46.1 | 2026-04-10 | 新增AGV五极控制规格模块(control/grade_control.py); 五极PID/安全监控/轨迹规划器; grade_control_tests.py 100项测试; SPEC.md第22章五极控制规格; 583项测试全通过 |
| v2.45.0 | 2026-04-10 | 新增velocity_control.py速度控制模块(S曲线规划/摩擦补偿/PID闭环/AGV五级规格); velocity_control_tests.py 78项测试全通过; SPEC.md第21章速度控制规格; 483项测试全通过 |
| v2.43.0 | 2026-04-10 | 补充SPEC.md第20章AGV五级规格总表(7大子系统完整对照表); 传感器+融合+控制全链路五级规格完善; 2327项测试全通过 |
| v2.42.0 | 2026-04-10 | 新增核心目标系统(src/core/: core_goals/safety_shield/value_judgment/self_preservation); 更新MODULE_INDEX.md核心层章节; 2327项测试全通过 |
| v2.40.0 | 2026-04-10 | 完善触觉/力觉/IMU控制模块 + 修复标定管理器; 37项测试全通过; 2297项测试全通过 |
| v2.39.0 | 2026-04-10 | 具身智能仿真环境(embodied_sim.py, 720行) + 42项测试全通过; 2261项测试全通过 |
| v2.38.0 | 2026-04-10 | 传感器标定管理器(calibration_manager.py, 750行) + 33项测试全通过; 2259项测试全通过 |
| v2.37.0 | 2026-04-10 | 仿真控制接口 + MODULE_INTERFACE五级规格总表完善; 418项测试全通过 |
| v2.23.0 | 2026-04-10 | 新增sensorimotor_integration_tests.py(30项); 2009项测试全通过 |
| v2.15.0 | 2026-04-09 | 补充SPEC.md第18章; 378项传感器+融合测试全通过 |
| v2.14.0 | 2026-04-09 | 新增SurfaceFollowingController与AssemblyController; AGV五级具身控制完整规格表 |
| v2.13.0 | 2026-04-09 | 完善SPEC.md接口使用示例、数据流与状态机、错误处理规范 |
| v2.12.0 | 2026-04-09 | 版本同步，1937项测试全通过 |
