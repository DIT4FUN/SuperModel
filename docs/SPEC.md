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

#### TrajectoryPlanner (轨迹规划器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `plan_line(start, end)` | Waypoint, Waypoint | List[TrajectoryPoint] | 直线规划 |
| `plan_arc(start, end, curv)` | Waypoint, Waypoint, float | List[TrajectoryPoint] | 圆弧规划 |
| `smooth_trajectory(traj)` | List[TrajectoryPoint] | List[TrajectoryPoint] | 轨迹平滑 |
| `plan_path(waypoints)` | List[Waypoint] | List[TrajectoryPoint] | 多路点规划 |

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

## 4. 测试规范

```bash
# 运行所有测试
pytest tests/ -v

# 传感器模块测试 (89项)
pytest tests/sensor_tests.py -v

# 传感器融合测试 (24项)
pytest tests/fusion_tests.py -v

# 控制模块测试 (220项)
pytest tests/control_tests.py -v

# 运行关键测试子集
pytest tests/sensor_tests.py tests/fusion_tests.py tests/control_tests.py -q
```
