# SuperModel 技术规格文档 (SPEC.md)

## 1. 模块接口设计

### 1.1 传感器模块接口

#### TactileSensor (触觉传感器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `read(timestamp)` | float | TactileData | 读取触觉数据 |
| `calibrate(ref)` | np.ndarray | bool | 校准传感器 |
| `get_sensitivity()` | - | float | 获取灵敏度 |

#### ForceSensor (力觉传感器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `read(timestamp)` | float | ForceData | 读取力觉数据 |
| `set_bias(data)` | ForceData | None | 设置零偏 |
| `apply_calibration(raw)` | np.ndarray | np.ndarray | 校准变换 |
| `compute_tcp_wrench(offset)` | np.ndarray | np.ndarray | 计算TCP力/力矩 |

#### IMUSensor (IMU传感器)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `read(timestamp)` | float | IMUData | 读取IMU数据 |
| `update_orientation(gyro, dt)` | np.ndarray, float | None | 更新姿态 |
| `get_euler_from_quat(q)` | np.ndarray | np.ndarray | 四元数转欧拉 |
| `calibrate_gyro_bias(samples)` | int | None | 陀螺仪偏置校准 |

### 1.2 控制模块接口

#### Motor (电机控制)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `enable()` | - | None | 使能电机 |
| `disable()` | - | None | 禁用电机 |
| `set_target(target, mode)` | float, MotorControlMode | None | 设置目标值 |
| `step(dt)` | float | MotorState | 步进控制 |
| `get_state()` | - | MotorState | 获取状态 |

#### TrajectoryPlanner (轨迹规划)

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `plan_line(start, end)` | Waypoint, Waypoint | List[TrajectoryPoint] | 直线规划 |
| `plan_arc(start, end, curv)` | Waypoint, Waypoint, float | List[TrajectoryPoint] | 圆弧规划 |
| `smooth_trajectory(traj)` | List[TrajectoryPoint] | List[TrajectoryPoint] | 轨迹平滑 |
| `plan_path(waypoints)` | List[Waypoint] | List[TrajectoryPoint] | 多路点规划 |

## 2. AGV五级规格表

| 等级 | 负载能力 | 导航方式 | 定位精度 | 安全标准 | 典型场景 | 代表型号 |
|------|---------|---------|---------|---------|---------|---------|
| **L1** | ≤500kg | 磁条/二维码 | ±10mm | ISO 3691-2 | 仓储拣选 | 潜伏式AGV |
| **L2** | 500-1500kg | 激光导航 | ±5mm | ISO 3691-4 | 产线配送 | 叉式AGV |
| **L3** | 1500-3000kg | SLAM视觉 | ±3mm | ISO 3691-4 | 柔性制造 | 复合AGV |
| **L4** | 3000-5000kg | 多传感器融合 | ±1mm | IEC 61508 SIL2 | 重载车间 | 重载AGV |
| **L5** | >5000kg | 具身智能超模态 | <±0.5mm | IEC 61508 SIL3 | 无人化工厂 | 超级AGV |

### L5级 SuperModel 核心规格

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

### TactileData
- 压力: Pa (帕斯卡)
- 触感阵列: `np.ndarray[rows, cols]`
- 归一化向量: `to_vector()` → `np.ndarray`

### ForceData
- 六维力: `[Fx, Fy, Fz, Mx, My, Mz]` 单位: N, Nm
- 力矩补偿: `M' = M + F × offset`

### IMUData
- 加速度: `m/s²` → `[ax, ay, az]`
- 角速度: `rad/s` → `[wx, wy, wz]`
- 磁场: `μT` → `[mx, my, mz]`
- 欧拉角: `rad` → `[roll, pitch, yaw]`
- 四元数: `[w, x, y, z]`

## 4. 测试规范

```bash
# 运行所有测试
pytest tests/ -v

# 传感器模块测试
pytest tests/sensor_tests.py -v

# 传感器融合测试
pytest tests/fusion_tests.py -v

# 控制模块测试
pytest tests/control_tests.py -v
```
