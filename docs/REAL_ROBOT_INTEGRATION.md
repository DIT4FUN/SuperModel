# SuperModel 真实机器人集成指南

> **文档版本**: v1.0.0
> **最后更新**: 2026-04-01
> **项目**: SuperModel 超模态机器人具身智能大脑

---

## 概述

本文档描述如何将 SuperModel 从仿真环境部署到真实机器人平台。

---

## 1. 硬件平台概览

### 1.1 推荐硬件配置

| AGV等级 | 计算平台 | 传感器 | 执行器 | 典型场景 |
|---------|----------|--------|--------|----------|
| **S** | RK3588 / Jetson Nano | RealSense D435i + MPU6050 | 直流伺服电机 | 教育/实验室 |
| **M** | RK3588 + FPGA | RealSense D455 + BMI088/ETT10A-PW | 步进/伺服电机 | 服务机器人 |
| **L** | RK3588 x2 + FPGA | 双目 + ADIS16470 | 伺服电机 + 液压 | 工业搬运 |
| **XL** | RK3588集群 | 多目 + LiDAR + ADIS16470 | 高性能伺服 | 复杂物流 |
| **XXL** | RK3588集群 + GPU | 多目 + 3D LiDAR + 工业IMU | 重型伺服 | 柔性制造 |

### 1.2 已采购硬件

| 组件 | 型号 | 数量 | 来源 |
|------|------|------|------|
| 激光雷达 | 镭神 N10P | 1 | https://detail.tmall.com/item.htm?id=661907723595 |
| IMU | ETT10A-PW | 1 | https://item.taobao.com/item.htm?id=622844097690 |
| 从动轮 | ESUN 2.5寸 | 2 | https://detail.tmall.com/item.htm?id=591810849491 |
| RGB相机 | 奥比中光 C100 | 1 | https://item.taobao.com/item.htm?id=641692244195 |
| 深度相机 | 奥比中光 Astra Pro Plus | 1 | https://item.taobao.com/item.htm?id=646073233035 |
| 电机驱动器 | 中菱 ZLAC8015D | 1 | https://item.taobao.com/item.htm?id=677349695836 |

### 1.3 通信接口

```
真实机器人通信架构:

  SuperModel (RK3588)
       │
       ├── CANopen/RS485 ───→ 电机驱动器 (ZLAC8015D)
       ├── Ethernet ────────→ ROS2 Humble (多机协同)
       ├── USB2.0 ──────────→ 奥比中光 Astra/C100
       ├── RS485/CAN ───────→ IMU (ETT10A-PW)
       ├── UART ────────────→ 激光雷达 (N10P)
       ├── GPIO ────────────→ 安全急停/限位开关
       └── USB ────────────→ 里程计编码器
```

---

## 2. 软件架构

### 2.1 ROS2 Humble 集成

SuperModel 通过 `src/control/ros2_interface.py` 与 ROS2 Humble 无缝集成:

```python
from control.ros2_interface import (
    ROS2JointTrajectoryInterface,
    ROS2TopicInterface,
    ROS2ServiceInterface,
    ROS2ActionInterface,
)

# 创建 ROS2 接口
joint_if = ROS2JointTrajectoryInterface(node_name="supermodel_joint_control")
topic_if = ROS2TopicInterface()
service_if = ROS2ServiceInterface()

# 订阅传感器话题
joint_if.subscribe_joint_states("/robot/joint_states")

# 发布控制指令
joint_if.send_position_cmd(
    joint_names=["waist", "shoulder", "elbow", "wrist"],
    positions=[0.0, 0.5, -0.3, 0.2],
    duration=2.0
)
```

### 2.2 传感器驱动映射

| 仿真模块 | 真实传感器 | 驱动接口 | 话题/服务 |
|----------|-----------|----------|-----------|
| `BinocularCamera` | RealSense D455 | librealsense2 | `/camera/realsense` |
| `BinauralMic` | Respeaker 4-mic | pyusb/ALSA | `/audio/audio` |
| `TactileArray` | Digi Sensing 电子皮肤 | I2C/SPI | `/tactile/raw` |
| `ForceTorqueSensor` | ATI Nano25 | Ethernet/USB | `/ft/raw` |
| `IMUSensor` | BMI088/ADIS16470 | SPI | `/imu/data` |

---

## 3. 标定流程

### 3.1 相机标定

```bash
# RealSense 内参标定
ros2 run realsense2_camera realsense2_camera
ros2 run camera_calibration cameracalibrator.py \
    --approx 0.0 \
    --size 8x6 \
    --square 0.025 \
    image:=/camera/color/image_raw

# 双目外参标定
ros2 run camera_calibration cameracalibrator.py \
    --size 8x6 \
    --square 0.025 \
    left:=/camera/left/image_raw \
    right:=/camera/right/image_raw
```

### 3.2 IMU 标定

```python
from sensors.imu import IMUSensor, IMUSensorType

imu = IMUSensor(sensor_type=IMUSensorType.BMI088, sensor_id="imu_0")
imu.open()

# 陀螺仪偏置标定 (静止状态)
imu.calibrate_gyro_bias(num_samples=500, duration_sec=5.0)

# 加速度计标定 (已知朝向)
imu.calibrate_accel(known_orientation="level")
imu.calibrate_accel(known_orientation="up")
imu.calibrate_accel(known_orientation="right")

imu.close()
```

### 3.3 力传感器标定

```python
from sensors.force import ForceTorqueSensor, ForceSensorType

ft = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS, sensor_id="ft_0")
ft.open()

# 偏置校准 (无负载状态)
ft.calibrate_bias(num_samples=100)

# 工具中心设置 (用于重力补偿)
ft.set_tool_center(tool_mass=0.55, tool_com=np.array([0.0, 0.0, 0.05]))

ft.close()
```

### 3.4 触觉传感器标定

```python
from sensors.tactile import TactileArray, TactileCalibration

tactile = TactileArray(array_size=(16, 16), sensor_id="tactile_0")
tactile.open()

# 零压力基准
import numpy as np
frames = [tactile.capture() for _ in range(50)]
zero_pressure = np.mean([f.pressure_map for f in frames], axis=0)

# 力标定 (已知砝码)
known_weights = [1.0, 2.0, 5.0]  # kg
tactile.calibrate(zero_pressure=zero_pressure, known_weights=known_weights)

tactile.close()
```

---

## 4. 从仿真到真实的迁移

### 4.1 仿真参数 → 真实参数

```python
# 仿真参数 (Gymnasium/MuJoCo)
SIM_PARAMS = {
    "mass": 5.5,           # kg (负载质量)
    "gravity": 9.81,       # m/s^2
    "friction": 0.3,      # 摩擦系数
    "max_torque": 100.0,  # Nm (最大关节力矩)
    "max_velocity": 2.0,   # rad/s
}

# 真实参数 (需要从电机规格书获取)
REAL_PARAMS = {
    "mass": 5.48,          # 实际测量 5.48kg
    "gravity": 9.794,      # 当地重力加速度
    "friction": 0.28,      # 实测摩擦系数
    "max_torque": 85.0,   # 真实电机峰值力矩 (留20%余量)
    "max_velocity": 1.8,   # 真实最大转速 (留10%余量)
}

# 补偿因子
COMPENSATION = {
    "mass": REAL_PARAMS["mass"] / SIM_PARAMS["mass"],  # 0.996
    "torque": REAL_PARAMS["max_torque"] / SIM_PARAMS["max_torque"],  # 0.85
    "velocity": REAL_PARAMS["max_velocity"] / SIM_PARAMS["max_velocity"],  # 0.90
}
```

### 4.2 安全限制

```python
from control.safety_controller import SafetyController, SafetyConfig, SafetyLevel

safety_config = SafetyConfig(
    joint_limits=[
        {"min": -2.356, "max": 2.356},   # waist
        {"min": -1.571, "max": 1.571},   # shoulder
        {"min": -2.094, "max": 2.094},   # elbow
        {"min": -3.142, "max": 3.142},   # wrist
    ],
    velocity_limits=[1.5, 1.5, 1.5, 2.0],  # rad/s
    torque_limits=[80.0, 80.0, 40.0, 20.0],  # Nm
    safety_level=SafetyLevel.AGGRESSIVE,
    stop_torque_threshold=90.0,  # Nm
    collision_force_threshold=50.0,  # N
    workspace_bounds={
        "x": (-1.0, 1.0),
        "y": (-1.0, 1.0),
        "z": (0.0, 1.5),
    },
)

safety = SafetyController(config=safety_config)
```

---

## 5. AGV 五级集成指南

### 5.1 S 级 (教育/实验室)

最小配置，适合快速原型验证:

```python
from src.control.agv import AGVMotionController, AGVGrade

agv = AGVMotionController(
    grade=AGVGrade.S,
    drive_type=DriveType.DIFFERENTIAL,
    wheel_radius=0.05,
    wheelbase=0.3,
)

# 差速控制
agv.set_velocity(v_linear=0.2, omega=0.0)  # 直线运动
agv.set_velocity(v_linear=0.0, omega=0.5)  # 原地旋转

# 里程计积分
pose = agv.get_pose()
print(f"Pose: x={pose.x:.3f}, y={pose.y:.3f}, theta={pose.theta:.3f}")
```

### 5.2 M 级 (服务机器人)

标准配置，支持 SLAM 和自主导航:

```python
from src.perception.scene_understanding import SceneUnderstanding

scene = SceneUnderstanding(grade="M")

# 实时场景理解
scene_obs = scene.observe(
    rgb_image=camera.frame,
    depth_image=depth.frame,
    return_semantic=True,
    return_3d=True,
)

print(f"Detected objects: {len(scene_obs['objects'])}")
print(f"Navigation waypoints: {scene_obs['nav_waypoints']}")
```

### 5.3 L/XL/XXL 级 (工业/重载)

高性能配置，多机协同:

```python
from src.control.multi_agent import MultiAgentCoordinator, FormationType

coordinator = MultiAgentCoordinator(
    num_agents=3,
    formation_type=FormationType.LINE,
    leader_id="agent_0",
)

# 协同控制
tasks = coordinator.create_formation_task(
    target_positions=[
        (0.0, 0.0),
        (-1.0, 0.0),
        (-2.0, 0.0),
    ]
)

# 执行协同任务
for task in tasks:
    coordinator.assign_task(task)
    coordinator.monitor_formation()

# 碰撞检测
risk = coordinator.check_collision_risk()
if risk.level > CollisionRisk.LOW:
    coordinator.avoid_collision(risk)
```

---

## 6. 性能基准

### 6.1 感知延迟目标

| AGV等级 | 视觉延迟 | 力觉延迟 | IMU延迟 | 融合延迟 |
|---------|----------|----------|---------|----------|
| **S** | <100ms | <20ms | <5ms | <50ms |
| **M** | <50ms | <10ms | <5ms | <30ms |
| **L** | <20ms | <5ms | <2ms | <15ms |
| **XL** | <10ms | <2ms | <1ms | <8ms |
| **XXL** | <5ms | <1ms | <0.5ms | <4ms |

### 6.2 端到端延迟预算

```
端到端感知-控制延迟链路:

传感器采集 ──→ 预处理 ──→ 特征提取 ──→ 融合 ──→ 决策 ──→ 控制 ──→ 执行器
     │            │           │          │        │        │
   Camera      预处理      特征网络    融合网络   策略    电机驱动
   IMU等       滤波        注意力      跨模态    RL/DQN   PWM/CAN
     │            │           │          │        │        │
   Real        <5ms        <20ms      <30ms    <10ms    <10ms
   (真实)
```

---

## 7. 调试与故障排除

### 7.1 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 关节力矩异常增大 | 摩擦力估计不准确 | 重新标定摩擦模型，降低速度增益 |
| IMU漂移严重 | 陀螺仪偏置未校准 | 执行 `calibrate_gyro_bias()` |
| 深度相机噪声大 | 基线标定偏移 | 使用官方标定工具重新标定 |
| CAN 通信丢包 | 线缆过长/电磁干扰 | 缩短线缆，加屏蔽，使用双绞线 |
| 力传感器偏置漂移 | 温度漂移 | 启用温漂补偿，周期性重校准 |

### 7.2 调试工具

```python
from src.evaluation.benchmark import SensorBenchmark, LatencyTracker

# 传感器延迟基准测试
benchmark = SensorBenchmark(grade="M")
results = benchmark.run_sensor_benchmarks(
    sensors=["camera", "imu", "ft_sensor"],
    duration_sec=10.0,
)

for name, metrics in results.items():
    print(f"{name}: p95={metrics.p95_latency_ms:.2f}ms, fps={metrics.fps:.1f}")

# 实时延迟跟踪
tracker = LatencyTracker(window_size=100)
while True:
    start = time.perf_counter()
    # 处理
    end = time.perf_counter()
    tracker.update((end - start) * 1000)  # ms
    print(f"P95: {tracker.get_percentile(95):.2f}ms")
```

---

## 8. 下一步

1. **RK3588 NPU 部署**: 使用 RKNN-Toolkit2 量化模型，部署到 NPU 加速推理
2. **真机验证**: 在目标平台上进行完整的端到端功能验证
3. **长期测试**: 进行 24h+ 连续运行测试，验证稳定性
4. **性能调优**: 根据真实数据调整控制参数和安全限制

---

_本文档与 SuperModel v1.42.0 同步更新_
