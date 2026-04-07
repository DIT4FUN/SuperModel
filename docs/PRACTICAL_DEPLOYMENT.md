# SuperModel AGV五级部署实战指南

> **文档版本**: v1.0.0
> **更新**: 2026-04-07
> **项目**: SuperModel 超模态机器人具身智能大脑

本文档提供 SuperModel 从传感器选型到完整部署的实战指南，按 AGV 五级 (S/M/L/XL/XXL) 分级说明。

---

## 1. 部署概览

```
部署流程:
  ① 选型定级 → ② 硬件连接 → ③ 传感器配置 → ④ 融合调参 → ⑤ 控制调参 → ⑥ 仿真验证 → ⑦ 实机部署
```

### AGV五级选型决策树

| 需求 | 推荐等级 | 理由 |
|------|:-------:|------|
| 教学/科研，预算 < ¥15K | S | MPU6050 + 单目，ROS2 Humble |
| 标准室内AGV，¥15-50K | M | 双目D435i + 6轴力矩，BMI088 |
| 精密装配/力控，¥50-150K | L | 双目D455 + 24×24触觉，阻抗+MPC |
| 重载/高速，¥150-500K | XL | 事件相机 + ADIS16470，500Hz控制 |
| 全功能旗舰，>¥500K | XXL | 多目+LiDAR + 48×48触觉，具身智能 |

---

## 2. 硬件连接

### 2.1 传感器接线 (M级示例)

```
AGV-M 传感器连接图:

  [双目相机D435i] ── USB3.0 ──┐
                               │
  [IMU BMI088]     ── SPI ────┼── [RK3588 NPU] ── Ethernet ── 上位机
                               │
  [六轴力矩]       ── USB ────┤
                               │
  [触觉阵列16×16] ── SPI ────┘
```

### 2.2 各级传感器接口速查

| 等级 | 触觉接口 | 力觉接口 | IMU接口 | 主控接口 |
|------|:-------:|:-------:|:-------:|:--------:|
| S | I2C | USB HID | I2C | Raspberry Pi USB |
| M | SPI | CAN | SPI/I2C | RK3588 USB |
| L | SPI | EtherCAT | SPI | Jetson Orin USB/Ethernet |
| XL | USB3.0 | EtherCAT | SPI | Jetson Orin AGX |
| XXL | USB3.0 | Ethernet UDP | SPI×2 | Orin AGX×2 + GPU |

---

## 3. 各级传感器配置

### 3.1 S级 (教学级)

```python
"""
S级配置: 低成本快速部署
- IMU: MPU6050 I2C@100kHz, ±8g, ±1000°/s, 100Hz
- 触觉: 8×8 电阻式, 12bit, 0-500kPa, 50Hz
- 力觉: 3轴 ±100N, 100Hz
- 融合: LATE融合, 128d
"""

from src.sensors.imu import IMUSensor, IMUSensorType
from src.sensors.tactile import TactileArray, TactileSensorType
from src.sensors.force import ForceTorqueSensor, ForceSensorType

# IMU配置
imu = IMUSensor(
    sensor_type=IMUSensorType.MPU6050,
    sensor_id="imu_0",
    accel_range=8,
    gyro_range=1000,
    sample_rate=100
)
imu.open()
imu.calibrate_gyro_bias(num_samples=200)
imu.self_test()

# 触觉配置
tactile = TactileArray(
    array_size=(8, 8),
    sensor_type=TactileSensorType.RESISTIVE,
    sensor_id="tactile_0"
)
tactile.open()

# 力觉配置
force = ForceTorqueSensor(
    sensor_type=ForceSensorType.THREE_AXIS,
    sensor_id="force_0"
)
force.open()
force.calibrate_bias(num_samples=100)

print("S级传感器初始化完成")
```

### 3.2 M级 (标准工业级)

```python
"""
M级配置: 标准AGV配置
- IMU: BMI088 SPI, ±16g, ±2000°/s, 200Hz
- 触觉: 16×16 电容式, 12bit, 0-1000kPa, 100Hz
- 力觉: 6轴 ±200N/±20N·m, 500Hz
- 融合: HYBRID融合, 256d
"""

from src.sensors.imu import IMUSensor, IMUSensorType
from src.sensors.tactile import TactileArray, TactileSensorType
from src.sensors.force import ForceTorqueSensor, ForceSensorType

# IMU: BMI088 双6轴
imu = IMUSensor(
    sensor_type=IMUSensorType.BMI088,
    sensor_id="imu_0",
    accel_range=16,
    gyro_range=2000,
    sample_rate=200
)
imu.open()
imu.calibrate_gyro_bias(num_samples=500)
imu.calibrate_accel(known_orientation="level")

# 触觉: 电子皮肤
tactile = TactileArray(
    array_size=(16, 16),
    sensor_type=TactileSensorType.CAPACITIVE,
    sensor_id="tactile_0"
)
tactile.open()
tactile.calibrate()

# 六轴力矩: ATI风格
force = ForceTorqueSensor(
    sensor_type=ForceSensorType.SIX_AXIS,
    sensor_id="ft_0",
    ip_address="192.168.1.100"  # Net F/T UDP
)
force.open()
force.set_tool_center(tool_mass=0.5, tool_com=np.array([0, 0, 0.05]))
force.calibrate_bias(num_samples=200)

# 姿态估计: Madgwick
pose_estimator = PoseEstimator(
    algorithm="madgwick",
    sample_rate=200.0,
    beta=0.1
)

print("M级传感器初始化完成")
```

### 3.3 L级 (精密工业级)

```python
"""
L级配置: 精密操作
- IMU: BMI088 + 磁力计, ±24g, ±4000°/s, 500Hz
- 触觉: 24×24 压电式, 14bit, 0-2000kPa, 200Hz + 温度 + 接近觉
- 力觉: 6轴 ±500N/±50N·m, 1000Hz EtherCAT
- 融合: HYBRID融合, 512d
- 控制: 阻抗控制 + RRT*规划, 200Hz
"""

from src.sensors.imu import IMUSensor, IMUSensorType, PoseEstimator
from src.sensors.tactile import TactileArray, TactileSensorType
from src.sensors.force import ForceTorqueSensor, ForceSensorType
from src.fusion.cross_modal_fusion import CrossModalFusion, FusionConfig

# IMU: 高性能6轴+磁力计
imu = IMUSensor(
    sensor_type=IMUSensorType.MPU9250,  # 9轴版本
    sensor_id="imu_0",
    accel_range=24,
    gyro_range=4000,
    sample_rate=500
)
imu.open()
imu.calibrate_gyro_bias(num_samples=500)
imu.calibrate_accel(known_orientation="level")

# 触觉: 高分辨率电子皮肤
tactile = TactileArray(
    array_size=(24, 24),
    sensor_type=TactileSensorType.PIEZOELECTRIC,
    sensor_id="tactile_0"
)
tactile.open()

# 六轴力矩: EtherCAT接口
force = ForceTorqueSensor(
    sensor_type=ForceSensorType.SIX_AXIS,
    sensor_id="ft_0",
    ip_address="192.168.1.101",
    ethernet_type="TCP"
)
force.open()
force.set_tool_center(tool_mass=1.0, tool_com=np.array([0, 0, 0.1]))
force.calibrate_bias(num_samples=500)

# 姿态估计: 卡尔曼滤波
pose_estimator = PoseEstimator(
    algorithm="kalman",
    sample_rate=500.0
)

# 融合配置
fusion_config = FusionConfig(
    fusion_strategy="hybrid",
    hidden_dim=512,
    num_heads=6,
    num_layers=3
)
fusion = CrossModalFusion(fusion_config)

print("L级传感器和融合初始化完成")
```

### 3.4 XL级 (高性能工业级)

```python
"""
XL级配置: 高性能/高速
- IMU: ADIS16470×1, ±40g, ±4000°/s, 1000Hz
- 触觉: 32×32 光学式, 14bit, 0-5000kPa, 500Hz
- 力觉: 6轴 ±1000N/±100N·m, 2000Hz
- 融合: EARLY+HYBRID, 768d
- 控制: MPC + RRT*, 500Hz
"""

from src.sensors.imu import IMUSensor, IMUSensorType, PoseEstimator
from src.sensors.tactile import TactileArray, TactileSensorType
from src.sensors.force import ForceTorqueSensor, ForceSensorType
from src.fusion.cross_modal_fusion import CrossModalFusion, FusionConfig

imu = IMUSensor(
    sensor_type=IMUSensorType.ADIS16470,
    sensor_id="imu_0",
    accel_range=40,
    gyro_range=4000,
    sample_rate=1000
)
imu.open()

tactile = TactileArray(
    array_size=(32, 32),
    sensor_type=TactileSensorType.OPTICAL,
    sensor_id="tactile_0"
)
tactile.open()

force = ForceTorqueSensor(
    sensor_type=ForceSensorType.SIX_AXIS,
    sensor_id="ft_0",
    ip_address="192.168.1.102",
    ethernet_type="UDP"
)
force.open()

pose_estimator = PoseEstimator(
    algorithm="madgwick",
    sample_rate=1000.0,
    beta=0.05
)

fusion_config = FusionConfig(
    fusion_strategy="early",
    hidden_dim=768,
    num_heads=8,
    num_layers=4
)
fusion = CrossModalFusion(fusion_config)

print("XL级配置完成")
```

### 3.5 XXL级 (旗舰全功能)

```python
"""
XXL级配置: 全功能旗舰
- IMU: ADIS16470×2 (冗余), ±80g, ±8000°/s, 2000Hz
- 触觉: 48×48 光学式, 16bit, 0-10000kPa, 1000Hz
- 力觉: 6轴 ±5000N/±500N·m, 5000Hz
- 融合: EARLY+HYBRID+LATE, 1024d
- 控制: MPC + 多次RRT*, 1000Hz
"""

from src.sensors.imu import IMUSensor, IMUSensorType, PoseEstimator
from src.sensors.tactile import TactileArray, TactileSensorType
from src.sensors.force import ForceTorqueSensor, ForceSensorType
from src.fusion.cross_modal_fusion import CrossModalFusion, FusionConfig

# 双IMU冗余配置
imu_primary = IMUSensor(
    sensor_type=IMUSensorType.ADIS16470,
    sensor_id="imu_0",
    accel_range=80,
    gyro_range=8000,
    sample_rate=2000
)
imu_secondary = IMUSensor(
    sensor_type=IMUSensorType.ADIS16470,
    sensor_id="imu_1",
    accel_range=80,
    gyro_range=8000,
    sample_rate=2000
)
imu_primary.open()
imu_secondary.open()

# 高分辨率触觉
tactile = TactileArray(
    array_size=(48, 48),
    sensor_type=TactileSensorType.OPTICAL,
    sensor_id="tactile_0"
)
tactile.open()

# 重载力矩
force = ForceTorqueSensor(
    sensor_type=ForceSensorType.SIX_AXIS,
    sensor_id="ft_0",
    ip_address="192.168.1.103",
    ethernet_type="UDP"
)
force.open()

# 高性能融合
fusion_config = FusionConfig(
    fusion_strategy="early",
    hidden_dim=1024,
    num_heads=12,
    num_layers=6
)
fusion = CrossModalFusion(fusion_config)

print("XXL级配置完成")
```

---

## 4. 传感器-控制集成流水线

### 4.1 M级完整闭环示例

```python
"""
M级: 传感器采集 → 融合 → 控制 → 执行 完整闭环
控制频率: 100Hz, 闭环延迟 < 70ms
"""

import numpy as np
import time
import sys
sys.path.insert(0, 'src')

from sensors.imu import IMUSensor, IMUSensorType, PoseEstimator
from sensors.tactile import TactileArray, TactileSensorType
from sensors.force import ForceTorqueSensor, ForceSensorType
from fusion.cross_modal_fusion import CrossModalFusion, FusionConfig, MultimodalInput
from control.imu_control import AttitudeStabilizer
from control.tactile_control import TactileServoController
from control.force_control import ForceController

# --- 初始化 ---
imu = IMUSensor(IMUSensorType.BMI088, sensor_id="imu_0", sample_rate=200)
imu.open()

tactile = TactileArray((16, 16), TactileSensorType.CAPACITIVE, sensor_id="tactile_0")
tactile.open()

force = ForceTorqueSensor(ForceSensorType.SIX_AXIS, sensor_id="ft_0")
force.open()

pose_est = PoseEstimator("madgwick", sample_rate=200.0)
attitude_ctrl = AttitudeStabilizer(kp=2.0, ki=0.1, kd=0.5)
tactile_ctrl = TactileServoController(kp=1.5)
force_ctrl = ForceController(force_threshold=20.0)

fusion = CrossModalFusion(FusionConfig(hidden_dim=256, num_heads=4))

# --- 主循环: 100Hz ---
dt = 0.01  # 10ms
rate_hz = 100

print("启动传感器-控制闭环...")
for step in range(500):
    loop_start = time.time()
    
    # 1. 传感器采集
    imu_frame = imu.capture()
    tactile_frame = tactile.capture()
    force_wrench = force.capture()
    
    # 2. 姿态估计
    pose = pose_est.update(
        imu_frame.accel,
        imu_frame.gyro,
        imu_frame.mag
    )
    euler = pose.to_euler()
    
    # 3. 接触检测
    contact = force.detect_contact(force_wrench)
    contacts = tactile.detect_contacts(tactile_frame)
    slip = tactile.get_slip_signal(tactile_frame)
    grip_quality = tactile.estimate_grip_quality(tactile_frame)
    
    # 4. 传感器融合
    multimodal = MultimodalInput(
        vision=None,
        audio=None,
        tactile=tactile_frame.pressure_map.flatten(),
        force=force_wrench.to_vector(),
        imu=np.concatenate([imu_frame.accel, imu_frame.gyro]),
        language=None,
        timestamp=time.time()
    )
    fused = fusion.fuse(multimodal)
    
    # 5. 控制器计算
    # 姿态稳定控制
    attitude_cmd = attitude_ctrl.compute(
        current_rpy=euler,
        target_rpy=np.array([0.0, 0.0, 0.0]),  # 保持水平
        current_angular_vel=imu_frame.gyro
    )
    
    # 力觉响应
    if contact.is_contact:
        force_response = force_ctrl.compute(contact)
    else:
        force_response = np.zeros(3)
    
    # 触觉伺服
    tactile_cmd = tactile_ctrl.compute(
        tactile_frame,
        target_force=5.0,
        grip_quality=grip_quality
    )
    
    # 6. 安全检查
    # - 速度限制
    # - 边界限制
    # - 力限幅
    safety_ok = (
        np.linalg.norm(attitude_cmd) < 10.0 and
        contact.contact_force < 50.0 and
        grip_quality['overall'] > 0.3
    )
    
    if not safety_ok:
        attitude_cmd = np.zeros(3)
        print(f"[警告] 安全触发! step={step}, grip={grip_quality['overall']:.2f}")
    
    # 7. 执行 (实际部署时发送到电机驱动)
    # motor_driver.send_velocity(attitude_cmd)
    
    # 8. 周期控制
    elapsed = time.time() - loop_start
    sleep_time = dt - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)
    
    if step % 100 == 0:
        print(f"Step {step}: rpy={[f'{x:.3f}' for x in euler]}, "
              f"force={force_wrench.magnitude:.2f}N, "
              f"grip={grip_quality['overall']:.2f}, "
              f"fused_dim={fused.state.shape}, "
              f"loop_time={elapsed*1000:.1f}ms")

print("闭环运行完成")
imu.close()
tactile.close()
force.close()
```

---

## 5. 仿真验证流程

### 5.1 使用PyBullet验证

```python
"""
部署前: 在PyBullet仿真中验证控制器
"""

from src.simulation.pybullet_sim import PyBulletSimulator

# 创建仿真环境
sim = PyBulletSimulator(gui=True)

# 加载AGV URDF
sim.load_agv(
    urdf_path="sim_demos/urdf/agv_m.urdf",
    base_position=[0, 0, 0],
    grade='M'
)

# 运行传感器-控制闭环
for step in range(200):
    # 获取仿真IMU数据
    imu_sim = sim.get_imu_data()
    
    # 运行控制器
    cmd = attitude_ctrl.compute(...)
    
    # 发送控制指令
    sim.send_joint_velocity(cmd)
    
    # 步进仿真
    sim.step()
    
    # 检查碰撞
    if sim.check_collision():
        print("碰撞检测!")
        break

sim.close()
```

### 5.2 五级性能基准测试

```python
"""
验证各级性能是否满足规格
"""

from src.simulation.real_time_monitor import RealTimeMonitor

grades = ['S', 'M', 'L', 'XL', 'XXL']

for grade in grades:
    print(f"\n{'='*50}")
    print(f"测试 {grade} 级性能基准")
    print(f"{'='*50}")
    
    monitor = RealTimeMonitor(grade=grade)
    results = monitor.run_benchmark(num_samples=100)
    
    # 验证延迟
    target_latency = {
        'S': 0.200, 'M': 0.070, 'L': 0.035, 'XL': 0.015, 'XXL': 0.007
    }[grade]
    
    actual_latency = results['sensor_to_control_latency_p95']
    passed = actual_latency < target_latency
    
    print(f"目标延迟: {target_latency*1000:.1f}ms")
    print(f"实际延迟: {actual_latency*1000:.1f}ms")
    print(f"通过: {'✅' if passed else '❌'}")
```

---

## 6. 实机部署检查清单

### 部署前检查

- [ ] 传感器连接测试 (IMU/触觉/力觉/相机)
- [ ] 传感器标定 (IMU零偏、触觉零点、力矩偏置)
- [ ] 通信测试 (ROS2话题/服务/动作)
- [ ] 安全控制器测试 (急停、碰撞检测)
- [ ] 边界测试 (关节限位、速度限制)

### S级部署检查项

```
[ ] MPU6050 I2C 通信正常
[ ] 单目相机USB连接正常
[ ] ROS2 Humble 节点启动
[ ] 50Hz 控制循环稳定
[ ] 安全限速配置 (<0.5m/s)
```

### M级部署检查项

```
[ ] BMI088 SPI 通信正常 (200Hz采样)
[ ] 双目D435i USB3.0连接
[ ] 六轴力矩 USB/CAN 连接
[ ] 16×16触觉 SPI 连接
[ ] RK3588 NPU 推理延迟 <5ms
[ ] 100Hz 控制循环稳定
[ ] 激光导航建图完成
[ ] 安全力限幅配置 (±50N)
```

### L级部署检查项

```
[ ] ADIS/BMI088 高精度IMU 校准
[ ] 双目D455 立体校正
[ ] EtherCAT 实时通信 (1000Hz同步)
[ ] 24×24触觉 标定完成
[ ] 阻抗控制参数整定 (K=500, B=50)
[ ] RRT* 路径规划测试
[ ] 200Hz 控制循环稳定
[ ] ISO 3691-4 安全认证
```

### XL级部署检查项

```
[ ] ADIS16470 双IMU 冗余验证
[ ] 事件相机 1000Hz 低延迟测试
[ ] EtherCAT 2000Hz 同步
[ ] MPC 参数调优 (预测时域=0.5s)
[ ] 500Hz 控制循环稳定 (RT-PREEMPT)
[ ] 多层避障 DWA+APF 测试
[ ] IEC 61508 SIL2 功能安全
```

### XXL级部署检查项

```
[ ] ADIS16470×2 冗余切换测试
[ ] 多目相机+3D LiDAR 外参标定
[ ] 48×48触觉 高速采集验证
[ ] 6轴±5000N 力矩 校准
[ ] MPC+RRT* 多次规划测试
[ ] 1000Hz 控制 (Xenomai+FPGA)
[ ] 多AGV 20台协同测试
[ ] IEC 61508 SIL3 功能安全
[ ] 端到端 <5ms 延迟验证
```

---

## 7. 故障排查

### 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| IMU数据全是零 | I2C地址错误 | 检查0x68/0x69地址配置 |
| 触觉数据噪声大 | SPI信号干扰 | 添加CRC校验，降低采样率 |
| 力矩偏置漂移 | 温度漂移 | 启用温度补偿，每小时重新标定 |
| 融合延迟过高 | GPU内存不足 | 降低hidden_dim或减少融合层数 |
| 控制响应慢 | 实时性不足 | 启用Xenomai/RT-PREEMPT内核 |

### 调试工具

```python
# 使用实时监控器调试
from src.simulation.real_time_monitor import RealTimeMonitor

monitor = RealTimeMonitor(grade='M', verbose=True)
monitor.run_diagnostic(num_samples=50)
# 输出: 传感器延迟、抖动、吞吐量、融合各阶段延迟
```

---

## 8. 版本记录

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| v1.0.0 | 2026-04-07 | 初始版本，包含S/M/L/XL/XXL五级部署指南 |
