# SuperModel 传感器模块 API 实用指南

> 版本: v1.63.0 | 更新: 2026-04-07
> 本指南提供触觉、力觉、IMU 传感器的实用代码示例

---

## 1. 触觉传感器 (TactileArray) 实用指南

### 1.1 快速初始化

```python
from src.sensors.tactile import (
    TactileArray, TactileSensorType, TactileCalibration,
    VirtualTactileSensor, PressureProcessor
)

# 真实传感器初始化
tactile = TactileArray(
    array_size=(16, 16),
    sensor_type=TactileSensorType.CAPACITIVE,
    sensor_id="finger_tip_0",
    calibration=TactileCalibration(force_scale=50.0)
)
tactile.open()
```

### 1.2 连续采集与接触检测

```python
# 采集 100 帧并检测接触
for i in range(100):
    frame = tactile.capture()
    
    # 检测接触区域
    contacts = tactile.detect_contacts(frame)
    
    # 计算滑移信号
    slip = tactile.get_slip_signal(frame)
    
    # 评估抓取质量
    quality = tactile.estimate_grip_quality(frame)
    
    if quality['overall'] > 0.8:
        print(f"Frame {i}: Good grasp! overall={quality['overall']:.3f}")

tactile.close()
```

### 1.3 使用虚拟触觉传感器仿真

```python
from src.sensors.tactile import VirtualTactileSensor

with VirtualTactileSensor((24, 24), "test_tactile") as sensor:
    # 模拟单点接触
    frame = sensor.simulate_contact(
        contact_pos=(0.5, 0.5),
        contact_radius=0.3,
        contact_force=15.0
    )
    
    # 模拟滑移动作 (30帧)
    sliding_frames = sensor.simulate_sliding(
        direction=(0.1, 0.0),
        speed=0.05,
        duration_frames=30
    )
    
    # 模拟多点接触
    multi_frame = sensor.simulate_multi_contact([
        ((0.3, 0.4), 10.0, 0.2),
        ((0.7, 0.6), 8.0, 0.15)
    ])
    
    # 模拟滑移检测
    slip_result = sensor.simulate_slip_detection(
        normal_force=10.0,
        friction_coeff=0.3,
        velocity=(0.05, 0.0)
    )
    print(f"Slip state: {slip_result['slip_state']}")
```

### 1.4 压力信号处理

```python
processor = PressureProcessor(filter_window=3, drift_compensation=True)

# 采集原始帧
frame = tactile.capture()
raw = frame.pressure_map

# 滤波去噪
filtered = processor.filter(raw)

# 基线补偿
compensated = processor.compensate_baseline(filtered, set_baseline=False)

# 计算接触力
contact_area = 1e-4  # 1cm² 传感单元面积
total_force = processor.compute_force(compensated, contact_area)

# 计算压力质心
centroid = processor.compute_centroid(compensated)

# 压力直方图分析
hist, edges = processor.compute_pressure_histogram(compensated, bins=20)
```

### 1.5 AGV五级触觉规格速查

```python
from src.sensors.tactile import get_tactile_spec, AGV_TACTILE_GRADES

# 获取各等级规格
for grade in ['S', 'M', 'L', 'XL', 'XXL']:
    spec = get_tactile_spec(grade)
    print(f"{grade}: {spec['array']} @ {spec['freq_hz']}Hz, "
          f"{spec['res']}bit, {spec['range_kpa']}kPa")
```

---

## 2. 力觉传感器 (ForceTorqueSensor) 实用指南

### 2.1 快速初始化

```python
from src.sensors.force import (
    ForceTorqueSensor, ForceSensorType, ForceCalibration,
    Wrench, VirtualForceSensor, WrenchProcessor
)

# 真实传感器初始化 (六维力矩)
ft_sensor = ForceTorqueSensor(
    sensor_type=ForceSensorType.SIX_AXIS,
    sensor_id="ft_0",
    calibration=ForceCalibration(),
    ip_address="192.168.1.100"
)
ft_sensor.open()

# 关节力矩传感器
joint_ft = ForceTorqueSensor(
    sensor_type=ForceSensorType.JOINT_TORQUE,
    sensor_id="joint_ft_0"
)
joint_ft.open()
```

### 2.2 六维力数据采集与分析

```python
# 采集并处理力数据
processor = WrenchProcessor(filter_alpha=0.3, outlier_threshold=3.0)
wrench_history = []

for i in range(100):
    wrench = ft_sensor.capture()
    wrench_history.append(wrench.to_vector())
    
    # 滤波
    filtered = processor.filter(wrench.to_vector())
    
    # 接触检测
    contact = ft_sensor.detect_contact(wrench, threshold=5.0)
    if contact.is_contact:
        print(f"Contact detected! Force={contact.contact_force:.2f}N, "
              f"Slip={contact.slip_probability:.3f}")
    
    # 负载估计
    payload = ft_sensor.estimate_payload(wrench)
    
    # 力方向
    direction = processor.compute_force_direction(wrench.to_vector())

ft_sensor.close()
```

### 2.3 工具坐标系标定

```python
# 设置工具中心参数
ft_sensor.set_tool_center(
    tool_mass=0.5,       # 工具质量 0.5kg
    tool_com=np.array([0.0, 0.0, 0.05])  # 质心偏移 5cm
)

# 偏置校准 (无负载状态下)
ft_sensor.calibrate_bias(num_samples=200)
```

### 2.4 使用虚拟力传感器仿真

```python
from src.sensors.force import VirtualForceSensor

with VirtualForceSensor("test_force", noise_level=0.02) as sensor:
    # 模拟接触力
    wrench = sensor.simulate_contact(
        force=(0.0, 0.0, -10.0),  # 10N 向下的力
        torque=(0.0, 0.0, 0.0)
    )
    
    # 模拟负载重力
    payload_wrench = sensor.simulate_payload(
        mass=1.0,
        com_offset=(0.0, 0.0, 0.05),
        gravity=9.81
    )
    
    # 模拟碰撞事件 (100ms)
    collision_frames = sensor.simulate_collision(
        direction=(1.0, 0.0, 0.0),
        peak_force=50.0,
        duration_ms=100.0,
        decay="exponential"
    )
    print(f"Collision: {len(collision_frames)} frames")
    
    # 模拟表面接触 (弹簧阻尼模型)
    surface_wrench = sensor.simulate_surface_contact(
        surface_normal=(0.0, 0.0, 1.0),
        contact_point=(0.0, 0.0, 0.0),
        penetration_depth=0.001,  # 1mm 穿透
        stiffness=1000.0,         # 1000 N/m 刚度
        damping=50.0
    )
    
    # 模拟摩擦力
    friction_wrench = sensor.simulate_friction_contact(
        normal_force=10.0,
        velocity=(0.1, 0.0, 0.0),
        friction_coeff=0.3,
        object_mass=1.0
    )
```

### 2.5 Wrench 数据变换与协方差估计

```python
from src.sensors.force import Wrench

wrench = ft_sensor.capture()

# Wrench 属性
print(f"Force magnitude: {wrench.magnitude:.3f} N")
print(f"Torque magnitude: {wrench.torque_magnitude:.3f} N·m")
print(f"Vector: {wrench.to_vector()}")

# 坐标变换 (传感器坐标系 → 世界坐标系)
rotation = np.eye(3)  # 单位旋转
translation = np.array([0.1, 0.0, 0.05])  # 5cm 偏移
world_wrench = wrench.transform(rotation, translation)

# 协方差估计
cov = processor.estimate_covariance(wrench_history)
print(f"Force covariance shape: {cov.shape}")
```

### 2.6 AGV五级力觉规格速查

```python
from src.sensors.force import get_force_spec, AGV_FORCE_GRADES

for grade in ['S', 'M', 'L', 'XL', 'XXL']:
    spec = get_force_spec(grade)
    print(f"{grade}: {spec['axes']}轴, "
          f"力范围±{spec['force_range']}N, "
          f"采样{spec['sampling_hz']}Hz")
```

---

## 3. IMU传感器 (IMUSensor) 实用指南

### 3.1 快速初始化

```python
from src.sensors.imu import (
    IMUSensor, IMUSensorType, IMUCalibration,
    VirtualIMUSensor, PoseEstimator, Pose
)

# 真实IMU初始化
imu = IMUSensor(
    sensor_type=IMUSensorType.BMI088,
    sensor_id="imu_0",
    calibration=IMUCalibration(),
    accel_range=16,    # ±16g
    gyro_range=2000,   # ±2000°/s
    sample_rate=200    # 200Hz
)
imu.open()
```

### 3.2 姿态估计 (Madgwick/互补滤波)

```python
# 初始化姿态估计器
pose_estimator = PoseEstimator(
    algorithm="madgwick",
    sample_rate=200.0,
    beta=0.1
)

# 采集并进行姿态估计
for i in range(1000):
    frame = imu.capture()
    
    # 更新姿态估计
    pose = pose_estimator.update(
        accel=frame.accel,
        gyro=frame.gyro,
        mag=frame.mag,
        dt=1.0/200.0
    )
    
    # 获取欧拉角
    rpy = pose_estimator.get_euler()
    roll, pitch, yaw = rpy * 180 / 3.14159  # 转换为度
    
    # 获取旋转矩阵
    R = pose_estimator.get_rotation_matrix()

print(f"Current pose: position={pose.position}, "
      f"orientation={pose.orientation}")

imu.close()
```

### 3.3 IMU 自检与标定

```python
imu.self_test()  # 自检

# 陀螺仪偏置校准 (静止状态)
imu.calibrate_gyro_bias(num_samples=500, duration_sec=5.0)

# 加速度计标定 (已知朝向)
imu.calibrate_accel(known_orientation="level")
```

### 3.4 使用虚拟IMU仿真

```python
from src.sensors.imu import VirtualIMUSensor

with VirtualIMUSensor("test_imu", accel_noise=0.01, gyro_noise=0.001) as sensor:
    # 模拟静止状态 (水平放置)
    frame = sensor.simulate_static(orientation=(0.0, 0.0, 0.0))
    
    # 模拟运动 (线性加速度 + 角速度)
    frame = sensor.simulate_motion(
        linear_accel=(1.0, 0.5, 0.0),  # m/s²
        angular_vel=(0.0, 0.0, 0.1),    # rad/s
        dt=0.01
    )
    
    # 模拟轨迹
    trajectory_frames = sensor.simulate_trajectory(
        trajectory_type="circle",
        duration_s=2.0,
        dt=0.01
    )
    print(f"Trajectory: {len(trajectory_frames)} frames")
    
    # 模拟AGV运动
    agv_frame = sensor.simulate_agv_motion(
        linear_velocity=(0.5, 0.0),  # 0.5m/s 前进
        angular_velocity=0.1,          # 0.1rad/s 左转
        grade="M"                      # M级AGV噪声特性
    )
    
    # 模拟人类步行
    walk_frames = sensor.simulate_human_walking(
        step_frequency=1.5,
        walk_speed=1.0,
        duration_s=5.0,
        dt=0.01
    )
```

### 3.5 速度/位置积分 (仅短时有效)

```python
pose_estimator = PoseEstimator(algorithm="madgwick", sample_rate=200.0)

imu.open()
for i in range(200):
    frame = imu.capture()
    
    # 更新姿态
    pose = pose_estimator.update(frame.accel, frame.gyro, frame.mag)
    
    # 积分加速度获得速度/位置
    vel, pos = pose_estimator.integrate_velocity(
        frame.accel,
        dt=0.005,
        remove_gravity=True
    )
    
    if i % 50 == 0:
        print(f"Step {i}: vel={vel}, pos={pos}")

# 重置积分漂移
pose_estimator.reset()

imu.close()
```

### 3.6 AGV五级IMU规格速查

```python
from src.sensors.imu import get_imu_spec, AGV_IMU_GRADES

for grade in ['S', 'M', 'L', 'XL', 'XXL']:
    spec = get_imu_spec(grade)
    print(f"{grade}: {spec['type']}, "
          f"Accel±{spec['accel_range']}g, "
          f"Gyro±{spec['gyro_range']}°/s, "
          f"{spec['sample_hz']}Hz, "
          f"噪声密度{spec['noise_density']}μg/√Hz")
```

---

## 4. 多传感器联合使用示例

### 4.1 完整流水线: 触觉+力觉+IMU

```python
import numpy as np
from src.sensors.tactile import TactileArray, TactileSensorType
from src.sensors.force import ForceTorqueSensor, ForceSensorType
from src.sensors.imu import IMUSensor, IMUSensorType, PoseEstimator

# 初始化所有传感器
tactile = TactileArray((16, 16), TactileSensorType.CAPACITIVE, "tactile_0")
force_sensor = ForceTorqueSensor(ForceSensorType.SIX_AXIS, "ft_0")
imu = IMUSensor(IMUSensorType.BMI088, "imu_0", sample_rate=200)
pose_est = PoseEstimator("madgwick", sample_rate=200.0)

# 打开所有传感器
tactile.open()
force_sensor.open()
imu.open()

# 抓取任务主循环
for step in range(500):
    # 采集触觉
    tactile_frame = tactile.capture()
    contacts = tactile.detect_contacts(tactile_frame)
    grip_quality = tactile.estimate_grip_quality(tactile_frame)
    
    # 采集力觉
    wrench = force_sensor.capture()
    contact = force_sensor.detect_contact(wrench)
    
    # 采集IMU并估计姿态
    imu_frame = imu.capture()
    pose = pose_est.update(imu_frame.accel, imu_frame.gyro, imu_frame.mag)
    rpy_deg = pose.to_euler() * 180 / np.pi
    
    # 综合判断
    if contact.is_contact and grip_quality['overall'] > 0.7:
        if abs(wrench.force[2]) > 5.0:  # 有足够的抓取力
            print(f"Step {step}: Stable grasp - "
                  f"force={wrench.magnitude:.1f}N, "
                  f"quality={grip_quality['overall']:.2f}, "
                  f"rpy=[{rpy_deg[0]:.1f}, {rpy_deg[1]:.1f}, {rpy_deg[2]:.1f}]°")

# 关闭所有传感器
tactile.close()
force_sensor.close()
imu.close()
```

### 4.2 虚拟传感器联合仿真

```python
from src.sensors.tactile import VirtualTactileSensor
from src.sensors.force import VirtualForceSensor
from src.sensors.imu import VirtualIMUSensor

# 创建虚拟传感器组
vt = VirtualTactileSensor((16, 16), "v_tactile")
vf = VirtualForceSensor("v_force")
vi = VirtualIMUSensor("v_imu")

vt.open()
vf.open()
vi.open()

# 仿真抓取
print("Simulating grasp...")
for i in range(30):
    # 触觉: 接触中心移动
    if i < 10:
        pos = (0.5 + i*0.01, 0.5)
    else:
        pos = (0.6, 0.5 - (i-10)*0.005)
    
    tactile_frame = vt.simulate_contact(
        contact_pos=pos,
        contact_radius=0.25,
        contact_force=10.0 + np.sin(i*0.3)*2.0
    )
    
    # 力觉: 对应接触力
    force_n = 10.0 + np.sin(i*0.3)*2.0
    wrench = vf.simulate_contact(
        force=(0.0, 0.0, -force_n),
        torque=(0.0, 0.0, 0.5)
    )
    
    # IMU: 轻微摆动
    rpy_offset = (0.1*np.sin(i*0.2), 0.05*np.sin(i*0.3), 0)
    imu_frame = vi.simulate_static(orientation=rpy_offset)
    
    if i % 10 == 0:
        print(f"  Frame {i}: force={wrench.magnitude:.2f}N, "
              f"max_pressure={np.max(tactile_frame.pressure_map):.3f}, "
              f"accel={np.linalg.norm(imu_frame.accel):.2f}m/s²")

vt.close()
vf.close()
vi.close()
```

---

## 5. 编码器集成

### 5.1 传感器数据编码

```python
from src.sensors.encoders import (
    TactileEncoder, ForceEncoder, IMUEncoder,
    SensorEncoderWrapper, EncoderConfig
)

# 配置编码器 (AGV L级)
tactile_enc = TactileEncoder(output_dim=64, hidden_dim=128)
force_enc = ForceEncoder(output_dim=32, hidden_dim=64)
imu_enc = IMUEncoder(output_dim=64, hidden_dim=128)

# 编码触觉数据
tactile_emb = tactile_enc(tactile_frame.pressure_map[np.newaxis])

# 编码力觉数据
force_emb = force_enc(wrench.to_vector()[np.newaxis])

# 编码IMU数据
imu_emb = imu_enc(np.concatenate([
    imu_frame.accel, imu_frame.gyro
])[np.newaxis])

print(f"Tactile: {tactile_emb.shape}, Force: {force_emb.shape}, IMU: {imu_emb.shape}")
```

---

## 6. 故障排除

### 6.1 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 触觉数据全零 | I2C地址错误 | 检查 wiringPi/I2C 配置 |
| 力觉偏置漂移 | 温度变化 | 运行 `calibrate_bias()` |
| IMU角度跳变 | 磁干扰 | 使用 Madgwick 而非互补滤波 |
| 触觉噪声过大 | 电磁干扰 | 使用电容式替代电阻式 |
| 力觉数据饱和 | 超量程 | 降低负载或选大量程传感器 |

### 6.2 标定检查清单

```python
# 触觉标定
tactile.calibrate(
    zero_pressure=np.zeros((16, 16)),
    known_weights=[1.0, 5.0, 10.0]
)

# 力觉标定
ft_sensor.set_tool_center(0.5, np.array([0, 0, 0.05]))
ft_sensor.calibrate_bias(num_samples=200)

# IMU标定
imu.self_test()
imu.calibrate_gyro_bias(num_samples=500)
imu.calibrate_accel("level")
```

---

> 本指南版本: v1.63.0 | 最后更新: 2026-04-07
> 相关文档: `SPEC.md` (技术规格) | `MODULE_INTERFACE.md` (详细接口)
