# SuperModel 传感器-控制集成实战指南

> **文档版本**: v1.0.0
> **最后更新**: 2026-03-31
> **项目**: SuperModel 超模态机器人具身智能大脑

---

## 概述

本文档提供 SuperModel 传感器模块与控制模块集成的实战指南，涵盖从传感器数据采集到运动控制的完整闭环流程，适用于 AGV 五级规格 (S/M/L/XL/XXL) 的实际部署场景。

---

## 1. 典型集成架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        SuperModel Pipeline                        │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  传感器层  │→│  融合层   │→│  认知层   │→│   控制层     │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
│                                                                  │
│  Vision       CrossModal   WorldModel    MotionController        │
│  Audio        Fusion       Dreamer       Impedance               │
│  Tactile      (Attention)  Agent        MPC                     │
│  Force                      (Planning)   SafetyController        │
│  IMU                                                           │
│  Encoders                                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 传感器数据采集

### 2.1 初始化所有传感器

```python
import numpy as np
import sys
sys.path.insert(0, 'src')

from sensors.vision import BinocularCamera, get_stereo_spec
from sensors.audio import BinauralMic
from sensors.tactile import TactileArray, TactileSensorType, get_tactile_spec
from sensors.force import ForceTorqueSensor, ForceSensorType
from sensors.imu import IMUSensor, IMUSensorType, PoseEstimator
from sensors.manager import SensorManager
from sensors.encoders import JointEncoder

# 创建传感器管理器 (AGV-M 等级)
grade = 'M'
manager = SensorManager()

# 添加各传感器
manager.add_sensor('camera', BinocularCamera(
    resolution=get_stereo_spec(grade)['resolution'],
    fps=get_stereo_spec(grade)['fps']
))
manager.add_sensor('mic', BinauralMic(sample_rate=16000))
manager.add_sensor('tactile', TactileArray(
    array_size=get_tactile_spec(grade)['array'],
    sensor_type=TactileSensorType.CAPACITIVE
))
manager.add_sensor('force', ForceTorqueSensor(
    sensor_type=ForceSensorType.SIX_AXIS
))
manager.add_sensor('imu', IMUSensor(sensor_type=IMUSensorType.BMI088))
manager.add_sensor('encoders', JointEncoder(num_joints=6))

# 一键打开所有传感器
manager.open_all()

print(f"已打开 {manager.num_sensors} 个传感器")
```

### 2.2 周期性数据采集

```python
import time

def capture_cycle(manager, dt=0.01):
    """控制周期内的数据采集"""
    data = {}
    
    # 视觉 (降低频率，与控制周期解耦)
    if manager.get_sensor('camera'):
        camera_data = manager.get_sensor('camera').capture()
        data['vision'] = {
            'left': camera_data.left_image,
            'right': camera_data.right_image,
            'timestamp': camera_data.timestamp
        }
    
    # 听觉
    if manager.get_sensor('mic'):
        audio_data = manager.get_sensor('mic').capture()
        direction = manager.get_sensor('mic').get_sound_direction(audio_data)
        data['audio'] = {
            'direction': direction,
            'timestamp': audio_data.timestamp
        }
    
    # 触觉
    if manager.get_sensor('tactile'):
        tactile_data = manager.get_sensor('tactile').capture()
        contacts = manager.get_sensor('tactile').detect_contacts(tactile_data)
        data['tactile'] = {
            'pressure_map': tactile_data.pressure_map,
            'contacts': contacts,
            'timestamp': tactile_data.timestamp
        }
    
    # 力觉
    if manager.get_sensor('force'):
        wrench = manager.get_sensor('force').capture()
        contact = manager.get_sensor('force').detect_contact(wrench)
        data['force'] = {
            'wrench': wrench,
            'contact_detected': contact.is_contact,
            'timestamp': wrench.timestamp
        }
    
    # IMU
    if manager.get_sensor('imu'):
        imu_data = manager.get_sensor('imu').capture()
        data['imu'] = {
            'accel': imu_data.accel,
            'gyro': imu_data.gyro,
            'timestamp': imu_data.timestamp
        }
    
    # 关节编码器
    if manager.get_sensor('encoders'):
        encoder_data = manager.get_sensor('encoders').read()
        data['encoders'] = {
            'positions': encoder_data.positions,
            'velocities': encoder_data.velocities
        }
    
    return data
```

---

## 3. 跨模态融合

### 3.1 创建融合网络

```python
import torch
from fusion.cross_modal_fusion import (
    CrossModalFusion, FusionConfig, FusionStrategy
)

# AGV-M 等级融合配置
config = FusionConfig(
    vision_dim=512,    # 双目视觉特征维度
    audio_dim=128,      # 双耳音频特征维度
    tactile_dim=64,     # 触觉特征维度
    force_dim=32,       # 力觉特征维度
    imu_dim=64,         # IMU特征维度
    hidden_dim=256,     # 融合隐层维度
    num_heads=4,        # 注意力头数
    num_layers=2,       # 融合层数
    strategy=FusionStrategy.HYBRID  # 混合融合策略
)

fusion_net = CrossModalFusion(config)

# 注册模态重要性权重 (AGV-M 等级)
fusion_net.register_modality_weights(
    vision=1.0,
    audio=0.8,
    tactile=0.9,
    force=1.0,
    imu=0.7
)
```

### 3.2 特征提取与融合

```python
def extract_features(data):
    """从原始传感器数据提取融合特征"""
    features = {}
    
    # 视觉特征 (简化: 使用均值池化)
    if 'vision' in data and data['vision'] is not None:
        left_img = data['vision']['left']
        # 实际应用中应使用 CNN/Transformer 提取特征
        features['vision'] = torch.tensor(
            left_img.mean(axis=(0,1)),  # 简化为全局均值
            dtype=torch.float32
        ).unsqueeze(0)
    
    # 音频特征 (简化为方向信息的编码)
    if 'audio' in data and data['audio']['direction'] is not None:
        direction = data['audio']['direction']
        features['audio'] = torch.tensor(
            [np.sin(direction), np.cos(direction)],
            dtype=torch.float32
        ).unsqueeze(0)
    
    # 触觉特征 (压力图的空间统计)
    if 'tactile' in data:
        pm = data['tactile']['pressure_map']
        features['tactile'] = torch.tensor([
            pm.mean(), pm.std(), pm.max(),
            len(data['tactile']['contacts'])
        ], dtype=torch.float32).unsqueeze(0)
    
    # 力觉特征 (末端六维力矩)
    if 'force' in data:
        w = data['force']['wrench']
        features['force'] = torch.tensor(
            list(w.force) + list(w.torque),
            dtype=torch.float32
        ).unsqueeze(0)
    
    # IMU特征 (加速度/角速度/姿态)
    if 'imu' in data:
        accel = data['imu']['accel']
        gyro = data['imu']['gyro']
        features['imu'] = torch.tensor(
            list(accel) + list(gyro),
            dtype=torch.float32
        ).unsqueeze(0)
    
    return features


def fuse_sensor_data(fusion_net, features):
    """执行跨模态融合"""
    from fusion.cross_modal_fusion import MultimodalInput
    
    # 构建多模态输入
    multimodal = MultimodalInput(
        vision=features.get('vision'),
        audio=features.get('audio'),
        tactile=features.get('tactile'),
        force=features.get('force'),
        imu=features.get('imu')
    )
    
    # 前向传播
    fused = fusion_net(multimodal)
    return fused
```

---

## 4. 运动控制

### 4.1 运动控制器初始化

```python
from control.motion import MotionController, JointState, ControlMode
from control.impedance import ImpedanceController, ImpedanceParams
from control.safety_controller import SafetyController, SafetyConfig, SafetyLevel
from control.agv import AGVMotionController, AGVGrade

num_joints = 6

# 关节空间 PID 控制器
motion_ctrl = MotionController(
    num_joints=num_joints,
    control_rate=100.0,  # AGV-M: 100Hz
    control_mode=ControlMode.JOINT_TORQUE
)
motion_ctrl.set_pid_gains(
    kp=np.ones(num_joints) * 2.0,
    ki=np.ones(num_joints) * 0.1,
    kd=np.ones(num_joints) * 0.5
)

# 阻抗控制器 (用于人机协作)
imp_ctrl = ImpedanceController(
    impedance_params=ImpedanceParams.default_6d()
)

# 安全控制器 (AGV-M 等级: M级安全)
safety_cfg = SafetyConfig(
    safety_level=SafetyLevel.M,
    max_joint_velocity=np.array([1.0] * num_joints),
    max_joint_accel=np.array([5.0] * num_joints),
    collision_threshold=50.0,
    workspace_limits=np.array([
        [-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5],
        [-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]
    ])
)
safety_ctrl = SafetyController(config=safety_cfg)
```

### 4.2 安全控制循环

```python
def control_loop(motion_ctrl, safety_ctrl, data, fused_features):
    """主控制循环 (100Hz)"""
    
    # 1. 获取当前关节状态
    if 'encoders' in data:
        current_state = JointState(
            position=data['encoders']['positions'],
            velocity=data['encoders']['velocities'],
            torque=np.zeros(num_joints)
        )
    else:
        current_state = JointState(
            position=np.zeros(num_joints),
            velocity=np.zeros(num_joints),
            torque=np.zeros(num_joints)
        )
    
    # 2. 目标轨迹 (从融合特征解码或手动设定)
    target_position = np.array([0.5, 0.3, -0.2, 0.1, 0.4, 0.0])
    
    # 3. 安全检查
    safety_result = safety_ctrl.check_all(
        joint_state=current_state,
        commanded_position=target_position,
        external_wrench=data.get('force', {}).get('wrench')
    )
    
    # 4. 根据安全响应执行
    if safety_result.response.level == SafetyLevel.XXL:
        # 紧急停止
        torque = np.zeros(num_joints)
    elif safety_result.response.level == SafetyLevel.XL:
        # 看门狗触发，降低速度
        target_position = safety_result.response.suggested_position
        torque = motion_ctrl.compute_joint_torque(target_position)
        motion_ctrl.update_joint_state(current_state)
    else:
        # 正常控制
        torque = motion_ctrl.compute_joint_torque(target_position)
        motion_ctrl.update_joint_state(current_state)
    
    # 5. 力矩限幅
    torque = np.clip(torque, -50.0, 50.0)
    
    return torque, safety_result
```

---

## 5. AGV 运动控制

### 5.1 AGV 运动学控制

```python
from control.agv import (
    AGVMotionController, DriveType, AGVGrade,
    DifferentialKinematics, MecanumKinematics
)

# 创建 AGV-M 运动控制器 (麦克纳姆轮)
agv_ctrl = AGVMotionController(
    grade=AGVGrade.M,
    drive_type=DriveType.MECANUM,
    wheel_radius=0.05,    # 0.05m
    track_width=0.4,      # 0.4m
    wheelbase=0.5         # 0.5m
)

# 设置目标速度 (vx, vy, omega)
target_twist = {'vx': 0.5, 'vy': 0.0, 'omega': 0.0}  # 0.5m/s 前进

# 逆运动学: 末端速度 → 轮速
wheel_velocities = agv_ctrl.inverse_kinematics(
    vx=target_twist['vx'],
    vy=target_twist['vy'],
    omega=target_twist['omega']
)
print(f"轮速命令: {wheel_velocities}")

# 正运动学: 轮速 → 末端速度
odometry = agv_ctrl.forward_kinematics(wheel_velocities)
print(f"估计速度: vx={odometry.vx:.3f}, vy={odometry.vy:.3f}, omega={odometry.omega:.3f}")
```

### 5.2 轨迹跟踪

```python
from control.agv import TrajectoryTracker
import numpy as np

# 创建轨迹跟踪器
tracker = TrajectoryTracker(
    max_linear_speed=0.5,   # AGV-M: 0.5m/s
    max_angular_speed=1.0,  # 1.0 rad/s
    look_ahead_time=0.5,     # 前瞻时间
    k_alpha=1.0,            # 角度误差增益
    k_beta=-0.5,            # 横向误差增益
    k_delta=1.0             # 距离误差增益
)

# 定义目标轨迹 (圆弧)
t = np.linspace(0, 2*np.pi, 100)
reference_path = np.column_stack([
    1.0 * np.cos(t),        # x
    1.0 * np.sin(t),        # y
    0.2 * t                 # theta (朝向)
])

# 轨迹跟踪
current_pose = {'x': 1.0, 'y': 0.0, 'theta': 0.0}
current_twist = {'vx': 0.0, 'vy': 0.0, 'omega': 0.0}

for i in range(len(reference_path)):
    cmd = tracker.compute_command(
        current_pose=current_pose,
        current_twist=current_twist,
        reference_path=reference_path,
        current_idx=i
    )
    # 发送速度命令给 AGV
    # wheel_velocities = agv_ctrl.inverse_kinematics(cmd['vx'], cmd['vy'], cmd['omega'])
```

---

## 6. 完整闭环示例

```python
def supermodel闭环(grade='M', duration=10.0, dt=0.01):
    """SuperModel 完整闭环示例"""
    
    # 初始化
    manager, fusion_net, motion_ctrl, safety_ctrl = setup_pipeline(grade)
    
    # 姿态估计器 (用于 IMU 融合)
    pose_estimator = PoseEstimator(algorithm='madgwick', sample_rate=1.0/dt)
    
    # 主循环
    num_steps = int(duration / dt)
    for step in range(num_steps):
        # 1. 传感器采集
        sensor_data = capture_cycle(manager, dt)
        
        # 2. 特征提取
        features = extract_features(sensor_data)
        
        # 3. 跨模态融合
        fused = fuse_sensor_data(fusion_net, features)
        
        # 4. IMU 姿态估计
        if 'imu' in sensor_data:
            pose = pose_estimator.update(
                sensor_data['imu']['accel'],
                sensor_data['imu']['gyro']
            )
        
        # 5. 控制输出
        torque, safety = control_loop(
            motion_ctrl, safety_ctrl, sensor_data, fused
        )
        
        # 6. 发送力矩命令到执行器
        # send_torque_command(torque)
        
        # 7. 日志记录 (每100步)
        if step % 100 == 0:
            print(f"Step {step}/{num_steps}: "
                  f"torque={[f'{t:.2f}' for t in torque[:3]]} "
                  f"safety={safety.response.level.name}")
    
    # 关闭
    manager.close_all()
    print("闭环完成!")


def setup_pipeline(grade='M'):
    """初始化完整流水线"""
    # 传感器
    manager = SensorManager()
    # ... 添加传感器 ...
    manager.open_all()
    
    # 融合网络
    config = FusionConfig(vision_dim=512, audio_dim=128, 
                          tactile_dim=64, force_dim=32, imu_dim=64,
                          hidden_dim=256, num_heads=4, num_layers=2)
    fusion_net = CrossModalFusion(config)
    
    # 控制器
    motion_ctrl = MotionController(num_joints=6, control_rate=100.0)
    safety_cfg = SafetyConfig(safety_level=SafetyLevel.M)
    safety_ctrl = SafetyController(config=safety_cfg)
    
    return manager, fusion_net, motion_ctrl, safety_ctrl
```

---

## 7. AGV 五级集成对照表

| 功能 | S | M | L | XL | XXL |
|------|:---:|:---:|:---:|:---:|:---:|
| 传感器管理 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 同步采集 | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| 特征提取 | 均值 | 统计 | CNN | CNN+Transformer | 多尺度CNN |
| 跨模态融合 | 拼接 | 拼接+注意力 | 混合融合 | 动态权重 | 动态权重+图 |
| PID控制 | 位置 | 位置+速度前馈 | 力矩+位置 | 自适应 | 非线性MPC |
| 安全控制器 | S级 | M级 | L级 | XL级 | XXL级 |
| AGV驱动 | 差速 | 麦克纳姆 | 麦克纳姆+阿克曼 | 全向+阿克曼 | 全向+故障容忍 |

---

## 8. 常见问题与解决方案

### Q1: 传感器数据延迟不一致
```python
# 使用时间戳对齐不同频率的传感器数据
sensor_buffer = {}

def align_sensors(data, target_timestamp, max_dt=0.05):
    """时间戳对齐"""
    aligned = {}
    for key, value in data.items():
        if hasattr(value, 'timestamp'):
            dt = abs(value.timestamp - target_timestamp)
            if dt < max_dt:
                aligned[key] = value
    return aligned
```

### Q2: 力觉传感器漂移
```python
# 定期零点标定
def auto_zero(force_sensor, num_samples=100):
    """自动零点漂移补偿"""
    samples = []
    for _ in range(num_samples):
        wrench = force_sensor.capture()
        samples.append(np.concatenate([wrench.force, wrench.torque]))
    
    zero_offset = np.mean(samples, axis=0)
    return zero_offset
```

### Q3: IMU 姿态发散
```python
# 使用互补滤波融合加速度计和陀螺仪
from sensors.imu import ComplementaryFilter

comp_filter = ComplementaryFilter(
    alpha=0.98,  # 陀螺仪权重
    accel_bias=0.02
)

def fused_imu(accel, gyro, dt=0.01):
    """互补滤波姿态估计"""
    pitch_accel = np.arctan2(accel[1], np.sqrt(acc[0]**2 + accel[2]**2))
    roll_accel = np.arctan2(-accel[0], accel[2])
    
    # 陀螺仪积分
    pitch_gyro += gyro[0] * dt
    roll_gyro += gyro[1] * dt
    
    # 互补融合
    pitch = alpha * pitch_gyro + (1-alpha) * pitch_accel
    roll = alpha * roll_gyro + (1-alpha) * roll_accel
    
    return pitch, roll
```

---

*文档版本: v1.0.0*
*最后更新: 2026-03-31*
*维护者: SuperModel Dev Team / DIT4FUN*
