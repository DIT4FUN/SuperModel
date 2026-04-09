# SuperModel 具身智能实机部署指南

> 实用指南：SuperModel 超模态大模型机器人的实机部署流程、校准步骤和运行检查清单
> 版本: v2.35.0 | 更新: 2026-04-10

---

## 目录

1. [概述](#1-概述)
2. [实机部署流程](#2-实机部署流程)
3. [传感器校准](#3-传感器校准)
4. [控制模块配置](#4-控制模块配置)
5. [五级规格对照表](#5-五级规格对照表)
6. [运行检查清单](#6-运行检查清单)
7. [故障排查](#7-故障排查)

---

## 1. 概述

SuperModel 超模态机器人具身智能大脑支持 S/M/L/XL/XXL 五级配置，从低成本单传感器到全模态高精密系统均可用同一代码库部署。

### 部署前提

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| Python | 3.10+ | 3.12+ |
| 操作系统 | Linux (Ubuntu 22.04+) | Linux + RT 内核 |
| 计算平台 | RK3588 / x86_64 | RK3588 + NPU / x86_64 + GPU |
| 内存 | 4GB | 8GB+ |
| 存储 | 16GB | 32GB+ SSD |

### 支持的五级配置

```python
AGV_GRADE = 'M'  # 可选: S / M / L / XL / XXL
```

---

## 2. 实机部署流程

### 2.1 传感器连接检查

```python
# === 视觉 ===
from src.sensors.vision import BinocularCamera
camera = BinocularCamera(left_id=0, right_id=1)
assert camera.open(), "双目相机连接失败"

# === 触觉 (电子皮肤) ===
from src.sensors.tactile import TactileArray, TactileSensorType
tactile = TactileArray(
    array_size=(16, 16),
    sensor_type=TactileSensorType.CAPACITIVE,
    sensor_id="tactile_0"
)
assert tactile.open(), "触觉传感器连接失败"

# === 力觉 (六维力矩传感器) ===
from src.sensors.force import ForceTorqueSensor, ForceSensorType
ft_sensor = ForceTorqueSensor(
    sensor_type=ForceSensorType.SIX_AXIS,
    sensor_id="ft_0",
    ip_address="192.168.1.100"  # ATI Net F/T
)
assert ft_sensor.open(), "力觉传感器连接失败"

# === IMU ===
from src.sensors.imu import IMUSensor, IMUSensorType
imu = IMUSensor(
    sensor_type=IMUSensorType.BMI088,
    sensor_id="imu_0"
)
assert imu.open(), "IMU传感器连接失败"
```

### 2.2 快速启动模板

```python
#!/usr/bin/env python3
"""
SuperModel 实机启动模板
"""
import numpy as np
from src.sensors.tactile import TactileArray, TactileSensorType
from src.sensors.force import ForceTorqueSensor, ForceSensorType
from src.sensors.imu import IMUSensor, IMUSensorType
from src.control.tactile_control import TactileServoController, TactileServoParams
from src.control.force_control import ForceController, ForceControlParams
from src.control.imu_control import AttitudeStabilizer, IMUControlParams
from src.control.sensor_fusion_control import SensorFusionController, FusionControlConfig
from src.hardware.rk3588 import RK3588Platform

GRADE = 'M'  # S/M/L/XL/XXL

def create_agents(grade: str):
    """根据AGV等级创建具身控制器"""
    
    # 触觉控制
    tactile = TactileArray(array_size=(16, 16), sensor_id="t0")
    tactile_ctrl = TactileServoController(TactileServoParams(grade=grade))
    
    # 力觉控制
    force = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS, sensor_id="f0")
    force_ctrl = ForceController(ForceControlParams(grade=grade))
    
    # IMU控制
    imu = IMUSensor(sensor_type=IMUSensorType.BMI088, sensor_id="i0")
    imu_ctrl = AttitudeStabilizer(IMUControlParams(grade=grade))
    
    # 融合控制
    fusion_cfg = FusionControlConfig(grade=grade)
    fusion_ctrl = SensorFusionController(fusion_cfg)
    
    return {
        'tactile': (tactile, tactile_ctrl),
        'force': (force, force_ctrl),
        'imu': (imu, imu_ctrl),
        'fusion': (fusion_ctrl),
    }

def main():
    # 初始化平台
    platform = RK3588Platform()
    platform.set_power_mode('performance')
    
    # 创建控制器
    agents = create_agents(GRADE)
    
    # 打开所有传感器
    for name, (sensor, ctrl) in agents.items():
        if name != 'fusion':
            sensor.open()
    
    # 运行控制循环
    print(f"[SuperModel] 启动 {GRADE} 级具身智能大脑...")
    
    for step in range(1000):
        # 采集
        t_frame = agents['tactile'][0].capture()
        f_wrench = agents['force'][0].capture()
        i_frame = agents['imu'][0].capture()
        
        # 融合控制
        ctrl_input = {
            'tactile': t_frame,
            'force': f_wrench,
            'imu': i_frame,
        }
        ctrl_output = agents['fusion'].update(ctrl_input)
        
        # 执行
        if ctrl_output.command is not None:
            platform.send_motor_command(ctrl_output.command)
        
        if step % 100 == 0:
            print(f"Step {step}: "
                  f"T={t_frame.pressure_map.mean():.3f}, "
                  f"F={f_wrench.magnitude:.2f}N, "
                  f"IMU={i_frame.accel_magnitude:.2f}m/s²")
    
    # 清理
    for name, (sensor, _) in agents.items():
        if name != 'fusion':
            sensor.close()
    
    print("[SuperModel] 具身智能大脑关闭")

if __name__ == '__main__':
    main()
```

---

## 3. 传感器校准

### 3.1 IMU 校准

```python
from src.sensors.imu import IMUSensor, IMUSensorType, IMUCalibration
from src.control.bias_compensation import IMUBiasCompensator

def calibrate_imu(imu: IMUSensor, duration: int = 5):
    """IMU 静态校准"""
    imu.open()
    
    # 自检
    assert imu.self_test(), "IMU 自检失败"
    
    # 陀螺仪偏置校准 (静止状态)
    imu.calibrate_gyro_bias(num_samples=500)
    
    # 加速度计标定 (水平放置)
    imu.calibrate_accel(known_orientation="level")
    
    # 在线偏置补偿
    compensator = IMUBiasCompensator(imu_type='bmi088', grade='M')
    
    for _ in range(100):
        frame = imu.capture()
        compensated = compensator.compensate(frame)
    
    return imu.calibration

# AGV五级IMU校准规格
IMU_CALIBRATION_GRADES = {
    'S':  {'bias_samples': 200,  'duration_s': 2,   'method': 'static'},
    'M':  {'bias_samples': 500,  'duration_s': 5,   'method': 'static'},
    'L':  {'bias_samples': 1000, 'duration_s': 10,  'method': 'static+temp'},
    'XL': {'bias_samples': 2000, 'duration_s': 20,  'method': 'adaptive'},
    'XXL': {'bias_samples': 5000,'duration_s': 30,  'method': 'adaptive+temp'},
}
```

### 3.2 力觉传感器校准

```python
from src.sensors.force import ForceTorqueSensor, ForceCalibration

def calibrate_force(ft: ForceTorqueSensor):
    """力觉零点校准"""
    ft.open()
    
    # 偏置校准 (无负载状态)
    ft.calibrate_bias(num_samples=100)
    
    # 设置工具中心参数 (用于重力补偿)
    ft.set_tool_center(
        tool_mass=0.55,       # kg
        tool_com=np.array([0, 0, 0.05])  # m, TCP偏移
    )
    
    return ft.calibration

# 校准后验证
wrench = ft.capture()
assert abs(wrench.force[2]) < 1.0, f"力觉零点偏移过大: {wrench.force}"
print(f"力觉零点校准完成: {ft.calibration.bias}")
```

### 3.3 触觉传感器校准

```python
from src.sensors.tactile import TactileArray, TactileCalibration

def calibrate_tactile(tactile: TactileArray):
    """触觉传感器标定"""
    tactile.open()
    
    # 采集零点压力
    frames = [tactile.capture() for _ in range(50)]
    zero_pressure = np.mean([f.pressure_map for f in frames], axis=0)
    
    # 设置标定参数
    calibration = TactileCalibration(
        pressure_min=0.0,
        pressure_max=1.0,
        force_scale=100.0,  # N 满量程
        offset_map=zero_pressure
    )
    
    tactile.calibrate(zero_pressure=zero_pressure)
    print(f"触觉标定完成: scale={calibration.force_scale}N")
    
    return calibration
```

---

## 4. 控制模块配置

### 4.1 五级控制参数

```python
# AGV五级控制参数速查
AGV_CONTROL_PARAMS = {
    'S': {
        'control_rate': 50,        # Hz
        'tactile_rate': 50,        # Hz
        'force_rate': 100,         # Hz
        'imu_rate': 100,           # Hz
        'fusion_rate': 50,         # Hz
        'pid_kp': 1.0,
        'pid_ki': 0.1,
        'pid_kd': 0.05,
        'impedance_stiffness': 100,
        'max_velocity': 1.0,       # m/s
        'safety_distance': 0.5,     # m
    },
    'M': {
        'control_rate': 100,
        'tactile_rate': 100,
        'force_rate': 500,
        'imu_rate': 200,
        'fusion_rate': 100,
        'pid_kp': 1.5,
        'pid_ki': 0.15,
        'pid_kd': 0.08,
        'impedance_stiffness': 200,
        'max_velocity': 1.5,
        'safety_distance': 0.3,
    },
    'L': {
        'control_rate': 200,
        'tactile_rate': 200,
        'force_rate': 1000,
        'imu_rate': 500,
        'fusion_rate': 200,
        'pid_kp': 2.0,
        'pid_ki': 0.2,
        'pid_kd': 0.1,
        'impedance_stiffness': 400,
        'max_velocity': 2.0,
        'safety_distance': 0.2,
    },
    'XL': {
        'control_rate': 500,
        'tactile_rate': 500,
        'force_rate': 2000,
        'imu_rate': 1000,
        'fusion_rate': 500,
        'pid_kp': 2.5,
        'pid_ki': 0.25,
        'pid_kd': 0.12,
        'impedance_stiffness': 600,
        'max_velocity': 2.5,
        'safety_distance': 0.15,
    },
    'XXL': {
        'control_rate': 1000,
        'tactile_rate': 1000,
        'force_rate': 5000,
        'imu_rate': 2000,
        'fusion_rate': 1000,
        'pid_kp': 3.0,
        'pid_ki': 0.3,
        'pid_kd': 0.15,
        'impedance_stiffness': 800,
        'max_velocity': 3.0,
        'safety_distance': 0.1,
    },
}
```

### 4.2 融合控制五级规格

```python
# 传感器融合控制五级规格表
FUSION_CONTROL_SPECS = {
    'S':  {'fused_modalities': 2,  'rate': 50,   'filter': 'complementary', '闭环延迟': '<100ms'},
    'M':  {'fused_modalities': 3,  'rate': 100,  'filter': 'EKF',          '闭环延迟': '<50ms'},
    'L':  {'fused_modalities': 4,  'rate': 200,  'filter': 'adaptive_EKF', '闭环延迟': '<20ms'},
    'XL': {'fused_modalities': 5,  'rate': 500,  'filter': 'neural',       '闭环延迟': '<10ms'},
    'XXL':{'fused_modalities': 6,  'rate': 1000, 'filter': 'GNN',          '闭环延迟': '<5ms'},
}
```

---

## 5. 五级规格对照表

### 5.1 感知子系统

| 感知参数 | S级 | M级 | L级 | XL级 | XXL级 |
|---------|-----|-----|-----|------|-------|
| 触觉阵列 | 8×8 | 16×16 | 24×24 | 32×32 | 48×48 |
| 触觉采样率 | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| 力觉轴数 | 3轴 | 6轴 | 6轴 | 6轴 | 6轴 |
| 力觉采样率 | 100Hz | 500Hz | 1000Hz | 2000Hz | 5000Hz |
| IMU型号 | MPU6050 | BMI088 | BMI088×2 | ADIS16470×2 | ADIS16470×4 |
| IMU采样率 | 100Hz | 200Hz | 500Hz | 1000Hz | 2000Hz |
| 相机 | 单目 | 双目 | 双目+深度 | 双目+深度+广角 | 多目+深度 |

### 5.2 控制子系统

| 控制参数 | S级 | M级 | L级 | XL级 | XXL级 |
|---------|-----|-----|-----|------|-------|
| 控制频率 | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| 控制策略 | PID | 阻抗 | 力位混合 | 自适应阻抗 | 预测控制 |
| 力控带宽 | — | 5Hz | 10Hz | 20Hz | 50Hz |
| 触觉反馈 | — | 接触检测 | 滑移检测 | 抓取质量 | 精细操作 |
| 故障容错 | 单传感器 | 双冗余 | 三冗余 | 多模态容错 | 智能切换 |

### 5.3 计算子系统

| 计算参数 | S级 | M级 | L级 | XL级 | XXL级 |
|---------|-----|-----|-----|------|-------|
| 计算平台 | RK3588 | RK3588+NPU | RK3588+NPU | x86+GPU | x86+GPU集群 |
| TOPS | 6 | 6+8 | 6+16 | 16+32 | 64+256 |
| 内存 | 4GB | 8GB | 16GB | 32GB | 128GB |
| 通信 | CAN | CAN+以太网 | EtherCAT | EtherCAT+光纤 | 多路光纤 |

---

## 6. 运行检查清单

### 部署前检查

```bash
# 1. 硬件连接检查
ls /dev/video*        # 相机
ls /dev/i2c-*        # I2C设备
ip link show         # 网络接口

# 2. 传感器连接测试
python -c "
from src.sensors.tactile import TactileArray
from src.sensors.force import ForceTorqueSensor
from src.sensors.imu import IMUSensor
# 测试各传感器连接
"

# 3. 校准数据加载
ls ~/.supermodel/calibration/
# 应包含: imu_calibration.yaml, force_calibration.yaml, tactile_calibration.yaml
```

### 运行时检查

```python
# 运行时状态监控
from src.control.sensor_fusion_control import SensorFusionController

def health_check(controller: SensorFusionController):
    """运行时健康检查"""
    issues = []
    
    # 检查传感器帧率
    if controller.fps < controller.target_fps * 0.8:
        issues.append(f"FPS过低: {controller.fps:.1f} < {controller.target_fps}")
    
    # 检查力觉偏置
    if controller.last_wrench and controller.last_wrench.magnitude > 50.0:
        issues.append(f"力觉偏置异常: {controller.last_wrench.magnitude:.1f}N")
    
    # 检查IMU温度
    if controller.last_imu and controller.last_imu.temperature > 45.0:
        issues.append(f"IMU温度过高: {controller.last_imu.temperature:.1f}°C")
    
    # 检查触觉接触
    contacts = controller.last_tactile_contacts
    if len(contacts) > 20:
        issues.append(f"触觉噪声过多: {len(contacts)}个接触点")
    
    if issues:
        print(f"[警告] 健康检查发现问题: {issues}")
        return False
    return True
```

---

## 7. 故障排查

### 常见问题

| 症状 | 可能原因 | 解决方案 |
|------|---------|---------|
| 触觉数据全零 | I2C未连接 | 检查接线/地址(0x18) |
| 力觉偏置漂移 | 未进行零点校准 | 运行calibrate_bias() |
| IMU姿态跳动 | 陀螺仪偏置未补偿 | 运行calibrate_gyro_bias() |
| 融合控制震荡 | PID参数过大 | 降低kp/ki，增设kd |
| 控制延迟 > 100ms | 计算负载过高 | 降低融合模态数量 |
| 触觉噪声过大 | 传感器线缆干扰 | 屏蔽线缆/降低采样率 |
| 力觉饱和 | 碰撞/过载 | 检查物理约束 |

### 偏置补偿故障排查

```python
from src.control.bias_compensation import BiasCompensationSystem

def diagnose_bias_issues():
    """偏置补偿问题诊断"""
    system = BiasCompensationSystem()
    
    # 检查各传感器偏置状态
    for sensor in ['tactile', 'force', 'imu']:
        status = system.get_status(sensor)
        print(f"{sensor}: bias_drift={status['bias_drift']:.4f}, "
              f"drift_rate={status['drift_rate']:.4f}/s, "
              f"confidence={status['confidence']:.2f}")
        
        if status['confidence'] < 0.5:
            print(f"  → 建议: 重新校准 {sensor}")
        if abs(status['drift_rate']) > 0.1:
            print(f"  → 警告: {sensor} 漂移率过高")
```

---

## 版本历史

- **v2.35.0** (2026-04-10): 新增实机部署指南; 包含传感器连接检查/校准流程/五级参数配置/运行检查清单/故障排查
- **v2.34.0** (2026-04-10): 具身智能大脑集成测试; 2215项测试全通过
- **v2.33.0** (2026-04-10): 传感器偏置补偿模块; AGV五级规格表完善
