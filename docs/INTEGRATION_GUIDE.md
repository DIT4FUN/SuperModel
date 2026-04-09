# SuperModel 超模态大模型 - 完整集成指南

> **版本**: v2.06.0  
> **更新**: 2026-04-09  
> **状态**: 完整可用

本文档提供 SuperModel 超模态大模型 AGV 具身智能系统的完整集成指南，包括传感器→融合→控制→执行的全链路接口规范、五级AGV配置对照、以及端到端集成示例。

---

## 1. 系统架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    SuperModel 具身智能大脑                        │
├─────────────┬─────────────┬──────────────┬─────────────────────┤
│  SENSOR层   │   FUSION层  │  CONTROL层   │    EXECUTION层      │
├─────────────┼─────────────┼──────────────┼─────────────────────┤
│ vision.py   │cross_modal  │ motor.py     │ simulation/          │
│ audio.py    │ _fusion.py  │ motion.py    │   pybullet_sim.py   │
│ tactile.py  │sensor_fusion│ agv.py       │   mujoco_sim.py     │
│ force.py    │             │ mpc.py       │   gazebo_sim.py     │
│ imu.py      │             │ safety_ctrl  │   gym_env.py        │
│ encoders.py │             │ imu_ctrl    │   physics_sim.py    │
│ manager.py  │             │ force_ctrl   │ hardware/           │
│             │             │ tactile_ctrl │   rk3588.py         │
│             │             │ supervisor   │   digu_robot.py     │
│             │             │ planner      │                     │
│             │             │ navigation  │                     │
└─────────────┴─────────────┴──────────────┴─────────────────────┘
```

### 1.1 数据流

```
传感器 → 感知编码 → 跨模态融合 → 决策规划 → 控制指令 → 执行器
  ↓          ↓            ↓            ↓          ↓        ↓
 原始数据  特征向量    融合状态      轨迹/策略   电机指令   物理响应
 (多模态)  (d维)     (统一表示)    (规划路径)  (PWM/CAN)  (仿真/真机)
```

---

## 2. 传感器→融合→控制 全链路接口

### 2.1 触觉链路 (Tactile Pipeline)

```
TactileArray.capture()
    ↓ TactileFrame (pressure_map, temperature_map, proximity, slip_signal)
TactileArray.detect_contacts()
    ↓ List[TactileContact] (center, area, peak_pressure, contact_force, slip_probability)
TactileServoController.compute_control_signal()
    ↓ np.ndarray [dx, dy, dz] (位置控制增量)
    OR
TactileServoController.detect_and_react_slip()
    ↓ np.ndarray [Fx, Fy, Fz] (滑移补偿力)
```

**示例代码**:
```python
from src.sensors.tactile import TactileArray, TactileSensorType, get_tactile_spec
from src.control.tactile_control import TactileServoController, TactileServoParams

# 按AGV等级初始化触觉系统
GRADE = 'M'
spec = get_tactile_spec(GRADE)

tactile = TactileArray(
    array_size=spec['array'],
    sensor_type=TactileSensorType.CAPACITIVE,
    sensor_id="tactile_main"
)
tactile.open()

params = TactileServoParams.from_grade(GRADE)
controller = TactileServoController(tactile, params)

# 主循环
for _ in range(100):
    frame = tactile.capture()
    contacts = tactile.detect_contacts(frame)
    
    if contacts:
        control = controller.compute_control_signal(target_force=5.0, current_frame=frame)
        slip_comp = controller.detect_and_react_slip(frame)
        
    quality = tactile.estimate_grip_quality(frame)
    print(f"抓取质量: {quality['overall']:.3f}")

tactile.close()
```

### 2.2 力觉链路 (Force Pipeline)

```
ForceTorqueSensor.capture()
    ↓ Wrench (force[3], torque[3], timestamp, frame_id)
WrenchProcessor.filter() / remove_outliers()
    ↓ np.ndarray [Fx, Fy, Fz, Tx, Ty, Tz] (滤波后)
ForceTorqueSensor.detect_contact()
    ↓ ContactState (is_contact, contact_force, slip_probability)
ForceTorqueSensor.estimate_payload()
    ↓ float (kg)
ForceControl.compute_force_control()
    ↓ np.ndarray [Fx, Fy, Fz] (力控指令)
ImpedanceController.compute_impedance()
    ↓ np.ndarray [dx, dy, dz] (位置调节量)
```

**示例代码**:
```python
from src.sensors.force import ForceTorqueSensor, ForceSensorType, get_force_spec, WrenchProcessor
from src.control.force_control import ImpedanceController

GRADE = 'M'
spec = get_force_spec(GRADE)

force = ForceTorqueSensor(
    sensor_type=ForceSensorType.SIX_AXIS,
    sensor_id="ft_0"
)
force.open()
force.calibrate_bias(num_samples=100)

processor = WrenchProcessor(filter_alpha=0.3)
impedance = ImpedanceController(K=500.0, D=50.0)

for _ in range(100):
    wrench = force.capture()
    wrench_vec = processor.filter(wrench.to_vector())
    
    contact = force.detect_contact(wrench)
    if contact.is_contact:
        impedance_delta = impedance.compute_impedance(
            wrench_vec[:3], target_force=10.0, dt=0.01
        )

force.close()
```

### 2.3 IMU链路 (IMU Pipeline)

```
IMUSensor.capture()
    ↓ IMUFrame (accel[3], gyro[3], mag[3], temperature, timestamp)
IMUSensor.self_test() → bool
PoseEstimator.update(accel, gyro, mag, dt)
    ↓ Pose (position[3], orientation[4])
Pose.to_euler() → np.ndarray [roll, pitch, yaw]
AttitudeStabilizer.compute_stabilization()
    ↓ np.ndarray [torque_x, torque_y, torque_z] (姿态稳定力矩)
IMUControlParams.from_grade(GRADE) → IMUControlParams
```

**示例代码**:
```python
from src.sensors.imu import IMUSensor, IMUSensorType, PoseEstimator, get_imu_spec
from src.control.imu_control import AttitudeStabilizer, IMUControlParams

GRADE = 'M'
spec = get_imu_spec(GRADE)

imu = IMUSensor(
    sensor_type=IMUSensorType.BMI088,
    sensor_id="imu_0",
    sample_rate=spec['sample_hz']
)
imu.open()
imu.calibrate_gyro_bias(num_samples=500)

estimator = PoseEstimator(algorithm='madgwick', sample_rate=spec['sample_hz'])
params = IMUControlParams.from_grade(GRADE)
stabilizer = AttitudeStabilizer(params)

for _ in range(200):
    frame = imu.capture()
    pose = estimator.update(frame.accel, frame.gyro, frame.mag, dt=1.0/spec['sample_hz'])
    
    euler = pose.to_euler()
    stabilization = stabilizer.compute_stabilization(pose)
    
    print(f"姿态: roll={euler[0]:.3f} pitch={euler[1]:.3f} yaw={euler[2]:.3f}")

imu.close()
```

---

## 3. 五级AGV完整配置对照表

### 3.1 整车规格

| 参数 | S级 | M级 | L级 | XL级 | XXL级 |
|------|:---:|:---:|:---:|:---:|:---:|
| **负载** | 30kg | 100kg | 300kg | 600kg | 1200kg |
| **自重** | 15kg | 35kg | 80kg | 150kg | 300kg |
| **车体尺寸** | 0.4×0.3×0.12m | 0.6×0.4×0.15m | 0.8×0.6×0.2m | 1.0×0.7×0.25m | 1.2×0.9×0.3m |
| **最高速度** | 0.5m/s | 1.5m/s | 2.0m/s | 2.5m/s | 3.0m/s |
| **定位精度** | ±10mm | ±5mm | ±3mm | ±1mm | ±0.5mm |
| **防护等级** | IP20 | IP30 | IP54 | IP65 | IP67 |

### 3.2 传感器配置

| 模态 | S级 | M级 | L级 | XL级 | XXL级 |
|------|:---:|:---:|:---:|:---:|:---:|
| **触觉阵列** | 8×8, 50Hz | 16×16, 100Hz | 24×24, 200Hz | 32×32, 500Hz | 48×48, 1000Hz |
| **力觉** | 3轴±100N, 100Hz | 6轴±200N, 500Hz | 6轴±500N, 1kHz | 6轴±1000N, 2kHz | 6轴±5000N, 5kHz |
| **IMU** | MPU6050, 100Hz | BMI088, 200Hz | BMI088×2, 500Hz | ADIS×2, 1kHz | ADIS×4, 2kHz |
| **视觉** | 单目640×480 | 双目D435i | 双目D455 | 双目+事件相机 | 多目+3D LiDAR |

### 3.3 控制配置

| 控制参数 | S级 | M级 | L级 | XL级 | XXL级 |
|--------|:---:|:---:|:---:|:---:|:---:|
| **控制频率** | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| **控制模式** | 位置 | 位置+速度 | 位置+速度+阻抗 | 全模态 | 全模态+MPC |
| **闭环延迟** | <200ms | <80ms | <35ms | <15ms | <7ms |

### 3.4 计算配置

| 计算参数 | S级 | M级 | L级 | XL级 | XXL级 |
|--------|:---:|:---:|:---:|:---:|:---:|
| **处理器** | RPi 4B | RK3588/Nano | Orin NX | Orin AGX | Orin AGX×2+GPU |
| **AI算力** | <5 TOPS | 5-20 TOPS | 20-100 TOPS | 100-300 TOPS | >300 TOPS |
| **功耗** | <10W | 15-30W | 30-80W | 80-150W | 150-500W |

---

## 4. 端到端集成示例

### 4.1 最小集成 (单AGV)

```python
"""
SuperModel 最小集成示例
适用: S/M级 AGV
功能: 传感器采集 → 融合 → 控制 → 仿真
"""
import numpy as np
import time
import sys
sys.path.insert(0, '/home/treeman/.openclaw/workspace/projects/SuperModel/src')

from sensors.tactile import TactileArray, TactileSensorType, get_tactile_spec
from sensors.force import ForceTorqueSensor, ForceSensorType, get_force_spec
from sensors.imu import IMUSensor, IMUSensorType, get_imu_spec, PoseEstimator
from control.tactile_control import TactileServoController, TactileServoParams
from control.force_control import ImpedanceController
from control.imu_control import AttitudeStabilizer, IMUControlParams

GRADE = 'M'

# === 1. 传感器初始化 ===
tactile_spec = get_tactile_spec(GRADE)
force_spec = get_force_spec(GRADE)
imu_spec = get_imu_spec(GRADE)

tactile = TactileArray(array_size=tactile_spec['array'], sensor_id="t0")
force = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS, sensor_id="f0")
imu = IMUSensor(sensor_type=IMUSensorType.BMI088, sensor_id="i0", 
                sample_rate=imu_spec['sample_hz'])

tactile.open()
force.open()
imu.open()

# === 2. 控制器初始化 ===
tactile_params = TactileServoParams.from_grade(GRADE)
tactile_ctrl = TactileServoController(tactile, tactile_params)

imu_params = IMUControlParams.from_grade(GRADE)
stabilizer = AttitudeStabilizer(imu_params)
estimator = PoseEstimator(algorithm='madgwick', sample_rate=imu_spec['sample_hz'])

impedance = ImpedanceController(K=500.0, D=50.0)

# === 3. 主循环 ===
print(f"[SuperModel] 启动 {GRADE}级 AGV 集成系统")
print(f"  触觉: {tactile_spec['array']} @ {tactile_spec.get('freq_hz', 100)}Hz")
print(f"  力觉: {force_spec['axes']}轴 @ {force_spec['sampling_hz']}Hz")
print(f"  IMU: {imu_spec['type']} @ {imu_spec['sample_hz']}Hz")

dt = 1.0 / tactile_params.control_rate
start_time = time.time()
frame_count = 0

try:
    while frame_count < 100:
        t0 = time.time()
        
        # 触觉采集
        t_frame = tactile.capture()
        contacts = tactile.detect_contacts(t_frame)
        slip = tactile.get_slip_signal(t_frame)
        
        # 力觉采集
        wrench = force.capture()
        contact = force.detect_contact(wrench)
        
        # IMU采集
        imu_frame = imu.capture()
        pose = estimator.update(imu_frame.accel, imu_frame.gyro, dt=dt)
        
        # 控制计算
        if contacts:
            tactile_ctrl.compute_control_signal(target_force=5.0, current_frame=t_frame)
            tactile_ctrl.detect_and_react_slip(t_frame)
        
        if contact.is_contact:
            impedance.compute_impedance(wrench.to_vector()[:3], target_force=10.0, dt=dt)
        
        stabilizer.compute_stabilization(pose)
        
        # 监控
        if frame_count % 20 == 0:
            quality = tactile.estimate_grip_quality(t_frame)
            euler = pose.to_euler()
            print(f"  t={time.time()-start_time:.1f}s | 接触={len(contacts)} | "
                  f"力={wrench.magnitude:.2f}N | 姿态=({euler[0]:.2f}, {euler[1]:.2f})")
        
        # 等待到下一个控制周期
        elapsed = time.time() - t0
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        frame_count += 1

finally:
    tactile.close()
    force.close()
    imu.close()
    print(f"[SuperModel] 集成系统关闭，共运行 {frame_count} 帧")
```

### 4.2 五级AGV自动配置

```python
"""
SuperModel 五级AGV自动配置系统
根据AGV等级自动配置所有传感器和控制器
"""

def create_agv_system(grade: str):
    """工厂函数: 按AGV等级创建完整系统"""
    from sensors.tactile import TactileArray, TactileSensorType, get_tactile_spec
    from sensors.force import ForceTorqueSensor, ForceSensorType, get_force_spec
    from sensors.imu import IMUSensor, IMUSensorType, get_imu_spec
    from control.tactile_control import TactileServoController, TactileServoParams
    from control.force_control import ImpedanceController
    from control.imu_control import AttitudeStabilizer, IMUControlParams

    spec_map = {
        'tactile': get_tactile_spec(grade),
        'force': get_force_spec(grade),
        'imu': get_imu_spec(grade),
    }
    
    # 传感器
    tactile = TactileArray(
        array_size=spec_map['tactile']['array'],
        sensor_id=f"tactile_{grade}"
    )
    force = ForceTorqueSensor(
        sensor_type=ForceSensorType.SIX_AXIS if spec_map['force']['axes'] == 6 else ForceSensorType.THREE_AXIS,
        sensor_id=f"force_{grade}"
    )
    imu = IMUSensor(
        sensor_type=getattr(IMUSensorType, spec_map['imu']['type'], IMUSensorType.BMI088),
        sensor_id=f"imu_{grade}",
        sample_rate=spec_map['imu']['sample_hz']
    )
    
    # 控制器
    tactile_params = TactileServoParams.from_grade(grade)
    tactile_ctrl = TactileServoController(tactile, tactile_params)
    
    imu_params = IMUControlParams.from_grade(grade)
    stabilizer = AttitudeStabilizer(imu_params)
    estimator = PoseEstimator(algorithm='madgwick', sample_rate=spec_map['imu']['sample_hz'])
    
    impedance = ImpedanceController(K=500.0, D=50.0)
    
    return {
        'sensors': {'tactile': tactile, 'force': force, 'imu': imu},
        'controllers': {'tactile': tactile_ctrl, 'imu': stabilizer, 'estimator': estimator, 'impedance': impedance},
        'specs': spec_map,
        'grade': grade
    }

# 使用示例
for grade in ['S', 'M', 'L', 'XL', 'XXL']:
    system = create_agv_system(grade)
    print(f"AGV {grade}: 触觉={system['specs']['tactile']['array']}, "
          f"力觉={system['specs']['force']['axes']}轴, "
          f"IMU={system['specs']['imu']['type']}")
```

---

## 5. 模块接口速查

| 模块文件 | 主要类/函数 | 输入 | 输出 |
|---------|-----------|------|------|
| `tactile.py` | `TactileArray.capture()` | - | `TactileFrame` |
| `tactile.py` | `TactileArray.detect_contacts()` | `TactileFrame` | `List[TactileContact]` |
| `tactile.py` | `TactileArray.get_slip_signal()` | `TactileFrame` | `np.ndarray` |
| `tactile.py` | `VirtualTactileSensor.simulate_contact()` | pos, force, radius | `TactileFrame` |
| `force.py` | `ForceTorqueSensor.capture()` | - | `Wrench` |
| `force.py` | `ForceTorqueSensor.detect_contact()` | `Wrench` | `ContactState` |
| `force.py` | `WrenchProcessor.filter()` | `np.ndarray[6]` | `np.ndarray[6]` |
| `force.py` | `VirtualForceSensor.simulate_collision()` | dir, peak, duration | `List[Wrench]` |
| `imu.py` | `IMUSensor.capture()` | - | `IMUFrame` |
| `imu.py` | `PoseEstimator.update()` | accel, gyro, dt | `Pose` |
| `imu.py` | `VirtualIMUSensor.simulate_agv_motion()` | vx, vy, omega, dt | `IMUFrame` |
| `tactile_control.py` | `TactileServoController.compute_control_signal()` | target_force, frame | `np.ndarray[3]` |
| `tactile_control.py` | `TactileServoController.detect_and_react_slip()` | frame | `np.ndarray[3]` |
| `force_control.py` | `ImpedanceController.compute_impedance()` | force, target, dt | `np.ndarray[3]` |
| `imu_control.py` | `AttitudeStabilizer.compute_stabilization()` | pose | `np.ndarray[3]` |

---

## 6. 测试与验证

```bash
# 运行传感器测试
cd ~/.openclaw/workspace/projects/SuperModel
python3 -m pytest tests/sensor_tests.py -v

# 运行融合测试
python3 -m pytest tests/fusion_tests.py -v

# 运行五级集成测试
python3 -m pytest tests/five_grade_integration_tests.py -v

# 运行全部测试
python3 -m pytest tests/ -v --tb=short
```

预期结果: **全部通过**

---

## 7. 版本历史

| 版本 | 日期 | 主要内容 |
|------|------|---------|
| v2.06.0 | 2026-04-09 | 新增完整集成指南，传感器→融合→控制全链路接口规范 |
| v2.05.1 | 2026-04-09 | 1857项测试全通过，新增部署清单 |
| v2.05.0 | 2026-04-09 | 传感器-控制模块详细接口规范 |
| v2.04.0 | 2026-04-09 | 具身传感控制仿真测试完成 |
