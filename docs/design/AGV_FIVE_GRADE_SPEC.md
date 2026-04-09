# SuperModel AGV五级规格总表
> **文档版本**: v1.0.0  
> **更新**: 2026-04-10  
> **项目**: SuperModel 超模态机器人具身智能大脑

本文档整合 SuperModel AGV 五级 (S/M/L/XL/XXL) 的全部规格参数，提供完整的规格对照表。

---

## 一、整车基础规格

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **负载能力** | 30kg | 100kg | 300kg | 600kg | 1200kg |
| **自重** | 15kg | 35kg | 80kg | 150kg | 300kg |
| **最大总重** | 45kg | 135kg | 380kg | 750kg | 1500kg |
| **车体尺寸** | 0.4×0.3×0.12m | 0.6×0.4×0.15m | 0.8×0.6×0.2m | 1.0×0.7×0.25m | 1.2×0.9×0.3m |
| **轮子配置** | 2轮驱动 | 2轮驱动 | 4轮驱动 | 4轮驱动 | 4轮驱动 |
| **轮子直径** | 100mm | 140mm | 140mm | 165mm | 200mm |
| **电机类型** | 57步进 | 5.5寸轮毂150W | 5.5寸150W×2 | 6.5寸200W×2 | 7.5寸300W×4 |
| **最高速度** | 0.5m/s | 1.5m/s | 2.0m/s | 2.5m/s | 3.0m/s |
| **最大扭矩** | 5Nm | 15Nm | 30Nm | 60Nm | 120Nm |
| **定位精度** | ±10mm | ±5mm | ±3mm | ±1mm | ±0.5mm |
| **防护等级** | IP20 | IP30 | IP54 | IP65 | IP67 |
| **典型价格** | ¥5-15K | ¥15-50K | ¥50-150K | ¥150-500K | >¥500K |

---

## 二、感知子系统规格

### 2.1 视觉 (Vision)

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **配置** | 单目640×480 | 双目D435i 720p | 双目D455 60fps | 双目+事件相机 | 多目+3D LiDAR |
| **基线** | — | 50mm | 50mm | 75mm | 100mm |
| **深度范围** | — | 0.2-5m | 0.2-8m | 0.1-10m | 0.1-30m |
| **分辨率** | 640×480 | 1280×720 | 1280×720 | 1920×1080 | 1920×1080×4 |
| **帧率** | 30fps | 30fps | 60fps | 90fps | 120fps |
| **编码器维度** | — | 256 | 512 | 768 | 1024 |

### 2.2 听觉 (Audio)

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **麦克风** | 1ch | 2ch阵列 | 4ch阵列 | 6ch阵列 | 8ch阵列 |
| **采样率** | 16000Hz | 16000Hz | 22050Hz | 32000Hz | 44100Hz |
| **拾音范围** | 1.0m | 3.0m | 5.0m | 8.0m | 10.0m |
| **波束形成** | ✗ | ✓ | ✓ | ✓ | ✓多波束 |
| **声源定位精度** | — | ±15° | ±5° | ±2° | ±1° |
| **编码器维度** | 64 | 128 | 128 | 256 | 256 |

### 2.3 触觉 (Tactile)

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **阵列尺寸** | 8×8 | 16×16 | 24×24 | 32×32 | 48×48 |
| **分辨率** | 12bit | 12bit | 14bit | 14bit | 16bit |
| **压力范围** | 0-500kPa | 0-1000kPa | 0-2000kPa | 0-5000kPa | 0-10000kPa |
| **采样频率** | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| **温度感知** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **接近觉** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **滑移检测** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **接口** | I2C | SPI | USB | USB/ETH | EtherCAT |

```python
# 使用示例
from src.sensors.tactile import TactileArray, TactileSensorType, get_tactile_spec
spec = get_tactile_spec('L')  # L级规格
ta = TactileArray(array_size=spec['array'], sensor_type=TactileSensorType.CAPACITIVE)
with ta:
    frame = ta.capture()
    contacts = ta.detect_contacts(frame)
    quality = ta.estimate_grip_quality(frame)
```

### 2.4 力觉 (Force)

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **轴数** | 3 | 6 | 6 | 6 | 6 |
| **力范围** | ±100N | ±200N | ±500N | ±1000N | ±5000N |
| **力矩范围** | ±10N·m | ±20N·m | ±50N·m | ±100N·m | ±500N·m |
| **分辨率** | 0.1N | 0.05N | 0.02N | 0.01N | 0.005N |
| **采样频率** | 100Hz | 500Hz | 1000Hz | 2000Hz | 5000Hz |

```python
# 使用示例
from src.sensors.force import ForceTorqueSensor, ForceSensorType, get_force_spec
spec = get_force_spec('L')
sensor = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
with sensor:
    wrench = sensor.capture()
    contact = sensor.detect_contact(wrench)
    payload = sensor.estimate_payload(wrench)
```

### 2.5 IMU

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **型号** | MPU6050 | BMI088 | BMI088 | ADIS16470 | ADIS16470 |
| **加速度量程** | ±8g | ±16g | ±24g | ±40g | ±80g |
| **陀螺仪量程** | ±1000°/s | ±2000°/s | ±4000°/s | ±4000°/s | ±8000°/s |
| **采样频率** | 100Hz | 200Hz | 500Hz | 1000Hz | 2000Hz |
| **噪声密度** | 400μg/√Hz | 120μg/√Hz | 60μg/√Hz | 20μg/√Hz | 10μg/√Hz |

```python
# 使用示例
from src.sensors.imu import IMUSensor, IMUSensorType, PoseEstimator, get_imu_spec
spec = get_imu_spec('L')
imu = IMUSensor(sensor_type=IMUSensorType.BMI088, sample_rate=spec['sample_hz'])
with imu:
    imu.calibrate_gyro_bias()
    frame = imu.capture()
    pose_est = PoseEstimator(algorithm='madgwick', beta=0.1)
    pose = pose_est.update(frame.accel, frame.gyro)
```

### 2.6 编码器 (Encoders)

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **分辨率** | 128d | 256d | 512d | 768d | 1024d |
| **类型** | 增量式 | 增量式256CPR | 增量式512CPR | 增量式768CPR | 增量式1024CPR |

---

### 2.7 偏置补偿 (Bias Compensation)

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **加速度计偏置限制** | ±0.5 m/s² | ±0.3 m/s² | ±0.2 m/s² | ±0.1 m/s² | ±0.05 m/s² |
| **陀螺仪偏置限制** | ±0.1 rad/s | ±0.05 rad/s | ±0.02 rad/s | ±0.01 rad/s | ±0.005 rad/s |
| **力偏置限制** | ±10 N | ±5 N | ±2 N | ±1 N | ±0.5 N |
| **力矩偏置限制** | ±1 N·m | ±0.5 N·m | ±0.2 N·m | ±0.1 N·m | ±0.05 N·m |
| **偏置适应率** | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 |
| **静止检测窗口** | 3.0 s | 2.0 s | 1.5 s | 1.0 s | 0.5 s |
| **静止加速度阈值** | 0.05 m/s² | 0.05 m/s² | 0.05 m/s² | 0.05 m/s² | 0.05 m/s² |
| **静止角速度阈值** | 0.02 rad/s | 0.02 rad/s | 0.02 rad/s | 0.02 rad/s | 0.02 rad/s |
| **温度补偿** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **漂移率限制** | 0.001 N/s | 0.001 N/s | 0.0005 N/s | 0.0002 N/s | 0.0001 N/s |
| **触觉偏置限制** | ±0.1 | ±0.1 | ±0.05 | ±0.02 | ±0.01 |
| **支持IMU** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **支持力传感器** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **支持触觉传感器** | ✗ | ✓ | ✓ | ✓ | ✓ |

**代码示例**:
```python
from src.control.bias_compensation import (
    MultiSensorBiasCompensator, IMUBiasEstimator, ForceBiasEstimator,
    get_agv_bias_spec_table
)

# 获取五级规格
specs = get_agv_bias_spec_table()

# 创建M级补偿器
comp = MultiSensorBiasCompensator(grade='M')
comp.initialize_tactile((16, 16))

# IMU偏置更新与补偿
g = np.array([0.0, 0.0, 9.81])
accel_bias = np.array([0.05, -0.03, 0.02])
comp.imu_estimator.accel_bias = accel_bias.copy()

raw_accel = g + accel_bias  # 含偏置的读数
comp_accel, comp_gyro = comp.compensate_imu(raw_accel, np.zeros(3))

# 力传感器偏置更新与补偿
comp.force_estimator.force_bias = np.array([1.0, -0.5, 0.2])
comp.force_estimator.torque_bias = np.array([0.05, -0.02, 0.01])
comp_f, comp_t = comp.compensate_force(
    np.array([1.0, -0.5, 0.2]),
    np.array([0.05, -0.02, 0.01]),
    dt=0.0
)
assert np.linalg.norm(comp_f) < 0.001  # 应补偿至接近零

# 获取统计信息
stats = comp.step()
print(f"Total bias magnitude: {stats['total_bias_mag']:.4f}")
print(f"Average bias: {stats['avg_bias_mag']:.4f}")
```

---

## 三、控制子系统规格

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **控制频率** | 50Hz | 100Hz | 200Hz | 500Hz | 1000Hz |
| **控制模式** | 位置 | 位置+速度 | 位置+速度+阻抗 | 全模态 | 全模态+MPC |
| **避障算法** | 人工势场 | DWA | DWA+VFH+APF | 混合 | 多层融合 |
| **轨迹规划** | 直线 | RRT | RRT*+样条 | MPC+RRT* | MPC+多次RRT* |
| **碰撞响应** | >100ms | <50ms | <20ms | <10ms | <5ms |
| **姿态稳定** | <500ms | <200ms | <100ms | <50ms | <20ms |
| **实时控制** | ✗ | ✗ | ✓ Xenomai | ✓ RT-PREEMPT | ✓ Xenomai+FPGA |

---

## 四、计算与通信规格

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **处理器** | RPi 4B | RK3588/Nano | Orin NX | Orin AGX | Orin AGX×2+GPU |
| **AI算力** | <5 TOPS | 5-20 TOPS | 20-100 TOPS | 100-300 TOPS | >300 TOPS |
| **内存** | 4GB | 8GB | 16-32GB | 64-128GB | 256+GB |
| **功耗** | <10W | 15-30W | 30-80W | 80-150W | 150-500W |
| **有线通信** | USB | USB/ETH | Ethernet | EtherCAT | EtherCAT+光纤 |
| **无线通信** | WiFi | WiFi | WiFi+5G | 5G+LoRa | 5G+卫星 |
| **多机协同** | ✗ | ✗ | ✗ | ✓ 5台 | ✓ 20台+ |

---

## 五、感知→控制闭环延迟规格

| 阶段 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **传感器采样** | 20ms | 10ms | 5ms | 2ms | 1ms |
| **特征提取** | 80ms | 30ms | 15ms | 5ms | 2ms |
| **融合推理** | 30ms | 10ms | 5ms | 2ms | 1ms |
| **决策规划** | 20ms | 10ms | 5ms | 2ms | 1ms |
| **控制计算** | 10ms | 5ms | 2ms | 1ms | 0.5ms |
| **电机响应** | 40ms | 15ms | 5ms | 2ms | 1ms |
| **总延迟** | <200ms | <80ms | <35ms | <15ms | <7ms |

---

## 六、融合网络规格

| 参数 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **融合策略** | 拼接 | 拼接+门控 | 跨模态注意力 | 多头跨模态 | 多头+层级 |
| **隐层维度** | 128 | 256 | 512 | 768 | 1024 |
| **注意力头数** | 2 | 4 | 8 | 12 | 16 |
| **编码器总数** | 3 | 5 | 8 | 12 | 16 |
| **参数量** | ~10M | ~50M | ~200M | ~500M | ~1B |

---

## 七、具身任务能力

| 任务 | S | M | L | XL | XXL |
|------|:--:|:--:|:--:|:--:|:--:|
| **抓取** | ✗ | ✓ 简单 | ✓ 中等 | ✓ 复杂 | ✓ 精密 |
| **放置** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **推** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **拉** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **表面追踪** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **插入** | ✗ | ✗ | ✗ | ✓ | ✓ |
| **抛光** | ✗ | ✗ | ✗ | ✓ | ✓ |
| **自主学习** | ✗ | 简单 | 中等 | 高级 | 在线 |

---

## 八、使用代码示例

### 八级完整流水线

```python
import numpy as np
import sys
sys.path.insert(0, 'src')

from sensors.tactile import TactileArray, TactileSensorType, get_tactile_spec
from sensors.force import ForceTorqueSensor, ForceSensorType, get_force_spec
from sensors.imu import IMUSensor, IMUSensorType, get_imu_spec, PoseEstimator
from fusion.cross_modal_fusion import CrossModalFusion, FusionConfig, MultimodalInput
from control.sensorimotor import SensorimotorIntegration, SensorimotorConfig
from control.supervisor import ControlSupervisor, SupervisorConfig, get_supervisor_config

# 选择AGV等级
GRADE = 'L'

# 1. 初始化传感器
tactile_spec = get_tactile_spec(GRADE)
force_spec = get_force_spec(GRADE)
imu_spec = get_imu_spec(GRADE)

tactile = TactileArray(
    array_size=tactile_spec['array'],
    sensor_type=TactileSensorType.CAPACITIVE
)
force = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
imu = IMUSensor(sensor_type=IMUSensorType.BMI088, sample_rate=imu_spec['sample_hz'])

# 2. 打开所有传感器
tactile.open()
force.open()
imu.open()

# 3. 初始化融合网络
fusion_cfg = FusionConfig(
    hidden_dim=512,
    num_heads=8,
    dropout=0.1,
    grade=GRADE
)
fusion = CrossModalFusion(fusion_cfg)

# 4. 初始化传感-运动控制
sensorimotor_cfg = SensorimotorConfig.from_grade(GRADE)
sensorimotor = SensorimotorIntegration(sensorimotor_cfg)

# 5. 初始化监管器
supervisor_cfg = get_supervisor_config(GRADE)
supervisor = ControlSupervisor(config=supervisor_cfg)

# 6. 主循环
pose_est = PoseEstimator(algorithm='madgwick', beta=0.1)

for step in range(1000):
    # 采集数据
    t_frame = tactile.capture()
    wrench = force.capture()
    imu_frame = imu.capture()
    
    # 姿态估计
    pose = pose_est.update(imu_frame.accel, imu_frame.gyro)
    
    # 触觉分析
    contacts = tactile.detect_contacts(t_frame)
    grip_quality = tactile.estimate_grip_quality(t_frame)
    
    # 力觉分析
    contact_state = force.detect_contact(wrench)
    
    # 构建多模态输入
    multimodal = MultimodalInput(
        vision=torch.randn(1, 3, 224, 224),  # 示例
        audio=torch.randn(1, 256),
        tactile=torch.from_numpy(t_frame.pressure_map).float().unsqueeze(0).unsqueeze(0),
        force=torch.from_numpy(wrench.to_vector()).float().unsqueeze(0),
        imu=torch.from_numpy(np.concatenate([imu_frame.accel, imu_frame.gyro])).float().unsqueeze(0),
        text=torch.randint(0, 1000, (1, 32)),
    )
    
    # 融合
    fused = fusion(multimodal)
    
    # 传感-运动控制
    sensorimotor_state = sensorimotor.update(
        tactile_frame=t_frame,
        wrench=wrench,
        imu_frame=imu_frame,
        desired_force=None
    )
    
    # 监管
    supervisor.update(sensorimotor_state, fused)
    
    if step % 100 == 0:
        print(f"Step {step}: grip={grip_quality['overall']:.3f}, "
              f"contact={contact_state.is_contact}, "
              f"pose=({pose.position[0]:.3f},{pose.position[1]:.3f})")

# 关闭
tactile.close()
force.close()
imu.close()
```
