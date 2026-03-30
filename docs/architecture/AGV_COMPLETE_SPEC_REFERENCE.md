# SuperModel AGV五级完整规格速查手册

> **文档版本**: v1.0.0
> **最后更新**: 2026-03-30
> **用途**: 一页速查所有 AGV 五级 (S/M/L/XL/XXL) 规格

---

## 一、传感器规格速查

### 1.1 视觉 (Vision)

```python
from sensors.vision import get_stereo_spec
for g in ['S','M','L','XL','XXL']:
    print(g, get_stereo_spec(g))
```

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 基线 mm | — | 50 | 50 | 75 | 100 |
| 深度范围 m | — | 0.2-5 | 0.2-8 | 0.1-10 | 0.1-30 |
| 分辨率 | — | 1280×720 | 1280×720 | 1920×1080 | 1920×1080×4 |
| 帧率 fps | 30 | 30 | 60 | 90 | 120 |
| 编码器维度 | — | 256 | 512 | 768 | 1024 |

### 1.2 听觉 (Audio)

```python
from sensors.audio import get_audio_spec
```

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 麦克风 | 1 | 2 | 4阵列 | 6阵列 | 8阵列 |
| 采样率 Hz | 16000 | 16000 | 22050 | 32000 | 44100 |
| 拾音范围 m | 1.0 | 3.0 | 5.0 | 8.0 | 10.0 |
| 波束形成 | ✗ | ✓ | ✓ | ✓ | ✓多波束 |
| 声源定位精度 | — | ±15° | ±5° | ±2° | ±1° |
| 编码器维度 | 64 | 128 | 128 | 256 | 256 |

### 1.3 触觉 (Tactile)

```python
from sensors.tactile import get_tactile_spec, TactileArray, TactileSensorType
# 使用示例:
spec = get_tactile_spec('L')
ta = TactileArray(array_size=spec['array'], sensor_type=TactileSensorType.CAPACITIVE)
ta.open()
frame = ta.capture()
contacts = ta.detect_contacts(frame)
```

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 阵列尺寸 | 8×8 | 16×16 | 24×24 | 32×32 | 48×48 |
| 分辨率 bit | 12 | 12 | 14 | 14 | 16 |
| 压力范围 kPa | 0-500 | 0-1000 | 0-2000 | 0-5000 | 0-10000 |
| 采样频率 Hz | 50 | 100 | 200 | 500 | 1000 |
| 温度感知 | ✗ | ✓ | ✓ | ✓ | ✓ |
| 接近觉 | ✗ | ✗ | ✓ | ✓ | ✓ |
| 滑移检测 | ✗ | ✓ | ✓ | ✓ | ✓ |
| 接口 | I2C | SPI | USB | USB/ETH | EtherCAT |

### 1.4 力觉 (Force)

```python
from sensors.force import get_force_spec, ForceTorqueSensor, ForceSensorType
# 使用示例:
spec = get_force_spec('L')
sensor = ForceTorqueSensor(sensor_type=ForceSensorType.SIX_AXIS)
sensor.open()
wrench = sensor.capture()
contact = sensor.detect_contact(wrench)
```

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 轴数 | 3 | 6 | 6 | 6 | 6 |
| 力范围 N | ±100 | ±200 | ±500 | ±1000 | ±5000 |
| 力矩范围 N·m | ±10 | ±20 | ±50 | ±100 | ±500 |
| 分辨率 | 0.1N | 0.05N | 0.02N | 0.01N | 0.005N |
| 采样频率 Hz | 100 | 500 | 1000 | 2000 | 5000 |

### 1.5 IMU

```python
from sensors.imu import get_imu_spec, IMUSensor, IMUSensorType, PoseEstimator
# 使用示例:
spec = get_imu_spec('L')
imu = IMUSensor(sensor_type=IMUSensorType.BMI088, sample_rate=spec['sample_hz'])
imu.open()
imu.calibrate_gyro_bias()
frame = imu.capture()
pose_est = PoseEstimator(algorithm='madgwick', beta=0.1)
pose = pose_est.update(frame.accel, frame.gyro)
```

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 型号 | MPU6050 | BMI088 | BMI088 | ADIS16470 | ADIS16470 |
| 加速度程 g | 8 | 16 | 24 | 40 | 80 |
| 陀螺仪程 °/s | 1000 | 2000 | 4000 | 4000 | 8000 |
| 采样频率 Hz | 100 | 200 | 500 | 1000 | 2000 |
| 噪声密度 μg/√Hz | 400 | 120 | 60 | 20 | 10 |

### 1.6 编码器融合维度

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 视觉编码器 | — | 256 | 512 | 768 | 1024 |
| 听觉编码器 | 64 | 128 | 128 | 256 | 256 |
| 触觉编码器 | 32 | 64 | 64 | 128 | 128 |
| 力觉编码器 | 16 | 32 | 32 | 64 | 64 |
| IMU编码器 | 32 | 32 | 64 | 64 | 128 |
| 融合隐层维度 | 128 | 256 | 512 | 768 | 1024 |
| 输出表示维度 | 64 | 128 | 256 | 512 | 1024 |

---

## 二、融合系统规格速查

```python
from fusion.cross_modal_fusion import FusionConfig, CrossModalFusion, FusionStrategy

config = FusionConfig(
    hidden_dim=512, num_heads=8, num_layers=4,
    strategy=FusionStrategy.HYBRID
)
fusion = CrossModalFusion(config)
```

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 融合策略 | 晚期 | 中期 | 中期 | 混合 | 全阶段 |
| 隐层维度 | 128 | 256 | 512 | 768 | 1024 |
| 注意力头数 | 2 | 4 | 8 | 12 | 16 |
| 融合层数 | 1 | 2 | 4 | 6 | 8 |
| 推理延迟 ms | <50 | <20 | <10 | <5 | <2 |

---

## 三、执行控制规格速查

### 3.1 运动控制

```python
from control.agv import AGVMotionController, AGVSpec, AGVGrade

spec = AGVSpec.from_grade(AGVGrade.L)
agv = AGVMotionController(spec)
```

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 控制频率 Hz | 50 | 100 | 200 | 500 | 1000 |
| 位置精度 mm | ±5 | ±1 | ±0.5 | ±0.1 | ±0.01 |
| 力控精度 N | ±1 | ±0.5 | ±0.2 | ±0.1 | ±0.05 |
| 最大负载 kg | 2 | 5 | 20 | 50 | 200 |
| 最大线速度 m/s | 0.5 | 1.0 | 2.0 | 3.0 | 5.0 |

### 3.2 控制模式

| 模式 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 关节位置控制 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 关节速度控制 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 关节力矩控制 | — | ✓ | ✓ | ✓ | ✓ |
| 笛卡尔位置控制 | — | ✓ | ✓ | ✓ | ✓ |
| 笛卡尔速度控制 | — | ✓ | ✓ | ✓ | ✓ |
| 位置阻抗控制 | — | ✓ | ✓ | ✓ | ✓ |
| 力阻抗控制 | — | — | ✓ | ✓ | ✓ |
| 力位混合控制 | — | — | ✓ | ✓ | ✓ |
| 导纳控制 | — | — | ✓ | ✓ | ✓ |
| 自适应阻抗 | — | — | — | ✓ | ✓ |
| MPC | — | ✓ | ✓ | ✓ | ✓ |

### 3.3 轨迹规划

| 算法 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 线性插值 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 三次多项式 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 五次多项式 | — | ✓ | ✓ | ✓ | ✓ |
| S曲线 | — | ✓ | ✓ | ✓ | ✓ |
| RRT | ✓ | ✓ | ✓ | ✓ | ✓ |
| RRT* | — | ✓ | ✓ | ✓ | ✓ |
| PRM | — | — | ✓ | ✓ | ✓ |
| CHOMP | — | — | — | ✓ | ✓ |

### 3.4 MPC控制器

```python
from control.mpc import MPCConfig, JointSpaceMPC, get_mpc_spec

spec = get_mpc_spec('L')
config = MPCConfig(
    horizon=30, control_horizon=15, dt=0.01,
    solver='osqp'
)
```

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 预测步数 | 10 | 20 | 30 | 40 | 50 |
| 控制步数 | 5 | 10 | 15 | 20 | 25 |
| 采样时间 ms | 20 | 10 | 10 | 5 | 2 |
| 求解器 | QP | QP | OSQP | OSQP | OSQP |
| 最大力矩 Nm | 50 | 100 | 200 | 500 | 1000 |

### 3.5 安全等级

| 功能 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 软限位 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 硬限位 | — | ✓ | ✓ | ✓ | ✓ |
| 碰撞检测(力) | — | ✓ | ✓ | ✓ | ✓ |
| 碰撞检测(视觉) | — | — | — | ✓ | ✓ |
| 碰撞预测 | — | — | — | ✓ | ✓ |
| 看门狗监控 | — | — | ✓ | ✓ | ✓ |
| 故障容忍 | — | — | — | ✓ | ✓ |

---

## 四、通信接口规格速查

```python
from control.ros2_interface import get_ros2_spec
spec = get_ros2_spec('L')
```

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 通信协议 | REST | ROS2 | ROS2 | ROS2 | ROS2+自定义 |
| 实时性 | 非实时 | 软实时 | 硬实时 | 硬实时 | 双系统 |
| ROS2话题数 | 5 | 10 | 20 | 30 | 50 |
| ROS2服务数 | 3 | 5 | 10 | 15 | 25 |
| 最大频率 Hz | 50 | 100 | 200 | 500 | 1000 |
| 安全加密 | — | — | ✓ | ✓ | ✓ |

---

## 五、硬件平台规格速查

| 参数 | S | M | L | XL | XXL |
|------|---|---|---|---|-----|
| 推荐平台 | RPi 5 | Jetson Nano | Jetson Orin Nano | Jetson AGX Orin | NVIDIA DRIVE |
| GPU TOPS | 1.5 | 5 | 40 | 275 | 1000+ |
| 内存 | 4GB | 8GB | 16GB | 32GB | 256GB |
| 功耗 W | <5 | 10-15 | 15-30 | 30-60 | >200 |
| 防护等级 | IP20 | IP30 | IP54 | IP65 | IP67 |

---

## 六、快速代码对照表

### 创建各级AGV

```python
from control.agv import AGVSpec, AGVGrade, AGVMotionController

# S级: 教育/实验
spec_s = AGVSpec.from_grade(AGVGrade.S)
# M级: 标准助手
spec_m = AGVSpec.from_grade(AGVGrade.M)
# L级: 专业工业
spec_l = AGVSpec.from_grade(AGVGrade.L)
# XL级: 高性能
spec_xl = AGVSpec.from_grade(AGVGrade.XL)
# XXL级: 旗舰全功能
spec_xxl = AGVSpec.from_grade(AGVGrade.XXL)
```

### 创建各级传感器

```python
# 触觉传感器
from sensors.tactile import TactileArray, get_tactile_spec
spec = get_tactile_spec('M')  # 16x16
ta = TactileArray(array_size=spec['array'])

# 力觉传感器
from sensors.force import ForceTorqueSensor, get_force_spec
spec = get_force_spec('L')  # 6轴 ±500N
fs = ForceTorqueSensor()

# IMU传感器
from sensors.imu import IMUSensor, get_imu_spec
spec = get_imu_spec('XL')  # ADIS16470, 1000Hz
imu = IMUSensor(sample_rate=spec['sample_hz'])
```

### 创建各级融合网络

```python
from fusion.cross_modal_fusion import FusionConfig, CrossModalFusion

for grade in ['S','M','L','XL','XXL']:
    dims = {'S':128,'M':256,'L':512,'XL':768,'XXL':1024}
    config = FusionConfig(hidden_dim=dims[grade], num_heads={2,4,8,12,16}[grades.index(grade)])
    fusion = CrossModalFusion(config)
```

### 创建各级MPC

```python
from control.mpc import MPCConfig, JointSpaceMPC

config = MPCConfig.for_grade('L', num_joints=6, dt=0.01)
mpc = JointSpaceMPC(config=config, num_joints=6)
```

---

*文档版本: v1.0.0*
*最后更新: 2026-03-30*
